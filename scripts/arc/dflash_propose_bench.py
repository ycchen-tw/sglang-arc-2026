#!/usr/bin/env python3
"""DFlash drafter propose on one GPU outside the server: what one decode cycle's draft step costs, by part.

The served shape is TP2 FP8 DFlash K4 with ARC_DFLASH_ATTN_TP=0: every rank runs the whole 5-layer dense drafter on
its batch (ring width 2 x 3 + 1 = 7 slots per request, 24 query / 2 KV heads, window 1024) and the target's
vocab-sharded lm_head top-k (124,160 rows per rank, merged by two all-gathers that this bench leaves out). The
drafter here is the deployed acc-v4 export with a random BF16 target embedding / lm_head (the step's cost does not
depend on their values) and random but well-formed ring states: full windows, 1-4 committed tokens pending, as a
verify leaves them.

  ring  [--batch 128] [--n 7]   ring_attention.attend against gather + SDPA (the reference the kernel was written
        against) and a tile sweep: per tile the time per layer, max / mean row error vs SDPA, and whether its output
        is bitwise the deployed tile's
  step  --drafter DIR [--batches 64,...,128] [--sizes 96,128,...] [--profile] [--reps 30]
        RingStatePool.step per batch size through the graphs captured for --sizes (a batch replays the smallest
        graph that holds it, as in serving), median of --reps CUDA-event timings; --profile: one eager step at the
        largest batch under the torch profiler, kernel time attributed to the innermost arc_dflash source line
  state --drafter DIR [--batch 128] [--steps 6]   the pool after N steps (tokens out, candidates, q, every pool
        tensor) hashed, for bitwise A/B of a code change (run once per arm, compare the printed digests)

Env knobs of the code under test (ARC_DFLASH_*) apply as in serving; ARC_DFLASH_COMPILE defaults to det here (the
TP2 default) and ARC_DFLASH_SPARSE_Q to 1 (the deployed setting).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import re
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("ARC_DFLASH_COMPILE", "det")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

W, HQ, HKV, D = 1024, 24, 2, 256
VOCAB, HIDDEN = 248320, 2560
SHARD = 124160  # rows of the target lm_head on one TP2 rank


# ------------------------------------------------------------------------------------------- ring attention
def _ring_inputs(batch, n, slots, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cols = W + 1
    pool_k = torch.randn(slots, HKV, cols + n, D, device="cuda", dtype=torch.bfloat16, generator=g)
    pool_v = torch.randn(slots, HKV, cols + n, D, device="cuda", dtype=torch.bfloat16, generator=g)
    positions = torch.full((slots, cols), -1, dtype=torch.long, device="cuda")
    cursor = torch.empty(slots, dtype=torch.long, device="cuda")
    for s in range(slots):
        length = [0, 1, 37, 600, 1023, 1024, 3000, 5000, 20000, 9000][s % 10]
        for p in range(max(0, length - W), length):
            positions[s, p % W] = p
        cursor[s] = max(length - 1, 0)
    idx = torch.randperm(slots, generator=g, device="cuda")[:batch]
    qpos = (cursor[idx] + 1)[:, None] + torch.arange(n, device="cuda")
    q = torch.randn(batch, HQ, n, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(batch, HKV, n, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(batch, n, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g).transpose(1, 2)
    return pool_k, pool_v, positions, idx, q, k, v, qpos


def _ring_reference(pool_k, pool_v, positions, idx, q, k, v, qpos):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cols = W + 1
    keys = torch.cat((pool_k[idx][:, :, :cols], k), 2)
    values = torch.cat((pool_v[idx][:, :, :cols], v), 2)
    kpos = torch.cat((positions[idx], qpos), 1)
    dist = qpos[:, :, None] - kpos[:, None, :]
    mask = (dist >= 0) & (dist < W) & (kpos[:, None] >= 0)
    batch, _, n, _ = q.shape
    group = HQ // HKV
    folded = q.reshape(batch, HKV, group * n, D)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        y = F.scaled_dot_product_attention(folded, keys, values, attn_mask=mask.repeat(1, group, 1)[:, None])
    return y.reshape(batch, HQ, n, D)


def _timeit(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def ring(args) -> None:
    from sglang.srt.speculative.arc_dflash import ring_attention as RA

    inputs = _ring_inputs(args.batch, args.n, args.batch + 16, 3)
    pool_k, pool_v, positions, idx, q, k, v, qpos = inputs

    def run():
        key, value = RA.RingWindow(pool_k, idx, W + 1), RA.RingWindow(pool_v, idx, W + 1)
        return RA.attend(q, k, v, qpos, key, value, positions[idx], W)

    ref = _ring_reference(*inputs).float()
    RA.BLOCK_M = RA.BLOCK_N = RA.NUM_WARPS = RA.NUM_STAGES = None
    base = run()
    t_ref = _timeit(lambda: _ring_reference(*inputs))
    t_base = _timeit(run)
    print(f"batch {args.batch}, n {args.n}, rows/program axis {HQ // HKV * args.n}: gather+SDPA {t_ref:.3f} ms/layer, "
          f"deployed tile {RA.TILES} -> {t_base:.3f} ms/layer", flush=True)
    tiles = [(bm, bn, w, s) for bm in (16, 32, 64, 128) for bn in (16, 32, 64) for w in (4, 8) for s in (1, 2, 3)]
    if args.tiles:
        tiles = [tuple(int(x) for x in t.split(",")) for t in args.tiles.split(";")]
    rows = []
    for bm, bn, warps, stages in tiles:
        RA.BLOCK_M, RA.BLOCK_N, RA.NUM_WARPS, RA.NUM_STAGES = bm, bn, warps, stages
        try:
            got = run()
            torch.cuda.synchronize()
            t = _timeit(run)
        except Exception as e:  # noqa: BLE001 -- a tile that does not compile / fit is reported, not fatal
            print(f"  {bm:3d}x{bn:2d} w{warps} s{stages}: fail ({type(e).__name__}: {str(e)[:80]})", flush=True)
            continue
        rel = (got.float() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-6)
        same = bool(torch.equal(got, base))
        rows.append((t, bm, bn, warps, stages, rel.max().item(), rel.mean().item(), same))
    RA.BLOCK_M = RA.BLOCK_N = RA.NUM_WARPS = RA.NUM_STAGES = None
    for t, bm, bn, warps, stages, mx, mean, same in sorted(rows):
        print(f"  {bm:3d}x{bn:2d} w{warps} s{stages}: {t:.3f} ms/layer ({t_base / t:.2f}x) | vs SDPA row err max "
              f"{mx:.2e} mean {mean:.2e} | bitwise = deployed tile: {same}", flush=True)


# ------------------------------------------------------------------------------------------- drafter step
class _StubTarget(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(0)
        embed = 0.02 * torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
        head = 0.02 * torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
        self.embed = nn.Embedding.from_pretrained(embed)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False, device="meta")
        self.lm_head.weight = nn.Parameter(head, requires_grad=False)
        self.lm_head.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=VOCAB)

    def get_input_embeddings(self):
        return self.embed


def _build(drafter: Path, max_batch: int, steps: int):
    from sglang.srt.speculative.arc_dflash import heads
    from sglang.srt.speculative.arc_dflash import worker as Wk
    from sglang.srt.speculative.arc_dflash.state_pool import RingStatePool

    one = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0])
    target = _StubTarget()
    if torch.cuda.get_device_capability() == (12, 0):
        # the server's fp8_utils.initialize_fp8_gemm_config (--fp8-gemm-backend auto): CUTLASS on sm_120, the block
        # FP8 path fp8.Fp8BlockLinear pins there (a bare process would dispatch to DeepGEMM)
        from sglang.srt.layers.quantization import fp8_utils

        fp8_utils.FP8_GEMM_RUNNER_BACKEND = fp8_utils.Fp8GemmRunnerBackend.CUTLASS
    # the step as a server without ARC_DFLASH_EXACT_PREFILL builds it (MTP3.refresh_entry only adds a refresh entry)
    model = Wk.build_dense_drafter(drafter, torch.device("cuda"), target, drafter, one, refresh_entry=False)
    ids = heads.draft_vocab(VOCAB, torch.device("cuda"))
    if ids is not None:  # ARC_DFLASH_DRAFT_VOCAB: the subset head the TP1 G4 deployment serves
        model.lm_head = heads.SubsetHead(target.lm_head.weight, ids)
    else:  # one TP2 rank's share of the head: top-k over its 124,160 rows (the group's all-gathers are left out)
        model.lm_head = heads.PairHead([(target.lm_head.weight[:SHARD], 0)], VOCAB, one)
    sparse = os.environ.setdefault("ARC_DFLASH_SPARSE_Q", "1") == "1"
    pool = RingStatePool(model, slots=max_batch + 8, max_batch=max_batch, device=torch.device("cuda"), steps=steps,
                         sparse_q=sparse)
    return model, pool


def _fill(pool, seed: int):
    """Random well-formed states in every slot but the scratch one: a full window ending where its pending block
    (1-4 committed tokens) starts."""
    c = pool.c
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots, w, b = pool.slots, c.window, c.block_size
    n = slots - 1
    lengths = torch.randint(2000, 40000, (n,), device="cuda", generator=g)
    p = (lengths[:, None] - w) + torch.arange(w, device="cuda")
    pool.positions[:n] = -1
    pool.positions[:n].scatter_(1, p % w, p)
    for kv in (pool.key, pool.value):
        if kv.dtype == torch.bfloat16:
            kv.normal_(generator=g)
        else:  # ARC_DFLASH_KV_DTYPE=fp8_e4m3: no normal_ for e4m3; BF16 draws cast per layer, as the commit casts
            for layer in kv:
                layer.copy_(torch.randn(layer.shape, device="cuda", dtype=torch.bfloat16, generator=g))
    if not pool.inject:  # an injected-context pool keeps no conv boundaries
        pool.boundary.normal_(generator=g)
    pending = torch.randint(1, 5, (n,), device="cuda", generator=g)
    pool.pending_length[:n] = pending
    pool.last_position[:n] = lengths - 1 + pending
    pool.pending_tokens[:n] = torch.randint(4, VOCAB, (n, b), device="cuda", generator=g)
    pool.pending_gr[:n] = torch.randn(n, b + 1, c.hc_count, c.hidden_size, device="cuda", dtype=torch.bfloat16,
                                      generator=g)
    pool.last_gr[:n] = torch.randn(n, c.hc_count, c.hidden_size, device="cuda", dtype=torch.bfloat16, generator=g)
    torch.cuda.synchronize()


def _commit(pool, slots, g, k=4):
    """What a K=4 verify leaves: [anchor, 3 drafts] with their GR, 1-4 of them accepted."""
    c = pool.c
    b = slots.shape[0]
    tokens = torch.randint(4, VOCAB, (b, k), device="cuda", generator=g)
    first = pool.last_position[slots] + 1
    positions = first[:, None] + torch.arange(k, device="cuda")
    gr = torch.randn(b, k, c.hc_count, c.hidden_size, device="cuda", dtype=torch.bfloat16, generator=g)
    span = torch.cat((pool.last_gr[slots][:, None], gr), 1)
    accepted = torch.randint(1, k + 1, (b,), device="cuda", generator=g)
    pool.defer_commit(slots, tokens, span, positions, accepted)


_FRAME = re.compile(r"arc_dflash/(\w+\.py)\((\d+)\): (\w+)")


def _attribute(prof) -> list:
    """Kernel time per innermost arc_dflash Python frame above the launching op (the profiler's with_stack trace
    records Python calls as events of their own, parents of the ops they run; kernel events are skipped so nothing
    counts twice)."""
    lines = collections.Counter()
    calls = collections.Counter()
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CPU:
            continue
        dt = getattr(e, "self_device_time_total", None)
        if dt is None:
            dt = e.self_cuda_time_total
        if not dt:
            continue
        where = "(no arc_dflash frame)"
        node = e.cpu_parent
        while node is not None:
            m = _FRAME.search(node.name)
            if m:
                where = f"{m.group(1)}:{m.group(2)} {m.group(3)}"
                break
            node = node.cpu_parent
        lines[where] += dt
        calls[where] += max(1, len(getattr(e, "kernels", None) or ()))
    return [(us / 1e3, calls[k], k) for k, us in lines.most_common()]


def step(args) -> None:
    batches = [int(x) for x in args.batches.split(",")]
    sizes = sorted({int(x) for x in args.sizes.split(",")} | {max(batches)})
    model, pool = _build(args.drafter, max(sizes), args.steps)
    _fill(pool, 1)
    g = torch.Generator(device="cuda").manual_seed(2)
    t0 = time.time()
    pool.capture(sizes)
    torch.cuda.synchronize()
    print(f"captured {sizes} in {time.time() - t0:.1f} s; compile mode {model.compile_mode}, sparse_q "
          f"{pool.sparse_q}, ring attention {pool.attention}", flush=True)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for b in batches:
        slots = torch.arange(b, device="cuda")
        anchor = torch.randint(4, VOCAB, (b,), device="cuda", generator=g)
        temp = torch.full((b,), 0.7, device="cuda")
        times = []
        for i in range(args.reps + 3):  # a step folds in the pending block; the verify after it leaves the next
            start.record()
            pool.step(slots, anchor, temp)
            end.record()
            torch.cuda.synchronize()
            _commit(pool, slots, g)
            if i >= 3:
                times.append(start.elapsed_time(end))
        print(f"batch {b:4d} (graph {pool.bucket(b)}): step median {statistics.median(times):.3f} ms "
              f"p10 {sorted(times)[len(times) // 10]:.3f} p90 {sorted(times)[9 * len(times) // 10]:.3f}", flush=True)
    if args.profile:
        b = max(batches)
        slots = torch.arange(b, device="cuda")
        anchor = torch.randint(4, VOCAB, (b,), device="cuda", generator=g)
        pool.idx.fill_(pool.scratch)
        pool.idx[:b] = slots
        pool.anchor[:b] = anchor
        pool.temperature[:b] = 0.7
        for _ in range(3):
            pool._step(b)
            _commit(pool, slots, g)
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            pool._step(b)
            torch.cuda.synchronize()
        rows = _attribute(prof)
        total = sum(r[0] for r in rows)
        kernels = sum(r[1] for r in rows)
        print(f"eager step at batch {b}: {total:.3f} ms of kernels in ~{kernels} launches, by source line:")
        for ms, n, where in rows:
            if ms >= 0.003:
                print(f"  {ms:7.3f} ms {n:5d}  {where}")
        by_kernel, count = collections.Counter(), collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                by_kernel[e.name[:110]] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
                count[e.name[:110]] += 1
        print("by kernel:")
        for name, us in by_kernel.most_common(40):
            print(f"  {us / 1e3:7.3f} ms {count[name]:4d}  {name}")


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def state(args) -> None:
    model, pool = _build(args.drafter, args.batch, args.steps_per_cycle)
    _fill(pool, 1)
    pool.capture([args.batch])
    g = torch.Generator(device="cuda").manual_seed(5)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    slots = torch.arange(args.batch, device="cuda")
    anchor = torch.randint(4, VOCAB, (args.batch,), device="cuda", generator=g)
    temp = torch.full((args.batch,), 0.7, device="cuda")
    temp[::5] = 0.0
    outs = []
    for _ in range(args.steps):
        tokens, q = pool.step(slots, anchor, temp)
        outs.append(tokens.clone())
        outs.extend(x.clone() for x in (q if isinstance(q, tuple) else (q,)))
        _commit(pool, slots, g)
    torch.cuda.synchronize()
    names = ("key", "value", "boundary", "positions", "last_gr", "last_position", "pending_tokens", "pending_gr",
             "pending_length")
    w = pool.c.window
    for name in names:
        t = getattr(pool, name)
        if name in ("key", "value"):  # the ring columns only: the garbage column takes racing duplicate writes
            t = t[:, :, :, :w]
        print(f"{name:15s} {_digest(t)}")
    print(f"{'outputs':15s} {_digest(torch.cat([o.flatten().view(torch.uint8) for o in outs]))}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ring")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n", type=int, default=7)
    p.add_argument("--tiles", help="bm,bn,warps,stages;... (default: a sweep)")
    p = sub.add_parser("step")
    p.add_argument("--drafter", type=Path, required=True)
    p.add_argument("--batches", default="64,80,88,96,104,112,120,128")
    p.add_argument("--sizes", default="1,2,4,8,12,16,24,32,40,48,56,64,80,96,128")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--profile", action="store_true")
    p = sub.add_parser("state")
    p.add_argument("--drafter", type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--steps-per-cycle", type=int, default=3)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    {"ring": ring, "step": step, "state": state}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
