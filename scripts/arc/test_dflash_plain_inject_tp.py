#!/usr/bin/env python3
"""Two-GPU check: the plain / injected-context drafter on an attention-TP pair (ARC_DFLASH_ATTN_TP=1, the L4 TP2
shape) against the same drafter on one GPU, on the same inputs.

  torchrun --nproc_per_node 2 scripts/arc/test_dflash_plain_inject_tp.py --drafter <deploy dir>
      [ARC_DFLASH_W4A8=int4:128 ARC_HUMMING_DIR=...]   # prints PLAIN_INJECT_TP_OK
  TP_TEST_ONE_GPU=1 torchrun --nproc_per_node 2 ...    # both ranks on cuda:0 over gloo (a functional check on one card)

Each rank builds the drafter twice: sharded over the pair (attention heads split, o_proj all-reduced; the pool holds
the rank's KV head) and whole on its own GPU. Both get the full random BF16 target lm_head (no vocab sharding: the
check is the drafter, not the head) and identical random inputs: prompts of 5 .. 2,500 positions refreshed in prefill
chunks into the pool, then cycles of defer_commit + the pool step (eager, so the proposal hidden can be captured).
  - the pair agrees (the server's ARC_DFLASH_CHECK_AGREE contract): both ranks' proposal hidden and tokens bitwise,
  - each rank's committed injected K/V equal the whole drafter's for that rank's KV head (relative L2),
  - the pair's proposal hidden against the whole drafter's (relative L2 / worst-row cosine; o_proj's partial sums
    and the sliced GEMMs only reorder BF16 sums), and the share of proposal tokens both draw under one seed.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

TREE = Path(__file__).resolve().parents[2]
K = 4  # DFLASH_K: verify width, 3 proposals per step


class PairGroup:
    """The attention-TP group interface build_dense_drafter / fp8.AllReduced use, over torch.distributed."""

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.ranks = list(range(self.world_size))

    def all_reduce(self, x):
        dist.all_reduce(x)
        return x

    def all_gather(self, x, dim=0):
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim)


def main(a):
    os.environ.setdefault("ARC_DFLASH_SPARSE_Q", "1")
    os.environ.setdefault("ARC_DFLASH_COMPILE", "det")
    os.environ["ARC_DFLASH_ATTN_TP"] = "1"
    rank = int(os.environ["LOCAL_RANK"])
    one_gpu = os.environ.get("TP_TEST_ONE_GPU") == "1"
    gpu = 0 if one_gpu else rank
    torch.cuda.set_device(gpu)
    if one_gpu:
        dist.init_process_group("gloo")
    else:
        dist.init_process_group("nccl", device_id=torch.device("cuda", gpu))
    sys.path.insert(0, str(TREE / "scripts" / "arc"))
    from dflash_propose_bench import VOCAB, _StubTarget

    from sglang.srt.speculative.arc_dflash import heads, serving, state_pool
    from sglang.srt.speculative.arc_dflash import worker as Wk
    from sglang.srt.speculative.arc_dflash.state_pool import RingStatePool

    device = torch.device("cuda", gpu)
    if torch.cuda.get_device_capability(device) == (12, 0):
        # the server's fp8_utils.initialize_fp8_gemm_config: CUTLASS on sm_120 (a bare process would pick DeepGEMM)
        from sglang.srt.layers.quantization import fp8_utils

        fp8_utils.FP8_GEMM_RUNNER_BACKEND = fp8_utils.Fp8GemmRunnerBackend.CUTLASS
    one = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0])
    pair = PairGroup()
    torch.manual_seed(0)
    target = _StubTarget()  # seeded: the same embedding / head on both ranks
    drafter = Path(a.drafter)
    models, pools = {}, {}
    build_head = Wk.build_head
    for name, group in (("pair", pair), ("whole", one)):
        # the pair's head would be the target's vocab-sharded one; both are replaced by the full head below, so the
        # pair is built with a placeholder (the stub target's lm_head is not sharded)
        Wk.build_head = (lambda *args, **kw: torch.nn.Module()) if group is pair else build_head
        path = drafter if group is pair else Path(a.whole_drafter or a.drafter)
        m = Wk.build_dense_drafter(path, device, target, path, group)
        Wk.build_head = build_head
        if m.config.residual != "plain" or m.config.context != "inject":
            raise SystemExit(f"{drafter}: not the plain / injected-context drafter")
        m.lm_head = heads.PairHead([(target.lm_head.weight, 0)], VOCAB, one)
        models[name] = m
        pools[name] = RingStatePool(m, slots=10, max_batch=8, device=device, steps=K - 1, sparse_q=True)
    c_pair, c_whole = models["pair"].config, models["whole"].config
    if c_pair.num_key_value_heads * pair.world_size != c_whole.num_key_value_heads:
        raise SystemExit(f"KV heads: pair {c_pair.num_key_value_heads} x {pair.world_size} vs whole {c_whole.num_key_value_heads}")

    captured = {}
    original = state_pool.fast_propose

    def capture(m, hidden, anchor, temperature):
        captured[id(m)] = hidden.detach().float().clone()
        return original(m, hidden, anchor, temperature)

    state_pool.fast_propose = capture

    def rel(x, y):
        x, y = x.float(), y.float()
        return float((x - y).norm() / y.norm().clamp_min(1e-12))

    g = torch.Generator().manual_seed(1)  # same draws on both ranks
    c = c_whole
    w = c.window
    batch = 6
    rids = [f"r{j}" for j in range(batch)]
    cursor = {}
    worst = {"rel": 0.0, "cos": 1.0, "kv": 0.0, "agree_hidden": True, "agree_tokens": True, "token_share": []}
    with torch.no_grad():
        for rid, n in zip(rids, (5, 300, 1023, 1024, 1025, 2500)):
            ids = torch.randint(4, VOCAB, (1, n), generator=g).to(device)
            pred = (0.5 * torch.randn(1, n, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
            positions = torch.arange(1, 1 + n, device=device)[None]
            last_gr = (0.5 * torch.randn(1, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
            for name, m in models.items():
                cache = None
                for b0 in range(0, n, 2048):
                    b1 = min(b0 + 2048, n)
                    _, cache = m.refresh(ids[:, b0:b1], pred[:, b0:b1], positions[:, b0:b1], cache)
                pools[name].write_request(rid, serving.DraftRequest(serving.decode_cache(cache, w), last_gr.clone(),
                                                                    positions[:, -1].clone()))
            cursor[rid] = n
        slots = {name: torch.tensor([p.slot_of[r] for r in rids], device=device) for name, p in pools.items()}
        temperature = torch.full((batch,), 0.7, device=device)
        for cycle in range(a.cycles):
            base = torch.tensor([cursor[r] for r in rids], device=device)
            positions = base[:, None] + 1 + torch.arange(K, device=device)
            tokens = torch.randint(4, VOCAB, (batch, K), generator=g).to(device)
            gr_span = (0.5 * torch.randn(batch, K + 1, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
            lengths = torch.randint(1, K + 1, (batch,), generator=g).to(device)
            anchor = torch.randint(4, VOCAB, (batch,), generator=g).to(device)
            out = {}
            for name, p in pools.items():
                p.defer_commit(slots[name], tokens, gr_span, positions, lengths)
                torch.cuda.manual_seed(1000 + cycle)
                tok, _ = p.step(slots[name], anchor, temperature)  # 8 > batch, no graphs: the eager step
                out[name] = (captured[id(models[name])][:batch], tok.clone())
            hp, tp = out["pair"]
            hw, tw = out["whole"]
            both_h = pair.all_gather(hp[None], 0)
            both_t = pair.all_gather(tp[None], 0)
            worst["agree_hidden"] &= bool(torch.equal(both_h[0], both_h[1]))
            worst["agree_tokens"] &= bool(torch.equal(both_t[0], both_t[1]))
            worst["rel"] = max(worst["rel"], rel(hp, hw))
            worst["cos"] = min(worst["cos"], torch.nn.functional.cosine_similarity(hp.flatten(1), hw.flatten(1), 1).min().item())
            worst["token_share"].append((tp == tw).float().mean().item())
            for j, r in enumerate(rids):
                n = int(lengths[j])
                cols = (positions[j, :n] % w)
                sp, sw = int(slots["pair"][j]), int(slots["whole"][j])
                if not torch.equal(pools["pair"].positions[sp, cols], pools["whole"].positions[sw, cols]):
                    raise SystemExit(f"rank {rank} cycle {cycle} {r}: committed positions differ")
                for i in range(c.num_layers):
                    for t in ("key", "value"):
                        mine = getattr(pools["pair"], t)[i, sp, 0][cols]
                        ref = getattr(pools["whole"], t)[i, sw, rank][cols]
                        worst["kv"] = max(worst["kv"], rel(mine, ref))
                cursor[r] = int(positions[j, n - 1])
    share = sum(worst["token_share"]) / len(worst["token_share"])
    w4a8 = os.environ.get("ARC_DFLASH_W4A8") or "off"
    print(f"rank {rank} (W4A8 {w4a8}, compile {models['pair'].compile_mode}): pair agrees bitwise -- hidden "
          f"{worst['agree_hidden']}, tokens {worst['agree_tokens']}; pair vs whole proposal hidden rel {worst['rel']:.4f}, "
          f"worst-row cos {worst['cos']:.5f}; committed K/V rel {worst['kv']:.2e}; same tokens under one seed {share:.3f}",
          flush=True)
    ok = (worst["agree_hidden"] and worst["agree_tokens"] and worst["rel"] < a.max_rel and worst["cos"] > a.min_cos
          and worst["kv"] < 2e-2)
    flags = pair.all_gather(torch.tensor([int(ok)], device=device), 0)
    dist.barrier()
    if rank == 0:
        print("PLAIN_INJECT_TP_OK" if bool(flags.all()) else "PLAIN_INJECT_TP_FAIL", flush=True)
    dist.destroy_process_group()
    return 0 if bool(flags.all()) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--whole-drafter", help="build the one-GPU drafter from this export instead (e.g. the BF16 deploy "
                    "against an FP8 pair: the quantization's own distance, the yardstick for the pair's)")
    ap.add_argument("--max-rel", type=float, default=0.05, help="pair vs whole proposal hidden, relative L2")
    ap.add_argument("--min-cos", type=float, default=0.99, help="pair vs whole proposal hidden, worst row")
    raise SystemExit(main(ap.parse_args()))
