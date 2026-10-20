#!/usr/bin/env python3
"""One-GPU A/B: the served plain / injected-context drafter built as stored (separate linears, per-layer injected
K/V) against the same export under ARC_DFLASH_MERGE=1 (q | k | v, gate | up, k | v as one linear each) and, with
--inject-gemm, ARC_DFLASH_INJECT_KV=gemm (every layer's injected K/V from one BF16 GEMM), on the same inputs.

  python scripts/arc/test_dflash_merge_ab.py --drafter <deploy dir> [--inject-gemm]
      [ARC_DFLASH_W4A8=int4:128 ARC_HUMMING_DIR=...]   # prints MERGE_AB_OK

Both flags are read when build_dense_drafter runs, so one process builds both drafters (the module-level flags --
ARC_DFLASH_FP8_OP, _FUSE_ONCE, _KV_DTYPE, the decode opt-ins -- apply to both). Requests with prompts of
5 .. 2,500 positions are refreshed into each pool, then cycles of defer_commit + the eager pool step:
  - the committed injected K/V in the two pools (relative L2): the merged k | v linear computes the same sums as
    the separate ones (block FP8: the same block scales and per-token-group activation quantization); the GEMM
    under --inject-gemm rounds the folded context_layernorm weights to BF16 once and skips the activation's FP8
    quantization, so a larger but small distance,
  - the proposal hidden states (relative L2 / worst-row cosine) and the share of proposal tokens both draw under
    one seed.
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

TREE = Path(__file__).resolve().parents[2]
K = 4  # DFLASH_K: verify width, 3 proposals per step


def main(a):
    os.environ.setdefault("ARC_DFLASH_SPARSE_Q", "1")
    for k in ("ARC_DFLASH_MERGE", "ARC_DFLASH_INJECT_KV"):
        if k in os.environ:
            raise SystemExit(f"{k} is set by this test per build; unset it")
    sys.path.insert(0, str(TREE / "scripts" / "arc"))
    from dflash_propose_bench import VOCAB, _StubTarget

    from sglang.srt.speculative.arc_dflash import heads, serving, state_pool
    from sglang.srt.speculative.arc_dflash import worker as Wk
    from sglang.srt.speculative.arc_dflash.state_pool import RingStatePool

    if torch.cuda.get_device_capability() == (12, 0):
        # the server's fp8_utils.initialize_fp8_gemm_config: CUTLASS on sm_120 (a bare process would pick DeepGEMM)
        from sglang.srt.layers.quantization import fp8_utils

        fp8_utils.FP8_GEMM_RUNNER_BACKEND = fp8_utils.Fp8GemmRunnerBackend.CUTLASS
    device = torch.device("cuda")
    one = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0])
    torch.manual_seed(0)
    target = _StubTarget()
    drafter = Path(a.drafter)
    flags = {"stored": {}, "merged": {"ARC_DFLASH_MERGE": "1"}}
    if a.inject_gemm:
        flags["merged"]["ARC_DFLASH_INJECT_KV"] = "gemm"
    models, pools = {}, {}
    for name, env in flags.items():
        os.environ.update(env)
        try:
            m = Wk.build_dense_drafter(drafter, device, target, drafter, one)
        finally:
            for k in env:
                del os.environ[k]
        if m.config.residual != "plain" or m.config.context != "inject":
            raise SystemExit(f"{drafter}: not the plain / injected-context drafter")
        m.lm_head = heads.PairHead([(target.lm_head.weight, 0)], VOCAB, one)
        models[name] = m
        pools[name] = RingStatePool(m, slots=10, max_batch=8, device=device, steps=K - 1, sparse_q=True)
    attn = models["merged"].layers[0].self_attn
    built = (type(attn.qkv_proj).__name__, type(models["merged"].layers[0].mlp.gate_up_proj).__name__,
             type(attn.kv_proj).__name__ if attn.kv_proj is not None else None,
             None if models["merged"].context_kv_weight is None else tuple(models["merged"].context_kv_weight.shape))
    print(f"merged: qkv {built[0]}, gate_up {built[1]}, kv {built[2]}, context_kv_weight {built[3]}; "
          f"stored FP8 {len(models['stored'].fp8_modules)} / merged FP8 {len(models['merged'].fp8_modules)}, "
          f"W4A8 {len(models['stored'].w4a8_modules)} / {len(models['merged'].w4a8_modules)}", flush=True)

    captured = {}
    original = state_pool.fast_propose

    def capture(m, hidden, anchor, temperature):
        captured[id(m)] = hidden.detach().float().clone()
        return original(m, hidden, anchor, temperature)

    state_pool.fast_propose = capture

    def rel(x, y):
        x, y = x.float(), y.float()
        return float((x - y).norm() / y.norm().clamp_min(1e-12))

    g = torch.Generator().manual_seed(1)
    c = models["stored"].config
    w = c.window
    batch = 6
    rids = [f"r{j}" for j in range(batch)]
    cursor = {}
    worst = {"rel": 0.0, "cos": 1.0, "kv": 0.0, "share": []}
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
            hs, ts = out["stored"]
            hm, tm = out["merged"]
            worst["rel"] = max(worst["rel"], rel(hm, hs))
            worst["cos"] = min(worst["cos"], torch.nn.functional.cosine_similarity(hm.flatten(1), hs.flatten(1), 1).min().item())
            worst["share"].append((tm == ts).float().mean().item())
            for j, r in enumerate(rids):
                n = int(lengths[j])
                cols = positions[j, :n] % w
                ss, sm = int(slots["stored"][j]), int(slots["merged"][j])
                if not torch.equal(pools["stored"].positions[ss, cols], pools["merged"].positions[sm, cols]):
                    raise SystemExit(f"cycle {cycle} {r}: committed positions differ")
                for t in ("key", "value"):
                    worst["kv"] = max(worst["kv"], rel(getattr(pools["merged"], t)[:, sm][:, :, cols],
                                                       getattr(pools["stored"], t)[:, ss][:, :, cols]))
                cursor[r] = int(positions[j, n - 1])
    share = sum(worst["share"]) / len(worst["share"])
    # a bug (a wrong row block, scale or layer order) is O(1); these bound rounding. Merge only: the same sums, a
    # different tiling at most. Inject gemm on an FP8 export: the per-layer path's FP8 activation quantization is gone.
    max_kv = 0.1 if a.inject_gemm else 1e-2
    max_rel = a.max_rel if a.max_rel is not None else 0.15 if a.inject_gemm else 0.05
    min_cos = a.min_cos if a.min_cos is not None else 0.95 if a.inject_gemm else 0.99
    print(f"merged vs stored ({'+ inject gemm' if a.inject_gemm else 'merge only'}, W4A8 "
          f"{os.environ.get('ARC_DFLASH_W4A8') or 'off'}, compile {models['merged'].compile_mode}): committed K/V rel "
          f"{worst['kv']:.2e} (max {max_kv:.0e}); proposal hidden rel {worst['rel']:.4f}, worst-row cos "
          f"{worst['cos']:.5f}; same tokens under one seed {share:.3f}", flush=True)
    ok = worst["kv"] < max_kv and worst["rel"] < max_rel and worst["cos"] > min_cos
    print("MERGE_AB_OK" if ok else "MERGE_AB_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--inject-gemm", action="store_true", help="also ARC_DFLASH_INJECT_KV=gemm on the merged build")
    ap.add_argument("--max-rel", type=float, help="proposal hidden, relative L2 (default 0.05; 0.15 with --inject-gemm)")
    ap.add_argument("--min-cos", type=float, help="proposal hidden, worst row (default 0.99; 0.95 with --inject-gemm)")
    raise SystemExit(main(ap.parse_args()))
