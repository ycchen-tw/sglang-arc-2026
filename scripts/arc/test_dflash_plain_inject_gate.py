#!/usr/bin/env python3
"""GPU gate: the served plain / injected-context drafter (RingStatePool._step_injected on the compiled layers and the
ring attention) against model.py's reference (an uncompiled MTP3, SDPA attention: refresh and _decode_step_injected),
run in FP32 as the truth and in BF16 as the yardstick: the served path must be no farther from FP32 than the BF16
reference is (random inputs grow the BF16 residual error to ~5% over the five layers, arc-forge
qwen38fn_dflash_plain_inject_20261019.md, so served vs BF16 reference alone compounds two such errors).

  python scripts/arc/test_dflash_plain_inject_gate.py --drafter <plan B deploy-bf16 dir>
      # prints PLAIN_INJECT_GATE_OK

Random BF16 target embedding / lm_head (as test_dflash_refresh_base.py), random target GRs. Requests with prompts
of 5 .. 2,500 positions (window wrap) are prefilled through the served model's refresh in prefill chunks and written
to the pool; then cycles of verify blocks [anchor, drafts] with random accept lengths go through defer_commit + the
pool step. Per cycle the reference runs decode_step (lengths) on the pool's window as it stood before the step, with
the same pending block, and is compared with the step:
  - the proposal hidden states the step hands to fast_propose (captured in the eager step): relative L2 to the FP32
    reference at most 1.5 x the BF16 reference's (+0.01), the worst row's cosine to FP32 within `slack` of the BF16
    reference's worst row in the same cycle: 0.01 eager (ARC_DFLASH_COMPILE=0, either ARC_DFLASH_RING_ATTN), 0.03
    compiled -- inductor's fusions move the rounding, measured 10-19 (served vs BF16 reference rel 0.12 compiled,
    0.06 eager; worst-row cos 0.950 compiled, 0.978 eager, BF16 reference 0.977) on the same inputs,
  - the committed rows' injected K/V written into the ring columns (and their positions), and
  - the window afterwards holds exactly the committed positions (cursor - window + 1 .. cursor that exist).
A CUDA-graph replay of the same step is compared with the eager step on a copy of the pool state: the commit (pool
K/V, positions) and position 0's candidates (the top-16 of the first slot's logits; later positions follow the sampled
predecessor, which the graph draws from a different generator offset).
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
    sys.path.insert(0, str(TREE / "scripts" / "arc"))
    from dflash_propose_bench import VOCAB, _StubTarget
    from safetensors.torch import load_file

    from sglang.srt.speculative.arc_dflash import heads, serving, state_pool
    from sglang.srt.speculative.arc_dflash import model as M
    from sglang.srt.speculative.arc_dflash import worker as Wk
    from sglang.srt.speculative.arc_dflash.state_pool import RingStatePool

    torch.manual_seed(0)
    device = torch.device("cuda")
    one = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0])
    target = _StubTarget()
    drafter = Path(a.drafter)
    model = Wk.build_dense_drafter(drafter, device, target, drafter, one)
    model.lm_head = heads.PairHead([(target.lm_head.weight, 0)], VOCAB, one)
    c = model.config
    if c.residual != "plain" or c.context != "inject":
        raise SystemExit(f"{drafter}: residual {c.residual!r} / context {c.context!r}; this gate is for plain/inject")
    if model.fp8_modules:
        raise SystemExit(f"{drafter}: FP8 modules {model.fp8_modules[:3]}...; the reference loads BF16 weights")
    steps = K - 1
    batch = 6
    slack = 0.01 if model.compile_mode == "0" else 0.03
    pool = RingStatePool(model, slots=batch + 4, max_batch=8, device=device, steps=steps, sparse_q=True)

    # the reference: model.py as is (no fast_drafter.install, no compile), the deploy's BF16 weights
    state = {}
    for f in sorted(drafter.glob("*.safetensors")):
        state.update(load_file(str(f), device=str(device)))
    state["embed_tokens.weight"] = target.embed.weight.detach()
    state["lm_head.weight"] = target.lm_head.weight.detach()

    def reference(dtype):
        r = M.MTP3(c).to(device=device, dtype=dtype).eval().requires_grad_(False)
        missing, unexpected = r.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()},
                                                strict=False)
        if missing or unexpected:
            raise SystemExit(f"reference load: missing {missing[:5]} unexpected {unexpected[:5]}")
        for layer in r.layers:
            layer.self_attn.backend = "sdpa"
        return r

    ref, ref32 = reference(torch.bfloat16), reference(torch.float32)

    captured = {}
    original = state_pool.fast_propose

    def capture(m, hidden, anchor, temperature):
        captured["hidden"] = hidden.detach().float().clone()
        return original(m, hidden, anchor, temperature)

    state_pool.fast_propose = capture

    g = torch.Generator().manual_seed(1)
    w = c.window
    lengths_prompt = (5, 300, 1023, 1024, 1025, 2500)
    rids = [f"r{j}" for j in range(batch)]
    cursor = {}
    worst = {"hidden_cos": 1.0, "bf16_cos": 1.0, "cos_excess": -1.0, "hidden_excess": -1.0, "served_rel": 0.0, "bf16_rel": 0.0, "kv_rel": 0.0}

    def rel(x, y):
        x, y = x.float(), y.float()
        return float((x - y).norm() / y.norm().clamp_min(1e-12))

    with torch.no_grad():
        # prefill: the served model's refresh in chunks (as the worker), then the pool; the reference refresh agrees
        for rid, n in zip(rids, lengths_prompt):
            ids = torch.randint(4, VOCAB, (1, n), generator=g).to(device)
            pred = (0.5 * torch.randn(1, n, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
            positions = torch.arange(1, 1 + n, device=device)[None]
            cache = ref_cache = None
            for b0 in range(0, n, 2048):
                b1 = min(b0 + 2048, n)
                _, cache = model.refresh(ids[:, b0:b1], pred[:, b0:b1], positions[:, b0:b1], cache)
                _, ref_cache = ref.refresh(ids[:, b0:b1], pred[:, b0:b1], positions[:, b0:b1], ref_cache)
            for x, y in zip(cache.layers, ref_cache.layers):
                worst["kv_rel"] = max(worst["kv_rel"], rel(x.key, y.key), rel(x.value, y.value))
                if x.boundaries or y.boundaries:
                    raise SystemExit("an injected context cache carries conv boundaries")
            if not torch.equal(cache.positions, ref_cache.positions):
                raise SystemExit(f"{rid}: prefill positions differ")
            last_gr = (0.5 * torch.randn(1, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
            pool.write_request(rid, serving.DraftRequest(serving.decode_cache(cache, w), last_gr, positions[:, -1].clone()))
            cursor[rid] = n
        slots = torch.tensor([pool.slot_of[r] for r in rids], device=device)

        def window(s, dtype=torch.bfloat16):
            ring = pool.positions[s, :w]
            order = torch.argsort(ring)
            layers = tuple(M.LayerCache(pool.key[i, s][:, order][None].clone().to(dtype),
                                        pool.value[i, s][:, order][None].clone().to(dtype), ())
                           for i in range(c.num_layers))
            return M.CleanCache(ring[order][None].clone(), layers)

        temperature = torch.full((batch,), 0.7, device=device)
        for cycle in range(a.cycles):
            # verify block [anchor, d1 .. d(K-1)] at cursor + 1 .., accept 0 .. K-1 drafts
            base = torch.tensor([cursor[r] for r in rids], device=device)
            positions = base[:, None] + 1 + torch.arange(K, device=device)
            tokens = torch.randint(4, VOCAB, (batch, K), generator=g).to(device)
            gr_span = (0.5 * torch.randn(batch, K + 1, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
            lengths = torch.randint(1, K + 1, (batch,), generator=g).to(device)
            anchor = torch.randint(4, VOCAB, (batch,), generator=g).to(device)
            pool.defer_commit(slots, tokens, gr_span, positions, lengths)
            before = [window(int(s)) for s in slots]
            before32 = [window(int(s), torch.float32) for s in slots]
            # reference steps per row on the window as it stood (FP32 truth, BF16 yardstick)
            ref_hidden, ref32_hidden, ref_committed = [], [], []
            for j in range(batch):
                hidden, committed, committed_positions = ref.decode_step(
                    tokens[j : j + 1], anchor[j : j + 1], gr_span[j : j + 1], positions[j : j + 1, 0], before[j],
                    lengths=lengths[j : j + 1],
                )
                ref_hidden.append(hidden[:, :steps].float())
                ref_committed.append((committed, committed_positions))
                hidden32, _, _ = ref32.decode_step(
                    tokens[j : j + 1], anchor[j : j + 1], gr_span[j : j + 1].float(), positions[j : j + 1, 0],
                    before32[j], lengths=lengths[j : j + 1],
                )
                ref32_hidden.append(hidden32[:, :steps].float())
            ref_hidden, ref32_hidden = torch.cat(ref_hidden), torch.cat(ref32_hidden)
            pool.step(slots, anchor, temperature)  # 8 > batch and no graphs captured: the eager step
            got = captured["hidden"][:batch]
            cos = torch.nn.functional.cosine_similarity(got.flatten(1), ref32_hidden.flatten(1), dim=1).min().item()
            cos16 = torch.nn.functional.cosine_similarity(ref_hidden.flatten(1), ref32_hidden.flatten(1), dim=1).min().item()
            worst["bf16_cos"] = min(worst["bf16_cos"], cos16)
            worst["cos_excess"] = max(worst["cos_excess"], (cos16 - slack) - cos)
            served, bf16 = rel(got, ref32_hidden), rel(ref_hidden, ref32_hidden)
            worst["hidden_cos"] = min(worst["hidden_cos"], cos)
            worst["hidden_excess"] = max(worst["hidden_excess"], served - max(1.5 * bf16, bf16 + 0.01))
            worst["served_rel"] = max(worst["served_rel"], served)
            worst["bf16_rel"] = max(worst["bf16_rel"], bf16)
            worst["vs_bf16"] = max(worst.get("vs_bf16", 0.0), rel(got, ref_hidden))
            # commits: the committed rows' K/V at their ring columns, the window = the newest committed positions
            for j, r in enumerate(rids):
                s = int(slots[j])
                n = int(lengths[j])
                committed, cpos = ref_committed[j]
                cols = cpos[0, :n] % w
                if not torch.equal(pool.positions[s, cols], cpos[0, :n]):
                    raise SystemExit(f"cycle {cycle} {r}: committed positions not in their ring columns")
                for i in range(c.num_layers):
                    worst["kv_rel"] = max(
                        worst["kv_rel"],
                        rel(pool.key[i, s][:, cols], committed.layers[i].key[0, :, :n]),
                        rel(pool.value[i, s][:, cols], committed.layers[i].value[0, :, :n]),
                    )
                cursor[r] = int(positions[j, n - 1])
                held = pool.positions[s, :w]
                want = torch.arange(max(1, cursor[r] - w + 1), cursor[r] + 1, device=device)
                if not torch.equal(torch.sort(held[held >= 0]).values, want):
                    raise SystemExit(f"cycle {cycle} {r}: window holds {held[held >= 0].numel()} positions, "
                                     f"want {want.numel()} ending at {cursor[r]}")
            if pool.last_position[slots].tolist() != [cursor[r] for r in rids]:
                raise SystemExit(f"cycle {cycle}: pool cursors {pool.last_position[slots].tolist()}")
        # the captured graph against the eager step on the same state and generator seed
        pool.capture([8])
        base = torch.tensor([cursor[r] for r in rids], device=device)
        positions = base[:, None] + 1 + torch.arange(K, device=device)
        tokens = torch.randint(4, VOCAB, (batch, K), generator=g).to(device)
        gr_span = (0.5 * torch.randn(batch, K + 1, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
        lengths = torch.randint(1, K + 1, (batch,), generator=g).to(device)
        anchor = torch.randint(4, VOCAB, (batch,), generator=g).to(device)
        saved = {n: getattr(pool, n).clone() for n in ("key", "value", "positions", "last_gr", "last_position",
                                                        "pending_tokens", "pending_gr", "pending_length")}
        outs = []
        for mode in ("graph", "eager"):
            for n, t in saved.items():
                getattr(pool, n).copy_(t)
            pool.defer_commit(slots, tokens, gr_span, positions, lengths)
            torch.cuda.manual_seed(1234)
            if mode == "eager":
                graphs, pool.graphs = pool.graphs, {}
            tok, (cand, prob) = pool.step(slots, anchor, temperature)
            if mode == "eager":
                pool.graphs = graphs
            outs.append((tok.clone(), cand.clone(), prob.clone(), pool.key.clone(), pool.positions.clone()))
        (tg, cg, pg, kg, posg), (te, ce, pe, ke, pose) = outs
        # position 0's candidates as sets (top-k order is not part of the contract)
        overlap = (cg[:, 0, :, None] == ce[:, 0, None, :]).any(-1).float().mean().item()
        kv_graph_rel = rel(kg, ke)
        pos_same = bool(torch.equal(posg, pose))
    print(f"worst over {a.cycles} cycles x {batch} rows, proposal hidden vs FP32 reference: served rel "
          f"{worst['served_rel']:.4f} (BF16 reference {worst['bf16_rel']:.4f}; worst excess over the allowance "
          f"{worst['hidden_excess']:+.4f}), worst-row cos served {worst['hidden_cos']:.5f} / BF16 reference "
          f"{worst['bf16_cos']:.5f} (worst excess {worst['cos_excess']:+.4f} at slack {slack}, compile "
          f"{model.compile_mode}); injected K/V rel {worst['kv_rel']:.2e}; served vs BF16 reference rel {worst['vs_bf16']:.4f}")
    print(f"graph vs eager: pool K rel {kv_graph_rel:.2e}, positions equal {pos_same}, position-0 candidate overlap "
          f"{overlap:.3f}")
    ok = (worst["cos_excess"] <= 0 and worst["hidden_excess"] <= 0 and worst["kv_rel"] < 1e-2 and pos_same
          and kv_graph_rel < 1e-2 and overlap >= 0.9)
    print("PLAIN_INJECT_GATE_OK" if ok else "PLAIN_INJECT_GATE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--cycles", type=int, default=12)
    raise SystemExit(main(ap.parse_args()))
