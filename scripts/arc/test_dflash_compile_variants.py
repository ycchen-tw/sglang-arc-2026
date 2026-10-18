#!/usr/bin/env python3
"""GPU check: the DFlash drafter's compiled layers stay within a fixed set of torch.compile variants, all compiled
before traffic, under the serving call patterns of ARC_DFLASH_EXACT_PREFILL / SESSION_PARK.

  python scripts/arc/test_dflash_compile_variants.py --drafter <deploy-bf16 dir>           # DFLASH_COMPILE_VARIANTS_OK
  python scripts/arc/test_dflash_compile_variants.py --drafter <dir> --legacy   # DFLASH_COMPILE_LEGACY_REPRODUCED

The deployed drafter (random BF16 target embedding / lm_head, as test_dflash_exact_park.py), compiled as in serving
(ARC_DFLASH_COMPILE, default 1: the one-GPU server's), its RingStatePool and the worker's own _append / _adopt / _park
/ _blank_state. Warm-up as at server start: fast_drafter.warm_refresh, then a fresh window per request and an eager
decode step per batch size (the eager warm-up before each CUDA-graph capture). Then traffic, a seeded shuffle of:
fresh windows over prompts of 1 .. 5,200 positions (chunked by ARC_DFLASH_PREFILL_CHUNK, the first chunk without a
prefix), appends of 1 .. 2,100 tokens to a pool slot's window (EXACT_PREFILL's continuing chunks, the park fold), a
blank window continued, decode steps of 1 .. 8 requests, and park / adopt cycles with and without the overshoot fold.

Asserts: traffic compiles nothing (dynamo's graph count does not move after the warm-up), no drafter code object holds
more than MAX_ENTRIES compiled variants (dynamo's recompile limit is 8), no recompile-limit warning. Values (eager):
the refresh's layer view (model.layer_view) leaves them alone, bitwise; a refresh shorter than model.REFRESH_MIN,
padded after its own tokens, is within rounding of the unpadded one (same math per row, other GEMM row counts).
--legacy runs the same traffic on the drafter as built without ARC_DFLASH_EXACT_PREFILL (MTP3.refresh_entry off: the
base refresh -- refresh and decode through one compiled layer entry, no layer view, no padding, no refresh warm-up):
the traffic compiles, past the limit.
"""

import argparse
import logging
import os
import random
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("ARC_DFLASH_COMPILE", "1")
os.environ.setdefault("ARC_DFLASH_SPARSE_Q", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from torch._dynamo import convert_frame  # noqa: E402
from torch._dynamo.eval_frame import _debug_get_cache_entry_list  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dflash_exact_park as T  # noqa: E402

# compiled variants a drafter code object may hold: 2 measured with the fix (Decoder.refresh_forward: with and
# without a prefix; Decoder.forward and the decode-only frames after its graph break: one request and more), half
# dynamo's limit of 8 as the bound
MAX_ENTRIES = 4
K = T.K

limit_events = []


class _LimitLog(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if "recompile_limit" in message or "cache_size_limit" in message:
            limit_events.append(message.splitlines()[0][:300])


def graphs():
    return counters["stats"]["unique_graphs"]


def drafter_entries():
    """compiled variants per drafter code object (dynamo's input codes from arc_dflash, resume functions included)"""
    out = {}
    for ref in convert_frame.input_codes.seen:
        code = ref()
        if code is None or "arc_dflash" not in code.co_filename:
            continue
        name = f"{Path(code.co_filename).name}:{code.co_firstlineno} {code.co_name}"
        out[name] = len(_debug_get_cache_entry_list(code))
    return out


def fresh_span(w, rid, ids, gr, start, end):
    """A window refreshed over positions start .. end - 1 (token t with the GR of t - 1), chunked as
    _prefill_exact's fresh window: the first chunk without a prefix, the next ones on the previous chunk's cache."""
    from sglang.srt.speculative.arc_dflash import serving

    c = w.model.config
    kept = torch.as_tensor(ids[start:end], device="cuda")[None]
    pred = gr[start - 1 : end - 1][None]
    positions = torch.arange(start, end, device="cuda")[None]
    cache = None
    for a in range(0, kept.shape[1], w.prefill_chunk):
        b = min(a + w.prefill_chunk, kept.shape[1])
        _, cache = w.model.refresh(kept[:, a:b], pred[:, a:b], positions[:, a:b], cache)
    if rid in w.pool.slot_of:
        w.pool.release(rid)
    w.pool.write_request(rid, serving.DraftRequest(serving.decode_cache(cache, c.window), gr[None, end - 1].clone(),
                                                   positions[:, -1].clone()))


def traffic(Wk, w, ids, gr, g, rng):
    pool = w.pool
    live = []  # rids whose windows decode can step (never deferred)

    def op_fresh():
        n = rng.choice([1, 2, 3, 17, 1023, 1024, 1025, 2047, 2048, 2049, 4100, 5200])
        rid = f"f{rng.randrange(6)}"
        fresh_span(w, rid, ids, gr, 1, 1 + n)
        if rid not in live:
            live.append(rid)

    def op_append():
        n = rng.choice([1, 2, 3, 4, 5, 7, 50, 2048, 2100])
        rid = f"a{rng.randrange(3)}"
        base = rng.choice([3, 900, 3000])
        fresh_span(w, rid, ids, gr, 1, base)
        w._append(rid, torch.as_tensor(ids[base : base + n], device="cuda"), gr[base : base + n], base)

    def op_blank():
        rid = "blank"
        if rid in pool.slot_of:
            pool.release(rid)
        pool.write_request(rid, w._blank_state(gr[None, 0].clone(), 0))
        n = rng.choice([1, 2, 40])
        w._append(rid, torch.as_tensor(ids[1 : 1 + n], device="cuda"), gr[1 : 1 + n], 1)

    def op_decode():
        if not live:
            return
        b = rng.randint(1, min(8, len(live)))
        T.step(w, rng.sample(live, b), [int(ids[7])] * b)

    def op_park():
        accept = rng.randint(1, K)
        e = rng.randrange(accept)
        T.run_case(Wk, w, g, flag=True, overshoot=rng.random() < 0.5, e=e, accept=accept,
                   match_after=rng.random() < 0.5)

    ops = [op_fresh] * 5 + [op_append] * 5 + [op_blank] + [op_decode] * 4 + [op_park] * 3
    for i in range(60):
        op, before = rng.choice(ops), graphs()
        op()
        if graphs() != before:
            print(f"  traffic op {i} {op.__name__} compiled {graphs() - before} graph(s)", flush=True)
        # run_case clears the pool's live requests: decode only what is still there
        live[:] = [rid for rid in live if rid in pool.slot_of]


def check_values(model, ids, gr):
    """eager: (1) a refresh whose layers see model.layer_view's layout equals one that sees the producers' layouts,
    bitwise; (2) a refresh of fewer than REFRESH_MIN tokens, padded after its own, against the same refresh unpadded:
    the rows' math is the same, the GEMMs' row count is not -- relative difference of each output at rounding level"""
    from sglang.srt.speculative.arc_dflash import model as M

    kept = torch.as_tensor(ids[1:3000], device="cuda")[None]
    pred = gr[0:2999][None]
    positions = torch.arange(1, 3000, device="cuda")[None]

    def outputs(n_new, prefixed):
        if prefixed:
            _, cache = model.refresh(kept[:, :2048], pred[:, :2048], positions[:, :2048])  # a chunk's cache
            h, c = model.refresh(kept[:, 2048 : 2048 + n_new], pred[:, 2048 : 2048 + n_new],
                                 positions[:, 2048 : 2048 + n_new], cache)
        else:
            h, c = model.refresh(kept[:, :n_new], pred[:, :n_new], positions[:, :n_new])
        return [h] + [t for layer in c.layers for t in (layer.key, layer.value, *layer.boundaries)]

    def patched(fn, **values):
        saved = {k: getattr(M, k) for k in values}
        for k, v in values.items():
            setattr(M, k, v)
        try:
            return fn()
        finally:
            for k, v in saved.items():
                setattr(M, k, v)

    with torch.compiler.set_stance("force_eager"):
        same = 0
        for n_new in (951, 5):
            a = outputs(n_new, True)
            b = patched(lambda: outputs(n_new, True), layer_view=lambda c: c, canonical=lambda t: t)
            for x, y in zip(a, b, strict=True):
                assert torch.equal(x.view(torch.int16), y.view(torch.int16)), "layer_view changed a refresh's values"
            same += len(a)
        worst = 0.0
        for n_new in (1, 2):
            for prefixed in (False, True):
                a = outputs(n_new, prefixed)
                b = patched(lambda: outputs(n_new, prefixed), REFRESH_MIN=1)
                for x, y in zip(a, b, strict=True):
                    assert x.shape == y.shape, (x.shape, y.shape)
                    rel = float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30))
                    worst = max(worst, rel)
        assert worst < 2e-2, worst
    return same, worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--legacy", action="store_true", help="MTP3.refresh_entry off (the base refresh, pre-fix)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.getLogger("torch._dynamo").addHandler(_LimitLog())
    from sglang.srt.speculative.arc_dflash import fast_drafter

    # --legacy: as a server without ARC_DFLASH_EXACT_PREFILL builds the drafter (the base refresh)
    Wk, model, pool, w = T.build(a.drafter, refresh_entry=not a.legacy)
    T.W = Wk
    w._blank_state = types.MethodType(Wk.DFlashDraftWorker._blank_state, w)
    if model.compile_mode == "0":
        raise SystemExit("ARC_DFLASH_COMPILE=0: nothing is compiled, nothing to check")
    g = torch.Generator().manual_seed(a.seed)
    rng = random.Random(a.seed)
    c = model.config
    total = 8000
    ids = torch.randint(4, T.VOCAB, (total,), generator=g).numpy()
    gr = (0.5 * torch.randn(total, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).cuda()

    # warm-up, as the server's start: the refresh variants, then decode's per batch size (capture's eager warm-up)
    if not a.legacy:
        begin = time.monotonic()
        fast_drafter.warm_refresh(model, w.device, w.prefill_chunk)
        torch.cuda.synchronize()
        print(f"warm_refresh (compiles the refresh entry's variants): {time.monotonic() - begin:.1f} s", flush=True)
    for i in range(8):
        fresh_span(w, f"d{i}", ids, gr, 1, 2000 + i)
    for b in range(1, 9):
        T.step(w, [f"d{i}" for i in range(b)], [int(ids[7])] * b)
    torch.cuda.synchronize()
    warm_graphs, warm_entries = graphs(), drafter_entries()
    print(f"warm-up ({'legacy: no refresh warm-up, producers layouts' if a.legacy else 'warm_refresh + decode'}): "
          f"{warm_graphs} compiled graphs, variants per code object {warm_entries}", flush=True)

    traffic(Wk, w, ids, gr, g, rng)
    torch.cuda.synchronize()
    new_graphs, entries = graphs() - warm_graphs, drafter_entries()
    worst = max(entries.values())
    print(f"traffic: {new_graphs} graphs compiled after the warm-up; variants per code object {entries}; "
          f"recompile-limit warnings {len(limit_events)}", flush=True)
    for event in limit_events[:4]:
        print("  ", event)
    if a.legacy:
        if new_graphs > 0 and (limit_events or worst >= torch._dynamo.config.recompile_limit):
            print("DFLASH_COMPILE_LEGACY_REPRODUCED")
            return 0
        print("legacy layouts did not reproduce the recompile-limit hit")
        return 1
    assert not limit_events, limit_events
    assert new_graphs == 0, f"{new_graphs} graphs compiled during traffic: a variant the warm-up does not cover"
    assert worst <= MAX_ENTRIES < torch._dynamo.config.recompile_limit, (worst, entries)
    same, worst = check_values(model, ids, gr)
    print(f"values (eager): {same} refresh outputs (hidden, K/V, boundaries) bitwise as with the producers' layouts; "
          f"refreshes of 1 and 2 tokens padded to REFRESH_MIN vs unpadded: max relative difference {worst:.1e}")
    print("DFLASH_COMPILE_VARIANTS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
