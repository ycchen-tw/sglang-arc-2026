#!/usr/bin/env python3
"""GPU check of ARC_DFLASH_SESSION_PARK with --session-exact-resume: the parked window continues at len(S) - 1.

  python scripts/arc/test_dflash_exact_park.py --drafter <deploy-bf16 dir>     # prints DFLASH_EXACT_PARK_OK

The deployed drafter (random BF16 target embedding / lm_head, as dflash_propose_bench.py), its RingStatePool and the
worker's own _park / _adopt / _append on a stand-in worker. A request's prompt is refreshed into its window, one
K=4 verify block is deferred with a random accept length L, and the request finishes with its stop at window step e
(P = len(S) - 1 = s + e + 1). The session's next request resumes the target at P (exact resume) with S + new tokens.

  no overshoot (the finishing block still pending at park, as without the overlap scheduler, or when the next
  batch was not a decode): with the flag the adopt folds the pending rows s .. P - 1 and continues; the window is
  checked against one refreshed from scratch over the same sequence and GR (positions, cursor, its GR, K/V and the
  next step's proposals q). Off, the pending rows are dropped as before and the window misses ("gap").
  overshoot (the overlap scheduler's extra cycle folded the finishing block, its own block pending): stop = bonus
  (e = L - 1) ends the window at P - 1 and hits; stop = an accepted draft ends it at P (the stop's own row, part of
  S: hits) or past it, where it hits only if the next request's ids equal the accepted rows after the stop, else
  misses (bootstrap) -- as before. K/V differences are measured against a floor: the same sequence refreshed and
  appended (EXACT_PREFILL's continuing chunks) vs refreshed in one go.
"""

import argparse
import collections
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("ARC_DFLASH_COMPILE", "det")
os.environ.setdefault("ARC_DFLASH_SPARSE_Q", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash_propose_bench import VOCAB, _StubTarget  # noqa: E402

K = 4  # DFLASH_K: verify width (anchor + 3 drafts), 3 proposals per step


def build(drafter, *, refresh_entry):
    """`refresh_entry`: as a server with ARC_DFLASH_EXACT_PREFILL (on) or without (MTP3.refresh_entry)"""
    from sglang.srt.speculative.arc_dflash import heads
    from sglang.srt.speculative.arc_dflash import worker as Wk
    from sglang.srt.speculative.arc_dflash.state_pool import RingStatePool

    one = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0])
    target = _StubTarget()
    model = Wk.build_dense_drafter(Path(drafter), torch.device("cuda"), target, Path(drafter), one,
                                   refresh_entry=refresh_entry)
    model.lm_head = heads.PairHead([(target.lm_head.weight, 0)], VOCAB, one)
    pool = RingStatePool(model, slots=24, max_batch=8, device=torch.device("cuda"), steps=K - 1, sparse_q=True)
    w = SimpleNamespace(pool=pool, model=model, device=torch.device("cuda"), prefill_chunk=2048,
                        parked=collections.OrderedDict(), state_stats=collections.Counter(), adopted=set(),
                        forward_stream=torch.cuda.current_stream())
    for name in ("_append", "_adopt", "_park"):
        setattr(w, name, types.MethodType(getattr(Wk.DFlashDraftWorker, name), w))
    return Wk, model, pool, w


def fresh(w, rid, ids, gr, end):
    """A window refreshed from scratch over positions .. end - 1 (the receptive tail), as _prefill_exact's fresh
    window: token t with the GR of t - 1."""
    from sglang.srt.speculative.arc_dflash import serving

    c = w.model.config
    first = max(1, end - c.num_layers * (c.window + 3))
    kept = torch.as_tensor(ids[first:end], device="cuda")[None]
    pred = gr[first - 1 : end - 1][None]
    positions = torch.arange(first, end, device="cuda")[None]
    cache = None
    for a in range(0, kept.shape[1], w.prefill_chunk):
        b = min(a + w.prefill_chunk, kept.shape[1])
        _, cache = w.model.refresh(kept[:, a:b], pred[:, a:b], positions[:, a:b], cache)
    w.pool.write_request(rid, serving.DraftRequest(serving.decode_cache(cache, c.window), gr[None, end - 1].clone(),
                                                   positions[:, -1].clone()))


def defer(w, rid, ids, gr, s, accept):
    """The verify at seq_len s: block [anchor, d1 .. d3] at s .. s + 3, GR of each, `accept` rows accepted."""
    pool = w.pool
    slot = torch.tensor([pool.slot_of[rid]], device="cuda")
    tokens = torch.as_tensor(ids[s : s + K], device="cuda")[None]
    span = torch.cat((pool.last_gr[slot][:, None], gr[None, s : s + K]), 1)
    positions = torch.arange(s, s + K, device="cuda")[None]
    pool.defer_commit(slot, tokens, span, positions, torch.tensor([accept], device="cuda"))


def step(w, rids, anchors):
    slots = torch.tensor([w.pool.slot_of[r] for r in rids], device="cuda")
    _, q = w.pool.step(slots, torch.as_tensor(anchors, device="cuda"), torch.ones(len(rids), device="cuda"))
    ids, probs = q
    return ids.clone(), probs.clone()


def window(pool, rid):
    s = pool.slot_of[rid]
    c = pool.c
    ring = pool.positions[s, : c.window]
    order = torch.argsort(ring)
    return (ring[order], pool.key[:, s][:, :, order], pool.value[:, s][:, :, order], pool.last_gr[s],
            int(pool.last_position[s]), int(pool.pending_length[s]))


def compare(pool, a, b):
    pa, ka, va, ga, la, na = window(pool, a)
    pb, kb, vb, gb, lb, nb = window(pool, b)
    assert torch.equal(pa, pb), "window positions differ"
    assert (la, na) == (lb, nb), ((la, na), (lb, nb))
    assert torch.equal(ga, gb), "cursor GR differs"
    valid = pa >= 0
    rel = max(float((ka[:, :, valid] - kb[:, :, valid]).float().norm() / kb[:, :, valid].float().norm()),
              float((va[:, :, valid] - vb[:, :, valid]).float().norm() / vb[:, :, valid].float().norm()))
    bitwise = torch.equal(ka, kb) and torch.equal(va, vb)
    return rel, bitwise


def fake_req(rid, sid, ids, origin_len, finished_len):
    output = list(ids[origin_len:])
    return SimpleNamespace(rid=rid, session_id=sid, origin_input_ids=list(ids[:origin_len]), output_ids=output,
                           finished_len=finished_len, output_ids_through_stop=output[:finished_len])


def run_case(Wk, w, g, *, flag, overshoot, e, accept, match_after):
    pool, c = w.pool, w.model.config
    hc, hid = c.hc_count, c.hidden_size
    L0 = 3000  # prompt
    total = L0 + 64 + 40
    ids = torch.randint(4, VOCAB, (total,), generator=g).numpy()
    gr = (0.5 * torch.randn(total, hc, hid, generator=g)).to(torch.bfloat16).cuda()
    s = L0 + 16  # the finishing verify's seq_len: 16 tokens decoded before it
    W.get_serving = lambda: SimpleNamespace(session_exact_resume=flag)
    w.parked.clear()
    w.state_stats.clear()
    w.adopted.clear()
    for r in list(pool.slot_of):
        pool.release(r)
    fresh(w, "a", ids, gr, s)  # the window through s - 1
    defer(w, "a", ids, gr, s, accept)
    P = s + e + 1  # the stop is the token at P (window step e); S = ids[:P + 1]
    if overshoot:
        step(w, ["a"], [int(ids[s])])  # the overshoot cycle's draft step folds the finishing block in
        defer(w, "a", ids, gr, s + accept, K)  # and its own verify block is pending
    origin_len = L0
    finished_len = P + 1 - origin_len
    req = fake_req("a", "sess", ids[: s + accept + 1], origin_len, finished_len)
    w._park(req)
    # the next request: S, then the next turn (or, for match_after, the accepted rows after the stop repeated)
    nxt = ids.copy()
    if not match_after:
        nxt[P + 1 :] = torch.randint(4, VOCAB, (total - P - 1,), generator=g).numpy()
    gr2 = gr.clone()
    gr2[P:] = (0.5 * torch.randn(total - P, hc, hid, generator=g)).to(torch.bfloat16).cuda()
    end2 = P + 40
    req2 = fake_req("b", "sess", nxt[:end2], end2 - 1, None)
    cursor = w._adopt(req2, P)
    stats = dict(w.state_stats)
    if cursor is None:
        return {"hit": False, "stats": stats}
    skip = cursor + 1 - P
    w._append("b", torch.as_tensor(nxt[P + skip : end2], device="cuda"), gr2[P + skip : end2], P + skip)
    # reference: one window refreshed over the next request's sequence, with the GR the adopt path used
    ref_gr = gr2.clone()
    ref_gr[:P] = gr[:P]
    if skip > 0:
        ref_gr[P : P + skip] = gr[P : P + skip]  # rows the parked window took from the finishing verify
    fresh(w, "ref", nxt, ref_gr, end2)
    rel, bitwise = compare(pool, "b", "ref")
    # the floor: the same sequence refreshed through s - 1, then appended (EXACT_PREFILL's continuing chunks)
    fresh(w, "split", nxt, ref_gr, s)
    w._append("split", torch.as_tensor(nxt[s:end2], device="cuda"), ref_gr[s:end2], s)
    floor, _ = compare(pool, "split", "ref")
    vs_split, split_bitwise = compare(pool, "b", "split")
    # the first proposal's q (later positions follow the sampled draws)
    qb, pb = step(w, ["b", "ref"], [int(nxt[end2])] * 2)
    same_cand = bool(torch.equal(qb[0, 0], qb[1, 0]))
    dq = float((pb[0, 0] - pb[1, 0]).abs().max())
    return {"hit": True, "cursor": cursor, "P": P, "stats": stats, "kv_rel": rel, "kv_bitwise": bitwise,
            "floor": floor, "vs_split": vs_split, "split_bitwise": split_bitwise,
            "q_same_candidates": same_cand, "q_max_abs_diff": dq}


W = None


def main() -> int:
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    a = ap.parse_args()
    Wk, model, pool, w = build(a.drafter, refresh_entry=True)  # SESSION_PARK needs EXACT_PREFILL
    W = Wk
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    # no overshoot, flag on: every stop step folds and hits at P - 1
    for accept in range(1, K + 1):
        for e in range(accept):
            r = run_case(Wk, w, g, flag=True, overshoot=False, e=e, accept=accept, match_after=False)
            assert r["hit"] and r["cursor"] == r["P"] - 1, r
            assert r["stats"].get("park_fold_tokens") == e + 1, r
            worst = max(worst, r["kv_rel"])
            print(f"no overshoot, accept {accept}, stop at step {e}: hit at P - 1 = {r['cursor']}, folded {e + 1} "
                  f"pending rows; K/V vs one refresh over the sequence {r['kv_rel']:.1e} (floor: refresh + append "
                  f"vs one refresh {r['floor']:.1e}); vs refresh + append {r['vs_split']:.1e} (bitwise "
                  f"{r['split_bitwise']}); next q vs one refresh max |dq| {r['q_max_abs_diff']:.1e}", flush=True)
    # flag off: the pending rows are dropped as before; the target resumes at P past the window: a miss
    r = run_case(Wk, w, g, flag=False, overshoot=False, e=1, accept=3, match_after=False)
    assert not r["hit"] and r["stats"].get("park_miss_gap") == 1, r
    print(f"flag off, no overshoot: miss {r['stats']}")
    # overshoot: stop = bonus hits at P - 1 without a fold; stop = draft past P - 1: hit iff the ids agree there
    for accept in range(1, K + 1):
        r = run_case(Wk, w, g, flag=True, overshoot=True, e=accept - 1, accept=accept, match_after=False)
        assert r["hit"] and r["cursor"] == r["P"] - 1 and "park_fold" not in r["stats"], r
        worst = max(worst, r["kv_rel"])
        print(f"overshoot, accept {accept}, stop = bonus: hit at P - 1, K/V vs one refresh {r['kv_rel']:.1e} "
              f"(floor {r['floor']:.1e}), next q max |dq| {r['q_max_abs_diff']:.1e}", flush=True)
        for e in range(accept - 1):
            # the overshoot cycle committed the block: the window ends at s + accept - 1 = P + (accept - 2 - e)
            past = accept - 2 - e
            r = run_case(Wk, w, g, flag=True, overshoot=True, e=e, accept=accept, match_after=False)
            if past == 0:  # the window ends at the stop token: S's own row
                assert r["hit"] and r["cursor"] == r["P"], r
                print(f"overshoot, accept {accept}, stop = draft at step {e}: window ends at P (S's last row): hit")
                continue
            assert not r["hit"] and r["stats"].get("park_miss_ids") == 1, r
            r = run_case(Wk, w, g, flag=True, overshoot=True, e=e, accept=accept, match_after=True)
            assert r["hit"] and r["cursor"] == r["P"] + past, r
            print(f"overshoot, accept {accept}, stop = draft at step {e}: {past} accepted rows past the stop in the "
                  f"window; miss when the next ids differ there, hit at {r['cursor']} when they repeat them "
                  f"(K/V vs one refresh {r['kv_rel']:.1e}, floor {r['floor']:.1e})", flush=True)
    assert worst < 5e-2, worst
    print(f"worst K/V relative difference vs a fresh refresh: {worst:.1e}")
    print("DFLASH_EXACT_PARK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
