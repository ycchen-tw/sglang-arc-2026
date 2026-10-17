#!/usr/bin/env python3
"""GPU check of --session-exact-resume's spec end snapshot (speculative/exact_end.py).

  python scripts/arc/test_exact_end_snapshot.py        # prints EXACT_END_SNAPSHOT_OK

1. plan(): which verify row fires, at which window step, with which P -- against a host model of
   Req.update_finish_state over many simulated requests (first stop among the accepted run, the max_new_tokens cap,
   ignore_eos, unarmed rows, padded accept rows), the seal (an overshoot verify after the finish never fires again),
   and request rows / slots handed to new requests (a new epoch fires again).
2. snapshot_after_advance() between the cursor advance and the commit, in a child process per mode:
   - no FOLD (fold every commit, 16-bit checkpoint): the live checkpoint and cursors are bitwise the ones without
     the snapshot launch; a snapshot at the window's last step is bitwise the live slot after the commit, and one at
     any step is bitwise the live slot after a commit that accepted through that step -- for the main commit with
     and without radix track rows (HAS_TRACK), whose fold rounds differently (exact_end.snapshot_after_advance);
   - ARC_GDN_FOLD_IN_VERIFY=1 (compact commit 0 and 1): the live checkpoint and cursors are untouched; the snapshot
     is bitwise the commit's track path at that step (full grid and compacted);
   - both: temporal against a float64 recompute of the window (rounding only), conv / PLE side states bitwise the
     verify's per-step intermediates.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import torch

L, HV, H, K, V, MAXC, KDRAFT = 4, int(os.environ.get("EXACT_END_TEST_HV", "48")), int(os.environ.get("EXACT_END_TEST_H", "16")), 128, 128, 16, 4
SLOTS, ROWS = 160, 48


# ---------------------------------------------------------------------------------------------------------------------
# 1. plan()
def _fake_req(rid, row, slot, epoch, origin_len, max_new, stops, ignore_eos=False):
    kv = SimpleNamespace(req_pool_idx=row, exact_end_slot=slot, exact_end_epoch=epoch, exact_end_row_written=False)
    sp = SimpleNamespace(ignore_eos=ignore_eos, stop_token_ids=set(stops[:1]), max_new_tokens=max_new)
    return SimpleNamespace(rid=rid, kv=kv, exact_end_verifies=0, sampling_params=sp, eos_token_ids=set(stops[1:]),
                           tokenizer=None, origin_input_ids=[0] * origin_len, output=[7], finished=False,
                           expect=None)


def _host_finish(req, new_tokens, stops):
    """Req.update_finish_state on the accepted run: (step, P) of the finish in this verify, or None."""
    out_before = len(req.output)
    first = None
    if not req.sampling_params.ignore_eos:
        for j, t in enumerate(new_tokens):
            if t in stops:
                first = j
                break
    cap = req.sampling_params.max_new_tokens - out_before - 1  # the step whose token is the max_new_tokens-th
    cand = [x for x in (first, cap if 0 <= cap < len(new_tokens) else None) if x is not None]
    if not cand:
        return None
    j = min(cand)
    finished_len = out_before + j + 1
    return j, len(req.origin_input_ids) + finished_len - 1


def test_plan() -> None:
    from sglang.srt.speculative import exact_end as E

    dev = "cuda"
    g = torch.Generator().manual_seed(0)
    pool = SimpleNamespace(req_to_token=torch.empty(ROWS + 1, 1),
                           mamba_pool=SimpleNamespace(size=SLOTS, replayssm_is_flush=torch.zeros(
                               ROWS + 1, dtype=torch.int8, device=dev)))
    state = E.SpecState(pool, dev)
    stops = [5, 6, 9]
    free_slots = list(range(1, SLOTS + 1))
    free_rows = list(range(1, ROWS + 1))
    running = []
    fired_total = checked = 0
    for cycle in range(400):
        # admit: new requests take free rows and slots (rows and slots of finished ones come back reused)
        while free_rows and len(running) < 24 and torch.rand(1, generator=g).item() < 0.7:
            row = free_rows.pop(int(torch.randint(len(free_rows), (1,), generator=g)))
            armed = torch.rand(1, generator=g).item() < 0.85
            slot = free_slots.pop(int(torch.randint(len(free_slots), (1,), generator=g))) if armed else None
            req = _fake_req(f"r{cycle}-{row}", row, None if slot is None else torch.tensor(slot, device=dev),
                            next(E._epochs) if armed else 0, int(torch.randint(1, 300, (1,), generator=g)),
                            int(torch.randint(1, 40, (1,), generator=g)), stops,
                            ignore_eos=torch.rand(1, generator=g).item() < 0.1)
            req.slot_id = slot
            running.append(req)
        if not running:
            continue
        bs = len(running)
        acc = torch.randint(1, KDRAFT + 1, (bs,), generator=g)
        tok = torch.randint(0, 12, (bs, KDRAFT), generator=g).to(torch.int32)
        ai = torch.full((bs, KDRAFT), -1, dtype=torch.int32)
        for i in range(bs):
            ai[i, : acc[i]] = i * KDRAFT + torch.arange(int(acc[i]), dtype=torch.int32)
        seq = torch.tensor([len(r.origin_input_ids) + len(r.output) - 1 for r in running])
        batch = SimpleNamespace(reqs=running, req_pool_indices=torch.tensor([r.kv.req_pool_idx for r in running],
                                                                            dtype=torch.int64, device=dev),
                                seq_lens=seq.to(dev))
        plan = state.plan(batch, tok.view(-1).to(dev), ai.to(dev), acc.to(torch.int32).to(dev))
        idx, step, pos = plan.idx.cpu().tolist(), plan.step.cpu().tolist(), plan.pos.cpu().tolist()
        still = []
        for i, req in enumerate(running):
            new = tok[i, : acc[i]].tolist()
            if req.finished:  # the overshoot verify: never fires again
                assert idx[i] == -1, (req.rid, "fired in its overshoot verify")
                free_rows.append(req.kv.req_pool_idx)
                if req.slot_id is not None:
                    free_slots.append(req.slot_id)
                continue
            exp = _host_finish(req, new, stops if not req.sampling_params.ignore_eos else [])
            checked += 1
            if exp is not None and req.slot_id is not None:
                assert (idx[i], step[i], pos[i]) == (req.slot_id, exp[0], exp[1]), (req.rid, idx[i], step[i], pos[i],
                                                                                     exp)
                fired_total += 1
            else:
                assert idx[i] == -1 and step[i] == -1 and pos[i] == -1, (req.rid, idx[i], exp)
            req.output.extend(new)
            if exp is not None:
                req.finished = True  # one more verify (the overlap overshoot) before its row and slot come back
            still.append(req)
        running = still
    assert fired_total > 200, fired_total
    print(f"plan: {checked} verify rows checked against the host finish model, {fired_total} snapshots fired, "
          f"no overshoot verify fired", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# 2. kernels
def _pools(seed, fold, bs):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[os.environ.get("EXACT_END_TEST_DTYPE", "bfloat16")]
    dev = "cuda"
    temporal = (0.1 * torch.randn(L, SLOTS + 1, HV, V, K, device=dev, generator=g)).to(dt)
    d = (0.1 * torch.randn(L, ROWS + 1, HV, MAXC, V, device=dev, generator=g)).to(dt)
    k = (0.1 * torch.randn(L, ROWS + 1, H, MAXC, K, device=dev, generator=g)).to(dt)
    gg = (-0.05 * torch.rand(L, ROWS + 1, HV, MAXC, device=dev, generator=g)).float()
    dr = (1e-3 * torch.randn(L, ROWS + 1, HV, MAXC, V, device=dev, generator=g)).to(dt)
    kr = (1e-3 * torch.randn(L, ROWS + 1, H, MAXC, K, device=dev, generator=g)).to(dt)
    conv_dim, km1 = 64, 3
    conv = torch.randn(L, SLOTS + 1, conv_dim, km1, device=dev, generator=g).to(dt)
    icw = torch.randn(L, bs, KDRAFT, conv_dim, km1, device=dev, generator=g).to(dt)
    sc = torch.randn(2, SLOTS + 1, 32, 4, device=dev, generator=g).to(dt)
    isc = torch.randn(2, bs, KDRAFT, 32, 4, device=dev, generator=g).to(dt)
    ng = torch.randint(0, 1000, (SLOTS + 1, 6), device=dev, generator=g)
    ing = torch.randint(0, 1000, (bs, KDRAFT, 6), device=dev, generator=g)
    assert 3 * bs <= SLOTS
    perm = torch.randperm(SLOTS, generator=g, device=dev) + 1
    state_idx = perm[:bs].to(torch.int64)
    snap_slots = perm[bs : 2 * bs].to(torch.int64)
    track_slots = perm[2 * bs : 3 * bs].to(torch.int64)
    replay = (torch.randperm(ROWS, generator=g, device=dev)[:bs] + 1).to(torch.int64)
    accept = torch.randint(1, KDRAFT + 1, (bs,), device=dev, generator=g).to(torch.int32)
    write_pos = torch.zeros(ROWS + 1, dtype=torch.int32, device=dev)
    base = torch.randint(0, MAXC, (ROWS + 1,), dtype=torch.int32, device=dev, generator=g)
    flush = torch.zeros(ROWS + 1, dtype=torch.int8, device=dev)
    if fold:
        # h20: the verify folded the previous window (cursors as the last advance left them: pending, is_flush 1)
        prev = torch.randint(1, KDRAFT + 1, (bs,), device=dev, generator=g).to(torch.int32)
        write_pos[replay] = prev
        flush[replay] = 1
    # fold every commit: the previous commit's finish emptied the ring (write_pos 0, is_flush 0)
    steps = (torch.rand(bs, device=dev, generator=g) * accept).long()
    steps[: bs // 3] = accept[: bs // 3].long() - 1  # a third at the window's last step (stop = bonus)
    snap_idx = torch.where(torch.rand(bs, device=dev, generator=g) < 0.8, snap_slots, -1)
    snap_step = torch.where(snap_idx >= 0, steps, -1)
    if bs > 3:
        state_idx[1] = -1  # a padding row: skipped even with a step
        snap_idx[1], snap_step[1] = snap_slots[1], 0
    spec_state = SimpleNamespace(temporal=temporal, replayssm_d=d, replayssm_k=k, replayssm_g=gg, replayssm_rawv=dr,
                                 replayssm_rawk=kr, conv=[conv], intermediate_conv_window=[icw])
    mamba_pool = SimpleNamespace(replayssm_spec_write_pos=write_pos, replayssm_cache_base=base,
                                 replayssm_is_flush=flush)
    req_pool = SimpleNamespace(mamba_pool=mamba_pool, exact_end_spec=SimpleNamespace(zero_flush=torch.zeros_like(flush)),
                               short_conv_pool=SimpleNamespace(conv_state=sc, intermediate_conv_state=isc),
                               ngram_pool=SimpleNamespace(context=ng, intermediate_context=ing))
    return dict(spec_state=spec_state, req_pool=req_pool, state_idx=state_idx, replay=replay, accept=accept,
                snap_idx=snap_idx, snap_step=snap_step, track_slots=track_slots)


def _clone(p):
    ss = p["spec_state"]
    rp = p["req_pool"]
    mp = rp.mamba_pool
    spec_state = SimpleNamespace(**{k: (v.clone() if torch.is_tensor(v) else [x.clone() for x in v])
                                    for k, v in vars(ss).items()})
    mamba_pool = SimpleNamespace(**{k: v.clone() for k, v in vars(mp).items()})
    req_pool = SimpleNamespace(
        mamba_pool=mamba_pool, exact_end_spec=rp.exact_end_spec,
        short_conv_pool=SimpleNamespace(conv_state=rp.short_conv_pool.conv_state.clone(),
                                        intermediate_conv_state=rp.short_conv_pool.intermediate_conv_state),
        ngram_pool=SimpleNamespace(context=rp.ngram_pool.context.clone(),
                                   intermediate_context=rp.ngram_pool.intermediate_context))
    return dict(p, spec_state=spec_state, req_pool=req_pool)


def _step(p, *, snapshot, track):
    """the circular branch of commit_mamba_states_after_verify: advance, (snapshot), commit with optional track rows"""
    from sglang.kernels.ops.attention.fla import gdn_replayssm_spec_decode as G
    from sglang.srt.speculative import exact_end as E

    ss, rp = p["spec_state"], p["req_pool"]
    mp = rp.mamba_pool
    G.commit_gdn_replayssm_spec(write_pos=mp.replayssm_spec_write_pos, cache_base=mp.replayssm_cache_base,
                                is_flush=mp.replayssm_is_flush, num_accepted=p["accept"], replay_indices=p["replay"],
                                max_cache_len=MAXC, max_spec_len=KDRAFT, fold_every_commit=True, null_block_id=-1)
    if snapshot:
        plan = SimpleNamespace(idx=p["snap_idx"], step=p["snap_step"])
        E.snapshot_after_advance(req_pool=rp, spec_state=ss, state_batch_indices=p["state_idx"],
                                 replay_indices=p["replay"], accept_lens=p["accept"], commit_has_track=track,
                                 plan=plan)
    G.commit_gdn_replayssm_circular(
        checkpoint_state=ss.temporal, d_cache=ss.replayssm_d, k_cache=ss.replayssm_k, g_cache=ss.replayssm_g,
        d_residual_cache=ss.replayssm_rawv, k_residual_cache=ss.replayssm_rawk, state_batch_indices=p["state_idx"],
        replay_indices=p["replay"], write_pos=mp.replayssm_spec_write_pos, cache_base=mp.replayssm_cache_base,
        is_flush=mp.replayssm_is_flush, accept_lens=p["accept"],
        mamba_track_indices=p["track_slots"] if track else None,
        mamba_steps_to_track=p["snap_step"] if track else None, null_block_id=-1)
    torch.cuda.synchronize()


def _reference(p, base_before, wp_before, flush_before, row_i):
    """float64 state after window step p['snap_step'][row_i] (the commit kernel's math, unrounded)"""
    ss = p["spec_state"]
    slot = int(p["state_idx"][row_i])
    rep = int(p["replay"][row_i])
    n_prev = int(wp_before[rep]) if int(flush_before[rep]) else 0  # FOLD: the verify folded the pending window
    start = (int(base_before[rep]) + n_prev) % MAXC
    t = int(p["snap_step"][row_i]) + 1
    phys = [(start + j) % MAXC for j in range(t)]
    h0 = ss.temporal[:, slot].double()  # [L, HV, V, K]
    g = ss.replayssm_g[:, rep][:, :, phys].double()  # [L, HV, t]
    upd = (ss.replayssm_d[:, rep][:, :, phys] + ss.replayssm_rawv[:, rep][:, :, phys].double()).double()
    keys = (ss.replayssm_k[:, rep][:, :, phys].double() + ss.replayssm_rawk[:, rep][:, :, phys].double())
    keys = keys.repeat_interleave(HV // H, dim=1)  # [L, HV, t, K]
    total = g.sum(-1)
    decay = torch.exp(total.unsqueeze(-1) - g.cumsum(-1))  # [L, HV, t]
    delta = torch.einsum("lhtv,lhtk,lht->lhvk", upd, keys, decay)
    return h0 * torch.exp(total)[..., None, None] + delta


def kernel_mode(fold: bool) -> int:
    torch.manual_seed(0)
    worst = 0.0
    # the main commit with and without radix track rows (HAS_TRACK: every extra_buffer verify has them); its fold's
    # rounding differs between the two, and the snapshot follows it (exact_end.snapshot_after_advance)
    for commit_track in ((False, True) if not fold else (False, True)):
        for case, bs in enumerate((1, 5, 17, 40)):
            worst = max(worst, _kernel_case(fold, commit_track, case, bs))
    print(f"temporal vs float64 recompute: max relative error {worst:.2e} (16-bit rounding)")
    return 0


def _kernel_case(fold, commit_track, case, bs):
    worst = 0.0
    p0 = _pools(100 + case, fold, bs)
    mp0 = p0["req_pool"].mamba_pool
    base_before = mp0.replayssm_cache_base.clone()
    wp_before = mp0.replayssm_spec_write_pos.clone()
    flush_before = mp0.replayssm_is_flush.clone()
    plain = _clone(p0)
    _step(plain, snapshot=False, track=commit_track)
    snap = _clone(p0)
    _step(snap, snapshot=True, track=commit_track)
    tracked = _clone(p0)
    _step(tracked, snapshot=False, track=True)
    # no FOLD: the live slot after a commit (same specialization) that accepted only through the snapshot step
    cut = _clone(p0)
    cut["accept"] = torch.where(p0["snap_step"] >= 0, p0["snap_step"] + 1, p0["accept"]).to(torch.int32)
    _step(cut, snapshot=False, track=commit_track)
    valid = (p0["snap_idx"] >= 0) & (p0["state_idx"] >= 0)
    # the snapshot launch leaves everything else as it was
    keep = torch.ones(SLOTS + 1, dtype=torch.bool, device="cuda")
    keep[p0["snap_idx"][p0["snap_idx"] >= 0]] = False
    a, b = plain["spec_state"].temporal, snap["spec_state"].temporal
    assert torch.equal(a[:, keep].view(torch.int16), b[:, keep].view(torch.int16)), "temporal: live disturbed"
    assert torch.equal(plain["spec_state"].conv[0][:, keep], snap["spec_state"].conv[0][:, keep])
    for k_ in ("replayssm_spec_write_pos", "replayssm_cache_base", "replayssm_is_flush"):
        assert torch.equal(getattr(plain["req_pool"].mamba_pool, k_), getattr(snap["req_pool"].mamba_pool, k_)), k_
    if bs > 3:  # the padding row's snapshot slot was not written
        pad_slot = int(p0["snap_idx"][1])
        assert torch.equal(p0["spec_state"].temporal[:, pad_slot].view(torch.int16),
                           snap["spec_state"].temporal[:, pad_slot].view(torch.int16)), "padding row written"
    n_last = n_cut = 0
    for i in range(bs):
        if not bool(valid[i]):
            continue
        s_slot = int(p0["snap_idx"][i])
        got = snap["spec_state"].temporal[:, s_slot]
        if fold:
            # the commit's own track path at the same step (mul.rn, the verify fold's rounding): the same bytes
            ref_track = tracked["spec_state"].temporal[:, int(p0["track_slots"][i])]
            assert torch.equal(got.view(torch.int16), ref_track.view(torch.int16)), (case, i, "!= track path")
        else:
            if int(p0["snap_step"][i]) == int(p0["accept"][i]) - 1:
                live = snap["spec_state"].temporal[:, int(p0["state_idx"][i])]
                assert torch.equal(got.view(torch.int16), live.view(torch.int16)), (case, i, "!= live (last)")
                n_last += 1
            live_cut = cut["spec_state"].temporal[:, int(p0["state_idx"][i])]
            assert torch.equal(got.view(torch.int16), live_cut.view(torch.int16)), (case, i, "!= live (cut)")
            n_cut += 1
        ref = _reference(p0, base_before, wp_before, flush_before, i)
        err = ((got.double() - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, err)
        assert err < 2e-2, (case, i, err)
        st = int(p0["snap_step"][i])  # conv / PLE: the verify's intermediates at that step
        assert torch.equal(snap["spec_state"].conv[0][:, s_slot], p0["spec_state"].intermediate_conv_window[0][:, i, st])
        rp = snap["req_pool"]
        assert torch.equal(rp.short_conv_pool.conv_state[:, s_slot], rp.short_conv_pool.intermediate_conv_state[:, i, st])
        assert torch.equal(rp.ngram_pool.context[s_slot], rp.ngram_pool.intermediate_context[i, st])
    if fold:
        what = f"{int(valid.sum())} snapshots bitwise the commit's track path"
    else:
        what = (f"{n_cut} snapshots bitwise the live slot of a commit that accepted through the snapshot step "
                f"({n_last} at the last step: bitwise the live slot of this commit)")
    print(f"{os.environ.get('EXACT_END_TEST_DTYPE', 'bfloat16')} FOLD={int(fold)} "
          f"compact={os.environ.get('ARC_GDN_FOLD_COMMIT_COMPACT', '0')} commit HAS_TRACK={int(commit_track)} "
          f"batch {bs:2d}: {what}; live / cursors untouched", flush=True)
    return worst


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--kernel":
        from sglang.kernels.ops.attention.fla import gdn_replayssm_spec_decode as G

        return kernel_mode(G._ARC_GDN_FOLD_IN_VERIFY)
    test_plan()
    modes = ({"ARC_GDN_FOLD_IN_VERIFY": "0"},
             {"ARC_GDN_FOLD_IN_VERIFY": "1", "ARC_GDN_FOLD_COMMIT_COMPACT": "0"},
             {"ARC_GDN_FOLD_IN_VERIFY": "1", "ARC_GDN_FOLD_COMMIT_COMPACT": "1"},
             {"ARC_GDN_FOLD_IN_VERIFY": "1", "ARC_GDN_FOLD_COMMIT_COMPACT": "check"})
    for env in [dict(m, EXACT_END_TEST_DTYPE=dt) for dt in ("bfloat16", "float16") for m in modes]:
        r = subprocess.run([sys.executable, __file__, "--kernel"], env={**os.environ, **env})
        if r.returncode != 0:
            print(f"FAILED with {env}")
            return 1
    print("EXACT_END_SNAPSHOT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
