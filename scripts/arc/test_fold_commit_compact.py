#!/usr/bin/env python3
"""GPU check: ARC_GDN_FOLD_COMMIT_COMPACT's track kernel writes exactly the compact commit kernel's bytes.

Random ReplaySSM states and rings at the served per-rank shape (36 GDN layers, 24 value heads / 8 key heads of 128,
ring 16, bf16 checkpoint with residual caches), batches of 1..128 rows mixing padding rows (slot -1), rows without a
track slot, rows with a slot but no step this cycle, and track rows (none / a few / all). For each batch the full
grid (launch_gdn_replayssm_compact_commit(fold_in_verify=True, compact="0")) and the compacted launch
(compact="1") run on copies of the same pool; the two pools must be equal byte for byte, and the track slots must
have been written (not the inputs). Also times both launches at 128 rows with 2 track rows.

  python scripts/arc/test_fold_commit_compact.py
"""

import os
import sys

os.environ.setdefault("ARC_GDN_FOLD_IN_VERIFY", "1")

import torch  # noqa: E402

from sglang.kernels.ops.attention.fla import gdn_replayssm_spec_decode as G  # noqa: E402

L, HV, H, K, V, MAXC = 36, 24, 8, 128, 128, 16


def make(batch, n_track, seed, slots=300, rows=260):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = torch.bfloat16
    state = (0.1 * torch.randn(L, slots, HV, V, K, device="cuda", generator=g)).to(dt)
    d = (0.1 * torch.randn(L, rows, HV, MAXC, V, device="cuda", generator=g)).to(dt)
    k = (0.1 * torch.randn(L, rows, H, MAXC, K, device="cuda", generator=g)).to(dt)
    gg = (-0.05 * torch.rand(L, rows, HV, MAXC, device="cuda", generator=g)).float()
    dr = (1e-3 * torch.randn(L, rows, HV, MAXC, V, device="cuda", generator=g)).to(dt)
    kr = (1e-3 * torch.randn(L, rows, H, MAXC, K, device="cuda", generator=g)).to(dt)
    perm = torch.randperm(slots - 1, generator=g, device="cuda")
    state_idx = perm[:batch].to(torch.int64)
    track_pool = perm[batch : 2 * batch].to(torch.int64)
    replay = torch.randperm(rows - 1, generator=g, device="cuda")[:batch].to(torch.int64)
    accept = torch.randint(1, 5, (batch,), device="cuda", generator=g)
    write_pos = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    base = torch.randint(0, MAXC, (rows + 1,), dtype=torch.int32, device="cuda", generator=g)
    flush = torch.randint(0, 2, (rows + 1,), dtype=torch.int8, device="cuda", generator=g)
    hist = accept + torch.randint(0, MAXC - 4, (batch,), device="cuda", generator=g)
    write_pos[replay] = hist.to(torch.int32)
    track_idx = torch.full((batch,), -1, dtype=torch.int64, device="cuda")
    steps = torch.full((batch,), -1, dtype=torch.int64, device="cuda")
    # every row has a track slot assigned, as under extra_buffer; n_track of them snapshot this cycle
    track_idx[:] = track_pool
    chosen = torch.randperm(batch, generator=g, device="cuda")[:n_track]
    steps[chosen] = (torch.rand(n_track, device="cuda", generator=g) * accept[chosen]).long()
    if batch > 3:
        state_idx[1] = -1  # a padding row (its step is set too: it must still be skipped)
        steps[1] = 0
        track_idx[2] = -1  # no track slot
        steps[2] = 0
    return dict(checkpoint_state=state, d_cache=d, k_cache=k, g_cache=gg, d_residual_cache=dr, k_residual_cache=kr,
                state_batch_indices=state_idx, replay_indices=replay, write_pos=write_pos, cache_base=base,
                is_flush=flush, accept_lens=accept, mamba_track_indices=track_idx, mamba_steps_to_track=steps,
                null_block_id=-1, live_rows=True, fold_in_verify=True)


def run(args, compact):
    out = dict(args)
    out["checkpoint_state"] = args["checkpoint_state"].clone()
    G.launch_gdn_replayssm_compact_commit(**out, compact=compact)
    torch.cuda.synchronize()
    return out["checkpoint_state"]


def main() -> int:
    worst = 0
    cases = [(1, 0), (1, 1), (7, 3), (64, 0), (120, 2), (128, 2), (128, 17), (128, 128), (97, 50)]
    for i, (batch, n_track) in enumerate(cases):
        args = make(batch, n_track, i)
        full = run(args, "0")
        comp = run(args, "1")
        same = torch.equal(full.view(torch.uint8), comp.view(torch.uint8))
        changed = int((full.view(torch.uint8) != args["checkpoint_state"].view(torch.uint8)).flatten(2).any(-1).sum())
        print(f"batch {batch:3d} track rows {n_track:3d}: bitwise equal {same}; (layer, slot) pairs written "
              f"{changed}", flush=True)
        if not same:
            return 1
        worst = max(worst, changed)
    if worst == 0:
        print("no case wrote a track slot: the test exercised nothing")
        return 1
    for n_track in (0, 2, 8, 32, 128):  # serving: 0-2 track rows per cycle; 128 is the worst case
        args = make(128, n_track, 99)
        for compact in ("0", "1"):
            out = dict(args)
            for _ in range(3):
                G.launch_gdn_replayssm_compact_commit(**out, compact=compact)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(20):
                G.launch_gdn_replayssm_compact_commit(**out, compact=compact)
            e.record()
            torch.cuda.synchronize()
            print(f"compact={compact}: {s.elapsed_time(e) / 20:.3f} ms per commit (128 rows, {n_track} track rows)")
    print("FOLD_COMMIT_COMPACT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
