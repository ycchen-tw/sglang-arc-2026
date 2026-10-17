"""--session-exact-resume under speculative decoding: where a finished request's sequence is exactly forwarded.

A finished request S = origin + output[:finished_len] has forwarded [0, P), P = len(origin) + finished_len - 1 (the
last token of S was sampled, never forwarded). The next request of the session can resume at P only from the mamba
state after exactly [0, P). Under spec decoding the request's live slot does not hold it:

- the stop token can be an accepted draft: the same verify forwarded the stop and the drafts accepted after it;
- the overlap scheduler runs a finished request one more verify (the overshoot), which moves the live slot on;
- with ARC_GDN_FOLD_IN_VERIFY the live checkpoint lags the last commit's window while conv / PLE do not.

So the commit of the verify in which the request outputs its first stop token (or its max_new_tokens-th token)
writes the state after that step -- temporal, conv and the PLE side states -- into a mamba slot of the request's own
(``ReqKvInfo.exact_end_slot``), with the kernels that write the radix cache's track rows, and seals the slot: a
later commit (the overshoot) does not write it again. All of it is device work inside the commit: no host sync.

Positions. A verify at seq_len s forwards [anchor, d1 .. d_{K-1}] at positions s .. s+K-1; window step j is the state
after position s+j and its prediction is the token at s+j+1 (output index s+j+1-len(origin)). The first accepted step
j whose prediction is a stop token, or the step that reaches max_new_tokens, gives P = s+j+1: the snapshot is step j.
The steps after j that the verify also accepted are output_ids[finished_len:]; their KV ([P, kv_committed_len)) is
the request's like any accepted run's, but it is not part of S (``ExactEnd.pos`` bounds S).

Device tables (``SpecState``, per req_to_token row, written on the forward stream at the request's first verify, so
an earlier owner's overshoot, earlier on that stream, still reads its own values): the end slot (-1: not armed), the
arm epoch, the length limit (len(origin) + max_new_tokens - 1) and the stop ids (Req._check_token_based_finish's
set). The seal is per slot and holds the epoch that sealed it: a slot armed again gets a new epoch, so nothing is
reset across owners.

CPU reconcile (``on_spec_finish``): the finish must be a stop token or the length limit, the position the device
wrote (``GenerationBatchResult.exact_end_pos``, copied with the result) must equal P, and kv_committed_len - P must
equal len(output_ids) - finished_len; otherwise exact_end is None and exact_end_fallback says why, one of the
contract's FALLBACK_* names (snapshot_mismatch: the reconcile failed or no snapshot was written / armed). A slot no
consumer took is freed in release_kv_cache.

Debug env (opt-in; off: nothing runs):
  ARC_EXACT_END_STATS=<path>   JSON of finish cases / fallback reasons / overshoots, rewritten every
                               ARC_EXACT_END_LOG_EVERY (500) spec finishes (a log line either way)
  ARC_EXACT_END_CHECK=1        bitwise device compares of the snapshot against the live slot where the live slot
                               holds the same state: no FOLD (16-bit checkpoint), stop at the window's last step, no
                               overshoot: all components; FOLD, stop at the last step, an overshoot verify (which
                               folded the window into the live checkpoint): temporal; FOLD, last step, no overshoot:
                               conv and PLE. The counters go with the stats report (a host sync)
  ARC_EXACT_END_DUMP=<dir>     torch.save the snapshot of up to ARC_EXACT_END_DUMP_MAX (64) exact ends per case, and
                               the live slot of prefill-finished requests whose rid starts with
                               ARC_EXACT_END_REF_PREFIX (needs --disable-overlap-schedule): the reference states
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

MAX_STOP_IDS = 16
_NO_LIMIT = 1 << 62
_epochs = itertools.count(1)
LOG_EVERY = int(os.environ.get("ARC_EXACT_END_LOG_EVERY", "500"))
STATS_PATH = os.environ.get("ARC_EXACT_END_STATS")
_check_env = os.environ.get("ARC_EXACT_END_CHECK", "0")
if _check_env not in ("0", "1"):
    raise ValueError(f"ARC_EXACT_END_CHECK must be 0 or 1, got {_check_env!r}")
CHECK = _check_env == "1"
DUMP_DIR = os.environ.get("ARC_EXACT_END_DUMP")
DUMP_MAX = int(os.environ.get("ARC_EXACT_END_DUMP_MAX", "64"))
REF_PREFIX = os.environ.get("ARC_EXACT_END_REF_PREFIX", "eeref-")

# req.exact_end_fallback: the contract's fixed set (exp/ser-park defines the same constants; its definition wins).
# The stats keep the finer detail (snapshot_mismatch: no_snapshot / reconcile_mismatch / no_snapshot_slot /
# too_many_stop_ids / vocab_boundary).
FALLBACK_RETRACT = "retract"
FALLBACK_ABORT = "abort"
FALLBACK_PREFILL_FINISHED = "prefill_finished"
FALLBACK_STOP_STRING = "stop_string"
FALLBACK_GRAMMAR = "grammar"
FALLBACK_THINK_BUDGET = "think_budget"
FALLBACK_SNAPSHOT_MISMATCH = "snapshot_mismatch"

COMPONENTS = ("temporal", "conv", "short_conv", "ngram")
CHECK_KINDS = ("nofold_last_noovershoot", "fold_last_overshoot", "fold_last_noovershoot")


def stop_ids(req: "Req") -> list[int]:
    """The token ids Req._check_token_based_finish stops at."""
    if req.sampling_params.ignore_eos:
        return []
    ids = set(req.sampling_params.stop_token_ids or ())
    if req.eos_token_ids:
        ids |= set(req.eos_token_ids)
    if req.tokenizer is not None:
        if req.tokenizer.eos_token_id is not None:
            ids.add(req.tokenizer.eos_token_id)
        if req.tokenizer.additional_stop_token_ids:
            ids |= set(req.tokenizer.additional_stop_token_ids)
    return sorted(ids)


def length_limit(req: "Req") -> int:
    """P of a length finish (finished_len = max_new_tokens)."""
    return len(req.origin_input_ids) + req.sampling_params.max_new_tokens - 1


# ---- slot lifetime (CPU, schedule time) ------------------------------------------------------------------------------
def arm_on_alloc(pool, req: "Req") -> None:
    """HybridReqToTokenPool.alloc, after the request's live and ping-pong slots: give a session's request (one with a
    session_id) an exact-end slot, once per lifetime (a retraction ends one). alloc_req_slots evicts for one more slot
    per request."""
    kv = req.kv
    if req.session_id is None:
        return  # the contract: only a session's requests have a next request to resume
    if kv.exact_end_slot is not None or kv.exact_end_unarmed is not None:
        return
    if req.grammar is not None:
        kv.exact_end_unarmed = "grammar"  # the grammar may cut the accepted run the device sees
        return
    if len(stop_ids(req)) > MAX_STOP_IDS:
        kv.exact_end_unarmed = "too_many_stop_ids"
        return
    slot = pool.mamba_allocator.alloc(1)
    if slot is None:
        kv.exact_end_unarmed = "no_snapshot_slot"
        return
    kv.exact_end_slot = slot[0]
    kv.exact_end_epoch = next(_epochs)
    kv.exact_end_row_written = False


def release(pool, req: "Req") -> None:
    """release_kv_cache, after the tree cache (and a consumer) handled the request: free the slot no consumer took
    (a consumer takes it by setting req.kv.exact_end_slot to None) and drop the ExactEnd on it."""
    kv = req.kv
    if req.exact_end is not None:
        if req.exact_end.mamba_slot is not kv.exact_end_slot:
            raise RuntimeError(f"exact end: {req.rid} reports slot {req.exact_end.mamba_slot} but holds another")
        req.exact_end = None  # no reference to a freed slot; the consumer recorded why it did not take it
    pool.mamba_allocator.free(kv.exact_end_slot.view(1))
    kv.exact_end_slot = None


# ---- device side -----------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Plan:
    """One verify's exact-end writes, per batch row: the snapshot slot (-1: none), the window step (-1: none), P (-1)."""

    idx: torch.Tensor  # int32 [bs] (the scatter kernels' index dtype)
    step: torch.Tensor  # int32 [bs]
    pos: torch.Tensor  # int64 [bs]
    verify_ct: list  # each request's exact_end_verifies at this launch


@triton.jit
def _plan_kernel(
    rows_ptr,
    seq_lens_ptr,
    predict_ptr,
    accept_index_ptr,
    accept_lens_ptr,
    slot_tab,
    epoch_tab,
    limit_tab,
    stop_tab,
    seal_ptr,
    idx_out,
    step_out,
    pos_out,
    accept_stride,
    KW: tl.constexpr,
    KW_POW2: tl.constexpr,
    N_STOP: tl.constexpr,
):
    """SpecState.plan, one program per batch row: the first accepted step whose token is a stop id, or the step of
    the length limit; fires if the row is armed and its slot is not sealed for this epoch, and seals it."""
    b = tl.program_id(0).to(tl.int64)
    row = tl.load(rows_ptr + b).to(tl.int64)
    slot = tl.load(slot_tab + row)
    epoch = tl.load(epoch_tab + row)
    limit = tl.load(limit_tab + row)
    s = tl.load(seq_lens_ptr + b).to(tl.int64)
    n_accepted = tl.load(accept_lens_ptr + b).to(tl.int64)
    o_j = tl.arange(0, KW_POW2)
    # topk 1: accept_index[b, j] = b * KW + j for the accepted steps j, -1 past them
    ai = tl.load(accept_index_ptr + b * accept_stride + o_j, mask=o_j < KW, other=-1).to(tl.int64)
    accepted = ai >= 0
    tokens = tl.load(predict_ptr + tl.where(accepted, ai, 0), mask=accepted, other=-1).to(tl.int64)
    stops = tl.load(stop_tab + row * N_STOP + tl.arange(0, N_STOP)).to(tl.int64)
    hit = tl.sum((tokens[:, None] == stops[None, :]).to(tl.int32), axis=1) > 0
    first_stop = tl.min(tl.where(hit & accepted, o_j, KW), axis=0).to(tl.int64)
    lim = limit - s - 1
    lim = tl.where((lim >= 0) & (lim < n_accepted), lim, KW)
    end = tl.minimum(first_stop, lim)
    sealed = tl.load(seal_ptr + tl.maximum(slot, 0))
    fire = (end < KW) & (slot >= 0) & (sealed != epoch)
    tl.store(seal_ptr + slot, epoch, mask=fire)
    tl.store(idx_out + b, tl.where(fire, slot, -1).to(tl.int32))
    tl.store(step_out + b, tl.where(fire, end, -1).to(tl.int32))
    tl.store(pos_out + b, tl.where(fire, s + end + 1, -1))


class SpecState:
    """Device tables of the exact-end snapshot, on the HybridReqToTokenPool (``pool.exact_end_spec``)."""

    def __init__(self, pool, device):
        mamba_pool = pool.mamba_pool
        rows = pool.req_to_token.shape[0]
        self.device = device
        self.forward_stream = None  # the stream of the first verify (run_eagle_verify's current stream)
        self.slot = torch.full((rows,), -1, dtype=torch.int64, device=device)
        self.epoch = torch.zeros((rows,), dtype=torch.int64, device=device)
        self.limit = torch.full((rows,), _NO_LIMIT, dtype=torch.int64, device=device)
        self.stop = torch.full((rows, MAX_STOP_IDS), -1, dtype=torch.int32, device=device)
        # seal[slot] = the epoch that sealed it (slots 0 .. size)
        self.seal = torch.full((mamba_pool.size + 1,), -1, dtype=torch.int64, device=device)
        # no FOLD: the snapshot launch must leave the live checkpoint to the commit's own launch
        self.zero_flush = torch.zeros_like(mamba_pool.replayssm_is_flush)
        self.minus_one = torch.full((), -1, dtype=torch.int64, device=device)
        self.stats = Stats(self)
        if CHECK:
            # [kind, component, (checked, mismatched)]; max |snapshot - live| of temporal per kind
            self.check_counts = torch.zeros((len(CHECK_KINDS), len(COMPONENTS), 2), dtype=torch.int64, device=device)
            self.check_maxdiff = torch.zeros((len(CHECK_KINDS),), dtype=torch.float32, device=device)

    def _write_rows(self, reqs, pending) -> None:
        values = np.full((len(pending), 3 + MAX_STOP_IDS), -1, dtype=np.int64)
        slots = []
        for j, i in enumerate(pending):
            req = reqs[i]
            kv = req.kv
            values[j, 0] = kv.req_pool_idx
            if kv.exact_end_slot is not None:
                values[j, 1] = kv.exact_end_epoch
                values[j, 2] = length_limit(req)
                ids = stop_ids(req)
                values[j, 3 : 3 + len(ids)] = ids
                slots.append(kv.exact_end_slot.view(()).to(torch.int64))
            else:  # not armed: the row must not keep an earlier owner's slot
                values[j, 1] = 0
                values[j, 2] = _NO_LIMIT
                slots.append(self.minus_one)
            kv.exact_end_row_written = True
        # pinned + non-blocking: the caching host allocator keeps the buffer until the copy ran (no stream sync)
        dev = torch.from_numpy(values).pin_memory().to(self.device, non_blocking=True)
        rows = dev[:, 0]
        self.slot.index_copy_(0, rows, torch.stack(slots))
        self.epoch.index_copy_(0, rows, dev[:, 1])
        self.limit.index_copy_(0, rows, dev[:, 2])
        self.stop.index_copy_(0, rows, dev[:, 3:].to(torch.int32))

    @torch.no_grad()
    def plan(self, batch: "ScheduleBatch", predict: torch.Tensor, accept_index: torch.Tensor,
             accept_lens: torch.Tensor) -> Plan:
        """run_eagle_verify, after the accept and before the mamba commit, on the forward stream."""
        if self.forward_stream is None:
            self.forward_stream = torch.cuda.current_stream()
        reqs = batch.reqs
        pending = []
        verify_ct = []
        for i, req in enumerate(reqs):
            req.exact_end_verifies += 1
            verify_ct.append(req.exact_end_verifies)
            if not req.kv.exact_end_row_written:
                pending.append(i)
        if pending:
            self._write_rows(reqs, pending)
        bs = len(reqs)
        if accept_index.shape[0] != bs or accept_index.stride(1) != 1 or accept_lens.shape != (bs,):
            raise RuntimeError(f"exact end: accept_index {tuple(accept_index.shape)} stride {accept_index.stride()}, "
                               f"accept_lens {tuple(accept_lens.shape)} for {bs} rows")
        for name, t in (("predict", predict), ("req_pool_indices", batch.req_pool_indices),
                        ("seq_lens", batch.seq_lens), ("accept_lens", accept_lens)):
            if not t.is_contiguous():
                raise RuntimeError(f"exact end: {name} is not contiguous ({tuple(t.shape)}, {t.stride()})")
        kw = accept_index.shape[1]
        idx = torch.empty((bs,), dtype=torch.int32, device=self.device)
        step = torch.empty((bs,), dtype=torch.int32, device=self.device)
        pos = torch.empty((bs,), dtype=torch.int64, device=self.device)
        _plan_kernel[(bs,)](
            batch.req_pool_indices, batch.seq_lens, predict, accept_index, accept_lens,
            self.slot, self.epoch, self.limit, self.stop, self.seal, idx, step, pos,
            accept_index.stride(0), KW=kw, KW_POW2=triton.next_power_of_2(kw), N_STOP=MAX_STOP_IDS,
        )
        return Plan(idx=idx, step=step, pos=pos, verify_ct=verify_ct)


def validate_and_init(model_runner, draft_worker, scheduler_pool) -> None:
    """Scheduler.__init__, once the request pool exists, under --session-exact-resume with spec decoding: refuse what
    the snapshot does not cover, then attach the device tables to the request pool."""
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import _ARC_GDN_FOLD_IN_VERIFY
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.runtime_context import get_memory, get_schedule
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    spec = get_spec()
    pool = model_runner.req_to_token_pool
    if pool is not scheduler_pool:
        raise RuntimeError("exact end: the scheduler's request pool is not the target model runner's")
    refused = []
    if type(draft_worker) is not EAGLEWorkerV2:
        # the plan runs in run_eagle_verify, called with the target model runner's pool by EAGLEWorkerV2.verify
        refused.append(f"the EAGLE v2 worker (NEXTN / EAGLE, with or without ARC_DFLASH_DRAFTER), not "
                       f"{type(draft_worker).__name__}")
    if spec.speculative_eagle_topk != 1:
        refused.append("a linear draft chain (--speculative-eagle-topk 1)")
    if type(pool) is not HybridReqToTokenPool:
        refused.append(f"a hybrid GDN model on the static HybridReqToTokenPool (got {type(pool).__name__})")
    else:
        mamba_pool = pool.mamba_pool
        if mamba_pool.replayssm_cache_base is None or mamba_pool.replayssm_is_kda:
            refused.append("the GDN ReplaySSM circular ring (--enable-linear-replayssm-spec on a GDN model)")
        else:
            spec_state = pool.get_speculative_mamba2_params_all_layers()
            if spec_state.temporal.dtype == torch.uint8 and not _ARC_GDN_FOLD_IN_VERIFY:
                refused.append("ARC_GDN_FOLD_IN_VERIFY with a packed GDN state (ARC_GDN_STATE_FMT)")
            if len(spec_state.conv) != 1 or len(spec_state.intermediate_conv_window) != 1:
                refused.append("one conv state tensor")
    if get_schedule().enable_mixed_chunk:
        refused.append("no --enable-mixed-chunk (a decoding request would step inside an extend)")
    if get_memory().enable_hisparse:
        refused.append("no --enable-hisparse (its session cache keeps its own snapshots)")
    if refused:
        raise ValueError("--session-exact-resume under speculative decoding needs " + "; ".join(refused))
    pool.exact_end_spec = SpecState(pool, model_runner.device)
    logger.info("session exact resume (spec): end snapshots in the verify commit (%s), stop ids <= %d, check %s, "
                "dump %s", "ARC_GDN_FOLD_IN_VERIFY" if _ARC_GDN_FOLD_IN_VERIFY else "fold at commit", MAX_STOP_IDS,
                CHECK, DUMP_DIR)


@torch.no_grad()
def snapshot_after_advance(*, req_pool, spec_state, state_batch_indices, replay_indices, accept_lens,
                           commit_has_track: bool, plan: Plan) -> None:
    """commit_mamba_states_after_verify (GDN ReplaySSM circular ring), after the cursor advance and before the
    commit's own launch: the track-row kernels with the plan's slots and steps.

    temporal: the commit kernel's track path, from the pre-window checkpoint and the window's ring rows. Without
    FOLD the commit's own launch folds the checkpoint in place, so this one runs first, with is_flush zero (track
    rows only). The fold's rounding depends on that launch's specialization: with track rows (HAS_TRACK, every
    extra_buffer verify) it is the default track store's, without them it compiles to an FMA, which TRACK_FMA
    reproduces (``commit_has_track``; scripts/arc/test_exact_end_snapshot.py checks both). With FOLD the commit
    writes track rows only and the next verify folds with the track store's rounding (mul.rn); this is the same
    launch over other rows. conv / PLE: the verify's per-step intermediates, as for the track rows. The plan's
    int32 indices are the scatter kernels' own dtype (no conversion launches)."""
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
        _ARC_GDN_COMMIT_LIVE_ROWS,
        _ARC_GDN_FOLD_IN_VERIFY,
        launch_gdn_replayssm_compact_commit,
    )
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend

    mamba_pool = req_pool.mamba_pool
    launch_gdn_replayssm_compact_commit(
        checkpoint_state=spec_state.temporal,
        d_cache=spec_state.replayssm_d,
        k_cache=spec_state.replayssm_k,
        g_cache=spec_state.replayssm_g,
        d_residual_cache=spec_state.replayssm_rawv,
        k_residual_cache=spec_state.replayssm_rawk,
        state_batch_indices=state_batch_indices,
        replay_indices=replay_indices,
        write_pos=mamba_pool.replayssm_spec_write_pos,
        cache_base=mamba_pool.replayssm_cache_base,
        is_flush=(mamba_pool.replayssm_is_flush if _ARC_GDN_FOLD_IN_VERIFY else req_pool.exact_end_spec.zero_flush),
        accept_lens=accept_lens,
        mamba_track_indices=plan.idx,
        mamba_steps_to_track=plan.step,
        null_block_id=-1,
        live_rows=_ARC_GDN_COMMIT_LIVE_ROWS,
        fold_in_verify=_ARC_GDN_FOLD_IN_VERIFY,
        track_fma=not _ARC_GDN_FOLD_IN_VERIFY and not commit_has_track,
        slots=1,  # FOLD compacted: a verify has 0 or a few end rows; 8 slots cost ~4x the programs for nothing
    )
    fused_conv_window_scatter_with_mask(spec_state.conv[0], spec_state.intermediate_conv_window[0], plan.idx,
                                        plan.step)
    for dst, src in _ple_pairs(req_pool):
        HybridLinearAttnBackend._scatter_speculative_state_with_mask(dst, src, plan.idx, plan.step)


def _ple_pairs(req_pool):
    """(state, per-step intermediate) of the Qwen4-Exp PLE side states, as _update_ple_state_after_mtp_verify."""
    pairs = []
    sc = req_pool.short_conv_pool
    if sc.conv_state is not None and sc.intermediate_conv_state is not None:
        pairs.append((sc.conv_state, sc.intermediate_conv_state))
    ng = req_pool.ngram_pool
    if ng.context is not None and ng.intermediate_context is not None:
        pairs.append((ng.context.unsqueeze(0), ng.intermediate_context.unsqueeze(0)))
    return pairs


def _components(req_pool, slot: torch.Tensor) -> dict:
    """The per-slot states the snapshot covers, gathered for one slot (``slot``: a 1-element device index)."""
    spec_state = req_pool.get_speculative_mamba2_params_all_layers()
    out = {"temporal": spec_state.temporal.index_select(1, slot), "conv": spec_state.conv[0].index_select(1, slot)}
    if req_pool.short_conv_pool.conv_state is not None:
        out["short_conv"] = req_pool.short_conv_pool.conv_state.index_select(1, slot)
    if req_pool.ngram_pool.context is not None:
        out["ngram"] = req_pool.ngram_pool.context.index_select(0, slot)
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


# ---- CPU side: finish, overshoot -------------------------------------------------------------------------------------
def on_prefill_finish(req: "Req", req_pool, overlap: bool) -> None:
    """A request that finished in its prefill step: no verify wrote a snapshot."""
    state = req_pool.exact_end_spec
    # an overshoot decode verify (overlap) starts where the prefill ended
    state.stats.note_finish(req.rid, len(req.origin_input_ids) + len(req.output_ids) - 1, (0, req.kv.kv_committed_len),
                            req.kv.kv_committed_len, get_spec().speculative_num_draft_tokens)
    if req.session_id is not None:
        req.exact_end = None
        req.exact_end_fallback = FALLBACK_PREFILL_FINISHED
        state.stats.fallback(req.exact_end_fallback)
    if DUMP_DIR and req.rid.startswith(REF_PREFIX):
        if overlap:
            raise RuntimeError("ARC_EXACT_END_DUMP reference states need --disable-overlap-schedule (an overshoot "
                               "decode would move the live slot past the prompt)")
        _dump(state, req_pool, req, "ref", req.kv.mamba_pool_idx.view(1).to(torch.int64), len(req.origin_input_ids),
              {})


def _fallback_reason(req: "Req") -> Optional[tuple[str, str]]:
    """(req.exact_end_fallback, the stats' detail) of a spec finish no snapshot can stand for, else None."""
    from sglang.srt.managers.schedule_batch import (
        FINISH_ABORT,
        FINISH_LENGTH,
        FINISH_MATCHED_STR,
        FINISH_MATCHED_TOKEN,
        FINISHED_MATCHED_REGEX,
    )

    reason = req.finished_reason
    if not isinstance(reason, (FINISH_ABORT, FINISH_MATCHED_STR, FINISHED_MATCHED_REGEX, FINISH_MATCHED_TOKEN,
                               FINISH_LENGTH)):
        raise RuntimeError(f"exact end: {req.rid} finished with an unknown reason {type(reason).__name__}")
    if isinstance(reason, FINISH_ABORT):
        return FALLBACK_ABORT, FALLBACK_ABORT
    if req.retraction_count > 0:
        return FALLBACK_RETRACT, FALLBACK_RETRACT
    if req.grammar is not None:
        return FALLBACK_GRAMMAR, FALLBACK_GRAMMAR
    if req.arc_think_armed_at >= 0:
        return FALLBACK_THINK_BUDGET, FALLBACK_THINK_BUDGET
    if isinstance(reason, FINISH_MATCHED_STR) and reason.matched == "NaN happened":
        return FALLBACK_SNAPSHOT_MISMATCH, "vocab_boundary"  # an out-of-vocab id was rewritten on the host
    if isinstance(reason, (FINISH_MATCHED_STR, FINISHED_MATCHED_REGEX)):
        return FALLBACK_STOP_STRING, FALLBACK_STOP_STRING
    if req.finished_len is None:
        raise RuntimeError(f"exact end: {req.rid} finished on {type(reason).__name__} without finished_len")
    if req.kv.exact_end_slot is None:
        if req.kv.exact_end_unarmed is None:
            raise RuntimeError(f"exact end: {req.rid} decoded without an exact-end slot or a reason for none")
        return FALLBACK_SNAPSHOT_MISMATCH, req.kv.exact_end_unarmed
    return None


def on_spec_finish(req: "Req", result, i: int, req_pool) -> None:
    """_handle_finish_state_updated_req: spec decode result ``result`` (row ``i``) finished ``req``; before
    release_kv_cache. Sets req.exact_end or req.exact_end_fallback."""
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import _ARC_GDN_FOLD_IN_VERIFY as fold

    state = req_pool.exact_end_spec
    finished_len = req.finished_len if req.finished_len is not None else len(req.output_ids)
    pos = len(req.origin_input_ids) + finished_len - 1
    after_stop = len(req.output_ids) - finished_len
    if result.exact_end_pos is None or not result.exact_end_pos.is_cpu:
        raise RuntimeError(f"exact end: {req.rid} finished in a spec decode result without a host exact_end_pos")
    # the finishing verify forwarded [s, s + K); an overshoot verify starts at kv_committed_len
    accept_len = int(result.accept_lens[i])
    s0 = req.kv.kv_committed_len - accept_len
    state.stats.note_finish(req.rid, pos, (s0, s0 + result.speculative_num_draft_tokens), req.kv.kv_committed_len,
                            result.speculative_num_draft_tokens)
    if req.session_id is None:
        state.stats.no_session += 1
        return
    fallback = _fallback_reason(req)
    device_pos = int(result.exact_end_pos[i])
    if fallback is None:
        if device_pos < 0:
            fallback = FALLBACK_SNAPSHOT_MISMATCH, "no_snapshot"
        elif device_pos != pos or req.kv.kv_committed_len - pos != after_stop:
            fallback = FALLBACK_SNAPSHOT_MISMATCH, "reconcile_mismatch"
            logger.error("exact end: %s device P %d, host P %d, kv_committed_len %d, after the stop %d (%s)", req.rid,
                         device_pos, pos, req.kv.kv_committed_len, after_stop, type(req.finished_reason).__name__)
    overshoot = req.exact_end_verifies > result.exact_end_verify_ct[i]
    last = after_stop == 0
    state.stats.finish(req, None if fallback is None else fallback[1], last, overshoot, after_stop, device_pos >= 0)
    if fallback is not None:
        req.exact_end = None
        req.exact_end_fallback = fallback[0]
        return
    from sglang.srt.managers.schedule_batch import ExactEnd

    req.exact_end = ExactEnd(pos=pos, mamba_slot=req.kv.exact_end_slot)
    req.exact_end_fallback = None
    if CHECK:
        _check(state, req_pool, req, last, overshoot, fold)
    if DUMP_DIR:
        _dump(state, req_pool, req, ("last" if last else "mid") + ("_os" if overshoot else ""),
              req.kv.exact_end_slot.view(1).to(torch.int64), pos,
              {"finished_len": finished_len, "output_len": len(req.output_ids), "overshoot": overshoot,
               "fold": fold, "finish": type(req.finished_reason).__name__})


def on_overshoot(req: "Req", result, i: int, req_pool) -> None:
    """process_batch_result_decode skipped row ``i`` of a spec result: ``req`` had finished (overlap)."""
    req_pool.exact_end_spec.stats.overshoot(req.rid, int(result.accept_lens[i]))


def _wait_forward(state: SpecState) -> None:
    """The debug reads run on the current (result processing) stream, behind everything launched on the forward
    stream so far: the overlap scheduler's forward stream, set by the first verify (a spec finish or overshoot comes
    after one); without overlap the forward runs on the result processing stream itself."""
    if state.forward_stream is not None:
        torch.cuda.current_stream().wait_stream(state.forward_stream)


def _check(state: SpecState, req_pool, req: "Req", last: bool, overshoot: bool, fold: bool) -> None:
    if not last:
        return
    spec_state = req_pool.get_speculative_mamba2_params_all_layers()
    if not fold and not overshoot:
        kind = 0
        names = COMPONENTS if spec_state.temporal.dtype != torch.float32 else COMPONENTS[1:]
    elif fold and overshoot:
        kind, names = 1, ("temporal",)
    elif fold:
        kind, names = 2, COMPONENTS[1:]
    else:
        return
    # result processing's (schedule) stream, behind the forward stream: after the overshoot verify (already launched)
    # and before any later owner of the live slot (its writes come later on this stream, or in a forward that waits
    # for it). Not on the forward stream: its allocations here could reuse a forward-pool block the schedule stream
    # still reads (the base frees those behind its stream waits; 10-17, device asserts in later gathers)
    _wait_forward(state)
    snap = _components(req_pool, req.kv.exact_end_slot.view(1).to(torch.int64))
    live = _components(req_pool, req.kv.mamba_pool_idx.view(1).to(torch.int64))
    for c, name in enumerate(COMPONENTS):
        if name not in names or name not in snap:
            continue
        state.check_counts[kind, c, 0] += 1
        state.check_counts[kind, c, 1] += (_bits(snap[name]) != _bits(live[name])).any().to(torch.int64)
        if name == "temporal" and snap[name].is_floating_point():
            state.check_maxdiff[kind] = torch.maximum(
                state.check_maxdiff[kind], (snap[name].float() - live[name].float()).abs().amax())
    state.stats.checked[CHECK_KINDS[kind]] += 1


def _check_counters(state: SpecState) -> dict:
    """ARC_EXACT_END_CHECK counters (a host sync: debug only)."""
    counts = state.check_counts.tolist()
    maxdiff = state.check_maxdiff.tolist()
    return {kind: {"max_abs_diff_temporal": maxdiff[k],
                   **{name: {"checked": counts[k][c][0], "mismatched": counts[k][c][1]}
                      for c, name in enumerate(COMPONENTS)}}
            for k, kind in enumerate(CHECK_KINDS)}


def _dump(state: SpecState, req_pool, req: "Req", kind: str, slot: torch.Tensor, pos: int, meta: dict) -> None:
    n = state.stats.dumped[kind]
    if n >= DUMP_MAX:
        return
    state.stats.dumped[kind] += 1
    _wait_forward(state)  # as _check
    comps = {k: v.to("cpu") for k, v in _components(req_pool, slot).items()}
    ids = list(req.origin_input_ids) + list(req.output_ids)
    out = Path(DUMP_DIR)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"rid": req.rid, "kind": kind, "pos": pos, "ids": ids[:pos], "meta": meta, "state": comps},
               out / f"{kind}-{n:04d}-{req.rid}.pt")


class Stats:
    """CPU counters of the spec exact ends: a log line and (ARC_EXACT_END_STATS) a JSON every LOG_EVERY finishes."""

    def __init__(self, state: "SpecState"):
        self.state = state
        self.finishes = 0
        self.cases = collections.Counter()
        self.fallbacks = collections.Counter()
        self.after_stop = collections.Counter()  # accepted tokens of the finishing verify after the stop
        self.overshoot_accept = collections.Counter()  # accept length of the overshoot verifies
        self.finished = collections.OrderedDict()  # rid -> P and the verify windows since its finish
        self.no_session = 0  # spec finishes of requests without a session_id (not armed, not reported)
        self.overshoot_multi = 0
        self.checked = collections.Counter()
        self.dumped = collections.Counter()

    def fallback(self, reason: str) -> None:
        self.fallbacks[reason] += 1

    def finish(self, req, reason, last, overshoot, after_stop, fired) -> None:
        self.finishes += 1
        if reason is not None:
            self.fallbacks[reason] += 1
        self.cases[f"{type(req.finished_reason).__name__}|{'bonus' if last else 'draft'}|"
                   f"{'overshoot' if overshoot else 'none'}|{'exact' if reason is None else 'fallback'}"
                   f"{'' if fired else '|nofire'}"] += 1
        self.after_stop[after_stop] += 1
        if self.finishes % LOG_EVERY == 0:
            self.report()

    def note_finish(self, rid: str, pos: int, window: tuple, committed: int, k: int) -> None:
        """A finish: P, the positions [start, end) the finishing forward (a verify, or the prefill) wrote, and where
        an overshoot verify starts (kv_committed_len)."""
        self.finished[rid] = {"P": pos, "windows": [window], "next": committed, "k": k}
        while len(self.finished) > 100000:
            self.finished.popitem(last=False)

    def overshoot(self, rid: str, accept_len: int) -> None:
        self.overshoot_accept[accept_len] += 1
        rec = self.finished.get(rid)
        if rec is None:
            raise RuntimeError(f"exact end: an overshoot verify of {rid}, whose finish was never seen")
        start = rec["next"]
        rec["windows"].append((start, start + rec["k"]))
        rec["next"] = start + accept_len
        if len(rec["windows"]) > 2:
            # one overshoot window is what the end snapshot's seal and ser-qsa's QSA ring slack assume
            self.overshoot_multi += 1
            raise RuntimeError(f"exact end: {rid} (P {rec['P']}) ran {len(rec['windows']) - 1} verifies after the "
                               f"one that finished it; windows written [start, end): {rec['windows']}")

    def as_dict(self) -> dict:
        return {"finishes": self.finishes, "cases": dict(self.cases), "fallbacks": dict(self.fallbacks),
                "after_stop": {str(k): v for k, v in sorted(self.after_stop.items())},
                "overshoot_accept": {str(k): v for k, v in sorted(self.overshoot_accept.items())},
                "overshoot_verifies": sum(self.overshoot_accept.values()), "overshoot_multi": self.overshoot_multi,
                "no_session": self.no_session,
                "checked": dict(self.checked), "dumped": dict(self.dumped)}

    def report(self) -> None:
        d = self.as_dict()
        if CHECK:
            d["check"] = _check_counters(self.state)  # a host sync: debug only
        logger.info("exact end (spec): %s", json.dumps(d, sort_keys=True))
        if STATS_PATH:
            tmp = Path(STATS_PATH + ".tmp")
            tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
            tmp.replace(STATS_PATH)
