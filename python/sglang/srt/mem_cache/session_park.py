"""Session tail park: a session's next request resumes exactly where the last one
ended, so every position of a multi-turn rollout is forwarded once
(--session-exact-resume; arc-forge docs/plans/session_exact_resume.md, phase 1).

Without it the hybrid-mamba radix cache keeps a finished request only up to its
last mamba track point T (page / track-interval aligned) and frees the rest: the
next turn re-prefills [T, end) -- forwarded twice, with MoE routing that may differ
from the first time.

Life of a park (one per finished request; a session keeps at most two):

  park     cache_finished_req of a request with a session_id that ended cleanly at
           a known ExactEnd P (non-spec: P = kv_committed_len, the live mamba slot).
           The tree gets the aligned prefix [0, T) exactly as at a chunk boundary
           (dedup, the row re-pointed at the tree's copies, the lock moved to the T
           node); its mamba snapshot at T stays a tree checkpoint, so a fallback
           still resumes at T. The request's record moves into the park: the row
           (with its QSA pending ring), the KV [T, P) (whole pages, the last one
           partial), the mamba slot after [0, P), the lock on the T node. The
           ping-pong track slots are freed.
  claim    match_prefix of a waiting request of the same session whose ids equal the
           park's on [0, P) under the same (cache_salt, extra_key), whose matchable
           key reaches P: prefix = P, cache_protected_len = T, last node = T node.
           Exclusive: a second request (concurrent, a resend) falls back to the tree
           (reason "claimed"). Nothing moves; a re-match returns the same result.
  consume  PrefillAdder commits the claimant (its own lock on the T node taken): the
           park's lock is released and the request takes the record (row, KV, mamba
           slot) and extends from P in place.
  release  alloc shortfall the tree cannot cover (evict_for_alloc, last resort),
           capacity, the per-session limit, or a claimant aborted while waiting
           (the claim is handed back, the park stays). reset (flush_cache) drops
           the parks; the caller clears the pools wholesale.

Accounting: an unclaimed park's KV pages and mamba slot count as evictable, a
claimed park's as session-held; every park holds one req row, taken from
``session_park_rows`` reserved on top of max_running_requests.

Every decision depends only on the request stream (logical sequence numbers, no
wall clock), so all TP ranks take the same ones.
"""

from __future__ import annotations

import logging
from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    FINISHED_MATCHED_REGEX,
    ExactEnd,
    ReqKvInfo,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_serving,
    get_spec,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

MAX_PARKS_PER_SESSION = 2
TOMBSTONE_LIMIT = 1 << 16

# QSA extends accept a prefix that is not a multiple of the compress ratio only when
# the request's own pending-ring row holds [floor4(P), P) and the forward batch says
# so (Req.qsa_ring_resume, exp/ser-qsa-misaligned). Until that lands such ends do
# not park (fallback "qsa_misaligned").
QSA_MISALIGNED_RESUME = False

# meta_info["session_resume"]["reason"] of a fallback: the contract's fixed names
# (arc-forge docs/plans/session_exact_resume.md); exp/ser-spec-snapshot uses the
# same set. Reasons a request sees at its match:
NO_PARK = "no_park"  # the session has nothing parked (first turn, new session, restart)
CLAIMED = "claimed"  # another request of the session claimed the matching park first
IDS_MISMATCH = "ids_mismatch"  # rewind / branch / another cache_salt or extra_key
KEY_LIMIT = "key_limit"  # ids agree, the matchable key ends before P (logprob start, no tail)
RETRACT = "retract"  # retracted: re-admitted from the tree
ABORT = "abort"  # aborted before admission
UNSUPPORTED = "unsupported"  # session_params (session controller) request
# ... and why the session's last request did not park (its tombstone):
EVICTED = "evicted"  # released for memory or capacity, or no park row was free
PREFILL_FINISHED = "prefill_finished"  # ended in its prefill
STOP_STRING = "stop_string"  # stop string / regex: output trimmed past the KV
GRAMMAR = "grammar"
THINK_BUDGET = "think_budget"
SNAPSHOT_MISMATCH = "snapshot_mismatch"  # spec: the end snapshot does not reconcile
QSA_MISALIGNED = "qsa_misaligned"  # P % compress ratio != 0 (until exp/ser-qsa-misaligned)
SPEC_UNSUPPORTED = "spec_unsupported"  # spec decode (until exp/ser-spec-snapshot)
# (ABORT, UNSUPPORTED -- positional embed overrides, beam search -- as well.)
RESUME_REASONS = frozenset(
    {
        NO_PARK,
        CLAIMED,
        IDS_MISMATCH,
        KEY_LIMIT,
        EVICTED,
        RETRACT,
        ABORT,
        PREFILL_FINISHED,
        STOP_STRING,
        GRAMMAR,
        THINK_BUDGET,
        SNAPSHOT_MISMATCH,
        UNSUPPORTED,
        QSA_MISALIGNED,
        SPEC_UNSUPPORTED,
    }
)

_REASON_RANK = {CLAIMED: 3, KEY_LIMIT: 2, IDS_MISMATCH: 1}

EXACT = ("exact", None)
NOT_A_SESSION = ("none", None)


def _fallback(reason: str) -> tuple:
    if reason not in RESUME_REASONS:
        raise AssertionError(f"session resume reason {reason!r} is not in the contract")
    return ("fallback", reason)


def session_resume_meta(req: Req, *, exact_resume: bool) -> dict:
    """meta_info["session_resume"] of every request:
    {"mode": "exact" | "fallback" | "none", "reason", "pos", "prefill_version"}.
    mode / reason come from the request's last scheduling match (req.session_resume);
    pos is the prefix of its last admission and prefill_version the weight version
    of the first prefill forward after it, both None before its first admission.
    "none": the flag is off or the request has no session_id."""
    pos, version = req.admitted_prefix_len, req.prefill_weight_version
    if not exact_resume:
        mode, reason = NOT_A_SESSION
        pos = None
    elif req.session_resume is None:
        # Never reached a scheduling match: aborted while queued.
        if pos is not None:
            raise AssertionError(f"session resume: {req.rid} admitted without a match")
        mode, reason = _fallback(ABORT)
    else:
        (mode, reason), park_pos = req.session_resume
        if mode == "none":
            pos = None
        elif mode == "exact" and pos is not None and pos != park_pos:
            raise AssertionError(
                f"session resume: {req.rid} claimed a park ending at {park_pos}, "
                f"admitted with prefix {pos}"
            )
    return {"mode": mode, "reason": reason, "pos": pos, "prefill_version": version}


def session_park_rows(max_running_requests: int) -> int:
    """Req rows reserved for parks on top of the running requests' rows."""
    if not get_serving().session_exact_resume:
        return 0
    n = get_serving().session_exact_resume_max_parks
    if n is None:
        return max_running_requests
    if n <= 0:
        raise ValueError(f"--session-exact-resume-max-parks must be > 0, got {n}")
    return n


def _ids_view(tokens) -> np.ndarray:
    if isinstance(tokens, array):
        if tokens.typecode != "q":
            raise TypeError(f"token array typecode {tokens.typecode!r}, need 'q'")
        return np.frombuffer(tokens, dtype=np.int64) if len(tokens) else np.empty(0, np.int64)
    return np.asarray(tokens, dtype=np.int64)


class _RowHolder:
    """ReqToTokenPool.free takes an owner with a ``kv`` record."""

    __slots__ = ("kv",)

    def __init__(self, kv: ReqKvInfo):
        self.kv = kv


@dataclass(eq=False)
class _Park:
    seq: int
    session_id: str
    cache_salt: Optional[str]
    extra_key: Optional[str]
    producer_rid: str
    ids: np.ndarray  # int64 ids of positions [0, P)
    kv: ReqKvInfo  # row; KV [cache_protected_len, P); mamba slot after [0, P)
    last_node: Any  # NodeId of the T node, locked by lock_receipt
    lock_receipt: DecLockRefParams
    claimed_by: Optional[str] = None  # rid of the waiting request that claimed it
    claimant: Optional[Req] = None

    @property
    def pos(self) -> int:
        return len(self.ids)


class SessionTailPark:
    def __init__(self, cache: UnifiedRadixCache):
        self.cache = cache
        self._validate(cache)
        self.capacity = cache.req_to_token_pool.session_park_rows
        if self.capacity <= 0:
            raise ValueError(
                "--session-exact-resume needs request rows reserved for parks "
                f"(req_to_token_pool.session_park_rows={self.capacity})"
            )
        self.page_size = cache.page_size
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        self.qsa_ratio = (
            kvcache.qsa_compress_ratio if isinstance(kvcache, QSATokenToKVPool) else 1
        )
        self.spec = get_spec().speculative_algorithm is not None
        self.counters: Counter = Counter()
        self.reset()

    @staticmethod
    def _validate(cache: UnifiedRadixCache) -> None:
        need = {
            "the radix cache on": not cache.disable,
            "a hybrid-mamba tree (FULL + MAMBA)": cache.is_mamba_enabled,
            "--mamba-radix-cache-strategy extra_buffer": cache.enable_mamba_extra_buffer
            and not get_exec().mamba.enable_mamba_extra_buffer_lazy,
            "no SWA component": not cache.is_swa_enabled,
            "no --enable-session-radix-cache": not cache.enable_session_radix_cache,
            "no --enable-streaming-session": not get_serving().enable_streaming_session,
            # It hands cache_finished_req a length short of the committed KV.
            "no --strip-thinking-cache": not get_serving().strip_thinking_cache,
            "no --enable-hierarchical-cache": not get_memory().enable_hierarchical_cache,
            "no unified-cache external linker": not get_memory().enable_unified_cache_external_linker,
            "no --enable-hisparse": not get_memory().enable_hisparse,
            "no --enable-int8-mamba-checkpoint": not get_exec().mamba.enable_int8_mamba_checkpoint,
            "no PD disaggregation": get_disagg().disaggregation_mode == "null",
            "no diffusion LLM": get_exec().dllm.dllm_algorithm is None,
            "SGLANG_RADIX_FORCE_MISS off": not envs.SGLANG_RADIX_FORCE_MISS.get(),
        }
        missing = [what for what, ok in need.items() if not ok]
        if missing:
            raise ValueError("--session-exact-resume requires: " + "; ".join(missing))

    def reset(self) -> None:
        self.parks: Dict[str, List[_Park]] = {}
        self.lru: "OrderedDict[int, _Park]" = OrderedDict()
        self.claims: Dict[str, _Park] = {}
        self.tombstones: "OrderedDict[str, tuple[int, str]]" = OrderedDict()
        self.seq = 0
        self._tokens = {False: 0, True: 0}  # KV tokens by claimed
        self._slots = {False: 0, True: 0}  # mamba slots by claimed

    # ---- accounting ---------------------------------------------------------

    def _park_tokens(self, park: _Park) -> int:
        return ceil_align(park.pos, self.page_size) - park.kv.cache_protected_len

    def _account(self, park: _Park, sign: int) -> None:
        claimed = park.claimed_by is not None
        self._tokens[claimed] += sign * self._park_tokens(park)
        self._slots[claimed] += sign

    def kv_tokens(self, *, claimed: bool) -> int:
        return self._tokens[claimed]

    def mamba_slots(self, *, claimed: bool) -> int:
        return self._slots[claimed]

    def rows(self) -> int:
        return len(self.lru)

    # ---- park (request end) -------------------------------------------------

    def try_park(self, req: Req, *, is_insert: bool, kv_len_to_handle: int) -> bool:
        """cache_finished_req entry: True when the request was parked (nothing left
        for the caller to do); False runs the normal finished-request path."""
        sid = req.session_id
        if sid is None or req.session is not None:
            return False
        if not req.finished():
            # release_req (retraction): the request starts over from the tree.
            req.session_resume = (_fallback(RETRACT), None)
            self.counters["retract"] += 1
            return False
        exact = self._resolve_exact_end(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        reason = req.exact_end_fallback
        if exact is not None:
            if exact.pos % self.qsa_ratio != 0 and not QSA_MISALIGNED_RESUME:
                reason = QSA_MISALIGNED
            elif not self._make_room(sid):
                reason = EVICTED  # every park row is claimed
        if reason is not None:
            self._tombstone(sid, reason)
            self.counters[f"no_park:{reason}"] += 1
            return False
        self._park(req, exact)
        return True

    def _resolve_exact_end(
        self, req: Req, *, is_insert: bool, kv_len_to_handle: int
    ) -> Optional[ExactEnd]:
        if req.exact_end is not None or req.exact_end_fallback is not None:
            return req.exact_end  # a decode path (spec snapshot) decided
        reason = self._nonspec_fallback(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        if reason is not None:
            req.exact_end_fallback = reason
            return None
        # Non-spec: the live slot has consumed exactly [0, kv_committed_len).
        req.exact_end = ExactEnd(
            pos=req.kv.kv_committed_len, mamba_slot=req.kv.mamba_pool_idx
        )
        return req.exact_end

    def _nonspec_fallback(
        self, req: Req, *, is_insert: bool, kv_len_to_handle: int
    ) -> Optional[str]:
        if self.spec:
            return SPEC_UNSUPPORTED
        fr = req.finished_reason
        if isinstance(fr, FINISH_ABORT) or not is_insert:
            return ABORT  # not inserted: a sampling-mask abort
        if isinstance(fr, (FINISH_MATCHED_STR, FINISHED_MATCHED_REGEX)):
            return STOP_STRING
        if not isinstance(fr, (FINISH_MATCHED_TOKEN, FINISH_LENGTH)):
            raise AssertionError(f"session park: unknown finish reason {fr!r}")
        if req.finished_len is not None and req.finished_len != len(req.output_ids):
            return STOP_STRING  # output trimmed before the KV's end
        if req.grammar is not None:
            return GRAMMAR
        if req.arc_think_armed_at >= 0:
            return THINK_BUDGET
        if req.beam_group is not None or req.positional_embed_overrides is not None:
            # KV another request with the same ids must not take
            return UNSUPPORTED
        if len(req.output_ids) <= 1:
            return PREFILL_FINISHED
        committed = req.kv.kv_committed_len
        if kv_len_to_handle != committed:
            raise AssertionError(
                f"session park: req {req.rid} hands {kv_len_to_handle} of {committed} "
                "committed KV (--strip-thinking-cache is excluded at startup)"
            )
        lead = committed - (len(req.origin_input_ids) + len(req.output_ids) - 1)
        if lead not in (0, 1):
            raise AssertionError(
                f"session park: req {req.rid} committed {committed} for "
                f"{len(req.origin_input_ids)}+{len(req.output_ids)} ids (lead {lead})"
            )
        if req.kv.kv_allocated_len != committed:
            raise AssertionError(
                f"session park: req {req.rid} non-spec allocated "
                f"{req.kv.kv_allocated_len} != committed {committed}"
            )
        if not req.kv.holds_mamba:
            raise AssertionError(f"session park: req {req.rid} holds no mamba slot")
        return None

    def _make_room(self, sid: str) -> bool:
        """Release older parks of the session beyond the per-session limit and the
        oldest unclaimed parks beyond capacity; False when every row is claimed."""
        while True:
            mine = self.parks.get(sid, [])
            if len(mine) < MAX_PARKS_PER_SESSION:
                break
            unclaimed = [p for p in mine if p.claimed_by is None]
            if not unclaimed:
                break  # both claimed: they will be consumed; capacity still applies
            self._release(unclaimed[0], "replaced")
        while len(self.lru) >= self.capacity:
            if not self.evict_lru():
                return False
        return True

    def _park(self, req: Req, exact: ExactEnd) -> None:
        cache = self.cache
        pool = cache.req_to_token_pool
        pos = exact.pos
        if req.kv_rotation_base is not None:
            raise AssertionError("session park does not support logical-page KV sharding")
        if req.last_node is None:
            raise AssertionError(f"session park: req {req.rid} holds no tree lock")
        if pos > req.kv.kv_committed_len:
            raise AssertionError(
                f"session park: exact end {pos} past committed {req.kv.kv_committed_len}"
            )
        if req.kv.kv_allocated_len > pos:
            # Spec: KV past the end (the verify that also ran the stop token) goes
            # back; the tail of the last page stays and the next extend overwrites it.
            start = ceil_align(pos, self.page_size)
            if start < req.kv.kv_allocated_len:
                cache.free_kv_row(req.kv, [(start, req.kv.kv_allocated_len)])
            req.kv.kv_allocated_len = pos
        req.kv.kv_committed_len = pos

        token_ids = (req.origin_input_ids + req.output_ids)[:pos]
        # The tree takes [0, T) the way a chunk boundary does.
        cache._cache_unfinished_tokens(req, token_ids, chunked=False)
        protected = req.kv.cache_protected_len
        if protected % self.page_size != 0 or protected > pos:
            raise AssertionError(
                f"session park: tree prefix {protected} (end {pos}, page {self.page_size})"
            )
        if (
            req.kv.mamba_cow_src_index is not None
            or req.kv.mamba_needs_clear
            or req.kv.mamba_host_restore is not None
        ):
            raise AssertionError(
                f"session park: req {req.rid} has a pending mamba COW/clear/restore"
            )

        # Mamba: the exact-end slot stays, the track slots go back.
        buf = req.kv.mamba_ping_pong_track_buffer
        if buf is None:
            raise AssertionError(f"session park: req {req.rid} has no track buffer")
        pool.mamba_allocator.free(buf)
        req.kv.mamba_ping_pong_track_buffer = None
        req.kv.mamba_next_track_idx = None
        req.kv.mamba_last_track_idx = None
        req.kv.mamba_last_track_seqlen = None
        if exact.mamba_slot is not req.kv.mamba_pool_idx:
            # Spec: the snapshot taken at the end; the live slot ran past it.
            if exact.mamba_slot is not req.kv.exact_end_slot:
                raise AssertionError("session park: spec ExactEnd slot is not req.kv.exact_end_slot")
            pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
            req.kv.mamba_pool_idx = exact.mamba_slot
            req.kv.exact_end_slot = None  # the park owns it now

        kv = req.detach_kv()
        self.seq += 1
        park = _Park(
            seq=self.seq,
            session_id=req.session_id,
            cache_salt=req.cache_salt or None,  # as RadixKey normalizes it
            extra_key=req.extra_key,
            producer_rid=req.rid,
            ids=_ids_view(token_ids).copy(),
            kv=kv,
            last_node=req.last_node,
            lock_receipt=req.lock_receipt,
        )
        req.last_node = None
        req.lock_receipt = DecLockRefParams()
        self.parks.setdefault(park.session_id, []).append(park)
        self.lru[park.seq] = park
        self.tombstones.pop(park.session_id, None)
        self._account(park, +1)
        self.counters["park"] += 1

    # ---- claim (scheduling match) -------------------------------------------

    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        """The scheduling match of a waiting request (init_next_round_input). Sets
        req.session_resume; returns a result only for a claimed park."""
        req = params.req
        if req is None or not params.cow_mamba:
            return None  # an internal re-match (cache_unfinished_req) or a probe
        sid = req.session_id
        if sid is None:
            req.session_resume = (NOT_A_SESSION, None)
            return None
        if req.session is not None:
            # A session_params (session controller) request: not parked.
            req.session_resume = (_fallback(UNSUPPORTED), None)
            return None
        if req.kv.holds_kv:
            raise AssertionError(f"session park: scheduling match of {req.rid} holding KV")
        park = self.claims.get(req.rid)
        if park is not None:
            return self._claimed_result(req, park, params)  # a re-match: same claim
        if req.retracted_stain:
            req.session_resume = (_fallback(RETRACT), None)
            return None
        if req.to_finish is not None:
            req.session_resume = (_fallback(ABORT), None)
            return None

        key = params.key
        candidates = self.parks.get(sid, [])
        reason = None
        if candidates:
            if key.is_bigram:
                raise AssertionError("session park: bigram (EAGLE) keys never park")
            tokens = _ids_view(key.token_ids)
            for park in reversed(candidates):  # newest first
                why = self._why_not(park, key, tokens)
                if why is None:
                    self._account(park, -1)
                    park.claimed_by, park.claimant = req.rid, req
                    self._account(park, +1)
                    self.claims[req.rid] = park
                    self.counters["claim"] += 1
                    return self._claimed_result(req, park, params)
                if reason is None or _REASON_RANK[why] > _REASON_RANK[reason]:
                    reason = why
        if reason is None:
            tomb = self.tombstones.get(sid)
            newest = candidates[-1].seq if candidates else -1
            reason = tomb[1] if tomb is not None and tomb[0] > newest else NO_PARK
        req.session_resume = (_fallback(reason), None)
        self.counters[f"fallback:{reason}"] += 1
        return None

    @staticmethod
    def _why_not(park: _Park, key, tokens: np.ndarray) -> Optional[str]:
        if park.cache_salt != key.cache_salt or park.extra_key != key.extra_key:
            return IDS_MISMATCH
        pos = park.pos
        if len(tokens) < pos or not np.array_equal(tokens[:pos], park.ids):
            return IDS_MISMATCH
        if pos > len(key):
            return KEY_LIMIT
        if park.claimed_by is not None:
            return CLAIMED
        return None

    def _claimed_result(self, req: Req, park: _Park, params: MatchPrefixParams) -> MatchResult:
        pos = park.pos
        if park.claimed_by != req.rid or len(params.key) < pos:
            raise AssertionError(f"session park: claim of {req.rid} ({park.claimed_by})")
        if req.kv.holds_mamba:
            raise AssertionError(f"session park: claimant {req.rid} holds a mamba slot")
        row = self.cache.req_to_token_pool.req_to_token[park.kv.req_pool_idx, :pos]
        req.session_resume = (EXACT, pos)
        return MatchResult(
            device_indices=row.to(dtype=torch.int64, copy=True),
            last_device_node=park.last_node,
            last_host_node=park.last_node,
            best_match_node=park.last_node,
            cache_protected_len=park.kv.cache_protected_len,
            full_kv_hit_length=pos,
        )

    def release_claim(self, rid: str) -> None:
        """A claimant left before admission (aborted while waiting)."""
        park = self.claims.pop(rid, None)
        if park is None:
            return
        park.claimant.session_resume = (_fallback(ABORT), None)
        self._account(park, -1)
        park.claimed_by, park.claimant = None, None
        self._account(park, +1)
        self.counters["claim_released"] += 1

    # ---- consume (admission) ------------------------------------------------

    def on_prefill_admitted(self, req: Req) -> None:
        park = self.claims.pop(req.rid, None)
        if park is None:
            return
        if park.claimed_by != req.rid or park.claimant is not req:
            raise AssertionError(f"session park: {req.rid} consumes {park.claimed_by}'s claim")
        if len(req.prefix_indices) != park.pos or req.last_node != park.last_node:
            raise AssertionError(
                f"session park: {req.rid} admitted with prefix {len(req.prefix_indices)} "
                f"at node {req.last_node}, park ends at {park.pos} on {park.last_node}"
            )
        if req.kv.holds_kv or req.kv.holds_mamba:
            raise AssertionError(f"session park: claimant {req.rid} already holds KV/mamba")
        if req.kv.cache_protected_len != park.kv.cache_protected_len:
            raise AssertionError("session park: claimant protected length differs")
        self._unlink(park)
        # The claimant took its own lock on the same node at commit.
        self.cache.dec_lock_ref(park.last_node, park.lock_receipt)
        req.kv = park.kv
        # The row (hence its QSA pending ring) is the producer's: it holds
        # [floor4(P), P) for this extend.
        req.qsa_ring_resume = True
        self.counters["consume"] += 1

    # ---- release ------------------------------------------------------------

    def evict_lru(self) -> bool:
        """Release the oldest unclaimed park; False when there is none."""
        for park in self.lru.values():
            if park.claimed_by is None:
                self._release(park, EVICTED)
                return True
        return False

    def _unlink(self, park: _Park) -> None:
        self._account(park, -1)
        del self.lru[park.seq]
        mine = self.parks[park.session_id]
        mine.remove(park)
        if not mine:
            del self.parks[park.session_id]

    def _release(self, park: _Park, reason: str) -> None:
        if park.claimed_by is not None:
            raise AssertionError("session park: releasing a claimed park")
        cache = self.cache
        pool = cache.req_to_token_pool
        self._unlink(park)
        cache.dec_lock_ref(park.last_node, park.lock_receipt)
        kv = park.kv
        if kv.kv_allocated_len > kv.cache_protected_len:
            cache.free_kv_row(kv, [(kv.cache_protected_len, kv.kv_allocated_len)])
        pool.free(_RowHolder(kv))
        pool.mamba_allocator.free(kv.mamba_pool_idx.unsqueeze(0))
        kv.mamba_pool_idx = None
        kv.mark_kv_released()
        if reason == EVICTED and park.session_id not in self.parks:
            self._tombstone(park.session_id, EVICTED)
        self.counters[f"release:{reason}"] += 1

    def _tombstone(self, sid: str, reason: str) -> None:
        _fallback(reason)  # a contract name
        self.seq += 1
        self.tombstones[sid] = (self.seq, reason)
        self.tombstones.move_to_end(sid)
        while len(self.tombstones) > TOMBSTONE_LIMIT:
            self.tombstones.popitem(last=False)

    # ---- checks -------------------------------------------------------------

    def sanity_check(self, *, idle: bool) -> None:
        """Bookkeeping invariants. ``idle`` (nothing waiting or running): no claim
        can be outstanding, and the parks' KV pages and mamba slots are not free."""
        tokens = {False: 0, True: 0}
        slots = {False: 0, True: 0}
        pool = self.cache.req_to_token_pool
        free_rows = set(pool.free_slots)
        for seq, park in self.lru.items():
            assert park.seq == seq and park in self.parks[park.session_id]
            assert park.kv.req_pool_idx not in free_rows
            assert park.kv.holds_mamba and park.kv.mamba_ping_pong_track_buffer is None
            assert park.kv.kv_committed_len == park.kv.kv_allocated_len == park.pos
            claimed = park.claimed_by is not None
            assert not claimed or self.claims[park.claimed_by] is park
            tokens[claimed] += self._park_tokens(park)
            slots[claimed] += 1
        assert sum(len(v) for v in self.parks.values()) == len(self.lru)
        assert all(p.seq in self.lru for p in self.claims.values())
        assert tokens == self._tokens and slots == self._slots, (tokens, self._tokens)
        if not idle or not self.lru:
            return
        if self.claims:
            raise AssertionError(
                f"session park: claims outstanding while idle: {sorted(self.claims)}"
            )
        page = self.page_size
        held_pages, held_slots = [], []
        for park in self.lru.values():
            kv = park.kv
            row = pool.req_to_token[kv.req_pool_idx]
            held_pages.append(row[kv.cache_protected_len : kv.kv_allocated_len] // page)
            held_slots.append(kv.mamba_pool_idx.reshape(1))
        free_pages = self.cache.token_to_kv_pool_allocator.get_all_free_pages()
        held = torch.cat(held_pages).to(free_pages.device, torch.int64)
        stale = held[torch.isin(held, free_pages.to(torch.int64))]
        if stale.numel():
            raise AssertionError(f"session park: parked KV pages are free: {stale[:8].tolist()}")
        mamba_free = pool.mamba_allocator.free_slots
        held_slots = torch.cat(held_slots).to(mamba_free.device, torch.int64)
        stale = held_slots[torch.isin(held_slots, mamba_free.to(torch.int64))]
        if stale.numel():
            raise AssertionError(f"session park: parked mamba slots are free: {stale.tolist()}")

    def stats(self) -> dict:
        return {
            "parks": len(self.lru),
            "claimed": len(self.claims),
            "kv_tokens": dict(self._tokens),
            "counters": dict(self.counters),
        }
