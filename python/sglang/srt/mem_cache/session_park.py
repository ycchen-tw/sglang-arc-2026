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
           (reason "claimed"). Nothing moves; a re-match returns the same result. A
           claimant the scheduler parks for its lazy media (ARC_MM_LAZY_MEDIA) keeps
           the claim until its fill brings it back to the queue.
  consume  PrefillAdder commits the claimant (its own lock on the T node taken): the
           park's lock is released and the request takes the record (row, KV, mamba
           slot) and extends from P in place.
  release  alloc shortfall the tree cannot cover (evict_for_alloc, last resort),
           capacity, the per-session limit, or /close_session of the session (its
           unclaimed parks; a claimed one stays with its claimant). A claimant
           aborted while waiting hands the claim back (the park stays). reset
           (flush_cache) drops the parks; the caller clears the pools wholesale.

Fallback reason of a request that claims nothing: "claimed" when a park it would
have matched is taken (waiting or running claimant); otherwise the newest request
of the session this one continues or repeats -- its ids extend that request's
input -- decides: its park's mismatch (a repeat, a rewind inside its output,
key_limit) or why it left no park (its tombstone: abort, stop_string, evicted,
...). A tombstone lasts until a later request of its lineage ends (that one is the
newest then). A request that extends none of the session's recent requests is a
rewind or branch ("ids_mismatch"); a session with no record at all is "no_park".

Accounting: an unclaimed park's KV pages and mamba slot count as evictable, a
claimed park's as session-held; every park holds one req row, taken from
``session_park_rows`` reserved on top of max_running_requests.

Every decision depends only on the request stream (logical sequence numbers, no
wall clock), so all TP ranks take the same ones.
"""

from __future__ import annotations

import hashlib
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
TOMBSTONE_LIMIT = 1 << 16  # sessions with tombstones
TOMBSTONES_PER_SESSION = 4
STATS_EVERY = 256  # parks between two stats log lines

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
QSA_MISALIGNED = "qsa_misaligned"  # (the transition before exp/ser-qsa-misaligned; unused)
# Under spec decoding speculative/exact_end.py names the finish's fallback (the same
# constants: retract, abort, prefill_finished, stop_string, grammar, think_budget,
# snapshot_mismatch).
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
    }
)

EXACT = ("exact", None)
NOT_A_SESSION = ("none", None)


def _fallback(reason: str) -> tuple:
    if reason not in RESUME_REASONS:
        raise AssertionError(f"session resume reason {reason!r} is not in the contract")
    return ("fallback", reason)


def session_resume_meta(req: Req) -> dict:
    """meta_info["session_resume"] of every request, with --session-exact-resume
    only (off, the scheduler sends none and meta_info has no such key):
    {"mode": "exact" | "fallback" | "none", "reason", "pos", "prefill_version"}.
    mode / reason come from the request's last scheduling match (req.session_resume);
    pos is the prefix of its last admission and prefill_version the weight version
    of the first prefill forward after it, both None before its first admission.
    "none": the request has no session_id."""
    if not get_serving().session_exact_resume:
        raise AssertionError("session_resume_meta with --session-exact-resume off")
    pos, version = req.admitted_prefix_len, req.prefill_weight_version
    if req.session_resume is None:
        # Never reached a scheduling match: aborted while queued.
        if pos is not None:
            raise AssertionError(f"session resume: {req.rid} admitted without a match")
        if req.session_id is None:
            mode, reason = NOT_A_SESSION
        else:
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


def _digest(ids: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(ids, dtype=np.int64), digest_size=16).digest()


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
    input_len: int  # the producer's input: a request whose ids extend [0, input_len) continues or repeats it
    ids: np.ndarray  # int64 ids of positions [0, P)
    row: int  # the producer's req_pool_idx: its QSA pending ring holds [floor4(P), P)
    # that row; KV [cache_protected_len, P); mamba slot after [0, P) (None once consumed)
    kv: Optional[ReqKvInfo]
    last_node: Any  # NodeId of the T node, locked by lock_receipt
    lock_receipt: DecLockRefParams
    claimed_by: Optional[str] = None  # rid of the waiting request that claimed it
    claimant: Optional[Req] = None

    @property
    def pos(self) -> int:
        return len(self.ids)


@dataclass(eq=False)
class _Tombstone:
    """A request of the session that ended without a park, or whose park was released."""

    seq: int
    reason: str
    cache_salt: Optional[str]
    extra_key: Optional[str]
    input_len: int  # a request whose ids extend its input [0, input_len) continues or repeats it
    digest: bytes  # of those ids


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
        # Parks consumed by requests still running, per session: a request of the
        # session that would have matched one sees "claimed", not "no_park".
        self.consumed: Dict[str, Dict[str, _Park]] = {}
        # Per session, the newest few ends that left no park (oldest first).
        self.tombstones: "OrderedDict[str, List[_Tombstone]]" = OrderedDict()
        self.seq = 0
        self._tokens = {False: 0, True: 0}  # KV tokens by claimed
        self._slots = {False: 0, True: 0}  # mamba slots by claimed

    # ---- accounting ---------------------------------------------------------

    def _park_tokens(self, park: _Park) -> int:
        kv = park.kv
        return ceil_align(park.pos, self.page_size) - kv.cache_protected_len + kv.private_kv_tokens

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
        mine = self.consumed.get(sid)
        if mine is not None and mine.pop(req.rid, None) is not None and not mine:
            del self.consumed[sid]
        if not req.finished():
            # release_req (retraction): the request starts over from the tree.
            req.session_resume = (_fallback(RETRACT), None)
            self.counters["retract"] += 1
            return False
        salt, input_ids = req.cache_salt or None, _ids_view(req.origin_input_ids)
        self._supersede(sid, salt, req.extra_key, input_ids)
        exact = self._resolve_exact_end(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        reason = req.exact_end_fallback
        token_ids = None
        if exact is not None:
            token_ids = (req.origin_input_ids + req.output_ids)[: exact.pos]
            self._replace_prefixes(sid, salt, req.extra_key, _ids_view(token_ids))
            if not self._make_room(sid):
                reason = EVICTED  # every park row is claimed
        if reason is not None:
            self._tombstone(sid, reason, salt, req.extra_key, input_ids)
            self.counters[f"no_park:{reason}"] += 1
            return False
        self._park(req, exact, token_ids)
        return True

    def _replace_prefixes(
        self, sid: str, salt: Optional[str], extra_key: Optional[str], ids: np.ndarray
    ) -> None:
        """A new end of the session is about to park: release its older unclaimed
        parks in the same namespace whose ids are a prefix of the new ones. Any
        request that extends one of them past its end either extends the new park
        too or rewinds into the new request's output; resuming at the old one's
        shorter end would be reported exact while the session's sequence ends
        further (the newest end, the ledger's S). It also frees their rows and
        mamba slots. A claimed one stays with its claimant."""
        for park in list(self.parks.get(sid, [])):
            if (
                park.claimed_by is None
                and park.cache_salt == salt
                and park.extra_key == extra_key
                and park.pos <= len(ids)
                and np.array_equal(ids[: park.pos], park.ids)
            ):
                self._release(park, "replaced")

    def _resolve_exact_end(
        self, req: Req, *, is_insert: bool, kv_len_to_handle: int
    ) -> Optional[ExactEnd]:
        if self.spec:
            return self._spec_exact_end(req, is_insert=is_insert)
        if req.exact_end is not None or req.exact_end_fallback is not None:
            raise AssertionError(f"session park: non-spec {req.rid} has an exact end already")
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

    def _spec_exact_end(self, req: Req, *, is_insert: bool) -> Optional[ExactEnd]:
        """Spec decoding: the end snapshot of the finishing verify's commit, reconciled
        by speculative/exact_end.py (on_spec_finish / on_prefill_finish), which names
        the fallback too. Here only what it does not see: an abort outside a decode
        result (the scheduler's abort, a sampling-mask abort) and KV another request
        with the same ids must not take. exact_end.release frees an untaken slot."""
        fr = req.finished_reason
        why = None
        if isinstance(fr, FINISH_ABORT) or not is_insert:
            why = ABORT
        elif req.beam_group is not None or req.positional_embed_overrides is not None:
            why = UNSUPPORTED
        if why is not None:
            req.exact_end, req.exact_end_fallback = None, why
            return None
        if req.exact_end is None:
            if req.exact_end_fallback is None:
                raise AssertionError(
                    f"session park: spec finish of {req.rid} was not reconciled "
                    "(exact_end.on_spec_finish / on_prefill_finish)"
                )
            return None
        if req.exact_end_fallback is not None:
            raise AssertionError(f"session park: {req.rid} has an exact end and a fallback")
        if req.exact_end.mamba_slot is not req.kv.exact_end_slot:
            raise AssertionError(f"session park: {req.rid} ExactEnd slot is not its end snapshot slot")
        if not req.kv.holds_mamba:
            raise AssertionError(f"session park: req {req.rid} holds no mamba slot")
        return req.exact_end

    def _nonspec_fallback(
        self, req: Req, *, is_insert: bool, kv_len_to_handle: int
    ) -> Optional[str]:
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

    def _park(self, req: Req, exact: ExactEnd, token_ids) -> None:
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
        track = req.kv.mamba_last_track_seqlen
        if track is not None and track > pos:
            # Spec: the finishing verify (or the overshoot) tracked past the end, over
            # ids accepted after the stop -- a checkpoint of no prefix of S. The tree
            # takes nothing here; the park keeps [cache_protected_len, P).
            req.kv.mamba_last_track_seqlen = None
            self.counters["track_past_end"] += 1

        if len(token_ids) != pos:
            raise AssertionError(f"session park: {len(token_ids)} ids for the exact end {pos}")
        # The tree takes [0, T) the way a chunk boundary does; the row keeps the
        # producer's own pages where the tree has a duplicate's copy (private_kv).
        if not cache.keeps_own_pages(req):
            raise AssertionError(f"session park: {req.rid} does not keep its own pages")
        cache._cache_unfinished_tokens(req, token_ids, chunked=False, keep_own=True)
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
        if req.kv.exact_end_slot is not None:
            raise AssertionError(f"session park: {req.rid} parks with an end snapshot slot left")
        # The record goes to the next turn: its own end snapshot is armed afresh at its
        # alloc (exact_end.arm_on_alloc), and its first verify writes the row's device
        # table again (exact_end_row_written False).
        req.kv.exact_end_epoch = 0
        req.kv.exact_end_row_written = False
        req.kv.exact_end_unarmed = None

        input_len = len(req.origin_input_ids)
        if input_len > pos:
            raise AssertionError(f"session park: exact end {pos} inside the input ({input_len})")
        kv = req.detach_kv()
        self.seq += 1
        park = _Park(
            seq=self.seq,
            session_id=req.session_id,
            cache_salt=req.cache_salt or None,  # as RadixKey normalizes it
            extra_key=req.extra_key,
            producer_rid=req.rid,
            input_len=input_len,
            ids=_ids_view(token_ids).copy(),
            row=kv.req_pool_idx,
            kv=kv,
            last_node=req.last_node,
            lock_receipt=req.lock_receipt,
        )
        req.last_node = None
        req.lock_receipt = DecLockRefParams()
        self.parks.setdefault(park.session_id, []).append(park)
        self.lru[park.seq] = park
        self._account(park, +1)
        self.counters["park"] += 1
        if self.counters["park"] % STATS_EVERY == 0:
            logger.info("session park: %s", self.stats())

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
        consumed = self.consumed.get(sid, {})
        tombs = self.tombstones.get(sid, [])
        reason = NO_PARK  # the session has no record
        if candidates or consumed or tombs:
            if key.is_bigram:
                raise AssertionError("session park: bigram (EAGLE) keys never park")
            tokens = _ids_view(key.token_ids)
            # (seq, reason) of the newest request of the session this one continues
            # or repeats; extending none of them is a rewind / branch.
            newest = (-1, IDS_MISMATCH)
            # the newest end this request extends that a running request of the
            # session already took (consumed)
            taken = -1
            for park in consumed.values():
                why = self._why_not(park, key, tokens)
                if why == CLAIMED:
                    taken = max(taken, park.seq)
                if park.seq > newest[0] and self._continues(park, key, tokens, park.ids):
                    newest = (park.seq, why)
            claimed = taken >= 0
            for park in reversed(candidates):  # newest first
                if park.seq < taken:
                    break  # older than an end it extends that is taken: see below
                why = self._why_not(park, key, tokens)
                if why is None:
                    self._account(park, -1)
                    park.claimed_by, park.claimant = req.rid, req
                    self._account(park, +1)
                    self.claims[req.rid] = park
                    self.counters["claim"] += 1
                    return self._claimed_result(req, park, params)
                if why != IDS_MISMATCH:
                    # The newest end this request extends decides: another request
                    # took it (a concurrent resend: "claimed") or the key stops short
                    # of it ("key_limit"). An older park it also extends ends earlier
                    # than the session's sequence; claiming that one would re-forward
                    # the gap and report it exact. Stop here.
                    claimed = claimed or why == CLAIMED
                    if park.seq > newest[0]:
                        newest = (park.seq, why)
                    break
                if park.seq > newest[0] and self._continues(park, key, tokens, park.ids):
                    newest = (park.seq, why)
            if claimed:
                reason = CLAIMED
            else:
                for tomb in reversed(tombs):  # newest first
                    if tomb.seq < newest[0]:
                        break
                    if self._continues(tomb, key, tokens, None) and (
                        self._input_digest(req, tokens, tomb.input_len) == tomb.digest
                    ):
                        newest = (tomb.seq, tomb.reason)
                        break
                reason = newest[1]
        req.session_resume = (_fallback(reason), None)
        self.counters[f"fallback:{reason}"] += 1
        return None

    @staticmethod
    def _continues(rec, key, tokens: np.ndarray, ids: Optional[np.ndarray]) -> bool:
        """Same cache namespace and the ids reach past rec's input (and, with ``ids``
        given, equal them there; a tombstone compares a digest instead)."""
        if rec.cache_salt != key.cache_salt or rec.extra_key != key.extra_key:
            return False
        n = rec.input_len
        return len(tokens) >= n and (ids is None or np.array_equal(tokens[:n], ids[:n]))

    @staticmethod
    def _input_digest(req: Req, tokens: np.ndarray, n: int) -> bytes:
        """Digest of the request's ids [0, n), kept on the request: a waiting
        request is matched again every scheduling round."""
        memo = req.session_lineage_digests
        if memo is None:
            memo = req.session_lineage_digests = (len(tokens), {})
        elif memo[0] != len(tokens):
            raise AssertionError(
                f"session park: {req.rid} matched with {len(tokens)} ids, earlier {memo[0]}"
            )
        digest = memo[1].get(n)
        if digest is None:
            digest = memo[1][n] = _digest(tokens[:n])
        return digest

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
        # The QSA extend from P (any P % 4) reads [floor4(P), P) from this row's
        # pending ring: it must be the producer's row, untouched since it parked.
        if park.kv.req_pool_idx != park.row or park.row in self.cache.req_to_token_pool.free_slots:
            raise AssertionError(
                f"session park: {req.rid} resumes on row {park.kv.req_pool_idx}, "
                f"the producer {park.producer_rid} parked row {park.row}"
            )
        self._unlink(park)
        # The claimant took its own lock on the same node at commit.
        self.cache.dec_lock_ref(park.last_node, park.lock_receipt)
        req.kv = park.kv
        park.kv = None  # the record is the request's now; ids stay for "claimed"
        park.claimant = None
        self.consumed.setdefault(park.session_id, {})[req.rid] = park
        req.qsa_ring_resume = True  # consumed by its next prepare_for_extend
        mamba_pool = self.cache.req_to_token_pool.mamba_pool
        if mamba_pool.replayssm_spec_write_pos is not None:
            # Spec ReplaySSM: the row's ring cursors are the producer's, past its end
            # (an overshoot verify still in flight may advance them); the slot holds
            # the whole state after [0, P). Zero them like a fresh row's, again on the
            # forward stream before this request's extend (HybridReqToTokenPool.alloc,
            # ModelRunner: replay_cursor_reset_rows).
            req.kv.replay_cursor_reset = True
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
        self._free(park)
        # Evicted or replaced by a newer end of the session: a request that
        # continues the producer sees "evicted".
        self._tombstone(
            park.session_id, EVICTED, park.cache_salt, park.extra_key, park.ids[: park.input_len]
        )
        self.counters[f"release:{reason}"] += 1

    def _free(self, park: _Park) -> None:
        """Unlink an unclaimed park and give back its lock, KV, row and mamba slot."""
        if park.claimed_by is not None:
            raise AssertionError("session park: releasing a claimed park")
        cache = self.cache
        pool = cache.req_to_token_pool
        self._unlink(park)
        cache.dec_lock_ref(park.last_node, park.lock_receipt)
        kv = park.kv
        if kv.kv_allocated_len > kv.cache_protected_len:
            cache.free_kv_row(kv, [(kv.cache_protected_len, kv.kv_allocated_len)])
        cache.free_private_kv(kv)
        pool.free(_RowHolder(kv))
        pool.mamba_allocator.free(kv.mamba_pool_idx.unsqueeze(0))
        kv.mamba_pool_idx = None
        kv.mark_kv_released()

    # ---- session close --------------------------------------------------------

    def on_session_close(self, sid: str) -> None:
        """/close_session: the client is done with the session. Its unclaimed parks
        are released now instead of waiting for eviction, and its tombstones go, so
        a later request under the same id starts like a new session ("no_park").
        A claimed park stays with its claimant: admission consumes it as usual (an
        abort before admission hands it back, an unclaimed park of a closed
        session then, left to eviction). Requests still running are not touched:
        one that ends cleanly parks again, likewise left to eviction."""
        mine = self.parks.get(sid, [])
        for park in [p for p in mine if p.claimed_by is None]:
            self._free(park)
            self.counters["release:closed"] += 1
        kept = len(self.parks.get(sid, []))
        if kept:
            self.counters["close:claimed_kept"] += kept
        self.tombstones.pop(sid, None)
        self.counters["close"] += 1

    def _supersede(
        self, sid: str, cache_salt: Optional[str], extra_key: Optional[str], input_ids: np.ndarray
    ) -> None:
        """A request of the session ended: it is the newest of the lineage of every
        tombstoned request it continues or repeats."""
        mine = self.tombstones.get(sid)
        if not mine:
            return
        keep = [
            tomb
            for tomb in mine
            if not (
                tomb.cache_salt == cache_salt
                and tomb.extra_key == extra_key
                and len(input_ids) >= tomb.input_len
                and _digest(input_ids[: tomb.input_len]) == tomb.digest
            )
        ]
        if keep:
            mine[:] = keep
        else:
            del self.tombstones[sid]

    def _tombstone(
        self,
        sid: str,
        reason: str,
        cache_salt: Optional[str],
        extra_key: Optional[str],
        input_ids: np.ndarray,
    ) -> None:
        _fallback(reason)  # a contract name
        self.seq += 1
        mine = self.tombstones.setdefault(sid, [])
        mine.append(
            _Tombstone(
                seq=self.seq,
                reason=reason,
                cache_salt=cache_salt,
                extra_key=extra_key,
                input_len=len(input_ids),
                digest=_digest(input_ids),
            )
        )
        del mine[:-TOMBSTONES_PER_SESSION]
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
            assert park.kv.private_kv_tokens % self.page_size == 0
            claimed = park.claimed_by is not None
            assert not claimed or self.claims[park.claimed_by] is park
            tokens[claimed] += self._park_tokens(park)
            slots[claimed] += 1
        assert sum(len(v) for v in self.parks.values()) == len(self.lru)
        assert all(p.seq in self.lru for p in self.claims.values())
        assert tokens == self._tokens and slots == self._slots, (tokens, self._tokens)
        if not idle:
            return
        if self.claims or self.consumed:
            raise AssertionError(
                f"session park: claims outstanding while idle: {sorted(self.claims)}, "
                f"consumed by running requests: {sorted(self.consumed)}"
            )
        if not self.lru:
            return
        page = self.page_size
        held_pages, held_slots = [], []
        for park in self.lru.values():
            kv = park.kv
            row = pool.req_to_token[kv.req_pool_idx]
            held_pages.append(row[kv.cache_protected_len : kv.kv_allocated_len] // page)
            if kv.private_kv is not None:
                held_pages.append(kv.private_kv // page)
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

    def check_tree_locks(self) -> None:
        """Idle (nothing waiting or running, no streaming session): the tree's device
        locks are exactly the parks'. Each park holds its T node's Full lock (taken
        on every node from T to the root) and, unless its receipt skipped it, the
        Mamba lock on T itself; nothing else is locked."""
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        core = self.cache.tree_core
        full, mamba = ComponentType.FULL, ComponentType.MAMBA
        if set(core.component_types) != {full, mamba}:
            raise AssertionError(f"session park: tree components {core.component_types}")
        want = {full: Counter(), mamba: Counter()}
        for park in self.lru.values():
            node = core.node_by_id(park.last_node)
            if mamba not in park.lock_receipt.skipped_lock_components:
                want[mamba][node.id] += 1
            while node is not core.root_node:
                want[full][node.id] += 1
                node = node.parent
        bad = []
        for node in core._collect_all_nodes():
            if node is core.root_node:
                continue
            for ct, counts in want.items():
                cd = node.component_data[ct]
                if cd.lock_ref != counts[node.id] or cd.host_lock_ref != 0:
                    bad.append((node.id, ct.name, cd.lock_ref, cd.host_lock_ref, counts[node.id]))
        if bad:
            raise AssertionError(
                "session park: idle tree locks (node, component, lock_ref, host_lock_ref, "
                f"parks' locks) differ: {bad[:8]}"
            )

    def stats(self) -> dict:
        return {
            "parks": len(self.lru),
            "capacity": self.capacity,
            "claimed": len(self.claims),
            "running_consumers": sum(len(v) for v in self.consumed.values()),
            "kv_tokens": {"unclaimed": self._tokens[False], "claimed": self._tokens[True]},
            # of those, the parks' own pages where the tree has a duplicate's copy
            "private_tokens": sum(p.kv.private_kv_tokens for p in self.lru.values()),
            "counters": dict(sorted(self.counters.items())),
        }
