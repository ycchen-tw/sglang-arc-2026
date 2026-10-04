"""Session cache for QSA HiSparse (arc-forge M1b h09; design §1-§2, §4.1, §5, §11).

Registered as ``--radix-cache-backend qsa_hisparse_session``. Reuse happens only
inside one radix-native session (the ``session_id`` in the request body,
``--enable-session-radix-cache``): each session keeps at most one record, the
prefix [0, P) of the last request that owned it, where P is the position of the
session's latest published GDN snapshot (h08). A record holds

    logical slots [0, P)       the QSA compressed-K index lives on them (h02 §4.2)
    host blocks [0, P/4)       the planar host copies of the KV (h07)
    one GDN host buffer        the GDN / PLE state after exactly P tokens (h08)

and never a request slot, slab or mamba slot. States: PARKED (no pin: its own
copies are authoritative, evictable in LRU order), PINNED (the one request that
resumed or created it runs: the request's req_to_token / req_to_host_pool rows
below P are the record's), released. The next request of the session resumes
only if its tokens agree with the record's on [0, P), token by token.

Ownership is by session (§11.1): owner[sid] is the one live request allowed to
resume, snapshot and park; a second concurrent request of the same session and
requests of a closed generation neither resume nor park (§11.2, R7, R8). The
rule model is arc-forge work/qsa_hisparse/session_rules.py (SessionSim); the
methods below mirror its steps and names.

Scheduler contract (every exit path goes through cache_finished_req, §11.5):
match_prefix (waiting) -> inc_lock_ref (PrefillAdder commit, pins) ->
adopt_resumed_prefixes (after alloc_for_extend) -> on_snapshot_published
(h08, commits [P_old, p) to the record) -> cache_finished_req (finish, abort,
retract: publish pending snapshots, copy rows back, free the rest, park or
release). A request aborted while waiting leaves through release_aborted_request.
"""

from __future__ import annotations

import logging
import time
from array import array
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.registry import register_radix_cache_backend

if TYPE_CHECKING:
    from sglang.srt.managers.qsa_hisparse_coordinator import QSAHiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext

logger = logging.getLogger(__name__)

BACKEND_NAME = "qsa_hisparse_session"
DEFAULT_IDLE_TIMEOUT_S = 1800.0


def _i64(tokens) -> np.ndarray:
    """A read-only int64 view of a token sequence (array('q') without a copy)."""
    if isinstance(tokens, array):
        if tokens.typecode != "q":
            raise TypeError(f"token array typecode {tokens.typecode!r}, need 'q'")
        return np.frombuffer(tokens, dtype=np.int64) if len(tokens) else np.empty(0, np.int64)
    return np.asarray(tokens, dtype=np.int64)


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


@dataclass(eq=False)
class SessionRecord:
    """design §2.1. logical / host_blocks / tokens are authoritative while PARKED only."""

    sid: str
    gen: int
    extra_key: Optional[str]
    cache_salt: Optional[str]
    P: int
    committed: int  # the producer's kv_committed_len when P was published
    gdn: int  # h08 GDN host buffer: the state after exactly P tokens
    logical: Optional[torch.Tensor] = None  # int64 [P] logical slots
    host_blocks: Optional[torch.Tensor] = None  # int64 [P / bt] host blocks (h07 contract)
    tokens: Optional[np.ndarray] = None  # the producer's input + output ids
    pin_req: Optional["Req"] = None
    pin_count: int = 0
    last_used: float = 0.0
    close_on_unpin: bool = False
    released: bool = False

    @property
    def state(self) -> str:
        if self.released:
            return "released"
        return "PINNED" if self.pin_count else "PARKED"


@dataclass(eq=False)
class _ReqSession:
    """What the cache knows of one live request of a session (SessionSim.SReq)."""

    req: "Req"
    sid: str
    gen: int
    second: bool = False  # another live request owns the session (R7)
    claim: Optional[SessionRecord] = None  # match result, until adopted
    record: Optional[SessionRecord] = None  # the record it pins (resumed or created)
    first_buf: Optional[int] = None  # GDN buffer of a snapshot taken before the record exists (§11.4)
    logged: Optional[tuple] = None  # last match decision logged (admission retries match again)


class QSAHiSparseSessionCache(ChunkCache):
    """ChunkCache semantics for a request's own KV (cache_protected_len is not the
    request's to free), plus per-session records of host-resident prefixes."""

    enable_session_radix_cache = True

    def __init__(self, params: "CacheInitParams", coordinator: "QSAHiSparseCoordinator"):
        from sglang.srt.environ import envs
        from sglang.srt.managers.qsa_hisparse_coordinator import QSAHiSparseCoordinator
        from sglang.srt.mem_cache.allocator.page_interleave import (
            page_interleave_shard_size,
        )
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        super().__init__(params)
        if not isinstance(coordinator, QSAHiSparseCoordinator):
            raise TypeError(
                f"{BACKEND_NAME} needs the QSA HiSparse coordinator (--enable-hisparse on a "
                f"QSA model), got {type(coordinator).__name__}"
            )
        if not coordinator.session_resume or coordinator.snapshots is None:
            raise ValueError(
                f"{BACKEND_NAME} needs hisparse_config qsa_session_resume=true (h07) and "
                "qsa_snapshot_interval (h08)"
            )
        if params.token_to_kv_pool_allocator is not coordinator.token_to_kv_pool_allocator:
            raise RuntimeError(f"{BACKEND_NAME}: the cache and the coordinator use different allocators")
        if not params.enable_session_radix_cache:
            raise ValueError(f"{BACKEND_NAME} needs --enable-session-radix-cache")
        layout = coordinator.layout
        if self.page_size != layout.page_size:
            raise ValueError(f"page_size {self.page_size} != QSA HiSparse layout page {layout.page_size}")
        if page_interleave_shard_size(self.token_to_kv_pool_allocator) != 1:
            raise NotImplementedError(f"{BACKEND_NAME} does not support KV page sharding")
        if envs.SGLANG_RADIX_FORCE_MISS.get():
            raise ValueError(f"{BACKEND_NAME}: SGLANG_RADIX_FORCE_MISS would keep a resume point without its prefix")
        timeout = parse_hisparse_config().sparse_extra_config.get(
            "qsa_session_idle_timeout_s", DEFAULT_IDLE_TIMEOUT_S
        )
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError(f"hisparse_config.qsa_session_idle_timeout_s must be >= 0, got {timeout!r}")

        self.coord = coordinator
        self.block_tokens = layout.block_tokens
        self.idle_timeout = float(timeout)  # 0: off
        # R15: before 0030 the prefill context is packed whole per QSA layer, so a
        # prefill batch takes resumed prompts up to 2 C tokens (the first always).
        self.resume_context_cap = 2 * coordinator.req_to_token_pool.max_context_len
        self.clock = time.monotonic
        self.reset_state()
        coordinator.set_snapshot_owner(self)
        logger.info(
            "QSA HiSparse session cache: snapshot T=%d, idle timeout %.0f s, "
            "resumed context cap %d tokens per prefill batch",
            coordinator.snapshots.interval,
            self.idle_timeout,
            self.resume_context_cap,
        )

    def reset_state(self) -> None:
        self.records: List[SessionRecord] = []  # live records, indexed or detached (closing)
        self.index: Dict[str, SessionRecord] = {}  # sid -> record of its current generation
        self.owner: Dict[str, "Req"] = {}  # sid -> the one live request that may resume / park
        self.gen: Dict[str, int] = {}  # sid -> generation; absent = closed / never opened
        self._incarnation = 0
        self._reqs: Dict[str, _ReqSession] = {}  # rid -> live session request
        self.stats: Counter = Counter()
        self.ledger_version = 0  # bumped whenever a record parks or goes (idle re-check)

    # ------------------------------------------------------------ cache traits

    @property
    def disable(self):
        # Prefix matching is on: the PrefillAdder path that pins (_lock_node,
        # _req_inc_lock_ref) runs for every request, ignore_eos included.
        return False

    def is_chunk_cache(self) -> bool:
        # evict_from_tree_cache skips chunk caches; PARKED records are evictable.
        return False

    def supports_mamba(self) -> bool:
        return False  # release_kv_cache frees the request's live mamba slot

    def supports_streaming_session(self) -> bool:
        return False

    # ------------------------------------------------------------ sessions

    def ensure_session_generation(self, session_id: str) -> int:
        if session_id not in self.gen:
            self._incarnation += 1
            self.gen[session_id] = self._incarnation
        return self.gen[session_id]

    def open_radix_session(self, session_id: str):
        # the scheduler's open_session refuses first; reaching this is a bug
        raise NotImplementedError(
            f"{BACKEND_NAME} serves radix-native sessions (session_id in the request body); "
            "/open_session streaming sessions are not supported"
        )

    def release_radix_session(self, session_id: str) -> int:
        """/close_session (design §2.5, §11.3): a PARKED record goes now, a PINNED one when unpinned."""
        if session_id not in self.gen:
            self.stats["close_unknown"] += 1
            return 0
        del self.gen[session_id]
        self.owner.pop(session_id, None)  # the in-flight request is stale from now on
        rec = self.index.pop(session_id, None)
        if rec is None:
            self.stats["close_empty"] += 1
        elif rec.state == "PARKED":
            self.stats["close_parked"] += 1
            self._release_record(rec)
        else:
            self.stats["close_pinned"] += 1
            rec.close_on_unpin = True  # detached: the id may start a new generation at once
        return 0

    def _stale(self, st: _ReqSession) -> bool:
        return self.gen.get(st.sid) != st.gen

    def _state(self, req: "Req") -> _ReqSession:
        st = self._reqs.get(req.rid)
        if st is None:
            if req.session_generation is None:
                raise RuntimeError(f"{req.rid}: session {req.session_id} has no generation")
            st = _ReqSession(req, req.session_id, req.session_generation)
            self._reqs[req.rid] = st
        elif st.req is not req or st.sid != req.session_id or st.gen != req.session_generation:
            raise RuntimeError(f"{req.rid}: two live requests share one rid, or its session changed")
        return st

    # ------------------------------------------------------------ match / pin / adopt

    def _miss(self) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def _session_role(self, req: "Req", st: _ReqSession) -> str:
        if self._stale(st):
            return "stale"
        owner = self.owner.get(st.sid)
        return "second" if owner is not None and owner is not req else "owner"

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """§4.1 with §11.2: role first (stale / second / owner), then the owner's
        record is checked token by token. Re-entrant: admission matches again."""
        req = params.req
        if req is None:
            raise RuntimeError(f"{BACKEND_NAME}.match_prefix needs the request")
        if req.kv.holds_kv:
            raise RuntimeError(f"{req.rid}: match_prefix on a request that holds KV")
        req.hisparse_resume = None
        req.kv.mamba_host_restore = None
        if req.session is not None:
            raise NotImplementedError(f"{BACKEND_NAME} has no streaming sessions")
        if req.session_id is None:
            return self._miss()
        st = self._state(req)
        st.claim = None
        role = self._session_role(req, st)
        st.second = role == "second"
        if role != "owner":
            self.stats[f"match_{role}"] += 1
            return self._miss()
        self.owner[st.sid] = req
        rec = self.index.get(st.sid)
        if rec is None:
            return self._miss()
        why = self._mismatch(rec, params, req, st)
        if why is not None:
            self.stats[f"mismatch_{why}"] += 1
            self.stats["mismatch_release"] += 1
            self._release_record(rec)
            return self._miss()
        st.claim = rec
        req.hisparse_resume = rec  # h07 adopt_host_prefix reads .host_blocks
        req.kv.mamba_host_restore = rec.gdn  # h08 deferred H2D before the first extend
        return MatchResult(
            device_indices=rec.logical,
            last_device_node=rec,
            last_host_node=rec,
            best_match_node=rec,
            cache_protected_len=rec.P,
        )

    def _mismatch(
        self, rec: SessionRecord, params: MatchPrefixParams, req: "Req", st: _ReqSession
    ) -> Optional[str]:
        """Why the owner's record cannot be resumed (None: it can); logs (P, LCP, committed)."""
        if rec.state != "PARKED" or rec.gen != req.session_generation:
            raise RuntimeError(
                f"{req.rid}: the owner's record of {rec.sid} is {rec.state}, generation "
                f"{rec.gen} vs {req.session_generation}"
            )
        key = params.key
        if key.is_bigram:
            raise NotImplementedError(f"{BACKEND_NAME} matches raw token keys only")
        full = _i64(key.token_ids)
        why = None
        if rec.extra_key != key.extra_key or rec.cache_salt != key.cache_salt:
            why = "extra_key"
        elif rec.P > len(key):  # len honours the limit: input_len - 1 or logprob_start_len
            why = "limit"
        elif not np.array_equal(full[: rec.P], rec.tokens[: rec.P]):
            why = "tokens"
        decision = (id(rec), rec.P, why)
        if st.logged != decision:  # a request waiting for admission matches every round
            st.logged = decision
            logger.info(
                "QSA session %s: %s at P=%d, LCP=%d, committed=%d, key %d/%d tokens",
                rec.sid,
                "resume" if why is None else f"mismatch ({why})",
                rec.P,
                _lcp(full, rec.tokens),
                rec.committed,
                len(key),
                len(full),
            )
        return why

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        """PrefillAdder: the temporary lock around the admission check, then the
        commit's pin (design §2.2 step 2); both on the owner's claimed record."""
        if node is None:
            return IncLockRefResult(delta=0)
        rec = self._live_record(node)
        req = self.owner.get(rec.sid)
        if req is None or self._reqs[req.rid].claim is not rec:
            raise RuntimeError(f"lock on a record of {rec.sid} its owner does not claim")
        if rec.pin_count == 0:
            rec.pin_req = req
        elif rec.pin_req is not req:
            raise RuntimeError(f"record of {rec.sid} pinned by {rec.pin_req.rid}, locked for {req.rid}")
        rec.pin_count += 1
        return IncLockRefResult(delta=0)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """The temporary lock's release. The commit's pin is released by cache_finished_req."""
        if node is None:
            return DecLockRefResult(delta=0)
        rec = self._live_record(node)
        if rec.pin_count < 1:
            raise RuntimeError(f"unlock of the unpinned record of {rec.sid}")
        rec.pin_count -= 1
        if rec.pin_count == 0:
            rec.pin_req = None
        return DecLockRefResult(delta=0)

    def _live_record(self, node: Any) -> SessionRecord:
        if not isinstance(node, SessionRecord) or node.released:
            raise RuntimeError(f"{BACKEND_NAME}: lock on {node!r}, not a live session record")
        return node

    def adopt_resumed_prefixes(self, reqs: List["Req"]) -> None:
        """After alloc_for_extend wrote record.logical into req_to_token (design
        §2.2 step 3): the rows take the record's host blocks (h07), and the
        record counts as held by the request's rows from here on."""
        for req in reqs:
            rec, st = req.hisparse_resume, self._reqs[req.rid]
            if st.claim is not rec or rec.pin_req is not req or rec.pin_count != 1:
                raise RuntimeError(
                    f"{req.rid}: adopting the record of {rec.sid} without holding its one pin "
                    f"(pins {rec.pin_count}, claim {st.claim is rec})"
                )
        self.coord.adopt_host_prefix(reqs)
        for req in reqs:
            st = self._reqs[req.rid]
            st.record, st.claim = req.hisparse_resume, None
            req.hisparse_resume = None
            self.stats["resume"] += 1

    # ------------------------------------------------------------ snapshots (h08 SnapshotOwner)

    def snapshot_buffer(self, req: "Req", p: int) -> Optional[int]:
        """The GDN buffer req's snapshot at p goes to, or None to skip (design §3.2)."""
        st = self._reqs.get(req.rid)
        if st is None:
            return None  # not a session request
        if st.req is not req:
            raise RuntimeError(f"{req.rid}: snapshot for a request the cache does not track")
        if st.second or self._stale(st):
            self.stats["snap_ineligible"] += 1
            return None
        if self.owner.get(st.sid) is not req:
            raise RuntimeError(f"{req.rid}: snapshot by a request that does not own {st.sid}")
        if st.record is not None:
            return st.record.gdn
        if st.first_buf is None:
            st.first_buf = self._alloc_gdn()
            if st.first_buf is None:
                self.stats["skip_gdn"] += 1
        return st.first_buf  # busy until published: h08 skips (§11.8)

    def _alloc_gdn(self) -> Optional[int]:
        snaps = self.coord.snapshots
        buf = snaps.alloc_buffer()
        if buf is None:
            victims = sorted(self._evict_candidates(), key=lambda r: r.last_used)
            if not victims:
                return None
            self.stats["evict_for_gdn"] += 1
            self._release_record(victims[0])
            buf = snaps.alloc_buffer()
            if buf is None:
                raise RuntimeError("released a record but no GDN buffer came back")
        return buf

    def on_snapshot_published(self, req: "Req", p: int, buf: int) -> None:
        """buf holds the state after p tokens, host blocks [0, p/4) are copied and
        p <= committed (h08): commit [P_old, p) to the record (design §2.2 step 4)."""
        st = self._reqs[req.rid]
        if st.req is not req or p % self.page_size:
            raise RuntimeError(f"{req.rid}: published p={p} is not a page multiple, or not tracked")
        if st.record is None:
            if buf != st.first_buf:
                raise RuntimeError(f"{req.rid}: first snapshot into buffer {buf}, allocated {st.first_buf}")
            st.first_buf = None
            if self._stale(st):  # closed between issue and publish: no record (§11.4)
                self.stats["first_snap_stale"] += 1
                self.coord.snapshots.free_buffer(buf)
                return
            if st.sid in self.index:
                raise RuntimeError(f"{req.rid}: a second record for session {st.sid}")
            rec = SessionRecord(
                st.sid,
                st.gen,
                req.extra_key,
                req.cache_salt,
                P=p,
                committed=req.kv.kv_committed_len,
                gdn=buf,
                pin_req=req,
                pin_count=1,
                last_used=self.clock(),
            )
            self.records.append(rec)
            self.index[st.sid] = rec
            st.record = rec
            self.stats["record_created"] += 1
        else:
            rec = st.record
            if rec.pin_req is not req or rec.gdn != buf or p <= rec.P:
                raise RuntimeError(
                    f"{req.rid}: publish p={p} into buffer {buf}; record P={rec.P}, "
                    f"buffer {rec.gdn}, pinned by {rec.pin_req.rid if rec.pin_req else None}"
                )
            rec.P, rec.committed = p, req.kv.kv_committed_len
        req.kv.cache_protected_len = p  # incremental commit of [P_old, p) to the record
        self.stats["publish"] += 1

    # ------------------------------------------------------------ exits

    def cache_finished_req(self, req: "Req", is_insert: bool = True, *, kv_len_to_handle: int):
        """Every exit of an admitted request: finish, abort, retract (design §2.2 step 5).

        Pending snapshots publish first; the record takes its rows [0, P) back;
        the request frees the rest (logical as ChunkCache, host blocks through
        release_host_row); then the record parks unless the session closed.
        """
        coord = self.coord
        coord.release_snapshots(req)
        st = self._reqs.get(req.rid)
        if st is not None and (st.req is not req or st.claim is not None or st.first_buf is not None):
            raise RuntimeError(
                f"{req.rid}: exit with a claim not adopted or an unpublished first snapshot"
            )
        rec = None if st is None else st.record
        keep = 0 if rec is None else rec.P
        if req.kv.cache_protected_len != keep or keep > kv_len_to_handle:
            raise RuntimeError(
                f"{req.rid}: cache_protected_len {req.kv.cache_protected_len}, record P {keep}, "
                f"committed {kv_len_to_handle}"
            )
        r = req.kv.req_pool_idx
        if rec is not None:  # the rows below P are the record's: copy them back
            rec.logical = self.req_to_token_pool.req_to_token[r, :keep].to(torch.int64, copy=True)
            rec.host_blocks = coord.req_to_host_pool[r, : keep // self.block_tokens].clone()
            rec.tokens = np.concatenate((_i64(req.origin_input_ids), _i64(req.output_ids)))
        super().cache_finished_req(req, is_insert, kv_len_to_handle=kv_len_to_handle)
        coord.release_host_row(r, keep // self.block_tokens)
        if rec is not None:
            self._unpin(rec, st)
        if st is not None and req.finished():  # a retracted request stays the owner
            self._exit(st)

    def _unpin(self, rec: SessionRecord, st: _ReqSession) -> None:
        if rec.pin_req is not st.req or rec.pin_count != 1:
            raise RuntimeError(f"{st.req.rid}: unpin of a record it does not hold alone")
        rec.pin_count, rec.pin_req = 0, None
        if rec.close_on_unpin != self._stale(st):
            raise RuntimeError(f"{st.req.rid}: close_on_unpin disagrees with the generation")
        st.record = None
        if rec.close_on_unpin:
            self.stats["unpin_release"] += 1
            self._release_record(rec)
        else:
            rec.last_used = self.clock()
            self.stats["park"] += 1
            self.ledger_version += 1

    def _exit(self, st: _ReqSession) -> None:
        req = st.req
        if self.owner.get(st.sid) is req:
            del self.owner[st.sid]  # the claim lapses once the claimant has finished
        req.hisparse_resume = None
        req.kv.mamba_host_restore = None
        del self._reqs[req.rid]

    def release_aborted_request(self, handle) -> None:
        """A request aborted while it holds no KV (waiting, possibly after a
        retract): its claim lapses; a claimed record stays PARKED (design §2.2)."""
        st = self._reqs.get(handle.rid)
        if st is None or st.req.kv.holds_kv:
            return  # no session, or it still exits through cache_finished_req
        self.stats["abort_waiting" + ("_with_record" if st.claim is not None else "")] += 1
        st.claim = None
        self._exit(st)

    # ------------------------------------------------------------ eviction

    def _evict_candidates(self) -> List[SessionRecord]:
        return [r for r in self.records if r.state == "PARKED"]

    def evictable_size(self) -> int:
        return sum(r.P for r in self.records if r.state == "PARKED")

    def protected_size(self) -> int:
        return sum(r.P for r in self.records if r.state == "PINNED")

    def evict(self, params: EvictParams) -> EvictResult:
        """LRU over PARKED records until params.num_tokens logical tokens are freed."""
        if params.swa_num_tokens or params.mamba_num:
            raise RuntimeError(f"{BACKEND_NAME} evicts logical tokens only, got {params}")
        freed = 0
        for rec in sorted(self._evict_candidates(), key=lambda r: r.last_used):
            if freed >= params.num_tokens:
                break
            freed += rec.P
            self.stats["evict"] += 1
            self._release_record(rec)
        return EvictResult(num_tokens_evicted=freed)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Callers (evict_from_tree_cache, check_decode_capacity) size the request
        against allocator.available_size(), the min of logical and prefill area.
        A record frees logical slots only (h06 note 1): evict the logical shortfall
        of the same demand, and nothing when only the prefill area is short."""
        alloc = self.token_to_kv_pool_allocator
        demand = params.num_tokens + alloc.available_size()
        short = demand - alloc.logical_attn_allocator.available_size()
        if short <= 0:
            return EvictResult()
        return self.evict(EvictParams(num_tokens=short))

    def reap_idle(self, now: float) -> None:
        """Records PARKED longer than the idle timeout go (design §2.5, R9)."""
        if not self.idle_timeout:
            return
        for rec in [r for r in self._evict_candidates() if now - r.last_used >= self.idle_timeout]:
            self.stats["reap"] += 1
            self._release_record(rec)

    def reset(self):
        """/flush_cache (the scheduler is fully idle)."""
        if self._reqs or self.owner or any(r.state != "PARKED" for r in self.records):
            raise RuntimeError(f"{BACKEND_NAME}.reset with live session requests")
        for rec in list(self.records):
            self._release_record(rec)

    def _release_record(self, rec: SessionRecord) -> None:
        if rec.pin_count or rec.logical is None or rec.released:
            raise RuntimeError(f"release of the {rec.state} record of {rec.sid} (P={rec.P})")
        self.token_to_kv_pool_allocator.free(rec.logical)
        self.coord.mem_pool_host.free(rec.host_blocks)
        self.coord.snapshots.free_buffer(rec.gdn)
        rec.released = True
        self.records.remove(rec)
        if self.index.get(rec.sid) is rec:
            del self.index[rec.sid]
        self.ledger_version += 1

    # ------------------------------------------------------------ checks

    def check_idle_invariants(self) -> None:
        """Nothing live: the pools hold exactly the parked records (M1a R3 plus the
        parked ledger, design §6), and every record is PARKED and indexed."""
        problems = []
        if self._reqs or self.owner:
            problems.append(f"live session requests {sorted(self._reqs)}, owners {sorted(self.owner)}")
        for rec in self.records:
            if rec.state != "PARKED" or self.index.get(rec.sid) is not rec:
                problems.append(f"record of {rec.sid} is {rec.state}, indexed {self.index.get(rec.sid) is rec}")
            elif rec.P % self.page_size or rec.P > rec.committed or rec.logical.numel() != rec.P or (
                rec.host_blocks.numel() * self.block_tokens != rec.P
            ):
                problems.append(f"record of {rec.sid}: P={rec.P}, committed {rec.committed}")
        if len(self.index) != len(self.records):
            problems.append(f"{len(self.index)} indexed of {len(self.records)} records")
        if problems:
            raise RuntimeError("QSA HiSparse session cache idle check: " + "; ".join(problems))
        parked = self.records
        self.coord.check_idle_invariants(
            parked_logical=torch.cat([r.logical for r in parked]) if parked else None,
            parked_host=torch.cat([r.host_blocks for r in parked]) if parked else None,
            parked_gdn=[r.gdn for r in parked],
        )
        logger.info(
            "QSA HiSparse session cache idle: %d parked records, %d tokens; stats %s; snapshots %s",
            len(parked),
            self.evictable_size(),
            dict(sorted(self.stats.items())),
            dict(sorted(self.coord.snapshots.stats.items())),
        )

    def pretty_print(self):
        return (
            f"{BACKEND_NAME}: {len(self.records)} records, evictable {self.evictable_size()}, "
            f"protected {self.protected_size()}, live session requests {len(self._reqs)}"
        )


def _factory(ctx: "TreeCacheBuildContext") -> QSAHiSparseSessionCache:
    return QSAHiSparseSessionCache(ctx.params, ctx.tp_worker.model_runner.hisparse_coordinator)


register_radix_cache_backend(BACKEND_NAME, _factory)
