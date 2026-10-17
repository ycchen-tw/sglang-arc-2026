"""Session tail park (--session-exact-resume, mem_cache/session_park.py) on the tree
cache the arc-forge RL rollout server runs: UnifiedRadixCache with FULL + MAMBA,
mamba extra_buffer, page 64, paged KV allocation.

Requests go through the scheduler's own calls: ``Req.init_next_round_input``
(match: the park is claimed there), ``inc_lock_ref`` + ``on_prefill_admitted``
(PrefillAdder._commit_prefill_admission: the claim is consumed), the request pool's
``alloc``, ``release_kv_cache`` -> ``cache_finished_req`` (the park is made), the
abort path's ``finish(handle, ABORT)``, retraction, ``evict_for_alloc`` and
``reset``. Every step checks the pool identities the scheduler's invariant checker
uses (available + evictable + protected + session_held + running == total, for KV
tokens, mamba slots and request rows). CPU only.
"""

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome, EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    default_radix_cache_factory,
)
from sglang.srt.mem_cache.session_park import (
    SessionTailPark,
    session_park_rows,
    session_resume_meta,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE = 64  # the rollout server's --page-size (QSA forces it)
TRACK = 256  # --mamba-track-interval
NUM_LAYERS = 8
FULL_LAYERS = (3, 7)
MAMBA_LAYERS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYERS]
RUN_ROWS = 6  # max_running_requests
PARK_ROWS = 3
KV_TOKENS = 64 * PAGE
MAMBA_SLOTS = 48


def _cpu_index(rows, device):
    # memory_pool._pinned_index pins host memory, which needs a CUDA driver.
    return torch.tensor(rows, dtype=torch.int64, device=device)


def _seq(start, n):
    return list(range(start, start + n))


class SessionTailParkHarness(CustomTestCase):
    EXACT_RESUME = True

    def setUp(self):
        patcher = mock.patch.object(memory_pool, "_pinned_index", _cpu_index)
        patcher.start()
        self.addCleanup(patcher.stop)

        server_args = ServerArgs(
            model_path="dummy",
            page_size=PAGE,
            session_exact_resume=self.EXACT_RESUME,
            session_exact_resume_max_parks=PARK_ROWS if self.EXACT_RESUME else None,
        )
        # mamba_cache_chunk_size otherwise loads the HF config of model_path.
        server_args._mamba_cache_chunk_size = PAGE
        set_global_server_args_for_scheduler(server_args)

        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=64,
                n_groups=1,
                num_heads=2,
                head_dim=16,
                state_size=16,
                conv_kernel=4,
            )
        park_rows = session_park_rows(RUN_ROWS)
        self.assertEqual(park_rows, PARK_ROWS if self.EXACT_RESUME else 0)
        # As KVCacheConfigurator._build_hybrid_req_pool sizes it.
        self.pool = HybridReqToTokenPool(
            size=RUN_ROWS + park_rows,
            mamba_size=MAMBA_SLOTS,
            mamba_spec_state_size=RUN_ROWS,
            max_context_len=4096,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=MAMBA_LAYERS),
            mamba_layer_ids=MAMBA_LAYERS,
            enable_mamba_extra_buffer=True,
            speculative_num_draft_tokens=3,
        )
        self.pool.session_park_rows = park_rows
        kv_pool = HybridLinearKVPool(
            size=KV_TOKENS,
            dtype=torch.bfloat16,
            page_size=PAGE,
            head_num=1,
            head_dim=8,
            full_attention_layer_ids=list(FULL_LAYERS),
            device="cpu",
            enable_memory_saver=False,
            mamba_pool=self.pool.mamba_pool,
        )
        self.allocator = PagedTokenToKVPoolAllocator(
            size=KV_TOKENS,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=PAGE,
            disable=False,
            enable_mamba_extra_buffer=True,
            eviction_policy="lru",
            session_exact_resume=self.EXACT_RESUME,  # as kv_cache_builder sets it
        )
        self.cache = default_radix_cache_factory(
            TreeCacheBuildContext(
                server_args=server_args,
                params=params,
                is_hybrid_swa=False,
                is_hybrid_ssm=True,
                enable_hierarchical_cache=False,
                disable_radix_cache=False,
                effective_chunked_prefill_size=8192,
                tp_worker=None,
                model_config=None,
                tp_size=1,
                tp_rank=0,
                tp_group=None,
            )
        )
        self.assertIsInstance(self.cache, UnifiedRadixCache)
        self.park = self.cache.tail_park
        if self.EXACT_RESUME:
            self.assertIsInstance(self.park, SessionTailPark)
        else:
            self.assertIsNone(self.park)
        self._rid = 0
        self.waiting = []  # matched, not admitted
        self.running = []  # admitted, not released

    # -- the scheduler's calls ------------------------------------------------

    def _req(self, tokens, *, session="sess-A", salt="ep-A", max_new=512):
        self._rid += 1
        return Req(
            rid=f"r{self._rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=max_new),
            session_id=session,
            cache_salt=salt,
        )

    def _match(self, req):
        """Scheduling match (Scheduler.get_new_batch_prefill)."""
        req.init_next_round_input(self.cache)
        if req not in self.waiting:
            self.waiting.append(req)
        return req

    def _admit(self, req):
        """PrefillAdder._commit_prefill_admission, then alloc_for_extend: the row,
        the prefix written into it, KV for the whole extend."""
        self.waiting.remove(req)
        req.set_extend_range(len(req.prefix_indices), len(req.full_untruncated_fill_ids))
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        self.cache.on_prefill_admitted(req)
        req.admitted_prefix_len = len(req.prefix_indices)
        req.prefill_weight_version = None
        self.pool.alloc([req])
        self.pool.write(
            (req.kv.req_pool_idx, slice(0, len(req.prefix_indices))),
            req.prefix_indices.to(torch.int32),
        )
        # Deferred COW / clear run on the forward stream; nothing to copy here.
        req.kv.mamba_cow_src_index = None
        req.kv.mamba_needs_clear = False
        self._extend(req, len(req.full_untruncated_fill_ids))
        # BatchResultProcessor: the first prefill result after the admission
        req.prefill_weight_version = f"v{self._rid}"
        self.running.append(req)
        return req

    def _extend(self, req, end):
        """KV for [committed, end) the way the paged allocator hands it out: the
        rest of the page the prefix ends in first, then fresh pages."""
        row = self.pool.req_to_token[req.kv.req_pool_idx]
        pos = req.kv.kv_committed_len or len(req.prefix_indices)
        if pos % PAGE and pos < end:
            n = min(end, ceil_align(pos, PAGE)) - pos
            last = int(row[pos - 1])
            row[pos : pos + n] = torch.arange(last + 1, last + 1 + n, dtype=row.dtype)
            pos += n
        if pos < end:
            new = self.allocator.alloc(ceil_align(end - pos, PAGE))
            self.assertIsNotNone(new)
            row[pos:end] = new[: end - pos].to(row.dtype)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = end

    def _run(self, req, output, *, overshoot=False):
        """Prefill the prompt and decode ``output``. extra_buffer tracks the mamba
        state at the last grid point of a prefill whose prefix is on the grid,
        and at every TRACK multiple a decode step reaches. Overlap scheduling
        leaves committed one past the last output id."""
        n_in = len(req.origin_input_ids)
        prefix = len(req.prefix_indices)
        self._extend(req, n_in)
        if prefix % PAGE == 0 and n_in - prefix >= PAGE:
            req.kv.mamba_last_track_seqlen = prefix + (n_in - prefix) // PAGE * PAGE
        req.output_ids = array("q", output)
        committed = n_in + len(output) - 1 + int(overshoot)
        self._extend(req, committed)
        decode_track = committed // TRACK * TRACK
        if decode_track > n_in:
            req.kv.mamba_last_track_seqlen = decode_track
        return committed

    def _release(self, req, reason):
        req.finished_reason = reason
        if reason is not None and not isinstance(reason, FINISH_ABORT):
            req.finished_len = len(req.output_ids)
        release_kv_cache(req, self.cache)
        self.running.remove(req)

    def _turn(self, tokens, output, *, overshoot=False, **ns):
        req = self._admit(self._match(self._req(tokens, **ns)))
        self._run(req, output, overshoot=overshoot)
        self._release(req, FINISH_MATCHED_TOKEN(matched=output[-1]))
        return req

    def _abort_waiting(self, req):
        """Scheduler.abort_request on a queued request."""
        self.waiting.remove(req)
        self.cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        req.finished_reason = FINISH_ABORT()
        release_kv_cache(req, self.cache)

    # -- checks ---------------------------------------------------------------

    def _accounting(self):
        c = self.cache
        held = [r for r in self.running if r.kv.holds_kv]
        uncached = sum(
            ceil_align(r.kv.kv_allocated_len, PAGE) - r.kv.cache_protected_len
            for r in held
        )
        self.assertEqual(
            self.allocator.available_size()
            + c.full_evictable_size()
            + c.full_protected_size()
            + c.session_held_full_tokens()
            + uncached,
            KV_TOKENS,
        )
        mamba_running = sum(
            int(r.kv.holds_mamba) + (0 if r.kv.mamba_ping_pong_track_buffer is None else 2)
            for r in self.running + self.waiting
        )
        self.assertEqual(
            self.pool.mamba_allocator.available_size()
            + c.mamba_evictable_size()
            + c.mamba_protected_size()
            + c.session_held_mamba_slots()
            + mamba_running,
            self.pool.mamba_pool.size,
        )
        self.assertEqual(
            len(self.pool.free_slots) + c.session_held_req_count() + len(held),
            self.pool.size,
        )
        if self.park is not None:
            self.park.sanity_check(idle=not self.waiting and not self.running)
        if not self.waiting and not self.running:
            c.sanity_check()

    def _drain(self):
        """Release every park and evict the whole tree: all pools are full again."""
        if self.park is not None:
            while self.park.evict_lru():
                pass
            self.assertEqual(self.park.rows(), 0)
        self.cache.evict_for_alloc(
            EvictParams(num_tokens=KV_TOKENS, mamba_num=MAMBA_SLOTS)
        )
        self.assertEqual(self.allocator.available_size(), KV_TOKENS)
        self.assertEqual(self.pool.mamba_allocator.available_size(), MAMBA_SLOTS)
        self.assertEqual(len(self.pool.free_slots), self.pool.size)
        self._accounting()

    def _resume(self, req):
        meta = session_resume_meta(req, exact_resume=self.EXACT_RESUME)
        self.assertEqual(set(meta), {"mode", "reason", "pos", "prefill_version"})
        return meta

    def _mrp(self, req):
        meta = self._resume(req)
        return meta["mode"], meta["reason"], meta["pos"]


class TestSessionTailPark(SessionTailParkHarness):
    S1_IN = _seq(1000, 3 * PAGE + 9)  # 201
    S1_OUT = _seq(5000, 150)

    def _first_turn(self, **kw):
        r1 = self._turn(self.S1_IN, self.S1_OUT, **kw)
        self.assertEqual(self._mrp(r1), ("fallback", "no_park", 0))
        self.assertIsInstance(self._resume(r1)["prefill_version"], str)
        self._accounting()
        return r1

    def _s1(self):
        return self.S1_IN + self.S1_OUT

    def test_next_turn_resumes_at_the_exact_end(self):
        self._first_turn()
        # committed 350 = 201 + 150 - 1; the tree holds [0, 256) (decode track),
        # the park [256, 350) on two pages.
        self.assertEqual(self.park.rows(), 1)
        self.assertEqual(self.park.kv_tokens(claimed=False), 2 * PAGE)
        (park,) = self.park.lru.values()
        self.assertEqual((park.pos, park.kv.cache_protected_len), (350, 256))
        row, slot = park.kv.req_pool_idx, park.kv.mamba_pool_idx
        kv_350 = self.pool.req_to_token[row, :350].clone()

        r2 = self._match(self._req(self._s1() + _seq(7000, 40)))
        self.assertEqual(len(r2.prefix_indices), 350)
        self.assertEqual(r2.kv.cache_protected_len, 256)
        self.assertFalse(r2.kv.holds_mamba)  # no copy-on-write: the park's slot
        # Not admitted yet: no prefix used, no prefill forward.
        self.assertEqual(self._mrp(r2), ("exact", None, None))
        self.assertIsNone(self._resume(r2)["prefill_version"])
        self.assertEqual(self.park.kv_tokens(claimed=True), 2 * PAGE)
        self._accounting()
        # A re-match while waiting (another scheduling round) keeps the claim.
        self._match(r2)
        self.assertEqual(len(r2.prefix_indices), 350)

        self._admit(r2)
        self.assertEqual(self._mrp(r2), ("exact", None, 350))
        self.assertEqual(self.park.rows(), 0)
        self.assertEqual(r2.kv.req_pool_idx, row)
        self.assertIs(r2.kv.mamba_pool_idx, slot)
        # P = 350 is inside a QSA compress group: the extend reads [348, 350)
        # from this row's pending ring (exp/ser-qsa-misaligned).
        self.assertTrue(r2.qsa_ring_resume)
        # The extend wrote [350, 391) behind the producer's KV, continuing its
        # last page; nothing before 350 moved.
        self.assertEqual(r2.kv.kv_committed_len, 391)
        self.assertTrue(torch.equal(self.pool.req_to_token[row, :350], kv_350))
        self.assertEqual(int(self.pool.req_to_token[row, 350]), int(kv_350[-1]) + 1)
        self._accounting()

        # Prefill [350, 391), decode 100: committed 490, no new track point (512
        # is not reached; the prefill from an off-grid prefix tracks nothing), so
        # the tree keeps [0, 256) and the park grows to [256, 490).
        self._run(r2, _seq(8000, 100))
        self._accounting()
        self._release(r2, FINISH_LENGTH(length=100))
        (park,) = self.park.lru.values()
        self.assertEqual((park.pos, park.kv.cache_protected_len), (490, 256))
        self.assertEqual(park.kv.req_pool_idx, row)
        self._accounting()

        # Third turn crosses 512: the tree takes [0, 512) at park time.
        s2 = r2.origin_input_ids.tolist() + r2.output_ids.tolist()
        r3 = self._admit(self._match(self._req(s2 + _seq(9000, 10))))
        self.assertEqual(len(r3.prefix_indices), 490)
        self._run(r3, _seq(10000, 60), overshoot=True)
        self._release(r3, FINISH_MATCHED_TOKEN(matched=0))
        (park,) = self.park.lru.values()
        # committed = 501 + 60 = 561 (overshoot: the last output id is forwarded)
        self.assertEqual((park.pos, park.kv.cache_protected_len), (561, 512))
        self._accounting()
        self._drain()

    def test_claim_is_exclusive(self):
        self._first_turn()
        a = self._match(self._req(self._s1() + _seq(7000, 40)))
        b = self._match(self._req(self._s1() + _seq(7500, 30)))  # a resend / sibling
        self.assertEqual(self._resume(a)["mode"], "exact")
        self.assertEqual(self._mrp(b), ("fallback", "claimed", None))
        self.assertEqual(len(b.prefix_indices), 256)  # the tree's track point
        self.assertTrue(b.kv.holds_mamba)  # copy-on-write from the tree's state
        self._accounting()
        for r in (a, b):
            self._admit(r)
        self.assertEqual(self._mrp(a), ("exact", None, 350))
        self.assertEqual(self._mrp(b), ("fallback", "claimed", 256))  # == cached_tokens
        self._accounting()
        self._run(a, _seq(8000, 20))
        self._run(b, _seq(8500, 20))
        self._release(a, FINISH_MATCHED_TOKEN(matched=0))
        self._release(b, FINISH_MATCHED_TOKEN(matched=0))
        self.assertEqual(len(self.park.parks["sess-A"]), 2)
        self._accounting()
        # b's continuation resumes b's end, a's continuation a's end.
        sb = b.origin_input_ids.tolist() + b.output_ids.tolist()
        b2 = self._match(self._req(sb + [1, 2, 3]))
        self.assertEqual(self._mrp(b2), ("exact", None, None))
        self.assertEqual(self.park.claims[b2.rid].pos, len(sb) - 1)
        sa = a.origin_input_ids.tolist() + a.output_ids.tolist()
        a2 = self._match(self._req(sa + [4]))
        self.assertEqual(self.park.claims[a2.rid].pos, len(sa) - 1)
        self._accounting()
        for r in (a2, b2):
            self._admit(r)
            self.assertEqual(self._mrp(r)[2], len(r.prefix_indices))
            self._run(r, [5, 6])
            self._release(r, FINISH_MATCHED_TOKEN(matched=6))
        self._accounting()
        self._drain()

    def test_mismatch_reasons(self):
        self._first_turn()
        s1 = self._s1()
        rewind = self._match(self._req(s1[:300] + [1, 2, 3]))
        self.assertEqual(self._resume(rewind)["reason"], "ids_mismatch")
        other_salt = self._match(self._req(s1 + [1], salt="ep-B"))
        self.assertEqual(self._resume(other_salt)["reason"], "ids_mismatch")
        self.assertEqual(len(other_salt.prefix_indices), 0)
        # logprob_start_len caps the matchable key below the park's end
        capped = self._req(s1 + [1, 2])
        capped.return_logprob, capped.logprob_start_len = True, 300
        self._match(capped)
        self.assertEqual(self._resume(capped)["reason"], "key_limit")
        no_session = self._match(self._req(s1 + [1], session=None))
        self.assertEqual(self._mrp(no_session), ("none", None, None))
        other_session = self._match(self._req(s1 + [1], session="sess-B"))
        self.assertEqual(self._resume(other_session)["reason"], "no_park")
        self._accounting()
        for r in list(self.waiting):
            self._abort_waiting(r)
        self.assertEqual(self.park.rows(), 1)  # nobody claimed it
        self._accounting()
        self._drain()

    def test_abort_before_admission_hands_the_claim_back(self):
        self._first_turn()
        tokens = self._s1() + _seq(7000, 40)
        a = self._match(self._req(tokens))
        self.assertEqual(self._resume(a)["mode"], "exact")
        self._abort_waiting(a)
        self.assertEqual(self._mrp(a), ("fallback", "abort", None))
        self.assertEqual(self.park.kv_tokens(claimed=True), 0)
        self.assertEqual(self.park.rows(), 1)
        self._accounting()
        b = self._match(self._req(tokens))
        self._admit(b)
        self.assertEqual(self._mrp(b), ("exact", None, 350))
        self._run(b, [1, 2, 3])
        self._release(b, FINISH_MATCHED_TOKEN(matched=3))
        self._accounting()
        self._drain()

    def test_abort_after_admission_releases_everything(self):
        self._first_turn()
        a = self._admit(self._match(self._req(self._s1() + _seq(7000, 40))))
        # The client's resend arrives while a (it took the park) still runs.
        early = self._match(self._req(self._s1() + _seq(7000, 40)))
        self.assertEqual(self._mrp(early), ("fallback", "claimed", None))
        self._abort_waiting(early)
        self._run(a, [1, 2, 3, 4])
        self._release(a, FINISH_ABORT())
        self.assertEqual(self.park.rows(), 0)
        self._accounting()
        nxt = self._match(self._req(self._s1() + _seq(7000, 40) + [1, 2, 3, 4, 5]))
        self.assertEqual(self._resume(nxt)["reason"], "abort")
        self._abort_waiting(nxt)
        self._drain()

    def test_retracted_request_starts_over_from_the_tree(self):
        self._first_turn()
        a = self._admit(self._match(self._req(self._s1() + _seq(7000, 40))))
        self._run(a, [1, 2, 3])
        # ScheduleBatch.release_req: retract a running request
        release_kv_cache(a, self.cache, is_insert=False)
        self.running.remove(a)
        a.reset_for_retract()
        self.assertEqual(self._resume(a)["reason"], "retract")
        self._accounting()
        self._match(a)
        self.assertEqual(self._resume(a)["reason"], "retract")
        self.assertEqual(len(a.prefix_indices), 256)
        cached_first = a.admitted_prefix_len
        self._admit(a)
        # pos is the last admission's prefix (cached_tokens keeps the first one)
        self.assertEqual((cached_first, self._mrp(a)), (350, ("fallback", "retract", 256)))
        self._run(a, [1, 2, 3, 4])
        self._release(a, FINISH_MATCHED_TOKEN(matched=4))
        self.assertEqual(self.park.rows(), 1)  # it parks again at its own end
        self._accounting()
        self._drain()

    def test_ends_that_do_not_park(self):
        cases = [
            (FINISH_MATCHED_STR(matched="</tool>"), "stop_string"),
            (FINISH_ABORT(), "abort"),
        ]
        for i, (reason, why) in enumerate(cases):
            sid = f"sess-{i}"
            r = self._admit(self._match(self._req(self.S1_IN, session=sid)))
            self._run(r, self.S1_OUT)
            self._release(r, reason)
            nxt = self._match(self._req(self._s1() + [1], session=sid))
            self.assertEqual(self._resume(nxt)["reason"], why, reason)
            self._abort_waiting(nxt)
        # stop-trimmed output, finished in prefill
        r = self._admit(self._match(self._req(self.S1_IN, session="sess-trim")))
        self._run(r, self.S1_OUT)
        r.finished_reason = FINISH_MATCHED_TOKEN(matched=0)
        r.finished_len = len(self.S1_OUT) - 1
        release_kv_cache(r, self.cache)
        self.running.remove(r)
        r = self._admit(self._match(self._req(self.S1_IN, session="sess-pf")))
        self._run(r, [7])
        self._release(r, FINISH_LENGTH(length=1))
        for sid, why in (("sess-trim", "stop_string"), ("sess-pf", "prefill_finished")):
            nxt = self._match(self._req(self.S1_IN + [1, 2], session=sid))
            self.assertEqual(self._resume(nxt)["reason"], why)
            self._abort_waiting(nxt)
        self.assertEqual(self.park.rows(), 0)
        self._accounting()
        self._drain()

    def test_eviction_releases_the_oldest_unclaimed_park_last(self):
        for sid in ("sess-A", "sess-B", "sess-C"):
            self._turn(self.S1_IN, self.S1_OUT, session=sid, salt=sid)
        self.assertEqual(self.park.rows(), 3)
        claimant = self._match(self._req(self._s1() + [1], session="sess-A", salt="sess-A"))
        self.assertEqual(self._resume(claimant)["mode"], "exact")
        self._accounting()
        # A page short: the tree has nothing evictable (the parks lock their
        # paths), so the oldest unclaimed park -- B, A is claimed -- goes, and its
        # two pages cover the shortfall.
        free = self.allocator.available_size()
        self.cache.evict_for_alloc(EvictParams(num_tokens=PAGE))
        self.assertEqual(sorted(self.park.parks), ["sess-A", "sess-C"])
        self.assertEqual(self.allocator.available_size(), free + 2 * PAGE)
        self._accounting()
        b = self._match(self._req(self._s1() + [1], session="sess-B", salt="sess-B"))
        self.assertEqual(self._resume(b)["reason"], "evicted")
        self._abort_waiting(b)
        # Everything: C goes, the claimed A stays.
        self.cache.evict_for_alloc(EvictParams(num_tokens=KV_TOKENS))
        self.assertEqual(sorted(self.park.parks), ["sess-A"])
        self._accounting()
        self._admit(claimant)
        self._run(claimant, [1])
        self._release(claimant, FINISH_LENGTH(length=1))  # finished in prefill
        self._accounting()
        self._drain()

    def test_capacity_and_per_session_limit(self):
        for i in range(PARK_ROWS + 1):
            self._turn(self.S1_IN, self.S1_OUT, session=f"s{i}", salt=f"s{i}")
            self._accounting()
        self.assertEqual(self.park.rows(), PARK_ROWS)
        self.assertNotIn("s0", self.park.parks)  # the oldest made room
        nxt = self._match(self._req(self._s1() + [1], session="s0", salt="s0"))
        self.assertEqual(self._resume(nxt)["reason"], "evicted")
        self._abort_waiting(nxt)
        self._drain()
        # Per session at most two: a third end of the session replaces the oldest.
        for out in ([1, 2], [3, 4], [5, 6]):
            self._turn(self.S1_IN, out)
        self.assertEqual([p.ids[-1] for p in self.park.parks["sess-A"]], [3, 5])
        self._accounting()
        self._drain()

    def test_reasons_are_the_contract_set(self):
        from sglang.srt.mem_cache import session_park

        self.assertEqual(
            session_park.RESUME_REASONS,
            {
                "no_park", "claimed", "ids_mismatch", "key_limit", "evicted",
                "retract", "abort", "prefill_finished", "stop_string", "grammar",
                "think_budget", "snapshot_mismatch", "unsupported",
                "qsa_misaligned", "spec_unsupported",
            },
        )
        with self.assertRaises(AssertionError):
            session_park._fallback("park_full")

    def test_reset_drops_parks(self):
        self._first_turn()
        self.cache.reset()  # Scheduler.flush_cache: then the pools are cleared
        self.pool.clear()
        self.allocator.clear()
        self.assertEqual(self.park.rows(), 0)
        self.assertEqual(self.park.kv_tokens(claimed=False), 0)
        nxt = self._match(self._req(self._s1() + [1]))
        self.assertEqual(self._resume(nxt)["reason"], "no_park")
        self._abort_waiting(nxt)
        self._drain()


class TestSessionTailParkOff(SessionTailParkHarness):
    """Flag off: no park, meta mode "none", and the next turn re-prefills from
    the tree's track point exactly as before."""

    EXACT_RESUME = False

    def test_off_is_the_tree_alone(self):
        s_in, s_out = _seq(1000, 3 * PAGE + 9), _seq(5000, 150)
        self._turn(s_in, s_out)
        nxt = self._match(self._req(s_in + s_out + [1]))
        self.assertEqual(len(nxt.prefix_indices), 256)
        self.assertEqual(self._mrp(nxt), ("none", None, None))
        self._admit(nxt)
        self.assertEqual(self._mrp(nxt), ("none", None, None))
        self.assertIsInstance(self._resume(nxt)["prefill_version"], str)
        self._run(nxt, [1])
        self._release(nxt, FINISH_LENGTH(length=1))
        self._accounting()
        self._drain()


if __name__ == "__main__":
    unittest.main()
