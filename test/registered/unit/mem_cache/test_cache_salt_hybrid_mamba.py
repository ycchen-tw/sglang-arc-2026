"""``cache_salt`` / ``extra_key`` scope radix prefix reuse on the tree cache the
arc-forge RL rollout server runs (Qwen3.8-Flash-Next, hybrid GDN): the default
factory's UnifiedRadixCache with FULL + MAMBA components, mamba extra_buffer,
page 64, python tree core. Requests go through the scheduler's own calls:
``Req.init_next_round_input`` (match + mamba COW), ``cache_unfinished_req``
(chunked prefill), ``release_kv_cache`` -> ``cache_finished_req``.

Same salt -> the episode's next turn hits its prefix; another salt -> no hit even
for identical tokens; no salt -> one shared namespace that salted entries are not
part of. CPU only.
"""

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    default_radix_cache_factory,
)
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 64  # the rollout server's --page-size
NUM_LAYERS = 8
FULL_LAYERS = (3, 7)
MAMBA_LAYERS = [i for i in range(NUM_LAYERS) if i not in FULL_LAYERS]


def _cpu_index(rows, device):
    # memory_pool._pinned_index pins host memory, which needs a CUDA driver.
    return torch.tensor(rows, dtype=torch.int64, device=device)


class TestCacheSaltHybridMamba(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(memory_pool, "_pinned_index", _cpu_index)
        patcher.start()
        self.addCleanup(patcher.stop)

        server_args = ServerArgs(model_path="dummy", page_size=PAGE)
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
        self.req_to_token_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=48,
            mamba_spec_state_size=8,
            max_context_len=4096,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=MAMBA_LAYERS),
            mamba_layer_ids=MAMBA_LAYERS,
            enable_mamba_extra_buffer=True,
            speculative_num_draft_tokens=3,
        )
        kv_pool = HybridLinearKVPool(
            size=8192,
            dtype=torch.bfloat16,
            page_size=PAGE,
            head_num=1,
            head_dim=8,
            full_attention_layer_ids=list(FULL_LAYERS),
            device="cpu",
            enable_memory_saver=False,
            mamba_pool=self.req_to_token_pool.mamba_pool,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=8192,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=PAGE,
            disable=False,
            enable_mamba_extra_buffer=True,
            eviction_policy="lru",
        )
        # The production selection chain, as the server log prints it:
        # "impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=True".
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
        self.assertEqual(
            self.cache.tree_components, (ComponentType.FULL, ComponentType.MAMBA)
        )
        self.assertTrue(self.cache.supports_mamba())
        self.assertTrue(self.cache.enable_mamba_extra_buffer)
        self._rid = 0
        # mamba slot -> the request whose state was donated into the tree there
        self.donated_by = {}
        # rid -> the mamba slot its match copied the state from
        self.cow_src = {}

    # -- one request, the way the scheduler drives it ---------------------------

    def _admit(self, tokens, *, cache_salt=None, extra_key=None, session_id=None):
        self._rid += 1
        req = Req(
            rid=f"r{self._rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
            session_id=session_id,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        req.init_next_round_input(self.cache)  # match + mamba copy-on-write
        src = req.kv.mamba_cow_src_index  # cleared again when the request ends
        self.cow_src[req.rid] = None if src is None else int(src)
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        self.req_to_token_pool.alloc([req])
        return req

    def _prefill(self, req, end):
        """Fresh KV for [committed, end): the first round starts right after the
        cached prefix the match handed the request."""
        start = req.kv.kv_committed_len or len(req.prefix_indices)
        new = self.allocator.alloc(end - start)
        self.assertIsNotNone(new)
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(start, end)), new.to(torch.int32)
        )
        req.kv.kv_committed_len = end
        req.kv.kv_allocated_len = end

    def _chunk(self, req, end):
        """One chunked-prefill round ending at ``end``; the mamba track buffer
        holds the state at the last page boundary, as extra_buffer tracks it."""
        start = len(req.prefix_indices)
        self._prefill(req, end)
        req.set_extend_range(start, end)
        req.kv.mamba_last_track_seqlen = end // PAGE * PAGE
        self.cache.cache_unfinished_req(req, chunked=True)

    def _finish(self, req, output):
        """Decode ``output`` and release the request (cache_finished_req)."""
        self._prefill(req, len(req.origin_input_ids))
        req.output_ids = array("q", output)
        self._prefill(req, len(req.origin_input_ids) + len(output))
        req.kv.mamba_last_track_seqlen = req.kv.kv_committed_len // PAGE * PAGE
        keep = self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
        slot = int(req.kv.mamba_ping_pong_track_buffer[keep])
        release_kv_cache(req, self.cache)
        self.donated_by[slot] = req.rid

    def _turn(self, tokens, output, **ns):
        """Admit, prefill, decode ``output``, finish. Returns the request: its
        prefix_indices are what the radix match gave it."""
        req = self._admit(tokens, **ns)
        self._finish(req, output)
        return req

    def _cow_source(self, req):
        """The request whose donated mamba state ``req``'s match copied."""
        src = self.cow_src[req.rid]
        return None if src is None else self.donated_by[src]

    @staticmethod
    def _seq(start, n):
        return list(range(start, start + n))

    # -- cases ------------------------------------------------------------------

    def test_episode_turns_share_only_their_own_salt(self):
        system = self._seq(1000, 3 * PAGE)  # identical in every episode
        turn1 = system + self._seq(5000, PAGE + 9)
        out1 = self._seq(6000, PAGE + 7)
        turn2 = turn1 + out1 + self._seq(7000, PAGE)

        a1 = self._turn(turn1, out1, cache_salt="ep-A", session_id="thread-A:0")
        self.assertEqual(len(a1.prefix_indices), 0)

        # Same episode, next turn: hits everything its turn 1 left in the tree
        # (page- and mamba-checkpoint-aligned), and its mamba state is copied
        # from the state turn 1 donated.
        a2 = self._turn(turn2, [1, 2, 3], cache_salt="ep-A", session_id="thread-A:0")
        cached_a1 = (len(turn1) + len(out1)) // PAGE * PAGE
        self.assertEqual(len(a2.prefix_indices), cached_a1)
        self.assertEqual(self._cow_source(a2), a1.rid)

        # Another episode, byte-identical tokens (same system prompt, same
        # history): no hit at all, not even the shared system prompt.
        b = self._turn(turn2, [1, 2, 3], cache_salt="ep-B", session_id="thread-B:0")
        self.assertEqual(len(b.prefix_indices), 0)
        self.assertIsNone(self._cow_source(b))

        # ...and episode B's own next turn hits B's entry, not A's.
        b2 = self._turn(turn2 + [4, 5], [6], cache_salt="ep-B")
        self.assertGreater(len(b2.prefix_indices), 0)
        self.assertEqual(self._cow_source(b2), b.rid)
        self.cache.sanity_check()

    def test_unsalted_requests_share_one_namespace_without_salted_entries(self):
        prompt = self._seq(1000, 5 * PAGE + 3)
        self._turn(prompt, [1], cache_salt="ep-A")

        u1 = self._turn(prompt, [1])
        self.assertEqual(len(u1.prefix_indices), 0)
        u2 = self._turn(prompt, [1])
        self.assertEqual(len(u2.prefix_indices), (len(prompt) - 1) // PAGE * PAGE)
        self.assertEqual(self._cow_source(u2), u1.rid)
        self.cache.sanity_check()

    def test_extra_key_and_cache_salt_are_distinct_namespaces(self):
        prompt = self._seq(1000, 4 * PAGE + 3)
        self._turn(prompt, [1], cache_salt="ep-A")
        by_extra_key = self._turn(prompt, [1], extra_key="ep-A")
        self.assertEqual(len(by_extra_key.prefix_indices), 0)
        again = self._turn(prompt, [1], extra_key="ep-A")
        self.assertEqual(len(again.prefix_indices), (len(prompt) - 1) // PAGE * PAGE)
        self.assertEqual(self._cow_source(again), by_extra_key.rid)
        self.cache.sanity_check()

    def test_chunked_prefill_inserts_under_the_salt(self):
        prompt = self._seq(1000, 6 * PAGE + 5)
        chunk_end = 4 * PAGE
        a = self._admit(prompt, cache_salt="ep-A")
        self._chunk(a, chunk_end)  # first chunk lands in the tree mid-prefill

        same = self._admit(prompt, cache_salt="ep-A")
        self.assertEqual(len(same.prefix_indices), chunk_end)
        other = self._admit(prompt, cache_salt="ep-B")
        self.assertEqual(len(other.prefix_indices), 0)
        unsalted = self._admit(prompt)
        self.assertEqual(len(unsalted.prefix_indices), 0)

        for req in (a, same, other, unsalted):
            self._finish(req, [1])
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
