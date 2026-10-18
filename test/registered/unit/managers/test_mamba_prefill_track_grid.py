"""The mamba prefill track point with --session-exact-resume off
(ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend, extra_buffer).

The session-exact-resume change added one condition that is not gated by the flag:
a prefill tracks only from a prefix on the page grid. The track point is
prefix + k * checkpoint_grid and the tree files the state under the page-aligned node
of that length, so from a prefix off the grid base filed the state under a shorter
node than it belongs to, silently. This is the only behavior difference with the
flag off:

- prefix on the page grid: the result is base's, case by case (mask, track seqlen,
  the forced read from h when the model's state chunk is coarser than the grid, the
  tracked length, the ping-pong swap);
- prefix off the page grid: no track (mask False, seqlen -1, nothing on the request
  changes), where base tracked an off-grid length.

CPU only.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.mamba import track_host as mamba_track_host
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.runtime_context import mamba_checkpoint_grid
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
SLOTS = (11, 12)  # the request's ping-pong track slots


def _base_entry(prefix, extend, grid, state_chunk):
    """Base's computation (before the page-grid condition), for a request without a
    branching point: (mask, track seqlen, tracked length or None)."""
    if extend < grid:
        return False, -1, None
    seqlen = prefix + extend
    aligned = prefix + extend // grid * grid
    if prefix + extend // state_chunk * state_chunk != aligned:
        seqlen = aligned + 1  # _force_track_h: read the state from h
    return True, seqlen, aligned


class TestMambaPrefillTrackGrid(CustomTestCase):
    def setUp(self):
        server_args = ServerArgs(model_path="dummy", page_size=PAGE, session_exact_resume=False)
        server_args._mamba_cache_chunk_size = PAGE  # otherwise the HF config of model_path is loaded
        set_global_server_args_for_scheduler(server_args)
        self.assertEqual(mamba_track_host.MAMBA_TRACK_HOST, "0")
        self.grid = mamba_checkpoint_grid(PAGE)
        self.assertEqual(self.grid, PAGE)
        self._rid = 0

    def _batch(self, state_chunk):
        """The ScheduleBatch fields the method reads."""
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(mamba_chunk_size=state_chunk)),
            tree_cache=SimpleNamespace(page_size=PAGE),
            req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i),
        )

    def _req(self, prefix, extend):
        self._rid += 1
        req = Req(
            rid=f"r{self._rid}",
            origin_input_text="",
            origin_input_ids=array("q", range(prefix + extend)),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
        )
        req.prefix_indices = torch.arange(prefix, dtype=torch.int64)
        req.set_extend_range(prefix, prefix + extend)
        req.kv.mamba_ping_pong_track_buffer = torch.tensor(SLOTS, dtype=torch.int64)
        req.kv.mamba_next_track_idx = 0
        return req

    def _run(self, prefix, extend, state_chunk):
        req = self._req(prefix, extend)
        entry = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(self._batch(state_chunk), req)
        return req, entry

    def test_prefix_on_the_page_grid_is_base(self):
        cases = 0
        for state_chunk in (PAGE, 2 * PAGE):  # equal to the grid, and coarser (the forced read from h)
            for prefix in (0, PAGE, 3 * PAGE, 40 * PAGE):
                for extend in (1, PAGE - 1, PAGE, PAGE + 1, 2 * PAGE - 1, 2 * PAGE, 5 * PAGE + 17):
                    req, entry = self._run(prefix, extend, state_chunk)
                    mask, seqlen, aligned = _base_entry(prefix, extend, self.grid, state_chunk)
                    msg = f"prefix {prefix} extend {extend} state chunk {state_chunk}"
                    self.assertEqual(entry.track_mask, mask, msg)
                    self.assertEqual(entry.track_seqlen, seqlen, msg)
                    self.assertEqual(entry.track_position, 0, msg)
                    self.assertEqual(entry.track_index, SLOTS[0], msg)
                    self.assertEqual(req.kv.mamba_last_track_seqlen, aligned, msg)
                    if mask:
                        self.assertEqual(aligned % PAGE, 0, msg)
                        self.assertEqual((req.kv.mamba_last_track_idx, req.kv.mamba_next_track_idx), (0, 1), msg)
                    else:
                        self.assertEqual((req.kv.mamba_last_track_idx, req.kv.mamba_next_track_idx), (None, 0), msg)
                    cases += 1
        self.assertEqual(cases, 56)

    def test_prefix_off_the_page_grid_does_not_track(self):
        for state_chunk in (PAGE, 2 * PAGE):
            for prefix in (1, PAGE - 1, PAGE + 9, 3 * PAGE + 33):
                for extend in (PAGE, 2 * PAGE, 5 * PAGE + 17):
                    msg = f"prefix {prefix} extend {extend} state chunk {state_chunk}"
                    base_mask, _, base_aligned = _base_entry(prefix, extend, self.grid, state_chunk)
                    # what base did: tracked, at a length off the page grid
                    self.assertTrue(base_mask, msg)
                    self.assertNotEqual(base_aligned % PAGE, 0, msg)
                    req, entry = self._run(prefix, extend, state_chunk)
                    self.assertFalse(entry.track_mask, msg)
                    self.assertEqual(entry.track_seqlen, -1, msg)
                    self.assertIsNone(req.kv.mamba_last_track_seqlen, msg)
                    self.assertEqual((req.kv.mamba_last_track_idx, req.kv.mamba_next_track_idx), (None, 0), msg)


if __name__ == "__main__":
    unittest.main()
