"""ARC_MAMBA_TRACK_HOST: the extend's mamba track plan from host mirrors.

Two properties are checked:
  * Equivalence (CPU tensors). The host plan gives the same values, dtypes and devices as the old device-tensor
    path on random batches. That covers _init_track_conv_indices + _init_track_ssm_indices for GDN and Mamba2,
    the GDN mask rows, the FlashInfer checkpoint plan, and the scheduler's slot gather.
  * No host synchronization (meta tensors). The "device" tensors live on the meta device, so any read of their
    data on the host raises: .item() / bool() / .tolist(), a boolean-mask index or nonzero() (data-dependent
    shape), and a copy to the CPU. The host path has to run through cleanly. The old path is the positive
    control and has to raise. A Python-level guard (Tensor.item / tolist / cpu / numpy / nonzero / __bool__ /
    __int__ / __index__ patched to raise) covers the same thing on CPU tensors.
"""

import random
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import Mamba2AttnBackend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels import gdn_flashinfer
from sglang.srt.layers.attention.mamba import track_host
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

CHUNK = 64  # model state chunk == mamba_cache_chunk_size (page 64), as in the RL deployment
CONV_STATE_LEN = 3
TRACK_OUTPUTS = (
    "mamba_track_rows",
    "track_conv_indices",
    "track_chunk_idx",
    "track_ssm_h_src",
    "track_ssm_h_dst",
    "track_ssm_h_batch_src",
    "track_ssm_final_src",
    "track_ssm_final_dst",
    "track_ssm_seq_idx",
    "track_ssm_end_locs",
    "track_ssm_recompute_dst",
)


@contextmanager
def _flag(value):
    with patch.object(track_host, "MAMBA_TRACK_HOST", value):
        yield


def _random_batch(rng: random.Random, *, grid: int, any_seqlen: bool = False):
    """Host lists like ScheduleBatch.prepare_for_extend builds them. any_seqlen also
    draws tracked seqlens off the scheduler's formula, to cover the planner's whole input domain."""
    bs = rng.randint(1, 12)
    extend, prefix, mask, seqlens = [], [], [], []
    for _ in range(bs):
        ext = rng.choice(
            [rng.randint(1, 3 * grid), grid * rng.randint(1, 4), rng.randint(1, 2000)]
        )
        pre = rng.choice([0, CHUNK * rng.randint(0, 40), rng.randint(0, 5000)])
        m = ext >= grid
        if not m:
            seqlen = -1
        elif any_seqlen and rng.random() < 0.4:
            seqlen = pre + rng.randint(1, ext)
        else:
            seqlen = pre + ext
            aligned = pre + (ext // grid) * grid
            if pre + (ext // CHUNK) * CHUNK != aligned:
                seqlen = aligned + 1  # _force_track_h
        extend.append(ext)
        prefix.append(pre)
        mask.append(m)
        seqlens.append(seqlen)
    return extend, prefix, mask, seqlens


def _forward_batch(extend, prefix, mask, seqlens, device, *, seed=0):
    bs = len(extend)
    g = torch.Generator().manual_seed(seed)
    track_slots = torch.randperm(1000, generator=g)[:bs].to(torch.int64)
    start_loc = [0] * bs
    for i in range(1, bs):
        start_loc[i] = start_loc[i - 1] + extend[i - 1]
    fb = SimpleNamespace(
        batch_size=bs,
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.arange(bs, dtype=torch.int64).to(device),
        extend_seq_lens=torch.tensor(extend, dtype=torch.int32).to(device),
        extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32).to(device),
        extend_start_loc=torch.tensor(start_loc, dtype=torch.int32).to(device),
        extend_seq_lens_cpu=list(extend),
        extend_prefix_lens_cpu=list(prefix),
        mamba_track_mask=torch.tensor(mask, dtype=torch.bool).to(device),
        mamba_track_seqlens=torch.tensor(seqlens, dtype=torch.int64).to(device),
        mamba_track_indices=track_slots.to(device),
        mamba_extend_track_mask_cpu=list(mask),
        mamba_extend_track_seqlens_cpu=list(seqlens),
        _original_batch_size=None,
        multi_item_delimiter_indices=None,
    )
    fb.mamba_track_aligned_lens = lambda: ForwardBatch.mamba_track_aligned_lens(fb)
    return fb


def _backend(cls, device, cache_indices):
    backend = object.__new__(cls)
    backend.device = device
    backend._mamba_chunk_size = CHUNK
    backend.conv_states_shape = (16, 8, CONV_STATE_LEN)
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda req_pool_indices: cache_indices[req_pool_indices],
        translate_mamba_indices=lambda indices: indices,
    )
    backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=True)
    return backend


def _query_start_loc(fb):
    bs = fb.batch_size
    qsl = torch.empty((bs + 1,), dtype=torch.int32, device=fb.extend_start_loc.device)
    qsl[:bs] = fb.extend_start_loc
    qsl[bs] = fb.extend_start_loc[-1] + fb.extend_seq_lens[-1]
    return qsl


class _ChunkSizeMixin:
    def setUp(self):
        # mamba_cache_chunk_size() reads the published config (see test_gdn_prefill_backend_policy).
        override = get_context().override_server_args(_mamba_cache_chunk_size=CHUNK)
        override.install()
        self.addCleanup(override.restore)


class TestTrackPlanEquivalence(_ChunkSizeMixin, unittest.TestCase):
    def _old(self, backend, qsl, cache_indices, fb):
        return (
            fb.mamba_track_mask.nonzero(as_tuple=True)[0],
            backend._init_track_conv_indices(qsl, fb),
        ) + tuple(backend._init_track_ssm_indices(cache_indices, fb))

    def _check_cls(self, cls, *, any_seqlen):
        rng = random.Random(1234)
        checked_rows = 0
        for case in range(400):
            grid = rng.choice([CHUNK, 256])
            extend, prefix, mask, seqlens = _random_batch(
                rng, grid=grid, any_seqlen=any_seqlen
            )
            fb = _forward_batch(extend, prefix, mask, seqlens, "cpu", seed=case)
            if not any(mask):
                continue
            cache_indices = torch.randperm(4096)[:64].to(torch.int32)
            backend = _backend(cls, "cpu", cache_indices)
            qsl = _query_start_loc(fb)
            mamba_cache = cache_indices[fb.req_pool_indices]
            old = self._old(backend, qsl, mamba_cache, fb)
            # "check" also compares internally and raises; the explicit comparison
            # below makes the test independent of that code.
            with _flag("check"):
                new = backend._init_track_indices_host(qsl, mamba_cache, fb)
            for name, a, b in zip(TRACK_OUTPUTS, old, new, strict=True):
                with self.subTest(cls=cls.__name__, case=case, output=name):
                    track_host.assert_same(name, a, b)
            checked_rows += sum(mask)
        self.assertGreater(checked_rows, 500)

    def test_gdn_matches_device_path(self):
        self._check_cls(GDNAttnBackend, any_seqlen=False)
        self._check_cls(GDNAttnBackend, any_seqlen=True)

    def test_mamba2_matches_device_path(self):
        self._check_cls(Mamba2AttnBackend, any_seqlen=False)
        self._check_cls(Mamba2AttnBackend, any_seqlen=True)

    def test_flashinfer_checkpoint_plan_matches(self):
        rng = random.Random(99)
        for case in range(400):
            extend, prefix, mask, seqlens = _random_batch(
                rng, grid=rng.choice([CHUNK, 256]), any_seqlen=True
            )
            fb = _forward_batch(extend, prefix, mask, seqlens, "cpu", seed=case)
            ref_src, ref_cu = gdn_flashinfer._flashinfer_checkpoint_plan_cpu(fb, CHUNK)
            src, cu = track_host.plan_flashinfer_checkpoints(
                mask, seqlens, prefix, extend, CHUNK
            )
            self.assertEqual(ref_src.tolist(), src)
            self.assertEqual(ref_cu.tolist(), cu)

    def test_full_metadata_matches_and_gdn_rows(self):
        rng = random.Random(7)
        for case in range(200):
            extend, prefix, mask, seqlens = _random_batch(rng, grid=256)
            if not any(mask):
                continue
            cache_indices = torch.randperm(4096)[:64].to(torch.int32)
            mds = []
            for flag in ("0", "check"):
                fb = _forward_batch(extend, prefix, mask, seqlens, "cpu", seed=case)
                backend = _backend(GDNAttnBackend, "cpu", cache_indices)
                with _flag(flag):
                    backend.init_forward_metadata(fb)
                mds.append(backend.forward_metadata)
            old, new = mds
            self.assertIsNone(old.mamba_track_rows)
            self.assertIsNotNone(new.mamba_track_rows)
            self.assertEqual(old.has_mamba_track_mask, new.has_mamba_track_mask)
            for name in TRACK_OUTPUTS[1:] + (
                "mamba_track_mask_indices",
                "conv_states_mask_indices",
                "state_checkpoint_cu_starts",
            ):
                with self.subTest(case=case, output=name):
                    track_host.assert_same(name, getattr(old, name), getattr(new, name))
            self.assertEqual(old.num_state_checkpoints, new.num_state_checkpoints)
            self.assertEqual(
                old.state_checkpoint_every_n_tokens, new.state_checkpoint_every_n_tokens
            )

    def test_check_mode_reports_coverage(self):
        # A check run must show it compared something, not just that nothing raised.
        extend, prefix, mask, seqlens = _meta_case()
        track_host._check_counts.clear()
        with (
            patch.object(track_host, "CHECK_LOG_EVERY", 2),
            self.assertLogs(track_host.logger, "INFO") as logs,
        ):
            for _ in range(2):
                fb = _forward_batch(extend, prefix, mask, seqlens, "cpu")
                backend = _backend(GDNAttnBackend, "cpu", torch.arange(64, dtype=torch.int32))
                with _flag("check"):
                    backend.init_forward_metadata(fb)
        self.assertEqual(track_host._check_counts["extend"], 2)
        self.assertEqual(track_host._check_counts["extend_rows"], 8)
        self.assertEqual(track_host._check_counts["checkpoints"], 2)
        self.assertIn("all comparisons matched", logs.output[-1])

    def test_no_tracked_row_skips_tracking(self):
        fb = _forward_batch([10, 20], [0, 64], [False, False], [-1, -1], "cpu")
        backend = _backend(GDNAttnBackend, "cpu", torch.arange(8, dtype=torch.int32))
        with _flag("check"):
            backend.init_forward_metadata(fb)
        self.assertFalse(backend.forward_metadata.has_mamba_track_mask)
        self.assertIsNone(backend.forward_metadata.mamba_track_rows)
        self.assertIsNone(backend.forward_metadata.track_conv_indices)

    def test_scheduler_gather_matches_item(self):
        rng = random.Random(5)
        for _ in range(100):
            bs = rng.randint(1, 16)
            buffers = [
                torch.tensor([rng.randint(0, 999), rng.choice([-1, rng.randint(0, 999)])])
                for _ in range(bs)
            ]
            positions = [rng.randint(0, 1) for _ in range(bs)]
            got = track_host.gather_track_slots(buffers, positions, "cpu")
            self.assertEqual(got.dtype, torch.int64)
            self.assertEqual(got.tolist(), [b[p].item() for b, p in zip(buffers, positions)])


class TestTrackPlanFailsLoud(_ChunkSizeMixin, unittest.TestCase):
    def test_plain_extend_without_mirrors_raises(self):
        fb = _forward_batch([300], [0], [True], [300], "cpu")
        fb.mamba_extend_track_mask_cpu = None
        backend = _backend(GDNAttnBackend, "cpu", torch.arange(8, dtype=torch.int32))
        with _flag("1"), self.assertRaisesRegex(RuntimeError, "no host mirrors"):
            backend.init_forward_metadata(fb)

    def test_mirror_length_mismatch_raises(self):
        fb = _forward_batch([300, 300], [0, 0], [True, True], [300, 300], "cpu")
        fb.mamba_extend_track_seqlens_cpu = [300]
        backend = _backend(GDNAttnBackend, "cpu", torch.arange(8, dtype=torch.int32))
        with _flag("1"), self.assertRaisesRegex(RuntimeError, "do not cover"):
            backend.init_forward_metadata(fb)

    def test_check_mode_catches_a_wrong_mirror(self):
        # Row 1's device seqlen is off the chunk grid, its mirror on it: check mode
        # must raise instead of tracking from the wrong state source.
        fb = _forward_batch([320, 320], [0, 0], [True, True], [320, 300], "cpu")
        fb.mamba_extend_track_seqlens_cpu = [320, 320]
        backend = _backend(GDNAttnBackend, "cpu", torch.arange(8, dtype=torch.int32))
        with _flag("check"), self.assertRaisesRegex(RuntimeError, "=check"):
            backend.init_forward_metadata(fb)


def _meta_case():
    # Grid 64 == state chunk, so a tracked row's seqlen is prefix + extend (no _force_track_h).
    extend = [300, 64, 1000, 130, 17]
    prefix = [0, 128, 4096, 64, 0]
    mask = [ext >= CHUNK for ext in extend]
    seqlens = [pre + ext if m else -1 for ext, pre, m in zip(extend, prefix, mask)]
    return extend, prefix, mask, seqlens


class TestTrackPlanIsSyncFree(_ChunkSizeMixin, unittest.TestCase):
    """Meta "device" tensors: every host read of device data raises."""

    def _run(self, flag):
        extend, prefix, mask, seqlens = _meta_case()
        fb = _forward_batch(extend, prefix, mask, seqlens, "meta")
        backend = _backend(
            GDNAttnBackend, "meta", torch.arange(64, dtype=torch.int32).to("meta")
        )
        with _flag(flag):
            backend.init_forward_metadata(fb)
        return backend.forward_metadata

    def test_host_path_runs_on_meta(self):
        md = self._run("1")
        self.assertTrue(md.has_mamba_track_mask)
        self.assertEqual(md.mamba_track_rows.device.type, "meta")
        self.assertEqual(tuple(md.mamba_track_rows.shape), (4,))
        self.assertEqual(md.track_ssm_h_src.device.type, "meta")
        # tracked rows 0-3; 300, 1000 and 130 are off the 64 grid, 64 is on it
        self.assertEqual(tuple(md.track_ssm_h_dst.shape), (3,))
        self.assertEqual(tuple(md.track_ssm_final_dst.shape), (1,))
        self.assertEqual(
            md.num_state_checkpoints, sum(ext // CHUNK for ext in _meta_case()[0])
        )

    def test_old_path_synchronizes_positive_control(self):
        with self.assertRaises((RuntimeError, NotImplementedError)):
            self._run("0")

    def test_scheduler_gather_runs_on_meta(self):
        buffers = [torch.tensor([3, 4]).to("meta"), torch.tensor([5, -1]).to("meta")]
        got = track_host.gather_track_slots(buffers, [1, 0], "meta")
        self.assertEqual(tuple(got.shape), (2,))
        self.assertEqual(got.dtype, torch.int64)


_GUARDED = ("item", "tolist", "cpu", "numpy", "nonzero", "__bool__", "__int__", "__index__")


@contextmanager
def _no_host_reads():
    def make(name):
        def guarded(self, *args, **kwargs):
            raise AssertionError(f"Tensor.{name} on the host path")

        return guarded

    patches = [patch.object(torch.Tensor, name, make(name)) for name in _GUARDED]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


class TestTrackPlanPythonGuard(_ChunkSizeMixin, unittest.TestCase):
    """Python-level guard on CPU tensors (catches .item()/bool()/... the meta test would also catch)."""

    def _run(self, flag):
        extend, prefix, mask, seqlens = _meta_case()
        fb = _forward_batch(extend, prefix, mask, seqlens, "cpu")
        backend = _backend(GDNAttnBackend, "cpu", torch.arange(64, dtype=torch.int32))
        with _flag(flag), _no_host_reads():
            backend.init_forward_metadata(fb)

    def test_host_path_passes_guard(self):
        self._run("1")

    def test_old_path_trips_guard_positive_control(self):
        with self.assertRaisesRegex(AssertionError, "on the host path"):
            self._run("0")


def _scheduler_req(buffer: torch.Tensor, *, prefix_len=0, extend_len=300):
    server_args = ServerArgs(model_path="dummy", page_size=CHUNK)
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)
    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1] * (prefix_len + extend_len)),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
    req.set_extend_range(prefix_len, prefix_len + extend_len)
    req.kv.mamba_ping_pong_track_buffer = buffer
    req.kv.mamba_next_track_idx = 1
    req.mamba_branching_seqlen = None
    batch = ScheduleBatch(reqs=[req])
    batch.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK)
    )
    batch.tree_cache = SimpleNamespace(page_size=CHUNK)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 0
    return batch, req


class TestSchedulerTrackEntry(unittest.TestCase):
    def test_host_mode_reads_no_slot(self):
        # A meta buffer: .item() would raise, so the entry must not read it.
        batch, req = _scheduler_req(torch.tensor([10, 11]).to("meta"))
        with _flag("1"):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        self.assertIsNone(entry.track_index)
        self.assertEqual(entry.track_position, 1)  # before the swap
        self.assertEqual(req.kv.mamba_next_track_idx, 0)  # swapped
        self.assertTrue(entry.track_mask)

    def test_default_mode_is_unchanged(self):
        batch, req = _scheduler_req(torch.tensor([10, 11]))
        with _flag("0"):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        self.assertEqual(entry.track_index, 11)
        self.assertEqual(entry.track_position, 1)

    def test_default_mode_reads_the_slot_positive_control(self):
        batch, req = _scheduler_req(torch.tensor([10, 11]).to("meta"))
        with _flag("0"), self.assertRaises((RuntimeError, NotImplementedError)):
            batch._mamba_radix_cache_v2_req_prepare_for_extend(req)


class TestPlannerArithmetic(unittest.TestCase):
    def test_python_floor_matches_torch_on_negatives(self):
        # Unmasked rows carry seqlen -1; the planner never reads them, but the
        # floor semantics it relies on must match torch's for any int.
        for a in range(-300, 300, 7):
            for b in (1, 64, 256):
                self.assertEqual(a // b, int(torch.tensor(a) // b))
                self.assertEqual(a % b, int(torch.tensor(a) % b))

    def test_upload_segments_are_32_byte_aligned(self):
        segs = [[1], [2, 3, 4, 5, 6], [], [7, 8]]
        views = track_host.upload_int64(segs, "cpu")
        for seg, view in zip(segs, views):
            self.assertEqual(view.tolist(), seg)
            self.assertEqual(view.dtype, torch.int64)
            self.assertTrue(view.is_contiguous())
            self.assertEqual(view.storage_offset() * view.element_size() % 32, 0)


if __name__ == "__main__":
    unittest.main()
