"""ARC_QWEN4_EXP_PREFILL_BCG: Qwen4-Exp text extend under the breakable prefill CUDA graph.

CPU coverage of the pieces that do not need a capture:
  * the opt-in flag, its guard, and the multimodal BCG allowlist it extends;
  * the PLE break against the eager forward's three steps (prepare at the top of the model, layer,
    commit after the last layer), with the bucket padding the replay adds: same layer output on the
    real rows, zeros on the padding, same N-gram history committed;
  * the QSA indexer break: live batch from the forward context, real rows copied, padding -1, and
    every shape / dtype / row-count mismatch raises;
  * the VL body's (hidden, hc_hidden) contract under BCG and its replay unpack;
  * the runner: the replayed pair goes through the unpack hook, the replay batch carries the host
    mirrors (req_pool_indices_cpu, the mamba track mirrors), the capture batch mirrors its static
    track slots, and batches with media stay eager.
Capture / replay itself and bitwise agreement with eager need a GPU:
scripts/arc/qwen4_exp_prefill_bcg_check.py.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs import model_config
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.srt.models import qwen4_exp, qwen4_exp_prefill_bcg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NGRAM = 3
EOS = 7
CHANNELS = 6


@contextmanager
def _piecewise_context(forward_batch):
    with set_tc_piecewise_forward_context(
        forward_batch,
        attention_layers=[],
        quant_config=None,
        moe_layers=[],
        moe_fusions=[],
    ):
        yield


class _FakePLEPool:
    """The req-to-token pool surface _prepare_ple_batch / _commit_ple_batch touch."""

    def __init__(self, req_slots: int):
        g = torch.Generator().manual_seed(1)
        # row 0 is the dummy state slot; request slot r owns state slot r + 1
        self.ngram = torch.randint(0, 1000, (req_slots + 1, NGRAM - 1), generator=g)
        self.state_of_req = torch.arange(req_slots) + 1
        self.ple_window_cache = None

    def get_mamba_indices(self, req_pool_indices):
        return self.state_of_req[req_pool_indices]

    def get_ngram_context(self, state_indices):
        return self.ngram[state_indices]

    def set_ngram_context(self, state_indices, context):
        self.ngram[state_indices] = context.to(self.ngram.dtype)


class _FakePLELayer:
    """Qwen4ExpPLELayer's contract: reads the real rows of the query and the batch's N-gram
    windows, zeroes invalid rows, pads the output to the physical token count."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, forward_batch, batch):
        self.calls.append((forward_batch, batch))
        rows = batch.processed_tokens
        windows = batch.ngram_context.unfold(1, NGRAM, 1)[
            batch.req_indices, batch.token_offsets
        ]
        out = query[:rows] * 2 + windows.sum(-1, keepdim=True).to(query.dtype)
        out = out.masked_fill(~batch.valid_tokens.unsqueeze(-1), 0)
        return qwen4_exp._pad_token_rows(out, batch.physical_tokens)


def _extend_batch(input_ids, extend_lens, req_pool_indices, real_tokens):
    bs = len(extend_lens)
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=input_ids,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        seq_lens=torch.tensor(extend_lens, dtype=torch.int64),
        out_cache_loc=torch.arange(1, input_ids.shape[0] + 1, dtype=torch.int64),
        seq_lens_sum=sum(extend_lens),
        positions=torch.zeros(input_ids.shape[0], dtype=torch.int64),
        seq_lens_cpu=torch.tensor(extend_lens, dtype=torch.int64),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int64),
        extend_prefix_lens=torch.zeros(bs, dtype=torch.int64),
        extend_start_loc=torch.zeros(bs, dtype=torch.int64),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=[0] * bs,
        global_num_token_non_padded_cpu=real_tokens,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        global_forward_mode=ForwardMode.EXTEND,
    )


class TestFlag(CustomTestCase):
    def test_parse_is_strict(self):
        self.assertFalse(qwen4_exp_prefill_bcg._parse("0"))
        self.assertTrue(qwen4_exp_prefill_bcg._parse("1"))
        for bad in ("", "true", "on", "2"):
            with self.assertRaises(ValueError):
                qwen4_exp_prefill_bcg._parse(bad)

    def test_guard_raises_when_off(self):
        with patch.object(qwen4_exp_prefill_bcg, "ENABLED", False):
            with self.assertRaisesRegex(RuntimeError, "ARC_QWEN4_EXP_PREFILL_BCG=1"):
                qwen4_exp_prefill_bcg.require_enabled("unit test")
        with patch.object(qwen4_exp_prefill_bcg, "ENABLED", True):
            qwen4_exp_prefill_bcg.require_enabled("unit test")

    def test_allowlist_only_with_the_flag(self):
        arch = ["Qwen4ExpForConditionalGeneration"]
        with patch.object(qwen4_exp_prefill_bcg, "ENABLED", False):
            self.assertFalse(model_config.is_multimodal_breakable_cuda_graph_supported(arch))
        with patch.object(qwen4_exp_prefill_bcg, "ENABLED", True):
            self.assertTrue(model_config.is_multimodal_breakable_cuda_graph_supported(arch))
            # the native MTP draft is not part of the opt-in
            self.assertFalse(
                model_config.is_multimodal_breakable_cuda_graph_supported(
                    ["Qwen4ExpForCausalLMMTP"]
                )
            )
        # archs already on the list are unaffected either way
        for enabled in (False, True):
            with patch.object(qwen4_exp_prefill_bcg, "ENABLED", enabled):
                self.assertTrue(
                    model_config.is_multimodal_breakable_cuda_graph_supported(
                        ["Qwen3_5MoeForConditionalGeneration"]
                    )
                )


class TestPLEBreak(CustomTestCase):
    def _eager(self, ids, extend_lens, req_pool_indices, query, pool):
        """The eager forward: prepare at the top of the model, the layer, commit at the end."""
        fb = _extend_batch(ids, extend_lens, req_pool_indices, real_tokens=None)
        layer = _FakePLELayer()
        backend = SimpleNamespace(req_to_token_pool=pool)
        with forward_context(ForwardContext(attn_backend=backend)):
            batch = qwen4_exp._prepare_ple_batch(
                fb.input_ids, fb, ngram_size=NGRAM, ngram_eos_token_id=EOS
            )
            out = layer(query, fb, batch)
            qwen4_exp._commit_ple_batch(batch, fb)
        return out

    def _breakable(self, ids, extend_lens, req_pool_indices, query, pool, bucket):
        real = ids.shape[0]
        padded_ids = torch.cat([ids, torch.zeros(bucket - real, dtype=ids.dtype)])
        fb = _extend_batch(padded_ids, extend_lens, req_pool_indices, real_tokens=real)
        g = torch.Generator().manual_seed(2)
        # padding rows of the replayed body hold whatever the graph computed there
        padded_query = torch.cat(
            [query, torch.randn(bucket - real, query.shape[1], generator=g)]
        )
        out = torch.full((bucket, query.shape[1]), float("nan"))
        layer = _FakePLELayer()
        backend = SimpleNamespace(req_to_token_pool=pool)
        with forward_context(ForwardContext(attn_backend=backend)), _piecewise_context(fb):
            # outside a capture eager_on_graph runs the body directly, as in warmup
            qwen4_exp._bcg_ple_with_output(
                layer, padded_query, out, ngram_size=NGRAM, ngram_eos_token_id=EOS
            )
        self.assertEqual(len(layer.calls), 1)
        self.assertIs(layer.calls[0][0], fb)
        return out

    def test_matches_eager_with_bucket_padding(self):
        g = torch.Generator().manual_seed(0)
        for extend_lens, req_pool_indices, bucket in (
            ([5], [2], 8),
            ([5, 3], [3, 1], 12),
            ([1, 6, 2], [0, 4, 2], 9),
            ([4, 4], [1, 2], 8),  # no padding at all
        ):
            with self.subTest(extend_lens=extend_lens, bucket=bucket):
                real = sum(extend_lens)
                ids = torch.randint(10, 500, (real,), generator=g)
                query = torch.randn(real, CHANNELS, generator=g)
                pool_eager, pool_bcg = _FakePLEPool(5), _FakePLEPool(5)
                before = pool_eager.ngram.clone()
                ref = self._eager(ids, extend_lens, req_pool_indices, query, pool_eager)
                out = self._breakable(
                    ids, extend_lens, req_pool_indices, query, pool_bcg, bucket
                )
                torch.testing.assert_close(out[:real], ref, rtol=0, atol=0)
                self.assertTrue(torch.equal(out[real:], torch.zeros_like(out[real:])))
                self.assertTrue(torch.equal(pool_bcg.ngram, pool_eager.ngram))
                touched = pool_eager.state_of_req[req_pool_indices]
                self.assertFalse(torch.equal(pool_eager.ngram[touched], before[touched]))

    def test_raises_on_output_mismatch(self):
        ids = torch.arange(10, 14)
        fb = _extend_batch(ids, [4], [1], real_tokens=4)
        backend = SimpleNamespace(req_to_token_pool=_FakePLEPool(3))
        query = torch.randn(4, CHANNELS)
        for out in (
            torch.empty(4, CHANNELS + 1),
            torch.empty(4, CHANNELS, dtype=torch.float64),
        ):
            with self.subTest(shape=tuple(out.shape), dtype=out.dtype):
                with forward_context(ForwardContext(attn_backend=backend)), _piecewise_context(fb):
                    with self.assertRaisesRegex(RuntimeError, "PLE break"):
                        qwen4_exp._bcg_ple_with_output(
                            _FakePLELayer(), query, out, ngram_size=NGRAM, ngram_eos_token_id=EOS
                        )

    def test_needs_the_forward_context(self):
        with self.assertRaisesRegex(RuntimeError, "outside the prefill CUDA graph"):
            qwen4_exp._ple_with_output_impl(
                _FakePLELayer(),
                torch.zeros(2, CHANNELS),
                torch.zeros(2, CHANNELS),
                ngram_size=NGRAM,
                ngram_eos_token_id=EOS,
            )


class _FakeQSALayer:
    def __init__(self, topk):
        self.topk = topk
        self.seen = None

    def _compute_qsa_topk_indices(self, hidden_states, positions, forward_batch):
        self.seen = (hidden_states, positions, forward_batch)
        return self.topk


class TestQSATopkBreak(CustomTestCase):
    def test_copies_real_rows_and_marks_padding(self):
        topk = torch.randint(-1, 50, (5, 7), dtype=torch.int32)
        layer = _FakeQSALayer(topk)
        fb = SimpleNamespace(global_num_token_non_padded_cpu=5)
        hidden, positions = torch.randn(8, 4), torch.zeros(3, 8, dtype=torch.int64)
        out = torch.full((8, 7), 12345, dtype=torch.int32)
        with _piecewise_context(fb):
            qwen4_exp._bcg_qsa_topk_with_output(layer, hidden, positions, out)
        self.assertIs(layer.seen[0], hidden)
        self.assertIs(layer.seen[1], positions)
        self.assertIs(layer.seen[2], fb)
        self.assertTrue(torch.equal(out[:5], topk))
        self.assertTrue(torch.equal(out[5:], torch.full((3, 7), -1, dtype=torch.int32)))

    def test_mismatches_raise(self):
        cases = {
            "rows != real tokens": (torch.zeros(4, 7, dtype=torch.int32), 5, (8, 7)),
            "more rows than the bucket": (torch.zeros(9, 7, dtype=torch.int32), 9, (8, 7)),
            "width": (torch.zeros(5, 6, dtype=torch.int32), 5, (8, 7)),
            "dtype": (torch.zeros(5, 7, dtype=torch.int64), 5, (8, 7)),
        }
        for name, (topk, real, out_shape) in cases.items():
            with self.subTest(name):
                fb = SimpleNamespace(global_num_token_non_padded_cpu=real)
                out = torch.zeros(out_shape, dtype=torch.int32)
                with _piecewise_context(fb):
                    with self.assertRaisesRegex(RuntimeError, "QSA indexer break"):
                        qwen4_exp._bcg_qsa_topk_with_output(
                            _FakeQSALayer(topk), torch.zeros(8, 4), torch.zeros(8), out
                        )


def _bare_vl_model():
    model = qwen4_exp.Qwen4ExpVLModel.__new__(qwen4_exp.Qwen4ExpVLModel)
    torch.nn.Module.__init__(model)
    model.last_hc_hidden_states = None
    return model


class TestVLBodyContract(CustomTestCase):
    def test_breakable_body_returns_both_streams(self):
        hidden, hc = torch.randn(4, 3), torch.randn(4, 12)
        model = _bare_vl_model()
        fb = SimpleNamespace(input_ids=torch.arange(4))
        with patch.object(
            qwen4_exp.Qwen4ExpModel, "forward", return_value=(hidden, hc)
        ), patch.object(qwen4_exp, "is_in_breakable_cuda_graph", return_value=True):
            out = model.forward(None, torch.zeros(4), fb)
        self.assertIsInstance(out, tuple)
        self.assertIs(out[0], hidden)
        self.assertIs(out[1], hc)
        # eager keeps its contract: hidden out, hc stashed
        with patch.object(
            qwen4_exp.Qwen4ExpModel, "forward", return_value=(hidden, hc)
        ), patch.object(qwen4_exp, "is_in_breakable_cuda_graph", return_value=False):
            out = model.forward(None, torch.zeros(4), fb)
        self.assertIs(out, hidden)
        self.assertIs(model.last_hc_hidden_states, hc)

    def test_unpack(self):
        model = _bare_vl_model()
        hidden, hc = torch.randn(4, 3), torch.randn(4, 12)
        self.assertIs(model.unpack_breakable_body_output((hidden, hc)), hidden)
        self.assertIs(model.last_hc_hidden_states, hc)
        for bad in (hidden, (hidden,), (hidden, hc, hc), (hidden, None)):
            with self.assertRaisesRegex(RuntimeError, r"\(hidden, hc_hidden\)"):
                model.unpack_breakable_body_output(bad)

    def test_model_forward_guard(self):
        model = qwen4_exp.Qwen4ExpModel.__new__(qwen4_exp.Qwen4ExpModel)
        with patch.object(qwen4_exp, "is_in_breakable_cuda_graph", return_value=True), patch.object(
            qwen4_exp_prefill_bcg, "ENABLED", False
        ):
            with self.assertRaisesRegex(RuntimeError, "ARC_QWEN4_EXP_PREFILL_BCG=1"):
                model.forward(None, None, SimpleNamespace(), inputs_embeds=torch.zeros(2, 3))


class _Slot:
    def __init__(self, buffer, axis):
        self.buffer = buffer
        self.axis = axis

    def slice_for(self, bs, num_tokens):
        return self.buffer[:bs] if self.axis == "bs" else self.buffer[:num_tokens]


class _Registry:
    def __init__(self, max_tokens=16, max_bs=4):
        self.slots = {
            "input_ids": _Slot(torch.zeros(max_tokens, dtype=torch.int64), "tokens"),
            "positions": _Slot(torch.zeros(max_tokens, dtype=torch.int64), "tokens"),
            "out_cache_loc": _Slot(torch.zeros(max_tokens, dtype=torch.int64), "tokens"),
            "mamba_track_indices": _Slot(torch.zeros(max_bs, dtype=torch.int64), "bs"),
            "mamba_track_mask": _Slot(torch.zeros(max_bs, dtype=torch.bool), "bs"),
            "mamba_track_seqlens": _Slot(torch.zeros(max_bs, dtype=torch.int32), "bs"),
        }

    def fill_from(self, forward_batch, *, raw_bs, **_kwargs):
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            self.slots[name].buffer[:raw_bs].copy_(getattr(forward_batch, name))

    def has_slot(self, name):
        return name in self.slots

    def get_slot(self, name):
        return self.slots[name]


class TestRunner(CustomTestCase):
    def _runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.capture_num_tokens = [8, 16]
        runner.buffer_registry = _Registry()
        runner.model_runner = SimpleNamespace(attn_tp_sequence_sharded=lambda _: False)
        runner.enable_cp_bcg_capture = False
        runner._is_full_backend = False
        runner.backend = SimpleNamespace()
        runner.has_mha_companion_layers = False
        runner._prefill_static_buffers = None
        runner.static_draft_hidden_states = None
        runner.capture_return_pooled_hidden_states = False
        runner._next_token_logits_buffer = lambda _rows: None
        runner._prefill_logits_buffer_rows = lambda _batch: 1
        runner.metadata_batches = []
        runner._prepare_forward_metadata_for_replay = (
            lambda fb, *_args: runner.metadata_batches.append(fb)
        )
        return runner

    def test_replay_batch_carries_host_mirrors(self):
        runner = self._runner()
        fb = _extend_batch(torch.arange(5), [3, 2], [2, 0], real_tokens=5)
        fb.req_pool_indices_cpu = torch.tensor([2, 0])
        fb.mamba_track_indices = torch.tensor([7, 9])
        fb.mamba_track_mask = torch.tensor([True, False])
        fb.mamba_track_seqlens = torch.tensor([64, 0], dtype=torch.int32)
        fb.mamba_extend_track_mask_cpu = [True, False]
        fb.mamba_extend_track_seqlens_cpu = [64, 0]

        static = runner.load_batch(fb)

        self.assertIs(static.req_pool_indices_cpu, fb.req_pool_indices_cpu)
        self.assertEqual(static.mamba_extend_track_mask_cpu, [True, False])
        self.assertEqual(static.mamba_extend_track_seqlens_cpu, [64, 0])
        # the mirrors describe the static slots the replay batch carries
        self.assertEqual(static.mamba_track_mask.tolist(), [True, False])
        self.assertEqual(static.mamba_track_seqlens.tolist(), [64, 0])
        self.assertEqual(len(static.input_ids), 8)
        self.assertEqual(runner.metadata_batches, [fb])

    def test_capture_batch_mirrors_its_track_slots(self):
        runner = self._runner()
        runner.model_runner = SimpleNamespace(
            attn_tp_sequence_sharded=lambda _: False,
            model_config=SimpleNamespace(context_len=1 << 20),
            attn_backend=None,
        )
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.max_bs = 4
        runner.device = "cpu"
        runner.require_mlp_tp_gather = runner.require_attn_tp_gather = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner._capture_num_token_non_padded = lambda _n: None
        runner._build_capture_spec_info = lambda _n: None
        runner._capture_lora = False
        runner.tbo_plugin = SimpleNamespace(capture_one_batch_size=lambda *a, **k: None)
        mask = runner.buffer_registry.get_slot("mamba_track_mask").buffer
        seqlens = runner.buffer_registry.get_slot("mamba_track_seqlens").buffer

        fb, _ = runner.capture_prepare(8)
        self.assertEqual(fb.mamba_extend_track_mask_cpu, [False])
        self.assertEqual(fb.mamba_extend_track_seqlens_cpu, [0])

        # mirrors are read from the slots, not assumed
        mask[0] = True
        seqlens[0] = 128
        fb, _ = runner.capture_prepare(8)
        self.assertEqual(fb.mamba_extend_track_mask_cpu, [True])
        self.assertEqual(fb.mamba_extend_track_seqlens_cpu, [128])

    def test_media_batches_stay_eager(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [8, 16]
        runner.max_num_tokens = 16

        def batch(with_media):
            return SimpleNamespace(
                batch_size=1,
                input_embeds=None,
                replace_embeds=None,
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                global_num_tokens_cpu=None,
                return_logprob=False,
                input_ids=list(range(8)),
                extend_prefix_lens_cpu=[0],
                contains_mm_inputs=lambda: with_media,
            )

        self.assertTrue(runner.can_run_graph(batch(True)))  # other models: unchanged
        runner._qwen4_exp_bcg = True
        self.assertFalse(runner.can_run_graph(batch(True)))
        self.assertTrue(runner.can_run_graph(batch(False)))

    def test_replay_unpacks_the_body_pair(self):
        hidden, hc = torch.randn(8, 3), torch.randn(8, 12)

        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_hc_hidden_states = None

            def forward(self, input_ids, positions, forward_batch, input_embeds=None):
                raise AssertionError("replay must not run the eager body")

            def unpack_breakable_body_output(self, output):
                hidden_states, self.last_hc_hidden_states = output
                return hidden_states

        body = Body()

        class Outer:
            def forward(self, input_ids, positions, forward_batch):
                out = body(None, positions, forward_batch)
                return out, body.last_hc_hidden_states

        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner._input_embeds_arg_idx = 3
        runner.buffer_registry = SimpleNamespace(has_slot=lambda _name: False)
        runner.backend = SimpleNamespace(replay=lambda *_a, **_k: (hidden, hc))
        runner.layer_model = body
        runner.model_runner = SimpleNamespace(model=Outer(), attn_backend=None)
        runner.attention_layers = runner.moe_layers = runner.moe_fusions = []
        runner.mha_companion_layers = runner.dsa_indexers = None
        runner.quant_config = None
        runner._body_output_unpacker = body.unpack_breakable_body_output
        static = SimpleNamespace(input_ids=torch.arange(8), positions=torch.zeros(8))
        fb = SimpleNamespace(mm_input_embeds=None)

        out, stashed = runner._execute_body_capture(fb, static, 8, 5, ShapeKey(size=8))

        self.assertIs(out, hidden)
        self.assertIs(stashed, hc)
        # the eager forward is restored afterwards
        self.assertIs(body.forward.__func__, Body.forward)


if __name__ == "__main__":
    unittest.main()
