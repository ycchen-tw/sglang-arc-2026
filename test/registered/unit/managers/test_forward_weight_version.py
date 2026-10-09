"""CPU regressions for old target results committed after in-place updates."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.weight_versions import (
    compute_weight_version_spans, record_forward_weight_version, record_weight_version_events,
)


def make_processor():
    metrics = MagicMock(num_generated_tokens=0, forward_ct_decode=0)
    return SchedulerBatchResultProcessor(
        is_generation=True, disaggregation_mode=None, enable_overlap=True,
        enable_overlap_mlx=False, model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(), tree_cache=SimpleNamespace(page_size=1),
        hisparse_coordinator=None, beam_coordinator=MagicMock(), req_to_token_pool=None,
        decode_offload_manager=None, metrics_collector=None, metrics_reporter=metrics,
        draft_worker=None, model_worker=MagicMock(), logprob_result_processor=None,
        output_streamer=MagicMock(), abort_request=MagicMock(),
    )


def labels(req, current, visible=None):
    spans = compute_weight_version_spans(req.weight_version_events, current,
                                         len(req.output_ids) if visible is None else visible)
    return [span.version for span in spans for _ in range(span.end - span.start)]


class TestForwardWeightVersion(unittest.TestCase):
    def setUp(self):
        context = get_context().override_server_args(enable_metrics=False)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_target_launch_captures_version_before_worker_returns(self):
        for spec in (False, True):
            with self.subTest(spec=spec):
                serving = SimpleNamespace(weight_version="0")
                result = GenerationBatchResult()
                algorithm = SimpleNamespace(is_none=lambda: not spec)
                scheduler = MagicMock()
                scheduler.enable_overlap = False
                scheduler.enable_pdmux = False
                scheduler.is_generation = True
                scheduler.forward_ct = 0
                scheduler.scripted_scheduler_hook = None
                scheduler.forward_sleep_time = None
                scheduler.spec_algorithm = algorithm
                scheduler.ps.pp_size = 1
                scheduler._forward_isolation.return_value = nullcontext()
                batch = ScheduleBatch(reqs=[])
                batch.forward_mode = SimpleNamespace(is_prebuilt=lambda: False)
                batch.spec_algorithm = algorithm
                batch.extend_num_tokens = 0

                def forward(*args, **kwargs):
                    # Even a changed current value before return cannot relabel
                    # this result. Real scheduler updates happen after return.
                    serving.weight_version = "1"
                    return result

                scheduler.model_worker.forward_batch_generation.side_effect = forward
                with patch("sglang.srt.managers.scheduler.get_serving", return_value=serving), \
                     patch("sglang.srt.managers.scheduler.resolve_forward_inputs"):
                    Scheduler.run_batch.__wrapped__(scheduler, batch)
                self.assertEqual(result.forward_weight_version, "0")

    def test_overlap_old_result_then_new_result_accept_lengths_eos_and_abort(self):
        # For K4: correction only, partial accept+correction, and full accept+bonus.
        for spec, width in ((False, 1), (True, 1), (True, 2), (True, 4)):
            for terminal in (None, "eos", "abort"):
                with self.subTest(spec=spec, width=width, terminal=terminal):
                    params = SamplingParams(max_new_tokens=64, stop_token_ids=[99])
                    params.normalize(None)
                    req = Req("r", "", [1, 2], params, vocab_size=128)
                    req.output_ids = [3] * 10
                    record_forward_weight_version(req, "0")
                    batch = ScheduleBatch(reqs=[req])
                    batch.spec_algorithm = SimpleNamespace(is_none=lambda: not spec)
                    proc = make_processor()
                    tokens = [20 + i for i in range(width)]
                    if terminal == "eos":
                        tokens[-1] = 99
                    if terminal == "abort":
                        req.finished_reason = FINISH_ABORT()
                    old = GenerationBatchResult(
                        forward_weight_version="0", next_token_ids=torch.tensor(tokens + [0] * (4 - width))
                        if spec else torch.tensor(tokens),
                        accept_lens=torch.tensor([width]) if spec else None,
                        speculative_num_draft_tokens=4 if spec else None,
                    )
                    # The forward is completed, but CPU processing is pending.
                    record_weight_version_events([req], "0")
                    # Multiple updates while paused must not relabel old outputs.
                    record_weight_version_events([req], "1")
                    observed = []
                    proc.output_streamer.stream_output.side_effect = lambda *a, **k: observed.append(labels(req, "2"))
                    with patch.object(SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"):
                        proc.process_batch_result_decode(batch, old)
                    expected = ["0"] * (10 if terminal == "abort" else 10 + width)
                    self.assertEqual(labels(req, "2"), expected)
                    self.assertEqual(observed[-1], expected)  # Correct before streaming, including EOS.
                    if terminal is None:
                        new = GenerationBatchResult(
                            forward_weight_version="2", next_token_ids=torch.tensor([55, 0, 0, 0])
                            if spec else torch.tensor([55]),
                            accept_lens=torch.tensor([1]) if spec else None,
                            speculative_num_draft_tokens=4 if spec else None,
                        )
                        with patch.object(SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"):
                            proc.process_batch_result_decode(batch, new)
                        self.assertEqual(labels(req, "2"), expected + ["2"])
                    # Aborts and speculative EOS clipping expose a prefix only.
                    self.assertEqual(labels(req, "2", visible=9), ["0"] * 9)

    def test_repeated_forwards_coalesce_and_missing_version_fails(self):
        req = SimpleNamespace(output_ids=[], weight_version_events=[])
        for _ in range(100):
            req.output_ids.append(1)
            record_forward_weight_version(req, "0")
        self.assertEqual(len(req.weight_version_events), 1)
        with self.assertRaisesRegex(RuntimeError, "lacks its target"):
            record_forward_weight_version(req, None)


if __name__ == "__main__":
    unittest.main()
