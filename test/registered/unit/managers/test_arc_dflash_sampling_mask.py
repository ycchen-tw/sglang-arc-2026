"""arc-forge: sampling supports of DFLASH-verified tokens and their ledger record, CPU only.

spec_sampling_mask_output packs, per opted-in request and block position, the positive support of the verify's
truncated target distribution, the committed token's support-renormalized logprob and a status; the result
processor expands the rows of a speculative step per request (rows_per_request); the session ledger records one
support per sampled token (--session-ledger-sampling-mask). The arc-forge repo runs the same path end to end on a
B200 (work/qwen38fn_rl/probes/topp_mask/).
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.layers.logits_processor import SamplingMaskOutput, SamplingMaskStatus
from sglang.srt.managers import session_ledger as sl
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.speculative.arc_dflash.sampling_mask import spec_sampling_mask_output
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def truncated(rows):
    """(bs, D, V) target probabilities: each row's given {id: weight}, renormalized, zero elsewhere."""
    bs, block = len(rows), len(rows[0])
    out = torch.zeros(bs, block, 12)
    for b, blocks in enumerate(rows):
        for j, kept in enumerate(blocks):
            total = sum(kept.values())
            for t, w in kept.items():
                out[b, j, t] = w / total
    return out


class TestSpecSamplingMaskOutput(unittest.TestCase):
    def test_support_logprob_and_status_per_block_position(self):
        probs = truncated([
            [{3: 0.5, 7: 0.3, 1: 0.2}, {4: 1.0}],
            [{2: 0.6, 5: 0.4}, {0: 0.25, 9: 0.75}],
            [{6: 1.0}, {8: 0.5, 2: 0.5}],
        ])
        tokens = torch.tensor([[7, 4], [2, 0], [6, 11]])        # request 2, position 1: 11 is outside its support
        rows = torch.tensor([0, 2])                              # request 1 did not opt in
        out = spec_sampling_mask_output(probs, tokens, rows, max_tokens=4)
        self.assertEqual(out.rows_per_request, 2)
        self.assertEqual(out.token_ids.shape, (4, 4))
        self.assertEqual(out.lengths.tolist(), [3, 1, 1, 2])
        self.assertEqual(out.token_ids[0, :3].tolist(), [3, 7, 1])   # by weight, descending
        self.assertEqual(sorted(out.token_ids[3, :2].tolist()), [2, 8])
        self.assertAlmostEqual(float(out.selected_logprobs[0]), math.log(0.3), places=6)
        self.assertAlmostEqual(float(out.selected_logprobs[1]), 0.0, places=6)
        self.assertEqual(
            out.statuses.tolist(),
            [SamplingMaskStatus.OK, SamplingMaskStatus.OK, SamplingMaskStatus.OK, SamplingMaskStatus.INVALID],
        )

    def test_overflow_past_the_cap(self):
        probs = truncated([[{0: 1.0, 1: 1.0, 2: 1.0}]])
        out = spec_sampling_mask_output(probs, torch.tensor([[1]]), torch.tensor([0]), max_tokens=2)
        self.assertEqual(out.statuses.tolist(), [SamplingMaskStatus.OVERFLOW])
        self.assertEqual(out.lengths.tolist(), [2])

    def test_shapes_are_checked(self):
        with self.assertRaises(ValueError):
            spec_sampling_mask_output(torch.zeros(2, 3, 5), torch.zeros(2, 2, dtype=torch.long), torch.tensor([0]), 4)
        with self.assertRaises(ValueError):
            spec_sampling_mask_output(torch.zeros(1, 1, 5), torch.zeros(1, 1, dtype=torch.long), torch.tensor([0]), 0)


class TestMaterializeSpecRows(unittest.TestCase):
    def test_rows_expand_per_request_and_position(self):
        output = SimpleNamespace(
            sampling_mask_output=SamplingMaskOutput(
                token_ids=torch.tensor([[5, 6], [7, 0], [8, 9], [1, 0]], dtype=torch.int32),
                lengths=torch.tensor([2, 1, 2, 1], dtype=torch.int32),
                selected_logprobs=torch.tensor([-0.1, 0.0, -0.7, 0.0]),
                statuses=torch.tensor([0, 0, 0, SamplingMaskStatus.INVALID], dtype=torch.int32),
                rows_per_request=2,
            )
        )
        reqs = [SimpleNamespace(return_sampling_mask=True), SimpleNamespace(return_sampling_mask=False),
                SimpleNamespace(return_sampling_mask=True)]
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(reqs, output)
        self.assertEqual(output.next_token_sampling_mask_idx, [[[5, 6], [7]], None, [[8, 9], None]])
        self.assertEqual(output.next_token_sampling_mask_status[0], [0, 0])
        self.assertEqual(output.next_token_sampling_mask_status[2], [0, SamplingMaskStatus.INVALID])
        self.assertAlmostEqual(output.next_token_sampling_logprobs[2][0], -0.7, places=6)
        self.assertIsNone(output.sampling_mask_output)

    def test_one_row_per_request_is_unchanged(self):
        output = SimpleNamespace(
            sampling_mask_output=SamplingMaskOutput(
                token_ids=torch.tensor([[5, 6]], dtype=torch.int32),
                lengths=torch.tensor([2], dtype=torch.int32),
                selected_logprobs=torch.tensor([-0.2]),
                statuses=torch.tensor([0], dtype=torch.int32),
            )
        )
        SchedulerBatchResultProcessor.materialize_sampling_mask_output([SimpleNamespace(return_sampling_mask=True)],
                                                                       output)
        self.assertEqual(output.next_token_sampling_mask_idx, [[5, 6]])
        self.assertEqual(output.next_token_sampling_mask_status, [0])


def unb64(value, dtype):
    import base64

    return np.frombuffer(base64.b64decode(value), dtype=dtype)


class TestLedgerRecord(unittest.TestCase):
    def test_truncated_sampling(self):
        sp = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=sl.TOP_K_ALL)
        self.assertFalse(sl.truncated_sampling(sp))
        self.assertTrue(sl.truncated_sampling(SimpleNamespace(temperature=1.0, top_p=0.95, top_k=sl.TOP_K_ALL)))
        self.assertTrue(sl.truncated_sampling(SimpleNamespace(temperature=1.0, top_p=1.0, top_k=20)))
        self.assertFalse(sl.truncated_sampling(SimpleNamespace(temperature=0.0, top_p=1.0, top_k=1)))
        self.assertTrue(sl.truncated_sampling_dict({"temperature": 1.0, "top_p": 0.95, "top_k": 20}))

    def test_record_round_trip(self):
        gen = np.array([4, 9, 2])
        record, why = sl._sampling_mask_record([[1, 4], [9], [2, 3, 5]], [-0.3, 0.0, -1.2], gen)
        self.assertIsNone(why)
        self.assertEqual(unb64(record["lens"], "<i4").tolist(), [2, 1, 3])
        self.assertEqual(unb64(record["ids"], "<i4").tolist(), [1, 4, 9, 2, 3, 5])
        np.testing.assert_allclose(unb64(record["logprobs"], "<f4"), [-0.3, 0.0, -1.2], rtol=1e-6)

    def test_record_refuses_misaligned_supports(self):
        gen = np.array([4, 9])
        self.assertIn("for 2 ids", sl._sampling_mask_record([[4]], [-0.1], gen)[1])
        self.assertIn("has support", sl._sampling_mask_record([[4], [8]], [-0.1, 0.0], gen)[1])
        self.assertIn("has support", sl._sampling_mask_record([[4], None], [-0.1, 0.0], gen)[1])
        self.assertIn("missing", sl._sampling_mask_record([[4], [9]], [-0.1, None], gen)[1])


if __name__ == "__main__":
    unittest.main()
