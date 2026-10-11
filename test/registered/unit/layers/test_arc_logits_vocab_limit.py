"""ARC_LOGITS_VOCAB_LIMIT: the padded lm_head rows past the tokenizer get -inf logits."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers import logits_processor as lp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ArcLogitsVocabLimitTestCase(CustomTestCase):
    def test_mask_sets_only_the_padded_columns(self):
        logits = torch.randn(3, 10)
        before = logits.clone()
        out = lp.arc_mask_padded_vocab(logits, 7)
        self.assertIs(out, logits)  # in place: the CUDA-graph logits buffer stays the buffer
        self.assertTrue(torch.equal(out[:, :7], before[:, :7]))
        self.assertTrue(torch.isneginf(out[:, 7:]).all())
        probs = torch.softmax(out, dim=-1)
        self.assertTrue((probs[:, 7:] == 0).all())
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(3)))

    def test_no_limit_or_limit_covering_the_width_is_a_no_op(self):
        logits = torch.randn(2, 5)
        before = logits.clone()
        lp.arc_mask_padded_vocab(logits, None)
        lp.arc_mask_padded_vocab(logits, 5)
        self.assertTrue(torch.equal(logits, before))

    def test_env_parsing(self):
        with mock.patch.dict("os.environ", {"ARC_LOGITS_VOCAB_LIMIT": "248077"}):
            self.assertEqual(lp._arc_logits_vocab_limit(), 248077)
        for raw in ("0", "-3", "1.5", "all"):
            with mock.patch.dict("os.environ", {"ARC_LOGITS_VOCAB_LIMIT": raw}):
                with self.assertRaises(ValueError):
                    lp._arc_logits_vocab_limit()
        with mock.patch.dict("os.environ", {}):
            __import__("os").environ.pop("ARC_LOGITS_VOCAB_LIMIT", None)
            self.assertIsNone(lp._arc_logits_vocab_limit())


if __name__ == "__main__":
    unittest.main()
