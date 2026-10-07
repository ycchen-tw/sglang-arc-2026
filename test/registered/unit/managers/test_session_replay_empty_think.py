"""arc-forge session replay (--session-replay-empty-think), CPU only.

A previous assistant turn whose thinking the model left empty in a non-canonical
way -- <think>\\n</think>\\n\\n before the answer, or <think>\\n straight into it --
is re-rendered by the client as <think>\\n\\n</think>\\n\\n. replay_ids treats the
two as the same message and gives the model its sampled ids back; anything else
(non-empty reasoning, a different answer) still stops replay as text_differs.

A toy tokenizer stands in for the real one: every id is one string, encode is a
greedy longest match. The arc-forge repo tests the same rule with the Qwen3.8
tokenizer through the whole ledger (work/tito/tests).
"""

import unittest
from array import array
from types import SimpleNamespace

import numpy as np

from sglang.srt.managers import session_ledger as sl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = {i: chr(ord("a") + i) for i in range(26)}
VOCAB.update({30: " ", 31: "\n", 32: "\n\n", 33: "ab", 40: "<think>", 41: "</think>", 42: "<|im_start|>",
              43: "<|im_end|>", 44: "assistant", 45: "<tool_call>", 46: "</tool_call>", 47: "<|endoftext|>",
              48: "user", 49: "�"})
TOKEN = {v: k for k, v in VOCAB.items()}
END, EOT = TOKEN["<|im_end|>"], TOKEN["<|endoftext|>"]
THINK, UNTHINK, NL, NL2 = TOKEN["<think>"], TOKEN["</think>"], TOKEN["\n"], TOKEN["\n\n"]
CALL, UNCALL = TOKEN["<tool_call>"], TOKEN["</tool_call>"]
STOP = np.array([END, EOT], dtype=np.int64)
BIG = 1 << 20


class ToyTokenizer:
    def convert_tokens_to_ids(self, name):
        return TOKEN.get(name)

    def convert_ids_to_tokens(self, t):
        return VOCAB.get(t)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(VOCAB[int(t)] for t in ids)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        names = sorted(TOKEN, key=len, reverse=True)
        while i < len(text):
            name = next(n for n in names if text.startswith(n, i))
            out.append(TOKEN[name])
            i += len(name)
        return out


TOK = ToyTokenizer()
DECODE = TOK.decode
ET = sl.EmptyThink.from_tokenizer(TOK, DECODE)


def enc(text):
    return TOK.encode(text)


def req(ids, **kw):
    base = dict(rid="r", session_id="s", input_ids=array("q", ids), input_embeds=None, mm_inputs=None,
                positional_embed_overrides=None, token_type_ids=None, multi_item_delimiter_indices=None,
                return_routed_experts=False, routed_experts_start_len=0, return_logprob=False,
                logprob_start_len=-1, need_wait_for_mm_inputs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def replay(record, ids, empty_think=ET, **kw):
    r = req(ids, **kw)
    res = sl.replay_ids(r, np.asarray(record, dtype=np.int64), STOP, END, DECODE, BIG, empty_think)
    return res, list(r.input_ids), r


PROMPT = enc("<|im_start|>user\nfix it<|im_end|>\n<|im_start|>assistant\n<think>\n")  # the generation prompt
TOOL = enc("<tool_call>\nls\n</tool_call>")
NEXT = enc("\n<|im_start|>user\nok<|im_end|>\n<|im_start|>assistant\n<think>\n")
CANON = enc("<think>\n\n</think>\n\n")


def turn_of(prompt_tail_and_gen):
    """The record: the prompt (ends in <think>\\n) plus what the model sampled, closed by <|im_end|>."""
    return PROMPT + prompt_tail_and_gen + [END]


def client(answer_ids):
    """What the client sends next: the previous turn rendered canonically (empty reasoning), then a new turn."""
    return PROMPT[:-2] + CANON + answer_ids + [END] + NEXT


class EmptyThinkTest(unittest.TestCase):
    def test_from_tokenizer(self):
        self.assertEqual(ET.canonical, (THINK, NL2, UNTHINK, NL2))
        self.assertEqual(ET.header, (TOKEN["<|im_start|>"], TOKEN["assistant"], NL))
        self.assertTrue(ET.is_space(NL) and ET.is_space(NL2) and ET.is_space(30))
        self.assertFalse(ET.is_space(THINK) or ET.is_space(0) or ET.is_space(49))

        class NoThink(ToyTokenizer):
            def convert_tokens_to_ids(self, name):
                return None if name == "<think>" else TOKEN.get(name)

        with self.assertRaisesRegex(ValueError, "no single <think> token"):
            sl.EmptyThink.from_tokenizer(NoThink(), DECODE)

    def test_think_then_unthink_is_replayed(self):
        """Shape 1: sampled <think>\\n</think>\\n\\n<tool_call>..., sent <think>\\n\\n</think>\\n\\n<tool_call>...:
        the model's ids come back, the request extends the record."""
        record = turn_of([UNTHINK, NL2] + TOOL)
        sent = client(TOOL)
        self.assertNotEqual(DECODE(sent[: len(record)]), DECODE(record))
        res, got, _ = replay(record, sent)
        self.assertEqual(got, record + NEXT)
        self.assertIsNone(res.skip)
        start = len(PROMPT) - 2 - 3 - 1  # the message starts after the user's <|im_end|>
        self.assertEqual(res.spans, [{"s": start, "lcp": len(PROMPT) - 1, "rec_end": len(record) - 1,
                                      "key_end": len(record) - 1, "delta": 0, "kind": "empty_think"}])

    def test_no_unthink_at_all_is_replayed(self):
        """Shape 2: sampled <think>\\n<tool_call>... (no </think>), sent canonically: two ids fewer."""
        record = turn_of(TOOL)
        res, got, _ = replay(record, client(TOOL))
        self.assertEqual(got, record + NEXT)
        self.assertIsNone(res.skip)
        self.assertEqual([(s["kind"], s["delta"]) for s in res.spans], [("empty_think", -2)])

    def test_any_whitespace_around_the_markers(self):
        for sampled, kind in (([NL, UNTHINK, NL2] + TOOL, None),  # the canonical text in other ids: plain replay
                              ([UNTHINK] + TOOL, "empty_think"),  # </think> with no whitespace after it
                              ([NL, NL, UNTHINK] + TOOL, "empty_think"),
                              ([30, UNTHINK, NL, NL] + TOOL, "empty_think"),
                              ([NL, NL2] + TOOL, "empty_think")):  # whitespace only, then straight into the answer
            record = turn_of(sampled)
            res, got, _ = replay(record, client(TOOL))
            self.assertEqual(got, record + NEXT, sampled)
            self.assertIsNone(res.skip)
            self.assertEqual([s.get("kind") for s in res.spans], [kind], sampled)

    def test_the_answer_may_differ_only_in_tokenization(self):
        """After the block the texts must agree; a non-canonical split there is replayed with it."""
        record = turn_of([UNTHINK, NL2, TOKEN["ab"], 2])
        res, got, _ = replay(record, client([0, 1, 2]))
        self.assertEqual(got, record + NEXT)
        self.assertEqual([s["kind"] for s in res.spans], ["empty_think"])

    def test_nonempty_reasoning_still_differs(self):
        sent_variants = {
            "rendered with its reasoning": PROMPT + [0, 1, NL, UNTHINK, NL2] + TOOL + [END] + NEXT,
            "reasoning dropped": client(TOOL),
            "reasoning moved into the answer": client([0, 1, UNTHINK, NL2] + TOOL),
        }
        record = turn_of([0, 1, UNTHINK, NL2] + TOOL)  # <think>\nab</think>\n\n<tool_call>...
        for name, sent in sent_variants.items():
            res, got, _ = replay(record, sent)
            self.assertEqual((res.skip, res.spans, got), ("text_differs", [], sent), name)

    def test_a_different_answer_still_splits(self):
        """The block matches, the rest of the message does not: nothing is replaced, not even the block."""
        record = turn_of([UNTHINK, NL2] + enc("<tool_call>\nls\n</tool_call>"))
        sent = client(enc("<tool_call>\npwd\n</tool_call>"))
        res, got, _ = replay(record, sent)
        self.assertEqual((res.skip, res.spans, got), ("text_differs", [], sent))
        record = turn_of(TOOL)  # shape 2, extra text after the call in the client's rendering
        sent = client(TOOL + [0])
        res, got, _ = replay(record, sent)
        self.assertEqual((res.skip, res.spans, got), ("text_differs", [], sent))

    def test_off_by_default_and_only_at_an_assistant_turn(self):
        record = turn_of([UNTHINK, NL2] + TOOL)
        res, got, _ = replay(record, client(TOOL), empty_think=None)
        self.assertEqual((res.skip, res.spans), ("text_differs", []))
        # the same block in a user message (no <|im_start|>assistant\n before it) is not a thinking block
        user = enc("<|im_start|>user\n")
        record = user + [THINK, NL, UNTHINK, NL2, 0, END]
        sent = user + list(CANON) + [0, END] + NEXT
        res, got, _ = replay(record, sent)
        self.assertEqual((res.skip, res.spans, got), ("text_differs", [], sent))

    def test_divergence_in_the_answer_is_not_this_rule(self):
        """The record's block is canonical; the request differs later in the answer: plain text_differs."""
        record = PROMPT[:-2] + CANON + [0, 1, END]
        sent = PROMPT[:-2] + CANON + [0, 2, END] + NEXT
        res, got, _ = replay(record, sent)
        self.assertEqual((res.skip, res.spans), ("text_differs", []))

    def test_keeps_matching_after_the_block(self):
        """Two earlier assistant turns: the first with sampled empty thinking, the second with a non-canonical
        split; both come back in one request, and logprob_start_len moves with both deltas."""
        turn1 = turn_of(TOOL)  # shape 2: two ids fewer when replayed
        user2 = enc("\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n")
        record = turn1 + user2 + [TOKEN["ab"], END]  # "ab" sampled as one id: one id fewer
        sent = PROMPT[:-2] + CANON + TOOL + [END] + user2 + [0, 1, END] + NEXT
        res, got, r = replay(record, sent, return_logprob=True, logprob_start_len=len(sent) - 3)
        self.assertEqual(got, record + NEXT)
        self.assertIsNone(res.skip)
        self.assertEqual([(s.get("kind"), s["delta"]) for s in res.spans], [("empty_think", -2), (None, -1)])
        self.assertEqual(r.logprob_start_len, len(sent) - 3 - 3)

    def test_replacement_character_is_not_equivalent(self):
        record = turn_of([UNTHINK, NL2, 49])
        res, _, _ = replay(record, client([49]))
        self.assertEqual(res.skip, "text_differs")


if __name__ == "__main__":
    unittest.main()
