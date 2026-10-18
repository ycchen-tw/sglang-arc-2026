"""arc-forge own-last-turn replay: the shared decision (managers/replay_own_turn.py), CPU only.

OwnTurn.from_tokenizer derives the think closure from the chat template and refuses what it cannot derive;
own_turn_replacement decides, at a replay's text_differs point, whether the diverging message is the session's
last generated one and builds its replacement. The routes (TITO, HiSparse) have their own test files.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_own_turn_fixtures import (  # noqa: E402
    A, B, DECODE, END, EOS, GEN_START, HEAD, NEXT, NL, NL2, OT, PROMPT, REPLY, SHAPES, SPLIT, START, STOP, THINK, TOK,
    TOKEN, UNTHINK, ToyTokenizer, client, enc, expected, shape)

from sglang.srt.managers import replay_own_turn as rot  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def decide(record, key, gen_start=GEN_START, gen_end=None, finished=True):
    """own_turn_replacement at the first divergence, as a replay loop calls it (gen_end: the record's end)."""
    toks, new = np.asarray(record, dtype=np.int64), np.asarray(key, dtype=np.int64)
    lcp = next(i for i, (x, y) in enumerate(zip(record, key)) if x != y)
    ends = np.flatnonzero(np.isin(toks[:lcp], STOP))
    start = int(ends[-1]) + 1 if ends.size else 0
    key_end = key.index(END, lcp)
    gen_end = len(toks) if gen_end is None else gen_end
    return rot.own_turn_replacement(OT, toks, new, lcp, start, key_end, gen_start, gen_end, STOP, finished=finished), start, key_end


class OwnTurnFromTokenizerTest(unittest.TestCase):
    def test_closure_comes_from_the_template(self):
        self.assertEqual(OT.closure_text, "\n</think>\n\n")
        self.assertEqual(OT.closure, (NL, UNTHINK, NL2))
        self.assertEqual((OT.think_start, OT.think_end, OT.message_end), (THINK, UNTHINK, END))
        self.assertEqual(rot.template_think_closure(TOK, "t"), "\n</think>\n\n")

    def test_refuses_what_it_cannot_derive(self):
        class NoUnthink(ToyTokenizer):
            def convert_tokens_to_ids(self, name):
                return None if name == "</think>" else TOKEN.get(name)

        class OtherClosure(ToyTokenizer):
            closure = "\n\n"

        class NoReasoning(ToyTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

        class NoTemplate(ToyTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                raise ValueError("no chat template")

        class LossyEncode(ToyTokenizer):
            def encode(self, text, add_special_tokens=False):
                return super().encode(text)[:-1]

        for tok, match in ((NoUnthink(), "no single </think> token"), (OtherClosure(), "closes reasoning"),
                           (NoReasoning(), "reasoning_content and content once"), (NoTemplate(), "failed"),
                           (LossyEncode(), "does not decode back")):
            with self.subTest(type(tok).__name__), self.assertRaisesRegex(ValueError, match):
                rot.OwnTurn.from_tokenizer(tok, tok.decode, "flag")
        with self.assertRaisesRegex(ValueError, "needs the tokenizer"):
            rot.OwnTurn.from_tokenizer(None, DECODE, "flag")

    def test_generation_bounds_are_checked(self):
        record, key, _ = shape("think_cut_prefix")
        toks, new, key_end = np.asarray(record), np.asarray(key), key.index(END, GEN_START)
        for bad in (-1, len(record) + 1, True, None, 3.0):
            with self.subTest(gen_start=bad), self.assertRaisesRegex(RuntimeError, "exact start of the session's"):
                rot.own_turn_replacement(OT, toks, new, GEN_START, START, key_end, bad, len(record), STOP, finished=True)
            with self.subTest(gen_end=bad), self.assertRaisesRegex(RuntimeError, "end of the session's last"):
                rot.own_turn_replacement(OT, toks, new, GEN_START, START, key_end, GEN_START, bad, STOP, finished=True)
        with self.assertRaisesRegex(RuntimeError, "exact start"):  # gen_start after gen_end
            rot.own_turn_replacement(OT, toks, new, GEN_START, START, key_end, GEN_START + 2, GEN_START + 1, STOP, finished=True)


class DecisionTest(unittest.TestCase):
    def test_shapes(self):
        for name in SHAPES:
            with self.subTest(name):
                record, key, closure = shape(name)
                d, start, key_end = decide(record, key)
                self.assertIsNone(d.why)
                self.assertEqual(d.rec_end, len(record) - 1 if record[-1] in EOS else len(record))
                self.assertEqual(key[:start] + d.ids.tolist() + key[key_end:], expected(record, closure))
                self.assertEqual(d.fields["closure"], len(closure))
                self.assertEqual(d.fields["client_len"], key_end - start)
                self.assertEqual(d.fields["client_sha256"], rot.client_sha256(np.asarray(key[start:key_end])))

    def test_not_the_last_generation(self):
        record, _, _ = shape("think_cut_prefix")
        d, _, _ = decide(record, client("ab cd"))  # the header differs, before gen_start
        self.assertEqual((d.ids, d.why), (None, "before_generation"))
        older = PROMPT + enc("ab\n</think>\n\nok") + [END]  # an older turn, then the cut one
        two = older + NEXT + record[GEN_START:]
        key = (client("<think>\nab\n</think>\n\nno")  # the client changed the older turn's answer
               + enc("ab cd\n</think>\n\n<|im_end|>") + NEXT)
        d, _, _ = decide(two, key, gen_start=len(older) + len(NEXT))
        self.assertEqual((d.ids, d.why), (None, "earlier_message"))

    def test_inconsistent_bounds_raise(self):
        record, key, _ = shape("think_cut_prefix")
        toks, new = np.asarray(record), np.asarray(key)
        key_end = key.index(END, GEN_START)
        with self.assertRaisesRegex(RuntimeError, "inconsistent message bounds"):
            rot.own_turn_replacement(OT, toks, new, GEN_START, START, key_end - 1, GEN_START, len(record), STOP, finished=True)

    def test_ids_past_the_generation_end(self):
        """A HiSparse record under speculative decoding: the verify that sampled the stop (or reached the length cut)
        accepted draft ids past it, which the client never got. They are not the last message: a divergence there is
        after_generation, and inside the generation the message ends at the stop / the cut, not at the record's end."""
        record, key, closure = shape("malformed_call_stop")
        overshoot = record + enc("\n<|im_start|>assistant")  # draft ids past the <|im_end|>
        d, _, _ = decide(overshoot, key, gen_end=len(record))
        self.assertEqual(d.rec_end, len(record) - 1)
        nxt = expected(record, closure)
        d2, start, key_end = decide(overshoot, nxt, gen_end=len(record))  # the next message diverges in them
        self.assertEqual((d2.ids, d2.why), (None, "after_generation"))
        record, key, closure = shape("tool_call_cut")
        d, start, key_end = decide(record + enc("al"), key, gen_end=len(record))  # past the length cut
        self.assertEqual(d.rec_end, len(record))
        self.assertEqual(key[:start] + d.ids.tolist() + key[key_end:], expected(record, closure))


class EntryTest(unittest.TestCase):
    """What a takeover leaves for later requests, and the abort exclusion."""

    def test_takeover_leaves_an_entry(self):
        for name in SHAPES:
            with self.subTest(name):
                record, key, closure = shape(name)
                d, start, key_end = decide(record, key)
                e = d.entry
                self.assertEqual((e.start, e.client_len, e.client_sha256),
                                 (start, key_end - start, d.fields["client_sha256"]))
                self.assertEqual((e.rep_len, e.rep_sha256), (len(d.ids), rot.client_sha256(d.ids)))
                want = np.asarray(expected(record, closure))
                self.assertTrue(e.held_by(want))
                self.assertEqual(int(want[e.end]), END)  # the client's <|im_end|> follows the replacement
                self.assertFalse(e.held_by(want[: e.end - 1]))

    def test_an_unfinished_generation_is_not_taken_over(self):
        for name in SHAPES:
            with self.subTest(name):
                record, key, _ = shape(name)
                d, _, _ = decide(record, key, finished=False)
                self.assertEqual((d.ids, d.why, d.entry), (None, "aborted", None))
        record, _, _ = shape("think_cut_prefix")
        d, _, _ = decide(record, client("ab cd"), finished=False)  # before the generation: that reason first
        self.assertEqual(d.why, "before_generation")
        with self.assertRaisesRegex(TypeError, "finished must be a bool"):
            decide(record, shape("think_cut_prefix")[1], finished=None)

    def test_a_closure_with_a_stop_token_raises(self):
        record = PROMPT + SPLIT + enc(" cd")
        key = enc(HEAD + "<|im_start|>assistant\n<think>\nab cd<|endoftext|>\n</think>\n\n<|im_end|>") + NEXT
        with self.assertRaisesRegex(RuntimeError, "holds a stop token"):
            decide(record, key)

    def test_kept_entries(self):
        record, key, closure = shape("tool_call_cut")
        d, _, _ = decide(record, key)
        ids = np.asarray(expected(record, closure))
        e = d.entry
        for cut, kept in ((e.end + 1, 1), (e.end, 0), (START, 0), (len(ids), 1)):
            with self.subTest(cut=cut):
                got, dropped = rot.kept_entries({e.start: e}, ids, cut, "t")
                self.assertEqual((len(got), dropped), (kept, 1 - kept))
        bad = ids.copy()
        bad[e.start + 1] = A if bad[e.start + 1] != A else B
        with self.assertRaisesRegex(RuntimeError, "not held"):
            rot.kept_entries({e.start: e}, bad, len(bad), "t")


class StickyTest(unittest.TestCase):
    """sticky_replacement at the request after the takeover (k+1), where the message is an earlier one."""

    def k1(self, name):
        """The record after k (the takeover, then REPLY) and the client's k+1 request; the loop's bounds there."""
        record, key, closure = shape(name)
        d, _, _ = decide(record, key)
        ids_k = expected(record, closure)
        toks = np.asarray(ids_k + REPLY, dtype=np.int64)
        new = np.asarray(key + enc("ok<|im_end|>") + NEXT, dtype=np.int64)  # the client re-renders the same way
        n = min(len(toks), len(new))
        lcp = int(np.flatnonzero(toks[:n] != new[:n])[0])
        ends = np.flatnonzero(np.isin(toks[:lcp], STOP))
        start = int(ends[-1]) + 1 if ends.size else 0
        key_end = int(np.flatnonzero(new[lcp:] == END)[0]) + lcp
        rec_end = int(np.flatnonzero(np.isin(toks[lcp:], STOP))[0]) + lcp
        return d.entry, toks, new, start, key_end, rec_end, len(ids_k)

    def test_the_same_client_ids_get_the_replacement_again(self):
        for name in SHAPES:
            if name == "think_cut_prefix":
                continue  # its replacement's text is the client's: the plain replay takes it, never sticky
            with self.subTest(name):
                e, toks, new, start, key_end, rec_end, gen_start = self.k1(name)
                self.assertEqual(start, e.start)
                d = rot.sticky_replacement(e, toks, new, start, key_end, rec_end, gen_start)
                self.assertFalse(d.drop)
                self.assertEqual(d.ids.tolist(), toks[e.start:e.end].tolist())
                self.assertEqual(d.fields, {"kind": "own_turn", "sticky": True, "closure": 0,
                                            "client_len": e.client_len, "client_sha256": e.client_sha256})

    def test_other_client_ids_drop_the_entry(self):
        e, toks, new, start, key_end, rec_end, gen_start = self.k1("tool_call_cut")
        other = new.copy()
        other[key_end - 1] = A if other[key_end - 1] != A else B  # same length, other ids
        shorter = np.concatenate((new[: key_end - 1], new[key_end:]))  # one id less
        for ids, end in ((other, key_end), (shorter, key_end - 1)):
            d = rot.sticky_replacement(e, toks, ids, start, end, rec_end, gen_start)
            self.assertEqual((d.ids, d.drop), (None, True))

    def test_inconsistencies_raise(self):
        e, toks, new, start, key_end, rec_end, gen_start = self.k1("tool_call_cut")
        bad = toks.copy()
        bad[e.end - 1] = A if bad[e.end - 1] != A else B
        with self.assertRaisesRegex(RuntimeError, "not held"):  # the record changed under the entry
            rot.sticky_replacement(e, bad, new, start, key_end, rec_end, gen_start)
        with self.assertRaisesRegex(RuntimeError, "message ends at"):
            rot.sticky_replacement(e, toks, new, start, key_end, rec_end + 1, gen_start)
        with self.assertRaisesRegex(RuntimeError, "last generation"):
            rot.sticky_replacement(e, toks, new, start, key_end, rec_end, start)
        with self.assertRaisesRegex(RuntimeError, "asked for the message"):
            rot.sticky_replacement(e, toks, new, start + 1, key_end, rec_end, gen_start)


if __name__ == "__main__":
    unittest.main()
