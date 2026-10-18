"""Shared pieces of the own-last-turn replay tests (arc-forge docs/plans/replay_own_last_turn.md), CPU only.

Not a test module: test_replay_own_turn.py (the shared decision), test_session_replay_own_turn.py (TITO route)
and mem_cache/test_qsa_session_replay_own_turn.py (HiSparse route) import it. ``RouteCases`` holds the scenarios
every route must pass with the same ids. The shapes are the f7 ones (ser-sync, 13 codex rewinds): (a) a tool
call cut by the token budget, re-rendered with empty arguments; (b) a malformed call (normal stop) normalized
by the parser; (c) thinking cut by the budget, closed by the template -- with the cut text a prefix of the
client's (closure = the client's suffix) or not (the template's think closure). Negatives: flag off, a
resend, a divergence before the generation, an older message, the existing blockers.

A toy tokenizer stands in for the real one (every id one string, encode a greedy longest match, a tiny Qwen3.8
chat template); arc-forge runs the 13 real f7 cases with the Qwen3.8 tokenizer.
"""

from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers import replay_own_turn as rot
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalProcessorOutput

VOCAB = {i: chr(ord("a") + i) for i in range(26)}
VOCAB.update({30: " ", 31: "\n", 32: "\n\n", 33: "ab", 40: "<think>", 41: "</think>", 42: "<|im_start|>",
              43: "<|im_end|>", 44: "assistant", 45: "<tool_call>", 46: "</tool_call>", 47: "<|endoftext|>",
              48: "user", 49: "�", 50: "<function=", 51: ">", 52: "</function>", 53: "<parameter=",
              54: "</parameter>", 55: "<|vision_start|>", 56: "<|image_pad|>", 57: "<|vision_end|>", 58: "-"})
TOKEN = {v: k for k, v in VOCAB.items()}
END, EOT, THINK, UNTHINK = TOKEN["<|im_end|>"], TOKEN["<|endoftext|>"], TOKEN["<think>"], TOKEN["</think>"]
NL, NL2, A, B, AB, FFFD = TOKEN["\n"], TOKEN["\n\n"], TOKEN["a"], TOKEN["b"], TOKEN["ab"], TOKEN["�"]
VS, PAD, VE = TOKEN["<|vision_start|>"], TOKEN["<|image_pad|>"], TOKEN["<|vision_end|>"]
EOS = [END, EOT]
STOP = np.array(sorted(EOS), dtype=np.int64)
BIG = 1 << 20


class ToyTokenizer:
    """Qwen3.8's shape in miniature, chat template included (assistant: <think>\\n{reasoning|trim}\\n</think>\\n\\n)."""

    closure = "\n</think>\n\n"

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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is False
        out = ""
        for m in messages:
            if m["role"] == "assistant":
                out += ("<|im_start|>assistant\n<think>\n" + m.get("reasoning_content", "").strip() + self.closure
                        + m["content"] + "<|im_end|>\n")
            else:
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return out


TOK = ToyTokenizer()
DECODE = TOK.decode
OT = rot.OwnTurn.from_tokenizer(TOK, DECODE, "test")


def enc(text):
    return TOK.encode(text)


HEAD = "<|im_start|>user\nfix it<|im_end|>\n"
PROMPT = enc(HEAD + "<|im_start|>assistant\n<think>\n")  # the last request's input: its generation starts here
GEN_START = len(PROMPT)
START = PROMPT.index(END) + 1  # the message starts after the user's <|im_end|>
NEXT = enc("\n<|im_start|>user\nok<|im_end|>\n<|im_start|>assistant\n<think>\n")
CLOSURE = list(OT.closure)


def client(assistant_text):
    """What the client sends next: the conversation re-rendered from text, canonically tokenized."""
    return enc(HEAD + "<|im_start|>assistant\n" + assistant_text + "<|im_end|>") + NEXT


# the model's ids after the generation prompt: "a", "b" where the canonical tokenization has "ab", so the client's
# re-render diverges right there -- the first non-canonical split, long before the cut in real data
SPLIT = [A, B]
SHAPES = {
    # (c) thinking cut by the budget; the template trims and closes it: the cut text is a prefix of the client's
    "think_cut_prefix": (SPLIT + enc(" cd"), "<think>\nab cd\n</think>\n\n", CLOSURE),
    # (c) the same with a trailing blank line the template's |trim drops: not a prefix, the think block is open
    "think_cut_not_prefix": (SPLIT + enc(" cd") + [NL2], "<think>\nab cd\n</think>\n\n", CLOSURE),
    # (a) cut inside a tool call's argument; the parser emits the call with empty arguments
    "tool_call_cut": (SPLIT + enc("\n</think>\n\n<tool_call>\n<function=f>\n<parameter=c>\nls -"),
                      "<think>\nab\n</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>", []),
    # (b) a malformed call that stopped normally; the parser rewrote it
    "malformed_call_stop": (SPLIT + enc("\n</think>\n\n<tool_call>\n<function=zz>\n<parameter=q>\nv\n</parameter>\n"
                                        "</function>\n</tool_call>") + [END],
                            "<think>\nab\n</think>\n\n<tool_call>\n<function=zz>\n</function>\n</tool_call>", []),
    # the same, stopped on the other eos token: the client's <|im_end|> stands in for it, as today
    "malformed_call_eot": (SPLIT + enc("\n</think>\n\n<tool_call>\n<function=zz>\n<parameter=q>\nv\n</parameter>\n"
                                       "</function>\n</tool_call>") + [EOT],
                           "<think>\nab\n</think>\n\n<tool_call>\n<function=zz>\n</function>\n</tool_call>", []),
    # cut inside a UTF-8 character: the record decodes to U+FFFD there, the client dropped it
    "think_cut_mid_char": (SPLIT + enc(" cd") + [FFFD], "<think>\nab cd\n</think>\n\n", CLOSURE),
    # cut after the answer's trailing blank line, which the template's content|trim drops (no split before it):
    # the client's <|im_end|> is the divergence itself (key_end == lcp)
    "answer_cut_trimmed": (enc("ab\n</think>\n\nok") + [NL2], "<think>\nab\n</think>\n\nok", []),
}


def shape(name):
    gen, rendered, closure = SHAPES[name]
    record = PROMPT + gen
    return record, client(rendered), closure


def expected(record, closure):
    """The request own-turn gives: the record's message, its closure, then the client's <|im_end|> on."""
    if record[-1] in EOS:
        return record[:-1] + [END] + NEXT
    return record + closure + [END] + NEXT


# ----------------------------------------------------------------------------- sessions (sticky)

USER = "\n<|im_start|>user\nok<|im_end|>\n"
ASK = "<|im_start|>assistant\n<think>\n"
REPLY = enc("ok") + [END]  # the model's reply to every request after the cut (a normal stop)
REPLY_TEXT = "<think>\nok"  # how the client renders it: the same text


class ToyChat:
    """The client: the conversation as text, re-rendered and canonically tokenized on every request."""

    def __init__(self):
        self.text = HEAD

    def request(self):
        return enc(self.text + ASK)

    def answered(self, rendered):
        self.text += "<|im_start|>assistant\n" + rendered + "<|im_end|>" + USER


def finish_of(gen):
    return "stop" if gen[-1] in EOS else "length"


class SessionCases:
    """Mixin: own-turn and its sticky entries over a session's requests, on one route. A subclass gives
    ``session(own=True)``, an object with ``turn(rid, key, gen, finish) -> SimpleNamespace(ids, spans, skip,
    own_skip)``, ``entries()`` (start -> OwnTurnEntry the route keeps with the record) and ``corrupt(pos)``
    (changes the record's id at pos, as only a bug could)."""

    def session(self, own=True):
        raise NotImplementedError

    def cut_then_replies(self, name, n, sess=None):
        """The cut turn (the shape's generation), then n requests each answered with REPLY. Returns the session,
        the ids each request got, and what it returned."""
        sess = self.session() if sess is None else sess
        chat = ToyChat()
        record, _, closure = shape(name)
        gen = record[GEN_START:]
        sess.turn("r0", chat.request(), gen, finish_of(gen))
        chat.answered(SHAPES[name][1])
        outs, ids = [], []
        for i in range(n):
            out = sess.turn(f"r{i + 1}", chat.request(), REPLY, "stop")
            outs.append(out)
            ids.append(out.ids)
            chat.answered(REPLY_TEXT)
        return sess, record, closure, ids, outs, chat

    def test_sticky_sequences(self):
        """k / k+1 / k+2 for every shape: the cut-following request takes the message over, every later one gets
        the same ids for it again (sticky where the texts differ, plain replay where they agree) and extends."""
        for name in SHAPES:
            with self.subTest(name):
                sess, record, closure, ids, outs, _ = self.cut_then_replies(name, 3)
                self.assertEqual(ids[0], expected(record, closure))
                self.assertEqual([s.get("kind") for s in outs[0].spans], ["own_turn"])
                self.assertNotIn("sticky", outs[0].spans[0])
                (entry,) = self.entries_of(sess)
                for i in (1, 2):
                    self.assertEqual(ids[i], ids[i - 1] + REPLY + NEXT)  # extends: the record, then the new turn
                    self.assertIsNone(outs[i].skip)
                    stable = name == "think_cut_prefix"  # the replacement's text is the client's: plain replay
                    sticky = [s for s in outs[i].spans if s.get("sticky")]
                    if stable:
                        self.assertEqual(sticky, [])
                    else:
                        (span,) = sticky
                        self.assertEqual((span["s"], span["rec_end"], span["closure"], span["kind"]),
                                         (START, entry.end, 0, "own_turn"))
                        self.assertEqual(span["delta"], entry.rep_len - entry.client_len)
                        self.assertEqual((span["client_len"], span["client_sha256"]),
                                         (entry.client_len, entry.client_sha256))
                    self.assertEqual(self.entries_of(sess), [entry])

    def test_several_entries_in_one_session(self):
        """Two cut turns in a row: the second request after them gets both messages back, two sticky spans."""
        sess = self.session()
        chat = ToyChat()
        gen = SHAPES["tool_call_cut"][0]
        sess.turn("r0", chat.request(), gen, "length")
        chat.answered(SHAPES["tool_call_cut"][1])
        k = sess.turn("r1", chat.request(), SHAPES["think_cut_not_prefix"][0], "length")
        self.assertEqual([s.get("sticky", False) for s in k.spans if s.get("kind") == "own_turn"], [False])
        chat.answered(SHAPES["think_cut_not_prefix"][1])
        k1 = sess.turn("r2", chat.request(), REPLY, "stop")
        self.assertEqual([(s.get("kind"), s.get("sticky", False)) for s in k1.spans],
                         [("own_turn", True), ("own_turn", False)])
        self.assertEqual(len(self.entries_of(sess)), 2)
        chat.answered(REPLY_TEXT)
        k2 = sess.turn("r3", chat.request(), REPLY, "stop")
        self.assertEqual([(s.get("kind"), s.get("sticky", False)) for s in k2.spans],
                         [("own_turn", True), ("own_turn", True)])
        self.assertEqual(k2.ids, k1.ids + REPLY + NEXT)
        self.assertIsNone(k2.skip)

    def test_client_changing_the_message_drops_its_entry(self):
        """The client sends other ids for the taken-over message: no replacement, text_differs, the entry goes."""
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        (entry,) = self.entries_of(sess)
        chat.text = chat.text.replace("<function=f>\n</function>", "<function=g>\n</function>")
        out = sess.turn("r9", chat.request(), REPLY, "stop")
        self.assertEqual((out.skip, out.own_skip), ("text_differs", "earlier_message"))
        self.assertEqual([s for s in out.spans if s.get("kind") == "own_turn"], [])
        self.assertEqual(self.entries_of(sess), [])
        self.assertEqual(self.dropped(sess, "client"), 1)

    def test_record_changed_under_an_entry_raises(self):
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        (entry,) = self.entries_of(sess)
        sess.corrupt(entry.end - 1)
        with self.assertRaisesRegex(RuntimeError, "not held"):
            sess.turn("r9", chat.request(), REPLY, "stop")

    def test_flag_off_keeps_no_entries(self):
        sess = self.session(own=False)
        sess, record, closure, ids, outs, _ = self.cut_then_replies("tool_call_cut", 2, sess=sess)
        self.assertEqual(self.entries_of(sess), [])
        self.assertEqual(outs[0].skip, "text_differs")

    def entries_of(self, sess):
        return sorted(sess.entries().values(), key=lambda e: e.start)

    def dropped(self, sess, why):
        return sess.counters()[f"replay_own_turn_sticky_drop_{why}"]


def recv(ids, **kw):
    base = dict(rid="r", session_id="s", input_ids=array("q", ids), input_embeds=None, mm_inputs=None,
                positional_embed_overrides=None, token_type_ids=None, multi_item_delimiter_indices=None,
                return_routed_experts=False, routed_experts_start_len=0, return_logprob=False,
                logprob_start_len=-1, need_wait_for_mm_inputs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def mrope(ids, image=None):
    """Toy M-RoPE: text rows equal and counting up; an image run's third row alternates."""
    pos = torch.arange(len(ids), dtype=torch.int64)
    rows = torch.stack([pos, pos, pos])
    if image is not None:
        a, b = image
        rows[2, a : b + 1] += torch.arange(b - a + 1) % 2
    return rows


def mm_output(ids, padded=False):
    runs = [i for i, t in enumerate(ids) if t == PAD]
    image = (runs[0], runs[-1]) if runs else None
    rows = mrope(ids, image)
    return MultimodalProcessorOutput(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, offsets=[image])] if image else [],
        input_ids=list(ids),
        padded_input_ids=[-7 if t == PAD else t for t in ids] if padded else None,
        im_token_id=PAD, im_start_id=VS, im_end_id=VE,
        mrope_positions=rows,
        mrope_position_delta=torch.tensor([[int(rows.max()) + 1 - len(ids)]]),
    )


class RouteCases:
    """Mixin: every test runs on one route (TITO or HiSparse); the subclasses pick it."""

    route = None

    def go(self, *a, **kw):
        return type(self).route(*a, **kw)

    # --------------------------------------------------------------- the shapes, flag on

    def test_shapes_become_extensions(self):
        for name in SHAPES:
            with self.subTest(name):
                record, key, closure = shape(name)
                self.assertNotEqual(key[: len(record)], record)
                got = self.go(record, key)
                want = expected(record, closure)
                self.assertEqual(got.ids, want)
                self.assertIsNone(got.skip)
                self.assertIsNone(got.own_skip)
                # the request extends the record through its last message
                last = len(record) - 1 if record[-1] in EOS else len(record)
                self.assertEqual(got.ids[:last], record[:last])
                (span,) = got.spans
                lcp = next(i for i, (x, y) in enumerate(zip(record, key)) if x != y)
                self.assertGreaterEqual(lcp, GEN_START)
                key_end = key.index(END, lcp)
                rep = last - START + len(closure)
                self.assertEqual(span, {"s": START, "lcp": lcp, "rec_end": last, "key_end": key_end,
                                        "delta": rep - (key_end - START), "kind": "own_turn",
                                        "closure": len(closure), "client_len": key_end - START,
                                        "client_sha256": rot.client_sha256(np.asarray(key[START:key_end]))})

    def test_cut_text_prefix_takes_the_clients_suffix(self):
        """The closure after a prefix cut is the client's own text after the cut, tokenized on its own."""
        record = PROMPT + SPLIT + enc(" cd\n")  # ends in "\n": the client's suffix is "</think>\n\n"
        key = client("<think>\nab cd\n</think>\n\n")
        got = self.go(record, key)
        self.assertEqual(got.ids, record + [UNTHINK, NL2, END] + NEXT)
        self.assertEqual(got.spans[0]["closure"], 2)

    def test_fffd_spelled_by_both_is_a_prefix(self):
        """The record and the client both spell U+FFFD at the cut: the cut text is a prefix of the client's."""
        record = PROMPT + SPLIT + [FFFD]
        key = enc(HEAD + "<|im_start|>assistant\n<think>\nab�\n</think>\n\n<|im_end|>") + NEXT
        self.assertEqual(self.go(record, key, own=False).ids, key)  # no flag: U+FFFD stops the replay
        got = self.go(record, key)
        self.assertEqual(got.ids, record + CLOSURE + [END] + NEXT)

    def test_earlier_message_is_still_replayed_as_text(self):
        """A non-canonical split in an older message (same text) is replayed as before, then own-turn on the last."""
        old = PROMPT + [A, B] + enc("\n</think>\n\nok") + [END]
        record = old + NEXT + SPLIT + enc(" cd")
        key = (enc(HEAD + "<|im_start|>assistant\n<think>\nab\n</think>\n\nok<|im_end|>") + NEXT
               + [AB] + enc(" cd\n</think>\n\n<|im_end|>") + NEXT)
        got = self.go(record, key, gen_start=len(old) + len(NEXT))
        self.assertEqual(got.ids, record + CLOSURE + [END] + NEXT)
        self.assertEqual([s.get("kind") for s in got.spans], [None, "own_turn"])

    # --------------------------------------------------------------- negatives

    def test_flag_off_is_unchanged(self):
        for name in SHAPES:
            with self.subTest(name):
                record, key, _ = shape(name)
                got = self.go(record, key, own=False)
                self.assertEqual(got.ids, key)
                self.assertEqual(got.spans, [])
                self.assertEqual(got.skip, "text_differs")
                self.assertIsNone(got.own_skip)

    def test_resend_is_unchanged(self):
        record, _, _ = shape("think_cut_prefix")
        for key in (record[:GEN_START], record[: GEN_START + 1], record):
            with self.subTest(len(key)):
                got = self.go(record, list(key))
                self.assertEqual((got.ids, got.spans, got.skip), (list(key), [], None))

    def test_divergence_before_the_generation_is_unchanged(self):
        """The client dropped the last turn's thinking (its header differs before gen_start): text_differs."""
        record, _, _ = shape("think_cut_prefix")
        key = client("ab cd")
        got = self.go(record, key)
        self.assertEqual((got.ids, got.spans, got.skip, got.own_skip), (key, [], "text_differs", "before_generation"))
        # and the same divergence with gen_start moved before it is own-turn: gen_start is what decides
        self.assertEqual(self.go(record, key, gen_start=START + 3).spans[0]["kind"], "own_turn")

    def test_older_message_divergence_is_unchanged(self):
        """The client changed an older assistant message: never touched, text_differs as before."""
        old = PROMPT + enc("ab\n</think>\n\nok") + [END]
        record = old + NEXT + SPLIT + enc(" cd")
        key = (enc(HEAD + "<|im_start|>assistant\n<think>\nab\n</think>\n\nno<|im_end|>") + NEXT
               + [AB] + enc(" cd\n</think>\n\n<|im_end|>") + NEXT)
        for gen_start in (len(old) + len(NEXT), 0):  # even with a gen_start before it
            with self.subTest(gen_start=gen_start):
                got = self.go(record, key, gen_start=gen_start)
                self.assertEqual((got.ids, got.spans, got.skip, got.own_skip),
                                 (key, [], "text_differs", "earlier_message"))

    def test_no_message_end_is_unchanged(self):
        """The client continues the cut message itself (no <|im_end|> after it): no_end, as before."""
        record, _, _ = shape("think_cut_prefix")
        key = enc(HEAD + "<|im_start|>assistant\n<think>\nab cd and more")
        got = self.go(record, key)
        self.assertEqual((got.ids, got.spans, got.skip), (key, [], "no_end"))

    def test_blockers_keep_priority(self):
        record, key, _ = shape("think_cut_prefix")
        key_end = key.index(END, GEN_START)
        cases = {
            "logprob_start": dict(return_logprob=True, logprob_start_len=GEN_START + 1),
            "mm_type": dict(mm_inputs=SimpleNamespace(input_ids=key)),
            "routed_experts": dict(return_routed_experts=True, routed_experts_start_len=GEN_START + 1),
            "embeds": dict(input_embeds=[[0.0]] * len(key)),
        }
        for why, kw in cases.items():
            with self.subTest(why):
                got = self.go(record, key, **kw)
                self.assertEqual((got.ids, got.spans, got.skip), (key, [], why))
        # an image inside the replaced message
        mm_key = key[:key_end] + [VS, PAD, VE] + key[key_end:]
        got = self.go(record, mm_key, mm_inputs=mm_output(mm_key))
        self.assertEqual((got.ids, got.spans, got.skip), (mm_key, [], "mm_overlap"))

    def test_input_len_is_unchanged(self):
        record, key, closure = shape("think_cut_prefix")
        want = expected(record, closure)
        got = self.go(record, key, max_input_len=len(want))
        self.assertEqual((got.ids, got.spans, got.skip), (key, [], "input_len"))

    # --------------------------------------------------------------- offsets that move with the delta

    def test_later_image_logprob_start_and_mrope_shift_by_the_delta(self):
        for name in ("tool_call_cut", "think_cut_prefix"):  # without and with a closure
            with self.subTest(name):
                record, key, closure = shape(name)
                image = [VS] + [PAD] * 4 + [VE]
                key = key + image + enc("go") + [END]
                want = expected(record, closure) + image + enc("go") + [END]
                mm = mm_output(key, padded=True)
                logprob = len(key) - 2
                got = self.go(record, key, mm_inputs=mm, return_logprob=True, logprob_start_len=logprob)
                self.assertEqual(got.ids, want)
                delta = got.spans[0]["delta"]
                self.assertEqual(len(want) - len(key), delta)
                self.assertNotEqual(delta, 0)
                a = want.index(PAD)
                self.assertEqual([tuple(o) for o in mm.mm_items[0].offsets], [(a, a + 3)])
                self.assertEqual(list(mm.input_ids), want)
                self.assertEqual(list(mm.padded_input_ids), want[:a] + [-7] * 4 + want[a + 4:])
                self.assertEqual(got.req.logprob_start_len, logprob + delta)
                self.assertTrue(torch.equal(mm.mrope_positions, mrope(want, (a, a + 3))))
