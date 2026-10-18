"""arc-forge own-last-turn replay on the HiSparse route (QSAHiSparseSessionCache.replay_request, h14;
hisparse_config.qsa_session_replay_own_last_turn).

RouteCases (managers/replay_own_turn_fixtures) on replay_request against a PARKED record whose generation starts
at its prompt_len; SessionCases over a session's requests, each parked the way cache_finished_req writes the
record (tokens, prompt_len, gen_end, then _park_entries); the counters, the hisparse_config keys, a record with raw
ids, the entries a release / truncate / close drops and the ones the parking request carries, and the abort
exclusion. The cache is built without its coordinator (only the replay state, and no-op pools for release and
truncate): arc-forge work/qsa_hisparse/tests runs the whole object. CPU only.
"""

import json
import logging
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "managers"))

from replay_own_turn_fixtures import (  # noqa: E402
    A, B, BIG, CLOSURE, END, EOS, GEN_START, NEXT, NL, PAD, PROMPT, REPLY, REPLY_TEXT, SHAPES, TOK, UNTHINK, RouteCases,
    SessionCases, ToyChat, client, enc, expected, recv, shape)

from sglang.srt.mem_cache import qsa_hisparse_session_cache as sc  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def bare_cache(own):
    """The replay state of a cache, its record bookkeeping, and pools whose frees do nothing."""
    cache = object.__new__(sc.QSAHiSparseSessionCache)
    cache.replay, cache.own_last_turn, cache.replay_own_turn = True, own, None
    cache.stats, cache._own_intake, cache.records, cache.index, cache.gen = Counter(), {}, [], {}, {"s": 1}
    cache.owner, cache.ledger_version, cache.page_size, cache.block_tokens = {}, 0, 1, 1
    nothing = SimpleNamespace(free=lambda *a: None, free_buffer=lambda *a: None)
    cache.token_to_kv_pool_allocator = nothing
    cache.coord = SimpleNamespace(mem_pool_host=nothing, snapshots=nothing)
    cache.set_replay_tokenizer(TOK, EOS)
    return cache


def hisparse_cache(record, gen_start, own, raw=None, gen_end=None, gen_finished=True):
    """gen_end: where the producer's output through its stop ends (cache_finished_req); default the record's end."""
    cache = bare_cache(own)
    rec = sc.SessionRecord("s", 1, None, None, P=0, committed=0, gdn=0, tokens=np.asarray(record, dtype=np.int64),
                           prompt_len=gen_start, raw=raw, gen_end=len(record) if gen_end is None else gen_end,
                           gen_finished=gen_finished)
    cache.index = {"s": rec}
    cache.records = [rec]
    return cache


HS_SKIPS = {"replay_text_differs": "text_differs", "replay_no_end": "no_end"}


def replay_logged(cache, r, max_input_len):
    """replay_request, and the QSA_REPLAY lines it logged."""
    cap = _Capture()
    sc.logger.addHandler(cap)
    level = sc.logger.level
    sc.logger.setLevel(logging.INFO)
    try:
        cache.replay_request(r, max_input_len)
    finally:
        sc.logger.removeHandler(cap)
        sc.logger.setLevel(level)
    return [json.loads(m.split(" ", 1)[1]) for m in cap.lines if m.startswith("QSA_REPLAY ")]


def outcome(stats, logged, r):
    skips = [HS_SKIPS.get(k, k[len("replay_skip_"):]) for k in stats if k in HS_SKIPS or k.startswith("replay_skip_")]
    own_skip = [k[len("replay_own_turn_skip_"):] for k in stats if k.startswith("replay_own_turn_skip_")]
    return SimpleNamespace(ids=list(r.input_ids), spans=logged[0]["spans"] if logged else [],
                           skip=skips[0] if skips else None, own_skip=own_skip[0] if own_skip else None,
                           stats=stats, req=r)


def run_hisparse(record, key, gen_start=GEN_START, own=True, max_input_len=BIG, gen_end=None, gen_finished=True,
                 **kw):
    cache = hisparse_cache(record, gen_start, own, gen_end=gen_end, gen_finished=gen_finished)
    r = recv(key, **kw)
    return outcome(cache.stats, replay_logged(cache, r, max_input_len), r)


def finished_req(rid, finish):
    """What _park_entries reads of the Req: finished() (False for a retract) and the finish reason's type."""
    return SimpleNamespace(rid=rid, finished=lambda: finish != "retract",
                           finished_reason=SimpleNamespace(to_json=lambda: {"type": finish}))


class HsSession:
    """SessionCases' route: one session of the cache; every request replays at intake and then parks the record
    the way cache_finished_req writes it (its ids + its output, no draft ids) and calls _park_entries."""

    def __init__(self, own):
        self.cache = bare_cache(own)

    def record(self):
        rec = self.cache.index.get("s")
        if rec is None:  # the request's first snapshot creates it (on_snapshot_published)
            rec = sc.SessionRecord("s", 1, None, None, P=0, committed=0, gdn=0, logical=torch.zeros(0),
                                   host_blocks=torch.zeros(0))
            self.cache.records.append(rec)
            self.cache.index["s"] = rec
        return rec

    def turn(self, rid, key, gen, finish, before_park=None):
        before = Counter(self.cache.stats)
        r = recv(key, rid=rid)
        logged = replay_logged(self.cache, r, BIG)
        if before_park is not None:
            before_park(self.cache)
        ids = list(r.input_ids)
        rec = self.record()
        rec.tokens = np.asarray(ids + gen, dtype=np.int64)
        rec.prompt_len, rec.gen_end, rec.raw = len(ids), len(ids) + len(gen), None
        if self.cache.own_last_turn:  # as cache_finished_req
            self.cache._park_entries(rec, finished_req(rid, finish))
        return outcome(self.cache.stats - before, logged, r)

    def abort_waiting(self, rid, key):
        """Aborted before admission: nothing parks; its intake lapses as in release_aborted_request."""
        replay_logged(self.cache, recv(key, rid=rid), BIG)
        if self.cache.own_last_turn and self.cache._own_intake["s"][0] == rid:
            del self.cache._own_intake["s"]

    def entries(self):
        rec = self.cache.index.get("s")
        return {} if rec is None else rec.own

    def corrupt(self, pos):
        rec = self.cache.index["s"]
        rec.tokens = rec.tokens.copy()
        rec.tokens[pos] = A if rec.tokens[pos] != A else B

    def counters(self):
        return self.cache.stats


class HiSparseSessionTest(SessionCases, unittest.TestCase):
    def session(self, own=True):
        return HsSession(own)

    def test_release_drops_entries(self):
        """A released record (evict, reap, close, a mismatch) takes its entries along; a new record starts without."""
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        sess.cache._release_record(sess.cache.index["s"])
        self.assertEqual(self.dropped(sess, "release"), 1)
        key = chat.request()
        self.assertEqual(sess.turn("r2", key, REPLY, "stop").ids, key)  # no record at intake: the client's ids
        self.assertEqual(sess.entries(), {})
        chat.answered(REPLY_TEXT)
        key = chat.request()
        out = sess.turn("r3", key, REPLY, "stop")  # the new record is the client's version: nothing to replay
        self.assertEqual((out.ids, out.spans, out.skip), (key, [], None))

    def test_close_drops_entries_and_the_intake(self):
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        sess.cache._own_intake["s"] = ("r-live", {}, 0)
        sess.cache.release_radix_session("s")
        self.assertEqual((self.dropped(sess, "release"), sess.cache._own_intake), (1, {}))

    def test_truncate_drops_the_records_entries_the_request_carries_its_own(self):
        """_decide truncates the record after the request's intake: the record's entries go, and the request parks
        the ones its ids hold (validated at its intake, checked again against what it parks)."""
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        (entry,) = self.entries_of(sess)
        rec = sess.cache.index["s"]
        rec.P, rec.anchor, rec.abuf = 8, 4, 1
        rec.logical, rec.host_blocks = torch.arange(8), torch.arange(8)
        out = sess.turn("r2", chat.request(), REPLY, "stop", before_park=lambda c: c._truncate(c.index["s"]))
        self.assertEqual(self.dropped(sess, "truncate"), 1)
        self.assertTrue(any(s.get("sticky") for s in out.spans))
        self.assertEqual(self.entries_of(sess), [entry])

    def test_a_request_that_leaves_the_record_early_does_not_carry_its_entries(self):
        """A retry of an earlier turn shares the record only before the entry's end: not carried, and the record
        it parks keeps none."""
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        sess.turn("r9", PROMPT, REPLY, "stop")
        self.assertEqual(sess.counters()["replay_own_turn_sticky_unshared"], 1)
        self.assertEqual(sess.entries(), {})

    def test_an_abort_parked_record_is_not_taken_over(self):
        """r1 takes r0's cut generation over, then aborts with a partial output it parks: that partial is not
        taken over (why aborted), while r0's message stays sticky (r1's ids hold it, so its park carries it)."""
        sess = self.session()
        chat = ToyChat()
        sess.turn("r0", chat.request(), SHAPES["tool_call_cut"][0], "length")
        chat.answered(SHAPES["tool_call_cut"][1])
        partial = SHAPES["think_cut_not_prefix"][0]
        sess.turn("r1", chat.request(), partial, "abort")
        rec = sess.cache.index["s"]
        self.assertIs(rec.gen_finished, False)
        self.assertEqual(len(rec.own), 1)
        chat.answered(SHAPES["think_cut_not_prefix"][1])  # the client got the partial output and renders it
        out = sess.turn("r2", chat.request(), REPLY, "stop")
        self.assertEqual((out.skip, out.own_skip), ("text_differs", "aborted"))
        self.assertEqual([s.get("sticky") for s in out.spans], [True])

    def test_a_retract_parks_unfinished_and_keeps_the_intake(self):
        sess = self.session()
        chat = ToyChat()
        sess.turn("r0", chat.request(), SHAPES["tool_call_cut"][0], "length")
        chat.answered(SHAPES["tool_call_cut"][1])
        key = chat.request()
        sess.turn("r1", key, enc("o"), "retract")
        rec = sess.cache.index["s"]
        self.assertIs(rec.gen_finished, False)
        self.assertEqual(sess.cache._own_intake["s"][0], "r1")  # it parks again when it exits for good
        self.assertNotIn("replay_own_turn_entries", sess.counters())
        # its final exit: the same intake (replay_request runs once per request), now finished
        rec.tokens = np.asarray(list(rec.tokens[: rec.prompt_len]) + REPLY, dtype=np.int64)
        rec.gen_end = rec.prompt_len + len(REPLY)
        sess.cache._park_entries(rec, finished_req("r1", "stop"))
        self.assertIs(rec.gen_finished, True)
        self.assertEqual((len(rec.own), sess.cache._own_intake), (1, {}))
        self.assertEqual(sess.counters()["replay_own_turn_entries"], 1)

    def test_a_park_whose_intake_is_not_the_latest_keeps_no_entries(self):
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        rec = sess.cache.index["s"]
        sess.cache._own_intake["s"] = ("someone-else", dict(rec.own), 0)
        sess.cache._park_entries(rec, finished_req("r1", "stop"))
        self.assertEqual((rec.own, self.dropped(sess, "no_intake")), ({}, 1))

    def test_a_request_beside_a_running_one_keeps_its_intake(self):
        """A second request of the session arrives while the record is PINNED (no replay): the running request's
        intake stays, so its park keeps the entries."""
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        rec = sess.cache.index["s"]
        sess.cache._own_intake["s"] = ("r-running", dict(rec.own), 0)
        rec.pin_count = 1
        sess.cache.replay_request(recv(chat.request(), rid="r-second"), BIG)
        self.assertEqual(sess.cache._own_intake["s"][0], "r-running")
        self.assertEqual(sess.counters()["replay_skip_pinned"], 1)

    def test_a_record_without_a_raw_view_keeps_no_entries(self):
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        rec = sess.cache.index["s"]
        sess.cache._own_intake["s"] = ("r9", dict(rec.own), 0)
        rec.raw_ok = False
        sess.cache._park_entries(rec, finished_req("r9", "stop"))
        self.assertEqual((rec.own, self.dropped(sess, "raw"), sess.cache._own_intake), ({}, 1, {}))

    def test_a_parked_entry_its_ids_do_not_hold_raises(self):
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        rec = sess.cache.index["s"]
        sess.cache._own_intake["s"] = ("r9", dict(rec.own), 0)
        rec.tokens = rec.tokens[:5]
        with self.assertRaisesRegex(RuntimeError, "not held"):
            sess.cache._park_entries(rec, finished_req("r9", "stop"))



class HiSparseRouteTest(RouteCases, unittest.TestCase):
    route = staticmethod(run_hisparse)

    def test_counters(self):
        record, key, _ = shape("think_cut_prefix")
        got = self.go(record, key)
        self.assertEqual(got.stats["replay_own_turn"], 1)
        self.assertEqual(got.stats["replay_own_turn_closure"], 3)
        self.assertNotIn("replay_text_differs", got.stats)
        got = self.go(record, key, own=False)
        self.assertEqual(dict(got.stats), {"replay_text_differs": 1})

    def test_flag_needs_replay(self):
        for extra, ok in (({}, (False, False)), ({"qsa_session_replay": True}, (True, False)),
                          ({"qsa_session_replay": True, "qsa_session_replay_own_last_turn": True}, (True, True))):
            self.assertEqual(sc.replay_flags(extra), ok)
        with self.assertRaisesRegex(ValueError, "needs qsa_session_replay"):
            sc.replay_flags({"qsa_session_replay_own_last_turn": True})
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            sc.replay_flags({"qsa_session_replay": True, "qsa_session_replay_own_last_turn": 1})

    def test_draft_ids_past_the_stop_are_not_the_last_message(self):
        """Speculative decoding: the verify that sampled <|im_end|> accepted draft ids past it, and rec.tokens keeps
        them (req.output_ids, not output_ids_through_stop). The client's next request has the same assistant message
        and then its tool response, which diverges from those draft ids: no part of the generation, so it stays
        text_differs and the request keeps the tool response. (GPU 6, 10-18: with the record's end taken as the
        generation's, own-turn replaced whole tool responses by "\\n<|im_start|><|im_start|>".)"""
        gen = enc("ab\n</think>\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>") + [END]
        rec_ids = PROMPT + gen
        nxt = rec_ids + enc("\n<|im_start|>user\nresult xyz<|im_end|>\n<|im_start|>assistant\n<think>\n")
        overshoot = rec_ids + enc("\n<|im_start|><|im_start|>")
        for own in (True, False):
            with self.subTest(own=own):
                got = self.go(overshoot, nxt, gen_end=len(rec_ids), own=own)
                self.assertEqual((got.ids, got.spans, got.skip), (nxt, [], "text_differs"))
                self.assertEqual(got.own_skip, "after_generation" if own else None)
        # what the gen_end guards against: the draft ids taken for the last message eat the tool response
        got = self.go(overshoot, nxt, gen_end=len(overshoot))
        self.assertEqual(got.spans[0]["kind"], "own_turn")
        self.assertNotIn(enc("result")[0], got.ids[len(rec_ids):])

    def test_own_turn_ends_at_the_generation_not_at_the_draft_ids(self):
        """The same record diverging inside the generation: own-turn replaces the message through the stop (or the
        cut), the draft ids past it are not copied, and the replay ends there (past the generation the record holds
        only draft ids)."""
        for name in ("malformed_call_stop", "tool_call_cut", "think_cut_prefix"):
            with self.subTest(name):
                record, key, closure = shape(name)
                overshoot = record + TOK.encode("\n<|im_start|>assistant")
                got = self.go(overshoot, key, gen_end=len(record))
                self.assertEqual(got.ids, expected(record, closure))
                (span,) = got.spans
                self.assertEqual(span["rec_end"], len(record) - 1 if record[-1] in EOS else len(record))
                self.assertEqual(got.stats["replay_own_turn"], 1)

    def test_draft_ids_with_the_closure_text_after_a_cut_do_not_replay_over_the_takeover(self):
        """A length cut inside thinking takes the template closure, and the last verify accepted draft ids past the
        cut whose text is exactly the closure's, split another way (\n </think> \n \n for the closure's
        \n </think> \n\n). The takeover is the last generation's last message and past gen_end the record holds only
        draft ids the client never saw, so the replay ends with it. (Review, 10-18: the loop went on at the same
        message, whose texts now agreed, replayed the record's ids with the draft ids over the takeover, and _carry
        raised in the scheduler.) The request gets the record's message, the closure, the client's rest; its entry
        is the takeover and is held by those ids."""
        gen, rendered, closure = SHAPES["think_cut_not_prefix"]
        record = PROMPT + gen
        key = client(rendered)
        for draft in ([NL, UNTHINK, NL, NL], list(CLOSURE), []):
            with self.subTest(draft=draft):
                self.assertEqual(TOK.decode(draft), TOK.decode(CLOSURE) if draft else "")
                cache = hisparse_cache(record + draft, GEN_START, True, gen_end=len(record))
                r = recv(key)
                got = outcome(cache.stats, replay_logged(cache, r, BIG), r)
                self.assertEqual(got.ids, expected(record, closure))
                self.assertEqual([s.get("kind") for s in got.spans], ["own_turn"])
                rid, entries, n_taken = cache._own_intake["s"]
                (entry,) = entries.values()
                self.assertEqual((rid, n_taken, entry.end), (r.rid, 1, len(record) + len(closure)))
                self.assertTrue(entry.held_by(np.asarray(got.ids)))
                self.assertNotIn("replay_text_differs", got.stats)

    def test_a_takeover_with_draft_ids_after_it_counts_nothing_else(self):
        """The same root in the counters: a takeover that ends on a stop or at a cut, with draft ids past the
        generation, went on into the draft ids and counted replay_text_differs + replay_own_turn_skip_after_generation
        (or replay_skip_stop, for a generation that ended on <|endoftext|>) on the takeover request. hs-on3's
        text_differs 99 = after_generation 99 included such requests."""
        for name in ("malformed_call_stop", "malformed_call_eot", "tool_call_cut", "think_cut_prefix",
                     "think_cut_not_prefix"):
            for draft in (enc("x"), TOK.encode("\n<|im_start|><|im_start|>")):
                with self.subTest(name=name, draft=draft):
                    record, key, closure = shape(name)
                    got = self.go(record + draft, key, gen_end=len(record))
                    self.assertEqual(got.ids, expected(record, closure))
                    self.assertEqual(got.stats["replay_own_turn"], 1)
                    for k in ("replay_text_differs", "replay_own_turn_skip_after_generation", "replay_skip_stop"):
                        self.assertNotIn(k, got.stats)

    def test_a_record_without_gen_end_raises(self):
        record, key, _ = shape("think_cut_prefix")
        for attr in ("gen_end", "gen_finished"):
            with self.subTest(attr):
                cache = hisparse_cache(record, GEN_START, True)
                setattr(cache.index["s"], attr, None)
                with self.assertRaisesRegex(RuntimeError, "no gen_end / gen_finished"):
                    cache.replay_request(recv(key), BIG)

    def test_an_unfinished_generation_is_not_taken_over(self):
        for name in SHAPES:
            with self.subTest(name):
                record, key, _ = shape(name)
                got = self.go(record, key, gen_finished=False)
                self.assertEqual((got.ids, got.spans, got.skip, got.own_skip), (key, [], "text_differs", "aborted"))

    def test_raw_record_with_images_uses_the_raw_ids(self):
        """A record whose tokens are padded keeps its raw ids for replay; own-turn reads those, at prompt_len."""
        record, key, closure = shape("think_cut_prefix")
        padded = [-5 if t == PAD else t for t in record]
        cache = hisparse_cache(padded, GEN_START, True, raw=np.asarray(record, dtype=np.int64))
        r = recv(key)
        cache.replay_request(r, BIG)
        self.assertEqual(list(r.input_ids), expected(record, closure))


if __name__ == "__main__":
    unittest.main()
