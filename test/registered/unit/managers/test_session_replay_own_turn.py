"""arc-forge own-last-turn replay on the TITO route (session_ledger.replay_ids, --session-replay-own-last-turn).

RouteCases (replay_own_turn_fixtures) on replay_ids with own_turn=, gen_start= and entries=; SessionCases on a
SessionLedger (sticky entries kept with S across requests); the ledger passes the session's gen_start, counts
own-turn spans, drops entries on a split / close / evict and never commits an aborted generation; the flag needs
--session-replay. CPU only.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_own_turn_fixtures import (  # noqa: E402
    A, B, BIG, DECODE, END, EOS, GEN_START, NEXT, OT, PROMPT, REPLY, REPLY_TEXT, SHAPES, STOP, TOK, RouteCases,
    SessionCases, ToyChat, enc, expected, recv, shape)

from sglang.srt.managers import session_ledger as sl  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def run_tito(record, key, gen_start=GEN_START, own=True, max_input_len=BIG, **kw):
    r = recv(key, **kw)
    res = sl.replay_ids(r, np.asarray(record, dtype=np.int64), STOP, END, DECODE, max_input_len, None,
                        own_turn=OT if own else None, gen_start=gen_start, entries={} if own else None)
    return SimpleNamespace(ids=list(r.input_ids), spans=res.spans if res.spans else [], skip=res.skip,
                           own_skip=res.own_turn_skip, req=r)


class TitoRouteTest(RouteCases, unittest.TestCase):
    route = staticmethod(run_tito)

    def test_gen_start_is_required(self):
        record, key, _ = shape("think_cut_prefix")
        with self.assertRaisesRegex(ValueError, "needs gen_start"):
            sl.replay_ids(recv(key), np.asarray(record, dtype=np.int64), STOP, END, DECODE, BIG, None, own_turn=OT)
        with self.assertRaisesRegex(ValueError, "own-turn entries"):
            sl.replay_ids(recv(key), np.asarray(record, dtype=np.int64), STOP, END, DECODE, BIG, None, own_turn=OT,
                          gen_start=GEN_START)
        with self.assertRaisesRegex(ValueError, "entries without own-turn"):
            sl.replay_ids(recv(key), np.asarray(record, dtype=np.int64), STOP, END, DECODE, BIG, None, entries={})


def make_ledger(own):
    return sl.SessionLedger(tokenizer=TOK, eos_token_ids=EOS, ledger_dir=None, record_all=False, logprobs=False,
                            routing=False, routing_server_enabled=False, max_bytes=1 << 20, tokenizer_worker_num=1,
                            server_meta={}, port=0, own_last_turn=own)


class TitoSession:
    """SessionCases' route: one session of a SessionLedger (replay only, no ledger dir)."""

    def __init__(self, own):
        self.led = make_ledger(own)

    def turn(self, rid, key, gen, finish):
        obj = SimpleNamespace(rid=rid, session_id="s", image_data=None, video_data=None)
        tok = recv(key, rid=rid, sampling_params=None)
        self.led.on_intake(obj, tok, BIG)
        res = self.led.pending[rid].replay
        reason = {"type": "stop", "matched": int(gen[-1])} if finish == "stop" else {"type": finish}
        self.led.on_finish(rid, SimpleNamespace(output_ids=gen), {}, reason)
        return SimpleNamespace(ids=list(tok.input_ids), spans=res.spans, skip=res.skip, own_skip=res.own_turn_skip)

    def abort_waiting(self, rid, key):
        self.turn(rid, key, [], "abort")  # an attempt: S is not touched

    def entries(self):
        sess = self.led.sessions.get("s")
        return {} if sess is None else sess.own

    def corrupt(self, pos):
        S = self.led.sessions["s"].S.copy()
        S[pos] = A if S[pos] != A else B
        self.led.sessions["s"].S = S

    def counters(self):
        return self.led.counters


class TitoSessionTest(SessionCases, unittest.TestCase):
    def session(self, own=True):
        return TitoSession(own)

    def test_rewind_after_an_entry_keeps_it(self):
        """A resend of the last request rewinds S into its generation, after the entry: kept, and used again."""
        sess = self.session()
        chat = ToyChat()
        sess.turn("r0", chat.request(), SHAPES["tool_call_cut"][0], "length")
        chat.answered(SHAPES["tool_call_cut"][1])
        key1 = chat.request()
        k = sess.turn("r1", key1, REPLY, "stop")
        (entry,) = self.entries_of(sess)
        again = sess.turn("r1b", key1, REPLY, "stop")  # the client resent r1 (its reply dropped)
        self.assertEqual(again.ids, k.ids)
        self.assertTrue(any(s.get("sticky") for s in again.spans))
        self.assertEqual(sess.counters()["rewind"], 1)
        self.assertEqual(self.entries_of(sess), [entry])
        chat.answered(REPLY_TEXT)
        nxt = sess.turn("r2", chat.request(), REPLY, "stop")
        self.assertTrue(any(s.get("sticky") for s in nxt.spans))
        self.assertEqual(nxt.ids, k.ids + REPLY + NEXT)

    def test_split_drops_entries(self):
        """A retry of an earlier turn opens a new segment: the entries go with the old one."""
        sess, record, closure, ids, outs, chat = self.cut_then_replies("tool_call_cut", 1)
        sess.turn("r9", PROMPT, REPLY, "stop")  # the very first request again: a branch (retry)
        self.assertEqual(sess.counters()["seg_open_retry"], 1)
        self.assertEqual(self.entries_of(sess), [])
        self.assertEqual(self.dropped(sess, "split"), 1)

    def test_close_and_evict_drop_entries(self):
        for how in ("close", "evict"):
            with self.subTest(how):
                sess, *_ = self.cut_then_replies("tool_call_cut", 1)
                if how == "close":
                    sess.led.close_session("s")
                else:
                    sess.led.max_bytes = 0  # below anything: the next _evict empties the ledger
                    sess.led._evict()
                self.assertEqual(sess.entries(), {})
                self.assertEqual(self.dropped(sess, how), 1)

    def test_aborted_output_never_enters_S(self):
        """An aborted request commits nothing: S keeps the last finished generation (the one own-turn may take over),
        and the aborted request's own takeover leaves no entry."""
        sess = self.session()
        chat = ToyChat()
        gen = SHAPES["tool_call_cut"][0]
        sess.turn("r0", chat.request(), gen, "length")
        S0 = sess.led.sessions["s"].S.tolist()
        chat.answered(SHAPES["tool_call_cut"][1])
        key = chat.request()
        aborted = sess.turn("r1", key, enc("o"), "abort")
        self.assertEqual([s["kind"] for s in aborted.spans], ["own_turn"])
        self.assertEqual(sess.led.sessions["s"].S.tolist(), S0)
        self.assertEqual(sess.entries(), {})
        again = sess.turn("r1b", key, REPLY, "stop")  # the retry takes the same generation over
        self.assertEqual(again.ids, aborted.ids)
        self.assertEqual(len(sess.entries()), 1)
        self.assertEqual(sess.led.sessions["s"].S.tolist(), again.ids + REPLY)


class TitoLedgerTest(unittest.TestCase):
    """The ledger passes the session's gen_start, counts own-turn spans, and the flag needs --session-replay."""

    def ledger(self, own):
        return sl.SessionLedger(tokenizer=TOK, eos_token_ids=EOS, ledger_dir=None, record_all=False, logprobs=False,
                                routing=False, routing_server_enabled=False, max_bytes=1 << 20,
                                tokenizer_worker_num=1, server_meta={}, port=0, own_last_turn=own)

    def turn(self, led, rid, ids, gen, finish):
        obj = SimpleNamespace(rid=rid, session_id="s", image_data=None, video_data=None)
        tok = recv(ids, rid=rid, sampling_params=None)
        led.on_intake(obj, tok, BIG)
        led.on_finish(rid, SimpleNamespace(output_ids=gen), {}, finish)
        return list(tok.input_ids)

    def test_cut_turn_extends_with_the_flag_and_rewinds_without(self):
        for own in (True, False):
            with self.subTest(own=own):
                led = self.ledger(own)
                record, key, closure = shape("think_cut_prefix")
                self.turn(led, "r1", PROMPT, record[GEN_START:], {"type": "length"})
                seen = self.turn(led, "r2", key, enc("ok") + [END], {"type": "stop", "matched": END})
                c = led.counters
                if own:
                    self.assertEqual(seen, expected(record, closure))
                    self.assertEqual((c["replay_own_turn"], c["replay_own_turn_closure"], c["rewind"]), (1, 3, 0))
                    self.assertEqual(led.sessions["s"].S.tolist(), seen + enc("ok") + [END])
                else:
                    self.assertEqual(seen, key)
                    self.assertEqual((c["replay_own_turn"], c["replay_skip_text_differs"], c["rewind"]), (0, 1, 1))

    def test_flag_needs_session_replay(self):
        from sglang.srt.managers import tokenizer_manager as tm_mod

        serving = SimpleNamespace(session_replay=False, session_replay_empty_think=False,
                                  session_replay_own_last_turn=True, session_ledger_dir=None, session_ledger_all=False,
                                  session_ledger_logprobs=False, session_ledger_routing=False)
        with mock.patch.object(tm_mod, "get_serving", lambda: serving):
            with self.assertRaisesRegex(ValueError, "--session-replay-own-last-turn needs --session-replay"):
                tm_mod.TokenizerManager.init_session_ledger(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
