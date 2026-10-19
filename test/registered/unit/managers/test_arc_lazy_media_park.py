"""arc lazy media (ARC_MM_LAZY_MEDIA=1) with --session-exact-resume (mem_cache/session_park.py), on the RL rollout
tree (UnifiedRadixCache + SessionTailPark): a waiting request the scheduler parks for its lazy media keeps its session
park claim -- nothing is handed back, the park cannot be evicted meanwhile -- and after the fill the re-match returns
the same claim, so the request still resumes exactly at the park's end. An abort or a failed fill while parked hands
the claim back. CPU only.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # test_arc_lazy_media_fill
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mem_cache"))  # test_session_tail_park

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

from test_arc_lazy_media_fill import IMG_LEN, _lazy, _with_pixels, bare_scheduler  # noqa: E402
from test_session_tail_park import PAGE, SessionTailParkHarness, _seq  # noqa: E402

from sglang.srt.managers.io_struct import AbortReq, ArcMmRematerializeReq  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN, MultimodalInputs  # noqa: E402
from sglang.srt.multimodal import arc_lazy_media  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")



class ParkedRequestKeepsItsClaim(SessionTailParkHarness):
    """--session-exact-resume on the RL rollout tree (UnifiedRadixCache + SessionTailPark)."""

    S1_IN = _seq(1000, 3 * PAGE + 9)
    S1_OUT = _seq(5000, 150)

    def _second_turn(self):
        self._turn(self.S1_IN, self.S1_OUT)
        (park,) = self.park.lru.values()  # [0, 350)
        tokens = self.S1_IN + self.S1_OUT + _seq(7000, 40)
        req = self._req(tokens)
        # an image of an earlier turn inside the park's prefix, this turn's frame in the extend: both seen before
        req.multimodal_inputs = MultimodalInputs(mm_items=[_lazy(11, 100), _lazy(12, 360)])
        return park, req, tokens

    def _fill(self, req):
        fresh = [_lazy(11, 100), _with_pixels(12, 360)]
        return SimpleNamespace(rid=req.rid, mm_inputs=MultimodalInputs(mm_items=fresh), arc_mm_refill=True)

    def test_claim_is_held_while_parked_and_the_fill_resumes_exactly(self):
        park, req, _ = self._second_turn()
        s = bare_scheduler(self.cache)
        s.waiting_queue.append(req)
        self._match(req)
        self.assertEqual(self._mrp(req), ("exact", None, None))
        needed = s._arc_lazy_needed(req)
        self.assertEqual(needed, [req.multimodal_inputs.mm_items[1].hash])  # only the lazy item in the extend
        s._arc_lazy_send_back([(req, needed)])

        self.assertEqual(s.waiting_queue, [])
        self.assertIs(s.arc_mm_lazy_parked[req.rid][0], req)
        (msg,) = s.to_tokenizer
        self.assertIsInstance(msg, ArcMmRematerializeReq)
        self.assertEqual((msg.rid, msg.hashes, msg.refill), (req.rid, needed, False))
        # the claim stays with the parked request: nothing handed back, the park cannot be evicted meanwhile
        self.assertEqual((park.claimed_by, park.claimant), (req.rid, req))
        self.assertIs(self.park.claims[req.rid], park)
        self.assertEqual(self.park.counters["claim_released"], 0)
        self.assertFalse(self.park.evict_lru())
        self._accounting()
        self.assertFalse(s.is_fully_idle())  # idle checks would see a claim outstanding
        self.assertTrue(s.is_fully_idle(for_health_check=True))

        s._arc_mm_refill(self._fill(req), None)
        self.assertEqual((s.waiting_queue, s.arc_mm_lazy_parked), ([req], {}))
        items = req.multimodal_inputs.mm_items
        self.assertIn(arc_lazy_media.LAZY_KEY, items[0].model_specific_data)  # in the prefix: stays lazy
        self.assertEqual(items[1].feature.tolist(), [[12.0, 12.0]] * IMG_LEN)
        self.assertEqual(items[1].offsets, [(360, 363)])

        # the next scheduling round: the same claim, exact at the park's end, nothing left to fill
        self._match(req)
        self.assertEqual(len(req.prefix_indices), park.pos)
        self.assertEqual(s._arc_lazy_needed(req), [])
        self._admit(req)
        self.assertEqual(self._mrp(req), ("exact", None, park.pos))
        self.assertEqual((self.park.counters["consume"], self.park.counters["claim_released"]), (1, 0))
        self._run(req, [1, 2, 3])
        self._release(req, FINISH_MATCHED_TOKEN(matched=3))
        self._accounting()
        self._drain()

    def test_abort_while_parked_hands_the_claim_back(self):
        park, req, tokens = self._second_turn()
        s = bare_scheduler(self.cache)
        s.waiting_queue.append(req)
        self._match(req)
        s._arc_lazy_send_back([(req, s._arc_lazy_needed(req))])
        s.abort_request(AbortReq(rid=req.rid))
        self.waiting.remove(req)
        self.assertEqual(s.arc_mm_lazy_parked, {})
        self.assertIsNone(park.claimed_by)
        self.assertEqual(self.park.counters["claim_released"], 1)
        self.assertEqual(self._mrp(req), ("fallback", "abort", None))
        abort = s.to_tokenizer[-1]
        self.assertIsInstance(abort, AbortReq)
        self.assertEqual(abort.rid, req.rid)
        self.assertTrue(s.is_fully_idle())
        self._accounting()
        # the fill on its way is dropped
        s._arc_mm_refill(self._fill(req), None)
        self.assertEqual((s.waiting_queue, s.arc_mm_refill_aborted), ([], set()))
        # the park is whole: the client's retry resumes exactly
        b = self._admit(self._match(self._req(tokens)))
        self.assertEqual(self._mrp(b), ("exact", None, park.pos))
        self._run(b, [1, 2, 3])
        self._release(b, FINISH_MATCHED_TOKEN(matched=3))
        self._accounting()
        self._drain()

    def test_a_failed_fill_aborts_the_parked_request(self):
        park, req, _ = self._second_turn()
        s = bare_scheduler(self.cache)
        s.waiting_queue.append(req)
        self._match(req)
        s._arc_lazy_send_back([(req, s._arc_lazy_needed(req))])
        bad = SimpleNamespace(rid=req.rid, mm_inputs=MultimodalInputs(mm_items=[_lazy(11, 100), _lazy(12, 360)]))
        s._arc_mm_refill(bad, None)
        self.waiting.remove(req)
        abort = s.to_tokenizer[-1]
        self.assertIsInstance(abort, AbortReq)
        self.assertIn("no feature for media item", abort.finished_reason["message"])
        self.assertEqual((s.arc_mm_lazy_parked, s.waiting_queue), ({}, []))
        self.assertIsNone(park.claimed_by)
        self._accounting()
        self._drain()

    def test_parked_twice_is_a_bug(self):
        _, req, _ = self._second_turn()
        s = bare_scheduler(self.cache)
        self._match(req)
        needed = s._arc_lazy_needed(req)
        s._arc_lazy_send_back([(req, needed)])
        with self.assertRaisesRegex(RuntimeError, "parked already"):
            s._arc_lazy_send_back([(req, needed)])
        s.abort_request(AbortReq(rid=req.rid))
        self.waiting.remove(req)
        self._drain()
