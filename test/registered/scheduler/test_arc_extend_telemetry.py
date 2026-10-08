"""arc-forge S0: extend-batch / prefill-delayer telemetry (ARC_EXTEND_TELEMETRY), CPU only."""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers import arc_extend_telemetry as tele
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeEvent:
    """A CUDA event stand-in: record() stamps a fake GPU clock; landed when the test says so."""

    gpu_ms = 0.0

    def __init__(self):
        self.t = None
        self.landed = True

    def record(self):
        self.t = FakeEvent.gpu_ms

    def query(self):
        return self.landed

    def elapsed_time(self, other):
        return other.t - self.t


def make_batch(prefix, extend, media=None, mode="extend"):
    """media: per request, a list of item offset lists (or None for a text-only request)."""
    media = media or [None] * len(prefix)
    reqs = []
    for i, items in enumerate(media):
        mm = None if items is None else SimpleNamespace(mm_items=[SimpleNamespace(offsets=o) for o in items])
        reqs.append(SimpleNamespace(rid=f"r{i}", multimodal_inputs=mm))
    fm = SimpleNamespace(is_mixed=lambda: mode == "mixed", is_idle=lambda: False)
    return SimpleNamespace(reqs=reqs, extend_lens=list(extend), prefix_lens=list(prefix), forward_mode=fm)


class TestExtendTelemetry(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.events = []
        self.landed = True

        def factory():
            e = FakeEvent()
            e.landed = self.landed
            self.events.append(e)
            return e

        self.t = tele.ExtendTelemetry(60.0, event_factory=factory, clock=lambda: self.now)

    def run_batch(self, batch, target_gpu, refresh_gpu, target_cpu, refresh_cpu):
        rec = self.t.begin(batch)
        FakeEvent.gpu_ms += target_gpu
        self.now += target_cpu
        self.t.target_done(rec)
        FakeEvent.gpu_ms += refresh_gpu
        self.now += refresh_cpu
        self.t.end(rec)
        return rec

    def test_shape_and_media(self):
        b = make_batch([0, 40000], [300, 500], media=[[[(10, 20)]], [[(39000, 39100)], [(40100, 40200)]]])
        # cold request's image at 10 is new; second request: one old image, one new
        self.assertEqual(tele.batch_shape(b), (2, 800, 40000, 40000, 1, 2))

    def test_contracts(self):
        with self.assertRaises(NotImplementedError):
            tele.batch_shape(make_batch([0], [10], mode="mixed"))
        with self.assertRaises(RuntimeError):
            tele.batch_shape(make_batch([0, 1], [10]))
        with self.assertRaises(RuntimeError):
            tele.batch_shape(make_batch([0], [10], media=[[None]]))
        rec = self.t.begin(make_batch([0], [10]))
        with self.assertRaises(RuntimeError):
            self.t.end(rec)  # end before target_done

    def test_bins_wait_for_events(self):
        self.landed = False
        rec = self.run_batch(make_batch([100, 200], [300, 400]), 90.0, 6.0, 0.150, 0.020)
        self.assertEqual(self.t._window.bins, {})  # unresolved: nothing aggregated, nothing synced
        self.assertEqual(len(self.t._pending), 1)
        rec.e2.landed = True
        self.landed = True
        self.run_batch(make_batch([5000], [3000]), 200.0, 15.0, 0.200, 0.030)
        self.assertEqual(len(self.t._pending), 0)
        bins = self.t._window.bins
        self.assertEqual(set(bins), {("k2", "<1024"), ("k1", "<4096")})
        b = bins[("k2", "<1024")]
        self.assertEqual((b.n, b.target_gpu_ms, b.refresh_gpu_ms, b.new_tokens), (1, 90.0, 6.0, 700))
        self.assertAlmostEqual(b.target_cpu_ms, 150.0)
        self.assertAlmostEqual(b.refresh_cpu_ms, 20.0)

    def test_flush_line_and_delayer(self):
        self.run_batch(make_batch([0], [100]), 80.0, 5.0, 0.1, 0.01)
        common = dict(input_estimation="all", queue_max_delay_ms=1000.0)
        self.t.observe_delayer(output_allow=False, output_reason="delay", actual_execution=False, wait_seconds=0.0, **common)
        self.t.observe_delayer(output_allow=True, output_reason="wait_success", actual_execution=True, wait_seconds=0.4, **common)
        self.t.observe_delayer(output_allow=True, output_reason="wait_success", actual_execution=True, wait_seconds=1.2, **common)
        self.t.observe_delayer(output_allow=True, output_reason="", actual_execution=False, wait_seconds=0.0,
                               input_estimation="none", queue_max_delay_ms=1000.0)
        w = self.t._window
        self.assertEqual(w.released["wait_success"], 1)
        self.assertEqual(w.released["wait_ms_cap"], 1)
        self.assertEqual(w.delayed_passes, 1)
        with self.assertRaises(RuntimeError):
            self.t.observe_delayer(output_allow=True, output_reason="bogus", actual_execution=True, wait_seconds=0.0, **common)
        with self.assertLogs(tele.logger, level="INFO") as logs:
            self.now += 61.0
            self.t.observe_delayer(output_allow=True, output_reason="no_wait", actual_execution=True, wait_seconds=0.0, **common)
        line = logs.output[-1]
        self.assertIn("1 extend batches (k1 1 / k2 0 / k3 0 / k4+ 0)", line)
        self.assertIn("k1/<512 n=1 gpu=80.0/5.0", line)
        self.assertIn("wait_ms_cap 1", line)
        self.assertEqual(self.t._window.bins, {})  # new window

    def test_single_thread(self):
        self.t.begin(make_batch([0], [10]))
        err = []
        th = threading.Thread(target=lambda: err.append(self._try_begin()))
        th.start()
        th.join()
        self.assertIsInstance(err[0], RuntimeError)

    def _try_begin(self):
        try:
            self.t.begin(make_batch([0], [10]))
        except RuntimeError as e:
            return e
        return None

    def test_env_resolution(self):
        bad = [
            {"ARC_EXTEND_TELEMETRY": "2"},
            {"ARC_EXTEND_TELEMETRY_INTERVAL_S": "5"},  # interval without the switch
            {"ARC_EXTEND_TELEMETRY": "1", "ARC_EXTEND_TELEMETRY_INTERVAL_S": "0"},
        ]
        for environ in bad:
            with self.subTest(environ=environ), mock.patch.dict(os.environ, environ, clear=True), \
                    mock.patch.object(tele, "_RESOLVED", False), mock.patch.object(tele, "_TELEMETRY", None):
                with self.assertRaises(ValueError):
                    tele.get_extend_telemetry()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(tele, "_RESOLVED", False), \
                mock.patch.object(tele, "_TELEMETRY", None):
            self.assertIsNone(tele.get_extend_telemetry())


if __name__ == "__main__":
    unittest.main()
