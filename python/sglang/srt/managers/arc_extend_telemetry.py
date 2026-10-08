"""arc-forge S0: what the extend (prefill) batches cost on the decode timeline (opt-in, default off).

ARC_EXTEND_TELEMETRY=1 records every extend batch the speculative worker (EAGLEWorkerV2: native MTP
and DFlash) runs, and every prefill-delayer verdict, and logs one aggregate INFO line per
ARC_EXTEND_TELEMETRY_INTERVAL_S seconds of wall time (default 60). Nothing synchronizes:

* batch shape comes from host metadata the scheduler already holds (ScheduleBatch.reqs /
  extend_lens / prefix_lens, the requests' media offsets);
* GPU time is three CUDA events on the forward stream -- before the target extend forward, after
  it, after the drafter's refresh -- resolved with query() on later calls. The target span includes
  the GPU gaps while the CPU is still launching the eager extend: that is the extend's cost on the
  decode timeline (every cycle that waits behind it), which is what this measures;
* host time is perf_counter at the same three points. The refresh's host time includes waiting for
  the target's GPU tail at its first host read (the DFlash `_prefill_one` cursor reads).

Line format (per interval):
  ARC extend telemetry [60.0 s]: <batches>, <requests>, <tokens>; target gpu/cpu, refresh gpu/cpu,
  share of wall | per (requests x new-token) bin: n, mean ms target gpu / refresh gpu / target cpu /
  refresh cpu, mean new tokens, mean prefix | delayer verdicts and waits
The bins are what a cost model fits on: ms = F (per forward) + r * requests + c * new tokens.

Not covered: MIXED batches (refused, HiSparse forbids --enable-mixed-chunk anyway), non-speculative
serving (no hook), GPU idle inside a span (needs a profiler trace).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_ENABLE = "ARC_EXTEND_TELEMETRY"
ENV_INTERVAL = "ARC_EXTEND_TELEMETRY_INTERVAL_S"

TOKEN_EDGES = (512, 1024, 2048, 4096)
REQ_BUCKETS = ("k1", "k2", "k3", "k4+")
TOKEN_BUCKETS = tuple(f"<{e}" for e in TOKEN_EDGES) + (f">={TOKEN_EDGES[-1]}",)
WAIT_EDGES_S = (0.25, 0.5, 1.0, 2.0)
# Records whose events have not landed. ~1 extend per second in production: thousands pending
# means the events never complete (a broken stream), not load.
MAX_PENDING = 4096

RELEASE_REASONS = ("no_wait", "wait_success", "wait_ms_cap", "wait_timeout", "token_watermark")


def _req_bucket(k: int) -> str:
    return f"k{k}" if k <= 3 else "k4+"


def _token_bucket(n: int) -> str:
    for edge in TOKEN_EDGES:
        if n < edge:
            return f"<{edge}"
    return f">={TOKEN_EDGES[-1]}"


def batch_shape(batch) -> Tuple[int, int, int, int, int, int]:
    """(requests, new tokens, prefix tokens, max prefix, cold requests, media items in the extend)
    of an EXTEND ScheduleBatch, from host metadata only."""
    if batch.forward_mode.is_mixed():
        raise NotImplementedError(f"{ENV_ENABLE} does not attribute MIXED batches")
    reqs, extend, prefix = batch.reqs, batch.extend_lens, batch.prefix_lens
    if extend is None or prefix is None or not (len(reqs) == len(extend) == len(prefix)):
        raise RuntimeError(
            f"{ENV_ENABLE}: extend batch with {len(reqs)} requests, extend_lens "
            f"{None if extend is None else len(extend)}, prefix_lens {None if prefix is None else len(prefix)}"
        )
    if not reqs:
        raise RuntimeError(f"{ENV_ENABLE}: extend batch without requests")
    new_media = 0
    for req, start, length in zip(reqs, prefix, extend):
        mm = req.multimodal_inputs
        if mm is None:
            continue
        end = start + length
        for item in mm.mm_items:
            if item.offsets is None:
                raise RuntimeError(f"{ENV_ENABLE}: {req.rid} has a media item without offsets")
            if any(start <= off[0] < end for off in item.offsets):
                new_media += 1
    return (
        len(reqs),
        int(sum(extend)),
        int(sum(prefix)),
        int(max(prefix)),
        sum(1 for p in prefix if p == 0),
        new_media,
    )


@dataclass
class _Record:
    reqs: int
    new_tokens: int
    prefix_tokens: int
    prefix_max: int
    cold: int
    new_media: int
    t0: float
    e0: object
    t1: Optional[float] = None
    e1: Optional[object] = None
    t2: Optional[float] = None
    e2: Optional[object] = None


@dataclass
class _Bin:
    n: int = 0
    target_gpu_ms: float = 0.0
    refresh_gpu_ms: float = 0.0
    target_cpu_ms: float = 0.0
    refresh_cpu_ms: float = 0.0
    new_tokens: int = 0
    prefix_tokens: int = 0

    def add(self, other: "_Bin") -> None:
        self.n += other.n
        self.target_gpu_ms += other.target_gpu_ms
        self.refresh_gpu_ms += other.refresh_gpu_ms
        self.target_cpu_ms += other.target_cpu_ms
        self.refresh_cpu_ms += other.refresh_cpu_ms
        self.new_tokens += other.new_tokens
        self.prefix_tokens += other.prefix_tokens


@dataclass
class _Window:
    start: float
    bins: Dict[Tuple[str, str], _Bin] = field(default_factory=dict)
    reqs_by_k: Counter = field(default_factory=Counter)
    reqs: int = 0
    cold: int = 0
    new_media: int = 0
    prefix_max: int = 0
    released: Counter = field(default_factory=Counter)
    delayed_passes: int = 0
    wait_s: float = 0.0
    wait_hist: Counter = field(default_factory=Counter)


class ExtendTelemetry:
    def __init__(
        self,
        interval_s: float,
        *,
        event_factory: Callable[[], object],
        clock: Callable[[], float] = time.perf_counter,
    ):
        if not interval_s > 0:
            raise ValueError(f"{ENV_INTERVAL} must be > 0, got {interval_s}")
        self.interval_s = interval_s
        self._event = event_factory
        self._clock = clock
        self._pending: Deque[_Record] = deque()
        self._window = _Window(start=clock())
        self._thread: Optional[int] = None

    # ---- the scheduler thread only -------------------------------------------------------------------
    def _check_thread(self) -> None:
        ident = threading.get_ident()
        if self._thread is None:
            self._thread = ident
        elif self._thread != ident:
            raise RuntimeError(f"{ENV_ENABLE}: called from two threads; the recorder is single-threaded")

    def _mark(self) -> object:
        event = self._event()
        event.record()
        return event

    # ---- extend batches ------------------------------------------------------------------------------
    def begin(self, batch) -> _Record:
        self._check_thread()
        reqs, new_tokens, prefix_tokens, prefix_max, cold, new_media = batch_shape(batch)
        return _Record(reqs, new_tokens, prefix_tokens, prefix_max, cold, new_media, self._clock(), self._mark())

    def target_done(self, record: _Record) -> None:
        if record.e1 is not None:
            raise RuntimeError(f"{ENV_ENABLE}: target_done twice for one extend batch")
        record.t1, record.e1 = self._clock(), self._mark()

    def end(self, record: _Record) -> None:
        """After the drafter refresh, or right after the target when the rank hosts no drafter."""
        if record.e1 is None or record.e2 is not None:
            raise RuntimeError(f"{ENV_ENABLE}: end without target_done, or twice")
        record.t2, record.e2 = self._clock(), self._mark()
        self._pending.append(record)
        if len(self._pending) > MAX_PENDING:
            raise RuntimeError(f"{ENV_ENABLE}: {len(self._pending)} extend records whose events never landed")
        self._drain()
        self._maybe_flush()

    def _drain(self) -> None:
        w = self._window
        while self._pending and self._pending[0].e2.query():
            r = self._pending.popleft()
            key = (_req_bucket(r.reqs), _token_bucket(r.new_tokens))
            b = w.bins.setdefault(key, _Bin())
            b.add(
                _Bin(
                    n=1,
                    target_gpu_ms=r.e0.elapsed_time(r.e1),
                    refresh_gpu_ms=r.e1.elapsed_time(r.e2),
                    target_cpu_ms=1e3 * (r.t1 - r.t0),
                    refresh_cpu_ms=1e3 * (r.t2 - r.t1),
                    new_tokens=r.new_tokens,
                    prefix_tokens=r.prefix_tokens,
                )
            )
            w.reqs_by_k[_req_bucket(r.reqs)] += 1
            w.reqs += r.reqs
            w.cold += r.cold
            w.new_media += r.new_media
            w.prefix_max = max(w.prefix_max, r.prefix_max)

    # ---- prefill delayer -----------------------------------------------------------------------------
    def observe_delayer(
        self,
        *,
        output_allow: bool,
        output_reason: str,
        input_estimation: str,
        actual_execution: bool,
        wait_seconds: float,
        queue_max_delay_ms: Optional[float],
    ) -> None:
        """One scheduler pass's verdict. ``queue_max_delay_ms`` is the queue trigger's wall-clock cap
        when it is enabled: a release that reached it is reported upstream as wait_success; here it
        is wait_ms_cap."""
        self._check_thread()
        if input_estimation == "none":  # nothing was prefillable on any rank: not a decision
            self._maybe_flush()
            return
        w = self._window
        if not output_allow:
            w.delayed_passes += 1
        else:
            reason = output_reason
            if (
                reason == "wait_success"
                and queue_max_delay_ms is not None
                and 1e3 * wait_seconds >= queue_max_delay_ms
            ):
                reason = "wait_ms_cap"
            if reason not in RELEASE_REASONS:
                raise RuntimeError(f"{ENV_ENABLE}: unknown delayer release reason {output_reason!r}")
            if actual_execution:
                w.released[reason] += 1
                w.wait_s += wait_seconds
                w.wait_hist[next((f"<={e}" for e in WAIT_EDGES_S if wait_seconds <= e), f">{WAIT_EDGES_S[-1]}")] += 1
        self._maybe_flush()

    # ---- output --------------------------------------------------------------------------------------
    def _maybe_flush(self) -> None:
        now = self._clock()
        if now - self._window.start < self.interval_s:
            return
        self._drain()
        logger.info(self.format_window(self._window, now))
        self._window = _Window(start=now)

    @staticmethod
    def format_window(w: _Window, now: float) -> str:
        wall_ms = 1e3 * (now - w.start)
        total = _Bin()
        for b in w.bins.values():
            total.add(b)
        ks = " / ".join(f"{k} {w.reqs_by_k[k]}" for k in REQ_BUCKETS)
        parts: List[str] = [
            f"ARC extend telemetry [{wall_ms / 1e3:.1f} s]: {total.n} extend batches ({ks}), "
            f"{w.reqs} reqs (cold {w.cold}, new media items {w.new_media}), new {total.new_tokens} tok, "
            f"prefix {total.prefix_tokens} tok (max {w.prefix_max}); "
            f"target gpu {total.target_gpu_ms / 1e3:.2f} s cpu {total.target_cpu_ms / 1e3:.2f} s, "
            f"refresh gpu {total.refresh_gpu_ms / 1e3:.2f} s cpu {total.refresh_cpu_ms / 1e3:.2f} s, "
            f"target+refresh gpu {100 * (total.target_gpu_ms + total.refresh_gpu_ms) / wall_ms:.1f}% of wall"
        ]
        for key in sorted(w.bins, key=lambda k: (REQ_BUCKETS.index(k[0]), TOKEN_BUCKETS.index(k[1]))):
            b = w.bins[key]
            parts.append(
                f"{key[0]}/{key[1]} n={b.n} gpu={b.target_gpu_ms / b.n:.1f}/{b.refresh_gpu_ms / b.n:.1f} "
                f"cpu={b.target_cpu_ms / b.n:.1f}/{b.refresh_cpu_ms / b.n:.1f} "
                f"tok={b.new_tokens / b.n:.0f} pfx={b.prefix_tokens / b.n:.0f}"
            )
        released = sum(w.released.values())
        rel = ", ".join(f"{r} {w.released[r]}" for r in RELEASE_REASONS)
        hist = ", ".join(
            f"{k} {w.wait_hist[k]}" for k in [f"<={e}" for e in WAIT_EDGES_S] + [f">{WAIT_EDGES_S[-1]}"]
        )
        delayer = (
            f"delayer: released {released} ({rel}), delayed passes {w.delayed_passes}, "
            f"wait mean {w.wait_s / released if released else 0.0:.3f} s ({hist})"
        )
        parts.append(delayer)
        return " | ".join(parts)


_TELEMETRY: Optional[ExtendTelemetry] = None
_RESOLVED = False


def get_extend_telemetry() -> Optional[ExtendTelemetry]:
    """The process's recorder, or None when ARC_EXTEND_TELEMETRY is off. Values are checked once."""
    global _TELEMETRY, _RESOLVED
    if _RESOLVED:
        return _TELEMETRY
    flag = os.environ.get(ENV_ENABLE)
    if flag is None or flag == "0":
        if ENV_INTERVAL in os.environ:
            raise ValueError(f"{ENV_INTERVAL} set without {ENV_ENABLE}=1")
        _RESOLVED = True
        return None
    if flag != "1":
        raise ValueError(f"{ENV_ENABLE} must be 0 or 1, got {flag!r}")
    raw = os.environ.get(ENV_INTERVAL, "60")
    try:
        interval = float(raw)
    except ValueError as e:
        raise ValueError(f"{ENV_INTERVAL} must be a number, got {raw!r}") from e
    import torch

    _TELEMETRY = ExtendTelemetry(interval, event_factory=lambda: torch.cuda.Event(enable_timing=True))
    _RESOLVED = True
    logger.info("%s=1: extend batch / prefill delayer telemetry every %.0f s", ENV_ENABLE, interval)
    return _TELEMETRY
