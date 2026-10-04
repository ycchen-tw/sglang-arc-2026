"""arc-forge experimental-dflash d02: GPU time of the speculative decode phases, per batch size.

ARC_SPEC_PHASE_TIMING=<json path> makes EAGLEWorkerV2 record CUDA events around each decode cycle's phases:
  draft    the drafter (native MTP: draft model steps; DFlash: one drafter forward + dense q)
  verify   run_eagle_verify: target verify forward, sampling (RS or greedy), mamba commit
  post     the QSA HiSparse post-verify work (h17 backup_after_accept, h18 snapshot bookkeeping)
  extend   draft extend (native MTP: draft model forward over the accepted tokens; DFlash: records the commit)
and the start-to-start gap between consecutive decode cycles of the same batch size (anything in between -- host
scheduling, prefill batches -- shows up there). Events are resolved lazily (query), never synchronized; the
aggregate is rewritten to the JSON every 20 s. It also counts which verify sampler ran: the chain rejection-sampling
kernel or the greedy tree verify.
"""

from __future__ import annotations

import collections
import json
import os
import time

import torch

PHASES = ("draft", "verify", "post", "extend")


class PhaseTimer:
    def __init__(self, path):
        import torch.distributed as dist

        # l4: one file per rank (TP4 ranks would race on a shared .tmp / replace)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            root, ext = os.path.splitext(path)
            path = f"{root}.rank{dist.get_rank()}{ext}"
        self.path = path
        self.pending = collections.deque()
        self.stats = {}
        self.calls = collections.Counter()
        self.last_dump = time.monotonic()
        self.prev = None  # (bs, start event) of the previous decode cycle
        self._wrap_samplers()

    def _wrap_samplers(self):
        from sglang.kernels.ops.speculative import reject_sampling
        from sglang.srt.speculative import eagle_utils

        def counted(fn, name):
            def wrapper(*args, **kwargs):
                self.calls[name] += 1
                return fn(*args, **kwargs)

            return wrapper

        reject_sampling.chain_speculative_sampling_triton = counted(
            reject_sampling.chain_speculative_sampling_triton, "rejection_sampling_chain"
        )
        eagle_utils.verify_tree_greedy_func = counted(eagle_utils.verify_tree_greedy_func, "greedy_tree_verify")

    def start(self, bs):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(PHASES) + 1)]
        events[0].record()
        gap = None
        if self.prev is not None and self.prev[0] == bs:
            gap = self.prev[1]
        self.prev = (bs, events[0])
        return [bs, events, gap]

    def mark(self, record, i):
        record[1][i].record()

    def finish(self, record):
        self.pending.append(record)
        while self.pending and self.pending[0][1][-1].query():
            bs, events, gap = self.pending.popleft()
            s = self.stats.setdefault(bs, {"n": 0, **{p: 0.0 for p in PHASES}, "total": 0.0, "gap_n": 0, "gap": 0.0})
            s["n"] += 1
            for i, p in enumerate(PHASES):
                s[p] += events[i].elapsed_time(events[i + 1])
            s["total"] += events[0].elapsed_time(events[-1])
            if gap is not None:
                s["gap_n"] += 1
                s["gap"] += gap.elapsed_time(events[0])
        if time.monotonic() - self.last_dump > 20:
            self.dump()

    def dump(self):
        out = {
            "batch": {
                str(bs): {
                    "cycles": s["n"],
                    **{f"{p}_ms": s[p] / s["n"] for p in PHASES},
                    "total_ms": s["total"] / s["n"],
                    "gap_ms": s["gap"] / s["gap_n"] if s["gap_n"] else None,
                    "gap_cycles": s["gap_n"],
                }
                for bs, s in sorted(self.stats.items())
            },
            "verify_sampler_calls": dict(self.calls),
            "env": {k: os.environ.get(k) for k in ("ARC_DFLASH_DRAFTER", "ARC_DFLASH_POOL", "ARC_DFLASH_GRAPH")},
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, self.path)
        self.last_dump = time.monotonic()


_TIMER = None


def get_timer():
    global _TIMER
    path = os.environ.get("ARC_SPEC_PHASE_TIMING")
    if path and _TIMER is None:
        _TIMER = PhaseTimer(path)
    return _TIMER
