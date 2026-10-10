"""Runtime GPU memory profile of the serving (scheduler) process -- analysis runs only, it slows the host.

`ARC_DFLASH_MEM_PROFILE=<dir>` makes DFlashDraftWorker call start(<dir>) at the end of its init (before the target
sizes its pools, so the pools, graphs and every runtime allocation after that are recorded); each GPU writes to
<dir>/cuda<device index>:
  - torch.cuda.memory._record_memory_history (python stacks) for every allocation and free;
  - an out-of-memory observer that dumps the snapshot at the failing allocation (oom-<t>.pickle);
  - a sampler thread: every 0.5 s one JSON line in memlog.jsonl (torch allocated / reserved, device used from
    cudaMemGetInfo -- device used minus torch reserved is what torch does not manage: CUDA context, library
    workspaces, JIT modules); and a snapshot dump (peak.pickle, overwritten) whenever torch's high-water mark
    (max_memory_allocated) passes the last dumped one by >= 64 MiB, after the first 2 minutes of serving. A transient
    peak lasts milliseconds, so the sampler never sees it; the snapshot's event trace (the last max_entries
    allocations and frees) holds it, and mem_report.py replays the trace to the peak instant. The mark is reset once,
    2 minutes in (start-up peaks must not mask serving ones); dumps are >= 20 s apart (each holds the allocator).
mem_report.py reads both.
"""

from __future__ import annotations

import json
import os
import threading
import time

import torch

STEP = 64 << 20


def start(directory: str, device, max_entries: int = 1_000_000) -> None:
    device = torch.device(device)  # the target model runner's device is the string "cuda"
    if device.index is None:  # cudaMemGetInfo / set_device in the sampler thread need the index
        device = torch.device("cuda", torch.cuda.current_device())
    # one subdirectory per GPU: on a multi-rank host (L4 TP4) every scheduler process gets the same <dir>, and four
    # samplers appending to one memlog.jsonl / replacing one peak.pickle would mix or tear them
    directory = os.path.join(directory, f"cuda{device.index}")
    os.makedirs(directory, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=max_entries, context="all", stacks="python")

    def on_oom(dev, alloc, device_allocated, device_free):
        path = os.path.join(directory, f"oom-{int(time.time())}.pickle")
        torch.cuda.memory._dump_snapshot(path)
        with open(os.path.join(directory, "oom.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "device": dev, "alloc": alloc, "device_allocated": device_allocated,
                                "device_free": device_free, "snapshot": path}) + "\n")

    torch._C._cuda_attach_out_of_memory_observer(on_oom)
    threading.Thread(target=_sample, args=(directory, device), daemon=True, name="arc-mem-profile").start()


def _sample(directory: str, device: torch.device) -> None:
    torch.cuda.set_device(device)
    t0 = time.time()
    dumped_peak, last_dump, armed = 0, 0.0, False
    with open(os.path.join(directory, "memlog.jsonl"), "a") as log:
        while True:
            free, total = torch.cuda.mem_get_info(device)
            alloc = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            now = time.time()
            log.write(json.dumps({"t": round(now - t0, 2), "alloc": alloc, "reserved": reserved,
                                  "used": total - free, "total": total}) + "\n")
            log.flush()
            if not armed and now - t0 > 120:
                # start-up peaks (pool sizing, graph capture) must not mask the serving ones
                torch.cuda.reset_peak_memory_stats(device)
                dumped_peak, armed = torch.cuda.memory_allocated(device), True  # dump only above the serving baseline
                continue
            mark = torch.cuda.max_memory_allocated(device)
            if armed and mark >= dumped_peak + STEP and now - last_dump > 20:
                tmp = os.path.join(directory, "peak.pickle.tmp")
                torch.cuda.memory._dump_snapshot(tmp)
                os.replace(tmp, os.path.join(directory, "peak.pickle"))
                with open(os.path.join(directory, "peak.json"), "w") as f:
                    json.dump({"t": round(now - t0, 2), "high_water": mark, "alloc": alloc, "reserved": reserved,
                               "used": total - free}, f)
                dumped_peak, last_dump = mark, time.time()
            time.sleep(0.5)
