"""ARC_HEAPDIAG: where a scheduler's glibc heap goes (l4opt 10-12; opt-in, off unless set).

On the L4 C96 deployment each scheduler's [heap] (glibc's brk main arena) holds ~3.55 GiB at plateau: ~1.4 GiB after
the weights, ~3.0 GiB when startup ends, +1.3 GiB over the first ~25 minutes of serving, then flat. Whether that is
live data or memory glibc keeps after a free decides what can be done about it, so ARC_HEAPDIAG=<seconds> logs per
scheduler:

  every 60 s        "ARC_HEAPDIAG tick {...}": [heap] size / Rss, RssAnon, glibc mallinfo2 over all arenas
                    (uordblks in use, fordblks free, keepcost trimmable top, hblkhd mmapped chunks), and from
                    malloc_info the main arena's system size and free bytes and the thread arenas' count and sums
  at init done and  "ARC_HEAPDIAG deep {...}": the tick fields, CPU torch storages grouped by the mapping that holds
  <seconds> later   them (the largest inside [heap]), Python objects of >= 4 KiB by type, then malloc_trim(0) and the
                    counters after it: what the trim gives back was free memory glibc held, the rest is live

A deep dump walks gc.get_objects() under the GIL (seconds) and the trim holds the arena locks (sub-second); the first
one runs before the event loop, the second on a timer thread while serving.
"""

from __future__ import annotations

import bisect
import ctypes
import gc
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import warnings
import weakref
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

TICK_S = 60
TOP = 12
BIG_OBJECT_BYTES = 4096
# a weakref proxy forwards every attribute (isinstance's __class__, __sizeof__) to its referent and raises once that is gone
_PROXIES = (weakref.ProxyType, weakref.CallableProxyType)
_MI2_FIELDS = ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks",
               "keepcost")
_state = {"late_s": None, "t0": None}


class _MallInfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in _MI2_FIELDS]


def _libc():
    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.restype = _MallInfo2
    libc.fopen.restype = ctypes.c_void_p
    libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.fclose.argtypes = [ctypes.c_void_p]
    libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    return libc


def late_seconds(value: str) -> int:
    """ARC_HEAPDIAG's value: the delay in seconds from init done to the second deep dump."""
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"ARC_HEAPDIAG={value!r}: the delay in seconds of the second deep dump (a positive integer)")
    return int(value)


def _maps():
    """Every mapping of this process: (lo, hi, path, Rss bytes)."""
    out, cur = [], None
    with open("/proc/self/smaps") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5 and "-" in p[0] and not p[0].endswith(":"):
                lo, hi = (int(x, 16) for x in p[0].split("-"))
                cur = [lo, hi, " ".join(p[5:]), 0]
                out.append(cur)
            elif cur is not None and p and p[0] == "Rss:":
                cur[3] = int(p[1]) * 1024
    return out


def _heap(maps):
    heap = [m for m in maps if m[2] == "[heap]"]
    if len(heap) != 1:
        raise RuntimeError(f"ARC_HEAPDIAG: expected one [heap] mapping, found {len(heap)}")
    return heap[0]


def _rss_anon():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("ARC_HEAPDIAG: no RssAnon in /proc/self/status")


def _malloc_info(libc):
    """Per malloc_info heap: [nr, system current bytes, free bytes (fast + rest)]."""
    fd, path = tempfile.mkstemp(prefix="arc_heapdiag_", suffix=".xml")
    os.close(fd)
    try:
        fp = libc.fopen(path.encode(), b"w")
        if not fp:
            raise RuntimeError(f"ARC_HEAPDIAG: fopen {path} failed")
        libc.malloc_info(0, fp)
        libc.fclose(fp)
        root = ET.parse(path).getroot()
    finally:
        os.unlink(path)
    heaps = []
    for heap in root.findall("heap"):
        system = [int(s.get("size")) for s in heap.findall("system") if s.get("type") == "current"]
        free = sum(int(t.get("size")) for t in heap.findall("total") if t.get("type") in ("fast", "rest"))
        if len(system) != 1:
            raise RuntimeError(f"ARC_HEAPDIAG: malloc_info heap {heap.get('nr')} without one current system size")
        heaps.append([int(heap.get("nr")), system[0], free])
    return heaps


def snapshot(libc, maps=None):
    maps = _maps() if maps is None else maps
    heap = _heap(maps)
    mi = libc.mallinfo2()
    heaps = _malloc_info(libc)
    main = [h for h in heaps if h[0] == 0]
    if len(main) != 1:
        raise RuntimeError(f"ARC_HEAPDIAG: malloc_info lists {len(main)} heaps numbered 0")
    threads = [h for h in heaps if h[0] != 0]
    return {"t": round(time.time() - _state["t0"], 1) if _state["t0"] is not None else None,
            "heap_size": heap[1] - heap[0], "heap_rss": heap[3], "rss_anon": _rss_anon(),
            "mallinfo2": {name: getattr(mi, name) for name in ("arena", "uordblks", "fordblks", "keepcost", "hblkhd")},
            "main_arena": {"system": main[0][1], "free": main[0][2]},
            "thread_arenas": {"count": len(threads), "system": sum(h[1] for h in threads),
                              "free": sum(h[2] for h in threads)}}


def deep(libc, trim: bool):
    maps = _maps()
    out = snapshot(libc, maps)
    t0 = time.time()
    los = [m[0] for m in maps]

    def where(ptr):
        i = bisect.bisect_right(los, ptr) - 1
        if i < 0 or ptr >= maps[i][1]:
            return "unmapped"
        lo, hi, path, _ = maps[i]
        return path or f"anon {(hi - lo) >> 20} MiB"

    torch = sys.modules.get("torch")
    storages, by_type, seen, unsized, subclass, proxies = {}, {}, set(), {}, {}, [0]
    plain = () if torch is None else (torch.Tensor, torch.nn.Parameter)

    def sized(o, t):
        """sys.getsizeof by the object's type only: no attribute of o is read (proxies, compile leftovers)."""
        if t in _PROXIES:
            proxies[0] += 1
            return -1
        size = sys.getsizeof(o, -1)  # -1: its __sizeof__ is missing or returns a non-int (seen on the L4 tree)
        if size < 0:
            unsized[t.__name__] = unsized.get(t.__name__, 0) + 1
        return size

    def count(o, size):
        e = by_type.setdefault(type(o).__name__, [0, 0])  # type(): never reads o's attributes
        e[0] += 1
        e[1] += size

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        objs = gc.get_objects()
        for o in objs:
            t = type(o)
            if torch is not None and issubclass(t, torch.Tensor):
                if t not in plain:  # FakeTensor / FunctionalTensor (torch.compile leftovers) have no data pointer
                    subclass[t.__name__] = subclass.get(t.__name__, 0) + 1
                elif o.device.type == "cpu" and o.layout == torch.strided:
                    st = o.untyped_storage()
                    if st.data_ptr():
                        storages.setdefault(st.data_ptr(), (st.nbytes(), list(o.shape), str(o.dtype)))
                continue
            size = sized(o, t)
            if size >= BIG_OBJECT_BYTES:
                count(o, size)
        for i in range(0, len(objs), 20000):  # untracked referents: bytes, str, numpy arrays, ...
            for r in gc.get_referents(*objs[i:i + 20000]):
                if gc.is_tracked(r) or id(r) in seen:
                    continue
                size = sized(r, type(r))
                if size >= BIG_OBJECT_BYTES:
                    seen.add(id(r))
                    count(r, size)
        n_objs = len(objs)
        del objs
    regions, in_heap = {}, []
    for ptr, (nbytes, shape, dtype) in storages.items():
        w = where(ptr)
        e = regions.setdefault(w, [0, 0])
        e[0] += 1
        e[1] += nbytes
        if w == "[heap]":
            in_heap.append([nbytes, shape, dtype])
    in_heap.sort(key=lambda x: -x[0])
    out.update({
        "gc_objects": n_objs,
        "torch_cpu_by_region": sorted(([k, v[0], v[1]] for k, v in regions.items()), key=lambda x: -x[2])[:TOP],
        "torch_cpu_in_heap": {"count": len(in_heap), "bytes": sum(x[0] for x in in_heap), "top": in_heap[:TOP]},
        "py_objects_ge_4k": sorted(([k, v[0], v[1]] for k, v in by_type.items()), key=lambda x: -x[2])[:TOP],
        "py_objects_unsized": sorted(([k, n] for k, n in unsized.items()), key=lambda x: -x[1])[:TOP],
        "tensor_subclasses_skipped": sorted(([k, n] for k, n in subclass.items()), key=lambda x: -x[1])[:TOP],
        "weakref_proxies_skipped": proxies[0],
        "walk_s": round(time.time() - t0, 2),
    })
    if trim:
        t1 = time.time()
        released = libc.malloc_trim(0)
        out["trim"] = {"released": bool(released), "s": round(time.time() - t1, 2), "after": snapshot(libc)}
    return out


def _ticks(libc):
    while True:
        time.sleep(TICK_S)
        logger.info("ARC_HEAPDIAG tick %s", json.dumps(snapshot(libc), separators=(",", ":")))


def _late(libc, delay):
    time.sleep(delay)
    logger.info("ARC_HEAPDIAG deep late %s", json.dumps(deep(libc, trim=True), separators=(",", ":")))


def start():
    """After the scheduler process configured its logging: the 60 s ticks, if ARC_HEAPDIAG is set."""
    value = os.environ.get("ARC_HEAPDIAG")
    if value is None:
        return
    _state["late_s"] = late_seconds(value)
    _state["t0"] = time.time()
    libc = _libc()
    logger.info("ARC_HEAPDIAG on: ticks every %d s, deep dumps + malloc_trim(0) at init done and %d s later",
                TICK_S, _state["late_s"])
    threading.Thread(target=_ticks, args=(libc,), name="arc-heapdiag-tick", daemon=True).start()


def init_done():
    """Right before the event loop: the first deep dump and trim, and the timer of the second."""
    if _state["late_s"] is None:
        return
    libc = _libc()
    logger.info("ARC_HEAPDIAG deep init %s", json.dumps(deep(libc, trim=True), separators=(",", ":")))
    threading.Thread(target=_late, args=(libc, _state["late_s"]), name="arc-heapdiag-late", daemon=True).start()
