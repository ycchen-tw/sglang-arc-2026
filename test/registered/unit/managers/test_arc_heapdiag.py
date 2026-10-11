"""ARC_HEAPDIAG (managers/arc_heapdiag.py): the env value, the counters, and a deep dump that finds a heap-held CPU
tensor and a big Python object, then trims. Linux / glibc only (mallinfo2 is glibc >= 2.33)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers import arc_heapdiag  # noqa: E402


def test_late_seconds():
    assert arc_heapdiag.late_seconds("2400") == 2400
    for bad in ("", "0", "-5", "60s", "1.5", " 60"):
        with pytest.raises(ValueError, match="ARC_HEAPDIAG"):
            arc_heapdiag.late_seconds(bad)


def test_snapshot_and_deep_dump():
    libc = arc_heapdiag._libc()
    snap = arc_heapdiag.snapshot(libc)
    assert snap["heap_size"] >= snap["heap_rss"] > 0 and snap["rss_anon"] > 0
    assert snap["mallinfo2"]["uordblks"] > 0
    assert snap["main_arena"]["system"] > 0 and snap["thread_arenas"]["count"] >= 0

    # a CPU tensor small enough for glibc's brk heap (below the mmap threshold) and a big Python object
    keep = torch.ones(16, 1000)
    blob = [0] * 10000

    class BadSizeof:  # the L4 tree holds an object whose __sizeof__ returns a non-int: counted, not fatal
        def __sizeof__(self):
            return "n/a"

    bad = [BadSizeof() for _ in range(3)]
    out = arc_heapdiag.deep(libc, trim=True)
    assert ["BadSizeof", 3] in out["py_objects_unsized"]
    json.dumps(out)  # one log line
    assert out["torch_cpu_in_heap"]["count"] >= 1 and out["torch_cpu_in_heap"]["bytes"] >= keep.numel() * 4
    assert any(t == "list" and n >= 1 and b >= 80000 for t, n, b in out["py_objects_ge_4k"])
    assert out["trim"]["after"]["heap_size"] > 0 and isinstance(out["trim"]["released"], bool)
    del blob, bad


def test_start_is_off_without_the_env(monkeypatch):
    monkeypatch.delenv("ARC_HEAPDIAG", raising=False)
    monkeypatch.setitem(arc_heapdiag._state, "late_s", None)
    arc_heapdiag.start()
    arc_heapdiag.init_done()
    assert arc_heapdiag._state["late_s"] is None
    monkeypatch.setenv("ARC_HEAPDIAG", "soon")
    with pytest.raises(ValueError, match="ARC_HEAPDIAG"):
        arc_heapdiag.start()
