"""ARC_HC_GATE_PRECOMPUTE parity + timing: hc_gate (side stream) + hc_apply == hc_combine, bit for bit.

The serving path splits each >32-row hyper-connection combine into hc_gate (the combine's phase 1, computed on a
side stream while the attention / MoE block runs) and hc_apply (phase 2, after the block's all-reduce). This
checks, per dtype and row count, that the two-kernel result is the same bytes as hc_combine's -- including when
hc_gate runs on another stream and the pair is captured in a CUDA graph -- and times what is left on the critical
path (hc_apply) against the combine it replaces.

  python scripts/arc/test_hc_gate_apply.py            # prints HC_GATE_APPLY_OK
"""

import torch

from sglang.kernels.ops.elementwise.hc_combine import hc_combine
from sglang.kernels.ops.elementwise.hc_gate import hc_apply, hc_gate

HC, H = 4, 2560
ROWS = (1, 7, 32, 33, 64, 100, 128, 476, 512, 2816, 8192)


def _inputs(m, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    y = torch.randn(m, H, dtype=dtype, device="cuda", generator=g)
    r = torch.randn(m, HC * H, dtype=dtype, device="cuda", generator=g)
    n = torch.randn(m, HC * H, dtype=dtype, device="cuda", generator=g)
    w = torch.randn(HC, HC * H, dtype=dtype, device="cuda", generator=g) * 0.02
    return y, r, n, w


def _same(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def parity():
    for dtype in (torch.bfloat16, torch.float16):
        for m in ROWS:
            y, r, n, w = _inputs(m, dtype, 1000 + m)
            ref = hc_combine(y, r, n, w, HC, H)
            gate = hc_gate(n, w, HC, H)
            out = hc_apply(y, r, gate, HC, H)
            if not _same(out, ref):
                diff = (out.float() - ref.float()).abs()
                raise RuntimeError(f"{dtype} m={m}: hc_gate+hc_apply != hc_combine, max diff {diff.max().item()}")
            # 3-D residual shapes as the model passes them
            out3 = hc_apply(y.view(m, 1, H), r.view(m, 1, HC * H), gate, HC, H)
            if out3.shape != (m, 1, HC * H) or not _same(out3.view(m, HC * H), ref):
                raise RuntimeError(f"{dtype} m={m}: 3-D shapes differ")
        print(f"parity {dtype}: rows {ROWS} bitwise")


def graph_two_streams():
    """The serving pattern: gate on a side stream forked from / joined to the main stream, captured in a graph."""
    side = torch.cuda.Stream()
    m = 128
    y, r, n, w = _inputs(m, torch.bfloat16, 7)
    ref = hc_combine(y, r, n, w, HC, H)
    busy = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

    def step():
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            gate = hc_gate(n, w, HC, H)
        n.record_stream(side)
        busy2 = busy @ busy  # stands in for the attention block running concurrently
        cur.wait_stream(side)
        gate.record_stream(cur)
        return hc_apply(y, r, gate, HC, H), busy2

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = step()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    if not _same(out, ref):
        raise RuntimeError("graph two-stream replay differs from hc_combine")
    print("graph two-stream: bitwise")


def _time(fn, reps=200):
    for _ in range(10):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (reps * 20)


def timing():
    print(f"{'rows':>6} {'combine_us':>10} {'gate_us':>8} {'apply_us':>9} {'saved_us':>9}")
    for m in (33, 64, 96, 128, 256, 512):
        y, r, n, w = _inputs(m, torch.bfloat16, m)
        gate = hc_gate(n, w, HC, H)
        out = torch.empty_like(r)
        t_c = _time(lambda: hc_combine(y, r, n, w, HC, H))
        t_g = _time(lambda: hc_gate(n, w, HC, H, out=gate))
        t_a = _time(lambda: hc_apply(y, r, gate, HC, H, out=out))
        print(f"{m:>6} {t_c:>10.2f} {t_g:>8.2f} {t_a:>9.2f} {t_c - t_a:>9.2f}")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    parity()
    graph_two_streams()
    timing()
    print("HC_GATE_APPLY_OK")
