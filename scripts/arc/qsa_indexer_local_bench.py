#!/usr/bin/env python3
"""Microbench + bitwise check of ARC_QSA_INDEXER_LOCAL (sequence-local prefill index scores).

  python qsa_indexer_local_bench.py [--shapes p10k,pmix,p45k] [--seed 0]

Per shape (the extend_profile shapes: k requests x new tokens on cached prefixes), one indexer
layer's prefill selection the way select_prefill_tokens does it (row chunks under the 128 MiB
logits budget; TileLang scores into [rows, all keys of the batch], zero fill, divide, mask kernel,
then top-k with row starts) against select_prefill_tokens_local ([rows, longest visible], one
kernel, top-k from column 0). Checks: every row's scores in its range are the same bits and the
rest of the local row is -inf; ARC_QSA_TOPK_EXACT's top-k is identical; the production fast_topk
selects the same sets (its order and tie choice are not deterministic run to run, so sets only, and
the old path against itself is printed as the floor). Times both paths (scores + top-k + expansion).
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import triton

from sglang.kernels.ops.elementwise import fast_topk as ft
from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices, qsa_fast_topk
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.attention.qsa.qsa_indexer import _qsa_prefill_row_chunk_size

RATIO, BLOCK_TOPK, HEADS, DIM = 4, 512, 4, 128
SHAPES = {
    "p10k": [(10240, 704)] * 4,
    "p10k_1408": [(10240, 1408)] * 4,
    "pmix": [(10240, 704), (20480, 704), (30720, 704), (46080, 704)],
    "p45k": [(46080, 704)] * 4,
    "k6": [(6000, 400), (9000, 500), (12000, 600), (7000, 300), (15000, 700), (8000, 350)],
}


def _inputs(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    extend = [n for _, n in shape]
    seqs = [p + n for p, n in shape]
    rows = sum(extend)
    q = torch.randn(rows, HEADS, DIM, device="cuda", generator=g).to(torch.bfloat16)
    keys = sum(s // RATIO for s in seqs)
    k = torch.randn(keys, 1, DIM, device="cuda", generator=g).to(torch.bfloat16)
    seq_t = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    tbi = torch.repeat_interleave(torch.arange(len(seqs), device="cuda", dtype=torch.int32),
                                  torch.tensor(extend, device="cuda"))
    pos = torch.cat([torch.arange(s - n, s, device="cuda") for s, n in zip(seqs, extend)])
    starts, ends, _ = build_qsa_row_ranges(seq_t, pos, tbi, RATIO)
    row_seq = seq_t.index_select(0, tbi.long())
    return q, k, starts, ends, pos, row_seq, extend, seqs


def _old(q, k, starts, ends, pos, row_seq, keep_logits=False):
    rows = q.shape[0]
    out = torch.empty((rows, BLOCK_TOPK * RATIO + RATIO - 1), dtype=torch.int32, device="cuda")
    chunk = _qsa_prefill_row_chunk_size(rows, k.shape[0], HEADS)
    kept, blocks = [], []
    for r0 in range(0, rows, chunk):
        sl = slice(r0, min(r0 + chunk, rows))
        logits = mqa.qsa_mqa_prefill(q[sl], k, starts[sl], ends[sl])
        b = qsa_fast_topk(logits, starts[sl], ends[sl], topk=BLOCK_TOPK)
        out[sl].copy_(expand_qsa_block_indices(b, pos[sl], row_seq[sl], RATIO, BLOCK_TOPK * RATIO))
        if keep_logits:
            kept.append(logits)
            blocks.append(b)
    return out, kept, blocks


def _new(q, k, starts, ends, pos, row_seq, extend, seqs, keep_logits=False):
    rows = q.shape[0]
    out = torch.empty((rows, BLOCK_TOPK * RATIO + RATIO - 1), dtype=torch.int32, device="cuda")
    lens = (ends - starts).to(torch.int32)
    block_q = 128 // HEADS
    _, width_all = mqa.qsa_prefill_local_tiles(extend, seqs, RATIO, 0, rows, block_q)
    chunk = _qsa_prefill_row_chunk_size(rows, max(width_all, 1), HEADS)
    kept, blocks = [], []
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        sl = slice(r0, r1)
        tiles, width = mqa.qsa_prefill_local_tiles(extend, seqs, RATIO, r0, r1, block_q)
        logits = mqa.tilelang_qsa_mqa_prefill_local(q[sl], k, lens[sl], tiles, width)
        b = qsa_fast_topk(logits, torch.zeros_like(lens[sl]), lens[sl], topk=BLOCK_TOPK)
        out[sl].copy_(expand_qsa_block_indices(b, pos[sl], row_seq[sl], RATIO, BLOCK_TOPK * RATIO))
        if keep_logits:
            kept.append((r0, logits))
            blocks.append(b)
    return out, kept, blocks


def _sets_equal(a, b):
    sa = torch.sort(a, dim=1).values
    sb = torch.sort(b, dim=1).values
    return (sa == sb).all(dim=1)


def run(name, shape, seed):
    q, k, starts, ends, pos, row_seq, extend, seqs = _inputs(shape, seed)
    rows = q.shape[0]
    _, old_logits, _ = _old(q, k, starts, ends, pos, row_seq, keep_logits=True)
    _, new_logits, _ = _new(q, k, starts, ends, pos, row_seq, extend, seqs, keep_logits=True)
    # scores: old is [rows, keys] in global columns (possibly in row chunks), new [rows, width] local
    old_full = torch.cat(old_logits)
    s_h, e_h = starts.tolist(), ends.tolist()
    mism = 0
    tail_bad = 0
    r = 0
    for r0, lg in new_logits:
        for i in range(lg.shape[0]):
            row = r0 + i
            n = e_h[row] - s_h[row]
            a = old_full[row, s_h[row] : e_h[row]]
            b = lg[i, :n]
            if not torch.equal(a.view(torch.int32), b.view(torch.int32)):
                mism += 1
            if n < lg.shape[1] and not torch.isneginf(lg[i, n:]).all():
                tail_bad += 1
    # top-k: exact (deterministic) and production
    ft._ARC_QSA_TOPK_EXACT = True
    o_ex, _, _ = _old(q, k, starts, ends, pos, row_seq)
    n_ex, _, _ = _new(q, k, starts, ends, pos, row_seq, extend, seqs)
    ft._ARC_QSA_TOPK_EXACT = False
    o1, _, _ = _old(q, k, starts, ends, pos, row_seq)
    o2, _, _ = _old(q, k, starts, ends, pos, row_seq)
    n1, _, _ = _new(q, k, starts, ends, pos, row_seq, extend, seqs)
    floor = float(_sets_equal(o1, o2).float().mean())
    prod = float(_sets_equal(o1, n1).float().mean())
    t_old = triton.testing.do_bench(lambda: _old(q, k, starts, ends, pos, row_seq), warmup=5, rep=50)
    t_new = triton.testing.do_bench(lambda: _new(q, k, starts, ends, pos, row_seq, extend, seqs), warmup=5, rep=50)
    print(
        f"{name:<10} rows {rows:5d} keys {k.shape[0]:6d} | scores bitwise: {rows - mism}/{rows} rows, "
        f"-inf past range: {rows - tail_bad}/{rows} | exact top-k identical: {bool(torch.equal(o_ex, n_ex))} | "
        f"production top-k same sets: old vs old {floor:.4f}, old vs local {prod:.4f} | "
        f"per layer {t_old * 1e3:7.1f} -> {t_new * 1e3:7.1f} us (x12 layers: {12 * (t_old - t_new):5.2f} ms saved)"
    )
    return mism == 0 and tail_bad == 0 and torch.equal(o_ex, n_ex)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="p10k,p10k_1408,pmix,p45k,k6")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ok = True
    for name in args.shapes.split(","):
        ok &= run(name, SHAPES[name], args.seed)
    print("✓ bitwise" if ok else "✗ NOT bitwise")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
