"""arc-forge ARC_DFLASH_FAST_TOPK (opt-in, default 0): the drafter head's top-k over the vocabulary as two Triton
passes instead of torch.topk.

The proposal step takes the top 16 of [rows, 124,160] BF16 logits per TP2 rank (rows = batch x 3 positions);
torch.topk's multi-block radix select (two digit-count passes and a gather) took 408 us at 384 rows on B200, twice
the lm_head GEMM in front of it; this takes 59 us. Pass 1 takes the top k of each CHUNK-wide column chunk of a row
(a tournament over GROUP-wide groups), pass 2 the top k of those candidates. Every comparison is one integer max
over a packed key -- the value's bits in an order-preserving integer form above the inverted position -- so a max
yields the largest value and, among equal values, the lowest column. Values are copied, not computed: the selected set and its order are exact, ties go to
the lower vocabulary id (torch.topk leaves the order of ties unspecified, so a tie can come out in another order
than torch's: the drafter's proposals move, the target's distribution does not -- rejection sampling).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = 4096
_CHUNK_BITS = 12  # log2(CHUNK)
GROUP = 32  # tournament group width (B200 sweep: 4096 / 32 / 1 warp best of chunk 2k-16k, group 16-128, 1-8 warps)


@triton.jit
def _key(bits):
    """16-bit float bits (int16) -> int32 in the float order (-inf < ... < -0.0 < +0.0 < ... < +inf)."""
    b = bits.to(tl.int32)
    return tl.where(b < 0, b ^ 0x7FFF, b)


@triton.jit
def _select(packed, base, OV, OI, IDS, ids_base, K: tl.constexpr, POS_BITS: tl.constexpr, FROM_TABLE: tl.constexpr,
            first_col):
    """k rounds of one int32 max over packed = (key << POS_BITS) | (MASK - pos): the largest value, the lowest
    position among equal values. Stores the value bits (int16) and the column: first_col + pos, or IDS[pos]."""
    MASK: tl.constexpr = (1 << POS_BITS) - 1
    GONE: tl.constexpr = (-32770) << POS_BITS  # below every packed key, masked ones (-32769) included
    for i in tl.static_range(K):
        best = tl.max(packed, 0)
        key = best >> POS_BITS
        tl.store(OV + base + i, tl.where(key < 0, key ^ 0x7FFF, key).to(tl.int16))
        pos = MASK - (best & MASK)
        if FROM_TABLE:
            col = tl.load(IDS + ids_base + pos)
        else:
            col = first_col + pos.to(tl.int64)
        tl.store(OI + base + i, col)
        packed = tl.where(packed == best, GONE, packed)


@triton.jit
def _topk_chunks_tournament(X, OV, OI, n_cols, s_x, n_chunks, K: tl.constexpr, CHUNK: tl.constexpr,
                            CHUNK_BITS: tl.constexpr, GROUP: tl.constexpr):
    """The chunk's top k as a tournament: the max of each GROUP-wide group, then k rounds of (max over the group
    maxima; reload the winner's group and take its max below the winner) -- O(CHUNK + k x (groups + GROUP)) where
    k rounds over the whole chunk are O(k x CHUNK) (159 vs 51 us at the served shape)."""
    MASK: tl.constexpr = CHUNK - 1
    GONE: tl.constexpr = (-32770) << CHUNK_BITS
    GROUPS: tl.constexpr = CHUNK // GROUP
    row = tl.program_id(0)
    c = tl.program_id(1)
    base_ptr = X + row.to(tl.int64) * s_x + c * CHUNK
    g = tl.arange(0, GROUPS)
    w = tl.arange(0, GROUP)
    pos = g[:, None] * GROUP + w[None, :]
    ok = c * CHUNK + pos < n_cols
    key = tl.where(ok, _key(tl.load(base_ptr + pos, mask=ok, other=0)), -32769)
    gmax = tl.max((key << CHUNK_BITS) | (MASK - pos), 1)
    out = (row * n_chunks + c) * K
    for i in tl.static_range(K):
        best = tl.max(gmax, 0)
        k_ = best >> CHUNK_BITS
        tl.store(OV + out + i, tl.where(k_ < 0, k_ ^ 0x7FFF, k_).to(tl.int16))
        p = MASK - (best & MASK)
        tl.store(OI + out + i, c.to(tl.int64) * CHUNK + p)
        winner = p // GROUP
        p32 = winner * GROUP + w
        ok32 = c * CHUNK + p32 < n_cols
        k32 = tl.where(ok32, _key(tl.load(base_ptr + p32, mask=ok32, other=0)), -32769)
        packed32 = (k32 << CHUNK_BITS) | (MASK - p32)
        nxt = tl.max(tl.where(packed32 < best, packed32, GONE), 0)
        gmax = tl.where(g == winner, nxt, gmax)


@triton.jit
def _topk_merge(CV, CI, OV, OI, n_cand, n_cols, K: tl.constexpr, BLOCK: tl.constexpr, POS_BITS: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    in_row = j < n_cand
    bits = tl.load(CV + row * n_cand + j, mask=in_row, other=0)
    col = tl.load(CI + row * n_cand + j, mask=in_row, other=n_cols)
    ok = in_row & (col < n_cols)  # a chunk with fewer than k columns pads with out-of-range ones
    key = tl.where(ok, _key(bits), -32769)
    packed = (key << POS_BITS) | ((BLOCK - 1) - j)
    _select(packed, row * K, OV, OI, CI, row * n_cand, K, POS_BITS, True, 0)


def topk(logits: torch.Tensor, k: int):
    """(values, int64 columns) of the k largest per row of 2-D BF16 / FP16 logits, descending, ties to the lower
    column. Candidates are taken chunk by chunk in column order, so the merge's lowest position among equal values
    is the lowest column."""
    if logits.ndim != 2 or logits.stride(-1) != 1:
        raise ValueError(f"fast topk: need a row-major 2-D tensor, got {tuple(logits.shape)} {logits.stride()}")
    if logits.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"fast topk: 16-bit floats only, got {logits.dtype}")
    rows, n_cols = logits.shape
    if not 1 <= k <= min(n_cols, CHUNK, 64):
        raise ValueError(f"fast topk: k {k} for {n_cols} columns")
    n_chunks = triton.cdiv(n_cols, CHUNK)
    n_cand = n_chunks * k
    block = triton.next_power_of_2(n_cand)
    pos_bits = max(1, (block - 1).bit_length())
    if pos_bits > 14:
        raise ValueError(f"fast topk: {n_cand} merge candidates")
    dev = logits.device
    cand_v = torch.empty((rows, n_cand), dtype=torch.int16, device=dev)
    cand_i = torch.empty((rows, n_cand), dtype=torch.int64, device=dev)
    values = torch.empty((rows, k), dtype=logits.dtype, device=dev)
    cols = torch.empty((rows, k), dtype=torch.int64, device=dev)
    if rows:
        _topk_chunks_tournament[(rows, n_chunks)](logits.view(torch.int16), cand_v, cand_i, n_cols, logits.stride(0),
                                                  n_chunks, K=k, CHUNK=CHUNK, CHUNK_BITS=_CHUNK_BITS, GROUP=GROUP,
                                                  num_warps=1)
        _topk_merge[(rows,)](cand_v, cand_i, values.view(torch.int16), cols, n_cand, n_cols,
                             K=k, BLOCK=block, POS_BITS=pos_bits, num_warps=4)
    return values, cols
