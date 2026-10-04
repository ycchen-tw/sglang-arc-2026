"""Drafter window attention straight from the ring pool (state_pool.RingStatePool), one Triton kernel per layer.

The SDPA path gathers every request's window out of the pool (`key[i][idx]`, ~1,040 columns x 2 KV heads x 256 per
layer, K and V), writes the block's own K/V into spare columns of that copy, and runs the memory-efficient kernel on
the result: at 110 requests the gathers are 1.7 ms and attention 1.6 ms per step. Here a program reads its request's
ring columns directly by slot, then the block's own keys, with the same mask as model.swa_mask plus validity:

    key j is visible to query q  iff  pos_j >= 0  and  0 <= pos_q - pos_j < window

(the ring's garbage column and never-written columns carry position -1). GQA: the query heads that share a KV head
are folded into the row axis, rows = (head in group) x (block slot), so one program reuses each K/V tile for all of
them. Online softmax in fp32, P cast to the value dtype for the PV product (as the flash/efficient kernels do).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class RingWindow:
    """Stands in for LayerCache.key / .value: the pool tensor [slots, kv heads, columns, head dim] and the batch's
    slot ids. Only the first `columns` (window + garbage column) are attended; spare columns are ignored."""

    pool: torch.Tensor
    idx: torch.Tensor
    columns: int


@triton.jit
def _ring_attn_kernel(
    Q, KP, VP, POS, IDX, KN, VN, QPOS, O,
    s_qb, s_qh, s_qt,
    s_ps, s_ph, s_pc,
    s_posb,
    s_nb, s_nh, s_nt,
    s_vb, s_vh, s_vt,
    s_qpb,
    s_ob, s_oh, s_ot,
    n, ncols, window, rows, group, scale,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NEW_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    kvh = tl.program_id(1)
    b = tl.program_id(2)
    r = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = r < rows
    t = r % n
    head = kvh * group + r // n
    d = tl.arange(0, D)
    q = tl.load(Q + b * s_qb + head[:, None] * s_qh + t[:, None] * s_qt + d[None, :], mask=row_ok[:, None], other=0.0)
    qpos = tl.load(QPOS + b * s_qpb + t, mask=row_ok, other=0)
    slot = tl.load(IDX + b)
    m_i = tl.full([BLOCK_M], -1.0e30, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    kbase = KP + slot * s_ps + kvh * s_ph
    vbase = VP + slot * s_ps + kvh * s_ph
    for start in range(0, ncols, BLOCK_N):
        c = start + tl.arange(0, BLOCK_N)
        c_ok = c < ncols
        kpos = tl.load(POS + b * s_posb + c, mask=c_ok, other=-1)
        k = tl.load(kbase + c[:, None] * s_pc + d[None, :], mask=c_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        dist = qpos[:, None] - kpos[None, :]
        ok = (kpos[None, :] >= 0) & (dist >= 0) & (dist < window) & row_ok[:, None]
        s = tl.where(ok, s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.where(ok, tl.exp(s - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vbase + c[:, None] * s_pc + d[None, :], mask=c_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    # the block's own keys (not in the pool yet)
    j = tl.arange(0, NEW_N)
    j_ok = j < n
    kpos = tl.load(QPOS + b * s_qpb + j, mask=j_ok, other=-1)
    nbase = b * s_nb + kvh * s_nh
    k = tl.load(KN + nbase + j[:, None] * s_nt + d[None, :], mask=j_ok[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
    dist = qpos[:, None] - kpos[None, :]
    ok = (kpos[None, :] >= 0) & (dist >= 0) & (dist < window) & row_ok[:, None] & j_ok[None, :]
    s = tl.where(ok, s, -1.0e30)
    m_new = tl.maximum(m_i, tl.max(s, 1))
    p = tl.where(ok, tl.exp(s - m_new[:, None]), 0.0)
    alpha = tl.exp(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    v = tl.load(VN + b * s_vb + kvh * s_vh + j[:, None] * s_vt + d[None, :], mask=j_ok[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(O + b * s_ob + head[:, None] * s_oh + t[:, None] * s_ot + d[None, :], out.to(O.dtype.element_ty),
             mask=row_ok[:, None])


# tiles measured on the G4 at 110 requests (test_ring_attention_gpu.py --bench): 32x64 for 12 x n <= 132 rows
# (K <= 6), 64x32 above (K = 8: 0.43 vs 0.51 ms per layer); bigger tiles or 3 stages run out of shared memory
TILES = {"small": (32, 64, 4, 2), "large": (64, 32, 4, 2)}
BLOCK_M = BLOCK_N = NUM_WARPS = NUM_STAGES = None  # the bench overrides the table through these


def attend(q, k, v, positions, key: RingWindow, value: RingWindow, key_positions, window):
    """q [B, Hq, n, D], k / v [B, Hkv, n, D] (this block), positions [B, n]; key / value: the ring pool;
    key_positions [B, columns] (the batch's ring positions). Returns [B, Hq, n, D]."""
    batch, hq, n, dim = q.shape
    hkv = k.shape[1]
    if key.pool.shape != value.pool.shape or key.idx is not value.idx or key.columns != value.columns:
        raise ValueError("key and value windows must be the same pool layout and batch")
    if key.pool.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("head dim must be contiguous")
    if key_positions.shape != (batch, key.columns) or positions.shape != (batch, n) or key.idx.shape != (batch,):
        raise ValueError(f"positions {tuple(key_positions.shape)} / {tuple(positions.shape)} vs batch {batch}")
    q = q.contiguous()
    out = torch.empty_like(q)
    group = hq // hkv
    rows = group * n
    bm, bn, warps, stages = TILES["large" if rows > 132 else "small"]
    if BLOCK_M is not None:
        bm, bn, warps, stages = BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES
    grid = (triton.cdiv(rows, bm), hkv, batch)
    kp, vp = key.pool, value.pool
    _ring_attn_kernel[grid](
        q, kp, vp, key_positions, key.idx, k, v, positions, out,
        q.stride(0), q.stride(1), q.stride(2),
        kp.stride(0), kp.stride(1), kp.stride(2),
        key_positions.stride(0),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),  # v is a transposed view in the model: its own strides
        positions.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        n, key.columns, window, rows, group, dim ** -0.5,
        D=dim, BLOCK_M=bm, BLOCK_N=bn, NEW_N=max(16, triton.next_power_of_2(n)),
        num_warps=warps, num_stages=stages,
    )
    return out
