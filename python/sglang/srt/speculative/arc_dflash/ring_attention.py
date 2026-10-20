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

import logging
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NEW_N: tl.constexpr, FP8_DOT: tl.constexpr,
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
    if FP8_DOT:  # the query as e4m3 with one scale per row, for e4m3 x e4m3 products against the pool
        q_scale = tl.maximum(tl.max(tl.abs(q.to(tl.float32)), 1), 1e-12) / 448.0
        q8 = (q.to(tl.float32) / q_scale[:, None]).to(KP.dtype.element_ty)
        s_scale = q_scale[:, None] * scale
    for start in range(0, ncols, BLOCK_N):
        c = start + tl.arange(0, BLOCK_N)
        c_ok = c < ncols
        kpos = tl.load(POS + b * s_posb + c, mask=c_ok, other=-1)
        if FP8_DOT:
            k8 = tl.load(kbase + c[:, None] * s_pc + d[None, :], mask=c_ok[:, None], other=0.0)
            s = tl.dot(q8, tl.trans(k8)).to(tl.float32) * s_scale
        else:
            # widened to the query dtype: a no-op for a BF16 pool, exact for an FP8 one (ARC_DFLASH_KV_DTYPE)
            k = tl.load(kbase + c[:, None] * s_pc + d[None, :], mask=c_ok[:, None], other=0.0).to(q.dtype)
            s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        dist = qpos[:, None] - kpos[None, :]
        ok = (kpos[None, :] >= 0) & (dist >= 0) & (dist < window) & row_ok[:, None]
        s = tl.where(ok, s, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.where(ok, tl.exp(s - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if FP8_DOT:  # P in [0, 1] as e4m3 scaled by 256 (keeps p >= 2^-14 normal), V straight from the pool
            v8 = tl.load(vbase + c[:, None] * s_pc + d[None, :], mask=c_ok[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot((p * 256.0).to(v8.dtype), v8).to(tl.float32) * (1.0 / 256.0)
        else:
            v = tl.load(vbase + c[:, None] * s_pc + d[None, :], mask=c_ok[:, None], other=0.0).to(q.dtype)
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


def _env_tile():
    """arc-forge ARC_DFLASH_RING_TILE (opt-in; unset = TILES). "bm,bn,warps,stages": that tile for the "small" case
    (rows <= 132, every K <= 6 drafter) at every batch. "auto": per launch, the deployed 32x64 tile or one 128x64
    (8 warps, 2 stages) program per (request, KV head) -- all 84 rows of a K = 4 step, each K/V tile read once
    instead of three times -- whichever needs less time by a wave count: both run one program per SM (shared
    memory), and on B200 a 128-row program takes AUTO_RATIO x as long as a 32-row one (88 vs ~40 us at window 1024,
    dflash_propose_bench.py ring). At 148 SMs that picks 128 rows for batches <= 74 and >= 99 (batch 128: 0.231 ->
    0.177 ms per layer; 64: 0.154 -> 0.095) and keeps 32 for 75-98, where 128-row programs need two waves for about
    1.3 waves of work (96: 0.162 vs 0.175). Another tile reorders the softmax's accumulation, so the drafter's
    attention moves by rounding (not bitwise); the target's distribution is untouched (rejection sampling). The
    choice is a function of the batch and the device only, so every rank of a group picks the same tile.
    AUTO_BIG is taken only where its compiled kernel fits the device's shared memory (_big_fits): B200 (227 KB) yes;
    RTX PRO 6000 / 4090 / L4 (99-101 KB) no -- the 128x64 tile needs 148 KB there and auto keeps the deployed tile."""
    value = os.environ.get("ARC_DFLASH_RING_TILE")
    if value is None:
        return None
    if value == "auto":
        return "auto"
    parts = value.split(",")
    if len(parts) != 4 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"ARC_DFLASH_RING_TILE={value!r}: auto or bm,bn,warps,stages")
    bm, bn, warps, stages = (int(p) for p in parts)
    for name, v in (("bm", bm), ("bn", bn), ("warps", warps)):
        if v < 1 or v & (v - 1):
            raise ValueError(f"ARC_DFLASH_RING_TILE={value!r}: {name} must be a power of two")
    if bm < 16 or bn < 16 or stages < 1:
        raise ValueError(f"ARC_DFLASH_RING_TILE={value!r}: bm, bn >= 16 (tl.dot) and stages >= 1")
    return bm, bn, warps, stages


AUTO_BIG = (128, 64, 8, 2)
AUTO_RATIO = 2.2
_SMS: dict = {}
_BIG_FITS: dict = {}  # (device index, D, NEW_N) -> whether AUTO_BIG's compiled kernel fits the shared memory


def _big_fits(args, consts, grid, device) -> bool:
    """Compile (no launch) AUTO_BIG for this shape once per device and compare its shared memory with the device's
    limit; the same answer on every rank of identical GPUs."""
    key = (device.index, consts["D"], consts["NEW_N"], consts["FP8_DOT"])
    fits = _BIG_FITS.get(key)
    if fits is None:
        bm, bn, warps, stages = AUTO_BIG
        kernel = _ring_attn_kernel.warmup(*args, **consts, BLOCK_M=bm, BLOCK_N=bn, num_warps=warps,
                                          num_stages=stages, grid=grid)
        limit = triton.runtime.driver.active.utils.get_device_properties(device.index)["max_shared_mem"]
        fits = _BIG_FITS[key] = kernel.metadata.shared <= limit
        logger.info("ARC_DFLASH_RING_TILE=auto: %s needs %d B shared memory, device limit %d B -> %s",
                    AUTO_BIG, kernel.metadata.shared, limit, "eligible" if fits else "deployed tile only")
    return fits


def _auto_tile(rows, programs_per_row_tile, device):
    """ARC_DFLASH_RING_TILE=auto: the deployed small tile or AUTO_BIG, by waves x time per program."""
    small = TILES["small"]
    if rows > AUTO_BIG[0]:
        return small
    sms = _SMS.get(device)
    if sms is None:
        sms = _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    waves_small = triton.cdiv(triton.cdiv(rows, small[0]) * programs_per_row_tile, sms)
    waves_big = triton.cdiv(programs_per_row_tile, sms)
    return AUTO_BIG if waves_big * AUTO_RATIO < waves_small else small


# arc-forge ARC_DFLASH_RING_OUT_T (opt-in, default 0): attend writes its output as [B, n, Hq, D] memory and returns
# the [B, Hq, n, D] view of it, so the model's y.transpose(1, 2).flatten(-2) * sigmoid(gate) reads contiguous rows.
# Inductor's dynamic-shape kernel for that product over the [B, Hq, n, D] layout ran 68 us per layer at batch 128
# (vs 21 us); the kernel computes and rounds the same values, only the store addresses change: bitwise.
OUT_T = os.environ.get("ARC_DFLASH_RING_OUT_T", "0")
if OUT_T not in ("0", "1"):
    raise ValueError(f"ARC_DFLASH_RING_OUT_T must be 0 or 1, got {OUT_T!r}")


# arc-forge ARC_DFLASH_RING_FP8_DOT (opt-in, default 0; needs ARC_DFLASH_KV_DTYPE=fp8_e4m3): over the pool's columns,
# QK^T as e4m3 x e4m3 (the query quantized per row, one fp32 scale each) and PV as e4m3 x e4m3 (P scaled by 256),
# fp32 accumulation; the block's own K/V (BF16, not in the pool) stay BF16. Widening the pool to BF16 for BF16 MMAs
# made the e4m3 pool slower than a BF16 one (the kernel is not bandwidth-bound). Lossy beyond the pool's rounding
# (query and P are rounded too): judged by accept length.
FP8_DOT = os.environ.get("ARC_DFLASH_RING_FP8_DOT", "0")
if FP8_DOT not in ("0", "1"):
    raise ValueError(f"ARC_DFLASH_RING_FP8_DOT must be 0 or 1, got {FP8_DOT!r}")

_ENV_TILE = _env_tile()
if _ENV_TILE is not None and _ENV_TILE != "auto":
    TILES = {**TILES, "small": _ENV_TILE}


def attend(q, k, v, positions, key: RingWindow, value: RingWindow, key_positions, window):
    """q [B, Hq, n, D], k / v [B, Hkv, n, D] (this block), positions [B, n]; key / value: the ring pool;
    key_positions [B, columns] (the batch's ring positions). Returns [B, Hq, n, D]."""
    batch, hq, n, dim = q.shape
    hkv = k.shape[1]
    if key.pool.shape != value.pool.shape or key.idx is not value.idx or key.columns != value.columns:
        raise ValueError("key and value windows must be the same pool layout and batch")
    if key.pool.dtype not in (q.dtype, torch.float8_e4m3fn) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"pool {key.pool.dtype} / block {k.dtype}, {v.dtype} vs query {q.dtype}: the pool is the "
                         "query dtype or e4m3, the block's own K/V the query dtype")
    if key.pool.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("head dim must be contiguous")
    if key_positions.shape != (batch, key.columns) or positions.shape != (batch, n) or key.idx.shape != (batch,):
        raise ValueError(f"positions {tuple(key_positions.shape)} / {tuple(positions.shape)} vs batch {batch}")
    q = q.contiguous()
    if OUT_T == "1":
        out = torch.empty((batch, n, hq, dim), dtype=q.dtype, device=q.device).transpose(1, 2)
    else:
        out = torch.empty_like(q)
    group = hq // hkv
    rows = group * n
    kp, vp = key.pool, value.pool
    args = (
        q, kp, vp, key_positions, key.idx, k, v, positions, out,
        q.stride(0), q.stride(1), q.stride(2),
        kp.stride(0), kp.stride(1), kp.stride(2),
        key_positions.stride(0),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),  # v is a transposed view in the model: its own strides
        positions.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        n, key.columns, window, rows, group, dim ** -0.5,
    )
    fp8_dot = FP8_DOT == "1"
    if fp8_dot and kp.dtype != torch.float8_e4m3fn:
        raise ValueError(f"ARC_DFLASH_RING_FP8_DOT=1 needs an e4m3 pool (ARC_DFLASH_KV_DTYPE=fp8_e4m3), got {kp.dtype}")
    consts = dict(D=dim, NEW_N=max(16, triton.next_power_of_2(n)), FP8_DOT=fp8_dot)
    bm, bn, warps, stages = TILES["large" if rows > 132 else "small"]
    if _ENV_TILE == "auto" and rows <= 132:
        tile = _auto_tile(rows, hkv * batch, q.device)
        if tile == AUTO_BIG and not _big_fits(args, consts, (triton.cdiv(rows, AUTO_BIG[0]), hkv, batch), q.device):
            tile = TILES["small"]
        bm, bn, warps, stages = tile
    if BLOCK_M is not None:
        bm, bn, warps, stages = BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES
    grid = (triton.cdiv(rows, bm), hkv, batch)
    _ring_attn_kernel[grid](*args, **consts, BLOCK_M=bm, BLOCK_N=bn, num_warps=warps, num_stages=stages)
    return out
