"""arc-forge h24: the ReplaySSM kernels of the h20 path on a packed GDN state (srt/.../linear/gdn_state_fmt.py).

gdn_replayssm_spec_circular_packed_kernel is gdn_replayssm_spec_circular_kernel restricted to the path h20 runs
(FOLD_PENDING, one verify launch, 16-bit compact records with low parts) with the checkpoint read from and written to
the packed planes: the fold dequantizes, folds with the deployed fp32 arithmetic, requantizes (one scale per key
channel and 64 value rows = the program's value tile) and stores; the projection reads back what was stored.
gdn_replayssm_track_packed_kernel is the h20 commit (track rows only) on the packed state.
The packed state is one uint8 pool; a (slot, head) block of NB bytes starts at slot * stride_slot + head * NB and holds
hi (int8 [V, K]) at 0, lo ([V, K * LO_BITS / 8]) at O_LO and the fp32 scales ([V / 64, K]) at O_SC. Byte j of a lo
row holds key channel i * (K * LO_BITS / 8) + j in bit field i, so a block of FOLD_BK = 32 key channels owns one bit
field of 32 contiguous bytes: the fold reads those bytes once, rewrites every field of them in turn and stores them
once, and the projection (32-channel key tiles) reads contiguous bytes and shifts by a per-tile constant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.linear.gdn_state_fmt import GROUP_V, StateFmt


@triton.jit
def _fold_state_blocks(p_src, p_dst, p_keys, p_key_residual, row_mask, upd_hi, upd_lo, decay, o_v, i_v,
                       K: tl.constexpr, BV: tl.constexpr, FOLD_BK: tl.constexpr, O_LO: tl.constexpr,
                       O_SC: tl.constexpr, LO_BITS: tl.constexpr, QMAX: tl.constexpr, IN_PLACE: tl.constexpr):
    """dst = src * decay + sum over the rows of key (x) update, for the value tile i_v of one (slot, head) block,
    requantized: per key channel one scale over the tile's value rows. p_keys / p_key_residual: [rows, 1] pointers to
    the normalized keys (high / low parts); upd_hi / upd_lo: [rows, BV] decayed updates; row_mask: the live rows.
    IN_PLACE: p_src == p_dst (each block is read before it is written; blocks are disjoint)."""
    PER: tl.constexpr = 8 // LO_BITS
    NBYTE: tl.constexpr = K // PER  # lo bytes per value row
    JG: tl.constexpr = NBYTE // FOLD_BK  # byte-column groups of FOLD_BK bytes
    MASK: tl.constexpr = (1 << LO_BITS) - 1
    for jg in tl.static_range(JG):
        o_j = jg * FOLD_BK + tl.arange(0, FOLD_BK)
        p_lo_src = p_src + O_LO + o_v[None, :] * NBYTE + o_j[:, None]
        p_lo_dst = p_dst + O_LO + o_v[None, :] * NBYTE + o_j[:, None]
        for i in tl.static_range(PER):
            o_kf = (i * JG + jg) * FOLD_BK + tl.arange(0, FOLD_BK)
            keys = tl.load(p_keys + o_kf[None, :], mask=row_mask[:, None], other=0.0)
            key_res = tl.load(p_key_residual + o_kf[None, :], mask=row_mask[:, None], other=0.0)
            delta = tl.dot(tl.trans(keys), upd_hi)
            delta += tl.dot(tl.trans(keys), upd_lo)
            delta += tl.dot(tl.trans(key_res), upd_hi)
            delta += tl.dot(tl.trans(key_res), upd_lo)
            hi_old = tl.load((p_src + o_v[None, :] * K + o_kf[:, None]).to(tl.pointer_type(tl.int8))).to(tl.int32)
            s_old = tl.load((p_src + O_SC + (i_v * K + o_kf) * 4).to(tl.pointer_type(tl.float32)))
            # field i of the source bytes is still the source's (earlier iterations rewrote other fields only)
            lo_src = tl.load(p_lo_src).to(tl.int32)
            q_old = hi_old * (1 << LO_BITS) + ((lo_src >> (i * LO_BITS)) & MASK)
            x = q_old.to(tl.float32) * s_old[:, None] * decay + delta
            amax = tl.max(tl.abs(x), axis=1)
            ok = amax > QMAX * 1e-30
            inv = tl.where(ok, QMAX / amax, 0.0)
            q = tl.floor(x * inv[:, None] + 0.5)
            q = tl.minimum(tl.maximum(q, -QMAX), QMAX).to(tl.int32)
            tl.store((p_dst + o_v[None, :] * K + o_kf[:, None]).to(tl.pointer_type(tl.int8)),
                     (q >> LO_BITS).to(tl.int8))
            tl.store((p_dst + O_SC + (i_v * K + o_kf) * 4).to(tl.pointer_type(tl.float32)),
                     tl.where(ok, amax / QMAX, 0.0))
            # rewrite field i of the destination bytes, keep the others (in place: the source's, still to be read)
            if IN_PLACE:
                lo_dst = lo_src
            else:
                lo_dst = tl.load(p_lo_dst).to(tl.int32)
            lo_dst = (lo_dst & ~(MASK << (i * LO_BITS))) | ((q & MASK) << (i * LO_BITS))
            tl.store(p_lo_dst, lo_dst.to(tl.uint8))
            tl.debug_barrier()


@triton.jit
def _deq_tile(p_blk, o_v, kk, i_v, K: tl.constexpr, BKT: tl.constexpr, O_LO: tl.constexpr, O_SC: tl.constexpr,
              LO_BITS: tl.constexpr):
    """[BV value rows, BKT = 32 key channels] of key tile kk, dequantized (fp32)."""
    PER: tl.constexpr = 8 // LO_BITS
    NBYTE: tl.constexpr = K // PER
    JG: tl.constexpr = NBYTE // BKT
    o_kt = kk * BKT + tl.arange(0, BKT)
    field = kk // JG
    o_j = (kk % JG) * BKT + tl.arange(0, BKT)
    hi = tl.load((p_blk + o_v[:, None] * K + o_kt[None, :]).to(tl.pointer_type(tl.int8))).to(tl.int32)
    lo = tl.load(p_blk + O_LO + o_v[:, None] * NBYTE + o_j[None, :]).to(tl.int32)
    q = hi * (1 << LO_BITS) + ((lo >> (field * LO_BITS)) & ((1 << LO_BITS) - 1))
    s = tl.load((p_blk + O_SC + (i_v * K + o_kt) * 4).to(tl.pointer_type(tl.float32)))
    return q.to(tl.float32) * s[None, :]


@triton.jit
def gdn_replayssm_spec_circular_packed_kernel(
    q, k, v, a, b, A_log, dt_bias, o,
    st,  # packed state pool, uint8, per layer [slots, HV, NB]
    d_cache, k_cache, g_cache, rawv_cache, rawk_cache,
    query_start_loc, ssm_state_indices, replay_indices, write_pos, cache_base, is_flush_flags,
    scale,
    stride_q_t: tl.constexpr, stride_k_t: tl.constexpr, stride_v_t: tl.constexpr, stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr, stride_o_t: tl.constexpr,
    stride_slot: tl.constexpr,
    stride_d_slot: tl.constexpr, stride_k_slot: tl.constexpr, stride_g_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr, stride_rawk_slot: tl.constexpr,
    stride_qsl: tl.constexpr, stride_indices: tl.constexpr, stride_replay_indices: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    BS: tl.constexpr, BC: tl.constexpr, NK: tl.constexpr, BKT: tl.constexpr, MAX_CACHE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr, NULL_BLOCK_ID: tl.constexpr, FOLD_BK: tl.constexpr,
    NB: tl.constexpr, O_LO: tl.constexpr, O_SC: tl.constexpr, LO_BITS: tl.constexpr, QMAX: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    bos = tl.load(query_start_loc + i_n * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc + (i_n + 1) * stride_qsl).to(tl.int64)
    spec_len = eos - bos
    mask_s = o_s < spec_len
    o_s_safe = tl.where(mask_s, o_s, 0)
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    replay_idx = tl.load(replay_indices + i_n * stride_replay_indices).to(tl.int64)
    p_o = o + (bos + o_s_safe[:, None]) * stride_o_t + i_hv * V + o_v[None, :]
    if state_idx <= NULL_BLOCK_ID:
        tl.store(p_o, tl.zeros([BS, BV], dtype=tl.float32).to(p_o.dtype.element_ty),
                 mask=(o_s < spec_len)[:, None] & mask_v[None, :])
        return
    b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
    b_write_pos = tl.load(write_pos + replay_idx).to(tl.int64)
    b_cache_base = tl.load(cache_base + replay_idx).to(tl.int32)
    out_mask = mask_s[:, None] & mask_v[None, :]
    b_wp_i = b_write_pos.to(tl.int32)
    n_fold = tl.where(b_is_flush, b_wp_i, 0)
    cache_valid = o_c < tl.where(b_is_flush, 0, b_write_pos)
    phys_c = (b_cache_base + o_c) & (MAX_CACHE_LEN - 1)
    phys_spec = (b_cache_base + b_wp_i + o_s) & (MAX_CACHE_LEN - 1)
    p_blk = st + state_idx * stride_slot + i_hv * NB

    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    a_s = tl.load(a + (bos + o_s_safe) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(tl.float32)
    b_s = tl.load(b + (bos + o_s_safe) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(tl.float32)
    x = a_s + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_s)), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    # committed history (non-flush rows; empty on the steady h20 path)
    p_g_main = g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c
    b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
    b_g_prefix = tl.cumsum(b_g_all, axis=0)
    b_g_total = tl.sum(b_g_all, axis=0)
    b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
    b_total_decay = tl.exp(b_g_total)
    p_d_main = d_cache + (replay_idx * stride_d_slot + (i_hv * MAX_CACHE_LEN + phys_c[None, :]) * V + o_v[:, None])
    b_d_all = tl.load(p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0.0).to(tl.float32)
    b_d_scaled = (b_d_all * b_replay_decay[None, :]).to(d_cache.dtype.element_ty)

    qnorm_acc = tl.zeros([BS], dtype=tl.float32)
    knorm_acc = tl.zeros([BS], dtype=tl.float32)
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        ld = mask_s[:, None] & mask_kt[None, :]
        qn = tl.load(q + (bos + o_s_safe[:, None]) * stride_q_t + i_h * K + o_kt[None, :], mask=ld, other=0.0).to(tl.float32)
        knn = tl.load(k + (bos + o_s_safe[:, None]) * stride_k_t + i_h * K + o_kt[None, :], mask=ld, other=0.0).to(tl.float32)
        qnorm_acc += tl.sum(qn * qn, axis=1)
        knorm_acc += tl.sum(knn * knn, axis=1)
    q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-6), 0.0)
    k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-6), 0.0)

    # fold the pending window (h20's arithmetic), requantize, store
    fold_mask = o_c < n_fold
    fold_g = tl.load(g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c, mask=fold_mask,
                     other=0.0).to(tl.float32)
    fold_g = tl.where(fold_mask, fold_g, 0.0)
    fold_prefix = tl.cumsum(fold_g, axis=0)
    fold_total = tl.sum(fold_g, axis=0)
    fold_decay = tl.where(fold_mask, tl.exp(fold_total - fold_prefix), 0.0)
    fold_scale = tl.exp(fold_total)
    p_fold_d = (i_hv * MAX_CACHE_LEN + phys_c[:, None]) * V + o_v[None, :]
    fold_updates = tl.load(d_cache + replay_idx * stride_d_slot + p_fold_d, mask=fold_mask[:, None] & mask_v[None, :],
                           other=0.0).to(tl.float32)
    fold_updates += tl.load(rawv_cache + replay_idx * stride_rawv_slot + p_fold_d,
                            mask=fold_mask[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
    fold_updates = fold_updates * fold_decay[:, None]
    fold_hi = fold_updates.to(k_cache.dtype.element_ty)
    fold_lo = (fold_updates - fold_hi.to(tl.float32)).to(k_cache.dtype.element_ty)
    if b_is_flush:
        p_fold_keys = k_cache + replay_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
        p_fold_key_residual = rawk_cache + replay_idx * stride_rawk_slot + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
        _fold_state_blocks(p_blk, p_blk, p_fold_keys, p_fold_key_residual, fold_mask, fold_hi, fold_lo, fold_scale,
                           o_v, i_v, K, BV, FOLD_BK, O_LO, O_SC, LO_BITS, QMAX, True)
        tl.debug_barrier()

    hw_q = tl.zeros([BV, BS], dtype=tl.float32)
    hw_k = tl.zeros([BV, BS], dtype=tl.float32)
    scores_q = tl.zeros([BC, BS], dtype=tl.float32)
    scores_k = tl.zeros([BC, BS], dtype=tl.float32)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)
    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        q_tile = tl.load(q + (bos + o_s_safe[:, None]) * stride_q_t + i_h * K + o_kt[None, :], mask=mask_kt[None, :],
                         other=0.0).to(tl.float32)
        k_tile = tl.load(k + (bos + o_s_safe[:, None]) * stride_k_t + i_h * K + o_kt[None, :], mask=mask_kt[None, :],
                         other=0.0).to(tl.float32)
        q_precise = q_tile * (q_rnorm * scale)[:, None]
        k_precise = k_tile * k_rnorm[:, None]
        k_store_tile = k_precise.to(k_cache.dtype.element_ty)
        k_residual_tile = (k_precise - k_store_tile.to(tl.float32)).to(k.dtype.element_ty)
        q_tile = q_precise.to(q.dtype.element_ty)
        k_tile = k_store_tile
        sc_tile = _deq_tile(p_blk, o_v, kk, i_v, K, BKT, O_LO, O_SC, LO_BITS)
        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT)
        kq_mat += tl.dot(k_tile, qT)
        k_residual_T = tl.trans(k_residual_tile)
        kk_mat += tl.dot(k_tile, k_residual_T)
        kk_mat += tl.dot(k_residual_tile, kT)
        kk_mat += tl.dot(k_residual_tile, k_residual_T)
        kq_mat += tl.dot(k_residual_tile, qT)
        sc_tile = sc_tile.to(k_tile.dtype)
        hw_q += tl.dot(sc_tile, qT)
        hw_k += tl.dot(sc_tile, kT)
        hw_k += tl.dot(sc_tile, k_residual_T)
        p_k = k_cache + (replay_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K + o_kt[None, :])
        khist_tile = tl.load(p_k, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0).to(k_cache.dtype.element_ty)
        scores_q += tl.dot(khist_tile, qT)
        scores_k += tl.dot(khist_tile, kT)
        if write_k:
            spec_kt_mask = mask_s[:, None] & mask_kt[None, :] & ((b_write_pos + o_s[:, None]) < MAX_CACHE_LEN)
            p_cur_rawk = rawk_cache + (replay_idx * stride_rawk_slot + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                                       + o_kt[None, :])
            tl.store(p_cur_rawk, k_residual_tile, mask=spec_kt_mask)
            p_cur_k = k_cache + (replay_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                                 + o_kt[None, :])
            tl.store(p_cur_k, k_store_tile, mask=spec_kt_mask)

    hw_q = b_total_decay * hw_q + tl.dot(b_d_scaled, scores_q.to(b_d_scaled.dtype))
    hw_k = b_total_decay * hw_k + tl.dot(b_d_scaled, scores_k.to(b_d_scaled.dtype))

    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)
    b_Ai = -A_mat
    for ii in range(2, BS):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)
    p_v = v + (bos + o_s_safe[None, :]) * stride_v_t + i_hv * V + o_v[:, None]
    v_tile = tl.load(p_v, mask=mask_v[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    D_spec = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]
    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF
    tl.store(p_o, tl.trans(O_tile).to(p_o.dtype.element_ty), mask=out_mask)

    spec_pos = b_write_pos + o_s
    spec_store_mask = mask_s & (spec_pos < MAX_CACHE_LEN)
    p_cur_d = d_cache + (replay_idx * stride_d_slot + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V + o_v[:, None])
    d_hi = D_spec.to(p_cur_d.dtype.element_ty)
    tl.store(p_cur_d, d_hi, mask=mask_v[:, None] & spec_store_mask[None, :])
    p_cur_v = rawv_cache + (replay_idx * stride_rawv_slot + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
                            + o_v[:, None])
    tl.store(p_cur_v, (D_spec - d_hi.to(tl.float32)).to(p_cur_v.dtype.element_ty),
             mask=mask_v[:, None] & spec_store_mask[None, :])
    if i_v == 0:
        p_cur_g = g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
        tl.store(p_cur_g, g_s, mask=spec_store_mask)


@triton.jit
def gdn_replayssm_track_packed_kernel(
    st,  # packed state pool, uint8 [layers, slots, HV, NB]
    d_cache, k_cache, g_cache, d_residual_cache, k_residual_cache,
    ssm_state_indices, replay_indices, write_pos, cache_base, accept_lens, mamba_track_indices, mamba_steps_to_track,
    stride_slot: tl.constexpr, stride_d_slot: tl.constexpr, stride_k_slot: tl.constexpr, stride_g_slot: tl.constexpr,
    stride_d_residual_slot: tl.constexpr, stride_k_residual_slot: tl.constexpr,
    stride_layer: tl.constexpr, stride_d_layer: tl.constexpr, stride_k_layer: tl.constexpr,
    stride_g_layer: tl.constexpr, stride_d_residual_layer: tl.constexpr, stride_k_residual_layer: tl.constexpr,
    stride_indices: tl.constexpr, stride_replay_indices: tl.constexpr, stride_accept: tl.constexpr,
    stride_track: tl.constexpr, stride_steps: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr, NULL_BLOCK_ID: tl.constexpr, FOLD_BK: tl.constexpr,
    NB: tl.constexpr, O_LO: tl.constexpr, O_SC: tl.constexpr, LO_BITS: tl.constexpr, QMAX: tl.constexpr,
):
    """h20's commit on the packed state: only rows with a track step, their track slot = the checkpoint (which the
    next verify has not folded yet) plus the accepted prefix up to the track step, as
    gdn_replayssm_compact_commit_kernel(FOLD_IN_VERIFY, LIVE_ROWS) computes it."""
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    replay_idx = tl.load(replay_indices + i_n * stride_replay_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
    track_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    if (track_idx <= NULL_BLOCK_ID) or (track_step < 0):
        return
    n_history = tl.load(write_pos + replay_idx).to(tl.int32)
    n_accepted = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    track_len = n_history - n_accepted + track_step + 1

    st += i_layer * stride_layer
    d_cache += i_layer * stride_d_layer
    k_cache += i_layer * stride_k_layer
    g_cache += i_layer * stride_g_layer
    d_residual_cache += i_layer * stride_d_residual_layer
    k_residual_cache += i_layer * stride_k_residual_layer
    base = tl.load(cache_base + replay_idx).to(tl.int32)
    o_t = tl.arange(0, MAX_CACHE_LEN)
    phys = (base + o_t) & (MAX_CACHE_LEN - 1)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_t = o_t < track_len

    p_src = st + state_idx * stride_slot + i_hv * NB
    p_dst = st + track_idx * stride_slot + i_hv * NB
    p_keys = k_cache + replay_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + phys[:, None]) * K
    p_key_residual = k_residual_cache + replay_idx * stride_k_residual_slot + (i_h * MAX_CACHE_LEN + phys[:, None]) * K
    updates = tl.load(d_cache + replay_idx * stride_d_slot + (i_hv * MAX_CACHE_LEN + phys[:, None]) * V + o_v[None, :],
                      mask=mask_t[:, None], other=0.0).to(tl.float32)
    updates += tl.load(d_residual_cache + replay_idx * stride_d_residual_slot
                       + (i_hv * MAX_CACHE_LEN + phys[:, None]) * V + o_v[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
    gates = tl.load(g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys, mask=mask_t,
                    other=0.0).to(tl.float32)
    track_g = tl.where(mask_t, gates, 0.0)
    track_prefix = tl.cumsum(track_g, axis=0)
    track_total = tl.sum(track_g, axis=0)
    track_decay = tl.where(mask_t, tl.exp(track_total - track_prefix), 0.0)
    track_updates = updates * track_decay[:, None]
    track_hi = track_updates.to(k_cache.dtype.element_ty)
    track_lo = (track_updates - track_hi.to(tl.float32)).to(k_cache.dtype.element_ty)
    _fold_state_blocks(p_src, p_dst, p_keys, p_key_residual, mask_t, track_hi, track_lo, tl.exp(track_total), o_v,
                       i_v, K, BV, FOLD_BK, O_LO, O_SC, LO_BITS, QMAX, False)


def _fmt_consts(fmt: StateFmt, v: int, k: int) -> dict:
    _, o_lo, o_sc = fmt.offsets(v, k)
    return dict(NB=fmt.head_bytes(v, k), O_LO=o_lo, O_SC=o_sc, LO_BITS=fmt.lo_bits, QMAX=float(fmt.qmax))


def launch_spec_verify_packed(*, fmt, q, k, v, a, b, A_log, dt_bias, out, state, d_cache, k_cache, g_cache,
                              rawv_cache, rawk_cache, query_start_loc, ssm_state_indices, replay_indices, write_pos,
                              cache_base, is_flush, scale, max_cache_len, max_spec_len, null_block_id, fold_bk,
                              block_v=64, num_warps=1, num_stages=2, bs_min=4):
    """h20's verify launch (fold the pending window, then project) on a per-layer packed state [slots, HV, NB]."""
    H, K = k.shape[1], k.shape[2]
    HV, V = v.shape[1], v.shape[2]
    if block_v != GROUP_V:
        raise ValueError(f"the packed state's value tile is {GROUP_V}, got block_v={block_v}")
    if state.dtype != torch.uint8 or state.dim() != 3 or state.shape[1:] != (HV, fmt.head_bytes(V, K)):
        raise ValueError(f"not a per-layer {fmt.name} state: {state.dtype} {tuple(state.shape)}")
    if state.stride(1) != state.shape[2] or state.stride(2) != 1:
        raise ValueError("packed state heads must be contiguous")
    if rawv_cache is None or rawk_cache is None:
        raise ValueError("the packed verify needs the compact D/K low-part rings")
    if max_cache_len < 16 or max_cache_len & (max_cache_len - 1):
        raise ValueError(f"ring length {max_cache_len} must be a power of two >= 16")
    BK = triton.next_power_of_2(K)
    if BK != K or fold_bk != 32:
        raise ValueError(f"the packed verify takes K a power of two and 32-channel key blocks, got K={K}, {fold_bk}")
    nk = K // 32  # 32-channel projection tiles: one lo bit field each
    B = query_start_loc.shape[0] - 1
    gdn_replayssm_spec_circular_packed_kernel[(triton.cdiv(V, block_v), B, HV)](
        q, k, v, a, b, A_log, dt_bias, out, state, d_cache, k_cache, g_cache, rawv_cache, rawk_cache,
        query_start_loc, ssm_state_indices, replay_indices, write_pos, cache_base, is_flush, scale,
        q.stride(0), k.stride(0), v.stride(0), a.stride(0), b.stride(0), out.stride(0),
        state.stride(0), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0), rawv_cache.stride(0),
        rawk_cache.stride(0), query_start_loc.stride(0), ssm_state_indices.stride(0), replay_indices.stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=block_v, BS=max(bs_min, triton.next_power_of_2(max_spec_len)),
        BC=max(16, triton.next_power_of_2(max_cache_len)), NK=nk, BKT=BK // nk, MAX_CACHE_LEN=max_cache_len,
        SOFTPLUS_THRESHOLD=20.0, NULL_BLOCK_ID=null_block_id, FOLD_BK=fold_bk, **_fmt_consts(fmt, V, K),
        num_warps=num_warps, num_stages=num_stages)


def launch_track_packed(*, fmt, state, d_cache, k_cache, g_cache, d_residual_cache, k_residual_cache,
                        state_batch_indices, replay_indices, write_pos, cache_base, accept_lens, mamba_track_indices,
                        mamba_steps_to_track, null_block_id, v_dim, k_dim):
    """h20's commit (track rows only) on the packed state [layers, slots, HV, NB]."""
    num_layers, _, HV, nb = state.shape
    if nb != fmt.head_bytes(v_dim, k_dim) or state.stride(2) != nb or state.stride(3) != 1:
        raise ValueError(f"not a contiguous {fmt.name} pool: {tuple(state.shape)} {state.stride()}")
    if d_residual_cache is None or k_residual_cache is None:
        raise ValueError("the packed commit needs the compact D/K low-part rings")
    H = k_cache.shape[2]
    B = state_batch_indices.shape[0]
    BV = GROUP_V
    gdn_replayssm_track_packed_kernel[(triton.cdiv(v_dim, BV), B, HV * num_layers)](
        state, d_cache, k_cache, g_cache, d_residual_cache, k_residual_cache,
        state_batch_indices, replay_indices, write_pos, cache_base, accept_lens, mamba_track_indices,
        mamba_steps_to_track,
        state.stride(1), d_cache.stride(1), k_cache.stride(1), g_cache.stride(1), d_residual_cache.stride(1),
        k_residual_cache.stride(1), state.stride(0), d_cache.stride(0), k_cache.stride(0), g_cache.stride(0),
        d_residual_cache.stride(0), k_residual_cache.stride(0), state_batch_indices.stride(0),
        replay_indices.stride(0), accept_lens.stride(0), mamba_track_indices.stride(0),
        mamba_steps_to_track.stride(0),
        H=H, HV=HV, K=k_dim, V=v_dim, BV=BV, MAX_CACHE_LEN=d_cache.shape[-2], NULL_BLOCK_ID=null_block_id,
        FOLD_BK=32, **_fmt_consts(fmt, v_dim, k_dim), num_warps=4, num_stages=2)


# ---------------------------------------------------------------------------------------------------------------
# The basis change and the prefill side (one launch each; the torch versions in gdn_state_fmt are the references)


@triton.jit
def _rotate_v_kernel(x, y, h, R, stride_x, stride_y, G: tl.constexpr, BR: tl.constexpr):
    """y[r, c0:c0 + G] = x[r, c0:c0 + G] @ H for the column block c0 = program_id(1) * G. H = H^T = H^-1 has entries
    +-1/sqrt(G) (exact in x's dtype for G = 64), so every product is exact and the sum is fp32. y may be x."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    o_r = (pid_r * BR + tl.arange(0, BR)).to(tl.int64)
    o_g = tl.arange(0, G)
    o_c = pid_c * G + o_g
    m = (o_r < R)[:, None]
    xt = tl.load(x + o_r[:, None] * stride_x + o_c[None, :], mask=m, other=0.0)
    ht = tl.load(h + o_g[:, None] * G + o_g[None, :])
    yt = tl.dot(xt, ht, out_dtype=tl.float32)
    tl.store(y + o_r[:, None] * stride_y + o_c[None, :], yt.to(y.dtype.element_ty), mask=m)


@triton.jit
def _gather_dequant_kernel(st, idx, out, n_slots, stride_slot, stride_out,
                           V: tl.constexpr, K: tl.constexpr, NB: tl.constexpr, O_LO: tl.constexpr,
                           O_SC: tl.constexpr, LO_BITS: tl.constexpr, BV: tl.constexpr):
    """out[n, head, value tile] = the packed block of slot idx[n] (negative: from the end, as torch indexing),
    dequantized: q * scale in fp32, then out's dtype."""
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_h = tl.program_id(2)
    PER: tl.constexpr = 8 // LO_BITS
    NBYTE: tl.constexpr = K // PER
    slot = tl.load(idx + i_n).to(tl.int64)
    slot = tl.where(slot < 0, slot + n_slots, slot)
    p = st + slot * stride_slot + i_h * NB
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    hi = tl.load((p + o_v[:, None] * K + o_k[None, :]).to(tl.pointer_type(tl.int8))).to(tl.int32)
    lo = tl.load(p + O_LO + o_v[:, None] * NBYTE + (o_k % NBYTE)[None, :]).to(tl.int32)
    q = hi * (1 << LO_BITS) + ((lo >> ((o_k // NBYTE) * LO_BITS)[None, :]) & ((1 << LO_BITS) - 1))
    s = tl.load((p + O_SC + (i_v * K + o_k) * 4).to(tl.pointer_type(tl.float32)))
    x = q.to(tl.float32) * s[None, :]
    p_out = out + i_n.to(tl.int64) * stride_out + i_h * (V * K) + o_v[:, None] * K + o_k[None, :]
    tl.store(p_out, x.to(out.dtype.element_ty))


@triton.jit
def _quantize_scatter_kernel(x, src, dst, st, n_slots, stride_x, stride_slot,
                             V: tl.constexpr, K: tl.constexpr, NB: tl.constexpr, O_LO: tl.constexpr,
                             O_SC: tl.constexpr, LO_BITS: tl.constexpr, QMAX: tl.constexpr, BV: tl.constexpr,
                             HAS_SRC: tl.constexpr):
    """The packed block of slot dst[n] = row src[n] of x (row n without src) quantized: per key channel one scale
    over the tile's value rows, q = floor(x / scale + 0.5) clamped, as gdn_state_fmt.quantize."""
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_h = tl.program_id(2)
    PER: tl.constexpr = 8 // LO_BITS
    NBYTE: tl.constexpr = K // PER
    MASK: tl.constexpr = (1 << LO_BITS) - 1
    if HAS_SRC:
        row = tl.load(src + i_n).to(tl.int64)
    else:
        row = i_n.to(tl.int64)
    slot = tl.load(dst + i_n).to(tl.int64)
    slot = tl.where(slot < 0, slot + n_slots, slot)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    xt = tl.load(x + row * stride_x + i_h * (V * K) + o_v[:, None] * K + o_k[None, :]).to(tl.float32)
    amax = tl.max(tl.abs(xt), axis=0)
    ok = amax > QMAX * 1e-30
    qmax = tl.full([K], QMAX, tl.float32)
    inv = tl.where(ok, tl.math.div_rn(qmax, amax), 0.0)
    q = tl.floor(xt * inv[None, :] + 0.5)
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX).to(tl.int32)
    p = st + slot * stride_slot + i_h * NB
    tl.store((p + o_v[:, None] * K + o_k[None, :]).to(tl.pointer_type(tl.int8)), (q >> LO_BITS).to(tl.int8))
    lo = tl.reshape(q & MASK, (BV, PER, NBYTE))  # key i * NBYTE + j -> [value, field i, byte j]
    packed = tl.sum(lo << (tl.arange(0, PER) * LO_BITS)[None, :, None], axis=1)
    tl.store(p + O_LO + o_v[:, None] * NBYTE + tl.arange(0, NBYTE)[None, :], packed.to(tl.uint8))
    tl.store((p + O_SC + (i_v * K + o_k) * 4).to(tl.pointer_type(tl.float32)),
             tl.where(ok, tl.math.div_rn(amax, qmax), 0.0))


def launch_rotate_v(x2: torch.Tensor, y2: torch.Tensor, h: torch.Tensor) -> None:
    """x2, y2: [rows, cols] views (unit column stride, cols a multiple of H's size; y2 may be x2)."""
    rows, cols = x2.shape
    g = h.shape[0]
    if y2.shape != x2.shape or x2.stride(1) != 1 or y2.stride(1) != 1 or cols % g or h.dtype != x2.dtype:
        raise ValueError(f"rotate_v: x {tuple(x2.shape)} {x2.stride()} {x2.dtype}, y {tuple(y2.shape)} "
                         f"{y2.stride()}, H {tuple(h.shape)} {h.dtype}")
    if rows == 0:
        return
    _rotate_v_kernel[(triton.cdiv(rows, 64), cols // g)](x2, y2, h, rows, x2.stride(0), y2.stride(0), G=g, BR=64,
                                                          num_warps=4)


def _check_pool(fmt: StateFmt, pool: torch.Tensor, v: int, k: int) -> None:
    nb = fmt.head_bytes(v, k)
    if pool.dtype != torch.uint8 or pool.dim() != 3 or pool.shape[2] != nb or pool.stride(2) != 1 \
            or pool.stride(1) != nb:
        raise ValueError(f"not a per-layer {fmt.name} pool [slots, HV, {nb}]: {pool.dtype} {tuple(pool.shape)} "
                         f"{pool.stride()}")


def launch_gather_dequant(fmt: StateFmt, pool: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    """out [n, HV, V, K] (contiguous rows) = pool[idx] dequantized."""
    n, hv, v, k = out.shape
    _check_pool(fmt, pool, v, k)
    if idx.dim() != 1 or idx.shape[0] != n or pool.shape[1] != hv or out.stride()[1:] != (v * k, k, 1):
        raise ValueError(f"gather_dequant: idx {tuple(idx.shape)}, out {tuple(out.shape)} {out.stride()}, "
                         f"pool {tuple(pool.shape)}")
    if n == 0:
        return
    _, o_lo, o_sc = fmt.offsets(v, k)
    _gather_dequant_kernel[(v // GROUP_V, n, hv)](
        pool, idx, out, pool.shape[0], pool.stride(0), out.stride(0),
        V=v, K=k, NB=pool.shape[2], O_LO=o_lo, O_SC=o_sc, LO_BITS=fmt.lo_bits, BV=GROUP_V, num_warps=4)


def launch_quantize_scatter(fmt: StateFmt, x: torch.Tensor, pool: torch.Tensor, dst: torch.Tensor,
                            src: torch.Tensor | None = None) -> None:
    """pool[dst[i]] = quantize(x[src[i]]) (x[i] without src); x [rows, HV, V, K] with contiguous rows."""
    _, hv, v, k = x.shape
    _check_pool(fmt, pool, v, k)
    n = dst.shape[0]
    if dst.dim() != 1 or pool.shape[1] != hv or x.stride()[1:] != (v * k, k, 1) or \
            (src is not None and (src.dim() != 1 or src.shape[0] != n)) or (src is None and x.shape[0] != n):
        raise ValueError(f"quantize_scatter: x {tuple(x.shape)} {x.stride()}, dst {tuple(dst.shape)}, "
                         f"src {None if src is None else tuple(src.shape)}, pool {tuple(pool.shape)}")
    if n == 0:
        return
    _, o_lo, o_sc = fmt.offsets(v, k)
    _quantize_scatter_kernel[(v // GROUP_V, n, hv)](
        x, dst if src is None else src, dst, pool, pool.shape[0], x.stride(0), pool.stride(0),
        V=v, K=k, NB=pool.shape[2], O_LO=o_lo, O_SC=o_sc, LO_BITS=fmt.lo_bits, QMAX=float(fmt.qmax), BV=GROUP_V,
        HAS_SRC=src is not None, num_warps=4)
