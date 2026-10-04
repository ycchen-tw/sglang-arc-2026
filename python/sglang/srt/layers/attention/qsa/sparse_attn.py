"""Validated sparse GQA operators migrated from the QSA reference branch."""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # The chunk-prefill K/V tensors are gathered from the KV pool and can
        # therefore carry the FP8 storage dtype, which Triton's dot rejects
        # (`Unsupported rhs dtype fp8e4nv`). Convert to Q's dtype; the QSA
        # backend writes the pool without per-tensor k/v scales, so this is a
        # plain cast (no-op for BF16 pools).
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(
    q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q=None
):
    """max_q: the longest query segment when the caller knows it on the host
    (arc-forge h15); otherwise it is read back from cu_q."""
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    if max_q is None:
        max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # One request per program: Triton caps a tile at 1M elements,
    # which [next_pow2(topk), next_pow2(batch)] exceeds at topk=2051, batch=512.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
    DST_PAGE_SIZE: tl.constexpr,
    DST_V_GROUP: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # 64-bit element offsets: slot * heads * dim exceeds int32 once the pool holds
    # more than 2^31 / (heads * dim) tokens (~4.2M for 2 x 256), which an FP8 pool
    # on one GPU does reach.
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    token = (pack_start + cols).to(tl.int64)
    if DST_PAGE_SIZE > 0:
        # Write straight into the consumer's paged [page, head, token, dim]
        # order instead of the token-major default, so a second pass over the
        # scratch is not needed to hand it to a kernel that wants HND. With
        # DST_V_GROUP set, the value plane additionally lands in token groups
        # of that size with dim interleaved inside each group -- the layout the
        # sm_100 trtllm-gen FP4 decode kernel reads its value scales in.
        page = token // DST_PAGE_SIZE
        slot = token % DST_PAGE_SIZE
        base = page * (heads * DST_PAGE_SIZE * dim) + head * (DST_PAGE_SIZE * dim)
        dst_k = base[:, None] + slot[:, None] * dim + dims[None, :]
        if DST_V_GROUP > 0:
            dst_v = (
                base[:, None]
                + (slot // DST_V_GROUP)[:, None] * (DST_V_GROUP * dim)
                + dims[None, :] * DST_V_GROUP
                + (slot % DST_V_GROUP)[:, None]
            )
        else:
            dst_v = dst_k
    else:
        dst_k = token[:, None] * heads * dim + head * dim + dims[None, :]
        dst_v = dst_k
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole pages,
        # so every slot in [valid_count, pad_cols) must hold zeros, never stale bytes.
        # `valid_count` here is the row's page-aligned stride, not its valid count, so
        # the store covers the full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    # Dequantize while gathering: the scratch is allocated in the query dtype, so an
    # FP8 pool is read as fp8 and stored as bf16. The QSA backend writes the pool
    # without per-tensor k/v scales (see set_kv_buffer calls in
    # qwen_sparse_attn_backend.py), so no scale is applied here either.
    out_dtype = out_k.dtype.element_ty
    tl.store(
        out_k + dst_k,
        tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )
    tl.store(
        out_v + dst_v,
        tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, without the packed cu_seqlens prefix sum."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
    dst_page_size: Optional[int] = None,
    dst_value_group: Optional[int] = None,
):
    """Gather the selected K/V rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the strided (page-aligned) layout used by the paged
    decode kernel: row ``b`` owns ``[cu_k[b], cu_k[b] + zero_fill_cols)`` and every slot
    past its valid rows is zero-filled. Paged kernels read whole pages and multiply the
    masked probabilities into V, so stale or uninitialized bytes there (NaN/Inf bit
    patterns) would otherwise leak into the output. ``0`` keeps the compact layout for
    the varlen fallback, whose rows are packed back-to-back.

    ``out_k``/``out_v`` may use a wider dtype than the pool (bf16 scratch for an FP8
    pool); rows are converted while gathering.

    Both layouts assume the valid entries of each ``indices`` row are contiguous at
    the front (``expand_qsa_block_indices`` sorts them that way): ``valid_count`` is a
    count, not a mask, so a ``-1`` in the middle of a row would shift the packing.

    ``dst_page_size`` switches the destination from token-major to the consumer's
    paged ``[page, head, token, dim]`` order; ``dst_value_group`` additionally lands
    the value plane in token groups of that size with ``dim`` interleaved inside each
    group. Both are only meaningful for the strided layout, so they require
    ``zero_fill_cols``.
    """
    dst_page_size = dst_page_size or 0
    dst_value_group = dst_value_group or 0
    if dst_page_size and zero_fill_cols <= 0:
        raise ValueError("dst_page_size only applies to the strided (zero-fill) layout")
    if dst_value_group and not dst_page_size:
        raise ValueError("dst_value_group needs dst_page_size")
    if dst_page_size and (
        zero_fill_cols % dst_page_size or dst_page_size % max(dst_value_group, 1)
    ):
        raise ValueError(
            "dst_page_size must divide zero_fill_cols and be a multiple of "
            "dst_value_group"
        )
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    _compact_kv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        DST_PAGE_SIZE=dst_page_size,
        DST_V_GROUP=dst_value_group,
        num_warps=8,
    )


@triton.jit
def _compact_kv_nvfp4_four_plane(
    k,
    v,
    ks,
    vs,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    out_ks,
    out_vs,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    sf_dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    DST_PAGE_SIZE: tl.constexpr,
    DST_V_GROUP: tl.constexpr,
):
    """Move an NVFP4 selection's four planes in one pass over the address chain.

    ``_compact_kv`` has to be called twice for an NVFP4 pool -- once for the
    packed payload, once for the block scales -- and each call re-runs the
    logical-position -> request-table -> slot lookup over a ``(rows, heads,
    tiles)`` grid. The scale pass moves an eighth of the bytes but pays the same
    addressing cost at a 16-byte feature width. Here the lookup happens once and
    the head axis is folded into the feature axis, so a slot's heads are one
    contiguous run in each plane. Destination layouts are unchanged.
    """
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    req = tl.load(req_indices + row)
    pack_start = tl.load(cu_k + row)
    valid_count = tl.load(cu_k + row + 1) - pack_start
    positions = tl.load(indices + row * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    # 64-bit element offsets, as in _compact_kv: slot * heads * dim overflows
    # int32 once the pool holds more than 2^31 / (heads * dim) tokens.
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    token = (pack_start + cols).to(tl.int64)
    # Same page-aligned zero fill as the two-pass path: the paged decode kernel
    # reads whole pages, so every slot up to the row stride must be written.
    store_rows = cols < pad_cols

    hd = tl.arange(0, heads * dim)
    src_p = slots[:, None] * (heads * dim) + hd[None, :]
    dst_p = token[:, None] * (heads * dim) + hd[None, :]
    load_p = valid[:, None] & (hd[None, :] < heads * dim)
    store_p = store_rows[:, None] & (hd[None, :] < heads * dim)
    tl.store(out_k + dst_p, tl.load(k + src_p, mask=load_p, other=0), mask=store_p)
    tl.store(out_v + dst_p, tl.load(v + src_p, mask=load_p, other=0), mask=store_p)

    sd = tl.arange(0, heads * sf_dim)
    head_i = sd // sf_dim
    feat_i = sd % sf_dim
    src_s = slots[:, None] * (heads * sf_dim) + sd[None, :]
    load_s = valid[:, None] & (sd[None, :] < heads * sf_dim)
    store_s = store_rows[:, None] & (sd[None, :] < heads * sf_dim)
    ks_val = tl.load(ks + src_s, mask=load_s, other=0)
    vs_val = tl.load(vs + src_s, mask=load_s, other=0)
    if DST_PAGE_SIZE > 0:
        # sm_100 trtllm-gen order: scale planes paged HND, the value plane
        # additionally grouped by DST_V_GROUP tokens with the channel blocks
        # interleaved inside a group. page/slot vary along the token axis, so
        # both must be broadcast as [:, None] -- a 1-D page aligns to the
        # feature axis and only happens to be right while a tile sits inside
        # one page.
        page = (token // DST_PAGE_SIZE)[:, None]
        slot = (token % DST_PAGE_SIZE)[:, None]
        base = page * (heads * DST_PAGE_SIZE * sf_dim) + head_i[None, :] * (
            DST_PAGE_SIZE * sf_dim
        )
        dst_ks = base + slot * sf_dim + feat_i[None, :]
        dst_vs = (
            base
            + (slot // DST_V_GROUP) * (DST_V_GROUP * sf_dim)
            + feat_i[None, :] * DST_V_GROUP
            + (slot % DST_V_GROUP)
        )
    else:
        # sm_120 XQA reads NHD with a linear scale plane: token-major, like the
        # payload.
        dst_ks = token[:, None] * (heads * sf_dim) + sd[None, :]
        dst_vs = dst_ks
    tl.store(out_ks + dst_ks, ks_val, mask=store_s)
    tl.store(out_vs + dst_vs, vs_val, mask=store_s)


_ARC_QSA_GATHER4_BLOCK_TOPK = int(os.environ.get("ARC_QSA_GATHER4_BLOCK_TOPK", "64"))
_ARC_QSA_GATHER4_WARPS = int(os.environ.get("ARC_QSA_GATHER4_WARPS", "8"))
for _name, _value in (("ARC_QSA_GATHER4_BLOCK_TOPK", _ARC_QSA_GATHER4_BLOCK_TOPK),
                      ("ARC_QSA_GATHER4_WARPS", _ARC_QSA_GATHER4_WARPS)):
    # tl.arange needs a power of two, and cdiv needs a positive tile; both
    # otherwise fail at first decode inside graph capture instead of here.
    if _value <= 0 or _value & (_value - 1):
        raise ValueError(f"{_name} must be a positive power of two, got {_value}")


def qwen_sparse_nvfp4_gather4_triton(
    k,
    v,
    ks,
    vs,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    out_ks,
    out_vs,
    rows,
    topk,
    stride,
    dst_page_size: int = 0,
    dst_value_group: int = 0,
):
    """Fused four-plane version of two ``qwen_sparse_kv_extraction_compact_triton``
    calls for an NVFP4 pool. Byte-for-byte identical output; see
    ``work/qwen38fn_eval/test_qsa_nvfp4_gather4.py``.

    ``dst_page_size``/``dst_value_group`` select the sm_100 consumer's paged
    scale layout exactly as they do for the split path; ``0`` keeps the
    token-major sm_120 layout.
    """
    if stride <= 0:
        raise ValueError("fused NVFP4 gather is only defined for the strided layout")
    if dst_value_group and not dst_page_size:
        raise ValueError("dst_value_group needs dst_page_size")
    if dst_page_size and (
        stride % dst_page_size or dst_page_size % max(dst_value_group, 1)
    ):
        raise ValueError(
            "dst_page_size must divide the row stride and be a multiple of "
            "dst_value_group"
        )
    for name, tensor in (
        ("k", k), ("v", v), ("ks", ks), ("vs", vs),
        ("out_k", out_k), ("out_v", out_v), ("out_ks", out_ks), ("out_vs", out_vs),
    ):
        if tensor.dtype != torch.uint8:
            raise ValueError(
                f"fused NVFP4 gather moves raw bytes; {name} is {tensor.dtype}"
            )
    _, heads, dim = k.shape
    sf_dim = ks.shape[2]
    _compact_kv_nvfp4_four_plane[(rows, triton.cdiv(stride, _ARC_QSA_GATHER4_BLOCK_TOPK))](
        k, v, ks, vs,
        req_to_token, req_indices, indices, seq_lens, cu_k,
        out_k, out_v, out_ks, out_vs,
        topk, heads, dim, sf_dim,
        req_to_token.stride(0), indices.stride(0), stride,
        BLOCK_TOPK=_ARC_QSA_GATHER4_BLOCK_TOPK,
        DST_PAGE_SIZE=dst_page_size,
        DST_V_GROUP=dst_value_group,
        num_warps=_ARC_QSA_GATHER4_WARPS,
        num_stages=1,
    )


@triton.jit
def _nvfp4_decode_nibble(byte, shift):
    """E2M1: bit3 sign, bits2-1 exponent, bit0 mantissa.

    Exponent 0 is the subnormal pair (0, 0.5); otherwise the value is
    2^(exponent-1) * (1 + mantissa/2), i.e. the {1,1.5,2,3,4,6} ladder.
    """
    nibble = (byte >> (4 * shift)) & 0x0F
    exponent = (nibble >> 1) & 0x03
    mantissa = (nibble & 0x01).to(tl.float32)
    pow2 = (1 << tl.maximum(exponent - 1, 0)).to(tl.float32)
    magnitude = tl.where(exponent == 0, mantissa * 0.5, pow2 * (1.0 + mantissa * 0.5))
    return tl.where(((nibble >> 3) & 0x01) == 1, -magnitude, magnitude)


@triton.jit
def _nvfp4_expand_kv(
    k_fp4,
    v_fp4,
    k_sf,
    v_sf,
    req_to_token,
    req_indices,
    cu_k,
    out_k,
    out_v,
    k_global,
    v_global,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    BLOCK_T: tl.constexpr,
    OUT_FP8: tl.constexpr,
):
    """Gather one request's packed NVFP4 context and write it out dequantized.

    Replaces index_select -> nvfp4_kv_dequantize(BF16) -> .to(FP8) -> torch.cat
    with a single pass: the packed bytes and their block scales are read once
    and the result lands directly in the caller's contiguous buffer at the
    request's cu_k offset, so no BF16 copy of the context ever exists.
    """
    PACK: tl.constexpr = dim // 2
    SF: tl.constexpr = dim // 16
    pid_b, pid_h, pid_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(cu_k + pid_b).to(tl.int64)
    length = tl.load(cu_k + pid_b + 1).to(tl.int64) - start
    tokens = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < length
    req = tl.load(req_indices + pid_b).to(tl.int64)
    slots = tl.load(
        req_to_token + req * req_stride + tokens, mask=live, other=0
    ).to(tl.int64)

    pack = tl.arange(0, PACK)
    packed_src = slots[:, None] * (heads * PACK) + pid_h * PACK + pack[None, :]
    scale_src = slots[:, None] * (heads * SF) + pid_h * SF + (pack // 8)[None, :]
    mask = live[:, None]
    kb = tl.load(k_fp4 + packed_src, mask=mask, other=0)
    vb = tl.load(v_fp4 + packed_src, mask=mask, other=0)
    k_step = tl.load(k_sf + scale_src, mask=mask, other=0.0).to(tl.float32) * tl.load(k_global)
    v_step = tl.load(v_sf + scale_src, mask=mask, other=0.0).to(tl.float32) * tl.load(v_global)
    dst = (start + tokens)[:, None] * (heads * dim) + pid_h * dim + 2 * pack[None, :]

    k_low = _nvfp4_decode_nibble(kb, 0) * k_step
    k_high = _nvfp4_decode_nibble(kb, 1) * k_step
    v_low = _nvfp4_decode_nibble(vb, 0) * v_step
    v_high = _nvfp4_decode_nibble(vb, 1) * v_step
    if OUT_FP8:
        k_low, k_high = k_low.to(tl.float8e4nv), k_high.to(tl.float8e4nv)
        v_low, v_high = v_low.to(tl.float8e4nv), v_high.to(tl.float8e4nv)
    else:
        k_low, k_high = k_low.to(tl.bfloat16), k_high.to(tl.bfloat16)
        v_low, v_high = v_low.to(tl.bfloat16), v_high.to(tl.bfloat16)
    tl.store(out_k + dst, k_low, mask=mask)
    tl.store(out_k + dst + 1, k_high, mask=mask)
    tl.store(out_v + dst, v_low, mask=mask)
    tl.store(out_v + dst + 1, v_high, mask=mask)


def qwen_sparse_nvfp4_expand_triton(
    k_fp4,
    v_fp4,
    k_sf,
    v_sf,
    req_to_token,
    req_indices,
    cu_k,
    out_k,
    out_v,
    k_global,
    v_global,
    max_length,
):
    """Expand every request's packed NVFP4 context into one contiguous K/V pair.

    ``out_k``/``out_v`` are ``[sum(seq_lens), heads, head_dim]`` in either FP8
    E4M3 (what the chunk-prefill kernel consumes) or BF16.
    """
    if k_fp4.dtype != torch.uint8 or v_fp4.dtype != torch.uint8:
        raise ValueError(f"Packed NVFP4 K/V must be uint8, got {k_fp4.dtype}/{v_fp4.dtype}")
    if out_k.dtype != out_v.dtype:
        raise ValueError("NVFP4 expansion outputs must share a dtype")
    # Only E4M3 and BF16 are ever written here, so the dtypes are named
    # outright rather than going through a generic is_fp8 predicate.
    out_fp8 = out_k.dtype == torch.float8_e4m3fn
    if not out_fp8 and out_k.dtype != torch.bfloat16:
        raise ValueError(f"NVFP4 expansion writes FP8 or BF16, got {out_k.dtype}")
    if k_global.dtype != torch.float32 or v_global.dtype != torch.float32:
        raise ValueError("NVFP4 global scales must be float32 device tensors")
    batch = req_indices.shape[0]
    heads, dim = out_k.shape[1], out_k.shape[2]
    if k_fp4.shape[1] != heads or k_fp4.shape[2] * 2 != dim:
        raise ValueError("Packed NVFP4 geometry does not match the expansion output")
    if k_sf.shape[2] * 16 != dim:
        raise ValueError("NVFP4 block-scale geometry does not match the expansion output")
    if not (out_k.is_contiguous() and out_v.is_contiguous()):
        raise ValueError("NVFP4 expansion outputs must be contiguous")
    block_t = 8
    _nvfp4_expand_kv[(batch, heads, triton.cdiv(max_length, block_t))](
        k_fp4,
        v_fp4,
        k_sf.view(torch.float8_e4m3fn),
        v_sf.view(torch.float8_e4m3fn),
        req_to_token,
        req_indices,
        cu_k,
        out_k,
        out_v,
        k_global,
        v_global,
        heads,
        dim,
        req_to_token.stride(0),
        BLOCK_T=block_t,
        OUT_FP8=out_fp8,
        num_warps=4,
    )


@triton.jit
def _nvfp4_expand_kv_segments(
    k_fp4,
    v_fp4,
    k_sf,
    v_sf,
    table,
    table_stride,
    segments,
    out_k,
    out_v,
    k_global,
    v_global,
    heads: tl.constexpr,
    dim: tl.constexpr,
    IDENTITY: tl.constexpr,
    BLOCK_T: tl.constexpr,
    OUT_FP8: tl.constexpr,
):
    """_nvfp4_expand_kv with the output placement and source rows given per
    segment (arc-forge h15): segment s = (table row, first table column, first
    output row, length). Source rows are table[row, col + t], or col + t with
    IDENTITY. The per-element arithmetic is _nvfp4_expand_kv's, line for line."""
    PACK: tl.constexpr = dim // 2
    SF: tl.constexpr = dim // 16
    pid_s, pid_h, pid_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seg = segments + pid_s * 4
    row = tl.load(seg).to(tl.int64)
    col = tl.load(seg + 1).to(tl.int64)
    start = tl.load(seg + 2).to(tl.int64)
    length = tl.load(seg + 3).to(tl.int64)
    tokens = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < length
    if IDENTITY:
        slots = col + tokens.to(tl.int64)
    else:
        slots = tl.load(
            table + row * table_stride + col + tokens, mask=live, other=0
        ).to(tl.int64)

    pack = tl.arange(0, PACK)
    packed_src = slots[:, None] * (heads * PACK) + pid_h * PACK + pack[None, :]
    scale_src = slots[:, None] * (heads * SF) + pid_h * SF + (pack // 8)[None, :]
    mask = live[:, None]
    kb = tl.load(k_fp4 + packed_src, mask=mask, other=0)
    vb = tl.load(v_fp4 + packed_src, mask=mask, other=0)
    k_step = tl.load(k_sf + scale_src, mask=mask, other=0.0).to(tl.float32) * tl.load(k_global)
    v_step = tl.load(v_sf + scale_src, mask=mask, other=0.0).to(tl.float32) * tl.load(v_global)
    dst = (start + tokens)[:, None] * (heads * dim) + pid_h * dim + 2 * pack[None, :]

    k_low = _nvfp4_decode_nibble(kb, 0) * k_step
    k_high = _nvfp4_decode_nibble(kb, 1) * k_step
    v_low = _nvfp4_decode_nibble(vb, 0) * v_step
    v_high = _nvfp4_decode_nibble(vb, 1) * v_step
    if OUT_FP8:
        k_low, k_high = k_low.to(tl.float8e4nv), k_high.to(tl.float8e4nv)
        v_low, v_high = v_low.to(tl.float8e4nv), v_high.to(tl.float8e4nv)
    else:
        k_low, k_high = k_low.to(tl.bfloat16), k_high.to(tl.bfloat16)
        v_low, v_high = v_low.to(tl.bfloat16), v_high.to(tl.bfloat16)
    tl.store(out_k + dst, k_low, mask=mask)
    tl.store(out_k + dst + 1, k_high, mask=mask)
    tl.store(out_v + dst, v_low, mask=mask)
    tl.store(out_v + dst + 1, v_high, mask=mask)


def qwen_sparse_nvfp4_expand_segments_triton(
    k_fp4,
    v_fp4,
    k_sf,
    v_sf,
    table,
    segments,
    out_k,
    out_v,
    k_global,
    v_global,
    max_length,
):
    """Expand packed NVFP4 rows segment by segment into ``out_k``/``out_v``
    (arc-forge h15). ``segments`` is int64 ``[n, 4]``: (table row, first table
    column, first output row, length); ``table`` None reads plane rows
    ``col .. col + length - 1`` directly. One launch covers pieces whose outputs
    are not one contiguous run (every request's device part of a resumed
    batch), with the bytes qwen_sparse_nvfp4_expand_triton would write for each.
    ``max_length`` (host int) is the longest segment; the table's row stride is
    a runtime argument, so tables of any width share one compiled kernel."""
    if k_fp4.dtype != torch.uint8 or v_fp4.dtype != torch.uint8:
        raise ValueError(f"Packed NVFP4 K/V must be uint8, got {k_fp4.dtype}/{v_fp4.dtype}")
    if out_k.dtype != out_v.dtype:
        raise ValueError("NVFP4 expansion outputs must share a dtype")
    out_fp8 = out_k.dtype == torch.float8_e4m3fn
    if not out_fp8 and out_k.dtype != torch.bfloat16:
        raise ValueError(f"NVFP4 expansion writes FP8 or BF16, got {out_k.dtype}")
    if k_global.dtype != torch.float32 or v_global.dtype != torch.float32:
        raise ValueError("NVFP4 global scales must be float32 device tensors")
    heads, dim = out_k.shape[1], out_k.shape[2]
    if k_fp4.shape[1] != heads or k_fp4.shape[2] * 2 != dim:
        raise ValueError("Packed NVFP4 geometry does not match the expansion output")
    if k_sf.shape[2] * 16 != dim:
        raise ValueError("NVFP4 block-scale geometry does not match the expansion output")
    if not (out_k.is_contiguous() and out_v.is_contiguous()):
        raise ValueError("NVFP4 expansion outputs must be contiguous")
    if (
        segments.dtype != torch.int64
        or segments.ndim != 2
        or segments.shape[1] != 4
        or not segments.is_contiguous()
    ):
        raise ValueError(
            f"segments must be contiguous int64 [n, 4], got {segments.dtype} {tuple(segments.shape)}"
        )
    identity = table is None
    if not identity and (table.ndim != 2 or table.stride(1) != 1):
        raise ValueError("the NVFP4 segment table must be 2-D with unit column stride")
    if segments.shape[0] == 0 or max_length <= 0:
        raise ValueError(f"empty NVFP4 segment expansion ({segments.shape[0]} segments, longest {max_length})")
    block_t = 8
    _nvfp4_expand_kv_segments[(segments.shape[0], heads, triton.cdiv(max_length, block_t))](
        k_fp4,
        v_fp4,
        k_sf.view(torch.float8_e4m3fn),
        v_sf.view(torch.float8_e4m3fn),
        segments if identity else table,
        0 if identity else table.stride(0),
        segments,
        out_k,
        out_v,
        k_global,
        v_global,
        heads,
        dim,
        IDENTITY=identity,
        BLOCK_T=block_t,
        OUT_FP8=out_fp8,
        num_warps=4,
    )



@triton.jit
def _sparse_gqa_decode_selected(
    q,
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    out,
    scale,
    k_scale,
    v_scale,
    topk: tl.constexpr,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    req_stride: tl.constexpr,
    si_m: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    """Sparse GQA over selected logical tokens without compacting K/V first."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    length = tl.load(seq_lens + row)
    req = tl.load(req_indices + row)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = kv_head * GROUP_SIZE
    q_values = tl.load(
        q + row * sq_m + (head_start + offs_h[:, None]) * sq_h + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, topk, BLOCK_N):
        current = start + offs_n
        position = tl.load(
            indices + row * si_m + current,
            mask=current < topk,
            other=-1,
        )
        valid = (current < topk) & (position >= 0) & (position < length)
        token = tl.load(
            req_to_token + req * req_stride + tl.where(valid, position, 0),
            mask=valid,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            k + token[None, :] * sk_n + kv_head * sk_h + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        ).to(q_values.dtype)
        values = tl.load(
            v + token[:, None] * sv_n + kv_head * sv_h + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        ).to(q_values.dtype)
        scores = tl.dot(q_values, keys)
        if KV_IS_FP8:
            scores *= k_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    if KV_IS_FP8:
        output *= v_scale
    tl.store(
        out
        + row * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def qwen_sparse_gqa_decode_selected_triton(
    q,
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    scale,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
):
    """Decode sparse GQA straight from paged K/V selected by logical indices."""
    # L4_FP8_DIRECT_SCALES
    kv_is_fp8 = _validate_sparse_gqa_dtypes(q, k, v)
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    group_size = q_heads // kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    if rows <= 128:
        block_n, warps, stages = 128, 4, 3
    elif rows <= 256:
        block_n, warps, stages = 64, 8, 3
    else:
        block_n, warps, stages = 64, 4, 3
    # SM89 shared memory is 99 KiB. Measured for 12 q-heads x head_dim 256:
    # BLOCK_N=128/stages=3 needs 140.5 KiB with BF16 K/V (does not fit) but only
    # 76.5 KiB with FP8 or packed FP4 (fits). Shrinking unconditionally cost 2.1x
    # (67 -> 140 us) on exactly the config that ships, so gate on the KV width.
    if torch.cuda.get_device_capability(q.device) == (8, 9) and not kv_is_fp8:
        block_n, stages = 64, 2
    out = torch.empty_like(q)
    if rows == 0:
        return out
    _sparse_gqa_decode_selected[(rows, kv_heads)](
        q,
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        out,
        scale,
        _unit_scale(k_scale),
        _unit_scale(v_scale),
        indices.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        req_to_token.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        KV_IS_FP8=kv_is_fp8,
        num_warps=warps,
        num_stages=stages,
    )
    return out



@triton.jit
def _sparse_gqa_decode_selected_fp4(
    q_even, q_odd, k, v, k_sf, v_sf, req_to_token, req_indices, indices, seq_lens, out,
    topk: tl.constexpr, req_stride: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    Q_HEADS: tl.constexpr, HEADS: tl.constexpr, PACK: tl.constexpr, SF: tl.constexpr,
):
    row, kv_head = tl.program_id(0), tl.program_id(1)
    length = tl.load(seq_lens + row)
    req = tl.load(req_indices + row)
    offs_h = tl.arange(0, BLOCK_M)
    offs_p = tl.arange(0, PACK)
    hmask = (offs_h < GROUP_SIZE)[:, None]
    head_start = kv_head * GROUP_SIZE
    qe = tl.load(q_even + row * (Q_HEADS * PACK) + (head_start + offs_h[:, None]) * PACK
                 + offs_p[None, :], mask=hmask, other=0.0)
    qo = tl.load(q_odd + row * (Q_HEADS * PACK) + (head_start + offs_h[:, None]) * PACK
                 + offs_p[None, :], mask=hmask, other=0.0)

    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    acc_even = tl.zeros([BLOCK_M, PACK], tl.float32)
    acc_odd = tl.zeros([BLOCK_M, PACK], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    offs_s = tl.arange(0, SF)                  # 1 scale per 16 features == per 8 bytes

    for start in range(0, topk, BLOCK_N):
        cur = start + offs_n
        pos = tl.load(indices + row * topk + cur, mask=cur < topk, other=-1)
        valid = (cur < topk) & (pos >= 0) & (pos < length)
        tok = tl.load(req_to_token + req * req_stride + tl.where(valid, pos, 0),
                      mask=valid, other=0).to(tl.int64)

        # K as [PACK, BLOCK_N] so it feeds tl.dot(q[BLOCK_M, PACK], k[PACK, BLOCK_N]).
        k_src = tok[None, :] * (HEADS * PACK) + kv_head * PACK + offs_p[:, None]
        k_ss = tok[None, :] * (HEADS * SF) + kv_head * SF + offs_s[:, None]
        kb = tl.load(k + k_src, mask=valid[None, :], other=0)
        kstep = tl.load(k_sf + k_ss, mask=valid[None, :], other=0.0).to(tl.float32)
        kstep = tl.reshape(tl.broadcast_to(tl.reshape(kstep, (SF, 1, BLOCK_N)),
                                           (SF, 8, BLOCK_N)), (PACK, BLOCK_N))
        k_low = (_nvfp4_decode_nibble(kb, 0) * kstep).to(qe.dtype)
        k_high = (_nvfp4_decode_nibble(kb, 1) * kstep).to(qe.dtype)

        scores = tl.dot(qe, k_low) + tl.dot(qo, k_high)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probs = tl.math.exp2(scores - next_max[:, None])

        v_src = tok[:, None] * (HEADS * PACK) + kv_head * PACK + offs_p[None, :]
        v_ss = tok[:, None] * (HEADS * SF) + kv_head * SF + offs_s[None, :]
        vb = tl.load(v + v_src, mask=valid[:, None], other=0)
        vstep = tl.load(v_sf + v_ss, mask=valid[:, None], other=0.0).to(tl.float32)
        vstep = tl.reshape(tl.broadcast_to(tl.reshape(vstep, (BLOCK_N, SF, 1)),
                                           (BLOCK_N, SF, 8)), (BLOCK_N, PACK))
        p = probs.to(qe.dtype)
        acc_even = tl.dot(p, (_nvfp4_decode_nibble(vb, 0) * vstep).to(qe.dtype),
                          acc_even * alpha[:, None])
        acc_odd = tl.dot(p, (_nvfp4_decode_nibble(vb, 1) * vstep).to(qe.dtype),
                         acc_odd * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probs, 1)
        max_value = next_max

    acc_even = acc_even / normalizer[:, None]
    acc_odd = acc_odd / normalizer[:, None]
    base = out + row * (Q_HEADS * 2 * PACK) + (head_start + offs_h[:, None]) * (2 * PACK)
    tl.store(base + 2 * offs_p[None, :], acc_even, mask=hmask)
    tl.store(base + 2 * offs_p[None, :] + 1, acc_odd, mask=hmask)


def qwen_sparse_gqa_decode_selected_fp4_triton(
    q, k, v, k_sf, v_sf, req_to_token, req_indices, indices, seq_lens, scale,
    k_global=1.0, v_global=1.0, block_n=None, stages=None, warps=None,
):
    """Sparse GQA decode straight off the packed FP4 pool.

    The sm_100/sm_120 path hands these planes to trtllm-gen/XQA. Ada has no such
    kernel, and the pool's fallback dequantizes the entire layer buffer on every
    read; this does the unpack inline instead. Tile config is the measured
    optimum on sm_89 (BLOCK_N=128, 2 stages, 16 warps). A 16-entry LUT gather, a
    bf16 decode chain and a tl.join-merged V dot were each tried and each came out
    slower, so the arithmetic decode stays.
    """
    rows, q_heads, dim = q.shape
    kv_heads, pack = k.shape[1], k.shape[2]
    if pack * 2 != dim:
        raise ValueError(f"packed width {pack} does not match head_dim {dim}")
    group = q_heads // kv_heads
    scaled = (q.float() * scale * 1.4426950408 * k_global).to(torch.bfloat16)
    q_even = scaled[..., 0::2].contiguous()
    q_odd = scaled[..., 1::2].contiguous()
    out = torch.empty(rows, q_heads, dim, device=q.device, dtype=torch.float32)
    if rows == 0:
        return out.to(q.dtype)
    _sparse_gqa_decode_selected_fp4[(rows, kv_heads)](
        q_even, q_odd, k, v, k_sf, v_sf, req_to_token, req_indices, indices,
        seq_lens, out,
        topk=indices.shape[1], req_stride=req_to_token.stride(0),
        GROUP_SIZE=group, BLOCK_M=max(16, triton.next_power_of_2(group)),
        BLOCK_N=128 if block_n is None else block_n,
        Q_HEADS=q_heads, HEADS=kv_heads, PACK=pack, SF=dim // 16,
        num_warps=16 if warps is None else warps,
        num_stages=2 if stages is None else stages,
    )
    return (out * v_global).to(q.dtype)



# ---------------------------------------------------------------- arc-forge h22: direct sparse decode
# Sparse GQA over the selected slots read straight from the pool (NVFP4 packed or FP8), through any
# (table, table row) pair: req_to_token over logical positions, or HiSparse's identity table over device
# rows. Split-K over the selection (flash-decoding) so a 55-row decode fills the GPU; one program per
# (row, KV head, split) shares the K/V decode across the head group. The query scaling happens in the
# kernel and the output is cast in the final store, so a call is one launch (two with splits).
# E2M1 decode variants (bit-identical values): 0 arithmetic (the 0020 ladder), 1 bit construction,
# 2 the sm_100a / sm_120a cvt.rn.f16x2.e2m1x2 instruction.


@triton.jit
def _e2m1_nibble_f32(nib, DECODE: tl.constexpr):
    if DECODE == 0:
        exponent = (nib >> 1) & 0x03
        mantissa = (nib & 0x01).to(tl.float32)
        pow2 = (1 << tl.maximum(exponent - 1, 0)).to(tl.float32)
        magnitude = tl.where(exponent == 0, mantissa * 0.5, pow2 * (1.0 + mantissa * 0.5))
        return tl.where(((nib >> 3) & 0x01) == 1, -magnitude, magnitude)
    else:
        # fp32 bits: exponent e + 126 and the mantissa bit m for em = e:m >= 2; 0.5 / 0 below
        em = nib & 7
        bits = tl.where(em >= 2, (em + 252) << 22, tl.where(em == 1, 0x3F000000, 0))
        return (bits | ((nib & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_pair_f32(b, DECODE: tl.constexpr):
    """uint8 packed E2M1 pairs -> (low nibble, high nibble) as fp32."""
    if DECODE == 2:
        h = tl.inline_asm_elementwise(
            asm="{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, $4; "
                "cvt.rn.f16x2.e2m1x2 $0, b0; cvt.rn.f16x2.e2m1x2 $1, b1; "
                "cvt.rn.f16x2.e2m1x2 $2, b2; cvt.rn.f16x2.e2m1x2 $3, b3; }",
            constraints="=r,=r,=r,=r,r", args=[b], dtype=tl.int32, is_pure=True, pack=4)
        lo = (h & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        hi = (h >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
        return lo, hi
    else:
        bi = b.to(tl.int32)
        return _e2m1_nibble_f32(bi & 0x0F, DECODE), _e2m1_nibble_f32(bi >> 4, DECODE)


@triton.jit
def _qsa_direct_decode(
    q, k, v, k_sf, v_sf, table, table_rows, indices, lens, out, part_o, part_ml,
    sm_scale, k_global, v_global, topk, table_stride, chunk,
    GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    Q_HEADS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
    SPLITS: tl.constexpr, FP4: tl.constexpr, DECODE: tl.constexpr,
):
    row, kv_head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    HALF: tl.constexpr = DIM // 2
    SF: tl.constexpr = DIM // 16
    length = tl.load(lens + row)
    trow = tl.load(table_rows + row).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_M)
    hmask = (offs_h < GROUP)[:, None]
    head_start = kv_head * GROUP
    q_row = q + row * (Q_HEADS * DIM) + (head_start + offs_h[:, None]) * DIM
    offs_n = tl.arange(0, BLOCK_N)
    if FP4:
        offs_p = tl.arange(0, HALF)
        offs_s = tl.arange(0, SF)
        qe = (((tl.load(q_row + 2 * offs_p[None, :], mask=hmask, other=0.0).to(tl.float32) * sm_scale)
               * 1.4426950408) * k_global).to(tl.bfloat16)
        qo = (((tl.load(q_row + 2 * offs_p[None, :] + 1, mask=hmask, other=0.0).to(tl.float32) * sm_scale)
               * 1.4426950408) * k_global).to(tl.bfloat16)
        acc_e = tl.zeros([BLOCK_M, HALF], tl.float32)
        acc_o = tl.zeros([BLOCK_M, HALF], tl.float32)
    else:
        offs_d = tl.arange(0, DIM)
        qv = (((tl.load(q_row + offs_d[None, :], mask=hmask, other=0.0).to(tl.float32) * sm_scale)
               * 1.4426950408) * k_global).to(tl.bfloat16)
        acc = tl.zeros([BLOCK_M, DIM], tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    begin = split * chunk
    end = tl.minimum(begin + chunk, topk)
    for start in range(begin, end, BLOCK_N):
        cur = start + offs_n
        pos = tl.load(indices + row * topk + cur, mask=cur < end, other=-1)
        valid = (cur < end) & (pos >= 0) & (pos < length)
        tok = tl.load(table + trow * table_stride + tl.where(valid, pos, 0), mask=valid, other=0).to(tl.int64)
        if FP4:
            kb = tl.load(k + tok[None, :] * (HEADS * HALF) + kv_head * HALF + offs_p[:, None],
                         mask=valid[None, :], other=0)
            ks = tl.load(k_sf + tok[None, :] * (HEADS * SF) + kv_head * SF + offs_s[:, None],
                         mask=valid[None, :], other=0.0).to(tl.float32)
            ks = tl.reshape(tl.broadcast_to(tl.reshape(ks, (SF, 1, BLOCK_N)), (SF, 8, BLOCK_N)), (HALF, BLOCK_N))
            k_lo, k_hi = _e2m1_pair_f32(kb, DECODE)
            scores = tl.dot(qe, (k_lo * ks).to(tl.bfloat16)) + tl.dot(qo, (k_hi * ks).to(tl.bfloat16))
        else:
            kt = tl.load(k + tok[None, :] * (HEADS * DIM) + kv_head * DIM + offs_d[:, None],
                         mask=valid[None, :], other=0.0).to(tl.bfloat16)
            scores = tl.dot(qv, kt)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        # a split whose slots so far are all invalid keeps the max at -inf: shift by 0 (alpha, p = 0)
        shift = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - shift)
        p = tl.math.exp2(scores - shift[:, None])
        if FP4:
            vb = tl.load(v + tok[:, None] * (HEADS * HALF) + kv_head * HALF + offs_p[None, :],
                         mask=valid[:, None], other=0)
            vs = tl.load(v_sf + tok[:, None] * (HEADS * SF) + kv_head * SF + offs_s[None, :],
                         mask=valid[:, None], other=0.0).to(tl.float32)
            vs = tl.reshape(tl.broadcast_to(tl.reshape(vs, (BLOCK_N, SF, 1)), (BLOCK_N, SF, 8)), (BLOCK_N, HALF))
            v_lo, v_hi = _e2m1_pair_f32(vb, DECODE)
            pb = p.to(tl.bfloat16)
            acc_e = tl.dot(pb, (v_lo * vs).to(tl.bfloat16), acc_e * alpha[:, None])
            acc_o = tl.dot(pb, (v_hi * vs).to(tl.bfloat16), acc_o * alpha[:, None])
        else:
            vt = tl.load(v + tok[:, None] * (HEADS * DIM) + kv_head * DIM + offs_d[None, :],
                         mask=valid[:, None], other=0.0).to(tl.bfloat16)
            acc = tl.dot(p.to(tl.bfloat16), vt, acc * alpha[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    heads = head_start + offs_h
    if SPLITS == 1:
        norm = tl.where(l_i > 0, l_i, 1.0)  # an empty row (a padded graph row) writes zeros
        base = out + row * (Q_HEADS * DIM) + heads[:, None] * DIM
        if FP4:
            tl.store(base + 2 * offs_p[None, :], (acc_e / norm[:, None] * v_global).to(out.dtype.element_ty), mask=hmask)
            tl.store(base + 2 * offs_p[None, :] + 1, (acc_o / norm[:, None] * v_global).to(out.dtype.element_ty), mask=hmask)
        else:
            tl.store(base + offs_d[None, :], (acc / norm[:, None] * v_global).to(out.dtype.element_ty), mask=hmask)
    else:
        slot = (row * Q_HEADS + heads) * SPLITS + split
        base = part_o + slot[:, None] * DIM
        if FP4:
            tl.store(base + 2 * offs_p[None, :], acc_e, mask=hmask)
            tl.store(base + 2 * offs_p[None, :] + 1, acc_o, mask=hmask)
        else:
            tl.store(base + offs_d[None, :], acc, mask=hmask)
        hm = offs_h < GROUP
        tl.store(part_ml + slot * 2, m_i, mask=hm)
        tl.store(part_ml + slot * 2 + 1, l_i, mask=hm)


@triton.jit
def _qsa_direct_combine(part_o, part_ml, out, v_global, Q_HEADS: tl.constexpr, DIM: tl.constexpr,
                        SPLITS: tl.constexpr, BLOCK_S: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    smask = offs_s < SPLITS
    slot = (row * Q_HEADS + head) * SPLITS + offs_s
    m = tl.load(part_ml + slot * 2, mask=smask, other=-float("inf"))
    l = tl.load(part_ml + slot * 2 + 1, mask=smask, other=0.0)
    mx = tl.max(m, 0)
    w = tl.math.exp2(m - tl.where(mx == -float("inf"), 0.0, mx))
    total = tl.sum(w * l, 0)
    offs_d = tl.arange(0, DIM)
    o = tl.load(part_o + slot[:, None] * DIM + offs_d[None, :], mask=smask[:, None], other=0.0)
    acc = tl.sum(w[:, None] * o, 0)
    total = tl.where(total > 0, total, 1.0)
    tl.store(out + (row * Q_HEADS + head) * DIM + offs_d, (acc / total * v_global).to(out.dtype.element_ty))


def qsa_sparse_decode_direct(
    q, k, v, k_sf, v_sf, table, table_rows, indices, lens, scale, k_global=1.0, v_global=1.0,
    *, block_n=64, warps=4, stages=2, chunk=None, decode=2,
):
    """arc-forge h22: sparse GQA decode straight off the pool for the selected slots.

    q [rows, q_heads, dim] BF16; k/v either packed NVFP4 (uint8 [slots, kv_heads, dim/2], low nibble = even
    feature, with k_sf/v_sf E4M3 [slots, kv_heads, dim/16]) or FP8 E4M3 [slots, kv_heads, dim] (k_sf/v_sf None);
    slot = table[table_rows[r], indices[r, j]] for 0 <= indices[r, j] < lens[r]. chunk (a multiple of block_n)
    splits the selection; None = one split. Returns BF16 [rows, q_heads, dim]; a row with no valid slot is 0.
    """
    rows, q_heads, dim = q.shape
    fp4 = k.dtype == torch.uint8
    if q.dtype != torch.bfloat16 or not q.is_contiguous() or not indices.is_contiguous():
        raise ValueError("direct decode needs a contiguous BF16 query and contiguous indices")
    if indices.dtype != torch.int32 or indices.shape[0] != rows:
        raise ValueError(f"indices must be int32 [{rows}, topk], got {indices.dtype} {tuple(indices.shape)}")
    if fp4:
        if k_sf is None or v_sf is None or k.shape[2] * 2 != dim:
            raise ValueError("packed NVFP4 K/V need their E4M3 scale planes and dim/2 bytes per head")
    elif k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn or k.shape[2] != dim:
        raise ValueError(f"direct decode reads packed NVFP4 or FP8 E4M3 K/V, got {k.dtype}/{v.dtype}")
    kv_heads = k.shape[1]
    group = q_heads // kv_heads
    topk = indices.shape[1]
    chunk = triton.cdiv(topk, block_n) * block_n if chunk is None else chunk
    if chunk <= 0 or chunk % block_n:
        raise ValueError(f"chunk {chunk} is not a multiple of block_n {block_n}")
    splits = triton.cdiv(topk, chunk)
    out = torch.empty(rows, q_heads, dim, device=q.device, dtype=q.dtype)
    if rows == 0:
        return out
    if splits > 1:
        part_o = torch.empty(rows, q_heads, splits, dim, device=q.device, dtype=torch.float32)
        part_ml = torch.empty(rows, q_heads, splits, 2, device=q.device, dtype=torch.float32)
    else:
        part_o = part_ml = out
    _qsa_direct_decode[(rows, kv_heads, splits)](
        q, k, v, k_sf if fp4 else k, v_sf if fp4 else v, table, table_rows, indices, lens, out, part_o, part_ml,
        float(scale), float(k_global), float(v_global), topk, table.stride(0), chunk,
        GROUP=group, BLOCK_M=max(16, triton.next_power_of_2(group)), BLOCK_N=block_n,
        Q_HEADS=q_heads, HEADS=kv_heads, DIM=dim, SPLITS=splits, FP4=fp4, DECODE=decode,
        num_warps=warps, num_stages=stages,
    )
    if splits > 1:
        _qsa_direct_combine[(rows, q_heads)](
            part_o, part_ml, out, float(v_global), Q_HEADS=q_heads, DIM=dim, SPLITS=splits,
            BLOCK_S=max(2, triton.next_power_of_2(splits)), num_warps=4,
        )
    return out


@triton.jit
def _qsa_nvfp4_expand_kv(
    k_fp4, v_fp4, k_sf, v_sf, req_to_token, req_indices, cu_k, out_k, out_v,
    k_global, v_global,
    heads: tl.constexpr, dim: tl.constexpr, req_stride: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Gather one request's packed FP4 context and write it out dequantized.

    Replaces index_select -> dequantize -> torch.cat: the packed bytes and their
    block scales are read once and the result lands directly at the request's
    cu_k offset, so no packed-to-dense copy of the whole layer buffer exists.
    """
    PACK: tl.constexpr = dim // 2
    SF: tl.constexpr = dim // 16
    pid_b, pid_h, pid_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(cu_k + pid_b).to(tl.int64)
    length = tl.load(cu_k + pid_b + 1).to(tl.int64) - start
    tokens = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    live = tokens < length
    req = tl.load(req_indices + pid_b).to(tl.int64)
    slots = tl.load(req_to_token + req * req_stride + tokens, mask=live, other=0).to(tl.int64)

    pack = tl.arange(0, PACK)
    packed_src = slots[:, None] * (heads * PACK) + pid_h * PACK + pack[None, :]
    scale_src = slots[:, None] * (heads * SF) + pid_h * SF + (pack // 8)[None, :]
    mask = live[:, None]
    kb = tl.load(k_fp4 + packed_src, mask=mask, other=0)
    vb = tl.load(v_fp4 + packed_src, mask=mask, other=0)
    k_step = tl.load(k_sf + scale_src, mask=mask, other=0.0).to(tl.float32) * tl.load(k_global)
    v_step = tl.load(v_sf + scale_src, mask=mask, other=0.0).to(tl.float32) * tl.load(v_global)
    dst = (start + tokens)[:, None] * (heads * dim) + pid_h * dim + 2 * pack[None, :]

    tl.store(out_k + dst, (_nvfp4_decode_nibble(kb, 0) * k_step).to(tl.bfloat16), mask=mask)
    tl.store(out_k + dst + 1, (_nvfp4_decode_nibble(kb, 1) * k_step).to(tl.bfloat16), mask=mask)
    tl.store(out_v + dst, (_nvfp4_decode_nibble(vb, 0) * v_step).to(tl.bfloat16), mask=mask)
    tl.store(out_v + dst + 1, (_nvfp4_decode_nibble(vb, 1) * v_step).to(tl.bfloat16), mask=mask)


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "qwen_sparse_nvfp4_expand_triton",
    "qwen_sparse_nvfp4_expand_segments_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "qwen_sparse_gqa_decode_selected_triton",
    "qwen_sparse_gqa_decode_selected_fp4_triton",
    "qsa_sparse_decode_direct",
]
