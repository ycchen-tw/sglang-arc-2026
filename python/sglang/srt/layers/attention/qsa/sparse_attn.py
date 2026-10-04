"""Validated sparse GQA operators migrated from the QSA reference branch."""

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


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
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
def _nvfp4_decode_nibble(byte, shift):
    """E2M1: bit3 sign, bits2-1 exponent, bit0 mantissa.

    Exponent 0 is the subnormal pair (0, 0.5); otherwise the value is
    2^(exponent-1) * (1 + mantissa/2), i.e. the {1,1.5,2,3,4,6} ladder.
    """
    nibble = (byte >> (4 * shift)) & 0x0F
    exponent = (nibble >> 1) & 0x03
    mantissa = (nibble & 0x01).to(tl.float32)
    magnitude = tl.where(
        exponent == 0,
        mantissa * 0.5,
        tl.exp2((exponent - 1).to(tl.float32)) * (1.0 + mantissa * 0.5),
    )
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


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "qwen_sparse_nvfp4_expand_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
]
