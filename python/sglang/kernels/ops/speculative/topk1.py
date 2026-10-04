from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_local_candidate_kernel(
    partial_vals,
    partial_indices,
    candidates,
    vocab_start,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    split = tl.argmax(vals, axis=0)
    max_val = tl.load(partial_vals + row * num_splits + split)
    local_index = tl.load(partial_indices + row * num_splits + split)
    tl.store(candidates + row * 2, max_val)
    # Global token ids are below 2^24, hence exactly representable in fp32.
    tl.store(candidates + row * 2 + 1, (vocab_start + local_index).to(tl.float32))


@triton.jit
def _draft_topk1_global_finalize_kernel(
    gathered_candidates,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    TP_SIZE: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    ADVANCE_POSITIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    mask = ranks < TP_SIZE
    row_base = row * (2 * TP_SIZE)
    vals = tl.load(
        gathered_candidates + row_base + ranks * 2,
        mask=mask,
        other=-float("inf"),
    )
    # Rank order follows global vocab order, so tl.argmax's first tie matches
    # argmax over the fully gathered logits.
    winner = tl.argmax(vals, axis=0)
    index = tl.load(gathered_candidates + row_base + winner * 2 + 1).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)
    if ADVANCE_POSITIONS:
        position = tl.load(positions + row)
        tl.store(positions + row, position + 1)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_splits: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``draft_tokens`` is given, the finalize kernel also stores the argmax
    into ``draft_tokens[:, draft_token_column]``, mutating the caller-owned
    buffer in place. ``topk_p`` is returned as constant 1.0: topk=1 drafting
    is greedy and the chain probabilities are unused downstream.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_splits,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


def draft_topk1_postprocess_distributed(
    local_next_token_logits: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    vocab_start_index: int,
    global_vocab_size: int,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """Exact global argmax from vocab-parallel draft logits.

    Each rank reduces only its local vocabulary shard, all-gathers one
    ``(max_logit, global_token_id)`` pair per row, then selects the winning
    rank. This replaces the full ``[batch, vocab]`` all-gather used by greedy
    top-1 MTP drafting. The candidate id is carried as fp32, which is exact for
    all supported vocabulary sizes. If ``positions`` is ``None``, the helper
    selects a token without advancing decode positions (draft-extend seeding).
    """
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    assert local_next_token_logits.ndim == 2
    assert local_next_token_logits.stride(1) == 1
    advance_positions = positions is not None
    if advance_positions:
        assert positions.ndim == 1 and positions.is_contiguous()
        assert positions.shape[0] == local_next_token_logits.shape[0]
        assert positions.device == local_next_token_logits.device
    assert vocab_start_index >= 0
    assert global_vocab_size > vocab_start_index
    assert global_vocab_size < 1 << 24

    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == local_next_token_logits.device
        assert draft_tokens.shape[0] == local_next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, local_width = local_next_token_logits.shape
    valid_local_vocab = min(local_width, global_vocab_size - vocab_start_index)
    assert valid_local_vocab > 0
    topk_p = torch.empty(
        (bs, 1), dtype=torch.float32, device=local_next_token_logits.device
    )
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=local_next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(valid_local_vocab, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=local_next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=local_next_token_logits.device
    )
    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        local_next_token_logits,
        partial_vals,
        partial_indices,
        local_next_token_logits.stride(0),
        valid_local_vocab,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    candidates = torch.empty(
        (bs, 2), dtype=torch.float32, device=local_next_token_logits.device
    )
    _draft_topk1_local_candidate_kernel[(bs,)](
        partial_vals,
        partial_indices,
        candidates,
        vocab_start_index,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    gathered = tensor_model_parallel_all_gather(candidates, dim=-1)
    tp_size = gathered.shape[-1] // 2
    _draft_topk1_global_finalize_kernel[(bs,)](
        gathered,
        topk_p,
        topk_index,
        positions if advance_positions else topk_index,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        TP_SIZE=tp_size,
        WRITE_DRAFT_TOKEN=write_draft_token,
        ADVANCE_POSITIONS=advance_positions,
        BLOCK=triton.next_power_of_2(tp_size),
        num_warps=1,
    )
    return topk_p, topk_index
