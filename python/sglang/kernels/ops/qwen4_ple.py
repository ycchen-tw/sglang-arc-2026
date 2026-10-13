"""Bitwise-exact fused kernels for decode-sized Qwen4 PLE paths."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_QWEN4_NGRAM_SIZE = 3
_QWEN4_HEADS_PER_NGRAM = 8
_QWEN4_NGRAM_HEADS = 16
_QWEN4_HC_COUNT = 4
_QWEN4_HIDDEN_SIZE = 2560
_QWEN4_MAX_SHORT_CONV_STATE_LEN = 16


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round fp32 to BF16 precision while retaining an fp32 register."""

    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _qwen4_ngram_hash_kernel(
    contexts_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_outputs,
    eos_token_id,
    NGRAM_SIZE: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    NGRAM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < num_outputs
    token_idx = output_idx // NGRAM_HEADS
    head_idx = output_idx % NGRAM_HEADS
    context_base = token_idx * NGRAM_SIZE

    token_0 = tl.load(contexts_ptr + context_base, mask=mask, other=0)
    token_1 = tl.load(contexts_ptr + context_base + 1, mask=mask, other=0)
    token_2 = tl.load(contexts_ptr + context_base + 2, mask=mask, other=0)
    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)

    # Only the final position of each three-token context is materialized.
    previous_1 = tl.where(token_1 == eos_token_id, eos_token_id, token_1)
    previous_2 = tl.where(
        (token_0 == eos_token_id) | (token_1 == eos_token_id),
        eos_token_id,
        token_0,
    )
    mixed = (token_2 * multiplier_0) ^ (previous_1 * multiplier_1)
    mixed_3 = mixed ^ (previous_2 * multiplier_2)
    mixed = tl.where(head_idx < HEADS_PER_NGRAM, mixed, mixed_3)

    vocab_size = tl.load(vocab_sizes_ptr + head_idx, mask=mask, other=1)
    offset = tl.load(offsets_ptr + head_idx, mask=mask, other=0)
    tl.store(output_ptr + output_idx, mixed % vocab_size + offset, mask=mask)


def can_fuse_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> bool:
    """Return whether inputs match the fixed Qwen4 PLE hash contract."""

    return (
        contexts.is_cuda
        and contexts.dtype == torch.long
        and contexts.dim() == 2
        and contexts.shape[1] == _QWEN4_NGRAM_SIZE
        and contexts.is_contiguous()
        and multipliers.is_cuda
        and multipliers.dtype == torch.long
        and multipliers.numel() == _QWEN4_NGRAM_SIZE
        and vocab_sizes.is_cuda
        and vocab_sizes.dtype == torch.long
        and vocab_sizes.numel() == _QWEN4_NGRAM_HEADS
        and offsets.is_cuda
        and offsets.dtype == torch.long
        and offsets.numel() == _QWEN4_NGRAM_HEADS
    )


def fused_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Return the 16 Qwen4 PLE N-gram IDs in one kernel launch."""

    if not can_fuse_qwen4_ngram_hash(contexts, multipliers, vocab_sizes, offsets):
        raise ValueError("unsupported input for fused Qwen4 PLE N-gram hash")
    output = torch.empty(
        (contexts.shape[0], _QWEN4_NGRAM_HEADS),
        dtype=torch.long,
        device=contexts.device,
    )
    num_outputs = output.numel()
    if num_outputs:
        block_size = 256
        _qwen4_ngram_hash_kernel[(triton.cdiv(num_outputs, block_size),)](
            contexts,
            multipliers,
            vocab_sizes,
            offsets,
            output,
            num_outputs,
            eos_token_id,
            NGRAM_SIZE=_QWEN4_NGRAM_SIZE,
            HEADS_PER_NGRAM=_QWEN4_HEADS_PER_NGRAM,
            NGRAM_HEADS=_QWEN4_NGRAM_HEADS,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return output


@triton.jit
def _qwen4_gate_value_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    num_tokens,
    HC_COUNT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_group = tl.program_id(0)
    token = token_group // HC_COUNT
    hidden = tl.arange(0, BLOCK_SIZE)
    mask = (token < num_tokens) & (hidden < HIDDEN_SIZE)

    # `gate` arrives already rounded to bf16 by the eager multiply/reduce/divide;
    # every remaining eager bf16 rounding boundary is reproduced below.
    gate = tl.load(gate_ptr + token_group).to(tl.float32)
    magnitude = tl.maximum(tl.abs(gate), 1.0e-6)
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
    transformed = _round_bf16_to_fp32(root * sign)
    activated = _round_bf16_to_fp32(tl.sigmoid(transformed))

    value = tl.load(value_ptr + token * HIDDEN_SIZE + hidden, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offset = token_group * HIDDEN_SIZE + hidden
    tl.store(output_ptr + output_offset, activated * value, mask=mask)


def can_fuse_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether inputs match Qwen4's fixed BF16 gate/value contract."""

    return (
        gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.dim() == 3
        and gate.shape[1:] == (_QWEN4_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.is_cuda
        and value.dtype == gate.dtype
        and value.shape == (gate.shape[0], _QWEN4_HIDDEN_SIZE)
        and value.is_contiguous()
    )


def fused_qwen4_gate_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply Qwen4's post-reduction gate and value broadcast in one kernel."""

    if not can_fuse_qwen4_gate_value(gate, value):
        raise ValueError("unsupported input for fused Qwen4 PLE gate/value")
    output = torch.empty(
        (gate.shape[0], _QWEN4_HC_COUNT, _QWEN4_HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        _qwen4_gate_value_kernel[(gate.shape[0] * _QWEN4_HC_COUNT,)](
            gate,
            value,
            output,
            gate.shape[0],
            HC_COUNT=_QWEN4_HC_COUNT,
            HIDDEN_SIZE=_QWEN4_HIDDEN_SIZE,
            BLOCK_SIZE=4096,
            num_warps=8,
        )
    return output


@triton.jit
def _qwen4_short_conv_state_kernel(
    state_ptr,
    state_indices_ptr,
    x_ptr,
    conv_input_ptr,
    num_tokens,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_STATE_LEN: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)[:, None]
    state_col = tl.arange(0, BLOCK_STATE_LEN)[None, :]
    channel_mask = (token < num_tokens) & (channel < CHANNELS)
    state_mask = channel_mask & (state_col < STATE_LEN)
    state_index = tl.load(state_indices_ptr + token, mask=token < num_tokens, other=0)
    state_base = state_index * CHANNELS * STATE_LEN
    state_offset = state_base + channel * STATE_LEN + state_col
    output_base = token * CHANNELS * (STATE_LEN + 1)
    output_offset = output_base + channel * (STATE_LEN + 1) + state_col

    # Materialize every old state value before advancing the in-place cache,
    # so the convolution input equals the native index_select + cat result.
    old_state = tl.load(state_ptr + state_offset, mask=state_mask, other=0.0)
    tl.store(conv_input_ptr + output_offset, old_state, mask=state_mask)
    x = tl.load(x_ptr + token * CHANNELS + channel, mask=channel_mask, other=0.0)
    tl.store(
        conv_input_ptr + output_base + channel * (STATE_LEN + 1) + STATE_LEN,
        x,
        mask=channel_mask,
    )
    tl.debug_barrier()

    # Slot 0 is the CUDA-graph padding slot and may appear in several rows;
    # its post-step value is unobservable, so skip it to avoid duplicate writers.
    update_mask = state_mask & (state_col < STATE_LEN - 1) & (state_index != 0)
    next_value = tl.load(
        conv_input_ptr + output_offset + 1, mask=update_mask, other=0.0
    )
    tl.store(state_ptr + state_offset, next_value, mask=update_mask)
    tl.store(
        state_ptr + state_base + channel * STATE_LEN + STATE_LEN - 1,
        x,
        mask=channel_mask & (state_index != 0),
    )


def can_fuse_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
) -> bool:
    """Return whether decode state movement can use the exact fused kernel."""

    return (
        state.is_cuda
        and state.dtype in (torch.bfloat16, torch.float16)
        and state.dim() == 3
        and state.is_contiguous()
        and 0 < state.shape[2] <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and state_indices.is_cuda
        and state_indices.dtype == torch.long
        and state_indices.dim() == 1
        and state_indices.is_contiguous()
        and x.is_cuda
        and x.dtype == state.dtype
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape == (state_indices.shape[0], state.shape[1])
    )


def fused_qwen4_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Build ``[selected state, x]`` and advance real decode slots in one launch."""

    if not can_fuse_qwen4_short_conv_state(state, state_indices, x):
        raise ValueError("unsupported input for fused Qwen4 short-conv state")
    state_len = state.shape[2]
    conv_input = torch.empty(
        (x.shape[0], x.shape[1], state_len + 1),
        dtype=x.dtype,
        device=x.device,
    )
    if x.shape[0]:
        block_channels = 128
        block_state_len = triton.next_power_of_2(state_len)
        _qwen4_short_conv_state_kernel[
            (x.shape[0], triton.cdiv(x.shape[1], block_channels))
        ](
            state,
            state_indices,
            x,
            conv_input,
            x.shape[0],
            CHANNELS=state.shape[1],
            STATE_LEN=state_len,
            BLOCK_CHANNELS=block_channels,
            BLOCK_STATE_LEN=block_state_len,
            num_warps=8,
        )
    return conv_input


@triton.jit
def _qwen4_short_conv_verify_window_kernel(
    conv_input_ptr,
    valid_ptr,
    cache_ptr,
    cache_stride_req,
    ROW_WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)  # request * ROW_WIDTH + step, the valid_tokens order
    req = row // ROW_WIDTH
    step = row % ROW_WIDTH
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)  # flat (channel, state column)
    mask = offs < CHANNELS * STATE_LEN
    channel = offs // STATE_LEN
    col = offs % STATE_LEN
    valid = tl.load(valid_ptr + row)
    src = (
        req.to(tl.int64) * CHANNELS * (STATE_LEN + ROW_WIDTH)
        + channel * (STATE_LEN + ROW_WIDTH)
        + step
        + 1
        + col
    )
    value = tl.load(conv_input_ptr + src, mask=mask & valid, other=0.0)
    dst = req.to(tl.int64) * cache_stride_req + step * (CHANNELS * STATE_LEN) + offs
    tl.store(cache_ptr + dst, value, mask=mask)


def can_fuse_qwen4_short_conv_verify_window(
    conv_input: torch.Tensor,
    valid_tokens: torch.Tensor,
    intermediate_cache: torch.Tensor,
    row_width: int,
) -> bool:
    """Return whether the verify step-window copy can use the fused kernel."""

    if not (
        conv_input.is_cuda
        and conv_input.dim() == 3
        and conv_input.is_contiguous()
        and intermediate_cache.is_cuda
        and intermediate_cache.dtype == conv_input.dtype
        and intermediate_cache.dim() == 4
        and valid_tokens.is_cuda
        and valid_tokens.dtype == torch.bool
        and valid_tokens.is_contiguous()
    ):
        return False
    requests, channels, width = conv_input.shape
    state_len = width - row_width
    return (
        0 < state_len <= _QWEN4_MAX_SHORT_CONV_STATE_LEN
        and valid_tokens.numel() == requests * row_width
        and intermediate_cache.shape[0] >= requests
        and intermediate_cache.shape[1] >= row_width
        and intermediate_cache.shape[2:] == (channels, state_len)
        and intermediate_cache[0].is_contiguous()
    )


def fused_qwen4_short_conv_verify_window(
    conv_input: torch.Tensor,
    valid_tokens: torch.Tensor,
    intermediate_cache: torch.Tensor,
    row_width: int,
) -> None:
    """Write every verify step's post-step conv state straight into the intermediate cache.

    ``intermediate_cache[r, t] = conv_input[r, :, t + 1 : t + 1 + state_len]`` where step ``t`` of request ``r`` is
    valid, zeros elsewhere: the same bytes as the native unfold / permute / contiguous / where / copy_ chain, which
    moves the (requests, row_width, channels, state_len) window five times (L4 verify at 48 requests: 35 MB per pass,
    ~0.97 ms per step), with one read of conv_input and one write of the cache.
    """

    if not can_fuse_qwen4_short_conv_verify_window(
        conv_input, valid_tokens, intermediate_cache, row_width
    ):
        raise ValueError("unsupported input for fused Qwen4 short-conv verify window")
    requests, channels, width = conv_input.shape
    state_len = width - row_width
    if requests == 0:
        return
    block = 1024
    _qwen4_short_conv_verify_window_kernel[
        (requests * row_width, triton.cdiv(channels * state_len, block))
    ](
        conv_input,
        valid_tokens,
        intermediate_cache,
        intermediate_cache.stride(0),
        ROW_WIDTH=row_width,
        CHANNELS=channels,
        STATE_LEN=state_len,
        BLOCK=block,
        num_warps=4,
    )
