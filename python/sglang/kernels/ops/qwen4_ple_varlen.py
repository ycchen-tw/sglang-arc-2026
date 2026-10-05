"""Unpadded (ragged) Qwen4 PLE short convolution for extend batches.

The padded path materialises (requests, longest row, channels) three times (padded input, conv input with the state,
conv output); with 8 prefill requests and an 8192-token chunk that is up to 8 x 8192 x 10240 x 2 B = 1.25 GiB each.
This computes the same dilated depthwise convolution straight from the flat token rows:

    out[t] = silu(sum_k w[c, k] * ext[o + k * dilation]),  ext = state (state_len cols) ++ the request's rows

with fp32 accumulation rounded to the input dtype before the SiLU (the conv1d output rounding) and the SiLU in fp32
rounded once more -- bitwise equal to the padded F.conv1d + F.silu path on sm_120 (BF16, 5 batch layouts incl.
8185 + 7 x 1 rows).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ple_conv_varlen_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    start_ptr,
    req_ptr,
    off_ptr,
    out_ptr,
    T,
    C,
    x_stride,
    s_stride_n,
    s_stride_c,
    s_stride_w,
    w_stride_c,
    out_stride,
    K: tl.constexpr,
    DIL: tl.constexpr,
    SL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mt = t < T
    mc = c < C
    r = tl.load(req_ptr + t, mask=mt, other=0)
    o = tl.load(off_ptr + t, mask=mt, other=0)
    start = tl.load(start_ptr + r, mask=mt, other=0)
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    m2 = mt[:, None] & mc[None, :]
    for k in tl.static_range(K):
        col = o + k * DIL  # column of [state | rows]
        in_seq = col >= SL
        xrow = start + col - SL
        xv = tl.load(
            x_ptr + xrow[:, None] * x_stride + c[None, :],
            mask=m2 & in_seq[:, None],
            other=0.0,
        )
        sv = tl.load(
            state_ptr
            + r[:, None] * s_stride_n
            + c[None, :] * s_stride_c
            + col[:, None] * s_stride_w,
            mask=m2 & (~in_seq)[:, None],
            other=0.0,
        )
        v = tl.where(in_seq[:, None], xv.to(tl.float32), sv.to(tl.float32))
        wk = tl.load(w_ptr + c * w_stride_c + k, mask=mc, other=0.0).to(tl.float32)
        acc += v * wk[None, :]
    y = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    y = y / (1.0 + tl.exp(-y))
    tl.store(
        out_ptr + t[:, None] * out_stride + c[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=m2,
    )


def qwen4_ple_conv_varlen(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    starts: torch.Tensor,
    req_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """silu(dilated depthwise conv) of an extend batch, (T, C) in x.dtype.

    x: (T, C) token rows (row stride free, column stride 1); state: (N, C, state_len) per request in x.dtype;
    weight: (C, 1, K) in x.dtype; starts: (N,) first row of each request; req_indices / token_offsets: (T,).
    """
    T, C = x.shape
    K = weight.shape[-1]
    state_len = (K - 1) * dilation
    if x.stride(1) != 1:
        raise ValueError(f"x must have unit column stride, got strides {x.stride()}")
    if state.shape[1:] != (C, state_len) or state.dtype != x.dtype:
        raise ValueError(f"state {tuple(state.shape)} {state.dtype} does not match x {tuple(x.shape)} {x.dtype}")
    if weight.dtype != x.dtype:
        raise ValueError(f"weight {weight.dtype} must be cast to {x.dtype} (the padded path's conv dtype)")
    out = torch.empty_like(x)
    if T == 0:
        return out
    w = weight.reshape(C, K)
    block_t, block_c = 32, 128
    grid = (triton.cdiv(T, block_t), triton.cdiv(C, block_c))
    _ple_conv_varlen_kernel[grid](
        x,
        state,
        w,
        starts,
        req_indices,
        token_offsets,
        out,
        T,
        C,
        x.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        w.stride(0),
        out.stride(0),
        K=K,
        DIL=dilation,
        SL=state_len,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
    )
    return out


def qwen4_ple_conv_state_at(
    x: torch.Tensor,
    state: torch.Tensor,
    starts: torch.Tensor,
    offsets: torch.Tensor,
    state_len: int,
) -> torch.Tensor:
    """The state_len columns [offsets, offsets + state_len) of each request's [state | rows], as (N, C, state_len):
    the padded path's conv_input.gather at those offsets, without building conv_input."""
    cols = offsets[:, None] + torch.arange(state_len, device=x.device, dtype=torch.long)[None, :]
    from_rows = cols >= state_len
    rows = (starts[:, None] + cols - state_len).clamp(min=0, max=max(x.shape[0] - 1, 0))
    row_vals = x[rows].transpose(1, 2)
    state_vals = state.gather(2, cols.clamp(max=state_len - 1)[:, None, :].expand(-1, state.shape[1], -1))
    return torch.where(from_rows[:, None, :], row_vals, state_vals)
