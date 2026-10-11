from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hc_gate_module(hc_count: int, hidden_size: int, dtype: torch.dtype) -> Module:
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"HC gate only supports BF16/FP16, got {dtype}")
    if (hc_count * hidden_size) % (256 * 8) != 0:
        raise RuntimeError("HC gate row must be divisible by 256 * 8")
    args = make_cpp_args(hc_count, hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "hc_gate",
        *args,
        cuda_files=["elementwise/hc_gate.cuh"],
        cuda_wrappers=[
            ("hc_gate", f"HcGateKernel<{args}>::run"),
            ("hc_apply", f"HcApplyKernel<{args}>::run"),
        ],
    )


def hc_gate(
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    normed = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty(
            (normed.shape[0], hc_count),
            dtype=torch.float32,
            device=normed.device,
        )
    else:
        out = out.reshape(normed.shape[0], hc_count)
    _jit_hc_gate_module(hc_count, hidden_size, normed.dtype).hc_gate(normed, inject_weight, out)
    return out


def hc_apply(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """arc-forge (ARC_HC_GATE_PRECOMPUTE): out = residual + gate * block_output per branch, gate from `hc_gate`.
    With the same gate this is hc_combine's output bit for bit (hc_combine_kernel's phase 2)."""
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    g = gate.reshape(-1, hc_count)
    if not (y.shape[0] == r.shape[0] == g.shape[0]):
        raise ValueError(f"hc_apply rows: block_output {y.shape[0]}, residual {r.shape[0]}, gate {g.shape[0]}")
    if out is None:
        out = torch.empty_like(r)
    _jit_hc_gate_module(hc_count, hidden_size, r.dtype).hc_apply(y, r, g, out)
    return out.view(residual.shape)


def compile_hc_gate(hc_count: int, hidden_size: int, dtype: torch.dtype) -> None:
    _jit_hc_gate_module(hc_count, hidden_size, dtype)
