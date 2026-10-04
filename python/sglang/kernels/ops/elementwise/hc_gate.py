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
        cuda_wrappers=[("hc_gate", f"HcGateKernel<{args}>::run")],
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


def compile_hc_gate(hc_count: int, hidden_size: int, dtype: torch.dtype) -> None:
    _jit_hc_gate_module(hc_count, hidden_size, dtype)
