"""Qwen4-only FlashInfer MNNVL all-reduce + HC apply epilogue."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _module():
    from flashinfer.jit import current_compilation_context, gen_jit_spec

    from sglang.kernels.jit.utils import KERNEL_PATH

    flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[10])
    return gen_jit_spec(
        "sglang_qwen4_mnnvl_hc",
        [KERNEL_PATH / "csrc/distributed/qwen4_mnnvl_hc.cu"],
        extra_cuda_cflags=flags,
    ).build_and_load()


def qwen4_mnnvl_hc_combine(
    input_: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    workspace,
    *,
    launch_with_pdl: bool = True,
) -> None:
    """Reduce ``input_`` and write ``residual + gate * reduced`` to output.

    The caller owns all rank-invariant eligibility checks. Once this function
    enters the collective it deliberately has no fallback: swallowing an error
    on one rank while peers remain in MNNVL would hang or corrupt results.
    """
    from flashinfer.comm.trtllm_mnnvl_ar import (
        MNNVLAllreduceFusionStrategy,
    )

    strategy = MNNVLAllreduceFusionStrategy.select_strategy(
        workspace.tp_size,
        input_.shape[0],
        input_.shape[1],
        input_.dtype,
    )
    if not workspace.is_buffer_size_sufficient(
        workspace.tp_size,
        input_.shape[0],
        input_.shape[1],
        input_.dtype,
        strategy,
    ):
        raise ValueError("MNNVL workspace is too small for Qwen4 HC fusion")

    _module().qwen4_mnnvl_hc_combine(
        input_,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        launch_with_pdl,
        strategy == MNNVLAllreduceFusionStrategy.ONESHOT,
        residual,
        gate,
        output,
    )


def compile_qwen4_mnnvl_hc() -> None:
    """Warm the isolated JIT module before CUDA-graph capture."""
    _module()
