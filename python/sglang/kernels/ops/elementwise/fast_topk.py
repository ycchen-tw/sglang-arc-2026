from __future__ import annotations

import os
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

_FAST_TOPK_SUPPORTED_K = (512, 2048)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


# Test flag (arc-forge QSA HiSparse gates): the radix kernel collects its picks
# with shared-memory atomics, so the output order, the choice among equal
# scores, and -- once more than 4096 candidates share the threshold bin -- the
# selected set itself change from run to run. With the flag on, every call
# takes the exact, deterministic top-k below instead, so two runs (or two arms)
# of one input select and attend identically. Off by default: production calls
# the kernel unchanged.
_ARC_QSA_TOPK_EXACT = _env_flag("ARC_QSA_TOPK_EXACT")


def _exact_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor],
) -> torch.Tensor:
    """Exact top-k of score[b, s : s + lengths[b]]: ties go to the lowest index,
    each row ascending, -1 padded (the kernel's contract with a fixed order).
    Torch only, no host sync, so CUDA-graph capturable."""
    batch, width = score.shape
    out = torch.full((batch, topk), -1, dtype=torch.int32, device=score.device)
    if batch == 0 or width == 0:
        return out
    if row_starts is None:
        starts = torch.zeros(batch, dtype=torch.int64, device=score.device)
    else:
        starts = row_starts.to(torch.int64)
    cols = torch.arange(width, device=score.device, dtype=torch.int64)
    rel = cols[None, :] - starts[:, None]
    valid = (rel >= 0) & (rel < lengths.to(torch.int64)[:, None])
    masked = torch.where(valid, score, torch.full_like(score, -float("inf")))
    k = min(topk, width)
    kth = torch.topk(masked, k, dim=1, sorted=True).values[:, -1:]
    greater = valid & (masked > kth)
    tie = valid & (masked == kth)
    need = k - greater.sum(dim=1, keepdim=True)
    chosen = greater | (tie & (torch.cumsum(tie.to(torch.int64), dim=1) <= need))
    big = torch.iinfo(torch.int64).max
    picked = torch.topk(
        torch.where(chosen, rel, torch.full_like(rel, big)),
        k,
        dim=1,
        largest=False,
        sorted=True,
    ).values
    out[:, :k] = torch.where(picked == big, -1, picked).to(torch.int32)
    return out


@cache_once
def _jit_fast_topk_module(topk: int) -> Module:
    """Compile and cache the JIT fast top-k module for a given k."""
    # Checks on the compile key live here, not in `fast_topk`: `cache_once`
    # keys on `topk`, so this runs once per specialisation.
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    args = make_cpp_args(topk, is_arch_support_pdl())
    return load_jit(
        "fast_topk",
        *args,
        cuda_files=["elementwise/fast_topk.cuh"],
        cuda_wrappers=[
            ("fast_topk", f"FastTopKKernel<{args}>::run"),
            (
                "fast_topk_zero_starts",
                f"FastTopKKernel<{args}>::run_zero_starts",
            ),
        ],
    )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-row top-k selection over a fp32 score matrix.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b]]`` and returns their
    indices relative to ``row_starts[b]``. Slots beyond ``lengths[b]`` are -1.
    Output order within a row is unspecified (atomic collection order);
    ARC_QSA_TOPK_EXACT=1 makes it exact and deterministic (see _exact_topk).

    Parameters
    ----------
    score      : CUDA fp32 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row; 512 or 2048
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros

    Returns
    -------
    CUDA int32 tensor [B, topk]
    """
    if _ARC_QSA_TOPK_EXACT:
        if topk not in _FAST_TOPK_SUPPORTED_K:
            raise RuntimeError(
                f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
            )
        return _exact_topk(score, lengths, topk, row_starts)
    batch = score.shape[0]
    indices = score.new_empty((batch, topk), dtype=torch.int32)

    module = _jit_fast_topk_module(topk)
    if row_starts is None:
        # The CUDA kernel already treats a null pointer as row_start=0. Use a
        # dedicated wrapper rather than allocate/fill a zeros tensor per call.
        module.fast_topk_zero_starts(score, indices, lengths)
    else:
        module.fast_topk(score, row_starts, indices, lengths)
    return indices
