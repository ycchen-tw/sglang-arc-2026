# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import numpy as np
import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache

# arc-forge: ARC_GDN_HOST_CHUNK_INDICES=1 (opt-in, default 0) builds the chunk indices of an extend forward from the
# host's extend lengths (plan_chunk_indices, called by the GDN backend's init_forward_metadata) instead of the
# .tolist() below, which synchronizes the stream at the first GDN layer of every extend: the scheduler waits there
# for the previous decode cycle to finish and launches the rest of the eager forward from an empty GPU queue. The
# chunk offsets likewise (prepare_chunk_offsets' new_tensor([0]) is a pageable copy followed by a stream sync).
# Same values, dtype and device. With the flag a chunk-index or chunk-offset request on a CUDA cu_seqlens that was
# not planned raises instead of synchronizing. "check" also recomputes every plan the old way (synchronizing) and
# raises on any difference.
HOST_CHUNK_INDICES_ENV = "ARC_GDN_HOST_CHUNK_INDICES"
HOST_CHUNK_INDICES = os.environ.get(HOST_CHUNK_INDICES_ENV, "0")
if HOST_CHUNK_INDICES not in ("0", "1", "check"):
    raise ValueError(
        f"{HOST_CHUNK_INDICES_ENV} must be 0, 1 or check, got {HOST_CHUNK_INDICES!r}"
    )
_PLANNED_MAX = 8
_planned: list = []  # (cu_seqlens, chunk_size, chunk_indices, chunk_offsets), newest last


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def _prepare_chunk_indices_device(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    planned = _lookup(cu_seqlens, chunk_size, "indices")
    if planned is not None:
        return planned[2]
    return _prepare_chunk_indices_device(cu_seqlens, chunk_size)


def _lookup(cu_seqlens: torch.Tensor, chunk_size: int, what: str):
    for planned in reversed(_planned):
        if planned[0] is cu_seqlens and planned[1] == chunk_size:
            return planned
    if HOST_CHUNK_INDICES != "0" and cu_seqlens.is_cuda:
        raise RuntimeError(
            f"{HOST_CHUNK_INDICES_ENV}: chunk {what} (chunk size {chunk_size}) for a cu_seqlens of "
            f"{tuple(cu_seqlens.shape)} that plan_chunk_indices did not plan; computing them here synchronizes"
        )
    return None


def host_chunk_indices(lens: np.ndarray, chunk_size: int) -> np.ndarray:
    """_prepare_chunk_indices_device's [chunks, 2] int64 (sequence index, chunk index within it), in numpy."""
    counts = -(-lens // chunk_size)
    total = int(counts.sum())
    if total == 0:
        raise RuntimeError(f"{HOST_CHUNK_INDICES_ENV}: no chunks in host lengths {lens.tolist()}")
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.stack([np.cumsum(within == 0) - 1, within], 1)


def host_chunk_offsets(lens: np.ndarray, chunk_size: int) -> np.ndarray:
    """_prepare_chunk_offsets_device's [sequences + 1] int64 (cumulative chunk count), in numpy."""
    return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(-(-lens // chunk_size))])


def plan_chunk_indices(
    cu_seqlens: torch.Tensor, seq_lens_cpu, chunk_sizes
) -> None:
    """prepare_chunk_indices(cu_seqlens, size) and prepare_chunk_offsets(cu_seqlens, size) for each size, from the
    sequence lengths the caller holds on the host (seq_lens_cpu[i] == cu_seqlens[i + 1] - cu_seqlens[i]): built in
    numpy, copied from pinned memory without blocking, and returned by those two for this cu_seqlens object. The
    sequence index column is the reference's (indices == 0).cumsum() - 1, so a zero-length sequence numbers the same
    way. The offsets are int64 like the reference's (cumsum promotes an int32 cu_seqlens)."""
    if not cu_seqlens.is_cuda:
        raise RuntimeError("plan_chunk_indices is for CUDA cu_seqlens")
    lens = np.asarray(seq_lens_cpu, dtype=np.int64)
    if lens.ndim != 1 or lens.shape[0] + 1 != cu_seqlens.shape[0] or (lens < 0).any():
        raise RuntimeError(
            f"{HOST_CHUNK_INDICES_ENV}: host lengths {lens.tolist()} for a cu_seqlens of {tuple(cu_seqlens.shape)}"
        )
    for size in dict.fromkeys(chunk_sizes):
        host = torch.from_numpy(host_chunk_indices(lens, size)).to(cu_seqlens.dtype).pin_memory()
        chunk_indices = host.to(cu_seqlens.device, non_blocking=True)
        chunk_offsets = torch.from_numpy(host_chunk_offsets(lens, size)).pin_memory().to(
            cu_seqlens.device, non_blocking=True
        )
        if HOST_CHUNK_INDICES == "check":
            for what, planned, reference in (
                ("indices", chunk_indices, _prepare_chunk_indices_device.__wrapped__(cu_seqlens, size)),
                ("offsets", chunk_offsets, _prepare_chunk_offsets_device.__wrapped__(cu_seqlens, size)),
            ):
                if reference.dtype != planned.dtype or not torch.equal(reference, planned):
                    raise RuntimeError(
                        f"{HOST_CHUNK_INDICES_ENV}=check: host chunk {what} (size {size}, lengths {lens.tolist()}) "
                        f"differ from the device computation"
                    )
        _planned.append((cu_seqlens, size, chunk_indices, chunk_offsets))
    del _planned[:-_PLANNED_MAX]


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    planned = _lookup(cu_seqlens, chunk_size, "offsets")
    if planned is not None:
        return planned[3]
    return _prepare_chunk_offsets_device(cu_seqlens, chunk_size)


@tensor_cache
def _prepare_chunk_offsets_device(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
