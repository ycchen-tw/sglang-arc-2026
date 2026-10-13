from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_GATHER_BLOCK_SIZE = 64


class HiSparseSpecState(NamedTuple):
    """Persistent cache state and reusable miss workspace for speculative swap.

    ``cache_index`` stores the two int64 hash banks as
    ``[num_requests, 2, hash_size]``. ``cache_policy`` uses a control-plane row
    for the packed CLOCK states followed by one reference-epoch row per
    request: ``[1 + num_requests, hot_buffer_size]``.

    ``scratch_locs`` and ``scratch_state`` hold reusable miss locations,
    counters, and metadata shared by all layers.
    """

    cache_index: torch.Tensor
    cache_policy: torch.Tensor
    scratch_locs: torch.Tensor
    scratch_state: torch.Tensor


@cache_once
def _jit_spec_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_steps: int,
    record_miss_plan: bool,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        item_size_bytes,
        num_steps,
        record_miss_plan,
        is_arch_support_pdl(),
    )
    return load_jit(
        "hisparse_spec",
        *template_args,
        cuda_files=["kvcacheio/hisparse_spec.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_spec",
                f"load_cache_to_device_buffer_spec<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_spec_mla(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseSpecState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the single-step HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.
    """
    _, num_steps, num_top_k = top_k_tokens.shape
    if not 2 <= num_steps <= 4:
        raise ValueError(
            f"HiSparse speculative swap requires 2-4 steps, got {num_steps}."
        )
    hot_buffer_size = state.cache_policy.size(1)
    page_size = device_buffer_tokens.size(1) - hot_buffer_size
    item_size_bytes = host_cache.stride(0) * host_cache.element_size()
    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32.")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32.")
        plan_capacity = num_steps * num_top_k
        batch_size = top_k_tokens.size(0)
        if (
            miss_src.ndim != 2
            or miss_dst.ndim != 2
            or miss_src.size(0) < batch_size
            or miss_dst.size(0) < batch_size
            or miss_src.size(1) < plan_capacity
            or miss_dst.size(1) < plan_capacity
        ):
            raise ValueError(
                "speculative miss_src/miss_dst must have shape "
                f"[batch, >= steps * top_k] (capacity {plan_capacity})."
            )
        if miss_count.ndim != 1 or miss_count.numel() < batch_size:
            raise ValueError("speculative miss_count must have shape [batch].")
        if miss_src.stride(0) != miss_dst.stride(0):
            raise ValueError("miss_src/miss_dst row strides must match.")
    else:
        if miss_dst is not None or miss_count is not None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        empty = torch.empty(0)
        miss_src = miss_dst = miss_count = empty

    module = _jit_spec_module(
        item_size_bytes,
        _GATHER_BLOCK_SIZE,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
    )

    module.load_cache_to_device_buffer_spec(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        state.cache_index,
        state.cache_policy,
        state.scratch_locs,
        state.scratch_state,
        num_real_reqs,
        page_size,
        miss_src,
        miss_dst,
        miss_count,
    )


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
    record_miss_plan: bool = False,
    skip_io: bool = False,
) -> Module:
    # record_miss_plan / skip_io are compile-time kernel flags; the
    # (False, False) production instantiation stays byte-identical.
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        record_miss_plan,
        skip_io,
    )
    cache_args = make_cpp_args(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        record_miss_plan,
        skip_io,
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_copy_planned_module(
    block_size: int,
    is_mla: bool,
    is_dsv4_layout: bool,
    skip_io: bool,
) -> Module:
    template_args = make_cpp_args(block_size, is_mla, is_dsv4_layout, skip_io)
    return load_jit(
        "sparse_copy_planned",
        block_size,
        is_mla,
        is_dsv4_layout,
        skip_io,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned",
                f"copy_cache_planned<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_dsv4_transfer_module(block_size: int) -> Module:
    template_args = make_cpp_args(block_size)
    return load_jit(
        "sparse_cache_dsv4_transfer",
        block_size,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "transfer_cache_dsv4_mla",
                f"transfer_cache_dsv4_mla<{template_args}>",
            )
        ],
    )


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers."""
    module = _jit_dsv4_transfer_module(block_size)
    module.transfer_cache_dsv4_mla(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
    )


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
    miss_src: torch.Tensor | None,
    miss_dst: torch.Tensor | None,
    miss_count: torch.Tensor | None,
    skip_io: bool,
) -> None:
    assert hot_buffer_size >= num_top_k, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    )

    record_miss_plan = miss_src is not None
    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
        record_miss_plan=record_miss_plan,
        skip_io=skip_io,
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    if record_miss_plan:
        assert miss_dst is not None and miss_count is not None
        assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
        assert miss_count.dtype == torch.int32
        # The kernel indexes both plan rows with one stride.
        assert miss_src.stride(0) == miss_dst.stride(0)
    else:
        # Unused sentinels; the RecordMissPlan=false instantiation never reads them.
        miss_src = miss_dst = miss_count = empty

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        miss_src,
        miss_dst,
        miss_count,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    Optional miss_src/miss_dst/miss_count record the miss plan for replay by
    copy_cache_planned_mla; skip_io elides only the KV bytes (timing probe).
    """
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
    )


def copy_cache_planned_mla(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_blocks: int = 4,
    block_size: int = 1024,
    is_dsv4_layout: bool = False,
    skip_io: bool = False,
) -> None:
    """Replay a recorded miss plan (host_cache -> device_buffer) for a skip layer.

    IO-only, no planning; the small fixed grid keeps the SM footprint low while
    overlapped on a side stream. The anchor's slot table stays valid (lockstep).
    """
    assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
    assert miss_count.dtype == torch.int32
    module = _jit_copy_planned_module(block_size, True, is_dsv4_layout, skip_io)
    empty = torch.empty(0)
    module.copy_cache_planned(
        miss_src,
        miss_dst,
        miss_count,
        num_real_reqs,
        host_cache,
        empty,
        device_buffer,
        empty,
        num_blocks,
        item_size_bytes,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
    )


# ---------------------------------------------------------------------------
# Planar records (arc-forge QSA HiSparse h01)
#
# A host item is one record packing every plane of a 4-token QSA block back to
# back ([K | V] for FP8, [K | V | K scale | V scale] for NVFP4); on the device
# each plane is its own buffer with per-item stride = that plane's bytes. The
# resolve/LRU is the same kernel as load_cache_to_device_buffer_mla; only the
# miss copy differs (see PlanarSpec / copy_record_planar in hisparse.cuh).
# ---------------------------------------------------------------------------

# lru_slots / s_lru_slots_out / s_hash_vals are int16, so the hot buffer must
# stay below 32768 slots; the largest multiple of 16 is 32752.
PLANAR_MAX_HOT_BUFFER = 32752
PLANAR_MAX_PLANES = 4


@functools.cache
def _jit_planar_module(
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    plane_bytes: tuple[int, int, int, int],
    skip_io: bool,
) -> Module:
    template_args = make_cpp_args(
        block_size, num_top_k, hot_buffer_size, *plane_bytes, skip_io
    )
    return load_jit(
        "sparse_cache_planar",
        *template_args,
        cuda_files=["kvcacheio/hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_planar",
                f"load_cache_to_device_buffer_planar<{template_args}>",
            )
        ],
    )


def _check_index_tensor(name: str, t: torch.Tensor, dtypes: tuple, ndim: int) -> None:
    if t.dtype not in dtypes or t.dim() != ndim or not t.is_cuda:
        raise ValueError(
            f"{name} must be a {ndim}-D CUDA tensor of {dtypes}, "
            f"got {t.dtype} {tuple(t.shape)} on {t.device}"
        )
    if t.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its last dim")


def load_cache_to_device_buffer_planar(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_records: torch.Tensor,
    device_planes: tuple[torch.Tensor, ...],
    plane_bytes: tuple[int, ...],
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor,
    skip_io: bool = False,
    rows_per_req: int = 1,
    ring_width: int = 1,
    overflow_flag: torch.Tensor | None = None,
    miss_stats: torch.Tensor | None = None,
) -> None:
    """HiSparse swap-in whose miss copy moves one planar host record per item.

    host_records: uint8 [items, sum(plane_bytes)] in registered host memory (one
    layer). device_planes[q]: contiguous CUDA tensor of any dtype whose bytes are
    items x plane_bytes[q]; item d of plane q starts at byte d * plane_bytes[q].
    Rows of top_k_tokens beyond num_real_reqs[0] get -1 and are not resolved.
    All checks are metadata-only, so the call is CUDA-graph capturable.

    Multi-row verify (arc-forge M5 h16): each request owns rows_per_req (K)
    consecutive rows of top_k_tokens / seq_lens / top_k_device_locs /
    req_pool_indices, resolved one after another against its tables; num_real_reqs
    counts requests. Selected blocks >= max(first row's seq_len - 1, hot_buffer_size)
    resolve to ring slot hot_buffer_size + block % ring_width. With K > 1,
    overflow_flag (int32 [1] on the device, never cleared here) is set when a
    row evicts a block an earlier row of its request selected.

    miss_stats (arc-forge 10-13 review E5, ARC_HISPARSE_MISS_STATS): int64 [8] on
    the device, never cleared here; every slow-path row adds [0] misses (records
    copied from host), [1] misses^2, [3] 1, [4] ring hits, [5] LRU hits and raises
    [2] to its misses (max); a fast-path row (seq_len <= hot_buffer_size) adds 1 to
    [6]. [7] unused.
    """
    num_planes = len(plane_bytes)
    if not 1 <= num_planes <= PLANAR_MAX_PLANES or len(device_planes) != num_planes:
        raise ValueError(
            f"need 1..{PLANAR_MAX_PLANES} planes and one tensor per plane, got "
            f"{num_planes} sizes and {len(device_planes)} tensors"
        )
    num_items = None
    for q, (plane, nbytes) in enumerate(zip(device_planes, plane_bytes)):
        if nbytes <= 0 or nbytes % 16 != 0:
            raise ValueError(
                f"plane {q}: bytes {nbytes} must be a positive multiple of 16"
            )
        if not plane.is_cuda or not plane.is_contiguous():
            raise ValueError(f"plane {q} must be a contiguous CUDA tensor")
        total = plane.numel() * plane.element_size()
        if total % nbytes != 0:
            raise ValueError(
                f"plane {q}: {total} bytes is not a whole number of {nbytes} B items"
            )
        if num_items is not None and total // nbytes != num_items:
            raise ValueError(
                f"plane {q} holds {total // nbytes} items, plane 0 holds {num_items}"
            )
        num_items = total // nbytes
    record_bytes = sum(plane_bytes)
    if (
        host_records.dtype != torch.uint8
        or host_records.dim() != 2
        or host_records.size(1) != record_bytes
        or host_records.stride() != (record_bytes, 1)
        or host_records.device.type != "cpu"
    ):
        raise ValueError(
            f"host_records must be contiguous uint8 [items, {record_bytes}] on the host, got "
            f"{host_records.dtype} {tuple(host_records.shape)} strides {host_records.stride()} "
            f"on {host_records.device}"
        )
    if (
        not num_top_k <= hot_buffer_size <= PLANAR_MAX_HOT_BUFFER
        or hot_buffer_size % 16 != 0
    ):
        raise ValueError(
            f"hot_buffer_size {hot_buffer_size} must be a multiple of 16 in "
            f"[num_top_k={num_top_k}, {PLANAR_MAX_HOT_BUFFER}]"
        )
    _check_index_tensor("top_k_tokens", top_k_tokens, (torch.int32,), 2)
    _check_index_tensor("top_k_device_locs", top_k_device_locs, (torch.int32,), 2)
    if (
        top_k_tokens.size(1) != num_top_k
        or top_k_device_locs.shape != top_k_tokens.shape
    ):
        raise ValueError(
            f"top_k_tokens {tuple(top_k_tokens.shape)} and top_k_device_locs "
            f"{tuple(top_k_device_locs.shape)} must both be [bs, {num_top_k}]"
        )
    _check_index_tensor("device_buffer_tokens", device_buffer_tokens, (torch.int32,), 2)
    _check_index_tensor("device_buffer_locs", device_buffer_locs, (torch.int32,), 2)
    if (
        device_buffer_tokens.shape != device_buffer_locs.shape
        or device_buffer_tokens.stride(0) != device_buffer_locs.stride(0)
        or device_buffer_tokens.size(1) <= hot_buffer_size
    ):
        raise ValueError(
            "device_buffer_tokens/locs must share shape and row stride [reqs, > hot_buffer_size] "
            f"(slot hot_buffer_size is the newest slot), got {tuple(device_buffer_tokens.shape)} "
            f"and {tuple(device_buffer_locs.shape)}"
        )
    _check_index_tensor("lru_slots", lru_slots, (torch.int16,), 2)
    if lru_slots.size(1) != hot_buffer_size:
        raise ValueError(
            f"lru_slots must be [reqs, {hot_buffer_size}], got {tuple(lru_slots.shape)}"
        )
    _check_index_tensor("host_cache_locs", host_cache_locs, (torch.int64,), 2)
    bs = top_k_tokens.size(0)
    for name, t in (("req_pool_indices", req_pool_indices), ("seq_lens", seq_lens)):
        _check_index_tensor(name, t, (torch.int32, torch.int64), 1)
        if t.numel() < bs:
            raise ValueError(f"{name} has {t.numel()} rows, batch has {bs}")
    if rows_per_req < 1 or bs % rows_per_req:
        raise ValueError(f"{bs} top-k rows are not a multiple of rows_per_req {rows_per_req}")
    if ring_width < 1 or hot_buffer_size + ring_width > device_buffer_tokens.size(1):
        raise ValueError(
            f"ring_width {ring_width} does not fit the {device_buffer_tokens.size(1) - hot_buffer_size} "
            "slots after the LRU slots"
        )
    if overflow_flag is None:
        if rows_per_req > 1:
            raise ValueError("a multi-row verify swap-in needs overflow_flag")
        overflow_flag = torch.empty(0, dtype=torch.int32, device=top_k_tokens.device)
    elif (
        overflow_flag.dtype != torch.int32
        or overflow_flag.numel() < 1
        or not overflow_flag.is_cuda
    ):
        raise ValueError("overflow_flag must be an int32 CUDA tensor with >= 1 element")
    if (
        num_real_reqs.dtype != torch.int32
        or num_real_reqs.numel() != 1
        or not num_real_reqs.is_cuda
    ):
        raise ValueError("num_real_reqs must be a 1-element int32 CUDA tensor")
    if miss_stats is None:
        miss_stats = torch.empty(0, dtype=torch.int64, device=top_k_tokens.device)
    elif (
        miss_stats.dtype != torch.int64
        or miss_stats.shape != (8,)
        or not miss_stats.is_cuda
        or miss_stats.stride() != (1,)
    ):
        raise ValueError(
            f"miss_stats must be a contiguous int64 [8] CUDA tensor, got {miss_stats.dtype} "
            f"{tuple(miss_stats.shape)} strides {miss_stats.stride()} on {miss_stats.device}"
        )

    padded_bytes = tuple(plane_bytes) + (0,) * (PLANAR_MAX_PLANES - num_planes)
    empty = torch.empty(0, dtype=torch.uint8, device=top_k_tokens.device)
    planes = tuple(device_planes) + (empty,) * (PLANAR_MAX_PLANES - num_planes)
    module = _jit_planar_module(
        block_size, num_top_k, hot_buffer_size, padded_bytes, skip_io
    )
    module.load_cache_to_device_buffer_planar(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_records,
        *planes,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        rows_per_req,
        ring_width,
        overflow_flag,
        miss_stats,
    )
