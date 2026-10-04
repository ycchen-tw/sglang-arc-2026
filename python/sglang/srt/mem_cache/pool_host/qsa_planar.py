"""Host mirror of the QSA HiSparse device pool (arc-forge M1a h02, design §1.5).

Layer-first: layer l is one registered uint8 [blocks, record] array, a record
holding every plane of one block back to back ([K | V] for FP8). The swap-in
kernel reads records straight from mapped host memory (h01); this pool only
allocates host blocks and writes records (device -> host backups). A resumed
session prefix (M1b h07) is read back two ways: load_blocks into the hot slab
at staging, fill_context_host into the landing zone for a prefill context.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

from sglang.kernels.ops.kvcache.hicache import transfer_hicache_all_layer
from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang.srt.mem_cache.qsa_hisparse_pool import (
    TRANSFER_BLOCK_QUOTA,
    HiSparseQSATokenToKVPool,
    QSAContextLandingZone,
    _check_transfer_indices,
    plane_ptr_tables,
)

logger = logging.getLogger(__name__)


class QSAPlanarHostPool(HiSparseHostPoolMixin, HostKVCache):
    """Host blocks for QSA HiSparse; page = one KV page of blocks, slots = blocks."""

    def __init__(
        self,
        *,
        device_pool: HiSparseQSATokenToKVPool,
        num_blocks: int,
        allocator_type: str = "default",
        admitted_bytes: Optional[int] = None,
    ):
        # Deliberately no HostKVCache.__init__: it sizes by tokens and ratio.
        spec = device_pool.plane_spec
        self.device_pool = device_pool
        self.pool_label = "qsa_hisparse"
        self.plane_spec = spec
        self.record_bytes = spec.record_bytes
        self.layer_num = device_pool.layer_num
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.page_size = device_pool.layout.page_blocks
        if num_blocks < self.page_size:
            raise ValueError(f"num_blocks {num_blocks} is below one host page ({self.page_size})")
        self.size = num_blocks
        self.size_per_token = self.record_bytes * self.layer_num
        self.dtype = torch.uint8
        self.device = "cpu"
        self.layout = "layer_first"
        self.pin_memory = True  # the kernel reads through mapped host pointers
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False
        self.lock = threading.RLock()

        requested = self.size * self.size_per_token
        available = host_memory_budget_bytes() if admitted_bytes is None else admitted_bytes
        if admitted_bytes is not None and requested != admitted_bytes:
            raise RuntimeError("QSA host pool differs from collective host admission")
        if requested > available:
            raise ValueError(
                f"QSA HiSparse host pool needs {requested / 1e9:.2f} GB "
                f"({num_blocks} blocks x {self.layer_num} layers x {self.record_bytes} B) but only "
                f"{available / 1e9:.2f} GB of host RAM is available after the reserve; "
                "the PLE table and everything else already loaded count against it"
            )
        gpu = torch.device(device_pool.device)
        alloc_func = ALLOC_MEMORY_FUNCS[gpu.type]
        self.kv_buffer = [
            alloc_func(
                (num_blocks, self.record_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            for _ in range(self.layer_num)
        ]
        tables = plane_ptr_tables(
            self.kv_buffer, spec.offsets, gpu, host_memory_registered=self.pin_memory
        )
        # Per K/V pair: host pointers at the pair's record offsets.
        self.pair_ptrs = [(tables[q], tables[q + 1]) for q, _ in spec.pairs()]
        logger.info(
            "QSA HiSparse host pool: %d blocks x %d layers x %d B = %.2f GB pinned",
            num_blocks,
            self.layer_num,
            self.record_bytes,
            requested / 1e9,
        )
        self.clear()

    def get_size_per_token(self):
        return self.size_per_token

    def init_kv_buffer(self):
        raise NotImplementedError("QSAPlanarHostPool allocates its buffers in __init__")

    def backup_blocks(self, device_blocks: torch.Tensor, host_blocks: torch.Tensor) -> None:
        """Device blocks -> host records, every layer and plane, on the current stream."""
        _check_transfer_indices(device_blocks, host_blocks)
        if device_blocks.numel() == 0:
            return  # a zero-length launch is an invalid grid
        for (_, nbytes), (hk, hv), (dk, dv) in zip(
            self.plane_spec.pairs(), self.pair_ptrs, self.device_pool.pair_ptrs
        ):
            transfer_hicache_all_layer(
                hk,
                hv,
                host_blocks,
                dk,
                dv,
                device_blocks,
                kv_cache_src_stride_bytes=nbytes,
                kv_cache_dst_stride_bytes=self.record_bytes,
                element_size=nbytes,
                block_quota=TRANSFER_BLOCK_QUOTA,
            )

    def load_blocks(self, host_blocks: torch.Tensor, device_blocks: torch.Tensor) -> None:
        """Host records -> device blocks, every layer and plane, on the current
        stream: backup_blocks reversed (M1b h07, the slab identity slots of a
        resumed prefix)."""
        _check_transfer_indices(host_blocks, device_blocks)
        if host_blocks.numel() == 0:
            return  # a zero-length launch is an invalid grid
        for (_, nbytes), (hk, hv), (dk, dv) in zip(
            self.plane_spec.pairs(), self.pair_ptrs, self.device_pool.pair_ptrs
        ):
            transfer_hicache_all_layer(
                dk,
                dv,
                device_blocks,
                hk,
                hv,
                host_blocks,
                kv_cache_src_stride_bytes=self.record_bytes,
                kv_cache_dst_stride_bytes=nbytes,
                element_size=nbytes,
                block_quota=TRANSFER_BLOCK_QUOTA,
            )

    def fill_context_host(
        self, layer_index: int, host_blocks: torch.Tensor, dst: QSAContextLandingZone
    ) -> None:
        """One layer's host records -> landing-zone blocks [0, n), on the current
        stream (M1b h07, design §4.3). The source table is the one-entry slice of
        this layer in the all-layer pointer table, so no table is built per call."""
        if not 0 <= layer_index < self.layer_num:
            raise ValueError(f"layer index {layer_index} outside [0, {self.layer_num})")
        if dst.plane_spec != self.plane_spec:
            raise ValueError("landing zone planes differ from the host records'")
        n = host_blocks.numel()
        if n > dst.capacity_blocks:
            raise ValueError(
                f"{n} host blocks exceed the landing zone's {dst.capacity_blocks}"
            )
        dst_blocks = dst.block_ids[:n]
        _check_transfer_indices(host_blocks, dst_blocks)
        if n == 0:
            return
        one = slice(layer_index, layer_index + 1)
        for (_, nbytes), (hk, hv), (zk, zv) in zip(
            self.plane_spec.pairs(), self.pair_ptrs, dst.pair_ptrs
        ):
            transfer_hicache_all_layer(
                zk,
                zv,
                dst_blocks,
                hk[one],
                hv[one],
                host_blocks,
                kv_cache_src_stride_bytes=self.record_bytes,
                kv_cache_dst_stride_bytes=nbytes,
                element_size=nbytes,
                block_quota=TRANSFER_BLOCK_QUOTA,
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        if device_pool is not self.device_pool or io_backend != "kernel":
            raise ValueError("QSAPlanarHostPool backs up its own device pool with the kernel backend")
        self.backup_blocks(device_indices, host_indices)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend, *, is_draft=False
    ) -> None:
        raise NotImplementedError(
            "QSA HiSparse loads through load_cache_to_device_buffer_planar"
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise NotImplementedError("QSA HiSparse host pool has no storage backend")

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise NotImplementedError("QSA HiSparse host pool has no storage backend")

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError("QSA HiSparse host pool has no storage backend")
