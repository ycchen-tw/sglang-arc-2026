"""QSA HiSparse device pool and allocator (arc-forge M1a h02, design §1.2-1.3, §5.3).

The full-attention KV of the QSA layers keeps only device rows: a prefill area
handed out while a request prefills or stages, and a static hot slab per
request slot (layout in qsa_hisparse_layout.py). Everything the scheduler and
the QSA index see stays *logical*: the paged logical allocator hands out
logical token slots, the compressed-K / pending rings are sized by the logical
capacity, and ``mapping`` (logical block -> device block) turns logical write
locations into device rows. The HiSparse coordinator owns every slab write
after staging.
"""

from __future__ import annotations

import logging
from typing import List

import torch

from sglang.kernels.ops.kvcache.hicache import transfer_hicache_all_layer
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
from sglang.srt.mem_cache.pool_host.common import make_kernel_ptr_table
from sglang.srt.mem_cache.qsa_hisparse_layout import (
    PlaneSpec,
    QSAHiSparseLayout,
    kv_plane_spec,
)
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.utils import get_num_new_pages

logger = logging.getLogger(__name__)

# CTAs per transfer_hicache_all_layer launch. G4 (h01 bench): staging D2D of
# 4096 blocks x 12 layers takes 0.29 ms at 32 and 3.4 ms at the default 2; D2H
# is PCIe-bound either way. Small decode backups launch fewer CTAs anyway.
TRANSFER_BLOCK_QUOTA = 32


def plane_ptr_tables(
    per_layer_bases: List[torch.Tensor],
    offsets: tuple[int, ...],
    device,
    *,
    host_memory_registered: bool,
) -> List[torch.Tensor]:
    """One uint64 [layers] pointer table per byte offset into each layer's base."""
    base = make_kernel_ptr_table(
        per_layer_bases, device, host_memory_registered=host_memory_registered
    ).view(torch.int64)
    return [(base + off).view(torch.uint64) for off in offsets]


def _check_transfer_indices(src: torch.Tensor, dst: torch.Tensor) -> None:
    if src.shape != dst.shape or src.dim() != 1:
        raise ValueError(
            f"block index lists must be 1-D and equal length, got {tuple(src.shape)} and {tuple(dst.shape)}"
        )
    if not (src.is_cuda and dst.is_cuda):
        raise ValueError("block index lists must be CUDA tensors")


class HiSparseQSATokenToKVPool(QSATokenToKVPool):
    """QSATokenToKVPool whose full-attention KV holds device rows, not logical slots."""

    def __init__(
        self,
        *,
        logical_size: int,
        prefill_tokens: int,
        hot_blocks: int,
        page_size: int,
        qsa_compress_ratio: int,
        num_request_slots: int,
        full_attention_layer_ids: List[int],
        post_capture_active: bool = False,
        **kwargs,
    ):
        layout = QSAHiSparseLayout(
            block_tokens=qsa_compress_ratio,
            page_size=page_size,
            hot_blocks=hot_blocks,
            num_slabs=num_request_slots - 1,
            prefill_tokens=prefill_tokens,
            logical_tokens=logical_size,
        )
        if post_capture_active:
            raise NotImplementedError(
                "QSA HiSparse fixes its device layout at startup; the post-capture "
                "KV resize cannot apply"
            )
        super().__init__(
            size=layout.device_tokens,
            index_size=layout.logical_tokens,
            page_size=page_size,
            qsa_compress_ratio=qsa_compress_ratio,
            num_request_slots=num_request_slots,
            full_attention_layer_ids=full_attention_layer_ids,
            post_capture_active=False,
            **kwargs,
        )
        self.layout = layout
        # Coordinator-facing: every per-layer table is indexed 0..layer_num-1.
        self.layer_num = len(full_attention_layer_ids)
        self.plane_spec = self._check_full_kv_pool()
        # logical block -> device block; 0 = unmapped (points at the dump page),
        # the trailing -1 is what a last_loc of -1 translates to.
        self.mapping = torch.zeros(
            layout.logical_blocks + 1, dtype=torch.int64, device=self.device
        )
        self.mapping[-1] = -1
        per_layer = [self.device_planes(i) for i in range(self.layer_num)]
        # Per K/V pair: uint64 [layers] device base pointers for transfer_hicache_all_layer.
        self.pair_ptrs = [
            tuple(
                make_kernel_ptr_table(
                    [planes[p] for planes in per_layer],
                    self.device,
                    host_memory_registered=False,
                )
                for p in (q, q + 1)
            )
            for q, _ in self.plane_spec.pairs()
        ]
        logger.info(
            "QSA HiSparse device pool: B=%d blocks, %d slabs, prefill area %d tokens, "
            "%d device rows for %d logical tokens, record %d B",
            layout.hot_blocks,
            layout.num_slabs,
            layout.prefill_tokens,
            layout.device_tokens,
            layout.logical_tokens,
            self.plane_spec.record_bytes,
        )

    def _check_full_kv_pool(self) -> PlaneSpec:
        fkv = self.full_kv_pool
        if type(fkv) is not MHATokenToKVPool:
            raise NotImplementedError(
                f"QSA HiSparse needs a plain MHATokenToKVPool, got {type(fkv).__name__}"
            )
        if fkv.is_quantized_kv_cache:
            raise NotImplementedError(
                "QSA HiSparse planes for quantized (NVFP4) KV are M3; only FP8/BF16 KV is wired"
            )
        if fkv.use_hnd or fkv.kv_cache_layout != "nhd":
            raise ValueError(
                f"QSA HiSparse needs a token-major NHD KV cache, got layout {fkv.kv_cache_layout!r}"
            )
        rows = self.layout.device_tokens + self.page_size
        for name, bufs in (("k", fkv.k_buffer), ("v", fkv.v_buffer)):
            for i, buf in enumerate(bufs):
                if not buf.is_contiguous() or buf.shape[0] != rows:
                    raise ValueError(
                        f"{name}_buffer[{i}] must be contiguous with {rows} rows, got "
                        f"{tuple(buf.shape)} contiguous={buf.is_contiguous()}"
                    )
        item = fkv.store_dtype.itemsize
        return kv_plane_spec(
            k_row_bytes=fkv.head_num * fkv.head_dim * item,
            v_row_bytes=fkv.head_num * fkv.v_head_dim * item,
            block_tokens=self.layout.block_tokens,
        )

    # ------------------------------------------------------------ addressing

    def full_layer_index(self, layer_id: int) -> int:
        """Model layer id -> 0..layer_num-1, the index of every coordinator table."""
        return self._transfer_full_attention_id(layer_id)

    def device_planes(self, layer_index: int) -> tuple[torch.Tensor, ...]:
        """Device planes of one layer in record order (K, V)."""
        fkv = self.full_kv_pool
        return fkv.k_buffer[layer_index], fkv.v_buffer[layer_index]

    def rows_of(self, loc: torch.Tensor) -> torch.Tensor:
        """Logical token slots -> device KV rows through ``mapping`` (int64)."""
        bt = self.layout.block_tokens
        return self.mapping[loc // bt] * bt + loc % bt

    def set_kv_buffer(
        self,
        layer,
        loc,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        dcp_kv_mask=None,
    ):
        loc, swa_loc, full_loc = unwrap_write_loc(loc)
        if swa_loc is not None or full_loc is not None:
            raise NotImplementedError("QSA HiSparse writes plain logical locations only")
        super().set_kv_buffer(
            layer,
            self.rows_of(loc),
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            dcp_kv_mask=dcp_kv_mask,
        )

    # ------------------------------------------------------------ block copies

    def copy_blocks(self, src_blocks: torch.Tensor, dst_blocks: torch.Tensor) -> None:
        """Device -> device copy of whole blocks, every layer and plane, on the current stream."""
        _check_transfer_indices(src_blocks, dst_blocks)
        if src_blocks.numel() == 0:
            return  # a zero-length launch is an invalid grid
        for (_, nbytes), (k_ptrs, v_ptrs) in zip(
            self.plane_spec.pairs(), self.pair_ptrs
        ):
            transfer_hicache_all_layer(
                k_ptrs,
                v_ptrs,
                dst_blocks,
                k_ptrs,
                v_ptrs,
                src_blocks,
                kv_cache_src_stride_bytes=nbytes,
                kv_cache_dst_stride_bytes=nbytes,
                element_size=nbytes,
                block_quota=TRANSFER_BLOCK_QUOTA,
            )

    # ------------------------------------------------------------ unsupported

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("QSA HiSparse pool has no CPU copy path")

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None, req_pool_index=None):
        raise NotImplementedError("QSA HiSparse pool has no CPU copy path")

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError("QSA HiSparse pool does not support KV moves (spec)")

    def finalize_backing(self, config) -> None:
        raise NotImplementedError("QSA HiSparse fixes its device layout at startup")

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("QSA HiSparse does not support PD transfer")

    def get_state_buf_infos(self):
        raise NotImplementedError("QSA HiSparse does not support PD transfer")

    def get_qsa_pending_state_buf_infos(self):
        raise NotImplementedError("QSA HiSparse does not support PD transfer")

    def get_qsa_compressed_state_buf_infos(self):
        raise NotImplementedError("QSA HiSparse does not support PD transfer")


class QSAContextLandingZone:
    """Persistent device buffer the host part of a resumed prefix lands in, one
    layer at a time, before the prefill context is assembled (arc-forge M1b h07,
    design §4.3). Same planes as the device pool (FP8 / BF16: token-major K and
    V); its one-entry pointer tables are built once, so a fill is one
    transfer_hicache_all_layer launch per K/V pair. NVFP4 (M3) records need a
    raw scratch plus expansion instead and are not wired here.
    """

    def __init__(self, pool: HiSparseQSATokenToKVPool, capacity_tokens: int):
        layout = pool.layout
        if capacity_tokens <= 0 or capacity_tokens % layout.page_size:
            raise ValueError(
                f"landing zone of {capacity_tokens} tokens is not a positive multiple "
                f"of page_size {layout.page_size}"
            )
        if len(pool.plane_spec.planes) != 2:
            raise NotImplementedError(
                "QSA HiSparse host-prefix fill is FP8/BF16 only; NVFP4 records are M3"
            )
        self.plane_spec = pool.plane_spec
        self.block_tokens = layout.block_tokens
        self.capacity_blocks = capacity_tokens // layout.block_tokens
        # Rows shaped like the pool's (tokens, heads, dim) in its store dtype.
        self.planes = tuple(
            torch.empty(
                (capacity_tokens,) + tuple(p.shape[1:]), dtype=p.dtype, device=p.device
            )
            for p in pool.device_planes(0)
        )
        self.pair_ptrs = [
            tuple(
                make_kernel_ptr_table(
                    [self.planes[p]], pool.device, host_memory_registered=False
                )
                for p in (q, q + 1)
            )
            for q, _ in self.plane_spec.pairs()
        ]
        self.block_ids = torch.arange(
            self.capacity_blocks, dtype=torch.int64, device=pool.device
        )
        self.nbytes = sum(p.numel() * p.element_size() for p in self.planes)


class QSAHiSparseTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Logical token allocator plus the device prefill-area block allocator (design §5.3).

    Admission sees min(logical, prefill area): a prompt must fit both. Decode
    needs logical slots only (its device rows are the static slab), so
    check_decode_capacity looks at the logical side alone.
    """

    def __init__(
        self,
        logical_size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: HiSparseQSATokenToKVPool,
        need_sort: bool,
    ):
        layout = kvcache.layout
        if logical_size != layout.logical_tokens or page_size != layout.page_size:
            raise ValueError(
                f"allocator (size={logical_size}, page={page_size}) does not match the pool layout "
                f"(logical={layout.logical_tokens}, page={layout.page_size})"
            )
        self._kvcache = kvcache
        self.layout = layout
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.need_sort = need_sort
        self.logical_attn_allocator = PagedTokenToKVPoolAllocator(
            logical_size, page_size, dtype, device, kvcache, need_sort
        )
        # Block ids, page = one KV page of blocks; its pages 1..N are exactly the
        # prefill area [page_blocks, hot_base) because page 0 is the dump page.
        self.prefill_block_allocator = PagedTokenToKVPoolAllocator(
            layout.prefill_blocks, layout.page_blocks, dtype, device, kvcache, need_sort
        )
        self.free_pages = None
        self.release_pages = None
        self.free_group = None
        self.clear()

    @property
    def size(self) -> int:
        return self.layout.logical_tokens

    @property
    def size_full(self) -> int:
        return self.layout.logical_tokens

    def available_size(self) -> int:
        return min(
            self.logical_attn_allocator.available_size(),
            self.prefill_block_allocator.available_size() * self.layout.block_tokens,
        )

    def check_decode_capacity(
        self, *, num_tokens: int, tree_cache, requests=None, spec_algorithm=None
    ) -> bool:
        self.evict_to_free_tokens(tree_cache, num_tokens)
        return self.logical_attn_allocator.available_size() >= num_tokens

    def debug_print(self) -> str:
        return (
            f"#logical-available-size: {self.logical_attn_allocator.available_size()}, "
            f"#prefill-area-available-blocks: {self.prefill_block_allocator.available_size()}, "
        )

    def alloc(self, need_size: int):
        raise NotImplementedError("QSA HiSparse allocates through alloc_extend / alloc_decode")

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        bt = self.layout.block_tokens
        if bool((prefix_lens_cpu % bt != 0).any()):
            raise RuntimeError(
                f"QSA HiSparse needs block-aligned prefixes, got {prefix_lens_cpu.tolist()}"
            )
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        blk_prefix_cpu = prefix_lens_cpu // bt
        blk_seq_cpu = (seq_lens_cpu + bt - 1) // bt
        # A block-aligned prefix makes device pages (page_blocks blocks) line up
        # with logical pages one to one.
        if (
            get_num_new_pages(
                seq_lens=blk_seq_cpu,
                page_size=self.layout.page_blocks,
                prefix_lens=blk_prefix_cpu,
            )
            != num_new_pages
        ):
            raise RuntimeError("device and logical page counts diverged")
        if num_new_pages > self.logical_attn_allocator.available_size() // self.page_size:
            return None
        if (
            num_new_pages
            > self.prefill_block_allocator.available_size() // self.layout.page_blocks
        ):
            return None

        mapping = self._kvcache.mapping
        # before any allocation: a rejected host prefix leaves both pools untouched
        device_last = self._device_last_block(
            mapping[last_loc // bt], prefix_lens, prefix_lens_cpu
        )
        logical = self.logical_attn_allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
        )
        if logical is None:
            raise RuntimeError("logical allocation failed after the capacity check")
        num_blocks = int((blk_seq_cpu - blk_prefix_cpu).sum())
        device_blocks = self.prefill_block_allocator.alloc_extend(
            prefix_lens // bt,
            blk_prefix_cpu,
            (seq_lens + bt - 1) // bt,
            blk_seq_cpu,
            device_last,
            num_blocks,
            num_new_pages=num_new_pages,
        )
        if device_blocks is None:
            raise RuntimeError("prefill-area allocation failed after the capacity check")
        # The in-page offset of a logical slot equals its position mod page, so
        # a slot opens a block iff slot % bt == 0; batch order matches.
        starts = logical[logical % bt == 0]
        if starts.numel() != num_blocks:
            raise RuntimeError(
                f"{starts.numel()} logical block starts for {num_blocks} device blocks"
            )
        mapping[starts // bt] = device_blocks
        return logical

    def _device_last_block(
        self,
        mapped_last: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Device block of each request's last prefix block, as the paged block
        allocator takes last_loc (arc-forge M1b h07, design §4.4).

        A prefix held on host (a resumed session) has no device rows: its last
        block maps to 0, the dump block no request is ever given. Its device
        side starts a fresh page, so pass -1 ("no partial page to continue",
        and the allocator's debug check (last + 1) % page == prefix % page
        holds). That is only right for a page-aligned prefix. Resume points are
        page-aligned by construction; a CPU-side unaligned prefix must be a
        device-resident chunk continuation, checked with a sync in that case.
        """
        on_host = mapped_last == 0
        if bool((prefix_lens_cpu % self.page_size != 0).any()) and bool(
            (on_host & (prefix_lens % self.page_size != 0)).any()
        ):
            raise RuntimeError(
                "QSA HiSparse: a prefix with no device rows (on host) is not page "
                f"aligned; prefix lengths {prefix_lens_cpu.tolist()}, page {self.page_size}"
            )
        return torch.where(on_host, torch.full_like(mapped_last, -1), mapped_last)

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        # Device rows of decode tokens are slab slots, set by the coordinator.
        return self.logical_attn_allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    def free_prefill_blocks(self, device_blocks: torch.Tensor) -> None:
        """Return prefill-area blocks (whole pages) to the pool."""
        self._check_prefill_blocks(device_blocks)
        self.prefill_block_allocator.free(device_blocks)

    def _check_prefill_blocks(self, device_blocks: torch.Tensor) -> None:
        layout = self.layout
        if bool(
            (
                (device_blocks < layout.prefill_base) | (device_blocks >= layout.hot_base)
            ).any()
        ):
            raise RuntimeError(
                "freeing device blocks outside the prefill area "
                f"[{layout.prefill_base}, {layout.hot_base}); a decoding request must "
                "go through the coordinator's request_finished before its KV is freed"
            )

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        # A request that never reached decode (finished or aborted during
        # prefill) still maps its blocks into the prefill area.
        mapping = self._kvcache.mapping
        blocks = torch.unique(free_index // self.layout.block_tokens)
        held = mapping[blocks]
        held = held[held != 0]
        if held.numel() > 0:
            self._check_prefill_blocks(held)  # before any state changes
        self.logical_attn_allocator.free(free_index)
        if held.numel() > 0:
            self.prefill_block_allocator.free(held)
        mapping[blocks] = 0

    def clear(self):
        self.logical_attn_allocator.clear()
        self.prefill_block_allocator.clear()
        self._kvcache.mapping[:-1].zero_()
        self.free_group = None

    # Frees are never deferred: free() must clear the mapping at once.
    def free_group_begin(self):
        return

    def free_group_end(self):
        return

    def resize(self, config) -> None:
        raise NotImplementedError("QSA HiSparse fixes its device layout at startup")
