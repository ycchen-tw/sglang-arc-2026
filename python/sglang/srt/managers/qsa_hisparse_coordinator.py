"""HiSparse coordinator for Qwen QSA block sparsity (arc-forge M1a h03, design §2).

A HiSparse item is one QSA block (``block_tokens`` = 4 positions); every
"token" of the upstream coordinator and swap-in kernel is a block here. The
kernel receives seq_blocks = ceil(seq_len / 4), so its newest item is the block
of the current token; top-k only ever selects complete blocks.

Per request slot r the device keeps a static slab of H = B + page_blocks blocks
(qsa_hisparse_layout.py). Lifecycle, all on block ids:

    prefill      logical slots + prefill-area blocks (allocator.alloc_extend)
    staging      D2H complete prompt blocks, D2D prompt blocks into the slab
    ready        prefill-area blocks back to the pool, LRU tables initialised
    decode       current block -> slab slot min(nb-1, B); the block completed by
                 the previous step goes to host before this step's forward
    finish       host blocks and tables released; the slab stays with the slot
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, NamedTuple, Tuple

import torch

from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_planar
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseAct,
    HiSparseCoordinator,
    HiSparseTokenStats,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host.qsa_planar import QSAPlanarHostPool
from sglang.srt.mem_cache.qsa_hisparse_pool import (
    HiSparseQSATokenToKVPool,
    QSAHiSparseTokenToKVPoolAllocator,
)
from sglang.srt.utils import get_device_module

device_module = get_device_module()

logger = logging.getLogger(__name__)

# Test flag (design §6): wait for staging right after admission so a request
# joins decode on the next round, as the full-resident baseline does.
_SYNC_STAGING_ENV = "ARC_HISPARSE_QSA_SYNC_STAGING"

# State the inherited HiSparseCoordinator methods read (design §2: the subclass
# skips super().__init__ and must set every one of them itself).
INHERITED_STATE = (
    "req_to_token_pool",
    "token_to_kv_pool_allocator",
    "mem_pool_device",
    "mem_pool_host",
    "top_k",
    "device_buffer_size",
    "padded_buffer_size",
    "device",
    "swap_in_block_size",
    "skip_io",
    "write_staging_stream",
    "decode_backup_stream",
    "ack_staging_queue",
    "decode_producer_stream",
    "_backup_done_event",
    "_has_pending_backup",
    "tp_group",
    "tp_world_size",
    "req_to_host_pool",
    "req_to_host_pool_allocated_len",
    "req_device_buffer_tokens",
    "req_device_buffer_token_locs",
    "lru_slots",
    "top_k_device_locs_buffer",
    "num_real_reqs",
    "_skip_first_backup",
    "enable_prefetch",
    "_is_shared_index_layer",
)


class QSAHiSparseDecodeMeta(NamedTuple):
    """Per-forward views of the coordinator's persistent decode buffers ([:bs])."""

    seq_blocks: torch.Tensor  # int32, ceil(seq_len / block_tokens): the kernel's seq_lens
    newest_dev: torch.Tensor  # int32, device block of slab slot min(seq_blocks - 1, B)
    rows: torch.Tensor  # int32 [bs, top_k * block_tokens + block_tokens - 1], hot rows (h04 writes)


class QSAHiSparseCoordinator(HiSparseCoordinator):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: QSAHiSparseTokenToKVPoolAllocator,
        top_k: int,
        device: str,
        tp_group,
        swap_in_block_size: int = 960,
        max_decode_rows: int = 0,
    ):
        # No super().__init__(): upstream's asserts an MLA pool and builds an MLA host pool.
        if not isinstance(token_to_kv_pool_allocator, QSAHiSparseTokenToKVPoolAllocator):
            raise TypeError(
                f"need QSAHiSparseTokenToKVPoolAllocator, got {type(token_to_kv_pool_allocator).__name__}"
            )
        pool = token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(pool, HiSparseQSATokenToKVPool):
            raise TypeError(f"need HiSparseQSATokenToKVPool, got {type(pool).__name__}")
        layout = pool.layout
        if top_k != pool.qsa_block_topk:
            raise ValueError(
                f"top_k {top_k} must be the QSA block top-k {pool.qsa_block_topk}"
            )
        if layout.hot_blocks < top_k:
            raise ValueError(f"hot_blocks {layout.hot_blocks} is below top_k {top_k}")
        num_slots = req_to_token_pool.req_to_token.shape[0]
        if num_slots != layout.num_slabs + 1:
            raise ValueError(
                f"{num_slots} request slots but the pool has {layout.num_slabs} slabs (+ slot 0)"
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = pool
        self.layout = layout
        self.top_k = top_k
        self.compress_ratio = layout.block_tokens
        self.device_buffer_size = layout.hot_blocks
        self.padded_buffer_size = layout.slab_stride
        self.page_size = layout.page_blocks
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device bytes (garbage output; benchmarking only).
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        self._sync_staging = os.environ.get(_SYNC_STAGING_ENV, "0") == "1"

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size != 1:
            raise NotImplementedError("QSA HiSparse is TP1 only (M1a)")

        max_context_len = req_to_token_pool.max_context_len
        max_blocks = -(-max_context_len // layout.block_tokens)
        self.mem_pool_host = QSAPlanarHostPool(
            device_pool=pool, num_blocks=layout.host_blocks(max_context_len)
        )

        layer_num = pool.layer_num
        hot, stride = layout.hot_blocks, layout.slab_stride
        # Static slab: slot i of request r is device block hot_base + (r-1)*H + i.
        # Row 0 (never allocated; CUDA-graph padding) points at the dump block.
        slab = torch.zeros((num_slots, stride), dtype=torch.int64)
        slab[1:] = layout.hot_base + (
            torch.arange(num_slots - 1, dtype=torch.int64)[:, None] * stride
            + torch.arange(stride, dtype=torch.int64)[None, :]
        )
        self.req_to_device_buffer = slab.to(device)
        self.req_device_buffer_token_locs = (
            self.req_to_device_buffer.to(torch.int32)[None]
            .repeat(layer_num, 1, 1)
            .contiguous()
        )
        self.req_device_buffer_tokens = torch.full(
            (layer_num, num_slots, stride), -1, dtype=torch.int32, device=device
        )
        self._lru_init = torch.arange(hot, dtype=torch.int16, device=device)
        self.lru_slots = (
            self._lru_init.view(1, 1, -1).repeat(layer_num, num_slots, 1).contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            hot, dtype=torch.int32, device=device
        )
        self.req_to_host_pool = torch.full(
            (num_slots, max_blocks + layout.page_blocks),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            num_slots, dtype=torch.int64, device="cpu"
        )
        # Decode rows: real requests plus CUDA-graph padding (max_decode_rows).
        max_rows = max(num_slots, max_decode_rows)
        self.top_k_device_locs_buffer = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        self._skip_first_backup = [False] * num_slots

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        # req_pool_idx -> (prefill-area blocks, logical blocks) held while staging
        self._staging: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        # No shared-index prefetch: QSA computes top-k in every layer.
        self.enable_prefetch = False
        self._is_shared_index_layer = [False] * layer_num

        # Persistent decode buffers for the QSA backend (design §3.6, §3.8, §4).
        bt = layout.block_tokens
        self.seq_blocks = torch.ones(max_rows, dtype=torch.int32, device=device)
        self.newest_dev = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.rows = torch.full(
            (max_rows, top_k * bt + bt - 1), -1, dtype=torch.int32, device=device
        )
        device_rows = layout.device_tokens + layout.page_size
        self.identity_rows = torch.arange(
            device_rows, dtype=torch.int32, device=device
        ).view(1, -1)
        self.zeros_i32 = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.rows_limit = torch.full(
            (max_rows,), device_rows, dtype=torch.int32, device=device
        )

        self._warmup_kernels()
        logger.info(
            "QSA HiSparse coordinator: top_k=%d blocks, B=%d, %d request slots, %d layers, "
            "host %d blocks",
            top_k,
            hot,
            num_slots - 1,
            layer_num,
            self.mem_pool_host.size,
        )

    # ------------------------------------------------------------ helpers

    def layer_index(self, layer_id: int) -> int:
        """Model layer id -> index of the per-layer tables."""
        return self.mem_pool_device.full_layer_index(layer_id)

    def _logical_blocks(self, req_pool_idx: int, num_tokens: int) -> torch.Tensor:
        """Logical block ids of positions [0, num_tokens) of one request."""
        bt = self.layout.block_tokens
        locs = self.req_to_token_pool.req_to_token[req_pool_idx, 0:num_tokens:bt]
        return locs.to(torch.int64) // bt

    def _free_host(self, req_pool_idx: int) -> None:
        host = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req_pool_idx,
            self.req_to_host_pool_allocated_len[req_pool_idx],
        )
        if host.numel() > 0:
            self.mem_pool_host.free(host)
        self.req_to_host_pool[req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req_pool_idx] = 0

    def _warmup_kernels(self) -> None:
        """Load the JIT modules now rather than inside a CUDA-graph capture (design §4)."""
        dump = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.mem_pool_device.copy_blocks(dump, dump)  # dump block onto itself
        # num_real_reqs = 0: every row takes the padded early return.
        self.num_real_reqs.zero_()
        self._run_swap_in_kernel(
            torch.zeros(1, dtype=torch.int64, device=self.device),
            torch.ones(1, dtype=torch.int32, device=self.device),
            torch.full((1, self.top_k), -1, dtype=torch.int32, device=self.device),
            0,
        )
        device_module.synchronize()

    # ------------------------------------------------------------ staging

    def admit_request_into_staging(self, req: Req) -> None:
        r = req.kv.req_pool_idx
        prompt_len = req.extend_range.end
        if prompt_len != req.kv.kv_allocated_len:
            raise RuntimeError(
                f"staging {req.rid}: prompt {prompt_len} != allocated {req.kv.kv_allocated_len}"
            )
        if r in self._staging:
            raise RuntimeError(f"request slot {r} is already staging")
        layout = self.layout
        plan = layout.staging_plan(prompt_len)
        logical_blocks = self._logical_blocks(r, prompt_len)
        prefill_blocks = self.mem_pool_device.mapping[logical_blocks]
        if bool(
            (
                (prefill_blocks < layout.prefill_base)
                | (prefill_blocks >= layout.hot_base)
            ).any()
        ):
            raise RuntimeError(f"staging {req.rid}: a prompt block is not in the prefill area")
        host_blocks = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            r,
            0,
            plan.full_blocks,
        )
        slab = self.req_to_device_buffer[r]
        src = [prefill_blocks[: plan.identity_blocks]]
        dst = [slab[: plan.identity_blocks]]
        if plan.newest_block is not None:
            src.append(prefill_blocks[plan.newest_block : plan.newest_block + 1])
            dst.append(slab[layout.hot_blocks : layout.hot_blocks + 1])
        d2d_src, d2d_dst = torch.cat(src), torch.cat(dst)

        req.hisparse_staging = True
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_blocks(
                prefill_blocks[: plan.full_blocks], host_blocks
            )
            self.mem_pool_device.copy_blocks(d2d_src, d2d_dst)
            finish_event.record()
            for t in (prefill_blocks, host_blocks, d2d_src, d2d_dst):
                t.record_stream(self.write_staging_stream)
        self._staging[r] = (prefill_blocks, logical_blocks)
        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))
        if self._sync_staging:
            self.write_staging_stream.synchronize()

    def alloc_device_buffer(self, req: Req) -> None:
        """Staging finished: hand the prefill blocks back and open the slab for decode."""
        r = req.kv.req_pool_idx
        prefill_blocks, logical_blocks = self._staging.pop(r)
        hot = self.layout.hot_blocks
        self.req_device_buffer_tokens[:, r, :hot] = self._device_buffer_arange_i32
        self.req_device_buffer_tokens[:, r, hot:] = -1
        self.lru_slots[:, r, :].copy_(self._lru_init)
        self.token_to_kv_pool_allocator.free_prefill_blocks(prefill_blocks)
        self.mem_pool_device.mapping[logical_blocks] = 0

    def abort_staging_request(self, req: Req) -> None:
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        self.write_staging_stream.synchronize()
        r = req.kv.req_pool_idx
        prefill_blocks, logical_blocks = self._staging.pop(r)
        self.token_to_kv_pool_allocator.free_prefill_blocks(prefill_blocks)
        self.mem_pool_device.mapping[logical_blocks] = 0
        self._free_host(r)
        self._skip_first_backup[r] = False
        req.hisparse_staging = False

    # ------------------------------------------------------------ decode (scheduler side)

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        if out_cache_loc.shape != seq_lens.shape:
            raise RuntimeError("QSA HiSparse decodes one token per request")
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )
        bt = self.layout.block_tokens
        slot = ((seq_lens + bt - 1) // bt - 1).clamp(max=self.layout.hot_blocks)
        self.mem_pool_device.mapping[out_cache_loc // bt] = self.req_to_device_buffer[
            req_pool_indices, slot
        ]

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Copy the block the previous step completed to host (design §2 decode row)."""
        device_blocks: List[int] = []
        host_blocks = []
        for i in range(len(seq_lens_cpu)):
            r = int(req_pool_indices_cpu[i])
            skip = self._skip_first_backup[r]
            self._skip_first_backup[r] = False
            backup = self.layout.decode_backup(int(seq_lens_cpu[i]), skip)
            if backup is None:
                continue
            block, slot = backup
            device_blocks.append(self.layout.slab_block(r, slot))
            host_blocks.append(
                self.mem_pool_host.alloc_paged_token_slots(
                    self.req_to_host_pool,
                    self.req_to_host_pool_allocated_len,
                    r,
                    block,
                    1,
                )
            )
        if not device_blocks:
            return
        device_locs = torch.tensor(device_blocks, dtype=torch.int64, device=self.device)
        host_locs = torch.cat(host_blocks)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_blocks(device_locs, host_locs)
            self._backup_done_event.record()
            device_locs.record_stream(self.decode_backup_stream)
            host_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    # ------------------------------------------------------------ decode (forward side)

    def prepare_decode_meta(
        self, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor, bs: int
    ) -> QSAHiSparseDecodeMeta:
        """Fill the persistent per-row buffers for one decode forward.

        No host sync, so it can run inside a CUDA-graph capture; seq_lens and
        req_pool_indices may be int32 or int64. The backend reads the buffers,
        not the returned views.
        """
        if bs > self.seq_blocks.numel():
            raise ValueError(f"batch {bs} exceeds {self.seq_blocks.numel()} decode rows")
        bt = self.layout.block_tokens
        seq_blocks = self.seq_blocks[:bs]
        seq_blocks.copy_((seq_lens[:bs] + bt - 1) // bt)
        slot = (seq_blocks.to(torch.int64) - 1).clamp(max=self.layout.hot_blocks)
        newest_dev = self.newest_dev[:bs]
        newest_dev.copy_(
            self.req_to_device_buffer[req_pool_indices[:bs].to(torch.int64), slot]
        )
        return QSAHiSparseDecodeMeta(seq_blocks, newest_dev, self.rows[:bs])

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
    ) -> torch.Tensor:
        """Resolve one layer's top-k blocks to slab blocks, loading misses from host."""
        if record_plan:
            raise NotImplementedError("QSA HiSparse has no shared-index prefetch")
        top_k_device_locs = self.top_k_device_locs_buffer[: req_pool_indices.size(0)]
        load_cache_to_device_buffer_planar(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_records=self.mem_pool_host.kv_buffer[layer_id],
            device_planes=self.mem_pool_device.device_planes(layer_id),
            plane_bytes=self.mem_pool_device.plane_spec.plane_bytes,
            top_k_device_locs=top_k_device_locs,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
        )
        return top_k_device_locs

    # ------------------------------------------------------------ release

    def request_finished(self, req: Req):
        r = req.kv.req_pool_idx
        if req.hisparse_staging or r in self._staging:
            raise RuntimeError(
                f"{req.rid} is staging; release it with abort_staging_request"
            )
        # Release only after a potentially overlapped batch has run.
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        layout = self.layout
        logical_blocks = self._logical_blocks(r, req.kv.kv_allocated_len)
        mapping = self.mem_pool_device.mapping
        held = mapping[logical_blocks]
        in_prefill = (held >= layout.prefill_base) & (held < layout.hot_base)
        slab = self.req_to_device_buffer[r]
        in_slab = (held >= slab[0]) & (held <= slab[-1])
        if bool((~(in_prefill | in_slab | (held == 0))).any()):
            raise RuntimeError(
                f"{req.rid}: mapping points outside its prefill blocks and slab"
            )
        # A request retracted mid-prefill still holds prefill-area blocks.
        prefill_held = held[in_prefill]
        if prefill_held.numel() > 0:
            self.token_to_kv_pool_allocator.free_prefill_blocks(prefill_held)
        mapping[logical_blocks] = 0
        self._free_host(r)
        self.req_device_buffer_tokens[:, r, :] = -1
        self.lru_slots[:, r, :].copy_(self._lru_init)
        self._skip_first_backup[r] = False

    # ------------------------------------------------------------ diagnostics

    def get_token_stats(self) -> HiSparseTokenStats:
        bt = self.layout.block_tokens
        prefill = self.token_to_kv_pool_allocator.prefill_block_allocator
        device_capacity = prefill.size * bt
        device_tokens = device_capacity - prefill.available_size() * bt
        host = self.mem_pool_host
        host_tokens = (host.size - host.available_size()) * bt
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=device_tokens / device_capacity,
            host_tokens=host_tokens,
            host_token_usage=host_tokens / (host.size * bt),
        )

    def check_idle_invariants(self) -> None:
        """With nothing running or staging, every pool and table is back at its start (R3)."""
        problems = []
        if self.ack_staging_queue or self._staging:
            problems.append(
                f"staging queue {len(self.ack_staging_queue)}, staging slots {sorted(self._staging)}"
            )
        alloc = self.token_to_kv_pool_allocator
        for name, sub, full in (
            ("logical", alloc.logical_attn_allocator, self.layout.logical_tokens),
            ("prefill area", alloc.prefill_block_allocator, self.layout.prefill_blocks),
        ):
            if sub.available_size() != full:
                problems.append(f"{name} allocator {sub.available_size()} / {full} free")
        host = self.mem_pool_host
        if host.available_size() != host.size or bool(host.slot_used.any()):
            problems.append(
                f"host {host.available_size()} / {host.size} free, "
                f"{int(host.slot_used.sum())} slots marked used"
            )
        mapping = self.mem_pool_device.mapping
        if bool(mapping[:-1].any()) or int(mapping[-1]) != -1:
            problems.append(f"mapping has {int((mapping[:-1] != 0).sum())} live entries")
        if bool((self.req_to_host_pool != -1).any()) or bool(
            self.req_to_host_pool_allocated_len.any()
        ):
            problems.append("req_to_host_pool not cleared")
        if bool((self.req_device_buffer_tokens != -1).any()):
            problems.append("req_device_buffer_tokens not cleared")
        if not torch.equal(
            self.lru_slots, self._lru_init.view(1, 1, -1).expand_as(self.lru_slots)
        ):
            problems.append("lru_slots not back to the initial order")
        if not torch.equal(
            self.req_device_buffer_token_locs,
            self.req_to_device_buffer.to(torch.int32)[None].expand_as(
                self.req_device_buffer_token_locs
            ),
        ):
            problems.append("req_device_buffer_token_locs no longer the static slab")
        if any(self._skip_first_backup):
            problems.append("skip-first-backup flags left set")
        if self._has_pending_backup:
            problems.append("a decode backup is still pending")
        if problems:
            raise RuntimeError("QSA HiSparse idle invariants violated: " + "; ".join(problems))

    # ------------------------------------------------------------ unsupported

    def admit_request_direct(self, req: Req) -> None:
        raise NotImplementedError("QSA HiSparse has no PD direct-to-host path")

    def _preload_to_device_buffer(self, req: Req) -> None:
        raise NotImplementedError("QSA HiSparse has no PD direct-to-host path")

    def naive_load_topk(self, req_pool_indices, seq_lens, top_k_tokens, layer_id):
        raise NotImplementedError("QSA HiSparse is checked against arc-forge's rules oracle")

    def _grow_device_buffers(self, *args, **kwargs):
        raise NotImplementedError("QSA HiSparse slabs are static")


def create_qsa_hisparse_coordinator(model_runner) -> QSAHiSparseCoordinator:
    """model_runner hook (h03): build the coordinator for a QSA model under --enable-hisparse."""
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import get_exec, get_memory

    pool = model_runner.token_to_kv_pool
    if not isinstance(pool, HiSparseQSATokenToKVPool):
        raise RuntimeError(
            "--enable-hisparse on a QSA model needs the QSA HiSparse KV pool "
            f"(experimental-hisparse h05 wiring), got {type(pool).__name__}"
        )
    profile = parse_qsa_profile(model_runner.model_config.hf_config)
    if profile is None:
        raise RuntimeError("QSA HiSparse pool on a model without a QSA profile")
    top_k = profile.block_topk
    raw = get_memory().hisparse_config
    user = json.loads(raw) if raw is not None else {}
    for key in ("device_buffer_size", "host_to_device_ratio"):
        if key in user:
            raise ValueError(
                f"hisparse_config.{key} does not apply to QSA: B comes from the pool "
                "(qsa_hot_blocks) and the host pool is sized for the worst case"
            )
    if user.get("top_k", top_k) != top_k:
        raise ValueError(
            f"hisparse_config.top_k={user['top_k']} but the QSA indexer selects "
            f"{top_k} blocks (indexer_budget / indexer_compress_ratio)"
        )
    decode_graph = get_exec().graph.cuda_graph_config.decode
    num_reqs = pool.layout.num_slabs
    if decode_graph.backend != Backend.DISABLED and (decode_graph.max_bs or 0) < num_reqs:
        # design §4: every decode batch replays a graph; larger ones would run eager
        raise ValueError(
            "QSA HiSparse with decode CUDA graphs needs --cuda-graph-max-bs-decode >= "
            f"--max-running-requests ({num_reqs}), got {decode_graph.max_bs}"
        )
    return QSAHiSparseCoordinator(
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        top_k=top_k,
        device=model_runner.device,
        tp_group=model_runner.tp_group.cpu_group,
        swap_in_block_size=parse_hisparse_config().swap_in_block_size,
        # one token per request (no spec), so graph rows = its max batch size
        max_decode_rows=int(decode_graph.max_bs or 0),
    )
