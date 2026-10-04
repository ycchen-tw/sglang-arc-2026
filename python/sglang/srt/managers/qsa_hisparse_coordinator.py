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

Session resume (arc-forge M1b h07, design §2.3, §4.3-4.4, §11.5), on with
session_resume: a request may start from a parked prefix, logical slots [0, P)
plus host blocks [0, P/4) owned by a session record. adopt_host_prefix puts the
record's host blocks in the request's row; the prefill context reads them
through the landing zone (fill_host_prefix_context); staging copies only
[P/4, nb) from the prefill area and loads the prefix's identity slots from
host. Host blocks are then released by release_host_row(r, keep) from the
cache's finish path, never by request_finished / abort_staging_request, which
run before it; without session_resume those two release the whole row as M1a.

With qsa_snapshot_interval set (M1b h08), decode also snapshots the GDN / PLE
state every T positions: D2D live -> staging mamba slot in the backup window,
async D2H to a pinned host buffer, published to an owner (h09) once the D2H is
seen done (QSAHiSparseSnapshots below).
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple

import torch

from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_planar
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseAct,
    HiSparseCoordinator,
    HiSparseTokenStats,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    HostTensorAllocator,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.qsa_planar import QSAPlanarHostPool
from sglang.srt.mem_cache.qsa_hisparse_pool import (
    HiSparseQSATokenToKVPool,
    QSAContextLandingZone,
    QSAHiSparseTokenToKVPoolAllocator,
)
from sglang.srt.utils import get_device_module

device_module = get_device_module()

logger = logging.getLogger(__name__)

# Test flag (design §6): wait for staging right after admission so a request
# joins decode on the next round, as the full-resident baseline does.
_SYNC_STAGING_ENV = "ARC_HISPARSE_QSA_SYNC_STAGING"
# Debug flag (shared with the indexer's selection check): before each eager
# swap-in, check that every block the kernel may copy from host has a record.
_DEBUG_ENV = "ARC_HISPARSE_QSA_DEBUG"

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
        session_resume: bool = False,
        snapshot: Optional["QSAHiSparseSnapshotConfig"] = None,
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
        debug = os.environ.get(_DEBUG_ENV, "0")
        if debug not in ("0", "1"):
            raise ValueError(f"{_DEBUG_ENV} must be 0 or 1, got {debug!r}")
        self._debug_host_records = debug == "1"

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        if self.tp_world_size != 1:
            raise NotImplementedError("QSA HiSparse is TP1 only (M1a)")

        max_context_len = req_to_token_pool.max_context_len
        max_blocks = -(-max_context_len // layout.block_tokens)
        self.mem_pool_host = QSAPlanarHostPool(
            device_pool=pool,
            num_blocks=(
                session_host_blocks(layout)
                if session_resume
                else layout.host_blocks(max_context_len)
            ),
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

        # Session resume (M1b h07). _host_prefix_blocks[r] = h: blocks [0, h) of
        # row r are an adopted record prefix (host only, no device rows).
        self.session_resume = session_resume
        self._host_prefix_blocks = [0] * num_slots
        self.landing_zone: Optional[QSAContextLandingZone] = None
        if session_resume:
            ps = layout.page_size
            self.landing_zone = QSAContextLandingZone(
                pool, -(-max_context_len // ps) * ps
            )

        # M1b h08: decode GDN snapshots, off unless configured.
        self.snapshots = (
            None if snapshot is None else QSAHiSparseSnapshots(self, snapshot)
        )

        self._warmup_kernels()
        logger.info(
            "QSA HiSparse coordinator: top_k=%d blocks, B=%d, %d request slots, %d layers, "
            "host %d blocks, session resume %s (landing zone %.1f MB)",
            top_k,
            hot,
            num_slots - 1,
            layer_num,
            self.mem_pool_host.size,
            session_resume,
            0.0 if self.landing_zone is None else self.landing_zone.nbytes / 1e6,
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

    def release_host_row(self, req_pool_idx: int, keep: int) -> None:
        """Free host blocks [keep, allocated) of row r and clear the row (M1b
        design §2.2 step 5, §11.5).

        Blocks [0, keep) stay allocated: the session record the request pinned
        owns them (keep = its P / 4; 0 when nothing is kept). With
        session_resume the cache's finish path calls this after request_finished
        / abort_staging_request; without it those two call it with keep = 0.
        Idempotent: a row with nothing allocated is left as is.
        """
        r = req_pool_idx
        page = self.layout.page_blocks
        if keep < 0 or keep % page:
            raise ValueError(f"keep={keep} host blocks is not a multiple of the page ({page})")
        allocated = int(self.req_to_host_pool_allocated_len[r])
        adopted = self._host_prefix_blocks[r]
        if allocated == 0:
            if adopted:
                raise RuntimeError(f"row {r} adopted {adopted} host blocks but holds none")
            return
        if not adopted <= keep <= allocated:
            raise RuntimeError(
                f"row {r}: keep={keep} outside [adopted prefix {adopted}, allocated {allocated}]"
            )
        if r in self._staging:
            raise RuntimeError(f"row {r} is staging; abort_staging_request comes first")
        # Blocks freed here may be handed out at once: order the frees after any
        # forward (swap-in reads) and decode backup (writes) still in flight.
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        freed = self.req_to_host_pool[r, keep:allocated]
        if bool((freed < 0).any()):
            raise RuntimeError(f"row {r}: unallocated entries inside [{keep}, {allocated})")
        if freed.numel() > 0:
            self.mem_pool_host.free(freed)
        self.req_to_host_pool[r, :] = -1
        self.req_to_host_pool_allocated_len[r] = 0
        self._host_prefix_blocks[r] = 0

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
        # Blocks [0, h) of a resumed request are its record's, on host only
        # (h07); every copy below that reads the prefill area covers [h, nb).
        h = self._host_prefix_blocks[r]
        if int(self.req_to_host_pool_allocated_len[r]) != h:
            raise RuntimeError(
                f"staging {req.rid}: host row holds "
                f"{int(self.req_to_host_pool_allocated_len[r])} blocks, adopted prefix {h}"
            )
        plan = layout.staging_plan(prompt_len, h)
        logical_blocks = self._logical_blocks(r, prompt_len)[h:]
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
            h,
            plan.full_blocks - h,
        )
        slab = self.req_to_device_buffer[r]
        src, dst = [], []
        from_prefill = plan.slab_from_prefill
        if len(from_prefill):
            src.append(prefill_blocks[from_prefill.start - h : from_prefill.stop - h])
            dst.append(slab[from_prefill.start : from_prefill.stop])
        if plan.newest_block is not None:
            if plan.newest_block < h:
                raise RuntimeError(f"staging {req.rid}: newest block inside the host prefix")
            src.append(prefill_blocks[plan.newest_block - h : plan.newest_block - h + 1])
            dst.append(slab[layout.hot_blocks : layout.hot_blocks + 1])
        empty = slab[:0]
        d2d_src = torch.cat(src) if src else empty
        d2d_dst = torch.cat(dst) if dst else empty
        from_host = plan.slab_from_host
        load_src = self.req_to_host_pool[r, from_host.start : from_host.stop]
        load_dst = slab[from_host.start : from_host.stop]

        req.hisparse_staging = True
        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_blocks(
                prefill_blocks[: plan.full_blocks - h], host_blocks
            )
            self.mem_pool_device.copy_blocks(d2d_src, d2d_dst)
            self.mem_pool_host.load_blocks(load_src, load_dst)
            finish_event.record()
            for t in (prefill_blocks, host_blocks, d2d_src, d2d_dst, load_src, load_dst):
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
        if not self.session_resume:
            self.release_host_row(r, 0)
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
        """Copy the block the previous step completed to host (design §2 decode row).

        With snapshots on, the same window also D2Ds every due GDN / PLE state into
        a staging slot (M1b design §3.2): one event covers both, and the next
        forward waits for it before it writes the live slot.
        """
        snaps = self.snapshots
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
            if snaps is not None:
                snaps.note_backup(r, block)
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
        due = [] if snaps is None else snaps.plan(seq_lens_cpu, req_pool_indices_cpu)
        if not device_blocks and not due:
            return
        if device_blocks:
            device_locs = torch.tensor(device_blocks, dtype=torch.int64, device=self.device)
            host_locs = torch.cat(host_blocks)
        if due:
            live, staging = snaps.staging_indices(due)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            if device_blocks:
                self.mem_pool_host.backup_blocks(device_locs, host_locs)
                device_locs.record_stream(self.decode_backup_stream)
                host_locs.record_stream(self.decode_backup_stream)
            if due:
                snaps.copy_to_staging(live, staging)
                live.record_stream(self.decode_backup_stream)
                staging.record_stream(self.decode_backup_stream)
            self._backup_done_event.record()
        self._has_pending_backup = True
        if due:
            snaps.start_d2h(due, self._backup_done_event)

    # ------------------------------------------------------------ GDN snapshots (M1b h08)

    def collect_ready_reqs(self) -> List[Req]:
        """Staging acks (upstream), plus the snapshot poll and decode registration."""
        if self.snapshots is None:
            return super().collect_ready_reqs()
        self.snapshots.poll()
        ready = super().collect_ready_reqs()
        for req in ready:
            self.snapshots.enter_decode(req)
        return ready

    def _require_snapshots(self) -> "QSAHiSparseSnapshots":
        if self.snapshots is None:
            raise RuntimeError(
                "QSA HiSparse snapshots are off (hisparse_config.qsa_snapshot_interval)"
            )
        return self.snapshots

    def set_snapshot_owner(self, owner: "SnapshotOwner") -> None:
        self._require_snapshots().set_owner(owner)

    def release_snapshots(self, req: Req) -> None:
        """Wait for req's in-flight snapshot D2H, publish them, drop its decode state.

        The owner calls it on every exit of an admitted request (h09:
        cache_finished_req), before the request slot can be reused.
        """
        self._require_snapshots().release(req)

    def load_snapshots(self, bufs: List[int], dst_slots: torch.Tensor) -> None:
        """H2D published GDN buffers into new mamba slots on the current stream
        (the deferred restore model_runner runs before an extend)."""
        self._require_snapshots().load(bufs, dst_slots)

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
        if self._debug_host_records and not torch.cuda.is_current_stream_capturing():
            self._check_host_records(req_pool_indices, compressed_seq_lens, top_k_result)
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

    def _check_host_records(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
    ) -> None:
        """ARC_HISPARSE_QSA_DEBUG: every block the swap-in kernel may copy from host has a record.

        On a slow-path row (seq_blocks > B) any selected block but the newest
        (slot B) can miss, and a miss copies from req_to_host_pool[r, block]. A -1
        there is a broken lifecycle rule (staging or decode backup skipped); the
        kernel would read before the host pool, an illegal memory access that
        poisons the CUDA context. Host sync, so eager only.
        """
        n = int(self.num_real_reqs)
        rpi = req_pool_indices[:n].to(torch.int64)
        nb = compressed_seq_lens[:n].to(torch.int64)
        top_k = top_k_result[:n].to(torch.int64)
        slow = nb > self.device_buffer_size
        may_miss = slow[:, None] & (top_k >= 0) & (top_k < nb[:, None] - 1)
        host = self.req_to_host_pool[rpi[:, None], top_k.clamp(min=0)]
        lost = may_miss & (host < 0)
        if bool(lost.any()):
            i, j = lost.nonzero()[0].tolist()
            raise RuntimeError(
                f"QSA HiSparse swap-in: request slot {int(rpi[i])} (seq_blocks "
                f"{int(nb[i])}) selected block {int(top_k[i, j])}, which can miss but "
                f"has no host record ({int(lost.sum())} such entries in this batch)"
            )

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
        if not self.session_resume:
            self.release_host_row(r, 0)
        self.req_device_buffer_tokens[:, r, :] = -1
        self.lru_slots[:, r, :].copy_(self._lru_init)
        self._skip_first_backup[r] = False

    # ------------------------------------------------------------ session resume (M1b h07)

    def adopt_host_prefix(self, reqs: Sequence[Req]) -> None:
        """Give each resumed request its session record's host blocks [0, P/4)
        (design §2.2 step 3). Call after alloc_for_extend has written the
        record's logical slots into req_to_token[r, :P] and before the extend
        forward; once per admission, on a row that holds no host blocks.

        Contract (h09): req.hisparse_resume.host_blocks is the record's int64
        CUDA tensor of P/4 host blocks, P = len(req.prefix_indices) =
        req.kv.cache_protected_len, a positive multiple of the page size. The
        record keeps owning the blocks; release_host_row(r, keep >= P/4) leaves them.
        """
        if not self.session_resume:
            raise RuntimeError("adopt_host_prefix needs session_resume")
        page, bt = self.layout.page_blocks, self.layout.block_tokens
        for req in reqs:
            r = req.kv.req_pool_idx
            host = req.hisparse_resume.host_blocks
            h = host.numel()
            if host.dtype != torch.int64 or host.dim() != 1 or host.device != self.req_to_host_pool.device:
                raise ValueError(
                    f"{req.rid}: record host blocks must be 1-D int64 on "
                    f"{self.req_to_host_pool.device}, got {host.dtype} {tuple(host.shape)} on {host.device}"
                )
            if h == 0 or h % page:
                raise ValueError(f"{req.rid}: {h} host blocks is not a positive multiple of {page}")
            if len(req.prefix_indices) != h * bt or req.kv.cache_protected_len != h * bt:
                raise RuntimeError(
                    f"{req.rid}: resume point {h * bt} but prefix {len(req.prefix_indices)}, "
                    f"cache_protected_len {req.kv.cache_protected_len}"
                )
            if (
                int(self.req_to_host_pool_allocated_len[r])
                or self._host_prefix_blocks[r]
                or r in self._staging
            ):
                raise RuntimeError(f"{req.rid}: row {r} already holds host blocks")
            self.req_to_host_pool[r, :h] = host
            self.req_to_host_pool_allocated_len[r] = h
            self._host_prefix_blocks[r] = h

    def host_prefix_lens(self, req_pool_indices: Sequence[int]) -> List[int]:
        """Tokens of each row's adopted host prefix (0 for none)."""
        bt = self.layout.block_tokens
        return [self._host_prefix_blocks[r] * bt for r in req_pool_indices]

    def has_host_prefix(self) -> bool:
        return any(self._host_prefix_blocks)

    def fill_host_prefix_context(
        self,
        layer_index: int,
        segments: Sequence[Tuple[int, int, int]],
        k_context: torch.Tensor,
        v_context: torch.Tensor,
        expand: Optional[Callable[[QSAContextLandingZone, List[Tuple[int, int, int]]], None]] = None,
    ) -> None:
        """Write the host part of resumed prefixes into a packed prefill context,
        on the current stream (design §4.3). segments: (req_pool_idx, P, first
        context row) per request with a host prefix of P tokens; rows
        [start, start + P) of k_context / v_context receive positions [0, P).

        Records go through the landing zone, as many requests per fill as it
        holds, then are copied out into the context (the extra D2D disappears
        once the zone doubles as the 0030 workspace). A zone of packed records
        (NVFP4, arc-forge h12) is not the context yet: ``expand(zone, [(blocks,
        first context row, first zone block), ...])`` writes each request's
        part instead of the copy.
        """
        zone = self.landing_zone
        if zone is None:
            raise RuntimeError("fill_host_prefix_context needs session_resume")
        if expand is None and len(zone.planes) != 2:
            raise RuntimeError("packed (NVFP4) host records need an expansion into the context")
        bt = self.layout.block_tokens
        group: List[Tuple[int, int, int, int]] = []

        def flush() -> None:
            host = torch.cat([self.req_to_host_pool[r, :n] for r, n, _, _ in group])
            self.mem_pool_host.fill_context_host(layer_index, host, zone)
            if expand is not None:
                expand(zone, [(n, start, off) for _, n, start, off in group])
                return
            zone_k, zone_v = (p.view(k_context.dtype) for p in zone.planes)
            for _, n, start, off in group:
                k_context[start : start + n * bt].copy_(zone_k[off * bt : (off + n) * bt])
                v_context[start : start + n * bt].copy_(zone_v[off * bt : (off + n) * bt])

        used = 0
        for r, prefix, start in segments:
            n = self._host_prefix_blocks[r]
            if prefix <= 0 or prefix != n * bt:
                raise RuntimeError(
                    f"row {r}: host segment of {prefix} tokens, adopted prefix {n * bt}"
                )
            if n > zone.capacity_blocks:
                raise RuntimeError(f"row {r}: {n} host blocks exceed the landing zone")
            if used + n > zone.capacity_blocks:
                flush()
                group, used = [], 0
            group.append((r, n, start, used))
            used += n
        if group:
            flush()

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

    def check_idle_invariants(
        self,
        parked_logical: Optional[torch.Tensor] = None,
        parked_host: Optional[torch.Tensor] = None,
        parked_gdn: Optional[List[int]] = None,
    ) -> None:
        """With nothing running or staging, every pool and table is back at its
        start (R3), except what parked session records hold (M1b h07):
        parked_logical / parked_host are their logical token slots and host
        blocks, concatenated (None: nothing parked, the M1a case); parked_gdn
        their GDN host buffers (h08: with snapshots on, exactly those are
        allocated)."""
        problems = []
        if self.ack_staging_queue or self._staging:
            problems.append(
                f"staging queue {len(self.ack_staging_queue)}, staging slots {sorted(self._staging)}"
            )
        alloc = self.token_to_kv_pool_allocator
        n_parked = 0 if parked_logical is None else parked_logical.numel()
        for name, sub, full in (
            ("logical", alloc.logical_attn_allocator, self.layout.logical_tokens - n_parked),
            ("prefill area", alloc.prefill_block_allocator, self.layout.prefill_blocks),
        ):
            if sub.available_size() != full:
                problems.append(f"{name} allocator {sub.available_size()} / {full} free")
        if n_parked:
            logical = alloc.logical_attn_allocator
            pages = torch.unique(parked_logical.to(torch.int64) // logical.page_size)
            if pages.numel() * logical.page_size != n_parked or bool(
                torch.isin(pages, logical.get_all_free_pages().to(pages.device)).any()
            ):
                problems.append("parked logical slots are not whole pages held off the free list")
        host = self.mem_pool_host
        used = host.slot_used.nonzero().flatten()
        parked = (
            torch.empty(0, dtype=torch.int64)
            if parked_host is None
            else torch.sort(parked_host.to("cpu", torch.int64)).values
        )
        if host.available_size() != host.size - parked.numel() or not torch.equal(used, parked):
            problems.append(
                f"host {host.available_size()} / {host.size} free, "
                f"{used.numel()} slots marked used, {parked.numel()} parked"
            )
        if any(self._host_prefix_blocks):
            problems.append("adopted host prefixes left on idle rows")
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
        if self.snapshots is not None:
            problems.extend(self.snapshots.idle_problems(parked_gdn or []))
        elif parked_gdn:
            problems.append(f"parked GDN buffers {parked_gdn} with snapshots off")
        if problems:
            raise RuntimeError("QSA HiSparse idle invariants violated: " + "; ".join(problems))

    def destroy(self) -> None:
        if self.snapshots is not None:
            self.snapshots.destroy()
        super().destroy()

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
    # M1b h07: session resume from host-resident prefixes. Only the session
    # cache (h09, --radix-cache-backend qsa_hisparse_session) parks prefixes and
    # calls release_host_row; with any other cache host rows would never be freed.
    session_resume = user.get("qsa_session_resume", False)
    if not isinstance(session_resume, bool):
        raise ValueError(f"hisparse_config.qsa_session_resume must be a bool, got {session_resume!r}")
    if session_resume and get_memory().radix_cache_backend != "qsa_hisparse_session":
        raise ValueError(
            "hisparse_config.qsa_session_resume needs --radix-cache-backend "
            f"qsa_hisparse_session (M1b h09), got {get_memory().radix_cache_backend!r}"
        )
    snapshot = _snapshot_config(user, num_reqs)
    if session_resume:
        # M1b h09: register the session cache backend (build_kv_cache comes next)
        # and check the host budget of what the records park before pinning it.
        import sglang.srt.mem_cache.qsa_hisparse_session_cache  # noqa: F401

        if snapshot is None:
            raise ValueError(
                "hisparse_config.qsa_session_resume needs qsa_snapshot_interval (h08): "
                "a session resumes from its latest GDN snapshot"
            )
        _check_session_host_budget(
            pool, user, snapshot, model_runner.req_to_token_pool.mamba_pool
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
        session_resume=session_resume,
        snapshot=snapshot,
    )


def session_host_blocks(layout) -> int:
    """M1b design §5.1, §11.9: host blocks follow L (L / bt), plus one page of
    slack per request slot. Every owner holds no more host pages than logical
    pages, so the host pool never runs out before the logical one."""
    return layout.logical_tokens // layout.block_tokens + layout.num_slabs * layout.page_blocks


def _check_session_host_budget(pool, user: dict, snapshot, mamba_pool) -> None:
    """Fail at startup unless the host KV pool and the GDN buffers fit next to
    hisparse_config.host_reserve_gib (RAM for the OS, agents, page cache; R3)."""
    import psutil

    reserve = user.get("host_reserve_gib")
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or reserve < 0:
        raise ValueError(
            "the QSA HiSparse session cache needs hisparse_config.host_reserve_gib >= 0, "
            f"got {reserve!r}"
        )
    kv = session_host_blocks(pool.layout) * pool.layer_num * pool.plane_spec.record_bytes
    gdn = snapshot.host_buffers * mamba_pool.host_slot_nbytes()
    need = kv + gdn + int(reserve * (1 << 30))
    available = psutil.virtual_memory().available
    if need > available:
        raise ValueError(
            f"QSA HiSparse session cache: host KV pool {kv / (1 << 30):.2f} GiB (L = "
            f"{pool.layout.logical_tokens} tokens) + GDN buffers {gdn / (1 << 30):.2f} GiB + "
            f"reserve {reserve} GiB exceed the {available / (1 << 30):.2f} GiB available; "
            "lower hisparse_config.qsa_logical_tokens or qsa_snapshot_host_buffers"
        )
    logger.info(
        "QSA HiSparse session host budget: KV %.2f GiB + GDN %.2f GiB + reserve %.1f GiB "
        "of %.2f GiB available",
        kv / (1 << 30),
        gdn / (1 << 30),
        reserve,
        available / (1 << 30),
    )


def _snapshot_config(user: dict, num_reqs: int) -> Optional["QSAHiSparseSnapshotConfig"]:
    """M1b h08 keys of hisparse_config: decode GDN snapshots are off unless
    qsa_snapshot_interval (T) is set; K staging slots default to 8 and the host
    buffers to R + 16 (design §3.3, §5.3)."""
    keys = (
        "qsa_snapshot_interval",
        "qsa_snapshot_staging_slots",
        "qsa_snapshot_host_buffers",
    )
    for key in keys:
        value = user.get(key)
        if key in user and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"hisparse_config.{key} must be an integer, got {value!r}")
    if keys[0] not in user:
        if any(key in user for key in keys[1:]):
            raise ValueError(f"hisparse_config.{keys[1]} / {keys[2]} need {keys[0]}")
        return None
    return QSAHiSparseSnapshotConfig(
        interval=user[keys[0]],
        staging_slots=user.get(keys[1], 8),
        host_buffers=user.get(keys[2], num_reqs + 16),
    )


# ---------------------------------------------------------------- GDN snapshots (M1b h08)


class QSAHiSparseSnapshotConfig(NamedTuple):
    interval: int  # T: snapshot when a decode input position p is a positive multiple of T
    staging_slots: int  # K mamba slots reserved for the D2D, for good
    host_buffers: int  # pinned host buffers of one mamba slot each (S_max)


class SnapshotOwner(Protocol):
    """Owner of the GDN host buffers and of what a published snapshot means (h09).

    snapshot_buffer(req, p): the buffer req's snapshot at position p goes to, or
      None to skip this boundary (req does not own its session, or no buffer).
      Called at most once per due boundary from the decode prepare, and only
      when a staging slot is free. The owner allocates buffers with
      QSAHiSparseSnapshots.alloc_buffer() and keeps them until it frees them.
    on_snapshot_published(req, p, buf): buf now holds exactly the GDN and PLE
      state of positions [0, p) -- the state after the forward whose last input
      is position p - 1 -- host blocks [0, p / 4) of req have finished copies
      and p <= req.kv.kv_committed_len. Per request in issue order, from
      collect_ready_reqs or release_snapshots.
    """

    def snapshot_buffer(self, req: Req, p: int) -> Optional[int]: ...

    def on_snapshot_published(self, req: Req, p: int, buf: int) -> None: ...


@dataclass(eq=False)
class _Snapshot:
    req: Req
    r: int  # request slot
    p: int
    buf: int
    staging: int  # staging mamba slot
    host_blocks: int  # host copies of blocks [0, host_blocks) issued by this window
    done: Optional[torch.cuda.Event] = None  # D2H finished


class QSAHiSparseSnapshots:
    """Decode GDN / PLE snapshots (M1b design §3.2, §11.6-11.8).

    In the decode backup window of the step whose input is position p (a
    multiple of T), the live mamba slot is D2D'd into a staging slot (same event
    as the KV backup, so the next forward cannot overwrite it first); a separate
    stream D2Hs the staging slot into the owner's pinned buffer; poll() (from
    collect_ready_reqs) publishes each snapshot once its D2H event is seen done.
    A buffer with an unpublished snapshot takes no new one (§11.8, counted).
    """

    def __init__(
        self, coord: QSAHiSparseCoordinator, config: QSAHiSparseSnapshotConfig
    ):
        pool = coord.req_to_token_pool
        if not isinstance(pool, HybridReqToTokenPool):
            raise TypeError(
                f"GDN snapshots need a HybridReqToTokenPool, got {type(pool).__name__}"
            )
        if (
            pool.mamba_v2p_table is not None
            or type(pool).translate_mamba_indices
            is not HybridReqToTokenPool.translate_mamba_indices
            or type(pool.mamba_allocator) is not MambaSlotAllocator
        ):
            raise NotImplementedError(
                "GDN snapshots need physical mamba slot ids (a static mamba pool)"
            )
        self.layout = coord.layout
        self.layout.snapshot_due(config.interval + 1, config.interval)  # validates T
        if config.staging_slots < 1 or config.host_buffers < 1:
            raise ValueError(f"need >= 1 staging slot and host buffer, got {config}")
        self.interval = config.interval
        self.pool = pool
        self.mamba = pool.mamba_pool
        self.device = pool.req_index_to_mamba_index_mapping.device
        need = self.layout.num_slabs + config.staging_slots
        free = pool.mamba_allocator.available_size()
        if free < need:
            raise ValueError(
                f"GDN snapshots reserve {config.staging_slots} staging mamba slots next "
                f"to {self.layout.num_slabs} running requests: --max-mamba-cache-size "
                f"must be >= {need}, {free} free"
            )
        self.staging_slots: List[int] = pool.mamba_allocator.reserve(
            config.staging_slots
        ).tolist()
        self.staging_free = list(self.staging_slots)
        # Compile the fused conv-slot copy now rather than in the first decode window
        # (slot 0 is the padding dummy; a staging slot is scratch until a D2D).
        self.mamba.copy_from(
            torch.zeros(1, dtype=torch.int64, device=self.device),
            torch.tensor(self.staging_slots[:1], dtype=torch.int64, device=self.device),
        )

        self.slot_nbytes = self.mamba.host_slot_nbytes()
        total = self.slot_nbytes * config.host_buffers
        available = host_memory_budget_bytes()
        if total > available:
            raise ValueError(
                f"GDN host buffers need {total / 1e9:.2f} GB ({config.host_buffers} x "
                f"{self.slot_nbytes} B) but only {available / 1e9:.2f} GB of host RAM "
                "is available"
            )
        alloc = ALLOC_MEMORY_FUNCS[torch.device(coord.device).type]
        self.host_raw = alloc(
            (config.host_buffers, self.slot_nbytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
            allocator=HostTensorAllocator(),
        )
        self.buffers = [
            self.mamba.host_slot_view(self.host_raw[i])
            for i in range(config.host_buffers)
        ]
        # pop() hands out buffer 0 first
        self.buffer_free = list(range(config.host_buffers - 1, -1, -1))

        self.busy: Dict[int, _Snapshot] = {}  # buffer -> unpublished snapshot (§11.8)
        self.pending: List[_Snapshot] = []  # issue order == completion order (one stream)
        self.decoding: Dict[int, Req] = {}  # request slot -> request, ready to release
        self.host_blocks: Dict[int, int] = {}  # request slot -> host copies issued [0, n)
        self.d2h_stream = device_module.Stream()
        self.owner: Optional[SnapshotOwner] = None
        self.stats: Counter = Counter()
        self._destroyed = False
        logger.info(
            "QSA HiSparse GDN snapshots: T=%d, staging mamba slots %s, %d host buffers "
            "x %d B = %.2f GB pinned",
            self.interval,
            self.staging_slots,
            config.host_buffers,
            self.slot_nbytes,
            total / 1e9,
        )

    # ------------------------------------------------------------ owner side

    def set_owner(self, owner: SnapshotOwner) -> None:
        if self.owner is not None:
            raise RuntimeError("the GDN snapshot owner is already set")
        self.owner = owner

    def alloc_buffer(self) -> Optional[int]:
        return self.buffer_free.pop() if self.buffer_free else None

    def free_buffer(self, buf: int) -> None:
        if buf in self.busy:
            raise RuntimeError(f"GDN buffer {buf} freed while a snapshot D2H writes it")
        if not 0 <= buf < len(self.buffers) or buf in self.buffer_free:
            raise RuntimeError(f"GDN buffer {buf} is not allocated")
        self.buffer_free.append(buf)

    # ------------------------------------------------------------ decode side

    def enter_decode(self, req: Req) -> None:
        """Staging finished: host copies cover the complete prompt blocks."""
        r = req.kv.req_pool_idx
        if r in self.decoding:
            raise RuntimeError(
                f"request slot {r} still holds the snapshot state of "
                f"{self.decoding[r].rid}; release_snapshots was not called"
            )
        if req.kv.kv_committed_len != req.kv.kv_allocated_len:
            raise RuntimeError(f"{req.rid} enters decode with uncommitted KV")
        self.decoding[r] = req
        self.host_blocks[r] = req.kv.kv_allocated_len // self.layout.block_tokens

    def note_backup(self, r: int, block: int) -> None:
        if r not in self.host_blocks:
            raise RuntimeError(f"decode backup for request slot {r}, which is not decoding")
        if block != self.host_blocks[r]:
            raise RuntimeError(
                f"decode backup of block {block} for slot {r}, but host copies cover "
                f"[0, {self.host_blocks[r]})"
            )
        self.host_blocks[r] = block + 1

    def plan(
        self, seq_lens_cpu: torch.Tensor, req_pool_indices_cpu: torch.Tensor
    ) -> List[_Snapshot]:
        """The snapshots this step's backup window takes (CPU only)."""
        due: List[_Snapshot] = []
        for i in range(len(seq_lens_cpu)):
            p = self.layout.snapshot_due(int(seq_lens_cpu[i]), self.interval)
            if p is None:
                continue
            r = int(req_pool_indices_cpu[i])
            req = self.decoding.get(r)
            if req is None:
                raise RuntimeError(f"decode row of request slot {r} never entered decode")
            if not self.staging_free:
                self.stats["skip_staging"] += 1
                continue
            if self.owner is None:
                raise RuntimeError("GDN snapshots are on but no owner is registered (h09)")
            buf = self.owner.snapshot_buffer(req, p)
            if buf is None:
                self.stats["skip_owner"] += 1
                continue
            if not 0 <= buf < len(self.buffers) or buf in self.buffer_free:
                raise RuntimeError(f"owner gave GDN buffer {buf}, which is not allocated")
            if buf in self.busy:
                self.stats["skip_busy"] += 1
                continue
            snap = _Snapshot(
                req, r, p, buf, self.staging_free.pop(), self.host_blocks[r]
            )
            self.busy[buf] = snap
            due.append(snap)
        return due

    def staging_indices(
        self, due: List[_Snapshot]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(live, staging) mamba slot ids on the device, built on the schedule stream."""
        rows = torch.tensor([s.r for s in due], dtype=torch.int64, device=self.device)
        live = self.pool.req_index_to_mamba_index_mapping[rows].to(torch.int64)
        staging = torch.tensor(
            [s.staging for s in due], dtype=torch.int64, device=self.device
        )
        return live, staging

    def copy_to_staging(self, live: torch.Tensor, staging: torch.Tensor) -> None:
        """D2D on the decode backup stream, inside the window."""
        self.mamba.copy_from(live, staging)

    def start_d2h(self, due: List[_Snapshot], after) -> None:
        with device_module.stream(self.d2h_stream):
            self.d2h_stream.wait_event(after)
            for snap in due:
                self.mamba.save_slot_to_host(
                    snap.staging, self.buffers[snap.buf], self.d2h_stream
                )
                snap.done = device_module.Event()
                snap.done.record(self.d2h_stream)
        self.pending.extend(due)
        self.stats["issued"] += len(due)

    def poll(self) -> None:
        """Publish, in issue order, every snapshot whose D2H is seen done."""
        while self.pending and self.pending[0].done.query():
            self._publish(self.pending.pop(0), d2h_done=True)

    def _publish(self, snap: _Snapshot, *, d2h_done: bool) -> None:
        req, p = snap.req, snap.p
        committed = req.kv.kv_committed_len
        if p > committed:  # §11.7: condition 1 always holds without spec
            raise RuntimeError(f"{req.rid}: snapshot p={p} beyond kv_committed_len {committed}")
        overwritten = any(o.buf == snap.buf for o in self.pending)
        if not self.layout.can_publish(
            p,
            committed=committed,
            host_blocks_done=snap.host_blocks,
            gdn_d2h_done=d2h_done,
            overwritten=overwritten,
        ):
            raise RuntimeError(
                f"{req.rid}: snapshot p={p} into GDN buffer {snap.buf} cannot publish: "
                f"host copies cover [0, {snap.host_blocks}) of "
                f"{p // self.layout.block_tokens} blocks, D2H done {d2h_done}, "
                f"overwritten {overwritten} (design §11.6, §11.8)"
            )
        if self.busy.pop(snap.buf) is not snap:
            raise RuntimeError(f"GDN buffer {snap.buf} busy with another snapshot")
        self.staging_free.append(snap.staging)
        self.stats["published"] += 1
        self.owner.on_snapshot_published(req, p, snap.buf)

    def release(self, req: Req) -> None:
        """Wait for req's in-flight D2H, publish in order, drop its decode state."""
        mine = [snap for snap in self.pending if snap.req is req]
        for snap in mine:
            snap.done.synchronize()
        for snap in mine:
            self.pending.remove(snap)
            self._publish(snap, d2h_done=True)
        slots = [r for r, q in self.decoding.items() if q is req]
        if len(slots) > 1:
            raise RuntimeError(f"{req.rid} is decoding in request slots {slots}")
        for r in slots:
            del self.decoding[r], self.host_blocks[r]

    def load(self, bufs: List[int], dst_slots: torch.Tensor) -> None:
        if len(bufs) != dst_slots.numel():
            raise ValueError(f"{len(bufs)} buffers for {dst_slots.numel()} mamba slots")
        for i, buf in enumerate(bufs):
            if buf in self.busy:
                raise RuntimeError(
                    f"restore from GDN buffer {buf} while a snapshot D2H writes it"
                )
            if not 0 <= buf < len(self.buffers) or buf in self.buffer_free:
                raise RuntimeError(f"restore from GDN buffer {buf}, which is not allocated")
            self.mamba.load_slot_from_host(
                self.buffers[buf], dst_slots[i : i + 1].to(torch.int64)
            )
        self.stats["restored"] += len(bufs)

    # ------------------------------------------------------------ checks

    def idle_problems(self, parked: List[int]) -> List[str]:
        problems = []
        held = sorted(set(range(len(self.buffers))) - set(self.buffer_free))
        if held != sorted(parked) or len(parked) != len(set(parked)):
            problems.append(f"GDN host buffers {held} allocated, parked records hold {sorted(parked)}")
        if self.pending or self.busy or self.decoding:
            problems.append(
                f"snapshots: {len(self.pending)} D2H pending, busy buffers "
                f"{sorted(self.busy)}, decoding slots {sorted(self.decoding)}"
            )
        if sorted(self.staging_free) != sorted(self.staging_slots):
            problems.append(
                f"staging mamba slots {sorted(self.staging_free)} free of "
                f"{self.staging_slots}"
            )
        alloc = self.pool.mamba_allocator
        if alloc.available_size() != alloc.size - len(self.staging_slots):
            problems.append(
                f"mamba allocator {alloc.available_size()} free of {alloc.size} with "
                f"{len(self.staging_slots)} staging slots reserved"
            )
        return problems

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self.d2h_stream.synchronize()
        _cuda_host_unregister(self.host_raw)
