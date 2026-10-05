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
through the landing zone (QSAHostPrefixFill, h15); staging copies only
[P/4, nb) from the prefill area and loads the prefix's identity slots from
host. Host blocks are then released by release_host_row(r, keep) from the
cache's finish path, never by request_finished / abort_staging_request, which
run before it; without session_resume those two release the whole row as M1a.

With qsa_snapshot_interval set (M1b h08), decode also snapshots the GDN / PLE
state every T positions: D2D live -> staging mamba slot in the backup window,
async D2H to a pinned host buffer, published to an owner (h09) once the D2H is
seen done (QSAHiSparseSnapshots below).

Speculative verify (arc-forge M5 h16, design §3-§4, §6): with verify_tokens K > 1
a TARGET_VERIFY forward has K rows per request (request-major). The swap-in
resolves a request's K rows one after another in one CTA, the window's blocks
>= B live in ring slots B + b % W (W = layout.ring_width(K)), the per-row decode
buffers hold max_bs * K rows, and verify_overflow is set when a row evicts a
block an earlier row of its request selected (checked with the step result:
check_verify_overflow). The decode-row path (map_last_loc_to_buffer and its
backup) is K = 1 only.

Speculative serving (arc-forge M5 h17, design §2, §5): a speculative allocation
maps the new window's logical blocks to their ring slots and gives the request
host pages for every logical page it holds (map_spec_window, from
alloc_for_spec_decode); after each verify, backup_after_accept copies the blocks
the accepted tokens completed from their ring slots to host on the forward
stream, before the next verify writes a ring slot or misses on those blocks.
host_done[r] counts the blocks on host (design §5.1 I1).
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
from sglang.srt.distributed import get_attn_tp_group
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
    """Per-forward views of the coordinator's persistent decode buffers ([:rows]); a verify
    forward has K rows per request (h16)."""

    seq_blocks: torch.Tensor  # int32, ceil(seq_len / block_tokens): the kernel's seq_lens
    newest_dev: torch.Tensor  # int32, device block of the row's tail slot ring_slot(seq_blocks - 1)
    rows: torch.Tensor  # int32 [rows, top_k * block_tokens + block_tokens - 1], hot rows (h04 writes)


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
        verify_tokens: int = 1,
        host_admission: Optional[Tuple[int, int]] = None,
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
        self.host_admission = host_admission
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = pool
        self.layout = layout
        # M5 h16: K rows per request in a verify forward; W ring slots after the LRU slots.
        self.verify_tokens = verify_tokens
        self.ring_width = layout.ring_width(verify_tokens)
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
        from sglang.srt.mem_cache.qsa_hisparse_distributed import (
            AttentionTPConsensus, distributed_hisparse_enabled, distributed_spec_enabled, distributed_mtp_steps, new_spec_protocols,
            tp4_attention_enabled,
        )
        self.distributed = distributed_hisparse_enabled()
        if self.tp_world_size != 1 and not self.distributed:
            raise NotImplementedError("QSA HiSparse is TP1 unless ARC_HISPARSE_DISTRIBUTED=1")
        self.distributed_spec = distributed_spec_enabled()
        expected_verify = distributed_mtp_steps() + 1 if self.distributed_spec else 1
        attention_tp = 4 if tp4_attention_enabled() else 2
        if self.distributed and (self.tp_world_size != attention_tp or verify_tokens != expected_verify):
            raise NotImplementedError(f"distributed HiSparse requires attention TP{attention_tp} and the explicitly "
                                      "gated verify width")
        if self.distributed and host_admission is None:
            raise ValueError("distributed HiSparse requires collective host admission before pin")
        if self.distributed:
            from sglang.srt.layers.dp_attention import get_attention_dp_rank
            self.namespace = f"attention-dp-{get_attention_dp_rank()}"
        else:
            self.namespace = "attention-dp-0"
        self.consensus = AttentionTPConsensus(tp_group, self.tp_world_size, self.namespace)
        self.spec_protocols = new_spec_protocols(self.namespace,
            torch.distributed.get_process_group_ranks(tp_group)) if self.distributed_spec else {}

        max_context_len = req_to_token_pool.max_context_len
        max_blocks = -(-max_context_len // layout.block_tokens)
        self.mem_pool_host = QSAPlanarHostPool(
            device_pool=pool,
            num_blocks=(
                session_host_blocks(layout)
                if session_resume
                else layout.host_blocks(max_context_len)
            )
            # h17: plus the page the verify backup's dummy entries write to
            + (layout.page_blocks if verify_tokens > 1 else 0),
            admitted_bytes=None if host_admission is None else host_admission[0],
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
        # Decode rows: real requests plus CUDA-graph padding (max_decode_rows), K rows
        # per request in a verify forward (h16; the caller scales max_decode_rows).
        max_rows = max(num_slots * verify_tokens, max_decode_rows)
        self.top_k_device_locs_buffer = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)
        # h16: set by a multi-row verify swap-in that evicted a block an earlier row of
        # the same request selected; never cleared (the server stops on it).
        self.verify_overflow = torch.zeros(1, dtype=torch.int32, device=device)
        self._skip_first_backup = [False] * num_slots
        # M5 h17: blocks [0, host_done[r]) of row r are on host (staging sets it to
        # the prompt's complete blocks, backup_after_accept advances it; design §5.1).
        self.host_done = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self.spec_backup_width = layout.spec_backup_width(verify_tokens)
        self._backup_offsets = torch.arange(
            self.spec_backup_width, dtype=torch.int64, device=device
        )
        # A verify's backup is a fixed list of bs x spec_backup_width blocks (no
        # host sync); an entry without a completed block copies device block 0 into
        # this reserved host page (the transfer kernel has no skip for -1).
        self._backup_dump_pages: Optional[torch.Tensor] = None
        self._backup_dump_host = -1
        if verify_tokens > 1:
            dump = self.mem_pool_host.alloc(self.page_size)
            if dump is None:
                raise RuntimeError("QSA HiSparse host pool has no page left for the verify backup dump")
            self._backup_dump_pages = dump
            self._backup_dump_host = int(dump[0])

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
        # h15: host -> landing-zone fills run here, one layer ahead of the
        # forward stream (QSAHostPrefixFill); the fill of the current forward.
        self.context_fill_stream = None
        self._host_fill: Optional["QSAHostPrefixFill"] = None
        if session_resume:
            ps = layout.page_size
            self.landing_zone = QSAContextLandingZone(
                pool, -(-max_context_len // ps) * ps
            )
            self.context_fill_stream = device_module.Stream()

        # M1b h08: decode GDN snapshots, off unless configured.
        self.snapshots = (
            None if snapshot is None else QSAHiSparseSnapshots(self, snapshot)
        )

        self._warmup_kernels()
        logger.info(
            "QSA HiSparse coordinator: top_k=%d blocks, B=%d, %d request slots, %d layers, "
            "host %d blocks, session resume %s (landing zone %.1f MB), verify rows %d "
            "(ring %d slots, %d decode rows)",
            top_k,
            hot,
            num_slots - 1,
            layer_num,
            self.mem_pool_host.size,
            session_resume,
            0.0 if self.landing_zone is None else self.landing_zone.nbytes / 1e6,
            verify_tokens,
            self.ring_width,
            max_rows,
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
        self.consensus.agree("host-row-release", [req_pool_idx, keep])
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
        self.consensus.agree("staging-admit", [req.rid, req.kv.req_pool_idx])
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
        # h17: every complete prompt block is on host once this staging's D2H runs
        self.host_done[r].fill_(plan.full_blocks)
        slab = self.req_to_device_buffer[r]
        src, dst = [], []
        from_prefill = plan.slab_from_prefill
        if len(from_prefill):
            src.append(prefill_blocks[from_prefill.start - h : from_prefill.stop - h])
            dst.append(slab[from_prefill.start : from_prefill.stop])
        if plan.newest_block is not None:
            if plan.newest_block < h:
                raise RuntimeError(f"staging {req.rid}: newest block inside the host prefix")
            # h16: the ring slot a verify window reads it from (slot B when K = 1)
            ring = layout.ring_slot(plan.newest_block, self.ring_width)
            src.append(prefill_blocks[plan.newest_block - h : plan.newest_block - h + 1])
            dst.append(slab[ring : ring + 1])
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
        self.mem_pool_device.mapping.index_fill_(0, logical_blocks.long(), 0)  # no CPU scalar copy (stream sync)

    def abort_staging_request(self, req: Req) -> None:
        self.consensus.agree("staging-abort", [req.rid, req.kv.req_pool_idx])
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        self.write_staging_stream.synchronize()
        self.consensus.agree("staging-abort-complete", [req.rid, req.kv.req_pool_idx])
        r = req.kv.req_pool_idx
        prefill_blocks, logical_blocks = self._staging.pop(r)
        self.token_to_kv_pool_allocator.free_prefill_blocks(prefill_blocks)
        self.mem_pool_device.mapping.index_fill_(0, logical_blocks.long(), 0)  # no CPU scalar copy (stream sync)
        if not self.session_resume:
            self.release_host_row(r, 0)
        self._skip_first_backup[r] = False
        self.host_done[r].fill_(0)
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
        if self.verify_tokens > 1:
            # h16: a speculative coordinator maps a verify window's blocks to ring slots
            # at allocation and backs up after accept (h17); slot B alone is the K = 1 rule.
            raise RuntimeError(
                f"QSA HiSparse decode-row mapping with verify_tokens={self.verify_tokens}: "
                "speculative decode goes through the verify path"
            )
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
        if not self.distributed:
            if self.snapshots is None:
                return super().collect_ready_reqs()
            self.snapshots.poll()
            ready = super().collect_ready_reqs()
        else:
            if self.snapshots is not None:
                self.snapshots.poll()
            count = 0
            for _, done, _ in self.ack_staging_queue:
                if not done.query():
                    break
                count += 1
            keys = [[act.req.rid, act.req.kv.req_pool_idx] for act in self.ack_staging_queue]
            count = self.consensus.ready_count("staging-ready", keys, count)
            ready = []
            for _ in range(count):
                _, _, req = self.ack_staging_queue.pop(0)
                self.alloc_device_buffer(req)
                self._skip_first_backup[req.kv.req_pool_idx] = True
                req.hisparse_staging = False
                ready.append(req)
        if self.snapshots is not None:
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
        """Fill the persistent per-row buffers for one decode or verify forward.

        bs counts rows: one per request in a decode forward, K per request in a
        verify forward (h16), where seq_lens / req_pool_indices are the rows' own
        (position + 1, and the row's request). A row's tail block (seq_blocks - 1)
        sits in ring_slot(seq_blocks - 1): its identity slot below B, B + b % W from
        B on (min(nb - 1, B) for W = 1). No host sync, so it can run inside a
        CUDA-graph capture; seq_lens and req_pool_indices may be int32 or int64.
        The backend reads the buffers, not the returned views.
        """
        if bs > self.seq_blocks.numel():
            raise ValueError(f"batch {bs} exceeds {self.seq_blocks.numel()} decode rows")
        bt = self.layout.block_tokens
        seq_blocks = self.seq_blocks[:bs]
        seq_blocks.copy_((seq_lens[:bs] + bt - 1) // bt)
        slot = self._ring_slots(seq_blocks.to(torch.int64) - 1)
        newest_dev = self.newest_dev[:bs]
        newest_dev.copy_(
            self.req_to_device_buffer[req_pool_indices[:bs].to(torch.int64), slot]
        )
        return QSAHiSparseDecodeMeta(seq_blocks, newest_dev, self.rows[:bs])

    # ------------------------------------------------------------ speculative verify (M5 h16)

    def _ring_slots(self, blocks: torch.Tensor) -> torch.Tensor:
        """layout.ring_slot on a device tensor of block ids (int64), no host sync."""
        hot = self.layout.hot_blocks
        return torch.where(blocks < hot, blocks, hot + blocks % self.ring_width)

    def ring_blocks(self, req_pool_indices: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """Device blocks of the slab slots blocks[i] of request req_pool_indices[i] live in
        while a verify window writes them: what h17's allocation writes into
        mapping[logical block] for a spec request's new blocks (int64, no host sync)."""
        if req_pool_indices.shape != blocks.shape:
            raise ValueError(
                f"ring_blocks: {tuple(req_pool_indices.shape)} request slots for "
                f"{tuple(blocks.shape)} blocks"
            )
        return self.req_to_device_buffer[
            req_pool_indices.to(torch.int64), self._ring_slots(blocks.to(torch.int64))
        ]

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        rows_per_req: int = 1,
    ) -> torch.Tensor:
        """Resolve one layer's top-k rows to slab blocks (upstream's entry; QSA has no
        shared-index prefetch). rows_per_req = K for a verify forward: every request
        owns K consecutive rows of all three inputs (h16)."""
        if not 1 <= rows_per_req <= self.verify_tokens:
            raise ValueError(
                f"{rows_per_req} rows per request, the coordinator verifies at most "
                f"{self.verify_tokens} (verify_tokens)"
            )
        return self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            rows_per_req=rows_per_req,
        )

    def check_verify_overflow(self, flag: Optional[torch.Tensor] = None) -> None:
        """Raise if a multi-row verify swap-in set the overflow flag (design §3.3).

        flag: a host copy of verify_overflow (the step result carries one, and the
        result processor checks it with raise_on_verify_overflow before the step's
        tokens leave); None reads the device flag, a host sync (eager and tests only)."""
        raise_on_verify_overflow(
            self.verify_overflow if flag is None else flag,
            f"B={self.device_buffer_size}, K={self.verify_tokens}, top-k={self.top_k}",
        )

    # ------------------------------------------------------------ speculative serving (M5 h17)

    def agree_spec_allocation(self, reqs, current, following, needed):
        if not self.distributed_spec:
            return
        from sglang.srt.mem_cache.qsa_hisparse_distributed import spec_phase_packed_enabled, spec_phase_allocation_packable, spec_packed_allocation_identity, agree_spec_transition
        if spec_phase_packed_enabled() and spec_phase_allocation_packable(reqs, current, following, needed):
            identity, invalid = spec_packed_allocation_identity(self, reqs, current, following, needed)
            agree_spec_transition(self.spec_protocols["allocation"], "spec-allocate", identity, invalid=invalid)
            return
        from sglang.srt.mem_cache.qsa_hisparse_distributed import agree_spec_transition, spec_tensor_identity
        logical, host = self.token_to_kv_pool_allocator.logical_attn_allocator, self.mem_pool_host
        rows = [req.kv.req_pool_idx for req in reqs]
        invalid = not len(reqs) == len(current) == len(following) or needed < 0
        tails, grow, pages = [], [], 0
        if not invalid:
            for row, a, n in zip(rows, current, following):
                if row is None or not 1 <= row < self.req_to_host_pool.shape[0] or not 0 <= a <= n <= self.req_to_token_pool.req_to_token.shape[1]:
                    invalid = True
                    continue
                tails.append(-1 if a == 0 else int(self.req_to_token_pool.req_to_token[row, a - 1].cpu()))
                pages += -(-n // logical.page_size) - -(-a // logical.page_size)
                have = int(self.req_to_host_pool_allocated_len[row])
                target = -(-self.layout.seq_blocks(n) // self.page_size) * self.page_size
                grow.append([row, have, target])
                invalid |= target > self.req_to_host_pool.shape[1]
            invalid |= needed != sum(n - a for a, n in zip(current, following))
        host_needed = sum(max(0, target - have) for _, have, target in grow)
        invalid |= pages * logical.page_size > logical.available_size() or host_needed > host.available_size()
        identity = [[req.rid for req in reqs], rows, current, following, int(needed), tails, grow,
            [spec_tensor_identity(logical.free_pages), [spec_tensor_identity(x) for x in logical.staged_pages]],
            [spec_tensor_identity(host.free_slots), [spec_tensor_identity(x) for x in host.release_slots]],
            logical.available_size(), host.available_size()]
        agree_spec_transition(self.spec_protocols["allocation"], "spec-allocate", identity,
            invalid=invalid)

    def agree_spec_allocation_result(self, locations):
        if not self.distributed_spec:
            return
        from sglang.srt.mem_cache.qsa_hisparse_distributed import agree_spec_transition
        agree_spec_transition(self.spec_protocols["allocation"], "spec-allocated",
            None if locations is None else locations.detach().cpu().tolist(), invalid=locations is None)

    def agree_spec_verify_input(self, batch, verify_input):
        if not self.distributed_spec or batch.forward_mode.is_idle():
            return
        from sglang.srt.mem_cache.qsa_hisparse_distributed import spec_phase_packed_enabled, spec_packed_verify_identity, agree_spec_transition
        if spec_phase_packed_enabled():
            identity = spec_packed_verify_identity(batch, verify_input)
            agree_spec_transition(self.spec_protocols["forward"], "spec-verify-input", identity,
                invalid=verify_input.draft_token_num != self.verify_tokens or verify_input.topk != 1)
            return
        from sglang.srt.mem_cache.qsa_hisparse_distributed import agree_spec_transition, spec_tensor_identity
        # QSA does not read the FULL_MASK prefix/padding; it may be uninitialized.
        # Agree the active tokens, positions and complete initialized tree tables.
        identity = [[req.rid for req in batch.reqs], batch.req_pool_indices.detach().cpu().tolist(),
                    batch.seq_lens.detach().cpu().tolist(),
                    [spec_tensor_identity(getattr(verify_input, name)) for name in
                     ("draft_token", "positions", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling")]]
        agree_spec_transition(self.spec_protocols["forward"], "spec-verify-input", identity,
            invalid=verify_input.draft_token_num != self.verify_tokens or verify_input.topk != 1)

    def agree_spec_accept(self, batch, predict, accepted, indices):
        if not self.distributed_spec or batch.forward_mode.is_idle():
            return
        from sglang.srt.mem_cache.qsa_hisparse_distributed import spec_phase_packed_enabled, spec_packed_accept_identity, agree_spec_transition
        if spec_phase_packed_enabled():
            identity, invalid = spec_packed_accept_identity(self, batch, predict, accepted, indices)
            agree_spec_transition(self.spec_protocols["forward"], "spec-accept", identity, invalid=invalid)
            return
        rows = batch.req_pool_indices.detach().cpu().tolist()
        before = batch.seq_lens.detach().cpu().tolist()
        counts = accepted.detach().cpu().tolist()
        # Required completed-block destinations are checked before mamba commit
        # and before backup_after_accept can write any accepted host KV.
        addresses = []
        invalid = not len(rows) == len(before) == len(counts)
        for row, seq, count in zip(rows, before, counts):
            if not 1 <= row < self.req_to_host_pool.shape[0] or not 1 <= count <= self.verify_tokens:
                invalid = True
                continue
            lo, hi = int(seq) // self.layout.block_tokens, (int(seq) + int(count)) // self.layout.block_tokens
            invalid |= not 0 <= lo <= hi <= self.req_to_host_pool.shape[1]
            addresses.append(self.req_to_host_pool[row, lo:hi].detach().cpu().tolist())
        flag = int(self.verify_overflow.detach().cpu().reshape(-1)[0])
        identity = [rows, before, counts, indices.detach().cpu().tolist(),
                    predict.detach().cpu().tolist(), addresses, flag]
        from sglang.srt.mem_cache.qsa_hisparse_distributed import agree_spec_transition
        agree_spec_transition(self.spec_protocols["forward"], "spec-accept", identity,
            invalid=invalid or bool(flag) or any(not 1 <= n <= self.verify_tokens for n in counts)
                    or any(loc < 0 for row in addresses for loc in row))

    def agree_spec_result(self, reqs, accepted, tracked, overflow):
        if not self.distributed_spec:
            return
        counts = [] if accepted is None else accepted.tolist()
        claims = None if tracked is None else tracked.tolist()
        flag = 0 if overflow is None else int(overflow.reshape(-1)[0])
        identity = [[[req.rid, req.kv.req_pool_idx, req.kv.kv_committed_len] for req in reqs], counts, claims, flag]
        from sglang.srt.mem_cache.qsa_hisparse_distributed import agree_spec_transition
        agree_spec_transition(self.consensus, "spec-result", identity,
            invalid=bool(flag) or len(counts) != len(reqs)
                    or (bool(reqs) and (claims is None or len(claims) != len(reqs)))
                    or any(not 1 <= n <= self.verify_tokens for n in counts))

    def map_spec_window(
        self,
        req_pool_indices: Sequence[int],
        allocated: Sequence[int],
        new_allocated: Sequence[int],
    ) -> None:
        """A speculative allocation grew request req_pool_indices[i] from allocated[i]
        to new_allocated[i] tokens (logical slots already in req_to_token, design
        §2.1). Map the new blocks (layout.spec_map_blocks) to their ring slots, the
        rule ring_slot(b) that does not depend on the accept length, and give the
        request host pages up to the new logical pages, so a backup after accept
        needs no allocation (host pages never outnumber logical pages, M1b §5.1).
        CPU lists in, device writes on the current stream, no host sync."""
        if self.verify_tokens <= 1:
            raise RuntimeError("map_spec_window needs a coordinator with verify_tokens > 1")
        rows = [int(r) for r in req_pool_indices]
        allocated = [int(a) for a in allocated]
        new_allocated = [int(n) for n in new_allocated]
        if not len(rows) == len(allocated) == len(new_allocated):
            raise ValueError("map_spec_window: lists of different lengths")
        if self.distributed_spec:
            self.spec_protocols["allocation"].agree("spec-map-window", [rows, allocated, new_allocated])
        self._map_spec_blocks(rows, allocated, new_allocated)
        self._grow_spec_host_rows(rows, new_allocated)

    def _map_spec_blocks(self, rows: List[int], allocated: List[int], new_allocated: List[int]) -> None:
        bt = self.layout.block_tokens
        idx_r: List[int] = []
        idx_b: List[int] = []
        for r, a, n in zip(rows, allocated, new_allocated):
            span = self.layout.spec_map_blocks(a, n)
            idx_r.extend([r] * len(span))
            idx_b.extend(span)
        if not idx_b:
            return
        idx = torch.tensor([idx_r, idx_b], dtype=torch.int64).pin_memory()
        idx = idx.to(self.device, non_blocking=True)
        r_dev, b_dev = idx[0], idx[1]
        logical = self.req_to_token_pool.req_to_token[r_dev, b_dev * bt].to(torch.int64) // bt
        self.mem_pool_device.mapping[logical] = self.req_to_device_buffer[
            r_dev, self._ring_slots(b_dev)
        ]

    def _grow_spec_host_rows(self, rows: List[int], new_allocated: List[int]) -> None:
        page = self.page_size
        grow: List[Tuple[int, int, int]] = []
        for r, n in zip(rows, new_allocated):
            need = -(-self.layout.seq_blocks(n) // page) * page  # host blocks, whole pages
            if need > self.req_to_host_pool.shape[1]:
                raise RuntimeError(
                    f"speculative allocation of {n} tokens for request slot {r} needs "
                    f"{need} host blocks, the row holds {self.req_to_host_pool.shape[1]}"
                )
            have = int(self.req_to_host_pool_allocated_len[r])
            if need > have:
                grow.append((r, have, need))
        if not grow:
            return
        total = sum(need - have for _, have, need in grow)
        host_blocks = self.mem_pool_host.alloc(total)
        if host_blocks is None:
            raise RuntimeError(
                f"QSA HiSparse host pool is out of blocks for a speculative window "
                f"({total} blocks for {len(grow)} requests)"
            )
        h_rows = torch.cat([torch.full((need - have,), r, dtype=torch.int64) for r, have, need in grow])
        h_cols = torch.cat([torch.arange(have, need, dtype=torch.int64) for _, have, need in grow])
        triple = torch.stack([h_rows, h_cols, host_blocks.to(torch.int64)]).pin_memory()
        triple = triple.to(self.device, non_blocking=True)
        self.req_to_host_pool[triple[0], triple[1]] = triple[2]
        for r, _, need in grow:
            self.req_to_host_pool_allocated_len[r] = need

    def backup_after_accept(
        self,
        req_pool_indices: torch.Tensor,
        committed: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        """After a verify at committed[i] tokens accepted accept_lens[i] (bonus
        included): copy blocks [s // bt, (s + a) // bt) (layout.verify_backup) from
        their ring slots to the host records map_spec_window gave them, and advance
        host_done (design §5.2). Runs on the current (forward) stream after the
        verify and before the next verify, which may overwrite those ring slots or
        miss on those blocks. A fixed-shape list, no host sync; an entry without a
        completed block goes to the dump page. A completed block without a host
        record sets bit 2 of verify_overflow (raised with the step result)."""
        bs = req_pool_indices.shape[0]
        if bs == 0:
            return
        if committed.shape[0] != bs or accept_lens.shape[0] != bs:
            raise ValueError(
                f"backup_after_accept: {bs} requests, {committed.shape[0]} lengths, "
                f"{accept_lens.shape[0]} accept lengths"
            )
        bt = self.layout.block_tokens
        rpi = req_pool_indices.to(torch.int64)
        s = committed.to(torch.int64)
        lo = s // bt
        hi = (s + accept_lens.to(torch.int64)) // bt
        if self._debug_host_records and not torch.cuda.is_current_stream_capturing():
            done = self.host_done[rpi].to(torch.int64)
            if not torch.equal(done, lo):
                i = int((done != lo).nonzero()[0])
                raise RuntimeError(
                    f"QSA HiSparse verify backup: request slot {int(rpi[i])} committed "
                    f"{int(s[i])} but {int(done[i])} blocks on host (want {int(lo[i])})"
                )
        blocks = lo[:, None] + self._backup_offsets[None, :]
        valid = blocks < hi[:, None]
        blocks = blocks.clamp(max=self.req_to_host_pool.shape[1] - 1)
        dev = self.req_to_device_buffer[rpi[:, None], self._ring_slots(blocks)]
        host = self.req_to_host_pool[rpi[:, None], blocks]
        lost = valid & (host < 0)
        self.verify_overflow.bitwise_or_(lost.any().to(torch.int32) * 2)
        use = valid & ~lost
        dev = torch.where(use, dev, torch.zeros_like(dev))
        host = torch.where(use, host, torch.full_like(host, self._backup_dump_host))
        self.mem_pool_host.backup_blocks(dev.reshape(-1), host.reshape(-1))
        self.host_done[rpi] = hi.to(torch.int32)

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
        rows_per_req: int = 1,
    ) -> torch.Tensor:
        """Resolve one layer's top-k blocks to slab blocks, loading misses from host."""
        if record_plan:
            raise NotImplementedError("QSA HiSparse has no shared-index prefetch")
        if self._debug_host_records and not torch.cuda.is_current_stream_capturing():
            self._check_host_records(
                req_pool_indices, compressed_seq_lens, top_k_result, rows_per_req
            )
        top_k_device_locs = self.top_k_device_locs_buffer[: top_k_result.size(0)]
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
            rows_per_req=rows_per_req,
            ring_width=self.ring_width,
            overflow_flag=self.verify_overflow,
        )
        return top_k_device_locs

    def _check_host_records(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        rows_per_req: int = 1,
    ) -> None:
        """ARC_HISPARSE_QSA_DEBUG: every block the swap-in kernel may copy from host has a record.

        On a slow-path row (seq_blocks > B) any selected block below the request's
        ring floor (its first row's seq_blocks - 1: the newest block of a decode
        row, floor(committed / 4) of a verify, h16) can miss, and a miss copies from
        req_to_host_pool[r, block]. A -1 there is a broken lifecycle rule (staging
        or a backup skipped); the kernel would read before the host pool, an
        illegal memory access that poisons the CUDA context. Host sync, so eager only.
        """
        n = int(self.num_real_reqs) * rows_per_req
        rpi = req_pool_indices[:n].to(torch.int64)
        nb = compressed_seq_lens[:n].to(torch.int64)
        top_k = top_k_result[:n].to(torch.int64)
        slow = nb > self.device_buffer_size
        ring_lo = nb.view(-1, rows_per_req)[:, :1].expand(-1, rows_per_req).reshape(-1) - 1
        may_miss = slow[:, None] & (top_k >= 0) & (top_k < ring_lo[:, None])
        host = self.req_to_host_pool[rpi[:, None], top_k.clamp(min=0)]
        lost = may_miss & (host < 0)
        if self.verify_tokens > 1:
            # h17: speculative host pages are allocated ahead with the logical pages,
            # so an entry >= 0 is not enough; the block must be backed up
            lost |= may_miss & (top_k >= self.host_done[rpi].to(torch.int64)[:, None])
        if bool(lost.any()):
            i, j = lost.nonzero()[0].tolist()
            raise RuntimeError(
                f"QSA HiSparse swap-in: request slot {int(rpi[i])} (seq_blocks "
                f"{int(nb[i])}) selected block {int(top_k[i, j])}, which can miss but "
                f"has no host record ({int(lost.sum())} such entries in this batch)"
            )

    # ------------------------------------------------------------ release

    def request_finished(self, req: Req):
        self.consensus.agree("request-finish", [req.rid, req.kv.req_pool_idx])
        r = req.kv.req_pool_idx
        if req.hisparse_staging or r in self._staging:
            raise RuntimeError(
                f"{req.rid} is staging; release it with abort_staging_request"
            )
        # Release only after a potentially overlapped batch has run.
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        self.consensus.agree("request-release", [req.rid, r])

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
        mapping.index_fill_(0, logical_blocks.long(), 0)
        if not self.session_resume:
            self.release_host_row(r, 0)
        self.req_device_buffer_tokens[:, r, :] = -1
        self.lru_slots[:, r, :].copy_(self._lru_init)
        self._skip_first_backup[r] = False
        self.host_done[r].fill_(0)

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

    def begin_host_prefix_fill(
        self, segments: Sequence[Tuple[int, int, int]]
    ) -> "QSAHostPrefixFill":
        """Start one extend forward's host-prefix fills (h15; design §4.3).

        segments: (req_pool_idx, P, first context row) per request with a host
        prefix of P tokens. Call once per forward, on the forward stream, before
        its first QSA layer; each layer then takes its part with
        QSAHostPrefixFill.consume, in layer order. The fills of the previous
        forward must all have been consumed (the forward stream has waited on
        every one of them, which is what orders the host frees after it)."""
        if self.landing_zone is None:
            raise RuntimeError("begin_host_prefix_fill needs session_resume")
        prev = self._host_fill
        if prev is not None and not prev.complete:
            raise RuntimeError(
                f"the previous extend forward consumed {prev.consumed_jobs} of its "
                f"{len(prev.jobs)} landing-zone fills"
            )
        self._host_fill = QSAHostPrefixFill(self, segments)
        return self._host_fill

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
        held = [] if parked_host is None else [parked_host.to("cpu", torch.int64)]
        if self._backup_dump_pages is not None:  # h17: the verify backup's dump page
            held.append(self._backup_dump_pages.to("cpu", torch.int64))
        parked = (
            torch.sort(torch.cat(held)).values if held else torch.empty(0, dtype=torch.int64)
        )
        if host.available_size() != host.size - parked.numel() or not torch.equal(used, parked):
            problems.append(
                f"host {host.available_size()} / {host.size} free, "
                f"{used.numel()} slots marked used, {parked.numel()} parked"
            )
        if any(self._host_prefix_blocks):
            problems.append("adopted host prefixes left on idle rows")
        if self._host_fill is not None and not self._host_fill.complete:
            problems.append(
                f"landing-zone fills left unconsumed ({self._host_fill.consumed_jobs} "
                f"of {len(self._host_fill.jobs)})"
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
        if bool(self.host_done.any()):
            problems.append("host_done not cleared")
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
        if self.context_fill_stream is not None:
            # fills read the host pool that super().destroy() unregisters
            self.context_fill_stream.synchronize()
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


class QSAHostPrefixFill:
    """One extend forward's host-prefix -> landing-zone fills (arc-forge h15).

    h07 filled the zone on the forward stream right before each QSA layer read
    it, so the H2D (about 0.15 ms per 1k prefix tokens per forward) never
    overlapped compute. Here the fills run on the coordinator's
    context_fill_stream, one job ahead: a job is (layer index, group), a group
    is the requests packed into one zone fill (h07's rule, in segment order),
    and the jobs run in (layer, group) order. With one zone:

      fill j      side stream: waits consumed[j-1] (the fork event for j = 0),
                  host records -> zone, then records filled[j]
      consume j   forward stream: waits filled[j], reads the zone into the
                  context (the backend's callback), records consumed[j], and
                  only then enqueues fill j+1

    So fill j+1 never overwrites the zone before the forward stream has read job
    j, and a layer never reads a zone its fill has not finished; fill j+1 runs
    while the forward stream computes the rest of layer i and the layers up to
    the next QSA layer. Every fill is waited on by the forward stream within
    the forward (the last consume waits the last fill), so all side-stream work
    is ordered before the forward stream's later work: release_host_row, which
    frees host blocks after waiting on the forward stream, and the next
    forward, whose fork event orders its first fill after this forward's last
    zone read. The host block lists are built once, on the forward stream
    before the fork, and marked used by the side stream.
    """

    def __init__(self, coord: QSAHiSparseCoordinator, segments: Sequence[Tuple[int, int, int]]):
        zone = coord.landing_zone
        bt = coord.layout.block_tokens
        self._coord, self._zone, self._stream = coord, zone, coord.context_fill_stream
        # groups[g] = [(req_pool_idx, blocks, first context row, first zone block)]
        self.groups: List[List[Tuple[int, int, int, int]]] = []
        group: List[Tuple[int, int, int, int]] = []
        used = 0
        for r, prefix, start in segments:
            n = coord._host_prefix_blocks[r]
            if prefix <= 0 or prefix != n * bt:
                raise RuntimeError(
                    f"row {r}: host segment of {prefix} tokens, adopted prefix {n * bt}"
                )
            if n > zone.capacity_blocks:
                raise RuntimeError(f"row {r}: {n} host blocks exceed the landing zone")
            if used + n > zone.capacity_blocks:
                self.groups.append(group)
                group, used = [], 0
            group.append((r, n, start, used))
            used += n
        if not group:
            raise RuntimeError("a host-prefix fill needs at least one segment")
        self.groups.append(group)
        self._host = []
        for g in self.groups:
            host = torch.cat([coord.req_to_host_pool[r, :n] for r, n, _, _ in g])
            host.record_stream(self._stream)
            self._host.append(host)
        layers = coord.mem_pool_device.layer_num
        self.jobs = [(li, g) for li in range(layers) for g in range(len(self.groups))]
        self._filled = [device_module.Event() for _ in self.jobs]
        self._consumed = [device_module.Event() for _ in self.jobs]
        self.consumed_jobs = 0
        self._issued = 0
        self._fork = device_module.Event()
        self._fork.record()
        self._issue()

    @property
    def complete(self) -> bool:
        return self.consumed_jobs == len(self.jobs)

    def _issue(self) -> None:
        j = self._issued
        layer_index, g = self.jobs[j]
        stream = self._stream
        with device_module.stream(stream):
            stream.wait_event(self._fork if j == 0 else self._consumed[j - 1])
            self._coord.mem_pool_host.fill_context_host(layer_index, self._host[g], self._zone)
            self._filled[j].record(stream)
        self._issued += 1

    def consume(
        self, layer_index: int, read_group: Callable[[QSAContextLandingZone, int], None]
    ) -> None:
        """This layer's groups, in order: wait for the fill, read_group(zone, g)
        on the current (forward) stream, then start the next fill."""
        stream = device_module.current_stream()
        for g in range(len(self.groups)):
            j = self.consumed_jobs
            if j >= len(self.jobs) or self.jobs[j] != (layer_index, g):
                raise RuntimeError(
                    f"landing-zone fill order: layer {layer_index} group {g} asked for, "
                    f"next job is {self.jobs[j] if j < len(self.jobs) else 'none'}"
                )
            if self._issued != j + 1:
                raise RuntimeError(f"fill job {j} consumed with {self._issued} issued")
            stream.wait_event(self._filled[j])
            read_group(self._zone, g)
            self._consumed[j].record(stream)
            self.consumed_jobs += 1
            if self.consumed_jobs < len(self.jobs):
                self._issue()


def create_qsa_hisparse_coordinator(model_runner) -> QSAHiSparseCoordinator:
    """model_runner hook (h03): build the coordinator for a QSA model under --enable-hisparse."""
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.runtime_context import (
        get_exec,
        get_memory,
        get_spec,
        max_speculative_num_draft_tokens,
    )

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
    verify_tokens = 1
    if get_spec().speculative_algorithm is not None:
        verify_tokens = int(max_speculative_num_draft_tokens() or 1)
    snapshot = _snapshot_config(user, num_reqs, verify_tokens)
    if session_resume:
        # M1b h09: register the session cache backend (build_kv_cache comes next)
        # and check the host budget of what the records park before pinning it.
        import sglang.srt.mem_cache.qsa_hisparse_session_cache  # noqa: F401

        if snapshot is None:
            raise ValueError(
                "hisparse_config.qsa_session_resume needs qsa_snapshot_interval (h08): "
                "a session resumes from its latest GDN snapshot"
            )
        from sglang.srt.mem_cache.qsa_hisparse_distributed import distributed_hisparse_enabled
        if not distributed_hisparse_enabled():
            _check_session_host_budget(
                pool, user, snapshot, model_runner.req_to_token_pool.mamba_pool
            )
    elif snapshot is not None and snapshot.anchor:
        raise ValueError(
            "hisparse_config.qsa_session_anchor (h23) needs qsa_session_resume: "
            "the anchors are session records' buffers"
        )
    # M5 h16: a TARGET_VERIFY forward has K rows per request (the decode graph
    # runner captures TARGET_VERIFY with bs * K tokens), so the per-row buffers hold
    # K x the graph's max batch size.
    if verify_tokens > 1:
        # h17 (design §3.3): B >= K x top-k makes the swap-in overflow impossible;
        # below it only the flag stands between a large selection and a raise.
        small_ok = user.get("qsa_spec_small_hot_ok", False)
        if not isinstance(small_ok, bool):
            raise ValueError(f"hisparse_config.qsa_spec_small_hot_ok must be a bool, got {small_ok!r}")
        hot = pool.layout.hot_blocks
        if hot < verify_tokens * top_k and not small_ok:
            raise ValueError(
                f"QSA HiSparse speculative verify with B={hot} hot blocks < K x top-k = "
                f"{verify_tokens} x {top_k}: a verify step may need more hot blocks than B "
                "and would stop the server (overflow flag). Raise B (memory, R, L) or set "
                "hisparse_config.qsa_spec_small_hot_ok=true to accept that risk"
            )
        if hot < verify_tokens * top_k:
            logger.warning(
                "QSA HiSparse speculative verify with B=%d < K x top-k = %d: relying on "
                "the overflow flag (qsa_spec_small_hot_ok)", hot, verify_tokens * top_k
            )
    from sglang.srt.mem_cache.qsa_hisparse_distributed import distributed_hisparse_enabled
    host_admission = None
    if distributed_hisparse_enabled():
        host_admission = _check_distributed_host_budget(
            model_runner, pool, user, snapshot, session_resume, verify_tokens
        )
    return QSAHiSparseCoordinator(
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        top_k=top_k,
        device=model_runner.device,
        tp_group=get_attn_tp_group().cpu_group,
        swap_in_block_size=parse_hisparse_config().swap_in_block_size,
        max_decode_rows=int(decode_graph.max_bs or 0) * verify_tokens,
        session_resume=session_resume,
        snapshot=snapshot,
        verify_tokens=verify_tokens,
        host_admission=host_admission,
    )


def raise_on_verify_overflow(flag: torch.Tensor, detail: str = "") -> None:
    """arc-forge M5 h16: raise when the QSA HiSparse verify swap-in flagged that a
    row evicted a block an earlier row of its request selected, so that step's
    attention read the wrong KV (B < K x top-k was not enough for the selection;
    the coordinator's startup log line gives B and K). Called with the step
    result's host copy of the flag, before the step's tokens are sent."""
    bits = int(flag.reshape(-1)[0])
    if bits & 1:
        raise RuntimeError(
            "QSA HiSparse verify swap-in overflow: a verify row evicted a block an "
            "earlier row of its request selected, so this step's tokens were computed "
            f"from wrong KV and are not sent ({detail or 'see the coordinator log for B and K'})"
        )
    if bits & 2:
        raise RuntimeError(
            "QSA HiSparse verify backup: an accepted, completed block had no host "
            "record (a speculative allocation skipped map_spec_window), so it could "
            f"not be backed up; this step's tokens are not sent ({detail})"
        )
    if bits:
        raise RuntimeError(f"QSA HiSparse verify flag has unknown bits {bits:#x}")


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


def _check_distributed_host_budget(model_runner, pool, user, snapshot, session_resume, verify_tokens):
    """Price every rank before any HiSparse pin; PLE/model RAM is already loaded."""
    from sglang.srt.mem_cache.qsa_hisparse_distributed import (
        host_admission_totals, host_memory_state,
    )
    from sglang.srt.mem_cache.pool_host.base import HICACHE_HOST_MEMORY_RESERVE_BYTES

    reserve = user.get("host_reserve_gib")
    if isinstance(reserve, bool) or not isinstance(reserve, (float, int)) or reserve < 0:
        raise ValueError("distributed HiSparse needs explicit hisparse_config.host_reserve_gib >= 0")
    group = model_runner.tp_group.cpu_group
    if torch.distributed.get_world_size(group=group) != 4:
        raise ValueError("h25 host admission requires all four model-TP ranks on one node")
    layout = pool.layout
    blocks = (
        session_host_blocks(layout) if session_resume
        else layout.host_blocks(model_runner.req_to_token_pool.max_context_len)
    ) + (layout.page_blocks if verify_tokens > 1 else 0)
    kv = blocks * pool.layer_num * pool.plane_spec.record_bytes
    gdn = (
        0 if snapshot is None else snapshot.host_buffers
        * model_runner.req_to_token_pool.mamba_pool.host_slot_nbytes()
    )
    import os
    flag = os.environ.get("ARC_NATIVE_MTP_KV_HOST", "0")
    if flag not in ("0", "1"):
        raise ValueError("ARC_NATIVE_MTP_KV_HOST must be 0 or 1")
    host_draft = flag == "1"
    draft_host = 0
    if host_draft:
        from sglang.srt.mem_cache.qsa_kv_pool import native_mtp_kv_host_enabled, native_mtp_host_kv_bytes
        native_mtp_kv_host_enabled()
        draft_host = native_mtp_host_kv_bytes(layout.logical_tokens, pool.page_size,
            pool.full_kv_pool.head_num, pool.full_kv_pool.head_dim, pool.full_kv_pool.v_head_dim)
    memory = host_memory_state()
    values = [kv + draft_host, gdn, max(int(reserve * (1 << 30)), HICACHE_HOST_MEMORY_RESERVE_BYTES),
              memory["available_bytes"]]
    if host_draft:
        values.append(draft_host)
    record = torch.tensor(values, dtype=torch.int64)
    peers = [torch.empty_like(record) for _ in range(4)]
    torch.distributed.all_gather(peers, record, group=group)
    total = host_admission_totals([peer.tolist()[:4] for peer in peers])
    draft_host_all_ranks = sum(int(peer[4]) for peer in peers) if host_draft else 0
    from sglang.srt.layers.dp_attention import get_attention_dp_rank

    dp_rank = get_attention_dp_rank()
    ranks = torch.distributed.get_process_group_ranks(get_attn_tp_group().cpu_group)
    receipt = dict(
        global_rank=torch.distributed.get_rank(), attn_tp_ranks=ranks,
        attn_tp_leader_global_rank=ranks[0], attn_dp_rank=dp_rank,
        namespace=f"attention-dp-{dp_rank}", max_running_requests_per_dp=layout.num_slabs,
        logical_tokens_per_dp=layout.logical_tokens, hot_blocks=layout.hot_blocks,
        verify_tokens=verify_tokens,
        spec_dummy_host_bytes_per_rank=(layout.page_blocks * pool.layer_num * pool.plane_spec.record_bytes if verify_tokens > 1 else 0),
        kv_host_bytes_per_rank=kv, snapshot_host_bytes_per_rank=gdn,
        kv_host_bytes_all_ranks=total[0] - draft_host_all_ranks, snapshot_host_bytes_all_ranks=total[1],
        host_reserve_bytes=total[2], host_available_bytes_before_pin=total[3],
        cgroup_limit_bytes=memory["cgroup_limit_bytes"],
        cgroup_current_bytes=memory["cgroup_current_bytes"],
        proc_available_bytes=memory["proc_available_bytes"],
    )
    if host_draft:
        receipt.update(draft_host_bytes_per_rank=draft_host, draft_host_bytes_all_ranks=draft_host_all_ranks,
                       planned_pinned_bytes_all_ranks=total[0] + total[1])
    logger.warning("ARC_HISPARSE_STARTUP=%s", json.dumps(receipt, sort_keys=True))
    return kv, gdn



def _snapshot_config(
    user: dict, num_reqs: int, verify_tokens: int = 1
) -> Optional["QSAHiSparseSnapshotConfig"]:
    """M1b h08 keys of hisparse_config: decode GDN snapshots are off unless
    qsa_snapshot_interval (T) is set; K staging slots default to 8 and the host
    buffers to R + 16 (design §3.3, §5.3). Under speculative decoding a staging
    slot stays claimed until the CPU sees the step and its D2H ends, about two
    steps: 16 by default (M5 design §7.2, §16 decision 2). h23:
    qsa_session_anchor (default off) keeps each record's resume point as a second
    restore point; a record then holds up to two buffers, so the host buffer default
    doubles to 2 (R + 16)."""
    keys = (
        "qsa_snapshot_interval",
        "qsa_snapshot_staging_slots",
        "qsa_snapshot_host_buffers",
    )
    for key in keys:
        value = user.get(key)
        if key in user and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"hisparse_config.{key} must be an integer, got {value!r}")
    anchor = user.get("qsa_session_anchor", False)
    if not isinstance(anchor, bool):
        raise ValueError(f"hisparse_config.qsa_session_anchor must be a bool, got {anchor!r}")
    if keys[0] not in user:
        if any(key in user for key in keys[1:]) or anchor:
            raise ValueError(f"hisparse_config.{keys[1]} / {keys[2]} / qsa_session_anchor need {keys[0]}")
        return None
    return QSAHiSparseSnapshotConfig(
        interval=user[keys[0]],
        staging_slots=user.get(keys[1], 16 if verify_tokens > 1 else 8),
        host_buffers=user.get(keys[2], (2 if anchor else 1) * (num_reqs + 16)),
        anchor=anchor,
    )


# ---------------------------------------------------------------- GDN snapshots (M1b h08)


class QSAHiSparseSnapshotConfig(NamedTuple):
    interval: int  # T: snapshot when a decode input position p is a positive multiple of T
    staging_slots: int  # K mamba slots reserved for the D2D, for good
    host_buffers: int  # pinned host buffers of one mamba slot each (S_max)
    anchor: bool = False  # h23: records keep their resume point as a second buffer


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
    after: Optional[torch.cuda.Event] = None  # spec (h18): the claiming verify's copy_done


class QSAHiSparseSnapshots:
    """Decode GDN / PLE snapshots (M1b design §3.2, §11.6-11.8).

    In the decode backup window of the step whose input is position p (a
    multiple of T), the live mamba slot is D2D'd into a staging slot (same event
    as the KV backup, so the next forward cannot overwrite it first); a separate
    stream D2Hs the staging slot into the owner's pinned buffer; poll() (from
    collect_ready_reqs) publishes each snapshot once its D2H event is seen done.
    A buffer with an unpublished snapshot takes no new one (§11.8, counted).

    Under speculative decoding (M5 design §7.2, h18) there is no D2D: the
    device claims a free staging slot for every verify row whose accept crosses
    a multiple of T (claim, before the mamba commit), the commit tracks the
    state at that multiple into it, and the CPU sees the claims with the step
    result one step late (on_verify_result): it asks the owner for a buffer
    and D2Hs, or frees the slot. staging_busy on the device is the truth about
    which staging slots are taken; the CPU only clears entries it saw claimed.
    """

    def __init__(
        self, coord: QSAHiSparseCoordinator, config: QSAHiSparseSnapshotConfig
    ):
        self.coord = coord
        self.consensus = coord.consensus
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
        self.anchor = config.anchor  # h23: read by the session cache
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
        self.spec = coord.verify_tokens > 1
        if self.spec:
            # M5 h18: one boundary per verify at most, and the commit tracks exactly
            # the entries a host slot holds (conv[0], temporal, the PLE siblings)
            if self.interval <= coord.verify_tokens:
                raise ValueError(
                    f"snapshot interval T={self.interval} must exceed K={coord.verify_tokens}"
                )
            if len(self.mamba.mamba_cache.conv) != 1:
                raise NotImplementedError(
                    f"the verify commit tracks conv[0] only, the pool has "
                    f"{len(self.mamba.mamba_cache.conv)} conv states"
                )
            n = len(self.staging_slots)
            # entry n: where rows without a claim write (never cleared, never read)
            self.staging_busy = torch.zeros(n + 1, dtype=torch.int32, device=self.device)
            self.staging_ids = torch.tensor(
                self.staging_slots + [-1], dtype=torch.int64, device=self.device
            )
            self._staging_index = {slot: i for i, slot in enumerate(self.staging_slots)}
        else:
            # Compile the fused conv-slot copy now rather than in the first decode
            # window (slot 0 is the padding dummy; a staging slot is scratch until a D2D).
            self.mamba.copy_from(
                torch.zeros(1, dtype=torch.int64, device=self.device),
                torch.tensor(self.staging_slots[:1], dtype=torch.int64, device=self.device),
            )

        self.slot_nbytes = self.mamba.host_slot_nbytes()
        total = self.slot_nbytes * config.host_buffers
        available = (
            host_memory_budget_bytes() if coord.host_admission is None
            else coord.host_admission[1]
        )
        if coord.host_admission is not None and total != available:
            raise RuntimeError("GDN snapshot pool differs from collective host admission")
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
        # M5 h18: claims seen with a verify result, D2H not issued yet (poll issues them, or
        # release, which alone knows whether a finishing request's P is past its stop)
        self.unissued: List[_Snapshot] = []
        self.pending: List[_Snapshot] = []  # issue order == completion order (one stream)
        self.decoding: Dict[int, Req] = {}  # request slot -> request, ready to release
        self.host_blocks: Dict[int, int] = {}  # request slot -> host copies issued [0, n)
        self.d2h_stream = device_module.Stream()
        self.owner: Optional[SnapshotOwner] = None
        self.stats: Counter = Counter()
        self._destroyed = False
        logger.info(
            "QSA HiSparse GDN snapshots: T=%d, staging mamba slots %s, %d host buffers "
            "x %d B = %.2f GB pinned, resume anchors %s",
            self.interval,
            self.staging_slots,
            config.host_buffers,
            self.slot_nbytes,
            total / 1e9,
            "on" if self.anchor else "off",
        )

    # ------------------------------------------------------------ owner side

    def set_owner(self, owner: SnapshotOwner) -> None:
        if self.owner is not None:
            raise RuntimeError("the GDN snapshot owner is already set")
        self.owner = owner

    def alloc_buffer(self) -> Optional[int]:
        return self.buffer_free.pop() if self.buffer_free else None

    def free_buffer(self, buf: int) -> None:
        self.consensus.agree("snapshot-buffer-release", buf)
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
        self.consensus.agree("snapshot-plan", [
            [int(x) for x in seq_lens_cpu], [int(x) for x in req_pool_indices_cpu],
            self.staging_free, self.buffer_free, sorted(self.busy),
        ])
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
        self.consensus.agree("snapshot-issue", self._snapshot_keys(due))
        return due

    # ------------------------------------------------------------ speculative decode (M5 h18)

    def track_before_commit(self, batch, accept_lens: torch.Tensor) -> Optional[int]:
        """run_eagle_verify's before_mamba_commit: claim staging slots for the rows
        this verify's accept carries across a multiple of T and hand them to the
        commit as mamba_track_indices (-1: no track), with T as its interval."""
        if batch.forward_mode.is_idle():
            return None
        if batch.mamba_track_indices is not None:
            raise RuntimeError("QSA HiSparse snapshots: the verify batch already has track indices")
        batch.mamba_track_indices = self.claim(batch.seq_lens, accept_lens)
        return self.interval

    def claim(self, seq_lens: torch.Tensor, accept_lens: torch.Tensor) -> torch.Tensor:
        """Device side of design §7.2 step 2, on the current (forward) stream, no host
        sync. Row i crosses when seq_lens[i] // T != (seq_lens[i] + accept_lens[i]) // T;
        the k-th crossing row takes the k-th free staging slot, and a crossing row
        past the free count gets none (the CPU counts it as skip_staging). Returns
        int64 [bs]: the claimed mamba slot, or -1."""
        t = self.interval
        pre = seq_lens.to(torch.int64)
        cross = pre // t != (pre + accept_lens.to(torch.int64)) // t
        n = len(self.staging_slots)
        busy = self.staging_busy
        if self.coord.distributed_spec:
            from sglang.srt.mem_cache.qsa_hisparse_distributed import spec_staging_busy
            busy = spec_staging_busy(self.coord.spec_protocols["forward"], busy)
        free_rank = torch.cumsum((busy[:n] == 0).to(torch.int64), 0)
        k = torch.cumsum(cross.to(torch.int64), 0)  # 1-based rank among crossing rows
        ok = cross & (k <= free_rank[-1])
        idx = torch.searchsorted(free_rank, k)  # the k-th free entry
        idx = torch.where(ok, idx, torch.full_like(idx, n))
        if self.coord.distributed_spec:
            self.coord.spec_protocols["forward"].agree("spec-snapshot-claim", [
                pre.detach().cpu().tolist(), accept_lens.detach().cpu().tolist(),
                busy.detach().cpu().tolist(), idx.detach().cpu().tolist(),
                self.staging_ids[idx].detach().cpu().tolist(),
            ])
        self.staging_busy.index_put_((idx,), torch.ones_like(idx, dtype=torch.int32))
        return self.staging_ids[idx]

    def on_verify_result(self, reqs, track_slots: torch.Tensor, accept_lens: torch.Tensor, after) -> None:
        """CPU side of design §7.2 step 4 for one verify's result, before its accept
        is committed on the host (req.kv.kv_committed_len is still the verify's
        seq_lens). track_slots / accept_lens: host copies from the result; after: its
        copy_done event (the commit's track and the backup after accept precede it).
        A claim of a request that finished, was retracted or left decode since the
        verify was launched (overlap) is discarded. The others wait unissued: whether
        this verify finishes the request is decided later in the same result, and a P
        past the stop must never reach the owner's buffer (a record's buffer would be
        overwritten with a state its P does not describe). poll issues them from the
        next iteration on; release issues or drops a finishing request's own."""
        slots, accepts = track_slots.tolist(), accept_lens.tolist()
        if not len(slots) == len(accepts) == len(reqs):
            raise RuntimeError(
                f"verify result with {len(slots)} track slots, {len(accepts)} accept lengths, "
                f"{len(reqs)} requests"
            )
        t, bt = self.interval, self.layout.block_tokens
        for req, slot, a in zip(reqs, slots, accepts):
            r = req.kv.req_pool_idx
            live = (
                not req.finished()
                and not req.is_retracted
                and r is not None
                and self.decoding.get(r) is req
            )
            if not live:
                if slot >= 0:
                    self.stats["claim_discarded"] += 1
                    self._free_staging(slot)
                continue
            pre = req.kv.kv_committed_len
            post = pre + a
            crossed = pre // t != post // t
            if slot < 0:
                if crossed:
                    self.stats["skip_staging"] += 1
                continue
            if not crossed or slot not in self._staging_index:
                raise RuntimeError(
                    f"{req.rid}: staging slot {slot} claimed for a verify {pre} -> {post} "
                    f"(T={t}, staging slots {self.staging_slots})"
                )
            self.stats["claimed"] += 1
            # the verify's backup after accept put blocks [0, post // bt) on host; buf is
            # filled in when the claim is issued
            self.unissued.append(_Snapshot(req, r, post // t * t, -1, slot, post // bt, after=after))

    def _issue(self, claims: List[_Snapshot]) -> None:
        """Ask the owner for each claim's buffer and D2H it, or free the staging slot."""
        due: List[_Snapshot] = []
        if claims and self.owner is None:
            raise RuntimeError("GDN snapshots are on but no owner is registered (h09)")
        for snap in claims:
            buf = self.owner.snapshot_buffer(snap.req, snap.p)
            if buf is None:
                self.stats["skip_owner"] += 1
                self._free_staging(snap.staging)
                continue
            if not 0 <= buf < len(self.buffers) or buf in self.buffer_free:
                raise RuntimeError(f"owner gave GDN buffer {buf}, which is not allocated")
            if buf in self.busy:
                self.stats["skip_busy"] += 1
                self._free_staging(snap.staging)
                continue
            snap.buf = buf
            self.busy[buf] = snap
            due.append(snap)
        # copy_done events of one forward stream, in order: the last covers the others
        if due:
            self.start_d2h(due, due[-1].after)

    def _free_staging(self, slot: int) -> None:
        self.consensus.agree("snapshot-staging-release", slot)
        if not self.spec:
            self.staging_free.append(slot)
            return
        # a device write on the scheduler stream: the next forward waits for it,
        # the one in flight may or may not see it (either is safe: it only ever
        # claims an entry it read as 0, and this entry reads 1 until now)
        i = self._staging_index[slot]
        self.staging_busy[i : i + 1].zero_()

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
        """Publish, in issue order, every snapshot whose D2H is seen done (spec: first
        issue the claims earlier results left, h18)."""
        if self.unissued:
            claims, self.unissued = self.unissued, []
            self._issue(claims)
        if self.consensus.world_size == 1:
            while self.pending and self.pending[0].done.query():
                self._publish(self.pending.pop(0), d2h_done=True)
            return
        count = 0
        for snap in self.pending:
            if not snap.done.query():
                break
            count += 1
        count = self.consensus.ready_count("snapshot-publish", self._snapshot_keys(self.pending), count)
        for _ in range(count):
            self._publish(self.pending.pop(0), d2h_done=True)

    @staticmethod
    def _snapshot_keys(snaps) -> list:
        return [[s.req.rid, s.r, s.p, s.buf, s.staging, s.host_blocks] for s in snaps]

    def _publish(self, snap: _Snapshot, *, d2h_done: bool) -> None:
        req, p = snap.req, snap.p
        committed = req.kv.kv_committed_len
        if p > committed:  # §11.7: condition 1 always holds without spec
            raise RuntimeError(f"{req.rid}: snapshot p={p} beyond kv_committed_len {committed}")
        if self.spec and req.finished() and p > self._emitted(req):
            # release drops these before they are issued (h18)
            raise RuntimeError(
                f"{req.rid}: snapshot p={p} past the stop ({self._emitted(req)} tokens emitted) was issued"
            )
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
        self._free_staging(snap.staging)
        self.stats["published"] += 1
        self.owner.on_snapshot_published(req, p, snap.buf)

    @staticmethod
    def _emitted(req: Req) -> int:
        return len(req.origin_input_ids) + len(req.output_ids_through_stop)

    def release(self, req: Req) -> None:
        """Wait for req's in-flight D2H, publish in order, drop its decode state.

        Spec (h18): req's unissued claims first. The finishing verify may commit tokens
        past the stop (EOS, stop string, max_new_tokens); a P among them is not a prefix
        of the next turn, so that claim is dropped before the owner sees it
        (skip_past_stop); the rest are issued now."""
        self.consensus.agree("snapshot-request-release", [req.rid, req.kv.req_pool_idx])
        claims = [snap for snap in self.unissued if snap.req is req]
        if claims:
            self.unissued = [snap for snap in self.unissued if snap.req is not req]
            keep = []
            for snap in claims:
                if req.finished() and snap.p > self._emitted(req):
                    self.stats["skip_past_stop"] += 1
                    self._free_staging(snap.staging)
                else:
                    keep.append(snap)
            self._issue(keep)
        mine = [snap for snap in self.pending if snap.req is req]
        for snap in mine:
            snap.done.synchronize()
        self.consensus.agree("snapshot-release-complete", self._snapshot_keys(mine))
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
        if self.pending or self.busy or self.decoding or self.unissued:
            problems.append(
                f"snapshots: {len(self.unissued)} claims unissued, {len(self.pending)} D2H pending, busy buffers "
                f"{sorted(self.busy)}, decoding slots {sorted(self.decoding)}"
            )
        if self.spec:
            busy = self.staging_busy[: len(self.staging_slots)].tolist()
            if any(busy):
                problems.append(f"staging slots claimed on the device: {busy}")
        elif sorted(self.staging_free) != sorted(self.staging_slots):
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
