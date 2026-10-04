"""Geometry and per-request rules of the QSA HiSparse offload (arc-forge M1a h02).

Pure Python on purpose: the pool, allocator and coordinator take every scalar
decision (slab numbering, staging copy sets, decode backups, newest slot) from
here, and arc-forge checks this file against its T0 rule model without torch.

A *block* is one QSA compression group of ``block_tokens`` consecutive
positions. HiSparse items, host records and hot-buffer slots are blocks; device
block d owns the token rows [d * block_tokens, (d + 1) * block_tokens).

Device layout in blocks (design §1.2):

    [0, page_blocks)             page 0, the dump that padding writes land in
    [page_blocks, hot_base)      prefill area, handed out page by page while a
                                 request is prefilling or staging
    [hot_base, device_blocks)    one static slab of H = B + page_blocks blocks per
                                 request slot r >= 1 (slot 0 is never allocated)

Slab slot i < B is an LRU slot, slot B holds the newest block once the request
is long (seq_blocks > B), slots B+1.. are the kernel's page padding.

Speculative verify (arc-forge M5 h16, design §4): a verify of K rows writes
positions committed .. committed + K - 1, which touch up to W = ring_width(K)
consecutive blocks that are not on host yet. Each block b >= B of them lives in
ring slot B + b % W (W <= page_blocks - 1, so the ring fits the padding slots);
a block b < B lives in its identity slot as always. W = 1 is the decode rule:
the newest block in slot B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# lru_slots / s_lru_slots_out / s_hash_vals are int16 in hisparse.cuh; the
# largest multiple of 16 below 2**15.
MAX_HOT_BLOCKS = 32752
HOT_BLOCKS_ALIGN = 16  # load_cache_to_device_buffer_planar requires hot % 16 == 0
MAX_PLANES = 4  # PlanarSpec<P0, P1, P2, P3>
TRANSFER_ALIGN = 128  # transfer_hicache_all_layer moves elements in 128 B units


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class PlaneSpec:
    """Planes of one host record in record order: ((name, bytes per block), ...).

    Planes come in K/V pairs of equal size because one transfer_hicache_all_layer
    call moves a K plane and a V plane with one element size (FP8: K|V; NVFP4:
    K|V payload, then K|V scale).
    """

    planes: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if not 1 <= len(self.planes) <= MAX_PLANES or len(self.planes) % 2:
            raise ValueError(f"need 2 or 4 planes in K/V pairs, got {self.planes}")
        names = [name for name, _ in self.planes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate plane names in {self.planes}")
        for name, nbytes in self.planes:
            if nbytes <= 0 or nbytes % TRANSFER_ALIGN:
                raise ValueError(
                    f"plane {name!r}: {nbytes} B per block is not a positive "
                    f"multiple of {TRANSFER_ALIGN}"
                )
        for q in range(0, len(self.planes), 2):
            if self.planes[q][1] != self.planes[q + 1][1]:
                raise ValueError(
                    f"planes {self.planes[q]} and {self.planes[q + 1]} form a K/V "
                    "pair and must have equal bytes"
                )

    @property
    def plane_bytes(self) -> tuple[int, ...]:
        return tuple(nbytes for _, nbytes in self.planes)

    @property
    def record_bytes(self) -> int:
        return sum(self.plane_bytes)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Byte offset of each plane inside a record."""
        out, acc = [], 0
        for nbytes in self.plane_bytes:
            out.append(acc)
            acc += nbytes
        return tuple(out)

    def pairs(self) -> tuple[tuple[int, int], ...]:
        """(index of the K plane, bytes per block) of each K/V pair."""
        return tuple((q, self.planes[q][1]) for q in range(0, len(self.planes), 2))


def kv_plane_spec(*, k_row_bytes: int, v_row_bytes: int, block_tokens: int) -> PlaneSpec:
    """Record of an unquantized (FP8 / BF16) token-major KV pool: [K | V]."""
    return PlaneSpec(
        (("k", k_row_bytes * block_tokens), ("v", v_row_bytes * block_tokens))
    )


# NVFP4 KV (M3): 2 heads x 256 dims x 4 tokens -> 1024 B payload and 128 B of
# per-16 scales per plane. The pool derives its spec from the NVFP4 buffers (h12);
# for REAP320 that is this one.
NVFP4_PLANES = PlaneSpec(
    (("k", 1024), ("v", 1024), ("k_scale", 128), ("v_scale", 128))
)


@dataclass(frozen=True)
class StagingPlan:
    """Copies made when a prompt of S tokens leaves prefill (design §2 staging row).

    D2H: prompt blocks [host_blocks, full_blocks) -> host (complete blocks only).
    Slab: prompt blocks [0, identity_blocks) -> slab slots [0, identity_blocks),
    plus newest_block -> slot B when a long prompt ends in a partial block (its
    ring slot ring_slot(newest_block, W) when the coordinator verifies K > 1 rows, h16).

    host_blocks = h > 0 when blocks [0, h) are a resumed prefix already on host
    with no device rows (arc-forge M1b h07, design §4.4): they are not copied to
    host again, and their identity slots are loaded from host (``slab_from_host``);
    every other copy reads the prefill area. With h = 0 this is the M1a plan.
    """

    host_blocks: int
    full_blocks: int
    identity_blocks: int
    newest_block: Optional[int]

    @property
    def d2h(self) -> range:
        return range(self.host_blocks, self.full_blocks)

    @property
    def slab_from_host(self) -> range:
        return range(0, min(self.identity_blocks, self.host_blocks))

    @property
    def slab_from_prefill(self) -> range:
        return range(self.host_blocks, self.identity_blocks)


@dataclass(frozen=True)
class QSAHiSparseLayout:
    block_tokens: int  # QSA compress ratio
    page_size: int  # KV page, tokens
    hot_blocks: int  # B
    num_slabs: int  # request slots owning a slab (R); slot 0 never does
    prefill_tokens: int  # prefill area (R_pf), tokens
    logical_tokens: int  # logical token capacity (L), what the scheduler sees

    def __post_init__(self):
        bt, ps = self.block_tokens, self.page_size
        if bt < 2 or ps <= 0 or ps % bt:
            raise ValueError(
                f"page_size {ps} must be a positive multiple of block_tokens {bt} >= 2"
            )
        if not (
            0 < self.hot_blocks <= MAX_HOT_BLOCKS
            and self.hot_blocks % HOT_BLOCKS_ALIGN == 0
        ):
            raise ValueError(
                f"hot_blocks {self.hot_blocks} must be a multiple of "
                f"{HOT_BLOCKS_ALIGN} in (0, {MAX_HOT_BLOCKS}]"
            )
        if self.num_slabs < 1:
            raise ValueError(f"num_slabs must be >= 1, got {self.num_slabs}")
        for name in ("prefill_tokens", "logical_tokens"):
            value = getattr(self, name)
            if value <= 0 or value % ps:
                raise ValueError(
                    f"{name} {value} must be a positive multiple of page_size {ps}"
                )

    # ------------------------------------------------------------ geometry

    @property
    def page_blocks(self) -> int:
        return self.page_size // self.block_tokens

    @property
    def slab_stride(self) -> int:
        """H = B + page_blocks slab slots per request."""
        return self.hot_blocks + self.page_blocks

    @property
    def prefill_blocks(self) -> int:
        return self.prefill_tokens // self.block_tokens

    @property
    def prefill_base(self) -> int:
        return self.page_blocks

    @property
    def hot_base(self) -> int:
        return self.page_blocks + self.prefill_blocks

    @property
    def device_blocks(self) -> int:
        """Device blocks including the dump page."""
        return self.hot_base + self.num_slabs * self.slab_stride

    @property
    def device_tokens(self) -> int:
        """Device KV rows excluding the dump page: the KV pool's ``size``."""
        return self.device_blocks * self.block_tokens - self.page_size

    @property
    def logical_blocks(self) -> int:
        """Logical blocks including the padding page (mapping length without its sentinel)."""
        return (self.logical_tokens + self.page_size) // self.block_tokens

    def slab_block(self, r: int, i: int) -> int:
        if not (1 <= r <= self.num_slabs and 0 <= i < self.slab_stride):
            raise ValueError(
                f"slab index (r={r}, i={i}) outside [1, {self.num_slabs}] x [0, {self.slab_stride})"
            )
        return self.hot_base + (r - 1) * self.slab_stride + i

    def host_blocks(self, max_context_len: int) -> int:
        """Worst-case host blocks: every slab owner holds a full context (design §5.2)."""
        return self.num_slabs * (
            _ceil_div(max_context_len, self.block_tokens) + self.page_blocks
        )

    # ------------------------------------------------------------ rules

    def seq_blocks(self, seq_len: int) -> int:
        """Blocks touched by seq_len tokens; what the swap-in kernel gets as seq_lens."""
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        return _ceil_div(seq_len, self.block_tokens)

    def newest_slot(self, seq_len: int) -> int:
        """s* = min(nb - 1, B): identity slot while short, slot B once long (design §1.7)."""
        return min(self.seq_blocks(seq_len) - 1, self.hot_blocks)

    # ------------------------------------------------------------ speculative verify (M5 h16)

    def ring_width(self, verify_tokens: int) -> int:
        """W: blocks one verify window of verify_tokens consecutive positions can
        touch, wherever it starts (ceil((K + bt - 1) / bt); K = 1 gives 1)."""
        if verify_tokens < 1:
            raise ValueError(f"verify_tokens must be >= 1, got {verify_tokens}")
        w = _ceil_div(verify_tokens + self.block_tokens - 1, self.block_tokens)
        if w > self.page_blocks - 1:
            raise ValueError(
                f"{verify_tokens} verify tokens need a ring of {w} slots; the slab "
                f"pads B with only {self.page_blocks - 1}"
            )
        return w

    def ring_slot(self, block: int, ring_width: int) -> int:
        """Slab slot of a block the current (or a later) verify window writes:
        its identity slot below B, ring slot B + block % W from B on. With
        ring_width 1 this is the decode rule min(block, B)."""
        if block < 0 or not 1 <= ring_width <= self.page_blocks - 1:
            raise ValueError(f"ring_slot(block={block}, ring_width={ring_width})")
        if block < self.hot_blocks:
            return block
        return self.hot_blocks + block % ring_width

    def ring_floor(self, committed: int) -> int:
        """First block not on host before a verify at `committed` tokens: blocks
        [0, committed // bt) are complete and backed up (h17's backup rule). The
        swap-in kernel reads it as row 0's seq_blocks - 1 (= committed // bt)."""
        if committed < 0:
            raise ValueError(f"committed must be >= 0, got {committed}")
        return committed // self.block_tokens

    def verify_backup(self, committed: int, accepted: int) -> range:
        """Blocks a verify at `committed` completes and h17 backs up to host once
        `accepted` (1 .. K) tokens are accepted: [committed // bt, (committed +
        accepted) // bt). Each still sits in ring_slot(block) until the next verify
        writes (design §5.2)."""
        if accepted < 1:
            raise ValueError(f"accepted must be >= 1, got {accepted}")
        bt = self.block_tokens
        return range(committed // bt, (committed + accepted) // bt)

    def staging_plan(self, prompt_len: int, host_blocks: int = 0) -> StagingPlan:
        """Staging copies of a prompt of prompt_len tokens whose blocks [0, host_blocks)
        are a resumed prefix on host (h07). The prefix is a whole number of pages
        below the prompt: resume points are published page-aligned (design §4.2)."""
        h = host_blocks
        if h < 0 or h % self.page_blocks or h * self.block_tokens >= prompt_len:
            raise ValueError(
                f"host prefix of {h} blocks must be a multiple of {self.page_blocks} "
                f"with {self.block_tokens} * h < prompt_len {prompt_len}"
            )
        nf, nb = prompt_len // self.block_tokens, self.seq_blocks(prompt_len)
        if self.seq_blocks(prompt_len + 1) <= self.hot_blocks:
            # The first decode step still takes the fast path: every block,
            # a partial tail included, sits in its identity slot.
            return StagingPlan(h, nf, nb, None)
        partial = prompt_len % self.block_tokens != 0
        return StagingPlan(h, nf, self.hot_blocks, nb - 1 if partial else None)

    def decode_backup(self, seq_len: int, skip_first: bool) -> Optional[tuple[int, int]]:
        """(block, slab slot) to copy to host before this decode step's forward, or None.

        seq_len counts the token this step appends. When that token opens block
        k, block k-1 was completed by the previous step; it still sits in slot
        min(k-1, B). The first step after staging skips: staging already copied
        every complete prompt block.
        """
        bt = self.block_tokens
        if skip_first or seq_len <= bt or (seq_len - 1) % bt:
            return None
        k = (seq_len - 1) // bt
        return k - 1, min(k - 1, self.hot_blocks)

    # ------------------------------------------------------------ GDN snapshots (M1b h08)

    def snapshot_due(self, seq_len: int, interval: int) -> Optional[int]:
        """M1b design §3.2: position p the decode step at seq_len snapshots, or None.

        seq_len counts the token this step appends, so before its forward the
        live GDN / PLE state covers p = seq_len - 1 tokens; the step snapshots
        when p is a positive multiple of the interval T (a multiple of the page
        size, so P is page aligned, §4.2).
        """
        if interval <= 0 or interval % self.page_size:
            raise ValueError(
                f"snapshot interval {interval} must be a positive multiple of "
                f"page_size {self.page_size}"
            )
        p = seq_len - 1
        return p if p > 0 and p % interval == 0 else None

    def can_publish(
        self,
        p: int,
        *,
        committed: int,
        host_blocks_done: int,
        gdn_d2h_done: bool,
        overwritten: bool,
    ) -> bool:
        """M1b design §2.4 as corrected in §11.6: p <= committed; blocks [0, p / bt)
        have finished host copies; the GDN D2H finished; no later D2H into the
        same buffer was issued."""
        return (
            p <= committed
            and host_blocks_done >= p // self.block_tokens
            and gdn_d2h_done
            and not overwritten
        )
