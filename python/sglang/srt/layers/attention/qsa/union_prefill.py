"""QSA prefix chunk prefill that reads each selected key once per tile of neighbouring queries.

arc-forge b200-perf-prefill, opt-in ``ARC_QSA_UNION_PREFILL=1`` (qwen_sparse_attn_backend.forward_extend).

The validated chunk kernel (``sparse_attn._sparse_gqa_chunk_prefill``) runs one program per
(query, kv head) and gathers that query's own 2,051 selected keys: neighbouring queries of one
sequence select largely the same compressed blocks, but every query re-reads them, so a TP2 rank of
Qwen3.8-FN reads ~2.95 GB per QSA layer at the RL production extend shape (4 x 704 new tokens on
10k cached prefixes) for a roofline of ~0.4 ms. It also needs the whole context gathered into a
tight copy first (``_extend_context``: an index_select + cat per request and plane).

Here T consecutive queries of one sequence form a tile. Each query's selection is the expansion of
its top-k compressed blocks (``qsa.kernel.expand_qsa_block_indices``): ``ratio`` consecutive tokens
per selected block, front-packed, then the up to ``ratio - 1`` tail tokens of the query's own
incomplete block. Per layer:

  1. ``_union_scatter``: every query sets the bits of its selected blocks in its own bitmap
     (one bitmap per query, ``ceil(seq / ratio / 32)`` words, laid out per tile);
  2. ``_union_compact``: per tile, the OR of its T bitmaps compacted into an ascending list of
     blocks: the T-bit mask of the queries that selected each, and its ``ratio`` pool slots
     (req_to_token resolved here, so the attention loop's gather depends on one load, not two);
  3. ``_union_attention``: per (tile, kv head) one program with T x (heads per kv head, padded to a
     power of two) query rows walks the union list, gathers each block's K/V once straight from
     the KV pool, masks every (query, key) pair the query did not select, then does the tail
     tokens as one more masked chunk.

Every query attends to exactly the multiset of keys the chunk kernel gives it (the selected blocks,
then its tail; a tail token is never in its own query's blocks, and if it were it would count twice
in both kernels), with the same Q pre-scaling and FP8 -> query-dtype cast. Only the order of the
online-softmax accumulation differs, so the output is not bitwise the chunk kernel's.

Only the plain FP8 / BF16 pool without HiSparse goes this way; the backend raises otherwise.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

_WORD_CHUNK = 32  # bitmap words per compaction step (1,024 blocks)


def _parse_mode() -> str:
    value = os.environ.get("ARC_QSA_UNION_PREFILL", "0")
    if value not in ("0", "1", "check"):
        raise ValueError(f"ARC_QSA_UNION_PREFILL must be 0, 1 or check, got {value!r}")
    return value


def _parse_config():
    """ARC_QSA_UNION_PREFILL_CFG=T,NB,warps,stages: queries per tile (any 1..16; the program's
    rows are T x q heads per kv head, padded to a power of two, so T = 5 / 10 fill 64 / 128 rows
    with 12 heads), union blocks per step (NB x ratio >= 16 keys), attention warps and stages."""
    raw = os.environ.get("ARC_QSA_UNION_PREFILL_CFG", "10,32,8,1")
    parts = raw.split(",")
    if len(parts) != 4:
        raise ValueError(f"ARC_QSA_UNION_PREFILL_CFG must be T,NB,warps,stages, got {raw!r}")
    tile, nb, warps, stages = (int(x) for x in parts)
    for name, value in (("NB", nb), ("warps", warps)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"ARC_QSA_UNION_PREFILL_CFG {name} must be a power of two, got {value}")
    if not 1 <= tile <= 16:
        raise ValueError(f"ARC_QSA_UNION_PREFILL_CFG T must be 1..16, got {tile}")
    if stages <= 0:
        raise ValueError(f"ARC_QSA_UNION_PREFILL_CFG stages must be positive, got {stages}")
    return tile, nb, warps, stages


ARC_QSA_UNION_PREFILL = _parse_mode()
ARC_QSA_UNION_PREFILL_CFG = _parse_config()


@triton.jit
def _union_scatter(
    indices,
    cu_q,
    bitmap,
    word_base,
    words,
    si_m,
    T: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    q_start = tl.load(cu_q + b)
    q_end = tl.load(cu_q + b + 1)
    if q_start + i >= q_end:
        return
    row = (q_start + i).to(tl.int64)
    tile = i // T
    t = i % T
    wb = tl.load(words + b).to(tl.int64)
    base = tl.load(word_base + b) + (tile * T + t).to(tl.int64) * wb
    j = tl.arange(0, BLOCK_TOPK)
    first = tl.load(indices + row * si_m + j * RATIO)
    last = tl.load(indices + row * si_m + j * RATIO + (RATIO - 1))
    # A selected block fills RATIO columns; the tail that follows the last block has at most
    # RATIO - 1 entries, so its group's last column is -1.
    valid = last >= 0
    blk = tl.where(valid, first // RATIO, 0)
    tl.atomic_or(bitmap + base + (blk >> 5), 1 << (blk & 31), mask=valid, sem="relaxed")


@triton.jit
def _union_compact(
    bitmap,
    cu_q,
    word_base,
    words,
    tile_base,
    req_to_token,
    req_pool_indices,
    req_stride,
    masks,
    slots,
    counts,
    CAP: tl.constexpr,
    T: tl.constexpr,
    TP: tl.constexpr,
    RATIO: tl.constexpr,
    WORD_CHUNK: tl.constexpr,
):
    """Per tile: the union's blocks in ascending order as (T-bit query mask, RATIO pool slots).
    TP: T rounded up to a power of two."""
    tile = tl.program_id(0)
    b = tl.program_id(1)
    n_q = tl.load(cu_q + b + 1) - tl.load(cu_q + b)
    if tile * T >= n_q:
        return
    wb = tl.load(words + b).to(tl.int64)
    base = tl.load(word_base + b) + tile.to(tl.int64) * T * wb
    gtile = (tl.load(tile_base + b) + tile).to(tl.int64)
    table = req_to_token + tl.load(req_pool_indices + b).to(tl.int64) * req_stride
    out_mask = masks + gtile * CAP
    out_slot = slots + gtile * (CAP * RATIO)
    tt = tl.arange(0, TP)
    bits = tl.arange(0, 32)
    lanes = tl.arange(0, WORD_CHUNK * 32)
    within = tl.arange(0, RATIO)
    count = tl.zeros([], tl.int32)
    for w0 in range(0, wb, WORD_CHUNK):
        w = w0 + tl.arange(0, WORD_CHUNK)
        live = w < wb
        word = tl.load(
            bitmap + base + tt[:, None] * wb + w[None, :],
            mask=(tt < T)[:, None] & live[None, :],
            other=0,
        )  # [TP, WORD_CHUNK]
        member = (word[:, :, None] >> bits[None, None, :]) & 1  # [T, WORD_CHUNK, 32]
        mask = tl.sum(member << tt[:, None, None], axis=0)  # [WORD_CHUNK, 32]
        mask = tl.reshape(mask, [WORD_CHUNK * 32])
        sel = mask != 0
        rank = tl.cumsum(sel.to(tl.int32), 0) - 1
        pos = count + rank
        tl.store(out_mask + pos, mask, mask=sel)
        token = (w0 * 32 + lanes)[:, None] * RATIO + within[None, :]  # [lanes, RATIO]
        slot = tl.load(table + token, mask=sel[:, None], other=0)
        tl.store(out_slot + pos[:, None] * RATIO + within[None, :], slot, mask=sel[:, None])
        count += tl.sum(sel.to(tl.int32), 0)
    tl.store(counts + gtile, count)


@triton.jit
def _softmax_step(q_values, keys, values, member, max_value, normalizer, accumulator):
    # keys [BN, dim] like values (row loads coalesce along dim)
    scores = tl.dot(q_values, tl.trans(keys))
    scores = tl.where(member, scores, -float("inf"))
    next_max = tl.maximum(max_value, tl.max(scores, 1))
    # a row with no key in this step (or none yet) keeps zero mass
    safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
    alpha = tl.math.exp2(max_value - safe_max)
    probabilities = tl.math.exp2(scores - safe_max[:, None])
    accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator * alpha[:, None])
    normalizer = normalizer * alpha + tl.sum(probabilities, 1)
    return next_max, normalizer, accumulator


@triton.jit
def _union_attention(
    q,
    k,
    v,
    out,
    req_to_token,
    req_pool_indices,
    cu_q,
    kv_lens,
    tile_base,
    masks,
    slots,
    counts,
    scale,
    req_stride,
    sq_m,
    sq_h,
    sq_d,
    sk_n,
    sk_h,
    sk_d,
    sv_n,
    sv_h,
    sv_d,
    so_m,
    so_h,
    so_d,
    CAP: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    RATIO: tl.constexpr,
    NB: tl.constexpr,
    TAIL_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Rows: row m = query t * GROUP_SIZE + head (m < T * GROUP_SIZE), M a power of two."""
    tile = tl.program_id(0)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    b = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + b)
    n_q = tl.load(cu_q + b + 1) - q_start
    if tile * T >= n_q:
        return
    kv_len = tl.load(kv_lens + b)
    req = tl.load(req_pool_indices + b).to(tl.int64)
    gtile = (tl.load(tile_base + b) + tile).to(tl.int64)
    count = tl.load(counts + gtile)
    # the tile's first query sits at logical position p0; query t at p0 + t
    p0 = kv_len - n_q + tile * T

    rows = tl.arange(0, M)
    t_of_row = rows // GROUP_SIZE
    head = rows % GROUP_SIZE
    q_rel = tile * T + t_of_row
    row_valid = (q_rel < n_q) & (rows < T * GROUP_SIZE)
    q_row = (q_start + q_rel).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q + q_row[:, None] * sq_m + (group * GROUP_SIZE + head)[:, None] * sq_h + offs_d[None, :] * sq_d,
        mask=row_valid[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    max_value = tl.full([M], -float("inf"), tl.float32)
    normalizer = tl.zeros([M], tl.float32)
    accumulator = tl.zeros([M, HEAD_DIM], tl.float32)

    BN: tl.constexpr = NB * RATIO
    k_head = k + group * sk_h
    v_head = v + group * sv_h
    table = req_to_token + req * req_stride
    mask_row = masks + gtile * CAP
    slot_row = slots + gtile * (CAP * RATIO)
    for start in range(0, count, NB):
        e = start + tl.arange(0, NB)
        live = e < count
        qmask = tl.load(mask_row + e, mask=live, other=0)
        key_mask = tl.reshape(tl.broadcast_to(qmask[:, None], [NB, RATIO]), [BN])
        n = start * RATIO + tl.arange(0, BN)
        key_live = n < count * RATIO
        slot = tl.load(slot_row + n, mask=key_live, other=0).to(tl.int64)
        keys = tl.load(
            k_head + slot[:, None] * sk_n + offs_d[None, :] * sk_d, mask=key_live[:, None], other=0.0
        ).to(q_values.dtype)
        values = tl.load(
            v_head + slot[:, None] * sv_n + offs_d[None, :] * sv_d, mask=key_live[:, None], other=0.0
        ).to(q_values.dtype)
        member = ((key_mask[None, :] >> t_of_row[:, None]) & 1) != 0
        max_value, normalizer, accumulator = _softmax_step(
            q_values, keys, values, member, max_value, normalizer, accumulator
        )

    # tails: query p attends to [floor((p + 1) / RATIO) * RATIO, p] (at most RATIO - 1 tokens)
    lo = (p0 + 1) // RATIO * RATIO
    token = lo + tl.arange(0, TAIL_N)
    last = tl.minimum(p0 + T, kv_len)  # one past the tile's last real query
    key_live = token < last
    p_row = p0 + t_of_row
    tail_start = (p_row + 1) // RATIO * RATIO
    member = (token[None, :] >= tail_start[:, None]) & (token[None, :] <= p_row[:, None]) & key_live[None, :]
    slot = tl.load(table + token, mask=key_live, other=0).to(tl.int64)
    keys = tl.load(
        k_head + slot[:, None] * sk_n + offs_d[None, :] * sk_d, mask=key_live[:, None], other=0.0
    ).to(q_values.dtype)
    values = tl.load(
        v_head + slot[:, None] * sv_n + offs_d[None, :] * sv_d, mask=key_live[:, None], other=0.0
    ).to(q_values.dtype)
    max_value, normalizer, accumulator = _softmax_step(
        q_values, keys, values, member, max_value, normalizer, accumulator
    )

    output = accumulator / normalizer[:, None]
    tl.store(
        out + q_row[:, None] * so_m + (group * GROUP_SIZE + head)[:, None] * so_h + offs_d[None, :] * so_d,
        output,
        mask=row_valid[:, None],
    )


def _host_ints_to_device(values, dtype, device) -> torch.Tensor:
    """No host sync: pinned staging + non-blocking copy (plain copy for a CPU device)."""
    if torch.device(device).type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


class UnionPrefillPlan:
    """The tile layout of one extend forward, from the scheduler's host lengths; every QSA layer
    of the forward reuses it and its scratch (the layers run one after another on one stream)."""

    def __init__(
        self,
        extend_lens: List[int],
        seq_lens: List[int],
        *,
        ratio: int,
        topk_width: int,
        device,
        tile: Optional[int] = None,
    ):
        if tile is None:
            tile = ARC_QSA_UNION_PREFILL_CFG[0]
        if len(extend_lens) != len(seq_lens) or not extend_lens:
            raise ValueError(f"union prefill plan: {len(extend_lens)} extend / {len(seq_lens)} seq lengths")
        block_topk = (topk_width - (ratio - 1)) // ratio
        if block_topk * ratio + ratio - 1 != topk_width or block_topk & (block_topk - 1):
            raise ValueError(
                f"top-k width {topk_width} is not a power-of-two block top-k x {ratio} + {ratio - 1} tail"
            )
        if any(e <= 0 or e > s for e, s in zip(extend_lens, seq_lens)):
            raise ValueError(f"union prefill: extend lengths {extend_lens} vs sequence lengths {seq_lens}")
        self.tile, self.ratio, self.block_topk = tile, ratio, block_topk
        self.cap = tile * block_topk
        tiles = [triton.cdiv(e, tile) for e in extend_lens]
        words = [triton.cdiv(triton.cdiv(s, ratio), 32) for s in seq_lens]
        tile_base, word_base, t_acc, w_acc = [], [], 0, 0
        for n, w in zip(tiles, words):
            tile_base.append(t_acc)
            word_base.append(w_acc)
            t_acc += n
            w_acc += n * tile * w
        self.total_tiles, self.total_words = t_acc, w_acc
        self.max_tiles = max(tiles)
        self.max_q = max(extend_lens)
        self.tile_base = _host_ints_to_device(tile_base, torch.int32, device)
        self.word_base = _host_ints_to_device(word_base, torch.int64, device)
        self.words = _host_ints_to_device(words, torch.int32, device)
        self.bitmap = torch.empty(w_acc, dtype=torch.int32, device=device)
        self.masks = torch.empty(t_acc * self.cap, dtype=torch.int32, device=device)
        self.slots = torch.empty(t_acc * self.cap * ratio, dtype=torch.int32, device=device)
        self.counts = torch.empty(t_acc, dtype=torch.int32, device=device)
        # the last _union_attention launch (the microbench reads its registers / spills)
        self.last_kernel = None

    def build(
        self,
        indices: torch.Tensor,
        cu_q: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        """Union lists of this layer's selection: per tile, ascending blocks with their query
        masks and pool slots."""
        if indices.ndim != 2 or indices.dtype != torch.int32 or indices.stride(1) != 1:
            raise ValueError(
                f"union prefill needs int32 [rows, width] indices with unit column stride, got "
                f"{indices.dtype} {tuple(indices.shape)} strides {indices.stride()}"
            )
        batch = cu_q.shape[0] - 1
        self.bitmap.zero_()
        _union_scatter[(self.max_q, batch)](
            indices,
            cu_q,
            self.bitmap,
            self.word_base,
            self.words,
            indices.stride(0),
            T=self.tile,
            RATIO=self.ratio,
            BLOCK_TOPK=self.block_topk,
            num_warps=4,
        )
        _union_compact[(self.max_tiles, batch)](
            self.bitmap,
            cu_q,
            self.word_base,
            self.words,
            self.tile_base,
            req_to_token,
            req_pool_indices,
            req_to_token.stride(0),
            self.masks,
            self.slots,
            self.counts,
            CAP=self.cap,
            T=self.tile,
            TP=triton.next_power_of_2(self.tile),
            RATIO=self.ratio,
            WORD_CHUNK=_WORD_CHUNK,
            num_warps=4,
        )


def union_prefill_attention(
    plan: UnionPrefillPlan,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: float,
    *,
    nb: Optional[int] = None,
    warps: Optional[int] = None,
    stages: Optional[int] = None,
) -> torch.Tensor:
    """Sparse GQA over every query's selection, K/V read from the pool through req_to_token.

    q [rows, q heads, dim]; k_buffer / v_buffer [slots, kv heads, dim] (FP8 or q's dtype);
    indices int32 [rows, block_topk * ratio + ratio - 1] (expand_qsa_block_indices); cu_q
    [bs + 1], kv_lens [bs] (int32, device); req_pool_indices [bs]."""
    _, cfg_nb, cfg_warps, cfg_stages = ARC_QSA_UNION_PREFILL_CFG
    nb = nb or cfg_nb
    warps = warps or cfg_warps
    stages = stages or cfg_stages
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    if num_q_heads % num_kv_heads or head_dim & (head_dim - 1):
        raise ValueError(f"union prefill: {num_q_heads} q heads / {num_kv_heads} kv heads, dim {head_dim}")
    if k_buffer.dtype != v_buffer.dtype or k_buffer.dtype not in (q.dtype, torch.float8_e4m3fn):
        raise ValueError(f"union prefill reads {q.dtype} or FP8 K/V, got {k_buffer.dtype}/{v_buffer.dtype}")
    if req_to_token.stride(1) != 1:
        raise ValueError("union prefill needs req_to_token with unit column stride")
    if nb * plan.ratio < 16:
        # tl.dot needs K >= 16 for the probabilities x values product (the Triton interpreter
        # does not check it, so CPU runs would pass a config CUDA cannot compile)
        raise ValueError(f"union prefill: NB x ratio = {nb} x {plan.ratio} keys per step, need >= 16")
    group_size = num_q_heads // num_kv_heads
    batch = cu_q.shape[0] - 1
    plan.build(indices, cu_q, req_to_token, req_pool_indices)
    out = torch.empty_like(q)
    tail_n = max(16, triton.next_power_of_2(plan.tile + plan.ratio - 1))
    plan.last_kernel = _union_attention[(plan.max_tiles, batch * num_kv_heads)](
        q,
        k_buffer,
        v_buffer,
        out,
        req_to_token,
        req_pool_indices,
        cu_q,
        kv_lens,
        plan.tile_base,
        plan.masks,
        plan.slots,
        plan.counts,
        scale,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        CAP=plan.cap,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        M=max(16, triton.next_power_of_2(plan.tile * group_size)),
        T=plan.tile,
        RATIO=plan.ratio,
        NB=nb,
        TAIL_N=tail_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


class UnionPrefillCheck:
    """ARC_QSA_UNION_PREFILL=check: every prefix-extend QSA layer also runs the chunk kernel and
    compares (host sync per layer; validation only). Reordering the online softmax moves an output
    element by a few bf16 ulps of its row's scale; a wrong key set moves it by O(1) of it. Raises on
    a non-finite output or an element off by more than 1/16 of its row's max |ref| (+1e-3); every
    100 layers logs the largest ratio seen."""

    LIMIT = 1.0 / 16

    def __init__(self):
        self.layers = 0
        self.worst = 0.0
        self.mean_ratio_sum = 0.0

    def compare(self, out: torch.Tensor, ref: torch.Tensor, layer_id: int, logger) -> None:
        o, r = out.float(), ref.float()
        if not torch.isfinite(o).all():
            raise RuntimeError(f"ARC_QSA_UNION_PREFILL=check: non-finite output in layer {layer_id}")
        err = (o - r).abs().amax(dim=-1)
        scale = r.abs().amax(dim=-1) + 1e-3
        ratio = err / scale
        worst = float(ratio.max())
        if worst > self.LIMIT:
            bad = int((ratio > self.LIMIT).sum())
            raise RuntimeError(
                f"ARC_QSA_UNION_PREFILL=check: layer {layer_id}, {bad} (row, head) outputs off by more "
                f"than {self.LIMIT} of their scale (worst {worst:.3e}) against the chunk kernel"
            )
        self.layers += 1
        self.worst = max(self.worst, worst)
        self.mean_ratio_sum += float(ratio.mean())
        if self.layers % 100 == 0:
            logger.info(
                f"ARC_QSA_UNION_PREFILL=check: {self.layers} layers matched; worst |diff|/scale "
                f"{self.worst:.3e}, mean {self.mean_ratio_sum / self.layers:.3e}"
            )


class PrefillDump:
    """Debug: ARC_QSA_PREFILL_DUMP=<dir> saves the inputs of the prefix chunk kernel for the
    microbench (scripts/arc/qsa_union_prefill_bench.py): TP rank 0, the first
    ARC_QSA_PREFILL_DUMP_MAX (3) extend forwards with at least ARC_QSA_PREFILL_DUMP_MIN_ROWS (2048)
    query rows over at least ARC_QSA_PREFILL_DUMP_MIN_SEQS (3) requests, layers
    ARC_QSA_PREFILL_DUMP_LAYERS (3,23,47). Host syncs; debug runs only."""

    def __init__(self, directory: str):
        self.dir = directory
        self.layers = {int(x) for x in os.environ.get("ARC_QSA_PREFILL_DUMP_LAYERS", "3,23,47").split(",")}
        self.max = int(os.environ.get("ARC_QSA_PREFILL_DUMP_MAX", "3"))
        self.min_rows = int(os.environ.get("ARC_QSA_PREFILL_DUMP_MIN_ROWS", "2048"))
        self.min_seqs = int(os.environ.get("ARC_QSA_PREFILL_DUMP_MIN_SEQS", "3"))
        # every request's cached prefix at least this long (steady-state agent turns, not cold starts)
        self.min_prefix = int(os.environ.get("ARC_QSA_PREFILL_DUMP_MIN_PREFIX", "0"))
        self.done = 0
        self.current = None
        os.makedirs(directory, exist_ok=True)

    def wants(self, plan, layer_id: int) -> bool:
        if self.done >= self.max or layer_id not in self.layers:
            return False
        if sum(plan.extend_lens) < self.min_rows or len(plan.extend_lens) < self.min_seqs:
            return False
        if min(plan.prefix_lens) < self.min_prefix:
            return False
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0

    def save(self, plan, layer_id, q, k_ctx, v_ctx, indices, scale) -> None:
        if self.current is not plan:
            if self.current is not None:
                self.done += 1
                if self.done >= self.max:
                    return
            self.current = plan
        rows = sum(plan.extend_lens)
        fp8 = k_ctx.dtype == torch.float8_e4m3fn
        torch.save(
            {
                "q": q[:rows].cpu(),
                "indices": indices[:rows].cpu(),
                "k": (k_ctx.view(torch.uint8) if fp8 else k_ctx).cpu(),
                "v": (v_ctx.view(torch.uint8) if fp8 else v_ctx).cpu(),
                "kv_fp8": fp8,
                "extend_lens": plan.extend_lens,
                "seq_lens": plan.seq_lens,
                "scale": float(scale),
                "layer_id": layer_id,
            },
            os.path.join(self.dir, f"b{self.done}-l{layer_id}.pt"),
        )
