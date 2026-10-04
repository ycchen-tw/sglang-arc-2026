"""QSA HiSparse pool sizing (arc-forge M1a h05, design §5.2). Pure Python.

Worst-case static sizing from the KV budget V left after weights and mamba:

    L    = R * C                         logical tokens (what the scheduler sees)
    R_pf = C + chunk                     prefill area (h06 admission: decode reserves never touch it)
    B    = floor16((V - fixed) / (R * (bt * kv + layers * 10)) - page_blocks)
           capped at MAX_HOT_BLOCKS and at C in blocks

fixed = L * index + R_pf * kv + L / bt * 8 (mapping) + R * (ceil(C / bt) + page_blocks) * 8
(host map); layers * 10 is tokens int32 + locs int32 + lru int16 per slot and layer.
Same numbers as arc-forge work/qsa_hisparse/capacity.py, which checks this file.

M1b h09 (session cache, design §5.1): L is a budget, not R * C: logical_tokens sets
it (parked sessions and running requests share it; retract is the last resort), and
the host-prefix landing zone (one layer of K/V for ceil_page(C) tokens) joins fixed.
The host pool follows L (L / bt + R * page_blocks blocks, the coordinator's check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.mem_cache.qsa_hisparse_layout import HOT_BLOCKS_ALIGN, MAX_HOT_BLOCKS

TABLE_BYTES_PER_SLOT_LAYER = 4 + 4 + 2


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def swap_in_smem_bytes(hot_blocks: int, top_k: int) -> int:
    """Dynamic smem of the swap-in kernel (hisparse.cuh SmemLayout)."""
    chunks = -(-hot_blocks // 32)
    int32_words = top_k + 2 * (chunks + 1) + 2 * top_k + 2
    int16_words = hot_blocks + 2 * top_k
    return 4 * int32_words + 2 * int16_words


@dataclass(frozen=True)
class QSAHiSparsePlan:
    logical_tokens: int  # L
    prefill_tokens: int  # R_pf
    hot_blocks: int  # B
    hot_blocks_vram: int  # before the MAX_HOT_BLOCKS / context caps and any override
    device_tokens: int  # KV pool rows without the dump page: R_pf + R * (B + page_blocks) * bt
    fixed_bytes: int
    per_slot_bytes: int


def plan_qsa_hisparse(
    *,
    budget_bytes: int,
    context_len: int,
    max_running_requests: int,
    chunked_prefill_size: int,
    kv_bytes_per_token: int,
    index_bytes_per_token: int,
    num_layers: int,
    block_tokens: int,
    page_size: int,
    top_k: int,
    hot_override: Optional[int] = None,
    smem_optin: Optional[int] = None,
    logical_tokens: Optional[int] = None,
    landing_zone_bytes: int = 0,
) -> QSAHiSparsePlan:
    r, c, kv, bt = max_running_requests, context_len, kv_bytes_per_token, block_tokens
    if min(r, c, chunked_prefill_size, kv, num_layers, top_k) <= 0:
        raise ValueError(
            "QSA HiSparse sizing needs positive R, context, chunk, KV bytes, layers and top-k; "
            f"got R={r} C={c} chunk={chunked_prefill_size} kv={kv} layers={num_layers} top_k={top_k}"
        )
    if page_size % bt:
        raise ValueError(f"page_size {page_size} is not a multiple of block_tokens {bt}")
    page_blocks = page_size // bt
    ctx_blocks = -(-c // bt)
    logical = _round_up(r * c, page_size)
    if logical_tokens is not None:
        if logical_tokens % page_size or logical_tokens < _round_up(c, page_size):
            raise ValueError(
                f"qsa_logical_tokens={logical_tokens} must be a multiple of page_size "
                f"{page_size} holding at least one context of {c} tokens"
            )
        logical = logical_tokens
    if landing_zone_bytes < 0:
        raise ValueError(f"landing_zone_bytes must be >= 0, got {landing_zone_bytes}")
    # h06 admits a prompt only when its whole rest fits the prefill area, and raises for a
    # prompt that could never fit: the area must hold the longest prompt, C rounded to pages.
    prefill = _round_up(c + chunked_prefill_size, page_size)
    if prefill < _round_up(c, page_size):
        raise ValueError(f"prefill area {prefill} is below the context length {c} in pages")
    fixed = (
        logical * index_bytes_per_token
        + prefill * kv
        + logical // bt * 8
        + r * (ctx_blocks + page_blocks) * 8
        + landing_zone_bytes
    )
    per_slot = r * (bt * kv + num_layers * TABLE_BYTES_PER_SLOT_LAYER)
    if budget_bytes <= fixed:
        raise ValueError(
            f"KV budget {budget_bytes} B does not cover the index, prefill area and maps "
            f"({fixed} B) for R={r}, C={c}; lower --max-running-requests or --context-length"
        )
    hot_vram = ((budget_bytes - fixed) // per_slot - page_blocks) // HOT_BLOCKS_ALIGN * HOT_BLOCKS_ALIGN
    hot = min(hot_vram, MAX_HOT_BLOCKS, _round_up(ctx_blocks, HOT_BLOCKS_ALIGN))
    if hot_override is not None:
        top = min(hot_vram, MAX_HOT_BLOCKS)
        if hot_override % HOT_BLOCKS_ALIGN or not top_k <= hot_override <= top:
            raise ValueError(
                f"qsa_hot_blocks={hot_override} must be a multiple of {HOT_BLOCKS_ALIGN} "
                f"in [{top_k}, {top}] (top-k .. what fits)"
            )
        hot = hot_override
    if hot < top_k:
        raise ValueError(
            f"only {hot} hot blocks fit for R={r}, C={c}; the swap-in kernel needs at least "
            f"top-k={top_k}. Lower --max-running-requests or --context-length"
        )
    if smem_optin is not None and swap_in_smem_bytes(hot, top_k) > smem_optin:
        raise ValueError(
            f"B={hot} needs {swap_in_smem_bytes(hot, top_k)} B of dynamic smem; "
            f"the device opt-in limit is {smem_optin} B"
        )
    return QSAHiSparsePlan(
        logical_tokens=logical,
        prefill_tokens=prefill,
        hot_blocks=hot,
        hot_blocks_vram=hot_vram,
        device_tokens=prefill + r * (hot + page_blocks) * bt,
        fixed_bytes=fixed,
        per_slot_bytes=per_slot,
    )
