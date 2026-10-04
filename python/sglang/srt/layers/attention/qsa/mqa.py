"""Weight-free TileLang MQA operators for the simple QSA indexer;
the torch implementations are the fallback and the reference."""

import math
import os
from typing import Optional

import torch

from sglang.srt.environ import envs

# Group 8 needs the same crossover treatment group 2 got: it beats group 4 by
# 1.10-1.30x from about 128 rows up (B=16/30/64/110 at ctx 8.1k and 32.1k) and
# loses by 1-2 us below that.
_GROUPED_QSA_MIN_ROWS_SCALE = {2: 2.0, 4: 1.0, 8: 2.0}
# Opt-in, default off. `__launch_bounds__(threads, n)`: caps register usage so
# n blocks fit per SM. ncu on the grouped verify kernel at B=110/ctx 96,000:
# 71.6% of warp stall cycles wait on an L1TEX scoreboard while theoretical
# occupancy is pinned at 50% *by registers* (block limit 8 vs SM limit 32).
# n=10 moves that to 62.5% with zero spilling and 1.09-1.12x -- but only once
# there are enough request groups to saturate; below about B=80 the sign is
# non-monotonic (B=48 1.09x, B=64 0.97x, B=80 1.20x at ctx 32,100), so there is
# no honest default to ship. Measurements in
# docs/research/qwen38fn_qsa_verify_kernels_20260922.md.
_ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM = int(
    os.environ.get("ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM", "0")
)
if _ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM < 0:
    raise ValueError("ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM must be >= 0 (0 = off)")


def _qsa_verify_group_size(
    queries_per_request: Optional[int],
    rows: Optional[int] = None,
    configured_min_rows: int = 0,
) -> Optional[int]:
    """Widest supported group that divides the verify window and earns its floor.

    The grouped kernel's contract is narrower than "the whole verify window":
    it needs ``queries_per_request`` consecutive rows of one request with
    monotonic context lengths. A window of 8 splits into groups of 8, 4 or 2 and
    all of them satisfy it, so a width with no table entry of its own can still
    use the widest divisor that has one. Without this, width 8 misses the table,
    falls back to the per-row kernel, and re-reads each request's whole
    compressed-K tile once per verify row -- 426.7 us against 148.7 us for
    groups of eight at B=110, ctx 32,100 on B200.

    Supported group sizes are the keys of ``_GROUPED_QSA_MIN_ROWS_SCALE``, which
    are powers of two, and that is not a stylistic choice: the MMA lowering
    requires ``N = group * heads`` to be divisible by 8 *and* ``N / 8`` to divide
    the warp count, so group 3 (N=12 raw, 24 padded) and group 9 (36 / 72) fail
    to build at all. **An odd verify window therefore cannot group** -- and the
    window is ``speculative_num_steps + 1``, so MTP 8 steps gives width 9 and
    gets nothing here, while 7 steps gives width 8 and gets 2.9x.

    ``rows`` lets a wider group be skipped when the batch is too small to pay for
    it; without it the widest divisor is returned.
    """
    if not queries_per_request or queries_per_request < 2:
        return None
    for group in sorted(_GROUPED_QSA_MIN_ROWS_SCALE, reverse=True):
        if queries_per_request % group:
            continue
        if rows is not None:
            floor = _grouped_qsa_effective_min_rows(group, configured_min_rows)
            if floor is not None and rows < floor:
                continue
        return group
    return None


def _grouped_qsa_effective_min_rows(
    queries_per_request: Optional[int], configured_min_rows: int
) -> Optional[int]:
    """Return the qpr-aware grouped-MQA crossover, or ``None`` if unsupported.

    The original qpr=4 path crosses over at 64 rows on B200; qpr=2 uses a
    conservative 2x threshold. qpr=3's N=24 needs a three-warp lowering that
    measured slower than baseline through rows=384, so it deliberately falls
    back. A configured value of zero still
    disables the crossover floor for explicit microbenching.
    """

    if configured_min_rows < 0:
        raise ValueError("SGLANG_QSA_GROUPED_VERIFY_MIN_ROWS must be non-negative")
    scale = _GROUPED_QSA_MIN_ROWS_SCALE.get(queries_per_request)
    if scale is None:
        return None
    scaled = math.ceil(configured_min_rows * scale)
    return math.ceil(scaled / queries_per_request) * queries_per_request


try:
    import flashinfer.comm  # noqa: F401
except ImportError:
    pass

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    k_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA; k_scale [pages, page_size]
    (arc-forge h21) dequantizes an FP8 k_cache and quantizes q the same way."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    if k_scale is not None:
        from sglang.srt.mem_cache.qsa_kv_pool import (
            dequantize_qsa_index_k,
            quantize_qsa_index_k,
        )

        k_cache = dequantize_qsa_index_k(k_cache[:, :, 0], k_scale).unsqueeze(2)
        q = dequantize_qsa_index_k(*quantize_qsa_index_k(q))
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        groups_per_cta: int = 1,
        num_stages: int = 3,
        threads: int = 128,
    ):
        # The MMA layout needs 64 GEMM rows; a compressed page has full_page // ratio,
        # so pages are packed as sub-pages of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], T.bfloat16),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.bfloat16),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (bx, group_block):
                q_shared = T.alloc_shared([heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                T.copy(Q[bx, 0, :, :], q_shared)
                context_len = ContextLens[bx]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_len:
                        # TileLang's pipeliner rejects dynamic loops around smem copies;
                        # unroll at the Python level instead.
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[bx, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, head in T.Parallel(GROUP, heads):
                            scores[token, head] = T.max(scores[token, head], 0.0)
                        T.reduce_sum(scores, reduced, dim=1, clear=True)
                        for token in T.Parallel(GROUP):
                            position = group * GROUP + token
                            if position < context_len:
                                Logits[bx, position] = reduced[token] / Scale

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            # (token, query) maps one-to-one to (row, position); TileLang's
            # symbolic checker cannot prove that row_base + query is unique.
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        }
    )
    def _tilelang_qsa_mqa_grouped_verify_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        queries_per_request: int = 4,
        groups_per_cta: int = 4,
        num_stages: int = 3,
        threads: int = 128,
        min_blocks_per_sm: int = 0,
    ):
        """Speculative paged MQA that loads each request's K tile only once.

        QSA target-verify and DRAFT_EXTEND_V2 layouts are request-major: two or
        four consecutive query rows share a page table and have monotonically
        increasing context lengths. Widening the GEMM across those queries
        removes duplicate K-cache traffic without changing the MMA reduction
        order per output.
        """

        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch * heads, head_dim], T.bfloat16),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.bfloat16),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(batch, queries_per_request),
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (request, group_block):
                if min_blocks_per_sm:
                    T.annotate_min_blocks_per_sm(min_blocks_per_sm)
                row_base = request * queries_per_request
                q_shared = T.alloc_shared(
                    [queries_per_request * heads, head_dim], T.bfloat16
                )
                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)
                scores = T.alloc_fragment(
                    [GROUP, queries_per_request * heads], T.float32
                )
                scores_3d = T.reshape(
                    scores, (GROUP, queries_per_request, heads)
                )
                reduced = T.alloc_fragment(
                    [GROUP, queries_per_request], T.float32
                )
                T.copy(Q[row_base * heads, 0], q_shared)
                # Speculative paged row lengths are monotonic within a request.
                context_max = ContextLens[
                    row_base + queries_per_request - 1
                ]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_max:
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_max:
                                T.copy(
                                    KCache[
                                        PageTable[
                                            row_base, group * sub_pages + sp
                                        ],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[
                                        sp * page_size : (sp + 1) * page_size, :
                                    ],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, query, head in T.Parallel(
                            GROUP, queries_per_request, heads
                        ):
                            scores_3d[token, query, head] = T.max(
                                scores_3d[token, query, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, query in T.Parallel(
                            GROUP, queries_per_request
                        ):
                            row = row_base + query
                            position = group * GROUP + token
                            if position < ContextLens[row]:
                                Logits[row, position] = reduced[token, query] / Scale

        return kernel


    # arc-forge h21 (ARC_QSA_INDEX_FP8): the same two paged kernels, FP8 x FP8. The index Q is
    # quantized per (row, head) with the keys' power-of-two rule (quantize_qsa_index_q_fp8); the
    # e4m3 K pages and the e4m3 Q tile feed the MMA directly (fp32 accumulate, no widening pass),
    # the head's Q scale multiplies its relu'd score before the head sum and the key's scale the
    # summed score at the store (relu(s * x) = s * relu(x) for s > 0). With power-of-two scales the
    # logits are the BF16 kernels' over the dequantized Q and K up to the MMA's accumulation.
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_fp8_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        groups_per_cta: int = 1,
        num_stages: int = 3,
        threads: int = 128,
    ):
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], T.float8_e4m3fn),  # type: ignore
            QScale: T.Tensor([batch, heads], T.float32),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.float8_e4m3fn),  # type: ignore
            KScale: T.Tensor([pages, page_size], T.float32),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (bx, group_block):
                q_shared = T.alloc_shared([heads, head_dim], T.float8_e4m3fn)
                k_shared = T.alloc_shared([GROUP, head_dim], T.float8_e4m3fn)
                q_scale = T.alloc_fragment([heads], T.float32)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                T.copy(Q[bx, 0, :, :], q_shared)
                T.copy(QScale[bx, :], q_scale)
                context_len = ContextLens[bx]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_len:
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[bx, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, head in T.Parallel(GROUP, heads):
                            scores[token, head] = T.max(scores[token, head], 0.0) * q_scale[head]
                        T.reduce_sum(scores, reduced, dim=1, clear=True)
                        for token in T.Parallel(GROUP):
                            position = group * GROUP + token
                            if position < context_len:
                                Logits[bx, position] = (
                                    reduced[token]
                                    * KScale[
                                        PageTable[bx, position // page_size],
                                        position % page_size,
                                    ]
                                    / Scale
                                )

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        }
    )
    def _tilelang_qsa_mqa_grouped_verify_fp8_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        queries_per_request: int = 4,
        groups_per_cta: int = 4,
        num_stages: int = 3,
        threads: int = 128,
        min_blocks_per_sm: int = 0,
    ):
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch * heads, head_dim], T.float8_e4m3fn),  # type: ignore
            QScale: T.Tensor([batch * heads], T.float32),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.float8_e4m3fn),  # type: ignore
            KScale: T.Tensor([pages, page_size], T.float32),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                T.ceildiv(batch, queries_per_request),
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (request, group_block):
                if min_blocks_per_sm:
                    T.annotate_min_blocks_per_sm(min_blocks_per_sm)
                row_base = request * queries_per_request
                q_shared = T.alloc_shared(
                    [queries_per_request * heads, head_dim], T.float8_e4m3fn
                )
                k_shared = T.alloc_shared([GROUP, head_dim], T.float8_e4m3fn)
                q_scale = T.alloc_fragment([queries_per_request * heads], T.float32)
                scores = T.alloc_fragment(
                    [GROUP, queries_per_request * heads], T.float32
                )
                scores_3d = T.reshape(
                    scores, (GROUP, queries_per_request, heads)
                )
                reduced = T.alloc_fragment(
                    [GROUP, queries_per_request], T.float32
                )
                T.copy(Q[row_base * heads, 0], q_shared)
                T.copy(QScale[row_base * heads], q_scale)
                context_max = ContextLens[
                    row_base + queries_per_request - 1
                ]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_max:
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_max:
                                T.copy(
                                    KCache[
                                        PageTable[
                                            row_base, group * sub_pages + sp
                                        ],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[
                                        sp * page_size : (sp + 1) * page_size, :
                                    ],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, column in T.Parallel(
                            GROUP, queries_per_request * heads
                        ):
                            scores[token, column] = (
                                T.max(scores[token, column], 0.0) * q_scale[column]
                            )
                        T.reduce_sum(scores_3d, reduced, dim=2, clear=True)
                        for token, query in T.Parallel(
                            GROUP, queries_per_request
                        ):
                            row = row_base + query
                            position = group * GROUP + token
                            if position < ContextLens[row]:
                                Logits[row, position] = (
                                    reduced[token, query]
                                    * KScale[
                                        PageTable[row_base, position // page_size],
                                        position % page_size,
                                    ]
                                    / Scale
                                )

        return kernel

def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # A torch.cat of the padding rows would copy the whole [rows, keys] fp32 matrix,
    # doubling the dominant prefill buffer; allocate pre-padded instead.
    logits = torch.zeros((padded_rows, keys), dtype=torch.float32, device=q.device)
    q_padded = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(heads=heads, head_dim=head_dim, block_q=block_q)(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    verify_queries_per_request: Optional[int] = None,
    fill_invalid_tail: bool = True,
    k_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed; k_scale (arc-forge h21)
    selects the FP8 kernels: k_cache float8_e4m3fn, k_scale [pages, page_size] fp32."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    fp8 = k_scale is not None
    if fp8 and (
        k_cache.dtype != torch.float8_e4m3fn
        or k_scale.dtype != torch.float32
        or tuple(k_scale.shape) != tuple(k_cache.shape[:2])
    ):
        raise ValueError(
            "QSA FP8 decode needs a float8_e4m3fn cache and fp32 scales [pages, page_size], got "
            f"{k_cache.dtype} {tuple(k_cache.shape)} / {k_scale.dtype} {tuple(k_scale.shape)}"
        )
    if not fp8 and k_cache.dtype == torch.float8_e4m3fn:
        raise ValueError("QSA decode got an FP8 cache without its scales")
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    if fill_invalid_tail:
        logits = torch.full(
            (q.shape[0], max_model_len),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        # Internal decode selection passes lengths directly to fast_topk, so
        # columns outside [0, length) are provably unread. Avoid a full
        # [rows, max_model_len] fill while preserving the public default.
        logits = torch.empty(
            (q.shape[0], max_model_len),
            dtype=torch.float32,
            device=q.device,
        )
    if not q.shape[0] or not max_model_len:
        return logits
    query_heads, head_dim = q.shape[1:]
    groups_per_cta = envs.SGLANG_QSA_MQA_GROUPS_PER_CTA.get()
    if groups_per_cta < 0:
        raise ValueError(
            "SGLANG_QSA_MQA_GROUPS_PER_CTA must be zero (auto) or positive"
        )
    if groups_per_cta == 0:
        rows = q.shape[0]
        groups_per_cta = 1 if rows <= 8 else 2 if rows <= 24 else 4
    configured_min_rows = envs.SGLANG_QSA_GROUPED_VERIFY_MIN_ROWS.get()
    grouped_size = _qsa_verify_group_size(
        verify_queries_per_request, q.shape[0], configured_min_rows
    )
    grouped_min_rows = _grouped_qsa_effective_min_rows(grouped_size, configured_min_rows)
    use_grouped_verify = (
        envs.SGLANG_QSA_GROUPED_VERIFY.get()
        and grouped_min_rows is not None
        and q.shape[0] >= grouped_min_rows
        # The producer's row layout is still checked against the *window*
        # width; the sub-group only subdivides a window that is already
        # request-major.
        and q.shape[0] % verify_queries_per_request == 0
    )
    # The validated MMA layout requires N (the Q-head dimension) to be a
    # multiple of eight, and for the per-row kernel N *is* `heads`, so 4 index
    # heads have to be zero-padded to 8. The grouped kernel's N is
    # `queries_per_request * heads`, which at the supported group sizes is
    # already a multiple of eight -- there the pad only buys a torch.cat per
    # call and a GEMM half of whose columns are zeros. Measured on B200 at
    # ctx 96,000 / B=110: 356 -> 231 us, logits bitwise identical.
    if use_grouped_verify and (grouped_size * query_heads) % 8 == 0:
        kernel_heads = query_heads
    else:
        kernel_heads = max(8, ((query_heads + 7) // 8) * 8)
    q_kernel = q.to(torch.bfloat16)
    if kernel_heads != query_heads:
        # Zero-padding preserves the weight-free head sum.
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    tail_args = (
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    if fp8:
        # arc-forge h21: FP8 x FP8; padded heads are zero rows (scale 1, all-zero e4m3)
        q8, q_scale = quantize_qsa_index_q_fp8(q_kernel.contiguous())
        if use_grouped_verify:
            _tilelang_qsa_mqa_grouped_verify_fp8_kernel(
                heads=kernel_heads,
                head_dim=head_dim,
                page_size=page_size,
                queries_per_request=grouped_size,
                groups_per_cta=groups_per_cta,
                min_blocks_per_sm=_ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM,
            )(q8.view(-1, head_dim), q_scale.view(-1), k_cache.contiguous(),
              k_scale.contiguous(), *tail_args)
        else:
            _tilelang_qsa_mqa_decode_fp8_kernel(
                heads=kernel_heads,
                head_dim=head_dim,
                page_size=page_size,
                groups_per_cta=groups_per_cta,
            )(q8.unsqueeze(1), q_scale, k_cache.contiguous(), k_scale.contiguous(), *tail_args)
        return logits
    kernel_args = (k_cache.to(torch.bfloat16).contiguous(), *tail_args)
    if use_grouped_verify:
        _tilelang_qsa_mqa_grouped_verify_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            queries_per_request=grouped_size,
            groups_per_cta=groups_per_cta,
            min_blocks_per_sm=_ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM,
        )(q_kernel.contiguous().view(-1, head_dim), *kernel_args)
    else:
        _tilelang_qsa_mqa_decode_kernel(
            heads=kernel_heads,
            head_dim=head_dim,
            page_size=page_size,
            groups_per_cta=groups_per_cta,
        )(q_kernel.unsqueeze(1).contiguous(), *kernel_args)
    return logits



try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _quantize_q_fp8_kernel(q, q8, scale, D: tl.constexpr):
        # one program per (row, head); the keys' rule (qsa_kv_pool.quantize_qsa_index_k): with
        # amax = m * 2^e, m in [0.5, 1): scale = 2^(e - 9) if m <= 0.875 else 2^(e - 8), i.e. the
        # fraction bits of amax <= 0x600000; e4m3 round to nearest even. amax = 0 (and a subnormal
        # amax, below 2^-126) gives scale 1.
        pid = tl.program_id(0)
        offs = tl.arange(0, D)
        x = tl.load(q + pid * D + offs).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        bits = amax.to(tl.int32, bitcast=True)
        e_field = (bits >> 23) & 0xFF
        frac = bits & 0x7FFFFF
        k = tl.where(frac <= 0x600000, e_field - 126 - 9, e_field - 126 - 8)
        normal = e_field > 0
        scale_v = tl.where(normal, ((k + 127) << 23).to(tl.float32, bitcast=True), 1.0)
        inv_v = tl.where(normal, ((127 - k) << 23).to(tl.float32, bitcast=True), 1.0)
        tl.store(q8 + pid * D + offs, (x * inv_v).to(tl.float8e4nv))
        tl.store(scale + pid, scale_v)


def quantize_qsa_index_q_fp8(q: torch.Tensor):
    """arc-forge h21: index Q [rows, heads, head_dim] (BF16) -> (e4m3 [rows, heads, head_dim], fp32
    [rows, heads]) per (row, head), bit for bit quantize_qsa_index_k (one launch; CUDA only)."""
    if q.ndim != 3 or not q.is_cuda or not q.is_contiguous():
        raise ValueError(f"index Q must be a contiguous CUDA [rows, heads, dim] tensor, got {tuple(q.shape)}")
    rows, heads, dim = q.shape
    q8 = torch.empty((rows, heads, dim), dtype=torch.float8_e4m3fn, device=q.device)
    scale = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    if rows * heads:
        _quantize_q_fp8_kernel[(rows * heads,)](q, q8, scale, D=dim, num_warps=1)
    return q8, scale

def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    verify_queries_per_request: Optional[int] = None,
    fill_invalid_tail: bool = True,
    k_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q,
            k_cache,
            page_table,
            context_lens,
            max_model_len,
            score_scale,
            verify_queries_per_request,
            fill_invalid_tail,
            k_scale,
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale, k_scale
    )


__all__ = [
    "HAS_TILELANG",
    "quantize_qsa_index_q_fp8",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
]
