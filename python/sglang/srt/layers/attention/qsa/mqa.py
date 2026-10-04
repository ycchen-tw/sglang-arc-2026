"""Weight-free TileLang MQA operators for the simple QSA indexer;
the torch implementations are the fallback and the reference."""

import math
import os
from typing import Optional

import torch

from sglang.srt.environ import envs

_GROUPED_QSA_MIN_ROWS_SCALE = {2: 2.0, 4: 1.0}
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


def _qsa_verify_group_size(queries_per_request: Optional[int]) -> Optional[int]:
    """Widest supported sub-group that evenly divides the verify window.

    The grouped kernel's contract is narrower than "the whole verify window":
    it needs ``queries_per_request`` consecutive rows of one request with
    monotonic context lengths. A width-8 window (MTP 8 steps) splits into two
    sub-groups of four that both satisfy it, so a width with no table entry of
    its own can still use the widest divisor that has one. Without this, width
    8 misses the table, falls back to the per-row kernel, and re-reads each
    request's whole compressed-K tile once per verify row -- measured at 329 us
    against 233 us for groups of four at B=110, ctx=32,100 on B200.

    Width 3 keeps returning None: its N=24 three-warp lowering measured slower
    than baseline, and 3 has no supported divisor above 1.
    """
    if not queries_per_request or queries_per_request < 2:
        return None
    if queries_per_request in _GROUPED_QSA_MIN_ROWS_SCALE:
        return queries_per_request
    for group in sorted(_GROUPED_QSA_MIN_ROWS_SCALE, reverse=True):
        if queries_per_request % group == 0:
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
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
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
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
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
    grouped_size = _qsa_verify_group_size(verify_queries_per_request)
    grouped_min_rows = _grouped_qsa_effective_min_rows(
        grouped_size, envs.SGLANG_QSA_GROUPED_VERIFY_MIN_ROWS.get()
    )
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
    kernel_args = (
        k_cache.to(torch.bfloat16).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
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
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


__all__ = [
    "HAS_TILELANG",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
]
