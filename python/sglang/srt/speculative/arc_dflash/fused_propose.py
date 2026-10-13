"""arc-forge ARC_DFLASH_FUSED_PROPOSE (opt-in, default 0): the selector chain of state_pool.fast_propose as one
Triton kernel.

fast_propose walks the proposal positions in Python: per position the selector score (a predecessor-codebook
gather, a product, an einsum against the candidates' successor rows, the unary add), `distribution` on the per-row
policy path (asserts, a 16-wide top-k sort, a nucleus branch that torch.where evaluates for every row, softmax,
scatter back, the greedy one-hot) and `sample_index` (a uniform draw, cumsum, compare, sum) -- about 45 small
kernels per position inside the drafter's CUDA graph. Here one program per request walks all positions:

    score_k = unary_k + bf16( sum_r bf16(pred[prev]_r * proj_r) * succ[cand_k]_r )
    q       = one-hot at argmax(score)          if temperature == 0
              softmax(score / temperature)      otherwise
    index   = #{ j : cumsum(q)_j <= u }  (clamped to top_k - 1);   prev = cand[index]

which is fast_propose's arithmetic (the drafter's selector runs top_p = 1, top_k = all 16, so `distribution`
reduces to that), with the intermediate roundings of the torch path (the codebooks' dtype for the product and the
einsum's output). Differences: the reduction orders (the 256-wide dot, the softmax sum, the cumsum) and the random
stream -- one [batch, steps] uniform draw from the default generator instead of one [batch, 1] draw per position.
q stays the exact distribution the token was drawn from, so rejection sampling's guarantee is unchanged; the
proposals are a different but identically distributed draw (not bitwise).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .sampling import Proposal


@triton.jit
def _chain_kernel(
    UNARY, PROJ, CAND, CAND_PRED, CAND_SUCC, PRED, SUCC, ANCHOR, TEMP, U, TOK, Q, SEL,
    s_ub, s_up, s_pb, s_pp, s_cb, s_cp, s_rpb, s_rpp, s_rsb, s_rsp, s_pred, s_succ, s_tb, s_qb, s_qp, s_sb,
    K: tl.constexpr, R: tl.constexpr, STEPS: tl.constexpr, TINY: tl.constexpr,
    CONTEXT_DTYPE: tl.constexpr, PAIR_DTYPE: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.arange(0, K)
    r = tl.arange(0, R)
    prev = tl.load(ANCHOR + b)  # the predecessor's codebook row (CandidateSelector.prune: not the token id)
    t = tl.load(TEMP + b)
    greedy = t == 0
    inv_t = 1.0 / tl.maximum(t, TINY)
    for p in tl.static_range(STEPS):
        cand = tl.load(CAND + b * s_cb + p * s_cp + k)
        cand_pred = tl.load(CAND_PRED + b * s_rpb + p * s_rpp + k)
        cand_succ = tl.load(CAND_SUCC + b * s_rsb + p * s_rsp + k)
        pred = tl.load(PRED + prev * s_pred + r)
        proj = tl.load(PROJ + b * s_pb + p * s_pp + r)
        context = (pred.to(tl.float32) * proj.to(tl.float32)).to(CONTEXT_DTYPE).to(tl.float32)
        succ = tl.load(SUCC + cand_succ[:, None] * s_succ + r[None, :]).to(tl.float32)
        pair = tl.sum(succ * context[None, :], 1).to(PAIR_DTYPE).to(tl.float32)
        score = tl.load(UNARY + b * s_ub + p * s_up + k) + pair
        top = tl.max(score, 0)
        first_top = tl.min(tl.where(score == top, k, K), 0)
        scaled = score * inv_t
        e = tl.exp(scaled - tl.max(scaled, 0))
        q = e / tl.sum(e, 0)
        q = tl.where(greedy, tl.where(k == first_top, 1.0, 0.0), q)
        u = tl.load(U + b * STEPS + p)
        index = tl.minimum(tl.sum((tl.cumsum(q, 0) <= u).to(tl.int32), 0), K - 1)
        prev = tl.sum(tl.where(k == index, cand_pred, 0), 0)
        tl.store(Q + b * s_qb + p * s_qp + k, q)
        tl.store(TOK + b * s_tb + p, tl.sum(tl.where(k == index, cand, 0), 0))
        tl.store(SEL + b * s_sb + p, index.to(tl.int64))


_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


def chain(selector, unary, candidates, projected, anchor, temperature, uniform=None) -> Proposal:
    """unary [B, S, K] fp32, candidates [B, S, K] int64, projected [B, S, R] (hidden_projection's output),
    anchor [B] int64, temperature [B] fp32 -> Proposal (tokens, candidates, q [B, S, K] fp32, selected).
    uniform: [B, S] fp32 draws in [0, 1) (tests); by default drawn here from the default generator.
    With pruned codebooks (CandidateSelector.prune) the kernel reads rows through the selector's id -> row tables:
    candidates and the anchor stay vocabulary ids, the tokens it writes are vocabulary ids."""
    batch, steps, k = unary.shape
    rank = projected.shape[-1]
    pred, succ = selector.predecessor_codebook, selector.successor_codebook
    if candidates.shape != unary.shape or projected.shape[:2] != (batch, steps) or anchor.shape != (batch,):
        raise ValueError(f"chain shapes: unary {tuple(unary.shape)}, candidates {tuple(candidates.shape)}, "
                         f"projected {tuple(projected.shape)}, anchor {tuple(anchor.shape)}")
    if unary.dtype != torch.float32 or temperature.dtype != torch.float32 or temperature.shape != (batch,):
        raise ValueError(f"chain dtypes: unary {unary.dtype}, temperature {temperature.dtype} {tuple(temperature.shape)}")
    if pred.shape[1] != rank or succ.shape[1] != rank or k & (k - 1) or rank & (rank - 1):
        raise ValueError(f"chain codebooks {tuple(pred.shape)} / {tuple(succ.shape)}, rank {rank}, top-k {k}")
    # the torch path's roundings: the product in result_type(pred, projected), the einsum in result_type(that, succ)
    context_dtype = torch.result_type(pred, projected)
    pair_dtype = torch.result_type(torch.empty((), dtype=context_dtype), succ)
    for dt in (pred.dtype, succ.dtype, projected.dtype, context_dtype, pair_dtype):
        if dt not in _TL:
            raise ValueError(f"chain: unsupported dtype {dt}")
    if pred.stride(1) != 1 or succ.stride(1) != 1 or projected.stride(-1) != 1 or unary.stride(-1) != 1 \
            or candidates.stride(-1) != 1:
        raise ValueError("chain: inner dimensions must be contiguous")
    if selector.pred_rows is None:
        cand_pred, cand_succ, anchor_row = candidates, candidates, anchor
    else:
        cand_pred, cand_succ = selector.pred_rows[candidates], selector.succ_rows[candidates]
        anchor_row = selector.pred_rows[anchor]
    if uniform is None:
        uniform = torch.empty((batch, steps), dtype=torch.float32, device=unary.device).uniform_()
    elif uniform.shape != (batch, steps) or uniform.dtype != torch.float32 or not uniform.is_contiguous():
        raise ValueError(f"chain: uniform {tuple(uniform.shape)} {uniform.dtype}, need contiguous fp32 [{batch}, {steps}]")
    tokens = torch.empty((batch, steps), dtype=torch.long, device=unary.device)
    selected = torch.empty((batch, steps), dtype=torch.long, device=unary.device)
    q = torch.empty((batch, steps, k), dtype=torch.float32, device=unary.device)
    if batch:
        _chain_kernel[(batch,)](
            unary, projected, candidates, cand_pred, cand_succ, pred, succ, anchor_row, temperature, uniform, tokens, q,
            selected, unary.stride(0), unary.stride(1), projected.stride(0), projected.stride(1),
            candidates.stride(0), candidates.stride(1), cand_pred.stride(0), cand_pred.stride(1),
            cand_succ.stride(0), cand_succ.stride(1), pred.stride(0), succ.stride(0),
            tokens.stride(0), q.stride(0), q.stride(1), selected.stride(0),
            K=k, R=rank, STEPS=steps, TINY=torch.finfo(torch.float32).tiny,
            CONTEXT_DTYPE=_TL[context_dtype], PAIR_DTYPE=_TL[pair_dtype],
            num_warps=4,
        )
    return Proposal(tokens, candidates, q, selected)
