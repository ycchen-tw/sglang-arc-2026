"""Exact chain rejection sampling on vocab-sharded verify logits (ARC_SHARDED_VERIFY=1).

With --enable-dp-lm-head every attention-TP rank holds the logits of its vocab shard. Stock verify gathers them
([rows, vocab] bf16 per rank: 24-32 MB at 96-128 rows on L4, 5.8-19.8 ms of host-mapped gather per verify step on
Runpod 10-08), copies them to FP32, softmaxes the whole vocabulary and runs the chain rejection-sampling kernel
(kernels/ops/speculative/reject_sampling.py) on dense target and draft probabilities -- on every rank, after which rank
0's result is broadcast. Here the target-verify LogitsProcessor keeps its shard and eagle_sample samples from the
shards directly. Same distribution as the kernel, not the same draws.

The kernel, per request with drafts c_1..c_{K-1} and r = the last accepted row (from 0):
  accept c_s  iff  coin_{s-1} * q_r(c_s) < p_r(c_s)          (then r = s)
  final token ~ relu(p_r - q_r) / Z   (first rejection)   or   ~ p_{K-1}   (every draft accepted)
p_r = softmax(logits_r / T). The arc DFlash drafter's q_r lives on the selector's candidates (selector_top_k ids,
worker.draft attaches them to the verify input), so per row only S_r = supp(q_r) + {c_{r+1}} matters beyond the
normaliser. Every rank summarises its shard per row (shard_summary: one pass over the bf16 shard, per-lane
accumulators, then a per-row merge):
  m, s     max and sum of exp(l/T - m)                     -> the global normaliser
  arg      the shard's argmax                              (kept for greedy checks)
  x        one id of the shard minus S_r drawn proportional to p (Gumbel-max, the rank's own Philox stream)
  l(S_r)   the scaled logits at S_r ids the shard owns, -inf elsewhere
The pair all-gathers the summaries (4 + |S_r| floats per row, ~12 KB per step) and every rank runs the accept loop on
p(S_r). The residual relu(p - q) equals p off S_r, so the final draw is a mixture: S_r (relu(p - q) there, inverse CDF
with the final coin) or shard n minus S_r (mass by the normaliser), whose draw is that shard's x, independent of the
coins (chain_verify: one program per request, outputs in the chain kernel's layout). Rank 0's result is
broadcast as before.

Batches this does not cover fall back to gathering the shards first (gather_full) and the stock path: greedy, top-k,
top-p, min-p, penalties, logit bias, grammar, seeded sampling, logprobs, target-only verify, trees. The codex harness
(/v1/responses with --sampling-defaults openai: temperature 0.7, top_p 1, no top_k) takes the sharded path.
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

NEG = float("-inf")
HEAD = 4  # m, s, arg, x; then l(S_r)
LOG_EVERY = 2000
_STATS = {"sharded": 0, "fallback": {}, "logged_at": 0}


def enabled() -> bool:
    v = os.environ.get("ARC_SHARDED_VERIFY", "0")
    if v not in ("0", "1"):
        raise ValueError(f"ARC_SHARDED_VERIFY must be 0 or 1, got {v!r}")
    return v == "1"


def require_supported(logits_processor) -> None:
    """The target LogitsProcessor's construction-time check: only the arc DFlash drafter attaches the sparse q this
    sampler needs, and softcapping would have to run on the shards."""
    from sglang.srt.speculative.arc_dflash import enabled as arc_dflash_enabled

    if not arc_dflash_enabled():
        raise RuntimeError("ARC_SHARDED_VERIFY=1 needs the arc DFlash drafter (ARC_DFLASH_DRAFTER) in the EAGLE worker")
    if logits_processor.final_logit_softcapping:
        raise RuntimeError("ARC_SHARDED_VERIFY=1 does not apply final-logit softcapping on vocab shards")
    if not _STATS.get("install_logged"):
        _STATS["install_logged"] = True
        logger.info("ARC_SHARDED_VERIFY installed: target-verify logits stay sharded over %d attention-TP ranks "
                    "(vocab %d)", logits_processor.attn_tp_size, logits_processor.vocab_size)


@triton.jit
def _row_support(q_ids_ptr, cand_ptr, b, r, stride_qb, stride_qr, stride_cb,
                 K: tl.constexpr, KQ: tl.constexpr, WP: tl.constexpr):
    """S_r of verify row r of request b as [WP] global ids (-1 = none): q_r's support for r < K-1 in columns
    0..KQ-1, then c_{r+1} in column KQ when it is not in that support. Row K-1 (every draft accepted) has none."""
    w = tl.arange(0, WP)
    in_q = (w < KQ) & (r < K - 1)
    ids = tl.load(q_ids_ptr + b * stride_qb + r * stride_qr + w, mask=in_q, other=-1)
    nxt = tl.load(cand_ptr + b * stride_cb + r + 1, mask=r < K - 1, other=-1)
    present = tl.max(((ids == nxt) & in_q).to(tl.int32), 0)
    return tl.where(w == KQ, tl.where((present == 0) & (r < K - 1), nxt, -1), ids)


@triton.jit
def _summary_partial_kernel(
    logits_ptr, stride_row, temp_ptr, q_ids_ptr, cand_ptr, seed_ptr, part_ptr,
    Vs, V, lo, chunk, splits, stride_qb, stride_qr, stride_cb,
    K: tl.constexpr, KQ: tl.constexpr, BLOCK: tl.constexpr,
):
    """One (row, split) of a shard: max, sum exp, first argmax and the Gumbel-max draw over the split minus S_r, with
    per-lane accumulators reduced once. part[row, split] = (m, s, argmax, gumbel max, gumbel argmax), shard-local."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    b = row // K
    r = row % K
    t = tl.load(temp_ptr + b)
    seed = tl.load(seed_ptr)
    has_q = r < K - 1
    nxt = tl.load(cand_ptr + b * stride_cb + r + 1, mask=has_q, other=-1) - lo  # in S_r whether in supp(q_r) or not
    begin = split * chunk
    end = tl.minimum(begin + chunk, Vs)
    m_v = tl.full((BLOCK,), -float("inf"), tl.float32)
    s_v = tl.zeros((BLOCK,), tl.float32)
    a_v = tl.zeros((BLOCK,), tl.int32)
    g_v = tl.full((BLOCK,), -float("inf"), tl.float32)
    i_v = tl.zeros((BLOCK,), tl.int32)
    base = logits_ptr + row.to(tl.int64) * stride_row
    for start in range(begin, end, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.math.div_rn(tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32), t)  # = logits / T
        nm = tl.maximum(m_v, x)
        s_v = s_v * tl.where(nm == -float("inf"), 0.0, tl.exp(m_v - nm)) + tl.where(mask, tl.exp(x - nm), 0.0)
        a_v = tl.where(x > m_v, offs, a_v)
        m_v = nm
        keep = mask & (offs != nxt)
        for j in tl.static_range(KQ):
            keep = keep & (offs != tl.load(q_ids_ptr + b * stride_qb + r * stride_qr + j, mask=has_q, other=-1) - lo)
        u = tl.rand(seed, row * V + lo + offs)
        g = tl.where(keep, x - tl.log(-tl.log(u)), -float("inf"))
        i_v = tl.where(g > g_v, offs, i_v)
        g_v = tl.maximum(g_v, g)
    m = tl.max(m_v, 0)
    s = tl.sum(tl.where(m_v == -float("inf"), 0.0, s_v * tl.exp(m_v - m)), 0)
    am = tl.min(tl.where(m_v == m, a_v, 2147483647), 0)  # each lane keeps its first max: the overall first occurrence
    gm = tl.max(g_v, 0)
    gi = tl.min(tl.where(g_v == gm, i_v, 2147483647), 0)
    out = part_ptr + (row * splits + split) * 5
    tl.store(out + 0, m)
    tl.store(out + 1, s)
    tl.store(out + 2, am.to(tl.float32))
    tl.store(out + 3, gm)
    tl.store(out + 4, gi.to(tl.float32))


@triton.jit
def _summary_finish_kernel(
    part_ptr, logits_ptr, stride_row, temp_ptr, q_ids_ptr, cand_ptr, out_ptr,
    Vs, lo, splits, stride_qb, stride_qr, stride_cb,
    K: tl.constexpr, KQ: tl.constexpr, WP: tl.constexpr, SP: tl.constexpr, PACK: tl.constexpr,
):
    """One row: the splits merged and l(S_r) read from the shard, packed as m, s, argmax, x (global ids), l(S_r)."""
    row = tl.program_id(0)
    b = row // K
    r = row % K
    sp = tl.arange(0, SP)
    live = sp < splits
    part = part_ptr + (row * splits + sp) * 5
    pm = tl.load(part + 0, mask=live, other=-float("inf"))
    ps = tl.load(part + 1, mask=live, other=0.0)
    pa = tl.load(part + 2, mask=live, other=3.0e38)
    pg = tl.load(part + 3, mask=live, other=-float("inf"))
    pi = tl.load(part + 4, mask=live, other=3.0e38)
    m = tl.max(pm, 0)
    s = tl.sum(tl.where(live, ps * tl.exp(pm - m), 0.0), 0)
    am = tl.min(tl.where(pm == m, pa, 3.0e38), 0)  # splits are in id order: the first split holding the max
    gm = tl.max(pg, 0)
    x = tl.min(tl.where(pg == gm, pi, 3.0e38), 0)
    t = tl.load(temp_ptr + b)
    ids = _row_support(q_ids_ptr, cand_ptr, b, r, stride_qb, stride_qr, stride_cb, K, KQ, WP)
    own = (ids >= lo) & (ids < lo + Vs)
    l = tl.load(logits_ptr + row.to(tl.int64) * stride_row + (ids - lo), mask=own, other=-float("inf")).to(tl.float32)
    l = tl.where(own, tl.math.div_rn(l, t), -float("inf"))
    out = out_ptr + row * PACK
    tl.store(out + 0, m)
    tl.store(out + 1, s)
    tl.store(out + 2, am + lo)
    tl.store(out + 3, x + lo)
    w = tl.arange(0, WP)
    tl.store(out + 4 + w, tl.where(w < KQ + 1, l, 0.0), mask=w < PACK - 4)


@triton.jit
def _row_probs(summ_ptr, R, row, ids, N: tl.constexpr, KQ: tl.constexpr, WP: tl.constexpr, PACK: tl.constexpr):
    """p(ids) of one verify row from the N shard summaries, with the log-normaliser terms (max M, Z * exp(-M))."""
    w = tl.arange(0, WP)
    big_m = -float("inf")
    for n in tl.static_range(N):
        big_m = tl.maximum(big_m, tl.load(summ_ptr + (n * R + row) * PACK))
    z = 0.0
    lmax = tl.full((WP,), -float("inf"), tl.float32)
    for n in tl.static_range(N):
        rec = summ_ptr + (n * R + row) * PACK
        z += tl.load(rec + 1) * tl.exp(tl.load(rec) - big_m)
        lmax = tl.maximum(lmax, tl.load(rec + 4 + w, mask=w < KQ + 1, other=-float("inf")))
    return tl.where(ids >= 0, tl.exp(lmax - big_m) / z, 0.0), big_m, z


@triton.jit
def _chain_verify_kernel(
    summ_ptr, R, cand_ptr, q_ids_ptr, q_probs_ptr, ridx_ptr, coins_ptr, coin_final_ptr,
    predict_ptr, accept_index_ptr, num_correct_ptr, Vs,
    stride_cb, stride_qb, stride_qr, stride_pb, stride_pr, stride_rb, stride_kb,
    N: tl.constexpr, K: tl.constexpr, KQ: tl.constexpr, WP: tl.constexpr, PACK: tl.constexpr,
):
    """One request: the chain accept loop on p(S_r), the final draw (S_r residual by inverse CDF, else the chosen
    shard's x), and predict / accept_index / num_correct_drafts as speculative_sampling_classic_kernel writes them."""
    b = tl.program_id(0)
    w = tl.arange(0, WP)
    acc = 0
    alive = 1
    for step in tl.static_range(1, K):
        r = step - 1
        ids = _row_support(q_ids_ptr, cand_ptr, b, r, stride_qb, stride_qr, stride_cb, K, KQ, WP)
        p, big_m, z = _row_probs(summ_ptr, R, b * K + r, ids, N, KQ, WP, PACK)
        q = tl.load(q_probs_ptr + b * stride_pb + r * stride_pr + w, mask=w < KQ, other=0.0)
        hit = ids == tl.load(cand_ptr + b * stride_cb + step)
        ok = (alive != 0) & (tl.load(coins_ptr + b * stride_kb + r) * tl.sum(tl.where(hit, q, 0.0), 0)
                             < tl.sum(tl.where(hit, p, 0.0), 0))
        acc += ok.to(tl.int32)
        alive = ok.to(tl.int32)

    row = b * K + acc
    ids = _row_support(q_ids_ptr, cand_ptr, b, acc, stride_qb, stride_qr, stride_cb, K, KQ, WP)
    p, big_m, z = _row_probs(summ_ptr, R, row, ids, N, KQ, WP, PACK)
    q = tl.load(q_probs_ptr + b * stride_pb + acc * stride_pr + w, mask=(w < KQ) & (acc < K - 1), other=0.0)
    part = tl.where(ids >= 0, tl.maximum(p - q, 0.0), 0.0)
    a = tl.sum(part, 0)
    total = a
    for n in tl.static_range(N):
        rec = summ_ptr + (n * R + row) * PACK
        mine = (ids >= n * Vs) & (ids < (n + 1) * Vs)
        total += tl.maximum(tl.load(rec + 1) * tl.exp(tl.load(rec) - big_m) / z - tl.sum(tl.where(mine, p, 0.0), 0), 0.0)
    u = tl.load(coin_final_ptr + b) * total
    last_col = tl.max(tl.where(part > 0, w, -1), 0)
    k = tl.minimum(tl.sum((tl.cumsum(part, 0) <= u).to(tl.int32), 0), tl.maximum(last_col, 0))
    tok_s = tl.sum(tl.where(w == k, ids, 0), 0)
    cum = a
    tok_x = tl.full((), -1, tl.int64)
    last_x = tl.full((), -1, tl.int64)
    for n in tl.static_range(N):
        rec = summ_ptr + (n * R + row) * PACK
        mine = (ids >= n * Vs) & (ids < (n + 1) * Vs)
        mass = tl.maximum(tl.load(rec + 1) * tl.exp(tl.load(rec) - big_m) / z - tl.sum(tl.where(mine, p, 0.0), 0), 0.0)
        x_n = tl.load(rec + 3).to(tl.int64)
        cum += mass
        tok_x = tl.where((tok_x < 0) & (u < cum) & (mass > 0), x_n, tok_x)
        last_x = tl.where(mass > 0, x_n, last_x)  # rounding can leave u at the very top
    tok_x = tl.where(tok_x < 0, last_x, tok_x)
    tok = tl.where((u < a) & (last_col >= 0), tok_s.to(tl.int64), tok_x)

    for j in tl.static_range(K):
        idx = tl.load(ridx_ptr + b * stride_rb + j)
        tl.store(accept_index_ptr + b * K + j, tl.where(j <= acc, idx, -1).to(tl.int32))
        nxt = tl.load(cand_ptr + b * stride_cb + j + 1, mask=j + 1 < K, other=0)
        tl.store(predict_ptr + idx, tl.where(j < acc, nxt, tl.where(j == acc, tok, 0)).to(tl.int32))
    tl.store(num_correct_ptr + b, acc)


def pack_width(kq: int) -> int:
    """Summary row width in floats (m, s, argmax, x, l(S_r) for |S_r| = kq + 1), padded to a multiple of 4 so that a
    bf16 view of the gather is a multiple of 8 elements."""
    return (HEAD + kq + 1 + 3) // 4 * 4


def shard_summary(logits, temperatures, candidates, q_ids, lo, vocab_size, *, seed=None, splits=None, block=1024):
    """This rank's summary of its shard. logits [bs*K, Vs] (rows contiguous; global ids [lo, lo+Vs)),
    temperatures [bs] (> 0), candidates [bs, K] int64, q_ids [bs, K-1, Kq] int64 (q_r's support). Returns
    [bs*K, pack_width(Kq)] float32: m, s, argmax (global id), x (global id), l(S_r) (scaled, -inf off the shard)."""
    R, Vs = logits.shape
    bs, K = candidates.shape
    kq = q_ids.shape[-1]
    if logits.stride(1) != 1 or R != bs * K or q_ids.shape[:2] != (bs, K - 1):
        raise ValueError(f"shard_summary: logits {tuple(logits.shape)} / candidates {tuple(candidates.shape)} / "
                         f"q_ids {tuple(q_ids.shape)}")
    if R * vocab_size >= 2**31:
        raise ValueError(f"{R} rows x {vocab_size} overflow the int32 Philox offsets")
    dev = logits.device
    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64, device=dev)
    if splits is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        splits = max(1, min(32, (4 * sms + R - 1) // R))
    chunk = triton.cdiv(triton.cdiv(Vs, splits), block) * block
    splits = triton.cdiv(Vs, chunk)
    part = torch.empty((R, splits, 5), dtype=torch.float32, device=dev)
    _summary_partial_kernel[(R, splits)](
        logits, logits.stride(0), temperatures, q_ids, candidates, seed, part, Vs, vocab_size, lo, chunk, splits,
        q_ids.stride(0), q_ids.stride(1), candidates.stride(0), K=K, KQ=kq, BLOCK=block, num_warps=4,
    )
    out = torch.empty((R, pack_width(kq)), dtype=torch.float32, device=dev)
    _summary_finish_kernel[(R,)](
        part, logits, logits.stride(0), temperatures, q_ids, candidates, out, Vs, lo, splits,
        q_ids.stride(0), q_ids.stride(1), candidates.stride(0),
        K=K, KQ=kq, WP=triton.next_power_of_2(kq + 1), SP=triton.next_power_of_2(splits), PACK=out.shape[1],
    )
    return out


def chain_verify(summaries, shard_width, candidates, q_ids, q_probs, retrieve_index, coins, coin_final):
    """summaries [N, bs*K, pack] of the N shards in vocab order (shard n = ids [n*shard_width, (n+1)*shard_width)),
    candidates [bs, K] int64, q_ids / q_probs [bs, K-1, Kq], retrieve_index [bs, K], coins [bs, >= K-1] (column s-1
    decides draft s), coin_final [bs]. Returns predict [bs*K], accept_index [bs, K], num_correct_drafts [bs] (int32),
    distributed as speculative_sampling_classic_kernel's."""
    N, R, pack = summaries.shape
    bs, K = candidates.shape
    kq = q_ids.shape[-1]
    if R != bs * K or pack != pack_width(kq) or not summaries.is_contiguous():
        raise ValueError(f"chain_verify: summaries {tuple(summaries.shape)} for {bs} x {K} rows, Kq {kq}")
    dev = summaries.device
    predict = torch.empty(R, dtype=torch.int32, device=dev)
    accept_index = torch.empty((bs, K), dtype=torch.int32, device=dev)
    num_correct = torch.empty(bs, dtype=torch.int32, device=dev)
    _chain_verify_kernel[(bs,)](
        summaries, R, candidates, q_ids, q_probs, retrieve_index, coins, coin_final,
        predict, accept_index, num_correct, shard_width,
        candidates.stride(0), q_ids.stride(0), q_ids.stride(1), q_probs.stride(0), q_probs.stride(1),
        retrieve_index.stride(0), coins.stride(0),
        N=N, K=K, KQ=kq, WP=triton.next_power_of_2(kq + 1), PACK=pack,
    )
    return predict, accept_index, num_correct


def fallback_reason(verify_input, batch, grammar_mask):
    """Why this batch needs the full logits (None: the sharded path covers it)."""
    from sglang.srt.runtime_context import get_spec

    si = batch.sampling_info
    if si.is_all_greedy:
        return "greedy"
    if si.need_top_k_sampling or si.need_top_p_sampling or si.need_min_p_sampling:
        return "top_k_p"
    if si.acc_additive_penalties is not None or si.acc_scaling_penalties is not None or si.logit_bias is not None:
        return "penalty"
    if grammar_mask is not None:
        return "grammar"
    if si.sampling_seed is not None:
        return "seed"
    if batch.return_logprob:
        return "logprob"
    if not get_spec().speculative_use_rejection_sampling:
        return "target_only"
    if verify_input.tree_topk != 1:
        return "tree"
    return None


def _note(kind):
    if kind is None:
        _STATS["sharded"] += 1
    else:
        _STATS["fallback"][kind] = _STATS["fallback"].get(kind, 0) + 1
    total = _STATS["sharded"] + sum(_STATS["fallback"].values())
    if total - _STATS["logged_at"] >= LOG_EVERY:
        _STATS["logged_at"] = total
        logger.info("ARC_SHARDED_VERIFY steps sharded=%d fallback=%s", _STATS["sharded"], _STATS["fallback"])


def gather_full(logits_output):
    """The stock layout: all-gather the shards into [rows, vocab] FP32 (LogitsProcessor._gather_attn_tp_logits +
    _copy_logits_to_buffer) and drop the shard marker, for the stock sampling path and anything after it."""
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    local = logits_output.next_token_logits
    group = get_attn_tp_group()
    n = group.world_size
    if local.shape[-1] * n != logits_output.next_token_logits_global_vocab_size:
        raise RuntimeError(f"vocab shard {tuple(local.shape)} x {n} != {logits_output.next_token_logits_global_vocab_size}")
    shards = torch.empty((n, *local.shape), dtype=local.dtype, device=local.device)
    group.all_gather_into_tensor(shards, local.contiguous())
    full = shards.permute(1, 0, 2).reshape(local.shape[0], -1).float()
    logits_output.next_token_logits = full
    logits_output.next_token_logits_vocab_start = None
    logits_output.next_token_logits_global_vocab_size = None
    return full


def sample(verify_input, batch, logits_output, grammar_mask):
    """eagle_sample on shard logits: (predict, num_correct_drafts, accept_index), or None after gather_full when the
    batch needs the stock path."""
    reason = fallback_reason(verify_input, batch, grammar_mask)
    _note(reason)
    if reason is not None:
        gather_full(logits_output)
        return None
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    q_ids = getattr(verify_input, "arc_q_ids", None)
    q_probs = getattr(verify_input, "arc_q_probs", None)
    if q_ids is None or q_probs is None:
        raise RuntimeError("ARC_SHARDED_VERIFY: the verify input carries no sparse draft q (arc DFlash drafter only)")
    group = get_attn_tp_group()
    local = logits_output.next_token_logits
    vocab = logits_output.next_token_logits_global_vocab_size
    lo = logits_output.next_token_logits_vocab_start
    bs = len(batch.seq_lens)
    K = verify_input.draft_token_num
    if local.shape[0] != bs * K or local.shape[-1] * group.world_size != vocab or lo != group.rank_in_group * local.shape[-1]:
        raise RuntimeError(f"ARC_SHARDED_VERIFY: shard {tuple(local.shape)} at {lo} of {vocab} on rank "
                           f"{group.rank_in_group}/{group.world_size}, {bs} x {K} verify rows")
    if verify_input.max_tree_depth != K:
        raise RuntimeError(f"ARC_SHARDED_VERIFY: chain verify expects max_tree_depth {K}, got {verify_input.max_tree_depth}")
    candidates = verify_input.draft_token.reshape(bs, K).to(torch.int64).contiguous()
    q_ids = q_ids.contiguous()
    q_probs = q_probs.float().contiguous()
    temps = batch.sampling_info.temperatures.view(-1).float().contiguous()
    mine = shard_summary(local, temps, candidates, q_ids, lo, vocab)
    both = torch.empty((group.world_size, *mine.shape), dtype=mine.dtype, device=mine.device)
    # a byte copy: the bf16 view goes through the host-mapped gather when ARC_HOSTCOMM is on
    group.all_gather_into_tensor(both.view(torch.bfloat16).view(-1), mine.view(torch.bfloat16).view(-1))
    coins = torch.rand_like(candidates, dtype=torch.float32)
    coin_final = torch.rand((bs,), dtype=torch.float32, device=candidates.device)
    predict, accept_index, num_correct = chain_verify(both, local.shape[-1], candidates, q_ids, q_probs,
                                                      verify_input.retrieve_index, coins, coin_final)
    if group.world_size > 1:
        group.broadcast(predict, src=0)
        group.broadcast(accept_index, src=0)
        group.broadcast(num_correct, src=0)
    return predict, num_correct, accept_index
