import torch
import triton
import triton.language as tl


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
            q_val = tl.load(q_ptr, mask=mask, other=0.0)
            # Treat NaN q (degenerate draft rows) as 0: residual falls back to p.
            q_val = tl.where(q_val == q_val, q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same NaN-q guard as pass 1.
                q_val = tl.where(q_val == q_val, q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
    )


# arc (DFlash ARC_DFLASH_SPARSE_Q=1): the same chain rejection sampling with the proposal q given as each row's
# support (selector candidates) and their probabilities instead of a dense [batch, num_slots - 1, vocab] tensor.
# q is zero off the support, so every value the dense kernel reads is reproduced exactly: q(x) for the accept test is
# the support entry equal to x (0 when there is none), and the residual block is p with q subtracted on the support
# lanes only. The residual vector, its reductions and the inverse CDF are the dense kernel's, op for op, so with the
# same p and coins the accept lengths and tokens are the dense kernel's (bitwise on GPU is to be checked, see
# test_arc_dflash_sparse_q.py). The degenerate-residual fallback (VOCAB_SIZE - 1) and the NaN-q guard are kept as
# they are for that comparison; hardening them is a separate change.


@triton.jit
def _sparse_q_block(
    ids_base,
    q_base,
    stride_ids_k,
    stride_q_k,
    v_start,
    v_offsets,
    BLOCK_V: tl.constexpr,
    SUPPORT: tl.constexpr,
):
    """q over vocabulary block [v_start, v_start + BLOCK_V) of one proposal row: zero but on the support ids that
    fall in the block. A support id outside the block costs one scalar compare, not a pass over the block."""
    q_val = tl.zeros([BLOCK_V], dtype=tl.float32)
    for s in tl.static_range(SUPPORT):
        sid = tl.load(ids_base + s * stride_ids_k)
        if (sid >= v_start) & (sid < v_start + BLOCK_V):
            q_val += tl.where(v_offsets == sid, tl.load(q_base + s * stride_q_k), 0.0)
    return q_val


@triton.jit
def speculative_sampling_sparse_q_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftIds,
    DraftQ,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_di_b,
    stride_di_s,
    stride_di_k,
    stride_dq_b,
    stride_dq_s,
    stride_dq_k,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SUPPORT: tl.constexpr,
    SUPPORT_P2: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b
    k_offsets = tl.arange(0, SUPPORT_P2)
    k_mask = k_offsets < SUPPORT

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0

    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        p = tl.load(TargetProbs + offset_prob)
        # q(draft_token): the row's support entry for it (ids are unique within a row: one entry or none)
        ids = tl.load(
            DraftIds + pid * stride_di_b + cur_prob_row * stride_di_s + k_offsets * stride_di_k,
            mask=k_mask,
            other=-1,
        )
        qs = tl.load(
            DraftQ + pid * stride_dq_b + cur_prob_row * stride_dq_s + k_offsets * stride_dq_k,
            mask=k_mask,
            other=0.0,
        )
        q = tl.sum(tl.where(ids == draft_token, qs, 0.0), axis=0)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # As in the dense kernel: on all-accept cur_prob_row == NUM_SLOTS - 1 has no q row, and only the rejection
    # branch below dereferences these pointers (cur_prob_row <= NUM_SLOTS - 2 there).
    di_base_ptr_safe = DraftIds + (pid * stride_di_b) + (cur_prob_row * stride_di_s)
    dq_base_ptr_safe = DraftQ + (pid * stride_dq_b) + (cur_prob_row * stride_dq_s)

    # Pass 1: Sum
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE

        p_ptr = tp_base_ptr + v_offsets * stride_tp_v
        p_val = tl.load(p_ptr, mask=mask, other=0.0)

        if all_drafts_accepted:
            val = p_val
        else:
            q_val = _sparse_q_block(
                di_base_ptr_safe, dq_base_ptr_safe, stride_di_k, stride_dq_k,
                v_start, v_offsets, BLOCK_V, SUPPORT,
            )
            # Same NaN-q guard as the dense kernel.
            q_val = tl.where(q_val == q_val, q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)

        norm_sum += tl.sum(val)

    # Pass 2: CDF (the dense kernel's, including its VOCAB_SIZE - 1 fallback on a zero residual).
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_val = _sparse_q_block(
                    di_base_ptr_safe, dq_base_ptr_safe, stride_di_k, stride_dq_k,
                    v_start, v_offsets, BLOCK_V, SUPPORT,
                )
                q_val = tl.where(q_val == q_val, q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_sparse_q_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_ids,
    draft_q,
):
    """chain_speculative_sampling_triton (topk=1 chain) with q given sparsely.

    draft_ids / draft_q: [batch, num_slots - 1, support]; row j is the proposal distribution of candidates[:, j + 1]
    as target-vocab ids (unique within a row) and their probabilities, q being zero elsewhere. Shapes and dtypes are
    checked here (host metadata only, no device sync); id range and uniqueness are the producer's contract.
    """
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    if tuple(target_probs.shape) != (batch_size, num_slots, vocab_size):
        raise ValueError(
            f"target_probs {tuple(target_probs.shape)} for candidates {tuple(candidates.shape)}"
        )
    if (
        draft_ids.ndim != 3
        or draft_ids.shape != draft_q.shape
        or tuple(draft_ids.shape[:2]) != (batch_size, num_slots - 1)
        or draft_ids.shape[2] < 1
    ):
        raise ValueError(
            f"sparse q ids {tuple(draft_ids.shape)} / probs {tuple(draft_q.shape)} for "
            f"{batch_size} rows of {num_slots - 1} proposals"
        )
    if draft_q.dtype != torch.float32 or draft_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"sparse q dtypes ids {draft_ids.dtype} / probs {draft_q.dtype}")
    support = draft_ids.shape[2]

    grid = (batch_size,)
    speculative_sampling_sparse_q_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_ids,
        draft_q,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_ids.stride(0),
        draft_ids.stride(1),
        draft_ids.stride(2),
        draft_q.stride(0),
        draft_q.stride(1),
        draft_q.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        SUPPORT=support,
        SUPPORT_P2=triton.next_power_of_2(support),
    )


# ---- arc-forge ARC_RS_FUSED_SOFTMAX: the sparse-q chain reading the target logits ---------------------------------
# softmax(logits / T) is never written: a stats pass keeps each row's max M and sum S of exp(logits / T - M), and the
# chain evaluates p = exp(logits / T - M) / S where it reads a probability (the accept test's one entry per step, the
# final sampling's two passes over one row). Replaces three passes over [bs x K, vocab] FP32 (divide, softmax read and
# write) by one read. Same distribution; not bitwise the torch softmax (different exp and summation order).


@triton.jit
def _rs_row_softmax_stats_kernel(
    Logits,
    Temperatures,
    RowMax,
    RowSum,
    stride_l_b,
    stride_l_s,
    stride_l_v,
    ROWS_PER_REQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // ROWS_PER_REQ
    t = tl.load(Temperatures + req)
    base = Logits + req * stride_l_b + (row % ROWS_PER_REQ) * stride_l_s
    m = -float("inf")
    s = 0.0
    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        x = tl.load(base + v_offsets * stride_l_v, mask=v_offsets < VOCAB_SIZE, other=-float("inf"))
        x = tl.math.div_rn(x, t)
        new_m = tl.maximum(m, tl.max(x, axis=0))
        # while every logit so far is -inf (a masked row: ARC_THINK_BUDGET forcing </think>, a vocab mask), new_m is
        # -inf and exp(-inf - -inf) is NaN; nothing has been summed yet, so the sum stays 0
        s = tl.where(new_m > -float("inf"), s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0), 0.0)
        m = new_m
    tl.store(RowMax + row, m)
    tl.store(RowSum + row, s)


@triton.jit
def speculative_sampling_sparse_q_logits_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    Logits,
    Temperatures,
    RowMax,
    RowSum,
    DraftIds,
    DraftQ,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_l_b,
    stride_l_s,
    stride_l_v,
    stride_di_b,
    stride_di_s,
    stride_di_k,
    stride_dq_b,
    stride_dq_s,
    stride_dq_k,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SUPPORT: tl.constexpr,
    SUPPORT_P2: tl.constexpr,
):
    """speculative_sampling_sparse_q_kernel with TargetProbs[pid, row, v] = exp(Logits[pid, row, v] / T[pid] -
    RowMax[pid * NUM_SLOTS + row]) / RowSum[pid * NUM_SLOTS + row]; control flow and arithmetic otherwise the same."""
    pid = tl.program_id(0)
    cur_prob_row = 0
    t = tl.load(Temperatures + pid)

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b
    k_offsets = tl.arange(0, SUPPORT_P2)
    k_mask = k_offsets < SUPPORT

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        row_m = tl.load(RowMax + pid * NUM_SLOTS + cur_prob_row)
        row_s = tl.load(RowSum + pid * NUM_SLOTS + cur_prob_row)
        logit = tl.load(Logits + pid * stride_l_b + cur_prob_row * stride_l_s + draft_token * stride_l_v)
        p = tl.exp(tl.math.div_rn(logit, t) - row_m) / row_s
        ids = tl.load(
            DraftIds + pid * stride_di_b + cur_prob_row * stride_di_s + k_offsets * stride_di_k,
            mask=k_mask,
            other=-1,
        )
        qs = tl.load(
            DraftQ + pid * stride_dq_b + cur_prob_row * stride_dq_s + k_offsets * stride_dq_k,
            mask=k_mask,
            other=0.0,
        )
        q = tl.sum(tl.where(ids == draft_token, qs, 0.0), axis=0)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    all_drafts_accepted = continue_verifying
    coin_final = tl.load(UniformSamplesFinal + pid)
    norm_sum = 0.0

    l_base_ptr = Logits + (pid * stride_l_b) + (cur_prob_row * stride_l_s)
    row_m = tl.load(RowMax + pid * NUM_SLOTS + cur_prob_row)
    row_s = tl.load(RowSum + pid * NUM_SLOTS + cur_prob_row)
    di_base_ptr_safe = DraftIds + (pid * stride_di_b) + (cur_prob_row * stride_di_s)
    dq_base_ptr_safe = DraftQ + (pid * stride_dq_b) + (cur_prob_row * stride_dq_s)

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < VOCAB_SIZE
        x = tl.load(l_base_ptr + v_offsets * stride_l_v, mask=mask, other=-float("inf"))
        p_val = tl.exp(tl.math.div_rn(x, t) - row_m) / row_s
        if all_drafts_accepted:
            val = p_val
        else:
            q_val = _sparse_q_block(
                di_base_ptr_safe, dq_base_ptr_safe, stride_di_k, stride_dq_k,
                v_start, v_offsets, BLOCK_V, SUPPORT,
            )
            q_val = tl.where(q_val == q_val, q_val, 0.0)
            diff = p_val - q_val
            val = tl.where(diff > 0.0, diff, 0.0)
        norm_sum += tl.sum(val)

    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE
            x = tl.load(l_base_ptr + v_offsets * stride_l_v, mask=mask, other=-float("inf"))
            p_val = tl.exp(tl.math.div_rn(x, t) - row_m) / row_s
            if all_drafts_accepted:
                val = p_val
            else:
                q_val = _sparse_q_block(
                    di_base_ptr_safe, dq_base_ptr_safe, stride_di_k, stride_dq_k,
                    v_start, v_offsets, BLOCK_V, SUPPORT,
                )
                q_val = tl.where(q_val == q_val, q_val, 0.0)
                diff = p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


def chain_speculative_sampling_sparse_q_from_logits(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    logits,
    temperatures,
    draft_ids,
    draft_q,
):
    """chain_speculative_sampling_sparse_q_triton on target_probs = softmax(logits / temperatures) without
    materializing it (ARC_RS_FUSED_SOFTMAX). logits: [batch, num_slots, vocab] FP32 (any strides); temperatures: [batch] or [batch, 1] FP32, one per request. Shapes and dtypes are checked here (host
    metadata only)."""
    batch_size, num_slots = candidates.shape
    vocab_size = logits.shape[-1]
    if logits.dtype != torch.float32 or tuple(logits.shape) != (batch_size, num_slots, vocab_size):
        raise ValueError(f"logits {logits.dtype} {tuple(logits.shape)} for candidates {tuple(candidates.shape)}")
    temperatures = temperatures.reshape(-1)
    if temperatures.dtype != torch.float32 or temperatures.shape[0] != batch_size:
        raise ValueError(f"temperatures {temperatures.dtype} {tuple(temperatures.shape)} for {batch_size} requests")
    if (
        draft_ids.ndim != 3
        or draft_ids.shape != draft_q.shape
        or tuple(draft_ids.shape[:2]) != (batch_size, num_slots - 1)
        or draft_ids.shape[2] < 1
    ):
        raise ValueError(
            f"sparse q ids {tuple(draft_ids.shape)} / probs {tuple(draft_q.shape)} for "
            f"{batch_size} rows of {num_slots - 1} proposals"
        )
    if draft_q.dtype != torch.float32 or draft_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"sparse q dtypes ids {draft_ids.dtype} / probs {draft_q.dtype}")
    support = draft_ids.shape[2]
    rows = batch_size * num_slots
    row_max = torch.empty(rows, dtype=torch.float32, device=logits.device)
    row_sum = torch.empty(rows, dtype=torch.float32, device=logits.device)
    _rs_row_softmax_stats_kernel[(rows,)](
        logits, temperatures, row_max, row_sum, logits.stride(0), logits.stride(1), logits.stride(2),
        ROWS_PER_REQ=num_slots, VOCAB_SIZE=vocab_size, BLOCK_V=4096,
    )
    speculative_sampling_sparse_q_logits_kernel[(batch_size,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        logits,
        temperatures,
        row_max,
        row_sum,
        draft_ids,
        draft_q,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        draft_ids.stride(0),
        draft_ids.stride(1),
        draft_ids.stride(2),
        draft_q.stride(0),
        draft_q.stride(1),
        draft_q.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=4096,
        SUPPORT=support,
        SUPPORT_P2=triton.next_power_of_2(support),
    )
