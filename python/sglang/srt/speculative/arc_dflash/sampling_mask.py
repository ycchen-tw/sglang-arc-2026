"""arc-forge: the sampling support of DFlash-verified tokens (``return_sampling_mask`` under DFlash).

Rejection sampling against the truncated target distribution ``p_S`` (top-k / top-p renormalized,
``build_dflash_verify_target_probs``) emits every committed token from ``p_S`` -- an accepted draft or the
residual draw alike; the residual ``relu(p_S - q)`` lives inside ``S``. So the support of a committed position is
the positive entries of its ``target_probs`` row, and its behaviour log-probability is ``log p_S(token)``. This
packs both for the rows of the opted-in requests, one row per block position (request-major), in the layout of the
sampler's ``SamplingMaskOutput``; the result processor keeps the rows of the tokens it commits. Everything stays on
the device: the overlap scheduler copies the result to the host with the rest of the batch result.
"""

import torch

from sglang.srt.layers.logits_processor import SamplingMaskOutput, SamplingMaskStatus


def spec_sampling_mask_output(
    target_probs: torch.Tensor,
    out_tokens: torch.Tensor,
    batch_rows: torch.Tensor,
    max_tokens: int,
) -> SamplingMaskOutput:
    """``target_probs`` (bs, D, V) the verify's truncated target distribution, ``out_tokens`` (bs, D) the committed
    block (positions past the commit length hold whatever the block layout puts there; their rows are never
    kept), ``batch_rows`` the opted-in batch indices (device, int64)."""
    if target_probs.dim() != 3 or out_tokens.shape != target_probs.shape[:2]:
        raise ValueError(
            f"target_probs {tuple(target_probs.shape)} and out_tokens {tuple(out_tokens.shape)} "
            "must be (bs, D, V) and (bs, D)"
        )
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    block, vocab = target_probs.shape[1], target_probs.shape[2]
    weights = target_probs.index_select(0, batch_rows).reshape(-1, vocab)
    tokens = out_tokens.index_select(0, batch_rows).reshape(-1).to(torch.int64)
    in_vocab = (tokens >= 0) & (tokens < vocab)
    selected = weights.gather(1, tokens.clamp(0, vocab - 1)[:, None])[:, 0].float()
    support = weights > 0
    mass = torch.where(support, weights, torch.zeros_like(weights)).sum(dim=-1, dtype=torch.float32)
    selected_logprobs = torch.log(selected / mass)
    invalid = ~(in_vocab & (selected > 0) & (mass > 0) & torch.isfinite(selected_logprobs))
    lengths = support.sum(dim=-1, dtype=torch.int32)
    statuses = torch.where(
        invalid,
        SamplingMaskStatus.INVALID,
        torch.where(lengths > max_tokens, SamplingMaskStatus.OVERFLOW, SamplingMaskStatus.OK),
    ).to(torch.int32)
    packed = min(max_tokens, vocab)
    _, positions = torch.topk(weights, k=packed, dim=-1, largest=True, sorted=True)
    return SamplingMaskOutput(
        token_ids=positions.to(torch.int32),
        lengths=lengths.clamp(max=packed),
        selected_logprobs=selected_logprobs,
        statuses=statuses,
        rows_per_request=block,
    )


def eagle_sampling_mask_output(
    target_probs: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    batch_rows: torch.Tensor,
    max_tokens: int,
) -> SamplingMaskOutput:
    """``spec_sampling_mask_output`` for the EAGLE v2 verify (the DFlash drafter, ARC_DFLASH_DRAFTER): committed token
    j of request b is ``predict[accept_index[b, j]]``, drawn from target row ``accept_index[b, j]`` of the flattened
    (bs * D, V) truncated ``target_probs``; -1 entries (not committed) pack whatever row 0 holds, never kept."""
    if accept_index.dim() != 2 or target_probs.dim() != 3:
        raise ValueError(f"accept_index {tuple(accept_index.shape)} / target_probs {tuple(target_probs.shape)}")
    vocab = target_probs.shape[-1]
    flat = target_probs.reshape(-1, vocab)
    rows = accept_index.index_select(0, batch_rows).to(torch.int64).clamp(min=0)
    sel, width = rows.shape
    probs = flat.index_select(0, rows.reshape(-1)).view(sel, width, vocab)
    tokens = predict.to(torch.int64).index_select(0, rows.reshape(-1)).view(sel, width)
    return spec_sampling_mask_output(
        probs, tokens, torch.arange(sel, device=rows.device), max_tokens
    )


def eagle_greedy_sampling_mask_output(
    predict: torch.Tensor, accept_index: torch.Tensor, batch_rows: torch.Tensor
) -> SamplingMaskOutput:
    """A greedy batch's supports: each committed token alone, logprob 0 (the sampler's greedy rule)."""
    rows = accept_index.index_select(0, batch_rows).to(torch.int64).clamp(min=0)
    sel, width = rows.shape
    tokens = predict.to(torch.int32).index_select(0, rows.reshape(-1))
    n = tokens.numel()
    return SamplingMaskOutput(
        token_ids=tokens.view(-1, 1),
        lengths=torch.ones(n, dtype=torch.int32, device=tokens.device),
        selected_logprobs=torch.zeros(n, dtype=torch.float32, device=tokens.device),
        statuses=torch.full((n,), SamplingMaskStatus.OK, dtype=torch.int32, device=tokens.device),
        rows_per_request=width,
    )
