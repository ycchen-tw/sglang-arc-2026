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
