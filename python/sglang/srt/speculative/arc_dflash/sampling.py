"""Selector-conditioned proposal probabilities and exact rejection sampling."""

from dataclasses import dataclass

import torch
from .model import require


def policy_column(value, logits, dtype=torch.float32):
    """A sampling knob as something broadcastable over one row of `logits`.

    A batch mixes policies -- one request at temperature 0, another nucleus
    sampling -- so every knob has to be able to vary per row. Floats stay
    scalars; tensors must already broadcast against `logits[..., :1]`.
    """
    if not torch.is_tensor(value):
        return torch.tensor(value, device=logits.device, dtype=dtype)
    if value.ndim and value.shape != logits.shape[: value.ndim]:
        raise ValueError(f"sampling policy {tuple(value.shape)} does not index rows of {tuple(logits.shape)}")
    return value.to(device=logits.device, dtype=dtype).reshape(*value.shape, *([1] * (logits.ndim - value.ndim)))


def nucleus(descending, top_p):
    """Mask everything past the nucleus. `descending` must already be sorted."""
    probabilities = descending.softmax(-1)
    keep = probabilities.cumsum(-1) - probabilities < top_p
    return descending.masked_fill(~keep, -torch.inf)


def keep_span(top_k):
    """The widest top-k a batch asks for, taken where the knobs are still Python.

    The per-row path sorts once for the whole batch and masks each row back to
    its own k, so it needs this bound before the sort. Reading it off the `top_k`
    tensor costs a device round trip per call -- seven of them per proposal
    chain, inside a compiled region -- and makes the sort width follow the
    traffic, which recompiles. `None` means a row asked for everything, which is
    `distribution`'s own default: the whole last dimension.
    """
    values = list(top_k) if isinstance(top_k, (list, tuple)) else [top_k]
    if not values:
        raise ValueError("a batch has at least one top-k")
    return None if any(k <= 0 for k in values) else max(values)


def distribution(logits, temperature=1.0, top_p=1.0, top_k=-1, *, span=None):
    """The sampling policy, applied row-wise.

    `temperature`, `top_p` and `top_k` are floats/ints for a uniform batch, or
    tensors indexing the leading dimensions of `logits` for a mixed one. Rows at
    temperature 0 come out one-hot on the argmax, whatever the other knobs say.

    `span` is `keep_span` of the batch's k values and only the per-row path uses
    it -- a uniform batch already knows its own k. Leaving it unset sorts the
    whole vocabulary, which is right and, at 248k wide, expensive.
    """
    uniform = not any(torch.is_tensor(x) for x in (temperature, top_p, top_k))
    if uniform:
        if temperature < 0 or not 0 < top_p <= 1 or top_k == 0 or top_k < -1:
            raise ValueError("invalid sampling policy")
        if temperature == 0:
            return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        scores = logits.float() / temperature
        if 0 < top_k < scores.shape[-1]:
            # Nucleus can only ever see the top-k survivors, so sort those k
            # values instead of the whole vocabulary. Sorting 248,320 logits per
            # verify row was 0.83 of the 0.99 ms this call cost; the k-wide sort
            # is 0.17 ms and the result is identical to 7e-9 with the same
            # support. topk already returns descending order.
            values, indices = scores.topk(top_k, -1)
            if top_p < 1:
                values = nucleus(values, top_p)
            return torch.zeros_like(scores).scatter_(-1, indices, values.softmax(-1))
        if top_p < 1:
            values, indices = scores.sort(descending=True, dim=-1)
            return torch.zeros_like(scores).scatter_(-1, indices, nucleus(values, top_p).softmax(-1))
        return scores.softmax(-1)

    raw = logits.float()
    hot = policy_column(temperature, raw)
    nucleus_p = policy_column(top_p, raw)
    width = raw.shape[-1]
    keep_k = policy_column(top_k, raw, torch.long)
    # On device: `.any()` produces a Python bool, and with a per-row policy this
    # check runs once per proposal step inside a compiled region.
    require(
        (hot >= 0).all() & ((nucleus_p > 0) & (nucleus_p <= 1)).all() & ((keep_k != 0) & (keep_k >= -1)).all(),
        "invalid sampling policy",
    )
    greedy = hot == 0
    scores = raw / hot.clamp_min(torch.finfo(raw.dtype).tiny)
    # One topk wide enough for every row, then mask each row back to its own k.
    span = width if span is None else min(span, width)
    if span < 1:
        raise ValueError(f"top-k span {span} must be at least one")
    bounded = keep_k.masked_fill(keep_k < 0, width).clamp(max=width)
    require((bounded <= span).all(), "the top-k span is narrower than a row asked for")
    values, indices = scores.topk(span, -1)
    values = values.masked_fill(torch.arange(span, device=raw.device) >= bounded, -torch.inf)
    values = torch.where(nucleus_p < 1, nucleus(values, nucleus_p), values)
    out = torch.zeros_like(scores).scatter_(-1, indices, values.softmax(-1))
    return torch.where(greedy, torch.zeros_like(raw).scatter_(-1, raw.argmax(-1, keepdim=True), 1.0), out)


@dataclass(frozen=True)
class Proposal:
    tokens: torch.Tensor
    candidates: torch.Tensor
    probabilities: torch.Tensor
    selected_indices: torch.Tensor


def sample_index(probabilities, generator=None):
    """Inverse-CDF categorical draw over the last dimension, one sample per row.

    Returns the first index whose cumulative mass exceeds a uniform draw, which
    is the textbook inverse CDF: a zero-probability entry spans an empty
    interval and can never be picked, and at temperature 0 the one-hot row
    always yields its own index -- including when `torch.rand` returns exactly
    zero, which is why the comparison is `<=` and not `<`.

    `torch.multinomial` is the same distribution but an opaque ATen op: over
    sixteen candidates its kernel costs more than the distribution it samples,
    and it stops inductor fusing the selector chain around it.
    """
    # Built from a tensor rather than a shape: with a batch the leading
    # dimension is symbolic under torch.compile, and `torch.rand` cannot take a
    # symbolic size through the fake-tensor path.
    uniform = torch.empty_like(probabilities[..., :1]).uniform_(generator=generator)
    index = (probabilities.cumsum(-1) <= uniform).sum(-1, keepdim=True)
    # Rounding can leave the final cumulative sum a hair below one.
    return index.clamp_(max=probabilities.shape[-1] - 1)


@torch.no_grad()
def propose_path(model, hidden, anchor, *, temperature=1.0, top_p=1.0, top_k=-1, generator=None):
    selector = model.candidate_selector
    logits = model.lm_head(hidden.reshape(-1, hidden.shape[-1])).view(*hidden.shape[:-1], -1).float()
    unary, candidates = logits.topk(selector.top_k, -1)
    # Nothing in the chain feeds these two, so they run once for all positions
    # instead of once per step. The chain itself was 89% of this function.
    projected = selector.hidden_projection(hidden)
    successor = selector.successor_codebook[candidates]
    previous = anchor
    tokens, probabilities, selected = [], [], []
    for position in range(hidden.shape[1]):
        scores = selector.scores(unary[:, position], projected[:, position], successor[:, position], previous)
        q = distribution(scores, temperature, top_p, top_k, span=scores.shape[-1])
        index = sample_index(q, generator)
        previous = candidates[:, position].gather(-1, index).squeeze(-1)
        tokens.append(previous)
        selected.append(index.squeeze(-1))
        probabilities.append(q)
    return Proposal(torch.stack(tokens, 1), candidates, torch.stack(probabilities, 1), torch.stack(selected, 1))


@torch.no_grad()
def reject_sample(proposal, target_probabilities, *, generator=None):
    """Exact rejection sampling for a batch of requests.

    Takes one request (`tokens` shaped [n], P shaped [n+1, V]) or a batch
    (`tokens` [B, n], P [B, n+1, V]), and returns the accepted proposal count
    and the correction/bonus token with the matching leading shape. The bonus is
    not refreshed here: the target has not read it, so it enters the next block
    as slot zero and is committed by that block's verify. Q is the actual
    conditional selector distribution, never unary softmax.

    Every position of every request is decided in one shot. The sequential form
    branched on the host once per position -- up to fourteen device->host round
    trips per request per cycle, each one idling the GPU. Drawing the uniforms
    up front is exactly equivalent: position i's decision depends only on u_i
    and its own accept probability, and every draw past the first rejection is
    discarded.
    """
    tokens, candidates, q = proposal.tokens, proposal.candidates, proposal.probabilities
    picked = proposal.selected_indices
    single = tokens.ndim == 1
    if single:
        tokens, candidates, q, picked = (x[None] for x in (tokens, candidates, q, picked))
        target_probabilities = target_probabilities[None]
    if tokens.ndim != 2 or candidates.ndim != 3 or q.shape != candidates.shape or picked.shape != tokens.shape:
        raise ValueError("proposal fields must agree on [batch, proposals] and [batch, proposals, candidates]")
    batch, proposals = tokens.shape
    if target_probabilities.shape[:2] != (batch, proposals + 1):
        raise ValueError("need one target distribution per proposal plus the bonus row")
    index = picked[..., None]
    chosen = q.gather(2, index).squeeze(2)
    require((candidates.gather(2, index).squeeze(2) == tokens).all(), "selected token and actual proposal Q disagree")
    require((chosen > 0).all(), "selected token carries zero proposal mass")
    require(((q.sum(-1) - 1).abs() < 1e-4).all(), "invalid conditional proposal distribution")
    rows = target_probabilities[:, :proposals]
    accept = (rows.gather(2, tokens[..., None]).squeeze(2) / chosen).clamp(max=1.0)
    uniform = torch.empty_like(accept).uniform_(generator=generator)
    accepted = (uniform < accept).long().cumprod(1).sum(1)
    # Residual at each request's first rejected position. Clamped so an
    # all-accept request still indexes a real row; `emit` then ignores it.
    first = accepted.clamp(max=proposals - 1)
    where = torch.arange(batch, device=q.device)
    row = rows[where, first]
    residual = (row - torch.zeros_like(row).scatter_add_(1, candidates[where, first], q[where, first])).clamp_min(0)
    total = residual.sum(1, keepdim=True)
    # A genuine rejection means p(token) < q(token), and p and q both sum to
    # one, so some mass must survive. An empty residual is only reachable on an
    # all-accept row, which `emit` does not use.
    require(
        torch.logical_or(accepted == proposals, total.squeeze(1) > 0).all(),
        "rejection with an empty residual distribution",
    )
    correction = residual / total.clamp_min(torch.finfo(total.dtype).tiny)
    emit = torch.where((accepted == proposals)[:, None], target_probabilities[:, proposals], correction)
    drawn = torch.multinomial(emit, 1, generator=generator).squeeze(1)
    return (accepted[0], drawn[0]) if single else (accepted, drawn)
