"""Request-local draft state. The drafter's target context lags its cursor by one.

The target only produces state for tokens it has read. The newest emitted token
(last cycle's bonus) has not been read, so it enters the next block as a token
embedding at slot zero and is committed by that block's verify -- the
DFlash/DSpark/NEXTN contract. Requiring target state at the anchor itself would
force a second target forward per cycle for no additional verified proposals.

Every per-cycle quantity here is a device tensor, including how many drafts were
accepted. That is what lets the scheduler overlap the next batch's preparation
with this forward: one `.tolist()` on the accept counts would drain the whole
pipeline back to a synchronous cycle.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from .model import ASYNC_REFRESH, CleanCache, LayerCache, canonical, require, row_of_block
from .sampling import propose_path


def decode_cache(cache, window):
    """The one shape every decode cycle reads: `window` slots ending at the cursor.

    Three differences from the growing trace `refresh` produces. The width is
    constant, so the compiled drafter sees a single specialization from the
    first cycle to the last, and the prefix is the cache itself rather than a
    gather. Short prefixes are left-padded at position -1, which attention
    already excludes. And only the newest boundary survives: `Decoder.forward`
    reads `boundaries[i][:, -1:]` and nothing else, so carrying the history cost
    ten times the cache's live size and copied all of it twice per cycle.
    """
    if cache.positions.shape[0] != 1:
        raise ValueError("a request's decode cache holds exactly one sequence")
    have = cache.positions.shape[1]
    if have < 1:
        raise ValueError("a decode cache needs at least one clean position")
    keep, pad = max(0, have - window), max(0, window - have)
    return CleanCache(
        canonical(F.pad(cache.positions[:, keep:], (pad, 0), value=-1)),
        tuple(
            LayerCache(
                canonical(F.pad(layer.key[:, :, keep:], (0, 0, pad, 0))),
                canonical(F.pad(layer.value[:, :, keep:], (0, 0, pad, 0))),
                tuple(canonical(b[:, -1:]) for b in layer.boundaries),
            )
            for layer in cache.layers
        ),
    )


def shift_in(cache, block, positions, lengths, window):
    """Slide every row's window left by its own commit count and append the block.

    How many tokens a request commits is the accepted draft count, which only
    the GPU knows, so the trim cannot be a slice. Concatenating the block and
    gathering `length + arange(window)` is the same window written without a
    host round trip: row `i` keeps `window - length[i]` old slots and takes
    `length[i]` new ones, and a row that commits nothing keeps its window whole.

    The gather index is one row per request, broadcast over heads and channels
    at stride zero, so it costs no bandwidth beyond the cache itself.
    """
    index = lengths[:, None] + torch.arange(window, device=lengths.device)
    edge = (lengths[:, None] - 1).clamp(min=0)
    committed = (lengths > 0)[:, None, None]
    layers = []
    for old, new in zip(cache.layers, block.layers, strict=True):
        wide = index[:, None, :, None].expand(-1, old.key.shape[1], -1, old.key.shape[3])
        layers.append(
            LayerCache(
                torch.cat((old.key, new.key), 2).gather(2, wide),
                torch.cat((old.value, new.value), 2).gather(2, wide),
                tuple(
                    torch.where(committed, fresh.gather(1, edge[..., None].expand(-1, -1, fresh.shape[-1])), stale)
                    for stale, fresh in zip(old.boundaries, new.boundaries, strict=True)
                ),
            )
        )
    return CleanCache(torch.cat((cache.positions, positions), 1).gather(1, index), tuple(layers))


def emitted_tokens(proposed, accepted, bonus):
    """Each row's accepted proposals followed by its bonus token.

    The obvious two-line Python loop needs `accepted` on the host, and that one
    read is the difference between a scheduler that overlaps the next batch with
    this forward and one that waits for it. The padding column is never selected
    -- a row accepting every proposal still leaves slot `block - 1` for the
    bonus -- so it exists only to make the widths line up.
    """
    if proposed.ndim != 2 or accepted.shape != (proposed.shape[0],) or bonus.shape != accepted.shape:
        raise ValueError("one row of proposals, one accepted count and one bonus token per request")
    slot = torch.arange(proposed.shape[1] + 1, device=proposed.device)
    keep = torch.cat((proposed, proposed.new_zeros(proposed.shape[0], 1)), 1)
    return torch.where(slot < accepted[:, None], keep, proposed.new_zeros(())).scatter(
        1, accepted[:, None], bonus[:, None]
    )


@dataclass(frozen=True)
class PendingCommit:
    """One verify block's tokens, of which the first `length` are real.

    `length` is a device tensor because it is the accepted draft count. Every
    other field is full block width whatever that count turns out to be, so the
    drafter, the compiler and a CUDA graph all see one shape per cycle.
    """

    tokens: torch.Tensor
    gr: torch.Tensor
    positions: torch.Tensor
    length: torch.Tensor


@dataclass(frozen=True)
class DraftRequest:
    """Draft states for every position the target has read, plus that cursor's GR.

    With a pending commit the cache deliberately lags the cursor by `length`
    positions; `advance` closes that gap inside the same forward that proposes
    the next block.
    """

    cache: CleanCache
    last_gr: torch.Tensor
    last_position: torch.Tensor
    pending: "PendingCommit | None" = None

    @classmethod
    @torch.no_grad()
    def prefill(cls, model, tokens, target_gr, positions):
        """target_gr[t] is the GR at positions[t]; token t pairs with target_gr[t-1]."""
        if tokens.shape[1] < 2:
            raise ValueError("prefill needs at least one position with a predecessor")
        cache = None
        for start in range(1, tokens.shape[1], 256):
            end = min(start + 256, tokens.shape[1])
            _, cache = model.refresh(
                tokens[:, start:end], target_gr[:, start - 1 : end - 1], positions[:, start:end], cache
            )
        if cache is None:
            raise ValueError("nonempty target prefill required")
        return cls(decode_cache(cache, model.config.window), target_gr[:, -1].clone(), positions[:, -1].clone())

    @torch.no_grad()
    def propose(self, model, anchor_token, *, temperature=1.0, top_p=1.0, top_k=-1, generator=None):
        """anchor_token is the pending bonus at last_position + 1; the target has not read it."""
        if self.pending is not None:
            raise RuntimeError("propose cannot run with a commit still pending; use advance")
        anchor = self.last_position + 1
        require((self.cache.positions[:, -1] == self.last_position).all(), "the clean cache must reach the cursor")
        hidden = model.propose(anchor_token, self.last_gr, anchor, self.cache)
        return propose_path(
            model, hidden, anchor_token, temperature=temperature, top_p=top_p, top_k=top_k, generator=generator
        )

    @torch.no_grad()
    def advance(self, model, anchor_token, *, temperature=1.0, top_p=1.0, top_k=-1, generator=None, propose=True):
        """Commit the pending tokens and propose the next block in one forward."""
        advanced, proposal = advance_batch(
            model,
            [self],
            anchor_token,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
            propose=propose,
        )
        return advanced[0], proposal

    @torch.no_grad()
    def defer_commit(self, model, committed_tokens, gr, positions, length=None):
        """Record verified tokens without running the drafter; `advance` folds them in."""
        return defer_commit_batch(model, [self], committed_tokens, gr, positions, length)[0]

    @torch.no_grad()
    def refresh_committed(self, model, committed_tokens, gr, positions):
        # Kept for chunked prefill, where there is no proposal block to merge with.
        """gr spans positions[0]-1 .. positions[-1]: one predecessor per token, plus the new cursor.

        Every committed token must already have passed through target verify. No
        masked proposal hidden or cache is accepted here.
        """
        if gr is None or gr.shape[1] != committed_tokens.shape[1] + 1:
            raise ValueError("need one predecessor GR per committed token plus the new cursor GR")
        if ASYNC_REFRESH:
            require((positions[:, 0] == self.last_position + 1).all(), "commit must follow the current cursor")
        elif not torch.equal(positions[:, 0], self.last_position + 1):
            raise ValueError("commit must follow the current cursor")
        if self.pending is not None:
            raise RuntimeError("refresh_committed cannot run with a commit still pending")
        _, cache = model.refresh(committed_tokens, gr[:, :-1], positions, self.cache)
        return DraftRequest(
            decode_cache(cache, model.config.window), gr[:, -1].clone(), positions[:, -1].clone()
        )


@torch.no_grad()
def defer_commit_batch(model, states, committed_tokens, gr, positions, lengths=None):
    """Record a whole batch's verified tokens as one owned block per request.

    The runtime reuses the buffers these come out of the moment the cycle ends,
    so the block has to be copied; copying it once for the batch and handing
    each request a row view costs four kernel launches instead of four per
    request, and at eight requests that per-request version was the largest
    slice of host time left in the cycle.

    `lengths` is the per-row count of real tokens. Left unset every row commits
    all of `committed_tokens`, which is what the tests and the warm-up want; the
    serving path always passes the device tensor so the host never learns it.
    """
    block = model.config.block_size
    rows, k = len(states), committed_tokens.shape[1]
    if committed_tokens.ndim != 2 or committed_tokens.shape[0] != rows:
        raise ValueError("one committed block per request")
    if not 1 <= k <= block:
        raise ValueError("a cycle commits between one token and one block")
    if gr is None or gr.shape[:2] != (rows, k + 1):
        raise ValueError("need one predecessor GR per committed token plus the new cursor GR")
    if positions.shape != (rows, k):
        raise ValueError("one position per committed token")
    for state in states:
        if state.pending is not None:
            raise RuntimeError("a commit is already pending")
    require(
        (positions[:, 0] == torch.cat([state.last_position for state in states]) + 1).all(),
        "commit must follow the current cursor",
    )
    place = positions[:, :1] + torch.arange(block, device=positions.device)
    require((positions == place[:, :k]).all(), "commit positions must be consecutive")
    if lengths is None:
        lengths = torch.full((rows,), k, device=positions.device, dtype=torch.int64)
    elif lengths.shape != (rows,):
        raise ValueError("one commit length per request")
    require(((lengths >= 1) & (lengths <= k)).all(), "a pending commit covers between one token and the block")
    # Rows shorter than the block read the cursor GR from their own length
    # onward, which is exactly what repeating the last row gives them. `F.pad`
    # and `torch.cat` both allocate even when they add nothing, so these four
    # tensors are the batch's owned copy.
    tokens = F.pad(committed_tokens, (0, block - k))
    span = torch.cat((gr, gr[:, -1:].expand(-1, block - k, -1, -1)), 1)
    length = lengths.clone()
    cursor_gr = span.gather(1, length[:, None, None, None].expand(-1, 1, *span.shape[2:]))[:, 0]
    cursor = place.gather(1, length[:, None] - 1)[:, 0]
    return [
        DraftRequest(
            state.cache,
            cursor_gr[row : row + 1],
            cursor[row : row + 1],
            PendingCommit(
                tokens[row : row + 1], span[row : row + 1], place[row : row + 1], length[row : row + 1]
            ),
        )
        for row, state in enumerate(states)
    ]


@torch.no_grad()
def advance_batch(model, states, anchor_tokens, *, temperature=1.0, top_p=1.0, top_k=-1, generator=None, propose=True):
    """One drafter forward for a whole batch of requests.

    Each request owns its clean cache and its pending commit; they meet only
    inside this forward. Every cache is exactly one window wide and every
    pending block exactly one block wide, so the batch is a concatenation and
    the shapes do not follow the traffic -- a width that did would recompile
    mid-request, and we measured those stalls in seconds.

    `temperature`/`top_p`/`top_k` may be per-request tensors: a batch mixes a
    greedy request with a nucleus one and must give each what it asked for.

    `propose=False` returns the advanced states and no proposal. Under expert
    parallelism every rank runs this forward, but only one may sample -- the
    others would draw different tokens -- and the sampling chain sits on the
    critical path of the broadcast that follows, so they skip it. The states do
    not depend on the proposal: the block's proposal slots are discarded and
    only the committed prefix reaches the cache.
    """
    if not states:
        raise ValueError("advancing needs at least one request")
    if anchor_tokens.ndim != 1 or anchor_tokens.shape[0] != len(states):
        raise ValueError("one anchor token per request")
    if anchor_tokens.stride(0) != 1:
        # The anchor is handed to a compiled function as-is, and dynamo guards
        # on its strides: a column view of a [batch, length] tensor is a
        # different specialization from the packed row serving passes, at the
        # same batch size. The warm-up used to build the first kind and traffic
        # the second, so every batch size recompiled on its first live cycle --
        # 0.27 s each, and about 1 s under expert parallelism where both ranks
        # pay it and meet at an all-reduce.
        #
        # `stride(0) != 1`, not `is_contiguous()`: a one-row batch has a size-1
        # leading axis, torch calls any stride on it contiguous, and `.contiguous()`
        # is then a no-op that leaves the guard value alone. That is the same
        # blind spot the clean cache hit, and it left batch 1 -- the single
        # request case -- still recompiling after the other sizes were fixed.
        raise ValueError("the anchor row must be packed; a strided one compiles a second time")
    block, window = model.config.block_size, model.config.window
    tokens, spans, lengths, cursors = [], [], [], []
    for state in states:
        if state.cache.positions.shape[1] != window:
            raise ValueError("a decode cache must be exactly one window wide")
        if state.pending is None:
            tokens.append(anchor_tokens.new_zeros(1, block))
            spans.append(state.last_gr[:, None].expand(-1, block + 1, -1, -1))
            lengths.append(torch.zeros(1, dtype=torch.int64, device=anchor_tokens.device))
        else:
            tokens.append(state.pending.tokens)
            spans.append(state.pending.gr)
            lengths.append(state.pending.length)
        cursors.append(state.last_position)
    length, cursor = torch.cat(lengths), torch.cat(cursors)
    first = cursor + 1 - length
    prefix = CleanCache.stack([state.cache for state in states])
    require((prefix.positions[:, -1] == first - 1).all(), "the clean cache must stop where the block starts")
    hidden, block_cache, positions = model.decode_step(
        torch.cat(tokens), anchor_tokens, torch.cat(spans), first, prefix, width=2 * block, lengths=length,
    )
    grown = shift_in(prefix, block_cache, positions, length, window)
    require((grown.positions[:, -1] == cursor).all(), "advance must leave the clean cache at the cursor")
    # Row views, not copies: a slice of a contiguous batch already carries the
    # strides a freshly allocated row would, and the compiled drafter guards on
    # exactly those.
    advanced = [
        DraftRequest(row_of_block(grown, row), state.last_gr, state.last_position, None)
        for row, state in enumerate(states)
    ]
    if not propose:
        return advanced, None
    return advanced, propose_path(
        model, hidden, anchor_tokens, temperature=temperature, top_p=top_p, top_k=top_k, generator=generator
    )
