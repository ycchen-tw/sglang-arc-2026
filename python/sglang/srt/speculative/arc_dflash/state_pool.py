"""Fixed slot pool for the drafter's per-request SWA state, and the CUDA-graphed decode step over it.

Every request owns one slot for its whole life: `window` K/V rows per layer, the newest conv boundaries, the cursor
and its GR, and the pending commit (the last verify block, folded in by the next step). The pool is allocated once,
before the target sizes its KV budget, so the drafter's state memory is a constant: slots x (layers x 2 x window x
kv_heads x head_dim + boundaries) -- 10.1 MiB per slot for the 5-layer drafter. A step rebuilds each window by a
gather of exactly `window` rows (serving.shift_in), so nothing outside the window survives a cycle, and a released
slot is simply reused.

The step reads the batch's slots by an index tensor, so for a fixed batch size every shape is static: one CUDA graph
per captured batch size, padded with a scratch slot (the last one) whose results are discarded. The graph covers the
drafter forward, the window shift, the scatter back into the pool, the 7-step selector chain (sampled with the
default CUDA generator, which graph capture supports) and the dense q for rejection sampling.
"""

from __future__ import annotations

import os

import torch

from . import serving
from .model import CleanCache, LayerCache, require
from .ring_attention import RingWindow
from .sampling import Proposal, distribution, sample_index


@torch.no_grad()
def fast_propose(model, hidden, anchor, temperature):
    """sampling.propose_path with the top-16 taken on the BF16 logits: casting [rows, vocab] to FP32 first is a
    full extra write and read of the logits (0.38 GB at 55 requests x 7). Only the 16 survivors become FP32.
    model.lm_head.topk (heads.py) is the full-vocabulary head on one GPU, or the attention-TP group's sharded one."""
    selector = model.candidate_selector
    unary, candidates = model.lm_head.topk(hidden, selector.top_k)
    unary = unary.float()
    projected = selector.hidden_projection(hidden)
    successor = selector.successor_codebook[candidates]
    ones = torch.ones_like(temperature)
    keep = torch.full_like(temperature, -1, dtype=torch.long)
    previous = anchor
    tokens, probabilities, selected = [], [], []
    for position in range(hidden.shape[1]):
        scores = selector.scores(unary[:, position], projected[:, position], successor[:, position], previous)
        q = distribution(scores, temperature, ones, keep, span=scores.shape[-1])
        index = sample_index(q)
        previous = candidates[:, position].gather(-1, index).squeeze(-1)
        tokens.append(previous)
        selected.append(index.squeeze(-1))
        probabilities.append(q)
    return Proposal(torch.stack(tokens, 1), candidates, torch.stack(probabilities, 1), torch.stack(selected, 1))


class StatePool:
    def __init__(self, model, slots: int, max_batch: int, device, *, steps: int = 7):
        c = model.config
        if not 1 <= steps < c.block_size:
            raise ValueError(f"{steps} proposals per cycle; the drafter's block holds {c.block_size - 1}")
        self.model, self.c, self.slots, self.device, self.steps = model, c, slots, device, steps
        w, b, layers = c.window, c.block_size, c.num_layers
        kv = (layers, slots, c.num_key_value_heads, w, c.head_dim)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.key = torch.zeros(kv, **bf16)
        self.value = torch.zeros(kv, **bf16)
        self.boundary = torch.zeros((layers, 4, slots, 1, c.hidden_size), **bf16)
        self.positions = torch.full((slots, w), -1, dtype=torch.long, device=device)
        self.last_gr = torch.zeros((slots, c.hc_count, c.hidden_size), **bf16)
        self.last_position = torch.zeros(slots, dtype=torch.long, device=device)
        self.pending_tokens = torch.zeros((slots, b), dtype=torch.long, device=device)
        self.pending_gr = torch.zeros((slots, b + 1, c.hc_count, c.hidden_size), **bf16)
        self.pending_length = torch.zeros(slots, dtype=torch.long, device=device)
        self.scratch = slots - 1
        # padding rows read the scratch slot: a blank window ending at its cursor (position 0), which is what
        # decode_step requires of every row; nothing reads its outputs
        self.positions[self.scratch, -1] = 0
        self.free = list(range(slots - 1))
        self.slot_of: dict[str, int] = {}
        # static graph inputs / outputs
        self.max_batch = max_batch
        self.idx = torch.full((max_batch,), self.scratch, dtype=torch.long, device=device)
        self.anchor = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.temperature = torch.ones(max_batch, device=device)
        self.tokens_out = torch.zeros((max_batch, steps), dtype=torch.long, device=device)
        self.probs_out = torch.zeros((max_batch, steps, c.vocab_size), dtype=torch.float32, device=device)
        self.cand_out = torch.zeros((max_batch, steps, c.selector_top_k), dtype=torch.long, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool = None

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (
            self.key, self.value, self.boundary, self.positions, self.last_gr, self.last_position,
            self.pending_tokens, self.pending_gr, self.pending_length, self.probs_out))

    # ---- slots -------------------------------------------------------------------------------------------------
    def acquire(self, rid):
        slot = self.slot_of.get(rid)
        if slot is None:
            if not self.free:
                raise RuntimeError(f"DFlash state pool exhausted ({self.slots - 1} slots, {len(self.slot_of)} live)")
            slot = self.free.pop()
            self.slot_of[rid] = slot
        return slot

    def release(self, rid):
        slot = self.slot_of.pop(rid, None)
        if slot is not None:
            self.pending_length[slot].fill_(0)  # fill_: `t[i] = 0` copies a CPU scalar, a stream sync
            self.free.append(slot)

    def live(self):
        return len(self.slot_of)

    # ---- writes from prefill / verify ----------------------------------------------------------------------------
    def write_request(self, rid, request: serving.DraftRequest):
        """A prefilled DraftRequest (decode_cache layout, no pending commit) into the request's slot."""
        if request.pending is not None:
            raise RuntimeError("prefill state carries no pending commit")
        s = self.acquire(rid)
        cache = request.cache
        for i, layer in enumerate(cache.layers):
            self.key[i, s] = layer.key[0]
            self.value[i, s] = layer.value[0]
            for j in range(4):
                self.boundary[i, j, s] = layer.boundaries[j][0, -1:]
        self.positions[s] = cache.positions[0]
        self.last_gr[s] = request.last_gr[0]
        self.last_position[s] = request.last_position[0]
        self.pending_length[s].fill_(0)
        self.pending_gr[s] = request.last_gr[0]
        self.pending_tokens[s].fill_(0)

    def read_request(self, rid) -> serving.DraftRequest:
        """The slot as a DraftRequest (for continuing a chunked prefill); refuses a pending commit."""
        s = self.slot_of[rid]
        if int(self.pending_length[s]) != 0:
            raise RuntimeError(f"{rid}: a pending commit cannot continue into a prefill chunk")
        layers = tuple(
            LayerCache(self.key[i, s : s + 1].clone(), self.value[i, s : s + 1].clone(),
                       tuple(self.boundary[i, j, s : s + 1].clone() for j in range(4)))
            for i in range(self.c.num_layers)
        )
        return serving.DraftRequest(CleanCache(self.positions[s : s + 1].clone(), layers),
                                    self.last_gr[s : s + 1].clone(), self.last_position[s : s + 1].clone())

    @torch.no_grad()
    def defer_commit(self, slots, tokens, gr_span, positions, lengths):
        """Record verify block [bonus, p1..p(K-1)] with its GR (one predecessor per token plus the cursor's).

        A verify narrower than the drafter's block (K < block_size) is padded to the block: the padding is past
        every row's length, so decode_step treats it as mask slots, exactly like a short accept."""
        b, k = self.c.block_size, tokens.shape[1]
        if not 1 <= k <= b or gr_span.shape[1] != k + 1 or positions.shape[1] != k:
            raise ValueError(f"a commit is at most one block: tokens {tuple(tokens.shape)}, gr {tuple(gr_span.shape)}")
        cursor_gr = gr_span.gather(1, lengths[:, None, None, None].expand(-1, 1, *gr_span.shape[2:]))[:, 0]
        cursor = positions.gather(1, lengths[:, None] - 1)[:, 0]
        if k < b:
            tokens = torch.nn.functional.pad(tokens, (0, b - k))
            gr_span = torch.cat((gr_span, gr_span[:, -1:].expand(-1, b - k, -1, -1)), 1)
        self.pending_tokens[slots] = tokens
        self.pending_gr[slots] = gr_span
        self.pending_length[slots] = lengths
        self.last_gr[slots] = cursor_gr
        self.last_position[slots] = cursor

    # ---- the step -----------------------------------------------------------------------------------------------
    def _step(self, batch):
        """Static-shape body: slots self.idx[:batch] -> proposals in tokens_out / probs_out, windows advanced."""
        model, c = self.model, self.c
        idx = self.idx[:batch]
        prefix = CleanCache(
            self.positions[idx],
            tuple(
                LayerCache(self.key[i][idx], self.value[i][idx], tuple(self.boundary[i, j][idx] for j in range(4)))
                for i in range(c.num_layers)
            ),
        )
        length = self.pending_length[idx]
        cursor = self.last_position[idx]
        first = cursor + 1 - length
        hidden, block, positions = model.decode_step(
            self.pending_tokens[idx], self.anchor[:batch], self.pending_gr[idx], first, prefix,
            width=2 * c.block_size, lengths=length,
        )
        grown = serving.shift_in(prefix, block, positions, length, c.window)
        self.positions[idx] = grown.positions
        for i, layer in enumerate(grown.layers):
            self.key[i][idx] = layer.key
            self.value[i][idx] = layer.value
            for j in range(4):
                self.boundary[i, j][idx] = layer.boundaries[j]
        self.pending_length.index_fill_(0, idx, 0)  # a Python scalar assignment is a host copy, not capturable
        proposal = fast_propose(model, hidden[:, : self.steps], self.anchor[:batch], self.temperature[:batch])
        self._emit(batch, proposal)

    def _emit(self, batch, proposal):
        """Proposals and the dense q: only the 16 x steps entries each row wrote last time are cleared, not the
        whole [batch, steps, vocab] buffer (a 0.77 GB memset per cycle at 110 requests)."""
        self.tokens_out[:batch] = proposal.tokens
        probs = self.probs_out[:batch]
        probs.scatter_(2, self.cand_out[:batch], 0.0)
        probs.scatter_(2, proposal.candidates, proposal.probabilities.float())
        self.cand_out[:batch] = proposal.candidates

    def bucket(self, batch):
        for size in sorted(self.graphs):
            if size >= batch:
                return size
        return None

    @torch.no_grad()
    def capture(self, sizes, target_pool=None, target_stream=None):
        """One graph per batch size. Warm-up runs eagerly first (allocator, autotuners); it only touches the
        scratch slot, as does the captured replay with the default index.

        target_pool / target_stream: capture into the target's graph memory pool, on the target's capture stream
        (the caching allocator hands a freed block back only to the stream that allocated it, so a new stream gets
        none of the target's freed blocks). Safe because the drafter's outputs live in static buffers outside the
        pool (tokens_out / probs_out / cand_out, the state pool) and its replays never overlap the target's."""
        if (target_pool is None) != (target_stream is None):
            raise ValueError("target_pool and target_stream go together")
        if target_pool is not None:
            if self.graph_pool is not None:
                raise RuntimeError("drafter graphs already captured into their own pool")
            self.graph_pool = target_pool
            stream = target_stream
        else:
            stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for size in sorted(sizes, reverse=True):
                self.idx.fill_(self.scratch)
                for _ in range(2):
                    self._step(size)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.graph_pool, stream=stream):
                    self._step(size)
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[size] = graph
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.pending_length[self.scratch] = 0

    @torch.no_grad()
    def step(self, slots, anchor, temperature):
        """slots/anchor/temperature: [batch] device tensors. Returns (tokens [batch, steps], probs [batch, steps, vocab])."""
        batch = slots.shape[0]
        size = self.bucket(batch)
        self.idx.fill_(self.scratch)
        self.idx[:batch] = slots
        self.anchor[:batch] = anchor
        self.temperature[:batch] = temperature
        if size is None:
            self._step(batch)
        else:
            self.graphs[size].replay()
        return self.tokens_out[:batch], self.probs_out[:batch]


class RingStatePool(StatePool):
    """The same pool with each slot's window kept as a ring: position p lives in column p % window.

    Attention masks keys by position (validity and SWA distance), so the order of the window's columns does not
    matter. A step therefore writes only the rows it commits -- at most one block per request -- into their ring
    columns, overwriting exactly the positions that just left the window, instead of gathering the whole window
    out, concatenating, re-gathering and scattering it back (five full-window copies per layer, ~5 ms at 55
    requests). Column `window` is a garbage column: uncommitted block rows are routed there with position -1, so
    the scatter needs no mask and nothing it holds is ever attendable.
    """

    def __init__(self, model, slots: int, max_batch: int, device, *, steps: int = 7, width: int | None = None):
        super().__init__(model, slots, max_batch, device, steps=steps)
        c = self.c
        w = c.window
        # A step forwards [committed <= K, anchor, mask...] and reads proposals at anchor + 0..steps-1, so it needs
        # K + steps = 2 x steps + 1 slots, not the 2 x block the shift layout runs: attention is causal and the
        # convolution has width 2, so the dropped trailing mask slots never reach a kept slot (model.decode_step).
        # window attention: "triton" (ring_attention, straight from the pool) or "sdpa" (gather + spare columns)
        self.attention = os.environ.get("ARC_DFLASH_RING_ATTN", "triton")
        if self.attention not in ("triton", "sdpa"):
            raise ValueError(f"ARC_DFLASH_RING_ATTN={self.attention}: triton or sdpa")
        self.width = 2 * steps + 1 if width is None else width
        if not 2 * steps + 1 <= self.width <= 2 * c.block_size:
            raise ValueError(f"step width {self.width} for {steps} proposals: need 2 x steps + 1 .. 2 x block")
        # columns: `window` ring entries, one garbage column, then `width` spare columns that each step's
        # gathered copy fills with the block's own K/V (fast_drafter.attention_forward), so attention never
        # concatenates; the spare columns of the pool itself are never read
        self.spare = self.width
        kv = (c.num_layers, slots, c.num_key_value_heads, w + 1 + self.spare, c.head_dim)
        self.key = torch.zeros(kv, dtype=torch.bfloat16, device=device)
        self.value = torch.zeros(kv, dtype=torch.bfloat16, device=device)
        self.positions = torch.full((slots, w + 1), -1, dtype=torch.long, device=device)
        # the scratch slot's window "ends" at position 0, the cursor every padding row carries
        self.positions[self.scratch, 0] = 0

    def write_request(self, rid, request):
        if request.pending is not None:
            raise RuntimeError("prefill state carries no pending commit")
        s = self.acquire(rid)
        w = self.c.window
        cache = request.cache
        pos = cache.positions[0]
        valid = pos >= 0
        col = torch.where(valid, pos % w, torch.full_like(pos, w))
        self.positions[s].fill_(-1)
        self.positions[s, col] = torch.where(valid, pos, torch.full_like(pos, -1))
        self.positions[s, w].fill_(-1)
        for i, layer in enumerate(cache.layers):
            self.key[i, s][:, col] = layer.key[0]
            self.value[i, s][:, col] = layer.value[0]
            for j in range(4):
                self.boundary[i, j, s] = layer.boundaries[j][0, -1:]
        self.last_gr[s] = request.last_gr[0]
        self.last_position[s] = request.last_position[0]
        self.pending_length[s].fill_(0)
        self.pending_gr[s] = request.last_gr[0]
        self.pending_tokens[s].fill_(0)

    def read_request(self, rid):
        s = self.slot_of[rid]
        if int(self.pending_length[s]) != 0:
            raise RuntimeError(f"{rid}: a pending commit cannot continue into a prefill chunk")
        w = self.c.window
        ring = self.positions[s, :w]
        order = torch.argsort(ring)  # padding (-1) first, then ascending positions: decode_cache's layout
        layers = tuple(
            LayerCache(self.key[i, s][:, order][None].clone(), self.value[i, s][:, order][None].clone(),
                       tuple(self.boundary[i, j, s : s + 1].clone() for j in range(4)))
            for i in range(self.c.num_layers)
        )
        return serving.DraftRequest(CleanCache(ring[order][None].clone(), layers),
                                    self.last_gr[s : s + 1].clone(), self.last_position[s : s + 1].clone())

    def _step(self, batch):
        """model.decode_step (per-row lengths) over ring windows, then commit only the committed rows."""
        model, c = self.model, self.c
        b, w = c.block_size, c.window
        n = self.width
        idx = self.idx[:batch]
        ring_pos = self.positions[idx]
        if self.attention == "triton":  # ring_attention reads the pool by slot
            kv = [(RingWindow(self.key[i], idx, w + 1), RingWindow(self.value[i], idx, w + 1)) for i in range(c.num_layers)]
        else:  # gathered copies with spare columns (fast_drafter's SDPA path)
            kv = [(self.key[i][idx], self.value[i][idx]) for i in range(c.num_layers)]
        prefix = CleanCache(
            ring_pos,
            tuple(
                LayerCache(*kv[i], tuple(self.boundary[i, j][idx] for j in range(4)))
                for i in range(c.num_layers)
            ),
        )
        length = self.pending_length[idx]
        first = self.last_position[idx] + 1 - length
        slot = torch.arange(n, device=self.device)
        positions = first[:, None] + slot
        require((positions[:, 0] >= 1).all(), "position 0 has no predecessor GR")
        require((ring_pos.max(1).values + 1 == first).all(), "the ring must end where the block starts")
        edge = length[:, None]
        committed = slot < edge
        spoken = slot <= edge
        tokens = self.pending_tokens[idx]
        ids = torch.where(committed, torch.nn.functional.pad(tokens, (0, max(0, n - b)))[:, :n], self.anchor[:batch, None])
        embeds = torch.where(spoken[..., None], model.embed_tokens(ids), model.mask_embedding)
        span = self.pending_gr[idx]
        grs = span.gather(1, slot.minimum(edge)[..., None, None].expand(-1, -1, *span.shape[2:]))
        hidden, block = model(embeds, grs, positions, prefix)
        take = edge + torch.arange(self.steps, device=self.device)
        proposal_hidden = hidden.gather(1, take[..., None].expand(-1, -1, hidden.shape[-1]))
        # commit: committed rows to their ring columns, the rest to the garbage column with position -1
        col = torch.where(committed, positions % w, torch.full_like(positions, w))
        rows = idx[:, None]
        self.positions[rows, col] = torch.where(committed, positions, torch.full_like(positions, -1))
        last = (length - 1).clamp(min=0)[:, None, None]
        has = (length > 0)[:, None, None]
        for i, layer in enumerate(block.layers):
            self.key[i][rows, :, col] = layer.key.transpose(1, 2)
            self.value[i][rows, :, col] = layer.value.transpose(1, 2)
            for j in range(4):
                fresh = layer.boundaries[j].gather(1, last.expand(-1, 1, layer.boundaries[j].shape[-1]))
                self.boundary[i, j][idx] = torch.where(has, fresh, self.boundary[i, j][idx])
        self.positions[:, w] = -1
        self.pending_length.index_fill_(0, idx, 0)
        proposal = fast_propose(model, proposal_hidden, self.anchor[:batch], self.temperature[:batch])
        self._emit(batch, proposal)
