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
default CUDA generator, which graph capture supports) and the q for rejection sampling: by default scattered into a
dense [max_batch, steps, vocab] FP32 buffer (0.71 GiB at 110 requests x 7, charged against the target's KV budget);
with sparse_q (ARC_DFLASH_SPARSE_Q=1) only each position's selector candidates and their probabilities,
[max_batch, steps, selector_top_k], and no dense buffer is allocated.
"""

from __future__ import annotations

import os

import torch

from . import serving
from .model import ASYNC_REFRESH, CleanCache, LayerCache, require
from .ring_attention import RingWindow
from .sampling import Proposal, distribution, sample_index


# arc-forge ARC_DFLASH_FUSED_PROPOSE (opt-in, default 0): the selector chain below as one Triton kernel
# (fused_propose.py); a different but identically distributed draw, q still the draw's exact distribution
FUSED_PROPOSE = os.environ.get("ARC_DFLASH_FUSED_PROPOSE", "0")
if FUSED_PROPOSE not in ("0", "1"):
    raise ValueError(f"ARC_DFLASH_FUSED_PROPOSE must be 0 or 1, got {FUSED_PROPOSE!r}")


# arc-forge ARC_DFLASH_BATCHED_COMMIT (opt-in, default 0): RingStatePool._step commits the block's K/V of all layers
# with one index_put each (stacked over layers) and every layer's four conv boundaries with one gather / where /
# index_put over the stacked rows, instead of 2 + 4 x 4 small kernels per layer. Same values to the same places
# (bitwise: scripts/arc/dflash_propose_bench.py state).
BATCHED_COMMIT = os.environ.get("ARC_DFLASH_BATCHED_COMMIT", "0")
if BATCHED_COMMIT not in ("0", "1"):
    raise ValueError(f"ARC_DFLASH_BATCHED_COMMIT must be 0 or 1, got {BATCHED_COMMIT!r}")


# arc-forge ARC_DFLASH_KV_DTYPE (opt-in, default bf16): the dtype of the ring pool's window K/V. fp8_e4m3 stores them
# as e4m3 with no scale (clamped to +-448, as sglang's fp8_e4m3 KV without checkpoint scales), half the window bytes;
# ring_attention widens each tile to BF16 on load, so attention computes with exactly the rounded values, and every
# read that leaves the pool (read_request, the SDPA gather) widens them the same way. Lossy: only the proposals change
# (rejection sampling keeps the output target-distributed), judged by accept length. Ring layout only.
KV_DTYPES = {"bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn}
KV_DTYPE = os.environ.get("ARC_DFLASH_KV_DTYPE", "bf16")
if KV_DTYPE not in KV_DTYPES:
    raise ValueError(f"ARC_DFLASH_KV_DTYPE must be one of {sorted(KV_DTYPES)}, got {KV_DTYPE!r}")
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def to_pool(x, dtype):
    """K/V rows on their way into a pool of `dtype` (a cast that saturates instead of making NaN for e4m3)."""
    if dtype == torch.bfloat16:
        return x
    return x.clamp(-FP8_MAX, FP8_MAX).to(dtype)


@torch.no_grad()
def fast_propose(model, hidden, anchor, temperature):
    """sampling.propose_path with the top-16 taken on the BF16 logits: casting [rows, vocab] to FP32 first is a
    full extra write and read of the logits (0.38 GB at 55 requests x 7). Only the 16 survivors become FP32.
    model.lm_head.topk (heads.py) is the full-vocabulary head on one GPU, or the attention-TP group's sharded one."""
    selector = model.candidate_selector
    unary, candidates = model.lm_head.topk(hidden, selector.top_k)
    unary = unary.float()
    projected = selector.hidden_projection(hidden)
    if FUSED_PROPOSE == "1":
        from .fused_propose import chain

        return chain(selector, unary, candidates, projected, anchor, temperature)
    successor = selector.succ_lookup(candidates)
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
    # Config.context "inject" (the plain drafter's injected context) is implemented by RingStatePool only
    supports_inject = False

    def __init__(self, model, slots: int, max_batch: int, device, *, steps: int = 7, sparse_q: bool = False):
        c = model.config
        if not 1 <= steps < c.block_size:
            raise ValueError(f"{steps} proposals per cycle; the drafter's block holds {c.block_size - 1}")
        # context "inject": a slot holds the injected K/V window and no conv boundaries (self.boundary is None)
        self.inject = c.context == "inject"
        if self.inject and not self.supports_inject:
            raise NotImplementedError(f"{type(self).__name__} does not implement the injected context (drafter config "
                                      "context 'inject'); use the ring pool (ARC_DFLASH_POOL=ring)")
        self.model, self.c, self.slots, self.device, self.steps = model, c, slots, device, steps
        self.sparse_q = sparse_q
        if KV_DTYPE != "bf16" and type(self) is StatePool:
            raise ValueError(f"ARC_DFLASH_KV_DTYPE={KV_DTYPE} needs the ring pool (ARC_DFLASH_POOL=ring)")
        w, b, layers = c.window, c.block_size, c.num_layers
        kv = (layers, slots, c.num_key_value_heads, w, c.head_dim)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.key = torch.zeros(kv, **bf16)
        self.value = torch.zeros(kv, **bf16)
        self.boundary = None if self.inject else torch.zeros((layers, 4, slots, 1, c.hidden_size), **bf16)
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
        # the q of the last step: cand_out holds each position's candidates (target-vocab ids, unique: a top-k);
        # dense mode scatters their probabilities into probs_out, sparse mode keeps them in cand_probs_out
        self.cand_out = torch.zeros((max_batch, steps, c.selector_top_k), dtype=torch.long, device=device)
        if sparse_q:
            self.probs_out = None
            self.cand_probs_out = torch.zeros((max_batch, steps, c.selector_top_k), dtype=torch.float32, device=device)
        else:
            self.probs_out = torch.zeros((max_batch, steps, c.vocab_size), dtype=torch.float32, device=device)
            self.cand_probs_out = None
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool = None

    def bytes(self):
        q = (self.cand_out, self.cand_probs_out) if self.sparse_q else (self.probs_out,)
        boundary = () if self.inject else (self.boundary,)
        return sum(t.numel() * t.element_size() for t in (
            self.key, self.value, *boundary, self.positions, self.last_gr, self.last_position,
            self.pending_tokens, self.pending_gr, self.pending_length, *q))

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
            self.free_slot(slot)

    def free_slot(self, slot):
        """A slot no request holds (a released or a parked one) back to the free list."""
        if slot in self.free or slot == self.scratch or slot in self.slot_of.values():
            raise RuntimeError(f"DFlash: freeing slot {slot}, which is free, the scratch slot or held")
        self.pending_length[slot].fill_(0)  # fill_: `t[i] = 0` copies a CPU scalar, a stream sync
        self.free.append(slot)

    # ---- parked slots (worker ARC_DFLASH_SESSION_PARK): a finished request's window kept for its session ------------
    def detach(self, rid) -> int:
        """Unmap a finished request from its slot without freeing it; the caller owns the slot from here on."""
        return self.slot_of.pop(rid)

    def attach(self, rid, slot):
        """Map a request to a parked slot (after `settle`)."""
        if rid in self.slot_of or slot in self.free or slot == self.scratch:
            raise RuntimeError(f"DFlash: attaching {rid} to slot {slot}: the request holds a slot or the slot is free")
        self.slot_of[rid] = slot

    def snapshot(self, slot, stream):
        """ARC_DFLASH_REFRESH_ASYNC: (cursor, pending length, window end) of a slot, copied to pinned host memory on
        `stream` (the forward stream, so after every cycle already launched for the slot) without waiting; `settle`
        reads it. Nothing writes a parked slot between the two."""
        if self.positions.device.type != "cuda":
            return torch.stack((self.last_position[slot], self.pending_length[slot], self.positions[slot].max())), None
        with torch.cuda.stream(stream):
            values = torch.stack((self.last_position[slot], self.pending_length[slot], self.positions[slot].max()))
            host = torch.empty(3, dtype=values.dtype, pin_memory=True)
            host.copy_(values, non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
        return host, done

    def settle(self, slot, snapshot=None, upto=None):
        """A parked slot as prefill leaves one: no pending commit, the cursor at the end of the clean window.

        The pending rows are the finished request's last verify block, or, under overlap scheduling, the block of the
        one cycle a finished request still runs before the scheduler sees it finish (whose step already folded the
        last verify in). Either way they are dropped, not folded in: the window stays exactly what the drafter
        committed, and the next request appends whatever lies past it. The GR at the window's end is the pending
        block's first predecessor. Returns that end (one host sync, or none from a `snapshot` taken at park time)
        and the fold: None, or -- with `upto` (session exact resume: the finished sequence S is forwarded through
        len(S) - 2) past the window's end -- (tokens, gr) of the pending rows end + 1 .. min(upto, cursor), copied
        before they are dropped, for the caller to commit (gr: one predecessor per token plus the new cursor's)."""
        if snapshot is None:
            last, pending, end = torch.stack(
                (self.last_position[slot], self.pending_length[slot], self.positions[slot].max())
            ).tolist()
        else:
            host, done = snapshot
            if done is not None:
                done.synchronize()  # long landed: the copy was queued when the slot was parked
            last, pending, end = host.tolist()
        if end != last - pending:
            raise RuntimeError(f"DFlash: parked slot {slot}: window ends at {end}, cursor {last}, {pending} pending")
        fold = None
        if upto is not None and upto > end and pending > 0:
            m = min(upto, last) - end
            fold = self.pending_tokens[slot, :m].clone(), self.pending_gr[slot, : m + 1].clone()
        gr = self.pending_gr[slot, 0].clone()
        self.last_gr[slot] = gr
        self.pending_gr[slot] = gr  # a step reads row 0 as the cursor's GR when nothing is pending
        self.last_position[slot].fill_(end)
        self.pending_length[slot].fill_(0)
        self.pending_tokens[slot].fill_(0)
        return end, fold

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
        if ASYNC_REFRESH:
            require(self.pending_length[s] == 0, f"{rid}: a pending commit cannot continue into a prefill chunk")
        elif int(self.pending_length[s]) != 0:
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
        """Static-shape body: slots self.idx[:batch] -> proposals in tokens_out, their q in probs_out (or
        cand_out / cand_probs_out), windows advanced."""
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
        """Proposals and their q. Dense: only the 16 x steps entries each row wrote last time are cleared, not the
        whole [batch, steps, vocab] buffer (a 0.77 GB memset per cycle at 110 requests). Sparse: the candidates and
        their probabilities as they are -- the same q, zero off the candidates."""
        self.tokens_out[:batch] = proposal.tokens
        if self.sparse_q:
            self.cand_out[:batch] = proposal.candidates
            self.cand_probs_out[:batch] = proposal.probabilities.float()
            return
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
        pool (tokens_out / probs_out or cand_probs_out / cand_out, the state pool) and its replays never overlap the
        target's."""
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
        """slots/anchor/temperature: [batch] device tensors. Returns (tokens [batch, steps], q): q is probs
        [batch, steps, vocab], or with sparse_q (candidate ids, their probabilities), each [batch, steps, top_k].
        All are views of static buffers the next step overwrites: consume them before it (verify does)."""
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
        if self.sparse_q:
            return self.tokens_out[:batch], (self.cand_out[:batch], self.cand_probs_out[:batch])
        return self.tokens_out[:batch], self.probs_out[:batch]


class RingStatePool(StatePool):
    """The same pool with each slot's window kept as a ring: position p lives in column p % window.

    Attention masks keys by position (validity and SWA distance), so the order of the window's columns does not
    matter. A step therefore writes only the rows it commits -- at most one block per request -- into their ring
    columns, overwriting exactly the positions that just left the window, instead of gathering the whole window
    out, concatenating, re-gathering and scattering it back (five full-window copies per layer, ~5 ms at 55
    requests). Column `window` is a garbage column: uncommitted block rows are routed there with position -1, so
    the scatter needs no mask and nothing it holds is ever attendable.

    Config.context "inject" (the plain drafter): the window holds each committed token's injected K/V and no conv
    boundaries; a step commits first and then runs the trunk over the proposal block alone (_step_injected).
    """

    supports_inject = True

    def __init__(self, model, slots: int, max_batch: int, device, *, steps: int = 7, width: int | None = None,
                 sparse_q: bool = False):
        super().__init__(model, slots, max_batch, device, steps=steps, sparse_q=sparse_q)
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
        # concatenates; the spare columns of the pool itself are never read. Context "inject": the trunk's block is
        # the `steps` proposal rows (_step_injected), so that many spare columns.
        self.spare = self.steps if self.inject else self.width
        kv = (c.num_layers, slots, c.num_key_value_heads, w + 1 + self.spare, c.head_dim)
        self.kv_dtype = KV_DTYPES[KV_DTYPE]
        self.key = torch.zeros(kv, dtype=self.kv_dtype, device=device)
        self.value = torch.zeros(kv, dtype=self.kv_dtype, device=device)
        self.positions = torch.full((slots, w + 1), -1, dtype=torch.long, device=device)
        # the scratch slot's window "ends" at position 0, the cursor every padding row carries
        self.positions[self.scratch, 0] = 0

    def _commit_batched(self, block, idx, rows, col, last, has):
        """ARC_DFLASH_BATCHED_COMMIT: the per-layer commit loop of _step with the layers stacked."""
        layers = block.layers
        if len(layers) != self.c.num_layers:
            raise RuntimeError(f"block has {len(layers)} layers, the pool {self.c.num_layers}")
        layer_ids = torch.arange(len(layers), device=self.device)[:, None, None]
        # pool [L, slots, Hkv, columns, D] indexed on (layer, slot, column): the value is [L, B, n, Hkv, D]
        keys = torch.stack([layer.key for layer in layers]).transpose(2, 3)
        values = torch.stack([layer.value for layer in layers]).transpose(2, 3)
        self.key[layer_ids, rows[None], :, col[None]] = to_pool(keys, self.kv_dtype)
        self.value[layer_ids, rows[None], :, col[None]] = to_pool(values, self.kv_dtype)
        fresh = torch.stack([b.gather(1, last.expand(-1, 1, b.shape[-1])) for layer in layers for b in layer.boundaries])
        fresh = fresh.view(len(layers), 4, *fresh.shape[1:])  # [L, 4, B, 1, H]
        self.boundary[:, :, idx] = torch.where(has, fresh, self.boundary[:, :, idx])

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
            self.key[i, s][:, col] = to_pool(layer.key[0], self.kv_dtype)
            self.value[i, s][:, col] = to_pool(layer.value[0], self.kv_dtype)
            if self.inject:
                if layer.boundaries:
                    raise RuntimeError("an injected context window carries no conv boundaries")
                continue
            for j in range(4):
                self.boundary[i, j, s] = layer.boundaries[j][0, -1:]
        self.last_gr[s] = request.last_gr[0]
        self.last_position[s] = request.last_position[0]
        self.pending_length[s].fill_(0)
        self.pending_gr[s] = request.last_gr[0]
        self.pending_tokens[s].fill_(0)

    def read_request(self, rid):
        s = self.slot_of[rid]
        if ASYNC_REFRESH:
            require(self.pending_length[s] == 0, f"{rid}: a pending commit cannot continue into a prefill chunk")
        elif int(self.pending_length[s]) != 0:
            raise RuntimeError(f"{rid}: a pending commit cannot continue into a prefill chunk")
        w = self.c.window
        ring = self.positions[s, :w]
        order = torch.argsort(ring)  # padding (-1) first, then ascending positions: decode_cache's layout
        layers = tuple(
            LayerCache(self.key[i, s][:, order][None].to(torch.bfloat16, copy=True),
                       self.value[i, s][:, order][None].to(torch.bfloat16, copy=True),
                       () if self.inject else tuple(self.boundary[i, j, s : s + 1].clone() for j in range(4)))
            for i in range(self.c.num_layers)
        )
        return serving.DraftRequest(CleanCache(ring[order][None].clone(), layers),
                                    self.last_gr[s : s + 1].clone(), self.last_position[s : s + 1].clone())

    def _step(self, batch):
        """model.decode_step (per-row lengths) over ring windows, then commit only the committed rows."""
        if self.inject:
            return self._step_injected(batch)
        model, c = self.model, self.c
        b, w = c.block_size, c.window
        n = self.width
        idx = self.idx[:batch]
        ring_pos = self.positions[idx]
        if self.attention == "triton":  # ring_attention reads the pool by slot
            kv = [(RingWindow(self.key[i], idx, w + 1), RingWindow(self.value[i], idx, w + 1)) for i in range(c.num_layers)]
        else:  # gathered copies with spare columns (fast_drafter's SDPA path)
            kv = [(self.key[i][idx].to(torch.bfloat16), self.value[i][idx].to(torch.bfloat16)) for i in range(c.num_layers)]
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
        grs = model.proposal_gr(span.gather(1, slot.minimum(edge)[..., None, None].expand(-1, -1, *span.shape[2:])), spoken)
        hidden, block = model(embeds, grs, positions, prefix)
        take = edge + torch.arange(self.steps, device=self.device)
        proposal_hidden = hidden.gather(1, take[..., None].expand(-1, -1, hidden.shape[-1]))
        # commit: committed rows to their ring columns, the rest to the garbage column with position -1
        col = torch.where(committed, positions % w, torch.full_like(positions, w))
        rows = idx[:, None]
        self.positions[rows, col] = torch.where(committed, positions, torch.full_like(positions, -1))
        last = (length - 1).clamp(min=0)[:, None, None]
        has = (length > 0)[:, None, None]
        if BATCHED_COMMIT == "1":
            self._commit_batched(block, idx, rows, col, last, has)
        else:
            for i, layer in enumerate(block.layers):
                self.key[i][rows, :, col] = to_pool(layer.key.transpose(1, 2), self.kv_dtype)
                self.value[i][rows, :, col] = to_pool(layer.value.transpose(1, 2), self.kv_dtype)
                for j in range(4):
                    fresh = layer.boundaries[j].gather(1, last.expand(-1, 1, layer.boundaries[j].shape[-1]))
                    self.boundary[i, j][idx] = torch.where(has, fresh, self.boundary[i, j][idx])
        self.positions[:, w] = -1
        self.pending_length.index_fill_(0, idx, 0)
        proposal = fast_propose(model, proposal_hidden, self.anchor[:batch], self.temperature[:batch])
        self._emit(batch, proposal)

    def _step_injected(self, batch):
        """Context "inject" (model.MTP3._decode_step_injected over ring windows): commit, then propose.

        1. The pending block's committed rows (the first `length` of the last verify's [bonus, drafts...]) become
           context: per layer their injected K/V (model.MTP3.context: the fusion of each token's embedding with its
           predecessor GR, the layer's context_layernorm and k/v projections) go to their ring columns, the other
           rows to the garbage column at position -1. No trunk rows, no conv boundaries.
        2. The proposal block [anchor, mask x (steps - 1)] at cursor + 1 .., every slot on the cursor GR (the
           pending GR at the row's commit count), runs the trunk on the ring (which now ends at the cursor) and its
           own keys; its convs start from zero (PlainDecoder.forward).
        Committing first is safe: a committed position p overwrites ring column p % window, i.e. position p - window,
        at least window + 1 behind the anchor and so outside every block query's window. The trunk runs `steps` rows
        where model.decode_step runs the block's b and drops the last: attention is causal by position and the convs
        read the row before, so rows past the last proposal verify reads never reach the ones it does.
        """
        model, c = self.model, self.c
        w, steps = c.window, self.steps
        kc = steps + 1  # a verify of width K = steps + 1 commits at most K tokens
        idx = self.idx[:batch]
        length = self.pending_length[idx]
        cursor = self.last_position[idx]
        first = cursor + 1 - length
        ring_pos = self.positions[idx]
        require((first >= 1).all(), "position 0 has no predecessor GR")
        require((ring_pos.max(1).values + 1 == first).all(), "the ring must end where the block starts")
        require((length <= kc).all(), "a row cannot commit more tokens than a verify block holds")
        slot = torch.arange(kc, device=self.device)
        positions = first[:, None] + slot
        committed = slot < length[:, None]
        span = self.pending_gr[idx]  # [batch, block + 1, branches, hidden]: one predecessor per row, then the cursor's
        fused = model.fuse(model.embed_tokens(self.pending_tokens[idx][:, :kc]), span[:, :kc])
        col = torch.where(committed, positions % w, torch.full_like(positions, w))
        rows = idx[:, None]
        self.positions[rows, col] = torch.where(committed, positions, torch.full_like(positions, -1))
        if BATCHED_COMMIT == "1":
            kv = [layer.context_kv(fused, positions) for layer in model.layers]
            layer_ids = torch.arange(len(kv), device=self.device)[:, None, None]
            # pool [L, slots, Hkv, columns, D] indexed on (layer, slot, column): the value is [L, B, kc, Hkv, D]
            self.key[layer_ids, rows[None], :, col[None]] = to_pool(torch.stack([k for k, _ in kv]).transpose(2, 3), self.kv_dtype)
            self.value[layer_ids, rows[None], :, col[None]] = to_pool(torch.stack([v for _, v in kv]).transpose(2, 3), self.kv_dtype)
        else:
            for i, layer in enumerate(model.layers):
                k, v = layer.context_kv(fused, positions)
                self.key[i][rows, :, col] = to_pool(k.transpose(1, 2), self.kv_dtype)
                self.value[i][rows, :, col] = to_pool(v.transpose(1, 2), self.kv_dtype)
        self.positions[:, w] = -1
        block = (cursor + 1)[:, None] + torch.arange(steps, device=self.device)
        embeds = torch.cat(
            (model.embed_tokens(self.anchor[:batch])[:, None], model.mask_embedding.expand(batch, steps - 1, -1)), 1
        )
        cursor_gr = span.gather(1, length[:, None, None, None].expand(-1, 1, *span.shape[2:]))
        grs = model.proposal_gr(cursor_gr.expand(-1, steps, -1, -1),
                                (torch.arange(steps, device=self.device) == 0)[None])
        if self.attention == "triton":  # ring_attention reads the pool by slot
            kv = [(RingWindow(self.key[i], idx, w + 1), RingWindow(self.value[i], idx, w + 1)) for i in range(c.num_layers)]
        else:  # gathered copies with `steps` spare columns (fast_drafter's SDPA path)
            kv = [(self.key[i][idx].to(torch.bfloat16), self.value[i][idx].to(torch.bfloat16)) for i in range(c.num_layers)]
        prefix = CleanCache(self.positions[idx], tuple(LayerCache(*kv[i], ()) for i in range(c.num_layers)))
        hidden, _ = model(embeds, grs, block, prefix)
        self.pending_length.index_fill_(0, idx, 0)
        proposal = fast_propose(model, hidden, self.anchor[:batch], self.temperature[:batch])
        self._emit(batch, proposal)
