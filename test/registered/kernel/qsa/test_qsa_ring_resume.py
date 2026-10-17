"""Session exact resume, QSA side: an extend whose prefix ends inside a compress group.

A parked session resumes on its own request row, so the keys of the group that
straddles the resume point P -- positions [floor4(P), P) -- are still in that
row's pending ring. The extend compresses that group from the ring head plus its
own chunk (qwen_sparse_attn_backend._qsa_ring_resume_plan,
ForwardBatch.qsa_ring_resume_cpu).

One QSA layer (real QSATokenToKVPool with an FP8 KV cache, real QSAIndexer and
QwenSparseAttnBackend) runs every request two ways:

  A  one extend of the whole turn [0, T), then TAIL decode steps;
  B  an extend of [0, P0), decode steps P0 .. P-1, the writes a finished
     speculative request's overshoot leaves past P, the resume extend [P, T)
     (optionally in two chunks, the second a chunked continuation), then the same
     TAIL decode steps.

The index projection is the identity (the "hidden states" are the projected q/k
rows), so both ways see the same key bits whatever the GEMM shapes. Compared:
compressed keys (bit for bit: same kernel, same members, same order), the KV
cache and the pending ring the decode tail continues from (bit for bit), the
selections of the resumed rows and of the decode tail (exactly;
ARC_QSA_TOPK_EXACT=1 makes fast_topk deterministic), the resumed rows' attention
(A runs the no-prefix kernel, B the prefix / union kernel: each within the
kernels' bf16 error of a float64 reference) and the decode tail's attention.

Run under the RL perf profile's QSA flags and without them, e.g.
  ARC_QSA_FP8_DIRECT=1 ARC_QSA_UNION_PREFILL=1 ARC_QSA_INDEXER_LOCAL=1 pytest ...
  pytest ...
  ARC_QSA_FP8_DIRECT=1 ARC_QSA_UNION_PREFILL=1 ARC_QSA_INDEXER_LOCAL=1 ARC_QSA_INDEX_FP8=1 pytest ...
(the module sets ARC_QSA_TOPK_EXACT=1 unless it is set).

The harness (Layer, Request, run_paths) keeps to the base tree's interfaces when
no resume mark and no ring slack is used, so a script can run the same flag-off
scenario on the base tree and on this one and compare the dumps bit for bit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

# The comparisons need a deterministic selection (fast_topk reads this at import;
# the exact_topk fixture skips if fast_topk was imported without it).
os.environ.setdefault("ARC_QSA_TOPK_EXACT", "1")

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

publish(ServerArgs(model_path="dummy"), role="test")

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

RATIO = 4
TOKEN_TOPK = 2048
IDX_HEADS = 4
IDX_DIM = 128
QK_WIDTH = (IDX_HEADS + 1) * IDX_DIM
Q_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
PAGE = 64
LAYER_ID = 3
VERIFY_WINDOW = 4  # DFlash K=4: the verify window the ring is sized for
TAIL = 6  # decode steps after the turn
DEVICE = "cuda"
BF16 = torch.bfloat16


def bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's raw bits, for exact comparison of any dtype (fp8 has no torch.equal)."""
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.contiguous().view(width)


class _IdentityProj(torch.nn.Module):
    def forward(self, x):
        return x, None


def _config():
    return SimpleNamespace(
        indexer_n_heads=IDX_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=IDX_DIM,
        indexer_budget=TOKEN_TOPK,
        indexer_compress_ratio=RATIO,
        hidden_size=QK_WIDTH,
        rms_norm_eps=1e-6,
    )


def make_indexer(seed: int) -> QSAIndexer:
    # Qwen3.8-Flash-Next's attention RoPE, which the indexer reuses: head 256,
    # partial rotary 0.25, interleaved M-RoPE [11, 11, 10], theta 1e7.
    rotary = MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM // 4,
        max_position_embeddings=32768,
        base=10000000,
        is_neox_style=True,
        dtype=BF16,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(BF16)
    try:
        indexer = QSAIndexer(_config(), layer_id=LAYER_ID, quant_config=None, rotary_emb=rotary)
        indexer.to(device=DEVICE)
    finally:
        torch.set_default_dtype(prev)
    indexer.index_qk_proj = _IdentityProj()
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        for norm in (indexer.q_layernorm, indexer.k_layernorm):
            w = (torch.randn(IDX_DIM, generator=gen) * 0.1).to(BF16).to(DEVICE)
            norm._weight_loader(norm.weight, w)
    return indexer


def mrope_coordinates(length, image=None):
    """[3, length] Qwen-VL M-RoPE coordinates: text has t = h = w = its position;
    an image (start, grid_h, grid_w) of grid_h * grid_w tokens sits at t = start,
    h = start + row, w = start + col, and the text after it continues from
    start + max(grid_h, grid_w), i.e. position + max(grid_h, grid_w) - grid_h * grid_w."""
    p = torch.arange(length, dtype=torch.int64)
    coords = p.unsqueeze(0).repeat(3, 1)
    if image is not None:
        start, gh, gw = image
        n = gh * gw
        idx = torch.arange(n)
        coords[0, start : start + n] = start
        coords[1, start : start + n] = start + idx // gw
        coords[2, start : start + n] = start + idx % gw
        coords[:, start + n :] += max(gh, gw) - n
    return coords.to(DEVICE)


class Request:
    """One sequence of `length` tokens: projected index q/k rows, attention q/k/v,
    its pages (slot of position p = slots[p])."""

    def __init__(self, name, length, gen, pages, image=None):
        self.name = name
        self.length = length
        self.coords = mrope_coordinates(length, image)
        self.qk = torch.randn(length, QK_WIDTH, generator=gen).to(BF16).to(DEVICE)
        self.q = torch.randn(length, Q_HEADS, HEAD_DIM, generator=gen).to(BF16).to(DEVICE)
        self.k = (torch.randn(length, KV_HEADS, HEAD_DIM, generator=gen) * 0.7).to(BF16).to(DEVICE)
        self.v = (torch.randn(length, KV_HEADS, HEAD_DIM, generator=gen) * 0.7).to(BF16).to(DEVICE)
        n_pages = -(-length // PAGE) + 1
        if len(pages) < n_pages:
            raise ValueError(f"{name}: {len(pages)} pages for {n_pages}")
        positions = torch.arange(n_pages * PAGE)
        page_ids = torch.tensor(pages[:n_pages])
        self.slots = (page_ids[positions // PAGE] * PAGE + positions % PAGE).to(DEVICE)

    def rope(self, start, end):
        """[3, n] M-RoPE coordinates of positions [start, end)."""
        return self.coords[:, start:end].contiguous()

    def group_slots(self, first, end):
        """Compressed slots of groups [first, end)."""
        return self.slots[torch.arange(first, end, device=DEVICE) * RATIO] // RATIO


class Layer:
    """One QSA layer: backend + indexer over one real pool; requests on fixed rows."""

    def __init__(self, num_pages, req_rows, indexer, verify, slack):
        kwargs = {}
        if slack:
            kwargs["qsa_ring_slack"] = slack  # base has no such kwarg
        self.pool = QSATokenToKVPool(
            size=num_pages * PAGE,
            dtype=torch.float8_e4m3fn,
            page_size=PAGE,
            head_num=KV_HEADS,
            head_dim=HEAD_DIM,
            full_attention_layer_ids=[LAYER_ID],
            device=DEVICE,
            mamba_pool=object(),
            qsa_index_kv_heads=1,
            qsa_index_head_dim=IDX_DIM,
            qsa_compress_ratio=RATIO,
            qsa_token_topk=TOKEN_TOPK,
            num_request_slots=req_rows,
            qsa_max_verify_tokens=verify,
            **kwargs,
        )
        self.ring = self.pool.qsa_pending_ring_size
        self.ctx = 4096 + 2 * PAGE
        self.req_to_token = torch.zeros(req_rows, self.ctx, dtype=torch.int32, device=DEVICE)
        runner = SimpleNamespace(
            device=DEVICE,
            token_to_kv_pool=self.pool,
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
            model_config=SimpleNamespace(
                context_len=self.ctx, hf_text_config=_config(), hf_config=_config()
            ),
            hisparse_coordinator=None,
            is_draft_worker=False,
        )
        self.backend = QwenSparseAttnBackend(runner)
        self.indexer = indexer
        self.attn = SimpleNamespace(
            layer_id=LAYER_ID, tp_q_head_num=Q_HEADS, head_dim=HEAD_DIM, scaling=HEAD_DIM**-0.5
        )
        self.rows = {}

    def place(self, req: Request, row: int):
        self.rows[req.name] = row
        self.req_to_token[row, : req.slots.numel()] = req.slots.to(torch.int32)

    def ring_slots(self, req: Request, start, end):
        return self.rows[req.name] * self.ring + torch.arange(start, end, device=DEVICE) % self.ring

    def extend(self, items, ring_resume=None):
        """items: [(req, start, end)] -> {name: (top-k rows, attention rows)}."""
        rows = [self.rows[r.name] for r, _, _ in items]
        seq = [e for _, _, e in items]
        ext = [e - s for _, s, e in items]
        pre = [s for _, s, _ in items]
        positions = torch.cat([torch.arange(s, e, device=DEVICE) for _, s, e in items])
        rope = torch.cat([r.rope(s, e) for r, s, e in items], dim=1)
        loc = torch.cat([r.slots[s:e] for r, s, e in items])
        n = positions.numel()
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(items),
            input_ids=torch.zeros(n, dtype=torch.int64, device=DEVICE),
            req_pool_indices=torch.tensor(rows, dtype=torch.int64, device=DEVICE),
            seq_lens=torch.tensor(seq, dtype=torch.int64, device=DEVICE),
            out_cache_loc=loc,
            seq_lens_sum=sum(seq),
            seq_lens_cpu=torch.tensor(seq, dtype=torch.int64),
            extend_seq_lens=torch.tensor(ext, dtype=torch.int32, device=DEVICE),
            extend_seq_lens_cpu=ext,
            extend_prefix_lens=torch.tensor(pre, dtype=torch.int32, device=DEVICE),
            extend_prefix_lens_cpu=pre,
            positions=positions,
            mrope_positions=rope,
            req_pool_indices_cpu=rows,
        )
        if ring_resume is not None:
            fb.qsa_ring_resume_cpu = ring_resume  # base has no such field
        hidden = torch.cat([r.qk[s:e] for r, s, e in items])
        q = torch.cat([r.q[s:e] for r, s, e in items])
        k = torch.cat([r.k[s:e] for r, s, e in items])
        v = torch.cat([r.v[s:e] for r, s, e in items])
        self.backend.init_forward_metadata(fb)
        md = self.backend.forward_metadata.indexer_metadata
        topk = self.indexer.forward_cuda(hidden, rope, fb, md)
        out = self.backend.forward_extend(q, k, v, self.attn, fb, topk_indices=topk)
        out = out.reshape(n, Q_HEADS, HEAD_DIM)
        result, at = {}, 0
        for r, s, e in items:
            result[r.name] = (topk[at : at + e - s].clone(), out[at : at + e - s].clone())
            at += e - s
        return result

    def decode(self, items):
        """items: [(req, position)]: one decode step each -> {name: (top-k row, attention row)}."""
        rows = [self.rows[r.name] for r, _ in items]
        pos = [p for _, p in items]
        seq = [p + 1 for p in pos]
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=len(items),
            input_ids=torch.zeros(len(items), dtype=torch.int64, device=DEVICE),
            req_pool_indices=torch.tensor(rows, dtype=torch.int64, device=DEVICE),
            seq_lens=torch.tensor(seq, dtype=torch.int64, device=DEVICE),
            out_cache_loc=torch.stack([r.slots[p] for r, p in items]),
            seq_lens_sum=sum(seq),
            seq_lens_cpu=torch.tensor(seq, dtype=torch.int64),
            positions=torch.tensor(pos, dtype=torch.int64, device=DEVICE),
            mrope_positions=torch.cat([r.rope(p, p + 1) for r, p in items], dim=1),
        )
        hidden = torch.stack([r.qk[p] for r, p in items])
        q = torch.stack([r.q[p] for r, p in items])
        k = torch.stack([r.k[p] for r, p in items])
        v = torch.stack([r.v[p] for r, p in items])
        self.backend.init_forward_metadata(fb)
        md = self.backend.forward_metadata.indexer_metadata
        topk = self.indexer.forward_cuda(hidden, fb.mrope_positions, fb, md)
        out = self.backend.forward_decode(q, k, v, self.attn, fb, topk_indices=topk)
        out = out.reshape(len(items), Q_HEADS, HEAD_DIM)
        return {r.name: (topk[i].clone(), out[i].clone()) for i, (r, _) in enumerate(items)}

    def overshoot(self, req: Request, resume: int, gen, window: int = VERIFY_WINDOW):
        """What a request that finished at P = resume under speculative decoding can
        leave past P: its last verify ran [s, s + K) with P = s + i + 1, and the
        overlap scheduler's one more verify wrote [s + L, s + L + K), so writes reach
        P + L + K - 2 - i <= P + 2K - 2. Garbage ring keys (carrying the true RoPE
        coordinates of the positions they were written for), KV, and compressed slots
        of every group those positions touch, over [P, P + 2K - 1). Returns the
        clobbered compressed slots, first group floor(P / 4)."""
        end = resume + 2 * window - 1
        if end > req.slots.numel():
            raise ValueError(f"{req.name}: overshoot past the request's pages")
        n = end - resume
        ring_slots = self.ring_slots(req, resume, end)
        self.pool.get_qsa_key_state_buffer(LAYER_ID)[ring_slots] = (
            torch.randn(n, 1, IDX_DIM, generator=gen).to(BF16).to(DEVICE)
        )
        self.pool.qsa_rope_position_buffer[ring_slots] = req.rope(resume, end).transpose(0, 1)
        self.pool.set_kv_buffer(
            self.attn,
            req.slots[resume:end],
            (torch.randn(n, KV_HEADS, HEAD_DIM, generator=gen) * 3).to(BF16).to(DEVICE),
            (torch.randn(n, KV_HEADS, HEAD_DIM, generator=gen) * 3).to(BF16).to(DEVICE),
        )
        cslots = req.group_slots(resume // RATIO, (end - 1) // RATIO + 1)
        self.pool.set_qsa_compressed_k_buffer(
            LAYER_ID,
            cslots,
            (torch.randn(cslots.numel(), 1, IDX_DIM, generator=gen) * 5).to(BF16).to(DEVICE),
        )
        return cslots

    def compressed_bits(self, slots):
        out = [bits(self.pool.get_qsa_compressed_k_buffer(LAYER_ID)[slots])]
        scale = self.pool.get_qsa_compressed_k_scale(LAYER_ID)
        if scale is not None:
            out.append(bits(scale[slots]))
        return out

    def kv_bits(self, slots):
        return [
            bits(self.pool.get_key_buffer(LAYER_ID)[slots]),
            bits(self.pool.get_value_buffer(LAYER_ID)[slots]),
        ]


def attention_reference(layer: Layer, req: Request, topk_rows, start, *, from_pool):
    """float64 softmax(q K^T * scale) V over each row's selection. Row i is the
    query at position start + i. K/V: read back from the FP8 pool (what the prefix
    kernels read), or the forward's own BF16 K/V (what the no-prefix kernel reads)."""
    out = torch.empty(topk_rows.shape[0], Q_HEADS, HEAD_DIM, dtype=torch.float64, device=DEVICE)
    group = Q_HEADS // KV_HEADS
    key_buf = layer.pool.get_key_buffer(LAYER_ID)
    val_buf = layer.pool.get_value_buffer(LAYER_ID)
    for i in range(topk_rows.shape[0]):
        sel = topk_rows[i][topk_rows[i] >= 0].long()
        if int(sel.max()) > start + i:
            raise AssertionError(f"{req.name}: row {start + i} selects position {int(sel.max())}")
        if from_pool:
            kk = key_buf[req.slots[sel]].double()
            vv = val_buf[req.slots[sel]].double()
        else:
            kk = req.k[sel].double()
            vv = req.v[sel].double()
        q = req.q[start + i].double()
        for g in range(KV_HEADS):
            qs = q[g * group : (g + 1) * group]
            p = torch.softmax(qs @ kk[:, g].T * layer.attn.scaling, dim=-1)
            out[i, g * group : (g + 1) * group] = p @ vv[:, g]
    return out


def err_ratio(out, ref):
    """max over rows of max |out - ref| / max |ref| (the union-prefill test's measure)."""
    return ((out.double() - ref).abs().amax(-1) / (ref.abs().amax(-1) + 1e-3)).max().item()


# (name, turn end T, first prefill end P0, resume point P, chunk boundary C or None,
#  image (start, grid_h, grid_w) or None)
# P % 4 in {0, 1, 2, 3}; extend lengths 1, 2, 3 (< 4: the straddling group may stay
# open for the decode tail to complete), several groups, page crossings (3072, 3136);
# P0 < P: the head was written by decode steps. C splits the resume into a first
# chunk ending inside a group (2nd chunk = misaligned chunked continuation) or on a
# page (the scheduler's first-chunk rule). r0_e1 is aligned and unmarked in the batch.
CASES = [
    ("r0_e1", 3001, 2990, 3000, None, None),
    ("r1_e1", 3010, 2990, 3009, None, None),  # one token, group stays open; head on a page start
    ("r2_e2", 3004, 2990, 3002, None, (100, 20, 20)),  # completes the group; after an image
    ("r3_e1", 3004, 2995, 3003, None, None),  # one token completes the group
    ("r1_e3", 3008, 2989, 3005, None, None),
    ("r3_e37", 3076, 2987, 3039, None, (2000, 8, 32)),  # several groups, crosses page 3072; image
    ("r2_e70", 3136, 3050, 3066, None, None),  # crosses page 3072, ends on page 3136
    ("r1_chunk", 3200, 3050, 3065, 3102, None),  # first chunk ends at 3102 % 4 == 2
    ("r3_chunk_pg", 3200, 3050, 3071, 3072, None),  # first chunk ends on a page
]
UNMARKED = {"r0_e1"}


def build(verify, slack, seed=0, cases=CASES):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    indexer = make_indexer(seed)
    pages_per = (max(c[1] for c in cases) + TAIL + 2 * VERIFY_WINDOW) // PAGE + 3
    n_pages = 1 + pages_per * len(cases)
    perm = (torch.randperm(n_pages - 1, generator=gen) + 1).tolist()  # page 0: padding
    reqs = [
        Request(name, t + TAIL, gen, perm[i * pages_per : (i + 1) * pages_per], image)
        for i, (name, t, _, _, _, image) in enumerate(cases)
    ]
    rows = 2 * len(cases) + 2
    a = Layer(n_pages, rows, indexer, verify, slack)
    b = Layer(n_pages, rows, indexer, verify, slack)
    for i, r in enumerate(reqs):
        a.place(r, 1 + i)
        b.place(r, 1 + len(cases) + i)  # other rows: other ring slots
    return gen, reqs, a, b


def decode_tail(layer, reqs, turn_end):
    steps = []
    for step in range(TAIL):
        steps.append(layer.decode([(r, turn_end[r.name] + step) for r in reqs]))
    return steps


def run_paths(verify, slack, *, overshoot=True, mark=True, seed=0, cases=CASES):
    """Path A and path B over `cases`; returns everything compare reads."""
    gen, reqs, a, b = build(verify, slack, seed, cases)
    case = {c[0]: c for c in cases}
    turn_end = {name: c[1] for name, c in case.items()}
    ref = a.extend([(r, 0, turn_end[r.name]) for r in reqs])
    ref_tail = decode_tail(a, reqs, turn_end)

    b.extend([(r, 0, case[r.name][2]) for r in reqs])
    step = 0
    while True:
        items = [(r, case[r.name][2] + step) for r in reqs if case[r.name][2] + step < case[r.name][3]]
        if not items:
            break
        b.decode(items)
        step += 1
    clobbered, garbage_seen = {}, {}
    if overshoot:
        for r in reqs:
            clobbered[r.name] = b.overshoot(r, case[r.name][3], gen)
            # the overshoot really left wrong compressed keys (the test is not vacuous)
            ka, kb = a.compressed_bits(clobbered[r.name]), b.compressed_bits(clobbered[r.name])
            garbage_seen[r.name] = int((ka[0] != kb[0]).flatten(1).any(1).sum())

    def marks(rs):
        if not mark:
            return None
        return [r.name not in UNMARKED for r in rs]

    first = [(r, case[r.name][3], case[r.name][4] or turn_end[r.name]) for r in reqs]
    got = b.extend(first, ring_resume=marks([r for r, _, _ in first]))
    second = [(r, case[r.name][4], turn_end[r.name]) for r in reqs if case[r.name][4] is not None]
    if second:
        more = b.extend(second, ring_resume=marks([r for r, _, _ in second]))
        for name, (t, o) in more.items():
            t0, o0 = got[name]
            got[name] = (torch.cat([t0, t]), torch.cat([o0, o]))
    got_tail = decode_tail(b, reqs, turn_end)
    torch.cuda.synchronize()
    return SimpleNamespace(
        reqs=reqs, a=a, b=b, case=case, ref=ref, got=got, ref_tail=ref_tail,
        got_tail=got_tail, clobbered=clobbered, garbage_seen=garbage_seen,
    )


def compare(run):
    """Assert path B equals path A (bitwise where the kernels are the same);
    returns per request numbers for the report."""
    report = {}
    for r in run.reqs:
        _, turn_end, _, resume, _, _ = run.case[r.name]
        end = turn_end + TAIL
        groups = r.group_slots(0, end // RATIO)
        for x, y in zip(run.a.compressed_bits(groups), run.b.compressed_bits(groups)):
            bad = (x != y).flatten(1).any(1).nonzero().flatten()
            assert bad.numel() == 0, f"{r.name}: compressed keys differ at groups {bad[:8].tolist()}"
        for x, y in zip(run.a.kv_bits(r.slots[:end]), run.b.kv_bits(r.slots[:end])):
            assert torch.equal(x, y), f"{r.name}: KV differs"
        # the pending ring the next decode step would continue from
        open_start = end // RATIO * RATIO
        ka = run.a.pool.get_qsa_key_state_buffer(LAYER_ID)[run.a.ring_slots(r, open_start, end)]
        kb = run.b.pool.get_qsa_key_state_buffer(LAYER_ID)[run.b.ring_slots(r, open_start, end)]
        assert torch.equal(bits(ka), bits(kb)), f"{r.name}: pending ring keys differ"
        pa = run.a.pool.qsa_rope_position_buffer[run.a.ring_slots(r, open_start, end)]
        pb = run.b.pool.qsa_rope_position_buffer[run.b.ring_slots(r, open_start, end)]
        assert torch.equal(pa, pb), f"{r.name}: pending ring RoPE coordinates differ"

        ref_topk, ref_out = run.ref[r.name]
        topk, out = run.got[r.name]
        assert topk.shape[0] == turn_end - resume
        bad = (ref_topk[resume:] != topk).any(1).nonzero().flatten()
        assert bad.numel() == 0, f"{r.name}: selection differs at rows {(bad + resume)[:8].tolist()}"
        # A's rows come from the no-prefix kernel (the forward's BF16 K/V), B's from
        # the prefix / union kernel (the FP8 cache): each against its own inputs
        err_a = err_ratio(ref_out[resume:], attention_reference(run.a, r, topk, resume, from_pool=False))
        err_b = err_ratio(out, attention_reference(run.b, r, topk, resume, from_pool=True))
        assert err_a < 0.03 and err_b < max(2 * err_a, 0.01), f"{r.name}: A {err_a:.3e} B {err_b:.3e}"
        tail_bitwise = True
        tail_err = 0.0
        for step_a, step_b in zip(run.ref_tail, run.got_tail):
            ta, oa = step_a[r.name]
            tb, ob = step_b[r.name]
            assert torch.equal(ta, tb), f"{r.name}: decode-tail selection differs"
            tail_bitwise &= torch.equal(bits(oa), bits(ob))
            tail_err = max(tail_err, (oa.float() - ob.float()).abs().max().item())
        assert tail_err < 1e-2, f"{r.name}: decode-tail attention differs by {tail_err}"
        report[r.name] = dict(
            err_a=err_a, err_b=err_b, tail_bitwise=tail_bitwise, tail_err=tail_err,
            garbage_groups=run.garbage_seen.get(r.name, 0),
            clobbered_groups=0 if r.name not in run.clobbered else run.clobbered[r.name].numel(),
        )
    return report


def _print_report(title, report):
    print(f"\n[{title}]")
    for name, row in report.items():
        print(
            f"  {name:12s} attn err A {row['err_a']:.2e} B {row['err_b']:.2e}  "
            f"tail bitwise {row['tail_bitwise']} (max {row['tail_err']:.1e})  "
            f"overshoot garbage groups {row['garbage_groups']}/{row['clobbered_groups']}"
        )


@pytest.fixture
def exact_topk():
    from sglang.kernels.ops.elementwise import fast_topk

    if not fast_topk._ARC_QSA_TOPK_EXACT:
        pytest.skip("needs ARC_QSA_TOPK_EXACT=1 (deterministic selection to compare)")


def test_resume_slack_ring_after_overshoot(exact_topk):
    """Spec ring (ratio + W + slack W = 12) with overshoot writes up to P + 2W - 2:
    from the head's first position P - 3 to the farthest write the distance is
    <= 2W + 1 = 9 < 12, so no write lands on a head slot."""
    from sglang.srt.mem_cache.qsa_kv_pool import qsa_resume_ring_slack

    slack = qsa_resume_ring_slack(VERIFY_WINDOW, True)
    assert slack == VERIFY_WINDOW and qsa_resume_ring_slack(VERIFY_WINDOW, False) == 0
    run = run_paths(VERIFY_WINDOW, slack, overshoot=True)
    assert run.a.ring == 12
    report = compare(run)
    # every request's overshoot left wrong compressed slots that the resume extend
    # (or the decode tail) had to rewrite before anything read them
    assert all(row["garbage_groups"] > 0 for row in report.values())
    _print_report("slack ring 12, overshoot", report)


def test_resume_nonspec_ring(exact_topk):
    """Non-spec ring (ratio slots): every chunk's pending tail maps onto the head's
    slots, so this checks the head is read before the store. No overshoot."""
    _print_report("ring 4, no overshoot", compare(run_paths(0, 0, overshoot=False)))


def test_unmarked_misaligned_prefix_raises():
    _, reqs, _, b = build(0, 0, cases=CASES[1:2])
    r = reqs[0]
    b.extend([(r, 0, 3008)])
    b.decode([(r, 3008)])
    with pytest.raises(ValueError, match="not marked"):
        b.extend([(r, 3009, 3010)], ring_resume=[False])
    with pytest.raises(ValueError, match="not marked"):
        b.extend([(r, 3009, 3010)], ring_resume=[])  # a short mark list covers no row


def test_marked_aligned_rows_take_the_base_plan(exact_topk):
    """A mark on aligned rows changes nothing: the same compressed keys, ring,
    selection and outputs as the unmarked (base) path, bit for bit."""
    dumps = []
    for mark in (None, [True]):
        _, reqs, _, b = build(VERIFY_WINDOW, 0, cases=CASES[:1])
        r = reqs[0]
        b.extend([(r, 0, 2996)])
        topk, out = b.extend([(r, 2996, 3001)], ring_resume=mark)[r.name]
        torch.cuda.synchronize()
        # request row 0's ring is the extend dump: several tokens race for its
        # slots (run to run, in base too), so it is left out
        dumps.append(
            [
                bits(b.pool.get_qsa_compressed_k_buffer(LAYER_ID)),
                bits(b.pool.get_qsa_key_state_buffer(LAYER_ID)[b.ring :]),
                b.pool.qsa_rope_position_buffer[b.ring :].clone(),
                topk,
                bits(out),
            ]
        )
    for x, y in zip(*dumps):
        assert torch.equal(x, y)


_SUBPROCESS = r"""
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("h", sys.argv[1])
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
mode = sys.argv[2]
if mode == "collide":
    # base spec ring (4 + W = 8, no slack); P % 4 == 3: head [P-3, P); the overshoot
    # writes P + 5, P + 6 onto the head's slots (P + 5 = P - 3 mod 8)
    case = [("r3_hit", 3008, 2995, 3003, None, None)]
    gen, reqs, a, b = h.build(h.VERIFY_WINDOW, 0, cases=case)
    r = reqs[0]
    b.extend([(r, 0, 2995)])
    for p in range(2995, 3003):
        b.decode([(r, p)])
    b.overshoot(r, 3003, gen)
    b.extend([(r, 3003, 3008)], ring_resume=[True])
elif mode == "unmarked_none":
    # no mark at all (the flag-off server): the base contract, a device-side assert
    case = [("r1", 3010, 2990, 3009, None, None)]
    gen, reqs, a, b = h.build(0, 0, cases=case)
    r = reqs[0]
    b.extend([(r, 0, 3008)])
    b.decode([(r, 3008)])
    b.extend([(r, 3009, 3010)])
else:
    raise SystemExit("unknown mode " + mode)
torch.cuda.synchronize()
print("NO-ERROR")
"""


@pytest.mark.parametrize(
    "mode,debug,expect",
    [
        ("collide", "1", "QSA ring resume: the pending-ring head does not hold"),
        # the device-side assert prints its message
        ("collide", "0", "a pending-ring head slot was overwritten"),
        # base's message-less torch._assert_async(prefix_lens % ratio == 0)
        ("unmarked_none", "0", "device-side assert"),
    ],
)
def test_contract_violations_fail_loudly(mode, debug, expect, tmp_path):
    """Without slack (ring 4 + W) one overshoot window reaches the head: the RoPE
    guard stops the resume (RuntimeError under ARC_HISPARSE_QSA_DEBUG=1, else a
    device-side assert). A misaligned extend with no mark keeps base's device assert."""
    script = tmp_path / "violate.py"
    script.write_text(_SUBPROCESS)
    env = dict(os.environ, ARC_HISPARSE_QSA_DEBUG=debug)
    proc = subprocess.run(
        [sys.executable, str(script), __file__, mode],
        env=env, capture_output=True, text=True, timeout=900,
    )
    text = proc.stdout + proc.stderr
    assert "NO-ERROR" not in text, text[-3000:]
    assert proc.returncode != 0
    assert expect in text, text[-3000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
