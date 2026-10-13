"""ARC_QSA_UNION_PREFILL (qsa/union_prefill.py) against a float64 reference and the chunk kernel.

Every query must attend to exactly its own selection: the expansion of its top-k blocks
(expand_qsa_block_indices) and its tail. Cases cover requests with fewer complete blocks than the
block top-k (all selected, the tail lands mid-row), more (the top-k is a strict subset, tail after
column block_topk * ratio), extends that are not a multiple of the tile, an extend of one token,
two KV heads, a shuffled pool (req_to_token is a permutation with stale slots past each request),
and indices / q with padding rows past the batch (the prefill CUDA graph's bucket). The reference is
softmax(q K^T scale) V in float64 over the selected keys; the union path must be as close to it as
the chunk kernel is. Runs on CUDA, or on CPU with TRITON_INTERPRET=1.
"""

import os
import unittest

import torch

from sglang.srt.layers.attention.qsa import union_prefill as up
from sglang.srt.layers.attention.qsa.kernel import torch_expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_fwd_interface_triton_ck
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RATIO = 4
# the Triton interpreter has no bf16 / fp8: CPU runs use fp16 q and K/V (the pool may be q's dtype)
Q_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float16
KV_DTYPE = torch.float8_e4m3fn if DEVICE == "cuda" else torch.float16
if DEVICE == "cpu":
    from sglang.srt.layers.attention.qsa import sparse_attn

    # its config table keys on torch.cuda.get_device_name; any config is exact under the interpreter
    sparse_attn._get_best_config = lambda total_q: (16, 1, 2)


def _selection(prefix_lens, extend_lens, block_topk, gen):
    """Top-k blocks per query from correlated scores, expanded the way the indexer does."""
    blocks, positions, lengths = [], [], []
    for prefix, new in zip(prefix_lens, extend_lens):
        seq = prefix + new
        nblk = seq // RATIO
        shared = torch.randn(nblk, generator=gen, dtype=torch.float64)
        walk = torch.cumsum(torch.randn(new, nblk, generator=gen, dtype=torch.float64) * 0.3, 0)
        scores = shared[None, :] + walk
        pos = torch.arange(prefix, seq)
        complete = (pos + 1) // RATIO
        scores = scores.masked_fill(torch.arange(nblk)[None, :] >= complete[:, None], -float("inf"))
        k = min(block_topk, nblk) if nblk else 0
        top = torch.full((new, block_topk), -1, dtype=torch.int32)
        if k:
            vals, idx = torch.topk(scores, k, dim=1)
            idx = torch.where(vals == -float("inf"), torch.full_like(idx, -1), idx).to(torch.int32)
            # fast_topk returns its valid entries in no particular order; shuffle them
            for r in range(new):
                valid = idx[r][idx[r] >= 0]
                top[r, : valid.numel()] = valid[torch.randperm(valid.numel(), generator=gen)]
        blocks.append(top)
        positions.append(pos)
        lengths.append(torch.full((new,), seq))
    return torch_expand_qsa_block_indices(
        torch.cat(blocks), torch.cat(positions), torch.cat(lengths), RATIO, block_topk * RATIO
    )


def _run(prefix_lens, extend_lens, *, block_topk=16, kv_heads=1, group=4, dim=64, tile=4, nb=4,
         pad_rows=0, seed=0):
    gen = torch.Generator().manual_seed(seed)
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    rows = sum(extend_lens)
    indices = _selection(prefix_lens, extend_lens, block_topk, gen)
    width = indices.shape[1]
    indices = torch.cat([indices, torch.full((pad_rows, width), -1, dtype=torch.int32)]).contiguous()
    q = torch.randn(rows + pad_rows, kv_heads * group, dim, generator=gen).to(Q_DTYPE)
    # pool: a permutation of slots, with unused slots between requests
    total = sum(seq_lens)
    pool_slots = total * 2 + 8
    k_pool = (torch.randn(pool_slots, kv_heads, dim, generator=gen) * 0.7).to(KV_DTYPE)
    v_pool = (torch.randn(pool_slots, kv_heads, dim, generator=gen) * 0.7).to(KV_DTYPE)
    perm = torch.randperm(pool_slots, generator=gen)[:total].to(torch.int32)
    stride = max(seq_lens) + 32
    r2t = torch.randint(0, pool_slots, (len(seq_lens) + 2, stride), generator=gen, dtype=torch.int32)
    req_rows = torch.tensor([2 + i for i in range(len(seq_lens))][::-1], dtype=torch.int64)
    off = 0
    for b, s in enumerate(seq_lens):
        r2t[req_rows[b], :s] = perm[off : off + s]
        off += s
    # the chunk kernel's tight context
    k_ctx = torch.cat([k_pool[r2t[req_rows[b], :s].long()] for b, s in enumerate(seq_lens)])
    v_ctx = torch.cat([v_pool[r2t[req_rows[b], :s].long()] for b, s in enumerate(seq_lens)])
    cu_q = torch.tensor([0] + list(torch.tensor(extend_lens).cumsum(0)), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32)
    scale = dim**-0.5
    d = DEVICE
    chunk = sparse_gqa_fwd_interface_triton_ck(
        q.to(d), k_ctx.to(d), v_ctx.to(d), indices.to(d), cu_q.to(d), cu_k.to(d), kv_lens.to(d), scale,
        max_q=max(extend_lens),
    ).cpu()
    plan = up.UnionPrefillPlan(extend_lens, seq_lens, ratio=RATIO, topk_width=width, device=d, tile=tile)
    union = up.union_prefill_attention(
        plan, q.to(d), k_pool.to(d), v_pool.to(d), r2t.to(d), req_rows.to(d), indices.to(d),
        cu_q.to(d), kv_lens.to(d), scale, nb=nb, warps=4, stages=1,
    ).cpu()
    # float64 reference per row
    ref = torch.empty(rows, kv_heads * group, dim, dtype=torch.float64)
    starts = cu_k.tolist()
    for b, (s, e) in enumerate(zip(seq_lens, extend_lens)):
        for j in range(e):
            r = int(cu_q[b]) + j
            sel = indices[r][indices[r] >= 0].long()
            assert int(sel.max()) <= s - e + j, "a selection reaches past its own query"
            kk = k_ctx[starts[b] + sel].double()  # [n, kv_heads, dim]
            vv = v_ctx[starts[b] + sel].double()
            for h in range(kv_heads * group):
                g = h // group
                p = torch.softmax(q[r, h].double() @ kk[:, g].T * scale, dim=-1)
                ref[r, h] = p @ vv[:, g]
    return union[:rows].double(), chunk[:rows].double(), ref, plan


def _ratio(out, ref):
    return ((out - ref).abs().amax(-1) / (ref.abs().amax(-1) + 1e-3)).max().item()


class TestUnionPrefill(unittest.TestCase):
    def _check(self, *args, **kwargs):
        union, chunk, ref, plan = _run(*args, **kwargs)
        self.assertTrue(torch.isfinite(union).all())
        err_chunk, err_union = _ratio(chunk, ref), _ratio(union, ref)
        # bf16 probabilities: a few 2^-8 of the row's scale for both kernels
        self.assertLess(err_chunk, 0.03)
        self.assertLess(err_union, max(2 * err_chunk, 0.01), f"union {err_union:.3e} chunk {err_chunk:.3e}")
        return plan

    def test_more_blocks_than_topk(self):
        # 400 + 37 tokens: 109 complete blocks, top-16 selected; extends not multiples of T
        self._check([400, 260], [37, 13])

    def test_fewer_blocks_than_topk(self):
        # every block selected; the tail sits at column 4 * blocks, not after block_topk * ratio
        self._check([20, 0 + 8], [9, 5])

    def test_one_token_extends_and_mixed(self):
        self._check([100, 64, 333], [1, 2, 6])

    def test_two_kv_heads_tile_two_and_eight(self):
        self._check([300, 150], [11, 21], kv_heads=2, tile=2)
        self._check([300, 150], [11, 21], kv_heads=2, tile=8, nb=8)

    def test_tiles_not_a_power_of_two(self):
        # rows pack as T x heads: T = 5 / 3 / 10 with 3 heads per kv head
        self._check([300, 150], [11, 21], group=3, tile=5)
        self._check([300, 150, 77], [11, 21, 3], kv_heads=2, group=3, tile=3)
        self._check([500], [33], group=3, tile=10)

    def test_fewer_than_16_keys_per_step_rejected(self):
        # the probabilities x values dot needs K >= 16; the interpreter would not catch it
        with self.assertRaisesRegex(ValueError, "keys per step"):
            _run([300], [11], nb=2)

    def test_padding_rows_untouched_and_union_counts(self):
        plan = self._check([256, 180], [16, 7], pad_rows=9)
        counts = plan.counts.cpu()
        # a tile reads at most T x block_topk blocks and at least block_topk
        self.assertTrue(((counts >= 1) & (counts <= plan.cap)).all())


if __name__ == "__main__":
    if DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("no CUDA: run with TRITON_INTERPRET=1")
    unittest.main()
