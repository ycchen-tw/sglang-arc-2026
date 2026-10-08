"""arc DFlash sparse q (ARC_DFLASH_SPARSE_Q=1): the sparse-q chain rejection-sampling kernel against the dense one.

With the same p, the same q (dense = the support scattered into the vocabulary) and the same coins, the two kernels
must give the same accept counts, accept_index and tokens -- at the G4 deployment shape (110 rows, K=8, 248,320-token
vocabulary, 16 candidates), for every accept length 0..K-1, and under CUDA graph replay with the batch size changing
between replays. A CPU (Triton interpreter) version of the parity plus an exact enumeration of the conditional chain
lives with the patch (arc-forge docs).
"""

import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_sparse_q_triton,
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

VOCAB, SUPPORT = 248320, 16
MAX_COIN = 1.0 - 2**-24


def make_case(gen, batch, width, mode):
    steps = width - 1
    dev = "cuda"
    p = torch.softmax(torch.randn(batch, width, VOCAB, device=dev, generator=gen) * 3, -1)
    if mode == "topk":
        top = p.topk(20, -1)
        p = torch.zeros_like(p).scatter_(-1, top.indices, top.values / top.values.sum(-1, keepdim=True))
    ids = torch.stack([torch.randperm(VOCAB, device=dev, generator=gen)[:SUPPORT] for _ in range(batch * steps)])
    ids = ids.view(batch, steps, SUPPORT)
    qs = torch.softmax(torch.randn(batch, steps, SUPPORT, device=dev, generator=gen) * 2, -1)
    dense = torch.zeros(batch, steps, VOCAB, device=dev).scatter_(2, ids, qs)
    if mode == "p_eq_q":
        p[:, :steps] = dense
    u = torch.rand(batch, steps, 1, device=dev, generator=gen)
    pick = (qs.cumsum(-1) <= u).sum(-1).clamp(max=SUPPORT - 1)
    drafts = ids.gather(2, pick[..., None]).squeeze(2)
    cand = torch.cat((torch.randint(0, VOCAB, (batch, 1), device=dev, generator=gen), drafts), 1)
    coins = torch.rand(batch, width, device=dev, generator=gen)
    if mode == "forced_a":
        a = torch.arange(batch, device=dev) % width
        coins = torch.where(torch.arange(width, device=dev)[None] < a[:, None], 0.0, MAX_COIN)
    final = torch.rand(batch, device=dev, generator=gen)
    return cand, p.contiguous(), ids, qs.contiguous(), dense, coins.contiguous(), final


def outputs(batch, width):
    return (torch.zeros(batch * width, dtype=torch.int32, device="cuda"),
            torch.full((batch, width), -1, dtype=torch.int32, device="cuda"),
            torch.empty(batch, dtype=torch.int32, device="cuda"))


def run_dense(cand, p, dense, coins, final):
    batch, width = cand.shape
    predict, accept_index, accepted = outputs(batch, width)
    retrieve = torch.arange(batch * width, device="cuda").view(batch, width)
    chain_speculative_sampling_triton(predict, accept_index, accepted, cand, retrieve, None, None, coins, final, p,
                                      dense, 1.0, 1.0, True)
    return predict, accept_index, accepted


def run_sparse(cand, p, ids, qs, coins, final, out=None):
    batch, width = cand.shape
    predict, accept_index, accepted = outputs(batch, width) if out is None else out
    retrieve = torch.arange(batch * width, device="cuda").view(batch, width)
    chain_speculative_sampling_sparse_q_triton(predict, accept_index, accepted, cand, retrieve, coins, final, p,
                                               ids, qs)
    return predict, accept_index, accepted


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestArcDflashSparseQ(CustomTestCase):
    def test_parity_deploy_shape(self):
        gen = torch.Generator(device="cuda").manual_seed(20261008)
        seen = set()
        for width in (2, 4, 8):
            for mode in ("plain", "topk", "p_eq_q", "forced_a"):
                cand, p, ids, qs, dense, coins, final = make_case(gen, 110, width, mode)
                want = run_dense(cand, p, dense, coins, final)
                got = run_sparse(cand, p, ids, qs, coins, final)
                for w, g in zip(want, got):
                    self.assertTrue(torch.equal(w, g), f"K={width} {mode}")
                if mode == "p_eq_q":
                    self.assertTrue(bool((want[2] == width - 1).all()))
                seen.update((width, int(a)) for a in want[2].tolist())
        for width in (2, 4, 8):
            self.assertEqual({a for w, a in seen if w == width}, set(range(width)))

    def test_graph_replay_alternating_batch(self):
        gen = torch.Generator(device="cuda").manual_seed(7)
        width, sizes = 8, (110, 55, 1)
        # static inputs at the largest size; each graph reads a prefix
        cand, p, ids, qs, dense, coins, final = make_case(gen, 110, width, "plain")
        graphs, outs = {}, {}
        for b in sizes:
            outs[b] = outputs(b, width)
            run_sparse(cand[:b], p[:b], ids[:b], qs[:b], coins[:b], final[:b], out=outs[b])  # warm-up / compile
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_sparse(cand[:b], p[:b], ids[:b], qs[:b], coins[:b], final[:b], out=outs[b])
            graphs[b] = graph
        for round_ in range(6):
            b = sizes[round_ % len(sizes)]
            mode = "forced_a" if round_ % 2 else "plain"
            n_cand, n_p, n_ids, n_qs, n_dense, n_coins, n_final = make_case(gen, 110, width, mode)
            for dst, src in ((cand, n_cand), (p, n_p), (ids, n_ids), (qs, n_qs), (coins, n_coins), (final, n_final)):
                dst.copy_(src)
            outs[b][0].zero_()  # the kernel writes only the accepted rows' slots, as eagle_sample's fresh buffers
            outs[b][1].fill_(-1)
            graphs[b].replay()
            want = run_dense(n_cand[:b], n_p[:b], n_dense[:b], n_coins[:b], n_final[:b])
            for w, g in zip(want, outs[b]):
                self.assertTrue(torch.equal(w, g), f"replay {round_} batch {b}")

    def test_rejects_bad_shapes(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        cand, p, ids, qs, dense, coins, final = make_case(gen, 4, 4, "plain")
        with self.assertRaises(ValueError):
            run_sparse(cand, p, ids[:, :2], qs[:, :2], coins, final)  # fewer q rows than proposals
        with self.assertRaises(ValueError):
            run_sparse(cand, p, ids, qs.double(), coins, final)


if __name__ == "__main__":
    unittest.main()
