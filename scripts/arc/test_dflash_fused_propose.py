"""ARC_DFLASH_FUSED_PROPOSE: the fused selector chain (fused_propose.chain) against fast_propose's torch chain.

  python scripts/arc/test_dflash_fused_propose.py      # prints FUSED_PROPOSE_OK

1. Same inputs and the same uniforms: q agrees to rounding (sampled rows), greedy rows are the identical one-hot,
   and the drawn indices agree except where a uniform sits within rounding of a cumulative-sum boundary.
2. Self-consistency on the kernel's own draws: candidates[selected] == tokens, q[selected] > 0, q sums to one.
3. Distribution: one fixed row repeated 400k times with fresh draws -- the first position's empirical index
   frequencies match q (|f - q| <= 5 sigma), and the second position's match q conditioned on the first.
4. Timing: the torch chain vs the kernel at the served shape (steps 3, top-k 16, rank 256), CUDA-graphed.
"""

import torch
from torch import nn

from sglang.srt.speculative.arc_dflash.fused_propose import chain
from sglang.srt.speculative.arc_dflash.sampling import distribution

VOCAB, RANK, TOPK, STEPS = 248320, 256, 16, 3


class Selector(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(3)
        self.top_k = TOPK
        self.predecessor_codebook = nn.Parameter(torch.randn(VOCAB, RANK, device="cuda", generator=g).to(dtype) * 0.3)
        self.successor_codebook = nn.Parameter(torch.randn(VOCAB, RANK, device="cuda", generator=g).to(dtype) * 0.3)

    def scores(self, unary, projected, successor, predecessor):  # model.CandidateSelector.scores
        context = self.predecessor_codebook[predecessor] * projected
        return unary + torch.einsum("...r,...kr->...k", context, successor)


def torch_chain(selector, unary, candidates, projected, anchor, temperature, uniform):
    """state_pool.fast_propose's loop with the uniforms supplied (sample_index's arithmetic)."""
    successor = selector.successor_codebook[candidates]
    ones = torch.ones_like(temperature)
    keep = torch.full_like(temperature, -1, dtype=torch.long)
    previous = anchor
    tokens, probabilities, selected, all_scores = [], [], [], []
    for position in range(unary.shape[1]):
        scores = selector.scores(unary[:, position], projected[:, position], successor[:, position], previous)
        q = distribution(scores, temperature, ones, keep, span=scores.shape[-1])
        index = (q.cumsum(-1) <= uniform[:, position, None]).sum(-1, keepdim=True).clamp_(max=q.shape[-1] - 1)
        previous = candidates[:, position].gather(-1, index).squeeze(-1)
        tokens.append(previous)
        selected.append(index.squeeze(-1))
        probabilities.append(q)
        all_scores.append(scores)
    return (torch.stack(tokens, 1), torch.stack(probabilities, 1), torch.stack(selected, 1),
            torch.stack(all_scores, 1))


def inputs(batch, seed, scale=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    unary = (torch.randn(batch, STEPS, TOPK, device="cuda", generator=g) * scale).to(torch.bfloat16).float()
    unary, _ = unary.sort(-1, descending=True)  # a top-k comes out descending
    candidates = torch.stack([torch.randperm(VOCAB, device="cuda", generator=g)[: STEPS * TOPK] for _ in range(batch)])
    candidates = candidates.view(batch, STEPS, TOPK)
    projected = torch.randn(batch, STEPS, RANK, device="cuda", generator=g).to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (batch,), device="cuda", generator=g)
    temperature = torch.tensor([0.0, 0.3, 0.7, 1.0, 1.3], device="cuda").repeat(batch // 5 + 1)[:batch]
    uniform = torch.rand(batch, STEPS, device="cuda", generator=g)
    return unary, candidates, projected, anchor, temperature, uniform


def agreement(dtype):
    selector = Selector(dtype)
    batch = 4000
    unary, candidates, projected, anchor, temperature, uniform = inputs(batch, 11)
    ref_tok, ref_q, ref_sel, ref_scores = torch_chain(
        selector, unary, candidates, projected, anchor, temperature, uniform)
    got = chain(selector, unary, candidates, projected, anchor, temperature, uniform)
    # compare each position only on rows whose earlier positions drew the same token (same predecessor)
    same_prefix = torch.ones(batch, dtype=torch.bool, device="cuda")
    greedy = temperature == 0
    worst_q, flips, near, greedy_ties = 0.0, 0, 0, 0
    for p in range(STEPS):
        rows = same_prefix & ~greedy
        worst_q = max(worst_q, (got.probabilities[rows, p] - ref_q[rows, p]).abs().max().item())
        differ = same_prefix & (got.selected_indices[:, p] != ref_sel[:, p])
        for b in differ.nonzero().flatten().tolist():
            if greedy[b]:  # only a near-tie at the top may flip (the 256-wide dot rounds in another order)
                s = ref_scores[b, p]
                gap = (s[ref_sel[b, p]] - s[got.selected_indices[b, p]]).abs().item()
                if gap > 0.02:
                    raise RuntimeError(f"{dtype} position {p} row {b}: greedy pick differs, score gap {gap}")
                greedy_ties += 1
            else:
                c = ref_q[b, p].cumsum(0)
                near += (c - uniform[b, p]).abs().min().item() < 1e-4
        g_rows = same_prefix & greedy & ~differ
        if not torch.equal(got.probabilities[g_rows, p], ref_q[g_rows, p]):
            raise RuntimeError(f"{dtype} position {p}: greedy one-hot differs")
        flips += int((differ & ~greedy).sum())
        same_prefix = same_prefix & ~differ
    if worst_q > 1e-4:
        raise RuntimeError(f"{dtype}: q differs by {worst_q:.2e}")
    if flips != near:
        raise RuntimeError(f"{dtype}: {flips} drawn indices differ, only {near} of them at a cumsum boundary")
    print(f"agreement {dtype}: q max |diff| {worst_q:.2e}, {flips} boundary flips, {greedy_ties} greedy near-ties "
          f"over {batch} x {STEPS}")
    consistency(got)


def consistency(got):
    picked = got.selected_indices[..., None]
    if not torch.equal(got.candidates.gather(2, picked).squeeze(2), got.tokens):
        raise RuntimeError("tokens are not the selected candidates")
    if not (got.probabilities.gather(2, picked) > 0).all():
        raise RuntimeError("a drawn index carries zero q")
    if ((got.probabilities.sum(-1) - 1).abs() > 1e-5).any():
        raise RuntimeError("q does not sum to one")


def distribution_check():
    selector = Selector(torch.bfloat16)
    n = 400_000
    unary, candidates, projected, anchor, temperature, _ = inputs(1, 5, scale=1.0)
    temperature.fill_(1.0)
    rep = lambda t: t.expand(n, *t.shape[1:]).contiguous()
    got = chain(selector, rep(unary), rep(candidates), rep(projected), rep(anchor), rep(temperature))
    consistency(got)
    q0 = got.probabilities[0, 0]
    if not torch.equal(got.probabilities[:, 0], q0.expand(n, -1)):
        raise RuntimeError("position 0's q varies across identical rows")
    freq = torch.bincount(got.selected_indices[:, 0], minlength=TOPK).double() / n
    sigma = (q0.double() * (1 - q0.double()) / n).sqrt()
    z0 = ((freq - q0.double()).abs() / sigma.clamp_min(1e-12)).max().item()
    # position 1 given the most frequent first index: q1 is a function of that prefix
    top = int(freq.argmax())
    rows = got.selected_indices[:, 0] == top
    q1 = got.probabilities[rows, 1][0]
    m = int(rows.sum())
    freq1 = torch.bincount(got.selected_indices[rows, 1], minlength=TOPK).double() / m
    sigma1 = (q1.double() * (1 - q1.double()) / m).sqrt()
    z1 = ((freq1 - q1.double()).abs() / sigma1.clamp_min(1e-12)).max().item()
    if z0 > 5 or z1 > 5:
        raise RuntimeError(f"empirical frequencies off q: max z {z0:.2f} (position 0), {z1:.2f} (position 1)")
    print(f"distribution: max z {z0:.2f} (position 0, n={n}), {z1:.2f} (position 1 | index {top}, n={m})")


def timing():
    selector = Selector(torch.bfloat16)
    print(f"{'batch':>6} {'torch_us':>9} {'fused_us':>9}")
    for batch in (64, 96, 128):
        unary, candidates, projected, anchor, temperature, uniform = inputs(batch, batch)
        ones = torch.ones_like(temperature)

        def torch_path():  # fast_propose's chain including its per-position uniform draws
            successor = selector.successor_codebook[candidates]
            keep = torch.full_like(temperature, -1, dtype=torch.long)
            previous = anchor
            out = []
            for position in range(STEPS):
                scores = selector.scores(unary[:, position], projected[:, position], successor[:, position], previous)
                q = distribution(scores, temperature, ones, keep, span=TOPK)
                u = torch.empty_like(q[..., :1]).uniform_()
                index = (q.cumsum(-1) <= u).sum(-1, keepdim=True).clamp_(max=TOPK - 1)
                previous = candidates[:, position].gather(-1, index).squeeze(-1)
                out.append(q)
            return torch.stack(out, 1)

        def fused_path():
            return chain(selector, unary, candidates, projected, anchor, temperature).probabilities

        times = []
        for fn in (torch_path, fused_path):
            for _ in range(3):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    fn()
            graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000 / 500)
        print(f"{batch:>6} {times[0]:>9.1f} {times[1]:>9.1f}")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    with torch.no_grad():
        agreement(torch.bfloat16)
        agreement(torch.float32)
        distribution_check()
        timing()
    print("FUSED_PROPOSE_OK")
