"""arc DFlash: the selector codebooks pruned to the ids that can reach them (CandidateSelector.prune).

Under ARC_TARGET_VOCAB a predecessor is the target's sampled bonus token or a drafted token and a candidate is a
draft-vocabulary id, so build_dense_drafter keeps the predecessor rows of the target list and the successor rows of
the draft vocabulary. Same rows, so the scores and the fused chain's draws must be bitwise those of the full codebooks.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.arc_dflash import fused_propose, state_pool
from sglang.srt.speculative.arc_dflash.model import CandidateSelector

VOCAB, RANK, TOP_K, HIDDEN = 4096, 256, 16, 128
FIELDS = ("tokens", "candidates", "probabilities", "selected_indices")


def selectors(device):
    torch.manual_seed(0)
    c = SimpleNamespace(selector_top_k=TOP_K, vocab_size=VOCAB, selector_rank=RANK, hidden_size=HIDDEN)
    full = CandidateSelector(c).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        full.successor_codebook.normal_(std=0.02)
    pruned = CandidateSelector(c).to(device=device, dtype=torch.bfloat16)
    pruned.load_state_dict(full.state_dict())
    perm = torch.randperm(VOCAB, device=device)
    pred_ids = perm[:2048].sort().values
    succ_ids = pred_ids[torch.randperm(2048, device=device)[:512]].sort().values  # draft list inside the target list
    pruned.prune(pred_ids, succ_ids)
    return full, pruned, pred_ids, succ_ids


def draws(pred_ids, succ_ids, batch, steps, device):
    candidates = succ_ids[torch.randint(len(succ_ids), (batch, steps, TOP_K), device=device)]
    anchor = pred_ids[torch.randint(len(pred_ids), (batch,), device=device)]
    unary = torch.randn(batch, steps, TOP_K, device=device)
    projected = torch.randn(batch, steps, RANK, device=device, dtype=torch.bfloat16)
    return candidates, anchor, unary, projected


def test_pruned_codebooks_hold_only_the_listed_rows():
    full, pruned, pred_ids, succ_ids = selectors("cpu")
    assert pruned.predecessor_codebook.shape == (2048, RANK) and pruned.successor_codebook.shape == (512, RANK)
    assert torch.equal(pruned.pred_lookup(pred_ids), full.predecessor_codebook[pred_ids])
    assert torch.equal(pruned.succ_lookup(succ_ids), full.successor_codebook[succ_ids])
    off = torch.tensor([int(torch.isin(torch.arange(VOCAB), succ_ids, invert=True).nonzero()[0])])
    with pytest.raises(IndexError):  # an id off the list fails, never reads another row
        pruned.succ_lookup(off)
    with pytest.raises(RuntimeError, match="already pruned"):
        pruned.prune(pred_ids, succ_ids)
    with pytest.raises(ValueError, match="sorted unique"):
        CandidateSelector(SimpleNamespace(selector_top_k=TOP_K, vocab_size=VOCAB, selector_rank=RANK,
                                          hidden_size=HIDDEN)).prune(pred_ids.flip(0), succ_ids)


def test_scores_are_bitwise_equal_on_cpu():
    full, pruned, pred_ids, succ_ids = selectors("cpu")
    candidates, anchor, unary, projected = draws(pred_ids, succ_ids, 6, 3, "cpu")
    for p in range(3):
        a = full.scores(unary[:, p], projected[:, p], full.succ_lookup(candidates)[:, p], anchor)
        b = pruned.scores(unary[:, p], projected[:, p], pruned.succ_lookup(candidates)[:, p], anchor)
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused chain is a Triton kernel")
def test_fused_chain_is_bitwise_equal():
    full, pruned, pred_ids, succ_ids = selectors("cuda")
    candidates, anchor, unary, projected = draws(pred_ids, succ_ids, 40, 3, "cuda")
    temperature = torch.tensor([0.0, 1.0] * 20, device="cuda")
    uniform = torch.rand(40, 3, device="cuda")
    a = fused_propose.chain(full, unary, candidates, projected, anchor, temperature, uniform)
    b = fused_propose.chain(pruned, unary, candidates, projected, anchor, temperature, uniform)
    for name in FIELDS:
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert torch.isin(b.tokens, succ_ids).all()  # tokens are vocabulary ids, not codebook rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fast_propose runs the selector on the drafter's device")
def test_fast_propose_torch_path_is_bitwise_equal(monkeypatch):
    full, pruned, pred_ids, succ_ids = selectors("cuda")
    monkeypatch.setattr(state_pool, "FUSED_PROPOSE", "0")
    batch, steps = 8, 3
    hidden = torch.randn(batch, steps, HIDDEN, device="cuda", dtype=torch.bfloat16)
    candidates, anchor, unary, _ = draws(pred_ids, succ_ids, batch, steps, "cuda")

    class Head:
        def topk(self, h, k):
            return unary.to(torch.bfloat16), candidates

    temperature = torch.zeros(batch, device="cuda")  # greedy: the walk does not draw
    out = []
    for selector in (full, pruned):
        model = SimpleNamespace(candidate_selector=selector, lm_head=Head())
        out.append(state_pool.fast_propose(model, hidden, anchor, temperature))
    for name in FIELDS:
        assert torch.equal(getattr(out[0], name), getattr(out[1], name)), name
