"""ARC_TARGET_VOCAB (arc_dflash/target_vocab.py): the compact <-> vocabulary tables, the full-width -inf scatter, and the
drafter's SubsetPairHead on a restricted target slice (non-contiguous row ids)."""

import types

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.speculative.arc_dflash import target_vocab as tv
from sglang.srt.speculative.arc_dflash.heads import SubsetPairHead
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")


def _ids(vocab, n, seed):
    g = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(vocab, generator=g)[:n].tolist())


@pytest.fixture
def tables():
    vocab, ids = 1000, _ids(1000, 256, 0)
    tv._T.clear()
    tv._T.update(tv.build_tables(ids, vocab, 2, 1, "cuda"), head=None)
    yield vocab, ids
    tv._T.clear()


def test_tables_round_trip(tables):
    vocab, ids = tables
    t = tv.tables()
    c2g = torch.tensor(ids, device="cuda")
    assert torch.equal(t["c2g"], c2g) and t["size"] == 256 and t["width"] == 128
    assert torch.equal(t["mine"], c2g[128:])  # rank 1 of 2 holds the upper half
    assert torch.equal(tv.to_vocab(tv.to_compact(c2g)), c2g)
    off = torch.tensor(sorted(set(range(vocab)) - set(ids))[:5] + [-1], device="cuda")
    assert bool((tv.to_compact(off) == -1).all())  # off the list and "none" both map to -1
    small = torch.arange(256, dtype=torch.int32, device="cuda")
    assert tv.to_vocab(small).dtype == torch.int32


def test_expand(tables):
    vocab, ids = tables
    compact = torch.randn(3, 256, device="cuda").to(torch.bfloat16)
    full = tv.expand(compact, vocab)
    assert full.shape == (3, vocab) and full.dtype == torch.bfloat16
    keep = torch.tensor(ids, device="cuda")
    assert torch.equal(full[:, keep], compact)
    rest = torch.ones(vocab, dtype=torch.bool, device="cuda")
    rest[keep] = False
    assert bool(torch.isneginf(full[:, rest]).all())
    with pytest.raises(RuntimeError):
        tv.expand(compact[:, :128], vocab)


def test_unaligned_list_refused():
    with pytest.raises(ValueError):
        tv.build_tables(_ids(1000, 200, 1), 1000, 2, 0, "cuda")  # 200 rows do not split into 2 x 64k


def test_subset_head_on_a_restricted_slice():
    """The drafter's head over a target slice holding non-contiguous vocabulary ids: proposals are draft ids of this
    slice only, scored with the full head's rows for those ids."""
    g = torch.Generator(device="cuda").manual_seed(0)
    vocab, hidden = 1000, 256
    full = torch.rand(vocab, hidden, generator=g, device="cuda").to(torch.bfloat16)
    mine = torch.tensor(_ids(vocab, 384, 2), device="cuda")
    draft = torch.tensor(sorted(set(mine[::3].tolist()) | set(_ids(vocab, 50, 3))), device="cuda")
    head = SubsetPairHead([(full.index_select(0, mine), mine)], draft, vocab, types.SimpleNamespace(world_size=1))
    held = draft[torch.isin(draft, mine)]
    assert head.rows == held.numel() and torch.equal(head.map, held)
    x = -torch.rand(2, 4, hidden, generator=g, device="cuda").to(torch.bfloat16)
    top, picked = head.topk(x, 64)
    assert bool(torch.isin(picked, held).all())
    ref_v, ref_local = F.linear(x.reshape(-1, hidden), full.index_select(0, held)).topk(64, -1)
    torch.testing.assert_close(top.reshape(-1, 64).float(), ref_v.float(), rtol=1e-2, atol=1e-2)
    agree = (picked.reshape(-1, 64) == held[ref_local]).float().mean().item()
    assert agree > 0.95, agree
