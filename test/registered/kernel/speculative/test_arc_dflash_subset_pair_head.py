"""SubsetPairHead pads its rows to a multiple of 64 for an aligned GEMM: the zero rows must never be proposed (all
real logits negative here, so an unsliced zero logit would win) and the top-k must match the unpadded head's."""

import types

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.speculative.arc_dflash.heads import SubsetPairHead
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")


@pytest.mark.parametrize("n_ids", [131, 192])  # padded and already aligned
def test_subset_pair_head_padding(n_ids):
    g = torch.Generator(device="cuda").manual_seed(0)
    vocab, hidden = 1000, 256
    weight = torch.rand(vocab, hidden, generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.randperm(vocab, generator=g, device="cuda")[:n_ids].sort().values
    head = SubsetPairHead([(weight, 0)], ids, vocab, types.SimpleNamespace(world_size=1))
    assert head.sub.shape[0] % 64 == 0 and head.rows == n_ids
    x = -torch.rand(3, 5, hidden, generator=g, device="cuda").to(torch.bfloat16)  # every real logit < 0
    top, picked = head.topk(x, 64)
    ref_v, ref_local = F.linear(x.reshape(-1, hidden), weight.index_select(0, ids)).topk(64, -1)
    assert torch.isin(picked, ids).all()
    torch.testing.assert_close(top.reshape(-1, 64).float(), ref_v.float(), rtol=1e-2, atol=1e-2)
    agree = (picked.reshape(-1, 64) == ids[ref_local]).float().mean().item()
    assert agree > 0.95, agree
