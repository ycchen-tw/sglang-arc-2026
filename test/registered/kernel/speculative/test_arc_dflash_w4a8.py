"""arc DFlash ARC_DFLASH_W4A8: drafter linears as Humming INT4 weights x per-token FP8 activations.

Lossy by design (only the proposals change); the checks are the spec parser, the module's output against the BF16
GEMM of the same FP8 weights at INT4's error level, and the 0-row IDLE batch.
"""

import os

import pytest
import torch

from sglang.srt.speculative.arc_dflash import fp8


def test_spec_parser(monkeypatch):
    monkeypatch.delenv("ARC_DFLASH_W4A8", raising=False)
    assert fp8.w4a8_spec() is None
    monkeypatch.setenv("ARC_DFLASH_W4A8", "int4:128")
    assert fp8.w4a8_spec() == ("int4", False, 128, "all")
    monkeypatch.setenv("ARC_DFLASH_W4A8", "uint4z:32:mlp")
    assert fp8.w4a8_spec() == ("uint4", True, 32, "mlp")
    for bad in ("int4", "int3:128", "int4:96", "int4:128:attn", "int4:128:all:x"):
        monkeypatch.setenv("ARC_DFLASH_W4A8", bad)
        with pytest.raises(ValueError, match="ARC_DFLASH_W4A8"):
            fp8.w4a8_spec()


def test_dequant_block_fp8():
    w = torch.randn(256, 384).to(torch.float8_e4m3fn)
    s = torch.rand(2, 3) + 0.5
    want = w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(fp8.dequant_block_fp8(w, s), want.to(torch.bfloat16))
    with pytest.raises(ValueError, match="block FP8"):
        fp8.dequant_block_fp8(w, s[:, :2])


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("ARC_HUMMING_DIR"),
                    reason="needs a GPU and ARC_HUMMING_DIR (unpacked humming-kernels)")
@pytest.mark.parametrize("spec", [("int4", False, 128, "all"), ("uint4", True, 64, "all")])
def test_humming_w4a8_linear(spec):
    torch.manual_seed(0)
    weight = (torch.randn(1024, 2560, device="cuda") * 0.02).to(torch.bfloat16)
    lin = fp8.HummingW4A8Linear(weight, spec)
    assert (lin.out_features, lin.in_features) == (1024, 2560)
    x = torch.randn(3, 128, 2560, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ weight.float().T
    y = lin(x)
    assert y.shape == (3, 128, 1024) and y.dtype == torch.bfloat16
    err = ((y.float() - ref).norm() / ref.norm()).item()
    assert err < 0.15, err  # INT4 RTN on Gaussian weights: ~0.11 at group 128
    assert lin(x[:0]).shape == (0, 128, 1024)
    with pytest.raises(ValueError, match="BF16"):
        fp8.HummingW4A8Linear(weight.float(), spec)


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("ARC_HUMMING_DIR"),
                    reason="needs a GPU and ARC_HUMMING_DIR (unpacked humming-kernels)")
@pytest.mark.parametrize("n,k,m", [(2560, 10240, 48), (6144, 2560, 384), (10240, 2560, 512)])
def test_humming_w4a8_is_deterministic_and_batch_invariant(n, k, m):
    """The pair's two ranks compute the replicated drafter ops each on their own: every repeat must be bitwise equal
    (Humming's default stream-K was not: down_proj K=10240 at M=48, q_proj at M=384), and a row's output must not
    depend on the rows batched with it."""
    torch.manual_seed(0)
    lin = fp8.HummingW4A8Linear((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16), ("int4", False, 128, "all"))
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = lin(x).clone()
    for _ in range(20):
        torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda")
        assert torch.equal(lin(x), ref)
    assert torch.equal(lin(x[: m // 2]), ref[: m // 2])
