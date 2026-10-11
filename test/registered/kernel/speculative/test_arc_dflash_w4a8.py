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
