"""arc DFlash on B200: block-FP8 drafter linears on sm_100, and the FP8 window pool (ARC_DFLASH_KV_DTYPE=fp8_e4m3).

Ring attention over an e4m3 pool must match ring attention over a BF16 pool holding the same rounded values up to
an output rounding (the kernel widens each tile on load). Fp8BlockLinear on sm_100 must take the
FlashInfer TRTLLM or Triton block GEMM, compute the dequantized weights' product, and refuse DeepGEMM (auto).
"""

import pytest
import torch

from sglang.srt.speculative.arc_dflash import fp8, state_pool
from sglang.srt.speculative.arc_dflash.ring_attention import RingWindow, attend

CUDA = torch.cuda.is_available()
SM100 = CUDA and torch.cuda.get_device_capability()[0] == 10


def test_to_pool_saturates():
    x = torch.tensor([-1e4, -3.0, 0.0, 0.01, 500.0], dtype=torch.bfloat16)
    y = state_pool.to_pool(x, torch.float8_e4m3fn)
    assert y.dtype == torch.float8_e4m3fn and torch.isfinite(y.float()).all()
    assert y.float()[0] == -448 and y.float()[-1] == 448
    assert state_pool.to_pool(x, torch.bfloat16) is x


def _case(batch=6, slots=9, hq=12, hkv=2, n=9, window=64, dim=256, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    cols = window + 1
    pool_k = torch.randn(slots, hkv, cols + 2 * n, dim, device=dev, generator=g).to(torch.bfloat16) * 2
    pool_v = torch.randn(slots, hkv, cols + 2 * n, dim, device=dev, generator=g).to(torch.bfloat16)
    idx = torch.randperm(slots, device=dev, generator=g)[:batch]
    end = torch.randint(window // 2, 4 * window, (batch,), device=dev, generator=g)
    pos = torch.full((batch, cols), -1, dtype=torch.long, device=dev)
    for b in range(batch):  # a ring holding positions end-window+1 .. end at column p % window, some unwritten
        p = torch.arange(int(end[b]) - window + 1, int(end[b]) + 1, device=dev)
        p = p[p >= 0][3:]
        pos[b, p % window] = p
    q = torch.randn(batch, hq, n, dim, device=dev, generator=g).to(torch.bfloat16)
    k = torch.randn(batch, hkv, n, dim, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(batch, hkv, n, dim, device=dev, generator=g).to(torch.bfloat16)
    qpos = end[:, None] + 1 + torch.arange(n, device=dev)
    return q, k, v, qpos, pool_k, pool_v, idx, pos, window


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("seed", [0, 1])
def test_ring_attention_fp8_pool_equals_rounded_bf16(seed):
    q, k, v, qpos, pk, pv, idx, pos, window = _case(seed=seed)
    cols = window + 1
    pk8, pv8 = state_pool.to_pool(pk, torch.float8_e4m3fn), state_pool.to_pool(pv, torch.float8_e4m3fn)
    rk, rv = pk8.to(torch.bfloat16), pv8.to(torch.bfloat16)
    got = attend(q, k, v, qpos, RingWindow(pk8, idx, cols), RingWindow(pv8, idx, cols), pos, window)
    want = attend(q, k, v, qpos, RingWindow(rk, idx, cols), RingWindow(rv, idx, cols), pos, window)
    # the e4m3-pointer specialization is a different compiled kernel: the same values up to an output rounding
    # (B200: ~2e-5 of the elements one BF16 ulp apart), not bitwise
    diff = (got.float() - want.float()).abs()
    assert (diff > 0).float().mean().item() < 1e-3
    assert torch.allclose(got.float(), want.float(), rtol=2 ** -7, atol=2 ** -9)
    exact = attend(q, k, v, qpos, RingWindow(pk, idx, cols), RingWindow(pv, idx, cols), pos, window)
    err = ((got.float() - exact.float()).norm() / exact.float().norm()).item()
    assert 0 < err < 0.08, err  # e4m3 K/V (3 mantissa bits) on Gaussian data


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_ring_attention_rejects_other_pool_dtypes():
    q, k, v, qpos, pk, pv, idx, pos, window = _case()
    cols = window + 1
    with pytest.raises(ValueError, match="e4m3"):
        attend(q, k, v, qpos, RingWindow(pk.half(), idx, cols), RingWindow(pv.half(), idx, cols), pos, window)


def _fp8_weight(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    blocks = w.view(n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax((1, 3)).clamp(min=1e-12) / 448
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.float().contiguous()


@pytest.mark.skipif(not SM100, reason="needs sm_100")
@pytest.mark.parametrize("backend", ["flashinfer_trtllm", "triton"])
@pytest.mark.parametrize("n,k,m", [(1024, 2560, 1), (2560, 1024, 37), (6144, 2560, 84), (2560, 10240, 512)])
def test_fp8_block_linear_sm100(monkeypatch, backend, n, k, m):
    from sglang.srt.layers.quantization import fp8_utils

    monkeypatch.setattr(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", fp8_utils.Fp8GemmRunnerBackend(backend))
    w, s = _fp8_weight(n, k)
    ref_w = fp8.dequant_block_fp8(w, s).float()
    lin = fp8.Fp8BlockLinear(w, s)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    y = lin(x)
    ref = x.float() @ ref_w.T
    err = ((y.float() - ref).norm() / ref.norm()).item()
    assert y.dtype == torch.bfloat16 and y.shape == (m, n)
    assert err < 0.05, err  # dynamic per-token-group e4m3 activations: ~0.03


@pytest.mark.skipif(not SM100, reason="needs sm_100")
def test_fp8_block_linear_sm100_refuses_deepgemm(monkeypatch):
    from sglang.srt.layers.quantization import fp8_utils

    monkeypatch.setattr(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", fp8_utils.Fp8GemmRunnerBackend.AUTO)
    w, s = _fp8_weight(256, 256)
    if fp8_utils.dispatch_w8a8_block_fp8_linear() in (fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback,
                                                      fp8_utils.triton_w8a8_block_fp8_linear):
        pytest.skip("auto already picks an accepted path here")
    with pytest.raises(RuntimeError, match="measured paths"):
        fp8.Fp8BlockLinear(w, s)
