"""QSA direct decode over a packed NVFP4 pool: the Ada-compatible E2M1 decodes (0 arithmetic, 1 fp32 bit
construction, 3 bf16 bit construction with a bf16 scale multiply) must give the same output bits, at verify-like
rows (several rows per request sharing a table row), split and unsplit, with invalid and out-of-length slots, for
every byte value of the packed K/V (all 16 nibbles in both halves).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.srt.layers.attention.qsa.sparse_attn import qsa_sparse_decode_direct


@pytest.mark.parametrize("chunk", [None, 128])
@pytest.mark.parametrize("stages", [2, 3])
def test_e2m1_decodes_bitwise(chunk, stages):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    req, per_req, q_heads, kv_heads, dim, hot, topk = 6, 4, 12, 1, 256, 1024, 320
    slots = req * hot
    k = torch.randint(0, 256, (slots, kv_heads, dim // 2), dtype=torch.uint8, device=dev)
    v = torch.randint(0, 256, (slots, kv_heads, dim // 2), dtype=torch.uint8, device=dev)
    k[:256, 0, 0] = torch.arange(256, dtype=torch.uint8, device=dev)  # every packed byte value
    v[:256, 0, 0] = torch.arange(256, dtype=torch.uint8, device=dev)
    k_sf = (torch.rand(slots, kv_heads, dim // 16, device=dev) * 4 + 2**-6).to(torch.float8_e4m3fn)
    v_sf = (torch.rand(slots, kv_heads, dim // 16, device=dev) * 4 + 2**-6).to(torch.float8_e4m3fn)
    table = torch.arange(slots, dtype=torch.int32, device=dev).view(req, hot)
    rows = req * per_req
    q = torch.randn(rows, q_heads, dim, device=dev, dtype=torch.bfloat16)
    table_rows = torch.arange(req, device=dev, dtype=torch.int32).repeat_interleave(per_req)
    lens = torch.full((rows,), hot, dtype=torch.int32, device=dev)
    lens[1] = hot // 2  # slots at or past the length are skipped
    indices = torch.stack([torch.randperm(hot, device=dev)[:topk] for _ in range(rows)]).to(torch.int32)
    indices[0, :256] = torch.arange(256, dtype=torch.int32, device=dev)
    indices[2, ::7] = -1  # invalid slots
    indices[3, :] = -1  # a row with no valid slot (a padded graph row) is zero
    outs = {
        d: qsa_sparse_decode_direct(q, k, v, k_sf, v_sf, table, table_rows, indices.contiguous(), lens, 0.0625,
                                    1.25, 0.75, block_n=64, warps=4, stages=stages, chunk=chunk, decode=d)
        for d in (0, 1, 3)
    }
    torch.cuda.synchronize()
    assert outs[0].abs().sum() > 0 and outs[0][3].abs().sum() == 0
    for d in (1, 3):
        assert torch.equal(outs[0].view(torch.int16), outs[d].view(torch.int16)), f"decode {d} differs from 0"


def test_decode_value_checked():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q = torch.zeros(1, 12, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="decode must be"):
        qsa_sparse_decode_direct(q, None, None, None, None, None, None, None, None, 1.0, decode=4)
