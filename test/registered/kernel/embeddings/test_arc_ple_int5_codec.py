"""arc_ple_int5 codec (pack / quantize / dequantize): CPU and GPU, no model needed."""

import pytest
import torch

from sglang.srt.models import arc_ple_int5
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _py_pack(row):
    """Independent reference: eight 5-bit values -> five bytes, little-endian, in plain python ints."""
    out = bytearray()
    for g in range(0, len(row), 8):
        word = 0
        for i in range(8):
            word |= int(row[g + i]) << (5 * i)
        out += word.to_bytes(5, "little")
    return bytes(out)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d", [16, 48, 160])
def test_pack_unpack_roundtrip_bit_exact(device, d):
    g = torch.Generator().manual_seed(d)
    unsigned = torch.randint(0, 32, (64, d), generator=g, dtype=torch.int32)
    packed = arc_ple_int5.pack_codes(unsigned.to(device))
    assert packed.dtype == torch.uint8 and packed.shape == (64, d * 5 // 8)
    for r in range(64):
        assert bytes(packed[r].cpu().tolist()) == _py_pack(unsigned[r].tolist())
    assert torch.equal(arc_ple_int5.unpack_codes(packed, d).cpu(), unsigned)


@pytest.mark.parametrize("device", DEVICES)
def test_every_code_value_in_every_slot(device):
    # all 32 values at all 8 positions of a group, the others filled with a different value
    rows = []
    for v in range(32):
        for slot in range(8):
            row = [(v + 11) % 32] * 8
            row[slot] = v
            rows.append(row * 2)
    unsigned = torch.tensor(rows, dtype=torch.int32)
    packed = arc_ple_int5.pack_codes(unsigned.to(device))
    assert torch.equal(arc_ple_int5.unpack_codes(packed, 16).cpu(), unsigned)
    assert bytes(packed[0].cpu().tolist()) == _py_pack(unsigned[0].tolist())
    # element 0 is the low 5 bits of byte 0; element 1 straddles bytes 0 and 1
    one = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8] * 2], dtype=torch.int32)
    b = arc_ple_int5.pack_codes(one.to(device)).cpu().tolist()[0]
    assert b[0] == 1 | ((2 & 0b111) << 5) and b[1] == (2 >> 3) | (3 << 2) | ((4 & 1) << 7)


@pytest.mark.parametrize("device", DEVICES)
def test_quantize_dequantize_known_values(device):
    # a group whose amax is 15 * 2^k has scale 2^k exactly: codes are the integers themselves
    grid = torch.arange(-15, 16, dtype=torch.float32)  # 31 values
    row = torch.cat([grid, torch.tensor([15.0])]).view(1, 32).repeat(1, 5)  # d = 160
    row = row.to(torch.float8_e4m3fn).to(device)
    packed, scales, subnormal = arc_ple_int5.quantize_rows(row)
    assert torch.all(scales.float() == 1.0) and subnormal == 0
    assert torch.equal(arc_ple_int5.dequantize_rows(packed, scales).cpu(), row.float().cpu())
    # stored unsigned = code + 16
    assert torch.equal(
        arc_ple_int5.unpack_codes(packed, 160)[0, :31].cpu(), (grid + 16).to(torch.int32)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_edge_rows(device):
    d = 160
    rows = torch.zeros((6, d))
    rows[1] = 448.0  # max
    rows[2] = -448.0
    rows[3, :16] = torch.tensor([448.0, -448.0] * 8)
    rows[3, 16:32] = 0.05  # subnormal scale (0.05 / 15 < 2^-6)
    rows[4, :16] = 2.0**-9  # smallest e4m3 subnormal: scale rounds to zero, decodes to zero
    rows[5] = torch.tensor([1.0, -1.0] * 80)
    rows = rows.to(torch.float8_e4m3fn).to(device)
    packed, scales, subnormal = arc_ple_int5.quantize_rows(rows)
    out = arc_ple_int5.dequantize_rows(packed, scales)
    assert torch.all(out[0] == 0)
    assert subnormal == 2  # row 3's 0.05 block and row 4's 2^-9 block; all-zero blocks are not counted
    # 448 / 15 -> e4m3 scale 30; code 15 -> 450 (the clamp handles 448 / 30 = 14.93 rounding to 15)
    assert torch.all(scales[1].float() == 30.0) and torch.all(out[1] == 450.0) and torch.all(out[2] == -450.0)
    assert torch.all(out[4, :16] == 0)
    # the subnormal block: error is at most half a step of its (coarse) scale
    s = scales[3, 1].float()
    assert 0 < float(s) < arc_ple_int5._E4M3_MIN_NORMAL
    assert float((out[3, 16:32] - rows[3, 16:32].float()).abs().max()) <= float(s) / 2 + 1e-9
    # +-1 in a group: amax / 15 = 1/15 -> e4m3 scale ~0.0664; codes +-15; all reconstruct within half a step
    err = (out[5] - rows[5].float()).abs().max()
    assert float(err) <= float(scales[5, 0].float()) / 2 + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_and_gpu_quantize_agree():
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(257, 160, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    cpu = arc_ple_int5.quantize_rows(x)
    gpu = arc_ple_int5.quantize_rows(x.cuda())
    assert torch.equal(cpu[0], gpu[0].cpu())
    assert torch.equal(cpu[1].view(torch.uint8), gpu[1].view(torch.uint8).cpu())
    assert cpu[2] == gpu[2]


@pytest.mark.parametrize("device", DEVICES)
def test_stored_row_layout_and_refusals(device):
    assert arc_ple_int5.row_bytes(160) == 110 and arc_ple_int5.code_bytes(160) == 100
    x = (torch.randn(5, 160) * 30).to(torch.float8_e4m3fn).to(device)
    packed, scales, _ = arc_ple_int5.quantize_rows(x)
    rows = arc_ple_int5.pack_rows(packed, scales)
    assert rows.shape == (5, 110) and rows.dtype == torch.uint8
    p2, s2 = arc_ple_int5.unpack_rows(rows, 160)
    assert torch.equal(p2, packed) and torch.equal(s2.view(torch.uint8), scales.view(torch.uint8))
    for bad in (8, 24, 100, 0):
        with pytest.raises(ValueError, match="divisible by 16"):
            arc_ple_int5.row_bytes(bad)
        with pytest.raises(ValueError, match="divisible by 16"):
            arc_ple_int5.check_table("pinned", torch.float8_e4m3fn, bad)
    with pytest.raises(ValueError, match="divisible by 16"):
        arc_ple_int5.quantize_rows(torch.zeros((2, 24)).to(torch.float8_e4m3fn).to(device))
    with pytest.raises(ValueError, match="pinned only"):
        arc_ple_int5.check_table("file", torch.float8_e4m3fn, 160)
    with pytest.raises(TypeError, match="needs an fp8 PLE table"):
        arc_ple_int5.check_table("pinned", torch.bfloat16, 160)
    with pytest.raises(ValueError, match="non-finite"):
        arc_ple_int5.quantize_rows(torch.tensor([[float("nan")] * 16], device=device))
    with pytest.raises(ValueError, match="exceed the e4m3 range"):
        arc_ple_int5.quantize_rows(torch.full((1, 16), 449.0, device=device))
    corrupt = packed.clone()
    corrupt[0, :] = 0  # unsigned 0 never occurs in a stored row
    with pytest.raises(ValueError, match="corrupt row"):
        arc_ple_int5.dequantize_rows(corrupt, scales)


def test_env_switch(monkeypatch):
    monkeypatch.delenv("ARC_PLE_INT5", raising=False)
    monkeypatch.delenv("ARC_PLE_NVFP4", raising=False)
    assert arc_ple_int5.enabled() is False
    monkeypatch.setenv("ARC_PLE_INT5", "1")
    assert arc_ple_int5.enabled() is True
    monkeypatch.setenv("ARC_PLE_NVFP4", "0")
    assert arc_ple_int5.enabled() is True
    monkeypatch.setenv("ARC_PLE_NVFP4", "1")
    with pytest.raises(ValueError, match="both set"):
        arc_ple_int5.enabled()
    monkeypatch.setenv("ARC_PLE_NVFP4", "yes")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        arc_ple_int5.enabled()
    monkeypatch.setenv("ARC_PLE_NVFP4", "0")
    monkeypatch.setenv("ARC_PLE_INT5", "2")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        arc_ple_int5.enabled()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
