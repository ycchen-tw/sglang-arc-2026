"""arc-forge: NVFP4 storage for the offloaded Qwen4-Exp PLE table (opt-in, ``ARC_PLE_NVFP4=1``).

The pinned PLE table is the largest host allocation of a c110 server (47.7 GiB in the checkpoint's FP8 for
Qwen3.8-Flash-Next, next to a 75.5 GiB host KV pool inside Kaggle's 175 GiB cgroup). With this on, every row is stored as
NVFP4 -- packed e2m1 codes (element 2j in the low nibble of byte j, 2j+1 in the high one) plus one e4m3 scale per 16
elements. Codes and scales of a row sit together (``row_bytes``): 90 B per 160-wide row instead of 160 B, 26.8 GiB for the
table. The gather reads host memory over PCIe a row at a time, so a row's bytes must be one span. Measured on a 4090,
random rows of a 20M-row pinned table, per gather of 15.8k / 131k / 524k rows: FP8 388 / 3,183 / 12,730 us; NVFP4 with
codes and scales in two tables 431 / 3,537 / 14,157 (10 % slower than FP8); one 90 B span 329 / 2,708 / 10,892 (15 %
faster); padded to 96 B 279 / 2,260 / 9,195. Host memory is what this is for, so no padding (the gather runs on the PLE
prefetch stream either way). The gather kernel decodes to bf16; the per-tensor ``weight_scale`` multiply downstream is
unchanged.

There is no second-level (global) scale. The source rows are e4m3 already (|x| <= 448; 24..288 in this checkpoint), so
the block scale amax / 6 <= 74.7 is itself e4m3-representable and, being a float, has the same relative precision at any
magnitude in the normal range. Only blocks whose amax is below 6 * 2^-6 get a subnormal (coarser) scale; they are counted
and logged at load.

Rows are quantized on the GPU, chunk by chunk, as the loader hands over checkpoint shards: the FP8 table is never
materialized in host memory. On Qwen3.8-FN-NVFP4-REAP320-arccode shards: relative RMS error 9.4-9.5 %, row cosine
0.9955 (0.1 % quantile 0.994) against the FP8 rows.

Refused, loudly: a non-pinned PLE backend; source rows that are not FP8; an embedding width not divisible by 16; a table
whose rows were not all loaded.
"""

from __future__ import annotations

import os

import torch

ENV = "ARC_PLE_NVFP4"
NAME = "NVFP4"
BLOCK = 16
E2M1_MAX = 6.0
# e2m1 magnitudes are [0, .5, 1, 1.5, 2, 3, 4, 6] for codes 0..7; these are the decision boundaries between them
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_MIN_NORMAL = 2.0**-6
# rows per GPU quantization step: fp32 copy 160 MiB, int32 codes 160 MiB at width 160
CHUNK_ROWS = 1 << 18


def row_bytes(embedding_dim: int) -> int:
    """Bytes per stored row: d/2 of codes, then d/16 of scales."""
    return embedding_dim // 2 + embedding_dim // BLOCK


def enabled() -> bool:
    raw = os.environ.get(ENV, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


def check_table(backend: str, source_dtype: torch.dtype, embedding_dim: int) -> None:
    if backend != "pinned":
        raise ValueError(f"{ENV}=1 supports --ple-offload-backend pinned only, got {backend!r}")
    if source_dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{ENV}=1 needs an fp8 PLE table (text_config.ple_embedding_dtype float8_e4m3fn), got {source_dtype}"
        )
    if embedding_dim % BLOCK:
        raise ValueError(f"{ENV}=1 needs the PLE width divisible by {BLOCK}, got {embedding_dim}")


@torch.no_grad()
def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """[n, d] float rows (on the GPU) -> ([n, d/2] uint8 packed e2m1, [n, d/16] e4m3 scales, subnormal-scale blocks)."""
    n, d = x.shape
    blocks = x.float().view(n, d // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    if not bool(torch.isfinite(amax).all()):
        raise ValueError(f"{ENV}: non-finite PLE rows")
    peak = float(amax.max()) if amax.numel() else 0.0
    if peak > 448.0:
        raise ValueError(f"{ENV}: PLE rows exceed the e4m3 range (amax {peak})")
    scale = (amax / E2M1_MAX).to(torch.float8_e4m3fn)
    s = scale.float().unsqueeze(-1)
    q = torch.where(s > 0, blocks / s, torch.zeros_like(blocks))
    mids = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=x.device)
    code = torch.bucketize(q.abs(), mids, out_int32=True)
    code = (code | ((q < 0).to(torch.int32) << 3)).to(torch.uint8).view(n, d)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    subnormal = int(((amax > 0) & (amax / E2M1_MAX < _E4M3_MIN_NORMAL)).sum())
    return packed.contiguous(), scale, subnormal


def pack_rows(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``quantize_rows`` output -> [n, row_bytes] uint8, the stored layout."""
    n, half = packed.shape
    return torch.cat([packed, scale.view(torch.uint8)], dim=1)


def unpack_rows(rows: torch.Tensor, embedding_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    half = embedding_dim // 2
    return rows[:, :half], rows[:, half : half + embedding_dim // BLOCK].view(torch.float8_e4m3fn)


def dequantize_rows(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference decode of ``quantize_rows`` output -> [n, d] float32 (tests; the gather kernel does this on device)."""
    n, half = packed.shape
    p = packed.to(torch.int32)
    code = torch.stack([p & 0xF, p >> 4], dim=-1).view(n, 2 * half)
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)[code & 7]
    values = torch.where((code & 8) != 0, -values, values)
    return (values.view(n, -1, BLOCK) * scale.float().unsqueeze(-1)).view(n, 2 * half)
