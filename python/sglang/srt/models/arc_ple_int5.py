"""arc-forge: INT5 storage for the offloaded Qwen4-Exp PLE table (opt-in, ``ARC_PLE_INT5=1``).

Same idea and same plumbing as ``arc_ple_nvfp4`` (read that docstring first), with a finer code: every row is stored as
symmetric INT5 in groups of 16 elements, one e4m3 scale per group. ``scale = amax / 15`` rounded to e4m3, and the codes are
chosen against the *rounded* scale: ``code = clamp(round(x / scale), -15, 15)``, stored unsigned as ``code + 16`` in
[1, 31] (0 never occurs). Eight codes make five bytes -- little-endian, bit-continuous: element ``i`` of a group of eight
occupies bits ``5 i .. 5 i + 4`` of the 40-bit word whose bytes are 5 g .. 5 g + 4. A row is its ``d * 5 / 8`` code bytes
followed by ``d / 16`` scale bytes, one contiguous span (the gather reads host memory over PCIe a row at a time, so a row
must not be split across two tables): 110 B for the 160-wide PLE row instead of 160 B (FP8) or 90 B (NVFP4), 32.8 GiB for
the 47.7 GiB FP8 table. The gather kernel decodes to bf16; ``code * scale`` has at most 8 significant bits, so it is exact
in bf16 and the per-tensor ``weight_scale`` multiply downstream is unchanged.

Why INT5 rather than NVFP4: the 15 uniform levels per group beat e2m1's 8 magnitudes on rows that are roughly Gaussian
within a group -- on ``Qwen3.8-FN-NVFP4-EAN320-v1`` layer 1 ``ngram_embedding.shard_7`` (first 400k rows) the relative RMS
error against the FP8 rows is 4.17 % (NVFP4 9.59 %), row cosine 0.99915 (0.1 % quantile 0.99872; NVFP4 0.99548 / 0.99363)
-- for 22 % more host bytes than NVFP4.

Gather speed, B200 (GPU 2), random rows of a 20M-row pinned table, median per gather of 15.8k / 131k / 524k rows: FP8 150 /
1,158 / 4,638 us; NVFP4 117 / 890 / 3,536; INT5 128 / 996 / 4,074 (12-15 % faster than FP8). The same kernels over
device-resident tables take the same time for all three formats (15 / 74 / 277-281 us): the decode is free, the gather is
bound by the PCIe reads, i.e. by bytes per row. Rows are not padded (host memory is the point); padding the 110 B stride
to 112 B measured 9 % faster still (906 vs 987 us at 131k rows), 128 B no faster than 110 B.

As for NVFP4 there is no second-level scale: the source is e4m3 (|x| <= 448), so ``amax / 15 <= 29.87`` is e4m3-
representable at full relative precision in the normal range. Only groups whose ``amax / 15`` is below 2^-6 get a
subnormal (coarser) scale; they are counted and logged at load (a group so small that its scale rounds to zero decodes to
zero).

Rows are quantized on the GPU, chunk by chunk, as the loader hands over checkpoint shards: the FP8 table is never
materialized in host memory.

Refused, loudly: ``ARC_PLE_INT5`` together with ``ARC_PLE_NVFP4``; a non-pinned PLE backend; source rows that are not FP8;
an embedding width not divisible by 16; a table whose rows were not all loaded.
"""

from __future__ import annotations

import os

import torch

ENV = "ARC_PLE_INT5"
# the other codec: switching both on is a configuration error (read here so this module stays the only place that knows)
_NVFP4_ENV = "ARC_PLE_NVFP4"
NAME = "INT5"
BLOCK = 16
QMAX = 15
# stored unsigned code = signed code + _ZERO, in [1, 31]
_ZERO = 16
_E4M3_MIN_NORMAL = 2.0**-6
# rows per GPU quantization step: fp32 copy 160 MiB, int64 packing words 80 MiB at width 160
CHUNK_ROWS = 1 << 18


def _check_width(embedding_dim: int) -> None:
    if embedding_dim <= 0 or embedding_dim % BLOCK:
        raise ValueError(f"{ENV}=1 needs the PLE width divisible by {BLOCK}, got {embedding_dim}")


def code_bytes(embedding_dim: int) -> int:
    """Bytes of packed codes per row: 5 for every 8 elements."""
    _check_width(embedding_dim)
    return embedding_dim * 5 // 8


def row_bytes(embedding_dim: int) -> int:
    """Bytes per stored row: d * 5 / 8 of codes, then d / 16 of scales."""
    return code_bytes(embedding_dim) + embedding_dim // BLOCK


def enabled() -> bool:
    raw = os.environ.get(ENV, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{ENV}={raw!r}: expected 0 or 1")
    if raw == "1":
        other = os.environ.get(_NVFP4_ENV, "0").strip()
        if other not in ("0", "1"):
            raise ValueError(f"{_NVFP4_ENV}={other!r}: expected 0 or 1")
        if other == "1":
            raise ValueError(f"{ENV}=1 and {_NVFP4_ENV}=1 are both set: pick one PLE storage format")
    return raw == "1"


def check_table(backend: str, source_dtype: torch.dtype, embedding_dim: int) -> None:
    if backend != "pinned":
        raise ValueError(f"{ENV}=1 supports --ple-offload-backend pinned only, got {backend!r}")
    if source_dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"{ENV}=1 needs an fp8 PLE table (text_config.ple_embedding_dtype float8_e4m3fn), got {source_dtype}"
        )
    _check_width(embedding_dim)


def pack_codes(unsigned: torch.Tensor) -> torch.Tensor:
    """[n, d] unsigned 5-bit values (0..31) -> [n, d * 5 / 8] uint8, eight values per five bytes, little-endian."""
    n, d = unsigned.shape
    _check_width(d)
    groups = unsigned.to(torch.int64).view(n, d // 8, 8)
    word = torch.zeros((n, d // 8), dtype=torch.int64, device=unsigned.device)
    for i in range(8):
        word |= groups[..., i] << (5 * i)
    out = torch.stack([(word >> (8 * k)) & 0xFF for k in range(5)], dim=-1)
    return out.to(torch.uint8).view(n, d * 5 // 8)


def unpack_codes(packed: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Inverse of ``pack_codes``: [n, d * 5 / 8] uint8 -> [n, d] int32 unsigned values (0..31)."""
    n = packed.shape[0]
    if packed.shape[1] != code_bytes(embedding_dim):
        raise ValueError(f"packed width {packed.shape[1]} != {code_bytes(embedding_dim)} for d={embedding_dim}")
    b = packed.to(torch.int64).view(n, embedding_dim // 8, 5)
    word = torch.zeros((n, embedding_dim // 8), dtype=torch.int64, device=packed.device)
    for k in range(5):
        word |= b[..., k] << (8 * k)
    out = torch.stack([(word >> (5 * i)) & 0x1F for i in range(8)], dim=-1)
    return out.to(torch.int32).view(n, embedding_dim)


@torch.no_grad()
def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """[n, d] float rows (any device) -> ([n, d*5/8] uint8 packed codes, [n, d/16] e4m3 scales, subnormal-scale blocks)."""
    n, d = x.shape
    _check_width(d)
    blocks = x.float().view(n, d // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    if not bool(torch.isfinite(amax).all()):
        raise ValueError(f"{ENV}: non-finite PLE rows")
    peak = float(amax.max()) if amax.numel() else 0.0
    if peak > 448.0:
        raise ValueError(f"{ENV}: PLE rows exceed the e4m3 range (amax {peak})")
    scale = (amax / QMAX).to(torch.float8_e4m3fn)
    s = scale.float().unsqueeze(-1)
    q = torch.where(s > 0, blocks / s, torch.zeros_like(blocks))
    code = q.round().clamp_(-QMAX, QMAX).to(torch.int32) + _ZERO
    packed = pack_codes(code.view(n, d))
    subnormal = int(((amax > 0) & (amax / QMAX < _E4M3_MIN_NORMAL)).sum())
    return packed, scale, subnormal


def pack_rows(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``quantize_rows`` output -> [n, row_bytes] uint8, the stored layout."""
    return torch.cat([packed, scale.view(torch.uint8)], dim=1)


def unpack_rows(rows: torch.Tensor, embedding_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cb = code_bytes(embedding_dim)
    return rows[:, :cb], rows[:, cb : cb + embedding_dim // BLOCK].view(torch.float8_e4m3fn)


def dequantize_rows(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference decode of ``quantize_rows`` output -> [n, d] float32 (tests; the gather kernel does this on device)."""
    n = packed.shape[0]
    d = scale.shape[1] * BLOCK
    unsigned = unpack_codes(packed, d)
    if unsigned.numel() and (int(unsigned.min()) < 1 or int(unsigned.max()) > 2 * QMAX + 1):
        raise ValueError(f"{ENV}: stored code outside [1, 31]: corrupt row")
    code = (unsigned - _ZERO).float()
    return (code.view(n, -1, BLOCK) * scale.float().unsqueeze(-1)).view(n, d)
