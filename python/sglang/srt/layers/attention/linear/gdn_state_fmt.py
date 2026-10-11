"""arc-forge h24: a packed ~10-bit GDN recurrent state (ARC_GDN_STATE_FMT, unset = the fp16 pool untouched).

The mamba pool's temporal state [layers, slots, HV, V, K] (fp16) becomes one byte tensor [layers, slots, HV, NB]; per
(slot, head) the NB bytes hold three planes, every value in a fixed value-axis Hadamard basis:
  hi  int8   [V, K]                  q >> LO_BITS (arithmetic shift)
  lo  uint8  [V, K * LO_BITS / 8]    q & (2^LO_BITS - 1); byte j of a value row holds, in bit field i, key channel
                                     i * K * LO_BITS / 8 + j (strided, so the kernels' 32-channel key blocks each
                                     own one bit field of a contiguous byte run)
  sc  fp32   [V / 64, K]             one scale per (key channel, 64 value rows): amax / QMAX
with q = round(x / sc) in [-QMAX, QMAX]. Formats:
  i10h  LO_BITS 2, QMAX 511    10.5 bits per value (NB = 21,504 for V = K = 128; 65.6% of fp16)
  i12h  LO_BITS 4, QMAX 2047   12.5 bits per value (NB = 25,600; 78.1%)
A scale that underflows (a group decayed to < 1e-30) stores zeros. Scales are fp32 so the group maximum keeps an
unbiased error: an fp16 scale rounded one way or the other shifts it the same way every requantization.

Basis: the value axis of each head is split in blocks of 64 and multiplied by the orthonormal Sylvester Hadamard
H64 (H = H^T = H^-1). The state is kept in that basis for good (DAMP's value-axis rotation): the GDN value input is
rotated before every kernel that updates the state and the kernel output is rotated back, so no kernel rotates the
state itself. Quantizing in the rotated basis spreads a value row's outliers over its block (real states, 400 folds:
relative state error 1.8% against 3.0% unrotated; bf16 2.6%, fp16 0.3%).

Runtime: rotate_v, gather_dequant and quantize_scatter are one Triton launch each (gdn_state_fmt_kernels.py);
quantize, dequantize and rotate_v_ref are the torch references they are checked against.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import torch

ENV = "ARC_GDN_STATE_FMT"
GROUP_V = 64  # value rows per scale; also the Hadamard block and the kernels' value tile
_FORMATS = {"i10h": 2, "i12h": 4}


@dataclass(frozen=True)
class StateFmt:
    name: str
    lo_bits: int

    @property
    def qmax(self) -> int:
        return 2 ** (7 + self.lo_bits) - 1

    @property
    def per_byte(self) -> int:  # lo values per byte
        return 8 // self.lo_bits

    def head_bytes(self, v: int, k: int) -> int:
        if v % GROUP_V or k % self.per_byte:
            raise ValueError(f"{self.name}: V={v} must be a multiple of {GROUP_V}, K={k} of {self.per_byte}")
        return v * k + v * k * self.lo_bits // 8 + (v // GROUP_V) * k * 4

    def offsets(self, v: int, k: int) -> tuple[int, int, int]:
        """byte offsets of hi, lo, sc inside one (slot, head) block"""
        lo = v * k
        sc = lo + v * k * self.lo_bits // 8
        if sc % 4:
            raise ValueError(f"{self.name}: scale plane at byte {sc} is not 4-aligned")
        return 0, lo, sc


@functools.cache
def current() -> StateFmt | None:
    raw = os.environ.get(ENV, "")
    if raw == "":
        return None
    if raw not in _FORMATS:
        raise ValueError(f"{ENV}={raw!r}: choose one of {sorted(_FORMATS)} (or leave it unset)")
    return StateFmt(raw, _FORMATS[raw])


def is_packed(t: torch.Tensor) -> bool:
    return t.dtype == torch.uint8


def planes(pool: torch.Tensor, fmt: StateFmt, v: int, k: int):
    """[..., HV, NB] uint8 -> (hi int8 [..., HV, V, K], lo uint8 [..., HV, V, K*lo/8], sc fp32 [..., HV, V/64, K])."""
    nb = fmt.head_bytes(v, k)
    if pool.dtype != torch.uint8 or pool.shape[-1] != nb:
        raise ValueError(f"not a {fmt.name} state: {pool.dtype} {tuple(pool.shape)}, need uint8 [..., HV, {nb}]")
    o_hi, o_lo, o_sc = fmt.offsets(v, k)
    lead = pool.shape[:-1]
    hi = pool[..., o_hi:o_lo].view(torch.int8).view(*lead, v, k)
    lo = pool[..., o_lo:o_sc].view(*lead, v, k * fmt.lo_bits // 8)
    sc = pool[..., o_sc:nb].view(torch.float32).view(*lead, v // GROUP_V, k)
    return hi, lo, sc


@torch.no_grad()
def quantize(x: torch.Tensor, fmt: StateFmt, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [..., HV, V, K] (float, already in the Hadamard basis) -> packed [..., HV, NB] uint8 (into out if given)."""
    *lead, v, k = x.shape
    x = x.float()
    blk = x.reshape(*lead, v // GROUP_V, GROUP_V, k)
    amax = blk.abs().amax(-2)  # [..., V/64, K]
    ok = amax > fmt.qmax * 1e-30
    s = torch.where(ok, amax / fmt.qmax, torch.zeros_like(amax))
    inv = torch.where(ok, fmt.qmax / amax, torch.zeros_like(amax))
    q = torch.floor(blk * inv[..., None, :] + 0.5).clamp_(-fmt.qmax, fmt.qmax).to(torch.int32)
    q = q.reshape(*lead, v, k)
    hi = torch.bitwise_right_shift(q, fmt.lo_bits)
    lo = q - hi * (1 << fmt.lo_bits)
    lo = lo.reshape(*lead, v, fmt.per_byte, k // fmt.per_byte)  # [.., field i, byte j]: key i * K / per + j
    shifts = torch.arange(fmt.per_byte, device=x.device, dtype=torch.int32)[:, None] * fmt.lo_bits
    packed_lo = (lo << shifts).sum(-2).to(torch.uint8)
    if out is None:
        out = torch.empty(*lead, fmt.head_bytes(v, k), dtype=torch.uint8, device=x.device)
    p_hi, p_lo, p_sc = planes(out, fmt, v, k)
    p_hi.copy_(hi.to(torch.int8))
    p_lo.copy_(packed_lo)
    p_sc.copy_(s)
    return out


@torch.no_grad()
def dequantize(packed: torch.Tensor, fmt: StateFmt, v: int, k: int) -> torch.Tensor:
    """packed [..., HV, NB] uint8 -> [..., HV, V, K] fp32 (still in the Hadamard basis)."""
    hi, lo, sc = planes(packed, fmt, v, k)
    shifts = torch.arange(fmt.per_byte, device=packed.device, dtype=torch.int32)[:, None] * fmt.lo_bits
    lo_vals = (lo.to(torch.int32)[..., None, :] >> shifts) & ((1 << fmt.lo_bits) - 1)  # [.., field i, byte j]
    lo_vals = lo_vals.reshape(*lo.shape[:-1], k)
    q = hi.to(torch.int32) * (1 << fmt.lo_bits) + lo_vals
    lead = q.shape[:-2]
    return (q.float().reshape(*lead, v // GROUP_V, GROUP_V, k) * sc[..., None, :]).reshape(*lead, v, k)


def _dev_key(device) -> str:
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


@functools.cache
def _hadamard_on(n: int, device: str) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device=device, dtype=torch.float32)


def hadamard(device) -> torch.Tensor:
    """The orthonormal H64 on a device, fp32 (built once; build it before any CUDA graph capture: prepare())."""
    return _hadamard_on(GROUP_V, _dev_key(device))


@functools.cache
def _hadamard_lp_on(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    return _hadamard_on(n, device).to(dtype)  # +-1/8: exact in bf16 and fp16


def hadamard_lp(device, dtype: torch.dtype) -> torch.Tensor:
    """H64 in an activation dtype, the rotation kernel's operand (exact)."""
    return _hadamard_lp_on(GROUP_V, _dev_key(device), dtype)


def prepare(device) -> None:
    """Build the device constants outside graph capture (a host->device copy cannot be captured)."""
    hadamard(device)
    for dtype in (torch.bfloat16, torch.float16):
        hadamard_lp(device, dtype)


def _rows(x: torch.Tensor) -> torch.Tensor:
    """x as a 2-D [rows, cols] view with unit column stride, cols a multiple of GROUP_V (no copy)."""
    for cols in ((x.shape[-2] * x.shape[-1],) if x.dim() >= 2 else ()) + (x.shape[-1],):
        try:
            x2 = x.view(-1, cols)
        except RuntimeError:
            continue
        if x2.stride(1) == 1:
            return x2
    raise ValueError(f"rotate_v: shape {tuple(x.shape)} strides {x.stride()} has no row view with unit stride")


def rotate_v(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Multiply the last (value) axis by blockdiag(H64): the basis change, its own inverse. One kernel: exact products,
    fp32 sums, x's dtype out (bf16 / fp16). out=x rotates in place; out=None returns a new contiguous tensor."""
    from sglang.kernels.ops.attention.fla import gdn_state_fmt_kernels as kernels

    if x.shape[-1] % GROUP_V:
        raise ValueError(f"value axis {x.shape[-1]} is not a multiple of {GROUP_V}")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"rotate_v takes bf16 / fp16 activations, got {x.dtype}")
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError(f"rotate_v: out {tuple(out.shape)} {out.dtype} for x {tuple(x.shape)} {x.dtype}")
    x2 = _rows(x)
    kernels.launch_rotate_v(x2, x2 if out is x else _rows(out), hadamard_lp(x.device, x.dtype))
    return out


def rotate_v_ref(x: torch.Tensor) -> torch.Tensor:
    """rotate_v in torch, fp32 math (the reference the kernel is checked against)."""
    v = x.shape[-1]
    if v % GROUP_V:
        raise ValueError(f"value axis {v} is not a multiple of {GROUP_V}")
    h = hadamard(x.device)
    y = x.float().reshape(*x.shape[:-1], v // GROUP_V, GROUP_V) @ h
    return y.reshape(x.shape).to(x.dtype)


def gather_dequant(pool: torch.Tensor, idx: torch.Tensor, fmt: StateFmt, v: int, k: int,
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """pool (per layer, [slots, HV, NB] uint8)[idx] dequantized -> [n, HV, V, K] of dtype, one kernel
    (== dequantize(pool[idx]).to(dtype))."""
    from sglang.kernels.ops.attention.fla import gdn_state_fmt_kernels as kernels

    out = torch.empty(idx.shape[0], pool.shape[1], v, k, dtype=dtype, device=pool.device)
    kernels.launch_gather_dequant(fmt, pool, idx, out)
    return out


def quantize_scatter(x: torch.Tensor, pool: torch.Tensor, dst: torch.Tensor, fmt: StateFmt,
                     src: torch.Tensor | None = None) -> None:
    """pool[dst] = quantize(x[src]) (x without src), one kernel; x [rows, HV, V, K] float, rows contiguous."""
    from sglang.kernels.ops.attention.fla import gdn_state_fmt_kernels as kernels

    kernels.launch_quantize_scatter(fmt, x, pool, dst, src)


def check_server_args(cfg) -> None:
    """Refuse every configuration h24 was not built for (called from attention_hook with h20's check)."""
    fmt = current()
    if fmt is None:
        return
    need = {
        "ARC_GDN_FOLD_IN_VERIFY=1 or check (the h20 verify is the only kernel that folds a packed state)": os.environ.get(
            "ARC_GDN_FOLD_IN_VERIFY", "0") in ("1", "check"),
        "--mamba-ssm-dtype float16 (the kernels' activation-side state dtype)": cfg.mamba_ssm_dtype == "float16",
        "--tp-size 1": cfg.tp_size == 1,
        "speculative decoding (every decode step is a ReplaySSM verify)": cfg.speculative_algorithm is not None,
        "no --enable-hierarchical-cache": not cfg.enable_hierarchical_cache,
        "no --enable-int8-mamba-checkpoint": not cfg.enable_int8_mamba_checkpoint,
        "no --enable-page-major-kv-layout": not cfg.enable_page_major_kv_layout,
        "no --enable-mis": not cfg.enable_mis,
        "--disaggregation-mode null": cfg.disaggregation_mode == "null",
    }
    missing = [what for what, ok in need.items() if not ok]
    if missing:
        raise ValueError(f"{ENV}={fmt.name} requires: " + "; ".join(missing))
