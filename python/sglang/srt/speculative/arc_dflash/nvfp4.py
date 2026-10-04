"""NVFP4 routed experts for the drafter on the target's own MoE kernel (FlashInfer CUTLASS, W4A4).

Weights use ModelOpt's two-level NVFP4: E2M1 values in 16-element blocks, one FP8 E4M3 scale per block and one
FP32 scale per matrix (amax / (448 * 6)). The kernel layout is what SGLang's ModelOptNvFp4FusedMoEMethod builds
for --moe-runner-backend flashinfer_cutlass: w13 = [up; gate] per expert, block scales swizzled into 128x4 tiles,
quant_scales = [1 / a1, w13 scales, g1 alphas, 1 / a2, w2 scales, g2 alphas].

Activations: GEMM1's input scale is computed per call from the batch's own amax (device-side, no host sync, CUDA
graph safe), so it needs no calibration. GEMM2's input is the SiLU-gated intermediate inside the kernel, so its
scale is static per layer and comes from calibration (`a2_amax`).
"""

from __future__ import annotations

import torch
from torch import nn

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX, FP8_MAX, BLOCK = 6.0, 448.0, 16


@torch.no_grad()
def quantize(w: torch.Tensor, chunk: int = 16):
    """[..., rows, k] float -> (packed uint8 [..., rows, k/2], block scale fp8 [..., rows, k/16], scale2 fp32 [...]).

    Round-to-nearest onto the E2M1 grid; element 2i in the low nibble, 2i+1 in the high one (ModelOpt's order).
    Leading matrices are processed `chunk` at a time so the FP32 working copies stay small.
    """
    if w.shape[-1] % BLOCK:
        raise ValueError(f"k={w.shape[-1]} is not a multiple of {BLOCK}")
    if w.ndim > 2 and w.shape[0] > chunk:
        parts = [quantize(w[i : i + chunk], chunk) for i in range(0, w.shape[0], chunk)]
        return tuple(torch.cat(x) for x in zip(*parts))
    w = w.float()
    scale2 = w.abs().amax(dim=(-2, -1)).clamp_min(1e-12) / (FP8_MAX * FP4_MAX)
    blocks = w.unflatten(-1, (-1, BLOCK))
    block_amax = blocks.abs().amax(-1)
    sf = (block_amax / FP4_MAX / scale2[..., None, None]).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    step = (sf.float() * scale2[..., None, None]).unsqueeze(-1)
    q = torch.where(step > 0, blocks / step.clamp_min(1e-30), torch.zeros_like(blocks))
    grid = torch.tensor(E2M1, device=w.device)
    code = torch.bucketize(q.abs(), (grid[1:] + grid[:-1]) / 2).clamp_(max=7).to(torch.uint8)
    code |= (q < 0).to(torch.uint8) << 3
    code = code.flatten(-2)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), sf.contiguous(), scale2.float()


@torch.no_grad()
def dequantize(packed, sf, scale2):
    grid = torch.tensor(E2M1, device=packed.device)
    low, high = packed & 0xF, packed >> 4
    code = torch.stack((low, high), -1).flatten(-2).long()
    value = grid[code & 7] * torch.where((code & 8) > 0, -1.0, 1.0)
    step = (sf.float() * scale2[..., None, None]).repeat_interleave(BLOCK, -1)
    return value * step


def swizzle(scale: torch.Tensor) -> torch.Tensor:
    """sglang.srt.layers.quantization.utils.swizzle_blockscale, copied so this module imports without SGLang."""
    if scale.dtype != torch.float8_e4m3fn or scale.ndim != 3:
        raise ValueError("expects [experts, rows, k/16] FP8 block scales")
    experts, rows, cols = scale.shape
    rows_p, cols_p = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    padded = torch.zeros((experts, rows_p, cols_p), dtype=scale.dtype, device=scale.device)
    padded[:, :rows, :cols] = scale
    tiles = padded.reshape(experts, rows_p // 128, 4, 32, cols_p // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return tiles.contiguous().reshape(experts, rows_p, cols_p)


class Nvfp4Experts(nn.Module):
    """Drop-in for model.Experts: forward(x, indices, weights) with renormalized top-k weights."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor, a2_amax: float):
        """gate_up [E, 2I, H] in the drafter's [gate; up] order, down [E, H, I]; both float, any device."""
        super().__init__()
        experts, two_i, hidden = gate_up.shape
        inter = two_i // 2
        if down.shape != (experts, hidden, inter):
            raise ValueError(f"down {tuple(down.shape)} does not match gate_up {tuple(gate_up.shape)}")
        if not a2_amax > 0:
            raise ValueError("GEMM2 input amax must be positive (calibration)")
        gate, up = gate_up[:, :inter], gate_up[:, inter:]
        w13, w13_sf, w13_s2 = quantize(torch.cat((up, gate), 1))  # the CUTLASS kernel reads [up; gate]
        w2, w2_sf, w2_s2 = quantize(down)
        self.num_experts, self.hidden, self.inter = experts, hidden, inter
        self.register_buffer("w13", w13)
        self.register_buffer("w13_sf", swizzle(w13_sf))
        self.register_buffer("w13_scale2", w13_s2)
        self.register_buffer("w2", w2)
        self.register_buffer("w2_sf", swizzle(w2_sf))
        a2_scale = torch.tensor(a2_amax / (FP8_MAX * FP4_MAX), dtype=torch.float32, device=w2.device)
        self.register_buffer("a2_gscale", 1.0 / a2_scale)  # 0-dim: the kernel takes a scalar or one per expert
        self.register_buffer("g2_alphas", (a2_scale * w2_s2).float())
        # the dequantized reference, for tests only (not kept)
        self._raw = None

    def forward(self, x, indices, weights):
        from flashinfer.fused_moe import cutlass_fused_moe

        x = x.to(torch.bfloat16).contiguous()
        a1 = x.abs().amax().float().clamp_min(1e-6) / (FP8_MAX * FP4_MAX)
        quant_scales = [
            1.0 / a1,
            self.w13_sf.view(torch.int32),
            (a1 * self.w13_scale2).float(),
            self.a2_gscale,
            self.w2_sf.view(torch.int32),
            self.g2_alphas,
        ]
        out = torch.empty_like(x)
        tokens = x.shape[0]
        return cutlass_fused_moe(
            output=out,
            input=x,
            token_selected_experts=indices.to(torch.int32).contiguous(),
            token_final_scales=weights.to(torch.float32).contiguous(),
            fc1_expert_weights=self.w13.view(torch.long),
            fc2_expert_weights=self.w2.view(torch.long),
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=quant_scales,
            tune_max_num_tokens=max(16, 1 << (tokens - 1).bit_length()),
        )[0]
