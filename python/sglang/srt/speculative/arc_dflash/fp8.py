"""Drafter linears stored in block FP8: e4m3 weight with one fp32 dequant scale per 128x128 block.

The layout and GEMM are the target's (arc_online_fp8, patch 0032): sglang's Fp8LinearMethod for block quant, its
process_weights_after_loading and CUTLASS block GEMM with dynamic per-token-group activation quant. No new kernel.
"""

import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.quantization import arc_online_fp8


class Fp8BlockLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale_inv: torch.Tensor):
        super().__init__()
        n, k = weight.shape
        block = arc_online_fp8.BLOCK
        if weight.dtype != torch.float8_e4m3fn or n % block or k % block:
            raise ValueError(f"block FP8 weight must be e4m3 [N, K] with N, K divisible by {block}: "
                             f"{weight.dtype} {tuple(weight.shape)}")
        if scale_inv.dtype != torch.float32 or scale_inv.shape != (n // block, k // block):
            raise ValueError(f"weight_scale_inv must be fp32 [{n // block}, {k // block}]: "
                             f"{scale_inv.dtype} {tuple(scale_inv.shape)}")
        if not torch.isfinite(scale_inv).all() or (scale_inv <= 0).any():
            raise ValueError("weight_scale_inv must be finite and positive")
        self.in_features, self.out_features = k, n
        self.weight = Parameter(weight, requires_grad=False)
        self.weight_scale_inv = Parameter(scale_inv, requires_grad=False)
        self.register_parameter("input_scale", None)
        self.orig_dtype = torch.bfloat16
        self.method = arc_online_fp8._method()
        self.method.process_weights_after_loading(self)

    def forward(self, x):
        shape = x.shape
        y = self.method.apply(self, x.reshape(-1, shape[-1]), bias=None)
        return y.reshape(*shape[:-1], self.out_features)
