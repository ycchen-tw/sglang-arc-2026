"""Drafter linears stored in block FP8: e4m3 weight with one fp32 dequant scale per 128x128 block.

The layout and GEMM are the target's online block FP8 (arc_online_fp8, patch 0032): sglang's Fp8LinearMethod for
block quant with dynamic per-token-group activation quant. No new kernel. The block GEMM sglang dispatches to is
pinned per device generation, the two measured paths: CUTLASS on sm_120 (G4, as 0032), Triton on sm_89 (L4, as the
target's ARC_HC_ONLINE_FP8 there). Anything else raises.
"""

import torch
from torch import nn
from torch.nn import Parameter

BLOCK = 128


def _method(device):
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[BLOCK, BLOCK])
    )
    if not method.block_quant or method.use_marlin or method.use_mxfp8:
        raise RuntimeError(f"Fp8LinearMethod is not plain block FP8 here (block_quant={method.block_quant}, "
                           f"marlin={method.use_marlin}, mxfp8={method.use_mxfp8})")
    capability = torch.cuda.get_device_capability(device)
    want = {(12, 0): fp8_utils.cutlass_w8a8_block_fp8_linear_with_fallback,
            (8, 9): fp8_utils.triton_w8a8_block_fp8_linear}.get(capability)
    if want is None or method.w8a8_block_fp8_linear is not want:
        raise RuntimeError(f"block-FP8 dispatch on sm_{capability[0]}{capability[1]} is "
                           f"{method.w8a8_block_fp8_linear.__name__}; measured paths: CUTLASS on sm_120, Triton on sm_89")
    return method


class Fp8BlockLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale_inv: torch.Tensor):
        super().__init__()
        n, k = weight.shape
        if weight.dtype != torch.float8_e4m3fn or n % BLOCK or k % BLOCK:
            raise ValueError(f"block FP8 weight must be e4m3 [N, K] with N, K divisible by {BLOCK}: "
                             f"{weight.dtype} {tuple(weight.shape)}")
        if scale_inv.dtype != torch.float32 or scale_inv.shape != (n // BLOCK, k // BLOCK):
            raise ValueError(f"weight_scale_inv must be fp32 [{n // BLOCK}, {k // BLOCK}]: "
                             f"{scale_inv.dtype} {tuple(scale_inv.shape)}")
        if not torch.isfinite(scale_inv).all() or (scale_inv <= 0).any():
            raise ValueError("weight_scale_inv must be finite and positive")
        self.in_features, self.out_features = k, n
        self.weight = Parameter(weight.contiguous(), requires_grad=False)
        self.weight_scale_inv = Parameter(scale_inv.contiguous(), requires_grad=False)
        self.register_parameter("input_scale", None)
        self.orig_dtype = torch.bfloat16
        self.method = _method(weight.device)
        self.method.process_weights_after_loading(self)

    def forward(self, x):
        shape = x.shape
        y = self.method.apply(self, x.reshape(-1, shape[-1]), bias=None)
        return y.reshape(*shape[:-1], self.out_features)


class AllReduced(nn.Module):
    """A row-parallel projection already narrowed to this rank's input slice: its partial output, all-reduced over
    the attention-TP group (the dense drafter's o_proj; the MoE path's fused_serving.ReducedLinear does the same
    for a BF16 weight it slices itself)."""

    def __init__(self, linear: nn.Module, group):
        super().__init__()
        self.linear, self.group = linear, group

    def forward(self, x):
        return self.group.all_reduce(self.linear(x))
