"""Drafter linears stored in block FP8: e4m3 weight with one fp32 dequant scale per 128x128 block.

The layout and GEMM are the target's online block FP8 (arc_online_fp8, patch 0032): sglang's Fp8LinearMethod for
block quant with dynamic per-token-group activation quant. No new kernel. The block GEMM sglang dispatches to is
pinned per device generation, the two measured paths: CUTLASS on sm_120 (G4, as 0032), Triton on sm_89 (L4, as the
target's ARC_HC_ONLINE_FP8 there). Anything else raises.
"""

import json

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


def dequant_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] with fp32 [N/128, K/128] block scales -> BF16 [N, K] (the values the block GEMM computes with)."""
    n, k = weight.shape
    if scale_inv.shape != (n // BLOCK, k // BLOCK) or n % BLOCK or k % BLOCK:
        raise ValueError(f"block FP8 {tuple(weight.shape)} with scales {tuple(scale_inv.shape)}")
    blocks = weight.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * scale_inv[:, None, :, None]
    return blocks.view(n, k).to(torch.bfloat16)


# ARC_DFLASH_W4A8=<format>:<group>[:<scope>] (opt-in): the drafter's block-FP8 linears served as Humming INT4 weights
# (round-to-nearest from the FP8 values, one scale per `group` inputs) with dynamic per-token FP8 e4m3 activations --
# the FP8 tensor cores of sm_89 at half the weight bytes. format int4 (symmetric) or uint4z (zero point); scope all
# (every FP8 linear) or mlp (the dense FFN only). Lossy: only the proposals change (rejection sampling keeps the output
# exact), so it is judged by accept rate and speed. Needs ARC_HUMMING_DIR (unpacked humming-kernels, as the
# hostcomm hook's Humming MoE) and Humming's CUDA root (v16_patch ARC_V16_HUMMING_CUDA_ROOT).
W4A8_FORMATS = {"int4": ("int4", False), "uint4z": ("uint4", True)}


def w4a8_spec():
    """None when ARC_DFLASH_W4A8 is unset, else (b_dtype, zero_point, group, scope)."""
    import os

    raw = os.environ.get("ARC_DFLASH_W4A8", "").strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) not in (2, 3) or parts[0] not in W4A8_FORMATS or parts[1] not in ("32", "64", "128") \
            or (len(parts) == 3 and parts[2] not in ("all", "mlp")):
        raise ValueError(f"ARC_DFLASH_W4A8={raw!r}: <int4|uint4z>:<32|64|128>[:<all|mlp>]")
    dtype, zero_point = W4A8_FORMATS[parts[0]]
    return dtype, zero_point, int(parts[1]), parts[2] if len(parts) == 3 else "all"


def _humming():
    import os
    import sys

    root = os.environ.get("ARC_HUMMING_DIR")
    if not root:
        raise RuntimeError("ARC_DFLASH_W4A8 needs ARC_HUMMING_DIR (the unpacked humming-kernels)")
    if root not in sys.path:
        sys.path.insert(0, root)
    from humming.layer import HummingLayer, HummingLayerMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    return HummingLayer, HummingLayerMethod, HummingWeightSchema, HummingInputSchema


class HummingW4A8Linear(nn.Module):
    """One drafter linear as Humming INT4 weights x per-token FP8 activations (ARC_DFLASH_W4A8).

    Always on Humming's batch-invariant tuning (no stream-K, a fixed K reduction): an attention-TP group's ranks run the
    replicated drafter ops each on their own and must propose the same tokens. Humming's default heuristic splits K
    across CTAs (stream-K) for K > 1024, and its fix-up order varies run to run -- the L4 pair diverged within a minute
    (ARC_SPEC_AGREE_DEVICE); on a 4090 the default gave 0/30 equal repeats for down_proj at M=48 and q_proj at M=384,
    the batch-invariant configs 30/30 for every shape, at about the same speed for M <= 384.
    """

    def __init__(self, weight: torch.Tensor, spec):
        super().__init__()
        dtype, zero_point, group, _ = spec
        if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.device.type != "cuda":
            raise ValueError(f"HummingW4A8Linear takes a BF16 [N, K] CUDA weight, got {weight.dtype} "
                             f"{tuple(weight.shape)} on {weight.device}")
        n, k = weight.shape
        if k % group:
            raise ValueError(f"W4A8: K {k} is not a multiple of the group {group}")
        HummingLayer, HummingLayerMethod, HummingWeightSchema, HummingInputSchema = _humming()
        hm = HummingLayer(shape_n=n, shape_k=k,
                          weight_config=HummingWeightSchema(b_dtype=dtype, weight_scale_group_size=group,
                                                            has_zero_point=zero_point),
                          input_config=HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=0),
                          torch_dtype=torch.bfloat16)
        hm.load_from_unquantized(weight)
        hm = hm.to(weight.device)
        hm.transform()
        self.hm = hm
        self.tuning = json.dumps(HummingLayerMethod.get_default_tuning_configs(hm, use_batch_invariant=True))
        self.in_features, self.out_features = k, n

    def forward(self, x):
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        if rows.shape[0] == 0:  # an attention-DP IDLE batch
            return x.new_empty(*shape[:-1], self.out_features)
        y = self.hm(rows.contiguous(), tuning_config=self.tuning)
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
