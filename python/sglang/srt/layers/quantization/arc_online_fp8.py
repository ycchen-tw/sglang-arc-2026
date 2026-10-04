"""arc-forge 0032: online block-FP8 for chosen BF16 linears of Qwen4-Exp (opt-in, ``ARC_ONLINE_FP8``).

At load time the named BF16 weights are quantized to e4m3 with one fp32 scale per 128x128 block (scale = block
amax / 448, the DeepSeek ``weight_scale_inv`` layout), and the layers get sglang's ``Fp8LinearMethod`` for block
quant -- the same method, ``process_weights_after_loading`` and CUTLASS block GEMM a block-FP8 checkpoint uses on
sm_120 (``cutlass_w8a8_block_fp8_linear_with_fallback``, dynamic per-token-group activation quant). No new kernel.

``ARC_ONLINE_FP8`` is a comma-separated subset of:
  gdn_in   Qwen3.5 GatedDeltaNet ``in_proj_qkvz`` (``in_proj_ba``, 2 x num_v_heads rows, stays BF16; the stacked
           BF16 qkvz+ba GEMM is dropped for the two separate ones)
  gdn_out  GatedDeltaNet ``out_proj``
  qsa_qkv  attention ``qkv_proj`` (on the decoder layer; ``self_attn.qkv_proj`` in the checkpoint)
  qsa_o    attention ``o_proj``
  lm_head  the target's ``lm_head``; the native Qwen MTP draft binds to this very module (eagle_worker_v2
           asserts ``draft_head is target_head``), so the draft's logits use it too
or ``all`` (all five). Unset or empty: nothing is touched. Everything else (shared expert, index projection, router,
hyper-connection mixers, the draft's own layers) stays as it is: FP8 is slower than BF16 on those shapes here.

Refused, loudly: an unknown name; tensor parallelism; overlapped startup weight load; a block-FP8 dispatch other
than CUTLASS; a layer that is not an unquantized BF16 linear with both dims divisible by 128; a LoRA-wrapped layer;
an lm_head tied to the input embedding; a requested name that matches no layer.
"""

from __future__ import annotations

import logging
import os
import re

import torch
from torch.nn import Parameter

logger = logging.getLogger(__name__)

ENV = "ARC_ONLINE_FP8"
TARGETS = ("gdn_in", "gdn_out", "qsa_qkv", "qsa_o", "lm_head")
BLOCK = 128
FP8_MAX = 448.0
_SUFFIX = {
    "gdn_in": re.compile(r"(^|\.)layers\.\d+\.linear_attn\.in_proj_qkvz$"),
    "gdn_out": re.compile(r"(^|\.)layers\.\d+\.linear_attn\.out_proj$"),
    # Qwen3.5's attention decoder layer holds qkv_proj / o_proj itself (checkpoint names say self_attn.)
    "qsa_qkv": re.compile(r"(^|\.)layers\.\d+\.(self_attn\.)?qkv_proj$"),
    "qsa_o": re.compile(r"(^|\.)layers\.\d+\.(self_attn\.)?o_proj$"),
}


def requested() -> tuple[str, ...]:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return ()
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if names == ["all"]:
        return TARGETS
    bad = [n for n in names if n not in TARGETS]
    if bad:
        raise ValueError(f"{ENV}={raw!r}: unknown {bad}; choose from {list(TARGETS)} or 'all'")
    return tuple(n for n in TARGETS if n in names)


@torch.no_grad()
def quantize_block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16 -> ([N, K] e4m3, [N / 128, K / 128] fp32 dequant scales); rows in chunks (lm_head is 248k)."""
    n, k = weight.shape
    if n % BLOCK or k % BLOCK:
        raise ValueError(f"block FP8 needs both dims divisible by {BLOCK}, got {tuple(weight.shape)}")
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n // BLOCK, k // BLOCK), dtype=torch.float32, device=weight.device)
    step = 16384
    for r in range(0, n, step):
        w = weight[r:r + step].to(torch.float32)
        rows = w.shape[0]
        blocks = w.view(rows // BLOCK, BLOCK, k // BLOCK, BLOCK)
        amax = blocks.abs().amax(dim=(1, 3))
        s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        q[r:r + rows] = (blocks / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).view(rows, k).to(torch.float8_e4m3fn)
        scale[r // BLOCK:(r + rows) // BLOCK] = s
    return q, scale


def _method():
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import cutlass_w8a8_block_fp8_linear_with_fallback

    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[BLOCK, BLOCK])
    )
    if not method.block_quant or method.use_marlin or method.use_mxfp8:
        raise RuntimeError(f"{ENV}: Fp8LinearMethod is not plain block FP8 here "
                           f"(block_quant={method.block_quant}, marlin={method.use_marlin}, mxfp8={method.use_mxfp8})")
    if method.w8a8_block_fp8_linear is not cutlass_w8a8_block_fp8_linear_with_fallback:
        raise RuntimeError(f"{ENV}: the block-FP8 dispatch is {method.w8a8_block_fp8_linear.__name__}, not the CUTLASS "
                           "path this was measured on (sm_120; pass --fp8-gemm-backend cutlass or leave auto there)")
    return method


def _convert(layer: torch.nn.Module, name: str) -> int:
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod, UnquantizedLinearMethod

    if hasattr(layer, "set_lora"):
        raise RuntimeError(f"{ENV}: {name} is LoRA-wrapped")
    qm = getattr(layer, "quant_method", None)
    if not isinstance(qm, (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)):
        raise RuntimeError(f"{ENV}: {name} has quant method {type(qm).__name__}, not an unquantized BF16 linear")
    weight = layer.weight
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or getattr(layer, "bias", None) is not None:
        raise RuntimeError(f"{ENV}: {name} weight is {weight.dtype} {tuple(weight.shape)} "
                           f"bias={getattr(layer, 'bias', None) is not None}; need BF16 [N, K], no bias")
    q, scale = quantize_block_fp8(weight.data)
    before = weight.numel() * weight.element_size()
    method = _method()
    del layer.weight
    layer.register_parameter("weight", Parameter(q, requires_grad=False))
    layer.register_parameter("weight_scale_inv", Parameter(scale, requires_grad=False))
    layer.register_parameter("input_scale", None)
    layer.orig_dtype = torch.bfloat16
    method.process_weights_after_loading(layer)
    layer.quant_method = method
    return before - (q.numel() + scale.numel() * scale.element_size())


def apply(model: torch.nn.Module, *, is_draft_worker: bool, tp_size: int, startup_overlap: bool) -> None:
    """Quantize the requested layers of a loaded target model in place (no-op when ARC_ONLINE_FP8 is unset)."""
    names = requested()
    if not names:
        return
    if is_draft_worker:
        # the draft's own layers are not targets; its lm_head is the target's module (bound after both load)
        return
    if tp_size != 1:
        raise RuntimeError(f"{ENV} supports --tp-size 1 only, got {tp_size}")
    if startup_overlap:
        raise RuntimeError(f"{ENV} needs the weights loaded inside load_model (no overlapped startup weight load)")
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    counts = {n: 0 for n in names}
    saved = 0
    for mod_name, module in model.named_modules():
        if "visual" in mod_name:
            continue
        for target in names:
            if target == "lm_head":
                continue
            if _SUFFIX[target].search(mod_name):
                saved += _convert(module, mod_name)
                counts[target] += 1
        if "gdn_in" in names and isinstance(module, Qwen3_5GatedDeltaNet) and module._fused_in_proj_weight is not None:
            # the stacked BF16 [qkvz; ba] GEMM is gone with qkvz in FP8: ba keeps its own BF16 rows
            module.in_proj_ba.weight.data = module.in_proj_ba.weight.data.clone()
            module._fused_in_proj_weight = None
            module._fused_in_proj_qkvz_width = 0
    if "lm_head" in names:
        head = getattr(model, "lm_head", None)
        if head is None:
            raise RuntimeError(f"{ENV}: the model has no lm_head")
        embed = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embed is not None and (head is embed or head.weight.data_ptr() == embed.weight.data_ptr()):
            raise RuntimeError(f"{ENV}: lm_head is tied to embed_tokens; quantizing it would change the embedding")
        saved += _convert(head, "lm_head")
        counts["lm_head"] = 1
    missing = [n for n, c in counts.items() if c == 0]
    if missing:
        raise RuntimeError(f"{ENV}: no layer matched {missing}")
    torch.cuda.empty_cache()
    logger.info("%s: block-FP8 %s; weights %.2f GiB smaller", ENV,
                ", ".join(f"{n} x{c}" for n, c in counts.items()), saved / 2**30)
