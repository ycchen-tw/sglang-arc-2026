"""Routed experts for the drafter on SGLang's NVFP4 Marlin MoE (W4A16), expert-parallel over the attention-TP group.

For GPUs without FP4 tensor cores (sm_89): activations stay BF16. Weights are the same two-level RTN NVFP4 as the
CUTLASS path (nvfp4.quantize), repacked by SGLang's prepare_moe_nvfp4_layer_for_marlin exactly as the target's NVFP4
experts on that GPU. Each rank of the group holds a contiguous slice of the kept experts; router ids outside the slice
map to -1 (SGLang's EP convention) and the routed sum is all-reduced over the group.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .nvfp4 import quantize


class ExpertRows:
    """A lazily read [experts, ...] stack: rows `ids` of a safetensors slice, materialized `a:b` at a time, so a
    layer's BF16 experts never sit whole in host memory (four ranks next to 48 GiB of pinned PLE tables)."""

    def __init__(self, sl, ids, shape):
        self.sl, self.ids, self.shape = sl, list(ids), (len(ids), *shape)

    def __getitem__(self, span):
        return torch.stack([self.sl[int(e)] for e in self.ids[span]])


def release_host_heap():
    """Hand freed glibc arenas back to the OS: the host admission that runs right after counts them as used."""
    import ctypes

    ctypes.CDLL("libc.so.6").malloc_trim(0)


class MarlinEPExperts(nn.Module):
    """Drop-in for model.Experts: forward(x, indices, weights) over this rank's slice of the kept experts.

    gate_up [E_local, 2I, H] in the checkpoint's [gate; up] order (SGLang's w13 = [w1; w3]: SiLU on the first
    half), down [E_local, H, I]. Quantized with nvfp4.quantize (ModelOpt two-level NVFP4) and repacked by
    SGLang's prepare_moe_nvfp4_layer_for_marlin, exactly as the target's NVFP4 experts on L4. `first` is the index
    of this rank's first expert among the `total` kept ones; router ids outside the slice map to -1 (SGLang's EP
    convention), and the routed sum is all-reduced over `group` (None: single rank, the unit test)."""

    CHUNK = 8  # experts quantized and repacked at a time: the target leaves ~4.6 GB per L4 card at this point

    def __init__(self, gate_up, down, first: int, total: int, group, device=None):
        """gate_up / down may live on the CPU (an mmap slice); chunks go to `device` (default: their own)."""
        super().__init__()
        from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

        device = gate_up.device if device is None else torch.device(device)
        experts, two_i, hidden = gate_up.shape
        inter = two_i // 2
        if down.shape != (experts, hidden, inter):
            raise ValueError(f"down {tuple(down.shape)} does not match gate_up {tuple(gate_up.shape)}")
        if not 0 <= first and first + experts <= total:
            raise ValueError(f"expert slice [{first}, {first + experts}) outside {total}")
        param = lambda t: nn.Parameter(t, requires_grad=False)  # noqa: E731
        names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale", "w13_weight_scale_2",
                 "w2_weight_scale_2")
        out, workspace = None, None
        for a in range(0, experts, self.CHUNK):
            b = min(a + self.CHUNK, experts)
            w13, w13_sf, w13_s2 = quantize(gate_up[a:b].to(device))
            w2, w2_sf, w2_s2 = quantize(down[a:b].to(device))
            layer = SimpleNamespace(
                quant_config=SimpleNamespace(group_size=16), moe_runner_config=SimpleNamespace(is_gated=True),
                intermediate_size_per_partition=inter, params_dtype=torch.bfloat16,
                w13_weight=param(w13), w2_weight=param(w2), w13_weight_scale=param(w13_sf),
                w2_weight_scale=param(w2_sf), w13_weight_scale_2=param(w13_s2), w2_weight_scale_2=param(w2_s2),
            )
            prepare_moe_nvfp4_layer_for_marlin(layer)  # per-expert repack: chunks stack along the expert axis
            if out is None:
                out = {n: torch.empty((experts, *getattr(layer, n).shape[1:]), dtype=getattr(layer, n).dtype,
                                      device=device) for n in names}
                workspace = layer.workspace
            for n in names:
                out[n][a:b] = getattr(layer, n).data
            del layer, w13, w13_sf, w2, w2_sf
        for n in names:
            self.register_buffer(n, out[n])
        self.register_buffer("workspace", workspace)
        local = torch.full((total,), -1, dtype=torch.int32, device=device)
        local[first : first + experts] = torch.arange(experts, dtype=torch.int32, device=device)
        self.register_buffer("local_map", local)
        self.num_local, self.total, self.hidden, self.inter = experts, total, hidden, inter
        self.group = group
        self.router_mass = None
        self.full_gate = None

    @torch.compiler.disable
    def forward(self, x, indices, weights):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

        x = x.to(torch.bfloat16).contiguous()
        ids = self.local_map[indices]
        weights = weights.float().contiguous()
        if self.router_mass is not None:
            # ARC_DFLASH_ROUTER_STATS: summed (renormalized) top-k weight of the drafter's whole router -- every
            # exported expert, not only the kept ones -- on this layer's live input (g4/make_keep.py's criterion)
            p = F.linear(x, self.full_gate).float().softmax(-1)
            w, i = p.topk(indices.shape[-1], -1)
            self.router_mass.scatter_add_(0, i.flatten(), (w / w.sum(-1, keepdim=True)).flatten())
        y = fused_marlin_moe(
            x, self.w13_weight, self.w2_weight, self.w13_weight_scale, self.w2_weight_scale,
            gating_output=weights, topk_weights=weights, topk_ids=ids,
            global_num_experts=self.total, expert_map=self.local_map,
            w1_global_scale=self.w13_weight_scale_2, w2_global_scale=self.w2_weight_scale_2,
            workspace=self.workspace, num_bits=4,
        )
        if self.group is not None:
            y = self.group.all_reduce(y)
        return y
