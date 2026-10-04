"""Serving-only transforms for the drafter. The training path never imports this.

`model.py` is shared with the trainer, so everything here rebinds behaviour from
the outside instead of editing it. Drafter numerics only move the acceptance
rate -- rejection sampling uses the actual proposal q, so emitted tokens stay
exactly target distributed whichever kernels run.

NOT WIRED IN -- the trade is bad, not the code. Measured 2026-09-14 on the trained
drafter: the fused kernel alone is 2.5x the hand-written sorted grouped GEMM (0.256 ->
0.104 ms at eight tokens), but the whole compiled forward only goes 1.355 -> 1.270 ms,
and the drafter hidden state moves by 8% relative L2. Eight percent of the signal that
sets the acceptance rate is not worth 0.085 ms. Two things here are worth keeping.
First, the `torch.library.custom_op` wrapper: calling flashinfer directly made the
forward *slower* (1.32 -> 1.88 ms) because its Python entry point breaks dynamo's graph
at every layer and the surrounding GR/norm/rotary work stops fusing. Second, the weight
layout: the kernel wants the gated halves as [up; gate], the opposite of how the
checkpoint stores them, which costs a factor of ten in relative error if you get it
backwards.
"""

from dataclasses import replace

import torch
from .model import Experts


def _trtllm_layout(w13, w2):
    """The one-time load-time shuffle the trtllm-gen kernel demands.

    Mirrors SGLang's `UnquantizedFusedMoEMethod.process_weights_after_loading`
    (`layers/quantization/unquant.py`). The kernel wants the gated halves in
    [up; gate] order, which is the opposite of how the checkpoint stores them.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    cache, epilogue_tile_m, block_k = {}, 128, 128
    gate, up = w13.chunk(2, 1)
    w13 = torch.cat((up, gate), 1).contiguous()
    first, second = [], []
    for expert in range(w13.shape[0]):
        rows = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13[expert].view(torch.uint8), epilogue_tile_m, is_gated_act_gemm=True
        )
        a = w13[expert].view(torch.uint8)[rows.to(w13.device)].contiguous()
        rows = get_w2_permute_indices_with_cache(cache, w2[expert].view(torch.uint8), epilogue_tile_m)
        b = w2[expert].view(torch.uint8)[rows.to(w2.device)].contiguous()
        first.append(convert_to_block_layout(a, block_k).view(torch.bfloat16))
        second.append(convert_to_block_layout(b, block_k).view(torch.bfloat16))
    return torch.stack(first).contiguous(), torch.stack(second).contiguous()


@torch.library.custom_op("mtp3::trtllm_bf16_moe", mutates_args=())
def trtllm_bf16_moe(
    hidden: torch.Tensor,
    routing_logits: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method: int,
    activation: int,
) -> torch.Tensor:
    """flashinfer's fused MoE behind a registered op.

    Called directly, flashinfer's Python entry point is opaque to dynamo, which
    breaks the graph at every layer; the surrounding GR/norm/rotary work then
    stops fusing and the whole forward gets slower than the unfused baseline
    even though the MoE kernel itself is 2.5x faster. Registering it keeps one
    graph and lets inductor schedule around an opaque call.
    """
    from flashinfer.fused_moe import trtllm_bf16_moe as kernel

    out = kernel(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routing_method_type=routing_method,
        activation_type=activation,
        norm_topk_prob=True,
    )
    # The single-tensor return is deprecated upstream in favour of a list.
    out = out[0] if isinstance(out, (list, tuple)) else out
    if out.shape != hidden.shape:
        raise RuntimeError(f"fused MoE returned {tuple(out.shape)} for input {tuple(hidden.shape)}")
    return out


@trtllm_bf16_moe.register_fake
def _(hidden, routing_logits, gemm1_weights, gemm2_weights, *rest):
    return torch.empty_like(hidden)


def _block_quant_fp8(weights):
    """Per-128x128-block FP8, the scheme the checkpoint and the kernel both use.

    Returns plain scales, not the power-of-two UE8M0 form: that rounding is for
    DeepGEMM's runner, and flashinfer's trtllm path takes the scales as given.
    """
    experts, rows, columns = weights.shape
    if rows % 128 or columns % 128:
        raise ValueError(f"block quantization needs 128-aligned weights, got {(rows, columns)}")
    blocks = weights.float().view(experts, rows // 128, 128, columns // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(1e-4) / 448.0
    quantized = (blocks / scale).to(torch.float8_e4m3fn).view(experts, rows, columns)
    return quantized.contiguous(), scale.view(experts, rows // 128, columns // 128).contiguous()


@torch.library.custom_op("mtp3::trtllm_fp8_moe", mutates_args=())
def trtllm_fp8_moe(
    hidden: torch.Tensor,
    routing_logits: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method: int,
    activation: int,
) -> torch.Tensor:
    """The block-scaled FP8 sibling of `trtllm_bf16_moe`, same graph treatment."""
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe as kernel
    from flashinfer.tllm_enums import Fp8QuantizationType
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    quantized, scale = per_token_group_quant_fp8(hidden, 128, column_major_scales=True)
    out = kernel(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=quantized,
        hidden_states_scale=scale.t(),
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=1.0,
        routing_method_type=routing_method,
        use_shuffled_weight=False,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=activation,
        norm_topk_prob=True,
    )
    out = out[0] if isinstance(out, (list, tuple)) else out
    if out.shape != hidden.shape:
        raise RuntimeError(f"fused MoE returned {tuple(out.shape)} for input {tuple(hidden.shape)}")
    return out


@trtllm_fp8_moe.register_fake
def _(hidden, routing_logits, *rest):
    return torch.empty_like(hidden)


class FusedRoutedExperts(torch.nn.Module):
    """flashinfer's trtllm-gen BF16 MoE, routing included.

    The kernel does softmax -> top-k -> renormalize itself (RenormalizeNaive),
    which is exactly what `MoE.forward` was doing by hand, so the gate logits go
    in raw and the top-k never materializes on our side.
    """

    def __init__(self, config, experts, quantization="bf16"):
        super().__init__()
        if quantization not in ("bf16", "fp8"):
            raise ValueError(f"unknown fused MoE quantization {quantization!r}")
        if not config.norm_topk_prob:
            raise ValueError("the trtllm routed kernel always renormalizes top-k weights")
        if config.moe_intermediate_size % 128:
            raise ValueError("trtllm block layout needs an intermediate size that is a multiple of 128")
        from flashinfer.fused_moe import RoutingMethodType
        from flashinfer.fused_moe.core import ActivationType

        self.config = config
        self.quantization = quantization
        self.routing = int(RoutingMethodType.RenormalizeNaive)
        self.activation = int(ActivationType.Swiglu)
        if quantization == "bf16":
            gemm1, gemm2 = _trtllm_layout(experts.gate_up_proj.data, experts.down_proj.data)
            self.register_buffer("gemm1_weights", gemm1, persistent=False)
            self.register_buffer("gemm2_weights", gemm2, persistent=False)
        else:
            # The FP8 path takes the weights unshuffled, but still wants the
            # gated halves the other way round.
            gate, up = experts.gate_up_proj.data.chunk(2, 1)
            gemm1, scale1 = _block_quant_fp8(torch.cat((up, gate), 1).contiguous())
            gemm2, scale2 = _block_quant_fp8(experts.down_proj.data)
            self.register_buffer("gemm1_weights", gemm1, persistent=False)
            self.register_buffer("gemm1_scale", scale1, persistent=False)
            self.register_buffer("gemm2_weights", gemm2, persistent=False)
            self.register_buffer("gemm2_scale", scale2, persistent=False)

    def forward(self, hidden, routing_logits):
        c = self.config
        if self.quantization == "fp8":
            return torch.ops.mtp3.trtllm_fp8_moe(
                hidden,
                routing_logits,
                self.gemm1_weights,
                self.gemm1_scale,
                self.gemm2_weights,
                self.gemm2_scale,
                c.num_experts,
                c.num_experts_per_tok,
                c.moe_intermediate_size,
                0,
                c.num_experts,
                self.routing,
                self.activation,
            )
        return torch.ops.mtp3.trtllm_bf16_moe(
            hidden,
            routing_logits,
            self.gemm1_weights,
            self.gemm2_weights,
            c.num_experts,
            c.num_experts_per_tok,
            c.moe_intermediate_size,
            0,
            c.num_experts,
            self.routing,
            self.activation,
        )


def enable_fused_moe(model, quantization="bf16"):
    """Swap every routed-expert block for the fused kernel and drop the originals."""
    for layer in model.layers:
        moe = layer.mlp
        fused = FusedRoutedExperts(model.config, moe.experts, quantization)
        # The unshuffled stacks are 2.5 GiB per layer and nothing reads them
        # again; keeping both copies would double the drafter's residency.
        del moe.experts
        moe.add_module("experts", fused)

        def forward(x, moe=moe, fused=fused):
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            y = fused(x, moe.gate(x).float())
            y = y + moe.shared_expert(x) * torch.sigmoid(moe.shared_expert_gate(x))
            return y.reshape(shape)

        moe.forward = forward
    torch.cuda.empty_cache()
    return model


class LocalExperts(torch.nn.Module):
    """This rank's slice of the routed experts, with the combine all-reduced.

    Routing stays global: the gate still scores all of them, so a token's top-k
    can land on either rank. Each rank computes only the picks it owns and
    produces a partial sum, and the all-reduce makes them whole. Non-local picks
    are zero-weighted and pointed at local expert 0, which keeps the sorted
    grouped GEMM's shape independent of how the routing happened to fall.

    Pointing them at local expert 0 concentrates half the batch's rows in one
    group, which looks like the wrong thing and is not: spreading them round
    robin over the local stack (`indices % count`) populates every group instead
    of a handful, and measured 8 x 5.5 s of first-call cost per batch size
    against 8 x 1.0 s for the concentrated form, plus ~0.2 ms on the steady-state
    draft. Concentrated stays.

    `Experts.forward` is called unbound rather than copied, so the sharded path
    and the replicated one cannot drift.
    """

    def __init__(self, experts, offset, count, group):
        super().__init__()
        if offset < 0 or count < 1 or offset + count > experts.gate_up_proj.shape[0]:
            raise ValueError("expert slice falls outside the stack")
        self.offset, self.count, self.group = offset, count, group
        self.fp32_output = experts.fp32_output
        self.stable_gemm = experts.stable_gemm
        self.gate_up_proj = torch.nn.Parameter(
            experts.gate_up_proj.data[offset : offset + count].clone(), requires_grad=False
        )
        self.down_proj = torch.nn.Parameter(
            experts.down_proj.data[offset : offset + count].clone(), requires_grad=False
        )

    def forward(self, x, indices, weights):
        mine = (indices >= self.offset) & (indices < self.offset + self.count)
        local = torch.where(mine, indices - self.offset, torch.zeros_like(indices))
        share = torch.where(mine, weights, torch.zeros_like(weights))
        partial = Experts.forward(self, x, local, share)
        return partial if self.group is None else self.group.all_reduce(partial)


def shard_experts_for_ep(model, rank, size, group):
    """Give each rank an equal slice of every routed-expert stack.

    The trained checkpoint is already split this way on disk -- FSDP2 shards
    every parameter on dim 0, and for the expert stacks dim 0 is the expert
    axis -- but the serving loader concatenates it back, so the split is redone
    here rather than threaded through the loader's coverage checks.
    """
    experts = model.config.num_experts
    if experts % size:
        raise ValueError(f"{experts} experts do not divide across {size} ranks")
    count = experts // size
    for layer in model.layers:
        layer.mlp.experts = LocalExperts(layer.mlp.experts, rank * count, count, group)
    torch.cuda.empty_cache()
    return model


class ReducedLinear(torch.nn.Module):
    """A row-parallel projection: each rank owns a slice of the input axis.

    Attention is sharded by head, so `o_proj` sees only this rank's heads and
    produces a partial sum of the whole output; the all-reduce makes it whole.
    It has to happen here rather than later in the residual, because the
    attention output is also a convolution boundary and a cache entry.
    """

    def __init__(self, linear, columns, group):
        super().__init__()
        self.group = group
        self.weight = torch.nn.Parameter(linear.weight.data[:, columns].clone(), requires_grad=False)

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight)
        return y if self.group is None else self.group.all_reduce(y)


def _keep_rows(linear, start, count):
    """Narrow a column-parallel projection to the rows this rank computes."""
    linear.weight = torch.nn.Parameter(linear.weight.data[start : start + count].clone(), requires_grad=False)
    linear.out_features = count


def shard_attention_for_tp(model, rank, size, group):
    """Give each rank an equal slice of every attention head.

    Query heads and key/value heads split the same way, so GQA still pairs each
    rank's queries with its own keys and the per-rank clean cache holds only the
    key/value heads that rank computes. `head_dim` and both head norms are
    per-head and stay whole.

    A measurement, not a recommendation: attention is 1.9% of the drafter's
    parameters, so halving it saves on the order of 25 us of weight reads per
    cycle while the three all-reduces it adds cost on the order of 200. It is
    here because which way that trade falls depends on the deployment -- batch
    size, sequence length, how much of the cycle is bandwidth-bound -- and not
    on the case it was first measured against.
    """
    c = model.config
    if c.num_attention_heads % size or c.num_key_value_heads % size:
        raise ValueError(f"{c.num_attention_heads}/{c.num_key_value_heads} heads do not divide across {size} ranks")
    heads, kv, dim = c.num_attention_heads // size, c.num_key_value_heads // size, c.head_dim
    for layer in model.layers:
        attention = layer.self_attn
        attention.c = replace(c, num_attention_heads=heads, num_key_value_heads=kv)
        # q_proj emits [head, 2 * head_dim] per head -- the value and its gate --
        # so one head's rows are contiguous and the slice is a single range.
        _keep_rows(attention.q_proj, rank * heads * 2 * dim, heads * 2 * dim)
        _keep_rows(attention.k_proj, rank * kv * dim, kv * dim)
        _keep_rows(attention.v_proj, rank * kv * dim, kv * dim)
        attention.o_proj = ReducedLinear(
            attention.o_proj, slice(rank * heads * dim, (rank + 1) * heads * dim), group
        )
    torch.cuda.empty_cache()
    return model
