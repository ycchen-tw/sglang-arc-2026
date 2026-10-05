"""Qwen4 native GR/MoE MTP copies with causal SWA and detached clean caches.

Tensor names follow the official HF checkpoint. GR/norm/attention equations
follow transformers' Qwen4Exp and SGLang's native MTP fusion. No target KV.
Each batch item is one isolated sequence/block; positions are absolute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    hidden_size: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    norm_topk_prob: bool = True
    num_layers: int = 3
    window: int = 1024
    block_size: int = 8
    conv_group_size: int = 16
    selector_top_k: int = 16
    selector_rank: int = 256
    # "moe": routed experts + router (the trained export); "dense_shared": one SwiGLU of width
    # dense_intermediate_size in place of the routed experts and router, shared expert and its gate unchanged
    ffn_type: str = "moe"
    dense_intermediate_size: int = 0

    def __post_init__(self):
        if self.hidden_size % self.conv_group_size or self.window < 1:
            raise ValueError("invalid conv group/window")
        if self.num_layers < 1:
            raise ValueError("need at least one draft decoder layer")
        if self.block_size < 2:
            raise ValueError("a block is the anchor plus at least one proposal")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("invalid GQA geometry")
        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("invalid routed expert count")
        if self.ffn_type not in ("moe", "dense_shared"):
            raise ValueError(f"unknown ffn_type {self.ffn_type!r}")
        if (self.ffn_type == "dense_shared") != (self.dense_intermediate_size > 0):
            raise ValueError("dense_intermediate_size is set exactly when ffn_type is dense_shared")
        if not 1 <= self.selector_top_k <= self.vocab_size:
            raise ValueError("invalid selector top-K")
        rotary = int(self.head_dim * self.partial_rotary_factor)
        if rotary <= 0 or rotary % 2:
            raise ValueError("rotary dimension must be positive and even")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps, group_size=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps, self.group_size = eps, group_size
        self.stable_kernel = False

    def forward(self, x):
        if self.stable_kernel and x.is_cuda:
            from stable_kernels import rms_norm

            return rms_norm(x, self.weight, self.group_size or x.shape[-1], self.eps)
        z = x.float()
        if self.group_size is not None:
            z = z.unflatten(-1, (-1, self.group_size))
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            z = z.flatten(-2)
        return (z * (1 + self.weight.float())).to(x.dtype)


class GatedResidual(nn.Module):
    def __init__(self, c, combine=True):
        super().__init__()
        self.c = c
        self.stable_math = False
        width = c.hc_count * c.hidden_size
        self.hc_norm = RMSNorm(width, c.rms_norm_eps, c.hidden_size)
        self.input_mix_weight_down = nn.Linear(width, c.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(c.hc_lowrank, width, bias=False)
        self.block_inject_weight = nn.Linear(width, c.hc_count, bias=False) if combine else None

    def forward(self, h):
        if h.shape[-2:] != (self.c.hc_count, self.c.hidden_size):
            raise ValueError("GR requires explicit [batch, sequence, branches, hidden]")
        z = self.hc_norm(h.flatten(-2))
        if self.stable_math:
            with torch.autocast(h.device.type, enabled=False):
                down = F.linear(z.float(), self.input_mix_weight_down.weight.float()) / self.c.hc_count
                mix = torch.sigmoid(F.linear(F.silu(down), self.input_mix_weight_up.weight.float()))
                x = (mix * z.float()).unflatten(-1, (self.c.hc_count, self.c.hidden_size)).mean(-2).to(h.dtype)
                if self.block_inject_weight is None:
                    return x
                gate = (
                    2 * torch.sigmoid(F.linear(z.float(), self.block_inject_weight.weight.float()) / self.c.hc_count)
                ).to(h.dtype)
            return x, gate
        mix = torch.sigmoid(self.input_mix_weight_up(F.silu(self.input_mix_weight_down(z) / self.c.hc_count)))
        x = (mix * z).unflatten(-1, (self.c.hc_count, self.c.hidden_size)).mean(-2)
        if self.block_inject_weight is None:
            return x
        gate = 2 * torch.sigmoid(self.block_inject_weight(z) / self.c.hc_count)
        return x, gate


class GroupedConv(nn.Module):
    """Released DFlash2 K2 equations, extended with clean boundary state.

    Both pre/post coefficients come from the operator's pre-conv input. The post
    boundary is the previous raw operator output, not its convolved output.
    """

    def __init__(self, c):
        super().__init__()
        self.groups = c.hidden_size // c.conv_group_size
        base = torch.zeros(2, 2, c.hidden_size)
        base[:, 0] = 1
        self.base_kernel = nn.Parameter(base)
        self.kernel_projection = nn.Linear(c.hidden_size, 4 * self.groups, bias=False)
        nn.init.zeros_(self.kernel_projection.weight)

    def convolve(self, x, delta, side, boundary):
        previous = torch.cat((boundary, x[:, :-1]), dim=1)
        coeff = self.base_kernel[side] + delta.repeat_interleave(x.shape[-1] // self.groups, -1)
        return coeff[..., 0, :] * x + coeff[..., 1, :] * previous

    def prepare(self, x, boundary):
        delta = self.kernel_projection(x).unflatten(-1, (2, 2, self.groups))
        return self.convolve(x, delta[..., 0, :, :], 0, boundary), delta[..., 1, :, :]


# Contract checks in the per-cycle decode path must not stall the host.
# `torch.equal` and `Tensor.any()` each produce a Python bool, which is a
# device->host round trip that idles the GPU mid-cycle; the path below has
# seven of them. `require` keeps the check on device instead. A violation
# still aborts the process -- loudly, but at the next synchronization rather
# than at the failing line, and it trashes the CUDA context on the way out.
# Set MTP3_STRICT_SYNC=1 to get the synchronous check with a usable traceback.
STRICT_SYNC = os.environ.get("MTP3_STRICT_SYNC") == "1"


def require(condition, message):
    """Assert a 0-dim boolean tensor without a device->host round trip."""
    if STRICT_SYNC:
        if not bool(condition):
            raise RuntimeError(message)
        return
    torch._assert_async(condition, message)


def canonical(tensor):
    """A fresh tensor carrying the default strides for its shape.

    `contiguous()` is a no-op when the only odd strides sit on size-1 axes --
    torch ignores those when deciding contiguity -- but torch.compile guards on
    the raw stride values, so a batch-of-one cache carved out of a longer trace
    still forces a recompile, once per trace length. Copying into a fresh
    allocation normalizes them.
    """
    return torch.empty_like(tensor, memory_format=torch.contiguous_format).copy_(tensor)


def swa_mask(query_positions, key_positions, window):
    distance = query_positions[:, :, None] - key_positions[:, None, :]
    return (distance >= 0) & (distance < window)


@dataclass(frozen=True)
class LayerCache:
    key: Tensor
    value: Tensor
    # Raw pre-conv input and raw operator output for attention and MoE.
    boundaries: tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class CleanCache:
    positions: Tensor
    layers: tuple[LayerCache, ...]

    def before(self, anchors: Tensor, window: int):
        """Gather independent clean prefixes for anchors from one clean trace.

        Excludes anchor itself: it will be recomputed as proposal slot zero.
        Fixed-width left padding has position -1 and is never attendable.
        """
        if self.positions.shape[0] != 1 or anchors.ndim != 1:
            raise ValueError("sampling requires one sequence and 1D anchors")
        require(
            (
                self.positions[0]
                == torch.arange(
                    self.positions[0, 0], self.positions[0, 0] + self.positions.shape[1], device=anchors.device
                )
            ).all(),
            "clean positions must be contiguous",
        )
        wanted = anchors[:, None] + torch.arange(-window, 0, device=anchors.device)
        valid = wanted >= self.positions[0, 0]
        require((anchors <= self.positions[0, -1] + 1).all(), "anchor beyond clean trace")
        indices = (wanted - self.positions[0, 0]).clamp(0, self.positions.shape[1] - 1)
        layers = tuple(
            LayerCache(
                layer_cache.key[0, :, indices].permute(1, 0, 2, 3),
                layer_cache.value[0, :, indices].permute(1, 0, 2, 3),
                tuple(b[0, indices[:, -1:]] * valid[:, -1:, None] for b in layer_cache.boundaries),
            )
            for layer_cache in self.layers
        )
        return CleanCache(wanted.masked_fill(~valid, -1), layers)

    @staticmethod
    def stack(views):
        """One padded batch out of several single-sequence window views.

        Each request keeps its own clean cache at a fixed width with -1 left
        padding (`serving.decode_cache`), so the batch is a concatenation and
        nothing needs re-gathering.
        """
        if not views:
            raise ValueError("stacking needs at least one window view")
        if len(views) == 1:
            # Concatenating a single view copies every tensor in it for nothing:
            # a one-row batch is already the layout the drafter wants, and a
            # slice of a contiguous batch already carries a fresh row's strides.
            return views[0]
        widths = {view.positions.shape for view in views}
        if len(widths) != 1:
            raise ValueError(f"window views disagree on width: {widths}")
        return CleanCache(
            torch.cat([view.positions for view in views], 0),
            tuple(
                LayerCache(
                    torch.cat([view.layers[i].key for view in views], 0),
                    torch.cat([view.layers[i].value for view in views], 0),
                    tuple(
                        torch.cat([view.layers[i].boundaries[j] for view in views], 0)
                        for j in range(len(views[0].layers[i].boundaries))
                    ),
                )
                for i in range(len(views[0].layers))
            ),
        )


def append_cache(prefix, new, positions, window=None):
    """Concatenate a freshly computed block onto a clean prefix, then trim to window.

    key/value are [batch, heads, sequence, dim] so they concatenate on dim 2;
    boundaries are [batch, sequence, hidden] and concatenate on dim 1.
    """
    if prefix is not None:
        layers = tuple(
            LayerCache(
                torch.cat((a.key, b.key), 2),
                torch.cat((a.value, b.value), 2),
                tuple(torch.cat((x, y), 1) for x, y in zip(a.boundaries, b.boundaries, strict=True)),
            )
            for a, b in zip(prefix.layers, new.layers, strict=True)
        )
        new = CleanCache(torch.cat((prefix.positions, positions), 1), layers)
    if window is None:
        return new
    return CleanCache(
        new.positions[:, -window:],
        tuple(
            LayerCache(
                layer_cache.key[:, :, -window:],
                layer_cache.value[:, :, -window:],
                tuple(x[:, -window:] for x in layer_cache.boundaries),
            )
            for layer_cache in new.layers
        ),
    )


def row_of_block(cache, row):
    """One request's slice of a cache computed for a whole batch."""
    return CleanCache(
        cache.positions[row : row + 1],
        tuple(
            LayerCache(
                layer_cache.key[row : row + 1],
                layer_cache.value[row : row + 1],
                tuple(b[row : row + 1] for b in layer_cache.boundaries),
            )
            for layer_cache in cache.layers
        ),
    )


def keep_prefix_of_block(cache, positions, keep):
    """Retain only the first `keep` slots of a freshly computed block."""
    return CleanCache(
        positions[:, :keep],
        tuple(
            LayerCache(
                layer_cache.key[:, :, :keep],
                layer_cache.value[:, :, :keep],
                tuple(x[:, :keep] for x in layer_cache.boundaries),
            )
            for layer_cache in cache.layers
        ),
    )


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.backend = "sdpa"
        self.q_proj = nn.Linear(c.hidden_size, 2 * c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm, self.k_norm = (RMSNorm(c.head_dim, c.rms_norm_eps) for _ in range(2))

    def rotary(self, x, positions):
        dim = int(self.c.head_dim * self.c.partial_rotary_factor)
        freq = 1.0 / self.c.rope_theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        angle = positions.float()[:, None, :, None] * freq
        angle = torch.cat((angle, angle), -1)
        a, b = x[..., :dim].chunk(2, -1)
        rotated = x[..., :dim] * angle.cos().to(x.dtype) + torch.cat((-b, a), -1) * angle.sin().to(x.dtype)
        return torch.cat((rotated, x[..., dim:]), -1)

    def forward(self, x, positions, prefix, prefix_positions):
        c = self.c
        q, gate = self.q_proj(x).unflatten(-1, (c.num_attention_heads, 2 * c.head_dim)).chunk(2, -1)
        q = self.rotary(self.q_norm(q).transpose(1, 2), positions)
        k = self.rotary(
            self.k_norm(self.k_proj(x).unflatten(-1, (c.num_key_value_heads, c.head_dim))).transpose(1, 2), positions
        )
        v = self.v_proj(x).unflatten(-1, (c.num_key_value_heads, c.head_dim)).transpose(1, 2)
        keys, values, key_positions = k, v, positions
        if prefix is not None:
            keys, values = torch.cat((prefix.key, k), 2), torch.cat((prefix.value, v), 2)
            key_positions = torch.cat((prefix_positions, positions), 1)
        if self.backend == "stable":
            from stable_kernels import causal_swa

            if prefix is not None and (prefix.key.requires_grad or prefix.value.requires_grad):
                raise ValueError("prefix cache must be detached")
            y = causal_swa(q, keys, values, positions, key_positions, c.window, keys.shape[2] - q.shape[2])
        elif self.backend == "fa4":
            from flash_attn.cute import flash_attn_varlen_func

            keep = key_positions >= 0
            lengths = keep.sum(-1).to(torch.int32)
            cu_k = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
            cu_q = torch.arange(x.shape[0] + 1, device=x.device, dtype=torch.int32) * x.shape[1]
            y = flash_attn_varlen_func(
                q.transpose(1, 2).flatten(0, 1),
                keys.transpose(1, 2)[keep],
                values.transpose(1, 2)[keep],
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=x.shape[1],
                max_seqlen_k=keys.shape[2],
                causal=True,
                window_size=(c.window - 1, 0),
            )
            y = y.unflatten(0, x.shape[:2]).transpose(1, 2)
        elif self.backend == "sdpa":
            mask = swa_mask(positions, key_positions, c.window) & (key_positions[:, None] >= 0)
            y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask[:, None], enable_gqa=True)
        else:
            raise ValueError(f"unknown attention backend: {self.backend}")
        y = y.transpose(1, 2).flatten(-2) * torch.sigmoid(gate.flatten(-2))
        return self.o_proj(y), k, v


class MLP(nn.Module):
    def __init__(self, c, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Experts(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fp32_output = False
        self.stable_gemm = False
        self.gate_up_proj = nn.Parameter(torch.empty(c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(c.num_experts, c.hidden_size, c.moe_intermediate_size))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x, indices, weights):
        output_dtype = x.dtype
        x = x.to(self.gate_up_proj.dtype)
        # Sorted grouped GEMMs avoid materializing [tokens, topk, expert weights].
        order = indices.flatten().argsort(stable=True)
        token = order // indices.shape[-1]
        # bincount syncs to size its own output; scatter_add_ counts the same
        # integers without leaving the device, so this stays CUDA-graph safe.
        flat = indices.flatten()
        counts = torch.zeros(self.gate_up_proj.shape[0], device=flat.device, dtype=torch.long)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        offsets = counts.cumsum(0).to(torch.int32)
        if x.is_cuda and self.stable_gemm:
            from stable_kernels import grouped_linear

            gate, up = grouped_linear(x[token], self.gate_up_proj, offsets).chunk(2, -1)
            y = grouped_linear(F.silu(gate) * up, self.down_proj, offsets)
        elif x.is_cuda and self.fp32_output:
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "FP32-output expert loop is an inference diagnostic; aten::mm out_dtype has no backward"
                )
            chunks, begin = [], 0
            for expert, count in enumerate(counts.tolist()):
                if count:
                    z = x[token[begin : begin + count]]
                    gate, up = torch.mm(z, self.gate_up_proj[expert].T, out_dtype=torch.float32).chunk(2, -1)
                    mid = (F.silu(gate) * up).to(self.down_proj.dtype)
                    chunks.append(torch.mm(mid, self.down_proj[expert].T, out_dtype=torch.float32))
                begin += count
            y = torch.cat(chunks)
        elif x.is_cuda:
            gate, up = torch._grouped_mm(x[token], self.gate_up_proj.transpose(1, 2), offs=offsets).chunk(2, -1)
            y = torch._grouped_mm(F.silu(gate) * up, self.down_proj.transpose(1, 2), offs=offsets)
        else:
            chunks, begin = [], 0
            for expert, count in enumerate(counts.tolist()):
                z = x[token[begin : begin + count]]
                gate, up = F.linear(z, self.gate_up_proj[expert]).chunk(2, -1)
                chunks.append(F.linear(F.silu(gate) * up, self.down_proj[expert]))
                begin += count
            y = torch.cat(chunks)
        weighted = y.to(output_dtype) * weights.flatten()[order, None].to(output_dtype)
        # Restore per-token top-k order, then reduce in FP32. A single BF16
        # index_add over all experts uses atomic additions whose order changes
        # with chunk size and GPU scheduling; three inherited GR layers amplify
        # that nondeterminism. There is no cross-token reduction here.
        inverse = torch.empty_like(order).scatter_(0, order, torch.arange(len(order), device=order.device))
        return weighted[inverse].unflatten(0, indices.shape).sum(1, dtype=torch.float32).to(output_dtype)


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.gate = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.experts = Experts(c)
        self.shared_expert = MLP(c, c.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probability = self.gate(x).float().softmax(-1)
        weights, indices = probability.topk(self.c.num_experts_per_tok, -1)
        if self.c.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        y = self.experts(x, indices, weights.to(x.dtype))
        y = y + self.shared_expert(x) * torch.sigmoid(self.shared_expert_gate(x))
        return y.reshape(shape)


class DenseWithShared(nn.Module):
    """The routed MoE replaced by one dense SwiGLU; the shared expert and its sigmoid gate as in MoE."""

    def __init__(self, c):
        super().__init__()
        self.routed = MLP(c, c.dense_intermediate_size)
        self.shared_expert = MLP(c, c.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, x):
        return self.routed(x) + self.shared_expert(x) * torch.sigmoid(self.shared_expert_gate(x))


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MoE(c) if c.ffn_type == "moe" else DenseWithShared(c)
        self.attn_hyper_connection, self.mlp_hyper_connection = GatedResidual(c), GatedResidual(c)
        self.attention_conv, self.mlp_conv = GroupedConv(c), GroupedConv(c)

    def forward(self, h, positions, prefix, prefix_positions):
        zero = h.new_zeros(h.shape[0], 1, h.shape[-1])
        boundary = (zero,) * 4 if prefix is None else tuple(x[:, -1:] for x in prefix.boundaries)
        raw_a, gate = self.attn_hyper_connection(h)
        x, delta = self.attention_conv.prepare(raw_a, boundary[0])
        raw_b, k, v = self.self_attn(x, positions, prefix, prefix_positions)
        y = self.attention_conv.convolve(raw_b, delta, 1, boundary[1])
        h = h + y.unsqueeze(-2) * gate.unsqueeze(-1)
        raw_c, gate = self.mlp_hyper_connection(h)
        x, delta = self.mlp_conv.prepare(raw_c, boundary[2])
        raw_d = self.mlp(x)
        y = self.mlp_conv.convolve(raw_d, delta, 1, boundary[3])
        h = h + y.unsqueeze(-2) * gate.unsqueeze(-1)
        return h, LayerCache(k, v, (raw_a, raw_b, raw_c, raw_d))


class CandidateSelector(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.top_k = c.selector_top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(c.vocab_size, c.selector_rank))
        self.successor_codebook = nn.Parameter(torch.zeros(c.vocab_size, c.selector_rank))
        self.hidden_projection = nn.Linear(c.hidden_size, c.selector_rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=0.02)

    def scores(self, unary, projected, successor, predecessor):
        """The selector score with the position-independent factors already built.

        Only the predecessor lookup is sequential, so a caller stepping through a
        proposal chain can hoist `hidden_projection` and the successor gather out
        of its loop. `forward` delegates here so the two cannot drift apart.
        """
        context = self.predecessor_codebook[predecessor] * projected
        return unary + torch.einsum("...r,...kr->...k", context, successor)

    def forward(self, candidates, unary, hidden, predecessor):
        return self.scores(unary, self.hidden_projection(hidden), self.successor_codebook[candidates], predecessor)


class MTP3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size).requires_grad_(False)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False).requires_grad_(False)
        self.mask_embedding = nn.Parameter(torch.zeros(c.hidden_size))
        self.pre_fc_norm_embedding = RMSNorm(c.hidden_size, c.rms_norm_eps)
        # Native MTP normalizes across all 4 branches here, unlike GR's norms.
        self.pre_fc_norm_hidden = RMSNorm(c.hc_count * c.hidden_size, c.rms_norm_eps)
        self.fc_embedding = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.layers = nn.ModuleList(Decoder(c) for _ in range(c.num_layers))
        self.hyper_connection_mixer = GatedResidual(c, combine=False)
        self.target_reducer = GatedResidual(c, combine=False).requires_grad_(False)
        self.candidate_selector = CandidateSelector(c)

    def forward(self, embeddings, predecessor_gr, positions, prefix=None):
        # predecessor_gr[i] pairs with embeddings[i]: the target GR one position back.
        if predecessor_gr.shape != (*embeddings.shape[:-1], self.config.hc_count, self.config.hidden_size):
            raise ValueError("invalid final pre-collapse target GR shape")
        h = self.pre_fc_norm_hidden(predecessor_gr.flatten(-2)).unflatten(
            -1, (self.config.hc_count, self.config.hidden_size)
        )
        h = self.fc_hidden(h) + self.fc_embedding(self.pre_fc_norm_embedding(embeddings)).unsqueeze(-2)
        caches = []
        for i, layer in enumerate(self.layers):
            h, cache = layer(
                h, positions, None if prefix is None else prefix.layers[i], None if prefix is None else prefix.positions
            )
            caches.append(cache)
        return self.hyper_connection_mixer(h), CleanCache(positions, tuple(caches))

    @torch.no_grad()
    def refresh(self, tokens, predecessor_gr, positions, prefix=None, *, retain_trace=False):
        """Append committed tokens; predecessor_gr[i] is the target GR at positions[i] - 1.

        The drafter never consumes target state at or beyond its own position:
        that state does not exist until the target reads the token, which is one
        verify later. Position 0 has no predecessor, so the draft context starts
        at position 1. This is the DFlash/DSpark/NEXTN contract.
        """
        if tokens.shape != positions.shape or tokens.shape[1] == 0:
            raise ValueError("refresh needs nonempty aligned tokens and positions")
        expected = positions[:, :1] + torch.arange(positions.shape[1], device=positions.device)
        if not torch.equal(positions, expected):
            raise ValueError("refresh positions must be consecutive")
        if (positions < 1).any():
            raise ValueError("position 0 has no predecessor GR; the draft context starts at position 1")
        if prefix is not None and not torch.equal(positions[:, 0], prefix.positions[:, -1] + 1):
            raise ValueError("refresh must append each committed position exactly once")
        # Deliberately the class function, not `self(...)`: serving replaces the
        # instance's `forward` with a compiled one specialized on shape, and a
        # prefill chunk is as long as the prompt leaves it. Compiling those is
        # unbounded work -- we measured 10 to 30 second stalls, one per distinct
        # prompt length, paid as time to first token. Prefill runs once per
        # request over the prompt, so eager is the right trade.
        hidden, new = MTP3.forward(self, self.embed_tokens(tokens), predecessor_gr, positions, prefix)
        new = append_cache(prefix, new, positions, None if retain_trace else self.config.window)
        return hidden, new

    def decode_step(self, committed_tokens, anchor_tokens, gr_span, first, prefix, *, width=None, lengths=None):
        """One forward per cycle: commit the verified tokens and propose the next block.

        Layout `[committed_1..k, anchor, mask x(b-1), pad...]` at consecutive
        positions starting at `first`. Committed slots carry real token
        embeddings and their own predecessor GR, so their state is exactly what
        a separate refresh would have produced; the anchor and the mask slots
        share the newest GR, exactly as the standalone proposal block did.
        Proposal slots sit strictly to the right of every committed slot, so
        causal attention and the width-2 convolution both keep them out of the
        retained state.

        `lengths` gives each request its own commit count, which is what a real
        batch looks like: requests accept different numbers of drafts. Without
        it every row commits all of `committed_tokens`. With it, the tokens are
        right-padded to a common width and each row's anchor sits at its own
        `lengths[row]`; the trailing slots are mask embeddings whose positions
        run past the block, so causal attention keeps them out of every real
        slot's context.

        Returns the proposal hidden states, the committed prefix of the block,
        and the committed positions -- the last two right-padded when `lengths`
        varies, so the caller trims per request.
        """
        b = self.config.block_size
        k = committed_tokens.shape[1]
        batch = committed_tokens.shape[0]
        if committed_tokens.ndim != 2 or anchor_tokens.ndim != 1 or anchor_tokens.shape[0] != batch:
            raise ValueError("decode_step takes [batch, committed] tokens and one anchor per row")
        if k > b:
            raise ValueError("a cycle commits at most one block of tokens")
        if gr_span.shape[:2] != (batch, k + 1):
            raise ValueError("need one predecessor GR per committed token plus the new cursor GR")
        if lengths is not None and (lengths.ndim != 1 or lengths.shape[0] != batch):
            raise ValueError("lengths gives one commit count per row")
        n = k + b if width is None else width
        if n < k + b:
            raise ValueError("block width must hold every committed token and the whole proposal block")
        slot = torch.arange(n, device=first.device)
        positions = first[:, None] + slot
        require(
            (positions[:, 0] >= 1).all(),
            "position 0 has no predecessor GR; the draft context starts at position 1",
        )
        if prefix is not None:
            require(
                (positions[:, 0] == prefix.positions[:, -1] + 1).all(),
                "decode_step must append each committed position exactly once",
            )
            require(
                ~((prefix.positions >= positions[:, :1]) & (prefix.positions >= 0)).any(),
                "prefix must exclude this block and every future position",
            )
        if lengths is None:
            embeds = torch.cat(
                (
                    self.embed_tokens(committed_tokens),
                    self.embed_tokens(anchor_tokens)[:, None],
                    self.mask_embedding.expand(batch, n - k - 1, -1),
                ),
                1,
            )
            # Slots at and after the anchor condition on the newest GR the target
            # has produced; the target has not read the anchor, so nothing later
            # exists.
            grs = torch.cat((gr_span[:, :-1], gr_span[:, -1:].expand(-1, n - k, -1, -1)), 1)
        else:
            require((lengths <= k).all(), "a row cannot commit more tokens than the block holds")
            edge = lengths[:, None]
            committed = slot < edge
            spoken = slot <= edge
            ids = torch.where(committed, F.pad(committed_tokens, (0, n - k)), anchor_tokens[:, None])
            embeds = torch.where(spoken[..., None], self.embed_tokens(ids), self.mask_embedding)
            # Row `i` reads gr_span[i] for its own committed slots and the cursor
            # GR from lengths[i] onward -- the same rule, clamped per row.
            index = slot.minimum(edge)
            grs = gr_span.gather(1, index[..., None, None].expand(-1, -1, *gr_span.shape[2:]))
        hidden, block = self(embeds, grs, positions, prefix)
        if lengths is None:
            # Slot k is the anchor, and under the next-token contract its output
            # is the first proposal. Its KV is still discarded: the anchor is
            # recomputed with its own GR once the target has read it.
            return hidden[:, k : k + b - 1], keep_prefix_of_block(block, positions, k), positions[:, :k]
        take = (lengths[:, None] + torch.arange(b - 1, device=first.device)).clamp(max=n - 1)
        proposal = hidden.gather(1, take[..., None].expand(-1, -1, hidden.shape[-1]))
        return proposal, keep_prefix_of_block(block, positions, k), positions[:, :k]

    def propose(self, anchor_tokens, predecessor_gr, anchors, prefix):
        """predecessor_gr is the target GR at anchor-1, the newest state the target has read.

        The anchor is the previous cycle's bonus token, which the target has not
        consumed yet, so no target state exists at the anchor position. It enters
        as a token embedding only, exactly as native DFlash/DSpark slot zero.

        Slot j predicts x[a+j+1] -- the native MTP next-token contract. The
        anchor slot is therefore a proposal like any other, and the first one is
        conditioned on the real anchor token in its own residual stream rather
        than through attention to a mask carrying the same GR.
        """
        b = self.config.block_size
        if (anchors < 1).any():
            raise ValueError("anchor 0 has no predecessor GR")
        positions = anchors[:, None] + torch.arange(b, device=anchors.device)
        if prefix is not None and ((prefix.positions >= anchors[:, None]) & (prefix.positions >= 0)).any():
            raise ValueError("proposal prefix must exclude anchor and every future position")
        embeds = torch.cat(
            (self.embed_tokens(anchor_tokens)[:, None], self.mask_embedding.expand(len(anchors), b - 1, -1)), 1
        )
        hidden, _ = self(embeds, predecessor_gr[:, None].expand(-1, b, -1, -1), positions, prefix)
        # The last slot would predict x[a+b], which a b-wide verify can never
        # reach, so it is dropped: b - 1 proposals for b - 1 verified positions.
        # Proposal KV is discarded whatever the outputs are used for.
        return hidden[:, :-1]


def tiny_config(**kwargs):
    return replace(
        Config(
            hidden_size=32,
            hc_lowrank=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            vocab_size=64,
            selector_rank=8,
            selector_top_k=8,
            window=8,
        ),
        **kwargs,
    )
