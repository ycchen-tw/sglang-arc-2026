"""Serving-side forward replacements for the drafter on sm_120, same math as model.py.

- Attention: SDPA with a boolean mask and enable_gqa falls back to the math kernel on sm_120 (34 ms per cycle at
  55 requests). The 12 query heads that share a KV head are folded into the query length instead, so SDPA sees
  plain MHA over 2 heads with a [12 x S, K] mask and can take the memory-efficient kernel.
- GroupedConv: `delta.repeat_interleave(16)` materializes a [tokens, 2, 2, 2560] coefficient tensor per call;
  the grouped view applies the same per-group coefficients by broadcasting.

`install(model)` swaps the instance methods; `check(model)` compares them with model.py's on random inputs.
"""

from __future__ import annotations

import os
import types

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import ring_attention
from . import serving
from .model import Attention, GroupedConv, swa_mask

_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]


def attention_forward(self, x, positions, prefix, prefix_positions):
    c = self.c
    q, gate = self.q_proj(x).unflatten(-1, (c.num_attention_heads, 2 * c.head_dim)).chunk(2, -1)
    q = self.rotary(self.q_norm(q).transpose(1, 2), positions)
    k = self.rotary(self.k_norm(self.k_proj(x).unflatten(-1, (c.num_key_value_heads, c.head_dim))).transpose(1, 2), positions)
    v = self.v_proj(x).unflatten(-1, (c.num_key_value_heads, c.head_dim)).transpose(1, 2)
    if prefix is not None and isinstance(prefix.key, ring_attention.RingWindow):
        # straight from the ring pool (no gathered copy, no spare columns)
        y = ring_attention.attend(q, k, v, positions, prefix.key, prefix.value, prefix_positions, c.window)
        y = y.transpose(1, 2).flatten(-2) * torch.sigmoid(gate.flatten(-2))
        return self.o_proj(y), k, v
    keys, values, key_positions = k, v, positions
    if prefix is not None:
        length = k.shape[2]
        if prefix.key.shape[2] == prefix_positions.shape[1] + length:
            # state_pool.RingStatePool hands over a gathered copy with `length` spare columns at the end: the
            # block's K/V go there in place instead of concatenating ~1,040 keys per layer
            keys, values = prefix.key, prefix.value
            keys[:, :, -length:] = k
            values[:, :, -length:] = v
        else:
            keys, values = torch.cat((prefix.key, k), 2), torch.cat((prefix.value, v), 2)
        key_positions = torch.cat((prefix_positions, positions), 1)
    batch, heads, length, dim = q.shape
    group = heads // c.num_key_value_heads
    mask = swa_mask(positions, key_positions, c.window) & (key_positions[:, None] >= 0)
    # query head h reads KV head h // group (SDPA's enable_gqa mapping): fold the group into the query axis
    folded = q.reshape(batch, c.num_key_value_heads, group * length, dim)
    with sdpa_kernel(_SDPA_BACKENDS):
        y = F.scaled_dot_product_attention(folded, keys, values, attn_mask=mask.repeat(1, group, 1)[:, None])
    y = y.reshape(batch, heads, length, dim)
    y = y.transpose(1, 2).flatten(-2) * torch.sigmoid(gate.flatten(-2))
    return self.o_proj(y), k, v


def convolve(self, x, delta, side, boundary):
    previous = torch.cat((boundary, x[:, :-1]), dim=1)
    base = self.base_kernel[side]  # [2, hidden]
    group = x.shape[-1] // self.groups
    xs = x.unflatten(-1, (self.groups, group))
    ps = previous.unflatten(-1, (self.groups, group))
    b0, b1 = base[0].view(self.groups, group), base[1].view(self.groups, group)
    d0, d1 = delta[..., 0, :, None], delta[..., 1, :, None]
    return ((b0 + d0) * xs + (b1 + d1) * ps).flatten(-2)


def install(model):
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = types.MethodType(attention_forward, module)
        elif isinstance(module, GroupedConv):
            module.convolve = types.MethodType(convolve, module)
    # default on (server: c110 K=4 draft 13.4 -> 10.9 ms, +25 s at start for the compile); ARC_DFLASH_COMPILE=0 off
    if os.environ.get("ARC_DFLASH_COMPILE", "1") == "1":
        compile_layers(model)


def compile_layers(model):
    """Fuse the drafter's elementwise work with inductor: each decoder layer (hyper-connection norms and low-rank
    mixes, grouped convs, residual updates, q/k norm and rotary, router softmax/top-k, shared expert) and the final
    mixer. The window attention (Triton), the NVFP4 experts and the block-FP8 linears (CUTLASS) stay opaque: graph
    breaks around them.
    dynamic=True keeps one compile per layer across the CUDA graph buckets; it happens in the eager warm-up that
    precedes each capture, never inside one."""
    from . import fp8, nvfp4

    ring_attention.attend = torch.compiler.disable(ring_attention.attend)
    nvfp4.Nvfp4Experts.forward = torch.compiler.disable(nvfp4.Nvfp4Experts.forward)
    fp8.Fp8BlockLinear.forward = torch.compiler.disable(fp8.Fp8BlockLinear.forward)
    for layer in model.layers:
        layer.forward = torch.compile(layer.forward, dynamic=True)
    model.hyper_connection_mixer.forward = torch.compile(model.hyper_connection_mixer.forward, dynamic=True)


@torch.no_grad()
def check(model, batch=3, length=16, prefix_len=1024, seed=0):
    """Max relative error of the replacements against model.py's own methods (same module, same inputs)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    attn = model.layers[0].self_attn
    conv = model.layers[0].attention_conv
    c = model.config
    x = torch.randn(batch, length, c.hidden_size, device="cuda", dtype=torch.bfloat16, generator=g)
    pos = torch.arange(prefix_len, prefix_len + length, device="cuda")[None].expand(batch, -1)
    prefix_pos = torch.arange(0, prefix_len, device="cuda")[None].expand(batch, -1).clone()
    prefix_pos[0, :100] = -1
    from .model import LayerCache

    kv = torch.randn(batch, c.num_key_value_heads, prefix_len, c.head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    prefix = LayerCache(kv, kv.flip(-1).contiguous(), ())
    ref = Attention.forward(attn, x, pos, prefix, prefix_pos)[0].float()
    got = attention_forward(attn, x, pos, prefix, prefix_pos)[0].float()
    out = {"attention": ((ref - got).norm() / ref.norm()).item()}
    spare = LayerCache(torch.cat((prefix.key, torch.zeros_like(prefix.key[:, :, :length])), 2),
                       torch.cat((prefix.value, torch.zeros_like(prefix.value[:, :, :length])), 2), ())
    inplace = attention_forward(attn, x, pos, spare, prefix_pos)[0].float()
    out["in_place_equal"] = bool(torch.equal(inplace, got))
    delta = torch.randn(batch, length, 2, conv.groups, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    boundary = torch.randn(batch, 1, c.hidden_size, device="cuda", dtype=torch.bfloat16, generator=g)
    ref = GroupedConv.convolve(conv, x, delta, 1, boundary).float()
    got = convolve(conv, x, delta, 1, boundary).float()
    out["conv"] = ((ref - got).norm() / ref.norm()).item()
    return out


def advance(worker, states, anchor, temperature):
    batch = len(states)
    ones = torch.ones(batch, device=anchor.device)
    return serving.advance_batch(
        worker.model, states, anchor, temperature=temperature, top_p=ones,
        top_k=torch.full((batch,), -1, device=anchor.device), propose=True,
    )


def warm(worker):
    """Nothing to compile on the eager path; CUDA-graph capture hooks in here."""
    if os.environ.get("ARC_DFLASH_CHECK_FAST") == "1":
        print("DFlash fast-path check:", check(worker.model), flush=True)
