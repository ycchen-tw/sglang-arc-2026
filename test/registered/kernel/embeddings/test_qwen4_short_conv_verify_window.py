"""Qwen4 PLE short conv, target verify: the fused step-window write must store the same bits as the native
unfold / permute / contiguous / where / copy_ chain, for every BF16 bit pattern (NaN payloads, -0, inf), invalid
steps and requests, a cache with more request slots and draft steps than the batch, and leave the rest untouched.
"""

import pytest
import torch

from sglang.kernels.ops.qwen4_ple import (
    can_fuse_qwen4_short_conv_verify_window,
    fused_qwen4_short_conv_verify_window,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")


def _native(conv_input, valid_tokens, cache, row_width, state_len):
    requests = conv_input.shape[0]
    window = conv_input.unfold(2, state_len, 1)[:, :, 1 : row_width + 1].permute(0, 2, 1, 3).contiguous()
    valid_steps = valid_tokens.reshape(requests, row_width, 1, 1)
    window = torch.where(valid_steps, window, torch.zeros_like(window))
    cache[:requests, :row_width].copy_(window.to(dtype=cache.dtype))


@pytest.mark.parametrize(
    "requests,row_width,channels,state_len,cache_requests,cache_steps",
    [(6, 4, 160, 9, 8, 5), (48, 4, 10240, 9, 49, 4), (3, 2, 96, 2, 3, 2), (1, 1, 64, 16, 2, 1)],
)
def test_verify_window_bitwise(requests, row_width, channels, state_len, cache_requests, cache_steps):
    g = torch.Generator(device="cuda").manual_seed(0)
    bits = torch.randint(-(2**15), 2**15, (requests, channels, state_len + row_width), generator=g, device="cuda")
    conv_input = bits.to(torch.int16).view(torch.bfloat16)
    conv_input.view(torch.int16)[0, : min(channels, 4), 0] = torch.tensor(
        [0x7FC1, -32768, 0x7F80, 0xFF80 - 65536][: min(channels, 4)], dtype=torch.int16, device="cuda"
    )  # NaN payload, -0, +inf, -inf
    valid_tokens = torch.rand(requests * row_width, generator=g, device="cuda") < 0.7
    valid_tokens[:row_width] = True
    if requests > 1:
        valid_tokens[row_width : 2 * row_width] = False  # a padded request
    sentinel = torch.full((cache_requests, cache_steps, channels, state_len), 0x1234, dtype=torch.int16, device="cuda")
    ref = sentinel.clone().view(torch.bfloat16)
    out = sentinel.clone().view(torch.bfloat16)
    assert can_fuse_qwen4_short_conv_verify_window(conv_input, valid_tokens, out, row_width)
    _native(conv_input, valid_tokens, ref, row_width, state_len)
    fused_qwen4_short_conv_verify_window(conv_input, valid_tokens, out, row_width)
    torch.cuda.synchronize()
    assert torch.equal(ref.view(torch.int16), out.view(torch.int16))
    assert (out.view(torch.int16)[requests:] == 0x1234).all() and (out.view(torch.int16)[:, row_width:] == 0x1234).all()


def test_verify_window_rejects_unsupported():
    conv_input = torch.zeros(2, 8, 13, dtype=torch.bfloat16, device="cuda")
    valid = torch.ones(8, dtype=torch.bool, device="cuda")
    cache = torch.zeros(2, 4, 8, 9, dtype=torch.bfloat16, device="cuda")
    assert can_fuse_qwen4_short_conv_verify_window(conv_input, valid, cache, 4)
    assert not can_fuse_qwen4_short_conv_verify_window(conv_input, valid, cache.float(), 4)  # dtype change
    assert not can_fuse_qwen4_short_conv_verify_window(conv_input, valid, cache[:1], 4)  # too few request slots
    assert not can_fuse_qwen4_short_conv_verify_window(conv_input, valid[:4], cache, 4)
    assert not can_fuse_qwen4_short_conv_verify_window(conv_input.transpose(1, 2), valid, cache, 4)
    with pytest.raises(ValueError, match="unsupported input"):
        fused_qwen4_short_conv_verify_window(conv_input, valid, cache.float(), 4)
