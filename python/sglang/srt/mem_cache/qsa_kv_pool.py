"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-request pending
index-key/RoPE ring and the paged compressed-K cache on top of the hybrid
full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool, MHATokenToKVPool

# State layer IDs are serialized as uint32 by the disaggregation protocols.
# Reserve the value below PLE's request-wide sentinel for QSA's request-wide
# RoPE ring, which is shared by all full-attention layers.
QSA_ROPE_STATE_LAYER_ID = (1 << 32) - 2


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


def qsa_index_fp8_enabled() -> bool:
    """arc-forge h21: ARC_QSA_INDEX_FP8=1 stores the compressed index keys as FP8 (e4m3) with one power-of-two fp32
    scale per compressed key (per key head), instead of BF16. Default 0 (off). Read wherever the cache is sized or
    built (planner, target and draft pools), so both sides agree."""
    value = os.environ.get("ARC_QSA_INDEX_FP8", "0")
    if value not in ("0", "1"):
        raise ValueError(f"ARC_QSA_INDEX_FP8 must be 0 or 1, got {value!r}")
    return value == "1"


QSA_INDEX_FP8_DTYPE = torch.float8_e4m3fn
QSA_INDEX_FP8_MAX = 448.0


def quantize_qsa_index_k(keys: torch.Tensor):
    """Reference FP8 quantization of compressed index keys [..., head_dim] (the fused compress kernel matches it bit
    for bit): scale = the smallest power of two with amax / scale <= 448, k8 = e4m3(key / scale) rounded to nearest
    even. The key is first rounded to BF16 (what the BF16 cache would hold). Dequantized keys k8 * scale are exact in
    BF16, so every consumer (decode / verify kernels scaling the logits, prefill dequantizing) scores the same keys.
    Returns (k8 [..., head_dim] float8_e4m3fn, scale [...] float32); an all-zero key gets scale 1."""
    x = keys.to(torch.bfloat16).float()
    amax = x.abs().amax(dim=-1)
    mantissa, exponent = torch.frexp(amax)  # amax = m * 2^e, m in [0.5, 1)
    # amax / 2^k <= 448 = 0.875 * 2^9  <=>  k >= e - 9 (m <= 0.875) or e - 8 (m > 0.875)
    k = torch.where(mantissa <= 0.875, exponent - 9, exponent - 8)
    scale = torch.where(amax > 0, torch.ldexp(torch.ones_like(amax), k), torch.ones_like(amax))
    k8 = (x / scale.unsqueeze(-1)).to(QSA_INDEX_FP8_DTYPE)
    return k8, scale


def dequantize_qsa_index_k(k8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """k8 [..., head_dim] e4m3 and scale [...] fp32 -> BF16 keys (exact: scale is a power of two)."""
    return (k8.float() * scale.unsqueeze(-1)).to(torch.bfloat16)


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # Full-KV pages are a multiple of the compress ratio, so no group straddles pages;
    # ``compressed_slot = full_slot // ratio`` needs no ownership bookkeeping;
    # lifecycle rides the full-KV allocator and radix tree.
    # Full slot 0 is the reserved padding slot; compressed slot 0 is the inert dump.
    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, compress_ratio: int, num_layers: int
    ) -> int:
        """Per-token QSA index-cache cost: compressed keys only;
        the per-request pending ring is budgeted with the other per-request buffers."""
        if qsa_index_fp8_enabled():
            # e4m3 key plus one fp32 scale per key head (arc-forge h21)
            index_k_bytes = kv_heads * (head_dim * QSA_INDEX_FP8_DTYPE.itemsize + 4)
        else:
            index_k_bytes = _index_k_bytes(
                kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
            )
        return index_k_bytes // compress_ratio * num_layers

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        qsa_max_verify_tokens: int = 0,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
        index_size: Optional[int] = None,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. With MambaRadixCache this "
                "needs the mamba extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # super().__init__ computes mem_usage via the overridden get_kv_size_bytes,
        # so the QSA buffers get placeholders first; mem_usage is recomputed last.
        self.qsa_key_state_buffer_pool = []
        self.qsa_compressed_k_buffer_pool = []
        self.qsa_compressed_scale_pool = []
        self.qsa_rope_position_buffer = torch.empty(0)
        self.qsa_index_fp8 = qsa_index_fp8_enabled()
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if (
            min(
                qsa_index_kv_heads,
                qsa_index_head_dim,
                qsa_compress_ratio,
                qsa_token_topk,
            )
            <= 0
        ):
            raise ValueError("QSA cache configuration values must be positive")
        if qsa_token_topk % qsa_compress_ratio != 0:
            raise ValueError("qsa_token_topk must be divisible by qsa_compress_ratio")
        self.qsa_compress_ratio = int(qsa_compress_ratio)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_token_topk = int(qsa_token_topk)
        self.qsa_block_topk = self.qsa_token_topk // self.qsa_compress_ratio
        # HiSparse (arc-forge h02): `size` counts device KV rows there, while the
        # index caches stay logically addressed and are sized by `index_size`.
        state_size = (size if index_size is None else index_size) + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        # ARC QSA speculative ring: a verify writes its complete window before
        # compression, so retain the preceding ratio-1 uncompressed keys too.
        # Rejected future keys are overwritten on continuation; committed tail
        # keys must survive without a rollback copy, including native MTP4.
        # Request slot 0 is never allocated and owns the inert extend dump.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        self.qsa_num_request_slots = int(num_request_slots)
        if qsa_max_verify_tokens < 0:
            raise ValueError("QSA verify token capacity must be nonnegative")
        self.qsa_pending_ring_size = self.qsa_compress_ratio + int(qsa_max_verify_tokens)
        ring_slots = self.qsa_num_request_slots * self.qsa_pending_ring_size
        # These buffers participate in Mooncake PD transfer just like the base
        # KV and Mamba pools.  Keep their allocation in the same memory-saver
        # and Mooncake custom-pool regions; otherwise MNNVL cannot resolve the
        # ordinary CUDA allocation when the first QSA state page is sent.
        allocation_pool = self.full_kv_pool
        with (
            allocation_pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(allocation_pool.custom_mem_pool)
                if allocation_pool.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            self.qsa_key_state_buffer_pool = [
                torch.zeros(
                    (
                        ring_slots,
                        self.qsa_index_kv_heads,
                        self.qsa_index_head_dim,
                    ),
                    dtype=self.index_state_dtype,
                    device=device,
                )
                for _ in full_attention_layer_ids
            ]
            # RoPE coordinates are layer-independent. Keep the exact Qwen4-Exp
            # MRoPE position of every incomplete key so compression can rotate
            # the pooled key with the group's real starting coordinate.
            self.qsa_rope_position_buffer = torch.zeros(
                (ring_slots, 3), dtype=torch.int64, device=device
            )
            # One contiguous allocation behind per-layer views: every layer's
            # compressed pages are addressable from a single base pointer.
            self.qsa_compressed_flat = torch.zeros(
                (
                    len(full_attention_layer_ids),
                    self.qsa_compressed_capacity
                    * self.qsa_index_kv_heads
                    * self.qsa_index_head_dim,
                ),
                dtype=(
                    QSA_INDEX_FP8_DTYPE if self.qsa_index_fp8 else self.index_state_dtype
                ),
                device=device,
            )
            if self.qsa_index_fp8:
                # arc-forge h21: one power-of-two scale per compressed key head; ones keep the
                # never-written slots (zeros in k8) at a zero key, as in the BF16 cache
                self.qsa_compressed_scale_flat = torch.ones(
                    (
                        len(full_attention_layer_ids),
                        self.qsa_compressed_capacity * self.qsa_index_kv_heads,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
        self.qsa_compressed_k_buffer_pool = [
            self.qsa_compressed_flat[layer_offset].view(
                self.qsa_compressed_capacity,
                self.qsa_index_kv_heads,
                self.qsa_index_head_dim,
            )
            for layer_offset in range(len(full_attention_layer_ids))
        ]
        if self.qsa_index_fp8:
            self.qsa_compressed_scale_pool = [
                self.qsa_compressed_scale_flat[layer_offset].view(
                    self.qsa_compressed_capacity, self.qsa_index_kv_heads
                )
                for layer_offset in range(len(full_attention_layer_ids))
            ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_qsa_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_key_state_buffer(
        self, layer_id: int, loc: torch.Tensor, token_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_key_state_buffer(layer_id)
        buffer[loc.long()] = token_k.to(buffer.dtype)

    def set_qsa_rope_position_buffer(
        self, loc: torch.Tensor, positions: torch.Tensor
    ) -> None:
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"QSA RoPE positions must be [tokens] or [3, tokens], got {positions.shape}"
            )
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc: torch.Tensor) -> torch.Tensor:
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def get_qsa_compressed_k_scale(self, layer_id: int) -> Optional[torch.Tensor]:
        """[compressed_slots, kv_heads] fp32 scales of the FP8 cache (h21), None for BF16."""
        if not self.qsa_index_fp8:
            return None
        return self.qsa_compressed_scale_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        if self.qsa_index_fp8:
            k8, scale = quantize_qsa_index_k(compressed_k)
            buffer[loc.long()] = k8
            self.get_qsa_compressed_k_scale(layer_id)[loc.long()] = scale
            return
        buffer[loc.long()] = compressed_k.to(buffer.dtype)

    @staticmethod
    def _get_paged_state_buf_infos(tensors, page_size: int):
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [tensor[0].nbytes * page_size for tensor in tensors],
        )

    def get_qsa_pending_state_buf_infos(self):
        """Per-request pending key-state and RoPE ring transfer buffers."""
        # A PP stage without a local QSA layer never writes the shared RoPE
        # ring.  Do not register it as a transfer source: otherwise that stage
        # can race with a QSA-owning stage and overwrite valid positions with
        # its zero-initialized or stale contents.
        if not self.full_attention_layer_id_mapping:
            return [], [], []
        if self.qsa_pending_ring_size != self.qsa_compress_ratio:
            # PD prefill and decode may use different speculative widths. The
            # existing protocol transfers request rows without remapping ring
            # offsets; advertising either stride alone cannot make that safe.
            raise NotImplementedError(
                "QSA speculative pending rings currently support aggregated "
                "serving only; PD requires negotiated ring sizes and remapping"
            )
        tensors = [*self.qsa_key_state_buffer_pool, self.qsa_rope_position_buffer]
        return self._get_paged_state_buf_infos(
            tensors,
            self.qsa_pending_ring_size,
        )

    def get_qsa_pending_state_layer_ids(self):
        """Global layer metadata for the compact QSA pending-state list."""
        if not self.full_attention_layer_id_mapping:
            return []
        return [
            *self.full_attention_layer_id_mapping.keys(),
            QSA_ROPE_STATE_LAYER_ID,
        ]

    def get_qsa_compressed_state_layer_ids(self):
        """Global layer metadata for the compact compressed-K list."""
        return list(self.full_attention_layer_id_mapping.keys())

    def get_qsa_compressed_state_buf_infos(self):
        """Per-full-page compressed-K transfer buffers.

        One full KV page maps to one compressed page because the full page size
        is an integer multiple of the compression ratio.
        """
        if self.qsa_index_fp8:
            raise NotImplementedError(
                "ARC_QSA_INDEX_FP8 (h21) does not transfer the compressed-key scales; PD is unsupported"
            )
        return self._get_paged_state_buf_infos(
            self.qsa_compressed_k_buffer_pool,
            self.qsa_compressed_page_size,
        )

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        qsa_k_size = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_key_state_buffer_pool
            )
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_k_buffer_pool
            )
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_scale_pool
            )
            + self.qsa_rope_position_buffer.numel() * 8
        )
        return k_size + qsa_k_size, v_size


import weakref
_NATIVE_MTP_HOST_KV_POOLS = weakref.WeakSet()

def native_mtp_kv_host_enabled() -> bool:
    value = os.environ.get("ARC_NATIVE_MTP_KV_HOST", "0")
    if value not in ("0", "1"):
        raise ValueError("ARC_NATIVE_MTP_KV_HOST must be 0 or 1")
    if value == "1" and (os.environ.get("ARC_HISPARSE_DISTRIBUTED") != "1"
                         or os.environ.get("ARC_HISPARSE_DISTRIBUTED_SPEC") != "1"):
        raise ValueError("native MTP host KV requires the guarded distributed HiSparse spec profile")
    return value == "1"


def native_mtp_host_kv_bytes(logical, page, heads, key_dim, value_dim) -> int:
    if min(logical, page, heads, key_dim, value_dim) <= 0 or logical % page:
        raise ValueError("native MTP host KV needs positive page-aligned geometry")
    return (logical + page) * heads * (key_dim + value_dim)  # FP8: one byte


class _NativeMTPHostByteAlias:
    def __init__(self, owner):
        import ctypes
        from pathlib import Path
        self.owner = owner
        rt = ctypes.CDLL(str(Path(os.environ["CUDA_HOME"]) / "targets/x86_64-linux/lib/libcudart.so"))
        rt.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        rt.cudaHostGetDevicePointer.restype = ctypes.c_int
        pointer = ctypes.c_void_p()
        code = rt.cudaHostGetDevicePointer(ctypes.byref(pointer), ctypes.c_void_p(owner.data_ptr()), 0)
        if code != 0 or pointer.value is None:
            raise RuntimeError(f"native MTP cudaHostGetDevicePointer failed: {code}")
        self.__cuda_array_interface__ = {"shape": tuple(owner.shape), "strides": None,
            "typestr": "|u1", "data": (pointer.value, False), "version": 3}


class NativeMTPMappedHostFP8Pool(MHATokenToKVPool):
    """Only native MTP's full K/V; logical indexing and all history are unchanged."""
    def __init__(self, *args, **kwargs):
        self._host_tables = []
        self._host_cuda_proxies = []
        self._host_closed = False
        self._host_device = None
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self.release_host_kv()  # Roll back only this constructor's registrations.
            raise
        _NATIVE_MTP_HOST_KV_POOLS.add(self)

    def _create_buffers_normal(self):
        from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator, alloc_with_host_register
        if not native_mtp_kv_host_enabled():
            raise ValueError("native MTP mapped host pool requires explicit opt-in")
        if (self.dtype != torch.float8_e4m3fn or self.store_dtype != torch.uint8
                or self.kv_cache_layout != "nhd" or self.layer_num != 1
                or self.is_quantized_kv_cache or self.post_capture_active or self.enable_custom_mem_pool):
            raise NotImplementedError("native MTP host KV supports one NHD E4M3 layer without custom/post-capture pools")
        device = torch.device(self.device)
        device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 9):
            raise NotImplementedError("native MTP host KV candidate is restricted to SM89 CUDA")
        self._host_device = device  # Preserve the exact registration/alias device for teardown.
        shapes = self._kv_buffer_shapes()
        for shape in shapes:
            table = alloc_with_host_register(shape, dtype=torch.uint8, device="cpu", pin_memory=True,
                allocator=HostTensorAllocator(), registration_granularity_bytes=shape[1] * shape[2])
            self._host_tables.append(table)  # Keep each base before the next allocation can fail.
            table.zero_()  # Match the resident pool's zeroed padding/dummy page.
        self._host_cuda_proxies = [_NativeMTPHostByteAlias(table) for table in self._host_tables]
        aliases = [torch.as_tensor(proxy, dtype=torch.uint8, device=device) for proxy in self._host_cuda_proxies]
        for alias, proxy in zip(aliases, self._host_cuda_proxies):
            if alias.device != device or alias.data_ptr() != proxy.__cuda_array_interface__["data"][0]:
                raise RuntimeError("native MTP CUDA alias copied storage or changed device")
        self.k_buffer, self.v_buffer = [aliases[0]], [aliases[1]]
        expected = native_mtp_host_kv_bytes(self.size, self.page_size, self.head_num, self.head_dim, self.v_head_dim)
        if self.get_host_kv_size_bytes() != expected:
            raise RuntimeError("native MTP host allocation differs from declared geometry")

    def get_kv_size_bytes(self):
        return 0, 0  # Payload is host-resident; QSATokenToKVPool adds GPU index bytes.

    def get_host_kv_size_bytes(self):
        return sum(table.numel() * table.element_size() for table in self._host_tables)

    def release_host_kv(self):
        if self._host_closed:
            return
        if self._host_tables:
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
            if self._host_device is None:
                raise RuntimeError("native MTP host KV owner device was not bound")
            torch.cuda.synchronize(self._host_device)
            for table in self._host_tables:
                _cuda_host_unregister(table)
                if table._sglang_cuda_host_registered_ranges:
                    raise RuntimeError("native MTP host KV registration was not released")
        self._host_closed = True

    def __del__(self):
        # Also covers an outer QSA constructor failing after this pool was built.
        try:
            self.release_host_kv()
        except BaseException:
            import logging
            logging.getLogger(__name__).exception("native MTP host KV finalizer failed")


def release_native_mtp_host_kv_pools():
    for pool in list(_NATIVE_MTP_HOST_KV_POOLS):
        pool.release_host_kv()  # Never calls global PLE cleanup or touches target tables.
