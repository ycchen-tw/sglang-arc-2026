"""Sparse-attention backend for Qwen4-Exp models with an indexer.

The backend is installed as the full-attention side of Qwen4-Exp's hybrid backend;
linear-attention layers continue to use GDN.
"""

from __future__ import annotations

import logging
import math
import os
from copy import copy
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.qsa.config import (
    is_qwen_qsa,
    parse_qsa_profile,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.metadata import (
    QSAHiSparseDecode,
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
    compressed_decode_view,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_nvfp4_expand_triton,
    qwen_sparse_nvfp4_gather4_triton,
    qwen_sparse_gqa_decode_selected_triton,
    qwen_sparse_gqa_decode_selected_fp4_triton,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.common import is_sm100_supported

logger = logging.getLogger(__name__)
from sglang.srt.environ import envs


_TRTLLM_SPARSE_PAGE_SIZE = 64
_QSA_DIRECT_DECODE = envs.SGLANG_QSA_DIRECT_DECODE.get()
_QSA_DIRECT_MIN_ROWS = envs.SGLANG_QSA_DIRECT_MIN_ROWS.get()
_ARC_QSA_FP4_FUSED = os.environ.get("ARC_QSA_FP4_FUSED", "1") == "1"
# NVFP4 target KV under QSA: the pool stays packed FP4 (E2M1 + block-16 E4M3
# scales); decode gathers the selection as bytes and XQA reads FP4 through
# kv_cache_sf, prefix prefill expands only the request's context transiently.
_ARC_QSA_NVFP4 = os.environ.get("ARC_QSA_NVFP4") == "1"
# The sm_100 trtllm-gen FP4 decode kernel reads the *value* scale plane in
# groups of this many tokens, with the channel blocks interleaved inside each
# group; the key plane it reads linearly. Derived and gated in
# work/qwen38fn_eval/probe_trtllm_fp4_kv_sm100.py.
_SF_TOKEN_GROUP = 4

# One fused four-plane gather instead of two same-shaped passes over the same
# address chain. Byte-for-byte identical output, so "split" exists only as a
# kill switch and for A/B, not because the two differ numerically.
_ARC_QSA_NVFP4_GATHER = os.environ.get("ARC_QSA_NVFP4_GATHER", "fused")
if _ARC_QSA_NVFP4_GATHER not in ("fused", "split"):
    raise ValueError(
        f"ARC_QSA_NVFP4_GATHER must be 'fused' or 'split', got "
        f"{_ARC_QSA_NVFP4_GATHER!r}"
    )
# Dtype the prefix-chunk expansion feeds the _ck kernel. FP8 halves what that
# kernel re-reads per query row (the dominant cost of chunk prefill); BF16
# exists so the downcast's quantization cost can be measured against it.
_ARC_QSA_NVFP4_PREFILL = os.environ.get("ARC_QSA_NVFP4_PREFILL", "fp8")


def _pool_kv_quant_name(pool) -> Optional[str]:
    if pool is None:
        return None
    method = pool.get_kv_cache_quant_method()
    return None if method is None else method.name


def _nvfp4_dequantize_tokens(pool, layer_id: int, slots: torch.Tensor, dtype: torch.dtype):
    """Gather packed NVFP4 K/V rows at ``slots`` and expand them to ``dtype``."""
    from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

    k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer_id)
    method = pool.get_kv_cache_quant_method()
    k_scale = method.k_scales_gpu[layer_id : layer_id + 1]
    v_scale = method.v_scales_gpu[layer_id : layer_id + 1]
    k = NVFP4KVQuantizeUtil.dequantize(
        k_fp4.index_select(0, slots), k_sf.index_select(0, slots), k_scale, dtype=dtype
    )
    v = NVFP4KVQuantizeUtil.dequantize(
        v_fp4.index_select(0, slots), v_sf.index_select(0, slots), v_scale, dtype=dtype
    )
    return k, v


@lru_cache(maxsize=1)
def _resolve_trtllm_sparse_decode():
    """FlashInfer paged decode for the post-gather sparse attention;
    the FA4 varlen fallback runs a prefill-shaped kernel at decode row counts."""
    from sglang.srt.utils import is_sm100_supported, is_sm120

    # This path is numerically validated on SM100 and SM120. Do not widen it
    # to every SM12x device: it silently corrupts long-context decode on
    # SM121/GB10.
    sm120 = is_sm120()
    mode = os.environ.get("ARC_QSA_SM120_DECODE", "xqa")
    if mode not in ("fa2", "xqa"):
        raise ValueError(f"Invalid ARC_QSA_SM120_DECODE={mode!r}")
    if sm120:
        # Upstream turns trtllm-gen (XQA on SM120) decode on by default; the
        # knob only keeps the FA2 arm reachable for the SM120 A/B, so "xqa"
        # is the default and the shipped behaviour matches upstream.
        logger.info("QSA_SM120_KERNEL=%s (selection and indexer unchanged)", mode)
        if mode == "fa2":
            return None
    elif "ARC_QSA_SM120_DECODE" in os.environ:
        raise RuntimeError("ARC_QSA_SM120_DECODE is only meaningful on SM120")
    if not (is_sm100_supported() or sm120):
        return None
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError:
        return None
    return trtllm_batch_decode_with_kv_cache


@lru_cache(maxsize=1)
def _resolve_flash_attn_varlen_func():
    from sglang.srt.utils import is_sm121

    if is_sm121():
        from sglang.kernels.ops.attention import (
            qwen38_qsa_sm121_varlen,
        )

        return qwen38_qsa_sm121_varlen
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func as cute_varlen_func

        def flash_attn_varlen_func(*args, **kwargs):
            output = cute_varlen_func(*args, **kwargs)
            # The cute interface returns (out, lse); lse is None here.
            return output[0] if isinstance(output, tuple) else output

        return flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "QSA decode requires flash_attn (FA2) or flash-attn-4 "
            "(FA4 cute) for its packed varlen fallback."
        ) from exc


class QwenSparseAttnMetadata(msgspec.Struct, frozen=True):
    """Per-forward metadata consumed by core sparse attention."""

    sequence_lengths: torch.Tensor
    token_to_batch_idx: torch.Tensor
    token_slot_table: torch.Tensor
    indexer_metadata: QSAIndexerMetadata
    row_req_pool_indices: Optional[torch.Tensor] = None
    is_cuda_graph: bool = False
    fa2_valid_counts: Optional[torch.Tensor] = None
    fa2_cu_seqlens_k: Optional[torch.Tensor] = None
    fa2_cu_seqlens_q: Optional[torch.Tensor] = None


class QSAMTPSharedSparseIndices:
    """Draft-extend top-k reused by one MTP iteration's decode steps,
    causally valid because a draft moves at most speculative_num_steps positions.
    Row ``num_requests`` is the trash row for padded/degenerate requests."""

    def __init__(
        self, *, layer_ids, num_requests, token_topk, tail_width, device
    ) -> None:
        self.layer_slots = {int(l): i for i, l in enumerate(sorted(layer_ids))}
        self.tail_width = tail_width
        self.trash_row = num_requests
        # Logical index 0 keeps never-captured rows (graph warmup dummies)
        # attending exactly the first token instead of an empty/invalid set.
        self.indices = torch.zeros(
            (len(self.layer_slots), num_requests + 1, token_topk + tail_width),
            dtype=torch.int32,
            device=device,
        )
        self.captured_len = torch.ones(
            (len(self.layer_slots), num_requests + 1),
            dtype=torch.int32,
            device=device,
        )
        self._tail_offsets = torch.arange(tail_width, device=device)

    def capture(
        self,
        topk_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        captured_lens: torch.Tensor,
        layer_id: int,
    ) -> None:
        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.to(torch.long)
        self.indices[slot, :, : topk_indices.shape[1]].index_copy_(
            0, rows, topk_indices.to(self.indices.dtype)
        )
        self.captured_len[slot].index_copy_(0, rows, captured_lens.to(torch.int32))

    def lookup(
        self,
        req_pool_indices: torch.Tensor,
        current_positions: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Frozen selection plus a tail of positions drafted since the capture;
        tail slots past current_position hold -1, which downstream drops."""
        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.to(torch.long)
        out = self.indices[slot, rows]
        base = self.captured_len[slot, rows].to(torch.int64)
        tail = base.unsqueeze(1) + self._tail_offsets.unsqueeze(0)
        valid = tail <= current_positions.to(torch.int64).unsqueeze(1)
        out[:, out.shape[1] - self.tail_width :] = torch.where(valid, tail, -1).to(
            out.dtype
        )
        return out


class QwenSparseAttnBackend(AttentionBackend):
    """QSA backend using trtllm-gen decode with a packed FA2/FA4 fallback."""

    # Every seq_lens_cpu read here has a device fallback (one readback);
    # the graphed decode path never reads it, so opting out is safe.
    needs_cpu_seq_lens: bool = False

    def __init__(self, runner=None) -> None:
        self.runner = runner
        self.token_to_kv_pool = getattr(runner, "token_to_kv_pool", None)
        self.device = getattr(runner, "device", None)
        self.arc_nvfp4 = _pool_kv_quant_name(self.token_to_kv_pool) == "nvfp4"
        self.arc_nvfp4_sm100 = self.arc_nvfp4 and is_sm100_supported()
        if self.arc_nvfp4:
            if not _ARC_QSA_NVFP4:
                raise RuntimeError(
                    "NVFP4 KV pool under QSA requires ARC_QSA_NVFP4=1; the pool "
                    "budget and prefill access were configured without it"
                )
            if _resolve_trtllm_sparse_decode() is None:
                raise RuntimeError(
                    "NVFP4 QSA decode needs the trtllm-gen/XQA paged decode "
                    "kernel and _resolve_trtllm_sparse_decode() returned None "
                    "(device is not SM100/SM120, or flashinfer.decode is not "
                    "importable)"
                )
            if _ARC_QSA_NVFP4_PREFILL not in ("fp8", "bf16"):
                raise RuntimeError(
                    "ARC_QSA_NVFP4_PREFILL must be fp8 or bf16, got "
                    + _ARC_QSA_NVFP4_PREFILL
                )
            logger.info(
                "ARC_QSA_NVFP4=1: packed FP4 gather + %s kv_cache_sf decode; "
                "prefix prefill uses the fused NVFP4 expansion to %s",
                "sm100 trtllm-gen" if self.arc_nvfp4_sm100 else "XQA",
                _ARC_QSA_NVFP4_PREFILL,
            )
        model_config = getattr(runner, "model_config", None)
        config = getattr(model_config, "hf_text_config", None)
        if config is None:
            config = getattr(model_config, "hf_config", None)
        self.qsa_profile = parse_qsa_profile(config)
        self.max_context_len = int(getattr(model_config, "context_len", 0))
        self.compress_ratio = (
            self.qsa_profile.compress_ratio
            if self.qsa_profile is not None
            else int(getattr(config, "indexer_compress_ratio", 4))
        )
        req_pool = getattr(runner, "req_to_token_pool", None)
        self.req_to_token = getattr(req_pool, "req_to_token", None)
        self.req_to_token_pool = req_pool
        self.forward_metadata: Optional[QwenSparseAttnMetadata] = None
        self._cuda_graph_metadata: Dict[
            Tuple[ForwardMode, int], QwenSparseAttnMetadata
        ] = {}
        self._cuda_graph_max_tokens = 0
        self._fa2_scratch: Dict[
            Tuple[int, int, torch.dtype, torch.device],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._graph_seq_lens = None
        self._graph_token_to_batch = None
        self._graph_cu_seqlens_q = None
        self._graph_fa2_valid_counts = None
        self._graph_fa2_cu_seqlens_k = None
        self._graph_write_locs = None
        self._graph_compressed_page_table = None
        self._graph_compressed_lengths = None
        self._graph_prefix_lengths = None
        self._graph_dummy_token_slot_table = None
        self._graph_dummy_out_cache_loc = None
        self._graph_row_req_pool_indices = None
        self._trtllm_sparse_tables = {}
        self._mtp_shared_sparse_indices = None
        self._trtllm_workspace = None
        self._graph_extend_lens = None
        self._graph_extend_lens_pin = None
        # QSA HiSparse offload (arc-forge h04). ModelRunner builds the
        # coordinator in alloc_memory_pool, before the attention backends.
        self.hisparse_coordinator = (
            None if runner is None else runner.hisparse_coordinator
        )
        if self.hisparse_coordinator is not None:
            self._validate_hisparse()

    def _validate_hisparse(self) -> None:
        """HiSparse decode reads KV through coordinator device rows, which only
        the trtllm-gen/XQA gather is wired for; every other decode path indexes
        the device pool with logical req_to_token slots."""
        if self.arc_nvfp4:
            raise NotImplementedError(
                "QSA HiSparse offload supports the FP8 KV pool only (M1a)"
            )
        if _QSA_DIRECT_DECODE:
            raise RuntimeError(
                "SGLANG_QSA_DIRECT_DECODE reads req_to_token slots; turn it off "
                "with HiSparse offload"
            )
        if _resolve_trtllm_sparse_decode() is None:
            raise RuntimeError(
                "QSA HiSparse offload needs the trtllm-gen/XQA paged decode "
                "kernel; _resolve_trtllm_sparse_decode() returned None"
            )
        if self.runner.is_draft_worker:
            raise NotImplementedError(
                "QSA HiSparse offload does not support speculative draft workers"
            )
        if torch.device(self.device).type != "cuda":
            raise RuntimeError(f"QSA HiSparse offload needs CUDA, got {self.device}")
        self._check_hisparse_buffers(1)
        coord = self.hisparse_coordinator
        if int(coord.rows_limit.max()) > coord.identity_rows.shape[1]:
            raise RuntimeError(
                "HiSparse rows_limit exceeds the identity table: "
                f"{int(coord.rows_limit.max())} > {coord.identity_rows.shape[1]}"
            )

    def _check_hisparse_buffers(self, rows: int) -> None:
        """The coordinator's persistent decode buffers must hold ``rows`` rows."""
        coord = self.hisparse_coordinator
        for name, tensor in (
            ("seq_blocks", coord.seq_blocks),
            ("newest_dev", coord.newest_dev),
            ("zeros_i32", coord.zeros_i32),
            ("rows_limit", coord.rows_limit),
        ):
            if tensor.dtype != torch.int32 or tensor.ndim != 1 or tensor.shape[0] < rows:
                raise RuntimeError(
                    f"HiSparse coordinator {name} must be int32 [>= {rows}], "
                    f"got {tensor.dtype} {tuple(tensor.shape)}"
                )
        width = self.token_to_kv_pool.qsa_token_topk + self.compress_ratio - 1
        buffer = coord.rows
        if (
            buffer.dtype != torch.int32
            or buffer.ndim != 2
            or buffer.shape[0] < rows
            or buffer.shape[1] != width
            or buffer.stride(1) != 1
        ):
            raise RuntimeError(
                f"HiSparse coordinator rows must be int32 [>= {rows}, {width}], "
                f"got {buffer.dtype} {tuple(buffer.shape)}"
            )
        identity = coord.identity_rows
        if identity.dtype != torch.int32 or identity.ndim != 2 or identity.shape[0] != 1:
            raise RuntimeError(
                "HiSparse identity_rows must be int32 [1, device rows], got "
                f"{identity.dtype} {tuple(identity.shape)}"
            )

    def _hisparse_decode_state(
        self, sequence_lengths: torch.Tensor, req_pool_indices: torch.Tensor
    ) -> QSAHiSparseDecode:
        """Refresh the coordinator's per-row decode buffers for these rows (on
        the forward stream) and view them for this forward."""
        coord = self.hisparse_coordinator
        bs = sequence_lengths.shape[0]
        state = QSAHiSparseDecode(
            coord=coord,
            seq_blocks=coord.seq_blocks[:bs],
            newest_dev=coord.newest_dev[:bs],
            rows=coord.rows[:bs],
        )
        if (
            state.seq_blocks.shape[0] != bs
            or state.newest_dev.shape[0] != bs
            or state.rows.shape[0] != bs
        ):
            raise RuntimeError(f"HiSparse coordinator buffers hold fewer than {bs} rows")
        coord.prepare_decode_meta(sequence_lengths, req_pool_indices, bs)
        return state

    def _refresh_hisparse_graph_state(self, metadata, bs: int) -> None:
        hisparse = metadata.indexer_metadata.hisparse
        if (hisparse is None) != (self.hisparse_coordinator is None):
            raise RuntimeError(
                "QSA CUDA graph metadata disagrees with the HiSparse coordinator"
            )
        if hisparse is None:
            return
        self.hisparse_coordinator.prepare_decode_meta(
            metadata.sequence_lengths, metadata.row_req_pool_indices, bs
        )

    def _store_kv(self, layer, loc, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.arc_nvfp4:
            # HybridLinearKVPool.set_kv_buffer defaults k/v scale to the Python
            # float 1.0, and MHATokenToKVPool._quantized_scales only substitutes
            # the quant method's per-layer global scales when it is given None.
            # A float would also be materialised host-side inside graph capture.
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, None, None)
            return
        self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)

    @staticmethod
    def _is_speculative_paged_mode(forward_mode) -> bool:
        if forward_mode is None:
            return False
        return forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()

    def _require_chain_speculation(self, forward_mode, spec_info) -> None:
        if forward_mode is None or not forward_mode.is_target_verify():
            return
        if int(getattr(spec_info, "topk", 1) or 1) != 1:
            raise NotImplementedError(
                "Qwen QSA target verification supports only speculative_eagle_topk=1"
            )
        draft_tokens = int(getattr(spec_info, "draft_token_num", 0) or 0)
        if draft_tokens > self.compress_ratio:
            # The pending-group ring keys state by position % ratio; a verify
            # window wider than the ratio would collide within one forward.
            raise NotImplementedError(
                "Qwen QSA requires speculative_num_draft_tokens <= the QSA "
                f"compress ratio ({self.compress_ratio}): the pending "
                f"index-key ring holds one group; got {draft_tokens}"
            )

    @staticmethod
    def _speculative_max_row_length(forward_batch, sequence_lengths) -> int:
        """Sync-free host-side over-bound for the token-slot gather width:
        request length plus the draft window covers every speculative row."""
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
            logger.warning_once(
                "QSA speculative metadata without CPU request lengths: the "
                "token-slot bound reads them back once per forward"
            )
            return max(1, int(sequence_lengths.max()))
        spec_info = forward_batch.spec_info
        # Target verify exposes ``draft_token_num`` while draft-extend exposes
        # ``num_tokens_per_req``. Both modes use this gather-width bound.
        draft_window = int(
            getattr(
                spec_info,
                "draft_token_num",
                getattr(spec_info, "num_tokens_per_req", 0),
            )
            or 0
        )
        return max(1, int(seq_lens_cpu.max()) + draft_window)

    @staticmethod
    def _speculative_row_to_request(forward_batch, num_rows: int) -> torch.Tensor:
        batch_size = int(forward_batch.req_pool_indices.numel())
        if batch_size == 0:
            if num_rows == 0:
                return torch.zeros(
                    0,
                    dtype=torch.long,
                    device=forward_batch.req_pool_indices.device,
                )
            raise ValueError(
                "QSA speculative query rows cannot be mapped to an empty batch: "
                f"rows={num_rows}"
            )
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None:
            # DP token padding rows belong to no request; alias them to request row 0,
            # as the CUDA-graph layout does, so gathers stay in bounds.
            repeats = extend_seq_lens[:batch_size].to(dtype=torch.long)
            real_rows = int(repeats.sum().item())
            if real_rows > num_rows:
                raise ValueError(
                    "QSA speculative query rows are fewer than the extend "
                    f"mapping: rows={num_rows}, mapped={real_rows}"
                )
            row_to_request = torch.repeat_interleave(
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=forward_batch.req_pool_indices.device,
                ),
                repeats,
            )
            padding = num_rows - real_rows
            if padding:
                row_to_request = torch.cat(
                    [
                        row_to_request,
                        row_to_request.new_zeros(padding),
                    ]
                )
            return row_to_request
        if num_rows % batch_size != 0:
            raise ValueError(
                "QSA speculative query rows cannot be mapped to requests: "
                f"rows={num_rows}, batch={batch_size}"
            )
        return torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.req_pool_indices.device,
        ).repeat_interleave(num_rows // batch_size)

    @staticmethod
    def _as_cpu_int_tensor(values, size: int) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values[:size].detach().cpu().to(torch.int32)
        return torch.tensor(values[:size], dtype=torch.int32)

    @classmethod
    def _graph_speculative_layout(
        cls,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu,
        forward_mode,
        spec_info,
        num_padding: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_lengths = cls._as_cpu_int_tensor(seq_lens_cpu, bs)
        if forward_mode.is_target_verify():
            extend_len = int(spec_info.draft_token_num)
            extend_lengths = torch.full((bs,), extend_len, dtype=torch.int32)
        else:
            extend_lengths_cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if extend_lengths_cpu is not None:
                extend_lengths = cls._as_cpu_int_tensor(extend_lengths_cpu, bs)
                if extend_lengths.numel() < bs:
                    extend_lengths = torch.cat(
                        [
                            extend_lengths,
                            torch.zeros(bs - extend_lengths.numel(), dtype=torch.int32),
                        ]
                    )
            else:
                extend_lengths = torch.full(
                    (bs,), num_tokens // max(bs, 1), dtype=torch.int32
                )
        num_padding = max(0, min(int(num_padding), bs))
        if num_padding:
            extend_lengths = extend_lengths.clone()
            extend_lengths[bs - num_padding :] = 0
        effective_lengths = (
            base_lengths + extend_lengths
            if forward_mode.is_target_verify()
            else base_lengths
        )

        row_lengths = []
        row_prefix_lengths = []
        for seq_len, extend_len in zip(
            effective_lengths.tolist(), extend_lengths.tolist()
        ):
            prefix_len = max(int(seq_len) - int(extend_len), 0)
            row_lengths.append(
                torch.arange(
                    prefix_len + 1,
                    prefix_len + int(extend_len) + 1,
                    dtype=torch.int32,
                ).clamp_max(int(seq_len))
            )
            row_prefix_lengths.append(
                torch.full((int(extend_len),), prefix_len, dtype=torch.int32)
            )
        row_lengths = (
            torch.cat(row_lengths) if row_lengths else torch.empty(0, dtype=torch.int32)
        )
        row_prefix_lengths = (
            torch.cat(row_prefix_lengths)
            if row_prefix_lengths
            else torch.empty(0, dtype=torch.int32)
        )
        actual_rows = row_lengths.numel()
        if actual_rows > num_tokens:
            raise ValueError(
                "QSA CUDA graph speculative layout has inconsistent token count: "
                f"capacity={num_tokens}, actual={actual_rows}"
            )
        repeats = extend_lengths.to(device=req_pool_indices.device, dtype=torch.long)
        row_req_pool_indices = torch.repeat_interleave(req_pool_indices[:bs], repeats)
        # A draft-extend replay may hold fewer accepted rows than the captured shape;
        # the runner packs real rows first, so give the tail inert metadata.
        padding = num_tokens - actual_rows
        if padding:
            row_lengths = torch.cat(
                [row_lengths, torch.ones(padding, dtype=torch.int32)]
            )
            row_prefix_lengths = torch.cat(
                [row_prefix_lengths, torch.zeros(padding, dtype=torch.int32)]
            )
            dummy_req = (
                req_pool_indices[0]
                if req_pool_indices.numel()
                else torch.zeros((), dtype=torch.int32, device=req_pool_indices.device)
            )
            row_req_pool_indices = torch.cat(
                [
                    row_req_pool_indices,
                    dummy_req.to(row_req_pool_indices.dtype).expand(padding),
                ]
            )
        return row_lengths, row_req_pool_indices, row_prefix_lengths

    def _empty_metadata(self, forward_batch) -> QwenSparseAttnMetadata:
        """Well-formed zero-row metadata for IDLE/zero-request forwards."""

        device = forward_batch.seq_lens.device
        sequence_lengths = torch.zeros(0, dtype=torch.int32, device=device)
        token_to_batch_idx = torch.zeros(0, dtype=torch.int32, device=device)
        token_slot_table = torch.zeros((0, 1), dtype=torch.int32, device=device)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            out_cache_loc = torch.zeros(0, dtype=torch.int64, device=device)
        indexer_metadata = QSAIndexerMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=self.token_to_kv_pool,
            compress_ratio=self.token_to_kv_pool.qsa_compress_ratio,
            block_topk=self.token_to_kv_pool.qsa_block_topk,
        )
        return QwenSparseAttnMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            indexer_metadata=indexer_metadata,
            row_req_pool_indices=torch.zeros(0, dtype=torch.int32, device=device),
        )

    @staticmethod
    def _qsa_write_plan(
        *,
        token_slot_table,
        start_blocks,
        end_blocks,
        capacity,
        compress_ratio,
        row_token_starts=None,
        prefix_lens=None,
    ):
        """Compact per-row block ranges into ``capacity`` write entries,
        a shape-derived bound (no sync); padding writes the inert reserved slot 0."""
        device = token_slot_table.device
        # The table width is a host-side bound;
        # assert on device so a short table fails loudly without a sync.
        torch._assert_async(
            (end_blocks * compress_ratio <= token_slot_table.shape[1]).all()
        )
        counts = (end_blocks - start_blocks).clamp_min(0)
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        entries = torch.arange(capacity, dtype=torch.long, device=device)
        # Which row owns each compacted entry, and its ordinal within the row.
        rows = torch.searchsorted(ends, entries, right=True)
        valid = entries < ends[-1] if counts.numel() else entries < 0
        rows = torch.where(valid, rows.clamp_max(max(counts.numel() - 1, 0)), 0)
        blocks = torch.where(valid, start_blocks[rows] + entries - starts[rows], 0)
        write_locs = torch.where(
            valid,
            token_slot_table[rows, blocks * compress_ratio].long() // compress_ratio,
            torch.zeros_like(blocks),
        ).to(torch.int32)
        group_end_positions = blocks * compress_ratio + (compress_ratio - 1)
        member_rows = None
        if row_token_starts is not None:
            # member_rows index this forward's packed token rows;
            # the chunk is group-aligned, so a group starts at block * ratio - prefix.
            member_rows = torch.where(
                valid,
                row_token_starts[rows] + blocks * compress_ratio - prefix_lens[rows],
                torch.zeros_like(blocks),
            )
        return write_locs, group_end_positions, rows, member_rows

    def _qsa_build_write_plan(
        self,
        *,
        forward_batch,
        speculative_paged: bool,
        token_slot_table,
        sequence_lengths,
    ):
        ratio = self.token_to_kv_pool.qsa_compress_ratio
        lengths = sequence_lengths.long()
        end_blocks = lengths // ratio
        if forward_batch.forward_mode.is_decode() or speculative_paged:
            # A paged row compresses exactly the block its length completes;
            # its members come from the per-request pending ring.
            start_blocks = end_blocks - (lengths % ratio == 0).long()
            return self._qsa_write_plan(
                token_slot_table=token_slot_table,
                start_blocks=start_blocks,
                end_blocks=end_blocks,
                capacity=int(lengths.numel()),
                compress_ratio=ratio,
            )
        extend_lens = forward_batch.extend_seq_lens
        if extend_lens is None:
            raise ValueError("QSA extend write plan requires extend_seq_lens")
        extend_lens = extend_lens.long()[: lengths.numel()]
        prefix_lens = (lengths - extend_lens).clamp_min(0)
        # Prefix sharing is page-granular and the page is a ratio
        # multiple, so a matched prefix always covers whole groups. A
        # misaligned prefix would leave a shared group half-written.
        torch._assert_async((prefix_lens % ratio == 0).all())
        # Each row spans at most ceil(extend_len / ratio) blocks, so the
        # token count and row count bound the plan without a sync.
        capacity = int(forward_batch.input_ids.numel()) // ratio + int(lengths.numel())
        row_token_starts = torch.cumsum(extend_lens, 0) - extend_lens
        return self._qsa_write_plan(
            token_slot_table=token_slot_table,
            start_blocks=prefix_lens // ratio,
            end_blocks=end_blocks,
            capacity=capacity,
            compress_ratio=ratio,
            row_token_starts=row_token_starts,
            prefix_lens=prefix_lens,
        )

    def _metadata_from_forward_batch(self, forward_batch) -> QwenSparseAttnMetadata:
        self._require_chain_speculation(
            forward_batch.forward_mode,
            getattr(forward_batch, "spec_info", None),
        )
        if self.device is None:
            self.device = forward_batch.seq_lens.device
        if not self.max_context_len:
            self.max_context_len = self.req_to_token.shape[1]
        if forward_batch.forward_mode.is_idle() or forward_batch.seq_lens.numel() == 0:
            # DP idle forwards reach metadata init even though layers skip attention;
            # so do the zero-row DECODE steps the MTP wrapper makes of them.
            return self._empty_metadata(forward_batch)
        original_mode = getattr(forward_batch, "_original_forward_mode", None)
        if original_mode is not None and self._is_speculative_paged_mode(original_mode):
            # DP MAX_LEN pseudo-extend sets extend_seq_lens == 1 per request,
            # losing the draft fan-out this mapping needs.
            raise ValueError(
                "QSA cannot build metadata for DP MAX_LEN pseudo-extend of "
                f"speculative mode {original_mode}: token rows would be "
                "mis-mapped to requests"
            )
        speculative_paged = self._is_speculative_paged_mode(forward_batch.forward_mode)
        if self.hisparse_coordinator is not None and (
            speculative_paged or self.should_reuse_mtp_sparse_indices(forward_batch)
        ):
            raise NotImplementedError(
                "QSA HiSparse offload does not support speculative decoding"
            )
        if speculative_paged:
            logical_positions = forward_batch.positions
            if logical_positions.ndim == 2:
                logical_positions = logical_positions[0]
            logical_positions = logical_positions.flatten()
            sequence_lengths = (logical_positions + 1).to(torch.int32)
            row_to_request = self._speculative_row_to_request(
                forward_batch, logical_positions.numel()
            )
            row_req_pool_indices = forward_batch.req_pool_indices.index_select(
                0, row_to_request
            )
            max_length = self._speculative_max_row_length(
                forward_batch, sequence_lengths
            )
            token_slot_table = self.req_to_token[
                row_req_pool_indices.long(), :max_length
            ].to(torch.int32)
            token_to_batch_idx = torch.arange(
                sequence_lengths.numel(),
                device=sequence_lengths.device,
                dtype=torch.int32,
            )
        else:
            sequence_lengths = forward_batch.seq_lens.to(torch.int32)
            batch_size = sequence_lengths.numel()
            if forward_batch.seq_lens_cpu is not None:
                max_length = int(forward_batch.seq_lens_cpu[:batch_size].max())
            else:
                max_length = int(sequence_lengths.max())
            row_req_pool_indices = forward_batch.req_pool_indices[:batch_size]
            token_slot_table = self.req_to_token[
                row_req_pool_indices.long(), :max_length
            ].to(torch.int32)
            positions = forward_batch.positions
            num_position_tokens = (
                positions.shape[-1] if positions.ndim == 2 else positions.numel()
            )
            if forward_batch.forward_mode.is_decode():
                if num_position_tokens != batch_size:
                    raise ValueError(
                        "QSA decode requires exactly one query row per request: "
                        f"rows={num_position_tokens}, batch={batch_size}"
                    )
                token_to_batch_idx = torch.arange(
                    batch_size,
                    device=sequence_lengths.device,
                    dtype=torch.int32,
                )
            else:
                extend_seq_lens = forward_batch.extend_seq_lens
                if extend_seq_lens is None:
                    raise ValueError("QSA extend metadata requires extend_seq_lens")
                token_to_batch_idx = torch.repeat_interleave(
                    torch.arange(
                        batch_size,
                        device=sequence_lengths.device,
                        dtype=torch.int32,
                    ),
                    extend_seq_lens.to(torch.long),
                )
                # More mapped rows than physical would index past them;
                # fewer is fine (DP MAX_LEN padding is trimmed downstream).
                if token_to_batch_idx.numel() > num_position_tokens:
                    raise ValueError(
                        "QSA extend request mapping exceeds token rows: "
                        f"mapping={token_to_batch_idx.numel()}, "
                        f"positions={num_position_tokens}"
                    )
        decode_page_table = None
        decode_lengths = None
        decode_logical_positions = None
        pending_ring_slots = None
        compress_group_ring_locs = None
        extend_rope_matrix = None
        write_locs, group_positions, group_sequence_ids, group_member_rows = (
            self._qsa_build_write_plan(
                forward_batch=forward_batch,
                speculative_paged=speculative_paged,
                token_slot_table=token_slot_table,
                sequence_lengths=sequence_lengths,
            )
        )
        decode_like = speculative_paged or forward_batch.forward_mode.is_decode()
        if decode_like:
            decode_logical_positions = (
                logical_positions.to(torch.int32)
                if speculative_paged
                else sequence_lengths - 1
            )
            ring_logical_positions = decode_logical_positions
        else:
            extend_positions = forward_batch.positions
            if extend_positions.ndim == 2:
                extend_positions = extend_positions[0]
            ring_logical_positions = extend_positions.flatten()[
                : token_to_batch_idx.numel()
            ]
        if not self.should_reuse_mtp_sparse_indices(forward_batch):
            if decode_like:
                pool = self.token_to_kv_pool
                decode_page_table, decode_lengths = compressed_decode_view(
                    compressed_page_size=pool.qsa_compressed_page_size,
                    compress_ratio=pool.qsa_compress_ratio,
                    sequence_lengths=sequence_lengths,
                    token_slot_table=token_slot_table,
                )
            pending_ring_slots = build_pending_ring_slots(
                token_to_batch_idx=token_to_batch_idx,
                req_pool_indices=row_req_pool_indices,
                sequence_lengths=sequence_lengths,
                logical_positions=ring_logical_positions,
                compress_ratio=self.compress_ratio,
                is_extend=group_member_rows is not None,
            )
            if write_locs.numel():
                if group_member_rows is not None:
                    rope_source = (
                        forward_batch.mrope_positions
                        if forward_batch.mrope_positions is not None
                        else forward_batch.positions
                    )
                    extend_rope_matrix = build_rope_position_matrix(
                        rope_source, token_to_batch_idx.numel()
                    )
                else:
                    compress_group_ring_locs = build_group_ring_slots(
                        req_pool_indices=row_req_pool_indices,
                        group_end_positions=group_positions.long(),
                        sequence_ids=group_sequence_ids.long(),
                        compress_ratio=self.compress_ratio,
                    )
        hisparse = None
        if (
            self.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode()
        ):
            hisparse = self._hisparse_decode_state(
                sequence_lengths, row_req_pool_indices
            )
        indexer_metadata = QSAIndexerMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            out_cache_loc=forward_batch.out_cache_loc,
            token_to_kv_pool=self.token_to_kv_pool,
            compress_ratio=self.token_to_kv_pool.qsa_compress_ratio,
            block_topk=self.token_to_kv_pool.qsa_block_topk,
            req_pool_indices=row_req_pool_indices,
            write_locs=write_locs,
            compress_group_positions=group_positions,
            compress_sequence_ids=group_sequence_ids,
            compress_member_rows=group_member_rows,
            decode_page_table=decode_page_table,
            decode_lengths=decode_lengths,
            decode_logical_positions=decode_logical_positions,
            pending_ring_slots=pending_ring_slots,
            compress_group_ring_locs=compress_group_ring_locs,
            extend_rope_matrix=extend_rope_matrix,
            hisparse=hisparse,
        )
        return QwenSparseAttnMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            indexer_metadata=indexer_metadata,
            row_req_pool_indices=row_req_pool_indices,
        )

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_idle():
            self.forward_metadata = None
            return
        self.forward_metadata = self._metadata_from_forward_batch(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch,
        in_capture: bool = False,
    ):
        if in_capture:
            num_tokens = (
                forward_batch.input_ids.shape[0]
                if self._is_speculative_paged_mode(forward_batch.forward_mode)
                else forward_batch.batch_size
            )
            self._capture_cuda_graph_metadata(
                bs=forward_batch.batch_size,
                num_tokens=num_tokens,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_batch.forward_mode,
                spec_info=forward_batch.spec_info,
            )
        else:
            num_padding = getattr(forward_batch, "num_padding", None)
            self._replay_cuda_graph_metadata(
                bs=forward_batch.batch_size,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_batch.forward_mode,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                num_padding=num_padding if num_padding is not None else 0,
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        if self.device is None:
            raise RuntimeError(
                "QSA backend requires a ModelRunner to initialize CUDA graph state"
            )
        if self.hisparse_coordinator is not None:
            self._check_hisparse_buffers(max_num_tokens)
        self._cuda_graph_max_tokens = max_num_tokens
        max_blocks = math.ceil(self.max_context_len / self.compress_ratio)
        max_pages = max(
            1,
            math.ceil(max_blocks / self.token_to_kv_pool.qsa_compressed_page_size),
        )
        self._graph_seq_lens = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_token_to_batch = torch.arange(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_cu_seqlens_q = torch.arange(
            max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._graph_fa2_valid_counts = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_fa2_cu_seqlens_k = torch.zeros(
            max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._graph_write_locs = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_compressed_page_table = torch.zeros(
            (max_num_tokens, max_pages), dtype=torch.int32, device=self.device
        )
        self._graph_compressed_lengths = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_prefix_lengths = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_dummy_token_slot_table = torch.zeros(
            (max_num_tokens, 1), dtype=torch.int32, device=self.device
        )
        self._graph_dummy_out_cache_loc = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=self.device
        )
        self._graph_row_req_pool_indices = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_logical_positions = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_state_slots = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=self.device
        )
        self._graph_ring_group_locs = torch.zeros(
            (max_num_tokens, self.compress_ratio),
            dtype=torch.int32,
            device=self.device,
        )
        # Two pinned halves: when a replay refills the pinned buffer,
        # the previous async HtoD copy from it may still be queued.
        self._graph_extend_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._graph_extend_lens_pin = [
            torch.zeros(max_bs, dtype=torch.int32, pin_memory=True) for _ in range(2)
        ]
        self._extend_lens_pin_idx = 0

    def _capture_cuda_graph_metadata(
        self,
        *,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info,
    ) -> None:
        self._require_chain_speculation(forward_mode, spec_info)
        if self.token_to_kv_pool is None:
            self.token_to_kv_pool = getattr(self.runner, "token_to_kv_pool", None)
        if self.req_to_token is None:
            req_pool = getattr(self.runner, "req_to_token_pool", None)
            self.req_to_token = getattr(req_pool, "req_to_token", None)
            self.req_to_token_pool = req_pool
        if self._graph_seq_lens is None or self.token_to_kv_pool is None:
            raise RuntimeError("QSA CUDA graph state is not initialized")
        pool = self.token_to_kv_pool
        speculative_paged = self._is_speculative_paged_mode(forward_mode)
        if self.hisparse_coordinator is not None and not forward_mode.is_decode():
            raise NotImplementedError(
                f"QSA HiSparse offload captures DECODE graphs only, got {forward_mode}"
            )
        metadata_rows = num_tokens if speculative_paged else bs
        if speculative_paged:
            row_lengths, row_req_pool_indices, row_prefix_lengths = (
                self._graph_speculative_layout(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens[:bs].detach().cpu(),
                    forward_mode,
                    spec_info,
                    num_padding=bs,
                )
            )
            self._graph_prefix_lengths[:metadata_rows].copy_(
                row_prefix_lengths.to(device=self.device)
            )
            self._graph_seq_lens[:metadata_rows].copy_(
                row_lengths.to(device=self.device)
            )
            self._graph_row_req_pool_indices[:metadata_rows].copy_(
                row_req_pool_indices.to(torch.int32)
            )
        else:
            self._graph_seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))
            self._graph_row_req_pool_indices[:bs].copy_(
                req_pool_indices[:bs].to(torch.int32)
            )
            self._graph_prefix_lengths[:bs].copy_(
                (seq_lens[:bs] - 1).clamp_min(0).to(torch.int32)
            )
        hisparse = None
        if self.hisparse_coordinator is not None:
            hisparse = self._hisparse_decode_state(
                self._graph_seq_lens[:bs], self._graph_row_req_pool_indices[:bs]
            )
        indexer_metadata = QSAIndexerMetadata(
            sequence_lengths=self._graph_seq_lens[:metadata_rows],
            token_to_batch_idx=self._graph_token_to_batch[:metadata_rows],
            token_slot_table=self._graph_dummy_token_slot_table[:metadata_rows],
            out_cache_loc=self._graph_dummy_out_cache_loc[:metadata_rows],
            token_to_kv_pool=pool,
            compress_ratio=pool.qsa_compress_ratio,
            block_topk=pool.qsa_block_topk,
            req_pool_indices=self._graph_row_req_pool_indices[:metadata_rows],
            is_cuda_graph=True,
            graph_write_locs=self._graph_write_locs[:metadata_rows],
            graph_compressed_page_table=self._graph_compressed_page_table[
                :metadata_rows
            ],
            graph_compressed_lengths=self._graph_compressed_lengths[:metadata_rows],
            graph_prefix_lengths=self._graph_prefix_lengths[:metadata_rows],
            decode_logical_positions=self._graph_logical_positions[:metadata_rows],
            pending_ring_slots=self._graph_state_slots[:metadata_rows],
            graph_ring_group_locs=self._graph_ring_group_locs[:metadata_rows],
            hisparse=hisparse,
        )
        metadata = QwenSparseAttnMetadata(
            sequence_lengths=self._graph_seq_lens[:metadata_rows],
            token_to_batch_idx=self._graph_token_to_batch[:metadata_rows],
            token_slot_table=self._graph_dummy_token_slot_table[:metadata_rows],
            indexer_metadata=indexer_metadata,
            row_req_pool_indices=self._graph_row_req_pool_indices[:metadata_rows],
            is_cuda_graph=True,
            fa2_valid_counts=self._graph_fa2_valid_counts[:metadata_rows],
            fa2_cu_seqlens_k=self._graph_fa2_cu_seqlens_k[: metadata_rows + 1],
            fa2_cu_seqlens_q=self._graph_cu_seqlens_q[: metadata_rows + 1],
        )
        self._cuda_graph_metadata[(forward_mode, bs)] = metadata
        self.forward_metadata = metadata

    def _replay_cuda_graph_metadata(
        self,
        *,
        bs,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        num_padding: int = 0,
    ) -> None:
        self._require_chain_speculation(forward_mode, spec_info)
        metadata = self._cuda_graph_metadata[(forward_mode, bs)]
        # Compressed addressing is arithmetic over req_to_token; pure reads below.
        if self._can_replay_with_gpu_kernels(metadata, seq_lens):
            self._replay_cuda_graph_metadata_gpu(
                metadata,
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                num_padding=num_padding,
            )
            self._refresh_hisparse_graph_state(metadata, bs)
            self.forward_metadata = metadata
            return
        if seq_lens_cpu is None:
            # Host-fallback staging only (non-CUDA pools): an explicit
            # slow-path readback, never the serving path.
            seq_lens_cpu = seq_lens[:bs].cpu()
        if self._is_speculative_paged_mode(forward_mode):
            row_lengths, row_req_pool_indices, row_prefix_lengths = (
                self._graph_speculative_layout(
                    bs,
                    metadata.sequence_lengths.numel(),
                    req_pool_indices,
                    seq_lens_cpu,
                    forward_mode,
                    spec_info,
                    num_padding=num_padding or 0,
                )
            )
            metadata.sequence_lengths.copy_(row_lengths.to(device=self.device))
            metadata.row_req_pool_indices.copy_(row_req_pool_indices.to(torch.int32))
            metadata.indexer_metadata.graph_prefix_lengths.copy_(
                row_prefix_lengths.to(device=self.device)
            )
        else:
            metadata.sequence_lengths.copy_(seq_lens[:bs].to(torch.int32))
            metadata.row_req_pool_indices.copy_(req_pool_indices[:bs].to(torch.int32))
            metadata.indexer_metadata.graph_prefix_lengths.copy_(
                (seq_lens[:bs] - 1).clamp_min(0).to(torch.int32)
            )
        self._update_qsa_cuda_graph_metadata(
            metadata.indexer_metadata, metadata.row_req_pool_indices
        )
        self._refresh_hisparse_graph_state(metadata, bs)
        self.forward_metadata = metadata

    def _can_replay_with_gpu_kernels(self, metadata, seq_lens) -> bool:
        if self.req_to_token is None:
            return False
        from sglang.srt.layers.attention.qsa.graph_metadata import (
            supports_graph_metadata_kernels,
        )

        return seq_lens.is_cuda and supports_graph_metadata_kernels(
            metadata.indexer_metadata.token_to_kv_pool, seq_lens.device
        )

    def _stage_extend_lens(self, spec_info, bs: int, num_tokens: int):

        extend_lens = getattr(spec_info, "extend_seq_lens_tensor", None)
        if extend_lens is not None and extend_lens.numel() >= bs:
            return extend_lens[:bs]
        pin = self._graph_extend_lens_pin[self._extend_lens_pin_idx]
        self._extend_lens_pin_idx = 1 - self._extend_lens_pin_idx
        values = getattr(spec_info, "extend_seq_lens_cpu", None)
        if values is None:
            # Mirrors the legacy fallback: an even per-request share of the
            # captured token capacity.
            pin[:bs].fill_(num_tokens // max(bs, 1))
        elif isinstance(values, torch.Tensor):
            pin[:bs].copy_(values[:bs].to(torch.int32))
        else:
            pin[:bs].copy_(torch.tensor(values[:bs], dtype=torch.int32))
        staged = self._graph_extend_lens
        staged[:bs].copy_(pin[:bs], non_blocking=True)
        return staged[:bs]

    def _replay_cuda_graph_metadata_gpu(
        self,
        metadata,
        *,
        bs,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        num_padding: int = 0,
    ) -> None:
        """Sync-free replay refresh: nothing here may read back to the host,
        since launch_graph_metadata rebuilds every per-row graph buffer on device."""

        from sglang.srt.layers.attention.qsa.graph_metadata import (
            launch_graph_metadata,
        )

        indexer = metadata.indexer_metadata
        pool = indexer.token_to_kv_pool
        speculative = self._is_speculative_paged_mode(forward_mode)
        num_rows = metadata.sequence_lengths.numel() if speculative else bs
        num_padding = max(0, min(int(num_padding or 0), bs))
        if bs <= 0 or num_rows <= 0:
            return

        if speculative:
            if forward_mode.is_target_verify():
                mode = 1
                extend_lens = None
                extend_len = int(spec_info.draft_token_num)
            else:
                mode = 2
                extend_len = 0
                extend_lens = self._stage_extend_lens(spec_info, bs, num_rows)
        else:
            mode = 0
            extend_lens = None
            extend_len = 0
        launch_graph_metadata(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_lens=extend_lens,
            extend_len=extend_len,
            num_padding=num_padding,
            metadata=metadata,
            req_to_token=self.req_to_token,
            pool=pool,
        )

    def _update_qsa_cuda_graph_metadata(
        self, metadata: QSAIndexerMetadata, req_pool_indices: torch.Tensor
    ) -> None:
        """Replay refresh for pools/devices without the graph_metadata kernels."""
        if self.req_to_token is None:
            raise RuntimeError("QSA req_to_token table is not initialized")
        pool = metadata.token_to_kv_pool
        lengths = metadata.sequence_lengths.long()
        req_indices = req_pool_indices.long()
        ratio = metadata.compress_ratio
        full_page = pool.qsa_compressed_page_size * ratio

        current_positions = (lengths - 1).clamp_min(0)
        compressed_lengths = torch.div(lengths, ratio, rounding_mode="floor")
        metadata.graph_compressed_lengths.copy_(compressed_lengths.to(torch.int32))

        # Boundary rows write their group's slot (last raw slot // ratio);
        # every other row keeps the inert reserved slot 0.
        boundary = (lengths % ratio == 0) & (lengths > 0)
        last_locs = self.req_to_token[req_indices, current_positions].long()
        write_locs = torch.where(
            boundary,
            last_locs // ratio,
            torch.zeros_like(lengths),
        )
        metadata.graph_write_locs.copy_(write_locs.to(torch.int32))

        metadata.decode_logical_positions.copy_(current_positions.to(torch.int32))
        metadata.pending_ring_slots.copy_(
            build_pending_ring_slots(
                token_to_batch_idx=metadata.token_to_batch_idx,
                req_pool_indices=req_pool_indices,
                sequence_lengths=metadata.sequence_lengths,
                logical_positions=current_positions,
                compress_ratio=ratio,
                is_extend=False,
            )
        )
        metadata.graph_ring_group_locs.copy_(
            build_group_ring_slots(
                req_pool_indices=req_pool_indices,
                group_end_positions=current_positions,
                sequence_ids=metadata.token_to_batch_idx.long(),
                compress_ratio=ratio,
            ).to(torch.int32)
        )

        # Page-table entries are full-KV page ids from the token-slot rows.
        page_table = metadata.graph_compressed_page_table
        max_pages = page_table.shape[1]
        row_width_pages = self.req_to_token.shape[1] // full_page
        num_pages = min(max_pages, row_width_pages)
        table = (
            self.req_to_token[req_indices, : num_pages * full_page : full_page].long()
            // full_page
        ).clamp_min(0)
        page_table[:, :num_pages].copy_(table.to(torch.int32))

    def get_indexer_metadata(self, layer_id: int, forward_batch):
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)
        assert self.forward_metadata is not None
        return self.forward_metadata.indexer_metadata

    def set_mtp_shared_sparse_indices(self, state) -> None:
        self._mtp_shared_sparse_indices = state

    def should_reuse_mtp_sparse_indices(self, forward_batch) -> bool:
        """Draft decode steps reuse the draft-extend selection."""
        return (
            self._mtp_shared_sparse_indices is not None
            and forward_batch.forward_mode.is_decode()
        )

    def should_capture_mtp_sparse_indices(self, forward_batch) -> bool:
        """Capture on both draft-extend flavors (a DP MAX_LEN rewrite is neither):
        DRAFT_EXTEND_V2, and the draft runner's plain post-prefill EXTEND."""
        if self._mtp_shared_sparse_indices is None:
            return False
        if forward_batch._original_forward_mode is not None:
            return False
        mode = forward_batch.forward_mode
        return mode.is_draft_extend_v2() or mode.is_extend_without_speculative()

    def capture_mtp_sparse_indices(
        self, topk_indices: torch.Tensor, forward_batch, layer_id: int, metadata=None
    ) -> None:
        """Store each request's final accepted row as the iteration seed."""
        if topk_indices.shape[0] == 0:
            return
        if metadata is None:
            metadata = self.get_indexer_metadata(layer_id, forward_batch)
        if metadata.is_cuda_graph or forward_batch.forward_mode.is_draft_extend_v2():
            self._capture_mtp_sparse_indices_from_extend_lens(
                topk_indices, forward_batch, metadata, layer_id
            )
            return
        if metadata.req_pool_indices is None or metadata.req_pool_indices.numel() == 0:
            return
        state = self._mtp_shared_sparse_indices
        row_to_req = metadata.get_token_to_batch_idx().long()
        row_req_pool_indices = metadata.req_pool_indices[
            row_to_req[: topk_indices.shape[0]]
        ]
        is_last = torch.ones_like(row_req_pool_indices, dtype=torch.bool)
        if row_req_pool_indices.numel() > 1:
            is_last[:-1] = row_req_pool_indices[:-1] != row_req_pool_indices[1:]
        anchor_rows = is_last.nonzero().flatten()
        req_rows = row_req_pool_indices[anchor_rows]
        captured_lens = metadata.get_seqlens_expanded()[anchor_rows]
        state.capture(topk_indices[anchor_rows], req_rows, captured_lens, layer_id)

    @staticmethod
    def _capture_extend_seq_lens(forward_batch) -> torch.Tensor:
        """Same resolution order as _stage_extend_lens,
        so the capture and in-graph layout read the same replay-refreshed buffer."""
        spec_info = getattr(forward_batch, "spec_info", None)
        extend_seq_lens = getattr(spec_info, "extend_seq_lens_tensor", None)
        if extend_seq_lens is None:
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            raise RuntimeError(
                "QSA draft-extend capture requires GPU extend lengths "
                "(spec_info.extend_seq_lens_tensor or "
                "forward_batch.extend_seq_lens)"
            )
        return extend_seq_lens

    def _capture_mtp_sparse_indices_from_extend_lens(
        self, topk_indices, forward_batch, metadata, layer_id: int
    ) -> None:
        """DRAFT_EXTEND_V2 packs each request as [front rows][draft-window rows],
        so its last accepted row is block_end - (window - front - num_accept);
        rows with no real request (zero extend, padding slot 0) go to the trash row."""

        state = self._mtp_shared_sparse_indices
        bs = int(forward_batch.batch_size)
        if bs == 0 or forward_batch.req_pool_indices.numel() < bs:
            return
        request_extend_lens = self._capture_extend_seq_lens(forward_batch)[:bs].long()
        block_ends = request_extend_lens.cumsum(0) - 1
        spec_info = forward_batch.spec_info
        num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
        if num_accept_tokens is not None and num_accept_tokens.numel() >= bs:
            # Only EagleDraftExtendInput carries accept counts;
            # prefill-side EXTEND has no draft window, so its rows end at the seed.
            front_tokens = int(spec_info.num_front_tokens)
            anchor_rows = block_ends - (
                request_extend_lens - front_tokens - num_accept_tokens[:bs].long()
            )
        else:
            anchor_rows = block_ends
        anchor_rows = anchor_rows.clamp(min=0, max=topk_indices.shape[0] - 1)
        req_pool_rows = forward_batch.req_pool_indices[:bs].long()
        req_rows = torch.where(
            (request_extend_lens > 0) & (req_pool_rows > 0),
            req_pool_rows,
            state.trash_row,
        )
        captured_lens = metadata.get_seqlens_expanded()[anchor_rows]
        state.capture(
            topk_indices.index_select(0, anchor_rows),
            req_rows,
            captured_lens,
            layer_id,
        )

    def lookup_mtp_sparse_indices(self, forward_batch, layer_id: int) -> torch.Tensor:
        metadata = self.get_indexer_metadata(layer_id, forward_batch)
        logical_positions = metadata.decode_logical_positions
        if logical_positions is None:
            logical_positions = metadata.get_seqlens_expanded() - 1
        return self._mtp_shared_sparse_indices.lookup(
            forward_batch.req_pool_indices,
            logical_positions,
            layer_id,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _resolve_metadata(self, forward_batch) -> QwenSparseAttnMetadata:
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)
        assert self.forward_metadata is not None
        return self.forward_metadata

    @staticmethod
    def _logical_to_physical(
        logical_indices: torch.Tensor, metadata: QwenSparseAttnMetadata
    ) -> torch.Tensor:
        sequence_ids = metadata.token_to_batch_idx.long()
        if sequence_ids.numel() != logical_indices.shape[0]:
            raise ValueError("QSA top-k rows do not match query rows")
        row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(
            0, sequence_ids
        )
        valid = (logical_indices >= 0) & (logical_indices < row_lengths.unsqueeze(1))
        safe = logical_indices.clamp(
            min=0, max=metadata.token_slot_table.shape[1] - 1
        ).long()
        slots = metadata.token_slot_table[sequence_ids[:, None], safe]
        return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk_indices is None:
            raise ValueError("QSA sparse attention requires topk_indices")
        if save_kv_cache:
            self._store_kv(layer, forward_batch.out_cache_loc, k, v)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        num_output_rows = q.shape[0]
        num_valid_rows = topk_indices.shape[0]
        if num_valid_rows > num_output_rows:
            raise ValueError(
                "QSA top-k rows exceed query rows: "
                f"topk={num_valid_rows}, query={num_output_rows}"
            )
        # DP attention may pad q beyond the indexer's query rows.
        # The kernels see only valid rows; the output is zero-padded to q's row count.
        q = q[:num_valid_rows]
        if self._is_speculative_paged_mode(forward_batch.forward_mode):
            output = self._forward_paged_attention(
                q, layer, forward_batch, topk_indices
            )
            return self._pad_extend_output(output, num_output_rows)
        if not q.is_cuda:
            if self.arc_nvfp4:
                raise RuntimeError("NVFP4 QSA has no CPU attention path")
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            pool = self.token_to_kv_pool
            output = qsa_sparse_attention(
                q,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
                slots,
                layer.scaling,
            )
            return self._pad_extend_output(output, num_output_rows)

        topk_indices = topk_indices.to(torch.int32).contiguous()
        extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        sequence_lens = [int(x) for x in forward_batch.seq_lens_cpu]
        prefix_lens = [
            sequence_lens[i] - extend_lens[i] for i in range(len(extend_lens))
        ]
        cu_seqlens_q = F.pad(
            forward_batch.extend_seq_lens.to(q.device, dtype=torch.int32).cumsum(0),
            (1, 0),
        ).contiguous()
        if not any(prefix_lens):
            output = sparse_gqa_fwd_interface_triton(
                q.contiguous(),
                k[:num_valid_rows].contiguous(),
                v[:num_valid_rows].contiguous(),
                max(sequence_lens, default=1),
                topk_indices,
                cu_seqlens_q,
                layer.scaling,
            )
            return self._pad_extend_output(output, num_output_rows)

        # The validated chunk-prefill kernel consumes tightly packed full-context
        # K/V. Current-chunk K/V has already been committed to the cache above.
        pool = self.token_to_kv_pool
        sequence_lens_tensor = torch.tensor(
            sequence_lens, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = F.pad(sequence_lens_tensor.cumsum(0), (1, 0)).contiguous()
        # Resumed session prefixes on host (arc-forge M1b h07); None without any.
        host_lens = self._hisparse_host_prefix_lens(forward_batch)
        if host_lens is not None:
            if self.arc_nvfp4:
                raise NotImplementedError(
                    "QSA HiSparse host-prefix context for NVFP4 KV is M3 (whole-record "
                    "H2D into a raw scratch, then expansion)"
                )
            k_context, v_context = self._hisparse_host_prefix_context(
                layer, forward_batch, sequence_lens, host_lens
            )
        elif self.arc_nvfp4:
            # One fused pass per layer: gather + dequantize + downcast write
            # straight into the buffer the chunk kernel consumes. The previous
            # index_select -> BF16 -> FP8 -> torch.cat chain held four copies of
            # the same context alive at once, which is what forced the pool cap.
            # The expansion folds the global scale in, and
            # _sparse_gqa_chunk_prefill casts FP8 K/V to the query dtype with no
            # scale of its own, so no k_scale/v_scale reaches the kernel.
            k_context, v_context = self._nvfp4_expand_context(
                layer, forward_batch, cu_seqlens_k, sequence_lens, q
            )
        else:
            slot_table, slot_rows = self._context_slot_table(
                forward_batch, sequence_lens
            )
            req_indices = slot_rows.tolist()
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
            k_context = torch.cat(
                [
                    k_buffer.index_select(
                        0, slot_table[req_indices[i], : sequence_lens[i]].long()
                    )
                    for i in range(len(sequence_lens))
                ]
            )
            v_context = torch.cat(
                [
                    v_buffer.index_select(
                        0, slot_table[req_indices[i], : sequence_lens[i]].long()
                    )
                    for i in range(len(sequence_lens))
                ]
            )
        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            k_context,
            v_context,
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence_lens_tensor,
            layer.scaling,
        )
        return self._pad_extend_output(output, num_output_rows)

    def _context_slot_table(
        self, forward_batch, sequence_lens, host_lens=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(table, row per request)`` a prefix chunk gathers its context
        through: ``table[row[i], :sequence_lens[i]]`` are request i's pool slots.

        Without HiSparse this is req_to_token and the requests' pool rows. With
        it the pool holds device rows only, so the logical slots are translated
        (``pool.rows_of``) into a per-request table read with rows ``arange(bs)``.
        ``host_lens[i]`` > 0 declares positions [0, host_lens[i]) a host-resident
        prefix (h07): they must have no device row, and are not read from here.
        """
        req_to_token = self.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices
        if self.hisparse_coordinator is None:
            return req_to_token, req_pool_indices
        device = req_to_token.device
        width = max(sequence_lens)
        lengths = torch.tensor(sequence_lens, dtype=torch.int64, device=device)
        positions = torch.arange(width, device=device)[None, :]
        live = positions < lengths[:, None]
        # Past a request's length the table holds stale slots; read slot 0 there.
        logical = torch.where(live, req_to_token[req_pool_indices.long(), :width], 0)
        rows = self.token_to_kv_pool.rows_of(logical)
        # An unmapped slot translates into device block 0, the reserved dump.
        no_row = rows < self.compress_ratio
        if host_lens is not None:
            hosts = torch.tensor(host_lens, dtype=torch.int64, device=device)
            on_host = positions < hosts[:, None]
            if bool((on_host & ~no_row).any()):
                raise RuntimeError(
                    "QSA HiSparse prefill context: a declared host prefix has device rows"
                )
            live = live & ~on_host
        if bool((no_row & live).any()):
            raise RuntimeError(
                "QSA HiSparse prefill context has a slot with no device row"
            )
        rows = rows.to(torch.int32)
        return rows, torch.arange(
            rows.shape[0], dtype=req_pool_indices.dtype, device=device
        )

    def _hisparse_host_prefix_lens(self, forward_batch) -> Optional[List[int]]:
        """Per request, the tokens of its adopted host prefix (M1b h07), or None
        when no request of this batch has one (the M1a path, no extra sync)."""
        coord = self.hisparse_coordinator
        if coord is None or not coord.has_host_prefix():
            return None
        lens = coord.host_prefix_lens(forward_batch.req_pool_indices.tolist())
        return lens if any(lens) else None

    def _hisparse_host_prefix_context(
        self, layer, forward_batch, sequence_lens, host_lens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tightly packed FP8 context of a batch with resumed prefixes (M1b h07,
        design §4.3): request i's rows [host_lens[i], seq) are gathered from its
        device rows, rows [0, host_lens[i]) come from its host records through
        the coordinator's landing zone. Same bytes as a full-resident gather."""
        pool = self.token_to_kv_pool
        coord = self.hisparse_coordinator
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        slot_table, _ = self._context_slot_table(forward_batch, sequence_lens, host_lens)
        total = sum(sequence_lens)
        k_context = k_buffer.new_empty((total,) + tuple(k_buffer.shape[1:]))
        v_context = v_buffer.new_empty((total,) + tuple(v_buffer.shape[1:]))
        req_indices = forward_batch.req_pool_indices.tolist()
        segments = []
        start = 0
        for i, (seq, host) in enumerate(zip(sequence_lens, host_lens)):
            if not 0 <= host < seq:
                raise RuntimeError(
                    f"QSA HiSparse context: host prefix {host} of a {seq}-token request"
                )
            rows = slot_table[i, host:seq].long()
            torch.index_select(k_buffer, 0, rows, out=k_context[start + host : start + seq])
            torch.index_select(v_buffer, 0, rows, out=v_context[start + host : start + seq])
            if host:
                segments.append((req_indices[i], host, start))
            start += seq
        coord.fill_host_prefix_context(
            coord.layer_index(layer.layer_id), segments, k_context, v_context
        )
        return k_context, v_context

    def _nvfp4_expand_context(
        self, layer, forward_batch, cu_seqlens_k, sequence_lens, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize every request's packed NVFP4 context into one K/V pair.

        The chunk-prefill kernel re-reads the selected rows once per query row,
        so FP8 -- half the bytes of the query dtype -- is what it should be
        handed; ``ARC_QSA_NVFP4_PREFILL=bf16`` keeps the dequantized precision
        at twice the footprint, for measuring what the downcast costs.
        """
        pool = self.token_to_kv_pool
        k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer.layer_id)
        method = pool.get_kv_cache_quant_method()
        heads, dim = k_fp4.shape[1], k_fp4.shape[2] * 2
        dtype = torch.float8_e4m3fn if _ARC_QSA_NVFP4_PREFILL == "fp8" else q.dtype
        shape = (sum(sequence_lens), heads, dim)
        k_context = torch.empty(shape, dtype=dtype, device=q.device)
        v_context = torch.empty(shape, dtype=dtype, device=q.device)
        layer_id = layer.layer_id
        slot_table, slot_rows = self._context_slot_table(forward_batch, sequence_lens)
        qwen_sparse_nvfp4_expand_triton(
            k_fp4,
            v_fp4,
            k_sf,
            v_sf,
            slot_table,
            slot_rows,
            cu_seqlens_k,
            k_context,
            v_context,
            method.k_scales_gpu[layer_id : layer_id + 1],
            method.v_scales_gpu[layer_id : layer_id + 1],
            max(sequence_lens),
        )
        return k_context, v_context

    @staticmethod
    def _pad_extend_output(output: torch.Tensor, num_rows: int) -> torch.Tensor:
        output = output.reshape(output.shape[0], -1)
        if output.shape[0] == num_rows:
            return output
        if output.shape[0] > num_rows:
            raise ValueError(
                "QSA attention output rows exceed padded query rows: "
                f"output={output.shape[0]}, query={num_rows}"
            )
        padded = output.new_zeros((num_rows, output.shape[1]))
        padded[: output.shape[0]].copy_(output)
        return padded

    def _get_fa2_scratch(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (num_kv_heads, head_dim, dtype, device)
        buffers = self._fa2_scratch.get(key)
        if buffers is None or buffers[0].shape[0] < capacity:
            shape = (capacity, num_kv_heads, head_dim)
            buffers = (
                torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device),
            )
            self._fa2_scratch[key] = buffers
        return buffers[0][:capacity], buffers[1][:capacity]

    def _get_trtllm_sparse_tables(self, batch, pages_per_row, page, device):
        key = (batch, pages_per_row, device)
        cached = self._trtllm_sparse_tables.get(key)
        if cached is None:
            stride = pages_per_row * page
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * stride
            block_tables = (
                torch.arange(batch, dtype=torch.int32, device=device)[:, None]
                * pages_per_row
                + torch.arange(pages_per_row, dtype=torch.int32, device=device)[None, :]
            ).contiguous()
            cached = (cu, block_tables)
            self._trtllm_sparse_tables[key] = cached
        return cached

    def _forward_trtllm_sparse(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        layer,
        forward_batch,
        metadata,
        topk_indices: torch.Tensor,
        trtllm_decode,
    ) -> torch.Tensor:
        """Pack selected KV at page-aligned row strides for FlashInfer's paged decode,
        driven by a static arange block table and the per-row valid counts."""
        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
        pages_per_row = (topk + page - 1) // page
        stride = pages_per_row * page
        device = q.device
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            if valid_counts is None:
                raise RuntimeError("QSA CUDA graph metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
        qwen_sparse_valid_counts_triton(
            sequence_lens, topk_indices, valid_counts, batch, topk
        )
        cu_strided, block_tables = self._get_trtllm_sparse_tables(
            batch, pages_per_row, page, device
        )
        capacity_rows = self._cuda_graph_max_tokens if metadata.is_cuda_graph else batch
        # Gather into the query dtype: an FP8 pool is dequantized on the way in, so the
        # paged kernel always runs the bf16 q + bf16 KV path.
        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
        )
        hisparse = metadata.indexer_metadata.hisparse
        if (hisparse is None) != (self.hisparse_coordinator is None):
            raise RuntimeError(
                "QSA decode metadata disagrees with the HiSparse coordinator"
            )
        if hisparse is None:
            gather_table = self.req_to_token_pool.req_to_token
            gather_rows = (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            )
            gather_indices = topk_indices
            gather_lens = sequence_lens
        else:
            # The pool holds device rows only: gather hisparse.rows through an
            # identity table (one row, bound rows_limit) instead of mapping the
            # logical indices through req_to_token. valid_counts stay logical;
            # the two index sets share one valid pattern.
            if hisparse.rows.shape != topk_indices.shape:
                raise RuntimeError(
                    f"HiSparse rows {tuple(hisparse.rows.shape)} do not match "
                    f"the top-k indices {tuple(topk_indices.shape)}"
                )
            gather_table = hisparse.coord.identity_rows
            gather_rows = hisparse.coord.zeros_i32[:batch]
            gather_indices = hisparse.rows
            gather_lens = hisparse.coord.rows_limit[:batch]
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            gather_table,
            gather_rows,
            gather_indices,
            gather_lens,
            cu_strided,
            packed_k,
            packed_v,
            batch,
            topk,
            zero_fill_cols=stride,
        )
        num_kv_heads = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]
        kc = (
            packed_k[: batch * stride]
            .view(-1, page, num_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        vc = (
            packed_v[: batch * stride]
            .view(-1, page, num_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        if self._trtllm_workspace is None:
            self._trtllm_workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        output = trtllm_decode(
            query=q.contiguous(),
            kv_cache=(kc, vc),
            workspace_buffer=self._trtllm_workspace,
            block_tables=block_tables,
            seq_lens=valid_counts,
            max_seq_len=stride,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
        )
        return output.reshape(q.shape[0], -1)

    def _trtllm_sparse_nvfp4_sm100(
        self, q, layer, trtllm_decode, packed_k, packed_v, packed_ks, packed_vs,
        n, page, num_kv_heads, packed_dim, scale_dim, sf_dtype,
        block_tables, valid_counts, stride, k_global, v_global, device,
    ):
        """Hand the gathered selection to sm_100's trtllm-gen FP4 decode kernel.

        That kernel is a different family from the XQA one sm_120 uses, and it
        differs in three ways that have to be honoured together: it reads HND,
        it has no BF16-query and no BF16-output FP4 cubin at head_dim 256, and
        it reads the value scale plane in groups of ``_SF_TOKEN_GROUP`` tokens
        with the channel blocks interleaved inside each group while reading the
        key plane linearly.

        The scale layout is the dangerous one. Feeding it the linear plane that
        ``nvfp4_kv_quantize`` produces raises nothing and returns a finite,
        strongly correlated, wrong answer, off in proportion to how much the
        per-block scales vary -- so there is no failure to notice at runtime.
        ``work/qwen38fn_eval/probe_trtllm_fp4_kv_sm100.py`` derives the layout
        and gates it.
        """
        pages = n // page
        kv_cache = (
            packed_k[:n].view(pages, page, num_kv_heads, packed_dim).permute(0, 2, 1, 3),
            packed_v[:n].view(pages, page, num_kv_heads, packed_dim).permute(0, 2, 1, 3),
        )
        # Already in the kernel's order: the gather wrote it there. Only two
        # strides reach the kernel for the scale planes (head and page), so
        # their inner [page, scale_dim] tile does have to be contiguous, which
        # this view is and a permuted one would not be.
        kv_cache_sf = (
            packed_ks[:n].view(pages, num_kv_heads, page, scale_dim).view(sf_dtype),
            packed_vs[:n].view(pages, num_kv_heads, page, scale_dim).view(sf_dtype),
        )
        # The query is cast without a scale, as sglang's own trtllm_mha backend
        # does for FP8 KV. E4M3 is a floating-point format, so folding a
        # constant into it buys no relative precision -- measured at 0.0267 vs
        # 0.0265 relative L2 -- while the amax reduction and the divide cost
        # ~32us per decode step. All it would buy is headroom against the 448
        # ceiling, and post-norm queries run three orders of magnitude below it.
        output = trtllm_decode(
            query=q.to(torch.float8_e4m3fn),
            kv_cache=kv_cache,
            workspace_buffer=self._trtllm_workspace,
            block_tables=block_tables,
            seq_lens=valid_counts,
            max_seq_len=stride,
            bmm1_scale=layer.scaling * k_global,
            bmm2_scale=v_global,
            kv_layout="HND",
            out_dtype=torch.float8_e4m3fn,
            kv_cache_sf=kv_cache_sf,
        )
        return output.to(q.dtype).reshape(q.shape[0], -1)

    def _forward_trtllm_sparse_nvfp4(
        self,
        q: torch.Tensor,
        layer,
        forward_batch,
        metadata,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Selected tokens are gathered as packed FP4 bytes plus their block
        scales into page-aligned scratch; trtllm-gen XQA reads FP4 directly
        through kv_cache_sf, so no dequantized copy of the selection exists."""
        trtllm_decode = _resolve_trtllm_sparse_decode()
        pool = self.token_to_kv_pool
        k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer.layer_id)
        k_global, v_global = pool.get_kv_cache_quant_method().get_bmm_scales(layer.layer_id)
        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
        pages_per_row = (topk + page - 1) // page
        stride = pages_per_row * page
        device = q.device
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            if valid_counts is None:
                raise RuntimeError("QSA CUDA graph metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
        qwen_sparse_valid_counts_triton(
            sequence_lens, topk_indices, valid_counts, batch, topk
        )
        cu_strided, block_tables = self._get_trtllm_sparse_tables(
            batch, pages_per_row, page, device
        )
        capacity_rows = self._cuda_graph_max_tokens if metadata.is_cuda_graph else batch
        rows = max(capacity_rows, batch) * stride
        num_kv_heads, packed_dim = k_fp4.shape[1], k_fp4.shape[2]
        scale_dim = k_sf.shape[2]
        if k_fp4.dtype != torch.uint8 or v_fp4.dtype != torch.uint8:
            raise RuntimeError(f"Packed NVFP4 K/V must be uint8, got {k_fp4.dtype}/{v_fp4.dtype}")
        packed_k, packed_v = self._get_fa2_scratch(
            rows, num_kv_heads, packed_dim, torch.uint8, device
        )
        packed_ks, packed_vs = self._get_fa2_scratch(
            rows, num_kv_heads, scale_dim, torch.uint8, device
        )
        req_indices = (
            metadata.row_req_pool_indices
            if metadata.row_req_pool_indices is not None
            else forward_batch.req_pool_indices
        )
        # sm_100 wants the scale planes paged and the value plane grouped; the
        # gather writes that order directly rather than paying a second strided
        # pass over the scratch afterwards. The packed byte planes stay
        # token-major because three strides reach the kernel for those, so a
        # permuted view is enough.
        sf_dst_layout = (
            {"dst_page_size": page, "dst_value_group": _SF_TOKEN_GROUP}
            if self.arc_nvfp4_sm100
            else {}
        )
        if _ARC_QSA_NVFP4_GATHER == "fused":
            qwen_sparse_nvfp4_gather4_triton(
                k_fp4,
                v_fp4,
                k_sf.view(torch.uint8),
                v_sf.view(torch.uint8),
                self.req_to_token_pool.req_to_token,
                req_indices,
                topk_indices,
                sequence_lens,
                cu_strided,
                packed_k,
                packed_v,
                packed_ks,
                packed_vs,
                batch,
                topk,
                stride,
                dst_page_size=sf_dst_layout.get("dst_page_size", 0),
                dst_value_group=sf_dst_layout.get("dst_value_group", 0),
            )
        else:
            for src_k, src_v, dst_k, dst_v, dst_layout in (
                (k_fp4, v_fp4, packed_k, packed_v, {}),
                (k_sf.view(torch.uint8), v_sf.view(torch.uint8), packed_ks, packed_vs, sf_dst_layout),
            ):
                qwen_sparse_kv_extraction_compact_triton(
                    src_k,
                    src_v,
                    self.req_to_token_pool.req_to_token,
                    req_indices,
                    topk_indices,
                    sequence_lens,
                    cu_strided,
                    dst_k,
                    dst_v,
                    batch,
                    topk,
                    zero_fill_cols=stride,
                    **dst_layout,
                )
        n = batch * stride
        if self._trtllm_workspace is None:
            self._trtllm_workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        if self.arc_nvfp4_sm100:
            return self._trtllm_sparse_nvfp4_sm100(
                q, layer, trtllm_decode, packed_k, packed_v, packed_ks, packed_vs,
                n, page, num_kv_heads, packed_dim, scale_dim, k_sf.dtype,
                block_tables, valid_counts, stride, k_global, v_global, device,
            )
        kv_cache = (
            packed_k[:n].view(-1, page, num_kv_heads, packed_dim),
            packed_v[:n].view(-1, page, num_kv_heads, packed_dim),
        )
        kv_cache_sf = (
            packed_ks[:n].view(-1, page, num_kv_heads, scale_dim).view(k_sf.dtype),
            packed_vs[:n].view(-1, page, num_kv_heads, scale_dim).view(v_sf.dtype),
        )
        output = trtllm_decode(
            query=q.contiguous(),
            kv_cache=kv_cache,
            workspace_buffer=self._trtllm_workspace,
            block_tables=block_tables,
            seq_lens=valid_counts,
            max_seq_len=stride,
            bmm1_scale=layer.scaling * k_global,
            bmm2_scale=v_global,
            kv_layout="NHD",
            out_dtype=q.dtype,
            kv_cache_sf=kv_cache_sf,
        )
        return output.reshape(q.shape[0], -1)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk_indices is None:
            raise ValueError("QSA sparse attention requires topk_indices")
        if save_kv_cache:
            self._store_kv(layer, forward_batch.out_cache_loc, k, v)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        return self._forward_paged_attention(q, layer, forward_batch, topk_indices)

    def _forward_paged_attention(
        self,
        q: torch.Tensor,
        layer,
        forward_batch,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        pool = self.token_to_kv_pool
        if self.arc_nvfp4:
            if not q.is_cuda:
                raise RuntimeError("NVFP4 QSA has no CPU attention path")
            metadata = self._resolve_metadata(forward_batch)
            return self._forward_trtllm_sparse_nvfp4(
                q, layer, forward_batch, metadata,
                topk_indices.to(torch.int32).contiguous(),
            )
        if (
            _ARC_QSA_FP4_FUSED
            and _QSA_DIRECT_DECODE
            and q.is_cuda
            and q.shape[0] >= _QSA_DIRECT_MIN_ROWS
            and getattr(getattr(pool, "full_kv_pool", pool), "k_scale_buffer", None)
            is not None
        ):
            # Take the packed plane before get_key_buffer(), which on a quantized
            # pool dequantizes the whole layer buffer on every call.
            metadata = self._resolve_metadata(forward_batch)
            k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer.layer_id)
            method = getattr(getattr(pool, "full_kv_pool", pool), "quant_method", None)
            getter = getattr(method, "get_bmm_scales", None)
            k_global, v_global = getter(layer.layer_id) if getter else (1.0, 1.0)
            req_indices = (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            )
            output = qwen_sparse_gqa_decode_selected_fp4_triton(
                q.contiguous(), k_fp4, v_fp4, k_sf, v_sf,
                self.req_to_token_pool.req_to_token, req_indices,
                topk_indices.to(torch.int32).contiguous(),
                metadata.sequence_lengths, layer.scaling,
                float(k_global), float(v_global),
            )
            return output.reshape(q.shape[0], -1)
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        if _QSA_DIRECT_DECODE and q.is_cuda and q.shape[0] >= _QSA_DIRECT_MIN_ROWS:
            # Ada: trtllm-gen/XQA is SM100+, and the FA4 cute fallback does not
            # build here, so this Triton path is the only working QSA decode.
            metadata = self._resolve_metadata(forward_batch)
            req_indices = (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            )
            k_scale, v_scale = self._kv_descales(layer, k_buffer.dtype)
            output = qwen_sparse_gqa_decode_selected_triton(
                q.contiguous(), k_buffer, v_buffer,
                self.req_to_token_pool.req_to_token, req_indices,
                topk_indices.to(torch.int32).contiguous(),
                metadata.sequence_lengths, layer.scaling,
                k_scale=k_scale, v_scale=v_scale,
            )
            return output.reshape(q.shape[0], -1)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)

        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        trtllm_decode = _resolve_trtllm_sparse_decode()
        if trtllm_decode is not None:
            return self._forward_trtllm_sparse(
                q,
                k_buffer,
                v_buffer,
                layer,
                forward_batch,
                metadata,
                topk_indices,
                trtllm_decode,
            )

        flash_attn_varlen_func = _resolve_flash_attn_varlen_func()
        batch, topk = topk_indices.shape
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            cu_seqlens_k = metadata.fa2_cu_seqlens_k
            cu_seqlens_q = metadata.fa2_cu_seqlens_q
            if valid_counts is None or cu_seqlens_k is None or cu_seqlens_q is None:
                raise RuntimeError("QSA CUDA graph FA2 metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
            cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
        qwen_sparse_fa2_cu_seqlens_triton(
            sequence_lens,
            topk_indices,
            valid_counts,
            cu_seqlens_k,
            batch,
            topk,
        )
        scratch_capacity = (
            self._cuda_graph_max_tokens * topk
            if metadata.is_cuda_graph
            else batch * topk
        )
        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
        )
        output = flash_attn_varlen_func(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            softmax_scale=layer.scaling,
            causal=True,
        )
        return output.reshape(q.shape[0], -1)


class QwenSparseMultiStepDraftBackend:
    """Per-step QSA metadata for consecutive Qwen4-Exp MTP draft decoding."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner, topk: int, speculative_num_steps: int):
        if topk != 1:
            raise NotImplementedError(
                "Qwen4-Exp QSA MTP currently supports speculative_eagle_topk=1"
            )
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            QwenSparseAttnBackend(model_runner)
            for _ in range(speculative_num_steps - 1)
        ]

    @staticmethod
    def _as_cpu_lengths(seq_lens_cpu, seq_lens: torch.Tensor) -> torch.Tensor:
        if seq_lens_cpu is None:
            return seq_lens.detach().cpu().to(torch.int32)
        if isinstance(seq_lens_cpu, torch.Tensor):
            return seq_lens_cpu.to(dtype=torch.int32)
        return torch.tensor(seq_lens_cpu, dtype=torch.int32)

    def _step_out_cache_loc(self, forward_batch, step: int):
        """Mirror EagleDraftWorker.draft_forward's per-step out_cache_loc slice;
        a mismatch makes every QSA step write into the first ``batch_size`` slots."""

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        steps = self.speculative_num_steps
        batch_rows = int(
            getattr(
                forward_batch,
                "batch_size",
                forward_batch.seq_lens.numel(),
            )
        )
        if out_cache_loc is None or steps <= 1:
            return out_cache_loc
        if out_cache_loc.numel() != batch_rows * self.topk * steps:
            # Idle/zero-row and non-draft batches do not use the interleaved layout.
            return out_cache_loc
        # Same expression chain as EagleDraftWorker.draft_forward (views only).
        return (
            out_cache_loc.reshape(batch_rows, self.topk, steps)
            .permute(2, 0, 1)
            .reshape(steps, -1)[step]
        )

    def _make_step_forward_batch(self, forward_batch, step: int, num_padding: int = 0):
        step_forward_batch = copy(forward_batch)
        step_forward_batch.forward_mode = ForwardMode.DECODE
        step_forward_batch.seq_lens = (forward_batch.seq_lens + step + 1).to(
            torch.int32
        )
        if forward_batch.seq_lens_cpu is None:
            # GPU-only serving: downstream metadata paths derive host bounds
            # from batch shape; do not force a per-step D2H here.
            step_forward_batch.seq_lens_cpu = None
        else:
            step_forward_batch.seq_lens_cpu = self._as_cpu_lengths(
                forward_batch.seq_lens_cpu, forward_batch.seq_lens
            ) + (step + 1)
        num_padding = max(
            0,
            min(int(num_padding), int(step_forward_batch.seq_lens.numel())),
        )
        if num_padding:
            step_forward_batch.seq_lens = step_forward_batch.seq_lens.clone()
            step_forward_batch.seq_lens[-num_padding:] = 1
            if step_forward_batch.seq_lens_cpu is not None:
                step_forward_batch.seq_lens_cpu = (
                    step_forward_batch.seq_lens_cpu.clone()
                )
                step_forward_batch.seq_lens_cpu[-num_padding:] = 1
        step_forward_batch.batch_size = int(step_forward_batch.seq_lens.numel())
        step_forward_batch.out_cache_loc = self._step_out_cache_loc(forward_batch, step)
        return step_forward_batch

    def set_mtp_shared_sparse_indices(self, state) -> None:
        for backend in self.attn_backends:
            backend.set_mtp_shared_sparse_indices(state)

    def init_forward_metadata(self, forward_batch):
        for step, backend in enumerate(self.attn_backends):
            backend.init_forward_metadata(
                self._make_step_forward_batch(forward_batch, step)
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for backend in self.attn_backends:
            backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if in_capture:
            for step, backend in enumerate(self.attn_backends):
                # Warmup must not write via shared req_pool_idx 0;
                # pad every capture row to stay below the first compression boundary.
                step_batch = self._make_step_forward_batch(
                    forward_batch,
                    step,
                    num_padding=forward_batch.batch_size,
                )
                backend._capture_cuda_graph_metadata(
                    bs=step_batch.batch_size,
                    num_tokens=step_batch.batch_size,
                    req_pool_indices=step_batch.req_pool_indices,
                    seq_lens=step_batch.seq_lens,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=step_batch.spec_info,
                )
            return

        num_padding = getattr(forward_batch, "num_padding", None)
        num_padding = num_padding if num_padding is not None else 0
        for step, backend in enumerate(self.attn_backends):
            step_batch = self._make_step_forward_batch(
                forward_batch, step, num_padding=num_padding
            )
            backend._replay_cuda_graph_metadata(
                bs=step_batch.batch_size,
                req_pool_indices=step_batch.req_pool_indices,
                seq_lens=step_batch.seq_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=step_batch.spec_info,
                seq_lens_cpu=step_batch.seq_lens_cpu,
                num_padding=num_padding,
            )

    def init_forward_metadata_in_graph(self, forward_batch) -> None:
        pass


__all__ = [
    "is_qwen_qsa",
    "QwenSparseAttnMetadata",
    "QwenSparseAttnBackend",
    "QwenSparseMultiStepDraftBackend",
]
