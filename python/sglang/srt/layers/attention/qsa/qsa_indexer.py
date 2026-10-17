"""QSA indexer for Qwen4-Exp checkpoints."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.qsa.kernel import (
    average_pool_qsa_keys,
    expand_qsa_block_indices,
    expand_qsa_block_indices_hot,
    qsa_fast_topk,
)
from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)
from sglang.srt.layers.attention.qsa.mqa import (
    qsa_mqa_decode,
    qsa_mqa_prefill,
    qsa_prefill_local_tiles,
    tilelang_qsa_mqa_prefill_local,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.runner import get_is_capture_mode

# Cap on the fp32 [query_rows, compressed_keys] prefill logits workspace;
# top-k is per row, so tiling rows does not change the selection.
_QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


# Test flag (arc-forge QSA HiSparse gate): fast_topk returns its picks in
# atomic-collection order, so two runs of one selection can attend in different
# orders. Sorting each row's valid blocks ascending (-1 padding stays last)
# fixes only the order: ties and threshold-bin overflow still change the set,
# so a bitwise end-to-end comparison needs ARC_QSA_TOPK_EXACT=1 (fast_topk.py).
_ARC_QSA_TOPK_SORT = _env_flag("ARC_QSA_TOPK_SORT")
# b200-perf-prefill (b): prefill scores in sequence-local columns (select_prefill_tokens_local)
_ARC_QSA_INDEXER_LOCAL = _env_flag("ARC_QSA_INDEXER_LOCAL")
# Eager-only HiSparse checks of the selection the swap-in kernel consumes.
_ARC_HISPARSE_QSA_DEBUG = _env_flag("ARC_HISPARSE_QSA_DEBUG")


def _sort_topk_blocks(block_indices: torch.Tensor) -> torch.Tensor:
    sentinel = torch.iinfo(block_indices.dtype).max
    keyed = torch.where(block_indices < 0, sentinel, block_indices)
    keyed = torch.sort(keyed, dim=1).values
    return torch.where(keyed == sentinel, -1, keyed)


def _check_hisparse_selection(
    block_indices: torch.Tensor,
    compressed_lengths: torch.Tensor,
    device_blocks: torch.Tensor,
    block_topk: int,
) -> None:
    """The swap-in kernel reads -1 as TOKEN_HIT and miscounts duplicates on its
    LRU path, and a -1 device block would be gathered as an empty key; none of
    that raises there. Require the fast_topk contract the kernel relies on:
    min(complete blocks, top-k) distinct complete blocks first, -1 after, and
    every valid entry resolved to a device block."""
    valid = block_indices >= 0
    count = valid.sum(dim=1)
    lengths = compressed_lengths.to(device=block_indices.device, dtype=torch.long)
    cols = torch.arange(block_indices.shape[1], device=block_indices.device)
    sentinel = torch.iinfo(block_indices.dtype).max
    ordered = torch.sort(torch.where(valid, block_indices, sentinel), dim=1).values
    checks = (
        (
            "-1 before a valid entry",
            (valid != (cols[None, :] < count[:, None])).any(dim=1),
        ),
        (
            "valid count != min(complete blocks, top-k)",
            count != lengths.clamp(max=block_topk),
        ),
        (
            "block past the complete blocks",
            (valid & (block_indices >= lengths[:, None])).any(dim=1),
        ),
        (
            "duplicate block",
            ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != sentinel)).any(
                dim=1
            ),
        ),
        (
            "selected block without a device block",
            (valid & (device_blocks < 0)).any(dim=1),
        ),
    )
    for what, bad in checks:
        if bool(bad.any()):
            rows = bad.nonzero().flatten()[:8].tolist()
            raise RuntimeError(f"QSA HiSparse selection: {what} in rows {rows}")


def _qsa_grouped_queries_per_request(
    forward_batch, *, is_target_verify: bool, is_draft_extend: bool
) -> Optional[int]:
    """Resolve the request-major speculative row width for grouped QSA MQA."""

    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is None:
        return None
    if is_target_verify:
        value = getattr(spec_info, "draft_token_num", None)
    elif is_draft_extend:
        value = getattr(spec_info, "num_tokens_per_req", None)
    else:
        return None
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _qsa_prefill_row_chunk_size(rows: int, keys: int, heads: int) -> int:
    if rows <= 0 or keys <= 0:
        return max(rows, 1)
    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, _QSA_PREFILL_LOGITS_BUDGET_BYTES // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


class QSAIndexer(MultiPlatformOp):
    """Config-driven fused-QK, weight-free sparse-attention indexer."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        rotary_emb=None,
    ) -> None:
        self._validate_config(config)
        super().__init__()
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_topk // self.compress_ratio
        if rotary_emb is None:
            raise ValueError("QSAIndexer must reuse its Qwen4-Exp attention RoPE")
        self.rotary_emb = rotary_emb
        if not 0 < self.rotary_emb.rotary_dim <= self.index_head_dim:
            raise ValueError(
                "Qwen4-Exp attention RoPE rotary_dim must fit the QSA index head: "
                f"{self.rotary_emb.rotary_dim=} {self.index_head_dim=}"
            )
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self._rope_axis_map_cache = None

    @staticmethod
    def _validate_config(config) -> None:
        names = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        missing = [name for name in names if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}")
        values = {name: int(getattr(config, name)) for name in names}
        if any(value <= 0 for value in values.values()):
            raise ValueError(f"QSA config values must be positive: {values}")
        if values["indexer_compress_ratio"] < 2:
            # DP token-padding rows carry logical length 1, which must never
            # reach a compression boundary; ratio >= 2 guarantees that.
            raise ValueError(
                "QSA requires indexer_compress_ratio >= 2, got "
                f"{values['indexer_compress_ratio']}"
            )
        if values["indexer_kv_heads"] != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if values["indexer_budget"] % values["indexer_compress_ratio"] != 0:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        block_topk = values["indexer_budget"] // values["indexer_compress_ratio"]
        if block_topk not in (512, 2048):
            raise ValueError(
                "fast_topk_v2 requires indexer_budget / indexer_compress_ratio "
                f"to be 512 or 2048, got {block_topk}"
            )

    def _use_fused_prep(self, tensor: torch.Tensor) -> bool:
        """Whether the fused indexer-prep kernels support this configuration."""
        return (
            tensor.is_cuda
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and self.index_head_dim in (64, 128, 256)
            and self.rotary_emb.rotary_dim % 2 == 0
            and not getattr(self.rotary_emb, "mrope_interleaved_glm", False)
            and len(getattr(self.rotary_emb, "mrope_section", None) or ()) in (0, 3)
            and getattr(self.rotary_emb, "cos_sin_cache", None) is not None
            and self.rotary_emb.cos_sin_cache.is_cuda
            and self.rotary_emb.cos_sin_cache.dtype == torch.float32
        )

    def _rope_axis_map(self, device) -> torch.Tensor:
        """axis_map[i] is the MRoPE position axis whose cos/sin rotary pair i reads."""
        cache = self._rope_axis_map_cache
        if cache is not None and cache.device == device:
            return cache
        half = self.rotary_emb.rotary_dim // 2
        section = getattr(self.rotary_emb, "mrope_section", None) or None
        axis_map = torch.zeros(half, dtype=torch.int32)
        if section is not None:
            s0, s1, s2 = (int(v) for v in section)
            if getattr(self.rotary_emb, "mrope_interleaved", False):
                pair = torch.arange(half, dtype=torch.int32)
                axis_map[((pair % 3) == 1) & (pair < s1 * 3)] = 1
                axis_map[((pair % 3) == 2) & (pair < s2 * 3)] = 2
            else:
                axis_map[s0 : s0 + s1] = 1
                axis_map[s0 + s1 :] = 2
        self._rope_axis_map_cache = axis_map.to(device)
        return self._rope_axis_map_cache

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        pool=None,
        cache_loc: torch.Tensor | None = None,
        q_heads_padded: int | None = None,
        max_position: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """max_position: an upper bound of positions.max() when the caller knows one on the host
        (h15: the extend metadata's max_position_cpu), so the RoPE cache check
        reads nothing back from the GPU."""
        qk, _ = self.index_qk_proj(hidden_states)
        token_k = qk[:, self.index_n_heads * self.index_head_dim :].reshape(
            -1, self.index_kv_heads, self.index_head_dim
        )
        if (
            pool is not None
            and cache_loc is not None
            and qk.shape[0] > 0
            and self._use_fused_prep(qk)
        ):
            from sglang.kernels.ops.attention.qsa_indexer import (
                qsa_index_q_norm_rope_store,
            )

            if not get_is_capture_mode() and hasattr(
                self.rotary_emb, "_ensure_cos_sin_cache_length"
            ):
                if max_position is None:
                    max_position = int(positions.max().item())
                elif _ARC_HISPARSE_QSA_DEBUG and max_position < int(positions.max().item()):
                    raise RuntimeError(
                        f"QSA indexer: host max position bound {max_position} < "
                        f"{int(positions.max().item())}"
                    )
                self.rotary_emb._ensure_cos_sin_cache_length(max_position)
            key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
            q = qsa_index_q_norm_rope_store(
                qk,
                positions.long(),
                self.rotary_emb.cos_sin_cache,
                self._rope_axis_map(qk.device),
                self.q_layernorm.weight.data,
                cache_loc[: qk.shape[0]].long(),
                key_state_buffer.view(key_state_buffer.shape[0], -1),
                pool.qsa_rope_position_buffer,
                self.index_n_heads,
                self.rotary_emb.rotary_dim,
                self.q_layernorm.variance_epsilon,
                self.rotary_emb.is_neox_style,
                q_heads_padded=q_heads_padded,
            )
            return q, token_k, True
        q_raw = qk[:, : self.index_n_heads * self.index_head_dim]
        q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(
            -1, self.index_n_heads, self.index_head_dim
        )
        q = self.apply_rope(positions, q)
        return q, token_k, False

    def normalize_compressed_keys(
        self, compressed_keys: torch.Tensor, block_positions: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.k_layernorm(
            compressed_keys.reshape(-1, self.index_head_dim)
        ).reshape(-1, self.index_kv_heads, self.index_head_dim)
        return self.apply_rope(block_positions, normalized)

    def _use_fused_compress(self, pool) -> bool:
        return getattr(
            pool, "qsa_rope_position_buffer", None
        ) is not None and self._use_fused_prep(
            pool.get_qsa_key_state_buffer(self.layer_id)
        )

    def _fused_compress_store(
        self,
        pool,
        group_locs: torch.Tensor,
        write_locs: torch.Tensor,
        source_keys: torch.Tensor | None = None,
        source_rope: torch.Tensor | None = None,
    ) -> None:
        """Fused mean -> gemma norm -> MRoPE -> compressed-cache store;
        a None source_keys/source_rope reads the members from the pending ring."""
        from sglang.kernels.ops.attention.qsa_indexer import (
            qsa_index_k_compress_store,
        )

        if source_keys is None:
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
        if source_rope is None:
            source_rope = pool.qsa_rope_position_buffer
        compressed_buffer = pool.get_qsa_compressed_k_buffer(self.layer_id)
        if getattr(pool, "qsa_index_fp8", False):
            # arc-forge h21: e4m3 keys + power-of-two scales, one kv head
            from sglang.kernels.ops.attention.qsa_indexer import (
                qsa_index_k_compress_store_fp8,
            )

            qsa_index_k_compress_store_fp8(
                source_keys.reshape(source_keys.shape[0], -1)
                .contiguous()
                .to(pool.index_state_dtype),
                group_locs.to(torch.int32),
                source_rope,
                self.rotary_emb.cos_sin_cache,
                self._rope_axis_map(source_keys.device),
                self.k_layernorm.weight.data,
                write_locs.to(torch.int32),
                compressed_buffer.view(compressed_buffer.shape[0], -1),
                pool.get_qsa_compressed_k_scale(self.layer_id).view(-1),
                self.compress_ratio,
                self.rotary_emb.rotary_dim,
                self.k_layernorm.variance_epsilon,
                self.rotary_emb.is_neox_style,
            )
            return
        qsa_index_k_compress_store(
            source_keys.reshape(source_keys.shape[0], -1)
            .contiguous()
            .to(pool.index_state_dtype),
            group_locs.to(torch.int32),
            source_rope,
            self.rotary_emb.cos_sin_cache,
            self._rope_axis_map(source_keys.device),
            self.k_layernorm.weight.data,
            write_locs.to(torch.int32),
            compressed_buffer.view(compressed_buffer.shape[0], -1),
            self.compress_ratio,
            self.rotary_emb.rotary_dim,
            self.k_layernorm.variance_epsilon,
            self.rotary_emb.is_neox_style,
        )

    def _pending_ring_slots(
        self, metadata, logical_positions: torch.Tensor, is_extend: bool
    ) -> torch.Tensor:
        return build_pending_ring_slots(
            token_to_batch_idx=metadata.token_to_batch_idx,
            req_pool_indices=metadata.req_pool_indices,
            sequence_lengths=metadata.sequence_lengths,
            logical_positions=logical_positions,
            compress_ratio=self.compress_ratio,
            is_extend=is_extend,
            ring_size=metadata.token_to_kv_pool.qsa_pending_ring_size,
        )

    def _group_ring_slots(
        self, metadata, group_end_positions: torch.Tensor, sequence_ids: torch.Tensor
    ) -> torch.Tensor:
        return build_group_ring_slots(
            req_pool_indices=metadata.req_pool_indices,
            group_end_positions=group_end_positions,
            sequence_ids=sequence_ids,
            compress_ratio=self.compress_ratio,
            ring_size=metadata.token_to_kv_pool.qsa_pending_ring_size,
        )

    def update_key_state_and_compress(
        self,
        token_k: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
        metadata,
        state_slots: torch.Tensor | None = None,
        state_stored: bool = False,
        head_keys: torch.Tensor | None = None,
    ) -> None:
        """Store the pending-group ring and compress each completed group.

        head_keys: the ring-held head keys of a ring-resumed extend
        (metadata.resume_head_ring_slots), read before this layer's ring store."""

        pool = metadata.token_to_kv_pool
        is_extend = metadata.compress_member_rows is not None
        if (head_keys is None) != (metadata.resume_head_ring_slots is None):
            raise RuntimeError(
                "QSA ring resume: head keys and resume_head_ring_slots must come together"
            )
        if not state_stored:
            if state_slots is None:
                state_slots = self._pending_ring_slots(
                    metadata, logical_positions, is_extend
                )
            pool.set_qsa_key_state_buffer(
                self.layer_id, state_slots[: token_k.shape[0]], token_k
            )
            pool.set_qsa_rope_position_buffer(
                state_slots[: token_k.shape[0]], rope_positions
            )

        if metadata.is_cuda_graph:
            self._compress_decode_cuda_graph(metadata)
            return

        if metadata.write_locs is None:
            raise RuntimeError(
                "QSA metadata is missing the precomputed write plan; the "
                "sparse-attention backend derives it from the batch lengths"
            )
        if metadata.write_locs.numel() == 0:
            return
        group_end_positions = metadata.compress_group_positions.long()
        compressed_locs = metadata.write_locs
        if is_extend and metadata.compress_member_locs is not None:
            # Ring resume: a row's first group starts in its pending ring. Compress
            # from the chunk keys followed by the head keys read before the store;
            # member_locs and extend_rope_matrix (chunk rows, then head rows) index
            # that source. Same members, same order, same kernel as a group that
            # lies wholly in one chunk.
            if head_keys is None:
                raise RuntimeError("QSA ring resume compress needs the head keys")
            num_chunk_rows = metadata.token_to_batch_idx.numel()
            if token_k.shape[0] != num_chunk_rows:
                raise RuntimeError(
                    f"QSA ring resume: {token_k.shape[0]} chunk keys, "
                    f"{num_chunk_rows} mapped rows"
                )
            source_keys = torch.cat(
                [token_k.to(pool.index_state_dtype), head_keys.to(pool.index_state_dtype)]
            )
            group_locs = metadata.compress_member_locs.long().clamp_max(
                source_keys.shape[0] - 1
            )
            source_rope = metadata.extend_rope_matrix
            if source_rope is None or source_rope.shape[0] != source_keys.shape[0]:
                raise RuntimeError(
                    "QSA ring resume: extend_rope_matrix must hold the chunk and head "
                    f"coordinates ({source_keys.shape[0]} rows), got "
                    f"{None if source_rope is None else source_rope.shape[0]}"
                )
        elif is_extend:
            # Extend chunks are group-aligned; each planned group lies in this forward,
            # so read its members from the packed chunk tensors.
            member_rows = metadata.compress_member_rows.long()
            group_locs = member_rows[:, None] + torch.arange(
                self.compress_ratio, device=member_rows.device, dtype=torch.long
            )
            source_keys = token_k
            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)
            source_rope = metadata.extend_rope_matrix
            if source_rope is None:
                source_rope = build_rope_position_matrix(
                    rope_positions, token_k.shape[0]
                )
        else:
            # Paged eager rows (speculative fallback) complete at most one
            # group each; its members are exactly the pending ring window.
            group_locs = metadata.compress_group_ring_locs
            if group_locs is None:
                group_locs = self._group_ring_slots(
                    metadata,
                    group_end_positions,
                    metadata.compress_sequence_ids.long(),
                )
            source_keys = pool.get_qsa_key_state_buffer(self.layer_id)
            source_rope = pool.qsa_rope_position_buffer
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                compressed_locs,
                source_keys=source_keys,
                source_rope=source_rope,
            )
            return
        key_groups = source_keys[group_locs]
        pooled = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            source_rope[group_locs[:, 0]]
        )
        normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
        pool.set_qsa_compressed_k_buffer(self.layer_id, compressed_locs, normalized)

    def _compress_decode_cuda_graph(self, metadata) -> None:
        """Fixed-shape graph-replay compression; non-boundary rows write slot 0."""

        if metadata.graph_write_locs is None or metadata.graph_ring_group_locs is None:
            raise RuntimeError("QSA CUDA graph compression metadata is incomplete")
        pool = metadata.token_to_kv_pool
        group_locs = metadata.graph_ring_group_locs
        if self._use_fused_compress(pool):
            self._fused_compress_store(
                pool,
                group_locs,
                metadata.graph_write_locs,
            )
            return
        key_groups = pool.get_qsa_key_state_buffer(self.layer_id)[group_locs]
        compressed = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            pool.qsa_rope_position_buffer[group_locs[:, 0]]
        )
        compressed = self.normalize_compressed_keys(
            compressed, compressed_rope_positions
        )
        pool.set_qsa_compressed_k_buffer(
            self.layer_id, metadata.graph_write_locs, compressed.contiguous()
        )

    def _rope_from_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """[n, 3] slot coordinates -> the layout apply_rope expects."""
        positions = positions.transpose(0, 1)
        if not getattr(self.rotary_emb, "mrope_section", None):
            return positions[0]
        return positions

    def apply_rope(self, positions: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        positions = positions.long()
        num_positions = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if num_positions != tensor.shape[0]:
            raise ValueError("QSA RoPE positions must match the token dimension")
        if not get_is_capture_mode() and hasattr(
            self.rotary_emb, "_ensure_cos_sin_cache_length"
        ):
            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))

        # position_cos/position_sin repeat cos/sin to the full rotary width;
        # apply_rotary_emb consumes one half.
        self.rotary_emb.get_cos_sin_with_position(positions)
        rotary_dim = self.rotary_emb.rotary_dim
        half_rotary_dim = rotary_dim // 2
        cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[
            :, :half_rotary_dim
        ]
        rotated = apply_rotary_emb(
            tensor[..., :rotary_dim],
            cos,
            sin,
            self.rotary_emb.is_neox_style,
        )
        return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)

    def select_prefill_tokens(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths_for_rows: torch.Tensor,
    ) -> torch.Tensor:
        rows = q.shape[0]
        output = torch.empty(
            (rows, self.token_topk + self.compress_ratio - 1),
            dtype=torch.int32,
            device=q.device,
        )
        if rows == 0:
            return output

        row_chunk_size = _qsa_prefill_row_chunk_size(
            rows, compressed_keys.shape[0], q.shape[1]
        )
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            chunk_slice = slice(row_start, row_end)
            if compressed_keys.shape[0] == 0:
                block_indices = torch.full(
                    (row_end - row_start, self.block_topk),
                    -1,
                    dtype=torch.int32,
                    device=q.device,
                )
                logits = None
            else:
                logits = qsa_mqa_prefill(
                    q[chunk_slice],
                    compressed_keys,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                )
                block_indices = qsa_fast_topk(
                    logits,
                    row_starts[chunk_slice],
                    row_ends[chunk_slice],
                    topk=self.block_topk,
                )
                if _ARC_QSA_TOPK_SORT:
                    block_indices = _sort_topk_blocks(block_indices)
            selected = expand_qsa_block_indices(
                block_indices,
                query_positions[chunk_slice],
                sequence_lengths_for_rows[chunk_slice],
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
            )
            output[chunk_slice].copy_(selected)
            del logits, block_indices, selected
        return output

    def select_prefill_tokens_local(
        self,
        q: torch.Tensor,
        compressed_keys: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths_for_rows: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        """ARC_QSA_INDEXER_LOCAL: select_prefill_tokens with each row's logits in its own
        sequence's coordinates ([rows, longest visible] instead of [rows, all keys of the
        batch]), written once by a column-split kernel that applies the scale and the -inf
        mask itself (no zero fill, divide or mask pass). The scores in a row's range are the
        same bits as the global layout's, so the top-k is too (exactly so under
        ARC_QSA_TOPK_EXACT=1; the production kernel's order and tie choice vary run to run
        either way)."""
        extend_lens = metadata.prefill_extend_lens_cpu
        seq_lens = metadata.prefill_seq_lens_cpu
        if extend_lens is None or seq_lens is None:
            raise RuntimeError("ARC_QSA_INDEXER_LOCAL needs the extend's host lengths in the indexer metadata")
        rows = q.shape[0]
        if sum(extend_lens) != rows:
            raise RuntimeError(f"ARC_QSA_INDEXER_LOCAL: {rows} query rows, host extend lengths sum to {sum(extend_lens)}")
        if not q.is_cuda:
            raise RuntimeError("ARC_QSA_INDEXER_LOCAL is a CUDA (TileLang) path")
        output = torch.empty(
            (rows, self.token_topk + self.compress_ratio - 1), dtype=torch.int32, device=q.device
        )
        if rows == 0:
            return output
        row_lens = (row_ends - row_starts).to(torch.int32)
        heads = q.shape[1]
        block_q = max(1, 128 // heads)
        _, width_all = qsa_prefill_local_tiles(
            extend_lens, seq_lens, self.compress_ratio, 0, rows, block_q
        )
        row_chunk_size = _qsa_prefill_row_chunk_size(rows, max(width_all, 1), heads)
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            chunk_slice = slice(row_start, row_end)
            tiles, width = qsa_prefill_local_tiles(
                extend_lens, seq_lens, self.compress_ratio, row_start, row_end, block_q
            )
            if compressed_keys.shape[0] == 0 or width == 0:
                block_indices = torch.full(
                    (row_end - row_start, self.block_topk), -1, dtype=torch.int32, device=q.device
                )
            else:
                logits = tilelang_qsa_mqa_prefill_local(
                    q[chunk_slice], compressed_keys, row_lens[chunk_slice], tiles, width
                )
                block_indices = qsa_fast_topk(
                    logits,
                    torch.zeros_like(row_lens[chunk_slice]),
                    row_lens[chunk_slice],
                    topk=self.block_topk,
                )
                if _ARC_QSA_TOPK_SORT:
                    block_indices = _sort_topk_blocks(block_indices)
                del logits
            output[chunk_slice].copy_(
                expand_qsa_block_indices(
                    block_indices,
                    query_positions[chunk_slice],
                    sequence_lengths_for_rows[chunk_slice],
                    compress_ratio=self.compress_ratio,
                    token_topk=self.token_topk,
                )
            )
        return output

    def select_decode_tokens(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        verify_queries_per_request: Optional[int] = None,
        compressed_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        block_indices = self._select_decode_blocks(
            q,
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            max_model_len,
            verify_queries_per_request=verify_queries_per_request,
            compressed_scale=compressed_scale,
        )
        return expand_qsa_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
        )

    def select_decode_tokens_hisparse(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        hisparse,
        req_pool_indices: torch.Tensor,
        pool,
        verify_queries_per_request: Optional[int] = None,
        compressed_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DECODE / TARGET_VERIFY selection under HiSparse offload: swap the
        selected blocks into each request's hot buffer, then expand them to the
        logical indices (returned, unchanged) and to device rows (written to
        hisparse.rows, which the sparse-attention gather reads). A verify forward
        (arc-forge M5 h16) has hisparse.rows_per_req = K rows per request,
        request-major; the swap-in resolves them one after another per request."""
        k = verify_queries_per_request or 1
        if hisparse.rows_per_req != k:
            raise ValueError(
                f"QSA HiSparse state has {hisparse.rows_per_req} rows per request, "
                f"the selection {k}"
            )
        block_indices = self._select_decode_blocks(
            q,
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            max_model_len,
            verify_queries_per_request=verify_queries_per_request,
            compressed_scale=compressed_scale,
        )
        rows = block_indices.shape[0]
        if (
            req_pool_indices.shape[0] != rows
            or hisparse.seq_blocks.shape[0] != rows
            or rows % k
        ):
            raise ValueError(
                "QSA HiSparse decode rows disagree: "
                f"top-k={rows}, req_pool_indices={req_pool_indices.shape[0]}, "
                f"seq_blocks={hisparse.seq_blocks.shape[0]}, rows per request {k}"
            )
        device_blocks = hisparse.coord.swap_in_selected_pages(
            req_pool_indices=req_pool_indices,
            compressed_seq_lens=hisparse.seq_blocks,
            top_k_result=block_indices,
            layer_id=pool.full_layer_index(self.layer_id),
            rows_per_req=k,
        )
        if _ARC_HISPARSE_QSA_DEBUG and not get_is_capture_mode():
            _check_hisparse_selection(
                block_indices, compressed_lengths, device_blocks, self.block_topk
            )
        logical, _ = expand_qsa_block_indices_hot(
            block_indices,
            device_blocks,
            query_positions,
            sequence_lengths,
            hisparse.newest_dev,
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
            out_rows=hisparse.rows,
        )
        return logical

    def _select_decode_blocks(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        verify_queries_per_request: Optional[int] = None,
        compressed_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_fast_decode_topk = q.is_cuda and self.block_topk == 512
        logits = qsa_mqa_decode(
            q,
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            max_model_len,
            verify_queries_per_request=verify_queries_per_request,
            fill_invalid_tail=not use_fast_decode_topk,
            k_scale=compressed_scale,
        )
        if use_fast_decode_topk:
            # Decode rows start at zero, so compressed lengths double as row lengths;
            # skip the generic zero-fill + subtract.
            from sglang.kernels.ops.elementwise.fast_topk import fast_topk

            block_indices = fast_topk(
                logits,
                compressed_lengths.to(torch.int32),
                topk=self.block_topk,
                row_starts=None,
            )
        else:
            row_starts = torch.zeros_like(compressed_lengths, dtype=torch.int32)
            block_indices = qsa_fast_topk(
                logits, row_starts, compressed_lengths, topk=self.block_topk
            )
        if _ARC_QSA_TOPK_SORT:
            block_indices = _sort_topk_blocks(block_indices)
        return block_indices

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        indexer_metadata,
    ) -> torch.Tensor:
        forward_mode = forward_batch.forward_mode
        is_target_verify = getattr(forward_mode, "is_target_verify", lambda: False)()
        is_draft_extend = getattr(forward_mode, "is_draft_extend_v2", lambda: False)()
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            # EAGLE/MTP may advance the model's RoPE coordinate independently
            # from the physical paged-KV position.  Compression and sparse
            # selection must use the latter; otherwise a draft step can index
            # token_slot_table[:, seq_len] one past the valid range.
            logical_positions = indexer_metadata.decode_logical_positions
            if logical_positions is None:
                logical_positions = indexer_metadata.get_seqlens_expanded() - 1
        else:
            logical_positions = getattr(forward_batch, "positions", None)
            if logical_positions is None:
                logical_positions = positions[0] if positions.ndim == 2 else positions
            logical_positions = logical_positions.flatten()
        # DP MAX_LEN padding adds token rows without assigning them to a
        # request. token_to_batch_idx is the source of truth for semantic rows.
        num_valid_tokens = indexer_metadata.get_token_to_batch_idx().numel()
        if logical_positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA logical positions are shorter than the request mapping: "
                f"positions={logical_positions.numel()}, mapping={num_valid_tokens}"
            )
        if hidden_states.shape[0] < num_valid_tokens:
            raise ValueError(
                "QSA hidden states are shorter than the request mapping: "
                f"hidden={hidden_states.shape[0]}, mapping={num_valid_tokens}"
            )
        position_tokens = (
            positions.shape[-1] if positions.ndim == 2 else positions.numel()
        )
        if position_tokens < num_valid_tokens:
            raise ValueError(
                "QSA RoPE positions are shorter than the request mapping: "
                f"positions={position_tokens}, mapping={num_valid_tokens}"
            )

        logical_positions = logical_positions[:num_valid_tokens]
        hidden_states = hidden_states[:num_valid_tokens]
        positions = (
            positions[:, :num_valid_tokens]
            if positions.ndim == 2
            else positions[:num_valid_tokens]
        )
        state_slots = indexer_metadata.pending_ring_slots
        if state_slots is None:
            state_slots = self._pending_ring_slots(
                indexer_metadata,
                logical_positions,
                indexer_metadata.compress_member_rows is not None,
            )
        head_keys = None
        if indexer_metadata.resume_head_ring_slots is not None:
            # Ring resume: read the head before project_qk / the compress step store
            # this chunk's pending tail, which can map onto the same ring slots.
            head_keys = indexer_metadata.token_to_kv_pool.get_qsa_key_state_buffer(
                self.layer_id
            ).index_select(0, indexer_metadata.resume_head_ring_slots)
        q, token_k, state_stored = self.project_qk(
            hidden_states,
            positions,
            pool=indexer_metadata.token_to_kv_pool,
            cache_loc=state_slots,
            q_heads_padded=(
                # The tilelang decode MQA kernel needs query heads in multiples of 8.
                ((self.index_n_heads + 7) // 8) * 8
                if (forward_mode.is_decode() or is_target_verify or is_draft_extend)
                else None
            ),
            max_position=indexer_metadata.max_position_cpu,
        )
        self.update_key_state_and_compress(
            token_k,
            logical_positions,
            positions,
            indexer_metadata,
            state_slots=state_slots,
            state_stored=state_stored,
            head_keys=head_keys,
        )
        if forward_mode.is_decode() or is_target_verify or is_draft_extend:
            compressed_cache, page_table, compressed_lengths, max_model_len = (
                indexer_metadata.get_decode_mqa_inputs(self.layer_id)
            )
            compressed_scale = indexer_metadata.get_decode_mqa_scale(self.layer_id)
            hisparse = indexer_metadata.hisparse
            if hisparse is not None:
                if not (forward_mode.is_decode() or is_target_verify):
                    raise NotImplementedError(
                        f"QSA HiSparse offload supports DECODE and TARGET_VERIFY, got {forward_mode}"
                    )
                return self.select_decode_tokens_hisparse(
                    q,
                    compressed_cache,
                    page_table,
                    compressed_lengths,
                    max_model_len,
                    logical_positions,
                    indexer_metadata.get_seqlens_int32(),
                    hisparse=hisparse,
                    req_pool_indices=indexer_metadata.req_pool_indices,
                    pool=indexer_metadata.token_to_kv_pool,
                    verify_queries_per_request=_qsa_grouped_queries_per_request(
                        forward_batch,
                        is_target_verify=is_target_verify,
                        is_draft_extend=False,
                    ),
                    compressed_scale=compressed_scale,
                )
            return self.select_decode_tokens(
                q,
                compressed_cache,
                page_table,
                compressed_lengths,
                max_model_len,
                logical_positions,
                indexer_metadata.get_seqlens_int32(),
                verify_queries_per_request=_qsa_grouped_queries_per_request(
                    forward_batch,
                    is_target_verify=is_target_verify,
                    is_draft_extend=is_draft_extend,
                ),
                compressed_scale=compressed_scale,
            )

        compressed_keys, row_starts, row_ends, sequence_lengths = (
            indexer_metadata.get_prefill_mqa_inputs(self.layer_id, logical_positions)
        )
        if getattr(indexer_metadata.token_to_kv_pool, "qsa_index_fp8", False) and q.shape[0]:
            # arc-forge h21: prefill scores the Q the FP8 decode kernels score (quantized per
            # (row, head) with the keys' power-of-two rule; dequantized exactly into BF16)
            from sglang.srt.layers.attention.qsa.mqa import quantize_qsa_index_q_fp8
            from sglang.srt.mem_cache.qsa_kv_pool import dequantize_qsa_index_k

            q = dequantize_qsa_index_k(*quantize_qsa_index_q_fp8(q.contiguous()))
        query_sequence_ids = indexer_metadata.get_token_to_batch_idx()
        row_sequence_lengths = sequence_lengths.index_select(
            0, query_sequence_ids.long()
        )
        if _ARC_QSA_INDEXER_LOCAL:
            return self.select_prefill_tokens_local(
                q,
                compressed_keys,
                row_starts,
                row_ends,
                logical_positions,
                row_sequence_lengths,
                indexer_metadata,
            )
        return self.select_prefill_tokens(
            q,
            compressed_keys,
            row_starts,
            row_ends,
            logical_positions,
            row_sequence_lengths,
        )


__all__ = [
    "QSAIndexer",
]
