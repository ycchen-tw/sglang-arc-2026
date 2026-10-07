import dataclasses
import logging
import math
from typing import Any, Dict, Optional, Set

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
    max_prefill_buffer_tokens,
    max_speculative_num_draft_tokens,
)
from sglang.srt.state_capturer.base import (
    BaseHostCache,
    TopkCaptureOutput,
    get_tensor_size_bytes,
)
from sglang.srt.utils.common import next_power_of_2

logger = logging.getLogger(__name__)

# Routed-expert ids are logical (pre-EPLB) global expert ids, so int16 holds
# any model with < 32768 experts at half the host/IPC footprint of int32. The
# FlashInfer TRT-LLM routing kernels write this dtype natively
# (routing_replay_out), and every consumer reads it from
# meta_info["routed_experts_meta"]["dtype"].
ROUTED_EXPERTS_DTYPE = torch.int16
ROUTED_EXPERTS_DTYPE_NAME = "int16"

# A routing row no forward wrote. The host buffer starts at it, and every target
# forward resets its device rows to it before the MoE layers run
# (on_forward_begin), so a token a capture site skipped reads back as -1 -- not
# as expert 0, and not as the previous occupant of its KV slot. Expert ids are
# >= 0.
ROUTING_UNWRITTEN = -1

# A real top-k row holds top_k *distinct* ids, so a row that holds the sentinel
# or is all zeros was never written by any capture site. The first forwards that
# carry tokens are scanned for such rows; the check is a one-time guard against
# a backend that silently skips the capture, not a per-step cost. Per request,
# invalid_rows() checks every finished request's rows.
_NEVER_WRITTEN_ROW_CHECK_FORWARDS = 32

_EXPERT_COUNT_KEYS = (
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "moe_num_experts",
)


def _is_scattered_a2a_backend() -> bool:
    """Return whether routed tokens are scattered across attention-TP ranks."""
    backend = get_moe_a2a_backend()
    return backend.is_deepep() or backend.is_deepep_v2()


def is_routed_experts_output_rank() -> bool:
    """Whether this rank's scheduler ships request outputs to the detokenizer.

    Mirrors ``Scheduler.init_ipc_channels``' ``is_rank_zero``: only these ranks
    ever return ``meta_info["routed_experts"]``, so only they need the host
    buffer, the per-forward D2H, and the host scatter. Every TP rank computes
    the same routing (replicated router over all-reduced hidden states), and
    under DP attention each DP group's attn-TP rank 0 is an output rank over
    its own KV pool.
    """
    parallel = get_parallel()
    return (
        parallel.pp_rank == 0
        and parallel.attn_tp_rank == 0
        and parallel.attn_cp_rank == 0
    )


def _num_logical_experts(hf_text_config: Any) -> int:
    for key in _EXPERT_COUNT_KEYS:
        value = getattr(hf_text_config, key, None)
        if value is not None:
            return int(value)
    raise RuntimeError(
        "--enable-return-routed-experts: cannot find the routed expert count on "
        f"the model config (looked for {_EXPERT_COUNT_KEYS}); it is needed to "
        f"prove expert ids fit {ROUTED_EXPERTS_DTYPE_NAME}."
    )


def _max_tokens_per_forward(max_running_requests: int) -> int:
    """Upper bound on the MoE rows of one target forward on this rank.

    Covers a full prefill chunk (or a whole non-chunked prefill batch) plus,
    for mixed chunks, every running request's decode / verify tokens.
    """
    schedule = get_schedule()
    prefill_tokens = max(
        schedule.chunked_prefill_size or 0,
        max_prefill_buffer_tokens() or 0,
        schedule.max_prefill_tokens or 0,
    )
    decode_tokens = max_running_requests * (max_speculative_num_draft_tokens() or 1)
    return prefill_tokens + decode_tokens


@dataclasses.dataclass
class RoutedExpertsCaptureOutput(TopkCaptureOutput):
    """``TopkCaptureOutput`` whose finalize also runs the never-written check."""

    capturer: Optional["RoutedExpertsCapturer"] = None

    def finalize(self):
        if self.capturer is None:
            raise RuntimeError("RoutedExpertsCaptureOutput built without capturer")
        self.capturer.check_rows_written(self.topk)
        super().finalize()


class RoutedExpertsCapturer:
    """Records each token's routed experts, keyed by its KV slot.

    Device side (every rank): a layer-major ``int16 [num_layers, rows, K]``
    buffer allocated once, before CUDA-graph capture. Layer ``l``'s
    ``buffer[l]`` is a contiguous ``[rows, K]`` view, which is exactly the
    ``routing_replay_out`` layout the FlashInfer TRT-LLM routing kernels write
    for the logits-based (BYPASSED top-k) backends; backends that materialize
    top-k ids in SGLang (STANDARD) copy them in through :meth:`capture`.

    Host side (output ranks only, see :func:`is_routed_experts_output_rank`):
    a pinned ``int16 [kv_pool_tokens, num_layers, top_k]`` buffer indexed by KV
    slot, filled by a D2H + scatter after every target forward and gathered
    through ``req_to_token`` when a request finishes.
    """

    @staticmethod
    def create(
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["RoutedExpertsCapturer"]:
        if not get_exec().features.enable_return_routed_experts:
            return None
        if get_parallel().pp_size > 1:
            # Each PP stage would capture only its own layers while just
            # pp_rank 0 ships outputs: the other stages' layers would come back
            # as zeros.
            raise NotImplementedError(
                "--enable-return-routed-experts does not support pipeline "
                f"parallelism (pp_size={get_parallel().pp_size})."
            )
        # The model's own attribute is the baked decision (0 when its gate
        # disabled fusion); the ACTIVE flag can be holding another runner's.
        num_fused_shared_experts = getattr(model, "num_fused_shared_experts", 0)
        hf_text_config = model_config.hf_text_config
        # Scale by dp_size so the buffer covers the full DP-concatenated batch
        # the MoE sees under DP attention.
        max_batch_size = (
            _max_tokens_per_forward(max_running_requests) * get_parallel().dp_size
        )
        return RoutedExpertsCapturer(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=hf_text_config.num_hidden_layers,
            topk_size=hf_text_config.num_experts_per_tok,
            num_fused_shared_experts=num_fused_shared_experts,
            num_logical_experts=_num_logical_experts(hf_text_config),
            device=device,
            is_output_rank=is_routed_experts_output_rank(),
        )

    def __init__(
        self,
        *,
        num_tokens: int,
        max_batch_size: int,
        num_layers: int,
        topk_size: int,
        num_fused_shared_experts: int,
        num_logical_experts: int,
        device: str,
        is_output_rank: bool,
    ):
        max_id = num_logical_experts + num_fused_shared_experts - 1
        if max_id > torch.iinfo(ROUTED_EXPERTS_DTYPE).max:
            raise RuntimeError(
                f"routed expert ids reach {max_id}, which does not fit "
                f"{ROUTED_EXPERTS_DTYPE_NAME}"
            )
        self.num_layers = num_layers
        self.topk_size = topk_size
        self.num_fused_shared_experts = num_fused_shared_experts
        # Extra columns for fused shared experts are dropped before the host
        # copy via [:topk_size].
        self.device_topk_size = topk_size + num_fused_shared_experts
        self.is_output_rank = is_output_rank
        # FlashInfer's autotuner replays the routing kernel (with the same
        # routing_replay_out) at every power-of-two bucket up to
        # next_power_of_2(num_tokens); the buffer must cover the top bucket.
        self.device_rows = next_power_of_2(max_batch_size)
        self.device_buffer = torch.full(
            (num_layers, self.device_rows, self.device_topk_size),
            ROUTING_UNWRITTEN,
            dtype=ROUTED_EXPERTS_DTYPE,
            device=device,
        )
        self.host_cache: Optional[BaseHostCache] = (
            BaseHostCache(
                num_tokens,
                num_layers,
                topk_size,
                name="routed_experts",
                device=device,
                dtype=ROUTED_EXPERTS_DTYPE,
                fill_value=ROUTING_UNWRITTEN,
            )
            if is_output_rank
            else None
        )
        # Layers whose capture site ran at least once (Python side, so CUDA
        # graph capture counts). Empty after a forward with tokens means the
        # MoE backend never reached a capture site: the result would be zeros.
        self.layers_recorded: Set[int] = set()
        self._row_checks_left = _NEVER_WRITTEN_ROW_CHECK_FORWARDS

        # Rebuild the full token batch before routed-expert readback.
        if _is_scattered_a2a_backend():
            attn_tp_size = (
                get_parallel().attn_tp_size if is_dp_attention_enabled() else 1
            )
            self.gather_buffer = torch.empty(
                (self.device_rows * attn_tp_size, self.device_topk_size),
                dtype=torch.int32,
                device=device,
            )
        logger.info(
            "RoutedExpertsCapturer: device buffer %s %s (%.1f MB); host buffer %s",
            tuple(self.device_buffer.shape),
            ROUTED_EXPERTS_DTYPE_NAME,
            get_tensor_size_bytes(self.device_buffer) / 2**20,
            (
                f"{tuple(self.host_cache.buffer.shape)} "
                f"({get_tensor_size_bytes(self.host_cache.buffer) / 2**30:.2f} GB)"
                if self.host_cache is not None
                else "not allocated (this rank does not send outputs)"
            ),
        )

    # ------------------------------------------------------------------ capture

    def capture(self, layer_id: Optional[int], topk_indices: torch.Tensor):
        """Copy SGLang-materialized top-k ids (STANDARD formats) for a layer."""
        if layer_id is None:
            raise RuntimeError(
                "routed-experts capture needs the MoE layer id; this TopK was "
                "built without layer_id"
            )
        if _is_scattered_a2a_backend():
            local_topk = topk_indices
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_parallel().attn_tp_size
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)
        num_rows = topk_indices.shape[0]
        if num_rows > self.device_rows:
            raise RuntimeError(
                f"routed-experts capture: {num_rows} rows exceed the device "
                f"buffer ({self.device_rows})"
            )
        self.layers_recorded.add(layer_id)
        self.device_buffer[layer_id, :num_rows].copy_(topk_indices)

    def routing_replay_out(self, layer_id: Optional[int], num_tokens: int):
        """This layer's ``int16 [rows, top_k]`` buffer for a routing kernel to
        write its selected (global, logical) expert ids into.

        Static storage, so it is safe to bake into a CUDA graph.
        """
        if layer_id is None:
            raise RuntimeError(
                "routing_replay_out needs the MoE layer id; this TopK was built "
                "without layer_id"
            )
        if self.num_fused_shared_experts:
            raise RuntimeError(
                "routing_replay_out cannot record fused shared experts "
                f"(num_fused_shared_experts={self.num_fused_shared_experts})"
            )
        if _is_scattered_a2a_backend():
            raise NotImplementedError(
                "routing_replay_out with a scattered (DeepEP) A2A backend"
            )
        if num_tokens > self.device_rows:
            # The kernel does not bound-check; this would write past the buffer.
            raise RuntimeError(
                f"routing_replay_out: {num_tokens} tokens exceed the device "
                f"buffer ({self.device_rows} rows)"
            )
        self.layers_recorded.add(layer_id)
        return self.device_buffer[layer_id]

    # ----------------------------------------------------------------- readback

    def on_forward_begin(self, forward_batch: ForwardBatch) -> None:
        """Reset this forward's device rows to ROUTING_UNWRITTEN before the MoE
        layers run (output ranks only: nothing reads the others' rows).

        One memset of ``L * n * K`` int16 on the current (forward) stream, outside
        any CUDA graph; the previous forward's rows were already copied out by
        its on_forward_end, earlier on the same stream. A split prefill resets
        only before its first split, so the later splits' layers are kept.
        """
        if self.host_cache is None or forward_batch.split_index != 0:
            return
        if is_dp_attention_enabled():
            # This rank reads a slice of the DP-gathered batch whose offset is
            # only known after the forward: reset every row.
            self.device_buffer.fill_(ROUTING_UNWRITTEN)
        else:
            num_tokens = forward_batch.out_cache_loc.shape[0]
            self.device_buffer[:, :num_tokens].fill_(ROUTING_UNWRITTEN)

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        """This rank's rows of the last forward, layer-major ``[L, n, K]``."""
        # Gathered rows start at buffer offset zero on every DP rank.
        if is_dp_attention_enabled() and not _is_scattered_a2a_backend():
            # GPU->CPU sync would break overlap; operate on CPU directly.
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.device_buffer[
            :, local_start_pos:local_end_pos, : self.topk_size
        ]

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[RoutedExpertsCaptureOutput]:
        """Hand this forward's rows to the host buffer (output ranks only).

        With ``no_copy_to_cpu`` return GPU tensors for the overlap thread's
        non-blocking D2H + finalize; otherwise D2H and scatter inline.
        """
        if forward_batch.out_cache_loc.shape[0] > 0 and not self.layers_recorded:
            raise RuntimeError(
                "--enable-return-routed-experts: a target forward with tokens "
                "finished but no MoE layer reached a routed-experts capture site "
                "(neither SGLang top-k nor a routing_replay_out kernel); every "
                "returned routing would be zeros. The active MoE runner backend / "
                "top-k format does not support routed-experts capture."
            )
        if self.host_cache is None:
            return None
        # [L, n, K] -> [n, L, K]; .contiguous() materializes a fresh tensor, so
        # the next (overlapped) forward cannot overwrite it.
        topk = self._get_local_slice(
            forward_batch, can_run_graph, cuda_graph_batch
        ).transpose(0, 1).contiguous()
        if no_copy_to_cpu:
            return RoutedExpertsCaptureOutput(
                out_cache_loc=forward_batch.out_cache_loc.clone(),
                topk=topk,
                host_cache=self.host_cache,
                capturer=self,
            )
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        topk_cpu = topk.cpu()
        self.check_rows_written(topk_cpu)
        self.host_cache.buffer[out_cache_loc_cpu] = topk_cpu
        return None

    def check_rows_written(self, topk_cpu: torch.Tensor) -> None:
        """Raise if a recorded layer left a token's row unwritten (the
        ROUTING_UNWRITTEN sentinel, or all zeros)."""
        if self._row_checks_left <= 0 or topk_cpu.shape[0] == 0:
            return
        self._row_checks_left -= 1
        layers = sorted(self.layers_recorded)
        rows = topk_cpu[:, layers, :]
        never_written = (rows == ROUTING_UNWRITTEN).any(dim=-1) | (rows == 0).all(
            dim=-1
        )
        if bool(never_written.any()):
            token, layer_pos = never_written.nonzero()[0].tolist()
            raise RuntimeError(
                "--enable-return-routed-experts: routed experts of token row "
                f"{token}, layer {layers[layer_pos]} were never written (the "
                f"{ROUTING_UNWRITTEN} sentinel or all {self.topk_size} ids 0; "
                f"{int(never_written.sum())} such (token, layer) rows in this "
                "forward). The MoE kernel accepted the capture buffer but did not "
                "fill it."
            )

    def invalid_rows(self, rows: torch.Tensor) -> Optional[str]:
        """Why a finished request's gathered rows ``[n, L, K]`` are not its
        routing, or None.

        Over the recorded layers (a layer no capture site runs in keeps the
        sentinel by design): a negative id is ROUTING_UNWRITTEN -- no forward
        wrote that row since its KV slot was allocated as host memory or reset
        as device rows -- and a row whose first two ids are both 0 is not a
        top-k of distinct ids (the all-zero row a capture that wrote nothing
        leaves). The common case is two vectorized passes over the rows, ~15% of
        the gather itself (20k tokens x 48 layers x top-10: 1.3 ms vs 8.4 ms).
        A per-request problem, so the caller drops this request's routing; it
        does not stop the server.
        """
        if rows.shape[0] == 0:
            return None
        if len(self.layers_recorded) != self.num_layers:
            layers = sorted(self.layers_recorded)
            rows = rows[:, layers]
        else:
            layers = list(range(self.num_layers))
        two_zeros = (
            (rows[..., 0] == 0) & (rows[..., 1] == 0)
            if self.topk_size > 1
            else torch.zeros(rows.shape[:2], dtype=torch.bool)
        )
        if not bool(rows.min() < 0) and not bool(two_zeros.any()):
            return None
        unwritten = (rows < 0).any(dim=-1)
        zero = (rows == 0).all(dim=-1) if self.topk_size > 1 else two_zeros
        bad = unwritten | zero
        if not bool(bad.any()):
            # two zero ids but not all: a duplicate, which the consumer's own
            # top-k validation reports.
            return None
        token, layer_pos = bad.nonzero()[0].tolist()
        return (
            f"invalid_rows: {int(unwritten.sum())} unwritten ({ROUTING_UNWRITTEN}) "
            f"and {int(zero.sum())} all-zero (token, layer) rows of "
            f"{rows.shape[0]}x{len(layers)}; first at row {token}, layer "
            f"{layers[layer_pos]}"
        )

    @staticmethod
    def effective_start(start_len: int, seqlen: int) -> int:
        if start_len < 0:
            raise ValueError(f"{start_len=} must be non-negative")
        return min(start_len, seqlen - 1)

    def get_topk(
        self,
        req_pool_idx: int,
        seqlen: int,
        req_to_token_pool: ReqToTokenPool,
        start_len: int = 0,
    ) -> torch.Tensor:
        """``int16 [seqlen - 1 - start, L, K]`` for tokens ``[start, seqlen - 1)``
        where ``start = effective_start(start_len, seqlen)``."""
        if self.host_cache is None:
            raise RuntimeError(
                "routed experts are only kept on output ranks; this rank must "
                "not collect them"
            )
        start_len = self.effective_start(start_len, seqlen)
        cache_pool_idx = (
            req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1]
            .cpu()
            .clone()
        )
        return self.host_cache.buffer[cache_pool_idx]

    @staticmethod
    def output_meta(routed_experts: torch.Tensor, start: int) -> Dict[str, Any]:
        """``meta_info["routed_experts_meta"]``: how to decode the base64 bytes.

        Row ``i`` is the routing of token ``start + i``.
        """
        if routed_experts.dtype != ROUTED_EXPERTS_DTYPE:
            raise RuntimeError(
                f"routed experts tensor is {routed_experts.dtype}, expected "
                f"{ROUTED_EXPERTS_DTYPE}"
            )
        return {
            "dtype": ROUTED_EXPERTS_DTYPE_NAME,
            "shape": [int(d) for d in routed_experts.shape],
            "start": int(start),
        }

    def destroy(self):
        if self.host_cache is not None:
            self.host_cache.destroy()


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:

    return get_resources().experts_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):

    get_resources().experts_capturer = capturer


def destroy_global_experts_capturer():
    if (capturer := get_resources().experts_capturer) is not None:
        capturer.destroy()
    get_resources().experts_capturer = None


def decode_routed_experts(
    routed_experts_base64: str, routed_experts_meta: Dict[str, Any]
) -> np.ndarray:
    """Decode ``meta_info["routed_experts"]`` with its ``routed_experts_meta``.

    Returns ``[n, num_layers, top_k]`` in the producer's dtype; row ``i`` is the
    routing of token ``routed_experts_meta["start"] + i``.
    """
    dtype = np.dtype(routed_experts_meta["dtype"])
    shape = tuple(int(d) for d in routed_experts_meta["shape"])
    flat = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=dtype
    )
    if flat.size != math.prod(shape):
        raise ValueError(
            f"routed_experts holds {flat.size} {dtype} values but "
            f"routed_experts_meta says shape {list(shape)}"
        )
    return flat.reshape(shape)


def extract_routed_experts_from_meta_info(data) -> np.ndarray:
    """``[n, num_layers, top_k]`` routed experts of one ``/generate`` result.

    The payload's dtype is declared by ``meta_info["routed_experts_meta"]``
    (int16 since the KV-slot host buffer shrank); a result without it is from
    a producer whose dtype cannot be known, so it is rejected rather than
    guessed.
    """
    meta_info = data["meta_info"]
    routed_experts_meta = meta_info.get("routed_experts_meta")
    if routed_experts_meta is None:
        raise ValueError(
            "meta_info has routed_experts but no routed_experts_meta; refusing "
            "to guess the dtype"
        )
    return decode_routed_experts(meta_info["routed_experts"], routed_experts_meta)


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft MoE ``TopK`` out of routed-experts (R3) capture.

    Capture is target-only; a draft ``TopK`` must never write the target's
    process-global buffer. ``HashTopK`` has no ``topk_config`` and never
    captures, so it is left untouched.
    """
    # Lazy import: ``layers.moe.topk`` imports ``get_global_experts_capturer``
    # from this module, so a top-level import here would be circular.
    from sglang.srt.layers.moe.topk import TopK

    for module in model.modules():
        if isinstance(module, TopK):
            module.topk_config.allow_routed_experts_capture = False
