from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.runtime_context import get_model

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    # Why the served weights cannot be trusted -- a keep_identity load failed
    # after it started writing -- or None. While set, generate requests are
    # refused (here and in the TokenizerManager); a keep_identity update that
    # succeeds clears it.
    poisoned: Optional[str] = None

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    @contextmanager
    def _fenced_weight_write(self, source: str) -> Iterator[None]:
        """Write weights only while no forward runs.

        Under the overlap scheduler, the batch launched before a
        ``pause_generation(mode="in_place")`` can still be on the forward stream
        when the update arrives, and the update writes the weights from the
        schedule stream (or a loader's own): a device-wide sync first keeps that
        forward entirely on the old weights. The sync after keeps the next
        forward off weights a loader is still writing on another stream. Both
        cost microseconds next to a load.
        """
        torch.get_device_module().synchronize()
        with self._observe_weight_load(source):
            yield
        torch.get_device_module().synchronize()

    def _on_every_rank(self, ok: bool) -> bool:
        """Whether ``ok`` holds on every rank of the TP group. A keep_identity
        decision (refuse, poison, succeed) is taken by all ranks together, or
        ranks would serve different weights."""
        flag = torch.tensor([int(ok)], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        return bool(flag.item())

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def record_weight_version_after_update(self, weight_version: Optional[str]) -> None:
        self.scheduler.record_weight_version_change(new_version=weight_version)

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        if recv_req.keep_identity:
            return self.update_weights_from_disk_keep_identity(recv_req)
        with self._fenced_weight_write("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if (
                tp_success
                and recv_req.flush_cache
                and not self.flush_cache(empty_cache=recv_req.torch_empty_cache)
            ):
                # Not an assert: a weight update sent while requests run (e.g.
                # under pause_generation(mode="in_place")) must not kill the
                # scheduler. The new weights stay loaded; the version and the
                # model path are not advanced.
                success, message = False, (
                    "Weights were loaded from disk but the cache was not flushed "
                    "(requests are running, queued or paused in place); the weight "
                    "version and model path were not updated."
                )
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success,
                message=message,
                num_paused_requests=0,
                weights_poisoned=self.poisoned,
            )

    def update_weights_from_disk_keep_identity(
        self, recv_req: UpdateWeightFromDiskReqInput
    ) -> UpdateWeightFromDiskReqOutput:
        """arc-forge RL weight sync: load ``recv_req.model_path`` (any subset of
        the model's tensors) into the serving model, keeping the server's
        identity, with no rollback.

        - Refused, nothing loaded, poisoned state unchanged: under speculative
          decoding (a draft runner would reload and re-postprocess its own
          weights from a delta that does not carry them), or with
          ``flush_cache`` while the scheduler is not fully idle on every rank.
          Without ``flush_cache`` there is no idle requirement: the mid-flight
          sync is ``pause_generation(mode="in_place")`` -> this ->
          ``continue_generation``, and the paused requests go on with the new
          weights over their cached (old-weight) KV.
        - A load that fails on any rank after it started writing poisons the
          weights on every rank: every request is aborted and generate is
          refused until a later keep_identity update succeeds (it must write
          every tensor the failed one may have written -- true for merged
          exports, which always carry the same keys).
        - Success clears the poisoned state and records ``weight_version``
          (tokens of requests alive across the update are attributed per span;
          under the overlap scheduler the batch in flight at the pause samples
          one more token per request with the old weights, which is attributed
          to the new version).
        """

        def refuse(message: str) -> UpdateWeightFromDiskReqOutput:
            logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=False, message=message, weights_poisoned=self.poisoned
            )

        if self.draft_worker is not None:
            return refuse(
                "keep_identity: speculative decoding is not supported (the draft "
                "would reload and re-postprocess its weights from a delta that "
                "does not carry them); nothing was loaded."
            )
        if recv_req.flush_cache and not self._on_every_rank(self.is_fully_idle()):
            return refuse(
                "keep_identity with flush_cache needs an idle scheduler (requests "
                "are running, queued or paused in place); nothing was loaded. "
                "Retry when idle, or sync mid-flight with flush_cache=false."
            )
        with self._fenced_weight_write("disk"):
            success, message, touched = (
                self.tp_worker.update_weights_from_disk_keep_identity(recv_req)
            )
            if not self._on_every_rank(success):
                if self._on_every_rank(not touched):
                    return refuse(
                        "keep_identity: the update failed before any weight was "
                        f"written on any rank; nothing changed. This rank: {message}"
                    )
                return self._poison(
                    f"the update from {recv_req.model_path} failed on a TP rank "
                    f"after loading started. This rank: {message}"
                )
            if recv_req.flush_cache and not self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            ):
                # Idle was checked above and nothing ran since: a server bug.
                return self._poison(
                    f"the cache flush failed after loading {recv_req.model_path} "
                    "although the scheduler was idle"
                )
        self.poisoned = None
        self.record_weight_version_after_update(recv_req.weight_version)
        logger.info(
            "keep_identity weight update from %s done (weight_version=%s)",
            recv_req.model_path,
            recv_req.weight_version,
        )
        return UpdateWeightFromDiskReqOutput(
            success=True, message=message, weights_poisoned=None
        )

    def _poison(self, reason: str) -> UpdateWeightFromDiskReqOutput:
        self.poisoned = reason
        message = f"Weights poisoned: {reason}"
        logger.error(
            "%s. Aborting every request; generate is refused until a "
            "keep_identity update succeeds.",
            message,
        )
        self.scheduler.abort_request(AbortReq(abort_all=True, abort_message=message))
        return UpdateWeightFromDiskReqOutput(
            success=False, message=message, weights_poisoned=reason
        )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._fenced_weight_write("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._fenced_weight_write("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._fenced_weight_write("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if success:
                self.record_weight_version_after_update(recv_req.weight_version)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _assert_weight_cache_inactive(self, op: str) -> None:
        """Reject freeing/restoring model weights while the CUDA IPC weight
        cache is active: the weights are shared with the daemon via CUDA IPC, so
        freeing them would leave the daemon and every peer pointing at released
        memory.
        """
        mode = get_model().weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self.is_fully_idle(), (
            "release_memory_occupation should be called only when server is idle."
        )

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("release_memory_occupation")
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("resume_memory_occupation")
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert draft_url is not None, (
                "draft_url must be provided when draft model is enabled"
            )
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
