"""h25: CPU consensus within one attention TP group; DP queues stay independent."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def distributed_hisparse_enabled() -> bool:
    value = os.environ.get("ARC_HISPARSE_DISTRIBUTED", "0")
    if value not in ("0", "1"):
        raise ValueError(f"ARC_HISPARSE_DISTRIBUTED must be 0 or 1, got {value!r}")
    return value == "1"


def distributed_shape_needs(view) -> dict[str, bool]:
    """Admit h25 fallback or the explicitly gated h29 NEXTN3 shape."""
    user = json.loads(view.hisparse_config) if view.hisparse_config else {}
    spec = distributed_spec_enabled()
    needs = {
        "ARC_HISPARSE_DISTRIBUTED: --tp-size 4 --ep-size 4":
            view.tp_size == 4 and view.ep_size == 4,
        "ARC_HISPARSE_DISTRIBUTED: --dp-size 2 --enable-dp-attention":
            view.dp_size == 2 and view.enable_dp_attention,
        "ARC_HISPARSE_DISTRIBUTED: one node": view.nnodes == 1,
        "ARC_HISPARSE_DISTRIBUTED: no speculative decoding":
            spec or view.speculative_algorithm is None,
        "ARC_HISPARSE_DISTRIBUTED: --mamba-ssm-dtype float16":
            view.mamba_ssm_dtype == "float16",
        "ARC_HISPARSE_DISTRIBUTED: --dtype bfloat16": view.dtype == "bfloat16",
        "ARC_HISPARSE_DISTRIBUTED: no packed GDN state":
            os.environ.get("ARC_GDN_STATE_FMT", "fp16") == "fp16",
        "ARC_HISPARSE_DISTRIBUTED: no ReplaySSM-spec":
            spec or not view.enable_linear_replayssm_spec,
        "ARC_HISPARSE_DISTRIBUTED: even --max-running-requests":
            view.max_running_requests is not None
            and view.max_running_requests > 0 and view.max_running_requests % 2 == 0,
        "ARC_HISPARSE_DISTRIBUTED: session idle expiry disabled":
            view.radix_cache_backend != "qsa_hisparse_session"
            or user.get("qsa_session_idle_timeout_s") == 0,
    }
    if spec:
        needs.update(distributed_spec_shape_needs(view, user))
    return needs


def host_admission_totals(records) -> tuple[int, int, int, int]:
    """One node: all rank pools, one reserve, least available reading before pin."""
    if len(records) != 4 or any(len(record) != 4 for record in records):
        raise ValueError("h25 host admission needs four [KV, GDN, reserve, available] records")
    kv = sum(record[0] for record in records)
    gdn = sum(record[1] for record in records)
    reserve = max(record[2] for record in records)
    available = min(record[3] for record in records)
    if min(kv, gdn, reserve, available) < 0 or kv + gdn + reserve > available:
        raise ValueError(
            f"QSA HiSparse whole-node host admission: KV {kv} + GDN {gdn} "
            f"+ reserve {reserve} bytes exceed available {available}"
        )
    return kv, gdn, reserve, available


def host_memory_state() -> dict:
    """Respect the Pod's cgroup v2 limit when /proc describes a larger host."""
    import psutil

    proc_available = psutil.virtual_memory().available
    available = proc_available
    cgroup = Path("/sys/fs/cgroup")
    limit = (cgroup / "memory.max").read_text().strip()
    used = int((cgroup / "memory.current").read_text())
    cap = None if limit == "max" else int(limit)
    if cap is not None:
        available = min(available, max(0, cap - used))
    return dict(available_bytes=available, proc_available_bytes=proc_available,
                cgroup_limit_bytes=cap, cgroup_current_bytes=used)


def host_available_bytes() -> int:
    return host_memory_state()["available_bytes"]


class AttentionTPConsensus:
    """Each call checks queue identity before agreeing an event-ready prefix.

    This never uses the model TP group: two DP replicas can call it with different
    requests, queue lengths and CUDA-event timing. It runs outside CUDA capture.
    A mismatched lifecycle operation raises before resources are recycled.
    """

    def __init__(self, group, world_size: int, namespace: str):
        self.group, self.world_size, self.namespace = group, world_size, namespace
        self.sequence = 0

    def ready_count(self, operation: str, identity, local_ready: int) -> int:
        import torch
        import torch.distributed as dist

        if self.world_size == 1:
            return local_ready
        payload = json.dumps(
            [self.namespace, self.sequence, operation, identity],
            sort_keys=True, separators=(",", ":"),
        ).encode()
        digest = int.from_bytes(hashlib.sha256(payload).digest()[:7], "big")
        own = torch.tensor([digest, local_ready], dtype=torch.int64)
        peers = [torch.empty_like(own) for _ in range(self.world_size)]
        dist.all_gather(peers, own, group=self.group)
        if any(int(peer[0]) != digest for peer in peers):
            raise RuntimeError(
                f"QSA HiSparse {self.namespace}: attention-TP lifecycle mismatch "
                f"at {operation} #{self.sequence}"
            )
        self.sequence += 1
        return min(int(peer[1]) for peer in peers)

    def agree(self, operation: str, identity) -> None:
        self.ready_count(operation, identity, 0)


def distributed_spec_enabled() -> bool:
    value = os.environ.get("ARC_HISPARSE_DISTRIBUTED_SPEC", "0")
    if value not in ("0", "1"):
        raise ValueError(f"ARC_HISPARSE_DISTRIBUTED_SPEC must be 0 or 1, got {value!r}")
    if value == "1" and not distributed_hisparse_enabled():
        raise ValueError("ARC_HISPARSE_DISTRIBUTED_SPEC=1 requires ARC_HISPARSE_DISTRIBUTED=1")
    return value == "1"


def distributed_spec_shape_needs(view, user) -> dict[str, bool]:
    return {
        "distributed spec: NEXTN3/EAGLE alias, top1, verify4":
            view.speculative_algorithm in ("NEXTN", "EAGLE")
            and view.speculative_num_steps == 3
            and view.speculative_num_draft_tokens == 4
            and view.speculative_eagle_topk == 1,
        "distributed spec: ReplaySSM-spec cache16":
            view.enable_linear_replayssm_spec and view.linear_replayssm_cache_len == 16,
        "distributed spec: native FP8 draft, prefill attention, overlap":
            view.speculative_draft_kv_cache_dtype == "fp8_e4m3"
            and view.speculative_attention_mode == "prefill" and not view.disable_overlap_schedule,
        "distributed spec: no adaptive or multilayer spec":
            not view.speculative_adaptive and not view.enable_multi_layer_eagle,
        "distributed spec: static ragged verify":
            os.environ.get("SGLANG_RAGGED_VERIFY_MODE") == "static",
        "distributed spec: full vocabulary draft path":
            os.environ.get("SGLANG_QWEN4_MTP_DISTRIBUTED_TOPK1", "0") == "0",
        "distributed spec: session snapshots and B>=2048":
            view.radix_cache_backend == "qsa_hisparse_session"
            and user.get("qsa_session_resume") is True
            and isinstance(user.get("qsa_snapshot_interval"), int)
            and not isinstance(user.get("qsa_snapshot_interval"), bool)
            and user["qsa_snapshot_interval"] > 4
            and isinstance(user.get("qsa_hot_blocks"), int)
            and not isinstance(user.get("qsa_hot_blocks"), bool)
            and user["qsa_hot_blocks"] >= 2048
            and user.get("qsa_snapshot_staging_slots") == 4
            and user.get("qsa_spec_small_hot_ok", False) is False,
    }


def new_spec_protocols(namespace, attention_ranks):
    """Separate Gloo channels avoid overlap-thread/scheduler collective ordering races.

    All four target ranks create the same groups in the same order. The draft
    creates no coordinator, so these channels are target-only and CPU-only.
    """
    import torch.distributed as dist

    if dist.get_world_size() != 4:
        raise ValueError("h29 protocol channels require four target ranks")
    own = dist.get_rank()
    if list(attention_ranks) != ([0, 1] if own < 2 else [2, 3]):
        raise ValueError("h29 protocol channels require contiguous attention TP2 ranks")
    result = {}
    for channel in ("allocation", "forward"):
        for ranks in ([0, 1], [2, 3]):
            group = dist.new_group(ranks, backend="gloo")
            if own in ranks:
                result[channel] = AttentionTPConsensus(group, 2, namespace + ":spec-" + channel)
    return result


def agree_spec_transition(consensus, operation, identity, *, invalid=False):
    """Every sibling sees disagreement/invalidity before callers mutate ownership."""
    consensus.agree(operation, [identity, bool(invalid)])
    if invalid:
        raise RuntimeError(f"distributed HiSparse invalid speculative transition at {operation}")


def spec_tensor_identity(tensor):
    """Hash ordered allocator state without putting large lists in the collective."""
    value = tensor.detach().cpu().contiguous()
    return [list(value.shape), str(value.dtype), hashlib.sha256(value.numpy().tobytes()).hexdigest()]


def spec_staging_busy(consensus, busy):
    """A staging slot is reusable only when every TP sibling sees it released."""
    import torch.distributed as dist

    own = busy.detach().cpu().clone()
    consensus.agree("spec-snapshot-state", [list(own.shape), str(own.dtype)])
    if consensus.world_size > 1:
        dist.all_reduce(own, op=dist.ReduceOp.MAX, group=consensus.group)
    return own.to(busy.device)
