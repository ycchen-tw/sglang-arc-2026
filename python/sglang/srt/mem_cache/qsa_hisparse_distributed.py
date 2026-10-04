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
    """Only the initial four-L4, non-speculative serving shape is admitted."""
    user = json.loads(view.hisparse_config) if view.hisparse_config else {}
    return {
        "ARC_HISPARSE_DISTRIBUTED: --tp-size 4 --ep-size 4":
            view.tp_size == 4 and view.ep_size == 4,
        "ARC_HISPARSE_DISTRIBUTED: --dp-size 2 --enable-dp-attention":
            view.dp_size == 2 and view.enable_dp_attention,
        "ARC_HISPARSE_DISTRIBUTED: one node": view.nnodes == 1,
        "ARC_HISPARSE_DISTRIBUTED: no speculative decoding":
            view.speculative_algorithm is None,
        "ARC_HISPARSE_DISTRIBUTED: --mamba-ssm-dtype float16":
            view.mamba_ssm_dtype == "float16",
        "ARC_HISPARSE_DISTRIBUTED: --dtype bfloat16": view.dtype == "bfloat16",
        "ARC_HISPARSE_DISTRIBUTED: no packed GDN state":
            os.environ.get("ARC_GDN_STATE_FMT", "fp16") == "fp16",
        "ARC_HISPARSE_DISTRIBUTED: no ReplaySSM-spec":
            not view.enable_linear_replayssm_spec,
        "ARC_HISPARSE_DISTRIBUTED: even --max-running-requests":
            view.max_running_requests is not None
            and view.max_running_requests > 0 and view.max_running_requests % 2 == 0,
        "ARC_HISPARSE_DISTRIBUTED: session idle expiry disabled":
            view.radix_cache_backend != "qsa_hisparse_session"
            or user.get("qsa_session_idle_timeout_s") == 0,
    }


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
