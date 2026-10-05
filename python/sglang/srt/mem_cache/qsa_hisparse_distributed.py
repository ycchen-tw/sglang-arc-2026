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


def _memory_cgroups() -> list:
    """[(cgroup v2 directory, its memory.max text)] that bound this process's memory.

    In a pod with its own cgroup namespace (Kaggle, Runpod, Vast: where h25 was built) /sys/fs/cgroup is the pod's
    cgroup and has memory.max: that one directory, as before. On a bare host (B200) /sys/fs/cgroup is the cgroup v2
    root, which has no memory.max: this process's own cgroup from /proc/self/cgroup and every ancestor below the root,
    any of which may hold the binding limit (none, for a process in the root cgroup)."""
    root = Path("/sys/fs/cgroup")
    try:
        return [(root, (root / "memory.max").read_text().strip())]
    except FileNotFoundError:
        pass
    entries = [line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")]
    if len(entries) != 1:
        raise RuntimeError(f"{root} has no memory.max and /proc/self/cgroup has no single cgroup v2 entry: {entries}")
    relative = entries[0][len("0::"):].strip("/")
    cgroups = []
    cgroup = root / relative if relative else root
    while cgroup != root:
        cgroups.append((cgroup, (cgroup / "memory.max").read_text().strip()))
        cgroup = cgroup.parent
    return cgroups


def host_memory_state() -> dict:
    """Respect the Pod's cgroup v2 limit when /proc describes a larger host."""
    import psutil

    proc_available = psutil.virtual_memory().available
    available = proc_available
    cap = used = tightest = None
    for cgroup, limit in _memory_cgroups():
        current = int((cgroup / "memory.current").read_text())
        if used is None:
            used = current  # the innermost cgroup's usage, reported when no cgroup has a limit
        if limit == "max":
            continue
        headroom = int(limit) - current
        if os.environ.get("ARC_ADMISSION_RECLAIMABLE_FILE") == "1":
            # arc-forge (Vast only, opt-in): memory.current counts page cache. Inactive file pages are reclaimable
            # (here: the checkpoint read at load time), so count them back; a container cannot drop_caches.
            for line in (cgroup / "memory.stat").read_text().splitlines():
                key, value = line.split()
                if key == "inactive_file":
                    headroom += int(value)
        if tightest is None or headroom < tightest:
            tightest, cap, used = headroom, int(limit), current
    if tightest is not None:
        available = min(available, max(0, tightest))
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


def _dflash_drafter_enabled() -> bool:
    """speculative/arc_dflash.enabled()'s contract, without importing sglang: this module also loads standalone (the
    L4 launcher's h25 shape check runs it with runpy under a 30 s timeout). Unset or empty is off; anything else must
    name the drafter's export directory."""
    path = os.environ.get("ARC_DFLASH_DRAFTER")
    if not path:
        return False
    if not os.path.isdir(path):
        raise RuntimeError(f"ARC_DFLASH_DRAFTER={path!r} is not a drafter export directory")
    return True


def distributed_mtp_steps() -> int:
    """Explicit reviewed depth; default retains h29 NEXTN3/verify4."""
    value = os.environ.get("ARC_HISPARSE_MTP_STEPS", "3")
    # arc-forge l01: the DFlash drafter (experimental-dflash/l4) proposes K-1 = 1..7 tokens; the verify-width
    # machinery (h16-h19: ring width, spec backup, union hot blocks) is generic in K. The native MTP keeps 1 / 3.
    allowed = ("1", "2", "3", "4", "5", "6", "7") if _dflash_drafter_enabled() else ("1", "3")
    if value not in allowed:
        raise ValueError(f"ARC_HISPARSE_MTP_STEPS must be one of {allowed}")
    if value == "1" and not distributed_spec_enabled():
        raise ValueError("ARC_HISPARSE_MTP_STEPS=1 requires guarded distributed spec")
    return int(value)


def distributed_spec_shape_needs(view, user) -> dict[str, bool]:
    steps = distributed_mtp_steps()
    verify = steps + 1
    return {
        f"distributed spec: NEXTN{steps}/EAGLE alias, top1, verify{verify}":
            view.speculative_algorithm in ("NEXTN", "EAGLE")
            and view.speculative_num_steps == steps
            and view.speculative_num_draft_tokens == verify
            and view.speculative_eagle_topk == 1,
        "distributed spec: one-step requires rejection sampling":
            steps != 1 or view.speculative_use_rejection_sampling,
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
        f"distributed spec: session snapshots and B>={512 * verify}":
            view.radix_cache_backend == "qsa_hisparse_session"
            and user.get("qsa_session_resume") is True
            and isinstance(user.get("qsa_snapshot_interval"), int)
            and not isinstance(user.get("qsa_snapshot_interval"), bool)
            and user["qsa_snapshot_interval"] > verify
            and isinstance(user.get("qsa_hot_blocks"), int)
            and not isinstance(user.get("qsa_hot_blocks"), bool)
            and user["qsa_hot_blocks"] >= 512 * verify
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



def spec_phase_packed_enabled():
    value = os.environ.get("ARC_HISPARSE_SPEC_PHASE_PACKED", "0")
    if value not in ("0", "1"):
        raise ValueError("ARC_HISPARSE_SPEC_PHASE_PACKED must be 0 or 1")
    if value == "1" and (os.environ.get("ARC_HISPARSE_DISTRIBUTED") != "1"
                         or os.environ.get("ARC_HISPARSE_DISTRIBUTED_SPEC") != "1"):
        raise ValueError("phase-packed agreement requires guarded distributed HiSparse spec")
    return value == "1"


def spec_phase_allocation_packable(reqs, current, following, needed):
    # No implicit integer casts: unsupported lengths/rows use the exact legacy
    # indexing/error path. Do not consume generators or invoke numeric coercion.
    if (type(reqs) not in (list, tuple) or type(current) not in (list, tuple)
            or type(following) not in (list, tuple) or type(needed) is not int):
        return False
    return (all(type(value) is int for value in (*current, *following))
            and all(req.kv.req_pool_idx is None or type(req.kv.req_pool_idx) is int for req in reqs))


def spec_phase_to_cpu(tensors):
    """One blocking live D2H packet per phase; exact dtype/shape/contiguous bytes.

    Caller supplies current-stream producer dependencies, as for legacy .cpu().
    No cross-call storage, asynchronous CPU reads, allocator mirror or hash cache.
    CPU tensors stay on CPU. Each returned GPU snapshot owns the copied packet.
    """
    import torch

    outputs, chunks, records, device, offset = [], [], [], None, 0
    for tensor in tensors:
        value = tensor.detach()
        if value.device.type == "cpu":
            outputs.append(value.contiguous())
            continue
        if value.device.type != "cuda":
            raise ValueError("phase-packed agreement supports CPU/CUDA tensors only")
        if device is not None and value.device != device:
            raise ValueError("one agreement phase cannot mix CUDA devices")
        device = value.device
        shape, dtype, size = tuple(value.shape), value.dtype, value.element_size()
        data = value.contiguous().reshape(-1).view(torch.uint8)
        padding = (-offset) % size
        if padding:
            chunks.append(torch.empty(padding, dtype=torch.uint8, device=device))
            offset += padding
        records.append((len(outputs), offset, data.numel(), shape, dtype))
        outputs.append(None)
        chunks.append(data)
        offset += data.numel()
    if records:
        # cat and copy stay on the caller's current stream, preserving producers.
        packet = torch.cat(chunks).cpu()
        for index, start, count, shape, dtype in records:
            outputs[index] = packet[start:start + count].view(dtype).reshape(shape)
    return outputs


def spec_packed_allocation_identity(coord, reqs, current, following, needed):
    import torch

    logical = coord.token_to_kv_pool_allocator.logical_attn_allocator
    host = coord.mem_pool_host
    rows = [req.kv.req_pool_idx for req in reqs]
    invalid = not len(reqs) == len(current) == len(following) or needed < 0
    tails, grow, pages, tail_rows, tail_cols, tail_places = [], [], 0, [], [], []
    if not invalid:
        for row, a, n in zip(rows, current, following):
            if (row is None or not 1 <= row < coord.req_to_host_pool.shape[0]
                    or not 0 <= a <= n <= coord.req_to_token_pool.req_to_token.shape[1]):
                invalid = True
                continue
            tails.append(-1)
            if a != 0:
                tail_rows.append(row)
                tail_cols.append(a - 1)
                tail_places.append(len(tails) - 1)
            pages += -(-n // logical.page_size) - -(-a // logical.page_size)
            have = int(coord.req_to_host_pool_allocated_len[row])
            target = -(-coord.layout.seq_blocks(n) // coord.page_size) * coord.page_size
            grow.append([row, have, target])
            invalid |= target > coord.req_to_host_pool.shape[1]
        invalid |= needed != sum(n - a for a, n in zip(current, following))
    host_needed = sum(max(0, target - have) for _, have, target in grow)
    invalid |= pages * logical.page_size > logical.available_size() or host_needed > host.available_size()
    table = coord.req_to_token_pool.req_to_token
    row_index = torch.tensor(tail_rows, dtype=torch.int64, device=table.device)
    col_index = torch.tensor(tail_cols, dtype=torch.int64, device=table.device)
    tensors = [table[row_index, col_index], logical.free_pages, *logical.staged_pages]
    snapshots = spec_phase_to_cpu(tensors)
    for place, tail in zip(tail_places, snapshots[0].tolist()):
        tails[place] = int(tail)
    identity = [[req.rid for req in reqs], rows, current, following, int(needed), tails, grow,
        [spec_tensor_identity(snapshots[1]), [spec_tensor_identity(x) for x in snapshots[2:]]],
        [spec_tensor_identity(host.free_slots), [spec_tensor_identity(x) for x in host.release_slots]],
        logical.available_size(), host.available_size()]
    return identity, invalid


def spec_packed_verify_identity(batch, verify_input):
    snapshots = spec_phase_to_cpu([batch.req_pool_indices, batch.seq_lens,
        *[getattr(verify_input, name) for name in
          ("draft_token", "positions", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling")]])
    return [[req.rid for req in batch.reqs], snapshots[0].tolist(), snapshots[1].tolist(),
            [spec_tensor_identity(value) for value in snapshots[2:]]]


def spec_packed_accept_identity(coord, batch, predict, accepted, indices):
    # Keep TWO phase-local blocking reads: sample results, then their exact
    # ragged destinations. No guard is moved past mamba commit / host backup.
    snapshots = spec_phase_to_cpu([batch.req_pool_indices, batch.seq_lens, accepted,
                                  indices, predict, coord.verify_overflow])
    rows, before, counts = [value.tolist() for value in snapshots[:3]]
    address_views = []
    invalid = not len(rows) == len(before) == len(counts)
    for row, seq, count in zip(rows, before, counts):
        if not 1 <= row < coord.req_to_host_pool.shape[0] or not 1 <= count <= coord.verify_tokens:
            invalid = True
            continue
        lo = int(seq) // coord.layout.block_tokens
        hi = (int(seq) + int(count)) // coord.layout.block_tokens
        invalid |= not 0 <= lo <= hi <= coord.req_to_host_pool.shape[1]
        # Slicing itself preserves legacy clipping, including invalid bounds.
        address_views.append(coord.req_to_host_pool[row, lo:hi])
    addresses = [value.tolist() for value in spec_phase_to_cpu(address_views)]
    flag = int(snapshots[5].reshape(-1)[0])
    identity = [rows, before, counts, snapshots[3].tolist(), snapshots[4].tolist(), addresses, flag]
    invalid = (invalid or bool(flag) or any(not 1 <= n <= coord.verify_tokens for n in counts)
               or any(loc < 0 for row in addresses for loc in row))
    return identity, invalid
