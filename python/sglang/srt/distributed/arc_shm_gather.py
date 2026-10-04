"""All-gather of small CPU tensors through POSIX shared memory, for process groups whose ranks share one host.

arc-forge (L4 DFlash optimization, 2026-10-04). The scheduler's per-cycle CPU collectives -- the h29 attention-TP
consensus (AttentionTPConsensus.ready_count, spec_staging_busy) and the DP-attention MLP sync -- go through Gloo,
which costs 0.36 ms per all-gather between two ranks and 0.7 ms among four on the Runpod L4 host. They sit on the
critical path between the verify and the next draft (the GPU is idle while the host runs them), about ten per
decode cycle. Here the same exchange is a write to this rank's slot of a shared buffer and a spin until every peer's
slot carries the same sequence number: tens of microseconds.

ARC_SHM_GATHER=1 turns it on (default off: Gloo, as before). The first call on a group, with a given byte size,
creates the buffer collectively (group rank 0 creates it and broadcasts its name over the group itself), so it
runs at a point all ranks of the group reach together -- exactly where the Gloo collective it replaces ran.

Protocol: two buffers, alternating by sequence parity. A rank writes its payload, then its sequence number (x86
stores become visible in program order), then waits for every slot of that buffer to show the same sequence. A rank
can only reuse a buffer after every peer has posted the next sequence, which each does only after reading the
previous one, so a payload is never overwritten while a peer may still read it. A peer slot ahead of this rank's
sequence, or a wait past ARC_SHM_GATHER_TIMEOUT seconds (default 600), raises: the ranks disagree on how many
exchanges happened, which Gloo would have turned into a hang or a mismatched payload.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

_SLOT_ALIGN = 64  # bytes: one cache line per slot header keeps ranks from sharing lines
_gatherers: dict = {}


def enabled() -> bool:
    value = os.environ.get("ARC_SHM_GATHER", "0")
    if value not in ("0", "1"):
        raise ValueError(f"ARC_SHM_GATHER must be 0 or 1, got {value!r}")
    return value == "1"


class ShmAllGather:
    def __init__(self, group, nbytes: int):
        import torch.distributed as dist
        from multiprocessing import shared_memory

        self.ranks = dist.get_process_group_ranks(group)
        self.world = len(self.ranks)
        self.me = self.ranks.index(dist.get_rank())
        self.nbytes = nbytes
        self.stride = -(-(8 + nbytes) // _SLOT_ALIGN) * _SLOT_ALIGN
        size = 2 * self.world * self.stride
        self.timeout = float(os.environ.get("ARC_SHM_GATHER_TIMEOUT", "600"))
        name = [None]
        if self.me == 0:
            self.shm = shared_memory.SharedMemory(create=True, size=size)
            self.shm.buf[:size] = bytes(size)
            name[0] = self.shm.name
        dist.broadcast_object_list(name, src=self.ranks[0], group=group)
        if self.me != 0:
            self.shm = shared_memory.SharedMemory(name=name[0])
        dist.barrier(group=group)
        if self.me == 0:
            self.shm.unlink()  # every rank holds a mapping; the name is no longer needed
        raw = np.ndarray((2, self.world, self.stride), dtype=np.uint8, buffer=self.shm.buf)
        self.seqs = [raw[b, :, :8].view(np.int64).reshape(self.world) for b in range(2)]
        self.payloads = [raw[b, :, 8 : 8 + nbytes] for b in range(2)]
        self.seq = 0

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """data: this rank's payload (uint8, nbytes); returns [world, nbytes] uint8 in group rank order."""
        if data.dtype != np.uint8 or data.shape != (self.nbytes,):
            raise ValueError(f"shm all-gather of {data.dtype}{data.shape}, buffer holds {self.nbytes} bytes")
        self.seq += 1
        b = self.seq & 1
        seqs, payloads = self.seqs[b], self.payloads[b]
        payloads[self.me] = data
        seqs[self.me] = self.seq
        spins, deadline = 0, None
        while True:
            low = int(seqs.min())
            if low == self.seq:
                break
            high = int(seqs.max())
            if high > self.seq:
                raise RuntimeError(f"shm all-gather ranks {self.ranks}: a peer is at exchange {high}, this rank at "
                                   f"{self.seq}")
            spins += 1
            if spins & 1023 == 0:
                now = time.monotonic()
                deadline = deadline or now + self.timeout
                if now > deadline:
                    raise RuntimeError(f"shm all-gather ranks {self.ranks}: exchange {self.seq} waited "
                                       f"{self.timeout:.0f} s, slots at {seqs.tolist()}")
                time.sleep(0)
        return payloads.copy()


def _gatherer(group, nbytes: int) -> ShmAllGather:
    key = (id(group), nbytes)
    g = _gatherers.get(key)
    if g is None:
        g = _gatherers[key] = ShmAllGather(group, nbytes)
    return g


def all_gather(tensor: torch.Tensor, group) -> torch.Tensor:
    """[world, *tensor.shape] on the CPU: every rank's `tensor` (a contiguous CPU tensor) in group rank order."""
    if tensor.device.type != "cpu":
        raise ValueError(f"shm all-gather takes CPU tensors, got {tensor.device}")
    flat = tensor.contiguous().reshape(-1)
    data = flat.view(torch.uint8).numpy()
    out = _gatherer(group, data.shape[0])(data)
    return torch.from_numpy(out).view(tensor.dtype).reshape(-1, *tensor.shape)
