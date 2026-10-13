"""arc-forge: ARC_MAMBA_TRACK_HOST -- plan an extend's mamba radix-track metadata from host mirrors.

With the extra_buffer mamba scheduler strategy, every plain extend that reaches a checkpoint grid snapshots the
conv / SSM state of the tracked requests into a ping-pong track slot. Building that plan synchronized the stream
on every such extend. The scheduler read each request's track slot with .item() and made three pageable
torch.tensor(list, device=) copies. The backend then ran mamba_track_mask.any(), indexed by a boolean mask in
_init_track_conv_indices and clamped there to a 0-dim tensor bound (the Scalar overload reads it with .item()),
made six .cpu() copies in _init_track_ssm_indices, ran a mask.nonzero() in the GDN backend, and made four .cpu()
copies in the FlashInfer checkpoint plan. Each of these waits for the work already queued on its stream (the
previous forward), so the overlap scheduler ran these extends in lockstep.

ARC_MAMBA_TRACK_HOST=1 (opt-in, default 0):
  * The scheduler keeps the per-request track mask and seqlens it already builds as host lists
    (ScheduleBatch / ForwardBatch .mamba_extend_track_mask_cpu / .mamba_extend_track_seqlens_cpu) and uploads
    them pinned and non_blocking.
  * The scheduler builds mamba_track_indices on the device. It stacks each request's ping-pong buffer and
    gathers the position the old .item() read. The gather runs in stream order on the schedule stream, the
    same stream that writes those buffers, so it reads the value the .item() would have read.
  * The backend plans every row selection on the host, from those mirrors and the extend lengths. It uploads
    the row lists in one pinned copy and gathers the device-only values (track / cache slot ids) with integer
    indices.
The values, dtypes and devices are the same as before. Only plain ForwardMode.EXTEND uses this path; decode,
verify, draft-extend and mixed batches keep the old one. A plain extend that carries a device track mask but no
host mirrors raises.
"check" runs both paths and raises on any difference. That mode synchronizes, so use it only for validation.
It also logs a line every CHECK_LOG_EVERY compared extends with what was compared, so a run shows that the path
was exercised.
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

MAMBA_TRACK_HOST_ENV = "ARC_MAMBA_TRACK_HOST"
MAMBA_TRACK_HOST = os.environ.get(MAMBA_TRACK_HOST_ENV, "0")
if MAMBA_TRACK_HOST not in ("0", "1", "check"):
    raise ValueError(
        f"{MAMBA_TRACK_HOST_ENV} must be 0, 1 or check, got {MAMBA_TRACK_HOST!r}"
    )

logger = logging.getLogger(__name__)

CHECK_LOG_EVERY = 100
_check_counts: Counter = Counter()


def note_checked(kind: str, **counts: int) -> None:
    """check mode: count one comparison that passed (kind: "extend", "checkpoints" or "slots") and its sizes;
    log the running totals every CHECK_LOG_EVERY extends."""
    _check_counts[kind] += 1
    for name, n in counts.items():
        _check_counts[f"{kind}_{name}"] += n
    if kind == "extend" and _check_counts["extend"] % CHECK_LOG_EVERY == 0:
        logger.info(
            "%s=check: all comparisons matched so far: %s",
            MAMBA_TRACK_HOST_ENV,
            dict(sorted(_check_counts.items())),
        )


# Every segment of a packed upload starts on a 32-byte boundary, the same as a fresh allocation is for the
# kernels that care (Triton pointer-divisibility specialization, FlashInfer GDN's 32-byte inputs).
_SEGMENT_ALIGN_ELEMS = 4  # int64


def _exclusive_cumsum(values: Sequence[int]) -> List[int]:
    out = [0] * len(values)
    acc = 0
    for i, v in enumerate(values):
        out[i] = acc
        acc += v
    return out


@dataclass(frozen=True)
class TrackSsmPlan:
    """Batch-row selections of MambaAttnBackendBase._init_track_ssm_indices, on the host."""

    rows: List[int]  # mask rows, ascending (== mask.nonzero())
    final_rows: List[int]  # rows whose tracked length is state-chunk aligned: final state
    unaligned_rows: List[int]  # the others (track_ssm_h_batch_src)
    h_rows: List[int]  # rows read from h (GDN: unaligned_rows; Mamba2: their on-grid part)
    h_src: List[int]
    chunk_idx: List[int]  # length bs, -1 except at unaligned rows
    seq_idx: Optional[List[int]]  # Mamba2 only: the off-grid unaligned rows (also the recompute rows)
    end_locs: Optional[List[int]]  # Mamba2 only


def plan_track_ssm(
    mask: Sequence[bool],
    track_seqlens: Sequence[int],
    prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    state_chunk_size: int,
    mamba2: bool,
) -> TrackSsmPlan:
    """_init_track_ssm_indices' selections in Python ints. Python's // and % floor like torch's on ints."""
    bs = len(mask)
    if not (len(track_seqlens) == len(prefix_lens) == len(extend_seq_lens) == bs):
        raise RuntimeError(
            f"{MAMBA_TRACK_HOST_ENV}: host mirror lengths differ: mask {bs}, track_seqlens "
            f"{len(track_seqlens)}, prefix_lens {len(prefix_lens)}, extend_seq_lens {len(extend_seq_lens)}"
        )
    chunk = state_chunk_size
    rows = [i for i in range(bs) if mask[i]]
    lens = {i: track_seqlens[i] - prefix_lens[i] for i in rows}
    final_rows = [i for i in rows if lens[i] % chunk == 0]
    unaligned_rows = [i for i in rows if lens[i] % chunk != 0]
    chunk_idx = [-1] * bs
    for i in unaligned_rows:
        chunk_idx[i] = lens[i] // chunk

    if mamba2:
        # `h` is one chunk grid over the flattened batch: on-grid rows read it, the rest recompute.
        starts = _exclusive_cumsum(extend_seq_lens)
        h_rows, h_src, seq_idx, end_locs = [], [], [], []
        for i in unaligned_rows:
            end = starts[i] + (lens[i] // chunk) * chunk
            if end % chunk == 0:
                h_rows.append(i)
                h_src.append(end // chunk)
            else:
                seq_idx.append(i)
                end_locs.append(end)
        return TrackSsmPlan(
            rows, final_rows, unaligned_rows, h_rows, h_src, chunk_idx, seq_idx, end_locs
        )

    offsets = _exclusive_cumsum([(n - 1) // chunk + 1 for n in extend_seq_lens])
    h_src = [offsets[i] + lens[i] // chunk for i in unaligned_rows]
    return TrackSsmPlan(
        rows, final_rows, unaligned_rows, unaligned_rows, h_src, chunk_idx, None, None
    )


def plan_flashinfer_checkpoints(
    mask: Sequence[bool],
    track_seqlens: Sequence[int],
    prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    every: int,
) -> Tuple[List[int], List[int]]:
    """maybe_build_flashinfer_checkpoint_plan's (track checkpoint src, checkpoint cu_starts) in Python ints."""
    bs = len(mask)
    if not (len(track_seqlens) == len(prefix_lens) == len(extend_seq_lens) == bs):
        raise RuntimeError(
            f"{MAMBA_TRACK_HOST_ENV}: host mirror lengths differ: mask {bs}, track_seqlens "
            f"{len(track_seqlens)}, prefix_lens {len(prefix_lens)}, extend_seq_lens {len(extend_seq_lens)}"
        )
    cu_starts = [0] * (bs + 1)
    for i, n in enumerate(extend_seq_lens):
        cu_starts[i + 1] = cu_starts[i] + n // every
    src = []
    for i in range(bs):
        rel = track_seqlens[i] - prefix_lens[i]
        if mask[i] and rel % every != 0:
            src.append(cu_starts[i] + rel // every - 1)
    return src, cu_starts


def _pin(device) -> bool:
    return torch.device(device).type == "cuda"


def to_device_pinned(values: Sequence, dtype: torch.dtype, device) -> torch.Tensor:
    """torch.tensor(values, dtype=dtype, device=device) without the pageable copy's stream sync."""
    return torch.tensor(values, dtype=dtype, pin_memory=_pin(device)).to(
        device, non_blocking=True
    )


def upload_int64(segments: Sequence[Sequence[int]], device) -> List[torch.Tensor]:
    """One pinned host buffer and one non_blocking copy; returns a contiguous int64 device view per segment."""
    flat: List[int] = []
    starts = []
    for seg in segments:
        flat.extend([0] * ((-len(flat)) % _SEGMENT_ALIGN_ELEMS))
        starts.append(len(flat))
        flat.extend(seg)
    packed = to_device_pinned(flat, torch.int64, device)
    return [packed[s : s + len(seg)] for s, seg in zip(starts, segments)]


def gather_track_slots(
    buffers: Sequence[torch.Tensor], positions: Sequence[int], device
) -> torch.Tensor:
    """int64 [bs]: buffers[i][positions[i]] on the device, i.e. the old per-request .item() without the sync."""
    if len(buffers) != len(positions):
        raise RuntimeError(
            f"{MAMBA_TRACK_HOST_ENV}: {len(buffers)} ping-pong buffers for {len(positions)} positions"
        )
    if not buffers:
        return torch.empty((0,), dtype=torch.int64, device=device)
    stacked = torch.stack(list(buffers))  # [bs, ping-pong size]
    pos = upload_int64([positions], device)[0]
    return torch.gather(stacked, 1, pos.unsqueeze(1)).squeeze(1).to(torch.int64)


def assert_same(name: str, ref: Optional[torch.Tensor], new: Optional[torch.Tensor]):
    """check mode: both None, or tensors with equal dtype, shape, device and values."""
    if ref is None or new is None:
        if ref is not None or new is not None:
            raise RuntimeError(
                f"{MAMBA_TRACK_HOST_ENV}=check: {name}: old {ref!r} vs host plan {new!r}"
            )
        return
    if (
        ref.dtype != new.dtype
        or ref.shape != new.shape
        or ref.device != new.device
        or not torch.equal(ref, new)
    ):
        raise RuntimeError(
            f"{MAMBA_TRACK_HOST_ENV}=check: {name} differs: old {ref.dtype} {tuple(ref.shape)} "
            f"{ref.device} {ref.tolist()} vs host plan {new.dtype} {tuple(new.shape)} {new.device} "
            f"{new.tolist()}"
        )
