"""Opt-in selection-overlap telemetry for QSA target verify (h22 direct path).

ARC_QSA_SELECTION_STATS=N with ARC_QSA_SELECTION_STATS_LAYERS=3,23,47 measures, over
the first N TARGET_VERIFY forwards, how much the K verify rows of one request
share the pool slots the direct kernel reads. It answers whether a kernel that
loads/decodes K/V once for several verify rows has anything to reuse, before
anyone writes one. Off (N=0, the default) the backend builds no collector and
the forward path is unchanged.

What is counted is exactly what ``_qsa_direct_decode`` reads: for row r and lane
j, slot = table[table_rows[r], indices[r, j]] when 0 <= indices[r, j] < lens[r],
nothing otherwise. Under HiSparse that is the device row of the hot buffer; a
request's K rows resolve one logical block to one device block within a forward
(the swap-in raises verify_overflow otherwise), so equal slots are equal tokens.

Per monitored layer and group size g (2, 4, 8 consecutive rows of one request,
and the whole request g=K) the device accumulates:
  groups        real groups (every row a real request row)
  slot_loads    sum_i |S_i|            (rows deduplicated first)
  union         |S_1 u ... u S_g|      -> slot_loads / union = ideal load reuse
  inter_loads   g * |S_1 n ... n S_g|  -> inter_loads / slot_loads = loads a
                                          common-history pass would cover
  exact_groups  every row identical to the first, order and -1 lanes included
  sameset_groups every row the same set as the first
plus rows, empty rows, duplicate valid entries inside a row, and a histogram of
max_i (|S_i| - |intersection|) per request (the largest private tail).

CUDA graphs: the statistics are fixed-shape torch ops. Inside a capture they are
recorded into the verify graph and run on every replay; a device counter gates
them so only the first N forwards contribute (later replays still execute the
ops, with zero weight -- restart without the flag for timing runs). Graph
warmup forwards are skipped by the caller. The host learns that N forwards are
done from the backend's per-forward metadata hook, synchronizes once, logs one
JSON line and stops. Transient memory per monitored layer is a handful of
copies of the [rows, width] selection plus int64 sort indices -- on the order
of 100 MB at 880 x 2051 -- taken from the graph pool when captured, so the
debug run may need a little more headroom than production.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

GROUP_KEYS: Tuple[str, ...] = ("g2", "g4", "g8", "req")
_GROUP_FIELDS: Tuple[str, ...] = (
    "groups",
    "slot_loads",
    "union",
    "inter_loads",
    "exact_groups",
    "sameset_groups",
)
_ROW_FIELDS: Tuple[str, ...] = ("rows", "empty_rows", "dup_entries")
# max_i(|S_i| - |intersection|) per request: [0], [1,4), [4,16), ..., [1024, inf)
PRIVATE_EDGES: Tuple[int, ...] = (1, 4, 16, 64, 256, 1024)
_HIST_LABELS: Tuple[str, ...] = ("0", "1-3", "4-15", "16-63", "64-255", "256-1023", ">=1024")

FIELDS: Tuple[str, ...] = (
    _ROW_FIELDS
    + tuple(f"{key}.{name}" for key in GROUP_KEYS for name in _GROUP_FIELDS)
    + tuple(f"private_max.{label}" for label in _HIST_LABELS)
)
NUM_STATS = len(FIELDS)
_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FIELDS)}


def parse_env() -> Optional[Tuple[int, Tuple[int, ...]]]:
    """(windows, layer ids) from the environment, or None when telemetry is off."""
    raw = os.environ.get("ARC_QSA_SELECTION_STATS", "0")
    if not raw.isdigit():
        raise ValueError(f"ARC_QSA_SELECTION_STATS must be a non-negative integer, got {raw!r}")
    windows = int(raw)
    layers_raw = os.environ.get("ARC_QSA_SELECTION_STATS_LAYERS")
    if windows == 0:
        if layers_raw is not None:
            raise ValueError("ARC_QSA_SELECTION_STATS_LAYERS is set but ARC_QSA_SELECTION_STATS is 0")
        return None
    if not layers_raw:
        raise ValueError(
            "ARC_QSA_SELECTION_STATS=N needs ARC_QSA_SELECTION_STATS_LAYERS, a comma-separated "
            "list of QSA (full-attention) layer ids, e.g. 3,23,47"
        )
    parts = [part.strip() for part in layers_raw.split(",")]
    if not all(part.isdigit() for part in parts):
        raise ValueError(f"ARC_QSA_SELECTION_STATS_LAYERS must be layer ids, got {layers_raw!r}")
    layers = tuple(sorted({int(part) for part in parts}))
    return windows, layers


def read_slots(
    table: torch.Tensor,
    table_rows: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
) -> torch.Tensor:
    """[rows, width] int32 pool slot each lane of _qsa_direct_decode reads, -1 where it reads none."""
    if indices.ndim != 2 or table.ndim != 2:
        raise ValueError(f"indices/table must be 2-D, got {tuple(indices.shape)}/{tuple(table.shape)}")
    rows = indices.shape[0]
    if table_rows.shape != (rows,) or lens.shape != (rows,):
        raise ValueError(
            f"table_rows {tuple(table_rows.shape)} / lens {tuple(lens.shape)} must be [{rows}]"
        )
    valid = (indices >= 0) & (indices < lens[:, None])
    safe = torch.where(valid, indices, torch.zeros_like(indices)).long()
    slots = table[table_rows.long()[:, None], safe].to(torch.int32)
    return torch.where(valid, slots, torch.full_like(slots, -1))


def _group_sizes(k: int) -> Tuple[Tuple[str, int], ...]:
    return (("g2", 2), ("g4", 4), ("g8", 8), ("req", k))


def private_edges(device) -> torch.Tensor:
    """Histogram edges as a device tensor; allocate it before any graph capture (a host list ->
    device copy is not capturable)."""
    return torch.tensor(PRIVATE_EDGES, dtype=torch.int64, device=device)


def selection_overlap_counts(
    slots: torch.Tensor, real_rows: torch.Tensor, k: int, edges: torch.Tensor
) -> torch.Tensor:
    """int64 [NUM_STATS] counts for one forward's selection (layout FIELDS).

    slots: [rows, width] int32, -1 = no read; rows are request-major, k per
    request. real_rows: bool [rows]. edges: private_edges(device). Fixed-shape
    torch ops only (no host sync, no host->device copy, no data-dependent
    shapes), so it can be recorded into a CUDA graph. Group sizes that do not
    divide k contribute zeros.
    """
    if slots.ndim != 2 or slots.dtype != torch.int32:
        raise ValueError(f"slots must be int32 [rows, width], got {slots.dtype} {tuple(slots.shape)}")
    rows, width = slots.shape
    if real_rows.dtype != torch.bool or real_rows.shape != (rows,):
        raise ValueError(f"real_rows must be bool [{rows}], got {real_rows.dtype} {tuple(real_rows.shape)}")
    if k < 1 or rows % k:
        raise ValueError(f"{rows} rows are not whole requests of {k} verify rows")
    if edges.dtype != torch.int64 or edges.shape != (len(PRIVATE_EDGES),) or edges.device != slots.device:
        raise ValueError(f"edges must be private_edges({slots.device}), got {edges.dtype} {tuple(edges.shape)}")
    device = slots.device
    zero = torch.zeros((), dtype=torch.int64, device=device)
    values = {name: zero for name in FIELDS}
    if rows == 0 or width == 0:
        return torch.stack([values[name] for name in FIELDS])
    real = real_rows.to(torch.int64)

    ordered, _ = torch.sort(slots, dim=1)
    repeat = (ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])
    dedup = torch.cat(
        [ordered[:, :1], torch.where(repeat, torch.full_like(ordered[:, 1:], -1), ordered[:, 1:])],
        dim=1,
    )
    # each repeat became a -1 in place; sort again so equal sets are equal rows
    dedup, _ = torch.sort(dedup, dim=1)
    card = (dedup >= 0).sum(1)  # |S_i|, int64
    values["rows"] = real.sum()
    values["empty_rows"] = ((card == 0).to(torch.int64) * real).sum()
    values["dup_entries"] = (repeat.sum(1) * real).sum()

    for key, g in _group_sizes(k):
        if k % g:
            continue
        groups = rows // g
        greal = real_rows.view(groups, g).all(1).to(torch.int64)
        merged, _ = torch.sort(dedup.view(groups, g * width), dim=1)
        valid = merged >= 0
        first = torch.cat([valid[:, :1], valid[:, 1:] & (merged[:, 1:] != merged[:, :-1])], dim=1)
        union = first.sum(1)
        run = torch.cumsum(first, dim=1, dtype=torch.int64) - 1  # run id of each sorted element
        multiplicity = torch.zeros_like(run).scatter_add_(1, run.clamp(min=0), valid.to(torch.int64))
        inter = (multiplicity == g).sum(1)  # rows are deduplicated, so multiplicity g = in every row
        loads = card.view(groups, g).sum(1)
        grouped = slots.view(groups, g, width)
        exact = (grouped == grouped[:, :1]).all(2).all(1).to(torch.int64)
        sets = dedup.view(groups, g, width)
        same = (sets == sets[:, :1]).all(2).all(1).to(torch.int64)
        values[f"{key}.groups"] = greal.sum()
        values[f"{key}.slot_loads"] = (loads * greal).sum()
        values[f"{key}.union"] = (union * greal).sum()
        values[f"{key}.inter_loads"] = (inter * g * greal).sum()
        values[f"{key}.exact_groups"] = (exact * greal).sum()
        values[f"{key}.sameset_groups"] = (same * greal).sum()
        if key == "req":
            private = (card.view(groups, g) - inter[:, None]).amax(1)
            bins = torch.bucketize(private, edges, right=True)
            hist = torch.zeros(len(_HIST_LABELS), dtype=torch.int64, device=device)
            hist.scatter_add_(0, bins, greal)
            for i, label in enumerate(_HIST_LABELS):
                values[f"private_max.{label}"] = hist[i]
    return torch.stack([values[name] for name in FIELDS])


def summarize(counts: Dict[str, int]) -> Dict[str, object]:
    """Ratios from one layer's accumulated FIELDS counts (host ints)."""
    out: Dict[str, object] = {name: counts[name] for name in _ROW_FIELDS}
    for key in GROUP_KEYS:
        groups = counts[f"{key}.groups"]
        if groups == 0:
            out[key] = None  # this group size does not divide K
            continue
        loads = counts[f"{key}.slot_loads"]
        union = counts[f"{key}.union"]
        out[key] = {
            "groups": groups,
            "mean_slots_per_row": loads / max(1, counts["rows"]) if key == "req" else None,
            "load_reuse": loads / union if union else None,
            "common_fraction": counts[f"{key}.inter_loads"] / loads if loads else None,
            "exact_fraction": counts[f"{key}.exact_groups"] / groups,
            "same_set_fraction": counts[f"{key}.sameset_groups"] / groups,
        }
    out["private_max_hist"] = {label: counts[f"private_max.{label}"] for label in _HIST_LABELS}
    return out


class QSASelectionStats:
    """Device accumulators plus the host-side window bookkeeping (one per backend)."""

    def __init__(self, windows: int, layers: Tuple[int, ...], device) -> None:
        if windows < 1 or not layers:
            raise ValueError(f"selection stats need windows >= 1 and layers, got {windows}, {layers}")
        self.windows = windows
        self.layers = layers
        self._slot = {layer: i for i, layer in enumerate(layers)}
        self._acc = torch.zeros(len(layers), NUM_STATS, dtype=torch.int64, device=device)
        self._layer_forwards = torch.zeros(len(layers), dtype=torch.int64, device=device)
        # verify forwards that reached the last monitored layer while the window was open
        self._landed = torch.zeros(1, dtype=torch.int64, device=device)
        self._edges = private_edges(device)
        self._host_forwards = 0
        self._last_landed = -1
        self._k: Optional[int] = None
        self.done = False

    def observe(
        self,
        *,
        layer_id: int,
        k: int,
        table: torch.Tensor,
        table_rows: torch.Tensor,
        indices: torch.Tensor,
        lens: torch.Tensor,
        logical_lens: torch.Tensor,
        row_req: torch.Tensor,
    ) -> None:
        """Queue one layer's counts (eager, or recorded into a graph capture)."""
        if self.done or layer_id not in self._slot:
            return
        if k < 2:
            raise RuntimeError(f"selection stats observe TARGET_VERIFY rows, got K={k}")
        if self._k is None:
            self._k = k
        elif self._k != k:
            raise RuntimeError(f"verify width changed from {self._k} to {k} inside the stats window")
        rows = indices.shape[0]
        for name, tensor in (("table_rows", table_rows), ("lens", lens),
                             ("logical_lens", logical_lens), ("row_req", row_req)):
            if tensor.ndim != 1 or tensor.shape[0] < rows:
                raise RuntimeError(f"selection stats: {name} {tuple(tensor.shape)} has fewer than {rows} rows")
        slots = read_slots(table, table_rows[:rows], indices, lens[:rows])
        # graph padding rows have logical length 1 (or the reserved request slot 0); a real
        # verify row sees at least its prefix token and itself
        real = (logical_lens[:rows] >= 2) & (row_req[:rows] != 0)
        counts = selection_overlap_counts(slots, real, k, self._edges)
        slot = self._slot[layer_id]
        active = (self._landed < self.windows).to(torch.int64)
        self._acc[slot].add_(counts * active)
        self._layer_forwards[slot : slot + 1].add_(active)
        if layer_id == self.layers[-1]:
            self._landed.add_(active)

    def on_verify_forward(self) -> None:
        """Host hook, once per TARGET_VERIFY forward before it is launched (eager or replay)."""
        if self.done:
            return
        self._host_forwards += 1
        if self._host_forwards <= self.windows:
            return
        # forwards 1..N are all enqueued; wait for them once
        if self._acc.is_cuda:
            torch.cuda.synchronize(self._acc.device)
        landed = int(self._landed.item())
        if landed < self.windows:
            if landed == self._last_landed:
                per_layer = dict(zip(self.layers, self._layer_forwards.tolist()))
                raise RuntimeError(
                    f"ARC_QSA_SELECTION_STATS: {self._host_forwards} verify forwards launched but only "
                    f"{landed} reached layer {self.layers[-1]} (per-layer {per_layer}); every id in "
                    "ARC_QSA_SELECTION_STATS_LAYERS must be a QSA layer on the h22 direct path "
                    "(ARC_QSA_NVFP4_DIRECT=1 on sm_120, or the sm_89 HiSparse route)"
                )
            self._last_landed = landed
            return
        logger.info("ARC_QSA_SELECTION_STATS %s", json.dumps(self.summary(), sort_keys=True))
        self.done = True

    def summary(self) -> Dict[str, object]:
        acc = self._acc.tolist()
        forwards = self._layer_forwards.tolist()
        return {
            "windows": self.windows,
            "verify_width": self._k,
            "layers": {
                str(layer): dict(
                    forwards=forwards[i],
                    **summarize(dict(zip(FIELDS, acc[i]))),
                )
                for i, layer in enumerate(self.layers)
            },
        }
