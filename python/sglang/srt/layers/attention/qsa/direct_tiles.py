"""Measured tile table for the h22 direct sparse decode (ARC_QSA_DIRECT_TILE_TABLE=path.json).

Unset (the default) the backend keeps its built-in tiles bit for bit: G4 NVFP4
(64, 4, 3, 512, 2), FP8 (64, 4, 3, 1024, 0), and the sm_89 HiSparse route
(64, 4, 2, 512|1024, 0). A table only adds measured entries for exact shapes;
a forward whose shape has no entry runs the built-in tile, and every distinct
resolution is logged once with the entry's provenance, so a run's log says
which tile each captured shape used.

The tile is chosen on the host per call -- at capture time for a CUDA graph,
so a captured graph keeps the tile of its own row count. Ranks of one
attention-TP group see the same rows, so they resolve the same tile.

File format (unknown keys, overlapping row ranges, a decode the device cannot
run, or a chunk that is not a multiple of block_n raise at startup):

{
  "format": "arc-qsa-direct-tiles/v1",
  "entries": [
    {"capability": [12, 0], "kind": "nvfp4", "q_heads": 24, "kv_heads": 2,
     "head_dim": 256, "width": 2051, "rows": [513, 1024],
     "tile": {"block_n": 64, "warps": 4, "stages": 3, "chunk": null, "decode": 2},
     "provenance": "bench_qsa_verify_split.py G4 <date> <json path>"}
  ]
}

chunk null = one split (no partials, no combine launch).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

FORMAT = "arc-qsa-direct-tiles/v1"
KINDS = ("nvfp4", "fp8")
_ENTRY_KEYS = frozenset(
    ("capability", "kind", "q_heads", "kv_heads", "head_dim", "width", "rows", "tile", "provenance")
)
_TILE_KEYS = frozenset(("block_n", "warps", "stages", "chunk", "decode"))
# DECODE=2 is the cvt.rn.f16x2.e2m1x2 instruction: sm_100a / sm_120a only
_E2M1_CVT_MAJORS = (10, 12)

Tile = Tuple[int, int, int, Optional[int], int]  # block_n, warps, stages, chunk, decode


class TileEntry(NamedTuple):
    capability: Tuple[int, int]
    kind: str
    q_heads: int
    kv_heads: int
    head_dim: int
    width: int
    rows_lo: int
    rows_hi: int
    tile: Tile
    provenance: str


def _int(entry_no: int, name: str, value, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"tile table entry {entry_no}: {name} must be an integer >= {minimum}, got {value!r}")
    return value


def _parse_tile(entry_no: int, raw, capability: Tuple[int, int]) -> Tile:
    if not isinstance(raw, dict) or set(raw) != _TILE_KEYS:
        raise ValueError(f"tile table entry {entry_no}: tile needs exactly {sorted(_TILE_KEYS)}, got {raw!r}")
    block_n = _int(entry_no, "block_n", raw["block_n"], 16)
    if block_n & (block_n - 1):
        raise ValueError(f"tile table entry {entry_no}: block_n {block_n} is not a power of two")
    warps = _int(entry_no, "warps", raw["warps"])
    if warps & (warps - 1) or warps > 16:
        raise ValueError(f"tile table entry {entry_no}: warps {warps} is not a power of two <= 16")
    stages = _int(entry_no, "stages", raw["stages"])
    chunk = raw["chunk"]
    if chunk is not None and (_int(entry_no, "chunk", chunk) % block_n):
        raise ValueError(f"tile table entry {entry_no}: chunk {chunk} is not a multiple of block_n {block_n}")
    decode = raw["decode"]
    if decode not in (0, 1, 2) or isinstance(decode, bool):
        raise ValueError(f"tile table entry {entry_no}: decode must be 0, 1 or 2, got {decode!r}")
    if decode == 2 and capability[0] not in _E2M1_CVT_MAJORS:
        raise ValueError(
            f"tile table entry {entry_no}: decode 2 (cvt e2m1x2) needs sm_100a/sm_120a, entry is sm_{capability}"
        )
    return block_n, warps, stages, chunk, decode


def load_tile_table(path: str) -> Tuple[TileEntry, ...]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"format", "entries"} or data["format"] != FORMAT:
        raise ValueError(f"{path}: expected {{'format': {FORMAT!r}, 'entries': [...]}}")
    if not isinstance(data["entries"], list) or not data["entries"]:
        raise ValueError(f"{path}: entries must be a non-empty list")
    entries = []
    for no, raw in enumerate(data["entries"]):
        if not isinstance(raw, dict) or set(raw) != _ENTRY_KEYS:
            raise ValueError(f"{path}: entry {no} needs exactly {sorted(_ENTRY_KEYS)}, got {sorted(raw)}")
        cap = raw["capability"]
        if not (isinstance(cap, list) and len(cap) == 2 and all(isinstance(c, int) for c in cap)):
            raise ValueError(f"{path}: entry {no} capability must be [major, minor], got {cap!r}")
        capability = (cap[0], cap[1])
        if raw["kind"] not in KINDS:
            raise ValueError(f"{path}: entry {no} kind must be one of {KINDS}, got {raw['kind']!r}")
        rows = raw["rows"]
        if not (isinstance(rows, list) and len(rows) == 2):
            raise ValueError(f"{path}: entry {no} rows must be [lo, hi], got {rows!r}")
        lo, hi = _int(no, "rows lo", rows[0]), _int(no, "rows hi", rows[1])
        if lo > hi:
            raise ValueError(f"{path}: entry {no} rows [{lo}, {hi}] is empty")
        if not isinstance(raw["provenance"], str) or not raw["provenance"].strip():
            raise ValueError(f"{path}: entry {no} needs a provenance (which measurement chose this tile)")
        entries.append(
            TileEntry(
                capability=capability,
                kind=raw["kind"],
                q_heads=_int(no, "q_heads", raw["q_heads"]),
                kv_heads=_int(no, "kv_heads", raw["kv_heads"]),
                head_dim=_int(no, "head_dim", raw["head_dim"]),
                width=_int(no, "width", raw["width"]),
                rows_lo=lo,
                rows_hi=hi,
                tile=_parse_tile(no, raw["tile"], capability),
                provenance=raw["provenance"],
            )
        )
    for i, a in enumerate(entries):
        for b in entries[i + 1 :]:
            if a[:6] == b[:6] and a.rows_lo <= b.rows_hi and b.rows_lo <= a.rows_hi:
                raise ValueError(f"{path}: entries overlap in rows: {a} / {b}")
    return tuple(entries)


def resolve_tile(
    entries: Tuple[TileEntry, ...],
    *,
    baseline: Tile,
    capability: Tuple[int, int],
    kind: str,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    width: int,
    rows: int,
) -> Tuple[Tile, Optional[TileEntry]]:
    """The entry covering this exact shape, else (baseline, None)."""
    key = (capability, kind, q_heads, kv_heads, head_dim, width)
    for entry in entries:
        if entry[:6] == key and entry.rows_lo <= rows <= entry.rows_hi:
            return entry.tile, entry
    return baseline, None
