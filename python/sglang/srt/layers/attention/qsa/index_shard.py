"""QSA index cache split across the attention-TP pair by layer (ARC_QSA_INDEX_SHARD=layer; opt-in, default off).

The QSA indexer has one KV head, so attention TP does not split its compressed-key cache: both ranks of an
attention-TP pair hold the whole index of all QSA layers (L4: 12 layers x 33 B per logical token, 2.10 GiB per rank at
L = 5.70M). With ``ARC_QSA_INDEX_SHARD=layer`` QSA layer number ``i`` of this stage (in layer order) belongs to
attention-TP rank ``i % 2``. Only the owner keeps that layer's index state (compressed keys, their FP8 scales and the
pending-key ring) and runs its indexer (projection, compression, scores, top-k). It then hands its
``[rows, block_topk]`` int32 block top-k to the pair, and both ranks expand the same blocks (and under HiSparse swap
the same blocks into their own hot slabs). Every other step is unchanged, so the selection and the attention outputs
are the unsharded ones bit for bit.

Hand-off: one attention-TP ``all_gather_into_tensor`` per QSA layer and forward, in layer order on both ranks (like
the o_proj all-reduce that follows it). A block index is a block of the row's own sequence, -1 <= b < ceil(C / 4), so
it travels as the low 16 bits of the int32 (C < 262,140 tokens, checked at startup; -1 stays -1, b >= 32768 is
restored by + 65536) and the int16 rows as a 1-D bf16 view: a pure byte copy, which the hostcomm hook carries as
ag_bf16 on Kaggle and pynccl inside CUDA graphs elsewhere. The owner keeps its own int32 tensor.

Off (unset, ``0`` or empty) nothing changes. On, it raises at startup unless: attention TP is exactly 2, the stage has
an even, nonzero number of QSA layers, one pipeline stage, no attention CP / DCP, a context below 262,140 tokens, and
the pool is the target's (a native-MTP draft keeps one QSA layer and cannot split it). Paths that are not covered raise
where they are reached (the breakable prefill CUDA graph, draft-extend, PD transfer).

``ARC_QSA_INDEX_SHARD=layer-check`` is the in-server equivalence check (no memory saving): every rank keeps and
computes every layer's index as unsharded, the owner hands its top-k over as under ``layer``, and the other rank runs
the ``layer`` receiving path (expansion, HiSparse swap-in) on the received top-k after comparing it with the one it
computed itself, row by row, on the device (CUDA graphs included). Any differing row raises at the next eager
hand-off; the running counts are logged every minute. Both ranks compute the same top-k only with deterministic
top-k (ARC_QSA_TOPK_EXACT=1), so run the check with it.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

ENV = "ARC_QSA_INDEX_SHARD"
_MODES = {"": None, "0": None, "layer": "layer", "layer-check": "layer-check"}
CHECK_LOG_SECONDS = 60.0
# A block index on the wire: the low 16 bits of its int32, so every valid one stays below 0xFFFF (the bits of -1).
WIRE_MAX_BLOCKS = 0xFFFF


def _parse(raw: str) -> Optional[str]:
    if raw not in _MODES:
        raise ValueError(f"{ENV}={raw!r}: expected one of {sorted(_MODES)}")
    return _MODES[raw]


_MODE = _parse(os.environ.get(ENV, ""))


def mode() -> Optional[str]:
    return _MODE


def enabled() -> bool:
    return _MODE is not None


def checking() -> bool:
    """layer-check: every rank keeps every QSA layer's index (and checks the hand-off against its own top-k)."""
    return _MODE == "layer-check"


def require_off(where: str) -> None:
    """Raise if the shard is on: ``where`` is a path the shard does not cover."""
    if enabled():
        raise NotImplementedError(f"{ENV}={_MODE} does not cover {where}; unset it")


def _attn_tp_group():
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    return get_attn_tp_group()


def _check_parallel(world: int) -> None:
    from sglang.srt.runtime_context import get_parallel

    ps = get_parallel()
    if world != 2:
        raise NotImplementedError(
            f"{ENV}={_MODE} splits the QSA index over an attention-TP pair; attention TP is {world} "
            "(ARC_HISPARSE_TP4 / pure TP4 replicate the index on four ranks)"
        )
    if ps.pp_size != 1:
        raise NotImplementedError(f"{ENV}={_MODE} covers one pipeline stage, got pp_size={ps.pp_size}")
    if ps.attn_cp_size != 1 or ps.attn_dcp_size != 1:
        raise NotImplementedError(
            f"{ENV}={_MODE} does not cover attention CP / DCP (attn_cp_size={ps.attn_cp_size}, "
            f"attn_dcp_size={ps.attn_dcp_size})"
        )


def index_layers_per_rank(num_layers: int) -> int:
    """QSA layers whose index state one rank holds: all of them unsharded and under layer-check, num_layers / 2
    under the shard."""
    if not enabled():
        return num_layers
    world = _attn_tp_group().world_size
    _check_parallel(world)
    if num_layers <= 0 or num_layers % world:
        raise NotImplementedError(
            f"{ENV}={_MODE} splits the QSA layers evenly over the attention-TP pair; this stage has {num_layers} "
            "(a native-MTP draft pool has one)"
        )
    return num_layers if checking() else num_layers // world


def to_wire(blocks: torch.Tensor) -> torch.Tensor:
    """[rows, width] int32 block indices in [-1, WIRE_MAX_BLOCKS) -> their low 16 bits, [rows, width] int16
    (little-endian: element 0 of each int32's int16 pair)."""
    rows, width = blocks.shape
    return blocks.view(torch.int16).view(rows, width, 2)[:, :, 0].contiguous()


def from_wire(low: torch.Tensor) -> torch.Tensor:
    """to_wire's inverse: -1 (0xFFFF) comes back as -1, a block >= 32768 (negative as int16, below -1) + 65536."""
    low = low.to(torch.int32)
    return torch.where(low < -1, low + 65536, low)


class QSAIndexLayerShard:
    """Which attention-TP rank owns which QSA layer's index, and the owner -> pair hand-off of a layer's top-k.

    max_blocks: an upper bound of a sequence's compressed blocks (ceil((context + draft window) / ratio)); it must
    stay below WIRE_MAX_BLOCKS for the 16-bit wire format.
    check (layer-check): this rank holds every layer's index (held_layer_ids) and verify() compares a received top-k
    with its own; the counts live on ``device``."""

    def __init__(
        self,
        layer_ids: Sequence[int],
        rank: int,
        world: int,
        group=None,
        *,
        max_blocks: int = 0,
        check: bool = False,
        device=None,
    ):
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if world != 2:
            raise NotImplementedError(f"{ENV}: attention-TP world {world}, the shard needs a pair")
        if not 0 <= rank < world:
            raise ValueError(f"{ENV}: attention-TP rank {rank} outside [0, {world})")
        if not layer_ids or len(layer_ids) % world:
            raise NotImplementedError(f"{ENV}: {len(layer_ids)} QSA layers do not split evenly over {world} ranks")
        if len(set(layer_ids)) != len(layer_ids) or list(layer_ids) != sorted(layer_ids):
            raise ValueError(f"{ENV}: QSA layer ids must be distinct and ascending, got {list(layer_ids)}")
        if not 0 <= max_blocks < WIRE_MAX_BLOCKS:
            raise NotImplementedError(
                f"{ENV}: up to {max_blocks} compressed blocks per sequence; the 16-bit hand-off carries block "
                f"indices below {WIRE_MAX_BLOCKS} (context under {WIRE_MAX_BLOCKS * 4} tokens at ratio 4)"
            )
        if sys.byteorder != "little":
            raise NotImplementedError(f"{ENV}: the 16-bit hand-off takes the low half of a little-endian int32")
        self.layer_ids: Tuple[int, ...] = layer_ids
        self.rank = int(rank)
        self.world = int(world)
        self.max_blocks = int(max_blocks)
        self._owner = {layer_id: i % world for i, layer_id in enumerate(layer_ids)}
        self.owned_layer_ids: Tuple[int, ...] = tuple(
            layer_id for layer_id in layer_ids if self._owner[layer_id] == self.rank
        )
        # the layers whose index state this rank's pool keeps
        self.check = bool(check)
        self.held_layer_ids: Tuple[int, ...] = self.layer_ids if self.check else self.owned_layer_ids
        # group=None: ownership bookkeeping only (pool / sizing tests); exchange() then raises.
        self._group = group
        self._counts = None  # layer-check: [rows checked, rows differing] on the device
        if self.check:
            if device is None:
                raise ValueError(f"{ENV}=layer-check needs the device of its check counts")
            self._counts = torch.zeros(2, dtype=torch.int64, device=device)
            self._last_log = time.monotonic()

    def owner(self, layer_id: int) -> int:
        owner = self._owner.get(int(layer_id))
        if owner is None:
            raise RuntimeError(f"{ENV}: layer {layer_id} is not a QSA layer of this stage {list(self.layer_ids)}")
        return owner

    def owns(self, layer_id: int) -> bool:
        return self.owner(layer_id) == self.rank

    def exchange(
        self, layer_id: int, blocks: Optional[torch.Tensor], rows: int, width: int, device
    ) -> torch.Tensor:
        """The owner passes its [rows, width] int32 block top-k (returned as is); the other rank passes None and
        gets the owner's values. Both ranks must call it for every QSA layer of a forward, in layer order, with the
        same rows (they serve the same batch), like any attention-TP collective."""
        if self._group is None:
            raise RuntimeError(f"{ENV}: this shard was built without the attention-TP group")
        if self.check:
            self._poll_check()
        owner = self.owner(layer_id)
        if self.rank == owner:
            if blocks is None:
                raise RuntimeError(f"{ENV}: the owner of layer {layer_id} has no top-k to hand off")
            if blocks.dtype != torch.int32 or tuple(blocks.shape) != (rows, width) or not blocks.is_contiguous():
                raise RuntimeError(
                    f"{ENV}: layer {layer_id} top-k must be contiguous int32 [{rows}, {width}], got "
                    f"{blocks.dtype} {tuple(blocks.shape)} contiguous={blocks.is_contiguous()}"
                )
            if rows == 0:
                return blocks
            wire = to_wire(blocks)
        else:
            if blocks is not None:
                raise RuntimeError(f"{ENV}: rank {self.rank} computed a top-k for layer {layer_id} it does not own")
            if rows == 0:
                return torch.empty((0, width), dtype=torch.int32, device=device)
            wire = torch.empty((rows, width), dtype=torch.int16, device=device)
        payload = wire.view(-1).view(torch.bfloat16)
        gathered = torch.empty((self.world * payload.numel(),), dtype=torch.bfloat16, device=payload.device)
        self._group.all_gather_into_tensor(gathered, payload)
        if self.rank == owner:
            return blocks
        return from_wire(gathered.view(self.world, -1)[owner].view(torch.int16)).view(rows, width)

    def verify(self, layer_id: int, received: torch.Tensor, local: torch.Tensor) -> None:
        """layer-check, the receiving rank: count the rows where the owner's top-k differs from this rank's own (device
        ops only, so it is captured into the CUDA graphs; _poll_check reads the counts)."""
        if not self.check:
            raise RuntimeError(f"{ENV}={_MODE}: verify() is the layer-check path")
        if self.owns(layer_id):
            raise RuntimeError(f"{ENV}: rank {self.rank} owns layer {layer_id}; it receives no top-k to check")
        if received.shape != local.shape or received.dtype != local.dtype:
            raise RuntimeError(
                f"{ENV}: layer {layer_id} received top-k {received.dtype} {tuple(received.shape)}, computed "
                f"{local.dtype} {tuple(local.shape)}"
            )
        if received.shape[0] == 0:
            return
        self._counts[0].add_(received.shape[0])
        self._counts[1].add_((received != local).any(dim=1).sum())

    def _poll_check(self) -> None:
        from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

        if get_is_capture_mode():
            return
        checked, differing = self._counts.tolist()
        if differing:
            raise RuntimeError(
                f"{ENV}=layer-check: {differing} of {checked} received top-k rows differ from the rows attention-TP "
                f"rank {self.rank} computed itself"
            )
        now = time.monotonic()
        if now - self._last_log >= CHECK_LOG_SECONDS:
            self._last_log = now
            logger.info(
                "%s=layer-check: %d top-k rows of the pair's QSA layers checked against this rank's own, 0 differ",
                ENV,
                checked,
            )


def build_layer_shard(
    layer_ids: Sequence[int], *, is_draft_worker: bool, context_len: int, draft_tokens: int, compress_ratio: int
) -> Optional[QSAIndexLayerShard]:
    """The shard of this rank's QSA pool, or None when the flag is off. Raises for every unsupported shape."""
    if not enabled():
        return None
    if is_draft_worker:
        raise NotImplementedError(
            f"{ENV}={_MODE} splits the target's QSA layers; a draft worker with a QSA pool (native MTP) keeps one "
            "layer. Serve the DFlash drafter or SPEC=none"
        )
    group = _attn_tp_group()
    _check_parallel(group.world_size)
    shard = QSAIndexLayerShard(
        layer_ids,
        group.rank_in_group,
        group.world_size,
        group,
        max_blocks=-(-(int(context_len) + int(draft_tokens)) // int(compress_ratio)),
        check=checking(),
        device=torch.device("cuda", torch.cuda.current_device()) if checking() else None,
    )
    logger.info(
        "%s=%s: attention-TP rank %d of %d owns the index of QSA layers %s (%d of %d; the others' top-k comes "
        "from the pair, int16 on the wire, up to %d blocks per sequence)%s",
        ENV,
        _MODE,
        shard.rank,
        shard.world,
        list(shard.owned_layer_ids),
        len(shard.owned_layer_ids),
        len(shard.layer_ids),
        shard.max_blocks,
        "; layer-check: every layer's index kept and computed here, the received top-k compared with it"
        if shard.check
        else "",
    )
    return shard


__all__ = [
    "ENV",
    "QSAIndexLayerShard",
    "WIRE_MAX_BLOCKS",
    "build_layer_shard",
    "checking",
    "enabled",
    "from_wire",
    "index_layers_per_rank",
    "mode",
    "require_off",
    "to_wire",
]
