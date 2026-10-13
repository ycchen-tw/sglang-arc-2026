"""The drafter's embedding and lm_head, both taken from the target.

One GPU (G4): the target's lm_head is the whole vocabulary, `TargetHead`. Attention-TP group of N > 1 ranks (L4:
TP4 target, attention DP2, pairs): the target's lm_head is vocab-sharded over all TP ranks, so each rank adds the
rows of its counterparts in the other DP groups (read from the target checkpoint), takes the top-k of what it holds,
and the group merges by all-gather: `PairHead`. Both expose topk(x, k) -> (values, ids) over the full vocabulary,
which is all the proposal step (state_pool.fast_propose) reads. ARC_DFLASH_DRAFT_VOCAB restricts either to a fixed
id list (`SubsetHead`, `SubsetPairHead`). Under ARC_TARGET_VOCAB the target head holds a slice of a kept-id list
(target_vocab.py) and the drafter's subset head takes its rows from that slice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.arc_dflash import target_vocab


# arc-forge ARC_DFLASH_FAST_TOPK (opt-in, default 0): the vocabulary top-k of TargetHead / PairHead by fast_topk
# (two Triton passes; the same set and order, ties to the lower id) instead of torch.topk
FAST_TOPK = os.environ.get("ARC_DFLASH_FAST_TOPK", "0")
if FAST_TOPK not in ("0", "1"):
    raise ValueError(f"ARC_DFLASH_FAST_TOPK must be 0 or 1, got {FAST_TOPK!r}")


def _vocab_topk(logits, k: int):
    """top-k over the last dimension of 2-D logits."""
    if FAST_TOPK == "1":
        from .fast_topk import topk

        return topk(logits, k)
    return logits.topk(k, -1)


# arc-forge ARC_DFLASH_FAST_TOPK (opt-in, default 0): the vocabulary top-k of TargetHead / PairHead by fast_topk
# (two Triton passes; the same set and order, ties to the lower id) instead of torch.topk
FAST_TOPK = os.environ.get("ARC_DFLASH_FAST_TOPK", "0")
if FAST_TOPK not in ("0", "1"):
    raise ValueError(f"ARC_DFLASH_FAST_TOPK must be 0 or 1, got {FAST_TOPK!r}")


def _vocab_topk(logits, k: int):
    """top-k over the last dimension of 2-D logits."""
    if FAST_TOPK == "1":
        from .fast_topk import topk

        return topk(logits, k)
    return logits.topk(k, -1)


class TargetEmbedding(nn.Module):
    """The target's own token embedding (identical tensor)."""

    def __init__(self, module):
        super().__init__()
        self._target = [module]  # not a submodule: its weights belong to the target

    def forward(self, ids):
        return self._target[0](ids.reshape(-1)).view(*ids.shape, -1)


class TargetHead(nn.Module):
    """lm_head shared with the target, full vocabulary on this GPU."""

    def __init__(self, weight):
        super().__init__()
        self._weight = [weight]

    @property
    def weight(self):
        return self._weight[0]

    def forward(self, x):
        return F.linear(x, self._weight[0])

    def topk(self, x, k: int):
        if FAST_TOPK == "1":
            values, ids = _vocab_topk(self(x.reshape(-1, x.shape[-1])), k)
            return values.view(*x.shape[:-1], k), ids.view(*x.shape[:-1], k)
        logits = self(x.reshape(-1, x.shape[-1])).view(*x.shape[:-1], -1)
        return logits.topk(k, -1)


class SubsetHead(nn.Module):
    """The target lm_head restricted to a fixed draft vocabulary (ARC_DFLASH_DRAFT_VOCAB, a JSON {"ids": [...]}):
    the drafter's top-k candidates come from those rows only, mapped back to vocabulary ids. The draft head only
    proposes -- rejection sampling still verifies against the target's full distribution, so the output does not
    change; a target token outside the subset is never proposed (its position is rejected and resampled). The
    hs110 Kaggle run's 110 codex rollouts (20.7M output tokens) used 22,802 distinct ids and the top 16,384 cover
    99.92% of them, while the full head is a [rows, 248k] GEMM plus a top-k over it every draft step."""

    def __init__(self, weight, ids: torch.Tensor):
        super().__init__()
        self.register_buffer("ids", ids, persistent=False)
        self.register_buffer("sub", weight.index_select(0, ids).contiguous(), persistent=False)

    def topk(self, x, k: int):
        flat = x.reshape(-1, x.shape[-1])
        values, local = F.linear(flat, self.sub).topk(k, -1)
        return values.view(*x.shape[:-1], k), self.ids[local].view(*x.shape[:-1], k)


def draft_vocab(vocab: int, device):
    """ARC_DFLASH_DRAFT_VOCAB: path of {"ids": [...]} (unique ids in [0, vocab)), or None when unset."""
    path = os.environ.get("ARC_DFLASH_DRAFT_VOCAB")
    if not path:
        return None
    ids = json.loads(Path(path).read_text())["ids"]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= vocab:
        raise ValueError(f"ARC_DFLASH_DRAFT_VOCAB {path}: ids must be unique and in [0, {vocab}) ({len(ids)} given)")
    return torch.tensor(sorted(ids), dtype=torch.int64, device=device)


class PairHead(nn.Module):
    """Top-k of the target lm_head over the full vocabulary for one attention-TP group.

    `parts` are (weight [rows, H] BF16, first vocab id) pairs this rank scores; together the group covers the
    vocabulary exactly once (checked at build time)."""

    def __init__(self, parts, vocab: int, group):
        super().__init__()
        self.parts = [(w, int(first)) for w, first in parts]
        self.vocab, self.group = vocab, group
        for i, (w, _) in enumerate(self.parts):
            self.register_buffer(f"part{i}", w, persistent=False)

    def topk(self, x, k: int):
        flat = x.reshape(-1, x.shape[-1])
        values, ids = [], []
        for w, first in self.parts:
            v, i = _vocab_topk(F.linear(flat, w), k)
            values.append(v)
            ids.append(i + first)
        v, i = torch.cat(values, -1), torch.cat(ids, -1)
        if self.group.world_size > 1:
            v, i = self.group.all_gather(v.contiguous(), dim=-1), self.group.all_gather(i.contiguous(), dim=-1)
        top, pick = v.topk(k, -1)
        return top.view(*x.shape[:-1], k), i.gather(-1, pick).view(*x.shape[:-1], k)


class SubsetPairHead(nn.Module):
    """PairHead over a fixed draft vocabulary (ARC_DFLASH_DRAFT_VOCAB): each rank keeps the listed rows of the
    vocabulary it holds (build_head checked that the group holds every id exactly once), takes their top-k, and the
    group merges by all-gather exactly as PairHead does. Proposals only, as SubsetHead: the target still verifies
    against its full distribution. On L4 (dp-lm-head pairs) the full head is a [rows, 124k] BF16 GEMM per rank,
    ~2.9 ms of each draft step at C48 (e1 trace)."""

    MIN_ROWS = 64  # every rank must hold at least the draft top-k

    def __init__(self, parts, ids: torch.Tensor, vocab: int, group):
        """parts: (weight [rows, H], row_ids [rows] int64 vocabulary ids) this rank scores; the listed ids among them
        are kept, in row order."""
        super().__init__()
        weights, maps = [], []
        for w, row_ids in parts:
            if row_ids.shape != (w.shape[0],):
                raise ValueError(f"SubsetPairHead: weight {tuple(w.shape)} with row ids {tuple(row_ids.shape)}")
            pos = torch.isin(row_ids, ids).nonzero().squeeze(1)
            if pos.numel():
                weights.append(w.index_select(0, pos))
                maps.append(row_ids[pos])
        rows = sum(m.numel() for m in maps)
        if rows < self.MIN_ROWS:
            raise RuntimeError(f"ARC_DFLASH_DRAFT_VOCAB: this rank holds {rows} of the {ids.numel()} draft ids "
                               f"(< {self.MIN_ROWS}); the list does not span the group's vocabulary shards")
        # Zero rows up to a multiple of 64, sliced off before the top-k: an odd row count (17,131 on an L4 pair rank)
        # leaves the logits' row stride unaligned and cuBLAS falls back to an align1 kernel (L4 654 us per draft
        # step; 4090 215 -> 138 us padded).
        sub = torch.cat(weights)
        self.rows = sub.shape[0]
        pad = -self.rows % 64
        if pad:
            sub = torch.cat([sub, sub.new_zeros(pad, sub.shape[1])])
        self.register_buffer("sub", sub.contiguous(), persistent=False)
        self.register_buffer("map", torch.cat(maps), persistent=False)
        self.vocab, self.group = vocab, group

    def topk(self, x, k: int):
        flat = x.reshape(-1, x.shape[-1])
        v, local = F.linear(flat, self.sub)[:, : self.rows].topk(k, -1)
        i = self.map[local]
        if self.group.world_size > 1:
            v, i = self.group.all_gather(v.contiguous(), dim=-1), self.group.all_gather(i.contiguous(), dim=-1)
        top, pick = v.topk(k, -1)
        return top.view(*x.shape[:-1], k), i.gather(-1, pick).view(*x.shape[:-1], k)


def vocab_shard(vocab: int, rank: int, size: int) -> tuple[int, int]:
    """[start, end) of the original vocabulary on TP rank `rank` (VocabParallelEmbedding with page 64 padding)."""
    padded = -(-vocab // 64) * 64
    per = padded // size
    if per * size != padded:
        raise ValueError(f"padded vocab {padded} does not split over {size} ranks")
    return min(rank * per, vocab), min((rank + 1) * per, vocab)


def build_head(target_model, model_dir: Path, device, vocab: int, hidden: int, group):
    """TargetHead when this rank's target lm_head is the whole vocabulary, else the group's PairHead (SubsetHead /
    SubsetPairHead under ARC_DFLASH_DRAFT_VOCAB)."""
    from safetensors import safe_open

    from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    head = getattr(target_model, "lm_head", None)
    if head is None or head.weight.dtype != torch.bfloat16 or head.weight.shape[1] != hidden:
        raise RuntimeError(f"target lm_head is not a BF16 [*, {hidden}] matrix: {head}")
    if target_vocab.is_pruned(head):
        return _pruned_target_head(head, vocab, group)
    si = head.shard_indices
    own = (si.org_vocab_start_index, si.org_vocab_end_index)
    if own == (0, vocab):
        if group.world_size != 1:
            raise RuntimeError(f"full-vocabulary lm_head on an attention-TP group of {group.world_size}")
        if tuple(head.weight.shape) != (vocab, hidden):
            raise RuntimeError(f"target lm_head {tuple(head.weight.shape)} != [{vocab}, {hidden}]")
        ids = draft_vocab(vocab, head.weight.device)
        return TargetHead(head.weight) if ids is None else SubsetHead(head.weight, ids)
    rows = own[1] - own[0]
    parts = [(head.weight[:rows], own[0])]
    covered = [own]
    if not getattr(head, "use_attn_tp_group", False):  # --enable-dp-lm-head: the group's shards are the vocabulary
        tp, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        if vocab_shard(vocab, tp, tp_size) != own:
            raise RuntimeError(f"lm_head shard {own} != expected {vocab_shard(vocab, tp, tp_size)} on TP rank {tp}")
        ranks = list(group.ranks)
        if ranks != list(range(ranks[0], ranks[0] + len(ranks))) or tp_size % len(ranks):
            raise RuntimeError(f"attention-TP group {ranks} is not a contiguous block of TP{tp_size}")
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        extra = [d * len(ranks) + group.rank_in_group for d in range(tp_size // len(ranks)) if d * len(ranks) != ranks[0]]
        with safe_open(str(model_dir / index["lm_head.weight"]), framework="pt", device="cpu") as f:
            sl = f.get_slice("lm_head.weight")
            for r in extra:
                start, end = vocab_shard(vocab, r, tp_size)
                parts.append((sl[start:end].to(device=device, dtype=torch.bfloat16).contiguous(), start))
                covered.append((start, end))
    if group.world_size > 1:  # the group together must cover [0, vocab) exactly once
        gathered = [None] * group.world_size
        torch.distributed.all_gather_object(gathered, covered, group=group.cpu_group)
        spans = sorted(tuple(s) for c in gathered for s in c)
    else:
        spans = sorted(covered)
    edge = 0
    for start, end in spans:
        if start != edge:
            raise RuntimeError(f"lm_head spans {spans} do not tile [0, {vocab})")
        edge = end
    if edge != vocab:
        raise RuntimeError(f"lm_head spans {spans} stop at {edge}, vocab {vocab}")
    ids = draft_vocab(vocab, device)
    if ids is None:
        return PairHead(parts, vocab, group)
    return SubsetPairHead([(w, torch.arange(first, first + w.shape[0], device=w.device)) for w, first in parts],
                          ids, vocab, group)


def _pruned_target_head(head, vocab: int, group):
    """ARC_TARGET_VOCAB: the target head holds this rank's slice of the kept ids (target_vocab.prune_lm_head); the
    draft vocabulary must lie inside the list, and each draft id is then on exactly one rank of the group."""
    t = target_vocab.tables()
    if group.world_size != t["n"] or group.rank_in_group != t["rank"]:
        raise RuntimeError(f"ARC_TARGET_VOCAB: drafter group rank {group.rank_in_group}/{group.world_size}, target "
                           f"head sliced for {t['rank']}/{t['n']}")
    ids = draft_vocab(vocab, head.weight.device)
    if ids is None:
        raise RuntimeError("ARC_TARGET_VOCAB needs ARC_DFLASH_DRAFT_VOCAB (the drafter proposes inside the target list)")
    off = ids[t["g2c"][ids] < 0]
    if off.numel():
        raise RuntimeError(f"ARC_DFLASH_DRAFT_VOCAB: {off.numel()} draft ids are not in ARC_TARGET_VOCAB, e.g. "
                           f"{off[:5].tolist()}")
    if head.weight.shape[0] != t["mine"].numel():
        raise RuntimeError(f"ARC_TARGET_VOCAB: head {tuple(head.weight.shape)} vs {t['mine'].numel()} kept rows")
    return SubsetPairHead([(head.weight, t["mine"])], ids, vocab, group)
