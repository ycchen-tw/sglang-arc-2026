"""The drafter's embedding and lm_head, both taken from the target.

One GPU (G4): the target's lm_head is the whole vocabulary, `TargetHead`. Attention-TP group of N > 1 ranks (L4:
TP4 target, attention DP2, pairs): the target's lm_head is vocab-sharded over all TP ranks, so each rank adds the
rows of its counterparts in the other DP groups (read from the target checkpoint), takes the top-k of what it holds,
and the group merges by all-gather: `PairHead`. Both expose topk(x, k) -> (values, ids) over the full vocabulary,
which is all the proposal step (state_pool.fast_propose) reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


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
        logits = self(x.reshape(-1, x.shape[-1])).view(*x.shape[:-1], -1)
        return logits.topk(k, -1)


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
            v, i = F.linear(flat, w).topk(k, -1)
            values.append(v)
            ids.append(i + first)
        v, i = torch.cat(values, -1), torch.cat(ids, -1)
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
    """TargetHead when this rank's target lm_head is the whole vocabulary, else the group's PairHead."""
    from safetensors import safe_open

    from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    head = getattr(target_model, "lm_head", None)
    if head is None or head.weight.dtype != torch.bfloat16 or head.weight.shape[1] != hidden:
        raise RuntimeError(f"target lm_head is not a BF16 [*, {hidden}] matrix: {head}")
    si = head.shard_indices
    own = (si.org_vocab_start_index, si.org_vocab_end_index)
    if own == (0, vocab):
        if group.world_size != 1:
            raise RuntimeError(f"full-vocabulary lm_head on an attention-TP group of {group.world_size}")
        if tuple(head.weight.shape) != (vocab, hidden):
            raise RuntimeError(f"target lm_head {tuple(head.weight.shape)} != [{vocab}, {hidden}]")
        return TargetHead(head.weight)
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
    return PairHead(parts, vocab, group)
