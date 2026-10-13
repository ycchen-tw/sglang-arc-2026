"""The target's output vocabulary restricted to a fixed id list (ARC_TARGET_VOCAB, a JSON {"ids": [...]}).

The target lm_head keeps only the listed rows (sorted ascending), split evenly over the group its logits are sharded
on (the attention-TP group: dp-lm-head, or pure TP with one attention-DP group). Each rank's logits are then a
[rows, n_keep / N] slice of a compact id space, compact id c being vocabulary id c2g[c]. Every computed logit is the
full head's logit for its id; an id off the list gets none, i.e. probability 0:
  - sharded verify (sharded_verify.sample) runs in the compact space: the drafter's ids go in through g2c (an id off
    the list maps to -1, which the kernels treat as having p = 0) and the draws come back through c2g;
  - every other path (LogitsProcessor._get_logits, sharded_verify.gather_full) scatters the compact logits into a
    full-width row of -inf, so the stock samplers see the same distribution.
The DFlash drafter's subset head takes its rows from this head (heads.build_head), so its draft vocabulary has to lie
inside the list. L4 (dp-lm-head, 124,160 rows per rank): 98,304 kept ids cut each rank's head 636 -> 252 MB and its
verify GEMM in proportion; ARC2 codex rollouts of unseen tasks need an id off that list 0.3 times per task.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENV = "ARC_TARGET_VOCAB"
ROW_ALIGN = 64  # rows per rank stay a multiple of this (aligned lm_head GEMM)
_T: dict = {}


def path() -> Path | None:
    value = os.environ.get(ENV)
    return Path(value) if value else None


def active() -> bool:
    return bool(_T)


def tables() -> dict:
    if not _T:
        raise RuntimeError(f"{ENV} is set but the target lm_head was not restricted (target_vocab.prune_lm_head)")
    return _T


def is_pruned(lm_head) -> bool:
    """True for the restricted target head (other heads, e.g. a drafter's, keep the stock path)."""
    return bool(_T) and _T["head"] is lm_head


def load_ids(file: Path, vocab_size: int) -> list[int]:
    ids = json.loads(file.read_text())["ids"]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= vocab_size:
        raise ValueError(f"{ENV} {file}: ids must be unique and in [0, {vocab_size}) ({len(ids)} given)")
    return sorted(ids)


def eos_ids(model_dir: Path) -> set[int]:
    found = set()
    for name in ("generation_config.json", "config.json"):
        file = model_dir / name
        if not file.is_file():
            continue
        cfg = json.loads(file.read_text())
        for value in (cfg.get("eos_token_id"), (cfg.get("text_config") or {}).get("eos_token_id")):
            if value is None:
                continue
            found.update(value if isinstance(value, list) else [value])
    return found


def build_tables(ids: list[int], vocab_size: int, n: int, rank: int, device) -> dict:
    """c2g [K] (compact -> vocabulary id), g2c [vocab] (vocabulary -> compact, -1 off the list), this rank's slice."""
    if len(ids) % (ROW_ALIGN * n):
        raise ValueError(f"{ENV}: {len(ids)} ids do not split into {n} ranks x a multiple of {ROW_ALIGN} rows")
    c2g = torch.tensor(ids, dtype=torch.int64, device=device)
    g2c = torch.full((vocab_size,), -1, dtype=torch.int64, device=device)
    g2c[c2g] = torch.arange(len(ids), dtype=torch.int64, device=device)
    width = len(ids) // n
    return {"c2g": c2g, "g2c": g2c, "size": len(ids), "vocab": vocab_size, "n": n, "rank": rank, "width": width,
            "mine": c2g[rank * width:(rank + 1) * width]}


def to_compact(ids: torch.Tensor) -> torch.Tensor:
    """Vocabulary ids (>= 0, or -1 = none) -> compact ids; an id off the list becomes -1 (p = 0 in sharded verify)."""
    g2c = tables()["g2c"]
    return torch.where(ids >= 0, g2c[ids.clamp_min(0)], -1)


def to_vocab(compact: torch.Tensor) -> torch.Tensor:
    """Compact ids (>= 0) -> vocabulary ids, same dtype."""
    return tables()["c2g"][compact.long()].to(compact.dtype)


def expand(compact_logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """[rows, K] compact logits -> [rows, vocab_size] with -inf off the list."""
    t = tables()
    if compact_logits.shape[-1] != t["size"] or vocab_size != t["vocab"]:
        raise RuntimeError(f"{ENV}: logits {tuple(compact_logits.shape)} / vocab {vocab_size} vs tables "
                           f"{t['size']} / {t['vocab']}")
    full = compact_logits.new_full((compact_logits.shape[0], vocab_size), float("-inf"))
    full.index_copy_(1, t["c2g"], compact_logits)
    return full


def prune_lm_head(model, model_dir: Path, vocab_size: int) -> None:
    """Replace this rank's target lm_head rows by its slice of the kept ids (read from the checkpoint), once, after
    the target loads and before CUDA graphs and the DFlash drafter."""
    from safetensors import safe_open

    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.runtime_context import get_parallel

    if _T:
        raise RuntimeError(f"{ENV}: the target lm_head is already restricted")
    head = model.lm_head
    w = head.weight
    method = type(head.quant_method).__name__
    if w.dtype != torch.bfloat16 or w.dim() != 2 or method not in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"):
        raise RuntimeError(f"{ENV}: target lm_head {w.dtype} {tuple(w.shape)} {method}, need an unquantized BF16 head")
    ps = get_parallel()
    if not head.use_attn_tp_group and ps.attn_dp_size != 1:
        raise RuntimeError(f"{ENV}: lm_head sharded over TP{ps.tp_size} with attention DP{ps.attn_dp_size} "
                           "(needs dp-lm-head or one attention-DP group)")
    group = get_attn_tp_group()
    n, rank = group.world_size, group.rank_in_group
    if head.tp_size != n:
        raise RuntimeError(f"{ENV}: lm_head over {head.tp_size} ranks, attention-TP group of {n}")
    file = path()
    ids = load_ids(file, vocab_size)
    missing_eos = eos_ids(model_dir) - set(ids)
    if missing_eos:
        raise RuntimeError(f"{ENV} {file}: EOS ids {sorted(missing_eos)} are not in the list")
    t = build_tables(ids, vocab_size, n, rank, w.device)
    mine = t["mine"].cpu()
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(model_dir / index["lm_head.weight"]), framework="pt", device="cpu") as f:
        full = f.get_tensor("lm_head.weight")
    if full.dtype != torch.bfloat16 or full.shape[1] != w.shape[1] or full.shape[0] <= int(mine.max()):
        raise RuntimeError(f"{ENV}: checkpoint lm_head.weight {full.dtype} {tuple(full.shape)}")
    rows = full.index_select(0, mine).contiguous()
    del full
    # the kept ids inside this rank's stock shard must be the bytes it loaded
    lo, hi = head.shard_indices.org_vocab_start_index, head.shard_indices.org_vocab_end_index
    inside = ((mine >= lo) & (mine < hi)).nonzero().squeeze(1)
    if inside.numel() and not torch.equal(rows[inside].to(w.device), w.data[(mine[inside] - lo).to(w.device)]):
        raise RuntimeError(f"{ENV}: checkpoint rows differ from the loaded lm_head shard [{lo}, {hi})")
    checked = [None] * n
    torch.distributed.all_gather_object(checked, int(inside.numel()), group=group.cpu_group)
    if sum(checked) == 0:
        raise RuntimeError(f"{ENV}: no kept id falls in any rank's stock shard, the self-check compared nothing")
    before = torch.cuda.memory_allocated(w.device)
    old_rows = w.shape[0]
    head.weight = torch.nn.Parameter(rows.to(w.device), requires_grad=False)
    del w
    torch.cuda.empty_cache()
    freed = before - torch.cuda.memory_allocated(head.weight.device)
    t["head"] = head
    _T.update(t)
    logger.info("%s %s: %d of %d ids kept, attention-TP rank %d/%d holds %d rows (was %d), cuda_bytes_freed=%d, "
                "self-check %d rows (group %s)", ENV, file.name, len(ids), vocab_size, rank, n, t["width"], old_rows,
                freed, int(inside.numel()), checked)
