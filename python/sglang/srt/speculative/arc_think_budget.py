"""ARC_THINK_BUDGET=<N> (arc-forge, opt-in): close a request's thinking block after N output tokens.

Why: codex (arc-forge work/arc_code) sends no max_output_tokens; once in a while one response thinks for 30k-73k
tokens. A max_output_tokens cut is worse than nothing: sglang answers ``response.incomplete`` and codex 0.154 retries
the identical request up to stream_max_retries times. This closes the thinking block instead, so the model goes on
to its tool call and the response completes normally.

What is emitted: exactly what the chat template renders for that turn on the next request. Qwen3.5+ templates render
an assistant turn as ``<think>\\n`` + ``reasoning_content|trim`` + ``\\n</think>\\n\\n`` + ..., so the forced close is
``\\n</think>`` right after a non-whitespace token. Anything else (``...foo</think>``, or ``...foo\\n\\n\\n</think>``)
re-renders as different text: the QSA HiSparse session replay (h14) stops at it, the record falls back to its
anchor or is released (re-prefill of the whole reply, or the whole context), and the DFlash session park misses.

How (EAGLE v2 chain verify, ``run_eagle_verify``; candidates[:, 0] is the last committed token, logits row j is the
target's distribution after candidates[:, j]):

- CPU, per request: ``arc_think_open`` (the prompt ends inside ``<think>``), ``arc_think_scan`` (how far
  ``output_ids`` was scanned for ``</think>``), ``arc_think_armed``. A request is armed once ``len(output_ids)``
  reaches N while its block is open, and disarmed when its ``</think>`` shows up in ``output_ids``. With the overlap
  scheduler ``output_ids`` lags one step, so the CPU view alone cannot say what the last step committed.
- GPU, per armed request and verify row j (no host sync): ``closed[req_pool_idx]`` says a ``</think>`` was already
  committed (set after acceptance from the accepted tokens). Rows at or after a ``</think>`` among candidates[:, :j+1]
  are left alone. Otherwise, by the class of candidates[:, j]: a lone ``\\n`` -> force ``</think>``; whitespace or a
  partial UTF-8 piece -> leave the row alone this time (closing there would not round-trip); anything else -> force
  ``\\n``. Forcing a row is a one-hot target distribution; rejection sampling then commits exactly that token at that
  position (the draft if it matches, else the bonus).

Only the rows of armed requests are touched; with nothing armed the hook is a Python loop over the batch.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

ENV = "ARC_THINK_BUDGET"
#: how far back from the prompt's end to look for the generation prompt's ``<think>``
PROMPT_TAIL = 16

CLS_NORMAL, CLS_NL, CLS_WAIT = 0, 1, 2


def budget_from_env() -> Optional[int]:
    raw = os.environ.get(ENV)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{ENV}={raw!r} is not an integer") from None
    if value < 1:
        raise RuntimeError(f"{ENV}={raw!r}: the budget must be a positive token count")
    return value


def _single_id(tokenizer, text: str, special: bool) -> int:
    if special:
        tid = tokenizer.convert_tokens_to_ids(text)
        unk = getattr(tokenizer, "unk_token_id", None)
        if not isinstance(tid, int) or tid < 0 or (unk is not None and tid == unk):
            raise RuntimeError(f"{ENV}: the tokenizer has no {text!r} token")
        return tid
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{ENV}: {text!r} encodes to {ids}, not one token")
    return ids[0]


def token_classes(tokenizer, size: int) -> list[int]:
    """CLS_NL for the lone ``\\n`` token, CLS_WAIT for tokens that decode to whitespace only or to a partial UTF-8
    sequence (U+FFFD) and for ids past the tokenizer, CLS_NORMAL for the rest."""
    n = min(len(tokenizer), size)
    texts = tokenizer.batch_decode([[i] for i in range(n)], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    out = []
    for text in texts:
        if text == "\n":
            out.append(CLS_NL)
        elif text.strip() == "" or "�" in text:
            out.append(CLS_WAIT)
        else:
            out.append(CLS_NORMAL)
    out.extend([CLS_WAIT] * (size - n))
    return out


class ThinkBudget:
    """One per scheduler process (``setup``); GPU tensors are built on the first verify."""

    def __init__(self, budget: int, tokenizer):
        self.budget = budget
        self.tokenizer = tokenizer
        self.start_id = _single_id(tokenizer, "<think>", special=True)
        self.end_id = _single_id(tokenizer, "</think>", special=True)
        self.nl_id = _single_id(tokenizer, "\n", special=False)
        self.cls: Optional[torch.Tensor] = None
        self.closed: Optional[torch.Tensor] = None
        self.armed = 0
        self.disarmed = 0
        self.skipped_grammar = 0

    # ---- CPU side -------------------------------------------------------------------------------------------

    def _open_at_prompt_end(self, req: "Req") -> bool:
        ids = req.origin_input_ids
        for tid in reversed(ids[-PROMPT_TAIL:]):
            if tid == self.start_id:
                return True
            if tid == self.end_id:
                return False
        return False

    def _update(self, req: "Req") -> None:
        """Advance one request's CPU state from what ``output_ids`` shows now."""
        if req.arc_think_open is None:
            req.arc_think_open = self._open_at_prompt_end(req)
            req.arc_think_scan = 0
        if not req.arc_think_open:
            return
        out = req.output_ids
        end = len(out)
        for pos in range(req.arc_think_scan, end):
            if out[pos] == self.end_id:
                req.arc_think_open = False
                if req.arc_think_armed:
                    req.arc_think_armed = False
                    self.disarmed += 1
                    logger.info("ARC think budget: %s (session %s) closed at output token %d (armed at %d)",
                                req.rid, req.session_id, pos, req.arc_think_armed_at)
                break
        req.arc_think_scan = end

    def select(self, batch: "ScheduleBatch") -> list[int]:
        """Batch rows of armed requests; arms the ones that just reached the budget."""
        rows = []
        for i, req in enumerate(batch.reqs):
            self._update(req)
            if not req.arc_think_open:
                continue
            if not req.arc_think_armed and len(req.output_ids) >= self.budget:
                if req.grammar is not None:
                    req.arc_think_open = False  # stop looking at it: a forced token could break the grammar
                    self.skipped_grammar += 1
                    logger.warning("ARC think budget: %s has a grammar; its thinking is not cut", req.rid)
                    continue
                req.arc_think_armed = True
                req.arc_think_armed_at = len(req.output_ids)
                req.arc_think_slot = -1
                self.armed += 1
                logger.info("ARC think budget: %s (session %s) armed at %d output tokens (budget %d; %d armed, "
                            "%d closed so far)", req.rid, req.session_id, len(req.output_ids), self.budget, self.armed,
                            self.disarmed)
            if req.arc_think_armed:
                rows.append(i)
        return rows

    # ---- GPU side -------------------------------------------------------------------------------------------

    def _ensure_tensors(self, vocab: int, pool_rows: int, device) -> None:
        if self.cls is None:
            self.cls = torch.tensor(token_classes(self.tokenizer, vocab), dtype=torch.int8, device=device)
            if int(self.cls[self.nl_id]) != CLS_NL:
                raise RuntimeError(f"{ENV}: the newline token {self.nl_id} did not classify as a lone newline")
            self.closed = torch.zeros(pool_rows, dtype=torch.bool, device=device)
        if self.cls.shape[0] != vocab or self.closed.shape[0] != pool_rows:
            raise RuntimeError(f"{ENV}: vocab {vocab} / request pool {pool_rows} changed from "
                               f"{self.cls.shape[0]} / {self.closed.shape[0]}")

    def apply(self, batch: "ScheduleBatch", rows: list[int], candidates: torch.Tensor, logits: torch.Tensor,
              num_draft_tokens: int, pool_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Force rows of the armed requests in place; returns (their batch rows, their req_pool_indices) for
        ``commit``."""
        bs = len(batch.reqs)
        vocab = logits.shape[-1]
        if logits.shape[0] != bs * num_draft_tokens or candidates.numel() != bs * num_draft_tokens:
            raise RuntimeError(f"{ENV}: verify logits {tuple(logits.shape)} / candidates {candidates.numel()} "
                               f"for {bs} requests x {num_draft_tokens} rows")
        device = logits.device
        self._ensure_tensors(vocab, pool_rows, device)
        fresh = []
        for i in rows:
            req = batch.reqs[i]
            if req.arc_think_slot != req.kv.req_pool_idx:
                # armed just now, or the request came back from a retract on another slot: its CPU view is
                # complete then, and the block is still open, so nothing was committed past it
                req.arc_think_slot = req.kv.req_pool_idx
                fresh.append(req.kv.req_pool_idx)
        row_t = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        slots = batch.req_pool_indices.index_select(0, row_t).to(torch.int64)
        if fresh:
            fresh_t = torch.tensor(fresh, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
            self.closed.index_fill_(0, fresh_t, False)
        cand = candidates.view(bs, num_draft_tokens).index_select(0, row_t).to(torch.int64)
        cls = self.cls[cand]
        seen = self.closed[slots].unsqueeze(1) | (torch.cumsum((cand == self.end_id).to(torch.int32), dim=1) > 0)
        force = ~seen & (cls != CLS_WAIT)
        token = torch.where(cls == CLS_NL, torch.full_like(cand, self.end_id), torch.full_like(cand, self.nl_id))
        view = logits.view(bs, num_draft_tokens, vocab)
        sub = view.index_select(0, row_t)
        onehot = torch.full_like(sub, float("-inf"))
        onehot.scatter_(2, token.unsqueeze(2), 0.0)
        view.index_copy_(0, row_t, torch.where(force.unsqueeze(2), onehot, sub))
        return row_t, slots

    def commit(self, handle: tuple[torch.Tensor, torch.Tensor], predict: torch.Tensor,
               accept_index: torch.Tensor) -> None:
        """closed[slot] |= a ``</think>`` among the tokens committed for that request this step."""
        row_t, slots = handle
        idx = accept_index.index_select(0, row_t).to(torch.int64)
        toks = predict.view(-1)[idx.clamp(min=0)]
        hit = ((toks == self.end_id) & (idx >= 0)).any(dim=1)
        self.closed[slots] |= hit


_STATE: Optional[ThinkBudget] = None


def setup(spec_algorithm, server_args_spec_topk: Optional[int], tokenizer) -> Optional[ThinkBudget]:
    """Called once from Scheduler.__init__: validates the deployment and builds the CPU state."""
    global _STATE
    budget = budget_from_env()
    if budget is None:
        _STATE = None
        return None
    if not (spec_algorithm.is_eagle() or spec_algorithm.is_uno()):
        raise RuntimeError(f"{ENV} is applied in the EAGLE v2 verify (run_eagle_verify) only; the speculative "
                           f"algorithm here is {spec_algorithm!r}")
    if server_args_spec_topk != 1:
        raise RuntimeError(f"{ENV} needs a chain verify (--speculative-eagle-topk 1), got {server_args_spec_topk}")
    if tokenizer is None:
        raise RuntimeError(f"{ENV} needs the tokenizer (no --skip-tokenizer-init)")
    _STATE = ThinkBudget(budget, tokenizer)
    logger.info("ARC think budget: %d tokens (<think> %d, </think> %d, newline %d)",
                budget, _STATE.start_id, _STATE.end_id, _STATE.nl_id)
    return _STATE


def active() -> Optional[ThinkBudget]:
    return _STATE
