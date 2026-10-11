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

After the close, the template renders ``</think>\\n\\n`` and then the content (whitespace-only content renders as
nothing). A forced close is followed by ``\\n\\n`` and then a second ``\\n\\n`` in 0.5% of G4 c110 forced closes (never
after a natural one), which re-renders as one: so after a forced ``</think>`` the next token is forced to ``\\n\\n``
and the one after that may not be whitespace, ``<think>`` or ``</think>`` (anything else -- text, a tool call, a
compaction summary, end of turn -- stays the model's choice).

How (EAGLE v2 chain verify, ``run_eagle_verify``; candidates[:, 0] is the last committed token, logits row j is the
target's distribution after candidates[:, j]):

- CPU, per request: ``arc_think_open`` (the prompt ends inside ``<think>``), ``arc_think_scan`` (how far
  ``output_ids`` was scanned for ``</think>``), ``arc_think_end_at`` (where it was found), ``arc_think_armed``. A
  request is armed once ``len(output_ids)`` reaches N while its block is open, and disarmed once ``output_ids`` shows
  its ``</think>`` and the two tokens after it. With the overlap scheduler ``output_ids`` lags one step, so the CPU
  view alone cannot say what the last step committed.
- GPU, per armed request (no host sync): ``state[req_pool_idx]``, advanced over the accepted tokens after each
  verify: while thinking, WAIT (the last token is whitespace or a partial UTF-8 piece), OK (a word), NLOK (a word,
  then a lone ``\\n``); then END (``</think>`` is the last token), NN (``</think>\\n\\n``), DONE. Row j's state is
  that state advanced over candidates[:, 1:j+1]: OK -> force ``\\n``; NLOK -> force ``</think>``; WAIT -> leave the
  row alone (a close there would not round-trip: ``|trim`` drops the whitespace); END -> force ``\\n\\n``; NN -> mask
  whitespace, ``<think>``, ``</think>``. Forcing a row is a one-hot target distribution; rejection sampling then
  commits exactly that token at that position (the draft if it matches, else the bonus).

Only the rows of armed requests are touched; with nothing armed the hook is a Python loop over the batch.
When a request is armed, the step in flight is not in ``output_ids`` yet: its state starts from the last committed
token alone (a word -> OK, ``</think>`` -> END, anything else -> WAIT). Known hole: a request that closes on its own
inside that step with tokens after its ``</think>`` is taken as still thinking, so a ``\\n`` could be forced into its
content; that needs a natural close within ~3 tokens of the budget.
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

CLS_NORMAL, CLS_NL, CLS_WS, CLS_PARTIAL, CLS_PAD = 0, 1, 2, 3, 4
# thinking: WAIT (last token whitespace / partial: no close here), OK (last token a word: a newline may follow),
# NLOK (a word, then a lone newline: </think> may follow); then END (</think>), NN (</think> + blank line), DONE
ST_WAIT, ST_OK, ST_NLOK, ST_END, ST_NN, ST_DONE = 0, 1, 2, 3, 4, 5


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
    """CLS_NL for the lone ``\\n`` token, CLS_WS for other tokens that decode to whitespace only, CLS_PARTIAL for a
    partial UTF-8 sequence (U+FFFD), CLS_PAD for ids past the tokenizer (the model's padded vocab), CLS_NORMAL else."""
    n = min(len(tokenizer), size)
    texts = tokenizer.batch_decode([[i] for i in range(n)], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
    out = []
    for text in texts:
        if text == "\n":
            out.append(CLS_NL)
        elif text.strip() == "":
            out.append(CLS_WS)
        elif "\ufffd" in text:
            out.append(CLS_PARTIAL)
        else:
            out.append(CLS_NORMAL)
    out.extend([CLS_PAD] * (size - n))
    return out


class ThinkBudget:
    """One per scheduler process (``setup``); GPU tensors are built on the first verify."""

    def __init__(self, budget: int, tokenizer):
        self.budget = budget
        self.tokenizer = tokenizer
        self.start_id = _single_id(tokenizer, "<think>", special=True)
        self.end_id = _single_id(tokenizer, "</think>", special=True)
        self.nl_id = _single_id(tokenizer, "\n", special=False)
        self.nn_id = _single_id(tokenizer, "\n\n", special=False)
        self.cls: Optional[torch.Tensor] = None
        self.banned: Optional[torch.Tensor] = None  # [vocab] bool: not allowed right after a forced </think>\n\n
        self.state: Optional[torch.Tensor] = None  # [request pool] int8: ST_*
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
        out = req.output_ids
        end = len(out)
        if req.arc_think_open:
            for pos in range(req.arc_think_scan, end):
                if out[pos] == self.end_id:
                    req.arc_think_open = False
                    req.arc_think_end_at = pos
                    break
            req.arc_think_scan = end
        e = req.arc_think_end_at
        if req.arc_think_armed and e >= 0 and end >= e + 3:
            req.arc_think_armed = False
            self.disarmed += 1
            logger.info("ARC think budget: %s (session %s) closed at output token %d (armed at %d), then %s",
                        req.rid, req.session_id, e, req.arc_think_armed_at, list(out[e + 1 : e + 3]))

    def _cpu_state(self, req: "Req") -> int:
        """the state if ``output_ids`` shows the close, else -1 (taken from the last committed token on the GPU)"""
        e = req.arc_think_end_at
        if e < 0:
            return -1
        tail = req.output_ids[e + 1 : e + 3]
        if len(tail) == 0:
            return ST_END
        if tail[0] != self.nn_id:
            return ST_DONE
        return ST_NN if len(tail) == 1 else ST_DONE

    def select(self, batch: "ScheduleBatch") -> list[int]:
        """Batch rows of armed requests; arms the ones that just reached the budget."""
        rows = []
        for i, req in enumerate(batch.reqs):
            self._update(req)
            if req.arc_think_armed:
                rows.append(i)
                continue
            if not req.arc_think_open or len(req.output_ids) < self.budget:
                continue
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
            rows.append(i)
        return rows

    # ---- GPU side -------------------------------------------------------------------------------------------

    def _ensure_tensors(self, vocab: int, pool_rows: int, device) -> None:
        if self.cls is None:
            self.cls = torch.tensor(token_classes(self.tokenizer, vocab), dtype=torch.int8, device=device)
            if int(self.cls[self.nl_id]) != CLS_NL or int(self.cls[self.nn_id]) != CLS_WS:
                raise RuntimeError(f"{ENV}: newline tokens {self.nl_id} / {self.nn_id} classified as "
                                   f"{int(self.cls[self.nl_id])} / {int(self.cls[self.nn_id])}")
            self.banned = (self.cls == CLS_NL) | (self.cls == CLS_WS) | (self.cls == CLS_PAD)
            self.banned[[self.start_id, self.end_id]] = True
            self.state = torch.zeros(pool_rows, dtype=torch.int8, device=device)
        if self.cls.shape[0] != vocab or self.state.shape[0] != pool_rows:
            raise RuntimeError(f"{ENV}: vocab {vocab} / request pool {pool_rows} changed from "
                               f"{self.cls.shape[0]} / {self.state.shape[0]}")

    def _advance(self, st: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        c = self.cls[tok]
        thinking = torch.where(
            tok == self.end_id, ST_END,
            torch.where(c == CLS_NORMAL, ST_OK,
                        torch.where((c == CLS_NL) & (st == ST_OK), ST_NLOK, ST_WAIT)))
        after_end = torch.where(tok == self.nn_id, ST_NN, ST_DONE)
        return torch.where(st <= ST_NLOK, thinking, torch.where(st == ST_END, after_end, ST_DONE))

    def apply(self, batch: "ScheduleBatch", rows: list[int], candidates: torch.Tensor, logits: torch.Tensor,
              num_draft_tokens: int, pool_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Force / mask rows of the armed requests in place; returns (their batch rows, their req_pool_indices) for
        ``commit``."""
        bs = len(batch.reqs)
        k = num_draft_tokens
        vocab = logits.shape[-1]
        if logits.shape[0] != bs * k or candidates.numel() != bs * k:
            raise RuntimeError(f"{ENV}: verify logits {tuple(logits.shape)} / candidates {candidates.numel()} "
                               f"for {bs} requests x {k} rows")
        device = logits.device
        self._ensure_tensors(vocab, pool_rows, device)
        fresh, fresh_state = [], []
        for i in rows:
            req = batch.reqs[i]
            if req.arc_think_slot != req.kv.req_pool_idx:
                # armed just now, or back from a retract on another slot (its CPU view is complete then)
                req.arc_think_slot = req.kv.req_pool_idx
                fresh.append(req.kv.req_pool_idx)
                fresh_state.append(self._cpu_state(req))
        row_t = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        slots = batch.req_pool_indices.index_select(0, row_t).to(torch.int64)
        cand = candidates.view(bs, k).index_select(0, row_t).to(torch.int64)
        if fresh:
            fresh_t = torch.tensor(fresh, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
            val_t = torch.tensor(fresh_state, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
            # no close in the CPU view: start from the last committed token (the step in flight is not visible)
            pos = (slots.unsqueeze(0) == fresh_t.unsqueeze(1)).to(torch.int64).argmax(dim=1)
            c0 = cand[pos, 0]
            from_c0 = torch.where(c0 == self.end_id, ST_END,
                                  torch.where(self.cls[c0] == CLS_NORMAL, ST_OK, ST_WAIT))
            self.state.index_copy_(0, fresh_t, torch.where(val_t >= 0, val_t, from_c0).to(torch.int8))
        st = self.state[slots].to(torch.int64)
        states = [st]
        for j in range(1, k):
            st = self._advance(st, cand[:, j])
            states.append(st)
        states = torch.stack(states, dim=1)  # [n, k]: the state after candidates[:, :j+1]
        force = (states == ST_OK) | (states == ST_NLOK) | (states == ST_END)
        token = torch.where(states == ST_END, torch.full_like(cand, self.nn_id),
                            torch.where(states == ST_NLOK, torch.full_like(cand, self.end_id),
                                        torch.full_like(cand, self.nl_id)))
        view = logits.view(bs, k, vocab)
        sub = view.index_select(0, row_t)
        onehot = torch.full_like(sub, float("-inf"))
        onehot.scatter_(2, token.unsqueeze(2), 0.0)
        mask = (states == ST_NN).unsqueeze(2) & self.banned.view(1, 1, vocab)
        sub = torch.where(mask, torch.full_like(sub, float("-inf")), sub)
        view.index_copy_(0, row_t, torch.where(force.unsqueeze(2), onehot, sub))
        return row_t, slots

    def commit(self, handle: tuple[torch.Tensor, torch.Tensor], predict: torch.Tensor,
               accept_index: torch.Tensor) -> None:
        """state[slot] advanced over the tokens committed for that request this step (in order)."""
        row_t, slots = handle
        idx = accept_index.index_select(0, row_t).to(torch.int64)
        toks = predict.view(-1)[idx.clamp(min=0)].to(torch.int64)
        st = self.state[slots].to(torch.int64)
        for j in range(idx.shape[1]):
            st = torch.where(idx[:, j] >= 0, self._advance(st, toks[:, j]), st)
        self.state[slots] = st.to(torch.int8)


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
    logger.info("ARC think budget: %d tokens (<think> %d, </think> %d, newline %d, blank line %d)",
                budget, _STATE.start_id, _STATE.end_id, _STATE.nl_id, _STATE.nn_id)
    return _STATE


def active() -> Optional[ThinkBudget]:
    return _STATE
