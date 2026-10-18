"""arc-forge TITO session ledger (arc-forge docs/plans/tito_session_ledger.md).

Per request ``session_id`` the TokenizerManager keeps S, the exact token ids the
model has seen and sampled so far (raw ids: image/video placeholders are the real
pad token ids, not the scheduler's hash pad values). Every request is handled in
two steps:

intake (after tokenization, before the scheduler)
    ``replay_ids`` puts the session's sampled ids back wherever the client
    re-rendered the same text into a different tokenization (a non-canonical BPE
    split the model sampled), so the model always sees its own ids. The request
    is then an *extension* (its ids start with S) or a *branch* (they do not:
    compaction dropped the head, the parser rewrote a broken tool call, a client
    changed history, a retry of an earlier turn). With ``--session-replay-empty-think``
    a previous assistant turn whose thinking the model left empty in a non-canonical
    way (``<think>\n</think>\n\n``, or ``<think>\n`` straight into the answer) also
    counts as the same message when the client renders it canonically
    (``<think>\n\n</think>\n\n``): see ``EmptyThink``. With
    ``--session-replay-own-last-turn`` the model's own ids are authoritative for the
    session's last generated message even where the client's re-render has other text
    (a cut or malformed tool call, a thinking block the template closed), and stay
    so on every later request where the client sends the same ids for it (sticky
    entries, kept per session with S): see ``replay_own_turn``.

finish
    An extension appends ``tail`` (the ids after S) and ``gen`` (the sampled ids,
    stop token included) to S; a branch closes the session's segment and opens a
    new one. Aborted requests never change S.

With ``--session-ledger-dir`` every committed request is written as one
``append`` event (only the new ids), together with the raw bytes of new media,
optional sampled-token logprobs and optional routed experts. The replay itself is
a port of HiSparse h14 (``qsa_hisparse_session_cache.replay_request``) without
its record-state checks: the ledger's record is always raw. h14 stays untouched;
requests replayed here no longer diverge when they reach it.
"""

from __future__ import annotations

import atexit
import base64
import functools
import hashlib
import json
import logging
import os
import queue
import socket
import threading
import time
from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.managers.replay_own_turn import (
    OwnTurn,
    OwnTurnEntry,
    kept_entries,
    own_turn_replacement,
    sticky_replacement,
)

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1
REPLAY_END_TOKEN = "<|im_end|>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
EMPTY_THINK_TEXT = "<think>\n\n</think>\n\n"  # the chat template's assistant turn with empty reasoning
ASSISTANT_HEADER_TEXT = "<|im_start|>assistant\n"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VIDEO_PAD_TOKEN = "<|video_pad|>"
MM_UNSUPPORTED = ("vision_position_ids", "media_nums_per_sample", "visible_frame_counts", "token_type_ids")
MM_TOKEN_FIELDS = ("im_token_id", "im_start_id", "im_end_id", "slice_start_id", "slice_end_id", "video_token_id",
                   "audio_token_id", "audio_start_id", "audio_end_id")  # never inside a replayed message
EVENTS_ROTATE_BYTES = 1 << 30
ROUTING_ROTATE_BYTES = 4 << 30
QUEUE_MAX_EVENTS = 200_000
STATUS_EVERY_S = 300.0  # a status event this often: a killed server writes no "end"


def _i64(tokens) -> np.ndarray:
    """A read-only int64 view of a token sequence (array('q') without a copy)."""
    if isinstance(tokens, array):
        if tokens.typecode != "q":
            raise TypeError(f"token array typecode {tokens.typecode!r}, need 'q'")
        return np.frombuffer(tokens, dtype=np.int64) if len(tokens) else np.empty(0, np.int64)
    return np.asarray(tokens, dtype=np.int64)


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if diff.size else n


def _b64_i32(ids: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(ids, dtype="<i4").tobytes()).decode("ascii")


def _b64_f32(vals: Sequence[float]) -> str:
    return base64.b64encode(np.asarray(vals, dtype="<f4").tobytes()).decode("ascii")


# ----------------------------------------------------------------------------- replay


@dataclass
class ReplayResult:
    """What ``replay_ids`` did. ``spans`` non-empty means the request's ids changed."""

    spans: List[Dict[str, int]] = field(default_factory=list)
    skip: Optional[str] = None  # why it stopped before the record's end, None if it did not
    lcp0: int = 0  # first divergence before replay (len(record) or more: none)
    own_turn_skip: Optional[str] = None  # own-turn on and text_differs kept: why it did not apply
    # own-turn takeovers of this request (kept with S once it commits), and the starts of the session's
    # entries the client sent other ids for (dropped at intake)
    own_entries: List[OwnTurnEntry] = field(default_factory=list)
    own_dropped: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class EmptyThink:
    """How a previous assistant turn with empty thinking looks: the canonical rendering the client sends
    (``<|im_start|>assistant\n<think>\n\n</think>\n\n`` before the answer) and what the model may have
    sampled instead. ``replay_ids`` treats the two as the same message (``_empty_think_end``)."""

    header: Tuple[int, ...]  # <|im_start|>assistant\n
    canonical: Tuple[int, ...]  # <think>\n\n</think>\n\n
    think_start: int
    think_end: int
    is_space: Callable[[int], bool]  # the token decodes to whitespace only (and to something)

    @classmethod
    def from_tokenizer(cls, tokenizer, decode: Callable[[List[int]], str]) -> "EmptyThink":
        ids = {}
        for name in (THINK_START_TOKEN, THINK_END_TOKEN):
            t = tokenizer.convert_tokens_to_ids(name)
            if not isinstance(t, int) or tokenizer.convert_ids_to_tokens(t) != name:
                raise ValueError(f"--session-replay-empty-think: the tokenizer has no single {name} token (got {t!r})")
            ids[name] = int(t)
        start, end = ids[THINK_START_TOKEN], ids[THINK_END_TOKEN]

        @functools.lru_cache(maxsize=None)
        def is_space(t: int) -> bool:
            text = decode([int(t)])
            return bool(text) and not text.strip()

        header = tuple(int(t) for t in tokenizer.encode(ASSISTANT_HEADER_TEXT, add_special_tokens=False))
        canonical = tuple(int(t) for t in tokenizer.encode(EMPTY_THINK_TEXT, add_special_tokens=False))
        # <think>, whitespace, </think>, whitespace: the shape _empty_think_end relies on
        k = canonical.index(end) if end in canonical else -1
        if (len(canonical) < 3 or canonical[0] != start or canonical.count(start) != 1 or k < 2
                or canonical.count(end) != 1 or not all(is_space(t) for i, t in enumerate(canonical) if i not in (0, k))
                or decode(list(canonical)) != EMPTY_THINK_TEXT or decode(list(header)) != ASSISTANT_HEADER_TEXT):
            raise ValueError(f"--session-replay-empty-think: {EMPTY_THINK_TEXT!r} tokenizes to {canonical}, "
                             f"not <think> whitespace </think> whitespace")
        return cls(header=header, canonical=canonical, think_start=start, think_end=end, is_space=is_space)


def _empty_think_end(toks: np.ndarray, new: np.ndarray, lcp: int, start: int, rec_end: int, key_end: int,
                     et: EmptyThink, decode: Callable[[List[int]], str]) -> Optional[int]:
    """Where the record's empty-thinking block ends, if the request's message [start, key_end) is the
    record's [start, rec_end) with only that block rendered canonically; else None.

    The request diverges inside its canonical block ``<think>\n\n</think>\n\n``, which opens an
    assistant message (right after ``<|im_start|>assistant\n``, inside the message). The record has
    ``<think>`` there, then whitespace-only tokens, then either ``</think>`` and whitespace-only tokens,
    or directly something else -- and then no ``</think>`` in the rest of the message (the model never
    thought: reasoning that ends in a later ``</think>`` is not empty). After the two blocks the texts
    are the same. Anything else -- non-empty reasoning, another divergence -- is not equivalent."""
    C, H = et.canonical, et.header
    t = next((t for t in range(max(lcp - len(C) + 1, start + len(H)), lcp)
              if new[t] == et.think_start and tuple(new[t:t + len(C)].tolist()) == C
              and tuple(new[t - len(H):t].tolist()) == H), None)
    if t is None or key_end < t + len(C):
        return None
    # new[:lcp] == toks[:lcp] and t < lcp: the record has the header and <think> at t too
    i = t + 1
    while i < rec_end and et.is_space(int(toks[i])):
        i += 1
    if i < rec_end and toks[i] == et.think_end:
        i += 1
        while i < rec_end and et.is_space(int(toks[i])):
            i += 1
    elif (toks[i:rec_end] == et.think_end).any():
        return None  # text before a </think>: the model did think
    if lcp > i:
        return None  # the record's block ended before the divergence: it is in the answer
    rest = decode(toks[i:rec_end].tolist())
    if rest != decode(new[t + len(C):key_end].tolist()) or "\ufffd" in rest:
        return None
    return i


def _replay_blocker(tok, start: int) -> Optional[str]:
    """Inputs whose positions or side data a changed id count from `start` on would shift."""
    if tok.input_embeds is not None:
        return "embeds"
    if (
        tok.positional_embed_overrides is not None
        or tok.token_type_ids is not None
        or tok.multi_item_delimiter_indices is not None
    ):
        return "position_data"
    if tok.return_routed_experts and tok.routed_experts_start_len > start:
        return "routed_experts"
    return None


def _mm_view(mm, key: np.ndarray):
    """(item ranges [(item, a, b)] inclusive, special ids) of a processor output replay can move,
    or None when it has anything replay does not know how to keep consistent."""
    from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput

    if not isinstance(mm, MultimodalProcessorOutput) or mm.input_ids is None:
        return None
    if any(getattr(mm, f) is not None for f in MM_UNSUPPORTED):
        return None
    if len(mm.input_ids) != len(key) or not np.array_equal(_i64(mm.input_ids), key):
        return None  # the processor's ids must be the request's ids
    if mm.padded_input_ids is not None and len(mm.padded_input_ids) != len(key):
        return None
    pos = mm.mrope_positions
    if pos is not None and (not isinstance(pos, torch.Tensor) or tuple(pos.shape) != (3, len(key))):
        return None
    ranges = []
    for item in mm.mm_items:
        if item.offsets is None:
            return None
        for a, b in item.offsets:
            if not 0 <= a <= b < len(key):
                return None
            ranges.append((item, int(a), int(b)))
    ranges.sort(key=lambda r: r[1])
    if any(r[1] <= q[2] for q, r in zip(ranges, ranges[1:])):
        return None  # overlapping items
    special = np.array([v for f in MM_TOKEN_FIELDS if (v := getattr(mm, f)) is not None], dtype=np.int64)
    return ranges, special


def _text_positions(seg: torch.Tensor) -> bool:
    """M-RoPE positions of plain text: three equal rows counting up by one."""
    return bool((seg == seg[0:1]).all()) and bool(
        (seg[0] - seg[0, 0] == torch.arange(seg.shape[1], device=seg.device, dtype=seg.dtype)).all()
    )


def replay_ids(
    tok,
    record: np.ndarray,
    stop: np.ndarray,
    replay_end: int,
    decode: Callable[[List[int]], str],
    max_input_len: int,
    empty_think: Optional[EmptyThink] = None,
    own_turn: Optional[OwnTurn] = None,
    gen_start: Optional[int] = None,
    entries: Optional[Dict[int, OwnTurnEntry]] = None,
) -> ReplayResult:
    """Put the record's sampled ids back into a tokenized request (h14 replay, ported).

    Wherever the request's ids diverge from ``record`` before the record's end, the
    message around the divergence is compared as text: from the last message end
    before it (common to both), to the request's next ``replay_end`` and the record's
    next stop token (or the record's end). Same text (no U+FFFD): the request's ids for
    that message become the record's, and the search goes on after it. The first
    message whose text differs stops it. ``tok.input_ids`` -- and with multimodal
    inputs the processor's ``input_ids``/``padded_input_ids``, later image offsets and
    M-RoPE positions, and a ``logprob_start_len`` past a replaced message -- change in
    place. A replayed message may not touch an image.

    With ``empty_think``, a message whose text differs only because the record's leading
    empty thinking is non-canonical (``_empty_think_end``) is replayed the same way, as a
    span with ``"kind": "empty_think"``.

    With ``own_turn``, a message whose text differs is still replayed when it is the record's
    last generated message (``replay_own_turn.own_turn_replacement``; ``gen_start`` is where the
    session's last generation starts in ``record``): its ids become the record's message plus a
    closure, as a span with ``"kind": "own_turn"``, ``"closure"`` (closure tokens), ``"client_len"``
    and ``"client_sha256"`` (the client's ids for the message); ``own_entries`` gets its entry.
    A message starting at one of the session's ``entries`` (an earlier takeover; start ->
    entry) gets the record's ids again when the client sent the same ids for it
    (``replay_own_turn.sticky_replacement``: span kind own_turn, ``"sticky": true``, closure
    0); other ids drop the entry (``own_dropped``) and the replay goes on as without it.
    Otherwise the replay stops with text_differs as without the flag, and ``own_turn_skip``
    says why.
    """
    if own_turn is not None and (gen_start is None or entries is None):
        raise ValueError("own-turn replay needs gen_start (the start of the session's last generation) and the "
                         "session's own-turn entries")
    if own_turn is None and entries is not None:
        raise ValueError("own-turn entries without own-turn replay")
    if not isinstance(max_input_len, int) or max_input_len < 1:
        raise ValueError(f"max_input_len must be a positive int, got {max_input_len!r}")
    key = _i64(tok.input_ids)
    toks = record
    lcp = _lcp(key, toks)
    res = ReplayResult(lcp0=lcp)
    if lcp >= min(len(key), len(toks)):
        return res  # the key agrees through the record (or is a prefix of it)
    if np.isin(toks[lcp], stop):
        if lcp < len(toks) - 1:  # a message boundary moved: different text, not a split
            res.skip = "stop"
        return res  # else only the previous generation's stop token differs
    first = np.flatnonzero(np.isin(toks[:lcp], stop))
    blocker = _replay_blocker(tok, int(first[-1]) + 1 if first.size else 0)
    if blocker is not None:
        res.skip = blocker
        return res
    mm = tok.mm_inputs
    view = None
    if tok.need_wait_for_mm_inputs:  # encoder disaggregation: the images come later
        res.skip = "mm_type"
        return res
    if mm is not None:
        view = _mm_view(mm, key)
        if view is None:
            res.skip = "mm_type"
            return res
    ranges, special = view if view is not None else ([], None)
    shifts = {id(item): [list(map(int, ab)) for ab in item.offsets] for item, _, _ in ranges}
    padded = _i64(mm.padded_input_ids).copy() if view is not None and mm.padded_input_ids is not None else None
    pos = mm.mrope_positions if view is not None else None
    logprob = tok.logprob_start_len if tok.return_logprob else None
    new, spans, taken = key, [], []
    while lcp < min(len(new), len(toks)) and not np.isin(toks[lcp], stop):
        ends = np.flatnonzero(np.isin(toks[:lcp], stop))
        start = int(ends[-1]) + 1 if ends.size else 0
        after = np.flatnonzero(np.isin(toks[lcp:], stop))
        rec_end = lcp + int(after[0]) if after.size else len(toks)
        hits = np.flatnonzero(new[lcp:] == replay_end)
        if not hits.size:
            res.skip = "no_end"
            break
        key_end = lcp + int(hits[0])
        if logprob is not None and logprob >= 0 and start < logprob < key_end:
            res.skip = "logprob_start"
            break
        if view is not None:
            cur = [(a, b) for offs in shifts.values() for a, b in offs]
            if any(a < key_end and b >= start for a, b in cur) or np.isin(toks[start:rec_end], special).any():
                res.skip = "mm_overlap"
                break
        text = decode(toks[start:rec_end].tolist())
        kind, extra, rep = None, None, toks[start:rec_end]
        if text != decode(new[start:key_end].tolist()) or "�" in text:
            if empty_think is not None and _empty_think_end(toks, new, lcp, start, rec_end, key_end, empty_think,
                                                            decode) is not None:
                kind = "empty_think"
            else:
                entry = None if entries is None else entries.get(start)
                sticky = None if entry is None else sticky_replacement(entry, toks, new, start, key_end, rec_end,
                                                                        gen_start)
                if sticky is not None and sticky.drop:
                    res.own_dropped.append(start)  # the client sent other ids for the message
                if sticky is not None and sticky.ids is not None:
                    rep, extra = sticky.ids, sticky.fields
                else:
                    # the ledger's record is input + output through the stop: the last generation ends with
                    # it, and only a stop / length generation is ever committed
                    own = None if own_turn is None else own_turn_replacement(
                        own_turn, toks, new, lcp, start, key_end, gen_start, len(toks), stop, finished=True)
                    if own is None or own.ids is None:
                        res.own_turn_skip = None if own is None else own.why
                        res.skip = "text_differs"
                        break
                    if own.rec_end != rec_end:
                        raise RuntimeError(f"{tok.rid}: own-turn message end {own.rec_end}, replay's {rec_end}")
                    rep, extra = own.ids, own.fields
                    if view is not None and np.isin(rep[rec_end - start:], special).any():
                        res.skip = "mm_overlap"  # the closure holds a multimodal token
                        break
                    taken.append(own.entry)
        delta = len(rep) - (key_end - start)
        if pos is not None and delta:
            seg = pos[:, start:key_end]
            if not _text_positions(seg):
                res.skip = "mm_shift"
                spans = []
                break
            grown = seg[:, :1] + torch.arange(len(rep), dtype=pos.dtype, device=pos.device)
            pos = torch.cat((pos[:, :start], grown, pos[:, key_end:] + delta), dim=1)
        if view is not None:
            for offs in shifts.values():
                for ab in offs:
                    if ab[0] >= key_end:
                        ab[0] += delta
                        ab[1] += delta
            if padded is not None:
                if not np.array_equal(padded[start:key_end], new[start:key_end]):
                    raise RuntimeError(f"{tok.rid}: padded ids differ from the ids in a text message")
                padded = np.concatenate((padded[:start], rep, padded[key_end:]))
        new = np.concatenate((new[:start], rep, new[key_end:]))
        if logprob is not None and logprob >= key_end:
            logprob += delta
        spans.append({"s": start, "lcp": lcp, "rec_end": rec_end, "key_end": key_end, "delta": delta,
                      **({"kind": kind} if kind is not None else {}), **(extra or {})})
        nxt = _lcp(new, toks)
        if nxt < rec_end:  # own-turn after a cut: rec_end is the record's end
            raise RuntimeError(f"{tok.rid}: replayed message [{start}, {rec_end}) still diverges at {nxt}")
        lcp = nxt
        if lcp < min(len(new), len(toks)) and lcp < len(toks) - 1 and np.isin(toks[lcp], stop):
            res.skip = "stop"
    if not spans:
        return res
    if len(new) >= max_input_len:
        res.skip = "input_len"
        return res
    if view is not None:
        if pos is not None and mm.mrope_position_delta is not None:
            old_delta = int(mm.mrope_positions.max()) + 1 - len(key)
            if int(pos.max()) + 1 - len(new) != old_delta:
                raise RuntimeError(f"{tok.rid}: replay changed the M-RoPE position delta")
        for item, _, _ in ranges:
            item.offsets = [tuple(ab) for ab in shifts[id(item)]]
        mm.input_ids = new.tolist()
        if padded is not None:
            mm.padded_input_ids = padded.tolist()
        if pos is not None:
            mm.mrope_positions = pos
    tok.input_ids = array("q", new.tolist())
    if logprob is not None:
        tok.logprob_start_len = logprob
    res.spans = spans
    for e in taken:
        e.check_held_by(new, f"{tok.rid}'s ids after replay")
    res.own_entries = taken
    return res


# ----------------------------------------------------------------------------- media


def _media_bytes(item: Any) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """(raw bytes, mime, source) of one image_data / video_data entry.

    Data URLs and raw bytes are recorded; anything else (an http URL, a path, a
    preprocessed dict) is recorded by reference only and counted as unmapped."""
    from sglang.srt.utils.common import ImageData, VideoData

    if isinstance(item, (ImageData, VideoData)):
        item = item.url
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), None, None
    if isinstance(item, dict):
        url = item.get("url")
        if isinstance(url, str):
            item = url
    if isinstance(item, str) and item.startswith("data:"):
        head, sep, payload = item.partition(",")
        if not sep:
            raise ValueError("data URL without a comma")
        mime = head[5:].split(";", 1)[0] or None
        if ";base64" not in head:
            raise ValueError(f"data URL is not base64 ({head[:40]!r})")
        return base64.b64decode(payload), mime, None
    return None, None, (item if isinstance(item, str) else type(item).__name__)


def _runs(ids: np.ndarray, pad: Optional[int]) -> List[Tuple[int, int]]:
    """Inclusive [a, b] runs of `pad` in ids."""
    if pad is None:
        return []
    hit = ids == pad
    if not hit.any():
        return []
    d = np.diff(np.concatenate(([0], hit.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _grid_rows(mm, attr: str) -> Optional[List[List[int]]]:
    if mm is None:
        return None
    rows: List[List[int]] = []
    for item in mm.mm_items:
        g = getattr(item, attr, None)
        if g is None:
            continue
        g = g.tolist() if isinstance(g, torch.Tensor) else g
        rows.extend([list(map(int, r)) for r in (g if g and isinstance(g[0], (list, tuple)) else [g])])
    return rows or None


# ----------------------------------------------------------------------------- writer


class _Writer:
    """Background writer: events (one JSON line each, flushed), media blobs, routing sidecars."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "media"), exist_ok=True)
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=QUEUE_MAX_EVENTS)
        self.ev_idx = 0
        self.ev_f = open(os.path.join(root, f"events.{self.ev_idx}.jsonl"), "ab")
        self.rt_idx = 0
        self.rt_f = open(os.path.join(root, f"routing.{self.rt_idx}.bin"), "ab")
        self.media_written: set = set(n for n in os.listdir(os.path.join(root, "media")))
        self.bytes_written = 0
        # event lines queued (dropped puts excluded; the writer's own lossy markers included) and event lines
        # flushed to events.*.jsonl after their media and routing: both monotonic, written <= enqueued, equal once
        # the writer has drained
        self.enqueued_events = 0
        self.written_events = 0
        self.lossy = 0
        self.lossy_unreported = 0
        self.error: Optional[str] = None
        # shutdown waits for room instead of dropping: its seg_close/end events are what tells a clean stop
        # from a crash
        self.blocking = False
        self.thread = threading.Thread(target=self._run, name="session-ledger-writer", daemon=True)
        self.thread.start()

    def _enqueue(self, job: tuple, block: bool) -> None:
        # counted before the put, so that written_events never runs ahead of enqueued_events
        self.enqueued_events += 1
        try:
            if block:
                self.q.put(job)
            else:
                self.q.put_nowait(job)
        except BaseException:
            self.enqueued_events -= 1
            raise

    def put(self, event: dict, media: Sequence[Tuple[str, bytes]] = (), routing: Optional[bytes] = None) -> None:
        if self.error is not None:
            raise RuntimeError(f"session ledger writer died: {self.error}")
        if self.blocking:
            if self.lossy_unreported:
                self._enqueue(({"ev": "lossy", "t": time.time(), "dropped": self.lossy_unreported}, (), None), True)
                self.lossy_unreported = 0
            self._enqueue((event, media, routing), True)
            return
        try:
            if self.lossy_unreported:
                # the drop itself goes on disk as soon as there is room again
                self._enqueue(({"ev": "lossy", "t": time.time(), "dropped": self.lossy_unreported}, (), None), False)
                self.lossy_unreported = 0
            self._enqueue((event, media, routing), False)
        except queue.Full:
            self.lossy += 1
            self.lossy_unreported += 1
            logger.error("SESSION_LEDGER queue full (%d events): dropped one, ledger is lossy", self.q.qsize())

    def depth(self) -> int:
        return self.q.qsize()

    def close(self, timeout: float = 60.0) -> None:
        self.q.put(None)
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError(f"session ledger writer did not drain in {timeout}s ({self.q.qsize()} left)")
        if self.error is not None:
            raise RuntimeError(f"session ledger writer died: {self.error}")

    def _run(self) -> None:
        try:
            while True:
                job = self.q.get()
                if job is None:
                    break
                self._write(*job)
        except BaseException as exc:  # surfaced on the next put() and in status()
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("session ledger writer died")
        finally:
            self.ev_f.close()
            self.rt_f.close()

    def _write(self, event: dict, media, routing: Optional[bytes]) -> None:
        for sha, data in media:
            if sha in self.media_written:
                continue
            tmp = os.path.join(self.root, "media", f".{sha}.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(self.root, "media", sha))
            self.media_written.add(sha)
            self.bytes_written += len(data)
        if routing is not None:
            if self.rt_f.tell() + len(routing) > ROUTING_ROTATE_BYTES and self.rt_f.tell() > 0:
                self.rt_f.close()
                self.rt_idx += 1
                self.rt_f = open(os.path.join(self.root, f"routing.{self.rt_idx}.bin"), "ab")
            event["routing"]["file"] = f"routing.{self.rt_idx}.bin"
            event["routing"]["offset"] = self.rt_f.tell()
            self.rt_f.write(routing)
            self.rt_f.flush()
            self.bytes_written += len(routing)
        line = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        if self.ev_f.tell() + len(line) > EVENTS_ROTATE_BYTES and self.ev_f.tell() > 0:
            self.ev_f.close()
            self.ev_idx += 1
            self.ev_f = open(os.path.join(self.root, f"events.{self.ev_idx}.jsonl"), "ab")
        self.ev_f.write(line)
        self.ev_f.flush()
        self.bytes_written += len(line)
        self.written_events += 1


# ----------------------------------------------------------------------------- ledger


@dataclass
class _Session:
    sid: str
    seg: int
    S: np.ndarray  # int32
    version: int = 0
    k: int = 0  # appends in the current segment
    last_used: float = 0.0
    media: List[dict] = field(default_factory=list)  # committed media entries, S coordinates
    gen_start: int = 0  # where the last committed generation starts in S
    # --session-replay-own-last-turn: the messages own-turn took over, held by S (start -> entry)
    own: Dict[int, OwnTurnEntry] = field(default_factory=dict)


@dataclass
class _Pending:
    rid: str
    sid: Optional[str]
    key: np.ndarray  # int32, after replay
    kind: str  # "extend" | "branch" | "new" | "single"
    cause: Optional[str]
    base_version: int
    prefix_len: int
    stop_substituted: bool
    replay: ReplayResult
    media: List[dict]
    media_blobs: List[Tuple[str, bytes]]
    sampling: dict
    t0: float
    routing_start: int
    # an extension's media before prefix_len (committed earlier), in case it finishes as a branch
    earlier_media: List[dict] = field(default_factory=list)


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty",
                   "repetition_penalty", "max_new_tokens", "min_new_tokens", "stop_token_ids", "ignore_eos",
                   "skip_special_tokens", "sampling_seed", "n")


def _sampling_dict(sp) -> dict:
    """The effective sampling parameters (after preferred params and normalize()); SamplingParams is a
    msgspec Struct, its fields are __struct_fields__."""
    fields = set(type(sp).__struct_fields__)
    out = {}
    for name in SAMPLING_FIELDS:
        if name not in fields:
            continue
        v = getattr(sp, name)
        if isinstance(v, set):
            v = sorted(v)
        if v is None or isinstance(v, (bool, int, float, str, list)):
            out[name] = v
    return out


class SessionLedger:
    """See the module docstring. One per TokenizerManager; not thread-safe (the TM's event loop)."""

    def __init__(
        self,
        *,
        tokenizer,
        eos_token_ids: Sequence[int],
        ledger_dir: Optional[str],
        record_all: bool,
        logprobs: bool,
        routing: bool,
        routing_server_enabled: bool,
        max_bytes: int,
        tokenizer_worker_num: int,
        server_meta: Dict[str, Any],
        port: int,
        empty_think: bool = False,
        exact_resume: bool = False,
        own_last_turn: bool = False,
    ):
        if tokenizer is None:
            raise ValueError("--session-replay needs the tokenizer (do not use --skip-tokenizer-init)")
        if tokenizer_worker_num != 1:
            raise ValueError(f"--session-replay needs --tokenizer-worker-num 1 (got {tokenizer_worker_num}): "
                             "a session's requests must all reach the same ledger")
        if ledger_dir is None and (record_all or logprobs or routing):
            raise ValueError("--session-ledger-all/-logprobs/-routing need --session-ledger-dir")
        if routing and not routing_server_enabled:
            raise ValueError("--session-ledger-routing needs --enable-return-routed-experts")
        if not isinstance(max_bytes, int) or max_bytes < 1 << 20:
            raise ValueError(f"--session-ledger-max-bytes must be >= 1 MiB, got {max_bytes!r}")
        end = tokenizer.convert_tokens_to_ids(REPLAY_END_TOKEN)
        if not isinstance(end, int) or tokenizer.convert_ids_to_tokens(end) != REPLAY_END_TOKEN:
            raise ValueError(f"the tokenizer has no single {REPLAY_END_TOKEN} token (got {end!r})")
        eos = [int(t) for t in (eos_token_ids or [])]
        if not eos:
            raise ValueError(f"the session ledger needs the model's eos token ids, got {eos_token_ids!r}")
        self.replay_end = int(end)
        self.eos = np.array(sorted(set(eos)), dtype=np.int64)
        self.stop = np.array(sorted({self.replay_end, *eos}), dtype=np.int64)
        self.decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.empty_think = EmptyThink.from_tokenizer(tokenizer, self.decode) if empty_think else None
        self.own_turn = (OwnTurn.from_tokenizer(tokenizer, self.decode, "--session-replay-own-last-turn")
                         if own_last_turn else None)
        pads = {}
        for name in (IMAGE_PAD_TOKEN, VIDEO_PAD_TOKEN):
            t = tokenizer.convert_tokens_to_ids(name)
            pads[name] = int(t) if isinstance(t, int) and tokenizer.convert_ids_to_tokens(t) == name else None
        self.image_pad, self.video_pad = pads[IMAGE_PAD_TOKEN], pads[VIDEO_PAD_TOKEN]
        self.record_all = record_all
        # --session-exact-resume: appends and attempts carry "resume" (meta_info["session_resume"]);
        # off, the key is absent (readers go by server.json's server_args.session_exact_resume)
        self.exact_resume = exact_resume
        self.logprobs = logprobs
        self.routing = routing
        self.max_bytes = max_bytes
        self.sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self.pending: Dict[str, _Pending] = {}
        self.bytes = 0
        self.next_seg = 0
        self.version_seq = 0  # ledger-wide: a session's version never repeats, across segments or reopens
        self.counters: Counter = Counter()
        self._sweep_tick = 0
        self._next_status = time.time() + STATUS_EVERY_S
        # set by on_intake for the TokenizerManager: did the ledger turn these on for this request
        self.forced_logprobs = False
        self.forced_routing = False
        self._shut = False
        self.instance = None
        self.root = None
        self.writer: Optional[_Writer] = None
        if ledger_dir is not None:
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            self.instance = f"{socket.gethostname()}-p{port}-{stamp}-{os.getpid()}"
            self.root = os.path.join(ledger_dir, self.instance)
            os.makedirs(self.root, exist_ok=False)
            meta = dict(server_meta)
            meta.update(instance=self.instance, v=LEDGER_VERSION, t=time.time(), host=socket.gethostname(),
                        pid=os.getpid(), port=port, replay_end=self.replay_end, eos=self.eos.tolist(),
                        image_pad=self.image_pad, video_pad=self.video_pad,
                        ledger={"all": record_all, "logprobs": logprobs, "routing": routing, "max_bytes": max_bytes,
                                "empty_think": empty_think, "own_last_turn": own_last_turn})
            with open(os.path.join(self.root, "server.json"), "w") as f:
                json.dump(meta, f, indent=1, sort_keys=True, default=str)
            self.writer = _Writer(self.root)
            self.writer.put({"ev": "start", "v": LEDGER_VERSION, "instance": self.instance, "t": time.time()})
            atexit.register(self.shutdown)
        logger.info("SESSION_LEDGER on: replay=1 empty_think=%d own_last_turn=%d dir=%s all=%d logprobs=%d "
                    "routing=%d", empty_think, own_last_turn, self.root, record_all, logprobs, routing)

    # ------------------------------------------------------------------ intake

    def on_intake(self, obj, tok, max_input_len: int) -> None:
        """After tokenization, before the request goes to the scheduler."""
        self.forced_logprobs = self.forced_routing = False
        sid = obj.session_id
        if sid is None and not (self.writer is not None and self.record_all):
            return
        if tok.input_embeds is not None:
            self.counters["unrecorded_embeds"] += 1
            return
        sess = self.sessions.get(sid) if sid is not None else None
        res = ReplayResult(lcp0=0)
        if sess is not None:
            self.sessions.move_to_end(sid)
            res = replay_ids(tok, sess.S.astype(np.int64), self.stop, self.replay_end, self.decode, max_input_len,
                             self.empty_think, own_turn=self.own_turn, gen_start=sess.gen_start,
                             entries=sess.own if self.own_turn is not None else None)
            for start in res.own_dropped:  # the client sent other ids for a message own-turn had taken over
                del sess.own[start]
                self.counters["replay_own_turn_sticky_drop_client"] += 1
            if res.spans:
                self.counters["replay"] += 1
                self.counters["replay_messages"] += len(res.spans)
                think = sum(s.get("kind") == "empty_think" for s in res.spans)
                if think:
                    self.counters["replay_empty_think"] += think
                own = [s for s in res.spans if s.get("kind") == "own_turn" and not s.get("sticky")]
                if own:
                    self.counters["replay_own_turn"] += len(own)
                    self.counters["replay_own_turn_closure"] += sum(s["closure"] for s in own)
                sticky = sum(bool(s.get("sticky")) for s in res.spans)
                if sticky:
                    self.counters["replay_own_turn_sticky"] += sticky
            if res.skip is not None:
                self.counters[f"replay_skip_{res.skip}"] += 1
            if res.own_turn_skip is not None:
                self.counters[f"replay_own_turn_skip_{res.own_turn_skip}"] += 1
        key64 = _i64(tok.input_ids)
        kind, cause, prefix_len, substituted = "single", None, 0, False
        if sid is not None:
            kind, cause = "new", "new"
            if sess is not None:
                n = len(sess.S)
                lcp = _lcp(key64, sess.S.astype(np.int64))
                if lcp == n and len(key64) >= n:
                    kind, prefix_len = "extend", n
                elif (lcp == n - 1 and len(key64) >= n and int(sess.S[-1]) in self.eos
                      and int(sess.S[-1]) != self.replay_end and int(key64[n - 1]) == self.replay_end):
                    kind, prefix_len, substituted = "extend", n - 1, True
                elif sess.gen_start <= lcp < n:
                    # the request keeps the last request's input but not all of its generation: the
                    # client re-rendered the generation's tail (a turn cut by max_tokens gets "</think>"
                    # closed by the template), or resent the last request (codex drops a compaction
                    # reply that holds tool calls and asks again). Rewind S to the divergence; the
                    # replaced ids stay in their append event.
                    kind, prefix_len = "extend", lcp
                else:
                    kind = "branch"
                    cause = ("retry" if lcp == len(key64) else res.skip or "text_differs")
        if self.writer is None:
            # replay only: keep S current, nothing to write
            self.pending[obj.rid] = _Pending(obj.rid, sid, key64.astype(np.int32), kind, cause,
                                             sess.version if sess else 0, prefix_len, substituted, res,
                                             [], [], {}, time.time(), prefix_len)
            return
        media, blobs = self._media(obj, tok, key64, prefix_len)
        if self.logprobs and not tok.return_logprob:
            tok.return_logprob = True
            tok.logprob_start_len = -1  # the scheduler resolves it to len(input): output positions only
            self.forced_logprobs = True
        # the previous generation's last token was never forwarded: its routing is computed now
        routing_start = max(prefix_len - 1, 0)
        if self.routing:
            if tok.return_routed_experts:  # the client asked for its own range: record that one
                routing_start = int(tok.routed_experts_start_len)
            else:
                tok.return_routed_experts = True
                tok.routed_experts_start_len = routing_start
                self.forced_routing = True
        earlier = [m for m in sess.media if m["ranges"][-1][1] < prefix_len] if kind == "extend" else []
        self.pending[obj.rid] = _Pending(obj.rid, sid, key64.astype(np.int32), kind, cause,
                                         sess.version if sess else 0, prefix_len, substituted, res, media, blobs,
                                         _sampling_dict(tok.sampling_params), time.time(), routing_start,
                                         earlier_media=earlier)

    def _media(self, obj, tok, key: np.ndarray, prefix_len: int):
        """Media entries whose pad runs start at or after prefix_len (earlier ones are already recorded)."""
        out, blobs = [], []
        for modality, pad, data, grid_attr in (
            ("image", self.image_pad, obj.image_data, "image_grid_thw"),
            ("video", self.video_pad, obj.video_data, "video_grid_thw"),
        ):
            runs = _runs(key, pad)
            items = [] if data is None else (list(data) if isinstance(data, list) else [data])
            if not runs and not items:
                continue
            grid = _grid_rows(tok.mm_inputs, grid_attr)
            if modality == "image":
                groups = [[r] for r in runs]
            else:
                per = [int(g[0]) for g in grid] if grid is not None else None
                if per is None or sum(per) != len(runs):
                    groups = None
                else:
                    groups, i = [], 0
                    for t in per:
                        groups.append(runs[i:i + t])
                        i += t
            if groups is None or len(groups) != len(items) or (grid is not None and len(grid) != len(items)):
                self.counters[f"media_unmapped_{modality}"] += 1
                out.append({"modality": modality, "unmapped": True, "n_items": len(items),
                            "ranges": [list(r) for r in runs if r[0] >= prefix_len]})
                continue
            for j, (g, item) in enumerate(zip(groups, items)):
                if g[0][0] < prefix_len:
                    continue
                raw, mime, src = _media_bytes(item)
                entry = {"modality": modality, "ranges": [list(r) for r in g],
                         "grid_thw": grid[j] if grid is not None else None}
                if raw is None:
                    self.counters[f"media_by_reference_{modality}"] += 1
                    entry.update(sha256=None, src=src)
                else:
                    sha = hashlib.sha256(raw).hexdigest()
                    entry.update(sha256=sha, mime=mime, nbytes=len(raw))
                    blobs.append((sha, raw))
                out.append(entry)
        return out, blobs

    # ------------------------------------------------------------------ finish

    def on_finish(self, rid: str, state, meta_info: dict, finish_reason: Optional[dict]) -> None:
        """Once per request, when the TokenizerManager marks it finished."""
        p = self.pending.pop(rid, None)
        if p is None:
            return
        if self._shut:
            self.counters["finished_after_shutdown"] += 1  # the writer is closed: nothing can be recorded
            return
        gen = np.asarray(state.output_ids, dtype=np.int32)
        ftype = (finish_reason or {}).get("type")
        if ftype not in ("stop", "length"):
            self.counters[f"attempt_{ftype}"] += 1
            if self.writer is not None:
                self.writer.put({"ev": "attempt", "sid": p.sid, "rid": rid, "t": time.time(), "kind": ftype,
                                 "finish": finish_reason, "input_len": int(len(p.key)), "gen_len": int(len(gen)),
                                 # intake's prefix_len against S (as an append's)
                                 "prefix_len": int(p.prefix_len), "usage": _usage(meta_info),
                                 # a scheduler output: with the flag it always has session_resume
                                 **({"resume": meta_info["session_resume"]} if self.exact_resume else {})})
            return
        sess = self.sessions.get(p.sid) if p.sid is not None else None
        kind, prefix_len, cause = p.kind, p.prefix_len, p.cause
        if p.sid is not None and (sess.version if sess is not None else 0) != p.base_version:
            # the session changed since intake (versions are ledger-wide unique)
            p.stop_substituted = False
            if sess is None:
                kind, cause = "new", "evicted"  # evicted or closed
            else:
                # another request committed first: the intake rules again, against the new S
                n = len(sess.S)
                lcp = _lcp(p.key.astype(np.int64), sess.S.astype(np.int64))
                if lcp == n and len(p.key) >= n:
                    kind, prefix_len = "extend", n
                elif sess.gen_start <= lcp < n:
                    kind, prefix_len = "extend", lcp  # rewinds into the generation committed meanwhile
                else:
                    kind, cause = "branch", "concurrent"
        new_S = np.concatenate((p.key, gen))
        if kind == "extend":
            # entries before prefix_len were committed by this or a concurrent request
            media = [m for m in p.media if not m["ranges"] or m["ranges"][0][0] >= prefix_len]
            seg, k = sess.seg, sess.k
            rewind = len(sess.S) - prefix_len
            if rewind:
                self.counters["rewind"] += 1
            if self.own_turn is not None:
                # S keeps [0, prefix_len): an entry survives if it ends (its <|im_end|>) before that
                kept, dropped = kept_entries(sess.own, new_S, prefix_len, f"{rid}'s committed S")
                if dropped:
                    self.counters["replay_own_turn_sticky_drop_rewind"] += dropped
                for e in p.replay.own_entries:
                    e.check_held_by(new_S, f"{rid}'s committed S")
                    kept[e.start] = e
                if p.replay.own_entries:
                    self.counters["replay_own_turn_entries"] += len(p.replay.own_entries)
                sess.own = kept
            self._set_S(sess, new_S)
            sess.gen_start = len(p.key)
            sess.media.extend(m for m in media if "sha256" in m)
            self.version_seq += 1
            sess.version = self.version_seq
            sess.k += 1
        else:
            rewind = 0
            # an extension at intake that became a branch: its media before p.prefix_len were committed
            # earlier in the session (same ids, same coordinates), as the intake saw them
            media = p.earlier_media + p.media
            if sess is not None:
                self._drop_entries(sess, "split")
                self._close(sess, f"split:{cause}")
                del self.sessions[p.sid]
                self.bytes -= sess.S.nbytes
            if p.replay.own_entries:  # the new segment starts without entries
                self.counters["replay_own_turn_sticky_drop_split"] += len(p.replay.own_entries)
            seg, k = self._open(p.sid, cause if p.sid is not None else "single"), 0
            prefix_len = 0
            if p.sid is not None:
                self.version_seq += 1
                sess = _Session(p.sid, seg, np.empty(0, np.int32), version=self.version_seq, k=1,
                                last_used=time.time(), gen_start=len(p.key))
                self.sessions[p.sid] = sess
                self._set_S(sess, new_S)
                sess.media = [m for m in media if "sha256" in m]
        self.counters[f"commit_{kind}"] += 1
        if p.stop_substituted and kind == "extend":
            self.counters["stop_substituted"] += 1
        if sess is not None:
            sess.last_used = time.time()
        if self.writer is not None:
            self._write_append(p, state, meta_info, finish_reason, gen, seg, k, prefix_len, kind, media, rewind)
            now = time.time()
            if now >= self._next_status:
                self._next_status = now + STATUS_EVERY_S
                self.writer.put({"ev": "status", "t": now, "status": self.status()})
            if p.sid is None:
                self.writer.put({"ev": "seg_close", "sid": None, "seg": seg, "t": time.time(), "reason": "single",
                                 "n_tokens": int(len(new_S)), "sha256": hashlib.sha256(new_S.astype("<i4").tobytes()).hexdigest()})
        self._evict()

    def _write_append(self, p: _Pending, state, meta_info: dict, finish_reason: dict, gen: np.ndarray, seg: int,
                      k: int, prefix_len: int, kind: str, media: List[dict], rewind: int) -> None:
        tail = p.key[prefix_len:]
        ev = {"ev": "append", "sid": p.sid, "seg": seg, "k": k, "rid": p.rid, "t0": p.t0, "t1": time.time(),
              "prefix_len": int(prefix_len), "tail": _b64_i32(tail), "gen": _b64_i32(gen),
              "finish": finish_reason, "stop_substituted": bool(p.stop_substituted and kind == "extend"),
              "rewind": int(rewind),
              "sampling": p.sampling,
              "usage": _usage(meta_info),
              # how the scheduler resumed this prefill (--session-exact-resume only, session_park.py)
              **({"resume": meta_info["session_resume"]} if self.exact_resume else {}),
              "spec": {k2: meta_info[k2] for k2 in ("spec_verify_ct", "spec_accept_length", "spec_accept_rate")
                       if k2 in meta_info} or None,
              "replay": {"spans": p.replay.spans, "skip": p.replay.skip, "lcp0": p.replay.lcp0},
              "media": media, "logprobs": None, "routing": None}
        ev["weight_versions"], why = _weight_versions(meta_info, len(gen))
        if why is not None:
            self.counters["weight_versions_error"] += 1
            ev["weight_versions_error"] = why
        if self.logprobs:
            vals = [float(v) for v in state.output_token_logprobs_val]
            if len(vals) != len(gen):
                self.counters["logprobs_len_mismatch"] += 1
                ev["logprobs_error"] = f"{len(vals)} logprobs for {len(gen)} ids"
            else:
                ev["logprobs"] = _b64_f32(vals)
        routing_bytes = None
        if self.routing:
            server_error = meta_info.get("routed_experts_error")
            if server_error is not None:
                # the scheduler refused the rows (unwritten sentinel / all-zero)
                self.counters["routing_invalid"] += 1
                ev["routing_error"] = server_error
            else:
                routing_bytes, info = _routing_bytes(meta_info, p.routing_start,
                                                     len(p.key) + len(gen) - 1 - p.routing_start)
                if routing_bytes is None:
                    self.counters["routing_missing"] += 1
                    ev["routing_error"] = info
                else:
                    ev["routing"] = info
        self.writer.put(ev, p.media_blobs, routing_bytes)

    # ------------------------------------------------------------------ sessions

    def _open(self, sid: Optional[str], cause: str) -> int:
        seg = self.next_seg
        self.next_seg += 1
        self.counters[f"seg_open_{cause}"] += 1
        if self.writer is not None:
            self.writer.put({"ev": "seg_open", "sid": sid, "seg": seg, "t": time.time(), "cause": cause})
        return seg

    def _close(self, sess: _Session, reason: str) -> None:
        self.counters[f"seg_close_{reason}"] += 1
        if self.writer is not None:
            self.writer.put({"ev": "seg_close", "sid": sess.sid, "seg": sess.seg, "t": time.time(), "reason": reason,
                             "n_tokens": int(len(sess.S)),
                             "sha256": hashlib.sha256(sess.S.astype("<i4").tobytes()).hexdigest()})

    def _drop_entries(self, sess: _Session, why: str) -> None:
        if sess.own:
            self.counters[f"replay_own_turn_sticky_drop_{why}"] += len(sess.own)
            sess.own = {}

    def _set_S(self, sess: _Session, S: np.ndarray) -> None:
        self.bytes += S.nbytes - sess.S.nbytes
        sess.S = S

    def _evict(self) -> None:
        while self.bytes > self.max_bytes and self.sessions:
            sid, sess = self.sessions.popitem(last=False)
            self.bytes -= sess.S.nbytes
            self._drop_entries(sess, "evict")
            self._close(sess, "evicted")

    def close_session(self, session_id: str) -> None:
        sess = self.sessions.pop(session_id, None)
        if sess is None:
            self.counters["close_unknown"] += 1
            return
        self.bytes -= sess.S.nbytes
        self._drop_entries(sess, "close")
        self._close(sess, "close_session")

    def sweep(self, live_rids) -> None:
        """Drop pending entries whose request left the TokenizerManager without a finish
        (rejected before the scheduler, aborted with 503/500). Cheap: every 256 intakes."""
        self._sweep_tick += 1
        if self._sweep_tick % 256:
            return
        for rid in [r for r in self.pending if r not in live_rids]:
            p = self.pending.pop(rid)
            self.counters["pending_orphaned"] += 1
            if self.writer is not None:
                self.writer.put({"ev": "attempt", "sid": p.sid, "rid": rid, "t": time.time(), "kind": "orphaned",
                                 "finish": None, "input_len": int(len(p.key)), "gen_len": 0,
                                 "prefix_len": int(p.prefix_len),
                                 # no meta_info: it left without one (resume null with the flag, else absent)
                                 "usage": None, **({"resume": None} if self.exact_resume else {})})

    def status(self) -> dict:
        return {"instance": self.instance, "dir": self.root, "v": LEDGER_VERSION, "sessions": len(self.sessions),
                "pending": len(self.pending), "bytes": self.bytes, "max_bytes": self.max_bytes,
                "queue": self.writer.depth() if self.writer else 0,
                "written_bytes": self.writer.bytes_written if self.writer else 0,
                "enqueued_events": self.writer.enqueued_events if self.writer else 0,
                "written_events": self.writer.written_events if self.writer else 0,
                "lossy": self.writer.lossy if self.writer else 0,
                "writer_error": self.writer.error if self.writer else None,
                "logprobs": self.logprobs, "routing": self.routing, "all": self.record_all,
                "counters": dict(self.counters)}

    def shutdown(self) -> None:
        if self._shut:
            return
        self._shut = True
        if self.writer is not None:
            self.writer.blocking = True
        for sess in list(self.sessions.values()):
            self._close(sess, "shutdown")
        self.sessions.clear()
        if self.writer is not None:
            self.writer.put({"ev": "end", "t": time.time(), "status": self.status()})
            self.writer.close()


def _usage(meta_info: dict) -> dict:
    return {"prompt_tokens": meta_info.get("prompt_tokens"),
            "completion_tokens": meta_info.get("completion_tokens"),
            "cached_tokens": meta_info.get("cached_tokens")}


def _routing_bytes(meta_info: dict, routing_start: int, n_expected: int):
    """(bytes, info) of the routed experts of positions [routing_start, routing_start + n), or (None, why).

    The scheduler returns positions [start, seqlen - 1) (the last sampled token was never
    forwarded); ``routed_experts_meta`` carries dtype, shape [n, layers, top_k] and start."""
    val = meta_info.get("routed_experts")
    meta = meta_info.get("routed_experts_meta")
    if val is None or meta is None:
        return None, f"routed_experts {'missing' if val is None else 'present'}, meta {'missing' if meta is None else 'present'}"
    raw = base64.b64decode(val) if isinstance(val, str) else bytes(val)
    dtype, shape, start = meta["dtype"], list(meta["shape"]), int(meta["start"])
    itemsize = np.dtype(dtype).itemsize
    if start != routing_start or shape[0] != n_expected or len(raw) != int(np.prod(shape)) * itemsize:
        return None, (f"routing mismatch: start {start} (want {routing_start}), n {shape[0]} (want {n_expected}), "
                      f"{len(raw)} bytes for {shape} {dtype}")
    return raw, {"file": None, "offset": None, "nbytes": len(raw), "start": start, "n_tokens": shape[0],
                 "shape": shape, "dtype": dtype}


def _weight_versions(meta_info: dict, n_gen: int):
    """(spans, None) or (None, why): the weight version each generated token was sampled with.

    ``meta_info["weight_versions"]`` (the scheduler's per-request spans) as
    ``[{"version": str, "start": int, "end": int}, ...]``: half-open, in gen coordinates, tiling
    ``[0, len(gen))`` in order (one span unless a weight update happened while the request ran)."""
    spans = meta_info.get("weight_versions")
    if not spans:
        return None, f"meta_info weight_versions is {spans!r}"
    out, pos = [], 0
    for s in spans:
        version, start, end = s["version"], s["start"], s["end"]
        if not isinstance(version, str) or start != pos or end < start:
            return None, f"weight_versions {spans} do not tile [0, {n_gen})"
        out.append({"version": version, "start": int(start), "end": int(end)})
        pos = end
    if pos != n_gen:
        return None, f"weight_versions {spans} do not tile [0, {n_gen})"
    return out, None


MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
               "model.safetensors.index.json")


def server_meta(server_args) -> Dict[str, Any]:
    """What server.json records about the server: resolved args and model file digests."""
    import sglang

    model_path = server_args.model_path
    files = {}
    if os.path.isdir(model_path):
        for name in MODEL_FILES:
            path = os.path.join(model_path, name)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    files[name] = hashlib.sha256(f.read()).hexdigest()
    return {
        "server_args": server_args.resolved_dict(),
        "launch_command": server_args.launch_command,
        "sglang_file": sglang.__file__,
        "sglang_version": sglang.__version__,
        "model_files_sha256": files,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("ARC_", "SGLANG_")) and not any(w in k for w in ("KEY", "TOKEN", "SECRET", "PASS"))},
    }
