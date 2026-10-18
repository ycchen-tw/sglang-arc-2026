"""Own-last-turn replay (arc-forge docs/plans/replay_own_last_turn.md).

Both output-id replays -- the TITO ledger's ``session_ledger.replay_ids`` (TokenizerManager,
``--session-replay-own-last-turn``) and HiSparse h14's ``QSAHiSparseSessionCache.replay_request``
(scheduler, ``hisparse_config.qsa_session_replay_own_last_turn``) -- compare every message where the
request diverges from the session's record as text, put the record's ids back where the text is the
same, and stop at the first message whose text differs (``text_differs``).

In a multi-turn codex session the client re-renders the model's previous assistant output from parsed
objects, and for the session's last generation that is lossy in three ways seen in real data:

(a) the generation was cut by the token budget (finish ``length``) inside a tool call: the parser emits
    the call with empty arguments, so the half-written ``<parameter=...>`` is gone from the re-render;
(b) a malformed tool call (unknown function, odd parameters) is normalized by parse -> JSON -> template;
(c) the generation was cut inside ``<think>``: the template closes the block (Qwen3.8: ``\\n</think>\\n\\n``),
    the message text is no longer the record's, and the whole message is re-tokenized canonically -- it
    diverges from the sampled ids at the first non-canonical BPE split, often thousands of tokens before
    the cut.

Each one used to rewind the session (the request kept only a prefix of what the model sampled). With the
flag, the model's own sampled ids are authoritative for that one message: at the point where a replay
would stop with ``text_differs``, and only there, ``own_turn_replacement`` checks that the diverging
message is the session's last generated message and, if so, builds what replaces the client's message
``request[start:key_end)``: the record's message ``record[start:rec_end)`` and, when the generation was
cut (no stop token), a closure -- the tokens the client added after the model's text, which stand after
the record's end, i.e. in the next append's tail (prompt tokens, never counted as generated).

Only a generation that finished (stop or length) is taken over: a HiSparse record an abort or a retract
parked keeps text_differs (why ``aborted``); the TITO ledger never commits an aborted generation.

Sticky: codex re-renders the taken-over message the same lossy way in every later request, where it is
no longer the last generated one. Each takeover therefore leaves an ``OwnTurnEntry`` -- where the message
starts, the client's ids for it (length, sha256) and the replacement's (length, sha256) -- that the route
keeps with the session's record once the request's ids are committed to it. A later request that diverges
inside a message starting at an entry's ``start`` gets the record's ids for it again
(``sticky_replacement``) when the client sent exactly the same ids (else the entry goes and the replay
stops with text_differs) and the record still holds the replacement there (else it raises: the record
changed under the entry). Older history the client really changed is never touched. The routes drop an
entry when the record is cut or rewound at or before its end, and with the record or session itself.
Self-contained (numpy and the tokenizer only) so that a deploy tree can take it with either route.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
MESSAGE_END_TOKEN = "<|im_end|>"
SPAN_KIND = "own_turn"
# Sentinels rendered through the chat template to find its closure of a reasoning block; plain ASCII
# words, so the template's |trim leaves them alone and no tokenizer merges them with the template text.
_REASONING_SENTINEL = "zqOWNTURNREASONINGqz"
_CONTENT_SENTINEL = "zqOWNTURNCONTENTqz"


def _single_token(tokenizer, name: str, what: str) -> int:
    t = tokenizer.convert_tokens_to_ids(name)
    if not isinstance(t, int) or tokenizer.convert_ids_to_tokens(t) != name:
        raise ValueError(f"{what}: the tokenizer has no single {name} token (got {t!r})")
    return int(t)


def template_think_closure(tokenizer, what: str) -> str:
    """The text the chat template puts between an assistant message's reasoning and its content, rendered
    from one reasoning message (Qwen3.8: ``"\\n</think>\\n\\n"``). It must hold exactly one think-end and
    nothing but whitespace around it, else this raises."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": _REASONING_SENTINEL, "content": _CONTENT_SENTINEL},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as exc:  # re-raised: the flag cannot work without the template
        raise ValueError(f"{what}: rendering a reasoning message with the tokenizer's chat template failed: "
                         f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or text.count(_REASONING_SENTINEL) != 1 or text.count(_CONTENT_SENTINEL) != 1:
        raise ValueError(f"{what}: the chat template does not render reasoning_content and content once each "
                         f"({text!r})")
    a = text.index(_REASONING_SENTINEL) + len(_REASONING_SENTINEL)
    b = text.index(_CONTENT_SENTINEL)
    closure = text[a:b]
    if b < a or closure.count(THINK_END_TOKEN) != 1 or THINK_START_TOKEN in closure or closure.replace(
            THINK_END_TOKEN, "").strip():
        raise ValueError(f"{what}: the chat template closes reasoning with {closure!r}, not one {THINK_END_TOKEN} "
                         "between whitespace")
    return closure


@dataclass(frozen=True)
class OwnTurn:
    """What own-turn replay needs from the tokenizer, checked once at start-up (``from_tokenizer``)."""

    think_start: int
    think_end: int
    message_end: int  # <|im_end|>: ends the request's message (key_end)
    closure_text: str  # the template's canonical think closure
    closure: Tuple[int, ...]  # closure_text tokenized on its own
    encode: Callable[[str], List[int]]  # no special tokens added
    decode: Callable[[List[int]], str]

    @classmethod
    def from_tokenizer(cls, tokenizer, decode: Callable[[List[int]], str], what: str) -> "OwnTurn":
        """``what`` names the flag in errors. ``decode`` is the route's own replay decode (special tokens kept)."""
        if tokenizer is None:
            raise ValueError(f"{what} needs the tokenizer (do not use --skip-tokenizer-init)")
        start = _single_token(tokenizer, THINK_START_TOKEN, what)
        end = _single_token(tokenizer, THINK_END_TOKEN, what)
        message_end = _single_token(tokenizer, MESSAGE_END_TOKEN, what)
        closure_text = template_think_closure(tokenizer, what)

        def encode(text: str) -> List[int]:
            return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]

        closure = tuple(encode(closure_text))
        if decode(list(closure)) != closure_text or closure.count(end) != 1 or start in closure:
            raise ValueError(f"{what}: {closure_text!r} tokenizes to {closure}, which does not decode back to it "
                             f"with one {THINK_END_TOKEN}")
        return cls(think_start=start, think_end=end, message_end=message_end, closure_text=closure_text,
                   closure=closure, encode=encode, decode=decode)


def client_sha256(ids: np.ndarray) -> str:
    """sha256 of ids as little-endian int32 bytes (the ledger's id encoding): the client's ids for a message,
    or a replacement's."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i4").tobytes()).hexdigest()


@dataclass(frozen=True)
class OwnTurnEntry:
    """A message own-turn took over: the record holds the replacement, ``rep_len`` ids hashing to
    ``rep_sha256``, at ``[start, start + rep_len)``, where the client sends ``client_len`` ids hashing to
    ``client_sha256`` (followed by its <|im_end|>, which the record has at ``start + rep_len``)."""

    start: int
    client_len: int
    client_sha256: str
    rep_len: int
    rep_sha256: str

    @property
    def end(self) -> int:
        return self.start + self.rep_len

    def held_by(self, ids: np.ndarray) -> bool:
        """``ids`` (a record, or a request after replay) hold the replacement at its place."""
        return self.end <= len(ids) and client_sha256(ids[self.start:self.end]) == self.rep_sha256

    def check_held_by(self, ids: np.ndarray, what: str) -> None:
        if not self.held_by(ids):
            raise RuntimeError(f"own-turn entry at {self.start} (replacement of {self.rep_len} ids) is not held by "
                               f"{what} ({len(ids)} ids)")


def kept_entries(entries: Dict[int, OwnTurnEntry], ids: np.ndarray, cut: int,
                 what: str) -> Tuple[Dict[int, OwnTurnEntry], int]:
    """The entries that survive ids that agree with their record only before ``cut`` (a rewind, a
    truncation, the end of what a request shares with its record): those ending before it, each checked to
    be held by ``ids`` (it raises otherwise); and how many were dropped."""
    kept = {s: e for s, e in entries.items() if e.end < cut}
    for e in kept.values():
        e.check_held_by(ids, what)
    return kept, len(entries) - len(kept)


@dataclass(frozen=True)
class OwnTurnDecision:
    """``ids`` is None: own-turn does not apply (``why``), the replay stops with text_differs as before."""

    ids: Optional[np.ndarray] = None  # int64: replaces request[start:key_end)
    fields: Optional[Dict[str, object]] = None  # the span's own-turn keys
    rec_end: Optional[int] = None  # where the record's message ends: the stop token, or gen_end after a cut
    # "after_generation" | "earlier_message" | "before_generation" | "aborted"
    why: Optional[str] = None
    entry: Optional[OwnTurnEntry] = None  # what the route keeps once the request is committed (sticky)


@dataclass(frozen=True)
class StickyDecision:
    """At a message starting at an entry's start: ``ids`` the record's replacement again, or ``drop`` (the
    client sent other ids for the message: the entry goes, the replay goes on as without it)."""

    ids: Optional[np.ndarray] = None
    fields: Optional[Dict[str, object]] = None
    drop: bool = False


def sticky_replacement(
    entry: OwnTurnEntry,
    toks: np.ndarray,
    new: np.ndarray,
    start: int,
    key_end: int,
    rec_end: int,
    gen_start: int,
) -> StickyDecision:
    """A later request diverges inside the message at ``start`` (``toks`` the record, ``new`` the request so
    far, ``key_end`` its <|im_end|>, ``rec_end`` the record's message end, ``gen_start`` where the record's
    last generation starts) and ``entry.start == start``. The client's ids for the message must be exactly
    the ones the entry saw (length and sha256), else ``drop``; the record must still hold the replacement
    there, ending at ``rec_end``, else this raises (the record changed under the entry)."""
    if entry.start != start:
        raise RuntimeError(f"own-turn entry at {entry.start} asked for the message at {start}")
    if not start < gen_start:
        raise RuntimeError(f"own-turn entry at {start} lies in the record's last generation (from {gen_start})")
    if key_end - start != entry.client_len or client_sha256(new[start:key_end]) != entry.client_sha256:
        return StickyDecision(drop=True)
    entry.check_held_by(toks, "the session's record")
    if rec_end != entry.end:
        raise RuntimeError(f"own-turn entry at {start}: the record's message ends at {rec_end}, the replacement "
                           f"at {entry.end}")
    fields = {"kind": SPAN_KIND, "sticky": True, "closure": 0, "client_len": entry.client_len,
              "client_sha256": entry.client_sha256}
    return StickyDecision(ids=np.asarray(toks[start:entry.end], dtype=np.int64), fields=fields)


def _position(value, lo: int, hi: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise RuntimeError(f"own-turn replay needs {what} in [{lo}, {hi}], got {value!r}")
    return int(value)


def own_turn_replacement(
    ot: OwnTurn,
    toks: np.ndarray,
    new: np.ndarray,
    lcp: int,
    start: int,
    key_end: int,
    gen_start: int,
    gen_end: int,
    stop: np.ndarray,
    *,
    finished: bool,
) -> OwnTurnDecision:
    """At a replay's text_differs point: the ids that replace the request's message, if it is the session's
    last generated one and that generation finished (``finished``: stop or length; else why ``aborted``).

    ``toks`` is the record (raw ids), ``new`` the request's ids so far (earlier messages already replayed,
    so both agree on [0, lcp)), the message starts at ``start`` (after the last stop token before ``lcp``)
    and ends in the request at ``key_end``, its first <|im_end|> at or after ``lcp``. ``[gen_start,
    gen_end)`` is exactly where the session's last generation lies in the record, its stop token
    included (no guess: anything else raises). The TITO record ends there; a HiSparse record under
    speculative decoding goes on with the draft ids a verify accepted past the stop or the length cut,
    which the client never saw.

    It applies when the divergence lies in the last generation (``gen_start <= lcp < gen_end``) and the
    message is its last one: no stop token in ``toks[lcp:gen_end)`` but possibly the generation's final
    token. The replacement is the record's message ``toks[start:rec_end)`` -- ``rec_end`` is the stop
    token's position, or ``gen_end`` after a cut -- plus a closure:

    - the generation ended on a stop token: none -- the request's <|im_end|> at key_end follows as before;
    - it was cut (no stop) and its text is a prefix of the request message's text: the rest of the request's
      text, tokenized on its own (it must decode back exactly);
    - it was cut, not a prefix, inside an open thinking block (a think-start in the message, no think-end):
      the template's canonical closure (``OwnTurn.closure``);
    - otherwise none.

    A U+FFFD at the cut does not matter: the model may stop inside a UTF-8 character, its ids are kept.
    The decision's ``entry`` is what the route keeps for later requests (sticky); a closure holding a stop
    token would break the message bounds the entry relies on and raises.
    """
    if not isinstance(finished, bool):
        raise TypeError(f"own-turn replay: finished must be a bool, got {finished!r}")
    n = len(toks)
    gen_end = _position(gen_end, 0, n, f"the end of the session's last generation in a record of {n} ids")
    gen_start = _position(gen_start, 0, gen_end, "the exact start of the session's last generation")
    if not (0 <= start <= lcp < n and lcp <= key_end < len(new) and int(new[key_end]) == ot.message_end):
        raise RuntimeError(f"own-turn replay: inconsistent message bounds start {start}, lcp {lcp}, key_end "
                           f"{key_end} (request {len(new)} ids, record {n})")
    if lcp >= gen_end:
        return OwnTurnDecision(why="after_generation")
    after = np.flatnonzero(np.isin(toks[lcp:gen_end], stop))
    if after.size > 1 or (after.size == 1 and lcp + int(after[0]) != gen_end - 1):
        return OwnTurnDecision(why="earlier_message")
    if lcp < gen_start:
        return OwnTurnDecision(why="before_generation")
    if not finished:
        return OwnTurnDecision(why="aborted")
    ended = after.size == 1
    rec_end = gen_end - 1 if ended else gen_end
    rec = np.asarray(toks[start:rec_end], dtype=np.int64)
    closure: List[int] = []
    if not ended:
        rec_text = ot.decode(rec.tolist())
        req_text = ot.decode(new[start:key_end].tolist())
        if req_text.startswith(rec_text):
            suffix = req_text[len(rec_text):]
            if suffix:
                closure = ot.encode(suffix)
                if ot.decode(closure) != suffix:
                    raise RuntimeError(f"own-turn replay: the client's text after the cut {suffix!r} tokenizes to "
                                       f"{closure}, which does not decode back to it")
        elif (rec == ot.think_start).any() and not (rec == ot.think_end).any():
            closure = list(ot.closure)
    if np.isin(np.asarray(closure, dtype=np.int64), stop).any():
        raise RuntimeError(f"own-turn replay: the closure {closure} holds a stop token")
    ids = np.concatenate((rec, np.asarray(closure, dtype=np.int64)))
    client = client_sha256(new[start:key_end])
    fields = {"kind": SPAN_KIND, "closure": len(closure), "client_len": int(key_end - start), "client_sha256": client}
    entry = OwnTurnEntry(start=int(start), client_len=int(key_end - start), client_sha256=client,
                         rep_len=int(len(ids)), rep_sha256=client_sha256(ids))
    return OwnTurnDecision(ids=ids, fields=fields, rec_end=rec_end, entry=entry)
