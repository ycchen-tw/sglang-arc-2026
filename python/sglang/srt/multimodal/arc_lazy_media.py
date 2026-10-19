"""arc-forge: lazy media for the Qwen3-VL family (opt-in, ``ARC_MM_LAZY_MEDIA=1``).

An agent client (codex) resends the whole conversation every turn, every screenshot and video in it included. Without
this, the frontend decodes and preprocesses all of them on every request (videos are never cached: any request with a
video takes ``_process_mm_data_uncached``), ships all their pixels to the scheduler, and the scheduler then uses only
the items that overlap the extend -- the rest are inside the cached prefix. On G4 c110 that frontend work grew with the
history (three 111-frame videos in one game) until it was the bottleneck.

With this on, the frontend identifies each image / video by content (the h41 artifact key: digest of the bytes the
decoder would see + processor fingerprint + per-request preprocess options) and keeps, per key, what the processor made
of it the first time: its expanded token block, pad spans and grid (``MediaMeta``). A request then

- tokenizes its prompt text (one placeholder per item, as the chat template rendered it),
- runs the real processor only on items it has never seen, in one call with a prompt of just their placeholders,
- splices every item's expanded block in place of its placeholder, and computes M-RoPE on the spliced ids,
- sends pixels only for the items it processed; the others go as *lazy* items: offsets, hash and grid, no feature.

"Seen" is counted per content key for a request without a cache_salt. A request with one (a radix namespace the
scheduler shares with no other salt: an RL episode) counts per occurrence instead -- (cache_salt, extra_key, the item's
position among the prompt's media, content key) -- so an item goes lazy only where an earlier request of the same
namespace carried that very item at that place in the conversation, i.e. where that request's prefill left it in this
namespace's prefix. Without it, the same screenshot in another episode, or a repeated frame later in the same one,
went lazy and then sat in the extend.

A lazy item that the scheduler finds overlapping a prefill extend (its prefix was evicted, or the prefix match stops
short of it) cannot be embedded. The scheduler then takes the request out of the waiting queue before anything is
allocated for it and parks it (``Scheduler.arc_mm_lazy_parked``) as it is: its ids (after the session ledger's or
h14's replay), logprob / routed-experts flags forced at intake, and what its match holds (a session park claim, a
QSA session ownership) all stay. It sends ``ArcMmRematerializeReq`` with the hashes of those lazy items; the tokenizer
manager processes the request's media again with them materialized and sends the result back as a fill
(``arc_mm_refill``), from which the scheduler takes only those items' features (``fill_features``) before the request
re-enters the queue. Nothing upstream of the scheduler runs twice: no second session-ledger intake, no second
arrival in the scheduler. Prompt ids and hashes do not depend on which items carry pixels, so the fill's items line
up with the parked request's (same hash, modality, pad value and span lengths, else the request is aborted).

Videos expand into several pad runs (one per timestamped frame group). The scheduler's multi-offset path encodes all
of a request's videos under one combined key, which misses whenever a video is added; with this on, multi-offset items
go through the per-item path instead (one cache entry per item, only items overlapping the extend are encoded).

A retracted request whose items carry no feature (lazy, or released at prefill end with ARC_MM_RELEASE_AFTER_PREFILL)
is parked and filled the same way (refill: every featureless item).

With data parallelism the fill is routed back to the DP rank that parked the request
(``ArcMmRematerializeReq.dp_rank``).

Refused, loudly: audio, preprocessed / precomputed inputs, a prompt whose placeholders do not match the media, PD
disaggregation.
"""

from __future__ import annotations

import copy
import os
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Collection, Hashable, List, Optional, Sequence, Tuple

ENV = "ARC_MM_LAZY_MEDIA"
ENTRIES_ENV = "ARC_MM_LAZY_MEDIA_ENTRIES"
# an item goes lazy from its MIN_SEEN-th request on (1: as soon as one earlier request carried it). With a prefix
# cache that does not keep the previous turn's tail (plain radix + mamba keeps the state at the prefill end only) the
# newest media of the last turn is usually still in the extend; 2 sends it with pixels once more instead of a round trip
MIN_SEEN_ENV = "ARC_MM_LAZY_MEDIA_MIN_SEEN"
# model_specific_data key on an item sent without its feature
LAZY_KEY = "arc_lazy_media"
# GenerateReqInput attribute, set for a fill (the scheduler parked the request and asked for these): the hashes the
# processor must materialize. A fill counts nothing as seen: the request was counted when it arrived.
MATERIALIZE_ATTR = "arc_mm_materialize_hashes"
# GenerateReqInput attribute: every hash the scheduler asked for so far for lazy items of a waiting request (a lazy item
# asked for twice means a fill did not give it its feature)
ASKED_ATTR = "arc_mm_lazy_asked_hashes"


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{name}={raw!r}: expected 0 or 1")
    return raw == "1"


_ENABLED = _env_flag(ENV)


def enabled() -> bool:
    return _ENABLED


def min_seen() -> int:
    raw = os.environ.get(MIN_SEEN_ENV, "1").strip()
    n = int(raw)
    if n <= 0:
        raise ValueError(f"{MIN_SEEN_ENV}={raw!r}: expected a positive integer")
    return n


def cache_entries() -> int:
    raw = os.environ.get(ENTRIES_ENV, "16384").strip()
    n = int(raw)
    if n <= 0:
        raise ValueError(f"{ENTRIES_ENV}={raw!r}: expected a positive integer")
    return n


@dataclass(frozen=True)
class MediaMeta:
    """What the processor made of one image / video, minus its pixels."""

    modality: str  # "image" | "video"
    # the item's expanded tokens: an image's pad run; a video from its outer <|vision_start|> to its outer <|vision_end|>
    block: array
    # pad runs inside ``block`` (inclusive, relative to the block start); one for an image, one per frame group
    pad_spans: Tuple[Tuple[int, int], ...]
    # the processor's grid row for this item ([t, h, w])
    grid_thw: Tuple[int, int, int]
    hash: int


class MediaMetaCache:
    """LRU of ``MediaMeta`` by artifact key, with how many requests without a cache_salt carried each, and an LRU of
    how many requests of a salted namespace carried each occurrence (see the module docstring); both hold up to
    ``max_entries``. Touched only from the tokenizer manager's event loop."""

    def __init__(self, max_entries: int, min_seen: int):
        self.max_entries = max_entries
        self.min_seen = min_seen
        self._entries: "OrderedDict[str, list]" = OrderedDict()  # key -> [meta, unsalted requests that carried it]
        # ((namespace, position among the prompt's media), key) -> requests of that namespace that carried it there
        self._occurrences: "OrderedDict[tuple, int]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def lookup(self, key: str, occurrence: Optional[Hashable] = None, *, count: bool = True) -> Optional[MediaMeta]:
        """The meta if the item may go lazy in this request; counts this request as one more that carried it.

        ``occurrence`` None (no cache_salt): counted per content key. Otherwise ``(namespace, position)``
        (``seen_scope`` and the item's index among the prompt's media): counted per occurrence. ``count`` False (a
        fill re-processing a request that was counted when it arrived) only looks."""
        entry = self._entries.get(key)
        if occurrence is None:
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            seen = entry[1]
            if count:
                entry[1] = seen + 1
        else:
            okey = (occurrence, key)
            seen = self._occurrences.get(okey, 0)
            if count:
                self._occurrences[okey] = seen + 1
                self._occurrences.move_to_end(okey)
                while len(self._occurrences) > self.max_entries:
                    self._occurrences.popitem(last=False)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
        if seen < self.min_seen:
            self.misses += 1
            return None
        self.hits += 1
        return entry[0]

    def put(self, key: str, meta: MediaMeta, *, salted: bool = False) -> None:
        """What the processor made of ``key`` for this request. A salted request is counted by lookup (per
        occurrence), so the key's count of unsalted requests starts at 0."""
        entry = self._entries.get(key)
        if entry is not None:
            old = entry[0]
            if old.block != meta.block or old.grid_thw != meta.grid_thw or old.hash != meta.hash:
                raise RuntimeError(f"{ENV}: the processor expanded media {key} differently on two requests")
            self._entries.move_to_end(key)
            return
        self._entries[key] = [meta, 0 if salted else 1]
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def seen_scope(request_obj) -> Optional[Tuple[str, Optional[str]]]:
    """The namespace a request's media count as seen in: ``(cache_salt, extra_key)`` -- the radix cache's namespace
    -- for a request with a cache_salt, None (per content key) without one. EmbeddingReqInput has no cache_salt."""
    salt = getattr(request_obj, "cache_salt", None)
    if salt is None:
        return None
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"{ENV}: cache_salt {salt!r} of a single request should be a non-empty string")
    return salt, request_obj.extra_key


def pad_runs(block: Sequence[int], pad_token_id: int) -> Tuple[Tuple[int, int], ...]:
    runs, i, n = [], 0, len(block)
    while i < n:
        if block[i] != pad_token_id:
            i += 1
            continue
        j = i
        while j + 1 < n and block[j + 1] == pad_token_id:
            j += 1
        runs.append((i, j))
        i = j + 1
    return tuple(runs)


def placeholder_sequence(client_ids: Sequence[int], image_token_id: int, video_token_id: int) -> List[str]:
    """Modalities of the prompt's media placeholders, in order (an image is one pad, a video ``<vs><video_pad><ve>``)."""
    seq = []
    for token in client_ids:
        if token == image_token_id:
            seq.append("image")
        elif token == video_token_id:
            seq.append("video")
    return seq


def splice_blocks(
    client_ids: Sequence[int],
    blocks: Sequence[Sequence[int]],
    image_token_id: int,
    video_token_id: int,
    vision_start_id: int,
    vision_end_id: int,
) -> Tuple[List[int], List[int]]:
    """Replace each placeholder of ``client_ids`` by the next of ``blocks`` (one per placeholder, prompt order).

    An image placeholder is its single ``<|image_pad|>`` (the template's ``<|vision_start|>`` / ``<|vision_end|>``
    around it are kept); a video placeholder is ``<|vision_start|><|video_pad|><|vision_end|>`` and its block carries
    its own outer pair. Returns the spliced ids and each block's start in them. The same contract as
    base_processor.splice_timestamped_media_blocks, with the blocks given per item."""
    out: List[int] = []
    starts: List[int] = []
    k, i, n = 0, 0, len(client_ids)
    while i < n:
        token = client_ids[i]
        if token == image_token_id or token == video_token_id:
            if k == len(blocks):
                raise ValueError(f"the prompt has more media placeholders than the {len(blocks)} blocks")
            if token == video_token_id:
                if not out or out[-1] != vision_start_id or i + 1 >= n or client_ids[i + 1] != vision_end_id:
                    raise ValueError(f"video placeholder at {i} is not <|vision_start|><|video_pad|><|vision_end|>")
                out.pop()
                i += 1
            starts.append(len(out))
            out.extend(blocks[k])
            k += 1
        else:
            out.append(token)
        i += 1
    if k != len(blocks):
        raise ValueError(f"the prompt has {k} media placeholders for {len(blocks)} blocks")
    return out, starts


def item_needs_feature(item, prefix_len: int) -> bool:
    return any(end >= prefix_len for _, end in item.offsets)


def featureless_hashes(mm_inputs) -> List[int]:
    """Hashes of the items that carry no feature: lazy ones, and with ARC_MM_RELEASE_AFTER_PREFILL=1 every item of a
    request whose prefill completed. A retracted request is prefilled again from scratch and needs them back."""
    if mm_inputs is None:
        return []
    hashes = []
    for item in mm_inputs.mm_items:
        if item is None or item.feature is not None or item.precomputed_embeddings is not None:
            continue
        if item.hash not in hashes:
            hashes.append(item.hash)
    return hashes


def items_to_materialize(mm_inputs, prefix_len: int) -> List[int]:
    """Hashes (each once, prompt order) of the lazy items -- no feature -- that a prefill starting at ``prefix_len``
    would have to embed; [] when it needs none. Items that carry a feature are not listed: the scheduler parks the
    request as it is and a fill gives features to these only."""
    if mm_inputs is None:
        return []
    needed = []
    for item in mm_inputs.mm_items:
        if item is None or not item_needs_feature(item, prefix_len):
            continue
        if not item.model_specific_data.get(LAZY_KEY, False):
            continue
        if item.feature is not None or item.precomputed_embeddings is not None:
            raise RuntimeError(f"{ENV}: a lazy item carries a feature (hash {item.hash})")
        if item.hash not in needed:
            needed.append(item.hash)
    return needed


def _has_feature(item) -> bool:
    return item.feature is not None or item.precomputed_embeddings is not None


def fill_features(items: list, fresh_items: Sequence[Any], wanted: Optional[Collection[int]]) -> int:
    """Give a parked request's featureless items (``items``, changed in place) the features a fill brought
    (``fresh_items``: the tokenizer manager's processing of the same request with those items materialized).

    ``wanted`` None (a retracted request, prefilled again from scratch): every featureless item. Otherwise the hashes
    the scheduler asked for (lazy items in the extend); other featureless items stay as they are. A filled item is a
    copy of the fresh one (a request may carry the same picture twice; each occurrence keeps its own offsets) with the
    parked item's offsets (the session replay may have moved the text around them). Returns how many items were
    filled. Raises when the fill lacks a feature it was asked for, an item comes back as something else (modality,
    pad value, span lengths), or a wanted hash is not featureless in the parked request."""
    fresh = {}
    for item in fresh_items:
        if item is not None and _has_feature(item):
            fresh.setdefault(item.hash, item)
    missing = None if wanted is None else set(wanted)
    filled = 0
    for i, item in enumerate(items):
        if item is None or _has_feature(item):
            continue
        if wanted is not None and item.hash not in wanted:
            continue
        new = fresh.get(item.hash)
        if new is None:
            raise RuntimeError(f"the fill has no feature for media item {item.hash}")
        if (
            new.modality != item.modality
            or new.pad_value != item.pad_value
            or [e - s for s, e in new.offsets] != [e - s for s, e in item.offsets]
        ):
            raise RuntimeError(
                f"media item {item.hash} came back as {new.modality} spans {new.offsets}, "
                f"was {item.modality} spans {item.offsets}"
            )
        new = copy.copy(new)
        new.model_specific_data = dict(new.model_specific_data)
        new.offsets = item.offsets
        items[i] = new
        filled += 1
        if missing is not None:
            missing.discard(item.hash)
    if missing:
        raise RuntimeError(
            f"asked to fill media items {sorted(missing)}, which the parked request does not hold without a feature"
        )
    return filled
