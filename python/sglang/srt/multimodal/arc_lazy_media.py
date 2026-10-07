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

A lazy item that the scheduler finds overlapping a prefill extend (its prefix was evicted, or the prefix match stops
short of it) cannot be embedded. The scheduler then takes the request out of the waiting queue before anything is
allocated for it and sends ``ArcMmRematerializeReq`` with those items' hashes to the tokenizer manager, which
re-processes the request with them materialized and dispatches it again. Prompt ids and hashes do not depend on which
items carry pixels, so the resent request matches the same prefix.

Videos expand into several pad runs (one per timestamped frame group). The scheduler's multi-offset path encodes all
of a request's videos under one combined key, which misses whenever a video is added; with this on, multi-offset items
go through the per-item path instead (one cache entry per item, only items overlapping the extend are encoded).

Refused, loudly: audio, preprocessed / precomputed inputs, a prompt whose placeholders do not match the media, a lazy
item that is needed after a retraction (as with ARC_MM_RELEASE_AFTER_PREFILL).
"""

from __future__ import annotations

import os
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

ENV = "ARC_MM_LAZY_MEDIA"
ENTRIES_ENV = "ARC_MM_LAZY_MEDIA_ENTRIES"
# an item goes lazy from its MIN_SEEN-th request on (1: as soon as one earlier request carried it). With a prefix
# cache that does not keep the previous turn's tail (plain radix + mamba keeps the state at the prefill end only) the
# newest media of the last turn is usually still in the extend; 2 sends it with pixels once more instead of a round trip
MIN_SEEN_ENV = "ARC_MM_LAZY_MEDIA_MIN_SEEN"
# model_specific_data key on an item sent without its feature
LAZY_KEY = "arc_lazy_media"
# GenerateReqInput attribute: hashes the frontend must materialize (set when the scheduler sends a request back)
MATERIALIZE_ATTR = "arc_mm_materialize_hashes"


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
    """LRU of ``MediaMeta`` by artifact key, with how many requests carried each. Touched only from the tokenizer
    manager's event loop."""

    def __init__(self, max_entries: int, min_seen: int):
        self.max_entries = max_entries
        self.min_seen = min_seen
        self._entries: "OrderedDict[str, list]" = OrderedDict()  # key -> [meta, requests that carried it]
        self.hits = 0
        self.misses = 0

    def lookup(self, key: str) -> Optional[MediaMeta]:
        """The meta if the item may go lazy in this request; counts this request as one more that carried it."""
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        seen = entry[1]
        entry[1] = seen + 1
        if seen < self.min_seen:
            self.misses += 1
            return None
        self.hits += 1
        return entry[0]

    def put(self, key: str, meta: MediaMeta) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            old = entry[0]
            if old.block != meta.block or old.grid_thw != meta.grid_thw or old.hash != meta.hash:
                raise RuntimeError(f"{ENV}: the processor expanded media {key} differently on two requests")
            self._entries.move_to_end(key)
            return
        self._entries[key] = [meta, 1]
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


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
    """If a prefill starting at ``prefix_len`` would have to embed a lazy item (no feature): the hashes of every item
    it would embed, lazy or not -- the resent request is processed afresh, and an item that was new (sent with its
    pixels) is a seen one by then. Otherwise []."""
    if mm_inputs is None:
        return []
    overlapping, lazy = [], False
    for item in mm_inputs.mm_items:
        if item is None or not item_needs_feature(item, prefix_len):
            continue
        if item.model_specific_data.get(LAZY_KEY, False):
            if item.feature is not None or item.precomputed_embeddings is not None:
                raise RuntimeError(f"{ENV}: a lazy item carries a feature (hash {item.hash})")
            lazy = True
        if item.hash not in overlapping:
            overlapping.append(item.hash)
    return overlapping if lazy else []
