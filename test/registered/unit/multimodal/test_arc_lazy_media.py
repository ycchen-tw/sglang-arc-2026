"""arc lazy media (ARC_MM_LAZY_MEDIA=1): media seen before go to the scheduler without pixels.

The frontend keeps per media (by content) what the processor expanded it into and splices those blocks into the
prompt; the scheduler hands a request back when a lazy item overlaps its extend, and encodes videos (one placeholder
span per frame group) per item. See sglang.srt.multimodal.arc_lazy_media.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from array import array  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers import mm_schedule  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
)
from sglang.srt.multimodal import arc_lazy_media  # noqa: E402
from sglang.srt.multimodal.arc_lazy_media import (  # noqa: E402
    LAZY_KEY,
    MediaMeta,
    MediaMetaCache,
    featureless_hashes,
    items_to_materialize,
    pad_runs,
    splice_blocks,
)

IMG, VID, VS, VE = 9, 10, 7, 8
# client: text 1, an image, text 4, a video, text 5 (what the chat template renders)
CLIENT = [1, VS, IMG, VE, 4, VS, VID, VE, 5]
IMAGE_BLOCK = [IMG, IMG, IMG]
# a video of two frame groups with timestamp text 30 31 / 32 33, from its outer <vs> to its outer <ve>
VIDEO_BLOCK = [VS, 30, 31, VS, VID, VID, VE, 32, 33, VS, VID, VID, VE, VE]


def test_splice_blocks():
    ids, starts = splice_blocks(CLIENT, [IMAGE_BLOCK, VIDEO_BLOCK], IMG, VID, VS, VE)
    assert ids == [1, VS] + IMAGE_BLOCK + [VE, 4] + VIDEO_BLOCK + [5]
    assert starts == [2, 7]
    assert ids[starts[1] : starts[1] + len(VIDEO_BLOCK)] == VIDEO_BLOCK


def test_splice_blocks_rejects_mismatches():
    with pytest.raises(ValueError, match="more media placeholders"):
        splice_blocks(CLIENT, [IMAGE_BLOCK], IMG, VID, VS, VE)
    with pytest.raises(ValueError, match="2 media placeholders for 3 blocks"):
        splice_blocks(CLIENT, [IMAGE_BLOCK, VIDEO_BLOCK, IMAGE_BLOCK], IMG, VID, VS, VE)
    with pytest.raises(ValueError, match="not <\\|vision_start\\|>"):
        splice_blocks([1, VID, VE], [VIDEO_BLOCK], IMG, VID, VS, VE)


def test_pad_runs():
    assert pad_runs(IMAGE_BLOCK, IMG) == ((0, 2),)
    assert pad_runs(VIDEO_BLOCK, VID) == ((4, 5), (10, 11))


def _meta(h=1, block=IMAGE_BLOCK):
    return MediaMeta("image", array("i", block), pad_runs(block, IMG), (1, 2, 6), h)


def test_cache_min_seen():
    cache = MediaMetaCache(max_entries=4, min_seen=1)
    assert cache.lookup("a") is None
    cache.put("a", _meta())  # the request that processed it
    assert cache.lookup("a") is not None  # the next one sends it lazy

    cache = MediaMetaCache(max_entries=4, min_seen=2)
    assert cache.lookup("a") is None
    cache.put("a", _meta())
    assert cache.lookup("a") is None  # carried once before: still with pixels
    cache.put("a", _meta())
    assert cache.lookup("a") is not None


def test_cache_rejects_a_different_expansion_and_evicts_lru():
    cache = MediaMetaCache(max_entries=2, min_seen=1)
    cache.put("a", _meta())
    with pytest.raises(RuntimeError, match="differently"):
        cache.put("a", _meta(block=[IMG, IMG]))
    cache.put("b", _meta(2))
    assert cache.lookup("a") is not None  # a is now the most recent
    cache.put("c", _meta(3))
    assert cache.lookup("b") is None and len(cache) == 2


class _CacheBeforeSalts:
    """MediaMetaCache as it was before salted requests counted occurrences (f31bd1bcf), for the equivalence test."""

    def __init__(self, max_entries, min_seen):
        from collections import OrderedDict

        self.max_entries, self.min_seen = max_entries, min_seen
        self._entries = OrderedDict()
        self.hits = self.misses = 0

    def lookup(self, key):
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

    def put(self, key, meta):
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return
        self._entries[key] = [meta, 1]
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


@pytest.mark.parametrize("min_seen", [1, 2, 3])
def test_cache_without_salt_is_unchanged(min_seen):
    """No cache_salt (the Kaggle line): every lookup / put answers as before, hit and miss counts included."""
    import random

    rng = random.Random(min_seen)
    new, old = MediaMetaCache(max_entries=5, min_seen=min_seen), _CacheBeforeSalts(5, min_seen)
    metas = {k: _meta(h) for h, k in enumerate("abcdefgh")}
    for _ in range(3000):
        key = rng.choice("abcdefgh")
        if rng.random() < 0.6:
            got, want = new.lookup(key), old.lookup(key)
            assert (got is None) == (want is None)
            if got is None:
                new.put(key, metas[key])
                old.put(key, metas[key])
        else:  # a put without a lookup (another item of the same request processed it)
            new.put(key, metas[key])
            old.put(key, metas[key])
        assert (new.hits, new.misses, len(new)) == (old.hits, old.misses, len(old._entries))
        assert list(new._entries) == list(old._entries)
        assert [e[1] for e in new._entries.values()] == [e[1] for e in old._entries.values()]
    assert len(new._occurrences) == 0


def test_cache_salted_counts_occurrences():
    """A salted request: an item goes lazy only where an earlier request of the same namespace carried it at the
    same position among the prompt's media."""
    cache = MediaMetaCache(max_entries=64, min_seen=1)
    a, b = ("ep-A", None), ("ep-B", None)
    # episode A, turn 1: frame k at position 0, processed with pixels
    assert cache.lookup("k", (a, 0)) is None
    cache.put("k", _meta(), salted=True)
    # turn 2 of A: position 0 is in A's prefix (lazy); the same frame again at position 1 is new there
    assert cache.lookup("k", (a, 0)) is not None
    assert cache.lookup("k", (a, 1)) is None
    # episode B (another sample of the same level) never carried it: pixels, although the meta is known
    assert cache.lookup("k", (b, 0)) is None
    cache.put("k", _meta(), salted=True)  # B processes it; the meta must agree
    assert cache.lookup("k", (b, 0)) is not None
    # A's turn 3 carries both of its occurrences: both are in A's prefix now
    assert cache.lookup("k", (a, 0)) is not None and cache.lookup("k", (a, 1)) is not None
    # another extra_key is another radix namespace
    assert cache.lookup("k", (("ep-A", "x"), 0)) is None
    # salted requests never count for unsalted ones: a key first processed by a salted request starts at 0
    assert cache.lookup("k") is None
    assert cache.lookup("k") is not None


def test_cache_salted_min_seen_and_lru():
    cache = MediaMetaCache(max_entries=2, min_seen=2)
    ns = ("ep", None)
    assert cache.lookup("k", (ns, 0)) is None
    cache.put("k", _meta(), salted=True)
    assert cache.lookup("k", (ns, 0)) is None  # carried by one earlier request: still with pixels
    assert cache.lookup("k", (ns, 0)) is not None
    # occurrences are an LRU of max_entries too
    cache.lookup("k", (ns, 1))
    cache.lookup("k", (ns, 2))
    assert len(cache._occurrences) == 2 and ((ns, 0), "k") not in cache._occurrences
    # the meta itself went: a counted occurrence still misses
    cache.put("j", _meta(2), salted=True)
    cache.put("i", _meta(3), salted=True)
    assert cache.lookup("k", (ns, 2)) is None


def test_cache_fill_counts_nothing():
    """A fill (the scheduler parked the request and asked for its lazy items) re-processes a request that was
    counted when it arrived: it looks without counting, salted or not."""
    cache = MediaMetaCache(max_entries=8, min_seen=2)
    cache.put("k", _meta())
    assert cache.lookup("k", count=False) is None
    assert cache._entries["k"][1] == 1
    assert cache.lookup("k") is None and cache._entries["k"][1] == 2
    assert cache.lookup("k", count=False) is not None and cache._entries["k"][1] == 2
    ns = ("ep", None)
    assert cache.lookup("k", (ns, 0), count=False) is None
    assert ((ns, 0), "k") not in cache._occurrences


def test_seen_scope():
    from types import SimpleNamespace

    assert arc_lazy_media.seen_scope(SimpleNamespace(cache_salt=None, extra_key="x")) is None
    assert arc_lazy_media.seen_scope(SimpleNamespace(cache_salt="p/ep-3", extra_key=None)) == ("p/ep-3", None)
    assert arc_lazy_media.seen_scope(SimpleNamespace(cache_salt="p/ep-3", extra_key="x")) == ("p/ep-3", "x")
    assert arc_lazy_media.seen_scope(SimpleNamespace()) is None  # EmbeddingReqInput: no cache_salt
    with pytest.raises(ValueError, match="cache_salt"):
        arc_lazy_media.seen_scope(SimpleNamespace(cache_salt=["a"], extra_key=None))


def _item(h, offsets, lazy, modality=Modality.VIDEO, rows=None):
    item = MultimodalDataItem(modality=modality, offsets=offsets)
    if lazy:
        item.model_specific_data[LAZY_KEY] = True
    else:
        n = sum(b - a + 1 for a, b in offsets)
        item.feature = torch.zeros(n if rows is None else rows, 2)
    item.set_hash(h)
    return item


class _Inputs:
    def __init__(self, items):
        self.mm_items = items


def test_items_to_materialize():
    old = _item(11, [(2, 5), (8, 11)], lazy=True)
    last = _item(12, [(20, 23)], lazy=True, modality=Modality.IMAGE)
    new = _item(13, [(30, 33), (36, 39)], lazy=False)
    inputs = _Inputs([old, last, new])
    # the prefix covers the old video and the last image: nothing to send back
    assert items_to_materialize(inputs, 24) == []
    # the prefix stops inside the last image: only the lazy items in the extend (the parked request keeps the new
    # video's feature; a fill gives features to these alone)
    assert items_to_materialize(inputs, 22) == [last.hash]
    assert items_to_materialize(inputs, 0) == [old.hash, last.hash]
    assert items_to_materialize(_Inputs([new]), 0) == []
    # the same picture twice (a repeated frame): asked for once
    again = _item(12, [(26, 29)], lazy=True, modality=Modality.IMAGE)
    assert items_to_materialize(_Inputs([last, again, new]), 0) == [last.hash]
    bad = _item(14, [(40, 41)], lazy=True, modality=Modality.IMAGE)
    bad.feature = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="lazy item carries a feature"):
        items_to_materialize(_Inputs([bad]), 0)


def _fresh(h, offsets, modality=Modality.IMAGE):
    """An item of the fill: the tokenizer's processing of the same request, the item materialized."""
    item = _item(h, offsets, lazy=False, modality=modality)
    item.feature = torch.full((sum(b - a + 1 for a, b in offsets), 2), float(h))
    return item


def test_fill_features_gives_the_asked_items_their_features():
    # parked: a lazy image in the prefix, the same lazy frame twice in the extend, a new image with its feature
    in_prefix = _item(11, [(2, 5)], lazy=True, modality=Modality.IMAGE)
    frame_a = _item(12, [(10, 13)], lazy=True, modality=Modality.IMAGE)
    frame_b = _item(12, [(20, 23)], lazy=True, modality=Modality.IMAGE)
    new = _item(13, [(30, 33)], lazy=False, modality=Modality.IMAGE)
    kept = new.feature
    items = [in_prefix, frame_a, frame_b, new]
    # the fill's offsets are the client's (no session replay there): 3 tokens earlier
    fill = [_item(11, [(2, 5)], lazy=True, modality=Modality.IMAGE), _fresh(12, [(7, 10)]), _fresh(12, [(17, 20)]),
            _fresh(13, [(27, 30)])]
    assert arc_lazy_media.fill_features(items, fill, {12}) == 2
    assert items[0] is in_prefix and LAZY_KEY in items[0].model_specific_data  # not asked for: stays lazy
    assert items[3] is new and items[3].feature is kept  # had its feature: untouched
    a, b = items[1], items[2]
    assert a is not b  # each occurrence keeps its own offsets
    assert a.offsets == [(10, 13)] and b.offsets == [(20, 23)]  # the parked request's positions
    assert a.feature is not None and LAZY_KEY not in a.model_specific_data
    assert items_to_materialize(_Inputs(items), 6) == []


def test_fill_features_refill_takes_every_featureless_item():
    lazy = _item(11, [(2, 5)], lazy=True, modality=Modality.IMAGE)
    released = _item(12, [(8, 11)], lazy=False, modality=Modality.IMAGE)
    released.feature = None
    items = [lazy, released]
    assert arc_lazy_media.fill_features(items, [_fresh(11, [(2, 5)]), _fresh(12, [(8, 11)])], None) == 2
    assert all(item.feature is not None for item in items)


def test_fill_features_fails_loudly():
    def parked():
        return [_item(12, [(10, 13)], lazy=True, modality=Modality.IMAGE)]

    with pytest.raises(RuntimeError, match="no feature for media item"):
        arc_lazy_media.fill_features(parked(), [_item(12, [(10, 13)], lazy=True, modality=Modality.IMAGE)], {12})
    with pytest.raises(RuntimeError, match="came back as"):
        arc_lazy_media.fill_features(parked(), [_fresh(12, [(10, 14)])], {12})
    with pytest.raises(RuntimeError, match="came back as"):
        arc_lazy_media.fill_features(parked(), [_fresh(12, [(10, 13)], modality=Modality.VIDEO)], {12})
    with pytest.raises(RuntimeError, match="does not hold without a feature"):
        arc_lazy_media.fill_features(parked(), [_fresh(12, [(10, 13)]), _fresh(15, [(1, 2)])], {12, 15})


def test_featureless_hashes():
    # a retracted request gets back every item without a feature: lazy ones and ones released at prefill end
    lazy = _item(11, [(2, 5)], lazy=True)
    released = _item(12, [(8, 11)], lazy=False)
    released.feature = None
    kept = _item(13, [(20, 23)], lazy=False)
    assert kept.feature is not None
    assert featureless_hashes(_Inputs([lazy, released, kept])) == [lazy.hash, released.hash]
    assert featureless_hashes(_Inputs([kept])) == []
    assert featureless_hashes(None) == []


def _rows(item_index, n):
    # the encoder's output for one item: row j is (item_index, j)
    return torch.stack([torch.full((n,), float(item_index)), torch.arange(n, dtype=torch.float32)], dim=1)


@pytest.fixture
def per_item_videos(monkeypatch):
    monkeypatch.setattr(arc_lazy_media, "_ENABLED", True)
    mm_schedule.init_mm_embedding_cache(64 * 1024 * 1024)
    yield
    mm_schedule.init_mm_embedding_cache(0)


def test_videos_are_encoded_per_item(per_item_videos):
    # a lazy video inside the prefix, a new video of two frame groups whose second group a chunk boundary cuts
    old = _item(21, [(1, 2), (4, 5)], lazy=True)
    new = _item(22, [(8, 10), (13, 15)], lazy=False)
    seen = []

    def encode(items):
        seen.extend(item.hash for item in items)
        return [_rows(1, 6)]

    input_ids = torch.zeros(20, dtype=torch.long)
    chunk, _ = mm_schedule._get_chunked_prefill_embedding(
        encode, [old, new], [0, 2], [6], [8], [[(1, 2), (4, 5), (8, 10), (13, 15)]], input_ids
    )
    assert seen == [new.hash]  # the prefix video is neither encoded nor needs pixels
    # extend [6, 14): all of the first group (rows 0-2), the first token of the second (row 3)
    assert chunk.tolist() == _rows(1, 6)[:4].tolist()
    # the next chunk [14, 20) takes the rest from the cache, without encoding again
    chunk, _ = mm_schedule._get_chunked_prefill_embedding(
        encode, [old, new], [0, 2], [14], [6], [[(1, 2), (4, 5), (8, 10), (13, 15)]], input_ids
    )
    assert seen == [new.hash] and chunk.tolist() == _rows(1, 6)[4:].tolist()


def test_lazy_item_in_the_extend_fails_loudly(per_item_videos):
    lazy = _item(31, [(1, 2), (4, 5)], lazy=True)
    with pytest.raises(RuntimeError, match="reached the encoder"):
        mm_schedule._get_chunked_prefill_embedding(
            lambda items: [_rows(0, 4)], [lazy], [0, 1], [0], [8], [[(1, 2), (4, 5)]],
            torch.zeros(8, dtype=torch.long),
        )
