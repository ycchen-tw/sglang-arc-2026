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
    # the prefix stops inside the last image: that one and every other item in the extend, new or not
    assert items_to_materialize(inputs, 22) == [last.hash, new.hash]
    assert items_to_materialize(inputs, 0) == [old.hash, last.hash, new.hash]
    assert items_to_materialize(_Inputs([new]), 0) == []


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
