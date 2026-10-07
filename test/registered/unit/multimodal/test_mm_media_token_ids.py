"""ARC: --ple-mm-placeholder-ids -- media positions of input_ids hold the modality's placeholder token.

embed_mm_inputs clamps the radix pad hashes in input_ids to vocab_size - 1 in place. That is harmless for the token
embedding (the media embeddings overwrite those rows) but not for Qwen4's PLE, which hashes this same tensor
(forward_batch.input_ids) afterwards; HF and vLLM feed it <|image_pad|> / <|video_pad|>. A model that declares
``mm_media_token_ids`` gets those ids written at its media positions instead.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from sglang.srt.managers.mm_schedule import init_mm_embedding_cache  # noqa: E402
from sglang.srt.managers.mm_utils import (  # noqa: E402
    embed_mm_inputs,
    write_mm_media_token_ids_,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

VOCAB, HIDDEN, IMAGE, VIDEO = 100, 4, 90, 91
# embed_mm_inputs stages the placeholder ids in pinned host memory
DEVICE = "cuda"
init_mm_embedding_cache(0)  # every item is a miss, encoded whole and sliced to the chunk


class Model(nn.Module):
    """Media features are 1000 + the item's row index, so they are told apart from token embeddings."""

    def __init__(self, media_token_ids):
        super().__init__()
        self.mm_media_token_ids = media_token_ids

    @staticmethod
    def _feature(items):
        return torch.cat([item.feature for item in items])

    def get_image_feature(self, items):
        return self._feature(items)

    def get_video_feature(self, items):
        return self._feature(items)


def item(modality, start, end, seed):
    n = end - start + 1
    feature = 1000 + torch.arange(n * HIDDEN, dtype=torch.float32, device=DEVICE).reshape(n, HIDDEN) + 100 * seed
    data = MultimodalDataItem(modality=modality, offsets=[(start, end)], feature=feature)
    data.set_pad_value()
    return data


def request():
    image, video = item(Modality.IMAGE, 2, 4, 0), item(Modality.VIDEO, 6, 7, 1)
    ids = [1, 2, image.pad_value, image.pad_value, image.pad_value, 3, video.pad_value, video.pad_value, 4]
    return MultimodalInputs(mm_items=[image, video]), ids, image, video


def run(model, prefix):
    mm_inputs, ids, image, video = request()
    input_ids = torch.tensor(ids[prefix:], dtype=torch.int64, device=DEVICE)
    embedding = nn.Embedding(VOCAB, HIDDEN, device=DEVICE)
    embeds, _ = embed_mm_inputs(
        mm_inputs_list=[mm_inputs],
        extend_prefix_lens=[prefix],
        extend_seq_lens=[len(ids) - prefix],
        input_ids=input_ids,
        input_embedding=embedding,
        multimodal_model=model,
    )
    return input_ids, embeds, embedding, image, video


@pytest.mark.parametrize("prefix", [0, 4])  # whole prompt; a chunk starting at the image's last pad
def test_placeholder_ids_written_in_place(prefix):
    model = Model({Modality.IMAGE: IMAGE, Modality.VIDEO: VIDEO})
    input_ids, embeds, embedding, image, video = run(model, prefix)
    want = [1, 2, IMAGE, IMAGE, IMAGE, 3, VIDEO, VIDEO, 4][prefix:]
    assert input_ids.tolist() == want
    # the media rows still carry the media features, the text rows their token embeddings
    media = torch.tensor([i in (IMAGE, VIDEO) for i in want], device=DEVICE)
    assert torch.equal(embeds[~media], embedding(input_ids[~media]))
    rows = torch.cat([image.feature[prefix - 2 :] if prefix else image.feature, video.feature])
    assert torch.equal(embeds[media], rows)


def test_default_clamps_to_last_vocab_id():
    input_ids, _, _, _, _ = run(Model(None), 0)
    assert input_ids.tolist() == [1, 2, VOCAB - 1, VOCAB - 1, VOCAB - 1, 3, VOCAB - 1, VOCAB - 1, 4]


def test_undeclared_modality_raises():
    with pytest.raises(NotImplementedError, match="VIDEO"):
        run(Model({Modality.IMAGE: IMAGE}), 0)


def test_write_skips_absent_modalities_and_checks_mask_shape():
    ids = torch.tensor([5, 1_000_007, 6])
    write_mm_media_token_ids_(
        ids, [Modality.IMAGE, Modality.VIDEO], [None, torch.tensor([[False], [True], [False]])],
        {Modality.VIDEO: VIDEO},
    )
    assert ids.tolist() == [5, VIDEO, 6]
    with pytest.raises(RuntimeError, match="media mask"):
        write_mm_media_token_ids_(ids, [Modality.VIDEO], [torch.tensor([True, False, False])], {Modality.VIDEO: VIDEO})
