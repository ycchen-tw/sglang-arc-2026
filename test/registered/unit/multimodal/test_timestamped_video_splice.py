"""ARC: a pre-tokenized prompt that carries a video keeps its own ids.

Upstream SGLANG_MM_AVOID_RETOKENIZE keeps a pre-tokenized prompt's ids only when it has images and nothing else; with a
video the HF processor's re-tokenization of the decoded prompt was kept, so a non-canonical prompt drifted silently.
``splice_timestamped_media_blocks`` keeps the client's ids and takes only the expanded media blocks from the processor
(the Qwen3-VL video format: timestamped frame groups inside an outer <|vision_start|> ... <|vision_end|>).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
    splice_timestamped_media_blocks,
    timestamped_media_blocks,
)

IMG, VID, VS, VE = 9, 10, 7, 8
# Client: "ab" as two tokens 1 2 (non-canonical), an image, text 4, a video, text 5.
CLIENT = [1, 2, VS, IMG, VE, 4, VS, VID, VE, 5]
# Processor: "ab" re-tokenized to 12; image 3 pads; video of two frame groups with timestamps 30 31 / 32 33.
PROCESSOR = [12, VS, IMG, IMG, IMG, VE, 4, VS, 30, 31, VS, VID, VID, VE, 32, 33, VS, VID, VID, VE, VE, 5]
EXPANDED = [1, 2, VS, IMG, IMG, IMG, VE, 4, VS, 30, 31, VS, VID, VID, VE, 32, 33, VS, VID, VID, VE, VE, 5]


def splice(client, processor):
    return splice_timestamped_media_blocks(client, processor, IMG, VID, VS, VE)


def test_blocks():
    assert timestamped_media_blocks(PROCESSOR, IMG, VID, VS, VE) == [("image", 2, 4), ("video", 7, 20)]


def test_splice_keeps_client_ids_and_takes_expanded_blocks():
    assert splice(CLIENT, PROCESSOR) == EXPANDED
    # Already canonical text: identical to the processor's ids.
    canonical = [12] + CLIENT[2:]
    assert splice(canonical, PROCESSOR) == PROCESSOR


def test_two_videos_and_back_to_back_media():
    processor = [VS, 30, VS, VID, VE, VE, VS, 31, VS, VID, VE, 32, VS, VID, VE, VE, VS, IMG, VE]
    client = [VS, VID, VE, VS, VID, VE, VS, IMG, VE]
    assert timestamped_media_blocks(processor, IMG, VID, VS, VE) == [
        ("video", 0, 5), ("video", 6, 15), ("image", 17, 17)]
    assert splice(client, processor) == processor


@pytest.mark.parametrize("client,match", [
    (CLIENT[:3] + CLIENT[5:], "image / 1 video placeholders"),      # image placeholder missing
    (CLIENT + [VS, IMG, VE], "more image placeholders"),
    ([1, 2, VS, IMG, VE, 4, VID, VE, 5], "not <|vision_start|><|video_pad|><|vision_end|>"),
    ([1, 2, VS, IMG, VE, 4, VS, VID, VID, VE, 5], "not <|vision_start|><|video_pad|><|vision_end|>"),
])
def test_splice_rejects_prompts_that_do_not_match(client, match):
    with pytest.raises(ValueError, match=match):
        splice(client, PROCESSOR)


@pytest.mark.parametrize("processor,match", [
    ([VS, 30, VS, VID, VE], "never closes"),
    ([VID, VE, VE], "not wrapped"),
    ([VS, 30, VS, VID, VE, VS, IMG, VE, VE], "inside the video"),
    ([3] * 40 + [VS, VID, VE, VE], "no timestamp-delimited"),
    ([VE, 31, VS, VID, VE, VE], "outside any video"),
])
def test_blocks_reject_other_layouts(processor, match):
    with pytest.raises(ValueError, match=match):
        timestamped_media_blocks(processor, IMG, VID, VS, VE)


class _Stub(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _stub(splices):
    stub = _Stub.__new__(_Stub)
    stub._processor = SimpleNamespace(tokenizer=None)
    stub._tokenizer = None
    stub.splices_timestamped_video_blocks = splices
    stub.vision_start_token_id, stub.vision_end_token_id = VS, VE
    stub.use_cuda_ipc = False
    stub._process_and_collect_mm_items = lambda **kwargs: ([], torch.tensor(PROCESSOR), {})
    return stub


@pytest.mark.parametrize("splices,expected", [(True, EXPANDED), (False, PROCESSOR)])
def test_process_and_combine_takes_the_video_branch(splices, expected):
    out = BaseMultiModalProcessorOutput(input_text="decoded", input_ids=list(CLIENT), images=["img"], videos=["vid"])
    tokens = MultimodalSpecialTokens(image_token_id=IMG, video_token_id=VID)
    _, input_ids, _ = _stub(splices).process_and_combine_mm_data(out, tokens)
    assert input_ids.tolist() == expected
