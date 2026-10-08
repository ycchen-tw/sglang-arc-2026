"""ARC: the DFlash drafter reads an extend batch's ids before the target forward rewrites its media positions.

The drafter is text-only and was trained with <|image_pad|> at every image and video pad (arc-forge
capture_tito.drafter_ids). In serving each media position holds its out-of-vocab radix pad value, which
DFlashDraftWorker.prefill_draft_ids maps to <|image_pad|>. embed_mm_inputs rewrites those positions in place
(vocab_size - 1, or the modality placeholders when the model declares mm_media_token_ids) in the tensor model.forward
receives. That tensor is batch.input_ids itself when the eager runner does not copy its inputs
(SGLANG_EAGER_INPUT_NO_COPY, --enable-pdmux) or under a full prefill CUDA graph, so the drafter's ids are taken before
the forward and handed to _draft_extend_for_prefill.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import contextlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from sglang.srt.managers.mm_schedule import init_mm_embedding_cache  # noqa: E402
from sglang.srt.managers.mm_utils import embed_mm_inputs  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative import eagle_worker_v2  # noqa: E402
from sglang.srt.speculative.arc_dflash.worker import DFlashDraftWorker  # noqa: E402

VOCAB, HIDDEN, IMAGE, VIDEO, TIMESTAMP = 100, 4, 90, 91, 3
# embed_mm_inputs stages the placeholder ids in pinned host memory
DEVICE = "cuda"
init_mm_embedding_cache(0)  # every item is a miss, encoded whole and sliced to the chunk


class Model(nn.Module):
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


def item(modality, start, end):
    n = end - start + 1
    data = MultimodalDataItem(
        modality=modality, offsets=[(start, end)], feature=torch.zeros(n, HIDDEN, device=DEVICE)
    )
    data.set_pad_value()
    return data


def request():
    """A literal <|image_pad|> and the legal id vocab_size - 1 as text, an image, a timestamp, a video."""
    image, video = item(Modality.IMAGE, 2, 4), item(Modality.VIDEO, 6, 7)
    ids = [IMAGE, VOCAB - 1, *[image.pad_value] * 3, TIMESTAMP, *[video.pad_value] * 2, 4]
    return MultimodalInputs(mm_items=[image, video]), ids


# the drafter's training view: every media pad is <|image_pad|>, text (literal markers included) as is
DRAFT = [IMAGE, VOCAB - 1, IMAGE, IMAGE, IMAGE, TIMESTAMP, IMAGE, IMAGE, 4]
# what the target's mm embed leaves in the tensor it was given
TARGET = {
    False: [IMAGE, VOCAB - 1, VOCAB - 1, VOCAB - 1, VOCAB - 1, TIMESTAMP, VOCAB - 1, VOCAB - 1, 4],
    True: [IMAGE, VOCAB - 1, IMAGE, IMAGE, IMAGE, TIMESTAMP, VIDEO, VIDEO, 4],
}


def drafter():
    """prefill_draft_ids reads vocab and image_token; the prefill contract check reads nothing else."""
    worker = object.__new__(DFlashDraftWorker)
    worker.vocab, worker.image_token = VOCAB, IMAGE
    return worker


def target_embed(mm_inputs, ids, input_ids, prefix, placeholder_ids):
    model = Model({Modality.IMAGE: IMAGE, Modality.VIDEO: VIDEO} if placeholder_ids else None)
    embed_mm_inputs(
        mm_inputs_list=[mm_inputs],
        extend_prefix_lens=[prefix],
        extend_seq_lens=[len(ids) - prefix],
        input_ids=input_ids,
        input_embedding=nn.Embedding(VOCAB, HIDDEN, device=DEVICE),
        multimodal_model=model,
    )


@pytest.mark.parametrize("placeholder_ids", [False, True])  # --ple-mm-placeholder-ids off / on
@pytest.mark.parametrize("prefix", [0, 3])  # whole prompt; a chunk starting inside the image
def test_draft_ids_taken_before_the_target_rewrite(placeholder_ids, prefix):
    mm_inputs, ids = request()
    batch = SimpleNamespace(input_ids=torch.tensor(ids[prefix:], dtype=torch.int64, device=DEVICE))
    draft = drafter().prefill_draft_ids(batch)
    # the target's forward rewrites the batch's own tensor (no input copy)
    target_embed(mm_inputs, ids, batch.input_ids, prefix, placeholder_ids)
    assert batch.input_ids.tolist() == TARGET[placeholder_ids][prefix:]
    assert draft.tolist() == DRAFT[prefix:]
    # read after the forward, the media positions are in vocab and keep the target's ids
    assert drafter().prefill_draft_ids(batch).tolist() == TARGET[placeholder_ids][prefix:]


def test_prefill_requires_ids_taken_before_the_forward():
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, input_ids=torch.zeros(5, dtype=torch.int64, device=DEVICE)
    )
    with pytest.raises(TypeError, match="draft_ids"):
        drafter()._draft_extend_for_prefill(batch, None, None)
    for draft_ids in (None, torch.zeros(4, dtype=torch.int64, device=DEVICE)):
        with pytest.raises(RuntimeError, match="prefill_draft_ids"):
            drafter()._draft_extend_for_prefill(batch, None, None, draft_ids=draft_ids)


def test_eagle_worker_v2_takes_draft_ids_before_the_target_forward(monkeypatch):
    """EAGLEWorkerV2's extend branch with DFlash: the target forward rewrites batch.input_ids in place, the drafter
    still receives the pre-forward view."""
    for name in ("speculative_moe_backend_context", "speculative_moe_a2a_backend_context"):
        monkeypatch.setattr(eagle_worker_v2, name, contextlib.nullcontext)
    monkeypatch.setattr(eagle_worker_v2, "spec_stage_span", lambda name: contextlib.nullcontext())
    mm_inputs, ids = request()
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=True,
        input_ids=torch.tensor(ids, dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([len(ids)], device=DEVICE),
    )
    received = {}

    class Target:
        def forward_batch_generation(self, batch, **_):
            target_embed(mm_inputs, ids, batch.input_ids, 0, placeholder_ids=False)
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=None, mm_input_embeds=None), next_token_ids=None
            )

    class Drafter:
        prefill_draft_ids = DFlashDraftWorker.prefill_draft_ids
        vocab, image_token = VOCAB, IMAGE

        def _draft_extend_for_prefill(self, batch, hidden, next_ids, mm_input_embeds, *, draft_ids):
            received["batch"], received["draft"] = batch.input_ids.tolist(), draft_ids.tolist()
            return "next draft input"

    drafter_stub = Drafter()
    worker = SimpleNamespace(
        _dflash=True,
        _draft_worker=drafter_stub,
        draft_worker=drafter_stub,
        target_worker=Target(),
        speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
        _draft_tp_context=contextlib.nullcontext,
    )
    out = eagle_worker_v2.EAGLEWorkerV2.forward_batch_generation(worker, batch)
    assert out.next_draft_input == "next draft input"
    assert received["batch"] == TARGET[False]
    assert received["draft"] == DRAFT
