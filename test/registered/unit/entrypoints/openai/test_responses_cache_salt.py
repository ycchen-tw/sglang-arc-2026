"""A /v1/responses request's cache namespace reaches the scheduler's Req.

``cache_salt`` / ``extra_key`` are top-level body fields of ResponsesRequest,
forwarded into the GenerateReqInput the TokenizerManager receives, normalized
there, and carried across the tokenizer -> scheduler msgpack IPC in
TokenizedGenerateReqInput. ``prompt_cache_key`` -- what codex sends, its thread
id -- is not one of them: pydantic drops it and the request stays unsalted.
CPU only; the radix side is test_cache_salt_hybrid_mamba.py.
"""

from utils import make_serving  # stubs CUDA-only deps; import first

import asyncio
import unittest
from array import array

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _adapted_request(body: dict) -> GenerateReqInput:
    """Run create_responses on a JSON body; return the GenerateReqInput it hands
    to the TokenizerManager."""
    serving = make_serving()
    serving.default_chat_template_kwargs = {}
    serving.template_manager.chat_template_name = None
    captured = {}

    async def fake_generate(
        request_id, request_prompt, adapted_request, sampling_params, context, **kw
    ):
        captured["adapted_request"] = adapted_request
        context.append_output(
            {
                "text": "ok",
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                },
            }
        )
        yield context

    serving._generate_with_builtin_tools = fake_generate
    asyncio.run(serving.create_responses(ResponsesRequest.model_validate(body)))
    return captured["adapted_request"]


class TestResponsesCacheSalt(CustomTestCase):
    BODY = {"model": "x", "input": "hi", "store": False}

    def test_cache_salt_and_session_id_reach_the_generate_request(self):
        adapted = _adapted_request(
            {**self.BODY, "cache_salt": "ep-7", "session_id": "thread:0"}
        )
        self.assertEqual(adapted.cache_salt, "ep-7")
        self.assertIsNone(adapted.extra_key)
        self.assertEqual(adapted.session_id, "thread:0")

    def test_extra_key_reaches_the_generate_request(self):
        adapted = _adapted_request({**self.BODY, "extra_key": "ep-7"})
        self.assertEqual(adapted.extra_key, "ep-7")
        self.assertIsNone(adapted.cache_salt)

    def test_prompt_cache_key_is_not_a_cache_namespace(self):
        adapted = _adapted_request({**self.BODY, "prompt_cache_key": "thread-uuid"})
        self.assertIsNone(adapted.cache_salt)
        self.assertIsNone(adapted.extra_key)

    def test_namespace_survives_normalization_and_scheduler_ipc(self):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3], sampling_params={}, cache_salt="ep-7", extra_key=""
        )
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.cache_salt, "ep-7")
        self.assertIsNone(obj.extra_key)  # "" means no namespace

        # tokenizer_manager._tokenize_one_request copies both fields verbatim
        tokenized = TokenizedGenerateReqInput(
            rid="r1",
            input_text=None,
            input_ids=array("q", obj.input_ids),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_id="thread:0",
            extra_key=obj.extra_key,
            cache_salt=obj.cache_salt,
        )
        received = msgpack_decode(msgpack_encode(tokenized))
        self.assertEqual(received.cache_salt, "ep-7")
        self.assertIsNone(received.extra_key)

        # scheduler.handle_generate_request builds the Req from these fields
        req = Req(
            received.rid,
            received.input_text,
            received.input_ids,
            received.sampling_params,
            session_id=received.session_id,
            extra_key=received.extra_key,
            cache_salt=received.cache_salt,
        )
        self.assertEqual(req.cache_salt, "ep-7")
        self.assertIsNone(req.extra_key)


if __name__ == "__main__":
    unittest.main()
