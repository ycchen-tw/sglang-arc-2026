"""arc lazy media (ARC_MM_LAZY_MEDIA=1): a waiting request whose lazy items overlap its extend is parked in the
scheduler and filled by the tokenizer manager, not resent. CPU only.

RL line (TITO session ledger with forced logprobs / routed experts, --session-replay): the request the scheduler runs
after the fill is the one the ledger's intake made -- replayed ids, forced flags, its media offsets as the replay moved
them -- and the intake ran once; a lazy item asked for twice and a failed fill abort the request in the scheduler.
(The session park claim held while parked: test_arc_lazy_media_park, on trees with --session-exact-resume.)

Kaggle line (no ledger, no cache_salt; QSA HiSparse session cache, whose h14 replay rewrites ids in the scheduler at
arrival): the fill is the tokenizer's processing of the request (what the old resend sent), the request keeps its
h14-replayed ids and its session ownership while parked, and the request that finally runs equals the one the old
resend produced. Data parallelism routes the fill back to the parking rank; a request parked for its media (either
kind) keeps the scheduler from being idle.
"""

import asyncio
import shutil
import tempfile
from array import array
from collections import Counter
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers import session_ledger as sl  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    ArcMmRematerializeReq,
    GenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager  # noqa: E402
from sglang.srt.mem_cache import qsa_hisparse_session_cache as sc  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.multimodal import arc_lazy_media  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

IMG_LEN = 4  # pad tokens per image

# A toy tokenizer of Qwen3.8's shape (as managers/replay_own_turn_fixtures, kept here so that this file runs on every
# tree with lazy media): every id is one string, encode is a greedy longest match.
VOCAB = {i: chr(ord("a") + i) for i in range(26)}
VOCAB.update({30: " ", 31: "\n", 32: "\n\n", 33: "ab", 40: "<think>", 41: "</think>", 42: "<|im_start|>",
              43: "<|im_end|>", 44: "assistant", 47: "<|endoftext|>", 48: "user", 49: "\ufffd",
              55: "<|vision_start|>", 56: "<|image_pad|>", 57: "<|vision_end|>"})
TOKEN = {v: k for k, v in VOCAB.items()}
A, B, AB, END, EOT = TOKEN["a"], TOKEN["b"], TOKEN["ab"], TOKEN["<|im_end|>"], TOKEN["<|endoftext|>"]
VS, PAD, VE = TOKEN["<|vision_start|>"], TOKEN["<|image_pad|>"], TOKEN["<|vision_end|>"]
EOS = [END, EOT]
BIG = 1 << 20


class ToyTokenizer:
    def convert_tokens_to_ids(self, name):
        return TOKEN.get(name)

    def convert_ids_to_tokens(self, t):
        return VOCAB.get(t)

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(VOCAB[int(t)] for t in ids)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        names = sorted(TOKEN, key=len, reverse=True)
        while i < len(text):
            name = next(n for n in names if text.startswith(n, i))
            out.append(TOKEN[name])
            i += len(name)
        return out


TOK = ToyTokenizer()


def enc(text):
    return TOK.encode(text)


# ----------------------------------------------------------------------------- items


def _lazy(h, start):
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(start, start + IMG_LEN - 1)])
    item.model_specific_data[arc_lazy_media.LAZY_KEY] = True
    item.set_hash(h)
    return item


def _with_pixels(h, start):
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(start, start + IMG_LEN - 1)])
    item.feature = torch.full((IMG_LEN, 2), float(h))
    item.set_hash(h)
    return item


def _shape(items):
    """What the encoder sees of a request's items: (hash, pad value, offsets, lazy, feature)."""
    return [
        (i.hash, i.pad_value, list(i.offsets), arc_lazy_media.LAZY_KEY in i.model_specific_data,
         None if i.feature is None else i.feature.tolist())
        for i in items
    ]


# ----------------------------------------------------------------------------- the two processes, bare


def bare_scheduler(tree_cache):
    """The Scheduler state the lazy media park / fill / abort / idle paths touch."""
    s = Scheduler.__new__(Scheduler)
    s.tree_cache = tree_cache
    s.waiting_queue = []
    s.arc_mm_lazy_parked, s.arc_mm_refill_parked, s.arc_mm_refill_aborted = {}, {}, set()
    s.arc_lazy_sent_back_total = s.arc_mm_refill_total = 0
    s.beam_coordinator = mock.MagicMock()
    s.to_tokenizer = []
    s.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=lambda obj, req: s.to_tokenizer.append(obj))
    )
    # abort_request / is_fully_idle
    s.chunked_req = None
    s.mm_receiver = None
    s.dllm_config = None
    s.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
    s.grammar_manager = mock.MagicMock(grammar_queue=[])
    s.disaggregation_mode = DisaggregationMode.NULL
    s.collect_inflight_reqs = lambda: []
    s.hisparse_coordinator = None
    s.running_batch = SimpleNamespace(is_empty=lambda: True)
    s.last_batch = None
    s.enable_overlap = False
    s.ps = SimpleNamespace(pp_size=1, dp_size=1, dp_rank=None)
    s.enable_hisparse = False
    s.enable_hierarchical_cache = False
    s._engine_paused = False
    return s


def fill_message(fresh_tok):
    """What the scheduler receives of the tokenizer's fill (its mm_inputs as _get_multimodal_inputs makes them)."""
    mm = fresh_tok.mm_inputs
    if isinstance(mm, MultimodalProcessorOutput):
        mm = MultimodalInputs(mm_items=list(mm.mm_items))
    return SimpleNamespace(rid=fresh_tok.rid, mm_inputs=mm, arc_mm_refill=fresh_tok.arc_mm_refill)


class BareTokenizerManager:
    """TokenizerManager's fill path on a bare instance: _tokenize_one_request gives what the test's client
    tokenization gives (a new object each call), _send_one_request / _dispatch_to_scheduler record."""

    def __init__(self, ledger, tokenize):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.rid_to_state = {}
        tm.session_ledger = ledger
        tm.max_req_input_len = BIG
        tm._arc_mm_resend_tasks = set()
        self.tm, self.sent, self.dispatched, self.tokenized = tm, [], [], []

        async def tokenize_one(obj):
            self.tokenized.append(getattr(obj, arc_lazy_media.MATERIALIZE_ATTR, None))
            return tokenize(obj)

        async def send_one(tok):
            self.sent.append(tok)

        tm._tokenize_one_request = tokenize_one
        tm._send_one_request = send_one
        tm._dispatch_to_scheduler = self.dispatched.append

    def arrive(self, obj, tok):
        """generate_request: the request's state, then the ledger's intake on its tokenization."""
        self.tm.rid_to_state[obj.rid] = ReqState([], False, asyncio.Event(), obj, None)
        TokenizerManager._session_ledger_intake(self.tm, obj, tok)

    def deliver(self, msg):
        """The scheduler's ArcMmRematerializeReq arrives; the fill it starts runs to its end."""

        async def go():
            self.tm._handle_arc_mm_rematerialize(msg)
            await asyncio.gather(*list(self.tm._arc_mm_resend_tasks))

        asyncio.run(go())


def request(rid, session_id, cache_salt, n_images):
    """The tokenizer manager's request object (normalized as generate_request leaves a single request)."""
    obj = GenerateReqInput(input_ids=[1, 2, 3], image_data=[f"<png {i}>" for i in range(n_images)], rid=rid,
                           session_id=session_id, cache_salt=cache_salt)
    obj.normalize_batch_and_arguments()
    return obj


def req_from(tok):
    """handle_generate_request's Req for a tokenized request (the fields a fill must not change)."""
    req = Req(
        rid=tok.rid,
        origin_input_text="",
        origin_input_ids=array("q", tok.input_ids),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
        return_logprob=tok.return_logprob,
        return_routed_experts=tok.return_routed_experts,
        routed_experts_start_len=tok.routed_experts_start_len,
        session_id=tok.session_id,
        cache_salt=tok.cache_salt,
    )
    req.logprob_start_len = (
        len(req.origin_input_ids) if tok.return_logprob and tok.logprob_start_len == -1 else tok.logprob_start_len
    )
    mm = tok.mm_inputs
    req.multimodal_inputs = MultimodalInputs(mm_items=[_copy_item(i) for i in mm.mm_items])
    return req


def _copy_item(item):
    """The scheduler's own copy of an item (it arrived through IPC)."""
    new = MultimodalDataItem(modality=item.modality, offsets=list(item.offsets))
    new.model_specific_data = dict(item.model_specific_data)
    new.feature = item.feature
    new.set_hash(item.hash)
    return new


def _server_args():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


# ----------------------------------------------------------------------------- RL: the TITO ledger


HEAD_TEXT = "<|im_start|>user\nfix it<|im_end|>\n<|im_start|>assistant\n<think>\n"


def client_turn2(rid, salt):
    """The client's turn 2 as the tokenizer manager tokenizes it: the model's earlier answer "ab" re-rendered (one id
    where the model sampled two), then the user's message with a screenshot (expanded to its pad run) seen before."""
    ids = enc(HEAD_TEXT) + [AB, END] + enc("\n<|im_start|>user\nok")
    image_start = len(ids) + 1
    ids += [VS] + [PAD] * IMG_LEN + [VE] + enc("<|im_end|>\n<|im_start|>assistant\n<think>\n")
    mm = MultimodalProcessorOutput(
        mm_items=[_lazy(77, image_start)], input_ids=list(ids), im_token_id=PAD, im_start_id=VS, im_end_id=VE
    )
    return SimpleNamespace(
        rid=rid, session_id="s", input_ids=array("q", ids), input_embeds=None, mm_inputs=mm,
        positional_embed_overrides=None, token_type_ids=None, multi_item_delimiter_indices=None,
        return_routed_experts=False, routed_experts_start_len=0, return_logprob=False, logprob_start_len=-1,
        need_wait_for_mm_inputs=None, sampling_params=SamplingParams(max_new_tokens=16), cache_salt=salt,
        arc_mm_refill=False,
    ), image_start


class LedgerIntakeRunsOnce(CustomTestCase):
    def setUp(self):
        _server_args()
        self.dir = tempfile.mkdtemp(prefix="arc-lazy-ledger-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.ledger = sl.SessionLedger(
            tokenizer=TOK, eos_token_ids=EOS, ledger_dir=self.dir, record_all=True, logprobs=True, routing=True,
            routing_server_enabled=True, max_bytes=1 << 20, tokenizer_worker_num=1, server_meta={}, port=0,
        )
        self.addCleanup(self.ledger.shutdown)
        # turn 1 commits S: the prompt and what the model sampled, "a" "b" as two ids
        obj1 = SimpleNamespace(rid="t1", session_id="s", image_data=None, video_data=None)
        tok1 = SimpleNamespace(
            rid="t1", session_id="s", input_ids=array("q", enc(HEAD_TEXT)), input_embeds=None, mm_inputs=None,
            positional_embed_overrides=None, token_type_ids=None, multi_item_delimiter_indices=None,
            return_routed_experts=False, routed_experts_start_len=0, return_logprob=False, logprob_start_len=-1,
            need_wait_for_mm_inputs=None, sampling_params=SamplingParams(max_new_tokens=16),
        )
        self.ledger.on_intake(obj1, tok1, BIG)
        self.ledger.on_finish("t1", SimpleNamespace(output_ids=[A, B, END], output_token_logprobs_val=[-0.5] * 3), {},
                              {"type": "stop", "matched": END})
        self.S = [int(t) for t in self.ledger.sessions["s"].S]
        self.assertEqual(self.S, enc(HEAD_TEXT) + [A, B, END])

    def test_the_parked_request_is_the_one_the_intake_made(self):
        rid = "t2"
        obj = request(rid, "s", "p/ep-1", 1)
        tm = BareTokenizerManager(self.ledger, lambda o: client_turn2(rid, "p/ep-1")[0])
        tok, client_image = client_turn2(rid, "p/ep-1")
        client_ids = list(tok.input_ids)
        with mock.patch.object(self.ledger, "on_intake", wraps=self.ledger.on_intake) as intake:
            tm.arrive(obj, tok)
            # the intake: the model's own ids for "ab" (one id longer), the image moved with them, logprobs and
            # routed experts forced for the ledger
            replayed = self.S + client_ids[len(enc(HEAD_TEXT)) + 2 :]
            self.assertEqual(list(tok.input_ids), replayed)
            self.assertEqual(tok.mm_inputs.mm_items[0].offsets, [(client_image + 1, client_image + IMG_LEN)])
            self.assertTrue(tok.return_logprob and tok.return_routed_experts)
            self.assertEqual(tok.routed_experts_start_len, len(self.S) - 1)
            pending = self.ledger.pending[rid]
            counters = Counter(self.ledger.counters)
            sessions = {sid: (len(x.S), x.version) for sid, x in self.ledger.sessions.items()}

            # the scheduler: the request as it arrived, its lazy screenshot in the extend -> parked
            s = bare_scheduler(mock.MagicMock())
            req = req_from(tok)
            before = (list(req.origin_input_ids), req.return_logprob, req.logprob_start_len,
                      req.return_routed_experts, req.routed_experts_start_len, req.session_id, req.cache_salt)
            s.waiting_queue.append(req)
            needed = s._arc_lazy_needed(req)
            self.assertEqual(needed, [req.multimodal_inputs.mm_items[0].hash])
            s._arc_lazy_send_back([(req, needed)])

            # the tokenizer manager: a fill, no second intake
            tm.deliver(s.to_tokenizer[-1])
            self.assertEqual(intake.call_count, 1)
        self.assertEqual(tm.tokenized, [frozenset(needed)])
        (fill,) = tm.sent
        self.assertTrue(fill.arc_mm_refill)
        self.assertEqual(list(fill.input_ids), client_ids)  # the client's ids: the scheduler does not take them
        self.assertIs(self.ledger.pending[rid], pending)
        self.assertEqual(list(pending.key), replayed)
        self.assertEqual(Counter(self.ledger.counters), counters)
        self.assertEqual({sid: (len(x.S), x.version) for sid, x in self.ledger.sessions.items()}, sessions)

        # the fill: the screenshot's feature, at the offsets the replay gave it
        fill.mm_inputs.mm_items[0] = _with_pixels(req.multimodal_inputs.mm_items[0].hash, client_image)
        s._arc_mm_refill(fill_message(fill), None)
        self.assertEqual(s.waiting_queue, [req])
        after = (list(req.origin_input_ids), req.return_logprob, req.logprob_start_len,
                 req.return_routed_experts, req.routed_experts_start_len, req.session_id, req.cache_salt)
        self.assertEqual(after, before)
        (item,) = req.multimodal_inputs.mm_items
        self.assertEqual(item.offsets, [(client_image + 1, client_image + IMG_LEN)])
        self.assertIsNotNone(item.feature)
        self.assertEqual(s._arc_lazy_needed(req), [])

    def test_with_data_parallelism_the_fill_goes_back_to_the_parking_rank(self):
        rid = "t2"
        obj = request(rid, "s", None, 1)
        tm = BareTokenizerManager(self.ledger, lambda o: client_turn2(rid, None)[0])
        tok, _ = client_turn2(rid, None)
        tok.routed_dp_rank = None
        tm.arrive(obj, tok)
        s = bare_scheduler(mock.MagicMock())
        s.ps = SimpleNamespace(pp_size=1, dp_size=2, dp_rank=1)
        req = req_from(tok)
        s._arc_lazy_send_back([(req, s._arc_lazy_needed(req))])
        self.assertEqual(s.to_tokenizer[-1].dp_rank, 1)
        tm.deliver(s.to_tokenizer[-1])
        (fill,) = tm.sent
        self.assertEqual(fill.routed_dp_rank, 1)  # the DP controller routes it to rank 1, not round robin
        # a retracted request's refill too
        s.arc_mm_lazy_parked.clear()
        req.multimodal_inputs.mm_items[0].model_specific_data.pop(arc_lazy_media.LAZY_KEY)
        self.assertTrue(s._arc_mm_park_for_refill(req))
        self.assertEqual((s.to_tokenizer[-1].refill, s.to_tokenizer[-1].dp_rank), (True, 1))

    def test_a_lazy_item_asked_for_twice_aborts_in_the_scheduler(self):
        rid = "t2"
        obj = request(rid, "s", None, 1)
        tm = BareTokenizerManager(self.ledger, lambda o: client_turn2(rid, None)[0])
        tok, _ = client_turn2(rid, None)
        tm.arrive(obj, tok)
        msg = ArcMmRematerializeReq(rid=rid, hashes=[tok.mm_inputs.mm_items[0].hash])
        tm.deliver(msg)
        self.assertEqual(len(tm.sent), 1)
        tm.deliver(msg)
        self.assertEqual(len(tm.sent), 1)  # no second fill
        (abort,) = tm.dispatched
        self.assertIsInstance(abort, AbortReq)
        self.assertEqual(abort.rid, rid)
        self.assertIn("asked again", abort.finished_reason["message"])
        self.assertTrue(tm.tm.rid_to_state[rid].abort_sent)

    def test_a_failed_fill_aborts_in_the_scheduler(self):
        rid = "t2"
        obj = request(rid, "s", None, 1)

        def broken(o):
            raise ValueError("decoder said no")

        tm = BareTokenizerManager(self.ledger, broken)
        tok, _ = client_turn2(rid, None)
        tm.arrive(obj, tok)
        tm.deliver(ArcMmRematerializeReq(rid=rid, hashes=[tok.mm_inputs.mm_items[0].hash]))
        (abort,) = tm.dispatched
        self.assertIn("fill of a waiting request failed", abort.finished_reason["message"])
        self.assertEqual(tm.sent, [])


# ----------------------------------------------------------------------------- Kaggle


def kaggle_turn(rid):
    """A Kaggle request (no salt, no ledger): an old frame inside the session's prefix and this turn's frame in the
    extend, both seen before (lazy), and a picture never seen (with its pixels) in the extend too; the client's ids as
    the tokenizer manager makes them."""
    image = [VS] + [PAD] * IMG_LEN + [VE]
    ids = (list(range(1000, 1040)) + image + list(range(2000, 2030)) + image + list(range(3000, 3005)) + image
           + list(range(3005, 3010)))
    old, new, unseen = 41, 77, 88
    assert ids[old - 1] == ids[new - 1] == ids[unseen - 1] == VS
    mm = MultimodalProcessorOutput(mm_items=[_lazy(55, old), _lazy(66, new), _with_pixels(88, unseen)],
                                   input_ids=list(ids), im_token_id=PAD, im_start_id=VS, im_end_id=VE)
    return SimpleNamespace(
        rid=rid, session_id="g", input_ids=array("q", ids), mm_inputs=mm, return_routed_experts=False,
        routed_experts_start_len=0, return_logprob=False, logprob_start_len=-1, cache_salt=None, arc_mm_refill=False,
    ), (old, new)


def materialized(tok, hashes):
    """The tokenizer's processing of the request with these items materialized (their pixels): what a fill is, and
    what the old resend was with every item of the extend asked for."""
    tok.mm_inputs.mm_items = [
        _with_pixels(i.hash, i.offsets[0][0]) if i.hash in hashes else i for i in tok.mm_inputs.mm_items
    ]
    return tok


class KaggleFillIsTheOldResend(CustomTestCase):
    def setUp(self):
        _server_args()

    def test_without_ledger_or_salt(self):
        rid = "k1"
        obj = request(rid, "g", None, 3)
        tm = BareTokenizerManager(None, lambda o: materialized(kaggle_turn(rid)[0], o.arc_mm_materialize_hashes))
        tok, (old, new) = kaggle_turn(rid)
        tm.arrive(obj, tok)  # no ledger: the request goes as tokenized
        self.assertEqual(list(tok.input_ids), list(kaggle_turn(rid)[0].input_ids))
        self.assertFalse(tok.return_logprob or tok.return_routed_experts)

        s = bare_scheduler(mock.MagicMock())
        req = req_from(tok)
        req.prefix_indices = torch.arange(old + IMG_LEN + 5)  # the session's prefix covers the old frame
        s.waiting_queue.append(req)
        needed = s._arc_lazy_needed(req)
        self.assertEqual(needed, [req.multimodal_inputs.mm_items[1].hash])
        s._arc_lazy_send_back([(req, needed)])
        tm.deliver(s.to_tokenizer[-1])
        (fill,) = tm.sent
        # the fill is the tokenizer's processing of the request with the lazy item asked for: the old resend's ids
        # and items (the old resend also re-processed the extend's item that had its pixels; the parked request keeps
        # that feature instead)
        self.assertEqual(tm.tokenized, [frozenset(needed)])
        resend = materialized(kaggle_turn(rid)[0], {66, 88})  # before: every item of the extend, lazy or not
        resend_ids = [i.hash for i in resend.mm_inputs.mm_items]
        self.assertEqual(list(fill.input_ids), list(resend.input_ids))
        self.assertEqual([i.hash for i in fill.mm_inputs.mm_items], resend_ids)
        self.assertEqual(_shape(fill.mm_inputs.mm_items)[1], _shape(resend.mm_inputs.mm_items)[1])
        s._arc_mm_refill(fill_message(fill), None)

        # the request that runs is the one the old resend made the scheduler build
        old_req = req_from(resend)
        self.assertEqual(s.waiting_queue, [req])
        self.assertEqual(list(req.origin_input_ids), list(old_req.origin_input_ids))
        self.assertEqual(_shape(req.multimodal_inputs.mm_items), _shape(old_req.multimodal_inputs.mm_items))
        self.assertEqual((req.return_logprob, req.logprob_start_len, req.return_routed_experts),
                         (old_req.return_logprob, old_req.logprob_start_len, old_req.return_routed_experts))
        self.assertEqual(s._arc_lazy_needed(req), [])


def qsa_session_cache(record, P):
    """A QSA HiSparse session cache (the Kaggle backend) holding one PARKED record of session "g": its h14 replay,
    session ownership and resume decisions are real; its pools are not."""
    cache = object.__new__(sc.QSAHiSparseSessionCache)
    cache.replay, cache.own_last_turn, cache.replay_own_turn = True, False, None
    cache.stats, cache._own_intake, cache._reqs, cache.owner = Counter(), {}, {}, {}
    cache.gen, cache.ledger_version, cache.page_size, cache.block_tokens = {"g": 1}, 0, 1, 1
    nothing = SimpleNamespace(free=lambda *a: None, free_buffer=lambda *a: None)
    cache.token_to_kv_pool_allocator = nothing
    cache.coord = SimpleNamespace(mem_pool_host=nothing, snapshots=nothing)
    cache.set_replay_tokenizer(TOK, EOS)
    rec = sc.SessionRecord("g", 1, None, None, P=P, committed=P, gdn=7, tokens=np.asarray(record, dtype=np.int64),
                           logical=torch.arange(P), host_blocks=torch.arange(P), prompt_len=len(enc(HEAD_TEXT)))
    cache.records, cache.index = [rec], {"g": rec}
    return cache, rec


class KaggleQsaSessionOwnershipIsKept(CustomTestCase):
    """h14 replays at the scheduler's arrival; the parked request keeps those ids and stays the session's owner,
    so the fill neither replays again nor re-claims; an abort while parked lapses the ownership."""

    def setUp(self):
        _server_args()
        self.record = enc(HEAD_TEXT) + [A, B, END]
        self.cache, self.rec = qsa_session_cache(self.record, P=len(self.record) - 1)

    def _arrive(self, rid):
        tok, image_start = client_turn2(rid, None)
        tok.session_id = "g"
        client_ids = list(tok.input_ids)
        self.cache.replay_request(tok, BIG)  # handle_generate_request, before the Req exists
        self.assertEqual(list(tok.input_ids), self.record + client_ids[len(enc(HEAD_TEXT)) + 2 :])
        req = req_from(tok)
        req.session_generation = 1
        return req, client_ids, image_start

    def _match(self, req):
        return self.cache.match_prefix(MatchPrefixParams(
            key=RadixKey(token_ids=req.origin_input_ids, extra_key=None, limit=len(req.origin_input_ids) - 1,
                         cache_salt=None),
            req=req, cow_mamba=False))

    def test_parked_owner_resumes_with_its_replayed_ids(self):
        req, client_ids, image_start = self._arrive("k2")
        self.assertEqual(self._match(req).cache_protected_len, self.rec.P)
        s = bare_scheduler(self.cache)
        s.waiting_queue.append(req)
        needed = s._arc_lazy_needed(req)
        self.assertEqual(needed, [req.multimodal_inputs.mm_items[0].hash])
        s._arc_lazy_send_back([(req, needed)])
        self.assertIs(self.cache.owner["g"], req)  # still the owner while parked
        self.assertIs(self.cache._reqs["k2"].req, req)
        self.assertFalse(s.is_fully_idle())  # check_idle_invariants would fail on the live owner
        with self.assertRaisesRegex(RuntimeError, "live session requests"):
            self.cache.check_idle_invariants()
        replays = self.cache.stats["replay"]

        fresh, _ = client_turn2("k2", None)
        fresh.session_id = "g"
        fresh.mm_inputs.mm_items[0] = _with_pixels(needed[0], image_start)
        fresh.arc_mm_refill = True
        s._arc_mm_refill(fill_message(fresh), None)
        self.assertEqual(list(req.origin_input_ids), self.record + client_ids[len(enc(HEAD_TEXT)) + 2 :])
        self.assertEqual(req.multimodal_inputs.mm_items[0].offsets, [(image_start + 1, image_start + IMG_LEN)])
        self.assertEqual(self.cache.stats["replay"], replays)  # no second h14 replay
        # the next round: the same live request, resumed at P
        self.assertEqual(self._match(req).cache_protected_len, self.rec.P)
        self.assertIs(self.cache.owner["g"], req)
        self.assertEqual(s._arc_lazy_needed(req), [])

    def test_abort_while_parked_lapses_the_ownership(self):
        req, _, _ = self._arrive("k3")
        self._match(req)
        s = bare_scheduler(self.cache)
        s.waiting_queue.append(req)
        s._arc_lazy_send_back([(req, s._arc_lazy_needed(req))])
        s.abort_request(AbortReq(rid="k3"))
        self.assertEqual((self.cache.owner, self.cache._reqs), ({}, {}))
        self.assertEqual(self.rec.state, "PARKED")
        self.assertEqual(self.cache.stats["abort_waiting_with_record"], 1)
        self.assertTrue(s.is_fully_idle())


class RefillParkedIsNotIdle(CustomTestCase):
    """A retracted request parked for a refill stays the QSA session's owner (cache_finished_req: "a retracted
    request stays the owner"): before the fix is_fully_idle ignored it and the idle check raised."""

    def test_refill_parked_is_not_idle(self):
        _server_args()
        s = bare_scheduler(mock.MagicMock())
        self.assertTrue(s.is_fully_idle())
        s.arc_mm_refill_parked["r"] = object()
        self.assertFalse(s.is_fully_idle())
        self.assertTrue(s.is_fully_idle(for_health_check=True))
