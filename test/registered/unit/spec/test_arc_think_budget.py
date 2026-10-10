"""ARC_THINK_BUDGET (speculative/arc_think_budget.py): an over-budget thinking block is closed with exactly the text
the chat template renders for it on the next turn (``...<non-ws>\\n</think>``), once, under a lagging output_ids.

The verify loop here is a CPU stand-in for run_eagle_verify: a chain of K rows, greedy acceptance (a forced row is
one-hot, so greedy and rejection sampling commit the same token), and output_ids that see a step's tokens only after
the next step was launched (the overlap scheduler).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import os  # noqa: E402
import random  # noqa: E402
from array import array  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative import arc_think_budget as tb  # noqa: E402

TOKENIZER_DIR = os.environ.get("ARC_THINK_BUDGET_TEST_TOKENIZER")
# cuda: the verify-side tensors as the server has them (int32 draft tokens, fp32 logits over the model's padded vocab)
DEVICE = os.environ.get("ARC_THINK_BUDGET_TEST_DEVICE", "cpu")


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_DIR:
        pytest.skip("set ARC_THINK_BUDGET_TEST_TOKENIZER to a Qwen3.5+ checkpoint directory")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


@pytest.fixture(scope="module")
def budget_cls(tokenizer):
    state = tb.ThinkBudget(24, tokenizer)
    vocab = len(tokenizer)
    return state, tb.token_classes(tokenizer, vocab), vocab


def prompt_ids(tokenizer, **kw):
    text = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                         add_generation_prompt=True, **kw)
    return tokenizer.encode(text, add_special_tokens=False)


def make_req(rid, prompt, slot):
    """a real Req (the hook reads its attributes; a stand-in object would hide a renamed one)"""
    req = Req(rid, "", list(prompt), SamplingParams(max_new_tokens=1 << 20))
    req.kv.req_pool_idx = slot
    return req


def test_env_parsing(monkeypatch):
    monkeypatch.delenv(tb.ENV, raising=False)
    assert tb.budget_from_env() is None
    monkeypatch.setenv(tb.ENV, "32768")
    assert tb.budget_from_env() == 32768
    for bad in ("0", "-5", "1e4", "abc"):
        monkeypatch.setenv(tb.ENV, bad)
        with pytest.raises(RuntimeError):
            tb.budget_from_env()


def test_setup_refuses_paths_it_cannot_act_on(monkeypatch, tokenizer):
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    monkeypatch.setenv(tb.ENV, "100")
    with pytest.raises(RuntimeError):
        tb.setup(SpeculativeAlgorithm.NONE, 1, tokenizer)
    with pytest.raises(RuntimeError):
        tb.setup(SpeculativeAlgorithm.DFLASH, 1, tokenizer)
    with pytest.raises(RuntimeError):
        tb.setup(SpeculativeAlgorithm.EAGLE, 4, tokenizer)
    with pytest.raises(RuntimeError):
        tb.setup(SpeculativeAlgorithm.EAGLE, 1, None)
    assert tb.setup(SpeculativeAlgorithm.EAGLE, 1, tokenizer) is tb.active()
    monkeypatch.delenv(tb.ENV)
    assert tb.setup(SpeculativeAlgorithm.EAGLE, 1, tokenizer) is None and tb.active() is None


def test_token_classes(tokenizer, budget_cls):
    state, cls, _ = budget_cls
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    assert (state.start_id, state.end_id) == tuple(tokenizer.convert_tokens_to_ids(["<think>", "</think>"]))
    assert cls[state.nl_id] == tb.CLS_NL
    for ws in ("\n\n", " ", "  ", "\t"):
        ids = enc(ws)
        assert len(ids) == 1 and cls[ids[0]] == tb.CLS_WAIT, ws
    for word in ("the", " level", ".", "0"):
        assert cls[enc(word)[-1]] == tb.CLS_NORMAL, word


def test_open_at_prompt_end(tokenizer, budget_cls):
    state, _, _ = budget_cls
    on = prompt_ids(tokenizer)
    off = prompt_ids(tokenizer, enable_thinking=False)
    assert state._open_at_prompt_end(make_req("a", on, 1))
    assert not state._open_at_prompt_end(make_req("b", off, 2))


class Sim:
    """A chain verify loop over CPU tensors with output_ids lagging one step."""

    def __init__(self, state, cls, vocab, k, rng, lag=1):
        self.state, self.vocab, self.k, self.rng, self.lag = state, vocab, k, rng, lag
        self.cls = cls
        self.words = [i for i in range(1000, 60000) if cls[i] == tb.CLS_NORMAL][:2000]
        self.ws = [i for i in range(vocab) if cls[i] == tb.CLS_WAIT][:50]
        state.cls = torch.tensor(cls, dtype=torch.int8, device=DEVICE)
        state.closed = torch.zeros(64, dtype=torch.bool, device=DEVICE)

    def natural(self, prev, closed):
        """The 'model': keeps thinking (never closes on its own); after </think> it writes text."""
        r = self.rng.random()
        if r < 0.08:
            return self.state.nl_id
        if r < 0.16:
            return self.rng.choice(self.ws)
        return self.rng.choice(self.words)

    def run(self, reqs, steps):
        st, k = self.state, self.k
        pending = []  # committed tokens not yet visible in output_ids: [(req, [tokens])]
        committed = {r.rid: [] for r in reqs}
        last = {r.rid: r.origin_input_ids[-1] for r in reqs}
        for _ in range(steps):
            bs = len(reqs)
            cand = torch.zeros(bs, k, dtype=torch.int32)
            for i, r in enumerate(reqs):
                cand[i, 0] = last[r.rid]
                for j in range(1, k):  # drafts: sometimes the very tokens the close needs
                    x = self.rng.random()
                    cand[i, j] = (st.nl_id if x < 0.2 else st.end_id if x < 0.35
                                  else self.rng.choice(self.words))
            logits = torch.zeros(bs * k, self.vocab, device=DEVICE)
            # the natural distribution of each row: one dominant token
            closed_now = {r.rid: st.end_id in committed[r.rid] for r in reqs}
            nat = torch.zeros(bs, k, dtype=torch.int64)
            for i, r in enumerate(reqs):
                for j in range(k):
                    nat[i, j] = self.natural(int(cand[i, j]), closed_now[r.rid])
                    logits[i * k + j, nat[i, j]] += 100.0
            batch = SimpleNamespace(reqs=reqs, req_pool_indices=torch.tensor([r.kv.req_pool_idx for r in reqs],
                                                                             device=DEVICE))
            rows = st.select(batch)
            handle = None
            if rows:
                handle = st.apply(batch, rows, cand.flatten().to(DEVICE), logits, k, 64)
            pred = logits.view(bs, k, -1).argmax(-1).cpu()
            accept_index = torch.full((bs, k), -1, dtype=torch.int32)
            for i, r in enumerate(reqs):
                toks = []
                for j in range(k):
                    accept_index[i, j] = i * k + j
                    toks.append(int(pred[i, j]))
                    if j + 1 < k and int(cand[i, j + 1]) == int(pred[i, j]):
                        continue
                    break
                committed[r.rid].extend(toks)
                last[r.rid] = toks[-1]
                pending.append((r, toks))
            if handle is not None:
                st.commit(handle, pred.flatten().to(DEVICE, torch.int32), accept_index.to(DEVICE))
            # overlap: this step's tokens become visible after the next launch
            while len(pending) > self.lag * len(reqs):
                r, toks = pending.pop(0)
                r.output_ids.extend(toks)
        return committed


def round_trips(tokenizer, toks, end_id):
    """generated text == what the template renders from the parsed reasoning (the session replay criterion)"""
    assert toks.count(end_id) == 1, f"{toks.count(end_id)} </think> in the output"
    text = tokenizer.decode(toks, skip_special_tokens=False)
    reasoning, _, _ = text.partition("</think>")
    # the template renders "<think>\n" + reasoning|trim + "\n</think>"; the sim may start on whitespace, which is the
    # model's own business, so only the end is checked
    return reasoning.endswith("\n") and len(reasoning) > 1 and reasoning[:-1] == reasoning[:-1].rstrip()


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("lag", [0, 1])
def test_close_round_trips_once(tokenizer, budget_cls, seed, lag):
    state, cls, vocab = budget_cls
    rng = random.Random(seed)
    prompt = prompt_ids(tokenizer)
    reqs = [make_req(f"r{i}", prompt, slot=3 + i) for i in range(6)]
    sim = Sim(state, cls, vocab, k=4, rng=rng, lag=lag)
    committed = sim.run(reqs, steps=45)
    for r in reqs:
        toks = committed[r.rid]
        assert round_trips(tokenizer, toks, state.end_id), tokenizer.decode(toks[-40:])
        cut = toks.index(state.end_id)
        assert state.budget <= cut <= state.budget + 4 * 8, cut  # a few steps for whitespace waits and the lag
        assert not r.arc_think_armed and r.arc_think_open is False


def test_untouched_below_budget_and_when_closed(tokenizer, budget_cls):
    state, cls, vocab = budget_cls
    prompt = prompt_ids(tokenizer, enable_thinking=False)
    req = make_req("off", prompt, slot=5)
    req.output_ids.extend([sim_tok for sim_tok in range(1000, 1200)])
    batch = SimpleNamespace(reqs=[req], req_pool_indices=torch.tensor([5]))
    assert state.select(batch) == []
    on = prompt_ids(tokenizer)
    short = make_req("short", on, slot=6)
    short.output_ids.extend(range(1000, 1010))
    assert state.select(SimpleNamespace(reqs=[short], req_pool_indices=torch.tensor([6]))) == []


def test_grammar_requests_are_left_alone(tokenizer, budget_cls):
    state, _, _ = budget_cls
    on = prompt_ids(tokenizer)
    req = make_req("g", on, slot=7)
    req.grammar = object()
    req.output_ids.extend(range(1000, 1000 + state.budget))
    assert state.select(SimpleNamespace(reqs=[req], req_pool_indices=torch.tensor([7]))) == []
    assert req.arc_think_open is False and not req.arc_think_armed
