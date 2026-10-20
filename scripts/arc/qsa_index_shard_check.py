#!/usr/bin/env python3
"""ARC_QSA_INDEX_SHARD=layer check: the QSA index split over an attention-TP pair is bit-identical to the unsplit one.

Two GPUs (spawns an attention-TP2 group over cuda:0 / cuda:1), no model. Each rank builds the deployed indexer shape
(Qwen3.8-FN: 12 QSA layers 3, 7, ..., 47, hidden 2560, 4 index heads x 128, top-k 512 blocks of 4 tokens, MRoPE
[11, 11, 10] interleaved over rotary 64, FP8 index keys by default) with the same seeded weights, and two
QSATokenToKVPools fed through the real QwenSparseAttnBackend metadata builder:

  full    unsharded (shard None): every layer's index on both ranks
  shard   ARC_QSA_INDEX_SHARD=layer: QSA layer i's index on rank i % 2 only; the owner hands its top-k to the pair

Every forward runs all 12 layers on both arms with the same hidden states, and every layer's output (the logical
top-k token indices the attention reads) must be equal, bit for bit, on both ranks:

  extend    chunked prefill of R prompts (2.6k-9k tokens) in 2048-token forwards of up to 3 sequences (prefixes
            from earlier chunks; a prompt's last chunk leaves an incomplete group), so the top-k covers more than
            512 complete blocks
  decode    DECODE steps (one row per request) crossing group boundaries
  verify    TARGET_VERIFY steps, K = 4 rows per request, random accepts
  graph     DECODE and TARGET_VERIFY captured as CUDA graphs on both arms (backend graph metadata, the pair hand-off
            inside the graph via graph_capture's pynccl), replayed over more steps

Afterwards the owned layers' index state (compressed keys, FP8 scales, pending ring) of the shard pool equals the
full pool's, byte for byte; an unowned layer's accessors raise; the shard pool holds half the index bytes.
Determinism: ARC_QSA_TOPK_EXACT=1 (the production fast_topk orders its picks by atomics, so even two runs of the
unsharded arm differ in order).

--mode layer-check runs the in-server check mode instead (ARC_QSA_INDEX_SHARD=layer-check): the shard pool keeps every
layer, the receiving rank compares the owner's top-k with its own on the device, and the counts must show every
received row checked and none differing (with --negative-control: the counts must show the corrupted rows and the
eager poll must raise).

--ref-commit <commit> adds a third arm, "ref": the commit's own qsa_indexer.py and qsa_kv_pool.py (read from the
fork's object store), same weights and batches, which the flag-off arm must equal bit for bit (outputs and every
layer's index state): with --ref-commit 8d4dea09 the switch off is the branch point.

  python scripts/arc/qsa_index_shard_check.py [--mode layer|layer-check] [--index-fp8 0|1] [--requests 8] [--out r.json]
  python scripts/arc/qsa_index_shard_check.py --ref-commit 8d4dea09   # also: flag off == the branch point
  python scripts/arc/qsa_index_shard_check.py --negative-control   # must report mismatches (exit 0 if it does)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

QSA_LAYERS = list(range(3, 48, 4))
HIDDEN, INDEX_HEADS, INDEX_DIM, RATIO, BUDGET = 2560, 4, 128, 4, 2048
BLOCK_TOPK = BUDGET // RATIO
PAGE = 64
K = 4
CHUNK = 2048


def _cpu_checks() -> dict:
    """Ownership bookkeeping, flag parsing and the exchange contract (no GPU, no group)."""
    import torch

    from sglang.srt.layers.attention.qsa import index_shard as ish

    out = {}
    s0 = ish.QSAIndexLayerShard(QSA_LAYERS, 0, 2)
    s1 = ish.QSAIndexLayerShard(QSA_LAYERS, 1, 2)
    assert s0.owned_layer_ids == tuple(QSA_LAYERS[0::2]) and s1.owned_layer_ids == tuple(QSA_LAYERS[1::2])
    assert all(s0.owner(lid) == i % 2 for i, lid in enumerate(QSA_LAYERS))
    out["owned_rank0"], out["owned_rank1"] = list(s0.owned_layer_ids), list(s1.owned_layer_ids)
    for bad in (lambda: ish.QSAIndexLayerShard(QSA_LAYERS[:11], 0, 2),
                lambda: ish.QSAIndexLayerShard(QSA_LAYERS, 0, 4),
                lambda: ish.QSAIndexLayerShard(QSA_LAYERS[::-1], 0, 2),
                lambda: ish.QSAIndexLayerShard(QSA_LAYERS, 2, 2)):
        try:
            bad()
        except (NotImplementedError, ValueError):
            continue
        raise AssertionError("an invalid shard was accepted")
    try:
        s0.owner(4)
        raise AssertionError("a non-QSA layer has an owner")
    except RuntimeError:
        pass
    try:
        s0.exchange(3, torch.zeros(1, BLOCK_TOPK, dtype=torch.int32), 1, BLOCK_TOPK, "cpu")
        raise AssertionError("exchange ran without a group")
    except RuntimeError:
        pass
    # the 16-bit hand-off: every block index below WIRE_MAX_BLOCKS and -1 survive the round trip
    values = torch.tensor([[-1, 0, 1, 32766, 32767, 32768, 40000, ish.WIRE_MAX_BLOCKS - 2, -1, 7]],
                          dtype=torch.int32)
    wire = ish.to_wire(values)
    assert wire.dtype == torch.int16 and torch.equal(ish.from_wire(wire), values), (wire, ish.from_wire(wire))
    every = torch.arange(-1, ish.WIRE_MAX_BLOCKS - 1, dtype=torch.int32).view(1, -1)
    assert torch.equal(ish.from_wire(ish.to_wire(every)), every)
    out["wire_roundtrip_values"] = every.numel()
    try:
        ish.QSAIndexLayerShard(QSA_LAYERS, 0, 2, max_blocks=ish.WIRE_MAX_BLOCKS)
        raise AssertionError("a context past the 16-bit hand-off was accepted")
    except NotImplementedError:
        pass
    for raw, want in (("", None), ("0", None), ("layer", "layer"), ("layer-check", "layer-check")):
        assert ish._parse(raw) == want
    checker = ish.QSAIndexLayerShard(QSA_LAYERS, 1, 2, check=True, device="cpu")
    assert checker.held_layer_ids == tuple(QSA_LAYERS) and s1.held_layer_ids == s1.owned_layer_ids
    checker.verify(3, torch.zeros(2, BLOCK_TOPK, dtype=torch.int32), torch.zeros(2, BLOCK_TOPK, dtype=torch.int32))
    checker.verify(11, torch.ones(3, BLOCK_TOPK, dtype=torch.int32), torch.zeros(3, BLOCK_TOPK, dtype=torch.int32))
    assert checker._counts.tolist() == [5, 3], checker._counts
    try:
        checker.verify(7, torch.zeros(1, BLOCK_TOPK, dtype=torch.int32), torch.zeros(1, BLOCK_TOPK, dtype=torch.int32))
        raise AssertionError("the owner of layer 7 checked a received top-k")
    except RuntimeError:
        pass
    try:
        ish._parse("1")
        raise AssertionError("ARC_QSA_INDEX_SHARD=1 was accepted")
    except ValueError:
        pass
    out["ok"] = True
    return out


MODE = "layer"  # the shard arm's ARC_QSA_INDEX_SHARD (--mode)


@contextmanager
def _shard_mode(on: bool):
    """index_shard parses ARC_QSA_INDEX_SHARD once at import; the reference arm runs with it off."""
    from sglang.srt.layers.attention.qsa import index_shard as ish

    saved = ish._MODE
    ish._MODE = MODE if on else None
    try:
        yield
    finally:
        ish._MODE = saved


def _make_config():
    return SimpleNamespace(
        indexer_n_heads=INDEX_HEADS, indexer_kv_heads=1, indexer_head_dim=INDEX_DIM, indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO, hidden_size=HIDDEN, rms_norm_eps=1e-6,
    )


def _load_ref(ref_dir, name):
    """The --ref-commit version of sglang.srt.<...>.<name>.py, as its own module."""
    import importlib.util

    path = os.path.join(ref_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"arc_qsa_shard_check_ref_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_indexers(device, indexer_class=None):
    import torch

    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    QSAIndexer = indexer_class or QSAIndexer

    rotary = MRotaryEmbedding(head_size=256, rotary_dim=64, max_position_embeddings=262144, base=10000000,
                              is_neox_style=True, dtype=torch.bfloat16, mrope_section=[11, 11, 10],
                              mrope_interleaved=True).to(device)
    indexers = {}
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for i, layer_id in enumerate(QSA_LAYERS):
            ix = QSAIndexer(_make_config(), layer_id=layer_id, quant_config=None, rotary_emb=rotary).to(device=device)
            g = torch.Generator().manual_seed(1000 + i)
            with torch.no_grad():
                ix.index_qk_proj.weight.data.copy_(
                    (torch.randn((INDEX_HEADS + 1) * INDEX_DIM, HIDDEN, generator=g) * 0.02).to(torch.bfloat16))
                for norm in (ix.q_layernorm, ix.k_layernorm):
                    norm._weight_loader(norm.weight, (torch.randn(INDEX_DIM, generator=g) * 0.1).to(torch.bfloat16))
            indexers[layer_id] = ix
    finally:
        torch.set_default_dtype(prev)
    return indexers


def _make_pool(*, shard, slots, requests, device, pool_class=None):
    import torch

    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    kwargs = dict(
        size=slots, dtype=torch.float8_e4m3fn, page_size=PAGE, head_num=1, head_dim=256,
        full_attention_layer_ids=QSA_LAYERS, device=device, mamba_pool=None, qsa_index_kv_heads=1,
        qsa_index_head_dim=INDEX_DIM, qsa_compress_ratio=RATIO, qsa_token_topk=BUDGET,
        num_request_slots=requests + 1, qsa_max_verify_tokens=K,
    )
    if pool_class is not None:  # --ref-commit: the base commit's pool has no shard argument
        return pool_class(**kwargs)
    return QSATokenToKVPool(**kwargs, qsa_index_shard=shard)


def _make_backend(pool, req_to_token, ctx, device):
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend

    cfg = _make_config()
    runner = SimpleNamespace(
        token_to_kv_pool=pool, device=device, hisparse_coordinator=None,
        model_config=SimpleNamespace(hf_text_config=cfg, hf_config=cfg, context_len=ctx),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token), is_draft_worker=False,
    )
    return QwenSparseAttnBackend(runner)


class Workload:
    """Host-side request state shared by both arms; every forward batch is built once and fed to both backends."""

    def __init__(self, requests, ctx, device, seed):
        import torch

        self.torch = torch
        self.R, self.ctx, self.device = requests, ctx, device
        rng = random.Random(seed)
        self.prompt = [rng.randint(2600, 9000) for _ in range(requests)]
        self.done = [0] * requests  # committed tokens
        self.rng = rng
        # request slot r + 1 owns raw slots [(r + 1) * ctx, (r + 2) * ctx): page aligned, slot 0 is padding
        self.req_to_token = torch.zeros((requests + 1, ctx), dtype=torch.int32, device=device)
        for r in range(requests):
            base = (r + 1) * ctx
            self.req_to_token[r + 1] = torch.arange(base, base + ctx, dtype=torch.int32, device=device)
        self.forward_id = 0

    def _fb(self, **kw):
        # qsa_ring_resume_cpu: ForwardBatch's default (None, no session exact resume row), read by the write plan
        base = dict(_original_forward_mode=None, spec_info=None, mrope_positions=None, extend_seq_lens=None,
                    extend_seq_lens_cpu=None, num_padding=0, qsa_ring_resume_cpu=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def extend_batches(self):
        """Chunked prefill: each forward takes up to 3 requests' next chunks (2048 tokens per forward)."""
        torch = self.torch
        pending = list(range(self.R))
        while pending:
            budget, picked = CHUNK, []
            for r in list(pending):
                if budget <= 0 or len(picked) == 3:
                    break
                rest = self.prompt[r] - self.done[r]
                # a truncated chunk ends on a page (the paged scheduler's rule: the next chunk's prefix must
                # cover whole compression groups); a prompt's last chunk ends anywhere
                n = rest if rest <= budget else budget // PAGE * PAGE
                if n == 0:
                    break
                picked.append((r, n))
                budget -= n
            # positions / slots / lengths of the picked chunks
            pos, ext, seq = [], [], []
            for r, n in picked:
                pos.append(torch.arange(self.done[r], self.done[r] + n, dtype=torch.int64))
                ext.append(n)
                seq.append(self.done[r] + n)
            positions = torch.cat(pos).to(self.device)
            reqs = torch.tensor([r + 1 for r, _ in picked], dtype=torch.int64, device=self.device)
            out_loc = torch.cat([self.req_to_token[r + 1, self.done[r]:self.done[r] + n].long() for r, n in picked])
            fb = self._fb(
                forward_mode=_fm().EXTEND, batch_size=len(picked), req_pool_indices=reqs,
                seq_lens=torch.tensor(seq, dtype=torch.int64, device=self.device),
                seq_lens_cpu=torch.tensor(seq, dtype=torch.int64), positions=positions,
                mrope_positions=positions.unsqueeze(0).expand(3, -1).contiguous(),
                extend_seq_lens=torch.tensor(ext, dtype=torch.int32, device=self.device), extend_seq_lens_cpu=ext,
                input_ids=torch.zeros(positions.numel(), dtype=torch.int64, device=self.device),
                out_cache_loc=out_loc,
            )
            yield fb
            for r, n in picked:
                self.done[r] += n
                if self.done[r] == self.prompt[r]:
                    pending.remove(r)

    def decode_batch(self):
        torch = self.torch
        seq = [d + 1 for d in self.done]
        positions = torch.tensor(self.done, dtype=torch.int64, device=self.device)
        fb = self._fb(
            forward_mode=_fm().DECODE, batch_size=self.R,
            req_pool_indices=torch.arange(1, self.R + 1, dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor(seq, dtype=torch.int64, device=self.device),
            seq_lens_cpu=torch.tensor(seq, dtype=torch.int64), positions=positions,
            mrope_positions=positions.unsqueeze(0).expand(3, -1).contiguous(),
            input_ids=torch.zeros(self.R, dtype=torch.int64, device=self.device),
            out_cache_loc=torch.stack([self.req_to_token[r + 1, self.done[r]].long() for r in range(self.R)]),
        )
        return fb

    def commit_decode(self):
        self.done = [d + 1 for d in self.done]

    def verify_batch(self):
        torch = self.torch
        positions = torch.tensor([d + j for d in self.done for j in range(K)], dtype=torch.int64, device=self.device)
        fb = self._fb(
            forward_mode=_fm().TARGET_VERIFY, batch_size=self.R,
            req_pool_indices=torch.arange(1, self.R + 1, dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor(self.done, dtype=torch.int64, device=self.device),
            seq_lens_cpu=torch.tensor(self.done, dtype=torch.int64), positions=positions,
            mrope_positions=positions.unsqueeze(0).expand(3, -1).contiguous(),
            spec_info=SimpleNamespace(draft_token_num=K, topk=1),
            input_ids=torch.zeros(self.R * K, dtype=torch.int64, device=self.device),
            out_cache_loc=torch.cat([self.req_to_token[r + 1, d:d + K].long() for r, d in enumerate(self.done)]),
        )
        return fb

    def commit_verify(self):
        self.done = [d + self.rng.randint(1, K) for d in self.done]

    def hidden(self, rows, layer_index):
        """Same bf16 hidden states on both ranks and both arms for (forward, layer)."""
        torch = self.torch
        g = torch.Generator().manual_seed(7919 * self.forward_id + layer_index)
        return (torch.randn(rows, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).to(self.device)


def _fm():
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return ForwardMode


def _run_layers(indexers, backend, fb, work, positions_2d, sharded):
    """One forward of the 12 indexers on one arm; returns [12] logical top-k tensors."""
    backend.init_forward_metadata(fb)
    md = backend.forward_metadata.indexer_metadata
    rows = positions_2d.shape[1]
    outs = []
    with _shard_mode(sharded):
        for i, layer_id in enumerate(QSA_LAYERS):
            outs.append(indexers[layer_id].forward_cuda(work.hidden(rows, i), positions_2d, fb, md))
    return outs


def _compare(tag, full, shard, mismatches):
    for i, (a, b) in enumerate(zip(full, shard)):
        if a.shape != b.shape or a.dtype != b.dtype or not bool((a == b).all()):
            diff = -1 if a.shape != b.shape else int((a != b).sum())
            mismatches.append({"forward": tag, "layer": QSA_LAYERS[i], "shape": [list(a.shape), list(b.shape)],
                               "differing": diff})


def _graph_phase(kind, indexers, backends, work, steps, mismatches, stats, ref_mismatches):
    """Capture one DECODE / TARGET_VERIFY forward per arm as a CUDA graph (backend graph metadata, the hand-off in
    the graph), then replay both arms for `steps` steps on the live state."""
    import torch

    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    R = work.R
    rows = R * (K if kind == "verify" else 1)
    fb0 = work.verify_batch() if kind == "verify" else work.decode_batch()
    static_hidden = [torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=work.device) for _ in QSA_LAYERS]
    static_pos = torch.zeros(3, rows, dtype=torch.int64, device=work.device)
    graphs, outs = {}, {}
    for arm, backend in backends.items():
        backend.init_cuda_graph_state(R, R * K)
        with graph_capture() as ctx, _shard_mode(arm == "shard"):
            stream = ctx.stream
            with model_capture_mode():
                backend.init_forward_metadata_out_graph(fb0, in_capture=True)
                md = backend.forward_metadata.indexer_metadata
                with torch.cuda.stream(stream):
                    for _ in range(2):  # warm-up, as the graph runner does
                        for i, layer_id in enumerate(QSA_LAYERS):
                            indexers[arm][layer_id].forward_cuda(static_hidden[i], static_pos, fb0, md)
                stream.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    outs[arm] = [indexers[arm][layer_id].forward_cuda(static_hidden[i], static_pos, fb0, md)
                                 for i, layer_id in enumerate(QSA_LAYERS)]
        graphs[arm] = g
    torch.cuda.synchronize()
    for step in range(steps):
        work.forward_id += 1
        fb = work.verify_batch() if kind == "verify" else work.decode_batch()
        static_pos.copy_(fb.mrope_positions)
        for i in range(len(QSA_LAYERS)):
            static_hidden[i].copy_(work.hidden(rows, i))
        got = {}
        for arm, backend in backends.items():
            backend.init_forward_metadata_out_graph(fb, in_capture=False)
            graphs[arm].replay()
            got[arm] = [o.clone() for o in outs[arm]]
        torch.cuda.synchronize()
        _compare(f"graph-{kind}-{step}", got["full"], got["shard"], mismatches)
        if "ref" in got:
            _compare(f"graph-{kind}-{step}", got["full"], got["ref"], ref_mismatches)
        stats[f"graph_{kind}_valid_entries"] = stats.get(f"graph_{kind}_valid_entries", 0) + sum(
            int((o >= 0).sum()) for o in got["full"])
        (work.commit_verify if kind == "verify" else work.commit_decode)()


def _worker(rank, world, port, args, result_dir):
    global MODE
    MODE = args.mode
    checking = MODE == "layer-check"
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                          ARC_QSA_TOPK_EXACT="1", ARC_QSA_INDEX_FP8=str(args.index_fp8),
                          ARC_QSA_INDEX_SHARD=MODE)
        import torch

        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy"), role="test")
        from sglang.srt.distributed.parallel_state import (
            get_attn_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(world_size=world, rank=rank, local_rank=rank,
                                     distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world)
        from sglang.srt.layers.attention.qsa import index_shard as ish

        group = get_attn_tp_group()
        assert group.world_size == 2 and group.rank_in_group == rank
        shard = ish.QSAIndexLayerShard(QSA_LAYERS, group.rank_in_group, group.world_size, group, check=checking,
                                       device=device)
        poll = shard._poll_check
        if checking and args.negative_control:
            # count the corrupted rows on the device through the whole run, then poll once at the end
            shard._poll_check = lambda: None
        if args.negative_control:
            # the receiving rank shifts one selected block of the first row: the check must see it
            exchange = shard.exchange

            def corrupted(layer_id, blocks, rows, width, device):
                got = exchange(layer_id, blocks, rows, width, device)
                if blocks is None and rows:
                    got = got.clone()
                    got[0, 0] = torch.where(got[0, 0] > 0, got[0, 0] - 1, got[0, 0])
                return got

            shard.exchange = corrupted
        ctx = 16384
        work = Workload(args.requests, ctx, device, seed=args.seed)
        slots = (args.requests + 2) * ctx
        pools = {"full": _make_pool(shard=None, slots=slots, requests=args.requests, device=device),
                 "shard": _make_pool(shard=shard, slots=slots, requests=args.requests, device=device)}
        indexers = _make_indexers(device)
        arm_indexers = {"full": indexers, "shard": indexers}
        if args.ref_dir:
            pools["ref"] = _make_pool(shard=None, slots=slots, requests=args.requests, device=device,
                                      pool_class=_load_ref(args.ref_dir, "qsa_kv_pool").QSATokenToKVPool)
            arm_indexers["ref"] = _make_indexers(device, _load_ref(args.ref_dir, "qsa_indexer").QSAIndexer)
        backends = {arm: _make_backend(pool, work.req_to_token, ctx, device) for arm, pool in pools.items()}
        stats = {"rank": rank, "index_fp8": args.index_fp8, "prompts": work.prompt,
                 "owned": list(shard.owned_layer_ids)}

        # the shard pool keeps the owned layers only (layer-check: all of them)
        unowned = [lid for lid in QSA_LAYERS if lid not in shard.owned_layer_ids]
        for getter in ("get_qsa_compressed_k_buffer", "get_qsa_key_state_buffer", "get_qsa_compressed_k_scale"):
            if checking:
                break
            try:
                getattr(pools["shard"], getter)(unowned[0])
                if not (getter == "get_qsa_compressed_k_scale" and not args.index_fp8):
                    raise AssertionError(f"{getter} served an unowned layer")
            except RuntimeError:
                pass

        def index_bytes(pool):
            return (pool.qsa_compressed_flat.nbytes + sum(t.nbytes for t in pool.qsa_key_state_buffer_pool)
                    + (pool.qsa_compressed_scale_flat.nbytes if pool.qsa_index_fp8 else 0))

        stats["index_bytes"] = {arm: index_bytes(pool) for arm, pool in pools.items()}
        want = stats["index_bytes"]["full"] if checking else stats["index_bytes"]["full"] // 2
        assert stats["index_bytes"]["shard"] == want and stats["index_bytes"]["full"] % 2 == 0, stats["index_bytes"]
        stats["kv_size_bytes"] = {arm: int(sum(pool.get_kv_size_bytes())) for arm, pool in pools.items()}

        mismatches, ref_mismatches = [], []
        counts = {"extend": 0, "decode": 0, "verify": 0}

        def forward_all(fb, tag):
            outs = {arm: _run_layers(arm_indexers[arm], backends[arm], fb, work, fb.mrope_positions, arm == "shard")
                    for arm in pools}
            torch.cuda.synchronize()
            _compare(tag, outs["full"], outs["shard"], mismatches)
            if "ref" in outs:
                _compare(tag, outs["full"], outs["ref"], ref_mismatches)
            return outs

        for fb in work.extend_batches():
            work.forward_id += 1
            outs = forward_all(fb, f"extend-{counts['extend']}")
            stats["extend_valid_entries"] = stats.get("extend_valid_entries", 0) + sum(
                int((o >= 0).sum()) for o in outs["full"])
            counts["extend"] += 1
        stats["max_complete_blocks"] = max(p // RATIO for p in work.prompt)
        for step in range(args.decode_steps):
            work.forward_id += 1
            fb = work.decode_batch()
            forward_all(fb, f"decode-{step}")
            work.commit_decode()
            counts["decode"] += 1
        for step in range(args.verify_steps):
            work.forward_id += 1
            fb = work.verify_batch()
            forward_all(fb, f"verify-{step}")
            work.commit_verify()
            counts["verify"] += 1
        _graph_phase("decode", arm_indexers, backends, work, args.graph_steps, mismatches, stats, ref_mismatches)
        _graph_phase("verify", arm_indexers, backends, work, args.graph_steps, mismatches, stats, ref_mismatches)
        counts["graph_decode"] = counts["graph_verify"] = args.graph_steps

        # owned layers' index state: shard pool == full pool, byte for byte, outside the inert dumps (compressed
        # slot 0 takes every non-boundary row's write and request slot 0's pending ring every extend token that is
        # not pending: many writers per slot, so their final bytes are whichever write lands last, on either arm)
        ring = pools["full"].qsa_pending_ring_size
        state_diff, dump_diff = [], {}
        for lid in shard.held_layer_ids:
            for getter, dump_rows in (("get_qsa_compressed_k_buffer", 1), ("get_qsa_key_state_buffer", ring),
                                      ("get_qsa_compressed_k_scale", 1)):
                a, b = getattr(pools["full"], getter)(lid), getattr(pools["shard"], getter)(lid)
                if a is None and b is None:
                    continue
                rows_differ = (a.view(torch.uint8).reshape(a.shape[0], -1)
                               != b.view(torch.uint8).reshape(b.shape[0], -1)).any(dim=1)
                live = int(rows_differ[dump_rows:].sum())
                if live:
                    state_diff.append([lid, getter, live])
                dump_diff[f"{lid}.{getter}"] = int(rows_differ[:dump_rows].sum())
        rope_a, rope_b = pools["full"].qsa_rope_position_buffer, pools["shard"].qsa_rope_position_buffer
        stats["rope_buffer_equal"] = bool(torch.equal(rope_a[ring:], rope_b[ring:]))
        stats["dump_rows_differing"] = sum(dump_diff.values())
        stats.update(counts=counts, mismatches=mismatches[:20], num_mismatches=len(mismatches),
                     state_diff=state_diff, done=work.done)
        stats["ok"] = not mismatches and not state_diff and stats["rope_buffer_equal"]
        if args.ref_dir:
            # flag off == the ref commit: every output and every layer's index state, byte for byte (dumps aside)
            ref_state = []
            for lid in QSA_LAYERS:
                for getter, dump_rows in (("get_qsa_compressed_k_buffer", 1), ("get_qsa_key_state_buffer", ring),
                                          ("get_qsa_compressed_k_scale", 1)):
                    a, b = getattr(pools["full"], getter)(lid), getattr(pools["ref"], getter)(lid)
                    if a is None and b is None:
                        continue
                    live = int((a.view(torch.uint8).reshape(a.shape[0], -1)
                                != b.view(torch.uint8).reshape(b.shape[0], -1)).any(dim=1)[dump_rows:].sum())
                    if live:
                        ref_state.append([lid, getter, live])
            ref_rope = pools["ref"].qsa_rope_position_buffer
            stats["ref"] = {"commit": args.ref_commit, "num_mismatches": len(ref_mismatches),
                            "mismatches": ref_mismatches[:20], "state_diff": ref_state,
                            "rope_buffer_equal": bool(torch.equal(rope_a[ring:], ref_rope[ring:]))}
            stats["ok"] = (stats["ok"] and not ref_mismatches and not ref_state
                           and stats["ref"]["rope_buffer_equal"])
        if checking:
            checked, differing = shard._counts.tolist()
            stats["check_counts"] = {"checked_rows": checked, "differing_rows": differing}
            try:
                poll()
                stats["check_poll_raised"] = None
            except RuntimeError as e:
                stats["check_poll_raised"] = str(e)
            stats["ok"] = stats["ok"] and checked > 0 and differing == 0 and stats["check_poll_raised"] is None
    except BaseException:
        stats = {"rank": rank, "ok": False, "error": traceback.format_exc()}
    with open(os.path.join(result_dir, f"rank{rank}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except BaseException:
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--index-fp8", type=int, choices=(0, 1), default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=6)
    ap.add_argument("--verify-steps", type=int, default=6)
    ap.add_argument("--graph-steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=29731)
    ap.add_argument("--mode", choices=("layer", "layer-check"), default="layer")
    ap.add_argument("--ref-commit", help="also compare the flag-off arm with this commit's indexer and pool")
    ap.add_argument("--negative-control", action="store_true",
                    help="corrupt one received block per layer; succeeds only if the check reports mismatches")
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    import torch.multiprocessing as mp

    result = {"cpu": _cpu_checks()}
    if torch.cuda.device_count() < 2:
        raise SystemExit("the TP2 check needs two GPUs")
    with tempfile.TemporaryDirectory() as tmp:
        args.ref_dir = None
        if args.ref_commit:
            import subprocess

            fork = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            args.ref_dir = os.path.join(tmp, "ref")
            os.makedirs(args.ref_dir)
            for rel in ("python/sglang/srt/layers/attention/qsa/qsa_indexer.py",
                        "python/sglang/srt/mem_cache/qsa_kv_pool.py"):
                text = subprocess.run(["git", "-C", fork, "show", f"{args.ref_commit}:{rel}"], check=True,
                                      capture_output=True).stdout
                with open(os.path.join(args.ref_dir, os.path.basename(rel)), "wb") as f:
                    f.write(text)
        mp.spawn(_worker, args=(2, args.port, args, tmp), nprocs=2, join=True)
        ranks = [json.load(open(os.path.join(tmp, f"rank{r}.json"))) for r in range(2)]
    result["ranks"] = ranks
    if args.negative_control:
        result["negative_control"] = True
        result["ok"] = all("error" not in r and r["num_mismatches"] > 0 for r in ranks)
        if args.mode == "layer-check":  # the device counts saw the corrupted rows and the poll raised on them
            result["ok"] = result["ok"] and all(r["check_counts"]["differing_rows"] > 0 and r["check_poll_raised"]
                                                for r in ranks)
    else:
        result["ok"] = result["cpu"]["ok"] and all(r.get("ok") for r in ranks)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
