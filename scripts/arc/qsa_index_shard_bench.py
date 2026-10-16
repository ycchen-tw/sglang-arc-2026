#!/usr/bin/env python3
"""ARC_QSA_INDEX_SHARD=layer cost on an attention-TP pair (two GPUs, no model), CUDA-graph replay times.

  exchange   one verify step's 12 owner -> pair hand-offs of a [rows, 512] int32 block top-k (the
             QSAIndexLayerShard.exchange the indexer calls; rows = requests per DP x K), captured in one graph;
             per-step time = replay time. --rows takes several sizes (C128 verify 256, C96 192, prefill chunks).
  prefill    one EXTEND forward of the 12 QSA indexers (eager, as deployed), a 2048-row chunk after a long prefix:
             unsharded vs sharded GPU time between events around the 12 layers.
  indexer    one TARGET_VERIFY forward of the 12 QSA indexers at the deployed shape (R requests x K = 4 rows, top-k
             512, FP8 index, 4 x 128 index heads over hidden 2560), random index keys over contexts drawn from
             [--ctx-lo, --ctx-hi]: unsharded (each rank scores all 12 layers) vs sharded (each rank scores 6, receives
             6), both as CUDA graphs built from the real QwenSparseAttnBackend graph metadata. Production top-k
             (ARC_QSA_TOPK_EXACT=0).

Transport: the attention-TP group's all_gather_into_tensor -- pynccl inside graphs by default; --hostcomm installs
arc-forge's arc_hostcomm_hook (work/qwen38fn_l4/probes/hostcomm, the Kaggle hook) on the pair so the int16 wire rows go
through its ag_bf16 host-mapped gather, as on Kaggle (needs --hc-lib, a libhostcomm.so built from that hostcomm.cu).
Times are this machine's (PC: 2x RTX 4090 without P2P); the Kaggle L4 pair shares one PCIe Gen3 x16 uplink.

  python scripts/arc/qsa_index_shard_bench.py exchange --rows 192 256 2048 [--hostcomm --hook-dir D --hc-lib L]
  python scripts/arc/qsa_index_shard_bench.py indexer --requests 64 --ctx-lo 8192 --ctx-hi 65536
  python scripts/arc/qsa_index_shard_bench.py prefill --prefill-rows 2048 --prefill-prefix 32768
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qsa_index_shard_check as chk  # noqa: E402

QSA_LAYERS = chk.QSA_LAYERS
K = chk.K


def _init(rank, world, port, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      ARC_QSA_TOPK_EXACT="0", ARC_QSA_INDEX_FP8="1", ARC_QSA_INDEX_SHARD="layer")
    import torch

    torch.cuda.set_device(rank)
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
    group = get_attn_tp_group()
    transport = "pynccl"
    if args.hostcomm:
        os.environ.update(ARC_HC_LIB=args.hc_lib, ARC_HC_AG="1", ARC_HC_AR="1",
                          ARC_HC_SLOT_MB=str(args.hc_slot_mb))
        sys.path.insert(0, args.hook_dir)
        import arc_hostcomm_hook as hook

        from sglang.srt.layers import dp_attention

        hook.install(dp_attention)
        hook._setup_group(group, f"attn{min(group.ranks)}w{group.world_size}")
        transport = "hostcomm"
    return torch, group, transport


def _time_graph(torch, graph, reps, warm=20):
    for _ in range(warm):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _exchange_mode(rank, torch, group, args):
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.layers.attention.qsa import index_shard as ish

    shard = ish.QSAIndexLayerShard(QSA_LAYERS, group.rank_in_group, group.world_size, group)
    out = {}
    device = torch.device("cuda", rank)
    for rows in args.rows:
        owned = {lid: torch.randint(-1, 38400, (rows, chk.BLOCK_TOPK), dtype=torch.int32, device=device)
                 for lid in shard.owned_layer_ids}
        results = []
        g = torch.cuda.CUDAGraph()
        with graph_capture() as ctx:
            with torch.cuda.stream(ctx.stream):
                for _ in range(2):
                    for lid in QSA_LAYERS:
                        shard.exchange(lid, owned.get(lid), rows, chk.BLOCK_TOPK, device)
                ctx.stream.synchronize()
                with torch.cuda.graph(g, stream=ctx.stream):
                    results = [shard.exchange(lid, owned.get(lid), rows, chk.BLOCK_TOPK, device)
                               for lid in QSA_LAYERS]
        times = _time_graph(torch, g, args.reps)
        # every received top-k equals the owner's: gather the owners' tensors and compare
        torch.cuda.synchronize()
        ok = True
        for lid, got in zip(QSA_LAYERS, results):
            ref = got.clone()
            group.broadcast(ref, src=shard.owner(lid))
            ok &= bool(torch.equal(ref, got))
        out[rows] = {"wire_bytes_per_rank_per_layer": rows * chk.BLOCK_TOPK * 2,
                     "median_ms": statistics.median(times),
                     "p10_ms": sorted(times)[len(times) // 10], "p90_ms": sorted(times)[9 * len(times) // 10],
                     "received_equal_owner": ok}
    return out


def _arms(torch, group, R, ctx_hi, device):
    """Full and shard pools (random FP8 index keys everywhere, tiny main KV the indexer never reads), their
    backends over one req_to_token (request r + 1 owns a contiguous page-aligned row), the 12 indexers."""
    from sglang.srt.layers.attention.qsa import index_shard as ish
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    width = (ctx_hi // 64 + 2) * 64
    slots = (R + 2) * width
    shard = ish.QSAIndexLayerShard(QSA_LAYERS, group.rank_in_group, group.world_size, group)

    def pool(s):
        return QSATokenToKVPool(
            size=slots, dtype=torch.float8_e4m3fn, page_size=64, head_num=1, head_dim=8,
            full_attention_layer_ids=QSA_LAYERS, device=device, mamba_pool=None, qsa_index_kv_heads=1,
            qsa_index_head_dim=chk.INDEX_DIM, qsa_compress_ratio=chk.RATIO, qsa_token_topk=chk.BUDGET,
            num_request_slots=R + 1, qsa_max_verify_tokens=K, qsa_index_shard=s)

    pools = {"full": pool(None), "shard": pool(shard)}
    g = torch.Generator(device=device).manual_seed(5)
    for p in pools.values():
        flat = p.qsa_compressed_flat
        flat.copy_((torch.randn(flat.shape, generator=g, device=device) * 0.7).to(flat.dtype))
    req_to_token = torch.zeros((R + 1, width), dtype=torch.int32, device=device)
    for r in range(R):
        base = (r + 1) * width
        req_to_token[r + 1] = torch.arange(base, base + width, dtype=torch.int32, device=device)
    backends = {arm: chk._make_backend(p, req_to_token, width, device) for arm, p in pools.items()}
    return pools, backends, req_to_token, chk._make_indexers(device), g


def _prefill_mode(rank, torch, group, args):
    """One extend forward (eager, as the deployment runs prefill) of --prefill-seqs sequences, each a chunk of
    --prefill-rows / seqs tokens after a --prefill-prefix-token prefix whose index keys are random: the 12 indexers on
    each arm, GPU time between events around the 12 layers (both ranks start together after a barrier)."""
    device = torch.device("cuda", rank)
    n, prefix = args.prefill_seqs, args.prefill_prefix
    chunk = args.prefill_rows // n
    seq = prefix + chunk
    pools, backends, req_to_token, indexers, g = _arms(torch, group, n, seq, device)
    positions = torch.cat([torch.arange(prefix, seq, dtype=torch.int64) for _ in range(n)]).to(device)
    fb = SimpleNamespace(
        _original_forward_mode=None, forward_mode=chk._fm().EXTEND, batch_size=n,
        req_pool_indices=torch.arange(1, n + 1, dtype=torch.int64, device=device),
        seq_lens=torch.full((n,), seq, dtype=torch.int64, device=device), seq_lens_cpu=torch.full((n,), seq),
        positions=positions, mrope_positions=positions.unsqueeze(0).expand(3, -1).contiguous(), spec_info=None,
        extend_seq_lens=torch.full((n,), chunk, dtype=torch.int32, device=device), extend_seq_lens_cpu=[chunk] * n,
        input_ids=torch.zeros(n * chunk, dtype=torch.int64, device=device), num_padding=0,
        out_cache_loc=torch.cat([req_to_token[r + 1, prefix:seq].long() for r in range(n)]),
    )
    hidden = [(torch.randn(n * chunk, chk.HIDDEN, generator=g, device=device) * 0.5).to(torch.bfloat16)
              for _ in QSA_LAYERS]
    pos2 = fb.mrope_positions
    times = {"full": [], "shard": []}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for it in range(args.reps // 10 + 3):
        for arm in ("full", "shard"):
            with chk._shard_mode(arm == "shard"):
                backends[arm].init_forward_metadata(fb)
                md = backends[arm].forward_metadata.indexer_metadata
                torch.cuda.synchronize()
                group.barrier()
                start.record()
                for i, lid in enumerate(QSA_LAYERS):
                    indexers[lid].forward_cuda(hidden[i], pos2, fb, md)
                end.record()
                end.synchronize()
            if it >= 3:
                times[arm].append(start.elapsed_time(end))
    out = {"rows": n * chunk, "seqs": n, "prefix": prefix}
    for arm, t in times.items():
        out[arm] = {"median_ms": statistics.median(t), "min_ms": min(t), "max_ms": max(t), "n": len(t)}
    out["delta_ms"] = out["shard"]["median_ms"] - out["full"]["median_ms"]
    return out


def _indexer_mode(rank, torch, group, args):
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.layers.attention.qsa import index_shard as ish
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    device = torch.device("cuda", rank)
    R = args.requests
    rng = random.Random(args.seed)
    ctxs = [rng.randint(args.ctx_lo, args.ctx_hi) // 64 * 64 for _ in range(R)]
    pools, backends, req_to_token, indexers, g = _arms(torch, group, R, args.ctx_hi, device)
    positions = torch.tensor([c + j for c in ctxs for j in range(K)], dtype=torch.int64, device=device)
    fb = SimpleNamespace(
        _original_forward_mode=None, forward_mode=chk._fm().TARGET_VERIFY, batch_size=R,
        req_pool_indices=torch.arange(1, R + 1, dtype=torch.int64, device=device),
        seq_lens=torch.tensor(ctxs, dtype=torch.int64, device=device), seq_lens_cpu=torch.tensor(ctxs),
        positions=positions, mrope_positions=positions.unsqueeze(0).expand(3, -1).contiguous(),
        spec_info=SimpleNamespace(draft_token_num=K, topk=1), extend_seq_lens=None, extend_seq_lens_cpu=None,
        input_ids=torch.zeros(R * K, dtype=torch.int64, device=device), num_padding=0,
        out_cache_loc=torch.cat([req_to_token[r + 1, c:c + K].long() for r, c in enumerate(ctxs)]),
    )
    hidden = [(torch.randn(R * K, chk.HIDDEN, generator=g, device=device) * 0.5).to(torch.bfloat16)
              for _ in QSA_LAYERS]
    static_pos = fb.mrope_positions
    graphs = {}
    for arm, backend in backends.items():
        backend.init_cuda_graph_state(R, R * K)
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as ctx, chk._shard_mode(arm == "shard"):
            with model_capture_mode():
                backend.init_forward_metadata_out_graph(fb, in_capture=True)
                md = backend.forward_metadata.indexer_metadata
                with torch.cuda.stream(ctx.stream):
                    for _ in range(2):
                        for i, lid in enumerate(QSA_LAYERS):
                            indexers[lid].forward_cuda(hidden[i], static_pos, fb, md)
                ctx.stream.synchronize()
                with torch.cuda.graph(graph, stream=ctx.stream):
                    for i, lid in enumerate(QSA_LAYERS):
                        indexers[lid].forward_cuda(hidden[i], static_pos, fb, md)
        backend.init_forward_metadata_out_graph(fb, in_capture=False)
        graphs[arm] = graph
    torch.cuda.synchronize()
    out = {"contexts_mean": statistics.mean(ctxs), "contexts_min": min(ctxs), "contexts_max": max(ctxs),
           "rows": R * K}
    # interleave the arms so drift (clocks, temperature) hits both alike
    times = {"full": [], "shard": []}
    for _ in range(args.rounds):
        for arm in ("full", "shard"):
            group.barrier()
            times[arm] += _time_graph(torch, graphs[arm], args.reps, warm=5)
    for arm, t in times.items():
        out[arm] = {"median_ms": statistics.median(t), "p10_ms": sorted(t)[len(t) // 10],
                    "p90_ms": sorted(t)[9 * len(t) // 10], "n": len(t)}
    out["delta_ms"] = out["shard"]["median_ms"] - out["full"]["median_ms"]
    return out


def _worker(rank, world, port, args, result_dir):
    try:
        torch, group, transport = _init(rank, world, port, args)
        res = {"rank": rank, "mode": args.mode, "transport": transport,
               "device": torch.cuda.get_device_name(rank)}
        if args.mode == "exchange":
            res["exchange"] = _exchange_mode(rank, torch, group, args)
        elif args.mode == "prefill":
            res["prefill"] = _prefill_mode(rank, torch, group, args)
        else:
            res["indexer"] = _indexer_mode(rank, torch, group, args)
        if args.hostcomm:
            # the hook's per-group routing counters: the hand-offs must have gone through hostcomm, not NCCL
            res["hostcomm_counts"] = dict(group._arc_hc["counts"])
        res["ok"] = True
    except BaseException:
        res = {"rank": rank, "ok": False, "error": traceback.format_exc()}
    with open(os.path.join(result_dir, f"rank{rank}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("mode", choices=("exchange", "indexer", "prefill"))
    ap.add_argument("--rows", type=int, nargs="+", default=[192, 256, 1024, 2048])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--ctx-lo", type=int, default=8192)
    ap.add_argument("--ctx-hi", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill-rows", type=int, default=2048)
    ap.add_argument("--prefill-seqs", type=int, default=1)
    ap.add_argument("--prefill-prefix", type=int, default=32768)
    ap.add_argument("--hostcomm", action="store_true")
    ap.add_argument("--hook-dir")
    ap.add_argument("--hc-lib")
    ap.add_argument("--hc-slot-mb", type=int, default=32)
    ap.add_argument("--port", type=int, default=29741)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.hostcomm and not (args.hook_dir and args.hc_lib):
        raise SystemExit("--hostcomm needs --hook-dir and --hc-lib")
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(2, args.port, args, tmp), nprocs=2, join=True)
        ranks = [json.load(open(os.path.join(tmp, f"rank{r}.json"))) for r in range(2)]
    result = {"args": vars(args), "ranks": ranks, "ok": all(r.get("ok") for r in ranks)}
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
