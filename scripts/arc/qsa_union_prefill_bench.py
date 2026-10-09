#!/usr/bin/env python3
"""Microbench + accuracy of ARC_QSA_UNION_PREFILL against the validated QSA prefix chunk kernel.

  dump  FILE.pt [FILE.pt ...] [--configs T,NB,W,S;...] [--ref-rows 512]
        Inputs saved by a server with ARC_QSA_PREFILL_DUMP=<dir> (real traffic: q, the layer's
        top-k indices, the gathered FP8 context, lengths). The context becomes the "pool"
        (req_to_token row b = cu_k[b] + j), so both paths read the same bytes.
  synth [--bs 4 --new 704 --prefix 10240 --drift 0.15]
        Correlated synthetic selections (each query's block scores = shared + a random walk along
        the sequence), for a functional check before a dump exists.

Per case: union size (keys a tile reads vs T x the per-query keys), time of the chunk kernel
alone and with the context gather the backend does before it (`_extend_context`: index_select +
cat per request and plane), time of the union path (zero + scatter + compact + attention) and of
its attention kernel alone, and accuracy of both against a float64 reference on --ref-rows sampled
rows: per (row, head) max |out - ref| / max |ref|. The candidate must be as close to float64 as the
chunk kernel is (the chunk kernel is not exact either: bf16 probabilities and FP8 -> bf16 K/V).
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch
import triton

from sglang.srt.layers.attention.qsa import union_prefill as up
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_fwd_interface_triton_ck

RATIO = 4
BLOCK_TOPK = 512


def _case_from_dump(path: str):
    d = torch.load(path)
    dev = "cuda"
    q = d["q"].to(dev)
    k = d["k"].to(dev)
    v = d["v"].to(dev)
    if d["kv_fp8"]:
        k, v = k.view(torch.float8_e4m3fn), v.view(torch.float8_e4m3fn)
    return dict(
        name=f"{path.rsplit('/', 1)[-1]} ({len(d['extend_lens'])} seqs, {sum(d['extend_lens'])} new, "
        f"prefix {[s - e for s, e in zip(d['seq_lens'], d['extend_lens'])]})",
        q=q,
        k=k,
        v=v,
        indices=d["indices"].to(dev).to(torch.int32).contiguous(),
        extend_lens=list(d["extend_lens"]),
        seq_lens=list(d["seq_lens"]),
        scale=d["scale"],
    )


def _case_synth(bs: int, new: int, prefix: int, drift: float, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    seq = prefix + new
    extend_lens, seq_lens = [new] * bs, [seq] * bs
    heads, dim = 12, 256
    q = torch.randn(bs * new, heads, dim, device=dev, dtype=torch.bfloat16, generator=g)
    k = (torch.randn(bs * seq, 1, dim, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(bs * seq, 1, dim, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    rows = []
    positions = []
    lengths = []
    for b in range(bs):
        nblk = seq // RATIO
        shared = torch.randn(nblk, device=dev, generator=g)
        walk = torch.cumsum(torch.randn(new, nblk, device=dev, generator=g) * drift, 0)
        scores = shared[None, :] + walk
        pos = torch.arange(prefix, seq, device=dev)
        complete = (pos + 1) // RATIO
        cols = torch.arange(nblk, device=dev)
        scores = scores.masked_fill(cols[None, :] >= complete[:, None], -float("inf"))
        top = torch.topk(scores, BLOCK_TOPK, dim=1).indices.to(torch.int32)
        top = torch.where(
            torch.gather(scores, 1, top.long()) == -float("inf"), torch.full_like(top, -1), top
        )
        # fast_topk front-packs valid entries; topk on -inf rows already sorts them last
        rows.append(top)
        positions.append(pos)
        lengths.append(torch.full((new,), seq, device=dev))
    blocks = torch.cat(rows)
    indices = expand_qsa_block_indices(
        blocks, torch.cat(positions), torch.cat(lengths), RATIO, BLOCK_TOPK * RATIO
    )
    return dict(
        name=f"synth bs{bs} new{new} prefix{prefix} drift{drift}",
        q=q,
        k=k,
        v=v,
        indices=indices.contiguous(),
        extend_lens=extend_lens,
        seq_lens=seq_lens,
        scale=dim**-0.5,
    )


def _tensors(case):
    dev = "cuda"
    e, s = case["extend_lens"], case["seq_lens"]
    cu_q = torch.tensor([0] + list(np.cumsum(e)), dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(np.cumsum(s)), dtype=torch.int32, device=dev)
    kv_lens = torch.tensor(s, dtype=torch.int32, device=dev)
    width = max(s)
    r2t = torch.zeros(len(s), width + 64, dtype=torch.int32, device=dev)
    for b in range(len(s)):
        r2t[b, : s[b]] = torch.arange(int(cu_k[b]), int(cu_k[b]) + s[b], dtype=torch.int32, device=dev)
    rpi = torch.arange(len(s), dtype=torch.int64, device=dev)
    return cu_q, cu_k, kv_lens, r2t, rpi


def _reference(case, rows, cu_k_host):
    """float64 softmax(q K^T scale) V over each sampled row's selected keys."""
    q, k, v, idx = case["q"], case["k"], case["v"], case["indices"]
    e = case["extend_lens"]
    starts = np.cumsum([0] + e)
    out = []
    for r in rows:
        b = int(np.searchsorted(starts, r, side="right") - 1)
        sel = idx[r]
        sel = sel[sel >= 0].long() + int(cu_k_host[b])
        kk = k[sel, 0].double()
        vv = v[sel, 0].double()
        qq = q[r].double()
        p = torch.softmax(qq @ kk.T * case["scale"], dim=-1)
        out.append(p @ vv)
    return torch.stack(out)


def _err(out, ref):
    o = out.double()
    return ((o - ref).abs().amax(-1) / (ref.abs().amax(-1) + 1e-3)).flatten().cpu().numpy()


def _context(case, cu_k):
    """What _extend_context does: one index_select per request and plane, then cat."""
    k, v = case["k"], case["v"]
    s = case["seq_lens"]
    starts = [int(x) for x in cu_k[:-1].tolist()]
    tables = [torch.arange(st, st + n, device="cuda") for st, n in zip(starts, s)]
    kc = torch.cat([k.index_select(0, t) for t in tables])
    vc = torch.cat([v.index_select(0, t) for t in tables])
    return kc, vc


def run_case(case, configs, ref_rows):
    cu_q, cu_k, kv_lens, r2t, rpi = _tensors(case)
    q, idx = case["q"].contiguous(), case["indices"]
    e = case["extend_lens"]
    max_q = max(e)
    print(f"== {case['name']}")

    def old():
        return sparse_gqa_fwd_interface_triton_ck(
            q, case["k"], case["v"], idx, cu_q, cu_k, kv_lens, case["scale"], max_q=max_q
        )

    def old_ctx():
        kc, vc = _context(case, cu_k)
        return sparse_gqa_fwd_interface_triton_ck(q, kc, vc, idx, cu_q, cu_k, kv_lens, case["scale"], max_q=max_q)

    ref_out = old()
    torch.cuda.synchronize()
    t_old = triton.testing.do_bench(old, warmup=10, rep=60)
    t_old_ctx = triton.testing.do_bench(old_ctx, warmup=10, rep=60)
    rng = np.random.default_rng(0)
    total = sum(e)
    rows = np.sort(rng.choice(total, size=min(ref_rows, total), replace=False))
    ref = _reference(case, rows, cu_k.tolist())
    err_old = _err(ref_out[rows], ref)
    print(
        f"   chunk kernel {t_old * 1e3:7.1f} us, with context gather {t_old_ctx * 1e3:7.1f} us | "
        f"vs float64: p50 {np.quantile(err_old, .5):.2e} p99 {np.quantile(err_old, .99):.2e} "
        f"max {err_old.max():.2e}"
    )
    per_query = int((idx >= 0).sum(1).float().mean())
    for tile, nb, warps, stages in configs:
        plan = up.UnionPrefillPlan(
            e, case["seq_lens"], ratio=RATIO, topk_width=idx.shape[1], device="cuda", tile=tile
        )

        def new():
            return up.union_prefill_attention(
                plan, q, case["k"], case["v"], r2t, rpi, idx, cu_q, kv_lens, case["scale"],
                nb=nb, warps=warps, stages=stages,
            )

        try:
            out = new()
            torch.cuda.synchronize()
        except Exception as exc:  # report and continue: some configs exceed shared memory / registers
            print(f"   T{tile} NB{nb} w{warps} s{stages}: failed {type(exc).__name__}: {str(exc)[:160]}")
            continue
        counts = plan.counts.float()
        union_keys = float(counts.mean()) * RATIO
        t_new = triton.testing.do_bench(new, warmup=10, rep=60)
        t_build = triton.testing.do_bench(lambda: plan.build(idx, cu_q, r2t, rpi), warmup=10, rep=60)
        err_new = _err(out[rows], ref)
        d_old = _err(out[rows], ref_out[rows].double())
        kern = plan.last_kernel
        ptx = kern.asm.get("ptx", "")
        print(
            f"   T{tile} NB{nb} w{warps} s{stages}: regs {kern.n_regs} spills {kern.n_spills} | ptx tcgen05 "
            f"{ptx.count('tcgen05.mma')} mma.sync {ptx.count('mma.sync')} wgmma {ptx.count('wgmma')}"
        )
        print(
            f"   T{tile} NB{nb} w{warps} s{stages}: {t_new * 1e3:7.1f} us (build {t_build * 1e3:5.1f}) "
            f"x{t_old_ctx / t_new:4.2f} vs chunk+gather | union {union_keys:6.0f} keys/tile = "
            f"{union_keys / (tile * per_query):.3f} of T x {per_query} | vs float64: p50 "
            f"{np.quantile(err_new, .5):.2e} p99 {np.quantile(err_new, .99):.2e} max {err_new.max():.2e} | "
            f"vs chunk max {d_old.max():.2e} nonfinite {int((~torch.isfinite(out[:total])).sum())}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("synth")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--new", type=int, default=704)
    p.add_argument("--prefix", type=int, default=10240)
    p.add_argument("--drift", type=float, default=0.15)
    for p in sub.choices.values():
        p.add_argument("--configs", default="4,16,8,1;2,16,4,1;8,8,8,1;4,8,4,1;4,16,4,1")
        p.add_argument("--ref-rows", type=int, default=512)
    args = ap.parse_args()
    configs = [tuple(int(x) for x in c.split(",")) for c in args.configs.split(";")]
    if args.cmd == "dump":
        for f in args.files:
            run_case(_case_from_dump(f), configs, args.ref_rows)
    else:
        run_case(_case_synth(args.bs, args.new, args.prefix, args.drift), configs, args.ref_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
