#!/usr/bin/env python3
"""Prefix-extend parity gate for prefill-path opt-ins (e.g. ARC_QSA_UNION_PREFILL).

The prompt logprobs of an extend that sits on a cached prefix are what the prefix chunk prefill
computes (QSA attention over the cached context plus the new tokens), so they compare two prefill
paths directly. Serving is not batch-invariant (TP2 GDN switches kernel with the number of
sequences, the delayer changes batch composition), so the tolerance is measured, not assumed: the
same server runs the same extends in one concurrent batch and alone; A batch vs A solo is the floor.

  collect  --url URL --prompts prompts.json --out x.json [--n 32] [--extend 704] [--solo 16]
           [--passes batch,batch,solo]
           prompts.json from qwen4_exp_verify_parity.py prompts (8-16k tokens of the fork's sources).
           Per pass: flush the radix cache; warm every prefix (prompt[: P], P = (len - extend) // 64 * 64,
           the mamba checkpoint grid) with one request each, one at a time (so the cached state the solo
           extends read is the same on two servers); then the extends
           (the full prompts, max_new_tokens 1, logprob_start_len P, top-5): batch = all at once,
           solo = the first --solo one at a time. Every extend must report cached_tokens == P.
  compare  x.json:PASS y.json:PASS
  gate     x.json y.json [--slack 1.5] [--bitwise]  A = x (control), B = y (candidate): exits 1 unless
           B solo vs A solo and B batch vs A batch stay within A batch vs A solo (|dlp| quantiles x
           slack, top-1 / top-5 agreement minus 0.005) on the solo prompts. --bitwise (a path claimed
           bit-identical; both servers with the determinism switches): B solo vs A solo must be
           identical at every position (logprob and top-5 ids) and in the output token.

stdlib + numpy; does not import sglang.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from qwen4_exp_verify_parity import _flush, _post  # noqa: E402

TOP_K = 5


def _extend(url: str, prompt: List[int], prefix: int) -> dict:
    out = _post(
        f"{url}/generate",
        {
            "input_ids": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": prefix,
            "top_logprobs_num": TOP_K,
        },
    )
    meta = out["meta_info"]
    if meta["cached_tokens"] != prefix:
        raise RuntimeError(f"cached_tokens {meta['cached_tokens']} != prefix {prefix}")
    lp = meta["input_token_logprobs"]
    top = meta["input_top_logprobs"]
    # the first entry is the prefix's last token (no logprob of its own from this extend)
    return {
        "prefix": prefix,
        "input_logprobs": [x[0] for x in lp],
        "input_top_ids": [[t[1] for t in row] if row else [] for row in top],
        "output_logprob": meta["output_token_logprobs"][0][0],
        "output_id": out["output_ids"][0],
    }


def _warm(url: str, prompt: List[int]) -> None:
    _post(f"{url}/generate", {"input_ids": prompt, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}})


def collect(args) -> None:
    url = args.url.rstrip("/")
    with open(args.prompts) as fh:
        ps = json.load(fh)["prompts"][: args.n]
    prefixes = [(len(p) - args.extend) // 64 * 64 for p in ps]
    res: Dict = {"url": url, "prompts": args.prompts, "extend": args.extend, "passes": {}}
    for i, name in enumerate(args.passes.split(",")):
        _flush(url)
        # One prefix at a time: the cached KV / mamba state the extends sit on must not depend on how
        # concurrent warm-ups happened to batch (TP2 GDN switches kernel with the sequence count), or
        # two servers' solo passes are not comparable.
        for p, n in zip(ps, prefixes):
            _warm(url, p[:n])
        t = time.time()
        if name == "batch":
            with ThreadPoolExecutor(max_workers=len(ps)) as pool:
                recs = list(pool.map(lambda a: _extend(url, a[0], a[1]), zip(ps, prefixes)))
        elif name == "solo":
            recs = [_extend(url, p, n) for p, n in list(zip(ps, prefixes))[: args.solo]]
        else:
            raise ValueError(f"unknown pass {name!r}")
        key = f"{name}{sum(1 for x in args.passes.split(',')[: i + 1] if x == name)}"
        res["passes"][key] = recs
        print(f"· {key}: {len(recs)} extends in {time.time() - t:.1f} s")
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    print(f"-> {args.out}")


def _stats(xs: List[dict], ys: List[dict]) -> dict:
    n = min(len(xs), len(ys))
    dlp, top1, top5, cnt, out_diff = [], 0, 0, 0, 0
    for a, b in zip(xs[:n], ys[:n]):
        la, lb = a["input_logprobs"], b["input_logprobs"]
        if len(la) != len(lb) or a["prefix"] != b["prefix"]:
            raise RuntimeError("the two runs extended different prompts")
        for i in range(1, len(la)):
            dlp.append(abs(la[i] - lb[i]))
        for ta, tb in zip(a["input_top_ids"][1:], b["input_top_ids"][1:]):
            cnt += 1
            top1 += ta[:1] == tb[:1]
            top5 += set(ta) == set(tb)
        out_diff += a["output_logprob"] != b["output_logprob"]
    d = np.array(dlp)
    return {
        "requests": n,
        "positions": len(dlp),
        "dlp_zero_share": float((d == 0).mean()),
        "dlp_p50": float(np.quantile(d, 0.5)),
        "dlp_p90": float(np.quantile(d, 0.9)),
        "dlp_p99": float(np.quantile(d, 0.99)),
        "dlp_max": float(d.max()),
        "dlp_mean": float(d.mean()),
        "top1_agree": top1 / cnt,
        "top5_set_agree": top5 / cnt,
        "out_lp_differs": out_diff,
    }


def _fmt(label: str, s: dict) -> str:
    return (
        f"{label:<26} n={s['requests']:>3} pos {s['positions']:>6} | |dlp| zero {s['dlp_zero_share']:.3f} "
        f"p50 {s['dlp_p50']:.2e} p90 {s['dlp_p90']:.2e} p99 {s['dlp_p99']:.2e} max {s['dlp_max']:.2e} "
        f"mean {s['dlp_mean']:.2e} | top1 {s['top1_agree']:.4f} top5 set {s['top5_set_agree']:.4f} | "
        f"out lp differs {s['out_lp_differs']}"
    )


def _load(spec: str) -> List[dict]:
    path, _, key = spec.rpartition(":")
    with open(path) as fh:
        return json.load(fh)["passes"][key]


def gate(args) -> int:
    with open(args.a) as fh:
        a = json.load(fh)["passes"]
    with open(args.b) as fh:
        b = json.load(fh)["passes"]
    n = len(a["solo1"])
    rows = {
        "A batch1 vs A batch2": _stats(a["batch1"][:n], a["batch2"][:n]),
        "A batch1 vs A solo (floor)": _stats(a["batch1"][:n], a["solo1"]),
        "B batch1 vs B solo": _stats(b["batch1"][:n], b["solo1"]),
        "B solo vs A solo": _stats(a["solo1"], b["solo1"]),
        "B batch1 vs A batch1": _stats(a["batch1"][:n], b["batch1"][:n]),
        "B batch2 vs A batch2": _stats(a["batch2"][:n], b["batch2"][:n]),
        "B batch1 vs A batch1 (all)": _stats(a["batch1"], b["batch1"]),
    }
    for label, s in rows.items():
        print(_fmt(label, s))
    floor = rows["A batch1 vs A solo (floor)"]
    rc = 0
    for label in ("B solo vs A solo", "B batch1 vs A batch1", "B batch2 vs A batch2"):
        s, bad = rows[label], []
        for q in ("dlp_p50", "dlp_p90", "dlp_p99", "dlp_max", "dlp_mean"):
            if s[q] > floor[q] * args.slack:
                bad.append(f"{q} {s[q]:.2e} > floor {floor[q]:.2e} x {args.slack}")
        for q in ("top1_agree", "top5_set_agree"):
            if s[q] < floor[q] - 0.005:
                bad.append(f"{q} {s[q]:.4f} < floor {floor[q]:.4f}")
        print(f"{'✗' if bad else '✓'} {label}: " + ("; ".join(bad) if bad else "within the floor"))
        rc |= bool(bad)
    if args.bitwise:
        s = rows["B solo vs A solo"]
        exact = s["dlp_max"] == 0 and s["top5_set_agree"] == 1.0 and s["out_lp_differs"] == 0
        print(f"{'✓' if exact else '✗'} bitwise B solo vs A solo: max |dlp| {s['dlp_max']:.2e}, "
              f"top-5 sets {s['top5_set_agree']:.4f}, output logprob differs on {s['out_lp_differs']}")
        rc |= not exact
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--url", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--extend", type=int, default=704)
    p.add_argument("--solo", type=int, default=16)
    p.add_argument("--passes", default="batch,batch,solo")
    p = sub.add_parser("compare")
    p.add_argument("x")
    p.add_argument("y")
    p = sub.add_parser("gate")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--slack", type=float, default=1.5)
    p.add_argument("--bitwise", action="store_true")
    p.add_argument("--json")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        print(_fmt(f"{args.x} vs {args.y}", _stats(_load(args.x), _load(args.y))))
    else:
        return gate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
