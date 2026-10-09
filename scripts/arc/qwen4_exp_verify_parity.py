#!/usr/bin/env python3
"""Target-verify parity gate for decode-path opt-ins (e.g. ARC_QSA_FP8_DIRECT) under DFlash.

Under DFlash every output token after the first is scored by the target verify, so greedy decodes of
long prompts compare two verify paths directly: up to the first token where two runs choose
differently, both scored the same inputs, and their per-token logprobs, top-k ids and routed
experts must agree to within what serving itself moves them by. Serving is not batch-invariant, so
"what serving moves them by" is measured, not assumed: the same server decodes the same prompts in
one batch twice and alone (one request at a time); batch vs alone is the floor a candidate path is
judged against.

  prompts  --tokenizer MODEL_DIR --out prompts.json [--n 64] [--min-ctx 8192] [--max-ctx 16384]
           Disjoint slices of the fork's own python/sglang/srt sources (sorted paths), lengths spread
           over [min-ctx, max-ctx): natural text with long-range structure, so QSA selects sparsely.
           Both arms read the same file.
  collect  --url URL --prompts prompts.json --out x.json [--max-new 256] [--passes batch,batch,solo]
           [--solo 24]: per pass, flush the radix cache (every pass extends from scratch), then
           batch = all prompts at once (concurrent requests), solo = the first --solo prompts one at a time.
           Greedy; records output ids, chosen-token logprobs, top-5 (ids, logprobs), the routed
           experts of the output rows (routed_experts_start_len = prompt length), spec counters.
  compare  x.json:PASS y.json:PASS [--label L] [--json out.json]
           Per request: d = first differing output index. Over outputs 1 <= i < d (scored by the
           verify on equal inputs): |delta chosen logprob| and top-5 id-set agreement; over output
           rows j < d - 1: routed-expert agreement (per row and layer, as sets). Output 0 comes from
           the extend and is reported apart: bitwise between two solo passes (same extend, the
           control that the two servers are deterministic alike), not between batch passes (the
           concurrent prompts chunk differently; A batch1 vs A batch2 differs too).
  gate     x.json y.json   A = x (control), B = y (candidate): prints the floors (A batch vs A
           batch, A batch vs A solo; B's own) and the candidate (B vs A, batch and solo), and exits
           1 unless B-vs-A stays within the batch-vs-solo floor on every statistic (with --slack
           on the quantiles), on the floor's own requests (the solo prompts; a max over more requests
           is larger); B vs A over all prompts is printed too.

stdlib + numpy (+ transformers for `prompts`); does not import sglang.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

TOP_K = 5
FLUSH_WAIT_SECONDS = 180


def _post(url: str, payload: Optional[dict], timeout: float = 1800.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body.strip().startswith((b"{", b"[")) else body.decode()


def _flush(url: str) -> None:
    """sglang refuses /flush_cache (400) while a request is queued or running."""
    deadline = time.monotonic() + FLUSH_WAIT_SECONDS
    while True:
        try:
            _post(f"{url}/flush_cache", None)
            return
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            if time.monotonic() > deadline:
                raise RuntimeError(f"flush_cache refused for {FLUSH_WAIT_SECONDS} s: {e.read()!r}") from e
            time.sleep(2)


def prompts(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "sglang", "srt")
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    if not files:
        raise RuntimeError(f"no sources under {root}")
    lens = [args.min_ctx + (i * 2731) % (args.max_ctx - args.min_ctx) for i in range(args.n)]
    need = sum(lens)
    ids: List[int] = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            text = f"# file: {os.path.relpath(f, root)}\n" + fh.read() + "\n"
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(f"corpus has {len(ids)} tokens, the prompts need {need}")
    out, off = [], 0
    for n in lens:
        out.append(ids[off : off + n])
        off += n
    with open(args.out, "w") as fh:
        json.dump({"tokenizer": args.tokenizer, "lens": lens, "prompts": out}, fh)
    print(f"{len(out)} prompts, {min(lens)}-{max(lens)} tokens, {need} total -> {args.out}")


def _record(result: dict, prompt_len: int) -> dict:
    meta = result["meta_info"]
    out_ids = result["output_ids"]
    if meta["prompt_tokens"] != prompt_len:
        raise RuntimeError(f"prompt_tokens {meta['prompt_tokens']} != {prompt_len}")
    lps = [x[0] for x in meta["output_token_logprobs"]]
    if len(lps) != len(out_ids):
        raise RuntimeError(f"{len(lps)} output logprobs for {len(out_ids)} output ids")
    top = meta["output_top_logprobs"]
    rec = {
        "output_ids": out_ids,
        "output_logprobs": lps,
        "top_ids": [[t[1] for t in row] for row in top],
        "top_logprobs": [[t[0] for t in row] for row in top],
        "spec": {k: meta[k] for k in ("spec_verify_ct", "spec_accept_length")},
        "cached_tokens": meta.get("cached_tokens"),
        "finish": meta.get("finish_reason"),
    }
    if "routed_experts" not in meta:
        raise RuntimeError("no routed_experts in the response (server needs --enable-return-routed-experts)")
    m = meta["routed_experts_meta"]
    if int(m["start"]) != prompt_len:
        raise RuntimeError(f"routed experts start {m['start']} != prompt length {prompt_len}")
    if m["shape"][0] != len(out_ids) - 1:
        raise RuntimeError(f"routed experts rows {m['shape'][0]} != {len(out_ids) - 1} output rows")
    rec["routed"] = {"dtype": m["dtype"], "shape": m["shape"], "b64": meta["routed_experts"]}
    return rec


def _generate(url: str, prompt: List[int], max_new: int) -> dict:
    """One request. routed_experts_start_len is one int per /generate (a batched /generate applies it
    to every prompt), so a batch is sent as concurrent single requests; the scheduler batches them."""
    payload = {
        "input_ids": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": TOP_K,
        "return_routed_experts": True,
        "routed_experts_start_len": len(prompt),
    }
    out = _post(f"{url}/generate", payload)
    if isinstance(out, list):
        raise RuntimeError(f"{len(out)} results for one prompt")
    return _record(out, len(prompt))


def collect(args) -> None:
    url = args.url.rstrip("/")
    with open(args.prompts) as fh:
        ps = json.load(fh)["prompts"]
    info = json.loads(urllib.request.urlopen(f"{url}/server_info", timeout=60).read())
    res: Dict[str, Any] = {"url": url, "prompts": args.prompts, "max_new": args.max_new, "passes": {}}
    res["server_args"] = {k: info.get(k) for k in ("speculative_algorithm", "random_seed", "tp_size", "model_path")}
    for i, name in enumerate(args.passes.split(",")):
        _flush(url)
        t = time.time()
        if name == "batch":
            with ThreadPoolExecutor(max_workers=len(ps)) as pool:
                recs = list(pool.map(lambda p: _generate(url, p, args.max_new), ps))
        elif name == "solo":
            recs = [_generate(url, p, args.max_new) for p in ps[: args.solo]]
        else:
            raise ValueError(f"unknown pass {name!r}")
        key = f"{name}{sum(1 for x in args.passes.split(',')[: i + 1] if x == name)}"
        res["passes"][key] = recs
        acc = np.mean([r["spec"]["spec_accept_length"] for r in recs])
        print(f"· {key}: {len(recs)} requests x {args.max_new} tokens in {time.time() - t:.1f} s, accept {acc:.3f}")
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    print(f"-> {args.out}")


def _routed(rec: dict) -> np.ndarray:
    r = rec["routed"]
    return np.frombuffer(base64.b64decode(r["b64"]), dtype=np.dtype(r["dtype"])).reshape(r["shape"])


def _pair_stats(xs: List[dict], ys: List[dict]) -> dict:
    n = min(len(xs), len(ys))
    ds, dlp, top_eq, top_n, r_diff, r_n, out0 = [], [], 0, 0, 0, 0, 0
    acc_x, acc_y = [], []
    for a, b in zip(xs[:n], ys[:n]):
        ia, ib = a["output_ids"], b["output_ids"]
        m = min(len(ia), len(ib))
        d = next((i for i in range(m) if ia[i] != ib[i]), m)
        ds.append(d / m)
        if a["output_logprobs"][0] != b["output_logprobs"][0]:
            out0 += 1
        for i in range(1, d):
            dlp.append(abs(a["output_logprobs"][i] - b["output_logprobs"][i]))
            top_n += 1
            top_eq += set(a["top_ids"][i]) == set(b["top_ids"][i])
        ra, rb = _routed(a), _routed(b)
        rows = min(d - 1, ra.shape[0], rb.shape[0])
        if rows > 0:
            sa, sb = np.sort(ra[:rows], axis=-1), np.sort(rb[:rows], axis=-1)
            r_diff += int((sa != sb).any(axis=-1).sum())
            r_n += rows * ra.shape[1]
        acc_x.append(a["spec"]["spec_accept_length"])
        acc_y.append(b["spec"]["spec_accept_length"])
    dlp_a = np.array(dlp) if dlp else np.zeros(1)
    ds_a = np.array(ds)
    return {
        "requests": n,
        "identical": int(sum(1 for d in ds if d == 1.0)),
        "d_frac_median": float(np.median(ds_a)),
        "d_frac_mean": float(ds_a.mean()),
        "scored": len(dlp),
        "dlp_zero_share": float((dlp_a == 0).mean()),
        "dlp_p50": float(np.quantile(dlp_a, 0.5)),
        "dlp_p90": float(np.quantile(dlp_a, 0.9)),
        "dlp_p99": float(np.quantile(dlp_a, 0.99)),
        "dlp_max": float(dlp_a.max()),
        "dlp_mean": float(dlp_a.mean()),
        "top5_set_agree": top_eq / top_n if top_n else float("nan"),
        "routed_rowlayer_diff": r_diff / r_n if r_n else float("nan"),
        "out0_lp_differs": out0,
        "accept_x": float(np.mean(acc_x)),
        "accept_y": float(np.mean(acc_y)),
    }


def _load(spec: str) -> List[dict]:
    path, _, key = spec.rpartition(":")
    with open(path) as fh:
        return json.load(fh)["passes"][key]


def _fmt(label: str, s: dict) -> str:
    return (
        f"{label:<28} n={s['requests']:>3} identical {s['identical']:>3}  first-diff at {s['d_frac_median']:.2f}"
        f" (mean {s['d_frac_mean']:.2f}) of the output | {s['scored']:>6} scored: |dlp| zero {s['dlp_zero_share']:.3f}"
        f" p50 {s['dlp_p50']:.2e} p90 {s['dlp_p90']:.2e} p99 {s['dlp_p99']:.2e} max {s['dlp_max']:.2e}"
        f" | top5 set {s['top5_set_agree']:.4f} | routed row-layer diff {s['routed_rowlayer_diff']:.4f}"
        f" | out0 lp differs {s['out0_lp_differs']} | accept {s['accept_x']:.3f}/{s['accept_y']:.3f}"
    )


def compare(args) -> None:
    s = _pair_stats(_load(args.x), _load(args.y))
    print(_fmt(args.label or f"{args.x} vs {args.y}", s))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(s, fh, indent=1)


def gate(args) -> int:
    with open(args.a) as fh:
        a = json.load(fh)["passes"]
    with open(args.b) as fh:
        b = json.load(fh)["passes"]
    n_solo = len(a["solo1"])
    rows = {
        "A batch1 vs A batch2": _pair_stats(a["batch1"], a["batch2"]),
        "A batch1 vs A solo (floor)": _pair_stats(a["batch1"][:n_solo], a["solo1"]),
        "B batch1 vs B batch2": _pair_stats(b["batch1"], b["batch2"]),
        "B batch1 vs B solo": _pair_stats(b["batch1"][:n_solo], b["solo1"]),
        "B batch1 vs A batch1 (all)": _pair_stats(a["batch1"], b["batch1"]),
        "B batch2 vs A batch2 (all)": _pair_stats(a["batch2"], b["batch2"]),
        # gated on the floor's own requests (the solo prompts): a max over more requests is larger
        "B solo vs A solo": _pair_stats(a["solo1"], b["solo1"]),
        "B batch1 vs A batch1": _pair_stats(a["batch1"][:n_solo], b["batch1"][:n_solo]),
        "B batch2 vs A batch2": _pair_stats(a["batch2"][:n_solo], b["batch2"][:n_solo]),
    }
    for label, s in rows.items():
        print(_fmt(label, s))
    floor = rows["A batch1 vs A solo (floor)"]
    rc = 0
    for label in ("B solo vs A solo", "B batch1 vs A batch1", "B batch2 vs A batch2"):
        s = rows[label]
        bad = []
        for q in ("dlp_p50", "dlp_p90", "dlp_p99", "dlp_max"):
            if s[q] > floor[q] * args.slack:
                bad.append(f"{q} {s[q]:.2e} > floor {floor[q]:.2e} x {args.slack}")
        if s["top5_set_agree"] < floor["top5_set_agree"] - args.agree_slack:
            bad.append(f"top5 {s['top5_set_agree']:.4f} < floor {floor['top5_set_agree']:.4f}")
        if s["routed_rowlayer_diff"] > floor["routed_rowlayer_diff"] + args.agree_slack:
            bad.append(f"routed {s['routed_rowlayer_diff']:.4f} > floor {floor['routed_rowlayer_diff']:.4f}")
        if s["d_frac_mean"] < floor["d_frac_mean"] - args.d_slack:
            bad.append(f"first-diff mean {s['d_frac_mean']:.2f} < floor {floor['d_frac_mean']:.2f}")
        # output 0 comes from the extend: bitwise only where both extends ran alone (solo); a batch
        # extend depends on how the concurrent prompts were chunked (A batch1 vs A batch2 differs too)
        if label.startswith("B solo") and s["out0_lp_differs"]:
            bad.append(f"output 0 (extend) logprob differs on {s['out0_lp_differs']} solo requests")
        print(f"{'✗' if bad else '✓'} {label}: " + ("; ".join(bad) if bad else "within the floor"))
        rc |= bool(bad)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--min-ctx", type=int, default=8192)
    p.add_argument("--max-ctx", type=int, default=16384)
    p = sub.add_parser("collect")
    p.add_argument("--url", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--passes", default="batch,batch,solo")
    p.add_argument("--solo", type=int, default=24)
    p = sub.add_parser("compare")
    p.add_argument("x")
    p.add_argument("y")
    p.add_argument("--label")
    p.add_argument("--json")
    p = sub.add_parser("gate")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--slack", type=float, default=1.5)
    p.add_argument("--agree-slack", type=float, default=0.005)
    p.add_argument("--d-slack", type=float, default=0.1)
    p.add_argument("--json")
    args = ap.parse_args()
    if args.cmd == "prompts":
        prompts(args)
    elif args.cmd == "collect":
        collect(args)
    elif args.cmd == "compare":
        compare(args)
    elif args.cmd == "gate":
        return gate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
