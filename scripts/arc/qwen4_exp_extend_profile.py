#!/usr/bin/env python3
"""Where the GPU time of a large Qwen4-Exp extend batch goes (TP2 FP8 DFlash, the RL deployment shape).

  drive    --url --log --out-dir [--shapes a,b] [--reps 6] [--profile-reps 2] [--interval 10]
           [--expect-graph true|false]
           Against one DFlash server with ARC_EXTEND_TELEMETRY=1, ARC_EXTEND_TELEMETRY_INTERVAL_S=<interval>
           and SGLANG_ENABLE_COLOCATED_BATCH_GEN=1 (a batched /generate with images then reaches the
           scheduler as one group instead of one request per media processing step). Per shape (SHAPES):
             1. warm the shape's cached prefixes (one request each, cached_tokens checked later) and
                run one unmeasured batch of the shape (the first pays one-time compiles);
             2. telemetry: --reps batches, each ONE /generate of k prompts = prefix_i + a fresh new
                segment (text, or a new 512x512 image + text), one output token; an idle interval and
                a 17-token trigger flush the telemetry window; the k4+ bin of the windows logged
                after the shape started must hold exactly --reps extends;
             3. profile: /start_profile (CPU+GPU, stack, shapes), --profile-reps more batches,
                /stop_profile; traces <out-dir>/<shape>/<shape>-*-TP-{0,1}-*.trace.json.gz.
           Every batch must be one prefill batch in server.log (#new-seq k, the new and cached token
           sums), every request's cached_tokens must equal its prefix, and the log's
           "cuda graph" verdict must equal --expect-graph. Writes <out-dir>/drive.json.
  analyze  TRACE_TP0 [--peer TRACE_TP1] [--seqs 4] [--tokens 2800] [--out x.json] [--map eager.json]
           Kernel breakdown of every target extend step (gpu span step[EXTEND bs=<seqs> toks=<tokens>]
           up to the next draft_extend span) and of the drafter refresh (the draft_extend span):
           per category kernel time, calls, union busy and GPU idle inside the window. Eager traces
           attribute kernels by their launch's module stack (nn.Module / record_function frames);
           CUDA-graph replays launch from cudaGraphLaunch, so their kernels take the category
           mix that kernel name had in --map (the eager arm's analyze --out of the same shape).
           With --peer: the k-th all-reduce of TP0 and TP1 paired, min(d0, d1) = transfer, the
           rest = waiting for the other rank.
  roofline [--shapes]  per category floor (FLOPs / peak vs bytes / HBM) of each shape, per rank.
  table    ARM_DIR ...  per shape and arm: telemetry (no profiler), the trace's grouped kernel time
           (all-reduce as the paired transfer estimate) and the roofline.

stdlib + numpy; does not import sglang.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_exp_prefill_bcg_check as chk  # noqa: E402

# (prefix tokens, new tokens, new images) per request. Prefixes are multiples of 64 (the mamba
# checkpoint grid), so the follow-up hits the whole prefix. A new image is 258 of the new tokens.
# The H1 B arm's k>=4 extends of 2048-4095 new tokens: 4.9 seqs, 2885 new, 43.7k cached in total
# (8.9k per request), so p10k is the production shape; p45k / pmix stretch the prefix.
SHAPES: Dict[str, List[Tuple[int, int, int]]] = {
    "cold": [(0, 704, 0)] * 4,
    "p10k": [(10240, 704, 0)] * 4,
    "p10k_img": [(10240, 704, 1)] * 4,
    "p10k_352": [(10240, 352, 0)] * 4,
    "p10k_512": [(10240, 512, 0)] * 4,
    "p10k_1408": [(10240, 1408, 0)] * 4,
    "pmix": [(10240, 704, 0), (20480, 704, 0), (30720, 704, 0), (46080, 704, 0)],
    "p45k": [(46080, 704, 0)] * 4,
    "pmix_img": [(10240, 704, 1), (20480, 704, 1), (30720, 704, 1), (46080, 704, 1)],
}
# p10k_512: 2048 tokens, the largest flashinfer MoE autotune bucket the startup tuning caches (2816 tokens
# fall in the untuned 3072 bucket and run flashinfer's default tactic)
DEFAULT_SHAPES = "p10k,p10k_img,cold,p10k_352,p10k_512,p10k_1408,pmix,p45k"
TRIGGER_TOKENS = 17

_PREFILL = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[^\]]*\] Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), "
    r"#cached-token: (\d+).*?cuda graph: (True|False)"
)
_TELEMETRY = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[^\]]*\] ARC extend telemetry \[")


def _bin_re(key: str):
    return re.compile(
        re.escape(f"| {key} ") + r"n=(\d+) gpu=([0-9.]+)/([0-9.]+) cpu=([0-9.]+)/([0-9.]+) tok=(\d+) pfx=(\d+)"
    )


def _telemetry_key(seqs: int, tokens: int) -> str:
    req = f"k{seqs}" if seqs <= 3 else "k4+"
    for edge in (512, 1024, 2048, 4096):
        if tokens < edge:
            return f"{req}/<{edge}"
    return f"{req}/>=4096"


def _log_ts(s: str) -> float:
    import datetime

    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()


class _Log:
    """server.log read from an offset: the prefill-batch lines since the last mark."""

    def __init__(self, path: str):
        self.path = path
        self.offset = os.path.getsize(path)

    def mark(self) -> None:
        self.offset = os.path.getsize(self.path)

    def new_lines(self) -> List[str]:
        with open(self.path, errors="replace") as f:
            f.seek(self.offset)
            return f.read().splitlines()


def _prompt(text: chk._Text, seed: int, prefix: str, new_tokens: int, images: int) -> dict:
    n_text = new_tokens - images * chk.MM_IMAGE_TOKENS
    if n_text < 0:
        raise ValueError(f"{images} images do not fit in {new_tokens} new tokens")
    parts = [(text(seed, n_text), n_text)] + [("image", seed + 1 + i) for i in range(images)]
    p = chk._mm_prompt(*parts)
    return {"text": prefix + p["text"], "images": p["images"], "tokens": p["tokens"]}


def _send(url: str, prompts: List[dict]) -> List[dict]:
    """One /generate of len(prompts) prompts, one output token each, routed experts requested
    (the RL ledger records routing: the capture runs in production); no prompt logprobs."""
    payload = {
        "text": [p["text"] for p in prompts],
        "sampling_params": [{"temperature": 0.0, "max_new_tokens": 1} for _ in prompts],
        "return_routed_experts": True,
    }
    if any(p["images"] for p in prompts):
        payload["image_data"] = [p["images"] or None for p in prompts]
    out = chk._post(f"{url}/generate", payload)
    if not isinstance(out, list) or len(out) != len(prompts):
        raise RuntimeError(f"/generate returned {type(out)} for {len(prompts)} prompts")
    return out


def _check_batch(
    log: _Log, shape: str, spec, out: List[dict], prompts: List[dict], expect_graph: Optional[bool]
) -> dict:
    for (prefix, new, _), p, r in zip(spec, prompts, out, strict=True):
        meta = r["meta_info"]
        if p["tokens"] != new:  # p["tokens"]: the new segment only (the prefix text is prepended)
            raise RuntimeError(f"{shape}: new segment of {p['tokens']} tokens, expected {new}")
        if meta["prompt_tokens"] != prefix + p["tokens"]:
            raise RuntimeError(f"{shape}: prompt_tokens {meta['prompt_tokens']} != {prefix} + {p['tokens']}")
        if meta["cached_tokens"] != prefix:
            raise RuntimeError(f"{shape}: cached_tokens {meta['cached_tokens']} != prefix {prefix}")
    time.sleep(1.0)
    batches = [m.groups() for m in map(_PREFILL.search, log.new_lines()) if m]
    want = (len(spec), sum(s[1] for s in spec), sum(s[0] for s in spec))
    got = [(int(a), int(b), int(c)) for _, a, b, c, _ in batches]
    if got != [want]:
        raise RuntimeError(f"{shape}: prefill batches {got} in server.log, expected exactly [{want}]")
    graph = batches[0][4] == "True"
    if expect_graph is not None and graph != expect_graph:
        raise RuntimeError(f"{shape}: prefill ran with cuda graph {graph}, expected {expect_graph}")
    return {"graph": graph}


def _wait_traces(out_dir: str, prefix: str, timeout: float = 600.0) -> List[str]:
    """Both ranks' traces, once their sizes stop changing."""
    deadline = time.monotonic() + timeout
    last = None
    stable = 0
    while time.monotonic() < deadline:
        files = sorted(glob.glob(os.path.join(out_dir, f"{prefix}-*-TP-*.trace.json.gz")))
        sizes = tuple((f, os.path.getsize(f)) for f in files)
        if len(files) == 2 and sizes == last:
            stable += 1
            if stable >= 3:
                return files
        else:
            stable = 0
        last = sizes
        time.sleep(3)
    raise RuntimeError(f"no stable TP0+TP1 traces {prefix}-* in {out_dir} after {timeout} s (have {last})")


def _telemetry_bins(log_path: str, since: float, key: str) -> dict:
    """Sum of the `key` bin over the telemetry windows logged at or after `since`."""
    pat = _bin_re(key)
    tot = {"n": 0, "target_gpu": 0.0, "refresh_gpu": 0.0, "target_cpu": 0.0, "refresh_cpu": 0.0, "tok": 0.0, "pfx": 0.0}
    for line in open(log_path, errors="replace"):
        m = _TELEMETRY.search(line)
        if not m or _log_ts(m.group(1)) < since - 1:
            continue
        b = pat.search(line)
        if b is None:
            continue
        n = int(b.group(1))
        tot["n"] += n
        for i, k in enumerate(("target_gpu", "refresh_gpu", "target_cpu", "refresh_cpu", "tok", "pfx"), start=2):
            tot[k] += n * float(b.group(i))
    if tot["n"]:
        for k in ("target_gpu", "refresh_gpu", "target_cpu", "refresh_cpu", "tok", "pfx"):
            tot[k] /= tot["n"]
    return tot


def drive(args) -> None:
    url = args.url.rstrip("/")
    expect_graph = {"true": True, "false": False, None: None}[args.expect_graph]
    shapes = args.shapes.split(",")
    for s in shapes:
        if s not in SHAPES:
            raise SystemExit(f"unknown shape {s}; have {sorted(SHAPES)}")
    os.makedirs(args.out_dir, exist_ok=True)
    text = chk._Text(url)
    chk._flush(url)
    log = _Log(args.log)
    prefixes: Dict[Tuple[int, int], str] = {}  # (request index, length) -> text, warmed once
    result = {"url": url, "reps": args.reps, "profile_reps": args.profile_reps, "shapes": {}}
    seed = 100000
    for shape in shapes:
        spec = SHAPES[shape]
        # 1. warm the prefixes this shape needs (cached_tokens of every batch below proves the hit)
        for i, (plen, _, _) in enumerate(spec):
            if plen and (i, plen) not in prefixes:
                prefixes[(i, plen)] = text(7000 + 97 * i + plen, plen)
                chk._post(
                    f"{url}/generate",
                    {"text": prefixes[(i, plen)], "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}},
                )
        graph = set()

        def batch() -> None:
            nonlocal seed
            prompts = []
            for i, (plen, new, images) in enumerate(spec):
                prompts.append(_prompt(text, seed, prefixes.get((i, plen), ""), new, images))
                seed += 10
            log.mark()
            out = _send(url, prompts)
            graph.add(_check_batch(log, shape, spec, out, prompts, expect_graph)["graph"])

        # one unmeasured batch of the shape: the first one pays one-time compiles (r1 p10k, both arms:
        # ~1.8 s for the first k=4 batch vs ~77 ms after)
        batch()
        # flush the telemetry window that holds the warm-ups (and the previous shape)
        time.sleep(args.interval + 2)
        chk._post(f"{url}/generate", {"text": text(seed, TRIGGER_TOKENS), "sampling_params": {"max_new_tokens": 1}})
        seed += 1
        time.sleep(2)
        t_start = time.time()

        # 2. telemetry
        for _ in range(args.reps):
            batch()
        time.sleep(args.interval + 2)
        chk._post(f"{url}/generate", {"text": text(seed, TRIGGER_TOKENS), "sampling_params": {"max_new_tokens": 1}})
        seed += 1
        time.sleep(3)
        key = _telemetry_key(len(spec), sum(s[1] for s in spec))
        tel = _telemetry_bins(args.log, t_start, key)
        if tel["n"] != args.reps:
            raise RuntimeError(f"{shape}: telemetry bin {key} holds {tel['n']} extends since the shape began, expected {args.reps}")
        print(
            f"{shape}: {len(spec)} x {[s[1] for s in spec]} new on prefixes {[s[0] for s in spec]} "
            f"({sum(s[2] for s in spec)} new images): target {tel['target_gpu']:.1f} ms GPU / {tel['target_cpu']:.1f} ms CPU, "
            f"refresh {tel['refresh_gpu']:.1f} ms GPU / {tel['refresh_cpu']:.1f} ms CPU (n={tel['n']}, bin {key}, "
            f"cuda graph {sorted(graph)})",
            flush=True,
        )
        # 3. profile
        prof_dir = os.path.join(args.out_dir, shape)
        traces = []
        if args.profile_reps:
            os.makedirs(prof_dir, exist_ok=True)
            chk._post(
                f"{url}/start_profile",
                {
                    "output_dir": prof_dir,
                    "activities": ["CPU", "GPU"],
                    "profile_prefix": shape,
                    "with_stack": True,
                    "record_shapes": True,
                },
            )
            for _ in range(args.profile_reps):
                batch()
            chk._post(f"{url}/stop_profile", None)
            traces = _wait_traces(prof_dir, shape)
        result["shapes"][shape] = {
            "spec": spec,
            "telemetry_key": key,
            "telemetry": tel,
            "graph": sorted(graph),
            "traces": traces,
        }
        json.dump(result, open(os.path.join(args.out_dir, "drive.json"), "w"), indent=1)
    print(f"wrote {os.path.join(args.out_dir, 'drive.json')}")


# ---------------------------------------------------------------------------------------------------
# trace analysis

_MODULE_SUFFIX = re.compile(r"_\d+$")
_ALLREDUCE = re.compile(r"allreduce|all_reduce|AllReduce|nccl|cross_device_reduce|twoshot|oneshot|lamport", re.I)

# (category, module-path regex, kernel-name regex), first match wins. The module path is the
# launch's stack of nn.Module classes and record_function names ("@name"), outermost first, joined
# by "/". A rule with kernel regex None matches any kernel.
RULES: List[Tuple[str, Optional[str], Optional[str]]] = [
    ("allreduce", None, _ALLREDUCE.pattern),
    ("ViT/mm embed", r"Qwen3VLMoeVisionModel|@sglang\.vlm\.mm_embedding", None),
    ("refresh: drafter", r"@draft_extend", None),
    ("MoE routed: FC1", r"FusedMoE", r"^bmm_E4m3_E4m3E4m3|^bmm_.*E4m3E4m3.*t128x64x128u2"),
    ("MoE routed: FC2", r"FusedMoE", r"^bmm_Bfloat16"),
    ("MoE routed: activation", r"FusedMoE", r"activation"),
    ("MoE routed: finalize", r"FusedMoE", r"finalize"),
    ("MoE routed: routing", r"FusedMoE", r"routing|Histogram|Coop"),
    ("MoE routed: other", r"FusedMoE", None),
    ("MoE router (topk)", r"/TopK", None),
    ("MoE shared expert", r"Qwen2MoeMLP", None),
    ("MoE gate/misc", r"Qwen2MoeSparseMoeBlock", None),
    ("QSA indexer", r"QSAIndexer", None),
    ("QSA attention core", r"Qwen4ExpAttentionDecoderLayer.*RadixAttention", None),
    ("QSA proj (dense)", r"Qwen4ExpAttentionDecoderLayer.*(QKVParallelLinear|RowParallelLinear|ColumnParallelLinear)", None),
    ("GDN core", r"RadixLinearAttention", None),
    ("GDN proj (dense)", r"Qwen3_5GatedDeltaNet.*(MergedColumnParallelLinear|RowParallelLinear|ColumnParallelLinear|QKVParallelLinear)", None),
    ("PLE", r"Qwen4ExpPLE", None),
    ("norm", r"GroupedGemmaRMSNorm|/RMSNorm$", None),
    ("QSA misc", r"Qwen4ExpAttentionDecoderLayer/(?!.*Qwen2Moe)[A-Za-z]", None),
    ("GDN misc", r"Qwen3_5GatedDeltaNet", None),
    ("HC/layer level", r"Qwen4Exp(Linear|Attention)DecoderLayer", None),
    ("LM head/logits", r"LogitsProcessor", None),
    ("sampling", r"Sampler", None),
    ("embedding", r"VocabParallelEmbedding", None),
]
_RULES = [(c, re.compile(m) if m else None, re.compile(k) if k else None) for c, m, k in RULES]


def _categorize(path: str, kernel: str) -> str:
    for cat, mre, kre in _RULES:
        if mre is not None and not mre.search(path):
            continue
        if kre is not None and not kre.search(kernel):
            continue
        return cat
    tail = path.rsplit("/", 2)[-2:] if path else ["(no module)"]
    return "other: " + "/".join(tail)


_FAMILIES = [
    ("allreduce", _ALLREDUCE.pattern),
    ("moe", r"^bmm_|moe::|routing|finalize"),
    ("attention", r"fmha|attention|attn|flash|sparse_gqa|paged"),
    ("gemm", r"nvjet|cutlass|gemm|Gemm|cublas|deep_gemm|sm100_fp8|splitK"),
    ("norm", r"norm"),
    ("copy/fill", r"Memcpy|Memset|copy|fill|Fill|index|gather|scatter|cat_"),
    ("elementwise", r"elementwise|triton_poi|triton_red|act_and_mul|silu|quant"),
]
_FAMILIES_RE = [(f, re.compile(r)) for f, r in _FAMILIES]


def _family(kernel: str) -> str:
    for fam, r in _FAMILIES_RE:
        if r.search(kernel):
            return fam
    return "other"


def _short(name: str) -> str:
    name = re.sub(r"\(.*", "", name) if not name.startswith(("bmm_", "nvjet")) else name
    return name[:110]


def _load(path: str) -> list:
    return json.load(gzip.open(path))["traceEvents"]


def _launch_stacks(events: list) -> Dict[int, str]:
    """correlation id -> the launch's module path (nn.Module classes and record_function names)."""
    by_tid: Dict[Tuple[int, int], list] = collections.defaultdict(list)
    for e in events:
        cat = e.get("cat")
        if cat == "python_function":
            name = e["name"]
            if name.startswith("nn.Module: "):
                by_tid[(e["pid"], e["tid"])].append((e["ts"], 0, -(e["ts"] + e["dur"]), _MODULE_SUFFIX.sub("", name[11:])))
        elif cat == "user_annotation":
            name = e["name"]
            if name.startswith("## Call CompiledFxGraph"):
                name = "CompiledFxGraph"
            by_tid[(e["pid"], e["tid"])].append((e["ts"], 0, -(e["ts"] + e["dur"]), "@" + name))
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                by_tid[(e["pid"], e["tid"])].append((e["ts"], 1, 0, corr))
    paths: Dict[int, str] = {}
    for items in by_tid.values():
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        stack: List[Tuple[float, str]] = []
        for ts, kind, neg_end, val in items:
            while stack and stack[-1][0] <= ts:
                stack.pop()
            if kind == 0:
                stack.append((-neg_end, val))
            else:
                paths[val] = "/".join(n for _, n in stack)
    return paths


def _union(intervals: List[Tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _merged(spans: list, lo: float, hi: float) -> List[Tuple[float, float]]:
    """GPU spans of one name that start in [lo, hi), merged when they overlap: kineto emits one
    gpu_user_annotation per CUDA stream the CPU range launched on."""
    out: List[List[float]] = []
    for e in sorted((e for e in spans if lo <= e["ts"] < hi), key=lambda e: e["ts"]):
        if out and e["ts"] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e["ts"] + e["dur"])
        else:
            out.append([e["ts"], e["ts"] + e["dur"]])
    return [tuple(x) for x in out]


def _windows(events: list, seqs: int, tokens: int, spec: bool = True) -> List[dict]:
    """GPU windows of every target extend step of the shape and of its drafter refresh (without
    speculative decoding: the step span alone, no refresh)."""
    ann = [e for e in events if e.get("cat") == "gpu_user_annotation"]
    by_name = collections.defaultdict(list)
    for e in ann:
        by_name[e["name"]].append(e)
    step_name = re.compile(rf"^step\[EXTEND bs={seqs} toks={tokens}[\] ]")
    step_spans = [e for n, v in by_name.items() if step_name.match(n) for e in v]
    steps = _merged(step_spans, float("-inf"), float("inf"))
    if not steps:
        names = collections.Counter(e["name"] for e in ann if e["name"].startswith("step[EXTEND"))
        raise RuntimeError(f"no gpu span step[EXTEND bs={seqs} toks={tokens}]; extend spans: {dict(names)}")
    refresh = _merged(by_name.get("draft_extend", []), float("-inf"), float("inf"))
    out = []
    for s0, s1 in steps:
        if spec:
            after = [r for r in refresh if r[0] >= s0]
            if not after:
                raise RuntimeError(f"no draft_extend span after the extend step at {s0}")
            r = after[0]
            end = r[0]
        else:
            r = None
            end = s1
        mm = _merged(by_name.get("sglang.vlm.mm_embedding", []), s0, end)
        lm = _merged(by_name.get("sglang.vlm.language_model_prefill", []), s0, end)
        if len(lm) != 1 or len(mm) > 1:
            raise RuntimeError(f"extend step at {s0}: language-model spans {lm}, media spans {mm}")
        out.append({"step": (s0, s1), "target": (s0, end), "refresh": r, "lm": lm[0], "mm": mm[0] if mm else None})
    return out


def _gpu_ops(events: list) -> list:
    ops = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ops.sort(key=lambda e: e["ts"])
    return ops


def _in(ops: list, starts: List[float], lo: float, hi: float) -> list:
    i = bisect.bisect_left(starts, lo)
    j = bisect.bisect_left(starts, hi)
    return ops[i:j]


def _breakdown(
    ops: list, paths: Dict[int, str], name_map: Optional[dict], lo: float, hi: float, prefix: Optional[str] = None
) -> dict:
    """Kernel time per category inside [lo, hi) (GPU us), union busy, idle, top kernels. With
    `prefix` every kernel is "<prefix>: <kernel family>" (the drafter refresh: its kernel names
    also occur in the target, so neither module paths of graph replays nor a name map apply)."""
    cats: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: {"us": 0.0, "calls": 0})
    kernels: Dict[Tuple[str, str], Dict[str, float]] = collections.defaultdict(lambda: {"us": 0.0, "calls": 0})
    by_name: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    intervals = []
    graph_us = 0.0
    mapped_split_us = 0.0
    unmapped = collections.Counter()
    for e in ops:
        s, d = e["ts"], e["dur"]
        if s >= hi:
            break
        d = min(d, hi - s)
        intervals.append((s, s + d))
        name = e["name"]
        args = e.get("args", {})
        in_graph = args.get("graph id", 0) not in (0, None)
        if in_graph:
            graph_us += d
        if prefix is not None:
            parts = {f"{prefix}: {_family(name)}": 1.0}
        else:
            # a graph replay launches from cudaGraphLaunch: its module path says nothing
            cat = None if in_graph and not _ALLREDUCE.search(name) else _categorize(
                paths.get(args.get("correlation"), ""), name
            )
            if cat is not None and (not cat.startswith("other: ") or name_map is None or name not in name_map):
                parts = {cat: 1.0}
            elif name_map is None:
                raise RuntimeError(f"graph kernel {name[:80]} in an extend window: analyze the eager arm first, pass --map")
            elif name not in name_map:
                unmapped[_short(name)] += d
                parts = {"unmapped (graph kernel)": 1.0}
            else:
                parts = name_map[name]
                if len(parts) > 1:
                    mapped_split_us += d
        for cat, w in parts.items():
            cats[cat]["us"] += w * d
            cats[cat]["calls"] += w
            kernels[(cat, _short(name))]["us"] += w * d
            kernels[(cat, _short(name))]["calls"] += w
            by_name[name][cat] += w * d
    busy = _union(intervals)
    top = collections.defaultdict(list)
    for (cat, k), v in sorted(kernels.items(), key=lambda kv: -kv[1]["us"]):
        if len(top[cat]) < 4:
            top[cat].append((k, round(v["us"] / 1e3, 3), round(v["calls"], 1)))
    per_kernel = collections.defaultdict(lambda: [0.0, 0])
    for e in ops:
        if e["ts"] >= hi:
            break
        k = per_kernel[_short(e["name"])]
        k[0] += min(e["dur"], hi - e["ts"]) / 1e3
        k[1] += 1
    return {
        "window_ms": (hi - lo) / 1e3,
        "busy_ms": busy / 1e3,
        "kernels": dict(sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:60]),
        "idle_ms": (hi - lo - busy) / 1e3,
        "kernel_sum_ms": sum(v["us"] for v in cats.values()) / 1e3,
        "graph_ms": graph_us / 1e3,
        "graph_split_ms": mapped_split_us / 1e3,
        "unmapped_graph_kernels": {k: round(v / 1e3, 3) for k, v in unmapped.most_common(10)},
        "categories": {c: {"ms": v["us"] / 1e3, "calls": v["calls"]} for c, v in cats.items()},
        "top": dict(top),
        "by_name": {n: dict(v) for n, v in by_name.items()},
    }


def _allreduce_pairs(ops0: list, ops1: list, w0: Tuple[float, float], w1: Tuple[float, float]) -> dict:
    a0 = [e["dur"] for e in ops0 if w0[0] <= e["ts"] < w0[1] and _ALLREDUCE.search(e["name"])]
    a1 = [e["dur"] for e in ops1 if w1[0] <= e["ts"] < w1[1] and _ALLREDUCE.search(e["name"])]
    if len(a0) != len(a1):
        raise RuntimeError(f"{len(a0)} all-reduces on TP0 vs {len(a1)} on TP1 in the paired window")
    mins = np.minimum(np.array(a0), np.array(a1))
    return {
        "calls": len(a0),
        "tp0_ms": sum(a0) / 1e3,
        "tp1_ms": sum(a1) / 1e3,
        "transfer_ms": float(mins.sum()) / 1e3,
        "transfer_median_us": float(np.median(mins)) if len(mins) else 0.0,
    }


_SYNC_CALLS = re.compile(r"Synchronize|cudaMemcpy(?!.*Async)|cudaStreamQuery|cudaEventQuery")


def _cpu_spans(events: list, seqs: int, tokens: int, spec: bool) -> List[dict]:
    """CPU ranges (user_annotation) of each target extend step and its draft_extend, in order."""
    step_name = re.compile(rf"^step\[EXTEND bs={seqs} toks={tokens}[\] ]")
    ua = sorted((e for e in events if e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
    steps = [e for e in ua if step_name.match(e["name"])]
    refresh = [e for e in ua if e["name"] == "draft_extend"]
    out = []
    for s in steps:
        r = next((x for x in refresh if x["ts"] >= s["ts"]), None) if spec else None
        if spec and r is None:
            raise RuntimeError(f"no CPU draft_extend range after the CPU extend step at {s['ts']}")
        out.append({"target": (s["ts"], r["ts"] if r else s["ts"] + s["dur"], s["tid"]), "refresh": (r["ts"], r["ts"] + r["dur"], r["tid"]) if r else None})
    return out


def _host_syncs(events: list, span: Tuple[float, float, int]) -> dict:
    """Blocking CUDA runtime calls on the launching thread inside a CPU range: each one lets the GPU
    run dry before the host launches more (memcpy without Async is synchronous for pageable host
    memory; DtoH .item()/.tolist() reads show up as cudaMemcpyAsync + cudaStreamSynchronize)."""
    lo, hi, tid = span
    calls = collections.Counter()
    us = collections.Counter()
    for e in events:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver") or e.get("tid") != tid:
            continue
        if not (lo <= e["ts"] < hi) or not _SYNC_CALLS.search(e["name"]):
            continue
        key = e["name"]
        calls[key] += 1
        us[key] += e.get("dur", 0.0)
    return {"calls": dict(calls), "ms": {k: v / 1e3 for k, v in us.items()}, "cpu_range_ms": (hi - lo) / 1e3}


def analyze(args) -> dict:
    events = _load(args.trace)
    ops = _gpu_ops(events)
    starts = [e["ts"] for e in ops]
    paths = _launch_stacks(events)
    name_map = None
    if args.map:
        name_map = json.load(open(args.map))["name_map"]
    wins = _windows(events, args.seqs, args.tokens, spec=not args.no_spec)
    peer = None
    if args.peer:
        pev = _load(args.peer)
        pops = _gpu_ops(pev)
        pwins = _windows(pev, args.seqs, args.tokens, spec=not args.no_spec)
        if len(pwins) != len(wins):
            raise RuntimeError(f"{len(wins)} extend steps on TP0, {len(pwins)} on the peer")
        peer = (pops, pwins)
        del pev
    cpu = _cpu_spans(events, args.seqs, args.tokens, spec=not args.no_spec)
    if len(cpu) != len(wins):
        raise RuntimeError(f"{len(cpu)} CPU extend step ranges vs {len(wins)} GPU ones")
    reps = []
    learned: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for i, w in enumerate(wins):
        rep = {"windows": {k: (v[1] - v[0]) / 1e3 for k, v in w.items() if v is not None}}
        rep["host_syncs"] = {part: _host_syncs(events, cpu[i][part]) for part in ("target", "refresh") if cpu[i][part]}
        for part in ("target", "refresh"):
            if w[part] is None:
                continue
            lo, hi = w[part]
            b = _breakdown(_in(ops, starts, lo, hi), paths, name_map, lo, hi, prefix="refresh" if part == "refresh" else None)
            by_name = b.pop("by_name")
            if part == "target":
                for n, mix in by_name.items():
                    for c, us in mix.items():
                        if not c.startswith(("other: ", "unmapped")):
                            learned[n][c] += us
            rep[part] = b
        lo, hi = w["lm"]
        lm = _breakdown(_in(ops, starts, lo, hi), paths, name_map, lo, hi)
        lm.pop("by_name")
        rep["lm"] = {k: lm[k] for k in ("window_ms", "busy_ms", "idle_ms")}
        if w["mm"] is not None:
            lo, hi = w["mm"]
            mm = _breakdown(_in(ops, starts, lo, hi), paths, name_map, lo, hi)
            rep["mm"] = {k: mm[k] for k in ("window_ms", "busy_ms", "idle_ms")}
        if peer is not None:
            pops, pwins = peer
            rep["allreduce_target"] = _allreduce_pairs(ops, pops, w["target"], pwins[i]["target"])
            if w["refresh"] is not None:
                rep["allreduce_refresh"] = _allreduce_pairs(ops, pops, w["refresh"], pwins[i]["refresh"])
        reps.append(rep)
    # name -> category mix (fractions), for analyzing graph replays of the same shape
    nm = {}
    for n, mix in learned.items():
        tot = sum(mix.values())
        if tot > 0:
            nm[n] = {c: v / tot for c, v in mix.items() if v / tot >= 0.01}
            s = sum(nm[n].values())
            nm[n] = {c: v / s for c, v in nm[n].items()}
    result = {"trace": args.trace, "peer": args.peer, "seqs": args.seqs, "tokens": args.tokens, "reps": reps, "name_map": nm}
    _print_analysis(result)
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
    return result


def _print_analysis(result: dict) -> None:
    reps = result["reps"]
    print(f"== {result['trace']}: {len(reps)} extend steps of bs={result['seqs']} toks={result['tokens']}")
    for part in ("target", "refresh"):
        if part not in reps[0]:
            continue
        cats = collections.defaultdict(list)
        for r in reps:
            for c, v in r[part]["categories"].items():
                cats[c].append(v["ms"])
        n = len(reps)
        mean = lambda k: np.mean([r[part][k] for r in reps])  # noqa: E731
        print(
            f"-- {part}: window {mean('window_ms'):.2f} ms, busy {mean('busy_ms'):.2f}, idle {mean('idle_ms'):.2f}, "
            f"kernel sum {mean('kernel_sum_ms'):.2f} (graph kernels {mean('graph_ms'):.2f}, of which split by the eager "
            f"name map {mean('graph_split_ms'):.2f})"
        )
        for c, vals in sorted(cats.items(), key=lambda kv: -sum(kv[1])):
            ms = sum(vals) / n
            calls = np.mean([r[part]["categories"].get(c, {"calls": 0})["calls"] for r in reps])
            tops = reps[-1][part]["top"].get(c, [])
            print(f"   {ms:8.2f} ms  {calls:6.0f} calls  {c:28s} {'; '.join(f'{k[:60]} {t:.2f}' for k, t, _ in tops[:3])}")
        if "allreduce_" + part in reps[0]:
            ar = [r["allreduce_" + part] for r in reps]
            print(
                f"   all-reduce pairs: {np.mean([a['calls'] for a in ar]):.0f} calls, TP0 {np.mean([a['tp0_ms'] for a in ar]):.2f} ms, "
                f"TP1 {np.mean([a['tp1_ms'] for a in ar]):.2f} ms, transfer (per-call min) {np.mean([a['transfer_ms'] for a in ar]):.2f} ms "
                f"(median call {np.mean([a['transfer_median_us'] for a in ar]):.1f} us)"
            )
        um = reps[-1][part]["unmapped_graph_kernels"]
        if um:
            print(f"   graph kernels without an eager mapping: {um}")
    for part in ("target", "refresh"):
        if part in reps[-1]["host_syncs"]:
            hs = reps[-1]["host_syncs"][part]
            print(f"-- {part} CPU range {hs['cpu_range_ms']:.2f} ms (under the profiler), blocking calls: "
                  + (", ".join(f"{k} x{hs['calls'][k]} {hs['ms'][k]:.2f} ms" for k in hs["calls"]) or "none"))
    lm = np.mean([r["lm"]["window_ms"] for r in reps])
    lmb = np.mean([r["lm"]["busy_ms"] for r in reps])
    mm = [r["mm"] for r in reps if "mm" in r]
    print(
        f"-- language-model span {lm:.2f} ms (busy {lmb:.2f})"
        + (f"; media span {np.mean([m['window_ms'] for m in mm]):.2f} ms (busy {np.mean([m['busy_ms'] for m in mm]):.2f})" if mm else "")
    )


# ---------------------------------------------------------------------------------------------------
# roofline: per category, the larger of FLOPs / peak and bytes / bandwidth, per rank and forward,
# TP2 + EP2 (dense layers TP-split, HC / router / indexer / PLE projections replicated, 256 local
# experts per rank). Weights: routed experts block-FP8; every other linear bf16 (the checkpoint's
# modules_to_not_convert). Spec-sheet B200 peaks; "ideal" bytes are each operand once from HBM.
B200 = {"hbm": 8.0e12, "bf16": 2.25e15, "fp8": 4.5e15, "nvlink": 0.9e12}
H, LAYERS, L_GDN, L_QSA = 2560, 48, 36, 12
BF16, FP8 = 2, 1


def roofline_ms(spec: List[Tuple[int, int, int]], tp: int = 2) -> Dict[str, Tuple[float, str]]:
    """{category: (ms, note)} for one target extend + drafter refresh of `spec`."""
    T = sum(n for _, n, _ in spec)
    ctx = [p + n for p, n, _ in spec]
    news = [n for _, n, _ in spec]

    def t(flops, nbytes, peak):
        return max(flops / B200[peak], nbytes / B200["hbm"]) * 1e3

    out = {}
    # routed experts: E_act local experts' FP8 weights + permuted activations; tile padding ignored
    local = 512 // tp
    assign = T * 10 / tp
    e_act = local * (1 - np.exp(-T * 10 / 512))
    w = 3 * 640 * H
    out["MoE routed"] = (
        LAYERS * t(2 * assign * w, e_act * w * FP8 + assign * 12800, "fp8"),
        f"{e_act:.0f} of {local} local experts active, {assign:.0f} assignments/layer: weight-read bound",
    )
    sh = H * 2 * 640 // tp + 640 // tp * H
    out["MoE shared expert"] = (LAYERS * t(2 * T * sh, sh * BF16 + T * (H + 640 // tp * 3 + H) * BF16, "bf16"), "bf16, TP-split")
    out["MoE gate/misc"] = (LAYERS * t(2 * T * H * 512, H * 512 * BF16 + T * (H + 512) * BF16, "bf16"), "router 2560x512 bf16, replicated")
    gdn = H * (10240 + 6144 + 48 + 48) // tp + 6144 // tp * H
    out["GDN proj (dense)"] = (L_GDN * t(2 * T * gdn, gdn * BF16 + T * (H * 2 + 16480 // tp + 6144 // tp) * BF16, "bf16"), "in_proj qkv/z/a/b + out_proj, bf16 TP-split")
    qsa = H * (12288 + 512 + 512) // tp + 6144 // tp * H
    out["QSA proj (dense)"] = (L_QSA * t(2 * T * qsa, qsa * BF16 + T * (H * 2 + 13312 // tp + 6144 // tp) * BF16, "bf16"), "q(+gate)/k/v/o bf16 TP-split")
    kc = [c / 4 for c in ctx]
    score_flops = sum(n * k * 4 * 128 * 2 for n, k in zip(news, kc))
    score_bytes = sum(n * k * 4 * 2 for n, k in zip(news, kc)) + sum(k * 128 * BF16 for k in kc)
    out["QSA indexer"] = (
        L_QSA * (t(2 * T * H * 640, H * 640 * BF16 + T * (H + 640) * BF16, "bf16") + t(score_flops, score_bytes, "bf16")),
        "index_qk_proj + own-row scores (fp32 written+read once); top-k ignored",
    )
    att_flops = sum(n * min(c, 2048) * (24 // tp) * 256 * 4 for n, c in zip(news, ctx))
    att_bytes = sum(c * 512 for c in ctx) + T * (24 // tp) * 256 * BF16 * 2
    nore = sum(n * min(c, 2048) * 512 for n, c in zip(news, ctx))
    out["QSA attention core"] = (
        L_QSA * t(att_flops, att_bytes, "bf16"),
        f"2048-key budget, context K/V read once; with no reuse across queries {L_QSA * nore / B200['hbm'] * 1e3:.1f} ms of gathers",
    )
    gdn_core_bytes = T * (2 * 16 * 128 + 2 * 48 * 128 + 2 * 48) // tp * BF16 + T * 16384 // tp * BF16 * 2
    gdn_core_flops = T * (48 // tp) * 2 * (3 * 128 * 128 + 64 * 256)
    out["GDN core"] = (L_GDN * t(gdn_core_flops, gdn_core_bytes, "bf16"), "chunked delta rule + conv1d, activations once")
    hc_flops = 2 * T * (2 * 10240 * 320) * 2
    hc_bytes = 2 * 3 * T * 10240 * BF16
    out["HC/layer level + norm"] = (LAYERS * t(hc_flops, hc_bytes, "bf16"), "2 HC blocks/layer, replicated: low-rank mix GEMMs; 4-wide stream read twice + written once")
    ar = T * H * BF16
    out["allreduce"] = (
        (2 * LAYERS + 1) * (ar / B200["nvlink"]) * 1e3,
        f"{2 * LAYERS + 1} x {ar / 1e6:.1f} MB two-shot at 900 GB/s per direction",
    )
    out["PLE"] = (t(2 * T * H * 12800, H * 12800 * BF16, "bf16"), "key/value proj replicated; host gather ignored")
    out["LM head/logits"] = (t(2 * len(spec) * H * 248320 // tp, 248320 // tp * H * BF16, "bf16"), "last token per request: vocab shard read once")
    # drafter refresh: 5 dense bf16 layers (24/2/256 attention, FFN 10240), replicated, over the new tokens
    d_layer = H * (12288 + 1024) + 6144 * H + 3 * H * 10240
    out["refresh (drafter)"] = (5 * t(2 * T * d_layer, d_layer * BF16, "bf16"), "5 layers x (attn + FFN 10240) bf16 over the new tokens, replicated")
    return out


def roofline(args) -> None:
    for shape in args.shapes.split(","):
        spec = SHAPES[shape]
        r = roofline_ms(spec)
        total = sum(v[0] for k, v in r.items() if not k.startswith("refresh"))
        print(f"== {shape}: {len(spec)} requests, new {[s[1] for s in spec]}, prefix {[s[0] for s in spec]}")
        for k, (ms, note) in sorted(r.items(), key=lambda kv: -kv[1][0]):
            print(f"   {ms:7.2f} ms  {k:24s} {note}")
        print(f"   {total:7.2f} ms  target total (refresh {r['refresh (drafter)'][0]:.2f} ms)")


_GROUPS = [
    ("MoE routed", lambda c: c.startswith("MoE routed")),
    ("MoE shared expert", lambda c: c == "MoE shared expert"),
    ("MoE gate/misc", lambda c: c in ("MoE gate/misc", "MoE router (topk)")),
    ("GDN proj (dense)", lambda c: c == "GDN proj (dense)"),
    ("GDN core", lambda c: c in ("GDN core", "GDN misc")),
    ("QSA proj (dense)", lambda c: c == "QSA proj (dense)"),
    ("QSA indexer", lambda c: c == "QSA indexer"),
    ("QSA attention core", lambda c: c in ("QSA attention core", "QSA misc")),
    ("HC/layer level + norm", lambda c: c in ("HC/layer level", "norm")),
    ("allreduce", lambda c: c == "allreduce"),
    ("PLE", lambda c: c == "PLE"),
    ("LM head/logits", lambda c: c in ("LM head/logits", "sampling", "embedding")),
    ("ViT/mm embed", lambda c: c == "ViT/mm embed"),
    ("other", lambda c: True),
]


def _grouped(rep_part: dict) -> Dict[str, float]:
    out: Dict[str, float] = collections.defaultdict(float)
    for c, v in rep_part["categories"].items():
        g = next(name for name, f in _GROUPS if f(c))
        out[g] += v["ms"]
    return out


# kernels the coordinator asked about, per shape and arm (target window, mean over the profiled reps)
KEY_KERNELS = [
    ("QSA sparse prefill (prefix)", r"^_sparse_gqa_chunk_prefill"),
    ("QSA sparse prefill (cold)", r"^_sparse_gqa_prefill"),
    # TileLang names its kernels kernel_kernel: the prefill index scores and the -inf mask (mqa.py)
    ("QSA index scores+mask (tilelang)", r"^kernel_kernel$"),
    ("QSA top-k", r"fast_topk|topk_kernel|gatherTopK|radixSort"),
    ("QSA block expand", r"expand_qsa_block"),
    ("MoE FC1", r"^bmm_E4m3"),
    ("MoE activation", r"activationDeepSeek"),
    ("MoE FC2", r"^bmm_Bfloat16"),
    ("MoE finalize", r"finalizeKernel"),
    ("MoE routing", r"routingIndices"),
    ("HC combine", r"hc_combine"),
    ("all-reduce (TP0 incl. wait)", _ALLREDUCE.pattern),
    ("GDN chunk (flashinfer)", r"gated_delta_net|flashinfergdn"),
    ("causal conv1d", r"causal_conv1d"),
]


def _key_kernels(an: dict) -> Dict[str, Tuple[float, float, str]]:
    out = {}
    reps = an["reps"]
    for label, pat in KEY_KERNELS:
        r = re.compile(pat)
        ms, calls, names = [], [], collections.Counter()
        for rep in reps:
            m = c = 0.0
            for k, (kms, kc) in rep["target"]["kernels"].items():
                if r.search(k):
                    m += kms
                    c += kc
                    names[k[:70]] += kms
            ms.append(m)
            calls.append(c)
        out[label] = (float(np.mean(ms)), float(np.mean(calls)), names.most_common(1)[0][0] if names else "")
    return out


def table(args) -> None:
    """Per shape: telemetry (no profiler) and the trace's grouped kernel time per arm, next to the
    roofline. all-reduce: kernel time on TP0 and the paired transfer estimate."""
    arms = [(os.path.basename(d.rstrip("/")), d) for d in args.arm_dirs]
    drives = {a: json.load(open(os.path.join(d, "xprof", "drive.json"))) for a, d in arms}
    shapes = list(drives[arms[0][0]]["shapes"])
    for shape in shapes:
        spec = SHAPES[shape]
        roof = roofline_ms(spec)
        print(f"== {shape}: {len(spec)} x {[s[1] for s in spec]} new, prefixes {[s[0] for s in spec]}, {sum(s[2] for s in spec)} new images")
        cols = {}
        for a, d in arms:
            tel = drives[a]["shapes"][shape]["telemetry"]
            path = os.path.join(d, "xprof", shape, "analysis.json")
            an = json.load(open(path)) if os.path.exists(path) else None
            cols[a] = (tel, an)
            line = f"   {a}: telemetry target {tel['target_gpu']:.1f} ms GPU / {tel['target_cpu']:.1f} CPU, refresh {tel['refresh_gpu']:.1f} GPU / {tel['refresh_cpu']:.1f} CPU"
            if an:
                reps = an["reps"]
                busy = np.mean([r["target"]["busy_ms"] for r in reps])
                ar = [r["allreduce_target"] for r in reps if "allreduce_target" in r]
                wait = np.mean([x["tp0_ms"] - x["transfer_ms"] for x in ar]) if ar else 0.0
                line += (
                    f" | trace: target busy {busy:.1f} ms (all-reduce wait {wait:.1f}), so kernels without the wait "
                    f"{busy - wait:.1f} ms and idle+wait at telemetry {tel['target_gpu'] - (busy - wait):.1f} ms"
                )
                if "refresh" in reps[0]:  # absent only in a SPEC=none analysis (--no-spec)
                    line += f"; refresh busy {np.mean([r['refresh']['busy_ms'] for r in reps]):.1f} ms"
            print(line)
        print(f"   {'group':24s} " + " ".join(f"{a[-12:]:>12s}" for a, _ in arms) + f" {'roofline':>9s}")
        groups = [g for g, _ in _GROUPS]
        for g in groups + ["refresh (drafter)"]:
            vals = []
            for a, _ in arms:
                an = cols[a][1]
                if an is None:
                    vals.append(float("nan"))
                    continue
                if g == "refresh (drafter)":
                    vals.append(np.mean([r["refresh"]["busy_ms"] for r in an["reps"]]) if "refresh" in an["reps"][0] else float("nan"))
                elif g == "allreduce":  # the paired transfer estimate; NaN for an analysis without --peer
                    vals.append(
                        np.mean([r["allreduce_target"]["transfer_ms"] for r in an["reps"]])
                        if "allreduce_target" in an["reps"][0]
                        else float("nan")
                    )
                else:
                    vals.append(np.mean([_grouped(r["target"]).get(g, 0.0) for r in an["reps"]]))
            rf = roof.get(g, (float("nan"), ""))[0]
            if all(v == 0 for v in vals) and np.isnan(rf):
                continue
            print(f"   {g:24s} " + " ".join(f"{v:12.2f}" for v in vals) + f" {rf:9.2f}")
        keyed = {a: _key_kernels(cols[a][1]) for a, _ in arms if cols[a][1] is not None}
        if keyed:
            print("   key kernels (ms / calls per target extend):")
            for label, _ in KEY_KERNELS:
                cells = []
                for a, _ in arms:
                    if a in keyed:
                        ms, calls, _name = keyed[a][label]
                        cells.append(f"{ms:7.2f}/{calls:4.0f}")
                    else:
                        cells.append(f"{'-':>12s}")
                name = next((keyed[a][label][2] for a in keyed if keyed[a][label][2]), "")
                print(f"     {label:30s} " + " ".join(cells) + f"  {name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--url", required=True)
    d.add_argument("--log", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--shapes", default=DEFAULT_SHAPES)
    d.add_argument("--reps", type=int, default=6)
    d.add_argument("--profile-reps", type=int, default=2)
    d.add_argument("--interval", type=float, default=10.0)
    d.add_argument("--expect-graph", choices=("true", "false"))
    a = sub.add_parser("analyze")
    a.add_argument("trace")
    a.add_argument("--peer")
    a.add_argument("--seqs", type=int, default=4)
    a.add_argument("--tokens", type=int, default=2816)
    a.add_argument("--map")
    a.add_argument("--no-spec", action="store_true", help="a SPEC=none trace: the step span alone, no refresh")
    a.add_argument("--out")
    r = sub.add_parser("roofline")
    r.add_argument("--shapes", default=DEFAULT_SHAPES)
    t = sub.add_parser("table")
    t.add_argument("arm_dirs", nargs="+")
    args = p.parse_args()
    if args.cmd == "drive":
        drive(args)
    elif args.cmd == "analyze":
        analyze(args)
    elif args.cmd == "roofline":
        roofline(args)
    else:
        table(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
