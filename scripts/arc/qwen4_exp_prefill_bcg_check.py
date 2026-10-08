#!/usr/bin/env python3
"""ARC_QWEN4_EXP_PREFILL_BCG GPU check: the breakable prefill CUDA graph against the eager extend.

Two servers of the same build and the same flags except the opt-in, run one after the other on the
same GPUs (start, `collect`, stop, start the other, `collect`), then `compare`:

  A (eager):  prefill CUDA graph off (the default for Qwen4-Exp)
  B (graph):  ARC_QWEN4_EXP_PREFILL_BCG=1 [--cuda-graph-bs-prefill ...]

Both need the three determinism switches, or nothing below is reproducible even eager vs eager
(memory qsa-fast-topk-nondeterminism): ARC_QSA_TOPK_EXACT=1 SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0
ARC_HC_MIX_DETERMINISTIC=1. Add --enable-return-routed-experts to both to compare routing. With
DFlash also the same --random-seed on both: its verify draws from the server RNG, and unseeded
servers do not reproduce the decode even eager vs eager.

Control first: start A twice (A1, A2) and `compare a1.json a2.json`. It must pass; if it does not,
the A-vs-B result means nothing (serving is not batch-invariant, so a restart alone can move
logprobs). Then `compare a1.json b.json`.

  collect  --url http://127.0.0.1:PORT --out a.json [--exact-sizes 1024,2048] [--routed-experts]
           Flushes the radix cache, then sends the cases one at a time with greedy sampling:
             exact_<n>   one prompt of n tokens, n a captured bucket (no padding)
             pad_<n>     one prompt the replay pads to the next bucket
             pair_<a>_<b> one /generate with two prompts (one extend batch of two requests)
             iso_<n>_pad / iso_<n>_fill   padding isolation, after a cache flush each: a prompt P
                         of n tokens, alone (the replay pads it to the bucket) or next to a filler
                         request of (bucket - n) tokens (one extend of exactly the bucket); one
                         output token each. Then iso_<n>_<variant>_follow: P + 256 tokens, which
                         extends on P's cached KV and the mamba state the extend tracked at the
                         last multiple of 64 <= n (P's very end for n = 1088, 3328; 704 for 722;
                         none for 17), 16 output tokens alone (bs=1)
             iso_<n>_pad2 / iso_<n>_fill2   the same with an 8-token companion request after P in
                         both variants ([P, C] padded vs [P, C, filler]): every extend holds >= 2
                         sequences. FlashInfer's GDN prefill (use_cp="auto") switches a lone
                         sequence to its context-parallel kernel when num_seqs x v-heads-per-rank
                         x 4 < SMs (TP2 on B200: 24 heads, so 1 sequence -> CP, 2 -> not), which
                         makes pad (1 sequence) vs fill (2) differ for a reason that is not
                         padding; pad2/fill2 never switch
             prefix      a 10240-token prompt, then it + 1024 tokens: the second extend runs on a
                         cached prefix (mamba/PLE track restore, QSA prefix path) at an exact bucket
             pad_prefix_800  the same cached prefix + 800 other tokens (padded to 896)
             mm_*        text prompts with 512x512 noise images (258 tokens each, every count
                         checked against prompt_tokens): an image in a 1024 extend; the same
                         prompt cached under a 512-token text extend (an agent's later turns:
                         the request holds media, the extend none); a new image after the cached
                         one; a batch of a media and a text request (2048); an image cut by the
                         8192 chunk boundary (8704); a padded one (700). isomm_704_pad2/fill2:
                         the iso pair with an image in P.
           and records per request: every prompt token's logprob and top-5 ids, 16 greedy output
           tokens and their logprobs, cached_tokens, spec counters, routed experts.
  compare  a.json b.json   exit 1 unless every case whose extends all run at an exact bucket
           (exact_*, prefix, mm_exact/prefix/chunked_*, iso[mm]_*_fill[2], iso[mm]_<n>_fill[2]_follow
           for 64 | n) is bitwise identical. mm_pair_* is reported only, like pair_*: the media
           request reaches the scheduler later than the text one (its frontend processing), so the
           two may or may not share an extend (isomm_*_fill2 is the gated mixed batch).
           pad_* replays at a larger M than eager runs it, so GEMMs may pick other tiles; pair_*
           depends on both servers batching the two prompts alike; both reported only.
  isolation x.json [--expect-bitwise] [--gate-single]   within one server, iso_<n>_pad2 vs
           iso_<n>_fill2 (and iso_<n>_pad vs iso_<n>_fill): P's extend at the same M, once next to
           padding rows and once next to a real request. On the graph server (--expect-bitwise)
           P's routed experts and the whole follow-up (prompt and output logprobs, ids, routing)
           must match bitwise: a padding row leaking into P's rows, its KV or its GDN state shows
           up there. The single-sequence pair is gated only with --gate-single (where the GDN CP
           switch above never fires, e.g. TP1 on B200). P's own logprobs come from the eager LM head
           at a different M (n vs bucket rows) and are reported only. The first decode step after
           a fill extend would hold the finished filler too (overlap scheduler), so P's decode is
           not compared; the follow-up decodes alone. On the eager server this measures how far
           M alone (n vs bucket) moves the same prompt.
  vitcost  --url --out-dir   GPU time of the media embedding (ViT + merge) and of the language
           model in one 2048-token extend with 0 / 1 / 4 new 512^2 images and 1 new 1024^2 image,
           from a torch profiler trace of TP0.
  log      server.log   prefill graph capture time / memory, KV size, and how many prefill
           batches ran the graph ("cuda graph: True") vs eager.

stdlib + numpy only; does not import sglang.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import struct
import sys
import time
import zlib
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

TOKEN_LO, TOKEN_HI = 1000, 100000  # plain vocab, away from the special tokens at the top
DECODE_TOKENS = 16
TOP_K = 5
PREFIX_TOKENS, PREFIX_EXTRA, PREFIX_EXTRA_PAD = 10240, 1024, 800
PAD_SIZES = (17, 722, 1100)
# (P tokens, the bucket the replay pads P to) under the runbook's --cuda-graph-bs-prefill; collect
# checks this against the graph server's bucket list. An extend tracks the mamba state at the last
# multiple of the checkpoint grid (lcm(mamba chunk 64, page 64) = 64 here; collect checks
# page_size): 1088 and 3328 are multiples, so the follow-up restores the state at P's very end,
# where leaked padding tokens would land; 722 restores the state at 704, 17 none.
ISO_CASES = ((17, 32), (722, 768), (1088, 1280), (3328, 4096))
ISO_FOLLOW = 256  # a captured bucket
ISO_COMPANION = 8  # pad2/fill2: a second sequence in every extend (see the docstring)
MAMBA_TRACK_GRID = 64
# media cases: text prompts with the Qwen3-VL vision placeholder; a 512x512 image expands to
# (512 / (patch 16 x merge 2))^2 = 256 <|image_pad|> tokens between vision_start / vision_end
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
MM_IMAGE_SIZE = 512
MM_IMAGE_TOKENS = (MM_IMAGE_SIZE // 32) ** 2 + 2
MM_ISO = (704, 768)  # (P tokens, bucket): P holds an image; 704 is on the 64-token mamba grid
# candidates for one-token words (" word"); collect keeps the ones the server tokenizes to 1 token
WORDS = (
    "time year people way day man thing woman life child world school state family student group "
    "country problem hand part place case week company system program question work government "
    "number night point home water room mother area money story fact month lot right study book "
    "eye job word business issue side kind head house service friend father power hour game line "
    "end member law car city community name president team minute idea kid body information back "
    "parent face others level office door health person art war history party result change "
    "morning reason research girl guy moment air teacher force education foot boy age policy music "
    "market sense nation plan college interest death experience effect class control care field "
    "development role effort rate heart drug show leader light voice wife police mind price report "
    "decision son view relationship town road arm difference value building action model season "
    "society tax director position player record paper space ground form event official matter "
    "center couple site project activity star table need court oil situation cost industry figure "
    "street image phone data picture practice piece land product doctor wall patient worker news "
    "test movie north love support technology step baby computer type attention film tree source "
    "organization hair window evidence population site truth river"
).split()


def _gated(name: str) -> bool:
    if name.startswith(("exact_", "prefix")):
        return True
    if name.startswith(("mm_exact_", "mm_prefix_", "mm_chunked_")):
        return True
    m = re.fullmatch(r"iso(?:mm)?_(\d+)_fill2?(_follow)?", name)
    return bool(m) and (m.group(2) is None or int(m.group(1)) % MAMBA_TRACK_GRID == 0)


def _post(url: str, payload: Optional[dict], timeout: float = 600.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body.strip().startswith((b"{", b"[")) else body.decode()


FLUSH_WAIT_SECONDS = 180


def _flush(url: str) -> str:
    """POST /flush_cache. sglang answers 400 while any request is queued or running; right after
    readiness that is its own warmup request (the VLM warmup sends an image), so wait it out."""
    deadline = time.monotonic() + FLUSH_WAIT_SECONDS
    while True:
        try:
            return _post(f"{url}/flush_cache", None)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"flush_cache refused for {FLUSH_WAIT_SECONDS} s (pending requests): {e.read()!r}"
                ) from e
            time.sleep(2)


def _get(url: str, timeout: float = 60.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _tokens(seed: int, n: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


def _generate(
    url: str,
    prompts: List[List[int]],
    *,
    logprob_start: int,
    routed: bool,
    max_new_tokens: Optional[List[int]] = None,
):
    if max_new_tokens is None:
        max_new_tokens = [DECODE_TOKENS] * len(prompts)
    assert len(max_new_tokens) == len(prompts), (max_new_tokens, len(prompts))
    sampling = [{"temperature": 0.0, "max_new_tokens": m} for m in max_new_tokens]
    if len(prompts) == 1:
        sampling = sampling[0]
    payload = {
        "input_ids": prompts if len(prompts) > 1 else prompts[0],
        "sampling_params": sampling,
        "return_logprob": True,
        "logprob_start_len": logprob_start,
        "top_logprobs_num": TOP_K,
    }
    if routed:
        payload["return_routed_experts"] = True
    out = _post(f"{url}/generate", payload)
    return out if isinstance(out, list) else [out]


def _png_data_url(seed: int, size: int = MM_IMAGE_SIZE) -> str:
    """A deterministic RGB noise PNG (stdlib only)."""
    rng = random.Random(seed)
    raw = b"".join(b"\x00" + rng.randbytes(size * 3) for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


class _Text:
    """Text of an exact token count: one-token words (" word") the server's tokenizer keeps whole."""

    def __init__(self, url: str):
        self.url = url
        out = _post(f"{url}/tokenize", {"prompt": [" " + w for w in WORDS], "add_special_tokens": False})
        self.words = [w for w, n in zip(WORDS, out["count"], strict=True) if n == 1]
        if len(self.words) < 100:
            raise RuntimeError(f"only {len(self.words)} one-token words")

    def __call__(self, seed: int, n: int) -> str:
        rng = random.Random(seed)
        text = "".join(" " + rng.choice(self.words) for _ in range(n))
        count = _post(f"{self.url}/tokenize", {"prompt": text, "add_special_tokens": False})["count"]
        if count != n:
            raise RuntimeError(f"text of {n} one-token words tokenizes to {count}")
        return text


def _mm_prompt(*parts) -> dict:
    """parts: (text, tokens) or ("image", seed); returns {text, images, tokens}."""
    text, images, tokens = "", [], 0
    for part in parts:
        if part[0] == "image":
            text += VISION_BLOCK
            images.append(_png_data_url(part[1]))
            tokens += MM_IMAGE_TOKENS
        else:
            text += part[0]
            tokens += part[1]
    return {"text": text, "images": images, "tokens": tokens}


def _generate_text(
    url: str,
    prompts: List[dict],
    *,
    logprob_start: int,
    routed: bool,
    max_new_tokens: Optional[List[int]] = None,
):
    """_generate for text prompts with images; checks every prompt's token count."""
    if max_new_tokens is None:
        max_new_tokens = [DECODE_TOKENS] * len(prompts)
    assert len(max_new_tokens) == len(prompts), (max_new_tokens, len(prompts))
    sampling = [{"temperature": 0.0, "max_new_tokens": m} for m in max_new_tokens]
    images = [p["images"] or None for p in prompts]
    payload = {
        "text": [p["text"] for p in prompts] if len(prompts) > 1 else prompts[0]["text"],
        "sampling_params": sampling if len(prompts) > 1 else sampling[0],
        "return_logprob": True,
        "logprob_start_len": logprob_start,
        "top_logprobs_num": TOP_K,
    }
    if any(images):
        payload["image_data"] = images if len(prompts) > 1 else images[0]
    if routed:
        payload["return_routed_experts"] = True
    out = _post(f"{url}/generate", payload)
    out = out if isinstance(out, list) else [out]
    for p, r in zip(prompts, out, strict=True):
        if r["meta_info"]["prompt_tokens"] != p["tokens"]:
            raise RuntimeError(
                f"prompt of {len(p['images'])} images tokenized to {r['meta_info']['prompt_tokens']}, "
                f"expected {p['tokens']}"
            )
    return out


def _record(result: dict) -> dict:
    meta = result["meta_info"]
    rec = {
        "input_logprobs": [x[0] for x in meta["input_token_logprobs"]],
        "input_top_ids": [[t[1] for t in row] if row else None for row in meta["input_top_logprobs"]],
        "output_ids": result["output_ids"],
        "output_logprobs": [x[0] for x in meta["output_token_logprobs"]],
        "cached_tokens": meta.get("cached_tokens"),
        "spec": {k: meta[k] for k in ("spec_verify_ct", "spec_accept_length") if k in meta},
    }
    if "routed_experts" in meta:
        m = meta["routed_experts_meta"]
        flat = np.frombuffer(base64.b64decode(meta["routed_experts"]), dtype=np.dtype(m["dtype"]))
        rec["routed_experts"] = {"start": m["start"], "shape": m["shape"], "data": flat.tolist()}
    return rec


def collect(args) -> None:
    url = args.url.rstrip("/")
    server_info = _get(f"{url}/server_info")
    if server_info["page_size"] != MAMBA_TRACK_GRID:
        raise RuntimeError(
            f"page_size {server_info['page_size']}: the iso_* cases assume a {MAMBA_TRACK_GRID}-token "
            "mamba checkpoint grid"
        )
    buckets = server_info["cuda_graph_bs_prefill"]
    if buckets is not None:
        for n, bucket in ISO_CASES:
            for rows in (n, n + ISO_COMPANION):
                padded_to = min(b for b in buckets if b >= rows)
                if padded_to != bucket:
                    raise RuntimeError(
                        f"iso_{n}: {rows} rows pad to {padded_to} under --cuda-graph-bs-prefill "
                        f"{buckets}, but ISO_CASES sizes its filler for {bucket}"
                    )
        if ISO_FOLLOW not in buckets:
            raise RuntimeError(f"ISO_FOLLOW {ISO_FOLLOW} is not a captured bucket: {buckets}")
        for rows in (MM_ISO[0], MM_ISO[0] + ISO_COMPANION):
            if min(b for b in buckets if b >= rows) != MM_ISO[1]:
                raise RuntimeError(f"isomm: {rows} rows do not pad to {MM_ISO[1]} under {buckets}")
        for rows in (512, 1024, 2048, 8192):
            if rows not in buckets:
                raise RuntimeError(f"the mm cases need a {rows} bucket: {buckets}")
    flushed = _flush(url)
    exact = [int(x) for x in args.exact_sizes.split(",") if x]
    cases: Dict[str, List[dict]] = {}

    def run(name, prompts, logprob_start=0, max_new_tokens=None):
        print(f"  {name}: {[len(p) for p in prompts]} tokens", flush=True)
        cases[name] = [
            _record(r)
            for r in _generate(
                url,
                prompts,
                logprob_start=logprob_start,
                routed=args.routed_experts,
                max_new_tokens=max_new_tokens,
            )
        ]

    def run_text(name, prompts, logprob_start=0, max_new_tokens=None):
        print(f"  {name}: {[p['tokens'] for p in prompts]} tokens, {[len(p['images']) for p in prompts]} images", flush=True)
        cases[name] = [
            _record(r)
            for r in _generate_text(
                url,
                prompts,
                logprob_start=logprob_start,
                routed=args.routed_experts,
                max_new_tokens=max_new_tokens,
            )
        ]

    text = _Text(url)
    V = MM_IMAGE_TOKENS

    def isomm(variant: str) -> None:
        n, bucket = MM_ISO
        prompt = _mm_prompt((text(8101, 100), 100), ("image", 8106), (text(8102, n - 100 - V), n - 100 - V))
        companion = _mm_prompt((text(8103, ISO_COMPANION), ISO_COMPANION))
        fill = _mm_prompt((text(8104, bucket - n - ISO_COMPANION), bucket - n - ISO_COMPANION))
        prompts = {"pad2": [prompt, companion], "fill2": [prompt, companion, fill]}[variant]
        _flush(url)
        run_text(f"isomm_{n}_{variant}", prompts, max_new_tokens=[1] * len(prompts))
        follow = _mm_prompt((prompt["text"], n), (text(8105, ISO_FOLLOW), ISO_FOLLOW))
        follow["images"] = prompt["images"]
        run_text(f"isomm_{n}_{variant}_follow", [follow], logprob_start=n)

    # Phase 1 runs every case whose extends are all exact buckets on the graph server, before any
    # padded one: the graph and the eager server then share the request history up to the last
    # gated case. Spec decoding (DFlash) depends on that history, not only on the request: the
    # eager server alone decodes the same request differently after different earlier requests.
    exact_prompts = {n: _tokens(1000 + 1 + i, n) for i, n in enumerate(exact)}
    pad_seed = 1000 + len(exact)
    pair_seed = pad_seed + len(PAD_SIZES) + 1
    prefix_seed = pair_seed + 100

    def iso(n: int, bucket: int, variant: str) -> None:
        prompt = _tokens(3000 + n, n)
        companion = _tokens(6000 + n, ISO_COMPANION)
        prompts = {
            "pad": [prompt],
            "fill": [prompt, _tokens(5000 + n, bucket - n)],
            "pad2": [prompt, companion],
            "fill2": [prompt, companion, _tokens(7000 + n, bucket - n - ISO_COMPANION)],
        }[variant]
        # each variant extends P from scratch
        _flush(url)
        run(f"iso_{n}_{variant}", prompts, max_new_tokens=[1] * len(prompts))
        run(f"iso_{n}_{variant}_follow", [prompt + _tokens(4000 + n, ISO_FOLLOW)], logprob_start=n)

    for n in exact:
        run(f"exact_{n}", [exact_prompts[n]])
    run("pair_600_424", [_tokens(pair_seed, 600), _tokens(pair_seed + 50, 424)])
    phase1_iso = [(n, b) for n, b in ISO_CASES if n % MAMBA_TRACK_GRID == 0]
    for n, bucket in phase1_iso:
        for variant in ("fill", "fill2"):
            iso(n, bucket, variant)
    prefix = _tokens(prefix_seed, PREFIX_TOKENS)
    # warm the radix cache with the prefix (its chunks also run through the graph or eager)
    _generate(url, [prefix], logprob_start=PREFIX_TOKENS - 1, routed=False)
    run("prefix", [prefix + _tokens(prefix_seed + 1, PREFIX_EXTRA)], logprob_start=PREFIX_TOKENS)
    # media, every extend an exact bucket: an image in the extend; the same image in the cached
    # prefix under a text-only extend (every agent turn after the first); a new image after a
    # cached one; a batch of a media and a text request; an image cut by the chunk boundary
    _flush(url)
    p1 = _mm_prompt((text(8001, 300), 300), ("image", 8011), (text(8002, 1024 - 300 - V), 1024 - 300 - V))
    run_text("mm_exact_1024", [p1])
    run_text("mm_prefix_text_512", [_mm_prompt((p1["text"], 1024), (text(8003, 512), 512)) | {"images": p1["images"]}],
             logprob_start=1024)
    p3 = _mm_prompt((p1["text"], 1024), (text(8004, 200), 200), ("image", 8012), (text(8005, 1024 - 200 - V), 1024 - 200 - V))
    p3["images"] = p1["images"] + p3["images"]
    run_text("mm_prefix_img_1024", [p3], logprob_start=1024)
    run_text(
        "mm_pair_2048",
        [
            _mm_prompt((text(8006, 100), 100), ("image", 8013), (text(8007, 1000 - 100 - V), 1000 - 100 - V)),
            _mm_prompt((text(8008, 1048), 1048)),
        ],
    )
    run_text(
        "mm_chunked_8704",
        [_mm_prompt((text(8009, 8000), 8000), ("image", 8014), (text(8010, 8704 - 8000 - V), 8704 - 8000 - V))],
    )
    isomm("fill2")
    # Phase 2: padded extends
    _flush(url)
    run_text("mm_pad_700", [_mm_prompt((text(8015, 100), 100), ("image", 8016), (text(8017, 700 - 100 - V), 700 - 100 - V))])
    isomm("pad2")
    run(
        f"pad_prefix_{PREFIX_EXTRA_PAD}",
        [prefix + _tokens(prefix_seed + 2, PREFIX_EXTRA_PAD)],
        logprob_start=PREFIX_TOKENS,
    )
    for i, n in enumerate(PAD_SIZES):
        run(f"pad_{n}", [_tokens(pad_seed + 1 + i, n)])
    for n, bucket in ISO_CASES:
        done = ("fill", "fill2") if (n, bucket) in phase1_iso else ()
        for variant in ("pad", "fill", "pad2", "fill2"):
            if variant not in done:
                iso(n, bucket, variant)

    with open(args.out, "w") as f:
        json.dump({"server_info": server_info, "flush": flushed, "cases": cases}, f)
    print(f"wrote {args.out}")


def _diff(a: List[float], b: List[float]) -> tuple:
    x = np.array([np.nan if v is None else v for v in a], dtype=np.float64)
    y = np.array([np.nan if v is None else v for v in b], dtype=np.float64)
    if x.shape != y.shape:
        return (None, f"length {x.shape} vs {y.shape}")
    both = ~(np.isnan(x) & np.isnan(y))
    d = np.abs(x[both] - y[both])
    return (int((d != 0).sum()), float(d.max()) if d.size else 0.0)


def _server_arg_diff(a: dict, b: dict) -> List[str]:
    keys = sorted(set(a) | set(b))
    skip = {"port", "pid", "served_model_name", "version", "last_gen_throughput"}
    return [
        f"{k}: {a.get(k)!r} -> {b.get(k)!r}"
        for k in keys
        if k not in skip and not k.startswith("internal") and a.get(k) != b.get(k)
    ]


def compare(args) -> int:
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    diffs = _server_arg_diff(a["server_info"], b["server_info"])
    print("server_info differences (expect only the prefill graph config):")
    for line in diffs[:60]:
        print(f"  {line[:200]}")
    spec = a["server_info"]["speculative_algorithm"]
    if spec != b["server_info"]["speculative_algorithm"]:
        raise RuntimeError(f"speculative_algorithm {spec} vs {b['server_info']['speculative_algorithm']}")
    if spec is not None:
        # DFlash decode does not reproduce even eager vs eager with the same --random-seed (one
        # follow-up per run differs, 2026-10-08 r6 dA/dA2 and dB/dB2): gate the target side --
        # prompt logprobs, top-5, the prompt rows' routing, cached tokens, the first output token
        # (it comes from the extend) -- and report the decode
        print(f"speculative_algorithm {spec}: gating the target side; the decode is reported")
    failed = []
    for name in a["cases"]:
        gated = _gated(name)
        for i, (ra, rb) in enumerate(zip(a["cases"][name], b["cases"][name], strict=True)):
            n_in, max_in = _diff(ra["input_logprobs"], rb["input_logprobs"])
            top_same = ra["input_top_ids"] == rb["input_top_ids"]
            out_same = ra["output_ids"] == rb["output_ids"]
            n_out, max_out = _diff(ra["output_logprobs"], rb["output_logprobs"])
            routed = "-"
            if "routed_experts" in ra and "routed_experts" in rb:
                ea = np.array(ra["routed_experts"]["data"]).reshape(ra["routed_experts"]["shape"])
                eb = np.array(rb["routed_experts"]["data"]).reshape(rb["routed_experts"]["shape"])
                if ea.shape != eb.shape:
                    routed = f"shape {ea.shape} vs {eb.shape}"
                else:
                    # [tokens, layers, top_k]: the first MoE layer whose routing moved localizes where the
                    # body diverged (at or before that layer's MoE input).
                    bad_layers = np.flatnonzero((ea != eb).any(axis=(0, 2)))
                    routed = f"{int((ea != eb).any(axis=(1, 2)).sum())}/{ea.shape[0]} tokens differ"
                    if bad_layers.size:
                        routed += f" (first layer {int(bad_layers[0])}, {bad_layers.size} layers)"
            same = (
                n_in == 0
                and top_same
                and out_same
                and n_out == 0
                and ra["cached_tokens"] == rb["cached_tokens"]
                and (routed == "-" or routed.startswith("0/"))
            )
            if spec is not None:
                rows_a = ra["routed_experts"]["shape"][0] - len(ra["output_ids"]) + 1
                rows_b = rb["routed_experts"]["shape"][0] - len(rb["output_ids"]) + 1
                prompt_routed_same = rows_a == rows_b and _routed_diff(ra, rb, rows_a)[0]
                n_first, _ = _diff(ra["output_logprobs"][:1], rb["output_logprobs"][:1])
                target_same = (
                    n_in == 0
                    and top_same
                    and ra["output_ids"][:1] == rb["output_ids"][:1]
                    and n_first == 0
                    and ra["cached_tokens"] == rb["cached_tokens"]
                    and prompt_routed_same
                )
                routed += f"; target side {'same' if target_same else 'DIFF'}"
                same = target_same
            print(
                f"{'GATE' if gated else 'info'} {name}[{i}]: input lp {n_in} differ (max {max_in}), "
                f"top{TOP_K} {'same' if top_same else 'DIFF'}, output ids {'same' if out_same else 'DIFF'}, "
                f"output lp {n_out} differ (max {max_out}), cached {ra['cached_tokens']}/{rb['cached_tokens']}, "
                f"routed {routed}, spec {ra['spec']} / {rb['spec']}"
            )
            if gated and not same:
                failed.append(f"{name}[{i}]")
    if failed:
        print(f"FAIL: not bitwise identical: {failed}")
        return 1
    print("OK: every gated case is bitwise identical")
    return 0


def _routed(rec: dict) -> np.ndarray:
    if "routed_experts" not in rec:
        raise RuntimeError("no routed experts in this collect (collect --routed-experts)")
    r = rec["routed_experts"]
    return np.array(r["data"]).reshape(r["shape"])


def _routed_diff(ra: dict, rb: dict, rows: Optional[int] = None) -> tuple:
    ea, eb = _routed(ra), _routed(rb)
    if rows is not None:
        # the prompt rows only (routing of the decoded tokens follows the decode)
        if ra["routed_experts"]["start"] != 0 or rb["routed_experts"]["start"] != 0:
            raise RuntimeError("routed experts do not start at token 0")
        ea, eb = ea[:rows], eb[:rows]
    if ra["routed_experts"]["start"] != rb["routed_experts"]["start"] or ea.shape != eb.shape:
        return False, (
            f"start/shape {ra['routed_experts']['start']}/{ea.shape} vs "
            f"{rb['routed_experts']['start']}/{eb.shape}"
        )
    tok = np.flatnonzero((ea != eb).any(axis=(1, 2)))
    desc = f"{tok.size}/{ea.shape[0]} tokens differ"
    if tok.size:
        layers = np.flatnonzero((ea != eb).any(axis=(0, 2)))
        desc += f" (first token {int(tok[0])}, first layer {int(layers[0])})"
    return tok.size == 0, desc


def _iso_pair(
    cases: dict, n: int, bucket: int, pad: str, fill: str, gate: bool, decode_gated: bool,
    family: str = "iso",
) -> bool:
    """P's extend next to padding rows (iso_<n>_<pad>) vs next to a real request (iso_<n>_<fill>),
    and the follow-up after each; prints both, returns whether they match bitwise. Without
    decode_gated (spec decoding) the follow-up's decode is reported only."""
    tag = "GATE" if gate else "info"
    pa, *pad_rest = cases[f"{family}_{n}_{pad}"]
    pf, *fill_rest = cases[f"{family}_{n}_{fill}"]
    (fa,) = cases[f"{family}_{n}_{pad}_follow"]
    (ff,) = cases[f"{family}_{n}_{fill}_follow"]
    if len(fill_rest) != len(pad_rest) + 1:
        raise RuntimeError(f"{family}_{n}_{fill} holds {1 + len(fill_rest)} requests, {pad} {1 + len(pad_rest)}")
    for rec in [pa, pf, *pad_rest, *fill_rest]:
        if rec["cached_tokens"] != 0:
            raise RuntimeError(f"{family}_{n}_{pad}/{fill}: a request cached {rec['cached_tokens']} tokens after a flush")
        if len(rec["output_ids"]) != 1:
            raise RuntimeError(f"{family}_{n}_{pad}/{fill}: a request decoded {len(rec['output_ids'])} tokens, not 1")
    # P's extend: the body rows (routing) must match; its logprobs come from the LM head at a
    # different row count
    p_routed_same, p_routed = _routed_diff(pa, pf)
    n_in, max_in = _diff(pa["input_logprobs"], pf["input_logprobs"])
    print(
        f"{tag} {family}_{n} {pad}/{fill} (bucket {bucket}) P extend: routed {p_routed} | reported only: "
        f"prompt lp {n_in} differ (max {max_in}), top{TOP_K} "
        f"{'same' if pa['input_top_ids'] == pf['input_top_ids'] else 'DIFF'}, first output "
        f"{'same' if pa['output_ids'] == pf['output_ids'] else 'DIFF'}"
    )
    # the follow-up is the same request on P's cached KV / mamba state: all of it must match
    expect_cached = n // MAMBA_TRACK_GRID * MAMBA_TRACK_GRID
    f_in, f_max_in = _diff(fa["input_logprobs"], ff["input_logprobs"])
    f_out, f_max_out = _diff(fa["output_logprobs"], ff["output_logprobs"])
    f_routed_same, f_routed = _routed_diff(fa, ff, None if decode_gated else n + ISO_FOLLOW)
    f_top = fa["input_top_ids"] == ff["input_top_ids"]
    f_ids = fa["output_ids"] == ff["output_ids"]
    f_cached = fa["cached_tokens"] == ff["cached_tokens"]
    follow_same = f_in == 0 and f_top and f_routed_same and f_cached
    if decode_gated:
        follow_same = follow_same and f_out == 0 and f_ids
    decode = (
        f"output ids {'same' if f_ids else 'DIFF'}, output lp {f_out} differ (max {f_max_out})"
    )
    print(
        f"{tag} {family}_{n} {pad}/{fill} follow-up: cached {fa['cached_tokens']}/{ff['cached_tokens']} "
        f"(the 64-grid predicts {expect_cached}), prompt lp {f_in} differ (max {f_max_in}), "
        f"top{TOP_K} {'same' if f_top else 'DIFF'}, "
        + (f"{decode}, routed {f_routed}" if decode_gated else
           f"prompt rows routed {f_routed} | spec decode, reported only: {decode}")
    )
    return p_routed_same and follow_same


def isolation(args) -> int:
    collected = json.load(open(args.x))
    cases = collected["cases"]
    # spec decoding: the eager server alone decodes the same follow-up differently after different
    # earlier requests (DFlash, 2026-10-08), so only the follow-up's prompt side is gated
    decode_gated = collected["server_info"]["speculative_algorithm"] is None
    if f"iso_{ISO_CASES[0][0]}_pad2" not in cases:
        raise RuntimeError(f"{args.x} predates the iso_*_pad2/fill2 cases; collect it again")
    failed = []
    for n, bucket in ISO_CASES:
        if (
            not _iso_pair(cases, n, bucket, "pad2", "fill2", args.expect_bitwise, decode_gated)
            and args.expect_bitwise
        ):
            failed.append(f"iso_{n} pad2/fill2")
        gate_single = args.expect_bitwise and args.gate_single
        if not _iso_pair(cases, n, bucket, "pad", "fill", gate_single, decode_gated) and gate_single:
            failed.append(f"iso_{n} pad/fill")
    n, bucket = MM_ISO
    if (
        not _iso_pair(cases, n, bucket, "pad2", "fill2", args.expect_bitwise, decode_gated, "isomm")
        and args.expect_bitwise
    ):
        failed.append(f"isomm_{n} pad2/fill2")
    if failed:
        print(f"FAIL: padding rows are not isolated bitwise: {failed}")
        return 1
    print("OK: padding rows are isolated bitwise" if args.expect_bitwise else "(reported only)")
    return 0


VIT_RUNS = ((0, 512), (1, 512), (4, 512), (1, 1024))  # (new images, image side) per request
VIT_TOKENS = 2048  # every request: one extend of this many tokens
VIT_REPEATS = 3


def vitcost(args) -> None:
    """GPU time of the media embedding (ViT + merge) vs the language model in one extend, from a
    torch profiler trace (gpu_user_annotation spans of sglang.vlm.mm_embedding /
    sglang.vlm.language_model_prefill on TP0). Every request: VIT_TOKENS tokens, k new distinct
    images, one output token, after a cache flush."""
    import glob
    import gzip
    import os

    url = args.url.rstrip("/")
    text = _Text(url)
    os.makedirs(args.out_dir, exist_ok=True)
    plan = []
    for k, side in VIT_RUNS:
        image_tokens = (side // 32) ** 2 + 2
        for rep in range(VIT_REPEATS):
            parts = []
            for i in range(k):
                parts.append(("image", 9000 + 100 * len(plan) + i))
            n_text = VIT_TOKENS - k * image_tokens
            prompt = _mm_prompt((text(9500 + len(plan), n_text), n_text), *parts)
            if side != MM_IMAGE_SIZE:
                prompt["images"] = [_png_data_url(9000 + 100 * len(plan) + i, side) for i in range(k)]
                prompt["tokens"] = VIT_TOKENS
            plan.append((k, side, prompt))
    _flush(url)
    _post(f"{url}/start_profile", {"output_dir": args.out_dir, "activities": ["CPU", "GPU"], "profile_prefix": "vit"})
    for k, side, prompt in plan:
        _flush(url)
        _generate_text(url, [prompt], logprob_start=VIT_TOKENS - 1, routed=False, max_new_tokens=[1])
    _post(f"{url}/stop_profile", None)
    deadline = time.monotonic() + 300
    while True:
        traces = sorted(glob.glob(os.path.join(args.out_dir, "vit-*TP-0*.trace.json.gz")))
        if traces:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"no TP0 trace in {args.out_dir}")
        time.sleep(2)
    time.sleep(5)  # the export finishes writing
    events = json.load(gzip.open(traces[-1]))["traceEvents"]
    spans = {"sglang.vlm.mm_embedding": [], "sglang.vlm.language_model_prefill": []}
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and e.get("name") in spans:
            spans[e["name"]].append((e["ts"], e["dur"] / 1e3))
    lm = [d for _, d in sorted(spans["sglang.vlm.language_model_prefill"])]
    mm = [d for _, d in sorted(spans["sglang.vlm.mm_embedding"])]
    # the routine also wraps every decode forward: with one output token the overlap scheduler
    # still runs one decode step per request, so the spans come as (extend, decode) pairs
    if len(lm) != 2 * len(plan):
        raise RuntimeError(f"{len(lm)} language-model spans for {len(plan)} requests (expected 2 each)")
    extend, decode = lm[0::2], lm[1::2]
    if not all(d < 0.1 * e for e, d in zip(extend, decode)):
        raise RuntimeError(f"language-model spans are not (extend, decode) pairs: {lm}")
    lm = extend
    with_media = [i for i, (k, _, _) in enumerate(plan) if k > 0]
    if len(mm) != len(with_media):
        raise RuntimeError(f"{len(mm)} media spans for {len(with_media)} requests with media")
    mm_of = dict(zip(with_media, mm))
    rows = {}
    for i, (k, side, _) in enumerate(plan):
        rows.setdefault((k, side), []).append((mm_of.get(i, 0.0), lm[i]))
    for (k, side), vals in rows.items():
        m = np.mean([v[0] for v in vals])
        l = np.mean([v[1] for v in vals])
        print(
            f"{k} new images of {side}x{side} in a {VIT_TOKENS}-token extend: media embedding "
            f"{m:.2f} ms GPU, language model {l:.2f} ms GPU (n={len(vals)}, trace {traces[-1]})"
        )


_TELEMETRY_LINE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0[^\]]*\] ARC extend telemetry \[")
_TELEMETRY_BIN = re.compile(r"\| k1/<4096 n=(\d+) gpu=([0-9.]+)/([0-9.]+) cpu=([0-9.]+)/([0-9.]+)")
VITTEL_REQUESTS = 8
VITTEL_TRIGGER_TOKENS = 17


def vittel(args) -> None:
    """Media cost without a profiler, from the extend telemetry (spec serving with
    ARC_EXTEND_TELEMETRY=1 and a short ARC_EXTEND_TELEMETRY_INTERVAL_S). The telemetry prints a
    window when an extend arrives after the interval, so per VIT_RUNS config: VITTEL_REQUESTS
    sequential 2048-token extends (bin k1/<4096; cache flushed before each, one output token),
    an idle interval, then a 17-token trigger extend (bin k1/<512) that prints the window holding
    exactly that config's extends."""
    import datetime

    url = args.url.rstrip("/")
    text = _Text(url)
    triggers = []
    for c, (k, side) in enumerate(VIT_RUNS):
        image_tokens = (side // 32) ** 2 + 2
        for r in range(VITTEL_REQUESTS):
            seed = 20000 + 1000 * c + 10 * r
            n_text = VIT_TOKENS - k * image_tokens
            prompt = {
                "text": text(seed, n_text) + VISION_BLOCK * k,
                "images": [_png_data_url(seed + i, side) for i in range(k)],
                "tokens": VIT_TOKENS,
            }
            _flush(url)
            _generate_text(url, [prompt], logprob_start=VIT_TOKENS - 1, routed=False, max_new_tokens=[1])
        time.sleep(args.interval + 2)
        t = time.time()
        trigger = {"text": text(29000 + c, VITTEL_TRIGGER_TOKENS), "images": [], "tokens": VITTEL_TRIGGER_TOKENS}
        _generate_text(url, [trigger], logprob_start=VITTEL_TRIGGER_TOKENS - 1, routed=False, max_new_tokens=[1])
        triggers.append((k, side, t))
    time.sleep(3)
    lines = []
    for line in open(args.log, errors="replace"):
        m = _TELEMETRY_LINE.search(line)
        if m:
            ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
            lines.append((ts.timestamp(), line))
    for k, side, t in triggers:
        after = [line for ts, line in lines if ts >= t - 1]
        if not after:
            raise RuntimeError(f"no telemetry line after the ({k}, {side}) trigger")
        m = _TELEMETRY_BIN.search(after[0])
        if m is None or int(m.group(1)) != VITTEL_REQUESTS:
            raise RuntimeError(f"the ({k}, {side}) window does not hold its {VITTEL_REQUESTS} extends: {after[0][:400]}")
        print(
            f"{k} new images of {side}x{side} in a {VIT_TOKENS}-token extend: target {float(m.group(2)):.1f} ms GPU, "
            f"{float(m.group(4)):.1f} ms CPU; drafter refresh {float(m.group(3)):.1f} ms GPU "
            f"(mean of {VITTEL_REQUESTS} extends)"
        )


_CAPTURE = re.compile(r"Capture (\S+) prefill CUDA graph (begin|end)\. (.*)")
_PREFILL = re.compile(
    r"Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?cuda graph: (True|False)"
)
_LOG_BATCHES = 40


def log(args) -> None:
    graph = {"True": [], "False": []}
    batches = []
    for line in open(args.log, errors="replace"):
        if m := _CAPTURE.search(line):
            print(f"{m.group(1)} prefill capture {m.group(2)}: {m.group(3)}")
        elif "KV size" in line or "Load weight end" in line or "max_total_num_tokens" in line:
            print(line.rstrip()[-220:])
        elif m := _PREFILL.search(line):
            seqs, new, cached, verdict = m.groups()
            graph[verdict].append(int(new))
            batches.append(f"{seqs}/{new}/{cached}/{verdict[0]}")
    print(
        f"first {_LOG_BATCHES} prefill batches (seqs/new/cached/graph): "
        + " ".join(batches[:_LOG_BATCHES])
    )
    for verdict, tokens in graph.items():
        if tokens:
            print(
                f"prefill batches with cuda graph {verdict}: {len(tokens)}, new tokens "
                f"min/median/max {min(tokens)}/{int(np.median(tokens))}/{max(tokens)}"
            )
        else:
            print(f"prefill batches with cuda graph {verdict}: 0")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--exact-sizes", default="1024,2048")
    c.add_argument("--routed-experts", action="store_true")
    k = sub.add_parser("compare")
    k.add_argument("a")
    k.add_argument("b")
    v = sub.add_parser("vitcost")
    v.add_argument("--url", required=True)
    v.add_argument("--out-dir", required=True)
    t = sub.add_parser("vittel")
    t.add_argument("--url", required=True)
    t.add_argument("--log", required=True)
    t.add_argument("--interval", type=float, required=True)
    i = sub.add_parser("isolation")
    i.add_argument("x")
    i.add_argument("--expect-bitwise", action="store_true")
    i.add_argument("--gate-single", action="store_true")
    g = sub.add_parser("log")
    g.add_argument("log")
    args = p.parse_args()
    if args.cmd == "collect":
        collect(args)
        return 0
    if args.cmd == "compare":
        return compare(args)
    if args.cmd == "isolation":
        return isolation(args)
    if args.cmd == "vitcost":
        vitcost(args)
        return 0
    if args.cmd == "vittel":
        vittel(args)
        return 0
    log(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
