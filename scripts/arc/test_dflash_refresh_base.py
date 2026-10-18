#!/usr/bin/env python3
"""GPU check: without ARC_DFLASH_EXACT_PREFILL the drafter refreshes, compiles and decodes as the base tree does.

  python scripts/arc/test_dflash_refresh_base.py --drafter <deploy-bf16 dir> [--modes 0,det] [--rev ed837597f]
      # prints DFLASH_REFRESH_BASE_OK

MTP3.refresh_entry (the refresh entry, layer view, REFRESH_MIN padding and their start-up compile; on under
EXACT_PREFILL) must leave a server without the flag on the base path. Per ARC_DFLASH_COMPILE mode, one process on
this tree (refresh_entry off, as the worker builds it without the flag) and one on the base tree (`git archive` of
--rev, imported through PYTHONPATH) run the same seeded work on the deployed drafter (random BF16 target embedding /
lm_head, as test_dflash_exact_park.py) and its RingStatePool, the call patterns of the worker's legacy prefill:
fresh windows of 2 .. 4,100 positions chunked by the prefill chunk (the first chunk without a prefix, the next ones on
the previous chunk's cache), continuing chunks of 1 .. 2,048 tokens on a pool slot's window (refresh_committed), and
decode steps of 1, 3 and 8 requests; fast_drafter.warm first, as at server start. Every refresh output (hidden, K/V,
boundaries, positions), every DraftRequest written, every step's proposals and the pool after the steps (its ring
columns: the garbage column takes several rows per slot and step, whichever lands last) are hashed;
the two processes' hashes must all agree, and so must the number of graphs dynamo compiled and the variants per code
object. Probes on this tree, off: MTP3.forward never called with refresh=True, model.layer_view never called (no
padding: the padded branch goes through it), no layer holds a compiled refresh_forward and dynamo never saw that code
object, fast_drafter.warm never called warm_refresh. A third process on this tree with refresh_entry on shows the
probes see the other path. Modes: 0 (eager) and det (inductor, DETERMINISTIC_OPTIONS: the same kernels in every
process); mode 1 autotunes per process, so two processes of the same tree need not agree bitwise there.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

TREE = Path(__file__).resolve().parents[2]
K = 4  # DFLASH_K: verify width, 3 proposals per step
CHUNK = 2048  # ARC_DFLASH_PREFILL_CHUNK default (= the prefill tail)


def child(a):
    os.environ.setdefault("ARC_DFLASH_SPARSE_Q", "1")
    import torch
    from torch._dynamo import convert_frame
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    from torch._dynamo.utils import counters

    import sglang

    want = Path(a.pythonpath).resolve()
    got = Path(sglang.__file__).resolve()
    if want not in got.parents:
        raise RuntimeError(f"sglang imported from {got}, not from {want}")
    sys.path.insert(0, str(TREE / "scripts" / "arc"))
    from dflash_propose_bench import VOCAB, _StubTarget

    from sglang.srt.speculative.arc_dflash import fast_drafter, heads, serving
    from sglang.srt.speculative.arc_dflash import model as M
    from sglang.srt.speculative.arc_dflash import worker as Wk
    from sglang.srt.speculative.arc_dflash.state_pool import RingStatePool

    probes = collections.Counter()
    if a.tree != "base":
        forward, view, warm_refresh = M.MTP3.forward, M.layer_view, fast_drafter.warm_refresh

        def counted_forward(*args, refresh=False, **kw):
            probes["MTP3.forward(refresh=True)"] += int(refresh)
            return forward(*args, refresh=refresh, **kw)

        def counted_view(cache):
            probes["layer_view"] += 1
            return view(cache)

        def counted_warm(*args, **kw):
            probes["warm_refresh"] += 1
            return warm_refresh(*args, **kw)

        M.MTP3.forward, M.layer_view, fast_drafter.warm_refresh = counted_forward, counted_view, counted_warm

    torch.manual_seed(0)
    device = torch.device("cuda")
    one = SimpleNamespace(rank_in_group=0, world_size=1, ranks=[0])
    target = _StubTarget()
    drafter = Path(a.drafter)
    if a.tree == "base":
        model = Wk.build_dense_drafter(drafter, device, target, drafter, one)
    else:
        model = Wk.build_dense_drafter(drafter, device, target, drafter, one, refresh_entry=a.tree == "on")
    model.lm_head = heads.PairHead([(target.lm_head.weight, 0)], VOCAB, one)
    pool = RingStatePool(model, slots=24, max_batch=8, device=device, steps=K - 1, sparse_q=True)
    fast_drafter.warm(SimpleNamespace(model=model, device=device, prefill_chunk=CHUNK))

    out = {}

    def record(name, *tensors):
        for i, t in enumerate(tensors):
            # fresh strides: a size-1 dim keeps any stride through reshape / clone, and view(uint8) needs 1
            flat = torch.empty(t.numel(), dtype=t.dtype, device=t.device).copy_(t.detach().reshape(-1))
            raw = flat.view(torch.uint8) if flat.dtype != torch.bool else flat.to(torch.uint8)
            out[f"{name}/{i}"] = {
                "sha": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest(), "shape": list(t.shape),
                "dtype": str(t.dtype), "head": flat[:4096].float().cpu().tolist(),
                "tail": flat[-4096:].float().cpu().tolist(),
            }

    def cache_tensors(cache):
        return [cache.positions] + [x for layer in cache.layers for x in (layer.key, layer.value, *layer.boundaries)]

    def request_tensors(state):
        return cache_tensors(state.cache) + [state.last_gr, state.last_position]

    g = torch.Generator().manual_seed(1)
    c = model.config
    total = 9000
    ids = torch.randint(4, VOCAB, (total,), generator=g)
    gr = (0.5 * torch.randn(total, c.hc_count, c.hidden_size, generator=g)).to(torch.bfloat16).to(device)
    with torch.no_grad():
        # fresh windows (the legacy prefill's fresh window from a chunk's tail: positions 1 .. n)
        ends = {}
        for j, n in enumerate((2, 5, 17, 1024, 2047, 2048, 2049, 4100)):
            rid = f"f{j}"
            kept = ids[1 : 1 + n].to(device)[None]
            pred = gr[0:n][None]
            positions = torch.arange(1, 1 + n, device=device)[None]
            cache = None
            for k, b0 in enumerate(range(0, n, CHUNK)):
                b1 = min(b0 + CHUNK, n)
                hidden, cache = model.refresh(kept[:, b0:b1], pred[:, b0:b1], positions[:, b0:b1], cache)
                record(f"fresh{n}/chunk{k}/hidden", hidden)
                record(f"fresh{n}/chunk{k}/cache", *cache_tensors(cache))
            state = serving.DraftRequest(serving.decode_cache(cache, c.window), gr[None, n].clone(),
                                         positions[:, -1].clone())
            record(f"fresh{n}/request", *request_tensors(state))
            pool.write_request(rid, state)
            ends[rid] = n
        # continuing chunks on a pool slot's window (the legacy prefill's refresh_committed)
        for j, n in zip(range(8), (1, 2, 3, 7, 50, 2048, 1, 2)):
            rid = f"f{j}"
            start = ends[rid] + 1
            state = pool.read_request(rid)
            positions = torch.arange(start, start + n, device=device)[None]
            span = torch.cat((state.last_gr[:, None], gr[None, start : start + n]), 1)
            state = state.refresh_committed(model, ids[start : start + n].to(device)[None], span, positions)
            record(f"continue{rid}+{n}/request", *request_tensors(state))
            pool.write_request(rid, state)
            ends[rid] = start + n - 1
        # decode steps
        for b in (1, 3, 8):
            slots = torch.tensor([pool.slot_of[f"f{j}"] for j in range(b)], device=device)
            anchors = ids[[ends[f"f{j}"] + 1 for j in range(b)]].to(device)
            _, (q_ids, q_probs) = pool.step(slots, anchors, torch.ones(b, device=device))
            record(f"step{b}", q_ids, q_probs)
        w = c.window
        record("pool", pool.positions, pool.key[:, :, :, :w], pool.value[:, :, :, :w], pool.boundary, pool.last_gr,
               pool.last_position)
        # the garbage column (a step's uncommitted rows, several per slot: whichever write lands last) and the spare
        # columns are never read: reported, not compared
        record("unread/pool", pool.key[:, :, :, w:], pool.value[:, :, :, w:])
    torch.cuda.synchronize()

    entries = collections.Counter()
    for ref in convert_frame.input_codes.seen:
        code = ref()
        if code is not None and "arc_dflash" in code.co_filename:
            # resume functions are named after their line (`..._at_<line>`), which the trees' edits moved
            name = re.sub(r"_at_\d+$", "", code.co_name)
            entries[f"{Path(code.co_filename).name} {name}"] += len(_debug_get_cache_entry_list(code))
    probes["compiled refresh_forward on a layer"] = sum("refresh_forward" in vars(layer) for layer in model.layers)
    probes["dynamo variants of refresh_forward"] = sum(v for k, v in entries.items() if "refresh_forward" in k)
    result = {"tree": a.tree, "sglang": str(got), "compile_mode": model.compile_mode,
              "graphs": counters["stats"]["unique_graphs"], "entries": dict(entries), "probes": dict(probes),
              "tensors": out}
    Path(a.out).write_text(json.dumps(result))
    print(f"child {a.tree} mode {model.compile_mode}: {len(out)} tensors, {result['graphs']} graphs, "
          f"probes {dict(probes)}", flush=True)


def export_base(rev, base):
    if (base / "python" / "sglang" / "__init__.py").exists():
        return
    base.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", str(TREE), "archive", rev, "python/sglang"], check=True,
                             capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive, check=True)


def run_child(a, tree, mode, out):
    pythonpath = (a.base if tree == "base" else TREE) / "python"
    env = dict(os.environ, ARC_DFLASH_COMPILE=mode, PYTHONPATH=str(pythonpath))
    for name in ("ARC_DFLASH_EXACT_PREFILL", "ARC_DFLASH_SESSION_PARK", "ARC_DFLASH_REFRESH_ASYNC"):
        env.pop(name, None)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--tree", tree, "--drafter", a.drafter,
           "--out", str(out), "--pythonpath", str(pythonpath)]
    # the child imports from its PYTHONPATH, not from this script's directory's parents
    subprocess.run(cmd, env=env, check=True, cwd=str(a.work))
    return json.loads(out.read_text())


def compare(x, y):
    same, diff = 0, []
    if x["tensors"].keys() != y["tensors"].keys():
        raise AssertionError(f"different outputs: {sorted(x['tensors'].keys() ^ y['tensors'].keys())[:8]}")
    for name, tx in x["tensors"].items():
        if name.startswith("unread/"):
            continue
        ty = y["tensors"][name]
        if (tx["shape"], tx["dtype"]) != (ty["shape"], ty["dtype"]):
            raise AssertionError(f"{name}: {tx['shape']} {tx['dtype']} vs {ty['shape']} {ty['dtype']}")
        if tx["sha"] == ty["sha"]:
            same += 1
        else:
            sample = [abs(p - q) for p, q in zip(tx["head"] + tx["tail"], ty["head"] + ty["tail"], strict=True)]
            diff.append((name, max(sample)))
    return same, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--modes", default="0,det")
    ap.add_argument("--rev", default="ed837597f", help="the base tree")
    ap.add_argument("--work", type=Path, default=Path(os.environ["HOME"]) / "tmp" / "dflash_refresh_base")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tree", choices=("base", "off", "on"))
    ap.add_argument("--out")
    ap.add_argument("--pythonpath")
    a = ap.parse_args()
    if a.child:
        child(a)
        return 0
    a.work.mkdir(parents=True, exist_ok=True)
    a.base = a.work / f"base-{a.rev}"
    export_base(a.rev, a.base)
    for mode in a.modes.split(","):
        runs = {tree: run_child(a, tree, mode, a.work / f"{tree}-{mode}.json") for tree in ("base", "off", "on")}
        base, off, on = runs["base"], runs["off"], runs["on"]
        same, diff = compare(base, off)
        on_same, on_diff = compare(base, on)
        print(f"mode {mode}: off vs base {same}/{same + len(diff)} outputs bitwise; graphs {off['graphs']} vs "
              f"{base['graphs']}; variants per code object off {off['entries']} / base {base['entries']}")
        print(f"  probes off {off['probes']}; on {on['probes']}; on vs base {on_same}/{on_same + len(on_diff)} "
              f"bitwise, sampled max |diff| {max((d for _, d in on_diff), default=0.0):.3g}")
        unread = [n for n in base["tensors"] if n.startswith("unread/")]
        agree = sum(base["tensors"][n]["sha"] == off["tensors"][n]["sha"] for n in unread)
        print(f"  never-read pool columns (not compared) bitwise: {agree}/{len(unread)}")
        for name, d in diff[:8]:
            print(f"  DIFF {name}: sampled max |diff| {d:.3g}")
        assert not diff, f"mode {mode}: {len(diff)} outputs differ from the base tree"
        assert off["graphs"] == base["graphs"], (off["graphs"], base["graphs"])
        assert off["entries"] == base["entries"], (off["entries"], base["entries"])
        assert off["probes"].get("MTP3.forward(refresh=True)", 0) == 0, off["probes"]
        assert off["probes"].get("layer_view", 0) == 0, off["probes"]
        assert off["probes"].get("warm_refresh", 0) == 0, off["probes"]
        assert off["probes"]["compiled refresh_forward on a layer"] == 0, off["probes"]
        assert off["probes"]["dynamo variants of refresh_forward"] == 0, off["probes"]
        # the probes see the on path
        assert on["probes"]["MTP3.forward(refresh=True)"] > 0 and on["probes"]["layer_view"] > 0, on["probes"]
        if mode != "0":
            assert on["probes"]["warm_refresh"] == 1, on["probes"]
            assert on["probes"]["compiled refresh_forward on a layer"] > 0, on["probes"]
            assert on["probes"]["dynamo variants of refresh_forward"] > 0, on["probes"]
    print("DFLASH_REFRESH_BASE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
