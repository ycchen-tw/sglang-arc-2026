"""MTP5-DFlash drafter inside SGLang's EAGLE v2 worker: the target, verify and HiSparse paths stay as deployed.

The server runs `--speculative-algorithm NEXTN --speculative-num-steps K-1 --speculative-eagle-topk 1
--speculative-num-draft-tokens K` (2 <= K <= 8) with rejection sampling, so the scheduler, the target verify graph,
ReplaySSM, rejection sampling and every QSA HiSparse hook run exactly as for the native MTP. With ARC_DFLASH_DRAFTER
set, EAGLEWorkerV2 builds `DFlashDraftWorker` instead of EagleDraftWorker: no MTP model, no draft KV pool, no draft
graphs. Per cycle:

  draft()                     one drafter forward folds the previous verify's accepted block into each request's
                              window cache and proposes K-1 tokens (selector chain, sampled at the request's
                              temperature); q is scattered into the dense draft_probs rejection sampling reads
  verify                      SGLang's run_eagle_verify, unchanged (chain RS kernel: draft_probs row j <-> slot j+1)
  _draft_extend_for_decode    records [bonus, p1..p_{K-1}] with the verify GR and the accept length as a pending
                              commit; nothing runs (the next draft() folds it in)
  _draft_extend_for_prefill   builds or extends the request's window cache from the prefill GR

Where it runs: on the attention-TP group of this rank. One GPU (G4: TP1) is a group of one. With attention DP
(L4: TP4 target, attention DP2) each DP group's attention-TP ranks run one drafter together, with collectives only on
that group: the DP groups do not run the drafter in lockstep -- one may prefill (a data-dependent number of drafter
forwards) while the other decodes -- so a collective across DP groups would deadlock. On a group of N > 1 ranks the
routed experts are expert-parallel (marlin.MarlinEPExperts), attention heads are split (ARC_DFLASH_ATTN_TP, default 1),
the lm_head is the group's sharded one (heads.PairHead), and dense weights, the selector and the window state are
replicated: identical inputs through identical kernels, so every rank proposes the same tokens
(ARC_DFLASH_CHECK_AGREE=1 compares them every cycle and raises on a mismatch).

Routed experts: ARC_DFLASH_MOE=cutlass (FlashInfer CUTLASS W4A4, the target's kernel on sm_100/sm_120; needs
ARC_DFLASH_CALIB for GEMM2's static input scale; one rank only) or marlin (SGLang's NVFP4 Marlin W4A16, any group
size). Default: cutlass on compute capability >= 10, else marlin.
A dense drafter export (config ffn_type dense_shared, from arc-forge convert_dense.py) has no routed experts or router:
build_dense_drafter loads it as is, with the linears stored in block FP8 on fp8.Fp8BlockLinear (CUTLASS on sm_120,
Triton on sm_89). On a group of N > 1 the dense FFN is replicated (no collective) and, with ARC_DFLASH_ATTN_TP=1, the
attention heads are split as for the MoE drafter (q/k/v rows and o_proj columns sliced at load, 128-aligned so FP8
blocks stay whole; o_proj all-reduced). ARC_DFLASH_KEEP / CALIB / MOE / ROUTER_STATS must be unset.

Draft state per request is a fixed SWA window (`window` = 1024 tokens x layers, plus the newest conv boundary) in a
slot pool sized per DP group (max_running_requests / attention DP + 16 slots). The request's state is dropped when it
finishes (`note_request_finished`) and states that leave the running batch are evicted after ARC_DFLASH_EVICT_CYCLES
decode cycles (a request that comes back always re-enters through prefill, which rebuilds its window).

Prefill only refreshes the last ARC_DFLASH_PREFILL_TAIL tokens of each extend chunk (default 2048): the drafter's
receptive field is layers x window, but its state only steers proposals, and rejection sampling keeps the output
exactly target-distributed whatever the draft does. A chunk that does not continue the request's cached cursor
(session resume, HiSparse replay, retraction) starts a fresh window from the chunk itself.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from . import drafter_dir, fast_drafter, serving
from .heads import TargetEmbedding, build_head
from .model import MTP3, CleanCache, Config, LayerCache
from .state_pool import RingStatePool, StatePool

logger = logging.getLogger(__name__)

LOG_EVERY = int(os.environ.get("ARC_DFLASH_LOG_EVERY", "500"))
GRAPH_SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96)


def _env_int(name, default):
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def moe_backend(device) -> str:
    value = os.environ.get("ARC_DFLASH_MOE")
    if value is None:
        return "cutlass" if torch.cuda.get_device_capability(device)[0] >= 10 else "marlin"
    if value not in ("cutlass", "marlin"):
        raise ValueError(f"ARC_DFLASH_MOE must be cutlass or marlin, got {value!r}")
    return value


def build_drafter(directory: Path, device, target_model, model_dir: Path, keep, group, backend: str, calib):
    """The exported drafter with routers pruned to `keep`, routed experts in NVFP4 (`backend`), embedding from the
    target and the target's lm_head (heads.build_head). On a group of N > 1 ranks: this rank's EP slice of the
    experts, and (ARC_DFLASH_ATTN_TP) its share of the attention heads."""
    from safetensors import safe_open

    c = Config(**json.loads((directory / "config.json").read_text()))
    manifest = json.loads((directory / "initialization_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"{directory}: incomplete export")
    if c.block_size != 8:
        raise ValueError(f"drafter block {c.block_size}: this worker is written for the exported B8 drafter")
    rank, size = group.rank_in_group, group.world_size
    if backend == "cutlass" and size != 1:
        raise ValueError(f"ARC_DFLASH_MOE=cutlass runs on one rank; the attention-TP group has {size} (use marlin)")
    if backend == "cutlass" and calib is None:
        raise ValueError("ARC_DFLASH_MOE=cutlass needs ARC_DFLASH_CALIB (GEMM2 input amax per layer)")
    with torch.device("meta"):
        model = MTP3(c)
    model.embed_tokens = TargetEmbedding(target_model.get_input_embeddings())
    for i, layer in enumerate(model.layers):
        kept = len(keep[str(i)])
        if kept % size:
            raise ValueError(f"layer {i}: {kept} kept experts do not split over the attention-TP group of {size}")
        layer.mlp.gate = nn.Linear(c.hidden_size, kept, bias=False, device="meta")
        layer.mlp.experts = nn.Module()  # replaced below
    model.lm_head = nn.Module()  # replaced below
    model = model.to_empty(device=device).to(torch.bfloat16)
    loaded = set()
    full_gates = {}  # ARC_DFLASH_ROUTER_STATS: each layer's whole router, for statistics over every expert
    # cutlass reads the kept experts on the GPU (as the G4 drafter always has); marlin streams this rank's experts
    # from the CPU mmap a few at a time (a full BF16 layer is 4.7 GiB, more than an L4 has left at this point)
    read_device = str(device) if backend == "cutlass" else "cpu"
    for filename in manifest["files"]:
        with safe_open(str(directory / filename), framework="pt", device=read_device) as f:
            for key in f.keys():
                if key in ("embed_tokens.weight", "lm_head.weight"):
                    continue
                if key.endswith("mlp.experts.gate_up_proj") or key.endswith("mlp.experts.down_proj"):
                    continue
                parts = key.split(".")
                value = f.get_tensor(key)
                if key.endswith("mlp.gate.weight"):
                    if os.environ.get("ARC_DFLASH_ROUTER_STATS"):
                        full_gates[int(parts[1])] = value.to(device=device, dtype=torch.bfloat16)
                    value = value[torch.tensor(keep[parts[1]], device=value.device)]
                with torch.no_grad():
                    model.get_parameter(key).copy_(value)
                loaded.add(key)
            for i in range(c.num_layers):
                prefix = f"layers.{i}.mlp.experts."
                if prefix + "gate_up_proj" not in f.keys():
                    continue
                kept = keep[str(i)]
                if backend == "cutlass":
                    from .nvfp4 import Nvfp4Experts

                    ids = torch.tensor(kept, device=device)
                    gate_up = f.get_tensor(prefix + "gate_up_proj")[ids]
                    down = f.get_tensor(prefix + "down_proj")[ids]
                    model.layers[i].mlp.experts = Nvfp4Experts(gate_up, down, a2_amax=calib[str(i)]["a2"] * 1.25)
                    del gate_up, down
                else:
                    from .marlin import ExpertRows, MarlinEPExperts

                    per = len(kept) // size
                    mine = kept[rank * per : (rank + 1) * per]
                    gate_up, down = f.get_slice(prefix + "gate_up_proj"), f.get_slice(prefix + "down_proj")
                    gu = ExpertRows(gate_up, mine, gate_up.get_shape()[1:])
                    dn = ExpertRows(down, mine, down.get_shape()[1:])
                    model.layers[i].mlp.experts = MarlinEPExperts(gu, dn, rank * per, len(kept),
                                                                  group if size > 1 else None, device=device)
                    del gu, dn
                    torch.cuda.empty_cache()
                loaded.update({prefix + "gate_up_proj", prefix + "down_proj"})
    with torch.device("meta"):
        expected = {n for n, _ in MTP3(c).named_parameters()}
    missing = expected - loaded - {"embed_tokens.weight", "lm_head.weight"}
    if missing:
        raise RuntimeError(f"drafter export misses {sorted(missing)[:8]}")
    if backend == "marlin":
        from .marlin import release_host_heap

        release_host_heap()
    if full_gates and backend != "marlin":
        raise ValueError("ARC_DFLASH_ROUTER_STATS is implemented for the marlin experts")
    for i, gate in full_gates.items():
        model.layers[i].mlp.experts.full_gate = gate
    model.lm_head = build_head(target_model, model_dir, device, c.vocab_size, c.hidden_size, group)
    if size > 1 and os.environ.get("ARC_DFLASH_ATTN_TP", "1") == "1":
        # attention heads split over the group (L4 pair: 12 query / 1 KV head per rank): halves the attention weights
        # and the window state pool for one all-reduce of o_proj per layer. The config the pool and blank states read
        # carries the per-rank heads too.
        from . import fused_serving

        fused_serving.ReducedLinear.forward = torch.compiler.disable(fused_serving.ReducedLinear.forward)
        fused_serving.shard_attention_for_tp(model, rank, size, group)
        model.config = replace(c, num_attention_heads=c.num_attention_heads // size,
                               num_key_value_heads=c.num_key_value_heads // size)
    model.compile_mode = fast_drafter.install(model, replicas=size, device=device)
    return model.eval().requires_grad_(False)


def build_dense_drafter(directory: Path, device, target_model, model_dir: Path, group):
    """A dense-routed drafter export: every tensor loaded as stored, embedding from the target and the target's
    lm_head. A `<module>.weight` stored as e4m3 must come with `<module>.weight_scale_inv` and becomes an
    Fp8BlockLinear; the set of FP8 modules must match the config's quantization record."""
    import fnmatch

    from safetensors import safe_open

    from .fp8 import BLOCK, AllReduced, Fp8BlockLinear

    # the runner's device is "cuda" without an index; safetensors would read "cuda" as cuda:0 on every rank (on L4 the
    # other three ranks then put their drafter on GPU 0 and the TP group's KV budget fails there): name this rank's GPU
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the dense drafter loads onto a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    raw = json.loads((directory / "config.json").read_text())
    quant = raw.pop("quantization", None)
    c = Config(**raw)
    manifest = json.loads((directory / "initialization_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"{directory}: incomplete export")
    if c.ffn_type != "dense_shared" or c.block_size != 8:
        raise ValueError(f"{directory}: not a dense B8 drafter (ffn_type {c.ffn_type}, block {c.block_size})")
    rank, size = group.rank_in_group, group.world_size
    shard = size > 1 and os.environ.get("ARC_DFLASH_ATTN_TP", "1") == "1"
    kv_rep = 1  # ranks sharing one KV head
    if shard:
        if c.num_attention_heads % size:
            raise ValueError(f"{c.num_attention_heads} query heads do not divide across {size} ranks")
        if c.num_key_value_heads % size:
            # more ranks than KV heads (L4 pure TP4: 24 query / 2 KV heads over 4): each KV head is replicated on
            # size / num_kv ranks, each holding a contiguous part of that head's query group (GQA groups stay whole)
            if size % c.num_key_value_heads:
                raise ValueError(f"{c.num_key_value_heads} KV heads neither divide nor tile {size} ranks")
            kv_rep = size // c.num_key_value_heads
    heads, kv, dim = c.num_attention_heads // size, max(1, c.num_key_value_heads // size), c.head_dim

    def attention_slice(name):
        """(axis, start, count) of this rank's part of an attention projection, or None (kept whole)."""
        if not shard or ".self_attn." not in name:
            return None
        proj = name.rsplit(".", 1)[-1]
        if proj == "q_proj":  # [head, 2 * head_dim] rows per head: the value and its gate, contiguous
            return 0, rank * heads * 2 * dim, heads * 2 * dim
        if proj in ("k_proj", "v_proj"):
            return 0, (rank // kv_rep) * kv * dim, kv * dim
        if proj == "o_proj":
            return 1, rank * heads * dim, heads * dim
        return None

    def narrow(t, cut):
        axis, start, count = cut
        return t.narrow(axis, start, count)

    def narrow_fp8(weight, scale, cut):
        axis, start, count = cut
        if start % BLOCK or count % BLOCK:
            raise ValueError(f"attention slice {cut} cuts through a {BLOCK}x{BLOCK} FP8 block")
        return weight.narrow(axis, start, count), scale.narrow(axis, start // BLOCK, count // BLOCK)

    if quant is not None and quant.get("method") != "block_fp8":
        raise ValueError(f"{directory}: quantization {quant.get('method')!r}; this loader reads block_fp8 only")
    with torch.device("meta"):
        model = MTP3(c)
    model.embed_tokens = TargetEmbedding(target_model.get_input_embeddings())
    model.lm_head = nn.Module()  # replaced below
    model = model.to_empty(device=device).to(torch.bfloat16)
    loaded, fp8_modules = set(), []
    for filename in manifest["files"]:
        with safe_open(str(directory / filename), framework="pt", device=str(device)) as f:
            keys = set(f.keys())
            for key in sorted(keys):
                if key in ("embed_tokens.weight", "lm_head.weight"):
                    raise ValueError(f"{directory}: a deployment export does not carry {key} (shared with the target)")
                if key.endswith(".weight_scale_inv"):
                    if key[: -len("_scale_inv")] not in keys:
                        raise ValueError(f"{directory}: {key} without its weight")
                    continue
                value = f.get_tensor(key)
                name = key[: -len(".weight")] if key.endswith(".weight") else None
                cut = attention_slice(name) if name is not None else None
                if value.dtype == torch.float8_e4m3fn:
                    if name is None or name + ".weight_scale_inv" not in keys:
                        raise ValueError(f"{directory}: FP8 tensor {key} without weight_scale_inv")
                    old = model.get_submodule(name)
                    if not isinstance(old, nn.Linear) or tuple(old.weight.shape) != tuple(value.shape):
                        raise ValueError(f"{directory}: {key} {tuple(value.shape)} does not replace {type(old).__name__}")
                    scale = f.get_tensor(name + ".weight_scale_inv")
                    if cut is not None:
                        value, scale = narrow_fp8(value, scale, cut)
                    parent, _, child = name.rpartition(".")
                    setattr(model.get_submodule(parent), child, Fp8BlockLinear(value, scale))
                    fp8_modules.append(name)
                elif cut is not None:  # a BF16 attention projection: this rank's slice replaces the whole matrix
                    linear = model.get_submodule(name)
                    linear.weight = nn.Parameter(narrow(value, cut).contiguous(), requires_grad=False)
                    linear.in_features, linear.out_features = linear.weight.shape[1], linear.weight.shape[0]
                else:
                    with torch.no_grad():
                        model.get_parameter(key).copy_(value)
                loaded.add(key)
    with torch.device("meta"):
        expected = {n for n, _ in MTP3(c).named_parameters()}
    missing = expected - loaded - {"embed_tokens.weight", "lm_head.weight"}
    if missing or loaded - expected:
        raise RuntimeError(f"dense drafter export misses {sorted(missing)[:8]}, extra {sorted(loaded - expected)[:8]}")
    patterns = [] if quant is None else quant["modules"]
    # after the replacement, a declared module still in nn.Linear was stored in BF16
    declared = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and any(fnmatch.fnmatch(n, p) for p in patterns)]
    if declared:
        raise RuntimeError(f"{directory}: config declares FP8 for {declared[:4]}... but they are stored in BF16")
    stray = [n for n in fp8_modules if not any(fnmatch.fnmatch(n, p) for p in patterns)]
    if stray:
        raise RuntimeError(f"{directory}: FP8 modules {stray[:4]} are not declared in config quantization.modules")
    model.lm_head = build_head(target_model, model_dir, device, c.vocab_size, c.hidden_size, group)
    if shard:
        # the per-rank heads: the attention modules, the window state pool and the blank states all read them
        AllReduced.forward = torch.compiler.disable(AllReduced.forward)
        for layer in model.layers:
            attention = layer.self_attn
            attention.c = replace(c, num_attention_heads=heads, num_key_value_heads=kv)
            want = {"q_proj": (heads * 2 * dim, c.hidden_size), "k_proj": (kv * dim, c.hidden_size),
                    "v_proj": (kv * dim, c.hidden_size), "o_proj": (c.hidden_size, heads * dim)}
            for proj, shape in want.items():
                got = tuple(getattr(attention, proj).weight.shape)
                if got != shape:
                    raise RuntimeError(f"rank {rank}: {proj} is {got} after the attention split, want {shape}")
            attention.o_proj = AllReduced(attention.o_proj, group)
        model.config = replace(c, num_attention_heads=heads, num_key_value_heads=kv)
    model.fp8_modules = tuple(fp8_modules)
    model.compile_mode = fast_drafter.install(model, replicas=size, device=device)
    return model.eval().requires_grad_(False)


class SlotStager:
    """Request slot ids to the device without a stream sync. `torch.tensor(list, device="cuda")` copies from
    pageable memory, and CUDA synchronizes the stream before a pageable host-to-device copy: twice per decode cycle
    that drained the GPU and cost the overlap scheduler its lead (1.5-1.9 ms of idle GPU per cycle at c55-c110,
    against 0.1 ms with the native MTP). Pinned buffers in a ring instead, each reused only after the copy that
    last read it has executed (an event wait that the ring's depth makes a no-op in steady state)."""

    def __init__(self, depth, size, device):
        self.buffers = [torch.empty(size, dtype=torch.long, pin_memory=True) for _ in range(depth)]
        self.events = [None] * depth
        self.next = 0
        self.device = device

    def __call__(self, values):
        i = self.next
        self.next = (i + 1) % len(self.buffers)
        if len(values) > self.buffers[i].shape[0]:
            raise RuntimeError(f"{len(values)} slots staged, buffers hold {self.buffers[i].shape[0]}")
        if self.events[i] is not None:
            self.events[i].synchronize()
        host = self.buffers[i][: len(values)]
        host.numpy()[:] = values
        device = host.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.events[i] = event
        return device


class DFlashDraftWorker:
    """Duck-types EagleDraftWorker for EAGLEWorkerV2 (see its isinstance checks)."""

    is_dflash5 = True
    topk = 1

    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        from sglang.srt.distributed.parallel_state import get_attn_tp_group
        from sglang.srt.layers.dp_attention import get_attention_dp_size
        from sglang.srt.runtime_context import get_schedule, get_spec

        spec = get_spec()
        # Verify width K: the anchor plus K - 1 of the drafter's 7 proposals (2 <= K <= 8).
        self.width = spec.speculative_num_draft_tokens
        self.steps = self.width - 1
        if spec.speculative_eagle_topk != 1 or spec.speculative_num_steps != self.steps or not 2 <= self.width <= 8:
            raise ValueError("DFlash needs --speculative-eagle-topk 1 and --speculative-num-steps K-1 "
                             "--speculative-num-draft-tokens K with 2 <= K <= 8")
        self.group = get_attn_tp_group()
        dp = get_attention_dp_size()
        self.server_args, self.ps, self.target_worker = server_args, ps, target_worker
        self.device = target_worker.model_runner.device
        self.speculative_num_steps, self.speculative_num_draft_tokens = self.steps, self.width
        total = get_schedule().max_running_requests
        if total % dp:
            raise ValueError(f"max_running_requests {total} does not split over attention DP {dp}")
        self.max_running_requests = total // dp
        self.use_rejection_sampling = bool(spec.speculative_use_rejection_sampling)
        self.check_agree = os.environ.get("ARC_DFLASH_CHECK_AGREE") == "1"
        directory = Path(drafter_dir())
        raw = json.loads((directory / "config.json").read_text())
        dense = raw.get("ffn_type", "moe") == "dense_shared"
        if dense:
            for name in ("ARC_DFLASH_KEEP", "ARC_DFLASH_CALIB", "ARC_DFLASH_MOE", "ARC_DFLASH_ROUTER_STATS"):
                if os.environ.get(name):
                    raise ValueError(f"{name} is for the MoE drafter; {directory} is a dense export")
            keep, calib, self.backend = None, None, "dense"
        else:
            keep = json.loads(Path(os.environ["ARC_DFLASH_KEEP"]).read_text())
            self.backend = moe_backend(self.device)
            calib = json.loads(Path(os.environ["ARC_DFLASH_CALIB"]).read_text()) if self.backend == "cutlass" else None
        self.prefill_tail = _env_int("ARC_DFLASH_PREFILL_TAIL", 2048)
        # tokens per drafter forward when a window is rebuilt (default: the whole tail in one forward)
        self.prefill_chunk = _env_int("ARC_DFLASH_PREFILL_CHUNK", self.prefill_tail)
        self.evict_cycles = _env_int("ARC_DFLASH_EVICT_CYCLES", 512)
        self.use_graphs = os.environ.get("ARC_DFLASH_GRAPH", "1") == "1"
        begin = time.monotonic()
        before = torch.cuda.memory_allocated(self.device)
        if dense:
            self.model = build_dense_drafter(directory, self.device, target_worker.model_runner.model,
                                             Path(server_args.model_path), self.group)
        else:
            self.model = build_drafter(directory, self.device, target_worker.model_runner.model,
                                       Path(server_args.model_path), keep, self.group, self.backend, calib)
        torch.cuda.synchronize(self.device)
        self.weight_bytes = torch.cuda.memory_allocated(self.device) - before
        self._weight_load_time = time.monotonic() - begin
        self.vocab = self.model.config.vocab_size
        hf = target_worker.model_runner.model_config.hf_config
        image_token = getattr(hf, "image_token_id", None)
        if image_token is None or not 0 <= image_token < self.vocab:
            raise RuntimeError(f"target config has no usable image_token_id ({image_token})")
        self.image_token = image_token
        # ARC_DFLASH_ROUTER_STATS=<prefix>: accumulate each layer's summed top-k router weight per exported expert on
        # live traffic (CUDA-graph safe) and write <prefix>.rank<global rank>.json every ARC_DFLASH_LOG_EVERY cycles;
        # the input for choosing a keep set on the deployment's own workload
        self.router_stats = os.environ.get("ARC_DFLASH_ROUTER_STATS")
        if self.router_stats:
            for layer in self.model.layers:
                experts = layer.mlp.experts
                experts.router_mass = torch.zeros(experts.full_gate.shape[0], dtype=torch.float32, device=self.device)
        self.keep = keep
        self.dumped = 0
        # Allocated here, before the target measures its KV budget: the whole drafter state is this constant.
        # ARC_DFLASH_POOL: ring (default; writes only committed rows) | shift (whole-window gather/scatter)
        layout = os.environ.get("ARC_DFLASH_POOL", "ring")
        pool_cls = {"ring": RingStatePool, "shift": StatePool}[layout]
        self.pool = pool_cls(self.model, slots=self.max_running_requests + 16, max_batch=self.max_running_requests,
                             device=self.device, steps=self.steps)
        torch.cuda.empty_cache()  # the BF16 staging must not count as used when the target sizes its pools
        if os.environ.get("ARC_DFLASH_MEM_PROFILE"):  # analysis runs only: records every allocation (slows the host)
            from . import mem_profile

            mem_profile.start(os.environ["ARC_DFLASH_MEM_PROFILE"], self.device)
            logger.info("DFlash: GPU memory profile recording to %s", os.environ["ARC_DFLASH_MEM_PROFILE"])
        self.last_seen: dict[str, int] = {}
        self.stage_slots = SlotStager(8, self.max_running_requests, self.device)
        self.events = collections.deque()
        self.prefill_stats = [0, 0.0, 0]
        self.cycle = 0
        self.evicted = 0
        self.parents = torch.arange(-1, self.steps - 1, dtype=torch.long, device=self.device).repeat(self.max_running_requests + 1, 1)
        self.score_index = torch.arange(self.steps, dtype=torch.long, device=self.device).repeat(self.max_running_requests + 1, 1)
        from sglang.srt.speculative.eagle_utils import default_tree_mask_mode

        self.tree_mask_mode = default_tree_mask_mode()
        self.draft_runner = None
        self.draft_runners = []
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        self.draft_attn_backend = None
        self.draft_extend_attn_backend = None
        self.graph_seconds = 0.0
        self._prof = None
        if dense:
            kept = [f"dense {self.model.config.dense_intermediate_size}"] * self.model.config.num_layers
            local = [f"fp8 {len(self.model.fp8_modules)} linears"]
        else:
            experts = [self.model.layers[i].mlp.experts for i in range(self.model.config.num_layers)]
            kept = [len(keep[str(i)]) for i in range(self.model.config.num_layers)]
            local = [e.num_local if self.backend == "marlin" else e.num_experts for e in experts]
        logger.info(
            "DFlash drafter %s on attention-TP group %s (rank %d/%d, attention DP %d), %s experts: %d layers, experts "
            "kept %s (%s on this rank), weights %.2f GiB, state pool %d slots %.2f GiB (window %d, %s layout), "
            "lm_head %s, loaded in %.1f s; prefill tail %d, rejection sampling %s, verify width %d, compile %s",
            directory, list(self.group.ranks), self.group.rank_in_group, self.group.world_size, dp, self.backend,
            self.model.config.num_layers, kept, local,
            self.weight_bytes / 2**30, self.pool.slots, self.pool.bytes() / 2**30, self.model.config.window, layout,
            type(self.model.lm_head).__name__, self._weight_load_time, self.prefill_tail, self.use_rejection_sampling,
            self.width, self.model.compile_mode,
        )

    # ---- EagleDraftWorker surface used by EAGLEWorkerV2 / the scheduler ----------------------------------------
    def draft_tp_context(self, _group):
        return contextlib.nullcontext()

    @property
    def graph_memory_usage(self):
        return {}

    @property
    def graph_time_usage(self):
        return {"dflash_draft": self.graph_seconds} if self.graph_seconds else {}

    @property
    def weight_load_time(self):
        return self._weight_load_time

    @property
    def preloaded_weights_bytes(self):
        return 0

    def alloc_memory_pool(self, **_kwargs):
        pass

    def init_attention_backends(self):
        pass

    def init_cuda_graphs(self):
        fast_drafter.warm(self)
        if not self.use_graphs:
            return
        sizes = sorted({s for s in GRAPH_SIZES if s < self.max_running_requests} | {self.max_running_requests})
        begin = time.monotonic()
        free_before = torch.cuda.mem_get_info(self.device)[0]
        # the group's all-reduce / all-gather inside the graphs go through pynccl, which SGLang enables only here
        capture = self.group.graph_capture() if self.group.world_size > 1 else contextlib.nullcontext()
        with capture:
            self.pool.capture(sizes)
        torch.cuda.synchronize(self.device)
        self.graph_seconds = time.monotonic() - begin
        logger.info(
            "DFlash draft CUDA graphs for batch sizes %s in %.1f s, %.2f GiB",
            sizes, self.graph_seconds, (free_before - torch.cuda.mem_get_info(self.device)[0]) / 2**30,
        )

    def _rebuild_topk1_chain_buffers(self):
        pass

    def live_states(self):
        return self.pool.live()

    def note_request_finished(self, rid):
        self.pool.release(rid)
        self.last_seen.pop(rid, None)

    def clear_cache_pool(self):
        for rid in list(self.pool.slot_of):
            self.note_request_finished(rid)

    def _unpad(self, hidden, rows: int, what: str):
        """Attention TP pads a batch's token rows to a multiple of the group size (trailing rows); the target GR
        arrives with them. Drop exactly that padding; anything else is a real mismatch and raises. A group of one
        has no padding."""
        if hidden is None or hidden.shape[0] == rows:
            return hidden
        extra = hidden.shape[0] - rows
        if not 0 < extra < self.group.world_size:
            raise RuntimeError(f"DFlash: {what} GR has {hidden.shape[0]} rows for {rows} tokens "
                               f"(attention-TP padding is < {self.group.world_size})")
        return hidden[:rows]

    # ---- prefill ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _draft_extend_for_prefill(self, batch, target_hidden_states, next_token_ids, mm_input_embeds=None):
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        if batch.forward_mode.is_idle():
            return EagleDraftInput(
                topk_p=torch.empty((0, 1), dtype=torch.float32, device=self.device),
                topk_index=torch.empty((0, 1), dtype=torch.int64, device=self.device),
                bonus_tokens=next_token_ids, num_tokens_per_req=1, num_tokens_for_logprob_per_req=1,
            )
        c = self.model.config
        ids = batch.input_ids
        target_hidden_states = self._unpad(target_hidden_states, ids.numel(), "prefill")
        if target_hidden_states is None or tuple(target_hidden_states.shape) != (ids.numel(), c.hc_count * c.hidden_size):
            raise RuntimeError(
                f"target prefill GR {None if target_hidden_states is None else tuple(target_hidden_states.shape)} "
                f"!= [{ids.numel()}, {c.hc_count * c.hidden_size}]"
            )
        gr = target_hidden_states.view(-1, c.hc_count, c.hidden_size)
        # Multimodal prompts carry per-image hash values in place of the image pad ids (SGLang's radix keys);
        # the target reads vision embeddings there. The drafter is text-only: it sees <|image_pad|> instead.
        ids = torch.where((ids >= 0) & (ids < self.vocab), ids, torch.full_like(ids, self.image_token))
        offset = 0
        begin = time.perf_counter()
        for req, length, start in zip(batch.reqs, batch.extend_lens, batch.prefix_lens, strict=True):
            self._prefill_one(req.rid, ids[offset : offset + length], gr[offset : offset + length], int(start))
            self.last_seen[req.rid] = self.cycle
            offset += length
        if offset != ids.numel():
            raise RuntimeError("extend lengths do not cover the batch's tokens")
        # host time (the refresh is eager, and _prefill_one reads cursors on the host, so this tracks GPU time)
        self.prefill_stats[0] += 1
        self.prefill_stats[1] += time.perf_counter() - begin
        self.prefill_stats[2] += offset
        if self.prefill_stats[0] % 200 == 0:
            logger.info("DFlash prefill: %d extend batches, %.1f s in the drafter refresh, %d tokens (%.2f ms per batch)",
                        self.prefill_stats[0], self.prefill_stats[1], self.prefill_stats[2],
                        1e3 * self.prefill_stats[1] / self.prefill_stats[0])
        bs = len(batch.reqs)
        return EagleDraftInput(
            topk_p=torch.ones((bs, 1), dtype=torch.float32, device=self.device),
            topk_index=next_token_ids.view(bs, 1).to(torch.int64),
            bonus_tokens=next_token_ids, num_tokens_per_req=1, num_tokens_for_logprob_per_req=1,
        )

    def _prefill_one(self, rid, tokens, gr, start):
        """tokens/gr are this chunk at positions start .. start+n-1; gr[t] is the target GR at that position."""
        n = tokens.shape[0]
        slot = self.pool.slot_of.get(rid)
        continues = (
            slot is not None
            and int(self.pool.pending_length[slot]) == 0
            and int(self.pool.last_position[slot]) + 1 == start
        )
        if continues and n <= self.prefill_tail:
            state = self.pool.read_request(rid)
            positions = torch.arange(start, start + n, device=self.device)[None]
            span = torch.cat((state.last_gr[:, None], gr[None]), 1)
            self.pool.write_request(rid, state.refresh_committed(self.model, tokens[None].long(), span, positions))
            return
        if start == 0 and n == 1:
            # A one-token prompt (SGLang's /health check sends one): no position has a predecessor GR, so the
            # window starts blank; the next cycle's anchor at position 1 conditions on this token's GR.
            self.pool.write_request(rid, self._blank_state(gr[None, 0].clone(), start))
            return
        # A fresh window from the chunk's own tail. Token t pairs with the GR of t - 1; the first kept token's
        # predecessor is in the chunk unless the tail starts the chunk, where the cached prefix holds it and the
        # token's own GR stands in (it only steers proposals).
        keep_from = max(0, n - self.prefill_tail)
        if start + keep_from == 0:
            keep_from = 1  # position 0 has no predecessor at all
        kept = tokens[keep_from:].long()[None]
        pred = gr[keep_from - 1 : n - 1] if keep_from > 0 else torch.cat((gr[:1], gr[: n - 1]))
        positions = torch.arange(start + keep_from, start + n, device=self.device)[None]
        cache = None
        for a in range(0, kept.shape[1], self.prefill_chunk):
            b = min(a + self.prefill_chunk, kept.shape[1])
            _, cache = self.model.refresh(kept[:, a:b], pred[None, a:b], positions[:, a:b], cache)
        self.pool.write_request(rid, serving.DraftRequest(
            serving.decode_cache(cache, self.model.config.window), gr[None, -1].clone(), positions[:, -1].clone()
        ))

    def _blank_state(self, last_gr, position):
        """A window with one zero K/V slot at `position` and padding elsewhere: decode needs the cache to end
        at the cursor. The zero key only dilutes attention for the request's first block of proposals."""
        c = self.model.config
        positions = torch.full((1, c.window), -1, dtype=torch.long, device=self.device)
        positions[0, -1] = position
        kv = torch.zeros((1, c.num_key_value_heads, c.window, c.head_dim), dtype=torch.bfloat16, device=self.device)
        boundary = torch.zeros((1, 1, c.hidden_size), dtype=torch.bfloat16, device=self.device)
        cache = CleanCache(positions, tuple(LayerCache(kv, kv, (boundary,) * 4) for _ in range(c.num_layers)))
        return serving.DraftRequest(cache, last_gr, torch.tensor([position], device=self.device))

    # ---- decode ------------------------------------------------------------------------------------------------
    def _slots(self, rids):
        missing = [rid for rid in rids if rid not in self.pool.slot_of]
        if missing:
            raise RuntimeError(f"DFlash: decode for requests without a draft state: {missing[:4]}")
        return self.stage_slots([self.pool.slot_of[rid] for rid in rids])

    @torch.no_grad()
    def draft(self, batch):
        from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input

        draft_input = batch.spec_info
        bs = batch.seq_lens.shape[0]
        if batch.forward_mode.is_idle() or bs == 0:
            return build_eagle_verify_input(
                batch, draft_input, None, None, None, None, target_worker=self.target_worker, topk=1,
                num_steps=self.steps, num_draft_tokens=self.width, tree_mask_mode=self.tree_mask_mode, device=self.device,
            )
        self.cycle += 1
        self._profile_step(True)
        rids = [req.rid for req in batch.reqs]
        for rid in rids:
            self.last_seen[rid] = self.cycle
        self._evict()
        slots = self._slots(rids)
        if self.cycle % LOG_EVERY == 0:
            logger.info(
                "DFlash cycle %d: batch %d, live draft states %d of %d slots (evicted %d so far), torch allocated "
                "%.2f GiB, reserved %.2f GiB",
                self.cycle, bs, self.pool.live(), self.pool.slots - 1, self.evicted,
                torch.cuda.memory_allocated(self.device) / 2**30, torch.cuda.memory_reserved(self.device) / 2**30,
            )
        anchor = draft_input.bonus_tokens.view(-1).to(torch.long)
        temperature = batch.sampling_info.temperatures.view(-1).float()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        tokens, probs = self.pool.step(slots, anchor, temperature)
        end.record()
        self._timing(start, end, bs)
        self._profile_step(False)
        if self.router_stats and self.cycle % LOG_EVERY == 0 and self.cycle != self.dumped:
            self.dumped = self.cycle
            self._dump_router_stats()
        if self.check_agree and self.group.world_size > 1:
            both = self.group.all_gather(tokens.contiguous()[None], dim=0)
            if not all(torch.equal(both[0], both[r]) for r in range(1, self.group.world_size)):
                raise RuntimeError(f"DFlash: attention-TP group {list(self.group.ranks)} proposed different tokens "
                                   f"at cycle {self.cycle}")
        return build_eagle_verify_input(
            batch, draft_input, self.parents[:bs], self.score_index[:bs], tokens.clone(),
            probs if self.use_rejection_sampling else None,
            target_worker=self.target_worker, topk=1, num_steps=self.steps, num_draft_tokens=self.width,
            tree_mask_mode=self.tree_mask_mode, device=self.device,
        )

    def _timing(self, start, end, bs):
        """GPU ms of each draft step and of each whole cycle (draft start to the next draft start), averaged over
        the cycles whose events have landed by the time the line is logged; event queries never block."""
        self.events.append((start, end, bs))
        if len(self.events) < 2 * LOG_EVERY and self.cycle % LOG_EVERY:
            return
        draft, cycle, rows = [], [], []
        while len(self.events) > 1 and self.events[1][0].query():
            (s0, e0, b0), (s1, _, _) = self.events[0], self.events[1]
            draft.append(s0.elapsed_time(e0))
            cycle.append(s0.elapsed_time(s1))
            rows.append(b0)
            self.events.popleft()
        if draft and self.cycle % LOG_EVERY == 0:
            cycle.sort()
            logger.info(
                "DFlash timing over %d cycles: draft %.2f ms mean, cycle %.2f ms median (draft %.1f%%), batch %.1f mean",
                len(draft), sum(draft) / len(draft), cycle[len(cycle) // 2],
                100 * sum(draft) / sum(cycle), sum(rows) / len(rows),
            )

    def _evict(self):
        stale = [rid for rid, seen in self.last_seen.items() if self.cycle - seen > self.evict_cycles]
        for rid in stale:
            self.note_request_finished(rid)
        self.evicted += len(stale)

    def _dump_router_stats(self):
        mass = {str(i): layer.mlp.experts.router_mass.tolist() for i, layer in enumerate(self.model.layers)}
        path = Path(f"{self.router_stats}.rank{torch.distributed.get_rank()}.json")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cycle": self.cycle, "keep": self.keep, "router_mass": mass}))
        tmp.replace(path)

    def _profile_step(self, begin: bool):
        """ARC_DFLASH_PROFILE=<prefix>[:start[:cycles]]: torch.profiler over `cycles` draft steps from cycle `start`
        (run with ARC_DFLASH_GRAPH=0 so kernels are visible); writes <prefix>.rank<r>.txt (CUDA time table)."""
        spec = os.environ.get("ARC_DFLASH_PROFILE")
        if not spec:
            return
        prefix, start, cycles = (spec.split(":") + ["2000", "20"])[:3]
        start, cycles = int(start), int(cycles)
        prof = self._prof
        acts = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        if begin and self.cycle == start and prof is None:
            self._prof = torch.profiler.profile(activities=acts)
            self._prof.__enter__()
        elif begin and prof:
            self._prof.toggle_collection_dynamic(True, acts)  # collect only inside draft steps
        elif not begin and prof and self.cycle < start + cycles:
            torch.cuda.synchronize(self.device)
            self._prof.toggle_collection_dynamic(False, acts)
        elif not begin and prof and self.cycle >= start + cycles:
            torch.cuda.synchronize(self.device)
            self._prof.__exit__(None, None, None)
            table = self._prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            Path(f"{prefix}.rank{torch.distributed.get_rank()}.txt").write_text(
                f"{cycles} draft steps from cycle {start}\n{table}\n")
            self._prof = False

    @torch.no_grad()
    def _draft_extend_for_decode(self, batch, batch_result):
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        bs = batch.seq_lens.shape[0]
        next_draft_input = batch_result.next_draft_input
        next_draft_input.topk_p = torch.ones((bs, 1), dtype=torch.float32, device=self.device)
        next_draft_input.topk_index = next_draft_input.bonus_tokens.view(bs, 1).to(torch.int64)
        next_draft_input.hidden_states = None
        next_draft_input.draft_probs = None
        if batch.forward_mode.is_idle() or bs == 0:
            return
        verify = batch.spec_info
        if not isinstance(verify, EagleVerifyInput):
            raise RuntimeError(f"DFlash draft extend expected the verify input, got {type(verify).__name__}")
        c = self.model.config
        k = self.width
        gr = self._unpad(batch_result.logits_output.hidden_states, bs * k, "verify")
        if gr is None or tuple(gr.shape) != (bs * k, c.hc_count * c.hidden_size):
            raise RuntimeError(f"verify GR {None if gr is None else tuple(gr.shape)} != [{bs * k}, {c.hc_count * c.hidden_size}]")
        gr = gr.view(bs, k, c.hc_count, c.hidden_size)
        tokens = verify.draft_token.view(bs, k).to(torch.long)
        positions = batch.seq_lens.view(bs, 1).to(torch.long) + torch.arange(k, device=self.device)
        slots = self._slots([req.rid for req in batch.reqs])
        predecessors = torch.cat((self.pool.last_gr[slots][:, None], gr), 1)
        self.pool.defer_commit(slots, tokens, predecessors, positions, batch_result.accept_lens.to(torch.long))
