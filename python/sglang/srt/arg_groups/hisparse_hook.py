from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "triton", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if get_platform().is_hip:
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if get_platform().is_hip:
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""

    cfg = resolving_view(server_args)
    if not cfg.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = model_config_of(server_args).hf_config
    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

    if is_qwen_qsa(hf_config):
        validate_qsa_hisparse(server_args)
        return
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_hip = get_platform().is_hip
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert cfg.disable_radix_cache, (
        "Hierarchical sparse attention currently requires --disable-radix-cache."
    )

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)


# ---------------------------------------------------------------------------
# arc-forge QSA HiSparse (h05, M1a design §7): the one supported shape, then a
# source manifest check. SOURCE_SHA256 lists `<sha256>  <path>` lines (paths
# relative to the sglang package) for every installed file the QSA HiSparse
# path uses; the JIT kernels compile from the installed .cuh files, so a stale
# or partially patched tree must not start.
# ---------------------------------------------------------------------------
QSA_HISPARSE_MANIFEST_ENV = "ARC_HISPARSE_QSA_MANIFEST"


def validate_qsa_hisparse(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.model_override_base import mamba_extra_buffer_of
    from sglang.srt.environ import envs

    view = resolved_view(server_args)
    # arc-forge M1b h09: the session cache is the one prefix reuse QSA HiSparse has.
    session = view.radix_cache_backend == QSA_HISPARSE_SESSION_BACKEND
    nvfp4 = view.kv_cache_dtype == "nvfp4"
    qsa_nvfp4 = (
        os.environ.get("ARC_QSA_NVFP4") == "1" and os.environ.get("ARC_FULL_NVFP4") != "1"
    )
    need = {
        f"no --radix-cache-backend other than {QSA_HISPARSE_SESSION_BACKEND}": view.radix_cache_backend
        in (None, QSA_HISPARSE_SESSION_BACKEND),
        "--tp-size 1": view.tp_size == 1,
        "--pp-size 1": view.pp_size == 1,
        "--dp-size 1 and no DP attention": view.dp_size == 1
        and not view.enable_dp_attention,
        "no context parallelism (--dcp-size / --attn-cp-size 1)": view.dcp_size == 1
        and view.attn_cp_size == 1,
        "--disable-radix-cache": bool(view.disable_radix_cache),
        (
            "--enable-session-radix-cache with the session cache, not without"
        ): bool(view.enable_session_radix_cache) == session,
        # arc-forge M5 h17: speculative decoding in the shape _qsa_spec_needs lists
        "no non-speculative ReplaySSM (--enable-linear-replayssm)": not view.enable_linear_replayssm,
        "--enable-linear-replayssm-spec only with speculative decoding": view.speculative_algorithm
        is not None
        or not view.enable_linear_replayssm_spec,
        "no mamba extra buffer": not mamba_extra_buffer_of(view),
        "no --enable-mixed-chunk": not view.enable_mixed_chunk,
        "--max-running-requests set": view.max_running_requests is not None,
        "--kv-cache-dtype fp8_e4m3 or nvfp4": view.kv_cache_dtype in ("fp8_e4m3", "nvfp4"),
        # NVFP4 (arc-forge h12): the QSA packed-read path, no dequant workspace
        "ARC_QSA_NVFP4=1 and no ARC_FULL_NVFP4 with NVFP4 KV": not nvfp4 or qsa_nvfp4,
        "no PD disaggregation": view.disaggregation_mode == "null",
        "no hierarchical cache": not view.enable_hierarchical_cache,
        "no unified memory pool": not view.enable_unified_memory,
        "no page-major KV layout": not view.enable_page_major_kv_layout,
        "no post-capture KV sizing": not envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.get(),
    }
    if session:
        need.update(_qsa_session_needs(view))
    if view.speculative_algorithm is not None:
        need.update(_qsa_spec_needs(view))
    missing = [what for what, ok in need.items() if not ok]
    if missing:
        raise ValueError(
            "--enable-hisparse on a QSA model requires: " + "; ".join(missing)
        )
    tree = verify_qsa_hisparse_manifest()
    # warning: resolution runs before the server configures log levels
    logger.warning("ARC_HISPARSE_QSA_TREE=%s", tree)


QSA_HISPARSE_SESSION_BACKEND = "qsa_hisparse_session"


def _qsa_session_needs(view) -> dict:
    """M1b h09: session cache (design §6 hisparse_hook row). The snapshot interval T
    must be a multiple of the 64-token page (resume points are page aligned, §4.2),
    and the host reserve is stated, not guessed (R3)."""
    import json

    user = json.loads(view.hisparse_config) if view.hisparse_config else {}
    t = user.get("qsa_snapshot_interval")
    reserve = user.get("host_reserve_gib")
    return {
        # the registry would wrap the cache in a StreamingSession
        "no --enable-streaming-session": not view.enable_streaming_session,
        # cache-aware policies call match_prefix outside init_next_round_input
        "--schedule-policy fcfs / lof / random / priority / routing-key": view.schedule_policy
        not in ("lpm", "dfs-weight", "hrrn"),
        "hisparse_config qsa_session_resume true": user.get("qsa_session_resume") is True,
        "hisparse_config qsa_snapshot_interval a positive multiple of 64": isinstance(t, int)
        and not isinstance(t, bool)
        and t > 0
        and t % 64 == 0,
        "hisparse_config host_reserve_gib >= 0 (host RAM left to the OS and agents)": isinstance(
            reserve, (int, float)
        )
        and not isinstance(reserve, bool)
        and reserve >= 0,
        "no --enable-dynamic-chunking": not view.enable_dynamic_chunking,
        # it frees output KV below kv_committed_len, where a record's P can sit
        "no --strip-thinking-cache": not view.strip_thinking_cache,
    }


def _qsa_spec_needs(view) -> dict:
    """M5 h17 (design §9): the speculative shape QSA HiSparse serves (K <= 8 from h19).
    The target verifies K rows per request (h16) and backs up after accept (h17) only on the
    overlap EAGLEWorkerV2 path; the draft keeps a resident plain QSA pool priced
    per logical token (fp8_e4m3 / bfloat16 only). GDN snapshots (and so session
    resume) under spec take the ReplaySSM commit's track into staging slots the
    device claims after accept (h18), which needs one crossing per step: T > K."""
    import json

    from sglang.srt.environ import envs

    user = json.loads(view.hisparse_config) if view.hisparse_config else {}
    k = view.speculative_num_draft_tokens
    t = user.get("qsa_snapshot_interval")
    return {
        "--speculative-algorithm NEXTN / EAGLE": view.speculative_algorithm
        in ("EAGLE", "NEXTN"),
        "--speculative-eagle-topk 1": view.speculative_eagle_topk == 1,
        "--speculative-num-draft-tokens = --speculative-num-steps + 1": k is not None
        and view.speculative_num_steps is not None
        and k == view.speculative_num_steps + 1,
        # h19: the QSA pending ring holds ratio + K keys (experimental-qsa 0018), the
        # HiSparse ring W = ceil((K + 3) / 4) slots; K is sized for DFlash block 8
        "--speculative-num-draft-tokens <= 8": k is not None and 1 <= k <= 8,
        "no --enable-multi-layer-eagle": not view.enable_multi_layer_eagle,
        "no --speculative-adaptive (it changes K at runtime)": not view.speculative_adaptive,
        "--speculative-draft-kv-cache-dtype fp8_e4m3 or bfloat16": view.speculative_draft_kv_cache_dtype
        in ("fp8_e4m3", "bf16", "bfloat16"),
        "--speculative-attention-mode prefill": view.speculative_attention_mode == "prefill",
        "the overlap scheduler (no --disable-overlap-schedule)": not view.disable_overlap_schedule,
        "SGLANG_RAGGED_VERIFY_MODE=static (K rows per request)": envs.SGLANG_RAGGED_VERIFY_MODE.get()
        == "static",
        "no SGLANG_SIMULATE_ACC_LEN": envs.SGLANG_SIMULATE_ACC_LEN.get() < 0,
        # h18: the verify commit tracks the state of at most one boundary per step
        "--enable-linear-replayssm-spec with GDN snapshots": t is None
        or view.enable_linear_replayssm_spec,
        "hisparse_config qsa_snapshot_interval > --speculative-num-draft-tokens": t is None
        or (isinstance(t, int) and k is not None and t > k),
    }


def qsa_hisparse_spec_capture_bs(server_args: ServerArgs, max_bs: int):
    """M5 h17 (design §16 decision 5): the decode capture list for QSA HiSparse with
    speculative decoding when --cuda-graph-bs-decode is not given, else None.

    Three runners (target verify, draft decode, draft extend) capture every listed
    batch size, and the graph memory comes out of the headroom mem_fraction_static
    leaves; the stock spec list has 33 sizes up to 110 while a c110 batch sits
    near R. Fewer sizes only cost padding at small batches. Powers of two to 32,
    then every 16, plus max_bs."""
    view = resolving_view(server_args)
    if not view.enable_hisparse or view.speculative_algorithm is None:
        return None
    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

    if not is_qwen_qsa(model_config_of(server_args).hf_config):
        return None
    capture_bs = [1, 2, 4, 8, 16, 32] + list(range(48, max_bs + 1, 16))
    return [bs for bs in capture_bs if bs <= max_bs]


def verify_qsa_hisparse_manifest() -> str:
    """Re-hash the files ARC_HISPARSE_QSA_MANIFEST lists; return sha256 of the manifest text."""
    import sglang

    path = os.environ.get(QSA_HISPARSE_MANIFEST_ENV)
    if not path:
        raise ValueError(
            f"--enable-hisparse on a QSA model requires {QSA_HISPARSE_MANIFEST_ENV}=<SOURCE_SHA256>"
        )
    text = Path(path).read_text()
    root = Path(sglang.__file__).resolve().parent
    bad = []
    for n, line in enumerate(text.splitlines(), 1):
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            raise ValueError(f"{path}:{n} is not '<sha256>  <path>': {line!r}")
        target = root / rel
        if not target.is_file():
            bad.append(f"missing  {rel}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            bad.append(f"changed  {rel}")
    if bad:
        raise RuntimeError(
            f"installed sglang at {root} does not match {path}:\n" + "\n".join(bad)
        )
    return hashlib.sha256(text.encode()).hexdigest()
