"""arc-forge: Qwen4-Exp extend (text and media) under the breakable prefill CUDA graph (opt-in, default off).

``ARC_QWEN4_EXP_PREFILL_BCG=1`` lets the breakable (BCG) prefill runner capture the Qwen4-Exp
transformer body. The body is token-wise except for four places that read the live batch's host
metadata; those run as eager breaks, everything else (HC mix/combine, projections, norms, RoPE,
MoE with its routed-experts writes, the attention-TP all-reduces) replays from the graph:

* GDN core: RadixLinearAttention's existing BCG break (as for Qwen3.5);
* QSA attention core: RadixAttention's existing BCG break, fed ``topk_indices``;
* QSA indexer: one break per QSA layer that writes the live top-k rows into a buffer the
  captured segment allocated (padding rows -1), ``Qwen4ExpAttentionDecoderLayer``;
* PLE: one break at the PLE layer that prepares the batch from the live ForwardBatch, runs the
  layer and commits the N-gram history (equivalent to the eager forward's end-of-model commit
  because the model has exactly one PLE layer), writing into a captured buffer.

The model returns ``(hidden, hc_hidden)`` from the captured body; the runner hands the stored pair
back through ``Qwen4ExpVLModel.unpack_breakable_body_output`` so ``logits_output.hidden_states``
(the GR the DFlash drafter reads) is the replayed HC stream. Batches with media replay too: the
eager outer forward runs the ViT and the media merge into the static input_embeds slot, rewrites
the media ids in the static input_ids the PLE break reads, and M-RoPE positions come from their
static slot (prefill_cuda_graph_runner.py).

Off (the default) nothing changes: Qwen4-Exp stays off the multimodal BCG allowlist, and a BCG
capture of it (an explicit ``--cuda-graph-backend-prefill breakable``) raises instead of capturing
the indexer's host-planned shapes.
"""

from __future__ import annotations

import os

ENV = "ARC_QWEN4_EXP_PREFILL_BCG"
_VALUES = {"0": False, "1": True}

ARCH = "Qwen4ExpForConditionalGeneration"


def _parse(raw: str) -> bool:
    if raw not in _VALUES:
        raise ValueError(f"{ENV}={raw!r}: expected one of {sorted(_VALUES)}")
    return _VALUES[raw]


ENABLED = _parse(os.environ.get(ENV, "0"))


def require_enabled(where: str) -> None:
    """Raise unless the opt-in is on; ``where`` names the code that found itself under BCG."""
    if not ENABLED:
        raise RuntimeError(
            f"Qwen4-Exp {where} is running under the breakable prefill CUDA graph without "
            f"{ENV}=1. The QSA indexer and PLE size their work from the live batch's host "
            f"metadata; only {ENV}=1 runs them as eager breaks. Set it, or disable the "
            "prefill CUDA graph."
        )


def validate_runner(model_runner) -> None:
    """Configurations the opt-in has been reasoned through; anything else raises.

    The target worker of SPEC=none or of the DFlash drafter (ARC_DFLASH_DRAFTER), without DP
    attention, pipeline parallelism or QSA HiSparse.
    """
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative import arc_dflash

    require_enabled("prefill CUDA graph capture")
    archs = list(model_runner.model_config.hf_config.architectures)
    if model_runner.is_draft_worker:
        raise NotImplementedError(
            f"{ENV}=1 covers the Qwen4-Exp target; the native MTP draft's breakable prefill "
            f"({archs}) is not validated. Serve SPEC=none or the DFlash drafter, or unset {ENV}."
        )
    if archs != [ARCH]:
        raise NotImplementedError(f"{ENV}=1 covers {ARCH}; got {archs}")
    if is_dp_attention_enabled():
        raise NotImplementedError(f"{ENV}=1 does not cover DP attention")
    if get_parallel().pp_size != 1:
        raise NotImplementedError(f"{ENV}=1 does not cover pipeline parallelism")
    spec = model_runner.spec_algorithm
    if spec.is_speculative() and not (spec.is_eagle() and arc_dflash.enabled()):
        raise NotImplementedError(
            f"{ENV}=1 covers SPEC=none and the DFlash drafter (ARC_DFLASH_DRAFTER); got {spec}"
        )
    sparse_backend = resolve_qsa_sparse_backend(model_runner.attn_backend)
    if sparse_backend.hisparse_coordinator is not None:
        # The extend plan is keyed by ForwardBatch identity; a HiSparse forward builds it from the
        # live batch at metadata init and would rebuild (and refill) it for the replay batch.
        raise NotImplementedError(f"{ENV}=1 does not cover QSA HiSparse")
    from sglang.srt.layers.attention.qsa import index_shard

    # the indexer break would run the pair's top-k hand-off inside an eager break of the captured body
    index_shard.require_off(f"{ENV}=1 (the breakable prefill CUDA graph)")
