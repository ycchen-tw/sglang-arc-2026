import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLELayer,
)
from sglang.srt.utils import set_weight_attrs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _make_source_embedding(
    *,
    dtype=torch.bfloat16,
    embedding_dim=7,
    vocab_start=0,
    vocab_end=8,
    org_vocab_size=8,
    tp_size=1,
    num_added_embeddings=0,
):
    local_rows = vocab_end - vocab_start
    weight = nn.Parameter(
        torch.empty((local_rows, embedding_dim), dtype=dtype, device="cuda"),
        requires_grad=False,
    )
    set_weight_attrs(
        weight,
        {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": lambda *_args, **_kwargs: None,
        },
    )
    shard_indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=vocab_start,
        padded_org_vocab_end_index=vocab_end,
        padded_added_vocab_start_index=org_vocab_size,
        padded_added_vocab_end_index=org_vocab_size,
        org_vocab_start_index=vocab_start,
        org_vocab_end_index=vocab_end,
        added_vocab_start_index=org_vocab_size,
        added_vocab_end_index=org_vocab_size,
    )
    return SimpleNamespace(
        weight=weight,
        quant_config=None,
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=tp_size,
        num_embeddings=org_vocab_size + num_added_embeddings,
        org_vocab_size=org_vocab_size,
        padding_size=1,
        num_added_embeddings=num_added_embeddings,
        use_presharded_weights=False,
        org_vocab_size_padded=org_vocab_size,
        num_embeddings_padded=org_vocab_size + num_added_embeddings,
        shard_indices=shard_indices,
        embedding_dim=embedding_dim,
        weight_scale=None,
        quant_method=UnquantizedEmbeddingMethod(),
        num_embeddings_per_partition=local_rows,
        num_org_embeddings_per_partition=local_rows,
        num_added_embeddings_per_partition=0,
    )


def _load_rows(offloaded, rows, *, pinned=True):
    pointer = offloaded.weight.data_ptr()
    offloaded.weight_loader(offloaded.weight, rows)
    assert offloaded.weight.data_ptr() == pointer
    assert offloaded.weight.is_pinned() == pinned
    assert offloaded.weight.weight_loader.__self__ is offloaded
    assert offloaded.quant_method is None


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("embedding_dim", [7, 64, 257])
def test_qwen4_ple_pinned_gather_tp1(input_dtype, embedding_dim):
    source = _make_source_embedding(embedding_dim=embedding_dim)
    offloaded = Qwen4ExpPinnedHostEmbedding(source)
    rows = torch.arange(8 * embedding_dim, dtype=torch.bfloat16, device="cuda").reshape(
        8, embedding_dim
    )
    _load_rows(offloaded, rows)

    ids = torch.tensor([[0, 7, 3], [4, 1, 6]], dtype=input_dtype, device="cuda")
    expected = rows.index_select(0, ids.long().flatten()).reshape(
        *ids.shape, embedding_dim
    )
    actual = offloaded(ids)

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_ple_pinned_gather_shard_boundaries_and_out_buffer():
    embedding_dim = 13
    source = _make_source_embedding(
        embedding_dim=embedding_dim,
        vocab_start=4,
        vocab_end=8,
        org_vocab_size=8,
        tp_size=2,
    )
    offloaded = Qwen4ExpPinnedHostEmbedding(source)
    rows = torch.arange(8 * embedding_dim, dtype=torch.bfloat16, device="cuda").reshape(
        8, embedding_dim
    )
    _load_rows(offloaded, rows)

    ids = torch.tensor([[-1, 3, 4], [7, 8, 100]], device="cuda")
    output = torch.full(
        (*ids.shape, embedding_dim),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual = offloaded.gather(ids, out=output)
    expected = torch.zeros_like(output)
    expected[0, 2] = rows[4]
    expected[1, 0] = rows[7]

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_ple_pinned_gather_empty_input():
    offloaded = Qwen4ExpPinnedHostEmbedding(_make_source_embedding())
    _load_rows(offloaded, torch.zeros((8, 7), dtype=torch.bfloat16, device="cuda"))
    ids = torch.empty((0, 3), dtype=torch.int64, device="cuda")
    actual = offloaded.gather(ids)
    assert actual.shape == (0, 3, 7)
    assert actual.numel() == 0


def test_qwen4_ple_pinned_embedding_rejects_unsupported_weights():
    with pytest.raises(TypeError, match="requires bfloat16"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(num_added_embeddings=1))


def test_qwen4_ple_prefetch_buffer_lifecycle(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer.ple_embedding = SimpleNamespace(
        ngram_embedding=Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=layer.ple_embed_dim)
        )
    )
    layer._graph_prefetch_buffers = {}
    layer._eager_prefetch_buffer = None
    lookup_ids = torch.empty((0,), dtype=torch.int64, device="cuda")

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: False)
    eager_large = layer._get_prefetch_buffer(8, lookup_ids)
    eager_small = layer._get_prefetch_buffer(3, lookup_ids)
    assert eager_small.data_ptr() == eager_large.data_ptr()
    assert layer._eager_prefetch_buffer.shape == (8, layer.ple_embed_dim)

    eager_grown = layer._get_prefetch_buffer(12, lookup_ids)
    eager_grown_small = layer._get_prefetch_buffer(4, lookup_ids)
    assert eager_grown_small.data_ptr() == eager_grown.data_ptr()
    assert layer._eager_prefetch_buffer.shape == (12, layer.ple_embed_dim)

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: True)
    graph_three = layer._get_prefetch_buffer(3, lookup_ids)
    graph_five = layer._get_prefetch_buffer(5, lookup_ids)
    graph_three_reused = layer._get_prefetch_buffer(3, lookup_ids)
    assert graph_three_reused.data_ptr() == graph_three.data_ptr()
    assert graph_five.data_ptr() != graph_three.data_ptr()
    assert set(layer._graph_prefetch_buffers) == {3, 5}


def _file_backend_supported() -> bool:
    from sglang.srt.models.qwen4_exp_ple_table import device_uses_host_page_tables

    return (
        torch.cuda.is_available()
        and device_uses_host_page_tables(torch.cuda.current_device()) is True
    )


@pytest.mark.skipif(
    not _file_backend_supported(),
    reason="the file backend needs pageable host memory reachable through host page tables",
)
@pytest.mark.parametrize("embedding_dim", [7, 160])
def test_qwen4_ple_file_backend_matches_pinned(embedding_dim):
    with tempfile.TemporaryDirectory() as table_dir:
        pinned = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=embedding_dim)
        )
        filed = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=embedding_dim),
            backend="file",
            table_dir=table_dir,
        )
        assert pinned._file_prefetcher is None and filed._file_prefetcher is not None
        (name,) = os.listdir(table_dir)
        assert "rows0-8" in name  # this rank's vocabulary shard
        rows = torch.arange(
            8 * embedding_dim, dtype=torch.bfloat16, device="cuda"
        ).reshape(8, embedding_dim)
        _load_rows(pinned, rows)
        _load_rows(filed, rows, pinned=False)
        ids = torch.tensor([[0, 7, 3], [4, 1, 6]], dtype=torch.int64, device="cuda")
        torch.testing.assert_close(filed(ids), pinned(ids), rtol=0, atol=0)
        # A prefill-sized gather goes through the page-cache hint path.
        big = torch.randint(0, 8, (4096,), device="cuda")
        torch.testing.assert_close(
            filed(big), rows.index_select(0, big), rtol=0, atol=0
        )


@pytest.mark.skipif(
    not _file_backend_supported(),
    reason="the file backend needs pageable host memory reachable through host page tables",
)
def test_qwen4_ple_file_backend_fp8_table():
    embedding_dim = 160
    with tempfile.TemporaryDirectory() as table_dir:
        filed = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(
                embedding_dim=embedding_dim, dtype=torch.float8_e4m3fn
            ),
            backend="file",
            table_dir=table_dir,
        )
        assert filed.weight.dtype == torch.float8_e4m3fn
        rows = (
            torch.arange(8 * embedding_dim, dtype=torch.float32, device="cuda").reshape(
                8, embedding_dim
            )
            / 64
        ).to(torch.float8_e4m3fn)
        _load_rows(filed, rows, pinned=False)
        ids = torch.tensor([[0, 7, 3]], dtype=torch.int64, device="cuda")
        expected = (
            rows.index_select(0, ids.flatten())
            .to(torch.bfloat16)
            .reshape(1, 3, embedding_dim)
        )
        torch.testing.assert_close(filed(ids), expected, rtol=0, atol=0)


def _fp8_rows(n, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = torch.randn((n, d), generator=g, device="cuda") * 64
    rows[1, :16] = 0  # an all-zero block
    rows[2, 16:32] = 0.05  # a block whose scale is subnormal in e4m3
    return rows.clamp(-448, 448).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("vocab_start", [0, 4])
def test_qwen4_ple_nvfp4_gather_matches_reference(monkeypatch, vocab_start):
    from sglang.srt.models import arc_ple_nvfp4

    monkeypatch.setenv("ARC_PLE_NVFP4", "1")
    monkeypatch.setattr(arc_ple_nvfp4, "CHUNK_ROWS", 3)  # several chunks per load
    embedding_dim = 160
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(
            dtype=torch.float8_e4m3fn,
            embedding_dim=embedding_dim,
            vocab_start=vocab_start,
            vocab_end=8,
            org_vocab_size=8,
            tp_size=1 if vocab_start == 0 else 2,
        )
    )
    assert offloaded.weight.dtype == torch.uint8
    assert offloaded.weight.shape == (8 - vocab_start, 90)  # 80 code + 10 scale bytes
    assert offloaded.weight.is_pinned()
    rows = _fp8_rows(8, embedding_dim)
    # two checkpoint shards, as the Qwen4-Exp loader hands them over
    offloaded.load_rows(rows[:5], 0)
    offloaded.load_rows(rows[5:], 5)
    offloaded.finish_packed_load()
    assert offloaded.packed_subnormal_blocks == (1 if vocab_start <= 2 else 0)

    packed, scales, _ = arc_ple_nvfp4.quantize_rows(rows)
    reference = arc_ple_nvfp4.dequantize_rows(packed, scales).to(torch.bfloat16)
    ids = torch.tensor([[0, 7, 3], [4, 1, 2], [-1, 8, 6]], device="cuda")
    expected = torch.zeros((*ids.shape, embedding_dim), dtype=torch.bfloat16, device="cuda")
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            k = int(ids[i, j])
            if vocab_start <= k < 8:
                expected[i, j] = reference[k]
    out = torch.full_like(expected, torch.nan)  # tp_size 2 would allocate symmetric memory
    torch.testing.assert_close(offloaded.gather(ids, out=out), expected, rtol=0, atol=0)
    assert torch.all(reference[1, :16] == 0)
    stored = offloaded.weight.cuda()
    torch.testing.assert_close(
        stored, arc_ple_nvfp4.pack_rows(packed, scales)[vocab_start:], rtol=0, atol=0
    )

    # 4-bit rounding error only: ~9.5 % relative RMS on gaussian rows
    f = rows.float()
    rel = (arc_ple_nvfp4.dequantize_rows(packed, scales) - f).norm() / f.norm()
    assert 0.05 < float(rel) < 0.11


def test_qwen4_ple_nvfp4_dequantize_codes():
    from sglang.srt.models import arc_ple_nvfp4

    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    row = torch.cat([grid, -grid]).repeat(10).view(1, 160)
    packed, scales, _ = arc_ple_nvfp4.quantize_rows(row)
    assert torch.all(scales.float() == 1.0)
    torch.testing.assert_close(
        arc_ple_nvfp4.dequantize_rows(packed, scales), row, rtol=0, atol=0
    )
    assert int(packed[0, 0]) == 0x10  # element 0 low nibble, element 1 high


def test_qwen4_ple_nvfp4_refusals(monkeypatch):
    monkeypatch.setenv("ARC_PLE_NVFP4", "1")
    with pytest.raises(TypeError, match="needs an fp8 PLE table"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(embedding_dim=160))
    with pytest.raises(ValueError, match="divisible by 16"):
        Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=24)
        )
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=32)
    )
    with pytest.raises(TypeError, match="fp8 PLE rows only"):
        offloaded.load_rows(torch.zeros((8, 32), dtype=torch.bfloat16), 0)
    offloaded.load_rows(_fp8_rows(8, 32)[:6], 0)
    with pytest.raises(RuntimeError, match="quantized 6 PLE rows"):
        offloaded.finish_packed_load()
    monkeypatch.setenv("ARC_PLE_NVFP4", "yes")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=32)
        )


def _int5_edge_rows(n, d, seed=0):
    """fp8 rows with the edge cases: all zero, all +-448, tiny (subnormal / flushed scale)."""
    rows = _fp8_rows(n, d, seed).float()
    rows[3] = 448.0 * torch.where(torch.arange(d, device="cuda") % 3 == 0, -1.0, 1.0)
    rows[4] = 0.0
    rows[5, :16] = 448.0
    rows[5, 16:32] = 0.01
    rows[6, 48:64] = 2.0**-9  # the smallest e4m3 subnormal: amax / 15 rounds to a zero scale
    return rows.to(torch.float8_e4m3fn)


@pytest.mark.parametrize("vocab_start", [0, 4])
def test_qwen4_ple_int5_gather_matches_reference(monkeypatch, vocab_start):
    from sglang.srt.models import arc_ple_int5

    monkeypatch.setenv("ARC_PLE_INT5", "1")
    monkeypatch.setattr(arc_ple_int5, "CHUNK_ROWS", 3)  # several chunks per load
    embedding_dim = 160
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(
            dtype=torch.float8_e4m3fn,
            embedding_dim=embedding_dim,
            vocab_start=vocab_start,
            vocab_end=8,
            org_vocab_size=8,
            tp_size=1 if vocab_start == 0 else 2,
        )
    )
    assert offloaded.weight.dtype == torch.uint8
    assert offloaded.weight.shape == (8 - vocab_start, 110)  # 100 code + 10 scale bytes
    assert offloaded.weight.is_pinned()
    rows = _int5_edge_rows(8, embedding_dim)
    # two checkpoint shards, as the Qwen4-Exp loader hands them over
    offloaded.load_rows(rows[:5], 0)
    offloaded.load_rows(rows[5:], 5)
    offloaded.finish_packed_load()
    # blocks whose amax / 15 < 2^-6: row 2 (0.05), row 5 (0.01), row 6 (2^-9, scale flushed to zero)
    assert offloaded.packed_subnormal_blocks == sum(
        1 for r in (2, 5, 6) if r >= vocab_start
    )

    packed, scales, _ = arc_ple_int5.quantize_rows(rows)
    reference = arc_ple_int5.dequantize_rows(packed, scales).to(torch.bfloat16)
    ids = torch.tensor([[0, 7, 3], [4, 1, 2], [-1, 8, 6], [5, 4, 3]], device="cuda")
    expected = torch.zeros((*ids.shape, embedding_dim), dtype=torch.bfloat16, device="cuda")
    for i in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            k = int(ids[i, j])
            if vocab_start <= k < 8:
                expected[i, j] = reference[k]
    out = torch.full_like(expected, torch.nan)  # tp_size 2 would allocate symmetric memory
    torch.testing.assert_close(offloaded.gather(ids, out=out), expected, rtol=0, atol=0)
    assert torch.all(reference[4] == 0)  # all-zero row
    assert torch.all(reference[1, :16] == 0)  # all-zero block
    assert torch.all(reference[6, 48:64] == 0)  # scale rounded to zero
    stored = offloaded.weight.cuda()
    torch.testing.assert_close(
        stored, arc_ple_int5.pack_rows(packed, scales)[vocab_start:], rtol=0, atol=0
    )

    # 5-bit rounding error only: ~4 % relative RMS on the gaussian rows (0 and 7)
    f = rows[[0, 7]].float()
    rel = (arc_ple_int5.dequantize_rows(packed, scales)[[0, 7]] - f).norm() / f.norm()
    assert 0.03 < float(rel) < 0.055


@pytest.mark.parametrize("embedding_dim", [16, 48, 144, 176])
def test_qwen4_ple_int5_gather_other_widths(monkeypatch, embedding_dim):
    from sglang.srt.models import arc_ple_int5

    monkeypatch.setenv("ARC_PLE_INT5", "1")
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=embedding_dim)
    )
    assert offloaded.weight.shape == (8, embedding_dim * 5 // 8 + embedding_dim // 16)
    rows = _fp8_rows(8, embedding_dim, seed=embedding_dim)
    offloaded.load_rows(rows, 0)
    offloaded.finish_packed_load()
    packed, scales, _ = arc_ple_int5.quantize_rows(rows)
    reference = arc_ple_int5.dequantize_rows(packed, scales).to(torch.bfloat16)
    ids = torch.arange(8, device="cuda").flip(0).view(2, 4)
    torch.testing.assert_close(
        offloaded.gather(ids), reference[ids.flatten()].view(2, 4, embedding_dim), rtol=0, atol=0
    )


def test_qwen4_ple_int5_refusals(monkeypatch):
    monkeypatch.setenv("ARC_PLE_INT5", "1")
    with pytest.raises(TypeError, match="needs an fp8 PLE table"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(embedding_dim=160))
    for bad_width in (24, 8, 170):
        with pytest.raises(ValueError, match="divisible by 16"):
            Qwen4ExpPinnedHostEmbedding(
                _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=bad_width)
            )
    with pytest.raises(ValueError, match="pinned only"):
        Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=32),
            backend="file",
            table_dir="/nonexistent-never-touched",
        )
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=32)
    )
    with pytest.raises(TypeError, match="fp8 PLE rows only"):
        offloaded.load_rows(torch.zeros((8, 32), dtype=torch.bfloat16), 0)
    offloaded.load_rows(_fp8_rows(8, 32)[:6], 0)
    with pytest.raises(RuntimeError, match="quantized 6 PLE rows"):
        offloaded.finish_packed_load()
    monkeypatch.setenv("ARC_PLE_INT5", "yes")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=32)
        )
    # the two packed formats are exclusive
    monkeypatch.setenv("ARC_PLE_INT5", "1")
    monkeypatch.setenv("ARC_PLE_NVFP4", "1")
    with pytest.raises(ValueError, match="both set"):
        Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(dtype=torch.float8_e4m3fn, embedding_dim=32)
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
