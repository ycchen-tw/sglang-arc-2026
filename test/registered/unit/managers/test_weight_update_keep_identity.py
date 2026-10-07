"""arc-forge keep_identity weight updates (RL weight sync), CPU only.

The scheduler side (refuse / poison / succeed, every decision agreed over the TP
group, the fence around the write, no assert on a failed flush), the model
runner side (identity kept, no rollback, every delta name mapped onto a
parameter before anything is written) and the TokenizerManager side (served
name kept, weights_source, the poisoned gate on generate).
"""

import asyncio
import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    WeightsPoisonedError,
)
from sglang.srt.model_executor.model_runner_components import (
    weight_updater as runner_wu,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _TpGroup:
    """torch.distributed.all_reduce(MIN) over this rank and scripted peers."""

    def __init__(self, peers=()):
        self.peers = list(peers)  # one value per all_reduce call: the other ranks' MIN
        self.calls = 0

    def all_reduce(self, tensor, op=None, group=None):
        if self.calls < len(self.peers):
            tensor.copy_(torch.minimum(tensor, torch.tensor([self.peers[self.calls]])))
        self.calls += 1


class SchedulerKeepIdentityTest(CustomTestCase):
    def _updater(self, load=(True, "loaded", True), idle=True, flush=True, draft=None):
        self.events = []
        self.recorded = []
        self.aborts = []

        def tp_load(recv_req):
            self.events.append("load")
            return load

        def flush_cache(**kwargs):
            self.events.append("flush")
            return flush

        def default_load(recv_req):
            raise AssertionError("keep_identity must not take the default path")

        updater = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                update_weights_from_disk_keep_identity=tp_load,
                update_weights_from_disk=default_load,
            ),
            draft_worker=draft,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=flush_cache,
            is_fully_idle=lambda **kwargs: idle,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                ),
                abort_request=self.aborts.append,
            ),
        )
        device = SimpleNamespace(synchronize=lambda: self.events.append("sync"))
        for target, value in (
            ("torch.get_device_module", lambda *a: device),
            ("torch.distributed.all_reduce", self.group.all_reduce),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return updater

    def setUp(self):
        self.group = _TpGroup()

    @staticmethod
    def _request(**fields):
        base = dict(
            model_path="/delta/v000003",
            weight_version="3",
            keep_identity=True,
            flush_cache=True,
            torch_empty_cache=False,
            recapture_cuda_graph=False,
        )
        base.update(fields)
        return UpdateWeightFromDiskReqInput(**base)

    def test_success_fences_loads_flushes_and_records(self):
        """The write sits between two device syncs; the version advances only after the flush."""
        updater = self._updater()
        out = updater.update_weights_from_disk(self._request())
        self.assertTrue(out.success)
        self.assertIsNone(out.weights_poisoned)
        self.assertEqual(self.events, ["sync", "load", "flush", "sync"])
        self.assertEqual(self.recorded, ["3"])
        self.assertEqual(self.aborts, [])

    def test_mid_flight_sync_without_flush_needs_no_idle(self):
        """pause_generation(in_place) -> update(flush_cache=false) -> continue: requests stay, no flush."""
        updater = self._updater(idle=False)
        out = updater.update_weights_from_disk(self._request(flush_cache=False))
        self.assertTrue(out.success)
        self.assertEqual(self.events, ["sync", "load", "sync"])
        self.assertEqual(self.recorded, ["3"])

    def test_flush_while_busy_is_refused_untouched(self):
        """A flush would fail with requests alive: refuse before loading anything (no assert, no poison)."""
        updater = self._updater(idle=False)
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertIn("idle", out.message)
        self.assertNotIn("load", self.events)
        self.assertIsNone(out.weights_poisoned)
        self.assertIsNone(updater.poisoned)
        self.assertEqual(self.recorded, [])

    def test_a_busy_peer_rank_refuses_too(self):
        """Idle here but not on another rank: every rank must take the same decision."""
        self.group = _TpGroup(peers=[0])
        updater = self._updater(idle=True)
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertNotIn("load", self.events)

    def test_speculative_decoding_is_refused_untouched(self):
        updater = self._updater(draft=SimpleNamespace())
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertIn("speculative", out.message)
        self.assertEqual(self.events, [])

    def test_failure_after_loading_started_poisons_and_aborts(self):
        """No rollback: the weights are marked poisoned, every request aborted, the version kept."""
        updater = self._updater(load=(False, "boom", True))
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertIn("boom", out.weights_poisoned)
        self.assertEqual(updater.poisoned, out.weights_poisoned)
        self.assertNotIn("flush", self.events)
        self.assertEqual(self.recorded, [])
        (abort,) = self.aborts
        self.assertTrue(abort.abort_all)
        self.assertIn("poisoned", abort.abort_message)

    def test_failure_before_any_write_is_not_poison(self):
        updater = self._updater(load=(False, "no such file", False))
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertIsNone(updater.poisoned)
        self.assertEqual(self.aborts, [])
        self.assertIn("nothing changed", out.message)

    def test_a_peer_rank_failing_poisons_this_rank_too(self):
        """This rank loaded fine, another failed: the ranks now differ, so all are poisoned."""
        self.group = _TpGroup(peers=[1, 0, 0])  # idle ok; a peer's load failed; a peer's weights untouched
        updater = self._updater(load=(True, "loaded", True))
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertIsNotNone(updater.poisoned)
        self.assertEqual(self.recorded, [])

    def test_a_failed_flush_after_an_idle_check_poisons(self):
        updater = self._updater(flush=False)
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertIn("flush", updater.poisoned)
        self.assertEqual(self.recorded, [])

    def test_a_later_success_clears_the_poison(self):
        updater = self._updater(load=(False, "boom", True))
        updater.update_weights_from_disk(self._request())
        self.assertIsNotNone(updater.poisoned)
        updater.tp_worker.update_weights_from_disk_keep_identity = lambda r: (
            True,
            "loaded",
            True,
        )
        out = updater.update_weights_from_disk(self._request(weight_version="4"))
        self.assertTrue(out.success)
        self.assertIsNone(updater.poisoned)
        self.assertEqual(self.recorded, ["4"])

    def test_refusal_keeps_an_existing_poison(self):
        updater = self._updater(idle=False)
        updater.poisoned = "earlier"
        out = updater.update_weights_from_disk(self._request())
        self.assertFalse(out.success)
        self.assertEqual(out.weights_poisoned, "earlier")


class SchedulerDefaultPathTest(CustomTestCase):
    def test_failed_flush_returns_failure_instead_of_asserting(self):
        """A default update while requests run must not kill the scheduler."""
        recorded = []
        syncs = []
        updater = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(update_weights_from_disk=lambda r: (True, "ok")),
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: False,
            is_fully_idle=lambda **kwargs: False,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: recorded.append(
                    new_version
                )
            ),
        )
        device = SimpleNamespace(synchronize=lambda: syncs.append(1))
        with patch("torch.get_device_module", lambda *a: device):
            out = updater.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="/m", weight_version="2")
            )
        self.assertFalse(out.success)
        self.assertIn("not flushed", out.message)
        self.assertEqual(recorded, [])
        self.assertEqual(len(syncs), 2)


def _write_delta(directory, tensors):
    os.makedirs(directory, exist_ok=True)
    save_file(tensors, os.path.join(directory, "model.safetensors"))


class _ToyModel(torch.nn.Module):
    """q/k/v stack onto one parameter through its weight_loader; o_proj takes the
    default loader; an unknown name is dropped silently, as Qwen4-Exp's loader
    drops a name it cannot place."""

    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Parameter(torch.zeros(6, 2), requires_grad=False)
        self.o_proj = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.qkv_proj.weight_loader = self._stacked

    @staticmethod
    def _stacked(param, loaded_weight, shard_id):
        param.data[shard_id : shard_id + 2].copy_(loaded_weight)

    def load_weights(self, weights):
        params = dict(self.named_parameters(remove_duplicate=False))
        for name, weight in weights:
            for shard, offset in (("q_proj", 0), ("k_proj", 2), ("v_proj", 4)):
                if name.startswith(shard):
                    param = params["qkv_proj"]
                    param.weight_loader(param, weight, offset)
                    break
            else:
                name = name.removesuffix(".weight")
                if name in params:
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )


class _Loader(DefaultModelLoader):
    """A DefaultModelLoader over one local safetensors file, without the
    process-global load config."""

    def __init__(self, fail_load=False, consume=None):
        self.sources = []
        self.loads = 0
        self.fail_load = fail_load
        self.consume = consume

    def _prepare_weights(self, model_or_path, revision, fall_back_to_pt, overrides):
        path = os.path.join(model_or_path, "model.safetensors")
        if not os.path.isfile(path):
            raise RuntimeError(f"Cannot find any model weights with `{model_or_path}`")
        return model_or_path, [path], True

    def _get_weights_iterator(self, source, *, resolved_source=None, **kwargs):
        self.sources.append((source, resolved_source))
        return iter(load_file(resolved_source.weight_files[0]).items())

    def load_weights_and_postprocess(self, model, weights, target_device):
        self.loads += 1
        if self.consume is not None:
            weights = itertools.islice(weights, self.consume)
        model.load_weights(weights)
        if self.fail_load:
            raise RuntimeError("shape mismatch")


class RunnerKeepIdentityTest(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.delta = os.path.join(self.tmp.name, "v000003")
        self.tensors = {
            "q_proj.weight": torch.full((2, 2), 1.0),
            "k_proj.weight": torch.full((2, 2), 2.0),
            "v_proj.weight": torch.full((2, 2), 3.0),
            "o_proj.weight": torch.full((2, 2), 4.0),
        }
        _write_delta(self.delta, self.tensors)
        for name, value in (
            ("get_model", lambda: SimpleNamespace(weight_cache_mode="off")),
            ("_unsupported_derived_weight_cache_error", lambda: None),
            ("get_available_gpu_memory", lambda *a, **k: 0.0),
        ):
            patcher = patch.object(runner_wu, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _updater(self, loader, draft=False):
        self.model_config = SimpleNamespace(
            model_path="/base", revision=None, dtype=torch.float32
        )
        self.loader = loader
        self.model = _ToyModel()
        self.fields_updated = []
        patcher = patch.object(runner_wu, "get_model_loader", lambda *a: loader)
        patcher.start()
        self.addCleanup(patcher.stop)
        return runner_wu.WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=self.model_config,
            custom_weight_loaders={},
            get_model=lambda: self.model,
            update_model_fields=lambda *a, **k: self.fields_updated.append(k),
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: SimpleNamespace(is_draft_worker=draft),
        )

    def _untouched(self):
        return all(not bool(p.any()) for p in self.model.parameters())

    def test_identity_is_kept_and_every_tensor_written(self):
        """The delta is read through a source override; model_path and the runner record stay."""
        updater = self._updater(_Loader())
        ok, message, touched = updater.update_weights_from_disk_keep_identity(
            self.delta, None
        )
        self.assertTrue(ok and touched, message)
        self.assertIn("4 tensors, 4 weight_loader writes", message)
        self.assertEqual(self.model_config.model_path, "/base")
        self.assertEqual(self.fields_updated, [])
        ((source, resolved),) = self.loader.sources
        self.assertEqual(source.model_or_path, self.delta)
        self.assertEqual(
            resolved.weight_files, (os.path.join(self.delta, "model.safetensors"),)
        )
        self.assertTrue(
            torch.equal(self.model.qkv_proj[4:], self.tensors["v_proj.weight"])
        )
        self.assertTrue(torch.equal(self.model.o_proj, self.tensors["o_proj.weight"]))

    def test_a_name_no_parameter_takes_is_refused_before_any_write(self):
        """The loader would drop it silently and report success; keep_identity refuses the whole delta."""
        _write_delta(self.delta, {**self.tensors, "o_prj.weight": torch.ones(2, 2)})
        updater = self._updater(_Loader())
        ok, message, touched = updater.update_weights_from_disk_keep_identity(
            self.delta, None
        )
        self.assertFalse(ok or touched)
        self.assertIn("1 of 5 tensors", message)
        self.assertIn("o_prj.weight", message)
        self.assertEqual(self.loader.loads, 0)
        self.assertTrue(self._untouched())

    def test_a_load_that_consumes_less_than_planned_is_touched(self):
        updater = self._updater(_Loader(consume=2))
        ok, message, touched = updater.update_weights_from_disk_keep_identity(
            self.delta, None
        )
        self.assertFalse(ok)
        self.assertTrue(touched)
        self.assertIn("consumed 2 tensors, the plan has 4", message)

    def test_no_rollback_after_a_failed_load(self):
        """The default path reloads from the delta it just pointed model_path at; this must not."""
        updater = self._updater(_Loader(fail_load=True))
        ok, message, touched = updater.update_weights_from_disk_keep_identity(
            self.delta, None
        )
        self.assertFalse(ok)
        self.assertTrue(touched)
        self.assertIn("shape mismatch", message)
        self.assertEqual(self.loader.loads, 1)
        self.assertEqual(self.model_config.model_path, "/base")

    def test_failures_before_any_write_are_untouched(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        updater = self._updater(_Loader())
        ok, message, touched = updater.update_weights_from_disk_keep_identity(
            empty, None
        )
        self.assertFalse(ok or touched)
        self.assertIn("Cannot find any model weights", message)
        ok, message, touched = updater.update_weights_from_disk_keep_identity(
            os.path.join(self.tmp.name, "missing"), None
        )
        self.assertFalse(ok or touched)
        self.assertIn("not a directory", message)
        self.assertEqual(self.loader.loads, 0)
        self.assertTrue(self._untouched())

    def test_draft_runner_raises(self):
        updater = self._updater(_Loader(), draft=True)
        with self.assertRaises(RuntimeError):
            updater.update_weights_from_disk_keep_identity(self.delta, None)


class _TinyQwen4Exp(Qwen4ExpForConditionalGeneration):
    """Qwen4-Exp's own load_weights over a handful of named parameters, each
    with a weight_loader that must never run during a plan."""

    def __init__(self, param_names):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            num_experts=None,
            tie_word_embeddings=False,
            text_config=SimpleNamespace(split_ngram_parts=512),
            encoder_only=False,
        )
        self.language_model_only = True
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.model = SimpleNamespace(start_layer=0, end_layer=4)  # the layer-range properties read it
        self._fake = {}
        for name in param_names:
            param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            param.weight_loader = self._never
            self._fake[name] = param

    @staticmethod
    def _never(*args, **kwargs):
        raise AssertionError("a plan wrote a parameter")

    def named_parameters(self, *args, **kwargs):
        return iter(self._fake.items())

    def named_buffers(self, *args, **kwargs):
        return iter(())


class LoadPlanTest(CustomTestCase):
    """keep_identity_load_plan against the real Qwen4-Exp loader (checkpoint
    names: model.language_model.layers.N.{self_attn,linear_attn}.*)."""

    PARAMS = [
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.linear_attn.in_proj_ba.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.3.qkv_proj.weight",
        "model.layers.3.o_proj.weight",
    ]
    DELTA = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "model.language_model.layers.3.self_attn.v_proj.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
    ]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _plan(self, names):
        path = os.path.join(self.tmp.name, "model.safetensors")
        save_file({n: torch.zeros(2, 2, dtype=torch.bfloat16) for n in names}, path)
        return runner_wu.keep_identity_load_plan(_TinyQwen4Exp(self.PARAMS), [path])

    def test_a_merged_lora_delta_maps_completely(self):
        self.assertEqual(self._plan(self.DELTA), (9, 9, []))

    def test_names_the_loader_would_drop(self):
        for bad in (
            "model.language_model.layers.3.self_attn.q_prj.weight",  # misspelled
            "model.language_model.layers.7.self_attn.q_proj.weight",  # outside the layer range
            "model.language_model.layers.2.self_attn.q_proj.weight",  # no such parameter
            "mtp.layers.0.self_attn.q_proj.weight",  # MTP: the target loader skips it
            "model.visual.blocks.0.attn.qkv.weight",  # vision tower, language-only model
        ):
            self.assertEqual(self._plan(self.DELTA + [bad]), (10, 9, [bad]), bad)

    def test_header_is_read_without_tensor_data(self):
        path = os.path.join(self.tmp.name, "model.safetensors")
        save_file({"a": torch.ones(3, 5, dtype=torch.bfloat16)}, path)
        ((name, tensor),) = runner_wu._safetensors_tensors(path)
        self.assertEqual(
            (name, tuple(tensor.shape), tensor.dtype), ("a", (3, 5), torch.bfloat16)
        )
        self.assertTrue(tensor.is_meta)


class TokenizerManagerKeepIdentityTest(CustomTestCase):
    def _tm(self):
        tm = SimpleNamespace(
            model_path="/base",
            served_model_name="qwen38fn",
            weights_source="/base",
            weights_poisoned=None,
            renamed=[],
        )
        tm._update_model_path_info = lambda path, fmt: tm.renamed.append(path)
        return tm

    def test_keep_identity_success_keeps_the_served_name(self):
        tm = self._tm()
        obj = UpdateWeightFromDiskReqInput(model_path="/delta/v3", keep_identity=True)
        out = UpdateWeightFromDiskReqOutput(success=True, message="ok")
        TokenizerManager._record_weights_after_disk_update(tm, obj, [out])
        self.assertEqual(tm.renamed, [])
        self.assertEqual(tm.weights_source, "/delta/v3")
        self.assertIsNone(tm.weights_poisoned)

    def test_default_success_still_renames(self):
        tm = self._tm()
        obj = UpdateWeightFromDiskReqInput(model_path="/other")
        out = UpdateWeightFromDiskReqOutput(success=True, message="ok")
        TokenizerManager._record_weights_after_disk_update(tm, obj, [out])
        self.assertEqual(tm.renamed, ["/other"])
        self.assertEqual(tm.weights_source, "/other")

    def test_poison_mirrors_the_schedulers_and_clears(self):
        tm = self._tm()
        obj = UpdateWeightFromDiskReqInput(model_path="/delta/v3", keep_identity=True)
        bad = UpdateWeightFromDiskReqOutput(
            success=False, message="x", weights_poisoned="boom"
        )
        TokenizerManager._record_weights_after_disk_update(tm, obj, [bad])
        self.assertEqual(tm.weights_poisoned, "boom")
        self.assertEqual(tm.weights_source, "/base")
        good = UpdateWeightFromDiskReqOutput(success=True, message="ok")
        TokenizerManager._record_weights_after_disk_update(tm, obj, [good])
        self.assertIsNone(tm.weights_poisoned)

    def test_generate_is_refused_while_poisoned(self):
        """Inside the update lock: requests that queued behind a failed update are refused (503)."""

        async def run():
            tm = SimpleNamespace(
                weights_poisoned="boom",
                tokenizer=None,
                is_pause=False,
                is_pause_cond=asyncio.Condition(),
                model_update_lock=RWLock(),
                released=[],
                auto_create_handle_loop=lambda: None,
                _set_default_priority=lambda obj: None,
                _init_req_state=lambda obj, request: None,
                request_logger=SimpleNamespace(
                    log_received_request=lambda *a: None
                ),
            )
            tm._release_req_states_on_failure = tm.released.append
            obj = GenerateReqInput(text="hi", sampling_params={"max_new_tokens": 1})
            with patch(
                "sglang.srt.managers.tokenizer_manager.get_disagg",
                lambda: SimpleNamespace(language_only=False),
            ):
                async for _ in TokenizerManager.generate_request(tm, obj):
                    pass

        with self.assertRaises(WeightsPoisonedError) as err:
            asyncio.run(run())
        self.assertEqual(err.exception.status_code, 503)
        self.assertIn("boom", str(err.exception))


if __name__ == "__main__":
    unittest.main()
