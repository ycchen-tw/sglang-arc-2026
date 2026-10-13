"""arc-forge h24 under tensor parallelism: the packed GDN state on attention-TP head shards.

Attention-TP rank r of tp holds value heads [r HV/tp, (r+1) HV/tp) and key heads [r H/tp, (r+1) H/tp) of every GDN
layer; its mamba pool is [layers, slots, HV/tp, NB] (Mamba2StateShape.create(tp_world_size=attn_tp_size)). On the
deployed head counts (HV 48, H 16, V = K = 128) and the L4 shards (tp 2: attention DP2 x TP2, tp 4: TP4):

  shard     each packed launch -- quantize_scatter, gather_dequant, rotate_v, the h20 verify (fold, project, ring
            write), the track commit -- run on one rank's heads is the rank's slice of the same launch on all heads,
            bit for bit (state bytes, outputs, every ring)
  accuracy  on a rank's head counts over several verify steps: the stored state within half a quantization step of
            an fp32 fold of the previous stored state, encoded as the torch quantizer encodes that fold except at
            rounding midpoints; outputs against the deployed 16-bit kernel run on the stored state (bf16 noise); the
            key, key-residual and gate ring writes bit-identical to that kernel's, the D ring writes within bf16
            noise; track rows within half a step of the reference
  guards    check_server_args admits TP and attention DP and refuses context parallelism; distributed HiSparse admits
            a packed state only with ARC_GDN_STATE_FMT_DISTRIBUTED=1; a mis-sharded tensor makes a launch raise
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla import gdn_replayssm_spec_decode as dep
from sglang.kernels.ops.attention.fla import gdn_state_fmt_kernels as pk
from sglang.srt.layers.attention.linear import gdn_state_fmt as F
from sglang.srt.mem_cache import qsa_hisparse_distributed as qd
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

HV, H, V, K = 48, 16, 128, 128  # Qwen3.8-FN GDN, one GPU
L, S = 16, 4  # --linear-replayssm-cache-len 16, a K4 verify
DEV = "cuda"
FMTS = [F.StateFmt(name, bits) for name, bits in F._FORMATS.items()]
TPS = (2, 4)


def _rows_case(hv, h, seed, rows=6, layers=2):
    """A verify batch on hv / h heads: rows 0-2 fold a pending window (flush, 1..4 tokens), row 3 has an empty ring,
    row 4 reads committed history (not flush; off the steady h20 path, still a kernel branch), row 5 is padding."""
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*shape, s=1.0):
        return torch.randn(*shape, device=DEV, generator=g) * s

    nrep, slots, t = rows + 3, rows + 6, rows * S
    h0 = rn(layers, slots, hv, V, K, s=0.05)
    h0[..., :4] *= 20  # outlier key channels
    case = dict(
        h0=h0,
        d=rn(layers, nrep, hv, L, V, s=0.1).bfloat16(), dr=rn(layers, nrep, hv, L, V, s=1e-3).bfloat16(),
        k_ring=rn(layers, nrep, h, L, K, s=0.1).bfloat16(), kr=rn(layers, nrep, h, L, K, s=1e-3).bfloat16(),
        g=-torch.rand(layers, nrep, hv, L, device=DEV, generator=g) * 0.2,
        q=rn(t, h, K).bfloat16(), k=rn(t, h, K).bfloat16(), v=rn(t, hv, V, s=0.5).bfloat16(),
        a=rn(t, hv).bfloat16(), b=rn(t, hv).bfloat16(),
        A_log=torch.log(torch.rand(hv, device=DEV, generator=g) * 15 + 1), dt_bias=rn(hv, s=0.5),
    )
    state = torch.randperm(slots, device=DEV, generator=g)[:rows].to(torch.int64)
    state[rows - 1] = -1
    replay = torch.randperm(nrep, device=DEV, generator=g)[:rows].to(torch.int64)
    wp = torch.zeros(nrep, dtype=torch.int32, device=DEV)
    flush = torch.zeros(nrep, dtype=torch.int8, device=DEV)
    base = torch.randint(0, L, (nrep,), dtype=torch.int32, device=DEV, generator=g)
    for i, n in enumerate((1, 3, 4)):
        wp[replay[i]], flush[replay[i]] = n, 1
    wp[replay[4]] = 6
    case.update(state=state, replay=replay, wp=wp, flush=flush, base=base,
                qsl=torch.arange(0, t + 1, S, dtype=torch.int32, device=DEV))
    return case


_V_DIM = dict(h0=2, d=2, dr=2, g=2, v=1, a=1, b=1, A_log=0, dt_bias=0)
_K_DIM = dict(k_ring=2, kr=2, q=1, k=1)


def _shard(case, r, tp):
    """rank r's copy of a case: its value / key heads of every head-indexed tensor (contiguous, as a rank allocates)"""
    out = dict(case)
    for name, dim in _V_DIM.items():
        n = case[name].shape[dim] // tp
        out[name] = case[name].narrow(dim, r * n, n).contiguous()
    for name, dim in _K_DIM.items():
        n = case[name].shape[dim] // tp
        out[name] = case[name].narrow(dim, r * n, n).contiguous()
    return out


def _heads(t, dim, r, tp):
    n = t.shape[dim] // tp
    return t.narrow(dim, r * n, n)


def _verify_packed(fmt, c, pool, layer, wp=None, flush=None, base=None):
    out = torch.empty(c["q"].shape[0], c["v"].shape[1], V, dtype=torch.bfloat16, device=DEV)
    pk.launch_spec_verify_packed(
        fmt=fmt, q=c["q"], k=c["k"], v=c["v"], a=c["a"], b=c["b"], A_log=c["A_log"], dt_bias=c["dt_bias"], out=out,
        state=pool[layer], d_cache=c["d"][layer], k_cache=c["k_ring"][layer], g_cache=c["g"][layer],
        rawv_cache=c["dr"][layer], rawk_cache=c["kr"][layer], query_start_loc=c["qsl"],
        ssm_state_indices=c["state"], replay_indices=c["replay"], write_pos=c["wp"] if wp is None else wp,
        cache_base=c["base"] if base is None else base, is_flush=c["flush"] if flush is None else flush,
        scale=K ** -0.5, max_cache_len=L, max_spec_len=S, null_block_id=-1, fold_bk=32)
    return out


def _verify_deployed(c, h16, layer, wp, flush, base, ring):
    """the deployed 16-bit verify (no fold) on a [slots, HV, V, K] fp16 state; nk 4 = the packed kernel's key tiles,
    so the key norms sum in the same order"""
    out = torch.empty(c["q"].shape[0], c["v"].shape[1], V, dtype=torch.bfloat16, device=DEV)
    dep._launch_gdn_spec(
        c["q"], c["k"], c["v"], c["a"], c["b"], c["A_log"], c["dt_bias"], out, h16, ring["d"][layer],
        ring["k_ring"][layer], ring["g"][layer], ring["dr"][layer], ring["kr"][layer], None, c["qsl"], c["state"],
        c["replay"], wp, base, flush, K ** -0.5, L, S, True, False, 64, 1, 2, 4, 4, -1, "tf32", fold_pending=False)
    return out


def _fold_ref(old, c, layer, hv, h):
    """fp32 torch fold of every flush row's pending window into old [slots, hv, V, K]"""
    new = old.clone()
    for i in range(c["state"].shape[0]):
        s, r = int(c["state"][i]), int(c["replay"][i])
        n, b0 = int(c["wp"][r]), int(c["base"][r])
        if s < 0 or not int(c["flush"][r]) or n == 0:
            continue
        phys = [(b0 + j) % L for j in range(n)]
        g = c["g"][layer, r][:, phys].float()
        total = g.sum(1)
        decay = torch.exp(total[:, None] - torch.cumsum(g, 1))
        upd = (c["d"][layer, r][:, phys].float() + c["dr"][layer, r][:, phys].float()) * decay[:, :, None]
        hi = upd.bfloat16().float()
        lo = (upd - hi).bfloat16().float()
        keys = c["k_ring"][layer, r][:, phys].float() + c["kr"][layer, r][:, phys].float()
        keys = keys.repeat_interleave(hv // h, 0)
        new[s] = old[s] * torch.exp(total)[:, None, None] + torch.einsum("hjk,hjv->hvk", keys, hi + lo)
    return new


def _step(fmt, packed):
    """[..., HV, V, K]: each element's quantization step (its block's scale)"""
    _, _, sc = F.planes(packed, fmt, V, K)
    return sc.repeat_interleave(F.GROUP_V, dim=-2)


class TestShardBitwise(CustomTestCase):
    """a rank's launch == its slice of the all-heads launch"""

    def test_prefill_kernels(self):
        g = torch.Generator(device=DEV).manual_seed(11)
        x = (torch.randn(5, HV, V, K, device=DEV, generator=g) * 0.05).half()
        x[1, 7] = 0  # an all-zero head: zero scales
        dst = torch.tensor([3, 0, 9, -2, 6], device=DEV)
        for fmt in FMTS:
            nb = fmt.head_bytes(V, K)
            full = torch.zeros(12, HV, nb, dtype=torch.uint8, device=DEV)
            F.quantize_scatter(x, full, dst, fmt)
            back = F.gather_dequant(full, dst, fmt, V, K)
            for tp in TPS:
                for r in range(tp):
                    part = torch.zeros(12, HV // tp, nb, dtype=torch.uint8, device=DEV)
                    F.quantize_scatter(_heads(x, 1, r, tp).contiguous(), part, dst, fmt)
                    self.assertTrue(torch.equal(part, _heads(full, 1, r, tp)), (fmt.name, tp, r))
                    self.assertTrue(torch.equal(F.gather_dequant(part, dst, fmt, V, K), _heads(back, 1, r, tp)))

    def test_rotate_v(self):
        g = torch.Generator(device=DEV).manual_seed(12)
        t = 37
        for dt in (torch.bfloat16, torch.float16):
            v = (torch.randn(t, HV, V, device=DEV, generator=g) * 2).to(dt)
            full = F.rotate_v(v)
            for tp in TPS:
                for r in range(tp):
                    # a rank's value is a split of its own fused projection: [t, q | k | v] with its heads only
                    fused = torch.zeros(t, 2 * H // tp * K + HV // tp * V, dtype=dt, device=DEV)
                    view = fused[:, 2 * H // tp * K:].view(t, HV // tp, V)
                    view.copy_(_heads(v, 1, r, tp))
                    self.assertTrue(torch.equal(F.rotate_v(view), _heads(full, 1, r, tp)), (dt, tp, r))

    def test_verify(self):
        for fmt in FMTS:
            c = _rows_case(HV, H, seed=21)
            pool = F.quantize(c["h0"], fmt)
            rings0 = {n: c[n].clone() for n in ("d", "dr", "k_ring", "kr", "g")}
            shards = {tp: [_shard(c, r, tp) for r in range(tp)] for tp in TPS}  # before the all-heads launch
            pools = {tp: [_heads(pool, 2, r, tp).contiguous() for r in range(tp)] for tp in TPS}
            pool0 = pool.clone()
            outs = [_verify_packed(fmt, c, pool, layer) for layer in range(2)]
            self.assertFalse(torch.equal(pool, pool0), "the verify folded nothing")
            for n in rings0:
                self.assertFalse(torch.equal(c[n], rings0[n]), f"the verify wrote no {n} ring entry")
            for tp in TPS:
                for r in range(tp):
                    sc, sp = shards[tp][r], pools[tp][r]
                    for layer in range(2):
                        out = _verify_packed(fmt, sc, sp, layer)
                        self.assertTrue(torch.equal(out, _heads(outs[layer], 1, r, tp)), (fmt.name, tp, r, layer))
                    self.assertTrue(torch.equal(sp, _heads(pool, 2, r, tp)), (fmt.name, tp, r, "state"))
                    for n in rings0:
                        dim = _V_DIM.get(n, _K_DIM.get(n))
                        self.assertTrue(torch.equal(sc[n], _heads(c[n], dim, r, tp)), (fmt.name, tp, r, n))

    def test_track(self):
        for fmt in FMTS:
            c = _rows_case(HV, H, seed=31, rows=8, layers=3)
            pool = F.quantize(c["h0"], fmt)
            rows = c["state"].shape[0]
            acc = torch.tensor([1, 2, 3, 4, 2, 3, 4, 1], dtype=torch.int32, device=DEV)
            wp = torch.zeros_like(c["wp"])
            wp[c["replay"]] = acc + torch.tensor([0, 1, 0, 2, 0, 0, 3, 0], dtype=torch.int32, device=DEV)
            free = [s for s in range(pool.shape[1]) if s not in set(c["state"].tolist())]
            track = torch.full((rows,), -1, dtype=torch.int64, device=DEV)
            tstep = torch.full((rows,), -1, dtype=torch.int32, device=DEV)
            for j, i in enumerate((0, 2, 3, 6)):
                track[i], tstep[i] = free[j], int(acc[i]) - 1 - (i % 2)
            kw = dict(state_batch_indices=c["state"], replay_indices=c["replay"], write_pos=wp, cache_base=c["base"],
                      accept_lens=acc, mamba_track_indices=track, mamba_steps_to_track=tstep, null_block_id=-1,
                      v_dim=V, k_dim=K)
            shards = {tp: [_heads(pool, 2, r, tp).contiguous() for r in range(tp)] for tp in TPS}
            before = pool.clone()
            pk.launch_track_packed(fmt=fmt, state=pool, d_cache=c["d"], k_cache=c["k_ring"], g_cache=c["g"],
                                   d_residual_cache=c["dr"], k_residual_cache=c["kr"], **kw)
            self.assertFalse(torch.equal(pool, before), "the track kernel wrote nothing")
            for tp in TPS:
                for r in range(tp):
                    sc, sp = _shard(c, r, tp), shards[tp][r]
                    pk.launch_track_packed(fmt=fmt, state=sp, d_cache=sc["d"], k_cache=sc["k_ring"], g_cache=sc["g"],
                                           d_residual_cache=sc["dr"], k_residual_cache=sc["kr"], **kw)
                    self.assertTrue(torch.equal(sp, _heads(pool, 2, r, tp)), (fmt.name, tp, r))


class TestShardAccuracy(CustomTestCase):
    """on a rank's head counts: the packed path against fp32 / the deployed fp16 kernel"""

    def _verify_steps(self, fmt, hv, h, steps=4):
        worst = dict(half_step=0.0, mismatch=0.0, out=0.0, d_ring=0.0)
        c = _rows_case(hv, h, seed=41 + hv)
        pool = F.quantize(c["h0"], fmt)
        gen = torch.Generator(device=DEV).manual_seed(hv)
        for step in range(steps):
            for layer in range(2):
                old = F.dequantize(pool[layer], fmt, V, K)
                want = _fold_ref(old, c, layer, hv, h)
                ring0 = {n: c[n].clone() for n in ("d", "dr", "k_ring", "kr", "g")}
                out = _verify_packed(fmt, c, pool, layer)
                got = F.dequantize(pool[layer], fmt, V, K)
                live = c["state"][c["state"] >= 0]
                bound = (0.5 + 2e-5 * fmt.qmax) * _step(fmt, pool[layer])[live] + 1e-30
                ratio = ((got - want).abs()[live] / bound).max().item()
                eh, el, _ = F.planes(F.quantize(want[live], fmt), fmt, V, K)
                ph, pl, _ = F.planes(pool[layer][live], fmt, V, K)
                mism = ((eh != ph).float().mean().item() + (el != pl).float().mean().item()) / 2
                self.assertLessEqual(ratio, 1.01, (fmt.name, hv, step, layer, "state off the fp32 fold"))
                self.assertLessEqual(mism, 1e-3, (fmt.name, hv, step, layer, "encoding"))
                # the deployed kernel without a fold on the stored state, writing the same ring positions
                ref = {n: t.clone() for n, t in ring0.items()}
                fl = c["flush"].bool()
                base_ref = torch.where(fl, (c["base"] + c["wp"]) % L, c["base"]).to(torch.int32)
                wp_ref = torch.where(fl, torch.zeros_like(c["wp"]), c["wp"])
                o_ref = _verify_deployed(c, got.half(), layer, wp_ref, torch.zeros_like(c["flush"]), base_ref, ref)
                d = (out.float() - o_ref.float()).abs().max().item() / max(1.0, o_ref.float().abs().max().item())
                self.assertLessEqual(d, 0.02, (fmt.name, hv, step, layer, "outputs"))
                for n in ("k_ring", "kr", "g"):
                    self.assertTrue(torch.equal(c[n], ref[n]), (fmt.name, hv, step, layer, n))
                dsum = c["d"].float() + c["dr"].float()
                dref = ref["d"].float() + ref["dr"].float()
                dd = (dsum - dref).abs().max().item() / max(1e-6, dref.abs().max().item())
                self.assertLessEqual(dd, 0.02, (fmt.name, hv, step, layer, "D ring"))
                for key, val in (("half_step", ratio), ("mismatch", mism), ("out", d), ("d_ring", dd)):
                    worst[key] = max(worst[key], val)
            # the commit: advance the cursors by a fresh accept (fold every commit: flush rows next step)
            acc = torch.randint(1, S + 1, (c["state"].shape[0],), dtype=torch.int32, device=DEV, generator=gen)
            dep.commit_gdn_replayssm_spec(write_pos=c["wp"], cache_base=c["base"], is_flush=c["flush"],
                                          num_accepted=acc, replay_indices=c["replay"], max_cache_len=L,
                                          max_spec_len=S, fold_every_commit=True, null_block_id=-1)
            # the next verify's inputs
            g2 = torch.Generator(device=DEV).manual_seed(1000 * hv + step)
            for n in ("q", "k", "v", "a", "b"):
                c[n] = (torch.randn(c[n].shape, device=DEV, generator=g2) * (0.5 if n == "v" else 1.0)).bfloat16()
        return worst

    def test_verify_on_rank_heads(self):
        for fmt in FMTS:
            for tp in (1,) + TPS:
                w = self._verify_steps(fmt, HV // tp, H // tp)
                print(f"verify {fmt.name} tp{tp} (HV {HV // tp}, H {H // tp}): {w}")

    def test_track_on_rank_heads(self):
        for fmt in FMTS:
            for tp in TPS:
                hv, h = HV // tp, H // tp
                c = _rows_case(hv, h, seed=51 + tp, rows=8, layers=3)
                pool = F.quantize(c["h0"], fmt)
                rows = c["state"].shape[0]
                acc = torch.tensor([1, 2, 3, 4, 2, 3, 4, 1], dtype=torch.int32, device=DEV)
                wp = torch.zeros_like(c["wp"])
                wp[c["replay"]] = acc
                free = [s for s in range(pool.shape[1]) if s not in set(c["state"].tolist())]
                track = torch.full((rows,), -1, dtype=torch.int64, device=DEV)
                tstep = torch.full((rows,), -1, dtype=torch.int32, device=DEV)
                for j, i in enumerate((0, 2, 3, 6)):
                    track[i], tstep[i] = free[j], int(acc[i]) - 1 - (i % 2)
                before = F.dequantize(pool, fmt, V, K)
                pk.launch_track_packed(
                    fmt=fmt, state=pool, d_cache=c["d"], k_cache=c["k_ring"], g_cache=c["g"],
                    d_residual_cache=c["dr"], k_residual_cache=c["kr"], state_batch_indices=c["state"],
                    replay_indices=c["replay"], write_pos=wp, cache_base=c["base"], accept_lens=acc,
                    mamba_track_indices=track, mamba_steps_to_track=tstep, null_block_id=-1, v_dim=V, k_dim=K)
                after = F.dequantize(pool, fmt, V, K)
                worst = 0.0
                for i in range(rows):
                    s, r = int(c["state"][i]), int(c["replay"][i])
                    if int(track[i]) < 0:
                        continue
                    n = int(wp[r]) - int(acc[i]) + int(tstep[i]) + 1  # the kernel's track_len
                    sub = dict(c, state=torch.tensor([s], device=DEV), replay=torch.tensor([r], device=DEV),
                               wp=torch.zeros_like(c["wp"]), flush=torch.ones_like(c["flush"]))
                    sub["wp"][r] = n
                    for layer in range(3):
                        want = _fold_ref(before[layer], sub, layer, hv, h)[s]
                        bound = (0.5 + 2e-5 * fmt.qmax) * _step(fmt, pool[layer, int(track[i])]) + 1e-30
                        worst = max(worst, ((after[layer, int(track[i])] - want).abs() / bound).max().item())
                self.assertLessEqual(worst, 1.01, (fmt.name, tp))
                untouched = [j for j in range(pool.shape[1]) if j not in set(track.tolist())]
                self.assertTrue(torch.equal(after[:, untouched], before[:, untouched]))
                print(f"track {fmt.name} tp{tp}: half-step ratio {worst:.3f}")


class TestGuards(CustomTestCase):
    @staticmethod
    def _cfg(**kw):
        base = dict(mamba_ssm_dtype="float16", tp_size=4, dp_size=2, enable_dp_attention=True, dcp_size=1,
                    attn_cp_size=1, speculative_algorithm="NEXTN", enable_hierarchical_cache=False,
                    enable_int8_mamba_checkpoint=False, enable_page_major_kv_layout=False, enable_mis=False,
                    disaggregation_mode="null")
        base.update(kw)
        return SimpleNamespace(**base)

    def _check(self, env, **kw):
        with mock.patch.dict(os.environ, env):
            F.current.cache_clear()
            try:
                F.check_server_args(self._cfg(**kw))
            finally:
                F.current.cache_clear()

    def test_server_args(self):
        env = {"ARC_GDN_STATE_FMT": "i10h", "ARC_GDN_FOLD_IN_VERIFY": "1"}
        self._check(env)  # L4: TP4, attention DP2 x TP2
        self._check(env, dp_size=1, enable_dp_attention=False)  # pure TP4
        self._check(env, tp_size=2, dp_size=2)  # attention TP1
        for kw in (dict(dcp_size=2), dict(attn_cp_size=2), dict(mamba_ssm_dtype="bfloat16"),
                   dict(speculative_algorithm=None)):
            with self.assertRaisesRegex(ValueError, "ARC_GDN_STATE_FMT=i10h requires"):
                self._check(env, **kw)
        with self.assertRaisesRegex(ValueError, "ARC_GDN_FOLD_IN_VERIFY=1"):
            self._check({"ARC_GDN_STATE_FMT": "i10h", "ARC_GDN_FOLD_IN_VERIFY": "0"})

    def test_distributed_hisparse_gate(self):
        cases = [({}, True), ({"ARC_GDN_STATE_FMT": ""}, True), ({"ARC_GDN_STATE_FMT": "i10h"}, False),
                 ({"ARC_GDN_STATE_FMT": "i12h", "ARC_GDN_STATE_FMT_DISTRIBUTED": "0"}, False),
                 ({"ARC_GDN_STATE_FMT": "i10h", "ARC_GDN_STATE_FMT_DISTRIBUTED": "1"}, True)]
        for env, ok in cases:
            clean = {k: v for k, v in os.environ.items()
                     if k not in ("ARC_GDN_STATE_FMT", "ARC_GDN_STATE_FMT_DISTRIBUTED")}
            with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
                self.assertEqual(qd.packed_gdn_state_admitted(), ok, env)
        with mock.patch.dict(os.environ, {"ARC_GDN_STATE_FMT_DISTRIBUTED": "yes"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                qd.packed_gdn_state_admitted()

    def test_mis_sharded_launch_raises(self):
        fmt = FMTS[0]
        c = _rows_case(HV // 2, H // 2, seed=61)
        pool = F.quantize(c["h0"], fmt)
        for name, bad in (("A_log", c["A_log"].repeat(2)), ("k_ring", c["k_ring"].repeat(1, 1, 2, 1, 1))):
            with self.assertRaisesRegex(ValueError, "packed GDN kernels need"):
                _verify_packed(fmt, dict(c, **{name: bad}), pool, 0)
        with self.assertRaisesRegex(ValueError, "packed GDN kernels need"):
            pk.launch_track_packed(
                fmt=fmt, state=pool, d_cache=c["d"].repeat(1, 1, 2, 1, 1), k_cache=c["k_ring"], g_cache=c["g"],
                d_residual_cache=c["dr"], k_residual_cache=c["kr"], state_batch_indices=c["state"],
                replay_indices=c["replay"], write_pos=c["wp"], cache_base=c["base"],
                accept_lens=torch.ones_like(c["state"], dtype=torch.int32), mamba_track_indices=c["state"],
                mamba_steps_to_track=torch.zeros_like(c["state"], dtype=torch.int32), null_block_id=-1, v_dim=V,
                k_dim=K)


if __name__ == "__main__":
    unittest.main()
