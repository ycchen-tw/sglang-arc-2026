"""ARC_DFLASH_FAST_TOPK: fast_topk.topk against torch.topk, then the time at the served shape.

  python scripts/arc/test_dflash_fast_topk.py          # prints FAST_TOPK_OK

Per case: the values are torch.topk's values exactly (same multiset, descending); every returned id holds its
value; ids are unique; among equal values the ids ascend (the documented tie order); and the id set equals
torch's wherever the k-th value is not tied with the (k+1)-th (otherwise either choice is a valid top-k).
"""

import torch

from sglang.srt.speculative.arc_dflash.fast_topk import topk

K = 16


def check(logits, k, name):
    values, ids = topk(logits, k)
    ref_v, ref_i = logits.topk(k, -1)
    if values.dtype != logits.dtype or ids.dtype != torch.int64 or values.shape != ref_v.shape:
        raise RuntimeError(f"{name}: {values.dtype} {ids.dtype} {tuple(values.shape)}")
    if not torch.equal(values, ref_v):
        bad = (values != ref_v).any(-1).nonzero().flatten()[:3].tolist()
        raise RuntimeError(f"{name}: values differ from torch.topk at rows {bad}")
    if not torch.equal(logits.gather(-1, ids), values):
        raise RuntimeError(f"{name}: an id does not hold its value")
    if (ids.sort(-1).values.diff(dim=-1) == 0).any():
        raise RuntimeError(f"{name}: repeated id")
    tie = values[:, 1:] == values[:, :-1]
    if (tie & (ids[:, 1:] <= ids[:, :-1])).any():
        raise RuntimeError(f"{name}: ties not in ascending id order")
    if k < logits.shape[-1]:
        kth = values[:, -1]
        next_v = logits.float().masked_fill(
            torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, ids, True), -float("inf")).max(-1).values
        clean = next_v < kth.float()
        same = ids.sort(-1).values == ref_i.sort(-1).values
        if not same[clean].all():
            raise RuntimeError(f"{name}: id set differs from torch's on an untied row")
        tied_rows = int((~clean).sum())
    else:
        tied_rows = 0
    order_diff = int((ids != ref_i).any(-1).sum())
    print(f"{name}: ok ({order_diff} rows order ties differently from torch, {tied_rows} rows tied at the k-th)")


def main():
    torch.cuda.set_device(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(384, 2560, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(124160, 2560, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    served = x @ w.t()  # the drafter head's logits on one TP2 rank: [batch 128 x 3, 124,160] bf16
    check(served, K, "served shape bf16")
    check(served[:7].contiguous(), K, "7 rows")
    check(served[:1, :4097].contiguous(), K, "1 row, 4,097 columns (a 1-column chunk)")
    coarse = (torch.randn(64, 100000, device="cuda", generator=g) * 2).round().to(torch.bfloat16)  # mass ties
    check(coarse, K, "heavy ties bf16")
    special = torch.randn(32, 50000, device="cuda", generator=g)
    special[:, ::7] = -float("inf")
    special[0, :] = -float("inf")
    special[0, 123] = 5.0
    special[1, :20] = -0.0
    special[1, 20:] = -1.0
    check(special.to(torch.bfloat16), K, "bf16 with -inf / -0.0 rows")
    check(torch.randn(1024, 248320, device="cuda", generator=g).to(torch.bfloat16), K, "full vocab 1024 rows")
    check(served.half(), 8, "fp16 k=8")

    def t(fn, reps=30):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / reps * 1000

    for rows in (192, 288, 384):
        logits = served[:rows].contiguous()
        print(f"rows {rows}: torch.topk {t(lambda: logits.topk(K, -1)):.1f} us, fast {t(lambda: topk(logits, K)):.1f} us")
    print("FAST_TOPK_OK")


if __name__ == "__main__":
    with torch.no_grad():
        main()
