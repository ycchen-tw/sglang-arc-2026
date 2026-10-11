#!/usr/bin/env python3
"""Per-bin extend telemetry of server arms side by side (ARC_EXTEND_TELEMETRY=1 lines of TP0).

  extend_telemetry_compare.py LABEL=server.log [LABEL=server.log ...] [--skip-windows 3]

Every window line carries, per (requests x new tokens) bin, n and the mean target / refresh GPU and
CPU ms, mean new tokens and mean cached prefix. Windows after the first --skip-windows are pooled
per bin (n-weighted). Bins are coarse, so the mean new tokens and prefix are printed next to the
GPU ms: compare arms only where those agree. Also prints the pooled target GPU ms per 1k new tokens
for the k4+ bins (the production shape) and the total target GPU seconds per new-token million.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

_LINE = re.compile(r"\] ARC extend telemetry \[([0-9.]+) s\]: (.*)$")
_BIN = re.compile(
    r"\| (k\d\+?/[<>=0-9]+) n=(\d+) gpu=([0-9.]+)/([0-9.]+) cpu=([0-9.]+)/([0-9.]+) tok=(\d+) pfx=(\d+)"
)
_TP0 = re.compile(r"TP0[ \]]")


def load(path: str, skip: int):
    bins = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    windows = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if "ARC extend telemetry [" not in line or not _TP0.search(line):
                continue
            windows += 1
            if windows <= skip:
                continue
            for key, n, g, gr, c, cr, tok, pfx in _BIN.findall(line):
                n = int(n)
                b = bins[key]
                b[0] += n
                for i, v in enumerate((g, gr, c, cr, tok, pfx), start=1):
                    b[i] += n * float(v)
    out = {}
    for key, b in bins.items():
        n = b[0]
        out[key] = dict(n=n, gpu=b[1] / n, refresh=b[2] / n, cpu=b[3] / n, rcpu=b[4] / n, tok=b[5] / n, pfx=b[6] / n)
    return out, windows


def _order(key: str):
    k, t = key.split("/")
    edge = {"<512": 0, "<1024": 1, "<2048": 2, "<4096": 3, ">=4096": 4}[t]
    return (k.replace("+", "9"), edge)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--skip-windows", type=int, default=3)
    args = ap.parse_args()
    arms = []
    for spec in args.arms:
        label, _, path = spec.partition("=")
        data, windows = load(path, args.skip_windows)
        arms.append((label, data))
        print(f"{label}: {windows} windows ({windows - args.skip_windows} pooled) from {path}")
    keys = sorted({k for _, d in arms for k in d}, key=_order)
    head = f"{'bin':<12}" + "".join(f" | {label[:22]:>22} n  gpu  (tok/pfx)  refresh" for label, _ in arms)
    print(head)
    for key in keys:
        row = f"{key:<12}"
        for _, d in arms:
            b = d.get(key)
            row += " | " + (
                f"{b['n']:>5} {b['gpu']:6.1f} ({b['tok']:5.0f}/{b['pfx'] / 1e3:5.1f}k) {b['refresh']:5.1f}" if b else " " * 36
            )
        print(row)
    if len(arms) >= 2:
        base_label, base = arms[0]
        for label, d in arms[1:]:
            print(f"-- {label} minus {base_label} (target GPU ms per batch, same bin):")
            for key in keys:
                if key in base and key in d:
                    a, b = base[key], d[key]
                    print(
                        f"   {key:<12} {b['gpu'] - a['gpu']:+6.1f} ms ({100 * (b['gpu'] / a['gpu'] - 1):+5.1f}%)"
                        f"  n {a['n']}/{b['n']}  tok {a['tok']:.0f}/{b['tok']:.0f}  pfx {a['pfx'] / 1e3:.1f}k/{b['pfx'] / 1e3:.1f}k"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
