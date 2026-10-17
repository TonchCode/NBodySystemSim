#!/usr/bin/env python3
"""Per-SM interaction rate vs resident warps on the busiest SM (critic review 1, #9).

Reconstructs the round-1 claim "per-SM rate ~2.1 G/s per warp up to ~6 warps, ~14 G/s at 8,
~17 G/s at 32" from a sweep CSV, so the curve has a script behind it. Stdlib only.

    python tools/rate_curve.py results/2026-10-17_architect_sanity/sweep.csv [more.csv ...]
           [--out curve.csv]

Model inversion (M1, docs/03-design.md sec. 0.2): the kernel lasts as long as its busiest SM.
For one configuration row (kernel, n, block, q, median ms):
    P      = block / q                 bodies per block (q = 1 for tiled/tuned)
    B      = ceil(n / P)               grid blocks
    k      = ceil(B / 80)              blocks on the busiest SM
    work   = k * P * n_pad             interactions on that SM (n_pad = n rounded up to block)
    warps  = k * block / 32            resident warps on that SM
    rate   = work / time               G interactions / s for that SM
Rows are used only if all B blocks are co-resident (B <= 80 * blocks_per_sm), so the
busiest SM really runs its k blocks concurrently; multi-wave rows are skipped. Launch
overhead is not subtracted (t0 ~ 4 us; reported rows with time < 50 us are flagged).
Output: per warps value, the median and max rate over all rows with that warp count.
"""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict

SMS = 80


def load(paths):
    groups = defaultdict(list)
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["kernel"], int(r["n"]), int(r["block"]), int(r["q"]), int(r["unroll"]),
                       r.get("prefetch", "1"), int(r["blocks_per_sm"]))
                groups[key].append(float(r["ms"]))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--min-us", type=float, default=0.0, help="drop rows faster than this (launch-overhead bound)")
    args = ap.parse_args()
    pts = defaultdict(list)
    used = skipped = 0
    for (kern, n, block, q, unroll, pf, bps), ms in load(args.csv).items():
        t = statistics.median(ms) * 1e-3
        if t * 1e6 < args.min_us:
            skipped += 1
            continue
        P = block // q
        B = math.ceil(n / P)
        if B > SMS * bps:
            skipped += 1
            continue
        k = math.ceil(B / SMS)
        n_pad = math.ceil(n / block) * block
        work = k * P * n_pad
        warps = k * block / 32
        pts[warps].append((work / t / 1e9, f"{kern} n={n} b={block} q={q} u={unroll} pf={pf}"))
        used += 1
    print(f"# {used} configurations used, {skipped} skipped (multi-wave or below --min-us)")
    print(f"{'warps/SM':>9} {'n':>4} {'median G/s/SM':>14} {'max G/s/SM':>11} {'G/s per warp (median)':>22}  best config")
    rows = []
    for w in sorted(pts):
        rates = [r for r, _ in pts[w]]
        best = max(pts[w])
        med = statistics.median(rates)
        rows.append({"warps": w, "count": len(rates), "median": med, "max": best[0], "per_warp": med / w,
                     "best": best[1]})
        print(f"{w:9.1f} {len(rates):4d} {med:14.2f} {best[0]:11.2f} {med / w:22.2f}  {best[1]}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
