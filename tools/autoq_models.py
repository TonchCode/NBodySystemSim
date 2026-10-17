#!/usr/bin/env python3
"""Auto-q models for allpairs_multi and the loss computation behind D-008 (critic review 1, #5).

    python tools/autoq_models.py --in-sample results/2026-10-17_architect_sanity/sweep.csv \
                                 --oos results/2026-10-17_architect_sanity/sweep_oos.csv

Models (each returns q for (n, block)):
  v1    the pre-registered busiest-SM cost model, a line-by-line port of
        autoThreadsPerBody() in src/core/kernels_multi.cu (kTileCost = 2, 8-warp knee,
        2% tie band toward smaller q). This is what the bench runs for --q 0.
  v2    REJECTED model, kept for the record. Same busiest-SM work term, but the per-SM rate
        comes from the measured rate curve r(warps) (tools/rate_curve.py on the IN-SAMPLE
        data) instead of the 8-warp knee:  cost = k * P * (1 + 2q/block) / r(k * block / 32).
        PROVENANCE: the round-1 v2 code was never saved. This is a reconstruction in round 2
        from the D-008 / 03-design sec. 6 description ("fitted to the measured r(w)"). Its
        losses may not equal the round-1 figures; the script prints what this version gives.
  sat   the thread-saturation rule: smallest q with n * q >= 80 * 1536.

Loss for one (n, block) cell = 1 - rate(q_model) / max_q rate(q), using the median measured
rate per (n, block, q). The round-1 CSVs have only the RESOLVED q (auto and explicit rows
are merged, critic #9); the median is over both. New CSVs have q_requested and the script
then uses only explicit-q rows.
"""
import argparse
import bisect
import csv
import math
import statistics
import sys
from collections import defaultdict

SMS, MAX_THREADS = 80, 1536
QS = [1, 2, 4, 8, 16, 32]


def v1(n, block, sms=SMS):
    k_tile = 2.0
    best_q, best_cost = 1, 1e300
    q = 1
    while q <= 32 and q <= block:
        P = block // q
        B = math.ceil(n / P)
        per_sm = math.ceil(B / sms)
        warps = per_sm * block / 32.0
        lat = 1.0 if warps >= 8.0 else 8.0 / warps
        cost = per_sm * P * (1.0 + k_tile * q / block) * lat
        if cost < best_cost * 0.98:
            best_cost, best_q = cost, q
        q *= 2
    return best_q


def make_v2(curve):
    """curve: sorted list of (warps, rate). Piecewise-linear r(w), clamped, linear to 0 below."""
    ws = [w for w, _ in curve]
    rs = [r for _, r in curve]

    def r_of(w):
        if w <= ws[0]:
            return rs[0] * w / ws[0]
        if w >= ws[-1]:
            return rs[-1]
        i = bisect.bisect_left(ws, w)
        w0, w1, r0, r1 = ws[i - 1], ws[i], rs[i - 1], rs[i]
        return r0 + (r1 - r0) * (w - w0) / (w1 - w0)

    def v2(n, block, sms=SMS):
        best_q, best_cost = 1, 1e300
        q = 1
        while q <= 32 and q <= block:
            P = block // q
            B = math.ceil(n / P)
            per_sm = math.ceil(B / sms)
            warps = min(per_sm * block / 32.0, 48.0)
            cost = per_sm * P * (1.0 + 2.0 * q / block) / r_of(warps)
            if cost < best_cost * 0.98:
                best_cost, best_q = cost, q
            q *= 2
        return best_q

    return v2


def sat(n, block, sms=SMS):
    for q in QS:
        if q <= block and n * q >= sms * MAX_THREADS:
            return q
    return min(32, block)


def load_cells(path):
    """(n, block) -> {q: median G/s}, multi rows only."""
    per = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["kernel"] != "allpairs_multi":
                continue
            if "q_requested" in r and r["q_requested"] == "0":
                continue   # new CSVs: explicit-q rows only
            if r.get("prefetch", "1") != "1":
                continue   # the auto-q models were defined for the prefetch kernel
            per[(int(r["n"]), int(r["block"]), int(r["q"]))].append(float(r["ginteractions_per_s"]))
    cells = defaultdict(dict)
    for (n, b, q), v in per.items():
        cells[(n, b)][q] = statistics.median(v)
    return cells


def curve_from(cells_csv):
    """Median per-SM rate vs warps from the same CSV (tools/rate_curve.py logic, inline)."""
    groups = defaultdict(list)
    with open(cells_csv, newline="") as f:
        for r in csv.DictReader(f):
            groups[(r["kernel"], int(r["n"]), int(r["block"]), int(r["q"]), int(r["unroll"]),
                    int(r["blocks_per_sm"]))].append(float(r["ms"]))
    pts = defaultdict(list)
    for (kern, n, block, q, unroll, bps), ms in groups.items():
        t = statistics.median(ms) * 1e-3
        if t < 15e-6:
            continue
        P = block // q
        B = math.ceil(n / P)
        if B > SMS * bps:
            continue
        k = math.ceil(B / SMS)
        work = k * P * math.ceil(n / block) * block
        pts[k * block / 32].append(work / t / 1e9)
    return sorted((w, statistics.median(v)) for w, v in pts.items())


def losses(cells, model):
    out = []
    for (n, b), rates in sorted(cells.items()):
        q = model(n, b)
        best_q = max(rates, key=rates.get)
        if q not in rates:
            out.append((n, b, q, best_q, float("nan")))
            continue
        out.append((n, b, q, best_q, 1.0 - rates[q] / rates[best_q]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-sample", required=True)
    ap.add_argument("--oos")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    curve = curve_from(args.in_sample)
    print("v2 rate curve r(warps) from in-sample data: " + ", ".join(f"{w:g}:{r:.2f}" for w, r in curve))
    models = {"v1": v1, "v2(reconstructed)": make_v2(curve), "sat": sat}
    sets = [("in-sample", load_cells(args.in_sample))]
    if args.oos:
        sets.append(("out-of-sample", load_cells(args.oos)))
    print(f"{'model':<18} {'set':<14} {'cells':>5} {'mean loss':>9} {'worst':>7}  worst cell")
    for name, m in models.items():
        for sname, cells in sets:
            ls = losses(cells, m)
            vals = [x[4] for x in ls if not math.isnan(x[4])]
            worst = max(ls, key=lambda x: -1 if math.isnan(x[4]) else x[4])
            print(f"{name:<18} {sname:<14} {len(vals):5d} {100 * statistics.mean(vals):8.1f}% "
                  f"{100 * worst[4]:6.1f}%  n={worst[0]} b={worst[1]} q={worst[2]} (best q={worst[3]})")
            if args.detail:
                for n, b, q, bq, l in ls:
                    print(f"      n={n:6d} b={b:4d}  q={q:2d}  best={bq:2d}  loss {100 * l:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
