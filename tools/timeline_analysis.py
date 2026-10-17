#!/usr/bin/env python3
"""Per-SM load-imbalance metrics from nbody_bench --timeline CSVs (no ncu needed).

Stdlib only (matplotlib/numpy are not installed on this machine; the Gantt plot is written
as SVG directly).

    python tools/timeline_analysis.py results/<dir>/tl_*.csv [--svg-dir results/<dir>/plots]
                                      [--summary results/<dir>/timeline_metrics.csv]

Input: one row per block: launch,block,smid,t_start_ns,t_end_ns,cyc_start,cyc_end, with a
'# key=value ...' metadata line first (kernel, n, sms, ...) and '#cal,A|B,smid,tA,cA,tB,cB'
clock-calibration lines.

Clock calibration: %globaltimer ticks in 1.024 us steps on this board, too coarse for
10-30 us kernels. The bench anchors every SM's clock64 to a globaltimer tick edge before (A)
and after (B) the timeline launches. Block times are then
    t(c) = tA_A + (c - cA_A) / f_sm,   f_sm = (cA_B - cA_A) / (tA_B - tA_A)  [cycles/ns]
i.e. per-SM cycle counts placed on the global ns axis with sub-us alignment. If an SM has no
calibration record, the raw (quantised) globaltimer stamps are used and the file is flagged.
The calibrated block starts must agree with the raw stamps to within one tick (checked).

Per launch it computes (times relative to the first block start of that launch):
  span          last block end - first block start (the kernel as the blocks see it)
  busy[sm]      measure of the UNION of the block intervals on that SM (co-resident blocks
                overlap, so this is wall time the SM had >= 1 block, not the sum)
  max/mean      max busy / mean busy over ALL SMs (idle SMs count as 0): the imbalance
                factor; 1.0 = perfect. Also reported over active SMs only.
  sm_eff        mean busy / span (the ncu "SM active" analogue, averaged over SMs)
  idle_sms      SMs that got no block
  tail          last block end - median of the per-SM last-block-end (active SMs only);
                also as a fraction of span
  blocks/SM     min / max blocks per SM
  t_res         GCD of the differences between distinct raw %globaltimer values (the
                timer's tick; 1024 ns on this board)
  sm_clock      median over blocks of (cyc_end - cyc_start) / (t_end - t_start) in GHz
Medians over launches are printed per file.
"""
import argparse
import csv
import functools
import math
import os
import statistics
import sys


def read_timeline(path):
    meta, rows, cal = {}, [], {"A": {}, "B": {}}
    with open(path, newline="") as f:
        lines = f.readlines()
    body = []
    for ln in lines:
        if ln.startswith("#cal,"):
            _, ph, sm, tA, cA, tB, cB = ln.strip().split(",")
            cal[ph].setdefault(int(sm), (int(tA), int(cA), int(tB), int(cB)))
            continue
        if ln.startswith("#"):
            for tok in ln[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        else:
            body.append(ln)
    for r in csv.DictReader(body):
        rows.append({k: int(v) for k, v in r.items()})
    return meta, rows, cal


def calibrate(rows, cal, tick_ns=1024):
    """Replace t_start_ns/t_end_ns by clock64-derived times where both anchors exist.
    Returns (rows, info)."""
    info = {"calibrated_sms": 0, "uncalibrated_sms": 0, "f_ghz_min": float("nan"), "f_ghz_max": float("nan"),
            "max_dev_vs_raw_ns": 0.0}
    fs, out, bad_sms = [], [], set()
    anchors = {}
    for sm in {r["smid"] for r in rows}:
        a, b = cal["A"].get(sm), cal["B"].get(sm)
        if a and b and b[0] > a[0]:
            f = (b[1] - a[1]) / (b[0] - a[0])
            anchors[sm] = (a[0], a[1], f)
            fs.append(f)
        else:
            bad_sms.add(sm)
    for r in rows:
        r2 = dict(r)
        an = anchors.get(r["smid"])
        if an:
            tA, cA, f = an
            r2["t_start_ns"] = tA + (r["cyc_start"] - cA) / f
            r2["t_end_ns"] = tA + (r["cyc_end"] - cA) / f
            # raw stamp is the tick floor of the true time: true - raw in [0, tick) (+ read latency)
            dev = r2["t_start_ns"] - r["t_start_ns"]
            info["max_dev_vs_raw_ns"] = max(info["max_dev_vs_raw_ns"], abs(dev - tick_ns / 2) - tick_ns / 2, 0.0)
        out.append(r2)
    info["calibrated_sms"] = len(anchors)
    info["uncalibrated_sms"] = len(bad_sms)
    if fs:
        info["f_ghz_min"], info["f_ghz_max"] = min(fs), max(fs)
    return out, info


def union_length(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def metrics(rows, sms):
    t0 = min(r["t_start_ns"] for r in rows)
    t1 = max(r["t_end_ns"] for r in rows)
    span = t1 - t0
    by_sm = {}
    for r in rows:
        by_sm.setdefault(r["smid"], []).append((r["t_start_ns"] - t0, r["t_end_ns"] - t0))
    n_sm = max(sms, max(by_sm) + 1)
    busy = [union_length(by_sm.get(s, [])) for s in range(n_sm)]
    active = [b for s, b in enumerate(busy) if s in by_sm]
    ends = [max(e for _, e in by_sm[s]) for s in by_sm]
    mean_all = sum(busy) / sms if sms else 0.0
    mean_act = sum(active) / len(active)
    counts = [len(by_sm[s]) for s in by_sm]
    stamps = sorted({r.get("raw_t_start", r["t_start_ns"]) for r in rows} |
                    {r.get("raw_t_end", r["t_end_ns"]) for r in rows})
    diffs = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    clocks = [(r["cyc_end"] - r["cyc_start"]) / (r["t_end_ns"] - r["t_start_ns"])
              for r in rows if r["t_end_ns"] > r["t_start_ns"]]
    med_end = statistics.median(ends)
    return {
        "blocks": len(rows),
        "span_us": span / 1e3,
        "max_busy_us": max(busy) / 1e3,
        "mean_busy_us": mean_all / 1e3,
        "max_over_mean": max(busy) / mean_all if mean_all > 0 else float("nan"),
        "max_over_mean_active": max(active) / mean_act if mean_act > 0 else float("nan"),
        "sm_eff": mean_all / span if span > 0 else float("nan"),
        "active_sms": len(by_sm),
        "idle_sms": sms - len([s for s in by_sm if s < sms]),
        "tail_us": (t1 - t0 - med_end) / 1e3,
        "tail_frac": (t1 - t0 - med_end) / span if span > 0 else float("nan"),
        "blocks_per_sm_min": min(counts) if len(by_sm) == sms else 0,
        "blocks_per_sm_max": max(counts),
        "t_res_ns": functools.reduce(math.gcd, diffs) if diffs else 0,   # tick of the raw stamps
        "sm_clock_ghz": statistics.median(clocks) if clocks else float("nan"),
        "start_skew_us": (max(r["t_start_ns"] for r in rows) - t0) / 1e3 if len(rows) else 0.0,
    }


PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#9c755f"]


def gantt_svg(rows, sms, title, path):
    """One lane per SM; co-resident blocks are stacked as thin sub-lanes inside the SM lane."""
    t0 = min(r["t_start_ns"] for r in rows)
    span = max(r["t_end_ns"] for r in rows) - t0 or 1
    W, lane, left, top = 1000, 9, 60, 40
    n_sm = max(sms, max(r["smid"] for r in rows) + 1)
    H = top + n_sm * lane + 40
    by_sm = {}
    for r in rows:
        by_sm.setdefault(r["smid"], []).append(r)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + W + 20}" height="{H}" '
           f'font-family="sans-serif" font-size="10">',
           f'<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="16" font-size="12">{title}</text>',
           f'<text x="{left}" y="30" fill="#555">x: time since first block start ({span / 1e3:.1f} us total); '
           f'y: SM id; colour: block id mod {len(PALETTE)}; grey lanes = idle SMs</text>']
    for s in range(n_sm):
        y = top + s * lane
        if s not in by_sm:
            out.append(f'<rect x="{left}" y="{y}" width="{W}" height="{lane - 1}" fill="#eee"/>')
        if s % 10 == 0:
            out.append(f'<text x="{left - 6}" y="{y + lane - 1}" text-anchor="end">{s}</text>')
    for s, blocks in by_sm.items():
        # greedy sub-lane assignment for overlapping blocks
        ends, assign = [], []
        for r in sorted(blocks, key=lambda r: r["t_start_ns"]):
            for k, e in enumerate(ends):
                if r["t_start_ns"] >= e:
                    ends[k] = r["t_end_ns"]
                    assign.append((r, k))
                    break
            else:
                ends.append(r["t_end_ns"])
                assign.append((r, len(ends) - 1))
        sub = (lane - 1) / max(1, len(ends))
        for r, k in assign:
            x = left + (r["t_start_ns"] - t0) / span * W
            w = max(0.5, (r["t_end_ns"] - r["t_start_ns"]) / span * W)
            y = top + s * lane + k * sub
            out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{max(0.5, sub - 0.2):.2f}" '
                       f'fill="{PALETTE[r["block"] % len(PALETTE)]}"><title>block {r["block"]} SM {s} '
                       f'{(r["t_start_ns"] - t0) / 1e3:.2f}-{(r["t_end_ns"] - t0) / 1e3:.2f} us</title></rect>')
    for i in range(6):
        x = left + i * W / 5
        out.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{H - 35}" stroke="#999" stroke-width="0.5"/>')
        out.append(f'<text x="{x}" y="{H - 22}" text-anchor="middle">{span * i / 5 / 1e3:.1f} us</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


KEYS = ["blocks", "span_us", "max_busy_us", "mean_busy_us", "max_over_mean", "max_over_mean_active",
        "sm_eff", "active_sms", "idle_sms", "tail_us", "tail_frac", "blocks_per_sm_min", "blocks_per_sm_max",
        "t_res_ns", "sm_clock_ghz", "start_skew_us"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--svg-dir", help="write one Gantt SVG per file (launch 0 and the median-span launch)")
    ap.add_argument("--summary", help="append per-file median metrics to this CSV")
    ap.add_argument("--raw", action="store_true", help="ignore the clock calibration (raw globaltimer stamps)")
    args = ap.parse_args()
    summary_rows = []
    for path in args.files:
        meta, rows, cal = read_timeline(path)
        sms = int(meta.get("sms", 80))
        # Rebase to small integers first: globaltimer is ~1.8e18 ns, where a double resolves only
        # ~256 ns. All arithmetic below is then exact enough (values < 1e12).
        base = min(r["t_start_ns"] for r in rows)
        for r in rows:
            r["t_start_ns"] -= base
            r["t_end_ns"] -= base
            r["raw_t_start"], r["raw_t_end"] = r["t_start_ns"], r["t_end_ns"]
        for ph in cal:
            cal[ph] = {sm: (tA - base, cA, tB - base, cB) for sm, (tA, cA, tB, cB) in cal[ph].items()}
        cinfo = {"calibrated_sms": 0, "uncalibrated_sms": len({r["smid"] for r in rows}),
                 "f_ghz_min": float("nan"), "f_ghz_max": float("nan"), "max_dev_vs_raw_ns": float("nan")}
        if not args.raw and cal["A"] and cal["B"]:
            rows, cinfo = calibrate(rows, cal)
        launches = sorted({r["launch"] for r in rows})
        per = [metrics([r for r in rows if r["launch"] == l], sms) for l in launches]
        med = {k: statistics.median(m[k] for m in per) for k in KEYS}
        allst = sorted({r["raw_t_start"] for r in rows} | {r["raw_t_end"] for r in rows})
        dd = [b - a for a, b in zip(allst, allst[1:]) if b > a]
        med["t_res_ns"] = functools.reduce(math.gcd, dd) if dd else 0   # over all launches of the file
        label = (f'{meta.get("kernel", "?")} n={meta.get("n")} b={meta.get("block")} q={meta.get("q")} '
                 f'pf={meta.get("prefetch")} u={meta.get("unroll")}')
        print(f"== {os.path.basename(path)}: {label}  ({len(launches)} launches, median shown)")
        print(f'   span {med["span_us"]:.2f} us | max busy {med["max_busy_us"]:.2f} | mean busy (80 SMs) '
              f'{med["mean_busy_us"]:.2f} | max/mean {med["max_over_mean"]:.3f} (active SMs: '
              f'{med["max_over_mean_active"]:.3f}) | SM eff {med["sm_eff"]:.3f}')
        print(f'   clock calibration: {cinfo["calibrated_sms"]} SMs calibrated, {cinfo["uncalibrated_sms"]} raw; '
              f'f {cinfo["f_ghz_min"]:.4f}..{cinfo["f_ghz_max"]:.4f} GHz; calibrated vs raw start outside '
              f'one tick by <= {cinfo["max_dev_vs_raw_ns"]:.0f} ns'
              + ("" if cinfo["uncalibrated_sms"] == 0 else "  ** WARNING: raw 1.024 us stamps used for some SMs **"))
        print(f'   active {med["active_sms"]:.0f} / idle {med["idle_sms"]:.0f} SMs | blocks/SM '
              f'{med["blocks_per_sm_min"]:.0f}..{med["blocks_per_sm_max"]:.0f} | tail {med["tail_us"]:.2f} us '
              f'({100 * med["tail_frac"]:.1f}% of span) | start skew {med["start_skew_us"]:.2f} us | '
              f'timer res {med["t_res_ns"]:.0f} ns | SM clock {med["sm_clock_ghz"]:.3f} GHz')
        if args.svg_dir:
            os.makedirs(args.svg_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            spans = [m["span_us"] for m in per]
            lmed = launches[sorted(range(len(spans)), key=lambda i: spans[i])[len(spans) // 2]]
            for l, tag in sorted({(launches[0], "first"), (lmed, "median")}):
                sub = [r for r in rows if r["launch"] == l]
                svg = os.path.join(args.svg_dir, f"{base}_launch{l}.svg")
                gantt_svg(sub, sms, f"{label}  launch {l} ({tag})", svg)
        row = {"file": os.path.basename(path), **{k: meta.get(k, "") for k in
               ("kernel", "n", "block", "q", "q_requested", "prefetch", "unroll", "grid", "blocks_per_sm",
                "prod_median_ms", "tl_median_ms")}, "launches": len(launches), **med, **cinfo}
        summary_rows.append(row)
    if args.summary and summary_rows:
        exists = os.path.exists(args.summary)
        with open(args.summary, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(summary_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
