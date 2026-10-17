#!/usr/bin/env python3
"""Tree round-1 analysis: parse nbody_bench logs, timelines and per-warp CSVs (stdlib only).

    python tools/tree_analysis.py results/2026-10-17_tree_round1

Writes <dir>/tree_metrics.csv (one row per log) and <dir>/timeline_slots.csv, and prints
the tables used in summary.md. Loss decomposition for #6 vs #7 (docs/03-design.md §16):
  (a) idle lanes inside warps: 1 - E_simt, E_simt = issue-weighted SIMT efficiency
        E_simt = (c_I*sum I + c_V*sum V) / (32 * (c_I*sum leaderI + c_V*sum leaderV))
      with c_I = 17.4 issue slots per interaction (SASS, 03-design §0.1) and c_V = issue
      slots per pop (stated per run; a sensitivity range is printed);
      split into loop-level (lanes finished, waiting for the warp's slowest lane: pop-site
      efficiency) and branch-level (lanes inside the loop but on another branch);
  (b) uneven work between warps/blocks: from the timeline, the fraction of resident
      block-slot time lost after the first block on an SM finishes, i.e. the occupancy decay
      inside SMs, and the spread of block durations;
  (c) idle SMs and the tail: 1 - SM efficiency, idle SMs, tail % (timeline_analysis.py).
"""
import csv
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline_analysis as tla  # noqa: E402

NUM = r"([-+0-9.eE]+)"


def parse_log(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    r = {"name": os.path.basename(path)[:-4]}
    m = re.search(r"kernel=(\S+) n=(\d+) block=(\d+) q=(-?\d+).*?median=" + NUM + " ms", t)
    if not m:
        return None
    r.update(kernel=m.group(1), n=int(m.group(2)), block=int(m.group(3)), q=int(m.group(4)), total_ms=float(m.group(5)))
    m = re.search(r"# ic=(\S+) seed=\d+ theta=" + NUM, t)
    r["ic"], r["theta"] = (m.group(1), float(m.group(2))) if m else ("plummer", -1)
    m = re.search(r"keys " + NUM + "  sort " + NUM + "  tree " + NUM + "  order " + NUM + "  traverse " + NUM, t)
    if m:
        r.update(keys=float(m.group(1)), sort=float(m.group(2)), tree=float(m.group(3)), order=float(m.group(4)),
                 trav=float(m.group(5)))
        r["build"] = r["keys"] + r["sort"] + r["tree"]
    m = re.search(r"work: " + NUM + r" interactions/body \(" + NUM + r" total\), " + NUM +
                  r" nodes popped/body \| per-warp max/mean " + NUM + r" \(median warp " + NUM +
                  r"\) \| SIMT eff interactions " + NUM + ", pops " + NUM + r" \| warp-cost max/mean " + NUM +
                  r" \| traversal " + NUM, t)
    if m:
        keys = ["ipb", "itot", "vpb", "wmm", "wmm_med", "eff_i", "eff_v", "wcost", "rate"]
        r.update({k: float(v) for k, v in zip(keys, m.groups())})
    m = re.search(r"rms " + NUM + "  p50 " + NUM + "  p99 " + NUM + "  max " + NUM, t)
    if m:
        r.update(rms=float(m.group(1)), p50=float(m.group(2)), p99=float(m.group(3)), emax=float(m.group(4)))
    m = re.search(r"=> (PASS|FAIL)", t)
    r["gates"] = m.group(1) if m else "na"
    m = re.search(r"normMax " + NUM + " <= " + NUM, t)
    if m:
        r["normmax"], r["g1thr"] = float(m.group(1)), float(m.group(2))
    r["overflow"] = "overflow" in t.lower() and "stack ok" not in t
    if r.get("ipb") is None and r["kernel"].startswith("allpairs"):
        r["rate_ap"] = r["n"] * r["n"] / (r["total_ms"] * 1e-3) * 1e-9
    return r


def warp_issue_eff(path, c_i=17.4, c_v=20.0):
    """Issue-weighted SIMT efficiency and its split from a per-warp CSV."""
    si = sv = li = lv = 0
    with open(path) as f:
        for w in csv.DictReader(f):
            si += int(w["inter_sum"]); sv += int(w["visit_sum"])
            li += int(w["lead_inter"]); lv += int(w["lead_visit"])
    e = (c_i * si + c_v * sv) / (32.0 * (c_i * li + c_v * lv))
    return e, si, sv, li, lv


def slot_metrics(path):
    """Per launch: block-slot occupancy over the span, from a timeline CSV."""
    meta, rows, cal = tla.read_timeline(path)
    base = min(r["t_start_ns"] for r in rows)   # same preprocessing as timeline_analysis.main
    for r in rows:
        r["t_start_ns"] -= base
        r["t_end_ns"] -= base
        r["raw_t_start"], r["raw_t_end"] = r["t_start_ns"], r["t_end_ns"]
    for ph in cal:
        cal[ph] = {sm: (tA - base, cA, tB - base, cB) for sm, (tA, cA, tB, cB) in cal[ph].items()}
    if cal["A"] and cal["B"]:
        rows, _ = tla.calibrate(rows, cal)
    sms = int(meta.get("sms", 80))
    # round 2: warp-unit timelines carry slots_per_sm (resident warps/SM); block timelines = blocks/SM
    bps = int(meta.get("slots_per_sm", meta.get("blocks_per_sm", 12)))
    out = []
    for L in sorted({r["launch"] for r in rows}):
        rr = [r for r in rows if r["launch"] == L]
        m = tla.metrics(rr, sms)
        t0 = min(r["t_start_ns"] for r in rr)
        span = max(r["t_end_ns"] for r in rr) - t0
        dur = [r["t_end_ns"] - r["t_start_ns"] for r in rr]
        nb = len(rr)
        # slot fill: block-time / (slots that could be filled * span); capped by blocks available
        slots = sms * bps
        fill = sum(dur) / (min(slots, nb) * span) if span > 0 else float("nan")
        # within-SM occupancy decay: busy SM time weighted by resident blocks vs its peak
        by_sm = {}
        for r in rr:
            by_sm.setdefault(r["smid"], []).append((r["t_start_ns"], r["t_end_ns"]))
        decay_num = decay_den = 0.0
        for iv in by_sm.values():
            busy = tla.union_length(iv)
            peak = len(iv) if nb <= slots else bps
            decay_num += sum(e - s for s, e in iv)
            decay_den += min(peak, bps) * busy
        m.update(launch=L, blocks=nb, fill=fill, within_sm_occ=decay_num / decay_den if decay_den else float("nan"),
                 dur_max_mean=max(dur) / statistics.mean(dur), dur_cv=statistics.pstdev(dur) / statistics.mean(dur))
        out.append(m)
    keys = ["sm_eff", "max_over_mean", "idle_sms", "tail_frac", "fill", "within_sm_occ", "dur_max_mean", "dur_cv",
            "span_us", "sm_clock_ghz"]
    return {k: statistics.median(o[k] for o in out) for k in keys}, meta


def main():
    d = sys.argv[1]
    recs = [r for r in (parse_log(p) for p in sorted(glob.glob(os.path.join(d, "logs", "*.txt")))) if r]
    with open(os.path.join(d, "tree_metrics.csv"), "w", newline="") as f:
        keys = sorted({k for r in recs for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow(r)
    by = {r["name"]: r for r in recs}

    print("## Set A (theta 0.5): kernel ic N | total build trav | ipb vpb | wmm eff_i eff_v wcost | rate | rms p99 | gates")
    for ic in ["plummer", "collision", "core"]:
        for n in [65536, 262144, 1048576]:
            for k in ["bh_naive", "bh_sorted", "bh_hilbert"]:
                r = by.get(f"A_{k}_{ic}_{n}")
                if not r:
                    continue
                print(f"{k:10s} {ic:9s} {n:8d} | {r['total_ms']:8.3f} {r['build']:.3f} {r['trav']:8.3f} order {r['order']:.3f}"
                      f" | {r['ipb']:7.1f} {r['vpb']:7.1f} | {r['wmm']:.3f} {r['eff_i']:.3f} {r['eff_v']:.3f} "
                      f"{r['wcost']:.2f} | {r['rate']:6.1f} | {r['rms']:.2e} {r['p99']:.2e} | {r['gates']}")
    print("\n## Set C (plummer 262144, theta sweep; theta 0 at 65536)")
    for nm in sorted(n for n in by if n.startswith("C_")):
        r = by[nm]
        print(f"{nm:24s} th={r['theta']} N={r['n']} trav {r.get('trav', 0):9.3f} ipb {r.get('ipb', 0):9.1f} "
              f"wmm {r.get('wmm', 0):.3f} eff_i {r.get('eff_i', 0):.3f} eff_v {r.get('eff_v', 0):.3f} "
              f"rms {r.get('rms', float('nan')):.3e} p99 {r.get('p99', float('nan')):.3e} normMax {r.get('normmax', float('nan')):.2e}"
              f"/{r.get('g1thr', float('nan')):.2e} {r['gates']}")
    print("\n## Set D (crossover): tree totals vs best all-pairs")
    for n in [8192, 16384, 24576, 32768, 49152, 65536, 262144, 524288]:
        ap = [r for nm, r in by.items() if nm.startswith("D_") and r["kernel"].startswith("allpairs") and r["n"] == n]
        best = min(ap, key=lambda r: r["total_ms"]) if ap else None
        t4 = by.get(f"D_bh_naive_{n}"); t5 = by.get(f"D_bh_sorted_{n}")
        s = f"N={n:7d} best AP " + (f"{best['name']} {best['total_ms']:.4f} ms ({best['rate_ap']:.0f} G/s)" if best else "-")
        if t4 and t5:
            s += f" | bh_naive {t4['total_ms']:.4f} (AP/tree {best['total_ms'] / t4['total_ms']:.2f}) " \
                 f"bh_sorted {t5['total_ms']:.4f} (AP/tree {best['total_ms'] / t5['total_ms']:.2f}) " \
                 f"rates {t4['rate']:.1f}/{t5['rate']:.1f} ipb {t5['ipb']:.0f}"
        print(s)
    print("\n## Issue-weighted SIMT efficiency (c_V sensitivity 10/20/40 slots per pop)")
    for p in sorted(glob.glob(os.path.join(d, "warps", "A_*.csv"))) + sorted(glob.glob(os.path.join(d, "warps", "C_*.csv"))):
        es = [warp_issue_eff(p, c_v=c)[0] for c in (10, 20, 40)]
        print(f"{os.path.basename(p)[:-4]:34s} E_simt {es[0]:.3f} / {es[1]:.3f} / {es[2]:.3f}")
    print("\n## Timelines (set B): median over 5 launches")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "timeline", "tl_*.csv"))):
        m, meta = slot_metrics(p)
        m["file"] = os.path.basename(p)
        rows.append(m)
        print(f"{m['file']:34s} sm_eff {m['sm_eff']:.3f} max/mean {m['max_over_mean']:.3f} idle {m['idle_sms']:.0f} "
              f"tail {100 * m['tail_frac']:.1f}% | slot fill {m['fill']:.3f} within-SM occ {m['within_sm_occ']:.3f} "
              f"| block dur max/mean {m['dur_max_mean']:.2f} cv {m['dur_cv']:.2f} | span {m['span_us']:.0f} us "
              f"clk {m['sm_clock_ghz']:.2f}")
    # ---- loss decomposition for #6 vs #7 ----------------------------------------------
    # Reference = the theta = 0 control (every lane walks the identical path: SIMT eff 1, all
    # loads warp-uniform). Work in issue-slot units W = c_I*I + c_V*V per body. Achieved
    # fraction F = (W/t)_run / (W/t)_ref = E_lanes * E_mem * E_sm, with
    #   E_lanes = issue-weighted SIMT efficiency (counters)            -> (a) idle lanes
    #   E_sm    = timeline slot fill (block-slot time / (slots * span)) -> (b)+(c)
    #   E_mem   = F / (E_lanes * E_sm), the residual: scattered (non-broadcast) node, body and
    #             local-stack accesses, i.e. memory-side intra-warp incoherence
    ref = by.get("C_bh_sorted_th0_n65536")
    tl = {m["file"]: m for m in rows}
    if ref and rows:
        print("\n## Loss decomposition (c_V = 20; reference theta=0 N=65536)")
        print("run | F achieved | E_lanes (a) | E_mem (a') | E_sm (b+c) [sm_eff, tail%] | log-share a / a' / b+c")
        for c_v in (20,):
            wref = ref["n"] * (17.4 * ref["ipb"] + c_v * ref["vpb"]) / (ref["trav"] * 1e-3)
            for k in ["bh_naive", "bh_sorted"]:
                for ic in ["plummer", "core"]:
                    for n in [65536, 1048576]:
                        r = by.get(f"A_{k}_{ic}_{n}")
                        t = tl.get(f"tl_{k}_{ic}_{n}.csv")
                        wp = os.path.join(d, "warps", f"A_{k}_{ic}_{n}.csv")
                        if not (r and t and os.path.exists(wp)):
                            continue
                        F = r["n"] * (17.4 * r["ipb"] + c_v * r["vpb"]) / (r["trav"] * 1e-3) / wref
                        el = warp_issue_eff(wp, c_v=c_v)[0]
                        es = t["fill"]
                        em = F / (el * es)
                        import math
                        ls = [-math.log(x) for x in (el, em, es)]
                        tot = sum(ls)
                        print(f"{k:9s} {ic:8s} {n:8d} | {F:.3f} | {el:.3f} | {em:.3f} | {es:.3f} [{t['sm_eff']:.3f}, "
                              f"{100 * t['tail_frac']:.1f}] | {100 * ls[0] / tot:.0f}% / {100 * ls[1] / tot:.0f}% / "
                              f"{100 * ls[2] / tot:.0f}%")
    if rows:
        with open(os.path.join(d, "timeline_slots.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
