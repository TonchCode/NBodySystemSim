#!/usr/bin/env python3
"""Tree round-1 and round-2 analysis: parse nbody_bench logs, timelines and per-warp CSVs (stdlib only).

    python tools/tree_analysis.py results/2026-10-17_tree_round1
    python tools/tree_analysis.py results/2026-10-17_tree_round2 --round2   (round 2: scorecard.csv)

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


# =================================================================================================
# Tree round 2 (docs/03-design.md §17, docs/predictions/2026-10-17-tree-round2.md)
#
#     python tools/tree_analysis.py results/2026-10-17_tree_round2 --round2
#
# Writes <dir>/tree_metrics.csv, <dir>/timeline_slots.csv, <dir>/derived.csv and
# <dir>/scorecard.csv (one row per pre-registered item ID; the ranges below are copied verbatim
# from the committed predictions file; a multi-cell ID is a HIT only if every cell is in range).
# =================================================================================================
def parse_log2(path):
    r = parse_log(path)
    if not r:
        return None
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"# ic=(\S+) seed=\d+ theta=" + NUM + r" leaf=(\d+)", t)
    if m:
        r["leaf"] = int(m.group(3))
    m = re.search(r"rates: real " + NUM + r" G interactions/s over the full pipeline \(" + NUM + r" ms\), " + NUM +
                  r" over the traversal", t)
    if m:
        r.update(rate_real=float(m.group(1)), rate_trav=float(m.group(3)))
    m = re.search(r"TG3 \(error <= (\S+) at theta=" + NUM + r"\): rms " + NUM + " vs " + NUM + r" \(ratio " + NUM +
                  r"\), p99 " + NUM + " vs " + NUM + r" \(ratio " + NUM + r"\) => (PASS|FAIL)", t)
    if m:
        r.update(tg3_rms_ratio=float(m.group(5)), tg3_p99_ratio=float(m.group(8)), tg3=m.group(9))
    m = re.search(r"work vs (\S+): interactions x" + NUM + ", pops x" + NUM, t)
    if m:
        r.update(union_i=float(m.group(2)), union_v=float(m.group(3)))
    m = re.search(r"slots with fewer pops or interactions than \S+: (\d+)", t)
    if m:
        r["slots_below"] = int(m.group(1))
    m = re.search(r"exact vs (\S+): accelerations (bitwise equal|DIFFER), per-slot pop/interaction counts (\S+)", t)
    if m:
        r.update(exact_acc=m.group(2) == "bitwise equal", exact_counts=m.group(3))
    m = re.search(r"resident (\d+) blocks/SM", t)
    r["bps"] = int(m.group(1)) if m else -1
    m = re.search(r"occupancy cap: --occ-blocks (\d+) -> (\d+) B padding per block -> resident (\d+)", t)
    if m:
        r.update(occ_req=int(m.group(1)), occ_res=int(m.group(3)))
    r["control"] = "CONTROL kernel" in t
    m = re.search(r"gates \(D-009\): .*=> (PASS|FAIL)", t)
    if m:
        r["d009"] = m.group(1)
    r["exit_fail"] = "FAIL" in t and "=> FAIL" in t
    return r


def logcross(ns, ratio):
    """Crossover N where log(T_tree/T_ap) changes sign (linear interpolation in log N)."""
    import math
    pts = sorted(zip(ns, ratio))
    for (n0, r0), (n1, r1) in zip(pts, pts[1:]):
        l0, l1 = math.log(r0), math.log(r1)
        if l0 > 0 >= l1:
            x = math.log(n0) + (math.log(n1) - math.log(n0)) * l0 / (l0 - l1)
            return math.exp(x)
    return float("nan")


def round2(d):
    import math
    recs = [r for r in (parse_log2(p) for p in sorted(glob.glob(os.path.join(d, "logs", "*.txt")))) if r]
    with open(os.path.join(d, "tree_metrics.csv"), "w", newline="") as f:
        keys = sorted({k for r in recs for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow(r)
    by = {r["name"]: r for r in recs}
    A = lambda k, ic, n: by.get(f"A_{k}_{ic}_{n}")
    NS = [65536, 262144, 1048576]
    ICS = ["plummer", "core"]
    der = {}   # derived values, key -> value

    print("## Set A (theta 0.5, leaf 16, block 128): traverse ms | total | ipb vpb | eff_i eff_v | wmm | trav G/s | rms p99 | checks")
    for ic in ICS:
        for n in NS:
            for k in ["bh_naive", "bh_persistent_lane_naive", "bh_sorted", "bh_persistent", "bh_persistent_lane",
                      "bh_persistent_static", "bh_coherent", "bh_warp", "bh_warp_list", "bh_persistent_warp",
                      "bh_warp_naive"]:
                r = A(k, ic, n)
                if not r:
                    continue
                chk = []
                if "tg3" in r: chk.append(f"TG3 {r['tg3']} rms x{r['tg3_rms_ratio']:.3f} p99 x{r['tg3_p99_ratio']:.3f}")
                if "union_i" in r: chk.append(f"work x{r['union_i']:.3f}/x{r['union_v']:.3f}")
                if "exact_acc" in r: chk.append(f"exact {'ok' if r['exact_acc'] and r['exact_counts'] == 'yes' else 'FAIL'}")
                print(f"{k:24s} {ic:7s} {n:8d} | {r['trav']:8.3f} | {r['total_ms']:8.3f} | {r['ipb']:7.1f} {r['vpb']:7.1f} | "
                      f"{r['eff_i']:.3f} {r['eff_v']:.3f} | {r['wmm']:.3f} | {r['rate']:6.1f} | "
                      f"{r.get('rms', float('nan')):.2e} {r.get('p99', float('nan')):.2e} | {r['gates']} {'; '.join(chk)}")
    # ---- derived ratios
    for ic in ICS:
        for n in NS:
            g = lambda k: A(k, ic, n)
            s5, s4, c, w7, wl = g("bh_sorted"), g("bh_naive"), g("bh_coherent"), g("bh_warp"), g("bh_warp_list")
            if not (s5 and c and w7):
                continue
            der[f"rateC_{ic}_{n}"] = c["rate"]
            der[f"T5_TC_{ic}_{n}"] = s5["trav"] / c["trav"]
            der[f"T7_TC_{ic}_{n}"] = w7["trav"] / c["trav"]
            der[f"T5_T7_{ic}_{n}"] = s5["trav"] / w7["trav"]
            der[f"tot5_tot7_{ic}_{n}"] = s5["total_ms"] / w7["total_ms"]
            der[f"rate7_{ic}_{n}"] = w7["rate"]
            der[f"uI_{ic}_{n}"] = w7["ipb"] / s5["ipb"]
            der[f"uV_{ic}_{n}"] = w7["vpb"] / s5["vpb"]
            der[f"eff7_{ic}_{n}"] = min(w7["eff_i"], w7["eff_v"])
            der[f"effC_{ic}_{n}"] = min(c["eff_i"], c["eff_v"])
            der[f"wmmC_{ic}_{n}"] = c["wmm"]
            if wl: der[f"Tw_Tl_{ic}_{n}"] = w7["trav"] / wl["trav"]
            der[f"tg3rms_{ic}_{n}"] = w7.get("tg3_rms_ratio", float("nan"))
            der[f"tg3p99_{ic}_{n}"] = w7.get("tg3_p99_ratio", float("nan"))
            der[f"f_{ic}_{n}"] = math.log(s5["trav"] / w7["trav"]) / math.log(s5["trav"] / c["trav"])
            for k, base, key in [("bh_persistent", s5, "P1"), ("bh_persistent_lane", s5, "P3"),
                                 ("bh_persistent_lane_naive", s4, "P4"), ("bh_persistent_warp", w7, "P7"),
                                 ("bh_persistent_static", s5, "P8")]:
                r = g(k)
                if r and base:
                    der[f"{key}_{ic}_{n}"] = base["trav"] / r["trav"]
                    if k == "bh_persistent_lane_naive":
                        der[f"P6eff_{ic}_{n}"] = r["eff_v"]
                        der[f"P6eff4_{ic}_{n}"] = base["eff_v"]
            ps, pp = g("bh_persistent_static"), g("bh_persistent")
            if ps and pp:
                der[f"static_vs_dyn_{ic}_{n}"] = ps["trav"] / pp["trav"]
    for n in NS:
        wn, s4, w7 = A("bh_warp_naive", "plummer", n), A("bh_naive", "plummer", n), A("bh_warp", "plummer", n)
        if wn and s4 and w7:
            der[f"W16I_{n}"] = wn["ipb"] / s4["ipb"]
            der[f"W16T_{n}"] = s4["trav"] / wn["trav"]
            der[f"W17_{n}"] = wn["trav"] / w7["trav"]
    print("\n## Derived (A)")
    for k in sorted(der):
        print(f"  {k:28s} {der[k]:.4f}")
    # ---- A2 interleaved repeats
    print("\n## A2 interleaved #6 repeats at 1M (ratio sibling/arm per repetition)")
    a2 = {}
    for nm, r in by.items():
        mm = re.match(r"A2_r(\d)_(\S+?)_(plummer|core)_1048576_(\S+)", nm)
        if mm:
            a2.setdefault((mm.group(4), mm.group(3)), {}).setdefault(int(mm.group(1)), {})[mm.group(2)] = r["trav"]
    for (pair, ic), reps in sorted(a2.items()):
        k0, k1 = pair.split("-")
        rat = [v[k0] / v[k1] for v in reps.values() if k0 in v and k1 in v]
        if rat:
            der[f"A2_{pair}_{ic}"] = statistics.median(rat)
            print(f"  {pair:38s} {ic:7s} ratios {' '.join(f'{x:.4f}' for x in rat)}  median {statistics.median(rat):.4f}  "
                  f"spread {max(rat) - min(rat):.4f}")
    # ---- timelines
    print("\n## Timelines (B): median over 5 launches")
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "timeline", "tl*.csv"))):
        m, meta = slot_metrics(p)
        m["file"] = os.path.basename(p)
        rows.append(m)
        print(f"{m['file']:46s} sm_eff {m['sm_eff']:.3f} tail {100 * m['tail_frac']:.1f}% | slot fill {m['fill']:.3f} "
              f"| dur max/mean {m['dur_max_mean']:.2f} | span {m['span_us']:.0f} us clk {m['sm_clock_ghz']:.2f}")
    if rows:
        with open(os.path.join(d, "timeline_slots.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    tl = {m["file"]: m for m in rows}
    for ic in ICS:
        for n in [65536, 1048576]:
            for k in ["bh_sorted", "bh_warp", "bh_persistent", "bh_coherent"]:
                m = tl.get(f"tlw_{k}_{ic}_{n}.csv")
                if m:
                    der[f"fillw_{k}_{ic}_{n}"] = m["fill"]
                    der[f"smeffw_{k}_{ic}_{n}"] = m["sm_eff"]
            m = tl.get(f"tlb_bh_sorted_{ic}_{n}.csv")
            if m:
                der[f"fillb_bh_sorted_{ic}_{n}"] = m["fill"]
    # ---- set C
    print("\n## Set C (leaf x block, plummer, theta 0.5): traversal ms")
    best = {}
    for n in [262144, 1048576]:
        for k in ["bh_naive", "bh_sorted", "bh_warp"]:
            cells = {}
            for leaf in [4, 8, 16, 32]:
                for b in [64, 128, 256]:
                    r = by.get(f"C_{k}_n{n}_l{leaf}_b{b}")
                    if r:
                        cells[(leaf, b)] = r
            if not cells:
                continue
            bl = min(cells, key=lambda x: cells[x]["trav"])
            best[(k, n)] = (bl, cells[bl])
            print(f"  {k:10s} N={n:8d} best leaf {bl[0]:2d} block {bl[1]:3d} trav {cells[bl]['trav']:.3f} | " +
                  " ".join(f"l{l}b{b}={cells[(l, b)]['trav']:.2f}" for (l, b) in sorted(cells)))
            der[f"Lbestleaf_{k}_{n}"] = bl[0]
            l16 = {b: cells[(16, b)]["trav"] for b in [64, 128, 256] if (16, b) in cells}
            if l16 and 128 in l16:
                der[f"Lblk_{k}_{n}"] = l16[128] / min(l16.values()) - 1
            if k == "bh_warp" and (32, 128) in cells and (16, 128) in cells:
                der[f"L6I_{n}"] = cells[(32, 128)]["ipb"] / cells[(16, 128)]["ipb"]
                der[f"L6V_{n}"] = cells[(32, 128)]["vpb"] / cells[(16, 128)]["vpb"]
            if k == "bh_sorted" and (8, 128) in cells and (16, 128) in cells:
                der[f"L7I_{n}"] = cells[(8, 128)]["ipb"] / cells[(16, 128)]["ipb"]
                der[f"L7V_{n}"] = cells[(8, 128)]["vpb"] / cells[(16, 128)]["vpb"]
    if ("bh_sorted", 1048576) in best and ("bh_warp", 1048576) in best:
        bb = best[("bh_sorted", 1048576)][1]["trav"] / best[("bh_warp", 1048576)][1]["trav"]
        c5, c7 = by.get("C_bh_sorted_n1048576_l16_b128"), by.get("C_bh_warp_n1048576_l16_b128")
        der["L5_bestbest"] = bb
        if c5 and c7:
            der["L5_l16"] = c5["trav"] / c7["trav"]
            der["L5_rel"] = bb / der["L5_l16"] - 1
        print(f"  best-vs-best T5/T7 at 1M = {bb:.3f} (leaf-16 b128: {der.get('L5_l16', float('nan')):.3f})")
    for n in [262144, 1048576]:
        if ("bh_naive", n) in best and ("bh_sorted", n) in best:
            der[f"bestbest_T4_T5_{n}"] = best[("bh_naive", n)][1]["trav"] / best[("bh_sorted", n)][1]["trav"]
        if ("bh_sorted", n) in best and ("bh_warp", n) in best:
            der[f"bestbest_T5_T7_{n}"] = best[("bh_sorted", n)][1]["trav"] / best[("bh_warp", n)][1]["trav"]
            der[f"bestbest_tot5_tot7_{n}"] = best[("bh_sorted", n)][1]["total_ms"] / best[("bh_warp", n)][1]["total_ms"]
    # ---- set D
    print("\n## Set D (occupancy, plummer 262144): lane rate G/s at padded blocks/SM")
    for k in ["bh_sorted", "bh_coherent", "bh_warp"]:
        rr = {}
        for ob in [1, 2, 4, 8, 12]:
            r = by.get(f"D_{k}_occ{ob}")
            if r:
                rr[ob] = r
        u = by.get(f"D_{k}_occu")
        if not rr:
            continue
        print(f"  {k:12s} " + " ".join(f"B{ob}(res {rr[ob].get('occ_res', -1)}): {rr[ob]['rate']:.1f}" for ob in sorted(rr)) +
              (f" | unpadded: {u['rate']:.1f}" if u else ""))
        if 12 in rr and 4 in rr: der[f"O_12_4_{k}"] = rr[12]["rate"] / rr[4]["rate"]
        if 2 in rr and 1 in rr: der[f"O_2_1_{k}"] = rr[2]["rate"] / rr[1]["rate"]
        if u and 12 in rr: der[f"O_L1_{k}"] = rr[12]["trav"] / u["trav"]
    # ---- set E crossover
    print("\n## Set E (crossover)")
    xs, rat = [], []
    for n in [8192, 16384, 24576, 32768, 49152, 65536]:
        ap = [r for nm, r in by.items() if nm.startswith("E_") and r["kernel"].startswith("allpairs") and r["n"] == n]
        t7 = by.get(f"E_bh_warp_{n}")
        if not (ap and t7):
            continue
        b = min(ap, key=lambda r: r["total_ms"])
        xs.append(n); rat.append(t7["total_ms"] / b["total_ms"])
        tl_ = by.get(f"E_bh_warp_list_{n}")
        print(f"  N={n:6d} best AP {b['name']} {b['total_ms']:.4f} ms | bh_warp {t7['total_ms']:.4f} (AP/tree "
              f"{b['total_ms'] / t7['total_ms']:.2f})" + (f" | list {tl_['total_ms']:.4f}" if tl_ else ""))
        if n == 65536:
            der["X3_65536"] = b["total_ms"] / t7["total_ms"]
    der["X1"] = logcross(xs, rat) if xs else float("nan")
    print(f"  crossover N (bh_warp) = {der['X1']:.0f}")
    # 262144 and 1M: round-1 best all-pairs (51.0 ms; 822 ms [D]) against this session's bh_warp total
    for n, ap in [(262144, 50.98), (1048576, 822.0)]:
        r = A("bh_warp", "plummer", n)
        if r:
            der[f"X3_{n}"] = ap / r["total_ms"]
    # ---- F, G
    print("\n## Set F (theta sweep, plummer 262144) and G (theta 0)")
    for th in ["0.3", "0.5", "0.7", "1.0"]:
        s5 = by.get(f"F_bh_sorted_th{th}") if th != "0.5" else A("bh_sorted", "plummer", 262144)
        w7 = by.get(f"F_bh_warp_th{th}") if th != "0.5" else A("bh_warp", "plummer", 262144)
        if s5 and w7:
            der[f"F_uI_{th}"] = w7["ipb"] / s5["ipb"]
            der[f"F_T5T7_{th}"] = s5["trav"] / w7["trav"]
            print(f"  theta {th}: union I x{der[f'F_uI_{th}']:.3f}, T5/T7 {der[f'F_T5T7_{th}']:.3f}, TG3 {w7.get('tg3', 'na')}"
                  f" rms x{w7.get('tg3_rms_ratio', float('nan')):.3f} p99 x{w7.get('tg3_p99_ratio', float('nan')):.3f}")
    g5, g7 = by.get("G_bh_sorted_th0"), by.get("G_bh_warp_th0")
    if g5 and g7:
        der["G_ratio"] = g7["trav"] / g5["trav"]
        print(f"  theta 0: bh_warp D-009 {g7.get('d009')} trav {g7['trav']:.3f} vs bh_sorted {g5['trav']:.3f}")
    with open(os.path.join(d, "derived.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k in sorted(der):
            w.writerow([k, der[k]])
    scorecard(d, der, by)


def scorecard(d, der, by):
    """One row per pre-registered ID (docs/predictions/2026-10-17-tree-round2.md, committed)."""
    NS = [65536, 262144, 1048576]
    rows = []
    nan = float("nan")

    def rng(i, desc, keys, ranges, anchored=False, src="", note=""):
        vals = [der.get(k, nan) for k in keys]
        ok = all(lo <= v <= hi for v, (lo, hi) in zip(vals, ranges))
        rows.append([i, desc, "; ".join(f"[{lo}, {hi}]" for lo, hi in ranges),
                     "; ".join(f"{k}={v:.4g}" for k, v in zip(keys, vals)), src,
                     "HIT" if ok else "MISS", "yes" if anchored else "no", note])

    def dirn(i, desc, cond, measured, anchored=False, src="", note=""):
        rows.append([i, desc, "directional", measured, src, "HIT" if cond else "MISS", "yes" if anchored else "no", note])

    g = der.get
    srcA = "logs/A_*"
    rng("P2-C1", "control lane-rate, Plummer (G/s)", [f"rateC_plummer_{n}" for n in NS], [(75, 230), (95, 300), (110, 380)], src=srcA)
    rng("P2-C2", "control lane-rate, core (G/s)", ["rateC_core_65536", "rateC_core_1048576"], [(70, 220), (100, 360)], src=srcA)
    rng("P2-C3", "T5/TC Plummer", [f"T5_TC_plummer_{n}" for n in NS], [(2.2, 6.7), (3.1, 9.9), (3.8, 13)], src=srcA)
    rng("P2-C4", "T5/TC core", ["T5_TC_core_65536", "T5_TC_core_1048576"], [(2.4, 7.5), (3.7, 13.4)], src=srcA)
    v = [g(f"T5_TC_plummer_{n}", nan) for n in NS]
    dirn("P2-C5", "T5/TC grows with N (Plummer)", v[0] < v[1] < v[2], " < ".join(f"{x:.3f}" for x in v), src=srcA)
    cc = [g(f"effC_{ic}_{n}", nan) for ic in ["plummer", "core"] for n in NS] + \
         [g(f"wmmC_{ic}_{n}", nan) for ic in ["plummer", "core"] for n in NS]
    dirn("P2-C6", "(C) control SIMT eff 1.000 both sites, per-warp max/mean 1.000",
         all(abs(x - 1) < 5e-4 for x in cc), f"min {min(cc):.4f} max {max(cc):.4f}", src=srcA)
    rng("P2-W1", "union interactions ratio, Plummer", [f"uI_plummer_{n}" for n in NS], [(1.55, 1.95)] * 3, True, srcA)
    rng("P2-W2", "union interactions ratio, core", [f"uI_core_{n}" for n in NS], [(1.50, 2.10)] * 3, False, srcA)
    rng("P2-W3", "union pops ratio, Plummer", [f"uV_plummer_{n}" for n in NS], [(1.20, 1.50)] * 3, True, srcA)
    rng("P2-W4", "union pops ratio, core", [f"uV_core_{n}" for n in NS], [(1.15, 1.55)] * 3, False, srcA)
    dv = abs(g("uI_plummer_1048576", nan) - g("uI_plummer_65536", nan))
    dirn("P2-W5", "union interaction ratio flat in N: |r(1M) - r(65536)| <= 0.15", dv <= 0.15, f"{dv:.4f}", True, srcA)
    ee = [g(f"eff7_{ic}_{n}", nan) for ic in ["plummer", "core"] for n in NS]
    dirn("P2-W6", "(C) bh_warp SIMT eff >= 0.999 both sites", all(x >= 0.999 for x in ee), f"min {min(ee):.4f}", False, srcA)
    rng("P2-W7", "T7/TC, Plummer and core, all N", [f"T7_TC_{ic}_{n}" for ic in ["plummer", "core"] for n in NS],
        [(1.20, 1.80)] * 6, True, srcA)
    rng("P2-W8", "#7 speedup T5/T7 Plummer", [f"T5_T7_plummer_{n}" for n in NS], [(1.4, 4.6), (2.0, 6.9), (2.4, 9.0)], src=srcA)
    rng("P2-W9", "#7 speedup T5/T7 core", [f"T5_T7_core_{n}" for n in NS], [(1.5, 5.0), (2.0, 7.0), (2.4, 9.2)], src=srcA)
    rng("P2-W10", "total speedup Plummer", [f"tot5_tot7_plummer_{n}" for n in NS], [(1.3, 3.2), (1.9, 5.7), (2.4, 8.0)], src=srcA)
    rng("P2-W11", "bh_warp real traversal rate Plummer (G/s)", [f"rate7_plummer_{n}" for n in NS],
        [(85, 280), (110, 370), (120, 450)], src=srcA)
    p = [g(f"T5_T7_plummer_{n}", nan) for n in NS]
    c = [g(f"T5_T7_core_{n}", nan) for n in NS]
    dirn("P2-W12", "T5/T7 grows with N (Plummer and core)", p[0] < p[1] < p[2] and c[0] < c[1] < c[2],
         f"plummer {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}; core {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}", src=srcA)
    rng("P2-W13", "T(bh_warp)/T(bh_warp_list), 6 cells", [f"Tw_Tl_{ic}_{n}" for ic in ["plummer", "core"] for n in NS],
        [(0.95, 1.20)] * 6, True, srcA)
    rng("P2-W14a", "TG3 rms ratio Plummer (262144, 1M)", ["tg3rms_plummer_262144", "tg3rms_plummer_1048576"], [(0.55, 0.78)] * 2, True, srcA)
    rng("P2-W14b", "TG3 rms ratio core", ["tg3rms_core_262144", "tg3rms_core_1048576"], [(0.60, 0.85)] * 2, True, srcA)
    rng("P2-W14c", "TG3 p99 ratio Plummer", ["tg3p99_plummer_262144", "tg3p99_plummer_1048576"], [(0.48, 0.72)] * 2, True, srcA)
    rng("P2-W14d", "TG3 p99 ratio core", ["tg3p99_core_262144", "tg3p99_core_1048576"], [(0.75, 1.00)] * 2, True, srcA)
    tg = [r.get("tg3") for r in by.values() if "tg3" in r]
    dirn("P2-W15", "TG3 passes in every run (A, F)", bool(tg) and all(x == "PASS" for x in tg), f"{tg.count('PASS')}/{len(tg)} PASS", src="logs/A_*, F_*")
    rng("P2-W16", "warp_naive vs #4: interactions ratio (3 N) and T4/T(warp_naive) (65536, 1M)",
        [f"W16I_{n}" for n in NS] + ["W16T_65536", "W16T_1048576"], [(7, 20), (9, 28), (11, 35), (0.5, 2.2), (0.25, 1.3)], src=srcA)
    w17 = [g(f"W17_{n}", nan) for n in NS]
    dirn("P2-W17", "T(warp_naive) > 3 T(bh_warp) at every N", all(x > 3 for x in w17), " ".join(f"{x:.2f}" for x in w17), src=srcA)
    rng("P2-W18", "bh_warp core 65536 warp-unit SM eff", ["smeffw_bh_warp_core_65536"], [(0.65, 0.92)], src="timeline/tlw_bh_warp_core_65536.csv")
    S = "yes (background load)"
    rng("P2-P1", "warp queue on #5, Plummer", [f"P1_plummer_{n}" for n in NS], [(0.96, 1.03), (0.98, 1.05), (0.99, 1.04)], src=srcA, note=S)
    rng("P2-P2", "warp queue on #5, core", [f"P1_core_{n}" for n in NS], [(0.96, 1.03), (0.98, 1.07), (0.99, 1.05)], src=srcA, note=S)
    rng("P2-P3", "lane refill on #5, Plummer and core", [f"P3_{ic}_{n}" for ic in ["plummer", "core"] for n in NS],
        [(0.75, 1.00), (0.82, 1.05), (0.85, 1.08)] * 2, True, srcA, S)
    rng("P2-P4", "lane refill on #4, Plummer", [f"P4_plummer_{n}" for n in NS], [(0.78, 1.02), (0.95, 1.22), (0.97, 1.25)], src=srcA, note=S)
    rng("P2-P5", "lane refill on #4, core", [f"P4_core_{n}" for n in NS], [(0.78, 1.02), (1.02, 1.42), (1.08, 1.55)], src=srcA, note=S)
    e6 = []
    ok6 = True
    for ic in ["plummer", "core"]:
        a, b4 = g(f"P6eff_{ic}_65536", nan), g(f"P6eff4_{ic}_65536", nan)
        ok6 &= abs(a - b4) <= 0.05
        ok6 &= g(f"P6eff_{ic}_262144", nan) >= 0.88 and g(f"P6eff_{ic}_1048576", nan) >= 0.93
        e6.append(f"{ic}: {a:.3f} (#4 {b4:.3f}) / {g(f'P6eff_{ic}_262144', nan):.3f} / {g(f'P6eff_{ic}_1048576', nan):.3f}")
    rows.append(["P2-P6", "eff_pop of lane_naive: #4 +-0.05 / >= 0.88 / >= 0.93", "see desc", "; ".join(e6), srcA,
                 "HIT" if ok6 else "MISS", "no", ""])
    rng("P2-P7", "warp queue on #7, Plummer and core", [f"P7_{ic}_{n}" for ic in ["plummer", "core"] for n in NS],
        [(0.96, 1.03), (0.97, 1.07), (0.98, 1.04)] * 2, src=srcA, note=S)
    rng("P2-P8", "static partition on #5, Plummer and core", [f"P8_{ic}_{n}" for ic in ["plummer", "core"] for n in NS],
        [(0.80, 1.02), (0.90, 1.04), (0.93, 1.03)] * 2, src=srcA, note=S)
    pc = [g(f"P4_core_{n}", nan) for n in [262144, 1048576]]
    pp = [g(f"P4_plummer_{n}", nan) for n in [262144, 1048576]]
    dirn("P2-P9", "lane-refill gain on #4: core > Plummer at 262144 and 1M", pc[0] > pp[0] and pc[1] > pp[1],
         f"core {pc[0]:.3f} {pc[1]:.3f} vs plummer {pp[0]:.3f} {pp[1]:.3f}", src=srcA, note=S)
    ok10 = True
    m10 = []
    for ic in ["plummer", "core"]:
        x = [g(f"P4_{ic}_{n}", nan) for n in NS]
        ok10 &= x[2] >= x[1] > x[0]
        m10.append(f"{ic} {x[0]:.3f} {x[1]:.3f} {x[2]:.3f}")
    dirn("P2-P10", "lane-refill gain on #4: 1M >= 262144 > 65536 (both ICs)", ok10, "; ".join(m10), src=srcA, note=S)
    sv = [g(f"static_vs_dyn_{ic}_{n}", nan) for ic in ["plummer", "core"] for n in NS]
    dirn("P2-P11", "T(static) >= 0.99 T(warp queue) in every cell", all(x >= 0.99 for x in sv), " ".join(f"{x:.3f}" for x in sv), src=srcA, note=S)
    ex = [(r["name"], r["exact_acc"], r["exact_counts"]) for r in by.values() if "exact_acc" in r and r["name"].startswith("A_")]
    dirn("P2-P12", "(C) persistent runs bitwise equal + equal counts", bool(ex) and all(a and c_ == "yes" for _, a, c_ in ex),
         f"{sum(1 for _, a, c_ in ex if a and c_ == 'yes')}/{len(ex)}", src=srcA)
    f13 = [g(f"fillw_bh_sorted_{ic}_1048576", nan) for ic in ["plummer", "core"]]
    fb = [g(f"fillb_bh_sorted_{ic}_1048576", nan) for ic in ["plummer", "core"]]
    fp = [g(f"fillw_bh_persistent_{ic}_1048576", nan) for ic in ["plummer", "core"]]
    ok13 = all(0.90 <= a <= 0.98 and a < b and c_ >= a for a, b, c_ in zip(f13, fb, fp))
    rows.append(["P2-P13", "1M warp-unit fill bh_sorted in [0.90, 0.98], < block fill; persistent >= sorted",
                 "[0.90, 0.98] + directional", "; ".join(f"sorted w {a:.3f} b {b:.3f}, persistent w {c_:.3f}" for a, b, c_ in zip(f13, fb, fp)),
                 "timeline/", "HIT" if ok13 else "MISS", "no", S])
    for i, (lst, rr) in enumerate([("bh_sorted", {8, 16}), ("bh_naive", {8, 16}), ("bh_warp", {16, 32})]):
        bl = [g(f"Lbestleaf_{lst}_{n}", nan) for n in [262144, 1048576]]
        dirn(f"P2-L{i + 1}", f"best leaf for {lst} in {sorted(rr)} (both N)", all(x in rr for x in bl),
             f"262144: {bl[0]:.0f}, 1M: {bl[1]:.0f}", src="logs/C_*")
    lb = [g(f"Lblk_{k}_{n}", nan) for k in ["bh_naive", "bh_sorted", "bh_warp"] for n in [262144, 1048576]]
    dirn("P2-L4", "best block at leaf 16 within 5% of block 128", all(x <= 0.05 for x in lb), " ".join(f"{x:.3f}" for x in lb), src="logs/C_*")
    l5 = g("L5_rel", nan)
    dirn("P2-L5", "best-vs-best T5/T7 at 1M within +-25% of the leaf-16 b128 value", abs(l5) <= 0.25,
         f"best-best {g('L5_bestbest', nan):.3f} vs {g('L5_l16', nan):.3f} ({100 * l5:+.1f}%)", src="logs/C_*")
    rng("P2-L6", "bh_warp leaf 32/16: I ratio, V ratio (both N)", ["L6I_262144", "L6I_1048576", "L6V_262144", "L6V_1048576"],
        [(1.3, 1.8)] * 2 + [(0.60, 0.80)] * 2, src="logs/C_*")
    rng("P2-L7", "bh_sorted leaf 8/16: I ratio, V ratio (both N)", ["L7I_262144", "L7I_1048576", "L7V_262144", "L7V_1048576"],
        [(0.75, 0.95)] * 2 + [(1.15, 1.50)] * 2, src="logs/C_*")
    o = lambda k: g(k, nan)
    dirn("P2-O1", "bh_sorted rate(12)/rate(4) <= 1.35", o("O_12_4_bh_sorted") <= 1.35, f"{o('O_12_4_bh_sorted'):.3f}", src="logs/D_*")
    dirn("P2-O2", "bh_coherent rate(12)/rate(4) >= 1.8", o("O_12_4_bh_coherent") >= 1.8, f"{o('O_12_4_bh_coherent'):.3f}", src="logs/D_*")
    dirn("P2-O3", "bh_warp rate(12)/rate(4) >= 1.6", o("O_12_4_bh_warp") >= 1.6, f"{o('O_12_4_bh_warp'):.3f}", src="logs/D_*")
    o4 = [o(f"O_2_1_{k}") for k in ["bh_sorted", "bh_coherent", "bh_warp"]]
    dirn("P2-O4", "rate(2)/rate(1) >= 1.6 for each", all(x >= 1.6 for x in o4), " ".join(f"{x:.3f}" for x in o4), src="logs/D_*")
    rng("P2-O5", "L1: unpadded/padded-12 speed, bh_sorted", ["O_L1_bh_sorted"], [(1.00, 1.50)], src="logs/D_*")
    rng("P2-O6", "L1: unpadded/padded-12 speed, bh_coherent and bh_warp", ["O_L1_bh_coherent", "O_L1_bh_warp"], [(0.95, 1.25)] * 2, src="logs/D_*")
    rng("P2-D1", "share of the #5-control gap closed by #7 (Plummer)", ["f_plummer_65536", "f_plummer_1048576"], [(0.45, 0.90), (0.55, 0.95)], src=srcA)
    d2 = [1 - g(f"fillw_bh_warp_{ic}_1048576", nan) for ic in ["plummer", "core"]]
    dirn("P2-D2", "#7 SM-level loss at 1M (1 - warp-unit fill) <= 0.10", all(x <= 0.10 for x in d2), " ".join(f"{x:.3f}" for x in d2), src="timeline/")
    rng("P2-X1", "crossover N bh_warp vs best all-pairs", ["X1"], [(20000, 36000)], src="logs/E_*")
    dirn("P2-X2", "bh_warp crossover below 39,700", g("X1", nan) < 39700, f"{g('X1', nan):.0f}", src="logs/E_*")
    rng("P2-X3", "bh_warp total speedup over best all-pairs", ["X3_65536", "X3_262144", "X3_1048576"], [(2.4, 5.5), (11, 28), (55, 170)],
        src="logs/E_*, A_*", note="262144 and 1M against the round-1 all-pairs times (1M [D])")
    fu = [g(f"F_uI_{t}", nan) for t in ["0.3", "0.5", "1.0"]]
    ft = [g(f"F_T5T7_{t}", nan) for t in ["0.3", "1.0"]]
    dirn("P2-X4", "union ratio grows with theta (0.3 < 0.5 < 1.0) and T5/T7(0.3) > T5/T7(1.0)", fu[0] < fu[1] < fu[2] and ft[0] > ft[1],
         f"uI {fu[0]:.3f} {fu[1]:.3f} {fu[2]:.3f}; T5/T7 {ft[0]:.3f} vs {ft[1]:.3f}", src="logs/F_*")
    g7 = by.get("G_bh_warp_th0")
    gr = g("G_ratio", nan)
    dirn("P2-X5", "theta 0: bh_warp D-009 PASS, traversal within +-15% of bh_sorted", bool(g7) and g7.get("d009") == "PASS" and abs(gr - 1) <= 0.15,
         f"D-009 {g7.get('d009') if g7 else 'na'}, ratio {gr:.3f}", src="logs/G_*")
    allok = [r for r in by.values() if not r["name"].startswith("contam")]
    gf = [r["name"] for r in allok if r["gates"] == "FAIL"]
    dirn("P2-G1", "every A-G run exits 0 with all gates PASS", not gf, f"{len(allok)} runs, gate FAILs: {gf[:5]}", src="logs/index.txt")
    with open(os.path.join(d, "scorecard.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "item", "predicted_range", "measured", "source", "verdict", "anchored", "note"])
        w.writerows(rows)
    hits = sum(1 for r in rows if r[5] == "HIT")
    print(f"\n## Scorecard: {hits} HIT / {len(rows) - hits} MISS ({len(rows)} items) -> scorecard.csv")
    for r in rows:
        print(f"  {r[0]:8s} {r[5]:4s} {'A' if r[6] == 'yes' else ' '} {r[1][:60]:60s} | {r[3][:110]}")


if __name__ == "__main__":
    if "--round2" in sys.argv:
        round2(sys.argv[1])
    else:
        main()
