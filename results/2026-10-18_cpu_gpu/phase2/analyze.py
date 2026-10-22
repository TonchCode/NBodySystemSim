"""R3 #8 phase 2: parse logs/<cell>/<pass>/<entry>.txt, score docs/predictions/2026-10-18-cpu-gpu.md.

Value of an entry = mean of the two pass medians (p1, p2); `once` cells (L1) have p1 only.
Writes scorecard.csv, entries.csv (every parsed entry) and tables.md. Usage: python analyze.py [phase2 dir]
"""
import csv
import math
import re
import sys
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
L = R / "logs"

RX = {
    "med": r"^kernel=\S+ .*?median=([0-9.]+) ms",
    "compute": r"^phases \(host clock.*?compute ([0-9.]+)",
    "build": r"\| build\(keys\+sort\+tree\) ([0-9.]+)",
    "trav": r"\| build\(keys\+sort\+tree\) [0-9.]+\s+(?:order [0-9.]+\s+)?traverse ([0-9.]+)",
    "keys": r"^phases.*?keys ([0-9.]+)",
    "sort": r"^phases.*?sort ([0-9.]+)",
    "tree": r"^phases.*?tree ([0-9.]+)",
    "d2h": r"^host: d2h ([0-9.]+) ms",
    "h2d": r"^host: d2h [0-9.]+ ms\s+h2d ([0-9.]+)",
    "busy": r"worker busy max/mean ([0-9.]+)",
    "busyP": r"mean busy P ([0-9.]+) ms",
    "busyE": r"mean busy P [0-9.]+ ms E ([0-9.]+) ms",
    "itemsP": r"items on P ([0-9.]+)%",
    "share": r"GPU share ([0-9.]+)",
    "last": r"GPU share [0-9.]+ \(last ([0-9.]+)",
    "gpuonly": r"\(last [0-9.]+, (\d+) GPU-only\)",
    "gpart": r"GPU part ([0-9.]+) ms",
    "cpart": r"CPU part ([0-9.]+) ms",
    "xfer": r"transfer ([0-9.]+) ms",
    "hwall": r"transfer [0-9.]+ ms, wall ([0-9.]+) ms",
    "topo": r"topology updates (\d+)",
    "lag": r"mean lag ([-0-9.]+) launches",
    "topobuild": r"CPU topology build ([-0-9.]+) ms",
    "ipb": r"^work: ([0-9.]+) interactions/body",
    "itot": r"^work: [0-9.]+ interactions/body \(([0-9.e+]+) total\)",
    "norm": r"^validate vs FP64.*?this rms \S+ max \S+ norm ([0-9.e+-]+)",
    "nnorm": r"^validate vs FP64.*?naive rms \S+ max \S+ norm ([0-9.e+-]+)",
    "lrms": r"rms ratio ([0-9.]+)",
    "lwork": r"work ratio ([0-9.]+)",
}
SRX = {
    "gates": r"^gates .*=> (PASS|FAIL)",
    "same": r"^same work as \S+:.*=> (PASS|FAIL)",
    "exact": r"^exact vs \S+: accelerations (bitwise equal|[^,]+)",
    "lg1": r"LG1 \(TN1/TN2 on the stale tree\) (PASS|FAIL)",
    "lg2": r"LG2 \(rms <= 1.10 x fresh\) (PASS|FAIL)",
    "ltree": r"tree check (\S+) \| LG1",
}


def parse(path):
    t = path.read_text(errors="replace")
    d = {}
    for k, rx in RX.items():
        m = re.search(rx, t, re.M)
        if m:
            d[k] = float(m.group(1))
    for k, rx in SRX.items():
        m = re.search(rx, t, re.M)
        if m:
            d[k] = m.group(1)
    d["fail_word"] = bool(re.search(r"FAIL", re.sub(r"legacy TG1 \S+ .*?not gating", "", t)))
    d["error"] = bool(re.search(r"error|abort|CUDA_CHECK|exception", t, re.I)) and "med" not in d
    return d


E = {}      # (cell, entry) -> {"p1": d, "p2": d}
for cell in sorted(p for p in L.iterdir() if p.is_dir()):
    for pas in ("p1", "p2"):
        pd = cell / pas
        if not pd.is_dir():
            continue
        for f in sorted(pd.glob("*.txt")):
            if f.name == "index.txt":
                continue
            E.setdefault((cell.name, f.stem), {})[pas] = parse(f)


def v(cell, ent, key="med"):
    d = E.get((cell, ent))
    if not d:
        return math.nan
    xs = [d[p][key] for p in ("p1", "p2") if p in d and key in d[p]]
    return sum(xs) / len(xs) if xs else math.nan


def spread(cell, ent, key="med"):
    d = E.get((cell, ent), {})
    if "p1" in d and "p2" in d and key in d["p1"] and key in d["p2"]:
        a, b = d["p1"][key], d["p2"][key]
        return abs(a - b) / ((a + b) / 2)
    return math.nan


def s(cell, ent, key):
    d = E.get((cell, ent), {})
    return [d[p].get(key) for p in ("p1", "p2") if p in d]


def ents(cell, prefix=""):
    return sorted({e for (c, e) in E if c == cell and e.startswith(prefix)})


def tree_total(cell, ent):   # CPU compute only: keys + sort + tree + traverse (order is 0 on the CPU)
    return v(cell, ent, "build") + v(cell, ent, "trav")


SC = []


def item(iid, pred, rng, val, src, hit, anch="no", note=""):
    if isinstance(val, float):
        sval = "nan" if math.isnan(val) else f"{val:.4g}"
    else:
        sval = str(val)
    SC.append({"id": iid, "prediction": pred, "range": rng, "measured": sval, "source": src,
               "verdict": "HIT" if hit else "MISS", "anchored": anch, "note": note})


def inr(x, lo, hi):
    return (not math.isnan(x)) and lo <= x <= hi


# ---------------------------------------------------------------- §1 CPU baselines
for n in (16384, 65536, 262144):
    c = f"C1_{n}"
    best = min(((v(c, e, "compute"), e) for e in ents(c, "ap_iv") if "rsqrt" not in e))
    rate = n * n / (best[0] * 1e-3) / 1e9
    item(f"C1-{n}", "cpu_allpairs best rate, 20 LPs dynamic, compute only (G int/s)", "30-50", rate,
         f"logs/{c} ({best[1]})", inr(rate, 30, 50), "yes")
c = "C1_65536"
x = v(c, "ap_iv1_t2048", "compute") / v(c, "ap_iv2_t2048", "compute")
item("C2", "time IV1 / IV2 at 65536 (tile 2048)", "1.00-1.25", x, f"logs/{c}", inr(x, 1.0, 1.25))
x = v(c, "ap_iv2_t2048_rsqrt0", "compute") / v(c, "ap_iv2_t2048", "compute")
item("C3", "time rsqrt0 / rsqrt1 at 65536", "1.3-3.0", x, f"logs/{c}", inr(x, 1.3, 3.0))
c = "C1_262144"
a = v(c, "ap_iv2_t512", "compute") / v(c, "ap_iv2_t2048", "compute")
b = v(c, "ap_iv2_t8192", "compute") / v(c, "ap_iv2_t2048", "compute")
item("C4", "time tile512/tile2048 ; tile8192/tile2048 at 262144 (IV 2)", "0.95-1.05 each", f"{a:.4f} ; {b:.4f}",
     f"logs/{c}", inr(a, .95, 1.05) and inr(b, .95, 1.05))

bhrate = {}
for n, c in ((65536, "C5_65536"), (262144, "C5_262144"), (1048576, "C5_1048576")):
    best = max(((v(c, e, "itot") / (v(c, e, "trav") * 1e-3) / 1e9, e) for e in ents(c, "bh_g")
                if e.split("_")[1] in ("g16", "g32")))
    bhrate[n] = best
    item(f"C5-{n}", "cpu_bh best traversal rate, G {16,32} x leaf {8,16,32}, Plummer (G int/s)", "12-28", best[0],
         f"logs/{c} ({best[1]})", inr(best[0], 12, 28), "yes")
c = "C5_1048576"
x = v(c, "bh_g32_l16", "build")
item("C6", "cpu_bh build (keys+sort+tree) Plummer 1M, G32 leaf 16 (ms)", "5-25", x, f"logs/{c}", inr(x, 5, 25))


def bestG(g):
    return min((v(c, f"bh_g{g}_l{l}"), l) for l in (8, 16, 32))


g8, g16, g32, g64 = bestG(8), bestG(16), bestG(32), bestG(64)
x = g16[0] / g32[0]
item("C7", "time(G16)/time(G32), Plummer 1M, best leaf each (production median)", "0.85-1.10", x,
     f"logs/{c} (G16 leaf {g16[1]}, G32 leaf {g32[1]})", inr(x, .85, 1.10))
x = g8[0] / g16[0]
item("C8-a", "time(G8)/time(G16), Plummer 1M, best leaf", "1.00-1.40", x, f"logs/{c}", inr(x, 1.0, 1.4))
x = g64[0] / g32[0]
item("C8-b", "time(G64)/time(G32), Plummer 1M, best leaf", "0.95-1.30", x, f"logs/{c}", inr(x, .95, 1.3))
sc_ = min((v(c, f"bhs_l{l}", "trav"), l) for l in (8, 16, 32))
gr_ = min((v(c, e, "trav"), e) for e in ents(c, "bh_g") if e.split("_")[1] in ("g16", "g32"))
x = sc_[0] / gr_[0]
item("C9", "cpu_bh_scalar / cpu_bh traversal time, Plummer 1M, best leaf each", "3-10", x,
     f"logs/{c} (scalar leaf {sc_[1]}, {gr_[1]})", inr(x, 3, 10), "yes")

# C10: crossover of CPU tree vs CPU all-pairs (production medians; best over configs incl. p1)
cross = []
for n in (512, 1024, 2048, 4096, 8192, 16384):
    c = f"C10_{n}"
    ap = min(v(c, e, "compute") for e in ents(c, "ap_"))
    bh = min(tree_total(c, e) for e in ents(c, "bh_"))
    cross.append((n, ap, bh))
nx = math.nan
for (n0, a0, b0), (n1, a1, b1) in zip(cross, cross[1:]):
    r0, r1 = math.log(b0 / a0), math.log(b1 / a1)
    if r0 >= 0 > r1:
        nx = math.exp(math.log(n0) + (math.log(n1) - math.log(n0)) * r0 / (r0 - r1))
if math.isnan(nx) and cross[0][2] < cross[0][1]:
    nx = float("-inf")
item("C10", "CPU crossover N (cpu_bh best = cpu_allpairs best, compute only, log-log interpolation)", "1500-8000",
     nx if not math.isinf(nx) else "<512", "logs/C10_*", inr(nx, 1500, 8000) if not math.isinf(nx) else False, "partly")

# C11 / C12: accuracy over every validated CPU run
ap_ok, ap_n, ap_ratio = True, 0, []
bh_ok, bh_n, bh_same = True, 0, 0
for (c, e), d in E.items():
    for p, x in d.items():
        t = (L / c / p / f"{e}.txt").read_text(errors="replace")
        km = re.search(r"^kernel=(\S+)", t, re.M)
        k = km.group(1) if km else "?"
        if k == "cpu_allpairs":
            ap_n += 1
            ap_ok &= x.get("gates") == "PASS"
            if "norm" in x and "nnorm" in x:
                ap_ratio.append(x["norm"] / x["nnorm"])
        if k in ("cpu_bh", "cpu_bh_scalar"):
            bh_n += 1
            bh_ok &= x.get("gates") == "PASS" and x.get("same") in (None, "PASS")
            bh_same += x.get("same") == "PASS"
rr = (min(ap_ratio), max(ap_ratio)) if ap_ratio else (math.nan, math.nan)
item("C11", "cpu_allpairs D-009 PASS in every run; normMax/naive's within +-10%", "all PASS, +-10%",
     f"{ap_n} runs, all PASS={ap_ok}, ratio {rr[0]:.4f}-{rr[1]:.4f}", "logs/C1_*, C10_*, E1_*, S1_*",
     ap_ok and rr[0] >= 0.9 and rr[1] <= 1.1, "yes")
item("C12", "cpu_bh/cpu_bh_scalar TN1/TN2 PASS and same-work PASS in every run", "all PASS",
     f"{bh_n} runs, all PASS={bh_ok}; same-work check printed (G32 / scalar) and PASS in {bh_same}", "logs/C5_*, C10_*, E9_*, S5_*", bh_ok, "yes")

# ---------------------------------------------------------------- §2 P/E
c = "E1_65536"
N = 65536


def rate(e):
    return N * N / (v(c, e, "compute") * 1e-3) / 1e9


E_items = [
    ("E1", "rate(p,12)/rate(p1,6), dynamic", "1.00-1.25", rate("ap_p_dynamic") / rate("ap_p1_dynamic"), 1.0, 1.25, "no"),
    ("E2", "rate(e,8)/rate(p1,6), dynamic", "0.22-0.45", rate("ap_e_dynamic") / rate("ap_p1_dynamic"), .22, .45, "yes"),
    ("E3", "rate(all dyn)/rate(p dyn)", "1.15-1.45", rate("ap_all_dynamic") / rate("ap_p_dynamic"), 1.15, 1.45, "yes"),
    ("E4", "rate(all dyn)/rate(all static)", "1.40-2.00", rate("ap_all_dynamic") / rate("ap_all_static"), 1.4, 2.0, "yes"),
    ("E5", "rate(all static)/rate(p dyn)", "0.60-0.95", rate("ap_all_static") / rate("ap_p_dynamic"), .6, .95, "yes"),
    ("E6", "rate(all weighted)/rate(all dyn)", "0.93-1.02", rate("ap_all_weighted") / rate("ap_all_dynamic"), .93, 1.02, "no"),
]
for iid, pr, rg, x, lo, hi, an in E_items:
    item(iid, pr, rg, x, f"logs/{c}", inr(x, lo, hi), an)
for sch, lo, hi in (("static", 1.4, 2.2), ("dynamic", 1.0, 1.2), ("weighted", 1.0, 1.2)):
    x = v(c, f"ap_all_{sch}", "busy")
    item(f"E7-{sch}", f"worker busy max/mean, all 20, {sch}", f"{lo:.2f}-{hi:.2f}", x, f"logs/{c}", inr(x, lo, hi), "partly")
x = rate("ap_cores_dynamic") / rate("ap_all_dynamic")
item("E8", "rate(cores,14 dyn)/rate(all dyn)", "0.85-1.00", x, f"logs/{c}", inr(x, .85, 1.0))
c = "E9_1M"
for iid, a, b, key, lo, hi, pr in (("E9", "static", "dynamic", "trav", 1.5, 2.3, "traversal static/dynamic"),
                                    ("E10", "static", "weighted", "trav", 1.3, 2.0, "traversal static/weighted"),
                                    ("E11", "weighted", "dynamic", "trav", 1.0, 1.25, "traversal weighted/dynamic"),
                                    ("E12", "static", "dynamic", "build", 1.2, 2.0, "build static/dynamic")):
    x = v(c, f"bh_all_{a}", key) / v(c, f"bh_all_{b}", key)
    item(iid, f"cpu_bh 1M all 20: {pr}", f"{lo}-{hi}", x, f"logs/{c}", inr(x, lo, hi))

# ---------------------------------------------------------------- §3 CPU vs GPU
def gpu_ap(n):
    c = f"S1_{n}"
    return min(v(c, e) for e in ents(c, "gpu_") if not e.endswith("_wall"))


def gpu_ap_wall(n):
    c = f"S1_{n}"
    return min(v(c, e) for e in ents(c, "gpu_") if e.endswith("_wall"))


def cpu_ap(n):
    if n <= 4096:
        return min(v(f"S1_{n}", e, "compute") for e in ents(f"S1_{n}", "cpu_"))
    xs = [v(f"C1_{n}", e, "compute") for e in ents(f"C1_{n}", "ap_iv") if "rsqrt" not in e]
    if n == 16384:
        xs += [v("C10_16384", e, "compute") for e in ents("C10_16384", "ap_")]
    return min(xs)


x = cpu_ap(1024) / gpu_ap(1024)
item("S1", "all-pairs GPU/CPU speedup at 1024 (GPU event, CPU compute)", "4-15", x, "logs/S1_1024", inr(x, 4, 15))
x = cpu_ap(4096) / gpu_ap(4096)
item("S2", "all-pairs GPU/CPU at 4096", "15-35", x, "logs/S1_4096", inr(x, 15, 35))
for n in (16384, 65536, 262144):
    x = cpu_ap(n) / gpu_ap(n)
    item(f"S3-{n}", f"all-pairs GPU/CPU at {n}", "20-40", x, f"logs/S1_{n}, C1_{n}", inr(x, 20, 40))
cpu_faster = [n for n in (64, 128, 256, 512, 1024) if cpu_ap(n) <= gpu_ap(n)]
xo = max(cpu_faster) if cpu_faster else "none (GPU faster at 64)"
item("S4", "largest N in {64..1024} where CPU best <= GPU best (event-timed)", "64-384", xo, "logs/S1_64..1024",
     bool(cpu_faster) and 64 <= max(cpu_faster) <= 384)


def cpu_tree_best(n, ic="plummer"):
    if ic == "plummer":
        c = f"C5_{n}"
        es = [e for e in ents(c, "bh_g") if e.split("_")[1] in ("g16", "g32")] if n < 1048576 else ents(c, "bh_g")
    else:
        c = f"S5_{ic}_{n}"
        es = ents(c, "cpu_bh")
    return min((tree_total(c, e), e, c) for e in es)


def gpu_tree(n, ic="plummer", key="med", ent="gpu_bh_warp"):
    return v(f"S5_{ic}_{n}", ent, key)


rng5 = {65536: (6, 14), 262144: (9, 22), 1048576: (10, 24)}
for n, (lo, hi) in rng5.items():
    cb = cpu_tree_best(n)
    x = cb[0] / gpu_tree(n)
    item(f"S5-{n}", "tree GPU/CPU total (bh_warp best event vs cpu_bh best compute), Plummer", f"{lo}-{hi}", x,
         f"logs/S5_plummer_{n}, {cb[2]} ({cb[1]})", inr(x, lo, hi))
cbt = min((v("C5_1048576", e, "trav"), e) for e in ents("C5_1048576", "bh_g"))
x = cbt[0] / gpu_tree(1048576, key="trav")
item("S6", "tree traversal-only GPU/CPU, Plummer 1M", "8-25", x, f"logs/S5_plummer_1048576, C5_1048576 ({cbt[1]})",
     inr(x, 8, 25))
cbb = min((v("C5_1048576", e, "build"), e) for e in ents("C5_1048576", "bh_g"))
x = cbb[0] / gpu_tree(1048576, key="build")
item("S7", "tree build-only GPU/CPU, Plummer 1M", "8-40", x, f"logs/S5_plummer_1048576, C5_1048576 ({cbb[1]})",
     inr(x, 8, 40))
cb = cpu_tree_best(1048576, "cube_edges")
x = cb[0] / gpu_tree(1048576, "cube_edges")
item("S8", "tree GPU/CPU total, cube_edges 1M", "6-30", x, f"logs/S5_cube_edges_1048576 ({cb[1]})", inr(x, 6, 30))
gb = min(gpu_tree(1048576), gpu_tree(1048576, ent="gpu_bh_sorted"))
x = cpu_tree_best(1048576)[0] / gb
item("S9", "GPU best / CPU best over any kernel, Plummer 1M", "10-24", x, "logs/S5_plummer_1048576, C5_1048576",
     inr(x, 10, 24))

# ---------------------------------------------------------------- §4 hybrids
def gpu_wall_best(c):
    return min((v(c, e), e) for e in ents(c, "gpu_"))


for n, lo, hi in ((65536, .97, 1.02), (131072, .96, 1.0), (262144, .955, .995), (16384, 1.0, 1.10)):
    c = f"H1_{n}"
    g = gpu_wall_best(c)
    x = v(c, "hyb_ap_dyn") / g[0]
    iid = "H2" if n == 16384 else f"H1-{n}"
    item(iid, f"hyb_allpairs dynamic / GPU best all-pairs (--wall) at {n}", f"{lo}-{hi}", x, f"logs/{c} (vs {g[1]})",
         inr(x, lo, hi), note=f"vs tuned b128 wall: {v(c, 'hyb_ap_dyn') / v(c, 'gpu_tuned_b128_wall'):.4f}")
x = v("H1_262144", "hyb_ap_dyn", "last")
item("H3", "final GPU share f (dynamic) at 262144", "0.955-0.985", x, "logs/H1_262144", inr(x, .955, .985))
x = v("H4_262144", "hyb_ap_static") / v("H4_262144", "hyb_ap_dyn")
item("H4", "static f (session rates) / dynamic, time, 262144", "0.98-1.02", x, "logs/H4_262144", inr(x, .98, 1.02))

H5c = lambda ic, n, th="0.5": f"H5_{ic}_{n}_th{th}"
for n, lo, hi in ((1048576, .95, 1.02), (262144, .98, 1.10), (65536, 1.0, 1.30)):
    c = H5c("plummer", n)
    x = v(c, "hyb_bh_dyn") / v(c, "gpu_bh_warp_wall")
    item(f"H5-{n}", f"hyb_bh dynamic / bh_warp (--wall), Plummer {n}", f"{lo}-{hi}", x, f"logs/{c}", inr(x, lo, hi))
c = H5c("plummer", 1048576, "0.3")
x = v(c, "hyb_bh_dyn") / v(c, "gpu_bh_warp_wall")
item("H6", "hyb_bh dynamic / bh_warp, Plummer 1M theta 0.3", "0.93-1.00", x, f"logs/{c}", inr(x, .93, 1.0))
c = H5c("cube_edges", 1048576)
x = v(c, "hyb_bh_dyn") / v(c, "gpu_bh_warp_wall")
item("H7", "hyb_bh dynamic / bh_warp, cube_edges 1M", "0.98-1.05", x, f"logs/{c}", inr(x, .98, 1.05))
x = v(c, "hyb_bh_f095") / v(c, "gpu_bh_warp_wall")
item("H8", "hyb_bh static f 0.95 / bh_warp, cube_edges 1M", "2.0-5.0", x, f"logs/{c}", inr(x, 2.0, 5.0))
x = v(c, "hyb_bh_f095", "xfer")
item("H9", "tree D2H (hybrid transfer per launch, 56 B/body) at 1M (ms)", "2.0-3.5", x, f"logs/{c} (hyb_bh_f095)",
     inr(x, 2.0, 3.5))
lagK = {}
for ic, n, lo, hi, iid in (("plummer", 65536, .55, .95, "H10-65536"), ("plummer", 262144, .82, .97, "H10-262144"),
                           ("plummer", 1048576, .92, .99, "H10-1048576"), ("cube_edges", 262144, .50, .90, "H11-262144"),
                           ("cube_edges", 1048576, .55, .85, "H11-1048576")):
    c = H5c(ic, n)
    x = v(c, "hyb_phase") / v(c, "gpu_bh_warp_wall")
    item(iid, f"hyb_phase / bh_warp (--wall), {ic} {n}", f"{lo}-{hi}", x, f"logs/{c}", inr(x, lo, hi))
for ic, n in (("plummer", 65536), ("plummer", 262144), ("plummer", 1048576), ("cube_edges", 262144),
              ("cube_edges", 1048576)):
    c = H5c(ic, n)
    lag = v(c, "hyb_phase", "lag")
    K = 4 if abs(lag - 4) <= abs(lag - 16) else 16
    x = v(c, "hyb_phase") / v(c, f"gpu_refit_k{K}_wall")
    item(f"H12-{ic}-{n}", f"hyb_phase / bh_warp_refit K (nearer the mean lag {lag:.2f}) = {K}, {ic} {n}", "0.97-1.05", x,
         f"logs/{c}", inr(x, .97, 1.05))
c = H5c("cube_edges", 1048576)
x = v(c, "gpu_refit_k16_wall") / v(c, "gpu_bh_warp_wall")
item("H13", "bh_warp_refit K16 / bh_warp, cube_edges 1M", "0.55-0.85", x, f"logs/{c}", inr(x, .55, .85))
for ic, n, lo, hi in (("plummer", 1048576, 2, 6), ("cube_edges", 1048576, 12, 50), ("plummer", 65536, 2, 8)):
    c = H5c(ic, n)
    x = v(c, "hyb_phase", "lag")
    item(f"H14-{ic}-{n}", f"mean topology lag (launches), {ic} {n}", f"{lo}-{hi}", x, f"logs/{c}", inr(x, lo, hi))
for n, lo, hi in ((1048576, 2.0, 5.0), (65536, 1.5, 4.0)):
    c = H5c("plummer", n)
    x = v(c, "hyb_phase_sync") / v(c, "gpu_bh_warp_wall")
    item(f"H15-{n}", f"hyb_phase --hyb-sync / bh_warp, Plummer {n}", f"{lo}-{hi}", x, f"logs/{c}", inr(x, lo, hi))

# H16 verdict
worst, notes = [], []
for c in [f"H1_{n}" for n in (16384, 65536, 131072, 262144)] + ["H4_262144"] + \
         [p.name for p in L.iterdir() if p.name.startswith("H5_")]:
    g = min((v(c, e), e) for e in ents(c, "gpu_"))
    for h in ents(c, "hyb_"):
        if h == "hyb_phase_sync" or h == "hyb_bh_f095" or h == "hyb_ap_static":
            continue
        r = v(c, h) / g[0]
        worst.append((r, c, h, g[1]))
    if "hyb_phase" in ents(c, "hyb_"):
        hp, bw = v(c, "hyb_phase"), v(c, "gpu_bh_warp_wall")
        if hp < bw:
            rf = min(v(c, "gpu_refit_k4_wall"), v(c, "gpu_refit_k16_wall"))
            notes.append((rf / hp, c))
worst.sort()
best_h = worst[0]
part1 = best_h[0] >= 0.95
part2 = all(r <= 1.05 for r, _ in notes)
item("H16", "no hybrid beats the best GPU-only config of its cell by >5%; every hyb_phase gain over bh_warp "
     "reproduced within 5% by bh_warp_refit", "holds",
     f"best hybrid/GPU-best {best_h[0]:.4f} ({best_h[2]} in {best_h[1]} vs {best_h[3]}); "
     f"refit/hyb_phase max {max((r for r, _ in notes), default=float('nan')):.4f}",
     "logs/H*", part1 and part2)

# ---------------------------------------------------------------- §5 PCIe (cpu_bh G32 leaf16 host line)
def bw(n, key):
    c = {1048576: "C5_1048576", 262144: "C5_262144", 65536: "C5_65536"}[n]
    return 16 * n / (v(c, "bh_g32_l16", key) * 1e-3) / 1e9, c


for n in (1048576, 262144):
    x, c = bw(n, "d2h")
    item(f"T1-{n}", f"D2H effective bandwidth at {n}, 16 B/body (GB/s)", "18-27", x, f"logs/{c} (bh_g32_l16 host:)",
         inr(x, 18, 27))
for n in (1048576, 262144):
    x, c = bw(n, "h2d")
    item(f"T2-{n}", f"H2D effective bandwidth at {n}, 16 B/body (GB/s)", "18-27", x, f"logs/{c} (bh_g32_l16 host:)",
         inr(x, 18, 27))
x, c = bw(65536, "d2h")
item("T3", "D2H effective bandwidth at 65536 (GB/s)", "6-22", x, f"logs/{c} (bh_g32_l16 host:)", inr(x, 6, 22))

# ---------------------------------------------------------------- §6 lag gates
lg, lr, lw = [], [], []
for ic in ("plummer", "galaxies"):
    for e in ("hyb_phase_lag", "refit_k16_lag"):
        d = E.get((f"L1_{ic}", e), {}).get("p1", {})
        lg.append(d.get("lg1") == "PASS" and d.get("lg2") == "PASS" and d.get("ltree") == "ok")
        lr.append(d.get("lrms", math.nan))
        lw.append(d.get("lwork", math.nan))
item("L1", "LG1 and LG2, 4 runs", "all PASS", f"{sum(lg)}/4 PASS", "logs/L1_*", all(lg) and len(lg) == 4)
item("L2", "stale/fresh rms ratio, 4 runs", "0.97-1.05", " ".join(f"{x:.4f}" for x in lr), "logs/L1_*",
     all(inr(x, .97, 1.05) for x in lr))
item("L3", "stale/fresh interactions per body, 4 runs", "1.000-1.030", " ".join(f"{x:.4f}" for x in lw), "logs/L1_*",
     all(inr(x, 1.0, 1.03) for x in lw))

with open(R / "scorecard.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(SC[0].keys()))
    w.writeheader()
    w.writerows(SC)

# ---------------------------------------------------------------- entries.csv + tables.md
keys = ["med", "compute", "build", "trav", "d2h", "h2d", "busy", "itemsP", "share", "last", "gpart", "cpart", "xfer",
        "topo", "lag", "topobuild", "ipb"]
bad = []
with open(R / "entries.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cell", "entry", "passes", "spread_med"] + keys + ["gates", "same", "exact"])
    for (c, e), d in sorted(E.items()):
        w.writerow([c, e, "+".join(sorted(d)), f"{spread(c, e):.4f}"] +
                   [("" if math.isnan(v(c, e, k)) else f"{v(c, e, k):.5g}") for k in keys] +
                   ["/".join(str(x) for x in s(c, e, "gates")), "/".join(str(x) for x in s(c, e, "same")),
                    "/".join(str(x) for x in s(c, e, "exact"))])
        for p, x in d.items():
            if "med" not in x or (x.get("gates") not in (None, "PASS")) or x.get("same") == "FAIL":
                bad.append(f"{c}/{p}/{e}: med={'med' in x} gates={x.get('gates')} same={x.get('same')}")

T = []
T.append("## Scorecard\n")
hits = sum(r["verdict"] == "HIT" for r in SC)
T.append(f"{hits} HIT / {len(SC) - hits} MISS of {len(SC)} rows.\n")
T.append("| ID | Prediction | Range | Measured | Verdict |\n|---|---|---|---|---|")
for r in SC:
    T.append(f"| {r['id']} | {r['prediction']} | {r['range']} | {r['measured']} | {r['verdict']} |")

T.append("\n## Hybrid vs GPU-resident trees (wall clock, ms per step, mean of 2 pass medians)\n")
T.append("| cell | bh_warp | bh_sorted (leaf 32 b256) | hyb_bh dyn | hyb_bh f / last f | hyb_phase | lag | refit K4 | "
         "refit K16 | hyb_phase sync | hyb_bh/bh_warp | hyb_phase/bh_warp | hyb_phase/best GPU-only |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(p.name for p in L.iterdir() if p.name.startswith("H5_")):
    bw_, bs_ = v(c, "gpu_bh_warp_wall"), v(c, "gpu_bh_sorted_wall")
    hb, hp = v(c, "hyb_bh_dyn"), v(c, "hyb_phase")
    k4, k16, hs = v(c, "gpu_refit_k4_wall"), v(c, "gpu_refit_k16_wall"), v(c, "hyb_phase_sync")
    gbest = min(x for x in (bw_, bs_, k4, k16) if not math.isnan(x))
    f_ = lambda x: "-" if math.isnan(x) else f"{x:.3f}"
    T.append(f"| {c} | {f_(bw_)} | {f_(bs_)} | {f_(hb)} | {f_(v(c, 'hyb_bh_dyn', 'share'))} / "
             f"{f_(v(c, 'hyb_bh_dyn', 'last'))} | {f_(hp)} | {f_(v(c, 'hyb_phase', 'lag'))} | {f_(k4)} | {f_(k16)} | "
             f"{f_(hs)} | {f_(hb / bw_)} | {f_(hp / bw_)} | {f_(hp / gbest)} |")
    if c.startswith("H5_cube_edges_1048576"):
        T.append(f"| (same) hyb_bh f 0.95 | | | {f_(v(c, 'hyb_bh_f095'))} | xfer {f_(v(c, 'hyb_bh_f095', 'xfer'))} ms, "
                 f"CPU part {f_(v(c, 'hyb_bh_f095', 'cpart'))} ms | | | | | | {f_(v(c, 'hyb_bh_f095') / bw_)} | | |")

T.append("\n## Hybrid all-pairs (wall clock, ms)\n")
T.append("| N | tuned b128 | multi b256 | hyb_allpairs dyn | share (last) | GPU part | CPU part | transfer | hyb/GPU best |\n"
         "|---|---|---|---|---|---|---|---|---|")
for n in (16384, 65536, 131072, 262144):
    c = f"H1_{n}"
    g = gpu_wall_best(c)[0]
    T.append(f"| {n} | {v(c, 'gpu_tuned_b128_wall'):.4f} | {v(c, 'gpu_multi_b256_wall'):.4f} | {v(c, 'hyb_ap_dyn'):.4f} | "
             f"{v(c, 'hyb_ap_dyn', 'share'):.4f} ({v(c, 'hyb_ap_dyn', 'last'):.4f}) | {v(c, 'hyb_ap_dyn', 'gpart'):.4f} | "
             f"{v(c, 'hyb_ap_dyn', 'cpart'):.4f} | {v(c, 'hyb_ap_dyn', 'xfer'):.4f} | {v(c, 'hyb_ap_dyn') / g:.4f} |")
c = "H4_262144"
T.append(f"| 262144 (H4) | {v(c, 'gpu_tuned_b128_wall'):.4f} | | dyn {v(c, 'hyb_ap_dyn'):.4f} / static "
         f"{v(c, 'hyb_ap_static'):.4f} | static f {v(c, 'hyb_ap_static', 'share'):.4f} | | | | |")

T.append("\n## P/E cores: cpu_allpairs 65536 (IV 2, tile 2048), compute only\n")
T.append("| set | sched | workers | G int/s | busy max/mean | items on P | mean busy P / E (ms) |\n|---|---|---|---|---|---|---|")
for st in ("all", "p", "p1", "e", "cores"):
    for sch in ("static", "dynamic", "weighted"):
        e = f"ap_{st}_{sch}"
        T.append(f"| {st} | {sch} | | {65536**2 / (v('E1_65536', e, 'compute') * 1e-3) / 1e9:.2f} | "
                 f"{v('E1_65536', e, 'busy'):.3f} | {v('E1_65536', e, 'itemsP'):.1f}% | "
                 f"{v('E1_65536', e, 'busyP'):.2f} / {v('E1_65536', e, 'busyE'):.2f} |")
T.append("\n## P/E cores: cpu_bh Plummer 1M (G 32, leaf 16)\n")
T.append("| set | sched | build ms | traverse ms | busy max/mean | items on P |\n|---|---|---|---|---|---|")
for st in ("all", "p", "e"):
    for sch in ("static", "dynamic", "weighted"):
        e = f"bh_{st}_{sch}"
        T.append(f"| {st} | {sch} | {v('E9_1M', e, 'build'):.3f} | {v('E9_1M', e, 'trav'):.3f} | {v('E9_1M', e, 'busy'):.3f} | "
                 f"{v('E9_1M', e, 'itemsP'):.1f}% |")

T.append("\n## CPU vs GPU all-pairs (ms; GPU event-timed best, CPU compute-only best; wall GPU at N <= 4096 not scored)\n")
T.append("| N | GPU best (event) | GPU best (wall) | CPU best | CPU/GPU |\n|---|---|---|---|---|")
for n in (64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144):
    gw = gpu_ap_wall(n) if n <= 4096 else math.nan
    T.append(f"| {n} | {gpu_ap(n):.5f} | {'-' if math.isnan(gw) else f'{gw:.5f}'} | {cpu_ap(n):.5f} | {cpu_ap(n) / gpu_ap(n):.2f} |")
T.append("\n## CPU vs GPU trees (ms; GPU event-timed, CPU compute only)\n")
T.append("| cell | bh_warp total | build | traverse | bh_sorted | cpu_bh best total (config) | CPU/GPU |\n|---|---|---|---|---|---|---|")
for ic, n in (("plummer", 65536), ("plummer", 262144), ("plummer", 1048576), ("cube_edges", 1048576)):
    cb = cpu_tree_best(n, ic)
    T.append(f"| {ic} {n} | {gpu_tree(n, ic):.4f} | {gpu_tree(n, ic, 'build'):.4f} | {gpu_tree(n, ic, 'trav'):.4f} | "
             f"{gpu_tree(n, ic, ent='gpu_bh_sorted'):.4f} | {cb[0]:.3f} ({cb[1]}) | {cb[0] / gpu_tree(n, ic):.2f} |")
T.append("\n## CPU crossover (C10, compute only, ms)\n")
T.append("| N | cpu_allpairs best | cpu_bh best |\n|---|---|---|")
for n, a_, b_ in cross:
    T.append(f"| {n} | {a_:.4f} | {b_:.4f} |")
T.append("\n## PCIe, 16 B/body pinned (cpu_bh G32 leaf 16 host: line)\n")
T.append("| N | D2H ms | H2D ms | D2H GB/s | H2D GB/s |\n|---|---|---|---|---|")
for n in (65536, 262144, 1048576):
    c = {1048576: "C5_1048576", 262144: "C5_262144", 65536: "C5_65536"}[n]
    T.append(f"| {n} | {v(c, 'bh_g32_l16', 'd2h'):.4f} | {v(c, 'bh_g32_l16', 'h2d'):.4f} | {bw(n, 'd2h')[0]:.2f} | "
             f"{bw(n, 'h2d')[0]:.2f} |")

sp = sorted(((spread(c, e), c, e) for (c, e) in E if not math.isnan(spread(c, e))), reverse=True)
T.append("\n## Pass spread |p1 - p2| / mean (largest 10)\n")
for x, c, e in sp[:10]:
    T.append(f"- {c}/{e}: {x * 100:.1f}%")
T.append(f"\nEntries with a missing median, a non-PASS gate or a same-work FAIL: {len(bad)}")
T += [f"- {b}" for b in bad[:40]]
(R / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
print(f"{hits} HIT / {len(SC) - hits} MISS of {len(SC)}; entries {len(E)}; bad {len(bad)}")
