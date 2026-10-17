#!/usr/bin/env python3
"""Long-run energy drift summary from nbody_bench --energy-log CSVs (stdlib only).

    python tools/energy_analysis.py E1.csv E2.csv ... [--svg out.svg]

Per file: |dE/E| at the last row, the max |dE/E| over the run, the log-log slope alpha of the
running-max envelope over t in [1, t_end] (least squares on log env vs log t), and the virial
ratio 2K/|W| at start and end (also its min/max). --svg plots |dE/E| vs t (log-log), one line
per file.
"""
import argparse
import csv
import math
import sys


def load(path):
    meta = {}
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for ln in lines:
        if ln.startswith("#"):
            for tok in ln[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
    for r in csv.DictReader(l for l in lines if not l.startswith("#")):
        rows.append({k: float(v) for k, v in r.items()})
    return meta, rows


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--svg")
    a = ap.parse_args()
    series = []
    for p in a.files:
        meta, rows = load(p)
        env, m = [], 0.0
        for r in rows:
            m = max(m, abs(r["rel_drift"]))
            env.append(m)
        pts = [(math.log(r["time"]), math.log(e)) for r, e in zip(rows, env) if r["time"] >= 1.0 and e > 0]
        alpha = slope([x for x, _ in pts], [y for _, y in pts]) if len(pts) > 2 else float("nan")
        vir = [2 * r["kinetic"] / abs(r["potential"]) for r in rows]
        last = rows[-1]
        print(f"{p}: kernel={meta.get('kernel')} q={meta.get('q')} dt={meta.get('dt')} steps={int(last['step'])} "
              f"t_end={last['time']:.1f}")
        print(f"   dE/E(t_end) = {last['rel_drift']:+.3e}   max|dE/E| = {env[-1]:.3e}   envelope slope alpha "
              f"(t in [1, t_end]) = {alpha:.2f}")
        print(f"   virial 2K/|W|: start {vir[0]:.4f} end {vir[-1]:.4f} (min {min(vir):.4f}, max {max(vir):.4f})")
        series.append((p, rows))
    if a.svg:
        W, H, L, T = 700, 420, 70, 30
        tmin, tmax = 0.1, max(r["time"] for _, rows in series for r in rows)
        vals = [abs(r["rel_drift"]) for _, rows in series for r in rows if r["time"] >= tmin and r["rel_drift"] != 0]
        lo, hi = math.log10(min(vals)), math.log10(max(vals))
        lo, hi = math.floor(lo), math.ceil(hi)
        X = lambda t: L + (math.log10(t) - math.log10(tmin)) / (math.log10(tmax) - math.log10(tmin)) * (W - L - 20)
        Y = lambda v: T + (hi - math.log10(v)) / (hi - lo) * (H - T - 50)
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="{L}" y="18">|dE/E| vs time (N-body units), log-log; N = 4096 Plummer, eps = 0.01, KDK</text>']
        for e in range(lo, hi + 1):
            y = Y(10 ** e)
            out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - 20}" y2="{y:.1f}" stroke="#ddd"/>'
                       f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">1e{e}</text>')
        for t in [0.1, 1, 10, 100]:
            if t <= tmax:
                x = X(t)
                out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - 50}" stroke="#ddd"/>'
                           f'<text x="{x:.1f}" y="{H - 35}" text-anchor="middle">{t:g}</text>')
        for i, (p, rows) in enumerate(series):
            pts = [(X(r["time"]), Y(abs(r["rel_drift"]))) for r in rows if r["time"] >= tmin and r["rel_drift"] != 0]
            out.append(f'<polyline fill="none" stroke="{COLORS[i % 4]}" stroke-width="1.2" points="'
                       + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
            out.append(f'<text x="{L + 10}" y="{H - 12 - 14 * (len(series) - 1 - i)}" fill="{COLORS[i % 4]}">'
                       f'{p.replace(chr(92), "/").split("/")[-1]}</text>')
        out.append("</svg>")
        open(a.svg, "w").write("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
