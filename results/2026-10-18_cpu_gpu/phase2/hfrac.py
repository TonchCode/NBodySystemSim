"""H4: the static GPU share from this session's standalone rates (pre-registration §4, H4).

f = (T_x + t_c) / (t_g + t_c), with (all at N = 262144, mean of the two pass medians):
  t_g = allpairs_tuned b128 full --wall (cell H1_262144),
  t_c = cpu_allpairs IV 2 tile 2048, 20 LPs dynamic, compute phase (cell C1_262144; the CPU part's config),
  T_x = the hybrid's transfer per launch (cell H1_262144, hyb_allpairs dynamic, `hybrid:` line).
Prints f with 4 decimals. Usage: python hfrac.py <phase2 dir>
"""
import re
import sys
from pathlib import Path

R = Path(sys.argv[1])


def mean_passes(cell, name, rx):
    v = []
    for p in ("p1", "p2"):
        t = (R / "logs" / cell / p / f"{name}.txt").read_text(errors="replace")
        v.append(float(re.search(rx, t).group(1)))
    return sum(v) / len(v)


tg = mean_passes("H1_262144", "gpu_tuned_b128_wall", r"median=([0-9.]+) ms")
tx = mean_passes("H1_262144", "hyb_ap_dyn", r"transfer ([0-9.]+) ms")
tc = mean_passes("C1_262144", "ap_iv2_t2048", r"compute ([0-9.]+)")
f = (tx + tc) / (tg + tc)
sys.stderr.write(f"t_g {tg:.4f} ms, t_c {tc:.4f} ms, T_x {tx:.4f} ms -> f {f:.4f}\n")
print(f"{f:.4f}")
