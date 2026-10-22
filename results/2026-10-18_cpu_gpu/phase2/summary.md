# R3 #8 phase 2: CPU vs GPU and the CPU+GPU split (measured 2026-10-22)

Written by the main session from the `cuda-architect` report (sub-agents cannot write report
files); every number below is in `tables.md` / `scorecard.csv` / `logs/`. Predictions:
`docs/predictions/2026-10-18-cpu-gpu.md`. Decisions: D-034, D-035, D-036.

## Session
- 2026-10-22 02:19–02:50, 40 cells, 438 runs, every timed cell run forward then reversed.
- Idle GPU ≤ 1 %, CPU ≤ 4 % between cells (`idle_log.csv`, `cpu_log.csv`); no cell flagged.
  Clock and power logged throughout (`gpu_clock_log.csv`).
- Binary: the clean phase-1 build (md5 5745a8c1, `binary_md5.txt`), not rebuilt; selftest PASS at
  start (`selftest.txt`). No settings changed.
- **Scorecard: 70 HIT / 12 MISS of 82 items** (`scorecard.csv`).

## 1. The hybrid against the GPU-only tree (ms per force evaluation; ratio vs `bh_warp`)
| Case | `bh_warp` | `bh_sorted` | `hyb_bh` (body split) | `hyb_phase` (phase split) | `bh_warp_refit` (GPU only) |
|---|---|---|---|---|---|
| Plummer 65,536 | 0.787 | 2.115 | 1.002 (1.27×) | 0.624 (0.79×) | 0.617 |
| Plummer 262,144 | 1.856 | 8.148 | 1.975 (1.06×) | 1.681 (0.91×) | 1.681 |
| Plummer 1M | 6.639 | 35.07 | 6.654 (1.00×) | 6.367 (0.96×) | 6.351 |
| Plummer 1M, θ = 0.3 | 19.89 | 171.8 | 19.72 (0.99×) | – | – |
| cube_edges 262,144 | 0.350 | 0.379 | 0.445 (1.27×) | 0.180 (0.51×) | 0.177 |
| cube_edges 1M | 0.981 | 0.991 | 1.023 (1.04×); 3.58× with a forced 0.95 GPU share | 0.736 (0.75×) | 0.737 |

- **No split helps.** The best gain of any hybrid over the best GPU-only configuration of its cell
  is 1.3 % (`hyb_allpairs`, 262,144). The all-pairs split is 24 % / 8.3 % / 2.4 % slower at
  16,384 / 65,536 / 131,072.
- **The phase split's gain is not the CPU's.** GPU-only `bh_warp_refit` matches `hyb_phase` within
  1.4 % in all five cells: the gain is from refitting an older tree instead of rebuilding.
- **The timing loop flatters the refit.** Positions are static during timing, so staleness costs
  nothing there. In 500-step runs a tree rebuilt every 16 steps does 4–10 % more interactions, which
  eats most of the gain at Plummer 1M.

## 2. CPU against GPU (best tuned on both sides; D-034)
- All-pairs: GPU 6.2× at 1,024, 22.8× at 4,096, 29–36× from 16,384 up. CPU ≈ 37.8 G int/s
  (AVX2 + FMA, rsqrt + Newton, 20 threads, dynamic).
- Trees: 8.7× / 14.3× / 18.3× on Plummer at 65,536 / 262,144 / 1M; 15.6× on cube_edges 1M.
- Compute-only crossover ≈ N = 64, fragile (passes differ 30–86 % at N ≤ 512); with copies counted
  the GPU wins at every N.

## 3. P-cores and E-cores (D-035)
- All-pairs 65,536: dynamic on 20 threads 37.8 G/s, +22 % over P-cores alone (31.0); 77.8 % of the
  work lands on P-cores. Static even split 21.4 G/s, *below* P-only: each E-core thread finishes
  ≈ 2.4× later. Weighted = dynamic. One E-core ≈ 0.21 of a P-core; hyper-threading adds 0.1 %.
- Tree traversal 1M: static 1.64× slower than dynamic; 65.6 % of the work on P-cores.

## 4. Transfers (pinned memory)
- Positions (16 B/body): device→host 0.050 / 0.175 / 0.646 ms at 65,536 / 262,144 / 1M
  (21–26 GB/s); host→device 0.070 / 0.226 / 0.731 ms (15–23 GB/s).
- Tree body split (56 B/body): 2.26 ms per step at 1M, 2.3× a whole cube_edges `bh_warp` step, so
  the body split cannot win there. All-pairs split: 0.07–0.28 ms against 3.4–51 ms steps, not the limit.

## 5. Misses (12)
Group size (G16 1.23× slower than G32; best at 1M G64 leaf 8); CPU accuracy at N = 128 (0.83× the
reference error, outside ±10 %; all 128 runs pass the gate); tree-build static/dynamic 1.13 (pred.
1.2–2.0); lag-gate accuracy (0.968) and work (+10.2 %) outside range; tree lag far longer than
predicted (6.5 / 98 / 154 launches), see §6. Full list in `scorecard.csv`.

## 6. Defect found, not fixed
`hyb_phase`'s background CPU tree build takes 29–307 ms, against ≈ 1–10 ms standalone; in one pass
the tree was never replaced, which explains the long lags. Suspected: slow wake-up of the CPU worker
pool [T2, untested]. It changes no verdict (D-036).
