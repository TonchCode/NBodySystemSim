# 2026-10-17 — Tree round 1: R3 #4 (naive Barnes-Hut) and R3 #5 (space-filling-curve order)

Predictions: `docs/predictions/2026-10-17-tree-round1.md`, committed as **581fc47** before these
runs. Afterwards only a dated post-registration note was appended (21 lines added, 0 removed,
verified with `git diff --numstat`). Design: `docs/03-design.md` §10–§16. The cuda-architect
wrote the content; the harness blocked the agent from writing the file, so the main session saved
it. The analysis is reproducible with `tools/tree_analysis.py`.

Raw data in this folder:
- `logs/` (+ `index.txt` with exit codes)
- `setA.csv`, `setC.csv`, `setD_tree.csv`, `setD_ap.csv`
- `warps/`: per-warp counters
- `timeline/`: plus Gantt SVGs in `plots/`
- `tree_metrics.csv`, `timeline_slots.csv`, `gpu_clock_log.csv`
- `run_phase2.sh`: the exact commands

**Environment:**
- Driver 617.42, cl 19.51.36248, nvcc 13.4 V13.4.92.
- **Power limit 272 W** (default 320, unchanged). **Idle 0% before and after**, 1031 MiB.
- Under load: SM median 2730 MHz (2250–2775), power up to 273.9 W. All-pairs hits the cap; tree
  traversal (2.75 GHz) does not.
- `--selftest` on the measured binary: PASS (2530 lattice cases + 110 tree checks).
- **190 runs, all exit 0, no gate FAIL, no stack overflow.**

Settings: leaf 16, block 128, `--warmup-ms 300 --runs 11`, batch 5 for N ≤ 65536 (else 1). θ is
the bmax-θ (≈ 0.87 × classic θ). Times are medians in ms.

> **Reading the bench output:** for tree kernels, the headline `kernel=` line's "G interactions/s"
> is **N²-equivalent** (all-pairs work ÷ time), not the tree's actual interaction rate. The real
> traversal rate is on the `work:` line and in the "trav G int/s" column below. (Flagged by the
> main session; label fix pending.)

## 1. Main grid (set A), θ = 0.5

| IC | N | Variant | Total | Build | Traverse | Order | Int/body | Pops/body | Per-warp max/mean | SIMT eff int / pop | Warp-cost max/mean | Trav G int/s | rms / p99 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plummer | 65536 | naive #4 | 3.070 | 0.237 | 2.862 | 0.001 | 915.5 | 1166.4 | 1.197 | 0.092 / 0.752 | 1.08 | 21.0 | 6.67e-3 / 1.77e-2 |
| | | sorted #5 | 1.969 | 0.249 | 1.757 | 0.001 | 〃 | 〃 | 1.079 | 0.149 / 0.938 | 1.20 | 34.1 | identical |
| | | hilbert | 2.148 | 0.230 | 1.801 | 0.133 | 〃 | 〃 | 1.074 | 0.151 / 0.946 | 1.21 | 33.3 | identical |
| | 262144 | naive | 16.545 | 0.301 | 16.223 | 0.001 | 976.2 | 1424.2 | 1.197 | 0.110 / 0.794 | 1.11 | 15.8 | 6.55e-3 / 1.74e-2 |
| | | sorted | 8.918 | 0.404 | 8.475 | 0.001 | 〃 | 〃 | 1.063 | 0.184 / 0.963 | 1.26 | 30.2 | |
| | | hilbert | 9.278 | 0.303 | 8.852 | 0.139 | 〃 | 〃 | 1.060 | 0.184 / 0.968 | 1.26 | 28.9 | |
| | 1048576 | naive | 73.660 | 0.621 | 73.046 | 0.002 | 1062.6 | 1611.8 | 1.186 | 0.118 / 0.816 | 1.07 | 15.3 | 6.60e-3 / 1.74e-2 |
| | | sorted | 39.169 | 0.615 | 38.552 | 0.001 | 〃 | 〃 | 1.053 | 0.201 / 0.976 | 1.20 | 28.9 | |
| | | hilbert | 40.536 | 0.672 | 39.702 | 0.239 | 〃 | 〃 | 1.051 | 0.201 / 0.979 | 1.21 | 28.1 | |
| collision | 65536 | naive / sorted / hilbert | 2.885 / 1.766 / 1.845 | ≈0.23 | 2.664 / 1.568 / 1.499 | | 975.3 | 1065.9 | 1.168 / 1.087 / 1.081 | int 0.087 / 0.135 / 0.136 | 1.07 / 1.15 / 1.15 | 24.0 / 40.8 / 42.6 | 8.45e-3 / 2.09e-2 |
| | 262144 | | 16.051 / 7.859 / 8.038 | ≈0.28 | 15.762 / 7.456 / 7.584 | | 975.8 | 1376.8 | 1.207 / 1.067 / 1.064 | 0.112 / 0.174 / 0.174 | 1.09 / 1.23 / 1.23 | 16.2 / 34.3 / 33.7 | 8.61e-3 / 2.02e-2 |
| | 1048576 | | 71.039 / 36.188 / 36.888 | ≈0.61 | 70.398 / 35.553 / 36.102 | | 1047.5 | 1579.0 | 1.188 / 1.055 / 1.053 | 0.126 / 0.197 / 0.197 | 1.12 / 1.26 / 1.26 | 15.6 / 30.9 / 30.4 | 9.13e-3 / 1.95e-2 |
| core | 65536 | | 4.223 / 2.651 / 2.813 | ≈0.23 | 3.983 / 2.422 / 2.466 | | 1077.7 | 1404.0 | **1.619** / 1.075 / 1.070 | 0.091 / 0.150 / 0.152 | 1.06 / **1.59** / 1.60 | 17.7 / 29.2 / 28.6 | 1.01e-2 / 2.84e-2 |
| | 262144 | | 21.002 / 11.220 / 11.394 | ≈0.30 | 20.748 / 10.839 / 11.001 | | 1119.1 | 1693.1 | 1.541 / 1.057 / 1.054 | 0.110 / 0.196 / 0.197 | 1.08 / 1.58 / 1.58 | 14.1 / 27.1 / 26.7 | 1.01e-2 / 2.86e-2 |
| | 1048576 | | 90.948 / 47.820 / 49.065 | ≈0.62 | 90.303 / 47.252 / 48.230 | | 1214.4 | 1909.3 | 1.510 / 1.048 / 1.046 | 0.116 / 0.215 / 0.215 | 1.07 / 1.55 / 1.55 | 14.1 / 26.9 / 26.4 | 1.02e-2 / 2.84e-2 |

Build phases at 1M (keys / sort / tree): 0.05–0.07 / 0.22–0.25 / 0.33 ms, so at 1M the tree phase
is the largest. At 65536 and 262144 the sort is the largest.

## 2. Headline findings

1. **#5/#4 traversal speedup ≈ 1.6–2.1×.**
   - Plummer 1.63 / 1.91 / 1.89, collision 1.70 / 2.11 / 1.98, core 1.64 / 1.91 / 1.91.
   - It is flat above 262144 and nearly IC-independent.
   - Burtscher & Pingali's 2011 "order of magnitude" from sorting [burtscher2011barneshut] does
     **not** carry over to Ada.
2. **Sorting fixes uneven work between lanes, not divergence.**
   - Per-warp max/mean drops from 1.17–1.62 to 1.05–1.09.
   - Pop SIMT efficiency rises from 0.55–0.82 to 0.92–0.98.
   - At the interaction sites only 3–7 of 32 lanes are active in both variants (0.09–0.22). The
     cause is the leaf-bucket loop.
3. **The tree kernel runs at about 1/20 of its own coherent speed.**
   - In the θ = 0 control all lanes walk the identical path: 552 G int/s, SIMT efficiency 1.000,
     #4 = #5 within 0.3%.
   - At θ = 0.5 the same kernel does 14–43 G/s.
4. **Sorting moves imbalance between blocks; at large N the scheduler absorbs it.**
   - #5's warp-cost max/mean is 1.20–1.60, against 1.06–1.12 for #4.
   - In single-wave launches it shows as an SM tail: #5 core at 65536 has SM eff 0.810 and a 17.7%
     tail, against 0.945 and 3.5% for #4.
   - At 1M both reach 0.987–0.995.
5. **Hilbert ≈ Morton.** Per-warp max/mean is lower by only 0.2–0.6%, traversal is 1.5–4.4% slower
   in 8 of 9 cells, and the order phase adds 0.13–0.24 ms.
6. **Interactions per body are flat in N; node pops grow by about one tree level per doubling.**
   Pops per body go from 582 to 1612, so pops per interaction go from 0.61 to 1.52.
7. **Crossover vs the best all-pairs: N ≈ 39,700 (sorted), ≈ 59,600 (naive).**

## 3. Crossover and speedup vs the best all-pairs (set D, rule 8)

"Best all-pairs" is the fastest of tuned b128 u0 and multi b ∈ {128..1024} × q ∈ {1..16} at the
same N.

| N | Best all-pairs (config, ms, G/s) | bh_naive | bh_sorted | AP/naive | AP/sorted |
|---|---|---|---|---|---|
| 8192 | multi b1024 q8, 0.0657, 1021 | 0.867 | 0.653 | 0.08 | 0.10 |
| 16384 | multi b256 q8, 0.2387, 1125 | 1.135 | 0.862 | 0.21 | 0.28 |
| 24576 | multi b512 q8, 0.4999, 1208 | 1.278 | 0.986 | 0.39 | 0.51 |
| 32768 | multi b256 q8, 0.8903, 1206 | 1.570 | 1.074 | 0.57 | 0.83 |
| 49152 | tuned b128 u0, 1.9122, 1263 | 2.287 | 1.478 | 0.84 | **1.29** |
| 65536 | multi b256 q4, 3.3978, 1264 | 3.071 | 1.964 | **1.11** | 1.73 |
| 262144 | tuned b128 u0, 50.98, 1348 | 16.545 | 8.918 | 3.08 | 5.72 |
| 524288 | tuned b128 u0, 205.49, 1338 | — | — | — | — |
| 1048576 | [D] 4 × 205.49 = 822 (not run, TDR) | 73.66 | 39.17 | 11.2 [D] | 21.0 [D] |

## 4. Scorecard vs 581fc47: **91 HIT / 32 MISS (123 items)**

- **What held:** everything relative. IC ratios, imbalance metrics (P-T5/T7), error and work vs θ
  (P-T8; slopes 2.13 and −2.52), the crossover (P-T4a/b), the timeline structure, the θ = 0
  control, and "Burtscher's 10× fails on Ada".
- **What missed:** absolute speed. Traversal is 2–4× slower than predicted, and the speedups over
  all-pairs are about 2× smaller. The cause: the issue-slot-only model left out memory-side
  incoherence, and the rate barely grows with occupancy (16 → 34 → 29 G/s at 4 / 26 / 48 warps
  per SM).
- **Other misses:**
  - Plummer interactions/body came in slightly below the predicted range.
  - Pops per interaction grew rather than staying at 0.61.
  - The sort is not the largest build phase at 1M.
  - The #5/#4 speedup is flat above 262k instead of growing.
  - Hilbert's advantage is smaller than predicted.
  - All imbalance misses are "better than predicted".

The item-by-item table is in the architect's report as transcribed in
`docs/03-design.md` §16 (and in `tools/tree_analysis.py` output).

## 5. Implications for #6 vs #7 (requested by the coordinator; not pre-registered)

**Method:**
- Work is counted in issue slots: W = 17.4·I + c_V·V, with c_V = 20 (sensitivity checked at
  10–40).
- The reference is the θ = 0 control (SIMT efficiency 1.000), at 1.16e13 lane-slots/s.
- F = E_lanes × E_mem × E_sm, with shares of −ln F:
  - (a) idle lanes;
  - (a′) memory incoherence (the residual);
  - (b+c) block-slot fill / tail, from the timelines.

| Run (θ = 0.5) | F | (a) E_lanes | (a′) E_mem | (b+c) E_sm [SM eff, tail] | Share a / a′ / b+c |
|---|---|---|---|---|---|
| naive plummer 65536 | 0.077 | 0.193 | 0.423 | 0.947 [0.953, 2.7%] | 64 / 34 / 2% |
| naive plummer 1M | 0.063 | 0.259 | 0.247 | 0.977 [0.995, 0.3%] | 49 / 50 / 1% |
| naive core 65536 | 0.066 | 0.182 | 0.390 | 0.937 [0.945, 3.5%] | 63 / 35 / 2% |
| naive core 1M | 0.059 | 0.243 | 0.249 | 0.979 [0.995, 0.4%] | 50 / 49 / 1% |
| sorted plummer 65536 | 0.126 | 0.298 | 0.499 | 0.847 [0.902, 8.3%] | 58 / 34 / 8% |
| sorted plummer 1M | 0.119 | 0.406 | 0.300 | 0.976 [0.987, 1.0%] | 42 / 57 / 1% |
| sorted core 65536 | 0.109 | 0.302 | 0.501 | **0.721** [0.810, 17.7%] | 54 / 31 / **15%** |
| sorted core 1M | 0.113 | 0.432 | 0.268 | 0.978 [0.991, 0.8%] | 38 / 60 / 1% |

**Conclusions:**
1. **(b+c), between-block imbalance and the tail, costs only ≈ 2.1–2.4% at 1M.** A
   persistent-thread queue (#6) can recover at most that. It matters only for single-wave clustered
   launches: sorted core at 65536 loses 28% of slot time.
2. **Intra-warp incoherence (a + a′) is 85–99% of the loss** (98–99% at 1M). Traversal reaches only
   6–13% of the kernel's coherent throughput.
3. **Next: #7 as warp-cooperative traversal, not work stealing (D-019).** Work stealing
   [chen2026gmatch] targets loop-level and between-warp imbalance, about 2.4% and 2% for #5. A
   warp-uniform walk [burtscher2011barneshut] or per-warp interaction lists attack (a) and (a′).
   The θ = 0 control bounds the gain at ≈ 8.4× over #5 at 1M, minus the union walk's extra work.
   That extra work is the quantity to pre-register.
4. This is consistent with maeda2026gtap (stealing ≈ 1.05× end to end) and chen2026gmatch (stealing
   is supplementary).

**Caveats:**
- The reference is θ = 0 at 65536, which has a different cache footprint from 1M.
- E_mem is a residual; without ncu it can't be split further.
- c_V is assumed (±15 pts on the a/a′ split, but a + a′ and b + c are unaffected).

## 6. Validation and gates

- **Tree gates:** all 49 runs at θ > 0 pass TG1/TG2, with zero stack overflows.
  - θ = 0.5: rms 6.5e-3–1.0e-2 against the 0.03 gate; p99 1.7–2.9e-2 against 0.12.
  - θ = 1.0: rms 4.7e-2 against 0.12; p99 0.171 against 0.48.
- **θ = 0:** D-009 PASS. **All 132 all-pairs runs** pass D-009. #4, #5 and Hilbert errors are bitwise
  identical.

## 7. Independent re-check (main session)

`bh_naive` vs `bh_sorted`, plummer, N = 262144, θ = 0.5, 7 runs:

| | Agent | Re-check |
|---|---|---|
| naive traverse | 16.223 | 16.222 ms |
| sorted traverse | 8.475 | 8.523 ms |
| int/body, pops/body | 976.2 / 1424.2 | 976.2 / 1424.2 |
| per-warp max/mean | 1.197 / 1.063 | 1.197 / 1.063 |
| SIMT eff int (naive / sorted) | 0.110 / 0.184 | 0.110 / 0.184 |
| gates | PASS | PASS |

## 8. Process notes

- The first launch of `run_phase2.sh` failed immediately (missing log dir; nothing ran) and was
  relaunched unchanged.
- No git command and no system or power changes by the agent.
