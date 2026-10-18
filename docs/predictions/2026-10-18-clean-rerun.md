# Clean re-run predictions (bench-profiler), 2026-10-18: load-sensitive items of tree round 2 and the IC round

**Status: pre-registration.** CLAUDE.md rule 7: this file must be committed (separate commit) *before*
the timed runs in `results/2026-10-18_clean_rerun/`. Phase 2 does not start before that commit.
Commit IDs below are the current ones (`docs/HISTORY-NOTE.md`).

**What this is.** A **replication**, not a new prediction round. Tree round 2 (predictions **c6a12cc**)
ran under 20–22% uncontrolled idle GPU utilisation, the IC round (predictions **56505d8**) under
30–37%. Both summaries asked for the load-sensitive items to be re-run with the desktop apps closed.
This file fixes, before any timing: which items are re-run, how they are compared, what the load
model expects, and what would show the load explanation is wrong. Tags: [M] measured · [D] derived.

---

## 0. Already known before the timed runs (not predictions)

1. **Binary** [M]. `build/nbody_bench.exe`, SHA-256 `2b7896830db05c5977bde12e49be96e0c18ea14e4066d6c4a7656d4490dcfd43`
   (12,252,672 B, 2026-10-18 03:53). `cuobjdump -sass` of it has md5 `8e3d4a221b053bfce2267b46b7fd7677`,
   identical to `results/2026-10-18_viewer_round/bench_after.sass` (`sass_diff.txt`), and all 222
   functions match `bench_after_md5.txt` instruction for instruction. **Same binary as the viewer
   round.** `src/` is frozen for this run.
   - It is **not** the tree-round-2 binary: since then the per-thread stack went 64 → 96 (D-028; two
     SASS lines in the 18 per-thread-walk kernels, measured neutral 0.977–1.027 under load) and the
     gates became normalised (D-027). Work counts and the #7 walks are unchanged.
   - Viewer (V2 only): `build/src/viewer/nbody_viewer.exe`, SHA-256
     `73a55db5dcc095809a2199d65dd329894e40889ac4288de1721ad7ab7eb81aba`, the viewer-round build. It is
     **not** the IC-round viewer (host-side panel code changed; 222 of its 224 kernels are the
     IC-round ones). With `--screenshot` the panel is off and `--lb` defaults to off, as in the
     IC-round shot.
   - X (stack check) "old" binary: `scratchpad/nbody_bench_before.exe`, SHA-256
     `720996cd4d15be87a0580269a543ac82bc9fe7884c0bd8e3d5879dbd8e9811f5`, the same path the IC round's
     `logs/index.txt` used.
2. **Environment** [M, main session]. Idle utilisation 0–1%, SM 210 MHz, VRAM 1.5 GB. Light processes
   are still attached to the GPU with no measurable load (to be listed in `env_before.txt`).
   **Power limit 272 W** (default 320 W), not changed.
3. **Clean references that already exist** [M]. Tree round 1 ran at **0%** idle utilisation
   (`results/2026-10-17_tree_round1/env_*.txt`). Its #4/#5 times at leaf 16 b128, θ = 0.5 are clean
   anchors (binary differs only by the stack change):

   | kernel | Plummer 65536 / 262144 / 1M (ms) | core 65536 / 262144 / 1M (ms) |
   |---|---|---|
   | #4 bh_naive | 2.862 / 16.22 / 73.05 | 3.983 / 20.75 / 90.30 |
   | #5 bh_sorted | 1.757 / 8.475 / 38.55 | 2.422 / 10.84 / 47.25 |

   Tree round 2 measured the same cells at +2.5 to +4.8% (#5) and −1.4 to +7.5% (#4).
4. **What the old sessions show about the load** [M, from the committed results]:
   - Round 2: within-session drift up to 7% (#5 1M leaf 16 b128: 39.53 ms in set A, 42.18 ms in set C);
     adjacent-run ratios (A2) stable to ≤ 1.5%, except static on core (6.8%).
   - IC round: #5 Plummer 262144 leaf 16 b128 9.724 ms (+11.5% vs round 2, +14.7% vs round 1).
     But #5 at leaf 32 b256 was 7.881 ms, **faster** than round 2's set-C 8.335 ms. #7 Plummer 1M
     leaf 16 b128 in set B was 6.703 ms (+12% vs round 2's 5.982), while #7 at leaf 8 b256 in set A was
     6.017 ms (+0.8%). **The load was time-varying, and it was not confined to #5.**

---

## 1. Items re-run

Every command is copied from the original run scripts (`results/2026-10-17_tree_round2/run_phase2.sh`
and `results/2026-10-17_ic_round/run_phase2.sh`): same kernels, N, IC, θ, leaf/block, `--warmup-ms
300 --runs 11`, batch rule (5 for N ≤ 65536, else 1; set E always 5), validation and `--tree-stats`
as in the original. Only the order changes (§4).

### 1.1 Tree round 2 (sets A, A2, B, C, E, F, G; not D)

| group | items | sets |
|---|---|---|
| **#6 performance** | P2-P1 … P2-P11, P2-P13 (P6 is a counter, a check; P12 is (C)) | A, A2, B |
| **#7 speedups** | P2-W8 … P2-W12 | A |
| **Best vs best** | P2-L5, plus the reported best-vs-best T5/T7 and totals at 262144 and 1M | C (full 72-cell sweep) |
| **Crossover and θ** | P2-X1 … P2-X5 | E, F, G (X3 at 262144/1M also uses set A) |
| *secondary* (time-derived, free from the same runs) | C1–C5, W7, W13, W16, W17, W18, D1, D2, L1–L4 | A, B, C |
| *identity checks* (deterministic) | W1–W6, W14a–d, W15, P6, P12, L6, L7, C6 | must reproduce the old values |

Not re-run: **set D** (P2-O1 … O6). It was not named for the replication, and its misses are
dominated by the padding bug (post-registration note 3), not by load. The O verdicts stand as
recorded, still labelled with the round-2 load.

### 1.2 IC round (sets A, B, K, T, X, viewer V2)

| group | items | sets |
|---|---|---|
| **S** | S1–S5 (15 rows), S6, S7 | A (best-tuned, D-025) |
| **Q** | Q1–Q5, Q6 | A, T (galaxies 65536 timelines) |
| **K3** | #7/#5 speedup salpeter / equal, Plummer 262144 | K (all three spectra), B |
| **V2** | viewer bh_warp disk 1M real rate | viewer, after every bench run, nothing else on the GPU |
| **X** | stack 64 → 96, old/new binary, alternating | X (16 runs, as original) |
| *identity checks* | I, M, W, E, K1, K2, K4–K6, D, C rows | recomputed, must equal the old values |

Set B (leaf 16 b128, all ICs) is re-run in full: it carries K3's reference and gives same-config
cross-round anchors.

---

## 2. Comparison rule (fixed now)

1. **Same ranges.** Each item is re-scored against its **original committed range** (c6a12cc,
   56505d8), unchanged, with the original scoring rule: one ID = one row, a multi-cell ID is a HIT
   only if every cell is in range. No range is widened, narrowed or re-read.
2. **Value per cell.** Each kernel in a cell runs twice (pass 1 and pass 2, §4). The clean time is the
   **mean of the two pass medians** (each a median of 11). Ratios and rates come from those times. The
   set-C best config is chosen on the mean times. Pass 1 alone (closest to the original single pass)
   is reported as a sensitivity check. A verdict that differs between pass 1, pass 2 and the mean is
   marked **unstable**.
3. **Report for every item and cell:** old value (the committed `scorecard.csv`), clean value,
   **ratio clean/old**, old verdict, clean verdict.
4. **Canonical numbers.** If the session passes the environment checks (§5), the clean values replace
   the old ones as the canonical thesis numbers for these items. The old values stay in their folders,
   labelled. Deterministic quantities (work counts, depths, errors) are unaffected.
5. **Flagged cells** (idle utilisation > 5% between cells, §4) are re-run once after the load clears.
   The re-run is scored and both are kept. A flagged cell that cannot be re-run clean is reported but
   excluded from the verdict.

---

## 3. What the load model expects (numbers)

The load model as the two summaries stated it: *the background load inflated the per-thread walks
(#4, #5, #6) more than the coherent walks (#7, control), so #7/#5 speedups were inflated, and
adjacent-run ratios within one kernel family are load-proof.* Translated into expected clean/old ratios:

### 3.1 Tree round 2 (old load 20–22%)

| quantity | clean / old expected | central | consequence for verdicts |
|---|---|---|---|
| #4, #5, #6 times, set A | 0.93–0.99 | 0.97 | — |
| #7, #7-family, control times, set A | 0.97–1.02 | 0.995 | — |
| **#7/#5 speedups W8, W9, W10** | **0.93–1.00** | 0.97 (−3%) | all stay HIT (each ≥ 25% inside its range) |
| W11 #7 rate | 0.98–1.03 | 1.005 | stays HIT |
| #5 set C times | 0.88–0.99 | 0.95 (set C ran 7% above set A) | — |
| **L5 best-vs-best T5/T7 at 1M** (old 5.913) | 0.87–1.00 | 0.95 (≈ 5.6×) | stays HIT (L5 is relative to the leaf-16 cell, which moves alike) |
| **#6 ratios P1–P5, P7 (T_sibling / T_arm)** | **0.98–1.02** | 1.00 | **every #6 MISS stays MISS**: the nearest cell (P7 core 1M, 0.951 vs ≥ 0.98) needs +3% |
| P8, P11 (static arm) | 0.93–1.07 | 1.00 | P8, P11 stay MISS (P11: core 65536 static/queue 0.923) |
| P9, P10 (directional) | gains ±2% | — | P9 stays MISS (1M gap 3.6%), P10 stays HIT |
| P13, D2, W18 timeline fill / SM eff | ±0.02 absolute | 0 | P13 HIT; **D2 (0.108, bound 0.10) can go either way**, no prediction |
| X1 crossover (old 29,164) | ±10% (26,000–32,000) | 1.00 | stays HIT |
| X3 65536 (old 4.42) | 0.92–1.08 | 1.00 | stays HIT |
| X4, X5 | directional / θ = 0 ratio ±0.03 | — | stay HIT |
| C3, C4 T5/TC | 0.93–1.00 | 0.97 | stay HIT |
| D1 gap share f | ±0.03 | 0 | stays HIT |

### 3.2 IC round (old load 30–37%, time-varying)

| quantity | clean / old expected | central | consequence for verdicts |
|---|---|---|---|
| #5 (and #4, #6) times, set A | 0.85–1.02 | 0.95 | — |
| #7 times, set A | 0.95–1.02 | 0.99 | — |
| **S ratios (T5/T7), 3D ICs (disk, galaxies, uniform)** | **0.88–1.02** | **0.96 (−4%)** | S2-65536, S2-262144 stay MISS. **S1-262144** (4.229, low edge 4.0) flips to MISS if the ratio < 0.946 |
| S ratios, cube_shell | 0.88–1.02 | 0.96 | S3-65536 (1.594 vs 1.5) and S3-262144 (2.15 vs 2.0) are at risk below 0.94 / 0.93 |
| S ratios, cube_edges | **0.97–1.03** | 1.00 | sub-hypothesis H-coh, below. **S4-262144** (1.209, edge 1.2) is a coin flip; S4-1M stays MISS |
| S6 (#7 faster than #5 at N ≥ 262144, old min 1.04, cube_edges 1M) | — | — | stays HIT under H-coh. Under a uniform −4% it would fail |
| S7 (order at 1M) | — | — | stays HIT (disk/galaxies 5.6–6.0 vs cube_edges 1.04) |
| **Q ratios (T5/T6), every cell** | **0.97–1.03** | 1.00 | Q verdicts unchanged except the two within 2% of an edge: Q4-disk-1M (0.845 vs 0.84, could → MISS) and Q3 (0.848 vs 0.86, could → HIT) |
| **Q5 (#6 never beats #5, old max 0.967)** | — | — | **stays HIT: no IC where T5/T6 > 1.00** |
| Q6 (SM eff #6 ≥ #5 on galaxies 65536) | ±0.03 | — | stays HIT |
| K3 (ratio of two #7/#5 ratios, old 0.969) | 0.95–1.05 | 1.00 | stays HIT (range 0.88–1.12) |
| V2 viewer rate (old 318.2 at t = 3) | 0.97–1.12 | 1.03 (≈ 328) | stays HIT if < 350 (ratio < 1.10). Still a disputed reading (t = 3), with a viewer-binary confound (§0.1) |
| X stack, new/old per kernel × IC (old 0.977–1.027) | 0.98–1.02 each | 1.00 | D-028 "no measurable cost" confirmed |

**Sub-hypothesis H-coh** (stated now so it can fail): the load inflates a walk in proportion to its
scattered memory traffic. #5 on cube_edges already runs at a #7-like 220–230 G int/s, so it is
predicted as load-insensitive as #7. The cube_edges S ratios move by ≤ 3%, while the 3D-IC S ratios
move by 2–12%.

**The one-line expectation:** #5 runs 2–5% faster clean in the round-2 cells and 0–15% faster in the
IC-round cells, and #7 moves ±2%. #7/#5 speedups therefore fall by about 3% (round 2) and by 0–12%
(IC round, 3D ICs). **#6/#5 ratios move by ≤ 2% (≤ 3% in the IC round), so no #6 verdict is rescued
by the clean session.**

---

## 4. Run order and protocol

- `tools/capture_env.ps1 -Tag before`, then append the attached-process list (`nvidia-smi
  --query-compute-apps=pid,process_name,used_memory` and the `nvidia-smi` process table) and the
  binary hashes to `env_before.txt`. `--selftest` on the measured binary. `capture_env.ps1 -Tag after`
  at the end.
- **Clock/power log for the whole session:** `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,
  power.draw,power.limit,temperature.gpu,utilization.gpu,pstate,clocks_event_reasons.active
  --format=csv -lms 200` into `gpu_clock_log.csv`.
- **Interleaving.** A *cell* is one (set, IC, N[, leaf, block, θ, spectrum]) point. In each cell the
  kernels run in the original order (pass 1), then **in reverse order** (pass 2): A B C … C B A. A linear
  drift then biases no kernel's mean. Pass-1 and pass-2 outputs go to separate folders (`<set>/p1`,
  `<set>/p2`) with the original log names, so the existing analysis tools run on each pass unchanged.
  - Set C's cell is (N, leaf, block) with the three kernels interleaved. This reorders the original
    kernel-outer loop: best-vs-best is then taken from adjacent runs.
  - A2 and X were already alternating in the original. They run once, as written.
  - Round-2 set A's three `bh_warp_naive` runs (a separate loop in the original) are placed last in
    the Plummer cells.
  - Script: `results/2026-10-18_clean_rerun/run.sh`. Dry run (`DRYRUN=1`, no GPU work):
    **95 cells, 945 invocations** plus the selftest, listed in `dryrun_commands.txt`. That is 694
    tree-round-2 runs (set A 126, A2 60, B 72, C 144, E 276, F 12, G 4), 250 IC-round runs (A 144,
    B 72, K 12, T 6, X 16) and 1 viewer run. Expected session length is about 1 hour.
- **Idle check between cells:** 1.5 s after a cell, sample `utilization.gpu` 5 × 200 ms. If the median
  is > 5%, pause (re-sample every 15 s, up to 20 times), log it to `idle_log.csv`, flag the cell just
  finished in `flagged_cells.txt`, and re-run it once the load clears (§2.5).
- **Order of blocks:** selftest → tree round 2 A, A2, B, C, E, F, G → IC round A, B, K, T, X →
  viewer V2 (alone, after every bench process has exited; the viewer is not used during bench
  measurements) → env_after.
- **Power cap 272 W.** #4/#5 at 1M reached 270–271 W in both old sessions, so those cells are
  power-limited in every session alike. The cap is recorded, not changed. **Clock check:** the
  under-load SM median must be within ±2% of the old sessions' 2760 / 2745 MHz. Otherwise clock-normalised
  ratios are reported next to the raw ones.

## 5. What would show the load explanation is wrong

Any one of these, measured in a session whose idle checks all pass:

1. **Clean #4/#5 do not reproduce the clean round-1 anchors.** Fewer than 10 of the 12 leaf-16 b128
   cells (§0.3) are within ±3% of round 1, or the median |Δ| > 2%. That would mean the round-2 → round-1
   gap was not the load, or that the stack change was not neutral.
2. **Clean #5 is not faster than the old #5.** In round-2 set A, fewer than 5 of 6 #5 cells are ≥ 1%
   faster clean. Or the IC-round Plummer 262144 leaf 16 b128 #5 is not within ±4% of 8.475 ms (old 9.724).
3. **The speedups do not fall.** The median clean/old ratio of the W8–W9 cells or of the 3D-IC S cells
   is > 1.00 (#7/#5 *rises* when clean).
4. **#7 was inflated as much as #5.** The median clean/old of #7 times is < 0.97 in either round. The
   "#7 is load-proof" half of the model is then wrong, even if the speedups barely move. IC set B #7 1M
   (old 6.703 ms, +12%) is the test case.
5. **Same-family ratios were not load-proof.** The median |clean/old − 1| over the #6 cells (P1–P5, P7,
   Q) is > 2%, or any #6 verdict flips HIT ↔ MISS outside the two edge cells named in §3.2.
6. **The clean session still drifts.** The median |pass 2 / pass 1 − 1| of the same kernel in the same
   cell is > 2%, or the #5 1M leaf-16 b128 time differs between set A and set C by > 3% (old: 7%). The
   round-2 drift then was not (only) the desktop load.
7. **H-coh fails** if a cube_edges S ratio moves by > 3% while the 3D-IC ones do not. This tests the
   mechanism, not the verdicts.

If 1–6 hold, the load explanation is confirmed for both sessions, and the clean numbers are canonical.
