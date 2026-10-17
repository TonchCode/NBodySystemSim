# 2026-10-17 — IC round: initial conditions as a load-balancing workload suite

Predictions: `docs/predictions/2026-10-17-ic-round.md`, committed as **56505d8** before these runs.
Afterwards only a dated post-registration note was appended (14 lines added, 0 removed, verified).
Design: `docs/03-design.md` §19–§20. Decisions: D-027..D-031. The cuda-architect wrote the content;
the harness blocked the agent from writing this file, so the main session saved it.

**Analysis:** `python tools/ic_analysis.py results/2026-10-17_ic_round --viewer-rate 318.2` writes
`derived.csv`, **`scorecard.csv`** and `analysis_output.txt`. Timelines: `timeline_metrics.csv`.

**Environment:**
- Driver 617.42, cl 19.51.36248, nvcc 13.4 V13.4.92.
- **Power limit 272 W** (default 320).
- Under load: SM median 2745 MHz (p5 2670), power median 189 W, max 270 W. The clock log covers
  only part of the session.
- `--selftest` on the measured binary: PASS.

> ⚠ **UNCONTROLLED BACKGROUND LOAD: 30% idle utilisation before, 37% after** (desktop apps).
> - Same-session plummer at 262144: #5 +11% vs round 2, #7 −1%. **#7/#5 ratios are inflated by up
>   to ~10%.**
> - **Load-sensitive:** S (speedups), Q (#6 ratios), K3, V2, X.
> - **Deterministic, load-proof:** work counts, tree depths, imbalance ratios, errors.

**Runs:** 133 scripted + selftest, all exit 0, 0 stack overflows. 12 exploratory (`explore/`, not
pre-registered). 17 presentation runs. θ = 0.5, 11 runs, 300 ms warm-up.

Best-tuned settings (D-025):

| Kernel | Leaf / block |
|---|---|
| #4 | leaf 16 b256 (leaf 32 at 1M) |
| #5, #6 | leaf 32 b256 |
| #7 | leaf 16 b128 (leaf 8 b256 at 1M) |

## 1. Main grid (set A), traversal ms, best-tuned

| IC | N | #4 | #5 | #6 queue | #7 | #4/#5 | #7/#5 | #5/#6 | int/body #4/#5/#7 | per-warp max/mean #4/#5 | #5 warp-cost max/mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plummer | 65536 | 2.861 | 1.831 | 2.182 | 0.574 | 1.56 | 3.19 | 0.84 | 916/1671/1695 | 1.20/1.09 | 1.24 |
| | 262144 | 17.41 | 7.881 | 9.404 | 1.598 | 2.21 | 4.93 | 0.84 | 976/1433/1748 | 1.20/1.08 | 1.20 |
| | 1M | 77.38 | 36.97 | 41.94 | 6.017 | 2.09 | 6.14 | 0.88 | 1415/1415/1455 | 1.19/1.08 | 1.25 |
| disk | 65536 | 1.780 | 1.284 | 1.328 | 0.425 | 1.39 | 3.02 | 0.97 | 589/937/1083 | 1.81/1.16 | 2.22 |
| | 262144 | 10.58 | 4.526 | 5.686 | 1.070 | 2.34 | 4.23 | 0.80 | 785/1376/1402 | 1.49/1.13 | 1.79 |
| | 1M | 60.24 | 26.63 | 31.52 | 4.474 | 2.26 | 5.95 | 0.85 | 1532/1532/1185 | 1.35/1.12 | 1.49 |
| galaxies | 65536 | 1.573 | 0.967 | 1.010 | 0.423 | 1.63 | 2.29 | 0.96 | 540/857/992 | 1.95/1.16 | 2.61 |
| | 262144 | 10.50 | 3.971 | 4.967 | 1.053 | 2.65 | 3.77 | 0.80 | 736/1224/1323 | 1.60/1.13 | 1.79 |
| | 1M | 59.48 | 25.00 | 29.49 | 4.495 | 2.38 | 5.56 | 0.85 | 1535/1535/1159 | 1.45/1.12 | 1.54 |
| cube_shell | 65536 | 0.464 | 0.173 | 0.200 | 0.109 | 2.68 | 1.59 | 0.87 | 165/263/330 | 1.22/1.17 | 1.35 |
| | 262144 | 2.329 | 0.586 | 0.756 | 0.272 | 3.98 | 2.15 | 0.78 | 180/277/341 | 1.19/1.15 | 1.36 |
| | 1M | 10.52 | 2.268 | 3.003 | 1.039 | 4.64 | 2.18 | 0.76 | 289/289/290 | 1.21/1.14 | 1.42 |
| cube_edges | 65536 | 0.139 | 0.037 | 0.039 | 0.035 | 3.78 | 1.06 | 0.95 | 63/84/96 | 1.23/1.15 | 1.43 |
| | 262144 | 0.690 | 0.105 | 0.116 | 0.087 | 6.54 | 1.21 | 0.91 | 67/88/99 | 1.23/1.14 | 1.58 |
| | 1M | 2.979 | 0.423 | 0.441 | 0.407 | 7.04 | 1.04 | 0.96 | 92/92/93 | 1.29/1.14 | 1.54 |
| uniform | 65536 | 0.998 | 0.723 | 0.755 | 0.269 | 1.38 | 2.68 | 0.96 | 377/700/922 | 1.30/1.22 | 1.33 |
| | 262144 | 6.150 | 2.639 | 3.519 | 0.800 | 2.33 | 3.30 | 0.75 | 468/807/1117 | 1.27/1.17 | 1.28 |
| | 1M | 31.92 | 13.13 | 17.07 | 3.058 | 2.43 | 4.30 | 0.77 | 773/773/855 | 1.22/1.14 | 1.26 |

#6 is bitwise equal to #5, with equal per-slot counts, in every cell. #7's per-warp max/mean and
SIMT efficiency are 1.000 everywhere.

## 2. Headline findings

1. **Distribution shape decides which technique matters (D-030).**
   - On the 3D clustered ICs, #7 beats the best #5 by 3.8–6.1× at N ≥ 262144.
   - On cube_shell #7 gains only 2.2×, and on cube_edges 1.04–1.21×. There the **Morton sort alone
     already makes warps coherent**: #4 → #5 is 4.0–4.6× on cube_shell and 6.5–7.0× on cube_edges,
     against 2.1–2.7× on 3D ICs. On cube_edges #5 already runs at 220–230 G int/s, which is #7's rate
     on 3D ICs.
   - #5 and #7 attack the same loss; dimensionality decides which one is needed.
2. **#6 still never wins.**
   - The warp queue runs at 0.75–0.97× of #5, and worse at b256 for N ≥ 262144 (0.75–0.85×, D-031).
   - On galaxies at 65536 (single wave), #5's SM efficiency is 0.62 with a **37% tail**, and #7's is
     0.61 with a **39% tail**: the largest SM-level losses in the project. The queue lifts SM eff to
     0.87, but mean busy time rises 53%, so it ends at 0.96×.
   - Exploratory: the block-granular queue and the queue on #7 reach parity there (0.996–1.010×) and
     nowhere else.
3. **Intra-warp imbalance** (#4 per-warp max/mean at 262144): galaxies 1.60 > disk 1.49 > uniform
   1.27 > cube_edges 1.23 > plummer 1.20 > cube_shell 1.19. The predicted corner hot spots on the
   cube shapes did not appear.
4. **Between-warp imbalance:** galaxies 2.61 and disk 2.22 at 65536, the largest in the project (core
   was 1.59). It falls to about 1.5 at 1M.
5. **The normalised error gates (D-027) held in all 133 runs.** The old per-body relative TG1 fails on
   cube_edges and cube_shell at every N, which is why it was replaced. Error is flat in N, and #7 is
   more accurate than #5 on every IC.
6. **Mass spectrum:** Salpeter adds 10–11% to #5's work (the centre of mass sits off the cell centre,
   so cells open earlier); heavy-few adds 1%. Errors are unchanged, and heavy bodies are approximated
   no worse than light ones.
7. **Trees at 1M:** push depth ≤ 38, stack need ≤ 40 vs the 64-entry warp stack, 0 overflows.
   cube_edges: 2.08% duplicate Morton keys, max depth 48.
8. **Stack 64 → 96:** old/new traversal 0.977–1.027 under load, no measurable cost.

> **Main-session observation (not scored):** on cube_edges at 262144, the **tree build
> (≈ 0.27 ms) costs 2.5–3× the traversal** (#5 0.107 ms, #7 0.088 ms). For low-dimensional
> distributions the step is build-bound, not traversal-bound, so a faster build (or an
> overlapped / CPU-assisted build, R3 #8) would matter more there than any traversal technique.

## 3. Scorecard: **130 HIT / 43 MISS (173 rows)**, `scorecard.csv`

| Group | Rows | HIT | Notes on the misses |
|---|---|---|---|
| I | 47 | 36 | Uniform #5 and the cube shapes at small N high; I6 order (disks exceed plummer at 1M); I7 (plummer +8.3% only) |
| M | 32 | 16 | No corner hot spots on the cubes; uniform higher; disks at 65536 just above the range |
| W | 16 | 12 | Galaxies/disk spread wider in N |
| S | 17 | 14 | Galaxies at 65536/262144 low (2.29, 3.77); cube_edges at 1M 1.04 |
| Q | 17 | 10 | Queue at b256 for N ≥ 262144 worse than predicted |
| E, K, D | 38 | 38 | |
| C | 4 | 2 | C3: cube_shell union ratio 1.82–2.00 (second only to uniform); C4: cube_edges is not the most imbalanced for #4 |
| V | 2 | 2 | V2 is a disputed reading: taken at t = 3 on the evolved disk |

**Pattern:** the deterministic parts (trees, errors, mass spectrum) held. What missed: the imbalance
model for low-dimensional shapes, the size of #6's locality loss at b256, and the small-N galaxies
speedups, where the single-wave tail dominates.

## 4. Timelines (galaxies 65536, warp units)

| Kernel | SM eff | Tail | Mean busy | Span |
|---|---|---|---|---|
| #5 | 0.622 | 37.0% | 674 µs | 1084 µs |
| #6 | 0.869 | 12.9% | 1029 µs | 1176 µs |
| #7 | 0.613 | 38.7% | 277 µs | 451 µs |
| #6+#7 (exploratory) | 0.761 | 16.9% | 372 µs | 481 µs |

## 5. Mass spectrum (262144, leaf 16 b128)

| Case | #5 int/body | #5 / #7 normalised rms | #7/#5 |
|---|---|---|---|
| plummer equal | 976 | 3.09e-3 / 2.45e-3 | 6.13 |
| plummer salpeter | 1071 (×1.10) | 3.07e-3 / 2.44e-3 | 5.94 |
| plummer heavy-few | 990 (×1.01) | 3.00e-3 / 2.41e-3 | 6.06 |
| disk salpeter | 868 (×1.11 vs disk equal) | 5.03e-3 / 4.15e-3 | 5.15 |

## 6. Validation

- Selftest PASS: 7238 lattice cases, 8291 tree checks, 126 IC-invariant checks.
- TN1/TN2 in all runs, TG3n in every #7 run, #6 bitwise equal to #5.
- #5's FP64-sample error is at most 0.031.

## 7. Presentation (R6)

17 PNGs at 1600×900, dt = 1e-3, captions in `presentation/index.md`:
- **Galaxies:** t = 0 / 1 / 2 / 3 / 4.5 / 8 (approach → first passage → merged remnant).
- **Disk:** face-on t = 0 / 0.5 and edge-on.
- **Cubes:** cube_shell and cube_edges at t = 0 and during the cold collapse.
- **Plummer.**
- **Colour-by-mass:** heavy-few (16 heavy bodies in red) and Salpeter disk.

The cold, halo-less disks fragment into clumps within about one rotation, so the merger shows clumps,
not spiral arms. The captions say so.

## 8. Independent re-check (main session, background load 35%)

cube_edges, N = 262144, best-tuned settings, traversal ms:

| Kernel | Agent | Re-check |
|---|---|---|
| #4 bh_naive (leaf 16 b256) | 0.690 | 0.699 |
| #5 bh_sorted (leaf 32 b256) | 0.105 | 0.107 |
| #7 bh_warp (leaf 16 b128) | 0.087 | 0.088 |

## 9. Open

- Re-run the S and Q items with the desktop apps closed.
- The single-wave tail on galaxies (37–39%) is now the largest loss for both #5 and #7. The two
  arms that reached parity are the natural starting point for the next #6 attempt.
- Tree build dominates on low-dimensional shapes (§2, observation above).
