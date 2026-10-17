# 2026-10-17 — Tree round 2: R3 #7 (warp-cooperative walk) and R3 #6 (persistent threads)

Predictions: `docs/predictions/2026-10-17-tree-round2.md`, committed as **c6a12cc** before these
runs. Afterwards only a dated post-registration note on scoring readings was added. Design:
`docs/03-design.md` §17–§18. Decisions: D-021..D-026. The cuda-architect wrote the content; the
harness blocked the agent from writing this file, so the main session saved it.

**Analysis:** `python tools/tree_analysis.py results/2026-10-17_tree_round2 --round2` writes
`tree_metrics.csv`, `derived.csv`, `timeline_slots.csv` and **`scorecard.csv`** (committed, item
level). A console copy is in `analysis_output.txt`.

**Environment:**
- Driver 617.42, cl 19.51.36248, nvcc 13.4 V13.4.92.
- **Power limit 272 W** (default 320, unchanged).
- Under load: SM median 2760 MHz (p5 2640), power median 196 W, max 271 W.
- `--selftest` on the measured binary: PASS.

> ⚠ **UNCONTROLLED BACKGROUND LOAD: 20–22% idle utilisation** (desktop apps), before and after.
> - Measured effect on round-1 reference points: +2.4% (#4) and +2.7% (#5) traversal (`contam/`).
> - Absolute times drift up to about 7% between sets.
> - Adjacent-run ratios are stable to within 1.5% (set A2), except static core (6.8%).
> - **The #6 items hinge on 1–5% differences; re-run them with apps closed.**

**Runs:** 396 scripted runs, all exit 0, no gate FAIL, no stack overflow. 64 exploratory runs, all
exit 0. Settings: θ = 0.5 (bmax-θ), leaf 16, block 128, `--warmup-ms 300 --runs 11`, batch 5 for
N ≤ 65536.

## 1. Main grid (set A), traversal phase, ms (median)

| IC | N | #4 | #5 | control | **#7** | #7 list | #6 warp q | #6 lane | #6 lane on #4 | #6 static | #6+#7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plummer | 65536 | 2.822 | 1.804 | 0.348 | **0.576** | 0.590 | 2.017 | 2.824 | 4.177 | 2.089 | 0.646 |
| | 262144 | 16.57 | 8.723 | 1.061 | **1.607** | 1.697 | 9.714 | 11.73 | 19.31 | 9.764 | 1.715 |
| | 1M | 78.54 | 39.53 | 4.300 | **5.982** | 6.467 | 43.06 | 58.34 | 86.39 | 47.30 | 6.311 |
| core | 65536 | 3.964 | 2.511 | 0.552 | **0.729** | 0.753 | 2.767 | 3.590 | 5.621 | 2.553 | 0.797 |
| | 262144 | 21.83 | 11.19 | 1.300 | **1.759** | 1.875 | 12.09 | 15.44 | 24.90 | 12.48 | 1.891 |
| | 1M | 93.55 | 49.54 | 5.111 | **6.774** | 7.310 | 53.21 | 70.22 | 106.7 | 55.84 | 7.125 |

**Plummer 1M:**

| Kernel | Int/body | Pops/body | SIMT eff (int / pop) | Real traversal rate |
|---|---|---|---|---|
| #5 | 1063 | 1612 | 0.201 / 0.976 | 28.2 G/s |
| #7 | 1837 (×1.73) | 2036 (×1.26) | **1.000 / 1.000** | **322 G/s** |
| control | — | — | 1.000 / 1.000 | 259 G lane-int/s |

Every persistent arm is bitwise equal to its sibling with equal per-slot counts (36/36 set A, 24/24
exploratory). Error, TG3 (#7 vs #5 at the same θ): rms ×0.66–0.85, p99 ×0.60–0.91. **#7 is more
accurate.** TG3 PASS 60/60.

## 2. Headline findings

1. **R3 #7 works.**
   - Traversal is **3.1–7.3× faster than #5** and the gap grows with N: plummer 3.1 / 5.4 / 6.6×,
     core 3.4 / 6.4 / 7.3× at 65536 / 262144 / 1M. The full step is 2.6–6.0× faster.
   - It does 1.6–1.9× the interactions and 1.2–1.3× the pops at SIMT efficiency 1.000.
   - **Best vs best (set C, rule 8, θ = 0.5): 5.9× at 1M** (#5 leaf 32 b256, 35.30 ms vs #7 leaf 8
     b256, 5.97 ms) and 5.2× at 262144. Total step: 5.4× and 4.4×.
2. **The coherent control measures the ceiling directly** (critic review 2, #1).
   - #5's own kernel with all 32 lanes on one body runs 5.2–9.7× faster than #5 on the same lane
     work. Intra-warp incoherence therefore costs #5 a factor of 5–10.
   - #7 sits 1.32–1.65× above the control, matching its union overhead.
   - #7 closes **69–88% of the gap** (log terms).
3. **R3 #6 never beats the hardware block scheduler,** although each arm improves the balance metric
   it targets.

   | #6 arm | Speed vs its sibling | Balance it improves |
   |---|---|---|
   | Warp-granular atomic queue | 0.89–0.93× on #5, 0.89–0.95× on #7 | Single-wave SM eff (core 65536: 0.92 vs 0.81, tail 6.6% vs 18.3%); 1M warp-slot fill 0.98 vs 0.95 |
   | Lane-granular refill | 0.64–0.74× on #5, 0.68–0.91× on #4 | #4 core pop eff 0.61 → 0.91 at 1M |
   | Static cost-balanced partition | 0.84–0.98× | |

   - The lane arm is still 36% slower at 65536, where it does no refills at all.
   - **Exploratory diagnosis** (not pre-registered, `explore/`):
     - A block-granular queue (4 Morton-adjacent units per claim) recovers most of the warp queue's
       loss (0.94–0.98× vs 0.86–0.93×). So that loss is lost intra-block L1 locality.
     - Checking for idle lanes every 8 pops barely helps lane refill. Its loss is the loop
       structure, likely the warp-collective calls forcing reconvergence every pop (**not
       verified in SASS**).
   - **A measured negative result with a mechanism, not a strawman:** the lane-granular arm that
     critic review 2 asked for was built and measured.
4. **#5 is limited by L1 capacity, not occupancy** (set D, critic review 2, #6).
   - With the L1 carve-out pinned small by padding, #5 is flat from 2 to 11 blocks/SM
     (19.9–20.7 G/s). Unpadded it does 29.0 G/s (×1.51).
   - The control and #7 scale with occupancy (×1.75, ×1.87 from 4 to 11 blocks/SM), so they are
     latency-bound.
   - Caveat: the "12" cell ran at 11 blocks/SM.
5. **Leaf × block (set C):** #5's best leaf is **32**, not the predicted 8–16, and b256 is 11% faster
   than b128 at 1M. #7 is flat (±1%).
6. **#7 needs #5's ordering.** On unsorted warps (`bh_warp_naive`) the union does 12.9–14.8× #4's
   interactions. It is still 1.2–1.6× faster than #4, but 4.1–8.3× slower than `bh_warp`.
7. **Crossover vs the best all-pairs: N ≈ 29,200** for `bh_warp` (set E, same session), down from
   39,700 for #5. `bh_warp` total vs best all-pairs: 4.4× at 65536, 24.9× at 262144, 123× at 1M
   [D].
8. **θ dependence (set F, 262144):**

   | θ | 0.3 | 0.5 | 0.7 | 1.0 |
   |---|---|---|---|---|
   | Union overhead | ×1.55 | ×1.79 | ×2.08 | ×2.11 |
   | #7 speedup over #5 | 8.1× | 5.4× | 4.0× | 3.0× |

   At θ = 0, #7 equals #5 within 2% and passes D-009.

## 3. Scorecard: **39 HIT / 22 MISS (61 rows)**, `scorecard.csv`

- **HIT:** all coherent-control items (C1–C6), union work ratios and their flatness in N (W1–W5),
  T7/TC (W7), every #7 speedup range (W8–W12), TG3 ratios and passes (W14a–d, W15), W17, W18, P10,
  P12, P13, L5, O1, O3, O4, D1, X1–X5, G1.
- **MISS:**
  - **All #6 performance items** (P1–P9, P11): predicted 0.75–1.55× for lane refill and ≈ 1.0× for
    the queues; measured 0.64–0.98×.
  - W13 (the list variant is 2–7% slower, not 5% faster).
  - W16 (`bh_warp_naive` beats #4 at 1M by 1.59× vs ≤ 1.3×).
  - L1–L4, L6, L7 (leaf/block optima).
  - O2 (1.75 vs ≥ 1.8, at 11 blocks/SM).
  - O5, O6 (**disputed reading**: occupancy is confounded with L1 capacity).
  - D2 (1 − fill = 0.108 vs ≤ 0.10).
- **Anchored items** (built on pre-registration observations): W1, W3, W5, W7, W13, W14a–d, P3.
  W13 and P3 missed.
- **What the misses say:** the #7 model held (a coherent walk is bound by its dependent pop chain,
  and it was measured against a coherent control). The #6 model was wrong: it bounded the gain by
  balance metrics, but ignored that a software scheduler changes which units share an SM's L1, and
  the fixed cost of warp-collective calls inside the inner loop. Balance improved; speed fell.

## 4. Timelines (set B)

| Run | SM eff | Tail | Warp-slot fill |
|---|---|---|---|
| #5 plummer / core 65536 | 0.897 / 0.811 | 9.2% / 18.3% | 0.805 / 0.691 |
| #6 warp q plummer / core 65536 | **0.925 / 0.917** | 5.3% / 6.6% | 0.842 / 0.721 |
| #7 plummer / core 65536 | 0.792 / 0.733 | 21.8% / 27.3% | 0.584 / 0.556 |
| #5 plummer / core 1M | 0.988 / 0.990 | 1.0% / 1.1% | 0.945 / 0.946 |
| #6 warp q plummer / core 1M | 0.991 / 0.994 | 0.6% / 0.5% | **0.975 / 0.980** |
| #7 plummer / core 1M | 0.980 / 0.985 | 1.9% / 1.5% | 0.892 / 0.902 |

#7 makes the *single-wave tail* worse in relative terms: its warp costs spread 1.5–2.6× (max/mean)
over a launch about 3× shorter. **At 65536, between-warp imbalance is now #7's largest loss.** That
is where a cheap block-granular queue could pay off next.

## 5. Validation

- TG1/TG2 PASS in every θ > 0 run; TG3 PASS 60/60.
- D-009 PASS at θ = 0 and in all set-E all-pairs runs.
- Exact-sibling checks 60/60.
- 0 stack overflows.
- Selftest: 4450 lattice cases + 1604 tree checks, including the exact θ > 0 host check (worst 0.27
  of its bound). Exploratory binary: 2063 tree checks, PASS.

## 6. Independent re-check (main session, background load 20%)

Plummer 262144, θ = 0.5, leaf 16, b128, 7 runs, traversal:

| Kernel | Agent | Re-check |
|---|---|---|
| bh_sorted (#5) | 8.723 | 8.677 ms |
| bh_warp (#7) | 1.607 | 1.592 ms |
| bh_persistent (#6 warp q) | 9.714 | 9.579 ms |

The gates PASS. #7's rms error (4.59e-3) is below #5's (6.55e-3).

## 7. Process notes

- The phase-1 sanity run led to two code changes before registration (persistent-grid cap, batched
  refill), disclosed in predictions §0.5.
- The exploratory kernels were added after set A and checked to leave the SASS of every measured
  kernel unchanged.
- The agent ran no git command and made no system or power changes. It did not remove the
  background load, which comes from the owner's desktop session.

## 8. Constraints for the next round (new ICs, mass spectrum, live N), from the agent

1. **The per-body relative-error gates break on near-zero net force** (shells, the centre between
   colliding galaxies, small bodies next to a dominant mass). Switch to normalised error (Σ|f| or
   |a|_rms, D-009 style) first.
2. **A heavy central body:** the bmax criterion is geometric, so the error near the heavy body is
   dominated by its cell. The global eps = 0.01 and dt = 1e-3 will need a smaller dt.
3. **Stack depth 64:** dense cusps at 1M may exceed about 60 levels. That FAILs the gate (detected,
   not silent) but blocks the run.
4. **Morton quantization:** 21 bits per axis over the largest extent. Widely separated galaxies
   coarsen the cells, and dense nuclei fall into duplicate keys. That affects tree quality, not
   correctness.
5. **Equal masses** are assumed in `makeIC` normalisation and the selftest ICs (the tree itself is
   mass-weighted).
6. **Live N in the viewer:** the tree side is fine (`TreeCtx` grows on demand); the interop buffers
   must be resized.
7. **A uniform cube** is fine as is (the no-imbalance control).
