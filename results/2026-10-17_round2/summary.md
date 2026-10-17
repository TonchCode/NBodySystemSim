# 2026-10-17 — Round 2: imbalance measured, prefetch vs q separated, gates in code

Predictions were pre-registered in `docs/predictions/2026-10-17-round2.md` (commit `9bae5fa`)
**before** these runs. That file was only appended to afterwards: one dated post-registration
note, 17 added lines, 0 removed (verified with `git diff`). The cuda-architect wrote this
summary's content; the harness blocked the agent from writing the file, so the main session
saved it, plus an independent re-check (§8).

**Environment** (`env_before.txt`, `env_after.txt`, full `nvidia-smi -q` alongside):
driver 617.42, cl 19.51.36248, nvcc 13.4 V13.4.92, **power limit 272 W** (default 320 W,
unchanged), **idle 0% before and after**, 1027–1029 MiB VRAM. Under load
(`gpu_clock_log_phase2.csv`): SM median 2760 MHz (2565–2790), power up to 273.6 W, so **the
cap is reached**.

## 1. Load imbalance, measured (critic review 1, #1)

Method (D-011): instrumented twins record SM id + start/end per block. `%globaltimer` ticks in
1.024 µs steps on this board, so each SM's `clock64` is anchored to a timer tick edge before and
after the launches. Raw and calibrated metrics agree within about 1%. Medians of 5 launches.
Gantt plots are in `timeline/plots/`.

| N | Config | Idle SMs | Blocks/SM | Max/mean SM busy (80) | SM eff | Tail (% of span) |
|---|---|---|---|---|---|---|
| 1000 | tiled b256 | 76 | 1 | 20.07 | 0.050 | 0.4 |
| 1000 | tuned b128 full | 72 | 1 | 10.10 | 0.099 | 1.2 |
| 1000 | multi b512 q32 | 17 | 1 | 1.35 | 0.723 | 6.1 |
| 4096 | tiled b256 | 64 | 1 | 5.02 | 0.199 | 0.6 |
| 4096 | tuned b128 full | 48 | 1 | 2.52 | 0.396 | 0.7 |
| 4096 | multi b512 q8 | 16 | 1 | 1.27 | 0.786 | 1.4 |
| 10000 | tiled b256 | 40 | 1 | 2.005 | 0.497 | 0.5 |
| 10000 | tuned b128 full | 1 | 1 | 1.019 | 0.980 | 0.6 |
| 10000 | multi b1024 q8 | 1 | 1 | 1.018 | 0.970 | 1.5 |
| 65536 | tiled b256 | 0 | 3..4 | 1.231 | 0.812 | 23.5 |
| 65536 | tuned b128 full | 0 | 6..7 | 1.092 | 0.914 | 13.8 |
| 65536 | multi b256 q4 | 0 | 12..13 | 1.016 | 0.981 | 0.1 |

**Two regimes:**
- **Small N: starvation.** With B ≤ 80 blocks there is exactly one block on each of B SMs, and
  80 − B SMs stay idle the whole kernel. Active SMs are balanced to within 1.5%.
- **Large N: the wave tail.** The ceil(B/80) step is exact when all blocks are co-resident
  (tiled b256: 23.5% tail). With dynamic refill (multi q4: 1024 blocks over 480 slots) the tail
  shrinks to 0.1%.

**Instrumentation overhead misses the ≤1% prediction:** the instrumented twin runs −3.4…+6.2%
different from production. Production SASS is identical for all 57 kernels. The difference
comes from ptxas rescheduling the twin's force loop (same opcodes, different
schedule/registers), not from the stamp code. So **timelines are used for structure** (placement,
idle SMs, tail fraction) and **throughput always comes from production kernels**.

Exploratory, post-registration and not scored (`timeline_exploratory/`): at N = 65536, b64
reaches SM eff 0.972 vs 0.809 for b256, exactly the step M1 predicts. But it is only 8%
faster, because each b64 SM is about 12% slower per interaction. The round-1 P2a miss
therefore sits in M1's per-block cost term, not its step function (D-012).

## 2. Prefetch vs q, separated (critic #4)

G interactions/s, medians of 21 runs.

| N | Block | tuned u8 | q1 pf off | q1 pf on | pf @ q1 | q_best off → on | pf @ q_best |
|---|---|---|---|---|---|---|---|
| 4096 | 64 | 258 | 262 | 305 | +16.1% | q8 705 → 761 | +7.9% |
| 4096 | 128 | 273 | 275 | 303 | +10.1% | q16 808 → 853 | +5.6% |
| 4096 | 256 | 223 | 217 | 234 | +7.7% | q8 834 → 844 | +1.1% |
| 4096 | 512 | 132 | 138 | 139 | +0.5% | q8 838 → 894 | +6.7% |
| 65536 | 256 | 1045 | 1084 | 1080 | −0.3% | q4 1264 → 1264 | 0.0% |

At N = 4096 the 2.56× gain over the best-tuned baseline splits as **q ×2.40 · prefetch ×1.067**.
Prefetch matters (+10–16%) only at 2–4 warps per SM. Caveat: unroll is still confounded (multi
uses u8, the baseline uses full unroll).

## 3. Speedups vs the best-tuned baseline (CLAUDE.md rule 8)

Baseline = the best 1-thread/body config over tiled, tuned and multi q = 1. In this session that
was tuned b128 full at every N.

| N | Baseline G/s | Best multi G/s | vs baseline | vs tiled b256 |
|---|---|---|---|---|
| 1000 | 75 | 180 (b256 q16) | **2.40×** | 3.54× |
| 4096 | 348 | 894 (b512 q8) | **2.56×** | 3.91× |
| 10000 | 850 | 1188 (b1024 q8) | **1.40×** | 2.16× |
| 65536 | 1244 | 1266 (b128 q2) | **1.02×** | 1.16× |

## 4. Gates (D-009)

- All 4000 sweep runs pass G0, G1 and G2 (100 configs × 4 N × 10 runs, multi with pf on and off).
- The q = 1 order sits at 0.44–0.68 · u√N, a 1.47–2.27× margin to G1.
- κ_max of the new sample: 24 / 32 / 103 / 110 across the four N.
- `--selftest`: 1810 cases pass.

## 5. Probe (critic #2)

| Variant | SASS slots/interaction | Issue-model ratio | Round 1 | Round 2 |
|---|---|---|---|---|
| base | 17.38 | 1.000 | 1.000 | 1.000 |
| ftz | 14.38 (−3) | 0.827 | 0.90 | 0.900 / 0.899 |
| mufu2 | 19.38 (+2) | 1.115 | 1.19 | 1.189 / 1.184 |
| fmul3 | 20.38 (+3) | 1.173 | 1.185 | 1.185 / 1.185 |

- All four variants: 40 registers, 0 spills.
- The run was power-capped at 267–271 W with SM clock 2610–2640 MHz. The clock can't be
  resolved per variant, but the whole band is 1.1% and two passes agree, so the misfits are not
  clock artifacts.
- Only fmul3 fits the issue model. mufu2 overshoots by +7 pts, which points to the XU pipe;
  ftz undershoots, so something other than issue limits it.
- Conclusion stays **"issue-limited with a significant XU co-limit"**, pending ncu.

## 6. Long-run energy (N = 4096, 100 time units)

| Run | dt | ΔE/E @ t=100 | max \|ΔE/E\| | Growth exponent α |
|---|---|---|---|---|
| E1 tiled | 1e-3 | +6.2e-7 | 9.6e-7 | 0.98 |
| E2 tiled | 5e-4 | +1.09e-6 | 1.28e-6 | 0.71 |
| E3 multi b512 q8 | 1e-3 | +1.60e-6 | 1.60e-6 | 0.84 |

- **Halving dt increased the drift.** It is accumulated FP32 round-off, not leapfrog truncation,
  so dt = 1e-3 stays (D-014).
- Multi and tiled drift at the same order; the trajectories diverge chaotically.
- Virial ratio 0.980–1.016.

## 7. Scorecard vs `9bae5fa`: 18 HIT / 7 MISS (2 of the misses are better than predicted)

| Prediction | Verdict |
|---|---|
| P-TL1 (≤80 blocks → one per SM) | HIT |
| Imbalance table | 10/12 HIT. MISS: N=1000 multi (1.35 vs 1.23–1.31; a 2 µs kernel). MISS: N=65536 multi tail (0.1% vs 2–10%, better than predicted) |
| P-TL2 (clock 2.70–2.80 GHz, ±1% across configs) | **MISS**: 2.52–2.78 GHz; under the power cap, faster kernels clock lower |
| P-TL3 (starvation at small N, tail at large N) | HIT |
| P-PF0 (multi q1 pf off = tuned u8 ±3%) | **MISS** 3/5 (+4.8%, +3.7%) |
| P-PF1, P-PF2, P-PF3 | HIT |
| P-PF4 | q1 **MISS** by 0.3 pts (noise); q4 HIT |
| P-OV (overhead ≤1%) | **MISS** (loop rescheduled) |
| P-E1, P-E2, P-E3, virial | HIT |
| P-G1, P-G2 | HIT |
| P-G2b (error rises with the new sample) | **MISS** (only at N=1000) |
| P-PR1 / P-PR2 | HIT / HIT, weak |

## 8. Independent re-check (main session)

| Config | Agent | Re-check |
|---|---|---|
| tiled N=4096 b256 timeline | 64 idle, SM eff 0.199, max/mean 5.02 | 64 idle, 0.198, 5.017; overhead +5.5% |
| multi N=4096 b512 q8 timeline | 16 idle, 0.786, 1.27 | 16 idle, 0.790, 1.261; overhead +3.9% |

Also: `--selftest` 1810 PASS. The predictions file diff shows append-only changes.

## Process note

The agent ran `git status` once (read-only, no effect) despite its no-git instruction.
Recorded for transparency.

## Open (Toncho)

- The 272 W cap visibly couples SM clock to kernel speed.
- GPU performance counters are needed to name ftz's non-issue limiter.
- `-ftz` and higher-precision integrator state are numerics decisions.
