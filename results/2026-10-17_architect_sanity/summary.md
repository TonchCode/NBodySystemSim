# 2026-10-17 — cuda-architect sanity runs for R3 #2 and #3 (NOT the official benchmark)

**Sanity only.** Power limit was 272 W (default 320 W). The GPU was idle before the runs
(`env_before.txt`: 0%, 210 MHz), and the median SM clock under load was ≈ 2760 MHz
(`gpu_monitor*.csv`). Background apps were not controlled during the runs. The rigorous runs
belong to bench-profiler. The architect wrote the raw data in this folder. The harness blocked
the agent from writing this summary, so the main session wrote it from the agent's report and
re-checked the key rows independently (bottom of this file).

Timing method (D-007): time-based warm-up, batched launches. **Correction (critic review 1,
#9):** the logs show `runs=15` throughout, batch 1 at N = 65536 and batch 5 for part of the
out-of-sample sweep, not the "median of 10 / batch 50" first written here. The CSV `q`
column holds the resolved q, so auto and explicit rows are merged. Single timed launches at
small N were bimodal: the GPU idles at 210 MHz, and a 5–25 µs launch is mostly jitter. The
smoke-test N = 4096 figure suffers from the same problem.

## Occupancy (runtime occupancy API)

| Kernel | Block | Regs | Blocks/SM | Occupancy |
|---|---|---|---|---|
| tiled / tuned | 64 / 128 / 256 / 512 | 40 (tuned u1: 19, u4: 37) | 24 / 12 / 6 / 3 | 100% |
| any | 1024 | 40–56 | 1 | **67%**: one 1024-thread block per 1536-thread SM |
| multi | 64–512 | 40 | as above | 100% |
| multi b64 q32 | 64 | 44 | 20 | 83% (derived 87.5%; registers are allocated per partition) |

Shared memory never limits any of these configurations.

## Validation (gates pre-registered in D-006)

- `allpairs_tuned`: **bitwise equal** to naive in all 25 configurations, at every N.
- `allpairs_multi` vs naive. Ranges run from q2 to q32:

| N | rms rel | max rel |
|---|---|---|
| 1000 | 5.5e-7 → 4.8e-7 | ≤ 2.2e-6 |
| 4096 | 1.1e-6 → 9.3e-7 | ≤ 3.6e-6 |
| 10000 | 1.7e-6 → 1.5e-6 | ≤ 6.5e-6 |
| 65536 | 4.5e-6 → 3.7e-6 | ≤ 1.9e-5 |

- Odd N (999, 4097, 10007) pass (`validation_oddN.txt`).
- **Multi is more accurate than naive against FP64.** At N = 65536, rms vs FP64 falls from
  3.7e-6 (q1) to 1.4e-7 (q32), roughly 1/q, because q interleaved partial sums are shorter.
  The "difference vs naive" is mostly naive's own rounding error.
- Energy drift, N = 4096, 2000 KDK steps (`energy.txt`): tiled and tuned −2.76e-8, multi q4
  −2.97e-8, multi q32 −3.14e-8.
- Predicted max-rel spikes on near-cancelling bodies did not appear. κ_max ≈ 15 in this
  Plummer initial state. A more concentrated snapshot is needed to show them.

## Predictions vs measured (G interactions/s)

| Prediction (written before measuring) | Measured | Verdict |
|---|---|---|
| N = 4096 multi q4 ≈ 700 (550–850), ≈ 3× tiled | q4 771; best 893 = **3.9×** tiled | hit |
| N = 1000 in 120–300 | 170 | hit |
| multi q1 = tiled ± 3% | ± 1.8% | hit |
| full unroll at block 1024 ≥ 5% slower (instruction cache) | −18% | hit |
| q32 is the slowest multi at N = 65536 | 989 | hit |
| N = 65536: block 64 beats 256 by ~20% (tail effect) | +6%; block 128 best | **miss** |
| N = 4096 / 10000: block size doesn't matter | block 128 +23–24% | **miss** |
| unroll 16 / full gain 0–2% | +5% / +8% | **miss** |
| auto-q within 5% of best | loses 7–9% at b256, N = 4096 and 10000 | **miss** (local) |
| absolute levels at N = 65536 | 4–26% higher than predicted | **miss** (calibrated on the loaded smoke-test GPU) |

Auto-q across all 32 (N, block) cells: mean loss vs the best 1.8% (0.9% out-of-sample)
against 6.8% (4.8%) for the naive "fill every thread slot" rule (D-008). Root cause of the
misses: the model assumed 8 warps saturate an SM. Measured per-SM rate ≈ 2.1 G/s per warp up
to about 6 warps, ≈ 14 G/s at 8 warps, and ≈ 17 G/s at 32. A refit model did better on the
data it was fitted to and worse on new N values, so the pre-registered v1 was kept.

## Best configuration per N

> **Baseline note (critic review 1, #3):** "vs tiled" below is against tiled at the default
> b256. The fair baseline is the **best-tuned 1-thread/body config** (tuned b128 full: 349 G/s
> at N = 4096). Against that, multi's gain at N = 4096 is **2.56×**, not 3.9×. Part of the
> small-N gain is also register prefetch rather than q (#4); that is being separated.


| N | Best | G int/s | vs tiled |
|---|---|---|---|
| 1000 | multi b512 q32 | 173 | 3.4× |
| 4096 | multi b512 q8 | 893 | 3.9× |
| 10000 | multi b1024 q8 | 1182 | 2.15× |
| 65536 | multi b256 q4 (tuned b128 full: 1238) | 1256 | 1.15× |

## The MUFU hypothesis from the smoke test is wrong

ncu was blocked (ERR_NVGPUCTRPERM: GPU performance counters need admin or the "allow all users"
Control Panel setting). A timing probe was used instead (`mufu_probe.cu`,
`mufu_probe_out.txt`; b128, full unroll, N = 65536):

| Change per interaction | If issue-bound | If MUFU-bound | Measured |
|---|---|---|---|
| +1 MUFU | +11.5% | +100% | **+19%** |
| +3 FMUL | +17% | 0 | **+18.5%** |
| FTZ rsqrt (−3 non-MUFU instructions) | −17% | 0 | **−10%** |

~~The kernel is instruction-issue bound; MUFU co-limits only at ~2 per interaction.~~
**Revised after critic review 1 (#2):** only the fmul3 probe matches the pure issue model.
mufu2 (+19% measured vs ≈ +6% predicted by issue count) and ftz (−10% vs −17%) don't. The XU
pipe is ≈ 45% busy in the base kernel, so the honest statement is **"issue-limited with a
significant XU (MUFU) co-limit"**. ncu pipe utilisation and stall reasons will settle it.

## Open

- `-ftz` / `rsqrt.approx.ftz`: +10%. **Corrected (critic review 1, #12):** it removes
  *three* instructions per interaction (the non-FTZ denormal guard: FSETP + 2 FMUL; SASS counts
  in `results/2026-10-17_round2/probe/sass_counts.txt`), not one. It changes numerics, so it
  needs Toncho's call.
- Enable GPU performance counters for Nsight Compute. This is an admin/security setting, so
  Toncho has to do it.
- Tuned b256 u8 is 2–4% slower than tiled despite the same algorithm. Unexplained.
- Search block and q jointly. The winners at N = 4096–20000 have about one large block
  (16–32 warps) per SM.
- For R3 #6 (persistent threads), aim for ≥ 16–32 resident warps per SM: the per-SM rate keeps
  rising up to about 32.

## Independent re-check (main session, same day, GPU at ~43% background load)

| Config | Architect | Re-check | Validation |
|---|---|---|---|
| tiled N=4096 b256 | — | 217 | bitwise = naive |
| multi N=4096 b512 q8 | 893 | 886 | rms 9.5e-7 vs naive; FP64 rms 1.5e-7 (naive 9.4e-7) |
| multi N=4097 b256 auto (q=4) | — | 729 | rms 9.9e-7 |
| multi N=65536 b256 q4 | 1256 | 1182 (−6%, background load) | rms 3.9e-6; FP64 rms 9.6e-7 (naive 3.6e-6) |
| tuned N=65536 b128 full | 1238 | 1164 (−6%) | bitwise = naive |

The relative results reproduce. The absolute numbers sit about 6% lower under background load,
consistent with the uncontrolled environment.
