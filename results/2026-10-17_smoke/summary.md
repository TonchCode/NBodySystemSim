# 2026-10-17 — Smoke test (NOT a benchmark)

Purpose: prove the skeleton builds, runs, validates and conserves energy. **These numbers are
not for the report.** Background GPU load was about 20% (Wallpaper Engine, Discord, Riot,
NVENC), and the power limit was 272 W (default 320 W). No env capture, no sweep.

Build: CMake + Ninja, nvcc 13.4.92, MSVC 19.51.36248, `-arch=sm_89 -O3 -lineinfo`.
Driver 617.42 (CUDA 13.4). The CUDA runtime reported a 2610 MHz boost clock.
`nvidia-smi` showed 2745 MHz SM clock right after the runs.

## Resource usage (ptxas)

| Kernel | Registers | Spills | Occupancy @ block 256 |
|---|---|---|---|
| allpairs_naive | 39 | 0 | 6 blocks/SM → 1536 thr/SM = 100% (register-limited at exactly 6) |
| allpairs_tiled | 40 | 0 | 40 × 256 = 10,240 regs/block → 65,536 / 10,240 = 6 blocks → 100% |
| kickDrift | 14 | 0 | — |

## Results (median of 10 unless noted, block 256, Plummer seed 42)

| Kernel | N | median ms | G interactions/s | "GFLOP/s" @20 |
|---|---|---|---|---|
| naive | 16,384 | 0.830 | 323 | 6,467 |
| tiled | 16,384 | 0.323 | 831 | 16,614 |
| tiled | 131,072 (5 runs) | 17.21 | 998 | 19,966 |
| tiled | 4,096 (3 runs) | 0.075 | 224 | 4,487 |
| tiled | 10,000 | 0.196 | 511 | 10,222 |
| tiled, block 96 | 10,000 | 0.204 | 491 | 9,819 |

Energy: tiled, N = 4096, dt = 1e-3, eps = 0.01, 2000 KDK steps. Relative drift **−2.8e-8**.

## Observations (inputs for cuda-architect, not conclusions)

1. **Tiled is 2.6× naive at N = 16k.** Shared-memory tiling works as expected.
2. **About 1.0 T interactions/s at N = 131k.** That is about 20 "TFLOP/s" by the 20-FLOP
   convention, about 37% of the 53.5 TFLOPS FP32 peak. The convention counts `rsqrtf` as
   several FLOPs, but it actually runs on the SFU (MUFU) units, at 16/clk/SM vs 128 FMA/clk/SM.
   **Hypothesis:** the kernel is bound by MUFU throughput or instruction issue, not FMA.
   Nsight Compute must confirm before anyone writes it down.
3. **Small N starves the GPU.** N = 4096 at block 256 is only 16 blocks for 80 SMs (20% of
   SMs busy), so the rate falls to 224 G/s. This is the measured motivation for **R3 #3**
   (several threads per body), and it is a ready-made figure for the load-balancing chapter.
4. **Validation shows exactly 0 error, and that is legitimate here.** Naive and tiled sum the
   j-terms in the same order with the same FMA sequence. Padding bodies have mass 0, so they
   add an exact `+0`. Variants that **reorder** summation (multi-thread-per-body, trees) will
   show nonzero FP32 error. A tolerance must be set before those land (suggest max rel
   < 1e-4 for reordering; a tree method has its own θ-dependent error).
