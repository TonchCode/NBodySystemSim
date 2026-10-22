# Nsight Compute profile of the force kernels (2026-10-22)

Script: `tools/ncu_profile.bat` (run by Toncho from an administrator prompt: the driver still
refused counters to the non-elevated session after the "all users" setting and a restart).
`--set full`, one launch per kernel after two warm-up launches, `--cache-control all` (caches
flushed between replay passes), 45 replay passes each. **Diagnostic, not pre-registered**, and
the durations are not comparable with the bench (replay, cache flush, single launch). Reports
(`*.ncu-repz`) stay local: they can embed the process environment. Committed: `*_details.csv`,
`*_raw.csv` (exports), `stall_breakdown.txt`, `*.log`.

| | all-pairs #1 (65,536, b256) | #4 naive (1M) | #5 sorted (1M, leaf 32 b256) | #7 warp (1M, leaf 16 b64) |
|---|---|---|---|---|
| duration under ncu | 4.36 ms | 78.5 ms | 37.3 ms | 6.39 ms |
| SM clock reported | 2.48 GHz | 2.59 | 2.60 | 2.51 |
| issue slots busy | **68.2 %** | 21.0 % | 24.9 % | **68.9 %** |
| IPC (warp instr./cycle/SM) | 3.45 | 0.85 | 1.01 | 2.82 |
| active threads per warp | 32 | **6.1** | **10.8** | 32 |
| achieved / theoretical occupancy | **39 %** / 100 % | 96 / 100 | 92 / 100 | 93 / 100 |
| eligible warps per scheduler | 3.26 | 0.31 | 0.38 | 2.49 |
| top stall reason | not_selected 51 % | long_scoreboard 74 %, lg_throttle 20 % | **long_scoreboard 86 %** | long_scoreboard 39 %, wait 21 % |
| stall cycles per issued instruction | 5.5 | 54.5 | 43.7 | 15.7 |
| L1 hit rate (global loads) | 1.8 % | 18.6 % | 50.2 % | 51.9 % |
| L1 hit rate (local = stack) | – | 21.6 % | 30.5 % | – (no local memory) |
| L2 hit rate | 94.3 % | 99.7 % | 99.7 % | 97.2 % |
| **L2 throughput** | 1.6 % | **95.5 %** | **95.2 %** | 24.0 % |
| DRAM throughput | 0.5 % | 1.0 % | 0.9 % | 1.8 % |
| FMA pipe / XU pipe / ALU pipe | **68.6 / 39.2** / 13.9 % | 13.4 / 7.2 / 7.5 | 14.5 / 7.7 / 10.2 | 26.1 / 9.7 / **62.1** |
| shared-memory wavefronts (% of peak) | 31.3 % | 0.03 | 0.03 | 5.3 |
| global load sectors | 8.4 M | 4,599 M | 2,874 M | **200 M** |
| local load + store sectors | 0 | 2,004 M | 1,304 M | **0** |
| warp instructions executed | 2.36 G | 13.6 G | 7.70 G | 3.53 G |

Derived [D]:
- Local memory (the per-thread stack) is **39.3 %** of #5's L1-miss sectors (30.8 % for #4).
- #7 reads **14.4× fewer global sectors than #5** in total and **24.9× fewer per interaction**
  (1,837 vs 1,063 interactions per body), and executes 2.18× fewer warp instructions.

## What it confirms, and what it corrects
1. **All-pairs is issue-bound, not memory-bound: confirmed.** 68 % of issue slots busy, the
   largest stall is `not_selected` (a warp was ready but another one issued), DRAM 0.5 %, L2 1.6 %.
   FMA pipe 69 % with the XU (rsqrt) pipe at 39 %: the SFU co-limit of chapter 2, §2.2.3.
   The low achieved occupancy (39 %) is the level-1 starvation of chapter 4, §4.2: 256 blocks
   for 80 SMs is 3.2 blocks per SM, below the 6 that fit.
2. **#5 is bound by memory latency because L1 is too small: confirmed and sharpened.** 86 % of
   stalls are `long_scoreboard` (waiting for an L1TEX load). L1 serves only half the global
   loads and 30 % of the stack loads; the misses go to L2, which runs at **95 % of its
   throughput**, while DRAM is idle (L2 hit 99.7 %). So "L1-capacity bound" (D-026) means: the
   working set does not fit L1, and the misses saturate the L1→L2 path. Almost 40 % of that
   traffic is the stack in local memory.
3. **#4 vs #5:** the same picture plus `lg_throttle` (20 % of stalls: the load/store queue is full
   of divergent per-lane requests) and only 6.1 of 32 lanes active. Sorting raises active lanes
   to 10.8, the L1 hit rate from 19 % to 50 % and cuts the global sectors by 37 %.
4. **#7 removes the memory wall: confirmed.** 0 local memory, 14× fewer global sectors, L2 at
   24 %. **Correction to D-026:** at full occupancy #7 is not cleanly latency-bound. Issue slots
   are 69 % busy and the ALU pipe (integer, compare, vote, shuffle bookkeeping) is the busiest at
   62 %; the remaining stalls are split between `long_scoreboard` (39 %, the dependent node load)
   and `wait` (21 %, fixed-latency dependencies). It is close to balanced between the pop chain's
   latency and issue throughput, so the next optimisation is still fewer pops (wider nodes),
   which reduces both.
