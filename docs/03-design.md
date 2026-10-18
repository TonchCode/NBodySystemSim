# 03 — Kernel design: all-pairs variants (R3 #1–#3) and the Barnes-Hut tree code (R3 #4–#5, §10–§15)

Owner: `cuda-architect`. Target: the device in [`01-hardware.md`](01-hardware.md) (RTX 4080
SUPER, sm_89, 80 SMs). Tags: [M] measured · [S] spec · [D] derived · [P:key] paper ·
[N:doc] NVIDIA doc. Citation keys are those in `references/references.bib`.

**Order of writing matters here.** §1–§4 (design, occupancy, model, predictions) were written
on 2026-10-17 **before** any of the new kernels were timed. The only numbers used to build
the model are the smoke-test figures for `allpairs_tiled` (`results/2026-10-17_smoke`). The
sanity outcome is in §6 and in `results/2026-10-17_architect_sanity/summary.md`. The
predictions are kept as written, including the wrong ones.

---

## 0. Common ground

### 0.1 What one interaction costs on sm_89 (SASS, static) [M]

`cuobjdump -sass` of `allPairsTiledKernel` (unroll 8, CUDA 13.4, `-O3`, no `-use_fast_math`).
Per interaction the loop body issues:

| Instr | per interaction | pipe |
|---|---|---|
| FADD (r = xj − xi) | 3 | FP32 |
| FFMA (d² = r·r + eps²) | 3 | FP32 |
| FSETP + 2 × predicated FMUL | 1 + 2 | ALU + FP32 (the **non-FTZ denormal guard of `rsqrtf`**) |
| MUFU.RSQ | 1 | XU (SFU) |
| FMUL (m · inv³) | 3 | FP32 |
| FFMA (a += r·s) | 3 | FP32 |
| LDS.128 (tile[k], broadcast) | 1 | MIO |
| loop overhead (IADD3/ISETP/BRA ÷ 8) | ≈ 0.6 | |
| **total issue slots** | **≈ 17.6** | |

FP32 instructions: 14 of 17.6. MUFU: 1 of 17.6.

Throughputs per SM per clock for cc 8.9 [N:nvidia2026bestpractices §12.1.1 Table 5, column
"8.6 / 8.9", verified by literature-scout 2026-10-17. The table is in the Best Practices Guide,
not the Programming Guide]: 128 FP32 FMA/clk, 16 rsqrt/clk. 4 schedulers, 1 per partition
[N:nvidia2023adawhitepaper p.10]. The issue rate of 1 warp-instruction/clk per scheduler is
**not stated** in any source found (TODO(source: scheduler issue rate)); it is the standard
assumption, and the probe below is consistent with it.
- Issue ceiling: 4 / 17.6 warp-interactions/clk = **7.3 interactions/clk/SM** [D].
- MUFU ceiling: 16 interactions/clk/SM. A warp-wide MUFU takes 8 cycles on its
  partition's quarter-SFU, against ≈ 17.6 issue cycles per warp-interaction. The XU pipe
  is therefore ≈ 45% busy at the issue ceiling [D].
- FP32 ceiling: 128 / 14 = 9.1 interactions/clk/SM [D].

> **Round-2 status of P0** (critic #2; probe SASS + clock log, `results/2026-10-17_round2/probe/`):
> the honest wording is **"issue-limited with a significant XU co-limit"**. Static SASS per
> interaction (probe, full-unroll b128): base 17.38 slots, ftz 14.38 (−3), mufu2 19.38 (+2:
> MUFU + FSEL), fmul3 20.38 (+3). Measured ratios (power-capped run, 2610–2640 MHz, two passes):
> ftz 0.90, mufu2 1.19, fmul3 1.185. Against the pure-issue model (0.827 / 1.115 / 1.173), only
> fmul3 fits. mufu2's +7 pp excess is the XU (XU ≈ 83% busy in mufu2 vs 46% in base [D]).
> ftz's shortfall says a non-issue limiter (latency, dependencies or register banks) appears
> once the guard is gone. ncu is needed to name it. The 1 inst/clk/scheduler issue rate is
> still unsourced. The text below is the original pre-registration, kept as written.

**Prediction P0 (MUFU hypothesis from the smoke test):** *wrong*. The kernel is
**issue-bound with FP32 the dominant pipe**, not MUFU-bound. In Nsight Compute I expect
`Issue Slots Busy` at ≈ 70–80%, the FMA/FP32 pipe the hottest, and the XU pipe ≈ 30–40%.
Corollary, not implemented: compiling the force kernels with `-ftz=true` removes the 3-instruction
denormal guard (17.6 → 14.6 slots), which should be worth **≈ +15–20%** at large N. That
changes numerics for denormal d², which never happens with eps = 0.01 (d² ≥ 1e-4). It is an
R2 (numerics) decision for Toncho, so it is not done here (see §7).

### 0.2 Calibrating a per-SM rate from the smoke test [D]

The smoke data fits one simple model (M1): *an SM is a server with a fixed rate r, and the
kernel takes as long as its busiest SM*.

    T = W_max / r + t0,   W_max = ceil(B / 80) · bodies_per_block · N_pad · (1 + c·q / tile)

B = grid blocks, N_pad = N rounded up to the tile, t0 = launch and event overhead,
c = per-tile overhead (load + store + 2 barriers) in interaction units.

| smoke point | B | busiest SM work | measured | implied r (G/s/SM), t0 = 4 µs |
|---|---|---|---|---|
| tiled N=131072, b256 | 512 | 7 × 256 × 131072 = 235 M | 17.21 ms | 13.6 |
| tiled N=16384, b256 | 64 | 1 × 256 × 16384 = 4.19 M | 0.323 ms | 13.1 |
| tiled N=10000, b256 | 40 | 1 × 256 × 10240 = 2.62 M | 0.196 ms | 13.6 |
| tiled N=4096, b256 | 16 | 1 × 256 × 4096 = 1.05 M | 0.075 ms | 14.8 |

**r ≈ 13.6 G interactions/s per SM.** It holds whether the SM runs 1 block (8 warps) or 7
blocks (48 warps resident). This is assumption **A1**: *8 warps (2 per scheduler) already
saturate an SM for this kernel*. It means the kernel is throughput-bound, not latency-bound,
from 8 warps upward. **Below 8 warps** I assume the rate scales linearly, r · w/8. A 64-thread
block is 2 warps and occupies only 2 of the 4 schedulers.
At 2.6–2.7 GHz, r = 13.6 G/s is ≈ 5.0–5.2 interactions/clk/SM, or ≈ 70% of the 7.3 issue
ceiling above. Consistent with P0.

The important consequence: **utilization is set by how blocks land on SMs, not by
occupancy.** The kernel's efficiency is (B/80) / ceil(B/80), the wave-quantization (tail)
factor, as long as each busy SM holds ≥ 8 warps.

---

## 1. R3 #1 — `allpairs_tiled` (baseline, existing)

One thread per body, tile = block, two `__syncthreads` per tile [P:nyland2007nbody]. Perfectly
balanced *per thread* (every thread does N interactions). It is **not** balanced *per SM*:
B = N/256 blocks over 80 SMs (§3.3). Kept unchanged as the reference. The pairwise function
moved to `interaction.cuh` so that every variant computes bitwise-identical terms. The
SASS and the register count (40) are unchanged.

---

## 2. Memory traffic (all all-pairs variants) [D]

- Positions: N × 16 B (1 MB at N = 65536). They sit in the 64 MB L2 for the whole kernel, so
  DRAM traffic ≈ one read of pos plus one write of acc per step. **Not memory-bound.**
- L2 → SM traffic: every block streams all N positions once: B × N × 16 B.
  Per interaction that is **16 · q / tile bytes**. tiled/tuned (q = 1, tile = block): 0.06 B (b256)
  to 0.25 B (b64). multi at b256: 0.125 B (q = 2) … 2 B (q = 32).
  At ~1 T interactions/s, q = 32 needs ≈ 2 TB/s of L2 → SM bandwidth. The AD103 L2 bandwidth
  is not in our sources: TODO(source: AD103 L2 bandwidth, e.g. Ada whitepaper or a
  microbenchmark). **Risk: q = 32 at large N may be L2-bound.**
- Shared memory per interaction: one LDS.128 per thread. tiled/tuned: all lanes read the same
  address (broadcast, 1 wavefront). multi: see §4.2.

---

## 3. R3 #2 — `allpairs_tuned`: block × unroll sweep

### 3.1 Decomposition
Identical to `allpairs_tiled`: one thread per body, tile = block, same summation order. So
results are **bitwise equal** to naive and tiled. Block size BLOCK and unroll U are
**template parameters** (`kernels_tuned.cu`):
- BLOCK ∈ {64, 128, 256, 512, 1024}, U ∈ {1, 4, 8, 16, 0 = full tile}: 25 instantiations.
- The tile loop has a compile-time trip count, so `#pragma unroll U` is real. The tiled
  kernel's loop bound is the runtime `blockDim.x`, so it gets a remainder ladder (visible in its SASS).
- `__launch_bounds__(BLOCK)` lets ptxas budget registers for that block size. Shared memory
  is static (BLOCK × 16 B).
- One registry entry (`allpairs_tuned`). Dispatch is from `Params::blockSize` × `Params::unroll`
  (`--block`, `--unroll`). The registry stays readable: the viewer's keys 1–4 are naive,
  tiled, tuned, multi.

**Load-balancing problem addressed:** two levels. (a) *Inside an SM*: does a configuration
fill the 48 warp slots, the 24 block slots and the 4 schedulers (occupancy)? (b) *Across SMs*:
B = ceil(N / BLOCK) blocks spread over 80 SMs. Smaller blocks are finer-grained work items,
which shrinks the tail.

### 3.2 Occupancy table (sm_89) [M regs from ptxas `--resource-usage`; D occupancy]

Registers per warp are allocated in units of 256 (regs × 32 rounded up to 256). Limits per SM:
65,536 registers, 48 warps, 24 blocks, 100 KB smem (+1 KB reserved per block).
Runtime-confirmed values (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, printed by the bench
as `resident N blocks/SM`) are in §6.

| BLOCK | U=1 | U=4 | U=8 | U=16 | U=full | regs-limited warps | blocks/SM | occupancy | limiter |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 19 | 37 | 40 | 40 | 40 | 85 / 51 (84 / 48) | **24** | 100% | **24-block limit** (24 × 64 = 1536, exactly) |
| 128 | 19 | 37 | 40 | 40 | 40 | 85 / 51 (84 / 48) | 12 | 100% | warps (48) |
| 256 | 19 | 37 | 40 | 40 | 40 | 85 / 51 (84 / 48) | 6 | 100% | warps; at 40 regs also registers (per partition: exactly 12 warps × 1280 = 15,360 ≤ 16,384) |
| 512 | 19 | 37 | 40 | 40 | 40 | 85 / 51 (84 / 48) | 3 | 100% | warps |
| 1024 | 20 | 37 | 53 | 56 | 56 | 85 / 51 / 36 (84 / 48 / 36) | **1** | **67%** | **thread limit: 1536 / 1024 = 1.5 → 1 block**, regardless of registers |

**Correction (round 2, critic #11).** The first version of this column said "102 warps" at
19 regs. At 19 regs a warp needs 19 × 32 = 608 registers, rounded up to 768, and
65,536 / 768 = **85** warps from the whole-SM pool, not 102. **The register file is
allocated per SM sub-partition** (4 × 16,384 registers, one per scheduler), and a warp's
registers come from its own partition [D from M]. That is the model the runtime follows. b64
q32 at 44 regs (1536 per warp) gives floor(16,384 / 1536) = 10 warps per partition = 40 warps
= **20 blocks** (83%). The runtime occupancy API returns 20; the whole-SM pool would give
floor(65,536 / 1536) = 42 warps = 21 blocks (§6, round-1 summary). The values in brackets are
per partition (× 4): 84 / 48 / 36 warps at 19–20 / 37–40 / 53–56 regs. None of them changes a
blocks/SM entry above, because every block size is limited by warps, blocks or threads first.

Shared memory never limits: the worst case is 24 × (1 KB + 1 KB) = 48 KB (b64), and b1024 needs 17 KB.
**BLOCK = 1024 is the only configuration below 100%.** Its cause is not registers: 1536 is
not a multiple of 1024. By A1 that is harmless, since 32 resident warps is ≥ 8.

### 3.3 Tail effect: N / BLOCK blocks over 80 SMs [D]

With ≥ 8 warps per busy SM, time ∝ ceil(B/80). Efficiency = (B/80) / ceil(B/80):

| N | b64 | b128 | b256 | b512 | b1024 |
|---|---|---|---|---|---|
| 4096 | B=64 → 80% SMs, **2 warps/SM** | 32 → 40%, 4 warps | 16 → 20% | 8 → 10% | 4 → 5% |
| 10000 | 157 → 98% (2 blk/SM) | 79 → 99% | 40 → 50% | 20 → 25% | 10 → 12.5% |
| 65536 | 1024 → 12.8/13 = **98%** | 512 → 6.4/7 = **91%** | 256 → 3.2/4 = **80%** | 128 → 1.6/2 = **80%** | 64 → **80%** |
| 131072 | 2048 → 99% | 1024 → 98% | 512 → 91% | 256 → 80% | 128 → 80% |

> **Round-2 qualification of M1 (measured, `results/2026-10-17_round2`, timeline method D-011).**
> The ceil(B/80) step is an idealization [P:zhang2026wavetune]. The tail penalty is at most one
> wave, and block-latency variance plus greedy dispatch smooth the steps. Our timelines bound
> where it holds on this GPU:
> - *Single-wave launches (all B blocks co-resident):* the step is **exact**. tiled b256 at
>   N = 65536 had 16 SMs × 4 blocks and 64 × 3, a 23.5% tail, SM efficiency 0.81 (M1: 0.80).
>   tuned b64 had SM efficiency 0.97 (M1: 0.98). Every B ≤ 80 launch got exactly one block per
>   SM.
> - *Multi-wave launches (B > resident slots, dynamic refill):* the step is **smoothed far
>   below one block**. multi q4 at 65536 (1024 blocks, 480 slots) had a 0.1% tail. This is
>   the smoothing WaveTune describes.
> - The **P2a miss is not a step-model failure.** b64 removes the tail exactly as predicted
>   (0.97 vs 0.81 SM efficiency), but each b64 SM is ≈ 12% slower per interaction (4× the
>   barriers and tile loads). So M1's per-block cost term (c = 2 interaction units per tile) is
>   too small for block 64. M1 is kept as a *structural* model (which SMs idle, how long the
>   tail is) and is not used to rank block sizes. Block sizes are ranked by measurement (D-012).
>
> Supporting 2025–26 work for this technique: wave quantization is a first-order loss, fixed
> by refining the tail wave [P:zhang2025hytis, transferable: GEMM tiles, full text not read];
> occupancy-aware launch decisions on an RTX 4090 [P:shi2025libra, transferable: sparse].

Two failure modes in one table. **Starvation**: B < 80, idle SMs, the small-N problem that R3 #3
targets. **Wave quantization**: B slightly above a multiple of 80, so one extra block per SM is
needed on some SMs and the rest sit idle at the end. When all B blocks fit at once
(B ≤ 80 × blocks/SM), "waves" in the classic sense do not appear. The quantum is
ceil(B/80) *blocks per SM*, because the resident blocks of one SM share its single rate r.

### 3.4 Predictions, R3 #2 (written before measuring)

Model M1, r = 13.6 G/s/SM, t0 = 4 µs, c = 2, and below 8 warps r · w/8.

**N = 65536, U = 8, by block size** (busiest-SM work → time → rate):

| block | W_max | predicted ms | **predicted G int/s** |
|---|---|---|---|
| 64 | 13·64·65536·1.031 = 56.2 M | 4.13 | **1040** |
| 128 | 7·128·65536·1.016 = 59.6 M | 4.39 | **980** |
| 256 | 4·256·65536·1.008 = 67.6 M | 4.97 | **865** |
| 512 | 2·512·65536 = 67.4 M | 4.96 | **865** |
| 1024 | 1·1024·65536 = 67.1 M | 4.94 | **870** |

**P2a: at N = 65536, block 64 beats the default block 256 by ≈ 20%,** purely from the tail
(98% vs 80% SM efficiency). This happens even though block 64 has 4× the barriers per
interaction and is "24-block-limited". Order: 64 > 128 > 256 ≈ 512 ≈ 1024.

**N = 4096, U = 8:** b64 (64 blocks, 2 warps/SM → r/4) ≈ b128 (32 blocks, r/2) ≈ b256
(16 blocks) ≈ **205 G/s** each; b512 ≈ 105; b1024 ≈ 55. **P2b: no block size rescues small
N.** Smaller blocks spread over more SMs but give each SM too few warps. Under the linear
r·w/8 rule this cancels exactly. If 4 warps per SM do better than half rate, b128 wins, and
that would be a finding about A1.

**N = 10000, U = 8:** b128 (79 blocks, 1/SM, 4 warps → r/2): 128·10240 = 1.31 M / 6.8 G/s
+ 4 µs ≈ 197 µs → ≈ 505 G/s. b64 (157 blocks, 2 per SM = 4 warps, r/2): the same 1.31 M →
≈ 505. b256 (40 blocks, 8 warps, r): 2.62 M / 13.6 → ≈ 505 (smoke: 511). **P2c: b64 ≈ b128
≈ b256 ≈ 505 G/s at N = 10000.** Smaller blocks fix SM coverage exactly as much as they lose
per-SM rate under the linear rule. Any win for b64/b128 here measures how far below-8-warp
SMs beat r·w/8 (a direct test of A1's lower edge).

**Unroll at N = 65536, b256:** U=1 ≈ −15% (+3 loop instructions per interaction ⇒ 20.6 vs
17.6 slots); U=4 ≈ −3%; U=8 ≡ reference; U=16 and full ≈ +0–2% (fewer loop instructions,
and the full unroll drops the loop entirely). **P2d: at b1024, U=full is ≥ 5% slower than U=8**
because of instruction-cache pressure (1024 unrolled interactions ≈ 18 K instructions ≈
280 KB of SASS). I hold this one with low confidence.

**Control:** `tuned` b256 U8 vs `tiled` b256: within ±3%. Same algorithm. The only difference
is the compile-time trip count (no remainder ladder).

---

## 4. R3 #3 — `allpairs_multi`: q threads per body

### 4.1 Decomposition
2D block `dim3(Q, P)`, BLOCK = P·Q threads: P bodies × Q threads per body
[P:nyland2007nbody, small-N section: "We therefore increase the number of active threads"].
Thread (s, b) owns body i = blockIdx.x·P + b and the j-slice j ≡ s (mod Q). Each tile is still
BLOCK bodies (one `float4` load per thread). It is consumed by P bodies, each thread doing
BLOCK/Q interactions per tile. Grid = ceil(N/P) = Q × the tiled grid.
After the j-loop, the Q partial sums of a body are combined by a butterfly of
`__shfl_xor_sync` (log2 Q steps × 3 components). Lane s = 0 writes `acc[i]`.
Next-tile register prefetch: the global load for tile t+1 is issued before tile t is
consumed, so the L2 latency overlaps compute. This matters when BLOCK/Q is small (8 at Q=32).
Q ∈ {1, 2, 4, 8, 16, 32} × BLOCK ∈ {64 … 1024}: 30 instantiations. `Params::threadsPerBody`
(`--q`, 0 = auto).

Related 2025–26 work: splitting the work of the partial wave into finer pieces recovers the
idle SMs [P:zhang2025hytis, transferable]. Splitting only heavy items above a threshold, with an
occupancy-aware launch, is the same trade-off of split cost against balance
[P:shi2025libra, transferable: sparse SpMM on an RTX 4090]. Libra combines the pieces with
atomics. Our register-only shuffle butterfly (§4.2) has no atomic cost and is deterministic.
Prefetch is a separate template switch since round 2 (`Params::prefetch`), so its effect is
measured apart from q (§8.2).

**Load-balancing problem addressed:** *starvation across SMs* at small N (B = N/BLOCK < 80), and
more generally *wave quantization*. Q multiplies the number of work items (blocks) without
changing the per-SM rate, so B/80 can be tuned close to an integer.

### 4.2 Reduction choice: warp shuffles, with lane-major slices
Two options:
- **(A, chosen) lane-major.** `threadIdx.x = s`, so a body's Q threads are Q consecutive lanes of one
  warp (Q ≤ 32). The reduction is a register-only butterfly: 3·log2 Q SHFL + FADD, no shared
  memory, no extra `__syncthreads`, executed once per body (negligible against N/Q
  interactions). **Cost:** in the inner loop a warp reads Q *different* tile entries per step
  (the lanes with the same s across the 32/Q bodies share one). An LDS.128 of Q distinct
  consecutive `float4` needs max(1, Q·16 B / 128 B) wavefronts: 1 for Q ≤ 8, 2 for Q=16,
  **4 for Q=32**. The four schedulers each issue ≈ 1 LDS per 17.6 cycles, so shared-memory demand ≈
  4·w/17.6 of the 128 B/clk port: 23% (Q≤8), 45% (Q=16), **91% (Q=32)**. Q=32 is a
  co-bottleneck.
- **(B, not built) slice-major.** The Q threads of a body sit in different warps, so every lane of a warp
  reads the same tile entry (broadcast) and the LDS cost stays 1 wavefront. The price is a
  shared-memory reduction (P·Q·3 floats + a barrier) and P = 32 bodies minimum per warp row.
  Worth building only if Q=32 is measured to be LDS-bound. It is an open item.

### 4.3 Occupancy (sm_89) [M regs from ptxas; D occupancy]
40 registers for every (BLOCK ≤ 512, Q) except b64/Q32 (44 regs), and b1024 (55 regs, all Q).
Shared memory: BLOCK × 16 B, as for tuned. So: 100% for b64…b512 (b64/Q32: 44 regs →
1536 regs/warp → 42 warps by registers → 21 blocks → **87.5%**, to be confirmed by the
runtime API), 67% for b1024 (thread limit).

> **Correction (2026-10-17, cuda-architect, tree round 2; tech-writer finding):** the runtime gave
> **20 blocks/SM (83%)** for b64/q32, not 21 (87.5%). The SM-wide count above is wrong because
> registers are allocated per sub-partition (§3.2): floor(16,384 / 1536) = 10 warps per partition
> × 4 = 40 warps = 20 blocks of 2 warps. The text above is left as written.

### 4.4 Choosing q automatically
Two heuristics, implemented and compared:
1. **Thread saturation** (the simple rule): the smallest Q with N·Q ≥ 80 × 1536 = 122,880
   resident threads. It gives Q = 32 (N ≤ 3840 and 4096), 16 (10000), 2 (65536).
2. **Cost model (implemented as `autoThreadsPerBody`)**: M1 directly. For each Q, W(Q) =
   ceil(B/80) · P · (1 + 2Q/BLOCK) · max(1, 8/warps_on_busiest_SM). Take the minimum and
   break ties within 2% toward smaller Q. At BLOCK 256 it gives **Q = 16 (N=1000), 4 (4096),
   2 (10000), 4 (65536)**.
The rules disagree at small N. Rule 1 counts *threads*, but per A1 an SM is saturated by 8
warps, not 48. So filling all 48 slots buys nothing and costs per-tile overhead and LDS
wavefronts. **P3e: the cost model's Q is within 5% of the best measured Q at every N, and the
thread-saturation rule loses at N = 4096 (Q=32 ≥ 10% slower than Q=4).**

### 4.5 Predictions, R3 #3 (written before measuring), block 256

| N | Q=1 | Q=2 | Q=4 | Q=8 | Q=16 | Q=32 | auto → | vs tiled |
|---|---|---|---|---|---|---|---|---|
| 1000 | 43 | — | — | 152 | **186** | 182* | Q=16, **≈ 185** | ≈ 4× |
| 4096 | 205 | 389 | **700** | 685 | 655 | 670* | Q=4, **≈ 700** | ≈ 3.1× (224) |
| 10000 | 505 | **980** | 965 | 940 | 895 | 805* | Q=2, **≈ 980** | ≈ 1.9× (511) |
| 65536 | 865 | 980 | **1040** | 1010 | 950 | 865* | Q=4, **≈ 1040** | ≈ 1.2× |

(G interactions/s. * = before the Q=32 LDS/L2 co-bottleneck, which I expect costs a further 10–20%.)

Plain statements:
- **P3a: N = 4096, Q = 4 (and 8) reaches ≈ 700 G/s (accept 550–850), ≈ 3× tiled.** Reason: 64
  blocks on 64 SMs, each SM with its 8 warps at the saturated r.
  Ideal = 16.8 M interactions / (80 · 13.6 G/s) + 4 µs = 19.4 µs → 865 G/s. Q = 4 loses 20% to
  16 idle SMs, and Q = 8 loses the same to 2-vs-1.6 quantization.
- **P3b: N = 1000 is launch-overhead-bound.** The kernel itself is ≈ 1.4 µs against t0 ≈ 4 µs, so
  the absolute rate is a weak prediction (120–300 G/s). The ratio to Q = 1 (≈ 4×) is the robust part.
- **P3c: N = 65536: multi Q=4 ≈ tuned b64 ≈ 1040 G/s.** They are the same fix (finer work
  items) by different routes. Q=4 keeps a 256-wide tile, so it has fewer barriers.
- **P3d: Q=1 multi vs tiled within ±3%.** The difference is the prefetch only.
- **Q = 32 at N = 65536 is the slowest multi config** (LDS 4 wavefronts + 2 B/interaction of L2).

### 4.6 Validation tolerance (decided before measuring)
Changing Q changes the summation order, so results are no longer bitwise equal to naive.
- FP32 unit roundoff u = 2⁻²⁴ ≈ 6.0e-8. For a recursive sum of N terms with partial sums
  growing roughly linearly, the typical (not worst-case) relative error is ≈ u·√(N/3) ×
  (Σ|f| / |a|) [D, standard random-rounding argument; TODO(source: a numerical-analysis text,
  e.g. Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4, not yet in
  references.bib)]. A Q-split sum has Q shorter chains, so it is *more* accurate than naive. The
  naive−variant difference is therefore dominated by naive's own error.
- **κ_i = Σ_j |f_ij| / |Σ_j f_ij|** is the condition number of body i's sum. In a Plummer
  sphere, bodies near the centre feel pulls from all directions that nearly cancel
  (|a| ∝ r for r ≪ a). κ can reach 10²–10³ there. The *absolute* error stays ≈ u·√N·Σ|f|, but
  dividing by a tiny |a| inflates the *relative* error by κ. **That is why max-rel spikes on a
  few bodies while the force field is fine.** It is a property of the problem, not of the
  kernel. The same body spikes against FP64 for naive too.
- **Superseded by D-009 (round 2).** The rms-based G1 below is κ-inflated (rms-rel divides
  by |a_i|, like max-rel). The thresholds sat 6–8× above the measured noise, and a dropped body
  would pass them. The current gates are normMax ≤ 1.0·u·√N (calibrated to naive's measured
  0.44–0.68), normMax ≤ 1.5 × naive's, and max-rel ≤ 1e-2, all evaluated in the bench (exit code
  3). Missed interactions are caught by the exact `--selftest`. Original text kept:
- **Gates (PASS):**
  - **G1, primary: rms relative error vs naive ≤ 4·u·√N** (N=1000: 7.5e-6; 4096: 1.5e-5;
    10000: 2.4e-5; 65536: 6.1e-5).
  - **G2: normalised max error |Δa| / Σ|f| ≤ 4·u·√N** vs the FP64 sample. This is the
    max-error metric that is *not* inflated by cancellation.
  - **G3: FP64 rms of the variant ≤ 1.5 × FP64 rms of naive** (reordering must not make the
    answer worse).
  - max-rel is **reported, not gated**. A sanity ceiling of 1e-2 catches gross bugs (a missing
    tile shows up as O(1/tiles) everywhere, so rms catches it too).
- **Predicted values:** rms vs naive ≈ 1–3e-6 at N = 4096, ≈ 0.5–1e-5 at N = 65536. max-rel
  10⁻⁴–10⁻³ at N = 65536 (κ_max ~ 10²). FP64 rms of multi ≤ naive's. Energy drift over 2000
  steps at N = 4096: same as tiled (−2.8e-8) to within ±1e-8. The drift comes from the
  integrator (KDK, dt = 1e-3), not from summation-order noise.
- The smoke test's suggested max-rel < 1e-4 is too tight for N ≳ 10⁴. Naive itself fails it
  against FP64 on the cancellation bodies.

The bench now prints, for every run: error vs naive (max, rms), and vs an **FP64 host
reference** on 512 evenly spaced bodies plus the 16 smallest-|a| bodies (rms, max, normalised
max, κ_max), for both the variant and naive.

---

## 5. Metrics that would prove each claim (for bench-profiler)
| Claim | Metric |
|---|---|
| A1 (8 warps saturate an SM) | ncu `sm__inst_issued.avg.pct_of_peak_sustained_active` at 1 vs 6 blocks/SM; ratio of G/s per busy SM |
| P0 issue- not MUFU-bound | ncu pipe utilization: FMA vs XU; `Issue Slots Busy` |
| Tail (§3.3) | per-SM active cycles spread (`sm__cycles_active` min/max/avg); nsys kernel time vs ceil(B/80) |
| Starvation fix (§4) | `sm__cycles_active.avg / max` → 1, achieved SM coverage = min(B,80)/80 |
| Q=32 LDS co-bottleneck | `l1tex__data_pipe_lsu_wavefronts_mem_shared` per LDS, shared-pipe utilization |

---

## 6. Outcome (sanity runs, not the official benchmark)

Raw data: `results/2026-10-17_architect_sanity/` (`sweep.csv`, `sweep_oos.csv`, `energy.txt`,
`validation_oddN.txt`, `mufu_probe.cu` + `mufu_probe_out.txt`). Environment: 272 W cap, GPU idle
before the runs, SM clock median 2760 MHz under load. Sanity only, not the official benchmark.
The predictions in §0–§4 above are left exactly as written.

**Scorecard (short):**
- **P0: partial HIT** (revised in round 2, see the note in §0.1). It is not MUFU-*bound*, but it
  is "issue-limited with a significant XU co-limit". ncu was blocked (ERR_NVGPUCTRPERM), so a
  probe was run instead: +2 slots incl. 1 MUFU = +19% (issue model +11.5%), +3 slots (FFMA + 2 FMUL)
  = +18.5%, −3 slots (`rsqrt.approx.ftz`) = −10%. The `-ftz` gain is
  half of what I predicted (+10–11% vs +15–20%).
- **A1 wrong.** The per-SM rate is linear up to ~6 warps (≈ 2.1 G/s per warp), then saturates:
  8 warps ≈ 14 G/s, 32 warps ≈ 17 G/s. Consequences:
  - The tail effect (P2a) is real but smaller: b128 is best at N=65536, +8% over b256, not +20%.
  - P2b and P2c miss: b128 beats b256 at N = 4096 and 10000 by 23–24%.
  - Absolute rates are 4–26% above the model, which was calibrated on the loaded smoke-test GPU.
- Unroll: u1 −21%, u16 +5%, full +8% (b ≤ 256). Full unroll at b1024 is −18% (P2d HIT, I-cache).
- **P3a HIT.** N=4096 multi q4 = 771 G/s (predicted 700), best 893. That is 3.9× tiled b256,
  but **2.56× the best-tuned 1-thread/body baseline** (tuned b128 full, 349 G/s), which is the
  figure to quote (rule 8; §8.3 has every N).
  Validation: tuned is bitwise equal to naive. multi passes G1–G3 at every N, q and block,
  including odd N, and is **more accurate than naive vs FP64** (error ∝ 1/q).
  Energy drift: −2.76e-8 (tiled/tuned), −2.97e-8 / −3.14e-8 (multi q4 / q32).
- **P3e MISS at b256.** auto loses 7–9% at N = 4096 and 10000. Over all (N, block) cells, the cost
  model still beats the thread-saturation rule: mean loss 1.8% / 0.9% (in- / out-of-sample) vs
  6.8% / 4.8%. A v2 model fitted to the measured r(w) did better in-sample and worse
  out-of-sample, so it was rejected. **Withdrawn in round 2 (D-010):** the round-1 v2 code was
  never saved. A reconstruction (`tools/autoq_models.py`) gives 1.0% / 0.6%, better than v1
  out-of-sample, so the rejection reason is not supported.
- Methodology: small-N timings need `--warmup-ms` and `--batch` (cold clocks and event jitter
  made them bimodal). Both flags are added to the bench and to `tools/sweep.ps1`.

## 7. Open items
- `-ftz=true` (or `rsqrt.approx.ftz` in `interaction.cuh`) for the force kernels: +10–11%
  measured by the probe. This needs Toncho's numerics decision.
- Search (block, q) jointly. The winners at N = 4096–20000 are ≈ one fat block (16–32 warps)
  per SM, with B ≈ 64–79.
- Unroll as a template parameter in multi (tuned shows +5–8% from u16 or full).
- tuned b256 u8 is 2–4% *slower* than tiled b256 despite the identical algorithm. **Likely
  explained (round 2):** ptxas schedule sensitivity. The instrumented tiled twin has the
  same opcode multiset in its loop but a different schedule and register assignment, and runs
  +2…+6% differently (§8.1). Confirming this needs the same opcode-sequence diff for tuned vs tiled.
- Enable GPU performance counters (admin / NVIDIA Control Panel), then run ncu to confirm P0 and
  separate the Q=32 LDS bottleneck from the L2 one.
- Check max-rel spikes on an evolved, centrally concentrated snapshot (κ_max was only 15 in the ICs).
- Slice-major multi (§4.2 B), if Q=32 is confirmed LDS-bound.
- Tile larger than the block (each thread loads 2–4 bodies per tile) to cut barriers at
  high Q.
- AD103 L2 bandwidth figure (§2). TODO(source).
- Unroll is still confounded in the R3 #3 gain: multi is u8, the baseline is full unroll (§8.2).
- Power cap: at 272 W the faster kernels run at lower SM clocks (2.52 vs 2.70 GHz at N = 65536).
  Report G interactions per clock next to G/s once the power-limit decision is made.

---

## 8. Round 2 outcome (2026-10-17)

Predictions: `docs/predictions/2026-10-17-round2.md`, committed as 9bae5fa before these runs.
Data: `results/2026-10-17_round2/` (environment controlled: idle 0% before and after, 272 W
cap reached under load). Scorecard: 25 rows, 18 HIT and 7 MISS (two of the misses better than
predicted).

### 8.1 Load imbalance, measured without ncu (critic #1)

Method (D-011): an instrumented twin of each kernel records `%smid` plus `%globaltimer` and
`clock64` at block entry and exit. The globaltimer ticks in 1.024 µs steps on this board, so
clock64 is anchored to tick edges before and after the launches. Raw and calibrated metrics
agree within about 1%. The twin's force loop gets a different ptxas schedule (same opcodes),
so its speed differs from production by −3.4…+6.2%. Structure (placement, idle SMs, tail
fraction) carries over; absolute time comes from the production kernels.

| N | config | idle SMs | max/mean SM busy (80 SMs) | SM eff (mean busy / span) | tail % |
|---|---|---|---|---|---|
| 4096 | tiled b256 | 64 | 5.02 | 0.20 | 0.6 |
| 4096 | tuned b128 full (best 1 thread/body) | 48 | 2.52 | 0.40 | 0.7 |
| 4096 | multi b512 q8 | 16 | 1.27 | 0.79 | 1.4 |
| 65536 | tiled b256 | 0 | 1.23 | 0.81 | 23.5 |
| 65536 | tuned b128 full | 0 | 1.09 | 0.91 | 13.8 |
| 65536 | multi b256 q4 | 0 | 1.016 | 0.98 | 0.1 |

N = 1000 and 10000 are in the results summary. Two imbalance regimes, both now measured:
- **Starvation (B ≤ 80):** exactly one block per SM, 80 − B SMs idle for the whole kernel.
  R3 #3 removes it.
- **Wave tail (B > 80, single wave):** the ceil(B/80) step, exact. R3 #2 (smaller blocks) and
  #3 (more blocks, multi-wave refill) remove it.

### 8.2 Prefetch vs q (critic #4)

At N = 4096, `multi` with prefetch off equals `tuned u8` within −2.8…+4.8%. Prefetch is
worth +16% / +10% / +8% / +0.5% at q = 1 for 2 / 4 / 8 / 16 warps per SM. At the best config
(b512 q8) it is worth +6.7%. The N = 4096 gain of 2.56× over the best-tuned baseline is
**q ×2.40 × prefetch ×1.067**. At N = 65536, prefetch is worth 0%.

### 8.3 Speedups vs the best-tuned baseline (rule 8; round-2 sweep)

Baseline = the best 1-thread/body config over tiled, tuned (all block × unroll) and multi q = 1:

| N | baseline (G/s) | best R3 #3 (G/s) | speedup |
|---|---|---|---|
| 1000 | tuned b128 full (75) | multi b256 q16 (180) | **2.40×** |
| 4096 | tuned b128 full (348) | multi b512 q8 (894) | **2.56×** |
| 10000 | tuned b128 full (850) | multi b1024 q8 (1188) | **1.40×** |
| 65536 | tuned b128 full (1244) | multi b128 q2 (1266) | **1.02×** |

### 8.4 Validation and energy

- **Gates (D-009):** all 4000 sweep runs PASS. The exact `--selftest` passes 1810 cases and
  catches injected missed-interaction bugs.
- **Long-run energy:** N = 4096, 100 time units. |ΔE/E| = 6.2e-7 at dt = 1e-3 and 1.1e-6 at
  dt = 5e-4: a *smaller* dt drifts *more*, so FP32 round-off dominates (D-014). multi and
  tiled are the same order of magnitude.

---

## 9. Design inputs for R3 #6–#8 (from literature-scout round 3, `references/R3-evidence-map.md`)

Recorded now so that the #6–#8 predictions can be pre-registered against them. These are not
results.
- **#6 has three arms; name the one implemented.** (a) Static count-based mapping of work to
  blocks, no runtime scheduler [P:li2025staticbatching, transferable]. (b) A persistent kernel
  with a precomputed balanced partition [P:xia2025voltrix, transferable]. (c) A persistent
  kernel where each SM claims the next item through `atomicAdd` on a global counter
  [P:huang2026weave, transferable; the claiming mechanism is identical].
- **#7 vs #6: pre-register the gain as small.** With an atomic-counter pool of fine tasks,
  most of the balancing is already done, and stealing is supplementary [P:chen2026gmatch, RTX 4090,
  graph DFS]. In an FMM traversal, stealing gave ≈ 1.5× in the traversal phase but ≈ 1.05×
  end to end [P:maeda2026gtap]. Report phase time and full-step time separately.
- **#8: prefer a phase split (tree on the CPU, forces on the GPU) over a body split, and expect
  modest gains.** The one direct 2026 gravity source for the phase split (OpenGadget3, Ragagnin et
  al.) is not yet in `references.bib`, so it is not cited here. Dynamic CPU+GPU work stealing gave +12.9% over GPU-only
  [P:xu2025tascell, transferable]. Same-kernel co-execution gave 1.31× over GPU-only
  [P:nozal2025coexecution, transferable, T2: full text not read]. The 6P + 8E hybrid CPU
  (01-hardware §1) has no 2025–26 source.

---

## 10. Barnes-Hut tree code: overview (R3 #4–#5)

Written 2026-10-17 by `cuda-architect`, **before any timing run of the tree kernels**. What was
observed while building it (one functional check at N = 8192, plus exit-code checks) is listed
in §0 of `docs/predictions/2026-10-17-tree-round1.md`, so the reader can tell which numbers
below are predictions. Code: `src/core/kernels_tree.cu`.

**Why a tree, and why here.** All-pairs is perfectly balanced per thread (every thread does N
interactions), so its only imbalance is SM-level (starvation and the wave tail, §8.1). A tree
code replaces the far field by cell monopoles, O(N log N) instead of O(N²)
[P:barnes1986hierarchical]. Its work per body depends on where the body sits, and the walk is
data-dependent control flow, so it is where intra-warp imbalance and divergence appear
[P:nagarajan2025rtbarneshut: tree traversal maps poorly onto ordinary GPU cores, motivation
only; P:dale2026warpdivergence, T2: k divergent paths serialize, efficiency ≈ 32/k].

**Pipeline (one `KernelInfo::launch` = one force evaluation, 5 timed phases, all on the GPU):**

| phase | kernels | work | notes |
|---|---|---|---|
| keys | bbox partial (160 × 256) + bbox final (1 block) + Morton key | 2 reads of pos, 8 B/body written | no host sync: the key kernel reads the box from device memory |
| sort | `cub::DeviceRadixSort::SortPairs` (63 key bits, 32-bit index) + gather | ~8 radix passes over 12 B/body; gather 32 B/body | CUB ships with CUDA 13.4 (`include/cccl`), no download |
| tree | memset + radix-tree kernel (N−1 threads) + bottom-up multipole (N threads) | O(N) | binary radix tree over the sorted keys |
| order | nothing (#4, #5 Morton) / Hilbert key + second sort (#5 Hilbert) | | the *only* place the variants differ |
| traverse | one thread per body, private stack | O(N log N) | the R3 #4/#5 kernel |

The build is the same for every variant: the tree does not depend on the thread order. The
build stays GPU-resident: positions never leave the device (one copy engine, 01-hardware
§2.2), and 2025–26 tree codes keep the tree on the GPU [P:stucker2026jzfmm: sorting, tree
build and traversal all GPU-side; P:frontiere2025crkhacc: "GPU-resident tree solvers"]. A CPU
build is the R3 #8 question and is not done here.

**TDR.** One force evaluation is about 10 kernel launches. The longest is the traversal. The
worst case tried (N = 1M, core IC, θ = 0.3, unsorted) completed without a TDR (exit code only;
its time was not looked at). The exact reference for validation (`referenceAcc`) is launched
in chunks of ≤ 2³⁶ interactions, so validation at N = 1M is also TDR-safe.

**Memory** at N = 1M [D]: keys 3 × 8 MB, indices 3 × 4 MB, sortedPos 16 MB, nodes
(com 16 + lo 16 + hi 16 + info 16 + rc² 4 B) × (N−1) = 68 MB, parents and arrival counters
12 MB: about 130 MB of 16 GB. The traversal reads com + rc² + info + sortedPos = 52 MB, which
fits in the 64 MB L2.

## 11. Build design

### 11.1 Keys
63-bit Morton keys, 21 bits per axis, in the bounding **cube** of the bodies (side = the largest
extent × 1.0001, so the maximum coordinate stays inside). Bits are interleaved by the standard
magic-mask expansion. 21 bits per axis gives cells of side/2²¹. For the Plummer box (side ≈ 12,
from the r ≤ 10 Plummer-radius cut in `makePlummer`) that is 6e-6, far below the softening
0.01, so key collisions only happen for (near-)coincident bodies, and those are handled
exactly (§11.2). The tree is built on the Morton curve [P:stucker2026jzfmm, direct: Morton sort
plus bottom-up build].

### 11.2 Binary radix tree
N−1 internal nodes over the sorted keys, one thread per node. The node's direction and range
come from the common-prefix length δ with its neighbours, and the split from a binary search on
δ. Equal keys are made distinct by appending the index (δ = 64 + clz(i ⊕ j)), so duplicates are
legal. Every node then covers a **contiguous range** [first, last] of the sorted array, which
makes leaf buckets free: a small node is just a range. This is Karras' parallel radix-tree
construction [TODO(source): T. Karras, "Maximizing parallelism in the construction of BVHs,
octrees, and k-d trees", HPG 2012; not in `references.bib`, so not cited]. Child encoding:
≥ 0 = internal node, < 0 = ~(sorted body index).

*Why binary and not an octree:* the radix tree is built in one O(N) kernel with no atomics and
no level-by-level passes, and its nodes are exactly Morton prefix cells (tightened to the
bodies' AABB below). The price is about 3× the depth of an octree (31–43 levels at N = 65536,
§0 of the predictions), so more node pops per interaction. That costs #4 and #5 the same and
does not affect their comparison. It is a constant-factor choice for #6/#7 to revisit.

### 11.3 Multipoles: monopole only (justified)
Bottom-up pass, one thread per leaf: walk to the parent and `atomicAdd` on the node's arrival
counter. The first child to arrive stops; the second builds the node from both children (mass,
centre of mass, tight AABB, opening radius) and continues upward. Node data written by other
threads is read with `ld.global.cg` after a `__threadfence`, so stale L1 lines are never used.
The COM is clamped into the AABB (rounding), so bmax below really bounds every body of the node.

**Why monopole only.** (1) A cell interaction is then *the same* `bodyBodyInteraction` as a
body interaction (the COM acts as a pseudo-body). "Interactions" is one work unit with one
instruction cost across the all-pairs and tree kernels, so per-warp work counts and interaction
rates compare directly with §0–§8, which is what R3 is graded on. (2) The truncation error is
acceptable for a load-balancing study: rms ≈ 0.7% at θ = 0.5 in the functional check
(predictions §0). A quadrupole would cut the error at fixed θ, or give the same error at a
larger θ with fewer interactions, at about +30 FLOP per cell interaction and +24 B per node.
That is an accuracy-per-cost question and is deferred. It changes the absolute speed of #4 and
#5 equally and the #5/#4 ratio hardly at all.

### 11.4 Opening criterion: bmax
Accept node c for body i iff |x_i − com_c|² ≥ (bmax_c / θ)², where bmax_c is the distance from
the COM to the farthest corner of the node's tight AABB, so every body of the node lies within
bmax_c of the COM. If the node is not accepted: a node with ≤ `leafSize` bodies (default 16) is
summed directly over its contiguous range, a larger node pushes both children, and a
single-body child is one body interaction.
- *Safety:* any point inside the node's box is within bmax of the COM, so for θ ≤ 1 a body
  inside a node's box always opens it. The classic size criterion s/d < θ can accept a cell that
  contains the body when the COM sits near an edge [TODO(source): Salmon & Warren 1994, not in
  `references.bib`].
- *Meaning of θ:* θ trades accuracy against cost [P:barnes1986hierarchical]. Our θ is defined
  on bmax, not on the cell side, so it is not numerically the classic θ. For a cube with the COM
  at its centre bmax = 0.87 s, so our θ ≈ 0.87 × the classic one; with tight AABBs and binary
  cells the mapping is only approximate. Every θ in this repo is a bmax-θ.
- *θ = 0* sets rc² = +∞: nothing is ever accepted, so the walk sums every body through the
  leaf buckets. That is an exact, reordered all-pairs sum, which links the tree to the D-009
  gates and to the exact lattice self-test (§12.3).
- *Error:* per accepted cell the error scales as (bmax/d)² ≤ θ², because the dipole term
  vanishes about the COM and the first neglected term is the quadrupole [D, standard multipole
  argument; no source in the bib]. Cell errors add with partial cancellation, so the per-body
  error should scale between θ² and θ³.

## 12. R3 #4 (`bh_naive`) and R3 #5 (`bh_sorted`, `bh_hilbert`)

### 12.1 Decomposition, and the one difference
One kernel, `bhTraverseKernel`: one thread per body, block = `--block` (128 for every
pre-registered run), a 64-entry private stack in local memory, no shared memory, no warp vote,
no cooperation. Thread t processes body `order[t]` and writes `acc[order[t]]`:

| registry name | R3 | `order[]` | extra cost |
|---|---|---|---|
| `bh_naive` | #4 | identity (written once; CUB never writes its inputs, checked) | none |
| `bh_sorted` | #5 | the Morton permutation the build already produced | none: the sort is part of the build |
| `bh_hilbert` | #5, sub-variant | Hilbert permutation (Skilling transpose [TODO(source): Skilling 2004, not in bib], second radix sort) | one key kernel + one sort |

**The only difference between #4 and #5 is the content of `order[]`.** Same kernel, same SASS,
same tree and the same per-body visiting order, so the three variants produce
**bitwise-identical accelerations** (asserted by `--selftest` for 3 ICs × 6 N × 2 (θ, leaf)
configurations). This is the methodological point of D-005: #4 must not contain #5's ordering
or #7's warp vote, which the classic GPU code combines [P:burtscher2011barneshut: in-order body
sort plus warp-vote traversal]. Note that the tree *build* sorts the bodies in every variant,
because the radix tree needs sorted keys. "Unsorted" in #4 means that the *threads* run in the
original (spatially random) body order; the leaf bodies the walk reads come from the sorted
array in all variants. So #4 isolates exactly what #5 changes: which bodies share a warp.

**Load-balancing problem addressed by #5:** intra-warp imbalance and divergence. In #4 a warp's
32 bodies are a random sample of the whole system. They open different nodes, their loops have
different lengths (the warp runs until its slowest lane is done), and each lane fetches a
different node (up to 32 separate sectors per load). In #5 a warp's bodies are spatial
neighbours [P:thaler2026sfcneighbor, transferable: Hilbert runs of consecutive particles are
spatially compact on GPUs; P:stucker2026jzfmm: Morton order], so they walk similar paths and
mostly read the same node (one sector, broadcast). The price: block costs now differ by region
(a block in the dense core is expensive, one in the halo is cheap), so #5 moves imbalance from
inside warps to between blocks. Whether the dynamic block scheduler absorbs it is a measurement
(§14, P-T7 in the predictions).
Ada-specific 2026 evidence that regrouping uneven one-thread-per-item work pays:
[P:galgoczi2026ser: about 5.1× from the reordering itself, RTX 4090, photon Monte Carlo via
OptiX SER; transferable]. Morton vs Hilbert for traversal divergence has no 2025–26 source
(evidence map §5), so our measurement is new.

### 12.2 Occupancy, traffic, expected bottleneck [M regs from ptxas; D for the rest]
- `bhTraverseKernel<false>` (production) and `<true>` (instrumented): **40 registers, 0 spills,
  256 B stack frame** (the local-memory stack). 40 regs → 1280 per warp → 12 warps per
  partition → 48 warps per SM. Block 128 → **12 blocks/SM, 100% occupancy** (runtime-confirmed:
  "resident 12 blocks/SM"). Local-memory reserve: 256 B × 1536 × 80 ≈ 31 MB.
- Build kernels: radix tree 19 regs, multipole 38 regs, no spills.
- Grid at block 128: 64 blocks (N = 8192), 512 (65536: a single wave, 960 slots), 2048
  (262144: 2.1 waves), 8192 (1M: 8.5 waves).
- Per node pop: 16 B com + 4 B rc² (+ 16 B info if opened), i.e. 2–3 separate 32 B sectors. Per
  leaf body: 16 B from a contiguous range. In #4 each lane of a warp touches different nodes, so
  one warp-wide load is up to 32 sectors. The top of the tree stays in L1 and the rest comes
  from L2 (the whole traversal working set fits in the 64 MB L2 up to N = 1M). In #5 most loads
  are warp-uniform.
- **Expected bottleneck:** not FP32 throughput. #4: L2 latency and bandwidth on scattered node
  loads, plus divergence. #5: issue slots lost to divergence (the leaf-bucket loop is run by the
  few lanes that reached a bucket while the others wait; P-T5 in the predictions). The AD103 L2
  bandwidth is still unsourced (§2 TODO), so the #4 memory ceiling can only be bounded by
  measurement, not computed.

### 12.3 Validation (pre-registered in the predictions file)
- **θ = 0:** the D-009 gates G0/G1/G2 apply unchanged (an exact reordered sum). Checked once in
  the functional run (N = 8192: PASS, normMax 3.0e-6 ≤ 5.4e-6).
- **θ > 0, tree gates** (`kTG1`, `kTG2` in `src/bench/main.cpp`), over all bodies vs the exact
  FP32 reference (bitwise `allpairs_naive`, chunked): **TG1** rms rel ≤ 0.03·(θ/0.5)²,
  **TG2** p99 rel ≤ 0.12·(θ/0.5)², and **zero stack overflows**. They are bug catchers, about 4×
  above the expected truncation error (§11.4). The expected error itself is a prediction and is
  scored separately. A FAIL exits with code 3. FP64-sample numbers are printed as well.
- **`--selftest` additions** (exact, all pass):
  1. Every `bh_*` kernel at θ = 0, blocks {64, 128, 256} × leaf {1, 8, 16, 64}, instrumented and
     not, on the dyadic lattice at the 10 selftest N: bitwise equal to the analytic sum (720
     cases; the lattice has many duplicate positions, so the duplicate-key path is exercised).
     `--selftest-drop J` fails all of them. An injected bug (the leaf loop skips its last body)
     failed exactly the 486 cases where it can be visible (leaf > 1 and n > 1).
  2. The Hilbert key on a 16³ sub-grid is a bijection with unit steps.
  3. `checkLastTree` for 3 ICs × N ∈ {1, 2, 3, 1000, 10007, 65536} × 2 configs × 3 variants:
     orders are permutations (identity unmodified), sortedPos = pos[perm], keys monotone, every
     body in exactly one leaf, every internal node reached once, ranges contiguous, masses
     additive, boxes nested, COM inside its box, root mass and COM conserved.
  4. #5 is bitwise equal to #4.
  5. No stack overflow.

## 13. Initial conditions (`--ic`, `nbody::makeIC`)
All have total mass 1 and equal-mass bodies (1/N), and are shuffled with the seed so the
original order is spatially random. Otherwise the components of a two-part IC would be
contiguous runs and #4 would be partly sorted.
- `plummer`: `makePlummer`, bitwise unchanged. Mildly concentrated.
- `collision`: two Plummer spheres, N/2 each, virial radius 0.25 (mass 0.5), at (∓1, ∓0.125, 0),
  approaching at ±0.35 along x. Two dense clumps in an empty box.
- `core`: a Plummer halo (3N/4, virial radius 1) plus a dense Plummer core (N/4, virial radius
  0.03, 33× smaller). Strongly clustered and bimodal: tree depth 43 vs 31 for the Plummer at
  N = 65536. Each component is internally in equilibrium but the whole is not, so this is a
  force snapshot, not an evolution test.

## 14. Imbalance metrics for #4 vs #5 (no ncu)
From the instrumented twin (`Params::workCounters`, one uint4 per traversal thread). The counts
are algorithmic, so they are identical for the production kernel.
- **I_t, V_t:** interactions (cells + bodies) and nodes popped by thread t.
- **Per-warp max/mean** = Σ_w max_{t∈w} I_t / Σ_w mean_{t∈w} I_t (the median over warps is also
  printed). A warp's loop runs until its slowest lane is done, so this is the intra-warp load
  imbalance; 1 = perfect.
- **SIMT efficiency** (as executed, from `__activemask()`): at every pop and every interaction
  site the lowest active lane increments a leader counter, so Σ leaders = warp-level steps
  actually issued. eff_inter = Σ I / (32 Σ leader_I), eff_pop = Σ V / (32 Σ leader_V). This is
  the mean-active-lanes metric without ncu. Caveat: the counting code may itself change where
  ptxas places reconvergence points, so it measures the twin as executed.
- **Warp-cost max/mean** = max_w max_t I_t / mean_w max_t I_t: the spread of warp cost, i.e.
  the imbalance #5 moves *between* warps and blocks.
- **SM level:** per-block timelines (D-011, `--timeline`, `tools/timeline_analysis.py`) of the
  traversal kernel: SM efficiency, max/mean SM busy, tail.
- `--tree-stats out.csv` writes one row per warp, for histograms.

## 15. Interface changes (backward compatible)
- `Params`: `theta`, `leafSize`, `workCounters`, `phaseEvents`. `LaunchInfo`: `approximate`,
  `nPhases`, `phaseNames`. The all-pairs kernels ignore the new fields, and no all-pairs kernel
  source changed.
- `KernelInfo::launch` still computes acc from pos, with the build inside it. Phase times come
  from events the launcher records when `phaseEvents` is set. The bench times the production
  pipeline without events, and the phases in separate launches with events.
- `referenceAcc` (bitwise `allpairs_naive`, chunked) replaces the single naive launch in the
  bench's validation for every kernel. `forceScale` is chunked the same way (same results).
- Bench: `--ic`, `--theta`, `--leaf`, `--tree-stats`. The CSV has 18 new trailing columns
  (older CSVs are refused for appending, as before). Viewer: `--ic`, `--theta`, `--leaf`; keys
  5–7 select the tree kernels.
- CUB needs MSVC's conforming preprocessor: `-Xcompiler=/Zc:preprocessor`, scoped to
  `kernels_tree.cu` only.

---

## 16. Tree round-1 outcome (2026-10-17)

Predictions: `docs/predictions/2026-10-17-tree-round1.md`, committed as 581fc47 before these
runs. Data: `results/2026-10-17_tree_round1/` (272 W cap, idle 0% before and after; SM clock
2.75 GHz during the traversal). The summary's content is in the architect's phase-2 report: the
harness blocked the agent from writing `summary.md`. Scorecard: **91 HIT / 32 MISS** of 123
scored items. §10–§15 above are left as written.

> **Correction (2026-10-17, tree round 2):** the 91 / 32 scorecard is **disputed** (critic review 2,
> #4): no item table was committed, post-registration interpretation #2 moved a goalpost (P-T1a
> monotonicity is a MISS on set D), splitting P-T6d inflated the count, several P-T5 HITs restate
> §0.4 pre-registration observations, and P-T6c recomputes to 1.11 < 1.3 (a MISS). See
> `results/2026-10-17_tree_round1/summary.md` §4 for the annotated status. From round 2 on, the
> scorecard is a committed item table (§17.8).

### 16.1 What held and what didn't
- **Held (relative structure):**
  - IC ratios of work.
  - Every imbalance metric's direction and most of their ranges.
  - Error ∝ θ^2.13 and work ∝ θ^−2.52.
  - Crossover vs the best all-pairs: ≈ 39,700 sorted, ≈ 59,600 naive.
  - The timeline structure: #5 tail at single wave, absorbed by refill at 1M.
  - The θ = 0 control: #4 = #5 within 0.3%, SIMT efficiency 1.000.
- **Missed (absolute speed):**
  - Traversal 2–4× slower than predicted.
  - Speedups over all-pairs about 2× smaller: 1.73 / 5.72 / 21 (sorted) at 65536 / 262144 / 1M.
  - The #5/#4 speedup is flat at ≈ 1.9× above 262144 instead of rising to 3.5×.
  - The cause: the plateau rate came from an issue-slot model only. The measured rate barely
    grows with occupancy (16 → 34 → 29 G/s at 4 / 26 / 48 warps per SM). The kernel is bound by
    memory-side incoherence, which the model omitted.
- **Work model:** interactions/body are flat in N (≈ 900–1060 for the Plummer, 8192 → 1M). The
  log N growth is in node pops: 582 → 1612, pops/interaction 0.61 → 1.52. The binary radix
  tree adds one level per doubling of N.

### 16.2 Measured performance (Plummer, θ = 0.5, ms)

| N | build | #4 traverse | #5 traverse | #5/#4 | best all-pairs | AP / #5 total |
|---|---|---|---|---|---|---|
| 65536 | 0.24 | 2.86 | 1.76 | 1.63 | 3.40 | 1.73 |
| 262144 | 0.30–0.40 | 16.2 | 8.48 | 1.91 | 51.0 | 5.72 |
| 1M | 0.62 | 73.0 | 38.6 | 1.89 | 822 [D] | 21.0 [D] |

Interaction rates: #4 14–24, #5 27–43 G/s, i.e. 1–3% of the best all-pairs rate. The same
kernel at θ = 0 (coherent) does 552 G/s.

### 16.3 Imbalance, measured
- **Inside warps:**
  - Per-warp max/mean: #4 1.17–1.62 (core highest), #5 1.05–1.09.
  - Pop-site SIMT efficiency: #4 0.55–0.82, #5 0.92–0.98.
  - Interaction-site SIMT efficiency: #4 0.09–0.13, #5 0.13–0.22, i.e. 3–7 of 32 lanes. The
    leaf-bucket loop serializes (§0 of the predictions explains why sorting cannot fix it).
- **Between warps:** warp-cost max/mean #4 1.06–1.12, #5 1.15–1.60. Sorting moves the
  imbalance there.
- **SM level:**
  - Single wave (65536): #5 core SM eff 0.810, tail 17.7%; #4 0.945.
  - 8.5 waves (1M): 0.987–0.995 for both. The hardware scheduler hands each SM 92–132 blocks.
- **Hilbert vs Morton:** per-warp max/mean lower by only 0.2–0.6%, traversal 1.5–4.4% slower,
  plus a 0.13–0.24 ms order phase. Morton stays the #5 default (D-018).

### 16.4 Loss decomposition (design input for #6 / #7; not pre-registered)
Reference = the θ = 0 control. The achieved fraction F of the coherent throughput is factorised
into idle lanes (issue-weighted SIMT efficiency), a memory-incoherence residual, and block-slot
fill (timeline). Method: `tools/tree_analysis.py`.

| | F | idle lanes (a) | memory incoherence (a′) | blocks + tail (b+c) |
|---|---|---|---|---|
| #4, 1M | 0.06 | 49–50% of the loss | 49–50% | 1% |
| #5, 1M | 0.11–0.12 | 38–42% | 57–60% | 1% |
| #5 core, 65536 (single wave) | 0.11 | 54% | 31% | 15% |

**Decided input (D-019):** next round prioritises **#7 as warp-cooperative traversal**
(warp-uniform walk or per-warp interaction lists) over #6. The persistent queue (#6) can recover
≤ 2.4% at 1M. Its gain needs a single-wave clustered case to show at all.

> **Correction (2026-10-17, tree round 2):** the ≤ 2.4% bound holds for a **block-granular queue
> only** (critic review 2, #2). A lane-granular persistent queue targets the loop-level loss
> (per-warp max/mean 1.05–1.09 for #5, up to 1.62 for #4), which E_sm does not contain. The
> decomposition itself is also contested (critic review 2, #1: E_mem is a residual). Round 2
> replaces it with a measured coherent control (§17.4).

### 16.5 Open items (tree)
- Separate the memory residual (a′) without ncu: an exploratory stack-in-shared-memory variant
  would isolate the local-memory stack share.
- Why interactions/body are flat in N with the bmax criterion. Split the counters into cell
  accepts vs bucket bodies.
- An octree (or a wider radix node) to cut the 1.5 pops/interaction.
- Quadrupole (§11.3), deferred.
- Karras, Skilling and Salmon & Warren are still TODO(source).

---

## 17. Tree round 2: R3 #7 (warp-cooperative walk) and R3 #6 (persistent threads)

Written 2026-10-17 by `cuda-architect`, **before any timing run of these kernels**. It folds in
critic review 2 (#1, #2, #4, #5, #6, #7, #8, #11). What was observed while building (selftest,
mutation tests, one functional run at N = 8192) is in §0 of
`docs/predictions/2026-10-17-tree-round2.md`. Code: `src/core/kernels_tree.cu`. The tree, the bmax
criterion, `bodyBodyInteraction()` and the build are unchanged (§10–§13). **The production SASS of
all 192 pre-existing kernels is instruction-for-instruction unchanged** (`tools/sass_diff.py`
before/after the round-2 edits: 192 of 192 matched).

### 17.1 Registry and what each variant isolates

| name | R3 | walk | scheduler | isolates |
|---|---|---|---|---|
| `bh_sorted` | #5 | per thread | hardware block scheduler | baseline |
| `bh_warp` | **#7** | warp-union walk | hardware | intra-warp incoherence (lanes + memory) |
| `bh_warp_list` | #7, sub-variant | union walk + per-warp interaction lists | hardware | interleaving of pops and interactions |
| `bh_warp_naive` | #7 without #5 | union walk on original-order warps | hardware | how much #7 needs #5 (critic review 2, #3) |
| `bh_persistent` | **#6**, warp-granular | per thread | atomic queue of 32-slot units | between-warp, block-internal and tail loss |
| `bh_persistent_lane` | **#6**, lane-granular | per thread | per-lane refill from the queue | loop-level loss (lanes done, warp not) |
| `bh_persistent_lane_naive` | #6 on #4 | per thread, original order | per-lane refill | the same, where it is large (#4) |
| `bh_persistent_warp` | #6 + #7 | union walk | atomic queue of units | #6 with intra-warp coherence |
| `bh_persistent_static` | #6, static arm | per thread | precomputed cost-balanced partition | dynamic vs precomputed |
| `bh_coherent` | **control** | per thread, `order[t] = perm[t & ~31]` | hardware | the coherent ceiling at θ > 0 (critic review 2, #1) |

### 17.2 R3 #7: the warp-union walk (`warpWalk`)
- **Decomposition.** One warp = one work unit = 32 consecutive Morton slots. The warp pops one node
  at a time; every lane evaluates its own bmax test; the node is opened iff **any** valid lane opens
  it (`__any_sync`), the union of the lanes' decisions [P:burtscher2011barneshut: the warp traverses
  the union of its threads' paths with a warp vote; background]. Control flow is warp-uniform, so
  every interaction and pop runs on 32 lanes and every node or body load is a same-address
  (broadcast) load.
- **Accuracy.** A lane whose own test would accept an opened node gets the node's children or bucket
  instead. Per lane, I(#7) ≥ I(#5) and V(#7) ≥ V(#5) (induction over the walk; asserted per slot by
  `--selftest` and on every bench run). The aggregate error drops (TG3), but one body's error is
  not guaranteed to drop (cancellation), so TG3 is an aggregate gate.
- **Stack.** 64 entries in two registers per lane (entry k in lane k mod 32, register k / 32): push =
  a predicated move, pop = one `__shfl_sync`. No shared memory, no `__syncwarp` on the walk, no local
  memory (0 B stack frame; #5 has a 256 B local stack). This is the per-warp shared stack, held in
  the register file.
- **List variant.** Accepted cells (lane 0) and bucket bodies (one per lane, coalesced) are appended
  to a 64-entry per-warp list in shared memory (1 KB per warp, 4 KB per 128-thread block) and
  evaluated in a dense, 4× unrolled loop (LDS.128 broadcast + interaction) when the list fills. The
  summation order equals the direct walk's, so it is **bitwise equal to `bh_warp`**. It is a separate
  arm because it tests whether interleaving dependent pop chains with interactions still costs
  anything once the walk is coherent.
- **Resources** [M, ptxas `--resource-usage`]: production 40 regs, 0 spills, 0 B stack frame (both
  variants); 12 blocks/SM at block 128 (the list's 4 KB per block fits: 48 KB/SM). The instrumented
  list twin has 48 regs → 10 blocks/SM: its counters are algorithmic and valid, its timelines are
  not comparable.
- **Traffic and bottleneck [D].** Per pop: one broadcast com (16 B) + rc2 (4 B) + info if opened
  (16 B), i.e. one L1 wavefront per load instead of up to 32. Expected bottleneck: the dependent pop
  chain (shuffle → node load → test → vote → branch), i.e. latency hidden by resident warps; not
  FP32 and not L2 bandwidth.
- **Load-balancing problem addressed:** intra-warp imbalance and divergence (idle lanes at buckets,
  finished lanes waiting) and memory-side incoherence, by construction. The price is the union's
  extra work. The metrics that prove it: SIMT efficiency (must be 1.000 at both sites), the union
  work ratio (interactions and pops vs #5), and the traversal time against the coherent control
  (§17.4).

### 17.3 R3 #6: persistent threads
- **Grid.** min(resident blocks/SM × 80, blocks that have work). The cap matters below N = 122,880
  (3840 resident warps at 40 regs and block 128). Without it the excess warps race for the few units
  and the winners cluster on a few SMs: in the functional run at N = 8192 the uncapped warp queue
  took 0.75 ms against 0.51 ms for the normal grid (predictions §0).
- **Warp-granular queue** (`bh_persistent`, `bh_persistent_warp`): lane 0 claims the next unit with
  `atomicAdd` on a global counter and broadcasts it [P:huang2026weave, transferable: SMs claim the
  next tile with atomicAdd on a global counter]. It replaces the hardware block scheduler (already
  dynamic, at 4-warp granularity) by a warp-granular one, so it can recover the block-internal wait
  (3 warps done, block slot still held) and shorten the tail to one unit. It cannot touch the
  loop-level loss inside a warp. Below 122,880 bodies every warp gets at most one unit, so it reduces
  to the normal grid plus one atomic.
- **Lane-granular refill** (`bh_persistent_lane`, `_naive`), the critic-2 #2 design. Each lane runs
  #5's walk for one body. When its stack empties it writes the result, and the warp refills every
  idle lane from a warp-owned batch of 32 consecutive slots: one `atomicAdd(queue, 32)` per batch,
  issued one batch ahead so the atomic's latency is off the critical path. One loop iteration = one
  pop. It attacks the loop-level loss (eff_pop < 1, per-warp max/mean > 1). **Limit:** it can only
  rebalance when a lane gets more than one body, i.e. N ≫ 32 × resident warps = 122,880. **Price:**
  after the first refill a warp's lanes hold bodies from two Morton batches (two regions), so memory
  coherence degrades towards #4's. On #4 that costs nothing (already random). The per-body
  arithmetic is that of `bhTraverseKernel`, so the accelerations are bitwise equal. There is no
  2025–26 source in the bib for lane-granular persistent threads; the warp-level stealing analogue
  is [P:chen2026gmatch, transferable], persistent traversal in general [P:maeda2026gtap].
- **Static arm** (`bh_persistent_static`): per-unit costs (clock64 cycles of the previous launch,
  costzones-style; uniform on the first launch at a given N and grid) → CUB inclusive scan →
  contiguous unit ranges of equal cost per persistent warp (owner = cost midpoint × G / total)
  [P:xia2025voltrix, transferable: persistent kernel + precomputed balanced partition;
  P:li2025staticbatching, transferable: prefix sum of per-task cost, no runtime scheduler]. In the
  bench every timed launch uses the costs of the previous identical launch: the best case for a
  static partition. Its known disadvantage: contiguous ranges spread co-resident warps over
  far-apart regions (worse L1 sharing than the queue, where co-resident warps take neighbouring
  units).
- The **order phase** holds the scheduler set-up (queue reset: a 4 B memset; static: scan +
  partition), so the traversal phase is the kernel alone.
- **Resources** [M]: 38–40 regs in production, 256 B local stack for the per-thread walks, 0 for
  `bh_persistent_warp`; 12 blocks/SM at block 128.
- **Exactly once.** The persistent instrumented twins accumulate their counters with atomics into a
  zeroed buffer, so a slot processed twice shows doubled counts and a skipped slot shows zeros. Every
  bench run of a persistent kernel compares accelerations (bitwise) and per-slot pop and interaction
  counts (exactly) with its non-persistent sibling.

### 17.4 The coherent control (`bh_coherent`, critic review 2, #1)
The same kernel and SASS as #5 (`bhTraverseKernel`), the same tree and θ, with
`order[t] = perm[t & ~31]`: all 32 lanes of a warp walk the body of the warp's first slot. That gives
perfect lane coherence and same-address loads with the same instruction mix and the same local
stack (now with a uniform index), at θ = 0.5. Only 1 body in 32 gets a force (the 32 identical
writes are benign), so it has no gates, and its "interactions" are lane-work. Its traversal time is
what #5's kernel needs for #5-like per-lane work when the warp is coherent: a measured ceiling for
intra-warp fixes that replaces the residual E_mem. The selftest checks that exactly ⌈N/32⌉ bodies
are written, each bitwise equal to bh_sorted's. Caveat: a control warp's work is one body's path,
so its warp-cost spread samples #5's lanes rather than #5's warp maxima; the comparison is on lane
work per second (and on time, since the lane-work totals are within a few % of #5's).
Not built: a stack-in-shared-memory #5. At 64 ints × 128 threads = 32 KB per block it would run at
3 blocks/SM and confound the stack with occupancy. The register stack of #7 and the uniform-index
local stack of the control bracket the stack's share instead.

### 17.5 Measurement additions
- **Warp-unit timelines** (`--timeline-unit warp`): one record per 32-slot unit (per persistent warp
  for lane refill), so persistent kernels have a meaningful timeline and block-internal warp idling
  becomes visible. `slots_per_sm` in the header = resident warps/SM; `tools/tree_analysis.py` uses it.
- **Occupancy sweep** (`--occ-blocks B`, critic review 2, #6): dynamic shared-memory padding caps the
  resident blocks/SM at fixed N; padding = ⌊102,400 / B⌋ − 1024 − the kernel's own smem. The padding
  also pins the L1 carve-out near its minimum for every B, so B = 12 padded is compared with B = 12
  unpadded to separate L1 capacity from occupancy.
- **θ ≤ 1** in the bench and the launcher (critic review 2, #11).

### 17.6 Validation (pre-registered gates)
- TG1/TG2 and D-009 unchanged. **TG3 (new, union walks):** rms and p99 relative error over all bodies
  ≤ the sibling's (#5; #4 for `bh_warp_naive`) at the same θ, and per slot no fewer pops or
  interactions than the sibling. **Exact siblings:** `bh_warp_list` and `bh_persistent_warp` ≡
  `bh_warp`; the per-thread persistent arms ≡ `bh_sorted` (≡ `bh_naive`), bitwise and with equal
  per-slot counts. A FAIL exits with code 3.
- **`--selftest` additions** (all pass: 1494 new tree checks; the lattice now has 4450 cases because
  every new kernel runs the exact θ = 0 lattice too):
  1. **Exact θ > 0 host check** (critic review 2, #7) for 3 ICs × N ∈ {1, 2, 3, 1000, 10007} × θ ∈
     {0.5, 0.7}. Every node's mass, COM, AABB and bmax are recomputed in FP64 from its
     [first, last] range (AABB bitwise; mass rel ≤ 1e-5; COM and bmax ≤ 2e-5 × extent). Then the same
     tree is walked on the host with the device's FP32 test (the same `fmaf` expression and the
     device com/rc2) and FP64 accumulation, for the #5 walk and for the union walk. Per-body pop and
     interaction counts must equal the device's, and |a_dev − a_host| / Σ|f| must be ≤ (I + 10) u.
     Worst case seen: 0.27 of the bound. This checks the monopole-accept branch exactly.
  2. Instrumented = production, bitwise, for every new kernel; exact siblings as above; the static
     partition stable across re-partitions; union per-slot I, V ≥ sibling; TG3 at N ≥ 1000; at θ = 0
     the union walks ≡ `bh_sorted` on random ICs; the coherent control; warp-unit timeline records.
  3. **Mutation tests** (code reverted afterwards): a persistent queue that skips its last unit, the
     vote changed from `__any_sync` to `__all_sync`, and a 0.1% mass error in #5's monopole-accept
     branch. All three fail the selftest (480 failures). The third is caught **only** by the new
     exact host check; TG1/TG2 pass it. `--selftest-drop 5` still detects all 3560 visible drops.

### 17.7 Interface changes
- `Params`: `timelineWarp`, `smemPad`. `LaunchInfo`: `timelineRecords`, `timelineSlotsPerSM`,
  `persistent`, `control`, `errorSibling`, `exactSibling`, `counterSibling`, `dynSmem`.
- `checkLastTreeExact()` in `nbody.h`. `bhTraverseKernel` gained a defaulted template flag (TLW,
  warp-unit stamps); its production and round-1 instrumented instantiations are unchanged in SASS.
- **Rate label (critic review 2, #5).** For tree kernels the headline `kernel=` line now reads
  "G N^2-equivalent interactions/s (not the tree rate; see the work: line)" and drops GFLOP/s. A new
  `rates:` line gives the real rate over the pipeline and over the traversal. In the new CSV header,
  `ginteractions_per_s` and `gflops` hold the **real** interaction rate (all-pairs: N²; trees:
  counted interactions / pipeline time), `n2equiv_ginter_per_s` holds N² / time, and
  `trav_ginter_per_s` the traversal rate. Historic CSVs are unchanged, and the bench refuses to
  append to them (header check). The round-1 tree CSVs keep N²-equivalent values in
  `ginteractions_per_s`; they must not be read as tree rates.
- New CSV columns: `n2equiv_ginter_per_s, trav_ginter_per_s, occ_blocks, smem_pad, persistent,
  control, sib_rms, sib_p99, tg3, union_inter_ratio, union_visit_ratio, exact_match, counts_match`.

### 17.8 Scorecard rule (critic review 2, #4)
The phase-2 scorecard is a committed table file (`results/<date>_tree_round2/scorecard.csv`): one row
per pre-registered item ID, with the range, the measured value, the source log, HIT/MISS, and an
`anchored` flag for items built on a pre-registration observation. No item is split or reinterpreted
after registration; a disputed reading is listed as disputed.

---

## 18. Tree round-2 outcome (2026-10-17)

Predictions committed before the runs (`docs/predictions/2026-10-17-tree-round2.md`, with a dated
post-registration note on scoring readings). Data: `results/2026-10-17_tree_round2/` (the harness
blocked the agent from writing `summary.md`; its content is in the architect's phase-2 report).
**Session label: 20–22% idle GPU utilisation from desktop applications**, measured cost +2.4–2.7%
on round-1 reference points; ratios are taken from adjacent runs. Scorecard (committed table
`scorecard.csv`, one row per ID): **39 HIT / 22 MISS of 61**. §17 above is left as written.

### 18.1 R3 #7 held
- Traversal T5/T7: Plummer 3.1 / 5.4 / 6.6×, core 3.4 / 6.4 / 7.3× at 65536 / 262144 / 1M; total
  2.6 / 4.5 / 6.0× (Plummer). Best vs best (leaf × block, θ = 0.5): **5.9× at 1M** (#5 leaf 32
  b256, 35.30 ms; #7 leaf 8 b256, 5.97 ms), 5.2× at 262144.
- Union work: interactions ×1.64–1.85, pops ×1.22–1.31, flat in N (pre-registered, anchored, HIT).
  SIMT efficiency 1.000 at both sites. Error drops (rms ×0.66–0.85, p99 ×0.60–0.91; TG3 60/60).
- Real traversal rate 193 / 285 / 322 G/s (Plummer): about 11× #5's 28–33 G/s.
- The list variant is 2–7% **slower** than the direct walk (W13 MISS): once the walk is
  coherent, interleaving pops and interactions costs nothing, and the list adds STS/LDS traffic.
- Without the space-filling-curve order the union walk does 13–15× #4's interactions, yet it is
  still 1.2–1.6× faster than #4: divergence costs #4 more than an order of magnitude more work.

### 18.2 The coherent control replaces the residual decomposition (critic review 2, #1)
#5's own kernel with all 32 lanes on one body runs 5.2–9.7× faster than #5 on the same lane work
(172–259 G lane-int/s). That ratio is the whole intra-warp penalty (idle lanes + scattered loads +
divergent stack), measured, with no c_V or residual. #7 sits 1.32–1.65× above the control, i.e.
about its extra work, and closes 69–88% of the #5 → control gap in log terms. The round-1 "8.4×
ceiling" (D-019) was a residual-based estimate; the measured ceiling at 1M is 9.2× (Plummer) and
9.7× (core).

### 18.3 R3 #6 missed: balance improves, speed falls
- Every arm is slower than its sibling in every cell: warp queue 0.89–0.93× (#5) and 0.89–0.95×
  (#7); lane refill 0.64–0.74× (#5), 0.68–0.91× (#4); static partition 0.84–0.98×.
- The balance metrics move as designed: the warp queue lifts the single-wave core SM efficiency
  from 0.81 to 0.92 (tail 18% → 7%) and the 1M warp-slot fill from 0.95 to 0.98; lane refill lifts
  #4 core's pop efficiency from 0.61 to 0.91 at 1M.
- Exploratory diagnosis (not pre-registered): a block-granular queue (4 Morton-adjacent units per
  claim) recovers most of the warp queue's loss (0.94–0.98×), so the loss is chiefly **intra-block
  locality**: the hardware scheduler hands an SM four Morton-adjacent warps that share L1 lines; a
  warp-granular queue scatters them. Lane refill checked every 8 pops is barely better than every
  pop (0.65 vs 0.63× at 65536, where it does no refills), so its loss is the loop structure (most
  likely the per-pop warp collectives forcing reconvergence; not verified in SASS).
- Lesson for the design: on this GPU the hardware block scheduler is already a dynamic queue at
  4-warp granularity with good locality. A software queue has to preserve that locality (claim
  blocks of adjacent units) before it can win anything, and the remaining SM-level loss at
  N ≥ 262144 is only 1–2%.

### 18.4 What limits the walks (sets C, D)
- **#5 is L1-capacity-bound:** with the L1 carve-out pinned small, its rate is flat from 2 to 11
  blocks/SM (19.9–20.7 G/s); unpadded it does 29.0 G/s. Occupancy is not its limit (answers critic
  review 2, #6; the round-1 "16 → 34 → 29 G/s" was N, not occupancy).
- **The coherent walks are latency-bound:** the control and #7 scale ×1.75 and ×1.87 from 4 to 11
  blocks/SM, and depend less on L1 (×1.30, ×1.12). Caveat: the padded "12" cell ran at 11
  blocks/SM (256 B shared-memory allocation granule).
- Leaf/block: #5 prefers leaf 32 and block 256 (11% faster than block 128 at 1M); #4 likewise; #7
  is flat (±1%) over leaf 8–16 and all blocks. The pre-registered leaf directions missed (L1–L4).

### 18.5 Crossover and θ
- bh_warp crosses the best all-pairs at **N ≈ 29,200** (39,700 for #5). Total speedups over the
  best all-pairs: 4.4× (65536), 24.9× (262144), 123× (1M, all-pairs extrapolated [D]).
- θ sweep (262144): union overhead ×1.55 / 1.79 / 2.08 / 2.11 and #7 gain 8.1 / 5.4 / 4.0 / 3.0× at
  θ = 0.3 / 0.5 / 0.7 / 1.0. At θ = 0, #7 = #5 within 2%, D-009 PASS.

### 18.6 Open items
- Lane refill: compare its SASS with `bhTraverseKernel`, to confirm the reconvergence explanation.
- A block-granular queue on the #7 walk (the only #6 arm that might break even); #7's single-wave
  tail (27% at core 65536) is now its largest loss, which is the case where a fine-grained queue
  with locality could pay.
- Re-run the #6 items with the desktop applications closed (load-sensitive), and fix the
  `--occ-blocks` padding for the 256 B granule.
- #7 with leaf 8 vs 16 and an octree / wide node (fewer pops: the coherent walk is pop-bound).

---

## 19. IC round: initial conditions as a workload suite (2026-10-17)

Written by `cuda-architect` **before any timing run of this round**. What was observed while
building (selftest, gate calibration on the legacy ICs, one functional run per new IC at N = 8192,
viewer smoke runs) is listed in §0 of `docs/predictions/2026-10-17-ic-round.md`. Decisions: D-027
(normalised gates), D-028 (stack), D-029 (ICs, spectrum, viewer). The tree, the bmax criterion,
`bodyBodyInteraction()` and every #7 walk are unchanged; the per-thread walks changed only their
stack capacity (§19.2).

**Load-balancing framing.** A tree walk's cost per body is set by where the body sits. So the IC is
the *input* of every load-balancing technique in R3: the per-body cost distribution drives intra-warp
imbalance (what #5 and #7 fix), and the per-region cost distribution drives between-warp and
SM-level imbalance (what #6 targets). Each new IC is a different imbalance generator; uniform is the
control. The ranking and the per-IC expectations are in the predictions file (§2–§6).

### 19.1 The ICs (`--ic`, `nbody::makeIC(..., ICOptions)`)
All total mass 1 (G = 1), mass-weighted centre of mass and momentum removed, deterministic in the
seed, shuffled with the legacy shuffle so the original order carries no spatial information.

| IC | geometry | velocities | role |
|---|---|---|---|
| `disk` | exponential disk Σ ∝ e^(−R/Rd), Rd = 0.25, cut at 6 Rd, sech²(z/z0), z0 = 0.1 Rd; Plummer bulge (virial radius 0.08, 10% of the light bodies); one central body of mass `--bh-mass` (0.01) | v_c² = M(<r) R² / (r² + ε²)^1.5 from the realised enclosed mass in spheres; σ_R = σ_φ = 0.15 v_c; σ_z² = π G Σ z0; bulge: Plummer velocities × √(M_tot(<r)/M_bulge(<r)) | a dense 3D core inside a thin sheet: strong per-body cost gradient |
| `galaxies` | two disks of mass 1/2 at (∓1.5, ∓0.5, 0), tilted 30° and (−50°, 40°) | bulk ±(0.3, 0.05, 0): relative 0.60 < escape 0.80, a bound passage and merger | two compact expensive regions in an empty box: the most clustered IC |
| `cube_shell` | uniform on the 6 faces of [−1,1]³ | isotropic σ = 0.05 (almost cold) | **stress test**: hollow, 2D, flat AABBs |
| `cube_edges` | uniform on the 12 edges | σ = 0.05 | **stress test**: 1D, duplicate Morton keys, corner hot spots |
| `uniform` | uniform in [−1,1]³ | σ = 0.05 | the no-clustering control |

- *Disk approximation and limits.* The spherical enclosed-mass circular speed is exact for the bulge
  and the central body and approximate for the disk's own field [TODO(source): Freeman 1970, thin
  exponential disk]. The vertical relation is the isothermal sheet [TODO(source): Spitzer 1942]. No
  dark halo and no Toomre-Q control (Q ≪ 1 at these dispersions), so the disk is cold and
  self-gravitating: it develops bars, spirals and clumps within about one rotation (rotation period
  at Rd ≈ 1.7 time units). That is fine for a force snapshot and a demo; it is not a Milky Way model.
- *Mass spectrum* (`--mass-spectrum`, light bodies of every component, never the central body):
  `equal`; `salpeter`, dN/dm ∝ m^−2.35 on [1, 100] (4.3% of the bodies carry 31% of the mass);
  `heavy-few`, min(16, n_c/128) bodies with 1% of the component mass each. Masses are drawn
  independently of position and normalised per component, so the expected density profile is
  unchanged. With `equal`, `plummer`, `collision` and `core` are **bitwise** the pre-round ICs
  (FNV hashes in the selftest).
- *What unequal masses do to the tree:* the criterion is geometric, but bmax is measured from the
  COM, which a heavy body pulls off-centre, so small cells are opened at a larger distance (more work:
  ≈ +15% for salpeter at 8192, §0 of the predictions), while a cell dominated by one body has a small
  quadrupole (no error growth expected). The bench quantifies it: the FP64 sample adds the 16 heaviest
  bodies, and the normalised error is split heavy / light.
- *Softening / dt* [D]: ε is global. dt advisory = 0.1 √(ε³ / ((2/3^1.5) G m_max)), printed by the
  bench and the viewer (1.6e-3 for a 1% body at ε = 0.01). Force-snapshot runs do not depend on dt.

### 19.2 Stack capacity (D-028)
Along a root-to-node path of the binary radix tree the split prefix δ strictly increases (δ ∈ [1, 63]
for distinct keys, 64 + clz(i ⊕ j) for duplicates), so internal depth ≤ 62 + ⌈log2 n⌉ and a DFS that
pushes (right, left) never holds more than pushDepth + 2 ≤ 95 entries. The per-thread walks now have
96 entries: overflow is impossible (the counter stays as a guard). SASS: only the frame size and the
bound immediate changed in 18 kernels; registers 40, occupancy unchanged. The #7 register stack stays
at 64 (a 96-entry version costs +8 instructions on the pop chain and 4 registers in the instrumented
twin); every tree run prints `tree: max depth, push depth → stack bound (capacity …), duplicate keys`,
and an overflow fails the run. Measured stack bound ≤ 45 for every legacy IC at 1M.

### 19.3 Gates (D-027)
Primary tree gates are now normalised by S_i = Σ_j|f_ij| (the D-009 normaliser) over all bodies:
TN1 rms ≤ 0.02 (θ/0.5)², TN2 p99 ≤ 0.045 (θ/0.5)², zero overflows; TG3n compares the same metric
with the sibling. Calibrated on the legacy ICs only (worst rms 6.1e-3, p99 1.43e-2 per (θ/0.5)²); the
legacy verdicts are unchanged. Legacy TG1/TG2/TG3 are still printed (`not gating`). New CSV columns:
`mass_spectrum, bh_mass, ic_hash, mass_max_min, norm_rms, norm_p99, norm_max, tn1, tn2, sib_norm_rms,
sib_norm_p99, tg3n, glob_rms, glob_p99, heavy_norm_max, tree_depth, push_depth, stack_bound, dup_keys`
(new header; the bench refuses to append to older CSVs).

### 19.4 Selftest additions (all pass; 4 logged XFAILs)
- The round-1 and round-2 tree checks (structure, bitwise siblings, exactly-once, per-slot union
  invariants, TG3n, and the **exact θ > 0 host check** at n ≤ 10007) now run on 12 IC labels: the 8
  ICs plus plummer:salpeter, disk:heavy-few, galaxies:salpeter and cube_shell:heavy-few.
- IC invariants for 8 ICs × 3 spectra × n ∈ {1, 2, 3, 1000, 65536}: total mass 1, COM and momentum 0
  (new ICs), finite values, positive masses, determinism; legacy-IC hashes.
- An unequal-mass pass of the exact lattice (masses {1, 2, 4, 8} and 1% at 128, N up to 65536, every
  partial sum still exact): every kernel bitwise equal to the analytic sum. `--selftest-drop 5`
  detects all 6204 visible drops.
- XFAIL: TG3n of `bh_warp_naive` on `cube_edges` (D-027).

### 19.5 Viewer
Live N (PageUp/PageDown, `.`/`,`): the IC is regenerated at the new N, the device state reallocated,
the two GL buffers resized and re-registered with CUDA (unregister → `glBufferData` →
`cudaGraphicsGLRegisterBuffer`), the camera kept. `I` cycles the IC, `H` the mass spectrum, `M` colour
by mass (log-scaled red shift and larger points above 3× the mean mass). The title shows IC,
spectrum, N, kernel, ms/step and the real interaction rate: for tree kernels the interactions of one
instrumented launch (bitwise-equal accelerations, so the integration is undisturbed), refreshed every
2 s, divided by the measured ms/step (which includes the build and the integrator). N caps: 262144 for
all-pairs, 4,194,304 for tree kernels (TDR margin). Default kernel `bh_warp` (D-022). `--test-keys`
scripts key presses for smoke tests; `--dist/--yaw/--pitch` frame screenshots.

---

## 20. IC-round outcome (2026-10-17)

Predictions committed before the runs (`docs/predictions/2026-10-17-ic-round.md`, plus a dated
post-registration note on the V2 reading and the load label). Data: `results/2026-10-17_ic_round/`
(the harness blocked the agent from writing `summary.md`; its content is in the architect's phase-2
report). **Session label: 30% (before) / 37% (after) idle GPU utilisation from desktop applications.**
Same-session plummer, leaf 16 b128, 262144: #5 +11% slower than in tree round 2, #7 −1%, so #7/#5
ratios in this session are inflated by up to ~10%. Scorecard (`scorecard.csv`): **130 HIT / 43 MISS
of 173**. §19 is left as written.

### 20.1 The IC decides which technique matters
| IC class | #4 → #5 (order) | #5 → #7 (warp walk) at N ≥ 262144 | why |
|---|---|---|---|
| 3D clustered (plummer, disk, galaxies) | 2.1–2.7× | 3.8–6.1× | random warps diverge; Morton warps still diverge inside (bucket loops, different depths) |
| 3D uniform | 2.3–2.4× | 3.3–4.3× | the same, with the largest union overhead (×2.3–2.4) |
| sheet (cube_shell) | 4.0–4.6× | 2.2× | a Morton warp is a compact patch: nearly one path |
| line (cube_edges) | **6.5–7.0×** | **1.04–1.21×** | a Morton warp is a segment of one edge: #5 runs at 220–230 G int/s, like #7 on 3D ICs |

So intra-warp incoherence, the loss #7 removes, is a property of the *dimensionality* of the
distribution, not just of clustering: on lines and sheets the space-filling-curve sort alone already
removes it, and #7 is left with only its union overhead. This is the cleanest demonstration in the
project that #5 and #7 attack the same loss.

### 20.2 Imbalance, measured
- Intra-warp (#4 per-warp max/mean): the disks are the most imbalanced at every N (1.81–1.95 at 65536,
  1.35–1.60 above), then uniform (1.22–1.30: the boundary matters more than predicted), plummer
  (1.19–1.20), cube_edges (1.23–1.29) and cube_shell (1.19–1.22). The predicted corner/edge hot spots
  of the cube ICs do not appear (M4, M6 MISS).
- Between warps (#5 warp-cost max/mean): galaxies 2.61 and disk 2.22 at 65536 (core in round 2: 1.59),
  falling to 1.49–1.54 at 1M.
- SM level (warp-unit timelines, galaxies 65536, a single wave): #5 SM efficiency 0.62, tail 37%;
  #7 0.61, tail 39%: the largest SM-level losses measured in this project.

### 20.3 R3 #6 on the most imbalanced launch
The warp queue lifts galaxies-65536 SM efficiency from 0.62 to 0.87 (tail 37% → 13%), but the mean SM
busy time rises 53% (674 → 1029 µs) and the traversal is 0.96× of #5. At N ≥ 262144 it loses 15–25%
at the best-tuned block 256 (0.75–0.85×), more than round 2's 7–11% at block 128: a 256-thread block
holds 8 Morton-adjacent warps sharing L1, and the warp queue scatters them (D-023's mechanism, growing
with block size). Exploratory (not pre-registered): a block-granular queue and the queue on the #7
walk reach parity (0.996–1.010×) on galaxies 65536, and only there. **No #6 arm wins on any IC.**

### 20.4 Errors, mass spectrum, trees
- Normalised errors flat in N (±10%), #7 < #5 on every IC; TN1/TN2 PASS in all 133 runs; legacy TG1 FAILs
  on cube_shell and cube_edges at every N (the reason for D-027).
- Mass spectrum: salpeter +10–11% #5 work (COM pulled off-centre → larger bmax), heavy-few +1%; errors
  unchanged; heavy bodies not worse than light ones.
- 1M trees: push depth ≤ 38, stack bound ≤ 40 (warp stack 64 is enough, D-028); cube_edges 2.08%
  duplicate keys, max depth 48. The per-thread stack change is not measurable (0.98–1.03× old/new under
  load).

### 20.5 Open items
- Re-run the S and Q items with the desktop applications closed (30–37% load this session).
- The single-wave tail (37–39% at galaxies 65536) is now the largest loss for both #5 and #7. The two
  arms that reached parity (the block queue on #5 and the queue on #7) are the starting point;
  splitting expensive units (sub-warp work items for the dense bulges) would attack the tail directly.
- The disks fragment within a rotation (no halo): a halo potential would make the evolved-state
  workload (2857 int/body at t = 3 vs 1619 at t = 0) a stable benchmark too.

---

## 21. Viewer round: control panel and live statistics (2026-10-18)

Decision D-032. No force kernel, tree code or integrator changed; production SASS is byte-identical
(`results/2026-10-18_viewer_round/sass_diff.txt`). No performance claims in this round.

- **What the panel shows and where it comes from.** ms/step: CUDA events around the frame's steps
  (force + both kicks, + the well kick while active). Phases: `Params::phaseEvents` on the first
  step of each frame (EMA-smoothed). Real interaction rate: counted interactions of the
  instrumented twin / ms/step (all-pairs: N²). Load balance: the twin's per-thread counters reduced
  exactly as `nbody_bench` does (§14): per-warp max/mean, SIMT efficiency at interaction and pop
  sites, warp-cost max/mean; plus `lastTreeDepth()` (depth, push depth, stack need vs 96/64). The
  twin differs from production by −3.4…+6.2% in time (D-011); its *counts* are exact and its
  accelerations bitwise equal, so measuring does not perturb the run.
- **Measurement cost** (display, not a result): ≈7 ms per load-balance measurement at N = 262144,
  ≈110 ms at 4M (64 MB counter copy + host reduction + node-table copy for the depth); hence
  "every K frames".
- **Viewer-only kernels** (`src/viewer/viewer_kernels.cu`): `wellKickKernel`, one thread per body,
  v += dt·G·M_w·r/(r² + ε_w²)^1.5; `energyKernel`, 256-body shared-memory tiles, FP32 terms summed
  per tile, FP64 across tiles and for KE, self-pair skipped; ≤ 2^33 pairs per launch (TDR).
- **Origin tags** (§19.1 components → group/role byte): galaxy/cluster index = palette, bulge paler,
  central body = large white point. They follow each body through a merger, which makes mixing
  visible (and the per-galaxy origin of the expensive bulge walks).
- **Gravity well 2D → 3D:** a pixel defines a ray; the depth is fixed by the focal plane through the
  orbit target, perpendicular to the view direction f. With d = f + x·tanH·aspect·s + y·tanH·u,
  d·f = 1, so the hit is at t = dist and X = dist·tanH·(x·aspect·s + y·u). Orbiting tilts the plane,
  so any 3D point is reachable.
- **Presets** (viewer-only): `merger3` (three disks of 1/3, triangle of radius 1.6, bound),
  `flyby` (disk 0.8 + Plummer cluster 0.2, relative speed 0.75 vs escape 0.81), `unequal` (3:1 disk
  merger, bound). Not in the bench.
