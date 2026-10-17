# 03 — Kernel design: all-pairs variants (R3 #1–#3)

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
