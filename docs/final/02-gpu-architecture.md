# 2. Target GPU Architecture: NVIDIA GeForce RTX 4080 SUPER (Ada Lovelace, AD103)

> **Language note:** drafted in English; the Bulgarian/English choice for the final document is still open (`docs/00-course-requirements.md`, open questions).

**Source tags used in this chapter** (project convention, `CLAUDE.md` rule 5):
**[M]** measured on this machine on 2026-10-17 (`docs/raw/devicequery_2026-10-17.txt`,
`docs/raw/nvidia-smi-q_2026-10-17.txt`, or `docs/01-hardware.md` rows tagged M) ·
**[S]** public NVIDIA specification as recorded in `docs/01-hardware.md` ·
**[D]** derived arithmetically from [M]/[S] values ·
**[N:key]** NVIDIA document in `references/references.bib` ·
**[P:key]** paper in `references/references.bib`.
Where a needed fact has no verified source, the text says `TODO(source needed: …)` instead of
filling it in.

---

## 2.1 Overview and chip hierarchy

The test machine has a single NVIDIA GeForce RTX 4080 SUPER with 16 GB of memory, built by ASUS
(subsystem ID 0x89621043) [M]. The GPU belongs to the Ada Lovelace architecture [M], uses the
AD103-400-A1 die (device ID 0x2702) [M] and reports compute capability 8.9, so all device code in
this project is compiled for `sm_89` [M].

An Ada GPU is organised hierarchically. The chip is divided into Graphics Processing Clusters
(GPCs); each GPC contains Texture Processing Clusters (TPCs); each TPC contains Streaming
Multiprocessors (SMs), which are the units that actually execute CUDA thread blocks
[N:nvidia2023adawhitepaper]. The full AD103 die has **7 GPCs, 40 TPCs and 80 SMs**
[N:nvidia2023adawhitepaper], which gives two SMs per TPC [D].

Our board reports **80 SMs** through the CUDA runtime [M]. Because this equals the SM count of the
full AD103, the RTX 4080 SUPER uses the complete die, with no SMs disabled [D, from M and
N:nvidia2023adawhitepaper]. For contrast, the non-SUPER RTX 4080 16 GB described in the
whitepaper uses a cut-down AD103 with 76 SMs and 9,728 CUDA cores [N:nvidia2023adawhitepaper].
How the 40 TPCs are distributed across the 7 GPCs is not stated in our verified sources
(`TODO(source needed: TPCs per GPC on AD103)`); 40 is not divisible by 7, so the GPCs cannot all
be identical [D].

The whitepaper does not mention the RTX 4080 SUPER at all (`references/nvidia2023adawhitepaper.md`).
Every statement below about "the full AD103" is therefore from NVIDIA, and every statement that
our specific board matches it rests on our own measurements.

## 2.2 The Ada streaming multiprocessor (SM) in depth

The SM is where all of the N-body force computation happens, so this is the most important
section of the chapter.

### 2.2.1 Processing partitions and schedulers

Each Ada SM contains **128 FP32 CUDA cores** [N:nvidia2023adawhitepaper], which is
10,240 FP32 cores across 80 SMs [D]. These are organised as **4 processing partitions of 32 FP32
lanes** each [S] (`docs/01-hardware.md` §2.2; the 128-per-SM total is confirmed by
[N:nvidia2023adawhitepaper], the split into 4 × 32 is not yet confirmed in a verified reference:
`TODO(source needed: SM partition layout, e.g. whitepaper SM block diagram)`).

A warp (32 threads [M]) is executed by one partition, which issues instructions for the warps
assigned to it. The number of warp schedulers and dispatch units per partition, and the issue
rate per scheduler per clock, are needed to reason precisely about instruction-issue limits but
are not in our verified sources (`TODO(source needed: warp schedulers per partition and issue
rate on Ada SM)`).

Compared with compute capability 8.0 (A100), NVIDIA states that sm_89 executes
"2x more FP32 operations per cycle per SM" [N:nvidia2026adatuning]. This doubled FP32 rate is
the main reason the all-pairs N-body force loop, which is dominated by FP32 multiply-adds, is
compute-bound on this GPU (`references/nvidia2026adatuning.md`).

### 2.2.2 FP32 and INT32 datapaths

The 128 FP32 lanes per SM perform fused multiply-add (FMA), the operation that dominates the
force computation; counting an FMA as two floating-point operations gives the peak rate in §2.5
[D]. How the integer (INT32) datapath is shared with the FP32 lanes within a partition, and
therefore how much integer address and loop arithmetic competes with FP32 work, is not covered by
our verified sources (`TODO(source needed: FP32/INT32 datapath split per Ada partition)`).

### 2.2.3 Special function units (SFU / MUFU)

Transcendental and reciprocal functions such as `rsqrtf` are not executed on the FP32 FMA lanes but
on separate special function units, which appear in NVIDIA's SASS as `MUFU` instructions
(`results/2026-10-17_smoke/summary.md`, observation 2). This matters directly for N-body, because
every pairwise interaction needs one reciprocal square root, `1/sqrt(r² + ε²)`. The smoke-test
notes give the SFU throughput as 16 operations per clock per SM, against 128 FMA per clock per SM
(`results/2026-10-17_smoke/summary.md`), but that figure is not yet backed by a verified NVIDIA
document (`TODO(source needed: arithmetic-instruction throughput table for cc 8.9, CUDA
Programming Guide)`).

**Hypothesis, not a result.** If the 16 : 128 ratio holds, one `rsqrtf` per interaction costs as
much issue capacity as eight FMAs, and the force kernel may be limited by MUFU throughput or by
instruction issue rather than by FMA throughput (`results/2026-10-17_smoke/summary.md`,
observation 2). This has not been confirmed with Nsight Compute and must not be read as a finding.

### 2.2.4 Register file

Each SM has a **256 KB register file** [N:nvidia2023adawhitepaper], i.e. **65,536 32-bit
registers** [M; N:nvidia2026cudaguide], 20,480 KB for the whole AD103 [N:nvidia2023adawhitepaper].
A single thread can use at most 255 registers [M]. Registers are the fastest storage on the chip
and are private to a thread; in the N-body kernel they hold the body's own position, its
accumulated acceleration and the loop temporaries. Because the register file is shared by all
resident threads, register use per thread is one of the limits on occupancy (§2.3). Whether the
register file is physically split per partition is not stated in our verified sources
(`TODO(source needed: per-partition register file size)`).

### 2.2.5 Unified L1 data cache and shared memory

Each Ada SM has **128 KB of combined L1 data cache and shared memory**
[N:nvidia2023adawhitepaper; N:nvidia2026cudaguide], 10,240 KB across the full AD103
[N:nvidia2023adawhitepaper]. The same physical SRAM is divided between hardware-managed L1 cache
and programmer-managed shared memory. The shared-memory part (the *carve-out*) can be set to
**0, 8, 16, 32, 64 or 100 KB** per SM [N:nvidia2026adatuning], with the remainder used as L1
[D]. At the maximum carve-out, 100 KB is shared memory and 28 KB is left for L1 [D].

The CUDA runtime on our board confirms the per-SM shared-memory maximum of **102,400 B (100 KB)**
[M; N:nvidia2026adatuning] and a per-block maximum of **101,376 B (99 KB)** with opt-in
[M; N:nvidia2026cudaguide]; without the opt-in, a block is limited to 48 KB [S, `docs/01-hardware.md`
§2.3; `TODO(source needed: 48 KB default per-block limit)`].

Shared memory is the key resource for all-pairs N-body. In the tiled algorithm each thread block
loads a tile of body positions into shared memory once and every thread in the block reuses it,
which turns most global-memory reads into shared-memory reads [P:nyland2007nbody]. The carve-out
options above bound how large a tile can be (`references/nvidia2026adatuning.md`).

### 2.2.6 Tensor cores and RT cores (and why N-body FP32 does not use them)

The full AD103 has **320 fourth-generation Tensor cores** and **80 third-generation RT cores**
[N:nvidia2023adawhitepaper], i.e. 4 Tensor cores and 1 RT core per SM [D]. Fourth-generation
Tensor cores support FP8 [S]. Tensor cores accelerate dense matrix multiply-accumulate
(`TODO(source needed: Tensor core operation description, e.g. whitepaper Tensor core section)`); the direct
all-pairs force computation contains a reciprocal square root and a data-dependent scaling per
pair rather than a dense matrix product, so our FP32 force kernels are written for the CUDA cores
and SFUs only. Confirming that no tensor instructions appear in the compiled kernels requires
inspecting the SASS (`TODO(source needed: SASS listing of allpairs_tiled)`).

RT cores perform hardware traversal of bounding-volume hierarchies. Nagarajan et al. reformulate
Barnes-Hut as a ray-tracing problem and use RT cores, through OptiX, for the node-opening
decisions [P:nagarajan2025rtbarneshut]. That route therefore exists on our hardware, but it
requires OptiX rather than plain CUDA compute and is out of scope for the all-pairs FP32 baseline
(`references/nagarajan2025rtbarneshut.md`).

### 2.2.7 FP64 units

The Ada SM contains only **2 FP64 units**, and NVIDIA states that
"The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations." [N:nvidia2023adawhitepaper].
This is stated for AD102; the whitepaper's AD103 appendix describes the same SM
(`references/nvidia2023adawhitepaper.md`). The consequences are discussed in §2.5.

## 2.3 Execution model on sm_89: warps, blocks and occupancy

A CUDA kernel is launched as a grid of thread blocks; each block is assigned to one SM and stays
there until it finishes; inside the SM its threads are executed in groups of 32, called warps
(warp size 32 [M]; the programming-model sections of [N:nvidia2026cudaguide] are not yet read, so
`TODO(source needed: block-to-SM residency and warp execution, CUDA Programming Guide)`). A block
can contain at most 1,024 threads [S] (`TODO(source needed: max threads per block in a verified
reference)`). Several blocks can be resident on the same SM at once [M: up to 24], and the SM
hides latency by switching between resident warps (`TODO(source needed: latency hiding by warp
switching, e.g. Best Practices Guide "Execution Configuration Optimizations")`).

*Occupancy* is the ratio of resident warps to the maximum the SM supports. For compute
capability 8.9 the per-SM limits are:

| Per-SM limit (sm_89) | Value | Source |
|---|---|---|
| Resident warps | 48 (1,536 threads) | [M; N:nvidia2026adatuning] |
| Resident blocks | 24 | [M; N:nvidia2026adatuning] |
| 32-bit registers | 65,536 | [M; N:nvidia2026cudaguide] |
| Shared memory | 100 KB (102,400 B) | [M; N:nvidia2026adatuning] |

A block is resident only if **all four** limits are satisfied, so the number of blocks per SM is
the minimum over the four constraints.

### Worked example: our tiled all-pairs kernel

`ptxas` reports that `allpairs_tiled` uses **40 registers per thread with no spills**, and it was
launched with **256 threads per block** (`results/2026-10-17_smoke/summary.md`).

| Constraint | Calculation | Blocks / SM |
|---|---|---|
| Registers | 40 regs × 256 threads = 10,240 regs per block; ⌊65,536 / 10,240⌋ | **6** [D] |
| Warps | 256 threads = 8 warps per block; ⌊48 / 8⌋ | **6** [D] |
| Blocks | hardware limit | 24 [M] |
| Shared memory | depends on the tile size per block | ≥ 6, see note [D] |

The minimum is **6 blocks per SM = 1,536 threads = 48 warps, i.e. 100 % theoretical occupancy**
[D] (`results/2026-10-17_smoke/summary.md`). Across 80 SMs that is 122,880 resident threads
[D].

Three remarks follow from this calculation.

1. **Registers and warps bind at the same time.** Six blocks of 256 threads stay resident only
   while 6 × 256 × r ≤ 65,536, i.e. r ≤ 42 registers per thread [D]. At 43 registers the register
   limit drops to 5 blocks, which is 40 warps, or 83 % occupancy [D]. The kernel therefore has a
   headroom of only two registers. The hardware also allocates registers in fixed-size units, which
   may reduce this headroom further (`TODO(source needed: register allocation granularity for cc
   8.9)`). The naive kernel, at 39 registers, sits at the same 6-block limit
   (`results/2026-10-17_smoke/summary.md`).
2. **Shared memory is not the limit, on the assumption** that the tile holds one `float4`
   position (16 B) per thread, as in the classic design [P:nyland2007nbody]: 256 × 16 B = 4 KB per
   block, and 6 blocks need 24 KB out of 100 KB [D]. The actual shared-memory size of
   `allpairs_tiled` has not been recorded (`TODO(source needed: ptxas shared-memory usage for
   allpairs_tiled)`).
3. **Theoretical is not achieved occupancy.** The table gives the upper bound; Nsight Compute
   should report the achieved value before it is used in the results chapter.

The same arithmetic also shows the opposite problem, the one the load-balancing chapter addresses.
With N = 4,096 bodies and 256 threads per block, the grid has only 16 blocks for 80 SMs, so at
most 20 % of the SMs have any work; the smoke test measured 224 G interactions/s in that case
against 998 G interactions/s at N = 131,072 (`results/2026-10-17_smoke/summary.md`, **smoke-test
numbers, not benchmarks**). Full occupancy per SM is worthless if most SMs are idle.

## 2.4 Memory hierarchy

From fastest and smallest to slowest and largest, the levels available to a CUDA thread on this
GPU are as follows.

```
                         per thread / per SM                    whole GPU
  +----------------------------------------------+
  | Registers: 65,536 x 32-bit = 256 KB per SM   |  [M; N:whitepaper]   x 80 SMs = 20,480 KB [N]
  |            max 255 per thread                |  [M]
  +----------------------------------------------+
  | L1 data cache + shared memory: 128 KB per SM |  [N:whitepaper]      x 80 SMs = 10,240 KB [N]
  |   shared carve-out 0/8/16/32/64/100 KB       |  [N:adatuning]
  |   max 100 KB / SM, 99 KB / block (opt-in)    |  [M]
  +----------------------------------------------+
                         |
  +---------------------------------------------------------------+
  | L2 cache: 64 MB (67,108,864 B), shared by all 80 SMs          |  [M; N:whitepaper]
  +---------------------------------------------------------------+
                         |
  +---------------------------------------------------------------+
  | Device memory: 16 GB GDDR6X, 256-bit (8 x 32-bit controllers) |  [M; N:whitepaper]
  |   ~736 GB/s theoretical                                       |  [D]
  +---------------------------------------------------------------+
                         |
  +---------------------------------------------------------------+
  | Host: PCIe 4.0 x16, ~32 GB/s theoretical per direction        |  [M link; D rate]
  |   1 copy engine (asyncEngineCount = 1), Resizable BAR on      |  [M]
  +---------------------------------------------------------------+
```

*Figure 2.1: memory hierarchy of the RTX 4080 SUPER. Every number carries its source tag;
"whitepaper" = nvidia2023adawhitepaper, "adatuning" = nvidia2026adatuning.*

### 2.4.1 Registers and L1/shared memory

These are described in §2.2.4 and §2.2.5. Both are per-SM resources: data in one SM's shared
memory is not visible to another SM, which is why the tiled N-body kernel shares tiles only within
a block [P:nyland2007nbody].

### 2.4.2 The 64 MB L2 cache

The L2 cache is **64 MB** (67,108,864 B) [M], matching the 65,536 KB of the full AD103
[N:nvidia2023adawhitepaper]. This is unusually large: the previous-generation GA102 had 6 MB
[N:nvidia2023adawhitepaper], so the AD103 L2 is more than ten times larger [D].

For N-body the important question is which working sets fit in L2. One body position stored as a
`float4` (x, y, z, mass) is 16 B, so the L2 can hold 64 MB / 16 B ≈ 4.2 million positions [D].
With velocity stored alongside (32 B per body), it holds about 2.1 million bodies [D]. For the
all-pairs kernel, where every block streams through the whole position array once per step, this
means that every problem size up to a few million bodies can in principle be served from L2 rather
than DRAM after the first pass [D]. The N = 131,072 smoke-test run, for example, has a 2 MB
position array, which fits in L2 thirty-two times over [D]. Two cautions apply: not all of the L2 is
necessarily available to one array, and whether the hardware actually retains the array between
tiles has to be measured (`TODO(source needed: Nsight Compute L2 hit rate for allpairs_tiled)`).
The size of the L2 also means that DRAM-bandwidth arguments taken from older GPUs may not transfer
to this one.

### 2.4.3 Device memory (GDDR6X)

The board has 16 GB of device memory; `nvidia-smi` reports 16,376 MiB total with 331 MiB reserved
[M]. The memory type is GDDR6X [S] (`TODO(source needed: RTX 4080 SUPER memory type; the
whitepaper gives GDDR6X only for the non-SUPER RTX 4080)`). The interface is **256 bits wide**
[M], made of eight 32-bit memory controllers on the full AD103 [N:nvidia2023adawhitepaper].

The driver reports a maximum memory clock of 11,501 MHz [M], which `docs/01-hardware.md` reads as
23 Gbps effective per pin [M/D]. The theoretical bandwidth is then
23 Gbps × 256 bit / 8 = **736 GB/s** [D]. This is a derived figure, not a measurement; it rests on
an unsourced conversion from reported clock to data rate (`TODO(source needed: RTX 4080 SUPER
memory data rate, or a measured device-to-device bandwidth)`). For comparison, the whitepaper gives
22.4 Gbps for the non-SUPER RTX 4080 [N:nvidia2023adawhitepaper].

Because the all-pairs kernel performs O(N²) arithmetic on O(N) data, it is compute-bound rather
than bandwidth-bound for large N (`docs/01-hardware.md` §5; `references/nvidia2026adatuning.md`),
so DRAM bandwidth matters mostly for the integrator and for tree-building phases.

### 2.4.4 Host interface: PCIe, copy engine and Resizable BAR

The GPU is connected through **PCIe Gen 4 × 16**; the host supports Gen 5 but the card's maximum
is Gen 4 [M]. The theoretical rate is about 32 GB/s per direction [D] (`TODO(source needed: PCIe
4.0 per-lane rate and encoding)`).

The CUDA runtime reports **`asyncEngineCount = 1`** [M]. With a single copy engine, a
host-to-device copy and a device-to-host copy cannot be in flight at the same time
(`docs/01-hardware.md` §2.2). Whether one copy can overlap with kernel execution on this
configuration is expected but not yet sourced or measured (`TODO(source needed: copy/compute
overlap semantics for asyncEngineCount = 1)`). For a simulation that
keeps all state on the GPU and only reads results back for output, this rarely matters; it matters
for any CPU+GPU split (R3 #8), where data must flow both ways each step.

The BAR1 aperture is 16,384 MiB, the full size of device memory, from which `docs/01-hardware.md`
concludes that **Resizable BAR is enabled** [M/D]. The device also supports concurrent kernels and
unified addressing [M].

For visualisation, positions can be drawn directly from a buffer shared between CUDA and the
graphics API, avoiding a PCIe round trip per frame (`docs/01-hardware.md` §5).

## 2.5 Precision: FP32 versus FP64

At the CUDA-runtime clock of 2,610 MHz [M], the theoretical FP32 peak is
10,240 lanes × 2 FLOP per FMA × 2.61 GHz ≈ **53.5 TFLOPS** [D]. FP64 runs at 1/64 of the FP32 rate
[N:nvidia2023adawhitepaper], giving about **0.84 TFLOPS** [D]. A double-precision force kernel
would therefore be about 64 times slower than the same kernel in single precision [D].

The project uses **FP32 for all force kernels** (`CLAUDE.md` rule 3), storing bodies as `float4`
and using `rsqrtf` (`docs/01-hardware.md` §5). The accuracy cost of FP32 is handled in the
integrator and, if needed, by compensated summation rather than by FP64 arithmetic
(`docs/01-hardware.md` §5). As a first indication that FP32 is adequate, the smoke test integrated
4,096 bodies for 2,000 kick-drift-kick steps with a relative energy drift of −2.8 × 10⁻⁸
(`results/2026-10-17_smoke/summary.md`, **smoke-test number**).

Note that the 53.5 TFLOPS figure is not a hard ceiling: the SM clock rises above 2,610 MHz under
load (§2.6), and the "20 FLOP per interaction" accounting used to compare against it
[P:nyland2007nbody] counts `rsqrtf` as several FLOPs even though it executes on the SFU
(`results/2026-10-17_smoke/summary.md`).

## 2.6 Platform conditions that affect measurements

**Driver model and display.** The GPU runs under the Windows WDDM driver model with a display
attached and active [M], a 2560 × 1440 monitor at 180 Hz [M]. Under WDDM the Windows TDR
watchdog terminates kernels that run longer than about 2 s (`docs/01-hardware.md` §4;
`TODO(source needed: Microsoft documentation of the default TdrDelay)`), so every kernel launch
in this project is kept to a single timestep (`CLAUDE.md` rule 4).

**Power limit.** The board's default power limit is 320 W, but the current limit is **272 W**
(85 % of default), within an allowed range of 150–352 W [M]. The driver's software power-capping
counter already shows 1,683 s (about 28 minutes) of power-capped time [M]. A lower power limit
directly lowers sustained clocks for a compute-bound kernel, so the limit in force is recorded
with every benchmark (`docs/01-hardware.md` §2.5).

**Clocks.** Several clock values describe this GPU, and none of them alone is "the" clock:

| Clock | Value | Source |
|---|---|---|
| Reported by the CUDA runtime (`clockRate`) | 2,610 MHz | [M] |
| Observed by `nvidia-smi` right after the smoke runs | 2,745 MHz | `results/2026-10-17_smoke/summary.md` |
| Maximum SM clock reported by `nvidia-smi` | 3,105 MHz | [M] |
| Idle SM clock at capture time | 345 MHz | [M] |
| NVIDIA reference base / boost for the 4080 SUPER | — | `TODO(source needed: NVIDIA RTX 4080 SUPER product page)` |

The sustained clock during a run depends on temperature and power (`docs/01-hardware.md` §4),
which is why benchmark runs report the median of several repetitions and log the SM clock, power
draw and temperature alongside the timings.

**Background load.** At capture time the GPU was at 41 % utilisation with 5,198 MiB of memory in
use and three active NVENC encoder sessions [M], from desktop applications
(`docs/01-hardware.md` §4). Benchmarks are run after closing them.

## 2.7 Summary: measured versus specified values

| Property | Full AD103 (NVIDIA) | Our RTX 4080 SUPER | Source |
|---|---|---|---|
| Compute capability | — | 8.9 (`sm_89`) | [M] |
| GPCs / TPCs | 7 / 40 | not measured; 7 GPCs listed as spec | [N:nvidia2023adawhitepaper]; [S] |
| SMs | 80 | 80 | [N:nvidia2023adawhitepaper]; [M] |
| FP32 CUDA cores | 128 / SM, 10,240 total | 10,240 | [N:nvidia2023adawhitepaper]; [D] |
| SM partitions | — | 4 × 32 FP32 lanes | [S], unconfirmed |
| Tensor cores (4th gen) | 320 | 320 | [N:nvidia2023adawhitepaper]; [S] |
| RT cores (3rd gen) | 80 | 80 | [N:nvidia2023adawhitepaper]; [S] |
| FP64 units / rate | 2 per SM / 1/64 FP32 | 1/64 FP32 | [N:nvidia2023adawhitepaper] |
| Register file | 256 KB / SM | 65,536 regs / SM, ≤ 255 / thread | [N:nvidia2023adawhitepaper]; [M] |
| L1 + shared memory | 128 KB / SM | — | [N:nvidia2023adawhitepaper] |
| Shared memory / SM | up to 100 KB (carve-out 0–100 KB) | 102,400 B | [N:nvidia2026adatuning]; [M] |
| Shared memory / block | 99 KB opt-in | 101,376 B | [N:nvidia2026cudaguide]; [M] |
| Resident warps / threads per SM | 48 / 1,536 | 1,536 threads | [N:nvidia2026adatuning]; [M] |
| Resident blocks per SM | 24 | 24 | [N:nvidia2026adatuning]; [M] |
| Max threads per block | — | 1,024 | [S], unconfirmed |
| Warp size | — | 32 | [M] |
| L2 cache | 65,536 KB | 67,108,864 B (64 MB) | [N:nvidia2023adawhitepaper]; [M] |
| Memory interface | 8 × 32-bit controllers | 256-bit | [N:nvidia2023adawhitepaper]; [M] |
| Device memory | — | 16,376 MiB (331 MiB reserved), GDDR6X | [M]; [S] for the type |
| Memory clock / data rate | — | 11,501 MHz max reported; 23 Gbps | [M]; [D] |
| Memory bandwidth (theoretical) | — | ≈ 736 GB/s | [D] |
| Host link | — | PCIe Gen 4 × 16 | [M] |
| Host bandwidth (theoretical) | — | ≈ 32 GB/s per direction | [D] |
| Copy engines | — | 1 | [M] |
| BAR1 / Resizable BAR | — | 16,384 MiB / enabled | [M]; [D] |
| SM clock (CUDA runtime / max) | — | 2,610 / 3,105 MHz | [M] |
| Reference base / boost clock | — | — | TODO(source needed) |
| Peak FP32 / FP64 @ 2,610 MHz | — | ≈ 53.5 / ≈ 0.84 TFLOPS | [D] |
| Power limit (current / default) | — | 272 W / 320 W | [M] |
| Driver model | — | WDDM, display attached | [M] |

## 2.8 What this means for N-body (bridge to the following chapters)

All smoke-test numbers in this section come from `results/2026-10-17_smoke/summary.md` and were
**recorded under background load at 272 W; they are not benchmarks.**

- **The SM is the unit of balance.** Per SM, the tiled kernel already reaches 100 % theoretical
  occupancy (§2.3) [D]. Across the chip, small problems leave SMs idle: 16 blocks for 80 SMs at
  N = 4,096. This is the measured motivation for the load-balancing techniques of chapter 4.
- **Shared-memory tiling pays off.** The tiled kernel was 2.6× faster than the naive one at
  N = 16,384 (smoke-test number), as expected from tile reuse [P:nyland2007nbody].
- **Arithmetic, not bandwidth, is the expected limit.** At N = 131,072 the smoke test reached about
  1.0 × 10¹² interactions/s, about 37 % of the 53.5 TFLOPS FP32 peak by the 20-FLOP convention
  (smoke-test number). Whether the gap is due to the SFU (`rsqrtf`) or to instruction issue is the
  **hypothesis** in §2.2.3, to be tested with Nsight Compute.
- **The L2 changes the picture for large N.** Position arrays of up to about four million bodies
  fit in the 64 MB L2 (§2.4.2) [D], so tiling experiments should measure L2 hit rates rather than
  assume DRAM traffic.
