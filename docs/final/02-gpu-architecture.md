# 2. Target GPU Architecture: NVIDIA GeForce RTX 4080 SUPER (Ada Lovelace, AD103)

> **Language note:** drafted in English; the Bulgarian/English choice for the final document is still open (`docs/00-course-requirements.md`, open questions).

**Source tags used in this chapter** (project convention, `CLAUDE.md` rule 5):
**[M]** measured on this machine on 2026-10-17 (`docs/raw/devicequery_2026-10-17.txt`,
`docs/raw/nvidia-smi-q_2026-10-17.txt`, or `docs/01-hardware.md` rows tagged M) ·
**[S]** public NVIDIA specification as recorded in `docs/01-hardware.md` ·
**[D]** derived arithmetically from tagged values ·
**[N:key]** NVIDIA, Microsoft or standards-body document in `references/references.bib`, with the
section, table or page where the fact appears ·
**[P:key]** paper in `references/references.bib`.
Where a fact is still missing, the text says `TODO(source needed: …)` for a literature gap or
`TODO(measure: …)` for something that needs a build or profiler run.

---

## 2.1 Overview and chip hierarchy

The test machine has a single NVIDIA GeForce RTX 4080 SUPER with 16 GB of memory, built by ASUS
(subsystem ID 0x89621043) [M]. The GPU belongs to the Ada Lovelace architecture [M], uses the
AD103-400-A1 die (device ID 0x2702) [M] and reports compute capability 8.9, so all device code in
this project is compiled for `sm_89` [M].

An Ada GPU is organised hierarchically. The chip is divided into Graphics Processing Clusters
(GPCs); each GPC contains Texture Processing Clusters (TPCs); each TPC contains
"one PolyMorph Engine and two SMs" [N:nvidia2023adawhitepaper, p. 8]. The Streaming
Multiprocessors (SMs) are the units that execute CUDA thread blocks. The full AD103 die has
**7 GPCs, 40 TPCs and 80 SMs** [N:nvidia2023adawhitepaper, App. B p. 32].

Our board reports **80 SMs** through the CUDA runtime [M], and NVIDIA lists 10,240 CUDA cores for
the RTX 4080 SUPER [N:nvidia2026rtx4080family, "GPU Engine Specs"], which is 80 SMs × 128 cores
[D]. The RTX 4080 SUPER therefore uses the complete AD103, with no SMs disabled [D]. For
contrast, the non-SUPER RTX 4080 uses a cut-down AD103 with 76 SMs and 9,728 CUDA cores
[N:nvidia2023adawhitepaper, App. B; N:nvidia2026rtx4080family].

The whitepaper's description of "six TPCs" per GPC refers to the AD102 GPC
[N:nvidia2023adawhitepaper, p. 8, Fig. 2]. For AD103 it gives only the totals, and 40 TPCs cannot
be split evenly over 7 GPCs [D], so the GPCs of AD103 are not all identical
(`TODO(source needed: AD103 TPCs per GPC; not stated in the whitepaper, Programming Guide §5.1 or
Ada Tuning Guide)`).

## 2.2 The Ada streaming multiprocessor (SM) in depth

The SM is where all of the N-body force computation happens, so this is the most important
section of the chapter.

### 2.2.1 Processing partitions and schedulers

Each Ada SM contains **128 FP32 CUDA cores** [N:nvidia2023adawhitepaper, p. 8] and
"is divided into four processing blocks (or partitions)" [N:nvidia2023adawhitepaper, p. 10].
Each partition contains [N:nvidia2023adawhitepaper, p. 10]:

- one warp scheduler and one dispatch unit,
- a 64 KB register file,
- 16 CUDA cores dedicated to FP32 and 16 that can process FP32 **or** INT32 (32 FP32 lanes in
  total),
- one fourth-generation Tensor Core,
- four load/store units,
- one Special Function Unit (SFU),
- an L0 instruction cache.

The SM therefore has four warp schedulers [D]. A warp (32 threads [M; N:nvidia2026cudaguide,
§5.1.3 Table 30]) is issued by the scheduler of the partition it belongs to. The number of
instructions a scheduler can issue per clock is not documented in any of our sources
(`TODO(source needed: warp-scheduler issue rate per clock on Ada; not in the whitepaper, PG §5.1,
Best Practices §11–12 or the Ada Tuning Guide)`).

Compared with compute capability 8.0 (A100), NVIDIA states that sm_89 executes
"2x more FP32 operations per cycle per SM" [N:nvidia2026adatuning]. This doubled FP32 rate is
the main reason the all-pairs N-body force loop, which is dominated by FP32 multiply-adds, is
compute-bound on this GPU (`references/nvidia2026adatuning.md`).

### 2.2.2 FP32 and INT32 datapaths

The 128 FP32 lanes per SM perform fused multiply-add (FMA), the operation that dominates the
force computation. The Best Practices Guide gives **128 FP32 add/multiply/FMA results per clock
per SM** for compute capability 8.9 [N:nvidia2026bestpractices, §12.1.1 Table 5, column
"8.6 / 8.9"]. Counting an FMA as two floating-point operations gives the peak rate in §2.5 [D].

The FP32 and INT32 datapaths are not independent. In each partition, half of the 32 lanes can
execute either 16 FP32 or 16 INT32 operations per clock [N:nvidia2023adawhitepaper, p. 10]. In a
cycle where a partition uses that half for integer work (index arithmetic, loop counters), only
the 16 dedicated lanes remain for FP32 [D]. A tight FP32 force loop with little integer work can
use all 32 lanes.

**An inconsistency between two NVIDIA sources.** The whitepaper's layout implies at most
16 INT32 operations per clock per partition, i.e. 64 per SM [D, from N:nvidia2023adawhitepaper
p. 10], while the throughput table lists 128 results per clock per SM for 32-bit integer add on
cc 8.9 [N:nvidia2026bestpractices, §12.1.1 Table 5]. We do not resolve this and quote no INT32
peak; it does not affect the FP32 or SFU figures used below.

### 2.2.3 Special function units (SFU / MUFU)

Transcendental functions and reciprocal square roots are executed on the SFU, one per partition,
"which executes transcendental and graphics interpolation instructions"
[N:nvidia2023adawhitepaper, p. 10]; in SASS these appear as `MUFU` instructions
(`results/2026-10-17_smoke/summary.md`). This matters for N-body, because every pairwise
interaction computes one reciprocal square root, `rsqrtf(r² + ε²)`. NVIDIA recommends calling
`rsqrtf()` explicitly, because the compiler only rewrites `1.0f/sqrtf(x)` when IEEE-754 semantics
allow it [N:nvidia2026bestpractices, §12.1.6].

For cc 8.9 the throughput of 32-bit approximate reciprocal square root (and of reciprocal, log2,
exp2, sine and cosine) is **16 results per clock per SM**, against 128 for FP32 FMA
[N:nvidia2026bestpractices, §12.1.1 Table 5, column "8.6 / 8.9"], a ratio of **8 : 1** [D]. With
one SFU per partition, that is 4 results per clock per partition [D], so one warp-wide `rsqrtf`
occupies a partition's SFU for 8 clocks, while one warp-wide FP32 instruction occupies the FP32
lanes for 1 clock [D].

This 8 : 1 ratio is the SFU's own pipe throughput. It is **not** an issue limit: the scheduler
can issue other instructions to the FP32 lanes while the SFU is busy, and the scheduler's issue
rate is not documented (§2.2.1). Whether the force kernel is limited by the SFU or by the FP32
lanes therefore depends on its instruction mix per interaction. If each interaction contains
eight or more FP32 instructions per `rsqrtf`, the FP32 pipe is the tighter limit; if fewer, the
SFU is [D].

**Hypothesis, not a result.** The smoke-test notes suspect the kernel is bound by MUFU
throughput or instruction issue (`results/2026-10-17_smoke/summary.md`, observation 2). This
remains a hypothesis until Nsight Compute reports the per-pipe utilisation
(`TODO(measure: Nsight Compute pipe utilisation, SFU/XU vs FMA, for allpairs_tiled, plus the SASS
instruction count per interaction)`).

### 2.2.4 Register file

Each SM has a **256 KB register file** [N:nvidia2023adawhitepaper, p. 8], made of one 64 KB file
per partition [N:nvidia2023adawhitepaper, p. 10]. That is **65,536 32-bit registers** per SM
[M; N:nvidia2026cudaguide, §5.1.3 Table 31], and 20,480 KB for the whole AD103
[N:nvidia2023adawhitepaper, App. B]. A single thread can use at most 255 registers
[M; N:nvidia2026cudaguide, §5.1.3 Table 31]. Registers are the fastest storage on the chip and
are private to a thread; in the N-body kernel they hold the body's own position, its accumulated
acceleration and the loop temporaries. Because the register file is shared by all resident
threads, register use per thread is one of the limits on occupancy (§2.3).

### 2.2.5 Unified L1 data cache and shared memory

Each Ada SM has **128 KB of combined L1 data cache and shared memory**
[N:nvidia2023adawhitepaper; N:nvidia2026cudaguide, §5.1], 10,240 KB across the full AD103
[N:nvidia2023adawhitepaper, App. B]. The same physical SRAM is divided between hardware-managed L1
cache and programmer-managed shared memory. The shared-memory part (the *carve-out*) can be set
to **0, 8, 16, 32, 64 or 100 KB** per SM [N:nvidia2026adatuning], with the remainder used as L1
[D]. At the maximum carve-out, 100 KB is shared memory and 28 KB is left for L1 [D].

The CUDA runtime on our board confirms the per-SM shared-memory maximum of **102,400 B (100 KB)**
[M; N:nvidia2026adatuning] and a per-block maximum of **101,376 B (99 KB)**
[M; N:nvidia2026cudaguide, §5.1.3 Table 31]. A kernel that needs more than 48 KB per block must
use dynamic shared memory and an explicit opt-in through `cudaFuncSetAttribute` with
`cudaFuncAttributeMaxDynamicSharedMemorySize` [N:nvidia2026cudaguide, §3.2.6 and Table 31
footnote 3].

Shared memory is the key resource for all-pairs N-body. In the tiled algorithm each thread block
loads a tile of body positions into shared memory once and every thread in the block reuses it,
which turns most global-memory reads into shared-memory reads [P:nyland2007nbody]. The carve-out
options above bound how large a tile can be (`references/nvidia2026adatuning.md`).

### 2.2.6 Tensor cores and RT cores (and why N-body FP32 does not use them)

The full AD103 has **320 fourth-generation Tensor cores** and **80 third-generation RT cores**
[N:nvidia2023adawhitepaper, App. B], i.e. one Tensor core per partition, four per SM
[N:nvidia2023adawhitepaper, p. 10], and one RT core per SM [D]. NVIDIA describes the Tensor cores
as "tailored for the matrix multiply and accumulate math operations", and Ada includes the FP8
Transformer Engine from Hopper [N:nvidia2023adawhitepaper, p. 24]. The direct all-pairs force
computation contains a reciprocal square root and a data-dependent scaling per pair rather than a
dense matrix product, so our FP32 force kernels are written for the CUDA cores and SFUs only. That
the compiled kernels contain no tensor instructions still has to be checked
(`TODO(measure: cuobjdump -sass listing of allpairs_tiled)`).

RT cores perform hardware traversal of bounding-volume hierarchies. Nagarajan et al. reformulate
Barnes-Hut as a ray-tracing problem and use RT cores, through OptiX, for the node-opening
decisions [P:nagarajan2025rtbarneshut]. That route therefore exists on our hardware, but it
requires OptiX rather than plain CUDA compute and is out of scope for the all-pairs FP32 baseline
(`references/nagarajan2025rtbarneshut.md`).

### 2.2.7 FP64 units

The Ada SM contains only **2 FP64 units**, and NVIDIA states that
"The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations." [N:nvidia2023adawhitepaper].
The CUDA Programming Guide gives the same 64 : 1 FP32-to-FP64 throughput ratio for cc 8.9
[N:nvidia2026cudaguide, §5.1.3 Table 30], and the Best Practices Guide lists 2 FP64 results per
clock per SM [N:nvidia2026bestpractices, §12.1.1 Table 5]. The consequences are discussed in §2.5.

## 2.3 Execution model on sm_89: warps, blocks and occupancy

A CUDA kernel is launched as a grid of thread blocks. All threads of a block run on a single SM
and, in most cases, run to completion on that SM; there is no guaranteed order in which blocks are
scheduled [N:nvidia2026cudaguide, §1.2.2.1]. Inside the SM, threads are organised into
"groups of 32 threads called warps" and executed in SIMT fashion [N:nvidia2026cudaguide,
§1.2.2.2]. A block can contain at most **1,024 threads** [N:nvidia2026cudaguide, §5.1.3
Table 30]. Several blocks can be resident on one SM at the same time (up to 24 on cc 8.9 [M]),
and when one warp waits on memory "the SM switches to execute another warp"
[N:nvidia2026cudaguide, §3.2.5]. NVIDIA therefore ties latency hiding to the number of active
warps per SM [N:nvidia2026bestpractices, §11.3].

*Occupancy* is the ratio of active warps to the maximum the SM supports
[N:nvidia2026cudaguide, §2.3.7]. For compute capability 8.9 the per-SM limits are:

| Per-SM limit (sm_89) | Value | Source |
|---|---|---|
| Resident warps | 48 (1,536 threads) | [M; N:nvidia2026adatuning; N:nvidia2026cudaguide §5.1.3 Table 30] |
| Resident blocks | 24 | [M; N:nvidia2026adatuning; N:nvidia2026cudaguide §5.1.3 Table 30] |
| 32-bit registers | 65,536 | [M; N:nvidia2026cudaguide §5.1.3 Table 31] |
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
| Shared memory | depends on the tile size per block | ≥ 6 on the assumption below [D] |

The minimum is **6 blocks per SM = 1,536 threads = 48 warps, i.e. 100 % theoretical occupancy**
[D] (`results/2026-10-17_smoke/summary.md`). Across 80 SMs that is 122,880 resident threads
[D].

Three remarks follow from this calculation.

1. **Registers and warps bind at the same time, and the register headroom may be zero.** With
   registers counted per thread, six blocks of 256 threads fit while 6 × 256 × r ≤ 65,536, i.e.
   up to r = 42 registers, a headroom of two registers [D]. The Best Practices Guide, however,
   states that "register allocations are rounded up to the nearest 256 registers per warp"
   [N:nvidia2026bestpractices, §11.1.1]. That statement accompanies a cc 7.0 example, and the
   same paragraph notes that the granularity varies by compute capability, so it is not confirmed
   for cc 8.9. **If it applies to cc 8.9**, then 40 registers × 32 threads = 1,280 = exactly
   5 × 256 per warp, and 6 blocks fit; at 41 registers each warp rounds up to 1,536 registers, a
   block needs 12,288, and only ⌊65,536 / 12,288⌋ = 5 blocks fit, i.e. 40 warps or 83 %
   occupancy [D]. The headroom would then be **0 registers, not 2**
   (`TODO(measure: Nsight Compute launch__occupancy_limit_registers at 40 and at 41+ registers
   per thread)`). The naive kernel, at 39 registers, also sits at 6 blocks
   (`results/2026-10-17_smoke/summary.md`).
2. **Shared memory is not the limit, on the assumption** that the tile holds one `float4`
   position (16 B) per thread, as in the classic design [P:nyland2007nbody]: 256 × 16 B = 4 KB per
   block, and 6 blocks need 24 KB out of 100 KB [D]. The actual shared-memory size of
   `allpairs_tiled` has not been recorded; `nvcc --resource-usage` reports it
   [N:nvidia2026cudaguide, §2.3.7] (`TODO(measure: ptxas shared-memory usage for
   allpairs_tiled)`).
3. **Theoretical is not achieved occupancy.** The table gives the upper bound; Nsight Compute
   should report the achieved value before it is used in the results chapter.

The same arithmetic also shows the opposite problem, the one the load-balancing chapter addresses.
NVIDIA advises launching more blocks than there are SMs [N:nvidia2026bestpractices, §11.3]. With
N = 4,096 bodies and 256 threads per block, the grid has only 16 blocks for 80 SMs, so at most
20 % of the SMs have any work; the smoke test measured 224 G interactions/s in that case against
998 G interactions/s at N = 131,072 (`results/2026-10-17_smoke/summary.md`, **smoke-test numbers,
not benchmarks**). Full occupancy per SM does not help if most SMs are idle.

## 2.4 Memory hierarchy

From fastest and smallest to slowest and largest, the levels available to a CUDA thread on this
GPU are as follows.

```
                         per thread / per SM                    whole GPU
  +----------------------------------------------+
  | Registers: 65,536 x 32-bit = 256 KB per SM   |  [M; N:wp p.8]       x 80 SMs = 20,480 KB [N:wp]
  |   4 x 64 KB (one file per partition)         |  [N:wp p.10]
  |   max 255 per thread                         |  [M; N:pg T31]
  +----------------------------------------------+
  | L1 data cache + shared memory: 128 KB per SM |  [N:wp]              x 80 SMs = 10,240 KB [N:wp]
  |   shared carve-out 0/8/16/32/64/100 KB       |  [N:at]
  |   max 100 KB / SM, 99 KB / block (opt-in)    |  [M; N:pg T31]
  +----------------------------------------------+
                         |
  +---------------------------------------------------------------+
  | L2 cache: 64 MB (67,108,864 B), shared by all 80 SMs          |  [M; N:wp App. B]
  +---------------------------------------------------------------+
                         |
  +---------------------------------------------------------------+
  | Device memory: 16 GB GDDR6X, 256-bit, 23 Gbps                 |  [M; N:rtx; N:burnes]
  |   = 736 GB/s theoretical (not measured)                       |  [D]
  +---------------------------------------------------------------+
                         |
  +---------------------------------------------------------------+
  | Host: PCIe 4.0 x16, ~31.5 GB/s theoretical per direction      |  [M link; D rate]
  |   1 copy engine (asyncEngineCount = 1), Resizable BAR on      |  [M; D]
  +---------------------------------------------------------------+
```

*Figure 2.1: memory hierarchy of the RTX 4080 SUPER. Every number carries its source tag.
wp = nvidia2023adawhitepaper, at = nvidia2026adatuning, pg = nvidia2026cudaguide (§5.1.3),
rtx = nvidia2026rtx4080family, burnes = burnes2024rtx40super.*

### 2.4.1 Registers and L1/shared memory

These are described in §2.2.4 and §2.2.5. Both are per-SM resources: all threads of a block run
on the same SM [N:nvidia2026cudaguide, §1.2.2.1], so a block's shared memory is not visible to
blocks on other SMs. This is why the tiled N-body kernel shares tiles only within a block
[P:nyland2007nbody].

### 2.4.2 The 64 MB L2 cache

The L2 cache is **64 MB** (67,108,864 B) [M], matching the 65,536 KB of the full AD103
[N:nvidia2023adawhitepaper, App. B]. This is unusually large: the previous-generation GA102 had
6 MB [N:nvidia2023adawhitepaper], so the AD103 L2 is more than ten times larger [D].

For N-body the important question is which working sets fit in L2. One body position stored as a
`float4` (x, y, z, mass) is 16 B, so the L2 can hold 64 MB / 16 B ≈ 4.2 million positions [D].
With velocity stored alongside (32 B per body), it holds about 2.1 million bodies [D]. For the
all-pairs kernel, where every block streams through the whole position array once per step, this
means that problem sizes up to a few million bodies can in principle be served from L2 rather than
DRAM after the first pass [D]. The N = 131,072 smoke-test run, for example, has a 2 MB position
array, which fits in L2 thirty-two times over [D]. Two cautions apply: not all of the L2 is
necessarily available to one array, and whether the hardware actually keeps the array resident
between tiles has to be measured (`TODO(measure: Nsight Compute L2 hit rate for allpairs_tiled
at several N)`). The size of the L2 also means that DRAM-bandwidth arguments taken from older
GPUs may not transfer to this one.

### 2.4.3 Device memory (GDDR6X)

The board has 16 GB of device memory; `nvidia-smi` reports 16,376 MiB total with 331 MiB reserved
[M]. NVIDIA specifies **GDDR6X** on a **256-bit** interface for the RTX 4080 SUPER
[N:nvidia2026rtx4080family, "Memory Specs"], and our board reports a 256-bit bus [M]; on the full
AD103 this is eight 32-bit memory controllers [N:nvidia2023adawhitepaper]. NVIDIA's launch article
gives the memory data rate as **23 Gbps** [N:burnes2024rtx40super]; the product page does not list
it [N:nvidia2026rtx4080family]. The theoretical bandwidth is therefore
23 Gbps × 256 bit / 8 = **736 GB/s** [D]. This is a derived peak, not a measurement
(`TODO(measure: device-to-device copy bandwidth, bench-profiler)`).

The clock readings from the driver add a caveat. `nvidia-smi` reports a maximum memory clock of
11,501 MHz but a *current* memory clock of 11,701 MHz at capture time, i.e. above its own maximum
[M] (`docs/01-hardware.md` §2.4 flags this as either a factory overclock or a reporting quirk).
Until the bandwidth is measured, the operating data rate of this ASUS board is not known exactly.

Because the all-pairs kernel performs O(N²) arithmetic on O(N) data, it is compute-bound rather
than bandwidth-bound for large N (`docs/01-hardware.md` §5; `references/nvidia2026adatuning.md`),
so DRAM bandwidth matters mostly for the integrator and for tree-building phases.

### 2.4.4 Host interface: PCIe, copy engine and Resizable BAR

The GPU is connected through **PCIe Gen 4 × 16**; the host supports Gen 5 but the card's maximum
is Gen 4 [M], consistent with NVIDIA's specification [N:nvidia2026rtx4080family]. PCIe 4.0 signals
at 16 GT/s per lane [N:pcisig2011pcie4] with 128b/130b encoding [N:solomon2014pcie16gts], so a
×16 link carries 16 × 16 × 128/130 / 8 ≈ **31.5 GB/s per direction** in theory [D]. The primary
PCI-SIG specification could not be fetched (pcisig.com returned a bot-check page), so the rate
comes from a reprint of PCI-SIG's 2011 press release and the encoding from a 2014 secondary
vendor article written before the final specification (`references/pcisig2011pcie4.md`,
`references/solomon2014pcie16gts.md`; `TODO(source needed: PCI-SIG primary source for PCIe 4.0
128b/130b encoding)`).

The CUDA runtime reports **`asyncEngineCount = 1`** [M]. With a single copy engine the GPU
"can perform one asynchronous data transfer and execute kernels" concurrently
[N:nvidia2026bestpractices, §10.1.2], provided the host memory is pinned and the copy and kernel
are in different non-default streams [N:nvidia2026bestpractices, §10.1.2;
N:nvidia2026cudaguide, §2.5]. A host-to-device and a device-to-host copy, however, cannot be in
flight at the same time [N:nvidia2026bestpractices, §10.1.2]. For a simulation that keeps all
state on the GPU and only reads results back for output, this rarely matters; it matters for any
CPU+GPU split (R3 #8), where data must flow both ways each step.

The BAR1 aperture is 16,384 MiB [M], the full size of device memory, from which we infer that
**Resizable BAR is enabled** [D]; the card supports it [N:nvidia2026rtx4080family]. The device
also supports concurrent kernels and unified addressing [M].

For visualisation, positions can be drawn directly from a buffer shared between CUDA and the
graphics API, avoiding a PCIe round trip per frame (`docs/01-hardware.md` §5).

## 2.5 Precision: FP32 versus FP64

The theoretical FP32 peak is 10,240 lanes × 2 FLOP per FMA × clock [D]. At NVIDIA's reference
boost clock of 2.55 GHz [N:nvidia2026rtx4080family] this is ≈ **52.2 TFLOPS** [D], consistent with
the "52 TFLOPS" NVIDIA lists [N:nvidia2026rtx4080family]. At the 2,610 MHz the CUDA runtime reports
on our board [M] it is ≈ **53.5 TFLOPS** [D]. Neither figure is an upper bound: 2,745 MHz was
observed after the smoke runs (`results/2026-10-17_smoke/summary.md`), and at the 3,105 MHz
ceiling [M] the figure would be ≈ 63.6 TFLOPS [D] (`docs/01-hardware.md` §2.2).

FP64 runs at 1/64 of the FP32 rate [N:nvidia2023adawhitepaper; N:nvidia2026cudaguide, §5.1.3
Table 30], about **0.84 TFLOPS** at 2,610 MHz [D]. A double-precision force kernel would therefore
be about 64 times slower than the same kernel in single precision [D].

The project uses **FP32 for all force kernels** (`CLAUDE.md` rule 3), storing bodies as `float4`
and using `rsqrtf` (`docs/01-hardware.md` §5). The accuracy cost of FP32 is handled in the
integrator and, if needed, by compensated summation rather than by FP64 arithmetic
(`docs/01-hardware.md` §5). As a first indication that FP32 is adequate, the smoke test integrated
4,096 bodies for 2,000 kick-drift-kick steps with a relative energy drift of −2.8 × 10⁻⁸
(`results/2026-10-17_smoke/summary.md`, **smoke-test number**).

The "20 FLOP per interaction" accounting used to compare a kernel against this peak
[P:nyland2007nbody] counts `rsqrtf` as several FLOPs, even though it executes on the SFU at
one-eighth of the FMA rate (§2.2.3). Percentages of peak computed this way are a convention, not
a measure of FMA utilisation.

## 2.6 Platform conditions that affect measurements

**Driver model and display.** The GPU runs under the Windows WDDM driver model with a display
attached and active [M], a 2560 × 1440 monitor at 180 Hz [M]. Windows triggers a timeout
detection and recovery (TDR) when the GPU does not yield to a preemption request within
`TdrDelay`, whose default is 2 seconds; Microsoft calls it "effectively the timeout threshold"
[N:microsoft2025tdrkeys, section "TdrDelay"]. Microsoft advises end users not to change these keys
outside driver testing [N:microsoft2025tdrkeys], so every kernel launch in this project is kept to
a single timestep (`CLAUDE.md` rule 4).

**Power limit.** NVIDIA rates the RTX 4080 SUPER at 320 W total graphics power
[N:nvidia2026rtx4080family], which matches the board's default power limit of 320 W [M]. The
current limit, however, is **272 W** (85 % of default), within an allowed range of 150–352 W [M].
The driver's software power-capping counter already shows 1,683 s (about 28 minutes) of
power-capped time [M]. A lower power limit directly lowers sustained clocks for a compute-bound
kernel, so the limit in force is recorded with every benchmark (`docs/01-hardware.md` §2.5).

**Clocks.** Several clock values describe this GPU, and none of them alone is "the" clock:

| Clock | Value | Source |
|---|---|---|
| NVIDIA reference base / boost (Founders Edition) | 2.29 GHz / 2.55 GHz | [N:nvidia2026rtx4080family, "GPU Engine Specs"] |
| Reported by the CUDA runtime (`clockRate`) | 2,610 MHz | [M] |
| Observed by `nvidia-smi` right after the smoke runs | 2,745 MHz | `results/2026-10-17_smoke/summary.md` |
| Maximum SM clock reported by `nvidia-smi` | 3,105 MHz | [M] |
| Idle SM clock at capture time | 345 MHz | [M] |

NVIDIA's page notes that partner boards may differ from the Founders Edition figures
[N:nvidia2026rtx4080family]; ours is an ASUS board [M]. NVIDIA gives the base clock as 2.29 GHz;
the 2,295 MHz value used in earlier project notes has no source at that precision. The sustained
clock during a run depends on temperature and power (`docs/01-hardware.md` §4), which is why
benchmark runs report the median of several repetitions and log the SM clock, power draw and
temperature alongside the timings.

**Background load.** At capture time the GPU was at 41 % utilisation with 5,198 MiB of memory in
use and three active NVENC encoder sessions [M], from desktop applications
(`docs/01-hardware.md` §4). Benchmarks are run after closing them.

## 2.7 Summary: measured versus specified values

| Property | NVIDIA specification | Our RTX 4080 SUPER (measured / derived) | Source |
|---|---|---|---|
| Compute capability | — | 8.9 (`sm_89`) | [M] |
| GPCs / TPCs (full AD103) | 7 / 40; 2 SMs per TPC | not measured | [N:nvidia2023adawhitepaper p. 8, App. B] |
| SMs | 80 (full AD103) | 80 | [N:nvidia2023adawhitepaper App. B]; [M] |
| FP32 CUDA cores | 10,240 (4080 SUPER) | 80 × 128 = 10,240 | [N:nvidia2026rtx4080family]; [D] |
| SM partitions | 4 per SM; each 1 scheduler + 1 dispatch, 16 FP32 + 16 FP32/INT32 lanes, 64 KB registers, 1 SFU, 1 Tensor core | — | [N:nvidia2023adawhitepaper p. 10] |
| FP32 / rsqrt throughput | 128 / 16 results per clk per SM (8 : 1) | — | [N:nvidia2026bestpractices §12.1.1 Table 5] |
| INT32 throughput | 128 per clk per SM (table) vs ≤ 64 (layout) | — | NVIDIA sources disagree (§2.2.2) |
| Tensor cores (4th gen) / RT cores (3rd gen) | 320 / 80 (full AD103) | — | [N:nvidia2023adawhitepaper App. B] |
| FP64 rate | 1/64 FP32 (64 : 1); 2 results per clk per SM | — | [N:nvidia2023adawhitepaper; N:nvidia2026cudaguide §5.1.3 T30; N:nvidia2026bestpractices T5] |
| Register file | 256 KB / SM (4 × 64 KB) | 65,536 regs / SM, ≤ 255 / thread | [N:nvidia2023adawhitepaper p. 8, p. 10]; [M; N:nvidia2026cudaguide T31] |
| L1 + shared memory | 128 KB / SM | — | [N:nvidia2023adawhitepaper] |
| Shared memory / SM | up to 100 KB (carve-out 0–100 KB) | 102,400 B | [N:nvidia2026adatuning]; [M] |
| Shared memory / block | 99 KB; > 48 KB needs dynamic smem + opt-in | 101,376 B | [N:nvidia2026cudaguide T31, §3.2.6]; [M] |
| Resident warps / threads per SM | 48 / 1,536 | 1,536 threads | [N:nvidia2026adatuning; N:nvidia2026cudaguide T30]; [M] |
| Resident blocks per SM | 24 | 24 | [N:nvidia2026adatuning; N:nvidia2026cudaguide T30]; [M] |
| Max threads per block | 1,024 | — | [N:nvidia2026cudaguide T30] |
| Warp size | 32 | 32 | [N:nvidia2026cudaguide T30]; [M] |
| Register allocation unit | 256 per warp (stated for a cc 7.0 example) | unconfirmed for 8.9 | [N:nvidia2026bestpractices §11.1.1]; TODO(measure) |
| L2 cache | 65,536 KB (full AD103) | 67,108,864 B (64 MB) | [N:nvidia2023adawhitepaper App. B]; [M] |
| Memory type / interface | GDDR6X, 256-bit | 256-bit | [N:nvidia2026rtx4080family]; [M] |
| Memory data rate | 23 Gbps | max 11,501 MHz, current 11,701 MHz reported (anomaly) | [N:burnes2024rtx40super]; [M] |
| Memory bandwidth | — | ≈ 736 GB/s theoretical, not measured | [D]; TODO(measure) |
| Device memory | 16 GB | 16,376 MiB (331 MiB reserved) | [N:nvidia2026rtx4080family]; [M] |
| Host link | PCIe Gen 4 | Gen 4 × 16 | [N:nvidia2026rtx4080family]; [M] |
| Host bandwidth (theoretical) | 16 GT/s per lane, 128b/130b | ≈ 31.5 GB/s per direction | [N:pcisig2011pcie4; N:solomon2014pcie16gts]; [D] |
| Copy engines | 1 engine: one copy overlaps kernels; H2D and D2H don't overlap | `asyncEngineCount` = 1 | [N:nvidia2026bestpractices §10.1.2]; [M] |
| BAR1 / Resizable BAR | supported | 16,384 MiB → enabled | [N:nvidia2026rtx4080family]; [M]; [D] |
| Base / boost clock | 2.29 / 2.55 GHz (Founders Edition) | — | [N:nvidia2026rtx4080family] |
| SM clock | — | 2,610 MHz (CUDA runtime), 3,105 MHz max | [M] |
| Peak FP32 | 52 TFLOPS | 52.2 @ 2.55 GHz; 53.5 @ 2,610 MHz; not an upper bound | [N:nvidia2026rtx4080family]; [D] |
| Peak FP64 | — | ≈ 0.84 TFLOPS @ 2,610 MHz | [D] |
| Power | 320 W TGP | 320 W default, **272 W current** | [N:nvidia2026rtx4080family]; [M] |
| TDR timeout | `TdrDelay` default 2 s | WDDM, display attached | [N:microsoft2025tdrkeys]; [M] |

## 2.8 What this means for N-body (bridge to the following chapters)

All smoke-test numbers in this section come from `results/2026-10-17_smoke/summary.md` and were
**recorded under background load at 272 W; they are not benchmarks.**

- **The SM is the unit of balance.** Per SM, the tiled kernel already reaches 100 % theoretical
  occupancy (§2.3) [D], with possibly no register headroom. Across the chip, small problems leave
  SMs idle: 16 blocks for 80 SMs at N = 4,096. This is the measured motivation for the
  load-balancing techniques of chapter 4.
- **Shared-memory tiling pays off.** The tiled kernel was 2.6× faster than the naive one at
  N = 16,384 (smoke-test number), as expected from tile reuse [P:nyland2007nbody].
- **Arithmetic, not bandwidth, is the expected limit.** At N = 131,072 the smoke test reached about
  1.0 × 10¹² interactions/s, about 37 % of the 53.5 TFLOPS FP32 figure by the 20-FLOP convention
  (smoke-test number). Whether the SFU or the FP32 lanes set the limit depends on the instruction
  mix (§2.2.3) and is a **hypothesis** for Nsight Compute.
- **The L2 changes the picture for large N.** Position arrays of up to about four million bodies
  fit in the 64 MB L2 (§2.4.2) [D], so tiling experiments should measure L2 hit rates rather than
  assume DRAM traffic.
