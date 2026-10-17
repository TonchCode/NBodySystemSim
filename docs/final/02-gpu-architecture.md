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
[N:nvidia2023adawhitepaper, p. 10]; in SASS the reciprocal square root appears as `MUFU.RSQ`
(`docs/03-design.md` §0.1). This matters for N-body, because every pairwise
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

**What our force loop actually issues.** The static SASS of the tiled force loop contains, per
interaction, 3 FADD (r = x_j − x_i), 3 FFMA (d² = r·r + ε²), one FSETP and two predicated FMUL
(the denormal guard that non-FTZ `rsqrtf` adds), **one MUFU.RSQ**, 3 FMUL (m · inv³), 3 FFMA
(a += r·s), one LDS.128 for the shared-memory tile, and loop overhead [M]
(`docs/03-design.md` §0.1). Of roughly 17.6 issue slots per interaction, 14 are FP32 and 1 is
MUFU [M] (`docs/03-design.md` §0.1). In the full-unroll probe kernel the count is
**17.38 SASS slots per interaction** [M] (`results/2026-10-17_round2/probe/sass_counts.txt`).
With 8 or more FP32 instructions per `rsqrtf`, the SFU's 8 : 1 ratio alone does not make the
SFU the bottleneck [D].

Three ceilings follow for one SM, each in interactions per clock [D] (`docs/03-design.md` §0.1):
FP32 lanes 128 / 14 ≈ 9.1; SFU 16; and issue 4 schedulers / 17.38 slots × 32 threads ≈ 7.4.
The issue ceiling assumes one warp-instruction per clock per scheduler, which no source
states (§2.2.1). At that ceiling the SFU is busy about 46 % of the time (8 of 17.38 cycles) [D].

**The probe experiment.** To test which limit binds, the architect built four variants of the
force loop that change the instruction mix in known ways, all with 40 registers and no spills
[M], and timed them twice under the 272 W cap at an SM clock of 2,610–2,640 MHz [M]
(`results/2026-10-17_round2/summary.md` §5):

| Variant | Change | SASS slots / interaction [M] | Pure-issue model [D] | Measured time ratio [M] |
|---|---|---|---|---|
| base | — | 17.38 | 1.000 | 1.000 |
| ftz | flush-to-zero removes the 3-instruction guard | 14.38 | 0.827 | 0.900 / 0.899 |
| mufu2 | a second MUFU (+ FSEL) | 19.38 | 1.115 | 1.189 / 1.184 |
| fmul3 | 3 extra FP32 instructions | 20.38 | 1.173 | 1.185 / 1.185 |

Only fmul3 matches the pure-issue model. mufu2 is about 7 points slower than issue alone
predicts, which points to the SFU (XU) pipe, which would be about 83 % busy in that variant
[D] (`docs/03-design.md` §0.1). ftz gains less than the model predicts, so some other limit
(latency, dependencies or register banks) appears once the guard is gone
(`results/2026-10-17_round2/summary.md` §5). The project's current conclusion is that the force
kernel is **"issue-limited with a significant XU co-limit"**
(`results/2026-10-17_round2/summary.md` §5; `docs/reviews/2026-10-17-critic-1.md` #2). It is
neither purely FMA-bound nor purely MUFU-bound.

**Status of this conclusion.** It is an inference from timing and static SASS, not a profiler
measurement: GPU performance counters are blocked on this machine, so Nsight Compute cannot yet
report pipe utilisation (D-011, `docs/decisions.md`)
(`TODO(measure: Nsight Compute issue-slot and XU/FMA pipe utilisation for allpairs_tiled and the
probe variants, once counters are enabled)`).

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
dense matrix product, so our FP32 force kernels are written for the CUDA cores and SFUs only.
The compiled code agrees. The per-interaction instruction list of `allPairsTiledKernel`
(`docs/03-design.md` §0.1, from `cuobjdump -sass`) contains only FP32, MUFU, shared-memory load
and loop instructions [M]. The archived SASS of the four probe variants of the same force loop
(`results/2026-10-17_round2/probe/mufu_probe.sass`, 18,358 lines) contains no tensor-core
opcode at all: a search for HMMA, IMMA, HGMMA, DMMA, BMMA and QMMA returns zero matches [M].
The production `allpairs_tiled` kernel's SASS is archived at
`results/2026-10-17_round2/sass/allpairs_tiled_production.sass` (675 lines, non-instrumented
instantiation). The same search returns **zero** tensor-core opcodes. The kernel contains 90 FFMA
and 15 MUFU.RSQ (8-way unrolled main loop + remainder) [M].

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

`ptxas` reports that `allPairsTiledKernel` uses **40 registers per thread, with no spills and no
static shared memory** (`results/2026-10-17_architect_sanity/build_resource_usage.txt`). Its tile
is sized at launch, one `float4` per thread (tile = block, `docs/03-design.md` §1), so at
**256 threads per block** it uses 256 × 16 B = **4,096 B** of shared memory per block [D]. The
`allpairs_tuned` instantiations, which hold the same tile as static shared memory, confirm the
size: ptxas reports 4,096 B at block 256, 8,192 B at 512 and 16,384 B at 1,024 [M]
(`results/2026-10-17_architect_sanity/build_resource_usage.txt`; `docs/03-design.md` §3.1).

| Constraint | Calculation | Blocks / SM |
|---|---|---|
| Registers | 40 regs × 32 = 1,280 per warp; per partition ⌊16,384 / 1,280⌋ = 12 warps, × 4 = 48 warps | **6** [D] |
| Warps | 256 threads = 8 warps per block; ⌊48 / 8⌋ | **6** [D] |
| Blocks | hardware limit | 24 [M] |
| Shared memory | 4,096 B per block; ⌊102,400 / 4,096⌋ (20 if 1 KB per block is reserved, as `docs/03-design.md` §3.2 budgets: `TODO(source needed: per-block shared-memory reservation on cc 8.9)`) | 25 (or 20) [D] |

The minimum is **6 blocks per SM = 1,536 threads = 48 warps, i.e. 100 % theoretical occupancy**
[D]. The CUDA runtime's occupancy API returns the same 6 blocks per SM for this configuration
[M] (`results/2026-10-17_architect_sanity/summary.md`, occupancy table). Across 80 SMs that is
122,880 resident threads [D].

**How registers are actually allocated: measured evidence.** The register row above uses a
per-partition model, not one pool of 65,536 registers per SM. Two facts support it:

- The whitepaper describes one 64 KB register file per partition, i.e. 16,384 registers each
  [N:nvidia2023adawhitepaper, p. 10], and the Best Practices Guide states that
  "register allocations are rounded up to the nearest 256 registers per warp"
  [N:nvidia2026bestpractices, §11.1.1]. The latter is written next to a cc 7.0 example, so on
  its own it does not prove the rule for cc 8.9.
- The runtime confirms both rules on our GPU. The `allpairs_multi` kernel at block 64, q = 32
  uses 44 registers [M]. 44 × 32 = 1,408 registers per warp, rounded up to 1,536 [D]. A single
  pool of 65,536 registers would hold ⌊65,536 / 1,536⌋ = 42 warps = **21 blocks** of 2 warps.
  Per-partition allocation holds ⌊16,384 / 1,536⌋ = 10 warps per partition = 40 warps =
  **20 blocks** [D]. The occupancy API returns **20** [M]
  (`results/2026-10-17_architect_sanity/summary.md`; `docs/03-design.md` §3.2). Without the
  256-register rounding, both models would give more than 20 (22 per partition, 23 pooled) [D].
  So only "256 per warp, per partition" matches the measurement.

Two remarks follow.

1. **The tiled kernel has no register headroom.** At 40 registers a warp needs exactly
   5 × 256 = 1,280 registers, and 12 warps fill a partition to 15,360 of 16,384 [D]
   (`docs/03-design.md` §3.2). At 41 registers a warp rounds up to 1,536, a partition holds only
   10 warps, and the SM drops to 40 warps = 5 blocks = **83 % occupancy** [D]. The naive kernel
   (39 registers [M], `build_resource_usage.txt`) sits at the same 6-block limit. The 41-register
   case is a prediction from the model the runtime confirmed, not a separate measurement.
   Whether 83 % occupancy would actually be slower is a separate question. The round-1 design
   assumes 8 warps per SM already saturate this kernel (`docs/03-design.md` §0.2, assumption A1).
2. **Theoretical is not achieved occupancy.** The table gives the upper bound. Achieved
   occupancy needs Nsight Compute, which is blocked on this machine (D-011).

The same arithmetic also shows the opposite problem, the one the load-balancing chapter
addresses. NVIDIA advises launching more blocks than there are SMs
[N:nvidia2026bestpractices, §11.3]. With N = 4,096 bodies and 256 threads per block, the grid has
only 16 blocks for 80 SMs. The round-2 per-block timeline measured exactly that: **64 of 80 SMs
idle for the whole kernel**, with SM efficiency 0.199 [M]
(`results/2026-10-17_round2/summary.md` §1). Full occupancy per SM does not help if most SMs are
idle.

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
DRAM after the first pass [D]. At N = 65,536, the largest size in the round-2 sweep, the position
array is 1 MB and fits in L2 sixty-four times over [D] (`docs/03-design.md` §2). Two cautions apply: not all of the L2 is
necessarily available to one array, and whether the hardware actually keeps the array resident
between tiles has to be measured (`TODO(measure: Nsight Compute L2 hit rate for allpairs_tiled
at several N, once GPU counters are enabled; D-011)`). The size of the L2 also means that DRAM-bandwidth arguments taken from older
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
on our board [M] it is ≈ **53.5 TFLOPS** [D]. Neither figure is an upper bound on the clock: under
load the SM ran at up to 2.79 GHz (§2.6), and at the 3,105 MHz ceiling [M] the figure would be
≈ 63.6 TFLOPS [D] (`docs/01-hardware.md` §2.2).

For this kernel the FMA peak is not the relevant ceiling anyway. With 17.38 issue slots per
interaction and the assumed one instruction per scheduler per clock (§2.2.3), the issue-model
ceiling is about 7.4 interactions per clock per SM. At the 2.76 GHz median clock measured under
load that is ≈ **1.6 × 10¹² interactions/s** for 80 SMs [D]
(`docs/reviews/2026-10-17-critic-1.md` #11). The best measured all-pairs throughput in round 2 is
1.27 × 10¹² interactions/s (`allpairs_multi` b128 q2 at N = 65,536) [M]
(`results/2026-10-17_round2/summary.md` §3). That is about 78 % of the issue-model ceiling at
2.76 GHz [D], and more if that kernel ran at a lower clock, as §2.6 suggests.

FP64 runs at 1/64 of the FP32 rate [N:nvidia2023adawhitepaper; N:nvidia2026cudaguide, §5.1.3
Table 30], about **0.84 TFLOPS** at 2,610 MHz [D]. A double-precision force kernel would therefore
be about 64 times slower than the same kernel in single precision [D].

The project uses **FP32 for all force kernels** (`CLAUDE.md` rule 3), storing bodies as `float4`
and using `rsqrtf` (`docs/01-hardware.md` §5). The round-2 long-run energy test supports this
choice and shows where FP32 actually costs accuracy. For N = 4,096 over 100 time units, the
relative energy error was 6.2 × 10⁻⁷ at dt = 10⁻³ and 1.09 × 10⁻⁶ at dt = 5 × 10⁻⁴ [M]
(`results/2026-10-17_round2/summary.md` §6). A smaller time step drifted *more*, so the drift is
dominated by per-step FP32 round-off in the integrator state (positions and velocities), not by
the integrator's truncation error (D-014, `docs/decisions.md`). The lever for better
conservation is therefore the precision of the integrator state, not FP64 forces (D-014).

The "20 FLOP per interaction" accounting [P:nyland2007nbody] counts `rsqrtf` as several FLOPs,
even though it executes on the SFU at one-eighth of the FMA rate (§2.2.3). Percentages of the
FLOPS peak computed this way are a convention, not a measure of FMA utilisation. The issue-model
ceiling above is the more honest denominator.

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
In the round-2 session, power draw under load reached 273.6 W, so **our kernels hit the cap** [M]
(`results/2026-10-17_round2/summary.md`, environment;
`results/2026-10-17_round2/gpu_clock_log_phase2.csv`). The limit in force is recorded with every
benchmark.

**Clocks under the 272 W cap (measured).** Across the round-2 session the logged SM clock under
load had a median of 2,760 MHz (range 2,565–2,790 MHz) [M]
(`results/2026-10-17_round2/summary.md`, environment). The per-block timelines also give each
kernel's effective SM clock. These ranged from **2.52 GHz to 2.78 GHz**, and the kernels that
process more interactions per second ran at *lower* clocks [M]
(`results/2026-10-17_round2/timeline/timeline_analysis.txt`; scorecard P-TL2 in
`results/2026-10-17_round2/summary.md` §7). At N = 65,536, for example:

| Kernel at N = 65,536 | Throughput rank | Effective SM clock [M] |
|---|---|---|
| `allpairs_multi` b256 q4 | fastest | 2.522 GHz |
| `allpairs_tuned` b128 full | middle | 2.602 GHz |
| `allpairs_tiled` b256 | slowest | 2.700 GHz |

The small-N launches, which leave most SMs idle, ran at 2.73–2.79 GHz [M]
(`timeline_analysis.txt`). Under a power cap, a faster kernel draws more power and is clocked
down. Speedups between kernels are therefore measured at *different* clocks: part of a faster
kernel's gain is eaten by its lower clock. This is a property of the platform as configured, and
it is why the SM clock is logged with every run. These clocks come from the instrumented timeline
twins, whose speed differs from production by −3.4…+6.2 % (D-011), so they show the trend, not the
exact production clock. During the probe run (§2.2.3) the clock stayed within 2,610–2,640 MHz at
267–271 W [M] (`results/2026-10-17_round2/summary.md` §5).

Other clock values, for reference:

| Clock | Value | Source |
|---|---|---|
| NVIDIA reference base / boost (Founders Edition) | 2.29 GHz / 2.55 GHz | [N:nvidia2026rtx4080family, "GPU Engine Specs"] |
| Reported by the CUDA runtime (`clockRate`) | 2,610 MHz | [M] |
| Maximum SM clock reported by `nvidia-smi` | 3,105 MHz | [M] |
| Idle SM clock (round-2 environment capture) | 210 MHz | [M] (`results/2026-10-17_round2/env_before.txt`) |

NVIDIA's page notes that partner boards may differ from the Founders Edition figures
[N:nvidia2026rtx4080family]; ours is an ASUS board [M]. NVIDIA gives the base clock as 2.29 GHz;
the 2,295 MHz value used in earlier project notes has no source at that precision. The CUDA
runtime's 2,610 MHz is not the operating clock: measured clocks lie both below and above it.

**Timer resolution.** The GPU's `%globaltimer` register advances in steps of **1.024 µs** on this
board [M] (`results/2026-10-17_round2/timeline/timeline_analysis.txt`, "timer res 1024 ns";
D-011). That is comparable to an entire N = 1,000 kernel, so per-block start and end times are
taken from the SM cycle counter `clock64`, anchored to timer tick edges before and after the
launches. Calibrated and raw metrics agree within about 1 % (D-011;
`results/2026-10-17_round2/summary.md` §1). Any intra-kernel timing on this GPU has to account
for this resolution.

**Background load.** At the first hardware capture the GPU was at 41 % utilisation, with
5,198 MiB of memory in use and three active NVENC encoder sessions from desktop applications [M]
(`docs/01-hardware.md` §4). For round 2 those were closed: 0 % idle utilisation before and after
the runs, with 1,027–1,029 MiB VRAM in use [M] (`results/2026-10-17_round2/env_before.txt`,
`env_after.txt`).

## 2.7 Summary: measured versus specified values

| Property | NVIDIA specification | Our RTX 4080 SUPER (measured / derived) | Source |
|---|---|---|---|
| Compute capability | — | 8.9 (`sm_89`) | [M] |
| GPCs / TPCs (full AD103) | 7 / 40; 2 SMs per TPC | not measured | [N:nvidia2023adawhitepaper p. 8, App. B] |
| SMs | 80 (full AD103) | 80 | [N:nvidia2023adawhitepaper App. B]; [M] |
| FP32 CUDA cores | 10,240 (4080 SUPER) | 80 × 128 = 10,240 | [N:nvidia2026rtx4080family]; [D] |
| SM partitions | 4 per SM; each 1 scheduler + 1 dispatch, 16 FP32 + 16 FP32/INT32 lanes, 64 KB registers, 1 SFU, 1 Tensor core | — | [N:nvidia2023adawhitepaper p. 10] |
| FP32 / rsqrt throughput | 128 / 16 results per clk per SM (8 : 1) | — | [N:nvidia2026bestpractices §12.1.1 Table 5] |
| Force-loop instruction mix | — | 17.38 SASS slots per interaction (probe); ≈ 17.6 (tiled, u8), of which 14 FP32 and 1 MUFU | [M] `round2/probe/sass_counts.txt`; `03-design.md` §0.1 |
| Force-kernel limiter | — | "issue-limited with a significant XU co-limit" (timing inference, no ncu) | `results/2026-10-17_round2/summary.md` §5 |
| INT32 throughput | 128 per clk per SM (table) vs ≤ 64 (layout) | — | NVIDIA sources disagree (§2.2.2) |
| Tensor cores (4th gen) / RT cores (3rd gen) | 320 / 80 (full AD103) | no tensor opcodes in the probe SASS | [N:nvidia2023adawhitepaper App. B]; [M] |
| FP64 rate | 1/64 FP32 (64 : 1); 2 results per clk per SM | — | [N:nvidia2023adawhitepaper; N:nvidia2026cudaguide §5.1.3 T30; N:nvidia2026bestpractices T5] |
| Register file | 256 KB / SM (4 × 64 KB) | 65,536 regs / SM, ≤ 255 / thread | [N:nvidia2023adawhitepaper p. 8, p. 10]; [M; N:nvidia2026cudaguide T31] |
| Register allocation | 256 per warp (cc 7.0 example) | per partition (4 × 16,384), 256 per warp: the runtime gives 20 blocks (not 21) at 44 regs × 64 threads | [N:nvidia2026bestpractices §11.1.1]; [M] `architect_sanity/summary.md`; [D] |
| L1 + shared memory | 128 KB / SM | — | [N:nvidia2023adawhitepaper] |
| Shared memory / SM | up to 100 KB (carve-out 0–100 KB) | 102,400 B | [N:nvidia2026adatuning]; [M] |
| Shared memory / block | 99 KB; > 48 KB needs dynamic smem + opt-in | 101,376 B | [N:nvidia2026cudaguide T31, §3.2.6]; [M] |
| `allpairs_tiled` resources (b256) | — | 40 regs, 0 spills, 4,096 B smem → 6 blocks/SM, 100 %, register headroom 0 | [M] ptxas, runtime; [D] |
| Resident warps / threads per SM | 48 / 1,536 | 1,536 threads | [N:nvidia2026adatuning; N:nvidia2026cudaguide T30]; [M] |
| Resident blocks per SM | 24 | 24 | [N:nvidia2026adatuning; N:nvidia2026cudaguide T30]; [M] |
| Max threads per block | 1,024 | — | [N:nvidia2026cudaguide T30] |
| Warp size | 32 | 32 | [N:nvidia2026cudaguide T30]; [M] |
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
| SM clock | — | 2,610 MHz (CUDA runtime), 3,105 MHz max; under the 272 W cap 2.52–2.78 GHz per kernel, session median 2,760 MHz | [M] |
| Peak FP32 | 52 TFLOPS | 52.2 @ 2.55 GHz; 53.5 @ 2,610 MHz; not an upper bound | [N:nvidia2026rtx4080family]; [D] |
| Issue-model ceiling (force loop) | — | ≈ 1.6 × 10¹² interactions/s @ 2.76 GHz (assumes 1 inst/clk/scheduler) | [D] |
| Best measured all-pairs throughput | — | 1.27 × 10¹² interactions/s (multi b128 q2, N = 65,536) | [M] `round2/summary.md` §3 |
| Peak FP64 | — | ≈ 0.84 TFLOPS @ 2,610 MHz | [D] |
| Power | 320 W TGP | 320 W default, **272 W current**, reached under load (273.6 W) | [N:nvidia2026rtx4080family]; [M] |
| `%globaltimer` resolution | — | 1.024 µs | [M] |
| TDR timeout | `TdrDelay` default 2 s | WDDM, display attached | [N:microsoft2025tdrkeys]; [M] |

## 2.8 What this means for N-body (bridge to the following chapters)

Numbers in this section come from the round-2 session, with the environment controlled (idle 0 %
before and after) and the 272 W cap reached (`results/2026-10-17_round2/summary.md`), unless
labelled otherwise.

- **Per SM, occupancy is solved; across SMs, it is not.** The tiled kernel reaches 100 %
  theoretical occupancy with zero register headroom (§2.3). The measured per-block timelines show
  two imbalance regimes, both caused by how blocks land on the 80 SMs
  (`results/2026-10-17_round2/summary.md` §1):
  - **Starvation at small N.** With 80 or fewer blocks, each busy SM gets exactly one block and
    the rest stay idle for the whole kernel. Tiled b256 at N = 4,096 leaves 64 SMs idle (SM
    efficiency 0.199); `allpairs_multi` b512 q8 cuts that to 16 idle SMs (0.786).
  - **The wave tail at large N.** When all blocks are resident at once, the last wave is partial.
    Tiled b256 at N = 65,536 spends 23.5 % of its span in the tail; `allpairs_multi` b256 q4,
    with more blocks than resident slots, cuts it to 0.1 %.

  Chapter 4 treats both regimes and the techniques that remove them.
- **What load balancing buys.** Against the best-tuned one-thread-per-body baseline, splitting
  each body over several threads is worth 2.40× at N = 1,000, 2.56× at N = 4,096, 1.40× at
  N = 10,000 and 1.02× at N = 65,536 (`results/2026-10-17_round2/summary.md` §3). The gain is
  large exactly where starvation is measured and vanishes once every SM is busy.
- **Shared-memory tiling pays off.** The tiled kernel was 2.6× faster than the naive one at
  N = 16,384 (`results/2026-10-17_smoke/summary.md`, **smoke-test number**, not repeated in
  round 2), as expected from tile reuse [P:nyland2007nbody].
- **Instruction issue, not bandwidth, is the limit.** The best round-2 throughput,
  1.27 × 10¹² interactions/s, is about 78 % of the issue-model ceiling (§2.5). The force loop is
  issue-limited with a significant SFU co-limit (§2.2.3), an inference that still waits for
  Nsight Compute.
- **The power cap couples clock to speed.** Faster kernels ran at lower clocks (2.52 vs 2.70 GHz
  at N = 65,536, §2.6). The results chapter must report speedups together with the clocks they ran
  at.
- **The L2 changes the picture for large N.** Position arrays of up to about four million bodies
  fit in the 64 MB L2 (§2.4.2) [D], so tiling experiments should measure L2 hit rates rather than
  assume DRAM traffic.
