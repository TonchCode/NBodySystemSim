# 5. Parallelism and Memory: ILP, TLP, Where the Data Lives, Memory Optimisation and Profiling

> **Language note:** written in English; the Bulgarian/English choice for the final document is still open.
> §5.8 has an EN–BG glossary of the new terms.

**Source tags** (as in chapter 3): **[P:key]** = paper in `references/references.bib` ·
**[N:key]** = NVIDIA documentation · **[M]** = measured on this machine (folder named) ·
**[D]** = derived arithmetically from tagged values · **[std]** = textbook fact.

This chapter answers the lecturer's second set of questions:
- instruction-level parallelism (ILP) and thread-level parallelism (TLP);
- which memory of the GPU is used for what, and which suits this project best;
- where in memory the algorithm should live;
- memory optimisation;
- profiling, including memory (RAM) profiling.

Chapter 2 describes the hardware itself. This chapter is about **how our kernels use it**, and
it ties each claim to a measurement or a line of code.

New measurements made for this chapter (2026-10-22) are in `results/2026-10-22_memory_profile/`:
Nsight Systems traces, `analysis.txt` (produced by `tools/memprofile_analysis.py`) and
`host_ram.txt`. They were taken with the desktop in use, so the GPU had about 10–23 % background
load. **Their memory sizes are exact. Their kernel times are not canonical** (the canonical
times are the clean re-run of chapter 4, §4.1.1).

---

## 5.1 Three kinds of parallelism, from the bottom up

A GPU runs a program in parallel at three levels at once. It helps to separate them before
talking about any one of them:

| Level | What runs in parallel | Who creates it | In our code |
|---|---|---|---|
| **Data parallelism / SIMT** | the 32 threads of a warp execute the same instruction on different data | the hardware: one instruction, 32 lanes | 32 bodies per warp compute their forces together |
| **TLP** (thread-level) | many warps are resident on an SM; when one waits, the scheduler issues from another | the programmer, by launching enough threads; limited by occupancy | 1 thread per body (or q threads per body in #3); up to 48 warps per SM |
| **ILP** (instruction-level) | inside **one** thread, independent instructions issue back to back without waiting for each other | the compiler, from code that has independent work (unroll, prefetch) | the instructions of consecutive interactions are independent, except the final `a += …` |

TLP and ILP solve the **same problem**: **latency**. An FP32 instruction's result is not ready
on the next clock; a load from L2 or device memory takes hundreds of clocks. While a warp waits,
the SM needs something else to issue. It can take it from **another warp** (TLP) or from **the
same warp's next independent instruction** (ILP). Volkov's GTC 2010 talk is the classic statement
that TLP is not the only way: codes with enough ILP can be faster at *lower* occupancy, and
"multithreading is the only way to hide latency on GPU" is called a fallacy
[P:volkov2010lower, slide 4]. A 2026 study of database kernels on an L4 (an Ada GPU, the same
architecture as ours) and a GH200 reaches the matching modern conclusion: once memory bandwidth
stops being the limit, kernels hit an "instruction wall", and the fixes are "higher occupancy
and/or instruction level parallelism" and fewer instructions per access
[P:hepkema2026instructionwall].

### 5.1.1 ILP inside one interaction: the dependency chain

The interaction in `src/core/interaction.cuh` is the inner loop of every force kernel. Written
as a dependency graph (who has to wait for whom) [D, from the code]:

```
 rx = xj-xi   ry = yj-yi   rz = zj-zi        3 FADD, independent of each other   (ILP = 3)
        \          |          /
   d2 = rz*rz+eps2 -> ry*ry+d2 -> rx*rx+d2   3 FFMA, a chain                     (ILP = 1)
                   |
        denormal guard (FSETP + 2 FMUL)
                   |
           inv = rsqrt(d2)                   1 MUFU.RSQ, runs on the SFU, not the FP32 lanes
                   |
     s = mj*inv -> *inv -> *inv              3 FMUL, a chain                     (ILP = 1)
        /          |          \
 ax += rx*s   ay += ry*s   az += rz*s        3 FFMA, independent of each other   (ILP = 3)
```

- **Inside one interaction there is little ILP.** The critical path is about 11 dependent
  instructions long: FADD → 3 FFMA → guard → MUFU → 3 FMUL → FFMA [D].
- **Across interactions there is a lot of ILP.** Interaction k+1 does not need any result of
  interaction k, except the three accumulators `ax, ay, az`. Everything above the last line of
  interaction k+1 can run while interaction k is still in flight [D].
- **The compiler can only use that if it sees several iterations at once.** That is what loop
  unrolling does. `#pragma unroll 8` in the tiled kernel gives the scheduler 8 independent
  chains to interleave. In the compiled tiled kernel there are 15 `MUFU.RSQ` and 15 `LDS.128`
  for 2 `BAR.SYNC` [M] (`cuobjdump -sass`, 2026-10-22): the unrolled body plus its remainder.
- **Two pipes in parallel is also ILP.** `MUFU.RSQ` runs on the special-function unit while
  FP32 instructions of other interactions run on the FP32 lanes. Chapter 2, §2.2.3 shows that the
  force loop is "issue-limited with a significant XU co-limit": the SFU is busy about 46 % of the
  time, *in parallel* with the FP32 work [D; M probe experiment].

### 5.1.2 Measured evidence for ILP

| Experiment | What changes | Result | What it shows |
|---|---|---|---|
| Unroll 1 vs 8 vs full (all-pairs, round-1 sweep) | how many interactions the compiler sees at once | unroll 1: **−21 %**; unroll 16: +5 %; full: +8 % (blocks ≤ 256) [M] (`results/2026-10-17_architect_sanity/`) | without unroll, each interaction waits for its own chain plus loop overhead; with unroll, chains overlap |
| Full unroll at block 1024 | the loop body becomes very large | **−18 %** [M] | too much code: instruction-cache pressure (the design predicted this in advance). ILP has a cost too |
| Register prefetch in `allpairs_multi` | the next tile's global load is issued before the current tile is consumed (memory-level ILP) | **+16.1 %** at 2 warps/SM, +10.1 % at 4, +7.7 % at 8, +0.5 % at block 512, **0.0 %** at N = 65,536 [M] (`results/2026-10-17_round2/summary.md` §2) | prefetch helps *only* when there are too few warps to hide the load latency: ILP replaces missing TLP |
| Probe variants (`ftz`, `mufu2`, `fmul3`) | add or remove instructions in the chain | time follows the issue-slot count, with an extra penalty for a second MUFU [M] (chapter 2, §2.2.3) | the loop is limited by instruction issue, not by memory |

The prefetch row is the clearest result. It is Volkov's argument measured on our GPU:
**ILP matters exactly where TLP is missing** [D, interpreting M].

### 5.1.3 TLP: warps, occupancy and latency hiding

Each Ada SM has 4 schedulers and can hold at most 48 warps (1,536 threads), which is 12 warps per
scheduler (chapter 2, §2.3). **Occupancy** is resident warps ÷ 48.

What limits occupancy in our kernels [M] (`cuobjdump --dump-resource-usage`, 2026-10-22):

| Kernel | Registers / thread | Shared memory / block | Local stack | Occupancy limit |
|---|---|---|---|---|
| `allPairsTiledKernel` (#1) | 40 | 4 KB at block 256 (dynamic) | 0 | 6 blocks of 256 = 48 warps = **100 %**, with **zero register headroom**: at 41 registers it drops to 83 % (chapter 2, §2.3) |
| `allPairsMultiKernel` (#3) | 53–57 (block 1024 templates) | 16 KB static | 0 | registers decide it; smaller blocks run at 100 % (chapter 4, §4.4) |
| `bhTraverseKernel` (#4/#5) | 40 | 0 | **384 B** | 100 % at block 128 |
| `bhWarpKernel` (#7) | 40 | 0 | **0 B** | 100 % at block 128 |
| `multipoleKernel` (tree summary) | 38 | 0 | 0 | 100 % |
| `radixTreeKernel` (tree build) | 19 | 0 | 0 | 100 % |

**TLP in our measurements:**
- **TLP fixes starvation in all-pairs.** At small N there are fewer blocks than SMs, so most of
  the GPU has *no* threads at all. Technique #3 creates more threads by giving each body q
  threads: idle SMs drop from 64 to 16 at N = 4,096, and the kernel runs **2.56× faster** [M]
  (chapter 4, §4.4). This is TLP in its plainest form: more independent threads, more of the
  machine in use.
- **More TLP does not help the tree.** At fixed N = 262,144, #5's rate is **flat from 2 to
  11 blocks per SM** (19.9–20.7 G interactions/s) [M] (chapter 4, §4.5.1). Occupancy is not the
  limit. The limit is L1 capacity for the scattered loads (§5.3).
- **TLP cannot fix divergence.** The 32 lanes of a warp share one instruction stream. In #5,
  only 3–7 lanes are active at a leaf-bucket interaction [M] (chapter 4, §4.5.2). More warps do
  not change that: every warp has the same problem. #7 fixes it by changing *what* a warp does,
  not *how many* warps there are.

**A rough latency-hiding count** (an illustration, [T2]: the dependent-issue latency of an FP32
instruction on sm_89 is not in the sources this project has verified,
`TODO(source needed: sm_89 dependent FP32 issue latency)`; about 4 clocks is the commonly quoted
value). If a dependent instruction must wait about 4 clocks and a warp has ILP 1, a scheduler
needs about 4 warps to issue every clock. Our kernels give each scheduler up to 12 warps *and*
ILP across unrolled interactions. That is far more than enough, which is consistent with prefetch
adding 0.0 % at full occupancy and +16 % at 2 warps per SM [D, interpreting M].

---

## 5.2 The memories of the RTX 4080 SUPER and what each one is used for

The sizes come from chapter 2, §2.4. The "used for" column comes from our code and compiled
binary [M].

| Memory | Where / scope | Size (this GPU) | Speed class | **What our project uses it for** | Suited to N-body? |
|---|---|---|---|---|---|
| **Registers** | on the SM, private to each thread | 256 KB per SM, 64 K × 32 bit; ≤ 255 per thread | fastest, no load needed | the body's own position `bi`, the accumulators `a`, the temporaries of the interaction; in **#7, the traversal stack** (2 registers per lane hold 64 stack entries for the warp) | **best**: everything that is per-body and reused every interaction |
| **Local memory** | private to a thread, but **physically in device memory**, cached in L1/L2 | as needed (384 B per thread in #5) | L1 hit ≈ 32 cycles, otherwise like global memory | the per-thread traversal stack `int stack[96]` of #4/#5/#6 (`STACK:384`, 1 `LDL` + 3 `STL` in SASS [M]) | avoid: #7 removed it |
| **Shared memory** | on the SM, shared by one block | up to 100 KB per SM (carve-out from the 128 KB L1) | **≈ 30 cycles** (30.1 on RTX 4090) | the **tile of body positions** in all-pairs (#1/#2: blockDim × 16 B, 4 KB at block 256; #3: 16 KB); the per-warp interaction lists of `bh_warp_list`; the occupancy-sweep padding | **best for all-pairs**: every position loaded once per block is reused by all its threads |
| **L1 data cache** | on the SM, hardware-managed | 128 KB minus the shared carve-out | **≈ 32 cycles** on a hit | caches every global and local load: tree nodes, `sortedPos`, #5's stack | **critical for the tree**: #5 is bound by L1 *capacity* (§5.3) |
| **Constant memory** | read-only, cached, one copy for the whole GPU | small (kernel parameters + `__constant__` data) | very fast if **all lanes read the same address**; serialises otherwise | only the **kernel parameters** (`CONSTANT[0]`: 380–456 B per kernel [M]); no `__constant__` data | not for bodies or trees (far too large). #7's warp-uniform loads have the pattern it likes, but L1 broadcast already gives the same effect |
| **Read-only / texture path** | the non-coherent read-only data path into L1 | — | like L1 | used **implicitly**: `const __restrict__` pointers let the compiler emit `LDG.E.128.CONSTANT` (#1, #5) [M]. #7 passes its pointers in a struct (`TreeView`) and gets plain `LDG.E.128` [M] | small effect at most; a cheap experiment (§5.5.3) |
| **L2 cache** | on the chip, shared by all 80 SMs | **64 MB** | **≈ 270–285 cycles** | every global access passes through it; the hot top of the tree and, for N ≤ ~4 M, all positions can stay here | **the reason the tree code does not hit a DRAM wall** (§5.3) |
| **Global / device memory (GDDR6X)** | off-chip VRAM, visible to all threads | 16 GB, 736 GB/s theoretical | **≈ 570 cycles** | all arrays: positions, velocities, accelerations, keys, permutation, tree nodes (§5.6) | the home of the data; the goal is to touch it as rarely as possible |
| **Host memory (system RAM)** | CPU side, across PCIe 4.0 × 16 | 31.5 GB/s per direction theoretical | ~20× below VRAM bandwidth | initial conditions, validation reference, results; **pinned** buffers (`cudaHostAlloc`, `src/core/host.h`) only for the CPU + GPU split (#8) | keep it out of the time step: our state is GPU-resident |

**Latency numbers.** These are pointer-chase latencies measured on an **RTX 4090** by Luo et al.
[P:luo2025hopper, Table 3]: L1 32.0, shared memory 30.1, L2 273.0 cycles (about 285 in their text),
global memory 571 cycles. The RTX 4090 is AD102, a larger chip of the **same Ada generation and
compute capability 8.9** as our AD103. The SM-local numbers (L1, shared) should carry over. L2 and
DRAM latency depend on chip size and memory clock, so ours may differ
(`TODO(measure: pointer-chase latencies on the 4080 SUPER)`). The ladder that matters is: a
register costs nothing, L1 or shared about 30 cycles, L2 about 9× that, and DRAM about 19× that.

**One-sentence answer for the presentation:** registers hold what a thread reuses every step;
shared memory holds what a block reuses (the all-pairs tile); L1 and L2 hold what the hardware
can discover is reused (tree nodes); device memory is where everything lives between steps; host
memory is touched only at the start and the end.

---

## 5.3 Where the algorithm should live in memory

The right place depends on how often each byte is **reused**, not only on how big it is.

### All-pairs: compute-bound, the tile in shared memory

- Each body position (16 B) is used N times per step. A block of 256 threads loads one tile of
  256 positions from global memory **once** into shared memory, and then every thread reads all
  256 entries. Global and L2 traffic falls by a factor of the block size [D] (chapter 3, §3.2).
- Per interaction, a thread reads one 16 B `LDS.128` from shared memory and does about 20 FLOP.
  From global memory, it reads only 16 B per *tile* per thread [D].
- The whole position array is small: 1 MiB at N = 65,536 and 16 MiB at 1M [M]. That is well
  inside the 64 MB L2 [D].
- **Conclusion:** all-pairs lives in **registers + shared memory**. DRAM is not its limit. It is
  limited by instruction issue (§5.1.2, chapter 2, §2.2.3). Measured confirmation: at N = 1M the
  tiled kernel took **861 ms** per force evaluation in the profiling run [M]
  (`results/2026-10-22_memory_profile/analysis.txt`). That is the 870 ms the earlier work
  extrapolated from smaller N (chapter 3, §3.6), within the background-load noise. A
  memory-bound kernel would have slowed down once its data no longer fit a cache level. This one
  just scaled as N².

### Barnes-Hut: latency- and locality-bound, the tree in L1/L2

The traversal kernel reads these arrays [D, from `TreeView` in `src/core/kernels_tree.cu`]:

| Array | Bytes per body | Accessed |
|---|---|---|
| `nodeCom` (centre of mass + mass, `float4`) | 16 | every node visit |
| `nodeRc2` (opening radius², `float`) | 4 | every node visit |
| `nodeInfo` (children + body range, `int4`) | 16 | every opened node |
| `sortedPos` (bodies in Morton order, `float4`) | 16 | leaf interactions |
| `order`, `pos`, `acc` | 4 + 16 + 16 | once per body |

That is about **52 B per body of tree data** that is re-read during the walk, plus 36 B per body
read or written once [D]. At N = 1M the re-read set is about 52 MiB. That is just under the
64 MB L2, but L2 is shared with everything else. At 4M it is about 208 MiB, which is clearly
larger [D].

**What the profiling run shows** (`bhWarpKernel`, nsys, background load, so not canonical) [M]:

| N | traversal time | time per body |
|---|---|---|
| 262,144 | 2.07 ms | 7.9 ns |
| 1,048,576 | 7.68 ms | 7.3 ns |
| 4,194,304 | 27.8 ms | 6.6 ns |

**There is no cliff when the tree outgrows L2** [M]. The reason is reuse [D]:
- The **top levels of the tree** are visited by every warp. They are a few MB and stay hot in L2.
- The **leaves** are visited only by the warps of bodies nearby. With Morton order those warps
  run close together in time, so the leaves they need are in cache while they need them.

The space-filling-curve sort is therefore **a memory optimisation as much as a load-balancing
one**.

**The L1 measurement** is the strongest evidence of where the tree has to live. When the
occupancy sweep padded shared memory, it also shrank the L1 carve-out. #5 then ran at
19.9–20.7 G/s at every occupancy. Without padding, with the full L1, it ran at **29.0 G/s,
×1.51** [M] (chapter 4, §4.5.1; D-026). The persistent-thread arms (#6) lost speed, and the exploratory diagnosis
is that they broke the locality between Morton-adjacent warps on one SM (chapter 4, §4.6.4). Both results say
the same thing: **for the tree, keep neighbouring work on the same SM at the same time, so the
SM's own L1 serves it.**

**Conclusion:**
- the bodies' own data lives in **registers**;
- the tree lives in **device memory**, but it must be *read* from **L1/L2**;
- the code's job is to make that possible: sort the bodies, keep the order and leave L1 large.

---

## 5.4 Memory optimisations in this project, and their measured effect

| # | Optimisation | Where | Why it helps | Evidence |
|---|---|---|---|---|
| 1 | **`float4` packing** (x, y, z, mass in one 16 B value) | everywhere | one 128-bit load (`LDG.E.128`, `LDS.128`) instead of four 32-bit loads; mass travels with the position | in SASS [M] |
| 2 | **Shared-memory tiling** | all-pairs #1–#3 | reuse factor = block size (§5.3) | GPU Gems tiling [P:nyland2007nbody]; `allpairs_naive` (global memory only) is kept as the correctness reference |
| 3 | **Block size choice** (also a memory choice: more blocks → more barriers and tile loads) | #2 | b64 removes the wave tail but costs +12 % per interaction in barriers and tile loads | [M] chapter 4, §4.3 |
| 4 | **Register prefetch of the next tile** | #3 | issues the global load early: memory-level ILP | +16.1 % at 2 warps/SM, 0 % at full occupancy [M] |
| 5 | **Morton sort + gather into `sortedPos`** | tree | a leaf's bodies are contiguous, so leaf loads are coalesced; neighbouring warps share nodes in cache | #5/#4 = 2.0× [M] (part divergence, part locality) |
| 6 | **Stack in registers instead of local memory** | #7 | removes the 384 B per-thread local stack: no local-memory traffic, 0 B frame | part of #7's 5.8× [M] (not separated) |
| 7 | **Warp-uniform (broadcast) loads** | #7 | all 32 lanes read the same node: one L1 wavefront instead of up to 32 | SIMT efficiency 1.000 [M] (chapter 4, §4.7) |
| 8 | **Keep L1 large** (no shared memory in the traversal) | #5, #7 | the carve-out goes to L1 | ×1.51 for #5 [M] |
| 9 | **Allocate once, reuse every step** | tree builder (`TreeGpu`, 28 allocations) | no `cudaMalloc` in the time step | nsys: all allocations at start-up, none in the timed loop [M] |
| 10 | **Fused kick-drift integrator** | `kickDriftKernel` | position and velocity are read and written in one pass instead of two | chapter 3, §3.3 |
| 11 | **GPU-resident state** | whole simulation | positions never cross PCIe during a run; 2 host-to-device copies in total [M] | nsys trace (§5.6) |
| 12 | **FP32, not FP64** | all kernels | half the bytes per value; FP64 is 1/64 rate on sm_89 | chapter 2, §2.5 |

**Not done, and why** (for questions):
- *Structure of arrays (SoA)* would split x, y, z, m into four arrays. We keep `float4` because
  every interaction needs all four values together. One aligned 16 B load is the ideal access.
- *L2 persistence windows* (`cudaAccessPolicyWindow`) could pin the top of the tree in L2. They
  have not been tested. The "no cliff" result (§5.3) suggests a small gain at most
  (`TODO(measure)`).
- *FP16 / mixed precision* would halve the bytes again, but gravity needs the FP32 range for r²
  and r⁻³. Out of scope.
- *Unified (managed) memory* is not needed when the state never leaves the GPU.

---

## 5.5 Profiling

### 5.5.1 What tools exist and which ones work here

| Tool | What it measures | Works on this machine? |
|---|---|---|
| **Nsight Systems** (`nsys`) | timeline: kernels, copies, memsets, API calls, **device memory allocations over time**, CPU threads | **yes**. CUDA tracing does not need GPU performance counters. Used in §5.6 |
| **Nsight Compute** (`ncu`) | per-kernel hardware counters: achieved occupancy, cache hit rates, pipe utilisation, memory throughput | **no**: GPU performance counters are blocked on this machine (D-011, owner's decision). Every `TODO(measure: Nsight Compute …)` in this report waits for this |
| `cuobjdump --dump-resource-usage`, `-sass` | registers, stack, shared and constant memory per kernel; the actual instructions | yes. Used in §5.1–5.2 |
| CUDA occupancy API | theoretical blocks per SM | yes (chapter 2, §2.3) |
| **Our own instrumentation** | per-block and per-warp `%smid`, `%globaltimer`, `clock64` timelines; per-warp interaction counters | yes. This is how chapter 4 measured load imbalance *without* a profiler (D-011) |
| `cudaEvent` timing | kernel and phase times | yes. Every rate in chapters 3 and 4 |
| Windows process counters | host RAM (working set) of the process | yes. §5.6.3 |

The project's own timeline instrumentation exists *because* Nsight Compute was not available.
It is a legitimate profiling method: it reports which SM ran each block and when. That is enough
to compute per-SM busy time, idle SMs and the wave tail (chapter 4, §4.1).

### 5.5.2 How the memory profile was taken (repeatable)

```bash
nsys profile --trace=cuda --cuda-memory-usage=true --sample=none --cpuctxsw=none -o <name> \
    build/nbody_bench.exe --kernel bh_warp --n 1048576 --runs 5 --no-validate
nsys export --type sqlite -o <name>.sqlite <name>.nsys-rep
python tools/memprofile_analysis.py results/2026-10-22_memory_profile
```

The script reads `CUDA_GPU_MEMORY_USAGE_EVENTS` (every allocation and free with its size),
replays them to get the **peak live device memory**, and sums copies, memsets and kernel times
per kernel. Host RAM was sampled as the process's peak working set every 50 ms
(`host_ram.txt`).

**The raw traces are not in the repository.** An `.nsys-rep` (and its `.sqlite` export) records
the profiled process's full environment, which includes personal and account variables. The
traces stay on the profiling machine (`.gitignore`); `analysis.txt` and the `*.stdout.txt` files
are the committed outputs, and the commands above regenerate the traces.

### 5.5.3 Profiling work still open

- Nsight Compute, once counters are enabled: achieved occupancy, L1/L2 hit rates of #5 vs #7,
  FMA vs XU pipe utilisation of the all-pairs loop.
- Read-only path for #7: pass `__restrict__` pointers so that `bhWarpKernel` also gets
  `LDG.E.128.CONSTANT`, and time it. This is cheap, and the hypothesis is pre-registrable.

---

## 5.6 Profiling results: memory (VRAM and RAM)

### 5.6.1 Device memory (VRAM), measured with Nsight Systems [M]

| Kernel | N | Allocations | **Peak live device memory** | Per body |
|---|---|---|---|---|
| `allpairs_tiled` | 65,536 | 3 | 3.0 MiB | 48 B |
| `allpairs_tiled` | 1,048,576 | 3 | 48.0 MiB | 48 B |
| `bh_warp` | 262,144 | 28 | 53.2 MiB | 212.8 B |
| `bh_sorted` / `bh_warp` | 1,048,576 | 28 | 212.8 MiB | 212.8 B |
| `bh_warp` | 4,194,304 | 28 | 851.1 MiB | 212.8 B |

Memory grows **exactly linearly in N** for both algorithms [M]. **Barnes-Hut is O(N log N) in
time but O(N) in memory.** It needs 4.4× the memory of all-pairs per body [D]. In exchange, it is
145× faster at 1M (5.99 ms against about 870 ms, chapter 3) [M].

**Where the 212.8 B per body goes** [D, from the allocation sites in `src/core/kernels_tree.cu`,
`src/core/nbody.cu` and `src/bench/main.cpp`; the total matches the measurement to 0.1 B]:

| Group | Arrays | B / body |
|---|---|---|
| Simulation state | `pos`, `vel`, `acc` (3 × `float4`) | 48 |
| Per-body tree data | 3 key arrays (3 × 8), `iota` / `perm` / `hOrder` / `leafParent` / `cohOrder` (5 × 4), `sortedPos` (16) | 60 |
| Per-node tree data (N − 1 internal nodes) | `nodeCom`, `nodeLo`, `nodeHi`, `nodeInfo` (4 × 16), `nodeRc2`, `nodeParent`, `arrive` (3 × 4) | 76 |
| CUB radix-sort temporary storage | `sortTmp` (12.28 MiB at 1M) | 12.3 |
| Per-warp unit cost arrays (#6) | `unitCost`, `unitPre` (2 × 8 per 32 bodies) | 0.5 |
| **Bench instrumentation** (not needed in production) | work counters, 1 `uint4` per body | 16 |
| **Total** | | **212.8** |

A production build without the bench counters needs about **197 B per body**: about 197 MiB per
million bodies [D]. The 16 GB card, minus the driver's reserve and the desktop's share (about
2 GB in use at idle [M, `nvidia-smi`]), therefore holds roughly **70–80 million** bodies for
Barnes-Hut and about 300 million for all-pairs, which would then take hours per step [D].

### 5.6.2 Transfers and phases at N = 1M (`bh_warp`, profiling run) [M]

- **Host → device:** 2 copies, 32 MiB in total (initial positions and velocities, once).
- **Device → host:** 4 copies, 40 MiB (results and instrumentation, once at the end).
- **Memsets:** 157, 102 MiB in total (resetting the arrival counters before each tree build, and
  the bench's poison and zero fills).
- **No transfer happens inside a time step.** This is the "GPU-resident" design in action.

**Where the GPU time goes** in one Barnes-Hut step at 1M (share of kernel time) [M]:

| Phase | `bh_sorted` (#5) | `bh_warp` (#7) |
|---|---|---|
| Traversal (the force calculation) | **98.3 %** | **92.5 %** |
| Tree summary (`multipoleKernel`) | 1.1 % | 4.0 % |
| Radix sort (CUB, 8 passes) | 0.4 % | 2.3 % |
| Radix-tree build, bounding box, keys, gather | 0.3 % | 1.2 % |

Building the tree costs a few percent at most. Even after #7 made the traversal 5.8× faster, the
traversal is still above 90 % of the step. **This is why every load-balancing technique in
chapter 4 targets the traversal and not the build** [D].

### 5.6.3 Host memory (system RAM) [M]

| Kernel | N | Peak working set of the process |
|---|---|---|
| `allpairs_tiled` | 65,536 | 121.9 MiB |
| `bh_sorted` | 1,048,576 | 214.9 MiB |
| `bh_warp` | 1,048,576 | 214.6 MiB |
| `bh_warp` | 4,194,304 | 478.1 MiB |

About **120 MiB is a fixed cost** that does not depend on N: the CUDA runtime, the driver and the
loaded kernels [D: it is the value at the smallest N]. Above that, host memory grows by about
**88 B per body** [D: (478.1 − 214.6) MiB ÷ 3 M bodies]. That is the initial conditions and the
copies of results kept for checks. Host RAM is never the limit on this machine. Its role is to
create the initial conditions and to receive the results.

---

## 5.7 Summary: the four answers in one table

| Question | Short answer | Evidence |
|---|---|---|
| ILP? | Inside one interaction there is little; across interactions there is a lot. Unrolling and prefetch expose it. It matters most when TLP is missing | unroll 1 −21 %; prefetch +16 % at 2 warps/SM, 0 % at full occupancy |
| TLP? | One thread per body, up to 48 warps/SM. It fixes all-pairs starvation (#3: 2.56×) but does nothing for the tree (flat 2–11 blocks/SM) or for divergence (fixed by #7 instead) | chapter 4, §4.4, §4.5.1, §4.7 |
| Which memory, for what? | registers: the body and its sums (and #7's stack); shared: the all-pairs tile; L1/L2: the tree; VRAM: everything between steps; host: start and end only | §5.2 table, SASS and resource dump |
| Memory optimisation? | `float4`, tiling, Morton sort, register stack, broadcast loads, large L1, allocate once, GPU-resident state | §5.4 |
| Profiling (RAM)? | VRAM: 48 B/body all-pairs, 212.8 B/body BH (197 production), linear in N; host: 120 MiB fixed + 88 B/body; traversal > 90 % of the step | §5.6, nsys |

---

## 5.8 Glossary (EN → BG)

| English | Български |
|---|---|
| instruction-level parallelism (ILP) | паралелизъм на ниво инструкции |
| thread-level parallelism (TLP) | паралелизъм на ниво нишки |
| latency hiding | скриване на латентността |
| dependency chain | верига от зависимости |
| loop unrolling | разгъване на цикъл |
| prefetch | предварително зареждане |
| occupancy | заетост (на SM) |
| register file | регистров файл |
| local memory | локална памет |
| shared memory | споделена памет |
| constant memory | константна памет |
| cache hit / miss | попадение / пропуск в кеша |
| working set | работно множество |
| coalesced access | обединен (коалесциран) достъп |
| broadcast load | зареждане с разпространение (един адрес за всички нишки) |
| memory footprint | обем на използваната памет |
| profiler / profiling | профилиращ инструмент / профилиране |
| peak working set | пиково работно множество (RAM) |

---

## 5.9 Questions the lecturer is likely to ask

1. **"What is the difference between ILP and TLP?"** Both hide latency. TLP uses other warps
   while one waits. ILP uses independent instructions of the same thread. Our proof that they
   substitute for each other is the prefetch result: +16 % with 2 warps per SM, 0 % with a full
   SM.
2. **"Why not raise occupancy further for the tree?"** It is already 100 %, and the sweep shows
   the rate flat from 2 to 11 blocks per SM. The tree is limited by L1 capacity and divergence,
   not by the number of warps.
3. **"Why not put the tree in shared or constant memory?"**
   - The tree is 76 B per body: 76 MiB at 1M. Shared memory is 100 KB per SM and constant memory
     is far smaller still.
   - Shared memory would also have to be filled by each block separately.
   - Using shared memory would shrink L1, and we measured that a smaller L1 costs #5 a third of
     its speed.
4. **"Is all-pairs memory-bound?"** No. Its data fits in L2, the tile reuses each load
   256 times, and the 1M run scaled exactly as N² (861 ms measured against 870 ms extrapolated).
   It is limited by instruction issue (chapter 2, §2.2.3).
5. **"How did you profile without Nsight Compute?"**
   - Nsight Systems for the timeline and memory.
   - `cuobjdump` for registers, stack and SASS.
   - Our own per-block SM timelines for load balance.
   - Counter-based metrics (cache hit rates, pipe utilisation) are listed as open TODOs, not
     guessed.
6. **"How much memory does your simulation need?"** It is linear in N:
   - 48 B per body for all-pairs;
   - about 197 B per body for Barnes-Hut (212.8 B with the bench counters);
   - plus about 120 MiB of fixed host-side runtime cost.
   - So the 16 GB card holds roughly 70–80 M bodies for Barnes-Hut.
