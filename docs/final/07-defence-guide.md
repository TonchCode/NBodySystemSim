# 7. Defence Guide: The Whole Project, Step by Step

> **Purpose.** This chapter is for studying, not for the report. It walks through what the code
> does, one variant at a time, and for each step answers three questions: **what changed**, **what
> it measured**, and **why** (the mechanism). Understanding the *why* matters more than
> remembering the numbers. Every number here is copied from chapters 2–6, and the source is given
> so you can open it when asked.
>
> **Language note:** English; the Bulgarian/English choice for the final document is still open.

**How to use it.** Read §7.1–7.2 once. Then go through §7.3 variant by variant and try to explain
each "why" out loud without looking. §7.5 lists the questions to practise.

---

## 7.1 The problem in four sentences

1. N bodies (stars), each with a mass and a position. Every body pulls on every other body with
   gravity: a ∝ m / r² (chapter 3, §3.1).
2. Each time step: compute every body's acceleration (**the expensive part**), then move the
   bodies a little with the leapfrog integrator (cheap, O(N)) (§3.3).
3. Direct summation is N × N interactions: O(N²). At N = 1M that is 10¹² interactions per step.
4. **Barnes-Hut** replaces a far-away group of bodies by one point at its centre of mass, which
   gives about O(N log N) at a controlled error set by the opening angle θ (§3.4).

**The one function everything calls** (`src/core/interaction.cuh`):

```
r   = x_j − x_i                         3 FADD
d2  = r·r + ε²                          3 FFMA
inv = rsqrt(d2)                         1 MUFU.RSQ   (the special-function unit)
s   = m_j · inv³                        3 FMUL
a  += r · s                             3 FFMA
```

About 20 FLOP per interaction. All variants, all-pairs and tree, call **exactly this function**,
so "one interaction" means the same amount of work everywhere and rates can be compared directly.

---

## 7.2 The GPU in one page (what you need for every "why")

| Fact about the RTX 4080 SUPER | Why it matters for N-body | Chapter |
|---|---|---|
| 80 SMs, each runs up to 48 warps (1,536 threads) | you need ≥ 80 × (several) blocks or SMs sit idle: **starvation** | 2, §2.3 |
| a warp = 32 threads doing **the same instruction** | if the 32 threads want different things, they take turns: **divergence** | 2, §2.3 |
| 4 schedulers per SM, each issues 1 instruction per cycle | when one warp waits, another issues: **TLP hides latency** | 5, §5.1.3 |
| independent instructions in one thread can overlap | **ILP**: unrolling lets the chains of several interactions overlap | 5, §5.1.1 |
| registers (fastest) → shared/L1 (≈ 30 cycles) → L2 64 MB (≈ 270–285) → VRAM (≈ 570) | keep reused data as close as possible | 5, §5.2 |
| 1 MUFU (rsqrt) per interaction; rsqrt throughput is 16 results/clock/SM against 128 FP32 FMA (8 : 1) | the all-pairs loop is limited by **instruction issue**, with the SFU as a co-limit, not by memory | 2, §2.2.3 |

Latency figures are from microbenchmarks of the RTX 4090, the same architecture (cc 8.9)
[P:luo2025hopper, Table 3]. Our own card is not measured yet (`TODO(measure)` in ch.5).

**Three levels where load can be unbalanced** (chapter 4, §4.0). Each of our variants attacks one:

| Level | What is unbalanced | Symptom |
|---|---|---|
| 1. Between SMs | some SMs have blocks, others none; the last wave is half empty | idle SMs, tail |
| 2. Between warps / blocks | some warps have more work than others | per-warp max/mean > 1 |
| 3. Inside a warp | lanes of one warp follow different paths | SIMT efficiency < 1 |

---

## 7.3 The variants, one by one

Read each card as a chain: **previous variant → what was wrong with it → what we changed →
measured result → mechanism → what new limit appeared.** The new limit is the motivation for the
next card.

### Reference: `allpairs_naive`

- **What it does.** One thread per body; loops over all N bodies straight from global memory.
- **Role.** The **correctness reference**: every other kernel is validated against it (and against
  FP64 on a subset). It is not a speed baseline.

### #1 `allpairs_tiled`: static tiling (the baseline)

- **What changed.** The block loads a **tile** of 256 positions into shared memory once; all 256
  threads reuse it; two barriers per tile [P:nyland2007nbody]. Each position read from VRAM is now
  used 256 times (ch.5, §5.3).
- **Resources.** 40 registers, 4 KB shared per block, 6 blocks/SM = **100 % occupancy, with zero
  register headroom** (41 registers would drop it to 83 %) (ch.2, §2.3).
- **Measured.** Per-thread work is perfectly equal, but per-SM work is not: **76 / 64 / 40 idle SMs**
  at N = 1,000 / 4,096 / 10,000, and a **23.5 % tail** at N = 65,536 (ch.4, §4.2).
- **Why.** N = 1,000 makes only 4 blocks for 80 SMs. Balanced threads do not mean a balanced GPU.
- **New limit.** Level 1: not enough work items for small N, and a wave tail at medium N.

### #2 `allpairs_tuned`: block size × unroll

- **What changed.** 25 template instances: block ∈ {64…1024} × unroll ∈ {1, 4, 8, 16, full}.
- **Measured.** The best one (b128, full unroll) is **1.14–1.54× over tiled b256** and becomes the
  **official baseline** for every all-pairs speedup (rule 8, D-013).
- **Why (two separate effects):**
  - *Block size (TLP / level 1).* Smaller blocks → more blocks → more SMs busy and a smaller last
    wave. b64 does remove the tail (SM efficiency 0.97 vs 0.81 at 65,536), **but is only 8 %
    faster, not the predicted 20 %**: four times as many barriers and tile loads make each
    interaction ≈ 12 % more expensive (ch.4, §4.3; D-012). **A model miss we logged.**
  - *Unroll (ILP).* Unroll 1 is **−21 %**; full unroll +8 %. Without unroll each interaction's
    11-instruction chain waits for itself; unrolled, the compiler interleaves several chains. Full
    unroll at block 1024 is **−18 %**: the loop no longer fits the instruction cache. ILP has a
    cost (ch.5, §5.1.2).
- **New limit.** Still one thread per body, so at N = 1,000 there are still only 1,000 threads.

### #3 `allpairs_multi`: q threads per body

- **What changed.** Each body's sum is split over **q threads** (thread s takes j ≡ s mod q); the q
  partial sums are combined with `__shfl_xor_sync` in log₂ q steps, registers only. Plus a register
  **prefetch** of the next tile (ch.4, §4.4).
- **Measured vs the #2 baseline.** **2.40× / 2.56× / 1.40× / 1.02×** at N = 1,000 / 4,096 / 10,000 /
  65,536. Idle SMs 64 → 16 at 4,096.
- **Why.** This is **more TLP**: q× more threads, so small problems fill all 80 SMs. The gain is
  exactly as large as the starvation was, and disappears once every SM is busy (65,536).
- **The prefetch is the ILP/TLP lesson in one table.** +16.1 % at 2 warps/SM, +10.1 % at 4,
  +7.7 % at 8, **0 % at full occupancy**. Prefetch only helps when there are too few warps to hide
  the load: **ILP replaces missing TLP** [P:volkov2010lower] (ch.5, §5.1.2).
- **New limit.** None for all-pairs at large N; it is compute-bound. The only way forward is to do
  **fewer interactions**: the tree.

### #4 `bh_naive`: Barnes-Hut, bodies in their original order

- **What changed.** The 7-step GPU pipeline (ch.3, §3.5): bounding box → Morton keys → sort →
  radix tree (Karras, one thread per node, no locks) → bottom-up summary (atomic arrival counters)
  → **traversal** → integrate. But each thread walks the tree for a body in the **original,
  unsorted** order.
- **Measured.** Per-warp max/mean 1.17–1.62; **SIMT efficiency at interaction sites 0.09–0.13**:
  on average only 3–4 of 32 lanes do useful work (ch.4, §4.5).
- **Why.** The 32 bodies of a warp are scattered over the galaxy; each wants a different path;
  the warp executes the paths one after another (level 3).
- **Memory.** Each thread has a private stack `int stack[96]` = **384 B of local memory**, which
  physically lives in VRAM and is cached in L1 (SASS: `STACK:384`) (ch.5, §5.2).

### #5 `bh_sorted`: threads in Morton order

- **The one difference from #4.** Thread i handles the i-th body **along the Morton curve**, so a
  warp's 32 bodies are spatial neighbours (ch.4, §4.5).
- **Measured.** **2.01× / 2.06× vs #4** best against best at N = 262,144 / 1M (θ = 0.5). Hilbert
  order gives the same as Morton. Crossover with the best all-pairs kernel at **N ≈ 39,700**.
- **Why.** Neighbours walk almost the same path → less divergence, and neighbouring warps share
  tree nodes in L1 → better locality. (The two are not separated; the 2.0× is both.)
- **What limits it: L1 capacity.** Flat from 2 to 11 blocks/SM (more TLP does not help), but
  **×1.51 when the full L1 is available**. The working set, not the warp count, is the limit
  (ch.4, §4.5.1; ch.5, §5.3).
- **Still lost.** Only 0.201 SIMT efficiency at interaction sites remains; the coherent control
  says incoherence still costs a factor of ≈ 4–10 (ch.4, §4.5.5).

### #6 persistent threads + software queues (a **negative** result)

- **What changed.** Instead of the hardware block scheduler, a fixed set of resident warps pulls
  work from a queue: warp-granular, lane-granular (refill idle lanes), or a static partition
  (ch.4, §4.6).
- **Measured.** **Slower in every cell: 0.63–0.97×** of the hardware scheduler, **even though every
  arm improved its own balance metric**.
- **Why (exploratory diagnosis, not pre-registered).**
  - The warp queue scatters Morton-adjacent warps over different SMs, so they no longer share L1.
    #5 is L1-bound (above), so this costs directly. A block-granular queue that keeps 4 adjacent
    warps together recovers most of it (0.94–0.98×).
  - Lane refill pays warp-collective calls every pop (reconvergence). Not yet confirmed in SASS.
- **The lesson.** The hardware block scheduler **is already a dynamic queue** that keeps neighbours
  together. **Better balance ≠ better performance** when the kernel is bound by L1 capacity.
  A well-measured negative result is a result.

### #7 `bh_warp`: warp-cooperative traversal (union walk)

- **What changed.** One **warp** walks the tree together. Each lane does its own opening test; the
  node is opened if **any** lane needs it (`__any_sync`). The warp walks the union of its lanes'
  paths (Burtscher's warp vote) (ch.4, §4.7).
- **Memory change.** The stack moves from 384 B local memory per thread to **registers**: 64
  entries spread over 2 registers per lane; push = predicated move, pop = one `__shfl_sync`.
  0 B stack frame, 0 spills (ch.5, §5.4 #6).
- **Measured.** **5.8× vs #5** best against best at 1M (34.45 → 5.99 ms), 4.8× at 262,144.
  SIMT efficiency **1.000**. Real rate **321 G interactions/s** vs 28.8 for #5. Crossover with
  all-pairs drops to **N ≈ 28,700**.
- **The paradox to explain.** #7 does **1.64–1.85× more interactions** than #5 and is still 5.8×
  faster. Why? Because every instruction now runs on all 32 lanes, every load is one broadcast
  instead of up to 32 scattered ones, and there is no local-memory stack. **It adds work and
  removes incoherence.** It is *not* work stealing (D-022).
- **Bonus.** More accurate than #5 (rms error 4.59 × 10⁻³ vs 6.55 × 10⁻³ at 262,144), because each
  lane interacts with at least the cells it would have used alone.
- **Needs #5.** On unsorted bodies the union is huge and the method collapses (ch.4, §4.7.3).
- **New limit: latency.** It speeds up with more warps (×1.87 from 4 to 11 blocks/SM) but barely
  with more L1 (×1.12). The dependent pop chain (shuffle → load → test → vote → branch) is the
  limit, so the next step would be wider tree nodes (fewer pops), not more occupancy (D-026).

### #8 CPU + GPU split (a second **negative** result)

- **Idea.** Use the CPU too (i5-13600K: 6 P-cores + 8 E-cores): split the bodies between CPU and
  GPU (`hyb_bh`), or build the tree on the CPU while the GPU computes forces (`hyb_phase`).
- **Measured.** **No split beats the GPU alone**: best gain +1.3 %, most cells 0.99–1.27× *slower*
  (ch.4, §4.8.1).
- **Why.** The GPU is **18.3× the best CPU code** on the 1M tree and 29–36× on all-pairs, so the
  CPU can take only a tiny share; and the body split moves 56 B/body over PCIe every step
  (2.26 ms at 1M, 2.3× a whole cube_edges step).
- **The trap we caught.** `hyb_phase` *looked* 4–25 % faster. A GPU-only control that just reuses
  and refits the old tree (`bh_warp_refit`) is equally fast (within 1.4 %), so the gain was never
  the CPU's. And the timing loop keeps bodies still, so it does not charge for the older tree
  (+4–10 % interactions in a real 500-step run). **Always ask: what is the control?**
- **Load balancing on the CPU itself.** Splitting the work evenly over P- and E-cores is *slower
  than using the P-cores alone* (21.4 vs 31.0 G/s), because each E-core finishes ≈ 2.4× later.
  Dynamic scheduling fixes it: +22 % (37.8 G/s). Equal work ≠ equal time on unequal cores.
- **Memory.** The only variant that uses **pinned host memory** (`cudaHostAlloc`): 21–26 GB/s
  device→host. The CPU-built tree is bitwise identical to the GPU tree (phase 1).

---

## 7.4 The differences in one table

| # | Unit of work | Where the hot data lives | Level attacked | Kind of parallelism added | Result (best vs best) | Limited by |
|---|---|---|---|---|---|---|
| naive | thread per body | VRAM via L1/L2 | — | — | correctness reference | memory |
| 1 | thread per body | tile in **shared** | — | data reuse | idle SMs, 23.5 % tail | starvation / tail |
| 2 | thread per body | shared + registers | 1 | ILP (unroll), TLP (block) | 1.14–1.54× over #1 | issue + SFU |
| 3 | q threads per body | shared + registers | 1 | **TLP** (+ prefetch ILP) | 2.40–2.56× at small N, 1.02× at 65k | issue + SFU |
| 4 | thread per body, tree | tree in L1/L2, **stack in local** | — (tree baseline) | O(N log N) work | SIMT eff. 0.09–0.13 | divergence |
| 5 | thread per body, sorted | tree in **L1**, stack in local | 3 (and locality) | — (ordering) | 2.0× over #4 | **L1 capacity** |
| 6 | persistent warps + queue | as #5 | 1–2 | dynamic scheduling | 0.63–0.97× (**slower**) | lost L1 locality |
| 7 | **warp** per 32 bodies | tree in L1/L2, **stack in registers** | 3 | warp-uniform SIMT | 5.8× over #5 | **pop latency** |
| 8 | CPU + GPU split | host (pinned) + VRAM | between devices; P/E cores | heterogeneous | **no gain** (best +1.3 %); on the CPU, dynamic +22 % over P-only | PCIe, 18× rate gap |

**The story in one line per row:** tile it → tune it → split it → change the algorithm → sort it →
(don't hand-schedule it) → walk it as a warp → (don't offload it).

---

## 7.5 How to read the results (so you can defend them)

- **Speedups are best against best.** Each variant is tuned over its own block/leaf size before it
  is compared (rule 8, D-013, D-025). A speedup against an untuned baseline would be inflated.
- **Tree rates are real interaction rates**, not "N²-equivalent". The bench prints both; we quote
  only the real one (ch.4, rules).
- **Medians of repeated runs**, and the load-sensitive results were **replicated at 0 % background
  load** (2026-10-18). Those clean values are canonical (D-033).
- **Pre-registered predictions.** Before each round we wrote down what we expected. Tree round 2
  scored **39 HIT / 22 MISS**. The misses are reported, not hidden (b64 at +8 % instead of +20 %;
  #6 slower instead of faster).
- **Power cap.** Runs are at the 272 W cap; faster kernels run at lower clocks, so speedups compare
  different clocks. This favours the tree in the crossover numbers (ch.4, §4.5.3).
- **Memory profile** (nsys, ch.5, §5.6): VRAM is exactly linear in N, 48 B/body for all-pairs and
  ≈ 213 B/body for the tree in the bench (≈ 197 B/body in production); no transfers inside a time
  step; host RAM ≈ 120 MiB fixed + 88 B/body.

---

## 7.6 Practice questions (answer each with a mechanism)

1. *Why is #1 slow at N = 1,000 if every thread does the same work?* → 4 blocks for 80 SMs.
2. *Why did block 64 not give the predicted 20 %?* → 4× the barriers and tile loads per
   interaction: +12 % cost per interaction ate most of the tail gain.
3. *What is the difference between ILP and TLP? Give one experiment for each.* → unroll −21 % /
   +8 % (ILP); q-split 2.56× (TLP); prefetch +16 % → 0 % shows one replaces the other.
4. *Why does the prefetch help at 2 warps/SM and not at 48?* → with 48 warps the scheduler always
   has another warp ready while the load is in flight.
5. *Why is all-pairs not memory-bound?* → each position loaded once per block is reused 128–256
   times from shared memory; the loop is limited by issue slots and the SFU.
6. *Which GPU memory is best for which data, and why?* → per-body state and accumulators:
   registers; the all-pairs tile: shared; the tree: L1/L2 (read-only, re-read by neighbours);
   everything: VRAM as home; host only outside the time step.
7. *Why does sorting make the tree 2× faster?* → neighbours walk the same path (less divergence)
   and share nodes in L1 (locality).
8. *Why does #5 not speed up with more occupancy?* → L1 capacity bound: more warps share the same
   L1; the full-L1 run gives ×1.51.
9. *#6 balanced the work better. Why was it slower?* → it scattered Morton-adjacent warps across
   SMs and lost L1 sharing. The hardware scheduler is already a locality-preserving queue.
10. *#7 does 1.6–1.9× more interactions. How is it 5.8× faster?* → SIMT efficiency 0.20 → 1.00,
    broadcast loads, stack in registers instead of local memory.
11. *Why is #7 more accurate?* → each lane uses at least the cells it would have accepted alone.
12. *Why not use tensor cores or RT cores?* → ch.2, §2.2.6.
13. *Why not use the CPU as well?* → measured: no gain. The GPU is 18× the CPU on the 1M tree, PCIe
    costs 2.26 ms per step for the body split, and the phase split's apparent gain is matched by a
    GPU-only refit (ch.4, §4.8.1).
14. *How did you balance work between P-cores and E-cores?* → dynamic scheduling; a static even
    split was slower than P-cores alone (21.4 vs 31.0 G/s), dynamic gave 37.8.
15. *Did you try `__restrict__` / the read-only cache for #7?* → yes, pre-registered: the loads
    became `LDG.E.128.CONSTANT`, the time did not change (0.0–0.3 %), because #7 is latency-bound
    (D-037).
16. *How much memory does the simulation need, and how many bodies fit?* → ≈ 197 B/body in
    production BH → 70–80 M bodies in 16 GB (ch.5, §5.6).
17. *Where is Barnes-Hut used outside astronomy?* → t-SNE, graph layout (Gephi ForceAtlas2,
    d3-force with θ = 0.9) (ch.6, §6.2.3).
18. *Which result surprised you, and what did you learn from it?* → #6: better balance, worse
    performance. Balance is a means; the real limit was L1.

---

## 7.7 Where to look in the code (live demo order)

| Step | File | What to point at |
|---|---|---|
| the interaction | `src/core/interaction.cuh` | the 20-FLOP function, `rsqrtf` |
| #1 / #2 | `src/core/kernels_allpairs.cu`, `kernels_tuned.cu` | tile load, `__syncthreads`, the block × unroll templates |
| #3 | `src/core/kernels_multi.cu` | j ≡ s (mod q), `__shfl_xor_sync` butterfly, prefetch |
| tree build | `src/core/kernels_tree.cu` | Morton keys, CUB sort, `radixTreeKernel`, `multipoleKernel` |
| #4 / #5 | `kernels_tree.cu`, `bhTraverseKernel` | `int stack[96]`, the opening test |
| #7 | `kernels_tree.cu`, `bhWarpKernel` | `__any_sync` vote, register stack with `__shfl_sync` |
| #8 | `src/core/kernels_hybrid.cu` | CPU tree build, pinned buffers |
| live | `run.bat` | switch `bh_naive` → `bh_sorted` → `bh_warp` and watch SIMT efficiency go 0.1 → 0.2 → 1.0 |
