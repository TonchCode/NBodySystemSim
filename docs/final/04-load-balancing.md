# 4. GPU Load-Balancing Techniques

> **Language note:** drafted in English; the Bulgarian/English choice for the final document is still open (`docs/00-course-requirements.md`, open questions).

**Source tags** follow chapter 2: **[M]** measured, **[D]** derived from tagged values,
**[N:key]** NVIDIA or vendor document, **[P:key]** paper (all keys in `references/references.bib`,
all marked ✅ in `references/INDEX.md`). Every number below carries its results path. Missing
items are marked `TODO(source needed: …)`, `TODO(measure: …)` or `TODO(results: …)`.

**Rule for speedups** (`CLAUDE.md` rule 8, D-013): every speedup is quoted against the
**best-tuned baseline**, i.e. the fastest one-thread-per-body configuration measured in the same
session (tiled, tuned over all block × unroll, and multi with q = 1). In the round-2 session that
was `allpairs_tuned` b128 with full unroll at every N (`results/2026-10-17_round2/summary.md` §3).

**Rule for tree rates.** For the Barnes-Hut kernels (`bh_*`) the bench's headline
"G interactions/s" and "GFLOP/s" columns are N²-equivalent (all-pairs work divided by tree
time), not real rates (`docs/reviews/2026-10-17-critic-2.md` #5). This chapter never uses them.
Tree rates below are real traversal rates: the "trav G int/s" column of
`results/2026-10-17_tree_round1/summary.md` §1 for round 1, and the rates stated in
`results/2026-10-17_tree_round2/summary.md` for round 2. From round 2 on, the bench labels the
N²-equivalent figure as such and writes it to a separate column (D-021).

**Rule for tree speedups.** Tree speedups are quoted best against best over leaf size and block
size, with θ stated (D-025). Where a fixed protocol configuration (leaf 16, block 128) is used for
comparability across rounds, it is labelled as such.

**Platform caveat that applies to every number.** All runs used the 272 W power cap, which our
kernels reach. Under it, faster kernels run at lower SM clocks (2.52 GHz vs 2.70 GHz at
N = 65,536), so speedups compare kernels at *different* clocks (chapter 2, §2.6;
`results/2026-10-17_round2/summary.md` §7, P-TL2).

---

## 4.0 Where imbalance can arise on sm_89

Chapter 2 gives the structure: 80 SMs, each with 4 partitions and up to 48 resident warps and
24 resident blocks [M; N:nvidia2026adatuning]. Load imbalance can therefore appear at three
levels, and each technique in this chapter targets one or more of them:

1. **Across SMs.** Blocks are the unit of work an SM receives, and all threads of a block run on
   one SM [N:nvidia2026cudaguide, §1.2.2.1]. Too few blocks leave SMs idle (*starvation*). A
   block count that is not a multiple of the resident slots leaves some SMs finishing early
   (*the wave tail*).
2. **Between warps or blocks on the same SM.** Some warps carry more work than others.
3. **Inside a warp.** The 32 threads of a warp execute together [N:nvidia2026cudaguide,
   §1.2.2.2]. If one lane has more work or takes a different branch, the others wait.

The all-pairs kernels (#1–#3) are perfectly balanced *per thread*: every thread computes N
interactions. Their only imbalance is level 1 (`docs/03-design.md` §10). The tree kernels
(#4–#7) add levels 2 and 3, because the work per body depends on where the body sits.

## 4.1 Measuring imbalance without a profiler

GPU performance counters are blocked on this machine, so Nsight Compute's per-SM activity
metrics are unavailable (D-011, `docs/decisions.md`). Imbalance is therefore measured with our
own instrumentation.

**Per-block SM timelines (D-011).** Each force kernel has an instrumented twin that records, for
every block, the SM it ran on (`%smid`) and its start and end time (`nbody_bench --timeline`).
`tools/timeline_analysis.py` derives per-SM busy time, the number of idle SMs, the max/mean SM
busy time, the *SM efficiency* (mean busy time over span) and the *tail* (the share of the span
after the first SM goes idle) (D-011; `results/2026-10-17_round2/summary.md` §1). The metrics
are medians of 5 launches. Gantt plots are in `results/2026-10-17_round2/timeline/plots/` and
`results/2026-10-17_tree_round1/timeline/plots/`.

Tree round 2 added *warp-unit* timelines (one record per 32-slot unit, or per persistent warp), so
persistent kernels have a meaningful timeline and warps idling inside a block become visible
(`docs/03-design.md` §17.5).

**Clock calibration.** `%globaltimer` advances in 1.024 µs steps on this board [M], about the
length of a whole N = 1,000 kernel (D-011). The twins therefore time blocks with each SM's cycle
counter `clock64`, anchored to timer tick edges before and after the launches. Because the SM
clock changes under the power cap, the mapping can drift by up to 37 µs (1 % of span).
Calibrated and raw metrics agree within about 1 % on every headline number, and both are
reported (D-011).

**Per-warp work counters (tree kernels).** The instrumented tree traversal counts, per thread,
the interactions and the node pops it performs (`docs/03-design.md` §14). From these the
analysis computes:
- the *per-warp max/mean* (a warp runs until its slowest lane finishes, so this is the intra-warp
  imbalance; 1 is perfect);
- the *SIMT efficiency* at pop sites and at interaction sites, from `__activemask()` leader
  counts (the mean fraction of active lanes);
- the *warp-cost max/mean* (how unevenly cost is spread between warps).

The counts are algorithmic, so they are the same for the production kernel. The counting code
may, however, move ptxas's reconvergence points, so the SIMT efficiencies describe the twin as
executed (`docs/03-design.md` §14).

**The instrumentation-overhead caveat.** The pre-registered expectation was an overhead of at
most 1 %. The measured difference between twin and production is −3.4…+6.2 %
(`results/2026-10-17_round2/summary.md` §1). The production SASS is unchanged for all
57 kernels; the difference comes from ptxas scheduling the twin's force loop differently
(same opcodes, different schedule and registers) (D-011). Hence the rule used throughout:
**timelines give structure** (placement, idle SMs, tail fraction), **production kernels give
throughput**. The main session re-checked two timelines independently and reproduced them
(64 idle SMs, SM efficiency 0.198 vs 0.199; 16 idle, 0.790 vs 0.786)
(`results/2026-10-17_round2/summary.md` §8).

**Correctness gates.** Every validated run is gated in code: the D-009 gates for all-pairs and
the TG1/TG2 gates for trees, with an exact-arithmetic selftest that catches a single missed
interaction (D-009, D-017). A faster kernel that drops work therefore cannot enter this chapter.

### 4.1.1 Replicating without background load (the clean re-run)

The round-2 tree results (§4.5–4.7) were measured under 20–22 % uncontrolled background GPU load. Rather than
argue the effect away, the load-sensitive items were **replicated** (D-033;
`results/2026-10-18_clean_rerun/summary.md`; `docs/03-design.md` §22):
- **Pre-registered.** The replication plan was committed before any timed run (fbe0742), and each
  item was scored against its *original* committed range, not a new one. A resume option added
  later (bfe3c5b) only skips cells and changes no command.
- **Controlled.** The same binary (SASS identical to the previous round, 222 of 222 functions), the
  same 272 W cap, 0 % idle utilisation before and after, and at most 1 % at every scored
  between-cell check. Within each cell the kernels ran in the order A, B, …, B, A, so within-cell
  ratios are protected against drift. 945 scored runs, all exit 0, no gate failures.
- **An incident, contained.** At 04:56 the Windows screensaver started and loaded the GPU by
  30–44 %. The between-cell idle check flagged it at the next cell. The script ran on for about ten
  minutes before it stopped, so one more cell ran under the screensaver. **Nothing measured after
  04:56 in that sub-session was scored**; those logs are kept separately in `flagged/`. The owner
  disabled the screensaver, and the remaining cells were run from cell 39 in a second sub-session
  under a display-required hold, with no cell flagged. The only comparison that spans the two
  sub-sessions is the 1M best-against-best of set C, and the main session re-checked it
  independently afterwards (5.73× against 5.76×).
- **Outcome.** Of 59 primary items across tree round 2 and the IC round, 36 HIT against the original
  ranges, against 39 before. The three flips are two cells named in advance as edge cases and one
  disputed reading (D-033). **No #6 or #7 conclusion changed**, and all 153 deterministic values
  reproduced exactly. Within the tree-round-2 primary items, 13 of 23 HIT before and after
  (`results/2026-10-18_clean_rerun/summary.md`, scorecard).
- **The proposed mechanism was falsified.** The pre-registered explanation was that per-thread walks
  are load-sensitive and coherent walks are not. The data reject it: the inflation tracked **kernel
  length** and how late in a session a cell ran (in the more heavily loaded IC session, cells of
  about 8–10 ms or more lost 7–17 %, short cells at most 1–3 %), consistent with time-sharing the GPU
  with desktop applications. Round 2's own load cost about 3 % on per-thread walks, inside the ±4 %
  by which two clean sessions differ per cell (D-033). The consequences predicted from the load
  (speedups slightly lower when clean, #6 not rescued, the between-set drift gone) did hold.
- **Process lessons**, recorded for future sessions: disable the screensaver and verify it in the
  registry; never edit a running script; stop the script itself, not only its wrapper
  (`docs/03-design.md` §22).

The clean values are canonical for every re-run item. Set D (occupancy) and the exploratory runs were
not re-run and keep their load label (D-033).

---

## 4.2 Technique #1: static tiling (the baseline)

**Imbalance targeted.** None directly: it is the reference. One thread per body, with the block
size equal to the shared-memory tile, two barriers per tile [P:nyland2007nbody]
(`docs/03-design.md` §1).

**Mapping onto sm_89.** At block 256 the kernel uses 40 registers and 4 KB of shared memory per
block, which gives 6 blocks per SM and 100 % theoretical occupancy (chapter 2, §2.3). The grid
has B = ⌈N / 256⌉ blocks for 80 SMs.

**Measured imbalance.** The per-thread work is equal, but the per-SM work is not
(`results/2026-10-17_round2/summary.md` §1):

| N | Blocks | Idle SMs | SM efficiency | Tail (% of span) |
|---|---|---|---|---|
| 1,000 | 4 | 76 | 0.050 | 0.4 |
| 4,096 | 16 | 64 | 0.199 | 0.6 |
| 10,000 | 40 | 40 | 0.497 | 0.5 |
| 65,536 | 256 (3–4 per SM) | 0 | 0.812 | 23.5 |

So even the "perfectly balanced" baseline leaves 95 % of the GPU idle at N = 1,000 and loses
almost a quarter of its span to the tail at N = 65,536.

**Literature.** No 2025–26 paper on the tiled all-pairs kernel design was found; codes that use
direct summation do not publish the kernel (`references/R3-evidence-map.md` §1). The 2025–26
support for #1 is NVIDIA's 2026 documentation for this device [N:nvidia2026adatuning;
N:nvidia2026bestpractices]. The kernel itself is a settled 2007 technique [P:nyland2007nbody].

**Limitations.** Tiled b256 is not the best one-thread-per-body configuration, so it is not the
rule-8 baseline (§4.3).

## 4.3 Technique #2: block size, tile size and unroll (and the wave tail)

**Imbalance targeted.** Level 1, both modes. Smaller blocks give more, finer work items, so they
spread over more SMs (starvation) and shrink the last partial wave (tail). Block size also
changes the per-SM work: more blocks per SM, more barriers per interaction
(`docs/03-design.md` §3.1). Unroll changes the loop overhead per interaction.

**Mapping onto sm_89.** `allpairs_tuned` instantiates block ∈ {64, 128, 256, 512, 1024} ×
unroll ∈ {1, 4, 8, 16, full} as templates (25 kernels). Blocks 64–512 run at 100 % occupancy;
block 1024 runs at 67 %, because only one 1,024-thread block fits in 1,536 thread slots
(`docs/03-design.md` §3.2; runtime API in `results/2026-10-17_architect_sanity/summary.md`).
At block 64 the 24-blocks-per-SM limit binds exactly (24 × 64 = 1,536).

**Model M1 and how it fared.** The pre-registered model treats an SM as a server with a fixed
rate, so a kernel takes as long as its busiest SM, ∝ ⌈B / 80⌉ blocks (`docs/03-design.md` §0.2,
§3.3). Zhang et al. show that this step function is an idealization: the tail penalty is at
most one wave and is smoothed by variance in block latency [P:zhang2026wavetune]. Our timelines
show where each view holds on this GPU (`docs/03-design.md` §3.3; D-012):
- **Single-wave launches** (all blocks resident at once): the step is **exact**. Tiled b256 at
  N = 65,536 (16 SMs with 4 blocks, 64 with 3) had a 23.5 % tail and SM efficiency 0.81,
  against M1's 0.80; tuned b64 had 0.97 against M1's 0.98.
- **Multi-wave launches** (more blocks than resident slots): the step is **smoothed far below
  one block**, as WaveTune describes. `allpairs_multi` q4 at N = 65,536 (1,024 blocks over
  480 slots) had a 0.1 % tail.
- **What M1 missed.** The pre-registered prediction P2a was that block 64 would beat block 256
  by ≈ 20 % at N = 65,536 (`docs/03-design.md` §3.4). Block 64 does remove the tail as
  predicted (SM efficiency 0.972 vs 0.809), but it is only 8 % faster, because each b64 SM is
  about 12 % slower per interaction: four times the barriers and tile loads
  (`results/2026-10-17_round2/summary.md` §1, exploratory, not scored; D-012). M1's per-block
  cost term (c = 2 interaction units per tile) is too small at block 64. The step function is
  right; the cost term is wrong. M1 is therefore kept to predict *which* SMs idle and *how
  long* the tail is, and block sizes are ranked by measurement (D-012).

**Measured effect.** In the round-2 session, tuned b128 with full unroll was the best
one-thread-per-body configuration at every N (`results/2026-10-17_round2/summary.md` §3). Its
gain over tiled b256 follows from the two columns of that table [D]:

| N | Best multi (G/s) | ÷ "vs baseline" | ÷ "vs tiled b256" | Tuned b128 / tiled b256 [D] |
|---|---|---|---|---|
| 1,000 | 180 | 2.40× | 3.54× | 1.48× |
| 4,096 | 894 | 2.56× | 3.91× | 1.53× |
| 10,000 | 1,188 | 1.40× | 2.16× | 1.54× |
| 65,536 | 1,266 | 1.02× | 1.16× | 1.14× |

The timelines show the mechanism (`results/2026-10-17_round2/summary.md` §1). At N = 4,096,
tuned b128 halves the starvation (48 idle SMs instead of 64, SM efficiency 0.396 vs 0.199). At
N = 10,000 it removes it (1 idle SM, 0.980). At N = 65,536 it shortens the tail from 23.5 % to
13.8 % (SM efficiency 0.914). Because tuned *is* the rule-8 baseline, technique #2 has no
speedup "against the baseline" by definition; its effect is the ratio to tiled shown above.

Unroll effects from the round-1 sanity sweep, not repeated in round 2: unroll 1 is −21 %,
unroll 16 +5 % and full unroll +8 % for blocks ≤ 256, while full unroll at block 1024 is −18 %,
which the design attributed in advance to instruction-cache pressure (`docs/03-design.md` §6;
data in `results/2026-10-17_architect_sanity/`).

**Literature.** Wave quantization as a first-order loss, fixed by refining the tail wave
[P:zhang2025hytis, transferable: GEMM tiles; full text not read]. Occupancy-aware launch
decisions on an RTX 4090 [P:shi2025libra, transferable: sparse kernels]. The step model and its
smoothing [P:zhang2026wavetune, direct for the wave model; GEMM-like kernels, no Ada]. Two of the
three are Chinese-led (`references/R3-evidence-map.md` §2).

**Limitations.** No block size rescues small N on its own: at N = 4,096 the best one
block per SM still leaves 48 SMs idle. The round-2 sweep and the timelines ran in the same
session but on different binaries (production vs twin).

## 4.4 Technique #3: several threads per body (j-split and reduction)

**Imbalance targeted.** Level 1, mainly starvation. Splitting each body's sum over q threads
multiplies the number of blocks by q without changing the per-SM rate, so small problems can
occupy all 80 SMs (`docs/03-design.md` §4.1).

**Mapping onto sm_89.** `allpairs_multi` uses a 2-D block of P bodies × q threads, with the q
threads of a body in consecutive lanes of one warp. Each thread sums the j-slice j ≡ s (mod q).
The q partial sums are combined by a register-only butterfly of `__shfl_xor_sync`, log₂ q steps
per component, with no shared memory and no extra barrier (`docs/03-design.md` §4.1–4.2). A
register prefetch issues the global load for the next tile before the current one is consumed.
All configurations up to block 512 use 40 registers and run at 100 % occupancy, except b64 q32
(44 registers, 20 blocks per SM, 83 %) (`docs/03-design.md` §4.3;
`results/2026-10-17_architect_sanity/summary.md`). The cost of the lane-major layout is shared
memory: at q = 32 a warp reads 32 different tile entries per step, which the design estimates at
91 % of the shared-memory port [D] (`docs/03-design.md` §4.2).

**Measured effect against the best-tuned baseline** (`results/2026-10-17_round2/summary.md` §3;
D-013):

| N | Baseline: tuned b128 full (G/s) | Best multi (G/s) | Speedup vs baseline |
|---|---|---|---|
| 1,000 | 75 | 180 (b256 q16) | **2.40×** |
| 4,096 | 348 | 894 (b512 q8) | **2.56×** |
| 10,000 | 850 | 1,188 (b1024 q8) | **1.40×** |
| 65,536 | 1,244 | 1,266 (b128 q2) | **1.02×** |

The gain is large exactly where starvation was measured (§4.2) and vanishes once every SM is
busy.

**Separating q from prefetch** (critic review 1, #4). Because multi also adds a register
prefetch, part of its gain could be the prefetch rather than the split. Round 2 measured both
with a `--prefetch` switch (`results/2026-10-17_round2/summary.md` §2):
- At N = 4,096 the 2.56× splits as **q × 2.40 · prefetch × 1.067**.
- The prefetch is worth +16.1 %, +10.1 % and +7.7 % at q = 1 for blocks 64, 128 and 256 (2, 4
  and 8 warps per SM), +0.5 % at block 512, and 0.0 % at N = 65,536.
- Prefetch therefore matters only when an SM holds 2–4 warps, which is when latency is not
  hidden by other warps.

**Timelines** (`results/2026-10-17_round2/summary.md` §1, Gantt plots in
`results/2026-10-17_round2/timeline/plots/`):

| N | Config | Idle SMs | SM efficiency | Tail |
|---|---|---|---|---|
| 1,000 | tiled b256 → multi b512 q32 | 76 → 17 | 0.050 → 0.723 | 0.4 % → 6.1 % |
| 4,096 | tiled b256 → multi b512 q8 | **64 → 16** | 0.199 → 0.786 | 0.6 % → 1.4 % |
| 10,000 | tiled b256 → multi b1024 q8 | 40 → 1 | 0.497 → 0.970 | 0.5 % → 1.5 % |
| 65,536 | tiled b256 → multi b256 q4 | 0 → 0 | 0.812 → 0.981 | 23.5 % → 0.1 % |

At N = 65,536 the mechanism is different: with 1,024 blocks for 480 resident slots, the hardware
refills SMs dynamically and the tail almost disappears.

**Accuracy side effect.** Splitting the sum changes the summation order. Multi passes all D-009
gates in all 4,000 sweep runs and is *more* accurate than the naive kernel against an FP64
reference, with error falling roughly as 1/q (`results/2026-10-17_round2/summary.md` §4;
`docs/reviews/2026-10-17-critic-1.md`, spot checks).

**Choosing q.** The bench's automatic q (cost model v1, D-008) searches q at a fixed block size.
The joint (block, q) optimum beats it by up to 12 % at N = 4,096 (D-010). Auto-q is therefore a
convenience, not the result to quote.

**Literature.** Splitting the partial wave into finer pieces to recover idle SMs
[P:zhang2025hytis, transferable, Chinese-led]. Splitting only heavy items above a threshold, with
the pieces combined by atomics [P:shi2025libra, transferable, Chinese-led]. Our shuffle
reduction avoids the atomic cost and is deterministic (`references/R3-evidence-map.md` §10.5).
Count-based static mapping of work items to blocks [P:li2025staticbatching, transferable: MoE
inference]. The original small-N idea is q threads per body in the tiled kernel
[P:nyland2007nbody].

**Limitations.**
- Unroll is still confounded: multi uses unroll 8, the baseline full unroll
  (`results/2026-10-17_round2/summary.md` §2).
- The split costs shared-memory bandwidth at large q; whether q = 32 is LDS- or L2-bound needs
  Nsight Compute (`docs/03-design.md` §7).
- The baseline was the same in every N of this session, but rule 8 makes the baseline
  session-specific: a different session can give slightly different values (348 G/s here,
  349 G/s in the round-1 sanity run, `docs/03-design.md` §6).

---

## 4.5 Techniques #4 and #5: tree traversal, naive vs space-filling-curve order

**Why a tree.** A Barnes-Hut tree replaces distant groups of bodies by their centre of mass, so
the work drops from O(N²) to O(N log N) [P:barnes1986hierarchical]. The work per body now
depends on where the body sits, and the walk is data-dependent control flow. Tree traversal is
known to map poorly onto ordinary GPU cores [P:nagarajan2025rtbarneshut], and divergent paths
serialize inside a warp [P:dale2026warpdivergence, T2: single author, no Ada]. This is where
levels 2 and 3 of §4.0 appear.

**Implementation** (D-015, D-016; `docs/03-design.md` §10–12). The tree is built and traversed
entirely on the GPU: 63-bit Morton keys sorted with CUB, a binary radix tree, bottom-up
monopoles, the bmax opening criterion, leaf buckets of 16 bodies and traversal block 128. The
traversal is one thread per body with a private 64-entry stack, 40 registers, 12 blocks per SM
and 100 % occupancy (`docs/03-design.md` §12.2). The build costs 12.6 %, 4.5 % and 1.6 % of the
sorted step at N = 65,536, 262,144 and 1M (D-015). θ here is a *bmax-θ*, roughly 0.87 times the
classic θ (D-015); critic review 2 (#10) notes that 0.87 is a best case, not an exact
conversion.

**The one difference between #4 and #5.** `bh_naive` (#4), `bh_sorted` (#5) and `bh_hilbert`
(#5 sub-variant) run *one* kernel on *one* tree. Only the `order[]` array that maps threads to
bodies differs: identity, the Morton permutation, or a Hilbert permutation. The accelerations
are therefore bitwise identical (D-016). In #4 a warp's 32 bodies are a random sample of the
system; in #5 they are spatial neighbours. A θ = 0 control, where every body walks the identical
path, gives #4 = #5 within 0.3 %, so the gain at θ = 0.5 is caused by the order alone (D-016).
The critic found no bias between #4 and #5 (`docs/reviews/2026-10-17-critic-2.md`, "Checked and
sound").

**Imbalance targeted by #5.** Levels 2 and 3: unequal loop lengths inside a warp and divergent
memory accesses (`docs/03-design.md` §12.1).

### 4.5.1 Measured effect

All at θ = 0.5, leaf 16, block 128, medians of 11 runs (`results/2026-10-17_tree_round1/summary.md`
§1–§2):

| IC | N | #4 traverse (ms) | #5 traverse (ms) | #5 / #4 | #4 / #5 trav G int/s |
|---|---|---|---|---|---|
| Plummer | 65,536 | 2.862 | 1.757 | 1.63 | 21.0 / 34.1 |
| Plummer | 262,144 | 16.223 | 8.475 | 1.91 | 15.8 / 30.2 |
| Plummer | 1,048,576 | 73.046 | 38.552 | 1.89 | 15.3 / 28.9 |
| collision | 65,536 / 262,144 / 1M | | | 1.70 / 2.11 / 1.98 | |
| core | 65,536 / 262,144 / 1M | | | 1.64 / 1.91 / 1.91 | |

The main session re-ran the N = 262,144 Plummer case and reproduced it (16.222 / 8.523 ms)
(`results/2026-10-17_tree_round1/summary.md` §7).

**Result: sorting the threads in Morton order speeds up the traversal by about 1.6–2.1× at
θ = 0.5.** The ratio is flat above N = 262,144 and nearly independent of the initial condition.
It depends on θ: 1.48 / 1.91 / 2.14 / 2.32 at θ = 0.3 / 0.5 / 0.7 / 1.0
(`docs/reviews/2026-10-17-critic-2.md` #3).

**What this result is not.** Burtscher and Pingali describe sorting as giving roughly an order
of magnitude [P:burtscher2011barneshut]. An earlier draft of our summary said that "Burtscher's
10× fails on Ada". That claim is withdrawn (`results/2026-10-17_tree_round1/summary.md` §2;
`docs/reviews/2026-10-17-critic-2.md` #3). The 2011 statement is an unquantified sentence about
a different code (an octree with warp-vote traversal), on a different GPU (a Quadro FX 5800,
without coalesced same-address reads) and at a different opening parameter. Our measurement is
of per-thread walks at bmax-θ 0.5 on sm_89, and it is not a like-for-like test of theirs. No
statement about Ada versus older GPUs is made.

**Rule 8 for the tree: best against best (tree round 2).** The 1.6–2.1× above is the fixed
protocol configuration (leaf 16, block 128), which critic review 2 (#8) flagged as untuned. Tree
round 2 swept leaf × block for each variant (set C, Plummer, θ = 0.5; D-025). The canonical values
are from the clean replication at 0 % background load (D-033; set C in
`results/2026-10-18_clean_rerun/tree2/merged/analysis_output.txt`):
- **#5 at 1M: best is leaf 32, block 256, at 34.45 ms**, 12 % faster than the protocol
  configuration leaf 16, block 128 (38.75 ms). At N = 262,144 the best is leaf 16, block 256
  (7.64 ms), with leaf 32, block 256 at 7.72 ms, within 1.1 % (D-033). The predicted leaf range
  8–16 therefore holds at 262,144 but not at 1M.
- **#4's best:** leaf 32, block 256 at 262,144 (15.32 ms); leaf 32, block 128 at 1M (71.11 ms).
- Best against best, **#5 / #4 = 2.01× at N = 262,144 and 2.06× at 1M**
  (`results/2026-10-18_clean_rerun/`). Under the 20–22 % load of the original round-2 session these
  had been 2.06× and 2.18× (`results/2026-10-17_tree_round2/`; D-025).
- The pre-registered leaf and block optima still miss in the clean run (L1, L2, L4, L6, L7);
  L3 (#7's best leaf) flips from MISS to HIT (`results/2026-10-18_clean_rerun/scorecard.csv`).

**What limits #5: L1 capacity, not occupancy (tree round 2, set D).** Round 1 had reported the
traversal rate as barely rising with occupancy, but that observation was confounded with N
(critic review 2, #6). Round 2 swept occupancy at fixed N = 262,144 by padding shared memory,
which also pins the L1 carve-out near its minimum (`docs/03-design.md` §17.5). With the small L1,
#5's rate is **flat from 2 to 11 blocks per SM (19.9–20.7 G/s)**; the unpadded kernel, with its
full L1, does **29.0 G/s (×1.51)** (`results/2026-10-17_tree_round2/summary.md` §2, finding 4;
D-026). #5 is therefore bound by the L1 capacity available to its scattered accesses, not by
occupancy. Caveat: the padded "12 blocks" cell actually ran at 11 blocks per SM, because shared
memory is allocated in 256 B units (D-026). **Load label:** set D was not part of the clean
replication, so these values were measured under 20–22 % background GPU load
(`results/2026-10-17_tree_round2/`; D-033).

### 4.5.2 Where the imbalance went

(`results/2026-10-17_tree_round1/summary.md` §1–§2; `docs/03-design.md` §16.3)

- **Inside warps, sorting helps.** Per-warp max/mean drops from 1.17–1.62 (#4; the core IC is
  worst) to 1.05–1.09 (#5). Pop-site SIMT efficiency rises from 0.55–0.82 to 0.92–0.98.
- **But the leaf loop still serializes.** At interaction sites only 3–7 of 32 lanes are active
  in both variants (SIMT efficiency 0.09–0.13 for #4, 0.13–0.22 for #5), because the few lanes
  that reach a leaf bucket run it while the others wait. Sorting reduces divergence but does not
  remove this leaf-bucket serialization (summary §2, reworded after critic review 2, #12).
- **Between warps, sorting makes it worse.** Warp-cost max/mean rises from 1.06–1.12 (#4) to
  1.15–1.60 (#5): blocks in the dense core now cost more than blocks in the halo.
- **At SM level, the hardware absorbs it when there are enough waves.** In the single-wave
  launch (N = 65,536, 512 blocks for 960 slots), #5 on the core IC has SM efficiency 0.810 and a
  17.7 % tail, against 0.945 and 3.5 % for #4. At 1M (8.5 waves), both reach 0.987–0.995.
- **Coherent speed is far away.** In the θ = 0 control, all lanes walk the same path: 552 G
  interactions/s at SIMT efficiency 1.000. At θ = 0.5 the same kernel does 14–43 G/s, about 1/20
  of its coherent speed (summary §2, finding 3). Round 2 measured this gap properly at θ = 0.5
  (§4.5.5): intra-warp incoherence costs #5 a factor of 4.4–9.5.
- **Hilbert ≈ Morton.** Hilbert lowers per-warp max/mean by only 0.2–0.6 %, its traversal is
  1.5–4.4 % slower in 8 of 9 cells, and it costs an extra 0.13–0.24 ms order phase. Morton is
  free (the build already produces it), so it is the #5 default (D-018). The compactness argument
  for Hilbert [P:thaler2026sfcneighbor] does not translate into a traversal gain here. No
  2025–26 source compares the two curves for GPU traversal (`references/R3-evidence-map.md` §5),
  so this measurement is new, within the caveat that the Hilbert selftest covers only the top
  4 of 21 bits per axis (`docs/reviews/2026-10-17-critic-2.md` #15).

### 4.5.3 Crossover against the best all-pairs kernel

The tree total (build + traversal) is compared with the best all-pairs configuration measured in
the same session at the same N (D-020; `results/2026-10-17_tree_round1/summary.md` §3):

| N | Best all-pairs (ms) | `bh_naive` (ms) | `bh_sorted` (ms) | AP / sorted |
|---|---|---|---|---|
| 32,768 | 0.890 | 1.570 | 1.074 | 0.83 |
| 49,152 | 1.912 | 2.287 | 1.478 | 1.29 |
| 65,536 | 3.398 | 3.071 | 1.964 | 1.73 |
| 262,144 | 50.98 | 16.545 | 8.918 | 5.72 |
| 1,048,576 | 822 [D] (4 × T(524,288), not run: TDR) | 73.66 | 39.17 | 21.0 [D] |

The crossover is at **N ≈ 39,700 for the sorted tree** and ≈ 59,600 for the naive one
(D-020). Two conditions attach to it (`docs/reviews/2026-10-17-critic-2.md` #9):
1. It holds **for θ = 0.5 only**; a different accuracy target moves it.
2. It compares a **power-capped** all-pairs kernel (clock down to 2,250 MHz) with a tree
   traversal that does **not** reach the cap (2.75 GHz) (`results/2026-10-17_tree_round1/summary.md`,
   environment). This favours the tree. A run at 320 W, or per-run clocks, is needed to remove
   the bias, and the 320 W decision is Toncho's.

The 1M figures are derived, not measured; a chunked 1M all-pairs measurement is possible and is
open (critic review 2, #9). Tree round 2 moves the crossover further down with #7 (§4.7).

### 4.5.4 Accuracy

All 47 gated tree runs at θ > 0 pass the TG1/TG2 gates with zero stack overflows. At θ = 0.5 the
rms relative error is 6.5 × 10⁻³ to 1.0 × 10⁻² against a gate of 0.03
(`results/2026-10-17_tree_round1/summary.md` §6; corrected from 49 runs, critic review 2, #13).
In round 1 the exact selftest did not exercise the θ > 0 force path (critic review 2, #7). Round 2
added an exact host check for θ > 0: every node's mass, centre of mass, box and bmax are recomputed
in FP64, and the same tree is walked on the host with the device's FP32 opening test. Per-body
counts must match the device exactly, and the force error must stay within a rounding bound; the
worst case seen is 0.27 of the bound (`docs/03-design.md` §17.6;
`results/2026-10-17_tree_round2/summary.md` §5). A 0.1 % mass error injected into the
monopole-accept branch is caught only by this new check; the statistical gates TG1/TG2 pass it
(`docs/03-design.md` §17.6).

### 4.5.5 The loss decomposition: a hypothesis generator, not a measurement

To decide what to build next, the architect factorised the sorted traversal's shortfall against
the θ = 0 control into idle lanes, a memory-incoherence term and SM-level loss
(`results/2026-10-17_tree_round1/summary.md` §5; `docs/03-design.md` §16.4). The suggested
reading was that 85–99 % of the loss is inside warps and only 1–2 % between blocks at 1M, which
is why #7 (warp-cooperative traversal) was prioritised over #6 (D-019).

**This decomposition does not measure the split** (`docs/reviews/2026-10-17-critic-2.md` #1):
- The memory term is a *residual*. It absorbs everything not modelled: the instruction-mix
  difference between θ = 0 and θ = 0.5, the cache footprint (the control ran at 65,536, the
  target at 1M), the local-memory stack, and the latency of dependent pop loads, which #7 would
  not remove.
- The headline fractions and the "8.4× ceiling" move with an assumed cost constant c_V
  (F = 0.089 / 0.119 / 0.166 at c_V = 10 / 20 / 40).
- Only the SM-level term (from the timelines) is independent of c_V.

It was used only to choose which experiments to run, and it is now superseded (D-024).

**The measured replacement: a coherent control at θ = 0.5 (tree round 2).** `bh_coherent` runs
#5's own kernel and instruction mix, on the same tree at the same θ, but with all 32 lanes of a
warp walking the body of the warp's first slot (`order[t] = perm[t & ~31]`). It therefore has
perfect lane coherence and same-address loads, and nothing else changes (`docs/03-design.md`
§17.4). Its traversal time is the time #5's kernel needs for #5-like lane work when the warp is
coherent. Measured in the clean replication: **#5's kernel runs 4.4–9.5× faster when coherent**,
at 127–261 G lane interactions/s, so intra-warp incoherence (idle lanes, scattered loads and the
divergent stack together) costs #5 a factor of about 4–10 (items C1–C4 in
`results/2026-10-18_clean_rerun/scorecard.csv`). At 1M the measured ceiling is 9.0× (Plummer) and
9.5× (core) (`results/2026-10-18_clean_rerun/`); the round-1 residual estimate had been 8.4×
(`docs/03-design.md` §18.2). The control is one body per
warp, so it has no gates, and its warp-cost spread samples #5's lanes rather than #5's warp maxima
(`docs/03-design.md` §17.4).

### 4.5.6 Scorecard

The tree round was pre-registered (commit 581fc47) before any timing run
(`results/2026-10-17_tree_round1/summary.md`; timestamps confirmed in critic review 2). The
first summary reported a scorecard of 91 HIT / 32 MISS. **That count is disputed and is not
used here** (`docs/reviews/2026-10-17-critic-2.md` #4): no item-level table is committed, one
post-registration interpretation moved a goalpost (P-T1a is a miss), one item was split into two,
some HITs restate values seen before registration, and P-T6c recomputes to a miss. The
qualitative pattern stands: the relative structure was predicted well (IC ratios, imbalance
directions, error ∝ θ^2.13 and work ∝ θ^−2.52, the crossover, the timeline structure), while
absolute speed was 2–4× slower than predicted because the model left out memory-side incoherence
(`results/2026-10-17_tree_round1/summary.md` §4). Even that rate-versus-occupancy observation
(16 → 34 → 29 G/s at 4 / 26 / 48 warps per SM) is confounded with N (critic review 2, #6);
round 2 resolved it (§4.5.1: L1 capacity, not occupancy).

**Limitations of #4/#5, in one list.** Still open: the θ dependence of every ratio; the disputed
round-1 scorecard; the power-cap bias in the crossover; the Hilbert selftest covering only the
top bits. Closed in round 2: the untuned configuration (best against best now quoted), the
residual-based decomposition (replaced by the coherent control), and the θ > 0 force path (now
checked exactly).

---

## 4.6 Technique #6: persistent threads and software work queues

**Imbalance targeted.** Levels 1 and 2 of §4.0, and, in its lane-granular form, the loop-level
imbalance inside a warp. Instead of the hardware block scheduler handing out a fixed grid, a
persistent kernel keeps a fixed set of resident warps that claim work from a queue until it is
empty.

**Motivation from round 1.** Between-block imbalance was real only in single-wave clustered
launches: sorted #5 on the core IC at N = 65,536 had a 17.7 % tail and SM efficiency 0.810
(`results/2026-10-17_tree_round1/summary.md` §2). Inside warps, per-warp max/mean was 1.05–1.09
for #5 and up to 1.62 for #4 (summary §1). D-019 first pre-registered #6 as small (≤ 3 %), from a
block-granular bound. Critic review 2 (#2) pointed out that this bound does not apply to a lane-
or warp-granular queue, and that pre-registering ≤ 3 % would make a graded technique a strawman.
Round 2 therefore built the lane-granular arm as well (`docs/03-design.md` §17.3).

### 4.6.1 The arms and how they map onto sm_89

All arms run the same tree with the same per-body arithmetic as their sibling, so their
accelerations are **bitwise equal** to the sibling's, with equal per-slot pop and interaction
counts. This was checked on every bench run (36/36 in set A, 24/24 exploratory)
(`results/2026-10-17_tree_round2/summary.md` §1; `docs/03-design.md` §17.3). The persistent grid
is capped at the blocks that actually have work. Without the cap, below N = 122,880 the excess
warps race for the few units and the winners cluster on a few SMs (`docs/03-design.md` §17.3).
All arms use 38–40 registers and 12 blocks per SM at block 128 [M] (`docs/03-design.md` §17.3).

| Arm | Registry name | How work is claimed | Targets | Literature (all transferable) |
|---|---|---|---|---|
| Warp-granular queue | `bh_persistent` (on #5), `bh_persistent_warp` (on #7) | lane 0 claims the next 32-slot unit with `atomicAdd` on a global counter | block-internal wait (3 warps done, block slot held) and the tail | [P:huang2026weave]: the same claiming mechanism |
| Lane-granular refill | `bh_persistent_lane` (on #5), `bh_persistent_lane_naive` (on #4) | when a lane's stack empties, the warp refills idle lanes from a warp-owned batch of 32 slots, claimed one batch ahead | loop-level loss (lanes done, warp not) | warp-level stealing analogue [P:chen2026gmatch]; persistent traversal [P:maeda2026gtap] |
| Static partition | `bh_persistent_static` | contiguous unit ranges of equal measured cost per warp, from the previous launch's per-unit clock64 costs and a prefix sum | between-warp imbalance, without runtime atomics | [P:xia2025voltrix]; [P:li2025staticbatching] |

The lane-granular arm has two built-in limits (`docs/03-design.md` §17.3):
- It can only rebalance when a lane gets more than one body, i.e. N ≫ 32 × 3,840 resident warps
  = 122,880. Bodies per lane are 0.53, 2.1 and 8.5 at N = 65,536, 262,144 and 1M
  (`docs/predictions/2026-10-17-tree-round2.md` §5).
- After a refill, a warp's lanes hold bodies from two Morton batches, so memory coherence degrades
  towards #4's.

The static arm is given its best case: every timed launch uses the costs measured on the previous
identical launch (`docs/03-design.md` §17.3).

### 4.6.2 What was predicted

The predictions were committed before the runs (c6a12cc;
`results/2026-10-17_tree_round2/summary.md`). For the warp-granular queue on #5 they gave a ratio
of about 1.00 (range 0.96–1.05). For lane refill they gave 0.90–0.97 on #5 and up to 1.28 (range up
to 1.55) on #4 at 1M, where #4's pop efficiency leaves the most to recover
(`docs/predictions/2026-10-17-tree-round2.md` §5, items P2-P1 to P2-P8).

### 4.6.3 Measured effect: every arm is slower than the hardware block scheduler

Speed relative to the sibling on the hardware block scheduler, traversal phase, θ = 0.5, leaf 16,
block 128, Plummer and core ICs at N = 65,536, 262,144 and 1M. Values are from the clean
replication at 0 % background load, which is canonical (D-033; `results/2026-10-18_clean_rerun/`,
summary and `scorecard.csv` items P1–P8):

| Arm | Speed vs sibling | Balance metric it targets, and what it did |
|---|---|---|
| Warp-granular queue on #5 | **0.88–0.93×** | single-wave SM efficiency (core, 65,536): 0.827 → 0.921, tail 14.7 % → 6.4 %; warp-slot fill at 1M: 0.946 → 0.976 (Plummer), 0.948 → 0.980 (core) |
| Warp-granular queue on #7 | **0.90–0.95×** | — |
| Lane-granular refill on #5 | **0.63–0.75×** | — |
| Lane-granular refill on #4 | **0.70–0.89×** | #4 core pop SIMT efficiency at 1M: 0.61 → 0.91 |
| Static cost-balanced partition | **0.85–0.97×** | — |

(Timeline metrics: `results/2026-10-18_clean_rerun/summary.md`, timelines.) At 1M, for example,
#5 takes 38.60 ms, the warp queue 41.54 ms, lane refill 55.51 ms and the static partition 45.53 ms
(Plummer; set A in `results/2026-10-18_clean_rerun/tree2/merged/analysis_output.txt`). Interleaved
repeats at 1M agree to within 0.8 % (set A2, same file).

**Result: a measured negative.** Each arm improves the balance metric it was designed to improve,
and each is slower in every cell. All #6 performance items of the scorecard are misses (P1–P9,
P11), in the original session and again in the clean replication
(`results/2026-10-17_tree_round2/scorecard.csv`; `results/2026-10-18_clean_rerun/scorecard.csv`).
The lane-refill arm is still 37 % slower at N = 65,536 (Plummer), where, with 0.53 bodies per lane,
it performs no refills at all (`results/2026-10-18_clean_rerun/scorecard.csv`, P3). The cost is therefore in the arm's
structure, not in the rebalancing it performs.

### 4.6.4 Why: exploratory diagnosis (not pre-registered)

Two exploratory variants, added after the main set and checked to leave the SASS of every measured
kernel unchanged, point to two different mechanisms (`results/2026-10-17_tree_round2/explore/`;
summary §2 and §7; `docs/03-design.md` §18.3). They were not part of the clean replication, so
their numbers carry the 20–22 % load label (D-033):

1. **The warp-granular queue loses L1 locality.** The hardware block scheduler hands an SM a block
   of four Morton-adjacent warps, which share L1 lines. A warp-granular queue scatters them over
   SMs. A *block-granular* queue that claims four Morton-adjacent units at a time recovers most of
   the loss: 0.94–0.98× against 0.86–0.93× for the warp-granular queue in the same exploratory
   runs. This agrees with §4.5.1: #5 is bound by L1 capacity, so anything that dilutes L1 sharing
   costs directly.
2. **Lane refill pays for its loop structure.** Checking for idle lanes only every 8 pops instead
   of every pop barely helps (0.65× against 0.63× at N = 65,536). The likely cause is the
   warp-collective calls in every iteration of the inner loop, which force reconvergence at each
   pop. **This explanation has not been verified in the SASS**
   (`docs/03-design.md` §18.6; `TODO(measure: compare the SASS of bh_persistent_lane with
   bhTraverseKernel to confirm the per-pop reconvergence)`).

**What this means on sm_89** (D-023). The hardware block scheduler is already a dynamic queue at
4-warp granularity that preserves locality, and the SM-level loss left at N ≥ 262,144 is only
1–2 %. A software queue has to claim blocks of adjacent units before it can win anything. The
D-019 bound of ≤ 3 % turned out right in size for the block-granular gain but wrong in sign for the
warp- and lane-granular arms (D-023). The cited papers describe the mechanisms; none of them
predicts a gain for this workload, and D-023 rules out citing them as if they did.

**Not a strawman.** The lane-granular arm that critic review 2 asked for was built, verified
bitwise against its sibling and measured, so the negative result covers the arm that could have
attacked the loop-level loss (D-023).

### 4.6.5 Limitations

- **Replicated without background load.** The original round-2 session ran with 20–22 % idle GPU
  utilisation, and the #6 verdicts hinged on 1–5 % differences. The pre-registered clean
  replication at 0 % load re-measured every #6 cell with A/B/B/A interleaving: **no #6 verdict
  changed**, and every arm is still slower than its sibling in every cell (§4.6.3; §4.1.1;
  `results/2026-10-18_clean_rerun/summary.md`; D-033). Attributing #6's loss to the background load
  is therefore ruled out (D-033).
- The diagnosis in §4.6.4 is exploratory and was not pre-registered.
- The block-granular queue was tested only on #5. Whether it can at least break even on the #7
  walk, whose single-wave tail is now its largest loss (§4.7.4), is open (`docs/03-design.md`
  §18.6; `TODO(results: block-granular queue on bh_warp)`).

## 4.7 Technique #7: warp-cooperative traversal (the union walk)

**Imbalance targeted.** Level 3: idle lanes and divergent memory accesses inside a warp. §4.5
showed that sorting leaves only 3–7 of 32 lanes active at interaction sites, and that incoherence
costs #5 a factor of 5–10 (§4.5.5).

### 4.7.1 How it maps onto sm_89

`bh_warp` treats one warp as one work unit of 32 consecutive Morton slots (`docs/03-design.md`
§17.2). The warp pops one node at a time; every lane evaluates its own opening test, and the node
is opened if **any** lane opens it (`__any_sync`). The warp therefore walks the union of its lanes'
paths, which is the warp-vote idea of [P:burtscher2011barneshut] (background). Consequences:
- **Control flow is warp-uniform.** Every pop and every interaction runs on all 32 lanes, and every
  node or body load is a same-address (broadcast) load: one L1 wavefront instead of up to 32.
- **The stack lives in registers.** The 64-entry stack is spread over two registers per lane (entry
  k in lane k mod 32), so a push is a predicated move and a pop is one `__shfl_sync`. There is no
  shared memory and no local memory: a 0 B stack frame, against #5's 256 B local stack.
- **Resources** [M]: 40 registers, 0 spills, 12 blocks per SM at block 128, 100 % occupancy
  (`docs/03-design.md` §17.2).
- **The price is extra work.** A lane whose own test would accept a node that the warp opens gets
  the node's children instead, so each lane does at least as many interactions and pops as in #5.
  This is asserted per slot on every run (`docs/03-design.md` §17.2).

D-022 fixes how the technique is named: warp-cooperative traversal, a union walk with a warp
vote, **not work stealing**. It does not remove work; it adds work and removes incoherence.

### 4.7.2 Measured effect

**Fixed protocol configuration** (leaf 16, block 128, θ = 0.5; set A of the clean replication,
canonical per D-033; `results/2026-10-18_clean_rerun/tree2/merged/analysis_output.txt`, set A, and
`scorecard.csv` items W8–W9):

| IC | N | #5 traversal (ms) | #7 traversal (ms) | #7 speedup |
|---|---|---|---|---|
| Plummer | 65,536 | 1.798 | 0.574 | 3.1× |
| Plummer | 262,144 | 8.408 | 1.609 | 5.2× |
| Plummer | 1,048,576 | 38.60 | 6.005 | 6.4× |
| core | 65,536 | 2.433 | 0.729 | 3.3× |
| core | 262,144 | 11.19 | 1.756 | 6.4× |
| core | 1,048,576 | 48.01 | 6.770 | 7.1× |

The traversal is **3.1–7.1× faster than #5**, and the gain grows with N. The full step (build +
traversal) is 2.6–6.6× faster [D, from the set-A totals in the same file]. Under the 20–22 % load of
the original session the range had been 3.1–7.3× (`results/2026-10-17_tree_round2/summary.md` §2).

**Best against best** (rule 8; set C, Plummer, θ = 0.5; `results/2026-10-18_clean_rerun/`):
**5.8× at 1M** (5.76×: #5 at leaf 32, block 256, 34.45 ms; #7 at leaf 16, block 64, 5.99 ms) and
**4.8× at N = 262,144**. For the total step, 5.3× and 4.3× (`results/2026-10-18_clean_rerun/summary.md`,
item L5). The main session re-ran the 1M pair afterwards at 0 % load and got 34.31 ms and 5.99 ms,
i.e. 5.73× (same summary, independent re-check). #7 is flat in leaf size: leaf 8 and leaf 16 are
within 0.6 % at 1M (D-033). Under load the figures had been 5.9× and 5.2× (D-025).

**Real interaction rate.** At 1M (Plummer), #7 performs 1,837 interactions and 2,036 pops per body
at **SIMT efficiency 1.000** at both interaction and pop sites, at **321 G interactions/s**,
against 1,063 interactions, 1,612 pops, SIMT efficiency 0.201 / 0.976 and 28.8 G/s for #5
(`results/2026-10-18_clean_rerun/tree2/merged/analysis_output.txt`, set A; the work counts are
deterministic and identical to round 2). Across N, #7's rate is 194 / 285 / 321 G/s (Plummer), about
11 times #5's 29–33 G/s (same file; scorecard item W11).

**The union overhead.** #7 does **1.64–1.85×** the interactions and **1.22–1.31×** the pops of #5,
flat in N, as pre-registered (W1–W5 HIT). These are deterministic counts and reproduce exactly
(`results/2026-10-17_tree_round2/scorecard.csv`; `results/2026-10-18_clean_rerun/scorecard.csv`).

**Against the coherent control.** #7 runs 1.31–1.65× slower than the coherent control, which is
about its extra work. It closes **70–87 % of the gap** between #5 and the control, measured in log
terms (`results/2026-10-18_clean_rerun/tree2/merged/analysis_output.txt`, derived values T7/TC and
f; D-024). Most of what remains is the union's extra work, not residual incoherence.

**Accuracy.** Because every lane interacts with at least the cells it would have used alone, #7 is
**more accurate than #5**: rms relative error ×0.66–0.85 and p99 ×0.60–0.91 of #5's at the same θ;
the TG3 gate passes 60/60 (`results/2026-10-17_tree_round2/summary.md` §1, §5; deterministic,
reproduced in the clean run). At N = 262,144, θ = 0.5 the rms error is 4.59 × 10⁻³ against
6.55 × 10⁻³ for #5. TG3 is an aggregate gate: an individual body's error is not guaranteed to drop
(`docs/03-design.md` §17.2).

**What limits #7: latency, not L1.** In the occupancy sweep (set D, N = 262,144) the control and #7
speed up ×1.75 and ×1.87 from 4 to 11 blocks per SM, and gain only ×1.30 and ×1.12 from the full
L1 (D-026). The coherent walk is bound by its dependent pop chain (shuffle, node load, test, vote,
branch), so its next optimisation is fewer pops per interaction, i.e. wider tree nodes, not more
occupancy (D-026). Set D was not re-run, so these numbers carry the 20–22 % load label (D-033).

### 4.7.3 #7 needs #5's ordering, and depends on θ

**Without the space-filling-curve order** (`bh_warp_naive`, the union walk on the original body
order), a warp's 32 bodies are scattered over the system and their union is large: **12.9–14.8×**
#4's interactions (deterministic). It is still 1.26–1.52× faster than #4, which shows how expensive
divergence is for #4, but it is **4.1–8.1× slower than `bh_warp`**
(`results/2026-10-18_clean_rerun/scorecard.csv`, items W16, W17). The scorecard item W16 misses: at
1M the naive union walk beats #4 by 1.51×, above the predicted ≤ 1.3× (same file). #5 and #7 are
therefore complements: sorting makes the union small, the union makes the warp coherent.

**θ dependence** (set F, N = 262,144; `results/2026-10-18_clean_rerun/tree2/merged/analysis_output.txt`,
set F):

| bmax-θ | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|
| Union overhead (interactions) | ×1.55 | ×1.79 | ×2.08 | ×2.11 |
| #7 speedup over #5 | 7.8× | 5.2× | 3.9× | 3.0× |

A larger θ accepts more cells, so lanes disagree more often and the union grows. At θ = 0, #7
equals #5 within about 2 % and passes D-009 (same file, set G).

**The interaction-list sub-variant** (`bh_warp_list`) buffers accepted cells and bucket bodies in a
per-warp shared-memory list and evaluates them in a dense loop. It is bitwise equal to `bh_warp`
but 3–8 % **slower** (W13 MISS; `results/2026-10-18_clean_rerun/scorecard.csv`): once the walk is
coherent, interleaving pops and interactions costs nothing, and the list adds shared-memory traffic
(`docs/03-design.md` §18.1; D-022). It stays registered as a measured negative.

### 4.7.4 Crossover and new bottleneck

`bh_warp` crosses the best all-pairs configuration of the same session at **N ≈ 28,700**, down from
≈ 39,700 for #5 (set E; `results/2026-10-18_clean_rerun/scorecard.csv`, item X1). Its total step is
4.4× faster than the best all-pairs at N = 65,536 and 27.3× at 262,144; at 1M the ratio is 124×,
against an extrapolated all-pairs time [D] (item X3, same file). The conditions of §4.5.3 apply:
the crossover holds for θ = 0.5 only, and per-run clocks for the two sides were not reported, so
the power-cap bias found in round 1 has not been re-checked. The clean session ran at 0 % idle load,
with a median power of 185 W and a maximum of 270 W under the 272 W cap
(`results/2026-10-18_clean_rerun/summary.md`, environment).

**The new bottleneck is between warps.** #7 makes the single-wave tail relatively worse: at
N = 65,536 (core) its SM efficiency is 0.734 with a 27.2 % tail, against 0.827 and 14.7 % for #5
(`results/2026-10-18_clean_rerun/summary.md`, timelines). Its warp costs spread 1.5–2.6× (max/mean)
over a launch about 3× shorter (`results/2026-10-17_tree_round2/summary.md` §4). At 1M the tail is
about 2 % (1.8–2.3 %, clean timelines in `tree2/merged/analysis_output.txt`). So at N = 65,536
between-warp imbalance is now #7's largest loss, which is exactly the case where a cheap,
locality-preserving block-granular queue (§4.6.4) might pay.

### 4.7.5 Literature

The warp vote over the union of paths is the classic GPU Barnes-Hut traversal
[P:burtscher2011barneshut] (background). A non-recursive tree walk on the GPU is in
[P:stucker2026jzfmm]. The 2025–26 scheduling papers [P:maeda2026gtap; P:chen2026gmatch] concern
work stealing, which #7 is not (D-022); they were used to set expectations for stealing as small,
and §4.6 found the software schedulers even below that. No 2025–26 source measures a warp-union walk
for gravity on Ada, so the coherent-control comparison is our own contribution.

### 4.7.6 Limitations

- **Replicated without background load.** The #7 cells were re-measured in the clean replication
  at 0 % load. The speedups fell slightly (by up to about 4 % per cell) and every #7 verdict stayed
  the same (`results/2026-10-18_clean_rerun/summary.md`, key cells; D-033). Coherent walks are not
  load-proof, though: in the IC session, #7 at 1M was inflated by 7–15 % under 30–37 % load (D-033).
- All ratios depend on θ (§4.7.3).
- The best-against-best speedups are for the Plummer IC only (set C).
- The 1M all-pairs comparison is extrapolated [D].

## 4.8 Technique #8: CPU + GPU split (planned)

**Imbalance targeted.** Between devices. The CPU is an Intel Core i5-13600K with 6 P-cores and
8 E-cores (`docs/01-hardware.md` §1). A static even split across its threads would be limited
by the slow E-cores, so the split itself needs balancing (`docs/01-hardware.md` §1).

**Motivation from round-1 data.** The tree build is never the bottleneck on the GPU: 12.6 %,
4.5 % and 1.6 % of the sorted step at N = 65,536, 262,144 and 1M (D-015). A CPU-side build
would therefore save little GPU time, and every step would add PCIe transfers through the single
copy engine (chapter 2, §2.4.4).

**Evidence.** The one direct 2026 gravity source, OpenGadget3 (Ragagnin et al.), rebuilds the tree
on the CPU and runs the short-range walk on the GPU (`references/R3-evidence-map.md` §8). It is not
yet in `references.bib`, so it is not cited here (`TODO(source needed: add Ragagnin et al. 2026
to references.bib with the full author list)`). Dynamic CPU–GPU work stealing gave +12.9 % over
GPU-only on a workstation [P:xu2025tascell, transferable]. Same-kernel co-execution gave 1.31×
over GPU-only [P:nozal2025coexecution, transferable, T2: full text not read]. A profiling-based
CPU/GPU split that minimises the slower side's time is an analogy from MoE inference
[P:zhong2025hybrimoe, Chinese-led]. On the other side, a production code argues for keeping the
solvers GPU-resident [P:frontiere2025crkhacc]. No 2025–26 source covers P-core/E-core hybrid CPUs
in a CPU+GPU split (`references/R3-evidence-map.md` §8).

**Planned measurement.** A phase split (tree on the CPU, forces on the GPU) against the
GPU-resident #5 of the same session, with the CPU work split statically and dynamically across P-
and E-cores. Not run in tree round 2 (`TODO(results: #8, future round)`).

---

## 4.9 Summary

| # | Technique | Imbalance targeted | Measured effect (rule 8 baseline unless noted) | Status |
|---|---|---|---|---|
| — | Measurement method | — | per-block / per-warp-unit SM timelines + per-warp counters; twin differs from production by −3.4…+6.2 %, so timelines for structure only; load-sensitive items replicated at 0 % load (§4.1.1) | done (D-011, D-033) |
| 1 | Static tiling | none (reference) | 76 / 64 / 40 idle SMs at N = 1,000 / 4,096 / 10,000; 23.5 % tail at 65,536 | done |
| 2 | Block / tile / unroll | SM level: starvation, wave tail | is the baseline: 1.14–1.54× over tiled b256 [D]; step model exact for single waves, per-block cost term wrong (D-012) | done |
| 3 | q threads per body | SM level: starvation | **2.40× / 2.56× / 1.40× / 1.02×** at N = 1,000 / 4,096 / 10,000 / 65,536; at 4,096: q × 2.40 · prefetch × 1.067; idle SMs 64 → 16 | done (unroll still confounded) |
| 4 | Naive Barnes-Hut | (imbalance baseline for trees) | per-warp max/mean 1.17–1.62; interaction-site SIMT eff. 0.09–0.13 | done |
| 5 | Morton / Hilbert order | inside warps | **2.01× / 2.06× vs #4** best vs best at N = 262,144 / 1M (θ = 0.5); 1.6–2.1× at the fixed protocol config; Hilbert ≈ Morton; L1-capacity-bound; crossover N ≈ 39,700 | done |
| 6 | Persistent threads + queues (warp, lane, static, on #7) | between warps / blocks; loop level (lane arm) | **slower in every cell: 0.63–0.97×** of the hardware scheduler, although each arm improves its balance metric; mechanism: lost L1 locality (warp queue), loop structure (lane refill, not verified in SASS) | measured negative; replicated clean 2026-10-18 |
| 7 | Warp-cooperative union walk | inside warps (idle lanes, divergent loads) | **5.8× / 4.8× vs #5** best vs best at 1M / 262,144 (θ = 0.5); 3.1–7.1× at the fixed config; SIMT eff. 1.000; ×1.6–1.9 interactions; more accurate (TG3 60/60); closes 70–87 % of the coherent gap; needs #5's order; crossover N ≈ 28,700 | done; replicated clean 2026-10-18 |
| 8 | CPU + GPU split | between devices, P/E cores | `TODO(results: #8, future round)` | planned |

**Scorecards.** All-pairs round 2: 18 HIT / 7 MISS (`results/2026-10-17_round2/summary.md` §7).
Tree round 1: disputed, not used (§4.5.6). Tree round 2: **39 HIT / 22 MISS of 61 items**, from the
committed item table `results/2026-10-17_tree_round2/scorecard.csv`; the #7 model held, all #6
performance items missed (`results/2026-10-17_tree_round2/summary.md` §3). **Clean replication**
(2026-10-18, 0 % load, §4.1.1): 36 of 59 primary items (tree round 2 and the IC round) HIT against
the original ranges, against 39 before; no #6 or #7 conclusion changed; the proposed load mechanism
was falsified, and the inflation tracked kernel length (`results/2026-10-18_clean_rerun/summary.md`;
D-033). Tree-rate numbers in rows 5–7 are the clean, canonical values.

Four conclusions, each limited to what was measured:
1. **For all-pairs, the imbalance is entirely at SM level**, and it is fixed by creating the right
   number of work items: smaller blocks (#2) or several threads per body (#3). The gain is 2.4–2.6×
   at small N and disappears once every SM has work (§4.3–4.4).
2. **For trees, the dominant loss is inside warps.** The coherent control measures it directly:
   incoherence costs #5 a factor of about 4–10 (§4.5.5). Sorting (#5) is the necessary first step, and
   the warp-union walk (#7) on top of it removes most of the remaining gap, 5.8× best against best
   at 1M, even though it does more work (§4.7).
3. **On sm_89 the hardware block scheduler is hard to beat.** It is already a dynamic queue that
   keeps Morton-adjacent warps together on one SM. Every software queue we built improved its
   balance metric and lost speed (§4.6). Better balance is not the same as better performance when
   the kernel is bound by L1 capacity.
4. **The two load-balancing levels interact.** #7 fixes the warp level and thereby makes the
   between-warp tail its largest loss at N = 65,536 (§4.7.4), which is where a locality-preserving
   block-granular queue remains worth testing.
