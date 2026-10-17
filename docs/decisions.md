# Decisions

Each entry: what was decided, why, what it rules out. Append only.

## D-001 — Visualization: standalone CUDA + OpenGL interop (2026-10-17)

**Decided by Toncho.** Option A from `02-approach.md`.
**Why:** the deliverable is CUDA + load-balancing analysis. Interop draws straight from the
device buffer (no per-frame PCIe copy; there is only one copy engine), timings stay clean,
and it is the standard approach in NVIDIA's own `nbody` sample.
**Rules out:** Unity compute shaders (not CUDA). Unity + native plugin is deferred, not
dead: `nbody_core` is a separate library on purpose.

## D-002 — Build: CMake + Ninja, sm_89 only (2026-10-17)

CMake and Ninja are both bundled with VS 2026 Insiders. `tools/build.bat` finds VS via
`vswhere -prerelease`. One architecture (`89`), because the project targets one device by
brief.

## D-003 — Numerics baseline (2026-10-17)

FP32 force kernels (`float4` pos+mass, `rsqrtf`, FMA). Plummer softening eps = 0.01.
Leapfrog KDK integrator. Plummer-sphere initial conditions in N-body units (G = M = 1).
Energy is checked in double on the host. FLOP convention: 20 per interaction (GPU Gems 3
ch. 31), always stated next to any GFLOP/s figure.

## D-004 — Viewer: GLFW 3.4 + glad, vendored (2026-10-17)

**Decided by Toncho**, on the recommendation. GLFW handles the window, context and input.
glad is the GL 3.3 loader, taken from GLFW's own `deps/`, so there was one download, not two.
Both are **committed into `third_party/`** (pinned, sha256 recorded in
`third_party/README.md`), so builds need no network.
**Why not raw Win32/WGL:** about 300–400 lines of window boilerplate worth zero grade. The
interop code is identical either way.
**Interop design:** the core keeps its own `cudaMalloc` buffers. Each frame does a
VRAM→VRAM `cudaMemcpyAsync` into the mapped VBOs: 1 MB per buffer at N = 32k, microseconds
at 736 GB/s. The bench and the viewer therefore run the exact same core code.
Verified: GL 3.3 core on the RTX 4080 SUPER, N = 32,768 at 180 FPS (vsync = monitor 180 Hz),
1.2 ms/step including the integrator. Screenshot: `docs/img/viewer_plummer_32k_2026-10-17.png`.

## D-005 — Literature first pass accepted, with caveats (2026-10-17)

`references/` holds 20 verified entries (13 from 2025–26, 3 background, 4 NVIDIA docs) and
about 17 unverified leads. Spot-check by the main session: 3 of 3 matched their arXiv or
Crossref records (maeda2026gtap, li2025staticbatching, yu2026cube2).
**Caveat for the professor:** no 2025–26 *Chinese GPU N-body* paper was found. The Chinese
entries are CUBE2 (a CPU code), swPHoToNs (Sunway, online 2023) and Alibaba static batching
(scheduling). We say this plainly rather than stretch.
**Open scope question (architect):** RT-core Barnes-Hut (nagarajan2025rtbarneshut) needs
OptiX, not plain CUDA. It is in or out of scope as an extra.
**Methodology consequence:** the naive tree baseline (R3 #4) must deliberately omit sorting
and warp-vote traversal. Report #6 vs #7 as phase time *and* total time (per maeda2026gtap).

## D-006 — Validation tolerance for reordered summation (2026-10-17, cuda-architect)

Variants that reorder the j-sum (`allpairs_multi`, later trees) are not bitwise equal to
naive. Gates (03-design §4.6): **G1** rms rel. error vs naive ≤ 4·u·√N (u = 2⁻²⁴); **G2**
max |Δa| / Σ|f_ij| vs an FP64 sample ≤ 4·u·√N; **G3** FP64 rms of the variant ≤ 1.5 × naive's.
max-rel is reported, not gated (1e-2 sanity ceiling).
**Why:** max-rel is inflated by κ = Σ|f| / |a| on bodies whose net force nearly cancels.
That is a property of the problem, not of the kernel. rms and the normalised max are not
inflated. **Rules out:** the smoke test's fixed "max rel < 1e-4", which naive itself fails
against FP64 on ill-conditioned bodies at large N.

## D-007 — Small-N timing needs time-based warm-up and batching (2026-10-17, cuda-architect)

`nbody_bench --warmup-ms` and `--batch` were added (defaults unchanged). Without them, timings
of 5–25 µs kernels were bimodal (clock ramp from 210 MHz idle, event jitter).
**Recommendation to bench-profiler:** `--warmup-ms 300 --batch 50` for N ≤ ~20k.

## D-008 — `allpairs_multi` auto-q = pre-registered cost model v1 (2026-10-17, cuda-architect)

Kept the busiest-SM cost model written before measuring (`autoThreadsPerBody`). Mean loss vs
the best measured q: 1.8% in-sample, 0.9% out-of-sample. The thread-saturation rule loses
6.8% / 4.8%. A v2 fitted to the measured per-SM rate curve was rejected because it was worse
out-of-sample (overfit).

## D-009 — Validation gates v2: conditioning-independent, calibrated, in code (2026-10-17, cuda-architect, round 2)

**Supersedes the D-006 thresholds** (D-006 stays as the record of what was decided first).
Pre-registered in `docs/predictions/2026-10-17-round2.md`, committed before the round-2 runs.
Implemented in `src/bench/main.cpp` (`kG0/kG1/kG2`). Every validated bench run prints PASS/FAIL,
and a FAIL exits with code 3.
- **G1 (primary):** normMax = max |Δa| / Σ_j|f_ij| vs the FP64 sample ≤ **1.0 · u · √N**.
- **G2 (no regression):** the variant's normMax ≤ **1.5 ×** naive's normMax on the same sample.
- **G0 (gross bugs):** max rel error vs naive ≤ 1e-2.
- **Missed interactions:** `nbody_bench --selftest`. An exact-arithmetic lattice (dyadic
  positions, masses {1, 2}, eps = 1), where every FP32 operation is exact in any order. All
  1810 variant × N × instrumented cases must be *bitwise* equal to the analytic sum.

**Why each change (critic review 1, #6):**
1. *Primary metric = normMax, not rms-rel.* rms-rel divides by |a_i|, so it is inflated by
   κ_i = Σ|f| / |a| exactly like max-rel. That makes the D-006 "rms is not κ-inflated"
   statement wrong. normMax divides by Σ|f|, so it is conditioning-independent.
2. *Calibrated, not generic.* Measured naive normMax / (u√N) = 0.47–0.68 over N = 1000…65536
   (round-1 sanity data, 8 values of N). The old 4·u·√N sat 6–8× above that noise. kG1 = 1.0
   = 1.5 × the worst measured ratio. Naive itself passes G1 with a ≥ 1.47× margin.
3. *Statistical gates cannot catch a dropped j at large N.* One missing body changes a_i
   by about |f_ij| / Σ|f| ≈ 1/N on average: 1.5e-5 at N = 65536, about G1's own threshold.
   So the dedicated exact test is the gate for that failure class. Shown to work: an injected
   "drop the last body" bug in tiled, and a "drop one j per tile" bug in one multi config,
   fail exactly those configs and nothing else. `--selftest-drop J` (body J's mass zeroed in
   the GPU input only) fails every case where the drop is visible
   (`results/2026-10-17_round2/phase1_checks/`).
4. *Worst-conditioned sample by κ, not smallest |a|.* The 16 extra FP64 bodies are the
   largest-κ ones: a GPU FP32 shortlist of 64, then ranked by FP64 κ.

**Rules out:** gating on rms-rel or max-rel. Using G1 for tree codes: those have a θ-dependent
truncation error by design and will need their own pre-registered gate.

## D-010 — Auto-q v2 rejection withdrawn; v1 stays the default (2026-10-17, cuda-architect, round 2)

D-008 said the v2 auto-q model (busiest-SM work over a measured r(warps) curve) was rejected
because it was "worse out-of-sample". The round-1 v2 code was never saved. A reconstruction
from that description (`tools/autoq_models.py`, run on the same round-1 CSVs) gives a mean loss
of **1.0% in-sample and 0.6% out-of-sample**, against v1's 1.8% / 0.9%, which v1 reproduces
exactly. On the round-2 sweep it gives 1.3% vs v1's 1.8% (in-sample only).
**Decided:** the rejection reason is withdrawn as unsupported. Either the lost original
differed or the claim was wrong, and that can't be told apart now. **v1 stays the bench
default.** It was the pre-registered model, the difference is ≤ 0.5 pp over 12–20 cells, and
switching on that evidence would be fitting to noise. The real loss is elsewhere: auto-q
searches q at a fixed block, and the joint (block, q) optimum beats it by up to 12% (N = 4096).
**Rules out:** citing "v2 overfit" anywhere.

> **Correction (2026-10-17, cuda-architect, tree round 2):** recomputed from
> `results/2026-10-17_round2/sweep/sweep.csv` (medians, prefetch on = the default, N = 4096,
> joint best = multi b512 q8, 0.01878 ms): auto-q at the bench default b256 (q4, 0.02177 ms) is
> **15.9% slower, i.e. 13.8% lower throughput**; the worst fixed block is b128 (q4, 0.02252 ms:
> 19.9% slower, 16.6% lower throughput); at b512 / b1024 auto-q is within 0.4% / 1.0%. So critic
> review 1's "14%" is right (throughput at the default block), and "up to 12%" above is wrong: it
> matches no cell. The correct statement is "up to 16.6% lower throughput (b128), 13.8% at the
> default b256".

## D-011 — Load imbalance is measured with per-block timelines (2026-10-17, cuda-architect, round 2)

GPU performance counters stay blocked (Toncho's call), so ncu's `sm__cycles_active` is not
available. Each force kernel has an instrumented twin (template flag `TL`; production SASS is
identical to round 1 for all 57 kernels). The twin records `%smid`, `%globaltimer` and `clock64`
per block (`nbody_bench --timeline`), and `tools/timeline_analysis.py` derives per-SM busy time,
max/mean, idle SMs, tail and SM efficiency.
**Why the extra clock calibration:** `%globaltimer` ticks in 1.024 µs steps on this board
(measured), comparable to a whole N = 1000 kernel. `clockCalibration` anchors each SM's
clock64 to a tick edge before and after the launches. Over multi-ms windows the SM clock
changes (power cap), so the mapping can drift by up to 37 µs (1% of span). Raw and
calibrated metrics agree within about 1% on every headline number, and both are reported.
**Known limitation:** ptxas schedules the twin's force loop differently (same opcodes, similarity
0.73), and the twin runs −3.4…+6.2% differently from production. So timelines give
*structure* (placement, idle SMs, tail fraction); all throughput numbers come from production
kernels. A future fix: stamp from a separate `__noinline__` function so the loop is compiled
identically, then re-verify with `tools/sass_diff.py`.

## D-012 — Model M1 kept as a structural model only (2026-10-17, cuda-architect, round 2)

Measured (D-011): for single-wave launches the ceil(B/80) step is exact (tiled b256 at
N = 65536: 23.5% tail, SM efficiency 0.81 vs M1's 0.80). For multi-wave launches it is smoothed
far below one block (multi q4: tail 0.1%), as [zhang2026wavetune] describes. The round-1 P2a
miss (block 64 +6% vs predicted +20%) is **not** a step failure: b64 removes the tail exactly
(SM efficiency 0.97), but runs ≈ 12% slower per SM per interaction (per-tile cost: 4× the
barriers and loads).
**Decided:** M1 predicts *which* SMs idle and *how long the tail is*. It is not used to rank
block sizes, because its per-tile cost constant c = 2 is too small at block 64. Rankings come
from the sweep, quoted against the best-tuned baseline (rule 8).

## D-013 — Prefetch stays on in `allpairs_multi`; the best-tuned baseline is defined (2026-10-17, cuda-architect, round 2)

Measured with the new `--prefetch` switch: register prefetch is worth +6.7% at the best N = 4096
config, +10–16% at 2–4 warps per SM, and 0% at N = 65536. It is never more than 3% negative,
so it stays the default.
**Best-tuned baseline (rule 8)** = the best configuration with one thread per body over tiled,
tuned (all block × unroll) and multi q = 1 (prefetch on or off), measured in the same session.
On 2026-10-17 that was tuned b128 full at every N. R3 #3 speedups vs that baseline: 2.40×,
2.56×, 1.40×, 1.02× (N = 1000, 4096, 10000, 65536). At N = 4096 the gain splits as q 2.40× ×
prefetch 1.067×.

## D-014 — Energy drift is FP32 round-off; dt = 1e-3 stays (2026-10-17, cuda-architect, round 2)

N = 4096, 100 time units (`results/2026-10-17_round2/energy/`): |ΔE/E| = 6.2e-7 at dt = 1e-3
and 1.1e-6 at dt = 5e-4. Halving dt *increases* the drift, so it is dominated by per-step FP32
round-off in pos/vel, not by KDK truncation. A smaller dt buys nothing. tiled and multi drift
by the same order of magnitude (6e-7 vs 1.6e-6; the trajectories diverge chaotically).
**Rules out:** shrinking dt to improve energy conservation. If better conservation is needed,
the lever is the integrator's state precision (e.g. compensated position updates), which is an
R2 decision for Toncho. FP32 in the force kernels is unaffected.

## D-015 — Tree code: GPU-resident binary radix tree, monopole, bmax criterion (2026-10-17, cuda-architect, tree round 1)

**Decided:** the Barnes-Hut build and traversal are fully GPU-resident:
- Morton keys (63 bits), sorted with CUB radix sort from the CUDA 13.4 toolkit.
- A binary radix tree in one kernel.
- Bottom-up monopoles with atomic arrival counters.
- The bmax opening criterion.
- Leaf buckets of 16, traversal block 128.

**Why:**
- Measured build = 0.24 / 0.30–0.40 / 0.62 ms at N = 65536 / 262144 / 1M, i.e. 12.6% / 4.5% /
  1.6% of the sorted step: never the bottleneck.
- Monopole-only keeps a cell interaction identical to a body interaction (one work unit).
- bmax opens every node whose box contains the body (for θ ≤ 1).
- Measured error at θ = 0.5: rms 6.5e-3 (Plummer) to 1.0e-2 (core).

**Rules out:** a CPU build in the R3 #4/#5 numbers (that is #8), and quoting our θ as the
classic θ: it is a bmax-θ, ≈ 0.87 × the classic one.
**Known cost:** 1.27–1.57 node pops per interaction at N ≥ 65536 (binary depth). That is an
octree or wide-node question for later rounds.

## D-016 — R3 #4 vs #5 differ only in the thread → body map (2026-10-17, cuda-architect)

**Decided:** `bh_naive`, `bh_sorted` and `bh_hilbert` run one kernel on one tree. Only the
`order[]` array differs (identity / Morton permutation / Hilbert permutation), so accelerations
are bitwise identical (selftest).
**Why:** D-005 requires #4 to leave out sorting and warp cooperation. The tree build sorts in
every variant, so "unsorted" means the threads run in the original, spatially random order.
The θ = 0 control (identical walk for every body) gives #4 = #5 within 0.3%. At θ = 0.5 the
measured #5/#4 gain (1.6–2.1×) is therefore caused by the order alone.
**Rules out:** comparing #5 against a #4 with a different tree or kernel.

## D-017 — Tree validation gates TG1/TG2 adopted (2026-10-17, cuda-architect)

**Decided:** for θ > 0, gate on rms rel error ≤ 0.03·(θ/0.5)² and p99 rel error ≤ 0.12·(θ/0.5)²
over all bodies vs the chunked exact reference, plus zero stack overflows. θ = 0 uses D-009.
Pre-registered in 581fc47.
**Why:** all 49 tree runs pass with 2.3–4.6× margin on rms.
*(Correction 2026-10-17, tree round 2: **47** gated runs, not 49 (critic review 2, #13); the
other θ > 0 logs are timeline runs with `--no-validate`.)* Bug detection is the exact selftest's
job: 720 θ = 0 lattice cases, `--selftest-drop`, and an injected leaf-loop bug that failed
exactly its 486 visible cases. The statistical gates only catch gross approximation errors.
**Rules out:** gating trees with D-009's FP32-roundoff thresholds (θ > 0 is truncation-limited
by design), and gating on max rel (κ-inflated: 0.15–0.56 on a few near-centre bodies).

## D-018 — Morton, not Hilbert, is the R3 #5 default (2026-10-17, cuda-architect)

**Measured:**
- Hilbert lowers per-warp max/mean by only 0.2–0.6%.
- Its traversal is 1.5–4.4% slower in 8 of 9 cells.
- It needs an extra 0.13–0.24 ms key + sort phase.

The Morton order is free: the build already produced it.
**Decided:** `bh_sorted` (Morton) is the #5 result to quote; `bh_hilbert` stays registered as
the measured sub-variant. Thaler & Keller's compactness argument [thaler2026sfcneighbor] does
not translate into a traversal gain here. That is the new data point the evidence map (§5) said
was missing.

## D-019 — Next round: #7 (warp-cooperative traversal) before #6 (2026-10-17, cuda-architect)

**Decomposition** (not pre-registered; `results/2026-10-17_tree_round1`, `tools/tree_analysis.py`,
θ = 0 control as the coherent reference):
- The sorted traversal reaches only 11–13% of the kernel's coherent throughput.
- Of the loss, at N = 1M: 38–42% is idle lanes inside warps (interaction-site SIMT eff ≈ 0.2),
  57–60% is memory-side incoherence, and 1% is blocks + tail (slot fill 0.976–0.978, SM eff
  ≥ 0.987, no idle SMs). The naive variant is 49–50% / 49–50% / 1%.
- Only single-wave clustered launches show a real SM-level loss (sorted core at 65536: 15% of
  the loss, tail 17.7%).

**Decided:** R3 #7 is implemented first, as a warp-cooperative walk (warp vote over the union of
the lanes' decisions [burtscher2011barneshut], or per-warp interaction lists). Work stealing is
not the first choice: the loop-level imbalance it targets is ≈ 2.4% for #5. R3 #6 is still built
(it is a graded technique), but pre-registered as small (≤ 3% at N ≥ 262144), with its test case
in the single-wave clustered regime. The extra work of the union walk is the quantity to
pre-register for #7.

> **Correction (2026-10-17, tree round 2):** "≤ 3%" applies to a **block-granular queue only**
> (critic review 2, #2). Round 2 builds #6 at warp and at lane granularity and pre-registers the
> lane-granular bound from 1 − eff_pop and per-warp max/mean (§17.3,
> `docs/predictions/2026-10-17-tree-round2.md`). The 85–99% / 8.4× figures rest on a residual
> (critic review 2, #1) and are replaced by a measured coherent control (§17.4).

## D-020 — How tree speedups over all-pairs are quoted (2026-10-17, cuda-architect)

**Decided:** tree totals (build + traversal, one force evaluation) are compared with the best
all-pairs configuration measured in the same session at the same N (rule 8: max over tuned
b128 u0 and multi b × q). Measured: 1.73× at 65536 and 5.72× at 262144 (sorted); 1.11× and 3.08×
(naive). At N = 1M all-pairs is not run (≈ 0.82 s per launch, too close to the TDR margin), so
it is extrapolated as 4 × T(524288) = 822 ms and every 1M speedup (21.0× sorted, 11.2× naive)
carries the [D] tag. Crossover: N ≈ 39,700 (sorted) and ≈ 59,600 (naive).

## D-021 — Tree rates are labelled; the CSV holds the real rate (2026-10-17, cuda-architect, tree round 2)

**Decided:** for approximate (tree) kernels the bench headline says "G N^2-equivalent
interactions/s (not the tree rate)" and drops GFLOP/s; a `rates:` line gives the real rate over the
pipeline and over the traversal. In the round-2 CSV header `ginteractions_per_s` and `gflops` hold
the **real** rate (all-pairs: N²; trees: counted interactions / pipeline time), and
`n2equiv_ginter_per_s` and `trav_ginter_per_s` are separate columns.
**Why:** critic review 2, #5: round-1 tree rows carry N²-equivalent values (e.g. 28,071 G/s) in
the column every all-pairs row uses for its real rate.
**Rules out:** reading `ginteractions_per_s` of a round-1 tree CSV as a tree rate. Historic CSVs are
unchanged; the bench refuses to append to them (header check).

## D-022 — R3 #7 is the warp-union walk; bh_warp is the tree to quote (2026-10-17, cuda-architect, tree round 2)

**Measured** (`results/2026-10-17_tree_round2`, background-load label): traversal 3.1–7.3× faster
than #5 (best vs best 5.9× at 1M, 5.2× at 262144, θ = 0.5), SIMT efficiency 1.000, 1.6–1.9×
interactions, lower error (TG3 60/60), crossover vs the best all-pairs N ≈ 29,200.
**Decided:** `bh_warp` (direct union walk, register stack, Morton warps) is the R3 #7 result and
the tree to use in later rounds (making it the viewer default is a pending code change). `bh_warp_list` stays registered as a measured
negative (2–7% slower). Quote #7 as "warp-cooperative traversal (union walk with a warp vote)",
not as work stealing.
**Rules out:** describing #7's gain as removing work: it adds work and removes incoherence.

## D-023 — R3 #6 result: the software queues lose to the hardware block scheduler (2026-10-17, cuda-architect, tree round 2)

**Measured:** warp-granular queue 0.89–0.95×, lane-granular refill 0.64–0.91×, static cost-balanced
partition 0.84–0.98× of their siblings, in every cell, while the balance metrics they target improve
(SM efficiency, warp-slot fill, pop efficiency). Exploratory: a block-granular queue recovers most
of the warp queue's loss (0.94–0.98×); checking refill every 8 pops does not rescue lane refill.
**Decided:** report #6 as a measured negative result with its mechanism: on sm_89 the block
scheduler is already a dynamic, locality-preserving queue at 4-warp granularity, and the remaining
SM-level loss at N ≥ 262144 is 1–2%. A software queue must claim Morton-adjacent units per block to
keep L1 sharing. The pre-registered "≤ 3%" bound (D-019) was right in size for the block-granular
gain and wrong in sign for the warp and lane arms. The critic's lane-granular arm was built and
measured, so #6 is not a strawman.
**Rules out:** claiming any #6 speedup; citing huang2026weave or chen2026gmatch as predicting a gain
for this workload.

## D-024 — The loss decomposition is measured with the coherent control (2026-10-17, cuda-architect, tree round 2)

**Decided:** intra-warp loss is quoted as T(#5) / T(control), where the control is #5's kernel with
`order[t] = perm[t & ~31]` (5.2–9.7× measured), and #7 as the share of that gap it closes (69–88%,
log terms). The round-1 E_lanes × E_mem × E_sm split (D-019, §16.4) is kept only as a historical
estimate.
**Why:** critic review 2, #1: E_mem was a residual and the ceiling moved with c_V.

## D-025 — Best-tuned tree configurations (rule 8) (2026-10-17, cuda-architect, tree round 2)

**Measured** (set C, Plummer, θ = 0.5): #4 best leaf 32 b256 (1M) / leaf 16 b256 (262144); #5 best
leaf 32 b256 at both N (11% faster than leaf 16 b128 at 1M); #7 flat within ±1% over leaf 8–16
and all blocks (best leaf 8 b256 at 1M, leaf 16 b128 at 262144).
**Decided:** tree speedups are quoted best vs best with θ stated: #5/#4 2.06× (262144), 2.18× (1M);
#7/#5 5.2× and 5.9×. Leaf 16 block 128 stays the protocol default for cross-round comparability,
always labelled as such.

## D-026 — #5 is L1-capacity-bound; the coherent walks are latency-bound (2026-10-17, cuda-architect, tree round 2)

**Measured** (set D, 262144): with the L1 carve-out pinned small, #5's rate is flat from 2 to 11
blocks/SM, and the unpadded kernel is ×1.51 faster; the control and #7 scale ×1.75–1.87 from 4 to 11
blocks/SM and gain ×1.12–1.30 from L1.
**Decided:** design notes say "#5: L1-capacity-bound scattered accesses; #7: latency-bound pop
chain", not "memory-bound". Next optimisations for #7 target pops per interaction (wider nodes),
not occupancy.
**Caveat:** padded "12" ran at 11 blocks/SM (256 B allocation granule); fix the padding formula
before reusing `--occ-blocks`.

## D-027 — Normalised tree gates TN1/TN2 replace TG1/TG2 as the primary tree gates (2026-10-17, cuda-architect, IC round)

**Supersedes D-017 as the gating rule** (D-017 stays as the record; TG1/TG2 are still computed and
printed as "legacy", and TG3 as "legacy relative"). Implemented in `src/bench/main.cpp`
(`kTN1`, `kTN2`, `normStats`); the selftest's TG3 uses the same metric.
- **Metric:** e_i = |a_tree,i − a_exact,i| / S_i over all bodies, S_i = G Σ_j m_j |r_ij| / (r_ij² + ε²)^1.5
  (the D-009 normaliser, from `forceScale`). Also printed: |Δa_i| / |a|_rms (global), and with
  unequal masses the split heavy (m ≥ 10× median) vs light.
- **TN1:** rms(e) ≤ 0.02 (θ/0.5)². **TN2:** p99(e) ≤ 0.045 (θ/0.5)². Plus zero stack overflows.
- **TG3n:** the union walk's normalised rms and p99 ≤ its sibling's (#5, or #4 for `bh_warp_naive`).

**Why:**
1. *Per-body relative error breaks on near-zero net force* (tree round 2, summary §8.1): it divides by
   |a_i|, which cancels on sheets and lines, between galaxies and next to a dominant mass. Measured in
   phase 1: on `cube_edges` at 8192 the legacy rms rel error is 0.150 and p99 0.65 (legacy TG1/TG2 FAIL)
   while the normalised rms is 9.3e-3. S_i ≥ |a_i| always, and does not cancel.
2. *It is IC-independent by construction [D]:* for an accepted cell the monopole error is O(θ²) times
   that cell's share of S_i, so e is O(θ²) for every body whatever the geometry. The legacy metric has
   no such bound (it scales with κ_i = S_i / |a_i|).
3. *Calibrated on the legacy ICs only, before any new-IC measurement:* 24 bh_warp validation runs (each
   also evaluating #5) on plummer, collision, core × N 65536–1M × θ 0.3–1.0 × leaf 8/16/32
   (`results/2026-10-17_ic_round/phase1/calib/`). Worst normalised rms / (θ/0.5)² = 6.1e-3 and p99 /
   (θ/0.5)² = 1.43e-2 (#5, core/collision at θ = 0.3: the error falls slower than θ² at small θ).
   The gates sit ≈ 3.3× and 3.1× above, as TG1/TG2 did (bug catchers; the exact host check and the
   lattice selftest catch real bugs).
4. *Verdicts on the legacy ICs do not change:* all 24 runs PASS TN1/TN2/TG3n and also PASS the legacy
   TG1/TG2/TG3; the pre-round binary prints identical error values on the same runs.

**Known exception (XFAIL, logged, not hidden):** TG3n fails for `bh_warp_naive` on `cube_edges`
(n = 10007 and 65536, θ 0.5 and 0.7): the union of 32 *random* bodies has ≈ 13% higher normalised
(and legacy) error than #4, although every cell it uses passes the lane's own criterion and the exact
θ > 0 host check passes. On collinear distributions the children of a refined cell do not cancel as
well as their parent did, so "finer is more accurate in aggregate" (TG3's premise) is false there. The
selftest reports it as XFAIL; the bench still gates it (such a run exits 3). `bh_warp_naive` is not in
this round's measured set. The Morton-ordered `bh_warp` passes TG3n on every IC.
**Rules out:** gating trees on per-body relative error; quoting a legacy TG1/TG2 FAIL on a degenerate
IC as an accuracy regression.

## D-028 — Traversal stack: 96 entries per thread (provable), 64 in the #7 register stack (measured) (2026-10-17, cuda-architect, IC round)

**Decided:** `kStack = 96` for every per-thread walk (#4/#5 traversal, the per-thread persistent arms);
the #7 warp stack stays at 64 entries in two registers (`WARP_STACK_ENTRIES`, a build flag), with the
overflow counter as an explicit run FAIL, and the bench prints the tree's push depth and stack bound
on every tree run (`tree:` line, `lastTreeDepth()`).
**Why:**
- *Bound [D]:* along any root-to-node path of the radix tree the split prefix length δ strictly
  increases; δ ∈ [1, 63] for distinct 63-bit keys and 64 + clz(i ⊕ j) for duplicates (⌈log2 n⌉
  values). So the deepest internal node is at depth ≤ 62 + ⌈log2 n⌉, and a right-then-left DFS holds
  ≤ pushDepth + 2 ≤ 64 + ⌈log2 n⌉ ≤ 95 entries for any int n. 96 cannot overflow.
- *Cost of 96 per thread [M]:* exactly two SASS lines change in the 18 per-thread-walk kernels (frame
  0x100 → 0x180, bound 0x3f → 0x5f); same instruction count, 40 registers, occupancy unchanged; the
  local-memory reserve grows from 31 to 47 MB. The 204 other kernels are SASS-identical
  (`results/2026-10-17_ic_round/phase1/sass_diff_stack.txt`). Phase 2 times the old and new binary on
  round-2 reference points to confirm no speed change.
- *Why not 96 for #7 [M]:* a third stack register keeps production at 40 registers but adds 8
  instructions to the latency-bound pop chain (296 → 304; D-026 says this walk is pop-latency-bound)
  and takes the instrumented twin from 40 to 44 registers (fewer resident blocks for the timelines).
  The measured need is far below 64: stack bound ≤ 45 for every legacy IC at 1M (core: push depth
  41), ≤ 24 for every new IC at 8192. If a future IC exceeds it, the run FAILs loudly, and the build
  flag `-DWARP_STACK_ENTRIES=96` is the documented fallback.
**Rules out:** a silent overflow; a 64-entry per-thread stack (correct only for push depth ≤ 62).

## D-029 — IC workload suite, mass spectrum, viewer live N (2026-10-17, cuda-architect, IC round)

**Decided:**
- New `--ic`: `disk` (exponential disk + Plummer bulge + one central body of `--bh-mass`, default
  0.01), `galaxies` (two disks, inclined, bound collision course), `cube_shell` and `cube_edges`
  (stress tests: hollow and 1D), `uniform` (the no-clustering control). `--mass-spectrum equal |
  salpeter | heavy-few`. All total mass 1, mass-weighted COM and momentum removed, shuffled. Design
  §19.1, `src/core/init.cpp`.
- The legacy ICs with the equal spectrum are **bitwise unchanged** (hash-checked in the selftest).
- Every IC is measured with #4, #5, #6, #7: ICs are imbalance generators, not decorations.
- Viewer: live N (PageUp/PageDown or `.`/`,`), `I` cycles the IC, `H` the mass spectrum, `M` colour by
  mass; title shows IC, spectrum, N, kernel, ms/step and the **counted** interaction rate for tree
  kernels; N capped at 262144 (all-pairs) and 4,194,304 (tree) for the TDR margin; the default kernel
  is now `bh_warp` (the pending item of D-022).
**Why these models:** the disk velocities use the softened enclosed-mass circular speed of a sphere
with the same M(r) — exact for the bulge and central body, an approximation for the thin disk's own
field [TODO(source): Freeman 1970]. There is no dark halo and no Toomre-Q control, so the disk is a
self-gravitating cold disk that will form bars and clumps within about a rotation: a valid force
snapshot and a demo, not a stable galaxy. Masses are drawn independently of position, so the expected
density profile (and the collisionless equilibrium of the Plummer) is unchanged; heavy bodies will
sink by dynamical friction over time.
**Softening and dt:** ε stays global (0.01). The bench and viewer print the advisory dt ≤ 0.1 √(ε³ /
(0.385 G m_max)) (1.6e-3 for a 1% body); the viewer's dt = 2e-3 exceeds it near the central body (a
visual). This round's measurements are force snapshots and do not integrate.
**Rules out:** labelling `cube_shell`/`cube_edges` as physical systems, or the disk as an equilibrium
model.
