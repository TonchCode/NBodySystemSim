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
**Why:** all 49 tree runs pass with 2.3–4.6× margin on rms. Bug detection is the exact selftest's
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

## D-020 — How tree speedups over all-pairs are quoted (2026-10-17, cuda-architect)

**Decided:** tree totals (build + traversal, one force evaluation) are compared with the best
all-pairs configuration measured in the same session at the same N (rule 8: max over tuned
b128 u0 and multi b × q). Measured: 1.73× at 65536 and 5.72× at 262144 (sorted); 1.11× and 3.08×
(naive). At N = 1M all-pairs is not run (≈ 0.82 s per launch, too close to the TDR margin), so
it is extrapolated as 4 × T(524288) = 822 ms and every 1M speedup (21.0× sorted, 11.2× naive)
carries the [D] tag. Crossover: N ≈ 39,700 (sorted) and ≈ 59,600 (naive).
