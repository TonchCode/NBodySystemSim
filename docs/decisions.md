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
