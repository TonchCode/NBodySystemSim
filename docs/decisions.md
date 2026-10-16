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

## D-004 — Viewer windowing/GL loader: OPEN

Options: **(a)** GLFW + glad fetched by CMake `FetchContent`. Standard, about 20 lines of
setup, but it downloads third-party code at configure time. **(b)** Raw Win32 + WGL, loading
the ~15 GL functions we need by hand. No dependencies, more boilerplate, Windows-only (fine:
one machine by brief). Needs Toncho's call.

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
