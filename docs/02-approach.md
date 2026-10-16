# 02 — Approach & Open Decisions

Written 2026-10-17 before any code. This holds the decisions that must be made before the
project skeleton exists.

## Decision 1: Visualization (CUDA-GL interop vs Unity)

| | **A. Standalone C++ / CUDA + OpenGL interop** (recommended) | **B. Unity + CUDA native plugin** | **C. Unity compute shaders** |
|---|---|---|---|
| Is the simulation CUDA? | Yes | Yes | **No** (HLSL). Fails the brief |
| Per-frame cost | Zero-copy: the kernel writes straight into the VBO | Needs interop through Unity's native rendering plugin API (D3D11/12 ↔ CUDA), or a PCIe copy every frame | Zero-copy |
| Benchmark purity | Clean: we control everything | Unity's own GPU work pollutes timings | n/a |
| Presentation polish | Good (point sprites, additive blending, colour by velocity) | Best (post-processing, camera, UI) | Best |
| Effort | Medium | High (plugin plumbing, D3D interop) | Low, but wrong |

**Recommendation: A.** The deliverable is CUDA + load-balancing analysis. Interop rendering
proves a GPU-resident pipeline, and it is the standard approach in NVIDIA's own `nbody`
sample. The CUDA core should still be built as a **separate library** (`nbody_core`), so a
Unity front end can be added later via B if presentation polish becomes worth it (for
example if this continues into a thesis).

## Decision 2: Build system

CMake (bundled with VS 2026 Insiders) + `CMAKE_CUDA_ARCHITECTURES=89`. One build for the
library, the benchmark CLI and the viewer.

## Proposed layout (once you're ready for code)

```
NBodyProblemSim/
  CLAUDE.md               project context for agents
  .claude/agents/         role agents
  docs/                   our documentation (this folder)
  references/             papers: bib + notes (literature-scout)
  src/core/               kernels: all-pairs variants, Barnes-Hut, load-balancing strategies
  src/bench/              CLI benchmark harness: CSV output
  src/viewer/             OpenGL interop viewer
  results/                benchmark CSVs + plots (python)
```

## Not decided: needs Toncho

- [x] Visualization: **A, OpenGL interop** (D-001, 2026-10-17)
- [ ] Doc language (BG/EN)
- [ ] Power limit for benchmarks: 272 W as-is, or 320 W default
- [x] Git repo: local repo initialized 2026-10-17 (no remote yet)
