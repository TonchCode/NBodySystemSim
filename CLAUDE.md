# CLAUDE.md — N-Body CUDA Simulation (course project)

Owner: Toncho Nikolov. Master's course project (TU Sofia, Plovdiv Branch, FEA), separate grade, may continue
into a thesis. Read `docs/00-course-requirements.md` first: it is the grading contract.

## The brief in one paragraph

Simulate the gravitational N-body problem in **CUDA**, optimized specifically for **this**
machine's GPU (RTX 4080 SUPER, Ada AD103, sm_89, 80 SMs, 16 GB). Implement and **measure
several GPU load-balancing techniques**. Describe the GPU architecture in depth. Back every
design decision with **2025–2026 publications** (professor prefers leading Chinese
authors) and NVIDIA documentation. Present it with a live visualization.

## Hardware facts: the single source of truth

`docs/01-hardware.md`. Don't restate specs from memory; read that file. If a fact there is
tagged [S] (spec, not measured), measure it before relying on it in a benchmark claim.

## Toolchain (verified 2026-10-17)

- CUDA 13.4 (nvcc V13.4.92), driver 617.42 (latest)
- VS Community 2026 **Insiders**, MSVC 14.51 (default) and 14.44. Build from a
  `vcvars64.bat` environment. `vswhere` needs `-prerelease` to find it.
- `nvcc -arch=sm_89`. CMake: `CMAKE_CUDA_ARCHITECTURES=89`
- Nsight Compute 2026.3.1, Nsight Systems 2026.3.2

## Rules

1. **No fabricated citations, ever.** Every reference must be fetched and verified (title,
   authors, year, venue, DOI/arXiv). Unverified = not cited.
2. **Every performance claim has a number behind it** in `results/`, with environment
   recorded: driver, `cl` version, power limit, clocks, background GPU load.
3. **FP32 in force kernels.** FP64 is 1/64 rate on this GPU. Deviations need a written reason.
4. **Kernels stay well under ~2 s.** WDDM TDR watchdog; display is on this GPU.
5. Source tags in docs: [M] measured · [S] spec · [D] derived · [P:key] paper ·
   [N:doc] NVIDIA doc.
6. Log decisions with the *why* in `docs/decisions.md` (create on first decision).

## Agents (`.claude/agents/`)

| Agent | Role (professor's framing) |
|---|---|
| `cuda-architect` | programmer: designs and writes kernels tuned to sm_89 |
| `literature-scout` | scientist: finds and verifies 2025–2026 papers and NVIDIA docs |
| `critic` | critic: attacks claims, methodology, citations, code |
| `bench-profiler` | runs builds, benchmarks and Nsight; owns `results/` |
| `tech-writer` | writes the final documentation and presentation text from verified material only |

Typical loop: scout → architect → profiler → critic → writer.
