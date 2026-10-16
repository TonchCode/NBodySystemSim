# nvidia2026bestpractices (NVIDIA DOC)

**Citation:** NVIDIA Corporation. *CUDA C++ Best Practices Guide*, 13.4 (CUDA 13.4 documentation set). No last-updated date was visible in the fetched portion.
**Link:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
**Verified on:** 2026-10-17 (page fetched: title, version and table of contents).
**Lead-group country:** USA (vendor).

## Summary
1. NVIDIA's official optimisation guide. Its top-level chapters run from Application Profiling and Parallelizing Your Application through to Deploying CUDA Applications.
2. Relevant chapters: Performance Metrics, Memory Optimizations, Execution Configuration Optimizations, Instruction Optimization, Control Flow.
3. "Execution Configuration Optimizations" covers occupancy and block-size choice (R3 #2).
4. "Control Flow" covers branch divergence (R3 #4/#5/#7).
5. "Performance Metrics" covers timing and bandwidth methodology for `results/`.

## What it lets us claim
An official methodology reference for how we time kernels and pick launch configurations. Only the table of contents was read so far; read the specific sections before quoting their content.

## Design decision / R3 technique supported
R3 #2 (execution configuration) and R3 #4/#5 (control flow / divergence).

## Quote
None captured yet.
