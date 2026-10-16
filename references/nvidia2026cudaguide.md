# nvidia2026cudaguide (NVIDIA DOC)

**Citation:** NVIDIA Corporation. *CUDA Programming Guide*, v13.4.2. Last updated 10 Sep 2026.
**Link:** https://docs.nvidia.com/cuda/cuda-programming-guide/ (landing page; the old URL `cuda-c-programming-guide/index.html` serves the same guide). Compute capability tables: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
**Verified on:** 2026-10-17 (landing page + §5.1 Compute Capabilities fetched).
**Lead-group country:** USA (vendor).

## Key facts (compute capability 8.9, §5.1)
- 64 K 32-bit registers per SM; up to 100 KB shared memory per SM and 99 KB per block. The unified L1+shared size is 128 KB.
- The page's table extraction was poorly aligned for the per-SM warp, thread and block limits (48 / 1536 / 24). Those same values are confirmed independently by the Ada Tuning Guide.
- **Version note:** the online guide is **v13.4.2**. The local toolkit reports nvcc V13.4.92 (CLAUDE.md). Record which 13.4 update is installed before claiming the two match.

## Summary
1. The primary reference for the CUDA programming model: threads, blocks, warps, memory spaces.
2. §5.1 lists per-compute-capability limits used in R1 and R2.
3. Expected to cover warp-level primitives (R3 #7) and atomics (R3 #6). Those sections are **not yet read**; cite them only after reading.
4. The current guide lives under a new "cuda-programming-guide" URL with "05-appendices" numbering.
5. Tables 29–32 in §5.1 hold the features and technical specifications.

## What it lets us claim
The official sm_89 resource limits, to cite in R1.

## Design decision / R3 technique supported
R2, plus R3 #2, #6 and #7, once the relevant sections are read.

## Quote
None captured yet.
