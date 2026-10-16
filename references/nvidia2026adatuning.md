# nvidia2026adatuning (NVIDIA DOC)

**Citation:** NVIDIA Corporation. *NVIDIA Ada GPU Architecture Tuning Guide*, v13.4 (CUDA 13.4 documentation set). Last updated 12 Oct 2026.
**Link:** https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
**Verified on:** 2026-10-17 (page fetched).
**Lead-group country:** USA (vendor).

## Key facts (compute capability 8.9)
- Maximum concurrent warps per SM: 48, i.e. 1536 threads. This matches our [M].
- Maximum thread blocks per SM: 24. Matches [M].
- Shared memory capacity per SM: up to 100 KB, with carveout options 0, 8, 16, 32, 64 and 100 KB. Matches [M] 102,400 B.
- "2x more FP32 operations per cycle per SM" than compute capability 8.0 (A100).
- L2 is quoted for AD102 (98,304 KB). AD103's 64 MB comes from the whitepaper, not this guide.

## Summary
1. NVIDIA's official tuning advice for sm_89.
2. Gives the occupancy limits that bound the R3 #2 block-size sweep.
3. Gives the shared-memory carveout choices that bound the tile size in R3 #1/#2.
4. Documents the doubled FP32 throughput per SM, which explains why all-pairs is compute-bound on Ada.
5. Points back to the CUDA C++ Best Practices Guide for general advice.

## What it lets us claim
The occupancy limits used in the design are NVIDIA-documented as well as measured.

## Design decision / R3 technique supported
R3 #2 (occupancy and tile sweep), R3 #1 (shared-memory tile size).

## Quote
"The maximum number of concurrent warps per SM is 48." (Ada Tuning Guide v13.4)
