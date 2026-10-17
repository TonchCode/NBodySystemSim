# xia2025voltrix

**Citation:** Y. Xia, W. Wang, D. Yang, X. Zhou, D. Cheng. "Voltrix: Sparse Matrix-Matrix Multiplication on Tensor Cores with Asynchronous and Balanced Kernel Optimization." In *2025 USENIX Annual Technical Conference (USENIX ATC '25)*, Boston, MA, July 2025, pp. 699–714.
**Link:** https://www.usenix.org/conference/atc25/presentation/xia
**Verified on:** 2026-10-17 (USENIX landing page). Affiliations: Xia, Wang and Cheng at Wuhan University; Yang at Nvidia Corporation; Zhou at the University of Macau.
**Lead-group country:** China (Wuhan University).

## Summary
1. SpMM on Tensor Cores.
2. Asynchronous loading pipeline: bit-compressed sparse input, bulk copies for dense input.
3. A **persistent kernel** with a two-stage partition that balances both input (I/O) and output work across blocks.
4. Argues that earlier methods cannot manage data imbalance across computations.
5. Reports 1.7x average over RoDe and 1.8x over DTC-SpMM (CUDA 12.6), per a search summary of the abstract.

## What it lets us claim
A 2025 Chinese systems paper (USENIX ATC, CCF-A) that fixes imbalance in irregular work with a persistent kernel plus a precomputed balanced partition.

## Design decision / R3 technique supported
R3 #6, the persistent-thread arm. Voltrix computes its partition ahead of time, so it is closer to "persistent kernel + static balanced partition" than to an atomic queue. That makes it a third arm to name next to li2025staticbatching (static, no persistence) and maeda2026gtap (persistent + stealing). Strength: **transferable** (sparse, not N-body; per-cell interaction counts in a tree play the role of nonzeros per row).

## Quote
"a persistent and I/O co-balanced kernel mechanism" (Xia et al., USENIX ATC '25, abstract)
