# shi2025libra

**Citation:** J. Shi, S. Li, Y. Xu, X. Wang, R. Fu, Z. Ma, T. Wu. "Libra: Unleashing GPU Heterogeneity for High-Performance Sparse Matrix Multiplication." arXiv:2506.22714, v1 28 Jun 2025, v2 22 Dec 2025. Preprint; no journal ref or venue comment on the arXiv page. (A search result gives the v1 title as "Libra: Synergizing CUDA and Tensor Cores for High-Performance Sparse Matrix Multiplication"; not checked.)
**Link:** https://arxiv.org/abs/2506.22714
**Verified on:** 2026-10-17 (arXiv abstract + HTML v2). Affiliation line (all 7 authors): "Beijing University of Posts and Telecommunications, Beijing, China."
**Lead-group country:** China (BUPT).

## Summary
1. SpMM/SDDMM that splits work between Tensor Cores and CUDA cores on the same GPU.
2. "Hybrid load balancing" splits long windows/rows into fixed-size groups only when they exceed a threshold, and combines the split segments with atomicAdd. The splitting is precomputed once per matrix (static).
3. Occupancy-aware scheduling computes O = B_launch / (N_SM × B_max^SM) for each path and picks multi-stream or sequential launch from calibrated thresholds.
4. Evaluated on an **RTX 4090 (Ada)** and an H100 PCIe.
5. Reports 1.77x average over FlashSparse and 2.9x over DGL end to end for GNNs (abstract, per search summary).

## What it lets us claim
A 2025 Chinese paper, evaluated on an Ada consumer GPU, that (a) treats "blocks launched / (SMs × resident blocks per SM)" as the quantity to balance, the same quantity as our M1 tail model, and (b) splits only the heavy work items, paying for the split with atomics, while light items stay whole.

## Design decision / R3 technique supported
R3 #2: occupancy and wave fill. The O formula is our B / (80 × blocks/SM).
R3 #3: split heavy work items. Libra combines the pieces with atomicAdd. We combine them with a warp-shuffle reduction, which avoids the atomic cost that Libra weighs against balance.
Strength: **transferable** (sparse linear algebra, not N-body; same GPU family).

## Quote
"Libra further incorporates hybrid load balancing, occupancy-aware task scheduling" (Shi et al., 2025, arXiv abstract)
