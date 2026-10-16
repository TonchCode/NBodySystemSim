# nanda2025amrexlb

**Citation:** A. Nanda, M. K. H. Chowdhury, H. Ross, K. Gott. "Exploring Dynamic Load Balancing Algorithms for Block-Structured Mesh-and-Particle Simulations in AMReX." *PEARC '25*, ACM, Article 5, 2025. DOI 10.1145/3708035.3736022. arXiv:2505.15122.
**Link:** https://doi.org/10.1145/3708035.3736022 · https://arxiv.org/abs/2505.15122
**Verified on:** 2026-10-17 (arXiv abstract page with journal-ref and publisher DOI). Work done at Lawrence Berkeley National Laboratory.
**Lead-group country:** USA.

## Summary
1. Compares dynamic load-balancing algorithms for mesh and particle simulations in AMReX.
2. Covers a brute-force optimum, Knapsack, and space-filling-curve (SFC) partitioning.
3. Introduces a painter's-partition SFC algorithm and a topology-aware Knapsack+SFC hybrid.
4. Finds Knapsack and painter's-partition worth trying first when weight variation is limited.
5. Their advantage shrinks as the variety of weights grows. GPUs are not discussed.

## What it lets us claim
SFC ordering plus contiguous partitioning (painter's partition) is a standard, cheap balancer. Its quality depends on how skewed the per-item costs are, which is exactly what clustered N-body distributions change.

## Design decision / R3 technique supported
R3 #5 (SFC ordering used for partitioning, not only for locality) and R3 #8 (splitting a Morton-ordered body list between CPU and GPU at a cost-weighted cut point).

## Quote
None captured verbatim from the abstract.
