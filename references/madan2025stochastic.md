# madan2025stochastic

**Citation:** A. Madan, N. Sharp, F. Williams, K. Museth, D. I. W. Levin. "Stochastic Barnes-Hut Approximation for Fast Summation on the GPU." *SIGGRAPH Conference Papers '25*, ACM, Vancouver, pp. 1–10, 2025. DOI 10.1145/3721238.3730725. arXiv:2506.02219.
**Link:** https://doi.org/10.1145/3721238.3730725 · https://arxiv.org/abs/2506.02219
**Verified on:** 2026-10-17 (arXiv abstract page + Crossref DOI record). Affiliations: University of Toronto; NVIDIA.
**Lead-group country:** Canada / USA (U. Toronto + NVIDIA).

## Summary
1. Proposes a stochastic Barnes-Hut variant in which level-of-detail approximations act as control variates.
2. This gives an unbiased estimator of the full kernel sum, unlike deterministic BH.
3. Built for GPU fast summation and tested on graphics kernels (winding numbers, smooth distance).
4. Compared against a GPU-optimised deterministic Barnes-Hut implementation.
5. Reports equal median error in up to 9.4x less time. The kernels are graphics kernels, not gravity.

## What it lets us claim
GPU Barnes-Hut is still an active research topic at a top venue in 2025. Accuracy against time (via the opening angle θ) is a legitimate axis of comparison, alongside raw speed.

## Design decision / R3 technique supported
Background for R3 #4 (Barnes-Hut traversal), and support for reporting error-vs-time curves instead of a single θ. It is not a load-balancing paper.

## Quote
"equal median error in up to 9.4x less time" (Madan et al., 2025, abstract)
