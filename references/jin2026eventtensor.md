# jin2026eventtensor

**Citation:** H. Jin, B. Hou, G. Wang, R. Lai, J. Chen, Z. Ye, Y. Cai, Y. Dong, X. Cheng, Z. Zhang, Y. Zhao, Y. Huang, L. Yang, J. Jiang, G. Oliaro, J. Ji, X. Miao, V. Grover, T. C. Mowry, Z. Jia, T. Chen. "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel." arXiv:2604.13327 [cs.DC], v2, 21 Apr 2026. Accepted to MLSys 2026, per the arXiv comments.
**Link:** https://arxiv.org/abs/2604.13327
**Verified on:** 2026-10-17 (arXiv abstract page). arXiv shows no affiliations. The author list matches the CMU / NVIDIA group (Jia, Chen, Mowry, Grover), but that is **not confirmed** from the landing page.
**Lead-group country:** USA (likely; unconfirmed).

## Summary
1. Targets launch overhead and coarse synchronization between kernels in GPU workloads, mainly LLM inference.
2. Existing megakernels (fused persistent kernels) handle dynamic shapes and data-dependent work poorly.
3. Event Tensor encodes dependencies between tiled tasks, including data-dependent ones.
4. Its compiler applies both static and dynamic scheduling transformations to produce persistent kernels.
5. Reports state-of-the-art serving latency and lower warm-up overhead. The abstract does not compare static and dynamic scheduling separately.

## What it lets us claim
In 2026, persistent kernels with on-GPU scheduling are mainstream (MLSys). The static-vs-dynamic choice is treated as a compiler decision, which matches our R3 framing.

## Design decision / R3 technique supported
Context for R3 #6 (persistent threads; static vs dynamic queue). Weak for N-body specifically.

## Quote
"static and dynamic scheduling transformations to generate high-performance persistent kernels" (Jin et al., 2026, abstract)
