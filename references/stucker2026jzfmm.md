# stucker2026jzfmm

**Citation:** J. Stücker, O. Foldal. "JZ-FMM: GPU-native differentiable N-body simulations with the Fast Multipole Method." arXiv:2609.09307 [astro-ph.CO], v1, 8 Sep 2026. Preprint, no journal reference yet.
**Link:** https://arxiv.org/abs/2609.09307 · code: github.com/jstuecker/jzfmm
**Verified on:** 2026-10-17 (arXiv abstract + HTML full text). Affiliation: Department of Astrophysics, University of Vienna.
**Lead-group country:** Austria.

## Summary
1. A GPU-native FMM: sorting, tree build, interaction lists and force evaluation all stay on the GPU.
2. Particles are sorted along a Morton (z-order) curve, and the tree is built bottom-up from the sorted order.
3. A non-recursive, level-by-level dual-tree walk stores interactions in dense index arrays.
4. One thread block per receiving parent node; sources are staged in shared memory, and no atomics are needed.
5. Benchmarked on A100 (Leonardo). The abstract reports more than 10x speedup over CPU-native and hybrid codes.

## What it lets us claim
In 2026, the standard way to make tree methods GPU-friendly is to remove recursion and branching and to order the data along a Morton curve. The paper uses no dedicated dynamic load balancer. It is a preprint, not yet peer-reviewed.

## Design decision / R3 technique supported
R3 #5 (Morton ordering) and R3 #7 (block-cooperative traversal, one block per target node). It also favours building the tree on the GPU over a CPU build, which bears on R3 #8.

## Quote
"a GPU-native differentiable implementation of the fast multipole method" (Stücker & Foldal, 2026, abstract)
