# thaler2026sfcneighbor

**Citation:** F. Thaler, S. Keller. "GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout." arXiv:2602.19873 [cs.CE; astro-ph.IM, cs.DS], v1 23 Feb 2026. Comments: "Accepted at IPDPS 2026".
**Link:** https://arxiv.org/abs/2602.19873
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliation (both authors): "Swiss National Supercomputing Centre (CSCS)", "Zürich, Switzerland".
**Lead-group country:** Switzerland.

## Summary
1. GPU neighbor lists for particle codes (SPH, Lennard-Jones) built on a Hilbert SFC memory layout.
2. Consecutive particles along the curve are close in space on average, so clusters are simply runs of consecutive particles. No separate reordering step is needed, unlike in GROMACS.
3. Supports a different interaction radius per particle.
4. Runs on NVIDIA GH200 and AMD MI300A; the Evrard collapse test scales to 1024 GH200 GPUs on Alps.
5. Performance is similar to GROMACS in a neighborhood pass. The SFC particle order also sped up a LAMMPS-like kernel, but the paper does not analyse why.

## What it lets us claim
2026 evidence (IPDPS, CCF-B) from an astrophysical particle code that Hilbert ordering makes runs of consecutive particles spatially compact on GPUs. A warp or block of consecutive bodies therefore works on nearby space, which is the premise of R3 #5.

## Design decision / R3 technique supported
R3 #5 (Morton/Hilbert sort). This paper chose Hilbert for compactness; stucker2026jzfmm chose Morton. Neither compares the two on traversal divergence, so the Hilbert-vs-Morton gap in INDEX stands, and our measurement would be new. Strength: **transferable** (neighbor search, not a gravity tree walk, but the same locality argument).

## Quote
"the average compactness of the Hilbert curve" (Thaler & Keller, 2026, arXiv HTML)
