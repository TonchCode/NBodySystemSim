# frontiere2025crkhacc

**Citation:** N. Frontiere, J. D. Emberson, M. Buehlmann, E. M. Rangel, S. Habib, K. Heitmann, P. Larsen, V. Morozov, A. Pope, C.-A. Faucher-Giguère, A. Georgiadou, D. Lebrun-Grandié, A. Prokopenko. "Cosmological Hydrodynamics at Exascale: A Trillion-Particle Leap in Capability." arXiv:2510.03557 [cs.DC], 3 Oct 2025.
**Link:** https://arxiv.org/abs/2510.03557
**Verified on:** 2026-10-17 (arXiv abstract page). arXiv shows no affiliations. An Argonne news page (seen only as a search result) links the work to Argonne and collaborators as a Gordon Bell finalist. **Venue NOT confirmed.**
**Lead-group country:** USA (Argonne according to the news page; treat as likely, not confirmed).

## Summary
1. Presents CRK-HACC, a cosmological hydrodynamics and gravity code for exascale machines.
2. The techniques it names are separation of scales, GPU-resident tree solvers, in-situ analysis and tiered I/O.
3. It ran "Frontier-E", a four-trillion-particle full-sky simulation.
4. It reports 513.1 PFLOPs peak and 46.6 billion particles/s.
5. The abstract does not discuss load balancing inside a GPU.

## What it lets us claim
In the 2025 state of the art, the short-range tree solver runs entirely on the GPU and long-range gravity is split off (a TreePM/P3M-style split). This supports our choice to keep the tree on the GPU.

## Design decision / R3 technique supported
Context for TreePM/P3M scale splitting and for R3 #8: keep the work on the GPU rather than shipping it to the CPU. It gives no direct load-balancing technique.

## Quote
"GPU-resident tree solvers" (Frontiere et al., 2025, abstract)
