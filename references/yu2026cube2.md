# yu2026cube2

**Citation:** H.-R. Yu, B.-H. Chen, K. Xu, M.-J. Sheng, J. Han, Y. Jing, H. Cui. "CUBE2: A parallel N-body simulation code for scalability, accuracy, and memory efficiency." *Science China Physics, Mechanics & Astronomy* 69(6), 269511, 2026. DOI 10.1007/s11433-025-2926-0. arXiv:2512.12629.
**Link:** https://doi.org/10.1007/s11433-025-2926-0 · https://arxiv.org/abs/2512.12629
**Verified on:** 2026-10-17 (Crossref DOI record + arXiv abstract + arXiv HTML). Affiliations: Xiamen University; Shanghai Jiao Tong University; University of Pennsylvania; Dawning Information Industry.
**Lead-group country:** China.

## Summary
1. A multi-level particle-mesh (PM) code with a particle-particle (PP) short-range correction, which puts it in the P3M family.
2. It is a **CPU code** (Coarray Fortran + nested OpenMP). The authors say GPU offload is planned but not yet done.
3. Positions and velocities are compressed to 1–2-byte integers relative to the mesh ("Information Optimized Storage").
4. Load balancing: subtiles are sorted by maximum coarse-grid density and handed out heaviest first.
5. The paper reports that this cuts load imbalance by 11.4%, reaching 99.6% load balance. Runs reached 6144^3 particles.

## What it lets us claim
A leading Chinese cosmology group (SJTU/Xiamen) balances P3M work by sorting work units by estimated cost and scheduling the heaviest first (longest-processing-time-first). That is a cost-model approach, which we can contrast with runtime dynamic approaches.

## Design decision / R3 technique supported
R3 #8, the CPU side of the CPU+GPU split: cost-sorted assignment matters on the i5-13600K with its P- and E-cores. It is also an analogy for R3 #6 (order the work queue by estimated cost). It is **not** GPU evidence.

## Quote
"written in Coarray FORTRAN (CAF)" (Yu et al., 2026, arXiv HTML)
