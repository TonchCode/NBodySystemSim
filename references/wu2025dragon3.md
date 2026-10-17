# wu2025dragon3

**Citation:** K. Wu, P. Cho, R. Spurzem, L. Wang, F. Flammini Dotti, V. Amiri. "DRAGON-III simulation: modelling million-body globular and nuclear star clusters." arXiv:2510.03933 [astro-ph.GA], v1 4 Oct 2025. Comments: accepted for the proceedings of IAU Symposium 398 & MODEST-25.
**Link:** https://arxiv.org/abs/2510.03933
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliations: ARI, Universität Heidelberg (Wu, Cho, Spurzem, Amiri); NAOC, Chinese Academy of Sciences, and KIAA, Peking University (Spurzem); School of Physics and Astronomy, Sun Yat-sen University, Zhuhai (L. Wang); NYU Abu Dhabi (Flammini Dotti).
**Lead-group country:** Germany (Heidelberg), with Chinese co-affiliations (NAOC, KIAA-PKU, SYSU).

## Summary
1. Simulations of million-body globular and nuclear clusters over 10 Gyr.
2. The main code is NBODY6++GPU (direct N-body with GPU force evaluation).
3. The CUDA part of NBODY6++GPU was ported to HIP.
4. Targets: DCU-accelerated clusters in China and the AMD-GPU LUMI system.
5. This 4-page proceedings paper gives no performance or load-balancing figures.

## What it lets us claim
Context for the "preferably Chinese" search. The 2025 Chinese-involved GPU N-body work found is a port of an existing direct code to Chinese DCU accelerators, not new GPU load-balancing methods.

## Design decision / R3 technique supported
None directly; background for the R4 narrative only. Strength: **context**.

## Quote
"We have ported the CUDA code in NBODY6++GPU to HIP software stacks" (Wu et al., 2025, arXiv HTML)
