# wang2021photonsgpu (BACKGROUND, 2021)

**Citation:** Q. Wang, C. Meng. "PhotoNs-GPU: A GPU accelerated cosmological simulation code." *Research in Astronomy and Astrophysics* 21(11), 281 (2021). DOI 10.1088/1674-4527/21/11/281. arXiv:2107.14008 [astro-ph.IM], v1 28 Jul 2021 (the only version).
**Link:** https://arxiv.org/abs/2107.14008
**Verified on:** 2026-10-17 (arXiv abstract page: title, authors, journal ref and both DOIs). The author affiliations were **not** extracted. RAA is published by NAOC/CAS with IOP.
**Lead-group country:** China, inferred from the PhotoNs/NAOC lineage. The affiliation line was not fetched, so this is unconfirmed.

## Summary
1. GPU port of PhotoNs, a PM-FMM (particle mesh + fast multipole) cosmological code.
2. Interpolates the truncated-gravity special functions to make the GPU kernels cheaper.
3. Mixed-precision tests: single precision is ≈ 2x faster than double in practice and adds a small, unbiased noise to the power spectrum.
4. Much faster than CPU PhotoNs and Gadget-2.
5. Reaches a peak of 48% and an average of ≈ 35% of theoretical GPU speed with its memory-access, kernel and concurrency optimizations.

## What it lets us claim
This is the best Chinese **GPU** gravity code found, but it dates from 2021, so it is BACKGROUND only. It backs FP32 force kernels (CLAUDE.md rule 3) with a measured accuracy cost, and it is a reference point for reporting fraction of peak.

## Design decision / R3 technique supported
Background for R3 #1 (FP32 force kernel) and for the FMM context of #4–#7. The abstract has no load-balancing content. Strength: **direct** for FP32 in gravity kernels, but outside 2025–26.

## Quote
None captured verbatim this pass. Paraphrase only, with citation.
