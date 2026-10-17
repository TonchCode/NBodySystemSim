# zhang2025hytis

**Citation:** Z. Zhang, H. Wang, H. Xu, D. Yang, X. Zhou, D. Cheng. "HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality." In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)*, St. Louis, MO, Nov 2025, pp. 1604–1618. DOI 10.1145/3712285.3759771.
**Link:** https://doi.org/10.1145/3712285.3759771
**Verified on:** 2026-10-17 via Crossref DOI metadata (title, authors, affiliations, venue, pages) and the Semantic Scholar abstract. **The ACM DL page returned 403**, so the full text was not read. Affiliations (Crossref): Zhang, Wang, Xu and Cheng at Wuhan University; Yang at NVIDIA Corporation; Zhou at the University of Macau.
**Lead-group country:** China (Wuhan University).

## Summary (abstract as summarized by Semantic Scholar; full text not read)
1. Targets **wave quantization**: the last, partially filled wave of blocks leaves SMs idle.
2. Two-level tile scheduling: the first level maximizes utilization in full waves, and the second uses finer tiles in the partial wave.
3. Offline profiling picks efficient micro-kernels, so runtime tuning is cheap.
4. Models how the tile layout affects L2 data movement.
5. Reports up to 1.95x (H100) and 2.08x (A100) over cuBLAS.

## What it lets us claim
SC '25 (CCF-A) evidence from a Chinese group that wave quantization is a first-order loss on current NVIDIA GPUs, and that the fix is a finer-grained tail wave. This is the mechanism behind R3 #2 (smaller blocks) and R3 #3 (more, smaller work items via q).

## Design decision / R3 technique supported
R3 #2 and #3. A natural extension of #3 is to use q > 1 only for the tail wave (HyTiS's two-level idea). Strength: **transferable**. GEMM tiles and all-pairs tiles are both regular, equal-cost work items, so this is the closest analogue found.

## Quote
None captured verbatim (full text not accessible). Paraphrase only, with citation.
