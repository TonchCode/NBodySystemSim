# liu2025swphotons

**Citation:** Z. Liu, Q. Wang, X. Lv, X. Liu, H. Meng, G. Zhu, L. Gao, H. Fu, G. Yang. "swPHoToNs: Toward trillion-body-scale cosmological N-body simulations on Sunway TaihuLight supercomputer." *Engineering Reports* 7(3), e12640, 2025. DOI 10.1002/eng2.12640.
**Link:** https://doi.org/10.1002/eng2.12640
**Verified on:** 2026-10-17 via the Crossref DOI record (the Wiley page returned 403). Affiliations: Tsinghua University; National Supercomputing Center in Wuxi; NAOC, Chinese Academy of Sciences; China Ship Scientific Research Center; Huirong Electronic Systems.
**Lead-group country:** China.
**⚠ Date caveat:** Crossref gives a first online date of **19 Feb 2023**. 2025 is only the issue date. If we cite it as recent work, say "published online 2023, issue 2025".

## Summary
1. PM + FMM cosmological N-body code (PHoToNs) ported to Sunway TaihuLight (SW26010 many-core, **not a GPU**).
2. Multilevel domain decomposition with dynamic load balancing.
3. Tree traversal is pipelined with the gravity calculation.
4. Compute-kernel and MPI optimisations for the many-core architecture.
5. Up to 1.6 trillion particles, 56.3 PFlops sustained, 80.9% weak-scaling efficiency, 44.9% computational efficiency.

## What it lets us claim
A leading Chinese HPC group (Tsinghua / NSCC-Wuxi / NAOC) treats dynamic load balancing and pipelining of traversal with computation as first-class requirements for tree/FMM gravity. Its balancing is between nodes, not inside a GPU.

## Design decision / R3 technique supported
General support for R3 #6 (dynamic rather than static assignment) and for overlapping traversal with force evaluation. The hardware differs from ours, so cite it as an analogy only.

## Quote
None captured. The publisher page was blocked (403) and Crossref returned only a paraphrase of the abstract. Fetch the PDF before quoting.
