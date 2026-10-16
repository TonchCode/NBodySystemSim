# dale2026warpdivergence

**Citation:** A. Dale. "Characterizing Warp Divergence from Pascal to Blackwell." arXiv:2607.23402 [cs.AR], v1, 26 Jul 2026. 6 pages. Preprint by a single author, not peer-reviewed.
**Link:** https://arxiv.org/abs/2607.23402
**Verified on:** 2026-10-17 (arXiv abstract page). No affiliation shown.
**Lead-group country:** Unknown.

## Summary
1. Microbenchmarks divergence on Pascal, Ampere, Hopper and Blackwell. **Ada is not tested.**
2. k divergent paths serialize linearly, with no super-linear reconvergence penalty.
3. Warp execution efficiency falls as 32/k, independent of occupancy. Predication removes the cost.
4. The same cost model holds on Pascal, i.e. before Independent Thread Scheduling.
5. Reconvergence mechanisms changed between generations, but the cost stays predictable.

## What it lets us claim
A simple, recent model for the divergence penalty in R3 #4: efficiency ≈ 32/k, where k is the number of distinct traversal paths per warp. Ada sits between Ampere and Hopper, so the model is plausible for sm_89 **but must be measured** ([M] needed). Thin evidence (one author, 6 pages): tag [T2] if used.

## Design decision / R3 technique supported
Gives the analysis for R3 #4 (naive traversal) vs R3 #5 (sorted). We can measure average active threads per executed instruction in Nsight Compute and compare it with 32/k. Check the exact metric name in the Nsight Compute 2026.3 docs, which are not yet catalogued.

## Quote
None captured verbatim. The abstract was received only as a summary.
