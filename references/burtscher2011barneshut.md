# burtscher2011barneshut (BACKGROUND)

**Citation:** M. Burtscher, K. Pingali. "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm." Chapter 6 in *GPU Computing Gems Emerald Edition*, pp. 75–92. Elsevier, 2011. DOI 10.1016/B978-0-12-384988-5.00006-1. ISBN 9780123849885.
**Link:** https://doi.org/10.1016/B978-0-12-384988-5.00006-1 · author PDF: https://userweb.cs.txstate.edu/~mb92/papers/gcg11.pdf · code (ECL-BH v4.5): https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/
**Verified on:** 2026-10-17 (Crossref record, the author's project page, and the full chapter PDF text).
**Lead-group country:** USA (Texas State; UT Austin).
**Label:** Background classic. Not 2025–2026.

## Summary
1. The first Barnes-Hut implementation that runs entirely on the GPU, as six kernels: bounding box, tree build (lock-based insertion), summarisation, sort, force, integrate.
2. Kernel 4 sorts bodies in octree in-order, which places spatially close bodies next to each other.
3. The chapter says this sorting eliminates most thread divergence and speeds up the force kernel by an order of magnitude.
4. The force kernel makes each warp traverse the union of its threads' tree prefixes, using a warp vote (`__all`). One thread per warp fetches node data into shared memory.
5. 5M bodies on a Quadro FX 5800: 5.2 s per step, 74x faster than an optimised serial Xeon code.

## What it lets us claim
On GPUs, ordering bodies spatially is the single most important optimisation for Barnes-Hut. Warp-uniform traversal (the warp vote) removes divergence by having every thread in a warp do the union of the work.

## Design decision / R3 technique supported
**R3 #4 / #5 / #7.** Important: this "reference" GPU Barnes-Hut code already combines sorting (#5) and warp-cooperative traversal (#7). Our naive variant (#4) must therefore leave out both deliberately to show the imbalance, and the chapter's claimed order-of-magnitude gain is a falsifiable prediction for our #4 → #5 measurement.

## Quote
"we group similar work together to minimize divergence" (Burtscher & Pingali, 2011, chapter abstract)
