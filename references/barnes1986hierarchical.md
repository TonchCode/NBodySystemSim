# barnes1986hierarchical (BACKGROUND)

**Citation:** J. Barnes, P. Hut. "A hierarchical O(N log N) force-calculation algorithm." *Nature* 324(6096), 446–449, December 1986. DOI 10.1038/324446a0.
**Link:** https://doi.org/10.1038/324446a0
**Verified on:** 2026-10-17 (Crossref DOI record).
**Lead-group country:** Not verified. The Crossref record gives no affiliations; check the Nature page before stating one.
**Label:** Background classic. Not 2025–2026.

## Summary
1. The original tree-code paper: space is recursively subdivided into an octree of cells.
2. Distant groups of bodies are replaced by their centre of mass when they subtend a small enough angle.
3. This reduces the all-pairs O(N²) force cost to O(N log N).
4. The opening criterion (θ) controls the trade-off between accuracy and cost.
5. Every GPU tree code in this folder descends from this algorithm.

## What it lets us claim
The origin and complexity of Barnes-Hut. It is the reason a tree method beats all-pairs at large N.

## Design decision / R3 technique supported
The algorithm behind R3 #4–#7.

## Quote
Title used as the citable phrase: "A hierarchical O(N log N) force-calculation algorithm".
