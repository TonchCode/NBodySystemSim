# nozal2025coexecution

**Citation:** R. Nozal, J. L. Bosque. "CPU-GPU co-execution through the exploitation of hybrid technologies via SYCL." *The Journal of Supercomputing* 81(3), article 452, 2025 (online 31 Jan 2025). DOI 10.1007/s11227-025-06963-y.
**Link:** https://doi.org/10.1007/s11227-025-06963-y
**Verified on:** 2026-10-17 via Crossref DOI metadata and abstract. The Springer page redirected to a login and was not read. Crossref lists the first author with given name "Nozal" and family name "Raúl"; we take the order to be Raúl Nozal. The metadata gives no affiliations.
**Lead-group country:** not confirmed.

## Summary
1. Extends the CoexecutorRuntime so that one co-execution API and scheduler drives backends from different technologies.
2. CPU and GPU run the same kernel on different parts of the index space at the same time.
3. The goal is performance portability for co-execution.
4. Average efficiency 0.91 and speedup 1.31 over GPU-only, across the programs studied.
5. Full text not read, so the scheduling policies and hardware were not checked.

## What it lets us claim
A 2025 measurement of what CPU+GPU co-execution of a single kernel buys over GPU-only: ≈1.3x on average, at 0.91 efficiency.

## Design decision / R3 technique supported
R3 #8, the data-parallel arm (same kernel, split index range), next to the phase split (Ragagnin et al. 2026: tree on the CPU, forces on the GPU). Strength: **transferable**; [T2] for the details, since only the abstract was read.

## Quote
"average efficiencies of 0.91 and speedups of 1.31 over using only the GPU" (Nozal & Bosque, 2025, abstract)
