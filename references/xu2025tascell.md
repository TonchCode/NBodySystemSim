# xu2025tascell

**Citation:** J. Xu, T. Hiraishi, Z. Bai, K. Fukazawa, M. Yasugi. "Implementation of Hybrid CPU-GPU Programs in a Backtracking-based Load Balancing Framework." *Journal of Information Processing* 33 (2025), pp. 494–506. DOI 10.2197/ipsjjip.33.494.
**Link:** https://www.jstage.jst.go.jp/article/ipsjjip/33/0/33_494/_article
**Verified on:** 2026-10-17 (J-STAGE landing page). Affiliations: Xu, Graduate School of Informatics, Kyoto University; Hiraishi, Kyoto Tachibana University; Bai, RIKEN R-CCS; Fukazawa, RIHN and ACCMS, Kyoto University; Yasugi, Kyushu Institute of Technology.
**Lead-group country:** Japan.

## Summary
1. Extends the Tascell work-stealing framework (backtracking-based load balancing) to GPUs.
2. Each worker has both a CPU and a GPU implementation of a task.
3. A worker picks CPU or GPU for each task from the task size and GPU availability.
4. On a workstation, the hybrid version beats both CPU-only and GPU-only.
5. 2D stencil: up to 12.9% over GPU-only and 25.2% over CPU-only.

## What it lets us claim
2025 evidence that dynamic (work-stealing) CPU+GPU splits pay off on a single workstation. The gain over GPU-only is modest (≈13%), which is the right expectation for R3 #8 on a GPU that is far faster than the CPU.

## Design decision / R3 technique supported
R3 #8 (heterogeneous CPU+GPU split): a dynamic alternative to a static profiled ratio. Strength: **transferable** (stencil code at workstation scale; the task-size rule maps to "small subtrees on the CPU").

## Quote
"the hybrid approach outperformed both CPU-only and GPU-only configurations" (Xu et al., 2025, abstract)
