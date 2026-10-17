# huang2026weave

**Citation:** Z. Huang, Y. Zhou, C. Zhu, P. Yu, Z. Liu, J. Liu, S. Zhang, X. Tang, H. Yan, X. Luo, M. Guo, X. Lin, Y. Yu, G. Yang, L. Zhang, S. Sun, J. Leng. "Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap." arXiv:2609.21483, v3 25 Sep 2026 (v1 18 Sep 2026). Preprint.
**Link:** https://arxiv.org/abs/2609.21483
**Verified on:** 2026-10-17 (arXiv abstract + HTML v3). Affiliations listed: "Shanghai Jiao Tong University, Shanghai, China"; "National University of Singapore, Singapore"; "Fudan University, Shanghai, China"; "Alibaba Group, China". The per-author mapping was not extracted.
**Lead-group country:** China (SJTU is the first listed affiliation).

## Summary
1. A persistent MoE megakernel on 4x H100 SXM that splits SMs between communication and computation.
2. The split is recomputed per layer at runtime, once routing makes the work volumes known.
3. Inside the kernel, each SM claims its next GEMM tile with atomicAdd on a global counter.
4. Communication SMs join the computation once dispatch is done ("bubble stealing").
5. 2.89x geomean MoE-layer speedup, 1.33x end to end.

## What it lets us claim
A 2026 Chinese source for the exact R3 #6 mechanism: persistent blocks pull work items from a global atomic counter, chosen over a fixed (static) SM assignment.

## Design decision / R3 technique supported
R3 #6 (persistent threads + global atomic work queue). Strength: **transferable**. It is an ML megakernel, but the work-claiming mechanism is identical to what #6 needs for tree-traversal work items.

## Quote
"each SM claims its next tile via atomicAdd on a global counter" (Huang et al., 2026, arXiv HTML)
