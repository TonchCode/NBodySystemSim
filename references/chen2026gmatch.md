# chen2026gmatch

**Citation:** W. Chen, S. Sun, C. Chen, Y. Hu, Y. Hu, M. Guo. "gMatch: Fine-Grained and Hardware-Efficient Subgraph Matching on GPUs." arXiv:2604.10601 [cs.DB], v1 12 Apr 2026. Preprint (17 pages), no venue given.
**Link:** https://arxiv.org/abs/2604.10601
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliations: Weitian Chen and Shixuan Sun at "Shanghai Jiao Tong University"; Cheng Chen, Yongmin Hu and Yingqian Hu at "ByteDance Inc"; Minyi Guo at "Guizhou University".
**Lead-group country:** China (SJTU).

## Summary
1. GPU subgraph matching is an irregular DFS over a search tree, the same shape as a tree walk with data-dependent depth.
2. A BFS phase first builds a large pool of partial matches (τ = 10⁶), and warps take tasks from it with an **atomic counter**.
3. When the pool runs out, **warp-level work stealing** starts. An idle warp registers in a per-block queue in shared memory, and a busy warp splits its stack and hands over half. The same scheme repeats across blocks through global memory.
4. The authors call stealing supplementary, because the large initial pool keeps it rare.
5. The default testbed is an **RTX 4090 (Ada)**. The main text has no ablation that isolates stealing; the τ study is in an appendix that was not read.

## What it lets us claim
A 2026 Chinese paper on our GPU family that has both R3 #6 (atomic-counter task pool) and R3 #7 (warp-level stealing, half-split) in one system. Its design lesson: a large, fine-grained initial pool does most of the balancing.

## Design decision / R3 technique supported
R3 #6 and #7. The prediction it supports: in Barnes-Hut, per-body cost probably varies by about an order of magnitude, far less than the long tails of subgraph mining. So #6 with fine work items should capture most of the gain, and #7 should add little. Strength: **transferable** (graph DFS, not N-body; same Ada hardware).

## Quote
"idle warps receive half of the work from busy warps" (Chen et al., 2026, arXiv HTML)
