# maeda2026gtap

**Citation:** Y. Maeda, K. Taura. "GTaP: A GPU-Resident Fork-Join Task-Parallel System with a Pragma-Based Interface." arXiv:2604.05982 [cs.DC], v2, 24 Aug 2026 (v1 7 Apr 2026). Preprint.
**Link:** https://arxiv.org/abs/2604.05982
**Verified on:** 2026-10-17 (arXiv abstract + HTML full text). Affiliation: Department of Information and Communication, The University of Tokyo.
**Lead-group country:** Japan.

## Summary
1. Runs fine-grained fork-join task programs entirely inside one **persistent GPU kernel**.
2. Scheduling stays on the GPU and uses **random work stealing**: one deque per block (block mode) or per warp (thread mode). Owners pop from the tail; thieves steal from the head.
3. Optional Divergence-Aware Queueing splits the queues to reduce warp divergence.
4. Evaluated on one H100 (GH200 node, Miyabi-G). It includes an **FMM dual-tree traversal** benchmark.
5. In the traversal phase (N = 5×10⁷, θ = 0.2) it is about 1.5x faster than a GPU worklist baseline, but only about 1.05x end-to-end. The abstract says "up to 1.6x".

## What it lets us claim
The most directly relevant 2026 evidence for R3: on a modern NVIDIA GPU, a persistent kernel with work stealing beats a centralized global queue on tree traversal. The end-to-end gain is small when other phases dominate, which is a caution for how we report results.

## Design decision / R3 technique supported
**R3 #6 (persistent threads + queue) vs R3 #7 (warp/block-level work stealing).** It supports implementing both and expecting #7 to win at high contention. It also argues for measuring phase time and total time separately.

## Quote
"Work stealing outperforms a centralized global queue, which quickly becomes a scalability bottleneck." (Maeda & Taura, 2026, arXiv HTML)
