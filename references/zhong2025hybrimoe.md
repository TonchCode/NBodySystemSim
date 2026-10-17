# zhong2025hybrimoe

**Citation:** S. Zhong, Y. Sun, L. Liang, R. Wang, R. Huang, M. Li. "HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference." arXiv:2504.05897 [cs.LG], v1 8 Apr 2025. Comments: "Accepted by DAC 25" (not checked against the DAC proceedings).
**Link:** https://arxiv.org/abs/2504.05897
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliations: Institute for Artificial Intelligence, Peking University; School of Integrated Circuits, Peking University; Institute of Electronic Design Automation, Peking University (Wuxi); Beijing Advanced Innovation Center for Integrated Circuits; School of Computer Science and Engineering, Beihang University.
**Lead-group country:** China (Peking University).

## Summary
1. MoE inference with the experts split between CPU and GPU.
2. A warmup phase profiles CPU speed, GPU speed and transfer latency.
3. The split minimizes max(T_CPU, T_GPU), so both sides should finish together.
4. Before each layer, it simulates the CPU, GPU and transfer timelines and greedily assigns work to the one that finishes first.
5. Hardware: RTX A6000 + Xeon Gold 5220R (limited to 10 cores).

## What it lets us claim
A 2025 Chinese source for profile-then-split CPU+GPU balancing whose objective is equal finish times, which is the natural objective for R3 #8.

## Design decision / R3 technique supported
R3 #8. Implementation hint: calibrate per-device rates at startup (separately for P-cores and E-cores), then split the work so that all of them finish together. Strength: **analogy**: ML inference has a few large tasks, while the N-body split would cover many cheap interactions. The paper says its objective ignores when transfers finish. We should not copy that gap, since positions cross PCIe every step.

## Quote
"a dynamic intra-layer scheduling strategy to balance workloads across CPU and GPU" (Zhong et al., 2025, arXiv abstract)
