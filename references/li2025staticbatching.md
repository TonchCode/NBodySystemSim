# li2025staticbatching

**Citation:** Y. Li, Y. Li, J. Zhang, B. Chen, X. Chen, L. Duan, Y. Jin, Z. Li, X. Liu, H. Wang, W. Wang, Y. Wang, J. Yang, P. Zhang, L. Zheng, W. Yu. "Static Batching of Irregular Workloads on GPUs: Framework and Application to Efficient MoE Model Inference." arXiv:2501.16103 [cs.DC], v1, 27 Jan 2025. Preprint, no venue given.
**Link:** https://arxiv.org/abs/2501.16103
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliation: all authors at Alibaba Group.
**Affiliation re-verified 2026-10-17 (round 3, critic #8), from https://arxiv.org/html/2501.16103v1:** each of the 16 authors carries the footnote "Affiliation: Alibaba Group". Y. Li and Y. Li are marked "Equal contribution". The contact line reads `{lyh238099,lyf383659,jiejing.zjj,wenyuan.ywy}@alibaba-inc.com`. No university affiliation appears. The abstract page does not show affiliations, which is why the critic could not confirm it there.
**Scope caveat (critic #8):** this is MoE inference, not N-body. Cite it only as transferable evidence for the static, count-based mapping arm of R3 #6/#3, never as N-body evidence.
**Lead-group country:** China (industry).

## Summary
1. A general framework that batches many irregular tasks into **one** GPU kernel.
2. The host computes a prefix sum of per-task tile counts (TilePrefix). In the kernel, each block finds its task with a warp vote + popcount over TilePrefix.
3. The mapping is precomputed, so no runtime scheduler or atomics are needed for task assignment.
4. Contrasted with grouped GEMM's in-kernel dynamic tile scheduling, which adds overhead.
5. Reaches up to 91% of peak Tensor Core throughput on H800 and 95% on H20 for MoE inference.

## What it lets us claim
When per-task cost is known before launch, a static prefix-sum mapping can beat dynamic scheduling. This is the counter-hypothesis to persistent threads, and a 2025 Chinese source for it. Its domain is ML, not N-body.

## Design decision / R3 technique supported
A strong **alternative arm for R3 #6**: per-cell interaction counts → prefix sum → block-to-work mapping, against an atomic queue. It also fits R3 #3 (assigning several threads to heavy bodies or cells by count).

## Quote
"We propose a general framework for statically batching irregular workloads on GPUs." (Li et al., 2025, arXiv HTML)
