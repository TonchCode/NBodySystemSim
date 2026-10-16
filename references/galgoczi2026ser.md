# galgoczi2026ser

**Citation:** G. Galgóczi. "Mitigating warp divergence in GPU optical photon Monte Carlo: an order of magnitude speedup with Shader Execution Reordering." arXiv:2608.21396 [physics.ins-det], v1, 3 Aug 2026. Preprint.
**Link:** https://arxiv.org/abs/2608.21396
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliation: Brookhaven National Laboratory.
**Lead-group country:** USA.

## Summary
1. One-thread-per-photon kernels diverge because photon lifetimes vary widely, so finished lanes sit idle.
2. Uses Shader Execution Reordering (SER, via OptiX `optixReorder`) to regroup surviving photons into coherent warps.
3. Measured on a single **RTX 4090 (Ada)**, the same architecture as our GPU.
4. Propagation-kernel cycles drop about 16.6x and end-to-end time about 14.7x, with bit-identical results.
5. About 3.2x comes from launch-configuration changes and about 5.1x from the reordering itself.

## What it lets us claim
On Ada, regrouping work by its future behaviour gives large gains for a "one thread per item, uneven lifetimes" workload, which has the same shape as naive Barnes-Hut. SER itself is an Ada hardware feature, but the paper uses it only through OptiX. For CUDA compute we would have to reproduce the effect in software (sort or compact).

## Design decision / R3 technique supported
Ada-specific evidence for R3 #5 (reorder so warps are coherent) and R3 #6 (compacting live work). It also warns us to separate the launch-configuration gain from the reordering gain when we report results.

## Quote
"from the Ada generation onward" (Galgóczi, 2026, arXiv HTML, on SER hardware support)
