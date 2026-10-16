# shetty2026gpuqueues

**Citation:** P. P. Shetty, T. R. W. Scogland, W. Feng. "Scalable Concurrent Queues for GPU." arXiv:2606.01693 [cs.DC], v1, 1 Jun 2026. Preprint.
**Link:** https://arxiv.org/abs/2606.01693
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliations: Virginia Tech (Shetty, Feng); Lawrence Livermore National Laboratory (Scogland).
**Lead-group country:** USA.

## Summary
1. Argues that SIMT execution and atomic contention change how concurrent queues should be designed on GPUs.
2. Presents three linearizable queues: G-WFQ-YMC (wait-free, segmented), G-LFQ (bounded lock-free, wave-batched), and G-WFQ (bounded wait-free, 64-bit CAS).
3. Evaluated with microbenchmarks, BFS (against Gunrock) and a persistent wavefront ray tracer.
4. G-LFQ had the best balanced throughput and beat Gunrock on BFS. Stream compaction sometimes beat every queue.
5. **Tested on AMD MI210 / MI300A only, not NVIDIA.**

## What it lets us claim
A naive global atomic queue is not the ceiling. Batching enqueue and dequeue per wave (warp) is the key trick, and stream compaction is a competitive alternative. The results do not carry over to Ada without our own measurements.

## Design decision / R3 technique supported
R3 #6, the queue design inside the persistent-threads variant: batch the queue operations per warp, and add a stream-compaction baseline for comparison.

## Quote
"wave-batched fast paths to maximize throughput" (Shetty et al., 2026, abstract)
