# 00 — Course Requirements (what the professor actually asked for)

Source: the audio transcript of the consultation with the professor (Bulgarian), plus a
summary of it. Both files are in `C:\Users\petya\Downloads\`. The transcript is
speech-to-text and partly garbled. Items I inferred rather than heard clearly are marked
*(inferred)*.

**Topic assigned to Toncho: the N-body problem.** (Another student was
discussing a shortest-path-in-graph topic. Two different projects, two different devices.)

---

## Hard requirements

| # | Requirement | How we satisfy it | Status |
|---|---|---|---|
| R1 | **Describe the GPU architecture** in detail: multiprocessors (SMs), memory and so on. "I want your GPU's architecture, well written." | [`01-hardware.md`](01-hardware.md), then a polished section in the final doc | 🟡 data captured, prose pending |
| R2 | **Parallel variants of the algorithm suited to *your* GPU.** Pick the algorithm and work out how it is best split across your own device. | `cuda-architect` agent, `03-design.md` | ⬜ |
| R3 | **Different load-balancing techniques on the GPU.** "I want different techniques for distributing the GPU's workload." This is the core of the grade. | Several kernels/strategies, each benchmarked; see the candidate list below | ⬜ |
| R4 | **Recent publications (2025–2026)** backing the algorithm and decisions. Well-known authors in the field, **"preferably Chinese"**. | `literature-scout` agent → `references/` | ⬜ |
| R5 | **Documentation from recent drivers / NVIDIA docs** for the specific device *(inferred: "documentation from the latest drivers")* | CUDA 13.4 Programming Guide, Ada tuning guide, Ada whitepaper | ⬜ |
| R6 | **Presentation** at a later consultation ("next time you'll come and show some presentation") | Slides + live demo of the visualization | ⬜ |

## Soft / contextual

- The professor suggested using AI agents in distinct roles, **scientist / critic /
  programmer**. That is the origin of `.claude/agents/`.
- **Counts as a separate grade** ("this can become your course project, a separate grade"),
  and is taken seriously.
- **Could continue into the diploma thesis** next semester. Optional, not required.
- The assistant runs the lab exercises and decides on that side. Check her requirements
  separately.

## Load-balancing techniques: candidate list (R3)

To be validated by `literature-scout` and designed by `cuda-architect`. Each technique needs a
*why it matters for N-body* line and a measurement.

1. **Static 1-thread-per-body, all-pairs, shared-memory tiling.** The baseline; perfectly
   balanced by construction.
2. **Block-size / tile-size / unroll sweep** against occupancy (1536 threads/SM, 24 blocks/SM).
   This is load balancing inside each SM.
3. **Multiple threads per body (j-loop split + reduction).** Matters at small N, when
   N < 80 SMs × resident threads and the GPU would otherwise be starved.
4. **Barnes-Hut tree traversal: naive one-thread-per-body.** Shows the imbalance (warp
   divergence, uneven tree depth).
5. **Space-filling-curve sorting (Morton/Hilbert)** so warps hold spatially close bodies and
   traverse similar paths.
6. **Persistent threads + global atomic work queue** (dynamic scheduling instead of static
   grid assignment).
7. **Warp-level work stealing / cooperative traversal.**
8. **Heterogeneous CPU+GPU split.** The i5-13600K has P-cores and E-cores, so the split
   itself needs balancing.

---

## Open questions for the professor / assistant

- Language of the documentation and presentation: Bulgarian or English?
- Is there a required document format (template, page count)?
- Is a visualization expected, or is it our own addition?
- Deadline for the presentation?
