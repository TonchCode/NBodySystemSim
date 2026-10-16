# nagarajan2025rtbarneshut

**Citation:** V. Nagarajan, R. Gangaraju, K. Sundararajah, A. Pelenitsyn, M. Kulkarni. "RT-BarnesHut: Accelerating Barnes-Hut Using Ray-Tracing Hardware." *PPoPP '25*, ACM, Las Vegas, pp. 43–56, 2025. DOI 10.1145/3710848.3710885.
**Link:** https://doi.org/10.1145/3710848.3710885 · https://ppopp25.sigplan.org/details/PPoPP-2025-Main-Conference-1/36/RT-BarnesHut-Accelerating-Barnes-Hut-Using-Ray-Tracing-Hardware
**Verified on:** 2026-10-17 (Crossref DOI record, PPoPP'25 program page, Virginia Tech repository record). The ACM DL page itself returned 403.
**Lead-group country:** USA (Purdue; UT Austin; Virginia Tech).

## Summary
1. Argues that tree traversal maps poorly onto ordinary GPU shader (CUDA) cores.
2. Reformulates Barnes-Hut as a ray-tracing problem and implements it with NVIDIA OptiX.
3. Uses RT cores (hardware BVH traversal) to make the node-opening decisions.
4. The abstract claims it outperforms current state-of-the-art GPU Barnes-Hut implementations.
5. The abstract gives neither the speedup range nor the GPU model. A search snippet claimed 2x–40x, which is **not verified; do not cite it**.

## What it lets us claim
A peer-reviewed 2025 result at PPoPP (a CCF-A venue) showing that tree traversal is a poor fit for SIMT cores. That irregularity is the subject of our load-balancing chapter. The RTX 4080 SUPER has 80 RT cores [S: nvidia2023adawhitepaper], so this route exists on our hardware.

## Design decision / R3 technique supported
Motivation for R3 #4–#7. It could become an optional extra technique (offload traversal to the RT cores), but that needs OptiX rather than plain CUDA compute. Scope decision for the architect.

## Quote
"outperforms current state-of-the-art GPU-based implementations" (Nagarajan et al., 2025, abstract)
