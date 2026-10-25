# GPU memory visualizer (Bulgarian UI)

`gpu-memory-viz.html` is a single self-contained page. Open it in a browser (it needs internet only
for the IBM Plex fonts). It steps through one leapfrog time step very slowly and shows where the data
of each body sits (registers, shared, L1, L2, VRAM, local memory, host) on a schematic RTX 4080 SUPER.

- **Real:** the physics and the algorithms. The JS ports Morton keys, the Karras radix tree,
  the multipoles (rc² = bmax²/θ²), the #4/#5 per-thread-stack traversal, the #7 warp walk
  (`__any_sync`, shared stack), and tiled all-pairs.
- **Model:** the GPU. Block-to-SM placement is in order. L1 is a scaled LRU and the SIMT
  active-lane accounting is simplified. The panel is labelled "модел".
- **Measured:** only the Nsight Compute table and the chips marked "измерено". Each one names its
  configuration. Source: `results/2026-10-22_ncu/summary.md`.

Modes: #1, #3, #4, #5, #7 step by step; #6 and #8 as short explainers.
Keys: Space plays, ← and → step, Shift+→ jumps to the end of the kernel, + and − change the speed,
T switches the theme.

Made through designer → build → critic → fixes. The design spec is `design-spec.md`.
