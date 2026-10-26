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

Layout, top to bottom:

- the step caption;
- the memory path, one box per level from host to registers, with arrows for this step's reads
  (above) and writes (below);
- the galaxy, whose caption explains what its colours mean in this step;
- one detail panel whose tab follows the step: chip, one SM, tree + warp, or pipeline;
- the model counters next to the measured table.

The first 3 traversal iterations are split into sub-steps. Later ones are one step each, unless
"всяка итерация подробно" is ticked. A link like `gpu-memory-viz.html#m5-s40` opens mode 5 at
step 40.
Keys: Space plays, ← and → step, Shift+→ jumps to the end of the kernel, + and − change the speed,
T switches the theme.

Made through designer → build → critic → fixes. The design spec is `design-spec.md`.
Second pass (2026-10-26): self-review in the browser, layout rebuilt around the memory path.
