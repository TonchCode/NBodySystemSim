# nyland2007nbody (BACKGROUND)

**Citation:** L. Nyland, M. Harris, J. Prins. "Fast N-Body Simulation with CUDA." In H. Nguyen (ed.), *GPU Gems 3*, Chapter 31. Addison-Wesley, 2007. ISBN 9780321515261.
**Link:** https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
**Verified on:** 2026-10-17 (the NVIDIA-hosted chapter page and contributors page give the authors and affiliations; the KIT library catalogue gives the editor, publisher and ISBN). **Year note:** retail listings give August 2007, while the KIT catalogue records 2008. We use 2007.
**Lead-group country:** USA (NVIDIA; UNC Chapel Hill).
**Label:** Background classic. Not 2025–2026.

## Summary
1. All-pairs force computed in p×p tiles: each tile loads p bodies into shared memory and reuses them across p threads.
2. `float4` storage (xyz + mass in w) for coalesced loads; two `__syncthreads` per tile.
3. Unrolling the inner interaction loop cuts loop overhead.
4. On a GeForce 8800 GTX: more than 10 billion interactions/s and over 200 GFLOPS (counting 20 flops per interaction).
5. For N < 4096, splits each body's row across q threads (p·q ≤ 256). Two threads raised N = 1024 performance by 44%.

## What it lets us claim
The baseline all-pairs design, and the origin of our "20 FLOP per interaction" accounting in 01-hardware.md §5. It is also the earliest evidence that small N needs several threads per body.

## Design decision / R3 technique supported
**R3 #1** (tiled baseline), **R3 #2** (tile, block and unroll sweep), **R3 #3** (multiple threads per body at small N).

## Quote
"We therefore increase the number of active threads" (Nyland, Harris & Prins, 2007, ch. 31, small-N section)
