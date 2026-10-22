# Predictions: #7 tree reads through the read-only path (`__ldg`), 2026-10-22

Written **before any timing run** of this experiment (rule 7). Branch `exp/ldg-warpwalk`.

## Question
`bhWarpKernel` (#7) gets its pointers through the `TreeView` struct without `__restrict__`, so
its tree loads compile to plain `LDG.E.128`. #1 and #5 get `LDG.E.128.CONSTANT` (the
non-coherent read-only path) (ch.5, §5.2, from `cuobjdump`). Does the read-only path speed up #7?

## The change (one variable)
Inside `warpWalk` only, the reads of `nodeCom`, `nodeRc2`, `nodeInfo` and `sortedPos` go through
`__ldg()`. The arithmetic and summation order are unchanged, so the accelerations must be
**bitwise identical** to `bh_warp` on main. Every other kernel's SASS must be unchanged except
the kernels that inline `warpWalk` (`bhWarpKernel`, the #6 warp-walk persistent arm and the #8
GPU part).

## Prediction (with reasoning)
| ID | Item | Range |
|---|---|---|
| G1 | SASS of `bhWarpKernel<false,false,false>`: tree loads become `LDG.E.*.CONSTANT` | yes |
| G2 | registers stay at 40, 0 spills | yes |
| G3 | accelerations bitwise equal to main `bh_warp` (bench validation / selftest) | yes |
| G4 | traversal-time ratio main / exp at N = 1M, Plummer, θ = 0.5, leaf 16, block 64 | **0.98–1.05** |
| G5 | same at N = 262,144 | **0.98–1.05** |

**Why a small effect.** On sm_89 the L1 and the read-only (texture) path are the same unified
L1 data cache. #7's loads are warp-uniform (one address per warp), so they are already a single
broadcast wavefront. #7 was measured as latency-bound on its dependent pop chain, not L1-bound
(×1.12 from the full L1, D-026). What `.CONSTANT` can change is cache policy and whether
the compiler may reorder or batch loads past the global store `v.acc[body]`. That store happens
only after the loop, so little reordering freedom is gained. Expected gain: 0–3 %. A loss is
possible but unlikely.

## Protocol
Idle PC, after the #8 phase-2 run has finished (no overlap). Same N, IC, θ, leaf and block for
both builds. A/B/B/A passes of `--runs 11` each; the ratio is the mean of the two pass medians
per binary. Record background load.
