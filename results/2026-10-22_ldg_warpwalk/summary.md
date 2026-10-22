# #7 tree reads through the read-only path (`__ldg`), 2026-10-22

Predictions: `docs/predictions/2026-10-22-ldg-warpwalk.md`, committed on the experiment branch
`exp/ldg-warpwalk` (d76d7c9) **before** any timing. Code change: `exp_ldg.patch` (one function,
`warpWalk`; four tree arrays read via `__ldg()`). The branch is not merged (D-037).

## Session
- 2026-10-22, right after the #8 phase-2 session, PC idle (GPU 0 %, 7 W before the runs, `env_before.txt`).
- Both binaries built from the same commit except the patch; `nbody_bench --kernel bh_warp --ic plummer
  --theta 0.5 --leaf 16 --block 64 --runs 11`.
- Round 1: A/B/B/A with validation (`logs/`). Round 2: 4 alternating pairs per N, `--no-validate`
  (`logs/round2/`). Traversal phase median of 11 runs per pass.

## Results
| ID | Item | Range | Measured | Verdict |
|---|---|---|---|---|
| G1 | tree loads become `.CONSTANT` | yes | 10 × `LDG.E.128.CONSTANT` + 1 × `LDG.E.CONSTANT` (main: 11 × `LDG.E.128` + 2 × `LDG.E`) (`sass_bhWarpKernel.txt`) | HIT |
| G2 | 40 registers, 0 stack | yes | REG:40 STACK:0 in both | HIT |
| G3 | accelerations identical | yes | validation errors identical to 4 digits at both N (max rel 1.234e-01 / 2.704e-01, rms 4.591e-03 / 4.712e-03); bitwise equality **not checked directly** | HIT (indirect) |
| G4 | main/exp traversal, 1M | 0.98–1.05 | **1.000** (round 2, median of 4 per binary; pairs 0.996–1.012) | HIT |
| G5 | main/exp traversal, 262,144 | 0.98–1.05 | **1.003** (round 2; pairs 0.995–1.009) | HIT |

Round 1 gave 1.043 at 262,144 and 0.987 at 1M, but main itself moved 7 % between its two passes
(1.699 → 1.588 ms), and all of round 1 at 262,144 ran ≈ 6 % faster than round 2 for **both**
binaries: a session-level clock effect, which is why the passes alternate. Round 2 is the
measurement; round 1 is kept as raw data.

## Conclusion
The read-only path changes the instruction (G1) and nothing else: **no measurable effect
(|Δ| < 1 %)**. The predicted mechanism holds: #7's loads are warp-uniform broadcasts that already
hit the same unified L1, and #7 is limited by the latency of its dependent pop chain (D-026),
not by the load path. The prediction's point estimate (0–3 % gain) was slightly optimistic; the
range held.
