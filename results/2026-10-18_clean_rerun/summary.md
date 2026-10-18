# 2026-10-18 — Clean re-run of the load-sensitive items (replication)

**Pre-registration:** `docs/predictions/2026-10-18-clean-rerun.md`, committed as **fbe0742** before
any timed run. A resume option was added after the pause, committed as **bfe3c5b** (skips cells,
changes no command). Each item is scored against its **original** committed range (c6a12cc,
56505d8); this is a replication, not a new prediction.

The bench-profiler wrote the content; the harness blocked the agent from writing this file, so the
main session saved it.

**Analysis:** `python tools/clean_rerun_analysis.py results/2026-10-18_clean_rerun`. This new
combiner runs the existing `tree_analysis.py --round2` and `ic_analysis.py` per pass and merged, and
writes `scorecard.csv` (original range, old value, clean value, clean/old ratio, old / clean /
pass-1 / pass-2 verdicts), `load_tests.txt` and `clock_summary.txt`.

## Environment

- **Binary:** `build/nbody_bench.exe`, SHA-256 `2b789683…cfdcfd43`. SASS identical to the viewer
  round (222/222 functions).
- **Toolchain:** driver 617.42, cl 19.51.36248, nvcc 13.4.
- **Power limit:** 272 W (default 320, unchanged).
- **Clocks under load:** SM median 2760 MHz in both sub-sessions (p5 2610). Power median 185 W,
  max 270 W.
- **Idle load:** **0% before and after**, ≤ 1% at every scored between-cell check. Light processes
  attached to the GPU are listed in `env_before.txt`.
- **Selftest:** PASS before both sub-sessions.
- **Runs:** 945 scored invocations, all exit 0, no gate failures, no stack overflows.
- **Order:** kernels A/B/…/B/A within each cell (two passes, p1/p2), so within-cell ratios are
  drift-protected.

## Session history (full detail in `flagged_cells.txt`, `session_times.txt`)

1. **04:32–04:56, cells 1–38**, all clean.
2. **At 04:56 the Windows screensaver started** (`Mystify.scr`, 30–44% GPU load) and the idle check
   flagged cell 39. The agent's stop ended only the wrapper shell, so the script ran on until about
   05:06: cell 40 ran under the screensaver (flagged, 19%), then the script died after the agent
   edited `run.sh` while bash was still reading it. **Nothing measured after 04:56 in this
   sub-session is scored.** Those logs are kept in `flagged/`, and the clock log has a gap from
   04:57:45 to 05:06.
3. **11:29–11:50, cells 39–95** (`RESUME_FROM=39`). The owner turned the screensaver off. The main
   session's first display-required hold (pid 23120) failed to compile and held nothing; a working
   one (pid 17844) was active from about 11:30. No cell was flagged.

The only cross-cell comparison that spans the gap is the set-C best-vs-best at 1M (cell 40 vs cell
29).

## Scorecard (original committed ranges)

| Block | Rows | Old HIT | Clean HIT | Changed |
|---|---|---|---|---|
| Tree round 2 primary (P1–P11, P13, W8–W12, L5, X1–X5) | 23 | 13 | 13 | none |
| IC round primary (S, Q, K3, V2) | 36 | 26 | 23 | S4-262144 (1.209 → 1.186, edge 1.2) and Q4-disk-1M (0.845 → 0.838, edge 0.84; passes disagree): both **named in advance as edge cells**. V2 (318.2 → 367.9 G int/s vs 150–350): a disputed reading, now also a different viewer build |
| Secondary (time-derived) | 16 | 9 | 10 | L3 MISS → HIT |
| Identity (deterministic) | 153 | 118 | 118 | none; every value reproduces |

**No #6 and no #7 conclusion changed.**

## Key cells, old → clean

| Item | Old | Clean |
|---|---|---|
| W8 #7/#5 plummer (65536 / 262144 / 1M) | 3.135 / 5.429 / 6.608 | 3.134 / 5.226 / 6.428 |
| W9 #7/#5 core | 3.444 / 6.363 / 7.314 | 3.339 / 6.371 / 7.091 |
| L5 best vs best, 1M / 262144 | 5.913 / 5.2 | **5.756 / 4.82** (totals 5.33× / 4.28×) |
| X1 crossover vs best all-pairs | 29,164 | **28,708** |
| X3 bh_warp vs best all-pairs (65536 / 262144 / 1M [D]) | 4.42 / 24.9 / 123 | 4.39 / 27.3 / 124 |
| #6 warp queue on #5 / on #7 | 0.89–0.93 / 0.89–0.95 | **0.88–0.93 / 0.90–0.95** |
| #6 lane refill on #5 / on #4 | 0.64–0.74 / 0.68–0.91 | **0.63–0.75 / 0.70–0.89** |
| #6 static partition | 0.84–0.98 | **0.85–0.97** |
| IC S #7/#5 at 1M: plummer / disk / galaxies / uniform | 6.14 / 5.95 / 5.56 / 4.30 | **5.70 / 5.43 / 5.02 / 3.91** |
| IC Q (single cells) | — | moved up to +7.3% (uniform 262144: 0.750 → 0.804) |
| K3 | 0.969 | 0.995 |
| X stack 64 → 96 (new/old) | — | 0.993–1.008 |

## Load-model tests (pre-registered falsifiers)

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Clean #4/#5 match clean round 1 (≥ 10/12 within ±3%) | 9/12, −2.7 … +4.2% | **FAIL** |
| 2a | Round-2 #5 ≥ 1% faster clean in ≥ 5/6 cells | 4/6 (median 0.973) | **FAIL** |
| 2b | IC #5 plummer 262144 within ±4% of round 1 | 8.516 vs 8.475 ms (old 9.724) | PASS |
| 3 | Speedups don't rise when clean | median 0.971 / 0.986 | PASS |
| 4 | #7 not inflated like #5 | median 0.999 / 0.998, **but** IC set-B #7 at 1M was inflated 7–15% | PASS (median) |
| 5 | Same-family ratios within 2% | median 1.2% | PASS |
| 6 | No drift in the clean session | 0.21% median; set A vs set C 0.996 (old 7%) | PASS |
| 7 | cube_edges speedups move ≤ 3% | max 1.9% (can't tell the models apart) | PASS |

**Interpretation:**
- **The predicted consequences held.** Speedups fell slightly, #6 was not rescued, and the round-2
  between-set drift disappeared.
- **The proposed mechanism was wrong.** The IC-round inflation follows **kernel length** and how
  late in the session the cell ran, not the walk type:
  - cells of about 8–10 ms or more lost 7–17% (including #7 at 1M in late set B);
  - short cells lost at most 1–3%.
  - That fits time-sharing of the GPU with desktop apps, not "per-thread walks are load-sensitive,
    coherent walks are not".
- **Round 2's load effect can't be told apart from session-to-session noise** (±4% per cell between
  two clean sessions). Tests 1 and 2a fail on exactly that.
- **Band misses:** W10 at 262144 (1.037), two P cells (1.048, 0.970), three Q cells, and V2 (1.156).

## Timelines (warp units)

| Run | SM eff | Tail | Slot fill |
|---|---|---|---|
| core 65536 #5 | 0.827 | 14.7% | 0.705 |
| core 65536 #6 warp queue | 0.921 | 6.4% | 0.721 |
| core 65536 #7 | 0.734 | 27.2% | 0.555 |
| 1M plummer / core #5 | | | 0.946 / 0.948 |
| 1M plummer / core #6 | | | 0.976 / 0.980 |
| 1M plummer / core #7 | | | 0.889 / 0.896 |

IC galaxies 65536, SM efficiency: #5 0.621, #6 0.808 (old 0.869), #7 0.614.

## Canonical numbers (D-033)

The clean values are canonical for every re-run item, quoted with this folder. Deterministic
quantities keep their original sources. Set D and the exploratory runs keep their old values,
labelled with the load. D-025's best-tuned settings stand.

## Independent re-check (main session, after the run, idle 0%)

plummer 1M, θ = 0.5, 5 runs, traversal:

| Kernel | Agent (clean) | Re-check |
|---|---|---|
| bh_sorted leaf 32 b256 | 34.45 ms | 34.31 ms |
| bh_warp leaf 16 b64 | 5.99 ms | 5.99 ms |
| Best vs best | 5.76× | **5.73×** |

## Process lessons (also in `docs/03-design.md` §22)

- **Disable the screensaver** (and confirm the registry, not just the Settings page) before long
  sessions. Hold display-required for the whole session.
- **Never edit a running bash script:** bash reads scripts incrementally.
- **Stopping a wrapper doesn't stop its child script:** kill the script itself.
