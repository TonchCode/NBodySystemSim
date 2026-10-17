#!/usr/bin/env bash
# IC round, phase 1: ONE small functional run per new IC (and per non-equal mass spectrum, on plummer), N = 8192,
# bh_warp (which also runs its #5 sibling for TG3n and the work ratio). Not timing runs. Everything these logs
# show is disclosed as a pre-registration observation in docs/predictions/2026-10-17-ic-round.md §0.
set -u
R=results/2026-10-17_ic_round/phase1/functional
B=./build/nbody_bench.exe
mkdir -p $R
V="--n 8192 --runs 1 --warmup 0 --block 128 --theta 0.5 --leaf 16 --kernel bh_warp"
for ic in disk galaxies cube_shell cube_edges uniform; do
  $B $V --ic $ic --csv $R/functional.csv > $R/$ic.txt 2>&1; echo "$ic exit=$?"
done
for sp in salpeter heavy-few; do
  $B $V --ic plummer --mass-spectrum $sp --csv $R/functional.csv > $R/plummer_$sp.txt 2>&1; echo "plummer:$sp exit=$?"
done
