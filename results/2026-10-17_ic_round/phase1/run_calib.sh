#!/usr/bin/env bash
# IC round, phase 1: calibration of the normalised tree gates TN1/TN2 (D-027) on the LEGACY ICs only, and
# re-validation that the legacy verdicts (TG1/TG2/TG3) do not change. Validation runs, not timing runs
# (--runs 1 --warmup 0; the times in these logs are not results). bh_warp runs also evaluate the #5 sibling
# (bh_sorted, bitwise = bh_naive and every per-thread persistent arm), so they cover #4/#5/#6/#7 errors.
# Run from the repo root:  bash results/2026-10-17_ic_round/phase1/run_calib.sh
set -u
R=results/2026-10-17_ic_round/phase1/calib
B=./build/nbody_bench.exe
mkdir -p $R
V="--runs 1 --warmup 0 --block 128"
for ic in plummer collision core; do
  for n in 65536 262144 1048576; do
    $B --kernel bh_warp --n $n --ic $ic --theta 0.5 --leaf 16 $V --csv $R/calib.csv > $R/${ic}_${n}_t0.5_l16.txt 2>&1
    echo "$ic $n 0.5 16 exit=$?"
  done
  for th in 0.3 0.7 1.0; do
    $B --kernel bh_warp --n 262144 --ic $ic --theta $th --leaf 16 $V --csv $R/calib.csv > $R/${ic}_262144_t${th}_l16.txt 2>&1
    echo "$ic 262144 $th 16 exit=$?"
  done
  for leaf in 8 32; do
    $B --kernel bh_warp --n 1048576 --ic $ic --theta 0.5 --leaf $leaf $V --csv $R/calib.csv > $R/${ic}_1048576_t0.5_l${leaf}.txt 2>&1
    echo "$ic 1048576 0.5 $leaf exit=$?"
  done
done
# legacy verdict cross-check against the pre-IC-round binary (same args; error numbers must be identical)
OLD=${OLD_BENCH:-}
if [ -n "$OLD" ]; then
  for ic in plummer collision core; do
    $OLD --kernel bh_warp --n 262144 --ic $ic --theta 0.5 --leaf 16 $V > $R/OLD_${ic}_262144_t0.5_l16.txt 2>&1
    echo "OLD $ic exit=$?"
  done
fi
