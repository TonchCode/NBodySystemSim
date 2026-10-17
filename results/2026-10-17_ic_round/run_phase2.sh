#!/usr/bin/env bash
# IC round, phase 2 (protocol: docs/predictions/2026-10-17-ic-round.md §1, committed before these runs).
# Run from the repo root:  bash results/2026-10-17_ic_round/run_phase2.sh
# Session label: env_before.txt reports UNCONTROLLED BACKGROUND LOAD (desktop applications); every number
# carries that label and every ratio is taken from adjacent runs.
set -u
R=results/2026-10-17_ic_round
B=./build/nbody_bench.exe
OLD=${OLD_BENCH:-}
mkdir -p $R/logs $R/warps $R/timeline
COMMON="--warmup-ms 300 --runs 11 --theta 0.5"
batch() { if [ "$1" -le 65536 ]; then echo 5; else echo 1; fi; }
run() {   # name, args...
  local name=$1; shift
  echo "== $name: $*" >> $R/logs/index.txt
  "$@" > $R/logs/$name.txt 2>&1
  local rc=$?
  echo "   exit=$rc" >> $R/logs/index.txt
  echo "$name exit=$rc"
}

$B --selftest > $R/logs/selftest.txt 2>&1; echo "selftest exit=$?" | tee -a $R/logs/index.txt

# A: main grid, best-tuned settings (D-025), siblings adjacent: #4, #5, #6 (warp queue), #7
for ic in plummer disk galaxies cube_shell cube_edges uniform; do
  for n in 65536 262144 1048576; do
    b=$(batch $n)
    if [ $n -eq 1048576 ]; then L4=32; L7="--leaf 8 --block 256"; else L4=16; L7="--leaf 16 --block 128"; fi
    run A_bh_naive_${ic}_${n}      $B --kernel bh_naive      --n $n --ic $ic $COMMON --batch $b --leaf $L4 --block 256 --csv $R/setA.csv --tree-stats $R/warps/A_bh_naive_${ic}_${n}.csv
    run A_bh_sorted_${ic}_${n}     $B --kernel bh_sorted     --n $n --ic $ic $COMMON --batch $b --leaf 32 --block 256 --csv $R/setA.csv --tree-stats $R/warps/A_bh_sorted_${ic}_${n}.csv
    run A_bh_persistent_${ic}_${n} $B --kernel bh_persistent --n $n --ic $ic $COMMON --batch $b --leaf 32 --block 256 --csv $R/setA.csv
    run A_bh_warp_${ic}_${n}       $B --kernel bh_warp       --n $n --ic $ic $COMMON --batch $b $L7 --csv $R/setA.csv
  done
done

# B: protocol default leaf 16 b128 for #5 and #7 (round-to-round comparability; also K reference)
for ic in plummer disk galaxies cube_shell cube_edges uniform; do
  for n in 65536 262144 1048576; do
    b=$(batch $n)
    run B_bh_sorted_${ic}_${n} $B --kernel bh_sorted --n $n --ic $ic $COMMON --batch $b --leaf 16 --block 128 --csv $R/setB.csv
    run B_bh_warp_${ic}_${n}   $B --kernel bh_warp   --n $n --ic $ic $COMMON --batch $b --leaf 16 --block 128 --csv $R/setB.csv
  done
done

# K: mass spectrum at 262144, leaf 16 b128
for spec in "plummer salpeter" "plummer heavy-few" "disk salpeter"; do
  set -- $spec
  for k in bh_sorted bh_warp; do
    run K_${k}_${1}_${2} $B --kernel $k --n 262144 --ic $1 --mass-spectrum $2 $COMMON --leaf 16 --block 128 --csv $R/setK.csv
  done
done

# T: warp-unit timelines on the most imbalanced IC (galaxies, 65536)
run T_bh_sorted_galaxies_65536     $B --kernel bh_sorted     --n 65536 --ic galaxies $COMMON --batch 5 --leaf 32 --block 256 --no-validate --timeline-unit warp --timeline $R/timeline/T_bh_sorted_galaxies_65536.csv
run T_bh_persistent_galaxies_65536 $B --kernel bh_persistent --n 65536 --ic galaxies $COMMON --batch 5 --leaf 32 --block 256 --no-validate --timeline-unit warp --timeline $R/timeline/T_bh_persistent_galaxies_65536.csv
run T_bh_warp_galaxies_65536       $B --kernel bh_warp       --n 65536 --ic galaxies $COMMON --batch 5 --leaf 16 --block 128 --no-validate --timeline-unit warp --timeline $R/timeline/T_bh_warp_galaxies_65536.csv

# X: stack change (D-028) check: pre-round binary vs this binary, alternating, round-2 reference points
if [ -n "$OLD" ]; then
  for rep in 1 2; do
    for ic in plummer core; do
      for k in bh_naive bh_sorted; do
        run X_old_${k}_${ic}_r${rep} $OLD --kernel $k --n 262144 --ic $ic $COMMON --leaf 16 --block 128 --no-validate
        run X_new_${k}_${ic}_r${rep} $B   --kernel $k --n 262144 --ic $ic $COMMON --leaf 16 --block 128 --no-validate
      done
    done
  done
fi
