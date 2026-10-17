#!/usr/bin/env bash
# EXPLORATORY runs (tree round 2, phase 2), NOT pre-registered. Written after set A showed every
# #6 arm slower than its sibling. Two hypotheses, each with one new kernel:
#   H1 (locality): the warp queue loses because a block's warps no longer walk Morton-adjacent
#      units -> bh_persistent_block (block-granular claims of 4 adjacent units).
#   H2 (overhead): lane refill loses to its per-pop refill check (it is 36% slower at N = 65536,
#      where it has nothing to refill) -> bh_persistent_lane8 (check every 8 pops).
# Rebuilt binary: the production SASS of every kernel measured in sets A-G is checked unchanged
# (explore_sass_check.txt). Siblings are re-measured in the same sequence.
set -u
R=results/2026-10-17_tree_round2
B=./build/nbody_bench.exe
mkdir -p $R/explore
TREE="--leaf 16 --block 128 --theta 0.5 --warmup-ms 300 --runs 11"
run() { local name=$1; shift; $B "$@" > $R/explore/$name.txt 2>&1; echo "$name exit=$?" | tee -a $R/explore/index.txt; }
$B --selftest > $R/explore/selftest.txt 2>&1; echo "selftest exit=$?" | tee -a $R/explore/index.txt
for rep in 1 2; do
  for ic in plummer core; do
    for n in 65536 1048576; do
      b=1; [ $n -le 65536 ] && b=5
      for k in bh_sorted bh_persistent bh_persistent_block bh_persistent_lane bh_persistent_lane8 \
               bh_naive bh_persistent_lane_naive bh_persistent_lane8_naive; do
        v=""; [ $rep -gt 1 ] && v="--no-validate"
        run X_r${rep}_${k}_${ic}_${n} --kernel $k --n $n --ic $ic $TREE --batch $b $v
      done
    done
  done
done
echo ALL DONE | tee -a $R/explore/index.txt
