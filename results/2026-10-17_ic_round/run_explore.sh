#!/usr/bin/env bash
# EXPLORATORY (not pre-registered), added after set A/T: the single-wave galaxies launch is the first case where
# #5's SM efficiency drops to 0.62 (tail 37%) and #7's to 0.61 (tail 39%). Does a queue on the #7 walk
# (bh_persistent_warp) or the block-granular queue (bh_persistent_block) recover any of it? Siblings adjacent, 2 repeats.
set -u
R=results/2026-10-17_ic_round
B=./build/nbody_bench.exe
mkdir -p $R/explore
COMMON="--warmup-ms 300 --runs 11 --theta 0.5 --batch 5 --n 65536 --ic galaxies"
for rep in 1 2; do
  for k in bh_warp bh_persistent_warp; do
    $B --kernel $k $COMMON --leaf 16 --block 128 --csv $R/explore/explore.csv > $R/explore/${k}_l16b128_r$rep.txt 2>&1; echo "$k l16 r$rep exit=$?"
  done
  for k in bh_sorted bh_persistent_block bh_persistent; do
    $B --kernel $k $COMMON --leaf 32 --block 256 --csv $R/explore/explore.csv > $R/explore/${k}_l32b256_r$rep.txt 2>&1; echo "$k l32 r$rep exit=$?"
  done
done
for k in bh_warp bh_persistent_warp; do
  $B --kernel $k $COMMON --leaf 16 --block 128 --no-validate --timeline-unit warp --timeline $R/explore/T_$k.csv > $R/explore/T_$k.txt 2>&1; echo "T $k exit=$?"
done
