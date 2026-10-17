#!/usr/bin/env bash
# Phase-2 runs for the tree round 1 (protocol: docs/predictions/2026-10-17-tree-round1.md §1).
# Run from the repo root:  bash results/2026-10-17_tree_round1/run_phase2.sh
set -u
R=results/2026-10-17_tree_round1
B=./build/nbody_bench.exe
mkdir -p $R/logs $R/warps $R/timeline
COMMON="--leaf 16 --block 128 --warmup-ms 300 --runs 11"
batch() { if [ "$1" -le 65536 ]; then echo 5; else echo 1; fi; }
run() {   # name, args...
  local name=$1; shift
  echo "== $name: $*" | tee -a $R/logs/index.txt
  $B "$@" > $R/logs/$name.txt 2>&1
  local rc=$?
  echo "   exit=$rc" | tee -a $R/logs/index.txt
}

$B --selftest > $R/logs/selftest.txt 2>&1; echo "selftest exit=$?" | tee -a $R/logs/index.txt

# A: main grid
for ic in plummer collision core; do
  for n in 65536 262144 1048576; do
    for k in bh_naive bh_sorted bh_hilbert; do
      run A_${k}_${ic}_${n} --kernel $k --n $n --ic $ic --theta 0.5 $COMMON --batch $(batch $n) \
          --csv $R/setA.csv --tree-stats $R/warps/A_${k}_${ic}_${n}.csv
    done
  done
done

# B: timelines
for ic in plummer core; do
  for n in 65536 1048576; do
    for k in bh_naive bh_sorted; do
      run B_${k}_${ic}_${n} --kernel $k --n $n --ic $ic --theta 0.5 $COMMON --batch $(batch $n) --no-validate \
          --timeline $R/timeline/tl_${k}_${ic}_${n}.csv --timeline-launches 5
    done
  done
done

# C: theta sweep (theta = 0 at 65536)
for th in 0.3 0.5 0.7 1.0; do
  for k in bh_naive bh_sorted; do
    run C_${k}_th${th} --kernel $k --n 262144 --ic plummer --theta $th $COMMON --batch 1 --csv $R/setC.csv \
        --tree-stats $R/warps/C_${k}_th${th}.csv
  done
done
for k in bh_naive bh_sorted; do
  run C_${k}_th0_n65536 --kernel $k --n 65536 --ic plummer --theta 0 $COMMON --batch 5 --csv $R/setC.csv \
      --tree-stats $R/warps/C_${k}_th0_n65536.csv
done

# D: crossover
for n in 8192 16384 24576 32768 49152 65536; do
  for k in bh_naive bh_sorted; do
    run D_${k}_${n} --kernel $k --n $n --ic plummer --theta 0.5 $COMMON --batch 5 --csv $R/setD_tree.csv
  done
  run D_tuned_b128u0_${n} --kernel allpairs_tuned --n $n --block 128 --unroll 0 --warmup-ms 300 --runs 11 --batch 5 \
      --csv $R/setD_ap.csv
  for b in 128 256 512 1024; do
    for q in 1 2 4 8 16; do
      run D_multi_b${b}q${q}_${n} --kernel allpairs_multi --n $n --block $b --q $q --warmup-ms 300 --runs 11 --batch 5 \
          --csv $R/setD_ap.csv
    done
  done
done
for n in 262144 524288; do
  run D_tuned_b128u0_${n} --kernel allpairs_tuned --n $n --block 128 --unroll 0 --warmup-ms 300 --runs 11 --batch 1 \
      --csv $R/setD_ap.csv
  run D_multi_b128q2_${n} --kernel allpairs_multi --n $n --block 128 --q 2 --warmup-ms 300 --runs 11 --batch 1 \
      --csv $R/setD_ap.csv
  run D_multi_b256q4_${n} --kernel allpairs_multi --n $n --block 256 --q 4 --warmup-ms 300 --runs 11 --batch 1 \
      --csv $R/setD_ap.csv
done
echo ALL DONE | tee -a $R/logs/index.txt
