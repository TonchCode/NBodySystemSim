#!/usr/bin/env bash
# Phase-2 runs for tree round 2 (protocol: docs/predictions/2026-10-17-tree-round2.md §1).
# Run from the repo root:  bash results/2026-10-17_tree_round2/run_phase2.sh
# Session note: 20-22% idle GPU utilisation from desktop applications (env_before.txt), measured
# contamination +2.4-2.7% on round-1 reference points (contam/). Every number carries that label.
set -u
R=results/2026-10-17_tree_round2
B=./build/nbody_bench.exe
mkdir -p $R/logs $R/warps $R/timeline
COMMON="--warmup-ms 300 --runs 11"
TREE="--leaf 16 --block 128 --theta 0.5 $COMMON"
batch() { if [ "$1" -le 65536 ]; then echo 5; else echo 1; fi; }
run() {   # name, args...
  local name=$1; shift
  echo "== $name: $*" >> $R/logs/index.txt
  $B "$@" > $R/logs/$name.txt 2>&1
  local rc=$?
  echo "   exit=$rc" >> $R/logs/index.txt
  echo "$name exit=$rc"
}

$B --selftest > $R/logs/selftest.txt 2>&1; echo "selftest exit=$?" | tee -a $R/logs/index.txt

# A: main grid (siblings adjacent to the arms that are compared with them)
for ic in plummer core; do
  for n in 65536 262144 1048576; do
    for k in bh_naive bh_persistent_lane_naive bh_sorted bh_persistent bh_persistent_lane bh_persistent_static \
             bh_coherent bh_warp bh_warp_list bh_persistent_warp; do
      run A_${k}_${ic}_${n} --kernel $k --n $n --ic $ic $TREE --batch $(batch $n) \
          --csv $R/setA.csv --tree-stats $R/warps/A_${k}_${ic}_${n}.csv
    done
  done
done
for n in 65536 262144 1048576; do
  run A_bh_warp_naive_plummer_${n} --kernel bh_warp_naive --n $n --ic plummer $TREE --batch $(batch $n) \
      --csv $R/setA.csv --tree-stats $R/warps/A_bh_warp_naive_plummer_${n}.csv
done

# A2: interleaved A/B repeats for the #6 cells (background-load noise estimate), no validation
for rep in 1 2 3; do
  for ic in plummer core; do
    for pair in "bh_sorted bh_persistent" "bh_sorted bh_persistent_lane" "bh_naive bh_persistent_lane_naive" \
                "bh_warp bh_persistent_warp" "bh_sorted bh_persistent_static"; do
      for k in $pair; do
        run A2_r${rep}_${k}_${ic}_1048576_$(echo $pair | tr ' ' '-') --kernel $k --n 1048576 --ic $ic $TREE \
            --batch 1 --no-validate
      done
    done
  done
done

# B: warp-unit timelines (+ block-unit for bh_sorted and bh_warp)
for ic in plummer core; do
  for n in 65536 1048576; do
    for k in bh_sorted bh_coherent bh_warp bh_persistent bh_persistent_lane bh_persistent_lane_naive bh_persistent_warp; do
      run B_${k}_${ic}_${n} --kernel $k --n $n --ic $ic $TREE --batch $(batch $n) --no-validate \
          --timeline $R/timeline/tlw_${k}_${ic}_${n}.csv --timeline-unit warp --timeline-launches 5
    done
    for k in bh_sorted bh_warp; do
      run Bb_${k}_${ic}_${n} --kernel $k --n $n --ic $ic $TREE --batch $(batch $n) --no-validate \
          --timeline $R/timeline/tlb_${k}_${ic}_${n}.csv --timeline-unit block --timeline-launches 5
    done
  done
done

# C: leaf x block sweep
for n in 262144 1048576; do
  for k in bh_naive bh_sorted bh_warp; do
    for leaf in 4 8 16 32; do
      for b in 64 128 256; do
        run C_${k}_n${n}_l${leaf}_b${b} --kernel $k --n $n --ic plummer --leaf $leaf --block $b --theta 0.5 $COMMON \
            --batch 1 --csv $R/setC.csv
      done
    done
  done
done

# D: occupancy at fixed N (padding caps blocks/SM; 'u' = unpadded)
for k in bh_sorted bh_coherent bh_warp; do
  for ob in 1 2 4 8 12; do
    run D_${k}_occ${ob} --kernel $k --n 262144 --ic plummer $TREE --batch 1 --no-validate --occ-blocks $ob --csv $R/setD.csv
  done
  run D_${k}_occu --kernel $k --n 262144 --ic plummer $TREE --batch 1 --no-validate --csv $R/setD.csv
done

# E: crossover (tree and best all-pairs in the same session)
for n in 8192 16384 24576 32768 49152 65536; do
  for k in bh_warp bh_warp_list; do
    run E_${k}_${n} --kernel $k --n $n --ic plummer $TREE --batch 5 --csv $R/setE_tree.csv
  done
  run E_tuned_b128u0_${n} --kernel allpairs_tuned --n $n --block 128 --unroll 0 $COMMON --batch 5 --csv $R/setE_ap.csv
  for b in 128 256 512 1024; do
    for q in 1 2 4 8 16; do
      run E_multi_b${b}q${q}_${n} --kernel allpairs_multi --n $n --block $b --q $q $COMMON --batch 5 --csv $R/setE_ap.csv
    done
  done
done

# F: theta sweep
for th in 0.3 0.7 1.0; do
  for k in bh_sorted bh_warp; do
    run F_${k}_th${th} --kernel $k --n 262144 --ic plummer --theta $th --leaf 16 --block 128 $COMMON --batch 1 \
        --csv $R/setF.csv --tree-stats $R/warps/F_${k}_th${th}.csv
  done
done

# G: theta = 0
for k in bh_sorted bh_warp; do
  run G_${k}_th0 --kernel $k --n 65536 --ic plummer --theta 0 --leaf 16 --block 128 $COMMON --batch 5 --csv $R/setG.csv
done
echo ALL DONE | tee -a $R/logs/index.txt
