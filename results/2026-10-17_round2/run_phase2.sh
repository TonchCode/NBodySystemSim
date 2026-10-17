#!/usr/bin/env bash
# Phase-2 protocol from docs/predictions/2026-10-17-round2.md (sections 1, 2, 3). Run from repo root.
set -u
R=results/2026-10-17_round2
B=./build/nbody_bench.exe
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,power.limit,temperature.gpu,utilization.gpu --format=csv -lms 100 -f $R/gpu_clock_log_phase2.csv &
SMI=$!
sleep 1.5
batch() { if [ "$1" -le 10000 ]; then echo 50; else echo 5; fi; }
best_multi() { case $1 in 1000) echo "512 32";; 4096) echo "512 8";; 10000) echo "1024 8";; 65536) echo "256 4";; esac; }
# ---- section 1/3: timelines + overhead
for N in 1000 4096 10000 65536; do
  bt=$(batch $N); read mb mq < <(best_multi $N)
  for cfg in "allpairs_tiled 256 --block 256" "allpairs_tuned b128u0 --block 128 --unroll 0" "allpairs_multi b${mb}q${mq} --block $mb --q $mq --prefetch 1"; do
    set -- $cfg; k=$1; tag=$2; shift 2
    echo "== $k $tag N=$N"
    $B --kernel $k --n $N "$@" --warmup-ms 300 --runs 21 --batch $bt --timeline $R/timeline/tl_${k#allpairs_}_${tag}_n$N.csv --timeline-launches 5 --csv $R/timeline/bench_timeline.csv
    echo "exit=$?"
  done
done
# ---- section 2: prefetch vs q
for spec in "4096 64 8" "4096 128 16" "4096 256 8" "4096 512 8" "65536 256 4"; do
  set -- $spec; N=$1; blk=$2; qb=$3; bt=$(batch $N)
  echo "== tuned u8 control N=$N b$blk"
  $B --kernel allpairs_tuned --n $N --block $blk --unroll 8 --warmup-ms 300 --runs 21 --batch $bt --csv $R/prefetch/prefetch.csv; echo "exit=$?"
  for q in 1 $qb; do for pf in 0 1; do
    echo "== multi N=$N b$blk q$q pf$pf"
    $B --kernel allpairs_multi --n $N --block $blk --q $q --prefetch $pf --warmup-ms 300 --runs 21 --batch $bt --csv $R/prefetch/prefetch.csv; echo "exit=$?"
  done; done
done
sleep 1; kill $SMI
