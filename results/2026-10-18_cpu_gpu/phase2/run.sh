#!/usr/bin/env bash
# R3 #8 phase 2: CPU baselines, P/E cores, CPU vs GPU, hybrid splits, PCIe, lag gates.
# Protocol: docs/predictions/2026-10-18-cpu-gpu.md §7 (written before any timing run of this round).
#
#   DRYRUN=1 bash results/2026-10-18_cpu_gpu/phase2/run.sh   # lists every command, touches no GPU
#            bash <copy of this script>                       # the timed session
#
# Run from the repo root, on a COPY of this file (the copy is never edited while it runs; to stop it,
# kill the bash process of the copy). Env capture (capture_env.ps1 / capture_cpu_env.ps1, before/after)
# is done outside this script.
#
# A cell = one (IC, N, theta) point; its entries run in order (pass p1) and then reversed (pass p2):
# A B .. B A. Logs: logs/<cell>/<pass>/<entry>.txt (+ .csv). Between cells the idle GPU utilisation
# (nvidia-smi, 5 x 200 ms after 1.5 s) and idle CPU utilisation (typeperf, 3 x 1 s) are sampled. GPU > 5%
# or CPU > 10% flags the cell: its logs move to flagged/<cell>/, the script waits for idle (15 s steps, up
# to 20) and re-runs the cell once. (The CPU threshold is set here, before the run; §7 asks only that the
# CPU idle utilisation be recorded.) A GPU clock/power log and a CPU utilisation/performance log run for
# the whole session. No system, driver, power or screensaver setting is changed.
set -u
R=results/2026-10-18_cpu_gpu/phase2
B=./build/nbody_bench.exe
DRY=${DRYRUN:-0}
RES=${RESUME_FROM:-1}
SFX=""; [ "$RES" -gt 1 ] && SFX="_resume$RES"
NRUNS=0; NCELLS=0
CELL=""; CDIR=""

GPU="--warmup-ms 300 --runs 11"
CPU="--warmup 2 --runs 7"
CPU5="--warmup 2 --runs 5"

out() { if [ "$DRY" = 1 ]; then echo "$*" >> $R/dryrun_commands.txt; else echo "$*"; fi; }
batch() { if [ "$1" -le 1024 ]; then echo 20; elif [ "$1" -le 65536 ]; then echo 5; else echo 1; fi; }

run() {   # pass name args...
  local pass=$1 name=$2; shift 2
  local d=$CDIR/$pass
  NRUNS=$((NRUNS + 1))
  if [ "$DRY" = 1 ]; then out "    $B $* --csv $d/$name.csv > $d/$name.txt"; return; fi
  mkdir -p $d
  echo "== $name: $B $*" >> $d/index.txt
  "$B" "$@" --csv "$d/$name.csv" > "$d/$name.txt" 2>&1
  local rc=$?
  echo "   exit=$rc $(date +%T)" >> $d/index.txt
  echo "  [$CELL $pass] $name exit=$rc"
}

idle_gpu() {
  local v=() i
  for i in 1 2 3 4 5; do
    v+=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -dc '0-9'))
    sleep 0.2
  done
  printf '%s\n' "${v[@]}" | sort -n | sed -n 3p
}
idle_cpu() {   # median of 3 x 1 s, integer percent (rounded up)
  typeperf "\\Processor(_Total)\\% Processor Time" -si 1 -sc 3 2>/dev/null | grep '^"[0-9]' \
    | sed 's/.*,"\([0-9.]*\)"/\1/' | sort -n | sed -n 2p | awk '{printf "%d\n", $1 + 0.999}'
}
idle_both() { sleep 1.5; local g c; g=$(idle_gpu); c=$(idle_cpu); echo "${g:-0} ${c:-0}"; }
log_idle() { echo "$(date '+%F %T'),$1,$2,$3,$4" >> $R/idle_log.csv; }   # time, cell, phase, gpu %, cpu %
busy() { [ "$1" -gt 5 ] || [ "$2" -gt 10 ]; }
wait_clear() {
  local t g c
  for t in $(seq 1 20); do
    sleep 15; read g c <<< "$(idle_both)"; log_idle "$1" "wait$t" "$g" "$c"
    busy "$g" "$c" || return 0
  done
  return 1
}

# ENT: array of "name|args" for the current cell; run forward (p1) then reversed (p2)
passes() {
  local mode=$1 i e
  for ((i=0; i<${#ENT[@]}; i++)); do e=${ENT[i]}; run p1 "${e%%|*}" ${e#*|}; done
  [ "$mode" = once ] && return
  for ((i=${#ENT[@]}-1; i>=0; i--)); do e=${ENT[i]}; run p2 "${e%%|*}" ${e#*|}; done
}
cell() {   # id mode(both|once) builder args...
  local id=$1 mode=$2; shift 2
  NCELLS=$((NCELLS + 1))
  if [ "$NCELLS" -lt "$RES" ]; then return; fi
  CELL=$id; CDIR=$R/logs/$id
  ENT=(); "$@"
  out "## cell $NCELLS: $id ($mode, ${#ENT[@]} entries)"
  if [ "$DRY" = 1 ]; then passes $mode; return; fi
  echo "$(date '+%F %T') cell $NCELLS $id start" >> $R/session_times.txt
  passes $mode
  local g c; read g c <<< "$(idle_both)"; log_idle "$id" after "$g" "$c"
  if busy "$g" "$c"; then
    echo "$(date '+%F %T') FLAGGED $id (gpu $g%, cpu $c%)" | tee -a $R/flagged_cells.txt
    mkdir -p $R/flagged; mv $CDIR $R/flagged/$id.$(date +%H%M%S)
    if wait_clear "$id"; then
      echo "$(date '+%F %T') RERUN $id" | tee -a $R/flagged_cells.txt
      passes $mode
      read g c <<< "$(idle_both)"; log_idle "$id" after_rerun "$g" "$c"
      busy "$g" "$c" && echo "$(date '+%F %T') FLAGGED-AGAIN $id (gpu $g%, cpu $c%) EXCLUDE" | tee -a $R/flagged_cells.txt
    else
      echo "$(date '+%F %T') NOT-CLEARED $id EXCLUDE" | tee -a $R/flagged_cells.txt
    fi
  fi
}

# ================================ cell builders ================================================
CA="--cpu-set all --cpu-sched dynamic"

# C1, C2, C3 (65536), C4, C11: cpu_allpairs IV x tile, 20 LPs dynamic
b_C1() {   # n
  local n=$1 r="$CPU" iv t; [ $n -ge 262144 ] && r="$CPU5"
  for iv in 1 2; do for t in 512 2048 8192; do
    ENT+=("ap_iv${iv}_t$t|--kernel cpu_allpairs --n $n $CA --cpu-iv $iv --cpu-tile $t $r")
  done; done
  [ $n = 65536 ] && ENT+=("ap_iv2_t2048_rsqrt0|--kernel cpu_allpairs --n $n $CA --cpu-iv 2 --cpu-tile 2048 --cpu-rsqrt 0 $r")
}
# C5-C9, C12, T1/T3: cpu_bh group x leaf (Plummer, theta 0.5); 1M adds G8/G64 and the scalar walk
b_C5() {   # n
  local n=$1 g l gs="16 32"; [ $n -ge 1048576 ] && gs="8 16 32 64"
  for g in $gs; do for l in 8 16 32; do
    ENT+=("bh_g${g}_l$l|--kernel cpu_bh --n $n --ic plummer --theta 0.5 --leaf $l --cpu-group $g $CA $CPU")
  done; done
  if [ $n -ge 1048576 ]; then for l in 8 16 32; do
    ENT+=("bhs_l$l|--kernel cpu_bh_scalar --n $n --ic plummer --theta 0.5 --leaf $l $CA $CPU")
  done; fi
}
# C10: CPU crossover tree vs all-pairs (all 20 and p1)
b_C10() {   # n
  local n=$1 s bt; bt=$(batch $n)
  for s in all p1; do
    ENT+=("ap_${s}|--kernel cpu_allpairs --n $n --cpu-set $s --cpu-sched dynamic --cpu-iv 2 --cpu-tile 2048 $CPU --batch $bt")
    ENT+=("bh_${s}_g16_l8|--kernel cpu_bh --n $n --ic plummer --theta 0.5 --leaf 8 --cpu-group 16 --cpu-set $s --cpu-sched dynamic $CPU --batch $bt")
    ENT+=("bh_${s}_g32_l16|--kernel cpu_bh --n $n --ic plummer --theta 0.5 --leaf 16 --cpu-group 32 --cpu-set $s --cpu-sched dynamic $CPU --batch $bt")
  done
}
# E1-E8: all-pairs 65536, IV 2, tile 2048, sets x schedules
b_E1() {
  local s c
  for s in all p p1 e cores; do for c in static dynamic weighted; do
    ENT+=("ap_${s}_$c|--kernel cpu_allpairs --n 65536 --cpu-set $s --cpu-sched $c --cpu-iv 2 --cpu-tile 2048 $CPU")
  done; done
}
# E9-E12: cpu_bh Plummer 1M, G 32, leaf 16, sets all/p/e x schedules
b_E9() {
  local s c
  for s in all p e; do for c in static dynamic weighted; do
    ENT+=("bh_${s}_$c|--kernel cpu_bh --n 1048576 --ic plummer --theta 0.5 --leaf 16 --cpu-group 32 --cpu-set $s --cpu-sched $c $CPU")
  done; done
}
# S1-S4: GPU all-pairs (event-timed best-tuned; wall-timed copies at N <= 4096, not scored) and the CPU at N <= 4096
b_S1() {   # n
  local n=$1 bt mb=256 s; bt=$(batch $n); [ $n -le 4096 ] && mb=512
  ENT+=("gpu_tuned_b128|--kernel allpairs_tuned --n $n --block 128 --unroll 0 $GPU --batch $bt")
  ENT+=("gpu_multi_b$mb|--kernel allpairs_multi --n $n --block $mb --q 0 $GPU --batch $bt")
  if [ $n -le 4096 ]; then
    ENT+=("gpu_tuned_b128_wall|--kernel allpairs_tuned --n $n --block 128 --unroll 0 $GPU --batch $bt --wall")
    ENT+=("gpu_multi_b${mb}_wall|--kernel allpairs_multi --n $n --block $mb --q 0 $GPU --batch $bt --wall")
    for s in all p1; do
      ENT+=("cpu_ap_$s|--kernel cpu_allpairs --n $n --cpu-set $s --cpu-sched dynamic --cpu-iv 2 --cpu-tile 2048 $CPU --batch $bt")
    done
    ENT+=("cpu_ap_1t|--kernel cpu_allpairs --n $n --cpu-set p1 --cpu-threads 1 --cpu-sched dynamic --cpu-iv 2 --cpu-tile 2048 $CPU --batch $bt")
  fi
}
# S5-S9: GPU trees event-timed (bh_warp best-tuned D-025/D-033; bh_sorted best leaf 32 b256); cube_edges adds cpu_bh
b_S5() {   # ic n
  local ic=$1 n=$2 wl=16 wb=128 g l; [ $n -ge 1048576 ] && { wl=8; wb=256; }
  ENT+=("gpu_bh_warp|--kernel bh_warp --n $n --ic $ic --theta 0.5 --leaf $wl --block $wb $GPU")
  ENT+=("gpu_bh_sorted|--kernel bh_sorted --n $n --ic $ic --theta 0.5 --leaf 32 --block 256 $GPU")
  if [ $ic = cube_edges ]; then for g in 16 32; do for l in 8 16; do
    ENT+=("cpu_bh_g${g}_l$l|--kernel cpu_bh --n $n --ic $ic --theta 0.5 --leaf $l --cpu-group $g $CA $CPU")
  done; done; fi
}
# H1-H3: hyb_allpairs dynamic vs GPU all-pairs, all wall-clock
b_H1() {   # n
  local n=$1 bt; bt=$(batch $n)
  ENT+=("gpu_tuned_b128_wall|--kernel allpairs_tuned --n $n --block 128 --unroll 0 $GPU --batch $bt --wall")
  ENT+=("gpu_multi_b256_wall|--kernel allpairs_multi --n $n --block 256 --q 0 $GPU --batch $bt --wall")
  ENT+=("hyb_ap_dyn|--kernel hyb_allpairs --n $n --block 128 --hyb-frac -1 $CA $GPU --batch $bt")
}
# H4: static f from this session's standalone rates (computed by hfrac.py from cells H1-262144 and C1-262144)
b_H4() {
  local f; f=$(python $R/hfrac.py $R 2>/dev/null || echo 0.97)
  [ "$DRY" = 1 ] && f="<from hfrac.py>"
  echo "H4 static f = $f" >> $R/session_times.txt
  ENT+=("gpu_tuned_b128_wall|--kernel allpairs_tuned --n 262144 --block 128 --unroll 0 $GPU --wall")
  ENT+=("hyb_ap_dyn|--kernel hyb_allpairs --n 262144 --block 128 --hyb-frac -1 $CA $GPU")
  ENT+=("hyb_ap_static|--kernel hyb_allpairs --n 262144 --block 128 --hyb-frac $f $CA $GPU")
}
# H5-H15 + the coordinator's GPU-resident references: one cell per (ic, n, theta)
b_H5() {   # ic n theta full(1: phase kernels too)
  local ic=$1 n=$2 th=$3 full=$4 wl=16 wb=128; [ $n -ge 1048576 ] && { wl=8; wb=256; }
  local T="--n $n --ic $ic --theta $th --leaf $wl --block $wb"
  ENT+=("gpu_bh_warp_wall|--kernel bh_warp $T $GPU --wall")
  ENT+=("gpu_bh_sorted_wall|--kernel bh_sorted --n $n --ic $ic --theta $th --leaf 32 --block 256 $GPU --wall")
  ENT+=("hyb_bh_dyn|--kernel hyb_bh $T --hyb-frac -1 $CA $GPU")
  [ $ic = cube_edges ] && [ $n -ge 1048576 ] && ENT+=("hyb_bh_f095|--kernel hyb_bh $T --hyb-frac 0.95 $CA $GPU")
  if [ "$full" = 1 ]; then
    ENT+=("hyb_phase|--kernel hyb_phase $T $CA $GPU")
    ENT+=("gpu_refit_k4_wall|--kernel bh_warp_refit $T --refit-every 4 $GPU --wall")
    ENT+=("gpu_refit_k16_wall|--kernel bh_warp_refit $T --refit-every 16 $GPU --wall")
    if [ $ic = plummer ] && { [ $n = 65536 ] || [ $n = 1048576 ]; }; then
      ENT+=("hyb_phase_sync|--kernel hyb_phase $T --hyb-sync $CA $GPU")
    fi
  fi
}
# L1-L3: lag gates, 500 steps (once; physics, not timing)
b_L1() {   # ic
  local ic=$1 T="--n 65536 --ic $1 --theta 0.5 --leaf 16 --block 128 --steps 500 --dt 1e-3 --lag-check --warmup 2 --runs 3"
  ENT+=("hyb_phase_lag|--kernel hyb_phase $T $CA")
  ENT+=("refit_k16_lag|--kernel bh_warp_refit $T --refit-every 16")
}

# ================================ session ======================================================
if [ "$DRY" = 1 ]; then
  : > $R/dryrun_commands.txt
  out "# DRY RUN $(date '+%F %T'): commands only. Binary: $B"
else
  mkdir -p $R/logs
  [ -f $R/idle_log.csv ] || echo "time,cell,phase,gpu_idle_util_pct,cpu_idle_util_pct" > $R/idle_log.csv
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,pstate,clocks_event_reasons.active \
      --format=csv -lms 500 -f $R/gpu_clock_log$SFX.csv &
  CLOG=$!
  typeperf "\\Processor(_Total)\\% Processor Time" "\\Processor Information(_Total)\\% Processor Performance" \
      -si 2 -f CSV -o $R/cpu_log$SFX.csv -y > /dev/null 2>&1 &
  TLOG=$!
  trap 'kill $CLOG $TLOG 2>/dev/null; taskkill //F //IM typeperf.exe > /dev/null 2>&1' EXIT
  md5sum $B > $R/binary_md5$SFX.txt
  echo "session start $(date '+%F %T') (resume from cell $RES)" | tee -a $R/session_times.txt
  read g c <<< "$(idle_both)"; log_idle start start "$g" "$c"
  $B --cpu-info > $R/cpu_info_start$SFX.txt 2>&1
  $B --selftest > $R/selftest$SFX.txt 2>&1; rc=$?
  echo "selftest exit=$rc $(tail -2 $R/selftest$SFX.txt | head -1)" | tee -a $R/session_times.txt
  if [ $rc != 0 ] || ! grep -q "^selftest: PASS" $R/selftest$SFX.txt; then echo "SELFTEST FAILED: abort"; exit 1; fi
fi

out "# ---- §1 CPU baselines"
for n in 16384 65536 262144; do cell "C1_$n" both b_C1 $n; done
for n in 65536 262144 1048576; do cell "C5_$n" both b_C5 $n; done
for n in 512 1024 2048 4096 8192 16384; do cell "C10_$n" both b_C10 $n; done
out "# ---- §2 P/E cores"
cell "E1_65536" both b_E1
cell "E9_1M" both b_E9
out "# ---- §3 CPU vs GPU"
for n in 64 128 256 512 1024 4096 16384 65536 262144; do cell "S1_$n" both b_S1 $n; done
for x in "plummer 65536" "plummer 262144" "plummer 1048576" "cube_edges 1048576"; do
  set -- $x; cell "S5_$1_$2" both b_S5 $1 $2
done
out "# ---- §4 hybrids"
for n in 16384 65536 131072 262144; do cell "H1_$n" both b_H1 $n; done
cell "H4_262144" both b_H4
for x in "plummer 65536 0.5 1" "plummer 262144 0.5 1" "plummer 1048576 0.5 1" "plummer 1048576 0.3 0" \
         "cube_edges 262144 0.5 1" "cube_edges 1048576 0.5 1"; do
  set -- $x; cell "H5_$1_$2_th$3" both b_H5 $1 $2 $3 $4
done
out "# ---- §6 lag gates"
for ic in plummer galaxies; do cell "L1_$ic" once b_L1 $ic; done

if [ "$DRY" = 1 ]; then
  out "# total: $NCELLS cells, $NRUNS invocations (+ selftest, 2 x --cpu-info)"
  echo "dry run: $NCELLS cells, $NRUNS invocations -> $R/dryrun_commands.txt"
else
  $B --cpu-info > $R/cpu_info_end$SFX.txt 2>&1
  echo "session end $(date '+%F %T'): $NCELLS cells, $NRUNS invocations" | tee -a $R/session_times.txt
  echo ALL DONE
fi
