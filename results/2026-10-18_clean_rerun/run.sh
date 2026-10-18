#!/usr/bin/env bash
# Clean re-run of the load-sensitive items of tree round 2 and the IC round.
# Protocol: docs/predictions/2026-10-18-clean-rerun.md (committed before this script is run timed).
#
#   DRYRUN=1 bash results/2026-10-18_clean_rerun/run.sh   # lists every command, touches no GPU
#            bash results/2026-10-18_clean_rerun/run.sh   # the timed session (after capture_env.ps1 -Tag before)
#
# Run from the repo root. Commands are copied from results/2026-10-17_tree_round2/run_phase2.sh and
# results/2026-10-17_ic_round/run_phase2.sh (same kernels, N, IC, theta, leaf/block, runs, warm-up,
# batch, validation, --tree-stats). What changes is the ORDER:
#   * a cell = one (set, IC, N[, leaf, block, theta, spectrum]) point; inside a cell the kernels run in the
#     original order (pass 1, output in <block>/p1/) and then in REVERSE order (pass 2, <block>/p2/): A B C .. C B A;
#   * A2 (tree round 2) and X (IC round) were already alternating; they run once (p1), as written;
#   * between cells the idle GPU utilisation is sampled (5 x 200 ms after 1.5 s). If the median is > 5%, the
#     script pauses (15 s re-samples, up to 20), logs it to idle_log.csv, flags the cell in flagged_cells.txt,
#     moves the first attempt's logs to flagged/<cell>/ and re-runs the cell once (pre-registration §2.5).
#     CSV rows are staged per cell and appended only for the accepted attempt.
# A clock/power log runs for the whole session (gpu_clock_log.csv). No system, driver or power setting is changed.
set -u
R=results/2026-10-18_clean_rerun
B=./build/nbody_bench.exe
V=./build/src/viewer/nbody_viewer.exe
OLD=${OLD_BENCH:-/c/Users/petya/AppData/Local/Temp/claude/D--life-system/cf99499d-78af-474a-80e1-0ba28ceb4299/scratchpad/nbody_bench_before.exe}
DRY=${DRYRUN:-0}
# RESUME_FROM=K (added 2026-10-18 after the session was stopped at cell 39 by the screensaver; commands unchanged):
# cells 1..K-1 are skipped (already measured), the clock log and selftest go to *_resumeK files.
RES=${RESUME_FROM:-1}
SFX=""; [ "$RES" -gt 1 ] && SFX="_resume$RES"
STAGE=$R/stage
NRUNS=0; NCELLS=0
STAGED=(); CELL_LOGS=()

out() { if [ "$DRY" = 1 ]; then echo "$*" >> $R/dryrun_commands.txt; else echo "$*"; fi; }
rev() { local a=("$@") i; for ((i=${#a[@]}-1; i>=0; i--)); do printf '%s\n' "${a[i]}"; done; }
batch() { if [ "$1" -le 65536 ]; then echo 5; else echo 1; fi; }

run() {   # dir name csvfile|- [@OLD] args...
  local dir=$1 name=$2 csvf=$3; shift 3
  local exe=$B
  if [ "$1" = "@OLD" ]; then exe=$OLD; shift; fi
  local extra=()
  if [ "$csvf" != "-" ]; then
    extra=(--csv "$STAGE/$csvf.$(basename $dir).$name.csv")
    STAGED+=("$dir|$csvf|$STAGE/$csvf.$(basename $dir).$name.csv")
  fi
  CELL_LOGS+=("$dir/logs/$name.txt")
  NRUNS=$((NRUNS + 1))
  if [ "$DRY" = 1 ]; then out "    $exe $* ${extra[*]} > $dir/logs/$name.txt"; return; fi
  echo "== $name: $exe $* ${extra[*]}" >> $dir/logs/index.txt
  "$exe" "$@" "${extra[@]}" > $dir/logs/$name.txt 2>&1
  local rc=$?
  echo "   exit=$rc $(date +%T)" >> $dir/logs/index.txt
  echo "$name exit=$rc"
}

idle_med() {
  sleep 1.5
  local v=() i
  for i in 1 2 3 4 5; do
    v+=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -dc '0-9'))
    sleep 0.2
  done
  printf '%s\n' "${v[@]}" | sort -n | sed -n 3p
}
log_idle() { echo "$(date '+%F %T'),$1,$2,$3" >> $R/idle_log.csv; }   # time, cell, phase, median util %
wait_clear() {
  local t m
  for t in $(seq 1 20); do
    sleep 15; m=$(idle_med); log_idle "$1" "wait$t" "$m"
    [ "$m" -le 5 ] && return 0
  done
  return 1
}
commit_staged() {
  local e dir csvf src dst
  for e in "${STAGED[@]}"; do
    IFS='|' read -r dir csvf src <<< "$e"
    dst=$dir/$csvf
    [ -f "$src" ] || continue
    if [ -s "$dst" ]; then tail -n +2 "$src" >> "$dst"; else cat "$src" > "$dst"; fi
    rm -f "$src"
  done
}

cell() {   # id mode(both|once) fn args...
  local id=$1 mode=$2; shift 2
  NCELLS=$((NCELLS + 1))
  if [ "$NCELLS" -lt "$RES" ]; then return; fi
  STAGED=(); CELL_LOGS=()
  out "## cell $NCELLS: $id ($mode)"
  if [ "$mode" = both ]; then "$@" p1; "$@" p2; else "$@" p1; fi
  [ "$DRY" = 1 ] && return
  local m; m=$(idle_med); log_idle "$id" after "$m"
  if [ "$m" -gt 5 ]; then
    echo "$(date '+%F %T') FLAGGED $id (idle $m%)" | tee -a $R/flagged_cells.txt
    local fd=$R/flagged/$(echo "$id" | tr ' /' '__')
    mkdir -p $fd
    local f; for f in "${CELL_LOGS[@]}"; do [ -f "$f" ] && cp "$f" $fd/; done
    if wait_clear "$id"; then
      local e; for e in "${STAGED[@]}"; do rm -f "${e##*|}"; done
      STAGED=(); CELL_LOGS=()
      echo "$(date '+%F %T') RERUN $id" | tee -a $R/flagged_cells.txt
      if [ "$mode" = both ]; then "$@" p1; "$@" p2; else "$@" p1; fi
      m=$(idle_med); log_idle "$id" after_rerun "$m"
      [ "$m" -gt 5 ] && echo "$(date '+%F %T') FLAGGED-AGAIN $id (idle $m%) EXCLUDE" | tee -a $R/flagged_cells.txt
    else
      echo "$(date '+%F %T') NOT-CLEARED $id EXCLUDE (first attempt kept)" | tee -a $R/flagged_cells.txt
    fi
  fi
  commit_staged
}

# ================================ tree round 2 =================================================
T2COMMON="--warmup-ms 300 --runs 11"
T2TREE="--leaf 16 --block 128 --theta 0.5 $T2COMMON"

t2_A() {   # ic n pass   (set A; bh_warp_naive moved into the Plummer cells, last in pass 1)
  local ic=$1 n=$2 P=$3 d=$R/tree2/$3 k
  local ks=(bh_naive bh_persistent_lane_naive bh_sorted bh_persistent bh_persistent_lane bh_persistent_static
            bh_coherent bh_warp bh_warp_list bh_persistent_warp)
  [ $ic = plummer ] && ks+=(bh_warp_naive)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    run $d A_${k}_${ic}_${n} setA.csv --kernel $k --n $n --ic $ic $T2TREE --batch $(batch $n) \
        --tree-stats $d/warps/A_${k}_${ic}_${n}.csv
  done
}
t2_A2() {  # rep ic pass   (as written: pairs adjacent, 1M, no validation)
  local rep=$1 ic=$2 d=$R/tree2/p1 pair k
  for pair in "bh_sorted bh_persistent" "bh_sorted bh_persistent_lane" "bh_naive bh_persistent_lane_naive" \
              "bh_warp bh_persistent_warp" "bh_sorted bh_persistent_static"; do
    for k in $pair; do
      run $d A2_r${rep}_${k}_${ic}_1048576_$(echo $pair | tr ' ' '-') - --kernel $k --n 1048576 --ic $ic $T2TREE \
          --batch 1 --no-validate
    done
  done
}
t2_B() {   # ic n pass   (warp-unit timelines + block-unit for bh_sorted, bh_warp)
  local ic=$1 n=$2 P=$3 d=$R/tree2/$3 s k u
  local ks=(w:bh_sorted w:bh_coherent w:bh_warp w:bh_persistent w:bh_persistent_lane w:bh_persistent_lane_naive
            w:bh_persistent_warp b:bh_sorted b:bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for s in "${ks[@]}"; do
    u=${s%%:*}; k=${s#*:}
    if [ $u = w ]; then
      run $d B_${k}_${ic}_${n} - --kernel $k --n $n --ic $ic $T2TREE --batch $(batch $n) --no-validate \
          --timeline $d/timeline/tlw_${k}_${ic}_${n}.csv --timeline-unit warp --timeline-launches 5
    else
      run $d Bb_${k}_${ic}_${n} - --kernel $k --n $n --ic $ic $T2TREE --batch $(batch $n) --no-validate \
          --timeline $d/timeline/tlb_${k}_${ic}_${n}.csv --timeline-unit block --timeline-launches 5
    fi
  done
}
t2_C() {   # n leaf b pass
  local n=$1 leaf=$2 b=$3 P=$4 d=$R/tree2/$4 k
  local ks=(bh_naive bh_sorted bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    run $d C_${k}_n${n}_l${leaf}_b${b} setC.csv --kernel $k --n $n --ic plummer --leaf $leaf --block $b --theta 0.5 \
        $T2COMMON --batch 1
  done
}
t2_E() {   # n pass
  local n=$1 P=$2 d=$R/tree2/$2 s b q
  local ks=(t:bh_warp t:bh_warp_list tuned)
  for b in 128 256 512 1024; do for q in 1 2 4 8 16; do ks+=(m:$b:$q); done; done
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for s in "${ks[@]}"; do
    case $s in
      t:*)   run $d E_${s#t:}_${n} setE_tree.csv --kernel ${s#t:} --n $n --ic plummer $T2TREE --batch 5 ;;
      tuned) run $d E_tuned_b128u0_${n} setE_ap.csv --kernel allpairs_tuned --n $n --block 128 --unroll 0 $T2COMMON --batch 5 ;;
      m:*)   b=$(echo $s | cut -d: -f2); q=$(echo $s | cut -d: -f3)
             run $d E_multi_b${b}q${q}_${n} setE_ap.csv --kernel allpairs_multi --n $n --block $b --q $q $T2COMMON --batch 5 ;;
    esac
  done
}
t2_F() {   # th pass
  local th=$1 P=$2 d=$R/tree2/$2 k
  local ks=(bh_sorted bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    run $d F_${k}_th${th} setF.csv --kernel $k --n 262144 --ic plummer --theta $th --leaf 16 --block 128 $T2COMMON --batch 1 \
        --tree-stats $d/warps/F_${k}_th${th}.csv
  done
}
t2_G() {   # pass
  local P=$1 d=$R/tree2/$1 k
  local ks=(bh_sorted bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    run $d G_${k}_th0 setG.csv --kernel $k --n 65536 --ic plummer --theta 0 --leaf 16 --block 128 $T2COMMON --batch 5
  done
}

# ================================ IC round =====================================================
ICCOMMON="--warmup-ms 300 --runs 11 --theta 0.5"

ic_A() {   # ic n pass   (best-tuned, D-025)
  local ic=$1 n=$2 P=$3 d=$R/ic/$3 k b L4 L7
  b=$(batch $n)
  if [ $n -eq 1048576 ]; then L4=32; L7="--leaf 8 --block 256"; else L4=16; L7="--leaf 16 --block 128"; fi
  local ks=(bh_naive bh_sorted bh_persistent bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    case $k in
      bh_naive)      run $d A_bh_naive_${ic}_${n} setA.csv --kernel bh_naive --n $n --ic $ic $ICCOMMON --batch $b --leaf $L4 --block 256 --tree-stats $d/warps/A_bh_naive_${ic}_${n}.csv ;;
      bh_sorted)     run $d A_bh_sorted_${ic}_${n} setA.csv --kernel bh_sorted --n $n --ic $ic $ICCOMMON --batch $b --leaf 32 --block 256 --tree-stats $d/warps/A_bh_sorted_${ic}_${n}.csv ;;
      bh_persistent) run $d A_bh_persistent_${ic}_${n} setA.csv --kernel bh_persistent --n $n --ic $ic $ICCOMMON --batch $b --leaf 32 --block 256 ;;
      bh_warp)       run $d A_bh_warp_${ic}_${n} setA.csv --kernel bh_warp --n $n --ic $ic $ICCOMMON --batch $b $L7 ;;
    esac
  done
}
ic_B() {   # ic n pass   (protocol default leaf 16 b128)
  local ic=$1 n=$2 P=$3 d=$R/ic/$3 k
  local ks=(bh_sorted bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    run $d B_${k}_${ic}_${n} setB.csv --kernel $k --n $n --ic $ic $ICCOMMON --batch $(batch $n) --leaf 16 --block 128
  done
}
ic_K() {   # ic spec pass
  local ic=$1 spec=$2 P=$3 d=$R/ic/$3 k
  local ks=(bh_sorted bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    run $d K_${k}_${ic}_${spec} setK.csv --kernel $k --n 262144 --ic $ic --mass-spectrum $spec $ICCOMMON --leaf 16 --block 128
  done
}
ic_T() {   # pass   (warp-unit timelines, galaxies 65536)
  local P=$1 d=$R/ic/$1 k
  local ks=(bh_sorted bh_persistent bh_warp)
  [ $P = p2 ] && ks=($(rev "${ks[@]}"))
  for k in "${ks[@]}"; do
    case $k in
      bh_warp) run $d T_${k}_galaxies_65536 - --kernel $k --n 65536 --ic galaxies $ICCOMMON --batch 5 --leaf 16 --block 128 --no-validate --timeline-unit warp --timeline $d/timeline/T_${k}_galaxies_65536.csv ;;
      *)       run $d T_${k}_galaxies_65536 - --kernel $k --n 65536 --ic galaxies $ICCOMMON --batch 5 --leaf 32 --block 256 --no-validate --timeline-unit warp --timeline $d/timeline/T_${k}_galaxies_65536.csv ;;
    esac
  done
}
ic_X() {   # rep ic pass   (as written: old then new binary, per kernel)
  local rep=$1 ic=$2 d=$R/ic/p1 k
  for k in bh_naive bh_sorted; do
    run $d X_old_${k}_${ic}_r${rep} - @OLD --kernel $k --n 262144 --ic $ic $ICCOMMON --leaf 16 --block 128 --no-validate
    run $d X_new_${k}_${ic}_r${rep} - --kernel $k --n 262144 --ic $ic $ICCOMMON --leaf 16 --block 128 --no-validate
  done
}
viewer_V2() {   # pass   (as in results/2026-10-17_ic_round/presentation/make_shots.sh, shot 17)
  local name=V2_disk_1M_rate_t3.0
  NRUNS=$((NRUNS + 1))
  local cmd="$V --dt 0.001 --steps-per-frame 10 --vsync 0 --frames 300 --screenshot $R/viewer/$name.bmp --ic disk --n 1048576 --kernel bh_warp --dist 3 --yaw 0 --pitch 0.4"
  if [ "$DRY" = 1 ]; then out "    $cmd > $R/viewer/$name.txt"; return; fi
  CELL_LOGS+=("$R/viewer/$name.txt")
  $cmd > $R/viewer/$name.txt 2>&1
  echo "$name exit=$? $(tail -1 $R/viewer/$name.txt)"
}

# ================================ session ======================================================
if [ "$DRY" = 1 ]; then
  : > $R/dryrun_commands.txt
  out "# DRY RUN $(date '+%F %T'): commands only, no GPU work. Binary: $B"
else
  for blk in tree2/p1 tree2/p2 ic/p1 ic/p2; do mkdir -p $R/$blk/logs $R/$blk/warps $R/$blk/timeline; done
  mkdir -p $STAGE $R/viewer
  [ -f $R/idle_log.csv ] || echo "time,cell,phase,idle_util_pct" > $R/idle_log.csv
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,pstate,clocks_event_reasons.active \
      --format=csv -lms 200 -f $R/gpu_clock_log$SFX.csv &
  CLOG=$!
  trap 'kill $CLOG 2>/dev/null' EXIT
  sha256sum $B $V $OLD > $R/binary_hashes$SFX.txt
  echo "session start $(date '+%F %T') (resume from cell $RES)" | tee -a $R/session_times.txt
  m=$(idle_med); log_idle start start "$m"
  $B --selftest > $R/selftest$SFX.txt 2>&1; echo "selftest exit=$?" | tee -a $R/session_times.txt
fi
[ "$DRY" = 1 ] && out "    $B --selftest > $R/selftest.txt"

out "# ---- tree round 2"
for ic in plummer core; do for n in 65536 262144 1048576; do cell "T2-A $ic $n" both t2_A $ic $n; done; done
for rep in 1 2 3; do for ic in plummer core; do cell "T2-A2 r$rep $ic" once t2_A2 $rep $ic; done; done
for ic in plummer core; do for n in 65536 1048576; do cell "T2-B $ic $n" both t2_B $ic $n; done; done
for n in 262144 1048576; do for leaf in 4 8 16 32; do for b in 64 128 256; do
  cell "T2-C n$n l$leaf b$b" both t2_C $n $leaf $b
done; done; done
for n in 8192 16384 24576 32768 49152 65536; do cell "T2-E $n" both t2_E $n; done
for th in 0.3 0.7 1.0; do cell "T2-F th$th" both t2_F $th; done
cell "T2-G th0" both t2_G
[ "$DRY" = 1 ] || echo "tree round 2 done $(date '+%F %T')" | tee -a $R/session_times.txt

out "# ---- IC round"
for ic in plummer disk galaxies cube_shell cube_edges uniform; do for n in 65536 262144 1048576; do
  cell "IC-A $ic $n" both ic_A $ic $n
done; done
for ic in plummer disk galaxies cube_shell cube_edges uniform; do for n in 65536 262144 1048576; do
  cell "IC-B $ic $n" both ic_B $ic $n
done; done
for spec in "plummer salpeter" "plummer heavy-few" "disk salpeter"; do set -- $spec; cell "IC-K $1 $2" both ic_K $1 $2; done
cell "IC-T galaxies 65536" both ic_T
for rep in 1 2; do for ic in plummer core; do cell "IC-X r$rep $ic" once ic_X $rep $ic; done; done
[ "$DRY" = 1 ] || echo "IC round done $(date '+%F %T')" | tee -a $R/session_times.txt

out "# ---- viewer V2 (alone; every bench process has exited)"
cell "V2 viewer disk 1M" once viewer_V2
if [ "$DRY" = 1 ]; then
  out "    python tools/bmp2png.py $R/viewer/*.bmp --delete"
  out "# total: $NCELLS cells, $NRUNS invocations (+ selftest)"
  echo "dry run: $NCELLS cells, $NRUNS invocations -> $R/dryrun_commands.txt"
else
  python tools/bmp2png.py $R/viewer/*.bmp --delete
  rmdir $STAGE 2>/dev/null
  echo "session end $(date '+%F %T'): $NCELLS cells, $NRUNS invocations" | tee -a $R/session_times.txt
  echo ALL DONE
fi
