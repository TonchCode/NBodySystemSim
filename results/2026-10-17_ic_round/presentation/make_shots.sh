#!/usr/bin/env bash
# R6 presentation screenshots (IC round). Run from the repo root:
#   bash results/2026-10-17_ic_round/presentation/make_shots.sh
# dt = 1e-3 (<= the 1.6e-3 advisory for a 1% central mass), 10 steps per frame, no vsync, 1600x900.
# Simulated time at the shot = frames x 10 x 1e-3 (printed by the viewer into the .txt next to each PNG).
set -u
P=results/2026-10-17_ic_round/presentation
V=./build/src/viewer/nbody_viewer.exe
shot() {   # name, frames, args...
  local name=$1 frames=$2; shift 2
  $V --dt 0.001 --steps-per-frame 10 --vsync 0 --frames $frames --screenshot $P/$name.bmp "$@" > $P/$name.txt 2>&1
  echo "$name exit=$? $(tail -1 $P/$name.txt)"
}
G="--ic galaxies --n 262144 --kernel bh_warp --pitch 0.7"
shot 01_galaxies_t0.0      1   $G --dist 6
shot 02_galaxies_t1.0      100 $G --dist 6
shot 03_galaxies_t2.0      200 $G --dist 5
shot 04_galaxies_t3.0      300 $G --dist 5
shot 05_galaxies_t4.5      450 $G --dist 6
shot 06_galaxies_t8.0      800 $G --dist 9
D="--ic disk --n 262144 --kernel bh_warp"
shot 07_disk_faceon_t0.0   1   $D --dist 3 --yaw 0 --pitch 0
shot 08_disk_faceon_t0.5   50  $D --dist 3 --yaw 0 --pitch 0
shot 09_disk_edgeon_t0.5   50  $D --dist 3 --yaw 0 --pitch 1.55
shot 10_cube_shell_t0.0    1   --ic cube_shell --n 262144 --kernel bh_warp --dist 5
shot 11_cube_shell_t1.2    120 --ic cube_shell --n 262144 --kernel bh_warp --dist 5
shot 12_cube_edges_t0.0    1   --ic cube_edges --n 262144 --kernel bh_warp --dist 5
shot 13_cube_edges_t1.2    120 --ic cube_edges --n 262144 --kernel bh_warp --dist 5
shot 14_plummer_t0.5       50  --ic plummer --n 262144 --kernel bh_warp --dist 4
shot 15_heavyfew_mass_t0.2 20  --ic plummer --mass-spectrum heavy-few --n 65536 --kernel bh_warp --dist 3.5 --mass-color 1
shot 16_disk_salpeter_mass_t0.2 20 --ic disk --mass-spectrum salpeter --n 65536 --kernel bh_warp --dist 3 --yaw 0 --pitch 0.5 --mass-color 1
# V2 (scored): real rate of bh_warp on disk at 1M, title after 300 frames (3000 steps)
shot 17_disk_1M_rate_t3.0  300 --ic disk --n 1048576 --kernel bh_warp --dist 3 --yaw 0 --pitch 0.4
python tools/bmp2png.py $P/*.bmp --delete
