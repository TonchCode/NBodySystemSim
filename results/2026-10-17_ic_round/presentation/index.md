# IC round: presentation screenshots (R6)

Made by `make_shots.sh` (in this folder) with `build/src/viewer/nbody_viewer.exe`, 1600×900, PNG via
`tools/bmp2png.py`. Every shot: dt = 1e-3 (below the 1.6e-3 advisory for a 1% central mass, D-029),
10 steps per frame, θ = 0.5, leaf 16, ε = 0.01, seed 42, colour = speed (blue slow → warm white fast)
unless "mass colour" is noted (red, larger points = bodies above 3× the mean mass, log-scaled).
t = simulated time at the shot (N-body units), printed by the viewer into the `.txt` next to each PNG
(which also holds the title line: ms/step and the counted interaction rate, under the session's
background load). Camera: distance / yaw / pitch in radians, looking at the origin, +y up.

| File | IC | N | Kernel | t | Camera (dist, yaw, pitch) | What it shows |
|---|---|---|---|---|---|---|
| `01_galaxies_t0.0.png` | galaxies | 262144 | bh_warp | 0.01 | 6, 0.6, 0.7 | the two inclined disks at the start |
| `02_galaxies_t1.0.png` | galaxies | 262144 | bh_warp | 1.0 | 6, 0.6, 0.7 | approach; both cold disks already form rings/bars |
| `03_galaxies_t2.0.png` | galaxies | 262144 | bh_warp | 2.0 | 5, 0.6, 0.7 | first contact, tidal bridge |
| `04_galaxies_t3.0.png` | galaxies | 262144 | bh_warp | 3.0 | 5, 0.6, 0.7 | first passage: disks disrupted into clumps |
| `05_galaxies_t4.5.png` | galaxies | 262144 | bh_warp | 4.5 | 6, 0.6, 0.7 | after the passage: merging core plus ejected clumps |
| `06_galaxies_t8.0.png` | galaxies | 262144 | bh_warp | 8.0 | 9, 0.6, 0.7 | merged remnant (spheroid) with escaping clumps |
| `07_disk_faceon_t0.0.png` | disk | 262144 | bh_warp | 0.01 | 3, 0, 0 | face-on initial disk + bulge |
| `08_disk_faceon_t0.5.png` | disk | 262144 | bh_warp | 0.5 | 3, 0, 0 | face-on: a ring forms (cold-disk instability, Q ≪ 1, no halo: expected, D-029) |
| `09_disk_edgeon_t0.5.png` | disk | 262144 | bh_warp | 0.5 | 3, 0, 1.55 | edge-on: thin disk (z0 = 0.025) and the bulge |
| `10_cube_shell_t0.0.png` | cube_shell | 262144 | bh_warp | 0.01 | 5, 0.6, 0.35 | stress test: bodies on the 6 faces |
| `11_cube_shell_t1.2.png` | cube_shell | 262144 | bh_warp | 1.2 | 5, 0.6, 0.35 | cold collapse: edges and corners concentrate first |
| `12_cube_edges_t0.0.png` | cube_edges | 262144 | bh_warp | 0.01 | 5, 0.6, 0.35 | stress test: bodies on the 12 edges |
| `13_cube_edges_t1.2.png` | cube_edges | 262144 | bh_warp | 1.2 | 5, 0.6, 0.35 | collapse of the wire frame towards the corners |
| `14_plummer_t0.5.png` | plummer | 262144 | bh_warp | 0.5 | 4, 0.6, 0.35 | the reference sphere (equilibrium) |
| `15_heavyfew_mass_t0.2.png` | plummer, heavy-few | 65536 | bh_warp | 0.2 | 3.5, 0.6, 0.35 | **mass colour**: the 16 bodies with 1% of the mass each (red) |
| `16_disk_salpeter_mass_t0.2.png` | disk, salpeter | 65536 | bh_warp | 0.2 | 3, 0, 0.5 | **mass colour**: Salpeter-like spectrum, heaviest bodies red |
| `17_disk_1M_rate_t3.0.png` | disk | 1048576 | bh_warp | 3.0 | 3, 0, 0.4 | 1M bodies live; its title is the V2 rate reading (318 G int/s real, at t = 3) |

**Honest caption for the galaxy sequence:** the disks have no dark halo and are cold (Toomre Q ≪ 1), so
they fragment into clumps within about one rotation, before and during the collision (D-029). The
sequence shows a real N-body merger of two self-gravitating disks, not a model of the Milky Way. All
viewer numbers (ms/step, rates) were taken under the session's uncontrolled background load (30–37%).
