# NBodyProblemSim

Gravitational N-body simulation in CUDA, tuned for one machine (RTX 4080 SUPER, sm_89),
with a study of GPU load-balancing techniques. Master's course project, Technical University of Sofia, Plovdiv Branch, Faculty of Electronics and Automation (FEA).

## Build

Needs CUDA 13.4 and VS 2026 Insiders with the C++ workload (CMake and Ninja are bundled).

```bat
tools\build.bat Release
```
Or just double-click `run.bat`: it builds if needed, then starts the viewer (see *Viewer* below).

**Visual Studio 2026 (Insiders):** File → Open → Folder… → `D:\NBodyProblemSim`. VS reads
`CMakePresets.json`. Pick **x64 Release**, then select `nbody_viewer.exe` or `nbody_bench.exe` as
the startup item and press F5. The build output goes to `out/build/<preset>/`. The CUDA 13.4 VS
integration is installed, so `.cu` files get syntax highlighting and Nsight menus.

**Reading order for studying the code:** `src/core/nbody.h` → `interaction.cuh` →
`kernels_allpairs.cu` → `kernels_multi.cu` → `kernels_tree.cu` (with `docs/03-design.md`
§10–§16). `src/bench/main.cpp` is the measurement and validation harness.

## Run

```bat
build\nbody_bench.exe --list
build\nbody_bench.exe --kernel allpairs_tiled --n 131072 --runs 10 --csv results\run.csv
build\nbody_bench.exe --kernel allpairs_tiled --n 4096 --steps 2000 --energy
build\src\viewer\nbody_viewer.exe --n 65536
```

### Viewer (live demo)

**One click:** `run.bat` (double-click, or from a prompt). It rebuilds Release first if the viewer is
missing or older than anything under `src\`, then starts two colliding disk galaxies with `bh_warp`,
N = 262144, colour by origin, load-balance statistics on, window maximised. Extra arguments are
appended and override the defaults: `run.bat --ic merger3`, `run.bat --kernel bh_naive --n 65536`.
`run-bench-selftest.bat` does the same for the bench and runs `--selftest` (last line PASS/FAIL).

**Panel** (F1 or Tab toggles; the window sections collapse). While the mouse is over the panel or a
field is being edited, the panel gets the mouse and keyboard, so camera keys and drags don't fire.
- *Scenario:* IC (the 8 suite ICs + viewer-only presets `merger3`, `flyby`, `unequal`), mass spectrum,
  bh-mass, seed, N (slider in powers of two + exact entry); **Apply** regenerates, **Reset** restarts
  the current scenario, **Revert** drops the edits.
- *Kernel:* every registry kernel (hover for its description), θ, leaf, block, steps/frame, dt,
  pause / step once.
- *Display:* colour by speed / mass / **origin** (one palette per galaxy or cluster: blue-white, gold,
  green, magenta; bulges paler; central bodies large white points), brightness, point size, vsync,
  auto-rotate, camera reset, UI scale (raise it on a projector).
- *Gravity well:* enable, strength (negative repels), softening.

**Live statistics:** FPS, simulated time, ms/step (+ rolling plot), the **real** interaction rate
(tree kernels: counted interactions, never N²-equivalent) and interactions/body; tree phase times
(keys / sort / tree / order / traverse); **measure load balance** runs the instrumented twin every K
frames and shows per-warp max/mean, SIMT efficiency, warp-cost spread, tree depth and stack need, with
plots. Switch `bh_naive` → `bh_sorted` → `bh_warp` live to watch R3 #4/#5/#7 change them. *Energy:*
"measure energy now" runs an exact O(N²) GPU sum (N ≤ 1M; auto every 5 s up to 262144); the first
measurement after a reset is E0. These are a display: quote `nbody_bench`, not the viewer.

**Gravity well (hold right mouse button):** a 2D cursor only fixes a ray, so the viewer takes the
point where that ray meets the plane through the orbit target perpendicular to the view direction
(the plane the camera orbits about). Orbit with the left button to tilt that plane and reach any 3D
point. The well is a softened point mass applied by a separate viewer-only kick after each step; the
bench, the force kernels and the validation never see it. While it acts the run is marked
**modified by user interaction** (energy is no longer conserved) until the next reset.

**Keys:** LMB drag orbit · wheel zoom · RMB hold gravity well · F1 / Tab panel · `1`..`9` kernel ·
Space pause · `+`/`-` steps per frame · `PageUp`/`.` double N · `PageDown`/`,` halve N · `I` next IC
(incl. presets) · `H` next mass spectrum · `M` speed/mass colour · `C` cycle speed/mass/origin ·
`[` `]` brightness · `R` reset · `V` vsync · Esc quit.
N is capped at 262144 for all-pairs and 4,194,304 for tree kernels (TDR margin); default kernel `bh_warp`.

**Command line:** `--ic plummer|collision|core|disk|galaxies|cube_shell|cube_edges|uniform|merger3|flyby|unequal`,
`--mass-spectrum equal|salpeter|heavy-few`, `--bh-mass 0.01` (presets are viewer-only; the bench
rejects them), `--color speed|mass|origin`, `--lb 1 --lb-every 30`, `--panel 0|1`, `--ui-scale 1.5`,
`--point-size`, `--auto-rotate 1`, `--maximize 1`, `--well-mass 0.5 --well-eps 0.1`, `--energy-check 1`.
`--frames F --screenshot out.bmp` renders F frames, saves the last one and exits (the panel is hidden
unless `--panel 1`; `--dist/--yaw/--pitch` set the camera, `--steps-per-frame`/`--vsync 0`
fast-forward, the simulated time is printed on exit, `python tools/bmp2png.py out.bmp` converts to PNG).
`--test-keys "F:act,..."` scripts actions for smoke tests: the old `up down ic spectrum mass` plus
`kernel=NAME ic=NAME n=N seed=S spectrum=S theta=X leaf=K block=B dt=X spf=K lb=0|1 lbevery=K energy
reset pause step panel=0|1 open=well|energy well=FX/FY wellup wellmass=X welleps=X color=MODE`.

![viewer](docs/img/viewer_plummer_32k_2026-10-17.png)

## Layout

| Path | What |
|---|---|
| `docs/` | requirements, hardware, approach, decisions, design |
| `references/` | verified papers + NVIDIA docs (literature-scout) |
| `src/core/` | kernels, integrator, initial conditions (`nbody_core` lib) |
| `src/bench/` | benchmark/validation CLI |
| `src/viewer/` | CUDA-OpenGL interop viewer (GLFW + glad + ImGui panel, viewer-only kernels) |
| `third_party/` | vendored GLFW 3.4 + glad + Dear ImGui 1.91.5 (pinned, see its README) |
| `results/` | benchmark sessions: env + CSV + summary |
| `.claude/agents/` | role agents: architect, scout, critic, profiler, writer |
