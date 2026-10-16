# NBodyProblemSim

Gravitational N-body simulation in CUDA, tuned for one machine (RTX 4080 SUPER, sm_89),
with a study of GPU load-balancing techniques. Master's course project, Technical University of Sofia, Plovdiv Branch, Faculty of Electronics and Automation (FEA).

## Build

Needs CUDA 13.4 and VS 2026 Insiders with the C++ workload (CMake and Ninja are bundled).

```bat
tools\build.bat Release
```

## Run

```bat
build\nbody_bench.exe --list
build\nbody_bench.exe --kernel allpairs_tiled --n 131072 --runs 10 --csv results\run.csv
build\nbody_bench.exe --kernel allpairs_tiled --n 4096 --steps 2000 --energy
```

## Layout

| Path | What |
|---|---|
| `docs/` | requirements, hardware, approach, decisions, design |
| `references/` | verified papers + NVIDIA docs (literature-scout) |
| `src/core/` | kernels, integrator, initial conditions (`nbody_core` lib) |
| `src/bench/` | benchmark/validation CLI |
| `src/viewer/` | OpenGL interop viewer (pending D-004) |
| `results/` | benchmark sessions: env + CSV + summary |
| `.claude/agents/` | role agents: architect, scout, critic, profiler, writer |
