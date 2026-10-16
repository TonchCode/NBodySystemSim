cmake_minimum_required(VERSION 3.24)
project(NBodyProblemSim LANGUAGES CXX CUDA)

# Single target device: RTX 4080 SUPER (Ada AD103). See docs/01-hardware.md.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NBODY_BUILD_VIEWER "Build the OpenGL interop viewer" OFF)

find_package(CUDAToolkit REQUIRED)

# -lineinfo: source correlation in Nsight Compute; --resource-usage: register/smem report per kernel
set(NBODY_CUDA_FLAGS -lineinfo --resource-usage)

# ---- core: kernels, integrator, initial conditions -------------------------
add_library(nbody_core STATIC
  src/core/nbody.cu
  src/core/kernels_allpairs.cu
  src/core/init.cpp
)
target_include_directories(nbody_core PUBLIC src/core)
target_compile_options(nbody_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NBODY_CUDA_FLAGS}>)
target_link_libraries(nbody_core PUBLIC CUDA::cudart)

# ---- bench: CLI harness, CSV output ----------------------------------------
add_executable(nbody_bench src/bench/main.cpp)
target_link_libraries(nbody_bench PRIVATE nbody_core)

# ---- viewer: CUDA-OpenGL interop (decision: docs/decisions.md D-001) -------
if(NBODY_BUILD_VIEWER)
  add_subdirectory(src/viewer)
endif()
