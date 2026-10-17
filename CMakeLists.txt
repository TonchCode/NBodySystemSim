cmake_minimum_required(VERSION 3.24)
project(NBodyProblemSim LANGUAGES C CXX CUDA)   # C: GLFW + glad loader

# Single target device: RTX 4080 SUPER (Ada AD103). See docs/01-hardware.md.
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NBODY_BUILD_VIEWER "Build the OpenGL interop viewer" ON)

find_package(CUDAToolkit REQUIRED)

# -lineinfo: source correlation in Nsight Compute; --resource-usage: register/smem report per kernel
set(NBODY_CUDA_FLAGS -lineinfo --resource-usage)

# ---- core: kernels, integrator, initial conditions -------------------------
add_library(nbody_core STATIC
  src/core/nbody.cu
  src/core/kernels_allpairs.cu
  src/core/kernels_tuned.cu     # R3 #2
  src/core/kernels_multi.cu     # R3 #3
  src/core/kernels_tree.cu      # R3 #4, #5
  src/core/init.cpp
)
target_include_directories(nbody_core PUBLIC src/core)
# CUB (CCCL, shipped with CUDA 13.4) requires MSVC's conforming preprocessor. Only the tree TU
# includes CUB, so the flag is scoped to it (host-side only; device code is unaffected).
if(MSVC)
  set_source_files_properties(src/core/kernels_tree.cu PROPERTIES COMPILE_OPTIONS "-Xcompiler=/Zc:preprocessor")
endif()
target_compile_options(nbody_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NBODY_CUDA_FLAGS}>)
target_link_libraries(nbody_core PUBLIC CUDA::cudart)

# ---- bench: CLI harness, CSV output ----------------------------------------
add_executable(nbody_bench src/bench/main.cpp)
target_link_libraries(nbody_bench PRIVATE nbody_core)

# ---- viewer: CUDA-OpenGL interop (decisions D-001, D-004) -------------------
if(NBODY_BUILD_VIEWER)
  set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)
  add_subdirectory(src/viewer)
endif()
