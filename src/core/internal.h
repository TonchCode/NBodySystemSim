// Helpers shared by the kernel translation units (not part of the public API).
#pragma once
#include "nbody.h"

namespace nbody {

inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }

// Fill LaunchInfo from the compiled kernel: registers, static smem, runtime occupancy.
LaunchInfo describeKernel(const void* fn, int gridBlocks, int blockThreads, size_t dynSmem);

} // namespace nbody
