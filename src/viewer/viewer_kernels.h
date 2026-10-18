// Viewer-only GPU helpers (D-032). These kernels live in the nbody_viewer target, NOT in nbody_core:
// the bench, the production force kernels and the validation never link or call them.
#pragma once
#include <cuda_runtime.h>

namespace viewer {

// Mouse gravity well: v_i += dt * G * Mw * (w - x_i) / (|w - x_i|^2 + epsW^2)^1.5 for every body.
// A separate kick after the KDK step (first-order operator splitting). It changes velocities only, so the
// stored acc (the force kernel's output at the current positions) stays valid for the next step's first kick.
// The well is an EXTERNAL, moving potential: total energy is not conserved while it acts.
void wellKick(float4* vel, const float4* pos, int n, float3 w, float massW, float epsW, float G, float dt,
              cudaStream_t st = 0);

// Total kinetic and potential energy, exact all-pairs, on the GPU (O(N^2), on request only).
// Per body: phi_i = sum_{j != i} m_j / sqrt(r_ij^2 + eps^2), pair terms in FP32 accumulated per 256-body
// tile in FP32 and across tiles in FP64; KE_i = m_i |v_i|^2 / 2 in FP64. PE = -G/2 sum_i m_i phi_i.
// Launched in chunks of bodies (<= 2^33 pairs per launch) to stay far below the WDDM TDR limit.
struct EnergyResult { double kinetic = 0, potential = 0, total = 0, ms = 0; };
EnergyResult energyGPU(const float4* pos, const float4* vel, int n, float softening, float G);

} // namespace viewer
