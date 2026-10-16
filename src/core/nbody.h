// Public API of nbody_core. All state is GPU-resident; the host only sees it on request.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace nbody {

struct Params {
    int   n          = 1 << 14;
    float dt         = 1e-3f;
    float softening  = 1e-2f;   // Plummer softening length eps; kernels use eps^2
    float G          = 1.0f;    // N-body units
    int   blockSize  = 256;
};

// Device buffers. pos.w = mass, vel.w unused (padding for 16-byte loads).
struct DeviceState {
    float4* pos = nullptr;
    float4* vel = nullptr;
    float4* acc = nullptr;
    int     n   = 0;
};

// Every force kernel variant has this signature: compute state.acc from state.pos.
// Load-balancing variants are added as new entries in the registry, not new APIs,
// so the bench can compare them by name.
using ForceLauncher = void (*)(const DeviceState&, const Params&, cudaStream_t);

struct KernelInfo {
    const char*   name;
    const char*   description;
    ForceLauncher launch;
};

const std::vector<KernelInfo>& kernelRegistry();
const KernelInfo*              findKernel(const std::string& name);

// Lifecycle
DeviceState allocate(int n);
void        release(DeviceState& s);
void        upload(DeviceState& s, const std::vector<float4>& pos, const std::vector<float4>& vel);
void        download(const DeviceState& s, std::vector<float4>& pos, std::vector<float4>& vel);
void        downloadAcc(const DeviceState& s, std::vector<float4>& acc);

// Leapfrog KDK. Requires acc to be valid on entry (call the force kernel once first).
void step(DeviceState& s, const Params& p, const KernelInfo& k, cudaStream_t stream = 0);

// Initial conditions (host). Plummer sphere in virial equilibrium, total mass 1.
void makePlummer(int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel);

// Diagnostics (host, double precision, O(N^2): use for N <= ~65k).
struct Energy { double kinetic, potential, total; };
Energy computeEnergy(const std::vector<float4>& pos, const std::vector<float4>& vel, const Params& p);

// Validation: max and RMS relative error of acc vs reference acc.
struct ErrorStats { double maxRel, rmsRel; };
ErrorStats compareAcc(const std::vector<float4>& acc, const std::vector<float4>& ref);

// FLOP convention for reporting: 20 FLOP per pairwise interaction
// (Nyland, Harris & Prins, GPU Gems 3 ch. 31). Stated in every report.
constexpr double kFlopsPerInteraction = 20.0;

} // namespace nbody

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),       \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                        \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)
