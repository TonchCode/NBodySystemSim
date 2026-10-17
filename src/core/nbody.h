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
    // R3 #2 (allpairs_tuned): inner-loop unroll factor, one of {1, 4, 8, 16}; 0 = full tile.
    int   unroll         = 8;
    // R3 #3 (allpairs_multi): threads cooperating on one body, one of {1,2,4,8,16,32};
    // 0 = automatic (autoThreadsPerBody, a cost model of SM-level load balance).
    int   threadsPerBody = 0;
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

// What a launcher would actually launch for (n, params): resolved template parameters,
// compiler resources (cudaFuncGetAttributes) and the occupancy the runtime computes
// (cudaOccupancyMaxActiveBlocksPerMultiprocessor). Used by the bench to report
// occupancy and wave quantization next to every timing.
struct LaunchInfo {
    int    gridBlocks     = 0;
    int    blockThreads   = 0;
    int    threadsPerBody = 1;   // as resolved (auto -> concrete q)
    int    unroll         = 0;   // as compiled (0 = full tile); -1 = not applicable
    int    regsPerThread  = 0;
    size_t smemPerBlock   = 0;   // static + dynamic, bytes (excl. the 1 KB system reservation)
    int    blocksPerSM    = 0;   // max resident blocks per SM for this config
};
using LaunchQuery = LaunchInfo (*)(int n, const Params&);

struct KernelInfo {
    const char*   name;
    const char*   description;
    ForceLauncher launch;
    LaunchQuery   query = nullptr;   // optional
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

// FP64 ground truth for a sample of bodies (host, O(|idx| * N)). For each sampled body:
// the softened acceleration in double, and scale = G * sum_j |f_ij|, the magnitude the
// FP32 sum is built from. |a| << scale means the net force nearly cancels: such a body is
// ill-conditioned and its *relative* error is amplified by kappa = scale / |a|.
struct Fp64Sample { std::vector<int> idx; std::vector<double> ax, ay, az, scale; };
Fp64Sample referenceFp64(const std::vector<float4>& pos, const Params& p, const std::vector<int>& idx);

// Errors of an FP32 acc array vs the FP64 sample. normMax = max |da| / scale, which is
// the conditioning-independent measure (bounded by ~ gamma_N * u for any summation order).
struct Fp64Error { double maxRel, rmsRel, normMax, kappaMax; };
Fp64Error compareFp64(const std::vector<float4>& acc, const Fp64Sample& ref);

// R3 #3: q chosen by a cost model of SM-level load balance (see docs/03-design.md §4.4).
int autoThreadsPerBody(int n, int blockSize);

// Device facts used by heuristics (cached cudaDeviceGetAttribute).
int deviceSMCount();

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
