// Public API of nbody_core. All state is GPU-resident; the host only sees it on request.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace nbody {

// Per-block timeline record (instrumented builds only, R3 imbalance measurement without ncu).
// t0/t1: %globaltimer (ns, one clock for the whole GPU) at block entry and after the block's
// last thread has finished (a final __syncthreads). c0/c1: clock64() on that SM (cycles; not
// synchronised across SMs, but finer-grained, and c/t gives the SM clock during the block).
struct BlockStamp {
    unsigned long long t0, t1, c0, c1;
    unsigned           smid, block;
};

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
    // R3 #3: register prefetch of the next tile in allpairs_multi (1 = on, the round-1
    // behaviour; 0 = off). An independent switch so prefetch and q can be measured
    // separately (critic review 1, #4).
    int   prefetch       = 1;
    // Device buffer of >= gridBlocks records. nullptr = production kernels. Non-null = the
    // instrumented instantiation (tiled, tuned, multi), which writes one BlockStamp per block.
    // The production instantiations are compiled without the stamping code (template flag).
    BlockStamp* timeline = nullptr;
    // R3 #4/#5 (bh_*): opening parameter of the bmax criterion (accept a node iff
    // |x - com| >= bmax / theta; theta = 0 opens everything = exact all-pairs order-changed sum)
    // and the leaf-bucket size (nodes with <= leafSize bodies are summed directly).
    float theta    = 0.5f;
    int   leafSize = 16;
    // bh_*: per-thread work counters of the instrumented traversal (uint4 per traversal thread:
    // nodes popped, interactions, leader pops, leader interactions). nullptr = not collected.
    // Non-null selects the instrumented twin, like timeline.
    uint4* workCounters = nullptr;
    // Kernels with LaunchInfo::nPhases > 0: if non-null, an array of nPhases + 1 events; the
    // launcher records [0] before the first phase and [k] after phase k (GTaP-style phase times).
    cudaEvent_t* phaseEvents = nullptr;
    // Tree round 2 (docs/03-design.md §17). timelineWarp = 1: the instrumented tree twins write
    // one BlockStamp per warp work unit (32 thread slots; for bh_persistent_lane one per
    // persistent warp) instead of one per block; LaunchInfo::timelineRecords says how many.
    int timelineWarp = 0;
    // Occupancy sweep: extra dynamic shared memory per traversal block (bytes, unused by the
    // kernel) to cap resident blocks/SM at fixed N (critic review 2, #6). 0 = off.
    int smemPad = 0;
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
    int    prefetch       = -1;  // multi: 0/1 as compiled; -1 = not applicable
    bool   instrumented   = false;
    int    regsPerThread  = 0;
    size_t smemPerBlock   = 0;   // static + dynamic, bytes (excl. the 1 KB system reservation)
    int    blocksPerSM    = 0;   // max resident blocks per SM for this config
    // Tree kernels: forces are approximate (theta-dependent truncation error), and the launch
    // is a pipeline of phases (build + traversal). gridBlocks/regs above describe the traversal.
    bool        approximate   = false;
    int         nPhases       = 0;
    const char* phaseNames[8] = {};
    // Tree round 2.
    int         timelineRecords   = 0;        // records per launch (0 = gridBlocks)
    int         timelineSlotsPerSM = 0;       // concurrent records per SM (blocks or warps/SM)
    bool        persistent        = false;
    bool        control           = false;    // diagnostic control: forces not all computed, no gates
    const char* errorSibling      = nullptr;  // kernel whose error this one must not exceed (TG3)
    const char* exactSibling      = nullptr;  // kernel this one must equal bitwise (+ equal counters)
    const char* counterSibling    = nullptr;  // kernel whose work counters are the overhead baseline
    size_t      dynSmem           = 0;        // dynamic shared memory of the traversal launch
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
// Named initial conditions (docs/03-design.md §13), all total mass 1, equal-mass bodies:
//   plummer    = makePlummer (bitwise identical)
//   collision  two Plummer spheres (N/2 each, virial radius 0.25) at x = -/+1, y = -/+0.125,
//              approaching at -/+0.35 along x
//   core       Plummer halo (3N/4, virial radius 1) + dense Plummer core (N/4, virial radius 0.03)
// Multi-component ICs are randomly shuffled (seeded), so the body order carries no spatial
// information, exactly like the single Plummer. Returns false for an unknown name.
bool makeIC(const std::string& name, int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel);
const char* icNames();   // "plummer|collision|core"

// Exact reference accelerations (device, FP32): the allpairs_naive loop, launched in chunks of
// bodies so each launch stays far below the TDR limit at N ~ 1M. Bitwise equal to allpairs_naive.
void referenceAcc(const DeviceState& s, const Params& p, std::vector<float4>& acc);

// Tree diagnostics (bh_* kernels). checkLastTree verifies the most recently built tree on the
// host: orders are permutations, sortedPos = pos[perm], keys monotone, every body in exactly
// one leaf, node ranges contiguous, masses additive, boxes nested, root mass / COM conserved.
struct TreeCheck {
    bool        ok = true;
    std::string msg;
    int         n = 0, internalNodes = 0, maxDepth = 0, leafBuckets = 0, leafBodies = 0;
    double      massRelErr = 0, comAbsErr = 0;
};
TreeCheck checkLastTree(const DeviceState& s);
int       treeStackOverflows(bool reset);
// Exact theta > 0 check (critic review 2, #7), host, FP64, for the most recent tree:
//  (1) every node's mass, COM and AABB recomputed from its [first, last] range of sortedPos and
//      compared with the device values (mass/COM at FP32-roundoff tolerance, AABB bitwise), and
//      the opening radius rc2 against the FP64 bmax;
//  (2) a host walk of the same tree with the same FP32 criterion (bitwise-identical decisions:
//      same fmaf expression, device com/rc2) and FP64 accumulation, per body: the interaction and
//      pop counts must equal the device counters exactly, and |a_dev - a_host| / sum|f| must be
//      within (I + 10) u (FP32 summation bound). unionWalk = the #7 warp-union criterion.
// acc: device result indexed by body; wc: device counters indexed by thread slot (Morton order).
struct TreeExactCheck {
    bool ok = true; std::string msg;
    int nodes = 0, nodeFails = 0, bodies = 0, countMismatch = 0, errFails = 0;
    double maxMassRel = 0, maxComAbs = 0, maxBmaxAbs = 0, maxErrOverBound = 0;
};
TreeExactCheck checkLastTreeExact(const std::vector<float4>& acc, const std::vector<uint4>* wc, float eps2, float G,
                                  bool unionWalk);   // traversal stack overflows since the last reset (must be 0)
unsigned long long hostHilbertKey(unsigned x, unsigned y, unsigned z);   // 21-bit coords (for --selftest)

// Conditioning pre-filter (device, FP32, O(N^2), diagnostic only, not a force kernel):
// scale[i] = G * sum_j |f_ij| / m_i. kappa_i ~ scale[i] / |a_i|. Used to pick the
// candidate worst-conditioned bodies, whose kappa is then recomputed in FP64.
void forceScale(const DeviceState& s, const Params& p, std::vector<float>& scale);

// Timeline clock calibration. %globaltimer ticks in 1.024 us steps on this board (measured,
// results/2026-10-17_round2/phase1_checks), too coarse for 10-30 us kernels, while clock64
// is per-SM and unsynchronised. Each record: on SM smid, the first clock64 value seen after
// a %globaltimer tick edge (tA, cA), then again after spinning ~spinUs (tB, cB). With one
// record before and one after the timeline launches, every SM's cycle counter maps onto
// global ns: t(c) = tA + (c - cA) / f, f = cycles per ns between the two anchors.
struct ClockCal { unsigned smid; unsigned long long tA, cA, tB, cB; };
void clockCalibration(std::vector<ClockCal>& out, int spinUs = 500);

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
double kappaOf(const Fp64Sample& r, size_t k);   // scale / |a| of sample k (0 if a == 0)
Fp64Sample referenceFp64(const std::vector<float4>& pos, const Params& p, const std::vector<int>& idx);

// Errors of an FP32 acc array vs the FP64 sample. normMax = max |da| / scale, which is
// the conditioning-independent measure. Worst case for recursive summation in any order:
// normMax <= gamma_{N+c} = (N+c)u / (1 - (N+c)u), u = 2^-24, c ~ a few (rounding inside
// each term); the typical (random-rounding) size is ~ u * sqrt(N).
struct Fp64Error { double maxRel, rmsRel, normMax, kappaMax; };
Fp64Error compareFp64(const std::vector<float4>& acc, const Fp64Sample& ref);

// R3 #3: q chosen by a cost model of SM-level load balance (see docs/03-design.md sec. 4.4).
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
