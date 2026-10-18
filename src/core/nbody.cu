// State management, integrator, kernel registry.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "nbody.h"
#include "interaction.cuh"
#include "internal.h"

namespace nbody {

void registerAllPairs(std::vector<KernelInfo>& r);
void registerTuned(std::vector<KernelInfo>& r);
void registerMulti(std::vector<KernelInfo>& r);
void registerTree(std::vector<KernelInfo>& r);
void registerHost(std::vector<KernelInfo>& r);          // host_kernels.cpp   (R3 #8 CPU baselines)
void registerHybridAllPairs(std::vector<KernelInfo>& r); // kernels_hybrid.cu  (R3 #8 body split, all-pairs)
void registerTreeHybrid(std::vector<KernelInfo>& r);     // kernels_tree.cu    (R3 #8 tree splits + GPU control)

int deviceSMCount()
{
    static const int sms = [] {
        int dev = 0, v = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
        return v;
    }();
    return sms;
}

LaunchInfo describeKernel(const void* fn, int gridBlocks, int blockThreads, size_t dynSmem)
{
    cudaFuncAttributes fa{};
    CUDA_CHECK(cudaFuncGetAttributes(&fa, fn));
    LaunchInfo li;
    li.gridBlocks    = gridBlocks;
    li.blockThreads  = blockThreads;
    li.unroll        = -1;
    li.regsPerThread = fa.numRegs;
    li.smemPerBlock  = fa.sharedSizeBytes + dynSmem;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&li.blocksPerSM, fn, blockThreads, dynSmem));
    return li;
}

const std::vector<KernelInfo>& kernelRegistry()
{
    static const std::vector<KernelInfo> registry = [] {
        std::vector<KernelInfo> r;
        registerAllPairs(r);   // naive, tiled: the references, keep them first (viewer keys 1, 2)
        registerTuned(r);      // R3 #2
        registerMulti(r);      // R3 #3
        registerTree(r);       // R3 #4, #5 (viewer keys 5, 6, 7)
        return r;
    }();
    return registry;
}

const KernelInfo* findKernel(const std::string& name)
{
    for (const auto& k : kernelRegistry())
        if (name == k.name) return &k;
    return nullptr;
}

const std::vector<KernelInfo>& hybridRegistry()
{
    static const std::vector<KernelInfo> registry = [] {
        std::vector<KernelInfo> r;
        registerHost(r);
        registerHybridAllPairs(r);
        registerTreeHybrid(r);
        return r;
    }();
    return registry;
}

const KernelInfo* findAnyKernel(const std::string& name)
{
    if (const KernelInfo* k = findKernel(name)) return k;
    for (const auto& k : hybridRegistry())
        if (name == k.name) return &k;
    return nullptr;
}

DeviceState allocate(int n)
{
    DeviceState s;
    s.n = n;
    CUDA_CHECK(cudaMalloc(&s.pos, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&s.vel, n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&s.acc, n * sizeof(float4)));
    return s;
}

void release(DeviceState& s)
{
    cudaFree(s.pos);
    cudaFree(s.vel);
    cudaFree(s.acc);
    s = DeviceState{};
}

void upload(DeviceState& s, const std::vector<float4>& pos, const std::vector<float4>& vel)
{
    CUDA_CHECK(cudaMemcpy(s.pos, pos.data(), s.n * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.vel, vel.data(), s.n * sizeof(float4), cudaMemcpyHostToDevice));
}

void download(const DeviceState& s, std::vector<float4>& pos, std::vector<float4>& vel)
{
    pos.resize(s.n);
    vel.resize(s.n);
    CUDA_CHECK(cudaMemcpy(pos.data(), s.pos, s.n * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel.data(), s.vel, s.n * sizeof(float4), cudaMemcpyDeviceToHost));
}

void downloadAcc(const DeviceState& s, std::vector<float4>& acc)
{
    acc.resize(s.n);
    CUDA_CHECK(cudaMemcpy(acc.data(), s.acc, s.n * sizeof(float4), cudaMemcpyDeviceToHost));
}

namespace {

// v += a*h ; optionally x += v*dt. Fused so a KDK step is: kick+drift, force, kick.
__global__ void kickDriftKernel(float4* pos, float4* vel, const float4* acc, int n,
                                float halfDt, float dt, bool drift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = vel[i];
    float4 a = acc[i];
    v.x = fmaf(a.x, halfDt, v.x);
    v.y = fmaf(a.y, halfDt, v.y);
    v.z = fmaf(a.z, halfDt, v.z);
    vel[i] = v;
    if (drift) {
        float4 p = pos[i];
        p.x = fmaf(v.x, dt, p.x);
        p.y = fmaf(v.y, dt, p.y);
        p.z = fmaf(v.z, dt, p.z);
        pos[i] = p;
    }
}

// Diagnostic, not a force kernel: scale_i = G * sum_j m_j |r_ij| / (r_ij^2 + eps^2)^(3/2),
// the sum of force magnitudes (per unit m_i) that body i's FP32 sum is built from. Only used
// to shortlist the worst-conditioned bodies (largest kappa = scale / |a|); their kappa is then
// recomputed in FP64 on the host (referenceFp64). FP32 is enough for ranking.
__global__ void forceScaleKernel(const float4* __restrict__ pos, float* __restrict__ scale, int n,
                                 int i0, int i1, float eps2, float G)
{
    int i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= i1) return;
    float4 bi = pos[i];
    float  sc = 0.f;
    for (int j = 0; j < n; ++j) {
        float4 bj = pos[j];
        float rx = bj.x - bi.x, ry = bj.y - bi.y, rz = bj.z - bi.z;
        float r2 = fmaf(rx, rx, fmaf(ry, ry, rz * rz));
        float d2 = r2 + eps2;
        float inv = rsqrtf(d2);
        sc = fmaf(bj.w * inv * inv * inv, sqrtf(r2), sc);
    }
    scale[i] = G * sc;
}

// Same loop as allPairsNaiveKernel (kernels_allpairs.cu), restricted to bodies [i0, i1):
// bitwise-equal results, but each launch is O(chunk * N), so N ~ 1M stays far under TDR.
__global__ void referenceChunkKernel(const float4* __restrict__ pos, float4* __restrict__ acc, int n,
                                     int i0, int i1, float eps2, float G)
{
    int i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= i1) return;
    float4 bi = pos[i];
    float3 a  = make_float3(0.f, 0.f, 0.f);
    for (int j = 0; j < n; ++j)
        a = bodyBodyInteraction(bi, pos[j], a, eps2);
    acc[i] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
}

// Bodies per launch: chunk * n <= 2^36 interactions (~0.1-0.2 s at the naive kernel's rate).
int referenceChunk(int n) { return int(std::max<long long>(256, std::min<long long>(n, (1ll << 36) / std::max(1, n)))); }

} // namespace

namespace {

__device__ __forceinline__ unsigned long long gtimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One thread per block. Edge = first clock64 after %globaltimer changes value, so (t, c) is
// a pair taken within one loop iteration of the tick; the latency is the same on every SM.
__global__ void clockCalKernel(ClockCal* out, unsigned long long spinNs)
{
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long t = gtimer(), tn, c;
    do { tn = gtimer(); c = clock64(); } while (tn == t);
    const unsigned long long tA = tn, cA = c;
    while (gtimer() < tA + spinNs) {}
    t = gtimer();
    do { tn = gtimer(); c = clock64(); } while (tn == t);
    out[blockIdx.x] = ClockCal{smid, tA, cA, tn, c};
}

} // namespace

void clockCalibration(std::vector<ClockCal>& out, int spinUs)
{
    // 4 blocks per SM of 32 threads: all co-resident, so every SM gets at least one (checked
    // by the analysis, which needs one record per SM per phase).
    const int grid = 4 * deviceSMCount();
    ClockCal* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, grid * sizeof(ClockCal)));
    clockCalKernel<<<grid, 32>>>(d, (unsigned long long)spinUs * 1000ull);
    CUDA_CHECK(cudaGetLastError());
    out.resize(grid);
    CUDA_CHECK(cudaMemcpy(out.data(), d, grid * sizeof(ClockCal), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
}

void forceScale(const DeviceState& s, const Params& p, std::vector<float>& scale)
{
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, s.n * sizeof(float)));
    const int chunk = referenceChunk(s.n);
    for (int i0 = 0; i0 < s.n; i0 += chunk) {
        const int i1 = std::min(s.n, i0 + chunk);
        forceScaleKernel<<<(i1 - i0 + 255) / 256, 256>>>(s.pos, d, s.n, i0, i1, p.softening * p.softening, p.G);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    scale.resize(s.n);
    CUDA_CHECK(cudaMemcpy(scale.data(), d, s.n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
}

void referenceAcc(const DeviceState& s, const Params& p, std::vector<float4>& acc)
{
    float4* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, size_t(s.n) * sizeof(float4)));
    const int chunk = referenceChunk(s.n);
    for (int i0 = 0; i0 < s.n; i0 += chunk) {
        const int i1 = std::min(s.n, i0 + chunk);
        referenceChunkKernel<<<(i1 - i0 + 255) / 256, 256>>>(s.pos, d, s.n, i0, i1, p.softening * p.softening, p.G);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    acc.resize(s.n);
    CUDA_CHECK(cudaMemcpy(acc.data(), d, size_t(s.n) * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
}

void step(DeviceState& s, const Params& p, const KernelInfo& k, cudaStream_t stream)
{
    const int block = 256;
    const int grid  = (s.n + block - 1) / block;
    kickDriftKernel<<<grid, block, 0, stream>>>(s.pos, s.vel, s.acc, s.n, 0.5f * p.dt, p.dt, true);
    k.launch(s, p, stream);
    kickDriftKernel<<<grid, block, 0, stream>>>(s.pos, s.vel, s.acc, s.n, 0.5f * p.dt, p.dt, false);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace nbody
