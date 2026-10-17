// State management, integrator, kernel registry.
#include <cstdio>
#include <cstdlib>
#include "nbody.h"
#include "internal.h"

namespace nbody {

void registerAllPairs(std::vector<KernelInfo>& r);
void registerTuned(std::vector<KernelInfo>& r);
void registerMulti(std::vector<KernelInfo>& r);

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

} // namespace

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
