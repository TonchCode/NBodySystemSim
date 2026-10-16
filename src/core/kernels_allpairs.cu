// All-pairs O(N^2) force kernels.
//
//  allpairs_naive  every thread streams all N bodies from global memory (via L1/L2).
//                  The correctness reference: no cleverness to get wrong.
//  allpairs_tiled  the classic shared-memory tiling (GPU Gems 3 ch. 31): each block
//                  cooperatively loads a tile of blockDim bodies into shared memory, then
//                  every thread consumes the tile. Statically balanced: every thread does
//                  exactly N interactions. This is R3 technique #1, the baseline.
//
// Both are FP32 by design: FP64 is 1/64 rate on sm_89 (docs/01-hardware.md).
#include <cstdio>
#include <cstdlib>
#include "nbody.h"

namespace nbody {
namespace {

__device__ __forceinline__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai, float eps2)
{
    float3 r  = make_float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z);
    float d2  = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, eps2)));
    float inv = rsqrtf(d2);
    float s   = bj.w * inv * inv * inv;     // m_j / (|r|^2 + eps^2)^(3/2)
    ai.x = fmaf(r.x, s, ai.x);
    ai.y = fmaf(r.y, s, ai.y);
    ai.z = fmaf(r.z, s, ai.z);
    return ai;
}

__global__ void allPairsNaiveKernel(const float4* __restrict__ pos, float4* __restrict__ acc,
                                    int n, float eps2, float G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 bi = pos[i];
    float3 a  = make_float3(0.f, 0.f, 0.f);
    for (int j = 0; j < n; ++j)
        a = bodyBodyInteraction(bi, pos[j], a, eps2);   // self-term is 0 thanks to softening
    acc[i] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
}

__global__ void allPairsTiledKernel(const float4* __restrict__ pos, float4* __restrict__ acc,
                                    int n, float eps2, float G)
{
    extern __shared__ float4 tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Out-of-range threads still help load tiles; they just don't write a result.
    float4 bi = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3 a  = make_float3(0.f, 0.f, 0.f);

    for (int base = 0; base < n; base += blockDim.x) {
        int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);  // mass 0 = no force
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < blockDim.x; ++k)
            a = bodyBodyInteraction(bi, tile[k], a, eps2);
        __syncthreads();
    }
    if (i < n) acc[i] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
}

int gridFor(int n, int block) { return (n + block - 1) / block; }

void launchNaive(const DeviceState& s, const Params& p, cudaStream_t st)
{
    allPairsNaiveKernel<<<gridFor(s.n, p.blockSize), p.blockSize, 0, st>>>(
        s.pos, s.acc, s.n, p.softening * p.softening, p.G);
    CUDA_CHECK(cudaGetLastError());
}

void launchTiled(const DeviceState& s, const Params& p, cudaStream_t st)
{
    size_t smem = p.blockSize * sizeof(float4);
    allPairsTiledKernel<<<gridFor(s.n, p.blockSize), p.blockSize, smem, st>>>(
        s.pos, s.acc, s.n, p.softening * p.softening, p.G);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// Registry hook used by nbody.cu. New variants: add a launcher here (or in a new
// kernels_*.cu) and register it in kernelRegistry().
void registerAllPairs(std::vector<KernelInfo>& r)
{
    r.push_back({"allpairs_naive", "all-pairs, global memory only (reference)", launchNaive});
    r.push_back({"allpairs_tiled", "all-pairs, shared-memory tiling, 1 thread/body (R3 #1)", launchTiled});
}

} // namespace nbody
