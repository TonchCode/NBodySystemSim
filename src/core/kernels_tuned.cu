// R3 #2: block-size / tile / unroll sweep (load balance inside an SM, and across SMs).
//
//  allpairs_tuned  the same algorithm as allpairs_tiled (tile = block, one body per
//                  thread, identical summation order => bitwise-equal results), but block
//                  size and inner-loop unroll are template parameters, so:
//                    * the tile loop has a compile-time trip count and the unroll is real;
//                    * __launch_bounds__(BLOCK) lets ptxas budget registers per block size;
//                    * shared memory is static (BLOCK * 16 B).
//                  Selected at run time from Params::blockSize x Params::unroll:
//                    blockSize in {64, 128, 256, 512, 1024}, unroll in {1, 4, 8, 16, 0=full}.
//
// What the sweep measures (docs/03-design.md §3): occupancy per configuration on sm_89
// (registers vs the 64K file, 24 blocks / 48 warps per SM) and the tail effect
// ceil(B/80) / (B/80) when B = N/blockSize blocks are spread over the 80 SMs.
#include <cstdio>
#include <cstdlib>
#include "nbody.h"
#include "interaction.cuh"
#include "internal.h"

namespace nbody {
namespace {

template <int BLOCK, int UNROLL>   // UNROLL == 0 means "unroll the whole tile"
__global__ void __launch_bounds__(BLOCK)
allPairsTunedKernel(const float4* __restrict__ pos, float4* __restrict__ acc, int n, float eps2, float G)
{
    constexpr int U = (UNROLL == 0) ? BLOCK : UNROLL;
    static_assert(BLOCK % U == 0, "unroll must divide the tile");
    __shared__ float4 tile[BLOCK];

    const int i  = blockIdx.x * BLOCK + threadIdx.x;
    float4    bi = (i < n) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3    a  = make_float3(0.f, 0.f, 0.f);

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);  // mass 0 = no force
        __syncthreads();
#pragma unroll U
        for (int k = 0; k < BLOCK; ++k)
            a = bodyBodyInteraction(bi, tile[k], a, eps2);
        __syncthreads();
    }
    if (i < n) acc[i] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
}

template <int BLOCK, int UNROLL>
void launchT(const DeviceState& s, const Params& p, cudaStream_t st)
{
    allPairsTunedKernel<BLOCK, UNROLL><<<ceilDiv(s.n, BLOCK), BLOCK, 0, st>>>(
        s.pos, s.acc, s.n, p.softening * p.softening, p.G);
}

struct Variant {
    ForceLauncher launch;
    const void*   kernel;
};

template <int BLOCK, int UNROLL>
Variant variant()
{
    return {&launchT<BLOCK, UNROLL>, reinterpret_cast<const void*>(&allPairsTunedKernel<BLOCK, UNROLL>)};
}

template <int BLOCK>
Variant byUnroll(int unroll)
{
    switch (unroll) {
    case 1:  return variant<BLOCK, 1>();
    case 4:  return variant<BLOCK, 4>();
    case 8:  return variant<BLOCK, 8>();
    case 16: return variant<BLOCK, 16>();
    case 0:  return variant<BLOCK, 0>();
    default: break;
    }
    fprintf(stderr, "allpairs_tuned: unroll %d not instantiated (use 1, 4, 8, 16 or 0 = full)\n", unroll);
    exit(2);
}

Variant select(const Params& p)
{
    switch (p.blockSize) {
    case 64:   return byUnroll<64>(p.unroll);
    case 128:  return byUnroll<128>(p.unroll);
    case 256:  return byUnroll<256>(p.unroll);
    case 512:  return byUnroll<512>(p.unroll);
    case 1024: return byUnroll<1024>(p.unroll);
    default: break;
    }
    fprintf(stderr, "allpairs_tuned: block %d not instantiated (use 64, 128, 256, 512, 1024)\n", p.blockSize);
    exit(2);
}

void launchTuned(const DeviceState& s, const Params& p, cudaStream_t st)
{
    select(p).launch(s, p, st);
    CUDA_CHECK(cudaGetLastError());
}

LaunchInfo queryTuned(int n, const Params& p)
{
    LaunchInfo li = describeKernel(select(p).kernel, ceilDiv(n, p.blockSize), p.blockSize, 0);
    li.unroll = p.unroll;
    return li;
}

} // namespace

void registerTuned(std::vector<KernelInfo>& r)
{
    r.push_back({"allpairs_tuned",
                 "all-pairs tiled, templated block x unroll (--block, --unroll) (R3 #2)",
                 launchTuned, queryTuned});
}

} // namespace nbody
