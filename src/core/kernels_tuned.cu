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
#include "timeline.cuh"

namespace nbody {
namespace {

// UNROLL == 0 means "unroll the whole tile". TL = instrumented (per-block timeline,
// timeline.cuh); the stamp parameter is last, so TL = false keeps the round-1 SASS.
template <int BLOCK, int UNROLL, bool TL>
__global__ void __launch_bounds__(BLOCK)
allPairsTunedKernel(const float4* __restrict__ pos, float4* __restrict__ acc, int n, float eps2, float G,
                    BlockStamp* tl)
{
    constexpr int U = (UNROLL == 0) ? BLOCK : UNROLL;
    static_assert(BLOCK % U == 0, "unroll must divide the tile");
    __shared__ float4 tile[BLOCK];
    stampBlockStart<TL>(tl);

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
    stampBlockEnd<TL>(tl);
}

template <int BLOCK, int UNROLL, bool TL>
void launchT(const DeviceState& s, const Params& p, cudaStream_t st)
{
    allPairsTunedKernel<BLOCK, UNROLL, TL><<<ceilDiv(s.n, BLOCK), BLOCK, 0, st>>>(
        s.pos, s.acc, s.n, p.softening * p.softening, p.G, p.timeline);
}

struct Variant {
    ForceLauncher launch;
    const void*   kernel;
};

template <int BLOCK, int UNROLL, bool TL>
Variant variantT()
{
    return {&launchT<BLOCK, UNROLL, TL>, reinterpret_cast<const void*>(&allPairsTunedKernel<BLOCK, UNROLL, TL>)};
}

template <int BLOCK, int UNROLL>
Variant variant(bool tl)
{
    return tl ? variantT<BLOCK, UNROLL, true>() : variantT<BLOCK, UNROLL, false>();
}

template <int BLOCK>
Variant byUnroll(int unroll, bool tl)
{
    switch (unroll) {
    case 1:  return variant<BLOCK, 1>(tl);
    case 4:  return variant<BLOCK, 4>(tl);
    case 8:  return variant<BLOCK, 8>(tl);
    case 16: return variant<BLOCK, 16>(tl);
    case 0:  return variant<BLOCK, 0>(tl);
    default: break;
    }
    fprintf(stderr, "allpairs_tuned: unroll %d not instantiated (use 1, 4, 8, 16 or 0 = full)\n", unroll);
    exit(2);
}

Variant select(const Params& p)
{
    const bool tl = p.timeline != nullptr;
    switch (p.blockSize) {
    case 64:   return byUnroll<64>(p.unroll, tl);
    case 128:  return byUnroll<128>(p.unroll, tl);
    case 256:  return byUnroll<256>(p.unroll, tl);
    case 512:  return byUnroll<512>(p.unroll, tl);
    case 1024: return byUnroll<1024>(p.unroll, tl);
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
    li.unroll       = p.unroll;
    li.instrumented = p.timeline != nullptr;
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
