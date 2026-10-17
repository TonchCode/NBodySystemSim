// R3 #3: multiple threads per body (j-loop split + warp-shuffle reduction).
//
//  allpairs_multi  2D block of P bodies x Q threads (blockDim = (Q, P), BLOCK = P*Q).
//                  Thread (s, b) handles body i = blockIdx.x*P + b and the j-slice
//                  j = s (mod Q). The block still loads one tile of BLOCK bodies per
//                  iteration (one float4 per thread), so the tile is consumed by P bodies,
//                  each thread doing BLOCK/Q interactions per tile. After the j-loop the Q
//                  partial sums of a body are combined with a butterfly of __shfl_xor_sync.
//
// Why: with one thread per body, N = 4096 at block 256 is 16 blocks for 80 SMs (smoke test:
// 224 G interactions/s, 20% of SMs busy). Q threads per body multiply the block count by Q,
// so the grid can cover all 80 SMs at small N [P:nyland2007nbody, small-N section].
//
// Layout choice (docs/03-design.md §4.2): threadIdx.x = slice is the fastest-varying index,
// so the Q threads of a body are consecutive lanes of ONE warp (Q <= 32). That makes the
// reduction a register-only shuffle butterfly (log2 Q steps, no shared memory, no extra
// __syncthreads). The price: a warp reads Q distinct tile entries per step, i.e.
// max(1, Q/8) shared-memory wavefronts per LDS.128 instead of 1 broadcast.
//
// Summation order: per thread, j = s, s+Q, s+2Q, ... ; then the butterfly. Different from
// allpairs_naive (so not bitwise equal) but deterministic run to run.
// One more difference from allpairs_tiled: the next tile's global load is issued before the
// current tile is consumed (register prefetch), so its L2 latency overlaps compute. With
// few interactions per tile (BLOCK/Q = 8 at Q = 32) an exposed L2 round trip per tile
// would otherwise dominate.
// Round 2 (critic review 1, #4): the prefetch is a template switch PF (Params::prefetch,
// bench --prefetch 0|1), so "q" and "prefetch" can be measured as separate effects.
// PF = false is the plain tiled loop (load, barrier, consume, barrier). PF = true, q = 1 is
// therefore "tiled + prefetch", the best 1-thread/body candidate with prefetch.
// TL = instrumented per-block timeline (timeline.cuh); the stamp parameter is last.
#include <cstdio>
#include <cstdlib>
#include "nbody.h"
#include "interaction.cuh"
#include "internal.h"
#include "timeline.cuh"

namespace nbody {
namespace {

template <int BLOCK, int Q, bool PF, bool TL>
__global__ void __launch_bounds__(BLOCK)
allPairsMultiKernel(const float4* __restrict__ pos, float4* __restrict__ acc, int n, float eps2, float G,
                    BlockStamp* tl)
{
    static_assert(Q >= 1 && Q <= 32 && (Q & (Q - 1)) == 0, "Q must be a power of two <= 32");
    static_assert(BLOCK % Q == 0, "BLOCK must be a multiple of Q");
    constexpr int P = BLOCK / Q;   // bodies per block
    constexpr int K = BLOCK / Q;   // interactions per thread per tile
    __shared__ float4 tile[BLOCK];
    stampBlockStart<TL>(tl);

    const int    s    = threadIdx.x;              // slice 0..Q-1 (consecutive lanes)
    const int    b    = threadIdx.y;              // local body 0..P-1
    const int    tid  = b * Q + s;                // linear thread id = warp lane order
    const int    i    = blockIdx.x * P + b;
    const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    const float4 bi   = (i < n) ? pos[i] : zero;
    float3       a    = make_float3(0.f, 0.f, 0.f);

    if constexpr (PF) {
        float4 next = (tid < n) ? pos[tid] : zero;    // tile 0
        for (int base = 0; base < n; base += BLOCK) {
            tile[tid] = next;                         // mass-0 padding past n adds exactly +0
            __syncthreads();
            const int jn = base + BLOCK + tid;
            next = (jn < n) ? pos[jn] : zero;         // prefetch tile t+1 while consuming tile t
#pragma unroll 8
            for (int m = 0; m < K; ++m)
                a = bodyBodyInteraction(bi, tile[m * Q + s], a, eps2);
            __syncthreads();
        }
    } else {
        for (int base = 0; base < n; base += BLOCK) {
            const int j = base + tid;
            tile[tid] = (j < n) ? pos[j] : zero;      // load tile t, then wait for it
            __syncthreads();
#pragma unroll 8
            for (int m = 0; m < K; ++m)
                a = bodyBodyInteraction(bi, tile[m * Q + s], a, eps2);
            __syncthreads();
        }
    }

    // Butterfly over the Q lanes of this body. All 32 lanes of every warp reach this point
    // (no early exit above), so the full mask is valid.
#pragma unroll
    for (int off = Q / 2; off > 0; off >>= 1) {
        a.x += __shfl_xor_sync(0xffffffffu, a.x, off);
        a.y += __shfl_xor_sync(0xffffffffu, a.y, off);
        a.z += __shfl_xor_sync(0xffffffffu, a.z, off);
    }
    if (s == 0 && i < n) acc[i] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
    stampBlockEnd<TL>(tl);
}

template <int BLOCK, int Q, bool PF, bool TL>
void launchT(const DeviceState& s, const Params& p, cudaStream_t st)
{
    constexpr int P = BLOCK / Q;
    allPairsMultiKernel<BLOCK, Q, PF, TL><<<ceilDiv(s.n, P), dim3(Q, P), 0, st>>>(
        s.pos, s.acc, s.n, p.softening * p.softening, p.G, p.timeline);
}

struct Variant {
    ForceLauncher launch;
    const void*   kernel;
};

template <int BLOCK, int Q, bool PF, bool TL>
Variant variantT()
{
    return {&launchT<BLOCK, Q, PF, TL>, reinterpret_cast<const void*>(&allPairsMultiKernel<BLOCK, Q, PF, TL>)};
}

template <int BLOCK, int Q>
Variant variant(bool pf, bool tl)
{
    if (pf) return tl ? variantT<BLOCK, Q, true, true>() : variantT<BLOCK, Q, true, false>();
    return tl ? variantT<BLOCK, Q, false, true>() : variantT<BLOCK, Q, false, false>();
}

template <int BLOCK>
Variant byQ(int q, bool pf, bool tl)
{
    switch (q) {
    case 1:  return variant<BLOCK, 1>(pf, tl);
    case 2:  return variant<BLOCK, 2>(pf, tl);
    case 4:  return variant<BLOCK, 4>(pf, tl);
    case 8:  return variant<BLOCK, 8>(pf, tl);
    case 16: return variant<BLOCK, 16>(pf, tl);
    case 32: return variant<BLOCK, 32>(pf, tl);
    default: break;
    }
    fprintf(stderr, "allpairs_multi: q %d not instantiated (use 0 = auto, 1, 2, 4, 8, 16, 32)\n", q);
    exit(2);
}

Variant select(int block, int q, const Params& p)
{
    if (p.prefetch != 0 && p.prefetch != 1) {
        fprintf(stderr, "allpairs_multi: prefetch must be 0 or 1 (got %d)\n", p.prefetch);
        exit(2);
    }
    const bool pf = p.prefetch == 1, tl = p.timeline != nullptr;
    switch (block) {
    case 64:   return byQ<64>(q, pf, tl);
    case 128:  return byQ<128>(q, pf, tl);
    case 256:  return byQ<256>(q, pf, tl);
    case 512:  return byQ<512>(q, pf, tl);
    case 1024: return byQ<1024>(q, pf, tl);
    default: break;
    }
    fprintf(stderr, "allpairs_multi: block %d not instantiated (use 64, 128, 256, 512, 1024)\n", block);
    exit(2);
}

int resolveQ(int n, const Params& p)
{
    return p.threadsPerBody > 0 ? p.threadsPerBody : autoThreadsPerBody(n, p.blockSize);
}

void launchMulti(const DeviceState& s, const Params& p, cudaStream_t st)
{
    select(p.blockSize, resolveQ(s.n, p), p).launch(s, p, st);
    CUDA_CHECK(cudaGetLastError());
}

LaunchInfo queryMulti(int n, const Params& p)
{
    const int  q  = resolveQ(n, p);
    LaunchInfo li = describeKernel(select(p.blockSize, q, p).kernel, ceilDiv(n, p.blockSize / q),
                                   p.blockSize, 0);
    li.threadsPerBody = q;
    li.unroll         = 8;
    li.prefetch       = p.prefetch;
    li.instrumented   = p.timeline != nullptr;
    return li;
}

} // namespace

// Cost model (docs/03-design.md §4.4, outcome in §6). Assumption A1, from the smoke test:
// one resident block of >= 8 warps keeps an SM's rate near its saturated value, so an SM's
// time is proportional to the interactions it owns. For each candidate q:
//   B        = ceil(n / P), P = block / q            blocks in the grid
//   perSM    = ceil(B / SMs)                          blocks owned by the busiest SM
//   work     = perSM * P                              bodies x N interactions on that SM
//   overhead = 1 + kTileCost * q / block              per-tile sync + load + store, in
//                                                     interaction units, spread over the
//                                                     block/q interactions of a tile
//   latency  = max(1, 8 / residentWarps)              A1 breaks below 8 warps per SM
// and take the q with the lowest work * overhead * latency. Ties within 2% go to the
// smaller q (fewer shuffles, more reuse per tile, less shared-memory bandwidth).
// Sanity result (results/2026-10-17_architect_sanity): mean loss vs the best measured q
// 1.8% in-sample (N = 1000..65536), 0.9% out-of-sample (N = 2048..32768), worst 12%.
// A variant with a measured r(warps) curve did better in-sample but worse out-of-sample,
// so it was not kept. Its code (reconstructed in round 2; the round-1 original was never
// saved) is in tools/autoq_models.py, with the loss computation (critic review 1, #5).
int autoThreadsPerBody(int n, int blockSize)
{
    constexpr double kTileCost = 2.0;
    const int sms = deviceSMCount();
    int    bestQ    = 1;
    double bestCost = 1e300;
    for (int q = 1; q <= 32 && q <= blockSize; q *= 2) {
        const int    P      = blockSize / q;
        const int    B      = ceilDiv(n, P);
        const int    perSM  = ceilDiv(B, sms);
        const double warps  = double(perSM) * blockSize / 32.0;   // resident on the busiest SM
        const double lat    = warps >= 8.0 ? 1.0 : 8.0 / warps;
        const double cost   = double(perSM) * P * (1.0 + kTileCost * q / blockSize) * lat;
        if (cost < bestCost * 0.98) { bestCost = cost; bestQ = q; }
    }
    return bestQ;
}

void registerMulti(std::vector<KernelInfo>& r)
{
    r.push_back({"allpairs_multi",
                 "all-pairs tiled, q threads/body + shuffle reduction (--q, 0 = auto; --prefetch 0|1) (R3 #3)",
                 launchMulti, queryMulti});
}

} // namespace nbody
