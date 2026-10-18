// R3 #8 (docs/03-design.md §23.4): CPU+GPU body split for all-pairs (co-execution of one kernel over
// a split index range [P:nozal2025coexecution]; profile-then-split with equal finish times as the
// objective [P:zhong2025hybrimoe]).
//
//  hyb_allpairs  the GPU computes bodies [0, ng) with allPairsRangeKernel (allpairs_tuned's block-full
//                loop with an i offset), the CPU bodies [ng, N) with host::allPairs (AVX2, pinned pool),
//                both over all N sources. Per launch:
//                  copy stream: D2H all positions (16 B/body; the CPU needs every source)
//                  main stream: GPU range kernel (overlaps the copy: one copy engine, but copy || kernel)
//                  host:        wait for the positions, compute [ng, N), H2D 16 B/body of results
//                ng = round(f N) to a multiple of 128. f: Params::hybFrac in [0, 1] (static), -1 dynamic
//                (every launch re-balanced from the previous launch's measured rates: equal finish times
//                with the transfer as a fixed CPU-side offset), -2 dynamic for 4 launches, then frozen.
//                When the model gives the CPU less than ~2 blocks per worker, the CPU part is skipped (f = 1).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "nbody.h"
#include "interaction.cuh"
#include "internal.h"
#include "host.h"

namespace nbody {
namespace {

// allPairsTunedKernel<BLOCK, 0 (full unroll), false> with bodies [i0, i1) instead of [0, n). The source
// loop and the summation order are identical (per body, j = 0..n-1).
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
allPairsRangeKernel(const float4* __restrict__ pos, float4* __restrict__ acc, int n, int i0, int i1, float eps2, float G)
{
    __shared__ float4 tile[BLOCK];
    const int i  = i0 + blockIdx.x * BLOCK + threadIdx.x;
    float4    bi = (i < i1) ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3    a  = make_float3(0.f, 0.f, 0.f);
    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        __syncthreads();
#pragma unroll BLOCK
        for (int k = 0; k < BLOCK; ++k)
            a = bodyBodyInteraction(bi, tile[k], a, eps2);
        __syncthreads();
    }
    if (i < i1) acc[i] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
}

struct HybAP {
    cudaStream_t cs = nullptr;
    cudaEvent_t  evPos = nullptr, evPosDone = nullptr, evG0 = nullptr, evG1 = nullptr, evUp = nullptr;
    host::Pinned<float4> pos, acc;
    int    n = -1, launches = 0;
    double frac = 0.95, rg = -1, rc = -1;   // GPU / CPU rate (bodies per ms)
} ha;

void initHA()
{
    if (ha.cs) return;
    CUDA_CHECK(cudaStreamCreateWithFlags(&ha.cs, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ha.evPos, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ha.evPosDone, cudaEventBlockingSync | cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ha.evG0, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&ha.evG1, cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&ha.evUp, cudaEventDisableTiming));
}

void launchRange(const DeviceState& s, const Params& p, cudaStream_t st, int i0, int i1)
{
    const int cnt = i1 - i0;
    if (cnt <= 0) return;
    const float eps2 = p.softening * p.softening;
    if (p.blockSize == 256)
        allPairsRangeKernel<256><<<ceilDiv(cnt, 256), 256, 0, st>>>(s.pos, s.acc, s.n, i0, i1, eps2, p.G);
    else
        allPairsRangeKernel<128><<<ceilDiv(cnt, 128), 128, 0, st>>>(s.pos, s.acc, s.n, i0, i1, eps2, p.G);
    CUDA_CHECK(cudaGetLastError());
}

void launchHybAllPairs(const DeviceState& s, const Params& p, cudaStream_t st)
{
    initHA();
    const int n = s.n;
    if (n != ha.n) { ha.n = n; ha.launches = 0; ha.frac = 0.95; ha.rg = ha.rc = -1; }
    const bool dynamic = p.hybFrac < 0.f && !(p.hybFrac <= -1.5f && ha.launches >= 4);
    double f = p.hybFrac >= 0.f ? std::min(1.0, double(p.hybFrac)) : ha.frac;
    const int W = cpuSetSize(p.cpuSet, p.cpuThreads);
    int ng = f >= 1.0 ? n : std::min(n, int(std::floor(f * n / 128.0)) * 128);
    const int minCpu = 2 * 16 * W;   // two 16-body register blocks per worker
    if (p.hybFrac < 0.f) {
        if (ha.rc < 0) ng = std::max(0, std::min(ng, (n - minCpu) / 128 * 128));   // CPU rate unknown: probe it once
        else if (n - ng < minCpu) ng = n;                                           // model: CPU share too small to pay
    }
    const int nc = n - ng;
    const double t0 = host::nowMs();
    float4* hp = nullptr;
    if (nc > 0) {
        hp = ha.pos.ensure(size_t(n));
        CUDA_CHECK(cudaEventRecord(ha.evPos, st));
        CUDA_CHECK(cudaStreamWaitEvent(ha.cs, ha.evPos, 0));
        CUDA_CHECK(cudaMemcpyAsync(hp, s.pos, size_t(n) * sizeof(float4), cudaMemcpyDeviceToHost, ha.cs));
        CUDA_CHECK(cudaEventRecord(ha.evPosDone, ha.cs));
    }
    if (ng > 0) {
        CUDA_CHECK(cudaEventRecord(ha.evG0, st));
        launchRange(s, p, st, 0, ng);
        CUDA_CHECK(cudaEventRecord(ha.evG1, st));
    }
    double tx = 0, tc = 0;
    if (nc > 0) {
        CUDA_CHECK(cudaEventSynchronize(ha.evPosDone));
        tx = host::nowMs() - t0;
        float4* hacc = ha.acc.ensure(size_t(nc));
        const double c0 = host::nowMs();
        host::allPairs(hp, n, ng, n, p, hacc);
        tc = host::nowMs() - c0;
        CUDA_CHECK(cudaMemcpyAsync(s.acc + ng, hacc, size_t(nc) * sizeof(float4), cudaMemcpyHostToDevice, ha.cs));
        CUDA_CHECK(cudaEventRecord(ha.evUp, ha.cs));
        CUDA_CHECK(cudaStreamWaitEvent(st, ha.evUp, 0));
    }
    float gms = 0.f;
    const bool needSync = dynamic || p.hybStats;
    if (needSync) {
        if (ng > 0) { CUDA_CHECK(cudaEventSynchronize(ha.evG1)); CUDA_CHECK(cudaEventElapsedTime(&gms, ha.evG0, ha.evG1)); }
        CUDA_CHECK(cudaStreamSynchronize(st));
    }
    if (dynamic) {
        if (ng > 0 && gms > 0.f) ha.rg = ng / double(gms);
        if (nc > 0 && tc > 0.0) ha.rc = nc / tc;
        if (ha.rg > 0 && ha.rc > 0) {
            // equal finish times: ng / rg = tx + (N - ng) / rc
            const double fs = (tx + n / ha.rc) / (n / ha.rg + n / ha.rc);
            ha.frac = std::clamp(ha.launches == 0 ? fs : 0.5 * ha.frac + 0.5 * fs, 0.0, 1.0);
        }
    }
    ++ha.launches;
    if (HybridStats* hs = p.hybStats) {
        hs->launches++;
        hs->fracSum += double(ng) / n;
        hs->lastFrac = double(ng) / n;
        hs->gpuMs += gms; hs->cpuMs += tc; hs->xferMs += tx;
        hs->wallMs += host::nowMs() - t0;
        if (nc == 0) hs->gpuOnlyLaunches++;
    }
}

LaunchInfo queryHybAllPairs(int n, const Params& p)
{
    const void* fn = p.blockSize == 256 ? reinterpret_cast<const void*>(allPairsRangeKernel<256>)
                                        : reinterpret_cast<const void*>(allPairsRangeKernel<128>);
    const int b = p.blockSize == 256 ? 256 : 128;
    LaunchInfo li = describeKernel(fn, ceilDiv(n, b), b, 0);
    li.unroll     = 0;
    li.hostTimed  = true;
    li.hybrid     = true;
    li.cpuWorkers = cpuSetSize(p.cpuSet, p.cpuThreads);
    return li;
}

} // namespace

void registerHybridAllPairs(std::vector<KernelInfo>& r)
{
    r.push_back({"hyb_allpairs",
                 "CPU+GPU body split, all-pairs: GPU bodies [0, fN) (tuned block-full loop), CPU the rest (AVX2); --hyb-frac F | -1 dynamic | -2 calibrate+freeze (R3 #8)",
                 launchHybAllPairs, queryHybAllPairs});
}

} // namespace nbody
