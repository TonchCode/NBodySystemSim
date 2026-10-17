// Per-block timeline stamps for the instrumented kernel instantiations (TL = true).
//
// Why: GPU performance counters are blocked on this machine (ERR_NVGPUCTRPERM), so ncu's
// sm__cycles_active cannot be read. Each block records where it ran (%smid) and when
// (%globaltimer, plus clock64 on its SM). tools/timeline_analysis.py turns that into
// per-SM busy time, max/mean SM busy, idle SMs and the tail.
//
// Cost when TL = true: thread 0 does two 16-byte-ish stores at entry, and at exit one extra
// __syncthreads (so t1 is taken after the block's LAST warp is done) plus two stores. No
// value is kept live across the force loop: the entry stamp goes straight to memory, so the
// register count of the hot loop should not change (checked with --resource-usage).
// When TL = false every function below compiles to nothing: the production kernels' SASS is
// unchanged (checked by diffing cuobjdump -sass before/after, results/2026-10-17_round2).
#pragma once
#include <cuda_runtime.h>
#include "nbody.h"

namespace nbody {

__device__ __forceinline__ unsigned long long globalTimerNs()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ unsigned smId()
{
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}

__device__ __forceinline__ bool firstThreadOfBlock()
{
    return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

template <bool TL>
__device__ __forceinline__ void stampBlockStart(BlockStamp* tl)
{
    if constexpr (TL) {
        if (firstThreadOfBlock()) {
            BlockStamp* r = tl + blockIdx.x;
            r->t0    = globalTimerNs();
            r->c0    = clock64();
            r->smid  = smId();
            r->block = blockIdx.x;
        }
    }
}

// Must be reached by every thread of the block (it contains a barrier).
template <bool TL>
__device__ __forceinline__ void stampBlockEnd(BlockStamp* tl)
{
    if constexpr (TL) {
        __syncthreads();
        if (firstThreadOfBlock()) {
            BlockStamp* r = tl + blockIdx.x;
            r->c1 = clock64();
            r->t1 = globalTimerNs();
        }
    }
}

} // namespace nbody
