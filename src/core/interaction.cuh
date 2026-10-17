// The one pairwise interaction every force kernel uses. Shared so that all variants
// compute bitwise-identical terms; only the *order* of summation may differ between them.
#pragma once
#include <cuda_runtime.h>

namespace nbody {

// a_i += m_j * r_ij / (|r_ij|^2 + eps^2)^(3/2). FP32: 3 FADD, 3 FFMA (d2), MUFU.RSQ (+ the
// non-FTZ denormal guard: 1 FSETP + 2 predicated FMUL), 3 FMUL, 3 FFMA.
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

} // namespace nbody
