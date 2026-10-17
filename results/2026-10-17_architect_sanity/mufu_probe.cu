// MUFU-vs-issue probe (cuda-architect, 2026-10-17). NOT part of the build; an experiment
// artifact kept next to its results. Replaces an ncu run that was blocked by
// ERR_NVGPUCTRPERM.
//
// Same tiled kernel (block 128, full unroll, N = 65536) in four variants of the interaction:
//   base      rsqrtf (MUFU.RSQ + non-FTZ guard: FSETP + 2 FMUL)           ~17.6 issue slots
//   ftz       rsqrt.approx.ftz.f32 (MUFU.RSQ only, no guard)              ~14.6 slots, same MUFU
//   mufu2     base + one extra MUFU.RSQ per interaction (s uses it)       +1 slot, +1 MUFU (2x)
//   fmul3     base + 3 extra dependent FMULs (value-neutral)              +3 slots, same MUFU
// If MUFU-bound: mufu2 ~ 2x slower, ftz and fmul3 ~ no change.
// If issue-bound: mufu2 ~ +6%, ftz ~ -17%, fmul3 ~ +17%.
//
// Build: nvcc -O3 -arch=sm_89 -o mufu_probe.exe mufu_probe.cu   (from a vcvars64 shell)
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

template <int V>
__device__ __forceinline__ float3 inter(float4 bi, float4 bj, float3 ai, float eps2)
{
    float3 r  = make_float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z);
    float d2  = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, eps2)));
    float inv, s;
    if (V == 1) {
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(d2));
        s = bj.w * inv * inv * inv;
    } else if (V == 2) {
        inv = rsqrtf(d2);
        float inv2;
        asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv2) : "f"(d2));
        s = bj.w * inv * inv * inv2;
    } else if (V == 3) {
        inv = rsqrtf(d2);
        float one;
        asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(one) : "f"(inv), "f"(0.f));   // 0
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(one) : "f"(one), "f"(inv), "f"(1.f)); // 1
        asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(inv) : "f"(inv), "f"(one));   // inv
        s = bj.w * inv * inv * inv;
    } else {
        inv = rsqrtf(d2);
        s = bj.w * inv * inv * inv;
    }
    ai.x = fmaf(r.x, s, ai.x);
    ai.y = fmaf(r.y, s, ai.y);
    ai.z = fmaf(r.z, s, ai.z);
    return ai;
}

template <int V>
__global__ void __launch_bounds__(128) k(const float4* __restrict__ pos, float4* acc, int n, float eps2)
{
    __shared__ float4 tile[128];
    int i = blockIdx.x * 128 + threadIdx.x;
    float4 bi = pos[i];
    float3 a = make_float3(0, 0, 0);
    for (int base = 0; base < n; base += 128) {
        tile[threadIdx.x] = pos[base + threadIdx.x];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 128; ++j) a = inter<V>(bi, tile[j], a, eps2);
        __syncthreads();
    }
    acc[i] = make_float4(a.x, a.y, a.z, 0);
}

template <int V>
float timeIt(const float4* p, float4* a, int n)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto w0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 0.3) {
        k<V><<<n / 128, 128>>>(p, a, n, 1e-4f);
        cudaDeviceSynchronize();
    }
    std::vector<float> ms;
    for (int r = 0; r < 15; ++r) {
        cudaEventRecord(e0);
        k<V><<<n / 128, 128>>>(p, a, n, 1e-4f);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int main()
{
    const int n = 65536;
    std::vector<float4> h(n);
    unsigned s = 1;
    for (auto& v : h) {
        auto rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.f / 16777216.f) - 0.5f; };
        v = make_float4(rnd(), rnd(), rnd(), 1.f / n);
    }
    float4 *p, *a;
    cudaMalloc(&p, n * 16); cudaMalloc(&a, n * 16);
    cudaMemcpy(p, h.data(), n * 16, cudaMemcpyHostToDevice);
    const char* names[] = {"base", "ftz", "mufu2", "fmul3"};
    float t[4];
    for (int rep = 0; rep < 2; ++rep) {   // two passes, interleaved, to expose clock drift
        t[0] = timeIt<0>(p, a, n); t[1] = timeIt<1>(p, a, n);
        t[2] = timeIt<2>(p, a, n); t[3] = timeIt<3>(p, a, n);
        for (int v = 0; v < 4; ++v)
            printf("pass %d  %-6s median %.3f ms  %.0f G int/s  ratio vs base %.3f\n", rep, names[v], t[v],
                   double(n) * n / (t[v] * 1e-3) * 1e-9, t[v] / t[0]);
    }
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
