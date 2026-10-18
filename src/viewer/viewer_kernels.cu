// Viewer-only kernels (D-032): the mouse gravity well and an on-request energy measurement.
// Not part of nbody_core: no bench code path, production kernel or validation uses this file.
#include <algorithm>
#include <chrono>
#include <vector>
#include "nbody.h"   // CUDA_CHECK
#include "viewer_kernels.h"

namespace viewer {
namespace {

__global__ void wellKickKernel(float4* __restrict__ vel, const float4* __restrict__ pos, int n, float3 w, float gm,
                               float eps2, float dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 p = pos[i];
    const float  rx = w.x - p.x, ry = w.y - p.y, rz = w.z - p.z;
    const float  d2 = fmaf(rx, rx, fmaf(ry, ry, fmaf(rz, rz, eps2)));
    const float  inv = rsqrtf(d2);
    const float  s = gm * inv * inv * inv * dt;
    float4 v = vel[i];
    v.x = fmaf(rx, s, v.x);
    v.y = fmaf(ry, s, v.y);
    v.z = fmaf(rz, s, v.z);
    vel[i] = v;
}

constexpr int kTile = 256;

// One thread per body i in [i0, i1); all j tiled through shared memory.
__global__ void energyKernel(const float4* __restrict__ pos, const float4* __restrict__ vel, int n, int i0, int i1,
                             float eps2, double2* __restrict__ out)
{
    __shared__ float4 sh[kTile];
    const int  i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < i1;
    const float4 bi = active ? pos[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    double phi = 0.0;
    for (int t = 0; t < n; t += kTile) {
        const int j = t + threadIdx.x;
        sh[threadIdx.x] = j < n ? pos[j] : make_float4(0.f, 0.f, 0.f, 0.f);   // m = 0 pads
        __syncthreads();
        float part = 0.f;
        const int m = min(kTile, n - t);
        for (int k = 0; k < m; ++k) {
            const float4 bj = sh[k];
            const float  dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
            const float  d2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
            const float  term = bj.w * rsqrtf(d2);
            part += (t + k == i) ? 0.f : term;
        }
        phi += double(part);
        __syncthreads();
    }
    if (active) {
        const float4 v = vel[i];
        const double ke = 0.5 * double(bi.w) * (double(v.x) * v.x + double(v.y) * v.y + double(v.z) * v.z);
        out[i] = make_double2(ke, -0.5 * double(bi.w) * phi);
    }
}

} // namespace

void wellKick(float4* vel, const float4* pos, int n, float3 w, float massW, float epsW, float G, float dt,
              cudaStream_t st)
{
    if (n <= 0) return;
    wellKickKernel<<<(n + 255) / 256, 256, 0, st>>>(vel, pos, n, w, G * massW, epsW * epsW, dt);
    CUDA_CHECK(cudaGetLastError());
}

EnergyResult energyGPU(const float4* pos, const float4* vel, int n, float softening, float G)
{
    EnergyResult r;
    if (n <= 0) return r;
    const auto t0 = std::chrono::steady_clock::now();
    double2* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, size_t(n) * sizeof(double2)));
    const long long chunk = std::max<long long>(kTile, std::min<long long>(n, (1ll << 33) / std::max(1, n)));
    for (long long a = 0; a < n; a += chunk) {
        const int i0 = int(a), i1 = int(std::min<long long>(n, a + chunk));
        energyKernel<<<(i1 - i0 + kTile - 1) / kTile, kTile>>>(pos, vel, n, i0, i1, softening * softening, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());   // one chunk at a time: each launch stays short (TDR)
    }
    std::vector<double2> h(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(h.data(), d, size_t(n) * sizeof(double2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    for (const double2& e : h) { r.kinetic += e.x; r.potential += e.y; }
    r.potential *= G;
    r.total = r.kinetic + r.potential;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

} // namespace viewer
