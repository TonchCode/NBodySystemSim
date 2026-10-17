// R3 #4 / #5: Barnes-Hut tree code, fully GPU-resident (docs/03-design.md §10-§13).
//
// Build (identical for every bh_* variant; the tree does not depend on the thread order):
//   1. keys      bounding cube (two-pass block reduction, no host sync) + 63-bit Morton key
//                per body (21 bits per axis)
//   2. sort      cub::DeviceRadixSort::SortPairs(key, index) over bits [0, 63), then gather
//                sortedPos[k] = pos[perm[k]]
//   3. tree      binary radix tree over the sorted keys (one thread per internal node: range
//                + split by binary search on the common-prefix length; duplicate keys are
//                broken by the body index), then a bottom-up pass (one thread per leaf, an
//                atomic arrival counter per node; the second child to arrive builds the
//                parent) computing mass, centre of mass, tight AABB and the opening radius
//   4. order     the thread -> body map. bh_naive: identity (R3 #4, original order);
//                bh_sorted: the Morton permutation from step 2 (free); bh_hilbert: a second
//                key (3D Hilbert index, Skilling's transpose) + a second radix sort.
//   5. traverse  one thread per body, a private stack in local memory, no warp cooperation.
//                bmax opening criterion: a node is accepted as one pseudo-body (its monopole)
//                iff |x_i - com|^2 >= (bmax / theta)^2, bmax = distance from the COM to the
//                farthest corner of the node's tight AABB. Otherwise: if the node holds
//                <= leafSize bodies (a contiguous range of the sorted array) they are summed
//                directly, else both children are pushed.
// Every interaction (cell or body) is bodyBodyInteraction() from interaction.cuh, so a cell
// interaction costs exactly what a body interaction costs, and "interactions" is a uniform
// work unit across all-pairs and tree kernels.
//
// THE ONLY DIFFERENCE between bh_naive and bh_sorted / bh_hilbert is the content of the
// order[] array the traversal kernel reads (same kernel, same SASS, same tree). Each body's
// sum is evaluated in the same order whichever thread runs it, so the three variants give
// BITWISE-identical accelerations (checked by --selftest).
//
// Phase timing (maeda2026gtap: report phases separately): when Params::phaseEvents is set,
// the launcher records an event before phase 1 and after every phase (6 events, 5 phases).
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include "nbody.h"
#include "interaction.cuh"
#include "internal.h"
#include "timeline.cuh"

namespace nbody {

// ---- keys (host + device, so --selftest can check the Hilbert curve on the host) -----------
__host__ __device__ inline unsigned long long expandBits3(unsigned v)
{
    unsigned long long x = v & 0x1fffffu;
    x = (x | x << 32) & 0x001f00000000ffffull;
    x = (x | x << 16) & 0x001f0000ff0000ffull;
    x = (x | x << 8)  & 0x100f00f00f00f00full;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ull;
    x = (x | x << 2)  & 0x1249249249249249ull;
    return x;
}

__host__ __device__ inline unsigned long long mortonKey(unsigned x, unsigned y, unsigned z)
{
    return (expandBits3(x) << 2) | (expandBits3(y) << 1) | expandBits3(z);
}

// 3D Hilbert index of 21-bit coordinates: Skilling's AxestoTranspose (TODO(source): J.
// Skilling, "Programming the Hilbert curve", AIP Conf. Proc. 707, 2004; not in references.bib),
// then the transposed bits are interleaved like a Morton key (X[0] most significant).
__host__ __device__ inline unsigned long long hilbertKey(unsigned x, unsigned y, unsigned z)
{
    unsigned X[3] = {x, y, z};
    const unsigned M = 1u << 20;
    for (unsigned Q = M; Q > 1; Q >>= 1) {          // inverse undo
        const unsigned P = Q - 1;
        for (int i = 0; i < 3; ++i) {
            if (X[i] & Q) X[0] ^= P;
            else { unsigned t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    X[1] ^= X[0];                                    // Gray encode
    X[2] ^= X[1];
    unsigned t = 0;
    for (unsigned Q = M; Q > 1; Q >>= 1)
        if (X[2] & Q) t ^= Q - 1;
    X[0] ^= t; X[1] ^= t; X[2] ^= t;
    return mortonKey(X[0], X[1], X[2]);
}

unsigned long long hostHilbertKey(unsigned x, unsigned y, unsigned z) { return hilbertKey(x, y, z); }

namespace {

constexpr int kStack       = 64;    // pending siblings per thread; overflow is flagged, not silent
constexpr int kBboxBlocks  = 160;   // 2 per SM
constexpr int kBboxThreads = 256;
constexpr int kBuildBlock  = 256;
enum class Order { Identity, Morton, Hilbert, Coherent };

// ---- GPU-resident tree context (one per process; grown on demand) ----------------------
struct TreeCtx {
    int cap = 0;
    unsigned long long *keyIn = nullptr, *keySorted = nullptr, *keyH = nullptr;
    int    *iota = nullptr, *perm = nullptr, *hOrder = nullptr;
    float4 *sortedPos = nullptr;
    float4 *nodeCom = nullptr, *nodeLo = nullptr, *nodeHi = nullptr;
    float  *nodeRc2 = nullptr;
    int4   *nodeInfo = nullptr;   // left, right, first, last (children: >= 0 internal, < 0 = ~sorted body)
    int    *leafParent = nullptr, *nodeParent = nullptr, *arrive = nullptr;
    float4 *bboxPartial = nullptr, *bbox = nullptr;   // [2*kBboxBlocks], [2] (lo, hi)
    int    *overflow = nullptr;
    // round 2
    int    *cohOrder = nullptr;                       // coherent control: perm[t & ~31]
    int    *queue = nullptr;                          // persistent work counter
    unsigned long long *unitCost = nullptr, *unitPre = nullptr;   // static partition
    int    *warpStart = nullptr;
    int     warpStartCap = 0, costN = -1, costG = -1;
    void   *scanTmp = nullptr;
    size_t  scanTmpBytes = 0;
    void   *sortTmp = nullptr;
    size_t  sortTmpBytes = 0;
    int     lastN = 0;
    float   lastTheta = 0.f;
    int     lastLeaf = 0;
    float   lastEps2 = 0.f;
    Order   lastOrder = Order::Identity;
} g;

__global__ void iotaKernel(int* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = i;
}

void ensureCapacity(int n)
{
    if (n <= g.cap) return;
    auto fr = [](auto*& p) { if (p) cudaFree(p); p = nullptr; };
    fr(g.keyIn); fr(g.keySorted); fr(g.keyH); fr(g.iota); fr(g.perm); fr(g.hOrder); fr(g.sortedPos);
    fr(g.nodeCom); fr(g.nodeLo); fr(g.nodeHi); fr(g.nodeRc2); fr(g.nodeInfo); fr(g.leafParent);
    fr(g.nodeParent); fr(g.arrive); fr(g.sortTmp); fr(g.cohOrder); fr(g.unitCost); fr(g.unitPre); fr(g.scanTmp);
    g.costN = -1;
    const size_t N = size_t(n), I = size_t(n > 1 ? n - 1 : 1);
    CUDA_CHECK(cudaMalloc(&g.keyIn, N * 8));
    CUDA_CHECK(cudaMalloc(&g.keySorted, N * 8));
    CUDA_CHECK(cudaMalloc(&g.keyH, N * 8));
    CUDA_CHECK(cudaMalloc(&g.iota, N * 4));
    CUDA_CHECK(cudaMalloc(&g.perm, N * 4));
    CUDA_CHECK(cudaMalloc(&g.hOrder, N * 4));
    CUDA_CHECK(cudaMalloc(&g.sortedPos, N * 16));
    CUDA_CHECK(cudaMalloc(&g.nodeCom, I * 16));
    CUDA_CHECK(cudaMalloc(&g.nodeLo, I * 16));
    CUDA_CHECK(cudaMalloc(&g.nodeHi, I * 16));
    CUDA_CHECK(cudaMalloc(&g.nodeRc2, I * 4));
    CUDA_CHECK(cudaMalloc(&g.nodeInfo, I * 16));
    CUDA_CHECK(cudaMalloc(&g.leafParent, N * 4));
    CUDA_CHECK(cudaMalloc(&g.nodeParent, I * 4));
    CUDA_CHECK(cudaMalloc(&g.arrive, I * 4));
    CUDA_CHECK(cudaMalloc(&g.cohOrder, N * 4));
    const size_t U = (N + 31) / 32;
    CUDA_CHECK(cudaMalloc(&g.unitCost, U * 8));
    CUDA_CHECK(cudaMalloc(&g.unitPre, U * 8));
    g.scanTmpBytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, g.scanTmpBytes, g.unitCost, g.unitPre, int(U)));
    CUDA_CHECK(cudaMalloc(&g.scanTmp, g.scanTmpBytes));
    if (!g.bbox) {
        CUDA_CHECK(cudaMalloc(&g.bboxPartial, 2 * kBboxBlocks * 16));
        CUDA_CHECK(cudaMalloc(&g.bbox, 2 * 16));
        CUDA_CHECK(cudaMalloc(&g.overflow, 4));
        CUDA_CHECK(cudaMemset(g.overflow, 0, 4));
        CUDA_CHECK(cudaMalloc(&g.queue, 4));
    }
    g.sortTmpBytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g.sortTmpBytes, g.keyIn, g.keySorted, g.iota, g.perm,
                                               n, 0, 63));
    CUDA_CHECK(cudaMalloc(&g.sortTmp, g.sortTmpBytes));
    // The identity order, written once. SortPairs never writes its (const) inputs, so iota
    // stays the identity; checkLastTree() verifies that.
    iotaKernel<<<ceilDiv(n, 256), 256>>>(g.iota, n);
    CUDA_CHECK(cudaGetLastError());
    g.cap = n;
}

// ---- 1. bounding box + keys ---------------------------------------------------------------
__global__ void bboxPartialKernel(const float4* __restrict__ pos, int n, float4* __restrict__ part)
{
    __shared__ float lo[3][kBboxThreads], hi[3][kBboxThreads];
    float l0 = FLT_MAX, l1 = FLT_MAX, l2 = FLT_MAX, h0 = -FLT_MAX, h1 = -FLT_MAX, h2 = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float4 p = pos[i];
        l0 = fminf(l0, p.x); l1 = fminf(l1, p.y); l2 = fminf(l2, p.z);
        h0 = fmaxf(h0, p.x); h1 = fmaxf(h1, p.y); h2 = fmaxf(h2, p.z);
    }
    const int t = threadIdx.x;
    lo[0][t] = l0; lo[1][t] = l1; lo[2][t] = l2; hi[0][t] = h0; hi[1][t] = h1; hi[2][t] = h2;
    __syncthreads();
    for (int s = kBboxThreads / 2; s > 0; s >>= 1) {
        if (t < s)
            for (int a = 0; a < 3; ++a) {
                lo[a][t] = fminf(lo[a][t], lo[a][t + s]);
                hi[a][t] = fmaxf(hi[a][t], hi[a][t + s]);
            }
        __syncthreads();
    }
    if (t == 0) {
        part[2 * blockIdx.x]     = make_float4(lo[0][0], lo[1][0], lo[2][0], 0.f);
        part[2 * blockIdx.x + 1] = make_float4(hi[0][0], hi[1][0], hi[2][0], 0.f);
    }
}

// One block: reduce the partials; bbox[0] = cube origin (xyz) + 1/side (w), bbox[1] = hi.
__global__ void bboxFinalKernel(const float4* __restrict__ part, int nPart, float4* __restrict__ bbox)
{
    __shared__ float4 slo[kBboxBlocks], shi[kBboxBlocks];
    const int t = threadIdx.x;
    if (t < nPart) { slo[t] = part[2 * t]; shi[t] = part[2 * t + 1]; }
    __syncthreads();
    if (t == 0) {
        float4 l = slo[0], h = shi[0];
        for (int b = 1; b < nPart; ++b) {
            l.x = fminf(l.x, slo[b].x); l.y = fminf(l.y, slo[b].y); l.z = fminf(l.z, slo[b].z);
            h.x = fmaxf(h.x, shi[b].x); h.y = fmaxf(h.y, shi[b].y); h.z = fmaxf(h.z, shi[b].z);
        }
        float side = fmaxf(h.x - l.x, fmaxf(h.y - l.y, h.z - l.z));
        side = side > 0.f ? side * 1.0001f : 1.f;   // keep the max coordinate strictly inside
        bbox[0] = make_float4(l.x, l.y, l.z, 1.f / side);
        bbox[1] = h;
    }
}

template <bool HILBERT>
__global__ void keyKernel(const float4* __restrict__ pos, const float4* __restrict__ bbox, int n,
                          unsigned long long* __restrict__ key)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 o = bbox[0];
    const float4 p = pos[i];
    const float  s = o.w * 2097152.f;   // 2^21 cells per axis
    auto q = [&](float v, float lo) {
        float c = (v - lo) * s;
        c = fminf(fmaxf(c, 0.f), 2097151.f);
        return unsigned(c);
    };
    const unsigned x = q(p.x, o.x), y = q(p.y, o.y), z = q(p.z, o.z);
    key[i] = HILBERT ? hilbertKey(x, y, z) : mortonKey(x, y, z);
}

// ---- 2. gather -----------------------------------------------------------------------------
__global__ void gatherKernel(const float4* __restrict__ pos, const int* __restrict__ perm, int n,
                             float4* __restrict__ sorted)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) sorted[k] = pos[perm[k]];
}

// ---- 3a. binary radix tree over sorted keys -------------------------------------------------
// Common-prefix length of sorted keys i and j; -1 outside [0, n). Equal keys are made unique by
// appending the index (64 + clz(i ^ j)), so the tree is well defined with duplicates.
__device__ __forceinline__ int delta(const unsigned long long* __restrict__ k, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    const unsigned long long a = k[i], b = k[j];
    if (a == b) return 64 + __clz(unsigned(i ^ j));
    return __clzll(a ^ b);
}

// Internal node i in [0, n-2]; node 0 is the root.
__global__ void radixTreeKernel(const unsigned long long* __restrict__ key, int n, int4* __restrict__ info,
                                int* __restrict__ leafParent, int* __restrict__ nodeParent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;
    const int d    = (delta(key, n, i, i + 1) - delta(key, n, i, i - 1)) >= 0 ? 1 : -1;
    const int dmin = delta(key, n, i, i - d);
    int lmax = 2;
    while (delta(key, n, i, i + lmax * d) > dmin) lmax <<= 1;
    int l = 0;
    for (int t = lmax >> 1; t >= 1; t >>= 1)
        if (delta(key, n, i, i + (l + t) * d) > dmin) l += t;
    const int j     = i + l * d;
    const int dnode = delta(key, n, i, j);
    int s = 0, t = l;
    do {
        t = (t + 1) >> 1;
        if (delta(key, n, i, i + (s + t) * d) > dnode) s += t;
    } while (t > 1);
    const int gamma = i + s * d + (d < 0 ? -1 : 0);
    const int first = min(i, j), last = max(i, j);
    const int left  = (first == gamma) ? ~gamma : gamma;
    const int right = (last == gamma + 1) ? ~(gamma + 1) : gamma + 1;
    info[i] = make_int4(left, right, first, last);
    if (left < 0) leafParent[~left] = i; else nodeParent[left] = i;
    if (right < 0) leafParent[~right] = i; else nodeParent[right] = i;
    if (i == 0) nodeParent[0] = -1;
}

// ---- 3b. bottom-up multipoles ----------------------------------------------------------------
struct ChildData { float3 lo, hi, com; float m; };

__device__ __forceinline__ ChildData loadChild(int c, const float4* __restrict__ sortedPos, const float4* nodeCom,
                                               const float4* nodeLo, const float4* nodeHi)
{
    ChildData d;
    if (c < 0) {
        const float4 p = sortedPos[~c];
        d.lo = d.hi = d.com = make_float3(p.x, p.y, p.z);
        d.m  = p.w;
    } else {   // written by another thread in this kernel: bypass L1 (ld.cg)
        const float4 a = __ldcg(nodeCom + c), l = __ldcg(nodeLo + c), h = __ldcg(nodeHi + c);
        d.com = make_float3(a.x, a.y, a.z);
        d.m   = a.w;
        d.lo  = make_float3(l.x, l.y, l.z);
        d.hi  = make_float3(h.x, h.y, h.z);
    }
    return d;
}

__global__ void multipoleKernel(const float4* __restrict__ sortedPos, const int4* __restrict__ info,
                                const int* __restrict__ leafParent, const int* __restrict__ nodeParent,
                                int* __restrict__ arrive, float4* nodeCom, float4* nodeLo, float4* nodeHi,
                                float* __restrict__ nodeRc2, int n, float invTheta2)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int node = leafParent[k];
    while (node >= 0) {
        __threadfence();                                   // publish this thread's last node
        if (atomicAdd(&arrive[node], 1) == 0) return;      // first child to arrive: stop here
        __threadfence();
        const int4      in = __ldcg(info + node);
        const ChildData L  = loadChild(in.x, sortedPos, nodeCom, nodeLo, nodeHi);
        const ChildData R  = loadChild(in.y, sortedPos, nodeCom, nodeLo, nodeHi);
        const float3 lo = make_float3(fminf(L.lo.x, R.lo.x), fminf(L.lo.y, R.lo.y), fminf(L.lo.z, R.lo.z));
        const float3 hi = make_float3(fmaxf(L.hi.x, R.hi.x), fmaxf(L.hi.y, R.hi.y), fmaxf(L.hi.z, R.hi.z));
        const float  M  = L.m + R.m;
        float3 c;
        if (M > 0.f) {
            const float im = 1.f / M;
            c = make_float3((L.m * L.com.x + R.m * R.com.x) * im, (L.m * L.com.y + R.m * R.com.y) * im,
                            (L.m * L.com.z + R.m * R.com.z) * im);
        } else {   // massless node (only with --selftest-drop): geometric centre, never NaN
            c = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
        }
        // keep the COM inside the box (rounding), so bmax below really bounds every body
        c.x = fminf(fmaxf(c.x, lo.x), hi.x); c.y = fminf(fmaxf(c.y, lo.y), hi.y); c.z = fminf(fmaxf(c.z, lo.z), hi.z);
        const float bx = fmaxf(c.x - lo.x, hi.x - c.x), by = fmaxf(c.y - lo.y, hi.y - c.y),
                    bz = fmaxf(c.z - lo.z, hi.z - c.z);
        const float b2 = bx * bx + by * by + bz * bz;
        nodeCom[node] = make_float4(c.x, c.y, c.z, M);
        nodeLo[node]  = make_float4(lo.x, lo.y, lo.z, 0.f);
        nodeHi[node]  = make_float4(hi.x, hi.y, hi.z, 0.f);
        nodeRc2[node] = invTheta2 > 0.f ? b2 * invTheta2 : __int_as_float(0x7f800000);   // theta = 0: never accept
        node = nodeParent[node];
    }
}

// ---- 5. traversal ------------------------------------------------------------------------------
// Instrumented twin (TL): per-block stamps (if tl) and per-thread work counters (if wc):
//   x = nodes popped, y = interactions (cells + bodies), z = pops at which this lane was the
//   lowest active lane, w = the same at the interaction sites. Sum(z) over a warp = warp-level
//   pop steps actually executed, so sum(x) / (32 sum(z)) = mean active lanes / 32 (SIMT
//   efficiency) as the hardware ran it (__activemask), without ncu.
__device__ __forceinline__ unsigned leaderInc()
{
    const unsigned m = __activemask();
    return (threadIdx.x & 31) == unsigned(__ffs(m) - 1) ? 1u : 0u;
}

// Round 2: per-warp-unit timeline records (TLW). One BlockStamp per work unit (= 32 consecutive
// thread slots, or one persistent warp's lifetime for the lane-refill kernel); `block` = unit id.
template <bool TLW>
__device__ __forceinline__ void unitStampStart(BlockStamp* tl, int u)
{
    if constexpr (TLW) {
        if (tl && (threadIdx.x & 31) == 0) {
            BlockStamp* r = tl + u;
            r->t0 = globalTimerNs(); r->c0 = clock64(); r->smid = smId(); r->block = unsigned(u);
        }
    }
}
template <bool TLW>
__device__ __forceinline__ void unitStampEnd(BlockStamp* tl, int u)
{
    if constexpr (TLW) {
        __syncwarp();
        if (tl && (threadIdx.x & 31) == 0) { BlockStamp* r = tl + u; r->c1 = clock64(); r->t1 = globalTimerNs(); }
    }
}

// TL: instrumented twin; TLW (only with TL): per-warp-unit stamps instead of per-block stamps.
template <bool TL, bool TLW = false>
__global__ void bhTraverseKernel(const float4* __restrict__ pos, float4* __restrict__ acc,
                                 const int* __restrict__ order, int n, const float4* __restrict__ sortedPos,
                                 const float4* __restrict__ nodeCom, const float* __restrict__ nodeRc2,
                                 const int4* __restrict__ nodeInfo, int leafSize, float eps2, float G,
                                 int* __restrict__ overflow, BlockStamp* tl, uint4* wc)
{
    if constexpr (TL && !TLW) { if (tl) stampBlockStart<true>(tl); }
    if constexpr (TLW) { if (((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 32 < n) unitStampStart<true>(tl, (blockIdx.x * blockDim.x + threadIdx.x) >> 5); }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) {
        const int    body = order[t];
        const float4 bi   = pos[body];
        float3       a    = make_float3(0.f, 0.f, 0.f);
        [[maybe_unused]] unsigned nVisit = 0, nInter = 0, lVisit = 0, lInter = 0;
        int stack[kStack];
        int sp = 0;
        stack[sp++] = n > 1 ? 0 : ~0;
        while (sp > 0) {
            const int node = stack[--sp];
            if constexpr (TL) { ++nVisit; lVisit += leaderInc(); }
            if (node < 0) {                                        // single-body child
                a = bodyBodyInteraction(bi, sortedPos[~node], a, eps2);
                if constexpr (TL) { ++nInter; lInter += leaderInc(); }
                continue;
            }
            const float4 c  = nodeCom[node];
            const float  dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
            const float  d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
            if (d2 >= nodeRc2[node]) {                             // far enough: monopole
                a = bodyBodyInteraction(bi, c, a, eps2);
                if constexpr (TL) { ++nInter; lInter += leaderInc(); }
            } else {
                const int4 in = nodeInfo[node];
                if (in.w - in.z + 1 <= leafSize) {                 // small node: direct sum
                    for (int k = in.z; k <= in.w; ++k) {
                        a = bodyBodyInteraction(bi, sortedPos[k], a, eps2);
                        if constexpr (TL) { ++nInter; lInter += leaderInc(); }
                    }
                } else {
                    if (sp + 2 > kStack) { atomicAdd(overflow, 1); break; }
                    stack[sp++] = in.y;
                    stack[sp++] = in.x;                            // left first
                }
            }
        }
        acc[body] = make_float4(G * a.x, G * a.y, G * a.z, 0.f);
        if constexpr (TL) { if (wc) wc[t] = make_uint4(nVisit, nInter, lVisit, lInter); }
    }
    if constexpr (TLW) { if (((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 32 < n) unitStampEnd<true>(tl, (blockIdx.x * blockDim.x + threadIdx.x) >> 5); }
    if constexpr (TL && !TLW) {
        if (tl) stampBlockEnd<true>(tl);
    }
}

// =================================================================================================
// Tree round 2 (docs/03-design.md §17): R3 #7 (warp-cooperative union walk), R3 #6 (persistent
// threads with a global atomic queue at warp and at lane granularity, and a static cost-balanced
// partition), and the coherent control. All walks use the same tree, criterion and
// bodyBodyInteraction() as bhTraverseKernel.
// =================================================================================================
constexpr unsigned kFull = 0xffffffffu;
constexpr int      kList = 64;   // per-warp interaction-list entries (bh_warp_list), float4 each

struct TreeView {
    const float4* pos; float4* acc; const int* order; int n;
    const float4* sortedPos; const float4* nodeCom; const float* nodeRc2; const int4* nodeInfo;
    int leafSize; float eps2; float G; int* overflow;
};

// Counters: persistent kernels accumulate (atomicAdd into a zeroed buffer), so a body processed
// twice shows doubled counts and one never processed shows zeros (exactly-once check).
__device__ __forceinline__ void putWc(uint4* wc, int t, uint4 c, bool accumulate)
{
    if (!wc) return;
    if (accumulate) {
        atomicAdd(&wc[t].x, c.x); atomicAdd(&wc[t].y, c.y); atomicAdd(&wc[t].z, c.z); atomicAdd(&wc[t].w, c.w);
    } else {
        wc[t] = c;
    }
}

// The #4/#5 per-thread walk for thread slot t, identical statement for statement to
// bhTraverseKernel's body (so its accelerations are bitwise equal), for the persistent kernels.
template <bool TL>
__device__ __forceinline__ void threadWalk(const TreeView& v, int t, uint4* wc, bool accWc)
{
    const int    body = v.order[t];
    const float4 bi   = v.pos[body];
    float3       a    = make_float3(0.f, 0.f, 0.f);
    [[maybe_unused]] unsigned nVisit = 0, nInter = 0, lVisit = 0, lInter = 0;
    int stack[kStack];
    int sp = 0;
    stack[sp++] = v.n > 1 ? 0 : ~0;
    while (sp > 0) {
        const int node = stack[--sp];
        if constexpr (TL) { ++nVisit; lVisit += leaderInc(); }
        if (node < 0) {
            a = bodyBodyInteraction(bi, v.sortedPos[~node], a, v.eps2);
            if constexpr (TL) { ++nInter; lInter += leaderInc(); }
            continue;
        }
        const float4 c  = v.nodeCom[node];
        const float  dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
        const float  d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
        if (d2 >= v.nodeRc2[node]) {
            a = bodyBodyInteraction(bi, c, a, v.eps2);
            if constexpr (TL) { ++nInter; lInter += leaderInc(); }
        } else {
            const int4 in = v.nodeInfo[node];
            if (in.w - in.z + 1 <= v.leafSize) {
                for (int k = in.z; k <= in.w; ++k) {
                    a = bodyBodyInteraction(bi, v.sortedPos[k], a, v.eps2);
                    if constexpr (TL) { ++nInter; lInter += leaderInc(); }
                }
            } else {
                if (sp + 2 > kStack) { atomicAdd(v.overflow, 1); break; }
                stack[sp++] = in.y;
                stack[sp++] = in.x;
            }
        }
    }
    v.acc[body] = make_float4(v.G * a.x, v.G * a.y, v.G * a.z, 0.f);
    if constexpr (TL) putWc(wc, t, make_uint4(nVisit, nInter, lVisit, lInter), accWc);
}

// ---- R3 #7: warp-cooperative union walk for work unit u (thread slots 32u .. 32u+31) ------------
// Every lane of the warp must call it. The warp pops one node at a time from a warp-shared stack;
// the node is opened iff ANY valid lane's bmax criterion opens it (__any_sync over the union of
// the lanes' decisions, burtscher2011barneshut), so control flow is warp-uniform: no idle lanes,
// all loads same-address (broadcast). A lane whose own criterion would have accepted an opened
// node gets its children instead (finer, never fewer interactions: I_lane(#7) >= I_lane(#5)).
// Stack: 64 entries held in two registers per lane (entry k lives in lane k % 32, register
// k / 32), pushed with a predicated move and popped with one __shfl_sync: no shared memory and no
// __syncwarp on the walk. LIST: accepted cells and bucket bodies are appended to a per-warp
// shared-memory list (bucket bodies loaded cooperatively, one per lane, coalesced) and evaluated
// in a dense loop when it fills; the summation order equals the direct walk's, so
// bh_warp_list is bitwise equal to bh_warp.
template <bool TL, bool LIST>
__device__ __forceinline__ void warpWalk(const TreeView& v, int u, float4* list, uint4* wc, bool accWc)
{
    const int    lane  = threadIdx.x & 31;
    const int    t     = u * 32 + lane;
    const bool   valid = t < v.n;
    const int    body  = valid ? v.order[t] : 0;
    const float4 bi    = valid ? v.pos[body] : make_float4(0.f, 0.f, 0.f, 0.f);
    float3       a     = make_float3(0.f, 0.f, 0.f);
    [[maybe_unused]] unsigned nVisit = 0, nInter = 0, lVisit = 0, lInter = 0;
    int s0 = v.n > 1 ? 0 : ~0, s1 = 0, sp = 1;
    [[maybe_unused]] int cnt = 0;
    auto push = [&](int x) {
        if (lane == (sp & 31)) { if (sp < 32) s0 = x; else s1 = x; }
        ++sp;
    };
    auto flush = [&]() {
        if constexpr (LIST) {
            __syncwarp();
#pragma unroll 4
            for (int j = 0; j < cnt; ++j) {
                a = bodyBodyInteraction(bi, list[j], a, v.eps2);
                if constexpr (TL) { nInter += valid; lInter += leaderInc(); }
            }
            __syncwarp();
            cnt = 0;
        }
    };
    while (sp > 0) {
        --sp;
        const int node = __shfl_sync(kFull, sp < 32 ? s0 : s1, sp & 31);
        if constexpr (TL) { nVisit += valid; lVisit += leaderInc(); }
        if (node < 0) {
            if constexpr (LIST) {
                if (cnt == kList) flush();
                if (lane == 0) list[cnt] = v.sortedPos[~node];
                ++cnt;
            } else {
                a = bodyBodyInteraction(bi, v.sortedPos[~node], a, v.eps2);
                if constexpr (TL) { nInter += valid; lInter += leaderInc(); }
            }
            continue;
        }
        const float4 c  = v.nodeCom[node];
        const float  dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
        const float  d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
        const bool   open = __any_sync(kFull, valid && !(d2 >= v.nodeRc2[node]));
        if (!open) {
            if constexpr (LIST) {
                if (cnt == kList) flush();
                if (lane == 0) list[cnt] = c;
                ++cnt;
            } else {
                a = bodyBodyInteraction(bi, c, a, v.eps2);
                if constexpr (TL) { nInter += valid; lInter += leaderInc(); }
            }
        } else {
            const int4 in = v.nodeInfo[node];
            if (in.w - in.z + 1 <= v.leafSize) {
                if constexpr (LIST) {
                    for (int k0 = in.z; k0 <= in.w; k0 += 32) {
                        const int m = min(32, in.w - k0 + 1);
                        if (cnt + m > kList) flush();
                        if (lane < m) list[cnt + lane] = v.sortedPos[k0 + lane];
                        cnt += m;
                    }
                } else {
                    for (int k = in.z; k <= in.w; ++k) {
                        a = bodyBodyInteraction(bi, v.sortedPos[k], a, v.eps2);
                        if constexpr (TL) { nInter += valid; lInter += leaderInc(); }
                    }
                }
            } else {
                if (sp + 2 > kStack) { if (lane == 0) atomicAdd(v.overflow, 1); break; }
                push(in.y);
                push(in.x);                                         // left first, as in #5
            }
        }
    }
    flush();
    if (valid) {
        v.acc[body] = make_float4(v.G * a.x, v.G * a.y, v.G * a.z, 0.f);
        if constexpr (TL) putWc(wc, t, make_uint4(nVisit, nInter, lVisit, lInter), accWc);
    }
}

// Normal grid: warp w of the grid = unit w. dynamic smem = list (LIST) + padding (--occ-blocks).
template <bool TL, bool TLW, bool LIST>
__global__ void bhWarpKernel(TreeView v, BlockStamp* tl, uint4* wc)
{
    extern __shared__ float4 smemList[];
    if constexpr (TL && !TLW) { if (tl) stampBlockStart<true>(tl); }
    const int u      = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nUnits = (v.n + 31) >> 5;
    if (u < nUnits) {                                               // warp-uniform
        unitStampStart<TLW>(tl, u);
        warpWalk<TL, LIST>(v, u, LIST ? smemList + (threadIdx.x >> 5) * kList : nullptr, wc, false);
        unitStampEnd<TLW>(tl, u);
    }
    if constexpr (TL && !TLW) { if (tl) stampBlockEnd<true>(tl); }
}

// ---- R3 #6: persistent blocks (grid = resident blocks/SM x SMs) ---------------------------------
// WARP-granular units (32 consecutive thread slots, Morton order) handed out by
//   STATIC = false: a global atomic counter, one atomicAdd per unit by lane 0 [huang2026weave:
//                   "each SM claims its next tile via atomicAdd on a global counter"];
//   STATIC = true:  a precomputed contiguous partition balanced on the previous launch's measured
//                   per-unit cost (clock64 cycles; costzones-style) [xia2025voltrix: persistent
//                   kernel + precomputed balanced partition; li2025staticbatching: prefix sum of
//                   per-task cost, no runtime scheduler].
// WARPWALK: the unit is walked by #7's union walk instead of #5's per-thread walk.
template <bool TL, bool TLW, bool WARPWALK, bool STATIC>
__global__ void bhPersistentKernel(TreeView v, int* queue, const int* warpStart, unsigned long long* unitCost,
                                   BlockStamp* tl, uint4* wc)
{
    if constexpr (TL && !TLW) { if (tl) stampBlockStart<true>(tl); }
    const int lane   = threadIdx.x & 31;
    [[maybe_unused]] const int gw     = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    [[maybe_unused]] const int nUnits = (v.n + 31) >> 5;
    int u = 0;
    [[maybe_unused]] int uEnd = 0;
    if constexpr (STATIC) { u = warpStart[gw]; uEnd = warpStart[gw + 1]; }
    for (;;) {
        if constexpr (STATIC) {
            if (u >= uEnd) break;
        } else {
            int x = 0;
            if (lane == 0) x = atomicAdd(queue, 1);
            u = __shfl_sync(kFull, x, 0);
            if (u >= nUnits) break;
        }
        [[maybe_unused]] long long c0 = 0;
        if constexpr (STATIC) c0 = clock64();
        unitStampStart<TLW>(tl, u);
        if constexpr (WARPWALK) {
            warpWalk<TL, false>(v, u, nullptr, wc, true);
        } else {
            const int t = u * 32 + lane;
            if (t < v.n) threadWalk<TL>(v, t, wc, true);
        }
        unitStampEnd<TLW>(tl, u);
        if constexpr (STATIC) {
            __syncwarp();
            if (lane == 0) unitCost[u] = (unsigned long long)(clock64() - c0);
            ++u;
        }
    }
    if constexpr (TL && !TLW) { if (tl) stampBlockEnd<true>(tl); }
}

// LANE-granular persistent threads: every lane runs #5's walk for one body at a time; when its
// stack empties it writes the result and the warp refills all idle lanes with the next bodies
// (Morton slots) from the global counter, one atomicAdd per refill (the leader adds popc(idle)).
// This targets the loop-level loss (lanes done, waiting for the warp's slowest lane) that a
// warp- or block-granular queue cannot touch. One loop iteration = one node pop (+ its leaf
// bucket), identical per-body arithmetic to bhTraverseKernel (bitwise-equal accelerations).
// TLW: one record per persistent warp (unit = global warp id; units are bodies here).
template <bool TL, bool TLW>
__global__ void bhLaneRefillKernel(TreeView v, int* queue, BlockStamp* tl, uint4* wc)
{
    if constexpr (TL && !TLW) { if (tl) stampBlockStart<true>(tl); }
    const int      lane = threadIdx.x & 31;
    const int      gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned lt   = (1u << lane) - 1u;
    unitStampStart<TLW>(tl, gw);
    int    t = -1, body = 0, sp = 0;
    float4 bi = make_float4(0.f, 0.f, 0.f, 0.f);
    float3 a  = make_float3(0.f, 0.f, 0.f);
    [[maybe_unused]] unsigned nVisit = 0, nInter = 0, lVisit = 0, lInter = 0;
    int  stack[kStack];
    // Batched claims: the warp owns a buffer of up to 32 consecutive slots [bufBase, bufBase +
    // bufLeft) and lane 0 holds the NEXT batch's atomicAdd result (pend), issued one batch ahead,
    // so an idle lane is refilled from registers; the atomic's latency is only exposed when the
    // prefetched result has not arrived yet (once per 32 bodies at most).
    int  bufBase = 0, bufLeft = 0, pend = 0;
    bool drained = false;
    {
        int b0 = 0;
        if (lane == 0) b0 = atomicAdd(queue, 32);
        bufBase = __shfl_sync(kFull, b0, 0);
        bufLeft = bufBase < v.n ? 32 : 0;
        drained = bufLeft == 0;
        if (lane == 0 && !drained) pend = atomicAdd(queue, 32);
    }
    for (;;) {
        if (t >= 0 && sp == 0) {                                    // body finished
            v.acc[body] = make_float4(v.G * a.x, v.G * a.y, v.G * a.z, 0.f);
            if constexpr (TL) putWc(wc, t, make_uint4(nVisit, nInter, lVisit, lInter), true);
            t = -1;
        }
        unsigned idle = __ballot_sync(kFull, t < 0);
        while (idle && !drained) {                                  // warp-uniform
            const int take = min(__popc(idle), bufLeft);
            const int rank = __popc(idle & lt);
            if (t < 0 && rank < take) {
                const int cand = bufBase + rank;
                if (cand < v.n) {
                    t = cand; body = v.order[t]; bi = v.pos[body]; a = make_float3(0.f, 0.f, 0.f);
                    sp = 0; stack[sp++] = v.n > 1 ? 0 : ~0;
                    if constexpr (TL) { nVisit = nInter = lVisit = lInter = 0; }
                }
            }
            bufBase += take; bufLeft -= take;
            if (bufBase >= v.n) { drained = true; break; }
            if (bufLeft == 0) {                                     // switch to the prefetched batch
                bufBase = __shfl_sync(kFull, pend, 0);
                bufLeft = 32;
                if (bufBase >= v.n) { drained = true; break; }
                if (lane == 0) pend = atomicAdd(queue, 32);
            }
            idle = __ballot_sync(kFull, t < 0);
        }
        if (__all_sync(kFull, t < 0)) break;                        // only once drained
        if (t >= 0) {
            const int node = stack[--sp];
            if constexpr (TL) { ++nVisit; lVisit += leaderInc(); }
            if (node < 0) {
                a = bodyBodyInteraction(bi, v.sortedPos[~node], a, v.eps2);
                if constexpr (TL) { ++nInter; lInter += leaderInc(); }
            } else {
                const float4 c  = v.nodeCom[node];
                const float  dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
                const float  d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                if (d2 >= v.nodeRc2[node]) {
                    a = bodyBodyInteraction(bi, c, a, v.eps2);
                    if constexpr (TL) { ++nInter; lInter += leaderInc(); }
                } else {
                    const int4 in = v.nodeInfo[node];
                    if (in.w - in.z + 1 <= v.leafSize) {
                        for (int k = in.z; k <= in.w; ++k) {
                            a = bodyBodyInteraction(bi, v.sortedPos[k], a, v.eps2);
                            if constexpr (TL) { ++nInter; lInter += leaderInc(); }
                        }
                    } else if (sp + 2 > kStack) {
                        atomicAdd(v.overflow, 1); sp = 0;               // abandon this body (flagged)
                    } else {
                        stack[sp++] = in.y;
                        stack[sp++] = in.x;
                    }
                }
            }
        }
    }
    unitStampEnd<TLW>(tl, gw);
    if constexpr (TL && !TLW) { if (tl) stampBlockEnd<true>(tl); }
}

// Static partition (bh_persistent_static): costs of the previous launch -> inclusive prefix P ->
// unit u belongs to warp floor((2 P[u] - c[u]) G / (2 T)) (its cost midpoint); warpStart[g] = the
// first unit of warp g (monotone owners, contiguous ranges), warpStart[G] = nUnits.
__global__ void fillCostKernel(unsigned long long* c, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) c[i] = 1ull;
}
__global__ void partitionKernel(const unsigned long long* __restrict__ cost, const unsigned long long* __restrict__ pre,
                                int nUnits, int G, int* __restrict__ warpStart)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nUnits) return;
    const unsigned long long T = pre[nUnits - 1] > 0 ? pre[nUnits - 1] : 1ull;
    auto owner = [&](int k) {
        const double mid = (2.0 * double(pre[k]) - double(cost[k])) / (2.0 * double(T));
        return min(G - 1, max(0, int(mid * G)));
    };
    const int o  = owner(u);
    const int op = u > 0 ? owner(u - 1) : -1;
    for (int gq = op + 1; gq <= o; ++gq) warpStart[gq] = u;
    if (u == nUnits - 1)
        for (int gq = o + 1; gq <= G; ++gq) warpStart[gq] = nUnits;
}

__global__ void coherentOrderKernel(const int* __restrict__ perm, int n, int* __restrict__ out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) out[t] = perm[t & ~31];
}

// ---- launcher -------------------------------------------------------------------------------------
const char* const kPhaseNames[] = {"keys", "sort", "tree", "order", "traverse"};
constexpr int     kPhases       = 5;

void checkParams(const Params& p)
{
    // theta <= 1 (critic review 2, #11): the bmax safety argument (a body inside a node's box always
    // opens it) holds only for theta <= 1.
    if (!(p.theta >= 0.f) || p.theta > 1.f) { fprintf(stderr, "bh: theta must be in [0, 1] (got %g)\n", p.theta); exit(2); }
    if (p.leafSize < 1)    { fprintf(stderr, "bh: leaf size must be >= 1 (got %d)\n", p.leafSize); exit(2); }
    if (p.blockSize < 32 || p.blockSize > 1024 || p.blockSize % 32) {
        fprintf(stderr, "bh: block must be a multiple of 32 in [32, 1024] (got %d)\n", p.blockSize); exit(2);
    }
}

// Traversal variants (round 1: Thread; round 2: the rest). See registerTree() for the names.
enum class Var { Thread, Warp, WarpList, PersUnit, PersLane, PersWarp, PersStatic };

bool isPersistent(Var v) { return v == Var::PersUnit || v == Var::PersLane || v == Var::PersWarp || v == Var::PersStatic; }

template <bool TL, bool TLW>
const void* traverseFnT(Var v)
{
    switch (v) {
    case Var::Thread:     return reinterpret_cast<const void*>(bhTraverseKernel<TL, TLW>);
    case Var::Warp:       return reinterpret_cast<const void*>(bhWarpKernel<TL, TLW, false>);
    case Var::WarpList:   return reinterpret_cast<const void*>(bhWarpKernel<TL, TLW, true>);
    case Var::PersUnit:   return reinterpret_cast<const void*>(bhPersistentKernel<TL, TLW, false, false>);
    case Var::PersWarp:   return reinterpret_cast<const void*>(bhPersistentKernel<TL, TLW, true, false>);
    case Var::PersStatic: return reinterpret_cast<const void*>(bhPersistentKernel<TL, TLW, false, true>);
    case Var::PersLane:   return reinterpret_cast<const void*>(bhLaneRefillKernel<TL, TLW>);
    }
    return nullptr;
}
const void* traverseFn(Var v, bool tl, bool tlw)
{
    if (!tl) return traverseFnT<false, false>(v);
    return tlw ? traverseFnT<true, true>(v) : traverseFnT<true, false>(v);
}

size_t dynSmemOf(Var v, const Params& p)
{
    return (v == Var::WarpList ? size_t(p.blockSize / 32) * kList * sizeof(float4) : 0) + size_t(std::max(0, p.smemPad));
}

// Resident blocks/SM for (kernel, block, dynamic smem), cached; also raises the dynamic-smem limit
// above 48 KB when needed (--occ-blocks). Cached so that timed launches make no extra API calls.
int residentBlocks(const void* fn, int block, size_t dyn)
{
    struct E { const void* fn; int block; size_t dyn; int bps; };
    static std::vector<E> cache;
    for (const E& e : cache) if (e.fn == fn && e.block == block && e.dyn == dyn) return e.bps;
    if (dyn > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dyn)));
    int bps = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, fn, block, dyn));
    cache.push_back({fn, block, dyn, bps});
    return bps;
}

int gridOf(Var v, const void* fn, int n, const Params& p)
{
    if (!isPersistent(v)) return ceilDiv(n, p.blockSize);
    // Persistent grid = what fits on the GPU, but never more blocks than there is work for: with
    // more resident warps than 32-body units (N < 122,880 at 12 blocks/SM x 4 warps), the excess
    // warps race for the few units and the winners cluster on a few SMs (sanity run, N = 8192:
    // 0.75 ms vs 0.51 ms for the normal grid). Capped, the hardware spreads the blocks as it does
    // for a normal grid.
    const int resident = std::max(1, residentBlocks(fn, p.blockSize, dynSmemOf(v, p)) * deviceSMCount());
    return std::max(1, std::min(resident, ceilDiv(ceilDiv(n, 32), p.blockSize / 32)));
}

void launchBH(const DeviceState& s, const Params& p, cudaStream_t st, Order ord, Var var)
{
    checkParams(p);
    const int n = s.n;
    ensureCapacity(n);
    auto mark = [&](int k) { if (p.phaseEvents) CUDA_CHECK(cudaEventRecord(p.phaseEvents[k], st)); };
    const int gb = ceilDiv(n, kBuildBlock);

    mark(0);
    bboxPartialKernel<<<kBboxBlocks, kBboxThreads, 0, st>>>(s.pos, n, g.bboxPartial);
    bboxFinalKernel<<<1, kBboxBlocks, 0, st>>>(g.bboxPartial, kBboxBlocks, g.bbox);
    keyKernel<false><<<gb, kBuildBlock, 0, st>>>(s.pos, g.bbox, n, g.keyIn);
    mark(1);
    size_t tmp = g.sortTmpBytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(g.sortTmp, tmp, g.keyIn, g.keySorted, g.iota, g.perm, n, 0, 63, st));
    gatherKernel<<<gb, kBuildBlock, 0, st>>>(s.pos, g.perm, n, g.sortedPos);
    mark(2);
    if (n > 1) {
        CUDA_CHECK(cudaMemsetAsync(g.arrive, 0, size_t(n - 1) * 4, st));
        radixTreeKernel<<<ceilDiv(n - 1, kBuildBlock), kBuildBlock, 0, st>>>(g.keySorted, n, g.nodeInfo,
                                                                              g.leafParent, g.nodeParent);
        const float invT2 = p.theta > 0.f ? 1.f / (p.theta * p.theta) : 0.f;
        multipoleKernel<<<gb, kBuildBlock, 0, st>>>(g.sortedPos, g.nodeInfo, g.leafParent, g.nodeParent, g.arrive,
                                                    g.nodeCom, g.nodeLo, g.nodeHi, g.nodeRc2, n, invT2);
    }
    mark(3);
    // ---- order phase: the thread -> body map, plus the scheduler set-up of the persistent kernels
    const int* order = g.iota;
    if (ord == Order::Morton) order = g.perm;
    if (ord == Order::Hilbert) {
        keyKernel<true><<<gb, kBuildBlock, 0, st>>>(s.pos, g.bbox, n, g.keyIn);   // Morton keyIn no longer needed
        tmp = g.sortTmpBytes;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(g.sortTmp, tmp, g.keyIn, g.keyH, g.iota, g.hOrder, n, 0, 63, st));
        order = g.hOrder;
    }
    if (ord == Order::Coherent) {
        coherentOrderKernel<<<gb, kBuildBlock, 0, st>>>(g.perm, n, g.cohOrder);
        order = g.cohOrder;
    }
    const bool   tl     = p.timeline || p.workCounters;
    const bool   tlw    = tl && p.timelineWarp && p.timeline;
    const void*  fn     = traverseFn(var, tl, tlw);
    const size_t dyn    = dynSmemOf(var, p);
    const int    grid   = gridOf(var, fn, n, p);
    const int    nUnits = ceilDiv(n, 32);
    if (dyn > 48 * 1024) residentBlocks(fn, p.blockSize, dyn);    // raises the opt-in limit once
    if (var == Var::PersUnit || var == Var::PersLane || var == Var::PersWarp)
        CUDA_CHECK(cudaMemsetAsync(g.queue, 0, 4, st));
    if (var == Var::PersStatic) {
        const int G = grid * p.blockSize / 32;
        if (G + 1 > g.warpStartCap) {
            if (g.warpStart) CUDA_CHECK(cudaFree(g.warpStart));
            CUDA_CHECK(cudaMalloc(&g.warpStart, size_t(G + 1) * 4));
            g.warpStartCap = G + 1;
        }
        if (g.costN != n || g.costG != G) {   // no previous launch with this N and grid: uniform costs
            fillCostKernel<<<ceilDiv(nUnits, 256), 256, 0, st>>>(g.unitCost, nUnits);
            g.costN = n; g.costG = G;
        }
        size_t stmp = g.scanTmpBytes;
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(g.scanTmp, stmp, g.unitCost, g.unitPre, nUnits, st));
        partitionKernel<<<ceilDiv(nUnits, 256), 256, 0, st>>>(g.unitCost, g.unitPre, nUnits, G, g.warpStart);
    }
    mark(4);
    // ---- traverse phase
    const float eps2 = p.softening * p.softening;
    BlockStamp* tlp  = tl ? p.timeline : nullptr;
    uint4*      wcp  = tl ? p.workCounters : nullptr;
    TreeView v{s.pos, s.acc, order, n, g.sortedPos, g.nodeCom, g.nodeRc2, g.nodeInfo, p.leafSize, eps2, p.G, g.overflow};
    const dim3 G3(grid), B3(p.blockSize);
    if (var == Var::Thread) {
        if (!tl)
            bhTraverseKernel<false><<<G3, B3, dyn, st>>>(s.pos, s.acc, order, n, g.sortedPos, g.nodeCom, g.nodeRc2,
                                                         g.nodeInfo, p.leafSize, eps2, p.G, g.overflow, nullptr, nullptr);
        else if (!tlw)
            bhTraverseKernel<true><<<G3, B3, dyn, st>>>(s.pos, s.acc, order, n, g.sortedPos, g.nodeCom, g.nodeRc2,
                                                        g.nodeInfo, p.leafSize, eps2, p.G, g.overflow, tlp, wcp);
        else
            bhTraverseKernel<true, true><<<G3, B3, dyn, st>>>(s.pos, s.acc, order, n, g.sortedPos, g.nodeCom,
                                                              g.nodeRc2, g.nodeInfo, p.leafSize, eps2, p.G, g.overflow,
                                                              tlp, wcp);
    } else {
        // Round-2 kernels: cudaLaunchKernel with the selected instantiation.
        int* q = g.queue; const int* ws = g.warpStart; unsigned long long* uc = g.unitCost;
        void* argsWarp[] = {&v, &tlp, &wcp};
        void* argsPers[] = {&v, &q, &ws, &uc, &tlp, &wcp};
        void* argsLane[] = {&v, &q, &tlp, &wcp};
        void** args = (var == Var::Warp || var == Var::WarpList) ? argsWarp : (var == Var::PersLane ? argsLane : argsPers);
        CUDA_CHECK(cudaLaunchKernel(fn, G3, B3, args, dyn, st));
    }
    mark(5);
    CUDA_CHECK(cudaGetLastError());
    g.lastN = n; g.lastTheta = p.theta; g.lastLeaf = p.leafSize; g.lastOrder = ord;
    g.lastEps2 = eps2;
}

template <Var V, Order O>
void launchV(const DeviceState& s, const Params& p, cudaStream_t st) { launchBH(s, p, st, O, V); }

LaunchInfo queryV(int n, const Params& p, Var var)
{
    const bool   inst = p.timeline || p.workCounters;
    const bool   tlw  = inst && p.timelineWarp && p.timeline;
    const void*  fn   = traverseFn(var, inst, tlw);
    const size_t dyn  = dynSmemOf(var, p);
    const int    grid = gridOf(var, fn, n, p);
    if (dyn > 48 * 1024) residentBlocks(fn, p.blockSize, dyn);
    LaunchInfo li   = describeKernel(fn, grid, p.blockSize, dyn);
    li.instrumented = inst;
    li.approximate  = true;
    li.nPhases      = kPhases;
    li.persistent   = isPersistent(var);
    li.dynSmem      = dyn;
    for (int k = 0; k < kPhases; ++k) li.phaseNames[k] = kPhaseNames[k];
    if (tlw) {
        li.timelineRecords    = var == Var::PersLane ? grid * p.blockSize / 32 : ceilDiv(n, 32);
        li.timelineSlotsPerSM = li.blocksPerSM * p.blockSize / 32;
    } else {
        li.timelineRecords    = grid;
        li.timelineSlotsPerSM = li.blocksPerSM;
    }
    return li;
}

LaunchInfo queryBH(int n, const Params& p) { return queryV(n, p, Var::Thread); }
LaunchInfo queryCoherent(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::Thread);
    li.control = true;
    return li;
}
LaunchInfo queryWarp(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::Warp);
    li.errorSibling = li.counterSibling = "bh_sorted";
    return li;
}
LaunchInfo queryWarpNaive(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::Warp);
    li.errorSibling = li.counterSibling = "bh_naive";
    return li;
}
LaunchInfo queryWarpList(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::WarpList);
    li.errorSibling = li.counterSibling = "bh_sorted";
    li.exactSibling = "bh_warp";
    return li;
}
LaunchInfo queryPersUnit(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::PersUnit);
    li.exactSibling = li.counterSibling = "bh_sorted";
    return li;
}
LaunchInfo queryPersLane(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::PersLane);
    li.exactSibling = li.counterSibling = "bh_sorted";
    return li;
}
LaunchInfo queryPersLaneNaive(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::PersLane);
    li.exactSibling = li.counterSibling = "bh_naive";
    return li;
}
LaunchInfo queryPersWarp(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::PersWarp);
    li.exactSibling = "bh_warp";
    li.errorSibling = li.counterSibling = "bh_sorted";
    return li;
}
LaunchInfo queryPersStatic(int n, const Params& p)
{
    LaunchInfo li = queryV(n, p, Var::PersStatic);
    li.exactSibling = li.counterSibling = "bh_sorted";
    return li;
}

} // namespace

int treeStackOverflows(bool reset)
{
    if (!g.overflow) return 0;
    int h = 0;
    CUDA_CHECK(cudaMemcpy(&h, g.overflow, 4, cudaMemcpyDeviceToHost));
    if (reset) CUDA_CHECK(cudaMemset(g.overflow, 0, 4));
    return h;
}

// Host-side structural check of the most recent tree (docs/03-design.md §12.3).
TreeCheck checkLastTree(const DeviceState& s)
{
    TreeCheck r;
    const int n = g.lastN;
    r.n = n;
    auto fail = [&](const char* m) { if (r.ok) { r.ok = false; r.msg = m; } };
    if (n <= 0 || n != s.n) { fail("no tree for this state (launch a bh_* kernel first)"); return r; }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> key(n);
    std::vector<int> iota(n), perm(n), hOrd(n);
    std::vector<float4> pos(n), spos(n);
    CUDA_CHECK(cudaMemcpy(key.data(), g.keySorted, n * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(iota.data(), g.iota, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(perm.data(), g.perm, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos.data(), s.pos, size_t(n) * 16, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(spos.data(), g.sortedPos, size_t(n) * 16, cudaMemcpyDeviceToHost));
    // permutations
    std::vector<int> seen(n, 0);
    for (int k = 0; k < n; ++k) {
        if (iota[k] != k) fail("identity order was modified");
        if (perm[k] < 0 || perm[k] >= n || seen[perm[k]]++) fail("Morton perm is not a permutation");
        if (k && key[k] < key[k - 1]) fail("sorted keys not monotone");
        if (perm[k] >= 0 && perm[k] < n && std::memcmp(&spos[k], &pos[perm[k]], 16) != 0) fail("sortedPos != pos[perm]");
    }
    if (g.lastOrder == Order::Hilbert) {
        CUDA_CHECK(cudaMemcpy(hOrd.data(), g.hOrder, n * 4, cudaMemcpyDeviceToHost));
        std::fill(seen.begin(), seen.end(), 0);
        for (int k = 0; k < n; ++k)
            if (hOrd[k] < 0 || hOrd[k] >= n || seen[hOrd[k]]++) fail("Hilbert order is not a permutation");
    }
    double M = 0, cx = 0, cy = 0, cz = 0, ext = 0;
    for (int k = 0; k < n; ++k) {
        M += pos[k].w; cx += double(pos[k].w) * pos[k].x; cy += double(pos[k].w) * pos[k].y;
        cz += double(pos[k].w) * pos[k].z;
        ext = std::max(ext, double(std::max(std::fabs(pos[k].x), std::max(std::fabs(pos[k].y), std::fabs(pos[k].z)))));
    }
    if (n == 1) { r.leafBodies = 1; r.maxDepth = 0; return r; }
    const int I = n - 1;
    r.internalNodes = I;
    std::vector<int4> info(I);
    std::vector<float4> com(I), lo(I), hi(I);
    std::vector<float> rc2(I);
    CUDA_CHECK(cudaMemcpy(info.data(), g.nodeInfo, size_t(I) * 16, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(com.data(), g.nodeCom, size_t(I) * 16, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lo.data(), g.nodeLo, size_t(I) * 16, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hi.data(), g.nodeHi, size_t(I) * 16, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rc2.data(), g.nodeRc2, size_t(I) * 4, cudaMemcpyDeviceToHost));
    // DFS from the root: every body must be reached exactly once, every internal node once,
    // node ranges contiguous = [left.first, right.last], masses additive, boxes nested.
    std::vector<int> bodyHits(n, 0), nodeHits(I, 0);
    struct Item { int node, depth, parentCount; };
    std::vector<Item> st{{0, 0, n + 1}};
    auto rangeOf = [&](int c, int& f, int& l) { if (c < 0) f = l = ~c; else { f = info[c].z; l = info[c].w; } };
    auto massOf  = [&](int c) { return c < 0 ? double(spos[~c].w) : double(com[c].w); };
    const float tol = float(1e-5 * (ext > 0 ? ext : 1.0));
    while (!st.empty()) {
        Item it = st.back(); st.pop_back();
        r.maxDepth = std::max(r.maxDepth, it.depth);
        if (it.node < 0) { if (++bodyHits[~it.node] > 1) fail("body reached twice"); continue; }
        if (it.node >= I || nodeHits[it.node]++) { fail("internal node reached twice / out of range"); continue; }
        const int4 in = info[it.node];
        int fl, ll, fr, lr;
        rangeOf(in.x, fl, ll); rangeOf(in.y, fr, lr);
        if (fl != in.z || lr != in.w || ll + 1 != fr) fail("node range is not the union of its children");
        const double m = massOf(in.x) + massOf(in.y);
        if (std::fabs(m - com[it.node].w) > 1e-5 * std::max(1e-30, std::fabs(m))) fail("node mass != sum of children");
        for (int c : {in.x, in.y}) {
            float3 cl, ch;
            if (c < 0) { cl = ch = make_float3(spos[~c].x, spos[~c].y, spos[~c].z); }
            else { cl = make_float3(lo[c].x, lo[c].y, lo[c].z); ch = make_float3(hi[c].x, hi[c].y, hi[c].z); }
            if (cl.x < lo[it.node].x || cl.y < lo[it.node].y || cl.z < lo[it.node].z || ch.x > hi[it.node].x ||
                ch.y > hi[it.node].y || ch.z > hi[it.node].z) fail("child box not inside parent box");
        }
        const float4 c = com[it.node];
        if (c.x < lo[it.node].x - tol || c.x > hi[it.node].x + tol || c.y < lo[it.node].y - tol ||
            c.y > hi[it.node].y + tol || c.z < lo[it.node].z - tol || c.z > hi[it.node].z + tol)
            fail("COM outside the node box");
        if (!(rc2[it.node] >= 0.f)) fail("opening radius NaN/negative");
        const int cnt = in.w - in.z + 1;   // terminal bucket: small node whose parent is not small
        if (cnt <= g.lastLeaf && it.parentCount > g.lastLeaf) ++r.leafBuckets;
        st.push_back({in.y, it.depth + 1, cnt});
        st.push_back({in.x, it.depth + 1, cnt});
    }
    for (int k = 0; k < n; ++k) if (bodyHits[k] != 1) { fail("body not in the tree exactly once"); break; }
    for (int i = 0; i < I; ++i) if (nodeHits[i] != 1) { fail("internal node unreachable"); break; }
    r.leafBodies = 0;
    for (int k = 0; k < n; ++k) r.leafBodies += bodyHits[k];
    r.massRelErr = M > 0 ? std::fabs(com[0].w - M) / M : 0.0;
    if (M > 0) {
        const double ex = cx / M - com[0].x, ey = cy / M - com[0].y, ez = cz / M - com[0].z;
        r.comAbsErr = std::sqrt(ex * ex + ey * ey + ez * ez);
    }
    if (r.massRelErr > 1e-5) fail("root mass != total mass");
    if (r.comAbsErr > 1e-5 * (ext > 0 ? ext : 1.0) + 1e-6) fail("root COM != centre of mass");
    return r;
}

// Exact theta > 0 check of the most recent tree and walk (critic review 2, #7); see nbody.h.
TreeExactCheck checkLastTreeExact(const std::vector<float4>& acc, const std::vector<uint4>* wc, float eps2, float G,
                                  bool unionWalk)
{
    TreeExactCheck r;
    const int n = g.lastN;
    auto fail = [&](const std::string& m) { if (r.ok) { r.ok = false; r.msg = m; } };
    if (n <= 0 || int(acc.size()) != n) { fail("no tree / acc size mismatch"); return r; }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float4> spos(n);
    std::vector<int> perm(n);
    CUDA_CHECK(cudaMemcpy(spos.data(), g.sortedPos, size_t(n) * 16, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(perm.data(), g.perm, size_t(n) * 4, cudaMemcpyDeviceToHost));
    const int I = n > 1 ? n - 1 : 0;
    std::vector<int4> info(I);
    std::vector<float4> com(I), lo(I), hi(I);
    std::vector<float> rc2(I);
    if (I) {
        CUDA_CHECK(cudaMemcpy(info.data(), g.nodeInfo, size_t(I) * 16, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(com.data(), g.nodeCom, size_t(I) * 16, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lo.data(), g.nodeLo, size_t(I) * 16, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hi.data(), g.nodeHi, size_t(I) * 16, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rc2.data(), g.nodeRc2, size_t(I) * 4, cudaMemcpyDeviceToHost));
    }
    double ext = 0;
    for (const float4& p : spos)
        ext = std::max(ext, double(std::max(std::fabs(p.x), std::max(std::fabs(p.y), std::fabs(p.z)))));
    if (ext <= 0) ext = 1;
    // (1) node data from first principles
    const double th = g.lastTheta;
    for (int i = 0; i < I; ++i) {
        const int4 in = info[i];
        double M = 0, cx = 0, cy = 0, cz = 0;
        float lx = FLT_MAX, ly = FLT_MAX, lz = FLT_MAX, hx = -FLT_MAX, hy = -FLT_MAX, hz = -FLT_MAX;
        for (int k = in.z; k <= in.w; ++k) {
            const float4 p = spos[k];
            M += p.w; cx += double(p.w) * p.x; cy += double(p.w) * p.y; cz += double(p.w) * p.z;
            lx = std::min(lx, p.x); ly = std::min(ly, p.y); lz = std::min(lz, p.z);
            hx = std::max(hx, p.x); hy = std::max(hy, p.y); hz = std::max(hz, p.z);
        }
        ++r.nodes;
        bool bad = false;
        const double mr = M > 0 ? std::fabs(com[i].w - M) / M : std::fabs(com[i].w);
        r.maxMassRel = std::max(r.maxMassRel, mr);
        if (mr > 1e-5) bad = true;
        if (lo[i].x != lx || lo[i].y != ly || lo[i].z != lz || hi[i].x != hx || hi[i].y != hy || hi[i].z != hz) bad = true;
        if (M > 0) {
            cx /= M; cy /= M; cz /= M;
            const double ce = std::max(std::fabs(com[i].x - cx), std::max(std::fabs(com[i].y - cy), std::fabs(com[i].z - cz)));
            r.maxComAbs = std::max(r.maxComAbs, ce);
            if (ce > 1e-5 * ext) bad = true;
            if (th > 0) {
                const double bx = std::max(cx - lx, hx - cx), by = std::max(cy - ly, hy - cy), bz = std::max(cz - lz, hz - cz);
                const double b64 = std::sqrt(bx * bx + by * by + bz * bz);
                const double bd  = std::sqrt(double(rc2[i])) * th;
                r.maxBmaxAbs = std::max(r.maxBmaxAbs, std::fabs(bd - b64));
                if (std::fabs(bd - b64) > 2e-5 * ext + 1e-5 * b64) bad = true;
            }
        }
        if (bad) { ++r.nodeFails; fail("node " + std::to_string(i) + ": mass/COM/AABB/bmax disagree with the FP64 recompute"); }
    }
    // (2) host walk with the device's FP32 criterion and FP64 accumulation
    const double u = 1.0 / 16777216.0;
    auto check = [&](int k, double ax, double ay, double az, double sc, unsigned nv, unsigned ni) {
        ++r.bodies;
        if (wc && ((*wc)[k].x != nv || (*wc)[k].y != ni)) {
            ++r.countMismatch;
            fail("slot " + std::to_string(k) + ": device pop/interaction counts differ from the host walk");
        }
        const float4 d = acc[perm[k]];
        const double ex = d.x - G * ax, ey = d.y - G * ay, ez = d.z - G * az;
        const double e  = std::sqrt(ex * ex + ey * ey + ez * ez) / (G * sc > 0 ? G * sc : 1.0);
        const double bound = (double(ni) + 10.0) * u;
        r.maxErrOverBound = std::max(r.maxErrOverBound, e / bound);
        if (e > bound) { ++r.errFails; fail("slot " + std::to_string(k) + ": |a_dev - a_host| / sum|f| above (I + 10) u"); }
    };
    auto term = [&](const float4& bi, const float4& bj, double& ax, double& ay, double& az, double& sc) {
        const double rx = double(bj.x) - bi.x, ry = double(bj.y) - bi.y, rz = double(bj.z) - bi.z;
        const double d2 = rx * rx + ry * ry + rz * rz + double(eps2);
        const double s  = double(bj.w) / (d2 * std::sqrt(d2));
        ax += rx * s; ay += ry * s; az += rz * s;
        sc += std::sqrt(rx * rx + ry * ry + rz * rz) * s;
    };
    auto accepts = [&](const float4& bi, int node) {
        const float4 c  = com[node];
        const float  dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
        const float  d2 = std::fmaf(dx, dx, std::fmaf(dy, dy, dz * dz));
        return d2 >= rc2[node];
    };
    const int leaf = g.lastLeaf;
    const int W    = unionWalk ? 32 : 1;
    for (int base = 0; base < n; base += W) {
        const int m = std::min(W, n - base);
        std::vector<double> ax(m, 0), ay(m, 0), az(m, 0), sc(m, 0);
        unsigned nv = 0, ni = 0;   // per lane (identical for all lanes of a union walk)
        std::vector<int> st{n > 1 ? 0 : ~0};
        while (!st.empty()) {
            const int node = st.back(); st.pop_back();
            ++nv;
            auto all = [&](const float4& bj) {
                for (int l = 0; l < m; ++l) term(spos[base + l], bj, ax[l], ay[l], az[l], sc[l]);
                ++ni;
            };
            if (node < 0) { all(spos[~node]); continue; }
            bool open = false;
            for (int l = 0; l < m && !open; ++l) open = !accepts(spos[base + l], node);
            if (!open) { all(com[node]); continue; }
            const int4 in = info[node];
            if (in.w - in.z + 1 <= leaf) { for (int k = in.z; k <= in.w; ++k) all(spos[k]); }
            else { st.push_back(in.y); st.push_back(in.x); }
        }
        for (int l = 0; l < m; ++l) check(base + l, ax[l], ay[l], az[l], sc[l], nv, ni);
    }
    return r;
}

void registerTree(std::vector<KernelInfo>& r)
{
    r.push_back({"bh_naive",
                 "Barnes-Hut, 1 thread/body, ORIGINAL body order, no warp cooperation (--theta, --leaf) (R3 #4)",
                 launchV<Var::Thread, Order::Identity>, queryBH});
    r.push_back({"bh_sorted",
                 "Barnes-Hut, same traversal, threads in Morton order (R3 #5)", launchV<Var::Thread, Order::Morton>, queryBH});
    r.push_back({"bh_hilbert",
                 "Barnes-Hut, same traversal, threads in Hilbert order (R3 #5, sub-variant)",
                 launchV<Var::Thread, Order::Hilbert>, queryBH});
    // ---- tree round 2 (docs/03-design.md §17)
    r.push_back({"bh_warp",
                 "Barnes-Hut, warp-cooperative union walk (warp vote, register stack), Morton warps (R3 #7)",
                 launchV<Var::Warp, Order::Morton>, queryWarp});
    r.push_back({"bh_warp_list",
                 "bh_warp + per-warp shared-memory interaction lists, dense evaluation (R3 #7, sub-variant)",
                 launchV<Var::WarpList, Order::Morton>, queryWarpList});
    r.push_back({"bh_warp_naive",
                 "bh_warp on ORIGINAL-order warps (R3 #7 without #5; union of 32 random bodies)",
                 launchV<Var::Warp, Order::Identity>, queryWarpNaive});
    r.push_back({"bh_persistent",
                 "persistent blocks, warp-granular atomic queue of 32-body units, #5 walk (R3 #6)",
                 launchV<Var::PersUnit, Order::Morton>, queryPersUnit});
    r.push_back({"bh_persistent_lane",
                 "persistent threads, lane-granular refill from an atomic queue, #5 walk (R3 #6)",
                 launchV<Var::PersLane, Order::Morton>, queryPersLane});
    r.push_back({"bh_persistent_lane_naive",
                 "bh_persistent_lane on ORIGINAL body order (R3 #6 on #4)",
                 launchV<Var::PersLane, Order::Identity>, queryPersLaneNaive});
    r.push_back({"bh_persistent_warp",
                 "persistent blocks, warp-granular atomic queue, #7 union walk (R3 #6 + #7)",
                 launchV<Var::PersWarp, Order::Morton>, queryPersWarp});
    r.push_back({"bh_persistent_static",
                 "persistent blocks, static partition balanced on the previous launch's unit costs, #5 walk (R3 #6, static arm)",
                 launchV<Var::PersStatic, Order::Morton>, queryPersStatic});
    r.push_back({"bh_coherent",
                 "CONTROL: #5 kernel with order[t] = perm[t & ~31] (all 32 lanes walk one body; forces of 31/32 bodies not computed)",
                 launchV<Var::Thread, Order::Coherent>, queryCoherent});
}

} // namespace nbody
