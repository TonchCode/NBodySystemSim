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
enum class Order { Identity, Morton, Hilbert };

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
    void   *sortTmp = nullptr;
    size_t  sortTmpBytes = 0;
    int     lastN = 0;
    float   lastTheta = 0.f;
    int     lastLeaf = 0;
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
    fr(g.nodeParent); fr(g.arrive); fr(g.sortTmp);
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
    if (!g.bbox) {
        CUDA_CHECK(cudaMalloc(&g.bboxPartial, 2 * kBboxBlocks * 16));
        CUDA_CHECK(cudaMalloc(&g.bbox, 2 * 16));
        CUDA_CHECK(cudaMalloc(&g.overflow, 4));
        CUDA_CHECK(cudaMemset(g.overflow, 0, 4));
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

template <bool TL>
__global__ void bhTraverseKernel(const float4* __restrict__ pos, float4* __restrict__ acc,
                                 const int* __restrict__ order, int n, const float4* __restrict__ sortedPos,
                                 const float4* __restrict__ nodeCom, const float* __restrict__ nodeRc2,
                                 const int4* __restrict__ nodeInfo, int leafSize, float eps2, float G,
                                 int* __restrict__ overflow, BlockStamp* tl, uint4* wc)
{
    if constexpr (TL) { if (tl) stampBlockStart<true>(tl); }
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
    if constexpr (TL) {
        if (tl) stampBlockEnd<true>(tl);
    }
}

// ---- launcher -------------------------------------------------------------------------------------
const char* const kPhaseNames[] = {"keys", "sort", "tree", "order", "traverse"};
constexpr int     kPhases       = 5;

void checkParams(const Params& p)
{
    if (!(p.theta >= 0.f)) { fprintf(stderr, "bh: theta must be >= 0 (got %g)\n", p.theta); exit(2); }
    if (p.leafSize < 1)    { fprintf(stderr, "bh: leaf size must be >= 1 (got %d)\n", p.leafSize); exit(2); }
    if (p.blockSize < 32 || p.blockSize > 1024 || p.blockSize % 32) {
        fprintf(stderr, "bh: block must be a multiple of 32 in [32, 1024] (got %d)\n", p.blockSize); exit(2);
    }
}

void launchBH(const DeviceState& s, const Params& p, cudaStream_t st, Order ord)
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
    const int* order = g.iota;
    if (ord == Order::Morton) order = g.perm;
    if (ord == Order::Hilbert) {
        keyKernel<true><<<gb, kBuildBlock, 0, st>>>(s.pos, g.bbox, n, g.keyIn);   // Morton keyIn no longer needed
        tmp = g.sortTmpBytes;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(g.sortTmp, tmp, g.keyIn, g.keyH, g.iota, g.hOrder, n, 0, 63, st));
        order = g.hOrder;
    }
    mark(4);
    const float eps2 = p.softening * p.softening;
    const int   grid = ceilDiv(n, p.blockSize);
    if (p.timeline || p.workCounters)
        bhTraverseKernel<true><<<grid, p.blockSize, 0, st>>>(s.pos, s.acc, order, n, g.sortedPos, g.nodeCom,
                                                            g.nodeRc2, g.nodeInfo, p.leafSize, eps2, p.G,
                                                            g.overflow, p.timeline, p.workCounters);
    else
        bhTraverseKernel<false><<<grid, p.blockSize, 0, st>>>(s.pos, s.acc, order, n, g.sortedPos, g.nodeCom,
                                                             g.nodeRc2, g.nodeInfo, p.leafSize, eps2, p.G,
                                                             g.overflow, nullptr, nullptr);
    mark(5);
    CUDA_CHECK(cudaGetLastError());
    g.lastN = n; g.lastTheta = p.theta; g.lastLeaf = p.leafSize; g.lastOrder = ord;
}

void launchNaiveBH(const DeviceState& s, const Params& p, cudaStream_t st)   { launchBH(s, p, st, Order::Identity); }
void launchSortedBH(const DeviceState& s, const Params& p, cudaStream_t st)  { launchBH(s, p, st, Order::Morton); }
void launchHilbertBH(const DeviceState& s, const Params& p, cudaStream_t st) { launchBH(s, p, st, Order::Hilbert); }

LaunchInfo queryBH(int n, const Params& p)
{
    const bool  inst = p.timeline || p.workCounters;
    const void* fn   = inst ? reinterpret_cast<const void*>(bhTraverseKernel<true>)
                            : reinterpret_cast<const void*>(bhTraverseKernel<false>);
    LaunchInfo li   = describeKernel(fn, ceilDiv(n, p.blockSize), p.blockSize, 0);
    li.instrumented = inst;
    li.approximate  = true;
    li.nPhases      = kPhases;
    for (int k = 0; k < kPhases; ++k) li.phaseNames[k] = kPhaseNames[k];
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

void registerTree(std::vector<KernelInfo>& r)
{
    r.push_back({"bh_naive",
                 "Barnes-Hut, 1 thread/body, ORIGINAL body order, no warp cooperation (--theta, --leaf) (R3 #4)",
                 launchNaiveBH, queryBH});
    r.push_back({"bh_sorted",
                 "Barnes-Hut, same traversal, threads in Morton order (R3 #5)", launchSortedBH, queryBH});
    r.push_back({"bh_hilbert",
                 "Barnes-Hut, same traversal, threads in Hilbert order (R3 #5, sub-variant)", launchHilbertBH, queryBH});
}

} // namespace nbody
