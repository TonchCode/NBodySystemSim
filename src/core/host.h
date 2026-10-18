// R3 #8 (docs/03-design.md §23): host-side building blocks shared by the CPU baselines
// (host_kernels.cpp) and the CPU+GPU splits (kernels_hybrid.cu, kernels_tree.cu).
// Internal: not part of the public API in nbody.h.
#pragma once
#include <functional>
#include <vector>
#include "nbody.h"

namespace nbody {
namespace host {

double nowMs();   // steady clock, ms

// ---- thread pool: one worker per logical processor of the chosen set, hard-pinned -------------------
// configurePool() (re)creates the workers when (set, threads) changes. The calling thread does not
// compute; it dispatches and waits (short spin, then blocking), so a worker never shares its LP with
// the dispatcher's spin for long.
void configurePool(int set, int threads);
int  poolWorkers();
bool poolWorkerIsP(int w);

// Schedules (Params::cpuSched): 0 static even split, 1 dynamic (atomic counter, chunk items per claim,
// 0 = auto), 2 static weighted by each worker's speed (items per ms, EMA) measured on the previous
// regions with the same site id. fn(begin, end, worker) is called once per contiguous item range.
enum Sched { kStatic = 0, kDynamic = 1, kWeighted = 2 };
enum Site {
    kSiteAllPairs = 1, kSiteBbox, kSiteKeys, kSiteSortHist, kSiteSortScatter, kSiteGather, kSiteRadix,
    kSiteMultipole, kSiteWalkGroup, kSiteWalkScalar, kSiteSoA, kSiteCount
};
void parallelFor(long long n, int sched, long long chunk, int site, CpuStats* stats,
                 const std::function<void(long long, long long, int)>& fn);
// Configure from Params (set, threads) and run with Params' schedule.
void parallelFor(const Params& p, long long n, int site, const std::function<void(long long, long long, int)>& fn);
// _mm256_rsqrt_ps of `samples` inputs on a P-core LP and on an E-core LP: number of results whose bits differ
// (-1 if the CPU has only one core type). An observation for the selftest (reproducibility of the CPU path).
int rsqrtCoreTypeMismatches(int samples);

// ---- CPU all-pairs (AVX2 + FMA, FP32) ------------------------------------------------------------------
// out[i - i0] = G * sum_j m_j r_ij / (r_ij^2 + eps^2)^1.5 for bodies i in [i0, i1), sources j = 0..n-1
// in index order (the allpairs_naive order, per body). Params: cpuRsqrt, cpuIV, cpuTile, schedule.
void allPairs(const float4* pos, int n, int i0, int i1, const Params& p, float4* out);

// ---- host Barnes-Hut tree: the GPU algorithm (§11) in the GPU's FP32 arithmetic ------------------------
// Same keys (bounding cube, 63-bit Morton), the same stable radix sort (ties by original index), the
// same binary radix tree and the same bottom-up multipoles with the GPU's FMA contractions, so on the
// same positions it is the GPU tree bit for bit (checked by --selftest).
struct Tree {
    int n = 0;
    float theta = 0.5f;
    int leaf = 16;
    std::vector<unsigned long long> key;   // sorted keys
    std::vector<int>    perm;              // slot -> body
    std::vector<float4> sortedPos;
    std::vector<float4> com, lo, hi;       // n - 1 internal nodes
    std::vector<float>  rc2;
    std::vector<int4>   info;              // left, right, first, last
    std::vector<int>    leafParent, nodeParent;
};
// full = false: topology only (keys, sort, radix tree, parents; no gather, no multipoles).
// phaseMs (optional, >= 3): ADDS keys (bbox + keys), sort (+ gather), tree (radix + multipoles).
void buildTree(const float4* pos, int n, const Params& p, Tree& t, bool full, double* phaseMs);

// A tree in the GPU layout (host copies of the device arrays, or a host Tree).
struct TreeArrays {
    int n = 0;
    const float4* sortedPos = nullptr;
    const float4* com = nullptr;
    const float*  rc2 = nullptr;
    const int4*   info = nullptr;
    int leaf = 16;
};
inline TreeArrays arrays(const Tree& t)
{
    TreeArrays a;
    a.n = t.n; a.sortedPos = t.sortedPos.data(); a.com = t.com.data(); a.rc2 = t.rc2.data(); a.info = t.info.data();
    a.leaf = t.leaf;
    return a;
}
// Walks over thread slots [s0, s1) (Morton slots; body of slot s = bodyOf[s]).
//   walkGroups: the CPU analogue of #7: Params::cpuGroup consecutive slots walk the union of their bmax
//               decisions together, building an interaction list that is evaluated 8 bodies per AVX2
//               instruction. Group 32 = the #7 warp: the same per-slot pop and interaction counts and
//               the same per-body summation order as bh_warp.
//   walkScalar: the #5 per-body walk (bhTraverseKernel statement for statement), scalar arithmetic.
// out[bodyOf ? bodyOf[s] : s - s0] = G a. wc (optional): wc[s - s0] = (pops, interactions, 0, 0).
void walkGroups(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc);
void walkScalar(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc);
int  overflows(bool reset);
const Tree& lastHostTree();   // the tree of the most recent cpu_bh / cpu_bh_scalar launch

// Pinned (page-locked) host buffer, grown on demand (cudaHostAlloc).
template <class T>
struct Pinned {
    T*     p   = nullptr;
    size_t cap = 0;
    T* ensure(size_t n)
    {
        if (n <= cap) return p;
        if (p) CUDA_CHECK(cudaFreeHost(p));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p), n * sizeof(T), cudaHostAllocDefault));
        cap = n;
        return p;
    }
};

} // namespace host
} // namespace nbody
