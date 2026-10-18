// R3 #8 (docs/03-design.md §23): honest CPU baselines on the i5-13600K (6 P-cores with HT + 8 E-cores).
//
//  cpu_allpairs   all-pairs, FP32, AVX2 + FMA, 8 bodies per instruction, 8 * cpuIV bodies held in
//                 registers per block, sources broadcast from the AoS array in j order (the
//                 allpairs_naive summation order, per body), j-tiled so that a tile is reused by 8 blocks
//                 from L1/L2. 1/sqrt: _mm256_rsqrt_ps + one Newton step (cpuRsqrt = 1, production) or
//                 sqrt + div (cpuRsqrt = 0, correctly rounded: the exact lattice selftest).
//  cpu_bh         Barnes-Hut with the GPU's algorithm and arithmetic: the same keys, stable radix sort,
//                 binary radix tree and multipoles (bitwise the GPU tree), and the CPU analogue of R3 #7:
//                 a group of cpuGroup Morton-consecutive bodies walks the union of its bmax decisions and
//                 evaluates the resulting interaction list 8 bodies per AVX2 instruction. Group 32 = the
//                 #7 warp: identical per-slot work counts and per-body summation order as bh_warp.
//  cpu_bh_scalar  the same tree, the #5 per-body walk in scalar arithmetic (counts = bh_sorted's).
//
// All three are threaded over the pinned host pool (host_pool.cpp) with the schedule in Params
// (static / dynamic / weighted). The bench's data lives on the GPU, so each launch copies positions
// D2H and accelerations H2D; the copies are timed separately (hostPhases[6], [7]) and are not part of
// the CPU compute time quoted as the baseline.
//
// Compiled with /arch:AVX2 (CMakeLists.txt). MSVC /fp:precise does not contract a*b+c into FMA, so
// every FMA below is explicit and placed where nvcc places it on the GPU (checked against the SASS of
// multipoleKernel and bitwise against the GPU tree by --selftest).
#define NOMINMAX
#include <immintrin.h>
#include <intrin.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>
#include "host.h"
#include "internal.h"

namespace nbody {
namespace host {
namespace {

// ---- arithmetic ---------------------------------------------------------------------------------------------
template <bool NR>
inline __m256 invSqrt8(__m256 d2)
{
    if constexpr (NR) {
        // y1 = y0 (1.5 - 0.5 d2 y0^2): one Newton step on the ~12-bit estimate (rel. error ~1.5 e0^2, always
        // from below, plus rounding); 4 extra instructions instead of a 2-unit sqrt + div.
        const __m256 y0 = _mm256_rsqrt_ps(d2);
        const __m256 hx = _mm256_mul_ps(d2, _mm256_set1_ps(0.5f));
        const __m256 yy = _mm256_mul_ps(y0, y0);
        return _mm256_mul_ps(y0, _mm256_fnmadd_ps(hx, yy, _mm256_set1_ps(1.5f)));
    } else {
        return _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_sqrt_ps(d2));
    }
}

// bodyBodyInteraction() (interaction.cuh) for 8 bodies i against one source j, the same operation order.
template <bool NR>
inline void interact8(__m256 xi, __m256 yi, __m256 zi, __m256 xj, __m256 yj, __m256 zj, __m256 mj, __m256 eps2,
                      __m256& ax, __m256& ay, __m256& az)
{
    const __m256 rx  = _mm256_sub_ps(xj, xi);
    const __m256 ry  = _mm256_sub_ps(yj, yi);
    const __m256 rz  = _mm256_sub_ps(zj, zi);
    const __m256 d2  = _mm256_fmadd_ps(rx, rx, _mm256_fmadd_ps(ry, ry, _mm256_fmadd_ps(rz, rz, eps2)));
    const __m256 inv = invSqrt8<NR>(d2);
    const __m256 s   = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(mj, inv), inv), inv);
    ax = _mm256_fmadd_ps(rx, s, ax);
    ay = _mm256_fmadd_ps(ry, s, ay);
    az = _mm256_fmadd_ps(rz, s, az);
}

inline float fma1(float a, float b, float c)
{
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

template <bool NR>
inline float invSqrt1(float d2)
{
    const __m128 x = _mm_set_ss(d2);
    if constexpr (NR) {
        const __m128 y0 = _mm_rsqrt_ss(x);
        const __m128 hx = _mm_mul_ss(x, _mm_set_ss(0.5f));
        const __m128 yy = _mm_mul_ss(y0, y0);
        return _mm_cvtss_f32(_mm_mul_ss(y0, _mm_fnmadd_ss(hx, yy, _mm_set_ss(1.5f))));
    } else {
        return _mm_cvtss_f32(_mm_div_ss(_mm_set_ss(1.f), _mm_sqrt_ss(x)));
    }
}

template <bool NR>
inline void interact1(const float4& bi, const float4& bj, float eps2, float& ax, float& ay, float& az)
{
    const float rx = bj.x - bi.x, ry = bj.y - bi.y, rz = bj.z - bi.z;
    const float d2  = fma1(rx, rx, fma1(ry, ry, fma1(rz, rz, eps2)));
    const float inv = invSqrt1<NR>(d2);
    const float s   = bj.w * inv * inv * inv;
    ax = fma1(rx, s, ax);
    ay = fma1(ry, s, ay);
    az = fma1(rz, s, az);
}

inline float fminf_(float a, float b) { return b < a ? b : a; }   // = fminf for non-NaN inputs
inline float fmaxf_(float a, float b) { return b > a ? b : a; }

// ---- all-pairs ----------------------------------------------------------------------------------------------
constexpr int kAPBlocks = 8;   // register blocks sharing one j-tile (cpu_allpairs)

template <bool NR, int IV>
void allPairsT(const float4* pos, int n, int i0, int i1, const Params& p, float4* out)
{
    constexpr int B = 8 * IV;   // bodies per register block
    constexpr int K = kAPBlocks;
    const int ni = i1 - i0;
    if (ni <= 0) return;
    const int nb = (ni + B - 1) / B;
    static std::vector<float> X, Y, Z;
    X.assign(size_t(nb) * B, 0.f); Y.assign(size_t(nb) * B, 0.f); Z.assign(size_t(nb) * B, 0.f);
    for (int i = 0; i < ni; ++i) { X[size_t(i)] = pos[i0 + i].x; Y[size_t(i)] = pos[i0 + i].y; Z[size_t(i)] = pos[i0 + i].z; }
    const float  eps2s = p.softening * p.softening;
    const float  G     = p.G;
    const int    tile  = std::max(64, std::min(p.cpuTile > 0 ? p.cpuTile : 2048, std::max(64, n)));
    parallelFor(p, nb, kSiteAllPairs, [&](long long b0, long long b1, int) {
        const __m256 eps2 = _mm256_set1_ps(eps2s);
        alignas(32) float acc[kAPBlocks][IV][3][8];
        for (long long bb = b0; bb < b1; bb += K) {
            const int kk = int(std::min<long long>(K, b1 - bb));
            std::memset(acc, 0, sizeof acc);
            for (int j0 = 0; j0 < n; j0 += tile) {
                const int j1 = std::min(n, j0 + tile);
                for (int k = 0; k < kk; ++k) {
                    const size_t base = size_t(bb + k) * B;
                    __m256 xi[IV], yi[IV], zi[IV], ax[IV], ay[IV], az[IV];
                    for (int v = 0; v < IV; ++v) {
                        xi[v] = _mm256_loadu_ps(&X[base + 8 * v]);
                        yi[v] = _mm256_loadu_ps(&Y[base + 8 * v]);
                        zi[v] = _mm256_loadu_ps(&Z[base + 8 * v]);
                        ax[v] = _mm256_load_ps(acc[k][v][0]);
                        ay[v] = _mm256_load_ps(acc[k][v][1]);
                        az[v] = _mm256_load_ps(acc[k][v][2]);
                    }
                    for (int j = j0; j < j1; ++j) {
                        const float* q = &pos[j].x;
                        const __m256 xj = _mm256_broadcast_ss(q), yj = _mm256_broadcast_ss(q + 1),
                                     zj = _mm256_broadcast_ss(q + 2), mj = _mm256_broadcast_ss(q + 3);
                        for (int v = 0; v < IV; ++v) interact8<NR>(xi[v], yi[v], zi[v], xj, yj, zj, mj, eps2, ax[v], ay[v], az[v]);
                    }
                    for (int v = 0; v < IV; ++v) {
                        _mm256_store_ps(acc[k][v][0], ax[v]);
                        _mm256_store_ps(acc[k][v][1], ay[v]);
                        _mm256_store_ps(acc[k][v][2], az[v]);
                    }
                }
            }
            for (int k = 0; k < kk; ++k)
                for (int v = 0; v < IV; ++v)
                    for (int l = 0; l < 8; ++l) {
                        const long long i = (bb + k) * B + 8 * v + l;
                        if (i < ni) out[i] = make_float4(G * acc[k][v][0][l], G * acc[k][v][1][l], G * acc[k][v][2][l], 0.f);
                    }
        }
    });
}

// ---- keys (the GPU's bboxPartial/bboxFinal/keyKernel in the same FP32 operations) ---------------------------
inline unsigned long long expandBits3(unsigned v)   // = kernels_tree.cu expandBits3
{
    unsigned long long x = v & 0x1fffffu;
    x = (x | x << 32) & 0x001f00000000ffffull;
    x = (x | x << 16) & 0x001f0000ff0000ffull;
    x = (x | x << 8)  & 0x100f00f00f00f00full;
    x = (x | x << 4)  & 0x10c30c30c30c30c3ull;
    x = (x | x << 2)  & 0x1249249249249249ull;
    return x;
}
inline unsigned long long mortonKey(unsigned x, unsigned y, unsigned z)
{
    return (expandBits3(x) << 2) | (expandBits3(y) << 1) | expandBits3(z);
}

inline int deltaK(const unsigned long long* k, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    const unsigned long long a = k[i], b = k[j];
    if (a == b) return 64 + int(__lzcnt(unsigned(i ^ j)));
    return int(__lzcnt64(a ^ b));
}

struct Scratch {
    std::vector<unsigned long long> keyA, keyB;
    std::vector<int> idxA, idxB;
    std::vector<unsigned> hist;
    std::unique_ptr<std::atomic<int>[]> arrive;
    size_t arriveCap = 0;
};
Scratch& scratch()
{
    static Scratch s;
    return s;
}

std::atomic<int> gOverflow{0};

// Stable LSD radix sort of (key, index) pairs by the 63-bit key, 8-bit digits; digits on which no key
// differs are skipped. Chunk-based histograms make every pass schedulable statically or dynamically
// while staying stable (ties keep the original index order = CUB SortPairs' result).
void radixSortPairs(std::vector<unsigned long long>& key, std::vector<int>& idx, int n, const Params& p)
{
    Scratch& S = scratch();
    S.keyB.resize(size_t(n)); S.idxB.resize(size_t(n));
    const int W = std::max(1, cpuSetSize(p.cpuSet, p.cpuThreads));
    // chunks of >= 16384 keys: below 32768 keys the whole sort is one region (per-region dispatch ~20 us)
    const int C = std::max(1, std::min(8 * W, n / 16384));
    std::vector<unsigned long long> orW(size_t(C), 0ull), andW(size_t(C), ~0ull);
    parallelFor(p, C, kSiteSortHist, [&](long long c0, long long c1, int) {
        for (long long c = c0; c < c1; ++c) {
            unsigned long long o = 0, a = ~0ull;
            for (long long i = c * n / C; i < (c + 1) * n / C; ++i) { o |= key[size_t(i)]; a &= key[size_t(i)]; }
            orW[size_t(c)] = o; andW[size_t(c)] = a;
        }
    });
    unsigned long long o = 0, a = ~0ull;
    for (int c = 0; c < C; ++c) { o |= orW[size_t(c)]; a &= andW[size_t(c)]; }
    const unsigned long long diff = (o ^ a) & ((1ull << 63) - 1);
    S.hist.resize(size_t(C) * 256);
    unsigned long long* kin = key.data(); unsigned long long* kout = S.keyB.data();
    int* iin = idx.data(); int* iout = S.idxB.data();
    if (C == 1) {   // small n: the whole sort is one region on one worker (in cache; no per-pass dispatch)
        int passes = 0;
        parallelFor(p, 1, kSiteSortScatter, [&](long long, long long, int) {
            unsigned h[256];
            for (int d = 0; d < 8; ++d) {
                const int sh = 8 * d;
                if (((diff >> sh) & 0xffull) == 0) continue;
                std::memset(h, 0, sizeof h);
                for (int i = 0; i < n; ++i) ++h[(kin[i] >> sh) & 0xff];
                unsigned off = 0;
                for (int dg = 0; dg < 256; ++dg) { const unsigned t = h[dg]; h[dg] = off; off += t; }
                for (int i = 0; i < n; ++i) { const unsigned dst = h[(kin[i] >> sh) & 0xff]++; kout[dst] = kin[i]; iout[dst] = iin[i]; }
                std::swap(kin, kout); std::swap(iin, iout);
                ++passes;
            }
        });
        if (passes & 1) { key.swap(S.keyB); idx.swap(S.idxB); }
        return;
    }
    for (int d = 0; d < 8; ++d) {
        const int sh = 8 * d;
        if (((diff >> sh) & 0xffull) == 0) continue;
        std::fill(S.hist.begin(), S.hist.end(), 0u);
        parallelFor(p, C, kSiteSortHist, [&](long long c0, long long c1, int) {
            for (long long c = c0; c < c1; ++c) {
                unsigned* h = &S.hist[size_t(c) * 256];
                for (long long i = c * n / C; i < (c + 1) * n / C; ++i) ++h[(kin[i] >> sh) & 0xff];
            }
        });
        unsigned off = 0;
        for (int dg = 0; dg < 256; ++dg)
            for (int c = 0; c < C; ++c) { unsigned& h = S.hist[size_t(c) * 256 + dg]; const unsigned t = h; h = off; off += t; }
        parallelFor(p, C, kSiteSortScatter, [&](long long c0, long long c1, int) {
            for (long long c = c0; c < c1; ++c) {
                unsigned* h = &S.hist[size_t(c) * 256];
                for (long long i = c * n / C; i < (c + 1) * n / C; ++i) {
                    const unsigned dst = h[(kin[i] >> sh) & 0xff]++;
                    kout[dst] = kin[i]; iout[dst] = iin[i];
                }
            }
        });
        std::swap(kin, kout); std::swap(iin, iout);
    }
    if (kin != key.data()) {   // odd number of passes: the result is in the scratch buffers
        key.swap(S.keyB); idx.swap(S.idxB);
    }
}

struct ChildData { float lx, ly, lz, hx, hy, hz, cx, cy, cz, m; };

inline ChildData loadChild(const Tree& t, int c)
{
    ChildData d;
    if (c < 0) {
        const float4 q = t.sortedPos[size_t(~c)];
        d.lx = d.hx = d.cx = q.x; d.ly = d.hy = d.cy = q.y; d.lz = d.hz = d.cz = q.z; d.m = q.w;
    } else {
        const float4 a = t.com[size_t(c)], l = t.lo[size_t(c)], h = t.hi[size_t(c)];
        d.cx = a.x; d.cy = a.y; d.cz = a.z; d.m = a.w;
        d.lx = l.x; d.ly = l.y; d.lz = l.z; d.hx = h.x; d.hy = h.y; d.hz = h.z;
    }
    return d;
}

// One node of multipoleKernel (kernels_tree.cu) in the GPU's operations. The FMA placement follows the
// SASS of multipoleKernel (CUDA 13.4, sm_89): com = fma(L.m, L.com, R.m * R.com) * (1 / M) and
// b2 = fma(bz, bz, fma(bx, bx, by * by)); 1 / M is IEEE-rounded on both sides.
inline void buildNode(Tree& t, int node, float invT2)
{
    const int4 in = t.info[size_t(node)];
    const ChildData L = loadChild(t, in.x), R = loadChild(t, in.y);
    const float lx = fminf_(L.lx, R.lx), ly = fminf_(L.ly, R.ly), lz = fminf_(L.lz, R.lz);
    const float hx = fmaxf_(L.hx, R.hx), hy = fmaxf_(L.hy, R.hy), hz = fmaxf_(L.hz, R.hz);
    const float M  = L.m + R.m;
    float cx, cy, cz;
    if (M > 0.f) {
        const float im = 1.f / M;
        cx = fma1(L.m, L.cx, R.m * R.cx) * im;
        cy = fma1(L.m, L.cy, R.m * R.cy) * im;
        cz = fma1(L.m, L.cz, R.m * R.cz) * im;
    } else {
        cx = 0.5f * (lx + hx); cy = 0.5f * (ly + hy); cz = 0.5f * (lz + hz);
    }
    cx = fminf_(fmaxf_(cx, lx), hx); cy = fminf_(fmaxf_(cy, ly), hy); cz = fminf_(fmaxf_(cz, lz), hz);
    const float bx = fmaxf_(cx - lx, hx - cx), by = fmaxf_(cy - ly, hy - cy), bz = fmaxf_(cz - lz, hz - cz);
    const float b2 = fma1(bz, bz, fma1(bx, bx, by * by));
    t.com[size_t(node)] = make_float4(cx, cy, cz, M);
    t.lo[size_t(node)]  = make_float4(lx, ly, lz, 0.f);
    t.hi[size_t(node)]  = make_float4(hx, hy, hz, 0.f);
    unsigned infBits = 0x7f800000u;
    float inf;
    std::memcpy(&inf, &infBits, 4);
    t.rc2[size_t(node)] = invT2 > 0.f ? b2 * invT2 : inf;
}

} // namespace

int overflows(bool reset)
{
    const int v = gOverflow.load();
    if (reset) gOverflow.store(0);
    return v;
}

void allPairs(const float4* pos, int n, int i0, int i1, const Params& p, float4* out)
{
    const bool nr = p.cpuRsqrt != 0;
    if (p.cpuIV == 1) { if (nr) allPairsT<true, 1>(pos, n, i0, i1, p, out); else allPairsT<false, 1>(pos, n, i0, i1, p, out); }
    else              { if (nr) allPairsT<true, 2>(pos, n, i0, i1, p, out); else allPairsT<false, 2>(pos, n, i0, i1, p, out); }
}

void buildTree(const float4* pos, int n, const Params& p, Tree& t, bool full, double* phaseMs)
{
    Scratch& S = scratch();
    t.n = n; t.theta = p.theta; t.leaf = p.leafSize;
    double tA = nowMs();
    // 1. bounding cube + keys (bboxPartialKernel, bboxFinalKernel, keyKernel<false>)
    const int W = std::max(1, cpuSetSize(p.cpuSet, p.cpuThreads));
    const int C = std::max(1, std::min(8 * W, n / 4096 + 1));
    std::vector<float> part(size_t(C) * 6);
    parallelFor(p, C, kSiteBbox, [&](long long c0, long long c1, int) {
        for (long long c = c0; c < c1; ++c) {
            float l0 = FLT_MAX, l1 = FLT_MAX, l2 = FLT_MAX, h0 = -FLT_MAX, h1 = -FLT_MAX, h2 = -FLT_MAX;
            for (long long i = c * n / C; i < (c + 1) * n / C; ++i) {
                const float4 q = pos[i];
                l0 = fminf_(l0, q.x); l1 = fminf_(l1, q.y); l2 = fminf_(l2, q.z);
                h0 = fmaxf_(h0, q.x); h1 = fmaxf_(h1, q.y); h2 = fmaxf_(h2, q.z);
            }
            float* r = &part[size_t(c) * 6];
            r[0] = l0; r[1] = l1; r[2] = l2; r[3] = h0; r[4] = h1; r[5] = h2;
        }
    });
    float lx = FLT_MAX, ly = FLT_MAX, lz = FLT_MAX, hx = -FLT_MAX, hy = -FLT_MAX, hz = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        const float* r = &part[size_t(c) * 6];
        lx = fminf_(lx, r[0]); ly = fminf_(ly, r[1]); lz = fminf_(lz, r[2]);
        hx = fmaxf_(hx, r[3]); hy = fmaxf_(hy, r[4]); hz = fmaxf_(hz, r[5]);
    }
    float side = fmaxf_(hx - lx, fmaxf_(hy - ly, hz - lz));
    side = side > 0.f ? side * 1.0001f : 1.f;
    const float inv = 1.f / side;
    const float sc  = inv * 2097152.f;
    S.keyA.resize(size_t(n)); S.idxA.resize(size_t(n));
    parallelFor(p, n, kSiteKeys, [&](long long b, long long e, int) {
        for (long long i = b; i < e; ++i) {
            const float4 q = pos[i];
            auto qz = [&](float v, float lo) {
                float c = (v - lo) * sc;
                c = fminf_(fmaxf_(c, 0.f), 2097151.f);
                return unsigned(c);
            };
            S.keyA[size_t(i)] = mortonKey(qz(q.x, lx), qz(q.y, ly), qz(q.z, lz));
            S.idxA[size_t(i)] = int(i);
        }
    });
    double tB = nowMs();
    // 2. sort (+ gather)
    radixSortPairs(S.keyA, S.idxA, n, p);
    t.key.swap(S.keyA);
    t.perm.swap(S.idxA);
    if (full) {
        t.sortedPos.resize(size_t(n));
        parallelFor(p, n, kSiteGather, [&](long long b, long long e, int) {
            for (long long k = b; k < e; ++k) t.sortedPos[size_t(k)] = pos[t.perm[size_t(k)]];
        });
    }
    double tC = nowMs();
    // 3. radix tree + multipoles
    const int I = n > 1 ? n - 1 : 0;
    t.info.resize(size_t(std::max(1, I))); t.leafParent.resize(size_t(n)); t.nodeParent.resize(size_t(std::max(1, I)));
    if (I > 0) {
        const unsigned long long* k = t.key.data();
        parallelFor(p, I, kSiteRadix, [&](long long b, long long e, int) {
            for (long long ii = b; ii < e; ++ii) {
                const int i = int(ii);
                const int d    = (deltaK(k, n, i, i + 1) - deltaK(k, n, i, i - 1)) >= 0 ? 1 : -1;
                const int dmin = deltaK(k, n, i, i - d);
                int lmax = 2;
                while (deltaK(k, n, i, i + lmax * d) > dmin) lmax <<= 1;
                int l = 0;
                for (int s = lmax >> 1; s >= 1; s >>= 1)
                    if (deltaK(k, n, i, i + (l + s) * d) > dmin) l += s;
                const int j     = i + l * d;
                const int dnode = deltaK(k, n, i, j);
                int s = 0, tt = l;
                do {
                    tt = (tt + 1) >> 1;
                    if (deltaK(k, n, i, i + (s + tt) * d) > dnode) s += tt;
                } while (tt > 1);
                const int gamma = i + s * d + (d < 0 ? -1 : 0);
                const int first = std::min(i, j), last = std::max(i, j);
                const int left  = (first == gamma) ? ~gamma : gamma;
                const int right = (last == gamma + 1) ? ~(gamma + 1) : gamma + 1;
                t.info[size_t(i)] = make_int4(left, right, first, last);
                if (left < 0) t.leafParent[size_t(~left)] = i; else t.nodeParent[size_t(left)] = i;
                if (right < 0) t.leafParent[size_t(~right)] = i; else t.nodeParent[size_t(right)] = i;
                if (i == 0) t.nodeParent[0] = -1;
            }
        });
    }
    if (full && I > 0) {
        t.com.resize(size_t(I)); t.lo.resize(size_t(I)); t.hi.resize(size_t(I)); t.rc2.resize(size_t(I));
        if (S.arriveCap < size_t(I)) { S.arrive.reset(new std::atomic<int>[size_t(I)]); S.arriveCap = size_t(I); }
        std::atomic<int>* arr = S.arrive.get();
        parallelFor(p, I, kSiteMultipole, [&](long long b, long long e, int) {
            for (long long i = b; i < e; ++i) arr[i].store(0, std::memory_order_relaxed);
        });
        const float invT2 = p.theta > 0.f ? 1.f / (p.theta * p.theta) : 0.f;
        // bottom-up: one task per leaf; the second child to arrive at a node builds it (multipoleKernel).
        // acq_rel on the arrival counter orders the first child's node writes before the builder's reads.
        parallelFor(p, n, kSiteMultipole, [&](long long b, long long e, int) {
            for (long long kk = b; kk < e; ++kk) {
                int node = t.leafParent[size_t(kk)];
                while (node >= 0) {
                    if (arr[node].fetch_add(1, std::memory_order_acq_rel) == 0) break;
                    buildNode(t, node, invT2);
                    node = t.nodeParent[size_t(node)];
                }
            }
        });
    }
    double tD = nowMs();
    if (phaseMs) { phaseMs[0] += tB - tA; phaseMs[1] += tC - tB; phaseMs[2] += tD - tC; }
}

// ---- walks ----------------------------------------------------------------------------------------------------
namespace {

constexpr int kHostStack = 160;   // >= 95 (the provable bound of docs/03-design.md §19.2)
constexpr int kHostList  = 512;   // interaction-list entries per group (8 KB)

template <bool NR, int V, bool COUNT>
void walkGroupT(const TreeArrays& t, long long s0, int m, float eps2s, float G, float4* out, const int* bodyOf,
                long long outBase, uint4* wc)
{
    __m256 xi[V], yi[V], zi[V], ax[V], ay[V], az[V], valid[V];
    alignas(32) float bx[8 * V], by[8 * V], bz[8 * V], bv[8 * V];
    for (int l = 0; l < 8 * V; ++l) {
        const bool ok = l < m;
        const float4 q = ok ? t.sortedPos[s0 + l] : make_float4(0.f, 0.f, 0.f, 0.f);
        bx[l] = q.x; by[l] = q.y; bz[l] = q.z;
        unsigned bits = ok ? 0xffffffffu : 0u;
        std::memcpy(&bv[l], &bits, 4);
    }
    for (int v = 0; v < V; ++v) {
        xi[v] = _mm256_load_ps(bx + 8 * v); yi[v] = _mm256_load_ps(by + 8 * v); zi[v] = _mm256_load_ps(bz + 8 * v);
        valid[v] = _mm256_load_ps(bv + 8 * v);
        ax[v] = ay[v] = az[v] = _mm256_setzero_ps();
    }
    const __m256 eps2 = _mm256_set1_ps(eps2s);
    float4 list[kHostList];
    int cnt = 0;
    unsigned nVisit = 0, nInter = 0;
    auto flush = [&]() {
        for (int j = 0; j < cnt; ++j) {
            const float* q = &list[j].x;
            const __m256 xj = _mm256_broadcast_ss(q), yj = _mm256_broadcast_ss(q + 1), zj = _mm256_broadcast_ss(q + 2),
                         mj = _mm256_broadcast_ss(q + 3);
            for (int v = 0; v < V; ++v) interact8<NR>(xi[v], yi[v], zi[v], xj, yj, zj, mj, eps2, ax[v], ay[v], az[v]);
        }
        cnt = 0;
    };
    auto push = [&](const float4& q) {
        if (cnt == kHostList) flush();
        list[cnt++] = q;
        if constexpr (COUNT) ++nInter;
    };
    int stack[kHostStack];
    int sp = 0;
    stack[sp++] = t.n > 1 ? 0 : ~0;
    while (sp > 0) {
        const int node = stack[--sp];
        if constexpr (COUNT) ++nVisit;
        if (node < 0) { push(t.sortedPos[size_t(~node)]); continue; }
        const float4 c  = t.com[size_t(node)];
        const __m256 cx = _mm256_set1_ps(c.x), cy = _mm256_set1_ps(c.y), cz = _mm256_set1_ps(c.z);
        const __m256 rc = _mm256_set1_ps(t.rc2[size_t(node)]);
        int open = 0;
        for (int v = 0; v < V; ++v) {
            // warpWalk: d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz)); open iff any valid lane has !(d2 >= rc2)
            const __m256 dx = _mm256_sub_ps(cx, xi[v]), dy = _mm256_sub_ps(cy, yi[v]), dz = _mm256_sub_ps(cz, zi[v]);
            const __m256 d2 = _mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
            open |= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(d2, rc, _CMP_NGE_UQ), valid[v]));
        }
        if (!open) { push(c); continue; }
        const int4 in = t.info[size_t(node)];
        if (in.w - in.z + 1 <= t.leaf) {
            for (int k = in.z; k <= in.w; ++k) push(t.sortedPos[size_t(k)]);
        } else {
            if (sp + 2 > kHostStack) { gOverflow.fetch_add(1); break; }
            stack[sp++] = in.y;
            stack[sp++] = in.x;
        }
    }
    flush();
    alignas(32) float rx[8 * V], ry[8 * V], rz[8 * V];
    for (int v = 0; v < V; ++v) { _mm256_store_ps(rx + 8 * v, ax[v]); _mm256_store_ps(ry + 8 * v, ay[v]); _mm256_store_ps(rz + 8 * v, az[v]); }
    for (int l = 0; l < m; ++l) {
        const long long s = s0 + l;
        out[bodyOf ? bodyOf[s] : s - outBase] = make_float4(G * rx[l], G * ry[l], G * rz[l], 0.f);
        if constexpr (COUNT) if (wc) wc[s - outBase] = make_uint4(nVisit, nInter, 0u, 0u);
    }
}

template <bool NR, int V, bool COUNT>
void walkGroupsT(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc)
{
    constexpr int GS = 8 * V;
    const long long ng = (s1 - s0 + GS - 1) / GS;
    const float eps2 = p.softening * p.softening;
    parallelFor(p, ng, kSiteWalkGroup, [&](long long g0, long long g1, int) {
        for (long long gI = g0; gI < g1; ++gI) {
            const long long a = s0 + gI * GS;
            const int m = int(std::min<long long>(GS, s1 - a));
            walkGroupT<NR, V, COUNT>(t, a, m, eps2, p.G, out, bodyOf, s0, wc);
        }
    });
}

template <bool NR, bool COUNT>
void walkScalarT(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc)
{
    const float eps2 = p.softening * p.softening;
    const float G = p.G;
    parallelFor(p, s1 - s0, kSiteWalkScalar, [&](long long b, long long e, int) {
        int stack[kHostStack];
        for (long long q = b; q < e; ++q) {
            const long long s = s0 + q;
            const float4 bi = t.sortedPos[s];
            float ax = 0.f, ay = 0.f, az = 0.f;
            unsigned nVisit = 0, nInter = 0;
            int sp = 0;
            stack[sp++] = t.n > 1 ? 0 : ~0;
            while (sp > 0) {   // bhTraverseKernel, statement for statement
                const int node = stack[--sp];
                if constexpr (COUNT) ++nVisit;
                if (node < 0) {
                    interact1<NR>(bi, t.sortedPos[size_t(~node)], eps2, ax, ay, az);
                    if constexpr (COUNT) ++nInter;
                    continue;
                }
                const float4 c  = t.com[size_t(node)];
                const float  dx = c.x - bi.x, dy = c.y - bi.y, dz = c.z - bi.z;
                const float  d2 = fma1(dx, dx, fma1(dy, dy, dz * dz));
                if (d2 >= t.rc2[size_t(node)]) {
                    interact1<NR>(bi, c, eps2, ax, ay, az);
                    if constexpr (COUNT) ++nInter;
                } else {
                    const int4 in = t.info[size_t(node)];
                    if (in.w - in.z + 1 <= t.leaf) {
                        for (int k = in.z; k <= in.w; ++k) {
                            interact1<NR>(bi, t.sortedPos[size_t(k)], eps2, ax, ay, az);
                            if constexpr (COUNT) ++nInter;
                        }
                    } else {
                        if (sp + 2 > kHostStack) { gOverflow.fetch_add(1); break; }
                        stack[sp++] = in.y;
                        stack[sp++] = in.x;
                    }
                }
            }
            out[bodyOf ? bodyOf[s] : s - s0] = make_float4(G * ax, G * ay, G * az, 0.f);
            if constexpr (COUNT) if (wc) wc[s - s0] = make_uint4(nVisit, nInter, 0u, 0u);
        }
    });
}

template <bool NR, bool COUNT>
void walkGroupsSel(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc)
{
    switch (p.cpuGroup) {
    case 8:  walkGroupsT<NR, 1, COUNT>(t, s0, s1, p, out, bodyOf, wc); return;
    case 16: walkGroupsT<NR, 2, COUNT>(t, s0, s1, p, out, bodyOf, wc); return;
    case 32: walkGroupsT<NR, 4, COUNT>(t, s0, s1, p, out, bodyOf, wc); return;
    case 64: walkGroupsT<NR, 8, COUNT>(t, s0, s1, p, out, bodyOf, wc); return;
    default: fprintf(stderr, "cpu group walk: --cpu-group must be 8, 16, 32 or 64 (got %d)\n", p.cpuGroup); exit(2);
    }
}

} // namespace

void walkGroups(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc)
{
    if (s1 <= s0) return;
    const bool nr = p.cpuRsqrt != 0;
    if (wc) { if (nr) walkGroupsSel<true, true>(t, s0, s1, p, out, bodyOf, wc); else walkGroupsSel<false, true>(t, s0, s1, p, out, bodyOf, wc); }
    else    { if (nr) walkGroupsSel<true, false>(t, s0, s1, p, out, bodyOf, wc); else walkGroupsSel<false, false>(t, s0, s1, p, out, bodyOf, wc); }
}

void walkScalar(const TreeArrays& t, int s0, int s1, const Params& p, float4* out, const int* bodyOf, uint4* wc)
{
    if (s1 <= s0) return;
    const bool nr = p.cpuRsqrt != 0;
    if (wc) { if (nr) walkScalarT<true, true>(t, s0, s1, p, out, bodyOf, wc); else walkScalarT<false, true>(t, s0, s1, p, out, bodyOf, wc); }
    else    { if (nr) walkScalarT<true, false>(t, s0, s1, p, out, bodyOf, wc); else walkScalarT<false, false>(t, s0, s1, p, out, bodyOf, wc); }
}

} // namespace host

// ---- launchers (the bench's data is on the GPU: D2H positions, compute on the host, H2D accelerations) -------
namespace {

struct HostCtx {
    host::Pinned<float4> pos, acc;
    std::vector<uint4> wc;
    host::Tree tree;
} hc;

void hostPhase(const Params& p, int k, double ms) { if (p.hostPhases) p.hostPhases[k] += ms; }

const float4* fetchPos(const DeviceState& s, const Params& p, cudaStream_t st)
{
    float4* hp = hc.pos.ensure(size_t(s.n));
    const double t0 = host::nowMs();
    CUDA_CHECK(cudaMemcpyAsync(hp, s.pos, size_t(s.n) * sizeof(float4), cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    hostPhase(p, 6, host::nowMs() - t0);
    return hp;
}

void storeAcc(const DeviceState& s, const Params& p, cudaStream_t st, const float4* ha)
{
    const double t0 = host::nowMs();
    CUDA_CHECK(cudaMemcpyAsync(s.acc, ha, size_t(s.n) * sizeof(float4), cudaMemcpyHostToDevice, st));
    if (p.workCounters && !hc.wc.empty())
        CUDA_CHECK(cudaMemcpyAsync(p.workCounters, hc.wc.data(), hc.wc.size() * sizeof(uint4), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    hostPhase(p, 7, host::nowMs() - t0);
}

void launchCpuAllPairs(const DeviceState& s, const Params& p, cudaStream_t st)
{
    const float4* hp = fetchPos(s, p, st);
    float4* ha = hc.acc.ensure(size_t(s.n));
    hc.wc.clear();
    const double t0 = host::nowMs();
    host::allPairs(hp, s.n, 0, s.n, p, ha);
    hostPhase(p, 0, host::nowMs() - t0);
    storeAcc(s, p, st, ha);
}

template <bool GROUP>
void launchCpuBH(const DeviceState& s, const Params& p, cudaStream_t st)
{
    if (!(p.theta >= 0.f) || p.theta > 1.f) { fprintf(stderr, "cpu_bh: theta must be in [0, 1]\n"); exit(2); }
    if (p.leafSize < 1) { fprintf(stderr, "cpu_bh: leaf size must be >= 1\n"); exit(2); }
    const float4* hp = fetchPos(s, p, st);
    float4* ha = hc.acc.ensure(size_t(s.n));
    double ph[3] = {0, 0, 0};
    host::buildTree(hp, s.n, p, hc.tree, true, ph);
    hc.wc.assign(p.workCounters ? size_t(s.n) : 0, make_uint4(0, 0, 0, 0));
    const double t0 = host::nowMs();
    const host::TreeArrays ta = host::arrays(hc.tree);
    if (GROUP) host::walkGroups(ta, 0, s.n, p, ha, hc.tree.perm.data(), p.workCounters ? hc.wc.data() : nullptr);
    else       host::walkScalar(ta, 0, s.n, p, ha, hc.tree.perm.data(), p.workCounters ? hc.wc.data() : nullptr);
    const double tw = host::nowMs() - t0;
    hostPhase(p, 0, ph[0]); hostPhase(p, 1, ph[1]); hostPhase(p, 2, ph[2]); hostPhase(p, 4, tw);
    storeAcc(s, p, st, ha);
}

LaunchInfo queryHostCommon(const Params& p)
{
    LaunchInfo li;
    li.unroll = -1;
    li.hostTimed  = true;
    li.cpuWorkers = cpuSetSize(p.cpuSet, p.cpuThreads);
    return li;
}
LaunchInfo queryCpuAllPairs(int, const Params& p)
{
    LaunchInfo li = queryHostCommon(p);
    li.nPhases = 1;
    li.phaseNames[0] = "compute";
    return li;
}
const char* const kTreePhaseNames[] = {"keys", "sort", "tree", "order", "traverse"};
LaunchInfo queryCpuBHCommon(const Params& p)
{
    LaunchInfo li = queryHostCommon(p);
    li.approximate = true;
    li.hostTree    = true;
    li.nPhases     = 5;
    for (int k = 0; k < 5; ++k) li.phaseNames[k] = kTreePhaseNames[k];
    return li;
}
LaunchInfo queryCpuBH(int, const Params& p)
{
    LaunchInfo li = queryCpuBHCommon(p);
    li.counterSibling = "bh_warp";
    if (p.cpuGroup == 32) li.countsSibling = "bh_warp";
    return li;
}
LaunchInfo queryCpuBHScalar(int, const Params& p)
{
    LaunchInfo li = queryCpuBHCommon(p);
    li.counterSibling = li.countsSibling = "bh_sorted";
    return li;
}

} // namespace

int hostTreeOverflows(bool reset) { return host::overflows(reset); }

TreeDepth hostLastTreeDepth()
{
    TreeDepth r;
    r.threadStack = 160; r.warpStack = 160;
    const host::Tree& t = hc.tree;
    const int n = t.n;
    if (n <= 1) { r.stackBound = 1; return r; }
    for (int k = 1; k < n; ++k) r.dupKeys += t.key[size_t(k)] == t.key[size_t(k) - 1];
    std::vector<std::pair<int, int>> st{{0, 0}};
    while (!st.empty()) {
        const auto [node, d] = st.back(); st.pop_back();
        r.maxDepth = std::max(r.maxDepth, d);
        const int4 in = t.info[size_t(node)];
        if (in.w - in.z + 1 > t.leaf) r.pushDepth = std::max(r.pushDepth, d);
        if (in.x >= 0) st.push_back({in.x, d + 1});
        if (in.y >= 0) st.push_back({in.y, d + 1});
    }
    r.stackBound = r.pushDepth + 2;
    return r;
}

namespace host {
const Tree& lastHostTree() { return hc.tree; }
}

void registerHost(std::vector<KernelInfo>& r)
{
    r.push_back({"cpu_allpairs",
                 "CPU all-pairs, AVX2+FMA FP32, pinned pool (--cpu-set/--cpu-sched/--cpu-threads; --cpu-rsqrt 1 = rsqrt+Newton) (R3 #8 baseline)",
                 launchCpuAllPairs, queryCpuAllPairs});
    r.push_back({"cpu_bh",
                 "CPU Barnes-Hut: the GPU's tree bit for bit, group-union walk (--cpu-group 32 = the #7 warp), AVX2 interaction lists (R3 #8 baseline)",
                 launchCpuBH<true>, queryCpuBH});
    r.push_back({"cpu_bh_scalar",
                 "CPU Barnes-Hut, the same tree, #5 per-body walk in scalar arithmetic (R3 #8, CPU ablation)",
                 launchCpuBH<false>, queryCpuBHScalar});
}

} // namespace nbody
