// nbody_bench: times force-kernel variants and validates them against the reference.
//
//   nbody_bench --list
//   nbody_bench --kernel allpairs_tiled --n 65536 --block 256 --runs 10 --csv results/x.csv
//   nbody_bench --kernel allpairs_tuned --n 65536 --block 128 --unroll 16      (R3 #2)
//   nbody_bench --kernel allpairs_multi --n 4096 --block 256 --q 8             (R3 #3; --q 0 = auto)
//               --prefetch 0|1    multi only: next-tile register prefetch (default 1)
//   --warmup-ms 300   time-based warm-up (recommended for small N: clocks ramp from idle)
//   --batch 20        time 20 back-to-back launches per run and report the mean (small N)
//
// Load imbalance without ncu (critic review 1, #1):
//   --timeline out.csv [--timeline-launches 3]
//       Times the production kernel and its instrumented twin interleaved (overhead), then runs
//       L instrumented launches back to back and writes one row per block:
//       launch,block,smid,t_start_ns,t_end_ns,cyc_start,cyc_end. Analyse with
//       tools/timeline_analysis.py.
//
// Correctness:
//   Validation vs allpairs_naive (all bodies) and vs an FP64 host reference on a sample
//   (--fp64-sample evenly spaced bodies + the 16 bodies with the largest FP64 kappa, chosen
//   from a GPU FP32 shortlist of 64). Gates (D-009, pre-registered) are evaluated and printed;
//   a FAIL exits with code 3 (after the CSV is written).
//   --selftest [--selftest-drop J]
//       Exact-arithmetic missed-interaction test for EVERY instantiated variant (all blocks,
//       unrolls, q, prefetch, instrumented or not) at several N: unit/dyadic masses on a dyadic
//       lattice with eps = 1, where every FP32 operation is exact, so the result must be
//       bitwise equal to the analytic sum. A dropped or duplicated j changes it.
//       --selftest-drop J zeroes body J's mass in the GPU input only (simulates a kernel that
//       drops j = J): every variant must then FAIL. Exit code 3 if any (unexpected) failure.
//
// Energy (critic review 1, #10):
//   --steps S --energy [--energy-log e.csv --energy-every K]
//       Integrates S KDK steps (one launch per kernel per step: TDR-safe) and logs
//       step,time,kinetic,potential,total,rel_drift every K steps (FP64 host energy, O(N^2)).
//
// Timing covers the force kernel only (cudaEvent), after warm-up. Environment that CUDA
// can't see (power limit, background load) is captured by tools/capture_env.ps1.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "nbody.h"

namespace {

struct Args {
    std::string kernel = "allpairs_tiled";
    std::string csv, timeline, energyLog;
    int  n = 1 << 14, block = 256, runs = 10, warmup = 3, steps = 0;
    int  q = 0, unroll = 8, prefetch = 1, fp64Sample = 512, warmupMs = 0, batch = 1;
    int  timelineLaunches = 3, energyEvery = 0, selftestDrop = -1;
    unsigned seed = 42;
    bool list = false, validate = true, energy = false, selftest = false;
    float dt = 1e-3f, softening = 1e-2f;
};

int toInt(const char* flag, const char* v)
{
    char* end = nullptr;
    long  x   = strtol(v, &end, 10);
    if (end == v || *end != '\0') { fprintf(stderr, "%s: not an integer: '%s'\n", flag, v); exit(2); }
    return int(x);
}

float toFloat(const char* flag, const char* v)
{
    char*  end = nullptr;
    double x   = strtod(v, &end);
    if (end == v || *end != '\0') { fprintf(stderr, "%s: not a number: '%s'\n", flag, v); exit(2); }
    return float(x);
}

Args parse(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); exit(2); }
            return argv[++i];
        };
        auto nexti = [&]() { const char* v = next(); return toInt(s.c_str(), v); };
        auto nextf = [&]() { const char* v = next(); return toFloat(s.c_str(), v); };
        if      (s == "--list")              a.list = true;
        else if (s == "--kernel")            a.kernel = next();
        else if (s == "--n")                 a.n = nexti();
        else if (s == "--block")             a.block = nexti();
        else if (s == "--q")                 a.q = nexti();
        else if (s == "--unroll")            a.unroll = nexti();
        else if (s == "--prefetch")          a.prefetch = nexti();
        else if (s == "--fp64-sample")       a.fp64Sample = nexti();
        else if (s == "--runs")              a.runs = std::max(1, nexti());
        else if (s == "--warmup")            a.warmup = nexti();
        else if (s == "--warmup-ms")         a.warmupMs = nexti();
        else if (s == "--batch")             a.batch = std::max(1, nexti());
        else if (s == "--seed")              a.seed = unsigned(nexti());
        else if (s == "--csv")               a.csv = next();
        else if (s == "--timeline")          a.timeline = next();
        else if (s == "--timeline-launches") a.timelineLaunches = std::max(1, nexti());
        else if (s == "--steps")             a.steps = nexti();
        else if (s == "--dt")                a.dt = nextf();
        else if (s == "--softening")         a.softening = nextf();
        else if (s == "--no-validate")       a.validate = false;
        else if (s == "--energy")            a.energy = true;
        else if (s == "--energy-log")        { a.energyLog = next(); a.energy = true; }
        else if (s == "--energy-every")      a.energyEvery = nexti();
        else if (s == "--selftest")          a.selftest = true;
        else if (s == "--selftest-drop")     { a.selftestDrop = nexti(); a.selftest = true; }
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); exit(2); }
    }
    // eps <= 0 makes the self-term 0 * rsqrt(0) = NaN (critic review 1, #12).
    if (!(a.softening > 0.f)) { fprintf(stderr, "--softening must be > 0 (got %g)\n", a.softening); exit(2); }
    if (a.n <= 0)             { fprintf(stderr, "--n must be > 0\n"); exit(2); }
    if (!(a.dt > 0.f))        { fprintf(stderr, "--dt must be > 0\n"); exit(2); }
    if (a.prefetch != 0 && a.prefetch != 1) { fprintf(stderr, "--prefetch must be 0 or 1\n"); exit(2); }
    return a;
}

void printDevice()
{
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int drv = 0, rt = 0, clk = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("# device  %s  sm_%d%d  %d SMs  boost %d MHz  CUDA driver %d runtime %d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, clk / 1000, drv, rt);
}

// Median: mean of the two middle values for an even count (critic review 1, #12).
double median(std::vector<float> v)
{
    std::sort(v.begin(), v.end());
    const size_t m = v.size();
    if (m == 0) return 0.0;
    return (m % 2) ? v[m / 2] : 0.5 * (double(v[m / 2 - 1]) + v[m / 2]);
}

constexpr double kU = 1.0 / 16777216.0;   // FP32 unit roundoff 2^-24

// ---------------------------------------------------------------------------------------
// Gates, D-009 (supersedes the D-006 thresholds). Pre-registered in
// docs/predictions/2026-10-17-round2.md before any run that tests them.
//   G1  normMax(variant vs FP64) <= kG1 * u * sqrt(N)      (absolute, conditioning-independent)
//       kG1 = 1.0 = 1.5 x the largest measured naive ratio normMax / (u sqrt N) = 0.68
//       (round-1 sanity data, N = 1000..65536, 8 values of N).
//   G2  normMax(variant) <= 1.5 x normMax(naive) on the same sample (no regression)
//   G0  sanity: max rel error vs naive <= 1e-2 (gross bugs)
// Missed interactions are caught by --selftest (exact), not by these statistical gates.
constexpr double kG1 = 1.0, kG2 = 1.5, kG0 = 1e-2;

// ---------------------------------------------------------------------------------------
// Exact missed-interaction self-test.
// Positions X_i = integers in [-16, 15] times 2^-20, masses m_j in {1, 2}, eps = 1, G = 1.
// Then |r| < 2^-15, r^2 < 2^-30, so every fmaf in d2 rounds to exactly eps^2 = 1, rsqrtf(1) =
// 1, s = m_j exactly, r * s exact, and every partial sum is an integer multiple of 2^-20
// bounded by 62 * N < 2^24 units for N <= 2^18: all FP32 arithmetic is exact in ANY order.
// So a_i = 2^-20 * (sum_j m_j X_j - M X_i) bitwise, for every variant.
struct SelftestCase { const nbody::KernelInfo* k; nbody::Params p; std::string label; };

std::vector<SelftestCase> allConfigs(int n)
{
    using namespace nbody;
    std::vector<SelftestCase> out;
    const int blocks[] = {64, 128, 256, 512, 1024};
    const int unrolls[] = {1, 4, 8, 16, 0};
    const int qs[] = {1, 2, 4, 8, 16, 32};
    for (const auto& k : kernelRegistry()) {
        std::string name = k.name;
        Params base;
        base.n = n; base.softening = 1.f; base.G = 1.f;
        char buf[128];
        if (name == "allpairs_naive") {
            base.blockSize = 256;
            out.push_back({&k, base, "naive b256"});
        } else if (name == "allpairs_tiled") {
            for (int b : blocks) {
                Params p = base; p.blockSize = b;
                snprintf(buf, sizeof buf, "tiled b%d", b);
                out.push_back({&k, p, buf});
            }
        } else if (name == "allpairs_tuned") {
            for (int b : blocks) for (int u : unrolls) {
                Params p = base; p.blockSize = b; p.unroll = u;
                snprintf(buf, sizeof buf, "tuned b%d u%d", b, u);
                out.push_back({&k, p, buf});
            }
        } else if (name == "allpairs_multi") {
            for (int b : blocks) for (int q : qs) for (int pf = 0; pf <= 1; ++pf) {
                Params p = base; p.blockSize = b; p.threadsPerBody = q; p.prefetch = pf;
                snprintf(buf, sizeof buf, "multi b%d q%d pf%d", b, q, pf);
                out.push_back({&k, p, buf});
            }
        } else {
            base.blockSize = 256;
            out.push_back({&k, base, name + " b256 (default params)"});
        }
    }
    return out;
}

int runSelftest(int dropBody, unsigned seed)
{
    using namespace nbody;
    const int ns[] = {1, 2, 31, 64, 255, 1000, 1025, 4097, 10007, 65536};
    const int kMaxN = 65536;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coord(-16, 15), mass(1, 2);
    BlockStamp* dTl = nullptr;
    CUDA_CHECK(cudaMalloc(&dTl, size_t(kMaxN) * sizeof(BlockStamp)));   // grid <= n for every variant
    std::vector<BlockStamp> hTl;
    int nCases = 0, nFail = 0, nStampFail = 0, nTrivial = 0, nExpectFail = 0, nMissed = 0;
    printf("selftest: exact dyadic lattice, eps = 1, masses {1,2}%s\n",
           dropBody >= 0 ? "  [--selftest-drop: every case is EXPECTED to fail]" : "");

    for (int n : ns) {
        std::vector<int> X(n), Y(n), Z(n), M(n);
        long long sx = 0, sy = 0, sz = 0, sm = 0;
        for (int i = 0; i < n; ++i) {
            X[i] = coord(rng); Y[i] = coord(rng); Z[i] = coord(rng); M[i] = mass(rng);
            sx += (long long)M[i] * X[i]; sy += (long long)M[i] * Y[i]; sz += (long long)M[i] * Z[i]; sm += M[i];
        }
        std::vector<float4> pos(n), vel(n, make_float4(0, 0, 0, 0)), acc;
        std::vector<float>  ex(n), ey(n), ez(n);
        const float h = std::ldexp(1.0f, -20);
        for (int i = 0; i < n; ++i) {
            pos[i] = make_float4(X[i] * h, Y[i] * h, Z[i] * h, float(M[i]));
            ex[i]  = float(std::ldexp(double(sx - sm * X[i]), -20));   // exact: |.| < 2^24 units
            ey[i]  = float(std::ldexp(double(sy - sm * Y[i]), -20));
            ez[i]  = float(std::ldexp(double(sz - sm * Z[i]), -20));
            if (ex[i] == 0.f && ey[i] == 0.f && ez[i] == 0.f) ++nTrivial;
        }
        // Drop mode: the case can only fail if dropping J changes some body's analytic answer
        // (not true for n = 1, or if every body sits on body J's lattice point).
        bool dropVisible = false;
        if (dropBody >= 0 && dropBody < n) {
            pos[dropBody].w = 0.f;   // GPU input only; ex/ey/ez keep body J
            for (int i = 0; i < n && !dropVisible; ++i)
                dropVisible = X[i] != X[dropBody] || Y[i] != Y[dropBody] || Z[i] != Z[dropBody];
        }
        DeviceState s = allocate(n);
        upload(s, pos, vel);

        for (const auto& c : allConfigs(n)) {
            for (int tl = 0; tl <= 1; ++tl) {
                Params p = c.p;
                if (tl) {
                    if (c.k->query == nullptr) continue;
                    p.timeline = dTl;
                    // Only kernels that have an instrumented instantiation report it.
                    if (!c.k->query(n, p).instrumented) continue;
                    CUDA_CHECK(cudaMemset(dTl, 0, size_t(kMaxN) * sizeof(BlockStamp)));
                }
                CUDA_CHECK(cudaMemset(s.acc, 0xFF, size_t(n) * sizeof(float4)));   // NaN poison
                c.k->launch(s, p, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
                downloadAcc(s, acc);
                int bad = 0, firstBad = -1;
                for (int i = 0; i < n; ++i) {
                    bool ok = std::memcmp(&acc[i].x, &ex[i], 4) == 0 && std::memcmp(&acc[i].y, &ey[i], 4) == 0 &&
                              std::memcmp(&acc[i].z, &ez[i], 4) == 0;
                    if (!ok) { if (firstBad < 0) firstBad = i; ++bad; }
                }
                ++nCases;
                if (dropVisible) { ++nExpectFail; if (!bad) ++nMissed; }
                if (bad) {
                    ++nFail;
                    if (dropBody < 0 || nFail <= 3)
                        printf("  FAIL n=%-6d %-22s%s  %d/%d bodies wrong; body %d got (%.9g %.9g %.9g) want (%.9g %.9g %.9g)\n",
                               n, c.label.c_str(), tl ? " +timeline" : "", bad, n, firstBad, acc[firstBad].x,
                               acc[firstBad].y, acc[firstBad].z, ex[firstBad], ey[firstBad], ez[firstBad]);
                }
                if (tl) {   // the stamps themselves: every block written once, sane times
                    const int grid = c.k->query(n, p).gridBlocks;
                    hTl.resize(grid);
                    CUDA_CHECK(cudaMemcpy(hTl.data(), dTl, grid * sizeof(BlockStamp), cudaMemcpyDeviceToHost));
                    int sb = 0;
                    for (int b = 0; b < grid; ++b)
                        if (hTl[b].block != unsigned(b) || hTl[b].t0 == 0 || hTl[b].t1 < hTl[b].t0 ||
                            hTl[b].c1 < hTl[b].c0 || hTl[b].smid >= 1024) ++sb;
                    if (sb) {
                        ++nStampFail;
                        printf("  STAMP FAIL n=%d %s: %d/%d block records invalid\n", n, c.label.c_str(), sb, grid);
                    }
                }
            }
        }
        release(s);
    }
    CUDA_CHECK(cudaFree(dTl));
    printf("selftest: %d cases, %d failed, %d timeline-record failures (%d of the analytic accelerations are 0)\n",
           nCases, nFail, nStampFail, nTrivial);
    if (dropBody >= 0) {
        const bool ok = nMissed == 0 && nFail == nExpectFail;
        printf("selftest-drop %d: %s (%d of %d cases where the drop is visible detected it; %d cases where it "
               "cannot be visible, e.g. n = 1)\n", dropBody,
               ok ? "PASS, the test detects a dropped j in every variant" : "FAIL, some variant missed it",
               nExpectFail - nMissed, nExpectFail, nCases - nExpectFail);
        return ok && nStampFail == 0 ? 0 : 3;
    }
    printf("selftest: %s\n", (nFail == 0 && nStampFail == 0) ? "PASS" : "FAIL");
    return (nFail == 0 && nStampFail == 0) ? 0 : 3;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace nbody;
    Args a = parse(argc, argv);

    if (a.list) {
        for (const auto& k : kernelRegistry()) printf("%-20s %s\n", k.name, k.description);
        return 0;
    }
    if (a.selftest) {
        printDevice();
        return runSelftest(a.selftestDrop, a.seed);
    }
    const KernelInfo* k = findKernel(a.kernel);
    if (!k) { fprintf(stderr, "unknown kernel '%s' (use --list)\n", a.kernel.c_str()); return 2; }

    printDevice();
    Params p;
    p.n = a.n; p.blockSize = a.block; p.dt = a.dt; p.softening = a.softening;
    p.unroll = a.unroll; p.threadsPerBody = a.q; p.prefetch = a.prefetch;

    // ---- launch configuration + occupancy (what will actually run) -------------
    LaunchInfo li;
    double occ = -1.0;
    cudaDeviceProp dp;
    CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
    const int sms = dp.multiProcessorCount;
    auto printLaunch = [&](const char* tag, const LaunchInfo& l) {
        const double o = double(l.blocksPerSM) * l.blockThreads / dp.maxThreadsPerMultiProcessor;
        const int slots = l.blocksPerSM * sms;
        printf("# %s grid=%d block=%d q=%d prefetch=%d unroll=%d  regs=%d smem=%zu B  "
               "resident %d blocks/SM (%.0f%% occupancy)  blocks/SM=%.2f (busiest %d)  waves=%.2f\n",
               tag, l.gridBlocks, l.blockThreads, l.threadsPerBody, l.prefetch, l.unroll, l.regsPerThread,
               l.smemPerBlock, l.blocksPerSM, 100.0 * o, double(l.gridBlocks) / sms, (l.gridBlocks + sms - 1) / sms,
               slots > 0 ? double(l.gridBlocks) / slots : 0.0);
        return o;
    };
    if (k->query) {
        li  = k->query(p.n, p);
        occ = printLaunch("launch ", li);
    }

    // ---- instrumented twin (timeline) ---------------------------------------
    const bool wantTl = !a.timeline.empty();
    BlockStamp* dTl = nullptr;
    Params pTl = p;
    LaunchInfo liTl;
    if (wantTl) {
        if (!k->query) { fprintf(stderr, "--timeline: kernel %s has no query\n", k->name); return 2; }
        CUDA_CHECK(cudaMalloc(&dTl, size_t(li.gridBlocks) * a.timelineLaunches * sizeof(BlockStamp)));
        pTl.timeline = dTl;
        liTl = k->query(p.n, pTl);
        if (!liTl.instrumented) { fprintf(stderr, "--timeline: kernel %s has no instrumented build\n", k->name); return 2; }
        printLaunch("timeline", liTl);
        if (liTl.blocksPerSM != li.blocksPerSM)
            printf("# WARNING: instrumented build has a different occupancy (%d vs %d blocks/SM)\n",
                   liTl.blocksPerSM, li.blocksPerSM);
    }

    std::vector<float4> pos, vel, acc, ref;
    makePlummer(p.n, a.seed, pos, vel);
    DeviceState s = allocate(p.n);
    upload(s, pos, vel);

    // ---- timing ------------------------------------------------------------
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < a.warmup; ++i) k->launch(s, p, 0);
    // Time-based warm-up: an idle GPU sits at a low clock state and needs tens of ms of
    // load to reach boost. A few us-long launches do not get it there, so small-N timings
    // are bimodal without this (results/2026-10-17_architect_sanity).
    if (a.warmupMs > 0) {
        auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() < a.warmupMs) {
            for (int i = 0; i < 20; ++i) k->launch(s, p, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --batch K: K back-to-back launches between the events, time / K. For us-scale
    // kernels a single event pair is dominated by event resolution and launch jitter.
    auto timeBatch = [&](const Params& pp) {
        float m = 0.f;
        CUDA_CHECK(cudaEventRecord(t0));
        for (int b = 0; b < a.batch; ++b) k->launch(s, pp, 0);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&m, t0, t1));
        return m / float(a.batch);
    };
    std::vector<float> ms(a.runs), msTl;
    if (wantTl) msTl.resize(a.runs);
    for (int i = 0; i < a.runs; ++i) {
        if (wantTl && (i & 1)) { msTl[i] = timeBatch(pTl); ms[i] = timeBatch(p); }   // alternate order
        else { ms[i] = timeBatch(p); if (wantTl) msTl[i] = timeBatch(pTl); }
    }
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const double med   = median(ms);
    const double inter = double(p.n) * p.n;
    const double ips   = inter / (med * 1e-3);
    const double gflop = ips * kFlopsPerInteraction * 1e-9;
    printf("kernel=%s n=%d block=%d q=%d prefetch=%d unroll=%d runs=%d batch=%d  median=%.4f ms  min=%.4f  max=%.4f  "
           "%.3f Ginteractions/s  %.1f GFLOP/s (@%.0f FLOP/interaction)\n",
           k->name, p.n, p.blockSize, li.threadsPerBody, li.prefetch, li.unroll, a.runs, a.batch, med, sorted.front(),
           sorted.back(), ips * 1e-9, gflop, kFlopsPerInteraction);

    if (wantTl) {
        const double medTl = median(msTl);
        printf("timeline: instrumented median %.4f ms vs production %.4f ms  overhead %+.2f%%\n", medTl, med,
               100.0 * (medTl / med - 1.0));
        CUDA_CHECK(cudaMemset(dTl, 0, size_t(li.gridBlocks) * a.timelineLaunches * sizeof(BlockStamp)));
        std::vector<ClockCal> calBefore, calAfter;
        clockCalibration(calBefore);   // synchronous; anchors every SM's clock64 to %globaltimer
        for (int l = 0; l < a.timelineLaunches; ++l) {   // back to back, like a batch
            Params pl = pTl;
            pl.timeline = dTl + size_t(l) * li.gridBlocks;
            k->launch(s, pl, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        clockCalibration(calAfter);
        std::vector<BlockStamp> h(size_t(li.gridBlocks) * a.timelineLaunches);
        CUDA_CHECK(cudaMemcpy(h.data(), dTl, h.size() * sizeof(BlockStamp), cudaMemcpyDeviceToHost));
        std::ofstream f(a.timeline);
        if (!f) { fprintf(stderr, "cannot write %s\n", a.timeline.c_str()); return 1; }
        f << "# kernel=" << k->name << " n=" << p.n << " block=" << p.blockSize << " q=" << li.threadsPerBody
          << " q_requested=" << a.q << " prefetch=" << li.prefetch << " unroll=" << li.unroll
          << " grid=" << li.gridBlocks << " sms=" << sms << " blocks_per_sm=" << liTl.blocksPerSM
          << " regs=" << liTl.regsPerThread << " regs_production=" << li.regsPerThread
          << " launches=" << a.timelineLaunches << " batch=" << a.batch << " runs=" << a.runs
          << " prod_median_ms=" << med << " tl_median_ms=" << median(msTl) << "\n";
        for (int ph = 0; ph < 2; ++ph)
            for (const ClockCal& c : ph ? calAfter : calBefore)
                f << "#cal," << (ph ? 'B' : 'A') << ',' << c.smid << ',' << c.tA << ',' << c.cA << ',' << c.tB
                  << ',' << c.cB << '\n';
        f << "launch,block,smid,t_start_ns,t_end_ns,cyc_start,cyc_end\n";
        for (int l = 0; l < a.timelineLaunches; ++l)
            for (int b = 0; b < li.gridBlocks; ++b) {
                const BlockStamp& r = h[size_t(l) * li.gridBlocks + b];
                f << l << ',' << r.block << ',' << r.smid << ',' << r.t0 << ',' << r.t1 << ',' << r.c0 << ','
                  << r.c1 << '\n';
            }
        printf("timeline: %d launches x %d blocks -> %s\n", a.timelineLaunches, li.gridBlocks, a.timeline.c_str());
        CUDA_CHECK(cudaFree(dTl));
    }

    // ---- validation vs reference kernel + gates (D-009) ------------------------
    double maxRel = -1.0, rmsRel = -1.0, fp64Rms = -1.0, fp64Max = -1.0, fp64Norm = -1.0;
    double naiveNorm = -1.0, kappaMax = -1.0;
    std::string g1 = "na", g2 = "na", gates = "na";
    bool gateFail = false;
    if (a.validate) {
        downloadAcc(s, acc);
        const KernelInfo* r = findKernel("allpairs_naive");
        r->launch(s, p, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        downloadAcc(s, ref);
        ErrorStats e = compareAcc(acc, ref);
        maxRel = e.maxRel; rmsRel = e.rmsRel;
        const bool bitwise = std::memcmp(acc.data(), ref.data(), acc.size() * sizeof(float4)) == 0;
        printf("validate vs allpairs_naive: max rel err %.3e  rms %.3e%s\n", maxRel, rmsRel,
               bitwise ? "  (bitwise equal)" : "");

        if (a.fp64Sample > 0) {
            // Evenly spaced sample + the 16 worst-conditioned bodies (largest kappa). The
            // shortlist (64) comes from a GPU FP32 estimate of kappa = sum|f| / |a_naive|; the
            // final 16 are ranked by kappa from the FP64 reference (critic review 1, #12).
            std::vector<int> idx;
            const int m = std::min(a.fp64Sample, p.n);
            for (int t = 0; t < m; ++t) idx.push_back(int((long long)t * p.n / m));
            std::vector<float> scale;
            forceScale(s, p, scale);
            std::vector<double> kEst(p.n);
            for (int t = 0; t < p.n; ++t) {
                double am = std::sqrt(double(ref[t].x) * ref[t].x + double(ref[t].y) * ref[t].y +
                                      double(ref[t].z) * ref[t].z);
                kEst[t] = am > 0 ? scale[t] / am : 1e300;
            }
            std::vector<int> order(p.n);
            for (int t = 0; t < p.n; ++t) order[t] = t;
            const int shortN = std::min(64, p.n);
            std::partial_sort(order.begin(), order.begin() + shortN, order.end(),
                              [&](int x, int y) { return kEst[x] > kEst[y]; });
            std::vector<int> shortlist(order.begin(), order.begin() + shortN);
            Fp64Sample sl = referenceFp64(pos, p, shortlist);
            std::vector<int> o2(shortN);
            for (int t = 0; t < shortN; ++t) o2[t] = t;
            std::sort(o2.begin(), o2.end(), [&](int x, int y) { return kappaOf(sl, x) > kappaOf(sl, y); });
            for (int t = 0; t < std::min(16, shortN); ++t)
                if (std::find(idx.begin(), idx.end(), shortlist[o2[t]]) == idx.end()) idx.push_back(shortlist[o2[t]]);

            Fp64Sample truth = referenceFp64(pos, p, idx);
            Fp64Error  ev    = compareFp64(acc, truth);
            Fp64Error  en    = compareFp64(ref, truth);
            fp64Rms = ev.rmsRel; fp64Max = ev.maxRel; fp64Norm = ev.normMax;
            naiveNorm = en.normMax; kappaMax = ev.kappaMax;
            printf("validate vs FP64 (%zu bodies, kappa_max %.1e): this rms %.3e max %.3e norm %.3e"
                   "  | naive rms %.3e max %.3e norm %.3e\n",
                   idx.size(), ev.kappaMax, ev.rmsRel, ev.maxRel, ev.normMax, en.rmsRel, en.maxRel, en.normMax);

            const double T1 = kG1 * kU * std::sqrt(double(p.n));
            const bool   ok1 = ev.normMax <= T1;
            const bool   ok2 = ev.normMax <= kG2 * en.normMax || ev.normMax == 0.0;
            const bool   ok0 = maxRel <= kG0;
            g1 = ok1 ? "PASS" : "FAIL"; g2 = ok2 ? "PASS" : "FAIL";
            gateFail = !(ok0 && ok1 && ok2);
            gates = gateFail ? "FAIL" : "PASS";
            printf("gates (D-009): G1 %s (normMax %.3e <= %.3e = %.1f u sqrtN; naive %.3e %s)  "
                   "G2 %s (<= %.1f x naive = %.3e)  G0 %s (max rel vs naive %.2e <= %.0e)  => %s\n",
                   g1.c_str(), ev.normMax, T1, kG1, en.normMax, en.normMax <= T1 ? "passes" : "FAILS",
                   g2.c_str(), kG2, kG2 * en.normMax, ok0 ? "PASS" : "FAIL", maxRel, kG0, gates.c_str());
        } else {
            gateFail = maxRel > kG0;
            gates = gateFail ? "FAIL" : "G0only";
            printf("gates: G1/G2 need --fp64-sample > 0; G0 %s\n", gateFail ? "FAIL" : "PASS");
        }
    } else {
        printf("gates: not evaluated (--no-validate)\n");
    }

    // ---- CSV ----------------------------------------------------------------
    if (!a.csv.empty()) {
        static const char* kHeader =
            "kernel,n,block,q_requested,q,prefetch,unroll,grid,regs,blocks_per_sm,occupancy,run,ms,"
            "ginteractions_per_s,gflops,max_rel_err,rms_rel_err,fp64_rms_rel,fp64_max_rel,fp64_norm_max,"
            "naive_fp64_norm_max,kappa_max,g1,g2,gates,batch,warmup_ms";
        std::string first;
        {
            std::ifstream in(a.csv);
            if (in.good()) std::getline(in, first);
        }
        if (!first.empty() && first != kHeader) {
            fprintf(stderr, "%s has a different header (older bench?); write to a new file\n", a.csv.c_str());
            return 1;
        }
        std::ofstream f(a.csv, std::ios::app);
        if (first.empty()) f << kHeader << '\n';
        for (int i = 0; i < a.runs; ++i) {
            double ipsi = inter / (ms[i] * 1e-3);
            f << k->name << ',' << p.n << ',' << p.blockSize << ',' << a.q << ',' << li.threadsPerBody << ','
              << li.prefetch << ',' << li.unroll << ',' << li.gridBlocks << ',' << li.regsPerThread << ','
              << li.blocksPerSM << ',' << occ << ',' << i << ',' << ms[i] << ',' << ipsi * 1e-9 << ','
              << ipsi * kFlopsPerInteraction * 1e-9 << ',' << maxRel << ',' << rmsRel << ',' << fp64Rms << ','
              << fp64Max << ',' << fp64Norm << ',' << naiveNorm << ',' << kappaMax << ',' << g1 << ',' << g2
              << ',' << gates << ',' << a.batch << ',' << a.warmupMs << '\n';
        }
    }

    // ---- integration + energy drift -----------------------------------------
    if (a.steps > 0) {
        upload(s, pos, vel);
        k->launch(s, p, 0);   // prime acc for KDK
        Energy e0{};
        std::ofstream elog;
        if (a.energy) e0 = computeEnergy(pos, vel, p);
        if (!a.energyLog.empty()) {
            elog.open(a.energyLog);
            if (!elog) { fprintf(stderr, "cannot write %s\n", a.energyLog.c_str()); return 1; }
            elog << "# kernel=" << k->name << " n=" << p.n << " block=" << p.blockSize << " q=" << li.threadsPerBody
                 << " prefetch=" << li.prefetch << " unroll=" << li.unroll << " dt=" << p.dt
                 << " eps=" << p.softening << " seed=" << a.seed << "\n";
            elog << "step,time,kinetic,potential,total,rel_drift\n";
            elog.precision(17);
            elog << 0 << ',' << 0.0 << ',' << e0.kinetic << ',' << e0.potential << ',' << e0.total << ',' << 0.0 << '\n';
        }
        std::vector<float4> hp, hv;
        auto w0 = std::chrono::steady_clock::now();
        for (int i = 1; i <= a.steps; ++i) {
            step(s, p, *k, 0);
            if (elog.is_open() && a.energyEvery > 0 && (i % a.energyEvery == 0 || i == a.steps)) {
                download(s, hp, hv);   // synchronises
                Energy e = computeEnergy(hp, hv, p);
                elog << i << ',' << double(i) * p.dt << ',' << e.kinetic << ',' << e.potential << ',' << e.total
                     << ',' << (e.total - e0.total) / std::abs(e0.total) << '\n';
                elog.flush();
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        if (a.energy) {
            download(s, pos, vel);
            Energy e1 = computeEnergy(pos, vel, p);
            printf("energy: E0=%.9f  E1=%.9f  rel drift=%.3e over %d steps = %.1f time units (dt=%g, eps=%g, wall %.1f s)\n",
                   e0.total, e1.total, (e1.total - e0.total) / std::abs(e0.total), a.steps, a.steps * double(p.dt),
                   p.dt, p.softening, wall);
        } else {
            printf("integrated %d steps\n", a.steps);
        }
    }

    release(s);
    return gateFail ? 3 : 0;
}
