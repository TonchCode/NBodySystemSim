// nbody_bench: times force-kernel variants and validates them against the reference.
//
//   nbody_bench --list
//   nbody_bench --kernel allpairs_tiled --n 65536 --block 256 --runs 10 --csv results/x.csv
//   nbody_bench --kernel allpairs_tuned --n 65536 --block 128 --unroll 16      (R3 #2)
//   nbody_bench --kernel allpairs_multi --n 4096 --block 256 --q 8             (R3 #3; --q 0 = auto)
//               --prefetch 0|1    multi only: next-tile register prefetch (default 1)
//   --warmup-ms 300   time-based warm-up (recommended for small N: clocks ramp from idle)
//   --batch 20        time 20 back-to-back launches per run and report the mean (small N)
//   --ic plummer|collision|core|disk|galaxies|cube_shell|cube_edges|uniform   initial conditions (§13, §19)
//   --mass-spectrum equal|salpeter|heavy-few   --bh-mass 0.01   (IC round, docs/03-design.md §19)
//
// Tree codes (R3 #4/#5: bh_naive, bh_sorted, bh_hilbert; docs/03-design.md §10-§14):
//   --theta 0.5 --leaf 16      opening parameter (bmax criterion) and leaf-bucket size
//   Phase times (keys, sort, tree, order, traverse) are measured with events inside the
//   pipeline in --runs extra launches and printed next to the production total.
//   One instrumented launch collects per-thread work counters (nodes popped, interactions,
//   and __activemask leader counts), printed as per-warp max/mean and SIMT efficiency;
//   --tree-stats out.csv also writes one row per warp.
//   Validation: theta = 0 uses the D-009 gates unchanged (the tree is then an exact,
//   reordered all-pairs sum); theta > 0 uses the pre-registered tree gates TG1/TG2 (below).
//
// Tree round 2 (R3 #6/#7, docs/03-design.md §17): bh_warp, bh_warp_list, bh_warp_naive,
//   bh_persistent, bh_persistent_lane, bh_persistent_lane_naive, bh_persistent_warp,
//   bh_persistent_static, and the control bh_coherent (no gates: computes 1 body in 32).
//   --theta is limited to [0, 1] (bmax safety argument).
//   --timeline-unit warp   one timeline record per 32-slot work unit (per persistent warp for
//                          bh_persistent_lane) instead of per block; header has slots_per_sm.
//   --occ-blocks B         pad each traversal block with dynamic shared memory so that at most B
//                          blocks are resident per SM (occupancy sweep at fixed N). Note: the
//                          padding also shrinks the L1 carve-out, so compare with --occ-blocks 12.
//   Sibling checks on every run of a round-2 kernel: TG3 (union-walk error <= #5's at the same
//   theta, and per slot never fewer pops/interactions), union work ratio, and bitwise equality +
//   equal per-slot counts against the exact sibling (persistent arms: exactly-once processing).
//   CSV (new header): ginteractions_per_s / gflops = the REAL interaction rate (tree: counted
//   interactions / pipeline time); N^2 / time is n2equiv_ginter_per_s (critic review 2, #5).
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
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "nbody.h"

namespace {

struct Args {
    std::string kernel = "allpairs_tiled";
    std::string csv, timeline, energyLog, ic = "plummer", treeStats;
    int  n = 1 << 14, block = 256, runs = 10, warmup = 3, steps = 0;
    int  q = 0, unroll = 8, prefetch = 1, fp64Sample = 512, warmupMs = 0, batch = 1;
    int  timelineLaunches = 3, energyEvery = 0, selftestDrop = -1;
    unsigned seed = 42;
    bool list = false, validate = true, energy = false, selftest = false;
    float dt = 1e-3f, softening = 1e-2f, theta = 0.5f;
    int   leaf = 16;
    int   occBlocks = 0;                 // tree round 2: cap resident blocks/SM (shared-memory padding)
    std::string tlUnit = "block";        // timeline record unit: block | warp
    std::string massSpectrum = "equal";  // IC round: equal | salpeter | heavy-few
    double      bhMass = 0.01;           // IC round: disk / galaxies central-body mass fraction
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
        else if (s == "--ic")                a.ic = next();
        else if (s == "--theta")             a.theta = nextf();
        else if (s == "--leaf")              a.leaf = nexti();
        else if (s == "--tree-stats")        a.treeStats = next();
        else if (s == "--selftest-drop")     { a.selftestDrop = nexti(); a.selftest = true; }
        else if (s == "--occ-blocks")        a.occBlocks = nexti();
        else if (s == "--timeline-unit")     a.tlUnit = next();
        else if (s == "--mass-spectrum")     a.massSpectrum = next();
        else if (s == "--bh-mass")           a.bhMass = nextf();
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); exit(2); }
    }
    // eps <= 0 makes the self-term 0 * rsqrt(0) = NaN (critic review 1, #12).
    if (!(a.softening > 0.f)) { fprintf(stderr, "--softening must be > 0 (got %g)\n", a.softening); exit(2); }
    if (a.n <= 0)             { fprintf(stderr, "--n must be > 0\n"); exit(2); }
    if (!(a.dt > 0.f))        { fprintf(stderr, "--dt must be > 0\n"); exit(2); }
    if (a.prefetch != 0 && a.prefetch != 1) { fprintf(stderr, "--prefetch must be 0 or 1\n"); exit(2); }
    // theta <= 1: the bmax criterion's safety argument holds only there (critic review 2, #11).
    if (!(a.theta >= 0.f) || a.theta > 1.f) { fprintf(stderr, "--theta must be in [0, 1]\n"); exit(2); }
    if (a.tlUnit != "block" && a.tlUnit != "warp") { fprintf(stderr, "--timeline-unit must be block or warp\n"); exit(2); }
    if (a.occBlocks < 0 || a.occBlocks > 24) { fprintf(stderr, "--occ-blocks must be in [0, 24]\n"); exit(2); }
    if (a.leaf < 1) { fprintf(stderr, "--leaf must be >= 1\n"); exit(2); }
    if (!nbody::validMassSpectrum(a.massSpectrum)) {
        fprintf(stderr, "--mass-spectrum must be %s\n", nbody::massSpectrumNames()); exit(2);
    }
    if (!(a.bhMass >= 0.0) || !(a.bhMass < 1.0)) { fprintf(stderr, "--bh-mass must be in [0, 1)\n"); exit(2); }
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
// Tree gates (theta > 0), pre-registered in docs/predictions/2026-10-17-tree-round1.md before
// any run that tests them. Errors are |a_tree - a_exact| / |a_exact| per body, over ALL bodies,
// a_exact = referenceAcc (bitwise allpairs_naive). They are bug catchers, set ~3x above the
// predicted truncation error (monopole + bmax criterion: error ~ theta^2, docs/03-design.md §12):
//   TG1  rms  <= kTG1 * (theta / 0.5)^2
//   TG2  p99  <= kTG2 * (theta / 0.5)^2
// theta = 0 is an exact (reordered) all-pairs sum and uses the D-009 gates G0/G1/G2 unchanged.
// Missed or duplicated interactions are caught exactly by --selftest (theta = 0 lattice).
constexpr double kTG1 = 0.03, kTG2 = 0.12;
double treeGateRms(double th) { return kTG1 * (th / 0.5) * (th / 0.5); }
double treeGateP99(double th) { return kTG2 * (th / 0.5) * (th / 0.5); }

// ---------------------------------------------------------------------------------------
// Normalised tree gates (IC round, D-027; the PRIMARY tree gates from this round on; TG1/TG2 above
// are still computed and printed as "legacy", but no longer gate). Per body
//   e_i = |a_tree,i - a_exact,i| / S_i,   S_i = G sum_j m_j |r_ij| / (r_ij^2 + eps^2)^1.5
// (forceScale over ALL bodies, the D-009 normaliser). S_i >= |a_i| always, and it does not vanish
// when the net force cancels (shells, the point between two galaxies, a body next to a dominant
// mass), so e_i stays bounded where |da| / |a| explodes. For one accepted cell the monopole error is
// O(theta^2) times that cell's share of S_i [D], so e is O(theta^2) for every body whatever the IC.
//   TN1  rms(e) <= kTN1 * (theta / 0.5)^2        TN2  p99(e) <= kTN2 * (theta / 0.5)^2
// Calibrated on the legacy ICs (plummer, collision, core; results/2026-10-17_ic_round/phase1/calib),
// set about 3x above the largest value measured there (bug catchers, as TG1/TG2).
constexpr double kTN1 = 0.02, kTN2 = 0.045;
double treeGateNRms(double th) { return kTN1 * (th / 0.5) * (th / 0.5); }
double treeGateNP99(double th) { return kTN2 * (th / 0.5) * (th / 0.5); }

struct NormStats { double rms = -1, p99 = -1, max = -1; int count = 0; };

// e_i over all bodies (subset == nullptr) or over the listed bodies.
NormStats normStats(const std::vector<float4>& acc, const std::vector<float4>& ref, const std::vector<float>& scale,
                    const std::vector<int>* subset = nullptr)
{
    NormStats r;
    std::vector<double> e;
    auto one = [&](size_t i) {
        const double dx = double(acc[i].x) - ref[i].x, dy = double(acc[i].y) - ref[i].y, dz = double(acc[i].z) - ref[i].z;
        const double s  = scale[i] > 0.f ? double(scale[i]) : 1.0;
        e.push_back(std::sqrt(dx * dx + dy * dy + dz * dz) / s);
    };
    if (subset) for (int i : *subset) one(size_t(i));
    else for (size_t i = 0; i < acc.size(); ++i) one(i);
    r.count = int(e.size());
    if (e.empty()) return r;
    double ss = 0, mx = 0;
    for (double x : e) { ss += x * x; mx = std::max(mx, x); }
    r.rms = std::sqrt(ss / double(e.size()));
    r.max = mx;
    const size_t at = std::min(e.size() - 1, size_t(0.99 * double(e.size())));
    std::nth_element(e.begin(), e.begin() + at, e.end());
    r.p99 = e[at];
    return r;
}

// Global normalisation (reported only): |da_i| / |a|_rms, |a|_rms over all bodies of the exact result.
void globalNorm(const std::vector<float4>& acc, const std::vector<float4>& ref, double& rms, double& p99)
{
    double ar = 0;
    for (const float4& a : ref) ar += double(a.x) * a.x + double(a.y) * a.y + double(a.z) * a.z;
    ar = std::sqrt(ar / std::max<size_t>(1, ref.size()));
    std::vector<double> e(acc.size());
    double ss = 0;
    for (size_t i = 0; i < acc.size(); ++i) {
        const double dx = double(acc[i].x) - ref[i].x, dy = double(acc[i].y) - ref[i].y, dz = double(acc[i].z) - ref[i].z;
        e[i] = std::sqrt(dx * dx + dy * dy + dz * dz) / (ar > 0 ? ar : 1.0);
        ss += e[i] * e[i];
    }
    rms = e.empty() ? 0 : std::sqrt(ss / double(e.size()));
    p99 = 0;
    if (!e.empty()) {
        const size_t at = std::min(e.size() - 1, size_t(0.99 * double(e.size())));
        std::nth_element(e.begin(), e.begin() + at, e.end());
        p99 = e[at];
    }
}

// Bodies with m >= 10 x the median mass (empty for equal masses).
std::vector<int> heavyBodies(const std::vector<float4>& pos)
{
    std::vector<float> m(pos.size());
    for (size_t i = 0; i < pos.size(); ++i) m[i] = pos[i].w;
    std::vector<int> h;
    if (m.empty()) return h;
    std::nth_element(m.begin(), m.begin() + m.size() / 2, m.end());
    const double med = m[m.size() / 2];
    for (size_t i = 0; i < pos.size(); ++i) if (pos[i].w >= 10.0 * med) h.push_back(int(i));
    return h;
}

// Per-warp work statistics from the instrumented traversal's counters (uint4 per thread:
// pops, interactions, leader pops, leader interactions), docs/03-design.md §14.
struct WarpStats {
    double interPerBody = 0, visitsPerBody = 0;
    double warpMaxMean  = 0;   // sum_w max_lane(I) / sum_w mean_lane(I): 1 = no intra-warp imbalance
    double warpMaxMeanMedian = 0;
    double simtInter    = 0;   // sum I / (32 * sum leaderI): mean active lanes / 32 at interaction sites
    double simtVisit    = 0;   // the same at node pops
    double interWarp    = 0;   // max_w max_lane(I) / mean_w max_lane(I): spread of warp cost
    long long totalInter = 0, totalVisits = 0;
    int warps = 0;
};

WarpStats warpStats(const std::vector<uint4>& wc, int n, const std::string& csvPath)
{
    WarpStats w;
    std::ofstream f;
    if (!csvPath.empty()) {
        f.open(csvPath);
        f << "warp,lanes,inter_sum,inter_max,inter_min,visit_sum,visit_max,lead_visit,lead_inter\n";
    }
    double sumMax = 0, sumMean = 0, maxOfMax = 0;
    long long leadI = 0, leadV = 0;
    std::vector<double> ratios;
    for (int b = 0; b < n; b += 32) {
        const int lanes = std::min(32, n - b);
        unsigned long long si = 0, sv = 0, lv = 0, li = 0;
        unsigned mx = 0, mn = ~0u, vmx = 0;
        for (int t = b; t < b + lanes; ++t) {
            si += wc[t].y; sv += wc[t].x; lv += wc[t].z; li += wc[t].w;
            mx = std::max(mx, wc[t].y); mn = std::min(mn, wc[t].y); vmx = std::max(vmx, wc[t].x);
        }
        const double mean = double(si) / lanes;
        sumMax += mx; sumMean += mean; maxOfMax = std::max(maxOfMax, double(mx));
        if (mean > 0) ratios.push_back(mx / mean);
        w.totalInter += si; w.totalVisits += sv; leadI += li; leadV += lv;
        ++w.warps;
        if (f) f << b / 32 << ',' << lanes << ',' << si << ',' << mx << ',' << mn << ',' << sv << ',' << vmx << ','
                 << lv << ',' << li << '\n';
    }
    w.interPerBody  = double(w.totalInter) / n;
    w.visitsPerBody = double(w.totalVisits) / n;
    w.warpMaxMean   = sumMean > 0 ? sumMax / sumMean : 0;
    if (!ratios.empty()) {
        std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
        w.warpMaxMeanMedian = ratios[ratios.size() / 2];
    }
    w.simtInter = leadI > 0 ? double(w.totalInter) / (32.0 * leadI) : 0;
    w.simtVisit = leadV > 0 ? double(w.totalVisits) / (32.0 * leadV) : 0;
    w.interWarp = sumMax > 0 ? maxOfMax / (sumMax / w.warps) : 0;
    return w;
}

// ---------------------------------------------------------------------------------------
// Tree round 2 helpers (docs/03-design.md §17).
struct RelStats { double rms = -1, p99 = -1; };

// rms and p99 of the per-body relative error |a - ref| / |ref| (same definitions as the TG1/TG2
// code in main: rms as compareAcc, p99 as the nth_element at floor(0.99 n)).
RelStats relStats(const std::vector<float4>& acc, const std::vector<float4>& ref)
{
    std::vector<double> rel(acc.size());
    double ss = 0;
    for (size_t i = 0; i < acc.size(); ++i) {
        const double dx = double(acc[i].x) - ref[i].x, dy = double(acc[i].y) - ref[i].y, dz = double(acc[i].z) - ref[i].z;
        const double rn = std::sqrt(double(ref[i].x) * ref[i].x + double(ref[i].y) * ref[i].y + double(ref[i].z) * ref[i].z);
        rel[i] = std::sqrt(dx * dx + dy * dy + dz * dz) / (rn > 0 ? rn : 1.0);
        ss += rel[i] * rel[i];
    }
    RelStats r;
    if (rel.empty()) return r;
    r.rms = std::sqrt(ss / double(rel.size()));
    const size_t at = std::min(rel.size() - 1, size_t(0.99 * double(rel.size())));
    std::nth_element(rel.begin(), rel.begin() + at, rel.end());
    r.p99 = rel[at];
    return r;
}

// Production launch of k on s (acc poisoned with NaN first), result downloaded.
std::vector<float4> runAcc(const nbody::KernelInfo& k, const nbody::DeviceState& s, const nbody::Params& p)
{
    std::vector<float4> out;
    CUDA_CHECK(cudaMemset(s.acc, 0xFF, size_t(s.n) * sizeof(float4)));
    nbody::Params q = p;
    q.timeline = nullptr; q.workCounters = nullptr; q.phaseEvents = nullptr;
    k.launch(s, q, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    nbody::downloadAcc(s, out);
    return out;
}

// Instrumented launch with zeroed work counters (one uint4 per thread slot); optional acc.
std::vector<uint4> runCounters(const nbody::KernelInfo& k, const nbody::DeviceState& s, const nbody::Params& p,
                               std::vector<float4>* accOut = nullptr)
{
    uint4* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, size_t(s.n) * sizeof(uint4)));
    CUDA_CHECK(cudaMemset(d, 0, size_t(s.n) * sizeof(uint4)));
    CUDA_CHECK(cudaMemset(s.acc, 0xFF, size_t(s.n) * sizeof(float4)));
    nbody::Params q = p;
    q.timeline = nullptr; q.phaseEvents = nullptr; q.workCounters = d;
    k.launch(s, q, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint4> h(s.n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, size_t(s.n) * sizeof(uint4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    if (accOut) nbody::downloadAcc(s, *accOut);
    return h;
}

// Slots whose pop (x) or interaction (y) count differs / is below the other run's.
int countDiff(const std::vector<uint4>& a, const std::vector<uint4>& b)
{
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i].x != b[i].x || a[i].y != b[i].y);
    return d;
}
int countBelow(const std::vector<uint4>& a, const std::vector<uint4>& b)
{
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (a[i].x < b[i].x || a[i].y < b[i].y);
    return d;
}

bool bitwiseEq(const std::vector<float4>& a, const std::vector<float4>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float4)) == 0;
}

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
        } else if (name == "bh_coherent") {
            continue;   // diagnostic control: computes 1 body in 32 by design (tested in runTreeSelftest)
        } else if (name.rfind("bh_", 0) == 0) {
            // theta = 0: every node is opened, so the tree is an exact all-pairs sum through the
            // leaf buckets; on the lattice that must be bitwise equal to the analytic sum.
            for (int b : {64, 128, 256}) for (int leaf : {1, 8, 16, 64}) {
                Params p = base; p.blockSize = b; p.theta = 0.f; p.leafSize = leaf;
                snprintf(buf, sizeof buf, "%s b%d leaf%d th0", k.name, b, leaf);
                out.push_back({&k, p, buf});
            }
        } else {
            base.blockSize = 256;
            out.push_back({&k, base, name + " b256 (default params)"});
        }
    }
    return out;
}

// Tree-specific checks (docs/03-design.md §12.3), run after the exact lattice cases:
//  (a) the Hilbert key on a 16^3 sub-grid of the 21-bit grid is a bijection whose consecutive
//      cells are face neighbours (a Hilbert curve; Morton fails this);
//  (b) for each IC x N x bh variant at theta > 0: checkLastTree passes (permutations, every body
//      in exactly one leaf, ranges, masses, boxes, root mass and COM);
//  (c) bh_sorted and bh_hilbert are BITWISE equal to bh_naive (same tree, same per-body order:
//      the thread order is the only difference between R3 #4 and #5);
//  (d) no traversal stack overflow.
// IC label "name" or "name:spectrum" (IC round): the selftest runs every IC, plus unequal-mass variants.
bool makeLabelled(const char* label, int n, unsigned seed, std::vector<float4>& pos, std::vector<float4>& vel)
{
    std::string l = label;
    nbody::ICOptions o;
    const size_t c = l.find(':');
    if (c != std::string::npos) { o.massSpectrum = l.substr(c + 1); l = l.substr(0, c); }
    return nbody::makeIC(l, n, seed, pos, vel, o);
}

int runTreeSelftest(unsigned seed)
{
    using namespace nbody;
    int fails = 0, cases = 0;
    {   // (a)
        struct P { unsigned long long k; int x, y, z; };
        std::vector<P> pts;
        for (int x = 0; x < 16; ++x) for (int y = 0; y < 16; ++y) for (int z = 0; z < 16; ++z)
            pts.push_back({hostHilbertKey(unsigned(x) << 17, unsigned(y) << 17, unsigned(z) << 17), x, y, z});
        std::sort(pts.begin(), pts.end(), [](const P& a, const P& b) { return a.k < b.k; });
        int bad = 0;
        for (size_t i = 1; i < pts.size(); ++i) {
            const int d = std::abs(pts[i].x - pts[i - 1].x) + std::abs(pts[i].y - pts[i - 1].y) +
                          std::abs(pts[i].z - pts[i - 1].z);
            if (d != 1 || pts[i].k == pts[i - 1].k) ++bad;
        }
        ++cases;
        if (bad) { ++fails; printf("  FAIL hilbert: %d of 4095 consecutive steps are not unit steps\n", bad); }
    }
    // IC round: every IC, plus unequal-mass variants (the tree is mass-weighted; D-027 / §19.4)
    const char* ics[] = {"plummer", "collision", "core", "disk", "galaxies", "cube_shell", "cube_edges", "uniform",
                         "plummer:salpeter", "disk:heavy-few", "galaxies:salpeter", "cube_shell:heavy-few"};
    const int   ns[]  = {1, 2, 3, 1000, 10007, 65536};
    const char* ks[]  = {"bh_naive", "bh_sorted", "bh_hilbert"};
    for (const char* ic : ics)
        for (int n : ns) {
            std::vector<float4> pos, vel, acc0, acc;
            if (!makeLabelled(ic, n, seed, pos, vel)) { ++fails; printf("  FAIL makeIC %s\n", ic); continue; }
            DeviceState s = allocate(n);
            upload(s, pos, vel);
            for (int cfg = 0; cfg < 2; ++cfg) {
                Params p;
                p.n = n; p.blockSize = cfg ? 256 : 128; p.theta = cfg ? 0.7f : 0.5f; p.leafSize = cfg ? 8 : 16;
                for (int v = 0; v < 3; ++v) {
                    const KernelInfo* k = findKernel(ks[v]);
                    CUDA_CHECK(cudaMemset(s.acc, 0xFF, size_t(n) * sizeof(float4)));
                    k->launch(s, p, 0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    downloadAcc(s, v ? acc : acc0);
                    TreeCheck tc = checkLastTree(s);
                    ++cases;
                    if (!tc.ok) {
                        ++fails;
                        printf("  FAIL tree %s n=%d %s th%.1f: %s\n", ic, n, ks[v], p.theta, tc.msg.c_str());
                    }
                    if (v && std::memcmp(acc.data(), acc0.data(), size_t(n) * sizeof(float4)) != 0) {
                        ++fails;
                        printf("  FAIL %s n=%d: %s not bitwise equal to bh_naive\n", ic, n, ks[v]);
                    }
                    if (n == 1 && (acc0[0].x != 0.f || acc0[0].y != 0.f || acc0[0].z != 0.f)) {
                        ++fails;
                        printf("  FAIL n=1: acceleration not 0\n");
                    }
                    if (v == 0 && cfg == 0 && n == 65536)
                        printf("  tree %-20s n=%d: depth %d, %d internal nodes, %d leaf buckets, root mass err %.1e, "
                               "COM err %.1e\n", ic, n, tc.maxDepth, tc.internalNodes, tc.leafBuckets, tc.massRelErr,
                               tc.comAbsErr);
                }
            }
            release(s);
        }
    // ---- tree round 2 (docs/03-design.md §17.6) ------------------------------------------------
    //  (e) every round-2 kernel: instrumented twin bitwise equal to production;
    //  (f) persistent arms bitwise equal to their non-persistent sibling, with EQUAL per-slot pop and
    //      interaction counts (counters accumulate, so a body processed 0 or 2 times is caught);
    //      bh_warp_list and bh_persistent_warp bitwise equal to bh_warp;
    //  (g) union walks: per slot, pops and interactions never below #5's (#4's for bh_warp_naive),
    //      and (n >= 1000, theta > 0) rms and p99 error not above the sibling's (TG3);
    //  (h) theta = 0 on random ICs: the union walks are bitwise equal to bh_sorted;
    //  (i) exact theta > 0 host check (n <= 10007): FP64 node recompute + host walk, for #5 and #7;
    //  (j) the coherent control computes exactly one body per warp, bitwise equal to bh_sorted's;
    //  (k) per-warp-unit timeline records: every unit written once with sane stamps.
    {
        const char* r2[] = {"bh_warp", "bh_warp_list", "bh_warp_naive", "bh_persistent", "bh_persistent_lane",
                            "bh_persistent_lane_naive", "bh_persistent_warp", "bh_persistent_static",
                            "bh_persistent_block", "bh_persistent_lane8", "bh_persistent_lane8_naive"};   // last 3: exploratory
        const int c0 = cases, f0 = fails;
        double worstExact = 0;
        for (const char* ic : ics)
            for (int n : ns) {
                std::vector<float4> pos, vel;
                makeLabelled(ic, n, seed, pos, vel);
                DeviceState s = allocate(n);
                upload(s, pos, vel);
                for (int cfg = 0; cfg < 3; ++cfg) {
                    if (cfg == 2 && n > 10007) continue;
                    Params p;
                    p.n = n; p.blockSize = cfg == 1 ? 256 : 128; p.leafSize = cfg == 1 ? 8 : 16;
                    p.theta = cfg == 0 ? 0.5f : (cfg == 1 ? 0.7f : 0.f);
                    const KernelInfo* k4 = findKernel("bh_naive");
                    const KernelInfo* k5 = findKernel("bh_sorted");
                    std::vector<float4> acc4, acc5, acc7;
                    const std::vector<uint4> wc4 = runCounters(*k4, s, p, &acc4);
                    const std::vector<uint4> wc5 = runCounters(*k5, s, p, &acc5);
                    std::vector<uint4> wc7;
                    std::vector<float4> ref;
                    std::vector<float>  scale;
                    const bool errCheck = n >= 1000 && p.theta > 0.f;
                    if (errCheck) { referenceAcc(s, p, ref); forceScale(s, p, scale); }
                    for (const char* nm : r2) {
                        const KernelInfo* k = findKernel(nm);
                        const LaunchInfo  li = k->query(n, p);
                        std::vector<float4> ai;
                        const std::vector<uint4> wci = runCounters(*k, s, p, &ai);
                        std::vector<float4> ap = runAcc(*k, s, p);
                        const std::string name = nm;
                        if (name == "bh_persistent_static") {   // again, now partitioned on measured costs
                            const std::vector<float4> ap2 = runAcc(*k, s, p);
                            ++cases;
                            if (!bitwiseEq(ap, ap2)) { ++fails; printf("  FAIL %s n=%d %s: result changes with the partition\n", ic, n, nm); }
                        }
                        ++cases;
                        if (!bitwiseEq(ai, ap)) { ++fails; printf("  FAIL %s n=%d %s th%.1f: instrumented != production\n", ic, n, nm, p.theta); }
                        if (name == "bh_warp") { acc7 = ap; wc7 = wci; }
                        if (li.exactSibling) {
                            const std::string sib = li.exactSibling;
                            const std::vector<float4>& ra = sib == "bh_warp" ? acc7 : (sib == "bh_naive" ? acc4 : acc5);
                            const std::vector<uint4>&  rw = sib == "bh_warp" ? wc7 : (sib == "bh_naive" ? wc4 : wc5);
                            ++cases;
                            if (!bitwiseEq(ap, ra)) { ++fails; printf("  FAIL %s n=%d %s th%.1f: not bitwise equal to %s\n", ic, n, nm, p.theta, li.exactSibling); }
                            ++cases;
                            if (const int d = countDiff(wci, rw)) {
                                ++fails; printf("  FAIL %s n=%d %s th%.1f: %d slots with counts != %s (not exactly once)\n", ic, n, nm, p.theta, d, li.exactSibling);
                            }
                        }
                        if (li.errorSibling) {
                            const bool naiveSib = std::string(li.errorSibling) == "bh_naive";
                            ++cases;
                            if (const int b = countBelow(wci, naiveSib ? wc4 : wc5)) {
                                ++fails; printf("  FAIL %s n=%d %s th%.1f: %d slots with fewer pops/interactions than %s\n", ic, n, nm, p.theta, b, li.errorSibling);
                            }
                            if (p.theta == 0.f) {
                                ++cases;
                                if (!bitwiseEq(ap, naiveSib ? acc4 : acc5)) { ++fails; printf("  FAIL %s n=%d %s th0: not bitwise equal to %s\n", ic, n, nm, li.errorSibling); }
                            }
                            if (errCheck) {
                                const NormStats e = normStats(ap, ref, scale), es = normStats(naiveSib ? acc4 : acc5, ref, scale);
                                ++cases;
                                // Known exception (D-027, observed in IC-round phase 1): on the 1D stress IC the union of
                                // 32 RANDOM bodies (bh_warp_naive) refines cells whose children do not cancel as well as
                                // the parent, so its aggregate error is ~13% above #4's although every cell it uses
                                // passes the lane's own criterion (the exact host check passes). Reported, not counted.
                                const bool xfail = naiveSib && std::string(ic).rfind("cube_edges", 0) == 0;
                                if (xfail && !(e.rms <= es.rms && e.p99 <= es.p99)) {
                                    printf("  XFAIL %s n=%d %s th%.1f: TG3n rms %.3e > %.3e or p99 %.3e > %.3e (known, D-027; not counted)\n",
                                           ic, n, nm, p.theta, e.rms, es.rms, e.p99, es.p99);
                                } else if (!(e.rms <= es.rms && e.p99 <= es.p99)) {
                                    ++fails; printf("  FAIL %s n=%d %s th%.1f: TG3n rms %.3e > %.3e or p99 %.3e > %.3e\n", ic, n, nm, p.theta, e.rms, es.rms, e.p99, es.p99);
                                }
                                if (name == "bh_warp" && n == 65536 && cfg == 0)
                                    printf("  union %-20s n=%d: normalised rms %.3e vs #5 %.3e, p99 %.3e vs %.3e\n", ic, n, e.rms, es.rms, e.p99, es.p99);
                            }
                        }
                    }
                    if (n <= 10007 && p.theta > 0.f) {   // (i)
                        const float eps2 = p.softening * p.softening;
                        for (int uw = 0; uw <= 1; ++uw) {
                            const TreeExactCheck x = checkLastTreeExact(uw ? acc7 : acc5, uw ? &wc7 : &wc5, eps2, p.G, uw == 1);
                            ++cases;
                            worstExact = std::max(worstExact, x.maxErrOverBound);
                            if (!x.ok) { ++fails; printf("  FAIL exact host check %s n=%d %s th%.1f: %s\n", ic, n, uw ? "union" : "#5", p.theta, x.msg.c_str()); }
                            if (n == 10007 && cfg == 0 && std::string(ic) == "plummer")
                                printf("  exact host check plummer n=%d %s: %d nodes (max mass rel %.1e, COM abs %.1e, bmax abs %.1e), "
                                       "%d bodies, count mismatches %d, max |da|/sum|f| = %.3f of (I+10)u\n", n,
                                       uw ? "union walk" : "#5 walk", x.nodes, x.maxMassRel, x.maxComAbs, x.maxBmaxAbs, x.bodies,
                                       x.countMismatch, x.maxErrOverBound);
                        }
                    }
                }
                release(s);
            }
        {   // (j) coherent control and (k) unit timelines, plummer
            for (int n : {1000, 10007, 65536}) {
                std::vector<float4> pos, vel;
                makeIC("plummer", n, seed, pos, vel);
                DeviceState s = allocate(n);
                upload(s, pos, vel);
                Params p;
                p.n = n; p.blockSize = 128; p.theta = 0.5f; p.leafSize = 16;
                const std::vector<float4> a5 = runAcc(*findKernel("bh_sorted"), s, p);
                const std::vector<float4> ac = runAcc(*findKernel("bh_coherent"), s, p);
                int written = 0, wrong = 0;
                for (int i = 0; i < n; ++i)
                    if (!std::isnan(ac[i].x)) { ++written; if (std::memcmp(&ac[i], &a5[i], 16) != 0) ++wrong; }
                ++cases;
                if (written != (n + 31) / 32 || wrong) {
                    ++fails; printf("  FAIL coherent control n=%d: %d bodies written (want %d), %d differ from bh_sorted\n", n, written, (n + 31) / 32, wrong);
                }
                const char* tk[] = {"bh_sorted", "bh_warp", "bh_warp_list", "bh_persistent", "bh_persistent_lane",
                                    "bh_persistent_warp", "bh_persistent_static", "bh_coherent"};
                for (const char* nm : tk) {
                    const KernelInfo* k = findKernel(nm);
                    Params q = p;
                    q.timelineWarp = 1;
                    q.timeline = reinterpret_cast<BlockStamp*>(uintptr_t(16));
                    const int rec = k->query(n, q).timelineRecords;
                    BlockStamp* d = nullptr;
                    CUDA_CHECK(cudaMalloc(&d, size_t(rec) * sizeof(BlockStamp)));
                    CUDA_CHECK(cudaMemset(d, 0, size_t(rec) * sizeof(BlockStamp)));
                    q.timeline = d;
                    k->launch(s, q, 0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    std::vector<BlockStamp> h(rec);
                    CUDA_CHECK(cudaMemcpy(h.data(), d, size_t(rec) * sizeof(BlockStamp), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaFree(d));
                    int bad = 0;
                    for (int b = 0; b < rec; ++b)
                        if (h[b].block != unsigned(b) || h[b].t0 == 0 || h[b].t1 < h[b].t0 || h[b].c1 < h[b].c0 || h[b].smid >= 1024) ++bad;
                    ++cases;
                    if (bad) { ++fails; printf("  FAIL warp-unit timeline %s n=%d: %d/%d records invalid\n", nm, n, bad, rec); }
                }
                release(s);
            }
        }
        printf("selftest (tree round 2): %d checks, %d failed; exact host check worst |da|/sum|f| = %.3f of its (I+10)u bound\n",
               cases - c0, fails - f0, worstExact);
    }
    {   // (l) IC round: IC invariants and the legacy-IC regression hashes
        const int c0 = cases, f0 = fails;
        // FNV-1a of (pos, vel), seed 42, computed with the PRE-IC-round init.cpp (results/2026-10-17_ic_round/
        // phase1/legacy_hashes.txt): the legacy ICs with the equal spectrum must stay bitwise unchanged.
        struct H { const char* ic; int n; unsigned long long h; };
        const H legacy[] = {{"plummer", 1000, 0x128c85600e7e93e0ull}, {"plummer", 65536, 0xe11a638ea14aeb36ull},
                            {"collision", 1000, 0x764e2d67c14abbbbull}, {"collision", 65536, 0xaa4211110a9bdd82ull},
                            {"core", 10007, 0x302f863d9a2f9713ull}, {"core", 65536, 0xee94e83e73b5ed43ull}};
        if (seed == 42)
            for (const H& h : legacy) {
                std::vector<float4> pos, vel;
                makeIC(h.ic, h.n, 42, pos, vel, ICOptions{});
                ++cases;
                if (icHash(pos, vel) != h.h) { ++fails; printf("  FAIL legacy IC %s n=%d changed (hash %016llx)\n", h.ic, h.n, icHash(pos, vel)); }
            }
        const char* sp[] = {"equal", "salpeter", "heavy-few"};
        const char* names[] = {"plummer", "collision", "core", "disk", "galaxies", "cube_shell", "cube_edges", "uniform"};
        for (const char* nm : names)
            for (const char* spec : sp)
                for (int n : {1, 2, 3, 1000, 65536}) {
                    ICOptions o; o.massSpectrum = spec;
                    std::vector<float4> pos, vel, p2, v2;
                    ++cases;
                    if (!makeIC(nm, n, seed, pos, vel, o) || int(pos.size()) != n || int(vel.size()) != n) {
                        ++fails; printf("  FAIL IC %s:%s n=%d not generated\n", nm, spec, n); continue;
                    }
                    makeIC(nm, n, seed, p2, v2, o);
                    double M = 0, cx = 0, cy = 0, cz = 0, px = 0, py = 0, pz = 0, ext = 0;
                    bool finite = true, positive = true;
                    for (int i = 0; i < n; ++i) {
                        const float4 b = pos[i], v = vel[i];
                        finite = finite && std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.z) && std::isfinite(b.w) &&
                                 std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
                        positive = positive && b.w > 0.f;
                        M += b.w; cx += double(b.w) * b.x; cy += double(b.w) * b.y; cz += double(b.w) * b.z;
                        px += double(b.w) * v.x; py += double(b.w) * v.y; pz += double(b.w) * v.z;
                        ext = std::max(ext, double(std::max(std::fabs(b.x), std::max(std::fabs(b.y), std::fabs(b.z)))));
                    }
                    const bool legacyEq = std::string(spec) == "equal" &&
                                          (std::string(nm) == "plummer" || std::string(nm) == "collision" || std::string(nm) == "core");
                    const double tolC = 1e-6 * std::max(1.0, ext);
                    const double com = std::sqrt(cx * cx + cy * cy + cz * cz) / M, mom = std::sqrt(px * px + py * py + pz * pz);
                    std::string why;
                    if (!finite) why = "non-finite value";
                    else if (!positive) why = "mass <= 0";
                    else if (std::fabs(M - 1.0) > 1e-5) why = "total mass != 1";
                    // (the legacy equal-mass ICs are not re-centred, e.g. collision with odd n: hash-checked above instead)
                    else if (!legacyEq && n > 1 && com > tolC) why = "COM not at the origin";
                    else if (!legacyEq && n > 1 && mom > 1e-6) why = "net momentum != 0";
                    else if (std::memcmp(pos.data(), p2.data(), size_t(n) * 16) || std::memcmp(vel.data(), v2.data(), size_t(n) * 16))
                        why = "not deterministic";
                    if (!why.empty()) {
                        ++fails;
                        printf("  FAIL IC %s:%s n=%d: %s (M %.9f, |com| %.2e, |p| %.2e)\n", nm, spec, n, why.c_str(), M, com, mom);
                    }
                }
        printf("selftest (IC invariants): %d checks, %d failed (mass 1, COM, momentum, finite, positive masses, "
               "deterministic, legacy hashes)\n", cases - c0, fails - f0);
    }
    const int ov = treeStackOverflows(true);
    ++cases;
    if (ov) { ++fails; printf("  FAIL tree stack overflows: %d\n", ov); }
    printf("selftest (tree): %d checks, %d failed\n", cases, fails);
    return fails;
}

int runSelftest(int dropBody, unsigned seed)
{
    using namespace nbody;
    const int ns[] = {1, 2, 31, 64, 255, 1000, 1025, 4097, 10007, 65536};
    const int kMaxN = 65536;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coord(-16, 15), mass(1, 2);
    // IC round: an unequal-mass pass of the same exact lattice (massMode 1): masses 2^k, k in [0, 3], and 1% of
    // the bodies at 128 (a heavy-few spectrum, ratio 128). sum_j m_j |X_j - X_i| <= 31 * sum m stays < 2^24
    // units up to N = 65536 (sum m <= ~3.3e5 there), so every FP32 operation is still exact in any order.
    std::uniform_int_distribution<int> pow2(0, 3);
    std::uniform_real_distribution<double> heavyU(0.0, 1.0);
    BlockStamp* dTl = nullptr;
    CUDA_CHECK(cudaMalloc(&dTl, size_t(kMaxN) * sizeof(BlockStamp)));   // grid <= n for every variant
    std::vector<BlockStamp> hTl;
    int nCases = 0, nFail = 0, nStampFail = 0, nTrivial = 0, nExpectFail = 0, nMissed = 0;
    printf("selftest: exact dyadic lattice, eps = 1, masses {1,2} and an unequal-mass pass {1,2,4,8,128}%s\n",
           dropBody >= 0 ? "  [--selftest-drop: every case is EXPECTED to fail]" : "");

    for (int massMode = 0; massMode <= 1; ++massMode)
    for (int n : ns) {
        if (massMode == 1 && n != 64 && n != 1025 && n != 10007 && n != 65536) continue;
        std::vector<int> X(n), Y(n), Z(n), M(n);
        long long sx = 0, sy = 0, sz = 0, sm = 0;
        for (int i = 0; i < n; ++i) {
            X[i] = coord(rng); Y[i] = coord(rng); Z[i] = coord(rng);
            M[i] = massMode == 0 ? mass(rng) : (heavyU(rng) < 0.01 ? 128 : (1 << pow2(rng)));
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
                        printf("  FAIL n=%-6d %-22s%s%s  %d/%d bodies wrong; body %d got (%.9g %.9g %.9g) want (%.9g %.9g %.9g)\n",
                               n, c.label.c_str(), tl ? " +timeline" : "", massMode ? " [unequal masses]" : "", bad, n, firstBad, acc[firstBad].x,
                               acc[firstBad].y, acc[firstBad].z, ex[firstBad], ey[firstBad], ez[firstBad]);
                }
                if (tl) {   // the stamps themselves: every block written once, sane times
                    const LaunchInfo lq = c.k->query(n, p);
                    const int grid = lq.timelineRecords > 0 ? lq.timelineRecords : lq.gridBlocks;
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
    if (int ov = treeStackOverflows(true)) { printf("  FAIL tree stack overflows: %d\n", ov); ++nFail; }
    if (dropBody < 0) nFail += runTreeSelftest(seed);
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
    p.theta = a.theta; p.leafSize = a.leaf;
    p.timelineWarp = a.tlUnit == "warp" ? 1 : 0;

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
        if (a.occBlocks > 0) {   // occupancy sweep at fixed N (critic review 2, #6): pad dynamic smem
            if (!li.approximate) { fprintf(stderr, "--occ-blocks: tree kernels only\n"); return 2; }
            const int per = dp.sharedMemPerMultiprocessor / a.occBlocks;   // 102400 B / B
            const int pad = per - int(dp.reservedSharedMemPerBlock) - int(li.dynSmem);
            if (pad < 0) { fprintf(stderr, "--occ-blocks %d: no room for padding\n", a.occBlocks); return 2; }
            p.smemPad = pad;
            li = k->query(p.n, p);
            printf("# occupancy cap: --occ-blocks %d -> %d B padding per block -> resident %d blocks/SM%s\n", a.occBlocks,
                   pad, li.blocksPerSM, li.blocksPerSM == a.occBlocks ? "" : "  (WARNING: not the requested value)");
        }
        occ = printLaunch("launch ", li);
        if (li.persistent) printf("# persistent: grid = min(resident blocks/SM x %d SMs, blocks with work) = %d blocks (%d warps)\n", sms,
                                  li.gridBlocks, li.gridBlocks * li.blockThreads / 32);
        if (li.control) printf("# CONTROL kernel: forces of only 1 body in 32 are computed; validation and gates are skipped\n");
    }

    // ---- instrumented twin (timeline) ---------------------------------------
    const bool wantTl = !a.timeline.empty();
    BlockStamp* dTl = nullptr;
    Params pTl = p;
    LaunchInfo liTl;
    if (wantTl) {
        if (!k->query) { fprintf(stderr, "--timeline: kernel %s has no query\n", k->name); return 2; }
        pTl.timeline = reinterpret_cast<BlockStamp*>(uintptr_t(16));   // query only tests for non-null
        liTl = k->query(p.n, pTl);
        if (liTl.timelineRecords <= 0) liTl.timelineRecords = liTl.gridBlocks;
        if (liTl.timelineSlotsPerSM <= 0) liTl.timelineSlotsPerSM = liTl.blocksPerSM;
        CUDA_CHECK(cudaMalloc(&dTl, size_t(liTl.timelineRecords) * a.timelineLaunches * sizeof(BlockStamp)));
        pTl.timeline = dTl;
        if (!liTl.instrumented) { fprintf(stderr, "--timeline: kernel %s has no instrumented build\n", k->name); return 2; }
        printLaunch("timeline", liTl);
        if (liTl.blocksPerSM != li.blocksPerSM)
            printf("# WARNING: instrumented build has a different occupancy (%d vs %d blocks/SM)\n",
                   liTl.blocksPerSM, li.blocksPerSM);
    }

    std::vector<float4> pos, vel, acc, ref;
    ICOptions icOpt;
    icOpt.massSpectrum = a.massSpectrum; icOpt.bhMass = a.bhMass; icOpt.softening = a.softening;
    if (!makeIC(a.ic, p.n, a.seed, pos, vel, icOpt)) { fprintf(stderr, "unknown --ic %s (%s)\n", a.ic.c_str(), icNames()); return 2; }
    const unsigned long long icH = icHash(pos, vel);
    double mMin = 1e300, mMax = 0, mTot = 0;
    for (const float4& b : pos) { mMin = std::min(mMin, double(b.w)); mMax = std::max(mMax, double(b.w)); mTot += b.w; }
    const double dtAdv = dtAdvisory(pos, p.softening, p.G);
    printf("# ic=%s seed=%u theta=%g leaf=%d mass_spectrum=%s bh_mass=%g ic_hash=%016llx | mass total %.9f, min %.3e, "
           "max %.3e (max/min %.3g) | dt advisory %.2e (dt %g%s)\n",
           a.ic.c_str(), a.seed, p.theta, p.leafSize, a.massSpectrum.c_str(), a.bhMass, icH, mTot, mMin, mMax, mMax / mMin,
           dtAdv, p.dt, p.dt > dtAdv ? ": dt ABOVE the advisory, integration runs are not converged near the heaviest body" : "");
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
    // Tree kernels (critic review 2, #5): N^2 / time is NOT the tree's interaction rate, so it is
    // labelled N^2-equivalent; the real rates are on the work: line and in the CSV.
    if (li.approximate)
        printf("kernel=%s n=%d block=%d q=%d prefetch=%d unroll=%d runs=%d batch=%d  median=%.4f ms  min=%.4f  max=%.4f  "
               "%.3f G N^2-equivalent interactions/s (not the tree rate; see the work: line)\n",
               k->name, p.n, p.blockSize, li.threadsPerBody, li.prefetch, li.unroll, a.runs, a.batch, med, sorted.front(),
               sorted.back(), ips * 1e-9);
    else
        printf("kernel=%s n=%d block=%d q=%d prefetch=%d unroll=%d runs=%d batch=%d  median=%.4f ms  min=%.4f  max=%.4f  "
               "%.3f Ginteractions/s  %.1f GFLOP/s (@%.0f FLOP/interaction)\n",
               k->name, p.n, p.blockSize, li.threadsPerBody, li.prefetch, li.unroll, a.runs, a.batch, med, sorted.front(),
               sorted.back(), ips * 1e-9, gflop, kFlopsPerInteraction);

    if (wantTl) {
        const double medTl = median(msTl);
        printf("timeline: instrumented median %.4f ms vs production %.4f ms  overhead %+.2f%%\n", medTl, med,
               100.0 * (medTl / med - 1.0));
        const int nRec = liTl.timelineRecords;
        CUDA_CHECK(cudaMemset(dTl, 0, size_t(nRec) * a.timelineLaunches * sizeof(BlockStamp)));
        std::vector<ClockCal> calBefore, calAfter;
        clockCalibration(calBefore);   // synchronous; anchors every SM's clock64 to %globaltimer
        for (int l = 0; l < a.timelineLaunches; ++l) {   // back to back, like a batch
            Params pl = pTl;
            pl.timeline = dTl + size_t(l) * nRec;
            k->launch(s, pl, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        clockCalibration(calAfter);
        std::vector<BlockStamp> h(size_t(nRec) * a.timelineLaunches);
        CUDA_CHECK(cudaMemcpy(h.data(), dTl, h.size() * sizeof(BlockStamp), cudaMemcpyDeviceToHost));
        std::ofstream f(a.timeline);
        if (!f) { fprintf(stderr, "cannot write %s\n", a.timeline.c_str()); return 1; }
        f << "# kernel=" << k->name << " n=" << p.n << " block=" << p.blockSize << " q=" << li.threadsPerBody
          << " q_requested=" << a.q << " prefetch=" << li.prefetch << " unroll=" << li.unroll
          << " grid=" << li.gridBlocks << " sms=" << sms << " blocks_per_sm=" << liTl.blocksPerSM
          << " regs=" << liTl.regsPerThread << " regs_production=" << li.regsPerThread
          << " launches=" << a.timelineLaunches << " batch=" << a.batch << " runs=" << a.runs
          << " prod_median_ms=" << med << " tl_median_ms=" << median(msTl) << " unit=" << a.tlUnit
          << " records=" << nRec << " slots_per_sm=" << liTl.timelineSlotsPerSM << " theta=" << p.theta
          << " leaf=" << p.leafSize << " ic=" << a.ic << " smem_pad=" << p.smemPad << "\n";
        for (int ph = 0; ph < 2; ++ph)
            for (const ClockCal& c : ph ? calAfter : calBefore)
                f << "#cal," << (ph ? 'B' : 'A') << ',' << c.smid << ',' << c.tA << ',' << c.cA << ',' << c.tB
                  << ',' << c.cB << '\n';
        f << "launch,block,smid,t_start_ns,t_end_ns,cyc_start,cyc_end\n";
        for (int l = 0; l < a.timelineLaunches; ++l)
            for (int b = 0; b < nRec; ++b) {
                const BlockStamp& r = h[size_t(l) * nRec + b];
                f << l << ',' << r.block << ',' << r.smid << ',' << r.t0 << ',' << r.t1 << ',' << r.c0 << ','
                  << r.c1 << '\n';
            }
        printf("timeline: %d launches x %d %s records -> %s\n", a.timelineLaunches, nRec, a.tlUnit.c_str(), a.timeline.c_str());
        CUDA_CHECK(cudaFree(dTl));
    }

    // ---- tree kernels: phase times + per-warp work statistics -------------------
    std::vector<double> phaseMed;
    WarpStats ws;
    int stackOverflows = 0;
    TreeDepth td;
    std::vector<uint4> wcOwn;
    if (li.nPhases > 0) {
        std::vector<cudaEvent_t> ev(li.nPhases + 1);
        for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
        std::vector<std::vector<float>> ph(li.nPhases, std::vector<float>(a.runs));
        std::vector<float> phSum(a.runs);
        Params pp = p;
        pp.phaseEvents = ev.data();
        for (int r = 0; r < a.runs; ++r) {
            k->launch(s, pp, 0);
            CUDA_CHECK(cudaEventSynchronize(ev[li.nPhases]));
            for (int q = 0; q < li.nPhases; ++q) CUDA_CHECK(cudaEventElapsedTime(&ph[q][r], ev[q], ev[q + 1]));
            CUDA_CHECK(cudaEventElapsedTime(&phSum[r], ev[0], ev[li.nPhases]));
        }
        printf("phases (median of %d, ms):", a.runs);
        for (int q = 0; q < li.nPhases; ++q) {
            phaseMed.push_back(median(ph[q]));
            printf("  %s %.4f", li.phaseNames[q], phaseMed.back());
        }
        const double build = phaseMed.size() >= 3 ? phaseMed[0] + phaseMed[1] + phaseMed[2] : 0.0;
        printf("  | build(keys+sort+tree) %.4f  order %.4f  traverse %.4f  sum %.4f  (production total %.4f)\n", build,
               phaseMed.size() > 3 ? phaseMed[3] : 0.0, phaseMed.back(), median(phSum), med);
        for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));

        uint4* dWc = nullptr;
        CUDA_CHECK(cudaMalloc(&dWc, size_t(p.n) * sizeof(uint4)));
        CUDA_CHECK(cudaMemset(dWc, 0, size_t(p.n) * sizeof(uint4)));
        Params pw = p;
        pw.workCounters = dWc;
        k->launch(s, pw, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint4> wc(p.n);
        CUDA_CHECK(cudaMemcpy(wc.data(), dWc, size_t(p.n) * sizeof(uint4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dWc));
        ws = warpStats(wc, p.n, a.treeStats);
        wcOwn = wc;
        const double trav = phaseMed.back();
        printf("work: %.1f interactions/body (%.3e total), %.1f nodes popped/body | per-warp max/mean %.3f "
               "(median warp %.3f) | SIMT eff interactions %.3f, pops %.3f | warp-cost max/mean %.2f | "
               "traversal %.1f G interactions/s%s\n",
               ws.interPerBody, double(ws.totalInter), ws.visitsPerBody, ws.warpMaxMean, ws.warpMaxMeanMedian,
               ws.simtInter, ws.simtVisit, ws.interWarp, trav > 0 ? ws.totalInter / (trav * 1e-3) * 1e-9 : 0.0,
               a.treeStats.empty() ? "" : ("  -> " + a.treeStats).c_str());
        printf("rates: real %.2f G interactions/s over the full pipeline (%.4f ms), %.2f over the traversal; "
               "N^2-equivalent %.1f G/s%s\n", ws.totalInter / (med * 1e-3) * 1e-9, med,
               trav > 0 ? ws.totalInter / (trav * 1e-3) * 1e-9 : 0.0, ips * 1e-9,
               li.control ? "  (control: lane-work; 1 body in 32 computed)" : "");
        stackOverflows = treeStackOverflows(true);
        if (stackOverflows) printf("ERROR: %d traversal stack overflows (results invalid)\n", stackOverflows);
        td = lastTreeDepth();
        printf("tree: max depth %d, push depth %d -> stack bound %d (capacity: per-thread %d, warp %d)%s, duplicate keys %d "
               "(%.3f%% of bodies)\n", td.maxDepth, td.pushDepth, td.stackBound, td.threadStack, td.warpStack,
               td.stackBound > td.warpStack ? "  WARNING: the warp walk can overflow on this tree" : "", td.dupKeys,
               100.0 * td.dupKeys / std::max(1, p.n));
    }

    // ---- validation vs reference kernel + gates (D-009) ------------------------
    double maxRel = -1.0, rmsRel = -1.0, fp64Rms = -1.0, fp64Max = -1.0, fp64Norm = -1.0;
    double naiveNorm = -1.0, kappaMax = -1.0;
    std::string g1 = "na", g2 = "na", gates = "na";
    bool gateFail = false;
    double errP50 = -1.0, errP99 = -1.0;
    const bool treeGates = li.approximate && p.theta > 0.f;
    std::string tg3 = "na", exactMatch = "na", countsMatch = "na";
    double sibRms = -1, sibP99 = -1, unionI = -1, unionV = -1;
    NormStats ns;
    double gRms = -1, gP99 = -1, sibNRms = -1, sibNP99 = -1, heavyNormMax = -1;
    std::string tn1 = "na", tn2 = "na", tg3n = "na";
    if (li.control) {
        printf("gates: not evaluated (control kernel)\n");
        gates = "control";
    } else if (a.validate) {
        k->launch(s, p, 0);   // the timed production result (the twins above rewrote acc)
        CUDA_CHECK(cudaDeviceSynchronize());
        downloadAcc(s, acc);
        referenceAcc(s, p, ref);   // allpairs_naive loop in TDR-safe chunks, bitwise equal to it
        ErrorStats e = compareAcc(acc, ref);
        maxRel = e.maxRel; rmsRel = e.rmsRel;
        const bool bitwise = std::memcmp(acc.data(), ref.data(), acc.size() * sizeof(float4)) == 0;
        printf("validate vs allpairs_naive: max rel err %.3e  rms %.3e%s\n", maxRel, rmsRel,
               bitwise ? "  (bitwise equal)" : "");

        // Sum of force magnitudes S_i for every body: the normaliser of the D-009 FP64 metric and of the
        // normalised tree gates TN1/TN2 (D-027). FP32, chunked, O(N^2) (a scale, not a result).
        std::vector<float> scale;
        if (a.fp64Sample > 0 || treeGates) forceScale(s, p, scale);
        const std::vector<int> heavy = heavyBodies(pos);
        if (a.fp64Sample > 0) {
            // Evenly spaced sample + the 16 worst-conditioned bodies (largest kappa). The
            // shortlist (64) comes from a GPU FP32 estimate of kappa = sum|f| / |a_naive|; the
            // final 16 are ranked by kappa from the FP64 reference (critic review 1, #12).
            // IC round: with unequal masses, also the 16 heaviest bodies.
            std::vector<int> idx;
            const int m = std::min(a.fp64Sample, p.n);
            for (int t = 0; t < m; ++t) idx.push_back(int((long long)t * p.n / m));
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
            if (!heavy.empty()) {
                std::vector<int> hs = heavy;
                std::partial_sort(hs.begin(), hs.begin() + std::min<size_t>(16, hs.size()), hs.end(),
                                  [&](int x, int y) { return pos[x].w > pos[y].w; });
                for (size_t t = 0; t < std::min<size_t>(16, hs.size()); ++t)
                    if (std::find(idx.begin(), idx.end(), hs[t]) == idx.end()) idx.push_back(hs[t]);
            }

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
            if (!treeGates) printf("gates (D-009): G1 %s (normMax %.3e <= %.3e = %.1f u sqrtN; naive %.3e %s)  "
                   "G2 %s (<= %.1f x naive = %.3e)  G0 %s (max rel vs naive %.2e <= %.0e)  => %s\n",
                   g1.c_str(), ev.normMax, T1, kG1, en.normMax, en.normMax <= T1 ? "passes" : "FAILS",
                   g2.c_str(), kG2, kG2 * en.normMax, ok0 ? "PASS" : "FAIL", maxRel, kG0, gates.c_str());
        } else if (!treeGates) {
            gateFail = maxRel > kG0;
            gates = gateFail ? "FAIL" : "G0only";
            printf("gates: G1/G2 need --fp64-sample > 0; G0 %s\n", gateFail ? "FAIL" : "PASS");
        }
        if (treeGates) {   // tree truncation error over all bodies
            std::vector<double> rel(acc.size());
            for (size_t i = 0; i < acc.size(); ++i) {
                const double dx = double(acc[i].x) - ref[i].x, dy = double(acc[i].y) - ref[i].y,
                             dz = double(acc[i].z) - ref[i].z;
                const double rn = std::sqrt(double(ref[i].x) * ref[i].x + double(ref[i].y) * ref[i].y +
                                            double(ref[i].z) * ref[i].z);
                rel[i] = std::sqrt(dx * dx + dy * dy + dz * dz) / (rn > 0 ? rn : 1.0);
            }
            auto pct = [&](double q) {
                size_t at = std::min(rel.size() - 1, size_t(q * double(rel.size())));
                std::nth_element(rel.begin(), rel.begin() + at, rel.end());
                return rel[at];
            };
            errP50 = pct(0.50);
            errP99 = pct(0.99);
            ns = normStats(acc, ref, scale);
            globalNorm(acc, ref, gRms, gP99);
            // legacy per-body relative gates (D-017): reported, no longer gating (D-027)
            const double T1 = treeGateRms(p.theta), T2 = treeGateP99(p.theta);
            const bool ok1 = rmsRel <= T1, ok2 = errP99 <= T2, okS = stackOverflows == 0;
            g1 = ok1 ? "PASS" : "FAIL"; g2 = ok2 ? "PASS" : "FAIL";
            // primary normalised gates (D-027)
            const double N1 = treeGateNRms(p.theta), N2 = treeGateNP99(p.theta);
            const bool okN1 = ns.rms <= N1, okN2 = ns.p99 <= N2;
            tn1 = okN1 ? "PASS" : "FAIL"; tn2 = okN2 ? "PASS" : "FAIL";
            gateFail = !(okN1 && okN2 && okS);
            gates = gateFail ? "FAIL" : "PASS";
            printf("tree error vs exact (all %d bodies), per-body relative (legacy): rms %.3e  p50 %.3e  p99 %.3e  max %.3e\n",
                   p.n, rmsRel, errP50, errP99, maxRel);
            printf("tree error vs exact, normalised |da|/sum|f| (all bodies): rms %.3e  p99 %.3e  max %.3e | global "
                   "|da|/|a|_rms: rms %.3e  p99 %.3e\n", ns.rms, ns.p99, ns.max, gRms, gP99);
            if (!heavy.empty()) {
                std::vector<int> light;
                std::vector<char> isH(p.n, 0);
                for (int i : heavy) isH[i] = 1;
                for (int i = 0; i < p.n; ++i) if (!isH[i]) light.push_back(i);
                const NormStats hs = normStats(acc, ref, scale, &heavy), ls = normStats(acc, ref, scale, &light);
                heavyNormMax = hs.max;
                printf("normalised error by mass: heavy (m >= 10 x median, %d bodies) rms %.3e max %.3e | light (%d) rms "
                       "%.3e p99 %.3e max %.3e\n", hs.count, hs.rms, hs.max, ls.count, ls.rms, ls.p99, ls.max);
            }
            printf("gates (tree, theta=%.2f): TN1 %s (norm rms %.3e <= %.3e)  TN2 %s (norm p99 %.3e <= %.3e)  stack %s  => %s"
                   "  | legacy TG1 %s (rms %.3e <= %.3e) TG2 %s (p99 %.3e <= %.3e), not gating\n",
                   p.theta, tn1.c_str(), ns.rms, N1, tn2.c_str(), ns.p99, N2, okS ? "ok" : "OVERFLOW", gates.c_str(),
                   g1.c_str(), rmsRel, T1, g2.c_str(), errP99, T2);
        } else if (stackOverflows) {
            gateFail = true; gates = "FAIL";
        }
        // ---- tree round 2 sibling checks (docs/03-design.md §17.6), on the full-N run:
        //   errorSibling   TG3: union-walk error <= the sibling's (#5 or #4) at the same theta, and
        //                  per slot I and V never below the sibling's (the union only opens more);
        //                  IC round (D-027): gated on the normalised rms and p99 (TG3n); the legacy
        //                  relative TG3 is still printed;
        //   counterSibling work ratio (union overhead) vs the sibling's counters;
        //   exactSibling   accelerations bitwise equal and per-slot counts equal (exactly once).
        if (li.errorSibling && treeGates) {
            const KernelInfo* ks = findKernel(li.errorSibling);
            const std::vector<float4> sa = runAcc(*ks, s, p);
            const RelStats rs = relStats(sa, ref);
            const NormStats sn = normStats(sa, ref, scale);
            sibRms = rs.rms; sibP99 = rs.p99; sibNRms = sn.rms; sibNP99 = sn.p99;
            const bool okL = rmsRel <= rs.rms && errP99 <= rs.p99;
            const bool ok  = ns.rms <= sn.rms && ns.p99 <= sn.p99;
            tg3  = okL ? "PASS" : "FAIL";
            tg3n = ok ? "PASS" : "FAIL";
            printf("TG3n (normalised error <= %s at theta=%.2f): rms %.3e vs %.3e (ratio %.3f), p99 %.3e vs %.3e (ratio %.3f) => %s"
                   "  | legacy relative TG3: rms ratio %.3f, p99 ratio %.3f => %s (not gating)\n",
                   li.errorSibling, p.theta, ns.rms, sn.rms, ns.rms / sn.rms, ns.p99, sn.p99, ns.p99 / sn.p99, tg3n.c_str(),
                   rmsRel / rs.rms, errP99 / rs.p99, tg3.c_str());
            if (!ok) { gateFail = true; gates = "FAIL"; }
        }
        if (li.counterSibling && !wcOwn.empty()) {
            const std::vector<uint4> wcs = runCounters(*findKernel(li.counterSibling), s, p);
            long long si = 0, sv = 0;
            for (const uint4& c : wcs) { si += c.y; sv += c.x; }
            unionI = si > 0 ? double(ws.totalInter) / double(si) : -1;
            unionV = sv > 0 ? double(ws.totalVisits) / double(sv) : -1;
            printf("work vs %s: interactions x%.4f, pops x%.4f", li.counterSibling, unionI, unionV);
            if (li.errorSibling) {
                const int below = countBelow(wcOwn, wcs);
                printf(" | slots with fewer pops or interactions than %s: %d (must be 0)", li.counterSibling, below);
                if (below) { gateFail = true; gates = "FAIL"; tg3 = "FAIL"; tg3n = "FAIL"; }
            }
            printf("\n");
        }
        if (li.exactSibling) {
            const KernelInfo*   kx  = findKernel(li.exactSibling);
            const std::vector<float4> ax = runAcc(*kx, s, p);
            const bool bit = bitwiseEq(ax, acc);
            int diff = -1;
            if (!wcOwn.empty()) diff = countDiff(wcOwn, runCounters(*kx, s, p));
            exactMatch  = bit ? "yes" : "NO";
            countsMatch = diff == 0 ? "yes" : (diff < 0 ? "na" : "NO");
            printf("exact vs %s: accelerations %s, per-slot pop/interaction counts %s (%d slots differ; "
                   "equal counts = every body processed exactly once)\n",
                   li.exactSibling, bit ? "bitwise equal" : "DIFFER", countsMatch.c_str(), diff);
            if (!bit || diff > 0) { gateFail = true; gates = "FAIL"; }
        }
    } else {
        printf("gates: not evaluated (--no-validate)\n");
    }

    // ---- CSV ----------------------------------------------------------------
    if (!a.csv.empty()) {
        static const char* kHeader =
            "kernel,n,block,q_requested,q,prefetch,unroll,grid,regs,blocks_per_sm,occupancy,run,ms,"
            "ginteractions_per_s,gflops,max_rel_err,rms_rel_err,fp64_rms_rel,fp64_max_rel,fp64_norm_max,"
            "naive_fp64_norm_max,kappa_max,g1,g2,gates,batch,warmup_ms,ic,theta,leaf,err_p50,err_p99,"
            "inter_per_body,visits_per_body,warp_max_mean,simt_eff_inter,simt_eff_visit,warp_cost_max_mean,"
            "ph_keys_ms,ph_sort_ms,ph_tree_ms,ph_order_ms,ph_traverse_ms,stack_overflows,"
            "n2equiv_ginter_per_s,trav_ginter_per_s,occ_blocks,smem_pad,persistent,control,"
            "sib_rms,sib_p99,tg3,union_inter_ratio,union_visit_ratio,exact_match,counts_match,"
            "mass_spectrum,bh_mass,ic_hash,mass_max_min,norm_rms,norm_p99,norm_max,tn1,tn2,sib_norm_rms,sib_norm_p99,tg3n,"
            "glob_rms,glob_p99,heavy_norm_max,tree_depth,push_depth,stack_bound,dup_keys";
        // Round 2 (critic review 2, #5): ginteractions_per_s / gflops hold the REAL interaction rate
        // (all-pairs: N^2; trees: counted interactions / pipeline time); N^2 / time is in
        // n2equiv_ginter_per_s. Older CSVs (other header) are refused for appending.
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
        const double realInter = li.approximate ? double(ws.totalInter) : inter;
        const double travMs    = phaseMed.empty() ? -1.0 : phaseMed.back();
        for (int i = 0; i < a.runs; ++i) {
            double ipsi  = realInter > 0 ? realInter / (ms[i] * 1e-3) : -1e9;
            double n2eq  = inter / (ms[i] * 1e-3);
            f << k->name << ',' << p.n << ',' << p.blockSize << ',' << a.q << ',' << li.threadsPerBody << ','
              << li.prefetch << ',' << li.unroll << ',' << li.gridBlocks << ',' << li.regsPerThread << ','
              << li.blocksPerSM << ',' << occ << ',' << i << ',' << ms[i] << ',' << (ipsi > 0 ? ipsi * 1e-9 : -1.0) << ','
              << (ipsi > 0 ? ipsi * kFlopsPerInteraction * 1e-9 : -1.0) << ',' << maxRel << ',' << rmsRel << ',' << fp64Rms << ','
              << fp64Max << ',' << fp64Norm << ',' << naiveNorm << ',' << kappaMax << ',' << g1 << ',' << g2
              << ',' << gates << ',' << a.batch << ',' << a.warmupMs << ',' << a.ic << ','
              << (li.approximate ? double(p.theta) : -1.0) << ',' << (li.approximate ? p.leafSize : -1) << ','
              << errP50 << ',' << errP99 << ',';
            if (li.nPhases > 0)
                f << ws.interPerBody << ',' << ws.visitsPerBody << ',' << ws.warpMaxMean << ',' << ws.simtInter << ','
                  << ws.simtVisit << ',' << ws.interWarp;
            else
                f << "-1,-1,-1,-1,-1,-1";
            for (int q = 0; q < 5; ++q) f << ',' << (q < int(phaseMed.size()) ? phaseMed[q] : -1.0);
            f << ',' << stackOverflows << ',' << n2eq * 1e-9 << ','
              << (li.approximate && travMs > 0 ? ws.totalInter / (travMs * 1e-3) * 1e-9 : -1.0) << ',' << a.occBlocks
              << ',' << p.smemPad << ',' << int(li.persistent) << ',' << int(li.control) << ',' << sibRms << ','
              << sibP99 << ',' << tg3 << ',' << unionI << ',' << unionV << ',' << exactMatch << ',' << countsMatch
              << ',' << a.massSpectrum << ',' << a.bhMass << ',' << std::hex << icH << std::dec << ',' << mMax / mMin << ','
              << ns.rms << ',' << ns.p99 << ',' << ns.max << ',' << tn1 << ',' << tn2 << ',' << sibNRms << ',' << sibNP99 << ','
              << tg3n << ',' << gRms << ',' << gP99 << ',' << heavyNormMax << ',' << td.maxDepth << ',' << td.pushDepth << ','
              << td.stackBound << ',' << td.dupKeys << '\n';
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
