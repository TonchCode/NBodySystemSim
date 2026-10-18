// R3 #8 (docs/03-design.md §23.2): host CPU topology (P-cores vs E-cores), a pinned worker pool and
// the three schedules of the heterogeneous-CPU load-balancing experiment (static even, dynamic,
// static weighted by measured worker speed).
//
// Why our own pool and not OpenMP: the experiment needs (1) hard affinity of every worker to one
// logical processor of a chosen core type, and (2) per-worker busy times per region. MSVC's OpenMP
// runtimes give neither portably, so a small std::thread pool is the more controllable choice.
// SetThreadAffinityMask changes only our own threads: no system, power or scheduler setting is touched.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include "host.h"

namespace nbody {

// ---- CPU topology ---------------------------------------------------------------------------------------
const HostCpuInfo& hostCpuInfo()
{
    static const HostCpuInfo info = [] {
        HostCpuInfo r;
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        std::vector<char> buf(len);
        struct Core { int eff; std::vector<int> lps; };
        std::vector<Core> cores;
        if (len && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
            for (DWORD off = 0; off < len;) {
                auto* e = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
                if (e->Relationship == RelationProcessorCore) {
                    Core c;
                    c.eff = e->Processor.EfficiencyClass;
                    const KAFFINITY m = e->Processor.GroupMask[0].Mask;   // one processor group (20 LPs < 64)
                    for (int b = 0; b < 64; ++b) if (m & (KAFFINITY(1) << b)) c.lps.push_back(b);
                    cores.push_back(c);
                }
                off += e->Size;
            }
        }
        int maxEff = 0;
        for (const Core& c : cores) maxEff = std::max(maxEff, c.eff);
        // Non-hybrid CPU (all cores in one class): every core counts as a P-core.
        for (const Core& c : cores) {
            const bool p = c.eff == maxEff;
            (p ? r.pCores : r.eCores)++;
            for (int lp : c.lps) (p ? r.pLPs : r.eLPs).push_back(lp);
        }
        // P LPs ordered core-major (core 0 LP a, core 0 LP b, core 1 ...): keep, but also remember the
        // first LP of each P-core for the "one LP per P-core" set.
        r.pLogical = int(r.pLPs.size());
        r.logical  = int(r.pLPs.size() + r.eLPs.size());
        int regs[4];
        char brand[49] = {};
        __cpuid(regs, 0x80000000);
        if (unsigned(regs[0]) >= 0x80000004u) {
            for (int k = 0; k < 3; ++k) { __cpuid(regs, 0x80000002 + k); std::memcpy(brand + 16 * k, regs, 16); }
        }
        r.brand = brand;
        while (!r.brand.empty() && r.brand.back() == ' ') r.brand.pop_back();
        while (!r.brand.empty() && r.brand.front() == ' ') r.brand.erase(r.brand.begin());
        return r;
    }();
    return info;
}

namespace {

std::vector<int> pFirstLPs()
{
    // first LP of each P-core: with HT the P LPs come in pairs (core-major); detect by core size
    const HostCpuInfo& c = hostCpuInfo();
    std::vector<int> out;
    if (c.pCores <= 0) return out;
    const int per = std::max(1, c.pLogical / c.pCores);
    for (int k = 0; k < c.pLogical; k += per) out.push_back(c.pLPs[size_t(k)]);
    return out;
}

std::vector<int> setLPs(int set, int threads)
{
    const HostCpuInfo& c = hostCpuInfo();
    std::vector<int> v;
    switch (set) {
    case 0: v = c.pLPs; v.insert(v.end(), c.eLPs.begin(), c.eLPs.end()); break;
    case 1: v = c.pLPs; break;
    case 2: v = pFirstLPs(); break;
    case 3: v = c.eLPs; break;
    case 4: v = pFirstLPs(); v.insert(v.end(), c.eLPs.begin(), c.eLPs.end()); break;
    default: break;
    }
    if (threads > 0 && size_t(threads) < v.size()) v.resize(size_t(threads));
    return v;
}

bool lpIsP(int lp)
{
    const HostCpuInfo& c = hostCpuInfo();
    return std::find(c.pLPs.begin(), c.pLPs.end(), lp) != c.pLPs.end();
}

} // namespace

const char* cpuSetName(int set)
{
    switch (set) {
    case 0: return "all";
    case 1: return "p";
    case 2: return "p1";
    case 3: return "e";
    case 4: return "cores";
    default: return "?";
    }
}
int parseCpuSet(const std::string& s)
{
    for (int k = 0; k <= 4; ++k) if (s == cpuSetName(k)) return k;
    return -1;
}
const char* cpuSchedName(int s) { return s == 0 ? "static" : (s == 1 ? "dynamic" : (s == 2 ? "weighted" : "?")); }
int parseCpuSched(const std::string& s)
{
    for (int k = 0; k <= 2; ++k) if (s == cpuSchedName(k)) return k;
    return -1;
}
int  cpuSetSize(int set, int threads) { return int(setLPs(set, threads).size()); }
bool cpuSetWorkerIsP(int set, int threads, int w)
{
    const std::vector<int> v = setLPs(set, threads);
    return w >= 0 && size_t(w) < v.size() && lpIsP(v[size_t(w)]);
}

// ---- clock probe ------------------------------------------------------------------------------------------
std::vector<double> probeCpuClocks(const std::vector<int>& lps, double ms)
{
    std::vector<double> ghz(lps.size(), 0.0);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> th;
    volatile unsigned long long seedA = 0x9E3779B97F4A7C15ull, seedB = 0xD1B54A32D192ED03ull;
    for (size_t k = 0; k < lps.size(); ++k)
        th.emplace_back([&, k] {
            SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << lps[k]);
            const unsigned long long a = seedA, b = seedB;
            unsigned long long x = a ^ k;
            ready.fetch_add(1);
            while (!go.load()) _mm_pause();
            // 16 dependent one-cycle integer ops per iteration (add, xor alternating; the compiler cannot
            // fold them), so cycles = 16 * iterations; the loop counter runs in parallel.
            const auto t0 = std::chrono::steady_clock::now();
            long long it = 0;
            double el = 0;
            do {
                for (int r = 0; r < 4096; ++r) {
                    x = (x + a) ^ b; x = (x + a) ^ b; x = (x + a) ^ b; x = (x + a) ^ b;
                    x = (x + a) ^ b; x = (x + a) ^ b; x = (x + a) ^ b; x = (x + a) ^ b;
                }
                it += 4096;
                el = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            } while (el < ms);
            ghz[k] = 16.0 * double(it) / (el * 1e6) + double(x & 1) * 0.0;   // x kept live
            if (x == 0x123456789ull) printf(" ");                          // never true; defeats DCE
        });
    while (ready.load() < int(lps.size())) std::this_thread::yield();
    go.store(true);
    for (auto& t : th) t.join();
    return ghz;
}

namespace host {

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

namespace {

class Pool {
public:
    ~Pool() { stop(); }
    void configure(const std::vector<int>& lps)
    {
        if (lps == lps_ && !th_.empty()) return;
        stop();
        lps_ = lps;
        quit_ = false;
        // the generation is read HERE, before any run() can bump it (a worker that read it itself could
        // miss the first job and wait forever)
        const unsigned long long g0 = gen_.load();
        for (size_t w = 0; w < lps_.size(); ++w) th_.emplace_back([this, w, g0] { loop(int(w), g0); });
    }
    int  size() const { return int(lps_.size()); }
    int  lp(int w) const { return lps_[size_t(w)]; }
    void run(const std::function<void(int)>& f)
    {
        job_ = &f;
        remaining_.store(size(), std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(m_);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        // The dispatcher is not pinned, so with every LP holding a pinned worker it shares an LP with one of
        // them: spinners yield (SwitchToThread) every 32 pauses so neither starves the other for a quantum.
        for (int spin = 0; remaining_.load(std::memory_order_acquire) != 0; ++spin) {
            if (spin < 4000) { if ((spin & 31) == 31) SwitchToThread(); else _mm_pause(); continue; }
            std::unique_lock<std::mutex> lk(m_);
            cvDone_.wait_for(lk, std::chrono::microseconds(200), [&] { return remaining_.load() == 0; });
        }
        job_ = nullptr;
    }

private:
    void stop()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            gen_.fetch_add(1);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
    }
    void loop(int w, unsigned long long seen)
    {
        SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << lps_[size_t(w)]);
        for (;;) {
            unsigned long long g;
            for (int spin = 0; (g = gen_.load(std::memory_order_acquire)) == seen; ++spin) {
                if (spin < 20000) { if ((spin & 31) == 31) SwitchToThread(); else _mm_pause(); continue; }   // then block
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_.load() != seen; });
            }
            seen = g;
            if (quit_) return;
            (*job_)(w);
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(m_);
                cvDone_.notify_all();
            }
        }
    }
    std::vector<int>                  lps_;
    std::vector<std::thread>          th_;
    std::atomic<unsigned long long>   gen_{0};
    std::atomic<int>                  remaining_{0};
    const std::function<void(int)>*   job_ = nullptr;
    std::mutex                        m_;
    std::condition_variable           cv_, cvDone_;
    bool                              quit_ = false;
};

Pool& pool()
{
    static Pool p;
    return p;
}
int  gSet = -1, gThreads = -1;
struct Speeds { std::vector<double> s; bool measured = false; };
std::map<int, Speeds>& speeds()   // site -> per-worker items/ms (EMA), for the weighted schedule
{
    static std::map<int, Speeds> m;
    return m;
}
std::mutex gRunMutex;   // one parallel region at a time (the phase split's background job vs the bench)

} // namespace

void configureLocked(int set, int threads)
{
    if (set == gSet && threads == gThreads && pool().size() > 0) return;
    std::vector<int> lps = setLPs(set, threads);
    if (lps.empty()) { fprintf(stderr, "host pool: CPU set '%s' is empty on this machine\n", cpuSetName(set)); exit(2); }
    pool().configure(lps);
    gSet = set; gThreads = threads;
    speeds().clear();
}
void configurePool(int set, int threads)
{
    std::lock_guard<std::mutex> lock(gRunMutex);
    configureLocked(set, threads);
}
int  poolWorkers() { return pool().size(); }
bool poolWorkerIsP(int w) { return lpIsP(pool().lp(w)); }

namespace {
void runFor(long long n, int sched, long long chunk, int site, CpuStats* stats,
            const std::function<void(long long, long long, int)>& fn)
{
    if (n <= 0) return;
    Pool& P = pool();
    const int W = P.size();
    std::vector<double>    busy(size_t(W), 0.0);
    std::vector<long long> items(size_t(W), 0);
    std::vector<long long> start(size_t(W) + 1, 0);
    if (sched == kWeighted) {
        Speeds& S = speeds()[site];
        if (S.s.size() != size_t(W) || !S.measured) S.s.assign(size_t(W), 1.0);
        const std::vector<double>& sp = S.s;
        double tot = 0;
        for (double s : sp) tot += s;
        double acc = 0;
        for (int w = 0; w < W; ++w) {
            start[size_t(w)] = (long long)(double(n) * acc / tot + 0.5);
            acc += sp[size_t(w)];
        }
        start[size_t(W)] = n;
        for (int w = 1; w <= W; ++w) start[size_t(w)] = std::max(start[size_t(w)], start[size_t(w) - 1]);
    } else {
        for (int w = 0; w <= W; ++w) start[size_t(w)] = n * w / W;
    }
    const long long ch = chunk > 0 ? chunk : std::max<long long>(1, n / (long long(W) * 32));
    std::atomic<long long> next{0};
    const double t0 = nowMs();
    P.run([&](int w) {
        const double b0 = nowMs();
        if (sched == kDynamic) {
            for (;;) {
                const long long b = next.fetch_add(ch, std::memory_order_relaxed);
                if (b >= n) break;
                const long long e = std::min(n, b + ch);
                fn(b, e, w);
                items[size_t(w)] += e - b;
            }
        } else if (start[size_t(w)] < start[size_t(w) + 1]) {
            fn(start[size_t(w)], start[size_t(w) + 1], w);
            items[size_t(w)] += start[size_t(w) + 1] - start[size_t(w)];
        }
        busy[size_t(w)] = nowMs() - b0;
    });
    const double wall = nowMs() - t0;
    // weighted schedule memory: every schedule updates the per-site worker speeds
    {
        Speeds& S = speeds()[site];
        const bool first = !S.measured || S.s.size() != size_t(W);
        if (first) S.s.assign(size_t(W), 0.0);
        for (int w = 0; w < W; ++w)
            if (items[size_t(w)] > 0 && busy[size_t(w)] > 1e-4) {
                const double v = double(items[size_t(w)]) / busy[size_t(w)];
                S.s[size_t(w)] = first ? v : 0.5 * S.s[size_t(w)] + 0.5 * v;
            }
        double mean = 0; int k = 0;
        for (double v : S.s) if (v > 0.0) { mean += v; ++k; }
        mean = k ? mean / k : 1.0;
        for (double& v : S.s) if (!(v > 0.0)) v = mean;   // a worker that got no items keeps the mean speed
        S.measured = true;
    }
    if (stats) {
        double mx = 0, sum = 0, ps = 0, es = 0;
        int np = 0, ne = 0;
        for (int w = 0; w < W; ++w) {
            mx = std::max(mx, busy[size_t(w)]);
            sum += busy[size_t(w)];
            if (lpIsP(P.lp(w))) { ps += busy[size_t(w)]; ++np; stats->pItems += items[size_t(w)]; }
            else                { es += busy[size_t(w)]; ++ne; stats->eItems += items[size_t(w)]; }
        }
        stats->regions++;
        stats->workers = W; stats->pWorkers = np; stats->eWorkers = ne;
        stats->wallMs += wall;
        stats->sumMaxBusyMs += mx;
        stats->sumMeanBusyMs += sum / W;
        stats->sumPBusyMs += np ? ps / np : 0;
        stats->sumEBusyMs += ne ? es / ne : 0;
    }
}

} // namespace

void parallelFor(long long n, int sched, long long chunk, int site, CpuStats* stats,
                 const std::function<void(long long, long long, int)>& fn)
{
    std::lock_guard<std::mutex> lock(gRunMutex);
    if (pool().size() == 0) configureLocked(0, 0);
    runFor(n, sched, chunk, site, stats, fn);
}

int rsqrtCoreTypeMismatches(int samples)
{
    const HostCpuInfo& c = hostCpuInfo();
    if (c.pLPs.empty() || c.eLPs.empty()) return -1;
    const int m = (samples + 7) / 8 * 8;
    std::vector<float> x(static_cast<size_t>(m)), rp(static_cast<size_t>(m)), re(static_cast<size_t>(m));
    for (int k = 0; k < m; ++k) {   // mantissas across [1, 4) and exponents 2^-20 .. 2^20
        const unsigned bits = (unsigned(0x3f800000u + unsigned(k % (1 << 21)) * 4u)) + (unsigned((k >> 21) % 40) << 23);
        float v;
        std::memcpy(&v, &bits, 4);
        x[size_t(k)] = std::ldexp(v, -20 + 2 * ((k * 7) % 21));
    }
    auto run = [&](int lp, std::vector<float>& out) {
        std::thread t([&] {
            SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << lp);
            for (int k = 0; k < m; k += 8) _mm256_storeu_ps(&out[size_t(k)], _mm256_rsqrt_ps(_mm256_loadu_ps(&x[size_t(k)])));
        });
        t.join();
    };
    run(c.pLPs[0], rp);
    run(c.eLPs[0], re);
    int d = 0;
    for (int k = 0; k < samples; ++k) d += std::memcmp(&rp[size_t(k)], &re[size_t(k)], 4) != 0;
    return d;
}

void parallelFor(const Params& p, long long n, int site, const std::function<void(long long, long long, int)>& fn)
{
    std::lock_guard<std::mutex> lock(gRunMutex);
    configureLocked(p.cpuSet, p.cpuThreads);
    runFor(n, p.cpuSched, p.cpuChunk, site, p.cpuStats, fn);
}

} // namespace host
} // namespace nbody
