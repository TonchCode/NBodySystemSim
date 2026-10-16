// nbody_bench: times force-kernel variants and validates them against the reference.
//
//   nbody_bench --list
//   nbody_bench --kernel allpairs_tiled --n 65536 --block 256 --runs 10 --csv results/x.csv
//   nbody_bench --kernel allpairs_tiled --n 4096 --steps 1000 --energy
//
// Timing covers the force kernel only (cudaEvent), after warm-up. Environment that CUDA
// can't see (power limit, background load) is captured by the bench-profiler agent.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nbody.h"

namespace {

struct Args {
    std::string kernel = "allpairs_tiled";
    std::string csv;
    int  n = 1 << 14, block = 256, runs = 10, warmup = 3, steps = 0;
    unsigned seed = 42;
    bool list = false, validate = true, energy = false;
    float dt = 1e-3f, softening = 1e-2f;
};

Args parse(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", s.c_str()); exit(2); }
            return argv[++i];
        };
        if      (s == "--list")        a.list = true;
        else if (s == "--kernel")      a.kernel = next();
        else if (s == "--n")           a.n = atoi(next());
        else if (s == "--block")       a.block = atoi(next());
        else if (s == "--runs")        a.runs = atoi(next());
        else if (s == "--warmup")      a.warmup = atoi(next());
        else if (s == "--seed")        a.seed = unsigned(atoi(next()));
        else if (s == "--csv")         a.csv = next();
        else if (s == "--steps")       a.steps = atoi(next());
        else if (s == "--dt")          a.dt = float(atof(next()));
        else if (s == "--softening")   a.softening = float(atof(next()));
        else if (s == "--no-validate") a.validate = false;
        else if (s == "--energy")      a.energy = true;
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); exit(2); }
    }
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

} // namespace

int main(int argc, char** argv)
{
    using namespace nbody;
    Args a = parse(argc, argv);

    if (a.list) {
        for (const auto& k : kernelRegistry()) printf("%-20s %s\n", k.name, k.description);
        return 0;
    }
    const KernelInfo* k = findKernel(a.kernel);
    if (!k) { fprintf(stderr, "unknown kernel '%s' (use --list)\n", a.kernel.c_str()); return 2; }

    printDevice();
    Params p;
    p.n = a.n; p.blockSize = a.block; p.dt = a.dt; p.softening = a.softening;

    std::vector<float4> pos, vel, acc, ref;
    makePlummer(p.n, a.seed, pos, vel);
    DeviceState s = allocate(p.n);
    upload(s, pos, vel);

    // ---- timing ------------------------------------------------------------
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < a.warmup; ++i) k->launch(s, p, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ms(a.runs);
    for (int i = 0; i < a.runs; ++i) {
        CUDA_CHECK(cudaEventRecord(t0));
        k->launch(s, p, 0);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    float  med   = sorted[sorted.size() / 2];
    double inter = double(p.n) * p.n;
    double ips   = inter / (med * 1e-3);
    double gflop = ips * kFlopsPerInteraction * 1e-9;
    printf("kernel=%s n=%d block=%d runs=%d  median=%.3f ms  min=%.3f  max=%.3f  "
           "%.3f Ginteractions/s  %.1f GFLOP/s (@%.0f FLOP/interaction)\n",
           k->name, p.n, p.blockSize, a.runs, med, sorted.front(), sorted.back(),
           ips * 1e-9, gflop, kFlopsPerInteraction);

    // ---- validation vs reference kernel -------------------------------------
    double maxRel = -1.0, rmsRel = -1.0;
    if (a.validate) {
        downloadAcc(s, acc);
        const KernelInfo* r = findKernel("allpairs_naive");
        r->launch(s, p, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        downloadAcc(s, ref);
        ErrorStats e = compareAcc(acc, ref);
        maxRel = e.maxRel; rmsRel = e.rmsRel;
        printf("validate vs allpairs_naive: max rel err %.3e  rms %.3e\n", maxRel, rmsRel);
    }

    // ---- CSV ----------------------------------------------------------------
    if (!a.csv.empty()) {
        bool exists = std::ifstream(a.csv).good();
        std::ofstream f(a.csv, std::ios::app);
        if (!exists) f << "kernel,n,block,run,ms,ginteractions_per_s,gflops,max_rel_err,rms_rel_err\n";
        for (int i = 0; i < a.runs; ++i) {
            double ipsi = inter / (ms[i] * 1e-3);
            f << k->name << ',' << p.n << ',' << p.blockSize << ',' << i << ',' << ms[i] << ','
              << ipsi * 1e-9 << ',' << ipsi * kFlopsPerInteraction * 1e-9 << ',' << maxRel << ','
              << rmsRel << '\n';
        }
    }

    // ---- integration + energy drift -----------------------------------------
    if (a.steps > 0) {
        upload(s, pos, vel);
        k->launch(s, p, 0);   // prime acc for KDK
        Energy e0{};
        if (a.energy) e0 = computeEnergy(pos, vel, p);
        for (int i = 0; i < a.steps; ++i) step(s, p, *k, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (a.energy) {
            download(s, pos, vel);
            Energy e1 = computeEnergy(pos, vel, p);
            printf("energy: E0=%.6f  E1=%.6f  rel drift=%.3e over %d steps (dt=%g, eps=%g)\n",
                   e0.total, e1.total, (e1.total - e0.total) / std::abs(e0.total), a.steps,
                   p.dt, p.softening);
        } else {
            printf("integrated %d steps\n", a.steps);
        }
    }

    release(s);
    return 0;
}
