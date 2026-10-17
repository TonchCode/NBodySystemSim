// nbody_bench: times force-kernel variants and validates them against the reference.
//
//   nbody_bench --list
//   nbody_bench --kernel allpairs_tiled --n 65536 --block 256 --runs 10 --csv results/x.csv
//   nbody_bench --kernel allpairs_tiled --n 4096 --steps 1000 --energy
//   nbody_bench --kernel allpairs_tuned --n 65536 --block 128 --unroll 16      (R3 #2)
//   nbody_bench --kernel allpairs_multi --n 4096 --block 256 --q 8             (R3 #3; --q 0 = auto)
//   --warmup-ms 300   time-based warm-up (recommended for small N: clocks ramp from idle)
//   --batch 20        time 20 back-to-back launches per run and report the mean (small N)
//
// Validation: vs allpairs_naive (all bodies), and vs an FP64 host reference on a sample of
// bodies (--fp64-sample, default 512 evenly spaced + the 16 worst-conditioned bodies).
//
// Timing covers the force kernel only (cudaEvent), after warm-up. Environment that CUDA
// can't see (power limit, background load) is captured by the bench-profiler agent.
#include <algorithm>
#include <chrono>
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
    int  q = 0, unroll = 8, fp64Sample = 512, warmupMs = 0, batch = 1;
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
        else if (s == "--q")           a.q = atoi(next());
        else if (s == "--unroll")      a.unroll = atoi(next());
        else if (s == "--fp64-sample") a.fp64Sample = atoi(next());
        else if (s == "--runs")        a.runs = atoi(next());
        else if (s == "--warmup")      a.warmup = atoi(next());
        else if (s == "--warmup-ms")   a.warmupMs = atoi(next());
        else if (s == "--batch")       a.batch = std::max(1, atoi(next()));
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
    p.unroll = a.unroll; p.threadsPerBody = a.q;

    // ---- launch configuration + occupancy (what will actually run) -------------
    LaunchInfo li;
    double occ = -1.0, perSM = -1.0;
    if (k->query) {
        li = k->query(p.n, p);
        cudaDeviceProp dp;
        CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
        const int sms = dp.multiProcessorCount;
        occ   = double(li.blocksPerSM) * li.blockThreads / dp.maxThreadsPerMultiProcessor;
        perSM = double(li.gridBlocks) / sms;
        const int slots = li.blocksPerSM * sms;
        printf("# launch  grid=%d block=%d q=%d unroll=%d  regs=%d smem=%zu B  "
               "resident %d blocks/SM (%.0f%% occupancy)  blocks/SM=%.2f (busiest %d)  waves=%.2f\n",
               li.gridBlocks, li.blockThreads, li.threadsPerBody, li.unroll, li.regsPerThread,
               li.smemPerBlock, li.blocksPerSM, 100.0 * occ, perSM, (li.gridBlocks + sms - 1) / sms,
               slots > 0 ? double(li.gridBlocks) / slots : 0.0);
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

    std::vector<float> ms(a.runs);
    for (int i = 0; i < a.runs; ++i) {
        // --batch K: K back-to-back launches between the events, time / K. For us-scale
        // kernels a single event pair is dominated by event resolution and launch jitter.
        CUDA_CHECK(cudaEventRecord(t0));
        for (int b = 0; b < a.batch; ++b) k->launch(s, p, 0);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], t0, t1));
        ms[i] /= float(a.batch);
    }
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    float  med   = sorted[sorted.size() / 2];
    double inter = double(p.n) * p.n;
    double ips   = inter / (med * 1e-3);
    double gflop = ips * kFlopsPerInteraction * 1e-9;
    printf("kernel=%s n=%d block=%d q=%d unroll=%d runs=%d batch=%d  median=%.3f ms  min=%.3f  max=%.3f  "
           "%.3f Ginteractions/s  %.1f GFLOP/s (@%.0f FLOP/interaction)\n",
           k->name, p.n, p.blockSize, li.threadsPerBody, li.unroll, a.runs, a.batch, med, sorted.front(), sorted.back(),
           ips * 1e-9, gflop, kFlopsPerInteraction);

    // ---- validation vs reference kernel -------------------------------------
    double maxRel = -1.0, rmsRel = -1.0, fp64Rms = -1.0, fp64Max = -1.0, fp64Norm = -1.0;
    if (a.validate) {
        downloadAcc(s, acc);
        const KernelInfo* r = findKernel("allpairs_naive");
        r->launch(s, p, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        downloadAcc(s, ref);
        ErrorStats e = compareAcc(acc, ref);
        maxRel = e.maxRel; rmsRel = e.rmsRel;
        printf("validate vs allpairs_naive: max rel err %.3e  rms %.3e\n", maxRel, rmsRel);

        if (a.fp64Sample > 0) {
            // Evenly spaced sample + the 16 bodies with the smallest |a| (net force nearly
            // cancels: the worst-conditioned bodies, where max-rel spikes).
            std::vector<int> idx;
            const int m = std::min(a.fp64Sample, p.n);
            for (int t = 0; t < m; ++t) idx.push_back(int((long long)t * p.n / m));
            std::vector<int> order(p.n);
            for (int t = 0; t < p.n; ++t) order[t] = t;
            auto mag2 = [&](int t) {
                return double(ref[t].x) * ref[t].x + double(ref[t].y) * ref[t].y + double(ref[t].z) * ref[t].z;
            };
            const int worst = std::min(16, p.n);
            std::partial_sort(order.begin(), order.begin() + worst, order.end(),
                              [&](int x, int y) { return mag2(x) < mag2(y); });
            idx.insert(idx.end(), order.begin(), order.begin() + worst);
            Fp64Sample truth = referenceFp64(pos, p, idx);
            Fp64Error  ev    = compareFp64(acc, truth);
            Fp64Error  en    = compareFp64(ref, truth);
            fp64Rms = ev.rmsRel; fp64Max = ev.maxRel; fp64Norm = ev.normMax;
            printf("validate vs FP64 (%zu bodies, kappa_max %.1e): this rms %.3e max %.3e norm %.3e"
                   "  | naive rms %.3e max %.3e norm %.3e\n",
                   idx.size(), ev.kappaMax, ev.rmsRel, ev.maxRel, ev.normMax,
                   en.rmsRel, en.maxRel, en.normMax);
        }
    }

    // ---- CSV ----------------------------------------------------------------
    if (!a.csv.empty()) {
        bool exists = std::ifstream(a.csv).good();
        std::ofstream f(a.csv, std::ios::app);
        if (!exists)
            f << "kernel,n,block,q,unroll,grid,regs,blocks_per_sm,occupancy,run,ms,ginteractions_per_s,"
                 "gflops,max_rel_err,rms_rel_err,fp64_rms_rel,fp64_max_rel,fp64_norm_max,batch\n";
        for (int i = 0; i < a.runs; ++i) {
            double ipsi = inter / (ms[i] * 1e-3);
            f << k->name << ',' << p.n << ',' << p.blockSize << ',' << li.threadsPerBody << ','
              << li.unroll << ',' << li.gridBlocks << ',' << li.regsPerThread << ',' << li.blocksPerSM
              << ',' << occ << ',' << i << ',' << ms[i] << ',' << ipsi * 1e-9 << ','
              << ipsi * kFlopsPerInteraction * 1e-9 << ',' << maxRel << ',' << rmsRel << ','
              << fp64Rms << ',' << fp64Max << ',' << fp64Norm << ',' << a.batch << '\n';
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
