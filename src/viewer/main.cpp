// nbody_viewer: real-time CUDA-OpenGL interop view of the simulation, with a Dear ImGui control panel
// and live statistics (D-032).
//
// The simulation state never leaves the GPU. Each frame: run K leapfrog steps with the selected force
// kernel, then copy pos/vel device-to-device into two GL vertex buffers that are mapped into CUDA
// (cudaGraphicsGLRegisterBuffer). That copy is VRAM->VRAM, not over PCIe (one copy engine,
// docs/01-hardware.md). A third, GL-only buffer holds one origin tag per body (which galaxy / cluster /
// bulge it was born in); bodies never change index, so it is uploaded once per IC. nbody_core is unchanged
// except that makeIC can also return those tags; the bench measures the same kernels.
//
//   nbody_viewer [--n 32768] [--kernel bh_warp] [--dt 0.002] [--block 256] [--seed 42]
//                [--frames F --screenshot out.bmp]   (run F frames, save the last one, exit)
//                [--ic plummer|collision|core|disk|galaxies|cube_shell|cube_edges|uniform|merger3|flyby|unequal]
//                [--mass-spectrum equal|salpeter|heavy-few] [--bh-mass 0.01]
//                [--theta 0.5 --leaf 16] [--dist 4 --yaw 0.6 --pitch 0.35]   (camera)
//                [--steps-per-frame 2 --vsync 1 --mass-color 1 --brightness 1]   (figure runs: simulated time
//                 at the last frame = frames x steps-per-frame x dt, printed on exit)
//                [--color speed|mass|origin] [--panel 0|1] [--ui-scale 1.0] [--point-size 1.0] [--auto-rotate 0|1]
//                [--lb 0|1 --lb-every 30]   (load-balance measurement: instrumented twin every K frames)
//                [--well-mass 0.5 --well-eps 0.1]   (mouse gravity well)
//                [--maximize 1]   (start maximised: more room around the panels; run.bat uses it)
//                [--energy-check 1]   (startup: GPU energy vs the host FP64 computeEnergy, N <= 32768)
//                [--test-keys "F:act,F:act,..."]   (scripted actions for smoke tests, see applyAction())
//   merger3 | flyby | unequal are viewer-only presets (nbody::makePreset; the bench does not know them).
//   The panel is shown by default, hidden by default when --screenshot is given (clean figures).
//
// Controls: LMB drag orbit · wheel zoom · RMB hold gravity well · F1 / Tab panel · 1..9 kernel · Space pause
//           +/- steps per frame · PageUp / . double N · PageDown / , halve N · I next IC · H next mass spectrum
//           M colour speed/mass · C cycle colour (speed, mass, origin) · [ ] brightness · R reset · V vsync
//           Esc quit. While the panel is hovered or a field is being edited, ImGui gets the mouse/keyboard.
// N limits (TDR margin, CLAUDE.md rule 4): all-pairs kernels <= 262144, tree kernels <= 4194304.
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "nbody.h"
#include "viewer_kernels.h"

namespace {

// ---- minimal column-major mat4 ---------------------------------------------
struct Mat4 { float m[16]; };

Mat4 mul(const Mat4& a, const Mat4& b)
{
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}

Mat4 perspective(float fovyRad, float aspect, float zn, float zf)
{
    float f = 1.f / std::tan(fovyRad * 0.5f);
    Mat4 r{};
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1.f;
    r.m[14] = 2.f * zf * zn / (zn - zf);
    return r;
}

// Camera basis for an eye at e looking at the origin with up = +y: forward f, right s = f x up, up u = s x f.
struct Basis { float e[3], f[3], s[3], u[3]; };

Basis cameraBasis(float ex, float ey, float ez)
{
    Basis b{};
    b.e[0] = ex; b.e[1] = ey; b.e[2] = ez;
    float fx = -ex, fy = -ey, fz = -ez;
    float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0.f - fz * 1.f, sy = fz * 0.f - fx * 0.f, sz = fx * 1.f - fy * 0.f;   // f x up
    float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
    sx /= sl; sy /= sl; sz /= sl;
    b.f[0] = fx; b.f[1] = fy; b.f[2] = fz;
    b.s[0] = sx; b.s[1] = sy; b.s[2] = sz;
    b.u[0] = sy * fz - sz * fy; b.u[1] = sz * fx - sx * fz; b.u[2] = sx * fy - sy * fx;   // s x f
    return b;
}

Mat4 lookAt(const Basis& b)   // target = origin, up = +y
{
    const float *e = b.e, *f = b.f, *s = b.s, *u = b.u;
    Mat4 r{};
    r.m[0] = s[0];  r.m[4] = s[1];  r.m[8]  = s[2];
    r.m[1] = u[0];  r.m[5] = u[1];  r.m[9]  = u[2];
    r.m[2] = -f[0]; r.m[6] = -f[1]; r.m[10] = -f[2];
    r.m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
    r.m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
    r.m[14] = (f[0] * e[0] + f[1] * e[1] + f[2] * e[2]);
    r.m[15] = 1.f;
    return r;
}

constexpr float kFovY = 0.785398f;   // 45 degrees

// ---- shaders ------------------------------------------------------------------
// uColorMode 0: speed (slow = deep blue, fast = warm white). 1: speed + mass: bodies heavier than the mean
// mass (uMassRef) shift towards red and grow, log-scaled (3x the mean starts to show, 100x is full red).
// 2: origin: one palette per galaxy / cluster (aTag >> 4: blue-white, gold, green, magenta), bulges paler,
// faster bodies whiter, the central massive bodies drawn as large white points.
const char* kVertexSrc = R"(#version 330 core
layout(location = 0) in vec4 aPos;   // xyz + mass
layout(location = 1) in vec4 aVel;
layout(location = 2) in uint aTag;   // group << 4 | role (nbody::originTag)
uniform mat4  uMVP;
uniform float uPointScale;
uniform float uSizeMul;
uniform float uSpeedRef;
uniform float uMassRef;
uniform int   uColorMode;
out vec3 vColor;
void main() {
    gl_Position  = uMVP * vec4(aPos.xyz, 1.0);
    float t = clamp(length(aVel.xyz) / uSpeedRef, 0.0, 1.0);
    vec3 speedCol = mix(vec3(0.15, 0.35, 1.0), vec3(1.0, 0.75, 0.45), t) + vec3(0.25 * t * t);
    float h = 0.0;
    if (uColorMode == 2) {
        const vec3 pal[4] = vec3[4](vec3(0.30, 0.55, 1.00), vec3(1.00, 0.68, 0.20),
                                    vec3(0.30, 1.00, 0.50), vec3(1.00, 0.30, 0.80));
        uint g = (aTag >> 4u) & 3u, r = aTag & 15u;
        float lift = 0.45 * t;
        if (r == 2u) lift += 0.30;                 // bulge
        vColor = mix(pal[g], vec3(1.0), clamp(lift, 0.0, 0.8));
        if (r == 3u) { vColor = vec3(4.0); h = 1.0; }   // central massive body
    } else {
        vColor = speedCol;
        if (uColorMode == 1) h = clamp(log(max(aPos.w / uMassRef, 1.0) / 3.0) / log(33.3), 0.0, 1.0);
        vColor = mix(vColor, vec3(4.0, 0.6, 0.15), h);
    }
    gl_PointSize = clamp(uPointScale * uSizeMul / gl_Position.w * (1.0 + 2.0 * h), 1.0, (8.0 + 16.0 * h) * uSizeMul);
})";

const char* kFragmentSrc = R"(#version 330 core
in vec3 vColor;
uniform float uIntensity;
out vec4 oColor;
void main() {
    float d = length(gl_PointCoord - vec2(0.5)) * 2.0;
    if (d > 1.0) discard;
    float a = (1.0 - d * d) * uIntensity;
    oColor = vec4(vColor * a, 1.0);   // additive blending: dense regions glow
})";

GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "shader compile error:\n%s\n", log);
        exit(1);
    }
    return s;
}

GLuint linkProgram()
{
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, kVertexSrc));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, kFragmentSrc));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "program link error:\n%s\n", log);
        exit(1);
    }
    return p;
}

constexpr int kMinN = 256, kMaxNAllPairs = 1 << 18, kMaxNTree = 1 << 22;
constexpr int kMaxNEnergy = 1 << 20;   // O(N^2) energy: ~1-2 s at 1M (chunked), refused above
constexpr int kHist = 240;             // rolling plot length

// Per-warp work statistics from the instrumented traversal's counters (uint4 per thread slot: pops,
// interactions, leader pops, leader interactions). SAME definitions as nbody_bench's warpStats
// (src/bench/main.cpp, docs/03-design.md §14); the bench stays the source of truth for reported numbers.
struct WarpStats {
    double interPerBody = 0, visitsPerBody = 0;
    double warpMaxMean = 0, warpMaxMeanMedian = 0;   // sum_w max_lane(I) / sum_w mean_lane(I); median over warps
    double simtInter = 0, simtVisit = 0;             // sum I / (32 sum leaderI); the same for pops
    double interWarp = 0;                            // max_w max_lane(I) / mean_w max_lane(I)
    long long totalInter = 0, totalVisits = 0;
    int warps = 0;
};

WarpStats warpStats(const std::vector<uint4>& wc, int n)
{
    WarpStats w;
    double sumMax = 0, sumMean = 0, maxOfMax = 0;
    long long leadI = 0, leadV = 0;
    std::vector<double> ratios;
    ratios.reserve(size_t(n / 32 + 1));
    for (int b = 0; b < n; b += 32) {
        const int lanes = std::min(32, n - b);
        unsigned long long si = 0, sv = 0, lv = 0, li = 0;
        unsigned mx = 0;
        for (int t = b; t < b + lanes; ++t) {
            si += wc[t].y; sv += wc[t].x; lv += wc[t].z; li += wc[t].w;
            mx = std::max(mx, wc[t].y);
        }
        const double mean = double(si) / lanes;
        sumMax += mx; sumMean += mean; maxOfMax = std::max(maxOfMax, double(mx));
        if (mean > 0) ratios.push_back(mx / mean);
        w.totalInter += si; w.totalVisits += sv; leadI += li; leadV += lv;
        ++w.warps;
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

struct Ring {
    float v[kHist] = {};
    int   head = 0, count = 0;
    void push(float x) { v[head] = x; head = (head + 1) % kHist; count = std::min(count + 1, kHist); }
    void clear() { head = count = 0; }
    // PlotLines wants a contiguous array with an offset; with count < kHist plot only the filled part.
    void plot(const char* label, const char* overlay, float lo, float hi, float height) const
    {
        if (count == 0) { ImGui::TextDisabled("%s: no samples yet", label); return; }
        if (count < kHist) ImGui::PlotLines(label, v, count, 0, overlay, lo, hi, ImVec2(0, height));
        else ImGui::PlotLines(label, v, kHist, head, overlay, lo, hi, ImVec2(0, height));
    }
    float maxv() const { float m = 0; for (int i = 0; i < count; ++i) m = std::max(m, v[i]); return m; }
};

// ---- app state ----------------------------------------------------------------
struct App {
    nbody::Params            params;
    nbody::DeviceState       state;
    const nbody::KernelInfo* kernel = nullptr;
    bool                     tree = false;    // cached isTree(kernel)
    nbody::LaunchInfo        li;              // cached query of the current kernel (phase names)
    std::vector<float4>      pos0, vel0;
    std::vector<unsigned char> tags;
    unsigned                 seed = 42;
    std::string              ic = "plummer";
    nbody::ICOptions         icOpt;
    std::vector<std::string> icList, presetList, allIC, spList;

    int    stepsPerFrame = 2;
    bool   paused = false, vsync = true, stepOnce = false;
    int    colorMode = 1;   // 0 speed, 1 mass, 2 origin
    float  yaw = 0.6f, pitch = 0.35f, dist = 4.0f, yaw0 = 0.6f, pitch0 = 0.35f, dist0 = 4.0f;
    bool   autoRotate = false;
    float  rotSpeed = 0.15f;   // rad / s
    float  intensity = 1.0f, baseIntensity = 1.0f, pointSize = 1.0f, uiScale = 1.0f;
    bool   dragging = false;
    double lastX = 0, lastY = 0;
    bool   statsReset = false;   // set when steps/frame, the kernel, N or the IC change: restart the ms/step window
    // pending changes requested by keys / panel, applied in the main loop
    int    pendingN = 0;
    int    pendingIC = 0, pendingSpectrum = 0;
    bool   pendingRebuild = false;          // panel Apply / scripted: rebuild from the req* fields
    std::string reqIC; nbody::ICOptions reqOpt; int reqN = 0; unsigned reqSeed = 42;
    // panel staging (edited in the Scenario section, applied with Apply)
    int    stIC = 0, stSpec = 0, stN = 0, stSeed = 42; float stBh = 0.01f;
    bool   showPanel = true;
    bool   openWell = false, openEnergy = false;   // scripted: expand these panel sections once
    std::string message;   // last status / refusal message, shown in the panel
    // GL interop
    GLuint vbo[3] = {0, 0, 0};   // pos, vel (CUDA-mapped), tag (GL only)
    cudaGraphicsResource* res[2] = {nullptr, nullptr};
    int    vboN = 0;
    // real interaction count per force evaluation (tree: counted; all-pairs: N^2), work statistics
    double interPerEval = -1.0;
    double countTimer = -1e9;
    double simTime = 0.0;   // simulated time since the last reset
    uint4* dWc = nullptr; int wcN = 0;
    WarpStats ws; bool wsValid = false;
    nbody::TreeDepth td; bool tdValid = false;
    int    overflows = 0;
    double measureMs = 0;
    bool   lbOn = false; int lbEvery = 30, framesSinceLB = 1 << 30;
    // phase times (tree kernels): events around the phases of the first step of each frame
    cudaEvent_t phaseEv[9] = {};
    float  phaseMs[8] = {}, phaseEma[8] = {};
    bool   phaseValid = false;
    // energy (on request, GPU all-pairs)
    bool   e0Valid = false, eValid = false, energyAuto = false, energyReq = false;
    viewer::EnergyResult e0, eLast;
    double e0Time = 0, eTime = 0, energyTimer = -1e9;
    bool   e0Modified = false;   // the reference was taken after a user interaction
    // gravity well
    bool   wellEnabled = true, wellActive = false, wellScripted = false;
    float  wellMass = 0.5f, wellEps = 0.1f, wellFx = 0.5f, wellFy = 0.5f;
    float3 wellPos = make_float3(0, 0, 0);
    bool   userModified = false;   // the well acted since the last reset: energy is not conserved
    // stats display
    double fps = 0, msPerStep = 0;
    Ring   histMs, histSimt, histWarp;
};

bool queryTree(const nbody::KernelInfo* k, const nbody::Params& p)
{
    return k->query && k->query(1024, p).approximate;
}

int maxNFor(const nbody::KernelInfo* k, const nbody::Params& p) { return queryTree(k, p) ? kMaxNTree : kMaxNAllPairs; }

void refreshKernelInfo(App& a)
{
    a.tree = queryTree(a.kernel, a.params);
    a.li = a.kernel->query ? a.kernel->query(std::max(a.state.n, 1024), a.params) : nbody::LaunchInfo{};
    a.phaseValid = false;
    for (float& x : a.phaseEma) x = 0.f;
    a.wsValid = a.tdValid = false;
    a.histSimt.clear(); a.histWarp.clear(); a.histMs.clear();
}

void resetSim(App& a)
{
    nbody::upload(a.state, a.pos0, a.vel0);
    a.kernel->launch(a.state, a.params, 0);   // prime acc for KDK
    CUDA_CHECK(cudaDeviceSynchronize());
    a.statsReset = true;
    a.countTimer = -1e9;
    a.framesSinceLB = 1 << 30;
    a.simTime = 0.0;
    a.e0Valid = a.eValid = false;
    a.userModified = false;
    a.histMs.clear();
}

void syncStage(App& a)
{
    a.stIC   = int(std::find(a.allIC.begin(), a.allIC.end(), a.ic) - a.allIC.begin());
    a.stSpec = int(std::find(a.spList.begin(), a.spList.end(), a.icOpt.massSpectrum) - a.spList.begin());
    a.stN    = a.state.n;
    a.stSeed = int(a.seed);
    a.stBh   = float(a.icOpt.bhMass);
}

// (Re)size the two interop vertex buffers to n bodies: unregister, reallocate, register again.
void resizeVbos(App& a, int n)
{
    for (int b = 0; b < 2; ++b)
        if (a.res[b]) { CUDA_CHECK(cudaGraphicsUnregisterResource(a.res[b])); a.res[b] = nullptr; }
    for (int b = 0; b < 2; ++b) {
        glBindBuffer(GL_ARRAY_BUFFER, a.vbo[b]);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(n) * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (int b = 0; b < 2; ++b)
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&a.res[b], a.vbo[b], cudaGraphicsRegisterFlagsWriteDiscard));
    a.vboN = n;
}

bool isPreset(const App& a, const std::string& ic)
{
    return std::find(a.presetList.begin(), a.presetList.end(), ic) != a.presetList.end();
}

// New IC / N / seed: regenerate on the host, reallocate the device state and the VBOs if N changed, keep the camera.
bool rebuild(App& a, const std::string& ic, const nbody::ICOptions& opt, int n, unsigned seed)
{
    std::vector<float4> p, v;
    std::vector<unsigned char> t;
    const bool ok = isPreset(a, ic) ? nbody::makePreset(ic, n, seed, p, v, opt, &t)
                                    : nbody::makeIC(ic, n, seed, p, v, opt, &t);
    if (!ok) { fprintf(stderr, "cannot make IC %s\n", ic.c_str()); a.message = "cannot make IC " + ic; return false; }
    a.pos0.swap(p); a.vel0.swap(v); a.tags.swap(t);
    a.ic = ic; a.icOpt = opt; a.seed = seed;
    if (a.state.n != n) {
        nbody::release(a.state);
        a.state = nbody::allocate(n);
        a.params.n = n;
    }
    if (a.vbo[0] && a.vboN != n) resizeVbos(a, n);
    if (a.vbo[2]) {   // origin tags: static per body index (no kernel reorders the state arrays)
        glBindBuffer(GL_ARRAY_BUFFER, a.vbo[2]);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(n), a.tags.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    // brightness auto-scales with N so 4k and 1M bodies both look reasonable
    a.baseIntensity = std::clamp(std::sqrt(32768.f / float(n)) * 0.6f, 0.03f, 1.0f);
    refreshKernelInfo(a);
    resetSim(a);
    syncStage(a);
    const double adv = nbody::dtAdvisory(a.pos0, a.params.softening, a.params.G);
    printf("ic=%s mass_spectrum=%s N=%d seed=%u ic_hash=%016llx%s\n", ic.c_str(), opt.massSpectrum.c_str(), n, seed,
           nbody::icHash(a.pos0, a.vel0), a.params.dt > adv ? "  (dt above the heavy-body advisory: orbits near the heaviest body are not converged)" : "");
    return true;
}

bool setKernel(App& a, const nbody::KernelInfo* k)
{
    if (k == a.kernel) return true;
    if (a.state.n > maxNFor(k, a.params)) {
        char b[256];
        snprintf(b, sizeof b, "%s is limited to N <= %d (TDR margin); lower N first", k->name, maxNFor(k, a.params));
        printf("%s\n", b);
        a.message = b;
        return false;
    }
    a.kernel = k;
    a.statsReset = true; a.countTimer = -1e9; a.framesSinceLB = 1 << 30;
    refreshKernelInfo(a);
    a.message.clear();
    return true;
}

// Real interactions of one force evaluation: one instrumented launch (bitwise-equal accelerations, so the
// integrator state is unchanged) with zeroed work counters, summed on the host. Tree kernels also give
// the per-warp statistics and (withDepth) the depth diagnostics of the tree that launch built.
// Runs OUTSIDE the timed window, so ms/step is always the production kernel.
void measureWork(App& a, bool withDepth)
{
    const int n = a.state.n;
    if (!a.tree) { a.interPerEval = double(n) * n; a.wsValid = a.tdValid = false; return; }
    const double t0 = glfwGetTime();
    if (a.wcN != n) {
        if (a.dWc) CUDA_CHECK(cudaFree(a.dWc));
        CUDA_CHECK(cudaMalloc(&a.dWc, size_t(n) * sizeof(uint4)));
        a.wcN = n;
    }
    CUDA_CHECK(cudaMemset(a.dWc, 0, size_t(n) * sizeof(uint4)));
    nbody::Params q = a.params;
    q.workCounters = a.dWc;
    a.kernel->launch(a.state, q, 0);
    std::vector<uint4> h(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpy(h.data(), a.dWc, size_t(n) * sizeof(uint4), cudaMemcpyDeviceToHost));
    a.ws = warpStats(h, n);
    a.wsValid = true;
    a.interPerEval = double(a.ws.totalInter);
    a.overflows += nbody::treeStackOverflows(true);
    if (withDepth) { a.td = nbody::lastTreeDepth(); a.tdValid = true; }
    a.measureMs = (glfwGetTime() - t0) * 1e3;
}

void measureEnergy(App& a)
{
    if (a.state.n > kMaxNEnergy) {
        a.message = "energy: N > 1048576 refused (O(N^2) all-pairs on the GPU)";
        return;
    }
    a.eLast = viewer::energyGPU(a.state.pos, a.state.vel, a.state.n, a.params.softening, a.params.G);
    a.eTime = a.simTime;
    a.eValid = true;
    if (!a.e0Valid) { a.e0 = a.eLast; a.e0Time = a.simTime; a.e0Valid = true; a.e0Modified = a.userModified; }
    printf("energy t=%.4f: E=%.9f (K %.6f, W %.6f), dE/|E0|=%.3e since t=%.4f, %.1f ms%s\n", a.eTime, a.eLast.total,
           a.eLast.kinetic, a.eLast.potential, (a.eLast.total - a.e0.total) / std::fabs(a.e0.total), a.e0Time,
           a.eLast.ms, a.userModified ? "  [user-modified run: not conserved]" : "");
}

std::vector<std::string> split(const char* s)
{
    std::vector<std::string> out;
    std::string cur;
    for (const char* c = s; *c; ++c) {
        if (*c == '|') { out.push_back(cur); cur.clear(); } else cur += *c;
    }
    out.push_back(cur);
    return out;
}

void requestRebuild(App& a, const std::string& ic, const nbody::ICOptions& opt, int n, unsigned seed)
{
    a.pendingRebuild = true;
    a.reqIC = ic; a.reqOpt = opt; a.reqN = n; a.reqSeed = seed;
}

int colorModeOf(const std::string& s) { return s == "speed" ? 0 : s == "mass" ? 1 : s == "origin" ? 2 : -1; }

// Scripted actions (--test-keys "F:act"): the panel-free equivalents of keys and panel widgets.
//   up | down | ic | spectrum | mass          as PageUp / PageDown / I / H / M
//   color=speed|mass|origin  kernel=NAME  ic=NAME  n=N  seed=S  spectrum=NAME  theta=X  leaf=K  block=B
//   dt=X  spf=K  lb=0|1  lbevery=K  energy  reset  pause  step  panel=0|1  autorot=0|1  open=well|energy
//   well=FX/FY (hold the gravity well at window fraction FX, FY from the top-left)  wellup (release)
//   wellmass=X  welleps=X
void applyAction(App& a, GLFWwindow* w, const std::string& act)
{
    const size_t eq = act.find('=');
    const std::string k = act.substr(0, eq), v = eq == std::string::npos ? "" : act.substr(eq + 1);
    if (act == "up") a.pendingN = 2;
    else if (act == "down") a.pendingN = -2;
    else if (act == "ic") a.pendingIC = 1;
    else if (act == "spectrum") a.pendingSpectrum = 1;
    else if (act == "mass") a.colorMode = a.colorMode == 1 ? 0 : 1;
    else if (k == "color" && colorModeOf(v) >= 0) a.colorMode = colorModeOf(v);
    else if (k == "kernel") { if (const nbody::KernelInfo* kk = nbody::findKernel(v)) setKernel(a, kk); }
    else if (k == "ic") requestRebuild(a, v, a.icOpt, a.state.n, a.seed);
    else if (k == "n") requestRebuild(a, a.ic, a.icOpt, std::clamp(atoi(v.c_str()), kMinN, maxNFor(a.kernel, a.params)), a.seed);
    else if (k == "seed") requestRebuild(a, a.ic, a.icOpt, a.state.n, unsigned(atoi(v.c_str())));
    else if (k == "spectrum") { nbody::ICOptions o = a.icOpt; o.massSpectrum = v; requestRebuild(a, a.ic, o, a.state.n, a.seed); }
    else if (k == "theta") { a.params.theta = std::clamp(float(atof(v.c_str())), 0.2f, 1.0f); refreshKernelInfo(a); a.statsReset = true; a.countTimer = -1e9; }
    else if (k == "leaf") { a.params.leafSize = std::clamp(atoi(v.c_str()), 1, 64); a.statsReset = true; a.countTimer = -1e9; }
    else if (k == "block") { a.params.blockSize = atoi(v.c_str()); refreshKernelInfo(a); a.statsReset = true; a.countTimer = -1e9; }
    else if (k == "dt") a.params.dt = std::clamp(float(atof(v.c_str())), 1e-4f, 1e-2f);
    else if (k == "spf") { a.stepsPerFrame = std::clamp(atoi(v.c_str()), 1, 64); a.statsReset = true; }
    else if (k == "lb") { a.lbOn = v != "0"; a.framesSinceLB = 1 << 30; }
    else if (k == "lbevery") a.lbEvery = std::clamp(atoi(v.c_str()), 1, 600);
    else if (act == "energy") a.energyReq = true;
    else if (act == "reset") resetSim(a);
    else if (act == "pause") a.paused = !a.paused;
    else if (act == "step") a.stepOnce = true;
    else if (k == "panel") a.showPanel = v != "0";
    else if (k == "open") { a.openWell |= v == "well"; a.openEnergy |= v == "energy"; }
    else if (k == "autorot") a.autoRotate = v != "0";
    else if (k == "well") {
        const size_t sl = v.find('/');
        a.wellFx = float(atof(v.substr(0, sl).c_str()));
        a.wellFy = sl == std::string::npos ? 0.5f : float(atof(v.substr(sl + 1).c_str()));
        a.wellScripted = a.wellActive = true;
    }
    else if (act == "wellup") a.wellScripted = a.wellActive = false;
    else if (k == "wellmass") a.wellMass = float(atof(v.c_str()));
    else if (k == "welleps") a.wellEps = std::max(1e-3f, float(atof(v.c_str())));
    else { printf("unknown test action '%s'\n", act.c_str()); return; }
    (void)w;
}

void onKey(GLFWwindow* w, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    if (key == GLFW_KEY_F1 || (key == GLFW_KEY_TAB && !ImGui::GetIO().WantTextInput)) {
        if (action == GLFW_PRESS) a.showPanel = !a.showPanel;
        return;
    }
    if (ImGui::GetIO().WantCaptureKeyboard) return;   // typing in a panel field
    const auto& reg = nbody::kernelRegistry();
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
        size_t i = size_t(key - GLFW_KEY_1);
        if (i < reg.size()) setKernel(a, &reg[i]);
    }
    const int spf = a.stepsPerFrame;
    switch (key) {
    case GLFW_KEY_ESCAPE:       glfwSetWindowShouldClose(w, 1); break;
    case GLFW_KEY_SPACE:        a.paused = !a.paused; break;
    case GLFW_KEY_R:            resetSim(a); break;
    case GLFW_KEY_V:            a.vsync = !a.vsync; glfwSwapInterval(a.vsync ? 1 : 0); break;
    case GLFW_KEY_M:            a.colorMode = a.colorMode == 1 ? 0 : 1; break;
    case GLFW_KEY_C:            a.colorMode = (a.colorMode + 1) % 3; break;
    case GLFW_KEY_I:            a.pendingIC = 1; break;
    case GLFW_KEY_H:            a.pendingSpectrum = 1; break;
    case GLFW_KEY_PAGE_UP:
    case GLFW_KEY_PERIOD:       a.pendingN = 2; break;
    case GLFW_KEY_PAGE_DOWN:
    case GLFW_KEY_COMMA:        a.pendingN = -2; break;
    case GLFW_KEY_EQUAL:
    case GLFW_KEY_KP_ADD:       a.stepsPerFrame = std::min(a.stepsPerFrame * 2, 64); break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:  a.stepsPerFrame = std::max(a.stepsPerFrame / 2, 1); break;
    case GLFW_KEY_LEFT_BRACKET: a.intensity /= 1.25f; break;
    case GLFW_KEY_RIGHT_BRACKET:a.intensity *= 1.25f; break;
    default: break;
    }
    if (a.stepsPerFrame != spf) a.statsReset = true;
}

void onMouseButton(GLFWwindow* w, int button, int action, int)
{
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    const bool overUi = ImGui::GetIO().WantCaptureMouse;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        a.dragging = action == GLFW_PRESS && !overUi;   // a press on the panel never starts an orbit
        glfwGetCursorPos(w, &a.lastX, &a.lastY);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && !a.wellScripted) a.wellActive = action == GLFW_PRESS && !overUi && a.wellEnabled;
}

void onCursor(GLFWwindow* w, double x, double y)
{
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    if (!a.dragging) return;
    a.yaw   -= float(x - a.lastX) * 0.005f;
    a.pitch += float(y - a.lastY) * 0.005f;
    a.pitch  = std::clamp(a.pitch, -1.55f, 1.55f);
    a.lastX = x;
    a.lastY = y;
}

void onScroll(GLFWwindow* w, double, double dy)
{
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    if (ImGui::GetIO().WantCaptureMouse) return;   // scrolling the panel
    a.dist = std::clamp(a.dist * std::pow(0.9f, float(dy)), 0.2f, 200.f);
}

// 24-bit BMP from the current back buffer. For report figures and automated checks.
void saveScreenshot(const char* path, int w, int h)
{
    int rowBytes = (w * 3 + 3) & ~3;
    std::vector<unsigned char> px(size_t(rowBytes) * h);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, px.data());   // bottom-up = BMP order
    unsigned int fileSize = 54u + unsigned(px.size());
    unsigned char hdr[54] = {'B', 'M'};
    auto put32 = [&](int off, unsigned int v) { for (int i = 0; i < 4; ++i) hdr[off + i] = (v >> (8 * i)) & 0xFF; };
    put32(2, fileSize); put32(10, 54); put32(14, 40); put32(18, unsigned(w)); put32(22, unsigned(h));
    hdr[26] = 1; hdr[28] = 24; put32(34, unsigned(px.size()));
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fwrite(hdr, 1, 54, f);
    fwrite(px.data(), 1, px.size(), f);
    fclose(f);
    printf("screenshot -> %s (%dx%d)\n", path, w, h);
}

// ---- the 2D -> 3D mapping of the gravity well ------------------------------------------------------
// A mouse position fixes a RAY, not a point: every depth along it projects to the same pixel. We pick the
// depth by intersecting the ray with the "focal plane": the plane through the orbit target (the origin)
// perpendicular to the view direction f. With eye e = -dist f, the ray is X(t) = e + t d, d = f + x tanH
// aspect s + y tanH u (x, y the cursor in normalised device coordinates, s, u the camera's right and up,
// tanH = tan(fovY / 2)). Since s, u are orthogonal to f, d . f = 1, so the plane condition (X . f = 0)
// gives t = dist, and X = dist tanH (x aspect s + y u): the cursor's point on the plane the camera orbits
// about. Orbiting the camera (LMB) rotates that plane, so any 3D point can be reached: orbit to choose the
// plane, then place the well in it. Zoom (wheel) scales the plane's extent, not its depth.
float3 wellFromCursor(const Basis& b, float dist, double cx, double cy, int ww, int wh)
{
    const float tanH = std::tan(kFovY * 0.5f), aspect = wh > 0 ? float(ww) / wh : 1.f;
    const float x = ww > 0 ? float(2.0 * cx / ww - 1.0) : 0.f, y = wh > 0 ? float(1.0 - 2.0 * cy / wh) : 0.f;
    const float kx = dist * tanH * x * aspect, ky = dist * tanH * y;
    return make_float3(kx * b.s[0] + ky * b.u[0], kx * b.s[1] + ky * b.u[1], kx * b.s[2] + ky * b.u[2]);
}

// ---- the panel -------------------------------------------------------------------------------------
void helpMarker(const char* text)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    ImGui::SetItemTooltip("%s", text);
}

void drawControls(App& a, GLFWwindow* win)
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(400.f * a.uiScale, 0.f), ImVec2(400.f * a.uiScale, 1e9f));
    ImGui::Begin("Controls  (F1 / Tab hides)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    const auto& reg = nbody::kernelRegistry();

    if (ImGui::CollapsingHeader("Scenario", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto icGetter = [](void* d, int i) -> const char* { return (*static_cast<std::vector<std::string>*>(d))[size_t(i)].c_str(); };
        ImGui::Combo("initial conditions", &a.stIC, icGetter, &a.allIC, int(a.allIC.size()));
        ImGui::SetItemTooltip("The 8 ICs of the workload suite (design §13, §19) and 3 viewer-only presets\n"
                              "(merger3, flyby, unequal: combinations of the disk and Plummer generators).");
        ImGui::Combo("mass spectrum", &a.stSpec, icGetter, &a.spList, int(a.spList.size()));
        ImGui::SliderFloat("bh-mass", &a.stBh, 0.0f, 0.2f, "%.3f");
        ImGui::SetItemTooltip("Central body of each disk, as a fraction of the galaxy's mass (disk, galaxies, presets).");
        ImGui::InputInt("seed", &a.stSeed);
        int lg = std::max(8, int(std::lround(std::log2(double(std::max(a.stN, 256))))));
        char nb[64];
        snprintf(nb, sizeof nb, "2^%d = %d", lg, 1 << lg);
        if (ImGui::SliderInt("N (powers of 2)", &lg, 8, 22, nb)) a.stN = 1 << lg;
        ImGui::InputInt("N exact", &a.stN, 1024, 65536);
        const int cap = maxNFor(a.kernel, a.params);
        a.stN = std::clamp(a.stN, kMinN, cap);
        ImGui::TextDisabled("N cap for %s: %d (TDR margin)", a.kernel->name, cap);
        const bool dirty = a.allIC[size_t(a.stIC)] != a.ic || a.spList[size_t(a.stSpec)] != a.icOpt.massSpectrum ||
                           a.stN != a.state.n || unsigned(a.stSeed) != a.seed || std::fabs(a.stBh - a.icOpt.bhMass) > 1e-7;
        if (ImGui::Button(dirty ? "Apply *" : "Apply")) {
            nbody::ICOptions o = a.icOpt;
            o.massSpectrum = a.spList[size_t(a.stSpec)];
            o.bhMass = std::clamp(double(a.stBh), 0.0, 0.99);
            requestRebuild(a, a.allIC[size_t(a.stIC)], o, a.stN, unsigned(a.stSeed));
        }
        ImGui::SetItemTooltip("Regenerate the IC with these settings (the camera is kept).");
        ImGui::SameLine();
        if (ImGui::Button("Reset (R)")) resetSim(a);
        ImGui::SetItemTooltip("Restart the current scenario from its initial conditions.");
        ImGui::SameLine();
        if (ImGui::Button("Revert")) syncStage(a);
        ImGui::SetItemTooltip("Discard the edits above.");
    }

    if (ImGui::CollapsingHeader("Kernel", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginCombo("force kernel", a.kernel->name)) {
            for (size_t i = 0; i < reg.size(); ++i) {
                char label[128];
                snprintf(label, sizeof label, i < 9 ? "%s  [%zu]" : "%s", reg[i].name, i + 1);
                if (ImGui::Selectable(label, a.kernel == &reg[i])) setKernel(a, &reg[i]);
                ImGui::SetItemTooltip("%s", reg[i].description);
            }
            ImGui::EndCombo();
        }
        ImGui::SetItemTooltip("%s", a.kernel->description);
        if (a.li.control) ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "CONTROL kernel: computes 1 body in 32, not a simulation");
        bool chg = false;
        ImGui::BeginDisabled(!a.tree);
        chg |= ImGui::SliderFloat("theta", &a.params.theta, 0.2f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Opening parameter (bmax criterion). Lower = more accurate, more interactions.\n"
                              "Limited to >= 0.2 here (theta -> 0 is an all-pairs walk: TDR risk at large N).");
        chg |= ImGui::SliderInt("leaf", &a.params.leafSize, 1, 64);
        ImGui::EndDisabled();
        static const int blocks[] = {64, 128, 256, 512, 1024};
        int bi = int(std::find(std::begin(blocks), std::end(blocks), a.params.blockSize) - std::begin(blocks));
        const char* bn[] = {"64", "128", "256", "512", "1024"};
        if (bi >= 5) { ImGui::Text("block %d (from the command line)", a.params.blockSize); }
        else if (ImGui::Combo("block", &bi, bn, 5)) { a.params.blockSize = blocks[bi]; chg = true; }
        if (chg) { refreshKernelInfo(a); a.statsReset = true; a.countTimer = -1e9; a.framesSinceLB = 1 << 30; }
        if (ImGui::SliderInt("steps / frame", &a.stepsPerFrame, 1, 64)) a.statsReset = true;
        ImGui::SliderFloat("dt", &a.params.dt, 1e-4f, 1e-2f, "%.5f", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("pause (Space)", &a.paused);
        ImGui::SameLine();
        if (ImGui::Button("step once")) a.stepOnce = true;
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::RadioButton("speed", &a.colorMode, 0); ImGui::SameLine();
        ImGui::RadioButton("mass", &a.colorMode, 1); ImGui::SameLine();
        ImGui::RadioButton("origin", &a.colorMode, 2);
        ImGui::SameLine(); helpMarker("C cycles. origin: one palette per galaxy / cluster (blue-white, gold,\n"
                                      "green, magenta), bulges paler, central bodies large white points.\n"
                                      "Follows each body through the merger: shows the mixing.");
        ImGui::SliderFloat("brightness", &a.intensity, 0.05f, 8.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("point size", &a.pointSize, 0.3f, 4.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::Checkbox("vsync (V)", &a.vsync)) glfwSwapInterval(a.vsync ? 1 : 0);
        ImGui::SameLine();
        ImGui::Checkbox("auto-rotate", &a.autoRotate);
        if (a.autoRotate) ImGui::SliderFloat("rotate speed", &a.rotSpeed, -1.f, 1.f, "%.2f rad/s");
        if (ImGui::Button("reset camera")) { a.yaw = a.yaw0; a.pitch = a.pitch0; a.dist = a.dist0; }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        if (ImGui::SliderFloat("UI scale", &a.uiScale, 0.75f, 2.5f, "%.2f")) ImGui::GetIO().FontGlobalScale = a.uiScale;
    }

    if (a.openWell) { ImGui::SetNextItemOpen(true); a.openWell = false; }
    if (ImGui::CollapsingHeader("Gravity well (hold right mouse)")) {
        ImGui::Checkbox("enabled", &a.wellEnabled);
        ImGui::SliderFloat("strength (mass)", &a.wellMass, -2.f, 2.f, "%.3f");
        ImGui::SetItemTooltip("Point mass of the well in units of the system's total mass (1). Negative repels.");
        ImGui::SliderFloat("softening", &a.wellEps, 0.01f, 1.f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::TextWrapped("The cursor picks a point on the plane through the orbit target, perpendicular to "
                           "the view. Orbit (LMB) to tilt that plane. Viewer-only kick after each step: the "
                           "bench and the force kernels never see it. Energy is not conserved while it acts.");
    }
    if (!a.message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.6f, 0.3f, 1));
        ImGui::TextWrapped("%s", a.message.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::End();
    (void)win;
}

void drawStats(App& a, int ww)
{
    // fixed width, height follows the content; anchored to the top-right corner (pivot 1, 0) on first use
    ImGui::SetNextWindowPos(ImVec2(float(ww) - 10.f, 10.f), ImGuiCond_FirstUseEver, ImVec2(1.f, 0.f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(470.f * a.uiScale, 0.f), ImVec2(470.f * a.uiScale, 1e9f));
    ImGui::Begin("Live statistics", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    const int n = a.state.n;
    ImGui::Text("%s (%s)  N = %d  seed %u", a.ic.c_str(), a.icOpt.massSpectrum.c_str(), n, a.seed);
    char tp[64] = "";
    if (a.tree) snprintf(tp, sizeof tp, ", theta %.2f, leaf %d", a.params.theta, a.params.leafSize);
    ImGui::Text("kernel %s  (block %d%s)", a.kernel->name, a.params.blockSize, tp);
    if (a.userModified)
        ImGui::TextColored(ImVec4(1, 0.35f, 0.25f, 1), "MODIFIED BY USER INTERACTION (gravity well): energy not conserved");
    ImGui::Separator();
    if (a.fps > 0) ImGui::Text("%.0f FPS   t = %.3f   %d steps/frame%s", a.fps, a.simTime, a.stepsPerFrame, a.paused ? "   PAUSED" : "");
    else ImGui::Text("-- FPS   t = %.3f   (first statistics after 0.5 s)", a.simTime);
    ImGui::Text("%.3f ms / step", a.msPerStep);
    helpMarker("CUDA events around the frame's steps: force kernel (for trees: build + walk) + the two\n"
               "kicks (+ the well kick while it acts). A live display, not a benchmark: quote nbody_bench.");
    const double rate = a.msPerStep > 0 && a.interPerEval > 0 ? a.interPerEval / (a.msPerStep * 1e-3) * 1e-9 : 0.0;
    ImGui::Text("%.1f G interactions/s real (%s)", rate, a.tree ? "counted" : "N^2, exact");
    helpMarker("Tree kernels: interactions counted by one instrumented launch (refreshed every 2 s, or every K\n"
               "frames with load-balance measurement on) divided by the measured ms/step. Never the\n"
               "N^2-equivalent rate.");
    ImGui::Text("%.0f interactions / body", n > 0 ? a.interPerEval / n : 0.0);
    char ov[64];
    snprintf(ov, sizeof ov, "ms/step, last %d frames", a.histMs.count);
    a.histMs.plot("##ms", ov, 0.f, std::max(0.001f, a.histMs.maxv() * 1.15f), 50.f * a.uiScale);

    if (a.tree && ImGui::CollapsingHeader("Tree phases (first step of each frame)", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!a.phaseValid) ImGui::TextDisabled("no samples yet (paused?)");
        else if (ImGui::BeginTable("phases", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            float sum = 0;
            for (int k = 0; k < a.li.nPhases; ++k) sum += a.phaseEma[k];
            for (int k = 0; k < a.li.nPhases; ++k) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%s", a.li.phaseNames[k]);
                ImGui::TableNextColumn(); ImGui::Text("%8.3f ms", a.phaseEma[k]);
                ImGui::TableNextColumn(); ImGui::ProgressBar(sum > 0 ? a.phaseEma[k] / sum : 0.f, ImVec2(160 * a.uiScale, 0));
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("build (k+s+t)");
            ImGui::TableNextColumn(); ImGui::Text("%8.3f ms", a.li.nPhases >= 3 ? a.phaseEma[0] + a.phaseEma[1] + a.phaseEma[2] : 0.f);
            ImGui::TableNextColumn(); ImGui::Text("pipeline %.3f ms", sum);
            ImGui::EndTable();
        }
        ImGui::TextDisabled("CUDA events inside the pipeline (as nbody_bench), smoothed");
    }

    if (ImGui::CollapsingHeader("Load balance", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Checkbox("measure load balance", &a.lbOn)) a.framesSinceLB = 1 << 30;
        helpMarker("Runs the kernel's INSTRUMENTED TWIN (same tree, same walk, plus per-thread work counters)\n"
                   "every K frames, outside the timed window. Accelerations are bitwise equal, so the run is\n"
                   "undisturbed; ms/step stays the production kernel. The twins run a few % differently\n"
                   "from production (-3.4..+6.2% measured, D-011), so the counts are exact, the times are not\n"
                   "theirs. FPS dips on measured frames (counter copy + host reduction).");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110 * a.uiScale);
        ImGui::SliderInt("every K frames", &a.lbEvery, 1, 120);
        if (!a.tree) {
            ImGui::TextWrapped("All-pairs kernel: every body does exactly N interactions, so there is no intra-warp "
                               "or between-warp work imbalance; its imbalance is SM-level (wave quantisation, "
                               "nbody_bench --timeline).");
        } else if (!a.lbOn || !a.wsValid) {
            ImGui::TextDisabled(a.lbOn ? "waiting for the first measurement" : "off");
        } else {
            const WarpStats& w = a.ws;
            ImGui::Text("per-warp max/mean   %.3f   (median warp %.3f)", w.warpMaxMean, w.warpMaxMeanMedian);
            ImGui::SetItemTooltip("sum over warps of the busiest lane's interactions / sum of the mean lane's.\n"
                                  "1 = every lane of a warp does the same work (no intra-warp imbalance).");
            ImGui::Text("SIMT efficiency     %.3f interactions, %.3f pops", w.simtInter, w.simtVisit);
            ImGui::SetItemTooltip("Mean fraction of a warp's 32 lanes active where interactions (node pops) are\n"
                                  "executed: sum I / (32 x leader I), from __activemask leader counts. #7's union walk\n"
                                  "is 1.000 by construction; its price shows up as more interactions/body.");
            ImGui::Text("warp-cost spread    %.2f  (max / mean warp)", w.interWarp);
            ImGui::SetItemTooltip("Most expensive warp / mean warp (by the busiest lane): between-warp imbalance,\n"
                                  "the input of the persistent-thread schedulers (R3 #6).");
            ImGui::Text("%.0f interactions/body, %.0f pops/body, %d warps", w.interPerBody, w.visitsPerBody, w.warps);
            a.histSimt.plot("##simt", "SIMT eff (interactions), 0..1", 0.f, 1.1f, 45.f * a.uiScale);
            a.histWarp.plot("##warp", "per-warp max/mean, from 1", 0.95f, std::max(1.2f, a.histWarp.maxv() * 1.05f), 45.f * a.uiScale);
            if (a.tdValid)
                ImGui::TextWrapped("tree depth %d, push depth %d -> stack need %d (capacity %d thread / %d warp), dup keys %d",
                            a.td.maxDepth, a.td.pushDepth, a.td.stackBound, a.td.threadStack, a.td.warpStack, a.td.dupKeys);
            ImGui::Text("stack overflows %d   measurement %.1f ms", a.overflows, a.measureMs);
            if (a.overflows) ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "STACK OVERFLOW: results invalid");
        }
    }

    if (a.openEnergy) { ImGui::SetNextItemOpen(true); a.openEnergy = false; }
    if (ImGui::CollapsingHeader("Energy")) {
        ImGui::BeginDisabled(n > kMaxNEnergy);
        if (ImGui::Button("measure energy now")) a.energyReq = true;
        ImGui::SameLine();
        ImGui::BeginDisabled(n > (1 << 18));
        ImGui::Checkbox("auto every 5 s", &a.energyAuto);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        helpMarker("Exact all-pairs energy on the GPU (FP32 pair terms, FP64 accumulation), O(N^2): about 0.1 s at\n"
                   "N = 262144, 1-2 s at 1M (chunked launches, TDR-safe); refused above 1M, auto only up to 262144.\n"
                   "The first measurement after a reset is the reference E0. Tree kernels do not conserve energy\n"
                   "exactly (approximate forces); this is a display, the bench --energy is the measurement.");
        if (a.eValid) {
            ImGui::Text("E = %.8f  (K %.6f, W %.6f) at t = %.3f", a.eLast.total, a.eLast.kinetic, a.eLast.potential, a.eTime);
            ImGui::Text("dE/|E0| = %+.3e  since t = %.3f   (%.0f ms)", (a.eLast.total - a.e0.total) / std::fabs(a.e0.total),
                        a.e0Time, a.eLast.ms);
            ImGui::Text("virial 2K/|W| = %.3f", a.eLast.potential != 0 ? 2 * a.eLast.kinetic / std::fabs(a.eLast.potential) : 0.0);
            if (a.userModified || a.e0Modified)
                ImGui::TextColored(ImVec4(1, 0.35f, 0.25f, 1), "the well acted: the drift is the work done by the user");
        } else {
            ImGui::TextDisabled(n > kMaxNEnergy ? "N > 1048576: not offered (O(N^2))" : "not measured since the last reset");
        }
    }
    ImGui::End();
}

} // namespace

int main(int argc, char** argv)
{
    App a;
    a.params.n  = 32768;
    a.params.dt = 0.002f;
    // bh_warp is the tree to use from tree round 2 on (D-022); all-pairs stays on keys 1-4.
    std::string kernelName = "bh_warp", screenshot, ic = "plummer";
    int maxFrames = 0, panelArg = -1;
    bool maximize = false;
    bool energyCheck = false;
    // --test-keys "F:act,F:act,..." (smoke tests without a human): at frame F apply act (applyAction()).
    std::string testKeys;
    for (int i = 1; i < argc; i += 2) {
        std::string k = argv[i];
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); return 2; }   // no silent dangling arg
        if      (k == "--n")          a.params.n = atoi(argv[i + 1]);
        else if (k == "--frames")     maxFrames = atoi(argv[i + 1]);
        else if (k == "--screenshot") screenshot = argv[i + 1];
        else if (k == "--kernel") kernelName = argv[i + 1];
        else if (k == "--dt")     a.params.dt = float(atof(argv[i + 1]));
        else if (k == "--block")  a.params.blockSize = atoi(argv[i + 1]);
        else if (k == "--q")      a.params.threadsPerBody = atoi(argv[i + 1]);
        else if (k == "--unroll") a.params.unroll = atoi(argv[i + 1]);
        else if (k == "--seed")   a.seed = unsigned(atoi(argv[i + 1]));
        else if (k == "--softening") a.params.softening = float(atof(argv[i + 1]));
        else if (k == "--prefetch")  a.params.prefetch = atoi(argv[i + 1]);
        else if (k == "--theta")     a.params.theta = float(atof(argv[i + 1]));   // bh_* kernels
        else if (k == "--leaf")      a.params.leafSize = atoi(argv[i + 1]);
        else if (k == "--ic")        ic = argv[i + 1];
        else if (k == "--mass-spectrum") a.icOpt.massSpectrum = argv[i + 1];
        else if (k == "--bh-mass")   a.icOpt.bhMass = atof(argv[i + 1]);
        else if (k == "--dist")      a.dist = float(atof(argv[i + 1]));
        else if (k == "--yaw")       a.yaw = float(atof(argv[i + 1]));
        else if (k == "--pitch")     a.pitch = float(atof(argv[i + 1]));
        else if (k == "--test-keys") testKeys = argv[i + 1];
        else if (k == "--steps-per-frame") a.stepsPerFrame = std::max(1, atoi(argv[i + 1]));
        else if (k == "--vsync")     a.vsync = atoi(argv[i + 1]) != 0;
        else if (k == "--mass-color") a.colorMode = atoi(argv[i + 1]) != 0 ? 1 : 0;
        else if (k == "--color")     { a.colorMode = colorModeOf(argv[i + 1]); if (a.colorMode < 0) { fprintf(stderr, "--color must be speed|mass|origin\n"); return 2; } }
        else if (k == "--brightness") a.intensity = float(atof(argv[i + 1]));
        else if (k == "--panel")     panelArg = atoi(argv[i + 1]) != 0;
        else if (k == "--ui-scale")  a.uiScale = std::clamp(float(atof(argv[i + 1])), 0.5f, 3.f);
        else if (k == "--point-size") a.pointSize = std::clamp(float(atof(argv[i + 1])), 0.1f, 8.f);
        else if (k == "--auto-rotate") a.autoRotate = atoi(argv[i + 1]) != 0;
        else if (k == "--lb")        a.lbOn = atoi(argv[i + 1]) != 0;
        else if (k == "--lb-every")  a.lbEvery = std::clamp(atoi(argv[i + 1]), 1, 600);
        else if (k == "--well-mass") a.wellMass = float(atof(argv[i + 1]));
        else if (k == "--well-eps")  a.wellEps = std::max(1e-3f, float(atof(argv[i + 1])));
        else if (k == "--energy-check") energyCheck = atoi(argv[i + 1]) != 0;
        else if (k == "--maximize")  maximize = atoi(argv[i + 1]) != 0;
        else { fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    a.yaw0 = a.yaw; a.pitch0 = a.pitch; a.dist0 = a.dist;
    a.showPanel = panelArg >= 0 ? panelArg != 0 : screenshot.empty();
    if (!(a.params.softening > 0.f)) { fprintf(stderr, "--softening must be > 0\n"); return 2; }
    if (a.params.n <= 0) { fprintf(stderr, "--n must be > 0\n"); return 2; }
    if (!(a.params.theta >= 0.f) || a.params.theta > 1.f) { fprintf(stderr, "--theta must be in [0, 1]\n"); return 2; }
    if (!nbody::validMassSpectrum(a.icOpt.massSpectrum)) {
        fprintf(stderr, "--mass-spectrum must be %s\n", nbody::massSpectrumNames()); return 2;
    }
    a.icOpt.softening = a.params.softening;
    a.kernel = nbody::findKernel(kernelName);
    if (!a.kernel) { fprintf(stderr, "unknown kernel '%s'\n", kernelName.c_str()); return 2; }
    if (a.params.n > maxNFor(a.kernel, a.params)) {
        fprintf(stderr, "%s is limited to N <= %d in the viewer (TDR margin)\n", a.kernel->name, maxNFor(a.kernel, a.params));
        return 2;
    }
    a.icList = split(nbody::icNames());
    a.presetList = split(nbody::presetNames());
    a.allIC = a.icList;
    a.allIC.insert(a.allIC.end(), a.presetList.begin(), a.presetList.end());
    a.spList = split(nbody::massSpectrumNames());
    if (std::find(a.allIC.begin(), a.allIC.end(), ic) == a.allIC.end()) {
        fprintf(stderr, "unknown --ic %s (%s|%s)\n", ic.c_str(), nbody::icNames(), nbody::presetNames());
        return 2;
    }

    // ---- window + GL ---------------------------------------------------------
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1600, 900, "N-body", nullptr, nullptr);
    if (!win) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }
    if (maximize) glfwMaximizeWindow(win);
    glfwMakeContextCurrent(win);
    if (!gladLoadGL(glfwGetProcAddress)) { fprintf(stderr, "gladLoadGL failed\n"); return 1; }
    glfwSwapInterval(a.vsync ? 1 : 0);
    glfwSetWindowUserPointer(win, &a);
    // Our callbacks first: ImGui's GLFW backend (install_callbacks = true) saves and chains them, so every
    // event reaches ImGui first and then us; ours check io.WantCaptureMouse / WantCaptureKeyboard.
    glfwSetKeyCallback(win, onKey);
    glfwSetMouseButtonCallback(win, onMouseButton);
    glfwSetCursorPosCallback(win, onCursor);
    glfwSetScrollCallback(win, onScroll);
    printf("GL %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;   // no imgui.ini next to the binary / in the repo
    io.FontGlobalScale = a.uiScale;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().Alpha = 0.92f;
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");   // the backend's own embedded loader (see CMakeLists.txt)

    // ---- GL buffers (pos/vel registered with CUDA in resizeVbos) --------------
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, a.vbo);
    for (int b = 0; b < 2; ++b) {
        glBindBuffer(GL_ARRAY_BUFFER, a.vbo[b]);
        glBufferData(GL_ARRAY_BUFFER, 16, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(GLuint(b), 4, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);
        glEnableVertexAttribArray(GLuint(b));
    }
    glBindBuffer(GL_ARRAY_BUFFER, a.vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, 16, nullptr, GL_STATIC_DRAW);
    glVertexAttribIPointer(2, 1, GL_UNSIGNED_BYTE, 1, nullptr);   // integer attribute: uint aTag
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    for (cudaEvent_t& e : a.phaseEv) CUDA_CHECK(cudaEventCreate(&e));

    // ---- simulation ----------------------------------------------------------
    if (!rebuild(a, ic, a.icOpt, a.params.n, a.seed)) return 2;

    if (energyCheck) {   // GPU energy vs the host FP64 reference (nbody::computeEnergy), on the initial state
        if (a.state.n > 32768) printf("energy-check: skipped (N > 32768, the host reference is O(N^2) in FP64)\n");
        else {
            const viewer::EnergyResult g = viewer::energyGPU(a.state.pos, a.state.vel, a.state.n, a.params.softening, a.params.G);
            const nbody::Energy h = nbody::computeEnergy(a.pos0, a.vel0, a.params);
            printf("energy-check N=%d: GPU K %.12f W %.12f E %.12f | host K %.12f W %.12f E %.12f | rel dE %.3e, dW %.3e\n",
                   a.state.n, g.kinetic, g.potential, g.total, h.kinetic, h.potential, h.total,
                   std::fabs(g.total - h.total) / std::fabs(h.total), std::fabs(g.potential - h.potential) / std::fabs(h.potential));
        }
    }

    GLuint prog = linkProgram();
    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uPointScale = glGetUniformLocation(prog, "uPointScale");
    GLint uSizeMul = glGetUniformLocation(prog, "uSizeMul");
    GLint uSpeedRef = glGetUniformLocation(prog, "uSpeedRef");
    GLint uIntensity = glGetUniformLocation(prog, "uIntensity");
    GLint uMassRef = glGetUniformLocation(prog, "uMassRef");
    GLint uColorMode = glGetUniformLocation(prog, "uColorMode");

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double titleTimer = glfwGetTime(), simMsAcc = 0.0, lastFrameT = glfwGetTime();
    int frames = 0, simSteps = 0, totalFrames = 0;
    char lastTitle[512] = "";

    std::vector<std::pair<int, std::string>> keyScript;
    for (size_t b = 0; b < testKeys.size();) {
        size_t e = testKeys.find(',', b);
        if (e == std::string::npos) e = testKeys.size();
        const std::string item = testKeys.substr(b, e - b);
        const size_t c = item.find(':');
        if (c != std::string::npos) keyScript.push_back({atoi(item.substr(0, c).c_str()), item.substr(c + 1)});
        b = e + 1;
    }

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        const double tNow = glfwGetTime(), frameDt = tNow - lastFrameT;
        lastFrameT = tNow;
        for (const auto& ks : keyScript)
            if (ks.first == totalFrames) {
                printf("test key at frame %d: %s\n", totalFrames, ks.second.c_str());
                applyAction(a, win, ks.second);
            }

        int fbw, fbh, ww, wh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        glfwGetWindowSize(win, &ww, &wh);

        // ---- UI (built before the simulation so this frame's edits take effect now) -------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (a.showPanel) { drawControls(a, win); drawStats(a, ww); }

        // ---- pending live changes (N, IC, mass spectrum, panel Apply) -------------------------
        if (a.pendingN || a.pendingIC || a.pendingSpectrum) {
            int n = a.state.n;
            std::string nic = a.ic;
            nbody::ICOptions opt = a.icOpt;
            if (a.pendingN > 0) n = std::min(n * 2, maxNFor(a.kernel, a.params));
            if (a.pendingN < 0) n = std::max(n / 2, kMinN);
            if (a.pendingIC) {
                size_t i = std::find(a.allIC.begin(), a.allIC.end(), a.ic) - a.allIC.begin();
                nic = a.allIC[(i + 1) % a.allIC.size()];
            }
            if (a.pendingSpectrum) {
                size_t i = std::find(a.spList.begin(), a.spList.end(), a.icOpt.massSpectrum) - a.spList.begin();
                opt.massSpectrum = a.spList[(i + 1) % a.spList.size()];
            }
            if (a.pendingN > 0 && n == a.state.n)
                printf("%s is limited to N <= %d (TDR margin)\n", a.kernel->name, maxNFor(a.kernel, a.params));
            if (n != a.state.n || nic != a.ic || opt.massSpectrum != a.icOpt.massSpectrum) rebuild(a, nic, opt, n, a.seed);
            a.pendingN = a.pendingIC = a.pendingSpectrum = 0;
        }
        if (a.pendingRebuild) {
            a.pendingRebuild = false;
            if (a.reqN > maxNFor(a.kernel, a.params)) a.reqN = maxNFor(a.kernel, a.params);
            rebuild(a, a.reqIC, a.reqOpt, std::max(a.reqN, kMinN), a.reqSeed);
        }
        const int n = a.state.n;

        // ---- work counters (outside the timed window) ---------------------------------------
        ++a.framesSinceLB;
        if (a.tree && a.lbOn) {
            if (a.framesSinceLB >= a.lbEvery) {
                measureWork(a, true);
                a.framesSinceLB = 0; a.countTimer = glfwGetTime();
                a.histSimt.push(float(a.ws.simtInter));
                a.histWarp.push(float(a.ws.warpMaxMean));
            }
        } else if (glfwGetTime() - a.countTimer > 2.0) { measureWork(a, false); a.countTimer = glfwGetTime(); }

        // ---- energy (on request) -------------------------------------------------------------
        if (a.energyAuto && n <= (1 << 18) && glfwGetTime() - a.energyTimer > 5.0) a.energyReq = true;
        if (a.energyReq) { a.energyReq = false; measureEnergy(a); a.energyTimer = glfwGetTime(); }

        // ---- gravity well: cursor -> 3D point (wellFromCursor) ---------------------------------
        if (a.autoRotate) a.yaw += float(frameDt) * a.rotSpeed;
        const float ex = a.dist * std::cos(a.pitch) * std::sin(a.yaw);
        const float ey = a.dist * std::sin(a.pitch);
        const float ez = a.dist * std::cos(a.pitch) * std::cos(a.yaw);
        const Basis cam = cameraBasis(ex, ey, ez);
        double cx = 0, cy = 0;
        if (a.wellScripted) { cx = a.wellFx * ww; cy = a.wellFy * wh; }
        else glfwGetCursorPos(win, &cx, &cy);
        const bool wellOn = a.wellActive && a.wellEnabled;
        if (wellOn) a.wellPos = wellFromCursor(cam, a.dist, cx, cy, ww, wh);

        // ---- simulate --------------------------------------------------------
        int stepsThisFrame = 0;
        const bool run = !a.paused || a.stepOnce;
        if (run) {
            CUDA_CHECK(cudaEventRecord(e0));
            stepsThisFrame = a.paused ? 1 : a.stepsPerFrame;
            for (int s = 0; s < stepsThisFrame; ++s) {
                nbody::Params pp = a.params;
                if (s == 0 && a.tree && a.li.nPhases > 0 && a.li.nPhases <= 8) pp.phaseEvents = a.phaseEv;
                nbody::step(a.state, pp, *a.kernel, 0);
                if (wellOn) {   // viewer-only external kick (viewer_kernels.cu), after the KDK step
                    viewer::wellKick(a.state.vel, a.state.pos, n, a.wellPos, a.wellMass, a.wellEps, a.params.G, a.params.dt);
                    a.userModified = true;
                }
            }
            a.simTime += stepsThisFrame * double(a.params.dt);
            CUDA_CHECK(cudaEventRecord(e1));
            a.stepOnce = false;
        }

        // ---- publish to GL (VRAM -> VRAM) -------------------------------------
        CUDA_CHECK(cudaGraphicsMapResources(2, a.res, 0));
        void* ptr[2];
        size_t bytes = 0;
        for (int b = 0; b < 2; ++b)
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr[b], &bytes, a.res[b]));
        CUDA_CHECK(cudaMemcpyAsync(ptr[0], a.state.pos, size_t(n) * sizeof(float4), cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(ptr[1], a.state.vel, size_t(n) * sizeof(float4), cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(2, a.res, 0));

        if (run) {
            float ms = 0.f;
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (a.statsReset) {   // window restarts with the first frame at the new setting
                simMsAcc = 0.0; simSteps = 0; a.statsReset = false;
            }
            simMsAcc += ms;
            simSteps += stepsThisFrame;   // the steps actually run this frame, not the current setting
            a.histMs.push(ms / float(std::max(1, stepsThisFrame)));
            if (a.tree && a.li.nPhases > 0 && a.li.nPhases <= 8) {
                for (int k = 0; k < a.li.nPhases; ++k) {
                    CUDA_CHECK(cudaEventElapsedTime(&a.phaseMs[k], a.phaseEv[k], a.phaseEv[k + 1]));
                    a.phaseEma[k] = a.phaseValid ? 0.9f * a.phaseEma[k] + 0.1f * a.phaseMs[k] : a.phaseMs[k];
                }
                a.phaseValid = true;
            }
        }

        // ---- draw ------------------------------------------------------------
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.005f, 0.005f, 0.015f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glDisable(GL_DEPTH_TEST);
        Mat4 mvp = mul(perspective(kFovY, fbh > 0 ? float(fbw) / fbh : 1.f, 0.01f, 1000.f), lookAt(cam));
        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp.m);
        glUniform1f(uPointScale, 0.010f * float(fbh));
        glUniform1f(uSizeMul, a.pointSize);
        glUniform1f(uSpeedRef, 1.2f);
        glUniform1f(uIntensity, a.baseIntensity * a.intensity);
        glUniform1f(uMassRef, 1.0f / float(n));   // every IC has total mass 1
        glUniform1i(uColorMode, a.colorMode);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, n);
        glBindVertexArray(0);

        // gravity-well marker: the 3D point is on the cursor's ray, so it projects to the cursor; the ring is
        // the well's softening length at the focal plane's scale (pixels per unit = (wh / 2) / (dist tanH)).
        if (wellOn) {
            const float ppu = 0.5f * float(wh) / (a.dist * std::tan(kFovY * 0.5f));
            const float r = std::max(6.f, a.wellEps * ppu);
            const ImU32 col = a.wellMass >= 0 ? IM_COL32(120, 220, 255, 230) : IM_COL32(255, 120, 90, 230);
            ImDrawList* dl = ImGui::GetForegroundDrawList();
            const ImVec2 c{float(cx), float(cy)};
            dl->AddCircle(c, r, col, 48, 2.f);
            dl->AddLine(ImVec2(c.x - r - 6, c.y), ImVec2(c.x - r + 4, c.y), col, 2.f);
            dl->AddLine(ImVec2(c.x + r - 4, c.y), ImVec2(c.x + r + 6, c.y), col, 2.f);
            dl->AddLine(ImVec2(c.x, c.y - r - 6), ImVec2(c.x, c.y - r + 4), col, 2.f);
            dl->AddLine(ImVec2(c.x, c.y + r - 4), ImVec2(c.x, c.y + r + 6), col, 2.f);
            char lbl[96];
            snprintf(lbl, sizeof lbl, "well M=%.2f  (%.2f, %.2f, %.2f)", a.wellMass, a.wellPos.x, a.wellPos.y, a.wellPos.z);
            dl->AddText(ImVec2(c.x - r, c.y + r + 8), col, lbl);
        }
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        ++totalFrames;
        bool last = maxFrames > 0 && totalFrames >= maxFrames;
        if (last && !screenshot.empty()) saveScreenshot(screenshot.c_str(), fbw, fbh);
        glfwSwapBuffers(win);
        ++frames;

        // ---- title / stats ----------------------------------------------------
        double now = glfwGetTime();
        if (now - titleTimer > 0.5 || last) {
            a.fps = frames / (now - titleTimer);
            a.msPerStep = simSteps ? simMsAcc / double(simSteps) : a.msPerStep;
            double rate = a.msPerStep > 0 && a.interPerEval > 0 ? a.interPerEval / (a.msPerStep * 1e-3) * 1e-9 : 0.0;
            snprintf(lastTitle, sizeof lastTitle,
                     "N-body | %s (%s) | N=%d | %s | %.3f ms/step | %.1f G int/s real (%s, incl. %sintegrator) | "
                     "%.0f int/body | %d steps/frame | %.0f FPS%s%s%s",
                     a.ic.c_str(), a.icOpt.massSpectrum.c_str(), n, a.kernel->name, a.msPerStep, rate,
                     a.tree ? "counted" : "N^2", a.tree ? "build + " : "", a.interPerEval / n, a.stepsPerFrame, a.fps,
                     a.vsync ? "" : " | no vsync", a.paused ? " | PAUSED" : "", a.userModified ? " | USER-MODIFIED" : "");
            glfwSetWindowTitle(win, lastTitle);
            titleTimer = now;
            frames = simSteps = 0;
            simMsAcc = 0.0;
        }
        if (last) {
            printf("%s\nsimulated time t = %.4f (dt %g, %d frames, camera dist %.3f yaw %.3f pitch %.3f)\n", lastTitle,
                   a.simTime, a.params.dt, totalFrames, a.dist, a.yaw, a.pitch);
            if (a.tree && a.phaseValid) {
                printf("phases (smoothed ms):");
                for (int k = 0; k < a.li.nPhases; ++k) printf(" %s %.4f", a.li.phaseNames[k], a.phaseEma[k]);
                printf("\n");
            }
            if (a.wsValid)
                printf("load balance: per-warp max/mean %.3f (median %.3f) | SIMT eff interactions %.3f, pops %.3f | "
                       "warp-cost max/mean %.2f | %.1f int/body | overflows %d\n", a.ws.warpMaxMean, a.ws.warpMaxMeanMedian,
                       a.ws.simtInter, a.ws.simtVisit, a.ws.interWarp, a.ws.interPerBody, a.overflows);
            if (a.tdValid)
                printf("tree: max depth %d, push depth %d, stack bound %d (capacity %d / %d), dup keys %d\n", a.td.maxDepth,
                       a.td.pushDepth, a.td.stackBound, a.td.threadStack, a.td.warpStack, a.td.dupKeys);
            if (a.userModified) printf("run modified by user interaction (gravity well)\n");
            glfwSetWindowShouldClose(win, 1);
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    for (int b = 0; b < 2; ++b) if (a.res[b]) cudaGraphicsUnregisterResource(a.res[b]);
    if (a.dWc) cudaFree(a.dWc);
    nbody::release(a.state);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
