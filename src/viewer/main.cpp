// nbody_viewer: real-time CUDA-OpenGL interop view of the simulation.
//
// The simulation state never leaves the GPU. Each frame: run K leapfrog steps with the
// selected force kernel, then copy pos/vel device-to-device into two GL vertex buffers that
// are mapped into CUDA (cudaGraphicsGLRegisterBuffer). That copy is VRAM->VRAM, not over
// PCIe (one copy engine, docs/01-hardware.md). For N = 131k it is 4 MB, a few microseconds
// at 736 GB/s. The core library stays untouched, so the bench measures the same code.
//
//   nbody_viewer [--n 32768] [--kernel bh_warp] [--dt 0.002] [--block 256] [--seed 42]
//                [--frames F --screenshot out.bmp]   (run F frames, save the last one, exit)
//                [--ic plummer|collision|core|disk|galaxies|cube_shell|cube_edges|uniform]
//                [--mass-spectrum equal|salpeter|heavy-few] [--bh-mass 0.01]
//                [--theta 0.5 --leaf 16] [--dist 4 --yaw 0.6 --pitch 0.35]   (camera)
//                [--test-keys "F:up,F:ic,..."]   (scripted key presses for smoke tests: up down ic spectrum mass)
//   The title shows IC, mass spectrum, N, kernel, ms/step and the REAL interaction rate: all-pairs
//   N^2 per step; tree kernels the interactions counted by one instrumented launch (refreshed every
//   2 s), divided by the measured ms/step (which includes the build and the two kicks).
//
// Controls: LMB drag orbit · wheel zoom · 1..9 kernel · Space pause · +/- steps per frame
//           PageUp / . double N · PageDown / , halve N · I next IC · H next mass spectrum
//           M colour by mass · [ ] brightness · R reset · V vsync · Esc quit
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
#include "nbody.h"

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

Mat4 lookAt(float ex, float ey, float ez)   // target = origin, up = +y
{
    float fx = -ex, fy = -ey, fz = -ez;
    float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0.f - fz * 1.f, sy = fz * 0.f - fx * 0.f, sz = fx * 1.f - fy * 0.f;   // f x up
    float sl = std::sqrt(sx * sx + sy * sy + sz * sz);
    sx /= sl; sy /= sl; sz /= sl;
    float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;          // s x f
    Mat4 r{};
    r.m[0] = sx;  r.m[4] = sy;  r.m[8]  = sz;
    r.m[1] = ux;  r.m[5] = uy;  r.m[9]  = uz;
    r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz;
    r.m[12] = -(sx * ex + sy * ey + sz * ez);
    r.m[13] = -(ux * ex + uy * ey + uz * ez);
    r.m[14] = (fx * ex + fy * ey + fz * ez);
    r.m[15] = 1.f;
    return r;
}

// ---- shaders ------------------------------------------------------------------
// Colour by speed (slow = deep blue, fast = warm white). With uMassColor, bodies heavier than the mean
// mass (uMassRef) shift towards red and grow, log-scaled: 3x the mean starts to show, 100x is full red.
const char* kVertexSrc = R"(#version 330 core
layout(location = 0) in vec4 aPos;   // xyz + mass
layout(location = 1) in vec4 aVel;
uniform mat4  uMVP;
uniform float uPointScale;
uniform float uSpeedRef;
uniform float uMassRef;
uniform int   uMassColor;
out vec3 vColor;
void main() {
    gl_Position  = uMVP * vec4(aPos.xyz, 1.0);
    float t = clamp(length(aVel.xyz) / uSpeedRef, 0.0, 1.0);
    vColor = mix(vec3(0.15, 0.35, 1.0), vec3(1.0, 0.75, 0.45), t) + vec3(0.25 * t * t);
    float h = 0.0;
    if (uMassColor != 0) h = clamp(log(max(aPos.w / uMassRef, 1.0) / 3.0) / log(33.3), 0.0, 1.0);
    vColor = mix(vColor, vec3(4.0, 0.6, 0.15), h);
    gl_PointSize = clamp(uPointScale / gl_Position.w * (1.0 + 2.0 * h), 1.5, 8.0 + 16.0 * h);
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

// ---- app state ----------------------------------------------------------------
struct App {
    nbody::Params            params;
    nbody::DeviceState       state;
    const nbody::KernelInfo* kernel = nullptr;
    std::vector<float4>      pos0, vel0;
    unsigned                 seed = 42;
    std::string              ic = "plummer";
    nbody::ICOptions         icOpt;

    int    stepsPerFrame = 2;
    bool   paused = false, vsync = true, massColor = true;
    float  yaw = 0.6f, pitch = 0.35f, dist = 4.0f;
    float  intensity = 1.0f, baseIntensity = 1.0f;
    bool   dragging = false;
    double lastX = 0, lastY = 0;
    bool   statsReset = false;   // set when steps/frame, the kernel, N or the IC change: restart the ms/step window
    // pending changes requested by keys, applied in the main loop
    int    pendingN = 0;
    int    pendingIC = 0, pendingSpectrum = 0;
    // GL interop
    GLuint vbo[2] = {0, 0};
    cudaGraphicsResource* res[2] = {nullptr, nullptr};
    int    vboN = 0;
    // real interaction count per force evaluation (tree: counted; all-pairs: N^2)
    double interPerEval = -1.0;
    double countTimer = -1e9;
};

bool isTree(const nbody::KernelInfo* k, const nbody::Params& p)
{
    return k->query && k->query(1024, p).approximate;
}

int maxNFor(const nbody::KernelInfo* k, const nbody::Params& p) { return isTree(k, p) ? kMaxNTree : kMaxNAllPairs; }

void resetSim(App& a)
{
    nbody::upload(a.state, a.pos0, a.vel0);
    a.kernel->launch(a.state, a.params, 0);   // prime acc for KDK
    CUDA_CHECK(cudaDeviceSynchronize());
    a.statsReset = true;
    a.countTimer = -1e9;
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

// New IC / N: regenerate on the host, reallocate the device state and the VBOs if N changed, keep the camera.
bool rebuild(App& a, const std::string& ic, const nbody::ICOptions& opt, int n)
{
    std::vector<float4> p, v;
    if (!nbody::makeIC(ic, n, a.seed, p, v, opt)) { fprintf(stderr, "cannot make IC %s\n", ic.c_str()); return false; }
    a.pos0.swap(p); a.vel0.swap(v);
    a.ic = ic; a.icOpt = opt;
    if (a.state.n != n) {
        nbody::release(a.state);
        a.state = nbody::allocate(n);
        a.params.n = n;
    }
    if (a.vbo[0] && a.vboN != n) resizeVbos(a, n);
    // brightness auto-scales with N so 4k and 1M bodies both look reasonable
    a.baseIntensity = std::clamp(std::sqrt(32768.f / float(n)) * 0.6f, 0.03f, 1.0f);
    resetSim(a);
    const double adv = nbody::dtAdvisory(a.pos0, a.params.softening, a.params.G);
    printf("ic=%s mass_spectrum=%s N=%d%s\n", ic.c_str(), opt.massSpectrum.c_str(), n,
           a.params.dt > adv ? "  (dt above the heavy-body advisory: orbits near the heaviest body are not converged)" : "");
    return true;
}

// Real interactions of one force evaluation: one instrumented launch (bitwise-equal accelerations, so the
// integrator state is unchanged) with zeroed work counters, summed on the host.
void countInteractions(App& a)
{
    const int n = a.state.n;
    if (!isTree(a.kernel, a.params)) { a.interPerEval = double(n) * n; return; }
    uint4* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, size_t(n) * sizeof(uint4)));
    CUDA_CHECK(cudaMemset(d, 0, size_t(n) * sizeof(uint4)));
    nbody::Params q = a.params;
    q.workCounters = d;
    a.kernel->launch(a.state, q, 0);
    std::vector<uint4> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, size_t(n) * sizeof(uint4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    double s = 0;
    for (const uint4& c : h) s += c.y;
    a.interPerEval = s;
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

void onKey(GLFWwindow* w, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    const auto& reg = nbody::kernelRegistry();
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
        size_t i = size_t(key - GLFW_KEY_1);
        if (i < reg.size() && a.kernel != &reg[i]) {
            if (a.state.n > maxNFor(&reg[i], a.params)) {
                printf("%s is limited to N <= %d (TDR margin); halve N first\n", reg[i].name, maxNFor(&reg[i], a.params));
            } else {
                a.kernel = &reg[i]; a.statsReset = true; a.countTimer = -1e9;
            }
        }
    }
    const int spf = a.stepsPerFrame;
    switch (key) {
    case GLFW_KEY_ESCAPE:       glfwSetWindowShouldClose(w, 1); break;
    case GLFW_KEY_SPACE:        a.paused = !a.paused; break;
    case GLFW_KEY_R:            resetSim(a); break;
    case GLFW_KEY_V:            a.vsync = !a.vsync; glfwSwapInterval(a.vsync ? 1 : 0); break;
    case GLFW_KEY_M:            a.massColor = !a.massColor; break;
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
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        a.dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &a.lastX, &a.lastY);
    }
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

void onScroll(GLFWwindow* w, double, double dy)
{
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    a.dist = std::clamp(a.dist * std::pow(0.9f, float(dy)), 0.2f, 200.f);
}

} // namespace

int main(int argc, char** argv)
{
    App a;
    a.params.n  = 32768;
    a.params.dt = 0.002f;
    // bh_warp is the tree to use from tree round 2 on (D-022); all-pairs stays on keys 1-4.
    std::string kernelName = "bh_warp", screenshot, ic = "plummer";
    int maxFrames = 0;
    // --test-keys "F:act,F:act,..." (smoke tests without a human): at frame F apply act = up | down | ic |
    // spectrum | mass, exactly as the keys PageUp / PageDown / I / H / M would.
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
        else { fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (!(a.params.softening > 0.f)) { fprintf(stderr, "--softening must be > 0\n"); return 2; }
    if (a.params.n <= 0) { fprintf(stderr, "--n must be > 0\n"); return 2; }
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
    const std::vector<std::string> icList = split(nbody::icNames()), spList = split(nbody::massSpectrumNames());

    // ---- window + GL ---------------------------------------------------------
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1600, 900, "N-body", nullptr, nullptr);
    if (!win) { fprintf(stderr, "glfwCreateWindow failed\n"); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGL(glfwGetProcAddress)) { fprintf(stderr, "gladLoadGL failed\n"); return 1; }
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(win, &a);
    glfwSetKeyCallback(win, onKey);
    glfwSetMouseButtonCallback(win, onMouseButton);
    glfwSetCursorPosCallback(win, onCursor);
    glfwSetScrollCallback(win, onScroll);
    printf("GL %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    // ---- GL buffers (registered with CUDA in resizeVbos) ----------------------
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, a.vbo);
    for (int b = 0; b < 2; ++b) {
        glBindBuffer(GL_ARRAY_BUFFER, a.vbo[b]);
        glBufferData(GL_ARRAY_BUFFER, 16, nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(GLuint(b), 4, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);
        glEnableVertexAttribArray(GLuint(b));
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ---- simulation ----------------------------------------------------------
    if (!rebuild(a, ic, a.icOpt, a.params.n)) {
        fprintf(stderr, "unknown --ic %s (%s)\n", ic.c_str(), nbody::icNames());
        return 2;
    }

    GLuint prog = linkProgram();
    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uPointScale = glGetUniformLocation(prog, "uPointScale");
    GLint uSpeedRef = glGetUniformLocation(prog, "uSpeedRef");
    GLint uIntensity = glGetUniformLocation(prog, "uIntensity");
    GLint uMassRef = glGetUniformLocation(prog, "uMassRef");
    GLint uMassColor = glGetUniformLocation(prog, "uMassColor");
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double titleTimer = glfwGetTime(), simMsAcc = 0.0;
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
        for (const auto& ks : keyScript)
            if (ks.first == totalFrames) {
                if (ks.second == "up") a.pendingN = 2;
                else if (ks.second == "down") a.pendingN = -2;
                else if (ks.second == "ic") a.pendingIC = 1;
                else if (ks.second == "spectrum") a.pendingSpectrum = 1;
                else if (ks.second == "mass") a.massColor = !a.massColor;
                printf("test key at frame %d: %s\n", totalFrames, ks.second.c_str());
            }

        // ---- pending live changes (N, IC, mass spectrum) ---------------------
        if (a.pendingN || a.pendingIC || a.pendingSpectrum) {
            int n = a.state.n;
            std::string nic = a.ic;
            nbody::ICOptions opt = a.icOpt;
            if (a.pendingN > 0) n = std::min(n * 2, maxNFor(a.kernel, a.params));
            if (a.pendingN < 0) n = std::max(n / 2, kMinN);
            if (a.pendingIC) {
                size_t i = std::find(icList.begin(), icList.end(), a.ic) - icList.begin();
                nic = icList[(i + 1) % icList.size()];
            }
            if (a.pendingSpectrum) {
                size_t i = std::find(spList.begin(), spList.end(), a.icOpt.massSpectrum) - spList.begin();
                opt.massSpectrum = spList[(i + 1) % spList.size()];
            }
            if (a.pendingN > 0 && n == a.state.n)
                printf("%s is limited to N <= %d (TDR margin)\n", a.kernel->name, maxNFor(a.kernel, a.params));
            if (n != a.state.n || nic != a.ic || opt.massSpectrum != a.icOpt.massSpectrum) rebuild(a, nic, opt, n);
            a.pendingN = a.pendingIC = a.pendingSpectrum = 0;
        }
        const int n = a.state.n;
        if (glfwGetTime() - a.countTimer > 2.0) { countInteractions(a); a.countTimer = glfwGetTime(); }

        // ---- simulate --------------------------------------------------------
        int stepsThisFrame = 0;
        if (!a.paused) {
            CUDA_CHECK(cudaEventRecord(e0));
            stepsThisFrame = a.stepsPerFrame;
            for (int s = 0; s < stepsThisFrame; ++s) nbody::step(a.state, a.params, *a.kernel, 0);
            CUDA_CHECK(cudaEventRecord(e1));
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

        if (!a.paused) {
            float ms = 0.f;
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (a.statsReset) {   // window restarts with the first frame at the new setting
                simMsAcc = 0.0; simSteps = 0; a.statsReset = false;
            }
            simMsAcc += ms;
            simSteps += stepsThisFrame;   // the steps actually run this frame, not the current setting
        }

        // ---- draw ------------------------------------------------------------
        int fbw, fbh;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.005f, 0.005f, 0.015f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        float ex = a.dist * std::cos(a.pitch) * std::sin(a.yaw);
        float ey = a.dist * std::sin(a.pitch);
        float ez = a.dist * std::cos(a.pitch) * std::cos(a.yaw);
        Mat4 mvp = mul(perspective(0.785398f, fbh > 0 ? float(fbw) / fbh : 1.f, 0.01f, 1000.f),
                       lookAt(ex, ey, ez));
        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp.m);
        glUniform1f(uPointScale, 0.010f * float(fbh));
        glUniform1f(uSpeedRef, 1.2f);
        glUniform1f(uIntensity, a.baseIntensity * a.intensity);
        glUniform1f(uMassRef, 1.0f / float(n));   // every IC has total mass 1
        glUniform1i(uMassColor, a.massColor ? 1 : 0);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, n);
        ++totalFrames;
        bool last = maxFrames > 0 && totalFrames >= maxFrames;
        if (last && !screenshot.empty()) saveScreenshot(screenshot.c_str(), fbw, fbh);
        glfwSwapBuffers(win);
        ++frames;

        // ---- title / stats ----------------------------------------------------
        double now = glfwGetTime();
        if (now - titleTimer > 0.5 || last) {
            double fps = frames / (now - titleTimer);
            double msPerStep = simSteps ? simMsAcc / double(simSteps) : 0.0;
            double rate = msPerStep > 0 && a.interPerEval > 0 ? a.interPerEval / (msPerStep * 1e-3) * 1e-9 : 0.0;
            const bool tree = isTree(a.kernel, a.params);
            snprintf(lastTitle, sizeof lastTitle,
                     "N-body | %s (%s) | N=%d | %s | %.3f ms/step | %.1f G int/s real (%s, incl. %sintegrator) | "
                     "%.0f int/body | %d steps/frame | %.0f FPS%s%s",
                     a.ic.c_str(), a.icOpt.massSpectrum.c_str(), n, a.kernel->name, msPerStep, rate,
                     tree ? "counted" : "N^2", tree ? "build + " : "", a.interPerEval / n, a.stepsPerFrame, fps,
                     a.vsync ? "" : " | no vsync", a.paused ? " | PAUSED" : "");
            glfwSetWindowTitle(win, lastTitle);
            titleTimer = now;
            frames = simSteps = 0;
            simMsAcc = 0.0;
        }
        if (last) {
            printf("%s\n", lastTitle);
            glfwSetWindowShouldClose(win, 1);
        }
    }

    for (int b = 0; b < 2; ++b) if (a.res[b]) cudaGraphicsUnregisterResource(a.res[b]);
    nbody::release(a.state);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
