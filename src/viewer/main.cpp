// nbody_viewer: real-time CUDA-OpenGL interop view of the simulation.
//
// The simulation state never leaves the GPU. Each frame: run K leapfrog steps with the
// selected force kernel, then copy pos/vel device-to-device into two GL vertex buffers that
// are mapped into CUDA (cudaGraphicsGLRegisterBuffer). That copy is VRAM->VRAM, not over
// PCIe (one copy engine, docs/01-hardware.md). For N = 131k it is 4 MB, a few microseconds
// at 736 GB/s. The core library stays untouched, so the bench measures the same code.
//
//   nbody_viewer [--n 32768] [--kernel allpairs_tiled] [--dt 0.002] [--block 256] [--seed 42]
//                [--frames F --screenshot out.bmp]   (run F frames, save the last one, exit)
//
// Controls: LMB drag orbit Â· wheel zoom Â· 1..9 kernel Â· Space pause Â· +/- steps per frame
//           [ ] brightness Â· R reset Â· V vsync Â· Esc quit
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
const char* kVertexSrc = R"(#version 330 core
layout(location = 0) in vec4 aPos;   // xyz + mass
layout(location = 1) in vec4 aVel;
uniform mat4  uMVP;
uniform float uPointScale;
uniform float uSpeedRef;
out vec3 vColor;
void main() {
    gl_Position  = uMVP * vec4(aPos.xyz, 1.0);
    gl_PointSize = clamp(uPointScale / gl_Position.w, 1.5, 8.0);
    float t = clamp(length(aVel.xyz) / uSpeedRef, 0.0, 1.0);
    // slow = deep blue, fast = warm white
    vColor = mix(vec3(0.15, 0.35, 1.0), vec3(1.0, 0.75, 0.45), t) + vec3(0.25 * t * t);
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

// ---- app state ----------------------------------------------------------------
struct App {
    nbody::Params           params;
    nbody::DeviceState      state;
    const nbody::KernelInfo* kernel = nullptr;
    std::vector<float4>     pos0, vel0;
    unsigned                seed = 42;

    int    stepsPerFrame = 2;
    bool   paused = false, vsync = true;
    float  yaw = 0.6f, pitch = 0.35f, dist = 4.0f;
    float  intensity = 1.0f;
    bool   dragging = false;
    double lastX = 0, lastY = 0;
};

void resetSim(App& a)
{
    nbody::upload(a.state, a.pos0, a.vel0);
    a.kernel->launch(a.state, a.params, 0);   // prime acc for KDK
    CUDA_CHECK(cudaDeviceSynchronize());
}

void onKey(GLFWwindow* w, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    App& a = *static_cast<App*>(glfwGetWindowUserPointer(w));
    const auto& reg = nbody::kernelRegistry();
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
        size_t i = size_t(key - GLFW_KEY_1);
        if (i < reg.size()) a.kernel = &reg[i];
    }
    switch (key) {
    case GLFW_KEY_ESCAPE:       glfwSetWindowShouldClose(w, 1); break;
    case GLFW_KEY_SPACE:        a.paused = !a.paused; break;
    case GLFW_KEY_R:            resetSim(a); break;
    case GLFW_KEY_V:            a.vsync = !a.vsync; glfwSwapInterval(a.vsync ? 1 : 0); break;
    case GLFW_KEY_EQUAL:
    case GLFW_KEY_KP_ADD:       a.stepsPerFrame = std::min(a.stepsPerFrame * 2, 64); break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:  a.stepsPerFrame = std::max(a.stepsPerFrame / 2, 1); break;
    case GLFW_KEY_LEFT_BRACKET: a.intensity /= 1.25f; break;
    case GLFW_KEY_RIGHT_BRACKET:a.intensity *= 1.25f; break;
    default: break;
    }
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
    std::string kernelName = "allpairs_tiled", screenshot;
    int maxFrames = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if      (k == "--n")          a.params.n = atoi(argv[i + 1]);
        else if (k == "--frames")     maxFrames = atoi(argv[i + 1]);
        else if (k == "--screenshot") screenshot = argv[i + 1];
        else if (k == "--kernel") kernelName = argv[i + 1];
        else if (k == "--dt")     a.params.dt = float(atof(argv[i + 1]));
        else if (k == "--block")  a.params.blockSize = atoi(argv[i + 1]);
        else if (k == "--q")      a.params.threadsPerBody = atoi(argv[i + 1]);
        else if (k == "--unroll") a.params.unroll = atoi(argv[i + 1]);
        else if (k == "--seed")   a.seed = unsigned(atoi(argv[i + 1]));
        else { fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    a.kernel = nbody::findKernel(kernelName);
    if (!a.kernel) { fprintf(stderr, "unknown kernel '%s'\n", kernelName.c_str()); return 2; }

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

    // ---- simulation ----------------------------------------------------------
    const int n = a.params.n;
    nbody::makePlummer(n, a.seed, a.pos0, a.vel0);
    a.state = nbody::allocate(n);
    resetSim(a);

    // ---- GL buffers registered with CUDA --------------------------------------
    GLuint vao, vbo[2];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, vbo);
    for (int b = 0; b < 2; ++b) {
        glBindBuffer(GL_ARRAY_BUFFER, vbo[b]);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(n) * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(GLuint(b), 4, GL_FLOAT, GL_FALSE, sizeof(float4), nullptr);
        glEnableVertexAttribArray(GLuint(b));
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsResource* res[2];
    for (int b = 0; b < 2; ++b)
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res[b], vbo[b], cudaGraphicsRegisterFlagsWriteDiscard));

    GLuint prog = linkProgram();
    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uPointScale = glGetUniformLocation(prog, "uPointScale");
    GLint uSpeedRef = glGetUniformLocation(prog, "uSpeedRef");
    GLint uIntensity = glGetUniformLocation(prog, "uIntensity");
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double titleTimer = glfwGetTime(), simMsAcc = 0.0;
    int frames = 0, simFrames = 0, totalFrames = 0;
    char lastTitle[256] = "";
    // brightness auto-scales with N so 4k and 1M bodies both look reasonable
    const float baseIntensity = std::clamp(std::sqrt(32768.f / float(n)) * 0.6f, 0.03f, 1.0f);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // ---- simulate --------------------------------------------------------
        if (!a.paused) {
            CUDA_CHECK(cudaEventRecord(e0));
            for (int s = 0; s < a.stepsPerFrame; ++s) nbody::step(a.state, a.params, *a.kernel, 0);
            CUDA_CHECK(cudaEventRecord(e1));
        }

        // ---- publish to GL (VRAM -> VRAM) -------------------------------------
        CUDA_CHECK(cudaGraphicsMapResources(2, res, 0));
        void* ptr[2];
        size_t bytes = 0;
        for (int b = 0; b < 2; ++b)
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr[b], &bytes, res[b]));
        CUDA_CHECK(cudaMemcpyAsync(ptr[0], a.state.pos, size_t(n) * sizeof(float4), cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(ptr[1], a.state.vel, size_t(n) * sizeof(float4), cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(2, res, 0));

        if (!a.paused) {
            float ms = 0.f;
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            simMsAcc += ms;
            ++simFrames;
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
        glUniform1f(uIntensity, baseIntensity * a.intensity);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, n);
        ++totalFrames;
        bool last = maxFrames > 0 && totalFrames >= maxFrames;
        if (last && !screenshot.empty()) saveScreenshot(screenshot.c_str(), fbw, fbh);
        glfwSwapBuffers(win);
        ++frames;
        if (last) {
            printf("%s\n", lastTitle);
            glfwSetWindowShouldClose(win, 1);
        }

        // ---- title / stats ----------------------------------------------------
        double now = glfwGetTime();
        if (now - titleTimer > 0.5) {
            double fps = frames / (now - titleTimer);
            double msPerStep = simFrames ? simMsAcc / (simFrames * double(a.stepsPerFrame)) : 0.0;
            double tips = msPerStep > 0 ? double(n) * n / (msPerStep * 1e-3) * 1e-12 : 0.0;
            char* title = lastTitle;
            snprintf(title, sizeof lastTitle,
                     "N-body | %s | N=%d | %.3f ms/step (%.2f T int/s incl. integrator) | %d steps/frame | %.0f FPS%s%s",
                     a.kernel->name, n, msPerStep, tips, a.stepsPerFrame, fps,
                     a.vsync ? "" : " | no vsync", a.paused ? " | PAUSED" : "");
            glfwSetWindowTitle(win, title);
            titleTimer = now;
            frames = simFrames = 0;
            simMsAcc = 0.0;
        }
    }

    for (int b = 0; b < 2; ++b) cudaGraphicsUnregisterResource(res[b]);
    nbody::release(a.state);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
