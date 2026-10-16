# 01 — Hardware & Software Environment

Machine documentation for the N-body CUDA project. Captured **2026-10-17** directly from
the machine (`Win32_*` CIM queries, `nvidia-smi -q`, `nvcc --version`). Raw GPU dump:
[`raw/nvidia-smi-q_2026-10-17.txt`](raw/nvidia-smi-q_2026-10-17.txt) (I removed the process
list).

Source tags: **[M]** measured on this machine · **[S]** NVIDIA spec / whitepaper
(public spec). **Update 2026-10-17:** the Ada whitepaper
[N:nvidia2023adawhitepaper] confirms these [S] rows: SMs/GPCs, 128 FP32 per SM, Tensor/RT
counts, 128 KB L1/shared, 64 MB L2 and the FP64 1/64 rate. It covers only the non-SUPER
4080, though, so the **2295/2550 MHz reference clocks are still unsourced**. Quote NVIDIA's
4080 SUPER product page or don't use them · **[D]** derived from [M]/[S].

---

## 1. System summary

| Component | Value | Src |
|---|---|---|
| OS | Windows 11 Pro 64-bit, build 22621 (10.0.22621) | M |
| Motherboard | ASUS TUF GAMING B760-PLUS WIFI (Intel B760, LGA1700) | M |
| BIOS | AMI 1820, 2025-05-15 | M |
| CPU | Intel Core i5-13600K (Raptor Lake), 14 cores / 20 threads | M |
| RAM | 32 GB DDR5-6000 (2 × 16 GB ADATA XPG AX5U6000C3016G, dual channel) | M |
| Storage | Samsung 990 PRO 2 TB NVMe · Lexar NM620 1 TB NVMe | M |
| GPU | NVIDIA GeForce RTX 4080 SUPER 16 GB (ASUS board, subsystem 0x89621043) | M |
| Display | 2560×1440 @ 180 Hz, connected to the RTX 4080 SUPER | M |

### CPU detail

| | | Src |
|---|---|---|
| Topology | 6 P-cores (Raptor Cove, HT) + 8 E-cores (Gracemont, no HT) = 14C/20T | M (14C/20T) + S (P/E split) |
| Base clock (reported) | 3.5 GHz (P-core base) | M |
| L2 / L3 | 20 MB / 24 MB | M |
| ISA | AVX2 (AVX-512 is not available on Raptor Lake) | S |

> **This matters for load balancing.** The CPU has two kinds of core. If we ever split
> work between the CPU and GPU, or use the CPU for tree building, a static even split across
> threads will be limited by the slow E-cores. That is a load-balancing example from our own
> hardware.

---

## 2. GPU — NVIDIA GeForce RTX 4080 SUPER

### 2.1 Identity

| Field | Value | Src |
|---|---|---|
| Architecture | Ada Lovelace | M |
| Die | AD103-400-A1 (Device ID 0x2702) | M |
| Compute capability | **8.9** (`sm_89`) | M |
| VBIOS | 95.03.44.00.b8 | M |
| Driver model | WDDM (display attached, display active) | M |
| PCIe link | Gen 4 ×16 (host supports Gen 5; card max Gen 4) | M |
| BAR1 aperture | 16384 MiB, so **Resizable BAR is enabled** | M |

### 2.2 Compute resources

| Resource | Value | Src |
|---|---|---|
| Streaming Multiprocessors (SMs) | **80** | M |
| GPCs | 7 | S |
| CUDA cores (FP32) | 10,240 (128 per SM, 4 partitions × 32) | S |
| Tensor cores | 320 (4th gen; FP8 support) | S |
| RT cores | 80 (3rd gen) | S |
| Base / Boost clock (NVIDIA reference) | 2295 / 2550 MHz | S |
| Boost clock reported by CUDA runtime (this board) | **2610 MHz** | M |
| Max SM clock reported by nvidia-smi | 3105 MHz (the ceiling; real sustained clocks are lower) | M |
| Peak FP32 | ≈ **53.5 TFLOPS** @ 2610 MHz (10240 × 2 FMA × 2.61 GHz) | D |
| Peak FP64 | ≈ 0.84 TFLOPS (1/64 of FP32) | S/D |
| Copy engines (`asyncEngineCount`) | **1**, so H2D and D2H copies cannot run at the same time | M |
| Concurrent kernels / unified addressing | yes / yes | M |

### 2.3 Per-SM limits (compute capability 8.9)

Rows marked M were read via `cudaGetDeviceProperties` from a test program built with nvcc
13.4 + MSVC on 2026-10-17.

| Limit | Value | Src |
|---|---|---|
| Max resident threads / SM | 1536 (48 warps) | M |
| Max resident blocks / SM | 24 | M |
| Max threads / block | 1024 | S |
| 32-bit registers / SM | 65,536 (max 255 per thread) | M |
| L1 + shared memory / SM | 128 KB unified | S |
| Max shared memory / SM | 100 KB (102,400 B) | M |
| Max shared memory / block | 99 KB (101,376 B; opt-in, 48 KB default) | M |
| Warp size | 32 | M |

### 2.4 Memory hierarchy

| Level | Value | Src |
|---|---|---|
| VRAM | 16 GB GDDR6X (16376 MiB visible, 331 MiB reserved) | M |
| Bus width | 256-bit | M |
| Memory clock | 11501 MHz max reported, which is **23 Gbps effective** | M/D |
| Bandwidth | **736 GB/s** (23 Gbps × 256 bit / 8) | D |
| L2 cache | 64 MB (67,108,864 B) | M |
| Host ↔ device | PCIe 4.0 ×16, ≈ 32 GB/s theoretical per direction | D |

### 2.5 Power and thermal

| | Value | Src |
|---|---|---|
| Default power limit | 320 W | M |
| **Current power limit** | **272 W** (85% of default) | M |
| Allowed range | 150 – 352 W | M |
| Slowdown / shutdown temp | 96 °C / 101 °C | M |
| Target temp | 65 °C | M |

> ⚠ **The power limit has been lowered to 272 W.** Someone set it by hand, or a tool did
> (ASUS GPU Tweak, MSI Afterburner or the NVIDIA app). The driver's "SW Power Capping"
> counter shows about 28 minutes of power-capped time already. This directly lowers the
> sustained clocks of a compute-bound N-body kernel. **Decide which limit to use, then
> record it in every benchmark.** 272 W is a valid choice too: it gives a perf-per-watt
> angle.

---

## 3. Software stack

| Item | Version | Status | Src |
|---|---|---|---|
| NVIDIA driver | **617.42 WHQL** (CUDA UMD 13.4) | ✅ **Latest** Game Ready driver (released 2026-10-06) | M + web |
| CUDA Toolkit | **13.4.3** (`version.json`), nvcc V13.4.92 (built 2026-09-01). The online docs are v13.4.2, so the installed toolkit is one update newer | ✅ Installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`, `CUDA_PATH` set | M |
| Nsight Compute | 2026.3.1 | ✅ (kernel-level profiling) | M |
| Nsight Systems | 2026.3.2 | ✅ (timeline profiling) | M |
| Nsight VS Edition | 2026.3 | ✅ | M |
| Visual Studio | **VS Community 2026 Insiders** (prerelease; `vswhere` only finds it with `-prerelease`) | ✅ | M |
| MSVC toolsets | **14.51** (cl 19.51.36248, default) and **14.44** (cl 19.44.35228, VS 2022 v143) | ✅ both tested | M |
| CMake | Bundled with VS (`Common7\IDE\CommonExtensions\Microsoft\CMake`) | ✅ but not on PATH; use the *x64 Native Tools* prompt | M |
| Python | 3.13.4 | ✅ (useful for plots and analysis) | M |
| Unity | 6000.2.10f1, 6000.3.23f1 | ✅ | M |

### Toolchain verified end to end

On 2026-10-17 a test `.cu` (device query + an `rsqrtf` kernel) was built with
`nvcc -arch=sm_89` under `vcvars64.bat`. It built and ran with **both** MSVC toolsets:
`kernel=no error`, CUDA runtime 13.4 = driver 13.4. The NVIDIA forum report that nvcc 13.4
rejects *VS Build Tools 2026* does **not** affect this setup.

**Caveat:** Insiders is a prerelease channel, and a VS update can bring an MSVC version newer
than nvcc 13.4 officially supports. If that happens, pin the older toolset with
`vcvars64.bat -vcvars_ver=14.44` (or `-allow-unsupported-compiler` as a last resort).
**Record the exact `cl` version in every benchmark report.**

### Driver status

617.42 is the newest Game Ready driver as of 2026-10-17. NVIDIA may still ship another
release before the end of October. Windows 10 support freezes after October 2026, but this
machine runs Windows 11, so that does not affect us.

---

## 4. Environment notes that affect benchmarks

1. **WDDM + display on the compute GPU.** The Windows TDR watchdog kills any kernel that runs
   longer than about 2 s. Keep each kernel launch short (one timestep per launch). Do not
   raise `TdrDelay` without a deliberate decision.
2. **Background GPU load.** At capture time the GPU sat at 33–41% utilization with 5.2 GB
   VRAM used, from Wallpaper Engine, Discord, Riot Client/Vanguard and 3 active NVENC
   encoder sessions. For benchmark runs: close them, and record idle VRAM and utilization
   beforehand.
3. **Clock variance.** Boost clocks depend on temperature and power. Report median of N runs
   and log `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` during runs.

---

## 5. What this hardware implies for N-body (first pass, to be verified)

- **Use FP32.** FP64 runs at 1/64 speed on GeForce Ada, so a double-precision kernel would
  be about 64× slower. Use `float4` (xyz + mass) and `rsqrtf`. If we need accuracy, handle it
  in the integrator (leapfrog / Hermite) or with compensated summation. Do not use FP64 for
  the force kernel.
- **All-pairs O(N²) is compute-bound** and maps well onto shared-memory tiling, which is the
  classic GPU Gems 3 ch. 31 approach. At about 20 FLOP per interaction, the peak is about
  2.6 × 10¹² interactions/s. Expect roughly 50–70% of that in practice. N = 10⁶ is about
  10¹² interactions, so around 0.5–1 s per step.
- **16 GB VRAM is not the limit.** 10⁷ bodies × (pos + vel, 32 B) = 320 MB. The 64 MB L2
  holds about 4M `float4` positions, which is unusually large and worth exploiting in tiling
  experiments.
- **Barnes-Hut / FMM for large N.** Tree methods are irregular: threads do uneven amounts of
  work. That is exactly the **load-balancing** problem the course wants us to study.
- **CPU-GPU interop for visuals.** CUDA ↔ OpenGL/Vulkan interop can draw positions straight
  from the VBO, with no PCIe copy per frame.
