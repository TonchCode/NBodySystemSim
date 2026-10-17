# nvidia2026bestpractices (NVIDIA DOC)

**Citation:** NVIDIA Corporation. *CUDA C++ Best Practices Guide*, 13.4 (CUDA 13.4 documentation set). No last-updated date was visible in the fetched portion.
**Link:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
**Verified on:** 2026-10-17 (page fetched: title, version and table of contents).
**Lead-group country:** USA (vendor).

**Second pass 2026-10-17:** full page fetched (HTML, 433 KB); footer reads "Last updated on Oct 12, 2026". Tables parsed from raw HTML with `colspan` resolved (plain-text extraction glues footnote numbers onto cell values, e.g. "16" + footnote 5 reads as "165").

## Exact facts for compute capability 8.9 (second pass)
**§12.1.1 Throughput of Native Arithmetic Instructions, Table 5** "(Number of Results per Clock Cycle per Multiprocessor)", anchor `#arithmetic-instructions-throughput-native-arithmetic-instructions`. Columns: 7.5 | 8.0 | **8.6 / 8.9** | 9.0 | 10.0 | 10.3 | 12.0 / 12.1. The table "reflects the max theoretical throughput".
| Row (PTX example) | cc 8.9 cell | Raw cell structure |
|---|---|---|
| 32-bit floating-point add, multiply, multiply-add (`add.f32`) | **128** | `64` spans 7.5–8.0; `128` spans 8.6/8.9 → 12.x |
| 32-bit approximate floating-point reciprocal, reciprocal square root, base-2 log, base-2 exp, sine, cosine (`lg2.approx.f32`) | **16** | one cell `16` with footnote 5 spanning all 7 columns; fn 5: "for reciprocal: approximate using __fdividef(1.F, x)" |
| 64-bit floating-point add, multiply, multiply-add (`add.f64`) | 2 | per-column |
| 32-bit integer add, subtract (`add.s32`) | 128 | spans all columns |
| 32-bit integer multiply, multiply-add (`mad.lo.s32`) | 64 (fn 6: 32 for extended precision) | spans all columns |
| warp shuffle (`shfl.sync.idx.b32`) | 32 | 7.5=16, rest 32 |
→ Per SM per clock on cc 8.9: **128 FP32 FMA results vs 16 rsqrt/MUFU results, ratio 8 : 1.**
Caveat: the whitepaper (p. 10) gives INT32 as 16 ops/clock per partition (64 per SM), while this table lists 128 for `add.s32`. The two NVIDIA documents do not agree on integer-add throughput; we do not resolve this, and it does not affect the FP32 or MUFU rows.

**§12.1.6 Reciprocal Square Root** (`#reciprocal-square-root`): invoke `rsqrtf()` explicitly; the compiler turns `1.0f/sqrtf(x)` into `rsqrtf()` "only when this does not violate IEEE-754 semantics".

**§11.1.1 Calculating Occupancy** (`#calculating-occupancy`): "register allocations are rounded up to the nearest 256 registers per warp". **Caveat:** stated next to a cc 7.0 example, and the same paragraph says the allocation granularity varies by compute capability. Applying it to cc 8.9 is [T2] until Nsight Compute's occupancy section confirms it.

**§11.3 Thread and Block Heuristics** (`#thread-and-block-heuristics`): "Latency hiding and occupancy depend on the number of active warps per multiprocessor"; the grid should have more blocks than there are SMs. **§11.2 Hiding Register Dependencies** (`#hiding-register-dependencies`): keep enough active threads per SM to hide register-dependency latency (the 4-cycle figure there is for cc 7.0).

**§10.1.2 Asynchronous and Overlapping Transfers with Computation**, sub-part "Concurrent copy and execute" (`#concurrent-copy-and-execute`): a GPU with a single copy engine "can perform one asynchronous data transfer and execute kernels"; two engines allow H2D + D2H + kernels at once. The number of copy engines is `asyncEngineCount`. The overlap needs pinned host memory and different non-default streams.

## Summary
1. NVIDIA's official optimisation guide. Its top-level chapters run from Application Profiling and Parallelizing Your Application through to Deploying CUDA Applications.
2. Relevant chapters: Performance Metrics, Memory Optimizations, Execution Configuration Optimizations, Instruction Optimization, Control Flow.
3. "Execution Configuration Optimizations" covers occupancy and block-size choice (R3 #2).
4. "Control Flow" covers branch divergence (R3 #4/#5/#7).
5. "Performance Metrics" covers timing and bandwidth methodology for `results/`.

## What it lets us claim
An official methodology reference for how we time kernels and pick launch configurations. Only the table of contents was read so far; read the specific sections before quoting their content.

## Design decision / R3 technique supported
R3 #2 (execution configuration) and R3 #4/#5 (control flow / divergence).

## Quote
"register allocations are rounded up to the nearest 256 registers per warp" (§11.1.1)
