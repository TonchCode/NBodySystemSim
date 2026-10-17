# nvidia2026cudaguide (NVIDIA DOC)

**Citation:** NVIDIA Corporation. *CUDA Programming Guide*, v13.4.2. Last updated 10 Sep 2026.
**Link:** https://docs.nvidia.com/cuda/cuda-programming-guide/ (landing page; the old URL `cuda-c-programming-guide/index.html` serves the same guide). Compute capability tables: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
**Verified on:** 2026-10-17 (landing page + §5.1 Compute Capabilities fetched).
**Lead-group country:** USA (vendor).

## Key facts (compute capability 8.9, §5.1)
- 64 K 32-bit registers per SM; up to 100 KB shared memory per SM and 99 KB per block. The unified L1+shared size is 128 KB.
- The page's table extraction was poorly aligned for the per-SM warp, thread and block limits (48 / 1536 / 24). Those same values are confirmed independently by the Ada Tuning Guide. **Resolved in the second pass** (see below): the raw HTML with colspans gives 48 / 1536 / 24 for cc 8.9.
- **Version note:** the online guide is **v13.4.2**. The local toolkit reports nvcc V13.4.92 (CLAUDE.md). Record which 13.4 update is installed before claiming the two match.

## Second pass, 2026-10-17: exact cells and sections (raw HTML fetched and parsed, colspans resolved)
Method: the pages were downloaded as HTML and table rows were parsed with their `colspan` attributes, because the plain-text extraction (WebFetch) misaligned merged cells. Base URL: `https://docs.nvidia.com/cuda/cuda-programming-guide/`.

**§5.1.3 Features and Technical Specifications** (`05-appendices/compute-capabilities.html#features-and-technical-specifications`). Column order: 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 10.3 | 10.7 | 11.0 | 12.x.
- **Table 30** "Device and Streaming Multiprocessor (SM) Information per Compute Capability" (`#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability`):
  - Maximum number of threads per block: **1024** (one cell across all 11 columns).
  - Warp size: 32 (all columns).
  - Maximum number of resident warps per SM: cell `48` spans 8.6/8.7/8.9 → **48 for cc 8.9**.
  - Maximum number of resident threads per SM: cell `1536` spans 8.6/8.7/8.9 → **1536**.
  - Maximum number of resident blocks per SM: 7.5=16, 8.0=32, 8.6–8.7=16, **8.9=24**.
  - Ratio of FP32 to FP64 throughput: cell `64:1` spans 8.6/8.7/8.9 → **64:1** (agrees with the whitepaper's 1/64).
  - Footnote [2]: the row is "Non-Tensor Core throughputs"; for throughput the page points to the Best Practices Guide (`#arithmetic-instructions-throughput-native-arithmetic-instructions`). **The 13.x Programming Guide no longer contains the arithmetic-instruction throughput table; it lives in nvidia2026bestpractices §12.1.1, Table 5.**
- **Table 31** "Memory Information per Compute Capability" (`#compute-capabilities-table-memory-information-per-compute-capability`): 64 K 32-bit registers per SM, 64 K per block, 255 per thread (all columns); shared memory per SM 8.9 = **100 KB**; per thread block 8.9 = **99 KB**, with footnote [3]: "Kernels relying on shared memory allocations over 48 KB per block must use dynamic shared memory" ... "and require an explicit opt-in".
- **Not on this page:** warp schedulers per SM, FP32 cores per SM, register allocation unit size. (Partition layout: see nvidia2023adawhitepaper p. 10; register granularity: see nvidia2026bestpractices §11.1.1.)

**§1.2.2.1 Thread Blocks and Grids** (`01-introduction/programming-model.html#thread-blocks-and-grids`): all threads of a block run on one SM and, "in most cases, run to completion on that SM". Also: no guarantee of scheduling order between blocks.
**§1.2.2.2 Warps and SIMT** (`01-introduction/programming-model.html#warps-and-simt`): "threads are organized into groups of 32 threads called warps", executed SIMT.
**§2.3.7 Kernel Launch and Occupancy** (`02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy`): occupancy defined as active warps over the SM maximum; high occupancy "hides latency and increases performance"; `nvcc --resource-usage` reports registers and shared memory per kernel.
**§3.2.5 Asynchronous Data Copies** (`03-advanced/advanced-kernel-programming.html#asynchronous-data-copies`): "the SM switches to execute another warp while memory operations complete."
**§3.2.6 Configuring L1/Shared Memory Balance** (`03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance`): allocations over 48 KB per block must use dynamic (not statically sized) shared memory and "require an explicit opt-in using cudaFuncSetAttribute" (`cudaFuncAttributeMaxDynamicSharedMemorySize`). This is the source for the 48 KB default.
**§2.5 Asynchronous Execution** (`02-basics/asynchronous-execution.html`): copies overlap other work only with pinned host memory (`cudaMallocHost`); otherwise `cudaMemcpyAsync` "will revert to a synchronous behavior". The copy-engine count semantics are in nvidia2026bestpractices §10.1.2, not here.

## Summary
1. The primary reference for the CUDA programming model: threads, blocks, warps, memory spaces.
2. §5.1 lists per-compute-capability limits used in R1 and R2.
3. Expected to cover warp-level primitives (R3 #7) and atomics (R3 #6). Those sections are **not yet read**; cite them only after reading.
4. The current guide lives under a new "cuda-programming-guide" URL with "05-appendices" numbering.
5. Tables 29–32 in §5.1 hold the features and technical specifications.

## What it lets us claim
The official sm_89 resource limits, to cite in R1.

## Design decision / R3 technique supported
R2, plus R3 #2, #6 and #7, once the relevant sections are read.

## Quote
"All threads of a thread block are executed by a single SM" (§1.2.2.1)
