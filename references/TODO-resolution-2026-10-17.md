# TODO resolution: `docs/final/02-gpu-architecture.md`, 2026-10-17

Produced by `literature-scout`, second pass. **The chapter was not edited.** This file maps each
`TODO(source needed: …)` to a source, for the tech-writer to apply.
Every source below was fetched on 2026-10-17. NVIDIA HTML pages were downloaded and their tables parsed
with `colspan` resolved. The whitepaper PDF was re-downloaded and read with `pdftotext`, page by page.

**Tally: 22 TODO markers** (the line-12 convention sentence is not counted).
**15 resolved**, **3 partly resolved**, **3 out of scope for literature** (they need a build or profiler run, not a citation), **1 not found**.

Status key: ✅ resolved · ◐ partial · ⛔ not a literature item · ✗ not found

| # | § | TODO text (abridged) | Status | Source key → exact location | Verified fact |
|---|---|---|---|---|---|
| 1 | 2.1 | TPCs per GPC on AD103 | ✗ | nvidia2023adawhitepaper, Appendix B p. 32; GPC description p. 8 | AD103 totals only (7 GPCs / 40 TPCs / 80 SMs). The "six TPCs" per GPC on p. 8 describes the **AD102** GPC (Fig. 2: 6 TPCs, 12 SMs). Searched: whitepaper full text, CUDA Programming Guide §5.1, Ada Tuning Guide. Keep the TODO, or drop the sentence and keep "40 is not divisible by 7" [D]. |
| 2 | 2.2.1 | SM partition layout | ✅ | nvidia2023adawhitepaper p. 10 (text for Fig. 5, p. 11) | The SM "is divided into four processing blocks (or partitions)". Per partition: 16 FP32-only cores + 16 FP32-or-INT32 cores, i.e. 32 FP32 lanes. 4 × 32 = 128 holds. |
| 3 | 2.2.1 | warp schedulers per partition and issue rate | ◐ | nvidia2023adawhitepaper p. 10 | "one warp scheduler, one dispatch unit" per partition, so 4 per SM [D]. **Issue rate per scheduler per clock: not stated** in the whitepaper, Programming Guide §5.1, Best Practices or the Ada Tuning Guide. |
| 4 | 2.2.2 | FP32/INT32 datapath split | ✅ | nvidia2023adawhitepaper p. 10 | Per partition, 16 lanes are FP32-only and 16 lanes do "16 FP32 operations per clock OR 16 INT32". INT32 work on the shared half takes those lanes away from FP32. ⚠ Best Practices Table 5 lists `add.s32` at 128/clk/SM for 8.9 while the whitepaper implies at most 64 INT32/clk/SM. The two NVIDIA sources disagree; cite the whitepaper for the datapath and do not quote an INT32 peak. |
| 5 | 2.2.3 | arithmetic-instruction throughput table for cc 8.9 | ✅ | **nvidia2026bestpractices §12.1.1, Table 5** (`cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions`). The 13.x Programming Guide no longer holds the table; its Table 30 footnote [2] points here. | Column "8.6 / 8.9", in results per clock per SM: **FP32 add/mul/FMA = 128**; **32-bit approximate reciprocal, rsqrt, log2, exp2, sin, cos = 16**. Ratio **8 : 1**, which confirms the smoke-test figure. Fix the citation: it is the Best Practices Guide, not the Programming Guide. |
| 6 | 2.2.4 | per-partition register file size | ✅ | nvidia2023adawhitepaper p. 10 | "a 64 KB register file" per partition. 4 × 64 KB = the 256 KB/SM on p. 8. |
| 7 | 2.2.5 | 48 KB default per-block limit | ✅ | nvidia2026cudaguide §3.2.6 "Configuring L1/Shared Memory Balance" (`03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance`); also §5.1.3 Table 31 footnote [3] | Allocations "over 48 KB per block" must use dynamic shared memory and an explicit opt-in via `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`. Tag becomes [N:nvidia2026cudaguide]. |
| 8 | 2.2.6 | Tensor core operation description | ✅ | nvidia2023adawhitepaper p. 24, "Ada Fourth-Generation Tensor Core" | Tensor Cores are "tailored for the matrix multiply and accumulate math operations". The same page says Ada "includes the Hopper FP8 Transformer Engine", so the FP8 [S] in §2.2.6 can also become [N]. The Ada Tuning Guide §1.4.1.2 says the same. |
| 9 | 2.2.6 | SASS listing of allpairs_tiled | ⛔ | — | Needs `cuobjdump -sass` / Nsight Compute on our binary (bench-profiler). No literature answer. |
| 10 | 2.3 | block-to-SM residency and warp execution | ✅ | nvidia2026cudaguide §1.2.2.1 (`01-introduction/programming-model.html#thread-blocks-and-grids`) and §1.2.2.2 (`#warps-and-simt`) | All threads of a block run on one SM and "in most cases, run to completion on that SM". There is no guaranteed scheduling order between blocks. Threads are grouped into "groups of 32 threads called warps" (SIMT). |
| 11 | 2.3 | max threads per block | ✅ | nvidia2026cudaguide §5.1.3 **Table 30** (`05-appendices/compute-capabilities.html#compute-capabilities-table-device-and-streaming-multiprocessor-sm-information-per-compute-capability`) | 1024, one cell spanning every compute capability. The same table gives cc 8.9 = 48 warps / 1536 threads / 24 blocks per SM, now cell-verified. §2.7 "Max threads per block [S], unconfirmed" → [M; N:nvidia2026cudaguide]. |
| 12 | 2.3 | latency hiding by warp switching | ✅ | nvidia2026bestpractices §11.3 "Thread and Block Heuristics" (`#thread-and-block-heuristics`); nvidia2026cudaguide §3.2.5 (`03-advanced/advanced-kernel-programming.html#asynchronous-data-copies`) and §2.3.7 (`02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy`) | BP: "Latency hiding and occupancy depend on the number of active warps per multiprocessor". PG §3.2.5: "the SM switches to execute another warp while memory operations complete". |
| 13 | 2.3 remark 1 | register allocation granularity for cc 8.9 | ◐ [T2] | nvidia2026bestpractices §11.1.1 "Calculating Occupancy" (`#calculating-occupancy`) | "register allocations are rounded up to the nearest 256 registers per warp". **Caveat:** this sits next to a cc 7.0 example, and the same paragraph says granularity varies by compute capability. Nothing found that states it for 8.9. **If 256/warp applies to 8.9, the chapter's "two-register headroom" is wrong; see Contradictions §A.** Confirm with Nsight Compute's occupancy section (`launch__occupancy_limit_registers`) at 40 and 41+ registers. |
| 14 | 2.3 remark 2 | ptxas shared-memory usage | ⛔ | — | Rebuild with `-Xptxas -v` or `nvcc --resource-usage` (PG §2.3.7 names that option). Measurement, not literature. |
| 15 | 2.4.2 | Nsight Compute L2 hit rate | ⛔ | — | Profiler measurement (bench-profiler). |
| 16 | 2.4.3 | RTX 4080 SUPER memory type | ✅ | **nvidia2026rtx4080family**, "Memory Specs": Memory Type | **GDDR6X**, 16 GB, Memory Interface Width **256-bit**. Tag [S] → [N:nvidia2026rtx4080family]. |
| 17 | 2.4.3 | 4080 SUPER memory data rate, or measured bandwidth | ◐ | **burnes2024rtx40super** (NVIDIA GeForce News, 8 Jan 2024) | GDDR6X "running at 23 Gbps". The product page does **not** list a data rate. 736 GB/s therefore becomes [D] from two NVIDIA-sourced inputs (23 Gbps × 256 bit / 8). The **measured** device-to-device bandwidth is still a bench-profiler job. |
| 18 | 2.4.4 | PCIe 4.0 per-lane rate and encoding | ✅ (encoding [T2]) | **pcisig2011pcie4** (PCI-SIG press release, 29 Nov 2011, via Design & Reuse reprint) for 16 GT/s; **solomon2014pcie16gts** (Synopsys) for 128b/130b | 16 GT/s per lane, 128b/130b → 16 × 128/130 / 8 ≈ 1.97 GB/s per lane per direction, so **≈ 31.5 GB/s per direction for ×16** [D]. The chapter's "~32" is rounded; consider writing 31.5. pcisig.com served a bot-check page and was not fetched. The encoding source is secondary and predates the final spec. NVIDIA's product page confirms "PCI Express Gen 4: Yes". |
| 19 | 2.4.4 | copy/compute overlap with asyncEngineCount = 1 | ✅ | nvidia2026bestpractices §10.1.2, "Concurrent copy and execute" (`#concurrent-copy-and-execute`) | A single copy engine "can perform one asynchronous data transfer and execute kernels". It needs pinned host memory and a non-default stream (also PG §2.5: non-pinned `cudaMemcpyAsync` reverts to synchronous). So one copy **can** overlap a kernel. H2D and D2H cannot overlap each other. |
| 20 | 2.6 | Microsoft documentation of default TdrDelay | ✅ | **microsoft2025tdrkeys**, section "TdrDelay" | "The default value is 2 seconds". TdrDelay is the time the GPU may delay a preempt request, "effectively the timeout threshold". The page tells end users not to change the keys. Suggested wording: "does not yield to preemption within ~2 s" rather than "runs longer than 2 s". |
| 21 | 2.6 clocks table | NVIDIA 4080 SUPER base / boost | ✅ | **nvidia2026rtx4080family**, "GPU Engine Specs" | Base **2.29 GHz**, boost **2.55 GHz** (Founders Edition figures; partner boards may differ). |
| 22 | 2.7 summary row | Reference base / boost clock | ✅ | same as #21 | 2.29 / 2.55 GHz [N:nvidia2026rtx4080family]. |

Tally: ✅ #2, 4, 5, 6, 7, 8, 10, 11, 12, 16, 18, 19, 20, 21, 22 (15) · ◐ #3, 13, 17 (3) · ⛔ #9, 14, 15 (3) · ✗ #1 (1) = 22.

## Other [S] / "unconfirmed" tags in the chapter that these sources upgrade (not TODO markers)
- §2.2.1 / §2.7 "SM partitions 4 × 32 [S], unconfirmed" → [N:nvidia2023adawhitepaper p. 10]. Add the 16 + 16 FP32/INT32 nuance.
- §2.2.6 "support FP8 [S]" → [N:nvidia2023adawhitepaper p. 24; nvidia2026adatuning §1.4.1.2].
- §2.7 "Max threads per block [S], unconfirmed" → [N:nvidia2026cudaguide Table 30].
- §2.7 GPCs: the "7 GPCs listed as spec [S]" row is already [N:nvidia2023adawhitepaper].
- §2.5 / §2.2.7 FP64 1/64: also in nvidia2026cudaguide Table 30 ("Ratio of FP32 to FP64 Throughput" 64:1 for 8.6/8.7/8.9).
- Consistency check [D]: product page "Shader Cores 52 TFLOPS" = 10,240 × 2 × 2.55 GHz = 52.2 TFLOPS.

## Contradictions and corrections for the tech-writer
**A. Register headroom (§2.3 remark 1), chapter only, [T2].** If cc 8.9 allocates registers in units of 256 per warp (BP §11.1.1):
40 regs × 32 = 1,280 = exactly 5 × 256, so 8 warps × 1,280 = 10,240 per block and 6 blocks fit.
At **41** regs: 1,312 rounds up to 1,536 per warp, giving 12,288 per block and ⌊65,536 / 12,288⌋ = **5 blocks**.
The headroom would be **0 registers, not 2**. One more register already costs 17 % occupancy. Keep this conditional until Nsight Compute confirms it.

**B. The SFU hypothesis (§2.2.3) should be reworded.** Table 5 gives a **pipe throughput** of 16 MUFU results per clock per SM, which is 4 per partition [D, with whitepaper p. 10's one SFU per partition]. A warp's `rsqrtf` therefore occupies a partition's SFU for 8 clocks, while an FP32 warp-instruction takes 1 clock on its 32 lanes [D]. That is "one rsqrt costs the FP32-pipe time of 8 FMAs". It is **not** sourced as "issue capacity", because the scheduler issue rate is undocumented (#3). Whether the kernel is MUFU-bound depends on how many FP32 instructions per interaction the SASS contains. If it has 8 or more (the 20-FLOP convention suggests it does), the FP32 pipe, not MUFU, is the tighter limit [D, hypothesis]. Nsight Compute pipe utilisation (`sm__inst_executed_pipe_xu`, `..._pipe_fma`) decides it.

**C. Throughput table location.** The chapter attributes it to the CUDA Programming Guide. In 13.x it is in the **Best Practices Guide §12.1.1, Table 5**.

## Against `docs/01-hardware.md`
- **Base clock:** 01-hardware says 2295 MHz [S]; NVIDIA's page says **2.29 GHz**. This is a precision difference, not a conflict, but 2295 has no source. Boost 2550 MHz matches 2.55 GHz.
- **Partitions "4 × 32":** correct in count, but 16 of each 32 lanes are shared with INT32 (whitepaper p. 10). 01-hardware does not mention this.
- **23 Gbps:** 01-hardware *assumes* it from an nvidia-smi clock reading. NVIDIA states it (burnes2024rtx40super), so the assumption was right, and the source is now NVIDIA, not the clock conversion.
- **Bandwidth 736 GB/s:** inputs are now NVIDIA-sourced. Still theoretical, still unmeasured.
- **PCIe ≈ 32 GB/s:** ≈ 31.5 GB/s per direction after 128b/130b. Rounding only.
- **TDR "kills any kernel that runs longer than ~2 s":** Microsoft defines the timeout as failure to honour a preempt request within TdrDelay (default 2 s). Same practical rule, more precise mechanism.
- No number in 01-hardware was found to be wrong.

## Searched and not found
- AD103 GPC/TPC distribution: whitepaper (full text), PG §5.1, Ada Tuning Guide. No NVIDIA statement.
- Warp-scheduler issue rate per clock on Ada: whitepaper, PG §5.1, BP §11–12, Ada Tuning Guide. Not stated.
- Register allocation unit for cc 8.9 specifically: PG §5.1 Tables 29–32 (absent); BP §11.1.1 (generic, cc 7.0 context). Nsight Compute docs not yet read.
- PCI-SIG primary page for 128b/130b: pcisig.com returned a JavaScript bot-check. Not bypassed.
