# nvidia2023adawhitepaper (NVIDIA DOC)

**Citation:** NVIDIA Corporation. *NVIDIA Ada GPU Architecture* (whitepaper), **V2.02**, © 2023. Includes "Appendix B – Ada AD103 GPU Full Specifications" (p. 32 ff.).
**Link:** https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
**Verified on:** 2026-10-17. The PDF was downloaded and its text extracted locally; the values below were read from that text.
**Lead-group country:** USA (vendor).

## What it confirms for docs/01-hardware.md (tags [S] → [N:nvidia2023adawhitepaper])
| 01-hardware claim | Whitepaper V2.02 says | Status |
|---|---|---|
| 80 SMs | Full AD103: 7 GPCs, 40 TPCs, 80 SMs | ✅ matches; our [M] 80 SMs means the 4080 SUPER uses the full AD103 |
| 7 GPCs | 7 GPCs | ✅ |
| 128 FP32 cores/SM, 10,240 total | "each SM containing 128 FP32 CUDA Cores", 10,240 total | ✅ |
| 320 Tensor (4th gen), 80 RT (3rd gen) | 80 RT Cores, 320 Tensor Cores (full AD103) | ✅ |
| 128 KB L1+shared per SM | Ada SM has 128 KB L1/Shared Memory; full AD103 has 10,240 KB L1 | ✅ |
| 65,536 registers/SM | 256 KB register file per SM; 20,480 KB for full AD103 | ✅ (256 KB / 4 B = 65,536) |
| 64 MB L2 | Full AD103: 65,536 KB L2 | ✅ |
| 256-bit bus | eight 32-bit memory controllers (256-bit) | ✅ |
| FP64 = 1/64 FP32 | AD102 text: 2 FP64 cores per SM; FP64 rate is 1/64 of FP32 | ✅ (stated for AD102; Appendix B says AD103 keeps the same SM) |
| Base/boost 2295/2550 MHz | **Not in the whitepaper.** It covers only the RTX 4080 16 GB (76 SMs, 9,728 cores, boost 2505 MHz, 22.4 Gbps, 320 W TGP) | ⚠ needs an NVIDIA RTX 4080 SUPER product page; still [S]-unsourced |

The whitepaper does not mention the RTX 4080 SUPER at all.

## Second pass, 2026-10-17 (PDF re-downloaded, `pdftotext -layout`, V2.02 confirmed on the title page)
Page numbers are the printed page footers.
| Topic | What V2.02 says | Where |
|---|---|---|
| SM partitions | AD10x SM "is divided into four processing blocks (or partitions)" | p. 10 (text above Figure 5, "Ada Streaming Multiprocessor (SM)", p. 11) |
| Per-partition register file | each partition contains "a 64 KB register file" (4 × 64 KB = the 256 KB/SM on p. 8) | p. 10 |
| Schedulers / dispatch | "one warp scheduler, one dispatch unit" per partition | p. 10 |
| FP32/INT32 split | per partition: "16 CUDA Cores that are dedicated for processing FP32 operations"; plus 16 that "can process FP32 or INT32 operations" (16 FP32 **or** 16 INT32 per clock) | p. 10 |
| Other per-partition units | one 4th-gen Tensor Core, four load/store units, an L0 instruction cache, and a Special Function Unit (SFU) "which executes transcendental and graphics interpolation instructions" | p. 10 |
| GPC/TPC layout | "Each TPC includes one PolyMorph Engine and two SMs." Each GPC has "six TPCs", but that is the **AD102** GPC (Figure 2, p. 8: 6 TPCs, 12 SMs). | p. 8 |
| AD103 TPCs per GPC | **Not stated.** Appendix B (p. 32 ff.) gives only the totals: 7 GPCs, 40 TPCs, 80 SMs. | — |
| Tensor Cores | "tailored for the matrix multiply and accumulate math operations"; Ada "includes the Hopper FP8 Transformer Engine" | p. 24, "Ada Fourth-Generation Tensor Core" (checked by extracting single PDF pages; PDF page index = printed footer) |
| Issue rate per scheduler | Not stated. | — |
| SFU lanes / throughput | Not stated here; see nvidia2026bestpractices Table 5 (16 results/clk/SM, i.e. 4 per partition [D]). | — |

Consequence: the "4 × 32 FP32 lanes" description is right in count, but half of each partition's 32 lanes are shared with INT32. In a cycle where a partition issues INT32 work on the shared half, only the 16 dedicated lanes are left for FP32.

## Summary
1. The Ada SM: 128 FP32 cores, 4th-gen Tensor cores, a 3rd-gen RT core, a 256 KB register file and 128 KB L1/shared.
2. A much larger L2 than Ampere (AD103: 64 MB against 6 MB on GA102).
3. FP64 hardware is token (2 units per SM), so FP64 runs at 1/64 rate.
4. Appendix B gives the full AD103 configuration and the RTX 4080 16 GB specifications.
5. Lists GDDR6X at 22.4 Gbps on the RTX 4080; our board's 23 Gbps is derived from nvidia-smi [M/D], not from this document.

## What it lets us claim
The R1 architecture section can cite NVIDIA for SM, cache and core counts. FP32 in the force kernels is justified by a primary source (CLAUDE.md rule 3).

## Quote
"The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations." (NVIDIA Ada whitepaper V2.02)
