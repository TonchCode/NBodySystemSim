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
