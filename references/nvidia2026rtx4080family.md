# nvidia2026rtx4080family (NVIDIA PRODUCT PAGE)

**Citation:** NVIDIA Corporation. *GeForce RTX 4080 Family: Specs*. Product web page. No publication date is shown, so the year is the access year.
**Link:** https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/ (specs section, which covers both the RTX 4080 SUPER and the RTX 4080)
**Verified on:** 2026-10-17 (page fetched).
**Lead-group country:** USA (vendor).

## Key facts: GeForce RTX 4080 SUPER column (labels as on the page)
| Label | RTX 4080 SUPER | RTX 4080 |
|---|---|---|
| NVIDIA CUDA Cores | 10240 | 9728 |
| Shader Cores | Ada Lovelace, 52 TFLOPS | 49 TFLOPS |
| Boost Clock (GHz) | **2.55** | 2.51 |
| Base Clock (GHz) | **2.29** | 2.21 |
| Standard Memory Config | 16 GB GDDR6X | 16 GB GDDR6X |
| Memory Type | **GDDR6X** | GDDR6X |
| Memory Interface Width | **256-bit** | 256-bit |
| Memory data rate | **not listed** (see burnes2024rtx40super: 23 Gbps) | not listed |
| PCI Express Gen 4 | Yes | Yes |
| Resizable BAR | Yes | Yes |
| Total Graphics Power (W) | **320** | 320 |
The page says the figures describe the Founders Edition design, and add-in-card partner boards (ours is ASUS) may differ.

## Notes against docs/01-hardware.md
- Boost 2.55 GHz matches 01-hardware's 2550 MHz.
- **Base: the page gives 2.29 GHz.** 01-hardware's 2295 MHz is not confirmed at that precision. Cite the base clock as "2.29 GHz [nvidia2026rtx4080family]".
- 320 W TGP matches our [M] default power limit of 320 W.
- Consistency check [D]: 10,240 × 2 × 2.55 GHz = 52.2 TFLOPS, which matches the page's "52 TFLOPS".

## Summary
1. This is NVIDIA's own specification for the RTX 4080 SUPER, which the Ada whitepaper does not cover.
2. It confirms 10,240 CUDA cores, i.e. the full AD103's 80 SMs × 128 lanes.
3. It confirms the memory is GDDR6X on a 256-bit interface.
4. It gives the reference base and boost clocks.
5. It does not give the memory data rate or bandwidth.

## What it lets us claim
The [S] rows of 01-hardware for core count, clocks, memory type, bus width and TGP become [N:nvidia2026rtx4080family].

## Design decision / R3 technique supported
R1/R2 platform description; the peak-FLOPS denominator at reference boost.

## Quote
"Memory Type: GDDR6X" (spec table, RTX 4080 SUPER column)
