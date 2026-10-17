# burnes2024rtx40super (NVIDIA NEWS ARTICLE)

**Citation:** Burnes, A. and Pacelli, C. "GeForce RTX 40 SUPER Series Graphics Cards Launching This January, For Supercharged Gaming & Creating, With Super-Fast AI". *NVIDIA GeForce News*, 8 January 2024.
**Link:** https://www.nvidia.com/en-us/geforce/news/geforce-rtx-4080-4070-ti-4070-super-gpu/
**Verified on:** 2026-10-17 (page fetched; title, authors and date confirmed).
**Lead-group country:** USA (vendor).

## Key fact
- The RTX 4080 SUPER has GDDR6X video memory "running at 23 Gbps".
- The article gives no bandwidth, clocks, L2 size or spec table for the 4080 SUPER. The 256-bit bus mentioned in it refers to the RTX 4070 Ti SUPER.

## Summary
1. NVIDIA's CES 2024 launch article for the RTX 4070 SUPER, 4070 Ti SUPER and 4080 SUPER.
2. It is the only NVIDIA source found that gives the 4080 SUPER memory data rate (23 Gbps).
3. Combined with the 256-bit width from nvidia2026rtx4080family, it gives 23 × 256 / 8 = 736 GB/s theoretical [D].
4. It is a marketing article, not a datasheet, so cite it for the data rate only.
5. The product page (nvidia2026rtx4080family) does not repeat the 23 Gbps figure.

## What it lets us claim
The 23 Gbps in 01-hardware moves from [M/D] (read off an nvidia-smi clock) to [N:burnes2024rtx40super]. The 736 GB/s bandwidth becomes [D] from two NVIDIA-sourced inputs. It is still not a measurement.

## Design decision / R3 technique supported
R1 memory section; the roofline bandwidth ceiling.

## Quote
"world's fastest GDDR6X video memory (VRAM) running at 23 Gbps" (NVIDIA, 2024)
