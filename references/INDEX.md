# references/INDEX.md: evidence base

Maintained by `literature-scout`. First pass: **2026-10-17**. Second pass (TODO closure for `docs/final/02-gpu-architecture.md`): **2026-10-17**, see [TODO-resolution-2026-10-17.md](TODO-resolution-2026-10-17.md).
Rule: ✅ = landing page fetched on 2026-10-17, with title, authors, year, venue and DOI/arXiv ID confirmed. ⬜ = lead, **do not cite**.
Country = country of the lead group's institution, not the authors' nationality.

Third pass (critic #8: Chinese leads, R3 evidence): **2026-10-17**. See [R3-evidence-map.md](R3-evidence-map.md) for technique → source → strength, and for the documented "preferably Chinese" search.

## Verified entries (38: 25 recent papers + 4 background + 4 NVIDIA docs + 5 spec/platform sources)

### A. GPU N-body / hierarchical methods (2025–2026)
| Key | Year | Venue | Topic | Country | R3 | ✓ |
|---|---|---|---|---|---|---|
| [madan2025stochastic](madan2025stochastic.md) | 2025 | SIGGRAPH '25 (ACM) | Stochastic Barnes-Hut on GPU | Canada/USA | bg for #4 | ✅ |
| [nagarajan2025rtbarneshut](nagarajan2025rtbarneshut.md) | 2025 | PPoPP '25 (ACM) | Barnes-Hut on RT cores (OptiX) | USA | motivation #4–#7; optional extra | ✅ |
| [stucker2026jzfmm](stucker2026jzfmm.md) | 2026 | arXiv preprint | GPU-native FMM, Morton sort, non-recursive dual-tree walk | Austria | #5, #7 | ✅ |
| [frontiere2025crkhacc](frontiere2025crkhacc.md) | 2025 | arXiv (venue unconfirmed) | CRK-HACC, GPU-resident tree solvers, 4T particles | USA | context #8 | ✅ |
| [yu2026cube2](yu2026cube2.md) | 2026 | Sci. China PMA | CUBE2 multi-level PM+PP (CPU); cost-sorted LB | **China** | #8 (CPU side), analogy #6 | ✅ |
| [liu2025swphotons](liu2025swphotons.md) | 2025* | Engineering Reports | PM-FMM on Sunway; dynamic LB, pipelined traversal | **China** | analogy #6 | ✅ *online 2023 |
| [thaler2026sfcneighbor](thaler2026sfcneighbor.md) | 2026 | arXiv; accepted IPDPS 2026 | Hilbert-SFC layout for GPU neighbor lists (astro SPH) | Switzerland | **#5** | ✅ |
| [wu2025dragon3](wu2025dragon3.md) | 2025 | arXiv; IAU Symp. 398 proc. | NBODY6++GPU ported to HIP for Chinese DCU clusters | Germany (+NAOC/PKU/SYSU) | context only | ✅ |
| [han2025jiutian](han2025jiutian.md) | 2025 | arXiv (venue unconfirmed) | Jiutian/CSST simulation suite: Gadget-3/4, Cube (CPU codes) | **China** (SJTU) | none; negative R4 result | ✅ (author count to re-check) |

### B. GPU load balancing / scheduling / divergence (2025–2026)
| Key | Year | Venue | Topic | Country | R3 | ✓ |
|---|---|---|---|---|---|---|
| [maeda2026gtap](maeda2026gtap.md) | 2026 | arXiv preprint | Persistent kernel + work stealing; FMM dual-tree traversal | Japan | **#6 vs #7** | ✅ |
| [li2025staticbatching](li2025staticbatching.md) | 2025 | arXiv preprint | Static prefix-sum task mapping for irregular work | **China** (Alibaba) | #6 (static arm), #3 | ✅ |
| [shetty2026gpuqueues](shetty2026gpuqueues.md) | 2026 | arXiv preprint | Lock-/wait-free GPU queues (AMD only) | USA | #6 | ✅ |
| [jin2026eventtensor](jin2026eventtensor.md) | 2026 | MLSys 2026 (per arXiv) | Static vs dynamic scheduling in persistent megakernels | USA (likely) | #6 | ✅ |
| [galgoczi2026ser](galgoczi2026ser.md) | 2026 | arXiv preprint | Reordering divergent work on **RTX 4090 (Ada)** | USA | #5, #6 | ✅ |
| [dale2026warpdivergence](dale2026warpdivergence.md) | 2026 | arXiv preprint (1 author) | Divergence cost ≈ 32/k, Pascal→Blackwell (no Ada) | unknown | #4 analysis | ✅ [T2] |
| [nanda2025amrexlb](nanda2025amrexlb.md) | 2025 | PEARC '25 (ACM) | SFC / knapsack / painter's-partition LB | USA | #5, #8 | ✅ |
| [zhang2026wavetune](zhang2026wavetune.md) | 2026 | arXiv preprint | Wave-aware latency model; the ceil(B/SM) step model is incomplete | **China** (SJTU + Alibaba) | **#2** | ✅ |
| [zhang2025hytis](zhang2025hytis.md) | 2025 | SC '25 (ACM) | Two-level tile scheduling against wave quantization | **China** (Wuhan U) | #2, #3 | ✅ (Crossref; ACM 403) |
| [shi2025libra](shi2025libra.md) | 2025 | arXiv preprint | Threshold splitting + occupancy-aware scheduling; **RTX 4090** | **China** (BUPT) | #2, #3 | ✅ |
| [xia2025voltrix](xia2025voltrix.md) | 2025 | USENIX ATC '25 | Persistent kernel + two-stage balanced partition | **China** (Wuhan U) | #6 | ✅ |
| [huang2026weave](huang2026weave.md) | 2026 | arXiv preprint | Persistent megakernel; atomicAdd tile claiming; dynamic SM roles | **China** (SJTU) | **#6** | ✅ |
| [chen2026gmatch](chen2026gmatch.md) | 2026 | arXiv preprint | Atomic task pool + warp-level work stealing; **RTX 4090** | **China** (SJTU) | **#6, #7** | ✅ |
| [zhong2025hybrimoe](zhong2025hybrimoe.md) | 2025 | arXiv; DAC '25 per comment | Profiled CPU+GPU split, min max(T_CPU, T_GPU) | **China** (PKU) | #8 (analogy) | ✅ |
| [xu2025tascell](xu2025tascell.md) | 2025 | J. Information Processing 33 | Hybrid CPU-GPU work stealing on a workstation | Japan | #8 | ✅ |
| [nozal2025coexecution](nozal2025coexecution.md) | 2025 | J. Supercomputing 81(3) | CPU-GPU co-execution: 0.91 efficiency, 1.31x | not confirmed | #8 | ✅ (Crossref) [T2] |

### C. Background classics (labelled BACKGROUND)
| Key | Year | Venue | Topic | Country | R3 | ✓ |
|---|---|---|---|---|---|---|
| [barnes1986hierarchical](barnes1986hierarchical.md) | 1986 | Nature 324 | Original Barnes-Hut | not verified | #4–#7 | ✅ |
| [nyland2007nbody](nyland2007nbody.md) | 2007 | GPU Gems 3, ch. 31 | Tiled all-pairs CUDA; q threads/body at small N | USA | **#1, #2, #3** | ✅ |
| [burtscher2011barneshut](burtscher2011barneshut.md) | 2011 | GPU Computing Gems Emerald, ch. 6 | All-GPU Barnes-Hut; sort + warp-vote traversal | USA | **#4, #5, #7** | ✅ |
| [wang2021photonsgpu](wang2021photonsgpu.md) | 2021 | RAA 21(11) 281 | PhotoNs-GPU: PM-FMM on GPU; FP32 ≈ 2x FP64 | China (affiliation not fetched) | bg #1 (FP32) | ✅ |

### D. NVIDIA documentation
| Key | Version | Date | Use | ✓ |
|---|---|---|---|---|
| [nvidia2026cudaguide](nvidia2026cudaguide.md) | CUDA Programming Guide v13.4.2 | updated 2026-09-10 | sm_89 limits (§5.1.3 Tables 30–31, cells resolved), block/warp model (§1.2.2), 48 KB opt-in (§3.2.6) | ✅ |
| [nvidia2026bestpractices](nvidia2026bestpractices.md) | CUDA C++ Best Practices Guide 13.4 | updated 2026-10-12 | **Table 5 throughput (FP32 128, rsqrt/MUFU 16 per clk/SM, cc 8.9)**, register granularity, copy engines, latency hiding | ✅ |
| [nvidia2026adatuning](nvidia2026adatuning.md) | Ada Tuning Guide v13.4 | updated 2026-10-12 | 48 warps / 24 blocks / 100 KB smem per SM | ✅ |
| [nvidia2023adawhitepaper](nvidia2023adawhitepaper.md) | Ada GPU Architecture whitepaper V2.02 | © 2023 | AD103 specs → confirms 01-hardware [S] rows; SM partitions, schedulers, FP32/INT32 split (p. 10) | ✅ |

### E. Product specifications and platform documentation (second pass)
| Key | Source | Date | Use | ✓ |
|---|---|---|---|---|
| [nvidia2026rtx4080family](nvidia2026rtx4080family.md) | NVIDIA RTX 4080 family product page | undated (accessed 2026-10-17) | 4080 SUPER: 10240 cores, base 2.29 / boost 2.55 GHz, GDDR6X, 256-bit, 320 W | ✅ |
| [burnes2024rtx40super](burnes2024rtx40super.md) | NVIDIA GeForce News | 2024-01-08 | 4080 SUPER GDDR6X at 23 Gbps | ✅ |
| [microsoft2025tdrkeys](microsoft2025tdrkeys.md) | Microsoft Learn, TDR registry keys | ms.date 2025-11-05 | TdrDelay default 2 s | ✅ |
| [pcisig2011pcie4](pcisig2011pcie4.md) | PCI-SIG press release (Design & Reuse reprint) | 2011-11-29 | PCIe 4.0 = 16 GT/s | ✅ (reprint; pcisig.com not fetched) |
| [solomon2014pcie16gts](solomon2014pcie16gts.md) | Synopsys technical article | 2014-07-21 | 128b/130b at 16 GT/s | ✅ [T2] secondary, pre-final-spec |

## R3 coverage map
| R3 # | Technique | Best sources |
|---|---|---|
| 1 | Tiled all-pairs baseline | nyland2007nbody (bg), nvidia2026adatuning, wang2021photonsgpu (bg, FP32). **No 2025–26 paper** |
| 2 | Block/tile/unroll sweep vs occupancy | **zhang2026wavetune**, zhang2025hytis, shi2025libra; nyland2007nbody (bg), nvidia2026adatuning, nvidia2026bestpractices |
| 3 | Multiple threads per body | **zhang2025hytis**, shi2025libra, li2025staticbatching; nyland2007nbody (bg) |
| 4 | Naive Barnes-Hut traversal | dale2026warpdivergence, nagarajan2025rtbarneshut, madan2025stochastic; burtscher2011barneshut (bg) |
| 5 | Morton/Hilbert ordering | stucker2026jzfmm, **thaler2026sfcneighbor**, galgoczi2026ser, nanda2025amrexlb; burtscher2011barneshut (bg) |
| 6 | Persistent threads + queue | maeda2026gtap, **huang2026weave**, **chen2026gmatch**, **xia2025voltrix**, li2025staticbatching (static arm), shetty2026gpuqueues, jin2026eventtensor |
| 7 | Warp-level stealing / cooperative traversal | maeda2026gtap, **chen2026gmatch**, stucker2026jzfmm; burtscher2011barneshut (bg) |
| 8 | CPU+GPU split | Ragagnin 2026 (OpenGadget3, not yet in bib), **xu2025tascell**, nozal2025coexecution, zhong2025hybrimoe, yu2026cube2, frontiere2025crkhacc (argues against shipping work to the CPU) |

Strengths (direct / transferable / analogy) and the quotes are in [R3-evidence-map.md](R3-evidence-map.md).

**Gaps:** no 2025–2026 paper for #1. None on Hilbert vs Morton for GPU *tree traversal*. None on CPU+GPU splitting with P/E-core hybrid CPUs. **No 2025–26 Chinese-led GPU N-body/tree/FMM paper found**; the Chinese 2025–26 support for #2, #3, #6, #7 and #8 is transferable GPU-scheduling work. The search log is in R3-evidence-map §9.

## Verified but not catalogued (relevance too low for this pass; landing pages fetched)
- Cheng et al., "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs", arXiv:2512.22219 (2025). Persistent megakernel with decentralized SM scheduling (LLM inference).
- Brandt, "On Parallel and Distributed N-Body Simulations", *Mathematics* 14(9) 1403, 2026, DOI 10.3390/math14091403. Costzones / hashed octree, distributed CPU.
- Ragagnin et al., "OpenGadget3 GPU solver tests", *Astronomy and Computing* 57, 101131, 2026, arXiv:2606.17145. GPU short-range gravity, 2–5x speedup. **Round 3:** HTML re-read. The tree is rebuilt on the CPU and offloaded each timestep, then walked (Barnes-Hut) on the GPU, overlapped with CPU work. This is now the direct #8 source in the evidence map. Promote it to the bib once the full author list is captured.
- Tozlu & Zhou, "CoopRT: Accelerating BVH Traversal for Ray Tracing via Cooperative Threads", ISCA '25, pp. 166–179, DOI 10.1145/3695053.3731118 (NC State; Crossref metadata only). Idle lanes help busy lanes in BVH traversal; it is a hardware proposal, so analogy for #7 only.
- Brotas, Nobre, Ilic, "Thievory: Graph Processing with Multi-GPU Memory Stealing", ICPP '25, DOI 10.1145/3754598.3754653 (Semantic Scholar metadata; ACM 403). It steals neighbouring GPUs' *memory* for out-of-memory graphs. Not compute load balancing, so **not relevant**.
- Lacopo et al., arXiv:2601.01935 (5 Jan 2026, 10 authors), "Accelerating cosmological simulations on GPUs: a step towards sustainability and green-awareness". GPU PINOCCHIO, 8x speed and 8x energy. Probably the same work as the *Astronomy and Computing* entry below; check that before citing either.
- Ringlein, Parnell, Stoica, "GPU Performance Portability needs Autotuning", arXiv:2505.03780 (v3 Jul 2025). Kernel-parameter autotuning for LLM kernels; weak for #2.
- Elvinger et al., "Understanding GPU Resource Interference One Level Deeper", arXiv:2501.16909 (v3 Feb 2026). Kernels saturate some units and leave others idle; not R3.
- Simonetti et al., "Mercury-Opal", arXiv:2601.19654 (A&A accepted). OpenACC GPU planet-formation N-body; no kernel-design content.
- Grimm & Stadel, "Terrestrial planet formation in the era of GPU computing", arXiv:2604.09184 (Springer book chapter, 2026). A GENGA review with no kernel-design content in the abstract.
- David--Cléris, Laibe, Lapeyre, "The Shamrock code: I- SPH on GPUs", arXiv:2503.09713 (MNRAS accepted). GPU tree with negligible build time. The abstract doesn't mention Morton; a search snippet says Morton codes are in an appendix, which was not checked.
- Yan et al., "Mock Observations for the CSST Mission: Multi-Channel Imager – Instrument Simulation", arXiv:2511.12481 (RAA accepted, 43 authors). Instrument simulation only, **not N-body** (resolves the CSST/PhotoNs-3.7 lead).
- Lacopo et al., *Astronomy and Computing* 55, 101060, 2026, DOI 10.1016/j.ascom.2026.101060 (PINOCCHIO on GPU; not a force solver).
- Meneses et al., "Ray Tracing Cores for General-Purpose Computing: A Literature Review", arXiv:2603.28771 (2026). Useful if the RT-core route is taken.

## Unverified leads ⬜ (seen second-hand only; do NOT cite)
| Lead | Where seen | Why it might matter |
|---|---|---|
| Turimbetov, Wahib, Unat, "A Device-Side Execution Model for Multi-GPU Task Graphs", ICS '25 | search result | device-side scheduling (not chased in round 3) |
| PhotoNs-3.7 (newest PhotoNs, "heterogeneous acceleration") | web-search summary of the CSST MCI paper only; the abstract does not mention it | if a PhotoNs-3.x code paper exists, it is the one Chinese GPU-capable gravity lead left |
| HyTiS / Voltrix full texts | ACM DL / USENIX PDF not read | would allow a verbatim HyTiS quote |
| Huang, Sadiq, Yokota, Hess, "Tensor Core Accelerated FMM for GROMACS", SC '25 poster | search snippet | Tensor-core M2L; Japan/Sweden |
| Lengvenis, Dachsel, Morgenstern, Kabadshow, "A New Spin on the FMM for GPUs", IPDPS 2025 pp. 789–800 | citation in a search snippet | 2025 GPU FMM operators (Germany) |
| HybriMoE at DAC '25; Thaler & Keller at IPDPS 2026 | arXiv comments only | confirm in the proceedings before citing the venue |
| dblp / OpenAlex venue sweeps (TPDS, JPDC, SC, PPoPP, IPDPS, JCST, SCIS) for Chinese GPU N-body 2025–26 | dblp blocked by Anubis; OpenAlex narrowed queries hit the tool session limit | needed to harden the "none found" result |
| Hierarchical rigid-body simulation on GPU, MDPI *Computation* 13(11) 250, 2025 | search result | block/warp grouping for balance |
| gevolution 2.0 (arXiv:2607.17929); StePS cylindrical (arXiv:2602.20787); Mercury-Opal (arXiv:2601.19654); GENGA review (arXiv:2604.09184); multi-GPU miniapps (arXiv:2511.02655) | arXiv API listing only | 2025–26 GPU N-body codes; abstracts not checked |
| Eberius et al., hierarchical load-imbalance metric for GPU+MPI (OSTI 1863305) | search result | LB metric definition |
| RT-BarnesHut "2x–40x" speedup figure | search snippet only | **do not quote**; the abstract gives no number |
| Nsight Compute 2026.3 documentation | not fetched | required by the role file; next pass. Needed to confirm the cc 8.9 register allocation granularity |
| PCI-SIG FAQ / PCIe 4.0 specification (primary for 128b/130b) | pcisig.com returned a bot-check page | would replace solomon2014pcie16gts as the encoding source; fetch by hand |
| AD103 TPCs per GPC | not in any NVIDIA source found (whitepaper gives only totals) | non-NVIDIA sources were not searched; do not cite anything unverified |
