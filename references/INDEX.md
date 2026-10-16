# references/INDEX.md: evidence base

Maintained by `literature-scout`. First pass: **2026-10-17**.
Rule: ✅ = landing page fetched on 2026-10-17, with title, authors, year, venue and DOI/arXiv ID confirmed. ⬜ = lead, **do not cite**.
Country = country of the lead group's institution, not the authors' nationality.

## Verified entries (20: 13 recent papers + 3 background + 4 NVIDIA docs)

### A. GPU N-body / hierarchical methods (2025–2026)
| Key | Year | Venue | Topic | Country | R3 | ✓ |
|---|---|---|---|---|---|---|
| [madan2025stochastic](madan2025stochastic.md) | 2025 | SIGGRAPH '25 (ACM) | Stochastic Barnes-Hut on GPU | Canada/USA | bg for #4 | ✅ |
| [nagarajan2025rtbarneshut](nagarajan2025rtbarneshut.md) | 2025 | PPoPP '25 (ACM) | Barnes-Hut on RT cores (OptiX) | USA | motivation #4–#7; optional extra | ✅ |
| [stucker2026jzfmm](stucker2026jzfmm.md) | 2026 | arXiv preprint | GPU-native FMM, Morton sort, non-recursive dual-tree walk | Austria | #5, #7 | ✅ |
| [frontiere2025crkhacc](frontiere2025crkhacc.md) | 2025 | arXiv (venue unconfirmed) | CRK-HACC, GPU-resident tree solvers, 4T particles | USA | context #8 | ✅ |
| [yu2026cube2](yu2026cube2.md) | 2026 | Sci. China PMA | CUBE2 multi-level PM+PP (CPU); cost-sorted LB | **China** | #8 (CPU side), analogy #6 | ✅ |
| [liu2025swphotons](liu2025swphotons.md) | 2025* | Engineering Reports | PM-FMM on Sunway; dynamic LB, pipelined traversal | **China** | analogy #6 | ✅ *online 2023 |

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

### C. Background classics (labelled BACKGROUND)
| Key | Year | Venue | Topic | Country | R3 | ✓ |
|---|---|---|---|---|---|---|
| [barnes1986hierarchical](barnes1986hierarchical.md) | 1986 | Nature 324 | Original Barnes-Hut | not verified | #4–#7 | ✅ |
| [nyland2007nbody](nyland2007nbody.md) | 2007 | GPU Gems 3, ch. 31 | Tiled all-pairs CUDA; q threads/body at small N | USA | **#1, #2, #3** | ✅ |
| [burtscher2011barneshut](burtscher2011barneshut.md) | 2011 | GPU Computing Gems Emerald, ch. 6 | All-GPU Barnes-Hut; sort + warp-vote traversal | USA | **#4, #5, #7** | ✅ |

### D. NVIDIA documentation
| Key | Version | Date | Use | ✓ |
|---|---|---|---|---|
| [nvidia2026cudaguide](nvidia2026cudaguide.md) | CUDA Programming Guide v13.4.2 | updated 2026-09-10 | sm_89 limits (§5.1) | ✅ |
| [nvidia2026bestpractices](nvidia2026bestpractices.md) | CUDA C++ Best Practices Guide 13.4 | no date shown | methodology, exec config, control flow | ✅ |
| [nvidia2026adatuning](nvidia2026adatuning.md) | Ada Tuning Guide v13.4 | updated 2026-10-12 | 48 warps / 24 blocks / 100 KB smem per SM | ✅ |
| [nvidia2023adawhitepaper](nvidia2023adawhitepaper.md) | Ada GPU Architecture whitepaper V2.02 | © 2023 | AD103 specs → confirms 01-hardware [S] rows | ✅ |

## R3 coverage map
| R3 # | Technique | Best sources |
|---|---|---|
| 1 | Tiled all-pairs baseline | nyland2007nbody, nvidia2026adatuning |
| 2 | Block/tile/unroll sweep vs occupancy | nyland2007nbody, nvidia2026adatuning, nvidia2026bestpractices |
| 3 | Multiple threads per body | nyland2007nbody; li2025staticbatching (count-based mapping) |
| 4 | Naive Barnes-Hut traversal | burtscher2011barneshut, nagarajan2025rtbarneshut, dale2026warpdivergence |
| 5 | Morton/Hilbert ordering | burtscher2011barneshut, stucker2026jzfmm, galgoczi2026ser, nanda2025amrexlb |
| 6 | Persistent threads + queue | maeda2026gtap, shetty2026gpuqueues, li2025staticbatching (counter-arm), jin2026eventtensor |
| 7 | Warp-level stealing / cooperative traversal | maeda2026gtap, burtscher2011barneshut, stucker2026jzfmm |
| 8 | CPU+GPU split | yu2026cube2, nanda2025amrexlb, frontiere2025crkhacc (argues against shipping work to the CPU) |

**Gap:** no 2025–2026 source on Hilbert vs Morton for GPU *tree traversal*, and none on CPU+GPU splitting for N-body on P/E-core hybrid CPUs.

## Verified but not catalogued (relevance too low for this pass; landing pages fetched)
- Cheng et al., "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs", arXiv:2512.22219 (2025). Persistent megakernel with decentralized SM scheduling (LLM inference).
- Brandt, "On Parallel and Distributed N-Body Simulations", *Mathematics* 14(9) 1403, 2026, DOI 10.3390/math14091403. Costzones / hashed octree, distributed CPU.
- Ragagnin et al., "OpenGadget3 GPU solver tests", *Astronomy and Computing* 57, 101131, 2026, arXiv:2606.17145. GPU short-range gravity, 2–5x speedup.
- Lacopo et al., *Astronomy and Computing* 55, 101060, 2026, DOI 10.1016/j.ascom.2026.101060 (PINOCCHIO on GPU; not a force solver).
- Meneses et al., "Ray Tracing Cores for General-Purpose Computing: A Literature Review", arXiv:2603.28771 (2026). Useful if the RT-core route is taken.

## Unverified leads ⬜ (seen second-hand only; do NOT cite)
| Lead | Where seen | Why it might matter |
|---|---|---|
| PhotoNs-GPU (PM-FMM GPU code, NAOC; ~2021, arXiv:2107.14008) | search results | Chinese GPU PM-FMM; outside 2025–26 but the best Chinese *GPU* gravity code found |
| PhotoNs-3.7 used in CSST mock observations (arXiv:2511.12481) | search snippet | possible 2025 Chinese GPU/heterogeneous N-body use |
| "Thievory: Graph Processing with Multi-GPU Memory Stealing", ICPP '25, DOI 10.1145/3754598.3754653 | search result | multi-GPU stealing (memory, not compute) |
| Turimbetov, Wahib, Unat, "A Device-Side Execution Model for Multi-GPU Task Graphs", ICS '25 | search result | device-side scheduling |
| Tascell work stealing on GPUs, J. Information Processing 33, 2025 | search result | GPU work stealing (Japan) |
| "Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel", arXiv:2609.21483 | search result | dynamic SM scheduling |
| Libra (CUDA + Tensor cores sparse MM, load balancing), arXiv:2506.22714 | search result | load-balanced sparse kernels; authorship unknown |
| Voltrix, USENIX ATC '25 (Weifeng Liu among authors per snippet) | search result | Chinese SpMM load balancing; not N-body |
| Hierarchical rigid-body simulation on GPU, MDPI *Computation* 13(11) 250, 2025 | search result | block/warp grouping for balance |
| gevolution 2.0 (arXiv:2607.17929); StePS cylindrical (arXiv:2602.20787); Mercury-Opal (arXiv:2601.19654); GENGA review (arXiv:2604.09184); multi-GPU miniapps (arXiv:2511.02655) | arXiv API listing only | 2025–26 GPU N-body codes; abstracts not checked |
| Eberius et al., hierarchical load-imbalance metric for GPU+MPI (OSTI 1863305) | search result | LB metric definition |
| RT-BarnesHut "2x–40x" speedup figure | search snippet only | **do not quote**; the abstract gives no number |
| NVIDIA RTX 4080 SUPER product spec page (2295/2550 MHz, 23 Gbps) | not fetched | needed to source 01-hardware clock rows |
| Nsight Compute 2026.3 documentation | not fetched | required by the role file; next pass |
