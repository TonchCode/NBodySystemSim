# R3 evidence map: load-balancing techniques → verified sources

Maintained by `literature-scout`. Round 3, **2026-10-17**, written in answer to critic review 1, finding #8.
Every key below is in `references.bib` and has a `<key>.md` file with its landing-page verification, except where marked *(not in bib)*.
Quotes are under 15 words and were copied from the fetched source.

**Strength scale** (how far the paper's evidence carries to N-body on one RTX 4080 SUPER, sm_89):
- **direct**: N-body or tree/FMM gravity on an NVIDIA GPU, or a GPU-scheduling result that doesn't depend on the kernel body.
- **transferable**: a different workload (graph, sparse, GEMM) using the same mechanism, so the result should carry over; we say how.
- **analogy**: a similar idea in a very different setting. It motivates a design, but doesn't predict a number.

**Background** (pre-2025) is labelled as such and never stands alone as the R4 source.

---

## Summary table

| R3 # | Technique | Best 2025–26 source | Strength | Chinese 2025–26? | Background |
|---|---|---|---|---|---|
| 1 | Tiled all-pairs baseline | **none found** (see §1) | — | no | nyland2007nbody; wang2021photonsgpu (FP32) |
| 2 | Block/tile/unroll sweep vs occupancy | zhang2026wavetune | direct (wave model) | **yes** (SJTU) | nyland2007nbody, NVIDIA docs |
| 3 | q threads per body (split + reduce) | zhang2025hytis (SC '25) | transferable | **yes** (Wuhan U) | nyland2007nbody |
| 4 | Naive Barnes-Hut, one thread/body | dale2026warpdivergence; nagarajan2025rtbarneshut | direct | no | burtscher2011barneshut, barnes1986hierarchical |
| 5 | Morton/Hilbert ordering | stucker2026jzfmm; thaler2026sfcneighbor | direct / transferable | no | burtscher2011barneshut |
| 6 | Persistent threads + atomic queue | maeda2026gtap; huang2026weave; chen2026gmatch | direct / transferable | **yes** (SJTU ×2, Wuhan U) | — |
| 7 | Warp-level stealing / cooperative traversal | maeda2026gtap; chen2026gmatch | direct / transferable | **yes** (SJTU) | burtscher2011barneshut |
| 8 | CPU+GPU split | Ragagnin et al. 2026 *(not in bib)*; xu2025tascell; zhong2025hybrimoe | direct / transferable / analogy | yes, analogy only (PKU) | — |

**Techniques without a 2025–26 source:** #1 has none. #8 has none for the P-core/E-core angle specifically.
**Techniques without a 2025–26 Chinese N-body source:** all eight. The Chinese 2025–26 support comes entirely from transferable GPU-scheduling papers. See §9.

---

## 1. Tiled all-pairs, shared-memory tiling (baseline)

| Key | Claim it supports | Strength |
|---|---|---|
| nyland2007nbody *(background, 2007)* | Tiled all-pairs, one thread per body, two barriers per tile | direct |
| wang2021photonsgpu *(background, 2021, China)* | FP32 ≈ 2x FP64 in a GPU gravity code, with small unbiased power-spectrum noise | direct (precision) |
| nvidia2026bestpractices, nvidia2026adatuning *(docs, 2026)* | Throughput and occupancy limits for cc 8.9 | direct |

**Flag: no 2025–26 paper.** Searches for a 2025–26 tiled direct-summation N-body kernel paper found codes that *use* GPU direct summation without publishing the kernel design. Mercury-Opal (arXiv:2601.19654, OpenACC), GENGA (Grimm & Stadel, arXiv:2604.09184, review chapter) and NBODY6++GPU (wu2025dragon3) are in this group, so none of them supports a design claim. The honest framing for the professor: the tiled all-pairs kernel is a settled 2007 technique, and the 2025–26 evidence for #1 is the 2026 NVIDIA documentation for the specific device, not a paper.

## 2. Block / tile / unroll sweep vs occupancy

| Key | Claim | Quote | Strength |
|---|---|---|---|
| **zhang2026wavetune** (SJTU + Alibaba, 2026) | The ceil(blocks/SMs) step model is an idealization. Tail penalty ≤ one wave, smoothed by block-latency variance. | "neither the rigid step function nor the simplified linear view adequately captures" | **direct** for the wave model; GEMM-like kernels, no Ada |
| **zhang2025hytis** (Wuhan U, SC '25) | Wave quantization is a first-order loss; a finer-grained tail wave fixes it | (no verbatim quote; full text 403) | transferable (GEMM tiles ≈ all-pairs tiles: regular, equal cost) |
| **shi2025libra** (BUPT, 2025) | Occupancy O = B_launch / (N_SM × B_max^SM) drives the launch decision; tested on RTX 4090 | "Libra further incorporates hybrid load balancing, occupancy-aware task scheduling" | transferable (sparse; same GPU family) |
| nvidia2026adatuning, nvidia2026cudaguide | 48 warps / 24 blocks / 100 KB per SM on sm_89 | (see entries) | direct |
| nyland2007nbody *(background)* | Block size as the tile size | | direct |

## 3. Multiple threads per body (j-split + reduction)

| Key | Claim | Quote | Strength |
|---|---|---|---|
| **zhang2025hytis** | Splitting work in the partial wave into finer pieces recovers idle SMs; that is #3's mechanism at small N | — | transferable |
| **shi2025libra** | Split only the heavy items (threshold), combine the pieces with atomics; split cost traded against balance | "triggers decomposition only when the workload exceeds these thresholds" (HTML) | transferable. We reduce with shuffles, not atomics |
| li2025staticbatching (Alibaba, 2025) | Count-based static mapping of work items to blocks | "We propose a general framework for statically batching irregular workloads on GPUs." | transferable (MoE, not N-body) |
| nyland2007nbody *(background)* | q threads per body at small N | | direct |

## 4. Naive Barnes-Hut, one thread per body (imbalance baseline)

| Key | Claim | Strength |
|---|---|---|
| dale2026warpdivergence | Divergence cost ≈ 32/k, Pascal→Blackwell (no Ada); [T2] one author | direct (divergence model) |
| nagarajan2025rtbarneshut (PPoPP '25) | Barnes-Hut traversal on GPU is irregular enough to justify RT-core offload | direct (motivation) |
| madan2025stochastic (SIGGRAPH '25) | GPU Barnes-Hut traversal with stackless BVH + warp voting | direct |
| burtscher2011barneshut, barnes1986hierarchical *(background)* | The algorithm and the classic GPU implementation | direct |

No Chinese 2025–26 source. The searches in §9 found no Chinese GPU Barnes-Hut paper from 2025–26.

## 5. Space-filling-curve sorting (Morton / Hilbert)

| Key | Claim | Quote | Strength |
|---|---|---|---|
| stucker2026jzfmm | GPU-native FMM: Morton sort, bottom-up tree, non-recursive traversal | (see entry) | **direct** |
| **thaler2026sfcneighbor** (CSCS, IPDPS 2026) | Hilbert layout makes consecutive particles spatially compact on GPUs; clusters = runs | "the average compactness of the Hilbert curve" | transferable (neighbor search, astro SPH) |
| galgoczi2026ser | Reordering divergent work on an RTX 4090 (Ada) | (see entry) | transferable |
| nanda2025amrexlb | SFC partitioning for load balance | (see entry) | analogy (distributed AMR) |

**Gap unchanged:** no 2025–26 source compares Hilbert and Morton for *traversal divergence* on a GPU. JZ-FMM uses Morton and Thaler & Keller use Hilbert, and neither measures the other. Our measurement would be original. No Chinese source.

## 6. Persistent threads + global atomic work queue

| Key | Claim | Quote | Strength |
|---|---|---|---|
| maeda2026gtap (Tokyo) | Persistent kernel; FMM dual-tree traversal on H100; stealing beats a central queue | "Work stealing outperforms a centralized global queue, …" | **direct** (FMM traversal) |
| **huang2026weave** (SJTU, 2026) | Persistent megakernel; each SM claims tiles from a global atomic counter; dynamic over fixed | "each SM claims its next tile via atomicAdd on a global counter" | transferable (identical claiming mechanism) |
| **chen2026gmatch** (SJTU, 2026, RTX 4090) | An atomic-counter pool of ~10⁶ fine tasks does most of the balancing | "idle warps receive half of the work from busy warps" | transferable (irregular DFS on Ada) |
| **xia2025voltrix** (Wuhan U, ATC '25) | Persistent kernel + precomputed two-stage balanced partition | "a persistent and I/O co-balanced kernel mechanism" | transferable |
| li2025staticbatching (Alibaba) | Counter-arm: static prefix-sum mapping, no runtime scheduler | (see entry) | transferable |
| shetty2026gpuqueues, jin2026eventtensor | GPU queue designs; static vs dynamic in persistent kernels | (see entries) | transferable |

## 7. Warp-level work stealing / cooperative traversal

| Key | Claim | Strength |
|---|---|---|
| maeda2026gtap | Per-warp and per-block deques with random stealing in a persistent kernel; ≈1.5x traversal phase, ≈1.05x end to end | **direct** |
| **chen2026gmatch** (SJTU, RTX 4090) | Intra-block warp stealing through a shared-memory queue (half-split), then inter-block through global memory; stealing is "supplementary" | transferable |
| burtscher2011barneshut *(background)* | Warp-vote cooperative traversal | direct |
| stucker2026jzfmm | Non-recursive dual-tree walk on GPU | direct |

Not catalogued: CoopRT (Tozlu & Zhou, ISCA '25, DOI 10.1145/3695053.3731118, NC State). Idle lanes help busy lanes in BVH traversal, but it is a *hardware* proposal on a simulator. Only the metadata was verified (Crossref).

## 8. Heterogeneous CPU+GPU split

| Key | Claim | Quote | Strength |
|---|---|---|---|
| Ragagnin et al. 2026, "OpenGadget3 GPU solver tests", *Astronomy and Computing* 57, 101131, arXiv:2606.17145 *(not in bib: full author list not yet captured)* | Production gravity code: tree rebuilt on the **CPU**, short-range Barnes-Hut walk on the **GPU**, overlapped asynchronously | "we offload the tree at each timestep." | **direct** (phase split) |
| **xu2025tascell** (Kyoto, JIP 2025) | Dynamic work stealing between CPU and GPU workers; +12.9% over GPU-only on a workstation | "the hybrid approach outperformed both CPU-only and GPU-only configurations" | transferable |
| nozal2025coexecution (J. Supercomputing 2025) | Same kernel co-executed on CPU+GPU: 0.91 efficiency, 1.31x over GPU-only | "average efficiencies of 0.91 and speedups of 1.31 over using only the GPU" | transferable [T2] |
| **zhong2025hybrimoe** (PKU, DAC '25 per arXiv) | Warmup profiling, then a split that minimizes max(T_CPU, T_GPU) | "a dynamic intra-layer scheduling strategy to balance workloads across CPU and GPU" | analogy |
| yu2026cube2 (China) | Cost-sorted load balance (CPU side) | (see entry) | analogy |
| frontiere2025crkhacc | Keeps solvers GPU-resident (argues against shipping work to the CPU) | (see entry) | direct (counter-evidence) |

**Flag:** still no 2025–26 source on P-core/E-core hybrid CPUs in a CPU+GPU split.

---

## 9. The "preferably Chinese" search, documented

**Result:** I found **no 2025–26 Chinese-led paper on GPU N-body, tree, FMM or P3M methods**. Chinese 2025–26 GPU *load-balancing* papers do exist at top venues (SC '25, USENIX ATC '25) and as recent SJTU/BUPT preprints, two of them evaluated on an RTX 4090. They support R3 #2, #3, #6 and #7 as transferable evidence.

| Lead / query | Where searched (2026-10-17) | Outcome |
|---|---|---|
| PhotoNs-GPU | arXiv abs 2107.14008 | ✅ verified: **2021** (RAA 21, 281). Background only. No 2025–26 PhotoNs GPU paper found |
| PhotoNs 2025 successors | web search "PhotoNs GPU cosmological N-body simulation 2025" | Only swPHoToNs (Sunway, already in INDEX) and the CSST MCI paper |
| CSST mock observations, arXiv:2511.12481 | arXiv abs | ✅ fetched. Instrument simulation (43 authors, RAA accepted). Abstract has no GPU, no PhotoNs, no N-body code. **Resolved: irrelevant** |
| CSST cosmological simulations (Jiutian) | arXiv 2503.21368 abs + HTML | ✅ han2025jiutian. Run with LGadget-3, Gadget-4, Cube: **CPU codes**; no GPU named |
| arXiv API: `abs:GPU AND abs:"N-body" AND (astro-ph.IM OR cs.DC)`, newest first | export.arxiv.org | 2025–26 hits: gevolution 2.0, OpenGadget3, GENGA, Mercury-Opal, Lacopo, Nyx-RT, Odisseo, multi-GPU miniapps, DISCO-DJ II, etc. **None with a Chinese lead group** in the visible part (40 of 78; the feed gives no affiliations, so judged from author names and known groups) |
| arXiv API: `abs:GPU AND (gravity OR gravitational) AND (tree OR multipole OR particle-mesh)` | export.arxiv.org | 9 hits from 2025+: FUGUE (Poland), OpenGadget3, JZ-FMM, Fast(er)PM, QTAM, Thaler & Keller (CSCS), pkdgrav3 SPH, CRK-HACC, QUOKKA (ANU). **No Chinese lead** |
| "N-body GPU CUDA 2025 Chinese Academy of Sciences" | web search | Only NBODY6++GPU (2015) and older |
| "Barnes-Hut GPU 2025/2026 … Tsinghua / Sun Yat-sen / NUDT / USTC" | web search | Only madan2025 and nagarajan2025 (non-Chinese) |
| "fast multipole method GPU tensor core 2025" | web search | SC '25 poster (Huang, Sadiq, Yokota, Hess: GROMACS FMM; snippet only, Japan/Sweden); IPDPS 2025 FMM paper (Lengvenis et al., Germany, snippet only) |
| Chinese-language query "N体模拟 GPU 加速 2025 树算法 快速多极子" | web search | Only JZ-FMM (Austria) and blogs |
| "DCU heterogeneous N-body … Hygon / new Sunway / Tianhe" | web search | Nothing from 2025–26. The only 2025 hit elsewhere: wu2025dragon3 (Heidelberg lead, NAOC/PKU/SYSU co-authors) **ports NBODY6++GPU to HIP for Chinese DCU clusters**. No method or performance content |
| PeTar / Long Wang (SYSU) 2025 | web search + arXiv 2510.03933 | PeTar 2020 (Tokyo-era); 2025 use only (DRAGON-III) |
| OpenAlex: `search=N-body GPU`, CN institutions, ≥2025 | api.openalex.org | 10,621 loose hits. The first 12 were unrelated (FHE, Eikonal, metadynamics, NBODY6++GPU applications). Narrower title/abstract filters were cut off by the tool's session limit (see "Not done") |
| dblp (venue-level: TPDS, JPDC, SC, PPoPP, IPDPS for "n-body gpu" 2025/2026) | dblp.org API | **Blocked** by a bot-protection page (Anubis), so not completed |
| ACM DL landing pages (SC '25, ICPP '25) | dl.acm.org | 403. Crossref and Semantic Scholar used instead |
| Chinese GPU load balancing, graph/sparse | web search + arXiv/USENIX/Crossref | ✅ shi2025libra (BUPT), xia2025voltrix (Wuhan, ATC '25), zhang2025hytis (Wuhan, SC '25), zhang2026wavetune (SJTU), huang2026weave (SJTU), chen2026gmatch (SJTU); plus li2025staticbatching (Alibaba) and zhong2025hybrimoe (PKU) |

**Not covered** (so the claim stays "not found", not "does not exist"): CNKI (not reachable with these tools); the JCST and *Science China Information Sciences* tables of contents; the TPDS/JPDC 2025–26 tables of contents (dblp was blocked); the OpenAlex narrowed queries. These are the next pass if the professor wants the null result hardened.

**How to say it to the professor:** Chinese groups lead in GPU *scheduling and load balancing* for irregular workloads (SJTU, Wuhan University, BUPT, PKU), and those papers back our techniques. For *GPU N-body itself* in 2025–26, the Chinese work found is either CPU or Sunway (swPHoToNs, CUBE2, the Jiutian/CSST suite on Gadget and Cube) or porting existing codes to DCU (DRAGON-III). The leading 2025–26 GPU N-body method papers come from Austria (JZ-FMM), Japan (GTaP), the USA (RT-BarnesHut, CRK-HACC) and Switzerland (CSCS).

## 10. Findings that bear on `docs/03-design.md` (read-only; for the architect)

1. **M1's step model is an idealization (zhang2026wavetune).** §3.3 and §3.4 use time ∝ ceil(B/80). WaveTune shows the tail penalty is bounded by one wave and smoothed by block-latency variance. That fits our miss on P2a (+8% measured vs +20% predicted). Cite it when revising M1, and fit a smoothed model rather than defending the step.
2. **For #6 vs #7, expect stealing to add little on top of a fine-grained pool** (chen2026gmatch: stealing is "supplementary"; maeda2026gtap: ≈1.05x end to end). Pre-register #7's gain over #6 as small, and report the phase and the full step separately.
3. **Three arms for #6, not two.** Static count-based mapping (li2025staticbatching), persistent + precomputed balanced partition (xia2025voltrix), and persistent + atomic counter (huang2026weave). The design should name which one it implements.
4. **#8 evidence favours a phase split over a body split.** The one direct 2026 gravity source (OpenGadget3) builds the tree on the CPU and walks it on the GPU, overlapped. The workstation-scale gains for dynamic splits are modest (+13%: xu2025tascell; 1.31x: nozal2025coexecution). frontiere2025crkhacc argues for keeping everything GPU-resident. Set expectations low for #8.
5. **#3's reduction choice holds up.** Libra combines split work with atomicAdd and trades balance against atomic cost. Our shuffle reduction avoids that cost, which is worth stating as a design advantage.
6. No finding contradicts FP32 (wang2021photonsgpu supports it, with a measured accuracy cost).
