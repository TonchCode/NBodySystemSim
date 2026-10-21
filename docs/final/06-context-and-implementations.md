# 6. Context: Where N-Body and Barnes-Hut Are Used, Why, and Other Implementations

> **Language note:** written in English; the Bulgarian/English choice for the final document is still open.

**Source tags** as in chapters 3 and 5. Every paper below is in `references/references.bib`. Every
repository was checked on GitHub on 2026-10-22: it exists, and its description, licence and last
activity were read. **Repository descriptions are those projects' own words, not our tests.**
None of these codes was built or benchmarked here.

---

## 6.1 Why the algorithm is useful

Any problem where **every element pulls or pushes every other element** costs O(N²) when done
directly. The same idea, "summarise far-away groups by one number", reduces it to about
O(N log N) (chapter 3, §3.4). Gravity is the textbook case, but the same structure appears
wherever a force falls off with distance:

| Field | What the "bodies" are | Why a tree / fast method is needed |
|---|---|---|
| **Cosmology** | dark-matter particles, 10⁹–10¹² of them | direct summation is impossible at that size |
| **Galaxy and star-cluster dynamics** | stars | collisions and mergers of galaxies (our `galaxies` initial condition) |
| **Planetary systems** | planets, planetesimals, test particles | long, accurate integrations; planet formation |
| **Data visualisation (t-SNE)** | data points in a 2-D embedding | repulsion between all points, N up to millions |
| **Graph drawing** | nodes of a network | every node repels every other node |
| **Molecular dynamics / plasma** | atoms, charges | electrostatics has the same 1/r² form (usually solved with FMM or PME, the relatives of Barnes-Hut) |

**Why on a GPU:** each body's force is independent of every other body's force in the same step.
That is millions of identical, independent jobs, which is exactly what a GPU's 10,240 FP32 lanes
are built for (chapter 2).

---

## 6.2 Where it is used: real codes and papers

### 6.2.1 Astrophysics and cosmology (production codes)

| Code | What it is | Method | Source |
|---|---|---|---|
| **GADGET-4** | the current version of one of the most widely used cosmology codes | tree, plus a fast multipole method (FMM) option; gravity + hydrodynamics; MPI on CPUs | [P:springel2021gadget4], MNRAS 506, 2021 |
| **PKDGRAV3** | the code behind 2-trillion-particle simulations calibrating the Euclid satellite | FMM on GPU-accelerated nodes; ran on Piz Daint with 4,000+ GPU nodes | [P:potter2017pkdgrav3] |
| **Bonsai** | a Barnes-Hut tree code that runs **entirely on the GPU**, build and walk | octree on the GPU, θ opening angle, CUDA | [P:bedorf2012bonsai]; github.com/treecode/Bonsai (Apache-2.0) |
| **ChaNGa** | a parallel tree code built on Charm++ (UIUC Parallel Programming Laboratory version) | tree gravity, multi-node | github.com/N-BodyShop/changa (GPL-2.0, active 2026); method details not checked here |
| **HACC / CRK-HACC** | an exascale cosmology code | GPU-resident solvers | [P:frontiere2025crkhacc] (chapter 3, §3.7.2) |
| **CUBE (2026)**, **Jiutian**, **PHOTONS** | Chinese cosmology and simulation programmes | particle-mesh and tree methods | [P:yu2026cube2], [P:han2025jiutian], [P:liu2025swphotons] (chapter 3, §3.7.2) |

### 6.2.2 Planetary dynamics

| Code | What it is | Source |
|---|---|---|
| **GENGA** | GPU N-body integrator for planet formation; up to 30× faster than the CPU code Mercury | [P:grimm2014genga], CUDA |
| **REBOUND** | open-source multi-purpose N-body code (CPU, C with Python bindings), very widely used | [P:rein2012rebound]; github.com/hannorein/rebound (GPL-3.0, active 2026) |

### 6.2.3 Outside physics: the same Barnes-Hut idea

| Use | What Barnes-Hut does there | Source |
|---|---|---|
| **t-SNE** (machine-learning visualisation) | approximates the repulsive forces between all embedded points in O(N log N), which makes millions of points feasible | [P:vandermaaten2014tsne], JMLR 15, 2014 |
| **ForceAtlas2** (Gephi's default graph layout) | Barnes-Hut approximation of node–node repulsion | [P:jacomy2014forceatlas2], PLoS ONE 2014 |
| **d3-force** (JavaScript graph layout in web pages) | `forceManyBody` builds a quadtree and uses a Barnes-Hut test with **θ = 0.9** by default (`theta2 = 0.81` in `src/manyBody.js`) | github.com/d3/d3-force (ISC) |
| **GPUGraphLayout** | experimental GPU implementation of ForceAtlas2 | github.com/govertb/GPUGraphLayout |
| **tsne-cuda** | GPU t-SNE (up to 1,200× faster than scikit-learn, by its own README). Note: the current version follows **FIt-SNE (FFT interpolation)**, not Barnes-Hut. The field moved on, which is itself a useful fact | github.com/CannyLab/tsne-cuda (BSD-3), arXiv:1807.11824 |

A web page that lays out a graph with d3 runs the same algorithm as our simulation, in 2-D, with
a looser θ (0.9 against our 0.5).

---

## 6.3 Other GPU implementations to compare ours with

| Implementation | Algorithm | How it differs from ours |
|---|---|---|
| **NVIDIA CUDA Samples `nbody`** (github.com/NVIDIA/cuda-samples, `cpp/5_Domain_Specific/nbody`) | all-pairs, shared-memory tiling; the code that accompanies GPU Gems 3 ch. 31 | the same tiling as our #1 [P:nyland2007nbody]. Ours adds block/unroll tuning (#2) and q threads per body (#3), measured at 2.40–2.56× at small N |
| **ECL-BH** (Burtscher, Texas State; userweb.cs.txstate.edu/~burtscher/research/ECL-BH/, v4.5) | Barnes-Hut entirely on the GPU: octree built with atomics, warp-vote traversal [P:burtscher2011barneshut] | the original warp-vote idea that our #7 builds on [P:burtscher2011barneshut]. Ours builds a **binary radix tree** (Karras) from sorted Morton keys instead of an octree with locks |
| **Bonsai** | GPU octree, build and walk both on the GPU [P:bedorf2012bonsai] | ours uses a binary radix tree and a warp-union walk; both keep the whole step on the GPU |
| **Stochastic Barnes-Hut (2025)** | randomised GPU Barnes-Hut | [P:madan2025stochastic] (chapter 3, §3.7.2) |
| **RT-core Barnes-Hut (2025)** | walks the tree with ray-tracing hardware | [P:nagarajan2025rtbarneshut]. Our chapter 2, §2.2.6 explains why we do not use RT cores |

**Forums.** The NVIDIA Developer Forums thread "a deep dive into instruction-level parallelism"
(forums.developer.nvidia.com/t/68567) discusses the ILP-versus-occupancy question of chapter 5. A
moderator there notes that Volkov's 2010 numbers come from much older hardware. That is why
chapter 5 relies on **our own** prefetch measurement rather than his figures.

---

## 6.4 Publications on the new topics (ILP, TLP, memory, profiling)

| Key | Year | What it says | Why it is here |
|---|---|---|---|
| [P:volkov2010lower] | 2010 | "Better Performance at Lower Occupancy" (GTC 2010, UC Berkeley): ILP can replace TLP, and faster codes often run at lower occupancy | the classic source on ILP vs TLP (BACKGROUND) |
| [P:hepkema2026instructionwall] | 2026 | "Over the Memory Wall, Into the Instruction Wall": on an **L4 (Ada)** and a GH200, once bandwidth stops being the limit, kernels become instruction-bound; fixes: occupancy and/or ILP, fewer instructions per access | matches our finding that the all-pairs loop is issue-limited (chapter 2, §2.2.3) |
| [P:luo2025hopper] | 2025 | microbenchmarks of the Hopper memory hierarchy, **compared against Ampere and Ada Lovelace**: L2 partitions, global-memory access, shared memory | **Chinese-led** (HKUST Guangzhou and HIT Shenzhen); it measures an **RTX 4090 (Ada, cc 8.9)**: L1 32.0, shared 30.1, L2 273, global 571 cycles (Table 3). Used in chapter 5, §5.2 |
| [P:jarmusch2025blackwell] | 2025 | microbenchmarks of Blackwell (RTX 5080) against Hopper: shared memory, L1/L2, pipelines | the generation after ours: what changes in the memory hierarchy |
| [N:nvidia2026bestpractices], [N:nvidia2026cudaguide], [N:nvidia2026adatuning] | 2026 | NVIDIA's own guides: memory spaces, coalescing, occupancy, Ada tuning | the R5 requirement (NVIDIA documentation) |

**R4 note (2025–26, Chinese):** for the memory and ILP topics, the Chinese-led 2025 source is
[P:luo2025hopper]. For load balancing, chapter 4 already cites Chinese-led 2025–26 papers
(e.g. [P:zhang2025hytis], [P:shi2025libra], [P:zhong2025hybrimoe]). For the algorithm itself,
chapter 3 cites Chinese-led cosmology codes ([P:yu2026cube2], [P:han2025jiutian]).
