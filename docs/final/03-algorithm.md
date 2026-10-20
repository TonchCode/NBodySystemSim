# 3. The Algorithm: The Gravitational N-Body Problem, Direct Summation and Barnes-Hut

> **Language note:** written in English; the Bulgarian/English choice for the final document is still open.
> §3.9 has an EN–BG glossary of the terms used here.

**Source tags** (project convention, `CLAUDE.md` rule 5): **[P:key]** = a paper in
`references/references.bib`, verified on a landing page or Crossref record (date in
`references/INDEX.md`) · **[M]** = measured on this machine (the results folder is named) ·
**[D]** = derived arithmetically from tagged values · **[std]** = standard textbook physics or
mathematics, stated without a citation.

This chapter explains **what** the algorithms are and **how they work, step by step**. It is
the chapter to present from. Chapter 2 covers the hardware and chapter 4 the load-balancing
measurements. §3.8 is a slide outline and §3.10 has the questions a lecturer is likely to ask.

---

## 3.1 The problem

We have N bodies (stars, or particles of a galaxy), each with a mass m_i, a position x_i and a
velocity v_i. Each body is pulled by **every other body** according to Newton's law of
gravitation. The task is to compute how the system evolves in time.

The acceleration of body i is [std]:

```
a_i = G · Σ_{j≠i} m_j · (x_j − x_i) / ( |x_j − x_i|² + ε² )^{3/2}
```

- **G** is the gravitational constant. The simulation uses units where G = 1 (N-body units) [std].
- **ε (softening)** is a small length (0.01 here, `docs/03-design.md` §11.1). Without it, two
  bodies that pass very close would get an almost infinite acceleration and the time step would
  explode. With it, every body behaves like a small "fuzzy" sphere at short range. ε also lets
  us drop the `j ≠ i` test: for j = i the numerator is 0, so the term contributes nothing.
- One evaluation of the sum for all i is called **one force evaluation**. A simulation is
  thousands of force evaluations, one per time step.

**Facts about the problem**
- N = 2 has an exact solution: Kepler's ellipses [std]. For N ≥ 3 there is no general closed-form
  solution, and the motion can be chaotic [std]. So the problem is solved **numerically**, by
  stepping through time.
- The cost lies in the force sum. Moving the bodies is O(N); computing the forces is O(N²) if
  done directly. That one fact drives every algorithm in this chapter.
- The test galaxy used most here is the **Plummer sphere**, a classic model of a star cluster
  from 1911 [P:plummer1911distribution]. Other shapes (disk, two colliding galaxies, cube shell
  and others) are listed in `docs/03-design.md` §13 and §19.

---

## 3.2 Algorithm 1: direct summation (all-pairs), O(N²)

### Idea
Compute the formula exactly as written: for every body, add up the pull of every other body.

### Step by step (one force evaluation)
1. For each body i (on the GPU: **one thread per body**):
2. set the accumulator a = (0, 0, 0);
3. for each body j = 0 … N−1:
   1. r = x_j − x_i (3 subtractions);
   2. d² = r·r + ε² (3 multiply-adds);
   3. inv = 1/√d² (one hardware **rsqrt** instruction);
   4. s = m_j · inv³;
   5. a += r · s (3 multiply-adds);
4. store a_i.

That is about **20 floating-point operations per interaction**, the standard count used since
[P:nyland2007nbody] (our exact SASS count is in `docs/03-design.md` §0.1). The inner loop is
`bodyBodyInteraction` in `src/core/interaction.cuh`, five lines of code.

### How it is made fast on a GPU (tiling)
Every thread needs every x_j. Reading them all from global memory would make the kernel
memory-bound. The **tiled** scheme [P:nyland2007nbody]:
1. a block of p threads loads p bodies (one each) into **shared memory**;
2. `__syncthreads()`;
3. every thread runs its p interactions against that tile, from fast on-chip memory;
4. `__syncthreads()`, then the next tile, until all N/p tiles are done.

Each body loaded from memory is reused p times. Positions are stored as `float4` (x, y, z, mass)
so a load is a single 16-byte transaction [P:nyland2007nbody].

### Facts
- **Perfectly balanced:** every thread does exactly N interactions. All of its load-balancing
  problems are at the SM level, from too few blocks or a partial last wave (chapter 4, §4.2–§4.4).
- **Exact** up to FP32 rounding. It is the reference against which the tree is validated.
- **Measured on the RTX 4080 SUPER:** about **1,264 G interactions/s** at N = 65,536 (best
  configuration, `results/2026-10-17_round2/summary.md` §2) [M]. At 20 FLOP per interaction that
  is ≈ 25 TFLOP/s [D], about half of the 52 TFLOPS FP32 peak (chapter 2, §2.5).
- **Doesn't scale:** at N = 1,048,576 one force evaluation is N² ≈ 1.1 × 10¹² interactions. At
  the measured rate that is ≈ 0.87 s per step [D: extrapolated from the 65,536 rate, not
  measured at 1M]. Ten times more bodies costs a hundred times more time.
- **Small N is the hard case on a GPU:** with N = 4,096 there are too few threads to fill 80
  SMs. Splitting each body's sum over q threads (R3 technique #3) gave **2.56×** there (chapter 4,
  §4.4) [M]. The idea dates from [P:nyland2007nbody].

---

## 3.3 Moving the bodies: the leapfrog (kick-drift-kick) integrator

After the forces, positions and velocities are advanced by one time step dt. We use
**leapfrog** in its kick-drift-kick form, a member of the Verlet family of integrators
[P:verlet1967computer]:

1. **kick:** v ← v + a · dt/2
2. **drift:** x ← x + v · dt
3. compute the new accelerations a (the force evaluation from §3.2 or §3.4)
4. **kick:** v ← v + a · dt/2

**Why leapfrog and not the simpler Euler method:** leapfrog is second-order accurate, needs only
**one** force evaluation per step (as Euler does), and is **symplectic**: energy errors oscillate
but don't drift over long runs, whereas Euler steadily gains energy and galaxies "evaporate"
[std]. Code: `kickDriftKernel` in `src/core/nbody.cu`, where kick and drift are fused into one
kernel.

The integrator is O(N) and trivially parallel (one thread per body), so all the interesting work
is in the force evaluation.

---

## 3.4 Algorithm 2: Barnes-Hut, O(N log N)

### The idea in one sentence
**A distant group of bodies pulls almost exactly like a single body of the same total mass
located at the group's centre of mass** [P:barnes1986hierarchical].

Looking at the Andromeda galaxy from Earth, you don't need its billion stars separately: one
point of mass M at its centre is accurate to a tiny fraction. Up close you have to look at
individual stars. Barnes-Hut organises space into a **tree** of boxes so that, for every body,
it can decide quickly which groups are "far enough" to treat as one point.

### The classic algorithm (Barnes & Hut, 1986) [P:barnes1986hierarchical]
1. **Build an octree.** Put all bodies in one cube. Split any cube that holds more than one
   body into 8 equal sub-cubes, and repeat until every leaf holds one body.
2. **Summarise.** For every cube compute its total mass M and centre of mass (COM), from the
   leaves up.
3. **Walk the tree for every body.** Start at the root. For a cube of side s at distance d from
   the body:
   - if **s / d < θ**, the cube is far enough: add **one** interaction with (M, COM) and skip
     everything inside it;
   - otherwise **open** it and look at its 8 children.
4. **Integrate** (§3.3) and rebuild the tree next step, because the bodies have moved.

**θ (theta), the opening angle**, is the accuracy knob. θ = 0 opens everything, giving exact
all-pairs. Larger θ means fewer interactions and more error. θ = 0.5 is a common choice and is
what we report.

### Worked example (2D, for the slide)
A body sits at the origin. Far away, at distance d = 10, is a box of side s = 2 holding 4 bodies
of mass 1 each, so the box has M = 4 and COM at (10, 0).
- s / d = 2 / 10 = 0.2 < θ = 0.5, so the box is accepted.
- The body does **1** interaction (with mass 4 at (10, 0)) instead of **4**.
- Error: with the 4 bodies at (9, ±1) and (11, ±1), the exact pull along x is 0.04057 and the
  point-mass pull is 4/10² = 0.04000, an error of **1.4%** [D: computed exactly, G = 1, no
  softening]. The error shrinks like (s/d)². If the box were at d = 3, then s / d = 0.67 > θ and
  it would be opened.

Applied over a whole tree, each body interacts with a few nearby bodies individually and with
ever-larger boxes further away, about log N "shells". That gives **O(N log N)** total work.

---

## 3.5 How our GPU version works, step by step

The classic algorithm builds the tree by inserting bodies one at a time, which is sequential.
A GPU needs every step to be massively parallel. Our pipeline (`src/core/kernels_tree.cu`,
design in `docs/03-design.md` §10–§12) does the **whole thing on the GPU**, as 2025–26 codes do
[P:stucker2026jzfmm; P:frontiere2025crkhacc: "GPU-resident tree solvers"]. Positions never go to
the CPU.

| # | Step | What it does | Parallelism | Cost |
|---|---|---|---|---|
| 1 | **Bounding box** | min/max of all positions → the enclosing cube | parallel reduction | O(N) |
| 2 | **Morton keys** | turn each 3D position into one 63-bit number | 1 thread per body | O(N) |
| 3 | **Sort** | sort bodies by key (CUB radix sort) | library radix sort | O(N) |
| 4 | **Build tree** | binary radix tree from the sorted keys (Karras) | 1 thread per internal node, no locks | O(N) |
| 5 | **Summarise** | mass, COM, tight box and bmax per node, from the leaves up | 1 thread per leaf, atomic counters | O(N) |
| 6 | **Traverse** | for each body, walk the tree with the opening test | 1 thread (or warp) per body | O(N log N) |
| 7 | **Integrate** | leapfrog kick/drift (§3.3) | 1 thread per body | O(N) |

### Step 2: Morton keys (the space-filling curve)
Divide the cube into a 2²¹ × 2²¹ × 2²¹ grid and take the cell coordinates (x, y, z) as 21-bit
integers. **Interleave their bits**: z₂₀ y₂₀ x₂₀ z₁₉ y₁₉ x₁₉ … z₀ y₀ x₀. The result is a 63-bit
key. Sorting by this key orders the bodies along the **Morton (Z-order) curve**, a path that
visits every cell and keeps nearby points mostly close together [std].

*2D example for the slide:* x = 5 = 101₂ and y = 3 = 011₂ interleave (y, x per bit) to
`01 10 11`₂ = **27**.

Two consequences:
- **Bodies close in the sorted array are close in space.** This is what makes the GPU fast in
  step 6 (§3.6).
- **Every box of the tree is a range of the sorted array.** All bodies whose keys share the
  first k bits lie in the same box, so a box is a pair of indices [first, last].

### Step 4: the parallel radix tree [P:karras2012maximizing]
For sorted keys, define δ(i, j) as the number of leading bits keys i and j share. Karras (2012)
showed that internal node i of the binary radix tree can find **on its own**, with no help from
other nodes:
1. its **direction**: does it extend left or right? It extends towards the neighbour with the
   longer common prefix;
2. its **other end**: search for the farthest key whose common prefix with key i is longer than
   the prefix key i shares with its neighbour on the other side;
3. its **split point**: binary-search for the first bit where the keys of its range differ. The
   two halves are its children.

Every one of the N−1 internal nodes is computed **in parallel, by its own thread, with no
locks**. That is the property that makes it a GPU algorithm. Earlier GPU methods built the tree
level by level [P:karras2012maximizing]. Duplicate keys are made unique by appending the index
(`docs/03-design.md` §11.2).

*Our choice:* a **binary** tree instead of an octree. It is built in a single kernel, but it is
about 3× deeper (31–43 levels at N = 65,536) [M], so walks pop more nodes. That is a
constant-factor trade-off (`docs/03-design.md` §11.2).

### Step 5: summarising bottom-up
Each leaf's thread climbs towards the root. At each parent it does an `atomicAdd` on an arrival
counter:
- the **first** child to arrive stops;
- the **second** knows both children are ready, so it computes the parent's mass, COM, tight
  bounding box and **bmax** (the distance from the COM to the farthest corner of the box), then
  continues upward.

Each node is computed exactly once, with no global synchronisation.

### Step 6: the traversal (the force kernel)
Each body runs a loop with a small **stack** of nodes to visit:

```
push(root)
while stack not empty:
    c = pop()
    if |x_i − COM_c|² ≥ (bmax_c / θ)²:     # far enough: accept
        a += interaction(x_i, COM_c, M_c)    # same 20-FLOP function as all-pairs
    elif c has ≤ 16 bodies:                  # small leaf bucket: just sum them
        for j in c.first .. c.last: a += interaction(x_i, x_j, m_j)
    else:
        push(left child); push(right child)
```

Our **opening test uses bmax instead of the box side s.** Because every body of the box lies
within bmax of the COM, a body can never "accept" a box it is inside (for θ ≤ 1). The classic
s/d test can do this when the COM is near an edge, a known failure case of the original
criterion [P:salmon1994skeletons]. Our θ is therefore about 0.87 × the classic one
(`docs/03-design.md` §11.4).

**Monopole only.** Each box is represented by its mass at its COM (no quadrupole). A box
interaction is then the same `bodyBodyInteraction` as a body interaction, so all-pairs and tree
"interactions" are the same unit of work and can be compared directly (`docs/03-design.md` §11.3).

### Why the sort is essential on the GPU
The 32 threads of a **warp** execute in lock-step. If the 32 bodies of a warp are scattered
around the galaxy, each wants to open different nodes. The warp runs all the different paths one
after another while the other threads idle (divergence) [P:dale2026warpdivergence, T2]. Sorting
along the Morton curve puts the 32 bodies of a warp next to each other in space, so they walk
almost the same path. This was the key optimisation of the first all-GPU Barnes-Hut
[P:burtscher2011barneshut], and in 2026 it is still the standard way to make tree methods
GPU-friendly [P:stucker2026jzfmm; P:thaler2026sfcneighbor].

Our step further (R3 technique #7, `bh_warp`): the warp walks the tree **together**, and opens a
node if **any** of its 32 bodies needs it (`__any_sync`). Every lane is then always busy; the
cost is some extra, harmless interactions (chapter 4, §4.7).

---

## 3.6 Facts and numbers (measured on our RTX 4080 SUPER)

| Fact | Value | Source |
|---|---|---|
| All-pairs complexity | O(N²): 1.1 × 10¹² interactions per step at N = 1M | [std]; [D] |
| Barnes-Hut complexity | O(N log N) | [P:barnes1986hierarchical] |
| Interactions per body at θ = 0.5 (Plummer) | ≈ 900–1,060, **almost flat** from N = 8,192 to 1M | [M] `docs/03-design.md` §16 |
| Work saved at N = 1M, θ = 0.5 | ≈ 1,000 interactions per body instead of 1,048,575: about **1,000× less work** | [D] |
| Force accuracy at θ = 0.5 | rms ≈ 0.7% (monopole) | [M] tree round 1, predictions §0 |
| Error and work vs θ | error ∝ θ^2.13, work ∝ θ^−2.52 | [M] chapter 4, §4.5 |
| Best all-pairs rate | ≈ 1,264 G interactions/s (N = 65,536) ≈ 25 TFLOP/s | [M] round 2; [D] |
| Best tree force evaluation, N = 1M (`bh_warp`, θ = 0.5) | **5.99 ms** for the traversal | [M] chapter 4, §4.7 |
| Same at N = 1M with all-pairs | ≈ 870 ms | [D] extrapolated, not measured |
| Crossover (tree faster than the best all-pairs) | N ≈ **28,700** (`bh_warp`); 39,700 (`bh_sorted`) | [M] chapter 4, §4.5, §4.7 |
| Gain from sorting the bodies (#5 vs #4) | **≈ 2.0×** at 262K and 1M | [M] chapter 4, §4.5 |
| Gain from the warp-cooperative walk (#7 vs #5) | **5.8×** at 1M, best vs best | [M] chapter 4, §4.7 |
| Tree depth (binary radix tree, N = 65,536) | 31 (Plummer) to 43 (dense core) levels | [M] `docs/03-design.md` §13 |
| Memory at N = 1M | ≈ 130 MB of 16 GB; traversal data 52 MB fits the 64 MB L2 | [D] `docs/03-design.md` §10 |

**Below the crossover, the simple O(N²) code wins:** it has perfect balance and no tree build.
Above it, the tree wins by a factor that grows with N. Choosing by N is part of the answer.

---

## 3.7 Publications

### 3.7.1 Foundations (the algorithm itself)
All verified on a DOI/Crossref record or publisher page (`references/INDEX.md`).

| Key | Publication | Why it matters |
|---|---|---|
| barnes1986hierarchical | J. Barnes, P. Hut, "A hierarchical O(N log N) force-calculation algorithm", *Nature* 324, 446–449, 1986. DOI 10.1038/324446a0 | **The** Barnes-Hut paper: octree, centre-of-mass approximation, θ |
| plummer1911distribution | H. C. Plummer, "On the Problem of Distribution in Globular Star Clusters", *MNRAS* 71(5), 460–470, 1911. DOI 10.1093/mnras/71.5.460 | the Plummer sphere, our main test galaxy |
| verlet1967computer | L. Verlet, "Computer 'Experiments' on Classical Fluids. I.", *Phys. Rev.* 159(1), 98–103, 1967. DOI 10.1103/PhysRev.159.98 | the Verlet/leapfrog integrator |
| warren1993hashed | M. S. Warren, J. K. Salmon, "A parallel hashed oct-tree N-body algorithm", *Supercomputing '93*, ACM, 12–21, 1993. DOI 10.1145/169627.169640 | parallel hashed oct-tree built on space-filling-curve (Morton) keys |
| salmon1994skeletons | J. K. Salmon, M. S. Warren, "Skeletons from the Treecode Closet", *J. Comput. Phys.* 111(1), 136–155, 1994. DOI 10.1006/jcph.1994.1050 | error analysis; failure of the classic s/d criterion → our bmax test |
| skilling2004hilbert | J. Skilling, "Programming the Hilbert curve", *AIP Conf. Proc.* 707, 381–387, 2004. DOI 10.1063/1.1751381 | the Hilbert key used by our `bh_hilbert` variant |
| karras2012maximizing | T. Karras, "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees", *High Performance Graphics* 2012, 33–37. DOI 10.2312/EGGH/HPG12/033-037 | the parallel radix-tree build (step 4) |
| nyland2007nbody | L. Nyland, M. Harris, J. Prins, "Fast N-Body Simulation with CUDA", *GPU Gems 3*, ch. 31, 2007 | tiled all-pairs on CUDA; q threads per body |
| burtscher2011barneshut | M. Burtscher, K. Pingali, "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm", *GPU Computing Gems Emerald*, ch. 6, 2011. DOI 10.1016/B978-0-12-384988-5.00006-1 | first all-GPU Barnes-Hut; sorting + warp-vote walk |

### 3.7.2 Recent work on the algorithm (2025–2026)
| Key | Publication | Country (lead) | What it adds |
|---|---|---|---|
| madan2025stochastic | A. Madan et al., "Stochastic Barnes-Hut Approximation for Fast Summation on the GPU", *SIGGRAPH '25*. DOI 10.1145/3721238.3730725 | Canada / USA (U. Toronto + NVIDIA) | a randomised, unbiased Barnes-Hut: "equal median error in up to 9.4x less time" (on graphics kernels) |
| nagarajan2025rtbarneshut | V. Nagarajan et al., "RT-BarnesHut: Accelerating Barnes-Hut Using Ray-Tracing Hardware", *PPoPP '25*, 43–56. DOI 10.1145/3710848.3710885 | USA | runs the tree walk on **RT cores** (our GPU has 80); argues tree walks fit SIMT cores poorly |
| stucker2026jzfmm | J. Stücker, O. Foldal, "JZ-FMM: GPU-native differentiable N-body simulations with the Fast Multipole Method", arXiv:2609.09307, 2026 (preprint) | Austria | the whole tree pipeline on the GPU, Morton sort, non-recursive walk: the same design choices as ours |
| thaler2026sfcneighbor | F. Thaler, S. Keller, "GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout", arXiv:2602.19873, accepted IPDPS 2026 | Switzerland | Hilbert ordering keeps consecutive particles compact on GPUs (our step 2 / #5) |
| frontiere2025crkhacc | N. Frontiere et al., "Cosmological Hydrodynamics at Exascale: A Trillion-Particle Leap in Capability", arXiv:2510.03557, 2025 | USA | state of the art: GPU-resident tree solvers, 4 × 10¹² particles |
| **yu2026cube2** | H.-R. Yu et al., "CUBE2: A parallel N-body simulation code for scalability, accuracy, and memory efficiency", *Sci. China Phys. Mech. Astron.* 69(6), 269511, 2026. DOI 10.1007/s11433-025-2926-0 | **China** (Xiamen U., SJTU) | the main alternative to trees: particle-mesh + particle-particle (P3M); cost-sorted load balancing (CPU code) |
| **liu2025swphotons** | Z. Liu et al., "swPHoToNs: Toward trillion-body-scale cosmological N-body simulations on Sunway TaihuLight", *Engineering Reports* 7(3), 2025 (online 2023). DOI 10.1002/eng2.12640 | **China** (Tsinghua, NSCC-Wuxi, NAOC) | tree/FMM gravity at 1.6 trillion bodies; dynamic load balancing |
| **wu2025dragon3** | K. Wu et al., "DRAGON-III simulation: modelling million-body globular and nuclear star clusters", arXiv:2510.03933, 2025 | Germany, with NAOC / PKU / SYSU co-affiliations | direct (all-pairs) N-body on GPUs, ported to Chinese DCU accelerators |
| **han2025jiutian** | J. Han et al., "The Jiutian simulations for the CSST extra-galactic surveys", arXiv:2503.21368, 2025 (venue not confirmed) | **China** (SJTU) | large Chinese N-body runs (Gadget / CUBE family, CPU codes) |
| *wang2021photonsgpu* (background) | Q. Wang, C. Meng, "PhotoNs-GPU: A GPU accelerated cosmological simulation code", *RAA* 21(11), 281, 2021 | China | Chinese GPU tree/FMM gravity code; FP32 ≈ 2× faster than FP64 at small accuracy cost |

**About the "preferably Chinese" request (R4)**, stated honestly: three literature passes found
**no 2025–2026 Chinese-led paper on Barnes-Hut or tree gravity on GPUs**. The Chinese 2025–26
N-body work we found is either a CPU/many-core code (CUBE2, swPHoToNs, Jiutian) or a port of a
direct code (DRAGON-III). The Chinese 2025–26 sources we do use are on the **GPU load-balancing**
side (chapter 4: HyTiS, Libra, Voltrix, Weave, G-Match, WaveTune, HybriMoE). The search log is
in `references/R3-evidence-map.md` §9. A further 2025 conference paper comparing direct and
Barnes-Hut (Balarezo & Pineda, CCIS vol. TICEC 2025, DOI 10.1007/978-3-032-08366-1_24) was found
but its content was not checked, so it is not used.

---

## 3.8 Slide outline (≈ 8 slides for this part)

1. **The problem.** N bodies and Newton's law; the formula with ε; "no closed form for N ≥ 3, so
   simulate". Visual: the viewer with two colliding galaxies.
2. **Direct summation.** The 5-line loop; O(N²); "10× bodies = 100× time". Visual: the
   N × N interaction matrix.
3. **Direct on a GPU.** One thread per body, shared-memory tiles [Nyland 2007]; 1,264 G
   interactions/s ≈ 25 TFLOP/s on our card.
4. **Leapfrog.** Kick-drift-kick; why not Euler (energy drift).
5. **Barnes-Hut idea.** "A far group = one body at its centre of mass" [Barnes & Hut 1986]; the
   2D worked example with s/d = 0.2 < θ.
6. **Our GPU pipeline.** The 7-step table from §3.5; Morton key example (5, 3) → 27.
7. **Parallel tree build and traversal.** Karras: every node independently; bmax test; the
   pseudocode; why sorting stops warp divergence.
8. **Numbers and publications.** The facts table (§3.6): 1,000× less work at 1M, 5.99 ms vs
   ≈ 870 ms, crossover ≈ 28,700. The recent-publications table (§3.7.2).

---

## 3.9 Glossary (EN → BG)

| English | Български |
|---|---|
| N-body problem | задача за N тела |
| direct summation / all-pairs | директно сумиране / всички двойки |
| softening (ε) | гравитационно омекотяване |
| time step, integrator | времева стъпка, интегратор |
| leapfrog (kick-drift-kick) | метод „прескачане" (leapfrog) |
| centre of mass | център на масата |
| octree / binary radix tree | октодърво / двоично радикс дърво |
| opening angle / opening criterion (θ) | ъгъл на отваряне / критерий за отваряне |
| space-filling curve (Morton, Hilbert) | запълваща пространството крива |
| tree traversal | обхождане на дървото |
| warp divergence | дивергенция в рамките на warp |
| load balancing | балансиране на натоварването |

---

## 3.10 Questions the lecturer is likely to ask

- **"Why not always use Barnes-Hut?"** Below about 28,700 bodies the tree build and the irregular
  walk cost more than they save; all-pairs is perfectly balanced and runs near half of peak.
  Measured, not assumed (§3.6).
- **"How accurate is the tree?"** rms force error ≈ 0.7% at θ = 0.5 with monopoles. It is
  validated against the exact all-pairs sum. At θ = 0 the tree reproduces all-pairs exactly,
  which is our self-test (`--selftest`).
- **"Why FP32?"** On this GPU FP64 runs at 1/64 of the FP32 rate (chapter 2). The softened force
  doesn't need FP64 for this study. FP64 is used only in the validation reference, never in the
  timed kernels.
- **"Why a binary tree and not an octree as in the 1986 paper?"** Karras' build is one
  lock-free kernel. The price is a deeper tree, a constant factor that is the same for all
  variants being compared.
- **"Why Morton and not Hilbert?"** We measured both: Hilbert was 1.5–4.4% *slower* in 8 of 9
  cells and needs a second sort, while Morton comes free with the build (chapter 4, §4.5).
- **"What is O(N log N) in practice?"** At θ = 0.5 we measured ≈ 1,000 interactions per body,
  almost constant from 8K to 1M bodies. The log N part shows up as tree depth, not interactions.
