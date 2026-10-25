# Design spec: „N-body на RTX 4080 SUPER: стъпка по стъпка“

Single self-contained HTML page, vanilla JS, canvas (galaxy, packets) + inline SVG (GPU diagram, SM zoom,
tree). All UI text in Bulgarian, English terms kept as-is. Audience: the student himself, then a lecturer on a
projector. Goal: every visual answers one "защо" he can repeat out loud.

**Honesty rule (governs everything):** the physics is real (the JS runs the real algorithms). The GPU is a
*schematic model*. Measured numbers come only from the Nsight Compute table. Two visual registers:
- **модел**: solid panels, normal text, label chip `МОДЕЛ`.
- **измерено**: panel with a 2px left rule in `--measured`, label chip `ИЗМЕРЕНО НА RTX 4080 SUPER (Nsight Compute)`.
Never mix both in one number. Any number derived from the model is never placed in the measured table.

---

## 1. Layout

### 1.1 Desktop / projector (≥ 1200 px wide; design target 1366×768 and 1920×1080)

```
┌──────────────────────────────── top bar (56 px) ───────────────────────────────────┐
│ Заглавие · [Режим ▾] · N [512|1024|2048|4096] · ⏮ ◀ ▶/❚❚ ▶| · Скорост ▾ · ◐ тема │
├──────────────────────────────┬──────────────────────────────────────────────────────┤
│ A. ГАЛАКТИКА (canvas)        │ B. GPU: AD103 (SVG), always visible, ~45% height    │
│ square, ~40% width           │                                                      │
│ body dots + overlays         ├──────────────────────────────────────────────────────┤
│                              │ C/D. ZOOM PANEL, tabs: [SM] [Дърво + warp] [Конвейер]│
│                              │ ~55% height, auto-switches by step, user can pin   │
├──────────────────────────────┴──────────────────────────────────────────────────────┤
│ E. STEPPER STRIP (≈150 px): kernel timeline · step caption (18 px) · „Защо?“ box    │
├──────────────────────────────────────────┬──────────────────────────────────────────┤
│ F. Брояч МОДЕЛ                           │ G. ИЗМЕРЕНО НА RTX 4080 SUPER            │
└──────────────────────────────────────────┴──────────────────────────────────────────┘
```
- Grid: `grid-template-columns: minmax(360px, 40%) 1fr`. Panels have 12 px gaps, 16 px padding, 8 px radius.
- At 1366×768 the F/G row collapses into a drawer (button „Броячи ▴“) so A/B/C stay large. At 1080p it is
  always shown.
- **Always visible:** A, B, E, top bar. **In the zoom panel:** SM detail, tree/warp detail, pipeline view.
  The zoom panel follows the current step (e.g. the tile load step shows tab SM, traversal shows tab Дърво +
  warp). A pin icon 📌 (use an SVG icon, not an emoji) locks the tab.
- Visual connector: when the zoom panel shows SM #k, that SM in B gets a 2 px outline in `--focus`, plus
  a thin leader line from SM #k to the zoom panel's top-left corner.

### 1.2 Phone (< 768 px)
Single column, 16 px gutters, no horizontal scroll: top bar wraps to 2 rows (mode + N / transport). Then
A (full width, square), E (caption), a segmented control [GPU | SM | Дърво | Броячи] that shows one of B/C/D/F+G
at a time. B is simplified: the 80 SMs as a 10×8 grid of 14 px cells, L2 as a bar under them, VRAM as one bar.
Packets are drawn but the latency labels are hidden.

### 1.3 Panel A: galaxy
- 2D projection, Plummer-like disc. Dot radius 2.5 px (N≤1024), 1.8 px (N≥2048). Footnote:
  „2D изглед; реалният код е 3D.“
- **Colour modes** (toggle under the canvas):
  1. **„Къде са данните“** (default): each body coloured by the memory level that currently holds the
     *active copy* of its data for the current step (see §3 palette). Bodies not involved in this step:
     `--dim` at 40% opacity.
  2. **„Кой ги обработва“**: selected warp = `--sel` (orange) with a 1 px white ring; the rest of its block
     = amber; bodies of other resident blocks = blue; bodies with no thread yet = grey.
- **Overlays** (auto per step, toggleable): tile outline (the 256 bodies of the current tile get a hollow
  ring); Z-order (Morton) curve as a 1 px polyline in body-sort order; BH cells: node bounding squares.
- Click a body → selects it and its warp (lane = body index mod 32 in thread order). Hover → tooltip:
  `тяло #417 · нишка 417 · warp 13 · блок 1 → SM 1 · данни: регистри`.

### 1.4 Panel B: GPU architecture (SVG, viewBox 1000×520)
```
 HOST (CPU+RAM) ══ PCIe 4.0 x16 · 31.5 GB/s/посока ══╗
                                                     ║
 ┌──── 7 GPC · 40 TPC · 80 SM (схематично) ─────────────────────────────┐
 │ row 1-4: 40 SM cells (TPC = pair of SMs, 2 px gap inside, 6 px btw) │
 │ ═══════════════ L2 · 64 MB (общ за всички SM) ═══════════════════════ │
 │ row 5-8: 40 SM cells                                                   │
 └────────────────────────────────────────────────────────────────────────┘
  MC MC MC MC   (4 left)        (4 right)   MC MC MC MC   → 8 memory controllers
  [VRAM 16 GB GDDR6X · 256-bit · 736 GB/s] as chips on both sides
```
- SM cell 40×28 px, numbered 0–79 (mono 10 px). Footnote under the frame: „Разпределението на 40-те TPC по
  7-те GPC не е публикувано: GPC рамката е схематична. Разпределението на блоковете по SM е модел (по
  ред); реалният планировчик не е документиран.“
- **SM states:** idle = `--idle` fill + 45° hatch + no number glow; resident block(s) = fill in `--sm-busy`
  with a mini bar of occupied warps (0–48) along the bottom edge; selected SM = `--focus` outline.
- Corner legend with the latency ladder (mono): `регистри ~0 · shared ~30 · L1 ~32 · L2 ~270–285 · VRAM ~570
  цикъла`, footnote „латентности: RTX 4090, същата архитектура (Luo et al. 2025)“.
- Footnote, one line, grey: „Tensor ядра (320) и RT ядра (80) съществуват, но проектът не ги използва.“
  Not drawn.

### 1.5 Panel C: SM zoom (tab „SM“, SVG 600×340)
- Four partition columns, each: `warp scheduler · 1 инстр./такт` box; 32 FP32 lane squares (8×4);
  `SFU` box with 4 slots (×4 partitions = 16 rsqrt/такт/SM, vs 128 FMA); a register-file slice
  (64 KB of 256 KB) drawn as a bar filled to current usage.
- Under the partitions: one wide bar `L1 / shared memory · 128 KB (обединени)` split into a `shared` segment
  (sized to what the kernel uses, e.g. tile = 256×16 B = 4 KB, highlighted) and the `L1` rest, which holds
  cached lines (small squares that light up on hit, flash on fill).
- Right edge: 48 warp slots (12 per partition), occupied slots filled, the selected warp in `--sel`.
- During rsqrt: FMA lanes flash 128 small ticks while the SFU shows 16 ticks: the ratio is visible.

### 1.6 Panel D: tree + warp (tab „Дърво + warp“)
- **Top 60%: tree.** Karras binary radix tree, drawn top-down, depth capped at 7 visible levels; deeper
  subtrees collapse into a triangle with a count. Node glyph = 10 px circle. States: not visited (outline),
  loaded this step (filled `--l1` if L1 hit, `--l2` if served from L2), **accepted** (filled + ✓ glyph),
  **opened** (outline thickened + ▾ glyph), **leaf summed** (square glyph). Mirrored in panel A: the node's
  bounding square is drawn, accepted = filled 15% with COM dot and a line to the body, opened = dashed.
- Under each visited node: a 32-px micro bar (1 px per lane) showing which lanes requested it. This bar is
  what makes #4 vs #5 vs #7 legible: scattered, partly overlapping, all-32.
- **Bottom 40%: the selected warp.** 32 lane columns (each ≥ 14 px wide at desktop):
  - lane state cell: **active** = filled square in branch colour; **masked** = hollow square with a
    diagonal slash; **finished** = hollow, grey dot. Shape always differs, not only hue.
  - branch letter on active cells: `П` (приет, accept), `Л` (листо, leaf), `О` (отвори, push children).
  - per-lane stack depth bar above each lane. #4/#5: hatched navy (`--local`) labelled „стек в local
    memory (физически VRAM, кеширан в L1/L2), 384 B/нишка“. #7: one shared strip of 64 cells (32 lanes ×
    2 регистъра) in `--reg` labelled „общ стек в регистри; pop = __shfl_sync“.
  - running text above: `активни ленти: 7 / 32` (big, mono 20 px).

### 1.7 Tab „Конвейер“ (BH only)
Seven chips in a row, `1 кутия → 2 Morton → 3 сортиране → 4 дърво → 5 обобщение → 6 обхождане → 7 интегриране`,
the current kernel filled. Under them a bar with a proportional time width per kernel where chip 6 is
labelled „>90% от времето (измерено)“; the other widths are drawn as "small", not to scale, with label
„не в мащаб“. Plus a 1D strip „масив тела в VRAM“ (one cell per body, colour = Morton key gradient), used
by steps 2–3 to show the reorder.

---

## 2. Stepper

### 2.1 Controls
- `⏮` reset step · `◀` step back · `▶/❚❚` play · `▶|` step forward · `⏭` „до края на ядрото“ (fast-forward the
  current kernel's repeated sub-steps, e.g. remaining tiles or the rest of a traversal).
- **Speed:** `1 стъпка/с · 1 / 2 с · 1 / 4 с · 1 / 10 с · ръчно` (default 1 / 4 с). Packet animation time
  is ≤ 70% of the step duration.
- **Back** is real: the engine snapshots state (positions, cache model, counters) at each micro-step of the
  current time step (ring buffer of ≤ 400 snapshots); crossing into the previous time step re-simulates from
  its start.
- **N**: 512 / 1024 / 2048 / 4096 (default 1024; resets the time step). **Mode ▾**: #1, #3, #4, #5, #7,
  then a separator and #6, #8 under „обяснения“.
- **Select:** click a body (A), an SM (B) or a lane (D). Inputs `warp №` and `тяло №` in the top bar's
  overflow menu.
- Keyboard: Space play/pause, ←/→ step, Shift+→ fast-forward, +/− speed, 1–7 mode, T theme, Esc unpin.
- **Timeline** in strip E: segmented bar of the steps of the current mode; the current one is filled,
  clickable. Kernels are separated by a taller tick and labelled with the kernel name in mono.

### 2.2 Caption format
Line 1 (18 px, semibold): what happens. Line 2 (15 px): where the data is, with inline level badges
`[R]` `[S]` `[L1]` `[L2]` `[V]` `[H]` `[loc]` in their colours. „Защо?“ box (right of the caption, 15 px):
appears only on steps marked ★ below.

### 2.3 Common prefix (all modes): one time step, leapfrog KDK
- **K0 ★** „Начало на времева стъпка (leapfrog: kick–drift–kick). Позициите и масите (float4) и скоростите
  са в VRAM `[V]`.“ Why: „Силата е скъпата част; kick/drift са евтини и ограничени от паметта.“
- **K1** „Kick ½ + drift: ядро *integrate*: всяка нишка чете своето тяло, обновява v и x, записва обратно.“
  `[V]→[L2]→[R]→[V]` packets for a sampled 32 bodies.
- **K2** „Изчисляване на силите, избраният режим.“ → mode-specific steps below.
- **K3** „Kick ½ с новите ускорения. Край на стъпката; галактиката се придвижва.“ Dots move.

### 2.4 #1 all-pairs tiled
1. **★ „Стартиране на ядрото: N/256 блока по 256 нишки.“** At N=1024: 4 блока → SM 0–3; „76 от 80 SM стоят
   празни.“ B greys 76 SMs. Why: „Малко N = твърде малко блокове; GPU-то е недонатоварено.“
2. „Всяка нишка зарежда *своето* тяло в регистри.“ `[V]→[L2]→[R]`; A: the block's bodies turn `--reg`.
3. „Плочка t: всяка от 256-те нишки зарежда по едно float4 в shared memory: съседни нишки, съседни адреси
   (coalesced).“ `[V]/[L2]→[S]`; A: the tile's 256 bodies ringed, coloured `--shared`.
4. „__syncthreads(): бариера. Никоя нишка не продължава, докато плочката не е цялата в shared.“ Lanes
   pulse once, all together.
5. **★ „Вътрешен цикъл j = 0..255: всички нишки четат shared[j], същия адрес за всички (broadcast).“**
   One `[S]→[R]` packet fans out to 32 lanes. Why: „Всяко заредено тяло се използва 256 пъти: изчисление на
   байт е високо → compute-bound.“
6. „Взаимодействие: r = xj − xi; d² = r·r + ε²; inv = rsqrt(d²) в SFU; s = mj·inv³; a += r·s (~20 FLOP).“
   C: FMA ticks 128 vs SFU 16. Shown in full for j = 0, 1; then `⏭` hint.
7. „__syncthreads() и следваща плочка.“ (loop 3–7, N/256 times; after 2 tiles, `⏭` is offered.)
8. „Запис на ускоренията: регистри → VRAM.“ `[R]→[L2]→[V]`.

### 2.5 #3 multi (q threads per body; default q = 4 at N=1024, user can set 2/4/8)
1. **★ „q = 4 нишки на тяло → 4× повече блокове: 16 блока.“** B lights 16 SMs; text: „при N=4096 това
   дава 2.56× (измерено)“ (chip ИЗМЕРЕНО). Why: „Същата работа, разделена на повече SM.“
2. „Всяка от q-те нишки обхожда своя 1/q част от телата (своите плочки).“ A: the 4 sub-ranges in 4
   lightness steps of blue.
3. „Плочки в shared, както в #1, но всяка група от q нишки чете различни части.“ (compressed: 1 step)
4. **★ „Събиране на частичните суми: __shfl_xor_sync, регистър → регистър, без памет.“** D: butterfly
   arrows inside the warp, log₂q rounds (2 rounds at q=4), all in `--reg`. Why: „Обменът между лентите в warp
   минава директно между регистрите, без shared и без L1.“
5. „Първата нишка от групата записва резултата.“ `[R]→[V]`.

### 2.6 BH pipeline, kernels 1–5 and 7 (shared by #4, #5, #7)
1. „Ядро 1, ограничаваща кутия: паралелна редукция min/max по всички тела.“ A: box shrinks onto the bodies.
2. **★ „Ядро 2, Morton ключове: координатите се квантуват и битовете им се преплитат → едно число на тяло.“**
   A: Z-curve overlay appears. Why: „Близки ключове ⇒ близки тела. Това е основата на #5 и #7.“
3. „Ядро 3, радикс сортиране (CUB) по ключ. Масивът в VRAM се пренарежда.“ Pipeline strip animates the
   reorder. Note: „библиотечно ядро, няколко прохода през паметта: не го разглобяваме.“
4. **★ „Ядро 4, дърво на Karras: по една нишка на вътрешен възел. Всяка намира своя диапазон от общия
   префикс на съседните ключове. Без заключвания (locks).“** D: internal nodes appear in parallel, not
   top-down. Why: „Всеки възел се определя само от сортираните ключове, затова всички се строят наведнъж.“
5. „Ядро 5, обобщение отдолу нагоре: маса, център на масата и bmax. При родителя: атомарен брояч.
   Първата пристигнала нишка спира, втората изчислява родителя.“ D: counters 0→1→2 on nodes.
6. → traversal (mode-specific, below).
7. „Ядро 7, интегриране.“ Same as K3.

### 2.7 Traversal: one iteration of the loop, shown for the selected warp
Shared step list `T1–T5`, captions differ per mode. Other warps run as aggregate counters only (no packets).

**#4 bh_naive**
- T0 **★** „Нишките вземат телата в *оригиналния* ред: 32-те тела на warp-а са разпръснати по галактиката.“
  A: selected warp's 32 dots in `--sel`, spread out. Why: „Различни тела ⇒ различни пътища в дървото.“
- T1 „Всяка лента изважда (pop) възел от *своя* стек: local memory (физически VRAM, кеширан в L1/L2).“
  `[loc]→[L1]` hit or `[L2]` miss per lane.
- T2 **★** „Всяка лента чете *различен* възел: до 32 отделни заявки. Повечето пропускат L1 → L2.“ Up to 32
  packets L2→L1. Why: „Пречката е L1↔L2 (L2 е 95% заето), не VRAM: дървото се събира в 64 MB L2 (DRAM ~1%).“
- T3 „Тест за отваряне във всяка лента: |x − COM|² ≥ (bmax/θ)², θ = 0.5.“ Lane cells show П / Л / О.
- T4 **★** „Дивергенция: клоновете П, Л и О се изпълняват *последователно*; във всеки са активни само част
  от лентите.“ D: three sub-flashes, one per branch, with masked lanes slashed. Why: „SIMT: една инструкция за
  целия warp; неучастващите ленти чакат. Измерено: средно 6.1 от 32 активни.“
- T5 „О: децата се записват (push) в стека в local memory.“ `[R]→[loc]`.
- Loop T1–T5; `⏭` until the warp ends; then „Запис на ускоренията `[R]→[V]`.“

**#5 bh_sorted**: same steps; changed captions:
- T0 **★** „Нишките вземат телата в *Morton* ред: warp = 32 пространствени съседи.“ A: tight cluster on the
  Z-curve. Why: „Съседите имат сходни пътища в дървото.“
- T2 „Съседните ленти често искат *същия* възел: една заявка, после попадения в L1.“ Node micro bars largely
  overlap. Caption adds: „измерено: L1 hit 50% (срещу 19% при #4)“.
- T4 „Дивергенцията остава, но е по-малка. Измерено: 10.8 от 32 активни; 2.06× спрямо #4.“
- T5 **★** „Стекът все още е в local memory.“ Why: „39% от трафика при L1-пропуски е самият стек
  (измерено). Това води към #7.“

**#7 bh_warp**
- T0 „Morton ред, както в #5, но целият warp обхожда *един общ* път.“
- T1 „Pop от общия стек: в регистри (2 на лента = 64 записа), чрез __shfl_sync. Без памет.“ `[R]→[R]`.
- T2 **★** „Един възел → *една* заявка → резултатът се разпраща до 32-те ленти.“ One packet, fan-out. Why:
  „14× по-малко сектори прочетени (измерено); L2 пада от 95% на 24%.“
- T3 „Всяка лента прави теста за себе си; __any_sync: ако *поне една* лента иска да отвори, warp-ът отваря.“
  Lane cells show votes, then all 32 adopt the result.
- T4 **★** „Няма дивергенция: всички 32 ленти правят едно и също.“ All cells filled. Why: „Някои ленти правят
  повече взаимодействия от нужното (1.6–1.9× повече), но с 32 активни ленти е 5.8× по-бързо от #5:
  compute-bound.“
- T5 „Децата → общия стек в регистри.“

### 2.8 Explainer modes (static-ish, 3–4 steps each, no traversal animation)
**#6 persistent + опашка**
1. „Пускат се само толкова блока, колкото се побират едновременно; те остават живи (persistent).“
2. „Всеки warp взема следваща порция работа от софтуерна опашка (атомарен брояч).“
3. **★** „Последователни порции (пространствени съседи) попадат на *различни* SM.“ B: neighbour warps
   coloured the same, scattered across SMs. Why: „L1 е на SM: съседите вече не делят L1. Измерено: 0.63–0.97×,
   т.е. по-бавно.“

**#8 CPU + GPU разделяне**
1. „Част от работата отива на CPU. Данните трябва да минат по PCIe 4.0 x16 (31.5 GB/s в посока).“ Packets
   on the PCIe bar, slow.
2. **★** „При 1M тела преносът струва 2.26 ms на стъпка (измерено). Печалба няма.“ Why: „PCIe е ~23× по-тесен от
   VRAM (736 GB/s); преносът изяжда спечеленото време.“

---

## 3. Highlighting language

### 3.1 Memory-level palette (same in dots, packets, badges, panels)
Rule: **near the cores = warm (orange), far = cool (blue), plus a lightness ladder**, so the order reads
in greyscale too. Every level also has a letter badge and a fixed pattern.

| level | badge | pattern | light hex | dark hex |
|---|---|---|---|---|
| registers | R | solid | `#D9600B` | `#FF8A3D` |
| shared memory | S | solid + 1 px inner ring | `#E9A23B` | `#F5C06A` |
| L1 (cache) | L1 | dotted fill | `#C9B458` | `#E0D07E` |
| L2 | L2 | solid | `#3A8FC4` | `#5FB0E6` |
| VRAM | V | solid | `#244C8F` | `#7C9FE0` |
| local memory | loc | VRAM colour + 45° hatch | `#244C8F` + hatch | `#7C9FE0` + hatch |
| host / PCIe | H | dashed outline | `#7A808A` | `#9AA0AA` |

(In dark mode the ladder flips lightness: near = lightest/warmest, so the order is still monotonic.)

Other tokens:
| token | light | dark | use |
|---|---|---|---|
| `--bg` | `#F7F7F5` | `#111317` | page |
| `--panel` | `#FFFFFF` | `#1A1D23` | panels |
| `--ink` | `#1B1E23` | `#E6E8EB` | text |
| `--ink-2` | `#5A616B` | `#9AA3AE` | secondary |
| `--rule` | `#E2E3E0` | `#2A2F37` | borders |
| `--idle` | `#D5D7DA` | `#2B2F36` | idle SM (+hatch) |
| `--sm-busy` | `#9CC6E8` | `#2D5878` | SM with resident block |
| `--sel` | `#D9600B` | `#FF8A3D` | selected warp/body (+ white/black ring to tell it apart from "in registers") |
| `--focus` | `#111111` | `#FFFFFF` | selected SM outline |
| `--measured` | `#7A4FB5` | `#B394E8` | left rule + chip of measured panel only |
| `--dim` | `#B8BCC2` | `#4A505A` | uninvolved bodies |

Theme: `:root` tokens; `@media (prefers-color-scheme: dark)` under `:root:not([data-theme="light"])` and again
`:root[data-theme="dark"]`; toggle in the top bar; `body` gets an explicit background.

### 3.2 Packets
- 6 px rounded rectangle in the *source* level's colour with its badge letter at ≥ 14 px zoom; it travels along
  fixed SVG paths: `VRAM → MC → L2 → SM(k).L1/shared → partition → lane`. Write-back reverses the path.
- Travel time ∝ √(latency), so VRAM visibly lags (registers instant, shared/L1 short, L2 ≈ 3×, VRAM ≈ 4.3×
  the shared time). Legend: „времената на анимацията са условни; цифрите в цикли са от RTX 4090“.
- Hit/miss: at L1 the packet either stops (✓ hit, short flash) or continues to L2 (✗ miss, the L1 line
  flashes as it fills on return). Coalesced = one fat packet labelled `×32`; scattered = many thin ones.
- Detailed packets only for the selected warp; other warps add to counters. Max 64 packets on screen.

### 3.3 SIMT lanes
Active = filled square, masked = hollow + slash, done = hollow + grey dot. Text `активни: k/32` always next
to the lane row. The same fraction is drawn as a gauge in F.

### 3.4 SM occupancy
Idle = grey + hatch, with a big count on B: `заети SM: 4 / 80` (mono 20 px). Never use red for idle; idle is
a fact, not an error.

---

## 4. Counters

### 4.1 F: МОДЕЛ (updates live, per current time step, chip `МОДЕЛ`)
- `заети SM: x / 80`
- `активни ленти (средно на инструкция): k / 32`, gauge
- `L1 попадения / пропуски: h / m (h%)`, modelled per SM: LRU, 128 B lines, 32 B sectors.
  **Capacity is scaled to N** (e.g. 1/64 of the tree's bytes, min 16 lines), otherwise at N=1024 the whole
  tree would fit and every access would hit. Label: „L1 е мащабиран модел; при истинския размер малкото N
  би се събрало изцяло.“
- `заявки към L2 (сектори): s`
- `взаимодействия: n` (for #7 vs #5, shows the 1.6–1.9× more)
- `зареждания от shared: …` (all-pairs only)
Under the panel: „Броячите описват модела в браузъра, не хардуера. Сравнявай *посоката* (кое е по-голямо),
не абсолютните стойности.“

### 4.2 G: ИЗМЕРЕНО НА RTX 4080 SUPER (static, Nsight Compute)
| | all-pairs | #4 | #5 | #7 |
|---|---|---|---|---|
| заети issue slots | 68% | 21% | 25% | 69% |
| активни ленти | 32 | 6.1 | 10.8 | 32 |
| L2 пропускателност | 2% | 95% | 95% | 24% |
| DRAM пропускателност | 0.5% | 1% | 1% | 2% |
| прочетени сектори | 8M | 4599M | 2874M | 200M |
The column of the current mode is highlighted. Footer note, always shown: „DRAM е ~1% дори при #4/#5: дървото
се събира в 64 MB L2. Пречката е L1↔L2, не VRAM.“ Second note: „конфигурация на измерването: както в
доклада.“ (The builder must not invent the N used for profiling.)
Extra measured facts shown as chips in the relevant steps only: 2.56× (#3, N=4096), 2.06× (#5 vs #4),
5.8× (#7 vs #5), 14× fewer sectors, 39% stack traffic, 0.63–0.97× (#6), 2.26 ms/step (#8, 1M).

---

## 5. Typography and accessibility
- **IBM Plex Sans** (400/600) for UI, **IBM Plex Mono** (400/500) for numbers, kernel names, code tokens;
  both with `&subset=cyrillic` (Google Fonts). Sizes: captions 18/15, labels 13, SM numbers 10, counters 20.
  Projector minimum: 13 px. Tabular numerals (`font-variant-numeric: tabular-nums`).
- Never hue alone: badges + patterns for memory levels, shapes for lane states, hatch for idle, ✓/▾/□ for
  node states. Blue/orange is the main contrast pair; avoid red/green.
- Text contrast ≥ 4.5:1 in both themes; packets ≥ 3:1 against the panel.
- `prefers-reduced-motion`: packets jump to their destination with a 150 ms fade; manual stepping still works.
- All controls are keyboard reachable; captions in an `aria-live="polite"` region; SVG has `<title>` on SMs and
  nodes.

## 6. Leave out (on purpose)
- Cycle-accurate timing, warp-scheduler arbitration, instruction caches, texture units, L2 slices or
  partitions, GDDR6X signalling, clocks or boost.
- Tensor/RT cores beyond the one footnote.
- Shared-memory bank conflicts (all-pairs uses broadcast; mention only in caption 2.4.5).
- CUB radix sort internals (one step), Karras prefix maths beyond one sentence, bounding-box reduction tree.
- 3D rendering; ε and θ tuning sliders (fixed ε, θ = 0.5).
- Packets for every warp: only the selected warp is detailed.
- Any number on the GPU diagram that looks measured but isn't. Any suggestion that VRAM limits the tree
  kernels.

## 7. Acceptance checks (for the builder)
1. At N=1024, #1 step 1 shows exactly 4 busy SMs and „76 от 80 SM стоят празни“.
2. Toggling #4 → #5 on the same body visibly turns the warp's 32 dots from scattered to clustered, and the
   node micro bars from sparse to overlapping.
3. In #7 every traversal step shows 32/32 filled lanes and one packet per node.
4. The measured panel never changes with N or play; the model panel always does.
5. Greyscale screenshot: memory levels, lane states and idle SMs are still distinguishable.
6. 375 px width: no horizontal scroll; caption readable without opening any tab.
