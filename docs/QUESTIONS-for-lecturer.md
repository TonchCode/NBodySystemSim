# Въпроси към преподавателя / Questions for the lecturer

Проект: **N-body симулация на CUDA с изследване на техники за балансиране на натоварването на GPU**
(RTX 4080 SUPER, sm_89). Автор: Тончо Николов.

Какво е направено: виж [README](../README.md); архитектура на GPU →
[docs/final/02-gpu-architecture.md](final/02-gpu-architecture.md); техники за балансиране →
[docs/final/04-load-balancing.md](final/04-load-balancing.md); литература →
[references/INDEX.md](../references/INDEX.md).

## Формат и срокове

1. **Език на документацията и презентацията:** български или английски? (В момента черновата е на
   английски.)
2. **Срок** за презентацията и за предаване на проекта?
3. **Изискван формат** на документа: шаблон, обем, структура?
4. **Визуализация:** очаква ли се, или е допълнение? (Има работеща визуализация в реално време,
   CUDA–OpenGL interop.)

## Съдържание

5. **Публикации 2025–2026, „предимно китайски автори“:** не открихме китайска публикация от
   2025–2026 за **N-body на GPU** (търсенето е документирано в
   [references/R3-evidence-map.md](../references/R3-evidence-map.md) §9). Намерихме 7 китайски
   публикации от 2025–2026 за **балансиране на натоварването на GPU** при нерегулярни задачи
   (SJTU, Wuhan U, BUPT, PKU; SC'25, USENIX ATC'25), които се пренасят към нашите техники.
   Приемливо ли е това?
6. **Отрицателен резултат:** техниката „persistent threads + опашка от задачи“ (#6) е реализирана в
   4 варианта и **измерено е, че е по-бавна** от хардуерния планировчик на блокове, с обяснен
   механизъм. Приема ли се отрицателен резултат като изследвана техника?
7. **Колко техники за балансиране** очаквате? В момента: 7 реализирани и измерени (статично
   разпределение, block/tile настройка, няколко нишки на тяло, наивно Barnes-Hut обхождане,
   сортиране по крива на Мортън, persistent threads, warp-кооперативно обхождане). Планирано: #8,
   разпределение CPU+GPU.
8. **Профилиране:** Nsight Compute изисква разрешение за броячите на GPU (административна
   настройка). Използвахме собствени измервания (времеви линии на блоковете по SM, броячи на
   работата по warp). Достатъчно ли е, или се изискват метрики от Nsight?
9. **Повторяемост:** кодът е настроен за конкретната видеокарта (sm_89). Трябва ли да се стартира
   и на машина в лабораторията?

## Продължение

10. Може ли проектът да продължи като **дипломна работа** (напр. FMM или разпределение CPU+GPU)?

---

## English summary

1. Language of docs/presentation (BG or EN)? 2. Deadlines? 3. Required document format?
4. Is a visualisation expected? 5. No Chinese-led 2025–26 *GPU N-body* paper exists that we could
find; 7 Chinese-led 2025–26 GPU load-balancing papers transfer to our techniques. Acceptable?
6. Is a measured negative result (#6 persistent threads, slower than the hardware scheduler, with a
mechanism) acceptable as a studied technique? 7. How many load-balancing techniques are expected
(7 done, #8 planned)? 8. Are custom timeline/counter measurements enough without Nsight Compute
counters? 9. Must it run on a lab machine? 10. Can it continue into the diploma thesis?
