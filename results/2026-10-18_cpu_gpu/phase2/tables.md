## Scorecard

70 HIT / 12 MISS of 82 rows.

| ID | Prediction | Range | Measured | Verdict |
|---|---|---|---|---|
| C1-16384 | cpu_allpairs best rate, 20 LPs dynamic, compute only (G int/s) | 30-50 | 38.32 | HIT |
| C1-65536 | cpu_allpairs best rate, 20 LPs dynamic, compute only (G int/s) | 30-50 | 37.71 | HIT |
| C1-262144 | cpu_allpairs best rate, 20 LPs dynamic, compute only (G int/s) | 30-50 | 37.85 | HIT |
| C2 | time IV1 / IV2 at 65536 (tile 2048) | 1.00-1.25 | 1.018 | HIT |
| C3 | time rsqrt0 / rsqrt1 at 65536 | 1.3-3.0 | 1.424 | HIT |
| C4 | time tile512/tile2048 ; tile8192/tile2048 at 262144 (IV 2) | 0.95-1.05 each | 1.0173 ; 0.9892 | HIT |
| C5-65536 | cpu_bh best traversal rate, G {16,32} x leaf {8,16,32}, Plummer (G int/s) | 12-28 | 23.07 | HIT |
| C5-262144 | cpu_bh best traversal rate, G {16,32} x leaf {8,16,32}, Plummer (G int/s) | 12-28 | 20.17 | HIT |
| C5-1048576 | cpu_bh best traversal rate, G {16,32} x leaf {8,16,32}, Plummer (G int/s) | 12-28 | 18.78 | HIT |
| C6 | cpu_bh build (keys+sort+tree) Plummer 1M, G32 leaf 16 (ms) | 5-25 | 10.54 | HIT |
| C7 | time(G16)/time(G32), Plummer 1M, best leaf each (production median) | 0.85-1.10 | 1.233 | MISS |
| C8-a | time(G8)/time(G16), Plummer 1M, best leaf | 1.00-1.40 | 1.306 | HIT |
| C8-b | time(G64)/time(G32), Plummer 1M, best leaf | 0.95-1.30 | 0.9663 | HIT |
| C9 | cpu_bh_scalar / cpu_bh traversal time, Plummer 1M, best leaf each | 3-10 | 7.466 | HIT |
| C10 | CPU crossover N (cpu_bh best = cpu_allpairs best, compute only, log-log interpolation) | 1500-8000 | 4208 | HIT |
| C11 | cpu_allpairs D-009 PASS in every run; normMax/naive's within +-10% | all PASS, +-10% | 128 runs, all PASS=True, ratio 0.8338-1.0400 | MISS |
| C12 | cpu_bh/cpu_bh_scalar TN1/TN2 PASS and same-work PASS in every run | all PASS | 128 runs, all PASS=True; same-work check printed (G32 / scalar) and PASS in 70 | HIT |
| E1 | rate(p,12)/rate(p1,6), dynamic | 1.00-1.25 | 1.001 | HIT |
| E2 | rate(e,8)/rate(p1,6), dynamic | 0.22-0.45 | 0.2771 | HIT |
| E3 | rate(all dyn)/rate(p dyn) | 1.15-1.45 | 1.22 | HIT |
| E4 | rate(all dyn)/rate(all static) | 1.40-2.00 | 1.764 | HIT |
| E5 | rate(all static)/rate(p dyn) | 0.60-0.95 | 0.6917 | HIT |
| E6 | rate(all weighted)/rate(all dyn) | 0.93-1.02 | 1 | HIT |
| E7-static | worker busy max/mean, all 20, static | 1.40-2.20 | 1.557 | HIT |
| E7-dynamic | worker busy max/mean, all 20, dynamic | 1.00-1.20 | 1.023 | HIT |
| E7-weighted | worker busy max/mean, all 20, weighted | 1.00-1.20 | 1.048 | HIT |
| E8 | rate(cores,14 dyn)/rate(all dyn) | 0.85-1.00 | 0.9919 | HIT |
| E9 | cpu_bh 1M all 20: traversal static/dynamic | 1.5-2.3 | 1.644 | HIT |
| E10 | cpu_bh 1M all 20: traversal static/weighted | 1.3-2.0 | 1.626 | HIT |
| E11 | cpu_bh 1M all 20: traversal weighted/dynamic | 1.0-1.25 | 1.011 | HIT |
| E12 | cpu_bh 1M all 20: build static/dynamic | 1.2-2.0 | 1.129 | MISS |
| S1 | all-pairs GPU/CPU speedup at 1024 (GPU event, CPU compute) | 4-15 | 6.2 | HIT |
| S2 | all-pairs GPU/CPU at 4096 | 15-35 | 22.82 | HIT |
| S3-16384 | all-pairs GPU/CPU at 16384 | 20-40 | 29.2 | HIT |
| S3-65536 | all-pairs GPU/CPU at 65536 | 20-40 | 33.33 | HIT |
| S3-262144 | all-pairs GPU/CPU at 262144 | 20-40 | 35.54 | HIT |
| S4 | largest N in {64..1024} where CPU best <= GPU best (event-timed) | 64-384 | 64 | HIT |
| S5-65536 | tree GPU/CPU total (bh_warp best event vs cpu_bh best compute), Plummer | 6-14 | 8.724 | HIT |
| S5-262144 | tree GPU/CPU total (bh_warp best event vs cpu_bh best compute), Plummer | 9-22 | 14.25 | HIT |
| S5-1048576 | tree GPU/CPU total (bh_warp best event vs cpu_bh best compute), Plummer | 10-24 | 18.32 | HIT |
| S6 | tree traversal-only GPU/CPU, Plummer 1M | 8-25 | 18.35 | HIT |
| S7 | tree build-only GPU/CPU, Plummer 1M | 8-40 | 16.23 | HIT |
| S8 | tree GPU/CPU total, cube_edges 1M | 6-30 | 15.63 | HIT |
| S9 | GPU best / CPU best over any kernel, Plummer 1M | 10-24 | 18.32 | HIT |
| H1-65536 | hyb_allpairs dynamic / GPU best all-pairs (--wall) at 65536 | 0.97-1.02 | 1.083 | MISS |
| H1-131072 | hyb_allpairs dynamic / GPU best all-pairs (--wall) at 131072 | 0.96-1.0 | 1.024 | MISS |
| H1-262144 | hyb_allpairs dynamic / GPU best all-pairs (--wall) at 262144 | 0.955-0.995 | 0.9979 | MISS |
| H2 | hyb_allpairs dynamic / GPU best all-pairs (--wall) at 16384 | 1.0-1.1 | 1.24 | MISS |
| H3 | final GPU share f (dynamic) at 262144 | 0.955-0.985 | 0.9736 | HIT |
| H4 | static f (session rates) / dynamic, time, 262144 | 0.98-1.02 | 1.02 | HIT |
| H5-1048576 | hyb_bh dynamic / bh_warp (--wall), Plummer 1048576 | 0.95-1.02 | 1.002 | HIT |
| H5-262144 | hyb_bh dynamic / bh_warp (--wall), Plummer 262144 | 0.98-1.1 | 1.064 | HIT |
| H5-65536 | hyb_bh dynamic / bh_warp (--wall), Plummer 65536 | 1.0-1.3 | 1.272 | HIT |
| H6 | hyb_bh dynamic / bh_warp, Plummer 1M theta 0.3 | 0.93-1.00 | 0.9914 | HIT |
| H7 | hyb_bh dynamic / bh_warp, cube_edges 1M | 0.98-1.05 | 1.043 | HIT |
| H8 | hyb_bh static f 0.95 / bh_warp, cube_edges 1M | 2.0-5.0 | 3.58 | HIT |
| H9 | tree D2H (hybrid transfer per launch, 56 B/body) at 1M (ms) | 2.0-3.5 | 2.259 | HIT |
| H10-65536 | hyb_phase / bh_warp (--wall), plummer 65536 | 0.55-0.95 | 0.7929 | HIT |
| H10-262144 | hyb_phase / bh_warp (--wall), plummer 262144 | 0.82-0.97 | 0.9058 | HIT |
| H10-1048576 | hyb_phase / bh_warp (--wall), plummer 1048576 | 0.92-0.99 | 0.9591 | HIT |
| H11-262144 | hyb_phase / bh_warp (--wall), cube_edges 262144 | 0.5-0.9 | 0.5135 | HIT |
| H11-1048576 | hyb_phase / bh_warp (--wall), cube_edges 1048576 | 0.55-0.85 | 0.751 | HIT |
| H12-plummer-65536 | hyb_phase / bh_warp_refit K (nearer the mean lag 154.25) = 16, plummer 65536 | 0.97-1.05 | 1.013 | HIT |
| H12-plummer-262144 | hyb_phase / bh_warp_refit K (nearer the mean lag 18.66) = 16, plummer 262144 | 0.97-1.05 | 0.9999 | HIT |
| H12-plummer-1048576 | hyb_phase / bh_warp_refit K (nearer the mean lag 6.50) = 4, plummer 1048576 | 0.97-1.05 | 0.9981 | HIT |
| H12-cube_edges-262144 | hyb_phase / bh_warp_refit K (nearer the mean lag 881.50) = 16, cube_edges 262144 | 0.97-1.05 | 1.013 | HIT |
| H12-cube_edges-1048576 | hyb_phase / bh_warp_refit K (nearer the mean lag 98.00) = 16, cube_edges 1048576 | 0.97-1.05 | 0.9997 | HIT |
| H13 | bh_warp_refit K16 / bh_warp, cube_edges 1M | 0.55-0.85 | 0.7512 | HIT |
| H14-plummer-1048576 | mean topology lag (launches), plummer 1048576 | 2-6 | 6.5 | MISS |
| H14-cube_edges-1048576 | mean topology lag (launches), cube_edges 1048576 | 12-50 | 98 | MISS |
| H14-plummer-65536 | mean topology lag (launches), plummer 65536 | 2-8 | 154.2 | MISS |
| H15-1048576 | hyb_phase --hyb-sync / bh_warp, Plummer 1048576 | 2.0-5.0 | 2.743 | HIT |
| H15-65536 | hyb_phase --hyb-sync / bh_warp, Plummer 65536 | 1.5-4.0 | 2.98 | HIT |
| H16 | no hybrid beats the best GPU-only config of its cell by >5%; every hyb_phase gain over bh_warp reproduced within 5% by bh_warp_refit | holds | best hybrid/GPU-best 0.9869 (hyb_ap_dyn in H4_262144 vs gpu_tuned_b128_wall); refit/hyb_phase max 1.0003 | HIT |
| T1-1048576 | D2H effective bandwidth at 1048576, 16 B/body (GB/s) | 18-27 | 25.98 | HIT |
| T1-262144 | D2H effective bandwidth at 262144, 16 B/body (GB/s) | 18-27 | 23.95 | HIT |
| T2-1048576 | H2D effective bandwidth at 1048576, 16 B/body (GB/s) | 18-27 | 22.97 | HIT |
| T2-262144 | H2D effective bandwidth at 262144, 16 B/body (GB/s) | 18-27 | 18.53 | HIT |
| T3 | D2H effective bandwidth at 65536 (GB/s) | 6-22 | 21.1 | HIT |
| L1 | LG1 and LG2, 4 runs | all PASS | 4/4 PASS | HIT |
| L2 | stale/fresh rms ratio, 4 runs | 0.97-1.05 | 0.9835 0.9676 1.0036 0.9909 | MISS |
| L3 | stale/fresh interactions per body, 4 runs | 1.000-1.030 | 1.0108 1.0411 1.0399 1.1021 | MISS |

## Hybrid vs GPU-resident trees (wall clock, ms per step, mean of 2 pass medians)

| cell | bh_warp | bh_sorted (leaf 32 b256) | hyb_bh dyn | hyb_bh f / last f | hyb_phase | lag | refit K4 | refit K16 | hyb_phase sync | hyb_bh/bh_warp | hyb_phase/bh_warp | hyb_phase/best GPU-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H5_cube_edges_1048576_th0.5 | 0.981 | 0.991 | 1.023 | 1.000 / 1.000 | 0.736 | 98.000 | 0.738 | 0.737 | - | 1.043 | 0.751 | 1.000 |
| (same) hyb_bh f 0.95 | | | 3.511 | xfer 2.259 ms, CPU part 0.477 ms | | | | | | 3.580 | | |
| H5_cube_edges_262144_th0.5 | 0.350 | 0.379 | 0.445 | 1.000 / 1.000 | 0.180 | 881.500 | 0.177 | 0.177 | - | 1.270 | 0.513 | 1.014 |
| H5_plummer_1048576_th0.3 | 19.890 | 171.822 | 19.720 | 0.945 / 0.944 | - | - | - | - | - | 0.991 | - | - |
| H5_plummer_1048576_th0.5 | 6.639 | 35.068 | 6.654 | 0.963 / 0.963 | 6.367 | 6.500 | 6.380 | 6.351 | 18.210 | 1.002 | 0.959 | 1.003 |
| H5_plummer_262144_th0.5 | 1.856 | 8.148 | 1.975 | 0.960 / 0.958 | 1.681 | 18.665 | 1.691 | 1.681 | - | 1.064 | 0.906 | 1.000 |
| H5_plummer_65536_th0.5 | 0.787 | 2.115 | 1.002 | 1.000 / 1.000 | 0.624 | 154.250 | 0.627 | 0.617 | 2.346 | 1.272 | 0.793 | 1.013 |

## Hybrid all-pairs (wall clock, ms)

| N | tuned b128 | multi b256 | hyb_allpairs dyn | share (last) | GPU part | CPU part | transfer | hyb/GPU best |
|---|---|---|---|---|---|---|---|---|
| 16384 | 0.2665 | 0.2404 | 0.2980 | 1.0000 (1.0000) | 0.2684 | 0.0000 | 0.0000 | 1.2396 |
| 65536 | 3.4670 | 3.4192 | 3.7032 | 0.9722 (0.9727) | 3.4245 | 3.4693 | 0.0708 | 1.0831 |
| 131072 | 12.9049 | 13.2300 | 13.2138 | 0.9728 (0.9736) | 12.6929 | 12.8338 | 0.1197 | 1.0239 |
| 262144 | 51.2222 | 52.4724 | 51.1123 | 0.9755 (0.9736) | 50.1502 | 49.8572 | 0.2787 | 0.9979 |
| 262144 (H4) | 51.2981 | | dyn 50.6236 / static 51.6320 | static f 0.9727 | | | | |

## P/E cores: cpu_allpairs 65536 (IV 2, tile 2048), compute only

| set | sched | workers | G int/s | busy max/mean | items on P | mean busy P / E (ms) |
|---|---|---|---|---|---|---|
| all | static | | 21.42 | 1.557 | 60.0% | 83.60 / 197.12 |
| all | dynamic | | 37.79 | 1.023 | 77.8% | 110.60 / 111.98 |
| all | weighted | | 37.79 | 1.048 | 78.2% | 109.98 / 109.91 |
| p | static | | 31.10 | 1.010 | 100.0% | 136.52 / 0.00 |
| p | dynamic | | 30.97 | 1.015 | 100.0% | 136.48 / 0.00 |
| p | weighted | | 31.24 | 1.006 | 100.0% | 136.57 / 0.00 |
| p1 | static | | 30.89 | 1.018 | 100.0% | 137.40 / 0.00 |
| p1 | dynamic | | 30.95 | 1.011 | 100.0% | 137.21 / 0.00 |
| p1 | weighted | | 31.18 | 1.007 | 100.0% | 137.28 / 0.00 |
| e | static | | 8.59 | 1.018 | 0.0% | 0.00 / 493.56 |
| e | dynamic | | 8.58 | 1.016 | 0.0% | 0.00 / 492.45 |
| e | weighted | | 8.42 | 1.023 | 0.0% | 0.00 / 495.81 |
| cores | static | | 14.91 | 1.538 | 42.8% | 60.50 / 281.46 |
| cores | dynamic | | 37.48 | 1.031 | 77.2% | 109.00 / 113.06 |
| cores | weighted | | 38.34 | 1.018 | 77.9% | 110.04 / 110.07 |

## P/E cores: cpu_bh Plummer 1M (G 32, leaf 16)

| set | sched | build ms | traverse ms | busy max/mean | items on P |
|---|---|---|---|---|---|
| all | static | 11.664 | 200.484 | 1.532 | 60.0% |
| all | dynamic | 10.328 | 121.953 | 1.023 | 65.6% |
| all | weighted | 10.812 | 123.307 | 1.033 | 66.1% |
| p | static | 14.682 | 164.290 | 1.062 | 100.0% |
| p | dynamic | 14.932 | 155.944 | 1.013 | 100.0% |
| p | weighted | 14.741 | 156.350 | 1.005 | 100.0% |
| e | static | 23.969 | 465.213 | 1.030 | 0.0% |
| e | dynamic | 23.817 | 457.132 | 1.012 | 0.0% |
| e | weighted | 23.947 | 454.759 | 1.011 | 0.0% |

## CPU vs GPU all-pairs (ms; GPU event-timed best, CPU compute-only best; wall GPU at N <= 4096 not scored)

| N | GPU best (event) | GPU best (wall) | CPU best | CPU/GPU |
|---|---|---|---|---|
| 64 | 0.00565 | 0.00565 | 0.00280 | 0.50 |
| 128 | 0.00530 | 0.00570 | 0.01140 | 2.15 |
| 256 | 0.00565 | 0.00570 | 0.01175 | 2.08 |
| 512 | 0.00580 | 0.00575 | 0.01220 | 2.10 |
| 1024 | 0.00575 | 0.00580 | 0.03565 | 6.20 |
| 4096 | 0.01960 | 0.02035 | 0.44730 | 22.82 |
| 16384 | 0.23990 | - | 7.00420 | 29.20 |
| 65536 | 3.41705 | - | 113.90085 | 33.33 |
| 262144 | 51.07530 | - | 1815.36750 | 35.54 |

## CPU vs GPU trees (ms; GPU event-timed, CPU compute only)

| cell | bh_warp total | build | traverse | bh_sorted | cpu_bh best total (config) | CPU/GPU |
|---|---|---|---|---|---|---|
| plummer 65536 | 0.8786 | 0.3115 | 0.5739 | 2.0386 | 7.665 (bh_g32_l8) | 8.72 |
| plummer 262144 | 1.9221 | 0.3957 | 1.5862 | 8.0133 | 27.388 (bh_g32_l8) | 14.25 |
| plummer 1048576 | 6.6036 | 0.6098 | 6.0149 | 35.0039 | 120.945 (bh_g64_l8) | 18.32 |
| cube_edges 1048576 | 1.0209 | 0.5768 | 0.4072 | 0.9885 | 15.959 (cpu_bh_g32_l16) | 15.63 |

## CPU crossover (C10, compute only, ms)

| N | cpu_allpairs best | cpu_bh best |
|---|---|---|
| 512 | 0.0162 | 0.1154 |
| 1024 | 0.0388 | 0.1469 |
| 2048 | 0.1285 | 0.2609 |
| 4096 | 0.4514 | 0.4648 |
| 8192 | 1.8138 | 0.8818 |
| 16384 | 7.0204 | 1.7308 |

## PCIe, 16 B/body pinned (cpu_bh G32 leaf 16 host: line)

| N | D2H ms | H2D ms | D2H GB/s | H2D GB/s |
|---|---|---|---|---|
| 65536 | 0.0497 | 0.0701 | 21.10 | 14.96 |
| 262144 | 0.1751 | 0.2264 | 23.95 | 18.53 |
| 1048576 | 0.6458 | 0.7305 | 25.98 | 22.97 |

## Pass spread |p1 - p2| / mean (largest 10)

- S1_128/cpu_ap_1t: 86.0%
- S1_128/gpu_multi_b512_wall: 74.7%
- S1_64/gpu_tuned_b128: 68.1%
- S1_256/cpu_ap_1t: 67.7%
- S1_128/cpu_ap_p1: 56.4%
- S1_256/cpu_ap_p1: 52.2%
- S1_512/cpu_ap_p1: 31.8%
- H5_cube_edges_262144_th0.5/hyb_bh_dyn: 28.4%
- S1_1024/cpu_ap_all: 24.4%
- S5_plummer_65536/gpu_bh_warp: 21.0%

Entries with a missing median, a non-PASS gate or a same-work FAIL: 0
