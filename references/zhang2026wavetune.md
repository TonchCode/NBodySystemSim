# zhang2026wavetune

**Citation:** K. Zhang, C. Ding, S. Qian, L. Wang, J. Cao, G. Xue, C. Huang, G. Yang, L. Zhang. "WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning." arXiv:2604.10187, v1 11 Apr 2026. Preprint, no venue given.
**Link:** https://arxiv.org/abs/2604.10187
**Verified on:** 2026-10-17 (arXiv abstract + HTML). Affiliations: Zhang, Ding, Qian, Cao and Xue at "Shanghai Jiao Tong University, Shanghai, China"; Wang, Huang, Yang and L. Zhang at "Alibaba Group, Hangzhou, China".
**Lead-group country:** China (SJTU).

## Summary
1. Kernel autotuning driven by an analytical latency model.
2. Finds that neither the step model (time ∝ ceil(blocks / SMs)) nor a linear model fits the measured latency.
3. Two reasons. The tail-wave penalty is bounded by one wave's duration, so its relative weight fades with problem size. Blocks in a wave do not finish together (latency variance, greedy dispatch), which smooths the steps. A discrete-event simulation backs this.
4. The model is T(G, L) ≈ αGL + βG + γL + δ (G = grid size, L = loop count), fitted per wave range.
5. Evaluated on A100, H20, B200, MI308X and MI355X (no Ada). Up to 1.83x kernel speedup.

## What it lets us claim
The pure step model M1 in 03-design §0.2/§3.3 is an idealization, and its tail-effect predictions should be expected to overshoot. That is what we measured: b128 vs b256 at N = 65536 was +8%, against a predicted +20% (03-design §6).

## Design decision / R3 technique supported
R3 #2 (block-size sweep and tail model). It supports replacing M1 with a smoothed, fitted model and measuring per-block start/end times (critic #1). Strength: **direct** for the wave model, which is about block scheduling on NVIDIA SMs and does not depend on the kernel body. The kernels tested are GEMM-like.

## Quote
"neither the rigid step function nor the simplified linear view adequately captures" (Zhang et al., 2026, arXiv HTML)
