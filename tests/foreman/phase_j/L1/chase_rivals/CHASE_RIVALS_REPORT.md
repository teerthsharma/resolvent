# Chase (rivals) — final report (received ~23:40 local), claims as filed

Directory SPJ/L1/chase_rivals: test_rivals.py, rivals_impl.py (float64 reimplementations from published equations), rivals_stub.py, red_run.txt, red_run2.txt, green_run.txt, SOURCES.txt. RED against the stub: 0/9, then 0/10 after T3c added, every row NotImplementedError; both RED runs logged on the board before GREEN. GREEN 10/10 (green_run.txt).

Mechanisms confirmed from primary sources:
- M1 DeepSeek V3 MLA (arXiv 2412.19437 §2.1.1; Kimi K2 arXiv 2507.20534): latent KV, decoupled RoPE; softmax over a key set (commutes); no gate. V3 code MIT, weights DeepSeek License v1.0; K2 modified MIT (display clause paraphrased, not verbatim — OPEN).
- M2 DeepSeek V3.2-Exp DSA (github deepseek-ai/DeepSeek-V3.2-Exp inference/model.py, README): fp8 indexer, top-2048, -inf mask on MLA scores; hard top-k on logits; paper equation and cost [U]; MIT.
- M3 NSA (arXiv 2502.11089 v2): o = sum over {cmp, slc, win} of g^c Attn(...), g in [0,1] by MLP+sigmoid; A100 64k vs FA2: fwd 9.0x, bwd 6.0x, decode 11.6x; no official code link.
- M4 Moonshot MoBA (arXiv 2502.13189; MoonshotAI/MoBA): top-k block gate on logits; "up to 6.5x when prefilling 1M"; MIT.
- M5 Kimi Linear KDA (arXiv 2510.26692 v2; MoonshotAI/Kimi-Linear; fla ops/kda): S_t = (I - beta k k^T) Diag(alpha) S_{t-1} + beta k v^T; non-diagonal, non-commuting; per-token eigenvalues in [0,1]; channel-wise decay + sigmoid output gate; NoPE on MLA layers, KDA:MLA 3:1; claims KV cache up to 75% smaller, decode 6.3x vs MLA at 1M, prefill 2.9x; MIT (fla MIT).
- M6 Qwen gated attention (arXiv 2505.06708; qiuzh20/gated_attention): Y' = Y ⊙ sigma(X W_theta) after SDPA; Table 1 (15A2B MoE) PPL baseline 6.026, elementwise G1 5.761 (+201M), headwise 5.792 (+1.6M); first-token attention mass 46.7% -> 4.8%; MIT.
- M7 Qwen3-Next (HF config/README; transformers modeling_qwen3_next.py L433-450, L856-858, L1038-1051): softmax layers attn_output * sigmoid(gate); GDN S <- S e^g; delta = (v - S^T k) beta; S += k delta^T; g = -e^{A_log} softplus(a + dt_bias); non-commuting, eigenvalues in [0,1]; partial RoPE 0.25, theta 1e7; 12 x (3 GDN + 1 gated attention); "10 times inference throughput for context over 32K"; Apache-2.0.
- M8 MiniMax lightning (arXiv 2501.08313, 2506.13585; HF modeling_minimax_m1.py): kv_t = lambda kv_{t-1} + k v^T, lambda = exp(-slope_rate) ALiBi-style, fixed; diagonal, commuting; output gate; M1 Apache-2.0, MiniMax-01 code MIT / weights MiniMax license; RoPE base conflict 10,000 (paper) vs 1e7 (config).
- M9 GLM-4.5 (arXiv 2508.06471): GQA softmax, 96 heads, QK-Norm, partial RoPE 0.5; MIT per README (LICENSE file 404).
- M10 Qwen3 (arXiv 2505.09388): GQA, QK-Norm, RoPE base 1e6; Apache-2.0.

Bound tests (all GREEN after RED against stub):
- T1a (M6): a2 equals Qwen headwise G1 with our bias folded into X, max diff 0.0.
- T1b (M7): a2 differs from shipped elementwise gate at generic weights (2.50); equals it with weights tied per head (4.4e-16) -> a2 strict subset.
- T2a (M7): GDN decay = FoX log-sigmoid gate at A_log = 0 (0.0); else FoX to a per-head power e^{A_log} (0.0).
- T2b (M7): GDN with orthonormal keys, beta 1 = linear attention with FoX mask (4.4e-16); differs from FoX softmax by 2.30.
- T3a (M5, M7): commutator norms KDA 1.19e-2, GDN 6.75e-2, diagonal decay 0.0.
- T3b (M5): det(KDA) = (1-beta) prod alpha to 1.1e-16; largest |det| over 200 draws 0.5276; two beta = 2 reflections compose to a 120 deg rotation.
- T3c (M5, M7): 2,000 shipped-range factors, 0 eigenvalues outside [0,1], max |Im| 1.8e-16; our SU(2) 3-cycle eigenvalues e^{±i pi/3}, U^3 = -I; unnormalised key (|k| 1.3, beta 0.9) gives eigenvalue -0.521 (bound depends on shipped L2 norm).
- T4 (M8): lightning decay = exp(ALiBi) (0.0) = FoX with constant f (1.1e-16).
- T5 (M1-M4, M9, M10): softmax over a key set order-blind under permutation (8e-17).
- T6 (M3): NSA with one branch = a2 (0.0).

Claims filed:
1. a2 (output mass gate) is prior art = Qwen headwise G1 exactly; Qwen3-Next ships a superset. Kill a2 as our arm; reprice as a replication of Qwen G1 built their way, and first read first-token attention mass on arm (a). Seat Foreman.
2. Our FoX win already ships in linear form (GDN gate law; KDA channel-wise, "KDA as position"). Kill any novelty claim for data-dependent forgetting as position; reprice the comparator to a 3:1 KDA/GDN + NoPE-softmax hybrid at matched params.
3. Learned phase: no lab ships it (fixed RoPE, partial or none) — corroborates our loss; retire.
4. Addendum L: bare non-commuting order at linear-attention cost ships (KDA, GDN, T3a). Survives: every shipped per-token factor has spectrum in [0,1]; a unit-circle (SU(2)) per-token spectrum is not in shipped work (T3c). Narrow the claim to "per-token unit-circle spectrum folded around SDPA"; reprice cost against fla KDA/GDN kernel, not SDPA; on the A5 bed build the floor GDN with beta in [0,1] and the rival beta in (0,2) (reflections; two per token make a rotation, T3b).

OPEN (as filed): DSA equation/cost [U]; T3b/T3c bound only full-space products and single-token factors (long products of near-projections may approximate orthogonal maps on a subspace; KDA ShortConv makes k depend on 4 tokens; no formal single-layer impossibility bound); why a2 lost unbound (needs weights); cost vs fla KDA unmeasured; unread prior art [U]: Gated DeltaNet (arXiv 2412.06464), negative-eigenvalue DeltaNet (Grazzi et al.), DeltaProduct — Addendum L cannot claim novelty over them until fetched; K2 license clause paraphrased; GLM LICENSE 404; MiniMax RoPE base conflict; NSA weights not stated released; QK-Clip vs dtype wall unbound.
