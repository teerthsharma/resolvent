# CHASE — Phase K, it.K0

The resolvent layer trains exactly, but only after three of its parts were replaced. As it was reused, Cameron's fs5 forward fails the 1e-5 bar under SSMax. The kernel-lse adjoint misses the 1e-4 gradient bar by about 10x. The first R-RANGE instrument returns negative masses on heads with content. The replacements pass every bar. The graphed layer costs **2.27x** one SDPA fwd+bwd at S 4096 and **2.26x** at S 8192. γ·ρ(W) < 1 holds for every logit scale, but it never tests anything. The quantity that breaks is γ·‖W‖∞ in bf16.

Directory: `SP/phase_k/K0/chase/`, where SP = `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad`.
Files:
- `resolvent.py`: the layer.
- `rrange.py`: the instrument.
- `test_chase_k0.py`: the bars, registered in its docstring before any measurement.
- `cost.py`, `cost_sparse.py`: cost measurements.
- `breaks.py`, `breaks_b2tail.py`: the "what breaks" measurements.
- Logs: `red_stub{,2..6}.log`, `run1.log`, `run_final.log`.
- Rows: `cost_rows.jsonl`, `cost_rows_eager_run0.jsonl`, `cost_sparse_rows.jsonl`, `breaks_rows.jsonl`.
- Diagnostics (not bars): `diag_*.py`.

Setup: torch 2.14.0+cu126, RTX 4060 Laptop, fp32, TF32 off. B1 H8 D64 and γ = 0.99 unless stated. Logits are z_ij = s_i·q_i·k_j/√D with s_i = a·ln(i+1) + b, applied by pre-scaling q; a and b are learned. Nurses dispatched: 0.

## The layer

- **Math.** x = (1−γ)(I−γW)⁻¹V at β = 1.
  - Adjoint: u = (I−γW)⁻ᵀ g, which is an upper-triangular solve.
  - Then dV = (1−γ)u and dW = γ·u·xᵀ, followed by the ordinary softmax (or sparsemax) Jacobian. Nothing is unrolled.
- **Training path `fusedcg`.**
  - Forward: `fs5c`, which is Cameron's fs5 with one change. The diagonal pivot is computed from the complementary mass, (1−γ) + γ·Σ_{j≠i}W_ij.
  - Backward: `backward_blocked`, one reverse pass over row blocks of c = 256.
    - Each block's rows of W are materialized with a max-shifted softmax.
    - The transposed diagonal block is solved with an upper trsm, using the same complementary pivot.
    - A right-looking push accumulates acc += W_{J,<lo}ᵀ u_J.
    - dq and dk are accumulated in the same pass.
  - Both forward and backward are captured as CUDA graphs with static buffers: inputs are copied in and outputs cloned out.
- **Sparsemax variant.** Path `densec`: W is materialized, the sparsemax is sort-based, the solve is a dense trsm, and the same complementary pivot is used.

## Bars

The RED path for every bar was a stub run first (`resolvent.py`/`rrange.py` raising `NotImplementedError`, or `CHASE_STUB=1`), posted to the board before the bar was measured.

| bar (chase.k0.*) | measured value | RED line | status |
|---|---|---|---|
| fwd_softmax_S1024_vs_f64: Cameron's fs5 reused unmodified, SSMax a=1 b=0, ≤ 1e-5 | 3.68e-5 | stub: `RED fwd_softmax_S1024_vs_f64: "EXC NotImplementedError: stub"`; measured: `RED fwd_softmax_S1024_vs_f64: {"rel_err": 3.6769499861787216e-05}` | **still RED** (kept as registered) |
| fwd_softmax_S1024_vs_f64_fs5c: complementary pivot, seeds 0 and 2, ≤ 1e-5 | 3.04e-6 / 4.08e-6 | `RED fwd_softmax_S1024_vs_f64_fs5c: "EXC NotImplementedError: stub"` | GREEN |
| fwd_sparsemax_S1024_vs_f64: dense fp32, naive pivot, ≤ 1e-5 | 1.78e-5 | stub, then measured `RED fwd_sparsemax_S1024_vs_f64: {"rel_err": 1.78001308427978e-05}` | **still RED** |
| fwd_sparsemax_S1024_vs_f64_pivot: densec, seeds 0 and 2, ≤ 1e-5 | 6.20e-6 / 5.80e-6 | `RED fwd_sparsemax_S1024_vs_f64_pivot: "EXC NotImplementedError: stub"` | GREEN |
| gradcheck_softmax_S64_f64: dense adjoint; inputs q, k, v, a, b | passes | `RED gradcheck_softmax_S64_f64: "EXC NotImplementedError: stub"` | GREEN (mutant with the untransposed solve → RED) |
| gradcheck_sparsemax_S64_f64 | passes | stub RED | GREEN |
| gradcheck_blocked_adjoint_S64_f64: the training backward, c = 16, 4 blocks | passes | `RED gradcheck_blocked_adjoint_S64_f64: "EXC NotImplementedError: stub"` | GREEN (mutants "push removed" and "dv×1.01" → RED; dv×1.0001 passes, which is inside rtol 1e-4) |
| fused_bwd_S1024_vs_f64: kernel lse + efficient-attention backward, ≤ 1e-4 | dq 1.14e-3, dk 1.05e-3, dv 1.85e-5 | stub, then measured `RED fused_bwd_S1024_vs_f64: {"dq": 0.0011404845058584565, ...}` | **still RED** |
| fusedc_bwd_S1024_vs_f64: fs5c + backward_blocked, ≤ 1e-4 | dq 2.95e-6, dk 9.74e-6, dv 2.15e-7 | stub; then measured with v1 (exp(z − lse)) `RED ... {"dq": 8.84e-05, "dk": 0.000313, "dv": 5.58e-07}`; v2 uses max-shifted softmax rows | GREEN |
| fusedcg_matches_fusedc_S1024: graphed vs eager, two different inputs, ≤ 1e-6 | 0.0 on all 8 tensors (bitwise) | stub RED | GREEN |
| y1_rebuild: \|κ − arccosh\| ≤ 5e-7 and 6-digit match to §9 | ≤ 8.6e-16 | stub RED | GREEN |
| y2_closed_form: \|Δm\| ≤ 1e-4 and relative ≤ 1e-4 | abs ≤ 1.8e-16, rel ≤ 1.9e-13 | stub RED | GREEN |
| y2_vs_pinned: vs `yukawa.out` 5-digit values, ≤ 5e-6 | ≤ 4.9e-6 (the print's rounding) | stub RED | GREEN |
| y2_tail_instrument: the replacement instrument | abs ≤ 1.8e-16, rel ≤ 1.9e-13 | `RED y2_tail_instrument: "EXC NotImplementedError: stub"` | GREEN |
| gamma_rho_lt_1: max γ·W_ii < 1 over a ∈ {0..8}, b ∈ {0, 2}, ALiBi on/off, S 4096 | 0.99 = γ·W_00 | stub RED | GREEN, but vacuous (see below) |
| cost_report: reported, not scored (R-COST is Cameron's at K4) | 2.27x at S 4096, 2.26x at S 8192 | `RED cost_report: {}` | GREEN (rows present) |

Final run: `run_final.log`, 16 bars, 13 GREEN and 3 RED; the 3 RED are the originals kept as registered. The process exits 1.

### Cost detail

Medians in ms, alternating arms, CUDA events. Fwd+bwd is compared with one SDPA fwd+bwd of the same shape.

| S | SDPA f+b | resolvent f+b (graphed) | ratio | eager ratio | fwd ratio | bwd-only ms (res / SDPA) | fwd rel err vs f64 | peak MiB |
|---|---|---|---|---|---|---|---|---|
| 4096 | 17.94 | 40.75 | **2.27** | 4.95 | 2.00 | 31.6 / 13.4 | 1.02e-5 | 763 |
| 8192 | 68.35 | 154.18 | **2.26** | 3.60 | 1.86 | 123.2 / 51.7 | 1.58e-5 | 2533 |

- The eager path is launch-bound. For the forward alone, CPU launch time is 24.8 ms and GPU time is 24.8 ms (`diag_cost.py`). A first eager run gave 5.44x (`cost_rows_eager_run0.jsonl`).
- Sparsemax (`densec`) at S 4096 costs **9.41x** with a 3.8 GiB peak. S 8192 was not run: the dense path needs about 15 GiB.

## Y1 / Y2: contract §9, the pinned instance and this rebuild

| id | contract §9 | pinned `yukawa.out` (sha feb03137…) | Chase rebuild (independent route) |
|---|---|---|---|
| Y1 m=0.1 | .099958 | 0.099958 | 0.09995838 (banded solve; \|Δ arccosh\| 8.6e-16) |
| Y1 m=0.5 | .494933 | 0.494933 | 0.49493292 (1.1e-16) |
| Y1 m=1.0 | .962424 | 0.962424 | 0.96242365 (2.2e-16) |
| Y2 λ=.5, γ=.9 | .06285, range 16 | 0.06285, 16 | 0.0628547235, range 15.91 |
| Y2 λ=.5, γ=.99 | .00647, 155 | 0.00647, 155 | 0.0064662613, 154.65 |
| Y2 λ=.5, γ=.999 | .00065, 1,542 | 0.00065, 1,542 | 0.0006485109, 1,541.99 |
| Y2 λ=1, γ=.9 | .15857, 6 | 0.15857, 6 | 0.1585650787, 6.31 |
| Y2 λ=1, γ=.99 | .01704, 59 | 0.01704, 59 | 0.0170368632, 58.70 |
| Y2 λ=1, γ=.999 | .00172, 582 | 0.00172, 582 | 0.0017168073, 582.48 |

The rebuild differs from the pinned script in method:
- **Y1** uses a banded solve.
- **Y2** averages the log-kernel over ten rows (2000..2900) instead of using one row. The tail-mass instrument gives the same digits.
- The closed form is matched to at most 1.9e-13 relative.

## What breaks

1. **Near-unit self-weights break the fused forward.**
   - Cameron's fs5 computes the pivot 1 − γ·exp(s_ii − l_tot). l_tot is a merged log-normalizer at |z| ≈ 37, where fp32's ulp is about 4e-6. That error is divided by a pivot near 1 − γ.
   - The error lands in the last block, where ln(i+1) is largest: 1.82e-4 there against 2e-6 to 6e-6 in blocks 0–2.
   - Replacing the pushes with exact float64 ones leaves it at 1.8165e-4, so the fused kernel is not the cause.
   - The same mechanism in the adjoint gives dq and dk errors near 1e-3.
2. **The fp32 floor grows with ln n under SSMax.** The fixed forward's error is 3.0e-6 at S 1024, 1.02e-5 at S 4096 and 1.58e-5 at S 8192. The 1e-5 standard holds at S 1024 only.
3. **What the read converges to on an ALiBi head.**
   - No trained ALiBi weights exist. `tests/foreman/phase_j/N2/foreman/ckpt_a_alibi_ss{0,1,2}/` hold `run_record.json` only.
   - On the ALiBi prior heads (slopes 2⁻ʰ, h = 1..8, n = 4096), the read's BOS weight follows **w0(i) ≈ e^{−m·i}**, with m = ln(γ + (1−γ)e^λ). At γ = .999, row 4095: measured 0.0701 vs e^{−mi} 0.0703; 0.312 vs 0.313; 0.577 vs 0.580.
   - As γ → 1, every row reads V_0. That is the D4 sink.
   - At γ = .999, BOS takes 51% to 99% of row 1023's read in all 8 heads.
   - At γ = .99, row 4095, it takes 49.5%, 68% and 79% in the three slowest heads.
   - With random content added (SSMax a = 1), the BOS weight at γ = .999, row 4095, runs from 2.8e-8 (slope 1/2) to 0.98 (slope 1/256).
   - So a single fixed γ makes the slow half of an ALiBi-initialised layer a BOS reader.
4. **The first R-RANGE instrument is not a mass on content heads.** The log-kernel fit, which passes Y2 to 2e-13, returns **m < 0 on 7 of 8** ALiBi+content heads at both γ. A single translation-invariant exponential does not describe those kernels.
5. **γ·ρ(W) < 1 never tests anything at β = 1.**
   - ρ(W) = max W_ii = W_00 = 1 exactly, so γ·ρ = γ for every a and b, in fp32 and in bf16. The law cannot break it, and the bar is GREEN for that reason alone.
   - The quantity that fails is γ·‖W‖∞. With W rounded to bf16 at γ = .999 and SSMax a = 1:
     - 19.3% (no ALiBi) and 21.2% (ALiBi) of rows have γ·rowsum ≥ 1;
     - the read's gain on V = 1, whose exact value is 1, reaches **2.53** and **1.91**.
   - At γ = .99 the gain is 1.15 and 1.09. At a = 0 it is 1.009.
6. **"Only one unit self-weight" fails in fp32 and never holds for sparsemax.**
   - Exact softmax has W_ii < 1 for i > 0. fp32 softmax at SSMax a = 2, 4, 8 gives 1, 5 and 17 of 32,768 rows with W_ii == 1.0.
   - Sparsemax gives 5–12 of 8,192 rows with W_ii == 1 at a = 0.5–8, and 38%–94% of all rows are one-hot.
   - Each of these rows is an absorber, where x_i = v_i exactly.

## Kills and replacement routes

| killed | replacement | state |
|---|---|---|
| fs5 as the training forward under SSMax | fs5c, with the complementary-mass pivot | GREEN at S 1024 |
| Adjoint from the kernel lse plus efficient-attention backward | backward_blocked: materialized max-shifted rows, complementary pivot, dq/dk in the same pass, graphed | GREEN, gradchecked, 2.27x/2.26x |
| Dense fp32 sparsemax with the naive pivot | densec | GREEN |
| Log-kernel mass fit on content heads | Tail mass T(r) = mean_i Σ_{j≤i−r}(1−γ)G(i,j) (non-increasing by construction), m = −d ln T/dr | Y2 GREEN; m > 0 on 16/16 content heads |
| γ·ρ(W) < 1 as the safety check | Use γ·max rowsum(W) < 1, and read gain on V = 1 within 1 ± 1e-5, as the K1 bar. Keep the resolvent layer's W and solve in fp32 (exempt from autocast). | route, untested |
| One fixed γ on slow ALiBi heads (BOS reader) | Per-head γ_h from Y2 inverted, so the dressed range equals the target depth: γ_h = (e^λ − e^{1/D})/(e^λ − 1). Or remove the BOS absorber term before the read. | route, untested |
| The premise "only position 0 is a unit self-weight" | Count absorbers per head on each checkpoint (the D1 count; R-BIND) as an L-TRAINED premise check | route |

## OPEN

- The fp32 forward error exceeds 1e-5 at S 4096 and 8192 under SSMax a = 1. No bar was set at those lengths. A fix would need logits centred per row inside the fused pushes, which has not been tried.
- Sparsemax needs a blocked, row-streamed path. The dense path is 9.41x at S 4096 and does not fit at S 8192.
- The layer has not trained on the chain bed yet; that is R-DEPTH at K1. There are no trained (f_R) or ALiBi checkpoints to measure. The trained-head version of item 3 of "What breaks", and R-RANGE on real heads, wait for Foreman's R0 checkpoints (with `save_model`).
- The bf16 route has not been measured on the fused path. Items 5 and 6 of "What breaks" are measured on materialized W only.
- Cost is at B1 H8 D64. The contract's R-COST at 16k belongs to Cameron at K4.
