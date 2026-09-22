# Chase, N2: R-DIAG′ on the trained (f) checkpoint

**Verdict: PAST A POLE.** At γ = 0.99, the Mode B equilibrium read of the trained family is not well-posed.

- γ* = 1 / max_i W_ii falls below 0.99 in **1,432 of 1,536 (window, layer, head) pairs (93.2%)**.
- Every layer fails: 424/512, 496/512 and 512/512.
- The worst self-weight is W_ii = 2.0677, so the pole sits at γ* = 0.4836.
- Re(W), the readout the head actually uses, gives the same numbers exactly.
- No token is an absorber, and there is no unit diagonal anywhere, position 0 included.

## Setup

**Checkpoint:** `SP/d45_ckpt_f_ss0_pair/model.pt`. It was rebuilt with `abstention_deciles.build_arm("f")`, using the hard-concrete gate and the repaired init.

**Data:** `abstention_deciles.eval_batches()`, which is 8 batches × 8 windows × 512 bytes, so 64 windows. Everything ran on CPU.

**How W was formed:** `ceq/arm_smprime.py::operator(q, k, u, θ, beta=at.beta, qk=at.qk, g=at.g)`, the same call `CEQAttention._smprime` makes. It ran in float32, then again in float64 from the same float32 inputs.

**Fidelity checks (all GREEN):**
- **The rebuilt model is the trained model.** Held-out NLL is 1.0430563 against the recorded 1.0430563 (CUDA).
- **W is the operator the model runs.** o_proj((W·V).real) matches the model's own self_attn output with max |diff| = 0.0.
- **β is the learned value.** It equals the state dict: 1.0605749 / 0.9300133 / 0.8995796.

**Reproducibility:** two runs gave identical JSON except the wall-clock field.

## Per-layer table

- Each layer has 262,144 diagonal entries, 512 (window, head) pairs and 32,768 gate positions.
- γ* is taken per (window, head) as 1 / max_i W_ii.

| layer | β (learned) | qk | exact gate zeros (min m) | unit diagonals: f32 `==1.0` / f64 within 1e−6 | max W_ii (window, head, pos) | γ* min / median | pairs with γ* < 0.99 | same, excluding pos 0 | entries W_ii > 1/0.99 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0605749 | 1.0872 | 0 (0.0472) | 0 / 0 | 1.2068455 (w0, h4, pos 0) | 0.8286 / 0.9541 | **424/512 = 82.8%** | 7/512 = 1.4% | 429 (0.16%) |
| 1 | 0.9300133 | 1.3335 | 0 (0.0967) | 0 / 0 | 1.4142937 (w30, h6, pos 134) | 0.7071 / 0.8439 | **496/512 = 96.9%** | 495/512 = 96.7% | 6,260 (2.39%) |
| 2 | 0.8995796 | 1.3895 | 0 (0.0925) | 0 / 0 | 2.0676823 (w22, h1, pos 242) | 0.4836 / 0.6824 | **512/512 = 100%** | 512/512 = 100% | 14,040 (5.36%) |
| all | | | 0 | 0 / 0 | 2.0677 | 0.4836 | **1,432/1,536 = 93.2%** | 1,014/1,536 = 66.0% | |

**Float32 against float64:** the counts are identical, and max W_ii agrees to within 2.2e−7 (layer 2: 2.0676822662 in f32). The f64 maxima are 1.2068454693 / 1.4142936757 / 2.0676820532.

**Re(W):**
- In every layer, diag(Re W) is bitwise equal to diag(W).real, and the imaginary part of every diagonal entry is exactly 0.
- Re(W) is lower triangular, so its spectrum is the same diagonal.
- Every Re(W) column therefore equals the W column: 424 / 496 / 512 pairs, γ* min 0.8286 / 0.7071 / 0.4836.

**The divergence, directly:**
- On the worst head, the Neumann iterate x ← 0.99·W·x grows by 2.0470 per step (float64, last 50 of 400 steps).
- That equals 0.99 × 2.0677 = ρ(γW). The series behind the Mode B read diverges.

**Mechanism:**
- **Layers 1–2 (β < 1):** the failures sit at interior self-peaked tokens.
  - The worst case has s_ii = 7.238 and self-weight p_ii = e^{s_ii}/Z_i = 0.99954.
  - So W_ii = p_ii^β e^{(1−β)s_ii} ≈ e^{0.727} = 2.068.
- **Layer 0 (β > 1):** the failures are 99.2% at position 0, where W_00 = e^{(1−β)s_00}.
  - The worst case has s_00 = −3.104, which gives 1.2068.
  - Excluding position 0, layer 0 is almost clean (7/512).

**Second implementation:** a nurse wrote `nurse1/wii_check.py` with its own hard-concrete, its own path product and float64, and no arm_smprime operator code. It reproduced:
- W_ii = 2.0676820534 at (layer 2, window 22, head 1, pos 242), a difference of 1.9e−10;
- W_ii = 1.2068454693 at (layer 0, window 0, head 4, pos 0), a difference of 1.6e−11.

## Absorbers

**None, other than position 0's trivially empty row.**
- Rows i > 0 whose off-diagonal entries are all exactly zero: 0 in every layer, in f32, f64 and Re(W).
- Exact gate zeros: 0 of 32,768 per layer, which agrees with Wilson.
- Position 0 is not a unit diagonal at the trained β, since W_00 = e^{(1−β)s_00} ≠ 1.
- So the trained operator has no closed class at all. D1's count of closed classes by unit diagonals returns 0 here, and D4 ("only W₀₀ = 1") does not hold on this checkpoint.

## Verdict against the registered bar

**The bar (registered before measuring, in the test docstring at 00:38):**
- WELL-POSED if γ* ≥ 0.99 in every (window, layer, head).
- PAST A POLE if γ* < 0.99 anywhere.

**Result: PAST A POLE, in 93.2% of pairs, and in 66.0% even with position 0 excluded.** The result binds.

## Tests

Test file: `C:/Users/seal/AppData/Local/Temp/claude/C--Users-seal-Desktop-New-folder--32-/870edeb1-6409-4414-98e7-00d0537b75dd/scratchpad/phase_j/N2/chase/test_rdiag_prime.py`, sha256 `9dab02a78a5c1075…`. The read itself is `rdiag_prime.py`, sha256 `9e4f1a4ead476b78…`.

**Order of events on the board:**
1. The stub RED came first: `RDIAG_IMPL=rdiag_stub`, 7/7 tests `NotImplementedError`, logged in `stub_red.log`.
2. Run 1 was all RED, from my own CPU guard (`assert device == "cpu"`). `CUDA_VISIBLE_DEVICES=""` did not hide the GPU here. The fix pins `D.DEVICE = "cpu"`.
3. Run 2 is final, logged in `run2.log`.

**Final status (run 2):**

| test | status |
|---|---|
| `test_rebuild_fidelity` | GREEN |
| `test_beta_is_learned_beta` | GREEN |
| `test_operator_is_model_path` | GREEN |
| `test_diag_real_and_ReW_diag_equal` | GREEN |
| `test_no_absorber_but_position_0` | GREEN |
| `test_bar_well_posed_gamma_099` | **still RED** |
| `test_bar_well_posed_gamma_099_ReW` | **still RED** |

**The exact RED line:**
```
RED   test_bar_well_posed_gamma_099 AssertionError: {0: (424, 512, 0.8286064635707269), 1: (496, 512, 0.707066741234268), 2: (512, 512, 0.48363330107807584)}
```

## Replacement route: the kill of "the Mode B read at γ = 0.99 is well-posed on the trained family"

### 1. Reprice (primary)

**The object:** read Mode B on W̃ = W / max_i W_ii per (window, head). This is the same as reading W at γ_eff = 0.99·γ*.
- The per-pair γ* values are already in `rdiag_prime.json` (`gstar_all`).
- Computing them is an O(n) diagonal read.

**The measurement:**
- Compute r = (1−γ)(I − γW̃)⁻¹V by blocked forward substitution at γ = 0.99, on the same 64 windows.
- Report σ₂/σ₁ of r.
- Report the share of r carried by the argmax-diagonal token.

**Seat:** Cameron, who already owns the forward-substitution read (R-COST-N, 1.76–1.80× SDPA).

**Why it is sharper:**
- It is well-posed by construction in 1,536/1,536 pairs.
- It leaves the trained weights and β untouched; the only change is one scalar per head.
- It names the equilibrium's attractor: the most self-peaked token, which is position 0 in only 0.8–2.7% of pairs in layers 1–2. That makes the "equilibrium is a sink" claim testable on the trained family instead of assumed.

### 2. Reroute (priced now): pin β = 1

**The measurement, evaluation only with the same weights** (`test_beta_pin.py`, stub RED first):
- **Well-posed:** γ* min is 1.0000. This is `test_beta1_gstar_at_least_1`, GREEN.
- **Not free:** held-out NLL rises from 1.0430563 to 1.0652995, **+0.0222 nats, 9.6% of C_win**. The registered FREE bar was ≤ 0.01, so `test_beta_pin_free` is RED.
- **One run voided:** the first GREEN of `test_beta1_gstar_at_least_1`, at 00:44, came from a buggy min (1/min_pairs of the maximum). It is filed VOID on the board, and the re-run is the one above.

**Next row:**
- **The object:** retrain (f) with β frozen at 1, at the grid shape, split seeds 0–2, CRN-paired with (f).
- **Seat:** Foreman, on GPU.
- **Bar:** if the loss exceeds (f)'s by more than 0.01 nats at all three seeds, β ≠ 1 is load-bearing. Mode B must then use route 1, and β = 1 is retired for any head whose equilibrium is read.

**Why it is sharper:** it turns "the diagonal theory needs β = 1" into a priced trade. Evaluation already bounds the cost at 0.022 nats.

## Limits

- **One checkpoint.** Only `d45_ckpt_f_ss0_pair` has weights; `d45_ckpt_f_ss1..4` hold only `run_record.json`. "The trained family" here means one split seed of arm (f).
- **The GPU was touched briefly.** In run 1, `build_arm` moved the 725k-parameter model to CUDA before my guard asserted. Nothing was computed there, and the process exited.
- **The unit-diagonal counts are 0/0**, so the float-rounding overcount found in N1 (fp32 manufacturing unit diagonals) never arises on this checkpoint.

## OPEN

- The pole result stands on one seed. It extends to the other seeds only if their weights are saved (`save_model`), which requires a retrain.
- Layer 0's failures are almost entirely position 0, where β > 1 and s_00 < 0. A Mode B read that excludes the BOS row still fails the bar, in 7/512 pairs of layer 0 and 495/512 and 512/512 in layers 1–2.
- Route 1's residue rank on W̃ is unmeasured.
- The cost of pinning β = 1 in a retrain is unmeasured. The +0.0222 nats is an evaluation-only upper bound.
- **Default next move:** Cameron runs route 1 (W̃ forward substitution, σ₂/σ₁ and argmax share) on CPU against these 64 windows. Foreman's β-frozen retrain waits until R-POS releases the GPU.
