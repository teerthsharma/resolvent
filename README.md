# ceq — consequence-equilibrium attention

An attention operator whose output is a **signed, strictly causal path sum** rather than a
one-hop similarity lookup:

```
out = v + A v + A² v + … + A^K v          A strictly lower triangular, signed
```

Every number below carries the test that produces it. Nothing here is claimed at a higher
confidence than its evidence supports.

---

## Limits, first

**THE OPERATOR THE FLAT-INFLUENCE NUMBERS DESCRIBE SHIPS NOWHERE.** `tgate` — the
denominator-free matrix `A_ij = g_i tanh(qhat_i . khat_j / tau)` that every context-stability
number is measured on — lives in `ceq/bench.py` and is not what this repository releases. The
Hub config ships `sgate` and `ceq/attention.py` ships the row-L1 form, and **both carry a
denominator that sums over context**, which `bench.py`'s own docstring names as the measured
cause of the 1/s decay. Pointed at the shipped matrix, the same routed measurement reads
[RUN, n=1024 draws, k=8, c drawn from the pivot set, PROTOCOL SCALING]:

| operator, hop 2 routed through k=8 pivots | s=8 | 32 | 128 | 512 | slope |
|---|---|---|---|---|---|
| `tgate` (bench only) | 0.02466 | 0.02856 | 0.03101 | 0.02881 | **+0.027** |
| **`sgate` (what ships)** | 0.15430 | 0.02344 | 0.00098 | **0.00000** | **−1.826** |

Routing fixes the PATH COUNT; it does not remove a DENOMINATOR, and the routed shipped arm
decays *faster* than its own dense control (−1.826 against −1.151) because restricting hop 2
to k pivots takes paths out of the numerator while the denominator still sums over all s.
**So the flat number is a property of denominator-freedom plus fixed path count, and the
released module has neither.** The measurements are not withdrawn — they are correct about
`tgate` — but nothing in this README about context-stable signed influence should be read as
a statement about the weights on the Hub until `tgate` ships.
(`scale/s2_probe.py`, `tests/cameron/test_s2_ablation.py`, table in `DONE.md`.)


**THE CAPABILITY NUMBER CAME BACK AND THE OPERATOR LOST IT.** COGS generalization, exact
match, 512 items, **3,652,096 parameters in both arms**, identical steps / lr / batch / seed
/ data / eval subsample, one seed:

| arm | COGS-gen | in-distribution (gate) |
|---|---|---|
| softmax control | **0.0293** (15/512) | **0.9258** |
| `sgate` | **0.0000** (0/512) | **0.7734** |
| published from-scratch encoder-decoder, arXiv:2010.05465 | 0.35 ± 0.06 | 0.96 |

One-sided Fisher exact for softmax > `sgate`: **p = 2.7502788939e-05** — recomputed exactly
this round; earlier documents rounded it to 2.8e-05. Not a tie — a loss. And the
deficit is already present **in-distribution**, on the split both arms trained on, so it is
not a story about generalization; it is the same shape as the val-loss gap that widened with
budget. Truncation ceiling at `max_new = 192` is 0.9512, identical for both arms.

SCAN addprim_jump, 3 seeds, 4000 steps, 512 items, 3,183,104 parameters both arms: softmax
median **0.0000** (0.0000 / 0.0000 / 0.0059), `sgate` median **0.0000** (0 / 0 / 0), against a
published vanilla Transformer at **0.034 ± 0.020 SEM** (arXiv:2107.01366 Table 3). At n = 512
a zero carries a 95% upper bound of 0.00583 and the published 0.034 would have shown ~17
solved items, so both arms genuinely miss the published floor — and **two arms at zero
separate nothing from each other.**

`python -m ceq.capability` · `results/capability.json` · `tests/cameron/test_capability_result.py`

**A reference this project cited for four rounds is a different task.** "Edge Transformer
0.874 ± 0.004 against a Universal Transformer control at 0.784" is the **graph-prediction**
reformulation of COGS (arXiv:2112.00578 Table 4, captioned "graph prediction accuracy"), not
the sequence-generation task scored here; the 0.784 carries no ± and is quoted from Ontanón
et al. 2021. The only altered-attention precedent on sequence-generation COGS is Csordás et
al. 2021 at 0.81 ± 0.01 against 0.80 ± 0.00 — **about four points.**

**There is no novel construction here, and the number that said otherwise was an instrument
bug.** Round 3 published "ParaFormer 0 of 2048 draws, at least 84×" as the last standing
novelty claim. The `paraformer` arm never applied its hop coefficients — it ran plain
single-hop softmax, so the 84× was softmax's zero. Corrected, the separation is 5.6× at
`s = 8` (conservative 95% bound 3.94×, peaking at 4.66× at `s = 16`). And the cell the claim
retreated to — *signed, content-dependent, multi-hop* — is occupied by DeltaNet's WY matrix,
which ships in sglang and `flash-linear-attention`, and which **reads at or above this module
on this module's own probe at every context length from 16 up**. The "non-overlapping 1.39× at
`s = 32`" that used to stand here is **withdrawn as a discard-floor artifact**: at `floor = 0`
the two are 0.054688 against 0.050781, a 1.08× overlap. The withdrawal restores nothing —
DeltaNet still reads ≥ this module, and the `sgate` matrix is Signed Dual Attention
(arXiv:2606.04833) Eq. 1–2 whatever the margin is. Full table with Clopper-Pearson intervals
below.

**The distinguishing property DECAYS with context — and every rate in this repository is
floor-sensitive, which no document said until now.** Content-conditional sign was measured at
the probe's default context of **s = 8** and never swept. `ceq/bench.sign_flip_rate` discards
any draw whose gradient falls below an **absolute** `floor` (default `1e-6`), so that setting
is part of every number it has ever produced. Swept, 1024 draws, relative positions held
fixed, at both settings:

| s | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| `sgate`, `floor = 0` | 0.181641 | 0.098633 | 0.050781 | 0.026367 | **0.012695** |
| `sgate`, `floor = 1e-6` — every rate published before this round | 0.174805 | 0.088867 | 0.026367 | 0.011719 | **0.003906** |
| softmax control, either floor | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The floor discards more of the signal the further the context runs — ratio **1.04 / 1.11 /
1.93 / 2.25 / 3.25×** over `s = 8…128` — which is exactly where the decay was being read.
**The decay is real** (0.181641 → 0.012695 is 14.3× over a 16× growth in context) **and the
published exponent −1.389, R² 0.9938 is withdrawn as a `floor = 1e-6` artifact.** No
replacement exponent is published: least squares on the five `floor = 0` rates above gives
**−0.958 (R² 0.9990)**, the audit that forced this correction reports **−1.221 (R² 0.9662)**,
and those two do not agree. Extrapolating the −0.958 line reaches ~1.8e-3 at `s = 1024` and
~9.3e-4 at `s = 2048` — arithmetic on a fitted line, **never measured at `s ≥ 1024` on this
probe**, and no longer "below the resolution of 10,000 draws" as previously published.

**The mechanism is that the property and its decay are the same fact.** A third token `c`
reaches the pair (i,j) only along a path *through* `c`, and the first term of `J = Σ A^k`
containing one is `k = 2` — measured at hops=1: 0.0273, at hops=2: 0.1445. The two-hop term
sums over ~`s` intermediates of which `c` is one, so its share is **1/s by construction**.
Multi-hop is what *creates* the property and multi-hop is what *dilutes* it.

**Round 4 corrects this: the decay is the softmax row normalizer, not the two-hop share.**
At a pinned path count of 3, `sgate` on a global row still falls 0.021484 → 0.001953 → 0 over
`s = 32/128/512`; DeltaNet, which has no row normalizer, holds 0.074219 / 0.076172 / 0.065430.

Nothing arrests it: `lam` 0.05→2.00, `rho` 0.5/1.5/4.0, depth 2 and 3, and head width
16/32/64/128 — all read **0.00000 at s=128** while holding 0.13–0.18 at s=8.

Honest control: on the sign-unconstrained *input* path, softmax decays too (0.03125 /
0.01172 / 0.00000). This is dilution of one token's leverage, common to attention. What is
specific to the signed operator is only that it has the property on the **value** path where
softmax is pinned at exactly zero — an infinite ratio over a vanishing absolute.

**The headline was never reported with an interval.** 0.1641 is 21/128; binomial 95%
interval ≈ [0.10, 0.24]. Re-runs give 0.1719 and 0.1445.

**A 3.34% val-loss gap is inside the band where loss stops predicting capability.**
arXiv:2605.20798 (verified by direct fetch): *"two significant failures converge to within
2-3% of baseline validation loss yet drop 6-16 CLIMB-points"*, Spearman ρ = −0.27 between
1.2B and 3B improver rank. The campaign's own win condition sits inside that band.

**A leap was attempted and falsified.** `A = ρ·sgn(w)·softmax(|w|)` — sign from an
unnormalized pairwise quantity, magnitude from a normalized one — decays identically
(0.0840 / 0.0332 / 0.0137 / 0.0039 / 0.0020). It is also Cog Attention's matrix
(arXiv:2411.07176), published. The normalizer was never the problem.


**Five of the six original requirements were deleted by their own falsifiers.** The
specification began with six behaviours; R1, R2-as-max-plus, R3, R4 and R6 each failed the
test written to kill them, and were deleted rather than defended. What ships is what
survived.

**Intervention generalization was NOT achieved.** On a held-out composition of two sign
flips, OOD NRMSE was attention 5.8198, APPNP 4.2107, signed 2.6151, median of 5 seeds.
NRMSE 1.0 is predict-the-mean, so **all three arms are worse than a constant predictor**.
The signed operator fails 1.6× less badly than the control. It does not generalize.
`tests/w4/test_w4_intervention.py::test_signed_arm_generalizes_rather_than_merely_degrading_less`
is RED and stays RED.

**PARITY was reached only after four operator knobs were tuned, and only at 3.3M
parameters.** The operator as originally shipped -- raw logits, row-L1 normalized -- loses
to softmax and the gap WIDENS with training. The operator that reaches parity is a
difference of two softmaxes (`rho=1.5, lam=0.10, hops=2`) at a learning rate swept for both
arms: median ratio **1.0334**, spread 1.0199-1.0413, under the 1.05 bar on **5/5 seeds**,
parameters exactly equal at 3,319,296, wall-clock **1.32x**. The signed arm had four knobs
tuned (rho, lam, hops, lr); softmax had one (lr), because it has no operator knobs. That
asymmetry is real and is stated rather than buried. Whether parity survives a longer budget
is UNTESTED -- the widening measured below was on the original operator.

**The one distinguishing property is a SHORT-CONTEXT property, and the size of its decay was
overstated.** Content-conditional sign — whether a *third* token can decide if `j` helps or
hurts `i` — is the property that separates this operator from every non-negative one, and the
0.1641 on record was measured at the probe's default context of **s = 8** and its default
`floor = 1e-6`. Swept over context on the same calibrated `ceq/bench.sign_flip_rate`, holding
relative positions fixed at `i = s−1, j = s/4, c = s/2`, 1024 draws, at both floors:

| s | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| `sgate`, `floor = 0` | 0.181641 | 0.098633 | 0.050781 | 0.026367 | **0.012695** |
| `sgate`, `floor = 1e-6` — as published through round 4 | 0.174805 | 0.088867 | 0.026367 | 0.011719 | **0.003906** |
| softmax control, either floor | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The decay is real; the exponent is withdrawn and no replacement is published (see "Limits,
first"). Three of the four controls offered for it still stand. The fourth was blind, and it
was the one that mattered.

*The discard floor IS a cause, and the control that said otherwise could not have seen it.*
That control ran at `j = 1` while every published headline uses `j = s/4`, and at 256 draws
whose resolution (1/256 = 0.00390625) is **exactly the value it reports** at `s = 64` and
`s = 128`. An instrument pinned at its own resolution cannot show a difference where the
difference lives. Re-run at the published geometry, the floor-off / floor-on ratio runs
**1.04 / 1.11 / 1.93 / 2.25 / 3.25×** over `s = 8…128` and the floor-off curve rises
monotonically against the floor-on one — the opposite of "the same curve". The claim that the
floor is not the cause is **withdrawn**.

*The negative half is what is being measured.* `lam = 0` deletes it and reads exactly
**0.0000 at every `s`**, identical to softmax; every `lam` from 0.05 to 2.00 reads 0.12–0.14
at `s = 8`.

*It is not specific to this operator.* The shipped row-L1 `signed` operator decays identically,
0.0957 → 0.0039 over the same range.

*No operating point escapes it.* Nine settings, `s = 8` → `s = 128`, 256 draws, from the
test's own grid: `lam` 0.05 **0.140625 → 0.007813**, 0.25 **0.160156 → 0.000000**, 1.00
**0.128906 → 0.000000**, 2.00 **0.144531 → 0.000000**; `rho` 0.5 **0.140625 → 0.003906**,
4.0 **0.167969 → 0.003906**; `d_head` 32 **0.183594 → 0.000000**, 64 **0.132813 → 0.003906**,
128 **0.128906 → 0.000000**. Depth 2 and 3 do not restore it either (0.03125 and 0.00391 at
`s = 64` against 0.109 and 0.082 at `s = 8`).

**Head width — the axis that actually grows from 3.3M to 300M — does not touch it.** A 300M
model runs `d_head` 64–128, not the probe's default 16. Swept at 192 draws, the rate at
`s = 128` is **0.00000 at d = 16, 32, 64 and 128 alike**, while `s = 8` holds at
0.16667 / 0.17708 / 0.13021 / 0.13542. Wider heads do not restore the property; at `s = 32`
they slightly worsen it (0.03646 → 0.00521 going from d = 16 to d = 64).

**It is not attention fading.** The suspected mechanism — that `softmax(-w)` spreads over the
prefix and becomes a running mean — is refuted: the negative-entry fraction of `A` is flat at
**0.4390 → 0.4673** from `s = 8` to `s = 1024`, `min A` is pinned at **−0.1364** = `−ρλ/(1+λ)`
throughout, and both branches stay at 0.05–0.12 of their uniform entropy ceiling.

**The cause is that the property and its decay are the same fact.** A third token `c` reaches
the pair `(i, j)` only along a path *through* `c`, and the first term of `J = Σ_k A^k`
containing one is `k = 2`. Measured by the test itself at 512 draws: at `hops = 1` the rate
is **0.017578** for `sgate` and **0.000000** for the pairwise-sign operator, against
**0.123047** and **0.083984** at `hops = 2`. So the
property is carried entirely by the two-hop term, which is a sum over ~`s` intermediate
tokens of which `c` is exactly one. Its share is `1/s` by construction. Any operator that
obtains content-conditional sign from a multi-hop path sum inherits the dilution.

**Round 4 corrects this.** That sweep moved `j = s/4` and `c = s/2`, so the intermediate
count and every softmax row normalizer grew with `s` together. Pinned at `j = i−4, c = i−2`
the path count is a constant 3, and `sgate` on a global row **still** falls 0.021484 →
0.001953 → 0.000000 over `s = 32/128/512`, while DeltaNet — signed entries, no row
normalizer — reads 0.074219 / 0.076172 / 0.065430 and is bit-for-bit identical under a
sliding window. The dilution is the **softmax row denominator summing over `s` tokens**, not
the two-hop share. `tests/foreman/test_paraformer_ratio_across_context.py::test_claim_the_context_decay_is_the_path_count_and_not_the_row_normalizer` — RED.

**And softmax does the same thing wherever it has a sign to lose.** On `wrt="x"`, the input
path, which is sign-unconstrained for every operator, softmax reads 0.025391 / 0.009766 /
0.000000 at `s` = 8 / 32 / 128 on cpu and 0.023438 / 0.009766 / 0.000000 on cuda, 512 draws.
The decay is dilution of one token's leverage in a context of
`s`; what is specific to the signed operator is only that it has the property on the *value*
path at all, where softmax is pinned at exactly zero.
`tests/foreman/test_sgate_scaling_defect.py::test_claim_content_conditional_sign_survives_a_longer_context`
`tests/foreman/test_sgate_scaling_defect.py::test_claim_some_knob_in_the_operator_family_arrests_the_context_decay`
`tests/foreman/test_sgate_scaling_defect.py::test_claim_the_property_does_not_live_only_in_the_multi_hop_term`
`tests/foreman/test_sgate_scaling_defect.py::test_claim_the_context_decay_is_specific_to_the_signed_value_path`
all RED.

**The ratio HOLDS on the axes that could be measured — and that is worse news, not better.**
The pathology the L1 operator showed (1.2191× at 250 steps → 1.337× at 600) does **not**
reproduce on `sgate` across context length or width. 600 steps, seed 0, lr 1e-3:

| seq | 64 | 128 | 256 | | `d_model` | 128 | 256 | 384 |
|---|---|---|---|---|---|---|---|---|
| ratio | 1.0356 | 1.0483 | **1.0340** | | ratio | 1.0777 | 1.0312 | **1.0551** |

Both non-monotone, both flat inside the 1.0199–1.0413 five-seed spread already on record
(single seed each, so a 0.024 swing is noise). `tests/foreman/test_sgate_scaling_defect.py::test_claim_the_parity_ratio_does_not_widen_with_context_length`
and `::test_claim_the_parity_ratio_does_not_widen_with_model_width`, both GREEN.

So the projected outcome at 300M is **not** a widening gap. It is validation-loss parity with
an operator that has, over exactly that context range, stopped being distinguishable from
softmax — the sign rate falls 0.17480 → 0.00391 while the ratio sits still. A widening ratio
would at least be a reason to stop. This is 1.32× wall-clock and 2.65×–8.06× training memory
spent to reproduce softmax.

**A 3% validation-loss gap is not parity at 1-3B, and there is a measurement saying so.**
`arXiv:2605.20798` (20 May 2026) tested 20 post-2021 Transformer modifications at 1.2B and 3B
under iso-data, iso-compute, iso-recipe control and reports, verbatim: "two significant
failures converge to within 2-3% of baseline validation loss yet drop 6-16 CLIMB-points",
and "1.2B improver rank is a weak predictor of 3B improver rank" at Spearman ρ = −0.27, with
observed rank moves 7→1, 10→3 and 1→6. This module's headline is **1.0334** — a 3.34% gap —
measured at 3.3M, three orders of magnitude below where that decoupling was characterised.

**The ORIGINAL operator loses to softmax and the gap WIDENS with training.** From scratch, byte-level TinyStories, identical
parameter counts (3,319,296 each), identical initialization, data, steps and optimizer:

| budget | softmax val | signed val | ratio |
|---|---|---|---|
| 250 steps, bs 16 | 1.8838 | 2.2965 | **1.2191×** |
| 600 steps, bs 32 | **1.5780** | **2.1103** | **1.337×** |

Same architecture in both rows (d=256, 4 layers, seq 128); only the training budget differs.
**More training makes it worse, not better** — softmax exploits its selectivity as budget
grows while the flat operator saturates. The signed arm also spends ~54% more wall-clock
(`hops=3` is three matmuls against one), so the gap is not an iso-FLOP artefact in its
favour. Both beat uniform `ln(256) = 5.5452`, so both genuinely learned.
`tests/w10/test_w10_from_scratch.py::test_pure_signed_operator_trains_to_parity_with_softmax`

**Why it saturates: the operator has no temperature channel, structurally.** `A = ρ·w/Σ|w|`
is homogeneous of degree **zero** in the logits, so `A(t·q, k) = A(q, k)` for every `t > 0`.
Scaling the queries by 64× moves the peak weight of a row by `0.000e+00` — it stays at
0.100336 — while softmax on the identical tensors sweeps 0.0504 → 1.0000. Selectivity by
logit scale is exactly what softmax exploits as budget grows, and it is the one thing this
operator cannot do: every row spends exactly ρ of absolute mass forever. The gradient shows
the same fact as an exact rank deficiency, `grad_q · q = 1.088e-14` against softmax's 10.561.

This is *not* a conditioning problem. The suspected `1/‖w‖₁` blow-up from the normalizer's
rank-one Jacobian correction does not appear: at identical initialization on identical
TinyStories bytes, total gradient norms are **0.9767×** softmax's, matched within 2.3%.
`tests/foreman/test_l1_normalizer_obstruction.py::test_claim_the_operator_can_sharpen_its_attention_by_scaling_its_logits`
`tests/foreman/test_l1_normalizer_obstruction.py::test_claim_the_query_magnitude_carries_gradient`
`tests/foreman/test_l1_normalizer_obstruction.py::test_claim_the_normalizer_makes_the_signed_arm_worse_conditioned_at_init`

**The operator is a triangular solve.** At `K = S−1` the path sum equals forward
substitution on `(I − A)z = v` to **1.776e-15** (Golub & Van Loan Alg. 3.1.1); at any `K` it
equals `K+1` steps of the Neumann/Jacobi iteration `z ← Az + v` to **8.882e-16**; and it
equals a recurrent linear attention (Katharopoulos et al., arXiv 2006.16236) whose `d×d`
state accumulates the layer's own **output** rather than the value, to **1.479e-14**
relative — so the quadratic form is a choice, not a requirement. The `ρ(A) = 0` terminating
resolvent is the ordinary nilpotency of a strictly triangular matrix, and it is already the
documented justification inside shipping kernels: sglang's Kimi-Delta-Attention prefill
computes `(I+L)⁻¹ = (I−L)(I+L²)(I+L⁴)(I+L⁸)` for `L` strictly lower triangular with `L¹⁶=0`.
`tests/foreman/test_operator_is_a_triangular_solve.py`

**The α gate is not superior to self-attention and cannot be.** At α = 0 it is bitwise
stock SDPA by construction, so it is self-attention *plus a term*, never a replacement. It
exists to get an honest cost curve on a pretrained checkpoint.
`tests/w8/test_w8_real_model.py::test_alpha_zero_is_bitwise_stock_attention`

**The Triton kernel is forward-only.** No backward pass exists. The CPU reference is
differentiable and gradcheck-clean; training must use it.

**Memory is far worse in TRAINING than the inference figures suggest.** The 1.31×–1.94×
on record is the forward-only kernel in the 128–512 band. Training forward+backward against
SDPA at identical parameters, fp32:

| seq | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| signed / SDPA peak memory | 1.44× | 1.74× | **2.65×** | **4.45×** | **8.06×** |

SDPA never forms `[S,S]`; this operator materializes it and autograd retains a measured 3.9
tensors of `[B,H,S,S]` per layer, so activations are O(S²) against O(S). Consequence: a 0.5B
signed model **fits no Colab GPU at seq 2048 batch 1** — 36.31 GiB against a 33.53 GiB A100-40GB
budget — without gradient checkpointing, which is measured at 0.379× (1287.5 → 487.7 MiB) and
is therefore mandatory rather than optional. bf16 autocast does **not** halve it: measured
0.748/0.772/0.797 at seq 256/512/1024, i.e. 2.89–3.36 effective bytes per element, not 2.
FLOP/token is only 1.197×, but wall-clock is 1.84× at seq 512 and 2.40× at 1024.

**The hop cache is exact forward and lossy backward.** Eviction is bitwise only if nothing
dependent has run yet; afterwards the contribution is already contracted into stored hop
vectors and cannot be unwound.
`tests/w9/test_w9_hopcache.py::test_eviction_is_not_retroactive`

**Prefill and decode agree to one ULP, not bitwise** — 1.898657e-16 relative, float64.
Different summation order, nothing more.

**No scaling-law claim is made.** There is no published ~30-architecture inversion study;
the nearest real work is Tay et al. ([arXiv:2207.10551](https://arxiv.org/abs/2207.10551)),
ten architectures at 2.9B dense, with no intercept-vs-slope decomposition.

**Perplexity is a sanity check, not a win condition.**

---

## What this is for, and which niche it could occupy

A stranger should know in one paragraph what this is: **a negative result, shipped with the
instrument that produced it and a machine-checked core.** The operator lost its capability test
— 0.0000 against softmax's 0.0293 on COGS generalization at 3,652,096 matched parameters,
Fisher **p = 2.7502788939e-05**, and behind in-distribution too — and every novelty claim this
project made has been withdrawn against prior art the project found itself. That is the
finding. It is not a disclaimer bolted to a product; it is the product.

The common view is that there are two options: claim novelty, or don't ship. There is a third,
and it is this one:

- **The negative result is the rare artifact.** Attention variants are proposed constantly and
  almost none ship a matched-parameter capability loss against their own control. This one
  does: identical steps / lr / batch / seed / data / eval subsample, one seed, and the deficit
  already present in-distribution (0.7734 against 0.9258), so it is not a generalization story.
- **The instrument is transferable and it reproduces itself.** `python -m ceq.diagnose` answers
  "does my non-standard attention operator actually differ from softmax?" in about two minutes
  on CPU, with softmax printed beside it as a control reading exactly 0.0. Calibration
  self-test **4/4 bit-identical**; 26/26 published probe cells reproduce exactly; a measurement
  replayed from a fresh interpreter comes back bit-identical.
- **The instrument's own defect is published against itself.** This round found that
  `sign_flip_rate`'s absolute discard `floor` biases arms whose row norms differ — `sgate`'s
  row-L1 is pinned at 1.5, DeltaNet's reaches 81.6 at `s = 512` — which inflated a decay
  exponent and manufactured a 2.07× margin where `floor = 0` shows 1.08×. Anyone running a
  gradient-sign probe across operators of different scale needs that warning, and it is worth
  more than the claim it destroyed.
- **The core is machine-checked.** `lake build CEQ` exit 0, zero `sorry`, no `sorryAx`.
- **The retractions are complete rather than quiet.** An "84×" headline was traced to an arm
  that never applied its own hop coefficients and withdrawn in full; five of six original
  requirements were deleted by their own falsifiers, each still on the page with the number
  that killed it.
- **The costs ship inside the package**, in `ceq.hf.modeling_ceq.COSTS`, because a cost recorded
  only in a markdown file does not reach the person who downloads the weights.

### The separation is not a capability claim — it is a depth-and-parameter claim

This has to be stated before any niche argument, because the strong version is false and this
repository has already measured it false. For a stack of **non-negative operators with linear
value paths**, `∂out_i/∂v_j` is a non-negative combination of path products times a fixed
matrix, so no third token can flip its sign — softmax reads exactly 0.0000 at depth 1 and
depth 2, three seeds each. **Put one GELU between two softmax layers and the sign flip comes
back.** A real transformer has an MLP.

`tests/cameron/test_negation_is_the_axis.py::test_stacked_softmax_layers_cannot_flip_that_sign_at_any_depth`
`tests/cameron/test_negation_is_the_axis.py::test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`

So the surviving claim is not "softmax cannot express content-conditional negation". It is
**"softmax needs a nonlinearity and the depth to route through it, where the signed operator
has it in one layer"** — a claim about depth and parameter efficiency, measurable only where
parameters are scarce. And it needs at least two hops to exist at all: at `hops = 1` the
coefficient is `A_ij`, built from `q_i` and `k_j` alone, and the measured flip rate is exactly
0.0000.

### What a niche would have to look like, and the one candidate that fits

Three constraints fall out of the measurements, and together they are narrow:

1. **The discriminating span must be short.** The rate is 0.181641 at `s = 8` and 0.050781 at
   `s = 32` (`floor = 0`), and it decays in window width as fast as in context (−1.159,
   R² 0.9933). Anything whose evidence is hundreds of tokens away is out.
2. **Parameters must be scarce**, because the claim is efficiency, not capability.
3. **The task must be scored on the model's own next-token probabilities**, since this is a
   text-generation model and the property lives in the attention block, not in a task head.

**The candidate that satisfies all three is minimal-pair NPI licensing under negation — the
BLiMP paradigms, inside the BabyLM strict-small regime.** BLiMP scores a minimal pair by which
sentence the LM assigns higher probability, which is exactly constraint 3; the licensor and the
NPI sit within one clause, which is constraint 1; and BabyLM's strict-small track fixes the
budget at 10 million words so a from-scratch small model is the norm rather than an outlier,
which is constraint 2. The paradigm is literally a third token deciding whether another token
is licensed.

Published accuracies (BLiMP, [ACL 2020.tacl-1.25](https://aclanthology.org/2020.tacl-1.25/),
5-gram / LSTM / Transformer-XL / GPT-2 / human):

| paradigm | 5-gram | LSTM | TXL | GPT-2 | human |
|---|---|---|---|---|---|
| `sentential_negation_npi_licensor_present` | 93 | 100 | 99 | 89 | 93 |
| **`sentential_negation_npi_scope`** | **45** | **23** | **53** | **95** | **81** |
| `only_npi_scope` | 30 | 36 | 45 | 85 | 72 |
| `npi_present_1` | 47 | 54 | 61 | 55 | 83 |

The `licensor_present` row is at ceiling for everything and separates nothing. **The `_scope`
rows are where models fail** — an LSTM at 23 is below chance, and the human is at 81 — so there
is a real dynamic range for a small model to be measured in, and "is there a *not* nearby" is
not enough to solve it.

**None of this has been run. It is a hypothesis with an address, not a result.** If it is
attempted, the kill is pre-registered here in the form this project uses for everything else:

> **KILL.** If `sgate` does not beat its own matched-parameter softmax control on
> `sentential_negation_npi_scope` and `only_npi_scope` by more than the seed spread, the niche
> hypothesis is deleted, like R1, R2, R3, R4 and R6 before it.

And the prior is against it: the **one** capability comparison ever run at matched parameters
went to softmax, in-distribution as well as out.

**What the niche is not.** Not general language modelling — the operator lost COGS and its
val-loss gap widened with budget on the original form. Not long context — the property decays
in both context length and window width. Not ARC-AGI — a 120-task exact-match benchmark cannot
resolve below ~5%, which `tests/cameron/test_arc_reality.py` derives from the scoring rule
rather than accepting on authority. Not SCAN — both arms score 0.0000 and two zeros separate
nothing. Not negation-scope *tagging* (CD-SCO, BioScope, SFU): those are token-classification
tasks whose published baselines are feature-based SVM/CRF systems or fine-tuned BERT-base at
110M parameters, so a 3.65M from-scratch generator has no comparison frame there.

### The scale gate, priced from this repository's own sizing module

The "912 A100-hours" figure that circulated in this project's checklists **has no derivation
anywhere in the repository** — it appears only in summary lines. Computed instead from
`ceq/sizing.py`, whose inputs are explicit (`TOKENS_PER_PARAM = 20` against 302,088,192
non-embedding parameters = 6.04e9 tokens; A100 bf16 peak 312 TFLOP/s; MFU 0.40 for the
**control**; 4 forward-equivalent passes with gradient checkpointing), and taking the operator's
cost as the control's cost times the **measured** step ratio (3.13× at seq 1024, 5.49× at
seq 2048, `tests/chase/test_scale_sizing.py`) rather than at an assumed MFU:

| 300M run, one seed | softmax control | `sgate` | 24-h Colab sessions | Colab compute units |
|---|---|---|---|---|
| seq 1024 | 41.4 A100-h | **129.7 A100-h** | 5.4 | ~1,945 |
| seq 2048 | 46.8 A100-h | **257.2 A100-h** | 10.7 | ~3,858 |

At Colab's published rate of roughly 15 compute units per A100-hour and 100 units per $9.99,
that is **≈ $194 (seq 1024) or ≈ $385 (seq 2048)** for the operator arm plus ≈ $62–70 for the
control. The gate is expensive, **not unreachable, and not 912 hours.** What blocks it is not
money: `ceq/hf/train.py::train()` trains `steps` from scratch and `save_pretrained`s at the
end, with **no resume**, so a run needing 5–11 sessions against Colab's 24-hour cap cannot
currently survive the gap between them. No Colab tier guarantees an A100 either.

**And there is a free step nobody has taken.** The shipped notebook's default shape
(`hidden 512, 8 layers, 8 heads, seq 512, batch 8`, byte vocab) is **25,707,520 parameters** —
**7.0× above the 3.65M ceiling that is the highest this project has ever run.** At a full
Chinchilla budget (5.04e8 tokens) it costs **2.3–3.7 T4-hours** for the operator arm and
1.3–2.0 for the control, at an assumed 25–40% MFU on a 65 TFLOP/s T4, and it **fits**: [RUN]
`ceq.hf.train.preflight(..., gpu="T4-16GB")` returns ok=True at **2.72 GiB total (0.43 GiB
state + 2.28 GiB activations) against a 13.50 GiB budget, headroom +10.78 GiB**, batch 8, no
gradient checkpointing; the softmax control at the same shape is 1.06 GiB. (This README
previously said 2.45 GiB against 14.5 GiB — stale numbers, unchanged conclusion.) That is inside one free 12-hour session, for $0. As shipped the notebook runs 2000
steps × batch 8 × seq 512 = 8.19e6 tokens, which is **1.6% of that budget** — a smoke run, not
a training run.

### What is trainable for free, and the one line of code that sets the ceiling

**The ceiling is not the GPU and not the quota. It is `ceq/hf/train.py::train()`.** [READ]
That function does `for step in range(steps)` from zero, builds a fresh model, constructs a
fresh `AdamW`, and calls `save_pretrained` **once, after the loop**. There is no `resume`
argument, no `global_step`, no `torch.save` of optimizer state, and no load path — `grep -rniE
"resume|load_state_dict|global_step|torch\.save" ceq/hf/` finds `from_pretrained` only in
`smoke.py`'s round-trip test. **Every free tier has a session cap, so the largest trainable
model is whatever finishes in ONE uninterrupted session, not whatever the weekly quota buys.**

Free tiers, fetched 2026-08-25, ranked by the column that actually matters:

| tier | GPU | session cap | quota | persistent disk |
|---|---|---|---|---|
| **Kaggle Notebooks** | P100-16GB or 2×T4-16GB | **12 h** | **30 GPU-h/week, published** | **20 GB auto-saved `/kaggle/working`** |
| Colab free | not published | 12 h | **not published** | none native (Drive mount only) |
| Paperspace/Gradient free | Quadro M4000 | 6 h | not published | 5 GB |
| Lightning AI free | T4 / A10 | ~4 h restart cycle | 80 GPU-h/month | 50–100 GB (sources disagree) |
| SageMaker Studio Lab | T4 (secondary source) | 4 h / 24 h | 4 GPU-h/day | 15 GB — **closed to new signups** |
| HF Spaces ZeroGPU | RTX Pro 6000 48/96GB | **60 s per call** | 5 min/day free | ephemeral |

Kaggle wins on all three: longest session, only published quota, and the only free tier whose
working directory survives a kill without mounting another product. ZeroGPU is unusable for
training at 60 s per call regardless of how good the card is.

Priced against it, byte vocab 256, seq 512, batch 8, Chinchilla 20 tokens/non-embedding
parameter, operator cost = control × the **measured** 3.13× step ratio:

| what | MFU 0.30 | MFU 0.15 |
|---|---|---|
| largest **operator** run finishing in ONE 12-h session — the real free ceiling today | **37.8M** (d=512, L=12, 11.9 h) | **25.2M** (d=512, L=8, 10.6 h) |
| same, with a resume path written, 4 weeks of Kaggle quota | 100.7M | 71.4M |
| same, 12 weeks | 177.0M | 127.4M |
| same, 26 weeks | 265.9M | 192.7M |
| memory ceiling only, free T4-16GB, signed arm, no checkpointing | 308.3M | — |
| memory ceiling only, with `grad_checkpoint=True` | 737.4M | — |

**Memory was never the binding constraint. Time is.** A 300M model FITS a free T4; it just
cannot be trained on one.

**The 300M gate is not reachable free.** A matched pair at seq 1024 is 41.4 A100-h (control) +
129.7 A100-h (operator) = **171.1 A100-hours**, which is **1,095 T4-hours at MFU 0.30 — 37
weeks of Kaggle's entire free weekly quota spent on nothing else**, against a 12-hour session
cap and no resume. At seq 2048 it is 304.0 A100-h, 65 weeks. It is a **paid** run, and the
cheapest honest price is not Colab compute units: 171.1 A100-hours is **$46–50 on vast.ai
A100-40GB at $0.27–0.29/h** (preemptible, host-dependent), **$204 on RunPod A100-80GB at
$1.19/h**, or **≈$256–455 in Colab compute units**. The vast.ai number is an order of magnitude
under the figure this README quoted before, and it is the one to plan against.

**So what ships.** At the 3.65M ceiling this project has actually trained, nothing ships. The
notebook's default 25.7M shape is 7.0× that and finishes a full Chinchilla budget inside one
free Kaggle session — **that is the release that exists today, for $0**. Getting to ~100M needs
one patch, not one dollar: save `opt.state_dict()` and the step counter next to the weights and
accept a `resume_from=` in `train()`. Getting to 300M needs about $50 and a card.

---

## The signed influence property, and who already had it

A token can **reduce** another token's contribution. This is not new, and the paragraph
below the table used to imply that it was.

| operator | min influence Jacobian `∂out_i/∂v_j` |
|---|---|
| non-negative control (softmax, same construction) | **+0.000000e+00 — exactly zero** |
| signed operator | **−9.000000e-01** |

This is not a tuning difference. The Kleene star of a non-negative matrix has a
non-negative influence Jacobian **in any ordered semiring** — measured exactly 0.000e+00
over 40 max-plus instances and 160 APPNP kernels. Softmax attention, APPNP and the max-plus
star are all inside that class, so none of them can express a negation. Dropping
non-negativity is the only escape.

`tests/w6/test_w6_attention.py::test_signed_operator_reaches_negative_influence`
with `::test_the_nonnegative_control_is_stuck_at_exactly_zero` as its calibration.

**Published attention is already outside that class**, and the escape was found in 2022.
Reimplemented from their defining equations and measured on this same instrument:

| operator | min influence Jacobian |
|---|---|
| SimA, arXiv 2206.08898 (2022), `Q/‖Q‖₁ · (K/‖K‖₁)ᵀ` | **−3.929583e-01** |
| Differential Transformer, arXiv 2410.05258, `(softmax(A₁) − λ·softmax(A₂))V` | negative at λ = 0.8 |
| DeltaNet UT transform, arXiv 2406.06484 Eq. (10) | strictly-triangular matrix, min entry **−10.875878** |

**And this module's own matrix is published.** Signed Dual Attention
([arXiv:2606.04833](https://arxiv.org/abs/2606.04833), v1 3 Jun 2026) Eq. 1–2 is
`A⁺ = softmax(QKᵀ/√d_k)`, `A⁻ = softmax(−QKᵀ/√d_k)`, `SDA(Q,K,V) = (A⁺ − A⁻)V` — this
campaign's `sgate` matrix at `lam = 1`, single-hop. (Differential Transformer is *not* the
same object and this repository has been sloppy about that: arXiv 2410.05258 Eq. 1 splits
`[Q₁;Q₂] = XW^Q` and differences two DISTINCT logit matrices from separate projection halves,
not one logit matrix and its negation.)

SimA states the property in its own text: L1-normalizing Q and K instead of applying
softmax means "the attention values can become negative, meaning that a token can affect
another one negatively." So signedness alone is a rediscovery.

**The multi-hop half of the pairing is occupied too, and this README used to deny it.**
ParaFormer ([arXiv:2512.14619](https://arxiv.org/abs/2512.14619), 16 Dec 2025) Eq. 9 is

```
Z = Σ_{k=0..K} γ_k Â^k V          Â = Softmax(QKᵀ/√d)
```

with "{γ_k ∈ ℝ | k = 0,1,2,⋯,K} … a set of learnable weights", and Theorem 1's construction
`γ_k = (−a)^k/2` is negative on odd k. That is multi-hop propagation with **signed hop
coefficients**, so the sentence "every multi-hop propagation requires a non-negative matrix"
was false as written. Signed *matrices* raised to powers are occupied as well, in graph
filtering: BernNet ([2106.10994](https://arxiv.org/abs/2106.10994)) takes `L^k`, ChebNetII
([2202.03580](https://arxiv.org/abs/2202.03580)) takes `T_k(L̂)`, PCNet
([2403.03676](https://arxiv.org/abs/2403.03676)) takes `(−L̃)^n` — all of a **fixed
structural** Laplacian, none of a content-dependent QK matrix.

**Round 4 retracts the 84x.** `ceq/bench.py::sign_flip_rate` built the `gam` vector for the
`paraformer` arm and then never referenced it: the arm fell through to
`else: a = _softmax_operator(...)` followed by `h = a @ h`, which is plain single-hop
softmax. "0 flips in 2048 draws, Clopper-Pearson ceiling 0.00146, at least 84×" was a
**softmax** number wearing ParaFormer's name — `test_the_paraformer_arm_is_not_the_softmax_arm`
was RED at 0.02734375 == 0.02734375 before the arm was routed through `Σ_k γ_k Â^k h`.

Both rates are binomial, so every entry below carries its exact Clopper-Pearson 95%
interval. `hops = 2`, 2048 draws per cell, positions `i = s−1, j = s/4, c = s/2`, seed 0,
CPU (`python scale/ratio_sweep.py --n 2048 --hops 2`). The last column is the
**conservative** separation `CP95_lo(signed matrix) / CP95_hi(signed coefficients)` — the
smallest factor consistent with the draws at 95%.

| s | sgate, `Σ A^k`, `A` signed | ParaFormer, `Σ γ_k Â^k`, `Â` non-negative | softmax | conservative ratio |
|---|---|---|---|---|
| 8 | 0.166016 [0.150143, 0.182850] | 0.029785 [0.022858, 0.038097] | 0 of 2048 [0, 0.001800] | **3.94×** |
| 16 | 0.078613 [0.067323, 0.091127] | 0.009277 [0.005595, 0.014450] | 0 of 2048 | **4.66×** |
| 32 | 0.027344 [0.020720, 0.035363] | 0.002930 [0.001076, 0.006366] | 0 of 2048 | **3.25×** |
| 64 | 0.008789 [0.005217, 0.013855] | 0.000488 [0.000012, 0.002717] | 0 of 2048 | **1.92×** |
| 128 | 0.003418 [0.001375, 0.007030] | 0.000000 [0, 0.001800] | 0 of 2048 | **0.76×** |
| 256 | 0.000000 [0, 0.001800] | 0.000488 [0.000012, 0.002717] | 0 of 2048 | **0.00×** |
| 512 | 0.000000 [0, 0.001800] | 0.000000 [0, 0.001800] | 0 of 2048 | **0.00×** |

The point ratio at `s = 8` is 5.6×, not 84×. The conservative bound peaks at **4.66× at
`s = 16`** and falls below 1 by `s = 128`, where 2048 draws simply run out of resolution —
at `s = 256` the ordering inverts inside the noise. Re-measured at **16,384 draws**, `s = 128`
is sgate 64/16384 = 0.003906 [0.003010, 0.004985] against ParaFormer 2/16384 =
0.000122 [0.000015, 0.000441], a conservative **6.83×**. So at `s = 128` the instrument ran
out, not the effect. It does not save the claim: at `s = 256` the same 16,384 draws give
1.92× and at `s = 512` they give **0.44×**, overlapping again.

The POINT ratio grows cleanly with context — 5.6×, 8.5×, 9.3×, 18×, 32× at `s = 8…128` — which
is the direction round 3's first-vs-second-order argument predicts. The CONSERVATIVE bound
never follows it: 3.94, 4.66, 3.25, 1.92, then 6.83, 1.92, 0.44. It never exceeds 7× and it
is below 1 by `s = 512` even at 16,384 draws. Log-log decay over `s = 8…512`: sgate **−1.437**
(R² 0.9959), ParaFormer **−1.261** (R² 0.8614). A ratio that grows over two absolute rates
that both vanish is not a capability.

**Every rate and every exponent in this section is `floor = 1e-6`, and the floor is not
neutral.** It discards up to 3.25× of `sgate`'s signal by `s = 128` (see "Limits, first"), so
these are instrument readings at one setting rather than properties of an operator. The floor
sweep has been run only for `sgate` and DeltaNet; the ParaFormer and Cog/SignGT columns have
**never** been re-measured at `floor = 0`. No ratio in this section is floor-independent, and
the exponents quoted just above are withdrawn on the same ground as the −1.389.

**The residual cell is occupied by a kernel that ships, and it wins.** DeltaNet's chunkwise
form ([arXiv:2406.06484](https://arxiv.org/abs/2406.06484) `eq:inverse`) is

```
T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)
```

— strictly lower triangular, built from `K` and `β` so **content-dependent**, entries
`−β_i (k_i · k_j)` with no non-negativity anywhere so **signed per entry**, and inverted
exactly. Strictly triangular is nilpotent, so that inverse *is* a finite path sum
`Σ_k (−M)^k`; [arXiv:2606.06034](https://arxiv.org/abs/2606.06034) writes it out as
`(I−A)⁻¹ = Σ_n Aⁿ`. It is in production: `fla/ops/utils/solve_tril.py` ("Compute the inverse
of the matrix I + A. A should be strictly lower triangular") and sglang's
`chunk_kda_fwd_kernel_inter_solve_fused`, wired into `RadixLinearAttention`.

Put on the same probe, same draws, same `hops = 2`:

| s | this module, `sgate` | **DeltaNet WY matrix** | Cog Attention / SignGT matrix | ParaFormer Eq. 9 |
|---|---|---|---|---|
| 8 | 0.166016 [0.150143, 0.182850] | 0.082031 [0.070507, 0.094768] | 0.105469 [0.092494, 0.119585] | 0.029785 [0.022858, 0.038097] |
| 16 | 0.078613 [0.067323, 0.091127] | **0.097656** [0.085139, 0.111339] | 0.035156 [0.027607, 0.044071] | 0.009277 [0.005595, 0.014450] |
| 32 | 0.027344 [0.020720, **0.035363**] | **0.059082** [**0.049264**, 0.070185] | 0.013184 [0.008706, 0.019124] | 0.002930 [0.001076, 0.006366] |
| 64 | 0.008789 [0.005217, 0.013855] | **0.021484** [0.015653, 0.028735] | 0.004395 [0.002011, 0.008326] | 0.000488 [0.000012, 0.002717] |
| 128 | 0.003418 [0.001375, 0.007030] | **0.010254** [0.006358, 0.015632] | 0.000977 [0.000118, 0.003523] | 0.000000 [0, 0.001800] |
| 256 | 0.000000 [0, 0.001800] | **0.003906** [0.001688, 0.007682] | 0.000000 | 0.000488 |
| 512 | 0.000000 [0, 0.001800] | **0.000977** [0.000118, 0.003523] | 0.000000 | 0.000000 |
| slope (`floor = 1e-6`, withdrawn) | −1.437 (R² 0.9959) | **−1.107** (R² 0.9300) | −1.651 (R² 0.9932) | −1.261 (R² 0.8614) |

**The "non-overlapping 1.39× win" read off `s = 32` is WITHDRAWN as a floor artifact.** The
floor is absolute and the two arms are not on the same scale: `sgate`'s row-L1 is pinned at
`rho = 1.5`, while DeltaNet's grows roughly linearly with context (4.23 / 21.53 / 81.62 at
`s = 32/128/512`), so one absolute threshold discards far more of `sgate` than of DeltaNet. At
`s = 32`:

| floor | `sgate` | DeltaNet | ratio |
|---|---|---|---|
| 1e-6 — as published | 0.026367 | 0.054688 | 2.07× |
| 1e-9 | 0.045898 | 0.054688 | 1.19× |
| **0** | **0.050781** | **0.054688** | **1.08×, overlapping** |

`sgate` moves **1.93×** across that sweep; DeltaNet does not move at all. The `s = 128`
16,384-draw separations quoted above are the same species of number and have not been
re-measured at `floor = 0`; they are withdrawn, not restated.

**This restores nothing.** DeltaNet still reads ≥ `sgate` at `floor = 0`, and the novelty
claim was never deleted on the size of a margin — it was deleted because Signed Dual Attention
([arXiv:2606.04833](https://arxiv.org/abs/2606.04833)) Eq. 1–2 **is** the `sgate` matrix, and
prior art does not care how large a probe margin is. DeltaNet still leads at every `s ≥ 16`
and is still the only arm nonzero at `s = 512`. The deletion stands.

**The residual novelty claim is deleted. This is the sentence that replaces it, and every
qualifier in it is forced by a measurement:**

> On a strictly-causal random-projection influence-Jacobian probe at 2,048 draws, two hops
> and `floor = 1e-6`, this module's signed content-dependent path sum separates from softmax
> absolutely (0.166016 [0.150143, 0.182850] against 0 of 2048, ceiling 0.001800, at context 8)
> and from ParaFormer's signed hop coefficients by a conservative 4.66× at context 16 — but
> DeltaNet's shipping WY matrix is a signed content-dependent path sum too and reads at or
> above this module at every context length from 16 up, including at `floor = 0`, where the
> two are 0.054688 against 0.050781 at context 32. The construction is not new: Signed Dual
> Attention arXiv:2606.04833 Eq. 1–2 is this module's own matrix. Every signed arm decays with
> context; the exponents on record are `floor = 1e-6` readings and are withdrawn.

Nothing in this repository is novel attention. What is left is an operator with a
Lean-verified nilpotent resolvent and a kernel that runs.

**A bounded receptive field was published as the one regime where the property does not
dilute. That is WITHDRAWN. What survives from the same table is round 3's mechanism,
corrected.** Round 3 wrote that the `1/s` decay is the third token's
share of the two-hop sum, "`1/s` by construction". That sweep moved `j = s/4` and `c = s/2`,
so the number of intermediates between `j` and `i` grew with `s` at the same time as every
softmax row normalizer did — two channels, one measurement. Pin the offsets instead
(`j = i−4`, `c = i−2`) so the path count between `j` and `i` is a constant 3 at every context
length, and the two channels come apart.

`hops = 2`, seed 0, `i = s−1, j = i−4, c = i−2`, exact Clopper-Pearson 95%
(`python scale/window_sweep.py --kinds sgate --n 2048 --sizes 32 128 512 2048` and
`--kinds deltanet paraformer softmax --n 1024 --sizes 32 128 512`). `sgate` at 2048 draws,
the rest at 1024:

| s | `sgate` w=8 | `sgate` global | DeltaNet w=8 | DeltaNet global | ParaFormer w=8 | ParaFormer global | softmax |
|---|---|---|---|---|---|---|---|
| 32 | 0.124023 [0.110050, 0.139083] | 0.021484 [0.015653, 0.028735] | 0.074219 | 0.074219 | 0.019531 | 0.002930 | 0.000000 |
| 128 | 0.133301 [0.118869, 0.148791] | 0.001953 [0.000532, 0.004993] | 0.076172 | 0.076172 | 0.031250 | 0.000000 | 0.000000 |
| 512 | 0.108398 [0.095258, 0.122671] | 0.000000 [0, 0.001800] | 0.065430 | 0.065430 | 0.019531 | 0.000000 | 0.000000 |
| 2048 | 0.120605 [0.106807, 0.135499] | — | — | — | — | — | — |

Three things fall out of that table and only one of them was expected.

**The windowed rate is flat across a 64× growth in context — and that is WITHDRAWN as
evidence of anything.** Every windowed `sgate` interval overlaps every other one out to
`s = 2048`, and three separate things are wrong with reading a non-vanishing regime off it.

*It is true by construction.* At `w = 8` with `j = i−4` and `c = i−2`, `out_i` depends only on
tokens `[i−16, i]`. Holding a fixed 20-token tail and varying `s`, the live row slice
`A[i, i−8:i]` is identical to **~7 significant figures** at `s = 32/128/512/2048`. The
measurement had no freedom to come out any other way.

*Its test never bound anything.* `::test_a_bounded_receptive_field_holds_the_separation_that_global_attention_dilutes`
was GREEN on first appearance and **never once RED**. In a campaign whose whole method is to
write the falsifier first, a test that was never red is not evidence.

*The decisive control was never run, and it kills the claim.* Every windowed number here is at
`w = 8`. Sweeping the window width at a fixed `s = 512` gives **0.0996 / 0.0371 / 0.0156 /
0.0078 / 0.0039 / 0.0000** for `w = 8/16/32/64/128/256` — a log-log slope of **−1.159
(R² 0.9933) in window width**, statistically indistinguishable from the −1.437 decay in
context length. **The property does not survive a bounded receptive field. It survives
`w = 8`.** And the sentence that used to stand here — that this is "the standard bounded field
of Mistral, Gemma and gpt-oss" — was **false as written**: those windows are 4096, 4096 and
128. At the smallest of the three the sweep already reads 0.0039, and by `w = 256` it reads
0.0000.

**But the cause is the row normalizer, not the path count.** At a FIXED path count of 3,
`sgate` on a global row still falls 0.021484 → 0.001953 → 0. A third token cannot change
which intermediates exist — it changes the softmax denominator that every entry of the row
divides by, and that denominator sums over `s` tokens. `test_claim_the_context_decay_is_the_path_count_and_not_the_row_normalizer`
is RED on exactly that.

**DeltaNet is bit-for-bit identical windowed and unwindowed** — 0.074219 / 0.076172 /
0.065430 either way. Its entries `−β_i (k_i · k_j)` have no row normalizer, so there is
nothing for the context to dilute. That is the same fact as its −1.107 slope and its win at
every `s ≥ 16`: the arm without a global denominator is the arm that keeps the property.

What is left of that table is a fact about normalizers, not about windows. Windowing lifts
every normalized arm (ParaFormer 0.002930 → 0.019531 at `s = 32`), leaves softmax at exactly
zero, and does nothing at all for the arm that never had a normalizer. It buys no regime,
because the lift is a function of the window width and dies with it at −1.159. (Every rate in
that table is `floor = 1e-6` and none of them has been re-measured at `floor = 0`.)

`tests/foreman/test_paraformer_ratio_across_context.py::test_a_bounded_receptive_field_holds_the_separation_that_global_attention_dilutes` — GREEN, cpu and cuda, **and unbound**: green on first appearance, never red, asserting something true by construction at `w = 8`. It stays in the suite as a regression guard and is cited here as evidence of nothing.
`tests/foreman/test_paraformer_ratio_across_context.py::test_claim_the_context_decay_is_the_path_count_and_not_the_row_normalizer` — RED, cpu and cuda

`tests/foreman/test_paraformer_ratio_across_context.py::test_claim_the_signed_content_dependent_path_sum_is_not_already_deltanet` — RED, cpu and cuda
`tests/foreman/test_paraformer_ratio_across_context.py::test_claim_the_separation_from_signed_hop_coefficients_survives_context` — RED at `s = 128`, cpu and cuda
`tests/foreman/test_paraformer_ratio_across_context.py::test_the_round_4_arms_reproduce_their_published_counts` — calibration

`tests/foreman/test_signedness_is_not_new.py::test_claim_no_published_attention_reaches_a_negative_influence_jacobian`

And the specific number **−9.000000e-01 is −ρ exactly**, which is reachable only at `A[1,0]`
— the row with a single predecessor, where `A[1,0] = ρ·w/|w| = ±ρ` is a frozen constant with
gradient **exactly 0.0**. Over 24 draws, 14 of 14 entries equal to −ρ sat at (1,0); the most
negative entry anywhere else was −8.845515e-01. The headline is read off the one entry no
training step can move.

`tests/foreman/test_signedness_is_not_new.py::test_claim_an_influence_entry_of_exactly_minus_rho_can_come_from_a_free_row`
`tests/foreman/test_l1_normalizer_obstruction.py::test_claim_the_first_attention_row_is_trainable`

---

## Verified core

Lean 4.7.0 + mathlib, `lake build CEQ` exit 0, **zero `sorry`**, and `#print axioms` on every
theorem lists only `[propext, Quot.sound, Classical.choice]` — **no `sorryAx`**. **27 theorems**
across five modules; the seven below are the ones load-bearing for claims in this README, and
`MODEL_CARD.md` covers `CEQ.Refcount`. `tests/chase/test_lean_refcount_binding.py` runs
`#print axioms` on the load-bearing theorems in one `lean` subprocess and calibrates that check
against a deliberate `sorry`, so a rename, a deletion or a broken build fails here.

| theorem | statement |
|---|---|
| `CEQ.Nilpotent.pow_card_eq_zero` | strictly lower-triangular ⇒ `A^n = 0`, over any `CommRing`. **No sign hypothesis** — which is what licenses dropping non-negativity for free. |
| `CEQ.Nilpotent.occupancy_is_exact_inverse` | the causal resolvent is a terminating finite sum and exact. No certificate, no Perron vector, no truncation error. |
| `CEQ.Nilpotent.one_not_nilpotent` | negative control: non-strict lower-triangular is nilpotent at no power, so `.tril(-1)` cannot be weakened to `.tril(0)`. |
| `CEQ.Contraction.weighted_contraction` | Perron certificate ⇒ contraction in the weighted sup norm. |
| `CEQ.Contraction.expander_expands_l2` | `!![1,0;1,0]` is row-stochastic with `σ_max = √2`, refuting the σ_max form of the contraction claim. |
| `CEQ.Occupancy.occupancy_telescope` | `(1−A)·∑_{k<N}Aᵏ = 1−Aᴺ`, over any ring. |
| `CEQ.OrbitBound.orbit_error_bound` | `err ≥ n − m` with injectivity explicit. |

`tests/w3b/` gates the build, checks for `sorry` with a **calibrated** detector, and greps
`ceq/nonnormal.py` for `.tril(-1)` so the proof and the code cannot drift apart silently.

---

## What else survived its falsifier

**Eviction forgets; gating does not.** Perturbing a token the gate crushed moves the settled
state by 2.154868e-05 under post-softmax gating and **0.000000e+00 in 24/24 draws** under
eviction. A gate cannot touch the softmax denominator, so a crushed token keeps its share of
every survivor's normalizer forever. Row-sum deficit: gating 2.307863e-03, eviction
1.110223e-16. `tests/w3/test_w3_eviction.py`

**Multi-zoom summarises rather than drops**, and that is the whole difference: 1622× / 44.7×
/ 1.67× better than dropping at the same geometry and budget; beats sliding-window+sinks
**3/3** where a dyadic schedule lost **0/15**; builder costs **0.036×** the attention it
feeds, against 53× for the dyadic version. Θ(N log N) at R² = 0.999833. **36.4%**
accelerator utilization (10.23 TFLOP/s), RTX 4060 Laptop.

**The hop cache makes α > 0 decodable** at K attention rows per token and K cache slots,
combining a LIFO scope stack (lifetime), a disjoint-set H0 island partition with min-index
representatives (granularity), and free-face admissibility. `tests/w9/`

---

## Wins and losses

Kept together deliberately. Sixteen iterations produced more deletions than results, and the
deletions are the more useful half — each one is a road nobody has to walk again.

### Wins

| # | result | evidence |
|---|---|---|
| 1 | Seven Lean theorems, `lake build` exit 0, **zero `sorry`** | `tests/w3b/`, calibrated `sorry` detector |
| 2 | `pow_card_eq_zero` stated over `CommRing` with **no sign hypothesis** — written two iterations before it was needed, and it is what let non-negativity be dropped for free | `lean/CEQ/Nilpotent.lean` |
| 3 | **Eviction forgets; gating does not.** Perturbing a crushed token: gating 2.154868e-05, eviction **0.000000e+00 in 24/24**. Row-sum deficit 2.307863e-03 vs 1.110223e-16 | `tests/w3/` |
| 4 | **Multi-zoom summarises rather than drops** — 1622× / 44.7× / 1.67× over dropping; beats window+sinks **3/3** where a dyadic schedule lost **0/15**; builder **0.036×** the attention against 53×; Θ(N log N) at R²=0.999833 | Chase, round 2 |
| 5 | Kernel meets C5 at **36.4%** utilization (10.23 TFLOP/s) | Chase, round 2 |
| 6 | α=0 is **bitwise** stock attention on a real checkpoint; generation character-identical on the decode path too | `tests/w8/` |
| 7 | Hop cache decodes at K rows/token; prefill↔decode agree to **one ULP** (1.898657e-16) | `tests/w9/` |
| 8 | **Content-conditional sign 0.1641 against softmax's exactly 0.0000** at `s = 8`, `floor = 1e-6`, on an instrument that reproduced its own published calibration numbers first — and whose floor sensitivity was later found and published against its own headline | `ceq/bench.sign_flip_rate` |
| 9 | Campaign **1.3124 → 1.0334**, four separable obstructions each measured, not guessed | `DONE.md` iterations 13–16 |

### Losses

| # | what died | how |
|---|---|---|
| 1 | **R1** settle-don't-glance | its falsifier does not falsify — an affine update moves 0.471 of the way on pass 1; Howard beats value iteration 7.9×/21.4× |
| 2 | **R2 as max-plus** | the star equals APPNP of its own greedy policy at **9.95e-14**; gradients at 1.65e-08 |
| 3 | **R3** amplify-few-crush-many | the baseline is **3.94× sharper** where R3 required it flatter |
| 4 | **R4** compress-by-forgetting | a *retained* filler scored −0.11224 against crushed −0.11446 — gap +0.00222 against 0.25 required |
| 5 | **R6** local/global agreement | the averaging baseline **wins by 0.0793 AUROC**; least squares *localizes* a contradiction rather than averaging it away |
| 6 | **W2** non-normality | departure from normality identical (1.40679/1.40682/1.40972) across a **4.08× ratio spread** — not the variable |
| 7 | **W7** Nash/QRE stance | beat the signed arm in **1/5 seeds**; the τ=4.0 optimum was a seed-0 artifact |
| 8 | **H7** identity double-count | **my own reasoning, wrong** — the block residual carries the pre-projection state, so `v` was never a duplicate. Removing it cost 1.0968 → 1.2887 |
| 9 | **Intervention generalization** | all three arms **worse than a constant predictor** on the held-out composition |
| 10 | **Signedness is not new** | SimA (arXiv:2206.08898, 2022) states the claim in its own text; also DiffAttn, Cog Attention, FAGCN, SignGT |
| 11 | **Nilpotency is not new either** | DeltaNet inverts a strictly-lower-triangular *signed* matrix; sglang's Kimi kernel documents `(I+L)⁻¹=(I−L)(I+L²)(I+L⁴)(I+L⁸), L¹⁶=0` |
| 12 | **The headline number was a dead entry** | −9.000e-01 is `−ρ` exactly, always at `A[1,0]` (14/14 draws), and `A[1,0]` has gradient **exactly 0.0**. Most negative anywhere else: −8.845515e-01 |
| 13 | **min-influence was the wrong probe** | a real block's `W_v`/`W_o` already supply minus signs, so "can be negative" separates nothing. The property that separates is content-conditional sign |
| 14 | **Published memory figures described inference** | training fwd+bwd is 2.65× / 4.45× / **8.06×** SDPA at seq 512/1024/2048 |
| 15 | **Five of my own instruments were internally consistent and externally wrong** | a parity test comparing my path against my own control (ppl 89,400 vs 1.667); a `sorry` detector firing on the sentence "No `sorry` anywhere"; an eviction test asserting bitwise in a window where it is provably false; a corpus split by literal value that measured embedding coverage; an uncalibrated sign-flip probe returning 0.0000 everywhere |

### Still open

**Parity has not been shown to scale.** Everything above is 3.3M parameters, one dataset, one
sequence length, one budget. The user's standard is that parity at 3.3M is false unless it
survives at 300M, and the original operator already showed the pathology — its gap *widened*
from 1.2191× to 1.337× with more training. That measurement is running and is the thing that
decides whether win #9 stands.

**The win condition was never met, and two evaluations were never attempted at all.** ARC-AGI
has **never been scored**. A Turing-style evaluation has **never been attempted — no file for
one exists in this repository**. Nothing has ever run above **3.65M parameters**; the 300M
gate is priced in the section above at 129.7-257.2 A100-hours from `ceq/sizing.py`, and the
"912 A100-hours" this project's checklists carried has **no derivation anywhere in this
repository**. Intervention generalization has every arm worse than a constant predictor. (`lean/CEQ/Refcount.lean` was on this list as cited-by-nothing and
gated-by-nothing; it now has both — see `MODEL_CARD.md` and
`tests/chase/test_lean_refcount_binding.py`.)

**The whole test suite has never completed.** `python -m pytest tests/ -q` reached ~9% in 15
minutes on CPU and is estimated at over three hours; it has not finished once in this campaign.
**No total pass/fail count exists — do not quote one.** What is verified is that **1,036 tests
collect** ([RUN] `python -m pytest --collect-only -q tests/`, 89.34 s, measured 2026-08-25) and
that every test name cited in this README resolves against that collection, enforced by
`tests/w11/test_w11_claims_resolve.py`. This README has carried 794, 809 and 829 at different
times; the count moves as tests land, so it is dated rather than asserted.

## Requirements, and how each ended

| # | requirement | verdict | evidence |
|---|---|---|---|
| R1 | settle, don't glance | **DELETED** | the falsifier does not falsify — an affine update moves 0.471 of the way on pass 1; Howard beats value iteration 7.9× CPU / 21.4× CUDA |
| R2 | consequence, not similarity | **DELETED as max-plus** | the star equals APPNP of its own greedy policy at 9.95e-14; gradients agree at 1.65e-08 |
| R3 | amplify few, crush many | **DELETED** | the baseline is 3.94× *sharper*, where R3 required it flatter |
| R4 | compress by forgetting | **DELETED** | a retained filler scored −0.11224 against crushed −0.11446 — a gap of +0.00222 against 0.25 required |
| R5 | multi-zoom reading | **ALIVE** | above |
| R6 | local/global agreement | **DELETED** | the averaging baseline wins by 0.0793 AUROC; least squares *localizes* a contradiction rather than averaging it away |

---

## The diagnostic, for somebody else's operator

The most transferable thing this project built is not the operator. It is the probe that
decides whether a non-standard attention operator has **content-conditional sign** — whether
a *third* token can decide if token j helps or hurts token i. Fixed value and output
projections cannot do that, and a non-negative `a_ij` can only rescale, so this is the one
property that separates a signed operator from softmax at block level rather than at matrix
level. If you are building a non-standard attention operator, this is a two-line answer to
whether it is actually different.

```bash
python -m ceq.diagnose                 # CPU, no GPU, no network, ~2 minutes
python -m ceq.diagnose --fast          # ~25 seconds
python -m ceq.diagnose --json          # same numbers, machine-readable
```

It prints three things and refuses to print the first without the second:

1. `sign_flip_rate` for every operator in this repository at depth 1 and 2, with **softmax
   as the control reading exactly 0.0** — so the reader does not have to supply a baseline
   from memory.
2. **The context-decay curve of that rate**, `s = 8 … 128`, with its fitted log-log slope.
   A single rate at a single context length is precisely the number that misled this
   project for three rounds: 0.1641 was measured at the probe's default `s = 8` and treated
   as a property of the operator. It is a property of the operator *at s = 8*, and it decays
   with context. At 1024 draws the tool reproduces the published curve entry for entry —
   0.17480 / 0.08887 / 0.02637 / 0.01172 / 0.00391 — at the shipped `floor = 1e-6`. **Those are
   `floor = 1e-6` rates, and the `s^-1.389` exponent the tool prints as `published_slope` is
   withdrawn** (see "Limits, first"). `ceq/diagnose.py` still hard-codes it and
   `tests/cameron/test_diagnose_package.py` still pins it, so the tool and this README disagree
   until the sweep is re-run at `floor = 0`. Reading a rate without its floor is the fifth
   instrument error this project has published against itself.
3. **The interventional code corpus with its scorer.** CPython is the oracle; there is no
   answer key in this repository. The held-out cell requires composing a subtraction with a
   negation — two sign flips seen separately in training and never together. The scorer is
   normalized by the target's own spread, so **1.0 is exactly predict-the-mean** and that
   line is printed next to the arms rather than left for the reader to remember.

Bound by `tests/cameron/test_diagnose_package.py`, which runs the command in a subprocess
with `CUDA_VISIBLE_DEVICES` emptied, so a hidden GPU dependency fails here rather than at a
reader's machine.

---

## The shippable package, and the one command that checks it

`ceq/hf/` is a self-contained `transformers` architecture — `configuration_ceq.py` plus
`modeling_ceq.py`, `auto_map` wired, `trust_remote_code=True` on load. Its default is the
operator that reached parity, `sgate` at **rho=1.5, lam=0.10, hops=2**, checked bitwise
against `ceq/lm.py` where the 5-seed run actually ran. `signed` — `rho·w/Σ|w|`, the arm that
measured 1.337× — stays selectable as the negative control and is no longer the default.

```bash
python -m ceq.hf.smoke        # CPU only, no GPU, no Triton, no network
```

Seven checks, each standing in for a failure this project has actually produced: no triton
import in either shipped file; the shipped operator is bitwise the parity operator; forward,
backward and `generate()`; `save_pretrained` → `AutoModelForCausalLM(trust_remote_code=True)`
round-trip identical; `lm_head` is materialized rather than stranded on the meta device;
`alpha = 0` is bitwise the model's own attention with identical greedy generation; and an
unrecognised attention mask is refused. It prints the version matrix it ran on and the price
list from `modeling_ceq.COSTS` — the memory table, the 1.32× wall clock, the forward-only
kernel, the missing KV cache, and the `s^-1.389` decay of the distinguishing property —
because a green check that does not say what it was green against is not evidence. **That last
figure is a `floor = 1e-6` reading and is withdrawn; `ceq/hf/modeling_ceq.py::COSTS` still
carries it, and it must be corrected before this package is pushed to the Hub.**

Four silent failures were removed in the process, each now raising or counted:

| was | is |
|---|---|
| an additive mask of `-1e4` returned an operator **bitwise equal to the unmasked one** while reporting as masked | `ValueError` naming the convention, plus a `STATS` counter of masked key positions |
| a chunked prefill (`q_len=4`, `kv_len=16`, no mask) attended to future keys — query 0 moved **1.8397** when key 15 was perturbed | `NotImplementedError`; a 1-token decode step still passes |
| `tie_word_embeddings=True` stranded `lm_head.weight` on meta and the forward returned **1.0054e+30** with nothing raised | `RuntimeError` naming the meta device and the rollback |
| `truncation_bound(1.5, 2)` returned **−6.75**, a negative error bound, at the shipped operating point | `ValueError`; there is no geometric bound at rho ≥ 1 and the docstring says so |

## Reproduce

```bash
python -m ceq.hf.smoke                        # the shippable package, CPU only
python -m pytest --collect-only -q tests/     # 1,036 collect, 89.3 s (2026-08-25)
python -m pytest tests/ -q                    # every falsifier -- >3 h, has NEVER completed
python -m ceq.diagnose                        # the CPU sign-flip diagnostic
python -m ceq.capability                      # COGS + SCAN, both arms, published refs
python -m pytest tests/w10/ -q -m slow        # from-scratch softmax vs signed
cd lean && lake build CEQ                     # the proofs
```

Every test parametrizes over `cpu` and `cuda`, skipping cuda when unavailable.

Machine for every number above unless stated otherwise: Windows, Python 3.11.9, torch
2.5.1+cu121, **RTX 4060 Laptop, sm_89, 8.0 GiB, 24 SMs**, triton 3.7.1. A utilization figure
from a laptop part does not transfer to an A100.

## License

MIT.
