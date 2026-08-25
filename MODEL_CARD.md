---
license: mit
library_name: transformers
pipeline_tag: text-generation
language:
  - en
datasets:
  - roneneldan/TinyStories
tags:
  - attention
  - custom_code
  - negative-results
  - research
---

# ceq — a signed causal path-sum correction for attention

> ## The operator does not work. This is the negative result and the harness that produced it.
>
> **The pre-registered kill fires on the operator the module ships.** Measured,
> `PROTOCOL: SCALING`, `c` drawn from the pivot set: sign-flip rate
> **0.16511 / 0.02732 / 0.00000 / 0.00000** at s = 8/32/128/512, slope
> **−1.298** against a bar of **−0.3**. **Zero flips at s ≥ 128.** Pivot routing
> makes it *worse* than dense, not better (dense reads −1.088).
>
> **The measurement that once said otherwise was measuring a different
> operator.** `_causal_tgate_operator` carried every headline — a flat
> **+0.0270** slope across a 256× context growth, a **61×** separation at
> s=1024 — and appears **nowhere in the shipped path**, which uses
> `ceq_operator` and `sgate_operator`. The arm names hid it:
> `scale/pivot_probe.py::ARMS` never contains the string `"tgate"`, because the
> arms are called `pivot_signed` and `dense_signed` and `build_arm` maps **both**
> to it. *The name describes a property, not an implementation.*
>
> **The Lean core is sound; it does not certify what ships.** 27 theorems,
> `lake build` exit 0, zero `sorry`, no `sorryAx`. `pow_card_eq_zero` is
> **confirmed against the shipped tensor** — `A^n = 0` at exactly
> `0.000000e+00` for s = 16/64/128/512. But `occupancy_is_exact_inverse` is
> stated at **N = n**, and the module truncates at **hops = 2..4** where
> `‖A^hops‖` is **0.880500** at s=128 (hops=2; **0.882030** at hops=4).
> **The 1.471448 previously printed here was struck by the iteration-35 audit —
> not reproducible at any of 1,800 settings.** **The theorems certify a computation the
> module does not perform**, and `truncation_bound` refuses at the shipped
> `rho = 1.5`, so no bound stands in either.
>
> **What is worth reading this repository for:** the negative result, the Lean
> core as mathematics, and a falsification harness that caught **seventeen** of
> its own broken instruments — including the one that invalidated its own
> headline. Full verdict in [`PROGNOSIS.md`](PROGNOSIS.md).


## Model Details

- **Type:** a `transformers`-registerable attention function plus a small causal-LM
  architecture (`CEQForCausalLM`), byte-level vocabulary. **No trained weights are published
  here.**
- **Operator:** `out = stock_attention(q, k, v) + Σ_{h=1..K} (α A)^h v`, `A` strictly lower
  triangular and signed; shipped default `sgate` at `rho=1.5, lam=0.10, hops=2`.
- **Sizes ever trained:** 3.3M and 3.65M parameters. **Nothing above 3.65M has ever run.**
- **Requires `trust_remote_code=True`.** Loading executes Python from this repository on your
  machine. Read `modeling_ceq.py` first and pin `revision=` to a commit hash so a later push
  cannot change what runs.
- **License:** MIT. **Contact / issues:** the source repository.
- **The headline result is a loss.** Read "Limits, first" before anything else.

**This card describes a MODULE, not a trained checkpoint.** There are no weights here. It
is a `transformers`-registerable attention function that adds one term to an existing model's
attention, and it is measured against that model rather than presented on its own. The one
capability comparison ever run at matched parameters went against it.

```
out = stock_attention(q, k, v) + Σ_{h=1..K} (α A)^h v      A strictly lower triangular, SIGNED
```

Every number below names the test that produces it. Reproduction commands are at the end.

---

## Limits, first

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

**Round 4 corrects this.** That sweep moved `j = s/4` and `c = s/2`, so the intermediate
count and every softmax row normalizer grew with `s` together. Pinned at `j = i−4, c = i−2`
the path count is a constant 3, and `sgate` on a global row **still** falls 0.021484 →
0.001953 → 0.000000 over `s = 32/128/512`, while DeltaNet — signed entries, no row
normalizer — reads 0.074219 / 0.076172 / 0.065430 and is bit-for-bit identical under a
sliding window. The dilution is the **softmax row denominator summing over `s` tokens**, not
the two-hop share. (These are `floor = 1e-6` rates and have not been re-measured at
`floor = 0`. A bounded receptive field was once published here as a regime where the property
survives; that is **withdrawn** — the rate decays in window width at −1.159, R² 0.9933, as
fast as it decays in context.) `tests/foreman/test_paraformer_ratio_across_context.py::test_claim_the_context_decay_is_the_path_count_and_not_the_row_normalizer` — RED.

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


**PARITY was reached only after four operator knobs were tuned, and only at 3.3M
parameters.** The operator as originally shipped -- raw logits, row-L1 normalized -- loses
to softmax and the gap WIDENS with training. The operator that reaches parity is a
difference of two softmaxes (`rho=1.5, lam=0.10, hops=2`) at a learning rate swept for both
arms: median ratio **1.0334**, spread 1.0199-1.0413, under the 1.05 bar on **5/5 seeds**,
parameters exactly equal at 3,319,296, wall-clock **1.32x**. The signed arm had four knobs
tuned (rho, lam, hops, lr); softmax had one (lr), because it has no operator knobs. That
asymmetry is real and is stated rather than buried. Whether parity survives a longer budget
is UNTESTED -- the widening measured below was on the original operator.

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

**At α = 0 this IS stock attention, bitwise.** Therefore the gate can never be superior to
attention — it is attention *plus a term*. That is by construction, not an accident.
`tests/w8/test_w8_real_model.py::test_alpha_zero_is_bitwise_stock_attention`

**Intervention generalization was NOT achieved.** On a held-out composition of two sign
flips: attention 5.8198, APPNP 4.2107, signed 2.6151 OOD NRMSE, median of 5 seeds. NRMSE
1.0 is predict-the-mean, so **all three arms are worse than a constant predictor**. The
signed operator fails 1.6× less badly. It does not generalize.
`tests/w4/test_w4_intervention.py::test_signed_arm_generalizes_rather_than_merely_degrading_less`

**Five of six original requirements were deleted by their own falsifiers** — R1, R2-as-max-plus,
R3, R4, R6. Only R5 (multi-zoom) survived. `README.md` has the table.

**The Triton kernel is forward-only.** No backward pass exists; training must use the CPU
reference, which is differentiable and gradcheck-clean.

**Kernel memory is worse at every length**, 1.31×–1.94× SDPA. It wins wall-clock only above
seq 8192 (7.48× at 32768); at seq 1024 it is **0.06×**, far slower.

**That 1.31×–1.94× is INFERENCE, forward-only, seq 128–512, and it is not the number a
trainer pays.** Training forward+backward against SDPA at identical parameters, fp32:

| seq | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| signed / SDPA peak memory | 1.44× | 1.74× | **2.65×** | **4.45×** | **8.06×** |

A 0.5B model at seq 2048 batch 1 needs **36.31 GiB** against a 33.53 GiB A100-40GB budget and
20.25 GiB on an L4, so gradient checkpointing (0.379×, 1287.5 → 487.7 MiB at seq 1024, loss
identical to 1e-4) is a requirement rather than an option. The same table ships inside the
package as `ceq.hf.modeling_ceq.COSTS`, because a cost recorded only in a markdown file does
not reach the person who downloads the checkpoint.

**α > 0 cannot decode without the hop cache**, and the hop cache is exact forward but lossy
backward — eviction is bitwise only if nothing dependent has run yet.
`tests/w9/test_w9_hopcache.py::test_eviction_is_not_retroactive`

**No scaling-law claim is made.** No published ~30-architecture inversion study exists; the
nearest real work is Tay et al. ([arXiv:2207.10551](https://arxiv.org/abs/2207.10551)), ten
architectures at 2.9B dense, no intercept-vs-slope decomposition.

**Every number was measured on one machine** — Windows, Python 3.11.9, torch 2.5.1+cu121,
RTX 4060 Laptop, sm_89, 8.0 GiB, 24 SMs, triton 3.7.1. A utilization figure from a laptop
part does not transfer to an A100.

**What has never been done, stated so nobody has to ask.** ARC-AGI has **never been scored**.
A Turing-style evaluation has **never been attempted — no file for one exists**. Nothing has
run above **3.65M parameters**. `python -m pytest tests/ -q` has never completed (~9% in 15
minutes on CPU, estimated >3 h), so **no total pass/fail count exists — do not quote one**;
what is verified is that **955 tests collect** in 9.6 s (measured 2026-08-25; this count
GROWS as tests are added, so re-measure rather than trusting it) and every test name cited in the
README resolves against that collection, enforced by `tests/w11/test_w11_claims_resolve.py`.

**This package requires `trust_remote_code=True`.** Loading it executes Python from this
repository on your machine. That is a real security decision and it is yours to make: read
`modeling_ceq.py` before loading it, and pin a `revision=` so a later commit cannot change
what runs.

---

## What it adds, and how much of it is already published

**A token can reduce another token's contribution.**

| operator | min influence Jacobian `∂out_i/∂v_j` |
|---|---|
| softmax attention | **+0.000000e+00 — exactly zero** |
| APPNP | **+0.000000e+00 — exactly zero** |
| max-plus Kleene star | **+0.000000e+00 — exactly zero** |
| **this module** | **−9.000000e-01** |

Not a tuning difference. The Kleene star of a non-negative matrix has a non-negative
influence Jacobian **in any ordered semiring** — measured exactly zero over 40 max-plus
instances and 160 APPNP kernels. Softmax, APPNP and max-plus are all inside that class, so
none of them can express a negation. Dropping non-negativity is the only escape.

`tests/w6/test_w6_attention.py::test_signed_operator_reaches_negative_influence`, calibrated
by `::test_the_nonnegative_control_is_stuck_at_exactly_zero`.

**Published attention is already outside that class.** Reimplemented from their defining
equations and measured on the same instrument: SimA (arXiv 2206.08898, 2022), which
L1-normalizes Q and K instead of applying softmax, reaches **−3.929583e-01** and states the
property in its own text; Differential Transformer (arXiv 2410.05258) is negative by
subtraction; DeltaNet's UT transform (arXiv 2406.06484 Eq. 10) inverts a **signed** strictly
lower triangular matrix, min entry **−10.875878**. Signedness alone is a rediscovery.

**And this module's own matrix is published.** Signed Dual Attention (arXiv:2606.04833,
v1 3 Jun 2026) Eq. 1–2 is `A⁺ = softmax(QKᵀ/√d_k)`, `A⁻ = softmax(−QKᵀ/√d_k)`,
`SDA = (A⁺ − A⁻)V` — the `sgate` matrix at `lam = 1`, single-hop.

**The signed-and-multi-hop pairing is occupied too, and the number that said otherwise was
an instrument bug.** `ceq/bench.py::sign_flip_rate` built the hop-coefficient vector for its
`paraformer` arm and never referenced it, so the arm ran plain single-hop softmax; round 3's
"ParaFormer 0 of 2048 draws, Clopper-Pearson ceiling 0.00146, at least 84×" was a softmax
number. Corrected — `Σ_k γ_k Â^k h`, arXiv:2512.14619 Eq. 9 — on `hops=2`, 2048 draws,
seed 0, `i=s-1, j=s/4, c=s/2`, exact Clopper-Pearson 95%:

| operator | base matrix | s = 8 | s = 16 |
|---|---|---|---|
| this module, `Σ A^k`, `sgate` | signed, content-dependent | 0.166016 [0.150143, 0.182850] | 0.078613 [0.067323, 0.091127] |
| **DeltaNet WY, arXiv 2406.06484 `eq:inverse`** | **signed, content-dependent** | 0.082031 [0.070507, 0.094768] | **0.097656** [0.085139, 0.111339] |
| Cog Attention 2411.07176 / SignGT 2310.11025 matrix | signed, content-dependent | 0.105469 [0.092494, 0.119585] | 0.035156 [0.027607, 0.044071] |
| ParaFormer 2512.14619 Eq. 9, `Σ γ_k Â^k` | non-negative, signed γ | 0.029785 [0.022858, 0.038097] | 0.009277 [0.005595, 0.014450] |
| softmax (control) | non-negative | 0.000000 [0, 0.001800] | 0.000000 [0, 0.001800] |

`T = (I + tril(diag(β) K Kᵀ, −1))⁻¹ diag(β)` is content-dependent, signed per entry, and
strictly triangular hence nilpotent, so its exact inverse **is** a finite path sum. It ships
in `fla/ops/utils/solve_tril.py` and sglang's `chunk_kda_fwd_kernel_inter_solve_fused`.

It does not merely tie: DeltaNet reads at or above `sgate` at every `s ≥ 16` and is the only
arm still nonzero at `s = 512`. **The margins that used to be quoted here are WITHDRAWN.**
Every cell above is `floor = 1e-6` — an ABSOLUTE discard threshold — and the arms are not on
the same scale: `sgate`'s row-L1 is pinned at `rho = 1.5` while DeltaNet's grows to 81.62 by
`s = 512`, so one absolute floor discards far more of `sgate`. At `s = 32`:

| floor | `sgate` | DeltaNet | ratio |
|---|---|---|---|
| 1e-6 — as published | 0.026367 | 0.054688 | 2.07× |
| 1e-9 | 0.045898 | 0.054688 | 1.19× |
| **0** | **0.050781** | **0.054688** | **1.08×, overlapping** |

`sgate` moves **1.93×** across that sweep; DeltaNet does not move at all. The "non-overlapping
1.39× at `s = 32`", the "conservative 1.92× at `s = 128`, 16,384 draws", and the log-log slopes
(−1.107 against −1.437) are all `floor = 1e-6` readings and are withdrawn — not restated at a
smaller value.

**None of that restores a novelty claim.** DeltaNet is still ≥ `sgate` at `floor = 0`, and the
claim was never deleted on the size of a margin: Signed Dual Attention (arXiv:2606.04833)
Eq. 1–2 **is** the `sgate` matrix, and prior art is indifferent to probe margins. **There is no
novel construction in this repository.** What is left is a Lean-verified nilpotent resolvent, a
kernel that runs, and a capability measurement that came back negative.

`tests/foreman/test_paraformer_ratio_across_context.py::test_claim_the_signed_content_dependent_path_sum_is_not_already_deltanet` (RED, cpu and cuda)
`tests/foreman/test_paraformer_ratio_across_context.py::test_the_paraformer_arm_is_not_the_softmax_arm` (GREEN, cpu and cuda)

The number **−9.000000e-01 is −ρ exactly**, reachable only at `A[1,0]`, the
single-predecessor row where `A[1,0] = ±ρ` is a frozen constant with gradient exactly 0.0.
Over 24 draws, 14 of 14 entries equal to −ρ sat at (1,0); the most negative entry elsewhere
was −8.845515e-01.

`tests/foreman/test_signedness_is_not_new.py::test_claim_no_published_attention_reaches_a_negative_influence_jacobian`
`tests/foreman/test_signedness_is_not_new.py::test_claim_an_influence_entry_of_exactly_minus_rho_can_come_from_a_free_row`

---

## Measured cost on a real checkpoint

`Qwen/Qwen2.5-0.5B`, fp32, GQA, 99-token technical passage:

| α | perplexity | vs stock |
|---|---|---|
| stock | 9.2150 | — |
| **0.00** | **9.2149** | **−0.00%** |
| 0.01 | 9.1779 | −0.40% |
| 0.05 | 9.0781 | −1.49% |
| 0.15 | 9.1789 | −0.39% |
| 0.30 | 10.7969 | +17.17% |

Greedy generation at α = 0 is character-identical to stock, on the decode path as well as
the teacher-forced one. **The −1.49% at α = 0.05 is one 99-token sample and is not a
result** — it is recorded because it is what was measured. Perplexity is a sanity check that
the gate is wired correctly; it is not a win condition.

`tests/w8/test_w8_real_model.py::test_the_cost_of_the_operator_on_untuned_weights_is_measured`

---

## Verified core

Lean 4.7.0 + mathlib, `lake build` exit 0, **zero `sorry`**. **27 theorems** across five
modules; the load-bearing one for the operator is `CEQ.Nilpotent.pow_card_eq_zero` — strictly
lower-triangular ⇒ `A^n = 0` over any `CommRing`, with **no sign hypothesis**, which is what
licenses dropping non-negativity for free. `tests/w3b/` gates the build, checks for `sorry`
with a calibrated detector, and greps the Python for `.tril(-1)` so proof and code cannot
drift apart silently.

**`CEQ.Refcount` is the provenance module**, and it is the only one whose two halves are both
the author's own prior work: the indistinguishability floor of `caustic` Theorem 1
([10.5281/zenodo.21997746](https://doi.org/10.5281/zenodo.21997746), formalised here as
`CEQ.OrbitBound`) and the refcount of `foliation`, a KV cache presented as a quotient by
block-aligned prefix. `floor_add_orbits` proves the floor is not merely *bounded* by refcount
data but *is* a sum over refcounts, `n − m = Σ_plaques (refcount − 1)`; `free_face_floor_unchanged`
proves that evicting a refcount-1 plaque moves the floor by exactly zero, and
`shared_plaque_floor_drops` proves the converse, without which "free faces are safe" would be
consistent with everything being safe. Scoring a KV block by removing it and measuring the
change is leave-one-out influence (Cook 1977, Koh–Liang 2017) and is externally owned; these
theorems replace the forward pass with an integer the cache already maintains.

**What that does and does not certify.** `CEQ.Refcount.IsFreeFace` is `refcount f p = 1`, a
fibre cardinality. The shipped `ceq/hopcache.py::HopCache.evict` admits on **scope depth**, not
on a refcount — no refcount is maintained anywhere in the Python. So the theorems certify the
criterion, not this implementation of it.
`tests/chase/test_lean_refcount_binding.py` binds all of the above: it runs `#print axioms` on
the five load-bearing theorems in one `lean` subprocess (a rename, a deletion, a broken build
or a `sorry` each make it exit non-zero), calibrates that check against a deliberate `sorry` in
the same process, re-derives the identity exhaustively over every map from 5 prefixes to 3
plaques, and asserts the refcount predicate is still absent from the Python so this paragraph
cannot go stale silently.

---

## Usage

```python
from ceq import hybrid

hybrid.register()                    # registers BOTH the attention and its mask
model.set_attn_implementation("ceq_hybrid")
```

Registering the attention function **without** its mask makes `transformers` pass
`attention_mask=None` and silently drop causal, padding, packing and sliding-window
constraints. `hybrid.register()` does both.
`tests/w6/test_w6_attention.py::test_both_interfaces_are_registered`

CPU works. `α = 0` is free and bitwise; raise `α` only with the cost curve above in view.

---

## Reproduce

```bash
python -m pytest --collect-only -q tests/     # 955 collect, 9.6 s (2026-08-25)
python -m pytest tests/ -q                    # every falsifier -- >3 h, has NEVER completed
python -m ceq.diagnose                        # the CPU sign-flip diagnostic
python -m ceq.capability                      # COGS + SCAN, both arms, published refs
python -m pytest tests/w10/ -q -m slow        # from-scratch softmax vs signed
cd lean && lake build CEQ                     # the proofs
```

Every test parametrizes over `cpu` and `cuda`, skipping cuda when unavailable.

## Uses

### Direct use

Research on attention operators. Concretely: (1) run `python -m ceq.diagnose` against your own
non-standard attention operator to find out whether it differs from softmax in
content-conditional sign, with softmax printed beside it as a control reading exactly 0.0;
(2) read the deletions — five of six original requirements and one 84× headline died here, each
with the number that killed it, and each is a road nobody else has to walk; (3) reuse the
`floor` warning below before trusting any gradient-sign probe of your own.

**Read the softmax 0.0 correctly — it is a floor, not a small number.** The diagnostic's read
is `d(out_i)/d(v_j)` through `out = v + Av + A²v`, whose influence matrix is `I + A + A²`. For
any entrywise non-negative `A` that matrix is non-negative, so the sign cannot be moved by any
context and the rate is **exactly 0 at every sequence length, by construction**. Post-multiplying
by a fixed `W_v` does not rescue it: a context-independent linear map cannot make a
context-independent sign context-dependent. So a 0.0000 from your operator means "cannot
represent a sign flip at all", never "flat".

**And the flatness of that rate in context is a separate axis from the sign.** The S2 selection
ablation (`scale/s2_probe.py`, `tests/cameron/test_s2_ablation.py`, table in `DONE.md`) measures
both on a magnitude-matched pair — the signed operator against `|A|`, the same matrix with the
sign stripped. Dilution-resistance, read as `E|term_c| / σ(background)`, comes out at
**slope -0.038 signed and -0.046 unsigned** when hop 2 is routed through a fixed set of k
pivots, and **-0.913 signed and -0.930 unsigned** when it is dense. Fixed path count carries
that property and carries it identically with or without signs — an axis Star-Transformer
([arXiv:1902.09113](https://arxiv.org/abs/1902.09113), 2019) already occupies with a
content-blind relay hub. Replacing content-selected pivots with **uniformly random** ones leaves
the sign-flip rate flat at **+0.081**, so content selection is not what holds it up either.
Signedness is the only axis on which the non-negative construction sits at the floor.

### Out-of-scope use

**As the operator the context-stability results describe.** Those results are measured on
`tgate`, a denominator-free matrix that lives in `ceq/bench.py` and **is not in this
release**. This model ships `sgate`. Under the same routed measurement the shipped operator
reads slope **−1.826** where `tgate` reads **+0.027** [RUN, n=1024, k=8, s=8..512] — the
kill bar is −0.3, so the shipped operator fires it by a factor of six. Pivot routing fixes
the path count; it does not remove the row denominator, which is the measured cause of the
decay. Do not read any flat-influence claim as a property of these weights.

**As a replacement for softmax attention.** It measurably is not one: at 3,652,096 matched
parameters it scored 0.0000 against softmax's 0.0293 on COGS generalization, one-sided Fisher
`p = 2.7502788939e-05`, and it was behind in-distribution too. At `α = 0` the gate is bitwise
stock attention by construction, so it is attention *plus a term* and can never exceed what it
wraps.

**As a long-context speedup below seq 8192**, where the kernel is slower than SDPA and uses more
memory — and the distinguishing property decays in context length *and* in window width, so it
is a short-range property either way.

**In production.** There are no weights, no KV cache, no backward pass for the Triton kernel,
and every number was measured on one laptop GPU.

## What this is for, and which niche it might occupy

This is **a negative result shipped with the instrument that produced it and a machine-checked
core.** The novelty claims were withdrawn against prior art this project found itself — SimA
(arXiv:2206.08898), Signed Dual Attention (arXiv:2606.04833, the `sgate` matrix itself),
DeltaNet (arXiv:2406.06484), ParaFormer (arXiv:2512.14619), RetNet, SignGT. What is left that
audits clean: a Lean core with zero `sorry` and no `sorryAx`; an instrument that reproduces
itself bit-identically; a complete 84× retraction; every cited test name resolving against
`pytest --collect-only`; the cost table shipping inside the package.

**The separation is not a capability claim.** For non-negative operators with **linear** value
paths a third token cannot flip the sign of `∂out_i/∂v_j` — softmax reads exactly 0.0000 at
depth 1 and 2. Put **one GELU** between two softmax layers and the flip returns. So the claim
that survives is "softmax needs a nonlinearity and the depth to route through it, where this
operator has it in one layer" — depth and parameter efficiency, not capability, and visible only
where parameters are scarce.
`tests/cameron/test_negation_is_the_axis.py::test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`

**The one candidate niche, untested:** minimal-pair NPI licensing under negation — BLiMP's
`sentential_negation_npi_scope` and `only_npi_scope` — inside the BabyLM strict-small regime
(10M words), because it is scored on the model's own next-token probabilities, the licensor and
the NPI sit inside one clause, and small from-scratch models are the norm there. Published
accuracies on `sentential_negation_npi_scope` are 5-gram 45, LSTM 23, Transformer-XL 53, GPT-2
95, human 81 ([BLiMP, TACL 2020](https://aclanthology.org/2020.tacl-1.25/)), so the failure is
not saturated and there is room to be measured. **It has never been run here, and the prior is
against it:** the one matched-parameter capability comparison this project ever ran went to
softmax. `README.md` carries the pre-registered kill.

## Training Details

Byte-level, `roneneldan/TinyStories` streamed (or a local text file), 90/10 contiguous split so
validation text is unseen. The published parity run is 600 steps, batch 32, seq 128, d=256,
4 layers, 5 seeds, lr swept for both arms. The COGS/SCAN capability runs are in
`ceq/capability.py` with results in `results/capability.json`. The Triton kernel has **no
backward pass**; training uses the pure-torch path, which is why a free-tier T4 can run it.

## Environmental Impact

Every number in this card was produced on one RTX 4060 Laptop (8 GiB). No large run has been
performed: **nothing above 3.65M parameters has ever been trained.** For scale, a 300M matched
pair at a Chinchilla budget is priced in `README.md` at 41–47 A100-hours for the softmax control
and 130–257 A100-hours for this operator, from `ceq/sizing.py` plus a measured step ratio — a
cost worth knowing before it is spent rather than after.

## Citation

No paper. Cite the repository and the commit hash you loaded.

## License

MIT.
