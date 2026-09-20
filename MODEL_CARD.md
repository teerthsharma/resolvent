---
license: apache-2.0
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
> **The Lean core is sound; it does not certify what ships.** **134 theorems +
> 41 lemmas**, counted by `python scripts/lean_count.py` (comments stripped,
> `@[attr]` prefixes admitted), 2026-09-11.
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
> headline. Full verdict in [`99777ab:PROGNOSIS.md`](https://github.com/teerthsharma/resolvent/blob/99777ab0f529702c62b49bf05d87a2eea3c116b5/PROGNOSIS.md).


---

# v17-K — THE SHIPPED MODEL (Q3): CARD TEMPLATE, LIMITS FIRST

> **THIS IS A TEMPLATE, NOT A FINISHED CARD.** Every `⟨SLOT …⟩` below is a hole a
> finished run fills with a measured number. **A slot is not a prediction and
> must never be read as one**, and this section is not publishable while any slot
> is unfilled. Filed as the documentary consequence of **RULING 2**
> (`V17K_RULINGS.md` §A2) and **RULING 3** (§A3).
>
> **Checked 2026-09-10: none of the twelve slots below are fillable yet.** Every
> one of them is downstream of the Q3 training run or the floor measurement
> `V17K_RULINGS.md` §A1.2 owes, and neither has produced a number:
> `COSTS.md` §4 still carries RULING 1's training noise floor as "☐ NOT YET MEASURED",
> and the one Kaggle attempt on record
> (`results/kaggle_v17k_output/ceq-v17-k.log`) halted at
> `GATE FAILED: a code source is configured` before any training step ran. Slots
> stay `NOT MEASURED` rather than carrying an invented number.

**Scope, before anything else.** This section describes the **§S-M′ arm** —
`ceq/arm_smprime.py`, wired into `CEQForCausalLM` as `operator="smprime"`
(`99777ab:V17_ARM_WIRING.md`). **That is a different object from the resolvent operator
the rest of this card describes.** No sentence from the sections below transfers
into this one, and no sentence here transfers out.

**The law of this section, from Ruling 2, verbatim: NO SENTENCE TRANSFERS ACROSS
CORNERS WITHOUT A BIND AT THE CORNER IT DESCRIBES.**

## Limits, first (v17-K)

- **There is no trained checkpoint for this v17-K Q3 arm (`CEQForCausalLM`,
  `operator="smprime"`) yet — a separate fact from the probe-arm weights this
  card ships elsewhere; see "Weights and checkpoints" below.** The one Kaggle
  attempt on record (`results/kaggle_v17k_output/ceq-v17-k.log`) halted at setup
  — `GATE FAILED: a code source is configured` — before any step ran.
  ⟨SLOT `Q3_CHECKPOINT`⟩ — **NOT MEASURED**, owed by the Q3 training node.
  Everything below the identity clause is a template until it lands.
- **The parameter counts are NOT equal, and the arm carries MORE.** Arm
  **25,736,232** against control **25,728,000** at `ceq/hf/train.py::DEFAULTS` —
  **+8,232 = +0.03200 %**. A reading favourable to the arm is the one that needs
  the caveat (`99777ab:V17_ARM_WIRING.md` §5) `[MEASURED]`.
- **The corner the arm's bind is claimed at is not the corner the LM runs at.**
  The module docstring claims `β = 0, QK-off`; the LM runs at `β = 1, qk = 1,
  g = 1`, which is `ArmSMPrime.__init__`'s own switch initialisation, because
  `qk = 0` deletes the content term exactly and an LM at that corner has no
  query-key channel (`99777ab:V17_ARM_WIRING.md` §3, third caveat) `[MEASURED]`.
- **Training here is not bitwise-reproducible and is not claimed to be.** Under
  L-TOL as amended (`V17K_RULINGS.md` §A1) training between checkpoints is held
  to a measured noise floor, and **that floor is not yet measured**: ⟨SLOT
  `FLOOR_TRAIN_ABS_DLOSS`⟩ — **NOT MEASURED**, owed by the floor node.
- **Nothing about the arm's quality is claimed anywhere in this section.**
  `99777ab:V17_ARM_WIRING.md`'s lead caveat governs everything it supplies: those numbers
  say the arm can be selected, built, forwarded, checkpointed and stepped. They
  do not say it is good, bad, or comparable to anything.
- **"PINNED" MEANS "INDISTINGUISHABLE AT THIS RESOLUTION", NEVER "EXACT", and
  the resolution is a number printed beside every verdict.** RULING 10′ decides
  pinning by likelihood ratio against `3.841` (χ²₁ at 0.95) and `ln n`, and
  ships each verdict with its minimum detectable departure
  `|β − 1|_min ≈ √(3.841 / (n · I_β))` — *a verdict printed without it is a
  defect* `[INHERITED]` `V17K_RULINGS.md`, RULING 10′. A departure smaller than
  that number is not absent; it is **below what this eval split can see**. The
  ruling's own worked case: *"a real 2 % departure is detected only 32 % of the
  time at n = 4000"* `[INHERITED]`, which this card reproduces as a conditional
  and not as a fact — see `99777ab:V17_R10P_LRT.md` §5, which recovers the `I_β ≈ 1.390`
  the figure silently assumes.
- **RULING 2a's `5·δ_β` criterion is RETIRED, and `δ_β` is now a DIAGNOSTIC.**
  It broke on measurement: the re-take read the identical-seed pair **bitwise**
  on the certified device, so the tolerance collapsed to exactly zero and every
  run read the "moved" branch for a reason about the pair rather than about
  training. RULING 10′ does not patch that floor — Wilks' randomness is over
  the **data**, so a training pair is irrelevant to the test. `δ_β` is still
  measured (`scripts/k_noise_floor.py`) and still printed, because a diagnostic
  that **contradicts** the verdict is a finding; it no longer decides anything.
- **The gradient census still gates the word "preferred"** (owed by the β node)
  and RULING 10′ leaves it standing explicitly as the scientific payload. It
  does not exist yet.
- **The Wald cross-check can contradict the likelihood ratio, and both are
  printed.** `(β − 1)²·I_β` against the same `3.841` is asymptotically the same
  statistic, but the equivalence holds **at the MLE** and `β_final` is fit on
  the training split, so it never is one. Measured locally at a 27,914-parameter
  shape: `Λ = 7.21` (rejects) against `Wald = 1.15` (does not) — opposite sides
  of the same constant on the same `β` (`99777ab:V17_R10P_LRT.md` §4 `[MEASURED]`).

---

## The split sentence (RULING 2)

**This card carried no sentence about the §S-M′ arm before now.**
`git show ab5b485:MODEL_CARD.md | grep -n "smprime\|S-M′\|corner"` returns
**zero lines** `[MEASURED]`. So there is nothing to retire: what Ruling 2 splits
is the sentence the finished run **will** carry, and it is filed here in advance
as **two clauses about two different objects**, printed apart so that neither can
be read as evidence for the other.

### Identity clause — CITES THE `β = 0` CERTIFICATE, UNCHANGED

**At `β = 0, QK-off` the arm's row weight is exactly the path product**
`G_ij = Π_{k=j+1}^{i} m_k e^{iθ_k}` (`99777ab:V16_ARM_SMPRIME.md` §1). This clause is
**not** rewritten for v17-K; it cites the existing certificate as it stands.

| row | what it certifies | reading | source |
|---|---|---|---|
| **(a)** | the oracle-gate bind holds on **BED-M's real support** `{−1, 0, +1}`, gates and drives read from `make_equilibrium_batch`, label recomputed by the bed's own `equilibrium_oracle` at float64 | `residual = 5.919777e-16` at `n=64, s=64`; `7.550528e-16` at `n=512, s=64` and at `n=256, s=128`, against a bar of `1e-6`; **real part exactly `0.000e+00`** — the whole residual is the `1.2246e-16` imaginary dust `polar(1, π)` injects | `99777ab:V16_ARM_SMPRIME.md` row (a) `[MEASURED]` |
| **(b)** | the reachable parameters | `(u, θ) = (1, π) → −1.0 + 1.2246467991473532e-16j`, `(0, ·) → 0.0 + 0.0j`, `(1, 0) → 1.0 + 0.0j`; **real part exact at all three, modulus exact at all three** (`1.0`, `0.0`, `1.0`) | `99777ab:V16_ARM_SMPRIME.md` row (b) `[MEASURED]` |
| **(c)** | the three corners | `β=1, g≡0, QK-on` **bitwise** against `#5a`'s own `softmaxAttn`; `β=0, g≡0, QK-on` **bitwise** against `exp(qk)`; **`β=0, QK-off` bitwise (cpu)** against an independent double-loop path product | `99777ab:V16_ARM_SMPRIME.md` row (c) `[MEASURED]` |

The statements these evaluate are proved in **`lean/CEQ/V16Domain.lean`** —
`corner_softmax`, `corner_linear`, `corner_path_product`,
`corner_path_product_is_the_gate_product`, `three_corners_containment`,
`corners_are_distinct`, and the census theorem `bedM_gate_exact` (every value
BED-M draws is an ordinary point of the gate) `[INHERITED]`, cited to that file.
The line-by-line correspondence from module object to Lean statement is
`99777ab:V16_ARM_SMPRIME.md` §2.

**Two caveats that travel with this clause and may not be dropped:**

1. **`bitwise (cpu)`, and the device is part of the claim.** `torch.cumprod` is
   sequential on cpu and a parallel scan on cuda, so the vectorized route
   re-associates: on cuda `11 / 64` entries move by at most `5.551115e-17`. The
   claim was **weakened to what is true and pinned per device**, with the zeros —
   the clause the label bind depends on — asserted equal on both
   (`99777ab:V16_ARM_SMPRIME.md` §0) `[MEASURED]`.
2. **This clause is about the `β = 0` corner and says nothing about any trained
   checkpoint.** It is a certificate about an operator at a setting of its
   switches. It is not evidence about weights.

### Trained-model clause — STATES ONLY WHAT WAS TRAINED (RULING 2a: THREE BRANCHES, CENSUS-GATED WORD)

Ruling 2: Q3 trains with **`β` LEARNABLE, INIT 1**, and the per-instance `β` is
logged as a column. **Ruling 2a** ruled that the original two branches below
("`β` moved off 1" / "`β` pinned at 1") were a **FALSE DICHOTOMY**, defined
"pinned", and replaced them with three, plus a gradient census that decides
which word the card is allowed to print. What follows is that ruling, filed as
a template.

#### The pin criterion (RULING 10′), and the resolution that must travel with it everywhere it appears

> **"PINNED" MEANS "INDISTINGUISHABLE AT THIS RESOLUTION". IT DOES NOT MEAN
> "EXACT", AND THE RESOLUTION IS A NUMBER ON THIS CARD.**
>
> ```
> |β − 1|_min  ≈  √( 3.841 / (n · I_β) )
> ```
>
> `n` is the eval count and `I_β` the observed Fisher information in `β`, per
> observation. A departure smaller than `|β − 1|_min` is **not absent** — it is
> below what this eval split can see. **A verdict printed without this number is
> a defect** (`V17K_RULINGS.md`, RULING 10′). The ruling's worked case:
> *"a real 2 % departure is detected only 32 % of the time at n = 4000"* — so
> even the ruling's own example sits below the resolution it prints.

`β_i` is decided by a **1-dof nested-model comparison** on the **held-out eval
split**, from **two deterministic forward passes** — no training, no
identical-seed pair, no floor:

```
Λ = 2·[ LL_eval(β_final) − LL_eval(β ≡ 1) ]
```

| `Λ` | verdict |
|---|---|
| **≤ 3.841** (χ²₁ at 0.95, Wilks) | **PINNED** — the data cannot reject that the shipped model is the softmax-corner object |
| **> ln n** (BIC, `n` = eval count) | **MOVED** — the dial buys its description length |
| **3.841 < Λ ≤ ln n** | **"rejected at 0.95, below description-length"** — both constants printed, no interpolation, no invented `k` |

**Both constants print beside every verdict, always** (`V17K_RULINGS.md`,
RULING 10′) `[INHERITED]`. Implemented at
`ceq/hf/modeling_ceq.py::beta_lrt` / `lrt_report`; the formatter is what prints
the two constants and the resolution together, so the defect above is
unreachable rather than merely discouraged.

**THE WALD CROSS-CHECK, AND THAT IT CAN CONTRADICT.** `(β_final − 1)²·I_β`
against the same `3.841` is the asymptotically equivalent parameter-space
statement — Fisher information IS the converter between parameter displacement
and likelihood displacement, which is how RULING 10′ closes the units objection
rather than waiving it. The equivalence holds **at the MLE**, and `β_final` is
fit on the **training** split, so it is never the eval split's maximiser.
Measured on this box at a 27,914-parameter shape: `Λ = 7.206` **rejects** while
`Wald = 1.150` **does not**, on the same `β` at the same `3.841`
(`99777ab:V17_R10P_LRT.md` §4 `[MEASURED]`). **Both are printed; neither is reported
alone.** `I_β` is the observed information, taken as an exact Hessian diagonal
by double backward (`torch.autograd.grad(..., create_graph=True)` then one
`grad` per parameter); where it measures **non-positive** — which it does at an
untrained model, because observed information is only guaranteed non-negative at
an MLE — the Wald and the resolution **do not exist** and are printed as
`undefined` with the measured value, never clamped.

#### `δ_β` — RETIRED AS THE CRITERION BY RULING 10′, KEPT AS A DIAGNOSTIC

RULING 2a defined PINNED as `|β_i,final − 1| ≤ 5·δ_β,i`, with `δ_β,i =
|β_i⁽ᵃ⁾ − β_i⁽ᵇ⁾|` across the Ruling-1 identical-seed pair. **That criterion
broke on measurement.** The re-take read the pair **bitwise** on the certified
device (`δ_nrmse = 0.0`), so `5·δ_β` collapsed to exactly zero and every run
read the "moved" branch for a reason about the **pair's determinism** rather
than about training. RULING 10′ does not patch the floor: **Wilks' randomness is
over the DATA, not the optimizer**, so a bitwise training pair is simply
irrelevant to the test `[INHERITED]` `V17K_RULINGS.md`, RULING 10′.

`δ_β` **survives as the diagnostic it always was.**
`scripts/k_noise_floor.py` keeps measuring it and that is correct;
`ceq/hf/modeling_ceq.py::beta_summary` keeps computing `pinned` /
`pinned_fraction` / `branch` / `n_degenerate_floor` off it, and its returned
`criterion` field says in the artifact — not only here — that it is a
diagnostic. **It is printed beside the verdict for one reason: a diagnostic that
CONTRADICTS the verdict is a finding, and it cannot contradict anything if it
stops being computed.** The two are not nested and will disagree by
construction: a `β` far from 1 in a direction the data dislikes reads
**MOVED** on the distance diagnostic and **PINNED** on the likelihood ratio,
because it fits *worse* than `β = 1` and is therefore no evidence against
`β = 1` (`tests/gate0/test_g16_lrt_pinned.py::
test_a_beta_far_from_one_that_the_data_dislikes_reads_pinned_and_that_is_the_point`)
`[MEASURED]`.

**In this architecture, "per-parameter" and "per-layer" name the same index.**
`β` is one scalar `nn.Parameter` per attention layer, **shared across all
heads in that layer** — `ceq/hf/modeling_ceq.py::CEQAttention.__init__`, and
`beta_column`'s own docstring states the granularity rather than choosing it:
*"a `[n_layers, n_heads]` beta would be a construction the arm does not
have"* `[INHERITED]`. So `i` in `β_i` ranges over **layers**, not heads —
noted here because it governs what Branch C below can and cannot report.

> ⟨SLOT `LRT_VERDICT`⟩ — **NOT MEASURED.** Per layer `i`: `Λ_i`, its verdict
> against `3.841` and `ln n`, `I_β,i` (total and per-observation), the Wald
> statistic and whether it agrees, and `|β_i − 1|_min`. Plus `n` — the eval
> count — and the joint `Λ` with its dof. Owed by the **Q3 training node**,
> from `ceq/hf/modeling_ceq.py::beta_lrt` on the held-out split; the notebook's
> Q3 chunk cell already calls it and writes
> `/kaggle/working/lrt_verdict.json`. **Requires no floor and no second run**,
> which is why RULING 10′ unblocked what RULING 2a could not: the verdict now
> waits on the training run alone.

> ⟨SLOT `EVAL_SPLIT_SPEC`⟩ — **NOT MEASURED.** Which bytes the eval split is,
> and the receipt that training never read them. `ceq/hf/train.py::ByteBatches`
> cuts 90/10 contiguously and `train()` draws only from `.train`; the Q3 cell
> draws from `.val` off the same object built from the same bytes, so the
> hold-out is by construction and not by an offset re-derived at read time. The
> spec still has to be printed, because `beta_lrt` scores whatever batch it is
> handed and cannot check the hold-out for the caller.

> ⟨SLOT `DELTA_BETA_PER_PARAM`⟩ — **NOT MEASURED, AND NO LONGER BLOCKING.**
> `δ_β,i` for every layer `i`: `β_i`'s own spread across the Ruling-1
> identical-seed pair. Owed by the **floor node**, `scripts/k_noise_floor.py`.
> **This is now a DIAGNOSTIC** (RULING 10′): the verdict above does not wait on
> it, and its value is read for whether it contradicts the verdict. The
> re-take's bitwise pair means it may well land at `0`, which is exactly the
> measurement that retired it as a criterion.

> ⟨SLOT `BETA_FINAL_DIST`⟩ — **NOT MEASURED.** `β_i,final` for every layer,
> printed as `min / median / max / n`, **plus the fraction PINNED** — where
> "pinned" is now ⟨`LRT_VERDICT`⟩'s per-layer verdict, not the retired
> `5·δ_β` test. Owed by the **β node**, out of the Q3 run, from the
> per-instance `β` column Ruling 2 requires
> (`ceq/hf/modeling_ceq.py::beta_column` / `beta_summary`, which compute
> `final_min` / `final_median` / `final_max` today). **The fraction-pinned
> figure no longer waits on ⟨`DELTA_BETA_PER_PARAM`⟩; it waits on the Q3 run.**

> ⟨SLOT `Q3_RUN_SPEC`⟩ — **NOT MEASURED.** data, steps, shape, seeds, optimizer,
> device, and the commit the run was made at. Owed by the Q3 training node.

> ⟨SLOT `Q3_MEASUREMENTS`⟩ — **NOT MEASURED.** The trained run's readings, each
> beside the control's at the counts in the table below, each with the tolerance
> L-TOL-as-amended gives it (`V17K_RULINGS.md` §A1). Owed by the Q3 training node.

#### The gradient census, and the word-gate it controls

Per pinned `β_i`, the **integrated `|∂L/∂β_i|` over training**, against the
same statistic for moved `β`'s (`V17K_RULINGS.md`, RULING 2a). **This is a
different quantity from anything `ceq/hf/modeling_ceq.py` computes today** —
`beta_summary`'s `max_abs_grad` is a **maximum over logged steps**, not a
**running integral**; the census needs an accumulator, not a max.

- **PINNED-WITH-SIGNAL** — a gradient reached `β_i` and it returned to `1`.
  Softmax is **genuinely preferred** there.
- **PINNED-WITHOUT-SIGNAL** — the dial was never exercised. The claim
  weakens to **"unused"**, and **may never be written as "preferred."**

> ⟨SLOT `GRADIENT_CENSUS`⟩ — **NOT MEASURED.** Per pinned `β_i`: the
> integrated `|∂L/∂β_i|` over training, and its PINNED-WITH-SIGNAL /
> PINNED-WITHOUT-SIGNAL classification. Owed by the **β node**,
> `ceq/hf/modeling_ceq.py` — a new per-parameter gradient accumulator, logged
> alongside the existing `beta_column`. No such accumulator exists in that
> file today.

**THE WORD-GATE.** The word **"preferred"**, applied to the softmax corner,
may not be written anywhere in this clause from any source other than
⟨SLOT `GRADIENT_CENSUS`⟩'s classification. It has exactly one point of entry —
⟨SLOT `GRADIENT_CENSUS_WORD`⟩ below — and that slot has exactly two legal
fills, chosen by table lookup and nothing else:

| ⟨`GRADIENT_CENSUS`⟩ over the pinned set | the ONLY legal fill of ⟨`GRADIENT_CENSUS_WORD`⟩ |
|---|---|
| **unanimous** PINNED-WITH-SIGNAL | *"a genuine preference — the gradient repeatedly returned `β` to the corner"* |
| **any** PINNED-WITHOUT-SIGNAL present | *"not established as a preference — the dial was never exercised for at least one pinned layer, so the corner's word is 'unused'"* |

The second row is the **default when the two disagree**. This extends RULING
2a's own PRECEDENCE principle — *"the criterion that licenses the weaker
sentence wins ties"* — from choosing between competing pin-criteria to
choosing between competing readings of a mixed census; a mixed census over the
pinned set is exactly such a tie, and this document is the one asserting the
extension, not RULING 2a itself. **A filled card that writes "preferred"
without citing a unanimous ⟨`GRADIENT_CENSUS`⟩ has failed this template**,
regardless of which branch below it fills.

> ⟨SLOT `GRADIENT_CENSUS_WORD`⟩ — **NOT FILLABLE** until ⟨`GRADIENT_CENSUS`⟩
> exists. No default value. No branch below may hard-code either row of the
> table above as prose in place of this slot.

#### The quantifier, and why three branches replace two

"Pins" is **per-parameter** (here, per-layer). The card reports the
**distribution**, never one number standing for all layers. The original two
branches were a false dichotomy; RULING 2a's three:

**BRANCH A — `≥ 95 %` of `β` parameters PINNED.**

> The shipped model is the softmax-corner object carrying an unused
> exact-propagation corner.
>
> This sentence carries two separate claims that must not be read as one:
>
> 1. **Structural, and true regardless of the census.** The parameters that
>    would reach the `β = 0, QK-off` corner are still in the checkpoint — the
>    identity clause above is a real certificate about that corner, and it is
>    **not evidence about this checkpoint**, whose `β` did not go there.
>    Nothing in the trained model reaches the exact-propagation corner; it is
>    carried and unused. This is the *"unused … corner"* in the sentence
>    above, and it does not depend on the gradient census.
> 2. **Whether settling near the softmax corner was PREFERRED — census-gated,
>    and not this document's word to choose:** ⟨SLOT `GRADIENT_CENSUS_WORD`⟩.
>
> **What "pinned" does NOT claim, and why the distinction matters.** `≥ 95 %`
> pinned means `≥ 95 %` of layers read `Λ_i ≤ 3.841` — **the data cannot
> reject `β_i = 1` at this eval split's resolution**, which is not equality and
> is not a claim that `β_i` is 1. The resolution itself,
> `|β_i − 1|_min ≈ √(3.841 / (n · I_β,i))`, travels with the branch; a
> departure below it is invisible here, not absent. The bitwise and exact-row-sum
> properties measured for the **operator at `β = 1` exactly** —
> `corner_softmax`, `softmax_row_sum_one`; `Σ_j|W_ij| = 1.000000` on every row
> at `β = 1` against `[1.312192, 0.724290, 2.563817, 2.264559, 10.293107,
> 2.721943, 3.096841, 1.337183]` at `β = 0`; bitwise against `#5a`'s own
> `softmaxAttn`, and **not** bitwise (`1.110223e-16` on `19/64` entries)
> against `ceq/lm.py`'s `Attention("softmax_x")` (`99777ab:V16_ARM_SMPRIME.md` rows
> (c), (e), §4.2) `[MEASURED]` — are properties of **that corner**, not of a
> layer the eval split merely fails to separate from it. **This branch does not
> assert the trained layers are bitwise softmax** — only that a `1`-dof
> likelihood ratio on the held-out split does not reject `β_i = 1` at the
> resolution printed beside it. Conflating "pinned" with "exactly at
> the corner" is exactly the transfer this section's own law forbids ("NO
> SENTENCE TRANSFERS ACROSS CORNERS WITHOUT A BIND AT THE CORNER IT
> DESCRIBES"); a non-rejection at `Λ ≤ 3.841` is its own, weaker bind — it is a
> statement about what this eval split can resolve, not about `β` — and the label
> bind's own failure at the exact corner (`1.335288` where the honest cell
> reads `1.110223e-16`, `99777ab:V16_ARM_SMPRIME.md` rows (f), (h) `[MEASURED]`) is
> Lean `#5a`'s content, not evidence against a layer merely sitting near it.
> The two per-position heads and three switches this corner's object still
> pays **+8,232 parameters** (**+0.03200 %**) to carry travel with the
> sentence regardless of branch (`99777ab:V17_ARM_WIRING.md` §5 `[MEASURED]`).
>
> Measurements: ⟨SLOT `Q3_RUN_SPEC`⟩, ⟨SLOT `BETA_FINAL_DIST`⟩ (fraction
> pinned `≥ 0.95`), ⟨SLOT `GRADIENT_CENSUS`⟩, ⟨SLOT `Q3_MEASUREMENTS`⟩.

**BRANCH B — `≤ 5 %` of `β` parameters PINNED.**

> Training left the softmax corner; the model is the interpolated object, `β`
> distribution printed.
>
> The corners are provably distinct objects — `|c₁−c₂| = 4.472918`,
> `|c₁−c₃| = 1.144938`, `|c₂−c₃| = 5.335671` (`99777ab:V16_ARM_SMPRIME.md` row (d),
> `corners_are_distinct`) `[MEASURED]` — so a bind proved at one corner is not
> a bind at an interpolated point between them, and **neither corner
> certificate applies to this branch's checkpoint**: not the `β = 0` identity
> clause above, not `corner_softmax` at `β = 1`. No "preferred"/"unused"
> language applies here — RULING 2a does not attach the word-gate to this
> branch, and this card does not add one.
>
> Measurements: ⟨SLOT `Q3_RUN_SPEC`⟩, ⟨SLOT `BETA_FINAL_DIST`⟩ (fraction
> pinned `≤ 0.05`), ⟨SLOT `Q3_MEASUREMENTS`⟩.

**BRANCH C — else: MIXED.** *(RULING 2a calls this the likely outcome and the
most informative one.)*

> Some layers are pinned near `β = 1`, some are not, and the finding is
> **WHERE** the split falls — a model that leaves the corner only in some
> layers is a finding about where exactness pays. Neither corner certificate
> applies whole-cloth; print the full distribution and the locations of the
> moved `β`'s.
>
> ⟨SLOT `BRANCH_C_LOCATIONS`⟩ — **NOT MEASURED.** Which **layers** carry a
> moved `β` (outside the pinned band) versus which are pinned. Owed by the
> **β node**, `ceq/hf/modeling_ceq.py`, out of ⟨`BETA_FINAL_DIST`⟩ once it
> exists. **"Which heads" — RULING 2a's own phrasing — is not answerable
> under the current parametrization.** `β` is one scalar per layer, shared
> across every attention head in that layer (`CEQAttention.__init__`,
> `beta_column`'s docstring, quoted above); there is no per-head `β` to
> locate. Reporting one would require widening the arm to a
> `[n_layers, n_heads]` `β`, which is exactly the construction
> `beta_column`'s own docstring already refuses as something "the arm does
> not have." **This is a real gap between what RULING 2a asks for and what
> the shipped parametrization can report: layer-level locations are
> reachable, head-level locations are not, without a new construction this
> round's first law forbids.** Flagged rather than papered over.
>
> Measurements: ⟨SLOT `Q3_RUN_SPEC`⟩, ⟨SLOT `BETA_FINAL_DIST`⟩ (fraction
> pinned strictly between `0.05` and `0.95`), ⟨SLOT `BRANCH_C_LOCATIONS`⟩,
> ⟨SLOT `Q3_MEASUREMENTS`⟩.

> ⟨SLOT `BRANCH_VERDICT`⟩ — **NOT RULED, and not a free choice.** Which of the
> three branches' text the finished card carries is selected by ⟨
> `BETA_FINAL_DIST`⟩'s fraction-pinned figure alone (`≥0.95` → A, `≤0.05` → B,
> else → C), once that figure exists. No other input selects the branch, and
> no branch may be filled before it does.

---

## Parameters — the exact counts (RULING 3)

Every comparison in this section is headed by the exact counts, so no reader has
to take "matched" on trust. `99777ab:V17_ARM_WIRING.md` §5 `[MEASURED]`, CPU, float32.

| shape | arm — `operator="smprime"`: **25,736,232 params** (at `DEFAULTS`) | control — `operator="sgate"`, softmax-shaped: **25,728,000 params** (at `DEFAULTS`) | difference |
|---|---|---|---|
| `d=32, L=2, H=4, V=32, S=16` | **27,914** | **27,776** | **+138** = `2·(2·(32+1)+3)` |
| `ceq/hf/train.py::DEFAULTS` — `d=512, L=8, H=8, V=256, seq=512` | **25,736,232** | **25,728,000** | **+8,232** = `8·(2·(512+1)+3)` = **+0.03200 %** |

**The excess is `n_layers · (2·(d+1) + 3)`, and it is exactly these tensors, by
name** — `m_head.weight [1,d]`, `m_head.bias [1]`, `theta_head.weight [1,d]`,
`theta_head.bias [1]`, `beta []`, `qk []`, `g []`, per layer — **and the control
has no parameter the arm lacks** (`99777ab:V17_ARM_WIRING.md` §5). It cross-checks the
same excess `99777ab:V16_ARM_SMPRIME.md` §1 recorded at `d_model = 16`:
`4,806 − 4,769 = 37 = 2·16 + 5` `[MEASURED]`.

**The residual is excluded as an explanation by magnitude, not closed.** Ruling 3:
`0.032 %` is MATCHED, and the arm is **not** to be re-architected to close it —
that is how new constructions sneak in. Three ways to make it exact were priced
and none taken: shrinking the arm's `d` breaks "same shape"; widening the
control's `d` is not expressible (the excess is not a multiple of any control
parameter block); deleting a head or a switch mutilates the arm
(`99777ab:V17_ARM_WIRING.md` §5).

**Result table for the trained comparison** — headed with the same counts, filled
by the Q3 run:

| reading | arm, **25,736,232 params** | control, **25,728,000 params** | Δ | tolerance (L-TOL as amended) |
|---|---|---|---|---|
| ⟨SLOT `Q3_MEASUREMENTS`⟩ | ⟨SLOT⟩ | ⟨SLOT⟩ | ⟨SLOT⟩ | ⟨SLOT `FLOOR_TRAIN_ABS_DLOSS`⟩ for training-side readings; **bitwise** for forward-only deciding cells |

---


## Model Details

- **Type:** a `transformers`-registerable attention function plus a small causal-LM
  architecture (`CEQForCausalLM`), byte-level vocabulary. **No language-model weights are
  published here** — what ships under `weights/` are 4,769-parameter trained task-probe arm
  checkpoints (`scale.m3_quintuple.QuintArm`), a different and much smaller object. See
  "Weights and checkpoints" below.
- **Operator:** `out = stock_attention(q, k, v) + Σ_{h=1..K} (α A)^h v`, `A` strictly lower
  triangular and signed; shipped default `sgate` at `rho=1.5, lam=0.10, hops=2`.
- **Sizes ever trained:** 3.3M and 3.65M parameters. **Nothing above 3.65M has ever run.**
- **Requires `trust_remote_code=True`.** Loading executes Python from this repository on your
  machine. Read `modeling_ceq.py` first and pin `revision=` to a commit hash so a later push
  cannot change what runs.
- **License:** MIT. **Contact / issues:** the source repository.
- **The headline result is a loss.** Read "Limits, first" before anything else.

**This card describes a MODULE; its language model is not a trained checkpoint.** There are no weights here
for `CEQForCausalLM`. What does ship, since the e3 ladder completed, is a set
of trained task-probe arm tensors under `ceq/hf_artifact/weights/` — 4,769-parameter
`QuintArm` modules trained by `scale/m3_quintuple.py`, each verified bit-exact against its
journal row before export. The card remains measured against softmax rather than presented on
its own, and the one capability comparison ever run at matched parameters went against it.

```
out = stock_attention(q, k, v) + Σ_{h=1..K} (α A)^h v      A strictly lower triangular, SIGNED
```

Every number below names the test that produces it. Reproduction commands are at the end.

---

## Limits, first

**THE e3 LADDER COMPLETED. ROW G FIRED AT t\* ∈ {2, 8, 32}; THE t\* = 1 RUNG IS
UNDERPOWERED — NOT A KILL, AND NOT A WIN.** The settled-vs-twin ladder over the chain-family
equilibrium tasks (`e3_t*`, 60 units: {settled, twin, softmax} × seeds 0–4 × t\* ∈
{1, 2, 8, 32}, n_train=2048, n_eval=2048, journal keys `*_taske3_t{1,2,8,32}` in
`results/m3_quintuple_v2.jsonl`, reading in `results/e_ladder_reading.txt`) closed today with
three findings, stated in the pre-registration's own order:

- **Row G (pre-registered) fired on three of four rungs.** At t\* ∈ {2, 8, 32} a cell sits at
  or above predict-the-mean — twin 0.996743 and settled 1.013958 at t\*=2; settled 1.096009
  and twin 1.091725 at t\*=8; settled 1.103711 and twin 1.119745 at t\*=32 (`results/e_ladder_reading.txt`,
  per-cell means) — so those rungs **credit nothing in either direction**. This is a statement
  about the arms' capacity budget at depth, not a settling kill.
- **t\* = 1 is underpowered, not a kill.** The settled−twin contrast reads **−0.036025**,
  95% CI **[−0.118936, +0.062209]**, N=5 paired seeds, verdict NO DIFFERENCE (reading's own
  convention: delta = NRMSE_twin − NRMSE_settled, positive favours settled). The interval
  spans zero and five seeds cannot resolve gaps below roughly 0.05 NRMSE
  (`99777ab:M3_QUINTUPLE_PREREGISTERED_READING.md` floor). The route owed is pre-registered:
  **seeds 5→13 at that rung**, at which the realised resolution is 0.027260
  (`results/e_ladder_reading.txt`). Until that run completes, no sentence about settling at
  t\* = 1 is licensed by this data.
- **The headline claim is unchanged and stays exactly what was earned, with its interval:**
  pivot-routed mixture attention (twin) beats the softmax baseline by **+0.111396 NRMSE,
  95% CI [+0.100873, +0.121920], 5/5 seeds**, on `negation_scope` at n_train=8192
  (`results/m3_quintuple_v2.jsonl`; contrast table in [`ceq/hf_artifact/README.md`](https://github.com/teerthsharma/resolvent/blob/master/ceq/hf_artifact/README.md)).
  No superiority claim beyond that interval is made anywhere in this card.

Scope caveat carried from the reading itself: every e3 task binds `equilibrium_oracle`, the
signed path sum the ceq resolvent computes, so only the settled-vs-twin contrast is
creditable on this ladder and a loss here is a statement about e3, not about the round's
prediction.

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

**Citation is stale — flagged rather than silently dropped.** The row above was measured by
two functions (`test signed operator reaches negative influence`, calibrated by
`test the nonnegative control is stuck at exactly zero`) in a module named
`test_w6_attention.py`, formerly under `tests/w6/`. That file was deleted from the live tree
at `c71527a` ("Remove the round reports from the tree..."), after an earlier move from
`tests/w6/` to `attic/tests/w6/` at `228a048` — `git log --follow -- attic/tests/w6/test_w6_attention.py`
shows both moves, and neither path exists at HEAD (`git ls-files | grep w6` returns only
`tests/w6/conftest.py`). The two functions survive only inside a Kaggle snapshot copy of this
repository under `kaggle/snapshot/repo/`, which `pytest.ini`'s own
`norecursedirs = attic kaggle ...` excludes from collection. **No live, collected test
currently reproduces this row**; the numbers above are not re-verified by the reproduction
commands at the end of this card until a replacement test is written.

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

Lean 4.7.0 + mathlib, `lake build` exit 0, **zero `sorry`**. **134 theorems + 41 lemmas**
across the thirteen files in `lean/CEQ/` and `lean/CEQ.lean`, counted by
`python scripts/lean_count.py` (comments stripped, `@[attr]` prefixes admitted),
2026-09-11; re-run it to check for drift. A raw `grep -c '^\s*theorem\s'` reads 140 + 35
instead: it counts six comment lines that open with the word "theorem" and misses six
`@[simp] lemma` declarations. (An earlier count of **39** covered only the six earliest
modules — `Contraction`, `Nilpotent`, `Occupancy`, `OracleSeparation`, `OrbitBound`,
`Refcount` — and is still exact for them; `V15`, `V15Fork`, `V15Kernel`, `V15Phase`,
`V15Source` and `V16Domain` came later.)
The load-bearing one for the operator is `CEQ.Nilpotent.pow_card_eq_zero` — strictly
lower-triangular ⇒ `A^n = 0` over any `CommRing`, with **no sign hypothesis**, which is what
licenses dropping non-negativity for free. `CEQ.OracleSeparation.oracle_ne_resolvent` is
the round-8 addition and it plays that nilpotency off against its negation: an operator
that is entrywise non-negative, has symmetric support and has one positive entry — which
is what a walk on an undirected graph gives — is never nilpotent, so the absorbing-chain
oracle proposed for the `t*` ladder is not the arm's own forward. `CEQ.OracleSeparation.truncation_never_exact` is the operational half: the finite
occupancy sum is not the inverse of `(1 − Q)` at ANY truncation, so every rung of the
ladder leaves a real residual. `tests/w3b/` gates the build, checks for `sorry`
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
**Citation is stale:** the function `test both interfaces are registered`, formerly in
`test_w6_attention.py` under `tests/w6/`, was deleted from the live tree at `c71527a`, after
an earlier archive move to `attic/tests/w6/` at `228a048`; neither path exists at HEAD, and
`pytest.ini` excludes `attic/` and `kaggle/` from collection, so the function is not currently
collected anywhere this card's own reproduction commands reach. See the same note beside the
influence-Jacobian
table above for the git evidence.

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
ablation (`scale/s2_probe.py`, `tests/cameron/test_s2_ablation.py`, table in `99777ab:DONE.md`) measures
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

**In production.** There are no language-model weights (only the 4,769-parameter probe arms
under `weights/`), no KV cache, no backward pass for the Triton kernel,
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

## Weights and checkpoints

**What ships here are trained probe-arm tensors, not language-model weights.** Five files
under [`ceq/hf_artifact/weights/`](https://github.com/teerthsharma/resolvent/blob/master/ceq/hf_artifact/weights/MANIFEST.json), one per seed, each
a trained `QuintArm` — 4,769 parameters, module class `scale.m3_quintuple.QuintArm`
(q/k projection + 2-layer MLP + scalar readout). **The `CEQForCausalLM` model in
`modeling_ceq.py` still has no published checkpoint**, and no file in `weights/` is named
`model.safetensors`, because none of them loads into it.

**What is tracked in the repository, and how it breaks down.** `git ls-files | grep -iE
'\.(safetensors|pt|pth|bin|ckpt)$' | wc -l` counts **171** weight-shaped tracked files,
2026-09-10. None of them is a `CEQForCausalLM` checkpoint; by directory
(`... | sed -E 's#/[^/]+$##' | sort | uniq -c`): **135** under
`results/m3_quintuple_v2_weights/` and **23** under `results/m3_quintuple_v2_cuda_weights/`
(per-cell `QuintArm`/settled/softmax probe checkpoints from the same m3 journal), **5** under
`ceq/hf_artifact/weights/` (the table below — the only ones packaged for HF export), **4**
under `results/paired/` (signed/unsigned probe pairs) and **3** loose in `results/`
(`phaseD_weights_*.pt`, a separate probe sweep); the remaining **1** is
`tests/gate0/fixtures/enwik8_short.bin`, a test fixture that only matches the extension
filter and is not a checkpoint at all. **`ceq/hf_artifact/` is a built HuggingFace package**
— `config.json`, `configuration_ceq.py`, `modeling_ceq.py`, `weights/MANIFEST.json` and the
five `.safetensors` above — but the package it builds is the `QuintArm` probe, not a
language model. **What does not ship, anywhere in this repository, is a trained
`CEQForCausalLM` weight file of any kind** — not in `weights/`, not in the 166 other tracked
tensors, and not the v17-K Q3 checkpoint described above, which has never been run to
completion.

| file (`ceq/hf_artifact/weights/`) | task | cell | seed | eval NRMSE (journal) |
|---|---|---|---|---|
| `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1.safetensors` | e3_t1 (t\*=1) | twin | 0 | 0.9231181827 |
| `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd1_taske3_t1.safetensors` | e3_t1 (t\*=1) | twin | 1 | 1.0785056996 |
| `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd2_taske3_t1.safetensors` | e3_t1 (t\*=1) | twin | 2 | 0.9293004878 |
| `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd3_taske3_t1.safetensors` | e3_t1 (t\*=1) | twin | 3 | 0.9568415797 |
| `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd4_taske3_t1.safetensors` | e3_t1 (t\*=1) | twin | 4 | 0.9041007794 |

Seed mean **0.958373** (`results/e_ladder_reading.txt`), and note honestly: **seed 1 sits
above predict-the-mean (NRMSE > 1.0)**. These weights belong to the underpowered t\* = 1 rung
described at the top of Limits; they are the ship candidate's arm at the only rung where both
cells beat the bar, and they carry no capability claim of their own.

- **Geometry string = journal key**: `{cell}_k{k}_s{s}_d{d}_st{steps}_ntr{n_train}_nev{n_eval}_b{t_max}_sd{seed}_task{task}`
  (`scale/m3_quintuple.py::_key`). Here: k=8 pivots, s=64, d=24 (task distance), d_model=16,
  150 steps, n_train=2048, n_eval=2048, t_max=21.
- **Provenance.** Metrics come from `results/m3_quintuple_v2.jsonl`, matched by key; the
  journal's last commit is `1cc7900` and the export HEAD is `1c56985`. Both are stamped into
  every safetensors metadata block and into `weights/MANIFEST.json`.
- **Verification, per file.** Exported by `scripts/export_hf_weights.py`: tensors reloaded
  from the safetensors file into a fresh `QuintArm`, forward re-run on the task's own eval
  batch (rebuilt as `bfn(n_eval, s, d, d_model=16, seed=seed+12345)`, exactly as training
  built it), recomputed NRMSE compared against the journal row. All five shipped files:
  tensor round-trip drift exactly **0.0**, |Δ NRMSE vs journal| exactly **0.000e+00**
  (bit-exact; acceptance bar was 1e-6). A checkpoint that fails is deleted and excluded from
  the manifest rather than shipped with a caveat.

## Compute

- **CPU lane (everything measured so far).** The registered geometry pins
  `torch.set_num_threads(2)` inside `scale/m3_quintuple.py:74`, before any unit runs, and the
  export script above pins the same count before verifying — CPU matmul reduction order
  depends on thread count, so an unpinned verifier could manufacture a phantom delta. Every
  journalled number this card cites from m3 ran under that pin on one machine: Windows,
  Python 3.11.9, torch 2.5.1+cu121, RTX 4060 Laptop (see the single-machine caveat below).
- **CUDA lane: opened, IN PROGRESS.** A CUDA execution lane for the same registered geometry
  exists as work in progress and has produced **no journalled number yet**. No figure in this
  card is a CUDA figure, and none should be quoted as one until it appears in
  `results/` with its own commit.

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
