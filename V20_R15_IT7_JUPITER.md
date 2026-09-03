# V20 R15 — it.7 — JUPITER (MYCROFT) — Q3 LEARNABILITY, BOTH WINGS

**Branch `v17k-gate0`, HEAD `207e7b9`.** Contract slot `CEQ_V20_R15_CONTRACT.md:94-113`.
Wings frozen at `V20_R15_WING_MANIFEST.md:20-21`: W1 `arm_smprime`, W3 `arm_pl`.

**`lake build` RE-RUN THIS ITERATION `[RUN]`: `cd lean && lake build; echo $?` → `0`.**
The standing gap this office named at it.6 §7 is closed. Every Lean citation below rests
on a green build observed at it.7, not on `V20_R15_IT1_WILSON.md:131,:137` alone.

---

## 0. THE CELL SHEET AFTER THIS ITERATION

| | W1 `arm_smprime` | W3 `arm_pl` |
|---|---|---|
| Q1 EXACT CLASS | F0 (it.6 §1) | F1 (it.6 §2) |
| Q2 OUTSIDE | F1 + constant (it.6 §3) | F1 + constant (it.6 §4) |
| **Q3 LEARNABILITY** | **FILLED — F2, §2** | **FILLED — F1, §3** |
| Q4–Q6 | not filled | not filled |

**Six of twelve.** Q3/W1 is graded **F2 and not lower**, and §2.5 says exactly why.

---

## 1. THE COLUMN THE ROUND HAS BEEN READING IS SATURATED, AND THE INSTRUMENT SHIPS THE ONE THAT IS NOT

### 1.1 The `−inf` fact, explained — not recorded as unexplained

it.6 asked this office to explain `lambda_hat == −inf` on cells where `unit_root == False`,
or file it unexplained. **It is explained, mechanically, from source.**

`[READ] scripts/v15_r1.py:371,:383`:

```
lg = torch.log(m)
...
lambda_hat=float(lg.mean()),
```

`lambda_hat` is the **mean of `log m` over every position**. One `m_k == 0` puts a `−inf`
in the sum, and the mean of a set containing `−inf` is `−inf`. **`unit_root` is not in the
mechanism at all** — it is `bool(a_max >= 1.0)` at `:386`, a different statistic on a
different reduction.

**THE LAW, and it has no exceptions:** over **32 cells** (16 `arm_smprime` + 8 `arm_pl`
+ 8 `softmax`),

> `lambda_hat == −inf`  ⟺  `frac_gate_annihilated > 0`.

`[RUN] tests/jupiter/test_v20_r15_it7_q3.py::test_lambda_hat_is_minus_inf_exactly_when_a_gate_died`
— 32 cells, `bad == []`. Seeds **3, 11 and 13** are `unit_root == False` with `−inf`, and
seed 2 is `unit_root == False` with a finite `−0.7953`. **The it.5 reading tied `−inf` to a
dead gate AND a unit root; the unit root was a coincidence of the eight cells it was read
on, and the conjunction is struck.**

**`lambda_hat = −inf` carries exactly one bit — "at least one gate is exactly zero".** On
15 of 16 W1 cells it is that one bit and nothing else. Every W1 relaxation reading this
round has taken has been taken on a saturated column.

### 1.2 `lambda_hat_live` — the column beside it, never used this round

`[READ] scripts/v15_r1.py:384`: `lambda_hat_live` is the same mean **over the finite
entries only** (`lg[fin].mean()`), and `:365-370` says in the source why both are shipped —
*the zeros are never quietly dropped into a mean that would read finite.* The instrument
already anticipated the failure the round then made anyway.

| W1 regime | n | `frac_gate_annihilated` | `lambda_hat` | **`lambda_hat_live`** | `eval_nrmse` |
|---|---|---|---|---|---|
| **crossing** (seed 2) | 1 | `0.0` | −0.7953 | **−0.7953** | **0.20392** |
| **intermediate** (seed 3) | 1 | `0.4967041015625` | −inf | **−0.4411** | 0.916258 |
| **flat band** | 12 | `0.5032958984375` | −inf | **−0.0009 … −0.0436** | 0.852–0.927 |
| **over-decayed** (11, 13) | 2 | `0.9768` / `0.9946` | −inf | **−2.4989 / −4.1869** | 1.121 / 1.203 |

**The good regime is INTERIOR.** Too slow (`λ̂_live ≈ 0`: twelve cells whose surviving
gates sit at unit modulus and relax not at all) fails at `0.85–0.93`. Too fast
(`λ̂_live ≤ −2.5`) fails **worse than doing nothing** — both cells exceed `nrmse = 1.0`.
The one cell that crosses sits at `−0.7953`. **A relaxation rate has a window, and the
window is what Q3 is about.** `[RUN] ::test_w1_relaxation_rate_is_a_WINDOW_not_an_extreme`.

---

## 2. Q3 / W1 `arm_smprime` — THE DATA CONDITION, AND WHAT IT IS NOT

### 2.1 The statement

> **Over all sixteen `arm_smprime` cells the trained `qk` coordinate partitions the outcome
> with zero overlap, and the trained `beta` coordinate does not order it at all.**

| coordinate | Spearman ρ vs `eval_nrmse`, n=16 | separation |
|---|---|---|
| `manifest.smp_values.qk` | **+0.717647** | **three regimes, no overlap** |
| `manifest.smp_values.beta` | **−0.032353** | none |

`[DERIVED]` from the 16 cells, computed here; `[RUN]`
`::test_w1_outcome_partitions_by_trained_qk_with_no_overlap`. Sorted `qk`:

```
0.5006 | 1.1694 1.2052 1.2347 1.2418 1.2622 1.2766 1.3206 1.3223 1.3970
       | 1.5597 1.6452 1.6525 | 1.8535 | 2.0305 2.0427
crosses|          the twelve-cell flat band              |  s3  |  over-decayed
```

`max(qk | crossing) < min(qk | flat band)`, and `max(qk | flat band) < min(qk | over-decayed)`.
**Zero overlap, sixteen of sixteen.**

### 2.2 THE it.5 SEED IS CORRECTED, AGAINST THIS OFFICE'S OWN §6

it.6 §6 carried forward, unchallenged: *"W1's failures are annex M1's predicted descent —
β descends toward the exact corner."* **Sixteen cells do not support it, and the number
that kills it is `ρ = −0.032`.**

`[READ] ceq/arm_smprime.py:12-14,:51`: `beta = 0` is the exact corner (`W = num`, no
normalizer, `Z^0 = 1` for every `Z` at `:241-242`); `beta = 1` is softmax. Then:

- the **one crossing cell** has `beta = 1.34393` — **further from the corner than 13 of the
  16 cells**;
- the cell that sits **at** the corner, seed 11 at `beta = −0.06023`, is the **worst cell in
  the tournament** (`nrmse = 1.120603`, above the trivial `1.0`);
- the twelve flat-band failures span `beta ∈ [0.58758, 1.00081]` — clustered on **softmax**,
  not on the corner.

**M1 is not struck. Its `[RUN]` instance (`+20.87`) is on path-product data
(`CEQ_V20_R15_CONTRACT.md:176-179`), and no cell in this journal is labelled with that
corpus.** What is struck is the round's *transfer* of M1's descent to W1's sixteen cells.
That transfer had no domain census behind it, which is the exact failure this round has now
paid for twice (M14/V-25, `pathProd_eq_Wp`). **It is filed as the third instance, and it is
this office's own.**

### 2.3 VENUS's asymmetry — half supported, half misnamed

> *"A norm cap arrests M2; a cap does not undo M1's descent direction."*

- **Arrest half — 8/8 correlational support**, §3. Not proved causal: no capped run exists.
- **Descent half — MISNAMED.** It names `beta`. `beta` does not order W1's outcomes
  (`ρ = −0.032`). The coordinate that does is `qk`, **and `qk` is a temperature on the QK
  score, not a norm** (`ceq/arm_smprime.py:4`: `W_ij = G_ij * exp(qk * q_i.k_j) / Z_i^beta`).
  A norm cap on the weights does not name `qk` at all. **The asymmetry is not refuted; it is
  stated in a coordinate the data does not use.**

### 2.4 The landscape theorems, by declaration and line

| declaration | `path:line` | what it contributes here |
|---|---|---|
| `pathProd_eq_zero_iff` | `lean/CEQ/V16Domain.lean:129` | the `m = 0` corner is an **interior, representable** point — so W1's flat band is a failure *inside* the class, not an escape from it |
| `pathProd_abs` | `lean/CEQ/V16Domain.lean:121` | `abs(pathProd) = ∏ m`, which is what makes `Σ log m` the arm's own relaxation exponent rather than a diagnostic |

Both re-verified under a **green `lake build` at it.7**.

### 2.5 GRADE: **F2.** HOW-BAD gap, and why not F1

**The window is NECESSARY on these sixteen cells and DEMONSTRABLY NOT SUFFICIENT.** Seed 3
sits at `λ̂_live = −0.4411`, **more than half the way** from the flat band to the crossing
cell's rate, and it **does not cross** (`0.916258`). One cell inside the qualitative window
fails. **A condition with a counterexample in its own sixteen cells is F2, not F1**, and no
constant is offered because a constant fitted to n=1 crossing is not a constant.

**Second gap, independent:** the partition is on **trained** coordinates read off
`manifest.smp_values`. A data condition stated in a *post-training* coordinate is a
**description, not a criterion** — it cannot be evaluated before spending the GPU-second.
**Replacement route, priced:** the 0-step control path already exists
(`scripts/v15_r1.py:801`); journalling `manifest.smp_values` on the `_0step` record makes
the same partition testable **pre-training at 0 additional GPU-s**. That is the same
0-cost repair Q1/W1's `0 of 24` gap needs, on the same four fields.

---

## 3. Q3 / W3 `arm_pl` — THE SIGN OF THE RELAXATION EXPONENT, 8/8

### 3.1 The statement

> **`lambda_hat < 0` ⟺ `eval_nrmse < 0.7`, on all eight `arm_pl` cells, zero overlap.**

| seed | `lambda_hat` | `a_hat_max` | `gate_r2` | `eval_nrmse` |
|---|---|---|---|---|
| 0 | **−1.43248** | 1.4105 | 0.98887 | 0.644673 |
| 1 | **−1.43660** | 1.2869 | 0.98976 | 0.644517 |
| 4 | **−1.38449** | 1.4536 | 0.98018 | 0.633739 |
| 5 | **−1.45285** | 1.5052 | 0.97187 | 0.641999 |
| 6 | **−1.47142** | 1.1029 | 0.97845 | 0.662128 |
| 3 | **+0.18896** | 49.6605 | 0.62686 | 1.113339 |
| 7 | **+1.21686** | 116.0061 | 0.04665 | 1.148927 |
| 2 | **+0.75843** | 12.7675 | 0.01112 | 1.152280 |

`max(nrmse | λ̂ < 0) = 0.662128` against `min(nrmse | λ̂ > 0) = 1.113339`. **A gap of
`0.451211` with nothing in it.**
`[RUN] ::test_w3_lambda_hat_sign_separates_all_eight_arm_pl_cells`.

**All eight `arm_pl` cells have `frac_gate_annihilated == 0.0`**, so `lambda_hat` here is
**not** saturated and *is* the live rate — §1.1's law is why W3 gets the clean column and
W1 does not. The two wings needed different columns for the same physical quantity, and
that is a fact about the instrument, not about the arms.

### 3.2 The theorem, and its instance

**M2 GATE-LANDSCAPE THEOREM** (`CEQ_V20_R15_CONTRACT.md:181-185`): *open-range
`a = 2σ(w) − 1`: infimum unattained, `|w| → ∞`.* Lean `#19 [M]`. The three failing cells are
that theorem's conclusion **measured**: `a_hat_max` at `12.77 / 49.66 / 116.01` with
`gate_r2` collapsing to `0.011–0.627`.

**GRADE: F1 with the constant.** The constant is the observed separation margin `0.451211`
on `eval_nrmse`, equivalently the sign threshold `λ̂ = 0` with nearest approach `+0.18896`
from above and `−1.38449` from below.

**HOW-BAD gap:** **it is a correlation over n=8 with an untested causal direction.** The
claim "a norm cap arrests M2" predicts that capping `|w|` moves seeds 2, 3 and 7 into
`λ̂ < 0`. **No capped run exists.** The clean separation is consistent with the cap being
causal and equally consistent with `λ̂`'s sign and the divergence sharing a common cause in
the initialisation. **Replacement route, priced from the record:** three capped cells at
seeds 2, 3, 7. `arm_pl` seed 0 cost `1.884 s` `[READ] results/v17k_r4_retake.jsonl` —
**~6 GPU-s for the decisive experiment**, against `146.399 s` spent at it.6 for a 0-of-8
result. **This is the cheapest live experiment in the round and no office has proposed it.**

---

## 4. THE PHYSICS FAMILIES — ASSESSED AS CHEEGER WAS, WITH THE DOMAIN CENSUS FIRST

The author's direction: *"prediction of causality is difficult but physics have been doing
it for a while."* Four families were on the table. **Each is graded on its L-DOM census
before its statement**, because M14 was machine-true and domain-empty and
`pathProd_eq_Wp`'s hypothesis is satisfied by **0 of 3 registered beds**.

Registered beds, by module: **BED-M**, **BED-K** (`ceq/beds/bed_k.py`), **BED-1**
(`ceq/beds/bed_1.py`). Three.

| family | what it would buy | **L-DOM: beds satisfying its hypothesis** | decision | grade |
|---|---|---|---|---|
| **Koopman / transfer-operator spectrum** | the relaxation structure IS the spectrum; a window on the leading exponent is a spectral-gap statement | **3 of 3, and already instrumented.** `lambda_hat_live` (`scripts/v15_r1.py:384`) is a leading log-modulus exponent on the arm's own recurrence; §1.2 measures the window on 16 cells; BED-1 carries the same object with **`λ₂ := ρ(Q)` defined, not assumed** (`MATHEMATICS.md` §7) | **ADOPT for Q3** | the §2/§3 cells |
| **Committor / transition-path theory** | "given this state, what happens next" as a proved object | **1 of 3.** BED-1 only — and there it is **not deferred, it is built**: `ceq/beds/bed_1.py:40` carries the closed form `(0, 1/4, 1/2, 3/4, 1)` and the solve is checked against it. **BED-M has no basins and no absorbing set; both frozen wings are BED-M arms.** | **DEFER, with the reason** — importing it onto W1/W3 would be M14's shape a third time | **not graded; no cell claimed** |
| **Mori–Zwanzig / memory kernel (M7)** | `K ≡ 0` on the chain, `K ≠ 0` exactly on BED-K; decides forced vs redundant composition per bed | **2 of 3 by construction** (BED-M chain, BED-K kernels) — **but 0 of 3 measured**, and the closed form it wants comes "from the Hankel residual", which is **M16, itself F1 by construction and resting on AAK/Glover, the citation this round already refused** (`CEQ_V20_R15_CONTRACT.md:246-249`) | **REPRICE, do not adopt at it.7** — M7 cannot reach its F0 target while M16 is F1 | **F4 as of it.7** (unattempted; dependency was unnamed in the annex until now) |
| **Fluctuation–dissipation** | ties the relaxation window to a noise/response identity | **0 of 3.** No bed journals a response function, a perturbation-response pair, or a noise amplitude | **RETIRE for this round** | **F4, domain-empty** |

**The one that earned its place is Koopman**, and it earned it by being **already in the
instrument and already measured** rather than by being imported. **The prettiest import —
the committor — is refused by its own census**, and refusing it here is worth more to this
round than a fourth theorem with an empty domain.

**No TDA / persistence / Mapper / sparse-mask claim is made in this report**, so no
`anthropic-skills:tda-tdd` invariant is invoked. Naming the skill for a claim not made
would be the padding this round has been striking.

---

## 5. TEST-BOUND — RED FIRST, VERBATIM, AND FOUR SEEDED MUTATIONS

`tests/jupiter/test_v20_r15_it7_q3.py`, five nodes.

**RED #1 — a real defect in this office's first draft, recorded rather than overwritten
`[RUN]`:**

```
>       assert len(plain) == 13 and sorted(high) == [11, 13]
E       assert (12 == 13)
E        +  where 12 = len([0, 1, 4, 5, 6, 7, ...])
tests\jupiter\test_v20_r15_it7_q3.py:83: AssertionError
1 failed, 4 passed in 0.54s
```

**The draft had bucketed seed 3 with the flat band** by testing
`0.49 < frac_gate_annihilated < 0.51`, which swallows `0.4967041015625` and
`0.5032958984375` together. They are **different states**: seed 3's `λ̂_live` is `−0.4411`,
the band's is `≈ 0`. **The wrong bucket would have published a window with no counterexample
in it and graded Q3/W1 F1.** The corrected node grades it F2. This is a tolerance chosen to
make a partition tidy, caught on this office's own draft.

**PLANTED NEGATIVES — four, named and seeded** via `JUP_IT7_MUTATE`, applied to loaded
cells, never to the journals `[RUN]`:

| mutation | what it breaks | nodes RED |
|---|---|---|
| `fga_law` | one cell's `frac_gate_annihilated → 0.0`, `lambda_hat` left at `−inf` | 3 |
| `qk_band` | seed 2's `qk → 1.3`, into the failing band | 2 |
| `live_band` | seed 2's `lambda_hat_live → −0.01` | 1 |
| `pl_sign` | one `arm_pl` cell's `lambda_hat` sign flipped | 2 |

Each mutation reaches at least the node it targets; **none is a no-op.**

**GREEN `[RUN]`:** `python -m pytest tests/jupiter/test_v20_r15_it7_q3.py -q` → **5 passed
in 0.24s**.

**No journal was written. No git command was run. Kaggle was not touched.**

---

## 6. F-GRADES AND HOW-BAD GAPS

| cell | grade | HOW-BAD gap | replacement route, priced |
|---|---|---|---|
| **Q3 / W1** | **F2** | seed 3 is a counterexample **inside** the window (`λ̂_live = −0.4411`, no crossing); and the criterion is stated in **post-training** coordinates, so it cannot be evaluated before the spend | journal `manifest.smp_values` on the `_0step` record — **0 GPU-s**, same four fields Q1/W1 needs |
| **Q3 / W3** | **F1 + constant** (margin `0.451211`) | correlation over **n=8**, causal direction **untested**; no capped run exists | three capped cells at seeds 2, 3, 7 — **~6 GPU-s** at `1.884 s`/cell |
| **M7 Mori–Zwanzig** | **F4** | unattempted; its closed form depends on **M16**, which is F1-by-construction on a refused citation, and the annex never stated the dependency | state the dependency in the annex; M7 cannot reach F0 while M16 is F1 |
| **Fluctuation–dissipation** | **F4** | **0 of 3** beds journal a response pair | retired for this round |

**Standing gap, named once:** everything in §2–§3 is read off **two journals, 24 distinct
cells, one box**. No cross-box reproduction exists for any of it.

---

## 7. SCOREBOARD

No new points. Phase B is **6 of 12** cells filled. The `+6` for the theory table needs all
N×Q6 graded — **6 outstanding across it.8–it.14**. Carried: **2 of 44**. `lake build` green
at it.7 removes the standing citation gap from all six filled cells.
