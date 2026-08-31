# V17-K — RULING 10′: "PINNED" BY LIKELIHOOD RATIO

**LEAD CAVEAT, BEFORE ANY NUMBER BELOW.** This is a **criterion and its
instrumentation**, not a reading. Every `Λ`, `I_β`, Wald and resolution in this
file comes from a model trained **30 gradient steps at a 27,914-parameter shape
on one repeated batch**, built to make the statistic exercise its branches. **No
number here says the arm is good, bad, comparable, pinned or moved.** Q3 has not
run; when it does, the verdict is read off the run, not off this file. The
`β` values printed below were *planted* wherever the text says "planted", and
where they were not planted they came from a 30-step probe that is not a
training result.

**SECOND CAVEAT: the criterion is a resolution, not a fact about `β`.** RULING
10′'s own instruction, and the reason the power line is mandatory: *"PINNED"
means "indistinguishable at this resolution", and the resolution is a number on
the card, never an implication of exactness.* Every PINNED below should be read
as "this eval split could not separate `β_i` from 1", and the number that says
how well it could is printed beside it.

**THIRD CAVEAT: one of the author's two `[RUN]` figures does not reproduce as
stated.** `ln n = 8.29 at n = 4000` reproduces exactly. The 32 % power figure is
**not arithmetic** — it is a power calculation that needs an `I_β` the ruling
never states. §5 recovers the value it implies (`I_β ≈ 1.390` per observation)
and reports what this box measured instead (`4.24` and `10.27`), at which the
same 2 % departure is detected **74 %** and **98 %** of the time. The figure is
reproducible as a **conditional**, not as a fact.

**Files written.** `ceq/hf/modeling_ceq.py` (edited), `MODEL_CARD.md` (edited),
`kaggle/ceq_v17k.ipynb` (cell 18 only, appended), `tests/gate0/test_g16_lrt_pinned.py`
(new), this file. **Nothing else was touched** — not `ceq/hf/train.py`, not
`ceq/arm_smprime.py`, not `scripts/*`, not `scale/*`, not `V17K_RULINGS.md`, not
`COSTS.md`, `MISTAKES.md`, `LOOP_PROMPT.md`, `CEQ_V16_CONTRACT.md`, and no other
`tests/gate0/test_g*.py`. **`kaggle` was not invoked, `~/.kaggle` was not
touched, nothing was uploaded, pushed or launched, and no writing git command
was run.**

## Provenance

| | `git rev-parse HEAD` | `git status --porcelain` |
|---|---|---|
| **start** | `ab5b48547884e04258276e6e808d5a71ea65f917` | 9 modified (`MODEL_CARD.md`, `ceq/arm_smprime.py`, `ceq/hf/configuration_ceq.py`, `ceq/hf/modeling_ceq.py`, `ceq/hf/train.py`, `house-events.jsonl`, `results/v15_r1.jsonl`, `scale/identity_manifest.py`, `scripts/v15_r1.py`) + 32 untracked |
| **end** | `ab5b48547884e04258276e6e808d5a71ea65f917` (unchanged; no writing git command run) | **10 modified + 36 untracked**, counted at the close of this file. The growth is **not all this node's**: `tests/chase/deq_run.jsonl`, `V17_ENWIK8_SOURCE.md` and `scripts/k_noise_floor.py` appeared mid-session from peers working the same tree. **Of those paths this node wrote exactly four**: `ceq/hf/modeling_ceq.py`, `MODEL_CARD.md`, `kaggle/ceq_v17k.ipynb` and `tests/gate0/test_g16_lrt_pinned.py`, plus this report |

**Box.** Windows 11, RTX 4060 Laptop 8 GiB, torch `2.5.1+cu121`, python 3.11.
Every measurement below is **this box, this run**, `float32`,
`hidden_size=32, n_layers=2, n_heads=4, vocab=32, seq=16, batch=8` unless the
row says `cuda`. CPU rows run at `torch.set_num_threads(1)`. L-TIME: no timing
is inherited and none is reported.

---

## VERDICT TABLE

| question | answer |
|---|---|
| **(a) Λ, in one line** | `Λ = 2·n·(loss(β≡1) − loss(β_final))` from two `torch.no_grad()` forwards on the held-out split, `n = (labels[:,1:] != -100).sum()`, since `CEQForCausalLM` returns a **mean** cross-entropy so `LL = −n·loss`. §2 `[MEASURED]` |
| **(b) Restore proof** | **`torch.equal` over all 39 parameters, True.** Plant: one float32 ulp on `beta` → the same comparator reads False. `.grad` and `model.training` also restored, each with its own test. §2.2 `[MEASURED]` |
| **(c) LRT vs Wald** | **They can give OPPOSITE verdicts, and both are printed.** At a planted `β = 1.02`: `Λ = 7.206` **rejects**, `Wald = 1.150` **does not** — opposite sides of the same `3.841`. At the eval-split MLE they track: rel gap **4.8 %** at `d/res = 0.21`, **37.5 %** at `d/res = 2.07`. The cause is the score term, which the Wald form drops. §4 `[MEASURED]` |
| **(d) Hessian-diagonal method** | Exact double backward: `torch.autograd.grad(loss, betas, create_graph=True)` then one `torch.autograd.grad(g_i, beta_i)` per parameter. No off-diagonal term formed. §3 |
| **(e) `ln n = 8.29 at n = 4000`?** | **REPRODUCES.** `ln 4000 = 8.294050` → `8.29`. §5.1 `[MEASURED]` |
| **(f) `32 % at a 2 % departure, n = 4000`?** | **REPRODUCES ONLY AS A CONDITIONAL.** It needs `I_β ≈ 1.390` per observation, which the ruling never states. At this box's measured `I_β` (`4.244`, `10.274`) the same departure is detected **74.1 %** and **98.2 %** of the time. §5.2 `[MEASURED]` |
| **(g) The power line** | `\|β−1\|_min = √(3.841/(n·I_β))`, printed on every row. Measured: `0.0874` and `0.0599` at `n = 120` CPU; `0.1129` and `0.0291` at `n = 120` cuda. `undefined` where `I_β ≤ 0`, with the measured value and the reason. §3.2 `[MEASURED]` |
| **(h) Non-positive `I_β`** | **HAPPENS, AND IS REPORTED NOT CLAMPED.** Untrained model at `β = 1`: `I_β,total = −0.00245` and `−0.00138`. Observed information is only non-negative **at an MLE** and `β_final` is not one. §3.3 `[MEASURED]` |
| **(i) Strict determinism** | **Λ is strict-safe; the Wald and the resolution are NOT.** On cuda under `use_deterministic_algorithms(True)` the forward repeats **bitwise**, and `beta_lrt` **RAISES** `cumsum_cuda_kernel does not have a deterministic implementation` from the Hessian's backward. §6 `[MEASURED]` |
| **(j) What was retired** | `5·δ_β` as **the criterion**, in `MODEL_CARD.md` (4 places) and `ceq/hf/modeling_ceq.py::beta_summary` / `_pinning`. The machinery is intact and still computes; `criterion` in the returned dict says it is a diagnostic. `scripts/k_noise_floor.py` untouched. §7 |
| **(k) `tests/gate0`** | **289 passed, 0 failed**, twice in a row (baseline **256**, +33 from this node's file). `tests/gate0/test_g16_lrt_pinned.py`: **31 passed** from **28 failed / 3 passed** RED. §1, §8 `[RUN]` |
| **(l) GREEN/RED/BLOCKED** | **GREEN.** The criterion is implemented, tested, carded and wired into the notebook, and — unlike the criterion it replaces — **it is not blocked on any measurement another node owes.** Three findings for the author in §9. |

---

## 1. RED

Tests were written first. Two RED transcripts are filed, because they answer two
different questions and neither alone is honest.

### 1.1 The first RED run — the whole node, before any file was touched

`tests/gate0/test_g16_lrt_pinned.py` run against the tree exactly as this node
found it (`ceq/hf/modeling_ceq.py` with no `beta_lrt`, `MODEL_CARD.md` still
carrying `5·δ_β` as the criterion, cell 18 with no Λ):

```
FAILED tests/gate0/test_g16_lrt_pinned.py::test_lambda_is_exactly_zero_when_beta_is_already_one_and_not_when_it_moved
FAILED tests/gate0/test_g16_lrt_pinned.py::test_lambda_is_two_n_times_the_loss_difference_the_ruling_writes
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_second_pass_sets_every_beta_to_exactly_one
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_model_is_bit_identical_after_the_statistic_and_the_check_can_fail
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_statistic_leaves_the_gradient_state_alone
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_statistic_restores_the_training_flag
FAILED tests/gate0/test_g16_lrt_pinned.py::test_labels_are_required_because_n_is_read_off_them
FAILED tests/gate0/test_g16_lrt_pinned.py::test_an_operator_without_beta_yields_no_verdict_rather_than_a_pinned_one
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_pinned_boundary_is_inclusive_at_3_841_and_the_constant_is_not_ours
FAILED tests/gate0/test_g16_lrt_pinned.py::test_moved_needs_strictly_more_than_ln_n
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_interval_verdict_is_the_rulings_exact_words
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_verdict_changes_only_where_a_constant_is_crossed
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_interval_is_empty_below_47_eval_items_and_that_is_not_an_interpolation
FAILED tests/gate0/test_g16_lrt_pinned.py::test_a_planted_moved_beta_crosses_and_a_planted_beta_at_one_does_not
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_observed_information_is_a_second_derivative_and_not_a_squared_first
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_wald_form_and_the_lrt_agree_near_the_corner
FAILED tests/gate0/test_g16_lrt_pinned.py::test_a_non_positive_observed_information_is_reported_and_never_clamped
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_power_line_is_the_formula_the_ruling_prints
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_minimum_detectable_departure_is_a_fifty_percent_power_point
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_authors_32_percent_figure_pins_an_unstated_fisher_information
FAILED tests/gate0/test_g16_lrt_pinned.py::test_every_verdict_prints_both_constants_and_its_resolution
FAILED tests/gate0/test_g16_lrt_pinned.py::test_a_report_with_no_resolution_still_says_why_rather_than_going_quiet
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_joint_statistic_names_its_degrees_of_freedom
FAILED tests/gate0/test_g16_lrt_pinned.py::test_beta_summary_calls_the_delta_beta_branch_a_diagnostic
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_card_carries_ruling_10_prime_and_no_longer_states_5_delta_beta_as_the_criterion
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_notebook_computes_lambda_at_the_end_of_q3_without_renumbering_anything
FAILED tests/gate0/test_g16_lrt_pinned.py::test_the_statistic_runs_on_the_certified_device_and_still_restores
27 failed, 1 passed, 2 warnings in 8.28s
```

**THE ONE THAT PASSED AT RED IS THE ONE THAT WAS SUPPOSED TO.**
`test_the_authors_ln_n_figure_reproduces` asserts `round(math.log(4000), 2) ==
8.29` and touches no module in this repo — it is a check of the **author's
arithmetic**, not of this node's code, so it passes before the change **by
construction**. That it passed at RED is the receipt that the `8.29` figure is
independent of anything built here.

### 1.2 The second RED run — the FINAL test file against the pre-change module

Three tests changed after the first RED, because two measurements (§4, §6)
falsified the premise they were written on and one finding had to be pinned.
Following the β node's own idiom (`V17_R2_BETA.md` §1), the **final** file was
therefore re-run against a reconstructed pre-change `ceq/hf/modeling_ceq.py` —
the live file copied aside, this node's block and its four docstring/key edits
inverse-applied, the suite run, the file restored and **sha256-verified
identical on both sides**:

```
live sha256:       c9c43e91386720b29fe6adc1132b668e997cd0aa6780a263f3bb2857f36dccc8
pre-change sha256: 9679ece653427a18eee72e745072aa93244834859382104a2a61d270d5a98e26
...
28 failed, 3 passed, 2 warnings in 8.71s
restored sha256:   c9c43e91386720b29fe6adc1132b668e997cd0aa6780a263f3bb2857f36dccc8
restore is byte-identical: True
```

**THE THREE THAT PASS HERE, AND WHY THAT IS A SCOPE LIMIT AND NOT A RESULT.**
The reconstruction reverts **only** `ceq/hf/modeling_ceq.py`. `MODEL_CARD.md`
and `kaggle/ceq_v17k.ipynb` were already edited by the time this transcript was
taken, so `test_the_card_carries_ruling_10_prime_…` and
`test_the_notebook_computes_lambda_…` pass here for a reason about **what was
reverted**, not about what they assert; their genuine RED is in §1.1, where both
are listed. The third is `test_the_authors_ln_n_figure_reproduces`, RED-passing
for the reason above. **This is stated rather than left for a reader to notice
that 28 + 3 ≠ 27 + 1.**

## 1b. GREEN

```
$ python -m pytest tests/gate0/test_g16_lrt_pinned.py -q
31 passed, 2 warnings in 11.05s

$ python -m pytest tests/gate0 -q          # run twice, back to back
289 passed, 4 warnings in 63.07s
289 passed, 4 warnings in 61.02s
```

Baseline before this node: **256 passed, 0 failed**. This node adds **33** tests
and breaks none.

---

## 2. Λ, AS IMPLEMENTED

### 2.1 The statistic

`ceq/hf/modeling_ceq.py::beta_lrt(model, **forward_kwargs)`.

```
LL = -n * loss          # CEQForCausalLM.forward returns a MEAN cross-entropy
Λ  = 2 * (LL_final - LL_at_beta_one)
   = 2 * n * (loss(beta == 1) - loss(beta_final))
```

`n` is **read off `labels`**, not inferred from a shape:
`(labels[:, 1:] != -100).sum()`, which matches the forward's own shift and
`F.cross_entropy`'s default `ignore_index`. Calling without `labels` **raises**,
because `n` sets `ln n` *and* the power line and a guessed `n` is an invented
tolerance wearing arithmetic's hat
(`test_labels_are_required_because_n_is_read_off_them`).

**Cross-checked against a second, independent path already in the file.**
`beta_substitution` (the β node's diagnostic) forwards at trained `β` and at
`β←1` and returns both losses; `Λ` must equal `2·n·(loss_at_beta_one −
loss_trained)` off those, to `rel=1e-9`
(`test_lambda_is_two_n_times_the_loss_difference_the_ruling_writes`) `[MEASURED]`.

**The second pass sets every `β` to exactly `1.0`**, asserted by recording what
the forward actually saw rather than by trusting the call site — the spy records
`[1.3, 0.8]` on pass one and `[1.0, 1.0]` bitwise on pass two
(`test_the_second_pass_sets_every_beta_to_exactly_one`) `[MEASURED]`.

**Determinism of the statistic itself**, `[MEASURED]`: two consecutive calls on
the same model and batch return `lambda_joint = 26.509805917739868` **identical**,
every per-parameter `Λ` identical, every `I_β` identical.

### 2.2 The restore proof, and its planted negative

```
n_params compared: 39
all torch.equal: True
after a ONE-ULP plant on beta, all torch.equal: False
```

`torch.equal`, never `allclose`, over **every** named parameter — and the plant
is one float32 ulp on `beta` itself, the tensor the substitution writes, so the
comparator is shown capable of failing `[MEASURED]`. Restore is in a `finally`
and copies back a pre-call clone.

Two further restores each have their own test, because each is a way the
statistic could quietly corrupt something RULING 10′ leaves standing:

* **`.grad` is untouched.** `torch.autograd.grad` does not accumulate into
  `.grad`, and that is asserted rather than assumed — `.grad` is `beta_census`'s
  input, and the census is the ruling's own "scientific payload"
  (`test_the_statistic_leaves_the_gradient_state_alone`).
* **`model.training` is saved and restored** around the `eval()`
  (`test_the_statistic_restores_the_training_flag`).

### 2.3 Non-degeneracy: the exact zero and its control

Both halves are in **one** test on purpose, because `Λ == 0.0` alone is passed by
a function that returns `0.0` unconditionally:

| plant | `lambda_joint` | per-parameter |
|---|---|---|
| `β = [1.0, 1.0]` | **exactly `0.0`** | both exactly `0.0` |
| `β = [1.4, 1.0]` | non-zero | layer 0 non-zero, **layer 1 still exactly `0.0`** |

The third row is what makes the per-parameter statistic per-parameter and not a
broadcast copy `[MEASURED]`.

---

## 3. `I_β` — THE HESSIAN-DIAGONAL METHOD, STATED

### 3.1 The method

**Exact double backward.** One
`torch.autograd.grad(loss, betas, create_graph=True)`, then one
`torch.autograd.grad(g_i, beta_i, retain_graph=True)` per parameter. That is the
`i`-th diagonal entry of the Hessian of the mean loss; **no off-diagonal term is
formed or needed**, and no finite difference is used.

**It is a second derivative and not a squared first**, checked against a central
finite difference of the same loss in `β` at `h = 1e-3` (agreement `rel < 2e-2`)
with the squared gradient as the planted negative — the number a careless Fisher
implementation returns instead
(`test_the_observed_information_is_a_second_derivative_and_not_a_squared_first`)
`[MEASURED]`.

### 3.2 Two `I_β`, because the ruling's two formulas use different ones

Since `LL = −n·loss`:

```
I_beta_total   = -d2 LL / dbeta2  =  n * d2 loss / dbeta2
I_beta_per_obs = I_beta_total / n =      d2 loss / dbeta2
```

The **Wald** form `(β−1)²·I_β` takes the **total**; the **power line**
`√(3.841/(n·I_β))` takes the **per-observation**. They are the same number
divided by `n`, and confusing them is a factor of `n` — so both are emitted by
name and both are printed. Both readings of the power line are asserted equal
(`√(3.841/(n·I_per_obs)) == √(3.841/I_total)`, `rel=1e-9`)
(`test_the_power_line_is_the_formula_the_ruling_prints`) `[MEASURED]`.

**What the printed resolution means, checked rather than asserted in prose.** A
departure of exactly `|β−1|_min` puts the noncentrality at `3.841`, which is
**50 % power** — not a threshold below which nothing is detected and above which
everything is. `[MEASURED]`: `power(λ = 3.841) = 0.50004`
(`test_the_minimum_detectable_departure_is_a_fifty_percent_power_point`).

### 3.3 A non-positive `I_β` is reported, never clamped

**Untrained model at `β = 1`**, `n = 120`, CPU `[MEASURED]`:

```
model.layers.0.self_attn.beta  I_beta_total = -0.00244615   I_beta_per_obs = -2.0385e-05
model.layers.1.self_attn.beta  I_beta_total = -0.00137515   I_beta_per_obs = -1.1460e-05
```

Observed information is only guaranteed non-negative **at an MLE**, and
`β_final` is not one. Negative here means `β = 1` sits at a local **maximum** of
the eval loss along `β`. Where `I_β ≤ 0`, `wald` and `beta_min_detectable` are
`None`, `note` carries the measured value and the reason, and `lrt_report`
prints `|beta-1|_min=undefined` with the note underneath. **An `abs()` there
would manufacture a resolution out of a curvature pointing the other way**
(`test_a_non_positive_observed_information_is_reported_and_never_clamped`).

The LRT verdict still exists in that case — the Wald is a cross-check, not a
gate.

---

## 4. LRT vs WALD: THE DISAGREEMENT, REPORTED RATHER THAN SMOOTHED

### 4.1 They can give OPPOSITE verdicts

`[MEASURED]`, CPU, `n = 120`, a 30-step model with `β` **planted** at `1.02`:

```
beta = 1.02:  Lambda = 7.205966 (rejects)   Wald = 1.150026 (does not reject)
wald_agrees: False   rel_gap: 0.8404
```

Same `β`, same `3.841`, **opposite sides of it**. This is a finding about the
**asymptotics**, not the implementation: the LRT–Wald equivalence is a
second-order expansion of `LL` **about the MLE**, and off the MLE that expansion
carries a first-order **score** term which the Wald form drops entirely. A
`β_final` from a real run is fit on the **training** split and is therefore
never the eval split's maximiser, so **this is the normal case, not a corner.**

**Consequence for the card: the Wald is not a free confirmation of the LRT, and
neither may be reported alone.** `lrt_report` prints `DISAGREES` on the row when
they split, and the card says so
(`test_the_two_forms_disagree_where_the_asymptotics_stop_and_that_is_reported`).

### 4.2 At the MLE they track, and where they stop tracking is measurable

`β` driven to the **eval-split** MLE by Newton on `β` alone (score
`[-1.32e-07, -1.86e-07]`), same batch, `n = 120` `[MEASURED]`:

| `d = \|β−1\|` | resolution `\|β−1\|_min` | `d/res` | `Λ` | Wald | rel gap |
|---|---|---|---|---|---|
| `0.011539` | `0.055816` | **0.21** | `0.1563` | `0.1642` | **4.8 %** |
| `0.179773` | `0.086842` | **2.07** | `26.3359` | `16.4602` | **37.5 %** |

Inside the resolution the two forms agree to 5 %; two resolutions out the
quadratic approximation fails on its own terms and they part by 37 %, **while
still agreeing on the verdict**. Both rows are asserted, in opposite directions,
so neither the agreement nor the disagreement is a tautology
(`test_the_wald_form_and_the_lrt_agree_at_the_mle_for_a_small_departure`,
`test_the_two_forms_disagree_…`).

### 4.3 The full report at three shapes

**CPU, trained 30 steps, `n = 120`** `[MEASURED]`:

```
RULING 10' LRT -- n_eval = 120, chi2_1(0.95) = 3.841, ln n = 4.7875, n_beta = 2
  JOINT  Lambda = +26.5098  [3.841 / ln n = 4.7875]  MOVED   dof = 2  <- 3.841 is the 1-dof constant; the per-parameter rows below are the 1-dof statistics
  model.layers.0.self_attn.beta  beta=1.200747  Lambda=+26.4055  [3.841 / ln n = 4.7875]  MOVED  |beta-1|_min=0.087370  Wald=+20.2776 (agrees)  I_beta_total=503.173
  model.layers.1.self_attn.beta  beta=0.998378  Lambda=+0.0408  [3.841 / ln n = 4.7875]  PINNED  |beta-1|_min=0.059908  Wald=+0.0028 (agrees)  I_beta_total=1070.23
  PINNED means indistinguishable AT THIS RESOLUTION, never exact: |beta-1|_min = sqrt(3.841 / (n * I_beta)).
```

**cuda, the smallest binding shape, trained 30 steps, `n = 120`** `[MEASURED]` —
and this one produced the **interval verdict** without being made to:

```
RULING 10' LRT -- n_eval = 120, chi2_1(0.95) = 3.841, ln n = 4.7875, n_beta = 2
  JOINT  Lambda = +50.4902  [3.841 / ln n = 4.7875]  MOVED   dof = 2  <- ...
  model.layers.0.self_attn.beta  beta=1.262072  Lambda=+45.3712  [...]  MOVED  |beta-1|_min=0.112919  Wald=+20.6897 (agrees)  I_beta_total=301.24
  model.layers.1.self_attn.beta  beta=0.960661  Lambda=+4.5635   [...]  rejected at 0.95, below description-length  |beta-1|_min=0.029114  Wald=+7.0128 (agrees)  I_beta_total=4531.49
restore, torch.equal, all 39 params: True
```

**Untrained at `β = 1`** — the `Λ = 0` half, with `undefined` resolutions and
their notes: §3.3 above.

**All three verdict strings therefore occurred on real measurements**, not only
in the boundary unit tests.

---

## 5. THE AUTHOR'S TWO `[RUN]` FIGURES

### 5.1 `ln n = 8.29 at n = 4000` — REPRODUCES

```
ln 4000 = 8.294050  ->  2dp  8.29     author says 8.29     REPRODUCES: True
```

Arithmetic, and it is right `[MEASURED]`.

### 5.2 `a real 2 % departure is detected only 32 % of the time at n = 4000` — REPRODUCES ONLY AS A CONDITIONAL

**Method.** For a 1-dof test the noncentral chi-squared is exactly `(Z + √λ)²`,
so the power is two normal tails and needs no scipy:

```
power(λ) = Φ(√λ − √3.841) + Φ(−√3.841 − √λ)
```

**What the figure requires.** Solving `power(λ) = 0.32` gives
**`λ = 2.224178`**. With `λ = (β−1)²·n·I_β`, `β−1 = 0.02` and `n = 4000`:

> **the figure implies `I_β = 1.390111` per observation.**

The ruling never states an `I_β`. So the 32 % is **not derivable from the
ruling's own text** — it is conditional on a value the reader is not given.

**Internal consistency, which does hold.** At that `I_β`,
`|β−1|_min = 0.026283`, and `0.02 < 0.026283`, so the 2 % departure sits **below
the printed resolution** and its power must be under 50 %. The two figures are
therefore mutually consistent with each other; they are just not self-contained.

**What this box measures instead.** From the eval-split-MLE model of §4.2,
`[MEASURED]`:

| parameter | `I_β` per observation | power of a 2 % departure at `n = 4000` |
|---|---|---|
| author's implied value | `1.390` | **32.0 %** |
| `model.layers.0.self_attn.beta` | `4.2443` | **74.1 %** |
| `model.layers.1.self_attn.beta` | `10.2742` | **98.2 %** |

**So the answer to "do you get 32 %" is: no — 74 % and 98 %, at an `I_β` three to
seven times the one the figure assumes.** No assumption was adjusted to make it
match. The caveat that matters: this box's `I_β` comes from a **27,914-parameter
model trained 30 steps on one repeated batch**, which is not the shape the figure
is about, and `I_β` is a property of the trained model and its data — so this
disagreement is **not** evidence that the author's figure is wrong for the shape
he had in mind. It is evidence that **the figure cannot be checked without its
`I_β`, and that `I_β` must therefore be printed on the card**, which is exactly
what §3.2 makes `lrt_report` do.

### 5.3 The notebook's own eval shape

`batch 8 × seq 512` → `n = 8·511 = 4088`, `ln n = 8.3158` — within 0.02 of the
figure's `n = 4000`, so the author's worked case is close to the shape Q3 will
actually read `[MEASURED]`.

### 5.4 A property of the two constants, recorded because nobody chose it

`ln 46 = 3.8286 < 3.841 < 3.8501 = ln 47`. **For `n ≤ 46` the interval verdict is
unreachable** and every rejection is a MOVED. That is a consequence of the two
shelf constants, **not a rule invented here**, and it is handled by printing
`ln n` beside every verdict rather than by patching an interpolation
(`test_the_interval_is_empty_below_47_eval_items_and_that_is_not_an_interpolation`).

---

## 6. WHERE RULING 10′ MEETS RULING 1'S HOLE

`[MEASURED]`, cuda, under `torch.use_deterministic_algorithms(True)` (strict, no
`warn_only`):

```
forward under STRICT: OK, repeat bitwise: True  (loss 0.28433388471603394)
beta_lrt under STRICT RAISED: cumsum_cuda_kernel does not have a deterministic
                              implementation, but you set
                              'torch.use_deterministic_algorithms(True)'
```

The split falls **exactly where RULING 1 says the hole is**:

| part of the criterion | passes taken | strict mode |
|---|---|---|
| `Λ` and its three-way verdict | **two forwards** | **executable, and bitwise** — inside B2's regime |
| `I_β`, the Wald cross-check, `\|β−1\|_min` | one forward + **two backwards** | **RAISES** — autograd differentiates `cumprod` with `cumsum`, and `cumsum_cuda_kernel` has no deterministic implementation |

This is the same measurement `COSTS.md` §1.6 recorded for the arm's gradient,
reached from a different direction.

**CONSEQUENCE, FLAGGED AND NOT DECIDED HERE.** RULING 10′ calls a verdict printed
without its resolution a **defect**, and the resolution is not computable under
strict mode. **Therefore a Λ cell cannot be a B2 strict deciding cell.** Under
the round's actual regime (`warn_only=True`, RULING 1) everything runs, and the
Q3 chunk cell is already in that regime because it trains. This is a question for
the **explicit deciding-cell list RULING 6f freezes at launch** — the envelope
node's row — and it is not something `beta_lrt` decides by swallowing the error.
Pinned by `test_lambdas_passes_are_strict_safe_and_the_hessian_read_is_not`.

---

## 7. WHAT WAS RETIRED, AND WHERE

`5·δ_β` is retired **as the criterion** and kept **as machinery**, everywhere it
was written as the criterion. Nothing was deleted; `scripts/k_noise_floor.py` was
not touched and keeps measuring `δ_β`, which is correct.

| file | what changed |
|---|---|
| `ceq/hf/modeling_ceq.py::beta_summary` | docstring gains a paragraph: `pinned`/`branch`/`pinned_fraction` **are now a diagnostic**, the criterion is `beta_lrt`, and the reason the old one went (bitwise pair → zero tolerance) |
| `ceq/hf/modeling_ceq.py::_pinning` | docstring retitled **"THE `5*delta_beta` DIAGNOSTIC. NOT the pin criterion any more."**; the retired rule is kept verbatim underneath because the code below still implements it; the degeneracy paragraph now records that the re-take **measured** it rather than that it was predicted |
| `ceq/hf/modeling_ceq.py::_pinning` **return value** | new `criterion` key on **both** branches, so a consumer that never reads a docstring is told in the artifact. `branch_reason` now adds *"The VERDICT does not wait on it — see `beta_lrt`."* |
| `MODEL_CARD.md` Limits | the `k = 5` heuristic bullet **replaced** by four: the resolution bullet, the retirement bullet, the census bullet, and the Wald-can-contradict bullet |
| `MODEL_CARD.md` pin-criterion section | rewritten as RULING 10′ — the formula, the three-row verdict table, both constants, the Hessian method, the `I_β ≤ 0` case, and the Wald contradiction with its measured numbers |
| `MODEL_CARD.md` new subsection | **"`δ_β` — RETIRED AS THE CRITERION BY RULING 10′, KEPT AS A DIAGNOSTIC"**, including *why* the two will disagree by construction |
| `MODEL_CARD.md` Branch A | the two sentences that **defined** pinned as `\|β−1\| ≤ 5·δ_β` now define it as `Λ ≤ 3.841` with the resolution attached. **The three branches' conditions and sentences are unchanged**, per the brief |
| `MODEL_CARD.md` slots | `DELTA_BETA_PER_PARAM` **relabelled a diagnostic and marked NO LONGER BLOCKING**; `BETA_FINAL_DIST`'s fraction-pinned now reads off the LRT; two new slots, `LRT_VERDICT` and `EVAL_SPLIT_SPEC` |

**Backwards compatibility is measured, not asserted:** `test_g13_beta_learnable.py`
(the β node's 32 tests + 2 xfails) is **green throughout** — `pinned`,
`pinned_fraction`, `branch`, `n_degenerate_floor`, `word` and `k` all still
compute and still return the same values. Only prose and one added key changed.

**A retirement that is verified rather than claimed.**
`test_the_card_carries_ruling_10_prime_and_no_longer_states_5_delta_beta_as_the_criterion`
asserts the card contains `RULING 10`, `3.841`, the interval verdict's exact
words and the power-line formula, **and that the pre-10′ criterion sentence is
absent verbatim**, and that `δ_β` is still present and labelled a diagnostic.

---

## 8. THE NOTEBOOK

`kaggle/ceq_v17k.ipynb`, **cell 18 only**, appended after `T.train(...)` and its
summary prints. `[MEASURED]` on the edit itself:

* **22 cells before, 22 cells after**; the only differing cell index is **18**;
  every cell `id` is preserved; the top-level notebook keys are byte-equal.
* Cell **15**, the Q2 tombstone, is untouched.
* CRLF line endings and `indent=1` JSON formatting preserved — the whole-file
  `diff` is **78 lines**, all inside cell 18.
* The appended 73 lines parse as Python (`ast.parse`).
* Nothing is stubbed.

**The eval split is held out by construction, not by arithmetic done in the
cell.** `ceq/hf/train.py::ByteBatches` cuts the corpus 90/10 contiguously and
`train()` draws only from `.train`; the cell builds the **same object from the
same bytes** (`max_bytes` read off `T.train`'s own signature rather than
retyped) and draws from `.val`. Nothing re-derives an offset that could drift
from the one training used. The generator seed is fixed so the eval batch is the
**same** batch every chunk — otherwise `Λ` is not comparable across the chunks it
is printed in.

The cell prints `lrt_report(...)`, then the **restore proof** (`torch.equal` over
every parameter, with an `assert`), then the retired `δ_β` diagnostic beside the
verdict it no longer decides, and writes `/kaggle/working/lrt_verdict.json`.
Pinned by `test_the_notebook_computes_lambda_at_the_end_of_q3_without_renumbering_anything`.

---

## 9. FINDINGS FOR THE AUTHOR

**(1) THE DOF SENTENCE AND THE `β ≡ 1` SENTENCE DO NOT AGREE WHEN `n_layers > 1`.**
RULING 10′ writes `Λ` over `β ≡ 1` — *every* `β` — and calls it a **1-dof**
comparison decided by **χ²₁**. `β` is one scalar **per layer**, so those are the
same statistic only at `n_layers == 1`. Nothing was invented to close this:
`beta_lrt` emits the ruling's literal joint `Λ` **with `dof_joint = n_beta`
printed beside it**, and the `per_parameter` rows — each holding **one** `β` out
at a time — are the genuinely 1-dof statistics `3.841` licenses, which is also
the per-parameter reporting 10′ leaves standing. At `n_layers = 1` the two are
**bitwise identical**, asserted. `lrt_report` prints
*"3.841 is the 1-dof constant; the per-parameter rows below are the 1-dof
statistics"* whenever `dof_joint > 1`. **No χ² constant for `dof > 1` was
introduced.** The author's call is which of the two the card's branch reads.

**(2) THE WALD CAN CONTRADICT THE LRT, AND IT IS NOT A RARE CASE.** §4.1. The
equivalence holds at the MLE; `β_final` is fit on the training split and is never
the eval split's maximiser, so a contradiction is the **expected** situation, not
a pathology. Measured: `Λ = 7.21` rejects while `Wald = 1.15` does not. The
implementation prints both and flags `DISAGREES`; **which one the card's branch
reads when they split is not a decision this node may make.**

**(3) A `Λ` CELL CANNOT BE A B2 STRICT DECIDING CELL.** §6. `Λ` is strict-safe;
its mandatory resolution is not. Under RULING 1 the round runs `warn_only=True`
and everything works, so this only bites if the Λ cell is put on RULING 6f's
frozen list as a strict forward-only cell. Flagged for the envelope node.

**A fourth, smaller one, and it is this node's own bug rather than the author's.**
`scripts/v15_r1.py:575` sets `torch.use_deterministic_algorithms(True,
warn_only=True)` process-wide and never restores it, and
`tests/gate0/test_g14_instrument_hash.py` calls into it — so by collection order
the rest of `tests/gate0` inherits `warn_only=True`. That is harmless on its own.
It became harmful when this node's first draft restored the flag from
`are_deterministic_algorithms_enabled()` alone, which **drops the `warn_only`
half** and therefore turned the ambient `(True, warn_only=True)` into **STRICT**
for every test that ran afterwards — a run of `tests/gate0` read `5 failed`, then
`2 failed`, then would have read differently again. **Fixed inside this node's
own file**, not by editing anyone else's: an autouse fixture pins the round's
regime and restores **both halves**, and the assertion that would have caught it
is now in the test. No other node's assertion was weakened. The underlying
non-restoring `set` in `scripts/v15_r1.py` is **left alone** — that file is not
this node's and the fixture makes this file immune to it — but it is recorded
here because the next file that flips that flag will hit the same thing.

---

## 10. LIMITS

Every number in this file is from a **27,914-parameter shape**, `float32`, on
probes of **0–30 gradient steps over one repeated batch**; none is a training
result, a device timing, or a reading of the arm. `I_β` is a property of the
trained model **and its data**, so the `I_β` values in §5.2 do not transfer to
Q3's shape and are not offered as predictions of it — they are offered as the
demonstration that the author's 32 % figure cannot be checked without an `I_β`
printed beside it. The Newton fit in §4.2 drives `β` to the **eval-split** MLE
purely to test a claim about the asymptotics; **it is not part of the criterion
and is not called by `beta_lrt`, the card or the notebook.** `Λ` is a
likelihood statistic and not a distance one: a `β` far from `1` that fits worse
than `β = 1` reads **PINNED**, correctly, and the retired `δ_β` diagnostic reads
**MOVED** on the same model — the two are not nested and will disagree by
construction (§7, and the measured case in
`test_a_beta_far_from_one_that_the_data_dislikes_reads_pinned_and_that_is_the_point`).
The eval split's hold-out is guaranteed by `ByteBatches`' own cut and **not** by
anything `beta_lrt` checks: it scores whatever batch it is handed, and a caller
that hands it training data gets an in-sample `Λ` with nothing raised. Q3 has not
run, so **no branch has been taken by anything here**, and the card's
⟨`LRT_VERDICT`⟩ is a slot, not a prediction.

---

*The criterion is now unblocked: unlike `5·δ_β`, it waits on no floor, no
identical-seed pair and no other node's measurement — only on the training run
itself. That is the one respect in which this file moved toward the north star,
and it moved a gate rather than a number.*
