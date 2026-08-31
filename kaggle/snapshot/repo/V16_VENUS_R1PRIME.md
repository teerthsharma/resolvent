# V16 VENUS (it. 6) — R1′, filed before the data exists

**Filed:** `2026-08-31T15:24:41Z` (UTC)
**git HEAD at filing:** `8bcef230e030806e68c92036d9fc967cf308cf1e`
(`8bcef23` — *"Rewrite the README from scratch, and file the status report with
every number's source"*)

Discharges `CEQ_V16_CONTRACT.md` PART VI it.6, VENUS half. Predicts
**R1′**: BED-M, `t* = 2`, `n = 2048`, `N = 8`, seeds `0..7`, on
**`ceq/arm_smprime.py::ArmSMPrime`** — not `ceq/arm_phase.py`, and not the
"phase gates" PART IV's prose names.

**Files written:** this one. Nothing under `ceq/`, `scale/`, `lean/`,
`scripts/`, `tests/`, `results/` or `MISTAKES.md` was touched. No git command
that writes was run. **Nothing was trained; no arm was run; no cell was run.**
The only computation performed is §2.3's closed-form minimisation, which
touches no corpus, no arm and no checkpoint.

---

## 0. THE TARGET DOES NOT EXIST YET — CHECKED, NOT ASSUMED

| thing | check | result |
|---|---|---|
| any R1′ row | `results/` searched for `smprime`, `sm_prime`, `r1prime`, `r1_prime`, `arm_smp` (ripgrep, case-insensitive, 155 files) | **no files found** |
| any runner | `ls scripts/` | `v16_device_probe.py` is the only `v16_*`; **no `v16_r1prime.py`, no R1′ script of any name** |
| `arm_smprime` **committed** anywhere in the tree | `git grep -l -i smprime` excluding `house-events*.jsonl` | **three hits, all it.5 artifacts**: `V16_ARM_SMPRIME.md`, `ceq/arm_smprime.py`, `tests/arm_smprime/test_arm_smprime.py` |
| `λ̂` / unit-root for any arm, **committed** | `results/v15_r1.jsonl` key census (81 keys), and `grep` over committed `scripts/v15_r1.py` | **no `lam`, `lambda`, `lyap`, `unit_root` or `winding` key exists** |

**THE TREE MOVED UNDER THIS FILING AND IT IS RECORDED RATHER THAN SMOOTHED.**
`HEAD` advanced twice while this node read (`deee6c4 → 91b862d → 8bcef23`), and
at the moment of writing `git status --porcelain` carries an **uncommitted**
`M scripts/v15_r1.py`, `+317 / −67`, which is another node building the R1′
runner concurrently. **VENUS wrote none of it.** Its staged content is read
read-only in §4.2 and it **sharpens the early-warning prediction rather than
weakening it**, so it is folded in here, before any data, with its uncommitted
status stated. **No R1′ cell, journal or result exists in either the committed
or the working tree.**

**If any reader finds an R1′ cell dated before the timestamp above, this filing
is void and must be reported as void rather than scored.**
`house-events.jsonl` was not consulted and was not grepped; nothing here needed
the ledger, so no `scale/ledger.py` parse was run.

---

## 1. THE FILING IN ONE TABLE

`floor₁ = 0.7071067812`. `NRMSE < floor₁ ⟺ ĥ > 1`, one event, paid once.

| quantity | AUTHOR's PREDICTION | AUTHOR's COUNTER (D-CALIB-1: the point estimate) | **VENUS** |
|---|---|---|---|
| seeds that converge (finite, 150 steps) | 8/8 | 8/8 | **8/8** — agreed, and §2.1 says why the agreement is not a prediction |
| seeds crossing `floor₁` | **5/8 banks** | **≤ 3/8** | **0/8**, 80% interval on the count **[0, 1]** |
| seed mean NRMSE | ≈ 0.64 (the crossing cluster) | ≈ 0.83 (R11's shape, one cluster thinned) | **0.92**, 80% interval **[0.83, 0.98]** |
| seed **sd** | — | — | **0.06**, 80% interval **[0.02, 0.13]**; hard claim **sd < 0.15** |
| 95% CI vs `floor₁` | does not straddle — **entirely below** | straddles, as in R11 | **does not straddle — entirely ABOVE.** Point CI `[0.870, 0.970]` |
| distribution shape | one cluster, below | two clusters, R11's gap `[0.663, 1.113]` still empty | **unimodal, and it FILLS the gap**: **≥ 6 of 8 seeds inside `[0.88, 1.00]`** |
| `â_max` | ≤ 1 | ≤ 1 | ≤ 1 — **non-discriminating by construction, see §2.1** |
| seeds above 1.10 (R11's divergent band) | 0 | 0 or more | **0 of 8** |

**Scoreboard consequence, stated before it.8:** under this filing R1′ pays
**0**, not `+12` and not the partial credit a `≤3/8` crossing would carry.

---

## 2. THE MECHANISM

### 2.1 `â_max ≤ 1` is not a prediction. It is `torch.clamp`.

`CEQ_V16_CONTRACT.md:196` credits the convergence half to *"`â_max ≤ 1` by
theorem #16"*. The shipped arm reaches it earlier and more cheaply:

```
ceq/arm_smprime.py:113   return torch.clamp(u, 0.0, 1.0)
ceq/arm_smprime.py:129   m = magnitude(torch.lerp(torch.ones_like(u), u, g))
```

`m ∈ [0,1]` is a property of the arithmetic at **every** value of the `g`
switch — `V16_ARM_SMPRIME.md` §7 records the one-line reordering that made it
so, and the sweep `g ∈ {−2, −0.5, 0, 0.5, 1, 2, 25}` over 4,096 draws that
holds it. R11's divergence mechanism was *an open range chasing a boundary
target*, `â_max` reaching `20.31 / 49.66 / 285.07` on seeds 2, 3 and 7. That
range is closed by a clamp on the forward pass.

**So "8/8 converge, `â_max ≤ 1`" cannot be scored as a correct prediction. It
is a restatement of a line of shipped code**, and the calibration column at
it.38 should record it as **not a statement under test** rather than as a hit.
VENUS agrees with the outcome and files the agreement as a non-event.

What is left to predict is the **cost side**, and both halves of the author's
filing put that cost in the wrong place.

### 2.2 The arm initialises at the corner where the label is NOT representable

This is the load-bearing fact of the filing and it is three lines of the
shipped module:

```
ceq/arm_smprime.py:508   self.beta = nn.Parameter(torch.tensor(1.0))
ceq/arm_smprime.py:509   self.qk   = nn.Parameter(torch.tensor(1.0))
ceq/arm_smprime.py:510   self.g    = nn.Parameter(torch.tensor(1.0))
```

and `identity_heads()` sets the same three to `1.0` (`:523`). The module's own
comment is explicit: *"the three switches, at the softmax corner's settings"*.

Set that against where the round's headline bind actually holds
(`V16_ARM_SMPRIME.md` §3.1, verdict (a)):

| setting | residual against BED-M's own `equilibrium_oracle` |
|---|---|
| **`β = 0`, QK **off**, `(u,θ) = (\|a\|, arg a)`, `V = b`** | **`5.919777e-16`**, real part exactly `0.000000e+00` |
| **`β = 1`** (the initialiser) | **`1.335288`** — the `beta_one` planted negative, verdict (f) |

`V16_ARM_SMPRIME.md` verdict (h), in the node's own words: at `β = 1` *"the
label bind **fails at `1.335288`** and no finite compensation exists at `m = 1`
… because the softmax row has divided out the magnitudes the label needs."*

**R1′ therefore begins training at a corner where the quantity it is being
trained to produce is provably not representable, and must travel `β: 1 → 0` on
a scalar MSE to reach the corner the round's headline result was measured at.**
`β` is one scalar with an unobstructed gradient (`∂W/∂β = −W log Z_i`), and
AdamW at `lr = 0.02` moves a scalar by at most `lr` per step, so the trip costs
**≈ 50 of the cell's 150 steps** if the gradient's sign is consistent — and it
is competing for those steps with a magnitude head that must find 2 live
positions in 64 and a phase head that cannot start until the magnitude head
has.

**The coupling is sequential and it is new.** `∂(m e^{iθ})/∂θ = i·m·e^{iθ}`,
which is **exactly zero at `m = 0`** — and `V16_ARM_SMPRIME.md` §6.1 measures
`55.5817%` of positions sitting at `m = 0` after 40 steps from the constructed
initialiser, on all four device/dtype configurations. The sign channel, which
is the whole of what BED-M's label carries, receives no gradient anywhere the
magnitude has annihilated. ARM PL had no such gate: its 5 crossing seeds read
`p = 1.000000` exactly, sign recovery was never the binding constraint, and it
crossed **while its expressibility proof did not cover the corpus at all**
(`V15_R1.md` §1: the `(L)` setting needs `0 < a < 1`; BED-M supplies
`{−1, 0, +1}`; `0 of 2048` sequences satisfy the hypothesis).

**That is the inversion this filing rests on.** R1 was never
expressibility-limited — it was optimisation-limited, and it crossed 5/8 anyway
with an operator that could not represent its target. R1′ repairs
expressibility to `5.9e-16` and **adds** an optimisation obstacle (`β`) plus a
sequential coupling (`θ` behind `m`). Repairing the half that was not binding
while loading the half that was does not raise the crossing count. It lowers it.

### 2.3 The author's counter names a mechanism that does not reach its conclusion

> *"the crossing shrinks to ≤ 3/8, because the magnitude cap removed gain the
> divergent seeds were exploiting — stability bought at capability's price."*

Two arithmetic objections, both computable before the data.

**(i) The divergent seeds contributed zero crossings.** R11's crossing count
was 5, from seeds `{0, 1, 4, 5, 6}`. The seeds carrying the large gain were
`{2, 3, 7}` at `â_max = 20.31 / 49.66 / 285.07` — and they read
`1.152430 / 1.113403 / 1.139404`, **above** `predict_the_mean = 1.0`. Removing
gain from three seeds that crossed nothing cannot lower a count they never
entered. To get `5 → ≤3` the cap must break at least two of
`{0, 1, 4, 5, 6}`, whose `â_max` were `1.4107, 1.2868, 1.4390, 1.5051, 1.1029`
— between `10%` and `51%` above the cap, and that loss must be worth the
`0.045–0.073` of NRMSE those seeds held below the floor.

**(ii) The cap does not bind at the optimum, and the round's own registered
NRMSE model says so in closed form.** `V15_VENUS_PREDICTIONS.md` §0 registers
`NRMSE(t*, c) = √(1 − Cov²/(t*·Var))`, `Cov = Σ_{m=1..t*} ρ^m c^m`,
`Var = Σ_{m=1..s−1} ρ^{2m}`, minimised over the decay `ρ`. Minimised here at
`s = 64, t* = 2` on a 300,000-point grid, with and without the constraint
`ρ ≤ 1` — **no corpus, no arm, no training, closed form only**:

| `c` | unconstrained `ρ*` | best NRMSE | `ρ ≤ 1`: `ρ*` | best NRMSE | **what the cap costs** |
|---|---|---|---|---|---|
| 1.0000 | **0.5000** | 0.395285 | 0.5000 | 0.395285 | **`+0.000000000`** |
| 0.9500 | 0.4913 | 0.513365 | 0.4913 | 0.513365 | `+0.000000000` |
| 0.9000 | 0.4819 | 0.600699 | 0.4819 | 0.600699 | `−0.000000000` |
| 0.8500 | 0.4717 | 0.669887 | 0.4717 | 0.669887 | `+0.000000000` |
| **0.8182** (the crossing threshold) | 0.4647 | **0.707139** | 0.4647 | 0.707139 | `+0.000000000` |
| 0.8000 | 0.4606 | 0.726541 | 0.4606 | 0.726541 | `+0.000000000` |
| 0.7000 | 0.4350 | 0.813616 | 0.4350 | 0.813616 | `−0.000000000` |

Closed-form check: `Cov²/(2·Var) = (1+ρ)³(1−ρ)/2`, maximised at
`ρ* = 1/2` exactly, giving `√(1 − 27/32) = 0.395285` at `c = 1`. The optimum
sits at **half** the cap, and the dead band's leak at `ρ^m` is what pushes it
there.

**A cap at `1.0` is inactive at an optimum at `0.5`. The author's stated
mechanism costs the crossing seeds nothing in the round's own model, and his
conclusion needs it to cost them everything.** He is filing the right
*direction* (the crossing shrinks) attached to the wrong *cause*. §2.2 supplies
a cause that reaches further than his conclusion does — which is why VENUS's
number is `0/8` and not `≤3/8`.

### 2.4 Where the non-crossing seeds land, and why it is a new place

A `β`-trapped arm is a softmax row with a phase in the numerator. The softmax
control was measured at **this exact cell, these exact eight seeds**
(`results/v15_r1.jsonl`, recomputed here from the journal):

```
softmax   n=8  mean=0.951767  sd=0.011824  min=0.933802  max=0.971432
arm_pl    n=8  mean=0.829151  sd=0.253673  min=0.634002  max=1.152430
```

`sd = 0.011824` against `0.253673` — a factor of **21.5**. That ratio is the
filing's cheapest instrument and it costs no extra run.

R11 read **bimodally with nothing between `0.663` and `1.113`**. VENUS predicts
R1′ reads **unimodally, in the band R11 left empty**, because `β` is a
*continuous* scalar and partial normalisation buys partial capability: a seed
whose `β` reaches `0.3` is neither at the softmax corner nor at the path-product
corner, and lands between them. A continuum of `β` produces a continuum of
NRMSE, which is precisely the shape R11 did not have.

---

## 3. THE DISCRIMINATING OBSERVATIONS

A prediction no observation could refute is not filed. Five, ranked by cost,
all computable from the same eight cells with no extra run.

### 3.1 The seed `sd` — one number, three-way

| reading | verdict |
|---|---|
| **`sd < 0.10`** | **VENUS** — unimodal; the two-cluster structure did not survive the arm change |
| `0.10 ≤ sd ≤ 0.18` | neither; the continuum reading of §2.4 takes partial credit and both filings are scored `wrong` |
| **`sd > 0.18`** | **the AUTHOR's COUNTER** — bimodality survived, the crossing cluster merely thinned |

R11's `sd = 0.253673`. Softmax's `sd = 0.011824`. The two regimes are 21×
apart; the boundary is not a fine judgement.

### 3.2 The sign of `mean − floor₁` — separates VENUS from VENUS-COUNTER

`sd` cannot tell a tight cluster **above** the floor from a tight cluster
**below** it. The mean's sign against `0.7071067812` does, and §5 files the
below-the-floor half against VENUS's own name at equal specificity.

### 3.3 Final `β` per seed — the mediation claim, and it is the mechanism itself

**VENUS predicts `β` is the mediator, and files it as a correlation on 8 points:**

- **`β_final ≥ 0.5` on ≥ 6 of 8 seeds**, and `Σ_j |W_ij| ∈ [0.95, 1.05]` on
  those seeds' rows — the `β = 1` signature, measured at
  `V16_ARM_SMPRIME.md` §5.1 as `1.000000` on **every** row at `β = 1` against
  `[1.312192 … 10.293107]` at `β = 0`.
- **`corr(β_final, NRMSE) > +0.6`** across the eight seeds.
- **`corr(β_final, gate-R²) < −0.6`.**
- Any seed that **does** cross has `β_final < 0.3`. No seed crosses with
  `β_final > 0.5`.

**The author's counter predicts none of this** — under "the cap removed gain",
`β` is not the axis and the crossing/non-crossing split is a gain story, so
`β_final` should be uninformative. **`β` is one scalar already in the
journal-able state of the module. Printing it per seed settles the two filings
in one column.** If `β_final` is near `0` on the non-crossing seeds, VENUS's
mechanism is refuted whatever the crossing count is, and VENUS says so here
rather than after.

### 3.4 Zero seeds above 1.10

R11's divergent band was `[1.113403, 1.152430]`. **VENUS predicts it is empty.**
`≥ 1` seed reading `> 1.10` refutes the `β`-trap story directly — it would mean
a divergence mode survived a clamp that makes the R11 mode impossible, and the
author's "capability's price" framing would be closer than this filing.

### 3.5 What is NOT a discriminator, said so it is not scored as one

`â_max` (§2.1, a clamp), and the **step-0 gate-`R²` column** (§4.1, already on
disk).

---

## 4. THE EARLY-WARNING COLUMN

`CEQ_V16_CONTRACT.md:201-203`: *"`λ̂` and gate-`R²` from step 0 per seed;
diverging seeds (if any) **must show `λ̂ > 0` before loss does**, or the
boundary story is wrong and gate-`R²` says where."*

### 4.1 Half of this column is a lookup, not a prediction

Step-0 gate-`R²` for `ArmSMPrime` on eight seeds is **already printed**
(`V16_ARM_SMPRIME.md` §8, at `n = 512`):

```
gate R^2, ZERO-STEP ArmSMPrime x8  = 0.291617
  per seed [0.196315, 0.146722, 0.246818, 0.594783, 0.086057, 0.322347, 0.095066, 0.64483]
```

and it is **identical to ARM PHASE's to the last printed digit**, because both
modules construct `wq, wk, m_head, theta_head` in the same order and draw the
same bytes under `torch.manual_seed`. **VENUS files no prediction for it and
asks that it.38 not score it as one**: a node that predicts `0.291617` and reads
`0.291617` has read `V16_ARM_SMPRIME.md`, not the future. The only movement
available is `n = 512 → 2048` and the eval batch; VENUS registers
**`|Δ| < 0.08` on the eight-seed mean** and treats a larger move as a finding
about the instrument's `n`-dependence rather than about the arm.

### 4.2 `λ̂ > 0` is not merely absent. For this arm it is ARITHMETICALLY IMPOSSIBLE.

The contract's must-fire is *"diverging seeds must show `λ̂ > 0` before the loss
does, or the boundary story is wrong."* The uncommitted runner defines the
column (`scripts/v15_r1.py::gate_columns`, working tree, read read-only):

```python
elif kind == "arm_smprime":
    m, theta = arm_smprime.blend(*model.heads(x), model.g)
    m, theta = m[:, live], theta[:, live]
    lg = torch.log(m)
...
lambda_hat      = float(lg.mean())
lambda_hat_live = float(lg[fin].mean())          # fin = isfinite(lg)
a_hat_max       = float(lg.max().exp())
unit_root       = bool(a_hat_max >= 1.0)
```

and `blend` returns `m = clamp(lerp(1, u, g), 0, 1)`, so **`m ∈ [0,1]`
unconditionally** (`ceq/arm_smprime.py:113,129`; the sweep in
`V16_ARM_SMPRIME.md` §7 holds it at every `g`). Therefore

```
m ≤ 1   ⇒   log m ≤ 0   ⇒   λ̂ = mean(log m) ≤ 0   and   λ̂_live ≤ 0
```

**at every step, on every seed, under every initialisation, for every value of
the three switches.** `λ̂ > 0` is not unlikely for this arm; it is unreachable.

**So the contract's early-warning must-fire has an EMPTY REJECTION REGION** —
the `V-24` class the round's own arm node names one page earlier: *"a planted
negative that cannot reach the object it mutilates is `V-24`'s empty rejection
region with the sign flipped."* The very clamp that buys the convergence the
author predicts is the clamp that makes his divergence detector unfireable.
**A round that records "the must-fire held" at it.7 records `0 ≤ 0`.**

This is stronger than the vacuity VENUS filed first. Two independent reasons the
antecedent is empty — no diverging seeds (§2), and no reachable `λ̂ > 0`
(here) — and only the second is immune to being wrong about the seeds.

**What the column actually reads, predicted per seed:**

| column | VENUS's prediction at step 0 | mechanism |
|---|---|---|
| `λ̂` (the headline) | **`−inf` on 8 of 8** | `lg.mean()` over the live band; `V16_ARM_SMPRIME.md` §6.1 measures `55.5817%` of positions at `m = 0`, so at least one of the `n·\|live\| = 4096` live entries is `log 0 = −inf` with overwhelming probability, and one `−inf` sets the mean |
| `λ̂_live` | **strictly negative on 8 of 8**, in `[−1.5, −0.2]` | the finite-only mean of `log m`, `m < 1` |
| `frac_gate_annihilated` | `0.30 … 0.55` on 8 of 8 | the same head that gives `55.58%` over all positions |
| `unit_root` (`= â_max ≥ 1`) | **`False` on ≥ 5 of 8** | `â_max = max(m) ≤ 1`, and `= 1` only if some live position sits **exactly** on the closed cap; at the default `nn.Linear` init the head has not yet been driven there |
| `dyn_range_bound` | **FINITE on ≥ 4 of 8 — the first finite reading in the campaign** | R11: *"undefined at every one of the eight seeds"*, `â_max ∈ [1.1029, 285.0719]` |

**`λ̂_live` is a PROGRESS meter, not a divergence meter, and the round should
relabel it before it.7 rather than after.** A trained arm that has found BED-M's
band reads `m = 1` on the live positions, hence `λ̂_live = 0.000000`,
`frac_gate_annihilated = 0`, `â_max = 1.0` and `unit_root = True`. **The flag
fires when the arm is RIGHT.** Its R11 semantics were the opposite: `unit_root`
meant the operator was expanding out of control. **VENUS predicts `λ̂_live`
rises monotonically toward `0` on the seeds whose `β` escapes and stalls below
`−0.2` on the trapped ones**, giving a third read on §3.3's mediation claim from
a column already being written.

**Provenance correction, filed rather than quietly dropped.** Against the
*committed* tree this section read `P(the λ̂ / unit-root column is produced at
it.7) = 0.35`, because the R1 journal's 81 keys contain no `lam`, `lambda`,
`unit_root` or `winding` key, `scripts/v15_r1.py` as committed computes none,
and the only `λ̂` estimator in the tree is `scripts/v13_tangent_kit.py`
(`ψ := ln R_ii`, `lam_hat = lsum / lcnt` at `:842`), written for iterated maps.
The working-tree change supersedes that: **`P` rises to `0.85`.** The earlier
number is left on the record because a pre-registration that silently re-bases
on new information is not one.

### 4.3 Does gate-`R²` still separate when nothing diverges?

**It stops separating on the axis R11 used, and starts separating on a different
one.**

R11's separation was clean and it was a *divergence* separation: crossing seeds
`0.9719 … 0.9898`, divergent seeds `0.0111 / 0.6268 / 0.0468`; trained mean
`0.699312`, zero-step mean `0.371850`. With no divergent population there is
nothing for that axis to separate, so:

- **trained gate-`R²` spread across the eight seeds `< 0.35`** (R11's spread was
  `0.9898 − 0.0111 = 0.9787`), with **≥ 6 of 8 seeds in `[0.25, 0.65]`** —
  moved little from the `0.291617` zero-step value, because at `β = 1` the
  heads receive only the weak indirect gradient that survives dividing the
  magnitudes out of the row;
- **it separates on `β`, not on NRMSE** — §3.3's `corr(β_final, gate-R²) < −0.6`.
  Any seed whose `β` escapes toward `0` gets real gradient into the heads and
  its gate-`R²` climbs; the trapped ones stay near `0.29`. So gate-`R²` remains
  the discriminating instrument the round found on data, and **what it
  discriminates has changed** — from *diverged / converged* to *escaped the
  softmax corner / did not*. Naming the new referent is L-DIAG's requirement,
  and it is named here before the data rather than after.

---

## 5. VENUS's COUNTER AGAINST VENUS — L-SIGN, equal specificity

R11's filing had **no numeric counter against itself**. §6 of
`V15_VENUS_PREDICTIONS.md` — *"the one prediction VENUS most wants to be wrong
about"* — is a **preference**, not a counter-prediction: it states which error
would be most valuable, files no number, and no observation separates it from
the main filing. **Under v16's L-SIGN, VENUS's own R11 filing would not have
been filed.** That is said here plainly because §6 below blocks the author's R2
for the same defect, and the rule has to cut both ways or it is not a rule.

> **VENUS-COUNTER.** `β` is a single scalar with an unobstructed gradient and a
> monotone one: normalising a row whose magnitudes carry the label is a strict
> loss increase, so `∂L/∂β > 0` consistently and AdamW at `lr = 0.02` drives it
> down at up to `0.02`/step. `β` reaches `≈ 0` by step **≈ 60 of 150** on most
> seeds, and the remaining **90** steps are R11's problem run on an operator
> that represents the target **exactly** (`5.919777e-16`) rather than one whose
> identity was proved on gates the corpus never supplies. The clamp then helps
> rather than hurts, because §2.3's optimum is `ρ* = 0.5000` and the clamp
> removes only the runaway region above `1`, which is where seeds 2, 3 and 7
> died. Under this half the crossing count goes **UP**: **6 of 8**, 80% interval
> on the count **[5, 8]**; seed mean **0.68**, 80% interval **[0.62, 0.76]**;
> **`sd` ≈ 0.04**; the 95% CI clears `floor₁` **from below**; and R1′ **banks
> `+12`**.

**What separates the two halves, and what does not.** `sd` does **not** — both
halves predict a tight unimodal cluster (`0.06` against `0.04`). §3.1 therefore
separates VENUS from the *author's counter* and is silent between VENUS and
VENUS-COUNTER. **The sign of `mean − floor₁` separates them, and `β_final` says
why**: VENUS has `β_final ≥ 0.5` on ≥ 6 of 8, VENUS-COUNTER has `β_final < 0.1`
on ≥ 6 of 8. One column, both questions.

**The split, filed honestly:**

| half | outcome | P |
|---|---|---|
| **VENUS** — unimodal, above the floor, `β` trapped | mean `0.92`, `sd 0.06`, `0/8` cross | **0.60** |
| **VENUS-COUNTER** — unimodal, below the floor, `β` escapes | mean `0.68`, `sd 0.04`, `6/8` cross | **0.25** |
| **the AUTHOR's COUNTER** — bimodality survives | mean ≈ `0.83`, `sd > 0.18`, `1–3/8` cross | **0.15** |

VENUS is betting `0.25` on the outcome it most wants — the `+12` — which is
lower than R11's `0.28` on the same seat, and the reduction is the whole content
of §2.2.

**Confidence, per clause:**

| clause | P |
|---|---|
| 8/8 converge, no non-finite gradient in 150 steps | **0.80** (§8 states the 20%) |
| `0` crossings | 0.55 |
| `≤ 1` crossing | 0.75 |
| `≤ 3` crossings (the author's counter's clause also holds) | 0.93 |
| seed mean in `[0.83, 0.98]` | 0.60 |
| `sd < 0.15` | 0.72 |
| 95% CI does not straddle **and** lies above `floor₁` | 0.58 |
| `0` of 8 seeds above `1.10` | 0.85 |
| `β_final ≥ 0.5` on ≥ 6 of 8 | 0.58 |
| `corr(β_final, NRMSE) > +0.6` | 0.55 |
| **no seed shows `λ̂ > 0` at ANY step** | **0.99** — §4.2, `m ≤ 1 ⇒ log m ≤ 0`; the 1% is a runner that redefines the column |
| `λ̂` reads `−inf` on 8/8 at step 0 | 0.88 |
| `dyn_range_bound` finite on ≥ 4 of 8 at step 0 (never once in R11) | 0.70 |
| the `λ̂` / unit-root column is produced at all at it.7 | **0.85** (was `0.35` against the committed tree) |
| trained gate-`R²` spread `< 0.35` | 0.62 |

---

## 6. D-CALIB-2 APPLIED TO THE CONTRACT — WHAT IS BLOCKED

D-CALIB-2 (`V16_CALIBRATION.md:162-170`), operational form: *"a node that
encounters a bare directional prediction in a contract clause does **not** apply
a discount to it and does not plan against it. It reports the missing counter
and blocks the cell."* The backing KILL is immutable at first datum:
**"Prediction without counter-prediction ⇒ not filed."**

Census of PART IV, by whether the clause carries a `COUNTER` of equal
specificity. `grep -n "COUNTER"` over `V16_*.md` returns four hits, none of
which is a counter for R2 or R-SKY.

| cell | prediction filed | counter filed | ruling |
|---|---|---|---|
| **R1′** (`:195-203`) | 8/8, no straddle, 5/8 banks | **YES**, explicit, equal specificity | filed; D-CALIB-1 applies; **runs** |
| **R2** (`:205-207`) | *"`floor₁ = 0.9354`; `ĥ > 1` for the first time"* | **NONE** | **NOT FILED. BLOCKED.** |
| **R-SKY** (`:209-211`) | *(none — see below)* | **NONE** | **BLOCKED, and for a worse reason** |
| **R3** (`:213-218`) | composed within resolution on both beds | **YES**, explicit | filed; **runs** |
| R7 (`:230-233`) | `≥ 2×` fewer examples | *"**counter: none** — `L_jac` helps only where…"* | **ambiguous, and fails on either reading** |

### 6.1 R2 — blocked twice, and the second blocker is physical

**Blocker 1, D-CALIB-2.** R2 states `ĥ > 1` *"for the first time"* and files no
counter. The author demonstrably knows the form — R1′ and R3 both carry one, in
the same PART IV, three lines apart. The omission is not stylistic. **R2 cannot
run until the author files a counter of equal specificity to R1′'s**, i.e. a
number for the crossing count or the seed mean at `t* = 8`, not a direction.

VENUS notes the obvious repair and rules it insufficient: **a counter supplied
by VENUS does not discharge this.** L-SIGN says *"the contract **author** files
a counter-prediction of equal specificity beside every prediction."* The law
exists to make the author's own sign measurable; a counter written by the node
that is scoring him measures nothing. **The block clears only from the author's
hand.**

**Blocker 2, memory, and it is independent of every contract question.**
`V16_DEVICE_CERT.md` §0 line 4 and §3.4: R2's registered `n = 32768` **does not
fit** — `7.768 GiB` of activations against `6.939 GiB` free and `7.996 GiB`
*total* on the certified device (RTX 4060 Laptop, sm_89), and it does not fit on
CPU either. The last power of two the complex arm affords is **`n = 16384`**.
The cert's own it.10 row marks R2's arm *"does not fit — §3.4"*. The cert's
`[ASSUMED]` there is `ceq/arm_phase.py`; the arm that will actually run is
`ceq/arm_smprime.py`, which materialises the same `[n, S, S]` complex operator
and for which the calibrated sizing model reads **`m/p = 1.842`** — it
**under-predicts**, the one direction a sizing gate must never have. **R2's
memory must be measured, not modelled, and the cert says so.**

Both blockers must clear. Clearing only the second leaves the cell unfiled;
clearing only the first leaves it unrunnable.

### 6.2 R-SKY — no counter, and no prediction either, against a `+2` line

R-SKY's full text is *"the scan skyline read beside R1′/R2; **`Δ_sky` column
mandatory**; **beats-softmax sentences BANNED on BED-M**."* That is a
measurement registration, a mandatory column and a banned sentence. **It
contains no number, no direction and no threshold.** There is nothing to
counter because nothing is predicted.

**And the SCOREBOARD pays it `+2`** (*"skyline + `Δ_sky`"*).

**This is worse than an uncountered prediction and VENUS files it as the
contract-level finding of this node.** An uncountered prediction can at least be
wrong. A scoreboard line that pays for *printing a column* cannot be lost by any
measurement: it is credit for labour, awarded regardless of what the labour
finds. Under L-SIGN's own logic — a calibration column exists so that
predictions can be scored against outcomes — a `+2` with no registered outcome
is `+2` of unfalsifiable scoreboard, and the round's ceiling of `41` includes it.

**The remedy is a choice and the author must make it, not VENUS:**

1. file a numeric prediction for `Δ_sky` **with its counter** — e.g. a number
   for `NRMSE_arm − NRMSE_skyline` at the R1′ cell with an interval — and the
   `+2` becomes scoreable; **or**
2. re-scope the `+2` to *"the column is printed"* and **state in the scoreboard
   that it is a labour credit, not an evidence credit**, so the `41` ceiling is
   read correctly.

**The same defect, unexamined, sits on three further scoreboard lines**: R4
*"architecture table"* `+2`, R5 *"state model passes CK"* `+4`, R6 *"detector at
CRB"* `+2`. R4 files no prediction at all. `9` of the `41`-point ceiling —
**22%** — is currently carried by lines with no registered prediction and
therefore no way to lose. VENUS reports the count and does not rule on it;
SATURN owns the scoreboard.

### 6.3 R7 — the ambiguity, and the R11 defect carried forward verbatim

*"**counter: none** — `L_jac` helps only where `a` and `b` were confounded"*
parses two ways. If *"none"* means **there is no counter**, R7 is blocked under
D-CALIB-2 like R2. If *"none"* **is** the counter (predicting no speedup), it
fails equal specificity: the prediction carries a number (`≥ 2×`) and the
counter carries a mechanism sentence. **Blocked on either reading**, and the
author should disambiguate rather than let a later node choose.

**Separately, R7 inherits a defect VENUS filed in R11 and that was never
repaired.** R7 predicts *"`≥2×` fewer examples to **R1′'s crossing**"* — a ratio
whose denominator is a crossing this filing predicts does not exist.
`V15_VENUS_PREDICTIONS.md` §4 filed exactly this against v15's R6: *"R6 as
written cannot be scored, because under section 1 there is no `ζ = 0` crossing
to be 2× cheaper than."* The clause was **carried into v16 as R7 with the
denominator changed from R1 to R1′ and the defect intact**. That is a
mechanism-level repeat, and it belongs in `MISTAKES.md` as one: *a target
defined as a ratio to an event the same contract has not established*.

---

## 7. SCORING R11 — BOTH HALVES

Adjudicated against `V15_R1.md`'s printed cells and `V15_VENUS_PREDICTIONS.md`
§7 as filed. No re-reading of a floor; NO READING counts as not-crossing, as the
filing itself required.

### 7.1 The forecast half

| # | VENUS filed | measured | verdict |
|---|---|---|---|
| R1 verdict | *does NOT cross*, CI does not clear `0.7071` | CI `[0.617075, 1.041227]` **straddles** ⇒ **NOT CROSSED** | **RIGHT** |
| R1 mean | `0.86`, 80% `[0.74, 0.96]` | **`0.829151`** | **RIGHT**, inside; point off by `0.031` |
| R1 `ĥ` | `[0.16, 0.90]`, point `0.52` | `0.625017` | **RIGHT**, inside |
| PL beats softmax by `> 2.345e-3` (P = 0.80) | | `0.829151` vs `0.951767` | **RIGHT** |
| `log a` probe returns NaN or `R² < 0.01` (P = 0.90) | | `SST = 0.000e+00`, `R²` **undefined**; pooled `3.2466e-04` | **RIGHT**, and it forced the contract's own diagnostic to be replaced (M-18) |
| trained gate-`R²` | `0.55`, 80% `[0.30, 0.75]` | mean **`0.699312`** | **inside, and hollow** — see below |
| **R1 discriminator**: `c ≥ 0.8182 → contract`, `c < 0.8182 → VENUS` | | **seed 3: `c = +1.0000` and NRMSE `1.113403`** | **REFUTED** |

**Score: 5 clear rights, 1 hollow right, 1 refutation.**

**How the discriminator was refuted, precisely.** It was filed as a
biconditional and only one direction survived. `c ≥ 0.8182 ⇒ crossing` is
**false**: seeds 3 and 2 and 7 all had `c` at or above the threshold on the
crossing side of the ledger for seed 3 (`c = 1.0000`, `p = 1.000000`) and did
not cross. `c < 0.8182 ⇒ no crossing` held. **`c` is necessary and not
sufficient**, and what killed seed 3 was `â_max = 49.6613` — a magnitude channel
the discriminator did not mention, on a seed with perfect signs. VENUS predicted
a one-dimensional failure surface for a two-dimensional arm.

**Why the gate-`R²` row is scored hollow rather than right.** `0.699312` is
inside `[0.30, 0.75]`, at its top edge, and the point `0.55` was low by `0.149`.
More to the point: the eight readings were `0.9889, 0.9898, 0.0111, 0.6268,
0.9807, 0.9719, 0.9785, 0.0468`. **No seed read anywhere near `0.55`.** VENUS's
interval covered a mean that no seed produced — **the same defect `V15_R1.md`
§"THE THREE NUMBERS" names in the contract's own reading**, where the seed mean
`0.829151` is *"a number no seed produced."* Being right about a mean while
blind to the bimodality that was the entire finding is not a forecasting
success, and it is scored as one row of credit and one lesson.

### 7.2 The counter half — L-SIGN, applied to VENUS

**There was none.** `V15_VENUS_PREDICTIONS.md` §6 files a *preference* ("the one
prediction VENUS most wants to be wrong about"), not a counter-prediction: no
number, no interval, no discriminating observation. **Under v16's L-SIGN and the
KILL it backs, VENUS's R11 filing was `prediction without counter-prediction ⇒
not filed`.** L-SIGN post-dates it, so this is not a retroactive strike — but
the sign is `+` and it goes in the column: filing only the half one believes is
the same one-sidedness D-CALIB exists to measure, and VENUS did it while
auditing the author for it.

### 7.3 What changed as a result — four things, each traceable to a row above

1. **Every discriminator is two-sided and has a stated verdict in every region,
   including the region where neither filing is right.** R11's `c ≥ 0.8182` had
   no clause for *"`c` high and still fails"*; seed 3 lived exactly there. §3.1
   states the outcome for `sd < 0.10`, for `sd > 0.18`, **and** for
   `0.10 ≤ sd ≤ 0.18` where both filings lose.
2. **The SHAPE of the seed distribution is predicted before the mean is.** §1
   leads with `sd` and unimodality; the mean is secondary. R11's mean was right
   and told nobody about the bimodality — the finding of the whole cell.
3. **A numeric counter against VENUS at equal specificity** (§5), with its own
   probability and its own discriminating column, which R11 had only as prose.
4. **Anything already on disk is declared a re-read, not a prediction** (§4.1,
   the step-0 gate-`R²`). R11 did not face this; R1′ does, and scoring a lookup
   as a forecast would inflate the very column this round opened.

---

## 8. LIMITS

Collected once.

- **The `β` mechanism is read from source, not measured under training.**
  `ceq/arm_smprime.py:508` and `V16_ARM_SMPRIME.md` verdict (h) establish that
  the arm initialises at `β = 1` and that the label bind fails there at
  `1.335288`. **Neither establishes that `β` fails to travel** — §5 is the half
  where it does, and it is filed at `0.25`. `V16_ARM_SMPRIME.md` §6.2 says the
  same in the node's own words: *"Whether R1′ trains an arm that drifts to
  `β = 1` is a question for it.7 and this node does not answer it."* This filing
  answers it in the other direction and may be wrong.
- **The `c`-model's level is optimistic by ≈ `0.24` NRMSE and VENUS's own R11
  data proves it.** §2.3's model predicts `0.395285` at `c = 1`; R11's five
  `c = 1.0000` seeds read `0.634002 … 0.662021`. The model's *crossing
  threshold* (`c ≥ 0.8182`) and its *`ρ`-optimality* argument survive that — the
  optimum's location is a derivative, not a level — but **no level claim in this
  filing rests on it.** §1's numbers come from the measured softmax cells
  (`mean 0.951767, sd 0.011824`), not from the model.
- **The convergence prediction's 20% has a named mechanism.** `β` is an
  `nn.Parameter` with **no clamp**, unlike `m`. If it goes negative while some
  `Z_i` is small, `Z_i^β` overflows. `Z_i ≥ exp(q_i·k_i) > 0` is bounded away
  from zero by the empty-product diagonal (`R_ii = 1`), so this is not the
  R11 mode — but the 8/8 40-step probe that licenses "converges" ran at
  `lr = 1e-3` on `torch.randn`, and R1′ runs at `lr = 0.02` — **20× the step
  size** — on the real corpus for `150` steps. **The probe does not cover the
  cell.** A non-finite failure, if it comes, shows as `|β|` growing with `â_max`
  pinned at `1`, which is a different signature from R11's and is worth printing
  `β` per step to catch.
- **The cell's `steps` and `lr` are assumed, not registered.** `150` and `0.02`
  are `V15_R1.md`'s and every prior `t* = 2, n = 2048` cell's. PART IV fixes
  `t*`, `n`, `N` and the seeds and fixes neither. **The ≈50-step `β` transit in
  §2.2 scales inversely with `lr`**, so a cell run at `lr = 0.002` moves this
  filing's probability mass sharply toward VENUS's main half and one at
  `lr = 0.2` toward VENUS-COUNTER. If MERCURY changes either, §5's split must be
  re-read before it.8 and this document says so before the choice is made.
- **`0/8` is a point estimate on a count, not a claim that crossing is
  impossible.** The arm represents BED-M's label at `5.919777e-16` at the right
  corner; nothing here says it cannot be trained there, only that it is not
  reachable from `β = 1` in the cell's budget while two coupled heads are also
  being found.
- **D-CALIB-3 was obeyed.** No number in §1 is R11's number times a factor. The
  `7/8` optimism at `p = 0.0352`, Wilson 95% `[0.5291, 0.9776]`, licensed one
  thing and one thing only: **ranking the author's counter ahead of his
  prediction**, which §1 does. Every level in this filing comes from a measured
  cell (`results/v15_r1.jsonl`), from shipped source, or from §2.3's closed form.
- **`corr` on 8 points is a weak instrument.** §3.3's `> +0.6` and `< −0.6`
  thresholds are chosen to be visible at `N = 8`, not to be significant at it;
  `r = 0.6` at `n = 8` is `p ≈ 0.12` two-sided. They are filed as *readings that
  separate two mechanisms*, not as tests, and §3.1's `sd` — a 21× ratio between
  two measured regimes — is the clause that carries the weight.
- **§4.2's staged runner is UNCOMMITTED and may change before it.7.** Its `λ̂`
  definition (`lg = torch.log(m)`) is what makes `λ̂ > 0` unreachable; a runner
  that instead read the growth rate off the **operator row** rather than off the
  gate magnitudes would restore a reachable positive exponent and §4.2's
  impossibility argument would not apply. The argument is therefore against
  *the column as currently staged*, and that is stated rather than generalised.
  The `≤ 0` bound on `log m` itself is unconditional and survives either choice.
- **`house-events.jsonl` was not consulted and was not grepped.**
