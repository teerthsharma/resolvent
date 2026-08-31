# V15 VENUS — competing numeric predictions, filed before the data exists

**Filed:** `2026-08-31T11:01:32Z` (UTC)
**git HEAD at filing:** `7e749d9a7dad7ecd8c9ec624a88ba88b5dfe0fba`
**Scheduled at it.8 by `CEQ_V15_CONTRACT.md:254`; filed at it.3.**

Filing early is not a courtesy. At it.8 the Lean gate is green and ARM PL is
built, so a prediction filed there is a prediction made after seeing the arm. At
it.3 none of that exists, and the pre-registration is stronger by exactly that
much.

## STATE OF THE TARGET AT FILING TIME — checked, not assumed

| thing | check | result |
|---|---|---|
| ARM PL | `grep -rln "ARM PL\|arm_pl\|path_product"` over the tree | one hit, `CEQ_V15_CONTRACT.md`. **The arm does not exist.** |
| R1 / R2 cells | `results/` scanned for any `pl` arm row | **none** |
| BED-K | `grep -rln "BED-K\|bed_k"` | three hits, all contract/audit/ledger prose. **The bed does not exist.** |
| `L_jac` / interventional channel | `grep -rln "L_jac\|l_jac"` | same three prose files. **Not built.** |
| Lean `V15.lean` | `lean/CEQ/V15.lean` present, train-gate not yet called | items #1,#2,#5,#6,#7 not yet reported green |

No measurement this document predicts against has been taken. If any reader
finds one dated before the timestamp above, this filing is void and must be
reported as void rather than scored.

Every number below is either (a) closed-form from the corpus builder, (b) run in
this session against the **corpus generator only** — `make_equilibrium_batch`,
which is not an arm and not a training run — or (c) taken from
`V15_CONTRACT_ARITHMETIC_AUDIT.md`, whose floors and power curve are used as
confirmed and are not re-derived.

---

## 0. THE FACT THE CONTRACT'S PART IV DOES NOT STATE, AND WHICH DECIDES ALL FOUR

BED-M's coefficients are **Rademacher**, not Gaussian.
`scale/negation_scope.py:428` draws `a` in `{-1,+1}` and `:429` zeroes it at and
before `head = s - 1 - t*`. So on the chain corpus:

```
|a_i| = 1   for i in [head+1, s-1]     ->  log|a_i| = 0
 a_i  = 0   for i in [0, head]         ->  log|a_i| = -inf
```

**`log a` is a two-valued function on this bed and it is constant on the live
band.** Everything the label carries is in `sign(a_i)`; the magnitude channel
`g` carries only the band mask.

Second fact, from `scale/m3_capability.py:110-142`: the harness arm takes
`q, k = wq(x), wk(x)` on the raw `[n, s, d_model]` tensor. **There is no
positional encoding anywhere in the arm or in `x`.** `make_batch` fills `x` with
`randn * 0.1` and overwrites three channels with content. A gate head therefore
sees content only.

Put together, these fix the learning problem ARM PL actually faces at R1:

- the true `g_i = log|a_i|` is an **even** function of the drive channel — `0` at
  `a = +1`, `0` at `a = -1`, `-inf` at `a = 0`;
- `g = -softplus(W x)` with `W` linear is **monotone** in `W x`, and `W x` is
  linear in `a_i`;
- a monotone function of an odd feature cannot be an even function of it, and
  there is no positional feature to route around it.

**Consequence, load-bearing for R1 and for the probe:** as parametrised in
`CEQ_V15_CONTRACT.md` PART I, the band mask is not representable, let alone
learnable. The escape is a *signed* gate — `a_hat_k = tanh(kappa . w . x_k)`,
`W_ij = prod_{j<k<=i} a_hat_k` — which supplies magnitude and sign in one
differentiable object (`tanh(0) = 0` kills the dead band, `|tanh(large)| -> 1`
keeps the live one). Whether ARM PL is built that way or built literally to the
contract's `-softplus` + hard-parity spec is the single largest free variable in
R1, and it is a *build* choice made at it.6-7, before any data.

### The one dial that sets every threshold below

Let `c = E[s_hat_k . sign(a_k)]` be the trained gate's per-hop **signed
recovery** on live positions (`c = 2p - 1` for per-position sign accuracy `p`).
Model the arm as an optimal-scale readout of `W_ij = rho^{i-j} . (recovered
signs)`, with the dead band leaking at weight `rho^m` and random sign. Then

```
NRMSE(t*, c) = sqrt( 1 - Cov^2 / (t* . Var) ),
Cov = sum_{m=1..t*} rho^m c^m ,   Var = sum_{m=1..s-1} rho^{2m},
```

minimised over `rho`. Run in this session, `s = 64`:

| `c` | best NRMSE, `t*=2` | crosses `0.7071`? | best NRMSE, `t*=8` | crosses `0.9354`? |
|---|---|---|---|---|
| 0.50 | 0.9211 | no | 0.9789 | no |
| 0.60 | 0.8761 | no | 0.9642 | no |
| 0.70 | 0.8136 | no | 0.9384 | no (by 0.0030) |
| 0.75 | 0.7738 | no | 0.9177 | **yes** |
| 0.80 | 0.7265 | no | 0.8878 | **yes** |
| 0.85 | 0.6699 | **yes** | 0.8431 | **yes** |
| 1.00 | 0.3953 | **yes** | 0.4287 | **yes** |

**Crossing thresholds, bisected:** `t*=2` needs `c >= 0.8182` (`p >= 0.9091`);
`t*=8` needs `c >= 0.7084` (`p >= 0.8542`).

**R2's bar is easier than R1's, in the only currency the arm pays in.** Crossing
`floor_1` means explaining one unit of label variance. At `t*=2` that is 1 of 2
(50%); at `t*=8` it is 1 of 8 (12.5%), and eight decaying hops supply partial
credit that two cannot. The already-measured softmax cells say the same thing
from the other side: `h-hat = 0.186` at `t*=2, n=2048` against `h-hat = 0.389` at
`t*=8, n=32768` (`workdonenew.md` section 4) — softmax is **twice as close** to
the bar at R2's cell as at R1's.

**Also recorded:** at every `t*`, `h-hat > 1` and `NRMSE < floor_1` are the
**same event**, not two. `h-hat = t*(1 - NRMSE^2)`, so `h-hat = 1` exactly at
`NRMSE = sqrt((t*-1)/t*)` — verified to ten decimals at `t*=2` and `t*=8`. R2's
two clauses are one clause and cannot disagree.

---

## 1. R1 — BED-M, `t* = 2`, `n = 2048`

**Contract:** `PL < floor_1 = 0.7071` with CI — the first floor crossing in the
campaign.

> **VENUS: ARM PL does NOT cross. The N=8 seed mean lands at `0.86`, 80%
> interval `[0.74, 0.96]`, and the seed CI does not clear `0.7071`. It is
> nonetheless a large, resolved improvement on softmax's `0.952349` — the
> contrast is negative and larger than the `2.345e-3` thread floor, which no
> prior arm has managed at this cell.**

Equivalently in the contract's own units: `h-hat` in `[0.16, 0.90]`, point
`0.52`, first `h-hat` above `0.389` in the campaign and still below `1`.

**Mechanism.** Three terms, all quantitative.

1. **The gate must be found from a scalar loss, through a 64-term product.** At
   `t*=2` the label is `z = a_63 a_62 b_61 + a_63 b_62` — both terms require
   `a_63`, so *no part of it is legible without the gate*. The gradient reaching
   `a_62` passes through the running product; at `nn.Linear` default init
   (`U(-1/4, 1/4)`, std `0.144`) the gate pre-activation carries signal
   `w_2 ~ 0.144` against dead-band leakage `sigma_eps ~ 0.154` from the other 15
   channels. **At initialisation the leak exceeds the signal**, and the 2-hop
   term's gradient is down by the product magnitude relative to the 1-hop term's.
2. **The budget is 150 full-batch Adam steps at `lr = 0.02`**
   (`scale/m3_capability.py:88`, and every prior `t*=2, n=2048` cell). The gate
   must saturate one channel and null fifteen inside that budget. `E2_STEPS =
   600` is already recorded in the same file as the smallest budget that passes
   for a *simpler* head.
3. **The contract's own `-softplus(W x)` cannot represent the band mask**
   (section 0). If ARM PL is built literally to spec, `g` collapses toward a
   constant, the operator degenerates to a uniform decay mask `rho^{i-j}`, and
   the arm's entire content is the parity mask — which is precisely the `c`-model
   above with `rho` free. That model's *best case at perfect signs* is `0.3953`,
   so a spec-literal build is not hopeless; it is `c`-limited.

Prior weight assigned honestly: the four dead hop-2 constructions get **near-zero
evidential weight against the operator** — they composed weights across positions
and this one composes values along paths, and `workdonenew.md` section 6 states
in the repo's own words that this corpus "asks for what a gated linear scan
computes natively", which ARM PL is. They get **substantial weight against the
optimisation**, because all four share one cause that survives the algebra fix:
the arm is *built* to satisfy a bind and then *trained* against a different
objective, and nobody has yet measured the gap. R1 is the fifth instance of that
gap, not the fifth instance of the wrong algebra.

**What must be true for the contract to win instead.** ARM PL's gate must reach
`c >= 0.8182` — per-position sign accuracy `p >= 90.9%` on live positions — from
2048 examples and 150 steps. That happens if and only if the arm is built with a
**differentiable signed gate** (`tanh`/cumprod, or a straight-through estimator
with a real surrogate gradient) rather than a hard `sign`-then-XOR parity. A
hard-XOR parity has exactly zero gradient into the sign weights, which pins them
at init (`c ~ 0.5`, giving NRMSE `0.921` by the table) and makes the contract's
prediction unreachable by construction.

**The discriminator, in one measurable number:** `c` (or `p`) read off the
trained arm. `c >= 0.8182` -> contract. `c < 0.8182` -> VENUS. Nothing else needs
to be adjudicated, and the sign of the disagreement is visible at the first seed.

**Confidence:** `P(no crossing) = 0.72`. `P(seed mean in [0.74, 0.96]) = 0.55`.
`P(PL beats softmax by more than the 2.345e-3 thread floor) = 0.80`.

---

## 2. R2 — BED-M, `t* = 8`, `n = 32768`

**Contract:** `PL < floor_1 = 0.9354`, `h-hat > 1` for the first time.

> **VENUS: R2 crosses — and it crosses BEFORE R1 does, which is the opposite of
> the contract's ordering. Seed mean `0.90`, 80% interval `[0.84, 0.95]`,
> `h-hat` in `[0.78, 2.36]` with point `2.5 x` the campaign's current maximum of
> `0.389`. The claim that carries the weight is the ORDERING: if exactly one of
> R1/R2 crosses, it is R2.**

**Mechanism.** Two independent reasons, both quantitative.

1. **The bar is lower in explained-variance units.** `c >= 0.7084` at `t*=8`
   against `c >= 0.8182` at `t*=2` (section 0, bisected). Eight decaying hops
   pay partial credit toward the one unit of variance that `floor_1` demands;
   two hops do not.
2. **The data is 16x larger** — 32768 examples against 2048, at the same 150
   steps. The gate is a 16-parameter linear map followed by one saturating
   nonlinearity; its sample complexity is trivial and `n` is exactly the axis
   that moves `c`.

Against it, and stated because it is real: the 8-hop path product needs eight
correct signs in a row, so the deepest term's contribution scales as `c^8` —
`0.27` at `c = 0.85`. That is why the predicted NRMSE is near the bar rather than
far below it, and why the interval's upper end sits above `floor_1`.

**What must be true for the contract to win instead.** The contract wins on R2
whenever VENUS does; the disagreement is *only* the ordering sentence "the first
floor crossing in the campaign". The contract loses that sentence if R2's CI
clears `0.9354` while R1's does not clear `0.7071`. Both cells are scheduled at
it.9-10, so the ordering is decidable in one iteration and needs no extra run.

**Confidence:** `P(R2 crosses) = 0.58`. `P(R2 crosses AND R1 does not) = 0.42`.
`P(R2 is the crossing | exactly one of R1/R2 crosses) = 0.78`. The last is the
number to score VENUS on.

---

## 3. R3 — BED-K, the delay bed

**Contract:** scan-only `>= 0.95` (theorem-backed), attention / fractional head
near 0, COMPOSED arm matches the native primitive.

Read in NRMSE, where `>= 0.95` is failure and `near 0` is success — the only
reading consistent with PART III's `[RUN: best first-order recurrence 0.990]`.

> **VENUS: the fractional head FAILS the delay bed. NRMSE in `[0.96, 1.00]`,
> point `0.99` — statistically indistinguishable from the scan it is supposed to
> beat, not "near 0". Scan-only `[0.97, 1.00]` (agreeing with the contract).
> Attention `[0.02, 0.20]` (agreeing with the contract) IF and ONLY IF BED-K
> supplies a positional feature; see the registered hazard below.**

**Mechanism, by construction rather than by measurement.** The Grünwald-Letnikov
weights under the audit's own verified recurrence `w_0 = 1`,
`w_k = w_{k-1}(alpha + k - 1)/k` are strictly positive and strictly decreasing
for `alpha in (0, 1)`. Run in this session:

```
alpha=0.10 : [1, 0.1000, 0.0550, 0.0385, 0.0298, ...]   argmax k = 0
alpha=0.25 : [1, 0.2500, 0.1562, 0.1172, 0.0952, ...]   argmax k = 0
alpha=0.45 : [1, 0.4500, 0.3262, 0.2664, 0.2298, ...]   argmax k = 0
alpha=0.90 : [1, 0.9000, 0.8550, 0.8265, 0.8058, ...]   argmax k = 0
```

**The kernel's maximum is at lag 0 for every admissible `alpha`.** A pure delay
`d >= 1` is a kernel whose *only* mass is at lag `d`. A one-scalar power-law
family cannot place more mass at lag `d` than at lag `0`, so the fractional head
is blind to a pure delay for the same structural reason the first-order
recurrence is — and the contract groups it with attention, which is the wrong
side of the line. This costs the contract nothing on the *power-law* half of
BED-K, where the head is native and `alpha-hat` recovery should work; it costs
it the delay half.

**Registered hazard, filed now so it cannot be discovered after the fact.** A
pure delay is a purely *positional* operation. `scale/m3_capability.py` gives its
arms **no positional feature** — `wq(x)`, `wk(x)` on content only. If BED-K is
generated in the m3 encoding without adding positions, softmax attention cannot
locate "d back" either, and the contract's "attention near 0" fails for a reason
that has nothing to do with kernels. **Saturn should register whether BED-K
carries a positional channel BEFORE the bed runs**, because after the fact the
two explanations are not separable. VENUS's fractional-head number is stated to
be independent of that choice; the attention number is not.

**What must be true for the contract to win instead.** The fractional head must
be given more than one scalar — a learned lag offset, a mixture of `alpha`s, or a
composition with a shift — in which case it is no longer the X34 head the
contract specifies and the win belongs to the composition, not to the head.

**Confidence:** `P(fractional head NRMSE >= 0.90 on the pure-delay bed) = 0.88`.
`P(the delay bed is built with a positional channel) = 0.60`.

---

## 4. R6 — interventional ablation, `zeta > 0` vs `zeta = 0`

**Contract:** at least 2x fewer examples to R1's crossing with `L_jac`.

> **VENUS: R6 as written cannot be scored, because under section 1 there is no
> `zeta = 0` crossing to be 2x cheaper than. Restated against a defined target:
> the `zeta > 0` arm CROSSES `floor_1` at `t*=2, n=2048` — the very cell where
> `zeta = 0` does not. Seed mean `0.46`, 80% interval `[0.30, 0.62]`. The
> sample-efficiency ratio is therefore not `2x`; it is unbounded on the grid, and
> R6 rather than R1 is the campaign's first floor crossing.**

**Mechanism.** The oracle Jacobian on this bed is not a hint, it is the answer.
`dz_{s-1}/db_j = prod_{k>j} a_k` — which is exactly `W_{s-1,j}`, the operator row
ARM PL is trying to learn. `L_jac` therefore converts the problem from "recover a
64-term product from a scalar MSE" into "regress the operator row on its
supervised target", which is the oracle-gate bind (`4.0e-15`) turned into a
training signal. That is a change of problem class, not a constant-factor speedup,
and a `2x` prediction is the wrong order of magnitude in the contract's own
favour. `L_jac` also fixes the exact failure section 1 identifies: it supervises
`sign(a_k)` directly, so `c` is no longer gated by whether a scalar loss can push
gradient through a saturating product.

**What would have to be true for the contract's `2x` to be the right number.**
`zeta = 0` must itself cross at `t*=2` (i.e. R1 must go the contract's way), and
`L_jac` must then buy only a factor of two on top. If R1 crosses, VENUS loses R1
and the `2x` becomes scoreable; if R1 does not cross, the `2x` is undefined and
this restatement is the only version of R6 that has a verdict.

**Confidence:** `P(zeta > 0 crosses floor_1 at t*=2, n=2048) = 0.65`.
`P(ratio > 2x, however the crossing target is defined) = 0.80`.
`P(R6 as literally written is scoreable) = 0.28`.

---

## 5. THE LINEAR PROBE — the number nobody has filed

R1's kill condition is *"bind passes, floor not crossed => `g` unlearnable at
budget — diagnose by linear probe on `log a` (should be near-exact)"*
(`CEQ_V15_CONTRACT.md:209`). This is the measurement that separates "the
architecture is wrong" from "the optimizer did not find `g` at n=2048", and it is
the measurement most likely to be misread.

**VENUS's finding: the specified probe is degenerate on BED-M, and its R-squared
is the same under both hypotheses. It cannot diagnose anything.**

Measured this session against the corpus generator only (`t*=2, n=2048, s=64,
seed=0`; no arm, no training):

| probe | rows | SST of target | R-squared |
|---|---|---|---|
| `x -> log\|a\|`, LIVE band only | 4,096 | **0.000e+00** | **undefined (NaN)** |
| `x -> 1{\|a\|>0}`, all positions | 131,072 | 3.968e+03 | **3.2466e-04** |
| `x -> log\|a\|` clamped at `-30` | 131,072 | — | **3.2466e-04** |
| `x -> log\|a\|` clamped at `-100` | 131,072 | — | **3.2466e-04** |
| `x -> sign(a)`, LIVE band only | 4,096 | — | **1.000000** |
| `x, x^2 -> 1{\|a\|>0}`, all positions | 131,072 | — | **1.000000** |

Read them in order. On the live band `log|a| == 0` identically, so the target has
**exactly zero variance** and `1 - SS_res/SS_tot` is `0/0`. Pooled over all
positions the target is an indicator of the band, and the population-optimal
*linear* predictor of it is the constant: every channel's covariance with `1{a
!= 0}` is zero by the Rademacher symmetry, so `R^2 = 0` in population and
`3.2e-04` in this sample against a `d/N = 1.22e-04` null. The clamp value does
not matter, which is the signature of a target that carries no linear signal
rather than one that is merely badly scaled. The last two rows prove the
obstruction is **evenness and not information**: the sign is recoverable at
`R^2 = 1.000000` exactly, and one squared feature recovers the band at
`R^2 = 1.000000` exactly.

### The two hypotheses, in numbers

| quantity | contract's number | VENUS's number |
|---|---|---|
| `R^2` of the contract's probe, `log a`, live band | "near-exact", `>= 0.95` | **undefined**, `SST = 0` |
| `R^2` of the contract's probe, `log a`, pooled/clamped | "near-exact", `>= 0.95` | **`0.0003 +- 0.0003`** |
| `R^2` of a probe on `sign(a)` from raw `x` | (not filed) | **`1.000 +- 0.000`** — under BOTH hypotheses; non-discriminating for the opposite reason |
| **`R^2(a_hat, a)` of the TRAINED ARM's own gate**, live band, `n=2048` | `>= 0.95` | **`0.55`, 80% interval `[0.30, 0.75]`** |
| same, `n=32768` (`t*=8` cell) | `>= 0.95` | **`0.72`, interval `[0.50, 0.90]`** |
| same, with `zeta > 0` (`L_jac`), `n=2048` | `>= 0.95` | **`0.94`, interval `[0.85, 0.99]`** |

**The probe that discriminates is the fourth row and only the fourth row.** It is
a probe on the *arm's recovered gate*, not on the corpus. Under the contract it
reads `>= 0.95` and the floor is crossed; under VENUS it reads about `0.55` and
the floor is not, and the two are the same number as `c` from section 0 up to
`R^2 = c^2` — so it is not an extra measurement, it is the one already required.

**Amendment VENUS asks Saturn to register, before it.9:** replace
"linear probe on `log a`" with "linear probe recovering `sign(a_i)` from the
arm's gate output on live positions, reported as per-position accuracy `p` with
its `n`". As written, the contract's diagnostic returns `NaN` or `0.0003`
whatever the arm does, and a `0.0003` read as "`g` is unlearnable" would blame
the architecture for a property of the corpus. That is the repository's own
**V-10** class — a gate whose value is fixed by construction rather than by the
thing it measures.

**Confidence:** `P(the contract's log-a probe returns NaN or R^2 < 0.01) = 0.90`.

---

## 6. THE ONE PREDICTION VENUS MOST WANTS TO BE WRONG ABOUT

**R1. Specifically the no-crossing clause in section 1.**

Being wrong there is the best available outcome for the project, and by a wide
margin over being wrong anywhere else.

- It is the only line on the SCOREBOARD worth `+12`, and it is the first floor
  crossing in nine cells and four dead constructions.
- It would mean the gate is learnable **end to end from a scalar loss**, with no
  intervention channel, no oracle gates and no positional feature. That is a much
  stronger sentence than R6's crossing, which only shows the operator is right
  when the operator is *supervised*. A crossing at R6 and not at R1 licenses
  "the architecture can express the path product and needs interventional data to
  find it"; a crossing at R1 licenses "the architecture finds it", and only the
  second one is an architecture claim rather than a training-data claim.
- It would retire section 0's obstruction argument by exhibiting the build that
  routes around it, which is worth more to `MISTAKES.md` than the obstruction is.

Being wrong on R2, R3 or the probe is worth much less: R2 wrong means the
crossing happens where the contract said it would, R3 wrong means one head is
better than its own weights suggest, and the probe finding is a defect report
that helps most when it is *right*.

VENUS is betting at `0.72` against the outcome it most wants, which is the
position the seat exists to occupy.

---

## 7. SCORING SHEET — fill in after it.10, it.17, it.23

| # | VENUS | contract | discriminating observation | VENUS bet |
|---|---|---|---|---|
| R1 | no crossing; mean `0.86`, `[0.74, 0.96]` | `< 0.7071` | trained gate `c` vs `0.8182` | 0.72 |
| R2 | crosses, `0.90`, `[0.84, 0.95]`; **crosses before R1** | crosses, second | which of the two cells clears its floor | 0.78 (ordering) |
| R3 | fractional head `[0.96, 1.00]` on the pure-delay bed | "near 0" | GL `argmax_k w_k = 0` for all admissible `alpha` | 0.88 |
| R6 | `zeta>0` crosses at `t*=2, n=2048`, `0.46`, `[0.30, 0.62]`; ratio unbounded, not `2x` | `2x` fewer examples | whether `zeta=0` ever crosses at all | 0.65 |
| probe | `log a` probe `NaN` / `0.0003`; gate probe `R^2 = 0.55` | both `>= 0.95` | `SST` of the probe target | 0.90 |

Adjudicate on the printed cells only. NO READING counts as not-crossing, as it
did for the it.8 filing. No re-reading of a floor, no post-hoc change to which
probe was meant.
