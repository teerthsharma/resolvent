# Retirement card: the per-coordinate corner rule

**Verdict: RETIRED.** The rule `beta_d = 1 - alpha_d` is withdrawn as a method for
assigning an attention corner to a latent coordinate. One card, not a chapter:
the operator family it was built on, the containment of softmax and linear
attention at two settings of one switch, is untouched and is not part of this
retirement.

Produced at commit `391a2d0` on `WIN-16QAL06O9GB` (python 3.11.9, torch
2.14.0+cpu), except the Kaggle rows, which carry their own provenance below.
Every number here names the command or file that produced it.

---

## What the rule claimed

Carrying units through `W = e^l / Z^beta`, a read carries `N^(1-beta)`: a mean
at `beta = 1`, a total at `beta = 0`. So if coordinate `d`'s representation
scales with the token count as `N^alpha_d`, the corner that reads it without a
residual count dependence is `beta_d = 1 - alpha_d`. The rule is a derivation,
not a fit, and that is what made it worth testing rather than tuning.

The claim it licensed: a model whose corner is assigned per coordinate from a
measured exponent generalises to longer contexts where a single-corner model
does not.

---

## Why it is retired

### 1. On the encoder, the exponent is zero by construction

`ceqjepa/pi_jepa.py:390-395` builds the encoder as
`nn.Sequential(Linear, GELU, Linear, GELU, Linear)` applied over the last axis,
with no reduction over the sequence axis. Permuting all fifteen context
positions leaves the last-position representation **bitwise identical, worst
change 0**. Its docstring one line above reads `x_{<=t} -> s_t`; the code is
`x_t -> s_t`.

A representation that does not depend on the count has `alpha = 0` for every
coordinate, at any initialisation, trained or not. Applied honestly to this
encoder the rule returns `beta = 1` for all nine coordinates and refuses none.
The mask actually in use was fitted on nine hand-built token statistics in
`ceqjepa/pi_assign.py` and reached the encoder's nine coordinates **by index
equality alone** — coordinate 3 there is `count_above`, coordinate 3 here is an
arbitrary row of a random `Linear(32, 9)`. Measured on the encoder's own
coordinates, 3 of the 7 assigned are off their own alpha by more than
`ALPHA_TOL = 0.10`.

Producer: `tests/curvature/test_pi_jepa_assignment_work.py`, 6 failed 3 passed
in 30.26 s.

### 2. On a null that isolates the variable, the assignment is below median

The first null shuffled the whole nine-vector with refusals travelling along.
Refused coordinates are dropped from the read, so **12 of 12 rows also
re-selected which 7 of 9 coordinates the read saw**. That null varies two things
and names one; under `L-NULL` it is void and the claim it licensed was unbound.

The replacement pins the refused coordinates at `{6, 7}` and permutes only which
of the seven assigned positions carry the softmax corner. The space is
`C(7, 4) = 35` and is **enumerated whole**, so there is no sampling fraction and
the null carries no seed.

| statistic | value |
|---|---|
| assigned arrangement | 0.6881 |
| best of 35 | 0.5285 |
| median of 35 | 0.6767 |
| worst of 35 | 0.8280 |
| rank of assigned | **21 of 35 (0.600)** |
| arrangements refused | 0 |

The assignment is worse than the median arrangement of its own corners, on a
null that varies only what the rule decides.

Producer: `tests/curvature/test_clean_permutation_null.py`, 1 failed 2 passed in
23.43 s.

### 3. On the value axis, the rule has no solution

The contraction `num @ v` over the key axis is the only place in the module
where a token count varies, so it is the only remaining axis on which the rule
could have a referent. Measuring the exponent there gives:

| `beta_at` | measured `alpha` | rule's output `1 - alpha` |
|---|---|---|
| 0.00 | +1.1934 | −0.1934 |
| 0.25 | +0.9427 | +0.0573 |
| 0.50 | +0.6920 | +0.3080 |
| 0.75 | +0.4413 | +0.5587 |
| 1.00 | +0.1906 | +0.8094 |

`d(alpha)/d(beta_at) = -1.0027`. Substituting `alpha(beta) = a0 - beta` into
`beta = 1 - alpha(beta)` gives `0 = 1 - a0`, and `a0 = 1.1934`. **The solution
set is empty.** This is not a degenerate fixed point; there is none. Every beta
the rule is evaluated at instructs the caller to use a different one, and its
answer spans −0.1934 to +0.8094 according only to where the measurement was
taken. The first of those is not a corner.

The axis also names one class and not the other. A planted extensive coordinate
is recovered at `alpha` between 0.9863 and 0.9988. A planted intensive
coordinate is **refused on all nine** as `NOT-A-POWER-LAW`: its read RMS still
grows, 4.246379e-03 at length 16 to 1.841639e-02 at length 128, because the
attention weights are positive and uncentred, so a centred value stream summed
against them grows anyway. A measurement that can only come out one way is not a
measurement.

Producer: `tests/curvature/test_value_axis_alpha.py`, 3 failed 3 passed in
3.64 s.

### 4. The mechanism survives adversarial control

The value-axis finding was attacked rather than defended.

| control | slope |
|---|---|
| shipped, reproducing the finding | −1.0025 |
| context positions permuted | −1.0019 |
| `Z` count-dependence removed | **−0.0254** |

The count dependence is not positional. Removing `Z = mod.sum(-1)`'s growth
removes the slope, so `Z` carrying the count exactly as the numerator does is
the whole mechanism, which makes the explanation mechanistic rather than fitted.
At `beta = 1` the freeze moves the worst exponent by 0.9771 while the
permutation moves it by 0.0000, and the freeze is inert at `beta = 0` where
`Z**0` is one by identity.

Producer: `tests/curvature/test_value_axis_mars.py`, 4 passed in 3.72 s.

---

## What the Kaggle run added, and what it did not

One T4 kernel, `Tesla T4`, torch 2.10.0+cu128, python 3.12.13, cloning the
public repo at `391a2d0`; 115,628 games and 8,089,783 plies from the Lichess
2013-01 dump, 104,132 train games against 11,496 holdout, split by game with
every one of the 8,089,783 position rows checked for cross-split leakage.

**Training helps on real data, and the synthetic bed said the opposite.** Against
a uniform two-head cross-entropy of `2*ln(64) = 8.3178`:

| arm | `probe_loss_last` | nats bought | `encoder_delta` |
|---|---|---|---|
| untrained | 8.0782 | 0.2396 | 0.0 |
| frozen random | 8.0830 | 0.2348 | 0.0 |
| trained | **7.7180** | **0.5998** | 1.0670 |

That retires, on real data, the earlier synthetic-bed finding that a frozen
random encoder beats a trained one. The bed's target moved with the encoder;
this one does not.

**It does not rescue the rule, and it kills the length claim from the other
side.** Top-1 accuracy on the square the human actually moved from:

| arm | L=16 (n=4096) | L=64 (n=4096) | L=128 (n=486) |
|---|---|---|---|
| trained | 0.0823 | 0.0256 | 0.0123 |
| frozen random | 0.0437 | 0.0325 | 0.0267 |
| untrained | 0.0403 | 0.0154 | 0.0144 |
| majority square | 0.0942 | 0.0398 | 0.0247 |

Every arm loses to the majority-square control at every length. The trained arm
degrades **fastest**, 0.0823 to 0.0123 across a factor of eight in context, where
the frozen arm falls only from 0.0437 to 0.0267; at L=128 the trained model is
below both the frozen and the untrained one. The predicted direction was that
the assigned model would hold up where a single corner would not. The measured
direction is that training helps at the training length and hurts at eight times
it.

The collapse detector never fired: `erank_min` 2.0259 against a floor of 1.5,
preflight `erank` 7.9961 and `std_min` 6.457e-03 against a floor of 1e-3.

---

## What this card does not claim

**This is not a result about the endgame target.** The kernel filters no material
and loads no distance-to-mate label, verified by grep at `391a2d0`; it keeps
every game of at least 20 plies. Nothing above is a statement about K+Q-vs-K.
The 13,388 exactly-labelled human decisions extracted during setup were never
used by the run.

**The 1x row is heavily contaminated and is reported, not corrected.** 65.66% of
holdout first-16-ply positions also occur in train (120,775 of 183,936), which
is opening-book recurrence and rises with corpus size — it was 42.73% at 7,498
games.

**Two claims cited as the reason for retirement were themselves unbound when
first made, and are named here rather than dropped.** The first null was void
under `L-NULL`, so the "carries no information" claim rested on nothing until
the enumerated null above replaced it. And `ceqjepa/t_length.py`'s 25.3x margin
for its `assigned-oracle` arm is a margin for the **literal** `(1.0, 0.0)` at
`t_length.py:300-308`, not for the rule's output, which enters only through
`load_measured_mask()`. Retiring the rule does not retire that number; it
retires the claim that the rule would select that mask.

---

## What survives

The operator family. `ceq/arm_smprime.readout` contains softmax attention at
`beta = 1` and linear attention at `beta = 0`, proved in Lean and matched in
code at worst `|factorised - shipped|` of 0.0000e+00 at `beta = 0` (bitwise),
4.4409e-16 at `beta = 1`, and 5.3291e-15 at `beta = 0.37`. Nothing in this card
touches that.

What is retired is the bridge from that containment to a prediction win — the
idea that *choosing* the corner per coordinate from a measured exponent is what
the containment buys. On the encoder the exponent does not exist; on the value
axis the rule has no solution; on a null that isolates it, the assignment is
below median. No third reroute is proposed.

## Limits

Every non-Kaggle figure is from one CPU bed at seed 5501, `D_LATENT = 9`,
`S_LEN = 16`, `N_STEPS = 120`, `S_GRID_VALUE = (16, 32, 64, 128)` and
`VALUE_SWEEP_BATCH = 256`. The clean null is exhaustive over arrangements but is
one seed, one bed and one initialisation: it bounds which corner belongs on
which coordinate here and says nothing about whether a different encoder would
rank them differently. The slopes are measured, not proved; the algebra predicts
exactly −1 and no machine-checked proof of that is offered. The Kaggle rows are
a single run at one seed with no repetition, so no interval is attached to any
of them, and the 8x column rests on 486 holdout games.
