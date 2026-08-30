# MARS / MORIARTY — R9 iteration 4: the three GREENs that had no adversary

Fast-forwarded `d95ae86 → 4d7ffb9` before touching anything. Filed **after**
Venus's dated prediction, so she cannot revise and this cannot have shaped her
number.

**All three attacks fire.** Two move a verdict; the third is a method defect the
clause's own margin absorbs, and that is reported at exactly that strength.

Running total across four iterations: **15 filed, 12 fired.**

Tests: `tests/mars/test_mars_green_attacks_it4.py` — **4/4 pass, 5.35 s.**
No wall clock taken; nothing owned by Mercury or Saturn is modified.

---

## Attack 1 — M4 eviction. FIRES. Venus is not too harsh; she understates it

**Her ruling:** the must-fire is insufficient because *"it moves the sibling
quantity, not the one reading `0.000000e+00`"*. She flags this as the call she
most expects to be wrong.

**She is right, and the true statement is stronger: no sufficient must-fire can
be written at all.**

`settle_evicted(x, keep, rho)` is
`settle_exact(evicted_operator(x, keep, rho), x[keep])`, and
`evicted_operator` is `rho * _causal_softmax(scores(x[keep]))`
(`ceq/eviction.py:103, 118-121`). **Both arguments are functions of `x[keep]`
alone.** The perturbed token is chosen by
`lowest_salience_token(x, rho, exclude=keep)`, which excludes `keep` by
construction. So the quantity reading `0.000000e+00` has an **empty rejection
region**, not an unexercised one.

Measured — the crushed slot overwritten with each of `1.0`, `1e6`, `1e12`,
`inf`, `nan`, across 8 draws, **40 checks, output bitwise identical every
time**:

| perturbation of the crushed token | `settle_evicted` moves | `settle_gated` moves |
|---|---|---|
| Gaussian redraw | **0 / 8** | **8 / 8** |
| `1.0`, `1e6`, `1e12`, `inf`, `nan` | **0 / 40** (bitwise) | — |

The `nan` arm is the load-bearing one. `nan` poisons every arithmetic path it
touches, so a `nan` in the crushed slot that does not reach the output is not
evidence of a small effect — it is proof there is **no path**.

**Consequence.** The existing must-fire (perturb a token *inside* `keep`) can
only ever show that `settle_evicted` reads its own arguments. M4's headline
`0.000000e+00` is a statement about the indexing, and the `0.000000e+00` against
`2.154868e-05` comparison is not like-for-like on the crushed arm — the two
functions are not measuring the same thing there. The existing test file already
concedes this in prose (*"this zero is STRUCTURAL … NOT worth reporting as a
measurement of exactness"*); what was missing was the value test that closes it,
and it is now in the tree.

**Verdict on Venus's self-doubt: unwarranted. Her count does not go to 2 on this
one.**

---

## Attack 2 — the E4′ gates. FIRES. The strike hangs on one unrepeatable draw

`test_the_degree_decoder_passes_at_criticality_so_e4_is_struck` conditions the
whole strike on `score < PASS_BAR = 0.5`. The score is `0.471045` — a margin of
**`0.028955`**.

`draw_balanced_marginal` seeds itself `random.Random(0x33960000 ^ case.seed)`
and splits on `np.random.RandomState(0)`. **Both are hardcoded; neither is a
parameter.** The sampling spread of the deciding number has never been measured.
Reproducing that draw verbatim with its two seeds exposed:

| | min | max | mean | sd | would reverse the strike |
|---|---|---|---|---|---|
| shipped draw, shipped split | — | — | **0.471045** | — | no (margin 0.028955) |
| 20 other **draw** seeds | 0.459390 | **0.506250** | 0.479472 | 0.010802 | **1 of 20** |
| 20 other **split** seeds | 0.471045 | 0.498577 | 0.482682 | 0.008182 | 0 of 20; closest 0.001423 off |

Two facts, both measured:

1. **1 draw in 20 reverses the strike.** The margin is `0.028955` against a draw
   sd of `0.010802` — **2.68 sd**.
2. **The shipped draw is on the favourable side of its own sampling
   distribution**, `0.008427` below the mean of the other twenty, **0.78 sd**.

This is not a claim that the strike is wrong. It is a claim that a verdict which
killed a research direction is reported **without the only error bar that could
support it**, and that the one draw it rests on happens to sit on the side that
licenses it. The number that would settle it is `P(score >= PASS_BAR)` over
draws; measured at **5%** on 20 seeds, which carries its own ±5% at that sample
size.

### One arithmetic correction to the dispatch

*"Note `+3hop` clears `FAIL_BAR` by `0.0049`."* `+3hop` reads `0.9951` and
`FAIL_BAR = 0.9`, so it clears the bar by **`0.0951`**. `0.0049` is its distance
**below the mean predictor at 1.0**, which is a different and much sharper
observation: all four decoders the gate requires to fail
(`1.0001`, `1.0018`, `1.0035`, `0.9951`) sit within `0.0049` of predicting the
mean, so the gate's FAIL half is demonstrated by decoders that are
indistinguishable from a constant. That is worth stating, but it is not what the
sentence said. Class `READ` on the recorded values — I did not re-run
`test_e4prime_registration.py`.

---

## Attack 3 — the calibrated M3 harness. FIRES as a method defect; no verdict moves

`calibrate_bar` clause 5 trains the two-feature positive control on `feats` and
scores `nrmse(net(feats), y)` on the **same** `feats`
(`scale/negation_scope.py:1502-1513`). Every arm the resulting bar gates is
scored on a **held-out** eval batch drawn at `seed + 12345`.

The E4′ decoder gate refuses exactly this, in its own docstring — *"an in-sample
reading would credit memorisation as decoding, and the strike below would be
unearned"* (`scale/rips_gate.py:163-168`). The M3 bar does the opposite, and the
two gates ship in the same tree.

Measured, same init, same 150 steps, same `lr = 0.02`:

| | `trained_two_feature` |
|---|---|
| shipped — train and score on all 2048 | `0.045561` |
| train on first half, score **in-sample** | `0.037681` |
| train on first half, score **held out** | `0.067680` |
| **in-sample bias** | **`+0.029998`** (80% relative) |

**No verdict moves.** The clause asks the control to beat `1.0` and it beats it
by `0.93` either way; `bar_verdict` returns `(True, 'BAR CALIBRATED')` under both
scorings, and that is asserted in the shipped test rather than argued in prose.

The defect is live only where the margin is not. At `e2_consequence` the same
clause read `2.446646` before Saturn's repair, and an in-sample bias pushes in
the direction that makes a **broken bar look calibrated** — the failure mode the
repair was for. That is the condition under which this becomes a verdict, and it
is not today's condition.

---

## Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | `settle_evicted` reads only `x[keep]` | `READ` | `ceq/eviction.py:103, 118-121` |
| 2 | crushed-slot `nan`/`inf`/`1e12` leaves the output bitwise identical, 40/40 | `RUN` | `torch.equal`, 8 draws × 5 magnitudes |
| 3 | Gaussian crushed perturbation: evicted 0/8, gated 8/8 | `RUN` | same file |
| 4 | M4's rejection region is empty, not unexercised | `DERIVED` | from 1 + 2; `nan` non-propagation is the step |
| 5 | shipped E4 strike score `0.471045`, margin `0.028955` | `RUN` | draw reproduced verbatim, asserted to 6 dp |
| 6 | 20 other draw seeds: `0.459390 .. 0.506250`, sd `0.010802` | `RUN` | same |
| 7 | 1 of 20 draws reverses the strike | `RUN` | `(scores >= 0.5).sum()` |
| 8 | shipped draw is 0.78 sd below the mean of the other 20 | `RUN` | same |
| 9 | 20 split seeds: max `0.498577`, 0 reversals | `RUN` | same |
| 10 | both draw seeds are hardcoded, neither a parameter | `READ` | `scale/rips_gate.py:207, 224` |
| 11 | bar control trains and scores on the same `feats` | `READ` | `scale/negation_scope.py:1502-1513` |
| 12 | in-sample `0.037681` vs held-out `0.067680`, gap `+0.029998` | `RUN` | same init, budget, lr |
| 13 | `bar_verdict` is `(True, 'BAR CALIBRATED')` either way | `RUN` | asserted |
| 14 | `+3hop 0.9951` clears `FAIL_BAR 0.9` by `0.0951`, not `0.0049` | `DERIVED` | arithmetic on recorded values |
| 15 | the recorded E4′ ladder values themselves | `READ` | dispatch; **not re-run** |

---

## What I could not validate

The E4′ attack varies the draw seed with the split seed held at 0, and the split
seed with the draw held — **not both together**, so the joint spread is wider
than either column and the 5% reversal rate is a floor on 20 seeds, not an
estimate with a usable interval. I did not re-run `test_e4prime_registration.py`,
so the ladder values `1.0001 / 1.0018 / 1.0035 / 0.9951 / 0.8220` and the planted
pair `0.0000 / 0.5530` are `READ` from the dispatch and only the arithmetic on
them is mine; in particular I did not check whether the planted binary control at
`0.5530` clears `PASS_BAR = 0.5` — on its face it does not, which would be a
second E4′ attack, and I ran out of budget before I could verify which bar that
clause actually enforces. The M4 result is a proof about indexing, so it says
nothing about whether eviction is a *good* mechanism — only that its published
zero is not evidence either way. The M3 bar attack shows an in-sample bias of
`0.029998` at `negation_scope`; I did not measure it at `e2_consequence`, which
is where I argue it would matter, so that step is `DERIVED` and the task in
question has still never been trained. Venus's count is unaffected by anything
here except the M4 line, and I did not attack the two GREENs she rules on that I
was not assigned.
