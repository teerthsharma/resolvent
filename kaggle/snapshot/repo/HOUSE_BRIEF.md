# BRIEF FOR DR HOUSE — the whole state, and the honest accounting

You have the run of the repository. You may edit `LOOP_PROMPT.md`, the governing
contract, including deleting whole sections of it. You may call the work stupid if
it is stupid. Nothing below is defended; it is reported.

---

## 1. THE GOAL, IN THE AUTHOR'S OWN WORDS, UNCHANGED FOR SEVEN ROUNDS

> "not the best token predictor — the next-equilibrium predictor. Attention that
> understands causality and consequences on Turing-grade problems at the smallest
> scale and the whole attention module build to survive equal to self attention or
> supersede it"

Trainable on free tiers. On HuggingFace. With a capability table.

---

## 2. THE ACCOUNTING. READ THIS BEFORE ANYTHING ELSE.

**Toward that goal, after seven rounds, the following exists:**

* No capability table.
* No HuggingFace upload.
* No Kaggle run.
* No comparison against self-attention on any causal task at any scale that is not
  a smoke cell.
* No certificate. Three attempts, three deaths, program closed by contract.
* No theoretical frame for the capability claim. The Hankel/nonnegative-rank frame
  died this iteration at its third joint (§4.9).

**What does exist, and it is not nothing:**

* 243 passing tests and the best falsification apparatus in the project's history.
* Eleven vacuous controls caught and struck, across five authors.
* An anytime-valid evidence process, calibrated in both directions, with a control
  seen to fire.
* A hash-sealed journal set that has been checked and holds.
* A dozen results that are *true*, small, and mostly negative.

**The one-sentence version.** Seven rounds produced a world-class machine for
killing this project's own ideas, and almost nothing that predicts an equilibrium.
Every instrument built has worked. Every theory those instruments were pointed at
has died. The scoreboard reads 23 and exactly **+1** of it was earned on the
current board.

---

## 3. WHAT HOLDS. Short list, because it is short.

| fact | number | provenance |
|---|---|---|
| `κ = β`, the chosen dial | holds **142/142** rows | `results/hilbert.jsonl`, `bound_respected` |
| e-process null calibration | worst crossing `0.0367 ± 0.0019` vs `α = 0.05` | `scale/eprocess.py`, `n_rep=10000` |
| the broken-λ control fires | `1.0000` | same |
| supermartingale property | holds, proved two ways, must-fire at `t=2` | `METHODS.md` §2 |
| Merkle seal, first use | 22 unchanged, 1 append, **0 edits**, of 23 | it.5 |
| fused settling step | `3.0 → 1.125` dispatches/step, journal bit-identical | `scale/settle.py` |
| trained cell reproduces | 3 seeds, `n_params = 4769`, `eval_nrmse` `0.7475/0.7202/0.7662` | `results/phaseD_weights_*.pt` |
| **the signed arm IS signed when trained** | min entry `−0.136364`, `neg_frac 0.0135285` | Foreman it.6 |

That last row is new and it reverses a standing kill. At random init every arm is
entrywise non-negative and the "signed" operator is the unsigned one times a gain
minus a per-row constant, to 4 parts in 100,000. **Training crosses the sign
floor**: `max_range` goes `0.204402 → 52.2851`, which is `45.4×` the threshold
`½ln(1/λ) = 1.151292546497023`.

---

## 4. EVERY DEATH, WITH ITS CAUSE. This is the part you were called for.

**4.1 The sphere certificate.** Died on an identity: `θ = arcsin(√TV)` exactly on a
one-token mask. The geometry was a restatement of the quantity it was supposed to
bound.

**4.2 Birkhoff on T.** Needs a positive **LINEAR** map. Six sources agree
(Lemmens–Nussbaum Thm 2.9 is an equality; nonlinear maps get Cor 2.7, nonexpansive
only). The map is not linear.

**4.3 Birkhoff on G — your own last leap.** The pivot-Gram factorisation failed its
own per-seed bind. One kill inside it was an error of mine (`0.961793` came from a
map carrying two Birkhoff factors), but the bind failure stood.

**4.4 The dial verdict, final.** `κ = β` is **chosen**, with a known transfer
function. Proposing another contraction certificate is a contract stop.

**4.5 Float saturation.** `tanh(Δ/4)` reads exactly `1.0` for `Δ ≥ 76.246190` in
float64. Repaired with the closed form `1−κ = 2/(e^(Δ/2)+1)`, representable to
`Δ ≈ 1400`. **The certificate is now readable and still fails**: at trained weights
`κ_G = 0.9893068617` at the median and the tail grew `0/512 → 2/512 → 26/512`.

**4.6 `delta_image`, the "only non-saturating diameter".** Saturates in **118 of
142** rows, ranges to `133.225686`, and reads exactly `0.0` in 6 rows where the
sampler found no two distinguishable image points.

**4.7 The 5-seed headline cell.** Cannot cross the evidence threshold in either
direction, ever, whatever numbers land. `|d| ≤ B` and `λ ≤ 1/2` force every factor
`≤ 1.5`, so `MIN_T_MIXTURE = 11` and the maximum reachable `E_5` is `3.801691`
against a threshold of `20`. Found before the first seed. Repriced at **436 units
against 10 planned**.

**4.8 The contract's own worked example.** Clauses C5a and C5b are false, on three
independent measurements. Shifting the counter's Hankel matrix does **not** give
`rank 3` — it gives **2**, because `H_f = a1ᵀ + 1aᵀ` already carries `1` in its
column space. And `rank₊ = 2` **exactly**, certified bitwise. **Gap zero.** The
error was reading "takes `n` distinct values" as "`rank₊` is `Ω(n)`". Hrubeš 2012
caps the **entire counter family** at `rk₊ ≤ 2log₂n + 2`, so the gap sought is not
merely absent here — it is unavailable in the family.

**4.9 THE FRESH ONE, AND IT IS THE WORST. The whole capability frame just died.**
The S2 program says: find a task whose Hankel matrix has `rank₊ > rank`, and the
non-negative arm must then lose at matched parameters. **Joint 3 is refuted, at
trained weights, measured.** The trained *non-negative* arm puts a **negative**
sign on a token's influence in `0.492188` of drawn interventions. Swap `GELU` for
`nn.Identity()`, same trained weights, one object changed: **`0.000000`**.

**The nonlinearity supplies the sign the operator withholds.** So the M3 arm is not
a nonnegative weighted automaton, and a bound on nonnegative automaton state count
bounds *nothing* about it. The readout is `Linear → GELU → Linear`; the theorem is
about linear value paths.

**4.10 Dyck-1**, the reroute target, has **no gap**: `rank = rank₊lb = MN = n+1`.

**4.11 The gap ceiling.** Best certifiable gap on the surviving task
(`counter_squared`) is **3**, and the instrument returns nothing at `n=7`. To reach
`d_model = 16` needs `n ≥ 10`. **The theorem's currency is smaller than the arm's.**

**4.12 S6's cost gate.** Unreachable by arithmetic: the settling arm does strictly
more work than the glance, so the floor is `7/6 = 1.1667` against a bar of `1.1`.

**4.13 L4 (Lean).** Dead as written. Fliess/Carlyle–Paz needs a **field**;
Berstel–Reutenauer switches hypothesis exactly at the definition of rank.

**4.14 Prior work.** `caustic`'s orbit partition is not a partition of a cone (87
blobs swept, zero hits). `sigmoid` has no Dobrushin, no TV, no Cheeger (30 modules,
zero) — its only contraction number is a spectral norm, and it exports a
`hilbert_coefficients` that is a graded-module **series**, a trap for anyone
grepping.

**4.15 No within-sequence norm-matched control exists.** Top-k by norm means every
other within-sequence set is strictly lower; the matcher closed **zero** of the gap
in **512/512** draws. Structural. No mechanism claim about *which* tokens is
permitted.

---

## 5. THE PATTERN, STATED SO YOU CAN REJECT IT

Every death above has the same shape: **an instrument was built to measure
something adjacent to the thing that matters, it worked perfectly, and the thing it
measured turned out not to be the thing.**

* Birkhoff measures contraction of a *linear* map. The map is not linear.
* `rank₊` bounds states of a *linear-value-path* automaton. The arm has a GELU.
* The Hilbert diameter measures a *cone*. The iteration visits an image.
* The sphere metric measured a quantity equal to the one it was bounding.

Seven rounds of this. The instruments are excellent. **The adjacency is the
disease.**

---

## 6. WHAT IS LIVE RIGHT NOW

* The `twin` arm is running — it had **zero rows** while three sibling arms were
  complete at five seeds, on the round's hard deadline, owned by nobody.
* Cameron is putting `counter_squared` into the M3 corpus. Whether it survives 4.9
  is now doubtful.
* `counter_squared`'s M3 batch is **below the sign floor at init**
  (`max_range 0.397573` vs `1.151292546497023`, `lam_needed 0.602710`).
* `lam = 0.10` is the shipped setting and it **cannot produce a sign**. The window
  that can is `λ ∈ (0.763205, 1.0)`, and moving there costs row sum
  `1.227273 → 0.204545`.

---

## 7. WHAT YOU ARE ASKED FOR

Leaps. Plural, if you have them. Toward **the goal in §1**, not toward the
scoreboard. The scoreboard is a proxy that has been optimised for seven rounds
while the goal has not moved.

You may:

* **Edit `LOOP_PROMPT.md`.** Delete items. Rewrite the board. Change what scores.
  If the contract is aiming the work at the wrong target, say so and re-aim it.
* **Declare the whole capability frame retired** and name what replaces it.
* **Say the goal itself is mis-stated**, if it is, and restate it.
* **Call the work what it is.**

You produce **hypotheses, not findings**. You are exempt from the
RED-test-first rule because five minutes does not fit a test. That exemption is
exactly why your output cannot enter a verdict directly — it re-enters the
differential and gets bound by a fellow, with a RED test, later. A leap that never
gets bound stays in Open forever.

What is *not* useful: another contraction certificate (contract stop), another
instrument to measure something adjacent, or a restatement of the goal in nicer
words.
