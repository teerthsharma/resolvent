# X29a — the differential second hop

`scripts/v13_deflated_hop2.py`. The second hop is propagated with the
common-mode deflated operator `a_d @ a_d` in place of `a @ a`. Hop 1 is
untouched, so the arm is `softmax`'s forward plus a differential second hop.

Every number below was produced by `python scripts/v13_deflated_hop2.py`
(self-check) and `--sweep` (trained comparison) at `torch 2.5.1+cu121`, CPU,
`threads=8`, on `feat/r9-causal-consequence` with `scale/` unmodified. No file
under `results/` was written.

## 1. What the operator is

`bench._softmax_operator` masks every `j >= i` with `finfo.min`, softmaxes the
row, then re-masks:

    (1)   a_ij = exp(w_ij) · 1[j < i] / Σ_{j' < i} exp(w_ij'),   w = q kᵀ / √d

Row 0 has empty support: the softmax over an all-`finfo.min` row returns the
uniform row `1/s`, and the trailing `masked_fill(nm, 0.0)` zeroes it. Hence

    (2)   a 𝟙 = c,   c_0 = 0,   c_i = 1 for i ≥ 1

Measured at `t*=2, n=16, seed 0, threads=8`, untrained: row sums run
`0.000000` to `1.000000` with row 0 exactly `0.000000`, and mass on or above
the main diagonal is `0.000000e+00`. `a` is **sub-stochastic, not stochastic**,
and the defect is exactly one row.

## 2. There is no Perron projection, and the deflation is not one

For a stochastic `A`, Perron–Frobenius supplies `ρ(A) = 1`, the right
eigenvector `𝟙`, the stationary left vector `πᵀA = πᵀ` with `πᵀ𝟙 = 1`, and the
spectral projection `P = 𝟙πᵀ` onto that eigenvalue. Deflation is then exact:

    (3)   A_d = A − 𝟙πᵀ  ⟹  A_d 𝟙 = 0,  πᵀA_d = 0,  A_d² = A² − 𝟙πᵀ

None of that is available here. `a` is strictly lower triangular, hence
nilpotent: `a^s = 0` and `σ(a) = {0}`. There is no eigenvalue 1, no stationary
left vector, and no spectral projection. Calling `𝟙πᵀ` "the Perron projection"
would be wrong, and the identity (3) does not hold.

What survives is the **right** half of the Perron structure — the action on the
constant direction. `a 𝟙 = c`, which equals `𝟙` on every row but row 0, so the
all-ones direction is invariant up to a one-row defect, and the second hop
inherits it (`a² 𝟙 = a c`, equal to `𝟙` off the first two rows). The dominance
is therefore a statement about the **action on the constant direction**, not
about the spectrum. The object to remove is the rank-1 map that reproduces that
action:

    (4)   P = diag(c) · 𝟙 π̂ᵀ,   π̂ = π / Σ_j π_j,   π_j = (1/s) Σ_i a_ij

`π` is the mean row of `a`, which is exactly `argmin_v Σ_i ‖a_i − v‖²`: `𝟙πᵀ` is
the least-squares best approximation to `a` **among constant-row matrices** —
the oblique projection onto the common-mode channel — and by construction
`P 𝟙 = c = a 𝟙`. This is a common-mode projector, not a Perron projector, and
the distinction is why the deflation below has to be built rather than quoted.

## 3. The naive projector breaks causality

`π_j > 0` for every `j ≤ s−2`, and `π_{s−1} = 0` exactly, because no row is
allowed to attend to the last position. So `a − 𝟙πᵀ` is nonzero on and above
the diagonal everywhere except column `s−1`:

    (5)   #{(i,j) : j ≥ i, (a − 𝟙πᵀ)_ij ≠ 0} = s(s+1)/2 − s = s(s−1)/2

Measured per example at `s=64`, untrained, `threads=8`:

| operator | mass on/above diag | max entry there | cm fraction |
|---|---|---|---|
| `a` (shipped) | `0.000000e+00` | `0.000000e+00` | 0.4006 |
| `𝟙πᵀ` | `2.598697e+02` | `7.412980e-02` | 1.0000 |
| `a − 𝟙πᵀ` (naive) | `2.598697e+02` | `7.412980e-02` | 0.0000 |
| `tril(a − 𝟙πᵀ, −1)` | `0.000000e+00` | `0.000000e+00` | 0.0296 |
| `a − diag(a𝟙)M` (renorm) | `0.000000e+00` | `0.000000e+00` | 0.6369 |

(`t*=2, n=16, seed 0`; the leak count is `2016/example`, exactly `s(s−1)/2`.)
The same table at `t*=8, n=8, seed 0` reads `1.299356e+02` / `7.407572e-02` for
`𝟙πᵀ` and the naive form, `0.4006` and `0.0296` for the two cm fractions —
reproducing an independent measurement of the same cells to four decimals
without adopting its numbers.

The leak compounds rather than cancelling: `(a − 𝟙πᵀ)²` still has
`2016/example` entries on or above the diagonal, `|mass| 1.262333e+02`, max
entry `1.064356e-01`. Substantively, row `i` of the naive deflated operator
carries weight on positions `j ≥ i`, so `a_d @ a_d @ x` moves future tokens into
the readout row. On `e3_t2` the label is a product of chain coefficients ending
at `s−1`, so an NRMSE taken through the naive form would be a measurement of the
leak, not of a capability. **The naive row-mean projector is not usable and no
reading through it counts.**

## 4. The two causal repairs, both built and both measured

    (6)   tril form:    a_d = tril(a − 𝟙πᵀ, −1) = a − mask ⊙ 𝟙πᵀ
    (7)   renorm form:  M_ij = π_j 1[j<i] / Σ_{j'<i} π_j' ,  a_d = a − diag(a𝟙) M

Both have exactly zero mass on and above the diagonal, and both stay strictly
lower triangular under squaring, so the hop count remains a hop count. They
differ in what they remove.

(6) leaves row masses `c_i − Σ_{j<i} π_j`, measured `max |a_d 𝟙| = 9.266e-01`.
A constant-along-positions input therefore still passes.

(7) satisfies `a_d 𝟙 = 0` exactly — measured `max |a_d 𝟙| = 2.421e-07` in
float32 — because the subtracted profile carries exactly the row's own mass;
scaling by `c_i` rather than by 1 is what makes it hold on the defective row 0
as well. Two structural consequences, both measured:

* Row 1, whose only admissible predecessor is position 0, is pure common mode
  and deflates to exactly zero (`row 1 max |a_d| = 0.000e+00`), as does row 0.
* Row `s−1` — the row the arm reads out — has the full prefix, so
  `M[s−1] = π̂` exactly (`max |M[s−1] − π/Σπ| = 0.000e+00`). The deflation at
  the readout row **is** the global common mode.

Projection property, with `M` held fixed at `M(a)`:

    (8)   P(h) = h − diag(h𝟙) M   ⟹   P(P(a)) = P(a)

measured `max |P(P(a)) − P(a)| = 6.706e-08`. Recomputing `M` from `a_d` instead
diverges (`1.204e+30` at this cell), because `M`'s renormalisation divides by
`Σ_{j<i} π_j`, a sum of non-negative terms for a softmax operator but a signed
sum for `a_d`, where it crosses zero. `common_mode` is defined for a
non-negative operator only; a signed operator (the `pivot_signed` arm) would
need a different projector. This is recorded rather than guarded away.

## 5. Leading behaviour of the deflated operator

Every row of `a_d` under (7) is a **contrast functional**: `Σ_j a_d,ij = 0`, so

    (9)   (a_d x)_i = Σ_j a_d,ij (x_j − λ)  for any constant λ

The row reads only how `x` varies across its admissible prefix, never its level.
The deflated second hop is a composition of two contrasts: it measures how the
variation row `i` sees differs from the variation the population sees.

Common-mode share of the energy, mean-row decomposition (every row equal to the
column mean, plus a differential remainder), `t*=2, n=16, seed 0, threads=8`,
untrained:

| second-hop term | cm fraction |
|---|---|
| `a @ a` (K=64, raw) | 0.7550 |
| `pivot_hop2` K=8 (raw) | 0.6287 |
| tril-deflated squared | 0.0075 |
| renorm-deflated squared | 0.5604 |

**The two repairs remove different things, and that is the central structural
result here.** The tril form removes the shared row *shape* (0.7550 → 0.0075)
but leaves a DC gain; the renorm form removes the DC gain exactly but leaves a
larger mean-row fraction than `a` itself (0.6369 at hop 1 against `a`'s 0.4006).
That is not a contradiction: the renorm residual's mean row is

    (10)  m̄_j = π_j (1 − w_j),   w_j = (1/s) Σ_{i>j} 1 / Σ_{j'<i} π_j'

which sums to zero, since every row of `a_d` does. It is a shared zero-sum
contrast profile — a common *shape*, not a common *mode*. It passes no DC, and
the deflation removed most of the operator's energy, so what remains has a
larger relative mean-row component.

**A disagreement, recorded not resolved.** The diagnosis this work was
commissioned from puts the `K=8` routed hop-2 term at `0.4899` of the
common-mode energy against `a @ a`'s `0.7550`. The `0.7550` reproduces to four
decimals at every cell tried; the `0.4899` does not. Measured across 18 cells
(`t*` ∈ {2,8} × `n` ∈ {8,16,64} × seeds {0,1,2}, untrained, threads=4), the
`K=8` fraction runs `0.5963` to `0.6796`, never below `0.59`, with `a @ a` at
`0.7549`–`0.7553` throughout. The qualitative claim survives — routing through 8
of 64 columns does filter common mode — but the gap is roughly `0.11`, not the
`0.27` the diagnosis states. The construction was not changed to chase this.

## 6. Must-fires

Both planted inputs are drawn from `M3_TASKS["e3_t2"][0]`, the batch builder the
arm trains on, and pushed through the production path: the arm's own `wq`/`wk`
and the arm's own `_operator`. Gain is the RMS of the deflated response over the
RMS of the raw response to the same input.

* **Common mode.** `x_cm` is each planted example's own position-mean broadcast
  over positions — a DC level taken from the batch, not a synthetic tensor.
* **Differential.** `x_d = x − 𝟙 (π̂ᵀ x)`, the same planted batch with its
  common-mode component removed under the operator's own `π̂`. This is exactly
  what the **readout row** sees as pure differential, because `M[s−1] = π̂`.

The must-fire is stated at the readout row because no input is pure differential
for every row at once. Requiring `⟨M_i, x⟩ = 0` for all `i` forces every partial
sum of `π_j x_j` to vanish, hence

    (11)  ⟨M_i, x⟩ = 0 ∀i  ⟺  π_j x_j = 0 ∀j  ⟺  x_j = 0 wherever π_j > 0

a degenerate input. The all-row figure is reported beside the readout-row one
rather than substituted for it.

Measured, `t*=2, n=16, seed 0, threads=8`, untrained, against the raw operator's
response to the same planted input:

| projector | common mode, all rows | differential, readout row | differential, all rows |
|---|---|---|---|
| renorm (7) | `7.944525e-08` | `1.000000` | `0.505241` |
| tril (6) | `3.589211e-01` | `1.000000` | `0.861934` |

**Both must-fires pass for the renorm projector (7)**: common-mode gain
`7.9e-08` against the `≈ 0` requirement, differential gain `1.000000` against
the `≥ 0.99` requirement.

**The tril projector (6) fails the common-mode must-fire**, at gain `0.358921`
— it rejects `9 dB` of DC, not all of it. It is retained and measured anyway
because it is the form that minimises the mean-row energy, and the trained
comparison in §8 adjudicates between the two rather than an argument.

## 7. CMRR

Amplifier convention: `CMRR = 20 log₁₀(raw DC output amplitude / deflated DC
output amplitude)`, the DC injection being the planted common-mode input of §6.
`t*=2, n=16, seed 0, threads=8`, untrained.

| configuration | raw DC out | renorm (7) | tril (6) |
|---|---|---|---|
| hop-1 operator | `4.162310e-02` | `3.307e-09` → **142.0 dB** | `1.494e-02` → **8.9 dB** |
| hop-2 term, K=64 | `3.894997e-02` | `5.407e-10` → **157.2 dB** | `9.372e-03` → **12.4 dB** |
| hop-2 term, K=8 routed | `5.159803e-03` | `8.512e-11` → **155.7 dB** | `2.004e-03` → **8.2 dB** |

The renorm figures are float32 noise floors, not physical limits: `142`–`157 dB`
is the ratio of a `1e-2` signal to an accumulation error near `1e-9`. The tril
figures are real attenuations of about one power of ten in energy.

The whole arm is not a differential amplifier and is not claimed to be: `z = x +
a@x + hop2@x` keeps a residual path and an undeflated hop 1, both of which pass
DC exactly as `softmax` does. The CMRR above is a spec for the **hop-2 term**,
which is the term being repaired.

## 8. The measured recovery against the pre-registered prediction

Pre-registered, before the run, in `scripts/v13_deflated_hop2.py:PREDICTION`:

> **Deflated `K=64` recovers to at most `0.960945`**, i.e. at least as good as
> raw `K=8`.

Cell, identical to the K sweep that produced the finding
(`scripts/v13_kpivot_sweep.py`): task `e3_t2`, `s=64`, `d=24`,
`n_train=2048`, `steps=150`, `lr=LR=0.02`, Adam, `y` standardised by train
mean/std and un-standardised at eval, `n_eval=4096` at `seed=12345`, seeds
`{0,1,2}`, `threads=8`, `torch.manual_seed(seed)` before each model is built.

| arm | mean eval NRMSE | sd | seeds | vs softmax | vs `1.000329` raw K=64 | ĥ | wall |
|---|---|---|---|---|---|---|---|
| softmax (1 hop, control) | **0.950252** | 0.019256 | 0.971432 / 0.933802 / 0.945523 | — | −0.050077 | 0.194041 | 289 s |
| raw K=64 | **1.000329** | 0.027601 | 1.013706 / 1.018692 / 0.968588 | +0.050077 | — | −0.001315 | 416 s |
| deflated renorm (7) K=64 | **0.951400** | 0.004973 | 0.952932 / 0.945841 / 0.955426 | +0.001148 | −0.048929 | 0.189677 | 611 s |
| deflated tril (6) K=64 | **0.958700** | 0.002995 | 0.961905 / 0.955972 / 0.958222 | +0.008448 | −0.041629 | 0.161790 | 365 s |

The softmax control reads `0.950252` (sd `0.019256`) and raw K=64 reads
`1.000329` (sd `0.027601`) — **both identical to six decimals, per seed, to the
rows in `results/r10_v13_kpivot_t2.txt`** that produced the finding. The
instrument is the same instrument; the deflated rows are the only new arithmetic.

### Verdict

**The prediction holds on the point estimate, for both projectors, and the
strong form holds only for the renorm form (7).**

* **renorm (7): `0.951400` against the pre-registered `≤ 0.960945`, clearing it
  by `0.009545`.** All **3/3 seeds** individually fall below the bound
  (`0.952932`, `0.945841`, `0.955426`). One-sample t against the fixed
  pre-registered bound: `t = 3.325`, df 2, one-sided `p = 0.0399`.
* **tril (6): `0.958700`, clearing the bound by `0.002245`** — but only **2/3
  seeds** fall below it (`0.961905` does not), and the one-sample t gives
  `t = 1.298`, one-sided `p = 0.1618`. **Inconclusive at 3 seeds.**

On the inconclusive-band criterion the two answers differ by which sd is used,
and both readings are stated rather than one chosen. Against the raw sweep's
`K=8` seed sd of `0.010289` the band is `2·sd/√3 = 0.011881` and **both** gaps
sit inside it, so the script prints INCONCLUSIVE for both. Against each arm's
own measured seed sd the bands are `0.005742` (renorm) and `0.003459` (tril):
the renorm gap of `0.009545` sits **outside** its band, the tril gap of
`0.002245` sits inside its own. The renorm arm's seed spread is a quarter of the
control's (`0.004973` against `0.019256`), which is itself part of the result.

**The recovery from the NO READING is the unambiguous part.** Raw K=64 sits
above `1.0` — predict-the-mean — on a mean of three seeds; both deflations bring
it back under the bar, and the renorm arm separates completely from it at the
seed level (`min` raw `0.968588` > `max` renorm `0.955426`; Welch `t = 3.022`,
one-sided `p = 0.0437`; exact permutation `p = 1/20 = 0.05`). Deflating the
second hop recovers `0.048929` of NRMSE, which is `98%` of the `0.050077` the
undeflated second hop destroyed.

**What the recovery is not.** The deflated arm is `+0.001148` from the 1-hop
softmax control, which at these seeds is nothing (Welch `t = −0.100`, two-sided
`p = 0.9286`). Its `ĥ = t*(1 − NRMSE²) = 0.189677` against softmax's `0.194041`,
and `floor_2 = √((t*−2)/t*) = 0` at this rung — a working second hop would drive
the error toward zero and this one does not move it at all. **X29a removes the
damage the second hop was doing; it does not make the second hop work.** The
`V13_PREDICTION_HOP2.md` reading that the hop-2 term is inert, rather than
merely underpowered, survives this repair: deflated, the term is inert instead
of harmful.

**Which projector.** The trained comparison and the must-fires agree, which is
the reason both were run: the renorm form (7) rejects DC exactly, clears the
pre-registered bound on 3/3 seeds, and lands `0.007300` below the tril form. The
tril form (6) minimises mean-row energy but passes `0.358921` of DC and clears
the bound on 2/3 seeds. **(7) is the construction; (6) is reported as its
control.** Mean-row energy fraction is not the operative quantity — DC gain is,
and the two rank the projectors in opposite orders (§5, §6).

## 9. Controls that ran with the measurement

* **Softmax control**, in the same process, same batches, same seeds — the top
  row of §8's table.
* **Identity check.** `DeflatedArm(hop2=False)` against `Arm("softmax")` at
  `torch.manual_seed(1234)` for both: **bitwise equal**, `max |diff| =
  0.000e+00`. With hop 2 off, the deflated arm's forward is `z = x + a @ x`
  followed by the shared head, which is `softmax`'s forward, and the subclass
  adds no parameters and no ops to that path.
* **Standing causality guard.** `assert_causal` runs inside
  `DeflatedArm.forward` on `a`, on `a_d` and on the hop-2 term, on **every**
  forward of **every** training step and eval — not as a one-time inspection. It
  raises on any nonzero mass on or above the main diagonal. Every number in §8
  was produced with the guard live and it never fired.

## Limits

* Three seeds. Against the raw sweep's `K=8` seed sd of `0.010289` the
  resolution band is `2·sd/√3 = 0.011881`, and **both** §8 gaps sit inside it;
  against each arm's own measured sd only the renorm gap clears. §8 states both
  readings and does not pick one. The `p = 0.0399` there is a one-sample t at
  df 2 — three points against a fixed bound, not a well-powered test.
* Recovery is not capability. `0.951400` is a return to the 1-hop control, not a
  crossing of `floor_2 = 0`. Nothing here shows the second hop computing
  anything; it shows it no longer destroying the reading.
* Depth. The differential content decays fast under composition. Measured at
  `t*=2, n=16, seed 0, threads=2`, untrained, mean Frobenius norm of the `h`-th
  power over the batch: `a` holds `2.17 → 2.26 → 2.37 → 2.10 → 1.52 → 0.90`
  across `h = 1..6`, the tril deflation falls `1.60 → 1.08 → 0.64 → 0.30 →
  0.11 → 0.035`, and the renorm deflation falls `0.808 → 0.159 → 0.034 →
  0.0068 → 0.0012 → 0.0002`. The per-hop decay ratios of the tril form
  (`1.49, 1.68, 2.12, 2.68, 3.21`) reproduce an independent measurement of the
  same quantity exactly; that measurement's absolute norms are a constant
  `4.0×` larger across every entry including `‖a^h‖`, which is a norm
  convention, not a disagreement. Nothing here licenses a claim about hops
  beyond the second, and the renorm form — the one that passes both must-fires
  — is the faster-decaying of the two.
* One cell. `t*=2, s=64, n_train=2048, steps=150`. `t*=2` was chosen because
  `floor_2 = √((t*−2)/t*) = 0`, so a working second hop has room to show; it is
  also the cell most favourable to the construction, and nothing here measures
  `t*=8` or `t*=32` trained.
* The `0.4899` figure in the commissioning diagnosis does not reproduce (§5);
  `0.7550`, `0.4006`, `0.0296` and `0.0075` all do.
* `common_mode` divides by a prefix sum of `π` and is defined for non-negative
  operators only. It is not valid for `pivot_signed`/`sgate` arms, where `π` has
  mixed signs and the denominator crosses zero (measured divergence `1.204e+30`
  when the projector is recomputed from an already-deflated operator).
* The renorm projector's exact DC null is exact only up to float32: measured
  `max |a_d 𝟙| = 2.421e-07`, and the CMRR figures of §7 are accumulation noise
  floors rather than physical rejection limits.
* The differential must-fire is exact at the readout row and only there;
  equation (11) shows no non-degenerate input is pure differential at every row,
  and the all-row gains (`0.505241` renorm, `0.861934` tril) are reported for
  that reason, not as failures of a check.
* CPU matmul reduction order varies with thread count, so every number here is
  bound to `threads=8` (the self-check figures and every trained number),
  `threads=4` (the 18-cell energy census in §5) or `threads=2` (the depth census
  above). `scale/m3_capability.py` pins 2 threads at
  import; this script overrides it after import, exactly as
  `scripts/v13_kpivot_sweep.py` does, so the trained numbers are comparable to
  that sweep's and not to the `m3_capability.py` log's.
* Two sibling measurement lanes were running on the same box during the trained
  sweep. That affects wall-clock timings only; the arithmetic is thread-count
  pinned and unaffected.
