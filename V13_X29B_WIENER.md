# X29b — the Wiener equalizer for the second hop

**The declared kill, stated before any result.** If the Wiener-equalized full
hop fails to match the plain deflation on the same cell, the noise model in
(9)–(11) is wrong, and the response is to say so rather than to adjust `S_i`
or `N_i` until the numbers agree. That kill is live, because the spectrum
measured in §6 assigns a gain below the zero-correlation null band to **every
one of the 64 modes**, not to the common mode alone: the equalizer this file
derives does not reproduce a deflation that keeps 63 modes at unit gain, it
approximately deletes the whole second hop (`‖P_wiener‖_F = 0.008173` against
`‖I‖_F = 8`).

**Where it stands.** The left-hand side is measured: the Wiener-equalized
`K = 64` arm reads `0.958816` (sd `0.005158`, seeds 0–2) against the plain
`K = 64` arm's `1.000329` and the softmax control's `0.950252`, both of which
reproduce their commissioned values to six decimals. So the equalizer does
take the arm from NO READING back to LEARNS, and it beats the raw `K = 8`
arm's `0.960945` while keeping the full-rank hop. The right-hand side is not
available — `V13_X29A_DEFLATION.md` §8 still carries a placeholder — so the
kill has not been adjudicated. §9 gives the comparison, the measured left-hand
side, and inequality (13), which is the exact arithmetic condition on the
sibling's number that fires it.

Evidence classes follow `MATHEMATICS.md`. **RUN** means executed this session
and the output read; **READ** means a file and line were opened; **DERIVED**
means it follows from those by steps written out here; **CITED** means an
external result is used and named as external.

Implementation and self-check: `scripts/v13_wiener_hop.py`. **RUN**, this
session: `python scripts/v13_wiener_hop.py` → exit 0, every assertion passed;
and `python scripts/v13_wiener_hop.py --arms` → exit 0, every assertion passed,
1 419 s wall on a box carrying five other pinned jobs.
Provenance for every number below unless stated otherwise: torch 2.5.1+cu121,
CPU, float32, `torch.set_num_threads(8)`, task `e3_t2` (`t* = 2`), `S = 64`,
`D = 24`, `D_MODEL = 16`, cell batch `n = 2048` at seed 0, `K = K_PIVOTS = 64`
so that `pivot_hop2(a, P) = a @ a` exactly.

---

## 1. What the second hop is, and what it is being asked to deliver

**READ**, `scale/m3_capability.py:141-166`. `Arm.forward` computes

    (1)   z = x + A x + M x,        M = a[:, P] a[P, :],   |P| = K

and returns `readout(mlp(z)).squeeze(-1)[:, s-1]` — **only the final sequence
position**. Every statement below is therefore about row `s-1` of `M`, not
about `M` as a whole.

**READ**, `ceq/bench.py:154`. The causal mask is `tril(-1)`, so `a` is
strictly lower triangular with row 0 identically zero, and its rows sum to 1
for `i ≥ 1` and to 0 for `i = 0`: sub-stochastic, non-normal, and — the fact
§4 turns on — nilpotent.

**READ**, `scale/negation_scope.py:286-330, 354-420`. At `t* = 2` and `s = 64`
the builder zeroes the chain sub-diagonal at and before `head = s-1-t*` and
sets `b_{s-1} = 0`, so the label is exactly

    (2)   y = a_{s-1} a_{s-2} b_{s-3}  +  a_{s-1} b_{s-2}
              \___ 2-hop term ___/       \_ 1-hop term _/

with `a = x[:, :, CH_DRIVE]` Rademacher and `b = x[:, :, CH_FLIP]` standard
normal. The first product is the only part of the label a second hop exists to
supply; the second is already reachable at hop 1, which is what the softmax
control gets. **RUN**: `Var(y) = 2.018575` against the family's law
`Var(y) = t*`, and `Var(y₂) = 1.023319` against 1.

---

## 2. The estimation problem

Fix a mode index `i`. Section 5 says what a mode is; nothing in this section
depends on that choice.

Let `c_i` be a scalar random variable — the mode's coordinate, one value per
example — and let `y₂` be the estimand supplied by the oracle. Take the batch
as the probability space with uniform measure and write `⟨f, h⟩` for the
centred inner product `E[(f − Ef)(h − Eh)]`.

The estimand `s_i` is the part of `c_i` that the oracle explains:

    (3)   s_i := argmin_{u ∈ span{y₂}} ⟨c_i − u, c_i − u⟩,
          n_i := c_i − s_i

`s_i` is the orthogonal projection of `c_i` onto a one-dimensional subspace,
so it exists and is unique. The **orthogonality principle** characterises it:
`u` solves (3) if and only if the residual is orthogonal to the approximating
subspace,

    (4)   ⟨c_i − s_i, y₂⟩ = 0    ⟺    ⟨n_i, y₂⟩ = 0

and since `s_i ∈ span{y₂}`, (4) gives `⟨s_i, n_i⟩ = 0` as well. Hence

    (5)   ⟨c_i, c_i⟩ = S_i + N_i,     S_i := ⟨s_i, s_i⟩,  N_i := ⟨n_i, n_i⟩

which is a **decomposition, not a modelling assumption**: it holds by
construction of (3), and no independence, stationarity or Gaussianity was used
to get it.

---

## 3. Why (8) is the MMSE gain and not a heuristic shrinkage

The filter question is: what single number `g` should mode `i` be multiplied
by, so that the gained observation is as close as possible to the part of it
that carries signal? That is

    (6)   J(g) = ⟨s_i − g c_i, s_i − g c_i⟩

`J` is a quadratic in `g` with leading coefficient `⟨c_i, c_i⟩ > 0`, so it is
strictly convex and its unique minimiser is its unique stationary point.
Differentiating,

    (7)   dJ/dg = −2 ⟨s_i − g c_i, c_i⟩ = 0   ⟺   ⟨s_i − g c_i, c_i⟩ = 0

which is again the orthogonality principle, now in its filtering form: **the
MMSE error is orthogonal to the data**. Expanding with `c_i = s_i + n_i` and
`⟨s_i, n_i⟩ = 0` from (5),

    ⟨s_i, c_i⟩ = ⟨s_i, s_i⟩ + ⟨s_i, n_i⟩ = S_i,
    ⟨c_i, c_i⟩ = S_i + N_i,

so (7) has the unique solution

    (8)   g_i = S_i / (S_i + N_i)

This is the Wiener gain. It is the minimiser of (6) over **all** linear
estimators of `s_i` from `c_i` — affine ones reduce to these, since both
quantities are centred in §2 so any intercept is zero at the optimum — and,
because `(s_i, c_i)` enter only through second moments, over all estimators
whenever the pair is jointly Gaussian. It is not a tuned shrinkage: no free
parameter was introduced, and the minimum is global by strict convexity.

Three consequences are used later.

- **Residual.** `J(g_i) = S_i N_i / (S_i + N_i) = g_i N_i`, so the MSE the
  equalizer leaves behind is the mode's noise power scaled by its own gain.
- **Both limits are the right ones.** `N_i → 0` gives `g_i → 1`, pass the mode
  through unchanged; `S_i → 0` gives `g_i → 0`, delete the mode. **Deflation
  is the second limit.** It requires no separate rule, no threshold, and no
  identification of which mode is the common mode.
- **(8) is a squared correlation.** Substituting (5) and (3),
  `g_i = ⟨c_i, y₂⟩² / (⟨c_i, c_i⟩ ⟨y₂, y₂⟩) = corr(c_i, y₂)²`. So
  `g_i ∈ [0, 1]` always: the equalizer is a contraction and can never inflate
  a mode. That is the sharp structural difference from water-filling (§8).

**Decoupling, stated as the condition it is.** The estimator of `y₂` from all
modes at once is `ŷ₂ = Σ_i g_i c_i` only when the coordinates are mutually
uncorrelated, since then the normal equations `R_c g = r_{cy}` are diagonal.
`R_c` is measured rather than assumed: §6 reports its off-diagonal mass.

---

## 4. Why an SVD basis, and not eigenmodes

This is where the operator's structure forces the hand, and the argument is
stronger than "the eigendecomposition is ill-conditioned".

`a` is strictly lower triangular, so `a @ a` is strictly lower triangular, so
the batch-mean hop operator `M̄` is strictly lower triangular. A strictly
lower triangular matrix is **nilpotent**: `M̄^s = 0` and every eigenvalue is
exactly 0, with algebraic multiplicity `s`. Its eigenvectors do not span
`R^s`; the only basis-like object the spectrum offers is a Jordan chain, which
is neither orthogonal nor computable at finite precision. An eigenmode
decomposition here carries **no information at all** — there is no eigenvalue
to rank modes by, and no eigenbasis to project onto.

**RUN**, measured before the SVD is used, so the choice is a reading and not a
preference (`check_decomposition`):

| Quantity | Measured |
|---|---|
| `M̄` strictly lower triangular | `True` |
| `max abs(eig(M̄))` | `0.000e+00` |
| `mean abs(eig(M̄))` | `0.000e+00` |
| `max abs(Vᵀ V − I)`, SVD right basis | `9.537e-07` (float32) |

The SVD supplies what the spectrum cannot: an orthonormal basis `{v_i}` of
`R^s`, singular values that rank the modes by energy, and — because the
equalizer built from it is `P = V diag(g) Vᵀ` — a symmetric positive
semidefinite filter with spectrum in `[0, 1]`. An eigenbasis filter would be
none of those things. **SVD is what this file uses.**

---

## 5. What plays the role of `S_i` and `N_i` here, exactly

This is the part most likely to be wrong, so every object is named with the
function that computes it.

**The observation.** Let `M_n` be example `n`'s own hop operator (built
batched by `batched_pivot_hop2(a, batched_select_pivots(k, K))`,
`scale/pivot_probe.py:103,120`), `w_n = M_n[s-1, :]` its read row, and
`b_n = x[n, :, CH_FLIP]` the driver channel — the only channel (2) depends on.
Let `V` hold the right singular vectors of `M̄ = mean_n M_n`. The mode
coordinate is

    (9)   c_{n,i} = (w_n · v_i)(v_i · b_n)

and it is **exact, not a truncation**: `{v_i}` is an orthonormal basis, so
`Σ_i c_{n,i} = (M_n b_n)_{s-1}` identically. **RUN**: the reconstruction error
is `9.317e-07` relative (float32 round-off), asserted in
`check_decomposition`. Note that (9) keeps each example's **own** operator; only
the basis is shared, which is what makes the resulting equalizer a fixed
instrument rather than a per-example oracle.

**The oracle.** The bed supplies the exact top-hop term of the label's own
path sum:

    (10)  y₂ = equilibrium_hop_reading(x, t*) − equilibrium_hop_reading(x, t*−1)

`equilibrium_hop_reading` (`scale/negation_scope.py:307`) recomputes the
truncated forward scan from `CH_DRIVE` and `CH_FLIP`; nothing is stored,
sampled, or fitted. **RUN**, `check_oracle`: at `t* = 2`,
`max |y₂ − a_{s-1} a_{s-2} b_{s-3}| = 2.384e-07`, so (10) is the 2-hop path
term of (2) and not some other object. That bind exists because a quietly
wrong oracle would still produce a plausible-looking table.

**The two quantities.** With `⟨·,·⟩` the centred batch inner product of §2,

    (11)  S_i = ⟨c_i, y₂⟩² / ⟨y₂, y₂⟩        N_i = ⟨c_i, c_i⟩ − S_i

`S_i` is the energy of the mode coordinate's projection **onto the oracle**;
`N_i` is the orthogonal remainder of the **same** coordinate. Stated as a
sentence: *`S_i` is how much of what mode `i` actually delivers at the read row
is explained by the exact 2-hop path term the bed computes from the drivers,
and `N_i` is everything else mode `i` delivers.*

**What the oracle buys, itemised**, since this is the whole reason the bed was
worth using:

1. The signal subspace is **given** (`span{y₂}`), not estimated by a PCA, a
   held-out split, or a variance-components fit.
2. `S_i` is a projection energy onto a known vector, not an estimated
   signal power.
3. `N_i` is the residual of that projection, so **no noise power spectral
   density is assumed or fitted anywhere**. The usual weakest link of a Wiener
   filter — inventing `N_i` — is absent here.

**A deployable variant is reported alongside.** A real filter cannot see `y₂`.
Substituting the raw label `y` for `y₂` in (11) gives the gains a trainer could
actually fit; §6 reports both columns, and they agree in magnitude.

`wiener_gains(c, y)` takes a coordinate matrix and a target and nothing else —
no basis vector, no index, no threshold. Every statement below about the common
mode is therefore a consequence of (8), not of a branch.

---

## 6. The two must-fires, and the spectrum

### Must-fire 2 — a mode at SNR 10 keeps `g ≈ 0.909`

`_orthogonal_pair` builds `c = σ + q` with `q` **exactly** orthogonal to `σ`
and both centred, scaled so `‖q‖² = ‖σ‖²/SNR`, and hands `(c, σ)` to
`wiener_gains`. Built by projection rather than by drawing independent noise,
so the check tests (8) to machine precision instead of testing a sampling
error. **RUN**, float64, `n = 4096`:

| SNR | `S_i` | `N_i` | measured `S/N` | `g` | `SNR/(SNR+1)` | abs diff |
|---|---|---|---|---|---|---|
| 0.1 | 4062.968051 | 40629.680507 | 0.100000000 | 0.090909091 | 0.090909091 | 0.000e+00 |
| 1.0 | 4062.968051 | 4062.968051 | 1.000000000 | 0.500000000 | 0.500000000 | 2.776e-16 |
| **10.0** | 4062.968051 | 406.296805 | 10.000000000 | **0.909090909** | 0.909090909 | 4.441e-16 |
| 100.0 | 4062.968051 | 40.629681 | 100.000000000 | 0.990099010 | 0.990099010 | 4.441e-16 |

**PASS** at tolerance `1e-9`, which the measurement beats by seven orders of
magnitude.

### Must-fire 1 — a pure DC mode gets `g ≈ 0`, out of (8)

Two readings, because they answer different questions.

**(a)** An exactly DC direction `𝟙/√s` is pushed through the **same**
coordinate construction (9) and the **same** `wiener_gains`. **RUN**:
`S = 8.061893e-03`, `N = 3.143870e+01`, **`g = 2.563664e-04`**.

**(b)** The gain (8) assigns to the real operator's top singular mode — the
near-DC one, `|cos(v₀, 𝟙/√s)| = 0.4088`, carrying `σ₀²/Σσ² = 0.862035` of the
mean operator's singular energy. **RUN**: **`g₀ = 1.580144e-05`**.

Null band at `n = 2048`: under zero population correlation `g = ρ²` has mean
`1/(n−2) = 4.887586e-04` and 99.9% quantile `χ²₀.₉₉₉(1)/n = 10.828/n =
5.287109e-03`. The bound was pre-registered in the script's `--dc-bound`
default at `0.02` — 41× the null mean and 3.8× the 99.9% quantile. Both
readings clear it by more than an order of magnitude. **PASS.**

**This falls out of the formula.** `wiener_gains` contains no DC test. The
mechanism is (2) and §3's third consequence: an all-ones mode reads the
**mean** of the drivers, `Σ_j b_j`, while the oracle is the signed path
product `a_{s-1} a_{s-2} b_{s-3}`; the Rademacher coefficients are
independent of `b` and mean-zero, so the population correlation is exactly 0
and `g = corr² → 0` at rate `1/n`. Deflation is a corollary of (8), not an
input to it.

### The measured gain spectrum, and the finding

**RUN**, `report_gains`, `K = 64`, `n = 2048`, seed 0, at the arm's initial
operator. `g(y₂)` uses the exact oracle (10); `g(y)` substitutes the raw label;
`g(all chans)` is `wiener_gains_block`, which credits a mode for signal in
**any** of the 16 channels, not only the drivers.

| mode | `σᵢ²` share | `S_i` | `N_i` | `g(y₂)` | `g(y)` | `g(all chans)` |
|---|---|---|---|---|---|---|
| 0 | 0.862035 | 7.3581e-04 | 4.6565e+01 | 0.000016 | 0.000014 | 0.000052 |
| 1 | 0.107478 | 1.0020e-02 | 1.1539e+01 | 0.000868 | 0.000060 | 0.000825 |
| 2 | 0.020521 | 9.2404e-05 | 2.5506e+00 | 0.000036 | 0.000413 | 0.000072 |
| 3 | 0.005844 | 3.7537e-04 | 7.7711e-01 | 0.000483 | 0.000405 | 0.000472 |
| 8 | 0.000133 | 7.1867e-05 | 2.0867e-02 | 0.003432 | 0.002311 | 0.003084 |
| 16 | 0.000005 | 2.4764e-06 | 9.7999e-04 | 0.002521 | 0.001051 | 0.002290 |
| 19 | 0.000002 | 1.4614e-06 | 4.6767e-04 | 0.003115 | 0.000312 | 0.002808 |
| 57 | 0.000000 | 1.8748e-08 | 4.6207e-06 | **0.004041** | 0.001148 | 0.003858 |

Rows shown are modes 0–3 and four of the eight largest-gain modes, including
the maximum at mode 57; the script prints modes 34, 55, 59 and 61 as well,
each in the same `0.0016`–`0.0027` band.

- `Σ_i g_i = 0.0328` over 64 modes.
- `max g = 0.004041` at mode 57, which carries `σ²` share `0.000000` to six
  places — the largest gain in the whole spectrum sits on one of the **weakest**
  modes.
- Modes above `g = 0.01`: **0**. Modes above the 99.9% zero-correlation
  quantile `0.005287`: **0** for the scalar form and **0** for the all-channel
  form.
- Coordinate Gram off-diagonal mass fraction: `0.0484`.
- At `K = 8` (**RUN**, same cell): `max g = 0.003238` at mode 49,
  `Σ g = 0.0242`, modes above the null quantile: **0**.

**The finding, stated plainly.** At the arm's initial operator, the Wiener
solution does not merely null the common mode. It assigns every one of the 64
modes a gain inside the zero-correlation null band, at `K = 64` and at `K = 8`,
against the exact oracle, against the raw label, and with every channel
credited. The MMSE answer to "how much of the second hop should reach the
readout" is, on this cell, **approximately none of it**.

That is a coherent answer rather than a broken one: the second hop's read row
applies a **non-negative** operator to `x`, while the object it is asked to
deliver is `a_{s-1} a_{s-2} b_{s-3}`, a product of two independent Rademacher
signs with a driver. No linear functional of `x` can produce that product, and
the operator's own data dependence — the only nonlinearity available before
the MLP — carries no such structure at initialisation.

### Subsumption of the deflation, checked as an identity

**RUN**:

- `‖V diag(1) Vᵀ − I‖_max = 4.768e-07`. Setting every gain to 1 reproduces the
  plain hop, so the equalizer is the plain arm plus a diagonal in mode space
  and not a different construction. Asserted.
- `‖P_wiener‖_F = 0.008173` against `‖I‖_F = 8.000000` — a factor of `979`.
  The equalizer the data asks for is, to that precision, **the zero matrix**:
  it attenuates the hop-2 input by three orders of magnitude rather than
  reshaping it.
- `‖P_wiener − (I − v₀v₀ᵀ)‖_F / ‖I − v₀v₀ᵀ‖_F = 0.9995`, with
  `‖I − v₀v₀ᵀ‖_F = 7.937253`. **P_wiener is not close to the deflation
  projector**, and the previous line says why: the deflation keeps 63 modes at
  unit gain while (8) keeps none. Reported rather than tuned — this is the
  number that makes §9's kill live.

---

## 7. Where the common-mode energy figure lands

**RUN**, `common_mode_fraction`, the rank-1 `𝟙πᵀ` Frobenius share with `π` the
column mean:

| Definition | `K = 64` | `K = 8` |
|---|---|---|
| Batch-mean operator `M̄`, `n = 2048`, seed 0 | **0.7551** | 0.7535 |
| Per-example mean, `n = 2048`, seed 0 | 0.7551 | 0.6517 |
| Per-example DC-left rank-1 share, `n = 512`, seed 0 | 0.7551 | 0.6532 |

The `K = 64` figure reproduces the reported `0.7550` to `1e-4` under all three
definitions. The `K = 8` figure does **not** reproduce the reported `0.4899`
under any of them; the closest is `0.6517`. That discrepancy is recorded, not
resolved, and appears again under Limits.

---

## 8. Water-filling, and when it is the right instrument

**CITED.** For `m` parallel Gaussian channels with noise-to-gain ratios
`d_i = N_i / |h_i|²` and a total power budget `P`, the allocation maximising
`Σ_i log(1 + p_i/d_i)` subject to `Σ_i p_i ≤ P`, `p_i ≥ 0` is the water-pouring
solution

    (12)  p_i = (μ − d_i)_+,     μ chosen so that Σ_i p_i = P

Optimality is Gallager's: R. G. Gallager, *Information Theory and Reliable
Communication*, Wiley, 1968, §7.5 (parallel Gaussian channels). The Kuhn–Tucker
conditions for the concave programme make `p_i + d_i` constant and equal to
`μ` on the support, and require `d_i ≥ μ` off it; (12) is the unique point
satisfying them.

`water_fill` finds the level by exact sorted scan rather than bisection: with
`d` ascending, the level implied by the `k` cheapest channels is
`μ_k = (P + Σ_{i<k} d_i)/k`, and the answer is the largest `k` with `μ_k > d_k`.
**RUN**, `check_water_filling`, `d = [0.05, 0.20, 0.50, 3.00, 1e9]`:

| `P` | `μ` | active | `p` |
|---|---|---|---|
| 0.1 | 0.150000 | 1/5 | `[0.1, 0, 0, 0, 0]` |
| 1.0 | 0.583333 | 3/5 | `[0.5333, 0.3833, 0.0833, 0, 0]` |
| 10.0 | 3.437500 | 4/5 | `[3.3875, 3.2375, 2.9375, 0.4375, 0]` |

Asserted at every budget: `Σ p_i = P` to `1e-9`; `p_i + d_i = μ` on the support
to `1e-9`; `d_i ≥ μ` off it; and the `d = 1e9` channel — the stand-in for a
pure DC mode — draws exactly zero power.

### How the two differ

| | per-mode Wiener (8) | water-filling (12) |
|---|---|---|
| Objective | minimise MSE per mode | maximise `Σ log(1 + p/d)` |
| Constraint | none | `Σ_i p_i ≤ P` binds |
| Range | `g_i ∈ [0, 1]`, a contraction | `p_i` unbounded above; a mode can be **amplified** |
| Weak modes | `g_i > 0` whenever `S_i > 0`; no hard cutoff | `p_i = 0` exactly for `d_i ≥ μ`; hard cutoff |
| Coupling | modes decouple (given uncorrelated coordinates) | modes couple through `μ` |

They agree on one thing and it is the thing this file cares about: a mode with
`S_i → 0` has `d_i → ∞` and receives zero from both, at **every** budget.
**RUN**, water-filling run on the measured modes with `d_i = N_i/S_i` at the
equal-power budget `P = Σ_i g_i = 0.0328` — chosen so the two solutions spend
the same total, rather than being compared at two different scales: water level
`μ = 246.489894`, **1 of 64** modes active, DC-mode power `0.000000e+00`
(asserted). So at equal power the two agree exactly on DC and disagree sharply
on the tail: Wiener spreads `0.0328` of gain thinly across all 64 modes,
water-filling puts all of it on the single best one.

That disagreement is a budget effect and it runs both ways. `P` here is tight
relative to the measured `d_i ≈ 1/g_i ≳ 250`, so `μ` clears only one channel;
as `P` grows the support grows, and in the slack limit `p_i → μ − d_i` with `μ`
large, which is near-uniform power and funds even the near-dead modes that
Wiener correctly zeroes. Neither limit is Wiener's answer.

### Which the bed's regime calls for

**Per-mode Wiener, on two grounds.**

*The objective matches.* `run_arm` trains every arm under
`torch.nn.functional.mse_loss` (`scale/m3_capability.py:200`), and the reported
statistic is NRMSE. Mean squared error is literally the bed's criterion, and
(8) is the exact minimiser of it. Water-filling maximises
`Σ log(1 + p_i/d_i)` — a throughput, not an error — and no quantity in this
project is scored that way.

*No budget binds, so water-filling has no well-posed input.* `z = x + Ax + Mx`
is an unnormalised residual sum: no norm constraint on the state, no shared
energy allocation across hops, no capacity to maximise. `P` would have to be
invented, and §8's measurement shows the answer swings from 1 active mode to
64 as that invented number moves. (8) needs no such input — it is fixed by the
data alone. The failure this file exists to explain is **interference**
(`0.7551` of the operator's energy in a mode with measured gain `1.58e-05`),
not power scarcity.

Water-filling becomes the right instrument the moment hops must genuinely share
a fixed budget: a normalised residual stream, a fixed total hop gain spread over
a depth ladder, or any construction where `Σ_i p_i` is capped by the
architecture rather than by a choice. It is implemented and checked here for
that case, which is why (12) is in the file rather than a footnote.

---

## 9. The declared kill

**The comparison.** Both arms trained on the identical cell —
`torch.optim.Adam`, `lr = m3_capability.LR = 0.02`, 150 steps, `y`
standardised by train mean/std, `S = 64`, `D = 24`, `D_MODEL = 16`,
`n_train = 2048`, eval batch 4096 at `seed = 12345`, `torch.manual_seed(seed)`
before model construction, seeds `0 1 2`, `threads = 8`, `K = 64`, task
`e3_t2`:

    python scripts/v13_wiener_hop.py --arms --t-star 2 --n-train 2048 \
        --steps 150 --seeds 0 1 2 --k-pivots 64 --threads 8

against the sibling's plain-deflation arm at the same cell.

**What fires the kill.** The Wiener equalizer is supposed to *subsume*
deflation — deflation is (8)'s `S_i → 0` limit on one mode. A construction that
subsumes another cannot be beaten by it. So the kill fires if the
Wiener-equalized arm's mean eval NRMSE **exceeds** the plain-deflation arm's by
more than the pooled seed standard deviation of the two.

**What the spectrum in §6 predicts, so the prediction is on the record before
the sibling's number arrives.** Every measured gain sits below the null band, so
`P_wiener ≈ 0` and the equalized arm's forward is the softmax control's up to a
negligible residual term. The Wiener arm should therefore land at the softmax
control, `0.950252` at this cell, and **not** below it.

Three outcomes, and what each means:

1. **Deflation also lands at ≈ 0.950**. Both constructions reduce to deleting a
   term the arm could not use, the kill does not fire, and the honest reading is
   that the second hop carries no signal at this cell rather than that its
   common mode was in the way.
2. **Deflation lands materially below the softmax control's `0.950252`** —
   moving toward this arm's own 2-hop floor `√((t*−2)/t*) = 0.000000` rather
   than sitting at the 1-hop reading.
   The kill **fires**. A deflation that keeps 63 modes at unit gain would then be
   extracting signal from modes that (11) scored inside the null band, which
   means `S_i` and `N_i` are measuring the wrong thing — most likely because the
   signal the MLP can use is a *product* of channels rather than a linear
   functional of one (see Limits 2).
3. **Deflation lands above 0.950**. The kill does not fire and the equalizer is
   strictly better than the special case, but nothing is credited to either: an
   arm at or above 1.0 is a NO READING.

**The response if it fires is written down here in advance: report that the
noise model is wrong.** `S_i` and `N_i` are not to be redefined until the
numbers agree. The specific replacement worth testing next is named in Limits 2
— a bilinear signal model in which the estimand is a product of two channel
coordinates — and it is a different derivation, not a retuning of this one.

### Measured, this session

**RUN**, `run_arms`, the command above, single process, same eval batch and
same seed set for every row:

| arm | mean eval NRMSE | seed sd | seeds `0, 1, 2` | wall |
|---|---|---|---|---|
| softmax control | **0.950252** | 0.019256 | `0.971432, 0.933802, 0.945523` | 284 s |
| plain `K = 64` | **1.000329** | 0.027601 | `1.013706, 1.018692, 0.968588` | 413 s |
| **Wiener-equalized `K = 64`** | **0.958816** | 0.005158 | `0.962968, 0.953041, 0.960437` | 462 s |

`wiener − softmax = +0.008563`.  `wiener − plain = −0.041513`.

**Both controls reproduce the commissioned numbers to six decimals** —
`0.950252` for softmax and `1.000329` for plain `K = 64` — which binds this
process to the K sweep that produced the finding rather than merely resembling
it.

**The prediction filed above holds.** The equalizer was predicted to land at
the softmax control and not below it. The standard error of the difference of
two 3-seed means is `√(0.019256²/3 + 0.005158²/3) = 0.011513`, so the measured
`+0.008563` is `0.74` standard errors: **not resolved**, i.e. the
Wiener-equalized full hop reads as the softmax control, which is what a
near-zero `P` predicts.

Three further readings, none of them predicted in advance and all reported as
found:

- **The equalizer removes the failure.** `−0.041513` against plain `K = 64`
  is `3.5` standard errors of that difference (`0.011744`), so the recovery
  from NO READING to LEARNS is resolved at three seeds even though the gap to
  softmax is not. The mechanism is not subtle: `‖P_wiener‖_F = 0.008173`
  against `‖I‖_F = 8`, so the recovery is achieved by attenuating the hop-2
  term by a factor of `979`.
- **`0.958816` beats the raw `K = 8` arm's `0.960945`.** Weighting all 64 modes
  by (8) does better than the accidental common-mode filtering that pivot
  routing supplied, and does so while keeping the full-rank hop.
- **Seed sd falls 3.7×**, `0.019256` → `0.005158`. The equalized arm is
  markedly more stable across seeds than the softmax control it matches in
  mean. Nothing in §2–§5 predicts this and no claim is made about why.

### The spectrum after training, and what it settles

**RUN**, `gains_after_training`, seed 0, plain `K = 64` arm after the same 150
steps (its own eval NRMSE `1.013706`):

| Quantity | At init | After 150 steps |
|---|---|---|
| common-mode share of `M̄` | 0.7551 | 0.7540 |
| `max g` (scalar) | 0.004041 (mode 57) | 0.002346 (mode 48) |
| `Σ_i g_i` | 0.0328 | 0.0295 |
| modes above `10.828/n = 0.005287` | 0 | 0 |
| `max g` (all channels) | 0.003858 | 0.002181 |
| `g₀`, top singular mode | 1.58e-05 | 3.037560e-04 |

Training does not lift a single mode out of the null band, and the common-mode
share does not move. So the "fitted at initialisation" objection to (9)–(11) is
**measured, not merely acknowledged**: the second hop acquires no linear signal
about the exact 2-hop oracle over the course of the run that its own arm fails.

### Kill status

**The kill has not fired, and it cannot fire on the evidence in this file.**
The left-hand side is measured (`0.958816 ± 0.005158`); the right-hand side is
not — `V13_X29A_DEFLATION.md` §8 still reads `SWEEP_TABLE_PLACEHOLDER` as of
this writing. The firing threshold is now arithmetic rather than a description:
with the sibling's deflation mean `m_d` and seed sd `s_d` over the same three
seeds, the kill fires when

    (13)  0.958816 − m_d  >  √(0.005158²/3 + s_d²/3)

At the sibling's own quoted resolution scale (`s_d ≈ 0.0103`, their `K = 8`
seed sd), (13) fires for `m_d < 0.952` and does not fire for `m_d ≥ 0.952`.
Their pre-registered target is *"deflated `K = 64` recovers to at most
`0.960945`"*; a deflation landing anywhere in `[0.952, 0.961]` leaves the kill
unfired and the two constructions indistinguishable at three seeds.

---

## Limits

1. **The equalizer is fitted at initialisation and frozen.** `bed` reads
   `q`/`k` from a freshly seeded `Arm` before any optimiser step, so the gains
   describe the operator the arm starts from, not the one it trains to. This
   limitation is measured rather than left open (§9, `gains_after_training`):
   the spectrum re-read from the *trained* operator is still entirely inside
   the null band, so the init-time fit is not what is doing the damage. It
   remains a limit because a *different* training regime — more steps, a
   different `lr`, a signed operator — could still move it.
2. **The signal is defined by linear correlation with the oracle.** The MLP is
   nonlinear and can multiply channels, so a mode carrying signal only
   multiplicatively — exactly the `a·a·b` shape of (2) — reads as pure noise
   under (11). `wiener_gains_block` credits every channel and is a partial
   mitigation only: it moves no gain out of the null band. This is the leading
   candidate explanation if §9's kill fires, and repairing it means a bilinear
   estimand, which is a different derivation.
3. **Diagonal gains are MMSE only for uncorrelated coordinates.** The measured
   Gram off-diagonal mass fraction is `0.0484`, so the diagonal form is an
   approximation here; the exact solution is the full normal-equations solve
   `R_c g = r_{cy}`, which is not implemented.
4. **The basis is shared, the operators are not.** `V` comes from the
   batch-mean operator `M̄` while each example contributes its own `M_n`. That
   is deliberate — a per-example basis is not a fixed instrument — but it means
   modes are optimal on average rather than per example.
5. **The `K = 8` common-mode share does not reproduce the quoted `0.4899`**
   under any of three definitions (§7: `0.7535`, `0.6517`, `0.6532`). The
   `K = 64` value reproduces to `1e-4`. Whatever definition produced `0.4899`
   was not recovered here, and no number in this file depends on it. This
   non-reproduction is independent of, and corroborated by,
   `V13_X29A_DEFLATION.md` (Limits), which reaches the same conclusion from a
   different decomposition — two agents failing to recover the same figure by
   different routes makes the figure itself the thing to re-derive.
6. **One rung, one width, one machine.** Everything is `e3_t2` at `s = 64`,
   `d_model = 16`, CPU float32, `threads = 8`, torch 2.5.1+cu121. CPU matmul
   reduction order varies with thread count (`scale/m3_capability.py:64`), so
   the sixth decimal of any arm number is thread-dependent.
7. **The null band uses the large-`n` χ²(1) approximation** for `ρ²`. For the
   all-channel column the effective degrees of freedom exceed 1, so the quoted
   `10.828/n` is a **lower** bound on the correct band there and the "0 modes
   above" reading for that column is conservative in the safe direction.
8. **Three seeds.** The softmax control's own seed sd is `0.019256`, so with
   3 seeds per arm the resolvable difference between two means is about
   `0.0115`. The `wiener − softmax` gap of `+0.008563` sits below that and is
   **not resolved**; only the `wiener − plain` gap of `−0.041513` is. No
   ordering between the Wiener arm and the softmax control should be read from
   this file.
9. **The 3.7× drop in seed sd is unexplained.** `0.019256` → `0.005158` is
   reported because it was measured, not because anything in §2–§5 predicts a
   variance reduction. At three seeds an sd ratio is itself poorly determined.
10. **No claim is made about the sibling's deflation number**, which was still
   a placeholder in `V13_X29A_DEFLATION.md` §8 at the time of writing. §9
   states the comparison, the measured left-hand side, and inequality (13);
   it does not state a verdict, and (13) must be evaluated against the
   sibling's final table rather than against any interim figure.
