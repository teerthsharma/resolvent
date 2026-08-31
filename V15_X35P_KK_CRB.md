# V15 X₃₅′ (c) KRAMERS–KRONIG AND (d) THE LOCALIZATION FLOOR

CEQ_V15_2_DELTA.md instruments **(c)** and **(d)**, against the kill *"any KK or
CRB number quoted before its must-fire ⇒ STRUCK."*

| instrument | must-fire | verdict |
|---|---|---|
| **(c)** KK causality residual | planted anticipating kernel reads nonzero | **FIRED. SHIPS.** `0.9899` against a null of `1.37e-15` — margin `7.2e14` |
| **(d)** Cramér–Rao localization floor | noise sweep against the CRB curve | **DOES NOT SHIP AS WRITTEN.** No CRB exists for this bed's parameter; a Ziv–Zakai bound is supplied in its place and its sweep is run |

**No CRB number is quoted anywhere in this file.** §7 contains numbers computed
from a *smoothed surrogate* of the CRB; they are reported as evidence that the
surrogate is a free knob, and are the argument for not shipping a floor rather
than a floor. The bound in §8–§10 is a Ziv–Zakai bound and is labelled as one
in every row it appears in.

Code: `ceq/x35p/kk.py`, `ceq/x35p/crb.py`. Tests: `tests/x35p/test_kk_crb.py`,
29 tests, RED pasted at §11 before either module existed, GREEN after.
float64 throughout. Nothing trains.

---

## PART (c) — THE KRAMERS–KRONIG CAUSALITY RESIDUAL

## 1. The relation, and what the probe computes

A real sequence `h` is causal exactly when its odd part is `sgn(k)` times its
even part. The DFT of the even part is `Re H` and the DFT of the odd part is
`i·Im H`, so under causality `Im H` is a function of `Re H` alone — the
discrete Kramers–Kronig relation. The probe throws `Im H` away, rebuilds it
from `Re H` under the causal hypothesis, and compares:

    residual = || Im H − Im Ĥ ||₂ / √M,     Ĥ = F[ P_causal F⁻¹ Re H ]

`P_causal` keeps bin 0 and Nyquist, doubles the first half, zeros the second.
The probe consumes `H(ω)` and never looks at the taps, which is the point: it
applies to a kernel observed only through its frequency response.

**It has a closed form, and that is what makes it an instrument rather than a
flag.** `H − Ĥ` is the DFT of the odd extension of the anticausal taps, so by
Parseval

    residual = √2 · || anticausal taps ||₂

exactly. `kk_residual` returns that as `closed_form` beside the
frequency-domain reading; the two are computed by different routes and their
agreement is a check, not an identity (`MISTAKES.md` V-3). Measured agreement
on the must-fire kernel: `1.110e-16`, and on 32 random 16-tap kernels with
6 anticausal lags, `rel ≤ 1e-10`.

There is **no threshold and no tuning knob**. The reading is the anticausal
amplitude in the kernel's own units, times √2.

## 2. THE MUST-FIRE — a planted anticipating kernel

Kernel `lags = [−3, 0, 1, 2]`, `taps = [0.7, 1.0, −0.5, 0.25]`, on `M = 16`
bins. The tap at lag `−3` is the plant.

| quantity | value |
|---|---|
| **KK residual** | **`0.98994949366116647`** |
| closed form `√2 × 0.7` | `0.98994949366116658` |
| route disagreement | `1.11e-16` |
| relative (`/ RMS Im H`) | `1.5628` |

**Predicted before the run.** A single anticausal tap of amplitude `a` must
read `a√2` — `0.7 × 1.41421356 = 0.98994949`. The expected value was computed
from the closed form before the probe was executed, which is the discipline
`MISTAKES.md` V-10 asks for and which the struck probe did not have.

## 3. THE OTHER HALF — a probe that fires on everything measures nothing

| causal kernel | KK residual |
|---|---|
| delay `d = 0` | `0.0000e+00` |
| delay `d = 1` | `7.8505e-17` |
| delay `d = 2` | `7.8505e-17` |
| delay `d = 5` | `1.0478e-16` |
| delay `d = 13` | `5.2388e-17` |
| delay `d = 40` | `4.2208e-17` |
| 256 random 24-tap causal kernels, **worst** | `1.3670e-15` |
| `bed_k.kernel_matrix("delay", 64, d=4)`, row 48 | `6.2354e-17` |
| `bed_k.kernel_matrix("powerlaw", 64, H=0.6)`, row 48 | `2.4951e-17` |
| `bed_k.kernel_matrix("powerlaw", 64, H=0.8)`, row 48 | `1.0770e-16` |
| `bed_k.kernel_matrix("powerlaw", 64, H=0.95)`, row 48 | `1.7446e-16` |

**Null maximum `1.3670e-15`**, which is float64 round-off on an FFT of that
size and not a small real reading.

> **MARGIN: `0.98995 / 1.367e-15` = `7.24e14`.**

The bed's own kernels are in that table because they are the only kernels this
round has: nothing trains (L-LEAN), so no learned kernel exists in scope. The
delta's *"on every learned kernel"* is therefore **owed, not discharged** — see
§12. The shipped attention operator's causality is enforced by a mask
(`ceq/attention.py::_strict_causal_mask`), so it is causal by construction and
the probe would read zero on it for a reason that has nothing to do with
learning.

## 4. The probe is graded in the violation, not thresholded

Kernel `lags = [−2, 0, 1]`, `taps = [a, 1.0, 0.3]`:

| `a` | residual | `a√2` | relative |
|---|---|---|---|
| `0` | `4.807e-17` | `0` | `2.27e-16` |
| `1e-9` | `1.414214e-09` | `1.414214e-09` | `6.67e-09` |
| `1e-6` | `1.414214e-06` | `1.414214e-06` | `6.67e-06` |
| `1e-3` | `1.414214e-03` | `1.414214e-03` | `6.67e-03` |
| `0.01` | `1.414214e-02` | `1.414214e-02` | `6.66e-02` |
| `0.1` | `1.414214e-01` | `1.414214e-01` | `6.32e-01` |
| `1.0` | `1.414214e+00` | `1.414214e+00` | `1.916` |
| `10.0` | `1.414214e+01` | `1.414214e+01` | `1.999` |

Linear over nine decades. An anticipation of `1e-9` still reads six orders of
magnitude above the `1.37e-15` null, so a kernel that is *slightly* acausal
gets a number rather than a verdict — the same property that makes X₃₅a's
truncation control useful (`V15_X35A_RESIDUAL.md` §5: FAR graded `0.0120 →
0.2667 → 1.0000` in the visible model's error).

**Report the absolute, not the relative.** The `relative` column saturates at
2.0 as the anticausal tap comes to dominate the kernel: it is the normalizer
growing with the numerator, which is exactly diagnosis candidate 4 of §5 in
mild form. `kk_residual` returns both and this file reads the absolute.

---

## 5. DIAGNOSIS — why the original read `0.000`

Four naive implementations, each written out in `ceq/x35p/kk.py` under a
section that shipping code may not call, run on one anticipating kernel:
`lags = [−5, −1, 0, 2, 6]`, `taps = [1.0, −0.8, 1.0, 0.4, −0.2]`, which
anticipates by five samples and carries `‖anticausal‖₂ = 1.2806`.

| variant | reading | reproduces `0.000`? |
|---|---|---|
| **rebuilt probe** | **`1.811077`** | — |
| **1. bare tap array, no lag axis** | `1.657e-16` | **YES, on every input** |
| **2. residual taken on `\|H\|`** | `3.269e-16` | **YES, on every input** |
| **3. FFT with no zero-padding** | `0.000000e+00` (aliased kernel) | **YES, conditionally** |
| 4. divided by `M` instead of `√M` | `5.66e-2` @ `M=1024` | no |

### 1 — the origin-forgetting bug. Reads zero on every input.

Taps handed over as a bare array, index 0 taken to be lag 0. **Causality is
not a property of a sequence; it is a property of a sequence relative to its
time origin.** A shift of the origin is a linear phase, and linear phase
preserves causality, so *every* finitely-supported sequence is causal once you
forget where its origin is. The reading is `1.66e-16` on a kernel that
anticipates by five samples, and it is `~0` on all inputs including causal
ones. This is `MISTAKES.md` V-16 in its purest form: the instrument has no
third outcome, and the outcome it collapses everything into is "pass".

### 2 — the residual on `|H|`. Reads zero by construction.

The KK partner of `|H|` is the **phase**, not the imaginary part.
Reconstructing it yields the minimum-phase spectrum `exp(A)` with
`Re A = log|H|`, whose magnitude is the input magnitude **by construction**, so
a residual taken between magnitudes has no rejection region at all — a gate
whose threshold is satisfied by construction, `MISTAKES.md` V-10, and the phase
where causality lives is never looked at. Measured `3.27e-16`, which is the
`1e-300` magnitude floor and float round-off, not a signal.

### 3 — no zero-padding. Reads zero conditionally, and that is worse.

FFT length set to the kernel's own support with no guard band. Measured on
`lags = [−7, −6, 0, 1]`: `M = 10`, the lags wrap to indices `[3, 4, 0, 1]`,
**both anticausal taps land below the Nyquist index 5** and are
indistinguishable from long positive delays. Reading `0.000000e+00` exactly,
where the rebuilt probe reads `2.000000`.

The same variant on the `−5` kernel reads `1.811077` — the correct value —
because there `M = 12` and the wrapped indices `[7, 11, 0, 2, 6]` happen to
land above Nyquist. **A probe that is right or wrong depending on the ratio of
the kernel's anticipation to its causal extent is the harder failure to catch**,
because it passes its own spot check. `kk_residual` refuses rather than
guessing: `ValueError: grid too small … leaves no guard band`.

### 4 — ELIMINATED as the mechanism.

The correct numerator divided by the bin count `M` rather than `√M`:
`5.660e-2` at `M = 1024`, `7.075e-3` at `M = 65536`, `1.769e-3` at `M = 2²⁰`.
It shrinks as `1/√M` — 64× more bins buys 8× — so reaching a printed `0.000`
would need `M > 1.3e7`. It prints `0.002` at a million bins. **A normalizer
that grows with the numerator cannot produce an exact zero**; only a numerator
that is identically zero does that robustly, which is why the three variants
above are candidates and this one is not.

### What the diagnosis reduces to

Candidates 1 and 3 are the **same root mechanism at two severities: the time
origin was not carried with the taps** — lost entirely, or lost modulo `M`.
Candidate 2 is a different mechanism: **the phase was discarded**. The single
reported datum (`0.000` on one anticipating kernel) does not separate them, and
this node will not guess which the author wrote.

**The discriminating experiment, run here so it is not owed as a suggestion.**

*(a) Negate the lag axis on the `−5` kernel.* A correct probe moves; a probe
that never had the lag axis cannot.

| | reading |
|---|---|
| rebuilt, lags as given | `1.811077` |
| rebuilt, lags negated | `0.632456` |
| candidate 1 (bare array), either way | `1.657e-16` |

*(b) A zero-phase kernel: `lags = [−1, 0, 1]`, `taps = [0.5, 1.0, 0.5]`.*

| | reading |
|---|---|
| rebuilt (absolute) | `0.707107` = `√2 × 0.5`, as predicted |
| `max\|Im H\|` | `0.000e+00` — **the spectrum is exactly real** |
| rebuilt (relative) | `nan` |
| candidate 2 (on `\|H\|`) | `3.964e-14` |
| candidate 1 (bare array) | `1.963e-17` |

Test (a) separates {1} from a correct probe. Test (b) separates a correct
*absolute* reading from every normalized one. Separating 1 from 2 needs the
struck source, which is not in this repository.

### 5b — the fifth observation, and it is the brief's real-spectrum candidate

The zero-phase kernel above is **maximally anticausal for its energy** — half
its weight sits at a negative lag — and its spectrum is **exactly real**,
`max|Im H| = 0.000e+00` to the bit. Two consequences, both measured:

- **Any probe keyed on `Im H` being "explained by" `Re H` reads zero here**, not
  because causality holds but because there is no imaginary part to explain.
  This is the brief's candidate *"a discrete Hilbert transform applied to a
  real-valued spectrum where the relation is trivially satisfied"*, and it is
  real — but note it is a property of the **kernel**, not of the probe: it
  strikes only the symmetric family, so it cannot by itself explain a `0.000`
  reported on an arbitrary planted anticipating kernel.
- **Every relative normalization is undefined here.** `‖Im H‖ = 0` makes the
  `relative` field `nan` while the absolute reading is a perfectly good
  `0.707107`. §4's "report the absolute" is therefore load-bearing rather than
  stylistic: on the one kernel family where a normalized probe is worst, it
  does not return a small number, it returns no number at all — and a caller
  that coerces `nan` to `0.0` has rebuilt V-16 one layer up, which is exactly
  where `MISTAKES.md` V-16 says these get written.

**MISTAKES.md entry owed either way** (this node does not write that file):
*an instrument whose input representation discards the property it measures.*
The lag axis is not metadata; it is the entire content of the causality
question, and a probe that accepts a kernel without one has already answered.

---

## PART (d) — THE LOCALIZATION FLOOR

## 6. What the bed's residual is, measured rather than assumed

With the oracle visible model, `ceq/x35/residual.py` gives
`r = observation noise + planted latent` exactly, and `bed_k._plant_field`
drives the plant with `u_i = magnitude · g_i` for `i ≥ index`. So

    r_i ~ N(0, σ²)          i < t
    r_i ~ N(0, σ² + m²)     i ≥ t

independent across `i`. **This is a variance change-point at a discrete index**
— not a delay of a known waveform, and not a mean shift.

Checked on the production path rather than asserted (`MISTAKES.md` M-18: a
diagnostic registered against a quantity whose distribution in the corpus was
never checked). 2000 runs, `n = 64`, `t = 32`, `σ = 1.0`, `m = 1.5`:

| | measured | model |
|---|---|---|
| `E[r²]`, `i < 32` (64 000 samples) | `0.99396` | `1.00000` |
| `E[r²]`, `i ≥ 32` (64 000 samples) | `3.27314` | `3.25000` |

## 7. THE CLASSICAL CRB DOES NOT EXIST FOR THIS PARAMETER

The parallel node's objection is **confirmed**, and it is stronger than
non-differentiability alone. Four independent reasons, three of them measured.

### 7.1 The parameter is an integer

The Cramér–Rao inequality requires `∂/∂θ log p(x; θ)` to exist and the
regularity conditions (interchange of `∂` and `∫`, finite non-zero Fisher
information) to hold. A discrete parameter has no score. There is nothing to
invert.

### 7.2 Embedding it in ℝ with the jump kept abrupt gives `I = 0` and a bound of `+∞`

`v_i(τ) = σ² + m²·1[i > τ]` is a **step in `τ`**, so `∂v_i/∂τ = 0` almost
everywhere and `I(τ) = Σᵢ ½(∂vᵢ/∂τ)²/vᵢ² = 0` identically. Measured:
`fisher = 0.0`, `crb = inf`.

The same fact from the data side — log-likelihood at `n = 64`, seed 11:

| `τ` | `log L` |
|---|---|
| `32.0` | `−87.987269894673858` |
| `32.25` | `−87.987269894673858` |
| `32.9` | `−87.987269894673858` |
| `32.999999` | `−87.987269894673858` |
| `33.0` | `−87.677751362587443` |
| `33.5` | `−87.677751362587443` |

Bit-identical across each unit interval. **A bound of `+∞` is satisfied by
every estimator**, which is `MISTAKES.md` V-10 — a gate with no rejection
region — reached from the bound side rather than the threshold side. Printing
"distance to CRB" against `+∞` would report a pass for every detector forever.

### 7.3 Smoothing creates a bound, and the smoothing parameter *is* the bound

Replace the step by a logistic ramp of width `w`. The Fisher information
becomes finite: `I(τ) = Σᵢ ½(m²/w)² φ′((i−τ)/w)² / vᵢ²`. Measured at `τ = 32`,
`σ = m = 1`, `n = 64` — **these are surrogate numbers, not a floor:**

| ramp width `w` | Fisher `I` | `1/I` | `√(1/I)` |
|---|---|---|---|
| `0.05` | `5.556e+00` | `1.800e-01` | `0.424` |
| `0.125` | `8.889e-01` | `1.125e+00` | `1.061` |
| `0.25` | `2.253e-01` | `4.439e+00` | `2.107` |
| `0.5` | `8.017e-02` | `1.247e+01` | `3.532` |
| `1.0` | `3.972e-02` | `2.518e+01` | `5.018` |
| `2.0` | `1.986e-02` | `5.035e+01` | `7.096` |
| `4.0` | `9.930e-03` | `1.007e+02` | `10.035` |
| `8.0` | `4.954e-03` | `2.019e+02` | `14.208` |
| `16.0` | `2.256e-03` | `4.432e+02` | `21.052` |

**`1/I` spans `0.18` to `443`, a factor of `2462`, over ramp widths the bed
does not specify** — because BED-K's plant has no ramp. And the two natural
regularizations converge to opposite vacuous answers: `1/I → 0` as `w → 0`,
while §7.2's abrupt reading is `+∞`. No limit defines the bound. Whatever
number a "distance-to-CRB" column contained would be a statement about `w`.

### 7.4 And at `w ≪ 1` it is not even a function of `w` alone

At `w = 0.05`, sweeping only where `τ` falls *between* samples:

| `τ` | `1/I` |
|---|---|
| `32.0` | `1.800e-01` |
| `32.1` | `5.681e-01` |
| `32.25` | `1.146e+02` |
| `32.5` | `1.941e+06` |

A factor of `1.08e7` from sub-sample phase alone. Sub-sample phase is not a
property of the estimation problem; a floor that moves seven orders of
magnitude with it is not measuring the detector.

### 7.5 The cited prior art does not transfer

`CEQ_V15_2_DELTA.md` PRIOR ART lists *"Cramér–Rao bounds for delay
estimation"*. That bound is derived for a **known waveform** `s(t − τ)` in
noise, where `I(τ) = (2/N₀)∫|s′(t)|²dt` is the mean-square bandwidth of `s`.
This bed's plant is an **iid Gaussian drive**: `E[r] = 0` under every onset,
the entire signal lives in the second moment, and there is no `s′` to
integrate. The bandwidth form is not hard to evaluate here — it has no argument
to take. This is `MISTAKES.md` V-17, a threshold imported out of its units,
applied to a bound instead of a constant: the source is real, the derivation is
real, and neither fact makes it applicable.

> **RULING. `CEQ_V15_2_DELTA.md` instrument (d), *"Cramér–Rao localization
> floor, per bed, printed beside every onset CI"*, is NOT ACHIEVABLE AS
> WRITTEN.** It is a contract-level finding, not a budget one: the bound does
> not exist for the parameter the bed has, and forcing one into existence makes
> its value a property of the forcing.

## 8. WHAT SHIPS INSTEAD — the Ziv–Zakai bound

Ziv–Zakai needs no differentiability: it is built from binary hypothesis tests
rather than from a score, and it is valid for a discrete parameter. Under a
uniform prior on `{0, …, n−1}`,

    MSE ≥ (1/n) Σ_{h=1}^{n−1} h (n − h) P_min(h)

`(n − h)` counts the pairs `(a, a+h)` fitting in the support, `1/n` is the
prior mass. **Not valley-filled** — the valley-filling refinement only tightens
the bound, so omitting it leaves a valid bound and errs in the safe direction
for a must-fire that asks whether an estimator ever falls *below* it.

`P_min(h)` is exact here, not simulated. Only the `h` samples in `[a, a+h)`
differ between the hypotheses — variance `v₁ = σ²+m²` under the earlier onset,
`v₀ = σ²` under the later — so the test reduces to `S = Σ rᵢ²` against a
threshold with `S/v ~ χ²_h`:

    c        = h·log(v₁/v₀)·v₀v₁/(v₁−v₀)
    P_min(h) = ½[ 1 − F_h(c/v₀) + F_h(c/v₁) ]

It does not depend on `a`, which is what makes the stationary form apply.

**Its own calibration, and it is exact rather than asymptotic.** At `m = 0` no
test beats a coin, `P_min ≡ ½`, and the sum must telescope to the prior
variance — the true MMSE when there is nothing to learn:

| `n` | ZZB at `m = 0` | `(n²−1)/12` | difference |
|---|---|---|---|
| 32 | `85.2500000000` | `85.2500000000` | `0.00e+00` |
| 64 | `341.2500000000` | `341.2500000000` | `0.00e+00` |
| 128 | `1365.2500000000` | `1365.2500000000` | `0.00e+00` |

A bound that missed this case would be wrong by a constant everywhere else too.

## 9. THE MUST-FIRE — the noise sweep against the curve

`n = 64`, `m = 1.0`, delay bed `d = 4`, oracle visible model, **16 000 runs per
point**, onset drawn uniformly from `{0, …, 63}` **per run** — the same prior
the bound is stated under. Scoring at one fixed onset while bounding under a
uniform prior is the mismatch that manufactures a spurious sub-bound point; it
is excluded by construction rather than checked for afterwards.

Two estimators. **Bayes** is the posterior mean under that prior with `σ, m`
known — the MMSE estimator, and therefore the tightest thing the bound can be
tested against: if *it* dips below, the bound is wrong. **Shewhart** is the
shipped detector, `ceq.x35.residual.detect`, at a threshold calibrated to
`α = 0.01` on 1000 disjoint no-plant runs, reporting the prior mean `31.5` on
runs where nothing exceeds threshold.

| `σ` | SNR `m²/σ²` | **ZZB** | √ZZB | MMSE (MC) | se | rms | **dist** | Shewhart MSE | rms | **dist** | no-call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.125 | 64 | 0.27 | 0.515 | 0.43 | 0.010 | 0.657 | **1.275** | 12.64 | 3.556 | **6.899** | 139/16000 |
| 0.25 | 16 | 0.98 | 0.989 | 1.73 | 0.044 | 1.314 | **1.329** | 34.63 | 5.884 | **5.952** | 437/16000 |
| 0.5 | 4 | 6.79 | 2.606 | 11.10 | 0.247 | 3.332 | **1.279** | 192.77 | 13.884 | **5.329** | 2472/16000 |
| 1 | 1 | 70.99 | 8.426 | 109.05 | 1.664 | 10.443 | **1.239** | 399.68 | 19.992 | **2.373** | 12566/16000 |
| 2 | 0.25 | 228.28 | 15.109 | 301.12 | 2.558 | 17.353 | **1.149** | 357.71 | 18.913 | **1.252** | 15522/16000 |
| 4 | 0.0625 | 309.51 | 17.593 | 336.65 | 2.438 | 18.348 | **1.043** | 344.15 | 18.551 | **1.054** | 15789/16000 |
| 8 | 0.0156 | 333.11 | 18.251 | 342.12 | 2.411 | 18.496 | **1.013** | 346.74 | 18.621 | **1.020** | 15823/16000 |
| 16 | 0.0039 | 339.20 | 18.417 | 338.11 | 2.401 | 18.388 | **0.998** | 342.06 | 18.495 | **1.004** | 15838/16000 |

`dist` is `rms / √ZZB` — the delta's *distance-to-floor* reporting, with the
floor being Ziv–Zakai and labelled as such.

**The estimator tracks the bound's shape.** ZZB rises monotonically over three
orders of magnitude (`0.27 → 339.20`) and saturates at the prior variance
`341.25`; the MMSE rises with it (`0.43 → 338.11`) and the ratio stays inside
`[0.998, 1.329]` across a 4096× span in SNR. The bound is loosest where it is
most informative — 1.28× at high SNR, where the discrete-parameter geometry the
ZZB cannot see costs it most — and asymptotically tight where the data run out.

**The shipped Shewhart detector is `6.9×` the floor at the lowest noise and
`1.00×` at the highest.** It is furthest from the floor exactly where the floor
carries the most information, and its distance is dominated by non-detection:
`139/16000` no-calls at `σ = 0.125` rising to `15838/16000` at `σ = 16`. That
is not a defect of the detector — a threshold calibrated to a 1% per-run
false-alarm rate is *supposed* to stay silent — it is the statement the delta
wanted: **the memoryless comparator is a factor of 6.9 above the
information-theoretic floor in the regime where localization is possible, and
the gap is a detection-rate gap rather than a localization gap.**

## 10. THE SUB-BOUND POINT — reported as an instrument failure, then resolved

> *"Report any sub-CRB point as an instrument failure, not as a result."*

**One point in the table is below the bound.** At `σ = 16`, MMSE `338.11` against
ZZB `339.20`, a deficit of `1.09` at `se = 2.401` — **`0.46 se` below**.

It is reported here before it is explained. Two independent tests resolve it as
Monte-Carlo noise, not as a wrong bound:

**Run-count test.** A real violation is a fixed deficit; MC noise shrinks as
`1/√runs` and changes sign. Same seed block, `σ = 16`:

| runs | ZZB | MMSE (MC) | deficit | se | margin |
|---|---|---|---|---|---|
| 4 000 | 339.2035 | 336.6331 | `+2.5704` | 4.834 | `−0.53 se` |
| 16 000 | 339.2035 | 338.1058 | `+1.0977` | 2.401 | `−0.46 se` |
| 64 000 | 339.2035 | 341.4389 | `−2.2354` | 1.203 | `+1.86 se` |

The deficit changes sign at 64 000 runs. For contrast, `σ = 8` is above the
bound at every run count and its margin *grows* in `se` as it should for a real
gap: `+1.90 → +3.73 → +6.01 se`.

**Analytic test, with no Monte Carlo at all.** As the data go uninformative the
posterior mean goes to the prior mean and the MMSE goes to the prior variance
`341.25`. The bound must therefore sit strictly below `341.25`, and does, by a
margin that shrinks but never changes sign: `+8.14` at `σ = 8`, `+2.05` at
`σ = 16`, `+0.13` at `σ = 64`. At `σ = 16` the true MMSE is `341.25` and the
bound is `339.20` — the bound holds; the MC estimate was low by `0.9 se`.

`tests/x35p/test_kk_crb.py::test_the_bound_is_still_below_the_mmse_where_it_is_nearly_tight`
is that analytic check as a regression, and
`::test_noise_sweep_never_falls_below_the_bound` carries the sweep with an
explicit `3·se` tolerance rather than an eyeballed one.

**What this costs, stated.** In the near-tight regime (`σ ≥ 8`) the sweep
cannot distinguish a correct bound from one that is a few units too high, at
16 000 runs. The regime is also the one where the bound is useless — everything
sits at the prior variance — so the sweep's power is lowest exactly where its
verdict matters least. §12 records that.

---

## 11. TDD — RED then GREEN

Tests written before either module existed.

```
$ python -m pytest tests/x35p/test_kk_crb.py -q
ImportError while importing test module '...\tests\x35p\test_kk_crb.py'.
tests\x35p\test_kk_crb.py:22: in <module>
    from ceq.x35p import kk, crb
E   ModuleNotFoundError: No module named 'ceq.x35p'
=========================== short test summary info ===========================
ERROR tests/x35p/test_kk_crb.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.64s
```

```
$ python -m pytest tests/x35p/test_kk_crb.py -q
.............................                                            [100%]
29 passed in 6.48s

$ python -m pytest tests/x35 -q          # sibling suite, unaffected
..............                                                           [100%]
14 passed in 3.09s
```

**Two assertions in the RED draft were guesses and were corrected before
implementation, not after.** `crbs[0] < 1.0` in the ramp test and
`round(fine, 3) == 0.0` in the bin-normalization test were written from a
plausible expectation rather than a derivation; both were replaced with
structurally-predicted forms (monotonicity and a `20×` span; the exact `1/√M`
scaling and an explicit elimination) before `crb.py` or the variant existed.
Recording this because tuning an assertion to a measured value after the fact
is the failure this file exists to repair, and the two edits are in the
transcript either way.

## 12. LIMITS

1. **`(c)` is demonstrated on generator kernels and planted synthetics, not on
   a learned one.** Nothing trains this round, so no learned kernel exists in
   scope. The delta's *"on every learned kernel"* is owed. `kernel_from_matrix`
   is the adapter that discharges it the moment one exists.
2. **`(c)` assumes a shift-invariant kernel** — a lag axis, one row of a
   Toeplitz operator. A learned attention operator that is not shift-invariant
   has a different `K[i, ·]` per row and needs a per-row reading with a stated
   aggregation; none is chosen here.
3. **The `−5` reading in §5 candidate 3 is one draw of the aliasing coin.**
   Whether the unpadded variant reads correctly depends on the ratio of
   anticipation to causal extent; §5 shows one kernel each way and does not
   characterise the boundary.
4. **The mechanism behind the struck `0.000` is narrowed to two families, not
   to one.** The discriminating experiment is stated in §5 and requires the
   struck source, which is not in this repository.
5. **`(d)`'s bound is Bayesian.** ZZB lower-bounds the Bayes risk under the
   stated uniform prior. It is not a bound on the error at a *particular*
   onset, and a detector optimised for one region of the support could sit
   below it there without contradiction.
6. **The sweep's power is lowest where the bound is nearly tight** (§10), and
   it is a single bed (`delay, d=4`) — justified because the oracle residual
   does not depend on the kernel (`ceq/x35/residual.py`, asserted in
   `tests/x35`), so a second kind would produce an identical row and read as
   coverage it is not.
7. **`σ = 0` is refused** by `pmin_binary` rather than handled: at zero
   observation noise the onset is exact and the bound is 0 trivially.
8. **Barankin was not computed.** It would tighten the high-SNR rows where the
   ZZB is loosest (`1.28×`), at the cost of a supremum over test points that
   this node did not need to reach a verdict.

## 13. RECOMMENDATION FOR (d)

Three options were on the table. Ranked, with costs:

1. **Ziv–Zakai, as built here. RECOMMENDED.** It exists for the parameter the
   bed actually has, it needs no change to the bed or the detector, its
   pairwise test has a closed form on this noise model, and it calibrates
   exactly against a known answer. Cost: it is Bayesian and prior-dependent
   (limit 5), and it is loose by ~1.28× at high SNR.
2. **Reparametrize the bed so the onset is continuous and the index is its
   floor.** This makes a genuine CRB exist, but §7.3–§7.4 are the measurement
   of what it costs: the bound becomes a function of the ramp width and of
   sub-sample phase, spanning `2462×` and `1.08e7×` respectively. **It changes
   the bed, and it buys a number whose value is set by the change.** Not
   recommended unless a physically-motivated ramp width is derivable from
   something other than convenience.
3. **The non-parametric substitute — the measured 95% onset-offset interval,
   which the X₃₅a node reports as `[0, 1]`.** Honest and cheap, and it should
   be kept *beside* the bound rather than instead of it: it says where the
   detector lands, and says nothing about whether that is the best any detector
   could do. §9's `dist` column is what it cannot supply.

**The contract text of (d) should read "localization floor" and name the bound,
not "Cramér–Rao floor".** As written it names a bound that does not exist for
this bed, and `CEQ_V15_2_DELTA.md`'s own kill — *any CRB number before its
must-fire is struck* — then binds every number that could ever be printed in
that column.
