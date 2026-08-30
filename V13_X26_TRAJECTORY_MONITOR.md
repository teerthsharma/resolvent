# V13 X26 — THE TRAJECTORY-STATE MONITOR

Round 13, contract item X₂₆. Paper derivation plus null-run calibration. Nothing
here is integrated into the training loop or the optimizer; that was never this
node's scope.

**Scope boundary.** X₂₆ is a *sequential change-detector over loss
trajectories*. The entropy-threshold fan-out detector `H_t − H̄_basin > τ_H` over
next-token entropy is a different instrument on a different signal, owned by the
Tier-12 node, and is neither derived nor restated here.

**Reproduction.** `python scripts/v13_cusum_arl_calib.py`. numpy + standard
library, fixed seeds, ~22 s. It exits non-zero on purpose: two pre-registered
checks fail, and those failures are the result.

---

## Verdict

The kill clause fires, on its second limb. The decision interval calibrates
cleanly against its target — `h = 5.7566` at `k = 0.5` delivers a simulated
ARL₀ of `999.9 ± 9.1` against a target of 1000, and Siegmund's closed form
agrees to 0.71% — but that calibration holds only on an i.i.d. null that the
loss-trajectory residual never supplies. On the shipped pipeline the theory
threshold overshoots its false-alarm budget by `15.1×` (realized ARL₀ 1459
against a nominal 22000, a 53.0% chance of at least one false alarm per run),
and closing that gap by calibrating `h` on the pipeline itself inflates the
decision interval `4.49×` while the memoryless comparator's limit inflates only
`1.08×`. That asymmetry is the mechanism: dependence taxes an integrating
detector far harder than a threshold, and it consumes exactly the early-warning
margin the item was proposing to sell. Against an ARL₀-matched comparator the
lead time is `−6.00` abscissa units with a bootstrap CI of `[−13.88, +1.50]` and
3/8 seeds positive, so the CI does not exclude zero and the sign test cannot
reach its floor. X₂₆ closes: the monitor is a plot and not a feature, and the
topologized-AdamW arms do not gain a declared trigger from it.

---

## 1. The statistic

Two training arms are compared at equal values of a shared abscissa `u` — tokens
seen, not step index — because the arms log at different cadences and different
batch sizes, so equal index is not equal progress. On the bed of §5 arm A logs
every step and arm B every third step at a 1.07× batch; the index-matched gap
averages `24.07 σ_noise` against the matched-abscissa gap's `1.32 σ_noise`, so
index-matching manufactures a gap 18× the real one out of nothing but bookkeeping.

**(1)** `ε(u) = L_A(u) − L̃_B(u)`, where `L̃_B` is arm B linearly interpolated
onto arm A's abscissa grid over the overlapping span.

The seed confidence interval on `ε` is the usual across-seed interval at fixed
`u`; it is reported for display only. The monitor consumes the *per-seed*
residual, never the seed mean, because a change detected in the mean of eight
arms is not a change any one arm could have acted on.

**Assumption, used at (1) and nowhere relaxed later:** the two arms share an
abscissa on which their losses are comparable. Where the arms differ in
tokenizer, sequence length, or data order this is false and `ε` is not a gap.

---

## 2. CUSUM, stated exactly

For a monitored statistic `x_t` with in-control mean `μ₀`, reference value
(slack) `k`, and decision interval `h`:

**(2)** `S⁺_t = max(0, S⁺_{t−1} + (x_t − μ₀) − k)`, `S⁺_0 = 0`

**(3)** `S⁻_t = max(0, S⁻_{t−1} − (x_t − μ₀) − k)`, `S⁻_0 = 0`

**(4)** alarm at `τ = min{ t : max(S⁺_t, S⁻_t) ≥ h }`

`k` is conventionally half the smallest shift worth detecting, in null standard
deviations; `k = 0.5` targets a 1σ shift and is the value the contract fixes.

### What `x_t` is

`x_t` is **not** the raw loss and **not** the raw gap. A monotonically
decreasing loss has a non-zero mean increment at every `t`, so (2) would ramp to
`h` under the null and the ARL would be a property of the learning-rate schedule
rather than of any change. `ε(u)` cancels the bulk of that decrease but leaves a
slow common-mode drift. The monitored statistic is therefore the residual of `ε`
against its own causal local trend, standardized and pre-whitened:

**(5)** `r_t = ε_t − ŵ · (ε_{t−G−W}, …, ε_{t−G−1})`

where `ŵ` is the fixed weight vector of an ordinary-least-squares line fitted on
a trailing window of length `W` and extrapolated to offset 0. The window stops
`G` points short of `t` — a **guard gap** — so that an incipient change cannot
enter its own baseline.

**(6)** `ρ_t = (r_t − μ̂₀) / σ̂₀`, with `μ̂₀`, `σ̂₀` estimated on a burn-in
segment that ends before any planted change.

**(7)** `x_t = (ρ_t − φ̂ ρ_{t−1}) / √(1 − φ̂²)`, with `φ̂` the lag-1
autocorrelation of `ρ` on the same burn-in.

After (7) the in-control mean is zero by construction, so `μ₀ = 0` in (2)–(3).

### The guard gap is not free

Equation (5) is a high-pass filter. If `G` is short relative to the duration of
the change, the trailing baseline climbs onto the transition and cancels it.
Measured on a noiseless unit-amplitude logistic transition of width 40:

| `W` | `G` | `G` / width | peak residual ÷ amplitude |
|---:|---:|---:|---:|
| 120 | 20 | 0.50 | **0.130** |
| 120 | 150 | 3.75 | 0.646 |
| 300 | 150 | 3.75 | 0.768 |
| 300 | 300 | 7.50 | 0.954 |

At `G`/width = 0.5 the detrender destroys 87% of the very signal the monitor
exists to catch. **The design constraint is `G ≫ ` transition duration**, and it
is the first thing to get wrong: an earlier revision of this node ran `W=120,
G=20` and lost 5 of 8 planted seeds to a detrender that had eaten the
transition, with no diagnostic pointing at the cause. The shipped geometry is
`W = 300, G = 150`, passing 76.8%.

---

## 3. The ARL relationship

Write `b = h + 1.166`, where `1.166 = 2 × 0.5826` is the expected overshoot of a
Brownian motion past a flat barrier. For one arm under a mean shift `δ` (null sd
units), with `Δ = δ − k`, the Siegmund approximation is

**(8)** `ARL(δ) = [ exp(−2Δb) + 2Δb − 1 ] / (2Δ²)`, and `ARL = b²` at `Δ = 0`.

In control, `δ = 0` so `Δ = −k`, and for the two-sided chart the two arms'
alarm hazards add:

**(9)** `ARL₀ = ARL(0) / 2 = [ exp(2kb) − 2kb − 1 ] / (4k²)`, which at `k = 0.5`
reduces to `ARL₀ = e^b − b − 1`.

Out of control, (8) with `Δ = δ − k > 0` gives the run length that *is* the lead
time; for `Δb` large it degenerates to the drift limit

**(10)** `ARL₁(δ) ≈ b / (δ − k) = (h + 1.166) / (δ − k)`.

**Assumptions of (8)–(10), all four load-bearing:** `x_t` is i.i.d.; `x_t` is
Gaussian; the barrier crossing is well approximated by a Brownian motion with
drift, i.e. the per-step increment is small relative to `h`; and `μ₀`, `σ₀` are
known rather than estimated.

**Where it degrades.** The Brownian overshoot constant is an asymptotic in
small increments, so (8) loses accuracy as `k` and `|δ|` grow (the discrete
overshoot stops being Brownian) and as `h` shrinks toward `k` (too few steps to
average). It degrades catastrophically — not gracefully — under positive
dependence, which is §4. Estimating `μ̂₀`, `σ̂₀` from a finite burn-in adds a
further inflation of the run-length variance not captured anywhere in (8).

**Measured, on the i.i.d. null the formula assumes** (12000 replicates,
censoring horizon 12000, fresh verification stream):

| quantity | value |
|---|---|
| `h` from (9) at `k=0.5`, ARL₀ = 1000 | `5.7496` |
| `h` from simulation, CRN bisection | **`5.7566`** |
| achieved ARL₀ at that `h` | `999.9 ± 9.1` (1 se), 0.000% censored |
| pre-registered tolerance | `±51.5` (4 sd of calibration + verification) |
| Siegmund ARL₀ at the simulated `h` | `1007.1`, **0.71%** relative error |
| ARL₁ at `δ=0.75` | simulated `19.99`, Siegmund `19.94`, 0.3% |
| ARL₁ at `δ=1.00` | simulated `11.88`, Siegmund `11.85`, 0.3% |
| ARL₁ at `δ=1.50` | simulated `6.51`, Siegmund `6.42`, 1.3% |

On its own terms the closed form is excellent. Every one of these checks passes.
That is precisely what makes it a trap.

---

## 4. Autocorrelation, and the number worth naming

Hold `h = 5.7566` fixed and hold the null's **marginal** variance at 1, so that
the only thing varying is dependence. With an AR(1) null
`x_t = φ x_{t−1} + √(1−φ²) e_t`:

| `φ` | LRV/γ₀ = (1+φ)/(1−φ) | ARL₀ achieved | vs nominal 1000 | P(false alarm) over H=1100 |
|---:|---:|---:|---:|---:|
| 0.000 | 1.00 | `1014.5 ± 9.0` | 1.015× | 66.2% |
| 0.300 | 1.86 | `149.1 ± 1.3` | 0.149× | 99.9% |
| 0.500 | 3.00 | `70.4 ± 0.6` | 0.070× | 100.0% |
| **0.709** | 5.87 | **`41.5 ± 0.3`** | **0.041×** | 100.0% |
| 0.800 | 9.00 | `35.5 ± 0.3` | 0.035× | 100.0% |

`φ = 0.709` is not a chosen value; it is `φ̂` measured on the burn-in of the
shipped pipeline, averaged over 2500 null beds (sd 0.059). **A monitor advertising
ARL₀ = 1000 delivers 41.5 — a 24.1× shortfall — at the autocorrelation this
pipeline actually produces.** That is the single number this node most needed to
name. Even at a mild `φ = 0.3` the shortfall is 6.7×.

Note also the first row: at `φ = 0` and ARL₀ = 1000, a single 1100-point training
run still carries a 66.2% chance of at least one false alarm. **ARL₀ is not a
per-run false-alarm probability**, and a target of 1000 is not deployable on a
run of comparable length. The deployable target used from §5 onward is
`ARL₀ = 20H`, giving a 4.88% per-run budget.

### The two textbook corrections, and why they are not enough

Pre-whitening by (7) and inflating the standard deviation by
`√((1+φ)/(1−φ))` both restore ARL₀ on a *true* AR(1), and both cost the same
first-order factor in `δ`: at `φ̂ = 0.709` the signal is attenuated to
`0.413×`, so `δ` more than halves and (10) more than doubles the lead. Measured:
pre-whitening returns ARL₀ `1000.3`; variance inflation by `2.422×` returns
`8183.8`, over-correcting, because the asymptotic long-run variance factor
`(1+φ)/(1−φ) = 5.87` describes an infinite sum, whereas a CUSUM that resets at
zero averages over an effective window of order `h/k ≈ 11` steps, where the
Bartlett factor is only ≈ 4.5.

**Neither correction rescues the real pipeline, because the pipeline's residual
is not AR(1).** After (7) its lag-1 autocorrelation is `−0.0999` — small, the
standard diagnostic passes — while its Bartlett long-run variance ratio keeps
climbing with bandwidth:

**(11)** `LRV(x)/γ₀ = 1 + 2 Σ_{l=1}^{L} (1 − l/(L+1)) γ_l / γ₀`

| bandwidth `L` | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|
| LRV/γ₀ | 1.20 | 1.82 | 2.52 | 2.78 |

**Whitening at lag 1 is not whitening.** A cumulative statistic integrates the
spectral density at zero frequency, not the lag-1 coefficient, and the sliding
window of (5) writes correlation into the residual at a range of order `W + G`
— longer, here, than the burn-in available to measure it.

---

## 5. The planted bed and the lead-time statistic

**(12)** the bed. Arm A: `L_A(u) = 1.20 + 1.8 e^{−u/300} + η_A(u)`. Arm B:
`L_B(u) = 1.18 + 1.8 e^{−u/330} − A·σ(u; u*, w) + η_B(u)`, with
`σ(u; u*, w) = [1 + e^{−(u−u*)/w}]^{−1}`. Both `η` are AR(1) with coefficient
0.35 and per-point sd `σ_noise = 0.02` nats. Amplitude `A = 12 σ_noise = 0.24`
nats, `u* = 1400`, `w = 40`, over a 2000-point run: a long flat plateau in the
gap followed by an abrupt late improvement in one arm, which is the grokking
shape the item names. Burn-in `[450, 900)`, monitored horizon `H = 1100`.

**(13)** the transition point. `u*` is known by construction; it is not the
operational quantity, because a monitor that beats a *known* changepoint has
proved nothing.

**(14)** visibility. The operational comparator is the point at which a reader
of the curve would call the transition — formalized as a **Shewhart individuals
chart** on the same standardized residual `ρ_t`, alarming at
`min{ t : |ρ_t| > L }`. This is the textbook memoryless baseline, and CUSUM's
entire published advantage is that it beats it on small persistent shifts.

**(15)** `L` and `h` are set to the **same** ARL₀. This is the step that decides
whether the result means anything. A comparator with no false-alarm budget of
its own can be pushed arbitrarily late — a naive 5σ rule realizes a per-run
false-alarm rate of 0.80% on this residual against the 4.88% the CUSUM is
allowed, and reporting a lead against it would be reporting the threshold
difference, not a detector difference.

**(16)** `Lead_s = τ^Shewhart_s − τ^CUSUM_s` for seed `s`, positive when the
CUSUM fires first.

**(17)** the estimator and its interval. `Lead` is a random variable over seeds.
The point estimate is the seed mean; the interval is a percentile bootstrap over
the 8 seed values (B = 20000), reported beside a Student-`t` interval and the
distribution-free order-statistic interval `[L_(2), L_(7)]`, whose exact
coverage for the median at `N = 8` is `1 − 2·(C(8,0)+C(8,1))/2⁸ = 0.9297`. The
accompanying sign test has a two-sided floor of `2·(1/2)⁸ = 0.0078125`,
reachable only at 8/8 or 0/8 — this project's N=8 convention.

**(18)** the pipeline calibration. Rather than trusting (8)–(9), `h` and `L` are
bisected directly on 2500 realized null beds until each hits the target per-run
false-alarm probability. This makes no independence assumption at all.

---

## 6. Measured results

Target `ARL₀ = 20H = 22000`, per-run false-alarm budget 4.88%.

| detector | threshold from theory | from pipeline | inflation |
|---|---:|---:|---:|
| CUSUM `k=0.5`, `h` | `8.833` | `39.683` | **`4.49×`** |
| Shewhart, `L` | `4.078` | `4.420` | **`1.08×`** |

Realized null false-alarm rates at the pipeline-calibrated thresholds: CUSUM
4.88%, Shewhart 4.84%, over 2500 beds. Using the *theory* value `h = 8.833` on
the real pipeline instead yields a realized per-run false-alarm probability of
**53.0%** against a nominal 4.88% — an implied ARL₀ of `1459` against `22000`,
**short by 15.1×**.

**Must-fires.** Planted bed alarms on **8/8** seeds. No-change bed alarms on
**0/8** seeds, against a pre-registered bound of ≤2 (`P(≥3) = 0.0060` at a
4.88% per-seed rate). Both pass.

**Lead time**, at the pipeline-calibrated thresholds:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `τ^CUSUM` | 1414 | 1413 | 1435 | 1386 | 1427 | 1396 | 1407 | 1418 |
| `τ^Shewhart` | 1404 | 1386 | 1420 | 1397 | 1418 | 1399 | 1403 | 1421 |
| lead | −10 | −27 | −15 | +11 | −9 | +3 | −4 | +3 |

mean `−6.00`, median `−6.50`, sd `11.96`.

| interval | value |
|---|---|
| percentile bootstrap 95% CI of the mean, B=20000 | **`[−13.88, +1.50]`** |
| Student `t` 95% CI of the mean | `[−16.00, +4.00]` |
| order-statistic 92.97% CI of the median | `[−15.00, +3.00]` |
| sign test | 3/8 positive; floor `0.0078125` unreachable |

**The CI does not exclude zero.** The kill clause fires.

### Sensitivity to transition width (thresholds held fixed)

| width `w` | `G`/width | both fire | median lead | min | max |
|---:|---:|---:|---:|---:|---:|
| 40 | 3.75 | 8/8 | −6 | −27 | +11 |
| 80 | 1.88 | 8/8 | **+56** | −40 | +268 |
| 120 | 1.25 | 6/8 | +11 | −50 | +121 |
| 200 | 0.75 | 2/8 | −74 | −141 | −6 |

There is a band — `w ≈ 80`, where the ramp is slow enough for the CUSUM's
integration to pay yet fast enough for the guard gap to pass it — in which the
lead is solidly positive. **This was not pre-registered and is not claimed.**
It is recorded because it names the only region where a future version of this
item could live, and because the min column shows the lead is not even
sign-stable there across 8 seeds.

---

## 7. Failed checks

Two pre-registered checks fail. Neither tolerance was relaxed.

**C1, null drift.** The monitored null mean of `x_t` is `−0.2153` sd, against a
pre-registered bound of 0.20. This is systematic, not sampling noise: it is a
mean over 2500 beds × 1100 points, and its sign is consistent with a
local-linear extrapolator overshooting a concave trend. It consumes **43% of the
`k = 0.5` slack** before any change occurs, biasing the two arms of (2)–(3)
against each other. `k` absorbs it here rather than alarming, but a `k` chosen
for detection sensitivity is not a free budget for detrender bias.

**C2, the kill clause.** Recorded in §6.

---

## 8. What would have to change

Recorded so the item's closure is a decision rather than a dead end. None of
this is proposed for the current round.

1. Calibrate every threshold on realized null beds, never from (8)–(9). The
   closed form is off by 4.49× in `h` on this pipeline and gives no warning.
2. Report the Bartlett long-run variance ratio (11) beside any residual
   autocorrelation figure. Lag-1 read `−0.0999` while the monitor was broken.
3. Fix the detrender's curvature bias (C1) before tuning anything downstream.
4. Establish the CUSUM-over-Shewhart advantage on the actual signal *before*
   building on it. It is a theorem about persistent mean shifts in i.i.d. noise,
   and a grokking transition in an autocorrelated gap is neither.

---

## Limits

- The bed is synthetic. `ε` is built from two exponential loss curves plus AR(1)
  noise; no CEQ training run was used, and the measured `φ̂ = 0.709` is a
  property of this generator and of the interpolation in (1), not of any
  observed CEQ run. Every number above is conditional on that generator.
- `N = 8` seeds. The finest two-sided sign-test p is `0.0078125`, and the
  bootstrap CI of §6 is a resample of eight atoms — a coarse lattice, and per
  this project's M-4, an interval over seeds, not over draws within a seed.
- The comparator of (14) is a Shewhart chart on the same residual, not a human
  reading a plot. It is the fair statistical baseline; it is not literally "the
  raw curve", and a reader with the whole finished curve in view has
  non-causal information neither detector has.
- After the transition passes the guard window, (5) produces a spurious
  opposite-sign excursion as the trailing baseline climbs onto the new level.
  Only the *first* crossing is interpretable; the sign of the alarming arm is not.
- ARL₁ from (10) is derived for a step shift; the bed's transition is a logistic
  ramp, so (10) bounds rather than predicts the alarm time.
- The `w ≈ 80` positive-lead band in §6 is post-hoc.
- The variance-inflation correction of §4 over-corrects by construction; the
  effective-window argument given there is an explanation of the measured `8183.8`,
  not an independently verified claim.
- Censoring is negligible for the §3 numbers (0.000% at the calibrated `h`) but
  the §4 rows at high `φ` were not separately checked for it.
- Nothing here was run against the topologized-AdamW arms. The item's proposed
  payoff — a pre-registered, alarm-triggered response for those arms — is not
  delivered, because the alarm has no demonstrated lead to trigger on.
