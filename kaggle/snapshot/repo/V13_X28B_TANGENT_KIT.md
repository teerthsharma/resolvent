# V13_X28B_TANGENT_KIT.md — the tangent kit

Instruments for measuring sensitivity and chaos on planted test beds, each
derived before it is coded. Evidence classes follow `MATHEMATICS.md`: `RUN`
means executed this session and the output read, `DERIVED` means it follows
from steps written out below, `CITED` means an external result is used and
named as external.

Every numbered equation `(K*)` that can be checked at finite precision is
checked by `scripts/v13_tangent_kit.py`, whose assertion messages carry the
same numbers. `RUN`, this session: **exit 0, every assertion passed**, 77.7 s wall
clock, numpy 1.26.4, torch 2.5.1+cu121, python 3.11.9, float64,
single-threaded, seed 28.

**Both must-fires fire.**

| gate | measured | closed form | absolute error |
|---|---|---|---|
| MF-1 logistic `r = 4` | `λ̂ = 0.693147181171` | `ln 2 = 0.693147180559945` | `6.1140e-10` |
| MF-2 logistic `r = 3.2` | `λ̂ = −0.916290731874071` | `ln 0.4 = −0.916290731874155` | `8.8818e-14` |
| MF-3 Hénon `a=1.4 b=0.3` | `λ̂₁ + λ̂₂ = −1.203972804326` | `ln 0.3 = −1.203972804325936` | `2.2204e-16` |

The negative control's sign is negative — the instrument can return "no chaos
here". The adjoint gradcheck **passes at `rtol = 1e-4`**; the tolerance was not
loosened and no fallback was taken. Training through the dynamics is licensed on
the field, dimension and horizon tested, and on nothing wider (§6). **No
downstream number is voided by this kit.**

Three findings were recorded rather than tuned around, each of which withdraws
credit the kit would otherwise have been given:

1. **The `r = 4` positive control cannot test the estimator's convergence
   (§3.2).** An assertion predicting `N^{−1/2}` fired at a measured `−0.9927`.
   The prediction was wrong, not the instrument: `ln|f'|` on that map is `ln 2`
   plus an exact coboundary, so the Birkhoff sum telescopes and the error falls
   as `N^{−1}`. The standard positive control is anomalously easy *because* it
   is smoothly conjugate to a linear map. The generic `N^{−1/2}` rate had to be
   measured elsewhere — on Hénon, where it reads `−0.5079` (§3.4).
2. **The obvious seeding guard is not implementable in float64 (§3.1).** Every
   float64 `u` is a dyadic rational, hence an exact preimage of the fixed point;
   a correct exact-preimage test rejects every seed, and the first version hung.
3. **A finite-difference control run on a quadratic tests nothing (§1.4).** The
   logistic map has `D³f ≡ 0`, so the truncation half of the step-size model is
   unfalsifiable there; the model was moved to a bed that can falsify it.

The single most important result in this file is not an instrument. It is §5:
**the X₂₈c consistency triangle is not intrinsically circular, but it is not a
mutual certification either**, and under the two conditions that actually hold
in this repo it degenerates into an arithmetic identity with zero discriminating
power. The measurements establishing this are in §5.3–§5.6; the verdict is §5.7.

---

## 0. WHAT THIS LEANS ON, NAMED

| Existing object | Where | Used at |
|---|---|---|
| Benettin QR named as the method behind `λ₁ = +0.42084` on Hénon | `ceq/nonnormal.py:15`, `NOTES.md:163`, `tests/w2/test_w2_nonnormal.py:7` | (K11) cross-check |
| Tél's `D₁⁽¹⁾ = 1 − κ/λ` and the ordering constraint `λ > κ` | `V13_X27_G1_PRIOR_ART.md` E7 | §5 throughout |
| Kantz–Grassberger as the origin of that relation, full text still owed | `V13_X27_G1_PRIOR_ART.md` E8, U3 | §6 Limits |
| Non-hyperbolic saddles decay algebraically, not exponentially | `V13_X27_G1_PRIOR_ART.md` E10 | §6 Limits |
| `α = κ/λ` derived and uncited, flagged as collapsing two instruments into one | `V13_X27_G1_PRIOR_ART.md` U4 | §5.4 |
| The `assert`-based self-check with no framework and no fixtures | `scripts/v13_derivation_check.py` | the whole script |

Nothing in `results/` is read or written. The script prints and asserts only.

---

## 1. INSTRUMENT 1 — FORWARD SENSITIVITY BY JVP

### 1.1 The tangent recursion

For a discrete map `x_{n+1} = f(x_n)` on `R^d`, differentiate both sides with
respect to the initial condition along a direction `v`:

```
    (K1)   v_{n+1} = Df(x_n) v_n ,      v_0 = v
```

This is the variational equation of the map. For a parameterised field the same
object carries a parameter tangent as well, `Df` acting on `(δz, δθ)` jointly.
`torch.func.jvp` evaluates `Df(x)v` in one forward pass without forming `Df`.

### 1.2 The finite-difference control and its step size

The control is a **central** difference, not a forward one:

```
    (K2)   D_h f(x)[v] := ( f(x + hv) − f(x − hv) ) / (2h)
                        = Df(x)v + (h²/6) D³f(x)[v,v,v] + O(h⁴)
```

The `O(h)` term cancels by symmetry, so the truncation error is `O(h²)` rather
than the forward difference's `O(h)`. Against that runs the roundoff in the
numerator, `~2ε_m|f|`, divided by `2h`:

```
    (K3)   E(h) = C₃ h² / 6 + ε_m M₀ / h ,
           dE/dh = 0  ⇒  h* = ( 3 ε_m M₀ / C₃ )^{1/3} ,
           E(h*) = O( ε_m^{2/3} )
```

with `C₃ = |D³f|` and `M₀ = |f|`. With `M₀ ~ C₃ ~ O(1)` and float64
`ε_m = 2.220446049250313e-16`:

```
    h*  ≈ ε_m^{1/3}  = 6.055454452393343e-06
    floor ≈ ε_m^{2/3} = 3.6669e-11
```

**The step used is `h = ε_m^{1/3} · max(1, ‖x‖_∞)`**, the state-magnitude
scaling being the `M₀` factor of (K3) in the only form available without
estimating `D³f`. A forward difference would give `h* ~ ε_m^{1/2} ≈ 1.5e-8` and
a floor of `~ε_m^{1/2} ≈ 1.5e-8` — three and a half decades worse — which is
why the control is central.

### 1.3 Achieved errors, `RUN`

Measured, not asserted. The assertion in the script is a loose sanity floor at
`1e-7`; these are what was achieved against it.

| bed | tangent | `h` used | rel `|jvp − closed form|` | rel `|jvp − central difference|` |
|---|---|---|---|---|
| logistic `r=4`, one step | state | `6.055454e-06` | `0.0000e+00` | `1.6641e-11` |
| Hénon `a=1.4 b=0.3`, one step | state | `6.055454e-06` | `0.0000e+00` | `5.6420e-12` |
| `tanh(Wz+b)`, `d=4` | state | `1.050333e-05` | `0.0000e+00` | `3.2915e-11` |
| `tanh(Wz+b)`, `d=4` | `(z, W, b)` jointly | — | `0.0000e+00` | — |

`torch.func.jvp` agrees with the closed-form Jacobian-vector product to
**exactly zero** in every case — bit-identical, not merely within tolerance.
The disagreements with the finite difference are all within a factor of two of
the (K3) floor `3.67e-11`, which is the finite difference being wrong, not the
JVP.

### 1.4 Finding: the logistic map has no truncation error

`RUN`. The step-size sweep was first run on the logistic map and produced no
minimum at all — the error fell monotonically all the way to `h = 1e-2`:

```
    logistic r=4:   h = 1e-08  rel err = 7.9990e-09
                    h = 1e-02  rel err = 1.9215e-15
```

The cause is (K2) read literally: **the logistic map is a quadratic, so
`D³f ≡ 0` and the truncation term is identically zero.** Its central difference
is exact up to roundoff, the error is `ε_m/h` alone, and (K3)'s optimum does not
exist. This is recorded rather than tuned around: **a finite-difference control
run only on a low-degree polynomial bed does not test the truncation half of
the error model at all,** and the derived `h*` is unfalsifiable there.

The V-curve was therefore measured on `tanh(Wz+b)`, where `D³f ≠ 0`:

| `h` | rel err | `h` | rel err |
|---|---|---|---|
| `1e-12` | `5.7170e-05` | `1e-06` | `3.8103e-11` |
| `1e-11` | `1.1793e-05` | **`1e-05`** | **`2.8397e-11`** |
| `1e-10` | `4.7078e-07` | `1e-04` | `3.2581e-09` |
| `1e-09` | `3.8662e-08` | `1e-03` | `3.2577e-07` |
| `1e-08` | `4.9494e-09` | `1e-02` | `3.2573e-05` |
| `1e-07` | `3.7065e-10` | | |

Measured minimum at `h = 1e-05` with rel err `2.8397e-11`, against the derived
`h* = 6.055e-06` and the derived floor `3.6669e-11`. The left branch falls as
`h⁻¹` and the right rises as `h²`, both as (K3) states. **The step-size model is
confirmed on the bed that can falsify it.**

### 1.5 The composed tangent

```
    (K4)   Df^n(x₀) = Df(x_{n−1}) ⋯ Df(x₀)
```

`RUN`, `n = 8` on the logistic map, amplification `2⁸ = 256`, measured
`|Df⁸ v| = 2.592625e+02`:

- `rel |composed jvp − iterated (K1) tangent| = 0.0000e+00` — the tangent
  recursion **is** the chain rule, bit for bit.
- `rel |composed jvp − central difference| = 1.7170e-06`. The composed map is a
  degree-`2⁸` polynomial, so its `C₃` is enormous and the (K3) truncation term
  dominates. The finite-difference control degrades under composition; the JVP
  does not.

---

## 2. INSTRUMENT 2 — LYAPUNOV SPECTRUM BY BENETTIN QR

### 2.1 Derivation

Let `J_n := Df(x_n)` and `A_N := J_N J_{N−1} ⋯ J_1`. Naively iterating (K1) on
`d` independent vectors fails: every vector aligns with the leading Oseledets
direction within a few tens of steps and the frame loses rank in float64. The
Benettin construction reorthonormalises at every step.

Start from an orthonormal frame `Q₀ = I` (or its first `k` columns). At step `n`:

```
    (K5)   M_n = J_n Q_{n−1} ,      M_n = Q_n R_n
```

with the QR factorisation made **unique** by forcing `diag(R_n) > 0`: for
`S = diag(sign(diag R))`, `S² = I`, so `M = (Q S)(S R)` and replacing
`Q ← Q S`, `R ← S R` leaves the factorisation valid with a positive diagonal.
Without this step `log` of the diagonal is complex on roughly half the
iterations and the accumulator is nonsense — LAPACK makes no sign promise.

Unrolling (K5):

```
    (K6)   A_N Q₀ = J_N (J_{N−1} ⋯ J_1 Q₀)
                  = J_N Q_{N−1} R_{N−1} ⋯ R_1
                  = Q_N R_N R_{N−1} ⋯ R_1
```

A product of upper-triangular matrices is upper triangular with diagonal equal
to the product of the diagonals, so the `i`-th diagonal entry of
`R_N ⋯ R_1` is `∏_n (R_n)_ii`. By Oseledets the singular values of `A_N` grow
as `e^{Nλ_i}`, and the `i`-th column of the transported frame spans the `i`-th
subspace of the Oseledets flag transverse to the previous `i−1`, so its growth
factor is exactly that diagonal entry. Hence, for a unit time step,

```
    (K7)   λ_i = lim_{N→∞} (1/N) Σ_{n=1}^{N} ln (R_n)_ii
```

and for a flow, divide by `N·Δt` instead of `N`.

**Convergence rate, and the trap in it.** Write `ψ := ln (R_n)_ii`. Then

```
    (K8)   λ̂_i(N) − λ_i  =  (1/N) Σ_{n<N} [ ψ(x_n) − λ_i ]
```

and the rate depends on which of two cases `ψ` falls into.

*Generic.* If `ψ − λ` is not a coboundary and the map is mixing with a
non-degenerate CLT variance `σ²`, the Birkhoff sum is `O(√N)`:

```
    (K8a)  | λ̂_i(N) − λ_i |  ~  σ_i N^{−1/2}
```

*Coboundary.* If `ψ = λ + (φ∘f − φ)` for some `φ`, the sum **telescopes**, its
CLT variance is exactly zero, and only a boundary term survives:

```
    (K8b)  λ̂_i(N) − λ_i  =  ( φ(x_N) − φ(x_0) ) / N   ~   N^{−1}
```

The two differ by half a decade of accuracy per decade of iterations, and §3.2
shows that **the standard positive control is the second case, not the first.**

A single-step variance calculation does not distinguish them — the observable's
own variance can be large while its CLT variance is zero. For the logistic map
at `r = 4`, the conjugate coordinate `x = sin²(πu)` of (K9) gives
`f'(x) = 4 − 8x = 4 cos(2πu)`, so with `θ := 2πu` the observable is
`ln|f'| = ln 4 + ln|cos θ|`, `θ` uniform, mean `ln 4 − ln 2 = ln 2`, and

```
    Var[ ln|cos θ| ]  =  ∫₀^π (ln|cos θ|)² dθ/π  −  (ln 2)²  =  π²/12
```

using `∫₀^{π/2} ln²(cos θ) dθ = (π/2)(ln²2 + π²/12)`, so `σ = 0.9069`. The
`log²` singularity at `θ = π/2` is integrable, so this variance is finite —
**and it is nevertheless the wrong number to predict the rate from.** §3.2.

An exact conservation law that (K7) must respect, used as the `d > 1` gate in
§3.4: since `det(Q R) = det R = ∏_i R_ii` and `Q_n` is orthogonal,

```
    Σ_i λ̂_i  =  (1/N) Σ_n ln |det J_n|
```

**exactly at every `N`**, with no statistical error whatsoever. Any map with a
constant Jacobian determinant therefore pins the sum of the measured exponents
to a closed form independent of iteration count.

---

## 3. THE MUST-FIRES

### 3.1 Positive control — logistic `r = 4`, closed form

The substitution `x = sin²(πu)` conjugates `x ↦ 4x(1−x)` to the doubling map
`u ↦ 2u mod 1`:

```
    4 sin²(πu)(1 − sin²(πu)) = 4 sin²(πu) cos²(πu) = sin²(2πu)
```

The conjugacy is smooth and invertible off a measure-zero set, and the doubling
map has `|du'/du| = 2` at every point, so

```
    (K9)   λ = ln 2 = 0.6931471805599453      exactly.
```

**The seeding pitfall.** In the conjugate coordinate the preimages of the fixed
point `0` are exactly the **dyadic rationals** `k/2^m`. `x₀ = 0.5` is `u = 1/4`,
which maps to `1`, then to `0`, and stays; every `x₀ = sin²(πk/2^m)` collapses
after `m` steps, and `ln|f'|` at the collapse point is `−∞`.

**Finding: the obvious guard is not implementable in float64.** The first
version of the seeder rejected any `u` whose doubling images came within
`2⁻⁴⁰` of an integer. It rejected every seed ever offered and the run hung.
The reason is exact: **every float64 `u` is itself a dyadic rational**, so
doubling it 53 times shifts the last mantissa bit out and yields exactly `0.0`.
An exact-preimage test therefore classifies the entire float64 grid as a
preimage of the fixed point — which, in exact arithmetic, it is. There is no
float64 seed that is *not* an exact preimage of `0`, and this is the same fact
as "float64 orbits are not true orbits" (§6) seen from the seeding end.

**The guard that is implementable** is on the computed orbit, which is the only
orbit the run has. Two layers, both reported:

1. `x₀ = sin²(πu)`, `u ~ U(0.05, 0.95)`, rejected if the **computed** float64
   forward orbit reaches `0.0`, `1.0` or `0.5` within 1,000 steps. `RUN`:
   0 seeds rejected.
2. For the remaining 210,000 iterations, the QR step counts every
   non-positive `R` diagonal. An orbit landing on `x = 0.5`, where `f' = 0`,
   produces one. `RUN`: 0.

**Transient discarded: 10,000 iterations** before any accumulation begins. This
is `≈ 6,900` Lyapunov times (`1/λ = 1.443` iterations each) and far longer
than the arcsine density's mixing time; its purpose is to remove dependence on the seeding
distribution, not to reach an attractor — `r = 4` has no attractor to reach.

`RUN`, seeds `512` drawn with rng seed 28, `200,000` measured iterations
each:

```
    MEASURED   λ̂ = 0.693147181171
    PREDICTED  λ  = 0.693147180559945          (ln 2, closed form)
    absolute error = 6.1140e-10
    ensemble s.d. 6.7696e-06, standard error of the mean 2.9918e-07, error/s.e. = 0.00
```

The achieved error is roughly `500×` *below* the standard error of the mean,
which is structure rather than luck: by (K8b) each seed's error is
`(φ(u_N) − φ(u_0))/N`, and `φ(u_N)` and `φ(u_0)` are identically distributed
under the invariant measure, so the boundary terms cancel in the ensemble mean
on top of the `1/N` already gained. Same coboundary structure as §3.2, same
reason the control is not representative.

### 3.2 The measured convergence rate, and a recorded assertion failure

`RUN`, from the same run at no extra cost — the running estimate is snapshotted
at four checkpoints.

| `N` | mean `λ̂` | `|mean − ln 2|` | per-seed RMS error |
|---|---|---|---|
| 200 | `0.693589617` | `4.424e-04` | `6.3535e-03` |
| 2,000 | `0.693144124` | `3.057e-06` | `6.7146e-04` |
| 20,000 | `0.693149218` | `2.037e-06` | `6.5909e-05` |
| 200,000 | `0.693147181` | `6.114e-10` | `6.7630e-06` |

**Fitted slope of `log(RMS error)` against `log N`: `−0.9927`.**

**An assertion fired here and was not weakened.** The first version of the
script predicted (K8a) from the finite single-step variance `π²/12` and
asserted `−0.62 < slope < −0.38`. The measured slope was `−0.9927` and the
assertion failed. **The prediction was wrong, not the instrument**, and the
reason is exact:

```
    (K9a)  ln|f'(h(u))| = ln 2 + φ(2u mod 1) − φ(u) ,
           h(u) = sin²(πu) ,      φ(u) := ln|h'(u)| = ln|π sin(2πu)|
```

`h` is the (K9) conjugacy. **`ln|f'|` on this map is `ln 2` plus an exact
coboundary**, so the Birkhoff sum telescopes and (K8b), not (K8a), governs:

```
    λ̂(N) − ln 2 = ( ln|h'(u_N)| − ln|h'(u_0)| ) / N   ~   N^{−1}
```

`RUN`, the identity checked at 499 points spanning the interval: max deviation
`9.5923e-14`, which is roundoff. The corrected assertion tests `−1`, and the script
now asserts the coboundary identity itself, so the `N^{−1}` rate has a stated
cause rather than an observed coincidence.

**The consequence is the substantive part.** The logistic map at `r = 4` is
smoothly conjugate to a linear map — which is *why* its exponent is exactly
`ln 2`, and is the same fact that makes its convergence anomalously fast.
**The standard positive control for a Lyapunov instrument does not exercise the
instrument's generic convergence at all.** Reading `|λ̂ − ln 2| = 6.1140e-10` at
`N = 200,000` and concluding that the instrument resolves exponents to that
accuracy on a real system would be wrong by orders of magnitude. The generic
branch (K8a) has to be measured on a map with no smooth linearisation, which is
§3.4.

### 3.3 Negative control — logistic `r = 3.2`, closed form

The period-2 orbit of `x ↦ r x(1−x)` has points
`p± = (r + 1 ± √((r−3)(r+1)))/(2r)` and multiplier

```
    (K10)  f'(p₊) f'(p₋) = 4 + 2r − r² ,      λ = ½ ln|4 + 2r − r²|
```

At `r = 3.2`: `4 + 6.4 − 10.24 = 0.16 = 0.4²`, so

```
    λ = ½ ln 0.16 = ln 0.4 = −0.9162907318741551      exactly.
```

`|4 + 2r − r²| = 0.16 < 1`, so the 2-cycle is attracting and **the exponent is
negative by construction** — `r = 3.2` sits inside the period-2 window, which is
what makes this a control rather than a coincidence.

`RUN`, 8 seeds `x₀ ~ U(0.2, 0.8)`, rng seed 29, transient 200,000,
measured 20,000:

```
    MEASURED   λ̂ = -0.916290731874071      (spread across seeds `2.220e-16`)
    PREDICTED  λ  = -0.916290731874155
    absolute error = 8.8818e-14
```

**Negative, and to 13 figures.** A kit that only fires positive is not
validated; this is the reading that shows the instrument returns "no chaos
here" when there is none.

### 3.4 The `d > 1` closed form — Hénon

In `d = 1` the QR step of (K5) is a `1×1` factorisation and reduces to
accumulating `ln|f'|`; **neither control above exercises the frame at all.**
The Hénon map supplies the two-dimensional gate. For
`x_{n+1} = 1 − a x_n² + y_n`, `y_{n+1} = b x_n`,

```
    J = [[ −2a x , 1 ] , [ b , 0 ]] ,     det J = −b   at every point
```

so by the conservation law of §2.1,

```
    (K11)  λ₁ + λ₂ = ln|det J| = ln b = ln 0.3 = −1.2039728043259361
```

exactly, at every iteration count. `RUN`, `a = 1.4`, `b = 0.3`, 64 seeds,
rng seed 30, transient 10,000, measured 100,000:

```
    MEASURED   λ₁ = +0.419022323     λ₂ = -1.622995127     sum = -1.203972804326
    PREDICTED  sum = -1.203972804325936
    absolute error = 2.2204e-16
```

`λ₂ < 0`: True — a second negative reading, this one along a contracting
direction of a genuinely chaotic map. Cross-check against the `λ₁ = +0.42084`
figure carried in `ceq/nonnormal.py:15` and `NOTES.md:163`: deviation `0.00182`.

**The generic convergence rate — the measurement §3.2 could not make.** Hénon
is not smoothly conjugate to a linear map, so `ln R₁₁` is not a coboundary and
(K8a) governs. The statistic is the **across-seed standard deviation** of
`λ̂₁`, which requires no known truth to compute:

| `N` | mean `λ̂₁` | across-seed s.d. |
|---|---|---|
| 100 | `+0.422106` | `4.0573e-02` |
| 1,000 | `+0.421050` | `1.2438e-02` |
| 10,000 | `+0.419285` | `3.5357e-03` |
| 100,000 | `+0.419022` | `1.2508e-03` |

**Fitted slope: `−0.5079`**, against the `−1/2` of (K8a). The generic branch of
(K8) is confirmed on the bed that can test it, and the contrast with §3.2 is
the point: **the same estimator, unchanged, converges at `N^{−1}` on the
positive control and `N^{−1/2}` here.** Only the second rate is representative
of what the instrument will do on a bed that is not secretly linear. Error bars
quoted for this kit must be taken from MF-3's slope, not MF-1's.

---

## 4. INSTRUMENT 3 — ADJOINT GRADIENTS

### 4.1 Derivation

The continuous-depth component is `dz/dt = f(z, θ)` on `[0, T]` with `z(0) = z₀`
and a loss `L = g(z(T))`. Perturb the parameters, `θ → θ + δθ`. The state
perturbation obeys the first-variation equation

```
    (K12)  d(δz)/dt = (∂f/∂z) δz + (∂f/∂θ) δθ ,      δz(0) = 0
```

Introduce a multiplier `λ(t)` and require it to satisfy the **adjoint equation**

```
    (K13)  dλ/dt = −(∂f/∂z)ᵀ λ        (equivalently  dλ/dt = −λᵀ ∂f/∂z )
```

Then the bilinear form `λᵀ δz` has a derivative in which the state term cancels
identically:

```
    (K14)  d/dt [ λᵀ δz ] = (dλ/dt)ᵀ δz + λᵀ d(δz)/dt
                          = −λᵀ (∂f/∂z) δz + λᵀ (∂f/∂z) δz + λᵀ (∂f/∂θ) δθ
                          = λᵀ (∂f/∂θ) δθ
```

**(K13) is not an ansatz — it is exactly the choice that makes (K14) hold.**
That is the whole content of the adjoint method. Integrating (K14) over
`[0, T]` with `δz(0) = 0` and `λ(T) = ∂L/∂z(T)`:

```
    (K15)  δL = λ(T)ᵀ δz(T) = ( ∫₀^T λ(t)ᵀ (∂f/∂θ) dt ) δθ

           ⇒   dL/dθ = ∫₀^T λ(t)ᵀ (∂f/∂θ) dt ,     dL/dz₀ = λ(0)
```

The field is `f(z, W, b) = tanh(Wz + b)`. With `s := sech²(Wz+b) = 1 − f²`, the
three pullbacks the backward pass needs are

```
    (∂f/∂z)ᵀ λ = Wᵀ (s ⊙ λ)      (∂f/∂W)ᵀ λ = outer(s ⊙ λ, z)      (∂f/∂b)ᵀ λ = s ⊙ λ
```

`RUN`: the closed form `∂f/∂z = diag(s) W` agrees with `torch.func.jacrev` at
relative deviation `0.000e+00`.

**Implementation.** The augmented state `(z, λ, g_W, g_b)` is integrated
**backwards from `t = T` to `t = 0`** by the same RK4 as the forward solve, with
`z(T)` from the forward pass, `λ(T) = ∂L/∂z(T)`, and `g(T) = 0`. Because the
forward-time integrand of `g` is `+λᵀ ∂f/∂θ`, the backward integration returns
`g(0) = −dL/dθ`, and the sign is corrected once at the end. **The state is
reconstructed by integrating the field backwards rather than by storing the
forward trajectory** — that is what makes this the continuous adjoint and not
backpropagation through the solver, and it is the source of the memory saving
that motivates the method at all. The reconstruction error is measured:
`‖z_rec(0) − z₀‖ = 6.8280e-14` at `T = 1`, `d = 4`, 128 RK4 steps.

### 4.2 The declared kill

`RUN`. Field `dz/dt = tanh(Wz + b)`, `d = 4`, `T = 1.0`, RK4 with 128 steps,
rng seed 31. 16 parameters in `W`, 4 in `b`, 4 in the state: 24 scalar inputs,
4 outputs, 96 Jacobian entries compared.

```
    max absolute deviation  =  1.3678e-09
    max relative deviation  =  3.5904e-07        (central difference, eps = 1e-06)

    torch.autograd.gradcheck(eps=1e-6, atol=1e-6, rtol=1e-4)   →   PASS
```

**The gradcheck passes at `rtol = 1e-4`.** The tolerance was not loosened and
no fallback was taken. Training through the dynamics is licensed this round —
on the field, dimension and horizon tested, and on nothing wider (§6).

### 4.3 Why the residual is a discretisation gap and not a bug

The comparison is between the **continuous** adjoint, discretised by RK4, and a
finite difference of the **discretised** forward map. These are two different
`O(Δt⁴)` approximations to the same continuous gradient, so their difference is
itself `O(Δt⁴)`:

```
    (K16)  | g_adjoint − g_finite-difference |  =  O(Δt⁴)
```

A residual that carries the solver's order is a discretisation gap. A residual
that does not move with `Δt` is a bug in the adjoint. `RUN`:

| RK4 steps | `Δt` | max abs deviation | ratio vs previous |
|---|---|---|---|
| 8 | `0.125000` | `1.2869e-06` | — |
| 16 | `0.062500` | `8.2307e-08` | `15.6` |
| 32 | `0.031250` | `5.3572e-09` | `15.4` |
| 64 | `0.015625` | `1.3703e-09` | `3.9` |
| 128 | `0.007813` | `1.3678e-09` | `1.0` |

The per-halving ratios are `15.6` and `15.4` against the `2⁴ = 16` that (K16)
predicts, until the run hits the finite-difference floor at `1.37e-09` — where
the *control* stops improving, not the adjoint. Fitted over the points above
`10×` that floor the slope is `−3.97` against the predicted `−4`.

**Consequence for the pass.** The gradcheck pass at `rtol = 1e-4` is conditional
on `Δt`. At 8 RK4 steps the residual is `1.29e-06`, three decades worse; the
128-step pass is a converged result, not a lucky one, and §6 states the
condition.

---

## 5. THE CIRCULARITY QUESTION

X₂₈c proposes a consistency triangle for transient chaos: escape rate `κ`,
Lyapunov exponent `λ`, and saddle dimension `d`, linked by

```
    κ = λ (1 − d)
```

with all three measured independently and the identity checked.
`V13_X27_G1_PRIOR_ART.md` (E7, E8) already records Tél's statement of
Kantz–Grassberger, `D₁⁽¹⁾ = 1 − κ/λ`, and notes that it makes the dimension and
the escape rate two names for one measurement once `λ` is known. This section
settles the question by construction and by measurement.

### 5.1 A planted bed on which all three are known in closed form

Take the open two-branch map on `[0,1]` with **unequal** slopes:

```
    f(x) = a x           for x ≤ 1/2       (survives iff x ≤ 1/a)
    f(x) = b (1 − x)     for x >  1/2      (survives iff x ≥ 1 − 1/b)
```

with `a, b > 2` so the two surviving preimages are disjoint. Everything mapped
outside `[0,1]` escapes and never returns. Used below: `a = 3`, `b = 5`.

**Escape rate.** One step maps each surviving interval *onto* `[0,1]` uniformly,
so Lebesgue measure is **exactly** conditionally invariant and the surviving
measure is `σ^t` with no transient at all:

```
    (K17)  σ = 1/a + 1/b = 8/15 ,     κ = −ln σ = ln(15/8) = 0.6286086594223742
```

**Natural measure on the saddle.** The Gibbs measure for the potential
`−ln|f'|` is Bernoulli with

```
    (K18)  p₁ = (1/a)/σ = 5/8 ,     p₂ = (1/b)/σ = 3/8
```

**Lyapunov exponent.**

```
    (K19)  λ = p₁ ln a + p₂ ln b = 1.2901718975803562
```

**Information dimension.** For a self-similar measure the information dimension
is `D₁ = H/λ` with `H = −Σ p_i ln p_i`. Substituting (K18),

```
    H = −Σ p_i ln p_i = Σ p_i ln(slope_i) + ln σ = λ − κ
```

because `p_i = (1/slope_i)/σ` gives `−ln p_i = ln(slope_i) + ln σ` and
`Σ p_i = 1`. Therefore

```
    (K20)  D₁ = H/λ = (λ − κ)/λ = 1 − κ/λ = 0.5127713906950743
```

**Kantz–Grassberger is recovered exactly on this bed, by construction.** `RUN`:
`H = 0.6615632381579821` against `λ − κ = 0.6615632381579820`, deviation
`1.110e-16`.

**Box-counting dimension.** `D₀` is the Moran root of `a^{−s} + b^{−s} = 1`:

```
    (K21)  D₀ = 0.5183702741223462 ,      D₀ − D₁ = +5.599e-03
```

`D₀ ≠ D₁` because `a ≠ b`. **The saddle is multifractal and the identity is
about `D₁` only.** This matters in §5.5.

### 5.2 Three routes, and whether each avoids the other two

The question posed is: for each of the three quantities, name a measurement
route that uses neither of the other two, and say whether such a route exists.

| quantity | route | inputs | reads `κ`? | reads `λ`? | reads `d`? | exists |
|---|---|---|---|---|---|---|
| `κ` | survivor counting: seed `N₀` uniformly, record `n(t)`, fit `ln n(t) = ln N₀ − κt` | counts only | — | no | no | **yes** |
| `λ` | Benettin (K5)–(K7) along orbits that remain in the region; in `d=1` the pooled mean of `ln|f'|` over surviving transitions | `Df` along the orbit | no | — | no | **yes** |
| `d` | the `ε`-scaling of the saddle's own natural measure: `I(ε) = −Σ P ln P`, `D₁ = lim I(ε)/ln(1/ε)` | the measure on the saddle | no | no | — | **yes, with two conditions** |

**All three routes exist.** The triangle is *not* intrinsically circular, and
the first finding of this section is that X₂₇'s reading — dimension and escape
rate as two names for one measurement — is too strong as stated. It is
conditionally true, not identically true.

**The two conditions on the `d` route**, both load-bearing:

1. **The measure must be the natural (two-sided) measure on the saddle, and
   only the dimension leg needs it.** On this bed the forward-conditioned limit
   — the conditionally invariant measure — is Lebesgue itself (K17), which is
   absolutely continuous and has information dimension **1**, not
   `D₁ = 0.5128`. Feeding it to the identity would demand `κ = 0`. The Lyapunov
   leg survives this because `λ` depends only on the *symbol* distribution at a
   single time, and conditioning on one further survival already draws branch
   `i` with probability exactly `p_i` — which is why route 2 below is unbiased
   with zero transient. The dimension leg does not survive it, because a
   dimension depends on the spatial distribution over infinitely many symbols in
   *both* time directions. Conditioning must run both ways (sprinkler, PIM
   triple, stagger-and-step), and that is the expensive part of the third leg.
2. **The `ε`-range must be wide enough for the log-periodic oscillation to
   damp** — see §5.4, where this is what sets the whole triangle's precision.

### 5.3 The three routes, measured

`RUN`, `N₀ = 4,000,000` uniform seeds, rng seed 32.

**Route 1, `κ` from survivor counts alone.** The surviving fraction tracks
`σ^t` with no transient, as (K17) requires:

| `t` | `n(t)` | `n(t)/N₀` | predicted `σ^t` |
|---|---|---|---|
| 1 | 2,134,191 | `5.335477e-01` | `5.333333e-01` |
| 4 | 323,173 | `8.079325e-02` | `8.090864e-02` |
| 8 | 25,984 | `6.496000e-03` | `6.546208e-03` |
| 12 | 2,157 | `5.392500e-04` | `5.296448e-04` |
| 15 | 318 | `7.950000e-05` | `8.034908e-05` |

```
    MEASURED   κ̂ = 0.628704 ± 0.001583        (s.e. from the binomial variance of ln n(t))
    PREDICTED  κ  = 0.628609
    deviation  9.571e-05   =  0.06 s.e.
```

**Route 2, `λ` from `ln|f'|` along surviving orbits alone.** Because Lebesgue is
exactly conditionally invariant, the points alive at `t` are Lebesgue
distributed and conditioning on one further survival draws branch `i` with
probability exactly `p_i` — so the estimator is **unbiased with zero transient**.
4,570,289 surviving transitions pooled.

```
    MEASURED   λ̂ = 1.290331 ± 0.000116
    PREDICTED  λ  = 1.290172
    deviation  1.592e-04   =  1.38 s.e.
```

**Route 3, `d` from the `ε`-scaling of the saddle measure alone.** Level-22
cylinder enumeration by inverse iteration: 4,194,304 intervals, longest
`3.187e-11`, well below the smallest `ε = 9.537e-07`, so no box straddles the
construction's own resolution. `I(ε)` binned at `ε = 2^{−j}`, `j = 8..20`.

The estimator's error bar is its **window dependence**, computed without
reference to the closed form: a self-similar measure with incommensurable
contraction ratios (`ln 3`, `ln 5`) carries log-periodic oscillations on
`I(ε)`, so the fitted slope moves with the window.

| window | `D̂₁` | `D̂₀` |
|---|---|---|
| `j = 8..16` | `0.519037` | `0.523877` |
| `j = 9..17` | `0.523576` | `0.527734` |
| `j = 10..18` | `0.515926` | `0.517689` |
| `j = 11..19` | `0.507564` | `0.506810` |
| `j = 12..20` | `0.506335` | `0.505614` |

```
    MEASURED   D̂₁ = 0.514160 ± 0.008620   (half-spread over sliding 8-octave windows)
    PREDICTED  D₁ = 0.512771              deviation 1.389e-03
    MEASURED   D̂₀ = 0.516869 ± 0.011060
    PREDICTED  D₀ = 0.518370              deviation 1.501e-03
```

**This leg is 75× less precise than `λ` and 5× less precise than `κ`, and it is
the only leg whose error is a fit systematic rather than a countable
statistic.**

### 5.4 The identity checked on three genuinely independent numbers

```
    κ̂ − λ̂ (1 − D̂₁)  =  +1.810495e-03
    combined uncertainty (κ s.e. 1.58e-03, λ s.e. 1.16e-04, D₁ systematic 8.62e-03)
                        =  1.123546e-02
    residual / uncertainty  =  0.16
```

The identity holds. And the error budget is the finding:

```
    the dimension leg supplies 98.0% of that variance
    DISCRIMINATING POWER: the check can only reject a violation larger than 1.79% of κ
```

**The triangle's entire resolving power is set by the one leg that in practice
is never measured independently.** `κ` is known to `0.25%` and `λ` to `0.009%`;
the check is blunt to `1.79%` because `d` is known to `1.7%`. Adding the third
leg does not tighten the other two — it loosens the conclusion by an order of
magnitude relative to what `κ` and `λ` alone support.

This is also, quantitatively, X₂₇'s owed relation U4 (`α = κ/λ`) seen from the
other side: the reason `d` is not independent *in practice* is not that the
mathematics forbids it, but that measuring it costs an order of magnitude in
precision, so the relation is used to supply it instead.

### 5.5 The check cannot detect the wrong dimension

`D₀` and `D₁` differ in closed form by `5.599e-03` (K21) — a real multifractal
gap, and substituting one for the other is the single most common way to get
this wrong, since box-counting is far easier to implement than an information
dimension. Feeding the box-counting dimension to the identity:

```
    κ̂ − λ̂ (1 − D̂₀)  =  +5.305251e-03   =  0.47 × the uncertainty  =  0.84% of κ
```

**Not detected.** The closed-form `D₀`/`D₁` gap (`5.60e-03`) is *smaller* than
the direct estimator's own window systematic (`8.62e-03`), so the triangle
cannot tell which dimension it was handed. A check that does not notice the
wrong dimension is not certifying the right one. This is asserted in the script
in that direction: the assertion fires if the substitution ever *becomes*
detectable, which would mean this paragraph needs rewriting.

### 5.6 If `d` comes from the relation, the check is a tautology

The case that actually obtains in the literature X₂₇ fetched. Tél (E7) presents
`D₁⁽¹⁾ = 1 − κ/λ` as a way to *obtain* the dimension; nothing in E7–E9 measures
`D₁` independently and then tests the relation. If `d := 1 − κ/λ`, then

```
    (K22)  κ − λ(1 − d) = κ − λ(1 − (1 − κ/λ)) = κ − κ = 0
```

identically, for **any** `(κ, λ)` whatsoever. `RUN`, 10⁵ random pairs drawn on
`[−3, 3]²`, deliberately including negative exponents that describe no chaotic
saddle at all:

```
    max absolute residual  =  8.882e-16
    max relative residual  =  4.232e-12
    fraction of nonsense pairs rejected at 1e-6 relative:  0.000000
    →  0 rejections out of 100000
```

The residual is roundoff. It carries no information about the dynamics, about
whether a chaotic saddle exists, or about whether the exponents even have the
right sign.

### 5.7 Verdict

**The consistency triangle is not three independent instruments certifying each
other. It is two cheap measurements, one expensive one, and an identity that is
satisfied by construction whenever the expensive one is skipped.**

Stated precisely, in the three parts the question asks for:

1. **Independent routes exist for all three.** `κ` from survivor counts, `λ` from
   Benettin on the variational equation, `d` from the `ε`-scaling of the
   saddle's natural measure. All three were run on a planted bed where all three
   are known in closed form, and all three agree with their closed forms —
   `κ̂` at `0.06` s.e., `λ̂` at `1.38` s.e., `D̂₁` within its own window
   systematic. X₂₇'s statement that the relation makes `d` and `κ` two names for
   one measurement is **too strong as a matter of mathematics**.
2. **But the triangle certifies almost nothing even when all three are
   measured.** The dimension leg carries `98.0%` of the error budget, the check
   is blunt to `1.79%` of `κ`, and it does not detect a `D₀`-for-`D₁`
   substitution that is a real `0.84%` error. **X₂₇'s statement is correct as a
   matter of practice**, for a reason X₂₇ does not give: not that `d` is
   analytically dependent, but that measuring it independently is an order of
   magnitude harder than the other two legs, so nobody does.
3. **And in the configuration this repo would actually be in, it certifies
   exactly nothing.** If `d` is taken as `1 − κ/λ`, the check is (K22), an
   algebraic identity with a measured `0/100000` rejection rate on deliberately
   nonsensical inputs.

**Recommendation on the evidence.** No number is published from the triangle
unless the report publishing it carries, for the `d` leg: the estimator used,
the `ε`-window in octaves, the window systematic obtained by sliding that
window, and an explicit statement that `D₁` and not `D₀` was measured. Absent
all four, the triangle is a rearrangement and must not be cited as a
consistency check.

**What survives and is worth keeping.** Tél's ordering constraint `λ > κ`
(X₂₇ E7) is a genuine falsifiable consequence that costs only the two cheap
legs and never touches `d`. `RUN` on the planted bed: `λ̂ = 1.290331 >
κ̂ = 0.628704`. That is the part of the triangle with real content, it is free,
and it is falsifiable. The `d` leg should be reported as a *derived* quantity
with its inherited error bar, labelled as derived — not entered into a
consistency check it cannot fail.

---

## 6. LIMITS

Everything qualifying any number above is collected here.

**Scope.** Every bed in this file is planted: three low-dimensional maps, one
piecewise-linear open map, and a 4-dimensional `tanh` field. **Nothing here is
measured on the CEQ bed.** The kit is validated, not applied; no claim about the
architecture follows from any number above.

**Benettin in `d > 1`.** The QR frame of (K5) is exercised only on Hénon,
`d = 2`, and only against the *sum* rule (K11). `λ₁` individually has no closed
form on that bed and is reported against a literature figure. Partial spectra
(`k < d`), `d > 2`, and non-square frames are untested. In `d = 1` — both
must-fires — the QR step is a `1×1` factorisation and tests nothing about the
frame.

**Float64 orbits are not true orbits.** The `r = 4` measurement averages
`ln|f'|` along a float64 trajectory, which is eventually periodic and is not the
true orbit of the map. The claim is only that the empirical average
approximates the ergodic average; shadowing is assumed, not proved. The seeding
guard and the non-positive-`R`-diagonal counter bound the *detectable* failure
mode (collapse onto the fixed point), not the undetectable one.

**Convergence rate.** (K8)'s `N^{−1/2}` was measured on the logistic map at
`r = 4`, where the observable's variance `π²/12` is finite and computable. Maps
with intermittency or with non-integrable `log` singularities in `ln|f'|` do not
satisfy the hypothesis and will not show this rate.

**The finite-difference control.** §1.4 records that a polynomial bed of degree
`≤ 2` cannot falsify the (K3) step-size model, because `D³f ≡ 0` there. The
model is confirmed on `tanh` only. The `h = ε_m^{1/3}·max(1,‖x‖_∞)` rule uses
the state magnitude as a stand-in for `M₀/C₃`; on a field where those differ by
several decades the rule will be off by the cube root of that ratio.

**The adjoint gradcheck is narrow.** `d = 4`, `T = 1`, one seed, one field
family (`tanh(Wz+b)`), non-stiff, fixed-step RK4. Stiff fields, longer horizons,
adaptive solvers, and larger `d` are untested. The backward reconstruction error
`6.83e-14` at `T = 1` grows with `T` and with the field's expansivity; on an
expanding field it grows exponentially, and at that point the continuous adjoint
must be replaced by checkpointing. **The pass is conditional on `Δt`**: at 8 RK4
steps the residual is `1.29e-06`, three decades above the 128-step value. The
licence to train through the dynamics extends to this field, this horizon and
`Δt ≤ 1/128`, and no further.

**What the gradcheck compares.** The continuous adjoint is checked against
finite differences of the *discretised* forward map. Both are `O(Δt⁴)`
approximations to the same continuous gradient, so the residual is a
discretisation gap (§4.3), not an exactness claim. A gradcheck against the
discrete adjoint of the RK4 stepper would be exact to roundoff, but that object
is backpropagation through the solver and is not what was built.

**The triangle bed is hyperbolic by fiat.** The open map of §5.1 is piecewise
linear with slopes `> 2` everywhere, so Kantz–Grassberger's hyperbolicity and
ergodicity conditions hold by construction. Those conditions are still owed as
`V13_X27_G1_PRIOR_ART.md` U3 — the original derivation has not been read — and
X₂₇ E10 records that non-hyperbolic saddles decay algebraically rather than
exponentially, which would break (K17)'s exponential fit before it broke the
identity. **Nothing here tests the relation where its hypotheses fail.**

**The dimension leg's precision is a floor, not a typical value.** Route 3
enumerates cylinders analytically because this bed's inverse maps are known in
closed form. On a bed whose saddle must be located numerically (PIM triple,
stagger-and-step, or sprinkler with two-sided conditioning) the dimension
estimate will be *worse* than the `±8.6e-03` measured here. The `1.79%`
discriminating power of §5.4 is therefore a best case for the triangle, and the
`98.0%` error share of the dimension leg is a lower bound on its dominance.

**`D₁` versus the partial dimension `D₁⁽¹⁾`.** Kantz–Grassberger's relation is
about the *partial* information dimension along the unstable direction. On this
1-dimensional bed the partial dimension and the information dimension of the
measure coincide, so §5 never has to separate them. In `d > 1` the partial
dimension requires the Oseledets splitting to know which direction to slice
along — and that splitting comes from the Lyapunov instrument. **The `d` leg's
independence from `λ` is therefore weaker in higher dimensions than it is here,
and by how much has not been measured.** Any transport of §5.2's "all three
routes exist" verdict to a higher-dimensional bed must re-establish it there.

**Provenance of the wall-clock figures.** One machine, Windows 11, python
3.11.9, numpy 1.26.4, torch 2.5.1+cu121, CPU float64, single process, all
BLAS thread pools pinned to 1 (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB` set before
the numpy import, `torch.set_num_threads(1)`). Pinning is not an
optimisation here: the Benettin loop factorises batches of `1x1` and `2x2`
matrices, on which multithreaded LAPACK spends more time in thread handoff
than in arithmetic. The box
was concurrently running several other measurement lanes during this run, so
the timings are load-contaminated and are not benchmarks. Every *numerical*
result above is deterministic given the stated seeds and is unaffected.
