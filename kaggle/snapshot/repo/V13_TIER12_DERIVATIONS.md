# V13_TIER12_DERIVATIONS.md — the Tier-1 and Tier-2 paper derivations

Evidence classes follow `MATHEMATICS.md`. `RUN` means executed this session and the
output read. `READ` means a file and line was opened and quoted. `DERIVED` means it
follows from `RUN`/`READ` by steps written out below. `CITED` means an external result
is used and named as external. Every numbered equation that can be checked at finite
precision is checked by `scripts/v13_derivation_check.py`, whose assertion messages
carry the same numbers. `RUN`, this session: **exit 0, every assertion passed**, 13.6 s
wall clock, numpy 1.26.4, python 3.11.9, float64.

These derivations are done **before** any architecture is built, so that the synthetic
test bed can be labelled in closed form rather than by simulation. Tier 1 supplies the
label `a*` — the true lowest-barrier exit — with an exact rate rather than a sampled
estimate. Tier 2 supplies `τ_H` with a computed false-positive rate rather than a
held-out percentile.

---

## 0. WHAT THIS LEANS ON, NAMED

Nothing below re-proves what the project already owns. The reuse is exact:

| Existing object | Where | Used at |
|---|---|---|
| `occupancy A N := Σ_{k<N} A^k` | `lean/CEQ/Occupancy.lean:42` | (T1.6) |
| `occupancy_telescope : (1−A)·occupancy A N = 1 − A^N` | `lean/CEQ/Occupancy.lean:52` | (T1.6), (T1.7) |
| `occupancy_eq_inverse_of_nilpotent` | `lean/CEQ/Occupancy.lean:83` | (T1.7) contrast |
| `PerronCertificate A w ρ` (`nonneg`, `w_pos`, `dominates`) | `lean/CEQ/Contraction.lean:59` | (T1.8) |
| `weighted_contraction` / `weighted_contraction_iterate` | `lean/CEQ/Contraction.lean:72, 99` | (T1.8) |
| `oracle_ne_resolvent`, `not_isNilpotent` | `lean/CEQ/OracleSeparation.lean:166, 148` | (T1.7) |
| `truncation_never_exact` | `lean/CEQ/OracleSeparation.lean:180` | (T1.7) |
| `N = (I − Q)^{-1}`, `B = N R`, `λ₂ := ρ(Q)` | `MATHEMATICS.md` §7 | (T1.5), (T1.9) |
| Kirchhoff grounded solve `ω_x = M_xa / M_aa` | `MATHEMATICS.md` §11 step 4 | (T1.15) |
| The `< 1e-10` instrument tolerance | `MATHEMATICS.md` §11 | every Tier-1 assert |

**Symbols are §7's, not new ones.** The basin's transient block is `Q`, its fundamental
matrix is `N = (I − Q)^{-1}`, absorption is `B = N R`, and the block's spectral radius
keeps §7's name `λ₂ := ρ(Q)` — §7 introduced that name precisely because the literal
second eigenvalue of a block-triangular `P` is `1` and says nothing.

---

## 1. TIER 1 — THE DISCRETE KRAMERS LAW ON A PLANTED MULTI-BASIN CHAIN

### 1.1 The planted chain, and why detailed balance is exact rather than approximate

The landscape is a finite state set `X`, an energy `E : X → R`, a temperature `T > 0`,
and a symmetric proposal graph `G`. **Assumption, used from here on:** `G` is connected
and `d_max := max_i deg(i)`.

```
    (T1.1)   G_ij = 1/d_max  on an edge,   G_ii = 1 − deg(i)/d_max,   G = Gᵀ
    (T1.2)   P_ij = G_ij · min(1, e^{−(E_j − E_i)/T})   for i ≠ j
             P_ii = 1 − Σ_{j≠i} P_ij
    (T1.3)   π_i = e^{−E_i/T} / Z,   Z = Σ_j e^{−E_j/T}
```

Detailed balance is exact, not asymptotic, because the acceptance rule collapses the
product to a symmetric expression:

```
    (T1.4)   π_i P_ij = G_ij · min(e^{−E_i/T}, e^{−E_j/T}) / Z = π_j P_ji
```

The right-hand side is invariant under `i ↔ j` since `G` is symmetric and `min` is
symmetric. `RUN`: `‖diag(π)P − Pᵀdiag(π)‖_max = 1.734723e-18` on the double well and
**exactly `0.000000e+00`** on the three-basin instance; row-sum deviation `0` and
`1.110e-16`. This is machine zero — the residual is one unit in the last place of the
smallest `π_i`, not a modelling error.

Two instances are built and used throughout.

**Double well**, nine states on a line,
`E = [0.00, 0.40, 1.10, 2.20, 3.00, 2.20, 1.10, 0.40, 0.10]`, basin `A = {0,1,2}`,
barrier `{3,4,5}` with the index-1 saddle at state 4, basin `B = {6,7,8}`.

**Three basin**, eleven states, basin `A = {0,1}` with two exit channels: one saddle at
`E = 2.00` leading to `B = {3,4}`, and **four parallel saddles at `E = 2.20`** leading
to `C = {9,10}`. This instance exists to settle §1.7.

### 1.2 The fundamental matrix IS the proved resolvent, at `γ → 1`, on a block

Fix a basin `A` and let `I` be the transient set (§1.4 states which `I` each question
uses). Partition `P` by `I` and its complement `∂`, and make `∂` absorbing:

```
    (T1.5)   Q := P_II  (substochastic),   R := P_{I∂},   N := (I − Q)^{-1}
             τ = N 1     is the MFPT vector out of I,     k = 1 / (π_A · τ)
```

`τ_i = 1 + Σ_{j∈I} Q_ij τ_j` is the first-step decomposition, i.e. `(I − Q)τ = 1`.

**The connection to the proved resolvent, made rigorous rather than gestural.**
`CEQ.Occupancy.occupancy_telescope` proves, over an arbitrary ring and hence at
`Matrix n n ℝ`,

```
    (T1.6)   (1 − A) · Σ_{k<N} A^k = 1 − A^N
```

That is a **finite** identity and it is deliberately all that is proved — the Lean
module's own header states the infinite limit is withheld because it needs a convergence
hypothesis. Instantiating (T1.6) twice makes the relation precise:

* At `A = γP` with `P` row-stochastic, `ρ(γP) = γ`. If `γ < 1` the tail `(γP)^N → 0` and
  the limit is the project's resolvent `Σ_t γ^t P^t = (I − γP)^{-1}`.
* At `A = Q = P_II`, the tail `Q^N → 0` **iff `ρ(Q) < 1`**, and the limit is `N`.

So `N` is the `γ → 1` case of the same lemma restricted to a substochastic block, and the
two differ only in **why** the spectral radius is below one. Discounting removes a fixed
fraction `1 − γ` of mass at every row uniformly; the substochastic block removes mass only
at the rows that touch `∂`. The sharp statement is therefore:

```
    (T1.7)   N = lim_{γ→1} (I − γQ)^{-1}   exists  ⟺  ρ(Q) = λ₂ < 1
             whereas  lim_{γ→1} (I − γP)^{-1}  diverges for stochastic P
```

The `γ → 1` limit exists on the block and does not exist on the full chain. **Condition,
stated exactly:** `λ₂ < 1` holds iff from every `i ∈ I` there is a path inside `I` to a
state with positive transition out of `I`. If some communicating class inside `I` had no
exit, `Q` restricted to it would be stochastic and would carry eigenvalue 1; conversely
if every state can exit within `m = |I|` steps then `Q^m 1 < 1` strictly, so
`‖Q^m‖_∞ < 1` and `λ₂ ≤ ‖Q^m‖_∞^{1/m} < 1`.

`CEQ.OracleSeparation.not_isNilpotent` is what forbids the cheap route: `Q` is
non-negative with symmetric support and at least one positive entry, so `Q^N` is never
exactly zero, and `truncation_never_exact` says no partial sum is ever the inverse. The
basin block is therefore genuinely the limit case of (T1.6), never the finite case that
`occupancy_eq_inverse_of_nilpotent` covers for the arm's own strictly-lower-triangular
operator.

**The MFPT vector is itself the Perron certificate the contraction lemma wants.** This is
the step that binds the convergence hypothesis to a proved object rather than asserting
it. From `(I − Q)τ = 1`, `Q τ = τ − 1`, and `τ ≥ 1 > 0` entrywise since
`τ_i = 1 + (non-negative)`. Hence for every `i`,

```
    (T1.8)   Σ_j Q_ij τ_j = τ_i − 1 ≤ (1 − 1/‖τ‖_∞) · τ_i
```

which is exactly `PerronCertificate Q τ ρ_A` with `ρ_A := 1 − 1/‖τ‖_∞`, all three fields
discharged: `nonneg` from (T1.2), `w_pos` from `τ ≥ 1`, `dominates` from (T1.8). Feeding
it to `weighted_contraction` gives `‖Q^t v‖_τ ≤ ρ_A^t ‖v‖_τ`, so `Q^N → 0`, so the limit
in (T1.7) exists, so `N` is the resolvent. Collatz–Wielandt then reads `λ₂ ≤ ρ_A`.

`RUN`, double well at `T = 0.4`, `I = A`:

```
    τ                 = [748.93442368, 743.49786002, 700.70339090]
    ‖Qτ − (τ − 1)‖    = 1.136868e-13
    λ₂ = ρ(Q)         = 0.998658649242
    1 − 1/‖τ‖_∞       = 0.998664769613          the certificate rate, and it holds
    ‖N1 − LU solve‖   = 2.273737e-13            second route
    ‖N1 − Σ_{t<2²⁰}Q^t 1‖ = 6.480150e-12        third route, matrix products only
    ‖(I−Q)S_64 − (I − Q^64)‖ = 8.104628e-15     (T1.6) itself, at N = 64
```

The third route never solves a linear system; it evaluates `CEQ.Occupancy.occupancy` by
the doubling recursion `S_{2m} = S_m + Q^m S_m`. Its agreement with the closed form is
(T1.6) verified numerically, and the fourth line is the Lean lemma verified directly.

### 1.3 Two escape rates, and they are not the same number

```
    (T1.9)    starting from the quasi-stationary distribution ν (νQ = λ₂ν, ν1 = 1),
              E_ν[τ] = ν N 1 = Σ_t λ₂^t = 1/(1 − λ₂)   exactly — escape is geometric
    (T1.10)   starting from π restricted to A and normalised,
              k_A := 1 / (π_A · τ)                      the pre-asymptotic rate
```

`RUN`: `E_ν[τ] = 745.517154088` against `1/(1 − λ₂) = 745.517154088`, relative gap
`1.799e-14`. The `π_A`-weighted MFPT is `745.384150701`, so `k_A = 1.341590104e-03`; the
two starting measures differ by a factor `1.000178436` on this deep well.

**This is §7's distinction, not a new one.** §7 records `t_rel = 1/(1 − λ₂) = 13.3333`
and then records that every ladder rung sits below the mode time, so `λ₂^{t*}` is the
wrong predictor at the rungs. (T1.9) says what `t_rel` *is* — a mean escape time from the
quasi-stationary start — and (T1.10) is the pre-asymptotic quantity the bed should label
with. **The bed must label with (T1.10).**

### 1.4 Splitting probabilities are exactly the committor

For the splitting question the transient set is `I_split := X \ ⋃_{r≠A} B_r`: the other
basins absorb, the barrier states do not. Writing `R_r := P_{I,B_r}`,

```
    (T1.11)   q^{(r)} := N R_r 1 = (I − P_II)^{-1} P_{I,B_r} 1
    (T1.12)   Σ_r q^{(r)} = 1  on I         (some basin is reached with probability 1)
```

`q^{(r)}_i` is the probability of exiting basin `A` into `B_r` starting from `i`. The
first-step decomposition `q_i = Σ_{j∈I} P_ij q_j + Σ_{j∈B_r} P_ij` rearranges to
`(I − Q)q = R_r 1`, which is (T1.11). **That same rearrangement is the discrete Dirichlet
problem.** With the generator `L := P − I`,

```
    (T1.13)   L q^{(r)} = 0   on I,     q^{(r)}|_{B_r} = 1,     q^{(r)}|_{B_s} = 0 (s ≠ r)
```

so `q^{(r)}` is the committor `q(x) = P_x(reach B_r before every other basin)`. The two
computations share no matrix: (T1.11) inverts the `|I| × |I|` block, (T1.13) solves a
full `|X| × |X|` system with pinned boundary rows.

`RUN`: `‖(I − Q)^{-1}R − Dirichlet‖ = 1.110223e-16` on **both** instances;
`‖Σ_r q^{(r)} − 1‖ = 1.110e-16` (double well) and `1.132e-14` (three basin).

**A third route, and it is the project's own instrument law.** Define the conductances

```
    (T1.14)   c_ij := π_i P_ij  (symmetric by T1.4),   L_c := D − C = diag(π)(I − P)
```

`L_c` is a symmetric weighted graph Laplacian with `L_c 1 = 0`. Contracting each boundary
set to a single node leaves the interior harmonic equations unchanged (every node of a
boundary set carries the same `q`), so §11 step 4 applies verbatim:

```
    (T1.15)   ground at the sink, M := (L_c with the sink row/column deleted)^{-1},
              ω_x = M_{x,α} / M_{α,α}        α the contracted source node
```

§11 states this for the unweighted graph Laplacian of the undamped walk; the derivation
there uses only that `L` is a symmetric Laplacian of a conductance network, which
(T1.14) supplies for any reversible chain. `RUN`: agreement with the absorbing-chain
committor `2.220446e-16`, well inside §11's own `1e-10` law.

### 1.5 The exact rate is capacity over partition function

`CITED`, classical potential theory for reversible chains (Bovier–den Hollander,
*Metastability*, the mean-hitting-time formula). Let `h(x) := P_x(τ_A < τ_B)`, solving
`Lh = 0` off `A ∪ B` with `h|_A = 1`, `h|_B = 0`. Then

```
    (T1.16)   cap(A,B) = Σ_{x∈A} π_x P_x(τ_B < τ_A⁺)
                       = Σ_{x∈A} π_x ((I − P)h)(x)
                       = ½ Σ_{x,y} c_xy (h(x) − h(y))²          the Dirichlet form
    (T1.17)   E_{ν_{A,B}}[τ_B] = ( Σ_x π_x h(x) ) / cap(A,B),
              ν_{A,B}(x) = π_x P_x(τ_B < τ_A⁺) / cap(A,B)   on A
```

The second line of (T1.16) is `DERIVED`: for `x ∈ A`,
`P_x(τ_B < τ_A⁺) = Σ_y P_xy (1 − h(y)) = 1 − (Ph)(x) = ((I − P)h)(x)` since `h(x) = 1`.

`RUN`, double well at `T = 0.4`, `A = {0,1,2}`, `B = {6,7,8}`: the two capacity formulas
give `4.593823098983e-05` and `4.593823098984e-05`, relative `1.003e-14`; the two sides
of (T1.17) give `11792.307004868` and `11792.307004868`, relative `3.995e-14`.

### 1.6 The continuous limit — Kramers, Eyring, and where the prefactor comes from

Taking `T → 0` in (T1.17) is a Laplace asymptotic on both factors. `h ≈ 1` on `A`'s basin
and `≈ 0` beyond the saddle, so `Σ_x π_x h(x) ~ Z_A := Σ_{x∈A} π_x`. The Dirichlet form
is a resistance network, and for a channel that is a series of bonds the capacity is the
series conductance, dominated by the smallest bond — the saddle. Hence

```
    (T1.18)   k = 1/E[τ] ~ cap(A,B) / Z_A ,     cap ~ 1/R_series ~ c_saddle
              c_saddle = π_s p_s = (1/Z) e^{−E_s/T} p_s ,
              Z_A      = (1/Z) e^{−E_min/T} · n_A^eff ,  n_A^eff := Σ_{x∈A} e^{−(E_x−E_min)/T}
        ⟹    k ~ (p_s / n_A^eff) · e^{−ΔE‡/T} ,          ΔE‡ := E_s − E_min
```

which is **Arrhenius with an explicit discrete prefactor** `p_s / n_A^eff`. The
correspondence to the continuum forms is a reading of that prefactor, not a separate
result: in a harmonic well `E ≈ E_min + ½ω_min²x²` the sum `n_A^eff` becomes the Gaussian
integral `√(2πT)/ω_min`, and the saddle's per-step commitment `p_s` carries the saddle
curvature and the friction, so `p_s/n_A^eff` is Kramers'
`ω_min ω_sad / 2πγ`. Eyring's `rate ∝ exp(−ΔE‡/kT)` is the same expression with the
prefactor asserted universal. **`CITED` as a correspondence, `DERIVED` only to the level
of (T1.18) — the identification of `p_s` with `ω_sad/2πγ` is a continuum reading and is
not checked numerically here.**

`RUN`, double well, escape from `A = {0,1,2}` (exit bond `2–3`, `ΔE‡ = 2.20`):

| `T` | `k` | `k·Z_A / c_saddle` | `k·Z_A·R_series` |
|---|---|---|---|
| 0.800 | 1.409956e-02 | 0.820184 | 1.114006 |
| 0.500 | 3.554877e-03 | 0.903466 | 1.028258 |
| 0.400 | 1.341590e-03 | 0.940057 | 1.010596 |
| 0.300 | 2.469842e-04 | 0.974612 | 1.001940 |
| 0.250 | 6.130373e-05 | 0.987618 | 1.000481 |
| 0.200 | 7.298715e-06 | 0.995865 | 1.000057 |
| 0.150 | 1.993392e-07 | 0.999342 | **1.000002** |

Two readings. The single-bond Arrhenius form converges to 1 but slowly — it is still
18 % off at `T = 0.8` and 6 % off at `T = 0.4`. **The series-resistance form
`k = 1/(Z_A · R_series)` is right to `2e-6` by `T = 0.15` and to 1 % by `T = 0.4`**, and
it costs one sum. The bed should carry the series form; the exponential is the asymptote,
not the label.

### 1.7 WHERE THE EXPONENTIAL LAW FAILS AS A RANK-ORDER PREDICTOR

This is the section the bed's design turns on. A barrier head is to be trained to rank
exits by `−ΔE‡`. **Within one basin `Z_A` is common to every channel and cancels from the
ratio**, so ranking channels by rate is ranking them by channel capacity, and by (T1.12)
the rate ranking is exactly the splitting-probability ranking:

```
              k_{A→r} = q̄_r · k_A ,   q̄_r := π_A · q^{(r)} ,   k_A common to all r
    ⟹        argmax_r k_{A→r} = argmax_r q̄_r
```

**So the question is whether `argmax_r q̄_r` can differ from `argmin_r ΔE‡_r` on a legal
instance. It can, and the counterexample is exact rather than asymptotic.**

The three-basin instance has channel `B` over **one** saddle at `E = 2.00` and channel
`C` over **four parallel** saddles at `E = 2.20`. Both channels are two bonds long and
structurally identical apart from saddle height and multiplicity, and the proposal graph
is a tree, so the network is exactly series–parallel and the capacity ratio is closed
form:

```
    (T1.19)   c_channel(B) = e^{−E_B/T} / (12 Z) ,   c_channel(C) = 4 e^{−E_C/T} / (12 Z)
    (T1.20)   q̄_C / q̄_B = m · e^{−ΔΔE‡/T} ,   m = 4 (multiplicity),  ΔΔE‡ = 0.20
```

`RUN`, and the identity is exact to float64, not asymptotic:

| `T` | `q̄_C / q̄_B` | `4 e^{−0.2/T}` | rel |
|---|---|---|---|
| 0.500 | 2.681280184142556 | 2.681280184142557 | 4.969e-16 |
| 0.350 | 2.258872488031034 | 2.258872488031037 | 9.830e-16 |
| 0.250 | 1.797315856468886 | 1.797315856468886 | 0.000e+00 |
| 0.200 | 1.471517764685770 | 1.471517764685769 | 1.509e-16 |
| 0.150 | 1.054388552462905 | 1.054388552462907 | 1.685e-15 |
| 0.120 | 0.755502411350246 | 0.755502411350247 | 2.351e-15 |

**The inversion, measured.** Barrier to `B` is `2.00`; barrier to `C` is `2.20`, i.e.
strictly higher.

```
    T = 0.50   q̄_B = 0.271644632   q̄_C = 0.728355368
               k_B = 9.686754e-04   k_C = 2.597290e-03      ← C is 2.68x FASTER
    T = 0.10   q̄_B = 0.648785520   q̄_C = 0.351214289
               k_B = 1.667280e-10   k_C = 9.025675e-11      ← rankings agree again
```

**Answer, unqualified: yes. A legal instance exists on which barrier ranking and true-rate
ranking disagree.** At `T = 0.50` a barrier head that ranks by `−ΔE‡` names `B` and the
truth is `C`, by a factor of 2.68 in rate and 2.68 in splitting probability. Nothing about
the instance is pathological: energies are ordinary, the chain is reversible by
construction, detailed balance is exactly zero, and the exit is a plain entropic channel —
four ways over a slightly higher pass beats one way over a slightly lower one.

**The crossover, and it gives the bed a guard rather than a warning.** Setting the ratio
in (T1.20) to one,

```
    (T1.21)   T* = ΔΔE‡ / ln m
              rankings AGREE  ⟺  ΔΔE‡ > T · ln(prefactor ratio)
```

`RUN`: bisection on the three-basin instance gives `T* = 0.144269504088897` against
`ΔΔE‡/ln m = 0.20/ln 4 = 0.144269504088896`, relative `1.731e-15`.

### 1.8 What the bed must therefore do — three options, and the third is the one taken

1. **Label `a*` by (T1.11), never by `−ΔE‡`.** The splitting probability is the exit the
   bed actually means, it is exact, and it costs one solve.
2. **Carry the guard (T1.21) as a generated flag.** For every drawn instance compute
   `min_{r≠s} |ΔE‡_r − ΔE‡_s|` and `T · ln(max_r C_r / min_r C_r)` where `C_r` is the
   channel's series-parallel conductance prefactor. The barrier ranking is safe on that
   instance iff the first exceeds the second, and both sides are closed form.
3. **Deliberately include the inverted regime as a labelled subset.** A bed that only
   generates guard-passing instances cannot distinguish a head that learned the rate from
   a head that learned the barrier, because on that subset the two labels coincide. The
   inverted instances are the only ones that separate them, so they belong in the bed as
   a named hard split rather than being filtered out.

### 1.9 Where the exact route stops being exact

The closed forms above are exact in exact arithmetic. In float64 the conditioning of
`(I − Q)` grows like `e^{ΔE‡/T}`, and it is the conservation law `Σ_r q^{(r)} = 1` that
detects the loss first. `RUN`, three-basin instance:

```
    (T1.22)   T      cond(I − P_II)      |q̄_B + q̄_C − 1|
              0.200  1.014699e+05        5.954126e-12
              0.150  3.383898e+06        9.023327e-11
              0.120  1.106203e+08        1.987510e-09
              0.100  3.527883e+09        1.911775e-07
              0.070  2.325604e+13        4.684276e-04
              0.050  3.363399e+16        9.873707e-01     ← the answer is gone
```

The bed must not draw instances below `T ≈ 0.12` at this barrier scale in float64, and
`|Σ_r q^{(r)} − 1|` is the cheap runtime check that says so. Note the failure is
survivable in exactly the way `MATHEMATICS.md` §11 describes for the off-by-one defect:
at `T = 0.07` every returned number still lies in `[0,1]` and looks plausible.

---

## 2. TIER 2 — THE NULL DISTRIBUTION OF THE FAN-OUT DETECTOR

The detector engages a state head when `H_t − H̄_basin > τ_H`. Setting `τ_H` from
held-out percentiles gives it no stated false-positive rate. What follows gives one.

### 2.1 The statistic and the null

`n` observations fall into `K` categories with probabilities `p`; `p̂ = counts/n`; the
plug-in estimator is `Ĥ = −Σ_k p̂_k ln p̂_k`. Under the **single-basin null** the
trajectory is genuinely inside one basin, `p` is the basin-conditional distribution, no
decision point exists, and the detector statistic is

```
    (T2.1)   D := Ĥ_n − H̄_m ,   H̄_m the running baseline over m observations
```

Two regimes are carried throughout: `m → ∞` (baseline known, the clean case) and finite
`m` (baseline itself a plug-in, independent of the window).

### 2.2 Bias — the plug-in is biased downward, to order `1/n` and `1/n²`

```
    (T2.2)   E[Ĥ] = H − (K−1)/(2n) + (1 − Σ_k 1/p_k)/(12 n²) + O(n^{-3})
```

The first correction is Miller's; the Miller–Madow estimator `Ĥ + (K̂−1)/(2n)` removes
it, with `K̂` the number of observed categories. The second is Harris's. `CITED` as
standard; `RUN` as checked. Both terms are needed: at `K = 32`, `n = 2048` the
first-order formula sits **8.53 standard errors** from the Monte Carlo bias and the
two-term formula sits **1.86**.

`RUN`, 400 000 draws per row:

| `K` | `n` | `p` | MC bias | se | Miller only | two-term |
|---|---|---|---|---|---|---|
| 8 | 500 | uniform | −7.02261935e-03 | 5.96e-06 | −3.80 se | **−0.27 se** |
| 8 | 500 | skew | −6.90937466e-03 | 5.64e-05 | +1.61 se | **+2.75 se** |
| 32 | 2048 | uniform | −7.59437598e-03 | 3.05e-06 | −8.53 se | **−1.86 se** |
| 32 | 2048 | zipf | −7.60955381e-03 | 3.98e-05 | −1.03 se | **+0.03 se** |

### 2.3 Variance, and the degenerate case that governs the detector

```
    (T2.3)   √n (Ĥ − H) → N(0, V₁)      provided V₁ > 0
    (T2.4)   V₁ := Σ_k p_k (ln p_k + H)² = Var_p(−ln p_K)
```

`RUN`: `V₁/n` matches the Monte Carlo variance to relative `0.0114` (`K=8, n=500`, skew)
and `0.0097` (`K=32, n=2048`, zipf).

**`V₁ = 0` exactly when `p` is uniform on its support**, since `−ln p_k` is then constant.
A detector whose null is uniform-ish is precisely the case that matters here, so the
degenerate law is derived rather than skipped. Expanding `Ĥ` about `p` with
`δ = p̂ − p`, `Σδ_k = 0`:

```
    (T2.5)   Ĥ − H = −Σ_k (1 + ln p_k) δ_k − ½ Σ_k δ_k²/p_k + O(δ³)
                   = Z_n/√n − X²_n/(2n) + O_P(n^{-3/2})
             Z_n → N(0, V₁),   X²_n = n Σ_k δ_k²/p_k → χ²_{K−1}   (Pearson)
```

Both terms are always present; which one dominates is the whole question.

```
    (T2.6)   at p uniform:  ln p_k ≡ −ln K, so the first term vanishes IDENTICALLY and
             2n ( ln K − Ĥ )  →  χ²_{K−1}
```

The mean of (T2.6) is `K−1`, i.e. `E[Ĥ] ≈ ln K − (K−1)/(2n)` — the same first-order bias
as (T2.2), which is an internal consistency check rather than a coincidence. `RUN`:

```
    K =  8, n =  200   mean 7.052795  (want 7)    var 14.272004  (want 14)
    K = 32, n =  256   mean 31.780087 (want 31)   var 65.807685  (want 62)
```

`Z_n` is linear and `X²_n` quadratic in the same multinomial fluctuation, so their
covariance is a third moment and `O(n^{-1/2})`: they are asymptotically independent, which
is what makes (T2.10) below a convolution rather than a joint law.

### 2.4 `τ_H` from a target false-positive rate

Under (T2.1) with independent window and baseline, biases partially cancel and variances
add:

```
    (T2.7)   E[D] = −(K−1)/2 · (1/n − 1/m),    Var[D] = V₁ (1/n + 1/m)
```

Everything below is written at `m → ∞`; for finite `m` substitute `1/n → 1/n + 1/m` in
the variance and `1/n → 1/n − 1/m` in the bias.

**Three candidate formulas, and two of them fail catastrophically outside their regime.**

```
    (T2.8)   normal:  τ_H = z_{1−α} √(V₁/n) − (K−1)/(2n)
    (T2.9)   chi2:    τ_H = − q_α(χ²_{K−1}) / (2n)                (the V₁ = 0 law)
    (T2.10)  convolution, branch-free:
             τ_H = q_{1−α}( Z/√n − X²/(2n) ),  Z ~ N(0,V₁) ⟂ X² ~ χ²_{K−1}
    (T2.11)  parametric bootstrap:
             τ_H = q_{1−α}( Ĥ(Multinomial(n, p̂_basin)) − H̄ ),  B draws
```

`RUN`, achieved `α` at nominal `α = 0.01`, 400 000 draws, `se = 1.57e-4`:

| `K` | `n` | null | `V₁` | normal (T2.8) | chi2 (T2.9) | conv (T2.10) | boot (T2.11) |
|---|---|---|---|---|---|---|---|
| 8 | 200 | skew | 0.62937 | 0.00688 | **0.40756** | 0.00624 | **0.01022** |
| 32 | 256 | zipf | 1.28676 | 0.00709 | **0.32726** | 0.00607 | **0.01048** |
| 8 | 200 | uniform | 0.00000 | **0.56482** | 0.00992 | 0.00992 | **0.01039** |
| 32 | 256 | uniform | 0.00000 | **0.49473** | 0.00834 | 0.00836 | **0.01026** |

**The normal formula's failure at a uniform null is the headline and it is not small.**
At `V₁ = 0` it returns `τ_H = −(K−1)/(2n) < 0`, while `D = Ĥ − ln K ≤ 0` almost surely
with `−D = X²/(2n)`. The achieved rate is therefore
`P(χ²_{K−1} < K−1) ≈ 0.58` for small `K` — measured **0.565 and 0.495 against a nominal
0.01, fifty times over budget.** A detector whose null is near-uniform and whose `τ_H`
came from (T2.8) fires half the time on nothing.

The chi-square formula fails just as hard the other way at a non-degenerate null: **0.408
and 0.327**.

**(T2.10) is the formula to code when a closed form is wanted.** It is branch-free, it
reduces to (T2.8) when `n V₁` dominates and to (T2.9) at `V₁ = 0`, and it is never
anti-conservative in these runs. It is, however, **conservative by roughly 1.6× at
`n = 256`** (0.0061–0.0062 achieved against 0.01) because (T2.5) drops an `O(n^{-3/2})`
skewness term; the gap closes as `n^{-1/2}`.

**(T2.11) is the formula to actually ship.** It is one line of numpy, it needs no
quantile tables and no regime test, and it is calibrated in every case measured:
achieved `α` of 0.01022, 0.01048, 0.01039, 0.01026 against 0.01, all within 5 standard
errors. Its cost is `B·n` multinomial draws once per basin.

### 2.5 Power, and the minimum detectable gap

Let the planted high-entropy state have `H_1 = H_0 + Δ` and its own `V₁^{(1)}`. Writing
`W_p := Z/√n − X²/(2n)` for the law in (T2.5) at distribution `p`,

```
    (T2.12)  power(Δ) = P( W_{p₁} > τ_H − Δ )
    (T2.13)  power ≥ 1 − β   ⟺   Δ ≥ τ_H − q_β(W_{p₁})
    (T2.14)  Δ_min = q_{1−α}(W_{p₀}) − q_β(W_{p₁})
```

When both distributions are non-degenerate the chi-square means cancel exactly and
(T2.14) collapses to the textbook two-sample form:

```
    (T2.15)  Δ_min = ( z_{1−α} √V₁^{(0)} + z_{1−β} √V₁^{(1)} ) / √n           (V₁ > 0 both sides)
    (T2.16)  Δ_min = z_{1−α} √(V₁^{(0)}/n) + [ q_{1−β}(χ²_{K−1}) − (K−1) ] / (2n)
                                                             (uniform alternative, V₁^{(1)} = 0)
```

`RUN`, the power formula against Monte Carlo at `K = 32`, `n = 256`, null `= zipf(32)`
(`H₀ = 2.875940`, `V₀ = 1.286760`), `τ_H = 0.1080266`, alternatives `p ∝ zipf^λ`:

| `λ` | `Δ` | `V₁^{(1)}` | predicted | empirical |
|---|---|---|---|---|
| 0.00 | 0.589796 | 0.00000 | 1.00000 | 1.00000 |
| 0.25 | 0.563490 | 0.05690 | 1.00000 | 1.00000 |
| 0.50 | 0.467479 | 0.27782 | 1.00000 | 1.00000 |
| 0.90 | 0.122577 | 1.04511 | 0.24349 | 0.22936 |
| 1.00 | 0.000000 | 1.28676 | 0.00995 | 0.00619 |

### 2.6 THE HEADLINE NUMBER — `Δ_min` for power ≥ 0.99 at `α = 0.01`

Design point, stated so the number means something: `K = 32` categories, null the
basin-conditional `zipf(32)` (`H₀ = 2.875940`, `V₀ = 1.286760`), planted state
**uniform** over the 32 categories (`H₁ = ln 32 = 3.465736`, `V₁ = 0`), `α = 0.01`,
target power `0.99`.

`RUN`, (T2.14) by convolution quantile against the closed form (T2.16):

| `n` | `Δ_min` (nats) | closed form (T2.16) | rel |
|---|---|---|---|
| 64 | 0.521803 | 0.495421 | 0.0506 |
| 128 | 0.326403 | 0.316027 | 0.0318 |
| **256** | **0.210421** | 0.206321 | 0.0195 |
| 512 | 0.138689 | 0.137319 | 0.0099 |
| 1024 | 0.093131 | 0.092813 | 0.0034 |
| 2048 | 0.063925 | 0.063486 | 0.0069 |
| 4096 | 0.043842 | 0.043820 | 0.0005 |

**At the design point `n = 256`: `Δ_min = 0.210421` nats (0.3036 bits).** The natural
gap available to the bed — `ln 32 − H(zipf 32) = 0.589796` nats — clears it by
**2.80×**, and a must-fire built that way is measured at power `1.00000` over 200 000
draws.

`Δ_min` falls as `n^{-1/2}` while the leading `V₀` term dominates and crosses to `n^{-1}`
once it does not; the closed form (T2.16) is within 2 % of the exact quantile from
`n = 256` upward and within 5.1 % at `n = 64`.

### 2.7 `Δ_min` IS ALTERNATIVE-SPECIFIC, AND THE BED WILL GET THIS WRONG BY DEFAULT

The most likely design error is to read `Δ_min = 0.210421` as a statement about the
entropy gap alone and then plant any state with that gap. **It is not.** (T2.14) depends
on `V₁^{(1)}`, and a state reached by tempering the null (`p ∝ p₀^λ`) has a large `V₁`
at the same `Δ`.

`RUN`, the trap made concrete: a state planted at exactly `Δ = 0.210421` by tempering
carries `V₁^{(1)} = 0.86026` rather than `0`, and its measured detection probability is

```
    (T2.17)  empirical power = 0.74372          against the intended 0.99
```

A must-fire the detector catches three times in four is a flaky gate, and it would have
been built by following the headline number without the variance.

The self-consistent requirement solves `Δ = Δ_min(p₀, p_Δ, n)` as a fixed point. `RUN`,
`n = 256`, tempered family:

```
    (T2.18)  fixed point:  Δ = 0.293911 nats,  V₁^{(1)} = 0.67670
             power under the closed-form τ_H = 0.108027 :  0.98364
             power under the bootstrap  τ_H = 0.097241 :  0.98956
```

Two things fall out. A tempered planted state needs **`0.293911` nats, 1.40× the uniform
figure**, for the same power. And the closed-form threshold's conservatism on `α` is paid
for on the power side: 0.98364 against a nominal 0.99, restored to 0.98956 by the
calibrated bootstrap (T2.11). **The bootstrap threshold out-powers the closed form at the
same nominal `α` because it is not spending margin it did not need.**

### 2.8 The cheapest design, if the bed is willing to build both states uniform

If the null is uniform on `k₀` categories and the planted state uniform on `K > k₀`, both
sides are degenerate, both variances are zero, and the whole problem lives at second
order:

```
    (T2.19)  τ_H = − q_α(χ²_{k₀−1}) / (2n)
             Δ_min = [ q_{1−β}(χ²_{K−1}) − q_α(χ²_{k₀−1}) ] / (2n)      — O(1/n), not O(1/√n)
```

`RUN`, `α = 0.01`, power `0.99`:

| `K` | `k₀` | `n` | `τ_H` | `Δ_min` | actual gap `ln(K/k₀)` | headroom |
|---|---|---|---|---|---|---|
| 32 | 16 | 256 | −0.0102136 | 0.091723 | 0.693147 | 7.56× |
| 32 | 16 | 1024 | −0.0025534 | 0.022931 | 0.693147 | 30.23× |
| 32 | 8 | 256 | −0.0024200 | 0.099516 | 1.386294 | 13.93× |
| 32 | 24 | 1024 | −0.0049784 | 0.020506 | 0.287682 | 14.03× |

At `n = 256` this design needs `0.0917` nats against the `0.2104` of §2.6 — 2.3× cheaper
— and the required gap then shrinks as `1/n` rather than `1/√n`. **The price is
fragility**: the `1/n` scaling holds only while `V₁` is exactly zero, and any departure
from uniformity reinstates the `z_{1−α}√(V₀/n)` term, which dominates as soon as
`V₀ > (K−1)²/(4 n z²_{1−α})`. The bed should design to §2.6's non-degenerate number and
treat §2.8 as headroom, not as budget.

### 2.9 What an implementer codes

```python
# per basin, once:
V0  = sum(p*(log(p) + H)**2)                       # (T2.4)
tau = quantile(Hhat(multinomial(n, p, B)) - H, 1-alpha)     # (T2.11), B = 200_000
# closed-form cross-check, no draws of Hhat:            (T2.10)
tau_cf = quantile(normal(0, sqrt(V0/n), B) - chisquare(K-1, B)/(2*n), 1-alpha)
# must-fire sizing:                                     (T2.14)
delta_min = tau - quantile(W_alt, 1-power)
```

Never (T2.8) alone. It is the formula a percentile heuristic would be replaced with, and
at a near-uniform basin null it is fifty times worse than the heuristic it replaces.

---

## 3. WHAT THE CHECK SCRIPT FOUND, RECORDED RATHER THAN QUIETLY FIXED

Two assertions fired during development and both were the script being right.

**T2.0 rejected a mis-remembered reference constant.** The chi-square quantile check was
seeded with `chi2_ppf(0.01, 7) = 1.239042304265752` quoted from memory. The hand-rolled
incomplete-gamma solver returned `1.2390423055679296`, and the assertion fired at a
`1.3e-9` gap. The solver was correct: `scipy.stats.chi2.ppf(0.01, 7)` is
`1.2390423055679296`. The quantile is sensitive there because the `χ²_7` density at that
point is only `0.0245`, so a `3e-11` error in the CDF moves the quantile by `1.3e-9`.
The reference values in the script are now scipy-verified and the tolerance stayed at
`1e-9`. **The constant was the defect, not the tolerance.**

**T2.17 rejected a wrong test, and the wrong test was the interesting one.** The first
version of the power check planted an alternative at the `Δ_min` computed for a *uniform*
alternative and asserted power `≥ 0.98`. It measured `0.74372` and fired. The derivation
was right and the test was wrong: `Δ_min` depends on the alternative's `V₁`, not on `Δ`
alone. Rather than retune the test, the failing configuration is kept as (T2.17) — it now
asserts that the trap *does* reproduce — and the correct self-consistent check was added
as (T2.18). **The failure is the finding, and it is the single most load-bearing sentence
in §2 for whoever builds the must-fire.**

---

## 4. LIMITS

Collected here rather than scattered.

**Tier 1.** (T1.18)'s identification of the discrete prefactor `p_s/n_A^eff` with
Kramers' `ω_min ω_sad / 2πγ` is a continuum correspondence, `CITED` in form and not
checked numerically; only the exact ratio `cap/Z_A` and its series-resistance evaluation
are `RUN`. (T1.16)–(T1.17) are classical potential theory for reversible chains, used as
`CITED` and verified numerically on one instance at one temperature rather than proved
here. The Kirchhoff route (T1.15) is §11's step 4 transported from the unweighted graph
Laplacian to the reversible chain's conductance Laplacian; §11's own scope note stands —
the forest identity is the identity for the undamped walk, and this transport is checked
on the double well only. The counterexample of §1.7 is exact *because* the proposal graph
is a tree with structurally identical channels; on a non-tree the ratio (T1.20) becomes
an approximation and the crossover (T1.21) an estimate, and neither case is measured here.
The whole of Tier 1 assumes reversibility; a bed that generated a non-reversible `P`
would break (T1.14) onward and the (T1.4) residual is the check that catches it. Every
number is float64 on this machine, and (T1.22) records that the exact route loses the
answer entirely below `T ≈ 0.07` at this barrier scale. `‖N1 − series‖ = 6.480150e-12`
is the loosest Tier-1 agreement and it reflects accumulated roundoff over 20 matrix
squarings, not a modelling gap. `MATHEMATICS.md` §7's `λ₂` is reused as a name; nothing
here re-derives §7's engineered `α`-scaling or bears on the ladder rungs.

**Tier 2.** (T2.2)'s two-term expansion is checked at four `(K, n, p)` points and holds
within `2.75` standard errors at the loosest; it is not checked at `K/n` above `1/64`,
and at `K = 32, n = 256` the third-order term is visible (the T2.5 variance reads
`65.807685` against `2(K−1) = 62`, 6 % high). The asymptotic independence of `Z` and
`X²` behind (T2.10) is `O(n^{-1/2})`, and the residual is exactly the conservatism
measured at 0.0061 against 0.01 at `n = 256`; no Edgeworth correction is supplied. (T2.7)
assumes the window and the baseline are independent samples from the same `p`, which a
running baseline over an autocorrelated trajectory is not — the effective `m` under
autocorrelation is smaller than the step count and nothing here estimates it. (T2.11)
plugs in `p̂_basin` for `p`, so its calibration inherits the baseline's own estimation
error, which is not measured. Every `α` and power figure is 200 000–400 000 draws at one
seed (`20260831`); the binomial standard error on a nominal `α = 0.01` at 400 000 draws
is `1.57e-4`. The headline `Δ_min = 0.210421` is specific to `K = 32`, `n = 256`, a
`zipf(32)` null and a **uniform** alternative — §2.7 exists because that specificity is
the thing most likely to be dropped. Nothing in Tier 2 has been run against a real
next-token distribution; `zipf(32)` is a stand-in chosen for being non-degenerate and
nothing more.

**Both.** No part of this document has been run against any component of the
architecture, because none of it is built. The check script is the only artefact and it
exercises hand-built instances, not the bed.
