# V13 — X₂₈c prior art: the `κ = λ(1 − d)` consistency triangle

Fetched 2026-08-31. Contract item X₂₈c (escape rate `κ`, Lyapunov exponent `λ`,
fractal dimension `d` of the chaotic saddle, linked by `κ = λ(1 − d)` and
attributed to Kantz and Grassberger). Contract law forbids the relation in any
load-bearing claim before this file lands. Classes: `[V]` source fetched and
read this session, `[V-t]` title/metadata/abstract only, `[U]` owed and
unfetched. Counts at the foot of the Owed section. Extends
`V13_X27_G1_PRIOR_ART.md`, which recorded the relation only through Tél's
restatement (its E7, E8) and left the derivation owed as its U3; this file
closes what can be closed and states precisely what cannot.

---

## Verdict

The relation is real and published in exactly the contract's form for a
one-dimensional map — Drótos, Hernández-García and López state it as
`D₁ = 1 − κ/λ` with general form `κ = Σⱼ λⱼ(1 − D₁⁽ʲ⁾)` over the unstable
directions, Altmann, Portela and Tél print it as Eq. (29) under the name
Kantz–Grassberger relation, and Tél's own memoir prints `D₁⁽¹⁾ = 1 − κ/λ` — but
`d` in it is the **information** dimension `D₁` of the natural measure on the
saddle, and every fetched statement of the relation says so explicitly. Box
counting is **not** a legitimate substitute: `D₁ ≤ D₀` always holds, so
`λ(1 − D₀) ≤ λ(1 − D₁) = κ`, and the substitution under-predicts the escape
rate with a known sign and an unbounded relative size — on an exactly solvable
two-branch linear repeller the shortfall computed here is 0.49% at slope ratio
1.5, 4.83% at ratio 3, 9.56% at ratio 5 and 24.5% at ratio 500, with equality
only when `|f′|` is constant on the saddle. The bed's restriction to
one-dimensional maps is exactly what licenses the simple form: there is one
unstable direction, the sum collapses to a single term, and `D₁` is the whole
information dimension of the repeller rather than a partial dimension, whereas
the two-dimensional statement reads `D = 2 − κ/λ` for the unstable manifold and
requires a second exponent for the second partial dimension. The triangle is
**not algebraically circular** — independent estimators for all three exist and
have been run together on a one-dimensional map, Drótos et al. taking `κ` from
the survival count, `λ` from finite-time Lyapunov averaging and `D₁` from the
slope of a Rényi entropy, and obtaining `λ = 0.54`, `κ = 0.075`, `D₁ = 0.86`
against `1 − κ/λ = 0.8611`. It is nevertheless **not a test of physics**:
`κ = λ(1 − D₁)` is a theorem for hyperbolic systems, assembled below from
Demers's escape-rate variational principle and Bowen's equation, so agreement
certifies three estimators against each other and nothing about the bed — and
that same paper records choosing its estimator definitions so that they
"satisfy a generalized Kantz–Grassberger relation with the smallest deviation"
and substituting `1 − κ/λ` for `D₁` "since it can be computed more precisely",
both of which turn the check into an identity confirmed against itself. The
contract's worked figure is arithmetically exact (`0.69 × (1 − 0.8) = 0.138` to
the digit) and physically empty: it follows from the formula alone, and the
relation's own ordering constraint `λ > κ` is satisfied automatically for any
`d ∈ (0, 1)`.

---

## The primary source and its published form

### K1 `[V-t]` — Kantz, Grassberger, "Repellers, semi-attractors, and long-lived chaotic transients", *Physica D* **17**(1), 75–86 (August 1985), DOI 10.1016/0167-2789(85)90135-6

Crossref-verified this session: title, journal, volume 17, **issue 1** (X₂₇'s
E8 records issue 2; Crossref returns 1), pages 75–86, dated 1985-08. OpenAlex
returns `oa_status: closed`, `any_repository_has_fulltext: false`, and a single
location (the publisher landing page). Semantic Scholar returns
`openAccessPdf.status: CLOSED` and records the abstract as **elided by the
publisher**. The ScienceDirect full-text PDF endpoint returns a captcha
challenge page (1.2 MB of HTML, `<title>ScienceDirect</title>`, tokens
`Captcha`/`challenge`) rather than a PDF. Kantz's own MPIPKS publication pages
return HTTP 403.

**The exact form as published, in Kantz and Grassberger's own notation and
equation numbering, is therefore not in hand and remains `[U]` (U1).** What is
in hand is the relation as restated by three independent fetched sources that
agree with one another to the symbol, plus a rigorous reconstruction of it from
two fetched theorems (below). The paper is on record as proposing "formulas
relating the average life time of the transient to dimensions of the repeller,
and to Lyapunov exponents of the flow on it", tested numerically; Drótos et al.
record that its worked system is the **logistic map**, which is the same class
of bed X₂₈c proposes.

Delta: origin of the relation X₂₈c would use; **zero**, and X₂₈c is an
application on a different substrate, not an extension.

### K2 `[V]` — Altmann, Portela, Tél, "Leaking chaotic systems", *Rev. Mod. Phys.* **85**, 869–918 (2013), DOI 10.1103/RevModPhys.85.869, arXiv:1208.0254

Full PDF read this session. The relation, verbatim as the source's Eq. (29):

```
D₁⁽¹⁾ = 1 − κ / λ̄                                              (29)
```

introduced as: "It is an important result of transient chaos theory that this
**partial information dimension** can be expressed in a simple way by the escape
rate and the average continuous-time Lyapunov exponent `λ̄` on the chaotic
saddle"; and named: "This relation, the Kantz-Grassberger relation (Kantz and
Grassberger, 1985) states that the dimension observed along the unstable
direction deviates from 1 the more, the larger the ratio of the escape rate (a
characteristic of the global instability of the saddle) to the average Lyapunov
exponent (a characteristic of the local instability on the saddle) is."

The dimension bookkeeping the relation sits inside, verbatim:

```
D₀ = D₀⁽¹⁾ + D₀⁽²⁾ ,   D₁ = D₁⁽¹⁾ + D₁⁽²⁾                        (26)
D₀,₁⁽ᵘ⁾ = 1 + D₀,₁⁽²⁾ ,  D₀,₁⁽ˢ⁾ = 1 + D₀,₁⁽¹⁾                    (27)
Hamiltonian:  D₀,₁ = 2 D₀,₁⁽¹⁾ ,  D₀,₁⁽ᵘ⁾ = D₀,₁⁽ˢ⁾ = 1 + D₀,₁/2  (28)
```

and, immediately after Eq. (26), the inequality that decides item 2: "The saddle
might also contain very rarely visited, and thus atypical, regions.
Consequently, the information dimension `D₁` cannot be larger than the
box-counting dimension `D₀`, which naturally holds for the partial dimensions,
too: `D₁⁽ʲ⁾ ≤ D₀⁽ʲ⁾, j = 1, 2`. The value of the box-counting dimension is found
often to be close to that of the information dimension and it is then sufficient
to use only one of them."

Worked closed-form example (§ on the leaky baker map): stretching rate 3
everywhere, leak of height 1/3, giving `γ = ln(3/2)`, `λ̄ = ln 3`, and via
Eq. (29) `D₁⁽¹⁾ = ln 2 / ln 3` — "i.e., the unstable manifold of this leaky baker
map carries the structure of the classical triadic Cantor set". Recomputed here:
`ln(3/2) = 0.4054651081`, `ln 3 = 1.098612289`, `1 − κ/λ = 0.6309297536`,
`ln 2 / ln 3 = 0.6309297536` — agreement to ten figures. **This example is the
uniform-expansion case, in which `D₀ = D₁` exactly**; it is reproduced as the
degenerate member of the counterexample family below.

Delta: this entry supplies the relation, its name, its inequality
`D₁ ≤ D₀`, and a constant-slope worked case; **the difference from X₂₈c is that
X₂₈c intends to measure `D₀` and use it where Eq. (29) requires `D₁`.**

### K3 `[V]` — Tél, "The joy of transient chaos", *Chaos* **25**, 097619 (2015), DOI 10.1063/1.4917287

Full PDF read this session (author's copy, `theorphys.elte.hu/tel/pdf_pub/Chaos25.pdf`).
Escape law, the source's Eq. (1):

```
p(t) ~ e^{−κ t}                                                  (1)
```

with `κ` "independent of the choice of the initial distribution of the `N₀`
tracers", "a unique property of the chaotic saddle", and — the condition —
"Relation (1) is not necessarily valid from the very beginning, it holds after
some time `t₀` needed for the ensemble to come sufficiently close to the saddle".
Natural measure defined operationally: "distributing an ensemble of points around
the saddle and following those with long lifetimes. The frequency of visiting
different regions of the saddle by these trajectories defines the natural
distribution." Ordering constraint: "In chaotic cases `λ > κ`."

Absorption generalisation, the source's Eq. (13), and the attribution:

```
D₁⁽¹⁾ = 1 − (κ s̄ + ln R̄) / λ̄                                    (13)
```

"This result is an extension of the Kantz-Grassberger formula
`D₁⁽¹⁾ = 1 − κ/λ`, **valid for the partial dimension in usual transient chaos**."

Footnote 81, which fixes the second partial dimension and is *not* the naive
guess: "For the other partial dimension, one also needs to know the negative
Lyapunov exponent `λ′`, and finds it as `D₁⁽²⁾ = D₁⁽¹⁾ λ/λ′`. The information
dimension `D⁽¹⁾` of the saddle and `D₁ᶜ` of the c-measure are then
`D⁽¹⁾ = D₁⁽¹⁾ + D₁⁽²⁾` and `D₁ᶜ = 1 + D₁⁽²⁾`."

Delta: X₂₇'s E7 recorded Eq. (13) and the attribution; this fetch adds Eq. (1)'s
stated validity condition, the `λ > κ` ordering, and footnote 81. **The
difference from X₂₈c is zero for the relation and non-zero for the dimension: the
contract's `d` is written unqualified where the source writes `D₁`.**

### K4 `[V]` — Drótos, Hernández-García, López, "Local characterization of transient chaos on finite times in open systems", *J. Phys. Complexity* **2**(2), 025014 (2021), DOI 10.1088/2632-072x/abe5f7, arXiv:2011.02283

Full PDF read this session. **The most important entry in this file**: it is the
only fetched source that states the one-dimensional case explicitly, gives the
general form, and measures all three quantities independently on a
one-dimensional map.

Stated conditions, verbatim: "Its simplest form is only strictly valid in
dynamical systems with a sufficient degree of ergodicity and hyperbolicity,
which we will assume in the following."

The relation, verbatim, with the one-dimensional case named as such:

```
For the case of one-dimensional chaotic open maps, for which there is only one
Lyapunov exponent λ, and the chaotic saddle becomes a chaotic repeller of
information dimension D₁, it reads

    D₁ = 1 − κ/λ                                                 (1)

In higher-dimensional chaotic open systems, the relation gets generalized to

    κ = Σⱼ λⱼ (1 − D₁⁽ʲ⁾)

where the sum is over the saddle's unstable directions, of positive Lyapunov
exponents {λⱼ}, and {D₁⁽ʲ⁾} are the partial information dimensions along these
directions. Notably, Eq. (1) remains valid for reversible two-dimensional maps.
```

Definitions used: `λ` is "the largest positive Lyapunov exponent averaged on the
chaotic saddle **with respect to its so-called natural probability measure** to
which the distribution of trajectories trapped on the saddle converges for
infinitely long times"; `N ~ e^{−κt}` defines `κ`.

The heuristic derivation, quoted because it is exactly the derivation the
box-counting instrument would reproduce, and exactly where it fails: "the number
of such trajectories, `N`, which is proportional to `exp(−κt)`, should also
satisfy `N ∝ ε^{1−D}`, or `κt ≈ −(1−D) log ε`. Noting that the interval `ε` needs
to be smaller for longer integration times as `exp(−λt)` … we immediately arrive
at Eq. (1), **although a more refined argument [Kantz and Grassberger] is needed
to show that the proper dimension to be used is the information dimension `D₁`**."

Bed: the logistic map `x_{t+1} = 1 − a x_t²` (their Eq. 11) at
`a = 1.75487767` on `X = [−1, 1]`, escape to the period-3 attractor through
leaks of width `w = 0.04` centred on the period-3 points, "the logistic map **as
in [Kantz and Grassberger]**".

Independent measurement of all three, verbatim: "The asymptotic global values
have been computed by regarding the whole phase space as a single box. For `λ`
and `κ`, the formulae of Appendix B, (B.2) and (B.1) … have been utilized with
`T = 80`. For `D₁`, the slope of `H′₁` has been taken at `ε/l = 2⁻¹⁵` for
`T = 40`." Values: `λ = 0.54`, `κ = 0.075`, `D₁ = 0.86`. Recomputed here:
`1 − 0.075/0.54 = 0.86111111` against the independently measured `0.86`.

Two admissions that bear directly on circularity, both verbatim:

- estimator selection — "We have chosen definitions such that numerical
  estimates satisfy a generalized Kantz–Grassberger relation with the smallest
  deviation. Beyond basic criteria like convergence to the asymptotic
  definitions, the fulfillment of this relationship ensures that the quantities
  involved are meaningful."
- substitution — "(Numerically, both coordinates of the latter point are taken
  to be `1 − κ/λ` for this analysis, **since it can be computed more precisely
  than `D₁`**.)"

Non-hyperbolic degeneracy, verbatim: "a feature of most non-hyperbolic open
systems is that the exponential decay of the depletion function is replaced by
power-law decay at long times, so that the asymptotic escape rate is zero. But
still in this case the Kantz-Grassberger relationship is satisfied, because the
chaotic saddle becomes locally space filling in non-hyperbolic regions and its
dimension the one of the full domain."

Delta: this is X₂₈c's proposed measurement already performed, on a
one-dimensional map, with the relation checked and the deviations quantified.
**The difference is the substrate and nothing else** — the instrument, the
estimand and the check are the same object.

### K5 `[V]` — Aref *et al.*, "Frontiers of chaotic advection", *Rev. Mod. Phys.* **89**, 025007 (2017), DOI 10.1103/RevModPhys.89.025007, arXiv:1403.2953

Full PDF read this session (§II.C, "Hyperbolicity and the Grassberger–Kantz
relation"). This is the prior art for **the exact substitution X₂₈c intends**,
including its own caveat. Verbatim:

```
Open hyperbolic systems have exponential decay: … Q(t) ~ exp(−κ_e t). κ_e is the
escape rate of the flow. It satisfies κ_e < λ … The fractal dimension D of the
unstable manifold, the Lyapunov exponent λ and the escape rate κ_e are related
by the Grassberger–Kantz formula (Kantz and Grassberger, 1985):

    D = 2 − κ_e/λ                                                (12)

More rigorously, we should have D₁, the information dimension (Falconer, 2003),
instead of the box-counting dimension D in Eq. (12), but since D and D₁ are
almost always very close for open flows, this approximation is valid in most
cases.
```

Non-hyperbolic falsifier, same source: `N(t) ~ t^{−γ}` (their Eq. 13), and then
`D = 2` for the manifolds, with an *effective* dimension
`D_eff(ε) = 2 − d ln f(ε)/d ln ε` (their Eq. 14) valid only on a finite range
`ε₁ < ε < ε₂`; measured `D_eff = 1.86` and `1.98` in two different cantori of one
system, i.e. **the "dimension" becomes position-dependent** once hyperbolicity
fails.

Delta: the review authorises the substitution X₂₈c wants, and does so on an
unquantified assertion ("almost always very close … valid in most cases") with no
bound and no error term. **The difference is that the assertion is a heuristic
about open *flows* in fluid mechanics, not a theorem, and the direction and size
of the error are computable — see the next section.**

### K6 `[V]` — Demers, "Dispersing billiards with small holes", chapter in *Ergodic Theory, Open Dynamics, and Coherent Structures* (Springer PROMS 70, 2014); author's copy `faculty.fairfield.edu/mdemers/research/2014.05.08.holes.chapter.v4.pdf`

Full PDF read this session. Supplies the rigorous half of the relation — the
escape-rate formula — with its definitions. Verbatim, §1.1:

```
Rates of escape.  ρ(µ) = lim_{n→∞} (1/n) log µ(M̊ₙ),             (1)
the exponential rate of escape with respect to µ is −ρ(µ).

Pressure on the survivor set.  P_C = sup_{ν∈C} P_ν  where
    P_ν = h_ν(T) − ∫ χ⁺(T) dν ,
h_ν(T) the Kolmogorov-Sinai entropy and χ⁺(T) the sum of positive Lyapunov
exponents, counted with multiplicity.
```

and the naming convention: "We say the open system satisfies a variational
principal if `ρ(µ) = P_C` … If there is an invariant measure `ν ∈ C` such that
`ρ(µ) = P_ν`, we say that `ν` satisfies an **escape rate formula**." Theorem 3
states it: `ρ(m) = log λ_H = sup_{ν∈G_H} {h_ν(T) − χ⁺_ν(T)}`, with the supremum
attained: `ρ(m) = h_{ν_H}(T) − χ⁺_{ν_H}(T)`.

In the notation of this file, with `κ = −ρ(m)`:

```
κ = χ⁺_{ν₁} − h_{ν₁} = inf_ν { χ⁺_ν − h_ν } ,   ν₁ = natural measure
```

Delta: this is `κ = λ − h` proved. **The difference from the contract is that the
contract's `κ = λ(1 − d)` is this theorem composed with a dimension formula, and
the contract states neither hypothesis.**

### K7 `[V]` — Climenhaga, "Bowen's equation in the non-uniform setting", arXiv:0908.4126v2 (2010); quoting Ruelle, "Repellers for real analytic maps", *Ergodic Theory Dynam. Systems* **2**, 99–107 (1982), DOI 10.1017/S0143385700009603

Full PDF read this session. Supplies the other rigorous half — the box-counting
side. Bowen's equation, verbatim as the source's Eq. (1.1) with Ruelle's
Theorem 1.1:

```
    P_J(−t φ) = 0 ,     φ(z) = log |f′(z)|                       (1.1)
```

"the Hausdorff dimension `t = dim_H J` is the unique root of the equation (1.1),
where `P_J` is the topological pressure of the map `f : J → J`, and `φ` is the
geometric potential". Ruelle's conditions, verbatim: `f : V → M` is `C^{1+ε}`
and conformal, and `J ⊂ V` is a repeller — (1) `J` compact; (2) `J` maximal,
`J = {x ∈ V | fⁿ(x) ∈ V for all n > 0}`; (3) `f` topologically mixing on `J`;
(4) `f` uniformly expanding on `J`. Barreira and Schmeling extend the same root
characterisation to arbitrary subsets `Z ⊂ J`, compact or not.

Delta: on a one-dimensional map the bed's saddle *is* a conformal repeller, so
`D₀` is this root while `D₁` is the natural measure's `h/λ`. **The difference is
that these are two different functionals of the same pressure function, and the
contract's `d` does not say which.**

### K8 `[V-t]` — Lai, Tél, *Transient Chaos: Complex Dynamics on Finite Time Scales*, Springer AMS 173 (2011), DOI 10.1007/978-1-4419-6987-3

Cited by K2, K3 and K4 as the monograph home of Eqs. (26)–(29) and of the
`D₀`/`D₁` distinction. Not fetched.

### K9 `[V-t]` — Ruelle, "Repellers for real analytic maps", *ETDS* **2**, 99–107 (March 1982), DOI 10.1017/S0143385700009603

Crossref-verified this session. The source of Theorem 1.1 as quoted by K7
(Climenhaga's ref [22], "Proposition 4"). Full text not fetched.

### K10 `[V-t]` — Gatzouras, Peres, "Invariant measures of full dimension for some expanding maps", *ETDS* **17**(1), 147–167 (1997)

Identifier from K7's reference list [7]; DOI not resolved (a plausible DOI guess
returned a different paper, so no DOI is asserted). The literature on **measures
of full dimension** — the object whose coincidence with the natural measure is
exactly the `D₀ = D₁` condition. Not fetched.

---

## Which dimension `d` is, and what the substitution costs

**`d` is `D₁`, the information dimension of the natural (conditionally
invariant) measure on the saddle. `κ = λ(1 − D₀)` is not valid in general.**
Four fetched sources say the dimension is `D₁`: K2 Eq. (29) ("this partial
**information** dimension"), K3 (`D₁⁽¹⁾`, and the whole paper's `D₁` notation),
K4 Eq. (1) ("a chaotic repeller of **information** dimension `D₁`") and K4's
derivation note ("a more refined argument is needed to show that the proper
dimension to be used is the information dimension `D₁`"). One fetched source, K5
Eq. (12), writes the box-counting `D` and then retracts it in the next sentence.

### The inequality, and its sign

K2 states `D₁⁽ʲ⁾ ≤ D₀⁽ʲ⁾` without proof. It follows from the two fetched
theorems. Write `P(t) = sup_ν {h_ν − t λ_ν}`, the topological pressure of
`−t log|f′|` (K7's `P_J(−tφ)`; the variational form is the standard variational
principle, `[U]` here — U3).

```
K6 (escape-rate formula) :  P(1) = −κ ,  attained at the natural measure ν₁ ,
                            with  P′(1) = −λ₁ ,  h₁ = λ₁ − κ
K7 (Bowen's equation)    :  P(D₀) = 0
Dimension of ν₁          :  D₁ = h₁/λ₁ = 1 − κ/λ₁                (U4)
```

`P` is convex and decreasing, so it lies above its tangent at `t = 1`, whose
root is `1 − κ/λ₁ = D₁`; the root of `P` itself is therefore no smaller:

```
    D₁ ≤ D₀ ,      with equality iff P is affine on [D₁, 1]
    λ(1 − D₀) ≤ λ(1 − D₁) = κ
```

**Box counting under-predicts `κ`.** Read the other way: if `κ` and `λ` are
measured and `d` is predicted as `1 − κ/λ`, a box-counted `D₀` comes out
systematically *too large*. The sign is fixed; a shortfall is the expected
result, not evidence of a broken relation.

### When `D₀ = D₁`

`P` is affine exactly when the geometric potential `log|f′|` is constant on the
saddle up to an additive constant and a coboundary — equivalently, when the
natural measure is also the measure of full dimension. The sufficient case
`|f′| = const` is realised in K2's own leaky-baker example (stretching rate 3
everywhere, `D₁⁽¹⁾ = ln 2/ln 3`, which is also the box dimension of the triadic
Cantor set). The general "if and only if" statement at equation level is `[U]`
(U2).

### The exact cost of the substitution, computed this session

Test family: a one-dimensional piecewise-linear map with two full branches of
slopes `s₁, s₂ > 1` mapping two disjoint intervals of lengths `r_i = 1/s_i` onto
the unit interval, everything else escaping. Lebesgue is conditionally invariant
with eigenvalue `r₁ + r₂`, the natural measure is Bernoulli with
`p_i = r_i/(r₁+r₂)`, and every quantity is closed-form:

```
κ  = −ln(r₁ + r₂)
λ  = Σ p_i ln s_i
h  = −Σ p_i ln p_i
D₁ = h/λ                       D₀ = root of  r₁^{D₀} + r₂^{D₀} = 1   (Bowen/Moran)
```

Evaluated at 20-digit precision (`mpmath`). The column `λ(1−D₁)` reproduced `κ`
to 10⁻¹⁹ relative in every row, which is the self-check on the arithmetic.

| `s₁, s₂` | `κ` | `λ` | `D₁` | `D₀` | `D₀ − D₁` | `λ(1 − D₀)` | shortfall vs `κ` |
|---|---|---|---|---|---|---|---|
| 3, 3 | 0.4054651 | 1.098612 | 0.6309298 | 0.6309298 | 1.1e−21 | 0.4054651 | **0.000%** |
| 2, 3 | 0.1823216 | 0.8553332 | 0.7868415 | 0.7878849 | 0.001043 | 0.1814291 | 0.490% |
| 3, 9 | 0.8109302 | 1.373265 | 0.4094876 | 0.4380179 | 0.02853 | 0.7717506 | 4.831% |
| 2, 10 | 0.5108256 | 0.9613868 | 0.4686576 | 0.5194632 | 0.05081 | 0.4619817 | 9.562% |
| 3, 100 | 1.069053 | 1.200745 | 0.1096749 | 0.2851601 | 0.1755 | 0.8583405 | 19.71% |
| 2, 1000 | 0.6911492 | 0.7055516 | 0.02041298 | 0.2606075 | 0.2402 | 0.5216795 | 24.52% |

Three readings:

1. The first row is K2's leaky-baker example (uniform slope 3, two surviving
   branches): `κ = ln(3/2)`, `λ = ln 3`, `D₀ = D₁ = ln 2/ln 3`. **The relation
   holds with box counting only because the measure is uniform.**
2. A slope ratio of 3 — an entirely ordinary non-uniformity for a trained
   system's return map — already costs 4.8%. A ratio of 5 costs 9.6%. There is
   no small-parameter regime in which the substitution is safe by default; K5's
   "almost always very close" is a fluid-mechanics heuristic, not a bound.
3. The error is unbounded in relative terms: `D₁ → 0` while `D₀` stays positive
   as the branch slopes separate.

### The instrument fix

The X₂₇a box-counting instrument is one line away from being the right
instrument. Box counting computes `ln N(ε)/ln(1/ε)`; the information dimension
is `Σᵢ pᵢ ln pᵢ / ln ε` over the **same boxes**, with `pᵢ` the visitation
frequency of long-lived trajectories. K4 does exactly this (their Eq. (9), the
slope of the mean generalised Rényi entropy `H′_q` against `ln ε`, evaluated at
`q = 1`). No new sampling, no new traversal — the counts are already there, they
only need to be weighted instead of thresholded.

---

## Partial dimensions and the one-dimensional special case

**The bed's restriction to one-dimensional maps is precisely what makes the
simple form legitimate.** The general relation, K4 verbatim:

```
    κ = Σⱼ λⱼ (1 − D₁⁽ʲ⁾)
```

summed over the saddle's unstable directions, with `{D₁⁽ʲ⁾}` the partial
information dimensions along those directions. In one dimension there is one
positive exponent, one term, and — K4 again — "the chaotic saddle becomes a
chaotic repeller of information dimension `D₁`", so the partial dimension *is*
the dimension of the set, with no decomposition to get wrong. That is the
contract's form.

Everything above one dimension requires more than `κ` and `λ`:

- K2 Eq. (26): the saddle's dimensions are sums, `D₁ = D₁⁽¹⁾ + D₁⁽²⁾`; Eq. (29)
  gives only `D₁⁽¹⁾`.
- K3 footnote 81: the stable-direction partial dimension needs the **negative**
  exponent, `D₁⁽²⁾ = D₁⁽¹⁾ λ/λ′`. Note this is *not* `1 − κ/|λ′|`; the naive
  symmetry guess is wrong, because the time-reversed dynamics has a different
  escape rate.
- K2 Eq. (27): the manifolds carry one smooth direction,
  `D₁⁽ᵘ⁾ = 1 + D₁⁽²⁾`, which is why K5's Eq. (12) reads `D = 2 − κ_e/λ` — that
  `2` is the embedding dimension of a two-dimensional flow, not a universal
  constant.
- K4: "Eq. (1) remains valid for reversible two-dimensional maps", the case
  where `|λ′| = λ` collapses the two partials together (K2 Eq. 28).

**Consequence for the contract.** `κ = λ(1 − d)` written with a bare `d` is
correct on a one-dimensional-map bed and silently wrong the moment the bed
acquires a second direction, where the same string of symbols denotes a partial
dimension and needs the negative exponent to be completed. Any migration of X₂₈c
off the 1-D bed invalidates the form, not just the numbers.

---

## Independence of `κ`, `λ` and `d`

**The triangle is not algebraically circular. Each of the three has a published
estimator that does not pass through the other two, and all three have been run
together on a one-dimensional map.**

| quantity | independent route | source |
|---|---|---|
| `κ` | fit `p(t) ~ e^{−κt}` to the survivor count; pure counting, no tangent-space and no geometry | K3 Eq. (1); K4 Eq. (6) |
| `κ` | leading eigenvalue of the Frobenius–Perron operator restricted to survivors, `µ(M̊ₙ) = λⁿ` | K6 Eq. (2); K2 §III |
| `κ` | cycle expansion over unstable periodic orbits that never hit the hole, `e^{−nγ} = Σᵢ 1/|Λ(Γᵢ⁽ᵒᵘᵗˢⁱᵈᵉ⁾)|` | K2 Eq. (33) |
| `λ` | tangent-space averaging along trajectories held on the saddle | K4 App. B (B.2); trajectories from K11/K12 |
| `d = D₁` | slope of the mean generalised Rényi entropy `H′₁` against `ln ε` on the reconstructed saddle | K4 Eq. (9) |
| `d = D₀` | box count of the reconstructed saddle; or `α = D − D₀` via the uncertainty exponent | K5; X₂₇'s E1–E4 |

`[V-t]` supporting entries for saddle reconstruction, both Crossref-verified this
session, neither fetched:

- **K11 `[V-t]`** — Nusse, Yorke, "A procedure for finding numerical trajectories
  on chaotic saddles", *Physica D* **36**, 137–156 (June 1989),
  DOI 10.1016/0167-2789(89)90253-4. The PIM-triple method.
- **K12 `[V-t]`** — Sweet, Nusse, Yorke, "Stagger-and-Step Method: Detecting and
  Computing Chaotic Saddles in Higher Dimensions", *Phys. Rev. Lett.* **86**,
  2261–2264 (2001-03-12), DOI 10.1103/PhysRevLett.86.2261.

### The demonstration that it can be done

K4 measures all three separately on the logistic map and prints the comparison:
`λ = 0.54` and `κ = 0.075` from their Appendix B formulae at `T = 80`, `D₁ = 0.86`
from the entropy slope at `ε/l = 2⁻¹⁵`, `T = 40`. Recomputed here,
`1 − κ/λ = 0.86111111` against a measured `0.86`. So the answer to the contract's
question, stated unambiguously: **two of the three are not "only ever obtained
via this relation", and "measure all three independently and check the identity"
is a procedure that exists and has been executed.**

### Why it still certifies nothing about the bed

Three separate reasons, in order of severity.

1. **`κ = λ(1 − D₁)` is a theorem, not an empirical law.** K6 proves
   `κ = λ − h` for the natural measure; the dimension formula `D₁ = h/λ` (U4)
   completes it. For a hyperbolic conformal repeller — which any well-behaved
   1-D expanding bed is — the identity holds by construction. A GREEN therefore
   certifies that three estimators are mutually consistent; it cannot certify
   anything the architecture does. A RED localises to exactly one of: broken
   hyperbolicity, non-exponential decay, the wrong measure in `λ`, finite-`ε`
   bias, or an estimator bug. That is a useful instrument-calibration gate. It is
   not a capability claim, and the contract must not let a GREEN be written as
   one.

2. **The literature's actual practice is to derive, not to check.** K4 substitutes
   `1 − κ/λ` for `D₁` in its own analysis "since it can be computed more precisely
   than `D₁`". K2 records a "nontrivial closed-system approximation `D*₁⁽¹⁾` of
   the information dimension … obtained from the Kantz-Grassberger relation (29)".
   `D₁` is the noisiest of the three and the one habitually eliminated. If X₂₈c's
   pipeline ever falls back on `1 − κ/λ` for `d` — including as a sanity default,
   a seed, or a fill for boxes where the entropy slope fails to converge — the
   triangle becomes an identity confirmed against itself and must not be reported
   as a check.

3. **Estimator tuning is the live failure mode, and it is documented.** K4 states
   plainly: "We have chosen definitions such that numerical estimates satisfy a
   generalized Kantz–Grassberger relation with the smallest deviation." Selecting
   estimator definitions by agreement with the relation makes subsequent
   agreement uninformative. X₂₈c must pre-register the three estimators —
   window, fit range, `ε` decades, discard of the initial transient `t₀`, box
   weighting — **before** the comparison, and record any later change as a
   defect.

### And with box counting it is not this check at all

Substituting `D₀` changes the estimand. `λ(1 − D₀)` and `κ` are two different
functionals of the pressure function — the tangent at `t = 1` versus the root —
so comparing them tests whether the saddle's natural measure is uniform, with a
guaranteed one-sided outcome (`λ(1 − D₀) ≤ κ`). A shortfall is the *expected*
result and measures multifractality, not architecture. Reported as a
"consistency triangle" it would read as a failed check of the Kantz–Grassberger
relation, which it is not.

### One degenerate case that must be gated

K4: in non-hyperbolic systems "still in this case the Kantz-Grassberger
relationship is satisfied, because the chaotic saddle becomes locally space
filling … and its dimension the one of the full domain", i.e. `κ → 0` and
`d → 1` and the relation degenerates to `0 = 0`. K5 supplies the same warning
from the other side: algebraic decay `N(t) ~ t^{−γ}`, `D = 2`, and a
position-dependent effective dimension. **A triangle that "passes" while `κ̂` is
drifting toward zero has passed vacuously.** The exponential fit needs a
goodness-of-fit gate and `κ̂` needs a floor before the identity is read — the
same gate X₂₇'s kill-clause already demanded for X₂₇d.

---

## Arithmetic audit of the contract's worked figure

Computed this session at 20 digits:

| quantity | value |
|---|---|
| `0.69 × (1 − 0.8)` | `0.138` exactly |
| `κ/λ = 0.138/0.69` | `0.2` exactly, i.e. `1 − d` |
| `λ` implied by `κ = 0.138`, `d = 0.8` | `0.69` exactly |
| `d` implied by `κ = 0.138`, `λ = ln 2` | `0.80090808` |
| `ln 2` | `0.6931471805599453` |
| `ln 2 × (1 − 0.8)` | `0.1386294361119891` |

**Consistent, and it tests the arithmetic and not the physics.** Four
observations:

1. `0.69 × 0.2 = 0.138` is exact decimal arithmetic. It follows from the formula
   alone; any `(λ, d)` pair produces a `κ` satisfying it, so the figure carries
   no information about the bed. The contract's own framing in
   `V13_DAG_TASKLIST.md:382` already says this and is correct.
2. The relation's only published side constraint, `λ > κ` (K3), is satisfied
   automatically: `κ/λ = 1 − d`, so `λ > κ` holds for every `d > 0`. The figure
   cannot violate the constraint and therefore cannot test it either.
3. `λ = 0.69` is `ln 2` to two decimals — the Lyapunov exponent of a
   constant-slope-2 map. A constant-slope map is exactly the case where
   `D₀ = D₁` and the box-counting substitution is harmless. **The worked figure
   sits on the one line in parameter space along which the report's principal
   finding is invisible.** If it is pre-registered as an expected value, it
   pre-registers the degenerate case.
4. Under the exact reading `λ = ln 2`, `d = 0.8` gives `κ = 0.1386294`, and
   holding `κ = 0.138` instead forces `d = 0.8009`. The 3-significant-figure
   rounding is internally consistent; nothing turns on it.

---

## Owed

**U1 — Kantz, Grassberger (1985) full text.** DOI 10.1016/0167-2789(85)90135-6.
Owed: the relation in the authors' own notation and equation numbering, the
hypotheses they attach (hyperbolicity, ergodicity, which measure `λ` is averaged
over), the derivation that upgrades `D` to `D₁` (K4 refers to it as "a more
refined argument" without reproducing it), and the numerical tolerances of their
own logistic-map test. Closed at the publisher (Crossref-verified metadata,
OpenAlex `oa_status: closed`, Semantic Scholar `CLOSED` with the abstract elided);
the ScienceDirect PDF endpoint returns a captcha page; Kantz's institutional
publication pages return 403. This is the same item as X₂₇'s U3 and it remains
open. Contract law is satisfied at *equation* level by three concordant fetched
restatements (K2 Eq. 29, K3, K4 Eq. 1) but **not** at *primary-source* level;
any sentence claiming the fetch reached Kantz and Grassberger's own text is
false.

**U2 — An equation-level source for the equality condition `D₀ = D₁`.** The
"if and only if" — that the natural measure is the measure of full dimension
exactly when `log|f′|` is cohomologous to a constant — is stated here as a
consequence of K6 and K7 plus the standard uniqueness of equilibrium states, and
verified numerically on the two-branch family, but no source was fetched that
states it. Nearest unfetched candidates: K10 (Gatzouras & Peres) and the Livšic /
Bowen rigidity literature. Owed before the report's equality condition is quoted
as published.

**U3 — The variational principle for topological pressure,
`P(φ) = sup_ν {h_ν + ∫φ dν}`.** Used above to identify `P(1) = −κ` with K6's
supremum and `P(D₀) = 0` with K7's root. Textbook (Walters, *An Introduction to
Ergodic Theory*, Springer GTM 79, ISBN 978-0-387-95152-2, Thm 9.10). Not fetched.

**U4 — The dimension formula `dim ν = h_ν/λ_ν` for an ergodic measure under a
conformal expanding map.** The second leg of the derivation of
`D₁ = 1 − κ/λ`. Attributed in the literature to Young (1982) and Ledrappier;
K3's reference list carries Ledrappier and Young, *Commun. Math. Phys.* **117**,
529 (1988). Not fetched at equation level.

**U5 — Kadanoff, Tang, "Escape from strange repellers", *PNAS* **81**(4),
1276–1279 (1984), DOI 10.1073/pnas.81.4.1276.** `[V-t]`, abstract read via PMC
(PMC344812); the full text is available only as page images and was not OCR'd
this session. Owed if the cycle-expansion route to `κ` (K2 Eq. 33) is used as the
independent estimator, since this is its origin.

**U6 — Gaspard, Dorfman, "Chaotic scattering theory, thermodynamic formalism,
and transport coefficients", arXiv:chao-dyn/9504014.** `[V-t]`, abstract read:
"relates the escape rate to the difference between the sum of the positive
Lyapunov exponents and the Kolmogorov-Sinai entropy for the fractal set of phase
space trajectories which are trapped forever in the open region" — the
escape-rate formula in the many-degree-of-freedom setting. The PDF downloaded
and extracted with `pymupdf`, but its Type-3 font encoding **drops every digit**
(`B-00`, `(April,)`), so no equation number or numerical value from it is
quotable. Owed from a clean source if the many-body form is ever needed.

**Counts: `[V]` 6 · `[V-t]` 6 · `[U]` 6.**
`[V]`: K2, K3, K4, K5, K6, K7.
`[V-t]`: K1, K8, K9, K10, K11, K12.
`[U]`: U1–U6.

---

## Kill-clause exposure

**Unavailable as written: any claim using `κ = λ(1 − d)` with `d` from box
counting.** The relation requires `D₁`. Substituting `D₀` gives a one-sided,
systematically low prediction of `κ` whose error is 4.8% at a slope ratio of 3
and 24.5% at a slope ratio of 500 on the exactly solvable family above. Every
downstream number that reads `d` off the X₂₇a instrument and puts it in this
formula is unavailable until the instrument is switched to the entropy-weighted
form (K4 Eq. 9). The switch is cheap — same boxes, weighted counts — so the
finding is a required repair, not a kill.

**Unavailable as written: "measure all three independently and check the
identity" reported as a validation of the architecture.** The identity is a
theorem for hyperbolic conformal repellers, assembled from K6 and K7. A GREEN is
an estimator-calibration result. It may be reported as such — it is a genuine
and useful gate, and it catches real defects — but no contract sentence may
present it as evidence that the bed does something, and no scoreboard line may
bank it as a capability. This is the layer's most dangerous item precisely
because the check *works*: passing it feels like a measurement.

**Unavailable without a pre-registration: the triangle at all.** K4 documents
that its estimator definitions were chosen to minimise deviation from the
relation. The same freedom exists here — fit window, `ε` decades, discard of the
initial transient `t₀`, box weighting, treatment of boxes with no escapes. The
triangle must not be run until those five choices are fixed on the record, and
any post-hoc change to them voids the result.

**Unavailable without a hyperbolicity gate: any triangle GREEN.** K4: in the
non-hyperbolic case `κ → 0`, `d → 1`, and the relation is satisfied trivially.
K5: decay becomes algebraic and the dimension becomes position-dependent, with
`D_eff` measured at 1.86 and 1.98 at two locations in one system. A GREEN
obtained while `κ̂` is small and `d̂` is near 1 is vacuous. The exponential fit
needs a goodness-of-fit test and `κ̂` a floor, both pre-registered, before the
identity is read.

**Unavailable off the 1-D bed: the form itself.** `κ = λ(1 − d)` is the
single-unstable-direction case of `κ = Σⱼ λⱼ(1 − D₁⁽ʲ⁾)`. On any bed with two
unstable directions, or on a two-dimensional map where `d` becomes a partial
dimension requiring the negative exponent (`D₁⁽²⁾ = D₁⁽¹⁾ λ/λ′`, K3 fn. 81), the
contract's string of symbols is wrong rather than imprecise. BED-1 must record
that the relation is licensed by the 1-D restriction and that lifting the
restriction retires the relation.

**Survives: the arithmetic.** `0.69 × (1 − 0.8) = 0.138` is exact, admissible
(`λ > κ` holds automatically), and empty. It may be printed as a worked example
of the formula. It may not be printed as a prediction, and if pre-registered as
an expected value it pre-registers `λ ≈ ln 2`, the constant-slope case in which
this file's central finding cannot show up.
