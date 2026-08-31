# V13 — X₂₇ G1 prior art: the transient-fractal layer

Fetched 2026-08-31. Contract item X₂₇ (α-instrument, exit-prediction ceiling,
`(H, α̂)` quadrants, escape-rate κ). Contract law forbids any X₂₇ claim before
this file lands. Classes: `[V]` source fetched and read this session, `[V-t]`
title/abstract/metadata only, `[U]` owed and unfetched. Counts at the foot of
the Owed section.

---

## Verdict

No published theorem states that no predictor can exceed exit accuracy
`1 − c·f(ε)` at state-resolution ε, in nonlinear dynamics or in learning
theory; the nearest published ceilings are the entropy-and-Fano bound of Song
et al. (*Science* **327**, 1018) and its exact restatement as the Bayes error
rate by Xu et al. (*EPL* **141**, 61003), neither of which carries an ε-scaling,
and no source fixes `c`. Under the Grebogi–Ott–Yorke definition of `f(ε)` as
the ε-uncertain phase-space fraction the claim is not merely unproved but false
for every `c > 0`, because an ε-ball counted uncertain may be arbitrarily
imbalanced; it becomes a three-line corollary of the Bayes error with `c = 1/2`
only if `f` is redefined as pairwise disagreement inside the ball and exits are
binary. The uncertainty exponent is already in use in machine learning: Ly &
Gong (arXiv:2510.05606) measure `φ = 0.0126 ± 0.0002` on a minimal network and
`φ = 0.000 ± 0.002` on VGG-12 with exactly the `f(ε) ∝ ε^φ` instrument X₂₇a
proposes, so the instrument is not new — though nothing in language modelling
uses it. The contract's `α = 0.2` figure is arithmetically consistent in its
exponent (`0.398 / 0.251 = 1.58566` against `10^{0.2} = 1.58489`, a `5×10⁻⁴`
rounding discrepancy) but its levels are the unit-prefactor evaluation:
`(10⁻²)^{0.2} = 0.398107` and `(10⁻³)^{0.2} = 0.251189` to six figures. The
published law is `f(ε) ~ C·(ε/L)^α` with a system-dependent prefactor — Daza et
al. print it explicitly as `n_k/ñ` — so setting `C = 1` asserts that every state
is uncertain at unit resolution, which is a claim about the bed and not a
consequence of the scaling law. `α = 0.2` is itself the forced damped
pendulum's number (`D₀ ≅ 1.8`, `D = 2`) from Ott's Scholarpedia article, so
pre-registering it imports a pendulum's boundary dimension into a claim about
this bed.

---

## Fractal basin boundaries and the uncertainty exponent

### E1 `[V]` — Ott, "Basin of attraction", *Scholarpedia* **1**(8):1701, doi:10.4249/scholarpedia.1701 (rev. #170495)

Defining equation, quoted at the level of the source:

```
ρ(ε) ~ ε^α ,      α = D − D₀
```

where ρ(ε) is "the probability that the uncertainty ε could cause us to make a
mistake in a determination of the attractor that the orbit goes to",
geometrically "what fraction of the area … is within a distance ε of the basin
boundary"; `D` is the dimension of initial-condition space and `D₀` the
box-counting dimension of the basin boundary. Worked system: the forced damped
pendulum

```
d²θ/dt² + 0.1 dθ/dt + sin θ = 2.1 cos t
```

for which `D₀ ≅ 1.8`, `D = 2`, hence `α ≅ 0.2`. The source's own consequence
statement: "if `α = 0.2`, to reduce `ρ(ε)` by a factor of 10, the uncertainty
`ε` would have to be reduced by a factor of `10⁵`" — i.e. `10^{1/α}`.

Delta: Scholarpedia defines ρ(ε) as an error probability for the attractor
label of a 2-D flow; X₂₇a computes the same functional on the bed's exit
basins with `f(ε) = P(final basin changes)`; **the difference is zero at the
level of definition** and non-zero only in substrate and in the perturbation
protocol (Scholarpedia's ρ is an area fraction within ε of the boundary, X₂₇a's
`f` is a resampled perturbation probability — see the audit below, the two are
not interchangeable inside a ceiling).

### E2 `[V-t]` — Grebogi, McDonald, Ott, Yorke, "Final state sensitivity: an obstruction to predictability", *Phys. Lett. A* **99**(9), 415–418 (1983), DOI 10.1016/0375-9601(83)90945-3

The origin of the exponent. Metadata resolved through Crossref and Semantic
Scholar (`CorpusId 37206366`); the abstract is elided by the publisher and the
full text is closed. Every later source in this file attributes `α = D − d` and
`f ~ ε^α` here or to E3. Reported error bars on the original `α̂` remain owed
(U1).

Delta: the 1983 paper defines the exponent and relates it to boundary
dimension; X₂₇a measures the same exponent on a different system; **the
difference is zero** — X₂₇a is an application, not an extension.

### E3 `[V-t]` — McDonald, Grebogi, Ott, Yorke, "Fractal basin boundaries", *Physica D* **17**(2), 125–153 (1985), DOI 10.1016/0167-2789(85)90001-6

The long version. Both E1 and E12 cite *this* paper, not E2, as the source of
the uncertainty exponent. Crossref-verified title, venue, volume, pages. Full
text owed (U2).

Delta: as E2; **zero**.

### E4 `[V]` — Wagemakers, "Basins of Attraction: A Dynamical Zoo", arXiv:2504.01580 (dated 2025-07-23), code at doi:10.5281/zenodo.15124200

Carries the numerical procedure of E2/E3 verbatim in prose: "sample initial
conditions within a ball of radius ε. If the ball contains initial conditions
leading to different basins, it is tagged as uncertain… The ratio `f(ε)`
between the number of uncertain and total number of balls is measured for a
range of ball sizes ε… A linear fit in logarithmic scale of `f(ε)` versus `ε`
recovers the uncertainty exponent `α = D − d`."

The canonical GOY test map, reproduced here because it is the system behind
the `α = 0.2` number's sibling figure:

```
θ_{n+1} = θ_n + a sin(2θ_n) − b sin(4θ_n) − x_n sin(θ_n)   (mod 2π)
x_{n+1} = −J₀ cos(θ_n)
```

with `J₀ = 0.3`, `a = 1.32`, `b = 0.9`; the authors "measured a fractal
dimension of `d = 1.8` for this boundary", i.e. `α = 0.2` again in `D = 2`.

Delta: the review states the ε-ball tagging procedure and log-log fit; X₂₇a
specifies a perturb-and-propagate procedure and a fit "over at least three
decades"; **the difference is the estimator, not the estimand** — ball-tagging
counts boundary-adjacent balls, perturbation counts label flips, and the two
agree in exponent but not in prefactor.

### E5 `[V]` — Daza, Wagemakers, Georgeot, Guéry-Odelin, Sanjuán, "Basin entropy: a new tool to analyze uncertainty in dynamical systems", *Sci. Rep.* **6**, 31416 (2016), DOI 10.1038/srep31416, arXiv:1605.02342

Gibbs entropy over an ε-box grid, `S_b = S/N` (Eq. 4), and — the load-bearing
result for X₂₇c — the closed form obtained by substituting box-counting scaling
`N_k = n_k ε^{−D_k}`, `N = ñ ε^{−D}`, `α_k = D − D_k`:

```
S_b = Σ_{k=1}^{k_max} (n_k / ñ) · ε^{α_k} · log(m_k)          (Eq. 6)
```

with `m_k` the number of colours in a boundary box. Fractality criterion on the
boundary-restricted entropy `S_bb = S/N_b` (Eq. 7):

```
S_bb > log 2  ⇒  α < 1                                        (Eq. 8)
```

stated as sufficient but not necessary. The paper also records `α = 1` for the
damped Duffing oscillator's smooth boundary, and `α` decreasing with energy for
Hénon–Heiles.

Delta: Daza et al. give an entropy `H` whose ε-dependence *is* `ε^α`; X₂₇c
proposes `(H, α̂)` as two coordinates of a quadrant policy; **the difference is
that Eq. 6 makes them one object measured at one scale and its scaling
exponent, not two independent axes.** A quadrant policy over `(H, α̂)` measured
at a single ε is measuring the same quantity twice, unless the ε-sweep that
produces `α̂` covers resolutions the `H` measurement does not.

### E6 `[V-t]` — Ott, Sommerer, Alexander, Kan, Yorke, "Scaling behavior of chaotic systems with riddled basins", *Phys. Rev. Lett.* **71**(25), 4134–4137 (1993), DOI 10.1103/PhysRevLett.71.4134

Crossref-verified. The `α → 0` limit: riddled basins, where every point of one
basin has points of another arbitrarily close, so `lim_{ε→0} f(ε) > 0`. Cited
by E5 as the case where `S_b > 0` for every box size.

Delta: riddling is the degenerate end of X₂₇a's axis; **the difference is
zero** — X₂₇a's `α → 0` regime is this literature's subject.

### Arithmetic audit of the `α = 0.2` figure

Computed this session:

| quantity | value |
|---|---|
| `(10⁻²)^{0.2}` | `0.39810717055` |
| `(10⁻³)^{0.2}` | `0.25118864315` |
| contract figures | `0.398`, `0.251` |
| relative deviation | `2.7×10⁻⁴`, `7.5×10⁻⁴` (rounding to 3 s.f.) |
| `0.398 / 0.251` | `1.58566` |
| `10^{0.2}` | `1.58489` |

**The exponent is handled correctly and the prefactor is silently 1.** The two
figures are not an independent calculation; they are `ε^α` evaluated with
`C = 1`, to the last digit that 3-significant-figure rounding permits. Three
consequences:

1. The published law is a scaling relation, `f(ε) ~ C ε^α`, and E5 Eq. 6 shows
   the prefactor explicitly as `n_k/ñ` — a ratio of boundary-box to total-box
   constants that is a property of the system's boundary length, not a
   universal 1.
2. `ε` is dimensional. `ε^α` is defined only after normalisation by a
   phase-space scale, `f = C (ε/L)^α`. Taking `C = 1` and `L = 1` together
   asserts `f(ε = 1) = 1`: *every* state uncertain at unit resolution. That is
   a statement about the bed's phase-space extent, and it must be measured, not
   assumed.
3. The **ratio** `1.586` per decade is safe to pre-register — it follows from
   `α = 0.2` alone. The **levels** `0.398` and `0.251` are not, and a ceiling
   computed from them (`1 − c·0.398` at `ε = 10⁻²`) inherits the unmeasured
   prefactor directly into the accuracy bar.

Separately: `α = 0.2` is not a prediction. It is E1's forced-damped-pendulum
value (`D₀ ≅ 1.8`) and E4's GOY-map value (`d = 1.8`), both in `D = 2`.
Pre-registering it as the bed's expected exponent imports a two-dimensional
pendulum's boundary dimension into a claim about a different system in a
different dimension, where `α = D − d` will have a different `D`.

---

## Transient chaos, escape rate, and the two-rate-theory claim

### E7 `[V]` — Tél, "The joy of transient chaos", *Chaos* **25**, 097619 (2015), DOI 10.1063/1.4917287

Escape law, quoted as the source's Eq. (1):

```
p(t) ~ e^{−κ t}
```

with κ "independent of the choice of the initial distribution of the `N₀`
tracers… thus a unique property of the chaotic saddle", valid only after a
transient `t₀` needed for the ensemble to approach the saddle, and with average
lifetime `τ ≈ 1/κ` as an order-of-magnitude estimate only. Topological entropy
by `L(t) ~ e^{h t}` (Eq. 2). Ordering constraint on a chaotic saddle: `λ > κ`.

Kantz–Grassberger relation, quoted from the source's discussion of Eq. (13):

```
D₁^{(1)} = 1 − κ / λ
```

"valid for the partial dimension in usual transient chaos", where `λ` is the
positive average Lyapunov exponent on the saddle. The paper's absorption
generalisation is

```
D₁^{(1)} = 1 − (κ τ̄ + ln R̄) / λ̄
```

with `τ̄` the mean return time and `R̄` the mean log reflection coefficient.
Worked numbers: an atmospheric chaotic saddle with `κ = 0.103 day⁻¹` (mean
lifetime ≈ 10 days) and `h ≈ 0.5 day⁻¹`.

Delta: Tél measures κ on a chaotic saddle of a physical flow and relates it to
`λ` and `D₁`; X₂₇d measures near-boundary transient lifetimes and prints κ;
**the difference is zero for the measurement and non-zero only for the claim
that κ and a Kramers rate are two theories of the same process** — see E11.

### E8 `[V-t]` — Kantz, Grassberger, "Repellers, semi-attractors, and long-lived chaotic transients", *Physica D* **17**(2), 75–86 (1985), DOI 10.1016/0167-2789(85)90135-6

Crossref-verified title, venue, volume, pages, date (1985-08). The origin of
`D₁ = 1 − κ/λ`. The relation is available in this file only through E7's
restatement; the original derivation and its stated conditions are owed (U3).

Delta: origin of the relation X₂₇d would use; **zero**.

### E9 `[V-t]` — Lai, Tél, *Transient Chaos: Complex Dynamics on Finite Time Scales*, Springer Applied Mathematical Sciences 173 (2011), DOI 10.1007/978-1-4419-6987-3

Crossref-verified. The monograph treatment of the exponential escape law,
chaotic saddles, and the dimension–escape-rate relations.

Delta: textbook home of X₂₇d's mathematics; **zero**.

### E10 `[V-t]` — Ott, Tél, "Chaotic scattering: an introduction", *Chaos* **3**(4), 417–426 (1993), DOI 10.1063/1.165949

Crossref-verified. Chaotic scattering as the canonical open-system setting in
which κ and the uncertainty exponent are measured together: hyperbolic
scattering gives exponential decay, non-hyperbolic (sticky, KAM-island) gives
algebraic decay — a falsifier for X₂₇d's exponential-lifetime assumption.

Delta: X₂₇d asserts exponential lifetimes; this lineage supplies the case where
that is wrong; **the difference is that X₂₇d has not stated the hyperbolicity
condition its exponential law requires.**

### E11 `[V-t]` — Hänggi, Talkner, Borkovec, "Reaction-rate theory: fifty years after Kramers", *Rev. Mod. Phys.* **62**, 251–341 (1990), DOI 10.1103/RevModPhys.62.251

Crossref-verified. The noise-activated side of X₂₇d's pairing, with the
Kramers/Eyring form the project already prints:

```
k = (ω₀ ω_b / 2πγ) · exp(−ΔE‡ / k_B T)          (moderate-to-high friction)
```

Delta: κ is the escape rate of a *deterministic* chaotic saddle, set by `λ` and
`D₁` with no temperature in it; the Kramers rate is the escape rate of a
*noise-activated* barrier crossing, set by `ΔE‡` and `T` with no Lyapunov
exponent in it. **The difference is that these are rate theories of two
different mechanisms, not two theories of one system**, and printing them side
by side is only meaningful if the bed demonstrably has both a chaotic saddle
and a stochastic driving term, and the paper says which mechanism sets the
measured κ.

### Derived, unverified: `α = κ/λ` couples X₂₇a to X₂₇d

Combining E7's `D₁^{(1)} = 1 − κ/λ` with `α = D − d` for a line transverse to
the stable manifold of a 2-D open map gives `α = κ/λ`. This is arithmetic on
two fetched relations, not a fetched claim, and a citation is owed (U4). If it
holds on the bed, `α̂` and `κ̂` are not independent measurements and X₂₇d's
"second rate theory" is a reparameterisation of X₂₇a.

---

## Uncertainty exponents in machine learning

The contract expected none and instructed verification. The expectation is
**wrong for machine learning generally and right for language modelling
specifically**.

### E12 `[V]` — Ly, Gong, "Riddled basin geometry sets fundamental limits to predictability and reproducibility in deep learning", arXiv:2510.05606 (2025-10-07)

The single most important entry in this file. Full PDF read this session.

Scaling law and dimension relation, verbatim:

```
f(ε) ∝ ε^φ ,      d_f = d − φ
```

with `f(ε)` defined in the text as "the probability of misclassification due to
uncertainty `ε` in the initial state", `d_f` the fractal dimension of the basin
boundary and `d` the dimension of parameter space. Same object as `α = D − D₀`
under a renamed exponent.

Measured values, with the paper's own error bars:

| system | φ |
|---|---|
| minimal 2-layer network | `0.0126 ± 0.0002` (weighted fit) |
| VGG-12, CPU (deterministic) | `0.000 ± 0.002` |
| VGG-12, GPU (non-deterministic) | `0.000 ± 0.002` |

Numerical procedure, verbatim in substance: initialisations sampled at fixed
distance from the invariant subspaces (`d±(θ) = 1`); "for every such
initialization, we generate another by perturbing each network parameter with a
uniform random value `U(−ε, ε)`"; `n = 10⁴` pairs per ε trained for `10³`
epochs; `f(ε)` = fraction of pairs whose training outcome differs; standard
error `√(f(ε)(1−f(ε))/n)`. For VGG-12, perturbations `U(−ε/2, ε/2)` about a
fixed reference initialisation, with `f(ε)` and its error from bootstrap
resampling. Bit-flip ensemble: with `p = 60.7%` reaching the same invariant
subspace, "we can analytically calculate the fraction of ε-uncertain pairs for
ε equal to the least significant bit, `f(ε) = 2p(1−p)`", giving `f = 0.477`.
Consequence stated: "ε must be reduced by almost 24 orders of magnitude to
merely halve `f(ε)`."

The paper states **no formal theorem** bounding achievable accuracy. Its
strongest claim is undecidability: "no algorithm can predict the destination
correctly for all initializations up to a zero-measure set" — a statement about
exact prediction on all inputs, not a numerical ceiling at resolution ε.

Delta: Ly & Gong measure `φ` in *initialisation/parameter* space under
*training* dynamics, with the outcome being which minimum is reached; X₂₇a
measures `α` in *state* space under the bed's inference-time dynamics, with the
outcome being which exit basin is entered. **The scaling law, the estimator,
the perturbation protocol, the error-bar convention and the interpretation are
identical; the difference is the space and the dynamics.** X₂₇a's instrument is
therefore not novel, and any contract sentence claiming the α-instrument as new
fails on this entry alone.

Note also that Ly & Gong's operational `f` is a **pairwise-disagreement**
probability (`2p(1−p)`), not GOY's ε-uncertain area fraction. That distinction
is what decides X₂₇b below.

### E13 `[V-t]` — Sohl-Dickstein, "The boundary of neural network trainability is fractal", arXiv:2402.06184 (2024-02-09)

Abstract read. Finds the boundary between hyperparameters giving stable versus
divergent training "fractal over more than ten decades of scale in all tested
configurations". No fractal dimension and no uncertainty exponent reported.

Delta: establishes fractality of a trainability boundary in *hyperparameter*
space; X₂₇a quantifies fractality of an *exit* boundary in state space with an
exponent; **the difference is that this entry does not produce the number
X₂₇a produces.**

### E14 `[V-t]` — Liu, "Complex fractal trainability boundary can arise from trivial non-convexity", arXiv:2406.13971 (2024)

Abstract read. Fractal trainability boundaries emerge from "simple non-convex
perturbations, i.e., adding or multiplying cosine type perturbations to
quadratic functions"; the observed fractal dimensions are controlled by
"roughness of perturbation".

Delta: deflationary control on the whole lineage — fractality of a boundary is
not evidence of interesting structure, since a cosine ripple on a quadratic
produces it. **X₂₇ owes the same control**: a measured `α < 1` on the bed is
not by itself evidence of anything the bed does, and needs a
trivially-non-convex null.

### E15 `[V-t]` — Torkamandi, "Mapping the Edge of Chaos: Fractal-Like Boundaries in the Trainability of Decoder-Only Transformer Models", arXiv:2501.04286 (2025-01-08, rev. 2025-02-15)

Abstract page read. Extends E13 to "medium-sized, decoder-only transformer
architectures", over the learning-rate landscape for attention and
fully-connected layers; finds "a region of stable convergence… surrounded by a
complex chaotic border" with "a self-similar yet seemingly random structure at
multiple scales". Does **not** compute a fractal dimension or an uncertainty
exponent.

Delta: this is the closest existing work to language modelling, and it is still
in hyperparameter space over training dynamics, with no exponent measured.
**The difference is non-zero and X₂₇a would be, as far as this fetch reaches,
the first uncertainty exponent measured on a language-model-adjacent system in
state space.**

### E16 `[V-t]` — Cohen, Rosenfeld, Kolter, "Certified Adversarial Robustness via Randomized Smoothing", arXiv:1902.02918 (2019)

Abstract read. Certified radius for a Gaussian-smoothed classifier:

```
R = (σ/2) · [ Φ⁻¹(p_A) − Φ⁻¹(p_B) ]
```

`p_A`, `p_B` the top and runner-up class probabilities under noise; below `R`
the smoothed prediction provably cannot change.

Delta: this is a *sufficient* per-point stability radius (a lower bound on how
far the prediction survives); X₂₇b wants a *necessary* upper bound on any
predictor's accuracy at a given resolution. **The difference is the direction
of the quantifier**, and no algebra converts one into the other.

### E17 `[V-t]` — Mahloujifar, Diochnos, Mahmoody, "The Curse of Concentration in Robust Learning", arXiv:1809.03063 (AAAI 2019)

Abstract read. "If the metric probability space of the test instance is
concentrated, any classifier with some initial constant error is inherently
vulnerable to adversarial perturbations"; for Lévy families the attack needs to
perturb by at most `O(√n)` in `n` dimensions.

Delta: the closest published object of the shape "no classifier can do better
than X at perturbation scale ε", but it bounds **adversarial** risk conditional
on non-zero clean error, from concentration of the input measure; X₂₇b bounds
**clean** exit accuracy from boundary geometry. **The difference is the risk
being bounded and the geometric hypothesis doing the bounding** — this entry
does not supply X₂₇b's theorem.

### E18 `[V-t]` — Shena, Kaloudis, Merkatas, Sanjuán, "On the approximation of basins of attraction using deep neural networks", arXiv:2109.06564 (2021)

Abstract read. Treats basin reconstruction "as a classification task and use[s]
a deep neural network as a classifier for predicting the attractor that
corresponds to any given initial condition", extracts an approximate basin
boundary from the trained model, and "provide[s] evidence relating the
complexity of the structure of the basins of attraction with the quality of the
obtained reconstructions, via the concept of basin entropy". Demonstrated on
the bistable Lorenz system.

Delta: this is X₂₇'s overall shape already published — a learned basin
classifier whose accuracy is explained by a geometric unpredictability measure
printed alongside it. **The difference is that X₂₇ substitutes `α̂` for `S_b`
and escalates the correlation into a ceiling theorem.** The correlation is
prior art; the escalation is what X₂₇ would have to defend.

### E19 `[V-t]` — Lamb, Le, "Final state sensitivity and fractal basin boundaries from coupled Chialvo neurons", arXiv:2511.03671 (2025)

Abstract read. Two asymmetrically coupled Chialvo neurons, multistable with two
qualitatively different attractors, boundary "found to be fractal, leading to
extreme final state sensitivity".

Delta: a current-year application of the α-instrument to a neural-*dynamics*
model; X₂₇a applies it to a trained network's bed. **The difference is the
system, not the method.**

### The negative result on language modelling

Searched this session, all returning nothing: `"uncertainty exponent"` crossed
with *language model*, *transformer*, *next token*, *NLP*; `fractal basin
boundary` crossed with *decision boundary* and *neural network*;
adversarial-robustness certified radii; decision-boundary fractality; chaos in
RNN state space; sensitivity of trained networks to input perturbation. The
only hits in the neighbourhood are E12–E15, E18, none of which measures an
uncertainty exponent on a language model's state space, and E15 is the only one
touching transformers at all. **No use of the uncertainty exponent in language
modelling was found.** The contract's expectation is correct for language
modelling and incorrect for machine learning at large, where E12 makes the
instrument standard.

---

## Status of the exit-prediction ceiling

X₂₇b is stated theorem-shaped: *at state-resolution ε, no predictor can exceed
exit accuracy `1 − c·f(ε)` on that state.* No source fetched or searched this
session publishes a bound of that form. What exists is the following, and none
of it is X₂₇b.

### E20 `[V]` — Xu, Bi, Hu, Chen, Yu, Li, Hu, Zhou, "Predictability of Complex Systems", arXiv:2510.16312

Full PDF read. The review's own framing of the field's central question is
"What are the limits of prediction accuracy?", and its two answers are:

**(a) The entropy-and-Fano ceiling.** With `S` the sequence entropy and `C` the
candidate-state count, `Π_max` is the unique solution of

```
S = S_F(Π_max) = −Π_max log₂ Π_max − (1 − Π_max) log₂(1 − Π_max)
                 + (1 − Π_max) log₂(C − 1)                        (Eq. 13)
```

with `S_F` concave and monotonically decreasing on `Π ∈ [1/C, 1]`, so
`Π_max ≥ Π` for every predictor.

**(b) Bayes-error equivalence.** `Π = 1 − R_B` exactly, with `R_B` the Bayes
error rate; the review attributes the proof to E21.

**(c) A resolution-indexed ceiling, empirical.** For human mobility at spatial
scale `s` with spatial uncertainty `μ = 1/s`, the review reports "an approximate
invariance relationship between `Π_max` and spatial uncertainty `μ`, expressed
as `μ^β Π_max ≈ κ`", with `Π_max` rising from ≈0.90 to ≈0.93 as the grid
coarsens from ≈0.15 km² to ≈3 km², attributed to E23.

Delta: (a) and (b) bound accuracy from *entropy*, with no ε in them; (c) is a
power-law-in-resolution ceiling, which is X₂₇b's *shape*, but it is an
empirical invariance in mobility data, not a theorem, and its exponent β is not
an uncertainty exponent. **The difference is that no published ceiling is
indexed by ε through the basin geometry.**

### E21 `[V-t]` — Xu, Zhou, Yu, Sun, Guo, "Equivalence between time series predictability and Bayes error rate", *EPL* **141**(6), 61003 (2023), DOI 10.1209/0295-5075/acc19e

Identifier taken from E20's reference list (ref. [141]). Establishes
`Π = 1 − R_B`.

Delta: if X₂₇b is proved, this entry says what it will have proved — that the
exit-prediction ceiling is the Bayes error of the exit-label problem at
resolution ε. **The difference is zero once `f` is defined so that the bound
holds**, which is the whole difficulty.

### E22 `[V-t]` — Song, Qu, Blumm, Barabási, "Limits of Predictability in Human Mobility", *Science* **327**(5968), 1018–1021 (2010), DOI 10.1126/science.1177170

The canonical published prediction ceiling: 93% potential predictability from
entropy via Eq. 13 above.

Delta: an entropy-derived ceiling on a prediction task, printed beside achieved
accuracy — exactly the reporting discipline X₂₇b proposes. **The difference is
that Song et al. derive their ceiling from a measured entropy, and X₂₇b would
derive its ceiling from a measured `f(ε)`, which requires a bound that does not
exist.**

### E23 `[V-t]` — Lin, Hsu, Lee, "Predictability of individuals' mobility with high-resolution positioning data", *UbiComp* 2012, 381–390, DOI 10.1145/2370216.2370274

Identifier from E20 ref. [121]. Source of `μ^β Π_max ≈ κ`.

Delta: as E20(c).

### What `c` is, and whether the claim survives

Nothing published fixes `c`. Two derivations settle its status, both done here
and both short enough that the contract can check them.

**Under the GOY/Scholarpedia definition of `f`, the claim is false for every
`c > 0`.** There, `f(ε)` is the fraction of ε-balls that contain more than one
basin label. A predictor at resolution ε sees the ball, not the point, and must
emit one label; its error on a ball is `1 − max_j p_j`, where `p_j` is the
within-ball label mass. A ball counted uncertain may be arbitrarily imbalanced:
`max_j p_j → 1` while the ball still contains two labels. So the only inequality
that holds in general is an **upper** bound on error,

```
err(ε) ≤ f(ε) · (1 − 1/N_A) ,
```

which is the wrong direction for a ceiling. The claim `acc ≤ 1 − c·f(ε)`
requires a lower bound on within-ball mixing that the GOY construction does not
provide and that E1–E4 never assert.

**Under a pairwise-disagreement definition of `f`, the claim is a three-line
corollary of the Bayes error with `c = 1/2`.** Take E12's operational
definition: `f(ε) = P(two independently ε-perturbed copies of the state exit
differently)`. For a binary exit with within-ball masses `p, 1−p`, this is
`f = 2p(1−p)`, and the Bayes error inside the ball is `q = min(p, 1−p)`, giving

```
q = (1 − √(1 − 2f)) / 2   ≥   f/2 ,
so   acc(ε) ≤ 1 − f(ε)/2 ,   i.e.   c = 1/2 .
```

Tightness, computed this session: `q / (f/2)` is `1.0051` at `f = 0.01`,
`1.0557` at `f = 0.1`, `1.3820` at `f = 0.4`. The bound is tight as `f → 0` and
loose by 38% at `f = 0.4` — i.e. loose exactly at the `ε = 10⁻²` value the
contract's worked figure would produce.

**Conclusion for X₂₇b.** It is not published, in dynamics or in learning
theory. It is false at the definition of `f` that the α-instrument as written
(perturb, propagate, count basin changes against a boundary-fraction reading)
would naturally be reported under, and provable with `c = 1/2` only under a
pairwise definition and a binary exit. For `N_A > 2` exits, `c` depends on the
within-ball distribution and is not `1/2`. The project would have to prove it,
and the theorem it would prove is the Bayes error of the exit problem, already
published in the form `Π = 1 − R_B` (E21).

---

## Owed

`[U]` entries, and why each is still owed.

**U1 — Grebogi, McDonald, Ott, Yorke (1983) full text.** DOI
10.1016/0375-9601(83)90945-3. Owed: the fit range in decades, the number of
initial conditions per ε, and the reported uncertainty on the original `α̂`.
Elsevier-closed and the abstract is elided by the publisher in the Semantic
Scholar record. The free copy at
`yorke.umd.edu/Yorke_papers_most_cited_and_post2000/` could not be retrieved:
three attempts this session, `curl` (connect timeout on port 443) and the fetch
tool (`ECONNREFUSED 129.2.148.17:443`). Retry from a different network.

**U2 — McDonald, Grebogi, Ott, Yorke (1985) full text.** DOI
10.1016/0167-2789(85)90001-6. Same host, same failure. This is the paper E1 and
E12 actually cite for the exponent, so it, not U1, is the primary owed source.
Owed: the original numerical procedure and its error analysis.

**U3 — Kantz, Grassberger (1985) full text.** DOI 10.1016/0167-2789(85)90135-6.
Closed. `D₁ = 1 − κ/λ` appears in this file only through E7's restatement; the
derivation's hyperbolicity and ergodicity conditions are owed before X₂₇d
relies on the relation.

**U4 — A citation for `α = κ/λ`.** Derived above from E7 Eq. (13) plus
`α = D − d`; not fetched. If X₂₇a and X₂₇d are to be reported as independent
measurements, this relation must be either found in the literature or shown not
to apply to the bed.

**U5 — Fano's inequality in its textbook statement.** Cover & Thomas,
*Elements of Information Theory*, 2nd ed. (ISBN 978-0-471-24195-9). Not
fetched; the Fano function used above is E20's Eq. (9)/(13). Owed if the
project states the ceiling information-theoretically rather than through the
Bayes error.

**Counts: `[V]` 6 · `[V-t]` 17 · `[U]` 5.**
`[V]`: E1, E4, E5, E7, E12, E20.
`[V-t]`: E2, E3, E6, E8, E9, E10, E11, E13, E14, E15, E16, E17, E18, E19, E21,
E22, E23.
`[U]`: U1–U5.

---

## Kill-clause exposure

**X₂₇a — the α-instrument. Buildable, novelty zero.** E12 runs precisely this
protocol on neural networks, with published exponents and error bars, using the
same perturbation scheme, the same log-log fit and the same
`√(f(1−f)/n)` error convention. E1, E3 and E4 own the estimand. Any contract
sentence presenting the α-instrument as new fails on E12 alone. What survives
is the measurement on this bed, and only if it is reported against E14's
trivially-non-convex null — a boundary being fractal is not evidence, since a
cosine ripple on a quadratic produces one.

**X₂₇b — the exit-prediction ceiling. Unbuildable as stated.** No published
theorem, no published `c`. Under GOY's `f` the inequality runs the other way.
Under a pairwise `f` with binary exits it is `acc ≤ 1 − f/2` by three lines of
Bayes-error algebra, and that object is already published as `Π = 1 − R_B`
(E21). This is the layer's single most dangerous item: an unproved
theorem-shaped claim that, if the accuracy tables start printing a "ceiling"
column, silently converts an unmeasured prefactor (`C = 1`) and an
unstated definition of `f` into the project's headline bar. Recommendation on
the evidence: X₂₇b does not enter any table until (i) the definition of `f` is
fixed to the pairwise form, (ii) `c` is derived on the record for the actual
number of exits, and (iii) the prefactor `C` is measured rather than set to 1.

**X₂₇c — the `(H, α̂)` quadrant policy. Buildable, but the axes are not
orthogonal.** E5 Eq. 6, `S_b = Σ (n_k/ñ) ε^{α_k} log m_k`, makes `α` the
ε-scaling exponent of the very entropy that would be the other coordinate. A
quadrant policy over `(H, α̂)` measured at one ε reads one object on two axes.
It becomes meaningful only if `H` is measured at resolutions the `α̂` sweep does
not cover, or if `H` is a predictive entropy of the head rather than a basin
entropy of the bed — in which case the contract must say so, because E18
already publishes the pairing of a basin classifier with basin entropy.

**X₂₇d — κ beside the Kramers/Eyring rate. Buildable, but the framing is
wrong twice.** First, E7 and E11 are rate theories of different mechanisms —
deterministic saddle escape set by `λ` and `D₁`, versus noise-activated barrier
crossing set by `ΔE‡` and `T`. "Two rate theories on one system" requires
demonstrating that the bed has both a chaotic saddle and a stochastic term, and
stating which one sets the measured κ. Second, if `α = κ/λ` (U4) holds, κ is
not independent of X₂₇a and the two-instrument framing collapses to one. Third,
E10 supplies the falsifier the sub-item has not stated: non-hyperbolic saddles
give algebraic, not exponential, decay, so the exponential fit needs a
goodness-of-fit gate rather than an assumption.

**Cross-cutting.** The contract's `α = 0.2` is a pendulum's number and its
`0.398 / 0.251` are `ε^α` at unit prefactor. Neither is a prediction about this
bed. If a pre-registration cites them as expected values, the expectation is
inherited from E1's forced damped pendulum, and the only part of the figure
that follows from `α = 0.2` alone is the per-decade ratio `1.586`.
