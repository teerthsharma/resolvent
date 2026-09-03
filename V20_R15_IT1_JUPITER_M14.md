# V20 R15 it.1 — JUPITER (MYCROFT) — M14 CHEEGER STRATIFICATION

Machine: `Windows-10-10.0.26200-SP0`, AMD64, Python 3.11.9, numpy 1.26.4.
Tree at `207e7b92c5611effbcac6877757dc1d0bb572e42` (branch `v17k-gate0`), plus the
untracked files this round wrote. No git writes were made. Nothing touched Kaggle.

New code: `scripts/v20_m14_cheeger.py`. New tests: `tests/jupiter/test_m14_cheeger.py`.
New artifact: `results/v20_m14_cheeger.txt`. No file owned by another node was edited.

---

## THE ANNEX ROW, IN THE ANNEX'S OWN FORMAT

**Statement.** `φ²/2 ≤ 1 − λ₂ ≤ 2φ`. Correct as written **only** when `λ₂` is the
second largest *eigenvalue* of an irreducible chain reversible w.r.t. `π`, and `φ` is
the *minimum* conductance over all cuts. Neither qualifier appears in the annex row,
and each is load-bearing: one is the difference between a spectral gap and a mixing
gap, the other between a theorem and a number that can be off by 27×.

**Route.** Exact minimum conductance by two independent paths — sweep cut over the
Fiedler ordering (`sweep_cut_conductance`), and brute force over all `2^(n-1)` cuts
(`exact_min_conductance`) — against `γ` from an independent `eigvalsh`.

**Instance.** 10 synthetic graphs with known bottlenecks, plus this repository's own
`LargestJoin_S2Rips_64` (n=18, all 131,071 cuts enumerated) and
`LargestJoin_S2Rips_1024` (n=62, `2^61` cuts, brute force refused).

**Decision.** **STRUCK as a corpus-difficulty dial for BED-M and BED-K**, on the
domain census, not on the check. **F1 when restated on the Rips graph chain**, the
only object in this repository whose hypotheses it fits.

**Grade.** **F1** on the Rips chain — a two-sided bound with printed constants
(`1/2`, `2`). **HOW-BAD gap: the bracket is 108.0× wide** at `LargestJoin_S2Rips_64`,
against a `γ` that one eigensolve returns *exactly*. Not upgradeable to F0: Cheeger's
quadratic half is tight on the cycle family. On BED-M/BED-K there is no grade,
because there is no object for the statement to be about.

**The contract's binding kill does not fire.** `CEQ_V20_R15_CONTRACT.md:262-263` —
"M14 cited before it passes its own check ⇒ struck". The check now passes: 43/43
GREEN, both halves, exact `φ`. A *different* kill fires, and it is `V-25`.

---

## 1. PROVENANCE — THE `[RUN]` TAG HAS NO PRODUCER, AND IS CONTRADICTED

The annex row is `CEQ_V20_R15_CONTRACT.md:239-243` (READ). That file is **untracked**
and was written at `Sep 2 00:47` during this session; it did not exist at
`207e7b9`. My first sweep of the tree ran before it landed and reported no M14
anywhere — that report was correct when made and is superseded here. Recording the
correction rather than quietly using the later reading.

**No conductance producer beyond one named cut has ever existed in this repository.**
`git log --all -S` across all 350 revisions (RUN):

| pickaxe term | commits |
|---|---|
| `sweep_cut` | **0** |
| `fiedler` | **0** |
| `min_conductance` | **0** |
| `phi_crude` | **0** |

The only conductance in the tree is `scale/foreman_lambda2.py:300 bridge_conductance`
(READ), and its own docstring at `:303` says what it is: *"ONE cut, so this is an
upper bound on the Cheeger constant `h` and never a value for it."*

**The check the annex says FAILED, passes.** `results/foreman_lambda2.txt:38` and
`:58` (READ):

```
LargestJoin_S2Rips_64      Phi_bridge 0.0370370370  mu_2 0.0327795158  mu_2 <= 2 Phi True
LargestJoin_S2Rips_1024    Phi_bridge 0.0117647059  mu_2 0.0035921143  mu_2 <= 2 Phi True
```

That is `scale/foreman_lambda2.py:683-684`, asserted at `:976`. Both halves also hold
with that φ, computed here: `φ²/2 = 6.859e-04 ≤ 3.278e-02 = γ` on the 64-case and
`6.920e-05 ≤ 3.592e-03` on the 1024-case (RUN, `scripts/v20_m14_cheeger.py`).

**And the crude φ turns out to be exact.** Brute force over all 131,071 cuts of
`LargestJoin_S2Rips_64` (RUN):

```
phi_exact = 0.037037037037037035   S = [0,1,2,3,4,6,8,10,11,14]
phi_bridge = 0.037037037037037035  crude/exact = 1.0     (bit-identical float64)
bridge lobes: left = [5,7,9,12,13,15,16,17]  right = [0,1,2,3,4,6,8,10,11,14]
exact minimising S == the bridge's right lobe: True
```

The minimum-conductance cut *is* the bridge cut, `φ = 1/27` exactly, and the sweep cut
independently returns the same cut expressed as its complement. So on the one instance
in this repository where exactness is computable, the crude φ was already the answer.

**Verdict on the tag.** `[RUN: my crude φ FAILED the sanity check]` names no producer,
and no run this repository can perform reproduces it. This is the `STRUCK.md` `0.743864`
shape — *"NO PRODUCER HAS EVER EXISTED"* — with one mitigation: it is a fresh
unversioned claim in a contract, not a constant that shipped. It should be withdrawn
from the annex rather than replaced with a number.

**A second, smaller defect, in the tree rather than the annex.** `cheeger_upper_holds`
(`scale/foreman_lambda2.py:683-684`) tests `μ₂ ≤ 2·Φ_bridge`. Since `Φ_bridge ≥ h`
always, and the theorem gives `μ₂ ≤ 2h`, this inequality is *implied* by the theorem
for any correct implementation. It is a regression check on the arithmetic, not
evidence that Cheeger binds — and it is printed under a name that reads as the latter.
It is not vacuous (a coding bug makes it fail, and mutation B below fires it), but it
cannot distinguish a tight φ from a useless one.

---

## 2. THE THEOREM, FROM ITS SOURCE, WITH ITS HYPOTHESES — `[V-eq]`

**CITED**, resolved, title and theorem matched. Levin and Peres, with contributions by
Wilmer, *Markov Chains and Mixing Times*, 2nd ed., **Theorem 13.10**, **page 183**,
ch. 13 §13.2.2 "The bottleneck ratio revisited", attributed there to
**Sinclair–Jerrum (1989)** and **Lawler–Sokal (1988)**. Fetched from
`pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf` (4.6 MB) and read with `pdftotext
-layout`. Its own sentence: *"Let λ₂ be the second largest eigenvalue of a reversible
transition matrix P"*, then eq. (13.6). `Φ_*` is that book's eq. (7.5)–(7.7) at p. 89;
`γ_*` is its eq. (12.6). Graph form cross-checked against Chung, *Spectral Graph
Theory* (CBMS 92), ch. 2 §2.3 pp. 25–26, split across **Lemma 2.1** (`λ₁ ≤ 2h_G`) and
**Theorem 2.2** (`λ₁ ≥ h_G²/2`).

**One hypothesis correction, against my own first statement of it.** I wrote
"irreducible and reversible". The theorem's own sentence states **reversibility
only**. Irreducibility is not in it; it enters as a standing convention of the
chapter, through the eigenvalue ordering `1 = λ₁ > λ₂ ≥ … ≥ −1` at eq. (12.7), which
needs Lemma 12.1(ii) — irreducibility makes the eigenvalue-`1` eigenspace
one-dimensional — for `λ₂` to be *well defined at all*. So irreducibility is required
to **state** the theorem rather than to prove it. `_require_graph` checks it either
way. Correcting this rather than leaving the stronger claim standing is the whole
point of the `[V-eq]` discipline.

For such a `P`, reversible w.r.t. `π`:

```
    Φ_*² / 2  ≤  γ  ≤  2 Φ_*
    γ    = 1 − λ₂,  λ₂ the second largest EIGENVALUE (not modulus)
    Φ_*  = min { Q(S,Sᶜ)/π(S) : π(S) ≤ 1/2 },   Q(S,Sᶜ) = Σ_{x∈S,y∉S} π(x)P(x,y)
```

On a connected undirected weighted graph with `W = Wᵀ ≥ 0` and `P = D⁻¹W`, this is
`Φ(S) = cut(S)/min(vol S, vol Sᶜ)` and `γ` is the second smallest eigenvalue of the
normalised Laplacian `I − D^{-1/2} W D^{-1/2}`. All five hypotheses are checked rather
than assumed in `_require_graph` (`scripts/v20_m14_cheeger.py`), which raises on
asymmetry, on a negative entry, on an isolated vertex and on disconnection.

**The annex's form is the right theorem for a quantity the annex then misnames.**
The row says *"corpus difficulty (**mixing**)"*. Cheeger constrains `γ = 1 − λ₂` and
says nothing about `λ_n`; mixing time is governed by the **absolute** gap
`γ_* = 1 − max_{i≥2}|λ_i|`. Measured (RUN,
`test_the_gap_cheeger_bounds_is_not_the_mixing_gap`): on the bipartite `K_{4,4}`,
`γ = 1.0` while `γ_* = 0.0` — Cheeger reports a healthy bottleneck for a chain that
never mixes. Laziness is the standard repair and is an **extra hypothesis M14 does not state** —
and the source says so in its own voice. That book's **Exercise 12.3** is what gives
`γ = γ_*`, *"if the chain is lazy"*; its **§17.4 at p. 249** needs the words *"For a
lazy reversible Markov chain"* before it may combine **Theorem 12.4** (which bounds
`t_mix` through `γ_*`) with **Theorem 13.10** (which bounds `γ`); and its **Remark
12.6** at p. 164 names the fix — when `λ_|X|` is near `−1` the slow mixing *"can be
rectified by passing to a lazy chain"*. This is `P-10` shape (a source read one clause
short of its hypotheses), caught before it entered a verdict.

**On this repository's own instances the defect is latent, not active**, and that is
recorded rather than elided: `γ = γ_* = 0.0327795158` on the 64-case and
`0.0035921143` on the 1024-case (RUN). The Rips graphs are not near-bipartite.

---

## 3. L-DOM — THE TWO-COLUMN DOMAIN CENSUS

Producer: `scripts/v20_m14_cheeger.py::domain_census`, run at
`results/v20_m14_cheeger.txt`. Generators taken from the registry at
`ceq/kdata.py:472-482`, drawn, never assumed from a docstring — `V-25`'s check
requires the **fraction of draws satisfying the hypothesis at the quantifier level the
theorem uses**, and `0%` is a blocked gate.

| Thm 13.10 hypothesis | BED-M (`ceq.corpus.build`, seed 0) | the `{−1,0,+1}` support (`scale/negation_scope.py:428-429`, n=2048 s=64) | BED-K (`bed_k.build_delay`, n=500 d=4 seed 7) | Rips chain (`build_chain`) |
|---|---|---|---|---|
| a square matrix over states | **False** — `build()` returns `train/test/vocab/held_out`; no square array | **False** — shape `(2048, 64)`, a coefficient array | True — `(500,500)` | True — `(18,18)` |
| `W = Wᵀ` (⇒ reversible) | n/a, no matrix | n/a | **False** | True |
| `W ≥ 0` (⇒ a `π` exists) | n/a | **False** — 61,348 / 131,072 entries negative | True | True |
| `P` row-stochastic | n/a | n/a | **False** — row sums 0.0000…1.0000; 4 zero rows | True by construction |
| `P` irreducible | n/a | n/a | **False** — strictly lower triangular, `Kⁿ = 0`, **nilpotent** | True |
| **draws satisfying the hypotheses** | **0 / 384** | **0 / 2048** | **0 / 8 kernels swept** | **2 / 2** |

**BED-K's failure is structural, not sampling** — every `(kind, params, n)` swept
(RUN): `delay d=1/4` and `powerlaw H=0.6/0.9` at `n = 32, 128`, all eight
`sym=False lower-tri=True`. `_delay_kernel_matrix` and `_powerlaw_kernel_matrix`
write a strictly lower triangular matrix by construction at every `n`, `d`, `H` and
seed. Its eigenvalue spectrum is identically zero (RUN: `max|eig| = 0.0` at
`powerlaw n=64 H=0.8`), so no spectral-gap dial of any kind exists on it, Cheeger's
or otherwise.

**BED-M's failure is categorical.** `ceq.corpus.build` emits token ids
(`int32 (384,12)`, 16 distinct) and CPython integer labels (44 distinct in
`[−36, 36]`). There is no operator, so there is no `π`, no `λ₂` and no `φ`.

**The measured conclusion.** The hypothesis set of Theorem 13.10 is satisfied by **0
of the campaign's registered beds**. It is satisfied by the Rips graph corpus, which
is a *different* corpus, is neither BED-M nor BED-K, and whose `λ₂` dial
`scale/foreman_lambda2.py:40-66` already records as refuted for a separate reason —
the conductance dial moves the *ergodic* `λ₂`, not the `ρ(Q)` the `t*` ladder
truncates. This is `V-25` exactly: a theorem whose hypothesis no draw in the corpus
satisfies.

---

## 4. TWO INDEPENDENT NUMERIC PATHS, AND WHERE THEY DISAGREE

Path A is the sweep cut over the Fiedler ordering — `O(n³)`, always available,
searches `n−1` of the `2^(n-1)` cuts, so it can only ever be **too large**. Path B is
brute force over every cut — exact, exponential, capped at `n ≤ 20`. Neither calls the
other. Agreement is evidence; disagreement is priced. Selected rows (RUN,
`results/v20_m14_cheeger.txt`):

| graph | n | γ | γ_* | φ_sweep (A) | φ_exact (B) | A/B | φ²/2 ≤ γ | γ ≤ 2φ |
|---|---|---|---|---|---|---|---|---|
| `dumbbell_6x6` | 12 | 0.0506582368 | 0.0506582368 | 0.0322580645 | 0.0322580645 | 1.000000 | ✓ | ✓ |
| `path_12` | 12 | 0.0405070264 | 0.0000000000 | 0.0909090909 | 0.0909090909 | 1.000000 | ✓ | ✓ |
| `cycle_14` | 14 | 0.0990311321 | 0.0000000000 | 0.1428571429 | 0.1428571429 | 1.000000 | ✓ | ✓ |
| **`hypercube_16`** | 16 | 0.5000000000 | 0.0000000000 | **0.3750000000** | **0.2500000000** | **1.500000** | ✓ | ✓ |
| `LargestJoin_S2Rips_64` | 18 | 0.0327795158 | 0.0327795158 | 0.0370370370 | 0.0370370370 | 1.000000 | ✓ | ✓ |
| `LargestJoin_S2Rips_1024` | 62 | 0.0035921143 | 0.0035921143 | 0.0117647059 | *refused, `2^61` cuts* | — | — | ✓ |

**They disagree on `Q4` by exactly 1.5×**, and the mechanism is measured, not guessed:
the normalised adjacency of the 4-cube has eigenvalues
`[-1, -0.5×4, 0×6, 0.5×4, 1]` (RUN), so `λ₂ = 0.5` carries **multiplicity 4**. `eigh`
returns an arbitrary vector of that eigenspace and the sweep lands on a suboptimal
cut (`0.375` against the true dimension cut `0.25`). This is the honest statement the
annex needs: **the sweep cut is an upper bound on `φ`, not `φ`**, and on a degenerate
Fiedler eigenspace it is 50% high. Any annex row that reports a sweep cut as `φ`
inherits that error.

At `n = 62` `φ` is not computable and only bracketed. `γ ≤ 2φ` inverts to
`φ ≥ γ/2`, so `φ ∈ [γ/2, φ_sweep] = [0.0017960571, 0.0117647059]` — a **6.55× window**
(RUN). Stated as a bracket, never as a value.

---

## 5. TEST-BOUND: THE RED, THE GREEN, AND TWO PLANTED NEGATIVES

**RED first, verbatim, against the tree as it stood:**

```
ImportError while importing test module '...\tests\jupiter\test_m14_cheeger.py'.
tests\jupiter\test_m14_cheeger.py:58: in <module>
    from scripts.v20_m14_cheeger import (  # noqa: E402
E   ModuleNotFoundError: No module named 'scripts.v20_m14_cheeger'
ERROR tests/jupiter/test_m14_cheeger.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.55s
```

**GREEN after:** `43 passed in 0.60s`. Whole directory unbroken: `92 passed in 44.33s`
(49 pre-existing + 43 new).

**A suite that cannot fail is the `V-16` class, so the suite was given planted
negatives and watched firing.** Two mutations, applied and reverted (RUN):

| mutation | what was broken | result |
|---|---|---|
| **A** | Fiedler coordinate `evec[:,-2]/√d` → a scrambled ordering | **7 failed, 36 passed** — `test_the_sweep_cut_obeys_its_own_quadratic_guarantee[cycle_14, dumbbell_6x6, dumbbell_6x6_w05, path_12]` and `test_a_correct_phi_detects_the_planted_bottleneck[4,6,8]` |
| **B** | conductance denominator `min(vol S, vol Sᶜ)` → `vol S`, in **both** paths | **6 failed, 37 passed** — `test_cheeger_two_sided_holds_with_the_exact_phi[complete_8, cycle_8, hypercube_16, star_10]`, `test_the_repo_crude_phi_is_an_upper_bound_on_the_exact_minimum`, `test_the_repo_crude_phi_is_bracketed_by_the_certificate_free_bounds` |

The two failure sets are **disjoint**, so the suite is not one assertion wearing eight
names. Restored and re-run: `43 passed`.

**The must-fire with its planted negative** (`test_a_single_named_cut_MISSES_the_planted_bottleneck`):
on a dumbbell with a bottleneck known in closed form, `φ = 1/(m(m−1)+1)`, a correct φ
returns it on both paths, while a `bridge_conductance`-shaped evaluation of one *named*
cut (the singleton `{0}`) reads `1.0` and misses it by `m(m−1)+1` — 13×, 31× and 57× at
`m = 4, 6, 8`.

**And the lower half genuinely breaks under that substitution**
(`test_the_lower_half_BREAKS_when_an_upper_bound_on_phi_is_substituted`): on a path of
`n = 8, 12, 16, 20`, `φ_named²/2 = 0.5` while `γ ≈ 4.93/n²`. `γ ≤ 2φ` survives any
upper bound; `φ²/2 ≤ γ` does not. This is the concrete reason the annex row cannot be
used with a named cut — and the reason the tree's own module says so at
`scale/foreman_lambda2.py:62-66`.

---

## 6. GRADE, AND THE HOW-BAD GAP

**F1** on the Rips chain — a two-sided bound with printed constants, both halves
verified with an exact `φ`.

**HOW-BAD.** The bracket M14 puts on `γ` is `[φ²/2, 2φ]`, of width ratio `4/φ`:

| instance | φ | bracket on γ | width | γ, measured directly |
|---|---|---|---|---|
| `LargestJoin_S2Rips_64` | `1/27` exact | `[6.859e-04, 7.407e-02]` | **108.0×** | `0.0327795158`, one `eigvalsh` |
| `LargestJoin_S2Rips_1024` | `∈ [1.796e-03, 1.176e-02]` | — | **340× to 2,227×** | `0.0035921143`, one `eigvalsh` |

**Not upgradeable to F0.** Cheeger's quadratic half is tight — the cycle family
attains `γ ≍ φ²` — so no implementation improves the bracket. F1 is the ceiling of
this route.

**REPRICE, and it is the decisive number.** M14 routes from the **expensive** quantity
to the **cheap** one. Certifying `φ = 1/27` on an 18-node graph took **131,071 cut
evaluations**; `γ` on the same graph is one `eigvalsh` on an `18×18` matrix, is
**exact**, and is already computed in the tree at
`scale/foreman_lambda2.py:286 ergodic_lambda_2`. At `n = 62` the exact route needs
`2^61 ≈ 2.3e18` cuts and is refused, while the eigensolve is instant. Minimum
conductance is NP-hard in general; `λ₂` is polynomial. **M14 cannot beat direct
measurement of `γ` on any instance where `γ` is computable — and `γ` is computable
wherever a chain exists at all.** As a *measurement* route it is strictly dominated;
its only defensible use is as a *design* statement (pick `φ`, get `γ`), which is
precisely what `scale/foreman_lambda2.py:40-66` already tried and recorded as refuted.

**What would move it.** Nothing moves it off F1 as a measurement. It would move from
STRUCK to gradeable *on the beds* only if a bed were re-stated so that a reversible
irreducible chain exists on it — which for BED-K means abandoning the strictly lower
triangular kernel that defines the bed, and for BED-M means inventing an operator the
corpus does not have.

---

## 7. EVERY KILL SHIPS A REPLACEMENT ROUTE

GOAL restated: *a corpus-difficulty dial that is a property of the chain, not of the arm.*

**REROUTE — for any corpus that has a chain: measure `γ` directly, drop `φ`.**
Already in the tree, zero new code: `scale/foreman_lambda2.py:286 ergodic_lambda_2`
returns `γ` exactly from `eigvalsh` on `D^{-1/2}WD^{-1/2}`. Measured:
`γ = 0.0327795158` (64) and `0.0035921143` (1024), and `γ_* = γ` on both, so no
laziness correction is needed on this substrate — checked, not assumed. Cost: one
`O(n³)` eigensolve. Strictly better than M14 on accuracy *and* cost.

**REROUTE — for BED-K: the singular spectrum of `K`, indexed by the `H` the bed
already ships.** `K` is nilpotent so its eigenvalue spectrum carries nothing
(`max|eig| = 0.0`), but its singular spectrum is a clean monotone dial. Measured at
`n = 128` (RUN):

| H | 0.55 | 0.60 | 0.70 | 0.80 | 0.90 | 0.99 |
|---|---|---|---|---|---|---|
| `cond(K)` | 7.64 | 8.74 | 11.57 | 15.52 | 21.10 | 28.10 |
| eff. rank (90% energy) | 84 | 79 | 69 | 56 | 41 | 29 |

Monotone in `H` on both readings, computed from `K` alone, so it is a property of the
bed and not of the arm — which is M14's stated goal, reached without a chain. `H` is
already the bed's parameter (`ceq/beds/bed_k.py:92-125`), so this reprices an existing
dial rather than adding one. The delay bed sits at `cond = 1.000` with effective rank
112/128, i.e. flat — correctly, since a pure delay is an isometry on its range.

**RETIRE — for BED-M.** `ceq.corpus.build` emits token ids and CPython integers. There
is no operator, so no spectral or bottleneck dial exists or can be constructed without
inventing a corpus the campaign does not have. Its difficulty dial is the one it
already ships: the held-out composition cell `('-', 1)` (`ceq/corpus.py:88-105`), which
is a property of the corpus and not of the arm. No replacement is needed; M14 should
simply not be cited there.

---

## 8. THE LEAN TARGET (annex owner's note)

Mathlib carries `Mathlib/Combinatorics/SimpleGraph/LapMatrix.lean` — the
**combinatorial** Laplacian `D − A`, its symmetry, positive-semidefiniteness, and
`card ConnectedComponent = rank ker L`. It carries **no** normalised Laplacian, **no**
`λ₂`, **no** conductance and **no** Cheeger: `grep -ril "cheeger"` and
`grep -ril "conductance"` over `Mathlib/` both return **zero files** (RUN).

A Lean target for M14 is therefore a from-scratch build of the normalised Laplacian,
the variational characterisation of `λ₂`, and the sweep argument. Against an F1
statement that direct eigensolve measurement strictly dominates (§6), **I do not
recommend it**, and M14 should not enter the Lean ledger this round.

---

## 9. WHAT I COULD NOT VALIDATE

1. **The inequality's printed symbols were not read directly; its direction is
   inferred.** The PDF's math font carries no `ToUnicode` mapping, so `pdftotext`
   silently drops `λ γ Φ π ≤ ≥` — the theorem's *name, number, page, attribution and
   hypothesis sentence* were read verbatim, but the glyphs of eq. (13.6) itself came
   back as blanks. The direction is therefore established by three independent legs
   and not by reading it: Chung's separately-fetched Lemma 2.1 / Theorem 2.2, the
   surrounding proof headings (*"Proof of the upper bound…"*, *"…the lower bound in
   Theorem 13.10"*, pp. 183–185), and my own numerical verification of both halves on
   11 graphs with an exact `φ`. I did not obtain a glyph-accurate rendering of (13.6).
2. **`φ` at `n = 62` is a bracket, not a value.** `2^61` cuts is not enumerable and no
   polynomial exact algorithm exists (sparsest cut is NP-hard). `[0.0017960571,
   0.0117647059]`, width 6.55×, is the strongest statement available there.
3. **Only two repository instances exist to test on**, both from `REROUTED_CASES`, and
   only one is small enough for the exact path. `n = 1` for the crude-versus-exact
   comparison. The `crude/exact = 1.0` result is one graph, not a law: it says the
   bridge cut happened to be the sparsest cut there, not that a named cut generally is.
4. **The `{−1,0,+1}` object's identity is contested in the tree and I did not resolve
   it.** `CEQ_V20_R15_CONTRACT.md:254` and `MISTAKES.md V-25` treat the
   `scale/negation_scope.py:428-429` gate array as "BED-M's support", while
   `ceq/kdata.py:473` registers `ceq.corpus.build` as BED-M's generator, and that
   function emits nothing of the sort. My census covers **both** objects and both read
   `0`, so the census verdict is insensitive to which one is meant — but the naming
   conflict itself is unresolved and belongs to whoever owns `ceq/kdata.py`.
5. **I did not re-derive Cheeger.** The lower half is cited, not proved here; only its
   numerical consequences are checked. The sweep guarantee `φ_sweep ≤ √(2γ)` is
   likewise asserted from the standard proof and confirmed on 10 graphs, not derived.
6. **`γ_* = γ` was checked on the two Rips instances only.** On any future corpus the
   laziness question reopens and must be re-measured; the bipartite `K_{4,4}` case in
   the suite shows how far apart the two can be (`1.0` against `0.0`).
7. **No claim is made about `ρ(Q)`, the quantity the `t*` ladder actually truncates.**
   `scale/foreman_lambda2.py:16-50` establishes that Cheeger's `λ₂` and the ladder's
   `ρ(Q)` are different numbers (`0.9672204842` against `0.9789623187` on the
   64-case). Nothing in this report narrows that gap, and M14 was never about `ρ(Q)`.
