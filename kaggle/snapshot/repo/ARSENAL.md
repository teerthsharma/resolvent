# CEQ MATHEMATICAL ARSENAL — post-audit consolidation, v2

**Scope rule:** nothing here is machinery already shipped in the repo (nilpotent
resolvents, path sums, CP intervals are STOCK, not listed). Everything below was
proposed against the measured record.

**USAGE RULE:** an item may be worked on only while its checklist hook is
UNTESTED or GREEN. **A hook going RED freezes every item attached to it.**

Citations marked **[V]** were verified by direct fetch. **[U]** are from memory
and MUST be fetched before use.

---

## TIER 1 — LOAD-BEARING (each attaches to a MANDATORY item)

### A1. ANTI-CONCENTRATION / LITTLEWOOD–OFFORD → M2, M3
Objects: small-ball probability of signed sums; Erdős–LO bounds; inverse-LO
structure theory; Tikhomirov singularity result; Campos–Jenssen–Michelen–
Sahasrabudhe [U].

**VERIFIED BY DIRECT FETCH 2026-08-25 (iteration 8):**
- **[V] Tikhomirov**, *Singularity of random Bernoulli matrices*, **Annals of
  Mathematics 191(2), 2020, pp. 593–634**, arXiv:1812.09016. For an n×n matrix
  with independent ±1 entries, P(singular) = **(1/2 + o_n(1))^n**. Settles the
  old problem. Attribution and venue as stated.
- **[V] Inverse Littlewood–Offord.** Posed by **Tao–Vu**, *Inverse
  Littlewood-Offord theorems and the condition number of random discrete
  matrices*, **Annals 169(2)**. Result: large concentration implies most
  coordinates lie in a **generalized arithmetic progression of small rank and
  small volume**, via Freiman-type additive combinatorics. Sharpened to an
  **optimal** form by **Nguyen–Vu**, arXiv:1004.3967, removing the error term.
  So the design principle A1 rests on — *anti-concentration stays large iff the
  weights carry additive structure* — is stated correctly and is attributable.
- **[U] still unfetched:** Campos–Jenssen–Michelen–Sahasrabudhe (symmetric
  case). Not load-bearing for A1's use here; fetch before citing.

Why it survives: `sign_flip_rate` IS the small-ball probability
`P(|background sum| < |one term|)`. It predicts the unexplained exponent
(−1 share × −1/2 CLT small-ball = −3/2; log-correlated corrections pull to −4/3;
measured −1.389 sits between) AND the pivot mechanism's core number: a k-term
background anti-concentrates as `k^(−1/2)`, independent of s. Inverse-LO gives
the only known design principle for arresting decay: additive structure (GAP
proximity) of path weights.

Kill: histogram the background sum per s; if its small-ball mass does not factor
the measured exponent, the theory is decoration.

**NOTE — first attempt VOID.** Run 2026-08-25 at fixed `i=7, j=1, c=4` put the
2-hop intermediate count at ~5 regardless of s, so there was no dilution to
measure; every exponent read ~0.00 at R² 0.003. Re-run under `PROTOCOL: SCALING`.

### A2. NUMERICAL RANGE / NON-NORMAL OPERATOR THEORY → M6
Objects: numerical range W(A); Berger power inequality; Crouzeix–Palencia;
pseudospectra (Trefethen–Embree [U]); Chebyshev polynomials extremal on W(A).

**VERIFIED BY DIRECT FETCH 2026-08-25 (iteration 9):**
- **[V] Berger power inequality:** `w(A^m) <= w(A)^m` for every operator and
  every positive integer m. Conjectured by **Halmos**, proved by **Berger**,
  elementary proof by **Pearcy (1966)**. As stated.
- **[V] Crouzeix–Palencia (2017):** the numerical range is a **(1+√2)-spectral
  set** — `||f(A)|| <= (1+√2) sup_{W(A)}|f|` for f holomorphic on a neighbourhood
  of the closure of W(A). arXiv:1708.08633 (remarks/simplified proof). Crouzeix
  conjectures the constant may be **2**, and 2 is best possible.
  **NEW, unverified:** arXiv:2608.03841 (2026) is titled "A solution to
  Crouzeix's conjecture" — if it holds, the constant is 2. Fetch before relying.

**[RUN] DOES THE CHAIN APPLY TO THE SHIPPED OPERATOR?** Checked on `tgate`
(s=64, seed 0) rather than asserted:

    w(A)   = 1.499315      ||A|| = 2.837508      ||A|| <= 2w(A): TRUE
    h=1  ||A^h|| = 2.838e+00   2w^h = 2.999e+00   holds
    h=2  ||A^h|| = 1.777e+00   2w^h = 4.496e+00   holds
    h=3  ||A^h|| = 7.119e-01   2w^h = 6.741e+00   holds
    h=4  ||A^h|| = 2.099e-01   2w^h = 1.011e+01   holds

**THE PROBLEM THIS EXPOSES, and it is M6's real content.** The bound holds but is
**vacuous as it stands**: `w(A) = 1.4993 > 1`, so `2w(A)^h` GROWS with h while
the true `||A^h||` falls (the operator is nilpotent). A certificate whose bound
grows while the quantity shrinks certifies nothing. M6's guard is only a real
constraint **if it actually binds w(A) <= 1** — and the shipped operator does not
satisfy that today. So M6 is not a formality; enforcing it is a genuine change to
the operator, and the kill ("training diverges under the constraint at every lr")
is a live possibility rather than a formality.

**REFINEMENT to how M6 should be stated.** `||A^h|| <= 2 w(A)^h` needs only
Berger plus the standard `||M|| <= 2w(M)` — Crouzeix–Palencia is not required for
it. Crouzeix–Palencia buys the strictly better thing: a bound on the WHOLE path
sum `p(A) = sum_h A^h`, which is what the operator actually computes, rather than
a term-by-term bound that is then summed. M6 should be stated on `p(A)`.

Why it survives: the operator's spectrum is **{0}** — every eigenvalue-based
guard is vacuous (this is exactly how T1 died) — yet transient growth is real
(row-L1 spread 1.5e4×; tgate measured 0.31 → 7.80 across one sequence). W(A) is
the unique non-vacuous control object: one scalar `w(A) <= rho` gives
`||A^h|| <= 2 rho^h` with NO denominator, hence nothing for context to dilute.
This is the entire mathematical content of M6.

Kill: enforce the constraint; if M2 decay reappears or training diverges at
every lr, the guard is wrong for this operator.

### A3. LGV / TOTAL POSITIVITY / SIGN VARIATION ON THE GRASSMANNIAN → M1
Objects: Lindström–Gessel–Viennot [certain]; variation-diminishing property of
totally nonnegative kernels; positroid stratification; sign variation
characterizations (Karp; Galashin–Karp–Lam [U]).

**VERIFIED BY DIRECT FETCH 2026-08-25 (iteration 11):**
- **[V] Variation diminishing property.** First studied by **Schoenberg (1930)**;
  total positivity was introduced by Schoenberg precisely *in his study of
  variation-diminishing kernels*, with generalizations by Schoenberg and by
  **S. Karlin**. Statement: if `k(x,y)` is a TP kernel and `h` has n sign
  changes, then `q(y) = ∫ k(x,y) h(x) dx` has **at most n** sign changes — a
  totally nonnegative kernel can only DESTROY sign changes, never create them.
  **This is the exact classical form of the repo's semiring theorem.**
- **[V] Gantmacher–Krein (1950)** — fundamental results relating total positivity
  to variation diminution. Newly surfaced; not previously in the arsenal.
- **[V] Characterization:** the variation-diminishing property together with the
  **sign non-reversal** property *characterize* totally nonnegative matrices
  (arXiv:2103.05624, arXiv:2007.09999). So "non-negative ⟺ cannot create a sign
  change" is an iff, not merely an implication.
- **[V] Postnikov**, *Total positivity, Grassmannians, and networks* — the
  positroid stratification of `Gr_≥0(k,n)`, cells cut out by vanishing Plücker
  coordinates, a special case of Lusztig's theory.
- **[U] NOT FOUND, stays unverified:** Karp's sign-variation characterization and
  Galashin–Karp–Lam. Searched, not located. Do not cite until fetched.

Why it survives: upgrades the repo's semiring theorem to its strongest classical
form — positive path weights force total nonnegativity, which provably DESTROYS
sign changes — and states the open theoretical question the build cannot answer
alone: which sign strata outside the TNN region are reachable by causal signed
path sums, and at what anti-concentration rate (join with A1).

Kill: none needed — it is a lens; it dies only if M1–M3 die.

### A4. DIRECTED POLYMERS / KPZ / GIBBS MEASURES ON PATHS → M2, M6
Objects: partition functions of directed polymers; free energy = growth rate;
KPZ exponents; directed landscape; partition-function zeros / Lee–Yang [U].

**VERIFIED BY DIRECT FETCH 2026-08-25 (iteration 12):**
- **[V] Dauvergne, Ortmann & Virág**, *The directed landscape*,
  arXiv:1812.00309, **Acta Mathematica 229(2), December 2022**. Constructs the
  Airy sheet, characterizes it via the Airy line ensemble; last-passage geodesics
  converge to random functions with **Hölder-2/3-continuous** paths. Described as
  completing the construction of the central object of the KPZ universality
  class. Attribution, venue and date as stated.
- **[V] KPZ exponents:** fluctuation **1/3** with a **Tracy–Widom** limit;
  transversal/roughness exponent **2/3**. The pair (2/3, 1/3) is standard, and
  the scaling relation `χ = 2ξ − 1` is documented (arXiv:1211.0992).
- **[U] Lee–Yang / partition-function zeros** — not fetched. Do not cite yet.

**CAVEAT THAT WEAKENS A4's PREDICTIVE CLAIM, and it is load-bearing.** Rigorous
proofs of either the 1/3 fluctuation exponent or the 2/3 transversal exponent
are **scarce outside a few integrable cases**. A4 predicts that trained path
weights localize on `O(s^{2/3})` corridors; that extrapolates an exponent proved
for integrable models to a *learned, non-integrable* operator. A4 therefore
**suggests** the measurement, it does not **predict** it. Stated at this strength
and no higher, per evidence-class discipline.

**A4's kill is still OWED and its earlier attempt was VOID.** The participation
ratio was measured once, in the PINNED geometry, where the 2-hop intermediate
count is ~5 regardless of s — it read a constant ~1.27–2.89 and meant nothing.
Must be re-run under `PROTOCOL: SCALING`, and the kill stands: if PR is flat in
s (no localization), the pivot design loses its physical justification and
stands on A1 alone.

Why it survives: `(A^h v)_i` is verbatim a polymer partition function on the
causal lattice. Path-weight LOCALIZATION (mass on O(s^{2/3}) corridors) is the
natural-law version of pivot routing — if trained operators localize, pivots
impose what the physics already prefers. Sign flips are partition-function zero
crossings, connecting M2 to Lee–Yang.

Kill: participation ratio of trained path weights vs s; if flat (no
localization), the pivot design loses its physical justification and stands on
A1 alone.

---

## TIER 2 — STRUCTURAL

### B1. CARNOT GROUPS / SUB-RIEMANNIAN GEOMETRY → S3
Objects: nilpotent graded Lie groups [V]; Chow–Rashevskii reachability [V];
Ball–Box anisotropic scaling eps^h; Pansu differentiability [U]; Mitchell
tangent-cone theorem [U].

**VERIFIED 2026-08-25 [RUN]:** the grading is exact — `[n_a, n_b] ⊆ n_{a+b}`,
no violations. Adjacent-token edges alone bracket-generate the full algebra
(15/15 at s=6, 5 rounds). Homogeneous/topological dimension = **(s+1)/3 exactly**
at every s from 4 to 512.

**THE CATCH, which is the finding:** a DENSE strictly-lower-triangular A occupies
every subdiagonal and is **NOT horizontal**, so Carnot structure is VACUOUS on
it. The geometry PRESCRIBES sparse local support. `hop = grading` holds only when
A ∈ n_1 (then A^h ∈ n_h exactly); dense A smears across layers.

Kill: S3 as written — if eps^h grading loses to learned scalars at matched
params, the geometry is reported as structure only.

### B2. DISCRETE MORSE THEORY → M4, M5, Lean track
Objects: Forman collapses; free pairs; critical cells; Morse inequalities [V].

Why it survives: the free-face eviction rule IS an elementary collapse, and
`CEQ.Refcount.free_face_floor_unchanged` is a special case of collapse
preserving homotopy type. Elementary enough for Lean. Only item that makes M4's
W9 limitation (no retroactive eviction) structure rather than apology.

Caveat: refcount was measured dead **as a selector** (constant within a
sequence). Morse upgrades the CERTIFICATE, not the ranker.

Kill: if the cache complex's critical cells fail to equal the measured floor on
synthetic cases, the identification is wrong.

### B3. MÖBIUS INVERSION ON INCIDENCE ALGEBRAS → M4 / W9
Objects: Rota's incidence algebra of the causal order; zeta and Möbius elements
[U]. Targets the one measured hopcache weakness: eviction after dependents read
is not retroactive, exact repair O(N).

Kill: derive the hops=2 correction for one evicted position; if cost is Ω(N)
anyway, delete permanently.

### B4. COMPLEX DYNAMICS / FILLED JULIA SETS → depth stacking, joins A2
Layers COMPOSE hop polynomials; stack boundedness is containment of W(A_l) in the
composed filled Julia set, checked by the escape-time loop.

Kill: if trained W(A_l) never approaches the boundary, demote to diagnostics.

### B5. MAGNITUDE HOMOLOGY → eviction/selection probes
Leinster–Shulman graded homology; magnitude function |tX| as scale sweeps [U].
The repo owns the calibrated judge (`dfloor_probe2`: flat 0.250 chance, cosine
control).

Kill: pre-registered from Round 12 — any magnitude-based selector failing to beat
cosine on that instrument dies the ΔFloor death, no appeal.

---

## TIER 3 — REINCARNATED

### C1. SIGN-SOLVABILITY / QUALITATIVE MATRIX THEORY → M2 hardening
Brualdi & Shader, *Matrices of Sign-Solvable Linear Systems*, Cambridge Tracts
in Mathematics 116, CUP **1995** [V].

Returns because it is the ONLY mathematics in which sign survival is
magnitude-INDEPENDENT. With |P| = k small, checking sign-determinacy on the k×k
pivot block is computable, which it never was globally.

**Standing caveat:** sign-DETERMINACY is insensitivity to magnitude, which pushed
to its limit drives the flip rate to ZERO (softmax's number). The salvageable
target is **path coherence** — that the j→i paths through c share a sign so c's
contribution adds instead of cancelling — not sign-nonsingularity per se.

Kill: fraction of sign-determined entries in trained pivot blocks ~0 and not
improvable without loss blowup.

### C2. FRUSTRATION INDEX / SWITCHING CLASSES (Zaslavsky [U]) → M1 integrity
**ALREADY RUN 2026-08-25 [RUN]. Result: the kill does NOT fire.**

| operator | unbalanced triangles | verdict |
|---|---|---|
| tgate | 0.5056 | maximally frustrated → truly signed |
| deltanet | 0.5641 | maximally frustrated |
| sgate | 0.2785 | partially frustrated |
| **softmax** | **0.0000** | balanced → non-negative in disguise (instrument validates itself) |

Gauge-invariance audit came free and is **exact**:
`max|(I−DAD)^{-1} − D(I−A)^{-1}D| = 0.000e+00` on all four operators, so
`sign_flip_rate` measures a switching-invariant quantity and was never counting
gauge artifacts.

Remaining work: re-run on TRAINED checkpoints, not random projections.

### C3. BLACKWELL APPROACHABILITY (1956 [U]) → training the conjunction
M1–M6 is a VECTOR of simultaneous targets; approachability drives a vector payoff
into a convex target set without scalarizing, where weighted sums trade one M
against another.

Kill: if a plain weighted sum reaches the same GREEN set, the machinery is
overhead.

### C4. NUCLEOLUS / CORE ALLOCATION (Schmeidler [U]) → the pivot selector
Pivot SELECTION is a coalition problem: choose P such that no coalition of
excluded tokens routes hop-2 flow better than P claims. Unique, LP-computable,
feasible at small k where hopeless at full s.

Kill: S2's ablation — if nucleolus-selected P loses to top-k salience, ship
salience and delete this.

---

## DEAD — do not revive without new evidence of the stated kind

- **DEQ / Hopfield settling** — direct method exists; Howard PI beat fixed-point
  search 7.9–21.4×. Revive only if a non-nilpotent variant ships.
- **Multigrid, spectral decimation, mixed-curvature manifolds** — R5 deleted;
  locality beat multiscale.
- **Sheaf Laplacian + Hodge harmonic energy** — R6: averaging won by 0.0793
  AUROC. The "contradiction surfaces" premise was false.
- **Nash/QRE, ESS selection, MFG master equation** — the equilibrium they select
  was deleted (W7: 1/5 seeds; DEQ multiplicity moot).
- **Lyapunov-spectrum shaping, Oseledets, Fenichel slow manifolds** — vacuous on
  spectrum {0}; live content migrated into **A2**.
- **IFS/Collage compression, Assouad/multifractal, free-probability hop scaling,
  Lorentzian polynomials** — attachment points deleted or superseded by **B1**.
- **ΔFloor-by-eviction as a selector** — Round 12, calibrated instrument, both
  pre-registered kills fired, CI at s=1024 entirely below chance.
- **Refcount-priced schedule as a selector** — killed by its own Lean proof:
  refcount is constant within a sequence, so the score carries zero bits.
