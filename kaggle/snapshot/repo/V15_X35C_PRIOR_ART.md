# V15 / X₃₅c — PRIOR ART AT EQUATION LEVEL, BEFORE NAMES

JUPITER-4, node `X₃₅c` of the CEQ v15.1 composition round. Discharges
`CEQ_V15_1_DELTA.md` §X₃₅c (`[U, before names]`) against the standing law **L-EQ**
(`CEQ_V15_CONTRACT.md` lines 51-55): `[V]` — page exists, intro matches — is
INADMISSIBLE for a load-bearing statement; `[V-eq]` requires the theorem's
statement WITH HYPOTHESES plus one numeric instance run.

The tag on this node is binding in its ordering: the prior art is fetched
**before** the construction is named or claimed, so that the naming can be
corrected by what is found. §5 and §6 are that correction.

## Admissibility legend

| tag | meaning |
|---|---|
| `[V-eq]` | defining equation transcribed from the source, hypotheses stated as the source states them, and a numeric instance under `scripts/v15_x35_probes/` executed with its real output pasted below |
| `[V]` | page or abstract reached, equation NOT verified — **INADMISSIBLE for any load-bearing statement**, flagged inline |

Every numeric instance carries a **control**: a deliberately mis-transcribed
variant of the same equation whose output differs by O(1). A probe that only
prints `~1e-16` proves nothing about transcription; a probe that prints `~1e-16`
for the equation and `~1e0` for the mis-transcribed variant does.

Reproduce: `python scripts/v15_x35_probes/pNN_*.py` (numpy 1.26.4, scipy 1.17.1,
CPython 3.11.9, Windows 11, this box). Outputs below are pasted verbatim from
those runs.

**Scoreboard for this node: 4 of 4 lineages reach `[V-eq]`, plus a fifth
`[V-eq]` probe on the composition itself (§5), which is the one that decides the
node.** Sub-statements that remain `[V]` are named inline. Nothing here trains.

---

## 1. Latent-variable inference `[V-eq]` — upgraded from the delta's `[V]`

The delta tags this lineage `[V]`, which under L-EQ is inadmissible. It is
upgraded here. The question the delta needs answered is *what is the actual
estimator, what does it assume, what does it guarantee* — and the answer splits
in two, because the estimator and the identifiability are separate results with
separate hypotheses, and the second is the one that binds X₃₅b.

### 1.1 The estimator: EM, and exactly what it guarantees

**Citation.** A. P. Dempster, N. M. Laird, D. B. Rubin, *Maximum Likelihood from
Incomplete Data via the EM Algorithm*, J. Roy. Statist. Soc. B 39(1):1-38, 1977.
Text extracted locally from the JSTOR scan with `pypdf` (see FETCH LOG). The
scan's OCR renders `φ` as `+`; symbols are restored below, structure and
numbering are the source's.

**Equations, as transcribed.**

- Definition of the algorithm (§2, general level):

  ```
  E-step: compute  Q(φ | φ^(p))  =  E[ log f(x | φ) | y, φ^(p) ]
  M-step: choose   φ^(p+1)  to be a value of φ ∈ Ω which maximizes Q(φ | φ^(p))
  ```

  where `y` is the observed (incomplete) data, `x` the complete data with
  `y = y(x)`, `f(x|φ)` the complete-data density, `g(y|φ)` the observed-data
  density, and `L(φ) = log g(y|φ)`.
- Eq (3.5), definition of a GEM algorithm: an iterative algorithm with mapping
  `M(φ)` is a **generalized EM algorithm** if `Q(M(φ) | φ) ≥ Q(φ | φ)` for every
  `φ ∈ Ω`.
- **Theorem 1** (Eq 3.7): *For every GEM algorithm* `L(M(φ)) ≥ L(φ)` *for all*
  `φ ∈ Ω`, with equality if and only if both `Q(M(φ)|φ) = Q(φ|φ)` and
  `k(x|y, M(φ)) = k(x|y, φ)` almost everywhere.
- Lemma 1 (Eq 3.3, Jensen): `H(φ'|φ) ≤ H(φ|φ)` with equality iff the conditional
  densities agree a.e.
- **Theorem 2**: suppose `{φ^(p)}` is an instance of a GEM algorithm such that
  (1) the sequence `L(φ^(p))` is bounded, and (2)
  `Q(φ^(p+1)|φ^(p)) − Q(φ^(p)|φ^(p)) ≥ λ (φ^(p+1) − φ^(p))(φ^(p+1) − φ^(p))^T`
  for some scalar `λ > 0` and all `p`. Then `φ^(p)` converges to some `φ*` in the
  closure of `Ω`.

**Hypotheses as the source states them.** `f(x|φ) > 0` almost everywhere in `X`
for all `φ ∈ Ω`; `Q(φ'|φ)` assumed to exist for all pairs; `y` fixed and known;
Theorem 2 additionally requires bounded `L(φ^(p))` and the curvature condition
(2). **What is NOT claimed is as important as what is:** Theorem 1 gives
monotone ascent of the observed-data likelihood and nothing more. DLR's
Theorem 2 gives convergence of the *iterates*, not convergence to a maximizer;
the convergence analysis was corrected by C. F. J. Wu, *On the Convergence
Properties of the EM Algorithm*, Ann. Statist. 11(1):95-103, 1983,
DOI `10.1214/aos/1176346060`, whose result is that limit points are **stationary
values** of the likelihood under regularity conditions — `[V]`, abstract only,
the full text is paywalled (FETCH FAILED, see log), so **no statement resting on
Wu's exact hypotheses may be load-bearing in this round.** The numeric instance
below exhibits the failure mode directly rather than resting on Wu's wording.

### 1.2 The identifiability: Kruskal's condition

**Citation.** Elizabeth S. Allman, Catherine Matias, John A. Rhodes,
*Identifiability of parameters in latent structure models with many observed
variables*, Ann. Statist. 37(6A):3099-3132, 2009, arXiv:0809.5032,
DOI `10.1214/09-AOS689`.

**Equations and theorems, as transcribed.**

- The model, Eq (1): a latent `Z ∈ {1,…,r}` with `r` known, `π = (π_i) ∈ (0,1)^r`,
  `Σ π_i = 1`; conditional on `Z = i` the `p` observed variables are independent
  with `p_ij ∈ [0,1]^{κ_j}`; `P_i = ⊗_{j=1..p} p_ij` and

  ```
  P = Σ_{i=1..r} π_i P_i                                                   (1)
  ```

  denoted `M(r; κ_1,…,κ_p)`. Kruskal rank of a matrix `M`: the largest `I` such
  that every set of `I` rows of `M` is independent.
- **Theorem 1 (Kruskal).** Let `I_j = rank_K M_j`. If `I_1 + I_2 + I_3 ≥ 2r + 2`,
  then `[M_1, M_2, M_3]` uniquely determines the `M_j`, up to simultaneous
  permutation and rescaling of the rows.
- **Corollary 2.** For `M(r; κ_1, κ_2, κ_3)` with all entries of `π` positive and
  `I_j` the Kruskal rank of the matrix whose rows are `p_ij`: if
  `I_1 + I_2 + I_3 ≥ 2r + 2`, the parameters are uniquely identifiable **up to
  label swapping**.
- **Corollary 3.** The parameters of `M(r; κ_1, κ_2, κ_3)` are **generically**
  identifiable up to label swapping provided
  `min(r,κ_1) + min(r,κ_2) + min(r,κ_3) ≥ 2r + 2`.

**Hypotheses as the source states them.** `r` known; conditional independence of
the observed variables given the latent; positive mixing proportions for
Corollary 2; "generic" means the non-identifiable set has measure zero, and the
paper states plainly that for `r > 1` the classical definition of identifiability
is too strong — parameters are never strictly identifiable, because the sum in
(1) can be reordered. The paper also records Kruskal's own observation that
**two-way tables from `M(r; κ_1, κ_2)` do not have a unique decomposition when
`r ≥ 2`.**

**Numeric instance.** `scripts/v15_x35_probes/p01_latent_em_identifiability.py`.

```
DLR Thm 1  min per-iteration increment of L   : -4.547473508864641e-13
control (inverted-density E-step) min increment: -0.09823100393509776
EM final loglik, best of 20 random starts     : -875.6821513996022
EM final loglik, worst of 20 random starts    : -922.2270525780536
O(1) spread over stationary points            : 46.54490117845148
label swap: |mu_A - mu_C| raw                 : 4.224352623866027
label swap: |sort(mu_A) - sort(mu_C)|         : 0.0
Kruskal/Cor.3  p=2: min(r,k) sum = 4   vs 2r+2 = 6  -> condition FAILS
Kruskal/Cor.3  p=3: min(r,k) sum = 6   vs 2r+2 = 6  -> condition holds
p=2 alternative parameters: table residual 2.776e-17 , parameter distance 0.2642
p=3 alternative parameters: table residual 1.129e-02 , parameter distance 0.1726
p=3 search at the TRUE pi (optimizer capability): residual 0.000e+00 , distance 0.000e+00
```

Reading: the `-4.5e-13` is float noise on a log-likelihood of magnitude `8.8e2`
(`5e-16` relative); the mis-transcribed E-step breaks Theorem 1 by `-9.8e-2`,
eleven orders larger. The `46.5`-nat spread over 20 starts is the guarantee's
actual shape — ascent, not the maximum. The last three lines are Corollary 3 at
its own boundary, checked constructively: with two features the condition fails
and a parameter vector at distance `0.26`, with a *different* mixing proportion,
reproduces the observed table to `2.8e-17`; with three features the condition
holds and the same optimizer cannot get below `1.1e-2` — and the capability line
proves the optimizer is not the limitation, since at the true `π` it recovers the
parameters exactly.

**What this occupies for X₃₅.** Positing a latent node and fitting it is EM (or a
GEM variant). The guarantee is monotone ascent to a stationary value. The
identifiability of *what was fitted* requires at least three conditionally
independent views of the latent and Kruskal's rank inequality, and even then only
**generically and up to label swapping**. A latent posited on a residual at a
single time index has one view, not three. **Any X₃₅b sentence of the form "the
model places the latent" is a statement about a fitted parameter, not about an
identified object, unless the bed plants the latent and the localization is
scored against the plant.**

---

## 2. Hidden-confounder discovery, FCI-class `[V-eq]`

**Citations.**
(a) Diego Colombo, Marloes H. Maathuis, Markus Kalisch, Thomas S. Richardson,
*Learning high-dimensional directed acyclic graphs with latent and selection
variables*, Ann. Statist. 40(1):294-321, 2012, arXiv:1104.5617. Fetched as PDF
and extracted locally; definitions and theorems below are verbatim from that
text.
(b) Jiji Zhang, *On the completeness of orientation rules for causal discovery in
the presence of latent confounders and selection bias*, Artificial Intelligence
172(16-17):1873-1896, 2008, DOI `10.1016/j.artint.2008.08.001` — **primary
FETCH FAILED (403)**; the completeness statement below is transcribed from
Colombo et al., who attribute it to Zhang. Zhang's own wording is `[V]`.
(c) Andreas Gerhardus, Jakob Runge, *High-recall causal discovery for
autocorrelated time series with latent confounders*, NeurIPS 33, 2020
(LPCMCI) — the time-series member of the class, fetched from the NeurIPS
proceedings PDF.

**Equations, definitions and theorems, as transcribed.**

- **Definition 2.1 (m-separation).** A path `π` in an ancestral graph is blocked
  by a set `Y` if and only if `π` contains a subpath `⟨X_i, X_j, X_k⟩` whose
  middle vertex `X_j` is a noncollider on this path and `X_j ∈ Y`, or `π`
  contains a v-structure `X_i *→ X_j ←* X_k` such that `X_j ∉ Y` and no
  descendant of `X_j` is in `Y`.
- **Definition 3.1 (FCI-PAG).** Let `G` be a DAG with partitioned vertex set
  `X ∪̇ L ∪̇ S`. Let `C` be a simple graph with vertex set `X` and edges of the
  type `→, ∘→, ∘∘, ↔, −, −∘`. `C` is an FCI-PAG representing `G` iff, for any
  distribution `P` of `X ∪̇ L ∪̇ S` faithful to `G`:
  1. absence of an edge between `X_i` and `X_j` in `C` implies there exists
     `Y ⊆ X \ {X_i, X_j}` with `X_i ⊥⊥ X_j | (Y ∪ S)` in `P`;
  2. presence of an edge implies `X_i ⊥̸⊥ X_j | (Y ∪ S)` in `P` **for all**
     `Y ⊆ X \ {X_i, X_j}`;
  3. an arrowhead at `X_j` implies `X_j ∉ an(G, X_i ∪ S)`;
  4. a tail at `X_j` implies `X_j ∈ an(G, X_i ∪ S)`.
- **The input to the theorem**, §3 verbatim in substance: the oracle problem is
  posed as being given *all* conditional independence relationships between pairs
  `X_i, X_j ∈ X` given sets `Y ∪ S` with `Y ⊆ X \ {X_i, X_j}`, under faithfulness
  to an unknown DAG `G`.
- **Theorem 3.1 (soundness).** Consider one of the oracle versions of `FCI_path`,
  `CFCI`, `CFCI_path`, `SCFCI` or `SCFCI_path`. Let the distribution of
  `V = X ∪̇ L ∪̇ S` be faithful to a DAG `G` and let conditional independence
  information among all variables in `X` given `S` be the input. Then the output
  is an FCI-PAG of `G`. (Soundness of FCI itself follows from Theorem 5 of
  Spirtes, Meek, Richardson.)
- **Completeness (Zhang 2008), as stated by Colombo et al.**: the output of FCI is
  *maximally informative*, in the sense that **for every circle mark there exists
  at least one MAG in the Markov equivalence class represented by the PAG where
  the mark is oriented as a tail, and at least one where it is oriented as an
  arrowhead**.
- **Theorem 3.2 (RFCI).** The output of RFCI is an RFCI-PAG of `G`; Definition 3.2
  replaces condition (ii) with the weaker (ii′) quantified over adjacency sets, so
  every FCI-PAG is an RFCI-PAG, RFCI-PAG skeletons are in general supergraphs of
  the FCI-PAG skeleton, and an RFCI-PAG can correspond to more than one Markov
  equivalence class.
- **Theorems 4.1 / 4.2 (consistency).** RFCI is consistent in sparse
  high-dimensional settings under (A1)-(A5) with adjacency sets bounded by
  `O(n^{1−b})`; FCI requires the stronger (A3′) bounding *Possible-D-SEP* sets by
  `O(n^{1−b})`.
- **LPCMCI, Theorem 2 (sound and complete).** Assume a process as in eq. (1)
  without causal cycles generating a distribution `P` faithful to its time series
  graph `G`; assume no selection variables and perfect statistical decisions about
  CI of observed variables in `P`. Then LPCMCI is sound and complete, i.e., **it
  returns the PAG `P(G)`.**

**Hypotheses as the sources state them.** Faithfulness of the full distribution
(including latents and selection variables) to a DAG; a CI oracle over the
observed variables; acyclicity; for LPCMCI additionally stationarity of the time
series process, no selection variables, and perfect CI decisions. Every
guarantee in this class is *conditional on the CI oracle* and returns an
*equivalence class*.

**Numeric instance.**
`scripts/v15_x35_probes/p02_fci_pag_latent_localization.py`.

```
(A) latent count in the two generating models     : 1  vs  3
(A) max |Sigma_obs(1 latent) - Sigma_obs(3 latents)|: 5.551115123125783e-17
(A) error variances of model B positive            : True
(A) CI oracles equal                               : True | both empty: True
(B) partial corr X1,X3 | X2  chain, no latent      : -4.334692657276431e-17
(B) partial corr X1,X3 | X2  chain + latent on pair: 0.27208759062772303
(B) O(1) separation                                : 0.2720875906277231
control (marginal corr, conditioning dropped)      : 0.3806168878173307 0.6573936226362445 -> declares dependence in BOTH, difference 0.27677673481891374
```

The probe computes exact population quantities from
`Σ = (I−B)^{-1} diag(err) (I−B)^{-T}`, using the linear-Gaussian equivalence
`m-separation ⟺ vanishing partial correlation`, so that the object compared is
precisely the theorem's input, the CI oracle.

### 2.1 THE CEILING ON `LATENT LOCALIZATION ACCURACY`

This is the answer the round asked for, stated so it can be attacked.

**(i) The number and location of latents is not identified, at all, by anything
that consumes a CI oracle.** Probe part (A): a single latent parent of
`{X, Y, Z}` and three *pairwise* latents produce observed covariance matrices
agreeing to `5.6e-17` with all error variances positive. Their CI oracles are
therefore identical (both empty). Theorem 3.1's input is that oracle, so FCI,
RFCI, FCI+, and LPCMCI must return the same PAG for a generator with one latent
and a generator with three. **A capability column that scores "which latent node
was placed" is not measuring an identifiable quantity.**

**(ii) What IS identified is weaker and pair-local.** Probe part (B): adding a
latent common cause on the pair `(X₁, X₃)` of a chain `X₁ → X₂ → X₃` moves
`ρ(X₁,X₃ | X₂)` from `−4.3e-17` to `0.272`. So the *existence* of a latent common
cause on a named pair is visible in the oracle, and that is what a `↔` edge
records — non-ancestry at both ends, per Definition 3.1(iii), not "there is one
node here".

**(iii) Even the orientation is only fixed up to the class.** By the completeness
statement, every circle mark in the returned PAG is a mark for which the class
contains a MAG with a tail there and a MAG with an arrowhead there. There is no
sample size at which a circle mark resolves; it is not a variance, it is a
quotient.

**(iv) The sequence setting does not escape this.** LPCMCI Theorem 2 is the
strongest available result for autocorrelated time series with latent
confounders, and its conclusion is *it returns the PAG* — the same quotient, in
X₃₅'s own regime. LPCMCI's contribution is recall under autocorrelation, not
identification of latent identity.

**Consequence for X₃₅b, stated plainly.** The column is admissible **only** as
*recovery of a planted node under a known generator* — the bed plants the cause,
the score is against the plant, and the number reported is a property of the
estimator at that SNR, not an identification result. Any sentence that reads
"the model localizes the hidden cause" without "on beds where the cause is
planted and the generator is known" claims something the identifiability
literature says is not there. The ceiling is set by theory, not by the method,
and the theory is not new: it is Definition 3.1 plus the completeness statement,
1997-2008, and 2020 for the time-series case.

---

## 3. Residual / anomaly-detection lineage `[V-eq]` — Shewhart, and dependence

**Citations.**
(a) Michèle Basseville, Igor V. Nikiforov, *Detection of Abrupt Changes: Theory
and Application*, Prentice-Hall 1993; the whole book is downloadable from
`people.irisa.fr/Michele.Basseville/kniga/`. Fetched as PDF, extracted locally.
(b) Thomas Mikosch, Olivier Wintenberger, *Some variations on the extremal
index*, arXiv:2106.05117. Fetched as PDF, extracted locally.
(c) S. M. Berman, *Limit theorems for the maximum term in stationary sequences*,
Ann. Math. Statist. 35:502-516, 1964 — Theorem 3.1, the condition
`lim_k |γ_k| ln k = 0`, transcribed from Turner & Chareka, *A Note on the Berman
Condition*, arXiv:1003.2831 (fetched), which quotes it as its Eq (1.1). Berman's
own text is `[V]`.
(d) R. A. Johnson, M. Bagshaw, *The effect of serial correlation on the
performance of CUSUM tests*, Technometrics 16(1):103-112, 1974,
DOI `10.1080/00401706.1974.10489155` — **`[V]`, abstract only**, paywalled.
(e) W. Schmid, *On the run length of a Shewhart chart for correlated data*,
Statistical Papers 36:111-130, 1995, DOI `10.1007/BF02926025` — **`[V]`,
bibliographic record only**, paywalled; Semantic Scholar returns no abstract.

**Equations and theorems, as transcribed.**

- Basseville & Nikiforov §5.1.1, Eqs (5.1.1)-(5.1.3): a Shewhart control chart is
  a repeated Neyman-Pearson test applied to samples of fixed size `N`; with
  `S₁^N(K)` the log-likelihood ratio of the `K`-th sample,

  ```
  t_a = N K*  =  N min{ K ≥ 1 : d_K = 1 },     d_K = 1 iff S₁^N(K) ≥ h
  S₁^N(K) = Σ_{i=(K−1)N+1}^{KN} ln [ p₁(y_i) / p₀(y_i) ]
  ```

- The ARL derivation, verbatim in its load-bearing step: *the number of samples
  `K` has a geometrical distribution* `P(K = k) = (1 − α₀)^k α₀`, where `α₀` is
  the probability of false alarms of the Neyman-Pearson test; therefore
  `E(K) = 1/α₀` and

  ```
  L(θ₀) = E₀(t_a) = N / α₀                                             (5.1.4)
  L(θ₁) = E₁(t_a) = N / (1 − β₁)                                       (5.1.5)
  L(θ)  = E_θ(t_a) = N / β(θ)                                          (5.1.6)
  ```

  with `β(·)` the power function of the Neyman-Pearson test.
- **Where independence enters, and where it does not.** `α₀` is a property of the
  *marginal* alone — it is the per-sample tail probability of the test statistic.
  The geometric law for `K`, and only that, is where i.i.d. sampling is used.
  This is the whole theoretical content of "memoryless": the threshold is
  calibrated by a marginal, and dependence can only reach the ARL through the
  law of the *count*, never through the law of the statistic.
- Leadbetter's **extremal index** (Mikosch & Wintenberger §1): assume for every
  `τ ∈ (0,∞)` there is a sequence `u_n(τ)` with `n F̄(u_n(τ)) → τ` and a number
  `θ_X` with

  ```
  P(M_n ≤ u_n(τ)) → exp(−τ θ_X),   θ_X ∈ [0,1],   M_n = max(X_1,…,X_n)
  ```

  then `θ_X` is the extremal index, independent of the choice of `u_n`, and is
  interpreted as the reciprocal of the expected extremal cluster size.
- **The Gaussian case, quoted in substance**: if `(X_t)` is a Gaussian stationary
  sequence whose autocovariance satisfies `cov(X_0, X_h) = o(1 / log h)` as
  `h → ∞`, then `θ_X = 1`. (Berman's condition, Berman 1964 Thm 3.1, given in
  Turner & Chareka as `lim_{k→∞} |γ_k| ln k = 0`.)
- **The consequence for a memoryless chart, derived here in one line from the two
  fetched statements:** the first-alarm time `T_u` satisfies
  `P(T_u > n) = P(M_n ≤ u)`, so `E[T_u] ≍ 1 / (θ_X α)`. For a Gaussian AR(1),
  `γ_h = φ^h` decays geometrically, hence `γ_h log h → 0`, hence `θ_X = 1`, hence
  **the i.i.d. value `1/α` is the asymptote of the memoryless chart's ARL₀ under
  dependence.** No such statement is available for a cumulative statistic,
  because a cumulative statistic's fluctuation scale is the long-run variance
  `Σ_k γ(k) = σ²(1+φ)/(1−φ)` for AR(1) and not the marginal — this is the content
  of Johnson & Bagshaw's Wiener-process approximation `[V]`, and it is the same
  factor already named in this campaign at `V13_X26_TRAJECTORY_MONITOR.md` §4.
- Schmid 1995 `[V]`: the in-control ARL of the correlated process is larger than
  in the independent case. Direction only; no equation fetched.

**Hypotheses as the sources state them.** Basseville: fixed sample size `N`,
known `p₀` and `p₁` (composite case handled by §4.2.4), and the geometric law for
`K`, i.e. independence across samples. Mikosch & Wintenberger: strict
stationarity; a high-threshold sequence `u_n(τ)` with `n F̄(u_n) → τ`; the limit
is asymptotic in `n`, so the `θ_X = 1` conclusion is a statement about high
thresholds, not about any particular finite threshold.

**Numeric instance.** `scripts/v15_x35_probes/p03_shewhart_arl_dependence.py`.
All arms hold the **marginal** at `N(0,1)`, so `α` is identical by construction
and only dependence varies.

```
== 1. Shewhart, i.i.d. Gaussian, Basseville Eq (5.1.4) with N = 1 ==
L = 2.50  alpha_0 = 2(1-Phi(L)) = 0.012419
predicted ARL0 = N/alpha_0        : 80.520
measured  ARL0 (B=40000)          : 80.106  +/- 0.401 (1 se), censored 0.0000
ratio measured/predicted          : 0.9949
CONTROL one-sided tail in (5.1.4) : 161.039  -> ratio 0.4974  (O(1) wrong)
run-length sd/mean            : 0.9998   (geometric law: sqrt(1-alpha) = 0.9938)

== 2. Shewhart, AR(1), SAME marginal N(0,1), same threshold ==
phi = 0.5 : ARL0 = 90.333 +/- 0.452 | ratio to i.i.d. = 1.1277 | sd/mean = 0.9962 | cens 0.0000
phi = 0.8 : ARL0 = 136.904 +/- 0.685 | ratio to i.i.d. = 1.7090 | sd/mean = 1.0087 | cens 0.0000

== 3. threshold sweep at phi = 0.8: ratio -> 1 as the threshold rises ==
(Leadbetter/Berman: theta_X = 1 for a Gaussian sequence, so ARL0 -> 1/alpha)
L = 2.0  1/alpha =     22.0  ARL0(iid) =     22.1  ARL0(AR .8) =     41.3  ratio = 1.8685
L = 2.5  1/alpha =     80.5  ARL0(iid) =     80.6  ARL0(AR .8) =    137.2  ratio = 1.7028
L = 3.0  1/alpha =    370.4  ARL0(iid) =    370.7  ARL0(AR .8) =    557.1  ratio = 1.5029
L = 3.5  1/alpha =   2149.3  ARL0(iid) =   2129.7  ARL0(AR .8) =   2914.1  ratio = 1.3684

== 4. CUSUM calibrated on i.i.d. to the SAME ARL0, then run on AR(1) ==
k = 0.50, calibrated h = 2.6455 -> i.i.d. ARL0 = 80.49 (target 80.11), censored 0.0000
phi = 0.5 : CUSUM ARL0 = 36.98 | collapse factor 2.18x | long-run var ratio 3.0 | censored 0.0000
           memoryless chart at the same phi: ARL0 = 90.35 | factor 0.89x
phi = 0.8 : CUSUM ARL0 = 37.86 | collapse factor 2.13x | long-run var ratio 9.0 | censored 0.0000
           memoryless chart at the same phi: ARL0 = 137.72 | factor 0.58x
```

**What this settles, and what it does not.**

1. **Eq (5.1.4) reproduces** to `0.5%` (`80.11 ± 0.40` against `80.52`, about one
   standard error), and the mis-transcribed one-sided tail is wrong by a clean
   factor of two — the control does its job.
2. **Dependence moves the memoryless chart's ARL₀ the SAFE way.** At matched
   marginal, `ARL₀` *rises* to `1.13×` at `φ = 0.5` and `1.71×` at `φ = 0.8`: the
   chart fires *less* often than nominal, not more. This is Schmid's direction,
   measured here rather than quoted.
3. **The inflation shrinks toward 1 as the threshold rises** — `1.87, 1.70, 1.50,
   1.37` at `L = 2.0, 2.5, 3.0, 3.5` — which is what `θ_X = 1` predicts, and it is
   the reason the memoryless choice is not merely lucky: the error is bounded and
   vanishing in the calibration regime that matters, and it is conservative
   throughout.
4. **The CUSUM at matched i.i.d. ARL₀ moves the UNSAFE way**, to `0.46×` and
   `0.47×` of nominal — more than twice the false-alarm budget. The direction is
   opposite to the memoryless chart's, so the two errors do not cancel: at
   `φ = 0.8` the two charts sit a factor of `137.7 / 37.9 = 3.6` apart on the same
   data with the same nominal ARL₀.
5. **Consistency with X₂₆.** `V13_X26_TRAJECTORY_MONITOR.md` §4 measured a `24.1×`
   CUSUM shortfall at `φ = 0.709` with `h = 5.7566` (nominal ARL₀ 1000); this node
   measures `2.1×` at `h = 2.6455` (nominal 80). Same construction, same
   direction, different depth: X₂₆'s own explanation — the CUSUM averages over an
   effective window of order `h/k`, so the Bartlett factor it sees grows with `h`
   — predicts exactly this ordering (`h/k ≈ 11.5` versus `≈ 5.3`). The two
   measurements agree.

**Verdict on the delta's claim.** The delta's line — *CUSUM's dependence
inflation killed X₂₆; the memoryless detector was the one that survived at
matched ARL₀* — is now backed by theory and not only by the campaign's
experiment: the memoryless chart's threshold is a function of the marginal, and
under Berman's condition its ARL₀ converges to the i.i.d. value from the
conservative side, while a cumulative statistic is calibrated by the long-run
variance and cannot be fixed by a marginal. **This is prior art, not a
discovery.** Basseville & Nikiforov 1993 and Leadbetter/Berman 1964-1983 own it
entirely; what the campaign owns is the measurement on its own residual.

---

## 4. Missing-mass inference in astronomy `[V-eq]` — MOTIVATION ONLY

The delta says *as the cited motivation only*, not as a method claim. That
instruction is discharged here in the strict form: the equation is fetched and
run, and the section then states explicitly what is **not** imported.

**Citations.** Gianfranco Bertone, Dan Hooper, Joseph Silk, *Particle dark
matter: evidence, candidates and constraints*, Phys. Rept. 405:279-390 (2005),
arXiv:hep-ph/0404175, §2.1. History and provenance: Gianfranco Bertone, Dan
Hooper, *A History of Dark Matter*, arXiv:1605.04909, Rev. Mod. Phys. 90, 045002
(2018).

**Equation, as transcribed.**

- Bertone-Hooper-Silk Eq (37): *in Newtonian dynamics the circular velocity is
  expected to be*

  ```
  v(r) = sqrt( G M(r) / r ),        M(r) ≡ 4π ∫ ρ(r) r² dr             (37)
  ```

  *and should be falling `∝ 1/√r` beyond the optical disc. The fact that `v(r)`
  is approximately constant implies the existence of an halo with `M(r) ∝ r` and
  `ρ ∝ 1/r²`.*

**Hypotheses as the source states them.** Newtonian dynamics; circular orbits;
spherically enclosed mass `M(r)`; `ρ(r)` the mass density profile.

**Provenance of the lineage, from the history review.** Le Verrier and Adams,
1846: anomalies in the motion of Uranus were explained by positing a new planet,
and Galle identified Neptune within 1 degree of the predicted position the same
evening. Le Verrier then applied the same residual argument to Mercury's
perihelion precession and posited Vulcan — which does not exist, the residual
being resolved instead by replacing the model with general relativity. Zwicky
1933/1937 applied the virial theorem to Coma and obtained a mass-to-light ratio
far above the luminous estimate.

**Numeric instance.** `scripts/v15_x35_probes/p04_missing_mass_astronomy.py`.

```
(a) Eq (37) with rho ~ 1/r^2 : v(r) over 2-30 kpc
    v mean = 23.2480 km/s, max|v - mean|/mean = 3.056e-16
    CONTROL (M ~ Int rho r dr): max|v-mean|/mean = 1.226, max/min = 3.555
(b) d log M / d log r        : 1.000000000000   (source says M(r) ~ r)
(c) M_dyn(<22 kpc) from Eq (37) = 6.883e+10 Msun
    M_lum(<22 kpc), exponential disc = 1.162e+10 Msun
    missing-mass ratio M_dyn / M_lum  = 5.92
    CONTROL (no sqrt) M_dyn          = 5.934e+08 Msun  -> off by 1.160e+02
```

**Stated explicitly, as the delta requires: this is a motivation and NOT a method
import.** The lineage supplies no estimator, no onset statistic, no threshold, no
false-alarm calibration and no localization procedure that X₃₅ could adopt. Its
inference is a static comparison of two mass estimates at the same radius, not a
sequential detection. **No methodological connection was found, and none is
manufactured here.**

**One structural point does survive, and it is a warning rather than a method.**
The same residual argument, applied twice by the same person with the same
rigour, produced Neptune and Vulcan. The difference between them is not in the
detector; it is in whether the *model being subtracted* was right. That is
precisely the delta's own KILL — *no-plant residual fails flatness ⇒ X₃₅ VOID*,
*the detector cannot outrun the model it subtracts* — and the astronomy lineage
is the canonical instance of its failure mode, 1859-1915, which is a better use
of the citation than the discovery half.

---

## 5. THE QUESTION THE ROUND NEEDS ANSWERED

> Is the composition — Shewhart onset on the residual of a learned visible model,
> feeding a latent node that the carrier then re-propagates — new, or is it a
> known pattern under another name?

**Verdict: it is a known pattern, under two names, and both are fetched at
equation level. X₃₅ as stated is occupied. `[V-eq]`**

### 5.1 The 1993 name: additive change detection in state-space models

**Citation.** Basseville & Nikiforov 1993, §7.2.4 *State-Space Models*, and
§7.2.5 *Statistical Decoupling for Diagnosis*, §7.2.6 *Statistical
Detectability*. The origin of the joint detection/identification idea is
attributed there to A. S. Willsky, H. L. Jones, *A generalized likelihood ratio
approach to the detection and estimation of jumps in linear systems*, IEEE Trans.
Automatic Control 21(1):108-112, 1976, DOI `10.1109/TAC.1976.1101146`.

**The composition, transcribed.**

- Base model (7.2.94): `X_{k+1} = F X_k + G U_k + W_k`, `Y_k = H X_k + J U_k + V_k`.
- The hidden cause (7.2.97):

  ```
  X_{k+1} = F X_k + G U_k + W_k + Γ Υ_x(k, t₀)
  Y_k     = H X_k + J U_k + V_k +   Υ_y(k, t₀)
  ```

  with `Υ_x(k,t₀) = Υ_y(k,t₀) = 0` for `k < t₀`, `t₀` the **unknown** change time,
  and (7.2.100) the scalar-magnitude case `… + ν Υ_x(k,t₀)` with `ν` unknown.
  The source states plainly that neither the gain matrices nor the profiles need
  be completely known a priori.
- The residual is the model's own innovation, and the hidden cause's effect on it
  is *propagated through the model's dynamics* — (7.2.109), (7.2.110):

  ```
  X_k = X_k⁰ + α(k,t₀),   X̂_{k|k} = X̂⁰_{k|k} + β(k,t₀),   ε_k = ε_k⁰ + ρ(k,t₀)

  α(k,t₀) = F α(k−1,t₀) + Γ Υ_x(k−1,t₀)
  β(k,t₀) = (I − K_k H) F β(k−1,t₀) + K_k [ H α(k,t₀) + Υ_y(k,t₀) ]
  ρ(k,t₀) = H [ α(k,t₀) − F β(k−1,t₀) ] + Υ_y(k,t₀)
  α(t₀,t₀) = 0,  β(t₀−1, t₀) = 0
  ```

  `ρ(k,t₀)` is named in the source the **signature of the change on the
  innovation**; at steady state it depends only on `k − t₀` (7.2.111-7.2.112).
- Onset and magnitude, jointly (7.2.123)-(7.2.126):

  ```
  ν̂_k(j)   = [ Σ_{i=j..k} ρ̃ᵀ(i,j) Σ_i^{-1} ε_i ] / [ Σ_{i=j..k} ρ̃ᵀ(i,j) Σ_i^{-1} ρ̃(i,j) ]
  sup_ν S_j^k = ½ [ Σ ρ̃ᵀ Σ^{-1} ε ]² / [ Σ ρ̃ᵀ Σ^{-1} ρ̃ ]
  g_k = max_{1 ≤ j ≤ k} sup_ν S_j^k ,      t̂₀ = argmax_{1 ≤ j ≤ t_a} S_j^{t_a}
  ```

  and the source states the mechanism in words: the basic computation is the
  correlation between the Kalman filter's innovations and the signatures of the
  changes on those innovations.
- **Detectability, Definition 7.2.1:** with `s(y) = ln[p₁(y)/p₀(y)]` and
  `K(θ₁,θ₀) = E₁[s(Y)] ≥ 0` the Kullback information, the change is *detectable*
  iff `K(θ₁,θ₀) > 0`; **Definition 7.2.2** extends this to composite hypotheses by
  `inf_{θ₀ ∈ Ω₀, θ₁ ∈ Ω₁} K(θ₁,θ₀) > 0`, with the explicit reading that when the
  parameter sets intersect the two measures cannot be discriminated;
  **Definition 7.2.3** is robust detectability for a chart tuned at wrong
  parameter values.

**Map to X₃₅, term by term.**

| X₃₅ (delta) | occupant (Basseville & Nikiforov 1993 §7.2.4) |
|---|---|
| `r = z_obs − z_model(visible)` | the innovation `ε_k` of the Kalman filter for the *unchanged* model — the source's own remark is that it is the innovation before `t₀` and the residual after |
| onset by a comparator on residual energy | (7.2.118), the known-magnitude log-likelihood ratio `Σ ρᵀΣ⁻¹ε − ½ Σ ρᵀΣ⁻¹ρ` — the innovation correlated with the signature, less half the signature's energy — and (7.2.124)/(7.2.125) its `ν`-maximized form, a squared correlation normalised by that energy; the memoryless comparator is §5.1.1's chart |
| report `(onset, magnitude, …)` | `t̂₀` by (7.2.123) and `ν̂` by (7.2.126), estimated jointly |
| S2 posits a latent node; S-M re-propagates with it | the change `(Γ, Υ_x, Υ_y)` is propagated by the model's own `F`, `H`, `K` through (7.2.110); the "re-propagation" *is* the signature |
| which latent (localization) | §7.2.5, statistical decoupling for diagnosis, over a **pre-specified** candidate set of changes |
| when localization is possible at all | §7.2.6, Definitions 7.2.1-7.2.3, a Kullback-information detectability condition |
| threshold calibrated on no-plant runs | §5.1.1, `α₀` of the Neyman-Pearson test |

**Numeric instance — the composition, run from the 1993 equations.**
`scripts/v15_x35_probes/p05_innovation_glr_composition.py`. The planted cause
enters a state coordinate that is never observed directly, so it is hidden in the
sense X₃₅ means.

```
== (7.2.110) signature, (7.2.123)/(7.2.126) GLR ==
  no-plant threshold at 5% FAR : g* = 3.289
  no plant  -> onset called    : 0.050   (X35a MUST-FIRE 2, target 0.05)
  nu = 0.15 : detect 1.000 | onset within +/-1 0.388 | median |t0hat-t0| 2.0 | median nuhat 0.1507
  nu = 0.30 : detect 1.000 | onset within +/-1 0.690 | median |t0hat-t0| 1.0 | median nuhat 0.2992
  nu = 0.60 : detect 1.000 | onset within +/-1 0.948 | median |t0hat-t0| 0.0 | median nuhat 0.5988

== CONTROL: rho = H alpha, feedback term dropped ==
  no-plant threshold at 5% FAR : g* = 3.024
  no plant  -> onset called    : 0.050   (X35a MUST-FIRE 2, target 0.05)
  nu = 0.15 : detect 1.000 | onset within +/-1 0.060 | median |t0hat-t0| 8.0 | median nuhat 0.0265
  nu = 0.30 : detect 1.000 | onset within +/-1 0.003 | median |t0hat-t0| 7.0 | median nuhat 0.0531
  nu = 0.60 : detect 1.000 | onset within +/-1 0.000 | median |t0hat-t0| 7.0 | median nuhat 0.1064
```

**Both of X₃₅a's MUST-FIREs are discharged by 1993 equations on the first
attempt.** Onset within ±1 in `94.8%` of 400 seeds at `ν = 0.60`, magnitude
recovered to `0.5988` against a true `0.60`, and no-plant onsets called at exactly
the calibrated `5%`. The control — the same signature recursion with the Kalman
feedback term `−F β(k−1,t₀)` dropped, an O(1) mis-transcription — keeps the
false-alarm rate (it is calibrated on its own null) but destroys localization
(`0.000` within ±1) and biases the magnitude by `5.6×`. That is the difference
between transcribing (7.2.110) and guessing it.

The sweep also prices the delta's second KILL: **localization accuracy is a
function of the plant magnitude and must be reported with it** — `0.39, 0.69,
0.95` within ±1 at `ν = 0.15, 0.30, 0.60` on the same generator.

### 5.2 The 2026 name: residual-anomaly change-point detection on a learned model

**Citation.** Yuhe Bai, Chengli Tan, Jiaqi Li, Xiangjun Wang, Zhikun Zhang,
*Residual-loss Anomaly Analysis of Physics-Informed Neural Networks: An Inverse
Method for Change-point Detection in Nonlinear Dynamical Systems with Regime
Switching*, arXiv:2604.25655v1 [stat.ML], 28 April 2026. Fetched as PDF,
extracted locally.

This is the *learned-model* version of the same composition, published four
months before this round. Its structure: a neural model is fitted; the local
residual energy is examined on overlapping subintervals; an elevation of that
energy localizes the transition; then the change point and the piecewise
parameters are estimated **jointly** under one loss.

**Transcribed, because the load-bearing part is a theorem and not a heuristic.**

- Assumption 1 (parameter-affine structure and local identifiability):
  `f(t,x;θ) = G(t,x) θ + b(t,x)` (3.15), with the information matrix
  `M(J) = ∫_J G(t,x*(t))ᵀ G(t,x*(t)) dt` (3.17) satisfying
  `λ_min(M(J)) ≥ α |J|` (3.18) for every subinterval `J ⊂ I` of positive length.
- Assumption 2: quadrature consistency of the discrete residual (3.19).
- **Theorem 3.1 (residual lower bound on change-point subintervals).** With
  `I_k^- = I_k ∩ [0,τ)`, `I_k^+ = I_k ∩ [τ,T]`, the continuous-time residual
  energy `R_k(θ) = ∫_{I_k} ‖ dx*/dt − f(t,x*;θ) ‖₂² dt` (3.20) and
  `R_k = inf_{θ∈Θ} R_k(θ)` (3.21): if `τ ∉ I_k` then `R_k = 0` in the absence of
  observational noise, and if `τ ∈ I_k` with `θ⁻ ≠ θ⁺` then

  ```
  R_k ≥ α · [ |I_k^-| |I_k^+| / ( |I_k^-| + |I_k^+| ) ] · ‖θ⁻ − θ⁺‖₂²        (3.22)
  R_k ≥ (α/2) · min{ |I_k^-|, |I_k^+| } · ‖θ⁻ − θ⁺‖₂²                        (3.23)
  ```

- **Theorem 3.3 (post-change residual lower bound under parameter mismatch)**
  gives the analogous bound for a residual computed with a stale parameter.

**This is X₃₅a's must-fire as a theorem.** (3.22) says the residual energy is
zero off the change point and bounded below by a constant times the squared jump
magnitude on any window straddling it — which is the formal content of *planted
hidden cause ⇒ onset*, *no plant ⇒ no onset*, and *localization degrades with the
plant magnitude*, all three, with the constant named. It is also the formal
content of the delta's KILL: `α` in (3.18) is a local-identifiability constant, so
where the dynamics are insensitive to the hidden parameter the bound is vacuous —
the detector cannot outrun the model it subtracts, stated as an eigenvalue.

### 5.3 What, precisely, is left unoccupied

Everything below is what survives after the two occupancies above, and it is thin.
It is written so it can be attacked rather than defended.

1. **The residual is that of a *learned sequence model with unknown structure*,
   not a Kalman filter for a known LTI system and not a PINN for a known ODE
   family.** Basseville's `ρ(k,t₀)` requires `F, H, K, Γ, Υ` — the signature is
   computed *from the known model*. The PINN paper requires Assumption 1's
   parameter-affine vector field. Neither transfers as written to a model whose
   forward operator is learned and whose `Γ` is not enumerable. The gap is real
   but it is a **transfer gap, not a new mechanism**: the composition is the same
   three pieces in the same order.
2. **The latent is posited as a NODE IN THE CARRIER'S GRAPH and re-propagated by
   the same learned operator (S-M), rather than as an additive input with a known
   direction `Γ`.** In the fetched literature the "re-propagation" is by the known
   `F/H/K`; making the *carrier* do it is the only structural difference this node
   could find, and it buys a testable difference only if `Γ` is not enumerated in
   advance — which is exactly the case §7.2.5 handles by pre-specifying the
   candidate set.
3. **Scoring `LATENT LOCALIZATION ACCURACY` as a capability column on beds with
   planted causes.** No fetched paper reports that column. But per §2.1 the column
   is only meaningful as recovery-against-a-plant, and per §5.1's probe the
   textbook estimator already scores `0.948` on it at `ν = 0.60` — so the column
   is a benchmark contribution, not a capability discovery, and it needs the
   1993 GLR as its baseline arm or it measures nothing.
4. **The `ARL₀`-matched comparison of a memoryless chart against CUSUM on a
   dependent learned residual** is a measurement this campaign has made
   (`X₂₆`, and §3 above) and is worth reporting; the *theory* backing it is
   Berman/Leadbetter plus Basseville (5.1.4) and is not the round's.

**A negative answer costs one node here and would have cost a capability column
at it.28, so it is filed plainly: X₃₅'s instrument (X₃₅a) is fully occupied, and
its architecture step (X₃₅b) is occupied except for the choice to make the
learned carrier perform the re-propagation. The round should name X₃₅ as a
*transfer* of GLR change detection in state-space models to a learned sequence
model, cite Willsky & Jones 1976 / Basseville & Nikiforov 1993 §7.2.4 before it
names anything of its own, and report the 1993 estimator as the baseline arm of
its own capability column.**

---

## 6. WHERE THE DELTA'S TEXT NEEDS CORRECTING, BEFORE ANYTHING IS NAMED

**(a) `LATENT LOCALIZATION ACCURACY` needs its scope in the column name.** As
written, X₃₅b promises a capability the identifiability results forbid off a
planted bed (§2.1(i): one latent and three latents give the same observed
covariance to `5.6e-17`). Admissible form: *localization accuracy of a planted
cause under a known generator, at stated noise*. Inadmissible form: any sentence
implying the location is identified from data.

**(b) "Onset by a MEMORYLESS comparator on residual energy (Shewhart)" is right
for a better reason than the delta gives.** The delta defends it by the campaign's
experiment. The theoretical defence is stronger and now transcribed: the
threshold is a functional of the marginal only, and under Berman's condition the
extremal index of a Gaussian residual is 1, so the ARL₀ error under dependence is
bounded, conservative, and vanishing as the threshold rises (§3, `1.87 → 1.37`
over `L = 2.0 → 3.5`). Add the citation; the claim then does not rest on X₂₆
alone.

**(c) The must-fire numbers should be reported against a baseline, not against
zero.** `[RUN: exact at 0 noise, 36 vs 37 at sd 0.05]` is a fine number, but §5.1
shows a 1993 estimator reaching `0.948` within ±1 at `ν = 0.60` on a hidden-state
plant. Any X₃₅ number without that baseline beside it is uninterpretable.

**(d) The astronomy citation is currently pointed at the wrong half.** Missing
mass is cited as motivation for *finding* a hidden cause; the more useful half of
the same lineage is Vulcan, which is the delta's own KILL condition realised
historically. Cite both or cite the second.

**(e) One `[V]` in this node must not become load-bearing.** Wu (1983), Johnson &
Bagshaw (1974), Schmid (1995) and Zhang (2008) are all `[V]` here — statements
obtained, primary equations not fetched. Nothing in §§1-5 rests on their wording:
§1's guarantee comes from DLR Theorem 1, §3's dependence result from
Mikosch-Wintenberger plus the fetched Basseville derivation, §2's completeness
from Colombo et al.'s own sentence. If a later node needs Wu's hypotheses or
Zhang's rule set, they must be fetched first.

---

## FETCH LOG (successes and failures, per the no-invention rule)

| target | route | result |
|---|---|---|
| Colombo et al., RFCI/FCI | `arxiv.org/pdf/1104.5617`, local `pypdf` | OK (Def 2.1, 3.1, 3.2, Thm 3.1, 3.2, 4.1, 4.2, the completeness sentence) |
| Colombo et al. | `ar5iv.labs.arxiv.org/html/1104.5617` via WebFetch | partial — summary was substantively right but compressed the theorem statements; **not used**, the PDF text was used instead |
| Zhang 2008 (AIJ completeness) | `commons.ln.edu.hk` PDF | **FETCH FAILED — HTTP 403.** Completeness taken from Colombo et al.'s transcription; Zhang's own wording `[V]` |
| Allman, Matias, Rhodes | `arxiv.org/pdf/0809.5032`, local `pypdf` | OK (Eq 1, Thm 1, Cor 2, Cor 3, the label-swap and generic-identifiability remarks) |
| Allman et al. | `ar5iv.labs.arxiv.org/html/0809.5032` | **FETCH FAILED** — "Conversion to HTML had a Fatal error"; recovered by PDF |
| Dempster, Laird, Rubin 1977 | `web.mit.edu/6.435/www/Dempster77.pdf`, local `pypdf` | OK (E/M steps, Eq 3.5, Lemma 1, Thm 1 Eq 3.7, Thm 2). OCR of the JSTOR scan renders `φ` as `+`; symbols restored, structure verbatim |
| Wu 1983 | `projecteuclid.org/.../10.1214/aos/1176346060.pdf` and `.full` | **FETCH FAILED** — Project Euclid returns an HTML block page for the PDF; abstract only. `[V]` |
| Neal & Hinton 1998 (EM as free energy) | `cs.toronto.edu/~radford/ftp/emk.pdf`, local `pypdf` | **FETCH FAILED** — Type-3 font encoding, extracted text is mojibake. Not used |
| Balakrishnan, Wainwright, Yu (EM guarantees) | `arxiv.org/pdf/1408.2156`, local `pypdf` | partial — extraction produced binary-flagged text; **not used** |
| Richardson & Spirtes 2002, ancestral graphs | Project Euclid PDF | **FETCH FAILED** — HTML block page. m-separation taken from Colombo et al. Def 2.1 instead |
| Basseville & Nikiforov 1993 (whole book) | `people.irisa.fr/Michele.Basseville/kniga/kniga.pdf`, local `pypdf` | OK, 469 pages (§5.1.1 Eq 5.1.1-5.1.6; §7.2.4 Eq 7.2.94-7.2.126; §7.2.5; §7.2.6 Def 7.2.1-7.2.3) |
| Mikosch & Wintenberger, extremal index | `arxiv.org/pdf/2106.05117`, local `pypdf` | OK (Leadbetter's definition, the exponential limit, the Gaussian `θ = 1` statement) |
| Turner & Chareka, Berman condition | `arxiv.org/pdf/1003.2831`, local `pypdf` | OK (Eq 1.1, attribution to Berman 1964 Thm 3.1 p.510) |
| Johnson & Bagshaw 1974 | tandfonline abstract; Semantic Scholar API | **FETCH FAILED** — paywalled, API returns no abstract. `[V]`, method described only |
| Schmid 1995 | `link.springer.com/article/10.1007/BF02926025`; Semantic Scholar API | **FETCH FAILED** — SSO redirect; API returns bibliographic record with no abstract. `[V]` |
| Gerhardus & Runge, LPCMCI | `proceedings.neurips.cc/.../94e70705…-Paper.pdf`, local `pypdf` | OK (Thm 2 sound and complete, Thm 3 order-independence) |
| Bai et al. 2026, PINN residual CPD | `arxiv.org/pdf/2604.25655`, local `pypdf` | OK (Assumptions 1-2, Thm 3.1 Eq 3.15-3.23, Thm 3.3 Eq 3.72-3.74) |
| Bertone, Hooper, Silk 2005 | `arxiv.org/pdf/hep-ph/0404175`, local `pypdf` | OK (Eq 37 with its stated hypotheses and implication) |
| Bertone & Hooper, history of dark matter | `arxiv.org/pdf/1605.04909`, local `pypdf` | OK (Le Verrier/Neptune/Vulcan, Zwicky's Coma numbers) |
| Zwicky 1937 equations | `ned.ipac.caltech.edu/level5/Sept01/Zwicky/Zwicky3.html` | **FETCH FAILED, and instructively.** The page's displayed equations (19)-(23) are GIF images; the raw HTML contains no formulas. The WebFetch summarizer nevertheless returned confident LaTeX for all of them, including `3<v²> = GM/R`, which is not the virial relation for a uniform sphere. **The summary was discarded.** The virial-theorem route is therefore `[V]`; §4 rests on Bertone-Hooper-Silk Eq (37), which was fetched |
| Biviano, Coma cluster review | `arxiv.org/pdf/astro-ph/9711251`, local `pypdf` | OK for history and Zwicky's `M > 5×10^14 M☉`; no virial equation printed |
| Willsky & Jones 1976 | Semantic Scholar API (title/venue/DOI) | OK for the citation; primary text not fetched, so only the attribution — made by Basseville & Nikiforov §2.5 — is used, and no Willsky-Jones equation is quoted |

---

## FILES

- `scripts/v15_x35_probes/p01_latent_em_identifiability.py` — DLR Theorem 1,
  EM's stationary-point spread, label swapping, Kruskal/Corollary 3 at its
  boundary, with an inverted-density E-step as the O(1) control.
- `scripts/v15_x35_probes/p02_fci_pag_latent_localization.py` — one latent versus
  three at identical observed covariance and identical CI oracle; the pair-local
  existence claim that *is* identified; marginal-for-partial correlation as the
  O(1) control.
- `scripts/v15_x35_probes/p03_shewhart_arl_dependence.py` — Basseville Eq (5.1.4)
  reproduced; memoryless ARL₀ under AR(1) at matched marginal; the threshold sweep
  toward `θ_X = 1`; CUSUM at matched i.i.d. ARL₀; one-sided-tail as the O(1)
  control.
- `scripts/v15_x35_probes/p04_missing_mass_astronomy.py` — Eq (37), the flat-curve
  implication `M(r) ∝ r`, the mass discrepancy; a one-power-of-`r` error in `M(r)`
  and a dropped square root as the two O(1) controls.
- `scripts/v15_x35_probes/p05_innovation_glr_composition.py` — the X₃₅
  composition run from Basseville & Nikiforov (7.2.110)/(7.2.123)/(7.2.126):
  onset within ±1, magnitude, and the no-plant control, with a dropped
  Kalman-feedback term in the signature as the O(1) control.
