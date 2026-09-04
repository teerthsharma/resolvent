# SWEEP — lineage S8: statistics and classical mathematics the paper cites

MERCURY-S8, 2026-09-03. Companion file: `bib_methods.bib` (same directory; every key below
is a key there). Repository read at branch `v17k-gate0`, HEAD `207e7b9`. No code written; no
git writes. Sixteen repository files were read for locators (`READ path:line` throughout).

## 0. Method and evidence classes

**What "owns" means.** A source *owns* a component of the shape when it states, in print,
the equation or mechanism the shape uses, with the same object on both sides — not a
neighbouring idea. Where a source states a neighbouring idea it is a NEAR-MISS and the
sweep says what is missing.

**Routes to [V].** For each source one of four pages was fetched this session and the
title compared with the record's claim: (i) the arXiv abs page; (ii) the DOI registry
record at `api.crossref.org/works/<doi>` — the identifier's own resolution, which returns
the registered title, authors, container, volume, issue, pages and year; (iii) the
publisher's landing page; (iv) a library catalogue record (numdam, Google Books by ISBN,
an author's laboratory publication list). Route (ii) carried most of the load: on the
first pass the publisher landing pages returned HTTP 403 or a login redirect for **17 of
the 24** DOIs tried (Springer 3, Taylor & Francis 3, Elsevier 3, SIAM 2, IEEE 2, AMS 1,
PNAS 1, AIP 1, OUP 1), while the registry record resolved every one of them. A registry
record verifies the *identifier–title pairing*, which is exactly the fabrication pattern
that must be excluded (a real DOI with an invented title); it does not verify a theorem
number inside a book. **Theorem locators therefore carry their own class**: [V] only when
the theorem statement was read from the body this session (one case: Levin–Peres Theorem
13.10), otherwise [U] with the record's locator quoted.

**Counts** (from `bib_methods.bib`, grep on `note = {[V]` / `note = {[U]`): **80 entries
[V]**, **9 entries [U]** (Cantelli 1928, Bonferroni 1936, Frobenius 1912, Neumann 1877,
Kantorovich–Rubinstein 1958, Grünwald 1867, Letnikov 1868, Kramers 1927, Littlewood–Offord
1943). A further 11 entries are [V] on the title but carry one field marked [U] (an ISBN,
a page range, a year) — each says which field.

**Quotation rule.** At most one quotation per source, under 15 words; the only quotation
in this file is the Theorem 13.10 inequality.

---

## 1. Sources, by group

Format per source: **key** — class — *owns* — *leaves open relative to the shape* — *found by*.

### 1.A Sequential and classical statistics (instruments for the beds)

- **ville-1939-collectif** — [V] numdam record — *owns* the maximal inequality for a
  nonnegative martingale, `P[sup M_n ≥ C] ≤ E[M_0]/C` (the record's METHODS.md:343
  locates it at Théorème 1 p. 84, [U]) — *leaves open* nothing about the shape; it is the
  type-I guarantee of the record's e-process (METHODS.md:16-41) — *found by* seed.
- **wang-2023-extended-ville** — [V] arXiv abs — *owns* the supermartingale form of
  Ville's inequality the record actually uses (METHODS.md:344 cites §2.1 eq. (3), [U]) and
  its extension to nonintegrable supermartingales — *leaves open* nothing — *found by* seed.
  Correction: the record carried this id as [U]; the title is now matched.
- **ramdas-2023-game-theoretic** — [V] arXiv abs + Crossref (Statistical Science 38(4), 2023)
  — *owns* the e-value / e-process framework and safe anytime-valid inference — *leaves
  open* nothing; it is the umbrella for the record's sequential verdicts — *found by* seed.
- **robbins-1971-almost-supermartingales** — [V] Crossref — *owns* the almost-supermartingale
  convergence theorem (the record's backstop, METHODS.md:165-172) — *leaves open* nothing.
  Correction: PRIOR_ART.md:519 and METHODS.md:346 mark the original NOT REACHED; the DOI
  record now resolves and pairs with the title (full text still unread) — *found by* seed.
- **durrett-2019-probability** — [V] Crossref (5th ed.) — *owns* the textbook statements the
  record cites (Thm 4.2.12, Ex. 4.8.2, both [U]) — *found by* seed.
- **schuirmann-1987-tost** — [V] Crossref — *owns* the two one-sided tests procedure for
  equivalence — *leaves open* the sequential (anytime-valid) form; widening search S8
  found 2026 preprints on anytime-valid and post-hoc-margin equivalence testing but none
  was fetched, so **the sequential TOST the brief's "TOST N = 70" would want is NOT FOUND
  at [V]** — *found by* seed.
- **clopper-1934-binomial** — [V] OUP landing — *owns* the exact binomial interval
  (MATHEMATICS.md:393) — *found by* seed.
- **mcnemar-1947-correlated** — [V] Cambridge landing — *owns* the paired-proportions test
  (MATHEMATICS.md:394,408) — *found by* seed.
- **wilson-1927-probable** — [V] Crossref — *owns* the score interval — *found by* seed.
- **page-1963-ordered** — [V] Crossref — *owns* Page's L and its use for ordered
  alternatives (MATHEMATICS.md:619-626 derives the exact null by convolution; that exact
  enumeration is the record's, Page gives tables and the normal approximation — [U] on
  what Page's body contains) — *found by* seed.
- **ayer-1955-empirical** — [V] Crossref — *owns* the pool-adjacent-violators algorithm
  (the origin) — *leaves open* nothing; **provenance note:** MATHEMATICS.md §12 attributes
  PAVA to "isotonic regression" generally; the citable origin is Ayer et al. 1955, the
  textbooks are Barlow et al. 1972 and Robertson–Wright–Dykstra 1988 — *found by* S7.
- **barlow-1972-order** — [V] via JRSS-A review record; ISBN [U] — *owns* the textbook
  theory of isotonic regression — *found by* seed + S7.
- **robertson-1988-order** — [V] via J. Appl. Econometrics review record (ISBN in the
  review's title string) — *owns* the second textbook — *found by* seed + S7.
- **hoeffding-1963-probability**, **azuma-1967-weighted** — [V] Crossref (Azuma pages [U])
  — *own* the bounded-increment concentration inequalities the contract's M9 F1/F1′ use
  (CEQ_V20_R15_CONTRACT.md:213-217) — *found by* seed.
- **cantelli-1928-confini**, **bonferroni-1936-classi** — **[U] NOT FETCHED** — the
  contract marks Cantelli and Boole/Bonferroni [V] (CEQ_V20_R15_CONTRACT.md:213-214)
  without a locator; this sweep found no fetchable primary page for either and records
  them as bibliographic-only. Any paper sentence resting on them should cite a textbook
  statement instead (Durrett, [V]) — *found by* seed.
- **sander-2023-topk** — [V] arXiv abs — *owns* the reduction of a differentiable sparse
  top-k to isotonic optimisation (results/r9_maths_survey.md:269) — *leaves open* nothing
  for the shape — *found by* a search to repair the record's missing id (a first guess,
  2302.11294, resolved to an unrelated paper and was discarded).

### 1.B Information theory and approximation (floors)

- **cover-2006-elements** — [V] Crossref + publisher sample PDF (ISBN and TOC read) —
  *owns* Fano's inequality (§2.10) and the Gaussian rate–distortion function (§10.3.2),
  the contract's M11 and M12 floors (CEQ_V20_R15_CONTRACT.md:226-234). **Theorem numbers
  2.10.1 and 10.3.2 are [U]** — the sample shows section headings, not theorem numbers —
  *found by* seed.
- **welch-1974-lower** — [V] Crossref — *owns* the Welch bound (MATHEMATICS.md:812).
- **ziv-1969-lower**, **bell-1997-extended** — [V] Crossref — *own* the Ziv–Zakai bound
  and its vector extension; the record ships the scalar one (`ceq/x35p/crb.py:48,178`).
  Correction: a first DOI guess for Bell et al. resolved to a different paper and was
  discarded; the bib carries the matched DOI.
- **johnson-1984-extensions** — [V] Crossref — *owns* the JL lemma.
- **eckart-1936-approximation**, **mirsky-1960-symmetric** — [V] Crossref — *own* the
  low-rank approximation theorem (Frobenius norm; every unitarily invariant norm).
- **adamjan-1971-analytic**, **glover-1984-hankel** — [V] Crossref — *own* Hankel-norm
  model reduction and the L∞ error bounds behind the contract's M16 Hankel bound
  (CEQ_V20_R15_CONTRACT.md:257-259). Correction: the contract carried AAK/Glover as
  `[U → V-eq at it.6]`; both are now title-matched at the registry.

### 1.C Markov chains, resolvents, harmonic functions, electrical networks — the shape's operator

- **levin-2017-markov** — [V] Crossref + the authors' second-edition PDF read — *owns*
  Theorem 13.10, read this session at printed p. 183 of that PDF: for `λ₂` the second
  largest eigenvalue of a **reversible** `P` and `γ = 1 − λ₂`,
  "Φ⋆²/2 ≤ γ ≤ 2Φ⋆" (eq. 13.6), attributed there to Sinclair–Jerrum (1989) and
  Lawler–Sokal (1988). This confirms PRIOR_ART.md:691-700 verbatim and the two caveats
  filed there (ergodic chain, not the transient block; one cut bounds only the upper
  half) stand. Bibliographic correction: the AMS record lists **Levin and Peres** as
  authors, Wilmer as contributor — PRIOR_ART.md:693 and the brief write "Levin, Peres,
  Wilmer"; the bib follows the record. *Leaves open* the transient-block quantity the
  ladder actually truncates (`scale/foreman_lambda2.py` per PRIOR_ART.md:702-707) —
  *found by* seed + S5.
- **sinclair-1989-approximate**, **lawler-1988-bounds**, **cheeger-1970-lower** — [V]
  Crossref — *own* the chain Cheeger inequality (two proofs) and the manifold original.
- **kemeny-1976-finite** — [V] Google Books catalogue (year field [U]: catalogue says
  1983, booksellers 1976; first edition Van Nostrand 1960 attested by two Crossref review
  records) — *owns* the fundamental matrix `N = (I − Q)^{-1}` and the absorption
  probabilities `B = N R` of an absorbing chain — which is **identity I3 of the brief
  exactly** (`q = (I − Q)^{-1} R 1_B`). *Leaves open* learning `Q`, learning the absorbing
  set, and reading `B` as a head — *found by* seed. **Caution recorded in the bib:** the
  Springer DOI `10.1007/978-1-4684-9455-6` is *Denumerable Markov Chains* (Kemeny, Snell,
  Knapp), a different book; the first fetch of this sweep resolved to it and it was not
  attached.
- **doyle-1984-random** — [V] arXiv abs — *owns* the harmonic-function / escape-probability
  reading of the committor and the electrical analogy behind `scale/kirchhoff.py`.
- **kirchhoff-1847-auflosung**, **chaiken-1982-combinatorial** — [V] Crossref — *own* the
  matrix-tree theorem and its all-minors form (MATHEMATICS.md §11).
- **perron-1907-matrices** — [V]; **frobenius-1912-matrizen** — [U] NOT FETCHED; **meyer-2000-matrix**,
  **horn-2013-matrix** — [V] Crossref — *own* Perron–Frobenius and the textbook Neumann
  series with its geometric tail. **The brief's identity I4** (`‖(I−γP)^{-1} − Σ_{k≤K}(γP)^k‖_∞
  ≤ γ^{K+1}/(1−γ)` for row-stochastic `P`) is the textbook tail bound with `‖P‖_∞ = 1`
  substituted; it is DERIVED, owned by the Neumann-series literature, and widening search
  S4 found no paper stating it as an *attention-mask certificate* — that packaging is NOT
  FOUND, the mathematics is OCCUPIED.
- **banach-1922-operations** — [V]; **neumann-1877-potential** — [U] — *own* the fixed
  point and the series.
- **sherman-1950-adjustment**, **hager-1989-updating** — [V] Crossref — *own* the rank-one
  (and rank-k) update of an inverse. **This is the algebra of the brief's `do(a)`
  displacement**: if an intervention changes `P` (or `V`) in one row, `Δz = z(do a) − z` is
  a Sherman–Morrison correction of the *same* factorisation, at `O(s·d)` after the first
  solve — DERIVED, and the owner is 1950 — *found by* the sweep's own derivation, then
  fetched.
- **kolmogorov-1931-analytischen**, **chapman-1928-brownian**, **prinz-2011-markov** — [V]
  Crossref — *own* the Chapman–Kolmogorov equation and its use as a Markov-state-model
  validation test (the CK test in `ceq/beds/bed_1.py`, CEQ_V16_CONTRACT.md:177).
- **e-2006-transition**, **metzner-2009-transition** — [V] Crossref — *own* the committor
  as the solution of the discrete Dirichlet problem with two absorbing sets, plus reactive
  flux and dominant pathways — *leaves open* `K ≥ 2` constraint sets scored against
  candidate moves, and any learning of the chain from context — *found by* S2.
- **zhu-2003-harmonic** — [V] author's laboratory listing — *owns* the harmonic
  extension `f_u = (I − P_uu)^{-1} P_ul f_l` with labelled nodes clamped (absorbing) — as a
  **learning layer**. This is the committor read with boundary conditions, in machine
  learning, 2003. *Leaves open* the causal (triangular) `P`, the softmax parity, the use
  inside attention, and `K ≥ 2` sets scored per candidate move — *found by* S11.
- **katz-1953-status** — [V]; **brin-1998-anatomy** — [V]; **gasteiger-2019-appnp** — [V];
  **gasteiger-2019-diffusion** — [V]; **yuan-2025-paraformer** — [V] — *own*, in
  chronological order, the resolvent of an adjacency matrix as an index (1953), the
  resolvent of a row-stochastic matrix as PageRank / personalized PageRank (1998), the
  **exact** propagation `α(I − (1−α)Â)^{-1} H` and its truncated power iteration as a
  neural-network layer (ICLR 2019), the same as a diffusion kernel (NeurIPS 2019), and a
  generalized-PageRank *polynomial* inside a graph transformer's attention (WSDM 2026).
  *Leave open* causality (strict triangularity), the softmax logits as the source of `P`,
  the `γ = 0` bitwise parity, absorbing sets, and the LM setting — *found by* S9.
- **dayan-1993-successor**, **bellman-1957-markovian**, **altman-1999-constrained**,
  **wang-2024-incontext-td**, **xie-2026-softmax-rl**, **bai-2019-deq** — [V] — see the
  SURPRISE in §3. Bellman owns `v = (I − γP)^{-1} r` and the argmin over actions; Dayan
  owns `M = (I − γP)^{-1}` as a *representation*; Altman owns the decision problem with
  several constraint functionals; Wang et al. and Xie et al. own the statement that a
  (linear, then softmax) transformer's forward pass implements TD policy evaluation
  in-context; Bai et al. own the fixed-point-as-layer framing — *found by* S1, S3, S12.
- **villani-2009-optimal** — [V] Crossref; **kantorovich-1958-space** — [U] NOT FETCHED —
  *own* Kantorovich–Rubinstein duality for `W₁` (contract M13, CEQ_V20_R15_CONTRACT.md:235;
  the contract's [V] on Kantorovich–Rubinstein has no fetched primary behind it in this
  sweep — Villani's textbook statement is the citable [V]).

### 1.D Operator theory and combinatorial optimisation

- **pearcy-1966-power** — [V] Crossref (pages [U]) — *owns* the citable proof of Berger's
  power inequality `w(Aⁿ) ≤ w(A)ⁿ`; Berger's own proof is unpublished per S6 — *found by*
  seed + S6.
- **crouzeix-2017-numerical** — [V] Crossref — *owns* the `(1+√2)` spectral-set constant.
- **nemhauser-1978-submodular** — [V] Crossref — *owns* the `1 − 1/e` greedy guarantee
  (MATHEMATICS.md:1107; results/r9_maths_survey.md:376-409, which also records that the
  captured-mass objective is *supermodular* and the guarantee does not apply there).

### 1.E Dynamics, memory kernels, symbolic dynamics, transfer operators

- **hosking-1981-fractional** — [V] OUP landing; **granger-1980-long-memory** — [V];
  **grunwald-1867-derivationen**, **letnikov-1868-theory** — [U] NOT FETCHED — *own*
  fractional differencing and the GL weights (CEQ_V15_CONTRACT.md:109-110).
- **zwanzig-1961-memory**, **mori-1965-transport**, **zwanzig-2001-nonequilibrium** (Google
  Books catalogue, [V]), **chorin-2000-optimal** — [V] — *own* the projection-operator
  memory kernel the contract's M7 uses (CEQ_V20_R15_CONTRACT.md:203-206). *Leaves open*:
  M7's claim "K ≡ 0 on the chain" is a statement about the record's own construction,
  not a theorem in these sources — DERIVED there, and the paper should say so.
- **bollt-2000-validity**, **bollt-2001-misplaced** — [V] Crossref — *own* the misplaced-
  partition result on symbolic dynamics.
- **pesin-1977-characteristic** — [V] mathnet landing — *owns* the Pesin identity.
- **koopman-1931-hamiltonian**, **mezic-2005-spectral**, **brunton-2022-koopman** — [V] —
  *own* the Koopman/transfer-operator lineage (V20_R15_THEORY_TABLE.md:325 names it as the
  leap for Q3/W1).
- **kronig-1926-dispersion** — [V] (last page [U]); **kramers-1927-diffusion** — [U] NOT
  FETCHED — *own* the causality–analyticity relation shipped in `ceq/x35p/kk.py`.

### 1.F The signed / anti-concentration lineage (dead programme, carried as negatives)

- **littlewood-1943-number** — [U]; **erdos-1945-lemma** — [V]; **tikhomirov-2020-singularity**
  — [V] arXiv + Crossref (pages [U]); **brualdi-1995-sign** — [V] Cambridge landing;
  **zaslavsky-1982-signed** — [V] Crossref. Corrections: CONTRACT.md:177,185 carried
  Brualdi–Shader and Zaslavsky as [U]; both now title-matched.

### 1.G Transformer obstructions and intervention in attention

- **sanford-2024-logdepth** — [V] arXiv abs (theorem numbers per the brief, [U]).
- **peng-2024-limitations** — [V] arXiv abs — the brief flagged this id as "VERIFY"; the
  title is *On Limitations of the Transformer Architecture*, primary class stat.ML.
- **karbalayghareh-2026-doformer** — [V] bioRxiv landing — *owns* a do-operator *inside
  attention*: the intervened token's value is fixed and it is forbidden from attending.
  That is an absorbing row in an attention operator. *Leaves open* the resolvent, the
  displacement read, several constraint sets, the LM setting — *found by* S10.

---

## 2. Searches run

**Seeds** (the lineage list in the task; every name resolved or recorded [U] above).

**Widening searches beyond the seeds — thirteen, each with query and outcome:**

| id | query (verbatim) | outcome |
|---|---|---|
| S1 | `successor representation (I - gamma P)^{-1} attention transformer resolvent` | SR closed form `M = (I − γP)^{-1}` confirmed across RL sources; no transformer paper surfaced; Dayan 1993 fetched [V] |
| S2 | `committor function transition path theory absorbing Markov chain Dirichlet problem Metzner Schütte Vanden-Eijnden` | E–Vanden-Eijnden 2006 and Metzner et al. 2009 fetched [V] |
| S3 | `constrained Markov decision process minimize maximum probability of reaching unsafe set multiple constraints safety reachability` | CMDP and reach-avoid formulations; Altman 1999 fetched [V]; the reach-avoid LP papers (arXiv:1209.2883, 1507.01585) NOT FETCHED |
| S4 | `Neumann series truncation error bound row-stochastic matrix geometric tail gamma^(K+1)/(1-gamma)` | generic Neumann-series pages only; **NOT FOUND** as an attention certificate |
| S5 | `Levin Peres Wilmer "Markov Chains and Mixing Times" second edition Theorem 13.10 Cheeger conductance bottleneck ratio` | authors' PDF located and read; Theorem 13.10 confirmed |
| S6 | `Berger power inequality numerical radius w(A^n) ≤ w(A)^n Pearcy 1966 elementary proof` | Pearcy 1966 fetched [V]; Berger unpublished |
| S7 | `Barlow Bartholomew Bremner Brunk 1972 "Statistical Inference under Order Restrictions" Wiley ISBN pool adjacent violators` | review records for both isotonic textbooks; Ayer 1955 fetched as PAVA origin |
| S8 | `anytime-valid equivalence testing e-process TOST two one-sided tests sequential` | 2026 preprints on anytime-valid equivalence tests exist (arXiv:2606.00878, 2603.16213) — **NOT FETCHED**, so sequential TOST is NOT FOUND at [V] |
| S9 | `personalized PageRank resolvent (I - alpha P)^{-1} attention mechanism transformer "attention"` | ParaFormer (WSDM 2026) fetched [V]; APPNP and GDC fetched [V] |
| S10 | `interventional attention do-operator transformer causal intervention consequence displacement fixed point re-solve` | DoFormer fetched [V]; attention-head-intervention interpretability papers not fetched (not a mechanism) |
| S11 | `absorbing states boundary conditions attention mechanism transformer Dirichlet problem harmonic attention "absorbing"` | only attention *sinks* and "Transformer Meets Boundary Value Inverse Problems" (arXiv:2209.14977, NOT FETCHED); **NOT FOUND** for absorbing constraint sets inside attention. Zhu 2003 was then searched directly and fetched [V] |
| S12 | `transformer attention implements policy evaluation Bellman resolvent in-context temporal difference learning "(I - \gamma P)^{-1}"` | Wang et al. 2024 and Xie et al. 2026 fetched [V] |
| S13 | `Zwanzig "Nonequilibrium Statistical Mechanics" Oxford University Press 2001 ISBN 0195140184` | catalogue record then fetched [V] |

Plus four bibliographic-repair searches (Kemeny–Snell reprint; Cover–Thomas theorem
numbers ×2; Sander et al. id) and one for Kantorovich–Rubinstein (no primary page).

---

## 3. Verdict

Component by component against the shape spine in BRIEF.md §1.

**OCCUPIED (owner named).**

1. `z = (I − γP)^{-1} V` with row-stochastic `P`, `γ ∈ [0,1)` — owned four times over:
   Katz 1953 (resolvent index), Bellman 1957 (policy evaluation `v = (I − γP)^{-1} r`),
   Dayan 1993 (successor representation `M = (I − γP)^{-1}`, so `z = M V`), Brin–Page 1998
   / Gasteiger et al. 2019 (personalized PageRank as an **exact neural propagation layer**,
   `α(I − (1−α)Â)^{-1} H`). The learnable `γ` is APPNP's teleport `α` and Bellman's discount.
2. `O = P z` — one more propagation step on the same object; APPNP's "predict then
   propagate" is the same family with the read on the other side. Trivially occupied.
3. `q^{(k)} = (I − Q)^{-1} R_k 1` — Kemeny–Snell `B = N R` (1960); Doyle–Snell harmonic
   reading (1984); E–Vanden-Eijnden 2006 / Metzner–Schütte–Vanden-Eijnden 2009 (the
   committor as the Dirichlet problem); **Zhu–Ghahramani–Lafferty 2003** (the same solve
   with clamped labelled nodes, as a learning layer). Identity I3 of the brief is a
   restatement of Kemeny–Snell.
4. The Neumann certificate (I4) — the textbook geometric tail (Meyer; Banach's a-priori
   estimate) with `‖P‖_∞ = 1`; the planted non-stochastic negative is the record's own.
5. The triangular solve (I5) and corner-3-is-a-resolvent (I2) — nilpotent strictly
   triangular ⇒ finite Neumann sum; classical; the record's Lean owns the machine check.
6. `do(a)` displacement *as algebra* — Sherman–Morrison 1950 / Hager 1989: a rank-one
   change of `P` or `V` updates `(I − γP)^{-1}` without re-solving.
7. Safest move `argmin_a max_k q^{(k)}(do a)` *as a decision object* — Bellman 1957
   (argmin over actions on evaluated values); Altman 1999 (several constraint
   functionals on one chain, LP-solvable); the reach-avoid literature (S3, not fetched).
8. Cheeger stratification (contract M14) — Levin–Peres Theorem 13.10, confirmed verbatim;
   the record's two caveats stand.
9. Every statistical instrument (TOST, Clopper–Pearson, McNemar, Page's L, PAVA, Wilson,
   e-processes, Azuma/Hoeffding), every floor (Fano, Gaussian rate–distortion, Ziv–Zakai,
   Welch, Hankel/AAK–Glover, JL, Eckart–Young–Mirsky), and every dynamics instrument
   (Mori–Zwanzig kernel, CK test, Bollt partitions, Pesin, Koopman) — occupied; the paper
   uses them and claims nothing on them.

**NEAR-MISS (what is missing named).**

- *Resolvent inside attention*: ParaFormer 2026 puts a generalized-PageRank **polynomial**
  in a graph transformer's attention — truncated, undirected, not causal, not built from
  the LM's own softmax logits, no `γ = 0` bitwise parity, no certificate.
- *Intervention inside attention*: DoFormer 2026 fixes an intervened token and forbids it
  from attending (an absorbing row) — for gene perturbation, one intervention at a time,
  no resolvent, no displacement read, no constraint sets.
- *Sequential equivalence testing*: exists in 2026 preprints (S8), not fetched; the
  brief's "TOST N = 70" is fixed-N Schuirmann and is fine as such.
- *Safest move with candidate moves in the context*: CMDPs solve for a policy; the shape
  scores **candidate moves that are tokens in the context** against `K` committors — the
  in-context packaging of the CMDP object is not in any fetched source.

**NOT FOUND (queries recorded in §2).**

- Learnable absorbing constraint sets `𝒜_k` inside a causal attention operator, with the
  committor into each set read as a head output (S11).
- The Neumann tail `γ^{K+1}/(1−γ)` printed as a per-mask certificate of an attention
  truncation, with a planted non-stochastic negative (S4).
- The displacement `Δz` of a resolvent fixed point under an in-context intervention,
  read as "the consequence", as an attention output (S10).
- The `γ = 0` bitwise-parity clause making a resolvent layer a strict superset of softmax
  by construction (S1, S9, S12 — none of the resolvent-layer owners has it, because none
  builds `P` from softmax logits over a causal prefix).

**SURPRISE — flagged loudly.** Three fetched sources, taken together, could occupy the
whole delta unless the delta is stated narrowly:

- **Bellman 1957 + Dayan 1993 + Gasteiger et al. 2019.** The shape's `z` is *policy
  evaluation of `V` under the chain `P` with discount `γ`* — the successor representation
  applied to `V` — and APPNP already ships the exact resolvent as a neural layer. The
  paper cannot present `(I − γP)^{-1} V` as its object; it must present *which `P`*
  (causal, row-stochastic, the softmax corner of the record's three-corner family with
  `γ = 0` parity bitwise, Lean-carried), *which read* (one triangular solve, certificate
  printed), and *which boundaries*.
- **Zhu–Ghahramani–Lafferty 2003.** The committor-with-absorbing-boundary as a learning
  layer is twenty-three years old. "A third token can veto through a boundary condition"
  is harmonic label propagation with the labelled node as the veto. The narrow delta is
  `K ≥ 2` learnable sets inside a *causal* operator, scored per candidate move.
- **Wang et al. 2024 and Xie et al. 2026.** A linear-attention transformer, and then a
  softmax transformer, implement TD policy evaluation in the forward pass, layer by
  layer. This is the "deeper softmax skyline" of BRIEF.md §1 made precise: softmax at
  depth computes the *same resolvent by iteration*. The paper's claim against softmax must
  therefore be stated at matched depth and parameters exactly as the brief demands, and
  the four obstructions must be read as *one-layer* obstructions (Peng et al. 2024 is a
  one-layer result; Sanford et al. 2024 is a depth law) — never as "softmax cannot
  compute this".

**What survives as the delta, narrowly:** (a) `P` is the record's causal softmax corner
and the layer is bitwise softmax at `γ = 0` (I1, Lean `three_corners_containment`);
(b) the resolvent is *exact* in one triangular solve with a printed Neumann δ and a
planted negative; (c) `K ≥ 2` absorbing constraint sets live inside the operator and the
committors are head outputs; (d) `do(a)` is a Sherman–Morrison displacement of the same
factorisation and the displacement is the consequence label; (e) the bed / floor /
control / counter-prediction discipline of the record. Items (a)–(d) are NEAR-MISS-to-
NOT-FOUND in this lineage; none is "novel"; each is a *composition* of owned parts, and
the paper should say exactly that.

---

## 4. Corrections and upgrades to the record's own citations

| record locator | as written | this sweep |
|---|---|---|
| METHODS.md:344 | Wang & Ramdas 2304.01163 `[U]` | title matched at arXiv abs: *The extended Ville's inequality for nonintegrable nonnegative supermartingales* → [V] |
| PRIOR_ART.md:519, METHODS.md:346 | Robbins–Siegmund 1971 ORIGINAL NOT REACHED | DOI registry record resolves and pairs with the title; full text still unread |
| CEQ_V20_R15_CONTRACT.md:258 | AAK/Glover `[U → V-eq at it.6]` | both title-matched at the registry → [V] |
| CONTRACT.md:177,185 | Brualdi–Shader `[U]`, Zaslavsky `[U]` | Cambridge landing page and registry record → [V] |
| CEQ_V20_R15_CONTRACT.md:213-214 | Cantelli `[V]`, Boole/Bonferroni `[V]` | no fetchable primary found; recorded [U] NOT FETCHED; cite a textbook statement instead |
| CEQ_V20_R15_CONTRACT.md:235 | Kantorovich–Rubinstein `[V]` | 1958 original not reachable; Villani 2009 is the [V] carrier |
| PRIOR_ART.md:693-700 | Levin, Peres, Wilmer 2nd ed. Thm 13.10 | confirmed verbatim, p. 183 of the authors' PDF; AMS record lists authors Levin and Peres, Wilmer as contributor |
| MATHEMATICS.md §12 | PAVA under "isotonic regression" | origin is Ayer et al. 1955 [V]; textbooks Barlow 1972, Robertson–Wright–Dykstra 1988 |
| results/r9_maths_survey.md:269 | Sander et al. ICML 2023, no id | arXiv:2302.01425 [V] |
| BRIEF.md §1 item 3 | Peng–Narayanan–Papadimitriou 2402.08164 "VERIFY" | verified: *On Limitations of the Transformer Architecture*, stat.ML |
| BRIEF.md §1 item 2 | Sanford–Hsu–Telgarsky 2402.09268 | title matched; theorem numbers not read |
| CEQ_V20_R15_CONTRACT.md:203-206 (M7) | "K ≡ 0 on the chain" attributed to Mori–Zwanzig | the sources own the projection formalism; the vanishing kernel on the record's chain is the record's own DERIVED claim |

---

## 5. What each proposal in this sweep is designed against (MISTAKES.md)

- **Owners cited before components are named** (§3, every OCCUPIED line) — against **P-7**
  (vocabulary with no referent) and BRIEF.md §5 rule 7.
- **Theorem 13.10 quoted from the theorem statement in the body, not from the record's
  paraphrase** — against **P-10** (a source's intro cited as its theorem).
- **Registry-record route with the route named on every entry, and [U] on every field
  not read** — against **P-1**'s citation analogue (a claim with no live producer) and
  **P-6** (line-reference drift: each locator carries the file:line and the session date).
- **Theorem numbers held at [U] until read (Cover–Thomas 2.10.1 / 10.3.2; Durrett
  4.2.12; Ville Thm 1 p. 84)** — against **P-3** (a stale claim never retracted) and
  **P-11** (a contract citing its own tagged theorem as settled).
- **The SURPRISE on Bellman/Dayan/APPNP** — against **D-1** (racing a baseline at its
  proven optimum): the shape's operator *is* policy evaluation, whose proven estimators
  are TD/LSTD; any arena that lets a bed collapse to scalar policy evaluation races LSTD
  on its own ground and repeats D-1. The paper's beds must ask for the vector `z`, the
  committor vector, or the argmin — never a scalar value at one state.
- **The SURPRISE on Wang 2024 / Xie 2026** — against **D-2** (an oracle equal to the arm's
  own resolvent) and the brief's R-SKY: the deeper-softmax skyline computes the same
  resolvent by iteration, so "beats softmax" is banned exactly where the brief bans it,
  and the oracle must run on the latent environment chain, never on the arm's `P`.
- **The Zhu 2003 occupancy** — against **V-12** (a single absorbing target makes the label
  constant) and **V-25** (a hypothesis no draw satisfies): BED-S needs `K ≥ 2` constraint
  sets and a domain census showing both are reachable from the query in the corpus.
- **The Neumann certificate as OCCUPIED-mathematics / NOT-FOUND-packaging** — against
  **V-24** (an identity bind whose rejection region is empty) and **V-15** (a condemning
  rule with no planted negative): the record's planted non-stochastic `P` (err 119.37 vs
  bound 1.143, BRIEF.md I4, RUN by the coordinator) is the rejection region and must ship
  with every printed δ.
- **The Cheeger caveat carried forward** — against **V-17** (a threshold imported out of
  its units): the ergodic-chain `γ` of Theorem 13.10 is not the transient-block Perron
  root the ladder truncates.
- **Sequential TOST recorded NOT FOUND at [V]** — against **M-13** (an equivalence margin
  registered without a reachability check) and **M-2** (a threshold refitted): the paper
  keeps fixed-N Schuirmann with the margin fixed before data.
- **Cantelli/Bonferroni/Kantorovich–Rubinstein downgraded to [U]** — against **P-11** and
  the brief's one fatal defect (a citation with no fetched identifier).

---

## 6. Open items

- NOT MEASURED — needs new code: none required by this sweep; all numbers above are READ
  or RUN by the coordinator.
- NOT FETCHED (bibliographic only, listed [U] in the bib): Cantelli 1928; Bonferroni 1936;
  Frobenius 1912; Neumann 1877; Kantorovich–Rubinstein 1958; Grünwald 1867; Letnikov 1868;
  Kramers 1927; Littlewood–Offord 1943.
- Theorem locators still [U]: Cover–Thomas Theorems 2.10.1 and 10.3.2; Durrett Theorem
  4.2.12 and Exercise 4.8.2; Ville Théorème 1 p. 84; Wang–Ramdas §2.1 eq. (3); Villani
  Theorem 5.10; Meyer §7.10; Kemeny–Snell Chapter III; Sanford et al. Thm 4.2 / Cor. 4.3.
- Not fetched but bearing on NEAR-MISS lines: the 2026 anytime-valid equivalence-testing
  preprints (S8); the reach-avoid LP papers (S3); "Transformer Meets Boundary Value
  Inverse Problems" (S11). A second pass should fetch these three before the paper's §7
  table is frozen.
