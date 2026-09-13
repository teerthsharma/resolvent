# PRIOR-ART SWEEP — causality, interventions, consequence learning in attention models

**Planet:** MARS-S4. **Date:** 2026-09-03. **Repository:** `<repo root>`,
branch `v17k-gate0`, HEAD `207e7b9`. **Companion file:** `bib_causality.bib` (same directory,
59 entries: 58 `[V]`, 1 `[U]`).

**Evidence classes used below** (brief §5.3): `RUN` = executed this session on this box;
`READ path:line` = quoted from the repository; `CITED [V]` = the arXiv abs page, the DOI /
CrossRef record, the publisher page or an ISBN-registry record was fetched this session and the
title matched; `CITED [U]` = reached through a search-index snippet or memory only; `DERIVED` =
steps written out. A content claim about a `[V]` source that came only from a search snippet is
marked `[U]` inline, even when the source itself is `[V]`.

**The question put to this planet.** Does anyone define *the consequence of an intervention* as
*the displacement of a jointly determined fixed point of an attention operator*, supervise it with
*exact `do()` oracles*, and score *sign / vector fidelity against a per-row control*?

**The short answer.** Each clause has an owner; the conjunction does not. The equilibrium-displacement
definition of consequence is owned, in structural-causal-model and dynamical-systems terms, by the
Mooij–Janzing–Schölkopf / Bongers–Forré–Peters–Mooij / Dash lineage (§2.B). The "fixed point of a
causally masked transformer, with `do()` by clamping and re-solving" is owned — this is the
**SURPRISE** — by Scetbon, Jennings, Hilmkil, Zhang and Ma 2024, *A Fixed-Point Approach for Causal
Generative Modeling* (§2.C, §3.4). Supervising a transformer with exact interventional targets from
synthetic SCMs is owned by the Do-PFN / CausalFM / CausalTimePrior line (§2.D). Auditing a model's
*imagined* intervention effect against an environment oracle with a no-change floor is owned, as of
four days before this sweep, by Vakalis 2026 (§2.F). What is **NOT FOUND** is the resolvent read
`P(I − γP)^{-1}V` of a *context* (not a data-generating SCM) as the attention output, with absorbing
constraint rows, the displacement `Δz` as a *vector* label scored jointly against a matched-depth
per-row softmax control, and the committor as the safety score inside the same read (§3.3).

---

## 0. The shape's components, as this lineage reads them

From the brief §1 (READ `BRIEF.md:59-71`), the components whose ownership this planet must settle:

| # | component | where it lives in the shape |
|---|---|---|
| C1 | `P` causal row-stochastic operator (three-corner family; `β = 1` is softmax) | `ceq/arm_smprime.py`, `lean/CEQ/V16Domain.lean` |
| C2 | `z = (I − γP)^{-1} V` the jointly determined state; `O = Pz` the read | identity I1, I2, I5 |
| C3 | absorbing sets `𝒜_k`; committor `q^{(k)} = (I − Q)^{-1} R_k 1` | identity I3; `ceq/beds/bed_1.py:188-198` (READ: solves `(Lq)_i = 0` interior, `q = 0` on A, `q = 1` on B) |
| C4 | `do(a)`: intervene on the context, re-solve; `Δz = z(do a) − z` is the consequence | the interventional channel, `CEQ_V16_CONTRACT.md:179` (READ: "per-position `do()` bumps with oracle responses; gates are Jacobians") |
| C5 | supervision by *exact* `do()` oracles on the latent environment chain (never the arm's own resolvent — D-2) | `CEQ_V15_CONTRACT.md:188-190` (READ: "interventions identify `a` directly; observations confound `a` with `b`") |
| C6 | sign / vector fidelity against a per-row control (softmax at matched depth) | `MATHEMATICS.md:384-392` (READ: `fidelity = fraction of interventions where sign(Δŷ_model) = sign(Δy_oracle)`, Clopper–Pearson, McNemar vs softmax on byte-identical draws) |
| C7 | safest move `argmin_a max_k q^{(k)}(do a)` | brief §1 |
| C8 | the `do()`-conditioned resolvent `p(· \| do(a_k))` (THEORY v1) | `THEORY.md:61-62` (READ: "the action-conditioned `T_k` blocks make it `p(· \| do(a_k))` rather than `p(· \| a_k)` — provided the training data contains real interventions") |

One record fact frames C6 and must travel with any fidelity number: at `s = 64, d = 24`, seed 0,
softmax already reads consequence fidelity `0.807843` with CP `[0.754044, 0.854329]`, and the
signed arm's `0.854902` does not survive McNemar (`p = 0.18181`) (READ `MATHEMATICS.md:399-412`).
Sign fidelity at one position is therefore *not* the ground softmax cannot occupy; only the joint
vector and the per-row-vs-joint contrast are (READ `MATHEMATICS.md:106-127`, §0.3, "currently
UNTESTED, because no vector-valued label exists in the repository").

---

## 1. Search log

Seeds fetched directly (S1–S13) are listed with the entries. WebSearch queries run this session,
in order, with the hits that were then fetched and verified:

| q | query (verbatim) | verified hits it produced |
|---|---|---|
| Q1 | `interventional world model transformer consequences of actions` | vakalis-2026-interventiongap, lin-2026-scratchworld, chen-2026-worldmodelroadmap |
| Q2 | `"causal attention" misnomer masking versus causality transformer` | karagodin-2024-clustering (via Q8) |
| Q3 | `transformer in-context learning interventional queries structural causal model do-operator supervised oracle` | robertson-2025-dopfn, wang-2026-inherently, anon-2025-iclcausalbma `[U]` |
| Q4 | `Basseville Nikiforov 1993 "Detection of Abrupt Changes" pdf` | basseville-1993-detection (author page + PDF TOC) |
| Q5 | `robot learn consequences of actions attention transformer intervention` | nothing on-lineage (robot-attention patents, decision transformers); CausalWorld reached from memory and verified |
| Q6 | `counterfactual transformer attention intervention fixed point equilibrium displacement` | saha-2025-cyclic, bottou-2013-counterfactual, scetbon-2024-fip, miller-2025-counterfactualicl |
| Q7 | `Dash 2005 "Restructuring dynamic causal systems in equilibrium" AISTATS equilibration manipulation` | dash-2005-emc, voortman-2010-manipulation |
| Q8 | `Karagodin Polyanskiy "clustering in causal attention masking" arXiv` | karagodin-2024-clustering |
| Q9 | `deep equilibrium model intervention causal effect fixed point displacement implicit layer do-operator` | **nothing** on interventions in DEQs (DEQ tutorials, monotone operator nets only) |
| Q10 | `sign agreement metric direction of intervention effect prediction evaluation causal model "sign accuracy"` | xun-2026-evidencetype, hoogland-2022-ite, chen-2024-calm |
| Q11 | `absorbing Markov chain committor attention constraint satisfaction safest action transformer` | makkuva-2024-attentionmarkov; **no** committor-in-attention hit |
| Q12 | `model-based reinforcement learning "consequence prediction" action-conditioned next state transformer causal intervention 2025` | nothing new verified (surveys, Decision Transformer) |
| Q13 | `successor representation transition revaluation fails Russek Momennejad 2017 predictive representations model-based` | momennejad-2017-sr, russek-2017-predictive (both via CrossRef) |
| Q14 | `linear structural equation model intervention closed form "(I-B)^{-1}" do-operator total causal effect path coefficients` | gische-2022-beyondmean (CrossRef) |
| Q15 | `reach-avoid probability safest action under multiple constraints transformer world model planning 2025 2026` | nakamura-2025-latentreach, latyshev-2025-safeplanning |
| Q16 | `transformers in-context learning linear structural causal model intervention prediction interventional distribution synthetic SCM 2024 2025 arXiv` | ma-2025-causalfm, thumm-2026-causaltimeprior |
| Q17 | `"How Transformers Learn Causal Structures In-Context" explainable mechanism theoretical guarantee arXiv` | dangelo-2025-selective; the OpenReview paper itself stayed `[U]` |
| Q18 | `"Learning why things change" Voortman Dash Druzdzel difference-based causality learner equilibrium manipulation` | weinberger-2023-equilibrium (CrossRef) |

Eighteen searches; twelve beyond the seed list. Landing-page fetches that failed and were routed
around: Cambridge (500 → DOI redirect → product page OK), MIT Press (403 ×2 → Open Library ISBN
record), OAPEN (ECONNRESET ×3), Google Books (429 ×2), DOAB (403), Elsevier linkinghub (login →
CrossRef), Springer (cookie redirect → CrossRef), OpenReview (bot wall ×2), Semantic Scholar (429),
dblp (0 hits), `www-lm2s.utt.fr` (ENOTFOUND). Three PDFs that WebFetch could not text-extract
(Basseville–Nikiforov book, Bottou 2013, Dash 2005) were read with a `python -X utf8 -c` one-liner
on the locally saved copies — a read, no file written (brief §5.1).

---

## 2. Sources, one block each

Format per source: **id** · class · *owns* (the equation or mechanism, in this planet's words) ·
*leaves open* (relative to C1–C8) · *found by*.

### 2.A SCM foundations and the linear closed form

**pearl-2009-causality** · `[V]` (Cambridge product page via DOI `10.1017/CBO9780511803161`;
2nd ed., 2009, ISBN 9780521895606) · *owns:* the `do()` operator as graph surgery — replace the
structural equation of the intervened variable by a constant, keep every other mechanism; the
three rungs (association, intervention, counterfactual); do-calculus. · *leaves open:* nothing
about attention, fixed points of an operator read, or a metric; the linear-SCM resolvent is present
as path analysis, not as an attention read. · S1.

**peters-2017-elements** · `[V]` (Open Library ISBN record 9780262037310; MIT Press page 403) ·
*owns:* the textbook form of SCMs, the principle of Independent Causal Mechanisms (ICM: mechanisms
do not inform or influence each other, so an intervention on one leaves the rest invariant), and
the statement that interventional data identify what observational data confound. · *leaves open:*
C1–C8 entirely; the ICM principle is the *licence* for the shape's claim that `do(a)` alters one
row of `P` and the rest of `P` is invariant — the paper must cite it as that licence and no more. ·
S2, S11.

**shimizu-2006-lingam** · `[V]` (JMLR 7:2003–2030) · *owns:* the linear SCM written as
`x = Bx + e`, solved as `x = (I − B)^{-1} e` — the identical algebraic object to `z = (I − γP)^{-1}V`
with `B ↔ γP`, `e ↔ V`; identification of `B` from non-Gaussian `e` by ICA. · *leaves open:* `B` is
the data-generating graph, not a context-dependent attention operator; no `γ`, no row-stochastic
constraint, no absorbing rows, no `do()` displacement as a supervised label. · S1 lineage, from
memory, verified.

**gische-2022-beyondmean** · `[V]` (CrossRef, Psychometrika 87(3):868–901) · *owns* `[U]` (search
snippet only): the total effect of `X_i` on `X_j` under `do(X_i = x_i)` as the entry
`(I − B)^{-1}_{ji}`, and interventional distributions in closed matrix form. · *leaves open:* same as
LiNGAM; the record's own `results/r9_maths_survey.md:313` (READ: "`X = (I−B)^{-1}Z`; `do(X_i=a)`
clamps row `i`. `ACE_i = B_do(1)[i] − B_do(0)[i]`") already states this and reuses
`e4_harmonic.fixed_point` — the record has this component and knew it was occupied. · Q14.

### 2.B Equilibrium / dynamical SCMs — consequence = displacement of an equilibrium

**mooij-2013-ode2scm** · `[V]` (arXiv:1304.7920, stat.OT, UAI 2013) · *owns:* the equilibria of a
first-order ODE system define a (possibly cyclic) deterministic SCM, and an intervention on the
ODE (clamp a variable) corresponds to graph surgery on that SCM; consequence of intervention =
the new equilibrium. **This is the earliest clean statement of "consequence = displacement of a
fixed point" in the causal literature.** · *leaves open:* no operator read of a context, no
attention, no learning, no supervision protocol, no metric; equilibria of ODEs, not of a resolvent
on a sequence. · Q6 lineage, from memory, verified.

**bongers-2021-cyclic** · `[V]` (arXiv:1611.06221, stat.ME; Ann. Statist. 49(5):2885–2915) ·
*owns:* solvability and uniqueness for cyclic SCMs ("simple SCMs"); interventions and
counterfactuals remain well defined when the structural equations are a fixed-point system rather
than a DAG. · *leaves open:* the shape's `P` is row-stochastic with `γ < 1`, which puts it inside
the "unique solution" class by construction (I4) — the paper should cite Bongers for the
*existence/uniqueness vocabulary* and prove the shape's instance itself (spectral radius `γ`). ·
Q6 lineage, verified.

**blom-2019-ccm** · `[V]` (arXiv:1805.06539, cs.AI, UAI 2019) · *owns:* the claim that SCMs cannot
represent every dynamical system in equilibrium; Causal Constraints Models generalise them. ·
*leaves open:* a warning, not a component: an equilibrium with absorbing constraints (C3) is a
constrained equilibrium; whether the shape's boundary-condition semantics survive as an SCM or need
a constraints-model reading is a question the paper should state, not assume. · Q18 lineage.

**dash-2005-emc** · `[V]` (PMLR R5:81–88; PDF read this session: Definition 4, Theorems 1–2) ·
*owns:* **the Equilibration–Manipulation Commutability (EMC) question** — whether "equilibrate,
then `do()`" equals "`do()`, then equilibrate"; Theorem 1 gives conditions under which EMC is
*violated* (feedback through the manipulated variable), Theorem 2 conditions under which it holds.
· *leaves open:* the shape's C4 defines consequence as "intervene on the context, re-solve" —
that is `do()`-then-equilibrate. A reader of Dash will ask whether the shape's `Δz` is the same
object as "equilibrate-then-`do()`" on the underlying environment chain. For a *strictly causal*
(triangular) `P` there is no feedback and EMC holds trivially (DERIVED below); for a `P` with
absorbing rows the two orders differ exactly when the intervened position feeds an absorbing set.
**This is a bind the shape owes and did not have.** · Q7.

**voortman-2010-manipulation** · `[V]` (PMLR 6:257–266) · *owns:* the first structure-learning
algorithm shown to predict manipulation effects correctly in systems that *violate* EMC. · *leaves
open:* time-series structure learning, not attention. · Q7.

**iwasaki-1994-abstraction** · `[V]` (CrossRef, AIJ 67(1):143–194; content not read) · *owns:*
the equilibration operator on a dynamic causal ordering (the source Dash builds on). · *leaves
open:* cited as lineage only. · Q7 lineage.

**weinberger-2023-equilibrium** · `[V]` (CrossRef, Erkenntnis 88(6):2467–2491; content not read) ·
*owns:* the philosophical adequacy of equilibrium causal models under intervention. · *leaves
open:* cited as lineage only. · Q18.

**saha-2025-cyclic** · `[V]` (arXiv:2510.25005, cs.AI, NeurIPS 2025) · *owns:* counterfactuals
in cyclic SCMs under shift–scale (soft) interventions, using the steady-state SCM of an ODE. ·
*leaves open:* the shape's `do(a)` on a context is a *hard* intervention on one position; the
soft-intervention algebra here is the natural generalisation if the paper ever wants "nudge a
token" rather than "replace a token" — record as future, not as a component. · Q6.

**bottou-2013-counterfactual** · `[V]` (JMLR 14:3207–3260; arXiv:1209.2355; PDF read: §7
"Equilibrium Analysis", §7.3 "Estimating the equilibrium response") · *owns:* estimating how a
small change `dθ` displaces the equilibrium of a feedback system by differentiating the first-order
equilibrium conditions, which yields a linear system linking the displacements — the first-order
(implicit-function) form of the shape's C4. · *leaves open:* first-order in `dθ`, a quasistatic
economic equilibrium (advertising auctions), no attention, no exact re-solve; but the *mechanism*
"differentiate the equilibrium equations" is precisely the Jacobian gate the record's contract
already prescribes (READ `CEQ_V16_CONTRACT.md:180`, "gates are Jacobians `[RUN 1e-9]`"). · Q6.

**DERIVED — the exact displacement identity, and why EMC is trivial for triangular `P`.**
Let `z = (I − γP)^{-1}V`. An intervention changes the operator and the value at the intervened
position: `P' = P + ΔP`, `V' = V + ΔV`, `z' = (I − γP')^{-1}V'`. Then

```
(I − γP') z' = V'                           (definition of z')
(I − γP') z  = V − γ ΔP z                    (expand (I − γP − γΔP) z, use (I − γP) z = V)
subtract:  (I − γP')(z' − z) = ΔV + γ ΔP z
so         Δz = (I − γP')^{-1} (ΔV + γ ΔP z)          … exact, no first-order step
```

Three consequences. (i) The consequence is *one more resolvent read* of a perturbation that is
supported on the intervened row(s) only — the same triangular solve as I5 and the same Neumann
certificate as I4 with the same `γ`. (ii) Bottou §7.3 is the linearisation `Δz ≈ (I − γP)^{-1}(…)`
of this identity; the shape can carry the exact form because `P'` is known. (iii) For strictly
causal `P` (triangular), position `i`'s row of `P'` depends only on positions `< i`, so
"`do()` then solve" and "solve, then `do()` and re-solve the suffix" produce the same `z'` —
EMC holds by nilpotency (`lean/CEQ/Nilpotent.lean`, `Occupancy.lean`, per brief §2); the two orders
differ only when `P` has feedback, i.e. when the shape is used with a non-triangular `P` or when
absorbing rows are re-targeted. That is the sentence the paper should write next to Dash 2005.
(Steps: 4 lines of algebra; no number claimed.)

### 2.C Fixed points and attention

**scetbon-2024-fip** · `[V]` (arXiv:2404.06969, cs.LG; abs + HTML read: Def. 2.3, §2.4, Def. 4.1,
Alg. 2) · *owns:* **an SCM written as a fixed-point problem `X = Pᵀ H(PX, PN)` on causally ordered
variables**, with `H` lower-triangular in its first argument (Condition 2.2); **a transformer whose
attention is masked to the causal order** (Def. 4.1, `M_{ij} = 0` if `i < j`, `+∞` otherwise —
i.e. the ordinary triangular mask); **interventions `do([PX]_i = a)` by composing `H` with a
clamping map `T_{i,a}` and re-solving** (§2.4); counterfactuals by abduction of the noise and
re-solve (Theorem 2.14); the fixed point is reached by **composing the map `d` times** (Alg. 2,
`d` = number of variables) — which is the nilpotent path-sum of the record's corner 3 in
different clothes. · *leaves open, exactly:* (a) the fixed point is the *data-generating* SCM over
`d` variables, not a read of an `s`-token context — the transformer is the mechanism `H`, not a
context operator; (b) no `γ`, no row-stochastic `P`, no resolvent `(I − γP)^{-1}` with `γ < 1` and
a printed truncation certificate — their solve is exact only because acyclic; (c) no absorbing
sets, no committor, no safest move (the HTML read confirms none of these words occur); (d) no
metric against a per-row control, no matched-depth softmax skyline; (e) no exact `do()` oracle
external to the model — their interventional distribution *is* the model's own re-solved fixed
point, which if imitated is the D-2 hazard (an oracle that is the arm's own resolvent). · Q6.
**Flagged loudly in §3.4.**

**bai-2019-deq** · `[V]` (arXiv:1909.01377, NeurIPS 2019) · *owns:* a layer defined as the fixed
point of a map, solved by root-finding, differentiated implicitly. · *leaves open:* no
intervention on the fixed point; Q9 found **nothing** on `do()` in DEQs. · S-memory, verified.

**ramsauer-2020-hopfield** · `[V]` (arXiv:2008.02217, cs.NE) · *owns:* the attention update as one
step of a modern-Hopfield energy descent whose fixed points are the retrieved patterns. · *leaves
open:* the fixed point is in pattern space per query (still a per-row object); no joint
determination across rows, no intervention. · S-memory, verified.

**dayan-1993-sr** · `[V]` (CrossRef, Neural Computation 5(4):613–624) · *owns:* the successor
representation `M = (I − γP)^{-1}` — the shape's C2 as an object, under a fixed policy chain. ·
*leaves open:* a cached quantity, not a context read; the record's `THEORY.md:61` already reads
the resolvent as "discounted expected future occupancy" and this is its owner. · Q13.

**momennejad-2017-sr** · `[V]` (CrossRef, Nat. Hum. Behav. 1(9):680–692) and
**russek-2017-predictive** · `[V]` (CrossRef, PLoS Comput. Biol. 13(9)) · *own* `[U]` (search
snippet for the mechanism): a cached SR cannot adapt to a change in the *transition* structure
without re-experiencing trajectories — humans show exactly that insensitivity — while a change in
*reward* is handled immediately. In the shape's terms: an intervention on `V` needs no re-solve;
an intervention on `P` (a `do()` that rewires who attends to whom) does. **This is the strongest
external argument that "consequence" requires a re-solve of `(I − γP')^{-1}` and not a re-read of
a cached mixture** — and it is the argument against a per-row softmax control that has memorised
`(I − γP)^{-1}` implicitly. · Q13.

### 2.D Transformers for causal inference; in-context interventional queries

**melnychuk-2022-causaltransformer** · `[V]` (arXiv:2204.07258, ICML 2022 PMLR 162) · *owns:*
counterfactual (potential-outcome) prediction over time with three transformer sub-networks
(covariates, treatments, outcomes) joined by cross-attention and a balancing adversarial term. ·
*leaves open:* potential outcomes under treatment sequences on longitudinal data, not a fixed
point; no `do()` oracle (observational data, unconfoundedness assumed); no per-row control. · S3.

**zhang-2023-cina** · `[V]` (arXiv:2310.00809) · *owns:* the duality between optimal covariate
balancing and self-attention, so a self-supervised transformer estimates treatment effects
zero-shot. · *leaves open:* per-query balancing weights are a *per-row mixture* — CInA is
evidence that softmax's own ground includes single-effect estimation, which strengthens the brief's
D-1 warning: single-position causal-effect labels are softmax's ground. · S4.

**robertson-2025-dopfn** · `[V]` (arXiv:2506.06039, NeurIPS 2025) · *owns:* **a prior-data-fitted
network trained on synthetic SCMs with paired observational and interventional data, predicting
interventional outcomes in-context from observational data alone.** This occupies C5 ("supervise
with exact `do()` oracles") for scalar per-query effects. · *leaves open:* the label is a scalar
outcome at one unit; no fixed point, no vector `Δz`, no absorbing sets, no matched-depth softmax
control (the PFN *is* a transformer, so the control question does not arise for them). · Q3.

**ma-2025-causalfm** · `[V]` (arXiv:2506.10914) · *owns:* a family of SCM-based priors so PFNs do
back-door / front-door / IV adjustment in context. · *leaves open:* as Do-PFN. · Q16.

**thumm-2026-causaltimeprior** · `[V]` (arXiv:2603.11090, ICLR 2026 TSALM workshop) · *owns:*
synthetic *temporal* SCMs with paired observational/interventional series, ground-truth
interventional targets by structural manipulation. · *leaves open:* time-series effects; no
fixed-point read, no committor. · Q16.

**ke-2022-induce** · `[V]` (arXiv:2204.04875) and **lorch-2022-avici** · `[V]` (arXiv:2205.12934,
NeurIPS 2022) · *own:* amortised causal *structure* inference from observational + interventional
data with a neural net (transformer-class). · *leave open:* the output is a graph, not a
consequence; relevant only as the licence that interventional data are the identifying signal. ·
memory, verified.

**xia-2021-ncm** · `[V]` (arXiv:2107.00793) · *owns:* the Neural Causal Model class and the
theorem that the causal hierarchy still holds for neural models — observational training alone
cannot yield interventional answers. · *leaves open:* the theorem is the formal backing for C5's
"exact `do()` oracles in training"; it does not touch attention or fixed points. · memory,
verified.

**zecevic-2021-gnnscm** · `[V]` (arXiv:2109.04173) · *owns:* message passing on the causal graph
as an SCM realisation, with interventions as edge removal. · *leaves open:* GNN not attention;
one message-passing step per hop (the depth law applies to them too). · memory, verified.

**scholkopf-2021-crl** · `[V]` (arXiv:2102.11107, Proc. IEEE) · *owns:* the ICM principle and
the sparse-mechanism-shift hypothesis. · *leaves open:* licence only. · S11.

**jin-2023-cladder** · `[V]` (arXiv:2312.04350, NeurIPS 2023) and **chen-2024-calm** · `[V]`
(arXiv:2405.00622) · *own:* benchmarks of causal reasoning with ground truth from causal models,
and a metric taxonomy ("intervention accuracy", "effect estimate accuracy"). · *leave open:*
natural-language questions to LLMs; no operator, no control, no vector label. · Q10, memory.

**miller-2025-counterfactualicl** · `[V]` (arXiv:2506.05188, NeurIPS 2025) · *owns:*
counterfactual reasoning as noise abduction in an in-context linear-regression task, with
attention heads identified as the mechanism. · *leaves open:* per-example scalar counterfactuals;
no fixed point. · Q6.

### 2.E "Causal attention" — the two meanings (the misnomer, settled by citation)

**karagodin-2024-clustering** · `[V]` (arXiv:2411.04990, NeurIPS 2024) · *owns:* the dynamics of
*causally masked* attention as an interacting particle system; "causal" = the triangular mask. ·
**nichani-2024-causalstructure** · `[V]` (arXiv:2402.14735, ICML 2024) and
**dangelo-2025-selective** · `[V]` (arXiv:2509.08184) · *own:* "causal structure" = which
earlier token a later token depends on in a latent Markov process, learned as an attention
pattern; no `do()`. · **wang-2026-inherently** · `[V]` (arXiv:2601.05647) · *owns:* extraction of
time-lagged dependency structure from gradients of an autoregressive transformer; "causal" =
Granger/dynamical. · **makkuva-2024-attentionmarkov** · `[V]` (arXiv:2402.04161, ICLR 2025) ·
*owns:* Markov chains as the controlled data process for analysing transformers; the chain is the
*data*, not the attention operator. · **yang-2021-causalattention** · `[V]` (arXiv:2103.03493) ·
*owns:* "causal attention" in the Pearl sense — front-door adjustment inside an attention module. ·
**anon-2025-iclcausalbma** · `[U]` (OpenReview bpF8zgSt41; authors not retrievable this session).

*What this settles for the paper:* the phrase "causal attention" has at least three published
meanings — the mask (Karagodin), the learned dependency graph (Nichani, D'Angelo, Wang), and
front-door adjustment (Yang). The shape's `P` is "causal" only in the first sense; its *capability*
claim is in the Pearl sense (a `do()` and its consequence). The paper must say so in one
sentence and never let the mask stand in for the semantics. Designed against **P-7** (vocabulary
with no referent) and **P-10** (a source's intro cited as its theorem).

### 2.F World models, consequence prediction, embodied intervention

**ha-2018-worldmodels** · `[V]`, **oh-2015-actionconditional** · `[V]` (NIPS 2015),
**hafner-2023-dreamerv3** · `[V]`, **schrittwieser-2020-muzero** · `[V]` (Nature; DOI on abs),
**micheli-2023-iris** · `[V]` (ICLR 2023) · *own:* action-conditioned next-state prediction as the
learned model (Oh: next frame given action; Ha/Dreamer: latent dynamics and imagination; MuZero:
a model that predicts only reward, value and policy and is *never* asked to reproduce the state;
IRIS: a discrete autoencoder plus an autoregressive transformer as the world model). · *leave
open:* consequences are learned from *on-policy experience*, not from `do()` oracles; the model is
scored by return, not by fidelity of the consequence against an exact oracle; MuZero in particular
is the proof that a planner does not need the state's consequence at all — which is the honest
skyline for C4: a *value* head can beat a consequence head on return. · S5.

**baradel-2020-cophy** · `[V]` (arXiv:1909.12000, ICLR 2020) · *owns:* counterfactual physical
outcome given a `do()` on initial conditions, with ground truth from a simulator that executes the
intervention. · *leaves open:* the model learns latent confounders and predicts trajectories; no
fixed point, no per-row control. This is the nearest *embodied* occupant of "supervise with an
exact `do()` oracle". · S13 lineage, memory, verified.

**wang-2022-cdl** · `[V]` (arXiv:2206.13452, ICML 2022) · *owns:* learning which state variables
causally drive which, so the dynamics model ignores spurious correlations. · *leaves open:* no
intervention semantics on the model's read. · memory, verified.

**ahmed-2020-causalworld** · `[V]` (arXiv:2010.04296, cs.RO) · *owns:* a robotic benchmark where
every environment variable can be intervened on by the experimenter (`do()` on masses, sizes,
colours). · *leaves open:* the interventions are on the *environment*, not on a context read; no
consequence label. · Q5 lineage, memory, verified.

**vakalis-2026-interventiongap** · `[V]` (arXiv:2608.29998, cs.LG, submitted 2026-08-30) · *owns:*
**the "intervention gap"** — the mismatch between a latent world model's *imagined* effect of an
action and the *actual* environment effect, measured by matched-intervention audits against
environment-endpoint oracles, with the finding that imagined effects can score *below a
no-change predictor* even when reward prediction is good. · *leaves open, exactly:* latent world
models trained by reconstruction/reward, not an attention read; no fixed point; the control is
"no change", not a per-row softmax at matched depth. But it **occupies the metric-with-floor
idea** for C6: an oracle-scored intervention effect with an information floor. The paper should
cite it as the owner of that discipline and state the delta as "per-row softmax control at
matched depth, joint vector label". · Q1.

**lin-2026-scratchworld** · `[V]` (arXiv:2606.31689, cs.SE) · *owns:* executable-consequence
evaluation of world models against a pinned VM oracle, with a changed-field F₁ that credits only
altered fields. · *leaves open:* program states, not attention; but "credit only what moved" is
exactly the record's own do()-bit discipline (READ `MISTAKES.md:755` D-5 "Declaring `0.0` without
a movement test"). · Q1.

**chen-2026-worldmodelroadmap** · `[V]` (arXiv:2607.06401) · *owns:* a definition of world
models as internal simulators. · *leaves open:* a perspective; cite for vocabulary only. · Q1.

### 2.G Safety, reachability, constrained planning

**nakamura-2025-latentreach** · `[V]` (arXiv:2502.00935, RSS 2025) · *owns:* Hamilton–Jacobi
reachability computed in the latent space of a learned world model to obtain safety-preserving
actions beyond collision avoidance. · *leaves open:* HJ value functions, not committors; a single
failure set, not `K` absorbing sets with a `max_k`; no attention read. Nearest occupant of C7. ·
Q15.

**latyshev-2025-safeplanning** · `[V]` (arXiv:2506.04828) · *owns:* model-based RL that switches
between planning in the model and direct policy execution under adaptive safety thresholds. ·
*leaves open:* same as above. · Q15.

*Q11 found no work that computes an absorption probability (committor) into a constraint set
inside an attention read and takes `argmin_a max_k` over candidate moves in the context.* The
absorbing-chain machinery itself is classical and belongs to the resolvent planet's bib; this
planet records only that the *attention-native* version is NOT FOUND.

### 2.H Metrics: sign / direction fidelity and the control

**xun-2026-evidencetype** · `[V]` (arXiv:2607.29484) · *owns:* sign-correctness of a causal
effect as the scored quantity for language models trained on interventional data, with
observational context as the noise floor (`38 %` mixed context vs `82 %` interventional-only, per
the abs page). · *leaves open:* LLM prompts; no operator; but it is the nearest published
"sign-fidelity with a floor", and it is four weeks old. · Q10.

**hoogland-2022-ite** · `[V]` (arXiv:2209.06101, stat.ME) · *owns:* discrimination and
calibration of individualised treatment-effect predictions, including whether predicted benefit
vs harm (the sign) is right. · *leaves open:* clinical ITE; per-unit scalar. · Q10.

The record's own metric (READ `MATHEMATICS.md:384-392`) is a sign-fidelity with a Clopper–Pearson
interval and a *paired* McNemar against softmax on byte-identical draws. Nothing found pairs the
sign test against a per-row control at matched depth on the *same* intervention draws; the
paired-control discipline is the record's and is NOT FOUND outside it (Q10). What *is* found is
the sign metric itself (directional accuracy is a standard forecasting statistic) — so the paper
must not call the sign metric new; it may call the *pairing* new, narrowly.

### 2.I Record-cited items re-verified this session

**kim-2026-topological** · `[V]` (arXiv:2603.02289, stat.ME, Kim & Lee, submitted 2026-03-02) ·
*owns:* "Topological Causal Effects" — treatment effects on persistence-diagram features of an
outcome distribution, doubly robust estimator, a test for whether treatment alters shape. · *what
the record says:* `results/r9_maths_survey.md:314,503` (READ: "`ψ_d(t) := E[φ¹_{i,d}(t) −
φ⁰_{i,d}(t)]`, a curve in `t`… verified to resolve and title-match: Kim & Lee, 2026-03-02"). The
identifier, authors and date **match**. · *leaves open:* a function-valued *label class*, not an
attention component; it is the licence for the paper's "topological layer" to score a consequence
by a persistence curve rather than a scalar. · S8.

**bai-2026-pinnchangepoint** · `[V]` (arXiv:2604.25655, stat.ML, submitted 2026-04-28; HTML read)
· *owns:* Theorem 3.1 "Residual lower bound on change-point subintervals": any fitting window that
straddles a regime change carries a strictly positive residual energy under a single constant
parameter, and a window inside one regime reads zero. · *what the record says:* `README.md:216-219`
and `workdonenewseal.md:464` (READ: "The learned-model form is arXiv:2604.25655, four months old,
whose Theorem 3.1 is the must-fire stated as a theorem"). The identifier, date and Theorem 3.1
**match**. **Two cautions for the paper:** (a) the paper does **not** cite Basseville & Nikiforov,
so "learned-model form of §7.2.4" is the *record's* reading, not the source's — write it as an
analogy owned by the record; (b) Theorem 3.1 is about a PINN's physics residual over a window,
not about a hidden *cause*; the must-fire it discharges is "no plant ⇒ flat residual", and only
that. Designed against **P-10**. · S10.

**basseville-1993-detection** · `[V]` (author-hosted page; PDF TOC read: Ch. 7 "Additive Changes
in Linear Models" p.209; §7.2 "Statistical Approach" p.214; **§7.2.4 "State-Space Models"
p.234**; §7.2.5 "Statistical Decoupling for Diagnosis" p.245; §7.2.6 "Statistical Detectability"
p.252) · *owns:* additive-change detection in state-space models via the innovation / residual
of a Kalman filter (GLR on the residual), which is the closed form the record says discharged
X₃₅'s must-fires (READ `README.md:216-218`). The section number and its subject **match** the
record's citation. · *leaves open:* X₃₅ is occupied and the record already conceded it; the paper
lists it under "refuted / occupied" and moves on. · S9, Q4.

**peng-2024-limitations** · `[V]` (arXiv:2402.08164, stat.ML) — identifier and title confirmed
for the brief's obstruction 3 (brief §1 asked for this VERIFY); **sanford-2024-logdepth** · `[V]`
(arXiv:2402.09268). Both belong to the expressivity planet; recorded here only because a
*consequence* is a composition (`z' = R(P') V'` after `P' = S(P, a)`) and the paper's obstruction
3 rests on Peng et al.

---

## 3. Verdict

### 3.1 OCCUPIED (owner named; the paper cites before it names)

| component | owner(s) | what they take |
|---|---|---|
| Linear SCM `X = (I − B)^{-1}Z`; `do()` as row surgery; total effect = entries of `(I − B)^{-1}` | Pearl 2009; Shimizu et al. 2006; Gische & Voelkle 2022 `[U]` on the formula | the algebra of C2 and C4 on a *data-generating* graph |
| Consequence of an intervention = displacement of an equilibrium / fixed point | Mooij–Janzing–Schölkopf 2013; Bongers et al. 2021; Bottou et al. 2013 §7.3 (first-order) | the *definition* in C4, in SCM/ODE terms |
| The order of `do()` and equilibration (EMC) | Dash 2005 (Def. 4, Thms 1–2); Voortman et al. 2010; Blom et al. 2019 | a *bind* the shape must carry (§2.B DERIVED) |
| `(I − γP)^{-1}` as an occupancy / successor operator; cached resolvent fails under a transition change | Dayan 1993; Momennejad et al. 2017 & Russek et al. 2017 `[U]` on the mechanism | C2 as an object; the argument that C4 needs a re-solve |
| Supervising a transformer with exact interventional targets from synthetic SCMs (in-context) | Do-PFN 2025; CausalFM 2025; CausalTimePrior 2026; Ke et al. 2022; Lorch et al. 2022 | C5 for *scalar per-query* effects |
| Attention for treatment effects; balancing ≡ attention | Melnychuk et al. 2022; Zhang et al. 2023 (CInA) | single-effect estimation is softmax's own ground (strengthens D-1) |
| The three meanings of "causal attention" | Karagodin et al. 2024; Nichani et al. 2024; D'Angelo et al. 2025; Wang et al. 2026; Yang et al. 2021 | the vocabulary; the paper must disambiguate |
| Attention as a fixed point (Hopfield update; DEQ layer) | Ramsauer et al. 2020; Bai et al. 2019 | fixed-point-ness of a layer; **not** interventions on it |
| Action-conditioned consequence prediction; planning with a learned model that never predicts the state | Oh et al. 2015; Ha & Schmidhuber 2018; Hafner et al. 2023; Schrittwieser et al. 2020; Micheli et al. 2023 | the world-model skyline: return without consequence fidelity |
| Counterfactual physics from a simulator `do()`; robot benchmark with experimenter `do()` | Baradel et al. 2020; Ahmed et al. 2020 | embodied exact oracles |
| Oracle-scored imagined intervention effect with a no-change floor | Vakalis 2026 (2026-08-30) | the metric-with-floor discipline for C6, on latent world models |
| Executable-consequence oracle, credit only changed fields | Lin & Zhang 2026 | the do()-bit movement discipline (D-5), on program states |
| X₃₅ residual hidden-cause inference | Basseville & Nikiforov 1993 §7.2.4 (State-Space Models, p.234); learned-model analogue Bai et al. 2026 Thm 3.1 | conceded by the record |
| Persistent-homology causal effect (function-valued label) | Kim & Lee 2026 | a label class for the topological layer |

### 3.2 NEAR-MISS (touches a component, does not take it)

- **Scetbon et al. 2024 (FiP)** — see 3.4; takes "fixed point of a causally masked transformer +
  `do()` by clamp-and-re-solve", leaves `γ`, row-stochasticity, the certificate, the context read,
  absorbing sets, the committor, the vector label and the control.
- **Saha et al. 2025** — soft interventions on cyclic equilibrium SCMs; the generalisation of C4 to
  "nudge", not the component.
- **Nakamura et al. 2025; Latyshev et al. 2025** — safety value in a learned latent space; not a
  committor into `K` sets, not `argmin_a max_k`, not an attention read (C7).
- **Xun 2026; Hoogland et al. 2022** — sign fidelity with a floor; not paired against a per-row
  control on identical draws (C6).
- **Miller et al. 2025** — counterfactual abduction in-context; scalar.
- **Zečević et al. 2021** — SCM as message passing with edge-removal interventions; GNN, one hop per
  layer.

### 3.3 NOT FOUND (absences, with the queries that failed to find them)

1. The resolvent read `P(I − γP)^{-1}V` of a *context* (row-stochastic `P` from the three-corner
   family, learnable `γ`) as the attention output, with **absorbing constraint rows** — Q6, Q9,
   Q11. (The resolvent planet owns the general resolvent literature; this planet records only that
   no *causal-inference* source reads a context this way.)
2. The consequence `Δz = z(do a) − z` as a **vector-valued supervised label** over all positions,
   scored **jointly** (not per coordinate) — Q1, Q3, Q6, Q16.
3. A **per-row softmax control at matched depth and parameters** paired on byte-identical
   intervention draws (McNemar / TOST) — Q10. The sign metric is standard; the pairing is not
   found.
4. The **committor into `K` absorbing constraint sets** computed inside the attention read, with
   `argmin_a max_k q^{(k)}(do a)` over candidate moves present *in the context* — Q11, Q15.
5. An **EMC (order-of-`do()`-and-settle) test** inside an attention model — Q7 (Dash's theorems
   exist; no attention instance).
6. An intervention on a **Deep Equilibrium** layer or on the Hopfield fixed point of attention —
   Q9.
7. A **robot / embodied** transformer that supervises consequences with exact `do()` oracles on
   an *attention fixed point* — Q5, Q12 (CoPhy and CausalWorld are simulators with `do()`, no
   fixed-point read).

### 3.4 SURPRISE — flagged loudly

**Scetbon, Jennings, Hilmkil, Zhang, Ma (2024), *A Fixed-Point Approach for Causal Generative
Modeling*, arXiv:2404.06969 `[V]`, full text read.** It defines an SCM as a fixed point
`X = Pᵀ H(PX, PN)` with `H` lower-triangular (Def. 2.3), parameterises `H` by a transformer whose
attention carries the ordinary triangular mask (Def. 4.1), performs `do()` by composing `H` with a
clamping map and re-solving (§2.4), computes counterfactuals by abduction (Thm 2.14), and reaches the
fixed point by composing the map `d` times (Alg. 2). A reader who has seen only the abstract of the
shape will say "this is FiP". The paper must pre-empt that reader in the prior-art section with the
five exact differences in §2.C: (a) data-generating SCM over `d` variables vs a read of an `s`-token
context; (b) acyclic `d`-fold composition vs a `γ`-resolvent with a printed Neumann `δ` (I4) on a
row-stochastic `P`; (c) no absorbing sets / committor / safest move; (d) no per-row control at
matched depth, no vector-fidelity metric; (e) their interventional answer *is* the model's own
re-solved fixed point — exactly the D-2 hazard the record's contract forbids (READ
`MISTAKES.md:710` "An oracle that is the arm's own resolvent"). The same authorship (Zhang, Ma)
owns CInA (§2.D), so the two together occupy "attention ⇄ causal inference" at the *foundation-model*
level; the shape's delta survives only as stated narrowly in §3.3, items 1–4.

Second surprise, smaller: **Vakalis 2026, arXiv:2608.29998 `[V]`**, dated 2026-08-30, owns the
"oracle-scored intervention effect with a no-change floor" discipline for world models. It does not
touch attention or fixed points, but it means the paper's C6 metric section must cite a paper four
days old as the owner of its *control philosophy* and claim only the pairing against a per-row
softmax at matched depth.

Third, a bind rather than a surprise: **Dash 2005 EMC**. The shape defines consequence as
"intervene, re-solve". For a triangular `P` the two orders coincide by nilpotency (§2.B DERIVED);
for any `P` with feedback or re-targeted absorbing rows they do not, and Dash's Theorem 1 says when.
The paper should file EMC as a proposition with both halves (holds for corner 3 by
`Nilpotent.pow_card_eq_zero`; can fail otherwise, with a planted feedback instance) — designed
against **V-24** (an identity bind whose rejection region is empty) and **D-7** (a prediction filed
without its counter).

---

## 4. Design against the taxonomy — what this lineage hands the plan, and the mechanism each is aimed at

Each item names the `MISTAKES.md` mechanism (READ headings, `MISTAKES.md:34-2190`).

1. **The `do()` oracle runs on the latent environment chain; the arm never sees its transition
   matrix.** Aimed at **D-2** (`MISTAKES.md:710`, an oracle that is the arm's own resolvent). FiP
   §2.4 is the published example of the hazard: its interventional distribution is the model's own
   fixed point. Literature licence for the design: Xia et al. 2021 (the hierarchy holds for neural
   models — interventional targets must come from outside the model); Do-PFN / CausalTimePrior
   (paired observational/interventional synthetic data as the training signal).
2. **Every `do()` draw ships a planted zero-effect intervention and a planted feedback instance.**
   Aimed at **V-24** (identity bind with empty rejection region) and **D-5** (`MISTAKES.md:755`,
   declaring `0.0` without a movement test). ScratchWorld's "credit only changed fields" and the
   record's own do()-bit (`results/r10_it17_battery.md:270-342`, READ: one boundary node moved the
   label by exactly `0.0`) are the two precedents; EMC violation (Dash Thm 1) is the feedback plant.
3. **The vector label `Δz` is scored jointly (relative `ℓ₂` and the full sign pattern), never only
   as a per-coordinate sign rate.** Aimed at **V-26** (a marginal assertion standing in for a joint
   claim) and **D-1** (racing a proven optimum): `MATHEMATICS.md:399-412` already shows softmax at
   `0.807843` sign fidelity on the scalar; CInA shows single-effect estimation is softmax's ground.
   The joint vector is the only ground §0.3 (`MATHEMATICS.md:106-127`) claims.
4. **The control is a per-row softmax at matched depth `⌊log₂ t*⌋ + 2` on byte-identical draws,
   with McNemar / TOST fixed before data (N ≥ 8, MDE, TOST N = 70 per brief §6).** Aimed at **M-2**
   (threshold refitted to the data it judges), **M-3** (pilot spread taken as realised spread),
   **M-9** (finest achievable `p` cannot reach the quoted `α`). Vakalis 2026 supplies the
   no-change floor; the record supplies the pairing.
5. **Two absorbing sets minimum on any committor bed.** Aimed at **V-12** (`MISTAKES.md:189`, a
   single absorbing target makes the label constant) and **V-8** (the PASS half's label is
   constant): with `K = 1`, `q ≡ 1` on every transient state that can reach the set.
6. **A domain census that every drawn `P` is row-stochastic including its absorbing rows, before
   the Neumann certificate is quoted.** Aimed at **V-25** (a theorem whose hypothesis no draw
   satisfies); I4's planted negative (rows summing `1.5`, err `119.37` vs bound `1.143`, RUN by
   the coordinator) is the rejection-region witness.
7. **The EMC proposition with both halves** (§3.4, third item). Aimed at **D-7** (a prediction
   filed without its counter).
8. **"Causal attention" disambiguated in one sentence with three citations** (§2.E). Aimed at
   **P-7** (vocabulary with no referent) and **P-10** (a source's intro cited as its theorem).
9. **The 2604.25655 attribution written as the record's analogy, not the source's claim** (§2.I).
   Aimed at **P-10** and **P-3** (a stale claim never retracted): the source does not cite
   Basseville & Nikiforov.
10. **The consequence read priced as one extra triangular solve** (§2.B DERIVED: `Δz = (I −
    γP')^{-1}(ΔV + γΔP z)`). Aimed at **M-8** (pricing every arm at one arm's rate) and **P-8** (a
    headline that states an upper bound as a price): the price is `≈ s²d` multiply-adds per
    intervention, the same as I5, and it must be measured on the certified RTX 4060, not asserted.
11. **MuZero as the honest skyline for any "consequence head".** Aimed at **D-1** and the R-SKY
    rule (brief §1): a planner that never predicts the state can win on return; the shape's claim
    is fidelity of the consequence with a certificate, not return.

---

## 5. Limits of this sweep

Collected once. (i) Seven landing pages were reached only through registry records (CrossRef,
Open Library) after publisher pages refused; titles and authors matched but section-level content
of Iwasaki–Simon 1994, Weinberger 2023, Momennejad 2017 and Russek 2017 was not read. (ii) One
source (OpenReview bpF8zgSt41) is `[U]`: two bot walls, dblp 0 hits, Semantic Scholar 429; it is
listed as a pointer only and must not carry a claim. (iii) The Gische–Voelkle closed-form
attribution rests on a search snippet; the record's own `r9_maths_survey.md:313` is the load-bearing
source for that formula. (iv) Searches were English-only and arXiv-weighted; the safety /
reach-avoid literature was sampled, not swept — the safety planet owns it. (v) No number in this
file was produced by new code; the only executions were text extraction from three locally saved
PDFs and `sed`/`grep` reads of the repository. (vi) The DERIVED identity in §2.B is four lines of
algebra and was not run numerically this session — it is a candidate for the coordinator's
`shape_identities.py` as I6 (`torch.allclose` of the two sides at `γ = 0.5` on a planted row
surgery), which this planet may not write.
