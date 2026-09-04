# SWEEP — safest move under constraints: committors, reachability, safe control

MARS-S5, 2026-09-03. Prior-art sweep for one lineage of the shape paper (`$SCRATCH/BRIEF.md`
§1, §7). Companion BibTeX: `$SCRATCH/sweep/bib_safety.bib` (62 entries, keys
`surname-year-keyword`). No code was written; no git write was made.

**The question put to this sweep.** Is

```
safest move  =  argmin_a  max_k  q^{(k)}(do a),     q^{(k)} = (I − Q)^{-1} R_k 1,
```

the committor / absorption probability into constraint set `𝒜_k`, computed by a resolvent
read inside an attention layer, occupied — and by whom; and what is the standard name of
each piece (the mentor rule: field vocabulary for things the author already built).

**Evidence classes used here.** `[V]` — the arXiv abs page or the DOI registry record
(`api.crossref.org/works/<doi>`) was fetched this session and the title matched; for two
books without a DOI, `[V-cat]` — a library catalogue record (Open Library, Internet
Archive) or the publisher-hosted text was fetched and matched. `[U]` — reached only through
a search-index snippet or the brief; used below only for venue sub-fields, never for a
source's existence. `READ path:line` — repository file. `DERIVED` — steps written out.
`RUN` — none this session (this sweep ran no code).

**Reading depth, so that P-10 (a source's intro cited as its theorem) cannot occur.**
Two sources were read at theorem/equation level: Grinstead–Snell §11.2 (Thm 11.1, Def 11.3,
Thm 11.4, Thm 11.6) and Roffo et al. 2026 (full HTML: operator, normalisation, convergence
condition, tasks). Every other "owns" line below is from the abstract or registry metadata
plus, where noted, the repository's own earlier full-text record
(`V13_TIER6_PRIOR_ART.md`, `PRIOR_ART.md`); such lines say what the paper is *about*, not
what its theorems *license*.

---

## 0. Search log (25 queries, each with the witness it recovered — V-7 discipline)

`MISTAKES.md` V-7 (`READ MISTAKES.md:117-130`) condemns a search structurally incapable of
finding anything read as absence. Every query below therefore records what it *did* find; a
query that recovered nothing on-point is marked so, and its absence claim is only as strong
as the neighbouring queries that recovered known items.

| # | query (verbatim) | witness recovered |
|---|---|---|
| 1 | `Altman "Constrained Markov Decision Processes" 1999 Chapman Hall CRC ISBN` | Open Library / Routledge records, ISBN 9780849303821 |
| 2 | `"safest action" selection attention transformer constraint` | Jeddi et al. 2021 (arXiv:2107.13944); Park et al. 2026 (arXiv:2606.09749) |
| 3 | `committor function transformer attention learned neural network arXiv` | Zhang et al. 2026 (arXiv:2606.31832); Khoo–Lu–Ying |
| 4 | `"reach-avoid" probability committor Markov chain absorption equivalence` | Avila–Junca 2018; Wisniewski–Bujorianu 2021; Grinstead–Snell §11.2 host page |
| 5 | `attention layer resolvent "(I - \gamma P)^{-1}" successor representation transformer` | two false leads (Johnson et al. 2020; Haggi-Mani–Rish 2026) — see §5 |
| 6 | `multiple constraints safe reinforcement learning minimax lexicographic constrained MDP arXiv` | Kushwaha et al. 2025; Park et al. 2026 max-min; LexiSafe 2026; Yao et al. 2023 |
| 7 | `"Neumann series" attention transformer iterative refinement fixed point equilibrium attention arXiv` | Anbar Jafari 2025; no Neumann-series attention on-point |
| 8 | `Decision Transformer causal confusion trajectory conditioning does not model consequences arXiv` | Brandfonbrener et al. 2022; Paster et al. 2022 (snippets; both then fetched) |
| 9 | `absorbing Markov chain fundamental matrix attention mechanism language model safety arXiv` | Kao et al. 2025; Wu et al. 2024; Makkuva et al. 2024 |
| 10 | `"argmin" "max" absorption probability action selection Markov decision process worst-case constraint set` | Dufour–Prieto-Rumeau 2023 (absorbing MDPs); nothing for the composite |
| 11 | `multiple control barrier functions compatibility conflicting constraints minimax safety filter arXiv` | Tan–Dimarogonas 2022; Cohen–Lavretsky–Ames 2025 |
| 12 | `harmonic measure Dirichlet problem graph neural network attention absorbing random walk layer` | nothing on-point (Dirichlet-energy / oversmoothing papers only) |
| 13 | `"successor representation" attention transformer in-context resolvent Neumann series arXiv` | nothing on-point |
| 14 | `"committor" safety control probabilistic "reach-avoid" OR "safe set" stochastic dynamics arXiv` | Gleason–Vinod–Oishi 2017; Schmid et al. 2026; Omidi et al. 2025; no control paper uses the word "committor" |
| 15 | `Fisac "Bridging Hamilton-Jacobi Safety Analysis and Reinforcement Learning" arXiv` | ICRA 2019, DOI 10.1109/ICRA.2019.8794107 |
| 16 | `Grinstead Snell "Introduction to Probability" chapter 11 absorbing Markov chains fundamental matrix pdf dartmouth` | chapter PDF (binary, unrenderable here) and the LibreTexts host page (read) |
| 17 | `transformer in-context learning Markov chain hitting probability absorption "one layer" cannot compute reachability` | Zhou–Tian–Diggavi 2024; Makkuva et al. 2024; nothing on hitting probabilities |
| 18 | `"committor" "attention" "absorbing" resolvent transformer layer safest action arXiv` | **Roffo et al. 2026 (arXiv:2603.00175)** via a secondary snippet — the SURPRISE |
| 19 | `"multiple unsafe sets" OR "multiple avoid sets" reach-avoid "max over" constraints minimum violation probability action selection Markov` | LP approach to reach-avoid (arXiv:1411.5925, snippet only, not fetched, not cited); nothing on max over K sets |
| 20 | `Kemeny Snell "Finite Markov Chains" 1976 Springer Undergraduate Texts in Mathematics DOI 978-0-387-90192-3` | ISBN 9780387901923 (index); Internet Archive record of the 1960 Van Nostrand edition (fetched) |
| 21 | `"consequence" prediction attention intervention "do-operator" transformer layer counterfactual next state safest action under constraints arXiv 2025 2026` | causal-transformer papers (another planet's lineage; not cited here) |
| 22 | `Chebyshev scalarization multi-objective reinforcement learning Van Moffaert Drugan Nowé 2013 max-min worst objective` | DOI 10.1109/ADPRL.2013.6615007 |
| 23 | `"least-restrictive" safety filter one-step lookahead argmax safety value function reach-avoid Hamilton-Jacobi discrete action` | Borquez et al. 2023; Ganai–Gao–Herbert 2024 |
| 24 | `"personalized PageRank" attention head transformer language model resolvent propagation inside attention arXiv` | ParaFormer (arXiv:2512.14619) |
| 25 | `Roffo Melzi Cristani "Infinite Feature Selection" ICCV 2015 DOI path integral adjacency (I - alpha A)^-1` | DOI 10.1109/ICCV.2015.478 |

Beyond the 25 searches, 58 landing pages / registry records were fetched (arXiv abs ×39,
Crossref ×14, LibreTexts, Internet Archive, Open Library, one arXiv full HTML, plus the
paywalled landing pages that returned 403 or cookie redirects: SIAM, Springer ×2, Annual
Reviews, Elsevier ×2, Nature, MIT Press, Routledge — each replaced by its Crossref record).

---

## 1. Per-source entries

Format: **id** · class · *owns* (the equation or mechanism, in this sweep's words) ·
*leaves open relative to the shape* · *query*.

### A. Absorbing chains, fundamental matrix, harmonic measure

**kemeny-1960-finitemarkov** · [V-cat] (Van Nostrand 1960 record; Springer 1976 reprint
[U]) · *owns:* the canonical form `P = [[Q, R], [0, I]]`, the fundamental matrix
`N = (I − Q)^{-1}`, absorption probabilities `B = N R`, expected absorption times `N 1`.
*Leaves open:* everything after the identity — no learning, no attention, no decision
rule, no intervention. *Query:* #20.

**grinstead-1997-probability** · [V] (§11.2 read on the LibreTexts host) · *owns:* the
textbook statement of the same four facts: Thm 11.1 absorption is certain (`Q^n → 0`),
Def 11.3 `N = (I − Q)^{-1}`, Thm 11.4 `n_ij` = expected visits to `j` from `i`, Thm 11.6
`B = N R`. The repository already quotes Def 11.3 / Thm 11.6 verbatim
(`READ V13_TIER6_PRIOR_ART.md:366-380`) and concedes that the identification
"committor = absorbing resolvent" has zero delta (`READ V13_TIER6_PRIOR_ART.md:389-398`).
*Leaves open:* the *use* — as a layer's read and as a label. *Query:* #16.

**doyle-1984-electric** · [V] (arXiv:math/0001057) · *owns:* the hitting probability as the
harmonic function with Dirichlet boundary data, escape probability, effective resistance,
the electrical reading of `P_x(τ_a < τ_b)`. `MATHEMATICS.md` §11 (`READ`, lines 542-609)
already derives the forest-ratio form `ω_x = M_xa / M_aa` and cross-checks it to `1e-10`;
that section is this book's material plus Kirchhoff. *Leaves open:* discounting `γ < 1`
(the forest identity is for the undamped walk — `READ MATHEMATICS.md:604-607`), K > 2
boundary sets, learning. *Query:* seed.

**katz-1953-status** · [V] · *owns:* the name "Katz index" for `(I − αA)^{-1} 1` — the
discounted count of walks of all lengths. *Leaves open:* stochastic `P`, boundaries, reads
of a value matrix. *Query:* via Roffo 2026's credits (#18) then Crossref.

**schweitzer-1968-perturbation** · [V] · *owns:* perturbation theory of the fundamental
matrix when the transition matrix changes — the field name for "how much does the
resolvent move when a row of `P` is intervened on". *Leaves open:* the interventional
semantics (`do`), attention, the decision. *Query:* memory of the name → Crossref fetch;
mark the *lead* as memory-originated, the record as [V].

**sherman-1950-inverse** · [V] · *owns:* the rank-one inverse update. Under the shape a
single-row intervention `P' = P + e_i uᵀ` (with `uᵀ = p'_i − p_i`, `uᵀ 1 = 0`) gives,
with `M = (I − γP)^{-1}` and `z = M V`,

```
Δz  =  γ · (M e_i) · (uᵀ z) / (1 − γ uᵀ M e_i)                     (DERIVED, S-M identity)
```

so one candidate move costs one column of `M` plus one `s·d` inner product — `O(s d)`
given the unintervened solve — not a fresh `O(s³)` inverse. Two consequences fall out:
(i) for `V = 1`, `z = 1/(1−γ)·1` and `uᵀ z = 0`, so `Δz ≡ 0` exactly — a planted identity
check for the interventional channel (designed against V-24: the bind has a rejection
region as soon as `V` is not constant); (ii) for causal `P`, `M` is lower-triangular
(brief I5), so `(M e_i)_j = 0` for `j < i`: the displacement is zero before the intervened
position — consequences propagate forward in time by construction, no mask needed.
*Leaves open:* multi-row interventions (Woodbury), stochastic-row projection after the
edit. *Query:* memory of the identity → Crossref fetch.

### B. Transition path theory and committors

**e-2006-transitionpaths** · [V] (Crossref) · *owns:* the committor `q(x) = P_x(τ_B < τ_A)`
as the central object of transition path theory; reactive probability current and flux in
terms of `q`. *Leaves open:* discrete learned graphs, discounting, more than two sets,
any decision over interventions. *Query:* seed.

**e-2010-tptreview** · [V] (Crossref) · *owns:* the review form of TPT; isocommittor
surfaces as the reaction coordinate (the brief's §4 "isocommittor surfaces" in the
topological layer inherits this name). *Leaves open:* as above. *Query:* seed.

**metzner-2009-tptjump** · [V] (Crossref) · *owns:* TPT on a finite state space: the
committor as the solution of the discrete Dirichlet problem `(Lq)_i = 0` on the interior,
`q|_A = 0`, `q|_B = 1`, plus discrete reactive flux — literally the problem
`ceq/beds/bed_1.py:188-198 committor` solves (`READ`) and `reactive_flux` at
`ceq/beds/bed_1.py:212-217` computes. *Leaves open:* the read inside a layer, the decision
rule, `K > 2` sets (TPT is two-set by definition; K sets need the `B = NR` column form).
*Query:* seed.

### C. Committor learning

**khoo-2019-committor** · [V] · *owns:* neural committor by minimising the Dirichlet
energy with soft boundary penalties (`READ V13_TIER6_PRIOR_ART.md:400-416` for the loss).
*Leaves open:* graph/discrete chains, resolvent reads, decisions. *Query:* seed.

**li-2019-committor** · [V] (abs + Crossref) · *owns:* deep-learning committor with
adaptive sampling; committor is a regression target. *Leaves open:* same. *Query:* seed.

**contrerasarredondo-2026-committor** · [V] (Crossref; arXiv abs under the longer preprint
title) · *owns:* GNN (geometric vector perceptrons) predicting the committor from atomic
positions without collective variables. Verification note for the brief: the journal
title is *Learning the committor without collective variables*, Nature Comput. Sci.
6(4):350-357, online 2026-02-17; the arXiv v-title carries the prefix "From Atoms to
Dynamics". Same six authors; cite the journal title. *Leaves open:* same as C. *Query:* seed.

**zhang-2026-committorpairformer** · [V] · *owns:* a simplified Pairformer (attention)
predicting the committor of biomolecules — attention *around* a committor target, not
attention *computing* a committor. *Leaves open:* the resolvent-in-layer; any decision.
*Query:* #3.

### D. Stochastic reachability and reach-avoid

**abate-2008-reachability** · [V] (Crossref) · *owns:* probabilistic safety of controlled
discrete-time stochastic hybrid systems as a multiplicative-cost DP; maximal
probabilistic safe sets. *Leaves open:* discrete learned chains, one-shot linear solve
(the DP is finite-horizon), attention. *Query:* seed.

**summers-2010-reachavoid** · [V] (Crossref) · *owns:* the reach-avoid probability
`r(x) = P(reach target K' while staying in K)` and its DP recursion; a single avoid set.
DERIVED naming identity: with target set `B` absorbing and unsafe set `A` absorbing, the
reach-avoid probability *is* the committor `q_{A→B}`; the constraint-set absorption
probability `q^{(k)}` of the shape is the committor into `𝒜_k` against all other absorbing
sets, and `1 − q^{(k)}` is the reach-avoid probability of "everything but `𝒜_k`". The two
literatures name one object. *Leaves open:* K > 1 avoid sets scored separately (the DP
folds them into one), the read inside a layer. *Query:* seed.

**gleason-2017-reachavoidlagrangian** · [V] · *owns:* grid-free underapproximation of
stochastic reach-avoid level sets. *Leaves open:* all of the shape. *Query:* #14.

**avila-2018-reachability** · [V] · *owns:* reach-avoid for Markov control models as a
long-run-average problem. *Leaves open:* same. *Query:* #4.

**wisniewski-2021-safedp** · [V] · *owns:* a safety function for MDPs with stopping times
by DP. *Leaves open:* same. *Query:* #4.

**schmid-2026-reachavoidmpc** · [V] · *owns:* reach-avoid probability maximised by a DP
layer over an MPC tracker, closed loop abstracted as an MDP. Nearest control-theory
relative of "score candidate moves by absorption probability" — but the score is a DP
value, not a resolvent read, and there is one avoid set. *Query:* #14.

**omidi-2025-averagereward** · [V] · *owns:* probabilistic safety as an average-reward MDP
solved by LP. *Query:* #14.

**dufour-2023-absorbingmdp** · [V] · *owns:* the name "absorbing MDP" and its
occupation-measure geometry. *Leaves open:* everything but the name. *Query:* #10.

### E. Hamilton–Jacobi reachability and safety filters

**bansal-2017-hjreach** · [V] (CDC 2017 venue [U]) · *owns:* the HJ reachability overview:
safety as the sub-zero level set of a min-over-time value function. *Leaves open:*
probabilistic (Markov) dynamics; the shape's linear solve replaces a PDE. *Query:* seed.

**fisac-2019-bridging** · [V] (Crossref) · *owns:* time-discounting the min-payoff safety
problem so the Bellman operator contracts, making it learnable. DERIVED correspondence:
the shape's `γ < 1` does for the Neumann certificate (brief I4,
`γ^{K+1}/(1−γ)`) exactly what Fisac's discount does for the HJ backup — buys a
contraction — and the same caveat applies: at `γ < 1`, `(I − γQ)^{-1} R_k 1` is the
*discounted* absorption probability `E[γ^{τ} 1{absorbed in k}]`, which rewards delaying
absorption, not only avoiding it. The paper must say which `γ` its safest-move objective
is stated at (see §4.1). *Query:* #15.

**hsu-2021-reachavoidrl** · [V] · *owns:* the discounted reach-avoid Bellman backup
(max of the current constraint value and min of next-step values) and reach-avoid
Q-learning. *Leaves open:* a one-solve linear form; K sets scored separately. *Query:*
memory → abs fetch.

**hsu-2023-safetyfilter** · [V] · *owns:* the umbrella name "safety filter": monitor a
safety value, intervene on the nominal action when it says so. The shape's
`argmin_a max_k q^{(k)}(do a)` is, in this vocabulary, *a one-step safety filter whose
safety value is the worst-constraint committor*. *Leaves open:* the committor as the value,
the resolvent as its computation, attention as the host. *Query:* memory → abs fetch.

**borquez-2023-lrf** · [V] · *owns:* the "least-restrictive filter": the maximal set of
controls keeping the value non-negative, with the nominal action projected onto it. The
shape's `argmin` is the *most*-restrictive point of that set (the single safest action);
the least-restrictive variant would be "any candidate with `max_k q^{(k)} ≤ threshold`".
Both should be stated; the threshold form is where L-CERT's printed δ lives. *Query:* #23.

**ganai-2024-hjrlsurvey** · [V] · *owns:* the survey of HJ value functions used as
safety critics in RL. *Query:* #23.

### F. Control barrier functions and multiple constraints

**ames-2019-cbf** · [V] (ECC venue [U]) · *owns:* CBF-QP safety-critical control: forward
invariance of a safe set enforced by a per-step QP. *Leaves open:* stochastic/Markov
semantics; probabilities. *Query:* seed.

**tan-2022-cbfcompat**, **cohen-2025-cbfcompat** · [V] · *own:* whether K barrier
constraints can be satisfied jointly (compatibility); closed-form QP under vector relative
degree. This is the deterministic analogue of the shape's K-set problem: CBFs ask
"is there an action inside all K constraints"; the shape asks "which action minimises the
worst absorption probability". *Leaves open:* probabilities, resolvents. *Query:* #11.

**park-2026-vlasafetyfilter** · [V] · *owns:* attention heads of a frozen VLA used to
localise target vs obstacle, then a CBF-QP filter. NEAR-MISS in wording only: attention
*informs* a safety filter; it does not *compute* a safety value. *Query:* #2.

### G. Constrained MDPs, safe RL, aggregation over constraints

**altman-1999-cmdp** · [V-cat] · *owns:* the CMDP: expected reward subject to K
expected-cost bounds; occupation-measure LP; Lagrangian duality. In CMDP vocabulary the
shape's `q^{(k)}` is the expected total cost for the indicator cost `1{absorbed in 𝒜_k}`,
and `max_k q^{(k)}` is a Chebyshev aggregation of K costs, which CMDPs do *not* do (they
bound each cost separately or Lagrange-weight them). *Query:* #1.

**achiam-2017-cpo** · [V] · *owns:* the CMDP made neural with per-iterate near-constraint
satisfaction. *Query:* memory → abs fetch.

**alshiekh-2018-shielding** · [V] (AAAI venue [U]) · *owns:* the shield — a reactive
override synthesised from a temporal-logic spec. Boolean safety: an action is allowed or
not; no absorption probability. *Query:* seed.

**kushwaha-2025-saferlsurvey** · [V] · *owns:* the CMDP-centred survey. *Query:* #6.

**yao-2023-gradientshaping** · [V] · *owns:* multi-constraint safe RL as MOO over
constraint gradients. *Query:* #6.

**yang-2026-lexisafe** · [V] · *owns:* lexicographic ordering of several safety costs
above reward — the "lexicographic form" the brief's safest-move line offers as an
alternative to `max_k`. *Query:* #6.

**park-2026-maxmin** · [V] · *owns:* the max-min (worst-objective) criterion with explicit
constraints in MORL — the policy-level cousin of `argmin_a max_k`. *Query:* #6.

**vanmoffaert-2013-chebyshev** · [V] (Crossref) · *owns:* the Chebyshev scalarisation as
an *action-selection* rule — the standard name for "pick the action by the worst of K
scores". The shape's `argmin_a max_k q^{(k)}(do a)` is Chebyshev action selection with
equal weights on K absorption probabilities. *Query:* #22.

**jeddi-2021-lyapunovsafe** · [V] · *owns:* a Transformer memory plus a risk-averse rule
choosing the action with the lowest predicted constraint-violation probability. The
closest existing sentence to "safest action selected by a transformer using a violation
probability" — but the probability comes from an uncertainty model, not from a resolvent
in the layer, and there is one constraint. NEAR-MISS. *Query:* #2.

### H. Decision Transformer and its lack of causal semantics

**chen-2021-decisiontransformer** · [V] (NeurIPS venue [U]) · *owns:* RL as
return-conditioned autoregressive sequence modelling with a causal Transformer. *Leaves
open:* any model of consequences — the causal mask is a temporal mask, not an
interventional semantics. *Query:* seed.

**paster-2022-luck**, **brandfonbrener-2022-rcsl**, **yang-2022-dichotomy** · [V] · *own
the negative:* return-conditioning imitates lucky trajectories in stochastic
environments (Paster: DT fails, ESPER conditions on cluster-average return);
RCSL is optimal only under assumptions stronger than DP's, with counterexample MDPs
(Brandfonbrener); the repair separates action-caused from environment-caused return
(Yang). The paper should cite these three *before* saying DT lacks causal semantics, and
should state its own claim narrowly: the shape computes the consequence *given* `P`; it
does not identify `P` from data, and D-2 (`READ MISTAKES.md:710-727`) still binds — the
oracle's chain must be the latent environment chain, never the arm's own resolvent.
*Query:* #8.

### I. The resolvent as a layer

**dayan-1993-successor** · [V] (Crossref) · *owns:* the successor representation
`M = (I − γP)^{-1}` — the field name for the shape's `z`-operator on a policy chain;
already cited at `READ THEORY.md:234`. *Leaves open:* attention, boundaries, decisions.

**barreto-2017-successorfeatures** · [V] · *owns:* successor features with generalised
policy improvement. *Query:* memory → abs fetch.

**gasteiger-2019-appnp** · [V] · *owns:* personalised-PageRank propagation
`α (I − (1−α) Â)^{-1} H` as a GNN layer, approximated by power iteration. This is a
resolvent read of a value matrix inside a layer, on a fixed graph, in 2019. *Leaves open:*
a content-dependent `P` (attention), causality, boundaries, decisions, a truncation
certificate with printed δ. *Query:* memory → abs fetch.

**gasteiger-2019-gdc** · [V] · *owns:* PPR / heat-kernel diffusion as a graph-convolution
preprocessor. **gu-2020-ignn**, **elghaoui-2019-implicit**, **bai-2019-deq** · [V] · *own:*
the state as the fixed point of a layer equation (joint determination as a layer),
trained by implicit differentiation — the brief's "one-read joint consistency" has these
as its nearest architectural relatives; the shape's fixed point is *linear* in `z`, which
is what makes I4 and I5 available. **wu-2023-difformer** · [V] · *owns:* attention weights
as the closed-form optimum of an energy-constrained diffusion. *Queries:* #7, #13, memory.

**roffo-2015-infinitefs**, **roffo-2017-infinitelatent** · [V] · *own:* ranking by the
resummed power series `(I − αA)^{-1}` over paths of all lengths with an absorbing-chain
reading. *Query:* #25.

**roffo-2026-infsa** · [V] (abs + full HTML) · **THE SURPRISE.** *Owns:* attention as a
discounted Neumann series over an attention matrix,

```
Ĉ = (I − γÂ)^{-1} − I = Σ_{k≥1} γ^k Â^k = γ Â (I − γÂ)^{-1},     Â = ReLU(QKᵀ) / (‖·‖_F + ε),
```

read as an absorbing Markov chain whose transient states are tokens and whose single
absorbing state is the leak `1 − γ Σ_j Â_ij`; token centrality = expected visits before
absorption (Grinstead–Snell Thm 11.4); convergence condition `γ ρ(Â) < 1`; a full `O(N²d)`
variant and a linear-time power-iteration variant with `O(d_h)` state; evaluated on
ImageNet-1K / ImageNet-V2 and attention-quality probes; credits Inf-FS 2015, Katz 1953,
PageRank, Kemeny–Snell 1960. DERIVED comparison with the shape's read
`O = P (I − γP)^{-1} V = Σ_{k≥0} γ^k P^{k+1} V`: **the two are the same series up to the
scalar `1/γ` and the choice of `Â` vs `P`.** What InfSA does *not* have, each item read
from the full text: (1) no causal mask — fully bidirectional; (2) `Â` is
Frobenius-normalised, not row-stochastic, so there is no softmax corner and no analogue
of I1 (at `γ → 0` InfSA's `Ĉ → 0`, it does not become softmax); (3) no absorbing
*constraint sets* inside the context and no Dirichlet data — the only absorbing state is
the leak, so no committor is computed; (4) no decision rule, no intervention, no safety;
(5) no quantitative truncation bound — only the spectral-radius convergence condition;
the shape's I4 bound `γ^{K+1}/(1−γ)` needs row-stochasticity, which `Â` lacks; (6) dense
`N×N`, not a triangular solve (I5); (7) vision tasks only. *Query:* #18.

**yuan-2025-paraformer** · [V] · *owns:* generalised-PageRank message passing inside the
attention block of a graph Transformer, with a linear-time approximation. Second
occupant of "resolvent-type propagation inside attention". *Query:* #24.

**johnson-2020-gfsa** · [V] · listed as a *discarded lead*: a search snippet attributed an
`(I − Q)^{-1}` successor step to it; the abs page does not. Not an occupant. *Query:* #5.

### J. Transformers on / as Markov chains

**makkuva-2024-attentionmarkov** · [V] · *owns:* loss-landscape analysis of one-layer
transformers on first-order Markov data. **zhou-2024-vomc** · [V] · *owns:* one-layer
transformers fail on variable-order Markov chains in context; `D+2` layers implement
context-tree weighting. Neither computes hitting probabilities; both are depth-law
relatives of the brief's obstruction 2, not occupants. *Queries:* #9, #17.

### K. Absorbing chains in language-model safety and decoding

**kao-2025-safetydepth** · [V] · *owns:* the LM as a Markov chain with refusal absorbing;
absorption analysis of alignment depth. Analysis, not selection. **wu-2024-hallucinationabsorbing**
· [V] · *owns:* decoding-time scoring of context tokens by absorbing-chain path sums.
Both are NEAR-MISS on vocabulary ("absorbing state", "fundamental matrix", "LLM") and far on
mechanism (no attention-layer resolvent, no constraint sets, no argmin). *Query:* #9.

### L. Equilibrium refinement in transformers

**anbarjafari-2025-closedloop** · [V] · *owns:* an energy-descent module reaching a latent
equilibrium before each token — nonlinear energy, not a linear resolvent; a relative of
the "next equilibrium state" phrasing, not of the mechanism. *Query:* #7.

---

## 2. Verdict — occupancy of the shape's components

| component of the shape | status | owner(s) | what is left |
|---|---|---|---|
| `N = (I − Q)^{-1}`, `B = N R` (absorption probabilities) | **OCCUPIED** | Kemeny–Snell 1960; Grinstead–Snell §11.2 | nothing — exposition only (the repository concedes this at `V13_TIER6_PRIOR_ART.md:389-398`) |
| committor as a discrete Dirichlet problem | **OCCUPIED** | Metzner–Schütte–Vanden-Eijnden 2009; E–Vanden-Eijnden 2006/2010 | nothing on the identity; `bed_1.py:188-198` is their equation |
| hitting probability as a harmonic function; electrical reading | **OCCUPIED** | Doyle–Snell 1984 | `MATHEMATICS.md` §11 is an instrument built on it, not a claim against it |
| reach-avoid probability ≡ committor | **OCCUPIED (two owners, one object)** | Summers–Lygeros 2010 (control); Metzner 2009 (TPT) | the *naming identity* itself is not stated in either literature as far as this sweep reached — state it as a remark, cite both, claim nothing |
| discounting `γ < 1` to buy a contraction / certificate | **OCCUPIED** in spirit | Fisac et al. 2019; Hsu et al. 2021 | the shape's I4 is a different inequality (Neumann tail in `‖·‖_∞`) on a different object (a linear resolvent); cite Fisac before calling `γ` "the safety discount" |
| `(I − γP)^{-1}` as a named operator | **OCCUPIED** | Dayan 1993 (successor representation); Katz 1953; PageRank | name it "the successor / Katz resolvent" and move on |
| **resolvent read of a value matrix inside a layer** | **OCCUPIED** | Gasteiger et al. 2019 (APPNP, GNN); **Roffo et al. 2026 (InfSA, attention)**; Yuan et al. 2025 (ParaFormer, graph attention) | the delta is *not* "attention as a Neumann series" — that sentence belongs to Roffo 2026. The delta is the conjunction: row-stochastic causal `P` with bitwise softmax parity at `γ = 0` (I1), absorbing constraint sets as Dirichlet data inside the context (I3), the quantitative truncation certificate that row-stochasticity pays for (I4), the triangular solve (I5), and the decision |
| `max_k` over K constraint scores as the action rule | **OCCUPIED (name)** | Van Moffaert et al. 2013 (Chebyshev scalarisation); Park et al. 2026 (max-min MORL) | apply the name; the scores being committors is not in either |
| lexicographic alternative | **OCCUPIED (name)** | LexiSafe 2026; Altman 1999 for the per-constraint-bound form | same |
| "safety filter" / least-restrictive filter framing | **OCCUPIED (name)** | Hsu–Hu–Fisac 2023; Borquez et al. 2023 | call the safest move a one-step safety filter whose value is the worst-constraint committor |
| displacement of the resolvent under a row intervention | **OCCUPIED (name)** | Schweitzer 1968; Sherman–Morrison 1950 | the `O(s d)`-per-candidate pricing and the forward-only displacement for causal `P` are DERIVED here (§1.A) and are, as far as this sweep reached, not stated for attention anywhere |
| Decision Transformer lacks causal semantics | **OCCUPIED (the negative)** | Paster 2022; Brandfonbrener 2022; Yang 2022 | cite them; the shape's claim must be "consequence given `P`", not "consequence from data" |
| transformer + violation-probability action choice | **NEAR-MISS** | Jeddi et al. 2021 | probability from an uncertainty model, not from the layer; one constraint |
| attention → safety filter | **NEAR-MISS** | Park et al. 2026 (VLA) | attention informs a CBF-QP; it computes no value |
| absorbing chains on LLM tokens | **NEAR-MISS** | Kao 2025; Wu 2024 | analysis / decoding scores, no in-layer resolvent, no constraint sets, no argmin |
| multiple constraints, compatibility | **NEAR-MISS** | Tan–Dimarogonas 2022; Cohen–Lavretsky–Ames 2025 | deterministic barriers; the probabilistic K-set worst-case score is not there |
| **the composite:** `argmin_a max_k q^{(k)}(do a)` with `q^{(k)}` a resolvent read inside an attention layer, with a certificate | **NOT FOUND** | — | queries #2, #5, #10, #12, #13, #17, #18, #19 (log §0) |
| a committor / absorption-probability *label class* for an attention bed | **NOT FOUND** (re-confirmed; `PRIOR_ART.md:636-639` said so in round 8) | — | #3, #17 |
| `Σ_k`-vs-`max_k` degeneracy remark (§4.1) | **NOT FOUND** as a stated design hazard | — | #10, #19 |
| the word "committor" in the control / safe-RL literature | **NOT FOUND** | — | #14 |

**Verdict paragraph.** Every *piece* of the safest-move component is owned, and the paper
must cite the owner before it names the piece: the absorption probability (Kemeny–Snell;
Grinstead–Snell), the committor and its Dirichlet problem (E–Vanden-Eijnden; Metzner et
al.), the harmonic-measure reading (Doyle–Snell), the reach-avoid probability
(Summers–Lygeros; Abate et al.), the discount-for-contraction move (Fisac et al. 2019), the
resolvent as a named operator (Dayan; Katz), the Chebyshev / max-min / lexicographic
aggregations (Van Moffaert; Park; LexiSafe; Altman), the safety-filter framing (Hsu–Hu–Fisac;
Borquez), the perturbation of the fundamental matrix (Schweitzer; Sherman–Morrison), and
the DT negative (Paster; Brandfonbrener; Yang). The one component the brief might have
believed was open — *the resolvent of an attention matrix as the layer's read* — is
**occupied, by Roffo et al. 2026 (arXiv:2603.00175)** in attention proper and by
APPNP (2019) and ParaFormer (2025) in graph learning; this is the SURPRISE and it must be
flagged loudly in §7 of the paper. The composite — K absorbing constraint sets as Dirichlet
data inside a causal, row-stochastic, softmax-parity attention operator; the committor
vector read from that operator; candidate moves scored by the worst-constraint committor
under a row intervention priced by Sherman–Morrison; and a printed Neumann truncation
certificate — was NOT FOUND under eight distinct query shapes. The delta, stated narrowly:
*not* "attention as a Neumann series", *not* "committor by neural network", *not* "safest
action by violation probability"; but the conjunction of boundary conditions, causal
row-stochastic parity, certificate, and decision inside one linear read.

---

## 3. Field vocabulary for what the author already built (mentor rule)

| the author's object | standard name(s) | owner to cite |
|---|---|---|
| `z = (I − γP)^{-1} V` | successor representation applied to `V`; Katz / PageRank-type resolvent; Neumann-series propagation | Dayan 1993; Katz 1953; Gasteiger 2019 |
| `O = P z` | one-hop-shifted resolvent read; InfSA's `Ĉ V / γ` | Roffo 2026 |
| `𝒜_k` with identity rows | absorbing states / absorbing sets; Dirichlet boundary | Kemeny–Snell; Doyle–Snell |
| `Q`, `R_k` | transient block and absorbing columns of the canonical form | Grinstead–Snell Def 11.3 |
| `q^{(k)} = (I − Q)^{-1} R_k 1` | column of `B = NR`: absorption probability; committor into `𝒜_k`; hitting probability; harmonic measure of `𝒜_k`; reach-avoid probability of the complement | Kemeny–Snell Thm 11.6; Metzner 2009; Doyle–Snell; Summers–Lygeros |
| `harmonic_residual = 0` (`bed_1.py:200-205`) | the discrete Dirichlet problem residual — `q` is `L`-harmonic on the interior | Metzner 2009; Doyle–Snell |
| `reactive_flux`, `label_by_committor` (`bed_1.py:212-236`) | TPT reactive flux; dominant reaction channel | E–Vanden-Eijnden 2006; Metzner 2009 |
| `γ < 1` | discount; the contraction that makes the certificate hold | Fisac 2019 (for the safety Bellman), Neumann series (for the linear solve) |
| brief I4 bound `γ^{K+1}/(1−γ)` | Neumann-series tail bound in the induced `∞`-norm for a row-stochastic matrix | textbook; no single owner (DERIVED in the brief, RUN by the coordinator) |
| `do(a)`, `Δz` | row intervention on the kernel; perturbation of the fundamental matrix; rank-one inverse update | Schweitzer 1968; Sherman–Morrison 1950 |
| `argmin_a max_k` | Chebyshev (max-norm) scalarised action selection; max-min criterion; one-step safety filter | Van Moffaert 2013; Park 2026; Hsu–Hu–Fisac 2023 |
| the lexicographic form | lexicographic safety hierarchy | LexiSafe 2026 |
| the threshold form `max_k q^{(k)} ≤ δ` | least-restrictive filter (the safe control set) | Borquez 2023 |
| "a third token can veto" via absorbing rows | boundary condition / Dirichlet data; a shield in the Boolean case | Doyle–Snell; Alshiekh 2018 |
| the second exact oracle, `ω_x = M_xa / M_aa` | harmonic measure by effective resistance; forest ratio | Doyle–Snell 1984 (Kirchhoff, all-minors — outside this sweep's fetch set) |
| `λ₂ := ρ(Q)` (`MATHEMATICS.md` §7) | spectral radius of the substochastic transient block; relaxation rate to absorption | Kemeny–Snell |

---

## 4. DERIVED remarks the paper should carry (each with the mechanism it is designed against)

### 4.1 `max_k` is degenerate unless a non-constraint absorbing set exists — V-12, V-8, D-3

If the only absorbing sets are the K constraint sets, Grinstead–Snell Thm 11.1 gives
absorption with probability 1 and Thm 11.6 gives `Σ_k q^{(k)}_i = Σ_j b_ij = 1` for every
transient `i`. Then `max_k q^{(k)} ≥ 1/K` at every position and every candidate, the
`argmin` only redistributes probability among constraints, and "avoid every constraint"
is unattainable by construction. The label's support collapses exactly as V-12
(`READ MISTAKES.md:189-202`: a single absorbing target makes the label constant) and V-8
(the PASS half's label is constant) describe. Three non-degenerate settings, each of which
changes what the objective *means*:

1. **a goal set `𝒜_0`** (reach-avoid proper): `q^{(0)} = 1 − Σ_{k≥1} q^{(k)}`; minimising
   `Σ_{k≥1} q^{(k)}` is maximising the reach-avoid probability (Summers–Lygeros);
   minimising `max_{k≥1} q^{(k)}` is the Chebyshev variant (Van Moffaert). Print both.
2. **`γ < 1` with no goal**: `(I − γQ)^{-1} R_k 1 = E[γ^{τ} 1{absorbed in k}]`, so
   `Σ_k < 1` and delaying absorption counts as safety (Fisac's discounted safety value has
   the same property). Say so; it is a feature at `γ` near 1 and a confound at small `γ`.
3. **an explicit kill / leak state** (InfSA's construction): equivalent to 2.

Design: BED-S's domain census must print, at construction time, the label sd and the
fraction of draws in which the goal set is reachable from the query position (V-25:
a theorem whose hypothesis no draw satisfies; D-3: a dial that does not vary).

### 4.2 The naming identity reach-avoid ≡ committor — P-7

Two literatures, one object (§1.D). Stating it costs one sentence and two citations and
removes a "vocabulary with no referent" hazard (P-7) from a paper that will use both words.

### 4.3 Pricing a candidate move — M-8, P-8

Given `z` and the intervened column `M e_i`, Sherman–Morrison prices each candidate at
`O(s d)`; with a triangular `M` the column costs one forward substitution. State this as a
price with its provenance (DERIVED, §1.A) and *not* as a measured throughput (P-8: an upper
bound stated as a price). A planted identity: `V = 1` must give `Δz ≡ 0` to machine
precision, and the paper must show a `V` for which `Δz ≠ 0` so the bind has a rejection
region (V-24).

### 4.4 The honest skyline for the resolvent read — D-1, R-SKY

Since APPNP, ParaFormer and InfSA compute a resolvent read inside a layer, the paper's
control for "the shape computes multi-hop absorption in one read" is *not* one softmax
layer (D-1, racing a proven optimum) and *not only* the deeper-softmax stack (the brief's
skyline): it must also include an InfSA-type Neumann-series attention *without* boundary
conditions, so that what is measured is the boundary-condition mechanism and not the series.

### 4.5 What to say about Decision Transformers — D-7, L-SIGN

The negative is owned (Paster; Brandfonbrener; Yang). The counter-prediction beside the
paper's prediction: on a bed whose `P` is *given in the context*, a DT-style
return-conditioned arm with matched depth should match the shape's safest move whenever
returns are deterministic functions of the trajectory (Brandfonbrener's sufficient
condition), and should fail only under environment stochasticity (Paster's regime). If the
shape "wins" on a deterministic bed, the win is not about causal semantics.

---

## 5. Discarded leads and false positives (recorded so that they are not re-found)

- **Johnson–Larochelle–Tarlow 2020 (GFSA)** — a search snippet claimed an `(I − Q)^{-1}`
  successor-representation step; the abs page does not mention it. Not cited as an occupant.
- **Haggi-Mani–Rish 2026, arXiv:2607.15449** — a snippet claimed a "resolvent" in a
  renormalisation-group analysis of attention; the abs page has no such content. Not
  entered in the bib.
- **arXiv:1411.5925 (LP approach to reach-avoid)** — snippet only; not fetched; not cited.
- **Roffo 2017 (Inf-LFS)** — the abs page does not mention absorbing chains, contrary to
  the 2026 paper's framing of its lineage; kept in the bib as the lineage's second paper
  with that caveat.

---

## 6. Limits of this sweep

The sweep reached abstracts and registry records for 60 sources and full text for two
(Grinstead–Snell §11.2; Roffo et al. 2026). "Owns" lines for the remaining sources describe
subject matter, not licensed theorems. Paywalled landing pages (SIAM, Springer, Annual
Reviews, Elsevier, Nature, MIT Press, Routledge) were replaced by Crossref registry records;
the title-to-DOI match is verified, the article text is not. Venue sub-fields for five
entries (Bansal CDC 2017; Ames ECC 2019; Alshiekh AAAI 2018; Chen NeurIPS 2021; Khoo Res.
Math. Sci. 2019) rest on the brief or the repository's earlier record and are marked [U].
The Kemeny–Snell entry is verified for the 1960 Van Nostrand edition; the 1976 Springer
reprint's ISBN is index-only. The search engine is a single US index; the eight NOT FOUND
claims are bounded by the 25 logged queries and their witnesses, not by exhaustiveness.
Nothing here was RUN; the DERIVED identities in §1.A and §4.1 are algebra written out, not
executed (the brief forbids new code; the coordinator's `shape_identities.py` could be
extended by someone licensed to write code to check the `V = 1 ⇒ Δz ≡ 0` planted identity).
