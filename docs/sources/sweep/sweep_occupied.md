# SWEEP — the record's prior-art table, re-verified, and the shape's occupancy

SATURN-S7, 2026-09-03. Companion file: `bib_occupied.bib` (same directory).

**Marks.** `[V]` = the arXiv abs page, DOI landing page, Crossref record, zbMATH record,
DBLP record or proceedings page was fetched this session and the title matched.
`[U]` = reached only through a search index or the repository's own record. Nothing is
marked `[V]` from memory. Equation-level statements below are `READ` from the
repository's prior-art files unless the line says the html was fetched this session.

**Scope.** Every identifier the record already cites (PRIOR_ART.md, RESEARCH.md,
V13_TIER6 / X25 / X27 prior-art files, MATHEMATICS.md, workdonenewseal.md §9) was
re-fetched; then the sweep widened with twelve recorded searches on the components of
the shape in `BRIEF.md` §1. Absence is written `NOT FOUND` with the query that failed,
never "novel".

---

## 0. Headline findings (for the paper's prior-art and negatives sections)

1. **SURPRISE — the read operator is published.** Fagnou, Caillon, Delattre, Allauzen,
   *Chain and Causal Attention for Efficient Entity Tracking*, EMNLP 2024,
   arXiv:2410.05565 `[V]` (abs + html fetched). Eq. (5) is
   `Y = (1−γ)·A(I−γA)⁻¹V`, `γ ∈ [0,1)`, with `A` the causal softmax attention matrix;
   §4.1 writes the path sum `A + A² + …` with closed form `A(I−A)⁻¹`; Eq. (7) solves it
   as the triangular system `(I−γA)Y = (1−γ)AV` and the paper says "triangular solver".
   Its motivation is the entity-tracking depth bound `log₂(n+1)`. **This is the shape's
   `O = P(I−γP)⁻¹V` on the β = 1 corner, with `I1` (γ = 0 gives `AV`) and `I5` (the
   triangular solve), in print eleven months before the brief.** The record has no
   mention of the identifier (grep over every `*.md`, 2026-09-03: 0 hits). The words
   *resolvent, Neumann, absorbing, committor, intervention, certificate, error bound* do
   not appear in it (html fetch, 2026-09-03).
2. **The Neumann-series / absorbing-chain framing of attention is also published.**
   Roffo, Abdelkawy, Lavie, Palmer, arXiv:2603.00175 `[V]` (abs + html fetched):
   Eq. (7) `Č = (I−γA)⁻¹ − I`, `γ ∈ (0, 1/ρ(A))`; §3.3 augments the token graph with a
   **single** absorbing state and reads the kernel as the fundamental matrix of an
   absorbing chain. `A` is ReLU-Frobenius-normalised, neither row-stochastic nor causal.
   The record knows this paper only inside the dead signed programme
   (`DONE_ARCHIVE_ROUND1.md:1317`).
3. **All 76 seed identifiers resolve and their titles match the record's description.**
   Three ids whose titles sit outside the attention domain were checked against their
   record context and are consistent: 2606.18303 (Miyagawa, shock-wave/SGD — used for
   the M8 divergence retrospective, `CEQ_V20_R15_CONTRACT.md:207`), 2604.25655 (Bai et
   al., PINN change-point — used as the learned-model form of Basseville–Nikiforov,
   `V15_X35C_PRIOR_ART.md:641`), 2603.02289 (Kim–Lee, topological causal effects —
   `results/r9_maths_survey.md:314`). No stale or wrong id was found among the seeds.
4. **One hazard the assembler must not inherit:** arXiv:2501.00663 is *Titans* (Behrouz,
   Zhong, Mirrokni), not the test-time-regression paper; the latter is arXiv:2501.12352
   (Wang, Shi, Fox) `[V]`. Both fetched this session; only 2501.12352 is in the bib.
5. **Two items remain `[U]`:** Butkus & Kriegeskorte, NeurIPS 2025 (OpenReview served a
   verification screen), and Grinstead & Snell 1997 (ISBN from the search index; the
   record's own V13_TIER6 file read Theorem 11.6 from the text in an earlier session).

---

## 1. Per-source sweep — the shape's components

Column "leaves open" is stated against the shape in `BRIEF.md` §1: causal row-stochastic
`P` from the three-corner family; `γ ∈ [0,1)`; `K` absorbing constraint sets `𝒜_k`;
`z = (I−γP)⁻¹V`; `O = Pz`; committors `q^{(k)}`; `do(a)` displacement; safest move
`argmin_a max_k q^{(k)}(do a)`; Neumann certificate with printed δ; triangular solve.

### 1.1 The resolvent read `O = P(I−γP)⁻¹V`

| id | mark | owns (in this file's words) | leaves open | found by |
|---|---|---|---|---|
| arXiv:2410.05565 Fagnou et al. 2024 | `[V]` | `Y=(1−γ)A(I−γA)⁻¹V` on causal softmax `A`, `γ∈[0,1)`, solved as a lower-triangular system; depth bound `log₂(n+1)` for `n` state changes | no absorbing sets, no committor, no intervention, no truncation certificate, no machine-checked parity theorem, `(1−γ)` prefactor differs from the shape's bare `P(I−γP)⁻¹` | search S7 (`"triangular solve" OR "linear system" causal … attention`) |
| arXiv:2603.00175 Roffo et al. 2026 | `[V]` | attention as a discounted Neumann series `(I−γA)⁻¹−I`; absorbing-chain reading with one terminal state; Katz/PageRank centrality link | non-causal, non-stochastic `A`; one absorbing state, not `K` labelled sets; no committor; no certificate; replaces softmax rather than containing it | search S3 (`attention mechanism resolvent … Neumann series`) and S4 |
| arXiv:1810.05997 APPNP 2019 | `[V]` | `Z = α(I−(1−α)Â)⁻¹H`, the personalised-PageRank resolvent on a normalised graph operator | fixed graph, not content-adaptive; no boundary sets; no decision | seed (`REQUIREMENTS.md:77`, `DONE_ARCHIVE_ROUND1.md:677`) |
| arXiv:1911.05485 GDC 2019 | `[V]` | generalised graph diffusion (PPR / heat kernel) as a preprocessing operator | same as APPNP | seed (`DONE_ARCHIVE_ROUND1.md:677`) |
| arXiv:2512.14619 ParaFormer 2025 | `[V]` | generalised-PageRank graph transformer (hop coefficients over a non-negative base) | graph substrate; no causal mask; no boundary sets | seed (RESEARCH.md table) |
| Dayan 1993 | `[V]` Crossref | successor representation `M = (I−γP)⁻¹` for TD learning | a representation for value prediction, not an attention read; no boundary sets | seed (`THEORY.md:234`); search S2 |
| Sutton–Precup–Singh 1999 | `[V]` Crossref | Eq. (7) discounted multi-step transition model over options | option model, not an operator on a context | seed (V13_TIER6 §5.1) |
| arXiv:2607.10677 Lin et al. 2026 | `[V]` | single-head attention as a connection-walk step; generator reduces to a random-walk connection Laplacian | one step, not the resolvent; no γ; no boundaries | search S4 |
| arXiv:2511.23239 Shi–Cao 2025 | `[V]` | a one-layer transformer trained on circle random walks learns "parent selector + one-step transition" | one hop; a learning-dynamics result, not an operator | search S4 |

### 1.2 Corner 3 is a resolvent; triangular solve (I2, I5)

| id | mark | owns | leaves open | found by |
|---|---|---|---|---|
| arXiv:2406.06484 DeltaNet 2024 | `[V]` | chunkwise parallel form of the delta rule via a WY/UT representation, i.e. a strictly-lower-triangular solve | signed, key-key, no row normaliser (the record's own table); no boundaries | seed (RESEARCH.md) |
| arXiv:2606.06034 Zhang et al. 2026 | `[V]` | inverse of strictly-lower-triangular chunk matrices as a finite Neumann sum; MatMul-only approximation | quantisation engineering; no γ, no boundary sets | seed (`DONE_ARCHIVE_ROUND1.md:1319`) |
| arXiv:2506.05233 MesaNet 2026 | `[V]` | a layer whose output is the exact minimiser of an in-context regression (a linear solve per position, CG) | least-squares solve, not a Markov resolvent; no γ; no boundaries | search S7 |
| arXiv:2501.12352 Wang–Shi–Fox 2025 | `[V]` | sequence models as test-time regression (associative memory as a solve) | same | search S7 (id corrected from 2501.00663) |
| arXiv:2502.01397 Trifonov et al. 2026 | `[V]` | message-passing GNNs cannot approximate sparse triangular factorisations | a negative for GNNs; the shape *solves* rather than approximates | seed (PRIOR_ART §4.2) |

### 1.3 Committor = resolvent read with boundaries (I3); absorbing sets

| id | mark | owns | leaves open | found by |
|---|---|---|---|---|
| Grinstead–Snell 1997, Thm 11.6 | `[U]` (ISBN from index; text READ by the record) | `B = NR`, `N = (I−Q)⁻¹` | textbook; no learning use | seed (V13_TIER6 §3.1) |
| E–Vanden-Eijnden 2006 | `[V]` Crossref | transition path theory; committor as the central object | continuous diffusions | seed (V13_TIER6 Owed #3) |
| Metzner–Schütte–Vanden-Eijnden 2009 | `[V]` Crossref | TPT for Markov jump processes: discrete committor `Lq = 0` with boundary values | not an attention layer; no `K` sets as *constraints on a decision* | seed (V13_TIER6 Owed #3) |
| Zhu–Ghahramani–Lafferty 2003 | `[V]` DBLP | node classification as the Dirichlet problem: labelled nodes clamped, harmonic extension elsewhere — absorbing sets as boundary conditions of a propagation | fixed graph, labels as boundaries, no attention, no decision | search S9 |
| Wu et al. 2012 | `[V]` NeurIPS page | partially absorbing random walks with per-node absorption rate `α_i`; absorption probabilities implement the cluster assumption | graph learning; no attention | search S10 |
| arXiv:2306.16976 Begga et al. 2023 | `[V]` | Dirichlet formulation with absorbing random walks inside a GNN | GNN, not attention | seed (PRIOR_ART §4.2) |
| arXiv:2205.01358 Azad 2022 | `[V]` | harmonic extension as node classification | same | seed |
| arXiv:2206.02911, 2206.14092 | `[V]` | inverse BVP on graphs; solution operator of BVPs by GNN | same | seed |
| arXiv:2206.11941 Velingker et al. 2023 | `[V]` | GNNs cannot compute single-source effective resistance (Thm C.1 per the record) | negative for GNNs | seed |
| arXiv:1802.10275, 1906.06285, 2507.17700 | `[V]` | committor regression from molecular configurations | not from a fundamental matrix; no context operator | seed (PRIOR_ART §4.2) |
| arXiv:1710.06012 VAMPnets; Prinz 2011 | `[V]` | metastable-state membership head; CK validation; implied timescales | the "phase" component of capability (c); no attention | seed (V13_TIER6 §1) |

**NOT FOUND — `K ≥ 2` absorbing constraint sets inside an attention operator yielding one
committor per set.** Queries: S4 (`absorbing random walk attention transformer boundary
conditions "absorbing" states attention layer`), S5 (`"committor" transformer attention
language model OR sequence model`). S5 returned no paper containing "committor" in a
transformer context; S4's only absorbing-state hit is InfSA's single terminal state.

### 1.4 The interventional channel `Δz = z(do a) − z`

| id | mark | owns | leaves open | found by |
|---|---|---|---|---|
| Pearl 2009 | `[V]` Crossref | `do(·)` as graph surgery; linear SCM `X = (I−B)⁻¹Z`, intervention clamps a row | structural, not an attention operator; the record's `r9_maths_survey.md:313` already uses this form | seed (r9 survey) |
| Sherman–Morrison 1950 | `[V]` Crossref | rank-one update of an inverse — the algebra of "re-solve after one row changes" | pure linear algebra | added for attribution of the displacement computation |
| Butkus–Kriegeskorte NeurIPS 2025 | `[U]` | a next-token transformer learns linear-Gaussian SCMs and answers counterfactual queries; a "mental SCM" is decodable from the residual stream | learned/emergent, not an operator with a displacement read | search S6 |
| arXiv:2412.07446 Rohekar et al. 2025 | `[V]` | a causal interpretation of GPT attention and a causal world model derived from it | interpretive; no resolvent; no constraint sets | search S6 |
| arXiv:2603.02289 Kim–Lee 2026 | `[V]` | causal effects defined on persistence-diagram summaries | a label class, not an operator | seed (r9 survey) |

**NOT FOUND — the displacement of an attention resolvent under a context intervention as a
trained output.** Query S6 (`intervention do-operator attention counterfactual "next
state" prediction transformer world model causal consequence`).

### 1.5 The safest move `argmin_a max_k q^{(k)}(do a)`

| id | mark | owns | leaves open | found by |
|---|---|---|---|---|
| Summers–Lygeros 2010 | `[V]` Crossref | the stochastic reach-avoid decision problem: maximise the probability of reaching a target while avoiding an unsafe set | control setting; no attention | search S8 |
| arXiv:1411.5925 Kariotoglou et al. 2014 | `[V]` | reach-avoid for MDPs as an infinite-dimensional LP | same | search S8 |
| arXiv:2302.13152 Misra et al. 2023 | `[V]` | reach a target while avoiding unsafe set(s) with probabilistic guarantees; multi-objective; Bellman optimality can fail in multichain CMDPs | same; the multichain warning is a hazard for the shape's own decision rule | search S8 |

**NOT FOUND — a min-max over per-constraint absorption probabilities computed by an attention
layer from candidate moves in the context.** Query S8 (`safest action multiple constraints
"absorption probability" OR "reach-avoid" Markov chain minimax constrained MDP`).

### 1.6 Why softmax cannot occupy the ground (the four obstructions)

| id | mark | owns | licenses exactly | found by |
|---|---|---|---|---|
| arXiv:2402.09268 Sanford–Hsu–Telgarsky 2024 | `[V]` | Thm 4.2 depth `⌊log₂k⌋+2` for `k`-hop; Cor. 4.3 `Ω(log k)` **conditional** on 1-vs-2-cycle | the depth law, conditionally | seed |
| arXiv:2405.18512 Sanford et al. NeurIPS 2024 | `[V]` | log depth necessary and sufficient for connectivity; nine-task hierarchy | reachability needs log depth | seed (OpenReview AfzbDw6DSp) |
| arXiv:2410.05565 Fagnou et al. 2024 | `[V]` | `log₂(n+1)` layers for entity tracking with `n` state changes | the same law, for state tracking | S7 |
| arXiv:2402.08164 Peng–Narayanan–Papadimitriou 2024 | `[V]` | a transformer layer cannot compose functions with large domains (communication complexity) | obstruction 3, title and abstract confirmed | seed (brief "VERIFY") |
| arXiv:2410.01537 Marion et al. 2025; arXiv:2509.21936 Duranthon et al. 2026 | `[V]` | single-location regression: attention attains Bayes risk asymptotically (`d→∞`, `L=o(d)`); statistical advantage of softmax | D-1's owner; the record's §17.5 caveat on `L/d = 4.00` stands | seed |
| arXiv:2008.02217 Ramsauer 2021; 2110.11773 Sinkformer 2022; 2311.12424 looped 2024; 2306.13596 Tarzanagh 2023; 1909.01377 DEQ 2019; 2402.06445 / 2410.15059 DEAR | `[V]` | one-step Hopfield convergence; doubly-stochastic fixed point; looped weight-tying; max-margin token selection; equilibrium layers; equilibrium algorithmic reasoners | the "next state toward equilibrium" ground on the representation side | seed |

### 1.7 The positional-encoding lineage (PRIOR_ART.md §5) — re-verified, unchanged

2505.16381 PaTH (NeurIPS 2025), 2406.10322 LieRE (ICML 2025 per abs), 2512.07805 GRAPE
(ICLR 2026), 2312.16045 APE (NeurIPS 2024 spotlight), 2104.09864 RoFormer, 2603.16123
Sargsyan (v2 2026-05-29; the record's `[U]` appendix question is still open), 2001.00610
DeBenedetto–Chiang (ICML 2020), 2510.08648 Wilson loops — all `[V]`, titles match. The
record's verdict (VGPE occupied by PaTH §2.1) is not disturbed.

### 1.8 The signed programme (RESEARCH.md table) — re-verified, unchanged

2206.08898 SimA, 2606.04833 SDA (AAAI 2026 AI4TS workshop), 2406.06484 DeltaNet,
2512.14619 ParaFormer (WSDM 2026), 2411.07176 Cog, 2310.11025 SignGT, 2307.08621 RetNet —
all `[V]`. Equation-level claims remain `READ` from the record, as RESEARCH.md itself says.

### 1.9 SSM / RNN gates (workdonenewseal §9) — re-verified

2206.11893 S4D, 2202.09729 SaShiMi, 2605.08966 VORT, 2303.06349 LRU, 1511.06464 uRNN
(modReLU's source) — `[V]`. Basseville–Nikiforov 1993 `[V]` via the author-hosted page
(ISBN 0-13-126780-9). 2604.25655 `[V]`, title as the record states.

### 1.10 The Hankel programme (PRIOR_ART.md §1–3) — re-verified

Fliess 1974 `[V]` zbMATH (Zbl 0315.94051; primary text still NOT REACHED); Carlyle–Paz 1971
`[V]` Crossref; Berstel–Reutenauer `[V]` Crossref (online 2010, print 2011); Sakarovitch
2009 `[V]`; Balle–Mohri 2015 `[V]`; van Heerdt et al. 1911.04404 `[V]`; Labai–Makowsky
1512.02430 `[V]`; Yannakakis 1991 `[V]`; Cohen–Rothblum 1993 `[V]`; Vavasis 2009 `[V]`
(SIAM + arXiv 0708.4149); Arora–Ge–Kannan–Moitra 1111.0952 `[V]`; Kwan–Sauermann–Zhao
2006.08836 `[V]` (title *Extension complexity of low-dimensional polytopes*; the record's
attribution of Thm 3.1 `rank₊/rank = n^{1−o(1)}` is `READ`, not re-read); FMPTdW 1111.0837
`[V]`; FKPT 1111.0444 `[V]`; Rothvoß 1311.2369 `[V]`; Liu et al. 2210.10749 `[V]` (no venue
on the abs page, as the record notes).

### 1.11 Over-squashing / spectral-gap diagnostics (PRIOR_ART.md §4.3) — re-verified

2111.14522, 2302.02941, 2302.06835, 2210.11790, 2206.08164, 2503.00547, 2607.21607 — all
`[V]`. 2607.21607's abs page reads "Submitted on 16 May 2026" against a July identifier;
recorded as the record already records it.

---

## 2. Searches run (recorded verbatim; hits counted against the shape)

| # | query | relevant hits |
|---|---|---|
| S1 | `Sanford Fatemi Hall Tsitsulin "Understanding Transformer Reasoning Capabilities via Graph Algorithms" arXiv` | arXiv:2405.18512 (id recovered) |
| S2 | `Dayan 1993 "successor representation" Neural Computation "Improving generalization for temporal difference learning"` | Neural Computation 5(4):613–624 |
| S3 | `attention mechanism resolvent "(I - \gamma P)^{-1}" OR "Neumann series" OR "successor representation" transformer multi-hop propagation arXiv 2025 2026` | arXiv:2603.00175 InfSA; arXiv:2604.15069 (doubly-stochastic GNN, truncated Neumann; not fetched) |
| S4 | `absorbing random walk attention transformer boundary conditions "absorbing" states attention layer arXiv` | 2603.00175; 2511.23239; 2607.10677 |
| S5 | `"committor" transformer attention language model OR sequence model arXiv` | **0** hits containing "committor" in a transformer context |
| S6 | `intervention do-operator attention counterfactual "next state" prediction transformer world model causal consequence arXiv 2025 2026` | Butkus–Kriegeskorte NeurIPS 2025 `[U]`; 2412.07446 |
| S7 | `attention layer "triangular solve" OR "linear system" causal "test-time regression" OR "mesa layer" transformer arXiv 2024 2025` | **2410.05565 (the surprise)**; 2506.05233; 2510.01450 (local linear attention, not fetched) |
| S8 | `safest action multiple constraints "absorption probability" OR "reach-avoid" Markov chain minimax constrained MDP probabilistic safety arXiv` | 2302.13152; 2403.15928 (not fetched); 1411.5925; Summers–Lygeros 2010 |
| S9 | `Zhu Ghahramani Lafferty 2003 "semi-supervised learning using Gaussian fields and harmonic functions" ICML` | DBLP conf/icml/ZhuGL03 |
| S10 | `"partially absorbing random walks" Wu Li Chang NeurIPS 2012 arXiv` | NeurIPS 2012 proceedings page |
| S11 | `Butkus "Causal Discovery and Inference through Next-Token Prediction" arXiv` | no arXiv id; OpenReview MMYTA3v66p (blocked) |
| S12 | `"successor representation" transformer attention in-context learning arXiv 2024 2025 2026` | **0** hits using the SR as an attention operator |

Twelve searches beyond the seeds; the brief asked for at least six.

---

## 3. Verdict

**OCCUPIED, with owners named.**
(a) *The read operator itself*, `O = P(I−γP)⁻¹V` on causal softmax `P`, `γ ∈ [0,1)`, with
softmax recovered at `γ = 0` and computed by a triangular solve — **Fagnou, Caillon,
Delattre, Allauzen, EMNLP 2024, arXiv:2410.05565** (Eqs. 5, 7). Identities I1 and I5 are
theirs as equations; the record holds only the machine-checked containment
(`three_corners_containment`) and the bitwise `torch.equal` run.
(b) *Attention as a discounted Neumann series read through an absorbing chain* —
**Roffo et al. 2026, arXiv:2603.00175** (Eq. 7, §3.3), on a non-causal non-stochastic
operator with one terminal state.
(c) *The resolvent of a stochastic propagation operator as a layer* — APPNP
(Gasteiger–Bojchevski–Günnemann 2019), GDC 2019, ParaFormer 2025; as a representation —
Dayan 1993; as a discounted multi-step model — Sutton–Precup–Singh 1999 Eq. (7).
(d) *Corner 3 is the resolvent of a strictly-lower-triangular chain, solved by forward
substitution* (I2) — DeltaNet 2024 (WY/UT form), Zhang et al. 2026 (explicit finite
Neumann sum), MesaNet 2026 and test-time regression 2025 (per-position linear solves).
(e) *Committor = `(I−Q)⁻¹R1_B`* (I3) — Grinstead–Snell Thm 11.6 `[U]`; Metzner–Schütte–
Vanden-Eijnden 2009 (discrete TPT); E–Vanden-Eijnden 2006.
(f) *Absorbing / labelled sets as boundary conditions of a propagation operator on a
graph* — Zhu–Ghahramani–Lafferty 2003; Wu et al. 2012; Begga et al. 2023; Azad 2022.
(g) *The safest move as a min–max over reach-avoid probabilities* — Summers–Lygeros 2010;
Kariotoglou et al. 2014; Misra et al. 2023 (which also warns that Bellman optimality can
fail on multichain CMDPs — a hazard the paper's decision rule must address).
(h) *Intervention as row surgery on a linear resolvent* — Pearl 2009 (linear SCM); the
rank-one re-solve — Sherman–Morrison 1950.
(i) *The four obstructions* — Sanford–Hsu–Telgarsky 2024 (conditional), Sanford et al.
2024, Fagnou et al. 2024 (`log₂(n+1)`), Peng–Narayanan–Papadimitriou 2024 (composition),
Marion et al. 2025 and Duranthon et al. 2026 (D-1's owner).

**NEAR-MISS.** (j) The Neumann truncation certificate `γ^{K+1}/(1−γ)` for row-stochastic
`P` (I4) is textbook; neither 2410.05565 nor 2603.00175 prints any truncation bound
(html fetches, 2026-09-03), so the *discipline* of shipping a printed δ beside every
truncated read is open, not the inequality. (k) InfSA's single absorbing state is one set
away from `K` constraint sets. (l) Butkus–Kriegeskorte 2025 and Rohekar et al. 2025 place
causal structure and counterfactuals inside next-token transformers, but as learned or
interpretive structure, not as a resolvent displacement read.

**NOT FOUND** (queries in §2). (m) `K ≥ 2` absorbing constraint sets inside an attention
operator with one committor per set (S4, S5). (n) The displacement `z(do a) − z` of an
attention resolvent as a trained output (S6). (o) `argmin_a max_k q^{(k)}(do a)` computed
by an attention layer over candidate moves placed in the context (S8). (p) "committor" in
any transformer or language-model paper (S5: 0 hits). (q) the successor representation
used as an attention operator (S12: 0 hits). (r) a machine-checked proof that the resolvent
family contains softmax bitwise at `γ = 0` and the path product at corner 3 — the record's
Lean declarations are the only instance reached, and this is the narrow, verifiable form
of the delta.

**The delta, stated narrowly.** What no source reached combines is: the *record's* causal
row-stochastic corner family (Lean-checked containment) as `P`; `K` absorbing constraint
sets as boundary conditions giving per-constraint committors; an interventional
displacement read; a min–max decision over those committors; and a printed Neumann δ
beside every truncated read. Each part has an owner (a–i above); the composition does not,
on this sweep. The paper must cite 2410.05565 in the first paragraph that writes
`P(I−γP)⁻¹V`, and 2603.00175 in the first paragraph that says "absorbing".

---

## 4. Design against the MISTAKES.md taxonomy (what this sweep's proposals guard)

- **V-7 (a search structurally incapable of finding anything, read as absence):** every
  NOT FOUND above carries the verbatim query and its hit count; S5 and S12 report 0 hits
  and are the only absences claimed. The surprise (2410.05565) was found by S7, a query
  built from the shape's *mechanism* (triangular solve) rather than its *name* — the paper's
  prior-art section should keep one mechanism-keyed search per component for that reason.
- **P-3 (a stale claim never retracted) / P-6 (line-reference drift):** all 76 seed ids were
  re-fetched rather than copied; no stale id found; the 2501.00663/2501.12352 confusion is
  recorded so it cannot enter the bibliography.
- **P-10 (a source's intro cited as its theorem):** the two occupants that decide the delta
  (2410.05565, 2603.00175) were read at equation level (html fetch) and the equation numbers
  are given; everything else equation-level is marked `READ` from the record.
- **D-1 (racing a proven optimum):** Marion 2025 / Duranthon 2026 are cited as the owners of
  the single-location task, so the label class the paper proposes is chosen *away* from it.
- **V-24 (empty rejection region):** for the prior-art table itself, the planted negative is
  the record's own dead signed programme (§1.8) — a lineage the table must show as
  *occupied and refuted*, proving the table can return a red.
- **L-CERT:** the NEAR-MISS (j) is the justification for the certificate column; the two
  occupants print none.

---

## 5. Counts

- Sources in `bib_occupied.bib`: 95 entries; `[V]` 93, `[U]` 2 (Grinstead–Snell 1997;
  Butkus–Kriegeskorte 2025).
- Seed identifiers re-verified: 76 (arXiv 60; Crossref/zbMATH/DBLP/proceedings/author-page
  16); mismatches: 0.
- New occupants found by widening: 2410.05565, 2603.00175, 2607.10677, 2511.23239,
  2506.05233, 2501.12352, 2412.07446, 2302.13152, 1411.5925, Summers–Lygeros 2010,
  Zhu–Ghahramani–Lafferty 2003, Wu et al. 2012, E–Vanden-Eijnden 2006, Metzner et al. 2009,
  Sherman–Morrison 1950, Pearl 2009, Dayan 1993 (Crossref), Butkus–Kriegeskorte 2025 `[U]`.
- Searches recorded: 12.
