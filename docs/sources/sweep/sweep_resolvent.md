# SWEEP — resolvent / propagation / equilibrium attention (JUPITER-S1, 2026-09-03)

Companion BibTeX: `$SCRATCH/sweep/bib_resolvent.bib` (48 entries, every one resolved this session;
the `note` field of each entry records the resolution route and marks any field supplied from
memory as `[U]`). Repository root `<repo root>`, HEAD `207e7b9`.

Evidence classes used below (BRIEF.md §5.3): `RUN` (executed this session), `READ path:line`,
`CITED [V]` (abs/DOI/ISBN/DBLP record fetched this session, title matched), `CITED [U]` (search
index, secondary paper, or memory only), `DERIVED` (steps written out). Every number carries one.

## 0. The question, and the rubric fixed before any paper was scored

The question the coordinator set: *does anyone put the resolvent of a content-dependent, causal,
row-stochastic attention matrix, with absorbing boundary rows, as the attention read, containing
softmax at γ = 0?*

To keep the answer from being refitted to whatever the search returned (MISTAKES.md `M-2`, a
threshold refitted to the data it judges; `V-26`, a marginal assertion standing in for a joint
claim), the shape of BRIEF.md §1 was decomposed into ten boxes **before** the first fetch, and each
source is scored on all ten. A component is OCCUPIED only if one named source ticks the box on a
content-dependent attention operator; NEAR-MISS if the box is ticked on a fixed graph or a
non-attention chain, or ticked with one of (a)–(c) missing; NOT FOUND otherwise.

| box | component of the shape (BRIEF.md §1) |
|---|---|
| (a) | the propagated operator `P` is **content-dependent** (built from `QKᵀ` of the input) |
| (b) | `P` is **causally masked** (lower triangular) |
| (c) | `P` is **row-stochastic softmax** (the β = 1 corner) |
| (d) | the read uses the **exact resolvent** `(I − γP)^{-1}` (a solve), not a truncated hop sum |
| (e) | **absorbing rows**: constraint sets `𝒜_k` whose rows of `P` are identity — boundary conditions inside the operator |
| (f) | **parity at γ = 0**: the read is softmax attention bitwise |
| (g) | the **interventional channel** `do(a)`: re-solve, read the displacement `Δz` |
| (h) | the **safest-move decision** `argmin_a max_k q^{(k)}(do a)` from committors into several sets |
| (i) | a **printed truncation certificate** `γ^{K+1}/(1−γ)` beside any Neumann approximation (I4) |
| (j) | **learnable γ** |

What the record already knew before this sweep (so the delta is stated against it, not against a
blank page — `P-3`, a stale claim never retracted, and `M-10`, filed without grepping for what
already existed):

- `THEORY.md:47-48` (READ): the solve `z = (I − A)^{-1} b` and the successor representation
  `M = (I − γP)^{-1}` are "the same algebraic object"; `THEORY.md:234` lists Dayan 1993 as
  unverified. **Now [V]** (Crossref, this session).
- `THEORY.md:225-235` (READ): Ramsauer 2020, Bai–Kolter–Koltun 2019 listed as unverified. **Now [V].**
- `RESEARCH.md:48` (READ): ParaFormer 2512.14619 recorded as "Nearest published multi-hop
  construction". `DONE_ARCHIVE_ROUND1.md:1321-1324` (READ): its `γ_k` is a scalar per hop, `K = 10`.
- `DONE_ARCHIVE_ROUND1.md:677-681` (READ): APPNP, GDC, MAGNA "require a non-negative matrix";
  DeltaNet named "the near miss — signed, strictly triangular, exactly inverted".
- `DONE_ARCHIVE_ROUND1.md:1317-1318` (READ): InfSA 2603.00175 recorded as a content-adaptive
  Neumann series with `A = ReLU(QKᵀ)` normalised, non-negative. The record read it as a *signed*
  near-miss and did not read its absorbing-chain paragraph; this sweep does (§2.3).
- `PRIOR_ART.md:567-713` (READ): the round-8 sweep already owns DEAR (2402.06445, 2410.15059),
  the committor-regression papers (Khoo–Lu–Ying, Li–Lin–Ren), Diffusion-Jump GNNs (absorbing
  random walks), Azad 2022 (node classification as a Dirichlet problem), Garrousian–Nouranizadeh
  (Dirichlet/Neumann boundary conditions on graphs). Those are not re-argued here; they are
  cross-referenced.
- **Absent from every `.md` in the tree** (RUN, `grep -rlE --include='*.md'`, 2026-09-03, `lake/`
  excluded): `ChaCAL`, `2410.05565`, `Fagnou`, `2605.22476`, `Diffuser`, `2210.11794`, `Piray`,
  `Todorov`, `Katz`, `Kemeny`, `2507.17657`, `EIGNN`, `GRAND`, `Zhu.*Ghahramani`, `Woodbury` — each
  `0 files`. The record has never seen the paper that occupies its read (§2.1).

## 1. The seeds (thirteen), scored

Format per source: **id — status — what it owns — what it leaves open relative to the shape —
query/route that found it.**

### 1.1 `gasteiger-2019-appnp` — arXiv:1810.05997, ICLR 2019 — [V]
Owns: propagation of a *fixed* prediction by personalised PageRank on the *graph adjacency*,
`Z = α(I − (1−α)Â)^{-1} H` in the limit, run as `K` power-iteration steps (DERIVED from the
abstract's "personalized PageRank" + memory of the body [U] for the closed form).
Leaves open: (a) — `Â` is the adjacency, not content; (b) no order; (e) no boundary rows;
(f) α → 1 gives the MLP, not softmax; (g)(h)(i) absent. Boxes ticked: (c) row-normalised
adjacency, (d) as a limit.
Route: seed; abs page fetched. First author's arXiv listing is "Gasteiger" (formerly Klicpera).

### 1.2 `yuan-2025-paraformer` — arXiv:2512.14619 — [V] (abs + HTML)
Owns: `Z = Σ_{k=0}^{K} γ_k (Softmax(QKᵀ))^k V` with learnable, sign-free-to-negative scalar
`γ_k`, on graphs, no causal mask; the generalized-PageRank family of Chien et al. put on a
content-dependent softmax base.
Leaves open: (b) causal; (d) exact solve — the HTML states the sum is truncated to a hyper-parameter
`K` in practice; (e); (f) — at `K = 0` it is `γ_0 V`, at `γ_k = γ^k`, `K → ∞` it is the resolvent;
so the shape's read is the geometric, infinite-`K` corner of ParaFormer's family (DERIVED);
(g)(h)(i) absent. Ticks (a)(c)(j).
Numbers (HTML, [V]): 3.4 % average relative gain over message-passing GNNs; 1.9 % over other graph
transformers; 7 node and 2 graph datasets.
Route: seed; already `RESEARCH.md:48`.

### 1.3 `chamberlain-2021-grand` — arXiv:2106.10934 — [V] (abs + ar5iv HTML)
Owns: `∂X/∂t = (A(X) − I)X` with `A` the softmax attention (the body calls it right-stochastic),
and the **implicit Euler step `(I − τĀ(x^{(k)})) x^{(k+1)} = x^{(k)}`, `Ā = A − I`** — one linear solve
against the content-dependent attention operator per step; GRAND-l freezes `A`, GRAND-nl
recomputes it.
DERIVED: `(I − τ(A − I))^{-1} = (1+τ)^{-1} (I − γ_τ A)^{-1}` with `γ_τ = τ/(1+τ)`. So one implicit
GRAND-l step **is** the shape's `z` up to the scalar `1/(1+τ)`, on an undirected, non-causal graph.
Further, `∫_0^∞ e^{-λt} e^{t(A−I)} dt = (λI − (A − I))^{-1} = (1+λ)^{-1}(I − (1+λ)^{-1}A)^{-1}`: the
shape's resolvent at `γ = 1/(1+λ)` is the Laplace transform of GRAND's frozen flow at rate
`λ = (1−γ)/γ` — the read is the discounted time-integral of the diffusion, not its endpoint.
Leaves open: (b) causal; (e) boundary rows (the body assumes undirected graphs, no self-edges,
no Dirichlet discussion); (f) `τ → 0` gives the identity, not softmax; (g)(h)(i)(j) absent.
Route: seed.

### 1.4 `gu-2020-ignn` — arXiv:2009.06211 — [V]
Owns: the fixed-point layer `X = φ(WXA + bΩ(U))` with Perron–Frobenius well-posedness conditions
(abstract). Nonlinear, fixed `A`.
Leaves open: (a)(b)(e)(f)(g)(h)(i); the equilibrium is of a nonlinear map, not a linear resolvent.
Route: seed.

### 1.5 `bai-2019-deq` — arXiv:1909.01377, NeurIPS 2019 — [V]
Owns: the equilibrium `z* = f_θ(z*; x)` of a weight-tied block found by root-finding, implicit
differentiation through `(I − J_f)^{-1}`, constant memory; instantiated on a Transformer-XL block
with input injection (search result, torchdeq zoo, [U] for the injection detail).
Leaves open: the equilibrium is of the whole block (attention + MLP), nonlinear; the shape's `z` is
the *linear* equilibrium `z = γPz + V` of one attention operator, solved exactly, not iterated.
DEQ owns the *training* mechanism the shape would use if `P` were made to depend on `z`; the
shape as written keeps `P` a function of the input only, so the solve is one triangular pass (I5)
and no root-finder is needed. `THEORY.md:203-206` risk 1 ("nested fixed points") is exactly this
distinction and should be carried into the paper as a design choice, not a risk.
Route: seed.

### 1.6 `georgiev-2024-dear` / `georgiev-2024-dearneurips` — arXiv:2402.06445, 2410.15059 — [V]
Own: algorithmic-reasoning targets treated as equilibria of a GNN, no ground-truth step count
needed, evaluated on CLRS-30 tasks. Already `PRIOR_ART.md:598-602`.
Leave open: attention (they are message passing); (b)(e)(f)(g)(h)(i).
Route: seed.

### 1.7 `fan-2020-feedback` — arXiv:2002.09402 — [V]
Owns: sequential feedback memory — every past position's top-layer state is visible to every
layer of the next position. This is depth-in-time recurrence, not a resolvent; it buys unbounded
composition at the price of sequential decoding.
Leaves open: everything in (d)–(j); the shape gets the same composition in one parallel solve.
Route: seed.

### 1.8 `yang-2023-looped` — arXiv:2311.12424 — [V]
Owns: looped (weight-tied, input-injected) transformers emulating iterative algorithms with under
10 % of the parameters (abstract). Iteration, not a solve; no fixed-point claim in the abstract.
Leaves open: (d)–(j).
Route: seed.

### 1.9 `dehghani-2018-universal` — arXiv:1807.03819, ICLR 2019 — [V]
Owns: recurrence in depth with per-position ACT halting. Same class as 1.8.
Route: seed.

### 1.10 `sander-2021-sinkformer` — arXiv:2110.11773, AISTATS — [V]
Owns: doubly-stochastic attention by Sinkhorn; in the infinite-depth/sample limit the network
runs a heat diffusion. Already `MATHEMATICS.md:204`.
Leaves open: it changes the *normaliser* of one step; no resolvent, no boundary rows, no causal
mask (Sinkhorn needs the full matrix). Relevant only as the diffusion reading whose Laplace
transform is the resolvent (§1.3 DERIVED).
Route: seed.

### 1.11 `ramsauer-2020-hopfield` — arXiv:2008.02217 — [V]
Owns: softmax attention as one update of a modern Hopfield energy, converging in one step.
Already `MATHEMATICS.md:200`.
Leaves open: the Hopfield fixed point is **per query** — each query's retrieval is independent of
the others' — so it is exactly the per-row-independence that BRIEF.md §1 obstruction 1 names.
It is the wrong equilibrium for the shape and the paper should say so in one sentence.
Route: seed.

### 1.12 `dayan-1993-successor` — Neural Computation 5(4):613–624 — [V] (Crossref)
Owns: the successor representation `M = (I − γP)^{-1}` as expected discounted future occupancy,
learned by TD (formula from memory [U]; the record `THEORY.md:47-48` states it the same way).
Leaves open: `P` is the environment's chain, not content; no attention, no boundary rows as
constraint sets (SR is on the ergodic/discounted chain), no read of `V` through it in the
attention sense.
Route: seed; MIT Press landing page returned 403, Crossref record used.

### 1.13 `barreto-2017-successorfeatures` — arXiv:1606.05312, NIPS 2017 — [V]
Owns: successor features `ψ^π = E[Σ γ^t φ_t]` and generalised policy improvement across reward
functions. This is the SR read through a feature map — the closest RL analogue of `(I−γP)^{-1}V`
with `V = Φ`.
Leaves open: as 1.12; and GPI evaluates several *policies*, not several *constraint sets* — the
shape's `max_k q^{(k)}` is over absorbing sets, a different index.
Route: seed.

## 2. Widening — what the twenty-two searches found (recorded in §6)

### 2.1 SURPRISE — `fagnou-2024-chacal` — arXiv:2410.05565, EMNLP 2024 Main — [V] (abs + HTML)
**Owns the read.** The ChaCAL layer is, in its own equation,

    Y = (1 − γ) · A (I − γA)^{-1} V,   A = causal softmax attention,  γ ∈ [0, 1)

solved as the triangular system `(I − γA) Y' = V` — the HTML body states it does not require
computing the inverse explicitly — with `γ = 0.9` fixed (Appendix C: 0.8–1 works), and the body
states that `γ = 0` reverts to standard attention. Motivation in the body: the series
`A + γA² + γ²A³ + …` counts paths of every length in the attention graph.

Scored: (a) ✓ (b) ✓ (c) ✓ (d) ✓ (f) ✓ — **five of the six structural boxes.** (e) ✗ (no absorbing
rows, sinks, Markov chains, boundary conditions anywhere — the body was queried for each term);
(g) ✗ (h) ✗ (i) ✗ (the series is motivation, no truncation bound is printed); (j) ✗ (fixed γ).

DERIVED, the exact relation to BRIEF.md §1: with `A = P`,
`O_shape = P (I − γP)^{-1} V = Y_ChaCAL / (1 − γ)`. **The shape's read is ChaCAL up to the scalar
`1/(1−γ)`, and identical at γ = 0.** Identities I1 (parity at γ = 0) and I5 (triangular solve) of
the brief are therefore re-derivations of ChaCAL's Eq. and its solver; I2 (corner 3 is a resolvent)
is the record's own and is not in ChaCAL.

DERIVED, a consequence the design panel must carry: for row-stochastic `A ≥ 0`,
`(I − γA)^{-1} 1 = Σ_k γ^k A^k 1 = 1/(1−γ) · 1`, so `(1−γ) A (I − γA)^{-1}` is itself
**row-stochastic and entrywise non-negative** — ChaCAL's read is a per-row *convex mixture* of `V`
with multi-hop weights, and so is the shape's read at any γ without boundary rows (up to the
scalar). Hence the influence Jacobian `∂O_i/∂V_j ≥ 0` still holds: **the resolvent alone does not
lift obstruction 4 of BRIEF.md §1** (a third token cannot subtract). Absorbing rows do not make it
signed either — `(I − γP)^{-1}` is a Neumann series of a non-negative matrix for any non-negative
`P` — they *redirect* mass (a walk that hits `a ∈ 𝒜_k` stays at `a`, so the weight on every token
the walk would have reached through `a` goes to zero). The paper should write "redirects", never
"subtracts"; the mechanism is the same family as the record's F0 zero-gate segmentation
(`V16Domain.pathProd_eq_zero_iff`), applied by boundary rows instead of by gates.

Numbers (HTML, [V]): toy entity tracking 100 % with 1 layer against 4–5 standard layers; Boxes
(advanced) 99.1 % vs 97.0 % with 2 layers; language-model perplexity 21.46 vs 20.15 (worse).
The abstract's theoretical constraint: transformers need at least `log₂(n+1)` layers for `n` state
changes — the same depth law the brief cites from Sanford–Hsu–Telgarsky, arrived at independently
(the two must be cited together and the constants compared by whoever holds the depth lineage).

Route: search 2 ("resolvent" attention … "(I − " inverse …) surfaced it through the follow-up
2605.22476; abs and HTML fetched; ACL Anthology id 2024.emnlp-main.731 seen in a search index only.

### 2.2 `zhao-2026-structuredsparse` — arXiv:2605.22476 — [V] (abs + HTML)
Owns: the same operator, named `𝒮_γ(A) := (1−γ)A(I−γA)^{-1}`, evaluated blockwise — exact on
diagonal tiles, cross-block interactions routed through a fixed row down-sampler
`Ã_res = P A_res Pᵀ ∈ ℝ^{k×k}` — in `O(n^{4/3} d)`; causal (the body states `A` is lower
triangular under the mask); 12–29 % wall-clock reduction against the dense operator at accuracy
parity on Boxes; up to 2.4× faster than a compact Transformer at comparable exact match (HTML, [V]).
Leaves open: (e)(g)(h)(i)(j); no truncation bound is printed for the reduced system.
Relevance to the brief's cost law (§6 of the paper): this is the existing kernel path for the
resolvent read; the record's `ceq/multizoom.py` fine-near/coarse-far construction is the same
block idea with a bound — the two must be cited side by side (the multizoom bound is the delta, if
its bound survives the coordinator's check).
Route: search 2.

### 2.3 `roffo-2026-infsa` — arXiv:2603.00175 — [V] (abs + HTML)
Owns: the **Neumann closure** `Č = (I − γÂ)^{-1} − I = Σ_{t≥1}(γÂ)^t` over a content-adaptive
attention, and — verbatim in its abstract — the reading that the Neumann kernel is the fundamental
matrix of an absorbing Markov chain, a token's centrality being its expected number of visits
before absorption. The chain is `M = γÂ` with per-step absorption `R_i = 1 − γΣ_j Â_ij`;
`N = (I − γÂ)^{-1}`; centrality `č(i) = [(I − γÂ)^{-1} e]_i − 1`. γ is learned per head (sigmoid).
Base: `Â = A_+ / (‖A_+‖_F + ε)` with `A_+ = ReLU(QKᵀ)` — **sub-stochastic by Frobenius
normalisation, not row-stochastic softmax**; no causal mask (vision); Linear-InfSA broadcasts a
rank-1 context from the principal eigenvector.
DERIVED: `č = (I − γÂ)^{-1}1 − 1` is Katz's 1953 index on the attention graph, exactly.
Scored: (a) ✓ (d) ✓ (closure; the linear variant approximates) (j) ✓; (b) ✗ (c) ✗ (e) ✗ —
absorption is *probabilistic and uniform per row*, there are no identity rows and no target sets,
so no committor; (f) ✗ — at γ = 0 the closure is the zero matrix, the layer degenerates (HTML);
(g)(h)(i) ✗ (the geometric bound is implicit in ρ(Â) < 1 but never printed as a certificate).
Numbers (HTML, [V]): 84.7 % ImageNet-1K top-1 (+3.2 pp); 231 img/s at 0.87 J/img, "13×" against
standard ViT.
Route: search 3 ("Neumann series" attention …). The record's line
`DONE_ARCHIVE_ROUND1.md:1317` read it as a signed near-miss only.

### 2.4 `wang-2021-magna` — arXiv:2009.14332, IJCAI 2021 — [V] (abs + ar5iv)
Owns: the first personalised PageRank **on a content-dependent attention matrix**:
`𝒜 = Σ_{i≥0} θ_i A^i, θ_i = α(1−α)^i`, `A` the row-softmaxed GAT scores, stated in the body as a
personalised PageRank on the graph with transition matrix `A`; computed by truncated power
iteration `Z^{(k+1)} = (1−α)AZ^{(k)} + αZ^{(0)}`, `K ∈ [3,10]`; a low-pass reading on the Laplacian
spectrum. Up to 5.7 % relative error reduction on Cora/Citeseer/Pubmed (abs, [V]).
Scored: (a) ✓ (c) ✓; (b) ✗ (d) ✗ truncated (e) ✗ (f) ✗ (g)(h)(i)(j) ✗.
Route: search 1 and search 4. Already `DONE_ARCHIVE_ROUND1.md:677` as "non-negative".

### 2.5 `feng-2022-diffuser` — arXiv:2210.11794 — [V] (abs + HTML)
Owns: MAGNA's construction on **sequences**: `𝒜 = Σ_k α(1−α)^k A^k` over a *sparse* softmax
attention (attention restricted to a sparsity pattern's edges), power-iterated `K` steps,
bidirectional (language pretraining, classification, QA, images, LRA); Proposition 4 in the body:
`lim_{K→∞} Z^{(K)} = 𝒜V`; no truncation bound. 1.67× memory saving, +2.30 % LRA average,
+0.94 % text classification (HTML, [V]).
Scored: (a) ✓ (c) ✓; (b) ✗ (d) ✗ (e)–(j) ✗.
Route: search 1.

### 2.6 `erel-2025-attentionchains` — arXiv:2507.17657, NeurIPS 2025 — [V] (abs + HTML)
Owns: the reading of one attention matrix as a discrete-time Markov chain; indirect attention by
power iteration `v_{n+1}ᵀ = v_nᵀ A`; TokenRank as the damped stationary vector with
`P' = αP + (1−α) 11ᵀ/n`; metastable token regions; post hoc, vision only.
Scored: (a) ✓ (c) ✓ as an *interpretation*; none of (b)(d)–(j) as a layer.
Route: search 5.

### 2.7 `candanedo-2026-diffusionattention` — arXiv:2604.09560 — [V]
Owns: softmax attention as the row-normalised operator of a diffusion map (Coifman–Lafon), with
a metric / node-potential / circulating decomposition of the score. No resolvent.
Route: search 5.

### 2.8 `liu-2021-eignn` — arXiv:2202.10720, NeurIPS 2021 — [V] (abs only)
Owns: the closed-form infinite-depth linear GNN with eigendecomposition on a **fixed** propagation
matrix. The body's formula was not readable this session (PDF binary; ar5iv conversion failed) —
recorded as [U] for the formula, [V] for the record. Ticks (d) on a fixed graph only.
Route: search 9.

### 2.9 `wu-2023-difformer` — arXiv:2301.09474, ICLR 2023 — [V]
Owns: attention as the pairwise diffusivity of an energy-constrained diffusion, closed-form
optimal diffusion strength per step; a step-wise scheme, not a resolvent.
Route: search 11.

### 2.10 `chien-2020-gprgnn` — arXiv:2006.07988, ICLR 2021 — [V]
Owns: generalised PageRank with learnable (signed) hop weights on a fixed graph — the family
ParaFormer lifts to attention. Route: follow-up of 1.2.

### 2.11 `gasteiger-2019-gdc` — arXiv:1911.05485, NeurIPS 2019 — [V]
Owns: graph diffusion convolution — PPR / heat-kernel diffusion of the fixed adjacency, sparsified,
as a drop-in preprocessing. Route: search 16. Already `DONE_ARCHIVE_ROUND1.md:677`.

### 2.12 `gilton-2019-neumann` — arXiv:1901.03707 — [V]
Owns: a truncated Neumann series of `(I − ηXᵀX)^{-1}` with a learned regulariser as the network
itself (inverse problems). The precedent for "Neumann series as layers"; no attention, no chain.
Route: search 12.

### 2.13 `tamar-2016-vin` — arXiv:1602.02867, NIPS 2016 — [V]
Owns: differentiable value iteration as a CNN planning module. The nearest neural owner of "plan
by a fixed-point solve inside the network" for a *decision*; the shape's committor solve is the
linear (policy-evaluation, absorbing) special case of what VIN iterates.
Route: search 13.

### 2.14 `xiao-2023-attentionsinks` (ICLR 2024), `gu-2024-sinkemerges` (ICLR 2025), `lee-2026-asap` — [V]
Own: the attention *sink* — a token (BOS / placeholder) that receives excess mass; Gu et al.:
sinks act as key biases storing non-informative scores, born of softmax's row normalisation; ASAP:
a ViT information flow as a lazy random walk with the sink as the dominant accumulator (the "relaxed
analog of an absorbing state" phrasing is from a search snippet of the body, [U]).
DERIVED, the distinction the paper must draw: a sink is a **column** phenomenon (every row
attends to it); the shape's absorbing set is a **row** condition (`P_{a,·} = e_a`, the constraint
token attends only to itself). The record's own value-zero BOS sink
(`V15Fork.Asink_row_sum`, `workdonenew.md` §1) is a column sink used to make the normaliser a
change of units; it is not an absorbing row. The two coexist and must not be conflated
(`P-7`, vocabulary with no referent).
Route: searches 15, 17, 22.

### 2.15 `abnar-2020-attentionflow` — arXiv:2005.00928 — [V]
Owns: attention rollout — the product of per-layer attention matrices as a path count across
depth — the depth-wise analogue of the within-layer hop sum. Route: memory, then abs fetched.

### 2.16 `zhao-2020-transformerxh` — ICLR 2020 — [V] (Microsoft Research page; OpenReview was
behind a bot check)
Owns: "eXtra hop" attention across linked documents — iterated hops, not a resolvent. Route:
search 3.

### 2.17 `whittington-2021-hippocampus` — arXiv:2112.04035 — [V]
Owns: transformers with recurrent positional encodings reproduce place/grid representations and
map onto hippocampal models — the published bridge between attention and the SR-like predictive
map; no resolvent read. Route: search 10 (the query returned only neuroscience SR papers; this one
from memory, abs fetched).

### 2.18 `geshkovski-2023-mathperspective` — arXiv:2312.10794 — [V]
Owns: attention as an interacting particle system, clustering in long time. The continuous-depth
flow whose Laplace transform is the resolvent (§1.3 DERIVED); no resolvent in the paper itself.

### 2.19 `yang-2024-deltanet`, `dao-2024-ssd` — arXiv:2406.06484, 2405.21060 (ICML 2024) — [V]
Own: the causal triangular solve as a *parallelisation device* for content-dependent recurrences
— DeltaNet's WY / Householder-product representation, Mamba-2's semiseparable-matrix duality
(the cumulative product is the 1-semiseparable case). This is the lineage of the record's corner 3
(I2, `ceq/arm_smprime.py:144`), already named at `DONE_ARCHIVE_ROUND1.md:679-681`. They tick
(a)(b)(d) on the *linear-attention* corner, not on softmax; no (e)–(j).

## 3. The classical owners of the boundary-condition read (fixed chain, no attention)

These are the sources that own box (e) — the absorbing-row / committor read — and box (g)/(h)
outside attention. They are cited so that the paper never writes "novel" beside a linear-algebra
fact that is a 1960 chapter.

### 3.1 `kemeny-1976-finitemarkov` — Open Library ISBN 9780387901923 — [V] (record), formulas [U]
Owns: the fundamental matrix `N = (I − Q)^{-1}` of an absorbing chain and the absorption
probabilities `B = N R` — BRIEF.md I3 verbatim. The record's `ceq/beds/bed_1.py:188-198 committor`
solves this Dirichlet problem (`READ` via BRIEF.md I3).

### 3.2 `zhu-2003-harmonic` — ICML 2003 (DBLP record [V]; AAAI abstract page [V])
Owns: labelled nodes as clamped boundary, the harmonic extension over unlabelled nodes obtained by
matrix methods, with — in the abstract's words — connections to random walks and electric networks;
the body's solution `f_u = (I − P_uu)^{-1} P_ul f_l` and its absorbing-walk reading are from memory
[U]. This is box (e) + (d) on a fixed similarity graph: **the committor as the read, with the
labelled set as the absorbing rows.**

### 3.3 `zhou-2003-consistency` — NIPS 2003 — [V] (proceedings page)
Owns: `F* = (I − αS)^{-1} Y` (memory [U]) — the resolvent of a *fixed* normalised similarity
applied to a label matrix, i.e. `(I − γP)^{-1} V` with `V = Y`.

### 3.4 `de-2014-absorbingtransduction` — arXiv:1402.4566 — [V] (abs; page carries a withdrawal)
Owns the vocabulary "labelled nodes as absorbing states" for transduction on directed graphs.
Cited only for the vocabulary; the withdrawal is recorded.

### 3.5 `baier-2008-modelchecking` — Open Library ISBN 9780262026499 — [V] (record; author names [U])
Owns: reachability probabilities of a DTMC as the unique solution of a linear system after making
the target set absorbing (Ch. 10, memory [U]); a search snippet (Journal of Automated Reasoning
2025, [U]) states that several target sets are handled by one linear system. This is box (h)'s
`q^{(k)}` for `k = 1..K`, computed and used as a safety verdict, in probabilistic model checking.

### 3.6 `todorov-2006-lmdp` (NIPS 19, [V]) and `todorov-2009-efficient` (PNAS 106(28):11478, [V] Crossref + Europe PMC abstract)
Own: the linearly-solvable MDP — the exponentiated value ("desirability") satisfies a *linear*
Bellman equation; the first-exit form fixes `z` on terminal states and solves a linear system on
the non-terminal block (the first-exit definition with terminal states is standard and was
confirmed only through a secondary abs, Jonsson–Gómez arXiv:1603.03267 [V], whose abstract speaks
of the linear Bellman equation; the boundary-value form itself is [U] this session); the optimal
policy is `u*(x'|x) ∝ p(x'|x) z(x')` (memory [U]). This is the nearest owner of **box (h)**: a move
chosen through a linear absorbing-chain solve. It has one cost, not `K` constraint sets, and the
decision is a softmax over `z`, not `argmin max_k`.

### 3.7 `piray-2021-linearrl` — Nature Communications 12:4942 — [V] (Crossref + Europe PMC full text PMC8368103)
Owns: the **default representation** — Eq. 4 of the body, `exp(v*) = M P exp(r)`, with `M` "the
key matrix" over the nonterminal states and terminal states distinguished as absorbing; the body
says it is similar to the SR except that it is for optimal rather than on-policy values; and
**Eq. 5, `M = M_old + M_B`**, the low-rank correction when a barrier is inserted, computable from
`M_old`. A citing paper (DROGO, arXiv:2602.00403, search snippet [U]) writes the DR as
`Z = [diag(exp(−r/λ)) − P]^{-1}`.
DERIVED: Eq. 5 is the Woodbury identity for a change of `r` rows of the chain, rank ≤ `r`. The
shape's `do(a)` — intervene on the context, re-solve, read `Δz` — is this identity on an
attention-built `P`: if a candidate move changes `r` rows of `P`, `(I − γP')^{-1}` is
`(I − γP)^{-1}` plus a rank-≤`r` correction. **Box (g) is therefore OCCUPIED on a fixed
environment chain by Piray–Daw** and the paper's interventional channel must be stated as
"Piray–Daw's barrier update, with the chain built from content and the boundary sets from the
prompt".

### 3.8 `esterhuysen-2026-terminal` — arXiv:2605.31289 — [V]
Owns: the terminal representation — the DR's information without eigendecomposition, embedded in
the top DR eigenvector; option discovery, shaping, transfer. Recent evidence that the
absorbing-chain resolvent is an active RL object; no attention.

### 3.9 `katz-1953-status` — Psychometrika 18(1):39–43 — [V] (Crossref)
Owns the object itself: `(I − αA)^{-1} 1 − 1` as a status index. InfSA's centrality is this
(§2.3 DERIVED).

### 3.10 `khoo-2018-committor`, `li-2019-committor` — [V]
Own the committor as a regression target for neural networks (already `PRIOR_ART.md:625-630`).
Not attention; the committor is the label, not the read.

### 3.11 Page–Brin–Motwani–Winograd 1999 — **NOT ENTERED IN THE BIB.** The Stanford InfoLab URL
failed (SSL error) and DBLP returned zero hits; the tech report is reached by memory only [U].
BRIEF.md §5.3 forbids a citation without a fetched identifier; APPNP and Katz carry the lineage.

## 4. Occupancy matrix (✓ ticked on a content-dependent attention operator; ○ ticked on a fixed graph / non-attention chain; · absent)

| source | (a) content | (b) causal | (c) softmax | (d) exact solve | (e) absorbing rows | (f) γ=0 parity | (g) do(a) re-solve | (h) safest move | (i) certificate | (j) learn γ |
|---|---|---|---|---|---|---|---|---|---|---|
| ChaCAL 2024 | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | · | · (0.9) |
| Structured-sparse 2026 | ✓ | ✓ | ✓ | ✓ tiles / reduced | · | ✓ | · | · | · | · |
| InfSA 2026 | ✓ | · | · sub-stoch. | ✓ | · uniform absorption | · degenerate | · | · | implicit | ✓ |
| MAGNA 2021 | ✓ | · | ✓ | · K∈[3,10] | · | · | · | · | · | · |
| Diffuser 2022 | ✓ | · | ✓ sparse | · K | · | · | · | · | · | · |
| ParaFormer 2025 | ✓ | · | ✓ | · K=10, signed | · | · | · | · | · | ✓ per hop |
| GRAND-l implicit 2021 | ✓ | · | ✓ | ✓ per step | · | · | · | · | · | · |
| Attention-as-DTMC 2025 | ✓ post hoc | · | ✓ | · powers | · | – | · | · | · | · |
| DEQ 2019 / IGNN 2020 / EIGNN 2021 / DEAR 2024 | · | · | · | ○ nonlinear / fixed S | · | · | · | · | · | · |
| APPNP 2019 / GDC 2019 / GPR-GNN 2021 | · | · | ○ | ○ limit | · | · | · | · | · | GPR ✓ |
| DeltaNet 2024 / Mamba-2 2024 | ✓ linear corner | ✓ | · | ✓ | · | · | · | · | · | · |
| Zhu 2003 / Zhou 2003 / Kemeny–Snell | · | · | ○ | ○ | ○ | · | · | · | · | · |
| Piray–Daw 2021 | · | · | ○ | ○ | ○ terminal | · | ○ Eq. 5 | · | · | · |
| Todorov 2006/2009 | · | · | ○ | ○ | ○ terminal | · | · | ○ one cost | · | · |
| Baier–Katoen 2008 | · | · | ○ | ○ | ○ target sets | · | · | ○ verdict | · | · |
| Attention sinks 2023–26 | ✓ | ✓ | ✓ | · | · column, not row | – | · | · | · | · |
| Dayan 1993 / Barreto 2017 | · | · | ○ | ○ | · | · | · | ○ GPI over policies | · | · |

The **conjunction** (a)(b)(c)(d)(f) is held by ChaCAL. The conjunction (a)(b)(c)(d)(e) — the
question as the coordinator set it — is held by no source found. (i) as a printed instrument is
held by no attention source found. (g) and (h) are held only on fixed chains.

## 5. DERIVED notes collected (each with its steps above)

1. `O_shape = Y_ChaCAL / (1−γ)`; equal at γ = 0 (§2.1).
2. `(1−γ)A(I−γA)^{-1}` is row-stochastic for row-stochastic `A`; the resolvent read is a convex
   mixture; non-negativity of the influence Jacobian survives the resolvent and survives absorbing
   rows; absorption redirects, it does not subtract (§2.1).
3. One implicit GRAND-l step is the shape's `z` at `γ = τ/(1+τ)` up to `1/(1+τ)`; the resolvent is
   the Laplace transform of the frozen attention flow at rate `(1−γ)/γ` (§1.3).
4. InfSA's centrality is Katz's index on the attention graph (§2.3).
5. Piray–Daw Eq. 5 is Woodbury for `r` changed rows, rank ≤ `r`; the shape's `do(a)` is that
   update on a content-built chain (§3.7).
6. ParaFormer's family contains the resolvent read as its geometric, infinite-`K` corner (§1.2).

## 6. Searches run (22 beyond the seeds; the brief asked for at least six), with what each returned

| # | query (verbatim or abbreviated) | yield |
|---|---|---|
| 1 | `"personalized PageRank" attention transformer sequence multi-hop attention diffusion` | Diffuser 2210.11794, MAGNA 2009.14332, Graph4MM 2510.16990 (hop-diffused attention, not fetched) |
| 2 | `"resolvent" attention mechanism transformer "(I - " inverse attention matrix infinite depth` | Shaped Transformer 2306.17759 (not relevant), **2605.22476 → ChaCAL 2410.05565** |
| 3 | `"Neumann series" attention transformer multi-hop propagation truncation` | **InfSA 2603.00175**, Diffuser, Transformer-XH |
| 4 | `"Multi-hop Attention Graph Neural Network" MAGNA attention diffusion personalized PageRank arXiv` | MAGNA abs/IJCAI page |
| 5 | `softmax attention row-stochastic Markov chain random walk over tokens interpretation absorbing state transformer` | Attention-as-DTMC 2507.17657, Diffusion–Attention Connection 2604.09560, Sinkhorn rank decay 2604.07925 (not fetched), "Transformers learning random walks" NeurIPS 2025 (not fetched) |
| 6 | `committor function neural network absorbing Markov chain linear system Dirichlet problem deep learning` | ocean-model committor papers (not fetched), Wikipedia; nothing on attention |
| 7 | `Todorov linearly solvable MDP desirability function terminal states linear system "efficient computation of optimal actions"` | NIPS 2006 page, hierarchical LMDP 1603.03267, Terminal Representation 2605.31289 |
| 8 | `Piray Daw "default representation" linear reinforcement learning terminal states matrix inverse Nature Communications 2021` | Nature landing (IDP redirect), PubMed 34400622 (timeout), DROGO 2602.00403 snippet with the DR formula |
| 9 | `EIGNN efficient infinite-depth graph neural networks closed-form fixed point eigendecomposition NeurIPS 2021` | EIGNN 2202.10720; IGNN-Solver 2410.08524 (not fetched) |
| 10 | `successor representation transformer attention in-context "successor" attention head predictive map` | neuroscience SR papers only; Learning Cognitive Maps from Transformer Representations 2401.05946 (not fetched) |
| 11 | `DIFFormer energy constrained diffusion transformer attention as diffusion step Wu ICLR 2023 arXiv` | DIFFormer 2301.09474 |
| 12 | `"Neumann networks" inverse problems Gilton Ongie Willett truncated Neumann series learned` | 1901.03707 |
| 13 | `"value iteration networks" Tamar 2016 arXiv planning module differentiable` | 1602.02867 |
| 14 | `reachability probabilities discrete-time Markov chain linear equation system Baier Katoen "Principles of Model Checking" absorbing` | JAR 2025 snippet (several target sets, one linear system), MIT Press confirmation |
| 15 | `attention sink BOS token absorbs attention mass StreamingLLM arXiv 2309.17453` | 2309.17453; When Attention Sink Emerges (ICLR 2025) |
| 16 | `"Diffusion Improves Graph Learning" graph diffusion convolution personalized PageRank Gasteiger NeurIPS 2019 arXiv` | 1911.05485 |
| 17 | `attention layer absorbing tokens fixed boundary condition Dirichlet "boundary" tokens harmonic extension transformer attention "absorbing"` | only sink/boundary-token interpretability hits; **no absorbing-row attention layer** |
| 18 | `"chain and causal attention" ChaCAL citing papers resolvent attention entity tracking follow-up 2025 2026` | ChaCAL pages only; the 2026 follow-up was found by search 2, not by this one |
| 19 | `interventional attention do-operator intervene on context tokens re-solve fixed point counterfactual transformer "intervention" consequence prediction attention mechanism` | interpretability interventions on heads/activations only; **no re-solve-and-read-displacement layer** |
| 20 | `deep equilibrium transformer attention fixed point implicit layer "DEQ" transformer language model equilibrium attention input injection` | DEQ, TorchDEQ zoo (input injection detail) |
| 21 | `safest action multiple unsafe sets reachability probability minimax absorbing Markov chain "probabilistic safety" planning linear system constraint sets` | reach-avoid / probabilistic safe sets (make target and avoid sets absorbing); **no attention** |
| 22 | `"absorbing random walk" OR "absorbing Markov chain" "attention" transformer layer "fundamental matrix" attention matrix tokens` (and the sibling `"absorption probability" OR "hitting probability" attention transformer …`) | InfSA again; ASAP 2605.22372; De et al. 1402.4566; nothing with identity rows in an attention layer |

Routes that could not see (recorded so absence is not over-read — `V-7`): MIT Press (403, Dayan),
Nature (IDP redirect, Piray–Daw), PNAS (redirect; Europe PMC full text 404 for PMC2705278),
PubMed (DNS timeout), Semantic Scholar API (429 ×2), Scholar Sidekick verifyCitation (rate-limited /
unsubscribed ×6), OpenReview (bot check, Transformer-XH), Stanford InfoLab (SSL), two PDFs returned
as binary (MAGNA — recovered via ar5iv; Zhu 2003 — recovered via DBLP + AAAI abstract), ar5iv
conversion failure (EIGNN — not recovered). The arXiv HTML fetches are summaries produced by a
reading model over the page, not the page itself; every load-bearing equation from them was
cross-checked against a second route where one existed (ChaCAL: abs + HTML + the 2026 follow-up's
restatement; InfSA: abs + HTML + search-22 snippet; MAGNA: abs + ar5iv + search-1 snippet).

## 7. NOT FOUND (with the queries that would have found it)

- **An attention layer whose operator carries absorbing rows** (identity rows for a constraint set)
  inside a content-dependent resolvent — searches 17, 22, and the boundary/committor terms queried
  inside the ChaCAL, structured-sparse and InfSA bodies (each returned "not mentioned").
- **The committor `q^{(k)} = (I − Q)^{-1} R_k 1` as an attention read** (as opposed to a regression
  label, which `PRIOR_ART.md:625-630` owns) — searches 6, 22, and `"committor" transformer
  attention …` (which returned Decision/Trajectory Transformer noise only).
- **An interventional channel on an attention resolvent** — intervene on context rows, re-solve,
  read `Δz` as the consequence label — search 19 (interpretability interventions on heads and
  activations only). The same operation on a fixed chain is Piray–Daw Eq. 5.
- **`argmin_a max_k q^{(k)}(do a)` as an attention read** — search 21 (probabilistic safety owns
  it as a verifier's verdict, not as a layer).
- **A printed Neumann truncation certificate `γ^{K+1}/(1−γ)` in any attention paper** — the series
  appears as motivation in ChaCAL, Diffuser (Prop. 4 is a limit, not a rate), MAGNA and the 2026
  follow-up; none prints a bound with a δ. The bound itself is a textbook inequality and cannot be
  claimed as mathematics — only as an *instrument* (`L-CERT`).
- **A sequence (language) model with the resolvent read AND learnable γ** — ChaCAL fixes γ = 0.9;
  InfSA learns γ but is non-causal vision. Search 2 and search 3.
- **"Next-equilibrium prediction" / "equilibrium labels as a capability bar" in any ML sense** —
  already NOT FOUND at `PRIOR_ART.md:653-655`; unchanged.
- Page et al. 1999 as a resolvable record — not found by DBLP; the lineage is carried by Katz 1953
  and APPNP.

## 8. VERDICT

**OCCUPIED.** The read `O = P(I − γP)^{-1}V` on a content-dependent, causally masked,
row-stochastic softmax `P`, solved exactly by a triangular solve, reducing to softmax at γ = 0, is
**owned by Fagnou, Caillon, Delattre and Allauzen, ChaCAL, EMNLP 2024 (arXiv:2410.05565)** up to the
scalar `1/(1−γ)`; its subquadratic blockwise evaluation is owned by Zhao, Caillon, Fagnou and
Allauzen (arXiv:2605.22476). Identities I1 and I5 of BRIEF.md are re-derivations of that paper.
The multi-hop PPR sum on a content-dependent softmax attention is owned earlier, truncated and
non-causal, by MAGNA (IJCAI 2021) on graphs and Diffuser (2022) on sequences, and generalised with
learnable signed hop weights by ParaFormer (2025). The fundamental-matrix / absorbing-chain
*reading* of `(I − γA)^{-1}` for attention, with learnable per-head γ, is owned by InfSA
(arXiv:2603.00175), on a sub-stochastic non-causal operator. The equilibrium framing (a fixed
point as the layer's output, implicit differentiation) is owned by DEQ (2019), IGNN (2020), EIGNN
(2021), DEAR (2024); the per-step resolvent of a content-dependent attention operator is owned by
GRAND's implicit scheme (2021). The successor / discounted-occupancy reading of the resolvent is
owned by Dayan (1993) and Barreto et al. (2017); the absorbing-boundary resolvent with a low-rank
re-solve after an intervention is owned, on a fixed environment chain, by Piray and Daw (2021,
Eq. 4–5); the linear first-exit solve that yields a move is owned by Todorov (2006, 2009); the
committor as `(I − Q)^{-1}R1` is Kemeny–Snell (1960/1976); clamped labels as absorbing rows with a
harmonic read is Zhu–Ghahramani–Lafferty (2003) and Zhou et al. (2003); several target sets in one
linear system with a safety verdict is probabilistic model checking (Baier–Katoen 2008).

**NEAR-MISS.** (i) InfSA's per-row uniform absorption `R_i = 1 − γΣ_jÂ_ij` is a *probabilistic*
boundary, one step short of identity rows on named sets; (ii) attention sinks (Xiao 2023, Gu 2025,
ASAP 2026) are column absorbers, the dual of the shape's row condition; (iii) GRAND-l's implicit
step is the shape's `z` on an undirected graph without a mask; (iv) ParaFormer's family contains
the resolvent read as a corner it never takes to `K = ∞`; (v) DeltaNet / Mamba-2 hold the causal
triangular solve on the linear-attention corner (the record's corner 3), which the shape lifts to
the softmax corner — but ChaCAL already did that lift.

**NOT FOUND.** Absorbing constraint rows inside a content-dependent causal attention resolvent;
the committor into `K` prompt-named sets as an attention read; the interventional re-solve with
`Δz` as the label on such an operator; the `argmin_a max_k q^{(k)}` decision as a read; a printed
Neumann certificate as an instrument in an attention paper; learnable γ on a causal language-model
resolvent. Each with the queries of §7.

**SURPRISE — flagged loudly.** ChaCAL occupies the *whole* of the shape's read as written in
BRIEF.md §1 lines 64–65, and the record has never cited it (`0 files`, RUN). What survives of the
delta is exactly and only boxes (e)(g)(h)(i)(j) in conjunction with (a)(b)(c)(d)(f): the shape is
**"ChaCAL with boundary conditions, an interventional channel, a multi-constraint committor
decision, a certificate, and a learnable γ"** — and each of (e)(g)(h) is separately owned on fixed
chains (Kemeny–Snell, Zhu et al., Piray–Daw, Todorov, Baier–Katoen). A second, smaller surprise:
InfSA has already put the words "fundamental matrix of an absorbing Markov chain" in an attention
abstract, so the paper cannot claim the *reading*, only the *rows*. A third: Piray–Daw's Eq. 5 is
the interventional channel, published in 2021 in a neuroscience journal the ML sweeps do not index.

## 9. Design against the taxonomy — what this sweep obliges the paper and the plan to do

Each item names the `MISTAKES.md` mechanism it is designed against (BRIEF.md §5.6).

1. **ChaCAL becomes a mandatory control arm on every bed where γ > 0 matters** — the fellow
   approximator, not the skyline. Any sentence of the form "the shape beats softmax at matched
   depth" on a bed ChaCAL can also solve is `D-1` (racing a proven optimum — here the *published*
   solver of the same operator) and `R-SKY` (`CEQ_V16_CONTRACT.md`, READ via BRIEF.md §1). The
   deeper-softmax skyline of BRIEF.md §1 stays; ChaCAL is added beside it.
2. **The identity bind I1 (γ = 0 parity) has an empty rejection region against ChaCAL** — both
   reduce to softmax at γ = 0, so I1 cannot separate the shape from ChaCAL (`V-24`, an identity
   bind whose rejection region is empty). The bind that separates them must plant the boundary
   rows: a bed where a constraint set `𝒜_k` is present in the context and the label is the
   committor or its `argmin max` — ChaCAL has no `𝒜_k` and must fail by construction, and the
   planted negative is "ChaCAL with the same γ". I4's non-stochastic `P` (rows sum 1.5) stays as
   the certificate's planted negative.
3. **The committor oracle must run on the latent environment chain, never on the arm's `P`**
   (`D-2`). Kemeny–Snell and Piray–Daw compute the same object from the *true* chain; the arm's
   `P` is content-built. If the bed's chain is handed to the arm as `P`, the arm is the oracle and
   the bed measures nothing.
4. **The paper's obstruction 4 sentence must be rewritten** (`P-7`, vocabulary with no referent;
   `V-23`, a plural claim whose central member is the counterexample): the resolvent read with
   row-stochastic `P` is a non-negative, row-normalised mixture (DERIVED §5.2); "a third token can
   veto" is licensed only as *redirection by absorption*, which is F0's zero-gate mechanism in
   boundary form, and the influence-Jacobian test of `ceq/attention.py` (min entry exactly 0) will
   read 0 on the shape too. State it, do not discover it later.
5. **Every hop-sum comparison names its truncation and prints δ** (`L-CERT`; `P-8`, an upper bound
   stated as a price). MAGNA `K ∈ [3,10]`, Diffuser `K`, ParaFormer `K = 10`, the 2026 follow-up's
   reduced system — none prints a bound. I4 is the instrument; its planted negative already exists.
6. **The prior-art section cites ChaCAL, InfSA, MAGNA, Diffuser, ParaFormer, GRAND and Piray–Daw
   before naming any component** (`BRIEF.md` §5.7; `M-10`, filed without grepping for what existed;
   `P-3`, stale claim — `THEORY.md:225-235` must be updated from "unverified" to the [V] records in
   the bib).
7. **The ACL Anthology id, the NeurIPS 2020 venue of IGNN, the ICML 2021 venue of GRAND, the
   AISTATS 2022 year of Sinkformer, the NeurIPS 2024 venue of DeltaNet and the ICLR 2022 venue of
   Whittington are [U]** and are so marked in the bib `note` fields; the assembler either resolves
   them or prints them as arXiv only (`P-10`, a source's intro cited as its theorem — here, a
   venue cited from memory).
8. **The `γ_k` signed-coefficient family (ParaFormer, GPR-GNN) must not be re-argued as a signed
   operator** — the record already closed that programme (`RESEARCH.md`, `PROGNOSIS.md`, READ via
   BRIEF.md §3); ParaFormer's base is non-negative and `DONE_ARCHIVE_ROUND1.md:1321-1324` records
   that no signed-base variant exists there. Design against `M-17` (classifying the correction
   record as the defect).
9. **Attention sinks vs absorbing rows** get one displayed definition each, with the record's own
   value-zero BOS sink placed on the sink side (`P-7`).
10. **What dies if what.** If a bed with prompt-named constraint sets shows ChaCAL-with-a-sink-token
    matching the shape's committor read within the TOST margin, box (e) is not a capability and
    the shape collapses to ChaCAL + certificate; the paper must pre-register that kill
    (`M-7`, pre-registration with a hole; `D-7`, prediction without its counter).

## 10. Open gaps handed to the coordinator

- EIGNN's closed-form equation was not read from the primary (all three routes failed); the
  assembler should either fetch the NeurIPS PDF by another route or cite EIGNN for the *claim* of
  a closed form only.
- Todorov's first-exit boundary-value statement is [U] from the 2009 body; the 2006 NIPS abstract
  [V] carries the linear-Bellman claim. A fetch of the PMC full text (PMC2705278) by another route
  would close it.
- Zhu 2003's formula and Zhou 2003's closed form are [U] from the bodies; the records are [V].
- Kemeny–Snell's edition year (1960 Van Nostrand / 1976 Springer) is [U]; the Open Library record
  lists a 1983 printing under the ISBN used.
- ChaCAL's `log₂(n+1)` depth bound and Sanford–Hsu–Telgarsky's `⌊log₂ k⌋ + 2` must be reconciled
  by the depth-lineage planet (same law, different constants and problems).
- The multizoom bound (`ceq/multizoom.py`) versus the 2026 structured-sparse reduced system is the
  one place in this lineage where the record may hold something the literature lacks (a printed
  block-approximation bound); it was not checked this session and should be by whoever holds the
  cost lineage.
