# Prior-art sweep — lineage S6: topology in and around the attention role

SATURN-S6, 2026-09-03. Companion file: `bib_topology.bib` (59 entries, one per source).
Repository: `<repo root>`, branch `v17k-gate0`, HEAD `207e7b9`.

Evidence classes as in `BRIEF.md` §5.3: `RUN` (executed this session), `READ path:line`,
`CITED [V]` (abs page / DOI metadata record / primary host API fetched this session, title
matched), `CITED [U]` (search index, reference list, or memory only), `DERIVED`.

**The question put to this planet.** Is any part of the shape's topological layer occupied:
(a) zero-gate segmentation as an exact block certificate; (b) β₀ persistence of the
influence graph across a threshold as a causal-structure barcode; (c) isocommittor surfaces
as decision boundaries inside an attention model; (d) a Mapper cover as the long-context
candidate builder for a resolvent read.

**The short verdict.** (a) mechanism NEAR-MISS (gated-linear-attention family), certificate
owned by the record's own Lean and NOT FOUND outside it; (b) the *descriptive* barcode of an
attention graph is OCCUPIED (Kushnareva et al. 2021 and its lineage), the *interventional*
reading has a statistical owner (Kim & Lee 2026), the influence-Jacobian filtration with a
printed δ is NOT FOUND; (c) isocommittor surfaces are OCCUPIED by transition-path theory,
their use inside an attention model is NOT FOUND; (d) Mapper is OCCUPIED (Singh–Mémoli–
Carlsson 2007), cluster covers as candidate builders are OCCUPIED (Routing Transformer,
Reformer, SBM attention), persistence-derived block schedules are OCCUPIED by the author's
own merged `triton-lang/kernels#22`, and the composition Mapper cover → CSR schedule →
resolvent read is NOT FOUND. No single paper occupies the whole delta. Five surprises,
§4; the loudest is that the record's three M10 arXiv ids are all misattributed.

Tally: **58 [V], 1 [U]** (Mapper's DOI landing page returned HTTP 403; two bibliographic
records agree). Searches run: **17** (§1.2), of which 15 go beyond the seed list.

---

## 1. Method

### 1.1 Verification protocol

Every arXiv source: `https://arxiv.org/abs/<id>` fetched, title read, author list read,
primary class read. Every DOI source: Crossref `api.crossref.org/works/<doi>` fetched (the
DOI registration agency's own record) after the publisher landing page either redirected to
an authorization endpoint (Springer, `10.1007/s00454-006-1276-5`) or returned 403
(Eurographics diglib, `10.2312/SPBG/SPBG07/091-100`). Two workshop papers absent from arXiv
were resolved through the primary host's API (`api.openreview.net`, `proceedings.mlr.press`).
The two GitHub pull requests were fetched at `github.com/<org>/<repo>/pull/<n>` and, for
`kernels#22`, at `/files`.

**Planted positive for the search instrument (`MISTAKES.md` V-7, READ `:117-135`: "a
reported absence needs a planted positive").** The same search tool that returned NOT FOUND
for the four composite queries in §5 returned, on identical settings, the seed papers it was
required to find: query 4 (§1.2) surfaced Kushnareva 2022 `2207.01903` and Perez–Reinauer
`2206.15195` beside the seeded Cherniavskii 2022; query 11 surfaced Bodnar 2022 beside the
sought Barbero 2022. The absences in §5 are therefore claims about the literature as indexed,
not about a search that could find nothing.

### 1.2 Queries run (in order; what each found)

| # | query | found (ids) |
|---|---|---|
| 1 | `"Sheaf Attention Networks" Barbero Bodnar Veličković Liò 2022` | pointer to NeurReps 2022; `2601.21207`, `2608.02558`, PMLR v196 `barbero22a` |
| 2 | `persistent homology sparse attention schedule transformer` | `2605.03163`, `2312.10702`, `2107.09031` |
| 3 | `topological attention sparsification transformer persistence block mask` | `2410.03462`, `2210.15541` |
| 4 | `attention graph persistent homology betti causal structure` | `2207.01903` (via ResearchGate title), `2206.15195`, `2205.09630` |
| 5 | `committor function neural network isocommittor surface decision boundary transition path theory` | `2507.21961`, `2012.06727`, `1802.10275`, Nature Comput. Sci. 2025 |
| 6 | `Mapper algorithm topological data analysis long context retrieval transformer candidate selection` | Mapper-only hits (`1706.00204` via JMLR URL, `2412.11631`); **no attention hit** |
| 7 | `forget gate zero resets state equivalent block diagonal attention mask segmentation gated linear attention` | `2312.06635`, `2503.02130`, `2502.01578`, `2605.22791` |
| 8 | `persistent homology attention maps large language model hallucination detection topological 2025` | `2504.10063`, `2601.01552`, `2605.05025`, `2604.10697` |
| 9 | `Levin Peres Wilmer Markov Chains and Mixing Times Theorem 13.10 bottleneck ratio Cheeger` | the authors' PDF at `pages.uoregon.edu/dlevin/MARKOV/` |
| 10 | `cellular transformer simplicial attention Hodge Laplacian attention 2024 arXiv` | `2405.14094`, `2403.06687`, `2607.10677`, `2509.01839`, `2204.09455` |
| 11 | arXiv API `ti:"Sheaf Attention Networks"` | 0 results (the paper is not on arXiv) |
| 12 | arXiv API `ti:"Betti numbers of attention graphs"` | `2207.01903v1` |
| 13 | `transformer learns committor absorbing Markov chain attention "committor" transformer` | `2402.04161`, `2410.05493`, `2110.05050`; **no committor-inside-attention hit** |
| 14 | `exact block-diagonal attention equivalence gate zero segment boundary theorem proof "block diagonal" gated linear attention reset` | `2606.02680`, `2604.14702`, `2608.28541`, `2606.17830`; **no exact-certificate hit** |
| 15 | `Reeb graph Mapper attention transformer tokens cover overlapping clusters sparse attention` | graph-transformer sparsity only (`2303.06147`, `2502.01659`); **no Mapper/Reeb hit** |
| 16 | `"zigzag persistence" OR "persistence barcode" attention influence Jacobian threshold connected components causal` | `2601.01552`, `2103.07353`, `2304.03828`; **no influence-Jacobian filtration hit** |
| 17 | dblp / OpenReview / Semantic Scholar record lookups for Mapper 2007 and Sheaf Attention Networks | dblp `conf/spbg/SinghMC07`; OpenReview forum `LIDvgVjpkZr` |

Seeds that were fetched directly without a search (ids known): `2109.04825`, `2205.09630`,
`2202.04579`, `2203.07485`, `2206.00606`, `2011.05804`, `1905.12200`, `1904.09378`,
`2010.08356`, `1906.00722`, `1906.09003`, `1307.6188`, `1207.3674`, `1701.00565`,
`1608.05432`, `1608.00365`, `1706.00204`, `2603.02289`, `2012.06333`, `1808.01513`,
`2003.05997`, `2001.04451`, `2502.11089`, `2507.07955`, the two GitHub PRs, and the Crossref
records for Cohen-Steiner 2007, E–Vanden-Eijnden 2010, Vanden-Eijnden–Venturoli 2009,
Levin–Peres 2017, Milnor 1963, Edelsbrunner–Harer 2009, Carlsson 2009, Reimann 2017,
Perea–Harer 2015.

### 1.3 What the record already owns in this lineage (READ)

- `lean/CEQ/V16Domain.lean:129-131` — `pathProd_eq_zero_iff`: the path product is `0` **iff**
  some gate magnitude on the path is `0` (an iff, machine-checked, no `sorry`; `README.md` §2
  per the brief).
- `lean/CEQ/V16Domain.lean:165-169` — `no_prefix_scan_represents_a_zero_gate`: for any prefix
  scan `C : ℕ → ℂ`, `exp(C i − C j) ≠ pathProd` whenever a gate on the path is zero, because
  `Complex.exp` is never zero. The docstring at `:157-164` names this a cost in the
  *computational route*, not the conclusion.
- `ceq/certs/topological.py:1-70` — three certificates that can *refuse*: `Z` winding,
  persistent `β₁` of a carrier trajectory, Euler–Poincaré against a Poincaré–Hopf index sum;
  the persistence threshold `τ = 4 · max nearest-neighbour distance` is derived from
  `d_B ≤ 2 d_H`, not fitted (`:47-56`, explicitly against `MISTAKES.md` M-2).
- `ceq/certs/topological.py:476-500` — `beta0_interleaving`: the sandwich
  `b0(X, r+2ε) ≤ b0(X_ε, r) ≤ b0(X, r−2ε)` with a `strict_span` measure so that a sandwich
  that never bites cannot report a pass (against V-16). This is a β₀-stability instrument on
  *point clouds*, not on an influence graph.
- `ceq/rips.py:1-30` — the S² geodesic Vietoris–Rips corpus ported from the author's
  `mujoco#3396`; a Rips 1-skeleton and `b0`; component count falls from 178 to 1 across six
  cases (docstring `:16-17`).
- `ceq/beds/bed_1.py:5-8` — BED-1's label is the committor `q`; "Guards are the `q = 1/2`
  isocommittor surfaces and the itinerary of guard crossings is the action vocabulary."
  `:382-390` `guard_itinerary` counts traversals of the `q = 1/2` surface by channel.
  `:188` `committor` solves the Dirichlet problem; `:336-343` `morse_census` reports
  `m₀ − m₁ = χ = V − E`.
- `workdonenewseal.md:265` — the record's own caveat: on a 1-complex `β₀ − β₁ = V − E`
  identically, so half of the Euler–Poincaré cross-check cannot fail.
- `THEORY.md:23` — `topoml` supplies `metric_cover / nerve_graph / mapper_graph`;
  `THEORY.md:114-118` — `kernels#22` consumes a causal CSR block schedule with a
  0D-persistence salience over key-block centroids; `THEORY.md:162-165` — schedule selection
  with the do-nothing (merged 0D-salience) schedule always entered.
- `THEORY.md:176-190` — the v1 pipeline: barcode → Hilbert-series embedding → hierarchical
  partition → CSR schedule → scheduled attention → resolvent solve `(I − γP)⁻¹ b` → sheaf
  consistency gate `‖R·û − ψ̂‖`.
- `tests/foreman/test_topology_washout.py:1-16` — the refutation instrument asking whether
  the resolvent stage preserves what the barcode stage measured; `AUDIT.md:562` keeps it as a
  refutation instrument whose RED is its deliverable.
- `CEQ_V20_R15_CONTRACT.md:220-226` — M10: "β₀ of the influence graph across the threshold ε
  is a persistence barcode of causal structure [RUN: 4096→…→1]; segmentation is its ε=0
  endpoint; the differentiable-persistence lineage (1905.12200, 1904.09378, 2011.05804 —
  occupied)". The three ids are misattributed; §4.1.
- `MATHEMATICS.md:455` — "A Cheeger obstruction bounds what any one-edge bridge can do.
  `g ≤ 2φ`". The cited theorem needs reversibility; §4.5.
- `PRIOR_ART.md:629-640` — the committor as a regression target already recorded [V] (Khoo,
  Lu, Ying `1802.10275`; Li, Lin, Ren `1906.06285`; Contreras Arredondo et al.
  `2507.17700`), and NOT FOUND for `B = N R` as a supervised label of a neural network.

---

## 2. Sources

Format per source: **key** — [V]/[U] — *owns* — *leaves open relative to the shape* —
*found by*.

### A. TDA of attention maps (descriptive barcodes of attention graphs)

**kushnareva-2021-tda-attention** (`2109.04825`, cs.CL, EMNLP 2021) — [V] — Owns the
construction: threshold a BERT head's attention matrix at a sweep of levels, read the
resulting graph's H0/H1 persistence and simple graph statistics as features, and classify
(here: machine-generated vs human text). — Leaves open: the filtration is by *attention
weight* of a trained softmax model, read post hoc; nothing is a certificate, nothing feeds
back into the operator, nothing is causal or interventional; there is no absorbing set and no
resolvent. — seed.

**cherniavskii-2022-acceptability** (`2205.09630`, cs.CL, Findings EMNLP 2022) — [V] — Owns
the same feature family applied to grammatical acceptability across languages. — Same
openings as above. — seed.

**kushnareva-2022-betti** (`2207.01903`, cs.CL, preprint; abs page records "submitted, not
accepted" to TDA&Beyond NeurIPS 2020) — [V] — Owns the *title-level* claim that Betti numbers
of thresholded attention graphs are sufficient classification features. — Same openings; and
by its own status it is a preprint, so the paper cites it as such. — query 4, 12.

**perez-2022-topological-bert** (`2206.15195`, cs.CL) — [V] — Owns "attention → graph →
topological features → classifier" as a full pipeline with an adversarial-robustness claim.
— Same openings. — query 4.

**bazarova-2025-toha** (`2504.10063`, cs.CL, ACL 2026 as printed) — [V] — Owns a
*divergence* between the topologies of two attention subgraphs (prompt vs response) as a
hallucination signal in RAG. — The two-subgraph contrast is the nearest existing thing to
"displacement of structure under a change of context", but the change is *which tokens*, not
an intervention `do(a)` on a jointly determined state; no resolvent, no barcode across ε as a
certificate. — query 8.

**samaga-2026-halluzig** (`2601.01552`, cs.CL) — [V] — Owns zigzag persistence over the
*layer* axis of causally-masked attention graphs (a filtration that goes up and down as the
graph changes layer to layer). — The zigzag axis is depth; the shape's axis is the threshold
ε on one operator's *influence*; open. — query 8, 16.

**balderas-2023-ph-bert-compression** (`2312.10702`, cs.LG, Applied Sciences 2025) — [V] —
Owns persistence-ranked neuron pruning of a Transformer. — Not an attention operator; not
causal. — query 2.

### B. Sheaf, simplicial, cellular and Hodge attention; operator views of attention

**barbero-2022-sheaf-attention** (OpenReview `LIDvgVjpkZr`, NeurReps 2022 Oral) — [V] via
`api.openreview.net`; not on arXiv (query 11: 0 results), not in dblp (0 hits); the reference
entry in `2608.02558` agrees. — Owns sheaf diffusion with input-dependent attention (the GAT
generalisation of neural sheaf diffusion, as `2608.02558`'s body states). — Attention here is
a *weighting of restriction maps on a graph*, not a row-stochastic resolvent with absorbing
rows; no boundary conditions, no committor. — query 1, 17.

**barbero-2022-connection-laplacians** (PMLR v196 pp. 28–36) — [V] — Owns sheaf construction
by Riemannian alignment of tangent spaces (orthogonal restriction maps). — Same openings. —
query 1.

**bodnar-2022-neural-sheaf-diffusion** (`2202.04579`, cs.LG, NeurIPS 2022) — [V] — Owns the
sheaf-Laplacian diffusion with the heterophily / oversmoothing analysis and the
asymptotic-behaviour control. — The shape's `(I − γP)⁻¹` is a *resolvent* of a stochastic
operator, not a diffusion to its kernel; `THEORY.md:215` already flags that `ker Δ_F` may be
trivial for generic restriction maps. — seed.

**hansen-2020-sheaf-neural-networks** (`2012.06333`, cs.LG, NeurIPS 2020 TDA&Beyond) — [V]
— Owns the sheaf-Laplacian GCN. — Same. — `THEORY.md:232` names it; fetched here.

**hansen-2019-spectral-cellular-sheaves** (`1808.01513`, math.AT) — [V] — Owns the sheaf
Laplacian and its Hodge theory on cell complexes. — The mathematical object behind any sheaf
consistency gate `‖R·û − ψ̂‖` (`THEORY.md:188`); no attention, no resolvent read. —
fetched directly.

**lin-2026-connection-laplacian-attention** (`2607.10677`, cs.LG) — [V] — Owns the
identification of single-head self-attention with a *connection walk* on the token graph
(messages transported by learned linear maps) and multi-head attention with an edge-dependent
connection walk; states when the generator reduces to a random-walk connection Laplacian
(stochasticity, reversibility). — **Near-miss to the definition of `P`, not to the shape**:
the abs page carries no resolvent, fixed point, absorbing boundary, committor or persistence.
If the paper's `P` is presented as a connection walk, this paper owns that reading and must
be cited before the word is used. — query 10.

**hu-2026-sheaf-attention-perspective** (`2601.21207`, cs.LG, AAAI 2026 MATH4AI workshop) —
[V] — Owns a cellular-sheaf reading of attention in graph neural models through local
consistency. — No resolvent, boundary, committor or barcode on the abs page. — query 1.

**giusti-2022-simplicial-attention** (`2203.07485`, cs.LG) — [V] — Owns attention over
upper/lower simplicial neighbourhoods. — Attention on a *given* complex; the shape's complex
(the influence graph) is produced by the operator. — seed.

**goh-2022-simplicial-attention-networks** (`2204.09455`, cs.LG, ICLR 2022 GTRL workshop) —
[V] — Owns signed (orientation-equivariant) simplicial attention. — Same. — query 10.

**hajij-2022-topological-deep-learning** (`2206.00606`, cs.LG) — [V] — Owns combinatorial
complexes and message passing over them (the survey/framework). — Same. — seed.

**ballester-2024-cellular-transformer** (`2405.14094`, cs.LG) — [V] — Owns a Transformer
over cell complexes with incidence-based attention and cellular positional encodings. — Same.
— query 10.

### C. Differentiable persistent homology (the M10 lineage; ids corrected)

**gabrielsson-2019-topology-layer** (`1905.12200`, cs.LG) — [V] — Owns a differentiable
persistence layer over level-set and edge filtrations, used as a regulariser / constraint /
adversary. — Trains through persistence of *data or activations*; the shape's filtration
would be the influence graph of a resolvent; open. — seed id; **the record attributes this id
to Hofer et al. (`CEQ_V20_R15_CONTRACT.md:223`); it is Brüel-Gabrielsson et al.** — §4.1.

**carriere-2019-perslay** (`1904.09378`, stat.ML) — [V] — Owns learnable vectorisations of
persistence diagrams (PersLay) and extended-persistence graph signatures. — Same. — seed id;
**the record attributes this id to Moor et al.; it is PersLay.** — §4.1.

**corcoran-2020-ph-gradient-regularization** (`2011.05804`, cs.LG) — [V] — Owns a grouping
regulariser that makes persistence gradients act on aggregates rather than single points. —
Same. — seed id; **the record attributes this id to Carrière et al. 2021; it is Corcoran &
Deng.** — §4.1.

**carriere-2021-optimizing-ph** (`2010.08356`, cs.CG) — [V] — Owns the differentiability
conditions and subgradient convergence for persistence-based objectives. This is the paper the
record *meant* by "Carriere et al. 2021". — Same openings. — fetched directly.

**hofer-2019-connectivity-optimized** (`1906.09003`, cs.LG) — [V] — Owns a persistence loss
that shapes latent-space connectivity (β₀ structure) for one-class learning. This is the
paper the record *meant* by "Hofer et al. 2019". — Same. — fetched directly.

**moor-2020-topological-autoencoders** (`1906.00722`, cs.LG, ICML 2020) — [V] — Owns the
topological autoencoder loss (preserve the input's persistence in the latent space). This is
the paper the record *meant* by "Moor et al. 2020". — Same. — fetched directly.

**zeng-2021-topological-attention** (`2107.09031`, cs.LG) — [V] — Owns attention *over*
local persistence features of a time series. — Persistence is the input to attention, not a
reading of attention's influence; no certificate. — query 2.

**faghihi-2026-topology-aware-attention** (`2605.03163`, cs.LG) — [V] — Owns persistence
(H0–H2) and anchored Euler-characteristic biases *added to attention logits* (exact PH under
a 28-point cap, smooth surrogates otherwise, per the search snippet [U]). — The nearest
"topology inside the logits" paper; it biases a softmax, does not certify a mask and has no
resolvent or boundary. — query 2, 4.

### D. Sparse / masked attention from graph or cluster structure

**reid-2024-topological-masking** (`2410.03462`, cs.LG) — [V] — Owns learnable *topological
masks* (functions of a weighted adjacency) approximated by graph random features in linear
time, with concentration bounds. — "Topological" here means graph-derived relative position
weighting; the mask is a bias, not an exact block certificate; no persistence. — query 3.

**cho-2022-sbm-attention** (`2210.15541`, cs.LG) — [V] — Owns per-head *sampled* attention
masks from a mixed-membership stochastic block model over node and cluster embeddings. —
**Nearest existing thing to a Mapper cover as candidate builder**: overlapping cluster
membership decides who attends to whom, per input. What is open: the cover is sampled, not
built from a lens with a Reeb-graph guarantee; no CSR schedule with a certificate; no
resolvent. — query 3.

**roy-2021-routing-transformer** (`2003.05997`, cs.LG, TACL) — [V] — Owns online k-means
routing of queries to key clusters. — Non-overlapping clusters; no cover; no certificate. —
fetched directly.

**kitaev-2020-reformer** (`2001.04451`, cs.LG, ICLR 2020) — [V] — Owns LSH bucketing as the
candidate builder. — Same. — fetched directly.

**yuan-2025-nsa** (`2502.11089`, cs.CL) — [V] — Owns the three-branch hierarchy
compression / selection / sliding window, hardware-aligned and trainable. — The nearest
owner of "fine-near / coarse-far" as a trained sparse attention; the record's
`ceq/multizoom.py:1-40` carries a coarsening *bound* rather than a learned selector, which is
the narrow delta and is not topological. — fetched directly.

**yang-2026-boundary-repair** (`2606.02680`, cs.LG) — [V] — Owns Theorem 1 (fixed-block
reachability barrier): under a fixed block-causal mask with only positionwise non-attention
ops, a token's reachable set stays inside its block's causal prefix at every depth; and
"Boundary Bridge Attention" as the repair. — **This is the converse of (a)**: it proves that a
block mask confines reach; the shape's certificate says a *zero gate* makes the resolvent
*exactly* block-diagonal. Neither implies the other; the paper is cited before the phrase
"block structure ⇒ no cross-block influence" is used. — query 14.

### E. Gates, segmentation, block structure

**lin-2025-forgetting-transformer** (`2503.02130`, cs.LG, ICLR 2025) — [V], HTML full text
also fetched — Owns the forget gate in softmax attention: `f_t = σ(w_fᵀx_t + b_f)`,
`F_ij = Π_{l=j+1}^{i} f_l`, computed as `D_ij = c_i − c_j` with `c` the cumulative sum of
`log f`, added to the logits (the paper's own equations, ≤ 15 words quoted in the bib note).
— **This is exactly the exponential-prefix-scan form that `no_prefix_scan_represents_a_zero_gate`
(`V16Domain.lean:165`) proves cannot carry a zero gate.** The paper's convention `log 0 = −∞`
is a limit, and the fetched text does not discuss an exact-zero gate or a block boundary. So
FoX owns the *mechanism* (a data-dependent cumulative decay on softmax logits); the record
owns the theorem that the mechanism's computational form has no exact zero. Narrow delta,
stated in those words. — query 7.

**yang-2023-gla** (`2312.06635`, cs.LG) — [V] — Owns gated linear attention with a
data-dependent gate on the 2-D state and a chunkwise-parallel training form. — A gate of `0`
resets the state, which *is* a segmentation in practice; no theorem, no certificate, and the
linear-attention corner is `β = 0` of the record's family (`V16Domain.gate_zero_beta_zero_is_linear_attention`,
brief §2). — query 7.

**hwang-2025-hnet** (`2507.07955`, cs.LG) — [V] — Owns learned, content-dependent chunk
boundaries trained end to end (H-Net). — The abs page makes no exactness or certificate
claim for the segmentation; open. — fetched directly.

**bathula-2026-gating-curvature** (`2604.14702`, cs.LG) — [V] — Owns the claim that
multiplicative gating lets attention realise non-flat decision boundaries that ungated
attention cannot. — Relevant only to the *phrase* "decision boundary inside attention"; not
a committor, not a level set of a harmonic function. — query 14.

**aguilar-2026-enclosed-mode** (`2608.28541`, cs.LG) — [V] — Owns "topology relative to
reach": a certified world model can be correct on the reachable region and wrong elsewhere;
a persistent-homology summary (`β₁`) and a "gate quotient" reachability framework;
planners exploit hidden channels. — **The closest existing combination of reachability
certificate + persistence + a decision-maker.** Not attention, not a resolvent, not an
absorbing chain. Watch it; cite it in §7 of the paper when "certified reach" is said. —
query 14.

**makkuva-2024-attention-with-markov** (`2402.04161`, cs.LG, ICLR 2025) — [V] — Owns the
analysis of transformers trained on first-order Markov chains (single-layer traps at unigram;
induction heads at depth ≥ 2). — Markov chains as *data*; nothing about a chain inside the
operator, committors or absorption. Cited as the nearest "Markov chain and attention" owner
so that (c) can be stated as NOT FOUND against it. — query 13.

### F. Mapper, Reeb graphs, persistence stability, directed persistence, foundations

**singh-2007-mapper** (DOI `10.2312/SPBG/SPBG07/091-100`, SPBG 2007) — **[U]**: DOI landing
403, Crossref 404 on this DOI form; dblp `conf/spbg/SinghMC07` and Semantic Scholar
CorpusId 5703368 both fetched and agree on title, authors, year, DOI. — Owns Mapper: a lens,
an overlapping cover of its range, partial clustering per cover element, the nerve graph. —
The shape's (d) uses exactly this object; the delta can only be *where the cover goes* (into a
CSR schedule feeding a resolvent read) and *what the lens is* (a committor or influence
score). — seed; query 17.

**carriere-2018-mapper-statistics** (`1706.00204`, cs.CG, JMLR 19) — [V] — Owns Mapper as an
optimal estimator of the Reeb graph, with parameter selection and confidence regions. — This
is the paper that lets (d) fix its cover parameters *before* seeing a bed (against M-2). —
query 6.

**cohen-steiner-2007-stability** (DOI `10.1007/s00454-006-1276-5`, DCG 37(1) 103–120) — [V]
via Crossref — Owns bottleneck stability of persistence diagrams under sup-norm perturbation
of the filtering function. — The δ in any "barcode with a printed δ" (L-CERT) is this
theorem's constant; the influence graph is *directed*, which this theorem does not cover
(see Turner, Chowdhury–Mémoli). — seed.

**chazal-2012-structure-stability** (`1207.3674`, math.AT) — [V] — Owns the
interleaving-distance form of stability for persistence modules over ℝ under weak finiteness.
— The form used by `ceq/certs/topological.py:476` (`2ε`-interleaving). — seed.

**turner-2019-quasimetric-rips** (`1608.00365`, math.AT, AGT 19) — [V] — Owns Rips-type
filtrations for *asymmetric* functions and quasi-metrics with bottleneck stability. — The
influence graph `∂z_i/∂x_j` is asymmetric; this is the stability result (b) would have to
invoke, not Cohen-Steiner's. — fetched directly.

**chowdhury-2016-dowker** (`1608.05432`, math.AT) — [V] — Owns Dowker persistence for
asymmetric networks with a functorial Dowker theorem. — Same role. — fetched directly.

**chowdhury-2017-path-homology** (`1701.00565`, math.AT) — [V] — Owns persistent *path*
homology of digraphs (direction-sensitive). — Same role; `β₀` of a digraph under path
homology is weak connectivity, which matters for how (b) is defined. — fetched directly.

**reimann-2017-directed-flag** (DOI `10.3389/fncom.2017.00048`) — [V] via Crossref — Owns
the directed flag complex of a directed network and its Betti numbers as a structure-function
link. — Same role; no learning. — fetched directly.

**carlsson-2009-topology-and-data** (DOI `10.1090/S0273-0979-09-01249-X`, BAMS 46(2)) — [V]
via Crossref — Owns the survey framing of persistence and Mapper. — Foundational citation. —
fetched directly.

**edelsbrunner-2009-computational-topology** (DOI `10.1090/mbk/069`) — [V] via Crossref —
Owns the textbook Euler–Poincaré and Morse-inequality material the record's `morse_census`
reports. — Foundational. — fetched directly.

**milnor-1963-morse-theory** (DOI `10.1515/9781400881802`) — [V] via Crossref — Owns Morse
theory. — Foundational. — fetched directly.

**perea-2015-sliding-windows** (`1307.6188`; DOI `10.1007/s10208-014-9206-z`, FoCM 15(3)
799–838) — [V] both — Owns persistence of sliding-window (delay) embeddings as a periodicity
score with convergence guarantees. — The `stratum` regime check in `THEORY.md:141-150` is
this object; the shape's next-transient-state prediction can *use* it as a regime detector
but it is not an attention operator. — seed.

**levin-2017-markov-mixing** (DOI `10.1090/mbk/107`) — [V] via Crossref; **Theorem 13.10 read
on p. 183 of the authors' PDF** (RUN: `pypdf` text extraction, page index 198): for a
reversible `P` with spectral gap `γ = 1 − λ₂` and bottleneck ratio `Φ⋆`,
`Φ⋆²/2 ≤ γ ≤ 2Φ⋆`, attributed to Sinclair–Jerrum (1989) and Lawler–Sokal (1988). — Owns the
Cheeger inequality in the form the record cites. — **Hypothesis: reversible.** The shape's `P`
is causal (strictly lower-triangular off the diagonal blocks, `I5` in the brief), hence not
reversible unless made so; §4.5. — seed; query 9.

**kim-2026-topological-causal-effects** (`2603.02289`, stat.ME, ICLR 2026) — [V] — Owns the
*causal estimand* "difference between potential outcomes measured through persistence
summaries" (power-weighted silhouettes). — **This owns "an intervention changes the
barcode" as a statistical object.** The shape's `Δz = z(do a) − z` read through a β₀ barcode
would be an instance and must cite it. Open: the intervention here is on exogenous
treatment, not on a token in a context; no operator. — `results/r9_maths_survey.md:314,503`;
fetched here.

### G. Committors, isocommittor surfaces, transition-path theory

**e-2010-transition-path-theory** (DOI `10.1146/annurev.physchem.040808.090412`) — [V] via
Crossref — Owns transition-path theory: the committor as the backward-Kolmogorov boundary
value problem, isocommittor surfaces, reactive flux. — The shape's `q^{(k)} = (I − Q)⁻¹R_k 1`
is the discrete instance (`ceq/beds/bed_1.py:188`, brief I3); TPT owns the object, the
record owns the finite-chain reading as a resolvent with absorbing rows. — seed area.

**vanden-eijnden-2009-string-method** (DOI `10.1063/1.3130083`, JCP 130, 194103) — [V] via
Crossref — Owns isocommittor surfaces as the objects a string method sweeps (reaction tubes).
— Same. — fetched directly.

**khoo-2018-committor-nn** (`1802.10275`) — [V] — Owns the variational neural committor. —
Regresses `q` from molecular configurations; the record's `PRIOR_ART.md:637` NOT FOUND for
the fundamental-matrix label stands. — query 5; already [V] in the record.

**li-2021-semigroup-committor** (`2012.06727`, math.NA, MSML 2021) — [V] — Owns the
semigroup (integral) formulation with boundary conditions built in. — Same. — query 5.

**chen-2025-committor-flow** (`2507.21961`, physics.comp-ph) — [V] — Owns strings lying on
isocommittor surfaces, iterated with a learned committor. — Same; no attention. — query 5.

### H. The author's own merged upstream work

**sharma-2026-mujoco-3396** — [V] PR page — merged 2026-07-20; touches `engine_island.c/.h`
and disjoint-set-forest helpers; the PR description references the S²–Vietoris–Rips
corpus. `ceq/rips.py:3-9` (READ) records that the Rips generator file was added and removed
inside the PR and is not in the merge commit. — What merged is a linear-memory island
discovery, not the corpus; the corpus lives in this repository. — fetched directly.

**sharma-2026-kernels-22** — [V] PR page and files tab — merged 2026-07-28; files
`kernels/topology_sparse_attention.py`, `test/test_topology_sparse_attention.py`,
`benchmarking/topology_sparse_attention.py` (+146), `benchmarking/benchmark_utils.py`
(+8/−1), `kernels/__init__.py` (+10); a forward-only Triton kernel consuming a causal CSR
block schedule built from sink blocks, local-window blocks and a persistence-style salience
over key-block centroids; the PR page states up to 3.48× against dense CSR on its measured
sequences (the PR's number, not re-measured here). — **The author already occupies
"persistence-derived block schedule consumed by an attention kernel."** The delta for (d) is
therefore not the schedule but the *cover* (Mapper/Reeb, overlapping) and the *consumer* (a
resolvent read with a Neumann certificate, brief I4), and `THEORY.md:223` already records that
the kernel is forward-only. — fetched directly.

---

## 3. Verdicts on the four components

### (a) Zero-gate segmentation as an exact block certificate

- OCCUPIED — the *mechanism* (a data-dependent multiplicative gate on the path from `j` to
  `i`, a small gate cutting the sequence): **lin-2025-forgetting-transformer** for softmax,
  **yang-2023-gla** for linear attention, **hwang-2025-hnet** for learned chunk boundaries.
- OCCUPIED — the *converse* statement (a block mask confines reach at every depth):
  **yang-2026-boundary-repair** Theorem 1.
- OWNED BY THE RECORD, NOT FOUND OUTSIDE IT — the *iff* certificate: `pathProd = 0 ⇔ ∃ k:
  m_k = 0` (`V16Domain.lean:129`) and the impossibility of carrying it in any exponential
  prefix scan (`:165`). FoX's own computation is `D_ij = c_i − c_j` with `c = cumsum log f`,
  which is precisely the form the theorem excludes at `f = 0` (FoX writes `log 0 = −∞` as a
  convention and does not treat the exact zero).
- DERIVED, elementary and not anyone's to own: if `P` is block-diagonal (after the gate zeroes
  every cross-block entry) then every power `P^k` is block-diagonal and so is
  `Σ_k (γP)^k = (I − γP)⁻¹`; hence the read `P(I − γP)⁻¹V` is exactly block-diagonal. The
  paper states this as a one-line proposition and does not call it new.
- Delta, narrowly: "a machine-checked iff certificate that a zero gate yields an exactly
  block-diagonal resolvent, and a proof that the cumulative-log-gate implementation family
  (FoX, GLA chunk forms) cannot represent that zero."

### (b) β₀ persistence of the influence graph across a threshold as a causal-structure barcode

- OCCUPIED — barcodes / Betti numbers of *attention-weight*-thresholded graphs as descriptive
  features: **kushnareva-2021-tda-attention**, **cherniavskii-2022-acceptability**,
  **kushnareva-2022-betti**, **perez-2022-topological-bert**; across layers by zigzag:
  **samaga-2026-halluzig**; as a two-subgraph divergence: **bazarova-2025-toha**.
- OCCUPIED — "an intervention changes the persistence summary" as a causal estimand:
  **kim-2026-topological-causal-effects**.
- OCCUPIED — persistence of *directed* networks and its stability: **turner-2019**,
  **chowdhury-2016-dowker**, **chowdhury-2017-path-homology**, **reimann-2017**.
- OCCUPIED — training through a persistence layer: the corrected M10 lineage
  (**gabrielsson-2019**, **carriere-2019-perslay**, **carriere-2021-optimizing-ph**,
  **hofer-2019**, **moor-2020**, **corcoran-2020**).
- NEAR-MISS — persistence *inside the logits* as a bias: **faghihi-2026**.
- NOT FOUND — a filtration by the *influence Jacobian of a resolvent read* (signed or
  absorbing), read as a certificate with a printed δ from a directed stability theorem, whose
  ε = 0 endpoint is the exact segmentation of (a). Queries 4, 8, 16.
- Delta, narrowly: the filtering function. Everything else in the sentence has an owner.

### (c) Isocommittor surfaces as decision boundaries inside an attention model

- OCCUPIED — isocommittor surfaces and the committor as a boundary-value problem:
  **e-2010**, **vanden-eijnden-2009**; neural committors: **khoo-2018**, **li-2021**,
  **chen-2025** (strings on isocommittor surfaces).
- OWNED BY THE RECORD — the `q = 1/2` guard as the action vocabulary of a bed
  (`ceq/beds/bed_1.py:5-8, :382`), the committor as a resolvent read with absorbing rows
  (brief I3; `MATHEMATICS.md` §7, §11).
- NEAR-MISS in phrase only — **bathula-2026-gating-curvature** ("decision boundaries" that
  gating enables in attention).
- NOT FOUND — a committor computed *by* the attention operator (an absorbing set inside the
  context, `q = (I − Q)⁻¹R1` as the read) and its `q = 1/2` level set used as the decision
  boundary of the model. Queries 5, 13; the record's own `PRIOR_ART.md:637` agrees.
- Delta, narrowly: the absorbing rows are positions of the context; the committor is the read.

### (d) A Mapper cover as the long-context candidate builder for a resolvent read

- OCCUPIED — Mapper: **singh-2007-mapper** [U-landing/V-records]; Mapper as a Reeb-graph
  estimator with pre-fixable parameters: **carriere-2018-mapper-statistics**.
- OCCUPIED — cluster covers as who-attends-to-whom: **cho-2022-sbm-attention** (overlapping,
  sampled), **roy-2021-routing-transformer** (k-means), **kitaev-2020-reformer** (LSH);
  hierarchical compress/select/window: **yuan-2025-nsa**; graph-derived masks:
  **reid-2024-topological-masking**.
- OCCUPIED BY THE AUTHOR — persistence-derived CSR block schedule consumed by a Triton
  kernel: **sharma-2026-kernels-22** (merged); the do-nothing schedule as the standing
  control (`THEORY.md:162-165`).
- NOT FOUND — a Mapper (lens + overlapping cover + nerve) used to build the candidate set of
  a *resolvent* read, with the candidate set carrying a truncation certificate (brief I4) and
  the cover parameters fixed by the Reeb-estimator rule. Queries 6, 15.
- Delta, narrowly: the cover's construction (nerve of a lens cover, not a sample or a
  hash) and its consumer (a certified resolvent, not a softmax).

---

## 4. Surprises (flagged loudly)

### 4.1 The record's M10 citations are all three misattributed (P-class defect)

`CEQ_V20_R15_CONTRACT.md:223-224` reads "(1905.12200, 1904.09378, 2011.05804 — occupied)"
under the names Hofer 2019, Moor 2020 and Carrière 2021 (as relayed in this planet's brief).
Fetched this session:

| id in the record | resolves to [V] | the record meant | correct id [V] |
|---|---|---|---|
| `1905.12200` | Brüel-Gabrielsson, Nelson, Dwaraknath, Skraba, Guibas, Carlsson — *A Topology Layer for Machine Learning* | Hofer et al. 2019 | `1906.09003` |
| `1904.09378` | Carrière, Chazal, Ike, Lacombe, Royer, Umeda — *PersLay* | Moor et al. 2020 | `1906.00722` |
| `2011.05804` | Corcoran, Deng — *Regularization of Persistent Homology Gradient Computation* | Carrière et al. 2021 | `2010.08356` |

The lineage verdict "occupied" survives (all six papers are differentiable-persistence
papers), but the paper must cite the six correct pairs. Mechanism this is filed against:
`MISTAKES.md` P-5 (doc rot pointing at nothing) and P-10 (a source cited without its
content read); the brief's rule 3 ("a fabricated citation is the one defect that ends the
paper") makes this the first correction the assembler applies.

### 4.2 A title already claims Betti numbers of attention graphs are "all you need"

**kushnareva-2022-betti** (`2207.01903`). It is a preprint the abs page marks as not
accepted, but it is prior art for the *descriptive* β₀/β₁ reading and must be cited before
(b) is stated.

### 4.3 "Intervention ⇒ barcode change" already has a causal-inference owner

**kim-2026-topological-causal-effects** (ICLR 2026). The shape's consequence channel, if it
is ever *read through persistence*, is an instance of this estimand and says so.

### 4.4 Attention as a connection walk (sheaf view of the *same* `P`)

**lin-2026-connection-laplacian-attention** identifies single-head attention with a
connection walk and multi-head attention with mixed transports, and states the
stochastic/reversible conditions under which the generator is a random-walk connection
Laplacian. It contains no resolvent, boundary or barcode (abs page). It occupies the
*operator identification*, which the paper must acknowledge if `P` is described in sheaf
or connection language (`THEORY.md:24, :188` do so).

### 4.5 The Cheeger inequality the record cites needs reversibility

Levin–Peres Theorem 13.10 (read on p. 183, 2nd ed.) is stated for a *reversible* `P`. The
shape's `P` is causal and, by brief I5, a triangular operator; it is not reversible.
`MATHEMATICS.md:455` ("`g ≤ 2φ`") must either (i) restrict to a symmetrised or lazy
reversible surrogate and say so, or (ii) cite a non-reversible bottleneck bound instead.
Mechanism: `MISTAKES.md` V-25 (a theorem whose hypothesis no draw satisfies) and P-10.

### 4.6 A near-miss to watch, not an occupant

**aguilar-2026-enclosed-mode** combines a reachability certificate, a `β₁` persistence
summary and a planner exploiting hidden channels. It is the only fetched paper that puts
"certified reach" and persistence in one frame with a decision-maker. It is not attention
and not a resolvent; it does not occupy the delta. It is cited in §7 so that "certified
reach" is not presented as unowned.

**No paper fetched or indexed occupies the whole delta** (resolvent read + absorbing
constraint rows + influence-filtered β₀ barcode with δ + Mapper cover as certified
candidate builder, in one attention operator).

---

## 5. NOT FOUND (with the queries that failed to find them)

| component | queries | nearest thing found |
|---|---|---|
| Exact iff certificate that a zero gate gives an exactly block-diagonal resolvent / attention read, outside the record's Lean | 7, 14 | FoX (`2503.02130`) mechanism; boundary-repair Theorem 1 (`2606.02680`) converse |
| A persistence filtration by the influence Jacobian of an attention or resolvent read, with a stability δ from a directed-network theorem | 4, 8, 16 | attention-weight filtrations (`2109.04825` lineage); zigzag over layers (`2601.01552`) |
| A committor / isocommittor level set computed by the attention operator itself with absorbing positions in the context | 5, 13 | neural committors on molecular configurations (`1802.10275`, `2012.06727`, `2507.21961`); Markov chains as data (`2402.04161`) |
| Mapper / Reeb-graph cover as the candidate builder for any attention read | 6, 15 | SBM-sampled overlapping masks (`2210.15541`); k-means / LSH clusters (`2003.05997`, `2001.04451`) |
| Mapper cover → CSR block schedule → resolvent read with a Neumann truncation certificate | 6, 15, 2 | the author's own `kernels#22` (0D-salience schedule → softmax kernel, forward-only) |
| "Sheaf Attention Networks" on arXiv or in dblp | 11, 17 | resolved on OpenReview (`LIDvgVjpkZr`) instead |

Each row is an absence in the indexed literature as of 2026-09-03 under the listed queries,
paid for by the planted positives in §1.1. None is written as "novel".

---

## 6. Design against the taxonomy (for the proposals this lineage feeds)

Each proposal the paper makes from this lineage names the `MISTAKES.md` mechanism it is
designed against (brief §5.6). Line numbers are the headings in `MISTAKES.md`.

1. **Correct the M10 ids before any citation is typeset** — against P-5 (`:343`) and P-10
   (`:1382`); the bib in this directory carries both the resolved and the intended ids.
2. **(a) The block certificate bind** — plant a gate at `1e-300` (nonzero) and require that
   the read is *not* block-diagonal (rejection region non-empty, V-24 `:1658`); run a domain
   census showing at least one draw with `m_k = 0` exactly (V-25 `:1954`); assert against a
   dense `(I − γP)⁻¹` solve with `torch.equal` on the zeroed blocks, not a tolerance
   (V-3 `:72`: the identity is the theorem's, not the test author's).
3. **(b) The influence barcode** — the threshold grid and the δ are fixed from Turner's
   stability constant and the sampling scale *before* the barcode is read (M-2 `:451`,
   following `ceq/certs/topological.py:47-56`); a null barcode from row-permuted influence
   at the same logit scale is reported beside every measured one (M-15 `:1289`); a barcode
   the instrument cannot compute (non-integer β₀, aliasing) is a refusal, not a pass
   (V-16 `:828`); the descriptive owners in §2.A are cited *before* the filtration is named
   (brief rule 7).
4. **(c) The committor read** — the oracle committor runs on the latent environment chain and
   the arm never sees `Q` or `R` (D-2 `:710`); the label is a vector over candidate moves, not
   a scalar at one position (D-1 `:677`); a single absorbing set makes the label constant
   (V-12 `:189`), so `K ≥ 2` constraint sets are required by construction.
5. **(d) The Mapper cover** — cover parameters fixed by the Reeb-estimator rule
   (`carriere-2018`) on a held-out relation, never on the bed (M-2); the do-nothing schedule
   (the merged 0D-salience builder) always entered (`THEORY.md:162-165`; V-9 `:154`: a repair
   that changes nothing must be a first-class outcome); every candidate set carries the
   Neumann δ of brief I4 (L-CERT) and the coarsening bound of `ceq/multizoom.py`.
6. **The Cheeger sentence** — restate with the reversibility hypothesis or replace
   (V-25, P-10); no `g ≤ 2φ` on a causal `P` without a stated symmetrisation.
7. **This sweep itself** — the NOT FOUND rows are paid for with planted positives
   (V-7 `:117`); every [V] carries the date and the page type fetched; every remembered venue
   is marked [U] in the bib note rather than silently printed (P-1 `:289`).

---

## 7. Limits

The sweep reads abs pages, DOI metadata records and one full text (FoX); it does not read
the body of the other 57 sources, so every "leaves open" clause is a statement about what the
*abstract* claims and the paper may contain more. Venues remembered rather than fetched are
marked [U] inside the bib notes and are not load-bearing. The Mapper 2007 entry is [U] on the
landing page only; its DOI, title and authors are confirmed by two independent bibliographic
records. The search index is one engine on one day (2026-09-03); the queries are recorded so
the absence claims can be re-run. The `kernels#22` speed figure (3.48×) is the PR page's own
number and was not re-measured. No code was written or run beyond a read-only `pypdf` text
extraction of the Levin–Peres PDF and the repository `grep`/`sed` reads listed in §1.3.
