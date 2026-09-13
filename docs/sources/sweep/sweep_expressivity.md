# SWEEP — expressivity, depth laws, optimality of the softmax role

JUPITER-S3, 2026-09-03. Companion file: `bib_expressivity.bib` (45 entries, all `[V]` for
identifier + title). Repository at `<repo root>`, branch `v17k-gate0`,
HEAD `207e7b9`. Evidence classes per `BRIEF.md` §5.3: `RUN` / `READ path:line` / `CITED [V]|[U]` /
`DERIVED`. `[V]` means the arXiv abs page (or the Crossref record for a DOI) was fetched this
session and the title matched; `[U]` means reached only through a search snippet, a summary
model's reading of a page, or memory, and is labelled as such wherever it appears.

The question this sweep was asked to answer, verbatim from the dispatch: *exactly which theorem
licenses "depth-1 softmax cannot compute a t\*-hop reachability/committor that the resolvent computes
in one operator", under what conditions, and what is conditional.* §3 answers it in five layers.
§4 carries the SURPRISE: the shape's operator `P (I − γP)^{-1} V` is published, with the same
triangular solve and the same depth-law motivation, as **ChaCAL** (Fagnou, Caillon, Delattre,
Allauzen, EMNLP 2024, arXiv:2410.05565) — a paper the repository never cites (`grep` 2026-09-03:
`0` files outside `kaggle/`).

---

## 0. Method and the search log

**Seeds fetched (from the dispatch):** 2402.09268, 2405.18512, 2402.08164, 2410.01537, 2509.21936,
2210.10749, 2306.13596, 1906.06755, 2207.00729, 2106.16213, 2311.00208, Barrington 1989 (DOI),
Krohn–Rhodes 1965 (DOI). Every seed identifier survived the fetch; the brief's `VERIFY` on
Peng–Narayanan–Papadimitriou is closed: **arXiv:2402.08164 is correct** (`[V]`, v1 13 Feb 2024, v2
26 Feb 2024, stat.ML).

**Searches run (WebSearch, 2026-09-03), beyond the seeds — 16, of which the dispatch required
at least six:**

| # | query | what it surfaced (ids later fetched `[V]`) |
|---|---|---|
| S1 | `Sanford Fatemi Hall Tsitsulin "Understanding Transformer Reasoning Capabilities via Graph Algorithms" arXiv` | 2405.18512 |
| S2 | `k-hop induction heads depth lower bound transformer 2024` | 2408.14332, 2505.23683, 2410.01686, 2407.17686 |
| S3 | `unconditional lower bound multi-layer transformer sequential function composition polynomial width Chen Peng Wu 2024` | 2412.02975, 2501.12997, 2503.22076 |
| S4 | `Yehudai depth-width tradeoffs algorithmic reasoning transformers k-hop 2025` | 2503.01805 |
| S5 | `Strassen attention function composition single layer transformer 2025 arXiv` | 2501.19215 |
| S6 | `attention layer matrix inverse resolvent multi-hop one layer "Neumann series" attention expressivity arXiv` | 2603.00175, 2009.14332, **2605.22476** |
| S7 | `transformer expressivity committor absorption probability Markov chain attention layer computes hitting probability` | 2402.04161, 2508.07208 (not fetched), 2510.18638 (not fetched) |
| S8 | `Merrill Sabharwal "A little depth goes a long way" log-depth transformers graph connectivity regular languages arXiv` | 2503.03961 |
| S9 | `attention as absorbing Markov chain "fundamental matrix" transformer layer arXiv` | 2603.00175 only |
| S10 | `graph diffusion attention Katz PageRank powers of attention matrix in one transformer layer multi-hop` | 2210.11794, 2303.00613, 2009.14332; "ParaFormer" (no id surfaced; see §2.C) |
| S11 | `softmax attention Bayes optimal theorem one layer in-context learning why softmax versus linear attention 2025` | 2512.11784, 2506.01582 |
| S12 | `one-layer transformer lower bound two-hop composition "grandparent" induction head Bietti "Birth of a Transformer"` | 2306.00802, 2408.14332, 2503.22076 |
| S13 | `transformers constant depth cannot solve directed graph reachability unless TC0 equals NL Merrill Sabharwal` | 2503.03961, 2503.01805, 2402.09268 |
| S14 | `counterfactual intervention do-operator attention mechanism expressivity theorem transformer causal consequence prediction 2025` | nothing in the expressivity lineage (applied causal-transformer papers only) |
| S15 | `attention head computes argmin over candidate actions of maximum absorption probability constraint sets safest action Markov chain transformer` | 2507.17657; no result matching the safest-move object |
| S16 | `matched-depth comparison single-layer resolvent attention versus deeper softmax stack exactness certificate multi-hop reachability transformer` | 2605.22476, 2404.01601; no matched-depth comparison paper |
| S17 | `"Chain and Causal Attention" Efficient Entity Tracking Fagnou Caillon Allauzen arXiv EMNLP 2024` | **2410.05565** (search index also printed 2410.05573 — the abs fetch confirms 2410.05565) |
| S18 | `attention "(I - A)^{-1}" OR "resolvent" causal attention matrix single layer arbitrary number of hops entity tracking Boxes transformer` | 2410.05565, 2605.22476 |

**Full-text reads (arXiv HTML, for theorem statements, not just titles):** 2402.09268, 2402.08164,
2410.01537, 2509.21936, 2405.18512, 2503.03961, 2207.00729, 2408.14332, 2501.19215, 2412.02975,
2603.00175, 2605.22476, 2410.05565. The theorem numbers quoted below come from those reads; where a
number or a detail is the summariser's reading rather than a displayed equation, it is marked `[U]`.

**Repository reads that anchor the sweep:** `BRIEF.md` §1 (the shape, I1–I5), `MATHEMATICS.md:284-306`
(§2 depth law), `MATHEMATICS.md:106-128` (§0.3 the narrow claim), `MATHEMATICS.md:488-521` (§9 the
novelty position, including *"no unconditional lower bound against one-layer softmax on this product
chain"*), `MATHEMATICS.md:1044-1057` (§17.5 erf-not-softmax, `L/d = 4.00`), `PRIOR_ART.md:1-110`
(Liu et al. delta, Theorem 4 quoted, `O(|Q|² log |Q|)` depth), `PRIOR_ART.md:585-610` (Sanford
NeurIPS 2024 as occupant of the capability hierarchy), `DONE.md:786-803` (the record's earlier
reading of Sanford as conditional and "pointer-chasing rather than this product chain"),
`DONE_ARCHIVE_ROUND1.md:672-681` and `:1312-1324` (APPNP, GDC, MAGNA, InfSA, ParaFormer, Gated
DeltaNet — all rejected during the *signed* programme for being non-negative), `THEORY.md:22,51,187`
(the resolvent `(I − γP)⁻¹` as "successor operator"), `MISTAKES.md:677-690` (D-1),
`ceq/attention.py:1-60` (the tier table), `CEQ_V16_CONTRACT.md:209` (R-SKY).

---

## 1. Per-source ledger — depth laws and composition lower bounds

Format: **id** · class · *owns* (the equation or mechanism, in this sweep's words) · *leaves open*
relative to the shape in `BRIEF.md` §1 · *found by*.

### 1.1 Sanford, Hsu, Telgarsky — *Transformers, parallel computation, and logarithmic depth* — arXiv:2402.09268 · `[V]`
- **Owns.** The `hop_k` task (`hop_k(X)_i = X_{find^k_X(i)}`, k-fold last-occurrence pointer chasing);
  **Thm 4.2** (upper): a causally-masked transformer computes `hop_k` at depth `L = ⌊log₂ k⌋ + 2`,
  `m = O(1)`, `H = 1`; **Cor 4.3** (lower): *assuming Conjecture 2.4 (one-versus-two-cycle in the
  (γ,δ)-MPC model)*, for `k = Θ(N^ξ)`, `ξ ∈ (0, 1/2]`, every transformer computing `hop_k` with
  `mH = O(k^{1−ε})` needs `L = Ω(log k)`. **Thm 3.1 / 3.4**: two-way simulation between depth-`L`
  transformers and `O(L)`-round MPC at `p = Θ(log N)` bits. **Cor 3.3 / 3.5**: connected components
  at `L = O(log D)`, `m = O(N^ε)`; conditionally optimal.
- **Leaves open.** (i) The lower bound is conditional and asymptotic in `N` with `k` polynomial in
  `N`; it says nothing at `s = 64`, `k = t* ≤ 8`, `d = 16`. (ii) It bounds *standard* transformer
  layers; a layer that is a triangular solve is outside the model, so the theorem neither licenses
  nor forbids the resolvent — it is silent. (iii) `hop_k` is pointer chasing in a *token-defined*
  graph; the shape computes reachability in the graph *the layer itself computes* (`P`), which is a
  different object (the record already said this: `DONE.md:802`, *"pointer-chasing rather than
  this product chain"*).
- **Found by.** Seed; theorem statements from the HTML read.

### 1.2 Sanford et al. — *Understanding Transformer Reasoning Capabilities via Graph Algorithms* — arXiv:2405.18512 · `[V]`
- **Owns.** The nine-problem hierarchy: **Thm 5** retrieval tasks (node/edge count, edge existence,
  degree) at depth 1 with `m = O(log N)`; **Thm 2/18** parallelizable tasks (connectivity, cycle
  check, bipartiteness) at `L = O(log N)`, `m = O(N^ε)`; **Thm 3/19** `L = Ω(log N)` for
  parallelizable tasks with `mH = O(N^ε)`, *conditional on Conjecture 13* (the same one-vs-two-cycle
  conjecture); **Thm 4** search tasks (shortest path, diameter) at `m = O(N^{1/2+ε})`.
- **Leaves open.** Same three gaps as 1.1. Additionally: the hierarchy places *undirected
  connectivity* in "parallelizable"; the committor is an *absorption probability* — a weighted,
  directed, quantitative reachability — which the hierarchy does not name. NOT FOUND: any class in
  that hierarchy for "absorption probability into one of K sets".
- **Found by.** S1 (the record cited it by OpenReview id only, `PRIOR_ART.md:593-596`).

### 1.3 Sanford, Hsu, Telgarsky — *One-layer transformers fail to solve the induction heads task* — arXiv:2408.14332 · `[V]`
- **Owns.** **Thm 1** (unconditional): a one-layer *softmax* transformer with `h` heads, embedding
  `m`, `p` bits, solving induction heads (= `hop_1`, the 2-hop pointer) at length `n` over a
  three-symbol alphabet has `h·m·p = Ω(n)`; proof by reduction from INDEX. Two-layer upper bound
  `h = O(1)`, `m = O(1)`, `p = O(log n)`.
- **Leaves open.** This is the cleanest *unconditional* one-layer statement in the lineage, and it
  is exactly the `t* = 2` case. It is asymptotic in `n`; at the record's `s = 64`, `d = 16`,
  float32 (`p = 32`), `h = 1`: `h·m·p = 512 ≥ 64` — **the hypothesis is not violated, so the theorem
  says nothing at that geometry** (the `V-25` mechanism: a theorem whose hypothesis no draw satisfies).
  The paper must say so.
- **Found by.** S2, S12.

### 1.4 Sanford, Hsu, Telgarsky — *Representational Strengths and Limitations of Transformers* — arXiv:2306.02896 · `[V]`
- **Owns.** Sparse averaging (attention needs `log`-scaling embedding; RNN/FFN polynomial); triple
  detection (Match3) needs width linear in `n` for one attention layer; the "attention is a
  pairwise object" limitation in its earliest clean form.
- **Leaves open.** Pairwise-vs-triple is about *one* layer's arity, not about hops; not the shape's
  question. Cited as the origin of the Match3 lower-bound line continued by 1.7.
- **Found by.** Memory of the lineage; abs fetched.

### 1.5 Peng, Narayanan, Papadimitriou — *On Limitations of the Transformer Architecture* — arXiv:2402.08164 · `[V]`
- **Owns.** **Thm 1** (unconditional, communication complexity): a single `H`-head softmax layer with
  embedding `d` and `p`-bit precision errs on function composition (`f(g(x))` with `n`-element
  domains) with probability `≥ R/(3 n log n)` where `R = n log n − H(d+1)p > 0`. **Thm 2**:
  `Ω(√(n/(Hdp)))` chain-of-thought steps for iterated composition. **Thm 3**: Derivability, 2-SAT,
  Horn-SAT, circuit evaluation not solvable by multi-layer transformers unless `L = NL` (resp. `L = P`).
- **Leaves open.** The identifier is confirmed; the brief's obstruction 3 may drop its `[U]`. The
  condition `n log n > H(d+1)p` at the record's geometry: `n = 64` gives `64·6 = 384` against
  `H(d+1)p = 1·17·32 = 544` — **not satisfied** (`DERIVED`, `log₂`), so Thm 1 is vacuous at `s = 64`,
  `d = 16`, float32; it bites at `n ≳ 100` for that width or at `p = 16`. "A consequence *is* a
  composition" (brief §1 item 3) is licensed as a *reduction sketch*, not as a theorem about the
  committor. Multi-layer statements are conditional (`L ≠ NL`).
- **Found by.** Seed.

### 1.6 Chen, Peng, Wu — *Theoretical limitations of multi-layer Transformer* — arXiv:2412.02975 · `[V]`
- **Owns.** **Thm 1.1** (unconditional; the first against multi-layer decoders): for constant
  `L ≤ Õ(log log n)`, an `L`-layer decoder-only softmax transformer cannot solve `L`-sequential
  function composition when `H·d·p ≤ n^{2^{−4L}}`, `p ≥ log n`. **Cor 1.2**: `L + 1` layers suffice
  with polylog parameters — the depth-`L` vs depth-`L+1` separation is exponential. **Cor 1.3**:
  `O(log L)`-layer encoder vs `L`-layer decoder, unconditional.
- **Leaves open.** This is the theorem the brief's obstruction 2 should cite for an *unconditional*
  statement that depth-1 softmax cannot do 2-hop and depth-`L` cannot do `(L+1)`-hop sequential
  composition. Its price: the exponent `2^{−4L}` — at `L = 1`, `H·d·p ≤ n^{1/16}`; `64^{1/16} ≈ 1.30`
  (`DERIVED`) — **vacuous at every geometry the record has run**. Also decoder-only (causal), which
  matches the shape's causal `P`. The task is sequential composition of *given* functions, not
  absorption in a learned chain.
- **Found by.** S3.

### 1.7 Kozachinskiy et al. — *Strassen Attention, Split VC Dimension and Compositionality in Transformers* — arXiv:2501.19215 · `[V]`
- **Owns.** **Thm 3.2**: one-layer softmax transformers at *infinite precision* satisfy
  `size(T) = split-VC(f)^{Ω(1)}`, `size = max{d, H, L_mlp}`; **Thm 3.4** function composition needs
  `size = n^{Ω(1)}`; **Thm 3.6** Match3; **Thm 3.8** binary-relation composition. **Strassen
  attention**: `a_i = Σ_{j,k} a_{ijk}(v_j ⊙ v_k)` with a trilinear softmax, `O(n^ω d)`.
- **Leaves open.** Removes the precision hypothesis from 1.5 — the only one-layer lower bound that
  survives float64. Still asymptotic (`n^{o(1)}` vs `n^{Ω(1)}`). Strassen attention is a competing
  *one-layer composition* operator (third-order, sub-cubic), a NEAR-MISS to "one operator computes the
  hops": it does **two** hops per layer by arity, the resolvent does **all** hops by inversion.
  No chain, no absorbing sets, no certificate.
- **Found by.** S5.

### 1.8 Barceló, Kozachinskiy, Steifer — *Ehrenfeucht-Haussler Rank and Chain of Thought* — arXiv:2501.12997 · `[V]`
- **Owns.** EH-rank of `f` = minimum CoT steps for a single-layer *hard*-attention transformer;
  composing `ℓ` functions needs exactly `ℓ` steps.
- **Leaves open.** Hard attention only; a CoT-steps law, not a depth law; not softmax.
- **Found by.** S3.

### 1.9 Yehudai et al. — *Depth-Width tradeoffs in Algorithmic Reasoning of Graph Tasks with Transformers* — arXiv:2503.01805 · `[V]`
- **Owns.** With width *linear* in `n`, constant depth suffices for a host of graph tasks; some need
  quadratic width. The log-depth "necessity" of 1.1/1.2 holds only under sub-linear width.
- **Leaves open.** Sharpens the skyline the brief demands: a *wide* shallow softmax stack is also a
  fellow approximator, so the matched-*parameter* control must fix width as well as depth (the
  brief's "matched depth and parameters" clause is exactly right, and this paper is its citation).
- **Found by.** S4.

### 1.10 Merrill, Sabharwal — *A Little Depth Goes a Long Way* — arXiv:2503.03961 · `[V]`
- **Owns.** **Thm 2**: a `(17,2,1)`-universal transformer unrolled `⌈log₂ n⌉` times solves connectivity
  on `n` vertices (directed or undirected; `18 + 2⌈log₂ n⌉` layers; `p ≥ c log n`; averaging-hard
  attention; input as `n²` adjacency tokens + `n³` padding). **Thm 1**: regular languages at
  `⌈log₂|w|⌉` unrolls. Fixed-depth impossibility *conditional on `TC⁰ ≠ NC¹`*.
- **Leaves open.** Occupies "log depth suffices for reachability" with a *uniform* construction —
  the honest skyline for the paper's control. Says nothing about weighted absorption probabilities.
- **Found by.** S8, S13.

### 1.11 Merrill, Sabharwal — *The Parallelism Tradeoff* — arXiv:2207.00729 · `[V]` (TACL 2023)
- **Owns.** **Thm 2**: log-precision, constant-depth transformers ⊆ logspace-uniform `TC⁰`.
  Consequences under `TC⁰ ≠ P` (the paper phrases the family as `L ≠ P`): cannot solve linear
  equalities `Ax = b` (P-complete), universal CFG recognition, and — explicitly named — graph
  connectivity (L-complete).
- **Leaves open.** This is the circuit-class root of every "constant-depth softmax cannot reach"
  sentence, and it is conditional. It also cuts the other way: **the exact committor is a linear
  solve `(I − Q) q = R 1`**, and the paper lists linear-system solving among the problems above
  `TC⁰`. So the shape's operator is not a constant-depth circuit either (§3.5).
- **Found by.** Seed.

### 1.12 Merrill, Sabharwal, Smith — *Saturated Transformers are Constant-Depth Threshold Circuits* — arXiv:2106.16213 · `[V]` (TACL 2022)
- **Owns.** Saturated (hard-limit) attention ⊆ `TC⁰`. Precursor of 1.11.
- **Leaves open.** Superseded by 1.11 for the soft case.
- **Found by.** Memory; abs fetched.

### 1.13 Merrill, Sabharwal — *A Logic for Expressing Log-Precision Transformers* — arXiv:2210.02671 · `[V]` (NeurIPS 2023)
- **Owns.** Log-precision transformers ⊆ `FO(M)` (first-order logic with majority).
- **Leaves open.** A tighter upper bound than `TC⁰`; same conditional consequences.
- **Found by.** Memory; abs fetched.

### 1.14 Merrill, Sabharwal — *The Expressive Power of Transformers with Chain of Thought* — arXiv:2310.07923 · `[V]` (ICLR 2024)
- **Owns.** CoT step-count hierarchy: log steps ≈ nothing; linear steps ⊇ regular languages;
  polynomial steps = P.
- **Leaves open.** The *autoregressive* route to the hops — a third fellow approximator (deeper
  stack, wider stack, longer decode). The brief's skyline should list it.
- **Found by.** Memory; abs fetched.

### 1.15 Hahn — *Theoretical Limitations of Self-Attention in Neural Sequence Models* — arXiv:1906.06755 · `[V]` (TACL 2020)
- **Owns.** Hard attention cannot recognise PARITY or 2DYCK at fixed layers/heads; soft attention
  cannot do so robustly (bounded Lipschitz argument).
- **Leaves open.** Periodic/hierarchical languages, not hops; historical anchor only.
- **Found by.** Seed.

### 1.16 Hahn, Rofin — *Why are Sensitive Functions Hard for Transformers?* — arXiv:2402.09963 · `[V]` (ACL 2024)
- **Owns.** Low-sensitivity bias: high-sensitivity functions sit in isolated regions of parameter space.
- **Leaves open.** A trainability bias, not a representational bound; relevant to "a third token can
  veto" (a veto is a high-sensitivity function), which the shape moves into boundary conditions.
- **Found by.** Memory; abs fetched.

### 1.17 Strobl, Merrill, Weiss, Chiang, Angluin — *What Formal Languages Can Transformers Express? A Survey* — arXiv:2311.00208 · `[V]` (TACL 12:543–561, 2024)
- **Owns.** The map of circuit/logic/automata bounds; the citation for "the field's consensus".
- **Leaves open.** Pre-dates 1.6, 1.7, 1.9, 1.10; the paper must cite those directly.
- **Found by.** Seed.

### 1.18 Liu, Ash, Goel, Krishnamurthy, Zhang — *Transformers Learn Shortcuts to Automata* — arXiv:2210.10749 · `[V]`
- **Owns.** `O(log T)`-depth simulation of any semiautomaton; `O(1)`-in-`T` depth for solvable ones
  (Krohn–Rhodes, with `O(|Q|² log |Q|)` depth, `PRIOR_ART.md:85-90`); **Thm 4** (Transformer
  Barrington): non-solvable semiautomata not simulable at depth independent of `T` unless
  `TC⁰ = NC¹` (`PRIOR_ART.md:44-50`).
- **Leaves open.** State tracking is *not* the shape's task, and the record already concedes the
  whole solvability axis (`PRIOR_ART.md:52-56`). Cited for the ceiling only.
- **Found by.** Seed; the record's own delta memo.

### 1.19 Barrington 1989 (JCSS 38(1):150–164, DOI 10.1016/0022-0000(89)90037-8) · `[V]` via Crossref; Krohn–Rhodes 1965 (Trans. AMS 116:450–464, DOI 10.1090/S0002-9947-1965-0188316-1) · `[V]` via Crossref
- **Own.** `NC¹ =` bounded-width branching programs; the prime decomposition of finite semigroups.
- **Leave open.** Context for 1.18 only. The Elsevier landing page returned HTTP 403 and the AMS
  page only redirected; the Crossref API records (title, authors, volume, pages) matched.
- **Found by.** Seed.

### 1.20 Cook 1985 — *A taxonomy of problems with fast parallel algorithms* (Information and Control 64:2–22, DOI 10.1016/S0019-9958(85)80041-3) · `[V]` via Crossref
- **Owns.** The class `DET` (integer determinant, matrix inversion, iterated matrix product) and its
  place `NL ⊆ DET ⊆ NC²`.
- **Leaves open.** Not a paper about attention; it is the reference for §3.5's `DERIVED` statement
  that the exact resolvent is a `DET`-class computation.
- **Found by.** Memory of the class; DOI fetched.

### 1.21 Bhattamishra, Hahn, Blunsom, Kanade — *Separations in the Representational Capabilities of Transformers and Recurrent Architectures* — arXiv:2406.09347 · `[V]`
- **Owns.** One-layer log-width index lookup; one-layer linear size for bounded Dyck and string
  equality, two-layer log size.
- **Leaves open.** One-vs-two-layer separations on decision tasks; corroborates 1.3's shape.
- **Found by.** Memory; abs fetched.

### 1.22 Chen, Zou — *What Can Transformer Learn with Varying Depth?* — arXiv:2404.01601 · `[V]`
- **Owns.** Case studies: one layer memorises, two reason/generalise, three do contextual
  generalisation. This is the "Chen et al." of the dispatch's seed line, and it is empirical/case-based.
- **Leaves open.** No lower bound.
- **Found by.** S16.

### 1.23 Wang et al. — *Learning Compositional Functions with Transformers from Easy-to-Hard Data* — arXiv:2505.23683 · `[V]` (COLT 2025)
- **Owns.** `k`-fold composition learnable by GD under an easy-to-hard curriculum; a
  statistical-computational gap under the standard distribution.
- **Leaves open.** A *learnability* result that the paper's BED-S curriculum (candidate moves in
  context) should cite as the reason a difficulty dial must vary (`MISTAKES.md` D-3).
- **Found by.** S2.

### 1.24 Strobl, Angluin, Frank — *Concise One-Layer Transformers Can Do Function Evaluation (Sometimes)* — arXiv:2503.22076 · `[V]`
- **Owns.** One-layer polylog transformers evaluate a function (one hop) under some encodings and
  fail when input/output are carried only by position; two layers repair it.
- **Leaves open.** Marks the *boundary* of the one-hop ground softmax owns — the shape's BED-M at
  `t* = 1` sits inside it.
- **Found by.** S3, S12.

### 1.25 Back de Luca et al. — *Positional Attention* — arXiv:2410.01686 · `[V]` (ICML 2025); Rajaraman et al. — *Transformers on Markov Data: Constant Depth Suffices* — arXiv:2407.17686 · `[V]`; Makkuva et al. — *Attention with Markov* — arXiv:2402.04161 · `[V]` (ICLR 2025); Bietti et al. — *Birth of a Transformer* — arXiv:2306.00802 · `[V]` (NeurIPS 2023)
- **Own.** Respectively: position-only attention keeps parallel expressivity at log depth; three
  layers represent the in-context conditional of `k`-th order Markov data; single-layer models on
  first-order Markov data fall into unigram minima; the two-layer induction head as associative memory.
- **Leave open.** None concerns absorption probabilities or interventions; 2402.04161 and
  2407.17686 are the nearest "attention + Markov chain" theory and both treat the chain as the *data*,
  not as the *operator*. The shape treats `P` itself as the chain.
- **Found by.** S2, S7, S12.

---

## 2. Per-source ledger — the softmax role, and the resolvent operators

### 2.A Optimality of the softmax role

### 2.1 Marion, Berthier, Biau, Boyer — *Attention layers provably solve single-location regression* — arXiv:2410.01537 · `[V]` (ICLR 2025)
- **Owns.** Task `Y = X_{J₀}ᵀ v* + ξ`; predictor `T(X) = erf(λXk)ᵀ X v` (**erf, not softmax**);
  **Cor 2**: risk `→ ε²` under `λ√d → ∞`, `λ√L → 0` (i.e. `d → ∞`, `L = o(d)`); **Prop 3**: the
  best *linear* predictor's risk `→ ε² + γ²`. Multiple-location regression is named as future work.
- **Leaves open.** Confirms `MATHEMATICS.md:1044-1057` line by line. The paper may say "softmax-like
  attention is asymptotically optimal on a scalar single-location label" and no more; the record's
  `L/d = 4.00` is the wrong direction for Cor 2. D-1 stands.
- **Found by.** Seed.

### 2.2 Duranthon, Marion, Boyer, Loureiro, Zdeborová — *Statistical Advantage of Softmax Attention: Insights from Single-Location Regression* — arXiv:2509.21936 · `[V]` (ICLR 2026)
- **Owns.** With softmax proper: **Prop 4.2** population-level Bayes optimality on SLR; **Cor 4.3**
  linear attention error `O(1/ν)` vs softmax `e^{−cν}` (spiked-SLR), and `→ 1` on max-SLR; **Cor 4.4**
  variable `L` hurts linear attention; **Result 5.1** proportional-limit finite-sample risk.
- **Leaves open.** Repairs the "erf" caveat: softmax itself is now the optimal object on SLR. Still
  single-location; no multi-hop, no vector label. This is the sharpest possible statement of *the
  ground softmax owns* and it should be cited where the brief says "D-1".
- **Found by.** Seed.

### 2.3 Boursier, Boyer — *Softmax as Linear Attention in the Large-Prompt Regime* — arXiv:2512.11784 · `[V]`
- **Owns.** Softmax attention → a linear operator on token measures as the prompt grows.
- **Leaves open.** A limit in which the three corners of the record's family (`V16Domain.three_corners_containment`) coalesce; cite when stating that parity at `γ = 0` is not a large-`s` distinction.
- **Found by.** S11.

### 2.4 Boncoraglio, Troiani, Erba, Zdeborová — *Bayes optimal learning of attention-indexed models* — arXiv:2506.01582 · `[V]` (NeurIPS 2025)
- **Owns.** Closed-form Bayes-optimal error and phase transitions for attention-indexed models.
- **Leaves open.** An information floor *methodology* (replica/AMP) that BED-S's Fano floor could
  borrow; not about hops.
- **Found by.** S11.

### 2.5 Veličković, Perivolaropoulos, Barbero, Pascanu — *Softmax is not Enough (for Sharp Size Generalisation)* — arXiv:2410.01104 · `[V]` (ICML 2025)
- **Owns.** Softmax circuits must disperse as item count grows; a size-generalisation failure of
  sharpness.
- **Leaves open.** A length-generalisation limitation of the *row mixture* itself — supports
  obstruction 1 (per-row independence) at the level of sharpness, not joint determination.
- **Found by.** Memory; abs fetched.

### 2.6 Tarzanagh, Li, Zhang, Oymak — *Max-Margin Token Selection in Attention Mechanism* — arXiv:2306.13596 · `[V]`; Tarzanagh, Li, Thrampoulidis, Oymak — *Transformers as Support Vector Machines* — arXiv:2308.16898 · `[V]`
- **Own.** GD on attention converges in direction to the max-margin token separator; the SVM
  equivalence with nuclear-norm objective.
- **Leave open.** Training-dynamics results about *selection*; the record used them to explain the
  one-hot control (`DONE.md:792-795`). Not a representational bound.
- **Found by.** Seed.

### 2.7 Dong, Cordonnier, Loukas — *Attention is Not All You Need* — arXiv:2103.03404 · `[V]`
- **Owns.** Pure attention stacks lose rank doubly exponentially with depth.
- **Leaves open.** A cost law for the deeper-softmax skyline that the paper's control section should
  carry beside the parameter count.
- **Found by.** Memory; abs fetched.

### 2.8 Ramsauer et al. — *Hopfield Networks is All You Need* — arXiv:2008.02217 · `[V]`
- **Owns.** The modern-Hopfield update rule *is* softmax attention; one-step retrieval, exponential
  capacity — the energy-descent reading of why one softmax step suffices for retrieval.
- **Leaves open.** Retrieval is Thm 5 territory of 1.2 (depth 1); the Hopfield fixed point is a
  *per-query* fixed point, so it does not touch joint determination. Another planet's lineage
  (equilibrium) owns the rest.
- **Found by.** Memory; abs fetched.

### 2.B The resolvent / Neumann / chain operators (the operator side of the shape)

### 2.9 **Fagnou, Caillon, Delattre, Allauzen — *Chain and Causal Attention for Efficient Entity Tracking* — arXiv:2410.05565 · `[V]` · EMNLP 2024 main, pp. 13174–13188, DOI 10.18653/v1/2024.emnlp-main.731 (ACL Anthology page fetched)** — **SURPRISE**
- **Owns.** **Eq. 5**: `Y = (1 − γ) A (I − γA)^{-1} V` with `A` the *causal, row-stochastic softmax*
  attention (diagonal removed), `γ ∈ [0,1)` a discount; series `A + γA² + γ²A³ + …`; computed by a
  **triangular solve** (`torch.linalg.solve_triangular` in training; row-by-row forward substitution
  at decode); **Thm 1**: standard attention needs `L_min = ⌈log₂(depth(G) + 1)⌉` layers to track `n`
  state changes (unconditional inside a receptive-field-one model; induction on `2^L − 1`); **§4.2**:
  ChaCAL is the fixed point of `f(Z) = A(γZ + (1 − γ)V)`; results: Boxes exact-match `99.1 %` with
  `1 + 1` layers vs `97.0 %` for a 5-layer standard transformer, `1.75×` faster `[U: summariser's
  reading of the HTML]`; cites MAGNA (2.13) as closest with "significant differences".
- **What it occupies in the shape (each line checked against `BRIEF.md` §1):**
  - `O = P (I − γP)^{-1} V` — occupied up to the scalar `(1 − γ)`.
  - I1 parity at `γ = 0` — occupied (`(1 − 0) A (I)^{-1} V = AV`, `DERIVED`).
  - I5 "the causal resolvent is a triangular solve" — occupied, same PyTorch call.
  - The depth-law motivation ("one layer instead of `log` layers") — occupied, with its own
    Theorem 1.
  - The fixed-point reading of the operator — occupied (§4.2).
- **What it leaves open (the shape's remaining delta, stated narrowly):**
  - **I2/nilpotency.** Its convergence argument is `γ < 1` (geometric) `[U]`; the record's
    `Nilpotent.pow_card_eq_zero` / `occupancy_is_exact_inverse` say the strictly-causal series is
    *finite and exact at any `γ`, sign-blind* — a stronger, machine-checked statement ChaCAL lacks.
  - **I3 boundary conditions.** No absorbing rows, no committor, no constraint sets, no
    "third token vetoes through the boundary". NOT FOUND in the paper.
  - **I4 certificate.** No truncation bound `γ^{K+1}/(1 − γ)`, no planted negative. NOT FOUND.
  - **do() / displacement / safest move.** NOT FOUND (`[V]` for absence: the HTML read reports
    none of "intervention", "counterfactual", "do", "committor", "hitting").
  - **Label class.** Boxes is a per-position state readout — still a scalar-at-a-position label
    (the D-1 class), not a jointly determined configuration.
  - **Row sums.** `(1 − γ)` restores row-stochasticity: for row-stochastic `A`,
    `A(I − γA)^{-1} 𝟙 = 𝟙/(1 − γ)` (`DERIVED`: Neumann series termwise), so ChaCAL's read is a
    convex mixture again and the shape's `O = P(I − γP)^{-1}V` without the factor has row sums
    `1/(1 − γ)`. The paper must choose and say which (I4's bound is in `‖·‖_∞` and the factor
    changes its constant).
- **Found by.** S6 → 2605.22476 → its cited foundation → S17/S18. **The repository does not cite it**
  (`grep -rlEi "ChaCAL|Fagnou|2410\.05565" --include="*.md"` outside `kaggle/` → `0` files, `RUN`
  2026-09-03). The record's own prior-art trawl (`DONE_ARCHIVE_ROUND1.md:672-681`) was run during
  the *signed* programme and discarded every non-negative multi-hop operator as "requires a
  non-negative matrix" — the shape now *is* a non-negative multi-hop operator, so that filter must be
  re-run without it. This is the `V-7` mechanism (a search structurally incapable of finding
  anything, read as absence), applied to prior art.

### 2.10 Zhao, Caillon, Fagnou, Allauzen — *Structured-Sparse Attention for Entity Tracking with Subquadratic Sequence Complexity* — arXiv:2605.22476 · `[V]` (v1 21 May 2026)
- **Owns.** Blockwise evaluation of `S_γ(A) = (1 − γ)A(I − γA)^{-1}`: `A = A_blk + A_res`, exact
  triangular solves on `m × m` tiles, cross-block interactions through a reduced `k × k` system;
  `Õ(n^{4/3} d)`; `γ` fixed; cites an `O(log K)` depth lower bound for `K` state updates; Boxes
  `100 %` at `12–29 %` lower latency than dense resolvent, `2.4×` vs a 5-layer dense transformer;
  WikiText-103 ppl `15.18` vs `16.46` `[U: summariser's numbers]`; head–property capacity
  constraint (`h ≥ p`).
- **Leaves open.** Occupies "blockwise resolvent" as an *approximation of learned locality*. The
  shape's block structure is *exact* — a zero gate annihilates the path product
  (`V16Domain.pathProd_eq_zero_iff`, `bedM_gate_exact`), with a Lean certificate — so the delta is
  "segmentation ⇒ block-diagonal resolvent, exactly, with a printed δ = 0" against "block-diagonal
  plus a learned residual, approximately". The paper must cite 2.10 before stating the segmentation
  proposition (`BRIEF.md` §6.4 "segmentation ⇒ block structure"). Same absences as 2.9 for I3, I4,
  do(), safest move.
- **Found by.** S6, S16, S18.

### 2.11 Roffo, Abdelkawy, Lavie, Palmer — *Self-Attention And Beyond the Infinite (InfSA)* — arXiv:2603.00175 · `[V]` (cs.CV)
- **Owns.** `Ĉ = (I − γÂ)^{-1} − I` with `Â` ReLU-activated, *Frobenius-normalised* (explicitly not
  row-stochastic, and the paper argues *against* softmax's stochasticity as oversmoothing); `γ`
  learnable per head via sigmoid; the absorbing-chain / fundamental-matrix reading
  (`N_ij` = expected visits before absorption; `R_i = 1 − γ Σ_j Â_ij`); Linear-InfSA rank-1
  principal-eigenvector approximation; ImageNet numbers.
- **Leaves open.** Non-causal; no committor into *named* absorbing sets (absorption is a uniform
  leak, not a constraint set); no do(); no truncation bound in the main text; no softmax parity. The
  record already had it (`DONE_ARCHIVE_ROUND1.md:1317`) and dismissed it for non-negativity. It
  occupies "attention as fundamental matrix of an absorbing chain" as a *reading*; the shape's I3
  (absorbing rows as constraint sets, committor `q = (I − Q)^{-1} R 𝟙_B`) is the *boundary-value*
  version, which InfSA does not state.
- **Found by.** S6, S9, S10.

### 2.12 Erel et al. — *Attention (as Discrete-Time Markov) Chains* — arXiv:2507.17657 · `[V]` (NeurIPS 2025, cs.CV)
- **Owns.** Softmax attention read as a DTMC transition matrix; indirect attention via chain
  powers; TokenRank from the stationary vector; an analysis/editing tool.
- **Leaves open.** Not a layer; no absorbing sets; no inverse. Occupies the *sentence* "the
  row-stochastic attention matrix is a Markov chain".
- **Found by.** S15.

### 2.13 Wang, Ying, Huang, Leskovec — *Multi-hop Attention Graph Neural Network (MAGNA)* — arXiv:2009.14332 · `[V]` (IJCAI 2021)
- **Owns.** Attention diffusion `Σ_k θ_k A^k` with personalised-PageRank weights on a *fixed graph*.
- **Leaves open.** Not causal, not a sequence operator, truncated in practice; the ancestor ChaCAL
  names. Already in the record (`DONE_ARCHIVE_ROUND1.md:677`).
- **Found by.** S6, S10.

### 2.14 Feng, Li, Jiang, Ying — *Diffuser* — arXiv:2210.11794 · `[V]`; Glickman, Yahav — *Diffusing Graph Attention* — arXiv:2303.00613 · `[V]`
- **Own.** Multi-hop attention diffusion over sparse patterns for long sequences (Diffuser); learned
  structural relations steering graph-transformer attention.
- **Leave open.** Neither is a causal resolvent nor certified; cited so the paper's occupancy table
  is complete on "multi-hop in one layer".
- **Found by.** S10.

### 2.C Seen in the record but NOT fetched this session (excluded from the `.bib`)
- ParaFormer (PageRank Transformer, `Σ_k γ_k Â^k`, scalar `γ_k` per hop, `K = 10`) —
  `READ DONE_ARCHIVE_ROUND1.md:1321-1324`; no identifier surfaced by S10; `[U]`.
- Gated DeltaNet's `(I − A)^{-1} = Σ A^n` — `READ DONE_ARCHIVE_ROUND1.md:1319`, arXiv:2606.06034 per
  the record; not fetched; `[U]`.
- APPNP 1810.05997, GDC 1911.05485 — `READ DONE_ARCHIVE_ROUND1.md:677`; not fetched; `[U]`.
- 2508.07208 (two-layer induction heads on any-order Markov chains), 2510.18638 — surfaced by S7,
  not fetched; `[U]`.

---

## 3. The answer to the dispatch's question — which theorem licenses what

The sentence under examination: *"depth-1 softmax cannot compute a `t*`-hop reachability / committor
that the resolvent computes in one operator."* No single theorem licenses it. Five theorems each
license a piece, under stated conditions, and one `DERIVED` observation limits what "one operator"
may claim.

### 3.1 `t* = 2` (one composition; the induction head) — UNCONDITIONAL against one softmax layer, asymptotic in `n`
- **1.3** Sanford–Hsu–Telgarsky 2408.14332 Thm 1: `h·m·p = Ω(n)` (softmax, `p` bits, 3-symbol
  alphabet). **1.5** Peng–Narayanan–Papadimitriou 2402.08164 Thm 1: error `≥ R/(3 n log n)` when
  `n log n > H(d+1)p`. **1.7** Kozachinskiy et al. 2501.19215 Thm 3.4: `size = n^{o(1)}` cannot compose,
  at *infinite* precision.
- **Conditions.** Domain/context `n` grows; parameters sublinear/polylog in `n`. **At the record's
  geometry** (`s = 64`, `d = 16`, `h = 1`, float32) 1.3's hypothesis reads `512 ≥ 64` and 1.5's reads
  `384 < 544` (`DERIVED`) — both vacuous. The paper may cite them for the *asymptotic* statement and
  must print those two inequalities beside the citation (`MISTAKES.md` V-25, P-10).

### 3.2 `t* = L + 1` against `L` layers — UNCONDITIONAL, doubly-exponentially vacuous at small `n`
- **1.6** Chen–Peng–Wu 2412.02975 Thm 1.1: `L`-layer decoder-only softmax cannot do `L`-sequential
  composition when `H·d·p ≤ n^{2^{−4L}}`, `L ≤ Õ(log log n)`; Cor 1.2: `L + 1` layers suffice at polylog.
- **Conditions.** `n^{1/16}` at `L = 1` is `1.30` for `n = 64` (`DERIVED`) — the theorem is silent at
  every `s` the record ran. It is the right citation for the *shape* of obstruction 2 (depth
  `L` cannot do `L + 1` hops) and the wrong citation for any number at `s ≤ 10⁴`.

### 3.3 `t*` general, depth `Ω(log t*)` — CONDITIONAL
- **1.1** SHT 2402.09268 Cor 4.3 (Conjecture 2.4, one-vs-two-cycle, `k = Θ(N^ξ)`, `mH = O(k^{1−ε})`);
  **1.2** Sanford et al. 2405.18512 Thm 3/19 (Conjecture 13, same conjecture, `mH = O(N^ε)`). Upper
  bounds Thm 4.2 (`⌊log₂ k⌋ + 2`) and Thm 2/18 are unconditional and are the **skyline**.
- **Conditions.** Sub-linear width — **1.9** Yehudai et al. removes the necessity at linear width. So
  the matched control must fix *both* depth and width (the brief's "matched depth and parameters"),
  and the skyline row in the occupancy table must list three fellow approximators: the
  `⌊log₂ t*⌋ + 2` stack (1.1), the wide constant-depth stack (1.9), and the CoT decoder (1.14).

### 3.4 Reachability / committor as a *problem class* — CONDITIONAL on circuit-class separations
- **1.11** Merrill–Sabharwal 2207.00729 Thm 2 (log-precision constant depth ⊆ uniform `TC⁰`;
  connectivity is L-complete; linear systems P-complete); **1.10** 2503.03961 (log depth suffices
  for STCON; fixed depth cannot under `TC⁰ ≠ NC¹`); **1.18** Liu et al. Thm 4 (`TC⁰ ≠ NC¹`).
- **Conditions.** All conditional; all asymptotic; all about *standard* layers.

### 3.5 What "one operator" may and may not claim — `DERIVED`, and the paper must say it
- The exact committor is the solution of `(I − Q) q = R 𝟙_B`; the exact `z` is `(I − γP)^{-1} V`.
  Matrix inversion and iterated product are `DET`-complete and `NL ⊆ DET ⊆ NC²` (Cook 1985, 1.20).
  So the resolvent *as a circuit* sits at or above the class the conditional lower bounds say
  reachability needs. **Nothing in the shape contradicts 3.3–3.4; the shape relocates the
  `log`-depth from the parameter stack into the linear solve.** I5's triangular solve is `O(s)`
  sequential steps, or `O(log s)` parallel rounds by prefix doubling with `O(s³)`-class work
  (`DERIVED`; not measured — `NOT MEASURED — needs a parallel-prefix kernel timing`).
- The honest sentence is therefore: *one **parameterised** operator (one `P`, one `γ`), not one
  parallel round; its hops are exact and certified (I2, I4) rather than learned layer by layer.*
  "Depth-1" in the paper must be defined as *one attention parameter set*, and the circuit-depth
  reading must be disclaimed in the same paragraph (`MISTAKES.md` P-8: a headline stating an upper
  bound as a price; V-17: a threshold imported out of its units — "depth" has two units here).
- The `hop_k` graph is token-defined; the shape's reachability is in the graph `P` *that the layer
  computes*. A reduction "any `hop_k` instance is a committor instance of some `P`" is plausible
  (put the pointer structure into `QKᵀ`) but **NOT FOUND** in the literature and not written in the
  record. Until written out, obstruction 2 is a *lineage argument*, not a theorem about the shape.

### 3.6 NOT FOUND (recorded, with the queries)
- An unconditional lower bound against one softmax layer at fixed `s = 64`, `d = 16`, float32/64 —
  none (S2, S3, S5, S12, S16; consistent with `MATHEMATICS.md:519-521`).
- A theorem about *absorption probabilities into named constraint sets* as a transformer task class —
  none (S7, S9, S15).
- An expressivity theorem about *interventional displacement* `Δz = z(do a) − z` for attention —
  none (S14).
- A matched-depth, matched-parameter comparison of a causal resolvent layer against the
  `⌊log₂ t*⌋ + 2` softmax stack with an exactness certificate — none (S16; ChaCAL compares
  `1 + 1` layers against a 5-layer stack on accuracy and latency, without a certificate).
- The `argmin_a max_k q^{(k)}(do a)` decision object as an attention read — none (S15).

---

## 4. Verdict

**OCCUPIED (owner named, cite before naming):**
- The operator `P(I − γP)^{-1}V` on causal row-stochastic softmax `P`, its triangular-solve
  evaluation, its `γ = 0` parity, and its fixed-point reading — **Fagnou et al. 2024 (ChaCAL)**,
  arXiv:2410.05565, EMNLP 2024. The record does not cite it. The paper's I1 and I5 become
  *reproductions* of ChaCAL's construction, and §4 "the shape — definition" must open with that
  attribution.
- Blockwise evaluation of the same resolvent with exact within-block solves — **Zhao et al. 2026**,
  arXiv:2605.22476 (approximate; the exact zero-gate version stays with the record's Lean F0).
- "Attention as the fundamental matrix of an absorbing chain" as a reading — **Roffo et al. 2026
  (InfSA)**, arXiv:2603.00175 (non-stochastic, non-causal).
- The depth law for `hop_k` — **SHT 2024** Thm 4.2 / Cor 4.3 (conditional); for graph connectivity —
  **Sanford et al. 2024** Thm 2/18, 3/19 (conditional); log-depth sufficiency with a uniform
  construction — **Merrill–Sabharwal 2025** Thm 2; `TC⁰` ceiling — **Merrill–Sabharwal 2023** Thm 2.
- Unconditional one-layer composition impossibility — **SHT 2408.14332** Thm 1, **Peng et al. 2024**
  Thm 1, **Kozachinskiy et al. 2025** Thm 3.4; unconditional multi-layer — **Chen–Peng–Wu 2024** Thm 1.1.
- The ground softmax owns (scalar single-location label) — **Marion et al. 2025** Cor 2 (erf,
  `L = o(d)`), **Duranthon et al. 2026** Prop 4.2 / Cor 4.3 (softmax proper).
- Depth is not necessary at linear width — **Yehudai et al. 2025**.

**NEAR-MISS (adjacent, does not take the component):**
- Strassen attention (1.7): one-layer composition by *arity* (two hops), not by inversion (all hops).
- MAGNA / Diffuser / Diffusing Graph Attention / ParaFormer (2.13, 2.14, 2.C): non-negative
  multi-hop sums, fixed-graph or truncated, uncertified.
- Erel et al. (2.12), Makkuva et al., Rajaraman et al. (1.25): Markov chain as *data* or as an
  *analysis lens*, not as the operator with boundary conditions.
- Hopfield (2.8): a per-query energy fixed point; not joint determination.

**NOT FOUND (see §3.6):** absorbing constraint sets as boundary conditions inside the resolvent
(I3 as a *layer*, with the committor `q^{(k)}` read out); the Neumann certificate with a planted
negative (I4); the nilpotency exactness theorem in the operator's own paper (I2 — the record's Lean
statement is stronger than ChaCAL's convergence argument `[U]`); the interventional displacement
`Δz`; the safest-move decision; any lower bound at the record's geometry; any matched-depth,
matched-parameter certified comparison.

**SURPRISE — flagged loudly.** **ChaCAL (arXiv:2410.05565, EMNLP 2024) occupies the operator, the
solver, the parity and the depth-law motivation of the shape, in one 2024 paper the repository never
cites, with a 2026 follow-up (arXiv:2605.22476) that already adds block structure.** It does *not*
occupy the boundary conditions, the certificate, the interventional channel, the decision, the
vector-valued label, or the Lean exactness — which is exactly the delta `BRIEF.md` §1 lists as I3,
I4, do(), safest move. The record's earlier prior-art filter (`DONE_ARCHIVE_ROUND1.md:672-681`)
discarded non-negative multi-hop operators wholesale while the signed programme was alive; with the
sign programme dead, that filter is the reason ChaCAL was missed (V-7 applied to search). The
paper must (i) cite ChaCAL in the abstract-level statement of the shape, (ii) state the delta as
"boundary conditions + certificate + intervention + decision on ChaCAL's operator", (iii) print
the `(1 − γ)` row-sum choice, and (iv) not describe I1 or I5 as contributions.

---

## 5. Design against the taxonomy — how the paper should use this lineage

| proposal | designed against | how |
|---|---|---|
| Cite theorem *numbers* (Thm 4.2, Cor 4.3, Thm 1.1, Thm 1, Thm 3.4, Prop 4.2, Cor 2, Thm 2) never abstracts | P-10 (a source's intro cited as its theorem) | every obstruction sentence in §5 of the paper carries a theorem number and the model class it is stated for |
| Print the two vacuity inequalities (`512 ≥ 64`; `384 < 544`; `64^{1/16} = 1.30`) beside every unconditional lower bound | V-25 (a theorem whose hypothesis no draw satisfies) | the reader sees that the unconditional bounds are asymptotic and do not bind at `s = 64` |
| Label Cor 4.3 / Thm 3/19 / `TC⁰ ≠ NC¹` / `L ≠ NL` results "conditional" in the same sentence as the bound | P-3 (a stale claim never retracted), P-11 (citing a conjecture as settled) | conditional bounds never appear without their conjecture |
| Define "depth-1" as *one parameter set* and disclaim the circuit-depth reading (§3.5) | V-17 (a threshold imported out of its units), P-8 (an upper bound stated as a price) | "one operator" cannot be read as "constant depth" |
| Skyline row with three fellow approximators: `⌊log₂ t*⌋ + 2` stack (1.1), wide constant-depth stack (1.9), CoT decoder (1.14); cost laws from 2.7 | D-1 (racing a proven optimum), R-SKY (`CEQ_V16_CONTRACT.md:209`) | no "beats softmax" sentence on a bed where any of the three is a fellow approximator |
| Attribute the operator to ChaCAL before defining the shape; delta = I3 + I4 + do() + safest move | P-7 (vocabulary with no referent), rule 7 of the brief (occupied components are cited before they are named) | I1 and I5 are reproductions, not contributions |
| Re-run the record's non-negative-operator filter without the sign criterion (2.9's finding) | V-7 (a search structurally incapable of finding anything, read as absence) | the prior-art census for the operator component is redone under the shape's own hypothesis |
| The single-location ground is stated with Duranthon et al. (softmax proper), not only Marion et al. (erf) | P-3, §17.5's own correction | the "erf, not softmax" caveat is closed by the 2026 paper and the paper says so |
| The `hop_k` → committor reduction is written out or marked `NOT FOUND` | V-3 (an assertion that is an algebraic identity of the author's own construction), D-2 | obstruction 2 is not presented as a theorem about the shape until the reduction exists |
| Cite Wang et al. 2025 (1.23) for the difficulty dial in BED-S | D-3 (a difficulty dial that does not vary) | the easy-to-hard curriculum is a cited necessity, not a convenience |

---

## 6. Counts for the coordinator

- Sources with a fetched identifier and matched title: **45** (`bib_expressivity.bib`), all `[V]`.
- Details inside those sources marked `[U]` (summariser readings, memory venues): ChaCAL's headline
  numbers and its "geometric, not nilpotent" convergence remark; Zhao et al.'s numbers; NeurIPS 2023
  venues for 2306.02896 / 2306.13596; ICML 2021 for 2103.03404; ICLR 2021 for 2008.02217; NeurIPS
  2025 spotlight for 2503.01805; ICML 2024 for 2402.09268 (attested by the record, not the arXiv page);
  Duranthon's given name.
- Sources mentioned but excluded from the `.bib` for lack of a fetched identifier: 6 (§2.C).
- Searches run beyond the seeds: 18 (table in §0).
- Files written: `sweep/bib_expressivity.bib`, `sweep/sweep_expressivity.md`. No code, no git writes.
