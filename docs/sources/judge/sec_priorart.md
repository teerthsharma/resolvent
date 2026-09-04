# 7. Prior art and occupancy

JUPITER (MYCROFT), 2026-09-03, HEAD `207e7b9`. Inputs read in full: the brief; both corrections;
the outline; `sec_intro_draft.md`; the eight sweeps `sweep_{resolvent,linrec,expressivity,causality,
safety,topology,occupied,methods}.md`; `judge/sec_shape.md` §4.0, whose occupancy paragraph this
section must agree with and go wider than; `bib_audit.md`; `bib_aliases.md`; `references.bib`
(canonical keys only — an alias key is never printed). Evidence classes: `RUN` (executed on this box
this session), `READ path:line`, `CITED [V]` (abs page, DOI/registry record, catalogue record or
proceedings page fetched by the named sweep this session and the title matched), `CITED [U]` (reached
through a search index, a secondary paper, or memory), `DERIVED` (steps written out). No number
without provenance. Struck constants (`STRUCK.md`, 12 entries) do not appear.

The rule this section exists to serve is the brief's rule 7: *occupied components are cited before
they are named, and absence is recorded as NOT FOUND with the query that bounds it, never as
"novel"*. It is the same rule the record breaks in §7.3.

---

## 7.0 How to read the table

**Verdicts.** `OCCUPIED` — one named source states, in print, the equation or mechanism with the
same object on both sides. `NEAR-MISS` — a named source states a neighbouring object and the table
says which clause is missing. `NOT FOUND` — no fetched source states it, and the row carries the
queries that failed; the totals in §7.2 bound how much that absence is worth (`MISTAKES.md` V-7, a
search structurally incapable of finding anything read as absence, `MISTAKES.md:117`).

**Marks.** `[V]` and `[U]` are per source, as the sweep that found it recorded them. `[V]` here is a
title match against a fetched identifier page; it is **not** an equation-level check. Where a sweep
read the body and transcribed the equation the row says so; those are the only rows on which an
equation number is load-bearing.

**Boxes.** Six of the shape's structural properties were fixed *before* the first fetch
(`sweep_resolvent.md` §0, against M-2, a threshold refitted to the data it judges): (a) `P` is
content-dependent, (b) causally masked, (c) row-stochastic softmax, (d) the read uses the exact
resolvent, (e) absorbing rows on named sets, (f) softmax parity at $\gamma=0$; and four more —
(g) the interventional re-solve, (h) the safest-move decision, (i) a printed truncation certificate,
(j) learnable $\gamma$. The delta letters in §7.2 are those boxes plus (k) machine-checked
containment and the zero-gate iff, (l) the influence barcode restricted to the F0 endpoint, and
(m) the cover-to-schedule path.

---

## 7.1 The occupancy table

| # | component of the shape | verdict | owner(s), canonical key, mark | what it owns (under 25 words) | what it leaves open relative to CEQ |
|---|---|---|---|---|---|
| 1 | the read operator $O=(1-\gamma)P(I-\gamma P)^{-1}V$ on causal row-stochastic softmax $P$ | **OCCUPIED** | `fagnou-2024-chacal` [V], Eq. 5, EMNLP 2024 (body read) | Discounted path sum over the causal softmax matrix, written as one closed-form read with $\gamma\in[0,1)$ fixed at 0.9. | No boundary rows, no committor, no intervention, no truncation bound; diagonal removed inside the inverse, so the read is sub-stochastic |
| 2 | the triangular solve | **OCCUPIED** | `fagnou-2024-chacal` [V] Eq. 7; `yang-2024-deltanet` [V] Eq. 10; `zhao-2026-structuredsparse` [V] | Solve $(I-\gamma A)Y=(1-\gamma)AV$ by forward substitution rather than inverting; DeltaNet solves the same unit-lower-triangular system per chunk. | Nothing on the solve itself. CEQ's I5 is a reproduction, not a contribution |
| 3 | $\gamma=0$ parity with softmax | **OCCUPIED** | `fagnou-2024-chacal` [V] (body: $\gamma=0$ reverts to standard attention) | At zero discount the layer is the ordinary attention matmul. | The *machine-checked* containment of softmax, linear attention and the path product in one family; that is the record's Lean and is NOT FOUND elsewhere (letter k) |
| 4 | the discount / horizon dial $\gamma$ | **OCCUPIED** | `bellman-1957-markovian` [V]; `dayan-1993-successor` [V]; `gasteiger-2019-appnp` [V]; learnable per head `roffo-2026-infsa` [V] | The discount of policy evaluation; APPNP's teleport $\alpha$; InfSA learns one $\gamma$ per head through a sigmoid. | A learnable discount inside a **causal language-model** read, pinned by a boundary-corrected likelihood-ratio test (letter j) |
| 5 | the row-stochastic base and its three corners | **OCCUPIED** | `vaswani-2017-attention` [V]; `katharopoulos-2020-linearattention` [V]; `dao-2024-ssd` [V] Def. 3.1 | Row-softmax; kernel-feature linear attention; the 1-semiseparable cumulative-product mask that is the record's path product. | That the three are one family with distinct corners, proved rather than asserted (letter k, `V16Domain.three_corners_containment`) |
| 5b | why softmax leaves that lineage | **OCCUPIED** | `hu-2025-ssdtheory` [V] | Row-softmax over $QK^\top$ has rank $T$ even at rank-1 logits, so it admits no finite-state dual. | Nothing. This is a gift: it *proves* the exit point the record had asserted |
| 6 | the chunked / blockwise cost path | **OCCUPIED** | `zhao-2026-structuredsparse` [V] ($\tilde O(n^{4/3}d)$); `yang-2024-deltanet` [V]; `beck-2025-tfla` [V] | Exact triangular solves on diagonal tiles with cross-block traffic routed through a reduced system; chunk tiling for linear-RNN kernels. | An **exact** block structure from a zero gate with a printed $\delta=0$, against a learned block-plus-residual approximation (letter l's sibling; §7.1 row 15) |
| 7 | absorbing rows as boundary conditions | **OCCUPIED on fixed graphs; NOT FOUND in a causal content-dependent attention operator** | `zhu-2003-harmonic` [V]; `zhou-2003-consistency` [V]; `wu-2012-partially-absorbing` [V]; `begga-2023-diffusion-jump` [V]; `azad-2022-harmonic-extension` [V]; one row inside attention `karbalayghareh-2026-doformer` [V] | Clamp labelled nodes, take the harmonic extension elsewhere; partially absorbing walks with per-node absorption; DoFormer fixes an intervened token and forbids it attending. | $K\ge 2$ constraint sets *plus* a goal set *plus* a declared sink, as identity rows of a causal row-stochastic content-dependent read (letter e). Queries: resolvent 17/22, linrec Q11/Q17, occupied S4/S5, methods S11 |
| 8 | the fundamental matrix and absorption probabilities | **OCCUPIED** | `kemeny-1960-finitemarkov` [V-cat]; `grinstead-1997-probability` [U], Thm 11.6 (§11.2 body read by the safety sweep) | $N=(I-Q)^{-1}$, $B=NR$, expected absorption times $N\mathbf 1$, in canonical form. | Nothing on the identity — the record concedes zero delta at `V13_TIER6_PRIOR_ART.md:389-398` (READ). The *use* as a layer's read is elsewhere in this table |
| 9 | the committor / reach-avoid reading | **OCCUPIED, twice, for one object** | `metzner-2009-tpt-markov-jump` [V]; `e-2006-transition-paths` [V]; `e-2010-tptreview` [V]; `doyle-1984-electric` [V]; `summers-2010-reach-avoid` [V]; `abate-2008-reachability` [V] | The discrete Dirichlet problem $(Lq)_i=0$ with $q|_A=0$, $q|_B=1$; the harmonic/electrical reading; reach-avoid probability by dynamic programming. | The naming identity itself (reach-avoid $\equiv$ committor) is stated in neither literature as far as the safety sweep reached; the paper states it as a remark and claims nothing. The **read inside a layer** stays open (letter h) |
| 10 | the safest-move decision rule | **OCCUPIED as names; NOT FOUND as a read** | `vanmoffaert-2013-chebyshev` [V]; `park-2026-maxmin` [V]; `yang-2026-lexisafe` [V]; `altman-1999-cmdp` [V-cat]; `bellman-1957-markovian` [V]; `baier-2008-modelchecking` [V-cat]; `todorov-2006-lmdp` [V] / `todorov-2009-efficient` [V]; NEAR-MISS `jeddi-2021-lyapunovsafe` [V]; framing `hsu-2023-safetyfilter` [V], `borquez-2023-lrf` [V] | Chebyshev (worst-objective) action selection; max-min and lexicographic orders; $K$ cost functionals on one chain; several target sets in one linear system with a verdict; a move from a linear first-exit solve; a transformer picking the lowest predicted violation probability. | The scores being **committors read out of the layer's own operator**, over candidate moves placed in the context (letter h). Queries: safety 2/5/10/12/13/17/18/19, occupied S8, expressivity S15 |
| 11 | the interventional re-solve and its rank-one update | **OCCUPIED as algebra; NOT FOUND as a trained read** | `pearl-2009-causality` [V]; `shimizu-2006-lingam` [V]; `sherman-1950-inverse-adjustment` [V]; `hager-1989-updating` [V]; `schweitzer-1968-perturbation` [V]; `piray-2021-linearrl` [V] Eq. 5; `mooij-2013-ode2scm` [V]; `bongers-2021-cyclic` [V]; `bottou-2013-counterfactual` [V] §7.3; `scetbon-2024-fip` [V] | $do(\cdot)$ as row surgery on $x=(I-B)^{-1}e$; rank-one inverse update; perturbation of the fundamental matrix; the barrier update of a default representation; consequence as displacement of an equilibrium; a causally masked transformer whose $do()$ is clamp-and-re-solve. | The displacement $\Delta z$ as a **jointly scored vector label** at two stated prices — row clamp $O(sd)$ against token rewrite $(s-i)^2d/2$ (letter g). Queries: resolvent 19, linrec Q15, causality Q6/Q9, methods S10 |
| 12 | the order-of-intervention question | **OCCUPIED as a theorem; NOT FOUND inside an attention model** | `dash-2005-emc` [V] Def. 4, Thms 1–2 (PDF read by the causality sweep); `voortman-2010-manipulation` [V]; `blom-2019-ccm` [V] | When "equilibrate then manipulate" differs from "manipulate then equilibrate", and when a reduced equilibrium model hides the feedback that makes it differ. | Nothing: for $\gamma<1$ the map $x\mapsto V+\gamma P'x$ is a contraction with one fixed point, so the orders coincide (`RUN[J]` $8.9\times10^{-16}$) and the paper retires EMC to a remark rather than claiming a bind |
| 13 | equilibrium as the layer's output | **OCCUPIED** | `bai-2019-deq` [V]; `gu-2020-ignn` [V]; `liu-2021-eignn` [V]; `georgiev-2024-dear-neurips` [V]; `ramsauer-2021-hopfield` [V]; `hoover-2023-energytransformer` [V] | A layer whose output is the root of $z=f(z,x)$, differentiated implicitly; algorithmic targets as equilibria; softmax attention as one Hopfield update. | All of these fixed points are **nonlinear** or **per query**. CEQ's is linear in $z$, which is what makes the certificate and the one-pass solve available; Hopfield's is the per-row independence obstruction, not its repair |
| 14 | the attention barcode | **OCCUPIED** | `kushnareva-2021-tda-attention` [V]; `kushnareva-2022-betti` [V] (preprint, abs marks it not accepted); `cherniavskii-2022-acceptability` [V]; `perez-2022-topological-bert` [V]; `samaga-2026-halluzig` [V]; `bazarova-2025-toha` [V] | Threshold a head's attention matrix over a sweep, read $H_0/H_1$ persistence as classification features; zigzag over the layer axis; a two-subgraph topological divergence. | The **filtering function**: a filtration by the influence Jacobian of a resolvent read, with a stability $\delta$ from a *directed* theorem, whose $\varepsilon=0$ endpoint is the exact segmentation (letter l). Queries: topology 4/8/16 |
| 15 | segmentation as an exact block certificate | **NEAR-MISS on mechanism; NOT FOUND as a certificate** | mechanism `lin-2025-forgetting-transformer` [V] (body read), `yang-2023-gla` [V], `hwang-2025-hnet` [V]; converse `yang-2026-boundary-repair` [V] Thm 1 | A data-dependent cumulative decay on softmax logits, $D_{ij}=c_i-c_j$ with $c=\mathrm{cumsum}\log f$; a fixed block mask confines reach at every depth. | The **iff**: an exponential prefix scan cannot carry an exact zero (`no_prefix_scan_represents_a_zero_gate`), so the annihilation certificate is the record's and is NOT FOUND outside its Lean (letter k). Queries: topology 7/14 |
| 16 | the truncation certificate $\gamma^{K+1}/(1-\gamma)$ | **OCCUPIED as mathematics; NOT FOUND as an instrument** | `meyer-2000-matrix` [V]; `horn-2013-matrix` [V]; global-bound NEAR-MISS `roffo-2026-infsa` [V] ($\|S_L\|\le\gamma/(1-\gamma)$) | The Neumann tail of a matrix with $\|P\|_\infty\le1$; the spectral-radius convergence condition for a content-adaptive attention series. | A per-$K$ bound **printed beside a truncated attention read**, with a planted non-stochastic negative and the $1/(1-\gamma)$ mask amplification (letter i). No attention paper fetched prints any $\delta$. Queries: resolvent 3, linrec Q6/Q14, methods S4 |
| 17 | intervention effects on persistence | **OCCUPIED** | `kim-2026-topological-causal` [V] | Treatment effects defined on persistence-diagram summaries of the outcome, with a doubly robust estimator and a shape test. | Nothing on the estimand. If CEQ ever reads $\Delta z$ through a barcode it is an instance of this and says so |
| 18 | the Mapper cover | **OCCUPIED** | `singh-2007-mapper` [U on the landing page; dblp `conf/spbg/SinghMC07` and one further record agree]; `carriere-2018-mapper-statistics` [V]; cluster covers `cho-2022-sbm-attention` [V], `roy-2021-routing-transformer` [V], `kitaev-2020-reformer` [V], `yuan-2025-nsa` [V] | Lens, overlapping cover, partial clustering, nerve graph; Mapper as a consistent Reeb-graph estimator with parameters fixable in advance; sampled/hashed/k-means covers deciding who attends to whom. | A **nerve-of-a-lens-cover** as the candidate builder for a *resolvent* read, cover parameters fixed by the Reeb-estimator rule, candidate set carrying the Neumann $\delta$ (letter m). Queries: topology 6/15 |
| 19 | the persistence-derived CSR schedule | **OCCUPIED BY THE AUTHOR** | `sharma-2026-kernels-22` [V] (PR page and files tab fetched; merged 2026-07-28) | A forward-only Triton kernel consuming a causal CSR block schedule built from sink blocks, local windows and a 0D-persistence salience over key-block centroids. | The consumer: the schedule feeds a softmax kernel, forward-only, with no backward and no truncation certificate. The cover and the certified consumer are letter m |
| 20 | the depth law | **OCCUPIED, and conditional where it is a lower bound** | `sanford-2024-logdepth` [V] Thm 4.2 / Cor. 4.3; `sanford-2024-graph-algorithms` [V]; `merrill-2025-littledepth` [V] Thm 2; `fagnou-2024-chacal` [V] Thm 1; `yehudai-2025-depthwidth` [V]; ceiling `merrill-2023-parallelism` [V]; class `cook-1985-taxonomy` [V] | $k$ hops at depth $\lfloor\log_2 k\rfloor+2$ (upper, unconditional); $\Omega(\log k)$ conditional on one-vs-two-cycle; $\lceil\log_2(n+1)\rceil$ for entity tracking; log depth suffices for connectivity; depth unnecessary at linear width; log-precision constant depth $\subseteq\mathrm{TC}^0$; $\mathrm{NL}\subseteq\mathrm{DET}\subseteq\mathrm{NC}^2$. | Nothing. CEQ does not evade the law; it relocates the log-depth into one linear solve, which is a DET-class computation. Every comparison therefore fixes depth *and* width, and three fellow approximators sit in every table |
| 21 | single-location optimality (why the record could not win) | **OCCUPIED** | `marion-2025-single-location` [V] Cor. 2 (erf gate, asymptotic $d\to\infty$, $L=o(d)$); `duranthon-2026-softmax-advantage` [V] Prop. 4.2 / Cor. 4.3 (softmax proper); `zhang-2023-cina` [V] | An attention layer attains Bayes risk on scalar single-location regression; softmax beats linear attention exponentially in the spike; balancing weights are a per-query mixture. | Nothing. This is D-1's owner and the reason the plan's label class is chosen *away* from it. The record's own geometry has $L/d=4.00$, the wrong direction for Cor. 2 (`MATHEMATICS.md` §17.5) |
| 22 | the statistical instruments | **OCCUPIED** | `schuirmann-1987-tost` [V]; `clopper-1934-binomial` [V]; `mcnemar-1947-correlated` [V]; `page-1963-ordered` [V]; `ayer-1955-empirical` [V]; `wilson-1927-probable` [V]; `ville-1939-collectif` [V]; `wang-2023-extended-ville` [V]; `ramdas-2023-game-theoretic` [V]; `hoeffding-1963-probability` [V]; `azuma-1967-weighted` [V]; floors `cover-2006-elements` [V], `welch-1974-lower` [V] | Two one-sided tests; exact binomial and score intervals; the paired-proportions test; ordered alternatives; pool-adjacent-violators; the martingale maximal inequality and e-processes; bounded-increment concentration; Fano and the Welch bound. | Nothing. The plan uses them and claims none. The *pairing* of a sign test against a per-row control on byte-identical draws is the record's and is NOT FOUND outside it (causality Q10) |

**Agreement with §4.0.** Every owner in `judge/sec_shape.md` §4.0 appears above under the same
canonical key and the same verdict; this table adds rows 5b, 6, 12, 14, 15, 16, 17, 19, 20, 21, 22
and splits row 7's fixed-graph owners from its attention near-miss. No row contradicts §4.0.

### 7.1.1 The cite-first sentences

Rule 7 requires that an occupied component be citable in one sentence *before* it is named. These
are those sentences; the assembler places each at the first occurrence of its component.

1. **Read operator.** "The read is that of ChaCAL (`fagnou-2024-chacal`, EMNLP 2024, Eq. 5), with the
   diagonal kept inside the inverse so that the mixing matrix stays row-stochastic."
2. **Triangular solve.** "ChaCAL solves it as a lower-triangular system (Eq. 7); the per-chunk form of
   the same primitive is DeltaNet's (`yang-2024-deltanet`, Eq. 10)."
3. **Discount.** "The discount is Bellman's (`bellman-1957-markovian`), the operator Dayan's successor
   representation (`dayan-1993-successor`); learning one per head is InfSA's (`roffo-2026-infsa`)."
4. **Base family.** "The three corners are Vaswani et al.'s softmax, Katharopoulos et al.'s linear
   attention, and the 1-semiseparable mask of `dao-2024-ssd` Def. 3.1."
5. **Absorbing rows.** "Clamping labelled positions and reading the harmonic extension is
   `zhu-2003-harmonic`; one absorbing row inside attention is `karbalayghareh-2026-doformer`."
6. **Fundamental matrix.** "$N=(I-Q)^{-1}$ and $B=NR$ are Kemeny–Snell (`kemeny-1960-finitemarkov`);
   `grinstead-1997-probability` Thm 11.6 states the same, and is `[U]`."
7. **Committor.** "The committor is the discrete Dirichlet problem of `metzner-2009-tpt-markov-jump`;
   its control-theoretic name is the reach-avoid probability (`summers-2010-reach-avoid`)."
8. **Decision rule.** "Selecting by the worst of several scores is Chebyshev scalarisation
   (`vanmoffaert-2013-chebyshev`); as a safety filter it is `hsu-2023-safetyfilter`."
9. **Re-solve.** "The rank-one re-solve is Sherman–Morrison (`sherman-1950-inverse-adjustment`);
   on a fixed chain with a barrier it is `piray-2021-linearrl` Eq. 5."
10. **Equilibrium layer.** "A layer defined as a fixed point is `bai-2019-deq`; on a causally masked
    transformer with $do()$ by clamp-and-re-solve it is `scetbon-2024-fip`."
11. **Barcode.** "Persistence of a thresholded attention graph is `kushnareva-2021-tda-attention`;
    an intervention read through persistence summaries is `kim-2026-topological-causal`."
12. **Cover and schedule.** "Mapper is `singh-2007-mapper`, its estimator theory
    `carriere-2018-mapper-statistics`; a persistence-derived CSR schedule inside an attention kernel
    is the author's own merged `sharma-2026-kernels-22`."
13. **Depth law.** "$k$ hops need depth $\lfloor\log_2 k\rfloor+2$ (`sanford-2024-logdepth` Thm 4.2;
    the matching lower bound is conditional, Cor. 4.3); the same law is reached independently for
    entity tracking by `fagnou-2024-chacal` Thm 1."
14. **Optimality.** "One attention layer attains Bayes risk on single-location regression
    (`marion-2025-single-location`; for softmax proper, `duranthon-2026-softmax-advantage`)."
15. **Certificate.** "The Neumann tail bound is textbook (`meyer-2000-matrix`); no attention paper
    fetched prints one."

---

## 7.2 The delta, narrowly

No fetched source combines the following in one attention operator, and every part of every clause is
owned above. **(e)** $K\ge2$ absorbing constraint sets, a goal set, and a declared sink as identity
rows of a content-dependent, causal, row-stochastic read — bounded by `sweep_resolvent` searches 17
and 22, `sweep_linrec` Q4/Q11/Q17, `sweep_occupied` S4/S5 (S5 returned **0** hits containing
"committor" in a transformer context), `sweep_methods` S11 and `sweep_topology` 5/13; the nearest
things found are InfSA's single uniform absorption leak, DoFormer's one clamped row, and the attention
*sink*, which is a column device and not a row condition. **(g)** the interventional re-solve with the
displacement as a jointly scored vector label at two stated prices — bounded by `sweep_resolvent` 19,
`sweep_linrec` Q15, `sweep_causality` Q1/Q3/Q6/Q9/Q16 and `sweep_methods` S10; Q9 returned nothing at
all on interventions in deep-equilibrium layers. **(h)** the committor vector and the two safest-move
rules over candidate moves placed in the context — bounded by `sweep_safety` 2/5/10/12/13/17/18/19
(query 14 established that the word "committor" does not occur in the control literature),
`sweep_occupied` S8, `sweep_expressivity` S15 and `sweep_causality` Q11/Q15. **(i)** a printed
certificate with a planted negative and the $1/(1-\gamma)$ mask amplification — bounded by
`sweep_resolvent` 3, `sweep_linrec` Q6/Q14 and `sweep_methods` S4; the inequality is textbook and is
claimed only as an instrument, never as mathematics. **(j)** a learnable discount in a *causal
language-model* read pinned by a boundary-corrected likelihood-ratio test — bounded by
`sweep_resolvent` 2 and 3 and `sweep_linrec` Q1/Q16 (Q16 and `sweep_occupied` S12 each returned **0**
hits for the successor representation used as an attention operator); ChaCAL fixes $\gamma=0.9$,
InfSA learns it and is non-causal vision. **(k)** machine-checked containment of the three corners and
the zero-gate iff — bounded by `sweep_topology` 7 and 14 and `sweep_expressivity` S16; the nearest
statements are FoX's cumulative-log gate, which the record's own theorem excludes at an exact zero,
and the block-mask converse of `yang-2026-boundary-repair` Thm 1. **(l)** the influence-Jacobian
barcode restricted to the F0 endpoint, with a directed stability $\delta$ — bounded by
`sweep_topology` 4/8/16. **(m)** the cover-to-schedule path, Mapper nerve to CSR blocks to a certified
resolvent read — bounded by `sweep_topology` 6/15/2. **The totals.** Across the eight sweeps, **143**
numbered queries were logged (resolvent 22, linrec 18, expressivity 18, causality 18, safety 25,
topology 17, occupied 12, methods 13), plus five bibliographic-repair searches in the methods sweep;
`references.bib` carries **408 canonical entries** (RUN, `grep -c "^@" references.bib` = 408, this
session), of which the audit resolved **278/278** arXiv identifiers and **91/92** DOIs against the
registries, with 38 entries carrying neither identifier; the five sweeps that print an entry-level
`[U]` count report **13** in total (methods 9, occupied 2, topology 1, causality 1, expressivity 0),
and the other three mark `[U]` per field rather than per entry, so no merged `[U]` count exists.
Absence here is exactly as strong as those 143 queries and no stronger, and two of the sweeps paid for
their absences with planted positives — `sweep_topology` §1.1 shows the same instrument returning the
seed papers it was required to find, and `sweep_occupied` §4 plants the record's own dead signed
programme as a lineage the table must return red.

---

## 7.3 The record's own prior-art defect

The paper that owns CEQ's read operator has been in print since October 2024 and the campaign never
cited it. The mechanism is a filter, and the filter is written down. At round 1 the record's
prior-art trawl concluded: *"Every multi-hop propagation found (APPNP 1810.05997, GDC 1911.05485,
MAGNA 2009.14332) requires a non-negative matrix; every signed attention found … is single-hop"*
(`READ DONE_ARCHIVE_ROUND1.md:676-680`). The clause "requires a non-negative matrix" was a
*disqualifier*: the programme then alive was the signed strictly-causal path sum, so a multi-hop
operator built on a non-negative base was recorded as inapplicable and dropped. Every candidate in
the class ChaCAL belongs to was therefore discarded by construction, and the same page later
concedes that the survey sentence built on that filter was false (`READ
DONE_ARCHIVE_ROUND1.md:1046-1050`). When the signed programme died (`RESEARCH.md`, `PROGNOSIS.md`,
per `BRIEF.md` §3), the filter was never re-run without its sign criterion. The shape is now itself a
non-negative multi-hop operator, which is to say the record's search was, for the whole life of the
campaign, structurally incapable of finding the paper that occupies its read.

The measurement of the gap is a grep. `grep -rlEi "ChaCAL|Fagnou|2410\.05565" --include='*.md'` over
the repository returns **0 files** (RUN, 2026-09-03, this box; confirmed twice); a full-tree
`grep -rlEi "2410\.05565"` returns exactly **one** file, `docs/references.bib` — the bibliography
this paper assembles. `2603.00175` (InfSA, which put "fundamental matrix of an absorbing Markov
chain" into an attention abstract in February 2026) appears in **2** `.md` files, both inside the
dead signed programme's archive, where it was read as a signed near-miss and its absorbing-chain
paragraph was not read (`READ DONE_ARCHIVE_ROUND1.md:1317-1318`).

This is `MISTAKES.md` **V-7** — *a search structurally incapable of finding anything, read as
absence* (`MISTAKES.md:117`) — applied to a **literature** search rather than to a code search, which
is the form the taxonomy had never recorded. V-7's existing instances condemn `grep` patterns and
test selectors that could not match; this instance condemns an inclusion criterion in a prior-art
census. It compounds **P-3** (a stale claim never retracted, `MISTAKES.md:316`): `THEORY.md:22` still
calls the successor operator new, and `THEORY.md:225-235` still marks Dayan 1993, Ramsauer 2020 and
Bai et al. 2019 unverified, all three of which the sweeps have now fetched. **The rule the plan
installs against it:** every prior-art census keeps *one mechanism-keyed query per component beside
every name-keyed query*, and no census criterion may name the programme's current hypothesis. ChaCAL
was found by `sweep_expressivity` S6 and `sweep_resolvent` search 2 — queries built from *triangular
solve* and *resolvent*, the shape's mechanism — and not by any query naming an operator family; the
name-keyed query `sweep_resolvent` 18 ("chain and causal attention … citing papers") returned only
ChaCAL's own pages after it had already been found. The census's planted positive is its own
refuted lineage: a table that cannot return a red on the signed programme is not calibrated.

One interval in the sweeps is not supported by anything in the tree. `sweep_occupied.md` §0.1 says
ChaCAL was in print "eleven months before the brief"; the identifier is arXiv 2410.05565, submitted
October 2024, twenty-three months before 2026-09-03, and the tree's own history runs only
2026-08-25 to 2026-09-01 (RUN, `git log --format=%ad --date=short`), so no in-tree date supports
either figure. The paper prints the identifier's date and no interval.

---

# 8. Limits

Collected once, here, and nowhere else.

**There is no result.** No CEQ arm has been trained and no capability number exists; the paper reports a plan and the
identities that justify it. Proposition 8 — the safest move — has no cell, no realised label standard deviation and no
measured hop depth, and the barcode and Mapper rows have no instrument at all (`beta0_interleaving` consumes point
clouds, `READ ceq/certs/topological.py:476-505`). The record's last deciding measurement crossed a one-hop threshold
on five seeds of eight, straddled it on the interval, and that threshold was afterwards shown not to be an information
floor (13 of 40 cells violate it, `V20_R15_THEORY_TABLE.md` §0.2). Every number here is an identity check, a price, or
a citation.

**Theorems cited beyond their hypotheses license nothing, so the hypotheses are printed.** The one-layer composition
bound of `peng-2024-transformer-limitations` Thm 1 requires $n\log n > H(d+1)p$; at $s=64$, $d=16$, one head, float32
this reads $384 < 544$ (DERIVED, $\log_2$), so the inequality is **vacuous** at the record's geometry and the theorem
is cited for its asymptotic shape with those numbers beside it. The same discipline applies to `sanford-2024-logdepth`
Cor. 4.3 and `sanford-2024-graph-algorithms` Thm 3/19, both conditional on the one-vs-two-cycle conjecture, and to
`chen-2024-multilayer` Thm 1.1, whose parameter budget $n^{2^{-4L}}$ reads $1.30$ at $L=1$, $n=64$. The Cheeger
restatement is owed: the record writes $g\le 2\varphi$ at `MATHEMATICS.md:455`, while Levin–Peres Thm 13.10
(`levin-2017-markov-mixing`, read at printed p. 183 by the methods sweep) states
$\Phi_\star^2/2\le\gamma\le 2\Phi_\star$ **for a reversible chain**; the shape's $P$ is causal and not reversible, so the sentence must be
restated on a named symmetrised or lazy surrogate, or replaced by a non-reversible bottleneck bound, and until it is,
no conductance sentence is written about $P$. A second hazard is carried, not resolved:
`misra-2023-safety-constrained-mdp` [V] warns that Bellman optimality can fail in multichain constrained MDPs with
several unsafe sets — the class the safest-move rule inhabits — so the rule is stated as a one-step filter over an
evaluated committor vector and no card claims policy-level optimality.

**Determinism and cost carry their own caveats.** `solve_triangular` was observed bitwise identical forward and
backward over eight repeats under `use_deterministic_algorithms(True)`, but torch documents no such guarantee for it,
so that is an observation on one box and one version, not a contract; `cumsum` raises under the same setting, which is
why the segmentation route is a mask and not a scan. The timings inherit an unsynchronised timer and a run-order
confound — the arms were not interleaved — so every price is a **per-op floor** under a measured dispatch gap of
$2.0\times$ to $6.6\times$ (`READ scale/m3_flops.py:101-121`). The measured increment of the solve over the softmax
head it contains is $2.514-1.473 = 1.041$ ms/step at $n=2048$ ($2.312$ at $4096$, $4.542$ at $8192$), float32, one
box, producer owed to `scripts/k_cert.py`; the chunked and CSR paths are `NOT MEASURED — needs a chunked kernel`, and
the dense control is not runnable at $n=2048$, $s=4096$ ($\approx137$ GB against $7.996$ GiB), so $n$ is stated per
$s$ everywhere.

**The bibliography is verified at the identifier, not at the equation.** The audit resolved 278 of 278 arXiv
identifiers and 91 of 92 DOIs with no identifier paired to a wrong title; the unresolved one is `singh-2007-mapper`,
`10.2312/SPBG/SPBG07/091-100`, absent from Crossref (Eurographics digital library), carried `[U]` with two independent
records agreeing on title, authors and year. Thirty-eight entries carry neither identifier — books, nineteenth- and
twentieth-century papers, unregistered proceedings — and rest on catalogue, ISBN, zbMATH or DBLP marks; nine of those
were never fetched at all (Cantelli 1928, Bonferroni 1936, Frobenius 1912, Neumann 1877, Kantorovich–Rubinstein 1958,
Grünwald 1867, Letnikov 1868, Kramers 1927, Littlewood–Offord 1943) and carry no load-bearing sentence, a textbook
statement standing in where one is needed. **The sweeps' `[V]` marks are abs-page title matches, not equation-level
`[V-eq]` marks.** Only a handful of sources were read at the body — ChaCAL, InfSA, DeltaNet, SSD, Hu et al., FoX, Dash
2005, Bottou §7.3, Grinstead–Snell §11.2, Levin–Peres Thm 13.10 — and every other equation number, theorem number and
hypothesis here was read through a rendered page or a summarising fetch and **must be re-checked against the compiled
PDF before typesetting**, ChaCAL's diagonal convention included, since it rests on one HTML fetch and decides two of
the paper's controls. That is not a formality; it is the record's own hazard. Extracting the LRU paper's §3.3 with
`pypdf` returned text whose Type-1 encoding mapped $-$ to `\x00` and $\infty$ to the glyph `1`, turning an open
interval into a closed one — *"the single sentence that decides that node is the one the extractor corrupts, and it
corrupts it into the claim under test"* (`READ workdonenewseal.md:468-473`; `V15_X36_PRIOR_ART.md:45-56`). The remedy
there was to read every ML source from arXiv LaTeX source, and it is the remedy owed here. One internal inconsistency,
recorded: `sweep_expressivity`'s header says sixteen searches where its own table and count section say eighteen;
eighteen is used.

**The provenance of this document.** It was produced by two multi-agent passes, both cut by a session cap, and the
reader is owed what was lost. In the **first** pass two of three design documents reached the first synthesis and no
refutations reached it at all: `design_instrument.md` (525 lines) and `design_falsify.md` (fourteen bets with counters
and killers) survive as inputs, `design_theory.md` was never written (`THESIS_CORRECTIONS_2.md` §2, `READ`;
`PAPER_OUTLINE.md` §8), and the theory design was produced fresh in the second pass. In the **second** pass the plan
synthesis was the casualty and was re-run separately. So this section was judged against six refutations, all present,
but the plan it points at was assembled on a second attempt rather than in one pass, and no single agent saw every
input at once. The judge's `RUN[J]` numbers are each one numpy float64 draw at $s=32$, $\gamma=0.6$, seed 0, CPU —
identity and counterexample checks, not statistics — and other planets' runs are quoted at their own differing
geometries. No planet wrote a code file or made a git write, so nothing here is reproduced by running this document;
the reproduction path is the plan's first-evening list, and it starts at zero GPU-seconds.
