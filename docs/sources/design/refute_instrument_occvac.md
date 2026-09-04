# REFUTATION — `design_instrument.md` under the OCCUPANCY, CITATION and VACUITY lens

*MARS holding the MORIARTY role, 2026-09-03, against HEAD `207e7b9`. Target:
`$SCRATCH/design/design_instrument.md` (525 lines). Inputs read in full: `BRIEF.md`,
`THESIS_NOTES.md`, `THESIS_CORRECTIONS.md`, `THESIS_CORRECTIONS_2.md`, the six `sections/sec_*.md`,
the eight `sweep/sweep_*.md`, `references.bib` (bib keys and `note` fields for every key the design
cites), `bib_aliases.md`, and `MISTAKES.md` at the headings the task names. Evidence classes: `RUN`
(executed this session, float64, torch 2.5.1, CPU, seed 0), `READ path:line`, `CITED [V]/[U]` by
canonical bib key, `DERIVED` (steps shown). Every KILL and REPAIR carries a number, a bib key or a
`file:line`. Default under uncertainty is KILL, and each such verdict says what would reverse it.*

## 0. The two facts this file rests on, run rather than read

**F-A (the design's §1 numbers are reproducible, so P-1 is repairable).** `RUN`, one-liner:
`torch.manual_seed(0); L = torch.randn(32,32,dtype=float64); P = softmax(L.masked_fill(triu(ones,1), -inf))`,
absorbing rows set to identity. `P[0,0] == 1.0` is `True` at `s = 32` and `s = 64` (R6); with
`{5},{17},{30}` absorbing and BOS undeclared, `ρ(Q) = 1.0`; with `𝒜_0 = {0,30}, 𝒜_1 = {5}, 𝒜_2 = {17}`,
`ρ(Q) = 0.6874342901390162`, `|T| = 28` — the design's R7 digits exactly; `q^{(1)} = q^{(2)} = 0.0` at
positions `1..4` (R10); `64^{1/16} = 1.2968`; zero-hop Fano `0.5 / 0.6667 / 0.75` at `m = 4/8/16`. The
design's "one-liner in this session's transcript" (`READ design_instrument.md:48-49`) is a P-1 hole the
command above closes (`READ MISTAKES.md:289-298`).

**F-B (a causal `P` cannot carry an undirected environment chain).** `DERIVED`, three lines. The
shape's `P` has support in `{(i,j): j ≤ i}` (`READ lean/CEQ/V16Domain.lean:366-378` via
`sec_proved.md` §2.2: `Hop β g qk i j := if j ≤ i then … else 0`). BED-S's oracle chain, as the design
inherits it from `sec_beds.md` §6.C.1, is the E4′ Rips walk or `bed_1`'s energy graph, whose transient
block satisfies `SymmSupport` (`READ sec_proved.md` §2.1: BED-1's `Q` reads `Nonneg True, SymmSupport
True, ρ(Q) = 0.9408612510154677`). If `Q_ij > 0` for some `i > j` then `SymmSupport` demands
`Q_ji > 0` with `j < i`, which a causal support forbids. Hence no draw of an undirected substrate has
a transition structure inside the arm's operator class, at any `γ`, with or without the diagonal —
the softmax-corner extension of `OracleSeparation.oracle_ne_resolvent` (`READ lean/CEQ/OracleSeparation.lean:166`
via `sec_proved.md` §2.1). The design's R6/R10 census lines ("position 0 inside `⊔𝒜_k`", "constraints
precede the query") are facts about the *arm's* token chain, while BED-S's label is computed on the
*environment* chain; the design never states the map between the two and its substrate makes the map
impossible. This is V-25 at the level of the whole bed (`READ MISTAKES.md:1954-2035`: the hypothesis
the mechanism needs is satisfied by `0 %` of draws) and it is the first FATAL below.

## 1. Attack 1 — novelty and citations, row by row

Rule of the brief (`READ BRIEF.md:232-234`): an occupied component is cited before it is named; the
delta is stated narrowly; absence is `NOT FOUND`. The eight sweeps supply the owners; the design's
citation set is checked against them and against `references.bib`.

### 1.1 Citation-class and key defects (mechanical, but the brief calls one of them fatal)

| # | design text | defect | mechanism | repair |
|---|---|---|---|---|
| C-1 | `CITED [V] grinstead-1997-probability` at `:59, :137` (R7 and B-E2) | the bib entry is `[U]`: `READ references.bib:1772` "% [U] ISBN from search index only; AMS bookstore page not fetched"; `sweep_occupied.md` §0.5 and `sweep_safety.md` §1.A both carry it `[U]` on the record, `[V]` only for the LibreTexts host page of §11.2 | P-10 / brief rule 3 (a mis-tagged citation is the class the brief calls fatal) | retag `[U]` or cite `kemeny-1976-finite` `[V]` (Google Books catalogue, `READ sweep_methods.md` §1.C) for `B = NR` |
| C-2 | `sherman-1950-inverse` (`:55, :170`), `kim-2026-topological-causal-effects` (`:361`), `peng-2024-limitations` (`:476`) | alias keys, not canonical (`READ bib_aliases.md:18, :43, :11`) | P-6 | `sherman-1950-inverse-adjustment`, `kim-2026-topological-causal`, `peng-2024-transformer-limitations` |
| C-3 | `sanford-2024-inductionheads` Thm 1, `chen-2024-multilayer` Thm 1.1 cited `[V]` with theorem numbers (`:476`) while §7 admits "not re-read" | theorem numbers come from `sweep_expressivity.md` §0's HTML read (2408.14332 and 2412.02975 are in its full-text list), so `[V]` on the theorem is the sweep's, not the design's; acceptable only if written "per `sweep_expressivity.md` §1.3/§1.6" | P-10 | add the provenance clause |
| C-4 | `momennejad-2017-sr`, `russek-2017-predictive` as the basis of G1's "cached-mixture" control (`:180-183`) | the transition-revaluation *mechanism* is `[U]` (`READ references.bib` note on `momennejad-2017-sr`: "attributed from a search snippet only [U]"; `sweep_causality.md` §2.C says the same) | P-10 | cite as `[U]` for the mechanism; the control must stand on its own operational definition (§2 G1 below) |
| C-5 | `sec_cost.md §4.x.3` vector bound `≈ 7.6e-05` carried at `:276-277` as if measured | the source marks `max|V| ≈ 5` `[ASSUMED … not printed by the run]` (`READ sec_cost.md` §4.x.3) | P-1 | carry the `[ASSUMED]` tag or drop the number |

### 1.2 Owners the design under-states or omits, per delta row

Verdict column: what the design's row would read as claiming, and the narrower sentence the sweeps
license.

- **E1 (`K ≥ 2` absorbing constraint sets as identity rows in the content-dependent causal read).**
  Missed owners: `zhu-2003-harmonic` `[V]` — labelled nodes clamped as absorbing rows with the
  harmonic read `f_u = (I − P_uu)^{-1} P_ul f_l` *as a learning layer*, with a multi-class label matrix,
  i.e. `K` one-hot value channels through one solve (`READ sweep_methods.md` §1.C, §3.3 and its SURPRISE;
  `sweep_resolvent.md` §3.2; `sweep_occupied.md` §1.3). `wu-2012-partially-absorbing` `[V]`
  (per-node absorption in a learning walk). `karbalayghareh-2026-doformer` `[V]` — a do-operator
  *inside attention* in which the intervened token's value is fixed and it is forbidden from
  attending, which `sweep_methods.md` §1.G names "an absorbing row in an attention operator". The
  design cites none of the three; its E1 sentence at `:23` and `:85` would read as claiming absorbing
  rows inside a read as new. The design does cite `fagnou-2024-chacal` and `roffo-2026-infsa` in the
  right places (`:91, :95`). **REPAIR (P-7, brief rule 7):** cite Zhu 2003 and DoFormer before the word
  "absorbing row"; state the delta as "*content-dependent, causal*, `K ≥ 2` prompt-named sets with
  identity rows", which is the `sweep_occupied.md` §3(m) form.
- **E1 identity half against ChaCAL (`:88-91`).** ChaCAL's `A` is the causal softmax *with the
  diagonal removed* per the expressivity sweep's HTML read (`READ sweep_expressivity.md` §2.9, "(diagonal
  removed)"; `[U]` on that detail until checked against the PDF); the shape's `β = 1` corner keeps
  `P_ii > 0` (`sec_proved.md` §2.8(5)). "`𝒜 = ∅ ⇒ torch.equal` with ChaCAL must be `True`" is
  therefore either false by convention or an equality with the design's *own* ChaCAL
  re-implementation on the shape's diagonal — V-3 (`READ MISTAKES.md:72`). **REPAIR:** declare the
  diagonal convention in the manifest, label half (i) as equality with a re-implementation, and make
  ChaCAL's diagonal-removed `A` a *planted negative* (it must break bitwise equality by O(1)).
- **E2 (the goal set).** The reach-avoid object with target and unsafe sets absorbing is owned by
  `summers-2010-reach-avoid` `[V]` and `abate-2008-reachability` `[V]` (`READ sweep_safety.md` §1.D,
  §4.1 item 1); several target sets in one linear system with a safety verdict is
  `baier-2008-modelchecking` (`READ sweep_resolvent.md` §3.5). The design cites
  `vanmoffaert-2013-chebyshev`, `yang-2026-lexisafe`, `fisac-2019-bridging`, `misra-2023-safety-constrained-mdp`
  and omits all three owners of the *object*. **REPAIR:** cite Summers–Lygeros and Abate before
  "reach-avoid vector".
- **G1 (the interventional re-solve).** Box (g) is OCCUPIED on a fixed environment chain by
  `piray-2021-linearrl` Eq. 5 (Woodbury after a barrier; `READ sweep_resolvent.md` §3.7, `[V]` Europe PMC
  full text) and the "fixed point of a causally-masked transformer, `do()` by clamp-and-re-solve" is
  the causality sweep's SURPRISE, `scetbon-2024-fip` `[V]` (`READ sweep_causality.md` §3.4: "a reader who
  has seen only the abstract of the shape will say 'this is FiP'"). Perturbation of the fundamental
  matrix is `schweitzer-1968-perturbation` `[V]` (`sweep_safety.md` §1.A). The design cites Sherman,
  Hager, Dash and omits Piray–Daw, FiP and Schweitzer. Its owner pointer `sweep_causality.md §3.3
  item 1` (`:25`) is the wrong item (item 1 is the read with absorbing rows; the displacement is item 2).
  **REPAIR:** cite Piray–Daw (fixed chain) and FiP (attention, data-generating SCM) in the first
  sentence that writes `do(a)`, with the five FiP differences of `sweep_causality.md` §2.C.
- **G2 (`Δz` as a trained vector label).** The *definition* "consequence = displacement of an
  equilibrium" is owned by `mooij-2013-ode2scm` and `bongers-2021-cyclic`, its first-order form by
  `bottou-2013-counterfactual` §7.3 (`READ sweep_causality.md` §3.1). The design cites only
  `vakalis-2026-interventiongap` and `duranthon-2026-softmax-advantage`. **REPAIR:** cite the three
  definition owners; the delta is "vector label on an attention resolvent scored jointly against a
  matched-depth per-row control" (`sweep_causality.md` §3.3 items 2–3), and the *pairing* — not the
  sign metric — is what is not found (`sweep_causality.md` §2.H).
- **H1 (per-constraint committor reads as head outputs).** The row's title at `:27` and `:221` drops
  "inside a content-dependent causal attention operator", which is the only clause that makes it
  NOT FOUND (`sweep_occupied.md` §3(m)); as written it would read as claiming committor reads as new
  against Zhu 2003 (a layer), `metzner-2009-tpt-markov-jump` and `e-2006-transition-paths` (the
  Dirichlet problem, `READ sweep_safety.md` §2 verdict table) and `kemeny-1976-finite`. The design's
  only citation on the row is `grinstead-1997-probability` `[U]` and `bed_1.py`. **REPAIR:** restore the
  qualifier and cite Zhu, Metzner, E–Vanden-Eijnden, Kemeny–Snell before "committor".
- **H2 (the safest-move rule).** Missed: `jeddi-2021-lyapunovsafe` `[V]` — a Transformer memory plus
  a rule choosing the action with the lowest predicted constraint-violation probability, the nearest
  existing sentence (`READ sweep_safety.md` §1.G, NEAR-MISS); `todorov-2006-lmdp` / `todorov-2009-efficient`
  (a move from a linear first-exit solve, `sweep_resolvent.md` §3.6); `bellman-1957-markovian`
  (argmin over actions on evaluated values) and `altman-1999-cmdp` (`sweep_methods.md` §3.7);
  `park-2026-maxmin`. **REPAIR:** cite Jeddi and Todorov beside Van Moffaert; the delta is the
  in-context packaging (`sweep_methods.md` §3 NEAR-MISS, last bullet).
- **I (the printed certificate).** Consistent with all sweeps (`sweep_methods.md` S4 NOT FOUND as
  packaging, OCCUPIED as mathematics; `sweep_occupied.md` §3(j)). No missed owner. The design's
  "labelled *definitional*" (`:273`) is correct. KEEP on citations.
- **J (learnable `γ` in a causal LM read).** The learnable-discount-in-a-propagation-layer lineage
  is `chien-2020-gprgnn` (learnable, signed hop weights) and `yuan-2025-paraformer` (learnable `γ_k`),
  `roffo-2026-infsa` (per-head sigmoid `γ`); the design cites only ChaCAL's fixed `0.9`. Missing on
  the *skyline* side: `wang-2024-incontext-td` and `xie-2026-softmax-rl` `[V]` — softmax transformers
  implement TD policy evaluation layer by layer, so the deeper stack computes the *same* resolvent by
  iteration; `THESIS_CORRECTIONS.md` C2 names both as mandatory and the design cites neither. **REPAIR:**
  add both to the skyline row of E1/H1/J (R-SKY, `READ CEQ_V16_CONTRACT.md:209`).
- **K (containment and segmentation iff).** The mechanism owners are `lin-2025-forgetting-transformer`
  (cited), `yang-2023-gla` and `hwang-2025-hnet` (not cited); the *converse* theorem — a block mask
  confines reach at every depth — is `yang-2026-boundary-repair` Thm 1 (`READ sweep_topology.md` §3(a)),
  which the sweep says must be cited before "block structure ⇒ no cross-block influence"; not cited.
  **REPAIR:** cite the three.
- **L (influence-Jacobian `β₀` barcode).** Descriptive owners the sweep says must be cited before (b)
  is stated: `kushnareva-2022-betti` (the title "Betti numbers of attention graphs are all you need",
  `READ sweep_topology.md` §4.2), `cherniavskii-2022-acceptability`, `perez-2022-topological-bert`,
  `samaga-2026-halluzig` (zigzag over layers); training-through-persistence: the corrected M10 six
  (`gabrielsson-2019-topology-layer`, `carriere-2019-perslay`, `carriere-2021-optimizing-ph`,
  `hofer-2019-connectivity-optimized`, `moor-2020-topological-autoencoders`,
  `corcoran-2020-ph-gradient-regularization`). The design cites `kushnareva-2021-tda-attention`,
  `kim-2026-topological-causal`, `turner-2019-quasimetric-rips`. **REPAIR:** cite the descriptive lineage
  and the corrected M10 ids (`sweep_topology.md` §4.1 says this is the first correction the assembler
  applies).
- **M (Mapper cover → CSR → certified resolvent).** `singh-2007-mapper` — the object itself — is not
  cited (bib carries it `[U]` on the landing page, records agree); `kitaev-2020-reformer`,
  `yuan-2025-nsa`, `reid-2024-topological-masking` (other candidate builders) are not cited; and
  `zhao-2026-structuredsparse` — the direct occupant of *blockwise evaluation of the same resolvent*
  — is cited in K but not in M, where `sweep_resolvent.md` §2.2 says it and `ceq/multizoom.py` "must be
  cited side by side". **REPAIR:** cite Mapper's origin and Zhao 2026 in row M.
- **The BOS sink sentence (`:66-71`).** "R6 is the one fact in this table the record did not have" is
  record-relative and true; but the reading of the sink as a relaxed absorbing state is in the sink
  literature (`xiao-2023-attentionsinks`, `gu-2024-sinkemerges`, `lee-2026-asap` `[U]` on the phrase,
  `ranmilo-2026-attentionsinks`; `READ sweep_resolvent.md` §2.14, `sweep_linrec.md` §2.7). **REPAIR (P-7):**
  cite the sink papers where BOS-as-row-device is introduced.

### 1.3 Obstruction theorems: hypotheses at `s = 64`, `d = 16` and at the proposed beds

The design's §5 row C10 prints `512 ≥ 64` (`sanford-2024-inductionheads` Thm 1), `384 < 544`
(`peng-2024-transformer-limitations` Thm 1), `64^{1/16} = 1.30` (`chen-2024-multilayer` Thm 1.1)
and rules "`0 %` admitted ⇒ the obstruction is not stated for that bed" — correct and matching
`sweep_expressivity.md` §3.1–3.2 (`RUN` here: `64^{1/16} = 1.2968`). Three residual defects:

1. `sanford-2024-logdepth` Thm 4.2 is used at `:103, :232` to *set the skyline's depth* for BED-S
   (`⌊log₂ t*⌋ + 2`). Thm 4.2 is stated for `hop_k` pointer chasing in a token-defined graph with
   `m = O(1), H = 1`; the reduction `hop_k → committor` is NOT FOUND (`READ sweep_expressivity.md` §3.5,
   §3.6). The depth is a *choice by analogy*, not licensed. **REPAIR (P-10, V-25):** write "depth chosen
   by analogy with Thm 4.2; the theorem does not cover BED-S's task".
2. `yehudai-2025-depthwidth` and `merrill-2024-cot` are cited as skylines with no width or step
   count fixed anywhere in the design; a skyline without a declared width/step budget is a control
   that cannot be built (P-4). **REPAIR:** fix width (`= n` per Yehudai) and CoT step count before
   the arena.
3. `turner-2019-quasimetric-rips` is invoked for row L's `δ` "not Cohen-Steiner's" — but its
   hypothesis (a quasi-metric or an asymmetric function inducing a Rips-type filtration) has not been
   checked against the influence matrix `Π_γ`, which is a similarity, not a distance, and the sweep
   read only the abs page (`READ sweep_topology.md` §2.F). **REPAIR (V-25 on a theorem):** state Turner's
   hypothesis as a predicate on `Π_γ` before the barcode is named, or hold at `NOT MEASURED`.

### 1.4 Sentences that would read as claiming an owned object new

Three: the H1 title at `:27`/`:221` (committor reads, §1.2), the E1 title at `:23`/`:85` without
Zhu 2003 / DoFormer beside it, and the M title at `:374` without `singh-2007-mapper` and
`zhao-2026-structuredsparse` — all REPAIR by citation. `:295` (learnable `γ` in a causal LM read) is
narrow and correct. Nothing in the design calls the resolvent, the successor representation, the
triangular solve, the sink or the discount new; §4 row 11 and §5 C11 forbid it. KEEP that clause.

## 2. Attack 2 — vacuity by the taxonomy, cell by cell

### 2.1 FATAL-1 — BED-S's substrate is outside the arm's operator class (V-25, D-2, V-10)

By F-B, on the E4′ / `bed_1` substrate the arm's `P̂` can never equal `P*`, so three of the design's
own instruments are vacuous by construction: (i) the D-2 kill "`‖P̂ − P*‖_∞ < 1e-3` on `≥ 6/8` ⇒
copied" (`:467`) has an empty rejection region — it can never fire (V-10, `READ MISTAKES.md:168-176`);
(ii) the sentence "the shape's read *is* the committor of the environment chain" is satisfied by
`0 %` of draws (V-25); (iii) the "one read computes all hops" mechanism is not the mechanism that
would produce the label, so any capability number would be a function-approximation reading on a
static bed — exactly C8's regime, which the design's H2 kill then strikes. **KILL** the E1/H1/H2
bed specification as inherited. **REPAIR that would reverse it:** specify the environment as a
random DAG on the *token order* with transitions toward earlier positions (a walk from `s₀` visits
ancestors), node tokens carrying their adjacency row as a multi-hot feature so that a depth-1 bilinear
logit `q_i·k_j = adj_i[j]` can realise `P̂ = P*` at `d_model ≥ n_nodes`, constraints and the goal placed
before the query (R10), BOS in `𝒜_0` (R6). Under that specification (a) the mechanism has a non-empty
domain, (b) the D-2 kill acquires a rejection region, and (c) `sec_refuted.md` C2 leak-clause (a)
("rows of `P*` in the context ⇒ copy task", carried at `:468`) must be *dropped*, because the graph is
the bed's input by definition and the label still requires the solve; keep clause (c) (`R² ≥ 0.5` on
`q` at order 0). Keep `bed_1` / E4′ only as *oracle-instrument* cross-checks (Kirchhoff at `K = 2`),
never as the shape lane's substrate. This REPAIR also requires the design to list, before any cell,
which contrasts are VOID on BED-S under D-2's letter (`READ MISTAKES.md:721-725`): the label is
`(I − Q*)^{-1}R*1` and the arm's read is `(I − Q̂)^{-1}R̂1` — the same operator — so `shape − softmax`
and `shape − skyline` are reproduction-vs-non-reproduction contrasts (VOID as capability numbers),
and only `shape − ChaCAL-with-sink` (boundary mechanism), `shape − Neumann-K` (exactness) and the
identification of `P̂` from context (a learnability reading) are creditable. The design's §4 row 1
lists only `shape − corner-3` on BED-M as VOID.

### 2.2 FATAL-2 — the committor label and the certificate pull `γ` in opposite directions (M-20)

The label of E2/H1 is the `γ = 1` committor (oracle `bed_1.committor`, `READ ceq/beds/bed_1.py:188-198`).
The arm's read at trained `γ̂ < 1` is `E[γ̂^{τ−1} 1_k]` (`THESIS_NOTES.md` P3), which the design's own
R11 measures at `γ = 0.6` as summing to `[0.136, 0.340]` on `T` — an O(1), draw- and position-dependent
shortfall no fixed readout can cancel. The read equals the label only as `γ̂ ↑ 1`, where row J's mirror
kill fires ("`γ̂ > 0.99` on `≥ 6/8` ⇒ the certificate is vacuous", `:317`) and `1/(1 − γ̂)` diverges
(`:311`). Two registrations that never met predict opposite outcomes for the same arm — M-20
(`READ MISTAKES.md:1723`). The E2 certificate line "print `1 − Σ_k q_γ^{(k)}` as the delay share" (`:155`)
records the contradiction; it does not resolve it. **KILL** the E2/H1 label-and-read pairing as
written. **REPAIR that would reverse it:** either (a) the committor head is the Dirichlet solve at
`γ = 1` on the transient block, `(I − Q̂)^{-1}R̂1`, with invertibility certified by the Perron
certificate `ρ(Q̂) < 1` (`sec_proved.md` §2.8(3b) `isUnit_one_sub_of_perron` [S]) and `γ` retained only
on the `z`/`Δz` channel; or (b) the label is redefined as the *discounted* reach-avoid value at a
fixed, pre-registered `γ` (`fisac-2019-bridging`, `hsu-2021-reachavoidrl`) and the word "committor" is
withdrawn from E2/H1. Either choice is one sentence; the design must take one.

### 2.3 Binds — rejection region, front door (V-24, V-14)

| bind | rejection region exists? | enters at the front door? | verdict |
|---|---|---|---|
| B-E1 (i) `𝒜 = ∅` bitwise vs ChaCAL | no — V-3 by construction and by the diagonal convention (§1.2) | no: identity-script level | REPAIR as §1.2 |
| B-E1 (ii) `𝒜 ≠ ∅` ⇒ O(1) gap downstream | true of any row edit that is reachable; a V-3 of the algebra, not of the arm | no | REPAIR: relabel "instrument test", not "bind" |
| B-E1 (iii) non-identity absorbing row breaks `Σ_k q = 1` | yes (breaks conservation) | no | KEEP as instrument test |
| B-E1 (iv) InfSA base breaks parity | `NOT MEASURED` (design says so) | — | KEEP as owed |
| B-E2 drop `𝒜_0` ⇒ `max_k q ≥ 1/K` | identity of `Σ = 1` (V-3); a demonstration, not a bind | no | REPAIR label |
| B-E2 drop BOS ⇒ solve must raise | yes (a guard with a witness, `ρ(Q) = 1.0` RUN) | builder-level: yes | KEEP |
| B-G1 (a)–(e) | (a) identity; (b) non-empty; (c),(e) exact algebra — V-3; (d) feedback plant `NOT MEASURED` | no: all on random logits, none through `arm.forward` | REPAIR: run the battery through `bed_s.build → arm.forward → journal`, hash it into the manifest |
| B-G2 metric plants (i)–(iii) | yes (V-26 defeated) | metric-level: yes | KEEP |
| B-H1 oracle identity `0.0` | V-3 of the builder, labelled so | oracle-level | KEEP as labelled |
| B-H2 argmin identity | V-3, labelled | — | KEEP as labelled; leak plants enter at the corpus (front door) — KEEP |
| B-I (i)–(iii) | (i) yes; (ii) yes; (iii) V-3 (labelled) | (ii) on the shipped mask — but no mask ships before row M (§2.6) | REPAIR: mark row I dormant until M |
| B-J `γ = 0` bitwise | empty against ChaCAL (`READ sweep_resolvent.md` §9.2) — the design knows | — | KEEP as definitional; see §2.5 for the kill |
| B-K Lean targets | a build is its own witness; `one_not_nilpotent` plant is real | — | KEEP |
| B-L `ε = 0` endpoint = segmentation count | **empty on BED-S**: see §2.6 | — | KILL on BED-S |
| B-M fill-in guard, do-nothing entered | yes (`READ ceq/mz_kernel.py:10-21`) | kernel-level | KEEP (unpriced) |

The global V-14 finding: every plant in R1–R12 exercised the *matcher* (resolvent algebra on random
logits), never the *reach* (arm forward, readout, journal row, `verdict()`; `READ MISTAKES.md:272-279`).
**REPAIR (V-14):** one planted-negative cell per row, produced by the banked cell's own script with
`kind`, `manifest_hash`, `instrument_hash`, FOUND under `results/` (`READ V20_R15_WING_MANIFEST.md:59-63`).

### 2.4 Controls — can the PASS half produce a non-constant label? (V-8, V-12)

- **E2 kill "argmin-unique fraction outside `(0.05, 0.95)`" (`:161-162`).** As written, a healthy
  bed whose argmin is unique on `100 %` of admitted draws (ties are discarded at `1e-9`, `:160`) is
  killed. The gate is inverted (V-10 in mirror). **REPAIR (M-2, frozen here):** "every class frequency
  of `a*` over `m` lies in `(0.05, 0.95)`" — which at `m = 8` (uniform `0.125`) is the intended check.
- **G2 census "`Var(Δz) > 0` per coordinate; `0` blocks registration" (`:215`).** `Δz[:i] = 0` exactly
  for the intervened row `i` (R4, RUN `4.4e-16`), and `Δz_0 = 0` on every draw since row 0 is `e₀` (R6).
  So coordinate 0 has `Var = 0` on `100 %` of draws and the rule blocks G2 by construction (V-8,
  `READ MISTAKES.md:146-152`). **REPAIR:** census over coordinates `≥ i_min` where `i_min` is the earliest
  intervened row in the batch, and print the zero-coordinate fraction.
- **H1 control "ChaCAL same-`γ` (no reads exist — it must emit chance)" (`:231`).** A control that must
  fail by construction is V-2 / V-10; and it contradicts E1's kill, which supposes ChaCAL-with-sink
  *can* match the committor read (`:126`). With the same `K+1`-way head, trained on the same label,
  ChaCAL is an arm like any other. **REPAIR:** ChaCAL + the same head, trained; delete "must emit
  chance".
- **G1 "cached-mixture arm" (`:180-183`).** A feed-forward attention arm recomputes `P̂` every forward;
  "re-reads `(I − γP)^{-1}` without re-solving after a `ΔP`" has no referent inside the harness (P-7).
  **REPAIR:** define it as `O_cached = P̂_base (I − γP̂_base)^{-1} V_int` — `P̂` frozen from the
  un-intervened context, values from the intervened one — which is what a cached SR computes and
  gives the Momennejad/Russek plant an operational form.
- **E2 census "`𝒜_0` reachable" (`:157`).** With BOS ∈ `𝒜_0` every un-absorbed walk ends at BOS
  (R7), so the precondition holds at every draw (V-11, `READ MISTAKES.md:178-187`). KEEP the print only.
- **H2 floor "zero-hop Fano `0.6667` at `m = 8`" (`:260`).** The floor assumes `I(X_{≤0}; a*) = 0`, but
  the zero-hop view includes the move token `(v_a, u_a)` and the membership flags of `u_a`
  (`READ sec_beds.md` §6.C.1); a move whose target lies in `𝒜_k` is legibly unsafe at zero hops, so
  `I > 0` and the printed floor is above the true floor (V-25 on a floor; V-10 on the C8 kill that
  compares to "the Fano floor's resolution", an undefined quantity). **REPAIR:** compute the plug-in
  `I(X_{≤0}; a*)` over the full zero-hop view, print `1 − (I + ln 2)/ln m`, and define "resolution" as the
  CP half-width at `N = 8`.

### 2.5 Thresholds, oracles, predictions, skylines (M-2, D-2, D-7, D-1/R-SKY)

- **Thresholds.** Every numeric threshold in §2 is frozen in the file (`6/8`, `2×`, `0.05`, `0.95`,
  `1e-9`, `1e-12`, `1,024`, `≥ 200`, `N = 70`) — M-2 satisfied on the face. Two are out of their units:
  J's "`|γ̂| < 0.05` on `≥ 6/8`" sits beside Ruling 10′'s `Λ ≤ 3.841` (`READ V17K_RULINGS.md:389-400`)
  as a *second* pinning criterion, and a cell can be MOVED by `Λ` with `γ̂ = 0.04` — two verdicts for
  one cell (M-20 in small; V-17 on the `0.05`). **REPAIR:** keep `Λ`, delete the `0.05` rule.
- **E1 kill's TOST tier (`:130-131`, `:489-491`, step (6)).** If the `N = 8` kill fires, E1 is dead;
  if it does not fire, a difference was detected and equivalence testing has nothing to decide. The
  `N = 70` TOST at `≈ 298 s` is therefore a step with no branch that consumes it (M-7,
  `READ MISTAKES.md:529-541`). **REPAIR:** state what TOST decides — presumably the *parity* half
  (shape at `γ̂` vs ChaCAL at the same `γ̂` with `𝒜 = ∅`), which is a different contrast from the kill's.
- **H2 certificate "a move is admitted … only if `δ_thr − δ·‖V‖_∞ > 0`" (`:261-262`).** `δ` is the
  truncation error (`0` on the exact route); the model error `P̂ ≠ P*` has no certificate, so "admitted"
  presents an uncertified quantity as a safety guarantee (V-17 across two error sources; P-7).
  **REPAIR:** "the certificate bounds the solve, not the model; no move is admitted by it".
- **Predictions and counters (D-7, L-SIGN).** §4 row 9 says "every kill in §2 is the counter to a
  prediction of equal specificity" — but the *prediction* half with its number is written nowhere in
  the design for rows E1, E2, G1, G2, H2, I, J, K, L, M; only H1 points at `sec_beds.md` P3. A counter
  without its prediction leaves no residual to sign (`READ MISTAKES.md:2074-2082`). **REPAIR:** one
  line per row: "prediction: `<statistic> <direction> <threshold>`; counter: the kill as written",
  both filed before step (4) of §6.
- **Skylines (D-1, R-SKY).** E1 and H1 carry the three skylines; G1, G2 and H2 do not (G1: per-row
  control only; G2: MuZero value head; H2: 0-hop, 1-hop, DT). `READ CEQ_V16_CONTRACT.md:209` makes
  `Δ_sky` mandatory on every bed. **REPAIR:** add the depth-`⌊log₂ t*⌋ + 2` stack, the wide stack and
  the CoT decoder to G1/G2/H2, with width and steps fixed (§1.3 item 2).
- **Matched parameters `4,769` (`:80`, `READ CEQ_V20_R15_CONTRACT.md:119`).** A `[m, K+1]` head or a
  vector readout changes the count; the design carries `4,769` unreconciled (Ruling 3, `READ
  V17K_RULINGS.md:56-59` via `sec_state.md` S.4). **REPAIR:** print the count per arm.

### 2.6 Rows L, K, I, M — domain census on the bed (V-25)

- **L (KILL on BED-S).** `DERIVED`: the filtration is `Π_γ = (1−γ)P(I−γP)^{-1}` (R12). For a softmax
  `P` every `P_{i0} > 0` (no `−∞` logit), so `Π_{i0} ≥ (1−γ)P_{i0} > 0` for every row `i`, absorbing
  rows included as *columns*; hence the `ε = 0` influence graph is weakly connected on `100 %` of draws
  and `β₀(ε = 0) = 1` — or, if `β₀` is read as strong components on a digraph, `= s` on `100 %` of draws
  (a causal chain with self-loops has no cycle). The design's own census line (`:369-370`) names this
  case as "decoration", and on BED-S (parity corner, `g ≡ 0`, no gate) it is the *only* case. The
  "`ε = 0` endpoint must equal the F0 segmentation count" bind is then `1 = 1` on every draw (V-3,
  V-25). Further, the instrument named — `beta0_interleaving` (`READ ceq/certs/topological.py:476-505`)
  — takes a point cloud and rotates it on `S²` (`_rotate_on_sphere`, `toolkit_edges`); it cannot consume
  an influence matrix (P-4, claimed scaffolding). **KILL** row L as specified on BED-S. **What would
  reverse it:** restrict L to gated corners with exact zeros (BED-M, `96.78 %` annihilation,
  `READ workdonenewseal.md:92`), or redefine the object as the *thresholded* barcode with the M-15
  permutation null — which is the occupied Kushnareva object with the influence filtration as the narrow
  delta (`sweep_topology.md` §3(b)) — and write "`NOT MEASURED — needs a digraph `β₀` instrument`".
- **K (REPAIR).** `pathProd_eq_zero_iff` and `segmentation_blockdiag` [M] (`StrictlyLower A`) need
  an exact zero gate; the lane's parity corner is `β = 1, g ≡ 0` with the diagonal kept (`:297,
  :416-418`; `one_not_nilpotent`), so on BED-S the K row's domain is `0 %`. Confining K to BED-M
  (`:337`) is right; **REPAIR:** say the theorems are silent on the lane's own read and gate BED-M only.
- **I (REPAIR).** "Every bed on which a mask ships" (`:282`) — in steps (1)–(6) of §6 the route is
  `solve_triangular` (`δ = 0`); a Neumann or CSR mask exists only in row M, which is last and
  `NOT MEASURED`. Row I's `1,024`-draw measurement has no object until then (D-4, registration
  without admission). **REPAIR:** mark I dormant until M ships a mask; keep the F0 `torch.equal` on
  the exact route as the only live clause.
- **M (REPAIR, P-8).** With `P` explicit the bed at `s ∈ {256, 1024, 4096}` is `O(s²)` memory
  (`READ sec_cost.md` §4.x.7): at `n = 2048, s = 4096` float32 the operator alone is `2048·4096²·4 B ≈
  137 GB` (`DERIVED`) against `7.996 GiB`, so the "dense resolvent at the same `s`" control (`:387`) is
  not runnable at the arena's `n`; and the kill's "MDE at a matched visited-tile budget" names no
  metric (V-17). **REPAIR:** state `n` per `s` and the metric the MDE is in.

### 2.7 The RUN table (§1) and the manifest (§3)

- **P-1:** R1–R12 have no producer in the file; F-A supplies it for R6, R7, R10 and the design must
  carry the one-liner. R8, R9, R11 were not re-run here and remain single-draw instances, as §7 says.
  KEEP with the producer added.
- **R9's sd `[0.083, 0.073, 0.029]`** includes the prefix before the first constraint, where
  `q^{(0)} = 1` exactly (`RUN`: `0.9999999999999997` at positions `1..4`); the census must be over the
  admitted query region or a constant block inflates it (V-8). REPAIR the census definition.
- **Manifest (§3).** KEEP: both rules repair L-2 (`READ V20_R15_LEAP_LEDGER.md:23`; `READ
  scale/identity_manifest.py:141-151`). REPAIR: add `diag_convention ∈ {kept, removed}` (§1.2) and
  `committor_route ∈ {gamma_limit, dirichlet_gamma1}` (§2.2), or the two fatal ambiguities are un-attributable cells.

## 3. Verdicts

| # | target | verdict | one-line reason (number / key / file:line) |
|---|---|---|---|
| 1 | BED-S substrate for E1/H1/H2 (`:107-108`, inherits `sec_beds.md` §6.C.1) | **KILL** | F-B: causal support vs `SymmSupport` (`sec_proved.md` §2.1, `ρ(Q) = 0.9408612510154677`); D-2 kill `‖P̂ − P*‖ < 1e-3` has an empty rejection region; reversed by the DAG-in-token-order spec of §2.1 with C2(a) dropped and the VOID list written |
| 2 | E2/H1 label (`γ = 1` committor) vs J mirror kill (`γ̂ > 0.99`) | **KILL** | M-20: R11 delay share `1 − [0.136, 0.340]` at `γ = 0.6`; reversed by the Dirichlet-at-`γ = 1` head (§2.2 (a)) or a fixed-`γ` discounted label (§2.2 (b)) |
| 3 | Row L on BED-S | **KILL** | `β₀(ε = 0) ∈ {1, s}` on `100 %` of softmax draws (§2.6 DERIVED); `beta0_interleaving` consumes point clouds (`READ ceq/certs/topological.py:476-505`); reversed by restriction to gated corners or the thresholded-barcode redefinition |
| 4 | `CITED [V] grinstead-1997-probability` (`:59, :137`) | **REPAIR** | `READ references.bib:1772` is `[U]`; retag or cite `kemeny-1976-finite` |
| 5 | alias keys `sherman-1950-inverse`, `kim-2026-topological-causal-effects`, `peng-2024-limitations` | **REPAIR** | `READ bib_aliases.md:18, :43, :11` |
| 6 | E1 citations and the ChaCAL identity half | **REPAIR** | add `zhu-2003-harmonic`, `karbalayghareh-2026-doformer`, `wu-2012-partially-absorbing`; declare the diagonal convention; ChaCAL's diagonal-removed `A` as the planted negative (`sweep_expressivity.md` §2.9) |
| 7 | E2 citations | **REPAIR** | add `summers-2010-reach-avoid`, `abate-2008-reachability`, `baier-2008-modelchecking` |
| 8 | E2 kill "argmin-unique fraction outside `(0.05, 0.95)`" | **REPAIR** | inverted gate; replace by class-frequency bounds (§2.4) |
| 9 | G1 citations, owner pointer, cached-mixture control | **REPAIR** | add `piray-2021-linearrl` Eq. 5, `scetbon-2024-fip`, `schweitzer-1968-perturbation`; pointer is `sweep_causality.md` §3.3 item 2; define `O_cached` (§2.4) |
| 10 | G1 bind battery location | **REPAIR** | V-14: run through `bed_s.build → arm.forward → journal`, hashed |
| 11 | G2 `Var(Δz) > 0` per coordinate | **REPAIR** | `Δz_0 = 0` on `100 %` of draws (R4, R6); census over coordinates `≥ i_min` |
| 12 | G2 citations | **REPAIR** | add `mooij-2013-ode2scm`, `bongers-2021-cyclic`, `bottou-2013-counterfactual` |
| 13 | H1 title wording and citations; ChaCAL "must emit chance" | **REPAIR** | restore "inside a content-dependent causal attention operator"; cite Zhu/Metzner/E/Kemeny; ChaCAL trained with the same head |
| 14 | H1/E1 harmonic-residual metric on a `[m, K+1]` label | **REPAIR** | `r(q̂)` needs `q̂` on all of `T` (`READ sec_beds.md` §6.B.3); label must be the full vector or P3 is unscoreable (P-1) |
| 15 | H2 citations; zero-hop Fano floor; "admitted" certificate | **REPAIR** | add `jeddi-2021-lyapunovsafe`, `todorov-2006-lmdp`, `bellman-1957-markovian`, `altman-1999-cmdp`; compute `I(X_{≤0}; a*)`; the truncation `δ` admits nothing |
| 16 | Row I | **KEEP** (with one repair) | citations and definitional label correct; mark dormant until M ships a mask; carry the `[ASSUMED]` on `7.6e-05` |
| 17 | Row J | **REPAIR** | delete the `|γ̂| < 0.05` rule, keep `Λ`; add `chien-2020-gprgnn`, `yuan-2025-paraformer` (learnable-discount lineage) and `wang-2024-incontext-td`, `xie-2026-softmax-rl` (skyline) |
| 18 | Row K | **REPAIR** | add `yang-2026-boundary-repair`, `yang-2023-gla`, `hwang-2025-hnet`; state the empty overlap with the parity corner and BED-S |
| 19 | Row M | **REPAIR** | add `singh-2007-mapper`, `zhao-2026-structuredsparse`, `kitaev-2020-reformer`, `yuan-2025-nsa`; state `n` per `s` (137 GB at `n = 2048, s = 4096`) and the kill's metric |
| 20 | Skyline depth from `sanford-2024-logdepth` Thm 4.2 | **REPAIR** | by analogy only (`sweep_expressivity.md` §3.5); fix width and CoT steps for the other two skylines |
| 21 | Predictions beside kills (§4 row 9) | **REPAIR** | D-7: no prediction half exists for ten of eleven rows |
| 22 | E1 TOST tier at `N = 70` | **REPAIR** | M-7: no branch consumes it; state the contrast it decides |
| 23 | `Δ_sky` on G1/G2/H2 | **REPAIR** | R-SKY (`READ CEQ_V16_CONTRACT.md:209`) |
| 24 | §1 RUN table | **KEEP** (producer added) | F-A reproduces R6/R7/R10 to the digit; R9 census region to be re-stated |
| 25 | §3 manifest | **KEEP** (two fields added) | `diag_convention`, `committor_route` |
| 26 | §4 / §5 answer tables | **KEEP** with rows 1, 4 (C1, C2) amended per §2.1 | VOID list on BED-S; C2(a) dropped |
| 27 | §6 price and order | **KEEP** | prices carry `[FITTED + RUN]` from `sec_cost.md` §4.x.8; the DERIVED TOST price is correct arithmetic and marked as a floor |
| 28 | Obstruction inequalities (C10) | **KEEP** | `512 ≥ 64`, `384 < 544`, `64^{1/16} = 1.30` (RUN `1.2968`) correctly printed as vacuous at `s = 64, d = 16` |

## 4. Limits of this refutation

Every RUN is one draw at `s ∈ {32, 64}`, float64, CPU, seed 0, on random logits; it reproduces the
design's R6/R7/R10 digits and nothing else in §1 was re-run. F-B is a support argument; the DAG repair
of §2.1 is a specification whose adjacency-feature representability at `d_model ≥ n_nodes` is
unmeasured. The ChaCAL "diagonal removed" detail is the expressivity sweep's HTML reading, `[U]` until
the PDF is read. The `β₀` argument of §2.6 assumes the parity corner (no `−∞` logits); a gated corner
is not killed. No citation was fetched this session; every key and tag is `references.bib`'s. No code
file, no git write, no Kaggle contact; the one Python invocation computed and printed only.
