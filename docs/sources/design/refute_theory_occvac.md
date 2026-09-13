# REFUTATION — THEORY-FIRST design, occupancy / citation / vacuity lens

MARS (MORIARTY), 2026-09-03. Target: `design/design_theory.md` (625 lines, JUPITER/MYCROFT). Inputs read in full and in order:
`BRIEF.md`, `THESIS_NOTES.md`, `THESIS_CORRECTIONS.md`, `THESIS_CORRECTIONS_2.md`, `sections/sec_{proved,measured,refuted,cost,beds,state}.md`,
`sweep/sweep_{resolvent,linrec,expressivity,causality,safety,topology,occupied,methods}.md`, `references.bib` (grep only), `bib_aliases.md`,
`MISTAKES.md` at `207e7b9` (all 2,235 lines). Repository root `<repo root>`.

Evidence classes: `RUN` — executed this session (three read-only probes, text below); `READ path:line`; `CITED [V]` — a key in `references.bib`
whose sweep marks it `[V]`; `DERIVED` — steps written out. Every KILL and REPAIR carries a number, a bib key or a `file:line`. Default under
uncertainty is KILL, and each such line says what would change the verdict.

**RUN this session (one numpy/scipy one-liner, seed 0, `s = 32`, `γ = 0.6`, causal softmax `P` with the diagonal, `d = 4`; one `grep -c` over
`references.bib` for 90 keys; one `Grep` of `lean/CEQ/OracleSeparation.lean`).** Outputs, verbatim: `gamma^32 7.958661109946392e-08`;
`max|(γP)^32| = 7.958661109946394e-08 at (0, 0)`; `max over rows ≥ 1 = 7.958641e-08`; `max diag i ≥ 1 = 6.273028e-13`; diagonal masked with
BOS kept: `max|(γP_m)^32| = 7.958661e-08`. Non-causal row-stochastic `W`, `γ = 0`, `solve_triangular(I − 0·W, V)` then `W @ z`:
`array_equal(W @ z, W @ V) = True`. `Hop β=1` row 0 `= 1.0`; `Hop β=0` row 0 `= 2.0138` at `qk_00 = 0.7`. Vacuity arithmetic:
`64·log₂64 = 384`, `H(d+1)p = 544`, `64^{1/16} = 1.2968`, `h·m·p = 512`. Bib: 89 of 90 keys resolve; `begga-2023-dirichlet` does not — the
canonical key is `begga-2023-diffusion-jump`; three Kemeny–Snell keys coexist (`kemeny-1960-finitemarkov`, `kemeny-1976-finitemarkov`,
`kemeny-1976-finite`). `OracleSeparation.lean:166-168` (`READ`): `oracle_ne_resolvent (A Q) (hA : StrictlyLower A) (hQ : Nonneg Q)
(hsupp : SymmSupport Q) (hij : 0 < Q i j) : Q ≠ A`.

---

## 1. Novelty and citations

### 1.1 Delta components against the eight sweeps — owners the design missed or under-stated

| component the design calls a delta (§5) | owner(s) the sweeps hold that the design omits or under-states | class | verdict |
|---|---|---|---|
| the mixing matrix `Π_γ = (1−γ)W(I−γW)^{-1}` as defined in **§1.2** | `fagnou-2024-chacal` Eq. 5 and `zhao-2026-structuredsparse`'s `𝒮_γ(A)` own this exact object (`sweep_resolvent.md` §2.1-2.2, `sweep_expressivity.md` §2.9). The design attributes it only at **§1.4**, two sections after it is introduced as "the record's operator" — a rule-7 breach (occupied components are cited before they are named) | CITED [V] | REPAIR: move the ChaCAL attribution to the §1.2 display; §1.2's first sentence names ChaCAL |
| the three corners in **§1.1** | corner 1 is `vaswani-2017-attention`, corner 2 is `katharopoulos-2020-linearattention` (`sweep_linrec.md` §4); the design attributes only corner 3 (`dao-2024-ssd`) | CITED [V] | REPAIR: cite all three owners in §1.1 |
| (e) absorbing constraint rows inside a content-dependent read | `begga-2023-diffusion-jump` (Dirichlet formulation with absorbing random walks *inside a GNN layer*, `sweep_occupied.md` §1.3) and `azad-2022-harmonic-extension` (harmonic extension as node classification) are the nearest learned-layer owners after `zhu-2003-harmonic`; neither is cited in §4 | CITED [V] | REPAIR: add both to the §4 occupancy paragraph beside Zhu 2003 |
| (g) consequence as the displacement of an equilibrium | `mooij-2013-ode2scm` (the earliest statement that an intervention's consequence *is* the displaced equilibrium), `bongers-2021-cyclic` (well-posedness vocabulary), `bottou-2013-counterfactual` §7.3 (first-order equilibrium displacement) — all `[V]` in `sweep_causality.md` §2.B, none cited in §1.6 or §4. `vakalis-2026-interventiongap` owns the "oracle-scored intervention effect with a no-change floor" discipline the P9 metric paragraph re-derives | CITED [V] | REPAIR: §1.6 cites Mooij–Bongers–Bottou before "displacement"; P9's metric paragraph cites Vakalis and states the delta as "paired per-row control at matched depth" only |
| (h) safest move over candidate moves | `jeddi-2021-lyapunovsafe` (a Transformer choosing the action with the lowest predicted constraint-violation probability, `sweep_safety.md` §1.G) and `park-2026-maxmin` are NEAR-MISS owners not cited in §1.7 or §4 | CITED [V] | REPAIR: cite Jeddi as the nearest "transformer + violation-probability action choice" and state what it lacks (one constraint, no resolvent) |
| (i) printed Neumann certificate | the inequality is textbook (`meyer-2000-matrix`, `horn-2013-matrix`; `sweep_methods.md` §1.C: "OCCUPIED mathematics / NOT FOUND packaging"); P4 cites no textbook and its status token reads PROVED-HERE | CITED [V] | REPAIR: P4 cites Meyer/Horn for the tail bound and downgrades the status token to "DERIVED (textbook) + RUN" |
| (j) learnable discount | `chien-2020-gprgnn` (learnable, signed hop weights on a fixed graph) is the lineage ParaFormer lifts; not cited | CITED [V] | minor REPAIR |
| (k) zero-gate segmentation iff | `hwang-2025-hnet` (learned chunk boundaries, `sweep_topology.md` §2.E) not cited; the FoX / GLA / boundary-repair owners are | CITED [V] | minor REPAIR |
| (m) Mapper cover → CSR schedule | `cho-2022-sbm-attention` (overlapping sampled cover as who-attends-to-whom — the nearest thing to a Mapper cover as candidate builder, `sweep_topology.md` §3(d)) not cited | CITED [V] | REPAIR: cite before "cover" |
| the hitting-time transform `E[γ^τ]` (§1.5, P3) | the probability generating function of the absorption time is Kemeny–Snell material (`kemeny-1960-finitemarkov` Ch. III, `[U]` locator per `sweep_methods.md` §6); the design presents the identity as its own derivation ("the `E[γ^τ]` identity is new" in `THESIS_NOTES.md` P3 is carried silently) | CITED [V-cat], locator [U] | REPAIR: state "classical (Kemeny–Snell); the *read* form `E[γ^{τ−1}]` on an attention `P` is the composition" |

Every other owner named in the eight sweeps is present in §4; the occupancy paragraph is the design's strongest section. The one
bibliographic hazard: `kemeny-1960-finitemarkov` is used in the design while `sweep_resolvent.md` uses `kemeny-1976-finitemarkov` and
`sweep_methods.md` uses `kemeny-1976-finite`; `references.bib` carries all three as separate entries (RUN: three `@book{kemeny-` lines). The
assembler must collapse them to one canonical key or the reference list prints the same book three times (P-5 / P-7).

### 1.2 Theorems cited as obstructions — hypotheses at `s = 64, d = 16, h = 1, p = 32` and at the proposed beds

| theorem (bib key, number) | hypothesis as the design states it | holds at the record's geometry? | holds at BED-S (`t* ∈ {2,8,32}`, `m = 8`, `K ∈ {2,3,4}`, 4,769 params)? | design's handling | verdict |
|---|---|---|---|---|---|
| `sanford-2024-inductionheads` Thm 1 | `h·m·p = Ω(n)` necessary for one softmax layer at `t* = 2` | `512 ≥ 64` (RUN) — not violated, theorem silent | same: `n = s` of the bed is unspecified; at any `s ≤ 512` silent | stated correctly (§3.2) | KEEP |
| `peng-2024-transformer-limitations` Thm 1 | `n log n > H(d+1)p` | `384 < 544` (RUN) — vacuous | vacuous unless `n ≳ 100` or `p = 16`; BED-S has no `n` fixed | stated correctly (§3.4) | KEEP |
| `chen-2024-multilayer` Thm 1.1 | `H·d·p ≤ n^{2^{−4L}}` | `512 ≤ 1.30` false (RUN) — vacuous | vacuous at every `n < 10⁴` | stated correctly (§3.2) | KEEP |
| `sanford-2024-logdepth` Cor 4.3, `sanford-2024-graph-algorithms` Thm 3/19 | conjecture-conditional, `k = Θ(N^ξ)`, sub-linear width | conditional, asymptotic | conditional, asymptotic | stated as conditional | KEEP |
| `merrill-2023-parallelism` Thm 2, `merrill-2025-littledepth` | log-precision; `TC⁰ ≠ NC¹` | conditional | conditional | stated | KEEP |
| `marion-2025-single-location` Cor 2, `duranthon-2026-softmax-advantage` Prop 4.2 | task model `Y = X_{J₀}ᵀ v* + ξ` (one token's features, latent index), `d → ∞`, `L = o(d)` | `L/d = 4.00` (`READ MATHEMATICS.md:1044-1057` via `sec_refuted.md` §3.0) — wrong direction; **and the label model is not BED-M's**: BED-M's label is a product `a_{s−1}⋯a_{s−t*}·b_{s−1−t*}` of `t*+1` features at `t*+1` positions (`READ scale/negation_scope.py:286-304` via `sec_beds.md` §6.A), which is single-location regression for no `t* ≥ 1` | not applicable by design (vector label) | §3.6 and P7(d) say "one layer is Bayes-optimal on single-location regression, **which every bed in the record was**" — a P-10 / V-25 exposure: the theorem's label model admits 0 of the record's 3 beds; "scalar at one position" is not "single-location regression" | REPAIR: replace "which every bed in the record was" with "no bed in the record leaves the scalar-readout class (D-1 as a mechanism); the SLR theorems are the nearest owned regime and their label model admits 0 of 3 record beds (census printed)" |
| `OracleSeparation.oracle_ne_resolvent` (`OracleSeparation.lean:166`) | `StrictlyLower A` (RUN grep) | the regime-S `P` has `P_ii > 0` (design §1.2, C8) — **hypothesis fails on the shape's own operator** | same | P3's mechanism line claims the theorem "proves the two operator classes disjoint and BED-1 instantiates its hypotheses" | REPAIR: the conclusion `Q ≠ P` for lower-triangular `P` against `SymmSupport Q` is one line DERIVED (an off-diagonal `Q_ij > 0` forces `Q_ji > 0`, which a lower-triangular `P` cannot carry); cite it as DERIVED and add Lean target `lowerTriangular_ne_symmSupport` with hypothesis `LowerTriangular`, not `StrictlyLower` — P-11 in potential form otherwise |
| `Contraction.rowStochastic_perron`, `weighted_contraction` | `RowStochastic P`, `0 ≤ γ` | at `β = 1` rows sum to `1 ± 2.2e-16` (`READ V16_ARM_SMPRIME.md:29` via `sec_measured.md` M.1) | Ruling 2 makes `β` learnable per instance (`READ V17K_RULINGS.md:47-54` via `sec_state.md` S.4); at `β̂ ≠ 1` rows sum to `1.31…10.29` (`READ V16_ARM_SMPRIME.md:28-32`) and every row-stochastic theorem (P3, P4, P5b, P7a) is vacuous on that cell | no census of `β̂` or of `row-sum(P̂)` on trained cells anywhere in the design | REPAIR: P4/P5 ship the domain-census line "`row-sum(P̂)` measured on every arena cell; `‖P̂‖_∞ > 1` ⇒ no certificate is printed" (V-25) |
| `V16Domain.pathProd_eq_zero_iff` (`:129`) | statement about `pathProd m θ` (magnitude-product form) | the design's **§1.1 defines `W` as `exp((scan g)_i − (scan g)_j + qk)`** (`Hop`, `READ V16Domain.lean:366-378`) — the exponential-prefix-scan form that `no_prefix_scan_represents_a_zero_gate` (`:165`) proves **cannot** carry a zero gate. P6's hypothesis `m_c = 0` is satisfiable by no draw of the §1.1 operator | at the softmax corner an exact zero needs a `−∞` logit (mask), which is F0-by-mask, not the gate theorem | P6 states the theorem on "the base with the path-product gate" and RUNs it by masking a block of `P` | REPAIR: define the base in §1.1 with the magnitude product (`ceq/arm_smprime.py:144 path_product`, `pathProd`) or state P6 as "F0 by mask in regime S; F0 by gate only in the `pathProd` base"; V-25 as written |
| `Nilpotent.pow_card_eq_zero`, `occupancy_is_exact_inverse` | `StrictlyLower` | regime N only; the design says so (C8) | — | correct | KEEP |

### 1.3 Sentences that would read as claiming an owned object as new

- **§1.2, the definitions block.** "the record's operator" followed by `Π_γ`, `z`, `O` with no owner named until §1.4. A reader stops at
  §1.2. REPAIR as in 1.1 row 1.
- **P3, Statement + Proof.** The four-line identity `(1−γ)z = E[γ^τ]`, `lim O = q = (I−Q)^{-1}R_k𝟙` is presented as "PROVED-HERE (DERIVED)".
  Every clause is Kemeny–Snell / Grinstead–Snell Thm 11.6 (`CITED [V]`, `sweep_safety.md` §1.A; the repository already concedes zero delta at
  `READ V13_TIER6_PRIOR_ART.md:389-398` via `sweep_safety.md`). The status token must read "CLASSICAL; RUN on the arm's `P`", not PROVED-HERE.
- **P4, Statement.** "The Neumann certificate is attained and has no sum over `s`" — the equality for non-negative row-stochastic `P` is the
  row-sum identity (the design's own C7 says so); the *attainment* is a `V-3` fact and the design labels it; but the heading's "certificate"
  with status PROVED-HERE reads as a result. REPAIR: heading "the textbook tail is attained on this class (V-3 declared)".
- **P5(a), cost line.** "`≈ s²d/2` MACs per head against `≈ s²d` for the softmax head it contains" reads as the solve costing *half* a softmax
  head. `sec_cost.md` §4.x.1 states the honest form: the solve is an **increment** of `+s²d/2` MACs and depth `s` over the `s²d` head, total
  `3s²d/2` (+50 %). M-8 wording.
- **P9, Statement.** The Sherman–Morrison row formula is attributed in §1.6 and §4 but P9's own text says "(Sherman–Morrison)" without the
  key and calls the forward-only fact "DERIVED; `sweep_safety.md` §1.A derives the same" — fine; the `Δz = (I−γP')^{-1}(ΔV + γΔPz)` identity
  is `bottou-2013-counterfactual` §7.3's exact form and `sweep_causality.md` §2.B's derivation — cite Bottou.
- **§3.7.** "the looped block … is the truncated Neumann sum with error **exactly** `γ^{K+1}/(1−γ)` (P4)" — P4's equality is the operator
  ∞-norm of the *matrix* residual; on the vector the bound is `≤ γ^{K+1}/(1−γ)·‖V‖_∞`. The design's own P4 files V-17 against exactly this
  units slip and then commits it three pages later. REPAIR: "with matrix-residual ∞-norm exactly … and vector error at most … `·‖V‖_∞`".
- **§1.3 F1.** "The record's value-zero BOS sink (`V15Fork.Asink`) is the same object seen from the value side (a column sink)" — `sweep_resolvent.md`
  §2.14 says the two "must not be conflated"; a column sink (every row attends to 0) and a row condition (row 0 attends only to itself)
  are different objects that happen to coexist on a causal softmax. "the same object" is the conflation the sweep forbids (P-7). REPAIR: "two
  distinct conditions that coexist on any causal softmax".
- No sentence in §4 or §5 claims the resolvent, the committor, the successor representation, the triangular solve, the sink or the discount
  as new. §5's "none is 'novel'" is correct and should be kept verbatim.

---

## 2. Vacuity by the `MISTAKES.md` taxonomy

### 2.1 Binds — does the rejection region exist, with a planted negative entering at the front door? (V-24, V-14)

| bind | planted negatives the design ships | front-door? | finding | verdict |
|---|---|---|---|---|
| **P1** parity at `γ = 0` | (i) `γ ≠ 0` (`2.3003` coord, `2.5685` here); (ii) "a non-causal `W`"; (iii) wrong normaliser (`β = 0`, `max|gap| > 0.5`) | (i) yes; (ii) **passes** — RUN: non-causal row-stochastic `W` at `γ = 0` gives `array_equal(W@z, W@V) = True`, because `(I − 0·W) = I` regardless of `W`'s support; (iii) breaks "= softmax", not "= `W V`" | plant (ii) is a mutilation the bind cannot see — the V-24 signature (the proof of the bind never mentions causality). Plant (ii) belongs to P5(a)/P9 (`solve_triangular` with `upper=False` on a dense `M` disagrees with the dense inverse), not to P1 | REPAIR: delete (ii) from P1's battery; keep (i) and (iii); add the plant the design itself names as the only separating one — "ChaCAL with the same `γ`" — as a **declared empty-region** row (`V-24` acknowledged at P1, `sec_refuted.md` C2) |
| **P4** certificate | rows `1.5` (`119.37` vs `1.143` coord; `7.29` vs `0.54` here) | the plant enters the certificate arithmetic, not the shipped mask pipeline; `sec_cost.md` §4.x.3's vector-units run (`6.13e-05` vs `≈ 7.6e-05`) carries `‖V‖_∞ ≈ 5` as **[ASSUMED, not printed]** | the design's P4 evidence quotes `≈ 7.6e-05` without the ASSUMED tag (P-1: a number whose producer did not print its input); the row-`1.5` plant fires on the hypothesis, not on the mask (`sec_refuted.md` C6), so the *mask* certificate has no planted negative yet | REPAIR: tag `≈ 7.6e-05 [ASSUMED ‖V‖_∞]`; add the mask plant `sec_cost.md` §4.x.9 #3 names (F1 mask with dropped mass above its printed `δ`) as the front-door negative; L-CERT |
| **P6** segmentation | `1e-300` in place of `0` (`2.4e-298`) | yes, on a masked `P` | the RUN masks a block of a softmax `P` (F0 by mask). The gate-form claim has no draw in the §1.1 base (V-25, §1.2 above). The dividend `31.06×` is DERIVED from a source line whose pair counts disagree by `2×` (`133,120` vs `266,240`, flagged at `sec_cost.md` §4.x.4); the design quotes both counts side by side (P6 Evidence) **without the flag** | REPAIR: carry `sec_cost.md`'s provenance flag; state the base | 
| **P9 / C6** displacement | `V ≡ 𝟙` (planted identity, `Δz = 2.2e-15`), Gaussian `V` (`0.363`), feedback plant (`0.0761`) | yes | correct pair; note the `V ≡ 𝟙` row is a must-pass identity (V-3 type, `uᵀ𝟙 = 0`), not a rejection region — the Gaussian row is the region; the design says so | KEEP |
| **P3** committor | BED-1 real sets `0.0`; the coordinator's `0.858` V-16 artefact retired (`sec_refuted.md` C12) | yes | no planted negative for the *read* (a wrong absorbing-set declaration, a re-targeted row) is listed | REPAIR: add "declare `𝒜_k` on the wrong set ⇒ `q` moves by `O(1)`" as the plant |
| **C8** regime boundary | "`max|(γP)^{32}| = 7.96e-08` with the diagonal, `0.0` strictly lower" | — | **V-4 (fired on the wrong cause).** RUN: `γ^{32} = 7.958661e-08`; the maximum is attained at `(0,0)`, the BOS absorbing row; the largest diagonal entry for `i ≥ 1` is `6.27e-13`; with the diagonal masked and BOS kept the number is **identical**, `7.958661e-08`. The evidence distinguishes nothing about the diagonal; it measures F1 | REPAIR: report `max_{i≥1} (γP)^{32}_{ii}` (`6.27e-13` on this draw) beside `γ^{32}`, and cite `one_not_nilpotent` as the theorem — the RUN then discriminates "diagonal kept" from "BOS absorbing" |

### 2.2 Controls — can the PASS half produce a non-constant label? (V-8, V-12)

- **P3 / P8, the committor vector.** The design's guard is the degeneracy lemma (`max_k q^{(k)} ≥ 1/K` without a goal set) and a
  construction-time census "label sd, goal reachability, class balance, discard count" — **none of which exists** (P8 status OPEN, "no BED-S
  cell exists"). Two structural facts the census will meet, DERIVED here: (i) on a causal softmax `P` every row `i ≥ 1` has `P_{i0} > 0`
  (`exp > 0`), so with `0 ∈ 𝒜_0` the one-step goal probability is bounded below by `P_{i0}` on every draw — `q^{(0)}` has a floor the
  arm reads at zero hops from `P` itself, and `t*` is bounded by the causal window (`≤ i` hops from position `i`); (ii) with `K` constraint
  sets all *before* the query (F2) and every row strictly positive on its window, absorption is a.s. in at most `s` steps, so the "next
  transient state" regime of P10 is short by construction. Neither fact is a kill; both are what the zero-hop control (`sec_refuted.md` C8) will
  measure first, and the design does not predict its value. REPAIR: P8 registers the zero-hop prediction (`1 − 1/m` chance for the argmin,
  `NRMSE ≥ 1 − GATE_TOL` for `q`) as a pre-data number with a counter.
- **F1 (BOS in the goal set).** With `0 ∈ 𝒜_0` the goal set is never empty and never unreachable — good — but it also means every draw has
  the same goal member at the same position; if the corpus adds no other goal nodes, `q^{(0)}` is the probability of hitting position 0
  before any constraint, which on a dense causal softmax is dominated by `P_{·,0}`. Print `sd(q^{(0)})` at construction (V-8). No REPAIR
  beyond the census the design already promises; flagged so the census has a predicted failure mode.
- **P10 pinning instrument.** PASS half: `Λ ≤ 3.841` PINNED. Can it fail? Yes (`> ln n` MOVED). Non-constant. KEEP.
- **P6 census.** `bedM_overlap_new_two = 3 of 3` is a census on `pathProd`'s `{−1, 0, +1}` support — on the *other* base (§1.2 row 9).
  On the §1.1 `Hop` base the admitted fraction for `m_c = 0` is `0 %`, which under V-25's rule "blocks the gate". REPAIR as above.

### 2.3 Theorems — non-empty domain census on the bed (V-25)

Covered in §1.2. Summary: five obstruction theorems correctly censused (KEEP); `oracle_ne_resolvent` cited outside its `StrictlyLower`
hypothesis (REPAIR); row-stochastic theorems with no `β̂` census on trained cells (REPAIR); `pathProd_eq_zero_iff` cited on a base that cannot
instantiate it (REPAIR); SLR theorems cited on a label model 0 of 3 beds satisfy (REPAIR).

### 2.4 Oracles — is the oracle the arm's own resolvent (D-2), or leakable from `x`?

**This is the fatal finding.** P3 and P8 rest on a single sentence (P3 Mechanism, inherited from `THESIS_NOTES.md` P3): "the oracle's `P` is
the latent environment chain built inside the bed; the arm's `P` is computed from tokens by `q, k`; … the two coincide only if the arm learns
the chain, and that coincidence is the capability being tested." Two horns, both mechanisms:

- **Horn A — the arm's `P` cannot be the bed's chain.** BED-S's substrate is "a random walk on a connected graph" (E4′ Rips or `bed_1`'s
  energy graph; `sec_beds.md` §6.C.1) — undirected, `SymmSupport`. The arm's `P` is causal (lower-triangular, `j ≤ i`). A lower-triangular
  matrix cannot equal a matrix with symmetric support that has one off-diagonal positive entry (DERIVED, one line; the theorem
  `oracle_ne_resolvent` says the same for the strictly-lower case). So the arm's exact resolvent read is exact for **its own** `P̂` and never
  for `P_env`; P3's identity licenses nothing about the label; the shape's separate claims — "exactness with a printed `δ`", "one-read joint
  consistency" — are properties of an object the bed does not score. The harmonic residual `r(ẑ) = ‖(I − γP_env)ẑ − V‖` (P9 metric) is then
  a function-approximation score like any other arm's, and "joint determination in one read" is a sentence about `P̂`, not about `P_env`.
- **Horn B — make the chain causal, and the oracle becomes the arm's own resolvent.** If BED-S's environment chain is instead a DAG
  consistent with token order (so that a causal `P̂` can equal `P_env`), then the label `(I − Q_env)^{-1} R_k 𝟙` and the arm's read
  `(1−γ)P̂(I − γP̂)^{-1}𝟙_{𝒜_k}` at `γ ↑ 1` are the **same operator** on the same object, and by `MISTAKES.md` D-2 (`:710-725`) the contrast
  is a reproduction check; the record's `e3_t*` ladder was VOIDed on exactly this shape. The design's P8 line "D-2 (the transition matrix never
  appears in `x`)" is contradicted by `sec_beds.md` §6.C.1, which places **edge tokens with endpoint ids** in `x` — for a uniform random walk,
  `P_env` is a deterministic function of those tokens (`P_{uv} = 1/deg(u)`), so `sec_refuted.md` C2's kill "(a) corpus-alone probe to rows of
  `P*` at `R² ≥ 0.99` makes the bed a copy task" fires **by construction** on the bed as specified.

Either horn makes P8 (and the (e)/(h) delta it carries) un-creditable as written. What would change the verdict: a written embedding that
puts a symmetric walk into a causal `P` — time-unroll the chain, tokens `(v, t)` for `t = 1..T`, edges only across consecutive `t`, `s = n·T`
— then `P̂` is causal, the read is the **finite-horizon** absorption probability `P_i(τ_k ≤ T)` (DERIVED: the unrolled `P` is nilpotent of
index `T`), its distance to the label is `≤ ρ(Q)^T`-class and printable (L-CERT), and `T` is a registered dial (D-3). Under that embedding the
contrast `shape − softmax` is still D-2-adjacent (the arm's class contains the oracle at `T = ∞`) and must be pre-registered as a
**reproduction check**, with the creditable contrasts being `shape − ChaCAL-with-sink` (same class, no boundary rows) and `shape −
depth-matched softmax` (`sec_refuted.md` C1, C3). The design must write the embedding, price `s = n·T`, and list the VOID contrasts *before*
any cell — or KILL P8's capability reading and keep BED-S as the boundary-row parity bind.

### 2.5 Thresholds — fixed before data (M-2)

`3.841`, `ln n` (Ruling 10′, `READ V17K_RULINGS.md:389-436` via `sec_state.md` S.4): fixed. `|γ̂| < 0.05` on `≥ 6/8`, ablation `>` one seed
sd (P10, `sec_refuted.md` C5): pre-registration candidates, stated as such. The `1e-9` argmin tie, `(0.05, 0.95)` argmin-unique band, `R² ≥
0.99 / 0.5` leak kills (`sec_refuted.md` C2, C4) are inherited and labelled. **The §5 pre-registered kill is not a threshold problem but a
reachability problem (M-13):** "if ChaCAL with a sink token matches the committor read **within the TOST margin**" — at the registered `N = 8`
TOST has no passing branch (half-width `0.8807σ` against `0.5σ`; power `0.000` at `N = 8`, `0.80` first at `N = 70`, `READ MISTAKES.md:1210-1247`,
`sec_beds.md` §6.D.3). The kill that decides whether component (e) is a capability **cannot fire** at any `N` the programme runs. REPAIR: state
the kill by the paired `n = 8` MDE cell of the realised sd (`0.039827` at sd `0.034451`, `sec_beds.md` §6.D.2): "`|shape − ChaCAL-sink| <
MDE₈` on the committor NRMSE ⇒ (e) not separated at `N = 8`, escalate to `N = 16`"; never "within TOST".

### 2.6 Predictions — counter of equal specificity, miss signed (D-7, L-SIGN)

The theory file makes three implicit predictions without counters: (i) §3.1 "the harmonic residual separates the arms while per-coordinate
NRMSE does not" (carried from `sec_beds.md` §6.E P3, which *does* file the counter and the SPLIT band — the design must point at it rather than
restate, M-20); (ii) P5's "the exact solve is cheaper than one hop at this geometry" — a measured fact at `s = 64`, extrapolated to "MAC
crossover none" as a law; the counter "at `s ≥ 1024` the depth-`s` chain makes one Neumann hop cheaper than the solve on the 4060" is not
filed (`sec_cost.md` §4.x.3 says NOT MEASURED past `s = 64`); (iii) P10's "`γ̂` distinguishable from `0`" has the C5 kill as its counter — fine.
REPAIR (ii): file the counter with the `s` at which it would decide.

### 2.7 Numbers — a producer for each (P-1)

- Every `RUN[here]` number (`0.5134`, `2.5685`, `8.33e-17`, `3.8e-07`, `4.4e-16`, `7.29 / 0.54`, `6.7e-16`, `−9.9e-17`, `2.4e-298`,
  `0.363`, `8.9e-16`, `0.0761`, `7.96e-08`, `1.78e-15`, `1.03e-15`): "two numpy float64 one-liners … no file written". The one-liners'
  text is not in the file. Under P-1 these are prose until the command is recorded. REPAIR: paste the two one-liners into §7 (markdown is
  permitted; no `.py` is created) — the same discipline this refutation follows above.
- `≈ 7.6e-05` (P4): input `‖V‖_∞ ≈ 5` ASSUMED and unprinted (`sec_cost.md` §4.x.3). REPAIR: tag.
- `2.514 / 1.473 / 3.001 ms` (P5): session RUN with the producer "owed, not done" (`sec_cost.md` §4.x.6 last sentence: must be added to
  `scripts/k_cert.py::determinism_at_64`). REPAIR: tag "producer owed".
- `31.06×` (P6): DERIVED from a fraction whose source line is internally inconsistent by `2×` (`sec_cost.md` §4.x.4). REPAIR: carry the flag.
- `8.9e-16, 595×`: correctly marked as having no producer. KEEP.
- `[0.136, 0.340]`, `0.605`, `0.687` (first pass): instrument-planet RUN, no file; same REPAIR as the first bullet.
- `4,769` params, `3/5/7` depths, `0.5/0.667/0.75` floors: producers named (`CEQ_V20_R15_CONTRACT.md:119`, Thm 4.2 DERIVED, VENUS RUN). KEEP.

### 2.8 "Depth-1 softmax cannot" — matched depth / width / CoT skyline in the same table (D-1, R-SKY)

§3.7 names the three fellow approximators and two controls in prose; P8 states the claim "at matched depth and parameters"; no table in the
design puts the claim and the skyline in one row. The only such table is `sec_beds.md` §6.A (native-skyline column). REPAIR: P7/P8 carry one
table with columns {claim, depth `⌊log₂t*⌋+2`, width (Yehudai: linear in `n`), CoT steps (Merrill–Sabharwal), ChaCAL-same-`γ`, InfSA-no-boundary,
params `4,769`}; a "cannot" sentence outside that table is refused. Note for the width column: at the record's geometry `d_model = 16 < s =
64`, so the wide-constant-depth skyline is *not* instantiated at matched parameters — say so rather than leave the cell blank.

---

## 3. Lean targets (§6) — statement checks

| # | target | finding | verdict |
|---|---|---|---|
| 2 | `bos_row_is_absorbing` : "causal softmax row 0 `= e₀`" | true only at `β = 1`: RUN `Hop β=0` row 0 `= 2.0138` (`Z^0 = 1`, `num = exp(qk_00)`). The statement omits `β = 1` | REPAIR: add `β = 1` |
| 4 | `softmax_corner_not_nilpotent` : `0 < P i i ⇒ (γ•P)^k i i > 0` | false without `0 < γ` and `Nonneg P` (`(P^k)_{ii} ≥ (P_{ii})^k` needs non-negative cross terms) | REPAIR: add both hypotheses |
| 14 | `emc_causal` : suffix re-solve `=` full re-solve for lower-triangular `P'` | also needs `∀ j < i, P' j = P j ∧ V' j = V j`; as stated (any lower-triangular `P'`) it is false | REPAIR: add the prefix-invariance hypothesis |
| 15 | `committor_is_resolvent_read` | one line; correct | KEEP |
| 18 | `reach_avoid_sum_one` | needs "the `𝒜_k` exhaust the non-transient states" (P3 says it; the target omits it) | REPAIR |
| 1, 3, 5, 6, 7, 9–13, 16, 17, 19–21 | — | statements consistent with their carriers; grades `[M]` for 5, 6, 19 rest on `[U]` Mathlib names as the design admits | KEEP |
| new | `lowerTriangular_ne_symmSupport` | the D-2 separation the design attributes to `oracle_ne_resolvent` needs a `LowerTriangular` hypothesis to cover regime S | ADD |

---

## 4. Verdicts

| target | verdict | reason (number / key / file:line) |
|---|---|---|
| §1.1 base family | REPAIR | corners 1–2 unattributed (`vaswani-2017-attention`, `katharopoulos-2020-linearattention`); base defined as the exp-scan `Hop` that cannot carry P6's zero gate (`V16Domain.lean:165`) |
| §1.2 two regimes | REPAIR | `Π_γ` introduced before its owner (`fagnou-2024-chacal` Eq. 5); otherwise sound |
| §1.3 F1 | REPAIR | "same object seen from the value side" conflates column sink and row condition (`sweep_resolvent.md` §2.14) |
| §1.3 F2 | KEEP | triangularity; `0.0` exact |
| §1.4 mixture / factor | REPAIR | "with boundary rows the bare read has row sums `1/(1−γ)`" is true with or without boundary rows (V-23 phrasing) |
| §1.5 horizon dial | KEEP | owners cited |
| §1.6 interventional channel | REPAIR | add `mooij-2013-ode2scm`, `bongers-2021-cyclic`, `bottou-2013-counterfactual` |
| §1.7 decision | REPAIR | add `jeddi-2021-lyapunovsafe` NEAR-MISS |
| P1 | REPAIR | plant "non-causal `W`" passes bitwise at `γ = 0` (RUN `True`) — V-24; delete it, keep `γ ≠ 0` and `β = 0` |
| P2 | KEEP | `0.0` exact; D-2 stated correctly (BED-M contained, not won) |
| P3 | REPAIR | status token PROVED-HERE → CLASSICAL (Kemeny–Snell); `oracle_ne_resolvent` cited outside `StrictlyLower` (`OracleSeparation.lean:167`); no read-side planted negative |
| P4 | REPAIR | cite `meyer-2000-matrix`; `≈ 7.6e-05` unflagged ASSUMED input; no `β̂` row-sum census; mask plant missing |
| P5 | REPAIR | cost stated as "`s²d/2` against `s²d`" instead of "+`s²d/2` over `s²d`" (M-8, `sec_cost.md` §4.x.1); `2.514 ms` producer owed |
| P6 | REPAIR | hypothesis `m_c = 0` unsatisfiable on the §1.1 base (V-25); `31.06×` provenance flag dropped |
| P7 | REPAIR | (d) "which every bed in the record was" — SLR label model admits 0 of 3 beds (P-10/V-25); no skyline table |
| P8 | KILL | D-2 on both horns (§2.4): causal `P̂` cannot equal the undirected `P_env` (label unlicensed by P3), or a causal `P_env` is recoverable from the edge tokens in `x` (`sec_beds.md` §6.C.1) and the contrast is a reproduction check (`MISTAKES.md:710-725`); the §5 kill that would test (e) cannot fire at `N = 8` (M-13). Revives only with the time-unrolled embedding written, priced (`s = n·T`), and VOID contrasts pre-registered |
| P9 | REPAIR | cite Bottou §7.3 and Vakalis 2026; otherwise the strongest proposition in the file |
| P10 | KEEP | corollary; instrument OPEN and labelled |
| C6 | KEEP (label) | (ii) is definitional for lower-triangular systems — label it V-3 declared as P4 does; both halves and the feedback plant are present |
| C8 | REPAIR | evidence `7.96e-08 = γ^{32}` attained at `(0,0)` (BOS), identical with the diagonal masked — V-4; report `max_{i≥1}` diagonal (`6.27e-13`) |
| §3.1–3.4 | KEEP | hypotheses printed; conditional bounds labelled |
| §3.5 obstruction 4 withdrawn | KEEP | correct |
| §3.6 D-1 | REPAIR | as P7(d) |
| §3.7 skylines | REPAIR | "error exactly `γ^{K+1}/(1−γ)`" on a vector (V-17, the design's own P4 rule); `wang-2024-incontext-td` / `xie-2026-softmax-rl` cited without theorem numbers on a load-bearing skyline claim (P-10) |
| §4 occupancy | REPAIR | add `begga-2023-diffusion-jump`, `azad-2022-harmonic-extension`, `cho-2022-sbm-attention`, `chien-2020-gprgnn`, `hwang-2025-hnet`; collapse the three Kemeny–Snell keys |
| §5 delta + kill | REPAIR | the pre-registered kill is unreachable at `N = 8` (TOST); restate by MDE₈ |
| §6 Lean list | REPAIR | targets 2, 4, 14, 18 mis-stated (§3 above); add `lowerTriangular_ne_symmSupport` |
| §7 Limits | REPAIR | paste the two `RUN[here]` one-liners (P-1) |

## 5. Fatal

1. **P8 / component (e)+(h): the oracle and the arm.** No causal `P̂` equals an undirected environment chain (DERIVED; `oracle_ne_resolvent`
   for the strict case), so P3 licenses nothing about BED-S's label; and the only bed geometry under which it could is the one where the label
   is the arm's own resolvent of a matrix determined by the edge tokens in `x` — D-2 verbatim (`MISTAKES.md:710-725`), the mechanism that VOIDed
   `e3_t*`. The design's D-2 line for P8 ("the transition matrix never appears in `x`") is contradicted by the bed it cites.
2. **The §5 pre-registered kill cannot fire.** "Within the TOST margin" at `N = 8` has no passing branch (`MISTAKES.md:1210-1247`); the
   shape can never collapse to ChaCAL-plus-certificate by its own rule at the `N` the programme runs.

Neither is a defect in the algebra; both are defects in what the algebra is allowed to certify. Both are repairable in one evening of markdown
(the unrolled embedding; the MDE₈ kill), which is why the file is graded REPAIR overall rather than struck.

## 6. Limits (once)

The three RUN probes are one seed, `s = 32`, float64, on this CPU, identity checks and not statistics. The bib check is a `grep -c` on `^@type{key,`
and verifies key existence only, not `[V]` status (inherited from the sweeps' notes). Line numbers are at `207e7b9`. The unrolled-embedding
repair in §2.4 is DERIVED and not run; its error bound `ρ(Q)^T`-class is a shape, not a number. No sweep was re-run; every "not cited" claim is a
`grep` of the design file for the key, and every owner claim is the sweep's, at the sweep's own evidence class. No code file, no git write, no
external fetch.
