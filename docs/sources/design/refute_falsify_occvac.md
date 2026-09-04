# REFUTATION — `design_falsify.md` under the OCCUPANCY, CITATION and VACUITY lens

*MARS / MORIARTY, 2026-09-03, HEAD `207e7b9`. Target: `$SCRATCH/design/design_falsify.md` (498
lines, VENUS). Inputs read in full: `BRIEF.md`, `THESIS_NOTES.md`, `THESIS_CORRECTIONS.md`,
`THESIS_CORRECTIONS_2.md`, the six `sections/sec_*.md`, the eight `sweep/sweep_*.md`,
`references.bib` (key list), `bib_aliases.md`, and the 66 `###` headings of `MISTAKES.md`.
Evidence classes: `RUN` (numpy one-liners executed this session, this box, float64, compute
only, no file written), `READ path:line`, `CITED [V]` by canonical key in `references.bib`,
`DERIVED` (steps shown). Every verdict carries a number, a bib key, or a `file:line`. Default
verdict under uncertainty is KILL; each KILL names what would change it. The design is dated
before `THESIS_CORRECTIONS_2.md` (its own §0 says so and CORRECTIONS_2 §2 lists it as an input),
so the two construction facts F1/F2 of CORRECTIONS_2 §1 are applied here as attacks, not as
faults of authorship.*

---

## 0. What was run this session (the two checks the verdicts lean on)

`RUN-1` (BED-S oracle at the design's own geometry, `s = 64`, causal softmax `P` from
`N(0,1)` logits, seed 0, absorbing sets `𝒜_1 = {30, 31}`, `𝒜_2 = {50}`, goal `{40}`, BOS
row 0 undeclared): `ρ(Q) = 1.000000`, `min|eig(I − Q)| = 0.000e+00` — the committor solve is
singular. With BOS added to the goal set: `ρ(Q) = 0.525557`, `Σ_k q^{(k)} ∈ [0.999999999999999,
1.000000000000000]` on `T`. This is CORRECTIONS_2 F1 reproduced at `s = 64`.

`RUN-2` (same chain, BOS in goal): committor into every set that sits *after* the query
position reads exactly `0.0` (`q^{(50)}` at query 20: `0.000e+00`; `q^{(30,31)}` at query 20:
`0.000000`); the positive control at query 60 reads `q_c1 = 0.0846, q_c2 = 0.0206, q_goal =
0.8948`, at query 35 `0.0654 / 0.0000 / 0.9346`. CORRECTIONS_2 F2 reproduced, with a planted
positive (V-7).

`RUN-3` (mixing matrix `(1−γ)P(I−γP)^{-1}`, `γ = 0.5`): with absorbing rows, min entry `0.0`
on the full matrix *and* on the lower triangle, `151` exact zeros in the lower triangle, row
sums `[1.000000000000000, 1.000000000000000]`; without absorbing rows the lower-triangle
min is `2.630e-04` and there are `0` exact zeros. So "min entry reads exactly 0" is decided by
the identity rows (and by the mask on the full matrix) — a theorem instance, not a reading.

`DERIVED-1` (Sherman–Morrison pricing at `m = 8, d = 16, s = 64`): `m` columns of
`M = (I−γP)^{-1}` cost `m·s²/2 = 16,384` MACs against the full solve's `s²d/2 = 32,768`; the
ratio is `m/d = 0.500`, not the design's `≤ 2×` (§3.2) — see §3.10 below.

---

## 1. Attack 1 — novelty and citations

### 1.1 Keys that are not canonical (`bib_aliases.md`; the coordinator's rule is CANONICAL keys)

| design cites | canonical key | where |
|---|---|---|
| `summers-2010-reachavoid` | `summers-2010-reach-avoid` | §2.2 |
| `prinz-2011-markov` | `prinz-2011-msm` | §2.3 |
| `sherman-1950-inverse` | `sherman-1950-inverse-adjustment` | §3.2 |
| `duranthon-2026-softmaxadvantage` | `duranthon-2026-softmax-advantage` | §5 rank 4 |

Four of the design's thirty-odd keys are aliases. The design's header also points at
`$SCRATCH/sweep/bib_*.bib` rather than `references.bib`. Mechanism: P-5 (doc rot pointing at
nothing) in potential form — an alias that a later dedupe deletes becomes a dangling key.
`references.bib` itself carries three Kemeny–Snell keys (`kemeny-1960-finitemarkov`,
`kemeny-1976-finite`, `kemeny-1976-finitemarkov`) and two Altman keys not in the alias table;
that is the coordinator's, recorded here so the assembler does not cite two of them as two books.

### 1.2 §0.3 / §7 — the owner list the design says it has, and does not

§7 states that the resolvent read, the Neumann series, the absorbing-chain reading, the
committor identity, the successor representation, the rank-one re-solve and the reach-avoid
rule are "each OCCUPIED with its owner in §0.3". §0.3 names six owners (`fagnou-2024-chacal`,
`roffo-2026-infsa`, `gasteiger-2019-appnp`, `zhu-2003-harmonic`, `piray-2021-linearrl`,
`scetbon-2024-fip`). Absent from §0.3, all `[V]` in the sweeps: the successor representation
(`dayan-1993-successor`; `bellman-1957-markovian` for policy evaluation, `sweep_methods.md` §3
item 1); the Neumann tail (`meyer-2000-matrix`, `banach-1922-operations`, `sweep_methods.md`
§1.C — "the mathematics is OCCUPIED, the packaging is NOT FOUND"); the committor identity
(`kemeny-1976-finitemarkov`, `grinstead-1997-probability` Thm 11.6 `[U]`,
`metzner-2009-tpt-markov-jump`, `e-2006-transition-paths`); the reach-avoid rule
(`summers-2010-reach-avoid`, `abate-2008-reachability`, `kariotoglou-2014-lp-reach-avoid`);
the multi-hop-on-attention lineage that ChaCAL itself cites as closest (`wang-2021-magna`,
`feng-2022-diffuser`, `yuan-2025-paraformer`, `chamberlain-2021-grand`, `sweep_resolvent.md`
§2.4–2.5, §1.3). Mechanism: P-7 (vocabulary with no referent — a cross-reference to owners
who are not there) and BRIEF.md §5 rule 7. **REPAIR:** §0.3 lists every owner §7 promises.

### 1.3 Per-component: owners the sweeps found that the design under-states or omits

| design item | cited | missed or under-stated (canonical keys) | sweep |
|---|---|---|---|
| §3.1 (e) boundary rows | none in-section | **`karbalayghareh-2026-doformer`** — a `do()` inside attention where the intervened token is fixed and forbidden from attending, "an absorbing row in an attention operator" (`sweep_methods.md` §1.G, §3 NEAR-MISS); `wu-2012-partially-absorbing`; `begga-2023-diffusion-jump`; `azad-2022-harmonic-extension`; `baier-2008-modelchecking` (several target sets, one linear system, a safety verdict); `ranmilo-2026-attentionsinks` / `xiao-2023-attentionsinks` (the sink the plant uses) | methods, occupied, resolvent, linrec |
| §3.2 (g) `do(a)` re-solve | `sherman-1950-*`, `hager-1989-updating` | `piray-2021-linearrl` Eq. 5 (named in §0.3, not where the bet is placed); `schweitzer-1968-perturbation`; `mooij-2013-ode2scm` (consequence = displacement of an equilibrium, the definition); `bottou-2013-counterfactual` §7.3; `shimizu-2006-lingam` / `gische-2022-beyondmean` (`(I−B)^{-1}` total effects); `vakalis-2026-interventiongap` (oracle-scored intervention effect with a no-change floor — the design's zero-hop control philosophy, four days old) | causality, safety |
| §3.3 (h) safest move | `vanmoffaert-2013-chebyshev`, `summers-*`, `hsu-2023-safetyfilter` | `bellman-1957-markovian` (argmin over actions on evaluated values); `altman-1999-cmdp`; `borquez-2023-lrf` (the threshold form); `todorov-2006-lmdp` / `todorov-2009-efficient` (a move from a linear first-exit solve); `park-2026-maxmin`; `yang-2026-lexisafe`; **`jeddi-2021-lyapunovsafe`** (a transformer choosing the action with the lowest predicted violation probability — NEAR-MISS, `sweep_safety.md` §1.G) | safety, methods |
| §3.4 (i) certificate | none | `meyer-2000-matrix` (the tail bound); InfSA's global bound is the `K = 0` case (`sweep_linrec.md` §4 NEAR-MISS) | methods, linrec |
| §3.5 (j) learnable `γ` | none | `roffo-2026-infsa` (learnable per-head `γ`, sigmoid), `yuan-2025-paraformer` / `chien-2020-gprgnn` (learnable hop weights) — learnable discount in attention is OCCUPIED; only "causal LM + learnable `γ`" is open (`sweep_resolvent.md` §7) | resolvent |
| §3.7 (l) barcode | `kushnareva-2021-tda-attention`, `turner-2019-quasimetric-rips`, three of six M10 papers | **`kim-2026-topological-causal`** ("an intervention changes the barcode" is a published estimand — the design's `Δz`-through-persistence reading is its instance, `sweep_topology.md` §4.3); `kushnareva-2022-betti` (the title already claims Betti numbers of attention graphs); `perez-2022-topological-bert`; `samaga-2026-halluzig`; `cohen-steiner-2007-stability`; `chowdhury-2016-dowker` / `chowdhury-2017-path-homology` (directed `β₀` is not one notion); the other three M10 papers (`gabrielsson-2019-topology-layer`, `carriere-2019-perslay`, `corcoran-2020-ph-gradient-regularization`) | topology |
| §3.7 / §3.6 segmentation | none | `lin-2025-forgetting-transformer` (the mechanism, `D_ij = c_i − c_j`), `yang-2023-gla`, `hwang-2025-hnet`, `yang-2026-boundary-repair` Thm 1 (the converse), `zhao-2026-structuredsparse` (blockwise resolvent, approximate) — the record owns only the *iff* certificate | topology, expressivity |
| §3.8 (m) Mapper | `carriere-2018-mapper-statistics`, `sharma-2026-kernels-22` | `singh-2007-mapper` (`[U]` landing, two records agree); **`zhao-2026-structuredsparse`** (the existing blockwise kernel path for *this* resolvent, `sweep_resolvent.md` §2.2 — the do-nothing control should be it, not only the 0D-salience schedule); `cho-2022-sbm-attention` (overlapping cluster cover as candidate builder); `roy-2021-routing-transformer`, `kitaev-2020-reformer`, `yuan-2025-nsa`, `guo-2025-loglinear` | topology, linrec |
| §3.9 (n) joint vector label | none | `vakalis-2026-interventiongap`, `lin-2026-scratchworld` (credit only changed fields — the D-5 discipline the design invokes), `xun-2026-evidencetype` (sign fidelity with a floor); the pairing is what is NOT FOUND, the metric is not (`sweep_causality.md` §2.H) | causality |
| §3.11 (p) committor label class | none | **`zhang-2026-committorpairformer`** — an attention model (Pairformer) trained on committor labels (`sweep_safety.md` §1.C); `khoo-2018-committor`, `li-2019-committor-deep`, `contrerasarredondo-2025-committor-gnn`; `cover-2006-elements` for Fano; `wang-2025-easytohard` for why the dial must vary (D-3) | safety, expressivity |
| §2.3 "phase" / horizon dial | `prinz-2011-*`, `fisac-2019-bridging` | `ramsauer-2021-hopfield` and `erel-2025-attentionchains` own "metastable state" vocabulary *for attention*; `bellman-1957-markovian` / `dayan-1993-successor` own the discount; `kemeny-1976-finitemarkov` owns "`γ ↑ 1` reads which basin" | linrec, methods |

### 1.4 Sentences that would read as claiming an occupied object as new

1. §2.3 *Holds*: "One learnable scalar interpolates next-step (softmax, bitwise at `γ = 0`)
   and equilibrium (absorption) reads" — no owner in the sentence. The discount is Bellman's,
   the SR limit is Dayan's, a learnable per-head `γ` on an attention Neumann series is
   InfSA's (`roffo-2026-infsa`), `γ ↑ 1 ⇒ absorption probability` is Kemeny–Snell. Mechanism:
   P-7. **REPAIR:** append "(the discount of `bellman-1957-markovian`; learnable as in
   `roffo-2026-infsa`; the `γ ↑ 1` limit of `kemeny-1976-finitemarkov`)".
2. §3.4 *Holds*: "every truncated read ships `δ·‖V‖_∞`" — fine as discipline; but §3.4's
   Prediction presents the bound as the design's instrument without naming the textbook. **REPAIR:**
   "the Neumann tail of `meyer-2000-matrix`, printed".
3. §3.1 *Holds* / §6.1: "boundary rows are the mechanism" with no NEAR-MISS named — DoFormer
   has one absorbing row inside attention. **REPAIR:** "`K + 1` absorbing sets, against DoFormer's
   single fixed row (`karbalayghareh-2026-doformer`)".
4. §6.1: "one-read joint consistency" as a "separate advantage" — `wang-2024-incontext-td` /
   `xie-2026-softmax-rl` show a softmax stack computes the same resolvent by iteration (§7 of
   the design cites them); the design's own §5 rank 4 reprices "one-read" as a *cost* statement.
   §6.1 should say "cost" in the optimistic sentence too.
5. §1 last paragraph: "the first arm in the record whose *training* step runs under
   `use_deterministic_algorithms(True)`" — the `softmax` control arm has no `cumsum` and its
   strict-mode training was never tested (`sec_measured.md` M.4.3 tests `arm_smprime` only).
   "First" is V-23 in potential form. **REPAIR:** "the first *gated* wing".

---

## 2. Attack 1, continued — theorem hypotheses at `s = 64, d = 16` and on the beds

| theorem as used | hypotheses stated by the design? | at `s = 64, d = 16` / on the bed | mechanism |
|---|---|---|---|
| `sanford-2024-logdepth` Thm 4.2 → "depth-5 skyline" (§1, §6.1) | no | Thm 4.2 is an *upper* bound for `hop_k` pointer chasing on a token-defined graph, `m = O(1), H = 1, p = Θ(log N)`; the reduction `hop_k → committor on the layer's own P` is NOT FOUND (`sweep_expressivity.md` §3.5). So "depth 5 computes the `t* = 8` label" is a lineage assertion; the skyline is empirical and *unmatched* in parameters (the design says so once, §1) | P-10, V-25 |
| `yehudai-2025-depthwidth`, `merrill-2024-cot` (§7) | no | named as fellow approximators; **no row in §1's price table and no bet carries them** — the skyline of `sec_beds.md` P2 (three fellows) is reduced to one, and even that one is not a bet (§3.4 below) | D-1, R-SKY |
| `duranthon-2026-softmax-advantage` Prop 4.2 (§5 rank 4: "says the marginal will not separate") | no | Prop 4.2 is Bayes optimality of softmax on *single-location regression*; BED-S's committor vector is not SLR; the record's own caveat is `L/d = 4.00` (`sec_refuted.md` §3.0). Used past its hypothesis as evidence | P-10 |
| `dash-2005-emc` Thm 1 (§3.10) | no | Thm 1's violation condition is feedback through the manipulated variable; on the causal (triangular) class it is satisfiable on **0 of N draws** — the design's own counter concedes this. "Solve, then re-solve the suffix" is undefined on the cyclic plant | V-25, V-24 |
| `turner-2019-quasimetric-rips` δ (§3.7) | no | Turner's stability is for Rips-type filtrations of an asymmetric *dissimilarity* under a sup-norm perturbation of the function; the design filters `|∂z_i/∂x_j| > ε` (a similarity, superlevel) on an undefined digraph with an undefined connectivity notion (weak / strong / path-homology `β₀`, `chowdhury-2017-path-homology`) and names no perturbation for δ to bound | V-17 |
| `misra-2023-safety-constrained-mdp` (§2.2 → Limits) | partly | multichain CMDP with a *policy*; BED-S is a one-step choice over `m` clamps on an absorbing chain — the warning may not apply at all; carrying it to Limits is correct, calling it a hazard is not yet licensed | P-10 (mild) |
| P5(b) row-stochastic mixing matrix, I4 certificate, C3 (§0.4, §3.4) | yes for β = 1 | holds on the softmax corner (`Σ_j Re W_ij = 1 ± 2.2e-16`, `sec_measured.md` M.1); **fails on the corner-3 base** the design prices at `≈ 133 s` (§1): at `β = 0` rows sum `1.312 … 10.293` (`V16_ARM_SMPRIME.md:28-32` via `sec_measured.md` M.3), so `‖P‖_∞ > 1`, the tail bound has no domain, and C3 does not hold. The price row carries no such note | V-25 |
| `segmentation_blockdiag` ε = 0 endpoint (§3.7) | no | needs a zero gate on the path; the softmax corner has none (`exp > 0`); on BED-S at the softmax corner the census reads **0 of N draws**; the `3 of 3` domain census (`sec_proved.md` §2.8 (6)) is BED-M's, corner 3 | V-25, V-22 |
| `e4_harmonic.cheeger_t_rel_floor` (§3.11, "one-bridge case read 1372.50") | no | Levin–Peres Thm 13.10 needs a *reversible* chain (`sweep_topology.md` §4.5, read at p. 183); the 1372.50 is E4′'s 1,200-node Rips graph, not an 11-node jittered `bed_1` landscape | V-25, V-22 |
| Grinstead–Snell Thm 11.6 `Σ_k q^{(k)} = 1` (§2.2 "C5 degeneracy") | no citation | holds (RUN-1: `[0.999999999999999, 1.0]`) **only once BOS is declared**; undeclared, the solve is singular (RUN-1: `ρ(Q) = 1.000000`) — the hypothesis "absorption a.s. from `T`" is violated on every draw of the design point as written | V-25 |

---

## 3. Attack 2 — vacuity by the taxonomy, bet by bet

### 3.1 Bet A — consequence displacement (§2.1)
- **Rejection region (V-24, V-14).** The `V = 1 ⇒ Δz ≡ 0` plant enters through `V`, not
  through the arm; the arm-side bind (cosine gap > MDE) has its region in the MDE. Acceptable.
  But an arm emitting `Δẑ ≡ 0` makes the cosine `0/0`; nothing refuses it (V-16).
- **PASS half non-constant (V-8).** `Δz` is exactly `0.0` on every row before the intervened
  position (design §4, RUN). If the intervened position is fixed per bed (E2's convention
  `f = s−1−d`, `sec_beds.md` §6.A), those coordinates have label sd `0.0` across draws,
  `nrmse` returns `nan`, and `nan ≥ 1.0` is `False` (`MISTAKES.md:149-153`). The design prints
  no per-coordinate sd for `Δz`. **REPAIR:** draw the intervened position uniformly, print the
  per-coordinate label sd, mask constant coordinates from the cosine.
- **Oracle (D-2).** The arm assembles `P̂` from edge tokens; if `‖P̂ − P*‖_∞ < 1e-3` the arm
  *is* the oracle (`sec_refuted.md` C1 kill). `sec_beds.md` §6.C.1 calls assembling `P` the
  capability; the design carries neither C1's measure nor a pre-registered VOID list (M-7).
  **REPAIR:** print `‖P̂ − P*‖_∞` per seed; pre-register which contrasts are VOID.
- **Threshold (M-2, V-22).** "softmax already reads `0.807843` CP `[0.754044, 0.854329]`
  there and the shape is predicted inside it" — the number is BED-M E2's (`MATHEMATICS.md:396-406`),
  carried to BED-S as the prediction's interval. **REPAIR:** predict against BED-S's own softmax
  control; the E2 number is context only.
- **Producer (P-1).** `1.36e-15`, `1.127`, `0.0` are "RUN this session" with no command text;
  CORRECTIONS_2 §1 reports the same identities at `s = 32` as `1.2e-15` and `0.1096`. Two RUNs,
  no producer either can point at. **REPAIR:** the one-liners go into a markdown appendix verbatim.

### 3.2 Bet B — safest move (§2.2) — the construction the design did not check
- **V-25 / D-4 (RUN-1).** On a causal softmax `P` row 0 is `e_0`; with BOS undeclared the
  transient block has `ρ(Q) = 1.000000` and `I − Q` is singular on **every** draw. The design's
  §2 preamble voids every bet if a guard fails at construction — and this guard is not in its
  list. **REPAIR:** adopt CORRECTIONS_2 F1 verbatim (BOS inside a declared boundary set; say
  which). RUN-2 then shows the goal set containing BOS absorbs `0.89–1.00` of the mass at every
  query — the argmin class balance (guard 2) is at risk from the fix itself (V-8); print it.
- **V-8 (RUN-2).** A constraint placed after the query has `q = 0.0` exactly. The design's
  §6.C.1 token order is unspecified. **REPAIR:** CORRECTIONS_2 F2; print reachability per set.
- **Control PASS half (V-8, V-12).** The plant "ChaCAL + same `γ` + sink token" is well-formed;
  but "same `γ`" is the *trained* `γ̂` of the shape, so the control is defined only after the
  shape trains — the pair is sequential, not paired (M-2 in potential form). State the order.
- **Leak thresholds (V-22, V-17).** `PASS_BAR = 0.5 / FAIL_BAR = 0.9` are `scale/rips_gate.py:60-61`
  constants set for E4′'s Rips fixed point (`0.973819`); carried to BED-S untouched.
  **REPAIR:** re-derive or declare them the record's, with the E4′ provenance printed.
- **Fano floor `k = 1` (P-1).** `I(X_{≤1}; a*)` "computable" by plug-in MI over an 8-class
  `a*` given a continuous window — no producer, no estimator named. Until produced, the only
  floor is `k = 0` (`0.6667`, RUN). The prediction's deciding number has no producer yet.
- **Counter specificity (L-SIGN).** Prediction: `CP_upper < floor`; counter: `CP_lower ≥ floor`;
  the band `CP_lower < floor ≤ CP_upper` is unfiled. **REPAIR:** name it SPLIT.

### 3.3 Bet C — the horizon dial (§2.3)
- **L-SIGN.** Counter: PINNED on `≥ 6/8`; prediction: "MOVED" with no seed count. The
  `3/8–5/8` pinned band belongs to neither. **REPAIR:** prediction "MOVED on `≥ 6/8`", band SPLIT.
- **Ruling 10′ transfer (V-17).** Ruling 10′ is stated for `β` against `β ≡ 1`
  (`V17K_RULINGS.md:389-436` via `sec_state.md` S.4); the design applies it to `γ` against
  `γ = 0` without saying so. Declare the null and the held-out set.
- **Occupancy.** §1.4 item 1 above. Otherwise the ablation (V-9) and the `1/(1−γ̂)` print
  (P-8) are the right instruments. KEEP after the two repairs.

### 3.4 The missing bet: the skyline (D-7, D-1, R-SKY)
`sec_beds.md` §6.E files three predictions; the design carries P1 (Bet B) and P3 (§3.9) and
**drops P2** — "the depth-5 softmax skyline reaches the same argmin accuracy within the MDE".
P2 then reappears as a *fact* inside §6.1's optimistic sentence ("the depth-5 softmax skyline
reaches the same argmin accuracy within the MDE"). A prediction asserted in the headline
sentence and filed nowhere with a counter is D-7 by the design's own §0.1. The §1 table prices
it (`≈ 7.6 s`, ASSUMED) and no killer runs it. **REPAIR:** add Bet D = P2 with its counter and
the `≈ 7.6 s` price, and add the wide-constant-depth (`yehudai-2025-depthwidth`) and CoT
(`merrill-2024-cot`) rows as `NOT MEASURED` in §1.

### 3.5 §3.1 (e) — the "predicted and printed" zero (V-3, V-10)
RUN-3: the mixing-matrix min entry is `0.0` because absorbing rows *are* zeros (151 in the
lower triangle) and the mask zeroes the upper triangle; without absorbing rows the lower
triangle has `0` exact zeros and min `2.6e-04`. So "min entry reads exactly 0" is an identity
of the construction, has no rejection region, and cannot be a prediction. **REPAIR:** file it as
the instance of P5(b) with the dead `sgate` (`107/200` signed, `ceq/attention.py` header via
BRIEF.md §1) as the planted positive that the test *can* read non-zero.

### 3.6 §3.2 (g) — the price band (M-2, P-8, DERIVED-1)
The design predicts the `m = 8` sweep adds `≤ 2 × 1.041 ms` and refutes at `≥ 8 × 1.041 ms`.
No derivation is shown. DERIVED-1: `m` resolvent columns cost `m·s²/2` MACs against the
solve's `s²d/2`, ratio `m/d = 0.500` at `(8, 16)`; so the *predicted* increment is `≈ 0.5×`
one solve, the counter is `m× = 8×` (a re-solve per candidate), and `(0.5, 8)` is a 16-fold
band the design leaves unfiled. Also: Sherman–Morrison needs `1 − γ uᵀ M e_i ≠ 0` and `uᵀ1 = 0`
(`sweep_safety.md` §1.A); a clamp `P_a[v,:] ← e_u` satisfies the second; the first is a
domain census the design does not print (V-25). **REPAIR:** prediction `0.5×`, counter `≥ 8×`,
SPLIT between, denominator census printed.

### 3.7 §3.4 (i) — a certificate bind with nothing to certify on the bed it is placed on
- The truncation half: on row-stochastic `P` the residual *equals* the bound to `1e-15`
  (`sec_proved.md` §2.8 (4), `sec_refuted.md` C6) — "exceeded on any draw" cannot fire (V-10;
  the design declares this definitional, then still lists the exceedance as the killer).
- The mask half: at `s = 64` the design ships the exact solve and refuses masks (§3.4 counter,
  §1: solve `2.514 ms` < one hop `3.001 ms`). No F1 mask exists on BED-S. The killer "1,024
  forward passes fire on one exceedance" has no object (V-6: the branch never runs).
- The vector-unit pass `6.13e-05 ≤ 7.6e-05` rests on `‖V‖_∞ ≈ 5` **ASSUMED, not printed**
  (`sec_cost.md` §4.x.3). The only RUN in the bind is not RUN (P-1).
- Only the planted non-stochastic `P` (`119.37` vs `1.143`) has a rejection region, and it fires
  on the hypothesis, not on a shipped object.
**KILL** as a BED-S bind. What changes the verdict: an F1 mask on the `s = 4096` CSR path
(§3.8), where `‖O_full − O_mask‖_∞` is a reading and not an identity, with `‖V‖_∞` printed.

### 3.8 §3.5 (j) — one measurement, two calibration rows (V-1, M-20)
Until an LM cell exists the bet is "placed on BED-S's `γ̂` (§2.3)". Bet C and (j) then score
the same `γ̂` twice in the calibration column. **REPAIR:** (j) has no row until its LM cell
runs; the BED-S reading belongs to Bet C only. The counter's evidence — ChaCAL's perplexity
`21.46 vs 20.15` — is a summariser reading `[U]` (`sweep_expressivity.md` §6) and cannot carry
a counter under L-EQ; replace with "ChaCAL fixed `γ = 0.9` and reports no LM gain (abstract
`[V]`)".

### 3.9 §3.6 (k) — `[M]` on `[U]` names (P-11)
`resolvent_fromBlocks` rests on Mathlib `inv_fromBlocks_zero₂₁_of_isUnit_iff` `[U]` and
`lower_triangular_isUnit` on `Matrix.det_of_lowerTriangular` `[U]` (`sec_proved.md` §2.8
(5)–(6)). An `[M]` grade citing unverified lemma names is the record's `#18/#19/#22` pattern
one step earlier. **REPAIR:** `[S]` until `lake build` is green; the design's own P-11 clause
requires it. Also V-25: `segmentation_blockdiag` has **0** BED-S draws in its domain (§2 table).

### 3.10 §3.7 (l) — the barcode
- The `ε = 0` endpoint "equals the exact F0 segmentation count": on the softmax corner there
  are no exact zeros in `∂z/∂x` except the mask's, so the endpoint is `β₀ = 1` (weak
  connectivity) on every draw, and the segmentation count is `1` — equal by construction
  (V-3, V-8). Absorbing rows add exact zeros (RUN-3) but a token that attends *to* an absorbing
  position keeps the digraph weakly connected; under strong connectivity a DAG has `s`
  components on every draw. Neither is a reading.
- Graph (positions? which `x` channels?), connectivity notion, filtering function (superlevel
  of a similarity is not Turner's quasi-metric) and the perturbation δ bounds are all
  unstated (V-17). The permutation null (M-15) and the frozen grid (M-2) are right.
**KILL** as written. What changes it: define the digraph and `β₀` (path homology or weak),
transform `|J|` to a dissimilarity, name the perturbation δ bounds (across seeds? draws?),
and run the plant on corner 3 where zero gates exist (BED-M, `3 of 3`).

### 3.11 §3.8 (m) — the fallback that is not a bet (D-7, V-25, V-22)
"Until then the bet is on the segmentation dividend … BED-M's `31.06×`" — a DERIVED number
with no prediction, counter or killer; and on the softmax corner (BED-S's base) the exact
dividend is `1×` (no zero gates), so §5 rank 7's "reroute to segmentation's exact dividend"
reroutes to a constant. The `(2, 4)` band between counter and prediction is unfiled.
**REPAIR:** `NOT MEASURED — needs the chunked kernel`, no row; the do-nothing control is
`zhao-2026-structuredsparse`'s blockwise resolvent *and* the 0D-salience schedule.

### 3.12 §3.9 (n) — units for the residual ratio (V-17)
`r(ẑ) = ‖(I − γP_env)ẑ − V‖_∞/‖V‖_∞` with error `e = ẑ − z*` satisfies (DERIVED)
`σ_min(I − γP_env)·‖e‖ ≤ ‖(I − γP_env) e‖ ≤ ‖I − γP_env‖_∞·‖e‖ ≤ (1 + γ)‖e‖`; on the
committor chain (`γ = 1`) `ρ(Q) = 0.9409` (`sec_proved.md` §2.8 (3)) puts the lower factor
near `0.06`. A `10×` ratio between arms at equal marginal error is therefore possible only if
one arm's error lies in the slow mode — which is a *statement about which mode the softmax
control mislearns*, not "joint consistency". **REPAIR (KEEP otherwise):** print the two
factors beside `r` so `≥ 10 / ≤ 2` is in units; cite `vakalis-2026-interventiongap` for the
no-change floor the residual is measured against.

### 3.13 §3.10 (o) — EMC (V-25, V-24)
The design's own counter is the point estimate under D-CALIB-1: the violation hypothesis of
`dash-2005-emc` Thm 1 is unsatisfiable on the causal class, the planted cyclic `P` is not a
member of the class the shape uses, and "solve then re-solve the suffix" is undefined once
`P` is cyclic. The `> 0.1 / < 1e-6` thresholds have no provenance. **KILL** as a proposition;
keep the remark the counter already writes, with `dash-2005-emc`, `mooij-2013-ode2scm` and
`voortman-2010-manipulation` cited. What changes it: a bed whose environment chain has
feedback *and* a re-targeted absorbing row, on which the two orders are both defined.

### 3.14 §3.11 (p) — admission (V-22, V-25)
The sd prediction `≤ 0.034451` is BED-M's row-2 sd carried as a BED-S bar; the Cheeger
floor `1372.50` is E4′'s and needs reversibility (§2 table). Add F1/F2 to the census (RUN-1,
RUN-2). Otherwise the sd read at `≈ 17.4 s` is the right first killer. **REPAIR** (citations
per §1.3; census lines added).

### 3.15 §4 table, §5 ranking, §6 sentences
- §4: the EMC row's rejection region is "NOT RUN"; the C6-causal row's is "—". A bind table
  with two empty rejection-region cells is V-24 in table form. Strike EMC; give C6-causal its
  plant (a non-causal `P` for which `Δz` is non-zero before the intervened row).
- §5 rank 1: the dominant death is right, but its cause list omits F1/F2, which kill at
  construction on **100 % of draws** as written (RUN-1, RUN-2) — cheaper than `17.4 s`: 0 GPU-s.
- §5 rank 4 cites Duranthon Prop 4.2 past SLR (§2 table). §5 rank 7 reroutes to `1×` (§3.11).
- §6.2 (the median sentence, every counter the point estimate) says "Six of its identities
  are machine-checked and the seventh … numerically"; §3.6's *counter* says "five are
  theorems; I3 is a numeric identity". The median sentence contains the optimistic count.
  Mechanism: P-11 and D-CALIB-1 violated in the one sentence the author is told to read first.
  **REPAIR:** "five".
- §6.1 asserts P2 (skyline within MDE) without a bet (§3.4 above) and "one-read joint
  consistency" as an advantage the design's own §5 reprices as cost (§1.4 item 4).

---

## 4. Verdicts

| # | target | verdict | reason (number / key / file:line) |
|---|---|---|---|
| 1 | §0.3 owner list + §7 cross-reference | REPAIR | §7 promises owners for SR, Neumann, committor, reach-avoid that §0.3 lacks: add `dayan-1993-successor`, `bellman-1957-markovian`, `meyer-2000-matrix`, `kemeny-1976-finitemarkov`, `metzner-2009-tpt-markov-jump`, `summers-2010-reach-avoid`, `wang-2021-magna`, `feng-2022-diffuser`, `yuan-2025-paraformer` (P-7) |
| 2 | bib keys | REPAIR | four aliases → canonical (`bib_aliases.md` rows 18, 24, 49, 14); point the header at `references.bib` (P-5) |
| 3 | §1 corner-3 base row (`≈ 133 s`) | REPAIR | rows sum `1.312 … 10.293` at `β = 0` (`V16_ARM_SMPRIME.md:28-32`): P4/P5(b)/C3 have no domain there; note it or drop the row (V-25) |
| 4 | §1 skyline rows; the missing Bet D | REPAIR | `sec_beds.md` §6.E P2 is asserted in §6.1 and filed nowhere; add it with counter and `≈ 7.6 s`; add `yehudai-2025-depthwidth` and `merrill-2024-cot` rows as NOT MEASURED (D-7, R-SKY) |
| 5 | §1 "first arm … deterministic" | REPAIR | `sec_measured.md` M.4.3 tested `arm_smprime` only; write "first gated wing" (V-23) |
| 6 | Bet A (§2.1) | REPAIR | per-coordinate sd of `Δz` printed and constant rows masked (`MISTAKES.md:149-153`, V-8); `0.807843` is E2's (V-22); `‖P̂ − P*‖_∞` per seed and a VOID list (D-2, M-7); one-liners recorded (P-1) |
| 7 | Bet B (§2.2) | REPAIR | RUN-1 `ρ(Q) = 1.000000` with BOS undeclared; RUN-2 `q = 0.0` for any set after the query — adopt CORRECTIONS_2 F1/F2 (V-25, V-8, D-4); `PASS_BAR/FAIL_BAR` from `scale/rips_gate.py:60-61` are E4′'s (V-22); Fano `k = 1` has no producer (P-1); SPLIT band named (L-SIGN) |
| 8 | Bet C (§2.3) | REPAIR | prediction needs a seed count to match "`≥ 6/8` PINNED" (L-SIGN); declare the Ruling 10′ null for `γ` (V-17); owner clause for the dial (`bellman-1957-markovian`, `roffo-2026-infsa`, `kemeny-1976-finitemarkov`) |
| 9 | §3.1 (e) | REPAIR | RUN-3: min entry `0.0` by the identity rows (151 exact zeros) — a theorem instance, not a prediction (V-3); cite `karbalayghareh-2026-doformer` as NEAR-MISS, `wu-2012-partially-absorbing`, `baier-2008-modelchecking` |
| 10 | §3.2 (g) | REPAIR | DERIVED-1 ratio `m/d = 0.500`, not `≤ 2×`; counter `8×`; band unfiled (M-2, P-8); denominator census `1 − γuᵀMe_i ≠ 0` (V-25); cite `piray-2021-linearrl` Eq. 5, `schweitzer-1968-perturbation`, `mooij-2013-ode2scm` |
| 11 | §3.3 (h) | REPAIR | citations only: `bellman-1957-markovian`, `altman-1999-cmdp`, `borquez-2023-lrf`, `todorov-2009-efficient`, `park-2026-maxmin`, `yang-2026-lexisafe`, `jeddi-2021-lyapunovsafe` (NEAR-MISS); mechanics KEEP |
| 12 | §3.4 (i) on BED-S | KILL | truncation half attained to `1e-15` (V-10); no mask ships at `s = 64` (`2.514 < 3.001 ms`, V-6); `‖V‖_∞ ≈ 5` ASSUMED (P-1). Changes with an F1 mask on the `s = 4096` CSR path and `‖V‖_∞` printed |
| 13 | §3.5 (j) | REPAIR | same `γ̂` as Bet C → one calibration row (V-1, M-20); `21.46 vs 20.15` is `[U]` and may not carry a counter (L-EQ); cite `roffo-2026-infsa`, `yuan-2025-paraformer` for learnable `γ` |
| 14 | §3.6 (k) | REPAIR | `[M]` grades rest on `[U]` Mathlib names (`sec_proved.md` §2.8 (5)–(6)) → `[S]` until green (P-11); `segmentation_blockdiag` census on BED-S = 0 draws (V-25) |
| 15 | §3.7 (l) | KILL | `ε = 0` endpoint = `1` = segmentation count on the softmax corner by construction (V-3, V-8); digraph, `β₀` notion, filtering function, δ's perturbation undefined (V-17); `turner-2019-quasimetric-rips` hypotheses unmet. Changes with the four definitions and a corner-3 bed |
| 16 | §3.8 (m) | REPAIR | `31.06×` fallback is not a bet (D-7) and reads `1×` on the softmax corner (V-25); `(2, 4)` band unfiled; control must include `zhao-2026-structuredsparse`; cite `singh-2007-mapper`, `cho-2022-sbm-attention` |
| 17 | §3.9 (n) | KEEP | after printing `σ_min(I−γP_env)` and `‖I−γP_env‖_∞` beside `r` (V-17) and citing `vakalis-2026-interventiongap`, `lin-2026-scratchworld` |
| 18 | §3.10 (o) | KILL | Dash Thm 1 hypothesis on 0 draws of the causal class (V-25); cyclic plant outside the class, "re-solve the suffix" undefined; `0.1 / 1e-6` without provenance. Keep the counter's remark |
| 19 | §3.11 (p) | REPAIR | `0.034451` and `1372.50` are BED-M's and E4′'s (V-22); Cheeger needs reversibility (V-25); add F1/F2 to the census; cite `zhang-2026-committorpairformer`, `khoo-2018-committor` |
| 20 | §4 identities table | REPAIR | two empty rejection-region cells (EMC "NOT RUN", C6-causal "—") (V-24); record the six one-liners (P-1) |
| 21 | §5 ranking | REPAIR | rank 1 cause list adds F1/F2 at 0 GPU-s; rank 4 drops Duranthon Prop 4.2 as evidence past SLR (P-10); rank 7 reroute reads `1×` |
| 22 | §6.1 | REPAIR | P2 asserted without a bet; "one-read joint consistency" is a cost statement per §5 rank 4; DoFormer NEAR-MISS named beside "boundary rows" |
| 23 | §6.2 | REPAIR | "six machine-checked" contradicts §3.6's counter "five" — the median sentence carries the optimistic count (P-11, D-CALIB-1) |
| 24 | §0.4 / obstruction-4 correction | KEEP | P5(b) stated with its hypothesis; RUN-3 confirms row sums `1.000000000000000` with absorbing rows |
| 25 | §1 price arithmetic | KEEP | `34 s`, `228 s`, `5.1 s`, `7.6 s`, `36 min`, `4.7 min`, Fano `0.5/0.6667/0.75` all re-derived here and agree |

Tally: KEEP 3, REPAIR 19, KILL 3.

---

## 5. Fatal (the coordinator's list)

1. **The BED-S design point has no admissible draw as written.** With BOS undeclared the
   oracle solve is singular on every draw (RUN-1, `ρ(Q) = 1.000000`); with any constraint set
   placed after the query its committor is `0.0` on every draw (RUN-2). Under the design's own
   §2 rule ("void if any guard fails at construction", D-4) Bets A, B, C and §3.1–3.3, 3.9,
   3.11 are void before a GPU-second is spent. Repair exists and is one paragraph
   (CORRECTIONS_2 §1 F1/F2); the design must carry it or every §2 number is a number about
   a bed that cannot be built.
2. **Two of the "eleven bets" have no rejection region on the bed they are placed on** —
   §3.4 (i) and §3.10 (o) — and the count "eleven NOT-FOUND components, one bet each" is
   therefore nine bets and two remarks. The calibration column must not carry rows for them.
3. **The skyline is not a bet.** §6.1's "the depth-5 softmax skyline reaches the same argmin
   accuracy within the MDE" is `sec_beds.md` P2 asserted as fact with no counter filed
   anywhere in the design (D-7, the mechanism §0.1 says disqualifies a bet from being filed).

---

## 6. Limits (collected once)

RUN-1/2/3 are numpy float64 on one seed at `s = 64` with Gaussian logits, not a BED-S draw
from a generator that does not exist; they instantiate the F1/F2 hypotheses, they do not
measure BED-S. DERIVED-1 counts MACs and ignores the `m` inner products and the diagonal
reciprocals (`< 2 %` at `s = 64`). The citation attack is bounded by the eight sweeps' own
coverage (recorded query lists; single US index; `[V]` by abs/registry page, equations `[U]`
except where the sweeps say otherwise); no source was fetched this session and no key outside
`references.bib` is proposed. Line numbers are at `207e7b9`; MISTAKES.md mechanisms are cited
by symbol and heading line. No code file was written; no git write was made; the three
`python -c` calls above read nothing from the tree and wrote nothing.
