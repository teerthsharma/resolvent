# DESIGN — THE INSTRUMENT FIRST: the apparatus that decides the shape

*SATURN holding the WATSON role, 2026-09-03, against HEAD `207e7b9` on `v17k-gate0`.
Written without reading the other two designers' files. Evidence classes: `RUN` (executed
this session on this box, torch `2.5.1+cu121`, float64, one draw, seed 0 unless stated),
`READ path:line` (quoted at HEAD), `CITED [V] bibkey` (a key in `$SCRATCH/sweep/bib_*.bib`
whose page the owning sweep fetched), `DERIVED` (steps written out), `DESIGN` (a choice this
file makes, with the `MISTAKES.md` mechanism it is made against). No number without one of
these. Nothing here is a result; every capability sentence is a bind, a control, a floor, a
certificate, a census or a kill, and the kill's threshold is frozen in this file before any
cell exists (M-2, `READ MISTAKES.md:451`).*

## 0. What this file decides, and the enumeration rule

`THESIS_CORRECTIONS.md` C1 lists the NOT-FOUND composition under eight letters, (e) and (g)
to (m), three of which name two objects each — (e) constraint sets *and* a goal set, (g) the
re-solve *and* the displacement as a trained vector output, (h) the per-constraint reads *and*
the safest-move rule. Splitting those three gives the **eleven rows** this apparatus is built
for:

| row | component (NOT FOUND per the sweeps) | owning sweep line |
|---|---|---|
| E1 | `K ≥ 2` absorbing constraint sets as identity rows inside the content-dependent causal read | `sweep_resolvent.md` §7; `sweep_occupied.md` §3 (m) |
| E2 | the goal set `𝒜_0` as a further absorbing set, so the label is a reach-avoid vector | `THESIS_CORRECTIONS.md` C5; `sweep_safety.md` §4.1 |
| G1 | the interventional re-solve `do(a)` on context rows | `sweep_causality.md` §3.3 item 1; `sweep_linrec.md` N3 |
| G2 | the displacement `Δz` as a trained, jointly scored vector output | `sweep_causality.md` §3.3 item 2 |
| H1 | per-constraint committor / reach-avoid reads as head outputs | `sweep_safety.md` §2 composite row |
| H2 | the safest-move rule over candidate moves placed in the context | `sweep_occupied.md` §3 (o) |
| I | a printed Neumann certificate with a planted negative (L-CERT) | `sweep_methods.md` §3 NOT FOUND item 2 |
| J | a learnable discount in a causal LM read, pinned by the LR test | `sweep_resolvent.md` §7 bullet 6 |
| K | machine-checked containment (three corners + `γ = 0`) and the zero-gate segmentation iff | `sweep_occupied.md` §3 (r); `sweep_topology.md` §3(a) |
| L | the influence-Jacobian `β₀` barcode with a directed-stability `δ` | `sweep_topology.md` §3(b) |
| M | Mapper cover → causal CSR schedule → certified resolvent | `sweep_topology.md` §3(d) |

Each row gets, in §2, the same nine cells: **bind** (with planted negatives and the O(1)
failure each must produce), **control arm**, **bed and label**, **metric**, **floor**,
**certificate**, **domain census**, **pre-registered kill with its frozen threshold**, and
the **mechanism** each cell is designed against. §3 fixes the identity manifest so that
`V20_R15_LEAP_LEDGER.md` L-2 (`0 of 24` attributable, `READ :23`) cannot recur. §4 answers
`sec_refuted.md` Part B's top ten and §5 answers Part C's twelve attacks by construction.
§6 is the price and the order. §7 is Limits, collected once.

## 1. The RUN facts this design rests on (this session, float64, `s = 32`, `d = 8`, `γ = 0.6`)

Every identity below was computed with `torch.linalg.solve` against a causal softmax `P`
built from `softmax(L.masked_fill(triu, −inf))`, two absorbing rows `{5, 17}` set to identity,
and one planted row edit at position `i = 9` moving half of `P₉₃` to column 7 (`u·1 = 0`, the
row stays stochastic). The command is the one-liner in this session's transcript; it wrote
no file.

| # | identity | reading | what it licenses in §2 |
|---|---|---|---|
| R1 | Sherman–Morrison planted identity: `V ≡ 1 ⇒ Δz ≡ 0` | `max|Δz| = 0.0` (bitwise) | G1's must-pass half (`sweep_safety.md` §4.3 asked for it, unrun) |
| R2 | the same edit with `V ~ N(0,1)` | `max|Δz| = 0.10960078193360684` | G1's rejection region is non-empty (V-24) |
| R3 | SM closed form `γ (Me_i)(uᵀz)/(1 − γ uᵀMe_i)` vs re-solve | `1.2339847026143769e-15` | G1's `O(sd)`-per-candidate price is exact algebra (C6, `CITED [V] sherman-1950-inverse`, `hager-1989-updating`) |
| R4 | displacement before the intervened row, `Δz[:i]` | `4.440892098500626e-16`, `torch.equal(z'[:i], z[:i]) = True` | consequences propagate forward only on a causal `P`; the EMC half that holds (C6, `CITED [V] dash-2005-emc`) |
| R5 | C6's exact identity `Δz = (I − γP')^{-1}(ΔV + γ ΔP z)` with both `ΔP` and `ΔV` planted | residual `1.2281842209915794e-15` | G1/G2's oracle is one more triangular solve, not a linearisation (`sweep_causality.md` §2.B DERIVED, now RUN) |
| **R6** | **row 0 of a causal softmax `P` is `[1, 0, 0, …]`; with constraint sets `{5},{17}` and goal `{30}` but position 0 undeclared, `ρ(Q) = 1.0` and `I − Q` is singular** | `P₀₀ == 1: True`; `ρ(Q) = 1.0` | **E1/E2's domain census must declare BOS inside an absorbing set, or no committor exists** (V-25, V-12) |
| R7 | with `𝒜_0 = {0, 30}`, `𝒜_1 = {5}`, `𝒜_2 = {17}`, `|T| = 28` | `ρ(Q) = 0.6874342901390162`; `Σ_k q^{(k)}` on `T` in `[0.9999999999999993, 1.0]` | C5's conservation row, RUN (`CITED [V] grinstead-1997-probability` Thm 11.6) |
| R8 | `max_k q^{(k)} ≥ 1/K` at `K = 3` everywhere; `min_T max_k = 0.6051682711182041` | `True` | the V-12 degeneracy is real and the goal set is what lifts it |
| R9 | `sd_T` of `(q^{(0)}, q^{(1)}, q^{(2)})` | `[0.0832, 0.0731, 0.0292]` | the label is non-constant on one draw; the census must print it on every draw (V-8) |
| R10 | positions `j < 5` (before every constraint) read `q^{(1)} = q^{(2)} = 0.0` exactly | `0.0` | on a causal `P` the walk moves to `j ≤ i`; a constraint **after** the query is unreachable — BED-S must place constraints before the query position (D-3, V-25) |
| R11 | discounted read at `γ = 0.6`: `Σ_k q_γ^{(k)}` on `T` | `[0.13643076236928928, 0.339994435489312]` | at `γ < 1` the reads are `E[γ^τ 1_k]` and do not sum to one — the "delay counts" confound (`CITED [V] fisac-2019-bridging`) is measurable and must be printed |
| R12 | mixing matrix `P_𝒜 (I − γP_𝒜)^{-1}` with absorbing rows | min entry `0.0`; every row sum `2.5 = 1/(1−γ)` | C3 confirmed by construction: obstruction 4 reads `0`, absorption redirects; the read must carry the `(1 − γ)` factor or declare row sums `1/(1−γ)` (`CITED [V] fagnou-2024-chacal` Eq. 5) |

R6 is the one fact in this table the record did not have. Every causal softmax row attends
to itself and to earlier positions; position 0 attends only to itself, so the walk on `P` is
absorbed at BOS with probability one unless BOS is a declared boundary. The record's value-zero
BOS sink (`V15Fork.Asink_row_sum`, `READ lean/CEQ/V15Fork.lean:81`) is a *column* device; the
shape needs BOS as a *row* device too, and the two must not be conflated (`sweep_resolvent.md`
§2.14, P-7).

## 2. The eleven rows, nine cells each

Conventions for every row. *Bind* = an identity the built arm must satisfy on the bed's real
draw, with a planted mutilation battery whose every member fails at O(1), counts printed —
the `V15_JUPITER2_FORK.md` pattern (`0.9749 / 0.9165 / 1.000 / 0.4845`, `READ
workdonenewseal.md:114-122`). *Control arm* = the fellow approximators the arm is paired
against on byte-identical draws at `4,769` matched parameters (`READ
CEQ_V20_R15_CONTRACT.md:119`), plus the three skylines at *unmatched* depth/width/decode
length (C2). *Floor* = an information floor, never `floor₁` (C15, `READ
V20_R15_JOURNAL.md:51`). *Kill* = a threshold frozen here. *Against* = the `MISTAKES.md`
mechanism, by symbol.

### E1 — `K ≥ 2` absorbing constraint sets as identity rows in the content-dependent causal read

- **Bind B-E1 (the separating bind, C2).** With constraint rows planted, the shape's read
  `O = (1−γ) P_𝒜 (I − γP_𝒜)^{-1} V` must differ from ChaCAL's `(1−γ) P (I − γP)^{-1} V` at
  O(1) on the rows downstream of the constraint positions, and must equal it **bitwise** when
  the constraint set is empty. Planted negatives: (i) `𝒜 = ∅` ⇒ `torch.equal` with ChaCAL
  (`CITED [V] fagnou-2024-chacal`) must be `True` — the identity half; (ii) `𝒜 ≠ ∅` ⇒
  `max|O_shape − O_ChaCAL|` must exceed a printed O(1) gap on the downstream rows (the
  rejection half); (iii) an absorbing row that is *not* identity (row sum `1` but
  off-diagonal mass) must break `Σ_k q^{(k)} = 1` (R7's row) by O(1); (iv) the InfSA base
  `Â = ReLU(QKᵀ)/‖·‖_F` (`CITED [V] roffo-2026-infsa`) substituted for `P` must break (i) —
  `NOT MEASURED — needs the Frobenius base wired into shape_identities.py`
  (`sweep_linrec.md` P2). *Against* V-24 (the parity bind alone cannot separate the shape
  from ChaCAL, `sweep_resolvent.md` §9.2), V-3 (half of the bind is an identity of the
  construction and is labelled so).
- **Control arm.** ChaCAL at the *same* `γ` with no boundary rows (mandatory on every bed);
  ChaCAL with a sink *token* (the C2 kill's planted control — a column sink, not a row);
  an InfSA-style Neumann read without boundaries (`sweep_safety.md` §4.4); depth-1 softmax
  at matched parameters; the three skylines — the `⌊log₂ t*⌋ + 2` stack (`CITED [V]
  sanford-2024-logdepth` Thm 4.2), the wide constant-depth stack (`CITED [V]
  yehudai-2025-depthwidth`), the CoT decoder (`CITED [V] merrill-2024-cot`). *Against* D-1,
  R-SKY (`READ CEQ_V16_CONTRACT.md:209`).
- **Bed, label.** BED-S with the goal set (E2); label the `[m, K+1]` reach-avoid tensor per
  draw. BED-1 as the fixed-landscape oracle instance (`READ ceq/beds/bed_1.py:188-198`).
  BED-M: containment only — the boundary mechanism's planted negative is the executed-program
  corpus's negation gate (`READ ceq/corpus.py:88-130`, `sec_beds.md` §6.A row 3), never a
  capability number.
- **Metric.** Fisher–Rao coordinate `φ(p) = 2 arcsin √p` per committor entry, position-matched
  NRMSE vector (mean, max) on `φ`, plus the harmonic residual
  `r(q̂) = ‖(I − Q_env) q̂ − R_env 1‖_∞` computed with the environment chain (C11). *Against*
  V-26 (marginal `W1` refuted, `READ V20_R15_THEORY_TABLE.md:221`), M-2 (`φ` declared here).
- **Floor.** Exact oracle at `0.0`; budget ceilings from the hop ladder
  `NRMSE(z_k, z*)` printed per draw batch at construction (`READ scale/e4_harmonic.py:194-199`).
- **Certificate.** None on the rows themselves (they are exact identity rows); the read's
  Neumann `δ` is row I's.
- **Domain census (L-DOM).** Printed per draw batch: fraction of drawn `P` with every row
  stochastic *including* absorbing rows (`|rowsum − 1| ≤ 1e-12`); **fraction with position 0
  inside `⊔_k 𝒜_k`** (must be `100 %`, R6); `ρ(Q) < 1` on `100 %` (R7's `0.687` is one
  instance); the fraction of constraint positions that precede the query position (must be
  `100 %`, R10); `K` support `{2, 3, 4}`. `0 %` on any line blocks the row. *Against* V-25,
  V-12.
- **Kill (frozen).** If ChaCAL-with-sink-token matches the shape's committor read within the
  `n = 8` MDE cell of the *realised* paired sd (the `sec_beds.md` §6.D.2 table; e.g.
  `0.039827` at sd `0.034451`, `RUN` by VENUS) on `≥ 6/8` seeds **and** its harmonic residual
  is within `2×` of the shape's, then E1 is not a capability and the paper's delta collapses
  to "ChaCAL + certificate". TOST at `N = 70` (`CITED [V] schuirmann-1987-tost`) is the
  second tier and is priced in §6; it is never claimed at `N = 8` (M-13, `READ
  MISTAKES.md:1210`). *Against* M-7 (the kill is registered before the bed runs), D-7.

### E2 — the goal set `𝒜_0`

- **Bind B-E2.** `Σ_{k=0}^{K} q^{(k)} = 1` on every transient position (R7, `CITED [V]
  grinstead-1997-probability` Thm 11.6), printed with its residual on every draw batch
  including when it fails (V-23). Planted negatives: drop `𝒜_0` ⇒ `max_k q^{(k)} ≥ 1/K`
  everywhere (R8, `min_T max_k = 0.605` on the instance) and the "avoid all constraints"
  objective becomes unattainable — this is the V-12 collapse made visible; drop BOS from
  `𝒜_0` ⇒ `ρ(Q) = 1.0`, the solve must **raise**, never return a number (R6; V-16).
- **Control arm.** The two reductions of the same tensor registered as *columns*, not beds:
  `argmax_a q^{(0)}(do a)` (reach the goal first), `argmin_a max_{k≥1} q^{(k)}(do a)`
  (Chebyshev, `CITED [V] vanmoffaert-2013-chebyshev`), and the lexicographic form (`CITED [V]
  yang-2026-lexisafe`); the disagreement fraction between them is printed (C4 attack). *Against*
  V-1 (one corpus, one registry key).
- **Bed, label.** BED-S; `𝒜_0 ⊇ {0}` by construction (the BOS-as-goal convention, R6); goal
  size `|𝒜_0| ∈ {1, 2} + BOS`.
- **Metric.** As E1; plus reach-avoid accuracy of the argmax with Clopper–Pearson (`CITED [V]
  clopper-1934-binomial`).
- **Floor.** Zero-hop Fano `1 − ln 2 / ln m`: `0.5 / 0.6667 / 0.75` at `m = 4 / 8 / 16` (`RUN`
  by VENUS, `sec_beds.md` §6.C.5; `CITED [V] cover-2006-elements` §2.10); `m = 2` refused.
- **Certificate.** The discounted-read confound: at `γ < 1` the reads sum to `< 1` (R11:
  `[0.136, 0.340]` at `γ = 0.6`); the paper states which `γ` the safest-move objective is
  evaluated at and prints `1 − Σ_k q_γ^{(k)}` as the "delay share" beside every reading
  (`CITED [V] fisac-2019-bridging`). *Against* V-17 (a threshold out of its units).
- **Domain census.** Fraction of draws on which `𝒜_0` is reachable from the query with
  `q^{(0)} > 0.05`; label sd per coordinate (R9 gives `[0.083, 0.073, 0.029]` on one draw —
  the census is per batch); argmin class balance over `m` (none above `1 − 1/m + 0.05`);
  discard count for ties within `1e-9`. *Against* V-8, D-3 (`READ MISTAKES.md:727`).
- **Kill (frozen).** `sd(q^{(k)}) = 0` on any `k`; argmin-unique fraction outside
  `(0.05, 0.95)`; reduction-disagreement fraction `= 0` (then only one reduction may be named).
  Limits carries `CITED [V] misra-2023-safety-constrained-mdp` (Bellman optimality can fail on
  multichain CMDPs) beside the decision rule.

### G1 — the interventional re-solve `do(a)`

- **Bind B-G1.** (a) `V ≡ 1 ⇒ Δz ≡ 0` bitwise (R1, `0.0`); (b) `V ~ N ⇒ max|Δz| = O(1)`
  (R2, `0.1096`); (c) Sherman–Morrison closed form vs re-solve `≤ 1e-12` (R3, `1.2e-15`;
  `CITED [V] sherman-1950-inverse`); (d) `Δz[:i] = 0` for a causal `P` (R4, `4.4e-16`); (e)
  the C6 identity residual `≤ 1e-12` (R5, `1.2e-15`). Planted negatives: (a) is the identity
  half and (b) its rejection region; for (d) a *feedback* plant — re-target an absorbing row
  so that a position `< i` attends to `i` — must make `Δz[:i] ≠ 0` (Dash Thm 1's EMC
  violation, `CITED [V] dash-2005-emc`; `NOT MEASURED — needs the feedback plant in
  shape_identities.py`); for (c) a two-row edit must break the rank-one formula by O(1) and be
  repaired by Woodbury (`CITED [V] hager-1989-updating`). *Against* V-24, D-5 (declaring `0.0`
  without a movement test, `READ MISTAKES.md:755`), D-7 (both EMC halves filed).
- **Control arm.** The per-row control at matched depth on byte-identical intervention draws
  (McNemar, `CITED [V] mcnemar-1947-correlated`); the "no-change" predictor as the floor arm
  (`CITED [V] vakalis-2026-interventiongap`); a cached-mixture arm that re-reads
  `(I − γP)^{-1}` without re-solving after a `ΔP` (the Momennejad/Russek argument, `CITED [V]
  momennejad-2017-sr`, `russek-2017-predictive`) — it must fail on `ΔP` plants and pass on
  `ΔV` plants, or the interventional channel is not needed.
- **Bed, label.** BED-S moves (each move is one row surgery `P_a[v_a, :] ← e_{u_a}`, `READ
  scale/negation_scope.py:718-732` for the clamping precedent); BED-M E2 consequence as the
  scalar containment case (`READ scale/negation_scope.py:803-850`); BED-K's `bump/rebuild`
  pair as the exact Jacobian oracle (`READ ceq/beds/bed_k.py:256-279`).
- **Metric.** Position-matched per-coordinate NRMSE on `Δz`, field cosine, magnitude ratio;
  sign fidelity kept as a column because softmax already reads it at `0.807843` (`READ
  MATHEMATICS.md:396-406`). *Against* V-17, V-26.
- **Floor.** Exact oracle at `0.0` on the environment chain; the no-change predictor's error
  printed beside it.
- **Certificate.** The re-solve inherits row I's `δ` at the same `γ`; the SM price `O(sd)`
  per candidate is DERIVED and labelled a price with direction `≤` until measured (P-8).
- **Domain census.** Fraction of moves that change `P` (needs a re-solve) vs only `V`
  (does not); fraction with `uᵀ1 = 0` and the edited row non-negative (`100 %` required).
- **Kill (frozen).** Cached-mixture control within one seed sd of the shape on `ΔP` plants on
  `≥ 6/8` seeds ⇒ G1 is a per-row control wearing a name (C5-type kill).

### G2 — the displacement `Δz` as a trained, jointly scored vector output

- **Bind B-G2.** The readout is `[n, s, d]` (or `[n, s]` per coordinate) — the
  `vector_readout` plumbing at `READ scale/m3_quintuple.py:368,483` — and the journal carries
  the vector, a histogram and quantiles (the Q6 F4 census read `0 of 40`, `READ
  V20_R15_THEORY_TABLE.md:213`). Planted negatives for the *metric*: (i) the oracle permuted
  must **not** score `0` (marginal `W1` does, `READ :221`); (ii) `oracle + 0.1σ` must be
  preferred to the permutation; (iii) `−Δz` must be distinguished from `Δz`. *Against* V-26,
  L-14, D-1 (a vector label leaves the single-location regime, `CITED [V]
  duranthon-2026-softmax-advantage` Prop 4.2 is the ground left).
- **Control arm.** Matched-depth softmax with the same vector readout; MuZero-style value head
  as the honest planning skyline (`sweep_causality.md` §4 item 11).
- **Bed, label.** BED-S `Δz` per move; a separate lane with its own journal because a vector
  label voids `PUBLISHED_SOFTMAX_8192` (`READ MISTAKES.md:701-708`). *Against* M-1.
- **Metric / floor / certificate.** As G1.
- **Domain census.** `Var(Δz) > 0` per coordinate on the corpus; `0` blocks registration
  (M-21/M-18 rule).
- **Kill (frozen).** The shape's field cosine within the `n = 8` MDE of the matched-depth
  softmax on `≥ 6/8` seeds ⇒ G2 carries no depth-1 capability; only exactness and the
  certificate survive as *properties*.

### H1 — per-constraint committor / reach-avoid reads as head outputs

- **Bind B-H1.** `q^{(k)} = (I − Q)^{-1} R_k 1` against `bed_1.committor` on the bed's real
  sets: `0.0` with `A = [0], B = [1], |T| = 9` (`RUN` by JUPITER/MARS, `sec_proved.md` §2.8(3),
  C12); the script raises on a missing key — the `basin_A` default that read `0.858` is the
  V-16 the bind now refuses. Planted negatives: the must-fire perturbation drives
  `harmonic_residual` from `1.04e-17` to `1e-6` (ratio `9.6e10`, `READ
  workdonenewseal.md:214-215`); a Kirchhoff second route for `K = 2` single-node boundaries
  (`READ scale/kirchhoff.py:1-90`, `1e-10`); for `K ≥ 3` the second route is `NOT MEASURED —
  needs the grounded Laplacian extended to multi-node boundaries` (P-4, recorded not assumed).
- **Control arm.** ChaCAL same-`γ` (no reads exist — it must emit chance), InfSA-style
  Neumann without boundaries, depth-1 softmax with a `K+1`-way head, the three skylines.
- **Bed, label.** BED-S; the `[m, K+1]` tensor; BED-1 as containment.
- **Metric.** `φ`-NRMSE vector plus harmonic residual (C11).
- **Floor.** Exact oracle `0.0`; restricted-view Fano at `k = 1` computed at construction by
  re-solving the latent chain conditional on the visible window (`sec_beds.md` §6.C.5).
- **Certificate.** Row I's `δ`; at `γ ↑ 1` the certificate factor `1/(1−γ)` is printed beside
  it (C5 attack's mirror).
- **Domain census.** As E1/E2.
- **Kill (frozen).** P3 of `sec_beds.md` §6.E in its refuting branch: softmax's harmonic
  residual within `2×` of the shape's ⇒ "joint determination in one read" is withdrawn
  (`READ MATHEMATICS.md:106-127`); `(2, 10)` is SPLIT and reported.

### H2 — the safest-move rule over candidate moves in the context

- **Bind B-H2.** The argmin over `m` moves computed from the oracle tensor equals the label's
  argmin on `100 %` of admitted draws (an identity of the builder, labelled V-3); the
  rejection region is the 0-hop control below. Planted negatives: a move token that leaks its
  own `q` must make the corpus-alone probe read `R² ≥ 0.99` — the leak detector must fire
  when the leak is planted (C2 attack, `READ MISTAKES.md:2133-2143`); the E4′ local-feature
  decoder must read above `FAIL_BAR = 0.9` (`READ scale/rips_gate.py:61`, `0.973819` on E4′).
- **Control arm.** 0-hop per-position MLP on the candidate's context row; 1-hop softmax;
  majority move; random move (`1/m`); predict-the-mean on the committor tensor; the
  return-conditioned DT-style arm on deterministic beds (`sweep_safety.md` §4.5, `CITED [V]
  brandfonbrener-2022-rcsl`, `paster-2022-luck`).
- **Bed, label.** BED-S argmin (classification over `m`, default `m = 8`); the safety-filter
  reading (`CITED [V] hsu-2023-safetyfilter`) and its least-restrictive threshold form
  `max_k q^{(k)} ≤ δ` (`CITED [V] borquez-2023-lrf`) registered as columns.
- **Metric.** Accuracy with Clopper–Pearson; McNemar paired against each control.
- **Floor.** Zero-hop Fano `0.6667` at `m = 8`; restricted-view Fano at `k = 1`.
- **Certificate.** The threshold form carries row I's `δ`: a move admitted at
  `max_k q̂^{(k)} ≤ δ_thr` is admitted only if `δ_thr − δ·‖V‖_∞ > 0`.
- **Domain census.** `I(s₀; a*) = 0` up to plug-in error; untrained arm within `1/m ± CP`
  (the 0-step RED gate, `GATE_TOL = 1e-3`, `READ COSTS.md:137`); class balance; discard count.
- **Kill (frozen).** C8: the 0-hop MLP's argmin accuracy within the Fano floor's resolution
  of the shape's, or McNemar `p > 0.05` against the 1-hop softmax at `N = 8` ⇒ BED-S is a
  third static task and is struck before any number is quoted.

### I — the printed Neumann certificate with a planted negative (L-CERT)

- **Bind B-I.** `‖(I − γP)^{-1} − Σ_{k≤K}(γP)^k‖_∞ = γ^{K+1}/(1−γ)` to `1e-15` at
  `K ∈ {1,2,4,8,16}` (`RUN` by the coordinator; attained, hence a V-3 identity on the
  row-stochastic class and labelled *definitional*). The bind is carried by (i) the planted
  non-stochastic `P` (rows `1.5`): err `119.37` vs bound `1.143`; (ii) **the shipped mask**:
  `max|O_full − O_mask| ≤ δ·‖V‖_∞` in vector units over `1,024` drawn cells (C6 attack;
  `sec_cost.md` §4.x.3 printed `6.13e-05` against bare `δ = 1.526e-05` and a vector bound
  `≈ 7.6e-05`); (iii) a float32 solve-vs-Neumann diff below `δ·‖V‖_∞` as the must-pass.
  *Against* V-3, V-10, V-17 (units), P-8.
- **Control arm.** The exact solve (`solve_triangular`, `1.8e-15` vs dense inverse); Neumann
  at `K` hops; the chunked path is `NOT MEASURED — needs a chunked kernel` (`sec_cost.md`
  §4.x.2; pattern `CITED [V] yang-2024-deltanet`).
- **Bed, label.** Every bed on which a mask ships; no capability number is read from this
  row.
- **Metric.** `δ·‖V‖_∞` printed beside every mask; `1/(1−γ̂)` printed beside every `δ`.
- **Floor.** None — a certificate is not a capability.
- **Certificate.** F0 for exact zeros (`δ = 0`, `torch.equal`); F1 for soft masks by a
  bounded-increment union bound — cited as the textbook statement (`CITED [V]
  durrett-2019-probability`, `hoeffding-1963-probability`, `azuma-1967-weighted`), never as
  Cantelli/Bonferroni primaries, which are `[U]` (`sweep_methods.md` §1.A).
- **Domain census.** Fraction of drawn cells with row-stochastic `P` (absorbing rows
  included) and `γ ∈ [0, 1)` — the hypotheses of the identity; the measured support of `γ̂`.
- **Kill (frozen).** Exceeded on any of the `1,024` draws; or `δ·‖V‖_∞ ≥ sd(label)`
  (uninformative); or the row-sum identity presented as evidence about `P`.

### J — the learnable discount `γ` in a causal LM read, pinned by the LR test

- **Bind B-J.** `γ = 0 ⇒ O = P V` bitwise (`torch.equal True`; I1), with the record's
  reference that even its own softmax corner is `1.110223e-16` off `ceq/lm.py` on `19/64`
  entries (`READ V16_ARM_SMPRIME.md:266-293`) — "bitwise" is stated against `softmaxAttn`,
  not against the fused kernel. Planted negatives: `γ = 0.5 ⇒ 2.3003` (`RUN`, coordinator);
  a non-causal `P`; a wrong normaliser (`β = 0` at the softmax corner, `> 0.5`, `READ
  V16_ARM_SMPRIME.md:336-339`). *Against* V-24.
- **Control arm.** ChaCAL at fixed `γ = 0.9` (its own setting, `CITED [V]
  fagnou-2024-chacal` App. C); the `γ` sweep `{0, .05, .10, .25, .50, .90}` as the
  precedent that killed hop-2 (`READ workdonenew.md:276`).
- **Bed, label.** Every bed; `γ̂` is journalled per seed with its CI.
- **Metric.** Ruling 10′'s likelihood ratio `Λ = 2[LL(γ̂) − LL(γ ≡ 0)]` on held-out:
  `≤ 3.841` PINNED, `> ln n` MOVED, between prints the interval verdict (`READ
  V17K_RULINGS.md:389-436`).
- **Floor.** None.
- **Certificate.** `1/(1 − γ̂)` beside every `δ`; the instance `1/(1 − â_max)` undefined at
  all eight R1 seeds (`READ V15_R1.md:56`) is the precedent.
- **Domain census.** Measured support of `γ̂` across seeds; fraction PINNED / MOVED /
  interval.
- **Kill (frozen).** `|γ̂| < 0.05` on `≥ 6/8` seeds, or the ablation
  `(I − γ̂P̂)^{-1} → I` moving NRMSE by less than one seed sd ⇒ the arm is softmax wearing a
  name (C5); mirror kill `γ̂ > 0.99` on `≥ 6/8` seeds ⇒ the certificate is vacuous.

### K — machine-checked containment and the zero-gate segmentation iff

- **Bind B-K.** The Lean targets of `sec_proved.md` §2.8 build with zero `sorry` and
  `[propext, Classical.choice, Quot.sound]` only: `gamma_zero_is_softmax` [M],
  `resolvent_fromBlocks` [M], `lower_triangular_isUnit` [M], `segmentation_blockdiag` [M];
  `pathprod_is_chain_resolvent`, `committor_is_resolvent_read` (b)(c),
  `neumann_truncation_bound`, `resolvent_is_triangular_solve` [S]. The float instances:
  corner 3 vs `(I − A)^{-1}` `0.0`; last row vs `equilibrium_oracle` `6.2e-15`; segmentation
  zeros by `torch.equal` (not `allclose`). Planted negatives: a gate at `−30` (F1, not F0)
  and at `1e-300` (nonzero) must leave the block **non**-zero (`sweep_topology.md` §6.2);
  keeping the diagonal must break `A^s = 0` (`one_not_nilpotent`, `READ
  lean/CEQ/Nilpotent.lean:105`) — the regime-N/regime-S boundary of C8 stated as a bind.
  *Against* P-11 (the contract's `#18/#19/#22 [M]` have no declarations, `sec_proved.md`
  §2.0), V-25 (`0 < a` holds on `1 of 3` BED-M values; the `pathProd_*` clauses on `3 of 3`).
- **Control arm.** The exponential prefix scan (`exp_scan`): `nan` on `133,120/133,120` pairs
  (`READ V16_ARM_SMPRIME.md:519-527`) and FoX's `log 0 = −∞` convention (`CITED [V]
  lin-2025-forgetting-transformer`); Zhao et al.'s block-plus-residual approximation
  (`CITED [V] zhao-2026-structuredsparse`) as the approximate fellow.
- **Bed, label.** BED-M as containment (the `96.78 %` annihilation fraction, `READ
  workdonenewseal.md:92`); the segmentation dividend `31.06×` is BED-M's corpus property and
  is not carried elsewhere (V-22).
- **Metric / floor.** None — theorems.
- **Certificate.** F0, `δ = 0`.
- **Domain census.** At least one draw with `m_k = 0` exactly (BED-M: `3 of 3` values).
- **Kill (frozen).** Any `[M]` target that does not build by the plan's Lean milestone
  demotes the row to `[S]` and the paper cites only declarations that build (L-LEAN).

### L — the influence-Jacobian `β₀` barcode with a directed-stability `δ`

- **Bind B-L.** The filtration function is the influence Jacobian `∂O_i/∂V_j` of the
  resolvent read (non-negative, R12); `β₀` across the threshold `ε` is computed by the
  record's own `beta0_interleaving` sandwich (`READ ceq/certs/topological.py:476-500`) with
  `strict_span` so a sandwich that never bites cannot pass (V-16); the `ε = 0` endpoint must
  equal the F0 segmentation count (`torch.equal` on block labels). Planted negatives: the
  row-permuted influence at the same logit scale (the M-15 null); an aliased filtration
  the instrument must **refuse** (the `Z`-winding precedent, `READ V15_X37_CERTS.md:155-169`).
  *Against* M-15, M-2 (threshold grid fixed from the stability constant before the read),
  V-16.
- **Control arm.** The attention-weight filtration of a trained softmax (`CITED [V]
  kushnareva-2021-tda-attention`) on the same cells — the descriptive owner as the fellow.
- **Bed, label.** BED-S; the label is the segmentation count and the barcode's bar lengths;
  an intervention's effect on the barcode is an instance of `CITED [V]
  kim-2026-topological-causal-effects` and is cited as such.
- **Metric.** Bottleneck distance to the null barcode; bar count.
- **Floor.** The null's `95 %` band from `≥ 200` permutations (the X₃₅′ FAR calibration
  pattern, `READ V15_X35A_RESIDUAL.md:147-151`).
- **Certificate.** `δ` from the directed-network stability theorem (`CITED [V]
  turner-2019-quasimetric-rips`), **not** Cohen-Steiner's (the influence graph is asymmetric);
  its constant is `NOT MEASURED — needs the Turner constant evaluated on the shipped
  filtration`.
- **Domain census.** Fraction of cells whose influence matrix has at least one exact zero
  (else the `ε = 0` endpoint is one block and the barcode is decoration).
- **Kill (frozen).** Measured barcode inside the null's `95 %` band on `≥ 6/8` seeds ⇒
  decoration; a non-integer or aliased `β₀` ⇒ refusal, not a pass.

### M — Mapper cover → causal CSR schedule → certified resolvent

- **Bind B-M.** The do-nothing schedule (the merged 0D-salience builder, `CITED [V]
  sharma-2026-kernels-22`; `READ THEORY.md:162-165`) is always entered; the Mapper cover's
  nerve edges become tiles; the resolvent stage on a schedule that drops `(k,l)` but keeps
  `(k,m)`, `(m,l)` must **report** the fill-in and refuse the exact solve unless a Neumann `δ`
  is printed (`sec_cost.md` §4.x.9 check 5); the empty-row `NaN` and negative-index loads of
  `READ ceq/mz_kernel.py:13-21` stay structural. Cover parameters are fixed by the
  Reeb-estimator rule on a held-out relation (`CITED [V] carriere-2018-mapper-statistics`),
  never on the bed. *Against* V-9 (a repair that changes nothing is a first-class outcome),
  M-2, L-CERT.
- **Control arm.** Do-nothing schedule; k-means routing (`CITED [V]
  roy-2021-routing-transformer`); SBM-sampled overlapping masks (`CITED [V]
  cho-2022-sbm-attention`); dense resolvent at the same `s`.
- **Bed, label.** Long-context BED-S at `s ∈ {256, 1024, 4096}`; the label unchanged.
- **Metric.** Visited-tile fraction, wall-clock with `torch.cuda.synchronize()` and
  randomised order (C9), `φ`-NRMSE, harmonic residual.
- **Floor.** Exact oracle `0.0`.
- **Certificate.** F1 union bound (dropped mass) + Neumann `δ` on the sparse `P`, in vector
  units; the coarsening bound of `READ ceq/multizoom.py:38-49` for the far field; a Mapper
  cover with a printed `δ` is `NOT FOUND` in the tree and enters only through the union
  bound.
- **Domain census.** Segment-length distribution of the corpus (the dividend `D = s²/Σ L_m²`
  is a corpus property, `sec_cost.md` (C5)); `NOT MEASURED` for BED-S.
- **Kill (frozen).** Mapper schedule not better than do-nothing by more than the `n = 8` MDE
  at a matched visited-tile budget, or its union `δ·‖V‖_∞ ≥ sd(label)` ⇒ refused. Backward
  through `kernels#22` does not exist (`READ THEORY.md:223-224`); the row is inference-side
  until it does.

## 3. The identity manifest — what every cell journals, so L-2 cannot recur

L-2 (`READ V20_R15_LEAP_LEDGER.md:23`): the F0 theorem is exact on a configuration described
by `β`/`qk`/`g` + normaliser, "no arena cell journals those as cell fields, so 0 of 24 banked
cells are attributable"; repaired by journalling four scalars, not by a theorem. The existing
`CONFIG_FIELDS` (`READ scale/identity_manifest.py:67-71`) are `cell, kind, task, s, d,
d_model, k_piv, beta, t_max, n_neumann, steps, n_train, n_eval, seed, device, torch_version` —
`beta` is there; `qk`, `g`, `gamma` are not, and the manifest **drops a missing declared field
silently** and reports it as `absent` outside the hash (`READ :141-151`). The shape lane's
manifest therefore (DESIGN, against L-2, P-1, V-16, M-10, M-16, V-22):

| field | type / support | why it is identity | mechanism |
|---|---|---|---|
| `beta` | float, `{0, 1}` at the corners, learnable per Ruling 2a | switch that decides softmax-class membership (`beta_one_row_is_one`, `READ lean/CEQ/V16Domain.lean:509`) | L-2 |
| `qk` | float switch, `{0, 1}` | QK on/off; effect size `3.522037` (`READ V16_ARM_SMPRIME.md:29`) | L-2 |
| `g` | gate-on flag plus the gate's parametrisation id | `g ≡ 0` is the parity corner; effect `0.673101` | L-2 |
| `gamma` | init, final `γ̂`, `Λ` and its verdict | row J | C5 |
| `route` | enum `{solve_triangular, neumann_K, segmented, csr}` + `K` + chunk `C` | which computation produced the read; `δ` is meaningless without it | L-CERT, P-8 |
| `boundary_sets` | sorted position lists `𝒜_1..𝒜_K`, sha256 of the list | the boundary condition is the mechanism under test | E1, V-12 |
| `goal_set` | sorted positions incl. BOS, sha256 | R6 | E2 |
| `K`, `m`, `t_star` | ints; `t_star` as the tolerance dial with `E2_DIAL_TOL` | the difficulty dials that must vary | D-3 |
| `S`, `D`, `d_model`, `n_train`, `n_eval`, `steps` | ints | `S = 64` on `40/40` was the D-3 instance | D-3, C9 |
| `seed`, `rng_plan` | int + the `RNG_PLAN` dict (`READ scale/identity_manifest.py:78`) | reproduction | P-1 |
| `device`, `threads`, `dtype`, `torch_version`, `cublas_workspace`, `deterministic_regime` | strings/ints | a cell's identity includes its thread count and flag regime (`READ workdonenew.md:363-366`; `READ V17_R4_RETAKE.md:14-16`) | M-10, M-16, Ruling 1 |
| `instrument_hash` | `instrument_manifest(__file__, reaches=…)["hash"]` (`READ scripts/v15_r1.py:174,830`) | which *scorer* read the cell — distinct from the cell's own `manifest["hash"]` (`READ :819-823`) | Ruling 5 |
| `manifest_hash` | `identity_manifest.manifest(...)["hash"]` over config/code/shapes/rng | which arm produced the cell | V-3 |
| `delta_vec`, `delta_bare`, `one_over_1mg` | floats | the certificate in vector units and its factor | L-CERT, V-17 |
| `census` | the L-DOM block of §2 for the row (fractions, sd, discards) | a theorem without its census is decoration | V-25, V-8 |
| `floor_exact`, `floor_fano_k0`, `floor_fano_k1`, `ceiling_hop_k` | floats | every capability number beside its floor | L-FLOOR |
| `sky_depth`, `sky_width`, `sky_cot`, `chacal_gamma` | the fellow approximators' settings and parameter counts | `Δ_sky` mandatory | R-SKY |
| `supersedes` | pointer to any earlier cell this one replaces; journals never move | append-only | L-G2, P-3 |

Two rules on the manifest itself. (i) In the shape lane a **missing declared field is a
refusal**, not an `absent` entry — the record's own reason for reporting rather than raising
("refusing would invalidate every stored manifest", `READ :146-148`) does not apply to a lane
with no stored manifests. (ii) The manifest ships its planted negative: flipping any one of
`beta / qk / g / gamma / boundary_sets` by one unit must move `manifest_hash`, and the
one-line-drift test of the wing manifest
(`test_moving_one_citation_by_one_line_fires_both_binds`, `READ V20_R15_WING_MANIFEST.md:148-152`)
is the pattern. A cell is FOUND iff `results/` holds a record with the arm's `kind` (`READ
V20_R15_WING_MANIFEST.md:59-63`); everything else is NAMED.

## 4. Part B's top ten, each answered by an apparatus feature

| rank | mechanism | apparatus feature | where |
|---|---|---|---|
| 1 | D-2 the oracle is the arm's own resolvent | every oracle runs on the latent environment chain built inside `bed_s.build`; the arm sees edge tokens only; the pre-registration lists `shape − corner-3` on BED-M as VOID *before* any cell; `‖P̂ − P*‖_∞` journalled per cell | §2 E1/H1, §3 `census` |
| 2 | D-1 racing the optimum | no bed in the lane scores a scalar at one position; labels are `z`, `Δz`, the `[m, K+1]` tensor, the argmin; the `L/d` geometry is printed with the D-1 citation (`CITED [V] marion-2025-single-location`, `duranthon-2026-softmax-advantage`) | §2 G2, H1 |
| 3 | V-24 empty rejection region | every bind carries a mutilation battery with counts; B-E1's identity half and rejection half are separately labelled; R1/R2 are the G1 pair | §2 every row |
| 4 | V-8 / V-12 constant label | the builder prints `sd(q^{(k)})`, class balance, discard count per batch; the goal set exists by construction; BOS declared absorbing (R6) | §2 E2 census |
| 5 | V-25 hypothesis no draw satisfies | the `census` manifest field carries the admitted fraction at the theorem's quantifier level; `0 %` blocks the gate | §3 |
| 6 | M-13 / M-9 unreachable margin | parity by identity bind at `N = 8`; TOST only at `N = 70` and priced (§6); every seed CI prints `n+` | §2 E1 kill |
| 7 | M-21 / M-18 prescribed diagnostic | the arena admits a statistic only after corpus-alone ceiling, zero-step floor, and headroom are printed; `Var = 0` blocks | §2 G2 census, H2 census |
| 8 | D-3 / P-8 dial that does not vary, bound as price | `S ∈ {64, 256, 1024, 4096}` with `synchronize()` and randomised order, `N = 8`; every price carries `≤` and the kernel it assumes | §2 M, §6 |
| 9 | D-7 / L-SIGN prediction without counter | every kill in §2 is the counter to a prediction of equal specificity; the calibration column (checked / wrong / sign) is kept across rounds (`READ V16_CALIBRATION.md:142-203`) | §2 kills |
| 10 | V-3 / V-10 identity of its own construction | row I's attained bound is labelled *definitional*; B-H2's argmin identity is labelled V-3; the shipped-mask measurement carries the certificate | §2 I, H2 |

The three one step behind: V-17 — every threshold is in NRMSE, `φ`, or Fano units, never in
`fd`; M-15 — row L's null barcode; M-2 — every threshold in §2 is frozen in this file.

## 5. Part C's twelve attacks, answered by construction

| attack | construction | the number that decides |
|---|---|---|
| C1 oracle = arm's resolvent | latent chain in the builder; `shape − corner-3` VOID on BED-M by registration | `‖P̂ − P*‖_∞ < 1e-3` on `≥ 6/8` ⇒ copied, no capability |
| C2 feature leak | corpus-alone linear probe to rows of `P*`, to `1_{𝒜_k}`, to `q`; planted positive must fire | `R² ≥ 0.99` on (a)/(b) ⇒ copy task; `R² ≥ 0.5` on (c) at order 0 ⇒ one-read claim void |
| C3 deeper skyline | three fellow approximators at unmatched depth/width/decode; `Δ_sky` column | skyline within `Δ_res` at every rung ⇒ only exactness, one-read consistency, boundary mechanism survive as properties |
| C4 sets trivialise the label | E2 census: `sd`, argmin-unique fraction, reduction-disagreement fraction | `sd = 0`; fraction outside `(0.05, 0.95)`; disagreement `0` |
| C5 `γ` pinned at 0 | Ruling 10′ LR test; ablation; `1/(1−γ̂)` beside `δ` | `|γ̂| < 0.05` on `≥ 6/8`; ablation `< 1` seed sd; `γ̂ > 0.99` on `≥ 6/8` |
| C6 certificate is an identity | attained bound labelled definitional; shipped-mask measurement in vector units over `1,024` draws | exceeded on any draw; `δ·‖V‖_∞ ≥ sd(label)` |
| C7 `W1` permutation-blind | position-matched `φ`-NRMSE + harmonic residual; the three metric plants | permuted oracle scores `0`, or preferred to `+0.1σ`, or `−Δz` indistinguishable ⇒ metric struck |
| C8 static label | 0-hop MLP and 1-hop softmax at matched params, McNemar on identical draws | 0-hop within Fano resolution, or `p > 0.05` at `N = 8` ⇒ BED-S struck |
| C9 cost law never varies `s` | `S ∈ {64, 256, 1024, 4096}`, synchronised, randomised order, `N = 8` | fitted exponent CI excludes `2` toward `3`; solve does not beat `QKᵀ` at the claimed `s`; any price quoted before the run |
| C10 obstructions past hypotheses | the vacuity inequalities printed beside each citation: `512 ≥ 64` (`CITED [V] sanford-2024-inductionheads` Thm 1), `384 < 544` (`CITED [V] peng-2024-limitations` Thm 1), `64^{1/16} = 1.30` (`CITED [V] chen-2024-multilayer` Thm 1.1); conditional bounds named conditional | `0 %` admitted on the bed ⇒ the obstruction is not stated for that bed |
| C11 operator occupied | ChaCAL cited in the first paragraph that writes the read; InfSA in the first that says "absorbing"; the delta is E–M in conjunction | any sentence calling the resolvent new |
| C12 identity script defaults | identity scripts raise on a missing key; B-H1's `0.858` reading is the filed V-16 | a default value in an identity check |

## 6. Price, order, and what is bought first

Prices are `[FITTED + RUN]` from `sec_cost.md` §4.x.8 on the certified RTX 4060 (laptop clock
not stationary, `±12 %`; every figure is a floor with the dispatch gap `2.0×–6.6×` beside it,
P-8): one bed-cell pair (shape on the softmax corner + its softmax control, `N = 8` each,
150 steps, `n = 2048`, `s = 64`) `≈ 34 s`; on the corner-3 base `≈ 133 s`; the BED-S oracle adds
`m·K` dense solves of `|T|³/3 ≈ 5.8e8` flops at `|T| = 1200`, under a second in float64, paid
once per draw (`sec_beds.md` §6.D.5).

DERIVED from those: TOST at `N = 70` for the E1 kill costs `70/8 × 34 s ≈ 298 s ≈ 5 GPU-min`
per pair on the softmax corner and `70/8 × 133 s ≈ 1,164 s ≈ 19 GPU-min` on corner 3 — the
`N = 70` requirement of M-13 is affordable in this lane and is scheduled, not waived. The
`s`-sweep of C9 has no price yet (`NOT MEASURED — needs the --seq-len flag and two
synchronize() calls`, priced by the record at `206–537 GPU-s` band only, `READ
V20_R15_THEORY_TABLE.md:104-109`).

Order (the critical path, D-1 dependency law): (1) Lean `[M]` targets of row K and the
extended manifest of §3 — 0 GPU-s, and L-LEAN forbids training before the identity theorems
are green; (2) `bed_s.build` with the E1/E2 census, guards, and the corpus-alone probes —
0 GPU-s; a bed not admitted produces no reading (D-4); (3) the identity binds B-E1, B-G1,
B-H1, B-I, B-J, B-K on BED-S's real draw with their batteries — seconds; (4) the eight-seed
arena at `t* = 8, m = 8, K = 2` with the control arms of §2 and the three skylines —
`≈ 34 s` per pair, seven arms, `≈ 4 GPU-min`; (5) the realised paired sd fixes the MDE row and
every §2 kill is scored against it; (6) the `N = 70` TOST for E1 only if step 5 leaves E1
alive; (7) rows L and M last — each needs an instrument (`Turner δ`, a Mapper-to-tile
quantiser) that does not exist and is `NOT MEASURED` until it does. Kaggle is not on this
path; the author's explicit yes is a node, not a formality (`READ kaggle/README.md:9-11`).

## 7. Limits (collected once)

Every RUN number in §1 is one draw at `s = 32`, `d = 8`, `γ = 0.6`, seed 0, float64, CPU, with
random logits and hand-placed absorbing rows; none is a statistic and none carries an
interval; the same identities on BED-S's real draws are owed to step (3) of §6. R6 is
structural (row 0 of any causal softmax is `e₀`) but its consequence — BOS declared inside a
boundary set — changes BED-S's semantics (walks move to `j ≤ i`, so constraints must precede
the query, R10) and has been run on one draw only. The E1 kill's threshold is a formula on a
paired sd that is `NOT MEASURED`; the `0.039827` quoted is the record's row-2 sd instance, not
BED-S's. The TOST price is DERIVED from a per-op microbenchmark floor and inherits its `2×–7×`
dispatch gap. Row L's directed-stability `δ` and row M's Mapper-to-tile quantiser have no
instrument and no number. The InfSA-base negative of B-E1 and the feedback plant of B-G1 are
`NOT MEASURED` because they need code this file may not write. The Kirchhoff second route
covers `K = 2` single-node boundaries only. Citations are by bib key from the sweeps' `[V]`
records; no theorem number beyond those the sweeps read at body level is claimed here;
`sanford-2024-inductionheads`, `chen-2024-multilayer` are cited for the inequalities the
expressivity sweep derived, not re-read. No code file, no git write, no Kaggle contact was
made; the three one-liners of §1 computed and printed only.
