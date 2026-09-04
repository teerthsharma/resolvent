# REFUTATION — THEORY, mathematics lens

MARS (MORIARTY), 2026-09-03. Target: `$SCRATCH/design/design_theory.md` (625 lines, JUPITER). Repository `C:\Users\seal\Desktop\New folder (32)`, HEAD `207e7b9`. Inputs read in full: `BRIEF.md`, `THESIS_NOTES.md`, `THESIS_CORRECTIONS.md`, `THESIS_CORRECTIONS_2.md`, the six `sections/sec_*.md`, `references.bib` key table, `bib_aliases.md`; the eight sweeps were read at the passages the design cites (`sweep_expressivity.md` §1.11, §1.20, §3.4–3.5; `sweep_safety.md` §1.A, §4.1–4.3; `sweep_causality.md` §2.B; `sweep_topology.md` §3(a); `sweep_methods.md` on Neumann and Fano; `sweep_linrec.md` §2.3) and otherwise by section headings. Four Lean definitions were read from the tree at HEAD (`V16Domain.lean:366-378, :85-97, :420-428`; `V15Fork.lean:60-70`; `Nilpotent.lean:44-47, :103-110`) and `scale/negation_scope.py:286-304`.

Evidence classes. `RUN[MARS]` — one numpy float64 draw executed this session, `s = 32`, `d = 4`, `γ = 0.6`, seed 0, causal softmax `P` from `N(0,1)` logits with the diagonal, goal set `𝒜_0 = {0, 5}`, constraints `𝒜_1 = {9, 10}`, `𝒜_2 = {15}`, intervened row `i = 12`, cut `c = 12` — the design's own geometry, a different logit scale (the design's draw reads `ρ(Q) = 0.5134`, this one `0.6927`; nothing below turns on the value). No file was written by the run. `READ path:line` — quoted from the tree or from the named section. `CITED [V]` — a `references.bib` key the sweeps fetched; `[U]` where the sweep says so. `DERIVED` — steps written out. No code file, no git write, no external fetch was made.

Verdict rule. KILL where a counterexample exists or the statement is false as written; REPAIR where the statement is true after a stated correction; KEEP where the attack failed. Every KILL and REPAIR names a number, a bib key or a `file:line`, and the `MISTAKES.md` mechanism the defect instantiates.

---

## 0. Summary of verdicts

| # | target | verdict | one line |
|---|---|---|---|
| 1 | §1.1 base + P6 annihilation | REPAIR | the base as defined (`Hop`, `exp(scan g)`, `V16Domain.lean:366-378`) has no zero; `pathProd_eq_zero_iff` is about a different object; `path_product_corner_fails_at_a_zero_gate` (`:424`) is the theorem that says so |
| 2 | §1.2 regimes / P5a diagonal interval | REPAIR | diagonal of `I − γP` lies in `[1−γ, 1)`, not `(1−γ, 1]`; both endpoints wrong; row 0 attains `1−γ` on every causal softmax |
| 3 | §1.3 F1, F2 | KEEP | with two additions: F1 is regime-S only; for causal `P`, `ρ(Q) = max_{i∈T} P_ii` and `I − Q` is invertible iff `0 ∈ 𝒜` — a diagonal read, no eigen-solve |
| 4 | §1.4 `(1−γ)` factor / row sums | KEEP | wording: the bare row sum `1/(1−γ)` holds with and without boundary rows (`2.5` both, RUN) — it is not a boundary-row effect |
| 5 | §1.6 support of `P' − P` | KILL | a token `do()` at position `i` in regime S changes rows `i..s−1` (20 of 32, rank 20, RUN), not "the intervened rows" |
| 6 | P1 | KEEP | |
| 7 | P2 | KEEP | |
| 8 | P3 identities | KEEP | the identities; REPAIR the evidence line `3.8e-07` (impossible: the gap is `≥ 1−γ = 1e-6` on every `i ∈ T`) |
| 9 | P4 equality | KEEP | REPAIR the clause "for `‖P‖_∞ > 1` no such bound holds" (nilpotent `A` with `‖A‖_∞ = 2` satisfies it at `K = s−1`) and label the `γ = 0.7`, rows-`1.5` plant as a divergent series |
| 10 | P5b hull, P5a solve | KEEP | REPAIR the "transitive closure" sentence: in regime S `supp Π_γ = supp P` exactly (RUN), so the sentence is empty |
| 11 | P6 block-diagonal + absorbing rows | KILL as composed with P3/P8 | a zero-gate cut at `c` makes `c` a new undeclared absorbing BOS (`P_cc = 1`, `ρ(Q) = 1`, `det(I−Q) = 0`) and zeroes every committor into a set before the cut for all `i ≥ c` (RUN); the design never states it |
| 12 | P7 / §3 arithmetic | KEEP | `64^{1/16} = 1.2968`, `384 < 544`, `3/5/7` all check |
| 13 | §3.2 "linear systems P-complete" vs §3.3 `DET ⊆ NC²` | REPAIR | the two paragraphs contradict each other; linear equalities over ℚ are in `DET ⊆ NC²` (`cook-1985-taxonomy`); P-completeness is linear *inequalities* |
| 14 | §3.3 "`O(log s)` parallel rounds by prefix doubling" | REPAIR | a dense triangular inverse by recursive block inversion is `O(log² s)` depth with `O(s³)`-class work; `O(log s)` counts a matmul as one round |
| 15 | P8 degeneracy lemma | KEEP | |
| 16 | P8 Fano floor `1 − ln 2/ln m` | REPAIR | the weak Fano form reads `0.5 / 0.667 / 0.75`; chance error is `1 − 1/m = 0.75 / 0.875 / 0.9375`; an arm at error `0.6`, `m = 4` "beats the floor" while worse than chance (V-17) |
| 17 | P8 cost clause "`m(K+1)` RHS when moves change only values" | KILL | `q^{(k)}` depends on `P` only; a value-only move leaves every reach-avoid label constant across moves — V-8 built into the design |
| 18 | P9 Sherman–Morrison at an absorbing row | KEEP | denominator `(1−γP'_ii)/(1−γP_ii) > 0` for `γ < 1`; RUN `1.4e-15` |
| 19 | P9 / C6 rank-one price `O(sd)` per candidate | KILL for token `do()` | valid for a row clamp on the latent chain (the oracle); invalid for the arm's softmax `P` (item 5); the honest price is C6(ii)'s suffix re-solve `O((s−i)² d/2)` |
| 20 | C6 (i)–(iii) | KEEP | hypothesis of (ii) restated: `P'_{<i,·} = P_{<i,·}` and `V'_{<i} = V_{<i}`, which a softmax `P` satisfies because the key of position `i` enters rows `≥ i` only |
| 21 | C8 first half | KEEP | |
| 22 | C8 masked-diagonal option | KILL as a matrix statement | with the sink row `e_0` the matrix is not nilpotent: `‖(I−γP)^{-1} − Σ_{t<s}(γP)^t‖ = 1.99e-07 = O(γ^s)`; exact on `V` with `V_0 = 0` (`4.4e-16`) and **not** on `V = 𝟙_{𝒜_0}` with `0 ∈ 𝒜_0` (`1.99e-07`) — F1 and the masked option contradict |
| 23 | P10 | KEEP | as a corollary; the `γ = 1` endpoint with absorbing rows is singular (`det(I − P') = 0`, RUN) and the design correctly takes a limit |
| 24 | §6 Lean target 16 grade | REPAIR | for the arm's causal `Q`, `IsUnit (1 − Q)` is a triangular determinant (`[M]`, target 5); the Perron route `[S]` is needed only for BED-1's non-causal oracle `Q` |
| 25 | §6 Lean target 4 statement | REPAIR | `0 < P i i ⇒ (γ•P)^k i i > 0` needs `0 < γ` and either `P ≥ 0` or lower-triangularity; the design's proof line `(A^k)_ii ≥ (A_ii)^k` uses non-negativity it does not state |
| 26 | P6 dividend `31.06×` | REPAIR | it is the pair-count ratio `s(s+1)/Σ L_m(L_m+1) = 31.04`, not (C5)'s `s²/Σ L_m²` (`58.5` on the implied distribution) — V-17 |

Fatal (would change the plan if not repaired): items 5/19 (the pricing of the safest-move search rests on an update that does not apply to the arm), 11 (segmentation and committors are incompatible as stated), 17 (a value-only move gives a constant label), 22 (the masked-diagonal route is not exact on the goal channel).

---

## 1. Definitions

### 1.1 The base family and P6's annihilation (REPAIR) — mechanism V-25, P-7

The design defines the base at §1.1 as `W_{β,qk,g}[i,j] = 𝟙[j≤i]·exp((scan g)_i − (scan g)_j + qk_ij)/Z_i^β`, citing `READ V16Domain.lean:366-378`. That definition was read this session: `num g qk i j := Real.exp ((scan g i − scan g j) + qk i j)` and `Hop β g qk i j := if j ≤ i then num g qk i j / (Znorm g qk i)^β else 0`. `Real.exp` is never zero, so **no setting of `(β, g, qk)` produces a zero entry**. P6 then states "in the base with the path-product gate, `m_c = 0` gives `W_ij = 0` (`pathProd_eq_zero_iff`, `:129`, no hypothesis)". `pathProd m θ i j := ∏ gateOf (m k) (θ k)` (`V16Domain.lean:97`) is a different object from `Hop`, and the tree's own theorem `path_product_corner_fails_at_a_zero_gate` (`:420-428`, READ) states that for every `g`, whenever a zero magnitude sits on the path, `Hop 0 g 0 i j ≠ pathProd m θ i j`. So the design's base and P6's annihilating operator are provably distinct objects, and P6 as written on §1.1's `W` is false: there is no `m_c` in `W_{β,qk,g}`.

**Repair.** Define the base with the multiplicative gate the arm actually runs (`ceq/arm_smprime.py:144 path_product`, READ via `sec_proved.md` §2.3: the product route, "segmented scan as the upgrade"): `W_ij = 𝟙[j≤i]·pathProd(m,θ)_{ij}·exp(qk_ij)/Z_i^β`, and state that the `exp(scan g)` form of §1.1 is its restriction to `m > 0` (`prefix_logit_mask_restated` clause 5, `:221`). Then `pathProd_eq_zero_iff` applies and P6's annihilation half is LEAN. Without the repair the design's P6 cites a theorem whose subject is absent from its own definition (V-25 in definitional form; P-7, a vocabulary with no referent).

### 1.2 Regimes and the diagonal interval of P5a (REPAIR) — mechanism V-17

P5a: "the diagonal `1 − γW_ii` lies in `(1 − γ, 1]` in regime S". For a causal softmax row `i ≥ 1` over `{0..i}` every weight is positive, so `P_ii ∈ (0, 1)` and `1 − γP_ii ∈ (1−γ, 1)`; for row 0 the window is `{0}` so `P_00 = 1` (F1) and the diagonal entry is exactly `1 − γ`; for a declared absorbing row `a`, `P_aa = 1` likewise. `RUN[MARS]`: `min diag(I − γP_b) = 0.4 = 1 − γ` (attained), `max = 0.99612 < 1`; without any declared absorbing row `min = 0.4` at row 0. The correct interval is **`[1−γ, 1)`**: the design's interval excludes the value every causal softmax attains at row 0 and includes a value none attains. The forward substitution is unaffected (`1 − γ > 0` for `γ < 1`); the cost line and `lower_triangular_isUnit` survive. `sec_cost.md` §4.x.1 already writes `≥ 1 − γ > 0` correctly; the design regressed it.

The regime table also needs one sentence: an absorbing row is not `StrictlyLower` (`Nilpotent.lean:46-47`, `A i j = 0` for `i ≤ j`, so `A_aa = 1` violates it), hence **regime N admits no absorbing row**, and at `γ = 1` a single absorbing row makes `I − A'` singular (`RUN[MARS]`: `det(I − A) = 1.0000` strictly lower, `det(I − A') = 0.0` with row 5 set to `e_5`). F1's "BOS absorbing by construction" is a regime-S fact: in regime N row 0 is the zero row (a source), not `e_0`. The design's P2 and F1 are each right in their regime and the file should say the two do not meet.

### 1.3 F1 and F2 (KEEP, with two additions) — mechanism V-25, V-8

`RUN[MARS]`: row 0 equals `e_0` exactly (`array_equal True`, `P_00 = 1.0`); with BOS undeclared `ρ(Q) = 1.0` and `det(I − Q) = 0.0`; with `0 ∈ 𝒜_0`, `ρ(Q) = 0.6927`. F2: `q^{(2)}_i = 0.0` exactly for every `i < 15`. Sum-to-one on `T`: `1.3e-15`. All three hold.

Addition 1 (DERIVED). For a causal `P`, `Q = P_TT` is lower-triangular, so its spectrum is its diagonal and `ρ(Q) = max_{i∈T} P_ii` — `RUN[MARS]`: `ρ(Q) = 0.692660 = max diag Q = 0.692660`. `I − Q` is invertible iff no transient row has `P_ii = 1`; for softmax rows `i ≥ 1` this is automatic, so **for the arm's `P` the whole of "absorption a.s." is the single condition `0 ∈ 𝒜`**. This is the cheapest possible domain census (a diagonal read) and it moves Lean target 16 (§6 item 24 below). The Perron certificate is needed only for the *oracle's* non-causal `Q` (BED-1, `SymmSupport`, `ρ(Q) = 0.9409`, `READ sec_proved.md` §2.1) — the design should say which `Q` each theorem is for, or it conflates the arm's regime-S operator with the oracle's undirected chain (the D-2 boundary drawn the wrong way round).

Addition 2. F2's "prints reachability at construction" must also print that the *query* position is in `T`; a query inside a boundary set reads its own indicator and the label is constant (V-8).

### 1.4 The `(1−γ)` factor (KEEP; wording) — mechanism V-17

`RUN[MARS]`: bare read `P(I−γP)^{-1}` row sums `[2.4999…, 2.5000…]` with no absorbing rows and `[2.4999…, 2.5000…]` with the three sets declared; `1/(1−γ) = 2.5`. The design's sentence "With boundary rows the bare read … has row sums `1/(1−γ)`" is true and misleading: it is true for every row-stochastic `P`, boundary rows or not (`P^t 𝟙 = 𝟙`). The paragraph should say the factor is a property of the class, not of the boundary rows, or a reader will look for a boundary-row mechanism that is not there (V-17, a constant imported with the wrong attribution).

### 1.6 The support of `P' − P` under `do(a)` (KILL) — mechanism P-8, M-8

The design: "`P' − P` supported on the intervened rows". In regime S the row `P_{j,·}` is a softmax over the keys `k_0..k_j`; rewriting the token at position `i` changes `q_i` (row `i`) **and `k_i`, which enters every row `j ≥ i` through both the logit `qk_{ji}` and the normaliser `Z_j`**. `RUN[MARS]` (perturb the logits of column 12 and row 12 of a `32 × 32` causal softmax): rows changed `20 of 32`, exactly rows `12..31`; `rank(ΔP) = 20`. The support statement holds only for an intervention defined as a *row clamp* (`P'_{i,·} ← e_u`), which is what BED-S's generator does on the latent chain (`sec_beds.md` §6.C.1, "clamps one transient node's outgoing row"). So the oracle's intervention is rank-one and the arm's is not. Everything downstream that prices a candidate by Sherman–Morrison (P8's "`O(sd)` per candidate after the first solve", C6's "rank-one price", `sweep_safety.md` §4.3) inherits the error — see §2.9 below.

---

## 2. Propositions

### P1 (KEEP)

The float clause is correct for finite entries: `0·x = +0`, `v − 0 = v`, `v/1 = v` are exact in IEEE-754 round-to-nearest; the only escape is a non-finite `x` (`0·∞ = NaN`), which the finiteness hypothesis excludes. The rejection region (`2.3003` at `γ = 0.5`, `RUN[coord]`) is non-empty. Nothing to break.

### P2 (KEEP)

`(I − A)^{-1}_{ij} = ∏_{k=j+1}^{i} a_k` for the sub-diagonal `A_{i,i−1} = a_i` is the standard bidiagonal inverse; `equilibrium_oracle` (`scale/negation_scope.py:298-304`, READ this session: `z = a[:, i]·z + b[:, i]`) is its last row. The proof's walk-counting argument is exact because a sub-diagonal matrix has one walk per `(i, j)`. KEEP.

### P3 identities (KEEP) and the `3.8e-07` evidence line (REPAIR) — mechanism P-1

The three identities hold with the stated conventions (`τ_k = 0` on `𝒜_k`, `γ^∞ = 0`): for `j ∈ 𝒜_k`, `z_j = Σ_t γ^t = 1/(1−γ)` so `(1−γ)z_j = 1 = γ^0`; for `j ∈ 𝒜_l`, `l ≠ k`, `z_j = 0 = γ^∞`; the read `O_i = Σ_j P_ij (1−γ)z_j = E_i[γ^{τ_k − 1}]` by first-step analysis with `τ_k ≥ 1` from `T`. The limit is monotone convergence (`γ^{τ−1} ↑ 𝟙[τ<∞]`). KEEP.

The evidence line "`(1−γ)z` at `γ = 1 − 10^{-6}` vs the Dirichlet committor to `3.8e-07` (the `O(1−γ)` gap)" cannot be what it says. DERIVED: for every `i ∈ T`, `q_i − E_i[γ^{τ}] = E_i[1 − γ^{τ}] ≥ (1 − γ)·P_i(τ ≥ 1) = 1 − γ = 10^{-6}`, because `τ ≥ 1` surely from a transient state and absorption is a.s. So the max-abs gap over `T` is **at least `1e-6`**, and `3.8e-07 < 1e-6` is impossible for `(1−γ)z_T` against `q`. `RUN[MARS]` at the same `γ`: max gap `4.33e-06`, min gap `2.47e-06`; the read `E[γ^{τ−1}]` gaps by `3.33e-06`. Either the number belongs to a different quantity (a row in `𝒜`, where both sides are `0` or `1`) or it was transcribed wrongly; either way it is a number without a live producer (P-1). Repair: re-run and print `max_T |(1−γ)z − q|` together with `(1−γ)·E_i[τ]` beside it, which is the quantity the gap tracks.

Minor: Limits says the tail of the `4000`-term series is "below `10^{-1000}`" from `ρ(Q) = 0.513`. A triangular `Q` is non-normal and `‖Q^u‖_∞ ≤ ρ(Q)^u` does not hold termwise; what holds is `‖Q^u‖_∞ ≤ 1` (sub-stochastic) so the tail is bounded by `Σ_{u>4000} γ^u ≈ 0.6^{4000} ≈ 10^{-887}` — still negligible, but the printed exponent is not licensed by the printed argument.

### P4 (KEEP the equality; REPAIR two clauses) — mechanism V-3, V-24

The equality for `P ≥ 0` row-stochastic is exact: the tail is entrywise non-negative with every row sum `γ^{K+1}/(1−γ)`, and the `∞`-norm of a non-negative matrix is its largest row sum. KEEP. The `≤` form for `‖P‖_∞ ≤ 1` is the triangle inequality. KEEP.

Clause "for `‖P‖_∞ > 1` no such bound holds" — KILL as a universal, REPAIR to "can fail". Counterexample `RUN[MARS]`: `A` sub-diagonal with `A_{i,i−1} = 2`, `‖A‖_∞ = 2`, `γ = 0.7`, `K = s − 1 = 31`: truncation error `1.98e-10` against bound `γ^{32}/(1−γ) = 3.68e-05` — the bound holds because `A` is nilpotent and the series terminates. More generally the series converges whenever `ρ(γP) < 1` regardless of `‖P‖_∞`. The design's planted negative (rows summing to `1.5` at `γ = 0.7`) has `‖γP‖_∞ = 1.05 > 1` and `ρ(γP) = 1.05` (Perron root of a positive matrix equals its constant row sum), so **the Neumann series diverges** and the reported `119.37` is a divergent partial sum, not the truncation error of an inverse that the series approaches; the bind still fires, but the paper should say why. The `γ = 0.6` plant (`7.29` vs `0.54`, `‖γP‖_∞ = 0.9 < 1`, series convergent) is the clean one and should be the one quoted.

### P5 (KEEP a, b; REPAIR the support sentence) — mechanism V-17, P-7

P5b: `Π_γ = (1−γ)Σ_t γ^t P^{t+1}` is a convex combination of row-stochastic matrices. KEEP. The limitation paragraph says "the support of `Π_γ` is the transitive closure of `P`'s support". True for non-negative `P` (no cancellation), and **empty in regime S**: a causal softmax `P` has support `{j ≤ i}`, which is already transitively closed, and an absorbing row `e_a` is a self-loop that removes nothing from the closure since every transient row reaches every `j ≤ i` in one step. `RUN[MARS]`: `supp(Π_γ) == supp(P_b)` is `True` with the three boundary sets declared. So in the regime the north star lives in, the shape adds **no** support at all; the whole gain is in the weights, and the sentence about transitive closure should be moved to regime N and the zero-gate family, where it has content.

### P6 composed with P3/P8 (KILL as composed; REPAIR the statement) — mechanism V-25, V-8, D-3

The block-inverse step is correct: a lower-triangular matrix with a zero lower-left block is block-diagonal and inverts blockwise. The design stops there. What it does not state, and what breaks the composition with absorbing rows:

- After a cut at `c` the row `c` has window `{c}` (every `j < c` annihilated) and renormalises to `P_cc = 1`. `RUN[MARS]`: `P_cc = 1.0` after cutting at `c = 12`. **Position `c` is a new absorbing BOS, undeclared.** With the three boundary sets declared and `c ∉ 𝒜`: `ρ(Q) = 1.0`, `det(I − Q) = 0.0` — the committor solve is singular, exactly F1's failure at every segment head.
- Declaring `c` absorbing (in some set) restores invertibility, and then every constraint set that lies before the cut is unreachable from every `i ≥ c`: `RUN[MARS]`: `q^{(0)}_i = 0.0` and `q^{(1)}_i = 0.0` for all `i > c` (`𝒜_0 = {0,5}`, `𝒜_1 = {9,10}` both before `c = 12`); only `q^{(2)}` (`𝒜_2 = {15}`, after the cut) is non-trivial.

So segmentation and the reach-avoid read are incompatible unless every segment carries its own goal set, its own constraint sets and a declared segment-head sink. The dividend of `sec_cost.md` (C5) is real for the *solve*; for the *label* a cut is a change of task. The design should carry this as a proposition ("a zero gate is a boundary condition: it makes the segment head absorbing and severs every boundary set behind it") with the RUN numbers above as its instance, and register the cut position as a dial that BED-S must vary (D-3) with the reachability print of F2 extended to "per segment".

### P7 and §3 arithmetic (KEEP)

`64^{1/16} = 2^{0.375} = 1.2968` ✓; `n log₂ n = 64·6 = 384`, `H(d+1)p = 1·17·32 = 544`, `384 < 544` ✓; `⌊log₂ t*⌋ + 2 = 3/5/7` at `t* = 2/8/32` ✓. The complexity-class sentences are attacked in §3 below.

### P8 degeneracy lemma (KEEP); Fano floor (REPAIR); the value-only cost clause (KILL) — mechanism V-17, V-8

Degeneracy: `Σ_k q^{(k)} = 𝟙` on `T` gives `max_k q^{(k)} ≥ 1/K` by pigeonhole. KEEP.

Fano. `1 − ln 2/ln m` is Fano's *weak* form (`h(P_e) ≤ ln 2`, `ln(m−1) ≤ ln m`) at `I = 0`; the exact zero-information floor is chance, `P_e ≥ 1 − 1/m`, and the exact Fano with the binary entropy recovers it (`h(0.75) + 0.75·log₂ 3 = 0.811 + 1.189 = 2.000 = log₂ 4` at `m = 4`). `RUN[MARS]`: weak form `0.5000 / 0.6667 / 0.7500` against chance `0.7500 / 0.8750 / 0.9375` at `m = 4/8/16`. `sec_beds.md` §6.E P1 itself calls `0.875` "chance" and `0.6667` "the floor" in one sentence. An arm reading error `0.60` at `m = 4` would be printed as `0.10` above the floor while being `0.15` worse than guessing. Repair: the zero-hop floor is `1 − 1/m`; the restricted-view floors at `k ≥ 1` use the exact Fano (solve `h(p) + p·ln(m−1) = ln m − I` for `p`), never the weak form. The design copies VENUS's number without the check (V-17: a threshold imported out of its units).

Cost clause "one solve with `m(K+1)` right-hand sides when moves change only values". `q^{(k)} = (I − Q)^{-1} R_k 𝟙` contains no `V`; a move that changes only values leaves every `q^{(k)}(a)` equal across `a`, and the argmax over `a` is a tie on every draw (V-8 by construction; the builder's own tie-discard rule, `sec_beds.md` §6.C.4 guard 2, would discard 100 % of such draws). KILL for the reach-avoid channel; the `m` right-hand-sides route is legal for the displacement channel `Δz` (P9), which does see `ΔV`, and the sentence must say so.

### P9 Sherman–Morrison (KEEP the formula; KILL its application to token `do()`) — mechanism M-8, P-8

Formula check at an absorbing row: `u = e_i − P_i`, `M = (I−γP)^{-1}` lower-triangular with `(Me_i)_i = 1/(1−γP_ii)` and `(Me_i)_j = 0` for `j < i`; `u` is supported on `j ≤ i`, so `uᵀMe_i = (P'_ii − P_ii)/(1−γP_ii)` and the denominator is `1 − γuᵀMe_i = (1 − γP'_ii)/(1 − γP_ii) > 0` for `γ < 1`. `RUN[MARS]`: denominator `0.492623` both ways; Sherman–Morrison vs re-solve `1.42e-15` with row 12 clamped to `e_12`. KEEP; the update is safe at an absorbing row for every `γ < 1` and blows up only at `γ = 1` (where `I − P'` is singular, §1.2).

Application. The formula needs `ΔP = e_i uᵀ`. By §1.6 above the arm's `ΔP` under a token intervention has rank `s − i` (`20` at `i = 12`, RUN). The design's price "`O(sd)` per candidate after the first solve" (P8, C6, `sweep_safety.md` §4.3) is therefore the price of the **oracle's** clamp on the latent chain and not of the arm's forward; the arm's exact price is C6(ii)'s suffix re-solve, `≈ (s−i)²d/2` MACs, which the design already proves equal to the full re-solve. Pricing the arm at the oracle's rate is M-8 in its literal form.

### P10 (KEEP)

Corollary of P1/P3/P2; the endpoint `γ = 1` with absorbing rows is singular and the design takes the limit, which is right.

### C6 (KEEP) — hypothesis of (ii) restated

(i) is four lines of algebra and holds for any invertible `I − γP'`. (ii) requires `P'_{<i,·} = P_{<i,·}` and `V'_{<i} = V_{<i}`, which the design phrases as "depends only on positions `< i`"; for a softmax `P` it holds because `k_i` and `v_i` enter rows `≥ i` only (RUN: rows `< 12` unchanged). (iii)'s feedback plant is the right rejection region. The design's D-7 filing of both halves is correct.

### C8 (KEEP the theorem; KILL the masked-diagonal sentence as a matrix identity) — mechanism V-25, V-3

"Mask the diagonal (rows attend to `j < i` with the `j = 0` value-zero sink absorbing row 0) and the finite sum is inherited exactly." With row 0 `= e_0` the matrix is not `StrictlyLower` and `(P^k)_00 = 1` for every `k`; `pow_card_eq_zero` does not apply. `RUN[MARS]`: `‖(I−γP_m)^{-1} − Σ_{t<32}(γP_m)^t‖_max = 1.99e-07`, entry `(0,0)` `2.5000` vs `2.49999980`; `‖(γP_m)^{32}‖_max = 7.96e-08 ≠ 0`. What is true (DERIVED, block form `P_m = [[1, 0],[r, A]]` with `A` strictly lower on rows `1..s−1`): `Σ_{t<s}(γP_m)^t V = (I−γP_m)^{-1}V` **iff the sink carries value zero**, `V_0 = 0`, because then column 0 never enters. `RUN[MARS]`: `4.4e-16` on Gaussian `V` with `V_0 = 0`; `1.99e-07` on `V = 𝟙_{𝒜_0}` with `0 ∈ 𝒜_0`. F1 puts BOS in the goal set, whose value channel is `𝟙_{𝒜_0}` with `V_0 = 1`. **The masked option and F1 contradict on the goal channel.** Repair, one of: (a) keep the diagonal and carry P4 (the design's other option, no change); (b) put BOS in its own sink set `𝒜_sink` with value `0` (InfSA's leak state, `roffo-2026-infsa`) — then the finite sum is exact for every channel, the reach-avoid identity becomes `q^{(0)} + Σ_k q^{(k)} + q^{(sink)} = 𝟙` and the sink share is printed beside the label (the design's own "delay share" discipline). Either way the sentence "inherited exactly" is withdrawn as a matrix identity and restated as a value-channel identity with the new Lean target `masked_sink_finite_sum : V 0 = 0 → occupancy (γ•P_m) n *ᵥ V = (1 − γ•P_m)⁻¹ *ᵥ V`, which is not `pow_card_eq_zero` and is not in the tree.

---

## 3. The complexity-class sentences

### 3.2 "linear systems P-complete" (REPAIR) — mechanism P-10

§3.2 lists, from `merrill-2023-parallelism`, "linear systems P-complete"; §3.3 says "matrix inversion is `DET`-complete and `NL ⊆ DET ⊆ NC²`" from `cook-1985-taxonomy` (CITED [V]). Both cannot stand: solving `Ax = b` over ℚ reduces to computing a determinant-class object and is in `DET ⊆ NC²`; if it were P-complete then `P ⊆ NC²`. The P-complete problem in the Greenlaw–Hoover–Ruzzo list is linear *inequalities* (linear programming). The sweep's line (`sweep_expressivity.md:191`, "linear equalities `Ax = b` (P-complete)") is a summariser reading marked as such at `:55`, and the design carried it into a paragraph that contradicts its own next section. Repair: drop the P-complete clause, cite Cook for the class, and keep the honest sentence "the exact solve is at or above the class the conditional bounds place reachability in" — which is all §3.3 needs.

### 3.3 "`O(log s)` parallel rounds by prefix doubling at `O(s³)`-class work" (REPAIR) — mechanism P-8, V-17

The solve `z_i = (v_i + γΣ_{j<i} P_ij z_j)/(1−γP_ii)` is a recurrence whose state at step `i` is all of `z_{<i}`; it is not a fixed-dimension first-order recurrence, so the affine-monoid prefix scan the record proved (`V15.scan_assoc`, `V15.lean:314`) does not apply. The parallel route is recursive block inversion, `M^{-1} = [[M_{11}^{-1}, 0],[−M_{22}^{-1}M_{21}M_{11}^{-1}, M_{22}^{-1}]]`, which has `log₂ s` levels each containing a matrix product of depth `O(log s)`: total depth `O(log² s)`, work `O(s³)` (or `O(s^ω)`). "`O(log s)` rounds" is right only if a dense matmul counts as one round, which is the unit the paragraph itself warns about ("depth has two units here"). Repair: write `O(log² s)` gate depth, `O(log s)` matmul rounds, `NOT MEASURED`.

### 3.3 `DET`-completeness of the *triangular* solve (KEEP)

Attack attempted: perhaps a triangular inverse is easier than `DET`. It is not: iterated matrix product `B_1 B_2 ⋯ B_m` embeds in `(I − A)^{-1}` for the block sub-diagonal `A = subdiag(B_1, …, B_m)`, so triangular inversion is `DET`-hard (DERIVED, the standard reduction). The design's sentence survives.

---

## 4. Regime conflations found

1. F1 (regime S) is quoted beside P2 (regime N) as if both hold at once; in regime N row 0 is a zero row (§1.2).
2. `Contraction.rowStochastic_perron` (arm, regime S, `γ < 1`) is quoted for `I − Q` invertibility (P3 hypothesis), where the arm's `Q` is triangular and needs only a diagonal read (§1.3); the Perron route belongs to the oracle's `Q`.
3. The rank-one update (oracle, clamp) is priced as if it were the arm's forward (§1.6, P9).
4. The masked-diagonal route (nilpotent, value-zero sink) is quoted for the goal channel `𝟙_{𝒜_0}` whose BOS value is `1` (C8).
5. P6's annihilation theorem (`pathProd`, product gate) is quoted for the base defined with `exp(scan g)` (`Hop`), which the tree proves cannot annihilate (§1.1).

---

## 5. Lean targets (§6 of the design)

| # | verdict | reason |
|---|---|---|
| 1 `gamma_zero_is_softmax` | KEEP | `Matrix.inv_one` suffices |
| 2 `bos_row_is_absorbing` | KEEP | `Finset.range 1`; add the hypothesis `β = 1` explicitly — at `β = 0` row 0 is `num g qk 0 0 = exp(qk 0 0)`, not `1` |
| 3 `later_boundary_unreachable` | KEEP | |
| 4 `softmax_corner_not_nilpotent` | REPAIR | state `0 < γ` and lower-triangularity (then `(γ•P)^k i i = (γ P i i)^k` exactly); the design's proof line uses `A ≥ 0` it never states |
| 5 `lower_triangular_isUnit` | KEEP | with the interval of P5a corrected to `[1−γ, 1)` in `diag_one_sub_smul_pos`'s docstring |
| 6 `segmentation_blockdiag` | KEEP for `StrictlyLower A` | add `segment_head_is_absorbing : cut c ⇒ P c c = 1` and `cut_severs_boundary_sets` (RUN §2.6) as the refusals it ships; the `Hop` form needs the §1.1 repair first |
| 7 `displacement_identity` | KEEP | |
| 8 P2 chain | KEEP | |
| 9–10 Neumann | KEEP | drop "no such bound holds" from the refusal's docstring; the `3/2`-row witness must be paired with `γ` such that `γ·3/2 < 1` (the `γ = 0.6` plant) or labelled divergent |
| 12 `mixing_matrix_rowStochastic` | KEEP | |
| 13, 14 causal forward-only, EMC | KEEP | |
| 15 committor read | KEEP | |
| 16 `isUnit_one_sub_of_perron` | REPAIR grade | for the arm's causal `Q` the statement is `[M]` via target 5 (`Q i i < 1` for `i ∈ T` ⇐ `0 ∈ 𝒜`); the Perron `[S]` version is for the oracle's `Q` only; say which |
| 18 `reach_avoid_sum_one` | KEEP | add the hypothesis "the sets exhaust the absorbing states" to the Lean statement; without it the sum is `< 𝟙` (the sink case of C8's repair) |
| 19 `sherman_morrison_row` | KEEP | with the docstring "applies to a row clamp; a token intervention is rank `s − i`" |
| new | `masked_sink_finite_sum` (`V 0 = 0`) | the theorem C8's masked option actually needs; `[S]` |
| new | `cut_makes_segment_head_absorbing` | the theorem P6 owes to P3; `[M]` (row `c`'s window is `{c}`) |

---

## 6. What would change the verdicts

- Item 5/19 (KILL): a definition of `do(a)` in the paper that is a *row clamp on the arm's `P`* (fix `q_i`, freeze `k_i`, `v_i`), not a token rewrite; then `ΔP = e_i uᵀ` and the `O(sd)` price returns. That is a different primitive from "rewrite the tokens it controls" and the paper must choose.
- Item 11 (KILL): a statement that BED-S never uses a zero gate, or that boundary sets are declared per segment with the segment head in a sink set; then P6 and P3 coexist.
- Item 17 (KILL): the sentence restricted to the `Δz` channel.
- Item 22 (KILL): BOS moved to a value-zero sink set, with the sink share printed; or the masked option dropped.
- Item 13 (REPAIR): a fetched line of `merrill-2023-parallelism` showing it says "linear equalities are P-complete" would move the defect from the design to the source and the paper would then have to cite Cook against it; it would not make the sentence true.
- Item 8 (REPAIR): a re-run printing a value `≥ 1e-6` for `max_T|(1−γ)z − q|` at `γ = 1 − 10^{-6}`.

## 7. Limits

Every `RUN[MARS]` number is one numpy float64 draw at `s = 32`, seed 0, `γ = 0.6`, on this CPU, an identity or counterexample check and not a statistic; the logit scale differs from the design's draw, so `ρ(Q)` reads `0.6927` here against `0.5134` there and no value is compared across the two draws. The `3.8e-07` finding is a DERIVED lower bound, not a reproduction of the design's run, which this planet could not execute (no script exists). The P-completeness item rests on `cook-1985-taxonomy` and the standard membership of rational linear systems in `DET`; no page of `merrill-2023-parallelism` was fetched this session and the sweep's `[U]` mark on that line is inherited. The `O(log² s)` depth is the recursive block-inversion count and no parallel-prefix kernel was timed. The Lean-grade remarks are about statements, not builds; nothing was compiled. No code file, no git write, no external fetch.
