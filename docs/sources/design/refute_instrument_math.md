# REFUTATION — the INSTRUMENT-FIRST design, mathematics lens

*MARS holding the MORIARTY role, 2026-09-03, against `$SCRATCH/design/design_instrument.md`
(525 lines, read in full first) and every input the dispatch named: `BRIEF.md`,
`THESIS_NOTES.md`, `THESIS_CORRECTIONS.md`, `THESIS_CORRECTIONS_2.md`, the six `sections/`
files and the eight `sweep/` files, `references.bib` and `bib_aliases.md`. Repository HEAD
`207e7b9`. Evidence classes: `RUN` (executed this session on this box, torch 2.5.1, float64,
CPU, python one-liners that computed and printed only — no file written); `READ path:line`;
`CITED [V] bibkey` (a canonical key of `references.bib`; the owning sweep fetched the page);
`DERIVED` (steps written out). Every KILL and REPAIR names a design line, a number, or a key,
and the `MISTAKES.md` mechanism it instantiates. Default under doubt is KILL; each such row
says what would reverse it.*

## 0. Reproduction of the design's §1 instance (so the attack is on the same object)

The design's §1 does not print its generator. The one-liner below re-derives every §1 number
that could be checked from `torch.manual_seed(0)`, `L = randn(32,32)`,
`P = softmax(L.masked_fill(triu(1), −inf), dim=1)`, absorbing rows `{0,30,5,17}` set to `e_i`,
`γ = 0.6` (`RUN`). Three of the design's numbers reproduce **to the last digit**, which fixes
the construction and licenses the counterexamples below as statements about the design's own
instance rather than about a different draw:

| design row | design value | `RUN` this session |
|---|---|---|
| R6 `ρ(Q)` with BOS undeclared (`READ design_instrument.md:58`) | `1.0`, `I − Q` singular | `1.0`; `torch.linalg.inv` raises *"diagonal element 1 is zero"* — the singularity is exact, not numerical |
| R7 `ρ(Q)` with `𝒜_0 = {0,30}` (`:59`) | `0.6874342901390162` | `0.6874342901390162`, and it **equals `max_{i∈T} P_ii`** (`0.6874342901390162`), see §2.3 |
| R7 `Σ_k q^{(k)}` on `T` (`:59`) | `[0.9999999999999993, 1.0]` | `[0.9999999999999993, 1.0]` |
| R8 `min_T max_k q^{(k)}` (`:60`) | `0.6051682711182041` | `0.6051682711182041` **with the goal set counted among the `K = 3`**; constraints-only `min_T max_{k≥1} = 0.0` |
| R11 `Σ_k q_γ^{(k)}` (`:63`) | `[0.13643076236928928, 0.339994435489312]` | `[0.1364307623692893, 0.3399944354893121]` — this is the **`z`-form** `(1−γ)·[(I−γP)^{-1}1_{𝒜_k}]_T = E[γ^τ 1_k]`; the **read** `O = (1−γ)P(I−γP)^{-1}1_{𝒜_k}` sums to `[0.2273846039488155, 0.5666573924821867]` and the ratio is exactly `γ = 0.6` on every position |
| R12 row sums (`:64`) | `2.5` | `(1−γ)`-form `[0.9999999999999998, 1.0000000000000004]`; bare form `2.5` |

## 1. Verdict table

| # | target (design line) | verdict | one-line reason | mechanism |
|---|---|---|---|---|
| 1 | R6 / F1 — BOS absorbing by construction (`:58`, `:66-71`) | **KEEP** | `RUN`: `P_00 = 1` exactly; `1 − Q_00 = 0.0`; DERIVED §2.1 shows `(I−Q)` is invertible **iff** `0 ∈ 𝒜` for a causal softmax with finite logits | V-25 |
| 2 | R7 `ρ(Q) < 1` census line (`:59`, `:122`) | **REPAIR** | for a lower-triangular `Q`, `ρ(Q) = max_{i∈T} P_ii` (`RUN`: both `0.6874342901390162`); the line is implied by row 1 and carries no information about absorption or mixing | V-10 |
| 3 | R8 and E2's planted negative "drop `𝒜_0` ⇒ `max_k q ≥ 1/K`" (`:60`, `:139-140`) | **KILL** | dropping `𝒜_0` drops BOS and `I − Q` is singular by row 1 — the negative cannot be run; the `0.605` was computed **with** the goal (pigeonhole on `Σ = 1`, V-3), and constraints-only `max_{k≥1}` reads `0.0` on this draw (`RUN`) | V-3, V-24 |
| 4 | R11 / E2 certificate "the reads are `E[γ^τ 1_k]`" (`:63`, `:153-156`) | **REPAIR** | the read is `E[γ^{τ−1} 1_k]` (DERIVED §2.4); the printed `[0.136, 0.340]` is the state `z`, the read sums to `[0.227, 0.567]`; the "delay share" `1 − Σ` changes by the factor `γ` | V-17 |
| 5 | R12 "min entry `0.0` confirms obstruction 4 reads 0" (`:64`) | **REPAIR** | the `0.0` comes from the upper triangle and the absorbing rows; on transient rows the lower-triangle minimum of `Π_γ` is `1.1114e-03 > 0` (`RUN`): absorbing rows change **weights, not support** in the softmax corner; C3's sentence "the support and weights" is half wrong | V-3, P-7 |
| 6 | Neumann tail equality on the row-stochastic class incl. absorbing rows (`:271-273`) | **KEEP** | `RUN` with `{0,30,5,17}` absorbing: `err − bound ∈ [4.4e-16, 6.7e-16]` at `K ∈ {1,2,4,8,16}`; DERIVED §2.5 | V-3 (declared, as the design does) |
| 7 | Row I bind (ii) "the shipped mask ... `6.13e-05` vs `7.6e-05`" (`:274-277`) | **REPAIR** | the measurement is a **truncation** (solve vs Neumann `K=16`), not a mask, and `‖V‖_∞ ≈ 5` is `[ASSUMED]` (`READ sections/sec_cost.md:84`); the bare `δ = 1.526e-05` **is exceeded** by `6.13e-05`, so the must-pass rests on an unprinted number | P-1, V-17 |
| 8 | Row M certificate "F1 union bound (dropped mass) + Neumann δ" and row I "F1 by a bounded-increment union bound" (`:392-395`, `:286-289`) | **KILL** | the dropped mass is amplified by the resolvent: `‖O_full − O_mask‖_∞ ≤ ε‖V‖_∞/(1−γ)`, not `ε‖V‖_∞`; counterexample §2.6: `s = 3`, `γ = 0.9`, `ε = 0.1` gives `0.5263` against the naive `0.1` (`5.26×`) | V-10, L-CERT |
| 9 | R1–R3, SM at an absorbing row (`:53-55`, `:168-171`) | **KEEP** (with the positivity proof added) | DERIVED §2.7: denominator `= (1 − γ p'_ii)/(1 − γ P_ii) > 0` for every causal row edit at `γ < 1`; `RUN` at the absorbing row 17 made non-absorbing: denominator `1.9 = (1−0.24)/0.4`, SM vs re-solve `1.45e-15` | — |
| 10 | R4 "`Δz[:i] = 4.44e-16` and `torch.equal = True`" (`:56`) | **REPAIR** | the two numbers come from two routes: `torch.linalg.solve` (LU, pivoting) gives `8.9e-16`, `equal False`; `solve_triangular` gives `0.0`, `equal True` (`RUN`); the design must name the route or the `0.0` is a D-5 declaration | D-5, P-1 |
| 11 | G1 planted negative "a feedback plant ... Dash Thm 1's EMC violation" (`:172-175`) | **KILL** | EMC cannot fail inside the shape: for any `γ < 1` the fixed point of `z = V + γPz` is unique, so settle-then-`do` equals `do`-then-solve for every `P`, causal or not — `RUN` on a dense non-causal `P` with feedback: `1.33e-15`; what the plant *does* show (`Δz[:i] = 0.0868 ≠ 0`) is loss of triangularity, not Dash's theorem; a non-causal `P` is also outside a causal LM read | V-24, D-7, P-10 |
| 12 | E1 planted negative (iii) "a non-identity absorbing row must break `Σ_k q = 1` (R7's row) by O(1)" (`:93-94`) | **KILL** | `B = NR` never reads the absorbing rows: `Σ_k q = N(1 − Q1) = 1` whenever the **transient** rows are stochastic; `RUN`: row 5 replaced by `(0.5 e_5 + 0.5 e_2)` — `Σ_k q ∈ [0.9999999999999993, 1.0]`, `max|q − q_identity| = 0.0`; row 5 replaced by `1.5 e_5` — still `[0.9999999999999993, 1.0]`. The rejection region is empty. The plant fires only on the full-`P` read at `γ < 1` (`0.1491` at `k = 5`, `RUN`) | V-24, V-3 |
| 13 | E1 bind (i) "bitwise `torch.equal` with ChaCAL at `𝒜 = ∅`" (`:88-91`) | **REPAIR** | bitwise across two implementations needs identical op order; the record's own softmax corner is `1.110223e-16` off `ceq/lm.py` on `19/64` entries (`READ design_instrument.md:298-300`); ChaCAL's `A` is reported with the **diagonal removed** (`READ sweep/sweep_expressivity.md:886`, `[U]`), which is regime N, not the design's regime-S `P` — the bind must state the diagonal convention and be bitwise only against the lane's own routine | V-3, C8 |
| 14 | K planted negative "a gate at `1e-300` (nonzero) must leave the block non-zero" (`:328`) | **REPAIR** | float64 underflow: `1e-300 · 2^{−80} = 0.0` and `1e-300 · 1e-300 = 0.0` (`RUN`), so a path of ≥ 80 gates at `0.5` (or two `1e-300` gates) annihilates the block exactly and the "nonzero" plant reads F0; the `−30` logit plant is safe (`e^{−30} = 9.36e-14`, exact zero only below `−745`: `exp(−746) = 0.0`) | V-2, V-16 |
| 15 | Segmentation block-diagonality with absorbing rows (`:321-327`) | **KEEP** | an identity row `e_a` respects every cut `(j < c ≤ i ⇒ P_ij = 0)`; `resolvent_fromBlocks` needs only `IsUnit` of the diagonal blocks, which `diag(I − γP) ≥ 1 − γ > 0` gives (`READ sections/sec_proved.md:560-561`) | — |
| 16 | H1 certificate "row I's `δ`; at `γ ↑ 1` the factor `1/(1−γ)` is printed" (`:237-238`) | **KILL** | the committor is read at `γ = 1` where row I's `δ = γ^{K+1}/(1−γ)` is `∞`; the sup-norm substitute uses `‖Q‖_∞ = 0.9580` and reads `19.2` at `K = 4` against a true error `2.5e-03` (`RUN`) — vacuous; the only certificate is the Perron-weighted one of `sec_proved.md` §2.8 (3)(b) `[S]`, `NOT MEASURED` | V-17, V-10, L-CERT |
| 17 | H2 certificate "admitted only if `δ_thr − δ·‖V‖_∞ > 0`" (`:261-262`) | **REPAIR** | the inequality never compares `q̂` to anything; the certified rule is `max_k q̂^{(k)} + δ·‖V‖_∞ ≤ δ_thr` (upper end of the certified interval below the threshold), with `‖V‖_∞ = 1` for indicator values and `δ` from row 16, not row I | V-17 |
| 18 | E2 / H2 floor "zero-hop Fano `1 − ln 2/ln m`; `m = 2` refused" (`:151-152`, `:260`) | **REPAIR** | at `I = 0` the exact floor is `1 − max_a π(a) = 1 − 1/m`: `0.75 / 0.875 / 0.9375` at `m = 4/8/16` (`RUN`), and Fano's `0.5 / 0.6667 / 0.75` is slack by `0.25 / 0.208 / 0.1875`; `m = 2` has an exact floor `0.5` and need not be refused; the `ln m` denominator is the weak Cover–Thomas form, `ln(m−1)` gives `0.7124` at `m = 8` | V-10, L-FLOOR |
| 19 | E1 kill's MDE cell `0.039827` at sd `0.034451`, `n = 8` (`:127-128`) | **KEEP** | `RUN` (noncentral `t`, paired, two-sided `α = 0.05`, power `0.80`): `0.0398266`, `ncp = 3.2698`; `n = 16`: `0.0258198` — both match `sections/sec_beds.md:715` | — |
| 20 | §6 "TOST at `N = 70`" priced at `70/8 × 34 s` (`:489-491`, `:130`) | **REPAIR** | `70` is the **two-sample** figure (`RUN`: power `0.7975` at `69`, `0.8073` at `70`, half-width `0.2799σ`); the arena is **paired on byte-identical draws**, where the margin is in units of `sd_d` and power `0.80` is reached at `N = 36` if `sd_d = σ` (`0.790` at `35`, `0.856` at `40`, `RUN`) — `N = 70` only if pairing buys nothing (`sd_d = √2σ`); the price is `2×` over in the safe direction, but the requirement is imported out of its units | V-17, M-13, P-8 |
| 21 | J metric `Λ ≤ 3.841` PINNED (`:307-309`) with `γ ∈ [0,1)` and kill `|γ̂| < 0.05` (`:315`) | **REPAIR** | if `γ` is constrained to `[0,1)`, `γ = 0` is a boundary and the null of the LR statistic is `½χ²_0 + ½χ²_1`, whose `95 %` point is `2.706`, not `3.841` (`RUN`); the `|γ̂|` in the kill implies a signed `γ`, for which the Neumann **equality** of row 6 fails (signed terms) and only the `≤` bound survives; the design must say which parameterisation it trains | V-17, V-25 |
| 22 | Row L on BED-S: "the `ε = 0` endpoint must equal the F0 segmentation count" (`:352-353`, `:359`) | **REPAIR** | on BED-S in the softmax corner `Π_ij > 0` for every `j ≤ i` on transient rows (row 5), so the `ε = 0` influence graph is one weak component and `β₀ = 1` on every draw — the row's own census line (`:369-370`) blocks it; the row needs BED-M's F0 zeros or an explicit F0 mask on BED-S | V-25, M-15 |
| 23 | Transitive-closure support (`THESIS_NOTES.md` P5, inherited at `:64` C3) | **REPAIR** | closure of the full lower triangle is the full lower triangle; with absorbing rows the transient-row support is unchanged (row 5); the "every hop at once" support claim is empty in regime S and bites only under F0 zeros | V-3 |
| 24 | DET-class / "O(log s) parallel rounds by prefix doubling" (`THESIS_CORRECTIONS.md` C4, inherited by §4 rank 8 and §5 C10) | **REPAIR** | `Π_{k<6}(I + A^{2^k}) = (I − A)^{-1}` to `8.9e-16` for strictly-lower `A` (regime N, exact); with a diagonal `0.3·I` the same product is a truncation with residual `1.07e-14 ≈ 0.3^{64}/0.7` (`RUN`): in regime S the doubling route is a certified Neumann truncation at `K = 2^k − 1`, not an exact solve, and the circuit depth is `O(log² s)` (log `s` matmuls of depth `log s`) with `O(s³ log s)` work | P-8, V-17 |
| 25 | §1 R11's "reach-avoid ... `E[γ^τ 1_k]`" vs `CITED [V] fisac-2019-bridging` (`:63`) | **KEEP** the citation, **REPAIR** the object | Fisac's discounted safety value is the read-form; see row 4 | P-10 |
| 26 | E1/H1 identity bind "on the bed's real draw" (`:87`, `:223-226`) | **REPAIR** | R6/R7/R10/F2 are facts about the **arm's causal `P`** (walks descend in position); BED-S's label is the committor of a **graph** chain `P_env` (`READ sections/sec_beds.md:557-562`) that is not causal and can revisit nodes; the bind must say which chain each identity is checked on, or it conflates the arm's operator with the environment's | D-2, C8 |
| 27 | Bib keys `sherman-1950-inverse` (`:55`, `:170`), `kim-2026-topological-causal-effects` (`:361`), `peng-2024-limitations` (`:476`) | **REPAIR** | none is a canonical key (`RUN grep` on `references.bib`: `0` entries each); the canonical keys are `sherman-1950-inverse-adjustment`, `kim-2026-topological-causal`, `peng-2024-transformer-limitations` (`READ bib_aliases.md`) | P-5 |
| 28 | §6 arena price "seven arms ≈ 4 GPU-min" (`:500-502`) | **REPAIR** | `7 × 8 × 1.680 s + 4.0 s ≈ 98 s ≈ 1.6 GPU-min` on the softmax corner (DERIVED from `:485`); the skylines at depth `5` and the corner-3 base are unpriced in that line; the `4` is a `2.4×` over-book in the safe direction | P-8 |
| 29 | `|T|³/3 ≈ 5.8e8 flops` (`:486`) | **REPAIR** | `1200³/3 = 5.76e8` **MACs**; house convention MAC = 2 FLOPs (`READ sections/sec_cost.md:3`) makes it `1.15e9` FLOPs; and `m·K` solves is `m` factorisations with `K` right-hand sides each | V-17, M-8 |
| 30 | R1 "`V ≡ 1 ⇒ Δz ≡ 0` bitwise" as a bind (`:53`, `:168`) | **KEEP** with a tolerance clause | `RUN` at `(s,γ) = (32,0.6), (256,0.9), (1024,0.95)`: `max|Δz| = 0.0` on both `solve` and `solve_triangular` routes, while `max|z − 1/(1−γ)|` is `8.9e-16 / 1.4e-14 / 2.1e-14` — the bitwise pass rests on rounding of `z` to the same neighbour on both sides and is not guaranteed; a `≤ 4 ulp·|z|` clause keeps the bind from failing on noise | D-5 |

## 2. Derivations and counterexamples (each with steps)

### 2.1 Invertibility of `I − Q` and `I − γP` (rows 1, 15) — DERIVED

`P` causal (`P_ij = 0` for `j > i`), softmax rows with finite logits, so `P_ij > 0` for all `j ≤ i`
and `P_ii < 1` for `i ≥ 1`; `P_00 = 1` because position `0` has one key. Absorbing rows are
`e_a`. `I − γP` is lower-triangular with diagonal `1 − γP_ii ≥ 1 − γ > 0` for `γ < 1`, hence
invertible with or without absorbing rows (the determinant is the product of the diagonal).
`Q = P_TT` is lower-triangular in position order, so `det(I − Q) = Π_{i∈T}(1 − P_ii)`, which is
zero iff some transient `i` has `P_ii = 1`, i.e. iff `0 ∈ T`. Hence `(I − Q)` is invertible **iff**
`0 ∈ 𝒜`. `RUN`: with `0 ∉ 𝒜`, `1 − Q_00 = 0.0` and `torch.linalg.inv` raises. At `γ = 1`, row `0`
makes `I − P` singular on every causal softmax; the design's `γ < 1` read never meets it.

### 2.2 The reach-avoid sum and absorption (rows 3, 12) — DERIVED

With `Q1 + R1 = 1` on transient rows (stochastic `P`) and `N = (I − Q)^{-1}`:
`Σ_k q^{(k)} = N R 1 = N(1 − Q1) = N1 − NQ1 = N1 − (N − I)1 = 1`. The identity uses only the
transient rows, so no change to an absorbing row can move it (row 12, `RUN` `0.0`). Absorption
is almost sure on a causal `P` because the walk never moves up in position and every transient
row has `P_i0 > 0` toward the declared BOS; that is why `CITED [V] grinstead-1997-probability`
Thm 11.6 applies, and why it stops applying the moment BOS is undeclared (row 1). The
pigeonhole `max_k q^{(k)} ≥ 1/(K+1)` over the `K+1` sets **including the goal** is what R8
measured (`0.605 ≥ 1/3`); the V-12 degeneracy is the statement about constraints **only**, and
on the design's own draw `min_T max_{k≥1} q^{(k)} = 0.0` (`RUN`, positions below every
constraint), so the degeneracy is not exhibited by R8 at all.

### 2.3 `ρ(Q)` on a causal chain (row 2) — DERIVED + RUN

A lower-triangular matrix has its eigenvalues on the diagonal, so `ρ(Q) = max_{i∈T} P_ii`. `RUN`:
`0.6874342901390162` for both. The census line "`ρ(Q) < 1` on `100 %`" is therefore equivalent
to "no transient position attends only to itself", which finite logits plus row 1 already give.
It is a V-10 gate: satisfied by construction once BOS is declared. The number that governs the
committor's hop ladder is not `ρ(Q)` either — `‖Q‖_∞ = 0.9580` on this draw (`RUN`) — see §2.8.

### 2.4 The discounted read is `E[γ^{τ−1} 1_k]` (row 4) — DERIVED + RUN

For `i ∈ T`, `z_i = Σ_t γ^t P_i(X_t ∈ 𝒜_k) = Σ_u P_i(τ = u, k) Σ_{t ≥ u} γ^t = E_i[γ^τ 1_k]/(1−γ)`
(absorbing set, so `X_t ∈ 𝒜_k ⇔ τ ≤ t` and absorbed in `k`). The read
`O_i = (1−γ)Σ_j P_ij z_j = Σ_j P_ij E_j[γ^τ 1_k] = E_i[γ^{τ−1} 1_k]` by first-step analysis
(`τ ≥ 1` from `T`). Equivalently `(I − γQ)^{-1} R_k 1 = Σ_t γ^t Q^t R_k 1 = Σ_t γ^t P(τ = t+1, k)`.
`RUN`: read-form `Σ_k ∈ [0.2274, 0.5667]`, `z`-form `Σ_k ∈ [0.1364, 0.3400]`, ratio `0.6000` on
every transient position, and the full-`P` read `(1−γ)P(I−γP)^{-1}1_{𝒜_k}` equals the read-form
to `1.67e-16`. The design's R11 printed the `z`-form and called it "the reads".
`CITED [V] fisac-2019-bridging` names the read-form quantity; the paper must print
`1 − Σ_k E[γ^{τ−1} 1_k]` as the delay share, not `1 − γ·(that)`.

### 2.5 Neumann tail — equality, and where it is lost (rows 6, 21) — DERIVED + RUN

Tail `= Σ_{k>K}(γP)^k`. For `P ≥ 0` row-stochastic (identity rows included) every term is
non-negative with row sum `γ^k`, so the `∞`-norm equals `γ^{K+1}/(1−γ)`. `RUN` with
`{0,30,5,17}` absorbing at `γ = 0.6`: `err − bound = 4.4e-16, 4.4e-16, 6.7e-16, 5.0e-16, 6.4e-16`
at `K = 1, 2, 4, 8, 16`. For a signed `P` with `‖P‖_∞ ≤ 1` the triangle inequality gives only
`≤`; for a signed `γ` the same. For `‖P‖_∞ = 1.5` at `K = 2`: `err 7.29` vs bound `0.54`
(`RUN`; the design's `119.37 / 1.143` is the same failure at `γ = 0.7`, `s = 64`).

### 2.6 An F1 mask is amplified by the resolvent (row 8) — DERIVED + RUN

Let `P_m` be the masked operator, `ε = ‖P − P_m‖_∞` (dropped row mass), `M = (I−γP)^{-1}`,
`M_m = (I−γP_m)^{-1}`, `‖M‖_∞, ‖M_m‖_∞ ≤ 1/(1−γ)` (sub-stochastic non-negative). Then
`PM − P_mM_m = (P − P_m)M + P_m(M − M_m)` and `M − M_m = γM(P − P_m)M_m`, so
`‖PM − P_mM_m‖_∞ ≤ ε/(1−γ) + γε/(1−γ)² = ε/(1−γ)²`, and for the `(1−γ)`-normalised read
`‖O_full − O_mask‖_∞ ≤ ε‖V‖_∞/(1−γ)`. The bound is attained up to a constant: `s = 3`,
`P = [[1,0,0],[ε,1−ε,0],[0,1,0]]` (causal, row-stochastic), mask drops `P_10`, `V = e_0`,
`γ = 0.9`, `ε = 0.1`: `|O_full − O_mask|_∞ = 0.5263157894736843` (`RUN`; closed form
`ε + (1−ε)γε/(1−γ+γε)`), against the naive dropped-mass certificate `0.1` — a `5.26×`
violation — and inside the derived `1.0`. The renormalised masked softmax (row 1 becomes
absorbing) gives the same `0.5263`. A random `s = 32` draw did **not** violate the naive bound
(ratios `0.21`, `0.16` at `γ = 0.6, 0.9`, `RUN`), which is why a bind on random draws would
pass the wrong certificate (V-10). Every "dropped mass + Neumann δ" line in rows I and M must
carry the `1/(1−γ)` factor. What would reverse this KILL: a proof that BED-S's masks only ever
drop mass on rows whose downstream `z` is bounded independently of `γ` — no such structure is
declared.

### 2.7 Sherman–Morrison at a causal row edit (row 9) — DERIVED + RUN

`P' = P + e_i uᵀ`, `u` supported on `j ≤ i`. `M e_i` is supported on rows `≥ i` (lower-triangular),
so `uᵀ M e_i = u_i M_ii = u_i/(1 − γP_ii)` and the denominator is
`1 − γu_i/(1−γP_ii) = (1 − γP_ii − γu_i)/(1−γP_ii) = (1 − γp'_ii)/(1 − γP_ii) > 0` for `γ < 1`,
whatever the row was — absorbing (`P_ii = 1`, giving `(1−γp'_ii)/(1−γ)`) or not. `RUN`, row
`17` from `e_17` to `0.4e_17 + 0.6e_3`: denominator `1.9`, closed form `(1 − 0.24)/0.4 = 1.9`,
`max|Δz_SM − Δz_resolve| = 1.45e-15`. The formula the design quotes
(`CITED [V] sherman-1950-inverse-adjustment`, `hager-1989-updating`) is right; the positivity
of its denominator was not stated and is what makes the `O(sd)` price unconditional.

### 2.8 The committor has no row-I certificate (row 16) — DERIVED + RUN

The committor is `(I − Q)^{-1}R_k1` at `γ = 1`. Its Neumann tail is `Σ_{t>K} Q^t R_k 1`; the
sup-norm bound `‖Q‖_∞^{K+1}/(1 − ‖Q‖_∞)` needs `‖Q‖_∞ < 1`, which holds (`P_i0 > 0` on every
transient row toward the declared BOS) but is useless: `RUN` `‖Q‖_∞ = 0.9580`, bound `19.24` at
`K = 4`, `5.80` at `K = 32`, while the true error is `2.5e-03` and `5.6e-17`. The equality of
§2.5 is lost because `Q` is not stochastic. The certificate that exists is the weighted Perron
form (`lean/CEQ/Contraction.lean` `PerronCertificate`, `READ sections/sec_proved.md:458-459`),
with `w` to be found and `ρ = ρ(Q) = max P_ii`; `ρ^{K+1}/(1−ρ)` is not a bound without `w`
(it reads `0.49` at `K = 4` here, which happens to dominate, `RUN`, but is not a theorem).
Until that `w` is computed the H1 and H2 certificate cells are `NOT MEASURED`.

### 2.9 EMC holds identically inside the shape (row 11) — DERIVED + RUN

"Settle then `do`": compute `z = (I−γP)^{-1}V`, replace row `i` of `P` by `p'_i`, iterate
`z ← V + γP'z` from `z`. The iteration is a `γ`-contraction in `‖·‖_∞` for any row-stochastic
`P'` (`CITED [V]` via `Contraction.rowStochastic_perron`, `READ sections/sec_proved.md:126-127`),
so it converges to the **unique** fixed point `(I−γP')^{-1}V` — which is "`do` then solve".
`RUN` on a dense (non-causal, feedback everywhere) row-stochastic `P` with two absorbing rows,
row `9` re-targeted to attend to `20 > 9`: `|settle-then-do − do-then-solve|_∞ = 1.33e-15`.
Dash's Theorem 1 (`CITED [V] dash-2005-emc`) needs a dynamic system whose equilibrium
manifold is not the SCM's; a linear contraction solved exactly has no such gap. The design's
"feedback plant" therefore has an empty rejection region for the EMC bind (V-24) and cannot
file "both halves" (D-7). What the plant does exhibit is `Δz[:i] = 0.0868 ≠ 0` — the loss of
forward-only propagation when `P` is not triangular — which is a **triangularity** bind with a
non-empty rejection region and should be filed under that name. What would reverse this: a
`P` that depends on `z` (a DEQ), which `THESIS_NOTES.md` P7 puts out of scope.

### 2.10 Regime N versus regime S in the design's own cells (rows 13, 24, 26) — READ + DERIVED

The design's §1 builds `P` **with** the diagonal (regime S). Three places lean on regime-N
facts: (a) E1's bitwise parity with ChaCAL, whose `A` is reported diagonal-free
(`READ sweep/sweep_expressivity.md:886`, `[U]`) — a regime-N operator whose row `0` has no key
and needs a sink; (b) the prefix-doubling exactness (`RUN` §1 row 24) which is exact only for
nilpotent `A`; (c) the E1/H1 "identity bind on the bed's real draw", where the label's chain
`P_env` is a graph walk (`READ sections/sec_beds.md:557-562`) that revisits nodes, while R6/R10
are theorems about the arm's descending walk. None of the three is wrong; each must say which
regime and which chain, or the bind checks an object other than the one it names (C8, D-2).

### 2.11 Floors and the LR threshold (rows 18, 21) — RUN

`1 − ln 2/ln m = 0.5000 / 0.6667 / 0.7500` at `m = 4/8/16` reproduces the design; the exact
`I = 0` floor `1 − 1/m = 0.7500 / 0.8750 / 0.9375` dominates it; the record's M11 instances
`0.1858` (`m = 8, I = 1`) and `0.2229` (`m = 32, I = 2`) reproduce (`CITED [V] cover-2006-elements`
§2.10 weak form). `χ²_1(0.95) = 3.8415`; the boundary null's `95 %` point is `2.7055`;
`ln 4096 = 8.318` so the interval verdict exists at the arena's `n_eval`, but for `n < 47`
(`ln 46 = 3.829`) the MOVED threshold falls below the PINNED one and the verdict grammar inverts.
A citation for the boundary null (Chernoff 1954 / Self–Liang 1987) is not in `references.bib`
(`RUN grep`: `0`) and was not fetched this session; it is owed, not cited.

### 2.12 What was checked and found sound (KEEPs, one line each)

R6 and F1 (row 1); the tail equality with absorbing rows (row 6); block-diagonality under a cut
with absorbing rows (row 15); SM validity at every causal edit (row 9); the MDE cell `0.039827`
(row 19) and its `n = 16` sibling `0.025820`; R1's bitwise zero on three geometries (row 30);
the C6 identity `Δz = (I−γP')^{-1}(ΔV + γΔPz)` (`RUN` residual `1.45e-15` on the §0 instance,
`Δz_C6[:i] = 0.0` exactly); the two-sample TOST figures `0.8807 / 0.6001 / 0.4955 / 0.2799` and
power `0.7975 → 0.8073` at `N = 69 → 70`.

## 3. Fatal findings (the ones that change the apparatus, not a sentence)

1. **Row 8** — the F1 mask certificate is wrong by `1/(1−γ)`; at ChaCAL's `γ = 0.9` a compliant
   mask can exceed its printed δ by `10×` and the `1,024`-draw must-pass would read as a
   violation, or a wrong δ would pass on random draws. Every L-CERT line in rows I and M rests
   on it.
2. **Row 12 + row 3 + row 11** — three of the design's planted negatives cannot fire (E1 (iii),
   E2 "drop `𝒜_0`", G1 EMC): the design's answer to V-24 is itself V-24 on three rows.
3. **Row 16** — the committor rows H1/H2 inherit a certificate that is `∞` at their `γ`; the
   safety-filter threshold of H2 is uncertified until the Perron weight exists.

## 4. Limits (collected once)

Every `RUN` here is one draw (seed 0, `s = 32`, `d = 8`, `γ = 0.6`, float64, CPU) plus the
named side instances (`s = 3` closed form; `s ∈ {256, 1024}` for R1; a dense `s = 32` chain for
EMC); none is a statistic. The §0 reproduction fixes the design's generator by matching three
16-digit numbers, not by reading its command, which the design did not print (P-1 on the
design's side). The ChaCAL diagonal convention is `[U]` in the owning sweep and is used here
only to demand that the design state its own. The paired-TOST `N = 36` is a Monte-Carlo figure
(`200,000` draws, `rng(0)`) at margin `0.5·sd_d`; the arena's realised `sd_d` is `NOT MEASURED`,
so the number is the form of the repair, not its value. No Lean was built; the `[M]/[S]` grades
cited are `sections/sec_proved.md`'s. No bib entry was fetched this session; the one owed
citation (boundary LR null) is named as owed. No code file, no git write.
