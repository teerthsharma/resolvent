# PLAN — JUPITER (MYCROFT): the mathematics roadmap

*2026-09-03, HEAD `207e7b9`, branch `v17k-gate0`. Every Lean target of `judge/sec_shape.md` §4.6 and `judge/sec_apparatus.md` §A.8.2 as a milestone card in build order, then the paper derivations owed. Format is `THESIS_CORRECTIONS_2.md` §0's, binding: one developer, alone, in evenings, on the certified RTX 4060, in any order the DAG allows; every card carries id, title, phase, prerequisites, independent-of, what to build, what to prove, what to measure, PASS, KILL, price, mechanism, deliverable, evenings. Evidence classes: `RUN[P]` — this planet's numpy float64 one-liner (seed 0, `s = 32`, `γ = 0.6`, causal softmax `P` with diagonal, sink `{0}`, goal `{5}`, constraints `{9,10}`, `{15}`; printed, no file written); `RUN[J]/[M]/[I]/[F]/[coord]` — other planets' runs as their files record them; `READ path:line`; `CITED [V]/[U]` by canonical `references.bib` key; `DERIVED` with steps. Nothing here is a result; every card is a specification (no `.py`, `.lean`, `.sh` is written by this file — L-LEAN: no arm trains before its identity theorems are green, `READ CEQ_V15_CONTRACT.md:57-58`).*

## 0. What this session checked before grading (the P-11 discharge)

The vendored Mathlib under `lean/.lake/packages/mathlib` is at rev `a45ae63747140c1b2cbad9d46f518015c047047a` (2024-04-04, `RUN[P] git log -1`), toolchain `leanprover/lean4:v4.7.0` (`READ lean/lean-toolchain`). Every Mathlib name the `[M]` cards below rest on was found by `grep` in that tree this session (`RUN[P]`), so the judge's "`[M]`→`[S]` until the `[U]` name is confirmed" (`judge/sec_apparatus.md` §A.8.2 rows 5–6) is discharged for these eight and only these:

| Mathlib declaration | file:line | what it gives a card |
|---|---|---|
| `Matrix.det_of_lowerTriangular (M) (h : M.BlockTriangular toDual) : M.det = ∏ i, M i i` | `Mathlib/LinearAlgebra/Matrix/Block.lean:265` | J-L3 (unit from a nonzero diagonal) |
| `Matrix.isUnit_iff_isUnit_det : IsUnit A ↔ IsUnit A.det` | `Mathlib/LinearAlgebra/Matrix/NonsingularInverse.lean:151` | J-L3 |
| `Matrix.blockTriangular_inv_of_blockTriangular [Invertible M]` | `Mathlib/LinearAlgebra/Matrix/Block.lean:346` | J-L4 (the inverse of a lower-triangular unit is lower-triangular) |
| `Matrix.inv_fromBlocks_zero₂₁_of_isUnit_iff (A B D) (hAD : IsUnit A ↔ IsUnit D) : (fromBlocks A B 0 D)⁻¹ = fromBlocks A⁻¹ (−(A⁻¹*B*D⁻¹)) 0 D⁻¹` | `Mathlib/LinearAlgebra/Matrix/SchurComplement.lean:202` | J-L6 (block inverse; the shape's cut has `B = 0` too) |
| `Matrix.inv_eq_left_inv (h : B * A = 1) : A⁻¹ = B`; `nonsing_inv_mul (h : IsUnit A.det)` | `NonsingularInverse.lean:527, :267` | J-L10, J-L13 |
| `mul_neg_geom_sum [Ring α] (x) (n) : (1 − x) * ∑ i in range n, x^i = 1 − x^n` | `Mathlib/Algebra/GeomSum.lean:247` | J-L12 (the tree's `occupancy_telescope` is this over a ring) |
| `tsum_geometric_of_lt_one (h₁ : 0 ≤ r) (h₂ : r < 1) : ∑' n, r^n = (1 − r)⁻¹` | `Mathlib/Analysis/SpecificLimits/Basic.lean:267` | J-L12 (the tail's scalar value) |
| `Matrix.linfty_opNorm_def`, `linfty_opNorm_mul`, `linfty_opNorm_mulVec` | `Mathlib/Analysis/Matrix.lean:268, :375, :392` | J-L12 (only if the norm form is chosen; the pointwise form of `Contraction.lean` avoids the instance) |

Absent from this Mathlib (`RUN[P] grep`, `0` files): any `hittingTime` object; the stopping-time machinery that exists (`IsStoppingTime`, `stoppedValue`, `Mathlib/Probability/Martingale/Basic.lean:508-510`) is for filtrations of a measure space, not for a finite chain's first-passage law. `hitting_time_transform` therefore stays `[D]` (J-L17) and its content is carried as a paper derivation (J-D3). The tree's own declarations that the cards extend were re-read verbatim this session: `CEQ.Contraction.PerronCertificate` / `RowStochastic` / `weighted_contraction` / `rowStochastic_perron` (`READ lean/CEQ/Contraction.lean:59-62, :72-75, :103-125`), `CEQ.Occupancy.occupancy` / `occupancy_telescope` / `occupancy_eq_inverse_of_nilpotent` (`READ Occupancy.lean:42, :52, :83`), `CEQ.Nilpotent.StrictlyLower` / `pow_entry_zero` / `pow_card_eq_zero` / `occupancy_is_exact_inverse` / `one_not_nilpotent` (`READ Nilpotent.lean:46-47, :52, :77, :96, :105`), `CEQ.V16Domain.pathProd` / `pathProd_eq_zero_iff` / `three_corners_containment` (`READ V16Domain.lean:97, :129, :433`), `CEQ.OracleSeparation.Nonneg` / `SymmSupport` / `not_isNilpotent` (`READ OracleSeparation.lean:76, :82, :148`). No declaration named `segment*`, `gamma_zero*`, `committor*` or `displacement*` exists in `lean/CEQ/` (`RUN[P] grep`, one prose hit at `V16Domain.lean:53`) — the contract's `#18/#19/#22 [M]` remain P-11 in potential form until J-L6/J-L10 build.

`RUN[P]` numbers used below (one draw, float64): Neumann tail `‖(I−γP)^{-1} − Σ_{k≤K}(γP)^k‖_∞` equals `γ^{K+1}/(1−γ)` to the printed digits at `(γ,K) = (0.6,2): 0.540000`, `(0.7,4): 0.5602333`, `(0.7,16): 7.754350e-3`; the planted rows-`1.5` matrix at `γ = 0.6, K = 2` reads `7.2900` against bound `0.5400` (reproduces `RUN[M]`); with sink `{0}`, goal `{5}`, constraints `{9,10}`, `{15}` the four reads sum on `T` to `[1.000000000000000, 1.000000000000001]`; `ρ(Q) = 0.692660 = max_{i∈T} P_ii` (reproduces `RUN[J]`); BOS undeclared: `ρ(Q) = 1.000000`, `det(I−Q) = 0.000e+00`; the displacement identity residual on a row clamp at `i = 12` is `2.897e-16` and `max|Δz_{<12}| = 0.000e+00`; with BOS placed *inside* a constraint set and no goal (`{0,9,10}`, `{5,15}`) the two committors sum to `1.000000000000000` on `T` and `min_T max_k q^{(k)} = 0.548718 ≥ 1/2` (the degeneracy lemma's instance); `χ²₁` quantiles `0.90 → 2.705543`, `0.95 → 3.841459` (the boundary null of J-D9).

## 1. The DAG at a glance

```
J-L0 ──┬─ J-L1  J-L2  J-L5  J-L8  J-L9          (independent [M] one-liners)
       ├─ J-L3 ─┬─ J-L4 ─┬─ J-L14
       │        ├─ J-L10 ─┬─ J-L15 ─ J-L16(c)
       │        └─ J-L13 ─┘
       ├─ J-L6 ─── J-L7
       ├─ J-L11
       ├─ J-L12 ─── J-L13(mixing)            J-L16(masked) needs J-L12
       └─ J-L17 [D]
J-L18 (build gate + census + #print axioms) needs every [M] card it certifies
J-D1..J-D9 (derivations) need only J-L0's notation; J-D2 needs J-L15's statement, J-D3 needs J-L10
```

Critical path (serial evenings): J-L0 → J-L3 → J-L10 → J-L13 → J-L12 → J-L15 → J-L16 → J-L18 = `1+1+1+2+2+2+1+1 = 11`. Everything else runs beside it. Total price of this roadmap: `0 GPU-s`; the numeric instances are CPU float64 one-liners (seconds). The five cheapest decisive items are marked ★.

## 2. Lean milestones, in build order

### J-L0 ★ — Lay the `Shape` module and run the Mathlib name census
- **phase** P0 · **prerequisites** none · **independent-of** every other card
- **what to build.** A specification for one new import root `lean/CEQ/Shape.lean` importing `CEQ.Contraction`, `CEQ.Occupancy`, `CEQ.Nilpotent`, `CEQ.V16Domain`, `CEQ.OracleSeparation`, and files `Shape/Parity.lean`, `Shape/Triangular.lean`, `Shape/Boundary.lean`, `Shape/Segment.lean`, `Shape/Displacement.lean`, `Shape/Neumann.lean`, `Shape/Committor.lean` (one file per card group below; no file may exceed the `V15Source.lean` scale, 201 lines / 7 declarations, `READ sec_proved.md` §2.0). Shared definitions, stated once: `LowerTri (M) := ∀ i j, (i:ℕ) < (j:ℕ) → M i j = 0` (the non-strict form; `StrictlyLower` of `Nilpotent.lean:46` is its strengthening); `Absorbing (P) (𝒜 : Finset (Fin n)) := ∀ a ∈ 𝒜, ∀ j, P a j = if j = a then 1 else 0`; `IsSink 0`. Journal fields for every card: declaration name, file:line, grade, `#print axioms` output, `lake build` exit code, Mathlib rev, the census line (§J-L18).
- **what to prove.** Nothing; the census is a `grep` over the vendored tree for the eight names of §0 plus any name a later card introduces, recorded as `[V-name]` with file:line or `[U]`.
- **what to measure.** `lake build` exit `0` on the empty module (the warm cache reads `[1530/1531]`, `READ workdonenewseal.md:41-45`).
- **PASS** all eight names resolve at the pinned rev (they do, §0). **KILL** a name missing ⇒ the dependent card is `[S]` and says which lemma it must supply itself (P-11).
- **price** `0 GPU-s` · **mechanism** P-11 (a contract citing its own `[M]` tag as settled), P-4 (claimed scaffolding that does not exist) · **deliverable** `docs/LEAN_SHAPE_MANIFEST.md` (the census table) · **evenings** 1

### J-L1 ★ — Prove parity at `γ = 0` with its refusal
- **phase** P0 · **prereq** J-L0 · **independent-of** J-L2…J-L17
- **what to build.** In `Shape/Parity.lean`:
```
theorem gamma_zero_is_softmax (P V : Matrix (Fin n) (Fin n) ℝ) :
    P * (1 - (0:ℝ) • P)⁻¹ * V = P * V            -- simp [zero_smul, sub_zero, Matrix.inv_one]
theorem gamma_half_is_not_softmax :                 -- refusal, a witness
    ∃ P V : Matrix (Fin 2) (Fin 2) ℝ, RowStochastic P ∧ LowerTri P ∧
      P * (1 - (1/2:ℝ) • P)⁻¹ * V ≠ P * V          -- P = !![1,0; 1/2,1/2], V = !![1,0;0,0] or any V with V₀ ≠ V₁
```
The corner identification is the tree's `V16Domain.three_corners_containment` clause 1 (`:433`); this card adds only the `γ` clause (Proposition 1 of `judge/sec_shape.md`).
- **what to prove.** Grade `[M]`, Mathlib objects: `Matrix.inv_one`, `zero_smul`, `sub_zero`; the witness by `decide`/`norm_num` on `Fin 2`.
- **what to measure.** The float instance stays RUN-class: `torch.equal(O(0), P V) = True` at `s = 64, d = 16`, rejection region `max|O(0.5) − O(0)| = 2.3002850040264393` (`RUN[coord]`); the record's own softmax corner is `1.110223e-16` off `ceq/lm.py` on `19/64` entries (`READ V16_ARM_SMPRIME.md:266-293`), so "bitwise" is against the lane's `softmaxAttn`, never the fused kernel.
- **PASS** both declarations build, axioms `[propext, Classical.choice, Quot.sound]`. **KILL** the witness fails to close ⇒ the parity proposition ships without a Lean rejection region and the paper's Proposition 1 reads RUN-only (V-24 in Lean form).
- **licenses if green:** the sentence "CEQ contains softmax at `γ = 0` by theorem" (`fagnou-2024-chacal` Eq. 5 owns the equation; the machine check is the record's, `sweep_occupied.md` §3 (r)). **dies if red:** nothing in the arena; the manifest field `gamma` still carries the RUN.
- **price** `0 GPU-s` · **mechanism** V-3 (the pass is by construction, so the refusal is mandatory), V-24 · **deliverable** `lean/CEQ/Shape/Parity.lean` (spec here; file written by the author) · **evenings** 1

### J-L2 — Prove that row 0 of a causal softmax is absorbing at `β = 1`
- **phase** P0 · **prereq** J-L0 · **independent-of** all but J-L10 (which cites it)
- **what to build.** In `Shape/Boundary.lean`, on the tree's `Hop β g qk i j = if j ≤ i then num/Znorm^β else 0` (`READ V16Domain.lean:366-378`):
```
theorem bos_row_is_absorbing (g qk) : ∀ j, Hop 1 g qk 0 j = if j = 0 then 1 else 0
   -- Znorm g qk 0 = num g qk 0 0 (range 1 = {0}); rpow 1; div_self (exp ≠ 0)
theorem bos_row_not_absorbing_at_beta_zero : ∃ qk, Hop 0 0 qk 0 0 ≠ 1      -- refusal: at β = 0 row 0 reads exp(qk 0 0)
```
- **what to prove.** `[M]`; objects: `Finset.sum_range_one`, `Real.exp_ne_zero`, `Real.rpow_one`, `div_self`. Ships the F1 fact of `THESIS_CORRECTIONS_2.md` §1 as a theorem: position 0 is absorbing whether or not declared.
- **what to measure.** `RUN[P]`: BOS undeclared ⇒ `ρ(Q) = 1.000000`, `det(I−Q) = 0.000e+00`; declared as `𝒜_sink = {0}` ⇒ `ρ(Q) = 0.692660`. At `β = 0` row 0 reads `exp(qk₀₀) = 2.0138` on MARS's draw (`RUN[M]`).
- **PASS** builds; the domain census (J-L18) reads `100 %` of BED-S draws with `0 ∈ 𝒜_sink`. **KILL** the theorem needs `0 < Znorm` as an extra hypothesis the tree does not supply ⇒ `[S]`; the census line stays a RUN.
- **licenses if green:** census line X-2 of `bind_ledger.md` is a theorem instance, and `isUnit_one_sub_transient_causal` (J-L10) may take `0 ∈ 𝒜` as its only hypothesis. **dies if red:** BED-S admission needs the RUN check on every batch instead.
- **price** `0 GPU-s` · **mechanism** V-25 (a hypothesis no draw satisfies — here the hypothesis is *always* satisfied at `β = 1` and *never* at `β = 0`, so both halves are stated), V-12 · **deliverable** `lean/CEQ/Shape/Boundary.lean` · **evenings** 1

### J-L3 ★ — Prove that `I − γP` is a unit on the causal class, with the diagonal interval
- **phase** P0 · **prereq** J-L0 · **independent-of** J-L1, J-L2, J-L5, J-L8, J-L9, J-L11, J-L12
- **what to build.** In `Shape/Triangular.lean`:
```
theorem lower_triangular_isUnit (M : Matrix (Fin n) (Fin n) ℝ) (hM : LowerTri M)
    (hd : ∀ i, M i i ≠ 0) : IsUnit M
   -- Matrix.isUnit_iff_isUnit_det; det_of_lowerTriangular (h : M.BlockTriangular toDual); Finset.prod_ne_zero_iff
theorem diag_one_sub_smul_pos (P) (hP : RowStochastic P) (γ) (h0 : 0 ≤ γ) (h1 : γ < 1) (i) :
    1 - γ ≤ (1 - γ • P) i i ∧ (1 - γ • P) i i ≤ 1        -- 0 ≤ P i i ≤ 1 from nonneg + row_sum
theorem zero_diag_not_unit : ∃ M : Matrix (Fin 2) (Fin 2) ℝ, LowerTri M ∧ M 0 0 = 0 ∧ ¬ IsUnit M   -- refusal
```
`LowerTri M ↔ M.BlockTriangular toDual` is one `simp` lemma (`BlockTriangular M b := ∀ ⦃i j⦄, b j < b i → M i j = 0`, `READ Block.lean:54`; with `b = toDual`, `toDual j < toDual i ↔ i < j`).
- **what to prove.** `[M]` now (both Mathlib names confirmed, §0). The interval is `[1−γ, 1]` closed at `1` in Lean (it is attained by a row with `P_ii = 0`, which a softmax row never has — the open bound `< 1` needs `0 < P i i`, the extra hypothesis J-L5 carries).
- **what to measure.** `RUN[M]` min diagonal `0.4` at `γ = 0.6` (row 0 and every absorbing row attain `1−γ`); `RUN[coord]` `solve_triangular` vs dense inverse `1.7763568394002505e-15`.
- **PASS** builds. **KILL** `det_of_lowerTriangular`'s `LinearOrder (Fin n)` instance clashes with the tree's `Fin n` usage ⇒ prove the determinant by induction on `n` instead (`[S]`, +1 evening).
- **licenses if green:** Proposition 5(a) and 10 of `judge/sec_shape.md`; every `[M]` card below that inverts a triangular matrix (J-L4, J-L10, J-L13, J-L14). **dies if red:** the whole triangular branch of the DAG stalls one evening.
- **price** `0 GPU-s` · **mechanism** M-8 (the shape is priced as a forward substitution, not an inverse; the theorem is what makes the substitution well-defined) · **deliverable** `lean/CEQ/Shape/Triangular.lean` · **evenings** 1

### J-L4 — Prove that consequences propagate forward only
- **phase** P0 · **prereq** J-L3 · **independent-of** J-L5…J-L9, J-L11, J-L12
- **what to build.** Same file:
```
theorem inv_lowerTri (M) (hM : LowerTri M) (hd : ∀ i, M i i ≠ 0) : LowerTri M⁻¹
   -- Matrix.blockTriangular_inv_of_blockTriangular after `Invertible` from lower_triangular_isUnit
theorem later_boundary_unreachable (M) (hM) (hd) (i j : Fin n) (hij : (i:ℕ) < (j:ℕ)) :
    (M⁻¹ *ᵥ Pi.single j 1) i = 0                           -- F2: a set after the query has committor 0
theorem causal_forward_only (M) (hM) (hd) (i j) (hji : (j:ℕ) < (i:ℕ)) :
    (M⁻¹ *ᵥ Pi.single i 1) j = 0                           -- Δz is zero before the intervened row
theorem dense_P_displaces_backward : ∃ M : Matrix (Fin 2) (Fin 2) ℝ, IsUnit M ∧ (M⁻¹ *ᵥ Pi.single 1 1) 0 ≠ 0   -- refusal
```
- **what to prove.** `later_boundary_unreachable` `[M]`, `causal_forward_only` `[S]→[M]` (the same lemma read on the other index; the judge graded it `[S]` because it was routed through `sherman_morrison_row`; routed through `inv_lowerTri` it is one line).
- **what to measure.** `RUN[P]` `max|Δz_{<12}| = 0.000e+00` on the row clamp at `i = 12`; `RUN[F]` `5.6e-17` on the non-nilpotent softmax corner; the triangularity plant (non-causal `P`) reads `0.0761 / 0.0868` (`RUN[M]`, `bind_ledger.md` B-G1).
- **PASS** builds. **KILL** `blockTriangular_inv_of_blockTriangular` needs `[Invertible M]` as an instance and the `IsUnit → Invertible` cast does not elaborate ⇒ `[S]`, prove `inv_lowerTri` by the block formula of J-L6 instead.
- **licenses if green:** census line X-4 and bind B-G1(d′) as theorem instances; Proposition 7(ii)'s "by triangularity, not nilpotency". **dies if red:** the forward-only claim stays a RUN with its plant.
- **price** `0 GPU-s` · **mechanism** V-8 (a boundary set after the query gives a constant `0` label — the theorem says why), V-24 (the dense refusal is the plant) · **deliverable** `lean/CEQ/Shape/Triangular.lean` · **evenings** 1

### J-L5 — Prove that the softmax corner is not nilpotent (the regime boundary)
- **phase** P0 · **prereq** J-L0 · **independent-of** everything but J-L18
- **what to build.** `Shape/Triangular.lean`:
```
theorem lowerTri_pow_diag (P) (hP : LowerTri P) (k : ℕ) (i) : (P ^ k) i i = (P i i) ^ k   -- induction; the off-diagonal terms vanish
theorem softmax_corner_not_nilpotent (P) (hP : LowerTri P) (γ) (hγ : 0 < γ) (i) (hi : 0 < P i i) (k) :
    (γ • P) ^ k i i = (γ * P i i) ^ k ∧ 0 < (γ • P) ^ k i i        -- so ¬ IsNilpotent (γ • P)
```
The refusal is already in the tree: `Nilpotent.pow_card_eq_zero` on `StrictlyLower` (`:77`) and `one_not_nilpotent` (`:105`).
- **what to prove.** `[M]`; objects: `Matrix.mul_apply`, `Finset.sum_eq_single`, `pow_pos`.
- **what to measure.** `RUN[M]` `max_{i≥1}(γP)^{32}_{ii} = 6.27e-13` beside `γ^{32} = 7.96e-8` attained at `(0,0)` — the discriminating pair the judge fixed after a V-4 (`judge/sec_shape.md` Prop. 11).
- **PASS** builds. **KILL** none foreseeable; if `Finset.sum_eq_single` needs the `Fin` order unfolded, +1 evening.
- **licenses if green:** the sentence "all `s` hops by nilpotency holds in regime N only" is a theorem pair (`pow_card_eq_zero` / this); the paper's default "keep the diagonal, carry the Neumann certificate" is forced. **dies if red:** nothing measurable; the RUN pair stands.
- **price** `0 GPU-s` · **mechanism** V-25 (the record's finite-sum theorems transfer to the softmax corner on `0 %` of draws — this theorem is the census), P-3 · **deliverable** `lean/CEQ/Shape/Triangular.lean` · **evenings** 1

### J-L6 ★ — Prove that a zero gate splits the resolvent into blocks
- **phase** P0 · **prereq** J-L0 · **independent-of** J-L1…J-L5, J-L8…J-L17
- **what to build.** `Shape/Segment.lean`:
```
theorem segmentation_blockdiag {A} (hA : StrictlyLower A) (c : ℕ)
    (hcut : ∀ i j : Fin n, (j:ℕ) < c → c ≤ (i:ℕ) → A i j = 0) :
    ∀ i j, (j:ℕ) < c → c ≤ (i:ℕ) → (CEQ.Occupancy.occupancy A n) i j = 0
   -- induction on k for A^k exactly as pow_entry_zero (Nilpotent.lean:52): (A^(m+1)) i j = Σ_l (A^m) i l * A l j;
   -- l < c kills the first factor (IH), c ≤ l kills the second (hcut); then sum over k < n
theorem chain_zero_gate_cuts (a : ℕ → ℝ) (c) (hc : a c = 0) : ∀ i j, (j:ℕ) < c → c ≤ (i:ℕ) → occupancy (subdiag a) n i j = 0   -- [S] via J-L11
theorem resolvent_fromBlocks (M₁₁ : Matrix m m ℝ) (M₂₂ : Matrix n' n' ℝ) (h₁ : IsUnit M₁₁) (h₂ : IsUnit M₂₂) :
    (Matrix.fromBlocks M₁₁ 0 0 M₂₂)⁻¹ = Matrix.fromBlocks M₁₁⁻¹ 0 0 M₂₂⁻¹
   -- inv_fromBlocks_zero₂₁_of_isUnit_iff with B = 0: the (1,2) block −(A⁻¹ * 0 * D⁻¹) simp's to 0
theorem tiny_gate_does_not_cut : ∃ (a : ℕ → ℝ) (c i j), a c = Real.exp (-30) ∧ (j:ℕ) < c ∧ c ≤ (i:ℕ) ∧ occupancy (subdiag a) n i j ≠ 0   -- refusal: F1 is not F0
```
- **what to prove.** `segmentation_blockdiag` `[M]` (the induction is `pow_entry_zero`'s), `resolvent_fromBlocks` `[M]` (name confirmed §0), the chain corollary `[S]` (needs J-L11), the refusal `[M]` (`Real.exp_pos`).
- **what to measure.** `RUN[M]` segmentation zeros `array_equal` on the gated corner; the `−30` plant (`e^{−30} = 9.36e-14`, exact zero only below `−745`) leaves the block non-zero; the `1e-300` plant is struck (underflow, `bind_ledger.md` B-K). Domain census: BED-M `3 of 3` values admit a zero gate (`READ V16Domain.lean:304`); BED-S at the softmax corner `0 of N` — the theorem is **silent** there and the paper says so.
- **PASS** builds. **KILL** `Matrix.fromBlocks` indexing (`Sum m n'`) does not match the `Fin n` cut ⇒ state `segmentation_blockdiag` only (entrywise, no reindexing) and demote `resolvent_fromBlocks` to a remark citing Mathlib; `[S]`.
- **licenses if green:** the contract's F0 line (`READ CEQ_V20_R15_CONTRACT.md:212-213`) finally has a declaration; certificate Z-F0 (`δ = 0`) is a theorem instance; the pair-count dividend `D = s(s+1)/Σ L_m(L_m+1) = 31.04×` on BED-M stays a corpus property (V-22). **dies if red:** the F0 row keeps its `torch.equal` RUN and no `[M]` tag.
- **price** `0 GPU-s` · **mechanism** L-CERT, P-11 (the `#22 [M]` with no declaration), V-25, V-2 (the refusal has a real witness) · **deliverable** `lean/CEQ/Shape/Segment.lean` · **evenings** 1

### J-L7 — Prove that a cut makes the segment head absorbing and severs the sets behind it
- **phase** P0 · **prereq** J-L6, J-L2 · **independent-of** J-L8…J-L17
- **what to build.** Same file, on `Hop` with a `−∞`-logit mask represented as a zero gate `m_c = 0` in the `pathProd` base (`READ V16Domain.lean:97`):
```
theorem cut_makes_segment_head_absorbing (m θ qk) (c) (hc : m c = 0) :
    HopGated 1 m θ qk c c = 1 ∧ ∀ j < c, HopGated 1 m θ qk c j = 0     -- window of row c is {c}; renormalises to 1
theorem cut_severs_boundary_sets (P) (hP : LowerTri P) (hd) (c) (hcut : ∀ i j, j < c → c ≤ i → P i j = 0)
    (j) (hj : (j:ℕ) < c) (i) (hi : c ≤ (i:ℕ)) : ((1 - P)⁻¹ *ᵥ Pi.single j 1) i = 0    -- via segmentation_blockdiag on 1 − P
```
`HopGated` is the gated base of Definition 1 (`judge/sec_shape.md` §4.1) — the `Hop` of `V16Domain.lean:366` carries no zero (`no_prefix_scan_represents_a_zero_gate`, `:165`), so this card defines the gated hop once, with the product taken directly.
- **what to prove.** `[M]` for both; objects as J-L2, J-L6.
- **what to measure.** `RUN[J]` after a cut at `c`: `P_cc = 1.0`, `ρ(Q) = 1.0` with `c ∉ 𝒜`; `RUN[M]` `q^{(0)} = q^{(1)} = 0.0` past the cut once `c` is declared (Proposition 6(ii)).
- **PASS** builds. **KILL** defining `HopGated` duplicates `Hop` and the containment of `three_corners_containment` must be re-proved for it ⇒ +1 evening, still `[M]`.
- **licenses if green:** the dial rule "every segment carries its own sink, goal and constraint sets; the cut position is registered" (`sec_apparatus.md` §A.2.1) is a theorem consequence, not a design preference. **dies if red:** segmentation and the committor channel are kept apart by rule only.
- **price** `0 GPU-s` · **mechanism** V-25, D-3 (the cut is a dial), V-8 · **deliverable** `lean/CEQ/Shape/Segment.lean` · **evenings** 1

### J-L8 ★ — Prove the displacement identity with its planted zero
- **phase** P0 · **prereq** J-L0 · **independent-of** everything but J-L14, J-L18
- **what to build.** `Shape/Displacement.lean`:
```
theorem displacement_identity (P P' : Matrix (Fin n) (Fin n) ℝ) (V V' z z' : Fin n → ℝ) (γ : ℝ)
    (hz : (1 - γ • P) *ᵥ z = V) (hz' : (1 - γ • P') *ᵥ z' = V') :
    (1 - γ • P') *ᵥ (z' - z) = (V' - V) + γ • ((P' - P) *ᵥ z)
   -- four lines: expand (1 − γP')z = V − γ(P' − P)z using hz; subtract from hz'
theorem const_value_zero_displacement (P P') (hP : RowStochastic P) (hP' : RowStochastic P') (γ) (h1 : γ < 1) (c : ℝ) :
    (1 - γ • P)⁻¹ *ᵥ (fun _ => c) = (1 - γ • P')⁻¹ *ᵥ (fun _ => c)      -- both equal c/(1−γ) · 𝟙 ; needs IsUnit from J-L3 or J-L12
```
- **what to prove.** `displacement_identity` `[M]` (ring arithmetic on `mulVec`; objects `Matrix.sub_mulVec`, `Matrix.smul_mulVec_assoc`, `Matrix.mulVec_sub`); the refusal `[M]` given a unit (J-L3 on the causal class, J-L12 on the stochastic class).
- **what to measure.** `RUN[P]` residual `2.897e-16`; `RUN[I]` `1.2e-15`, `RUN[F]` `1.36e-15`, `RUN[J]` `1.03e-15`; `V ≡ 𝟙 ⇒ max|Δz| ≤ 1e-15` (`RUN[F]` `1.1e-16`; "bitwise `0.0`" is a code-path accident, `refute_falsify_math` §3.7); Gaussian `V` gives `0.1096 / 0.363 / 1.127` on three draws — the rejection region.
- **PASS** builds. **KILL** none foreseeable.
- **licenses if green:** Proposition 7(i) is LEAN; the C6 identity of `THESIS_CORRECTIONS.md` is carried as a theorem beside `bottou-2013-counterfactual` §7.3 (its linearisation) and `sherman-1950-inverse-adjustment` (the rank-one form, J-L14). **dies if red:** nothing; the three independent RUNs stand.
- **price** `0 GPU-s` · **mechanism** V-24 (the constant-value plant is the identity half; Gaussian `V` the region), D-5 (`0.0` without a movement test) · **deliverable** `lean/CEQ/Shape/Displacement.lean` · **evenings** 1

### J-L9 — Prove that a causal operator is never the oracle's undirected chain (D-2 in regime S)
- **phase** P0 · **prereq** J-L0 · **independent-of** everything but J-L18
- **what to build.** `Shape/Committor.lean`, extending `OracleSeparation.oracle_ne_resolvent` (`:166`, `StrictlyLower` only) to the diagonal-kept class:
```
theorem lowerTriangular_ne_symmSupport (P Q : Matrix (Fin n) (Fin n) ℝ) (hP : LowerTri P)
    (hQ : SymmSupport Q) {i j : Fin n} (hij : (j:ℕ) < (i:ℕ)) (hpos : 0 < Q i j) : Q ≠ P
   -- SymmSupport gives 0 < Q j i; LowerTri gives P j i = 0; contradiction at (j, i)
```
Refusal pattern: `zero_not_a_counterexample` (`OracleSeparation.lean:193`) — dropping `hpos` makes the statement false (`Q = 0` is `SymmSupport`, lower-triangular).
- **what to prove.** `[M]`, one line.
- **what to measure.** BED-1's chain: `Nonneg True`, `SymmSupport True`, `ρ(Q) = 0.9408612510154677`, `Q^{11} ≠ 0` (`RUN[coord]`, `sec_proved.md` §2.1) — the hypotheses instantiated on the real oracle.
- **PASS** builds. **KILL** none.
- **licenses if green:** the ruling that BED-S's environment must be a **DAG in token order** (`sec_apparatus.md` §A.1) is a theorem consequence: an undirected substrate can never be matched by any causal `P̂`, so an arm on it measures nothing (`refute_instrument_occvac` FATAL-1). **dies if red:** the substrate ruling stands on the RUN alone.
- **price** `0 GPU-s` · **mechanism** D-2 (`READ MISTAKES.md:710`), V-25 · **deliverable** `lean/CEQ/Shape/Committor.lean` · **evenings** 1

### J-L10 — Prove the committor read and its causal invertibility
- **phase** P0 · **prereq** J-L3, J-L2 · **independent-of** J-L5…J-L9, J-L11, J-L12
- **what to build.** `Shape/Committor.lean`:
```
theorem committor_is_resolvent_read {t : ℕ} (Q : Matrix (Fin t) (Fin t) ℝ) (r q : Fin t → ℝ)
    (hharm : q = Q *ᵥ q + r) (hunit : IsUnit (1 - Q)) : q = (1 - Q)⁻¹ *ᵥ r       -- (a) [M]
theorem isUnit_one_sub_transient_causal (Q) (hQ : LowerTri Q) (hd : ∀ i, Q i i < 1) : IsUnit (1 - Q)   -- (b-causal) [M], from J-L3
theorem transient_diag_lt_one_of_bos_declared (P) (hP : RowStochastic P) (hcausal : LowerTri P)
    (hpos : ∀ i j, (j:ℕ) ≤ (i:ℕ) → 0 < P i j)                           -- a softmax row has every causal entry positive
    (T : Finset (Fin n)) (h0 : (0 : Fin n) ∉ T) : ∀ i ∈ T, P i i < 1        -- row i ≥ 1 puts positive mass on j = 0
theorem bos_undeclared_singular : ∃ P : Matrix (Fin 2) (Fin 2) ℝ, RowStochastic P ∧ LowerTri P ∧ ¬ IsUnit (1 - P)   -- refusal: P = 1 on row 0
```
- **what to prove.** (a) `[M]` (`sub_mulVec`, `nonsing_inv_mul_cancel_left`); (b-causal) `[M]`; the BOS-declared lemma `[M]` (a positivity argument: `P_ii = 1 − Σ_{j<i} P_ij < 1` once `P_i0 > 0`).
- **what to measure.** `RUN[P]` `ρ(Q) = max_{i∈T} P_ii = 0.692660`; BED-1's real sets `A = [0], B = [1], |T| = 9`: resolvent read vs `bed["q"]` `0.0`, residual `1.04e-17` (`RUN[coord]`); the must-fire perturbation drives the residual to `1e-6` (ratio `9.6e10`, `READ workdonenewseal.md:214-215`); the wrong-set plant moves `q` by O(1) (`bind_ledger.md` B-H1).
- **PASS** builds. **KILL** (b-causal) needs the strict-positivity hypothesis `hpos` that the gated corners do not satisfy (a zero gate zeroes `P_i0`) ⇒ the theorem is stated for the softmax corner only and the census line X-3 stays a RUN on gated corners.
- **licenses if green:** Proposition 10 of `judge/sec_shape.md` — "absorption a.s." is the single census line "BOS declared"; the committor head's exact route (`δ = 0`) rests on a theorem. **dies if red:** the head ships on the RUN `det ≠ 0` check per batch.
- **price** `0 GPU-s` · **mechanism** D-2, V-10 (the `ρ(Q̂) < 1` gate is implied by BOS-declared and is printed, not counted), V-16 (the solve raises when BOS is undeclared) · **deliverable** `lean/CEQ/Shape/Committor.lean` · **evenings** 1

### J-L11 — Prove that corner 3 is the sub-diagonal resolvent, entrywise
- **phase** P0 · **prereq** J-L0 · **independent-of** everything but J-L6(chain corollary), J-L18
- **what to build.** `Shape/Segment.lean` (or `Shape/Chain.lean`):
```
def subdiag (a : ℕ → ℝ) : Matrix (Fin n) (Fin n) ℝ := Matrix.of fun i j => if (i:ℕ) = (j:ℕ) + 1 then a i else 0
theorem subdiag_strictlyLower (a) : StrictlyLower (subdiag a)                                   -- [M]
theorem subdiag_pow_entry (a) (k) (i j) :
    (subdiag a ^ k) i j = if (i:ℕ) = (j:ℕ) + k then ∏ l in Ico ((j:ℕ)+1) ((i:ℕ)+1), a l else 0   -- [S]: induction on k with the surviving term tracked
theorem pathprod_is_chain_resolvent (a) (i j) (hij : j ≤ i) :
    (CEQ.Occupancy.occupancy (subdiag a) n) i j = ∏ l in Ico ((j:ℕ)+1) ((i:ℕ)+1), a l            -- [S]: exactly one k = i − j term survives
```
The `n = 3` instance is already in the tree (`V15Source.forward_map_fills_in`, `:168`); `V15.chain_path_product` (`:74`) gives the last row as the label.
- **what to prove.** `[S]` — the induction is `pow_entry_zero`'s with one surviving term; objects `Finset.sum_eq_single`, `Finset.prod_Ico_succ_top`.
- **what to measure.** `RUN[coord]` `max|G − (I−A)^{-1}| = 0.0`, last row vs `equilibrium_oracle` `6.217248937900877e-15`, `A^{64} = 0` exactly.
- **PASS** builds. **KILL** the `Ico` product bookkeeping does not close in one file ⇒ ship `subdiag_pow_entry` for the two cases `k = i − j` and `k ≠ i − j` separately (`[S]`, +1 evening).
- **licenses if green:** Proposition 2 is LEAN in both halves and BED-M is *contained* by theorem: any arm containing corner 3 reproduces `equilibrium_oracle` as its own forward, so `shape − corner-3` on BED-M is VOID by theorem (D-2 made exact). **dies if red:** the containment stays a `0.0` RUN — still VOID by registration.
- **price** `0 GPU-s` · **mechanism** D-2, D-1 · **deliverable** `lean/CEQ/Shape/Chain.lean` · **evenings** 2

### J-L12 — Prove the Neumann certificate, its attainment, and the mask amplification
- **phase** P0 · **prereq** J-L0 · **independent-of** J-L1…J-L11 (J-L13's mixing lemma and J-L16 depend on it)
- **what to build.** `Shape/Neumann.lean`, pointwise in the style of `Contraction.lean` (no norm instance):
```
theorem resolvent_sup_bound (P) (hP : RowStochastic P) (γ) (h0 : 0 ≤ γ) (h1 : γ < 1) (u x : Fin n → ℝ)
    (hx : x = γ • (P *ᵥ x) + u) (M) (hu : ∀ j, |u j| ≤ M) : ∀ i, |x i| ≤ M / (1 - γ)
   -- take i at the finite sup of |x|; weighted_contraction with w = 1 gives |x_i| ≤ γ‖x‖_∞ + M
theorem isUnit_one_sub_smul (P) (hP) (γ) (h0) (h1) : IsUnit (1 - γ • P)      -- injective (sup bound at u = 0) ⇒ unit on a finite type
theorem neumann_truncation_bound (P) (hP) (γ) (h0) (h1) (K : ℕ) (v M) (hv : ∀ j, |v j| ≤ M) :
    ∀ i, |((1 - γ • P)⁻¹ *ᵥ v - CEQ.Occupancy.occupancy (γ • P) (K + 1) *ᵥ v) i| ≤ γ ^ (K + 1) / (1 - γ) * M
   -- residual = (1 − γP)⁻¹ (γP)^{K+1} v by occupancy_telescope (:52); iterate weighted_contraction K+1 times (the iterate is the missing induction — weighted_contraction_iterate at :99 is one step)
theorem neumann_tail_attained (P) (hP) (γ) (h0) (h1) (K) :
    ∀ i, ∑ j, ((1 - γ • P)⁻¹ - occupancy (γ • P) (K + 1)) i j = γ ^ (K + 1) / (1 - γ)     -- row sums: every entry ≥ 0, P^k row-stochastic; the ∞-norm IS this (declared V-3)
theorem nonstochastic_breaks_bound : ∃ P : Matrix (Fin 2) (Fin 2) ℝ, (∀ i j, 0 ≤ P i j) ∧ (∀ i, ∑ j, P i j = 3/2) ∧
    ∃ K v M, (∀ j, |v j| ≤ M) ∧ ∃ i, γ₀ ^ (K+1)/(1 - γ₀) * M < |((1 - γ₀ • P)⁻¹ *ᵥ v - occupancy (γ₀ • P) (K+1) *ᵥ v) i|   -- γ₀ = 3/5, convergent (γ₀·3/2 < 1); docstring: "can fail", not "no bound holds"
theorem mask_amplification (P Pm) (hP hPm : RowStochastic) (γ) (h0) (h1) (ε) (hε : ∀ i, ∑ j, |P i j - Pm i j| ≤ ε) (v M) (hv) :
    ∀ i, |(P *ᵥ ((1 - γ • P)⁻¹ *ᵥ v) - Pm *ᵥ ((1 - γ • Pm)⁻¹ *ᵥ v)) i| ≤ ε * M / (1 - γ) * (1 + γ/(1-γ))    -- [S]; the s = 3 witness 0.5263 vs naive 0.1
```
- **what to prove.** all `[S]` (short; the `K`-fold iterate is the only induction). Objects: `Finset.exists_max_image`, `abs_sub`, `mul_neg_geom_sum` if the telescope is re-derived, `tsum_geometric_of_lt_one` only for the infinite-series remark.
- **what to measure.** `RUN[P]` equality at `(0.6,2)`, `(0.7,4)`, `(0.7,16)` to the printed digits; the convergent plant `7.2900` vs `0.5400`; the coordinator's `γ = 0.7` plant (`119.37` vs `1.143`) is a divergent partial sum (`ρ(γP) = 1.05`) and is labelled so; `RUN[J]` mask instance `s = 3, γ = 0.9, ε = 0.1`: `0.5263157894736843` against the naive `0.1`.
- **PASS** builds; the paper prints `δ·‖V‖_∞`, never bare `δ`, with `1/(1−γ̂)` beside it. **KILL** `neumann_tail_attained` needs `P^k` row-stochastic as a lemma (`RowStochastic (P^k)`) — an induction the tree lacks ⇒ +1 evening; if `mask_amplification`'s constant does not close cleanly, ship the `s = 3` witness as the theorem (`[M]`) and the general bound as `[D]`.
- **licenses if green:** Proposition 4(i)–(ii); certificate rows Z-NEU and Z-F1 (dormant until a mask ships, D-4). **dies if red:** every Neumann `δ` is printed from the textbook (`meyer-2000-matrix`, `horn-2013-matrix`) with the RUN plant; the arena loses no cell (the exact solve is cheaper than one hop at `s = 64`, `2.514 < 3.001` ms, `RUN[NEPTUNE]`).
- **price** `0 GPU-s` · **mechanism** V-3 / V-10 (attained ⇒ definitional, declared), V-24 (the plant), V-17 (units), L-CERT · **deliverable** `lean/CEQ/Shape/Neumann.lean` · **evenings** 2

### J-L13 — Prove that the resolvent is one forward substitution and that `Π_γ` is a mixture
- **phase** P0 · **prereq** J-L3, J-L12 · **independent-of** J-L5…J-L9, J-L11
- **what to build.** `Shape/Triangular.lean` and `Shape/Neumann.lean`:
```
def fwdSub (M : Matrix (Fin n) (Fin n) ℝ) (v : Fin n → ℝ) : Fin n → ℝ   -- well-founded on i: (v i − Σ_{j<i} M i j * fwdSub M v j) / M i i
theorem resolvent_is_triangular_solve (M) (hM : LowerTri M) (hd : ∀ i, M i i ≠ 0) (v) : M⁻¹ *ᵥ v = fwdSub M v   -- [S]: show M *ᵥ fwdSub M v = v by strong induction, then inv_eq_left_inv / mulVec injectivity
theorem mixing_matrix_rowStochastic (P) (hP : RowStochastic P) (γ) (h0) (h1) :
    RowStochastic ((1 - γ) • (P * (1 - γ • P)⁻¹))                                          -- [S]: entries ≥ 0 (Neumann series of nonneg terms), rows 1 (row sums γ^t)
theorem bare_read_row_sum (P) (hP) (γ) (h0) (h1) (i) : ∑ j, (P * (1 - γ • P)⁻¹) i j = 1 / (1 - γ)     -- refusal-shaped: the bare read is not a mixture
```
- **what to prove.** `[S]` all three. `mixing_matrix_rowStochastic` needs the entrywise non-negativity of `(1 − γP)^{-1}` — the finite-`N` occupancy is non-negative by induction, and the limit statement is the one place a `tsum` (`tsum_geometric_of_lt_one`) or a squeeze on `occupancy … N` as `N → ∞` enters; the pointwise route: `(1−γP)^{-1} = occupancy(γP, N) + (1−γP)^{-1}(γP)^N` for every `N`, both terms non-negative, so the inverse is non-negative.
- **what to measure.** `RUN[coord]` `solve_triangular` vs dense inverse `1.7763568394002505e-15`; `RUN[F]` `Π_γ` row sums `1.000000000000000` with and without absorbing rows; `RUN[J]`, `RUN[M]` bare row sums `2.5 = 1/(1−0.6)`; `RUN[J]` `supp Π_γ = supp P` on a dense causal softmax (the "transitive closure" sentence has content only under F0 zeros).
- **PASS** builds. **KILL** `fwdSub`'s well-founded recursion on `Fin n` does not elaborate ⇒ state the substitution as a `∀ i, M i i * z i = v i − Σ_{j<i} M i j * z j` characterisation of `z = M⁻¹ *ᵥ v` (`[M]`) and drop the executable definition.
- **licenses if green:** Proposition 5(a)–(c); the Q2/W3 hull bound (`READ V20_R15_THEORY_TABLE.md` Q2/W3) applies to the shape verbatim by theorem; the cost line "`+s²d/2` MACs, depth `s`" prices a substitution that exists. **dies if red:** the cost law stands on the RUN.
- **price** `0 GPU-s` · **mechanism** M-8, V-17 (the `(1−γ)` factor and the row sums are the units), V-23 · **deliverable** `lean/CEQ/Shape/Triangular.lean`, `Shape/Neumann.lean` · **evenings** 2

### J-L14 — Prove the suffix re-solve and the rank-one row clamp
- **phase** P0 · **prereq** J-L4, J-L8 · **independent-of** J-L5…J-L7, J-L9, J-L11…J-L13, J-L15…J-L17
- **what to build.** `Shape/Displacement.lean`:
```
theorem suffix_resolve_eq_full (P P') (hP hP' : LowerTri) (hd hd') (V V') (i : Fin n)
    (hpre : ∀ j, (j:ℕ) < (i:ℕ) → (∀ k, P' j k = P j k) ∧ V' j = V j) :
    ∀ j, (j:ℕ) < (i:ℕ) → ((1 - γ • P')⁻¹ *ᵥ V') j = ((1 - γ • P)⁻¹ *ᵥ V) j       -- prefix invariance; the suffix re-solve equals the full re-solve
theorem sherman_morrison_row (P) (hP : RowStochastic P) (γ) (h1 : γ < 1) (i : Fin n) (u : Fin n → ℝ) (hu : ∑ j, u j = 0)
    (hrow : ∀ j, 0 ≤ P i j + u j) (V) :
    let M := (1 - γ • P)⁻¹; let z := M *ᵥ V
    (1 - γ • (P + Matrix.of fun a b => if a = i then u b else 0))⁻¹ *ᵥ V - z
      = (γ * (u ⬝ᵥ z) / (1 - γ * (u ⬝ᵥ fun j => M j i))) • (fun j => M j i)      -- docstring: a ROW CLAMP only; a token rewrite is rank s − i
theorem sm_denominator_pos … : 0 < 1 - γ * (u ⬝ᵥ fun j => M j i)   -- equals (1 − γ p'_ii)/(1 − γ P_ii) > 0
theorem dense_P_breaks_suffix_resolve : ∃ P P' V V' i, … ∧ ((1 - γ • P')⁻¹ *ᵥ V') 0 ≠ ((1 - γ • P)⁻¹ *ᵥ V) 0   -- refusal
```
- **what to prove.** `[S]` all; `suffix_resolve_eq_full` follows from `inv_lowerTri` (J-L4) applied to the prefix block; Sherman–Morrison from `sherman-1950-inverse-adjustment` / `hager-1989-updating` `[V]` re-proved on `Fin n` (the algebra is `(M + γ M e_i uᵀ M /(1 − γ uᵀ M e_i))(1 − γP') = 1`).
- **what to measure.** `RUN[F]` `5.6e-17` suffix vs full on the non-nilpotent corner; `RUN[I]` closed form vs re-solve `1.2339847026143769e-15`, denominator `1.9`; `RUN[F]` `0.511918`; a two-row edit breaks the rank-one formula by O(1) (repaired by Woodbury, the plant); price `m` columns at `s²/2` MACs each, ratio `m/d = 0.5` at `(8,16)` (`judge/sec_shape.md` Prop. 7(iii)), SPLIT band `(0.5×, 8×)` filed.
- **PASS** builds. **KILL** the `Matrix.of` row-update expression does not simp ⇒ state the rank-one update for `P' = P + Matrix.vecMulVec (Pi.single i 1) u` (`[S]`, same content).
- **licenses if green:** Proposition 7(ii)–(iii); the pricing of the safest-move search ("one solve plus `m` columns") is a theorem about the *oracle's* row clamp and is never carried to the arm's token rewrite (M-8). **dies if red:** the price is a DERIVED band with the RUN identities.
- **price** `0 GPU-s` · **mechanism** M-8, P-8, V-24, D-5 · **deliverable** `lean/CEQ/Shape/Displacement.lean` · **evenings** 1

### J-L15 — Prove the reach-avoid conservation row and the degeneracy lemma
- **phase** P0 · **prereq** J-L10 · **independent-of** J-L5…J-L9, J-L11…J-L14
- **what to build.** `Shape/Committor.lean`:
```
theorem reach_avoid_sum_one (P) (hP : RowStochastic P) (T 𝒜 : Finset (Fin n)) (hpart : T ∪ 𝒜 = univ ∧ Disjoint T 𝒜)
    (habs : Absorbing P 𝒜) (sets : Fin K → Finset (Fin n)) (hexh : (⋃ k, sets k) = 𝒜 ∧ pairwise disjoint)
    (hunit : IsUnit (1 - P.submatrix T T)) :
    ∀ i ∈ T, ∑ k, ((1 - Q)⁻¹ *ᵥ (R k *ᵥ 𝟙)) i = 1        -- Q = P_TT, R_k = P_{T,sets k}; Σ_k R_k 𝟙 = 𝟙 − Q 𝟙 ⇒ N(𝟙 − Q𝟙) = 𝟙
theorem no_goal_no_sink_forces_max_ge_inv_K … : ∀ i ∈ T, 1/K ≤ ⨆ k, q k i     -- pigeonhole on the row above (Finset.exists_le_card_fiber / sum ≤ K · max)
theorem isUnit_one_sub_of_perron {Q w ρ} (hc : PerronCertificate Q w ρ) (hρ : ρ < 1) : IsUnit (1 - Q)    -- (b-chain) [S]: the oracle's non-causal Q; weighted sup bound as J-L12
theorem unit_needs_rho_lt_one : ∃ Q w, PerronCertificate Q w 1 ∧ ¬ IsUnit (1 - Q)     -- refusal: Q = 1
```
- **what to prove.** `[S]`; objects `Matrix.submatrix`, `Finset.sum_comm`, `Finset.exists_max_image`; `kemeny-1976-finite` Thm (`B = N R`) and `grinstead-1997-probability` Thm 11.6 `[U]` own the identity — the card machine-checks the *packaging* (sets exhaust the absorbing states).
- **what to measure.** `RUN[P]` sum over sink+goal+constraints `[1.000000000000000, 1.000000000000001]` on `T`; constraints-only with BOS inside a constraint set: sum `1.000000000000000`, `min_T max_k = 0.548718 ≥ 0.5` (the lemma's instance; `RUN[I]`'s `0.605` was computed *with* the goal counted and is not this lemma's instance, `bind_ledger.md` B-E2). The plant "drop `𝒜_0`" cannot run (dropping BOS makes `I − Q` singular); the lemma's plant is "BOS declared inside a constraint set".
- **PASS** builds. **KILL** `Finset` submatrix indexing over `{i // i ∈ T}` does not close in one file ⇒ state on the full `Fin n` with `Absorbing` rows and the vector `𝟙_T` (`[S]`, +1 evening).
- **licenses if green:** Proposition 8's conservation row and degeneracy lemma are LEAN; census line X-7 and bind B-E2's identity half are theorem instances; the C5 fix (a goal set and a declared sink) is forced by theorem. **dies if red:** the row is printed with its residual on every batch (V-23) and the lemma is DERIVED.
- **price** `0 GPU-s` · **mechanism** V-12 (a single absorbing target makes the label constant), V-8, V-23, V-25 · **deliverable** `lean/CEQ/Shape/Committor.lean` · **evenings** 2

### J-L16 — Prove the masked-sink finite sum and BED-1's solve as the committor read
- **phase** P0 · **prereq** J-L12, J-L10 · **independent-of** J-L5…J-L9, J-L11, J-L14
- **what to build.**
```
theorem masked_sink_finite_sum (P) (hP : LowerTri P) (Pm := P with row 0 and the diagonal zeroed) (V : Fin n → ℝ) (hV0 : V 0 = 0) (γ) :
    occupancy (γ • Pm) n *ᵥ V = (1 - γ • Pm)⁻¹ *ᵥ V        -- Pm is StrictlyLower ⇒ pow_card_eq_zero; exact iff the sink channel is value-zero
theorem goal_channel_breaks_finite_sum : ∃ P V, V 0 = 1 ∧ occupancy (γ • Pm) n *ᵥ V ≠ (1 - γ • P)⁻¹ *ᵥ V    -- refusal (RUN[M] 1.99e-7 on 𝟙_{𝒜₀} with 0 ∈ 𝒜₀)
theorem bed1_committor_eq (L : Matrix (Fin n) (Fin n) ℝ) (d : ℝ) (hd : 0 < d) (T B) … :
    (solve L_TT (−L_TB 𝟙)) = (1 - Q)⁻¹ *ᵥ (R *ᵥ 𝟙_B)  with Q = 1 + L_TT/d, R = L_TB/d      -- (c): the definitional unpacking of ceq/beds/bed_1.py:188-198
```
- **what to prove.** `[S]` both; the first is `occupancy_is_exact_inverse` on the masked operator plus `V 0 = 0` killing the one column where `Pm ≠ P`; the second is algebra on `READ ceq/beds/bed_1.py:173-174, :188-198` (`P = K/d`, `diag = 1 − rowsum(K)/d`).
- **what to measure.** `RUN[M]` masked route `4.4e-16` on a value-zero channel vs `1.99e-07` on `𝟙_{𝒜_0}` with `0 ∈ 𝒜_0`; BED-1 `harmonic_residual = 1.0408340855860843e-17` at `T = 0.25, jitter = 0.05, seed = 11`.
- **PASS** builds. **KILL** the BED-1 unpacking needs the bed's `K` as a Lean object ⇒ state it for an abstract generator `L` with `L_TT` diagonally dominant (`[S]`).
- **licenses if green:** the Definition-3 ruling "BOS is a value-zero sink set" is what makes the finite-sum route exact — a theorem, not a convention (Proposition 11); the oracle cross-check B-H1 is an instance of (c). **dies if red:** the paper keeps the diagonal and Proposition 4 everywhere (the default already).
- **price** `0 GPU-s` · **mechanism** V-25, V-4 (the C8 evidence was attained at `(0,0)` — the theorem says which entry carries the information), D-2 · **deliverable** `lean/CEQ/Shape/Committor.lean`, `Shape/Segment.lean` · **evenings** 1

### J-L17 — Register the two deferred targets
- **phase** P5 · **prereq** J-L0 · **independent-of** all
- **what to build.** Two declarations stated with `sorry`-free *statements only* in a separate file `Shape/Deferred.lean` that is **not** imported by `CEQ.lean` (so `lake build` of the root stays `sorry`-free): `hitting_time_transform` (P3's probability clause `(1−γ) z_i = E_i[γ^{τ_k} 𝟙_k]`) and `f1_cantelli_union` (the F1 union bound with the `1/(1−γ)` amplification). Both `[D]`: this Mathlib has no finite-chain hitting-time object (§0), and Cantelli/Bonferroni primaries are `[U]` (`sweep_methods.md` §1.A) — the F1 line cites `durrett-2019-probability` / `hoeffding-1963-probability` / `azuma-1967-weighted` `[V]` as textbook statements.
- **what to prove.** Nothing this round; the content is the paper derivation J-D3 (with a pointwise finite-sum proof that avoids probability altogether) and J-D5's constant.
- **what to measure.** `RUN[J]`/`RUN[M]` read vs `E[γ^{τ−1}]` to `8.33e-17` (Monte-Carlo-free, by the finite-sum identity).
- **PASS** the file parses (statements type-check). **KILL** a statement needs a measure-theoretic object the tree cannot name ⇒ the statement is written in the finite-sum form of J-D3 instead.
- **price** `0 GPU-s` · **mechanism** P-11 (a `[D]` is never cited as `[M]`), P-4 · **deliverable** `lean/CEQ/Shape/Deferred.lean` (statements), `docs/CEQ_SHAPE.md` §4 (the derivation) · **evenings** 1

### J-L18 ★ — Run the build gate, the axiom print and the domain census
- **phase** P0 gate · **prereq** every `[M]` card it certifies (J-L1–J-L10 at minimum) · **independent-of** the `[S]` cards it does not yet list
- **what to build.** (i) `lake build` from `lean/` with the new imports in `CEQ.lean`; (ii) a `#print axioms` block per new declaration, the pattern of `V16Domain.lean:586-594`; (iii) a **domain census file** in the `bedM_overlap_*` style (`READ V16Domain.lean:302-310`, `by decide` on the corpus value set): for each new theorem, the corpus (BED-M values `{−1, 0, +1}`; BED-S at the softmax corner: `β = 1`, row-stochastic, BOS-declared, sets before the query) and the fraction of that corpus satisfying the hypotheses at the quantifier level the theorem uses — `gamma_zero_is_softmax` `3/3`; `bos_row_is_absorbing` `3/3` at `β = 1`, `0/3` at `β = 0`; `segmentation_blockdiag` `3/3` on BED-M, `0/N` on BED-S's softmax corner (silent); `softmax_corner_not_nilpotent` `0/3` on BED-M (no diagonal), `N/N` on BED-S; `lowerTriangular_ne_symmSupport` `1/1` on BED-1's chain; `neumann_*` `N/N` on BED-S at `β̂ = 1`, `0/N` at `β̂ ≠ 1` (rows sum `1.31 … 10.29`, `READ V16_ARM_SMPRIME.md:28-32`). Journal fields: declaration, file:line, grade before/after, axioms, census fraction, corpus name.
- **what to prove.** The census rows by `decide` where the value set is finite (BED-M); by a printed RUN where it is a drawn batch (BED-S).
- **what to measure.** `lake build` exit code; `grep -c '^theorem\|^lemma'` on the new files (the record's count is `169` over 12 files, `READ BRIEF.md` §2; the new count is journalled beside it); `#print axioms` output per declaration.
- **PASS** exit `0`, zero `sorry`, zero `sorryAx`, axioms `[propext, Classical.choice, Quot.sound]` only, every `[M]` card green; a census line of `0 %` on the bed a theorem gates demotes that theorem to decoration on that bed (L-DOM). **KILL** any `[M]` target not building by this gate ⇒ `[S]` (K-K of `bind_ledger.md`); the paper cites only declarations that build; **no arm trains before this gate is green** (L-LEAN).
- **licenses if green:** the `+` on Lean in the paper's §2 and §8 tables; P0 of `bind_ledger.md` closes. **dies if red:** the arena waits; nothing else.
- **price** `0 GPU-s` (CPU minutes for `lake build`) · **mechanism** P-11, L-LEAN, L-DOM / V-25, V-16 (a build that "passes" with a `sorry` is a pass that cannot fail) · **deliverable** `docs/LEAN_SHAPE_CENSUS.md`; `lean/CEQ/Shape/Census.lean` · **evenings** 1

## 3. Paper derivations owed (0 GPU-s; each a numbered proposition or remark in `docs/CEQ_SHAPE.md` §4–§5)

### J-D1 — Write the EMC remark with both halves and the ruling
- **phase** P0 · **prereq** J-L0 (notation) · **independent-of** every other card
- **what to build.** A remark, not a proposition (the judge's ruling, `judge/sec_shape.md` "Remark (EMC)", `bind_ledger.md` B-EMC KILL→deleted). *Half that holds:* for `γ < 1` and any row-stochastic `P'` — cyclic or not — `x ↦ V + γP'x` is a `γ`-contraction in `‖·‖_∞` (`Contraction.rowStochastic_perron` + `weighted_contraction`), so the post-intervention fixed point is unique and "settle then intervene then re-settle" equals "intervene then solve" (`RUN[J]` `8.9e-16` on a dense feedback `P`; `RUN[I]` `1.33e-15`; `RUN[F]` `4.4e-16`). *Half where Dash applies:* `dash-2005-emc` Thm 1 concerns a *reduced* model whose equilibrated form hides feedback through the manipulated variable; an explicit linear fixed point has no such gap; the case where it would reappear is `P'` recomputed from `z` (a DEQ, `bai-2019-deq`), which is out of scope. *What survives as a bind:* the triangularity plant (J-L4, J-L14): a non-causal `P` displaces backward (`0.0761 / 0.0868`, `RUN[M]`). *External argument for the re-solve:* `momennejad-2017-sr` / `russek-2017-predictive` (mechanism `[U]`): a cached resolvent does not adapt to a change of *transition* structure. Cite `mooij-2013-ode2scm`, `bongers-2021-cyclic`, `voortman-2010-manipulation` for the equilibrium-SCM lineage.
- **what to prove.** Uniqueness: four lines from the contraction (DERIVED). **what to measure.** The three RUNs above, re-run on BED-S's real draw through the front door (P2.4 of `bind_ledger.md`).
- **PASS** the remark carries the three numbers and no proposition number. **KILL** a planted feedback instance on the shape's own class reads `> 0.1` — impossible for `γ < 1` by the contraction; if it ever reads so, the algorithm is wrong (the "violation" the design filed was a suffix-only solve on a non-triangular `P`, V-3).
- **price** `0 GPU-s` · **mechanism** V-24 (an empty rejection region is declared, not hidden), D-7 (both halves filed), V-3 · **deliverable** `docs/CEQ_SHAPE.md` §4.2 Remark · **evenings** 1

### J-D2 — Derive reach-avoid sum-to-one with a sink, and the degeneracy lemma
- **phase** P0 · **prereq** J-D1's notation; states J-L15 · **independent-of** all Lean cards
- **what to build.** Proposition 8's two clauses in full: (i) with `𝒜 = 𝒜_sink ⊔ 𝒜_0 ⊔ ⊔_k 𝒜_k` exhausting the absorbing states and `ρ(Q) < 1`, `Σ_• q^{(•)} = 𝟙` on `T` — proof: `Σ_• R_• 𝟙 = 𝟙 − Q𝟙` (rows of `P` sum to `1`), so `N Σ_• R_• 𝟙 = N(I − Q)𝟙 = 𝟙`; (ii) if the constraint sets alone exhaust the absorbing states, `max_k q^{(k)} ≥ 1/K` by pigeonhole — "avoid every constraint" is unattainable; the planted instance is "BOS inside a constraint set" (`RUN[P]` `min_T max_k = 0.548718`). (iii) The discounted reads sum below one: `Σ_k E[γ^{τ−1} 𝟙_k] < 1` on `T` (`RUN[I]` `[0.227, 0.567]` read-form at `γ = 0.6`; z-form/read-form ratio exactly `γ`), so the "delay share" `1 − Σ_k` is printed beside every safest-move reading (`fisac-2019-bridging` owns the discounted safety value). Owners cited before the object: `kemeny-1976-finite`, `grinstead-1997-probability` Thm 11.6 `[U]`, `summers-2010-reach-avoid` (reach-avoid ≡ committor, the naming identity stated as a remark with both owners), `vanmoffaert-2013-chebyshev`, `misra-2023-safety-constrained-mdp` (Limits: Bellman optimality can fail on multichain CMDPs).
- **PASS** the census of `bind_ledger.md` X-5/X-6/X-7 reads the printed row on every batch. **KILL** a batch where the row fails by more than `1e-12` ⇒ a set was left undeclared (V-16: the solve must raise, not return).
- **price** `0 GPU-s` · **mechanism** V-12, V-8, V-23, V-25 · **deliverable** `docs/CEQ_SHAPE.md` §4.2 Prop. 8 · **evenings** 1

### J-D3 — Derive the `E[γ^τ]` identity without a probability space
- **phase** P0 · **prereq** J-L10's statement · **independent-of** all
- **what to build.** Proposition 3 in a finite-sum form the Lean tree can eventually carry: for `i ∈ T` and `V = 𝟙_{𝒜_k}`, `z_i = Σ_{t≥0} γ^t (P^t 𝟙_{𝒜_k})_i`, and because `𝒜_k` is absorbing `(P^t 𝟙_{𝒜_k})_i = Σ_{u≤t} f_i(u)` with `f_i(u) := ((P^u − P^{u−1}) 𝟙_{𝒜_k})_i ≥ 0` the first-passage mass at step `u`; exchanging sums gives `(1−γ) z_i = Σ_u γ^u f_i(u) =: E_i[γ^{τ_k} 𝟙_k]`, the read `O_i = (1−γ)(Pz)_i = Σ_u γ^{u−1} f_i(u)` by first-step analysis, and `lim_{γ↑1} O_i = Σ_u f_i(u) = q^{(k)}_i` by monotone convergence of a non-negative series. The gap bound: `q_i − E_i[γ^{τ}𝟙_k] = E_i[(1−γ^τ)𝟙_k] ≥ (1−γ) q_i` — small where `q_i` is (`RUN[J]` at `γ = 1−10^{-6}` on channel `{15}`: `max_T = 1.42e-7`; `refute_theory_math` item 8's universal `≥ 1−γ` is overruled by that RUN). At `i ∈ 𝒜_k` the read is `1`, not `γ^{-1}`. Owners: `kemeny-1976-finite`, `dayan-1993-successor`, `bellman-1957-markovian`, `doyle-1984-electric`, `metzner-2009-tpt-markov-jump`, `e-2006-transition-paths`.
- **what to measure.** `RUN[J]`/`RUN[M]` read vs `E[γ^{τ−1}]` to `8.33e-17`; BED-1 real sets `0.0`.
- **PASS** the derivation carries the `(1−γ) q_i` line and the two RUNs. **KILL** none; the identity is classical.
- **price** `0 GPU-s` · **mechanism** D-2 (the oracle runs on `P_env`), V-25, P-10 (theorem numbers inside `grinstead-1997-probability` stay `[U]`) · **deliverable** `docs/CEQ_SHAPE.md` §4.2 Prop. 3 · **evenings** 1

### J-D4 — Write the DET-class statement and define "depth-1"
- **phase** P0 · **prereq** none · **independent-of** all
- **what to build.** §5.3 of `judge/sec_obstructions.md` as a displayed statement: the exact committor solves `(I − Q) q = R𝟙`, the exact `z` is `(I − γP)^{-1}V`; inversion and iterated product are DET-complete, `NL ⊆ DET ⊆ NC²` (`cook-1985-taxonomy` `[V]`); a triangular inverse is DET-hard (iterated product embeds in `(I − A)^{-1}` for block sub-diagonal `A` — J-L11's `subdiag_pow_entry` is that embedding with scalar blocks); rational linear *equalities* are in `DET` and the P-complete problem is linear *inequalities* (`refute_theory_math` item 13). So the shape relocates the log-depth from the parameter stack into the linear solve: `O(s)` sequential rounds by substitution, or recursive block inversion at `O(log² s)` circuit depth / `O(log s)` matmul rounds with `O(s³)`-class work, exact in regime N and a certified Neumann truncation at `K = 2^k − 1` in regime S — `NOT MEASURED — needs a parallel-prefix kernel timing`. "Depth-1" means **one attention parameter set** (`P`, `γ`); the circuit-depth reading is disclaimed in the same paragraph. Beside it the skyline facts: `sanford-2024-logdepth` Thm 4.2 (unconditional, `⌊log₂ k⌋+2`), Cor. 4.3 (conditional), `chen-2024-multilayer` Thm 1.1 and `sanford-2024-inductionheads` Thm 1 (unconditional, vacuous at `s = 64, d = 16, p = 32`: `64^{1/16} = 1.2968`, `512 ≥ 64`), `peng-2024-transformer-limitations` Thm 1 (`384 < 544`), `merrill-2023-parallelism` Thm 2, `merrill-2025-littledepth` Thm 2, `yehudai-2025-depthwidth`, `merrill-2024-cot`, `kozachinskiy-2025-strassen` Thm 3.4; `wang-2024-incontext-td` / `xie-2026-softmax-rl` (abstract level `[U]`) for "the deeper stack computes the same resolvent by iteration". The `hop_k → committor` reduction is NOT FOUND (`sweep_expressivity.md` §3.5) and is listed as an open derivation (J-D8).
- **PASS** every obstruction sentence carries its theorem number, model class and vacuity inequality (census X-15). **KILL** any "cannot" sentence outside the skyline table (K-10 of `bind_ledger.md`).
- **price** `0 GPU-s` · **mechanism** P-8 / V-17 ("depth" in two units), P-10, V-25, P-3 · **deliverable** `docs/CEQ_SHAPE.md` §5.3 · **evenings** 1

### J-D5 — Specify the directed-stability `δ` for the influence barcode (restricted)
- **phase** P5 · **prereq** J-D4's notation; J-L6 (the `ε = 0` endpoint) · **independent-of** all Lean cards
- **what to build.** The barcode row is KILLED on BED-S (`β₀(ε = 0) ∈ {1, s}` on `100 %` of softmax draws; `beta0_interleaving` consumes point clouds, `READ ceq/certs/topological.py:476-505`) and survives only on gated corners with exact zeros (BED-M, `3 of 3`). What is owed before any barcode is read (`proposition_ledger.md` T-1): (i) the filtered object — the influence matrix `J_{ij} = ∂O_i/∂V_j = (Π_γ)_{ij} ≥ 0` (non-negative by J-L13; a *directed* weighted graph, not a metric); (ii) the connectivity notion — weak `β₀` of the digraph `{J > ε}` (path homology of `chowdhury-2017-path-homology` counts weak components at degree 0); (iii) the dissimilarity transform `d_{ij} := −log J_{ij}` (∞ on zeros) so that a Rips-type filtration of an *asymmetric* function is defined; (iv) the stability constant — `turner-2019-quasimetric-rips` gives bottleneck stability for asymmetric functions under sup-norm perturbation, `cohen-steiner-2007-stability` / `chazal-2012-structure-stability` do **not** apply (symmetric hypothesis); the `δ` printed is Turner's constant times `‖d − d'‖_∞`, where `d'` is the influence of the arm's `P̂` against the environment's — **`NOT MEASURED — needs the Turner constant evaluated on the shipped filtration`**; (v) the null — row-permuted influence at the same logit scale, `≥ 200` permutations, the X₃₅′ FAR calibration pattern (`READ V15_X35A_RESIDUAL.md:147-151`); (vi) the `ε = 0` endpoint must equal the F0 segmentation count by `torch.equal` (J-L6). Descriptive owners cited before the filtration is named: `kushnareva-2021-tda-attention`, `kushnareva-2022-betti`; intervention effects on persistence `kim-2026-topological-causal`.
- **PASS** the predicate (i)–(vi) is written before any barcode is computed. **KILL** measured barcode inside the null's `95 %` band on `≥ 6/8` seeds ⇒ decoration; non-integer or aliased `β₀` ⇒ refusal (V-16).
- **price** `0 GPU-s` for the spec; the instrument `NOT MEASURED` · **mechanism** M-15, M-2, V-16, V-3, V-25, P-4 · **deliverable** `docs/CEQ_SHAPE.md` §4.4(a) and `docs/PLAN.md` T-1 · **evenings** 1

### J-D6 — Restate the Cheeger line on a symmetrised surrogate
- **phase** P0 · **prereq** none · **independent-of** all
- **what to build.** `MATHEMATICS.md:455` (`g ≤ 2φ`) cites `levin-2017-markov-mixing` Thm 13.10, read this session by the methods sweep at p. 183: `Φ⋆²/2 ≤ γ_gap ≤ 2Φ⋆` for a **reversible** `P` with `γ_gap = 1 − λ₂` (`sinclair-1989-approximate`, `lawler-1988-bounds`; manifold original `cheeger-1970-lower`). The shape's causal `P` is not reversible (a lower-triangular `P` with `P_{i0} > 0` has `P_{0i} = 0`), so the line is V-25 as written. The restatement: (i) define the additive symmetrisation `S := (Π + Π^T)/2` with `Π` the stationary-weighted transition `π_i P_{ij}` — but a causal chain with a sink has no positive stationary law on `T`, so (ii) the honest surrogate is the **transient block's lazy symmetrisation** `Q_s := (Q + Q^T)/2` on `T`, whose Cheeger bound controls the spectral gap of `Q_s`, **not** `ρ(Q)`; the sentence the paper may write is: "a one-edge bridge of conductance `1/vol(S)` on the symmetrised surrogate bounds the surrogate's relaxation time below by `vol(S)/2` (`READ scale/e4_harmonic.py:247-263`, `1372.50` on E4′'s own 1,200-node case, a V-22 constant for BED-S)", and the quantity the hop ladder truncates is `ρ(Q_env)` on the DAG substrate, read off the ladder, never predicted from the surrogate's spectrum (`sec_apparatus.md` §A.2.1: `0.9964` vs `0.9985` on `LargestJoin_S2Rips_1024`). (iii) On the DAG substrate `Q` is lower-triangular and `ρ(Q) = max_T P_ii` (J-L10) — there is no bottleneck spectrum to bound; the Cheeger line applies to the **oracle cross-check chains** (BED-1, E4′) only and is moved there.
- **what to prove.** DERIVED; the reversibility hypothesis is quoted (one quotation, under 15 words, already spent by the sweep: "Φ⋆²/2 ≤ γ ≤ 2Φ⋆").
- **PASS** the restated line names the surrogate, the object bounded and the bed it applies to. **KILL** any `g ≤ 2φ` on a causal `P` without a stated symmetrisation (K-10-type; `sweep_topology.md` §6.6).
- **price** `0 GPU-s` · **mechanism** V-25, P-10, V-17 (the ergodic-chain gap is not the transient-block Perron root), V-22 · **deliverable** `docs/CEQ_SHAPE.md` §4.4 and `MATHEMATICS.md` §7 correction pointer · **evenings** 1

### J-D7 — Write the mask-amplification and certificate-units derivation
- **phase** P0 · **prereq** J-L12's statement · **independent-of** all
- **what to build.** Proposition 4(ii) in full with the `s = 3` closed-form instance (`P = [[1,0,0],[ε,1−ε,0],[0,1,0]]`, `γ = 0.9`, `ε = 0.1`, `V = e_0`, `P_{10}` dropped and row 1 renormalised: `‖O_full − O_mask‖_∞ = 0.5263157894736843`, `ε/(1−γ) = 1.000`, naive `ε = 0.1` violated `5.26×`, `RUN[J]`); the rule that every F1 union bound carries `1/(1−γ)`; the units rule (`δ·‖V‖_∞` with `‖V‖_∞` printed — the `sec_cost.md` §4.x.3 vector bound carried `‖V‖_∞ ≈ 5 [ASSUMED]`); the `β̂` census guard (`‖P̂‖_∞ > 1 ⇒` no certificate); and the sentence that the exact solve on an F1 mask is refused (fill-in, `sec_cost.md` §4.x.5). Dormant until a mask ships (D-4).
- **PASS** written with the instance. **KILL** none this round.
- **price** `0 GPU-s` · **mechanism** V-10, V-17, L-CERT, V-25 · **deliverable** `docs/CEQ_SHAPE.md` §4.2 Prop. 4(ii) · **evenings** 1

### J-D8 — Attempt the `hop_k → committor` reduction, or file it NOT FOUND
- **phase** P5 · **prereq** J-D4 · **independent-of** all
- **what to build.** One page: given a `hop_k` instance (`sanford-2024-logdepth` §2: `hop_k(X)_i = X_{find^k_X(i)}`), construct logits `qk` whose causal softmax `P` puts mass `1 − η` on the pointer target and `η` elsewhere, declare the `k`-th target set absorbing, and show the committor into it from `i` is `≥ (1−η)^k` while the committor into any other set is `≤ 1 − (1−η)^k`; then the argmax over sets reads `hop_k`. What blocks a theorem: the softmax row also puts mass on *itself* (`P_ii > 0`, J-L5) and on BOS (J-L2), so the walk can stall or fall to the sink — the construction must bound the stall mass, and the bound depends on the logit scale, which the record's arms learn (a trained `β̂`, Ruling 2a). If the page closes, obstruction 2 becomes a theorem *about the shape's task*; if not, it is filed NOT FOUND with the blocking step named, and obstruction 2 stays a lineage argument (`judge/sec_obstructions.md` §5.2).
- **PASS** either outcome written. **KILL** the reduction is stated without the stall-mass bound (V-3 / P-10).
- **price** `0 GPU-s` · **mechanism** V-3, D-2, P-10 · **deliverable** `docs/CEQ_SHAPE.md` §5.2 · **evenings** 2

### J-D9 — Write the boundary-null likelihood-ratio derivation for `γ̂`
- **phase** P0 · **prereq** none · **independent-of** all (feeds `bind_ledger.md` P4.3 / J-1)
- **what to build.** Ruling 10′ (`READ V17K_RULINGS.md:389-436`) restated for `γ ∈ [0, 1)` against `γ = 0`: the null sits on the parameter boundary, so `Λ = 2[LL(γ̂) − LL(0)]` is asymptotically `½χ²₀ + ½χ²₁`, whose `95 %` point is the `χ²₁` `0.90` quantile `2.705543` (`RUN[P]`), not `3.841459`; PINNED at `Λ ≤ 2.7055`, MOVED at `Λ > ln n_eval` (`ln 4096 = 8.318`), interval between; the `|γ̂| < 0.05` rule is deleted (two verdicts for one cell, M-20); `1/(1−γ̂)` printed beside every `δ`; the held-out set and the seed rule "MOVED on `≥ 6/8`" declared before data. The citation for the boundary null is **owed** — not in `references.bib`, not fetched this session; the derivation is written from the chi-bar-square mixture and marked `[U]` until the assembler fetches a primary (the paper may not cite it by name until then).
- **PASS** the instrument spec carries the null, the quantile with its producer, the held-out set and the seed count. **KILL** the threshold is chosen after `Λ` is seen (M-2).
- **price** `0 GPU-s` · **mechanism** M-2, V-17, M-20, V-9 · **deliverable** `docs/PLAN.md` J-1; `docs/CEQ_SHAPE.md` Prop. 9 · **evenings** 1

## 4. What dies if what (the mathematics only)

| if this fails | what dies | what survives |
|---|---|---|
| J-L1 refusal | the Lean rejection region of parity | Proposition 1 as RUN (`2.3003`) |
| J-L3 | the `[M]` grade of every triangular card (J-L4, J-L10, J-L13, J-L14) for one evening | the substitution as RUN (`1.8e-15`) |
| J-L6 | the F0 declaration the contract has cited since R15 | `torch.equal` zeros, `pathProd_eq_zero_iff` (in tree) |
| J-L10 (b-causal) on gated corners | the theorem form of "BOS declared ⇒ solvable" off the softmax corner | the per-batch `det` check (V-16 raise) |
| J-L12 | the Lean certificate | the textbook tail with the RUN plant; no arena cell (exact solve is cheaper than one hop at `s = 64`) |
| J-L15 | the LEAN conservation row | the printed residual per batch (V-23) |
| J-L18 gate red | training start (L-LEAN) | everything else; the arena waits |
| J-D8 does not close | obstruction 2 as a theorem about the shape | obstruction 2 as a lineage argument with vacuity lines |

Nothing in this file launches on Kaggle; the author's explicit yes is a node on the plan, not a formality (`READ kaggle/README.md:9-11`).

## Limits (collected once)

Every `RUN[P]` number is one numpy float64 draw at `s = 32`, seed 0, CPU, an identity or counterexample check, never a statistic. Grades are statements about statements: nothing was compiled this session, no `.lean` file was written, and every `[M]` is conditional on `lake build` at J-L18; the Mathlib name census covers the eight names listed in §0 at rev `a45ae637` and no other, and a card that introduces a further name is `[S]` until that name is checked. `grinstead-1997-probability`'s theorem numbers, the theorem numbers inside `sanford-2024-logdepth`, `chen-2024-multilayer`, `sanford-2024-inductionheads` and `peng-2024-transformer-limitations`, and the `wang-2024-incontext-td` / `xie-2026-softmax-rl` claim are the sweeps' readings and inherit their `[U]` marks; the boundary-null citation of J-D9 is owed and uncited; the Momennejad/Russek mechanism is `[U]`; ChaCAL's diagonal convention is `[V-fetched]` by one HTML read (`fagnou-2024-chacal`) and is re-read against the PDF before J-L1's refusal names it. The Turner constant (J-D5), the parallel-prefix timing (J-D4) and the `hop_k → committor` reduction (J-D8) are `NOT MEASURED` / NOT FOUND. Evening estimates are small integers with no producer other than the `V15Source.lean` scale (201 lines, 7 declarations) as the unit. No git write, no Kaggle contact, no code file was made by this planet.
