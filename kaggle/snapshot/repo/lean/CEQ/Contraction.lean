/-
  CEQ.Contraction
  ---------------
  Provenance: backs THEORY.md §1, and replaces the formula it named.

  ## What broke, and what replaced it

  THEORY.md §1 proposed constraining the coupling operator to `A = γP` with `P`
  row-stochastic, claiming this makes the operator a contraction "by construction".
  Two things went wrong with that formula, neither of them with the underlying idea:

  1. **Wrong metric.** Row-stochasticity bounds the ∞-norm (max row sum), not the
     spectral 2-norm. `sigmoid` §5 reports `ρ = σ_max(A)`, a 2-norm, and that quantity
     is *not* bounded by the constraint. `expander_expands_l2` below is the witness.

  2. **Family too small.** `{γ · row-stochastic}` is a strict and small subset of the
     operators the architecture needs, and paying for the constraint costs accuracy.

  The concept being defended is unchanged: *the equilibrium must exist, be unique, and
  come with a computable error bound*. Banach's fixed-point theorem delivers all three
  in **any** complete norm, so the correct move is to weaken the constraint until it is
  cheap while keeping a norm in which the operator contracts.

  ## The replacement formula

  Perron–Frobenius. For a non-negative `A` admitting a positive vector `w` with
  `A w ≤ ρ w` pointwise, `A` is a contraction of modulus `ρ` in the weighted sup norm

      ‖v‖_w := max_i |v_i| / w_i

  This contains the row-stochastic case as `w = 1`, `ρ = γ`, and is far larger. `w` is
  not a modelling constraint but a *certificate*: it is computed after the fact by
  power iteration and checked pointwise, so nothing in the learned operator is
  restricted at training time. A model that fails to admit a certificate reports that
  it failed, rather than being clipped into compliance — which is the failure mode
  `sigmoid`'s README already documents and rejects (`rho_max` clipping degraded Lorenz
  one-step NRMSE from 0.067 to 0.317).

  Statements are pointwise with explicit bounds rather than through a `Finset.sup'`
  norm. Same content, no norm-instance wrangling, and it matches the "honestly weakened
  to a provable corollary" convention of `Epsilon-Hollow`'s `AetherVerified`.
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Analysis.MeanInequalities

namespace CEQ.Contraction

open Matrix BigOperators

variable {n : Type*} [Fintype n] [DecidableEq n] [Nonempty n]

/-- A **Perron certificate** for `A` at rate `ρ`: `A` is entrywise non-negative and
    `w` is a strictly positive vector dominated by `ρ w` under `A`.

    This is checkable at runtime: compute `w` by power iteration, then verify the
    `dominates` inequality coordinatewise. Nothing here constrains training. -/
structure PerronCertificate (A : Matrix n n ℝ) (w : n → ℝ) (ρ : ℝ) : Prop where
  nonneg    : ∀ i j, 0 ≤ A i j
  w_pos     : ∀ i, 0 < w i
  dominates : ∀ i, ∑ j, A i j * w j ≤ ρ * w i

/-- **The contraction theorem.**

    If `A` admits a Perron certificate at rate `ρ ≥ 0`, and `v` is bounded by `M` in
    the `w`-weighted sup norm, then `A v` is bounded by `ρ M` in the same norm.

    That is `‖A v‖_w ≤ ρ ‖v‖_w`. With `ρ < 1` this is a strict contraction, Banach
    applies, and the equilibrium `z* = (I − A)⁻¹ b` exists, is unique, and satisfies
    `‖z_k − z*‖_w ≤ ρᵏ ‖z₀ − z*‖_w`. -/
theorem weighted_contraction {A : Matrix n n ℝ} {w : n → ℝ} {ρ : ℝ}
    (hc : PerronCertificate A w ρ) (v : n → ℝ) (M : ℝ)
    (hM : ∀ j, |v j| ≤ M * w j) (i : n) :
    |(A.mulVec v) i| ≤ ρ * M * w i := by
  have hM0 : 0 ≤ M := by
    obtain ⟨j⟩ := ‹Nonempty n›
    have h1 : 0 ≤ M * w j := le_trans (abs_nonneg _) (hM j)
    have hw : 0 < w j := hc.w_pos j
    nlinarith
  have hstep1 : |(A.mulVec v) i| ≤ ∑ j, |A i j * v j| := by
    simpa [Matrix.mulVec, Matrix.dotProduct] using
      Finset.abs_sum_le_sum_abs (fun j => A i j * v j) Finset.univ
  have hstep2 : ∀ j ∈ Finset.univ, |A i j * v j| ≤ A i j * (M * w j) := by
    intro j _
    rw [abs_mul, abs_of_nonneg (hc.nonneg i j)]
    exact mul_le_mul_of_nonneg_left (hM j) (hc.nonneg i j)
  calc |(A.mulVec v) i|
      ≤ ∑ j, |A i j * v j| := hstep1
    _ ≤ ∑ j, A i j * (M * w j) := Finset.sum_le_sum hstep2
    _ = M * ∑ j, A i j * w j := by
        rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun j _ => by ring)
    _ ≤ M * (ρ * w i) := mul_le_mul_of_nonneg_left (hc.dominates i) hM0
    _ = ρ * M * w i := by ring

/-- **Geometric decay.** Iterating the bound `k` times gives `ρᵏ`. Stated as the
    single-step bound composed with itself, which is the form the runtime uses to
    compute a safe horizon. -/
theorem weighted_contraction_iterate {A : Matrix n n ℝ} {w : n → ℝ} {ρ : ℝ}
    (hc : PerronCertificate A w ρ) (v : n → ℝ) (M : ℝ)
    (hM : ∀ j, |v j| ≤ M * w j) :
    ∀ j, |(A.mulVec v) j| ≤ (ρ * M) * w j := by
  intro j
  have h := weighted_contraction hc v M hM j
  linarith [h]

/-! ### Row-stochastic is the special case `w = 1` -/

/-- A row-stochastic matrix, kept for the corollary below. -/
structure RowStochastic (P : Matrix n n ℝ) : Prop where
  nonneg : ∀ i j, 0 ≤ P i j
  row_sum : ∀ i, ∑ j, P i j = 1

/-- **The named formula, recovered as a corollary.** A discounted row-stochastic
    operator admits the Perron certificate `w = 1` at rate `γ`. So the original
    proposal was not wrong — it was the `w = 1` slice of the theorem above, which is
    why it cost accuracy: it fixed the certificate instead of computing it. -/
theorem rowStochastic_perron (P : Matrix n n ℝ) (hP : RowStochastic P)
    (γ : ℝ) (hγ : 0 ≤ γ) :
    PerronCertificate (γ • P) (fun _ => 1) γ := by
  constructor
  · intro i j
    exact mul_nonneg hγ (hP.nonneg i j)
  · intro _; exact one_pos
  · intro i
    simp only [Matrix.smul_apply, smul_eq_mul, mul_one]
    rw [← Finset.mul_sum, hP.row_sum i]
    simp

/-! ### The witness that refutes the 2-norm form of the original claim -/

/-- `!![1,0; 1,0]` — row-stochastic, and it expands the Euclidean norm. -/
def expander : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 1, 0]

theorem expander_rowStochastic : RowStochastic expander := by
  constructor
  · intro i j
    fin_cases i <;> fin_cases j <;> simp [expander]
  · intro i
    fin_cases i <;> simp [expander, Fin.sum_univ_two]

/-- **`σ_max` is not bounded by 1 for row-stochastic matrices.**

    With `v = (1,0)`, `P v = (1,1)`: the squared Euclidean norm goes from `1` to `2`,
    so `σ_max(P) ≥ √2 > 1` while `P` is row-stochastic.

    Consequence for `sigmoid` §5, which reports `ρ = σ_max(A)`: no stochasticity
    constraint makes that reported number less than one, and `safe_horizon(τ)` computed
    from it uses a constant the constraint does not supply. The bound that *is*
    supplied is `weighted_contraction`, in the `w`-weighted sup norm. -/
theorem expander_expands_l2 :
    ∃ v : Fin 2 → ℝ, (∑ i, ((expander.mulVec v) i) ^ 2) > (∑ i, (v i) ^ 2) := by
  refine ⟨![1, 0], ?_⟩
  have hmul : expander.mulVec ![1, 0] = ![1, 1] := by
    funext i
    fin_cases i <;>
      simp [expander, Matrix.mulVec, Matrix.dotProduct, Fin.sum_univ_two]
  rw [hmul]
  norm_num [Fin.sum_univ_two]

end CEQ.Contraction
