/-
  CEQ.Nilpotent
  -------------
  Provenance: backs `ceq/nonnormal.py`, whose `causal_mask` is `.tril(-1)`.

  `CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent` proves that when `A ^ N = 0`
  the truncated occupancy sum is the exact two-sided inverse of `(1 - A)` — no
  truncation error, no convergence hypothesis, no norm. What that theorem could
  not supply is a reason to believe any particular operator satisfies its
  hypothesis.

  This file supplies it. A strictly lower-triangular matrix — the diagonal
  EXCLUDED — is nilpotent at the cardinality of its index type.

  Why the diagonal must be excluded, and why this is the load-bearing detail:
  with the diagonal included the operator has a self-loop, `rho(A)` is the
  largest diagonal entry rather than `0`, and the resolvent stops being a finite
  sum. `tests/w3b` checks that the Python side really is `.tril(-1)` and not
  `.tril(0)`, because a proof about a different matrix than the one that ships is
  worse than no proof.

  The argument is a path-counting one. `A i j` can be nonzero only when `j < i`,
  so a nonzero entry of `A ^ k` needs a chain of `k` strictly increasing indices
  and therefore `j + k <= i`. At `k = n` that asks for an index at least `n`,
  which `Fin n` does not have.

  Measured counterpart, for comparison rather than as evidence: `tests/w2`
  reports `rho(A) < 1e-12` and resolvent residual `< 1e-12` at float64. That is a
  measurement. This is a proof.
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import CEQ.Occupancy

namespace CEQ.Nilpotent

open Matrix BigOperators

variable {n : ℕ} {R : Type*} [CommRing R]

/-- Strictly lower triangular: `A i j` vanishes unless `j < i`.

    Stated as "vanishes when `i ≤ j`" so the diagonal is covered by the
    hypothesis rather than by a side condition. -/
def StrictlyLower (A : Matrix (Fin n) (Fin n) R) : Prop :=
  ∀ i j : Fin n, (i : ℕ) ≤ (j : ℕ) → A i j = 0

/-- **Path-length bound.** A nonzero entry of `A ^ k` requires a strictly
    increasing chain of `k` indices from `j` up to `i`, hence `j + k ≤ i`.
    Contrapositive: `i < j + k` forces the entry to vanish. -/
theorem pow_entry_zero {A : Matrix (Fin n) (Fin n) R} (hA : StrictlyLower A) :
    ∀ (k : ℕ) (i j : Fin n), (i : ℕ) < (j : ℕ) + k → (A ^ k) i j = 0 := by
  intro k
  induction k with
  | zero =>
      intro i j h
      simp only [Nat.add_zero] at h
      rw [pow_zero, Matrix.one_apply_ne (Fin.ne_of_val_ne (Nat.ne_of_lt h))]
  | succ k ih =>
      intro i j h
      rw [pow_succ, Matrix.mul_apply]
      refine Finset.sum_eq_zero ?_
      intro l _
      by_cases hl : (l : ℕ) ≤ (j : ℕ)
      · -- the last hop `l → j` is on or above the diagonal, so `A l j = 0`
        rw [hA l j hl, mul_zero]
      · -- otherwise `j < l`, and the remaining `k` hops cannot reach `i` either
        push_neg at hl
        rw [ih i l (by omega), zero_mul]

/-- **Nilpotency at the cardinality.** `A ^ n = 0` for strictly lower-triangular
    `A : Matrix (Fin n) (Fin n) R`.

    Every index of `Fin n` is `< n`, so `i < j + n` holds for every pair and
    `pow_entry_zero` applies everywhere at once. -/
theorem pow_card_eq_zero {A : Matrix (Fin n) (Fin n) R} (hA : StrictlyLower A) :
    A ^ n = 0 := by
  ext i j
  rw [Matrix.zero_apply]
  exact pow_entry_zero hA n i j (by have := i.isLt; omega)

/-- **The resolvent of a strictly causal operator is a terminating finite sum,
    and it is exact.**

    Discharges the hypothesis of `CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent`
    for the operator `ceq/nonnormal.py` actually builds. Two consequences the
    numerical version does not get:

    * No contraction certificate is consulted. `rho(A) = 0` holds structurally,
      so the reducible-`A` failure mode — power iteration returning `w` with exact
      zero entries, leaving the weighted sup norm undefined — cannot arise,
      because no `w` exists in this statement.
    * No truncation error. `CEQ.Contraction.weighted_contraction` bounds a tail;
      here the tail is identically zero. -/
theorem occupancy_is_exact_inverse
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : StrictlyLower A) :
    (1 - A) * CEQ.Occupancy.occupancy A n = 1 ∧
    CEQ.Occupancy.occupancy A n * (1 - A) = 1 :=
  CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent A n (pow_card_eq_zero hA)

/-- The diagonal really is what matters: the identity is lower triangular in the
    non-strict sense and is not nilpotent at any power. Recorded so the strictness
    hypothesis cannot be quietly weakened later. -/
theorem one_not_nilpotent (hn : 0 < n) (k : ℕ) :
    (1 : Matrix (Fin n) (Fin n) ℝ) ^ k ≠ 0 := by
  rw [one_pow]
  intro h
  have hz : (1 : Matrix (Fin n) (Fin n) ℝ) ⟨0, hn⟩ ⟨0, hn⟩
          = (0 : Matrix (Fin n) (Fin n) ℝ) ⟨0, hn⟩ ⟨0, hn⟩ := by rw [h]
  rw [Matrix.one_apply_eq, Matrix.zero_apply] at hz
  exact one_ne_zero hz

end CEQ.Nilpotent
