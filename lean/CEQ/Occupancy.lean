/-
  CEQ.Occupancy
  -------------
  Provenance: THEORY.md §1, the successor-representation identification.

  Citation correction, verified by Wilson against the primary source. Dayan 1993,
  "Improving Generalization for Temporal Difference Learning: The Successor
  Representation", Neural Computation 5(4):613–624, prints the successor
  representation as

      [x_i]_j = [I]_ij + [Q]_ij + [Q²]_ij + … = [(I − Q)⁻¹]_ij      (eq. 3.1)

  with **no γ**, where `Q` is the transition matrix among the non-absorbing states of a
  finite *absorbing* Markov chain. Invertibility comes from the absorbing structure,
  not from a discount factor. The familiar `M = (I − γP)⁻¹` with `P` row-stochastic and
  `γ < 1` is later-literature notation. THEORY.md cited the second while attributing it
  to the first; the two are kept distinct here.

  What is proved is the finite telescoping identity — the algebraic core of the
  "resolvent equals discounted occupancy" reading, and what makes truncated occupancy
  computable. Stated over an arbitrary ring rather than over matrices specifically:
  strictly more general, and it instantiates at `Matrix n n ℝ` for `Fintype n` and
  `DecidableEq n` without further work.

  The infinite-series limit is deliberately not proved. It needs a convergence
  hypothesis, which `CEQ.Contraction.weighted_contraction` supplies in the weighted sup
  norm. Weakening to the provable finite statement follows the convention already used
  in `Epsilon-Hollow`'s `AetherVerified`.
-/

import Mathlib.Algebra.GeomSum
import Mathlib.Data.Matrix.Basic

namespace CEQ.Occupancy

open BigOperators

variable {R : Type*} [Ring R]

/-- The truncated discounted occupancy operator `∑_{k<N} A^k`.
    With `A = γ • P` this is expected discounted occupancy over a horizon `N`. -/
def occupancy (A : R) (N : ℕ) : R := ∑ k in Finset.range N, A ^ k

/-- **Finite Neumann / telescoping identity.**

    `(1 − A) · ∑_{k<N} A^k = 1 − A^N`.

    The truncated occupancy operator is an exact inverse of `(1 − A)` up to the tail
    `A^N`. When the operator contracts, the tail vanishes and the limit is `(1 − A)⁻¹`;
    `CEQ.Contraction.weighted_contraction` is exactly that hypothesis, in the weighted
    sup norm. -/
theorem occupancy_telescope (A : R) (N : ℕ) :
    (1 - A) * occupancy A N = 1 - A ^ N := by
  unfold occupancy
  induction N with
  | zero => simp
  | succ k ih =>
    -- `A` commutes with its own powers; a general ring is not commutative, so this
    -- step has to be supplied rather than assumed by `ring`.
    have hc : A * A ^ k = A ^ k * A := ((Commute.refl A).pow_right k).eq
    rw [Finset.sum_range_succ, mul_add, ih, pow_succ, sub_mul, one_mul, hc]
    abel

/-- The same identity on the other side. No commutation step is needed here: the
    occupancy sum multiplies `(1 − A)` from the left, so `A ^ k * A` appears directly. -/
theorem occupancy_telescope' (A : R) (N : ℕ) :
    occupancy A N * (1 - A) = 1 - A ^ N := by
  unfold occupancy
  induction N with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, add_mul, ih, pow_succ, mul_sub, mul_one]
    abel

/-- **The occupancy reading of the resolvent, finite and exact.**

    If `A^N = 0` — the exact finite-horizon case, and Dayan's situation once every
    trajectory of the absorbing chain has been absorbed — then truncated occupancy *is*
    the two-sided inverse of `(1 − A)`, with no limit argument required.

    This is the honest finite statement of "the resolvent is expected future occupancy".
    Anything beyond it needs a convergence hypothesis. -/
theorem occupancy_eq_inverse_of_nilpotent (A : R) (N : ℕ) (hA : A ^ N = 0) :
    (1 - A) * occupancy A N = 1 ∧ occupancy A N * (1 - A) = 1 := by
  constructor
  · rw [occupancy_telescope, hA, sub_zero]
  · rw [occupancy_telescope', hA, sub_zero]

end CEQ.Occupancy
