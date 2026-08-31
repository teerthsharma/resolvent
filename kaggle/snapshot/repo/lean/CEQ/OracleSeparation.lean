/-
  CEQ.OracleSeparation
  --------------------
  `oracle ≠ resolvent`, for the absorbing-chain oracle on a Rips graph.

  ## Why this file exists

  `LOOP_PROMPT.md` §1.1 and `STATE.md` open RED 18: `e1_anchor` is a *declared* rigged
  demo. Its label is the signed path sum `∑_j (∏_{k>j} a_k) b_j`, which is the last
  coordinate of `(I − A)⁻¹ b` for the strictly lower-triangular `A` that
  `ceq/nonnormal.py` builds — so an arm whose forward IS that resolvent is asked to
  reproduce its own forward, and every `e3_t*` rung inherits the same oracle
  (`scale/negation_scope.py:661-668`, keyed by oracle identity at
  `scale/m3_quintuple.py:102`). On that ladder only `settled − twin` carries credit.

  Round 8 proposes replacing the oracle with a coordinate of `B = N R`,
  `N = (I − Q)⁻¹`, for the transient block `Q` of a random walk on an E4′ Rips graph.
  The whole creditability of the new ladder rests on that oracle NOT being the arm's
  own forward. This file states that separation and proves it, rather than asserting
  it in prose where it would be one more unchecked claim.

  ## The separation, and what it turns on

  Not on the shape of the label, and not on the boundary values. On NILPOTENCY.

  * The arm's operator is strictly lower triangular — `ceq/attention.py` masks with
    `_strict_causal_mask`, `ceq/nonnormal.py` with `.tril(-1)` — so `A ^ n = 0`
    (`CEQ.Nilpotent.pow_card_eq_zero`). Its resolvent is a TERMINATING sum and
    `CEQ.Nilpotent.occupancy_is_exact_inverse` gives an exact finite inverse.
  * The oracle's operator is the transient block of a walk on an UNDIRECTED graph.
    Its entries are non-negative, its support is symmetric, and the merged component
    contains at least one transient edge. Such a matrix is never nilpotent: a walk
    that can step to a neighbour can step back, so some diagonal entry of every even
    power is strictly positive.

  Nilpotency is preserved by similarity, so this is not an artefact of how the tokens
  happen to be ordered: no reindexing of a causal operator produces `Q`. That
  invariance is stated here in prose and is NOT proved below — `oracle_ne_resolvent`
  is proved for the operator as given, and the permutation-conjugate version is left
  unproved rather than claimed.

  ## The second theorem is the one the `t*` ladder actually needs

  `truncation_never_exact` says the finite occupancy sum `∑_{k<N} Q^k` is not the
  inverse of `(1 − Q)` at ANY truncation `N`. For the arm's own operator the
  corresponding statement is false — it becomes an equality at `N = n`. So the two
  differ not merely as matrices but in the property the ladder is built to exploit:
  truncating the oracle's fixed point always leaves a residual, which is what makes
  `t* ∈ {1, 2, 8, 32}` a ladder at all rather than four copies of one exact answer.

  ## Hypotheses, and why each is the weakest that works

  `Nonneg` and `SymmSupport` are exactly what an undirected weighted graph gives:
  `Q = D⁻¹ W` with `W` symmetric and non-negative, and `D` a positive diagonal, so
  `Q i j > 0 ↔ W i j > 0 ↔ W j i > 0 ↔ Q j i > 0`. Symmetry of `Q` ITSELF is false
  whenever two transient nodes have different degrees, which is the generic case on a
  Rips graph, so assuming it would have been assuming something the corpus does not
  supply. `hij` is one positive transient-to-transient entry — one edge between two
  non-absorbing nodes — and without it the claim is false, since a `Q` that is
  identically zero IS nilpotent. `zero_not_a_counterexample` records that the
  hypothesis is doing work rather than decorating the statement.
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.RingTheory.Nilpotent
import CEQ.Nilpotent

namespace CEQ.OracleSeparation

open Matrix BigOperators

variable {n : ℕ}

/-- Entrywise non-negative. True of any sub-stochastic transition block. -/
def Nonneg (Q : Matrix (Fin n) (Fin n) ℝ) : Prop := ∀ i j, 0 ≤ Q i j

/-- Symmetric support: a step that is possible is possible in reverse.

    This is the undirected-graph hypothesis, and it is strictly weaker than symmetry
    of `Q`. The walk `D⁻¹ W` on an undirected graph satisfies it and is not symmetric. -/
def SymmSupport (Q : Matrix (Fin n) (Fin n) ℝ) : Prop := ∀ i j, 0 < Q i j → 0 < Q j i

theorem one_nonneg : Nonneg (1 : Matrix (Fin n) (Fin n) ℝ) := by
  intro i j
  rw [Matrix.one_apply]
  split
  · exact zero_le_one
  · exact le_refl 0

theorem mul_nonneg' {M N : Matrix (Fin n) (Fin n) ℝ} (hM : Nonneg M) (hN : Nonneg N) :
    Nonneg (M * N) := by
  intro i j
  rw [Matrix.mul_apply]
  exact Finset.sum_nonneg fun l _ => mul_nonneg (hM i l) (hN l j)

theorem pow_nonneg' {Q : Matrix (Fin n) (Fin n) ℝ} (hQ : Nonneg Q) (k : ℕ) :
    Nonneg (Q ^ k) := by
  induction k with
  | zero => simpa using (one_nonneg : Nonneg (1 : Matrix (Fin n) (Fin n) ℝ))
  | succ k ih => rw [pow_succ]; exact mul_nonneg' ih hQ

/-- **The doubling step.** For a non-negative `M`, the `(i, i)` entry of `M * M`
    dominates the single term `M i i * M i i`, because every other term in the sum is
    itself non-negative and so can only help. -/
theorem diag_sq_le {M : Matrix (Fin n) (Fin n) ℝ} (hM : Nonneg M) (i : Fin n) :
    M i i * M i i ≤ (M * M) i i := by
  rw [Matrix.mul_apply]
  exact Finset.single_le_sum (f := fun l => M i l * M l i)
    (fun l _ => mul_nonneg (hM i l) (hM l i)) (Finset.mem_univ i)

theorem diag_pos_double {M : Matrix (Fin n) (Fin n) ℝ} (hM : Nonneg M) {i : Fin n}
    (h : 0 < M i i) : 0 < (M * M) i i :=
  lt_of_lt_of_le (mul_pos h h) (diag_sq_le hM i)

/-- **A reversible step gives a positive diagonal at the square.**

    `(Q * Q) i i = ∑_l Q i l * Q l i ≥ Q i j * Q j i > 0`. This is the only place the
    symmetric-support hypothesis is used; everything after it is pure non-negativity. -/
theorem diag_sq_pos {Q : Matrix (Fin n) (Fin n) ℝ} (hQ : Nonneg Q)
    (hsupp : SymmSupport Q) {i j : Fin n} (hij : 0 < Q i j) : 0 < (Q * Q) i i := by
  rw [Matrix.mul_apply]
  refine lt_of_lt_of_le (mul_pos hij (hsupp i j hij)) ?_
  exact Finset.single_le_sum (f := fun l => Q i l * Q l i)
    (fun l _ => mul_nonneg (hQ i l) (hQ l i)) (Finset.mem_univ j)

/-- Every power of two beyond the first keeps that diagonal entry positive. -/
theorem diag_two_pow_pos {Q : Matrix (Fin n) (Fin n) ℝ} (hQ : Nonneg Q)
    (hsupp : SymmSupport Q) {i j : Fin n} (hij : 0 < Q i j) :
    ∀ k : ℕ, 0 < (Q ^ ((2 : ℕ) ^ (k + 1))) i i := by
  intro k
  induction k with
  | zero =>
      have h2 : (2 : ℕ) ^ (0 + 1) = 2 := by norm_num
      rw [h2, pow_two]
      exact diag_sq_pos hQ hsupp hij
  | succ k ih =>
      have hsplit : (2 : ℕ) ^ (k + 1 + 1) = 2 ^ (k + 1) + 2 ^ (k + 1) := by
        rw [pow_succ]; ring
      rw [hsplit, pow_add]
      exact diag_pos_double (pow_nonneg' hQ ((2 : ℕ) ^ (k + 1))) ih

/-- **The oracle's operator is not nilpotent.**

    A single transient-to-transient edge is enough. The proof is the doubling above
    plus `n < 2 ^ n`: any claimed nilpotency index `N` is dominated by some power of
    two, and a matrix power that vanishes forces every larger power to vanish. -/
theorem not_isNilpotent {Q : Matrix (Fin n) (Fin n) ℝ} (hQ : Nonneg Q)
    (hsupp : SymmSupport Q) {i j : Fin n} (hij : 0 < Q i j) : ¬ IsNilpotent Q := by
  rintro ⟨N, hN⟩
  have hlt : N < 2 ^ (N + 1) := lt_of_lt_of_le (Nat.lt_two_pow N)
    (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ N))
  have hsplit : 2 ^ (N + 1) = N + (2 ^ (N + 1) - N) := by omega
  have hzero : (Q ^ ((2 : ℕ) ^ (N + 1))) = 0 := by
    rw [hsplit, pow_add, hN, zero_mul]
  have hpos := diag_two_pow_pos hQ hsupp hij N
  rw [hzero, Matrix.zero_apply] at hpos
  exact lt_irrefl 0 hpos

/-- **`oracle ≠ resolvent`.**

    The absorbing chain's transient block is not the arm's causal operator. Stated as
    a disequality of matrices, which is the strongest form the round needs: it rules
    out the coincidence that makes `e1_anchor` a rigged demo, at the operator level,
    before any label is drawn. -/
theorem oracle_ne_resolvent (A Q : Matrix (Fin n) (Fin n) ℝ)
    (hA : CEQ.Nilpotent.StrictlyLower A) (hQ : Nonneg Q) (hsupp : SymmSupport Q)
    {i j : Fin n} (hij : 0 < Q i j) : Q ≠ A := by
  intro hEq
  subst hEq
  exact not_isNilpotent hQ hsupp hij ⟨n, CEQ.Nilpotent.pow_card_eq_zero hA⟩

/-- **Truncating the oracle's fixed point is never exact, at any budget.**

    `CEQ.Occupancy.occupancy_telescope` gives `(1 − Q) · ∑_{k<N} Q^k = 1 − Q^N`, so
    the truncated occupancy operator is the exact inverse precisely when `Q^N = 0`.
    For the arm's own operator that happens at `N = n`; for the oracle's it never
    happens. This is the statement the `t*` ladder rests on: the residual at
    `t* ∈ {1, 2, 8, 32}` is real at every rung, so the rungs measure something. -/
theorem truncation_never_exact {Q : Matrix (Fin n) (Fin n) ℝ} (hQ : Nonneg Q)
    (hsupp : SymmSupport Q) {i j : Fin n} (hij : 0 < Q i j) (N : ℕ) :
    (1 - Q) * CEQ.Occupancy.occupancy Q N ≠ 1 := by
  rw [CEQ.Occupancy.occupancy_telescope]
  intro hEq
  exact not_isNilpotent hQ hsupp hij ⟨N, sub_eq_self.mp hEq⟩

/-- **The edge hypothesis is load-bearing, not decoration.**

    Drop `hij` and the theorem is false: the zero matrix is non-negative, has
    symmetric support, and is nilpotent. Fourteen vacuous controls have been struck in
    this project; a hypothesis whose removal leaves the statement true is the same
    defect in proof form, so the witness is recorded rather than argued. -/
theorem zero_not_a_counterexample :
    Nonneg (0 : Matrix (Fin n) (Fin n) ℝ) ∧ SymmSupport (0 : Matrix (Fin n) (Fin n) ℝ)
    ∧ IsNilpotent (0 : Matrix (Fin n) (Fin n) ℝ) := by
  refine ⟨fun _ _ => by simp, fun _ _ h => by simp at h, ⟨1, by simp⟩⟩

/-- The other side of the separation, recorded so both halves are checked here rather
    than one being checked and the other remembered: the arm's operator IS nilpotent,
    which is what `oracle_ne_resolvent` plays it off against. -/
theorem resolvent_operator_isNilpotent {A : Matrix (Fin n) (Fin n) ℝ}
    (hA : CEQ.Nilpotent.StrictlyLower A) : IsNilpotent A :=
  ⟨n, CEQ.Nilpotent.pow_card_eq_zero hA⟩

end CEQ.OracleSeparation
