/-
  CEQ.V15Source — LEAN #15, `resolvent_inverse_is_difference`
  -----------------------------------------------------------
  Provenance: `CEQ_V15_2_DELTA.md` §LEAN #15. Backs `ceq/x35p/source.py`
  (instrument (a), the exact source solve) and `V15_X35P_SOURCE.md`.

  WHAT IS REFUSED AS THE STATEMENT.

  The delta calls the theorem trivial, and the trivial reading of it is
  `M * M⁻¹ = 1`. That reading is recorded below as `inverse_identity_is_vacuous`
  and then not used: it holds for every unit of every ring, so it separates this
  architecture from no other, and a file whose main theorem is true of every
  invertible matrix has proved nothing about a carrier. `V15_N3_LEAN.md`'s
  standard — trivial statements refused explicitly rather than shipped quietly —
  is why it appears at all.

  WHAT IS PROVED INSTEAD.

  The content is not that an inverse exists. It is WHICH operator the inverse
  is: for the strictly causal carrier `ceq/nonnormal.py` builds, the forward map
  is the terminating occupancy sum `W = ∑_{k<n} Aᵏ` — every path of every length
  — and its inverse is `1 - A`, ONE application of `A` and no powers. So the
  inverse problem on this architecture has a closed form:

      source = residual − A · residual

  a first-order difference. No iteration, no regularization, no conditioning
  question, because nothing is solved. Three theorems carry that, in the order
  the claim needs them:

  | theorem                            | what it says |
  |------------------------------------|--------------|
  | `source_is_first_order_difference` | `r = W ·ᵥ h` ⇒ `h = r − A ·ᵥ r`, exactly, with `A` strictly lower triangular the only hypothesis |
  | `no_fill_in`                       | every off-diagonal zero of `A` is a zero of `1 - A`: the inverse is as sparse as the hop, plus a diagonal |
  | `forward_map_fills_in`             | and the forward map is NOT — a witness where `A 2 0 = 0` while `(∑_{k<3} Aᵏ) 2 0 = 1` |

  The second and third together are the sharp form of "first order". Sparsity of
  an inverse is a claim only against something denser; `forward_map_fills_in` is
  that something, and it is why `ceq/x35p/source.py::exact_source` takes `A` and
  never takes `W`. Measured counterpart on the shipped carrier, for comparison
  and not as evidence: `nnz(1 - A) = 251` against `nnz(W) = 1703` at `n = 128`,
  `d = 5` (`tests/x35p/test_source.py`). That is a measurement. This is a proof.

  WHAT IS NOT RE-DERIVED. `CEQ.Occupancy.occupancy_eq_inverse_of_nilpotent` and
  `CEQ.Nilpotent.occupancy_is_exact_inverse` already prove that the truncated
  occupancy IS the two-sided inverse when `Aⁿ = 0`, and
  `CEQ.Nilpotent.pow_card_eq_zero` already discharges that hypothesis for a
  strictly lower-triangular matrix. This file composes with both and restates
  neither; `source_is_first_order_difference` is the vector-level consequence
  that the operator-level theorem does not state, and it is the one the
  instrument actually applies.

  WHAT IS NOT CLAIMED. Nothing here says the closed form is the estimator to
  USE. It is exact on `r = W ·ᵥ h` and says nothing about `r = W ·ᵥ h + noise`,
  where `1 - A` differences the noise; `V15_X35P_SOURCE.md` measures that and
  retires the exact inverse to the noiseless regime on the measurement. A
  theorem about the noiseless map is not a claim about the noisy one, and this
  file does not make one.

  No `sorry`. `#print axioms` on every theorem at the foot of the file.
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import CEQ.Occupancy
import CEQ.Nilpotent

namespace CEQ.V15Source

open Matrix BigOperators
open CEQ.Occupancy CEQ.Nilpotent

variable {n : ℕ}

/-! ## The reading that is refused -/

/-- **REFUSED as the statement of Lean #15.** `M · M⁻¹ = 1` holds for every unit
    of every ring. It is recorded here so that it is visibly not what any
    theorem below says: a statement true of every invertible matrix licenses
    nothing about the resolvent carrier, and in particular licenses no claim
    about the COST or the SPARSITY of the inverse, which is the whole of the
    delta's argument for the carrier. -/
theorem inverse_identity_is_vacuous {R : Type*} [Ring R] (u : Rˣ) :
    (u : R) * ((u⁻¹ : Rˣ) : R) = 1 := u.mul_inv

/-! ## The subtraction lemma -/

/-- **LEAN #15.** If the residual field `r` is what the resolvent carrier
    produced from an unobserved source `h`, then `h` is recovered from `r` by a
    **first-order difference**:

        `r = (∑_{k<n} Aᵏ) ·ᵥ h`   ⇒   `h = r − A ·ᵥ r`,

    exactly, for any strictly lower-triangular `A`.

    The forward map is the occupancy sum `CEQ.Occupancy.occupancy A n`, which
    `CEQ.Nilpotent.occupancy_is_exact_inverse` proves is the exact two-sided
    inverse of `1 - A` — no truncation error, no convergence hypothesis, no
    norm, because strict causality makes `Aⁿ = 0` structurally
    (`CEQ.Nilpotent.pow_card_eq_zero`). Only the hypothesis
    `StrictlyLower A` is assumed here; everything else is composed.

    Stated with the right-hand side as `r − A ·ᵥ r` rather than as
    `(1 - A) ·ᵥ r` on purpose. The two are equal by `Matrix.sub_mulVec`, and the
    first is the form in which the claim is legible: the source solve is one
    mat-vec against the hop and a subtraction, so it costs `O(nnz A)` and there
    is no linear system anywhere in it. -/
theorem source_is_first_order_difference
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : StrictlyLower A) (h r : Fin n → ℝ)
    (hfwd : r = CEQ.Occupancy.occupancy A n *ᵥ h) :
    h = r - A *ᵥ r := by
  have hinv : (1 - A) * CEQ.Occupancy.occupancy A n = 1 :=
    (occupancy_is_exact_inverse hA).1
  calc h = (1 : Matrix (Fin n) (Fin n) ℝ) *ᵥ h := (Matrix.one_mulVec h).symm
    _ = ((1 - A) * CEQ.Occupancy.occupancy A n) *ᵥ h := by rw [hinv]
    _ = (1 - A) *ᵥ (CEQ.Occupancy.occupancy A n *ᵥ h) :=
          (Matrix.mulVec_mulVec h (1 - A) (CEQ.Occupancy.occupancy A n)).symm
    _ = (1 - A) *ᵥ r := by rw [hfwd]
    _ = r - A *ᵥ r := by rw [Matrix.sub_mulVec, Matrix.one_mulVec]

/-- **Superposition, and why the two-source must-fire costs nothing.**

    The forward map is linear, so a field driven by two sources is the sum of
    the two fields; applying the closed form to it returns the sum of the two
    sources, exactly, with no separation step and no assumption that the sources
    are disjoint in support. `tests/x35p/test_source.py`'s superposition
    must-fire is this theorem with numbers in it. -/
theorem two_sources_recovered
    {A : Matrix (Fin n) (Fin n) ℝ} (hA : StrictlyLower A) (h₁ h₂ r : Fin n → ℝ)
    (hfwd : r = CEQ.Occupancy.occupancy A n *ᵥ h₁ + CEQ.Occupancy.occupancy A n *ᵥ h₂) :
    h₁ + h₂ = r - A *ᵥ r :=
  source_is_first_order_difference hA (h₁ + h₂) r
    (by rw [hfwd, Matrix.mulVec_add])

/-! ## Why "first order" is a claim and not a phrasing -/

/-- **No fill-in.** Every off-diagonal zero of the hop is a zero of the inverse,
    so the inverse's sparsity pattern is the hop's plus a diagonal — at most
    `nnz A + n` entries, however long the paths through `A` are.

    Stated over an arbitrary ring: it is a fact about `1 - A`, and needs neither
    causality nor ℝ. -/
theorem no_fill_in {R : Type*} [Ring R] {A : Matrix (Fin n) (Fin n) R}
    {i j : Fin n} (hij : i ≠ j) (hA : A i j = 0) : (1 - A) i j = 0 := by
  simp [Matrix.sub_apply, Matrix.one_apply_ne hij, hA]

/-- The unit sub-diagonal shift on `Fin 3` — one hop, one nonzero per row. -/
def shift3 : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of fun i j => if (i : ℕ) = (j : ℕ) + 1 then 1 else 0

@[simp] lemma shift3_apply (i j : Fin 3) :
    shift3 i j = if (i : ℕ) = (j : ℕ) + 1 then (1 : ℝ) else 0 := rfl

lemma shift3_strictlyLower : StrictlyLower shift3 := by
  intro i j hle
  rw [shift3_apply, if_neg (by omega)]

/-- **And the forward map is not sparse.** A witness in which the hop has a zero
    that the resolvent does not: `shift3 2 0 = 0`, so `no_fill_in` gives
    `(1 - shift3) 2 0 = 0`, while the forward map `∑_{k<3} shift3ᵏ` is nonzero
    there — the length-2 path `0 → 1 → 2` that `shift3²` counts.

    This is what makes `source_is_first_order_difference` a claim rather than a
    phrasing. "The inverse is sparse" says nothing until there is something
    denser to be sparse against, and the thing it is sparse against is the
    forward map it inverts. On the shipped carrier the same asymmetry is
    `nnz(1 - A) = 251` against `nnz(W) = 1703`; here it is a theorem. -/
theorem forward_map_fills_in :
    StrictlyLower shift3 ∧ shift3 2 0 = 0 ∧ (1 - shift3) 2 0 = 0 ∧
      (CEQ.Occupancy.occupancy shift3 3) 2 0 = 1 := by
  have hz : shift3 2 0 = 0 := by rw [shift3_apply, if_neg (by decide)]
  refine ⟨shift3_strictlyLower, hz, no_fill_in (by decide) hz, ?_⟩
  have hsq : ((shift3 ^ 2 : Matrix (Fin 3) (Fin 3) ℝ)) 2 0 = 1 := by
    rw [pow_two, Matrix.mul_apply, Fin.sum_univ_three]
    norm_num [show ((0 : Fin 3) : ℕ) = 0 from rfl, show ((1 : Fin 3) : ℕ) = 1 from rfl,
              show ((2 : Fin 3) : ℕ) = 2 from rfl]
  unfold CEQ.Occupancy.occupancy
  rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_one,
      Matrix.add_apply, Matrix.add_apply, pow_zero, pow_one, hz, hsq,
      Matrix.one_apply_ne (by decide)]
  norm_num

/-- The cost statement, spelled out at one index: the recovered source at `i` is
    the residual at `i` minus one weighted sum over `A`'s row `i`. With
    `no_fill_in`, that sum ranges over the hop's own row support — for the delay
    carrier `ceq/beds/bed_k.py` builds, a single term. Nothing in it depends on
    `n` except through `nnz A`. -/
theorem source_entry
    {A : Matrix (Fin n) (Fin n) ℝ} (r : Fin n → ℝ) (i : Fin n) :
    (r - A *ᵥ r) i = r i - ∑ j, A i j * r j := by
  simp [Matrix.mulVec, Matrix.dotProduct]

end CEQ.V15Source

#print axioms CEQ.V15Source.inverse_identity_is_vacuous
#print axioms CEQ.V15Source.source_is_first_order_difference
#print axioms CEQ.V15Source.two_sources_recovered
#print axioms CEQ.V15Source.no_fill_in
#print axioms CEQ.V15Source.shift3_strictlyLower
#print axioms CEQ.V15Source.forward_map_fills_in
#print axioms CEQ.V15Source.source_entry
