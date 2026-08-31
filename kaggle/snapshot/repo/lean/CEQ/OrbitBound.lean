/-
  CEQ.OrbitBound
  --------------
  Provenance: `caustic` Theorem 1, the Orbit Error Bound.
  https://doi.org/10.5281/zenodo.21997746

  Statement: for an injective ground relation `R` on a finite entity set, any model
  map `f` errs on at least `n − m` entities, where `m` is the number of distinct
  values `f` takes.

  This is the bound that consults no ground truth, which is what turns it from a
  diagnostic into a training objective. It is the reason `select_by_floor` can score
  candidates on data where no answer key exists.

  The precondition is load-bearing and is not decorative: `caustic`'s own README
  documents a case where injectivity failed silently at the token level and the bound
  would have certified 19 wrong answers on a model that answered all 20 correctly.
  `hR` below is that precondition, made explicit.
-/

import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card

namespace CEQ.OrbitBound

open Finset

variable {E A : Type*} [Fintype E] [DecidableEq E] [DecidableEq A]

/-- The entities on which `f` agrees with the ground relation. -/
def agree (R f : E → A) : Finset E := univ.filter (fun e => f e = R e)

/-- The entities on which `f` errs. -/
def errors (R f : E → A) : Finset E := univ.filter (fun e => f e ≠ R e)

/-- The orbit count `m`: the number of distinct values `f` takes. -/
def orbits (f : E → A) : ℕ := (univ.image f).card

/-- Agreement and error partition the entity set. -/
theorem card_agree_add_card_errors (R f : E → A) :
    (agree R f).card + (errors R f).card = Fintype.card E := by
  classical
  have h := Finset.filter_card_add_filter_neg_card_eq_card
    (s := (Finset.univ : Finset E)) (p := fun e => f e = R e)
  simpa [agree, errors, Finset.card_univ] using h

/-- **`f` is injective on the agreement set.**

    If `f` agrees with `R` at `e₁` and `e₂` and `f e₁ = f e₂`, then `R e₁ = R e₂`,
    and injectivity of `R` gives `e₁ = e₂`. This is the whole content of the bound. -/
theorem injOn_agree (R f : E → A) (hR : Function.Injective R) :
    Set.InjOn f (agree R f) := by
  intro e₁ h₁ e₂ h₂ hf
  simp only [agree, coe_filter, Set.mem_setOf_eq, mem_univ, true_and] at h₁ h₂
  exact hR (by rw [← h₁, ← h₂, hf])

/-- The agreement set is no larger than the number of orbits. -/
theorem card_agree_le_orbits (R f : E → A) (hR : Function.Injective R) :
    (agree R f).card ≤ orbits f := by
  classical
  have h : ((agree R f).image f).card = (agree R f).card :=
    Finset.card_image_of_injOn (injOn_agree R f hR)
  rw [← h]
  exact Finset.card_le_card (Finset.image_subset_image (Finset.filter_subset _ _))

/-- **caustic Theorem 1 — the Orbit Error Bound.**

    `err(f) ≥ n − m`, consulting no ground truth beyond the *injectivity* of `R`.
    Note the statement uses truncated ℕ subtraction, which is the honest form: when
    `m ≥ n` the bound is vacuous, exactly as it should be. -/
theorem orbit_error_bound (R f : E → A) (hR : Function.Injective R) :
    Fintype.card E - orbits f ≤ (errors R f).card := by
  have hpart := card_agree_add_card_errors R f
  have hle := card_agree_le_orbits R f hR
  omega

/-- **Tightness.** When `f` is itself injective, `m = n` and the bound reads `0 ≤ err`,
    which is attained by `f = R`. A bound that could not be attained would not be a
    bound worth reporting. -/
theorem orbit_error_bound_attained (R : E → A) :
    (errors R R).card = 0 := by
  simp [errors]

end CEQ.OrbitBound
