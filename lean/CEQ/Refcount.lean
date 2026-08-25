/-
  CEQ.Refcount
  ------------
  The linking theorem: `caustic` Theorem 1's floor is `foliation`'s refcount.

  Provenance. Both halves are the author's own prior results, and this file is the
  sentence that identifies them:

    * `caustic` Theorem 1 (Zenodo 10.5281/zenodo.21997746) -- an indistinguishability
      floor `n - m` computable with NO ground truth. Formalised in `CEQ.OrbitBound`.
    * `foliation` -- a KV cache presented as a quotient by block-aligned prefix, where
      a plaque's REFCOUNT is the cardinality of its fibre and eviction is the
      elementary collapse of a free face.

  Take entities := prefixes and values := plaques. Then the floor is not merely
  BOUNDED by refcount data, it IS a sum over refcounts:

        n - m  =  Sum over plaques of (refcount - 1)          `floor_add_orbits`

  and evicting one plaque moves the floor by exactly `refcount - 1`
                                                              `evict_floor_add_refcount_pred`
  so a FREE FACE -- refcount 1 -- leaves the floor untouched.
                                                              `free_face_floor_unchanged`
  while a shared plaque strictly lowers it.                   `shared_plaque_floor_drops`

  WHY THIS IS THE LOAD-BEARING FILE. Scoring a KV block by removing it and measuring
  the change is leave-one-out influence (Cook 1977, Koh-Liang 2017) and is externally
  owned. The theorems below remove the forward pass entirely: the score is an integer
  the cache already maintains.

  NO TRUNCATED SUBTRACTION IN ANY HYPOTHESIS. Natural-number subtraction floors at
  zero, so an identity written subtractively can hold vacuously with both sides 0
  while the real claim is false. This is not hypothetical here: the first draft of the
  DeltaFloor theorem stated the accounting as arithmetic on `(n, m, k)` and `omega`
  REFUSED it, correctly -- at `n = 3, m = 3, k = 3` the left side is `0` and the right
  side truncates to `2`. That triple is unreachable in a real cache (three distinct
  plaques cannot share one three-prefix fibre), but nothing in the arithmetic said so.
  The fix is not to add the missing inequality as a hypothesis; it is to state the
  theorem over the ACTUAL surviving prefix set, where the constraint is a consequence
  of the structure rather than an assumption about it. `survivors` below is that set.
-/

import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Basic

namespace CEQ.Refcount

open Finset BigOperators

variable {E P : Type*} [Fintype E] [DecidableEq E] [DecidableEq P]

/-- The fibre over a plaque: every prefix that resolves to it. -/
def fibre (f : E → P) (p : P) : Finset E := univ.filter (fun e => f e = p)

/-- `foliation`'s refcount: the cardinality of the fibre. Maintained in O(1) by the
    cache's own bookkeeping; nothing here needs a forward pass. -/
def refcount (f : E → P) (p : P) : ℕ := (fibre f p).card

/-- The plaques actually present in the cache. The `m` of `caustic` is this card. -/
def plaques (f : E → P) : Finset P := univ.image f

/-- A free face: a plaque held by exactly one prefix. -/
def IsFreeFace (f : E → P) (p : P) : Prop := refcount f p = 1

/-- The prefixes that remain after plaque `p` is evicted. -/
def survivors (f : E → P) (p : P) : Finset E := univ \ fibre f p

/-- Membership in a fibre is exactly resolving to that plaque. Stated once so the
    proofs below do not each depend on how `simp` happens to normalise `mem_filter`. -/
theorem mem_fibre {f : E → P} {e : E} {p : P} : e ∈ fibre f p ↔ f e = p := by
  simp [fibre]

/-- A present plaque has a nonempty fibre, so its refcount is at least one. -/
theorem one_le_refcount {f : E → P} {p : P} (hp : p ∈ plaques f) :
    1 ≤ refcount f p := by
  rw [plaques, mem_image] at hp
  obtain ⟨e, _, he⟩ := hp
  exact card_pos.mpr ⟨e, by simp [fibre, he]⟩

/-- **The linking theorem.** Stated additively: no truncated subtraction can hide here.

    `Sum over plaques of (refcount - 1)  +  m  =  n`

    which is `n - m = Sum (refcount - 1)` -- Theorem 1 of `caustic` read off the fibre
    cardinalities of `foliation`, with no ground truth and no forward pass. -/
theorem floor_add_orbits (f : E → P) :
    (∑ p in plaques f, (refcount f p - 1)) + (plaques f).card = Fintype.card E := by
  have hmem : ∀ e ∈ (univ : Finset E), f e ∈ plaques f := by
    intro e _; exact mem_image_of_mem f (mem_univ e)
  have hfib : Fintype.card E = ∑ p in plaques f, refcount f p := by
    rw [← card_univ]; exact card_eq_sum_card_fiberwise hmem
  rw [hfib, card_eq_sum_ones (plaques f), ← sum_add_distrib]
  exact sum_congr rfl fun p hp => Nat.sub_add_cancel (one_le_refcount hp)

/-- The floor in the subtractive form that `CEQ.OrbitBound` states it in. -/
theorem floor_eq_sum_refcount_pred (f : E → P) :
    Fintype.card E - (plaques f).card = ∑ p in plaques f, (refcount f p - 1) := by
  rw [← floor_add_orbits f]; omega

/-- Eviction removes exactly the fibre. -/
theorem card_survivors_add_refcount (f : E → P) (p : P) :
    (survivors f p).card + refcount f p = Fintype.card E := by
  have hle : (fibre f p).card ≤ (univ : Finset E).card := card_le_card (subset_univ _)
  rw [survivors, refcount, card_sdiff (subset_univ _), card_univ]
  rw [card_univ] at hle
  omega

/-- Eviction removes exactly the one plaque: the surviving prefixes hit `plaques f \ {p}`.

    This is the step that makes the arithmetic honest. It is where "the other `m - 1`
    plaques still have prefixes" stops being an assumption and becomes a consequence. -/
theorem image_survivors (f : E → P) (p : P) :
    (survivors f p).image f = plaques f \ {p} := by
  apply Finset.Subset.antisymm
  · intro q hq
    rw [mem_image] at hq
    obtain ⟨e, he, rfl⟩ := hq
    rw [survivors, mem_sdiff] at he
    rw [mem_sdiff, mem_singleton]
    exact ⟨mem_image_of_mem f (mem_univ e), fun h => he.2 (mem_fibre.mpr h)⟩
  · intro q hq
    rw [mem_sdiff, mem_singleton] at hq
    obtain ⟨hq1, hq2⟩ := hq
    rw [plaques, mem_image] at hq1
    obtain ⟨e, _, rfl⟩ := hq1
    rw [mem_image]
    refine ⟨e, ?_, rfl⟩
    rw [survivors, mem_sdiff]
    exact ⟨mem_univ e, fun h => hq2 (mem_fibre.mp h)⟩

/-- The surviving plaque count is exactly one less. -/
theorem card_image_survivors_add_one (f : E → P) {p : P} (hp : p ∈ plaques f) :
    ((survivors f p).image f).card + 1 = (plaques f).card := by
  rw [image_survivors, card_sdiff (singleton_subset_iff.mpr hp), card_singleton]
  exact Nat.sub_add_cancel (card_pos.mpr ⟨p, hp⟩)

/-- **DeltaFloor is the refcount, minus one.**

    post-eviction floor  +  (refcount - 1)  =  pre-eviction floor

    Both floors are the genuine `caustic` quantity of their own system, and the two are
    separated by exactly `refcount - 1`. No hypothesis is assumed about how the counts
    relate; it all comes out of `survivors`. -/
theorem evict_floor_add_refcount_pred (f : E → P) {p : P} (hp : p ∈ plaques f) :
    ((survivors f p).card - ((survivors f p).image f).card) + (refcount f p - 1)
      = Fintype.card E - (plaques f).card := by
  have hS := card_survivors_add_refcount f p
  have hI := card_image_survivors_add_one f hp
  have hle : ((survivors f p).image f).card ≤ (survivors f p).card := card_image_le
  have h1 : 1 ≤ refcount f p := one_le_refcount hp
  omega

/-- **The free-face corollary, and the reason the whole object exists.**

    Evicting a plaque of refcount 1 leaves the indistinguishability floor EXACTLY where
    it was. The safety of the eviction is a theorem, checked in O(1) against an integer
    the cache already holds -- not a measurement requiring a forward pass per candidate
    block. -/
theorem free_face_floor_unchanged (f : E → P) {p : P} (hp : p ∈ plaques f)
    (hfree : IsFreeFace f p) :
    (survivors f p).card - ((survivors f p).image f).card
      = Fintype.card E - (plaques f).card := by
  have h := evict_floor_add_refcount_pred f hp
  rw [IsFreeFace] at hfree
  omega

/-- The converse, which is what stops the criterion from being vacuous: evicting a
    plaque that is NOT a free face strictly LOWERS the surviving floor, i.e. it
    destroys distinctions the original system could certify.

    Without this, "free faces are safe" would be consistent with everything being safe,
    and the criterion would carry no bits. -/
theorem shared_plaque_floor_drops (f : E → P) {p : P} (hp : p ∈ plaques f)
    (hshared : ¬ IsFreeFace f p) :
    (survivors f p).card - ((survivors f p).image f).card
      < Fintype.card E - (plaques f).card := by
  have h := evict_floor_add_refcount_pred f hp
  have h1 : 1 ≤ refcount f p := one_le_refcount hp
  rw [IsFreeFace] at hshared
  omega

end CEQ.Refcount
