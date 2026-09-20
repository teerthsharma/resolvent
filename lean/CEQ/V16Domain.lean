/-
  CEQ.V16Domain — L-DOM: LEAN #2 RE-STATED ON THE CORPUS'S ACTUAL SUPPORT,
  the domain census, and #5a / #5b
  ---------------------------------------------------------------------------
  Provenance: `CEQ_V16_CONTRACT.md` STANDING LAWS (L-DOM), PART II items #2
  (re-statement clause), #5a `three_corners_containment`, #5b
  `gate_zero_beta_zero_is_linear_attention`, and PART V debt D-DOM.
  Companion report: `V16_LEAN_DOMAIN.md`.

  THE DEFECT THIS FILE EXISTS FOR.

  `CEQ.V15.prefix_logit_mask` (#2) is proved under `∀ k, 0 < a k`. BED-M draws
  its gate from `{−1, 0, +1}` (`scale/negation_scope.py:428-429`). One of the
  three values satisfies the hypothesis. A theorem green in Lean and applicable
  to one third of the values its corpus draws is decoration, and L-DOM is the
  law that says so.

  WHAT IS REFUSED AS THE RE-STATEMENT.

  Weakening `0 < a` to `0 ≤ a` and keeping the conclusion
  `exp(C_i − C_j) = ∏ a_k` is REFUSED, and refuted in-file twice:

  * `lean_log_junk_makes_the_scan_form_silently_false` — Lean's `Real.log 0 = 0`
    is a junk value, so the weakened statement does not fail loudly; it returns
    `1` where the path product is `0`.
  * `no_prefix_scan_represents_a_zero_gate` — the real reason, and it is
    structural, not a Lean artifact: `exp` is never zero, so NO prefix scan `C`
    whatsoever — not `scan (log m)`, not any other — can represent a hop that
    annihilates. `a = 0` is not a value the hypothesis can be relaxed to admit.
    It is a value the exponential FORM cannot carry.

  WHAT IS PROVED INSTEAD.

  `a_k = m_k · e^{iθ_k}` with `m_k ∈ [0,1]` CLOSED and `θ_k` free, the hop
  taken as the PATH PRODUCT `∏_{k=j+1}^{i} a_k` rather than as a difference of
  prefix scans.

  | theorem                              | what it says |
  |--------------------------------------|--------------|
  | `pathProd_abs`                       | `|∏ a| = ∏ m` — hypothesis `0 ≤ m` ONLY, both endpoints of `[0,1]` included, every phase |
  | `pathProd_eq_zero_iff`               | the product is `0` **iff** a zero magnitude is on the path: `m = 0` is a THEOREM (annihilation), not an excluded hypothesis |
  | `pathProd_eq_Wp`                     | on `m > 0` the path product IS `CEQ.V15Phase.Wp`, i.e. #2's exponential form, unchanged |
  | `no_prefix_scan_represents_a_zero_gate` | off `m > 0` no exponential form exists at all |
  | `prefix_logit_mask_restated`         | the five clauses above assembled into one statement over `m ∈ [0,1]` |
  | `bedM_gate_exact`                    | each of `{−1, 0, +1}` is `gateOf (|a|) (arg a)` exactly — `−1` is `m = 1, θ = π`, an ordinary point |
  | `constant_phase_gate_is_rope`        | a constant phase gate `≡ e^{iω}` is the RoPE kernel `e^{iω(i−j)}`, exactly |
  | `cumulative_phase_is_separable`      | `∏ e^{iθ_k} = e^{i(Θ_i − Θ_j)}` with `Θ` the prefix sum — the phase path-product IS the prefix-scan difference, for every `θ`, no hypothesis |
  | `magnitude_zero_not_separable`       | the phase always separates through a prefix scan; a planted magnitude zero on the same window has NO prefix-scan representation, by ANY `C` — quotes `no_prefix_scan_represents_a_zero_gate` rather than re-deriving it |

  WHAT WAS LOST. The conclusion was not weakened — `pathProd_abs`'s hypothesis
  is strictly weaker than #2's and its conclusion is the same path product, and
  `restatement_subsumes_phase_path_le_one` proves the two agree wherever #2 had
  a value. What was lost is the log-domain COMPUTATIONAL ROUTE: on a window
  containing a zero gate the parallel prefix scan plus subtraction does not
  compute the hop and cannot be repaired by any choice of `C`. The route
  survives only as a segmented scan that resets at zeros, or as the product
  itself. That cost is stated at
  `no_prefix_scan_represents_a_zero_gate` and nowhere adjusted away.

  #5a AND #5b. `Hop β g qk` is the §S-M′ family
  `W_ij = exp((C_i − C_j) + q_i·k_j) / Z_i^β`. `three_corners_containment`
  names three settings and proves each equals its named operator;
  `corners_are_distinct` is included because a containment whose corners
  coincide is decoration. `gate_zero_beta_zero_is_linear_attention` is #5b and
  it is a REFUTATION: at `g ≡ 0, β = 0` the hop is the unnormalized linear
  form, its row does not sum to `1`
  (`gate_zero_beta_zero_row_not_one`, composed with
  `CEQ.V15.gate_zero_not_stochastic`), and every softmax row does
  (`softmax_row_sum_one`). R11 refuted the parity clause three ways; this is
  the statement that stops it being re-claimed.

  WHAT IS NOT CLAIMED. Nothing here trains, and nothing here says the phase
  gate is learnable through the `{0, π}` restriction. `Hop` is a single-head,
  single-channel, real-logit object: no multi-head coupling, no value
  contraction, no `β` between `0` and `1`, and no claim that the learned
  switches land on a corner.

  No `sorry`. `#print axioms` on every declaration at the foot of the file.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import CEQ.V15
import CEQ.V15Kernel
import CEQ.V15Phase
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Notation
import Mathlib.Algebra.Ring.Idempotents
import Mathlib.Algebra.GroupWithZero.Units.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Mathlib.Analysis.Matrix
import Mathlib.Data.Matrix.Block
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

namespace CEQ.V16Domain

open BigOperators Finset

/-! ## 1. The gate on the corpus's actual support -/

/-- One gate in polar form: `a = m · e^{iθ}`. `m` is the magnitude the §X₃₆ cap
    holds in `[0,1]` CLOSED; `θ` is free on the circle. Sign lives here and not
    in `m`: `a = −1` is `m = 1, θ = π`. -/
noncomputable def gateOf (m θ : ℝ) : ℂ := (m : ℂ) * Complex.exp ((θ : ℂ) * Complex.I)

/-- The hop as the PATH PRODUCT `∏_{k=j+1}^{i} a_k`. This is the object #2
    claims to compute, taken directly rather than through a difference of
    prefix scans. Nothing in this definition mentions `log`. -/
noncomputable def pathProd (m θ : ℕ → ℝ) (i j : ℕ) : ℂ :=
  ∏ k in Ico (j + 1) (i + 1), gateOf (m k) (θ k)

lemma abs_gateOf (m θ : ℝ) : Complex.abs (gateOf m θ) = |m| := by
  rw [gateOf, map_mul, Complex.abs_ofReal, Complex.abs_exp_ofReal_mul_I, mul_one]

/-- The polar factorization of the path product: magnitude from the `m`, a
    unit-modulus factor from the `θ`, no cross term and no hypothesis. -/
lemma pathProd_polar (m θ : ℕ → ℝ) (i j : ℕ) :
    pathProd m θ i j = ((∏ k in Ico (j + 1) (i + 1), m k : ℝ) : ℂ)
      * Complex.exp (((∑ k in Ico (j + 1) (i + 1), θ k : ℝ) : ℂ) * Complex.I) := by
  have h1 : ∏ k in Ico (j + 1) (i + 1), ((m k : ℝ) : ℂ)
      = ((∏ k in Ico (j + 1) (i + 1), m k : ℝ) : ℂ) := (Complex.ofReal_prod _ _).symm
  have h2 : ∏ k in Ico (j + 1) (i + 1), Complex.exp ((θ k : ℂ) * Complex.I)
      = Complex.exp (((∑ k in Ico (j + 1) (i + 1), θ k : ℝ) : ℂ) * Complex.I) := by
    rw [← Complex.exp_sum, Complex.ofReal_sum, Finset.sum_mul]
  rw [pathProd]
  simp only [gateOf]
  rw [Finset.prod_mul_distrib, h1, h2]

/-- **The magnitude law on the WHOLE closed support.** `0 ≤ m` is the only
    hypothesis: `m = 0` and `m = 1` are both included, and `θ` is universally
    quantified. This is strictly weaker in hypothesis than
    `CEQ.V15.prefix_logit_mask`'s `0 < a` and identical in conclusion. -/
theorem pathProd_abs (m θ : ℕ → ℝ) (h0 : ∀ k, 0 ≤ m k) (i j : ℕ) :
    Complex.abs (pathProd m θ i j) = ∏ k in Ico (j + 1) (i + 1), m k := by
  rw [pathProd, Complex.abs_prod]
  exact Finset.prod_congr rfl fun k _ => by rw [abs_gateOf, abs_of_nonneg (h0 k)]

/-- **`m = 0` is a THEOREM, not an excluded hypothesis: the product
    annihilates, exactly, and only then.** An iff, so the band's `0` endpoint
    is characterized rather than dodged. -/
theorem pathProd_eq_zero_iff (m θ : ℕ → ℝ) (i j : ℕ) :
    pathProd m θ i j = 0 ↔ ∃ k ∈ Ico (j + 1) (i + 1), m k = 0 := by
  rw [pathProd, Finset.prod_eq_zero_iff]
  constructor
  · rintro ⟨k, hk, h⟩
    refine ⟨k, hk, ?_⟩
    have hz := congrArg Complex.abs h
    rw [abs_gateOf, map_zero] at hz
    exact abs_eq_zero.mp hz
  · rintro ⟨k, hk, h⟩
    exact ⟨k, hk, by simp [gateOf, h]⟩

/-! ### The two refusals: why `0 < a` cannot simply be relaxed -/

/-- **REFUSED: `0 ≤ a` with the conclusion kept.** Lean's `Real.log 0 = 0` is a
    junk value, so the naive relaxation of #2 does not fail loudly — it returns
    `1` where the path product is `0`. Filed so that no later file reads a green
    `Real.log`-based statement on `[0,1]` as evidence about the `m = 0` draw. -/
theorem lean_log_junk_makes_the_scan_form_silently_false :
    CEQ.V15Phase.Wp (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 = 1
      ∧ pathProd (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 = 0
      ∧ CEQ.V15Phase.Wp (fun _ => (0 : ℝ)) (fun _ => 0) 1 0
          ≠ pathProd (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 := by
  have hW : CEQ.V15Phase.Wp (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 = 1 := by
    simp [CEQ.V15Phase.Wp, CEQ.V15Phase.C]
  have hP : pathProd (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 = 0 :=
    (pathProd_eq_zero_iff _ _ 1 0).mpr ⟨1, Finset.mem_Ico.mpr ⟨le_refl 1, by norm_num⟩, rfl⟩
  exact ⟨hW, hP, by rw [hW, hP]; exact one_ne_zero⟩

/-- **THE STRUCTURAL REASON, and it is not a Lean artifact.** `Complex.exp` is
    never zero. So no prefix scan `C : ℕ → ℂ` whatsoever — not `scan (log m)`,
    not any repaired or extended-real variant — represents a hop that
    annihilates. `a = 0` is not a value #2's hypothesis can be relaxed to
    admit; it is a value the exponential FORM cannot carry. This is the exact
    cost of the re-statement and it is a cost in the COMPUTATIONAL ROUTE (the
    parallel prefix scan plus subtraction), not in the conclusion. -/
theorem no_prefix_scan_represents_a_zero_gate (C : ℕ → ℂ) (m θ : ℕ → ℝ) {i j : ℕ}
    (hz : ∃ k ∈ Ico (j + 1) (i + 1), m k = 0) :
    Complex.exp (C i - C j) ≠ pathProd m θ i j := by
  rw [(pathProd_eq_zero_iff m θ i j).mpr hz]
  exact Complex.exp_ne_zero _

/-! ### What survives unchanged where #2 had a value -/

/-- On `m > 0` the path product IS `CEQ.V15Phase.Wp`, #2's exponential form,
    entry for entry. Nothing about the positive part of the support is
    weakened, moved, or re-derived. -/
theorem pathProd_eq_Wp (m θ : ℕ → ℝ) (hm : ∀ k, 0 < m k) {i j : ℕ} (hij : j ≤ i) :
    pathProd m θ i j = CEQ.V15Phase.Wp m θ i j := by
  rw [CEQ.V15Phase.Wp_polar m θ hm hij, pathProd_polar]

/-- The diff, in one line: on the overlap of the two domains the moduli agree,
    so `CEQ.V15Phase.phase_path_le_one` and `phase_path_eq_one_iff_band` are
    recovered by the re-statement and are not competing readings. -/
theorem restatement_subsumes_phase_path_le_one (m θ : ℕ → ℝ) (h0 : ∀ k, 0 < m k)
    {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (pathProd m θ i j) = Complex.abs (CEQ.V15Phase.Wp m θ i j) := by
  rw [pathProd_eq_Wp m θ h0 hij]

/-! ### The band, with `0` and `1` both attainable -/

/-- `∏ f = 1 ↔ f ≡ 1`, for `f` in the CLOSED interval `[0,1]`.
    `CEQ.V15Phase.prod_eq_one_iff` needs `0 < f`; this is the same statement on
    the closed support, which is what the cap actually delivers. -/
lemma prod_eq_one_iff_of_nonneg {s : Finset ℕ} {f : ℕ → ℝ} (h0 : ∀ k ∈ s, 0 ≤ f k)
    (h1 : ∀ k ∈ s, f k ≤ 1) : (∏ k in s, f k = 1) ↔ ∀ k ∈ s, f k = 1 := by
  classical
  refine ⟨fun h k hk => ?_, Finset.prod_eq_one⟩
  by_contra hne
  have hlt : f k < 1 := lt_of_le_of_ne (h1 k hk) hne
  have hP1 : ∏ l in s.erase k, f l ≤ 1 :=
    Finset.prod_le_one (fun l hl => h0 l (Finset.mem_of_mem_erase hl))
      (fun l hl => h1 l (Finset.mem_of_mem_erase hl))
  have hsplit : f k * ∏ l in s.erase k, f l = 1 := by
    rw [Finset.mul_prod_erase _ _ hk]; exact h
  have hle : f k * ∏ l in s.erase k, f l ≤ f k * 1 :=
    mul_le_mul_of_nonneg_left hP1 (h0 k hk)
  rw [hsplit, mul_one] at hle
  linarith

/-- **#2 RE-STATED, assembled.** Hypotheses: `0 ≤ m k ≤ 1` for every `k`, `θ`
    free, `j ≤ i`. Five clauses, covering `m ∈ [0,1]` INCLUDING BOTH ENDPOINTS:

    1. the magnitude law — `|∏ a| = ∏ m`;
    2. the bound `|∏ a| ≤ 1`, arithmetic rather than luck;
    3. the band — `|∏ a| = 1` **iff** every magnitude on the path is `1`;
    4. **off the band the product is exactly `0`** — a theorem, not an
       excluded hypothesis;
    5. on the positive part the exponential form of #2 is recovered unchanged.

    Clause 4 is the one #2 could not state, and clause 5 is the guarantee that
    nothing was traded for it. -/
theorem prefix_logit_mask_restated (m θ : ℕ → ℝ) (h0 : ∀ k, 0 ≤ m k) (h1 : ∀ k, m k ≤ 1)
    {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (pathProd m θ i j) = ∏ k in Ico (j + 1) (i + 1), m k
      ∧ Complex.abs (pathProd m θ i j) ≤ 1
      ∧ (Complex.abs (pathProd m θ i j) = 1 ↔ ∀ k ∈ Ico (j + 1) (i + 1), m k = 1)
      ∧ ((∃ k ∈ Ico (j + 1) (i + 1), m k = 0) → pathProd m θ i j = 0)
      ∧ ((∀ k, 0 < m k) → pathProd m θ i j = CEQ.V15Phase.Wp m θ i j) := by
  have habs := pathProd_abs m θ h0 i j
  refine ⟨habs, ?_, ?_, fun hz => (pathProd_eq_zero_iff m θ i j).mpr hz,
    fun hp => pathProd_eq_Wp m θ hp hij⟩
  · rw [habs]
    exact Finset.prod_le_one (fun k _ => h0 k) (fun k _ => h1 k)
  · rw [habs]
    exact prod_eq_one_iff_of_nonneg (fun k _ => h0 k) (fun k _ => h1 k)

/-! ## 2. BED-M's support, and the census as a theorem -/

/-- BED-M's gate magnitude: `|a|`. -/
noncomputable def magOf (a : ℝ) : ℝ := |a|

/-- BED-M's gate phase: `π` for the negative draw, `0` otherwise. The sign is
    carried by the phase, exactly as §X₃₆ asks. -/
noncomputable def argOf (a : ℝ) : ℝ := if a < 0 then Real.pi else 0

/-- **The census, as a theorem: every value BED-M draws is an ordinary point of
    the re-stated #2.** For each of `{−1, 0, +1}` the polar gate reproduces the
    drawn value EXACTLY and its magnitude lies in the closed interval `[0,1]`.
    `−1` is `m = 1, θ = π`; `0` is `m = 0`, the annihilating case; `+1` is
    `m = 1, θ = 0`, the band. Overlap with the re-stated hypotheses: 3 of 3.
    Overlap with `CEQ.V15.prefix_logit_mask`'s `0 < a`: 1 of 3. -/
theorem bedM_gate_exact {a : ℝ} (ha : a = -1 ∨ a = 0 ∨ a = 1) :
    gateOf (magOf a) (argOf a) = (a : ℂ) ∧ 0 ≤ magOf a ∧ magOf a ≤ 1 := by
  rcases ha with rfl | rfl | rfl
  · refine ⟨?_, by norm_num [magOf], by norm_num [magOf]⟩
    have hneg : (-1 : ℝ) < 0 := by norm_num
    have hm : magOf (-1 : ℝ) = 1 := by norm_num [magOf]
    rw [gateOf, hm, argOf, if_pos hneg]
    rw [Complex.exp_pi_mul_I]
    norm_num
  · refine ⟨?_, by norm_num [magOf], by norm_num [magOf]⟩
    have hm : magOf (0 : ℝ) = 0 := by norm_num [magOf]
    rw [gateOf, hm]
    norm_num
  · refine ⟨?_, by norm_num [magOf], by norm_num [magOf]⟩
    have hpos : ¬ ((1 : ℝ) < 0) := by norm_num
    have hm : magOf (1 : ℝ) = 1 := by norm_num [magOf]
    rw [gateOf, hm, argOf, if_neg hpos]
    norm_num

/-- **The `a = −1` draw is an ordinary point.** The gate whose magnitude is `1`
    and whose phase is `π` is the real number `−1`, its modulus is `1`, and it
    sits on the band. No branch of `Real.log` is involved anywhere. -/
theorem negative_draw_is_on_the_band :
    gateOf 1 Real.pi = (-1 : ℂ) ∧ Complex.abs (gateOf 1 Real.pi) = 1 := by
  constructor
  · rw [gateOf, Complex.exp_pi_mul_I]; norm_num
  · rw [abs_gateOf]; norm_num

/-! ### The census counted, on decidable literals

    BED-M's support is integer-valued, so the overlap counts are computed
    rather than argued. `bedM` is `{−1, 0, +1}`
    (`scale/negation_scope.py:428-429`); `bedMProp` is `{−1, +1}`, the second
    builder (`make_propagate_batch`, line 658), which never zeroes the prefix
    and therefore still contains no positive-only-admissible value beyond
    `+1`. -/

def bedM : List ℤ := [-1, 0, 1]

def bedMProp : List ℤ := [-1, 1]

/-- `CEQ.V15.prefix_logit_mask`'s hypothesis `0 < a`. -/
def satOldTwo (a : ℤ) : Bool := decide (0 < a)

/-- The re-stated #2's hypothesis `0 ≤ |a| ∧ |a| ≤ 1`. -/
def satNewTwo (a : ℤ) : Bool := decide (0 ≤ |a| ∧ |a| ≤ 1)

/-- `CEQ.V15.bounded_gates_stable`'s image, `a ∈ (0,1)` open
    (`CEQ.V15Phase.softplus_gate_lt_one` closes the upper end strictly). -/
def satSix (a : ℤ) : Bool := decide (0 < a ∧ a < 1)

theorem bedM_overlap_old_two : bedM.countP satOldTwo = 1 := by decide

theorem bedM_overlap_new_two : bedM.countP satNewTwo = 3 := by decide

theorem bedM_overlap_six : bedM.countP satSix = 0 := by decide

theorem bedMProp_overlap_old_two : bedMProp.countP satOldTwo = 1 := by decide

theorem bedMProp_overlap_new_two : bedMProp.countP satNewTwo = 2 := by decide

theorem bedMProp_overlap_six : bedMProp.countP satSix = 0 := by decide

/-- **#6 is DECORATION on BED-M, and here is the empty overlap as a theorem
    over every `w` at once.** `bedM_overlap_six` counts it on the literals;
    this proves it for the whole parametrized family: no `w` whatsoever sends
    `exp(−softplus w)` to any of `−1`, `0`, `+1`. Consequence for PART III's
    identity bind: an arm whose gates are #6-parametrized cannot be set to
    BED-M's oracle gates AT ALL, so "oracle gates in ⇒ label ≤ 1e-6" is not a
    test #6's family can be given. This is what §X₃₆'s hard cap is for. -/
theorem six_misses_every_bedM_value (w : ℝ) :
    Real.exp (-CEQ.V15.softplus w) ≠ -1
      ∧ Real.exp (-CEQ.V15.softplus w) ≠ 0
      ∧ Real.exp (-CEQ.V15.softplus w) ≠ 1 := by
  refine ⟨?_, (Real.exp_pos _).ne', (CEQ.V15Phase.softplus_gate_lt_one w).ne⟩
  intro h
  have hp := Real.exp_pos (-CEQ.V15.softplus w)
  rw [h] at hp
  linarith

/-! ## 3. #12's overlap with BED-K is PARTIAL, and the gap is real -/

/-- **`d = 0` is outside #12 and must be.** `ceq/beds/bed_k.py`'s
    `_delay_kernel_matrix` rejects only `d < 0`, so `build_delay(n, 0, seed)`
    is a legal BED-K(a) cell; at `d = 0` the delay IS a first-order recurrence,
    `α = 0, β = 1`, and `CEQ.V15.first_order_cannot_delay`'s `1 ≤ d` is
    therefore load-bearing rather than decorative. #12's overlap with BED-K(a)
    is `d ≥ 1` — partial, with the excluded cell exhibited. -/
theorem delay_zero_is_first_order (y₀ : ℝ) (x : ℕ → ℝ) :
    CEQ.V15.IsDelay 0 (CEQ.V15.affineRec 0 1 y₀ x) x := by
  intro i
  cases i with
  | zero => simp [CEQ.V15.affineRec, CEQ.V15.firstOrder]
  | succ n => simp [CEQ.V15.affineRec, CEQ.V15.firstOrder]

/-! ## 4. #16's overlap is TOTAL and its content is NULL -/

/-- **L-DOM's test is necessary and not sufficient, and #16 is the witness.**
    `CEQ.V15Phase.unit_phase_product` has NO hypotheses, so its overlap with
    any corpus is total and it can never be declared decoration by an overlap
    count. It is decoration anyway: it reads `1` on the very draw where BED-M's
    gate is `0` and the path product annihilates. A census that counts only
    hypothesis overlap passes this theorem; the second column — what the
    theorem constrains on the drawn value — is what fails it. -/
theorem sixteen_is_silent_on_the_zero_draw (θ : ℕ → ℝ) (s : Finset ℕ) :
    Complex.abs (∏ k in s, Complex.exp ((θ k : ℂ) * Complex.I)) = 1
      ∧ pathProd (fun _ => (0 : ℝ)) θ 1 0 = 0 := by
  refine ⟨CEQ.V15Phase.unit_phase_product θ s, ?_⟩
  exact (pathProd_eq_zero_iff _ θ 1 0).mpr
    ⟨1, Finset.mem_Ico.mpr ⟨le_refl 1, by norm_num⟩, rfl⟩

/-! ## 4b. PHASE G / S1 — the phase axis is SEPARABLE, the magnitude axis is not

    The gate axis splits into MAGNITUDE `m ∈ [0,1]` and PHASE `θ` on the
    circle. `constant_phase_gate_is_rope` and `cumulative_phase_is_separable`
    show the phase half is SEPARABLE: a constant phase is exactly the RoPE
    kernel, and any phase schedule at all folds through a prefix scan
    (`cumsum`), which is what a fused kernel needs in order to absorb it into
    `q, k`. `magnitude_zero_not_separable` states the asymmetry this buys: the
    SAME window has a prefix-scan phase representation with no hypothesis on
    `θ` whatsoever, and, the moment a magnitude on it is `0`, NO prefix scan
    `C` represents its path product — `no_prefix_scan_represents_a_zero_gate`
    quoted directly, not re-derived. The kernel race is therefore a race over
    magnitude zeros only.

    SEPARABLE IS NOT FREE, and an earlier draft of this comment said free.
    Three measurements bound it, none of which these theorems assert:

    * the fold-in is exact only in float64 — worst `2.398082e-13` at
      n = 4096, d = 64 over 5 seeds. In half precision it costs about two
      extra ULPs beyond the rounding a plain `q @ kᵀ` already pays
      (`3.915×` the float16 floor against a no-phase control of `1.612×`),
      and only when the angles, the cumulative sum and `cos`/`sin` are
      computed in float32 and cast down afterwards. Computed entirely in
      the half dtype it costs `61.513×` the floor;
    * the rotation itself is `O(n·d)`. RoFormer §3.4.2 gives a separate
      element-wise realization precisely because the direct form of its
      Equation 16 is too expensive. Measured here at 0.66 % of attention
      time at n = 4096 and 0.05 % at n = 16384, with the table cached
      across batch and layers;
    * separately, and more sharply: in the CURRENT operator `θ` multiplies
      an already-exponentiated score, so it never reaches the real attention
      logit. `R_ij * e_ij` and `Z_i` are bitwise identical between `θ = 0`
      and random `θ`. RoPE rotates `q, k` BEFORE the dot product. These
      theorems describe a containment the shipped code does not yet
      implement. -/

/-- **`constant_phase_gate_is_rope`.** A gate with magnitude `≡ 1` and a
    CONSTANT phase `ω` is exactly the RoPE kernel `e^{iω(i−j)}` — the rotation
    angle is the phase times the token distance, nothing else. -/
theorem constant_phase_gate_is_rope (ω : ℝ) {i j : ℕ} (hij : j ≤ i) :
    pathProd (fun _ => (1 : ℝ)) (fun _ => ω) i j
      = Complex.exp (((ω * ((i : ℝ) - (j : ℝ)) : ℝ) : ℂ) * Complex.I) := by
  rw [pathProd_polar]
  have hcard : (Ico (j + 1) (i + 1)).card = i - j := by rw [Nat.card_Ico]; omega
  have hs : (∑ _k in Ico (j + 1) (i + 1), ω) = ω * ((i : ℝ) - (j : ℝ)) := by
    rw [Finset.sum_const, hcard, nsmul_eq_mul, Nat.cast_sub hij]; ring
  simp [hs]

/-- **`cumulative_phase_is_separable`.** The product of `e^{iθ_k}` over ANY
    window `Ico (j+1) (i+1)`, for ANY phase schedule `θ` (no hypothesis at
    all), equals `e^{i(Θ_i − Θ_j)}` with `Θ_n = Σ_{k ≤ n} θ_k` the prefix sum —
    the phase-path-product IS the prefix-scan difference, always. This is the
    fact that lets a fused kernel fold the whole phase gate into a rotation of
    `q` and `k`. -/
theorem cumulative_phase_is_separable (θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) :
    ∏ k in Ico (j + 1) (i + 1), Complex.exp ((θ k : ℂ) * Complex.I)
      = Complex.exp ((((∑ k in range (i + 1), θ k) - (∑ k in range (j + 1), θ k) : ℝ) : ℂ)
          * Complex.I) := by
  have hsum : (∑ k in Ico (j + 1) (i + 1), θ k)
      = (∑ k in range (i + 1), θ k) - (∑ k in range (j + 1), θ k) :=
    Finset.sum_Ico_eq_sub θ (Nat.succ_le_succ hij)
  rw [← hsum, Complex.ofReal_sum, Finset.sum_mul, Complex.exp_sum]

/-- **`magnitude_zero_not_separable`.** The asymmetry between the two halves of
    the split gate axis, as one theorem: for ANY phase schedule `θ` the phase
    factor over the window ALWAYS has a prefix-scan representation
    (`cumulative_phase_is_separable`, no hypothesis on `θ`), while the moment
    the window carries a planted magnitude zero, `pathProd m θ` has NO
    prefix-scan representation AT ALL — for every `C : ℕ → ℂ`, not some — by
    `no_prefix_scan_represents_a_zero_gate`, quoted rather than re-proved. The
    phase axis separates through `cumsum`; the magnitude axis does not, and
    `m = 0` is exactly where that stops being cosmetic. -/
theorem magnitude_zero_not_separable (m θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i)
    (hz : ∃ k ∈ Ico (j + 1) (i + 1), m k = 0) :
    (∏ k in Ico (j + 1) (i + 1), Complex.exp ((θ k : ℂ) * Complex.I)
        = Complex.exp ((((∑ k in range (i + 1), θ k) - (∑ k in range (j + 1), θ k) : ℝ) : ℂ)
            * Complex.I))
      ∧ ∀ C : ℕ → ℂ, Complex.exp (C i - C j) ≠ pathProd m θ i j :=
  ⟨cumulative_phase_is_separable θ hij, fun C => no_prefix_scan_represents_a_zero_gate C m θ hz⟩

/-! ## 5. #5a — the hop family and its three corners -/

/-- The §S-M′ numerator, `exp((C_i − C_j) + q_i·k_j)`. `qk i j` stands for the
    scalar `q_i · k_j`; nothing below needs it to factor. -/
noncomputable def num (g : ℕ → ℝ) (qk : ℕ → ℕ → ℝ) (i j : ℕ) : ℝ :=
  Real.exp ((CEQ.V15.scan g i - CEQ.V15.scan g j) + qk i j)

/-- The causal row normalizer `Z_i = Σ_{j ≤ i} num i j`. -/
noncomputable def Znorm (g : ℕ → ℝ) (qk : ℕ → ℕ → ℝ) (i : ℕ) : ℝ :=
  ∑ j in range (i + 1), num g qk i j

/-- **THE FAMILY.** `W_ij = exp((C_i − C_j) + q_i·k_j) / Z_i^β` for `j ≤ i`,
    `0` above the diagonal. `β` enters through `Real.rpow`, so `β = 0` gives
    `Z^0 = 1` for every `Z` including `Z = 0` and no positivity side condition
    leaks into the linear corner. -/
noncomputable def Hop (β : ℝ) (g : ℕ → ℝ) (qk : ℕ → ℕ → ℝ) (i j : ℕ) : ℝ :=
  if j ≤ i then num g qk i j / (Znorm g qk i) ^ β else 0

/-- Corner 1's named operator: causal softmax attention. -/
noncomputable def softmaxAttn (qk : ℕ → ℕ → ℝ) (i j : ℕ) : ℝ :=
  if j ≤ i then Real.exp (qk i j) / ∑ j' in range (i + 1), Real.exp (qk i j') else 0

/-- Corner 2's named operator: linear (unnormalized kernel) attention — the
    score with NO row normalizer, which is exactly what makes the row a causal
    scan rather than a softmax. -/
noncomputable def linearAttn (qk : ℕ → ℕ → ℝ) (i j : ℕ) : ℝ :=
  if j ≤ i then Real.exp (qk i j) else 0

lemma scan_zero_of_zero {g : ℕ → ℝ} (hg : ∀ k, g k = 0) (n : ℕ) : CEQ.V15.scan g n = 0 := by
  simp [CEQ.V15.scan, hg]

/-- **Corner 1: `β = 1, g ≡ 0, QK-on` is softmax attention.** -/
theorem corner_softmax {g : ℕ → ℝ} (hg : ∀ k, g k = 0) (qk : ℕ → ℕ → ℝ) (i j : ℕ) :
    Hop 1 g qk i j = softmaxAttn qk i j := by
  simp only [Hop, softmaxAttn, num, Znorm, scan_zero_of_zero hg, sub_self, zero_add,
    Real.rpow_one]

/-- **Corner 2: `β = 0, g ≡ 0` is LINEAR attention** — the unnormalized kernel
    score. This is #5b's first half. -/
theorem corner_linear {g : ℕ → ℝ} (hg : ∀ k, g k = 0) (qk : ℕ → ℕ → ℝ) (i j : ℕ) :
    Hop 0 g qk i j = linearAttn qk i j := by
  simp only [Hop, linearAttn, num, scan_zero_of_zero hg, sub_self, zero_add,
    Real.rpow_zero, div_one]

/-- **Corner 3: `β = 0, QK-off` is the path product** — `CEQ.V15.Wc`, the
    masked prefix-logit hop of #2. -/
theorem corner_path_product (g : ℕ → ℝ) (i j : ℕ) :
    Hop 0 g (fun _ _ => 0) i j = CEQ.V15.Wc g i j := by
  simp only [Hop, CEQ.V15.Wc, CEQ.V15.W, num, add_zero, Real.rpow_zero, div_one]

/-- Corner 3, bound to the gates themselves: on `0 < a` the corner IS
    `∏_{k=j+1}^{i} a_k`, by `CEQ.V15.prefix_logit_mask`. -/
theorem corner_path_product_is_the_gate_product (a : ℕ → ℝ) (ha : ∀ k, 0 < a k)
    {i j : ℕ} (hij : j ≤ i) :
    Hop 0 (fun k => Real.log (a k)) (fun _ _ => 0) i j = ∏ k in Ico (j + 1) (i + 1), a k := by
  rw [corner_path_product, CEQ.V15.Wc, if_pos hij, CEQ.V15.prefix_logit_mask a ha hij]

/-- **And corner 3 inherits #2's domain defect, unrepaired.** Wherever a zero
    magnitude sits on the path the real, exponential corner cannot equal the
    corpus's path product, for ANY gate schedule `g`. The `β = 0, QK-off`
    corner of §S-M′ is therefore the corner that the re-statement of §1
    replaces, not one it certifies. -/
theorem path_product_corner_fails_at_a_zero_gate (g : ℕ → ℝ) (m θ : ℕ → ℝ) {i j : ℕ}
    (hij : j ≤ i) (hz : ∃ k ∈ Ico (j + 1) (i + 1), m k = 0) :
    ((Hop 0 g (fun _ _ => 0) i j : ℝ) : ℂ) ≠ pathProd m θ i j := by
  rw [corner_path_product, CEQ.V15.Wc, if_pos hij, CEQ.V15.W,
    (pathProd_eq_zero_iff m θ i j).mpr hz]
  exact Complex.ofReal_ne_zero.mpr (Real.exp_ne_zero _)

/-- **#5a `three_corners_containment`.** One family, three named operators,
    each an instance of it at a stated setting of `(β, g, QK)`. -/
theorem three_corners_containment {g₀ : ℕ → ℝ} (hg : ∀ k, g₀ k = 0) (g : ℕ → ℝ)
    (qk : ℕ → ℕ → ℝ) :
    (∀ i j, Hop 1 g₀ qk i j = softmaxAttn qk i j)
      ∧ (∀ i j, Hop 0 g₀ qk i j = linearAttn qk i j)
      ∧ (∀ i j, Hop 0 g (fun _ _ => 0) i j = CEQ.V15.Wc g i j) :=
  ⟨corner_softmax hg qk, corner_linear hg qk, corner_path_product g⟩

/-- **REFUSED as the content of #5a: a containment whose corners coincide.**
    At the common setting `g ≡ 0, QK-off` the `β = 1` corner reads `1/2` at
    `(i,j) = (1,0)` and the `β = 0` corner reads `1`. The corners are distinct
    points of the family, so the containment is a statement about a family with
    interior and not three names for one operator. -/
theorem corners_are_distinct :
    Hop 1 (fun _ => (0 : ℝ)) (fun _ _ => 0) 1 0 = 1 / 2
      ∧ Hop 0 (fun _ => (0 : ℝ)) (fun _ _ => 0) 1 0 = 1
      ∧ Hop 1 (fun _ => (0 : ℝ)) (fun _ _ => 0) 1 0
          ≠ Hop 0 (fun _ => (0 : ℝ)) (fun _ _ => 0) 1 0 := by
  have hg : ∀ k : ℕ, (fun _ : ℕ => (0 : ℝ)) k = 0 := fun _ => rfl
  have h1 : Hop 1 (fun _ => (0 : ℝ)) (fun _ _ => 0) 1 0 = 1 / 2 := by
    rw [corner_softmax hg]
    norm_num [softmaxAttn, Finset.sum_range_succ]
  have h0 : Hop 0 (fun _ => (0 : ℝ)) (fun _ _ => 0) 1 0 = 1 := by
    rw [corner_linear hg]
    simp [linearAttn]
  refine ⟨h1, h0, ?_⟩
  rw [h1, h0]
  norm_num

/-! ## 6. #5b — the refutation, so the parity clause cannot be re-claimed -/

/-- Every softmax row sums to `1`. The class marker. -/
theorem softmax_row_sum_one (qk : ℕ → ℕ → ℝ) (i : ℕ) :
    ∑ j in range (i + 1), softmaxAttn qk i j = 1 := by
  have hS : 0 < ∑ j' in range (i + 1), Real.exp (qk i j') :=
    Finset.sum_pos (fun j _ => Real.exp_pos _)
      ⟨0, Finset.mem_range.mpr (Nat.succ_pos i)⟩
  have h : ∀ j ∈ range (i + 1), softmaxAttn qk i j
      = Real.exp (qk i j) / ∑ j' in range (i + 1), Real.exp (qk i j') := by
    intro j hj
    rw [softmaxAttn, if_pos (Nat.lt_succ_iff.mp (Finset.mem_range.mp hj))]
  rw [Finset.sum_congr rfl h, ← Finset.sum_div, div_self (ne_of_gt hS)]

/-- **#5b `gate_zero_beta_zero_is_linear_attention`.**

    At `g ≡ 0` and `β = 0` the §S-M′ hop is `linearAttn` — the unnormalized
    kernel score — and it is NOT softmax attention. Both halves are proved:
    the identification, and the separation by the class marker (`β = 0` rows do
    not sum to `1`; every softmax row does). R11 refuted the original parity
    clause three independent ways; this is the statement that stops it being
    re-claimed by anyone reading `g ≡ 0` as "standard attention". -/
theorem gate_zero_beta_zero_is_linear_attention {g : ℕ → ℝ} (hg : ∀ k, g k = 0)
    (qk : ℕ → ℕ → ℝ) :
    (∀ i j, Hop 0 g qk i j = linearAttn qk i j)
      ∧ ∃ (qk' : ℕ → ℕ → ℝ) (i j : ℕ), Hop 0 g qk' i j ≠ softmaxAttn qk' i j := by
  refine ⟨corner_linear hg qk, fun _ _ => 0, 1, 0, ?_⟩
  have h0 : Hop 0 g (fun _ _ => 0) 1 0 = 1 := by
    rw [corner_linear hg]; simp [linearAttn]
  have hs : softmaxAttn (fun _ _ => (0 : ℝ)) 1 0 = 1 / 2 := by
    norm_num [softmaxAttn, Finset.sum_range_succ]
  rw [h0, hs]
  norm_num

/-- **The separation at the level of the whole row, composed with
    `CEQ.V15.gate_zero_not_stochastic`.** At `g ≡ 0, β = 0, QK-off` the hop is
    `CEQ.V15.Wc`, whose row `i ≥ 1` sums to `i + 1 ≠ 1`. Read against
    `softmax_row_sum_one`: the `β = 0` corner is outside the softmax class for
    every `i ≥ 1`, not merely different at one entry. -/
theorem gate_zero_beta_zero_row_not_one (g : ℕ → ℝ) (hg : ∀ k, g k = 0) {i : ℕ}
    (hi : 1 ≤ i) : ∑ j in range (i + 1), Hop 0 g (fun _ _ => 0) i j ≠ 1 := by
  rw [Finset.sum_congr rfl (fun j _ => corner_path_product g i j)]
  exact CEQ.V15.gate_zero_not_stochastic g hg hi

/-- **And `β = 1` puts it back inside.** The same `g ≡ 0` gate at `β = 1` gives
    a row that sums to `1`. So `β`, not `g`, is the switch that decides
    membership in the softmax class — which is the precise sense in which the
    contract's original parity clause named the wrong parameter. -/
theorem beta_one_row_is_one (g : ℕ → ℝ) (hg : ∀ k, g k = 0) (qk : ℕ → ℕ → ℝ) (i : ℕ) :
    ∑ j in range (i + 1), Hop 1 g qk i j = 1 := by
  rw [Finset.sum_congr rfl (fun j _ => corner_softmax hg qk i j)]
  exact softmax_row_sum_one qk i

/-! ## 6. PHASE H — three operator-class facts, merged from the Phase-H round

    Provenance: Phase-H targets `scalar_gate_commutes`,
    `projector_idempotent_not_invertible`, `finite_semigroup_never_decidable`,
    each built and audited standalone before being merged here. They are
    grouped in `PhaseH` because they are about the operator CLASS a gate is
    drawn from — commutative scalars, idempotents, finite semigroups — and not
    about the `[0,1]`-magnitude re-statement the rest of this file exists for.

    **H.1 — WHAT THE PAIR MEANS.** `scalar_gate_commutes` says a scalar gate
    `gateOf m θ : ℂ`, composed along an ORDERED list of steps, gives the same
    product under any permutation of that list. `matrix_gate_not_commutes`
    exhibits two `2×2` complex matrices for which the same composition does
    NOT: `!![0,1;0,0] * !![0,0;1,0] = !![1,0;0,0]` but the other order gives
    `!![0,0;0,1]`. Together: a scalar carry CANNOT represent step order — the
    algebra makes every ordering equal, so there is no state in which to keep
    it — and a matrix carry can. The scalar half alone is a one-line
    specialization of Mathlib's `List.Perm.prod_eq` and carries no content;
    the contrast is the claim.

    **H.1 — SHAPE, STATED PLAINLY BECAUSE IT DOES NOT MATCH.** `pathProd`
    above is a `Finset.prod` over `Finset.Ico (j+1) (i+1)`, NOT a `List.prod`.
    A `Finset` has no order, so there is nothing in `pathProd` to permute:
    permutation-invariance of `pathProd` is free, unprovable-because-unstatable
    rather than proved, and `Finset.prod` could not even be TYPED over a
    noncommuting matrix codomain, since the definition requires a `CommMonoid`
    to be well defined at all. So `scalar_gate_commutes` and
    `matrix_gate_not_commutes` are a SEPARATE combinatorial statement about
    `List.prod`. They do not restate, strengthen, or apply to `pathProd` as
    this file defines it. What they establish is the algebraic reason the
    corpus's choice of a scalar gate forecloses order-sensitivity — a fact
    ABOUT the choice, not a theorem about the shipped operator.

    **H.1 — AMENDED BY §7 (PHASE H.1). The last sentence of the paragraph
    above is now WRONG and is retained only so the correction is legible.**
    `PhaseH1.pathProd_eq_pathProdL` proves that `pathProd` — this file's own
    `Finset.prod`, unchanged — IS a `List.prod` over
    `(Ico (j+1) (i+1)).toList`, and `PhaseH1.pathProdL_scalar_is_order_blind`
    then proves it equals that `List.prod` over EVERY permutation of the
    window. So order-blindness of the SHIPPED product is a theorem, not an
    unstatable free fact, and `matrix_gate_not_commutes` is the contrast
    against a product (`PhaseH1.pathProdMatrixL`) that the corpus now
    actually has. What remains true in the paragraph above is only the typing
    fact: `Finset.prod` needs a `CommMonoid` and cannot be written over a
    noncommuting codomain.

    **H.2 — WHAT IS AND IS NOT LOAD-BEARING.** `projector_idempotent_not_invertible`
    carries `hP0 : P ≠ 0` for parity with the round's contract, and the proof
    never uses it; only `hP1 : P ≠ 1` does work. Recorded rather than removed
    so the redundancy is a stated contract choice, not a hidden one.
    `projector_det_eq_zero` is a genuine strengthening, not a packaging
    conjunction: over ℝ it names the scalar witness `0` where the abstract
    form gives only a negative existential.

    **H.3 — WHICH RICE PRECONDITION FAILS.** Rice's proof needs a class rich
    enough to simulate unboundedly many distinct configurations, so that
    "never halts" can be realized as "never repeats a configuration, forever."
    A finite semigroup acting on a `Fintype` state space `α` has only
    `Fintype.card α` configurations, so the orbit cannot strictly grow beyond
    that without repeating, and a repeat is a decision — there is nowhere left
    to hide an unbounded computation. `finite_semigroup_never_decidable` is a
    `def`, not a `theorem`: the goal `DecidablePred (Reachable G s)` is
    Type-valued, and `theorem` compiles Type-valued goals opaquely, yielding a
    decision procedure that provably exists and cannot run. As a `def` it
    runs — see the `#eval` pair at the foot of this file, which computes
    `true` for a reachable target and `false` for a stalled generator. -/

namespace PhaseH

/-! ### H.1 A scalar gate is order-blind; a matrix gate is not -/

/-- **`scalar_gate_commutes`.** The path product of a SCALAR gate
    `gateOf (m k) (θ k)` over an ordered sequence of steps `l : List ι` is
    invariant under permuting that sequence. True because `ℂ` is a
    `CommMonoid` under multiplication and `List.Perm.prod_eq` holds over any
    `CommMonoid`. On its own this is Mathlib; its content is the contrast with
    `matrix_gate_not_commutes` below. NOTE the shape: this is `List.prod`, not
    the `Finset.prod` of `pathProd` — see the section docstring. -/
theorem scalar_gate_commutes {ι : Type*} (m θ : ι → ℝ) {l l' : List ι}
    (hperm : l.Perm l') :
    (l.map (fun k => gateOf (m k) (θ k))).prod
      = (l'.map (fun k => gateOf (m k) (θ k))).prod :=
  (hperm.map (fun k => gateOf (m k) (θ k))).prod_eq

/-- **`matrix_gate_not_commutes`.** The scalar case is not a free fact about
    list products — it consumes `ℂ`'s commutativity. Witness:
    `A = !![0,1;0,0]`, `B = !![0,0;1,0]`. `[A, B]` and `[B, A]` ARE
    permutations of each other (`List.Perm.swap`), yet `A * B = !![1,0;0,0]`
    and `B * A = !![0,0;0,1]` differ at entry `(0,0)`. The failure is
    exhibited by a witness, not asserted from an abstract non-commutativity
    lemma. Order is information a matrix carry can hold and a scalar carry
    provably cannot. -/
theorem matrix_gate_not_commutes :
    ∃ A B : Matrix (Fin 2) (Fin 2) ℂ,
      ([A, B] : List (Matrix (Fin 2) (Fin 2) ℂ)).Perm [B, A]
        ∧ ([A, B] : List (Matrix (Fin 2) (Fin 2) ℂ)).prod
            ≠ ([B, A] : List (Matrix (Fin 2) (Fin 2) ℂ)).prod := by
  refine ⟨!![0, 1; 0, 0], !![0, 0; 1, 0], List.Perm.swap _ _ _, ?_⟩
  simp only [List.prod_cons, List.prod_nil, mul_one, Matrix.mul_fin_two]
  intro h
  have h00 := congrFun (congrFun h 0) 0
  norm_num [Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons] at h00

/-! ### H.2 A nontrivial projector gate is not invertible -/

/-- **Helper, not a round target.** In any monoid, an idempotent that is not
    `1` is not a unit: `P*P = P*1` cancels `P` to force `P = 1`. This is why
    `P ≠ 1` alone suffices to kill invertibility — no determinant, no
    dimension. -/
theorem idempotent_ne_one_not_isUnit {M : Type*} [Monoid M] {P : M}
    (hP : IsIdempotentElem P) (hP1 : P ≠ 1) : ¬ IsUnit P := by
  intro hu
  exact hP1 (hu.mul_left_cancel (hP.eq.trans (mul_one P).symm))

/-- **TARGET.** A projector (`P * P = P`) that is neither the zero gate nor
    the identity gate is not invertible, over any ring. `hP0 : P ≠ 0` is
    carried for contract parity and is NOT used (hence bound as `_hP0`; it is
    still a required positional argument, so the statement is unchanged);
    `hP1 : P ≠ 1` is the
    load-bearing hypothesis. This separates "reset" from "rotate" as gate
    primitives: `gateOf m θ` with `m ≠ 0` is invertible at every `θ`,
    including `θ = 0`; a nontrivial reset is invertible at no parameter value
    whatsoever, because idempotence — not the value of a continuous
    parameter — forces the singularity. -/
theorem projector_idempotent_not_invertible {R : Type*} [Ring R] {P : R}
    (hP : IsIdempotentElem P) (_hP0 : P ≠ 0) (hP1 : P ≠ 1) : ¬ IsUnit P :=
  idempotent_ne_one_not_isUnit hP hP1

/-- **Sharper form.** Over `Matrix (Fin n) (Fin n) ℝ` the same hypotheses give
    `P.det = 0` outright. Genuinely sharper, not a repackaging: over a field
    `IsUnit` and `≠ 0` coincide, so the conclusion becomes a named scalar
    rather than a negated existential. -/
theorem projector_det_eq_zero {n : ℕ} {P : Matrix (Fin n) (Fin n) ℝ}
    (hP : IsIdempotentElem P) (hP0 : P ≠ 0) (hP1 : P ≠ 1) : P.det = 0 := by
  by_contra hdet
  exact projector_idempotent_not_invertible hP hP0 hP1
    (Matrix.isUnit_iff_isUnit_det P |>.mpr (isUnit_iff_ne_zero.mpr hdet))

/-! ### H.3 Finite-semigroup reachability is decidable, and the decision runs -/

section Finite

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- One step of the closure: everything already reached, plus everything one
    more generator-application reaches from it. Concrete and computable — no
    `Submonoid.closure`, no classical `sInf`, just union and image. -/
def step (G : Finset (α → α)) (S : Finset α) : Finset α :=
  S ∪ S.biUnion (fun x => G.image (fun f => f x))

/-- The orbit after `n` closure steps from `{s}`. `orbit G s 0 = {s}`:
    reachability is reflexive, as standard for automata reachability. -/
def orbit (G : Finset (α → α)) (s : α) : ℕ → Finset α
  | 0 => {s}
  | n + 1 => step G (orbit G s n)

/-- Each closure step only grows the set. -/
theorem subset_step (G : Finset (α → α)) (S : Finset α) : S ⊆ step G S := by
  unfold step
  apply Finset.subset_union_left

/-- The orbit sequence is `⊆`-monotone in `n`. -/
theorem orbit_subset_succ (G : Finset (α → α)) (s : α) (n : ℕ) :
    orbit G s n ⊆ orbit G s (n + 1) :=
  subset_step G (orbit G s n)

theorem orbit_mono (G : Finset (α → α)) (s : α) : Monotone (orbit G s) :=
  monotone_nat_of_le_succ (orbit_subset_succ G s)

/-- Once one closure step changes nothing it never changes anything again:
    `step` depends only on the current `Finset`, so a fixed point is fixed
    forever. Needs no cardinality bound. -/
theorem orbit_stable_forward (G : Finset (α → α)) (s : α) {n : ℕ}
    (h : orbit G s n = orbit G s (n + 1)) : ∀ k, orbit G s (n + k) = orbit G s n := by
  intro k
  induction k with
  | zero => rfl
  | succ j ih =>
    have heq : orbit G s (n + (j + 1)) = step G (orbit G s (n + j)) := rfl
    rw [heq, ih]
    exact h.symm

/-- Pure arithmetic: a `ℕ`-valued sequence that strictly increases at every
    step below `N` has grown by at least `N` by step `N`. -/
theorem sum_of_strict_steps (c : ℕ → ℕ) :
    ∀ N, (∀ n < N, c n < c (n + 1)) → c 0 + N ≤ c N := by
  intro N
  induction N with
  | zero => intro _; simp
  | succ k ih =>
    intro h
    show c 0 + (k + 1) ≤ c (k + 1)
    have hk : ∀ n < k, c n < c (n + 1) := fun n hn => h n (Nat.lt_succ_of_lt hn)
    have hlt : c k < c (k + 1) := h k (Nat.lt_succ_self k)
    have hprev := ih hk
    omega

/-- **THE PIGEONHOLE STEP — the single place `Fintype.card` is consumed.**
    Some closure step strictly below `Fintype.card α` changes nothing. This is
    exactly what fails for infinite `α`: with no upper bound on
    `(orbit G s n).card`, a sequence can grow forever and no stabilization
    step below any fixed `N` is forced to exist. -/
theorem exists_stable_step (G : Finset (α → α)) (s : α) :
    ∃ n < Fintype.card α, orbit G s n = orbit G s (n + 1) := by
  by_contra hcon
  push_neg at hcon
  have hstrict : ∀ n < Fintype.card α, (orbit G s n).card < (orbit G s (n + 1)).card :=
    fun n hn =>
      Finset.card_lt_card
        (lt_of_le_of_ne (orbit_subset_succ G s n) (hcon n hn))
  have hgrow : (orbit G s 0).card + Fintype.card α ≤ (orbit G s (Fintype.card α)).card :=
    sum_of_strict_steps (fun n => (orbit G s n).card) (Fintype.card α) hstrict
  have hpos : 1 ≤ (orbit G s 0).card := by
    show 1 ≤ ({s} : Finset α).card
    simp
  have hbound : (orbit G s (Fintype.card α)).card ≤ Fintype.card α :=
    (orbit G s (Fintype.card α)).card_le_univ.trans_eq (by simp)
  omega

/-- **`Fintype.card α` closure steps already contain every orbit.** This is
    what lets the reachability question be answered by ONE finite `Finset`
    rather than an unbounded search. -/
theorem orbit_le_card_stable (G : Finset (α → α)) (s : α) (n : ℕ) :
    orbit G s n ⊆ orbit G s (Fintype.card α) := by
  obtain ⟨m, hm, heq⟩ := exists_stable_step G s
  rcases le_or_lt n (Fintype.card α) with hn | hn
  · exact orbit_mono G s hn
  · have hcard : orbit G s (Fintype.card α) = orbit G s m := by
      have := orbit_stable_forward G s heq (Fintype.card α - m)
      rw [Nat.add_sub_cancel' hm.le] at this
      exact this
    have horb : orbit G s n = orbit G s m := by
      have := orbit_stable_forward G s heq (n - m)
      rw [Nat.add_sub_cancel' (hm.le.trans hn.le)] at this
      exact this
    rw [hcard, horb]

/-- **Reachability.** `t` is reachable from `s` by some finite number of
    applications of elements of `G`. A `Finset (α → α)` has no order, so
    nothing here privileges one composition order — H.1 and H.3 are about
    different objects: H.1 fixes a composition order over a linear index, H.3
    quantifies over how many times and which generators fire. -/
def Reachable (G : Finset (α → α)) (s t : α) : Prop := ∃ n, t ∈ orbit G s n

/-- **The reduction to one `Finset`.** The unbounded existential over `ℕ` is
    EQUIVALENT to membership in the single computed set
    `orbit G s (Fintype.card α)`. -/
theorem reachable_iff (G : Finset (α → α)) (s t : α) :
    Reachable G s t ↔ t ∈ orbit G s (Fintype.card α) := by
  constructor
  · rintro ⟨n, hn⟩
    exact orbit_le_card_stable G s n hn
  · intro ht
    exact ⟨Fintype.card α, ht⟩

/-- **TARGET: `finite_semigroup_never_decidable`.** Restricting the operator
    class to a finite semigroup on a finite state space turns reachability —
    the shape of question Rice's theorem forbids a general decision procedure
    for — into a decidable predicate, and the witness is a terminating
    computation (`Fintype.card α` closure steps), not `Classical.propDecidable`.
    `Fintype α` is used essentially, at `exists_stable_step`.

    Declared `def`, not `theorem`: `DecidablePred (Reachable G s)` is
    Type-valued, and `theorem` compiles Type-valued goals opaquely — that
    version proves a decision procedure exists and cannot run one. Which Rice
    precondition fails: a finite semigroup on a `Fintype` state space has only
    `Fintype.card α` configurations, so the orbit cannot strictly grow beyond
    that without repeating, and a repeat is a decision. The `#eval` pair at
    the foot of this file runs this instance. -/
def finite_semigroup_never_decidable (G : Finset (α → α)) (s : α) :
    DecidablePred (Reachable G s) := fun t =>
  decidable_of_iff (t ∈ orbit G s (Fintype.card α)) (reachable_iff G s t).symm

end Finite

/-- Demonstration generator, reaching: `not false = true`, so `true` is
    reached from `false` in one step. -/
def demoReachG : Finset (Bool → Bool) := {Bool.not}

/-- Demonstration generator, stalled: `id` never moves anything, so the orbit
    is stuck at `{false}` forever. -/
def demoStallG : Finset (Bool → Bool) := {id}

end PhaseH

/-! ## 7. PHASE H.1 — the product moved off `Finset.prod` onto `List.prod`

    Provenance: Phase-H.1 work order, targets `pathProdL`, the AGREEMENT
    LEMMA, `contraction_path_bounded`, `dilation_block_eq_sigma`. Each was
    built standalone in the Prove phase and is merged here only after the
    whole library rebuilt with every pre-existing declaration's axiom list
    unchanged.

    **THE BLOCKER THIS SECTION CLEARS, and it is a fact about the DEFINITION
    rather than about the mathematics.** `pathProd` above is a `Finset.prod`.
    `Finset.prod` is `protected def Finset.prod [CommMonoid β] (s : Finset α)
    (f : α → β) : β := (s.1.map f).prod` — it REQUIRES a `CommMonoid` to be
    well defined at all, so it cannot typecheck over a noncommuting codomain,
    and permutation-invariance over a `Finset` is vacuous because a `Finset`
    carries no order to permute. §6's `scalar_gate_commutes` /
    `matrix_gate_not_commutes` pair therefore said nothing about the shipped
    `pathProd`, and every matrix-gate sentence stayed unformalised.

    `pathProdL` is `List.prod`, which needs only a `Monoid`, so
    `Matrix n n R` for a noncommutative `Semiring R` typechecks.

    **WHAT PROTECTS THE EXISTING CORPUS.** `pathProd_eq_pathProdL` is the
    agreement lemma: the corpus's own `pathProd` — not a restated copy of it —
    equals `pathProdL` applied to the same gates over
    `(Ico (j+1) (i+1)).toList`. It is Mathlib's `Finset.prod_to_list`
    (`Mathlib/Algebra/BigOperators/Basic.lean`, `CommMonoid`-only) cited at the
    corpus's objects, not a duplicate bridge re-proved here.
    `gateList_multiset` is the anti-vacuity check on it: the list's underlying
    MULTISET is literally `(Ico (j+1) (i+1)).val.map gate`, i.e. the same
    multiset `Finset.prod` folds over, not a re-indexed or re-sorted one.

    **AND THE CORPUS IS RECOVERED, NOT MERELY COEXISTED WITH.** §H.1c derives
    four pre-existing results as statements about `pathProdL` by rewriting
    along the agreement lemma and then QUOTING the original — `pathProd_abs`,
    `pathProd_eq_zero_iff`, `no_prefix_scan_represents_a_zero_gate`,
    `constant_phase_gate_is_rope`. Nothing is re-proved. That is the sense in
    which the scalar family is the `CommMonoid` specialisation of the new
    mechanism rather than an orphan beside it, and
    `pathProdL_scalar_is_order_blind` closes the loop the other way: over the
    commutative codomain the `List.prod` is invariant under EVERY permutation
    of the window, so the scalar family provably cannot hold step order, which
    is what `pathProdMatrixL_order_sensitive` exhibits a matrix carry doing.

    **THE NORM IS NAMED, AND IT IS NOT `σ_max`.** `contraction_path_bounded`
    is stated in the `L∞`-operator norm (`Matrix.linftyOpSeminormedAddCommGroup`,
    activated `local` inside §H.1d only) because that is the matrix norm this
    Mathlib pin actually proves submultiplicative for a general entry ring
    (`Matrix.linfty_opNorm_mul`, `[NonUnitalSeminormedRing α]`). The plain
    entrywise sup norm carries no `norm_mul` field at all, and the Frobenius
    instance needs the stronger `[RCLike α]`. There is NO submultiplicative
    SPECTRAL-norm instance on `Matrix` in this pin, so the singular-value
    reading of `σ_max ≤ 1` is NOT what is formalised here, and that gap is
    stated rather than papered over.

    **THE CONCLUSION IS `≤ 1`, NOT SUBMULTIPLICATIVITY RENAMED.**
    `Matrix.linfty_opNorm_mul` needs no hypothesis and concludes
    `‖A*B‖ ≤ ‖A‖*‖B‖`; it says nothing about `1`.
    `contraction_hypothesis_is_load_bearing` proves the hypothesis-free version
    of §H.1d's statement is FALSE, by the witness `[2 • 1]` whose product has
    norm `2`. `contraction_path_bounded_noncommuting_witness` is the other
    side: the two `2×2` nilpotents of `matrix_gate_not_commutes` each have
    `L∞`-operator norm EXACTLY `1`, their two orderings give DIFFERENT
    products, and the bound holds for both — so the theorem has content on a
    genuinely noncommuting chain and is not carried by a commutative or
    degenerate instance.

    **WHAT `dilation_block_eq_sigma` DOES AND DOES NOT COVER.** The general
    DIAGONAL contraction `S : Fin n → ℝ` with entries in the closed `[0,1]`
    is the top-left block of an orthogonal matrix on the doubled index
    `Fin n ⊕ Fin n`. A non-diagonal contraction is NOT covered: the SVD
    reduction to the diagonal case is not formalised here.
    `dilation_2x2_rotation_by_arccos` ties the construction to the gate
    `m · e^{iθ}` by naming the angle: at one scalar, `θ = arccos S` gives
    `S = cos θ` and the off-diagonal `s = sin θ`. No `#eval` witness exists
    for either — `Real.sqrt` is `noncomputable` in this pin, so neither
    theorem can be run the way `finite_semigroup_never_decidable` can. The
    `#eval`-shaped witnesses in this section are the two `ℤ`-matrix products
    at the foot of §H.1a, which do reduce; the closest available check on the
    dilation is the exact rational instance `dilation_gate_instance_three_five`,
    closed by `norm_num` with no numerical tolerance. -/

namespace PhaseH1

/-! ### H.1a The product over a `List`, which a noncommuting codomain can type -/

/-- **TARGET `pathProdL`.** The path product of a list of monoid elements.
    `List.prod` needs only `[Monoid M]`; no commutativity appears in the
    signature and none is needed, which is the whole point — `Finset.prod`
    could not be written down here. -/
def pathProdL {M : Type*} [Monoid M] (L : List M) : M := L.prod

/-- The matrix instance, spelled to the work order's signature.
    `Matrix.semiring` supplies `Monoid (Matrix n n R)` through
    `Semiring → MonoidWithZero → Monoid`, so `R` is NOT assumed commutative
    and `n ≥ 2` typechecks. -/
def pathProdMatrixL {n : Type*} [Fintype n] [DecidableEq n] {R : Type*} [Semiring R]
    (L : List (Matrix n n R)) : Matrix n n R := pathProdL L

/-- **The new definition carries what the old one could not even be asked.**
    The same two `2×2` witnesses as `PhaseH.matrix_gate_not_commutes`: the
    two lists ARE permutations of each other, and `pathProdMatrixL` separates
    them. Over `pathProd` this statement is not false — it is unstatable. -/
theorem pathProdMatrixL_order_sensitive :
    ∃ A B : Matrix (Fin 2) (Fin 2) ℂ,
      ([A, B] : List (Matrix (Fin 2) (Fin 2) ℂ)).Perm [B, A]
        ∧ pathProdMatrixL [A, B] ≠ pathProdMatrixL [B, A] := by
  refine ⟨!![0, 1; 0, 0], !![0, 0; 1, 0], List.Perm.swap _ _ _, ?_⟩
  unfold pathProdMatrixL pathProdL
  simp only [List.prod_cons, List.prod_nil, mul_one, Matrix.mul_fin_two]
  intro h
  have h00 := congrFun (congrFun h 0) 0
  norm_num [Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons] at h00

/- `#eval` witnesses for `pathProdMatrixL`, over `ℤ` so the kernel reduces
   (`ℂ` does not). Entry `(0,0)` of the two reorderings: expected `1` then
   `0`. A `/-- -/` doc comment may not precede a `#eval`, hence the plain
   block comment. -/
#eval (pathProdMatrixL ([!![0, 1; 0, 0], !![0, 0; 1, 0]] : List (Matrix (Fin 2) (Fin 2) ℤ))) 0 0
#eval (pathProdMatrixL ([!![0, 0; 1, 0], !![0, 1; 0, 0]] : List (Matrix (Fin 2) (Fin 2) ℤ))) 0 0

/-! ### H.1b THE AGREEMENT LEMMA, stated over this file's own `pathProd` -/

/-- The window's gates as an ordered list, over the canonical representative
    `(Ico (j+1) (i+1)).toList`. -/
noncomputable def gateList (m θ : ℕ → ℝ) (i j : ℕ) : List ℂ :=
  (Ico (j + 1) (i + 1)).toList.map (fun k => gateOf (m k) (θ k))

/-- **TARGET: THE AGREEMENT LEMMA.** `pathProd` — the object all of §1–§4b's
    declarations are stated about, unchanged — IS a `pathProdL`. Proof: cite
    Mathlib's `Finset.prod_to_list`, which holds over any `CommMonoid` and is
    exactly "`List.prod` over a representative list equals `Finset.prod` over
    the same multiset". No duplicate bridge lemma is proved. -/
theorem pathProd_eq_pathProdL (m θ : ℕ → ℝ) (i j : ℕ) :
    pathProd m θ i j = pathProdL (gateList m θ i j) := by
  unfold pathProdL gateList pathProd
  exact (Finset.prod_to_list (Ico (j + 1) (i + 1)) (fun k => gateOf (m k) (θ k))).symm

/-- **The anti-vacuity check on the agreement lemma: SAME MULTISET, not a
    rewritten one.** `gateList`'s underlying multiset is literally
    `(Ico (j+1) (i+1)).val.map gate` — the multiset `Finset.prod` folds over.
    Without this, "equals the product over the same multiset" would be a
    claim about the prose and not about the terms. -/
theorem gateList_multiset (m θ : ℕ → ℝ) (i j : ℕ) :
    (gateList m θ i j : Multiset ℂ)
      = (Ico (j + 1) (i + 1)).val.map (fun k => gateOf (m k) (θ k)) := by
  unfold gateList
  rw [← Multiset.map_coe, Finset.coe_toList]

/-! ### H.1c The pre-existing corpus RECOVERED through the agreement lemma

    Each of the four below is `rw [← pathProd_eq_pathProdL]` followed by
    QUOTING the original declaration. Nothing above this section is restated,
    weakened, or re-proved: the scalar family is recovered as the `CommMonoid`
    case of the `List.prod` mechanism, which is the continuity claim the page
    needs in order to add matrix-gate sentences. -/

/-- `pathProd_abs`, recovered on `pathProdL`. -/
theorem pathProdL_abs (m θ : ℕ → ℝ) (h0 : ∀ k, 0 ≤ m k) (i j : ℕ) :
    Complex.abs (pathProdL (gateList m θ i j)) = ∏ k in Ico (j + 1) (i + 1), m k := by
  rw [← pathProd_eq_pathProdL]; exact pathProd_abs m θ h0 i j

/-- `pathProd_eq_zero_iff`, recovered on `pathProdL`: annihilation at a zero
    magnitude survives the move off `Finset.prod`. -/
theorem pathProdL_eq_zero_iff (m θ : ℕ → ℝ) (i j : ℕ) :
    pathProdL (gateList m θ i j) = 0 ↔ ∃ k ∈ Ico (j + 1) (i + 1), m k = 0 := by
  rw [← pathProd_eq_pathProdL]; exact pathProd_eq_zero_iff m θ i j

/-- `no_prefix_scan_represents_a_zero_gate`, recovered on `pathProdL`: the
    file's central cost is not an artifact of the `Finset` shape. -/
theorem pathProdL_no_prefix_scan (C : ℕ → ℂ) (m θ : ℕ → ℝ) {i j : ℕ}
    (hz : ∃ k ∈ Ico (j + 1) (i + 1), m k = 0) :
    Complex.exp (C i - C j) ≠ pathProdL (gateList m θ i j) := by
  rw [← pathProd_eq_pathProdL]; exact no_prefix_scan_represents_a_zero_gate C m θ hz

/-- `constant_phase_gate_is_rope`, recovered on `pathProdL`. -/
theorem pathProdL_is_rope (ω : ℝ) {i j : ℕ} (hij : j ≤ i) :
    pathProdL (gateList (fun _ => (1 : ℝ)) (fun _ => ω) i j)
      = Complex.exp (((ω * ((i : ℝ) - (j : ℝ)) : ℝ) : ℂ) * Complex.I) := by
  rw [← pathProd_eq_pathProdL]; exact constant_phase_gate_is_rope ω hij

/-- **The loop closed the other way, and this is what §6's `PhaseH` docstring
    could only assert.** `pathProd` equals the `List.prod` of the same gates
    over EVERY permutation of the window, not just the canonical one — so the
    scalar carry provably holds no step order. Read against
    `pathProdMatrixL_order_sensitive`, which exhibits a matrix carry that
    does, this is now a statement about the SHIPPED product rather than about
    a separate combinatorial object. -/
theorem pathProdL_scalar_is_order_blind (m θ : ℕ → ℝ) {i j : ℕ} {L : List ℕ}
    (hperm : (Ico (j + 1) (i + 1)).toList.Perm L) :
    pathProd m θ i j = pathProdL (L.map (fun k => gateOf (m k) (θ k))) := by
  rw [pathProd_eq_pathProdL]
  unfold pathProdL gateList
  exact (hperm.map (fun k => gateOf (m k) (θ k))).prod_eq

/-! ### H.1d The contraction bound on the operator axis -/

section Contraction

/- The `L∞`-operator norm on square matrices — the one
   `Matrix.linfty_opNorm_mul` is proved against — is declared
   `@[local instance]` inside `Mathlib.Analysis.Matrix`, so it does not leak
   in from the import. These two lines activate it for THIS SECTION ONLY, so
   no declaration outside §H.1d sees a `Norm (Matrix n n α)` instance.
   Plain block comments: `attribute` is a command and cannot carry a `/-- -/`. -/
attribute [local instance] Matrix.linftyOpSeminormedAddCommGroup
attribute [local instance] Matrix.linftyOpNormedSpace

/-- **TARGET `contraction_path_bounded`.** If EVERY factor of the list has
    operator norm at most `1`, the `List.prod` of the whole chain does too.
    The chain is submultiplicativity (`Matrix.linfty_opNorm_mul`, lifted to
    arbitrary length by induction on the list) and then the per-factor bound.
    This is `prefix_logit_mask_restated`'s clause 2 — `|∏ a| ≤ 1` from
    `m k ≤ 1` — carried onto the operator axis, and commutativity of the
    codomain is never used. -/
theorem contraction_path_bounded {n : Type*} [Fintype n] [DecidableEq n] [Nonempty n]
    {α : Type*} [NormedRing α] [NormOneClass α]
    (L : List (Matrix n n α)) (h : ∀ T ∈ L, ‖T‖ ≤ 1) :
    ‖pathProdMatrixL L‖ ≤ 1 := by
  unfold pathProdMatrixL pathProdL
  induction L with
  | nil => rw [List.prod_nil]; exact le_of_eq norm_one
  | cons T L ih =>
      rw [List.prod_cons]
      have hT : ‖T‖ ≤ 1 := h T (List.mem_cons_self T L)
      have hL : ‖L.prod‖ ≤ 1 := ih (fun S hS => h S (List.mem_cons_of_mem T hS))
      calc ‖T * L.prod‖ ≤ ‖T‖ * ‖L.prod‖ := Matrix.linfty_opNorm_mul T L.prod
        _ ≤ 1 * 1 := mul_le_mul hT hL (norm_nonneg _) (by norm_num)
        _ = 1 := by norm_num

/-- **REFUSED: `contraction_path_bounded` without its per-factor hypothesis.**
    Deleting `h` does not give submultiplicativity under a new name; it gives
    a FALSE statement, and here is the witness. `[2 • 1]` is a one-element
    chain whose product has operator norm `2`. So the hypothesis
    `∀ T ∈ L, ‖T‖ ≤ 1` is load-bearing and the conclusion `≤ 1` is not
    carried by `Matrix.linfty_opNorm_mul` alone. -/
theorem contraction_hypothesis_is_load_bearing :
    ∃ L : List (Matrix (Fin 2) (Fin 2) ℝ), ¬ ‖pathProdMatrixL L‖ ≤ 1 := by
  refine ⟨[(2 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)], ?_⟩
  unfold pathProdMatrixL pathProdL
  rw [List.prod_cons, List.prod_nil, mul_one, norm_smul, norm_one]
  norm_num

/-- **And the bound has content on a genuinely noncommuting chain.** The two
    `2×2` nilpotents of `PhaseH.matrix_gate_not_commutes` have operator norm
    EXACTLY `1` each — so they satisfy the hypothesis at its boundary, not
    with slack — the two orderings of the chain give DIFFERENT products, and
    `contraction_path_bounded` bounds both. Filed so the theorem cannot be
    read as holding only on commutative or degenerate instances. -/
theorem contraction_path_bounded_noncommuting_witness :
    ‖(!![0, 1; 0, 0] : Matrix (Fin 2) (Fin 2) ℝ)‖ = 1
      ∧ ‖(!![0, 0; 1, 0] : Matrix (Fin 2) (Fin 2) ℝ)‖ = 1
      ∧ pathProdMatrixL ([!![0, 1; 0, 0], !![0, 0; 1, 0]] : List (Matrix (Fin 2) (Fin 2) ℝ))
          ≠ pathProdMatrixL ([!![0, 0; 1, 0], !![0, 1; 0, 0]] : List (Matrix (Fin 2) (Fin 2) ℝ))
      ∧ ‖pathProdMatrixL ([!![0, 1; 0, 0], !![0, 0; 1, 0]]
            : List (Matrix (Fin 2) (Fin 2) ℝ))‖ ≤ 1
      ∧ ‖pathProdMatrixL ([!![0, 0; 1, 0], !![0, 1; 0, 0]]
            : List (Matrix (Fin 2) (Fin 2) ℝ))‖ ≤ 1 := by
  have hA : ‖(!![0, 1; 0, 0] : Matrix (Fin 2) (Fin 2) ℝ)‖ = 1 := by
    rw [Matrix.linfty_opNorm_def]
    norm_num [Fin.sum_univ_succ, Fin.univ_succ, Finset.sup_insert, Finset.sup_singleton]
  have hB : ‖(!![0, 0; 1, 0] : Matrix (Fin 2) (Fin 2) ℝ)‖ = 1 := by
    rw [Matrix.linfty_opNorm_def]
    norm_num [Fin.sum_univ_succ, Fin.univ_succ, Finset.sup_insert, Finset.sup_singleton]
  refine ⟨hA, hB, ?_, ?_, ?_⟩
  · unfold pathProdMatrixL pathProdL
    simp only [List.prod_cons, List.prod_nil, mul_one, Matrix.mul_fin_two]
    intro h
    have h00 := congrFun (congrFun h 0) 0
    norm_num [Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.head_cons] at h00
  · refine contraction_path_bounded _ ?_
    intro T hT
    rcases List.mem_cons.mp hT with rfl | hT
    · exact le_of_eq hA
    · rcases List.mem_cons.mp hT with rfl | hT
      · exact le_of_eq hB
      · simp at hT
  · refine contraction_path_bounded _ ?_
    intro T hT
    rcases List.mem_cons.mp hT with rfl | hT
    · exact le_of_eq hB
    · rcases List.mem_cons.mp hT with rfl | hT
      · exact le_of_eq hA
      · simp at hT

end Contraction

/-! ### H.1e The dilation -/

section Dilation

open Matrix

/-- **TARGET `dilation_block_eq_sigma`.** A DIAGONAL contraction
    `S : Fin n → ℝ` with entries in the CLOSED `[0,1]` is the top-left block
    of an orthogonal matrix on the doubled index: with
    `s i = sqrt (1 - (S i)^2)`,
    `D = fromBlocks (diag S) (diag s) (-(diag s)) (diag S)` satisfies
    `Dᵀ * D = 1` and `D (inl i) (inl j) = diagonal S i j`. Both endpoints of
    the band are ordinary points: `S i = 0` and `S i = 1` are admitted, not
    excluded. NOT covered: a non-diagonal contraction — the SVD reduction to
    this case is not formalised. -/
theorem dilation_block_eq_sigma {n : ℕ} (S : Fin n → ℝ)
    (hS0 : ∀ i, 0 ≤ S i) (hS1 : ∀ i, S i ≤ 1) :
    ∃ D : Matrix (Fin n ⊕ Fin n) (Fin n ⊕ Fin n) ℝ,
      Dᵀ * D = 1 ∧ ∀ i j : Fin n, D (Sum.inl i) (Sum.inl j) = Matrix.diagonal S i j := by
  set s : Fin n → ℝ := fun i => Real.sqrt (1 - (S i) ^ 2)
  have hs_sq : ∀ i, (s i) ^ 2 = 1 - (S i) ^ 2 := fun i =>
    Real.sq_sqrt (by nlinarith [hS0 i, hS1 i])
  have hsum1 : ∀ i, S i * S i + s i * s i = 1 := fun i => by
    have h := hs_sq i; nlinarith [h]
  have hsum2 : ∀ i, s i * s i + S i * S i = 1 := fun i => by
    have h := hs_sq i; nlinarith [h]
  refine ⟨Matrix.fromBlocks (Matrix.diagonal S) (Matrix.diagonal s)
      (-(Matrix.diagonal s)) (Matrix.diagonal S), ?_, ?_⟩
  · rw [Matrix.fromBlocks_transpose]
    simp only [Matrix.diagonal_transpose, Matrix.transpose_neg]
    rw [Matrix.fromBlocks_multiply]
    have e1 : Matrix.diagonal S * Matrix.diagonal S
        + -Matrix.diagonal s * -Matrix.diagonal s = (1 : Matrix (Fin n) (Fin n) ℝ) := by
      rw [neg_mul_neg, Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal,
        Matrix.diagonal_add]
      have hfun : (fun i => S i * S i + s i * s i) = (fun _ : Fin n => (1 : ℝ)) :=
        funext hsum1
      rw [hfun]; exact Matrix.diagonal_one
    have e2 : Matrix.diagonal S * Matrix.diagonal s
        + -Matrix.diagonal s * Matrix.diagonal S = (0 : Matrix (Fin n) (Fin n) ℝ) := by
      rw [neg_mul, Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal,
        Matrix.diagonal_neg, Matrix.diagonal_add]
      have hfun : (fun i => S i * s i + -(s i * S i)) = (fun _ : Fin n => (0 : ℝ)) :=
        funext (fun i => by ring)
      rw [hfun]; exact Matrix.diagonal_zero
    have e3 : Matrix.diagonal s * Matrix.diagonal S
        + Matrix.diagonal S * -Matrix.diagonal s = (0 : Matrix (Fin n) (Fin n) ℝ) := by
      rw [mul_neg, Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal,
        Matrix.diagonal_neg, Matrix.diagonal_add]
      have hfun : (fun i => s i * S i + -(S i * s i)) = (fun _ : Fin n => (0 : ℝ)) :=
        funext (fun i => by ring)
      rw [hfun]; exact Matrix.diagonal_zero
    have e4 : Matrix.diagonal s * Matrix.diagonal s
        + Matrix.diagonal S * Matrix.diagonal S = (1 : Matrix (Fin n) (Fin n) ℝ) := by
      rw [Matrix.diagonal_mul_diagonal, Matrix.diagonal_mul_diagonal, Matrix.diagonal_add]
      have hfun : (fun i => s i * s i + S i * S i) = (fun _ : Fin n => (1 : ℝ)) :=
        funext hsum2
      rw [hfun]; exact Matrix.diagonal_one
    rw [e1, e2, e3, e4]
    exact Matrix.fromBlocks_one
  · intro i j
    exact Matrix.fromBlocks_apply₁₁ _ _ _ _ i j

/-- **The `2×2` case, with the ANGLE named, which is what ties the dilation to
    this file's gate.** For one contraction scalar `S ∈ [0,1]`,
    `D = !![S, s; -s, S]` with `s = sqrt (1 - S^2)` is orthogonal, its
    top-left entry is `S`, and it IS the rotation by `θ = arccos S`:
    `S = cos θ`, `s = sin θ`. `gateOf m θ` at `m = 1` is exactly this
    rotation. Proved independently of `dilation_block_eq_sigma` rather than by
    specializing `Fin n ⊕ Fin n` to `Fin 1 ⊕ Fin 1`, so a defect in one does
    not hide inside the other. -/
theorem dilation_2x2_rotation_by_arccos (S : ℝ) (hS0 : 0 ≤ S) (hS1 : S ≤ 1) :
    ∃ D : Matrix (Fin 2) (Fin 2) ℝ, ∃ s : ℝ,
      D = !![S, s; -s, S]
        ∧ Dᵀ * D = 1
        ∧ D 0 0 = S
        ∧ s = Real.sqrt (1 - S ^ 2)
        ∧ S = Real.cos (Real.arccos S)
        ∧ s = Real.sin (Real.arccos S) := by
  refine ⟨!![S, Real.sqrt (1 - S ^ 2); -Real.sqrt (1 - S ^ 2), S], Real.sqrt (1 - S ^ 2),
    rfl, ?_, rfl, rfl, (Real.cos_arccos (by linarith) hS1).symm, (Real.sin_arccos S).symm⟩
  set s : ℝ := Real.sqrt (1 - S ^ 2) with hs_def
  have hs_sq : s * s = 1 - S * S := by
    rw [hs_def, Real.mul_self_sqrt (by nlinarith : (0:ℝ) ≤ 1 - S ^ 2)]; ring
  have hDT : (!![S, s; -s, S] : Matrix (Fin 2) (Fin 2) ℝ)ᵀ = !![S, -s; s, S] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  rw [hDT, Matrix.mul_fin_two]
  have h1 : S * S + -s * -s = (1 : ℝ) := by rw [neg_mul_neg]; linarith [hs_sq]
  have h2 : S * s + -s * S = (0 : ℝ) := by ring
  have h3 : s * S + S * -s = (0 : ℝ) := by ring
  have h4 : s * s + S * S = (1 : ℝ) := by linarith [hs_sq]
  rw [h1, h2, h3, h4]
  exact Matrix.one_fin_two.symm

/-- **The closest thing to an `#eval` witness available for the dilation.**
    `Real.sqrt` is `noncomputable` in this pin, so no instance of the two
    theorems above can be RUN. `S = 3/5` is a Pythagorean triple, so its
    complementary magnitude is `4/5` on the nose and the instance closes by
    `norm_num` with no numerical tolerance — a checked exact instance rather
    than a kernel reduction, and stated as such. -/
theorem dilation_gate_instance_three_five :
    Real.sqrt (1 - (3 / 5 : ℝ) ^ 2) = 4 / 5
      ∧ ((3 / 5 : ℝ)) ^ 2 + ((4 / 5 : ℝ)) ^ 2 = 1 := by
  refine ⟨?_, by norm_num⟩
  rw [show (1 : ℝ) - (3 / 5 : ℝ) ^ 2 = (4 / 5 : ℝ) ^ 2 by norm_num]
  rw [Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 4/5)]

end Dilation

end PhaseH1


end CEQ.V16Domain

/-! ## The census, printed -/

-- BED-M support `{-1, 0, +1}` against `CEQ.V15.prefix_logit_mask`'s `0 < a`: 1 of 3.
#eval CEQ.V16Domain.bedM.countP CEQ.V16Domain.satOldTwo

-- BED-M support against the re-stated #2's `0 <= |a| <= 1`: 3 of 3.
#eval CEQ.V16Domain.bedM.countP CEQ.V16Domain.satNewTwo

-- BED-M support against #6's image `(0,1)` open: 0 of 3.
#eval CEQ.V16Domain.bedM.countP CEQ.V16Domain.satSix

/-! ## Axiom check — every declaration in this file -/

#print axioms CEQ.V16Domain.gateOf
#print axioms CEQ.V16Domain.pathProd
#print axioms CEQ.V16Domain.abs_gateOf
#print axioms CEQ.V16Domain.pathProd_polar
#print axioms CEQ.V16Domain.pathProd_abs
#print axioms CEQ.V16Domain.pathProd_eq_zero_iff
#print axioms CEQ.V16Domain.lean_log_junk_makes_the_scan_form_silently_false
#print axioms CEQ.V16Domain.no_prefix_scan_represents_a_zero_gate
#print axioms CEQ.V16Domain.pathProd_eq_Wp
#print axioms CEQ.V16Domain.restatement_subsumes_phase_path_le_one
#print axioms CEQ.V16Domain.prod_eq_one_iff_of_nonneg
#print axioms CEQ.V16Domain.prefix_logit_mask_restated
#print axioms CEQ.V16Domain.magOf
#print axioms CEQ.V16Domain.argOf
#print axioms CEQ.V16Domain.bedM_gate_exact
#print axioms CEQ.V16Domain.negative_draw_is_on_the_band
#print axioms CEQ.V16Domain.bedM
#print axioms CEQ.V16Domain.bedMProp
#print axioms CEQ.V16Domain.satOldTwo
#print axioms CEQ.V16Domain.satNewTwo
#print axioms CEQ.V16Domain.satSix
#print axioms CEQ.V16Domain.bedM_overlap_old_two
#print axioms CEQ.V16Domain.bedM_overlap_new_two
#print axioms CEQ.V16Domain.bedM_overlap_six
#print axioms CEQ.V16Domain.bedMProp_overlap_old_two
#print axioms CEQ.V16Domain.bedMProp_overlap_new_two
#print axioms CEQ.V16Domain.bedMProp_overlap_six
#print axioms CEQ.V16Domain.six_misses_every_bedM_value
#print axioms CEQ.V16Domain.delay_zero_is_first_order
#print axioms CEQ.V16Domain.sixteen_is_silent_on_the_zero_draw
#print axioms CEQ.V16Domain.constant_phase_gate_is_rope
#print axioms CEQ.V16Domain.cumulative_phase_is_separable
#print axioms CEQ.V16Domain.magnitude_zero_not_separable
#print axioms CEQ.V16Domain.num
#print axioms CEQ.V16Domain.Znorm
#print axioms CEQ.V16Domain.Hop
#print axioms CEQ.V16Domain.softmaxAttn
#print axioms CEQ.V16Domain.linearAttn
#print axioms CEQ.V16Domain.scan_zero_of_zero
#print axioms CEQ.V16Domain.corner_softmax
#print axioms CEQ.V16Domain.corner_linear
#print axioms CEQ.V16Domain.corner_path_product
#print axioms CEQ.V16Domain.corner_path_product_is_the_gate_product
#print axioms CEQ.V16Domain.path_product_corner_fails_at_a_zero_gate
#print axioms CEQ.V16Domain.three_corners_containment
#print axioms CEQ.V16Domain.corners_are_distinct
#print axioms CEQ.V16Domain.softmax_row_sum_one
#print axioms CEQ.V16Domain.gate_zero_beta_zero_is_linear_attention
#print axioms CEQ.V16Domain.gate_zero_beta_zero_row_not_one
#print axioms CEQ.V16Domain.beta_one_row_is_one

/-! ## TRAIN-GATE axiom check, under the re-statement

    `CEQ_V16_CONTRACT.md` PART II: `#1, #2(re-stated), #3, #5a, #6, #7, #16`
    green. Each is printed here from THIS file, so the gate is checked in the
    presence of the re-statement rather than beside it. `#2` is printed twice —
    the `0 < a` original and the `[0,1]` re-statement — so the pair is
    diffable. `#5b` is printed with `#5a` because the refutation and the
    containment stand or fall together. -/

#print axioms CEQ.V15.chain_path_product                        -- #1
#print axioms CEQ.V15.prefix_logit_mask                         -- #2, original, 0 < a
#print axioms CEQ.V16Domain.prefix_logit_mask_restated          -- #2, re-stated, m ∈ [0,1]
#print axioms CEQ.V15.parity_sign                               -- #3
#print axioms CEQ.V16Domain.three_corners_containment           -- #5a
#print axioms CEQ.V16Domain.gate_zero_beta_zero_is_linear_attention  -- #5b
#print axioms CEQ.V15.bounded_gates_stable                      -- #6
#print axioms CEQ.V15.scan_assoc                                -- #7
#print axioms CEQ.V15Phase.unit_phase_product                   -- #16

/-! ## Phase H, printed and RUN

    The `#eval` pair is the point of `finite_semigroup_never_decidable` being
    a `def`: the decision procedure is executed here, not merely asserted to
    exist. Expected `true` then `false`. -/

#print axioms CEQ.V16Domain.PhaseH.scalar_gate_commutes
#print axioms CEQ.V16Domain.PhaseH.matrix_gate_not_commutes
#print axioms CEQ.V16Domain.PhaseH.idempotent_ne_one_not_isUnit
#print axioms CEQ.V16Domain.PhaseH.projector_idempotent_not_invertible
#print axioms CEQ.V16Domain.PhaseH.projector_det_eq_zero
#print axioms CEQ.V16Domain.PhaseH.exists_stable_step
#print axioms CEQ.V16Domain.PhaseH.orbit_le_card_stable
#print axioms CEQ.V16Domain.PhaseH.reachable_iff
#print axioms CEQ.V16Domain.PhaseH.finite_semigroup_never_decidable

#eval @Decidable.decide
  (CEQ.V16Domain.PhaseH.Reachable CEQ.V16Domain.PhaseH.demoReachG false true)
  (CEQ.V16Domain.PhaseH.finite_semigroup_never_decidable
    CEQ.V16Domain.PhaseH.demoReachG false true)

#eval @Decidable.decide
  (CEQ.V16Domain.PhaseH.Reachable CEQ.V16Domain.PhaseH.demoStallG false true)
  (CEQ.V16Domain.PhaseH.finite_semigroup_never_decidable
    CEQ.V16Domain.PhaseH.demoStallG false true)

/-! ## Phase H.1, printed

    17 declarations. The `#eval` pair at the foot of §H.1a is inside the file
    body above; it prints `1` then `0` — the `(0,0)` entries of the two
    reorderings of the same noncommuting chain. -/

#print axioms CEQ.V16Domain.PhaseH1.pathProdL
#print axioms CEQ.V16Domain.PhaseH1.pathProdMatrixL
#print axioms CEQ.V16Domain.PhaseH1.pathProdMatrixL_order_sensitive
#print axioms CEQ.V16Domain.PhaseH1.gateList
#print axioms CEQ.V16Domain.PhaseH1.pathProd_eq_pathProdL
#print axioms CEQ.V16Domain.PhaseH1.gateList_multiset
#print axioms CEQ.V16Domain.PhaseH1.pathProdL_abs
#print axioms CEQ.V16Domain.PhaseH1.pathProdL_eq_zero_iff
#print axioms CEQ.V16Domain.PhaseH1.pathProdL_no_prefix_scan
#print axioms CEQ.V16Domain.PhaseH1.pathProdL_is_rope
#print axioms CEQ.V16Domain.PhaseH1.pathProdL_scalar_is_order_blind
#print axioms CEQ.V16Domain.PhaseH1.contraction_path_bounded
#print axioms CEQ.V16Domain.PhaseH1.contraction_hypothesis_is_load_bearing
#print axioms CEQ.V16Domain.PhaseH1.contraction_path_bounded_noncommuting_witness
#print axioms CEQ.V16Domain.PhaseH1.dilation_block_eq_sigma
#print axioms CEQ.V16Domain.PhaseH1.dilation_2x2_rotation_by_arccos
#print axioms CEQ.V16Domain.PhaseH1.dilation_gate_instance_three_five
