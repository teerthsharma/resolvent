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
