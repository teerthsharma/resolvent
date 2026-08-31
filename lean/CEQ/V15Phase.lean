/-
  CEQ.V15Phase — LEAN #16, `unit_phase_product`, and the construction it is a
  corollary of
  ----------------------------------------------------------------------------
  Provenance: `CEQ_V15_3_DELTA.md` §X₃₆ (THE GATE IS A WAVE) and §LEAN #16.
  Companion report: `V15_X36_LEAN.md`.

  WHAT IS REFUSED AS THE STATEMENT.

  The delta writes #16 as `|Π e^{iθ_k}| = 1`. That is `unit_phase_product`
  below, it is three lines, and it is true of every phase in every context —
  RoPE, a Fourier basis, a random unitary, the number `1`. It separates this
  architecture from nothing. `unit_phase_does_not_bound_the_gate` is recorded
  immediately after it and gives the modulus of a unit-phase gate as
  `285.07` — the exact `â_max` R1 measured on its worst divergent seed. The
  phase is not what bounds the arm. `V15_N3_LEAN.md`'s standard — trivial
  statements refused in the file, not quietly shipped — is why both appear.

  WHAT IS PROVED INSTEAD.

  `a_k = m_k · e^{iθ_k}`, prefix-scanned as `C_i = Σ_{k≤i} (log m_k + i·θ_k)`.

  | theorem                            | what it says |
  |------------------------------------|--------------|
  | `Wp_polar`                         | `exp(C_i − C_j) = (∏_{k=j+1}^{i} m_k) · exp(i·Σ_{k=j+1}^{i} θ_k)`: the carrier splits exactly, magnitude from the `m`, phase from the `θ`, no interaction term |
  | `phase_modulus_is_the_real_carrier`| `|exp(C_i − C_j)| = CEQ.V15.W (log ∘ m) i j` for ALL `i j`, no hypothesis: the modulus of the phase carrier IS #2's real carrier, unchanged |
  | `prefix_phase_modulus`             | `|exp(C_i − C_j)| = ∏_{k=j+1}^{i} m_k`, `m > 0` the only hypothesis, `θ` free |
  | `phase_path_le_one`                | `m ≤ 1` pointwise ⇒ `|path| ≤ 1`, **for every** `θ` — the θ are universally quantified inside the statement, so no phase schedule can make it diverge |
  | `phase_path_eq_one_iff_band`       | `|path| = 1` ↔ `m ≡ 1` on that path, an iff, not an implication |

  The last two are the pair the delta's "≤ 1 by construction, = 1 on the band"
  is asking for, and they are what upgrades #6's inequality: `bounded_gates_*`
  bounds a product of positive reals, and says nothing at all once the gates are
  complex; `phase_path_le_one` bounds the modulus of a complex path product
  with no hypothesis on the phases.

  WHAT THE FILE FINDS AGAINST THE DELTA.

  **(1) #6's own parametrization cannot reach the band.**
  `six_never_reaches_the_band`: for `m_k = exp(−softplus w_k)` the modulus of
  every non-empty path is `< 1`, STRICTLY, for every `w` and every `θ`. The
  delta's "`on the band m ≡ 1` this becomes `= 1`" is therefore NOT a corollary
  of #6 — `m ≡ 1` is not in the image of `w ↦ exp(−softplus w)`. The equality
  needs the hard cap the delta calls for and #6 does not have, and
  `cap_band_attains` is the equality once the cap is in place. Same defect,
  same proof, for LRU's `λ = exp(−exp ν + iθ)` (`lru_modulus_lt_one`): its
  magnitude is open at `1`, which is exactly the "small delta" §X₃₆ claims over
  it, here at the strength of a strict inequality rather than a remark.

  **(2) `parity mask = Z₂ winding` is TRUE only after `exp`, and the delta's
  own phrasing of it is FALSE.** `parity_is_Z2_winding` proves the agreement in
  the only place both sides live: `exp(i·Σ_{path} θ) = χ(P_i − P_j)` with
  `θ ∈ {0, π}`. The delta writes "the prefix sum of phases mod 2π corresponds
  to the prefix-XOR of sign bits"; dropped to the phases themselves the
  correspondence fails, and `phase_sum_is_not_the_parity_phase` is the
  counterexample — `p ≡ 1` on `{1,2}` sums to `2π` while the parity bit is `0`,
  whose phase is `0`, and `2π ≠ 0`. This is not cosmetic: §X₃₇(a) asks for the
  **integer** `Z` winding, and `Z2_forgets_the_winding` exhibits a path that has
  wound twice and whose parity character is `1`. The parity mask is the
  reduction of the winding mod 2 and does not determine it. `[RUN: True]` in
  the delta is a true reading of a coarser statement than §X₃₇(a) needs.

  WHAT IS NOT CLAIMED. Nothing here says the phase gate trains, converges, or
  that R1's 8/8 pre-registration holds. `|a| ≤ 1` is proved for the carrier as
  parametrized; whether the optimizer reaches a useful `m` inside the cap, and
  whether the crossing survives the cap (the author's own counter-prediction),
  are measurements this file cannot make. The theorems are about one path in
  one channel: no multi-channel coupling, no Kuramoto order parameter, no
  `β₁`, and no claim that the discrete `Z₂` phase is trainable through the
  non-differentiable `{0, π}` restriction.

  No `sorry`. `#print axioms` on every declaration at the foot of the file.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.BigOperators
import CEQ.V15

namespace CEQ.V15Phase

open BigOperators Finset

/-! ## #16 as the delta writes it, and the refusal -/

/-- **#16 `unit_phase_product`.** The modulus of a finite product of
    unit-modulus complex numbers is `1`.

    Stated over `ℂ` rather than over `Circle` deliberately: the gates of §X₃₆
    are `m·e^{iθ}` and only the `θ` factor lives on the circle, so a statement
    inside the subgroup would not compose with the magnitude. Every hypothesis
    is discharged; there are none. -/
theorem unit_phase_product (θ : ℕ → ℝ) (s : Finset ℕ) :
    Complex.abs (∏ k in s, Complex.exp ((θ k : ℂ) * Complex.I)) = 1 := by
  rw [Complex.abs_prod]
  exact Finset.prod_eq_one fun k _ => Complex.abs_exp_ofReal_mul_I (θ k)

/-- **REFUSED as the content of #16.** A unit phase bounds the phase and
    nothing else. Here is a gate whose phase factor has modulus exactly `1` and
    whose modulus is `285.07` — R1's measured `â_max` on its worst divergent
    seed, the number X₃₆ exists to remove. What removes it is the cap on `m`;
    `unit_phase_product` is silent about `m` and therefore silent about the
    divergence. -/
theorem unit_phase_does_not_bound_the_gate (θ : ℝ) :
    Complex.abs (Complex.exp ((θ : ℂ) * Complex.I)) = 1 ∧
      Complex.abs ((285.07 : ℝ) * Complex.exp ((θ : ℂ) * Complex.I)) = 285.07 := by
  refine ⟨Complex.abs_exp_ofReal_mul_I θ, ?_⟩
  rw [map_mul, Complex.abs_ofReal, Complex.abs_exp_ofReal_mul_I, mul_one]
  norm_num

/-! ## The magnitude: closed by a cap, open in both prior parametrizations -/

/-- The hard cap of §X₃₆: `m ∈ [0,1]` CLOSED, so `0` and `1` are attained
    values and not limits. -/
noncomputable def cap (x : ℝ) : ℝ := min 1 (max 0 x)

lemma cap_nonneg (x : ℝ) : 0 ≤ cap x :=
  le_min zero_le_one (le_max_left 0 x)

lemma cap_le_one (x : ℝ) : cap x ≤ 1 := min_le_left _ _

/-- The closed end that the whole of X₃₆ turns on: `1` is attained. -/
lemma cap_eq_one_of_one_le {x : ℝ} (hx : 1 ≤ x) : cap x = 1 :=
  min_eq_left (le_max_of_le_right hx)

lemma cap_eq_zero_of_le_zero {x : ℝ} (hx : x ≤ 0) : cap x = 0 := by
  rw [cap, max_eq_left hx, min_eq_right zero_le_one]

/-- **#6's parametrization is open at `1`.** `exp(−softplus w) < 1` strictly,
    for every real `w`. `CEQ.V15.bounded_gates_stable` states `≤ 1`; the gap
    between `≤` and `<` is the whole of the band claim, and this lemma says the
    band is on the wrong side of it. -/
theorem softplus_gate_lt_one (w : ℝ) : Real.exp (-CEQ.V15.softplus w) < 1 :=
  Real.exp_lt_one_iff.mpr (by linarith [CEQ.V15.softplus_pos w])

/-- **LRU (Orvieto 2023) is open at `1` too.** `λ = exp(−exp ν + iθ)` has
    modulus `exp(−exp ν) < 1` for every `ν` and every `θ`. Recorded as the
    prior-art comparison §X₃₆ makes, at the strength of a proof: the delta's
    "OPEN magnitude" is this strict inequality. -/
theorem lru_modulus_lt_one (ν θ : ℝ) :
    Complex.abs (Complex.exp (((-Real.exp ν : ℝ) : ℂ) + (θ : ℂ) * Complex.I)) < 1 := by
  rw [Complex.abs_exp]
  simp only [Complex.add_re, Complex.ofReal_re, Complex.mul_I_re, Complex.ofReal_im, neg_zero,
    add_zero]
  exact Real.exp_lt_one_iff.mpr (by linarith [Real.exp_pos ν])

/-! ## Two facts about products of gains in `[0,1]`

    Both are about `ℝ`; they are separated out because the complex statements
    below reduce to them exactly, which is the point of `Wp_polar`. -/

lemma prod_lt_one_of_mem {s : Finset ℕ} {f : ℕ → ℝ} (h0 : ∀ k ∈ s, 0 < f k)
    (h1 : ∀ k ∈ s, f k ≤ 1) {k₀ : ℕ} (hk₀ : k₀ ∈ s) (hlt : f k₀ < 1) :
    ∏ k in s, f k < 1 := by
  classical
  rw [← Finset.mul_prod_erase _ _ hk₀]
  have hP1 : ∏ k in s.erase k₀, f k ≤ 1 :=
    Finset.prod_le_one (fun k hk => (h0 k (Finset.mem_of_mem_erase hk)).le)
      (fun k hk => h1 k (Finset.mem_of_mem_erase hk))
  have hP0 : 0 < ∏ k in s.erase k₀, f k :=
    Finset.prod_pos fun k hk => h0 k (Finset.mem_of_mem_erase hk)
  nlinarith [h0 k₀ hk₀]

lemma prod_eq_one_iff {s : Finset ℕ} {f : ℕ → ℝ} (h0 : ∀ k ∈ s, 0 < f k)
    (h1 : ∀ k ∈ s, f k ≤ 1) : (∏ k in s, f k = 1) ↔ ∀ k ∈ s, f k = 1 := by
  refine ⟨fun h k hk => ?_, Finset.prod_eq_one⟩
  by_contra hne
  exact absurd h (ne_of_lt (prod_lt_one_of_mem h0 h1 hk (lt_of_le_of_ne (h1 k hk) hne)))

/-! ## The prefix-PHASE construction -/

/-- The complex prefix scan of §X₃₆: `C_i = Σ_{k ≤ i} (log m_k + i·θ_k)`.
    Its real part is `CEQ.V15.scan (log ∘ m)` and its imaginary part is the
    running phase; `C_re` and `C_im` say exactly that. -/
noncomputable def C (m θ : ℕ → ℝ) (i : ℕ) : ℂ :=
  ∑ k in range (i + 1), ((Real.log (m k) : ℂ) + (θ k : ℂ) * Complex.I)

/-- The phase-gated hop, `W_ij = exp(C_i − C_j)`, the complex form of
    `CEQ.V15.W`. -/
noncomputable def Wp (m θ : ℕ → ℝ) (i j : ℕ) : ℂ := Complex.exp (C m θ i - C m θ j)

lemma C_re (m θ : ℕ → ℝ) (i : ℕ) :
    (C m θ i).re = CEQ.V15.scan (fun k => Real.log (m k)) i := by
  simp [C, CEQ.V15.scan, Complex.re_sum, Complex.mul_I_re]

lemma C_im (m θ : ℕ → ℝ) (i : ℕ) : (C m θ i).im = ∑ k in range (i + 1), θ k := by
  simp [C, Complex.im_sum, Complex.mul_I_im]

lemma C_sub (m θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) :
    C m θ i - C m θ j
      = ((∑ k in Ico (j + 1) (i + 1), Real.log (m k) : ℝ) : ℂ)
        + ((∑ k in Ico (j + 1) (i + 1), θ k : ℝ) : ℂ) * Complex.I := by
  have h := Finset.sum_Ico_eq_sub
    (fun k => ((Real.log (m k) : ℂ) + (θ k : ℂ) * Complex.I)) (Nat.succ_le_succ hij)
  rw [C, C, ← h, Finset.sum_add_distrib, ← Finset.sum_mul, ← Complex.ofReal_sum,
    ← Complex.ofReal_sum]

/-- **The construction, stated once.** The phase carrier factors exactly: a
    real magnitude which is the path product of the `m`, times a unit-modulus
    factor which is the path sum of the `θ`. There is no cross term, which is
    why every magnitude statement below holds for every phase schedule and
    every phase statement below holds for every magnitude schedule. -/
theorem Wp_polar (m θ : ℕ → ℝ) (hm : ∀ k, 0 < m k) {i j : ℕ} (hij : j ≤ i) :
    Wp m θ i j = ((∏ k in Ico (j + 1) (i + 1), m k : ℝ) : ℂ)
        * Complex.exp (((∑ k in Ico (j + 1) (i + 1), θ k : ℝ) : ℂ) * Complex.I) := by
  have hexp : Real.exp (∑ k in Ico (j + 1) (i + 1), Real.log (m k))
      = ∏ k in Ico (j + 1) (i + 1), m k := by
    rw [Real.exp_sum]
    exact Finset.prod_congr rfl fun k _ => Real.exp_log (hm k)
  rw [Wp, C_sub m θ hij, Complex.exp_add, ← Complex.ofReal_exp, hexp]

/-- **The phase changes nothing about the magnitude.** For every `i` and `j`,
    with no positivity and no ordering hypothesis, the modulus of the complex
    carrier is `CEQ.V15.W` of the log-magnitudes — #2's real carrier, entry for
    entry. So X₃₆ does not weaken or restate the magnitude semantics of V15; it
    adds a factor of modulus `1` on top of them. -/
theorem phase_modulus_is_the_real_carrier (m θ : ℕ → ℝ) (i j : ℕ) :
    Complex.abs (Wp m θ i j) = CEQ.V15.W (fun k => Real.log (m k)) i j := by
  rw [Wp, Complex.abs_exp, CEQ.V15.W, Complex.sub_re, C_re, C_re]

/-- **The modulus of the path is the path product of the magnitudes.**
    `m > 0` is the only hypothesis; `θ` is universally quantified and
    unconstrained. -/
theorem prefix_phase_modulus (m θ : ℕ → ℝ) (hm : ∀ k, 0 < m k) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp m θ i j) = ∏ k in Ico (j + 1) (i + 1), m k := by
  rw [phase_modulus_is_the_real_carrier, CEQ.V15.prefix_logit_mask m hm hij]

/-- **Half one of the pair: `≤ 1` by construction.** No hypothesis on the
    phases at all — the statement quantifies over every `θ : ℕ → ℝ`, so there
    is no phase schedule, learned or adversarial, that makes the path product
    exceed `1`. Divergence is not bounded here, it is unavailable. -/
theorem phase_path_le_one (m : ℕ → ℝ) (h0 : ∀ k, 0 < m k) (h1 : ∀ k, m k ≤ 1)
    (θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) : Complex.abs (Wp m θ i j) ≤ 1 := by
  rw [prefix_phase_modulus m θ h0 hij]
  exact Finset.prod_le_one (fun k _ => (h0 k).le) fun k _ => h1 k

/-- **Half two of the pair: `= 1` exactly on the band.** An iff. The modulus is
    `1` if and only if every magnitude on that path is `1`; anything strictly
    inside the cap, anywhere on the path, makes it strictly less. -/
theorem phase_path_eq_one_iff_band (m : ℕ → ℝ) (h0 : ∀ k, 0 < m k) (h1 : ∀ k, m k ≤ 1)
    (θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp m θ i j) = 1 ↔ ∀ k ∈ Ico (j + 1) (i + 1), m k = 1 := by
  rw [prefix_phase_modulus m θ h0 hij]
  exact prod_eq_one_iff (fun k _ => h0 k) fun k _ => h1 k

/-! ## What #6 gives, what it does not give, and what the cap adds -/

/-- #6's bound, recovered inside the complex carrier: with
    `m_k = exp(−softplus w_k)` the modulus of every path is `≤ 1`. -/
theorem six_recovered (w θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp (fun k => Real.exp (-CEQ.V15.softplus (w k))) θ i j) ≤ 1 :=
  phase_path_le_one _ (fun _ => Real.exp_pos _)
    (fun k => (CEQ.V15.bounded_gates_stable (w k)).2) θ hij

/-- **#6 CANNOT deliver the delta's equality.** For `m_k = exp(−softplus w_k)`
    the modulus of every non-empty path is STRICTLY less than `1`, for every
    `w` and every `θ`. So "on the band `m ≡ 1` this becomes `= 1`" is not a
    statement about #6's gates: `m ≡ 1` is outside the image of
    `w ↦ exp(−softplus w)`. The hard cap is not a convenience in §X₃₆, it is
    what makes the band reachable. -/
theorem six_never_reaches_the_band (w θ : ℕ → ℝ) {i j : ℕ} (hij : j < i) :
    Complex.abs (Wp (fun k => Real.exp (-CEQ.V15.softplus (w k))) θ i j) < 1 := by
  rw [prefix_phase_modulus _ θ (fun k => Real.exp_pos _) hij.le]
  refine prod_lt_one_of_mem (fun k _ => Real.exp_pos _)
    (fun k _ => (CEQ.V15.bounded_gates_stable (w k)).2)
    (Finset.mem_Ico.mpr ⟨le_refl _, Nat.succ_lt_succ hij⟩) ?_
  exact softplus_gate_lt_one (w (j + 1))

/-- **The cap delivers it.** With the magnitudes capped and every pre-cap value
    at or above `1`, the modulus of the path is exactly `1` — for every phase
    schedule. This is the band case #16 is supposed to name. -/
theorem cap_band_attains (x θ : ℕ → ℝ) (hx : ∀ k, 1 ≤ x k) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp (fun k => cap (x k)) θ i j) = 1 := by
  rw [prefix_phase_modulus _ θ (fun k => by rw [cap_eq_one_of_one_le (hx k)]; norm_num) hij]
  exact Finset.prod_eq_one fun k _ => cap_eq_one_of_one_le (hx k)

/-- **And the divergence needs an open magnitude, not a phase.** A carrier with
    `m ≡ 285.07` reproduces R1's worst `â_max` as the modulus of a single-step
    path, whatever the phases are. Read against `phase_path_le_one`: the cap is
    the load-bearing hypothesis, the phase is not. -/
theorem divergence_needs_an_open_magnitude (θ : ℕ → ℝ) :
    Complex.abs (Wp (fun _ => (285.07 : ℝ)) θ 1 0) = 285.07 := by
  rw [prefix_phase_modulus _ θ (fun _ => by norm_num) (Nat.zero_le 1)]
  rw [Finset.prod_Ico_succ_top (le_refl 1)]
  simp

/-! ## §X₃₆'s `parity mask = Z₂ winding`, and where it is false -/

/-- BED-M's `±1` gates read as phases: `θ ∈ {0, π}`. -/
noncomputable def phaseOf (p : ZMod 2) : ℝ := if p = 0 then 0 else Real.pi

/-- The one-step half of the identity: the `{0, π}` phase gate IS the sign
    character of `ZMod 2`. -/
lemma phase_gate_is_sign (p : ZMod 2) :
    Complex.exp ((phaseOf p : ℂ) * Complex.I) = ((CEQ.V15.chi p : ℝ) : ℂ) := by
  rcases CEQ.V15.zmod_two_cases p with rfl | rfl
  · simp [phaseOf, CEQ.V15.chi]
  · have h : (1 : ZMod 2) ≠ 0 := by decide
    simp [phaseOf, CEQ.V15.chi, h, Complex.exp_pi_mul_I]

/-- **`parity_is_Z2_winding`.** The phase accumulated along the path from `j`
    to `i` by `{0, π}` gates, exponentiated, is exactly `CEQ.V15.parity_sign`'s
    character of the prefix-XOR of the sign bits. The two mechanisms of V15 —
    the `ZMod 2` prefix mask (#3) and the complex phase scan of X₃₆ — are the
    same map. This is the delta's `[RUN: True]`, proved. -/
theorem parity_is_Z2_winding (p : ℕ → ZMod 2) {i j : ℕ} (hij : j ≤ i) :
    Complex.exp (((∑ k in Ico (j + 1) (i + 1), phaseOf (p k) : ℝ) : ℂ) * Complex.I)
      = ((CEQ.V15.chi (CEQ.V15.pscan p i - CEQ.V15.pscan p j) : ℝ) : ℂ) := by
  rw [CEQ.V15.parity_sign p hij, Complex.ofReal_prod, Complex.ofReal_sum, Finset.sum_mul,
    Complex.exp_sum]
  exact Finset.prod_congr rfl fun k _ => phase_gate_is_sign (p k)

/-- **The delta's phrasing of it is FALSE, and here is the counterexample.**
    §X₃₆ says the prefix sum of phases "corresponds to" the prefix-XOR of the
    sign bits. As a statement about the phases themselves it fails: with
    `p ≡ 1`, the path `Ico 1 3` accumulates `2π`, while the prefix-XOR over the
    same path is the bit `0`, whose phase is `0`. The correspondence holds only
    after the quotient by `2πℤ` that `exp` performs — which is what
    `parity_is_Z2_winding` states and this refutes any stronger reading of. -/
theorem phase_sum_is_not_the_parity_phase :
    (∑ k in Ico 1 3, phaseOf ((fun _ => (1 : ZMod 2)) k)) = 2 * Real.pi ∧
      phaseOf (CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 2
          - CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 0) = 0 ∧
      (∑ k in Ico 1 3, phaseOf ((fun _ => (1 : ZMod 2)) k))
        ≠ phaseOf (CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 2
            - CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 0) := by
  have hone : (1 : ZMod 2) ≠ 0 := by decide
  have hbit : CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 2
      - CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 0 = 0 := by decide
  have hsum : (∑ k in Ico 1 3, phaseOf ((fun _ => (1 : ZMod 2)) k)) = 2 * Real.pi := by
    simp only [phaseOf, hone, if_false]
    rw [Finset.sum_const, Nat.card_Ico]
    norm_num
  have hrhs : phaseOf (CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 2
      - CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 0) = 0 := by
    rw [hbit]; simp [phaseOf]
  refine ⟨hsum, hrhs, ?_⟩
  rw [hsum, hrhs]
  simpa using Real.pi_ne_zero

/-- **And `Z₂` is not the `Z` winding §X₃₇(a) asks for.** The same all-ones
    parity sequence over `Ico 1 5` accumulates `2·(2π)` — winding number `2` —
    and its character is `1`, the same value the empty path gives. The parity
    mask is the winding reduced mod 2; it does not determine the integer, so
    the `[RUN: True]` identity of §X₃₆ does not discharge §X₃₇(a)'s
    per-instance integer certificate. -/
theorem Z2_forgets_the_winding :
    (∑ k in Ico 1 5, phaseOf ((fun _ => (1 : ZMod 2)) k)) = 2 * (2 * Real.pi) ∧
      Complex.exp (((∑ k in Ico 1 5, phaseOf ((fun _ => (1 : ZMod 2)) k) : ℝ) : ℂ)
        * Complex.I) = 1 := by
  have hone : (1 : ZMod 2) ≠ 0 := by decide
  have hsum : (∑ k in Ico 1 5, phaseOf ((fun _ => (1 : ZMod 2)) k)) = 2 * (2 * Real.pi) := by
    simp only [phaseOf, hone, if_false]
    rw [Finset.sum_const, Nat.card_Ico]
    norm_num
    ring
  refine ⟨hsum, ?_⟩
  rw [hsum]
  have hc : ((2 * (2 * Real.pi) : ℝ) : ℂ) * Complex.I
      = ((2 : ℤ) : ℂ) * (2 * (Real.pi : ℂ) * Complex.I) := by push_cast; ring
  rw [hc, Complex.exp_int_mul_two_pi_mul_I]

end CEQ.V15Phase

/-! ## Axiom check — every declaration in this file -/

#print axioms CEQ.V15Phase.cap
#print axioms CEQ.V15Phase.C
#print axioms CEQ.V15Phase.Wp
#print axioms CEQ.V15Phase.phaseOf
#print axioms CEQ.V15Phase.unit_phase_product
#print axioms CEQ.V15Phase.unit_phase_does_not_bound_the_gate
#print axioms CEQ.V15Phase.cap_nonneg
#print axioms CEQ.V15Phase.cap_le_one
#print axioms CEQ.V15Phase.cap_eq_one_of_one_le
#print axioms CEQ.V15Phase.cap_eq_zero_of_le_zero
#print axioms CEQ.V15Phase.softplus_gate_lt_one
#print axioms CEQ.V15Phase.lru_modulus_lt_one
#print axioms CEQ.V15Phase.prod_lt_one_of_mem
#print axioms CEQ.V15Phase.prod_eq_one_iff
#print axioms CEQ.V15Phase.C_re
#print axioms CEQ.V15Phase.C_im
#print axioms CEQ.V15Phase.C_sub
#print axioms CEQ.V15Phase.Wp_polar
#print axioms CEQ.V15Phase.phase_modulus_is_the_real_carrier
#print axioms CEQ.V15Phase.prefix_phase_modulus
#print axioms CEQ.V15Phase.phase_path_le_one
#print axioms CEQ.V15Phase.phase_path_eq_one_iff_band
#print axioms CEQ.V15Phase.six_recovered
#print axioms CEQ.V15Phase.six_never_reaches_the_band
#print axioms CEQ.V15Phase.cap_band_attains
#print axioms CEQ.V15Phase.divergence_needs_an_open_magnitude
#print axioms CEQ.V15Phase.phase_gate_is_sign
#print axioms CEQ.V15Phase.parity_is_Z2_winding
#print axioms CEQ.V15Phase.phase_sum_is_not_the_parity_phase
#print axioms CEQ.V15Phase.Z2_forgets_the_winding
