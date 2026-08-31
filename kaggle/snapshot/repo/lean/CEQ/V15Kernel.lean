/-
  CEQ.V15Kernel
  -------------
  Provenance: `CEQ_V15_CONTRACT.md` PART II items #12 `first_order_cannot_delay`
  and #13 `GL_weights_powerlaw`, and PART III's BED-K paragraph, which calls the
  delay bed "attention-native, scan-blind **by Lean #12**". Before this file that
  scan-blindness was EMPIRICAL: `V15_N4_BEDK.md` reports a first-order recurrence
  fitted to the delay bed reaching `R² = −0.000170`, a 30-seed sweep in
  `[−0.0027, 0.0028]`, and the same fitting code recovering a true AR(1) at
  `R² = 1.000000`. Good evidence, not a proof.

  | # | name                                             | what is proved here |
  |---|--------------------------------------------------|---------------------|
  |12 | `first_order_cannot_delay`                       | no first-order recurrence whose STATE IS ITS OUTPUT — `f` arbitrary, not assumed linear — satisfies `y_{i+d} = x_i` for every input, for any `d ≥ 1` |
  |12 | `affine_first_order_cannot_delay`                | the same for `y_i = α y_{i−1} + β x_i`, the model `fit_first_order_recurrence` fits |
  |12 | `first_order_delays_constant_input`              | the PER-SEQUENCE reading is FALSE: on a constant drive `α=0, β=1` delays exactly |
  |12 | `delay_realizable_at_dimension_d`                | the NAIVE reading is FALSE: a shift register is first-order and delays exactly, at state dimension `d + 1` |
  |12 | `delay_forces_state_injective`                   | the mechanism with no dimension and no linearity hypothesis: a `d`-delay forces the state after `d` inputs to determine those `d` inputs |
  |12 | `linear_first_order_cannot_delay_beyond_state_dim`| LTI state space `Fin k → ℝ`: delay `d > k` is impossible, any `A`, `B`, `s₀`, readout `C` |
  |12 | `first_order_cannot_powerlaw` / `..._strict`     | the power-law bed: the scalar recurrence's kernel is geometric, the GL kernel is not, at both index alignments |
  |13 | `GL_weights_ratio_recurrence`                    | `(−1)^k C(−α,k) = w_k` with `w_0 = 1`, `w_k = w_{k−1}(α+k−1)/k` — the closed form the contract writes EQUALS the recurrence the implementation runs |
  |13 | `GL_weights_alpha_zero` / `_alpha_one`           | the two binds: `α = 0` gives `[1,0,0,…]`, `α = 1` gives all ones |
  |13 | `GL_weights_pos` / `_strictAnti`                 | positive and strictly decreasing on `0 < α < 1` |

  **THE NAIVE READING OF #12 IS FALSE, AND THE FILE PROVES IT FALSE.**
  "A first-order recurrence cannot produce a pure delay" is not a theorem: a shift
  register is a first-order recurrence, and it delays exactly.
  `delay_realizable_at_dimension_d` exhibits one — `window`, with its update
  `shiftStep` depending on nothing but (previous state, current input) — and proves
  it satisfies `IsDelay d`. So the theorem CANNOT be "no first-order recurrence
  delays", and any statement of #12 that does not pin the state down is refuted by
  a construction in this same file.

  What is pinned down, and where each theorem sits:

  * `first_order_cannot_delay` — STATE DIMENSION ONE, and the state is the output.
    `f` is an arbitrary function `ℝ → ℝ → ℝ`. This is the reading the BED-K
    instrument measures: `fit_first_order_recurrence` regresses on the label's own
    TRUE previous value, so the fitted machine's state is its own output, scalar.
    The theorem is strictly stronger than the measurement, which only rules out
    the AFFINE `f`.
  * `delay_forces_state_injective` — ANY state type, ANY `f`, ANY readout `g`, no
    linearity: realizing the `d`-delay forces the map (first `d` inputs) ↦ (state
    at time `d−1`) to be INJECTIVE. This is the mechanism — a `d`-delay must carry
    `d` numbers through the state — with the dimension hypothesis removed.
  * `linear_first_order_cannot_delay_beyond_state_dim` — state space `Fin k → ℝ`,
    `f` affine (an LTI system). Delay `d > k` is impossible. Sharp: the shift
    register does `d` at `k = d + 1`, so the file brackets the true threshold.

  **WHAT #13 DOES NOT PROVE.** The contract's `w_k ~ k^{α−1}/Γ(α)` is NOT proved
  here and is not claimed. `GL_weights_strictAnti` proves monotone decay on
  `0 < α < 1`; the RATE — that the decay is a power law with that exponent and that
  constant — stays `[S]`. What the power-law half of #12 needs is not the rate but
  the failure of GEOMETRIC decay, and that is proved outright
  (`GL_not_geometric`, `GL_not_geometric_tail`), from three weights each.

  **REGIME.** `first_order_cannot_powerlaw` is stated on `α ∈ (0, 1)`, which
  contains BED-K's registered stationarity box `α = H − 1/2 ∈ (0, 1/2)` entirely.
  VORT (arXiv:2605.08966) proves a related non-representability result by an
  `L²`-energy divergence argument, but only for `α > 1/2` — the regime where
  `∑ w_k²` diverges, which is exactly the complement of BED-K's box and the same
  threshold as ARFIMA's `|d| < 1/2`. In BED-K's box that energy CONVERGES, so the
  published argument is unavailable there and the theorems here are not a
  restatement of it. See `V15_JUPITER3_KERNEL.md`.

  Set-theoretically the scalar case does NOT generalize to a hidden scalar state:
  a pathological injection `ℝ² ↪ ℝ` exists, so `delay_forces_state_injective` alone
  cannot refute a nonlinear hidden-scalar-state machine. That is why the scalar
  theorem is proved for output-equals-state and the dimension theorem is proved for
  linear maps, and why neither is stated in the other's generality.

  No `sorry`.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Dimension.StrongRankCondition
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring

namespace CEQ.V15

open BigOperators Finset

/-! ## First-order recurrences, and the delay specification -/

/-- A first-order recurrence over a state of ANY type `S`: the next state is a
    function of the previous state and the CURRENT input, and of nothing else.
    `F` is arbitrary — no linearity, continuity or boundedness is assumed. -/
def vecRec {S : Type*} (F : S → ℝ → S) (s₀ : S) (x : ℕ → ℝ) : ℕ → S
  | 0 => F s₀ (x 0)
  | i + 1 => F (vecRec F s₀ x i) (x (i + 1))

@[simp] lemma vecRec_zero {S : Type*} (F : S → ℝ → S) (s₀ : S) (x : ℕ → ℝ) :
    vecRec F s₀ x 0 = F s₀ (x 0) := rfl

@[simp] lemma vecRec_succ {S : Type*} (F : S → ℝ → S) (s₀ : S) (x : ℕ → ℝ) (i : ℕ) :
    vecRec F s₀ x (i + 1) = F (vecRec F s₀ x i) (x (i + 1)) := rfl

/-- The scalar case: the state IS the output, `y_i = f (y_{i-1}) (x_i)`. This is
    the machine `ceq/beds/bed_k.py`'s `fit_first_order_recurrence` fits — it
    regresses the label on the label's own true previous value. -/
abbrev firstOrder (f : ℝ → ℝ → ℝ) (y₀ : ℝ) (x : ℕ → ℝ) : ℕ → ℝ := vecRec f y₀ x

/-- The affine instance, `y_i = α·y_{i−1} + β·x_i`: the exact model the instrument
    fits by least squares. -/
def affineRec (α β y₀ : ℝ) (x : ℕ → ℝ) : ℕ → ℝ :=
  firstOrder (fun y u => α * y + β * u) y₀ x

/-- `y` is the `d`-step delay of `x`. Written `y (i + d) = x i` rather than
    `y i = x (i − d)` so that no `ℕ`-subtraction truncation enters the statement
    and the `i < d` boundary is not quietly excused. -/
def IsDelay (d : ℕ) (y x : ℕ → ℝ) : Prop := ∀ i, y (i + d) = x i

/-- The unit impulse at position `p`. -/
def dirac (p : ℕ) : ℕ → ℝ := fun i => if i = p then 1 else 0

/-! ## #12, the delay bed -/

/-- **#12 `first_order_cannot_delay`.**

    HYPOTHESES, stated rather than assumed away: the state has dimension ONE and
    the state IS the output (`y_i = f y_{i−1} x_i`); the delay is `d ≥ 1`; the
    identity is required for EVERY input sequence. `f` is arbitrary — the theorem
    covers every nonlinear scalar recurrence, not just the affine one the
    instrument fits, and every initial condition `y₀`.

    The refutation is two explicit witnesses and is checkable by hand. Realizing
    the delay forces, at every input, `y_{d+1} = f (y_d) (x_{d+1})` with
    `y_d = x_0` and `y_{d+1} = x_1`, i.e. `x_1 = f (x_0) (x_{d+1})`. The all-zero
    input then reads `0 = f 0 0`; the impulse at position `1` reads `1 = f 0 0`,
    because `d ≥ 1` puts `d + 1` off the impulse. -/
theorem first_order_cannot_delay (f : ℝ → ℝ → ℝ) (y₀ : ℝ) {d : ℕ} (hd : 1 ≤ d) :
    ¬ ∀ x : ℕ → ℝ, IsDelay d (firstOrder f y₀ x) x := by
  intro h
  have key : ∀ x : ℕ → ℝ, x 1 = f (x 0) (x (d + 1)) := by
    intro x
    have h0 : firstOrder f y₀ x d = x 0 := by simpa using h x 0
    have h1 : firstOrder f y₀ x (d + 1) = x 1 := by
      have hx := h x 1
      rwa [show 1 + d = d + 1 from Nat.add_comm 1 d] at hx
    have hstep : firstOrder f y₀ x (d + 1) = f (firstOrder f y₀ x d) (x (d + 1)) := rfl
    rw [← h1, hstep, h0]
  have hzero : (0 : ℝ) = f 0 0 := by simpa using key (fun _ => 0)
  have hone : (1 : ℝ) = f 0 0 := by
    have hk := key (dirac 1)
    have e1 : dirac 1 1 = 1 := by simp [dirac]
    have e0 : dirac 1 0 = 0 := by simp [dirac]
    have ed : dirac 1 (d + 1) = 0 := by
      have hne : d + 1 ≠ 1 := by omega
      simp [dirac, hne]
    rw [e1, e0, ed] at hk
    exact hk
  exact absurd (hzero.trans hone.symm) (by norm_num)

/-- The instrument's own model, as a corollary: no `(α, β, y₀)` reproduces the
    delay bed's label for every drive. This is what `R² = −0.000170` was
    measuring, now at exact arithmetic and for every seed at once. -/
theorem affine_first_order_cannot_delay (α β y₀ : ℝ) {d : ℕ} (hd : 1 ≤ d) :
    ¬ ∀ x : ℕ → ℝ, IsDelay d (affineRec α β y₀ x) x :=
  first_order_cannot_delay _ y₀ hd

/-- **The per-sequence reading of #12 is FALSE, and the quantifier over inputs is
    therefore load-bearing.** On a CONSTANT drive the delayed label equals the
    undelayed one, and `α = 0, β = 1` reproduces it exactly, at every `d`. So
    "a first-order recurrence cannot fit this delayed label" is false as a
    statement about one fixed input sequence; it is only true as a statement about
    the delay MAP. `fit_first_order_recurrence`'s near-zero `R²` is evidence about
    one realization of an iid drive, which is why it needed a theorem. -/
theorem first_order_delays_constant_input (c y₀ : ℝ) (d : ℕ) :
    IsDelay d (affineRec 0 1 y₀ (fun _ => c)) (fun _ => c) := by
  have hall : ∀ n, affineRec 0 1 y₀ (fun _ => c) n = c := by
    intro n
    induction n with
    | zero => simp [affineRec]
    | succ n _ => simp [affineRec]
  intro i
  simp [hall]

/-! ## Why the statement is about the STATE DIMENSION: the shift register -/

/-- The `d`-window of the input: coordinate `j` holds `x (i − j)`, guarded to `0`
    before the input starts so that the recurrence has a state-independent base. -/
def window (d : ℕ) (x : ℕ → ℝ) (i : ℕ) : Fin (d + 1) → ℝ :=
  fun j => if (j : ℕ) ≤ i then x (i - (j : ℕ)) else 0

/-- The shift-register update: push the new input in at coordinate `0`, shift every
    other coordinate up, drop the oldest. A function of (previous state, current
    input) and of nothing else — first order, by inspection. -/
def shiftStep (d : ℕ) (s : Fin (d + 1) → ℝ) (u : ℝ) : Fin (d + 1) → ℝ :=
  Fin.cons u (fun k : Fin d => s k.castSucc)

lemma window_zero (d : ℕ) (x : ℕ → ℝ) : window d x 0 = shiftStep d 0 (x 0) := by
  funext j
  cases j using Fin.cases with
  | zero => simp [window, shiftStep]
  | succ k => simp [window, shiftStep]

lemma window_step (d : ℕ) (x : ℕ → ℝ) (i : ℕ) :
    window d x (i + 1) = shiftStep d (window d x i) (x (i + 1)) := by
  funext j
  cases j using Fin.cases with
  | zero => simp [window, shiftStep]
  | succ k =>
    simp only [window, shiftStep, Fin.cons_succ, Fin.val_succ, Fin.coe_castSucc,
      Nat.succ_sub_succ_eq_sub]
    by_cases hk : (k : ℕ) ≤ i
    · simp [hk, Nat.succ_le_succ hk]
    · simp [hk, fun h => hk (Nat.le_of_succ_le_succ h)]

/-- The window is literally the output of a first-order recurrence over the
    `(d+1)`-dimensional state, started from `0`. -/
lemma window_eq_vecRec (d : ℕ) (x : ℕ → ℝ) (i : ℕ) :
    window d x i = vecRec (shiftStep d) 0 x i := by
  induction i with
  | zero => simpa using window_zero d x
  | succ i ih => rw [window_step, ih, vecRec_succ]

lemma window_isDelay (d : ℕ) (x : ℕ → ℝ) (i : ℕ) :
    window d x (i + d) (Fin.last d) = x i := by
  simp [window, Fin.val_last]

/-- **The naive reading of #12, refuted by construction.** A first-order recurrence
    over a state of dimension `d + 1` DOES produce a pure `d`-delay: the shift
    register, with the last coordinate read out. Any statement of #12 that does not
    fix the state dimension is therefore false, and this is the reason
    `first_order_cannot_delay` is stated at dimension one. -/
theorem delay_realizable_at_dimension_d (d : ℕ) (x : ℕ → ℝ) :
    IsDelay d (fun i => vecRec (shiftStep d) 0 x i (Fin.last d)) x := by
  intro i
  show vecRec (shiftStep d) 0 x (i + d) (Fin.last d) = x i
  rw [← window_eq_vecRec]
  exact window_isDelay d x i

/-! ## The mechanism at any state dimension: a delay forces an injective state -/

/-- The zero-padded prefix of length `d`. -/
def pad {d : ℕ} (u : Fin d → ℝ) : ℕ → ℝ := fun i => if h : i < d then u ⟨i, h⟩ else 0

/-- **The mechanism, with the dimension hypothesis removed.** For ANY state type,
    ANY first-order update `F`, ANY initial state and ANY readout `g` — no
    linearity, no continuity — realizing the `(e+1)`-delay forces the state after
    the first `e+1` inputs to determine those inputs. A delay of `d` must carry `d`
    numbers through the state; that is the content of #12, and everything
    dimension-specific below is this lemma plus a counting argument. -/
theorem delay_forces_state_injective {S : Type*} (F : S → ℝ → S) (s₀ : S) (g : S → ℝ)
    (e : ℕ) (h : ∀ x : ℕ → ℝ, IsDelay (e + 1) (fun i => g (vecRec F s₀ x i)) x) :
    Function.Injective (fun u : Fin (e + 1) → ℝ => vecRec F s₀ (pad u) e) := by
  intro u v huv
  simp only at huv
  -- the states stay equal forever: after time `e` both padded inputs are `0`
  have hstep : ∀ m, vecRec F s₀ (pad u) (e + m) = vecRec F s₀ (pad v) (e + m) := by
    intro m
    induction m with
    | zero => simpa using huv
    | succ m ih =>
      have hpu : pad u (e + m + 1) = 0 := by
        have : ¬ (e + m + 1 < e + 1) := by omega
        simp [pad, this]
      have hpv : pad v (e + m + 1) = 0 := by
        have : ¬ (e + m + 1 < e + 1) := by omega
        simp [pad, this]
      have he : e + (m + 1) = (e + m) + 1 := by omega
      rw [he, vecRec_succ, vecRec_succ, ih, hpu, hpv]
  funext j
  have hj : (j : ℕ) + (e + 1) = e + ((j : ℕ) + 1) := by omega
  have hu := h (pad u) (j : ℕ)
  have hv := h (pad v) (j : ℕ)
  simp only at hu hv
  rw [hj] at hu hv
  have hpu : pad u (j : ℕ) = u j := by simp [pad, j.isLt]
  have hpv : pad v (j : ℕ) = v j := by simp [pad, j.isLt]
  rw [hpu] at hu
  rw [hpv] at hv
  rw [← hu, ← hv, hstep ((j : ℕ) + 1)]

/-! ## The linear case: delay `d` needs state dimension `≥ d` -/

variable {k : ℕ}

/-- One step of a linear time-invariant first-order system on `Fin k → ℝ`. -/
def linStep (A : (Fin k → ℝ) →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ)
    (s : Fin k → ℝ) (u : ℝ) : Fin k → ℝ := A s + u • B

lemma linRec_add (A : (Fin k → ℝ) →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ) (x y : ℕ → ℝ) :
    ∀ i, vecRec (linStep A B) 0 (x + y) i
      = vecRec (linStep A B) 0 x i + vecRec (linStep A B) 0 y i := by
  intro i
  induction i with
  | zero => simp [linStep, add_smul]
  | succ i ih =>
    simp only [vecRec_succ, linStep, ih, Pi.add_apply, map_add, add_smul]
    abel

lemma linRec_smul (A : (Fin k → ℝ) →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ) (c : ℝ) (x : ℕ → ℝ) :
    ∀ i, vecRec (linStep A B) 0 (c • x) i = c • vecRec (linStep A B) 0 x i := by
  intro i
  induction i with
  | zero => simp [linStep, mul_smul]
  | succ i ih =>
    simp only [vecRec_succ, linStep, ih, Pi.smul_apply, smul_eq_mul, map_smul, mul_smul, smul_add]

lemma pad_add {d : ℕ} (u v : Fin d → ℝ) : pad (u + v) = pad u + pad v := by
  funext i
  by_cases h : i < d <;> simp [pad, h]

lemma pad_smul {d : ℕ} (c : ℝ) (u : Fin d → ℝ) : pad (c • u) = c • pad u := by
  funext i
  by_cases h : i < d <;> simp [pad, h]

/-- The linear map (last `e+1` inputs) ↦ (state at time `e`), zero initial state. -/
def prefixMap (e : ℕ) (A : (Fin k → ℝ) →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ) :
    (Fin (e + 1) → ℝ) →ₗ[ℝ] (Fin k → ℝ) where
  toFun u := vecRec (linStep A B) 0 (pad u) e
  map_add' u v := by
    show vecRec (linStep A B) 0 (pad (u + v)) e
      = vecRec (linStep A B) 0 (pad u) e + vecRec (linStep A B) 0 (pad v) e
    rw [pad_add, linRec_add]
  map_smul' c u := by
    show vecRec (linStep A B) 0 (pad (c • u)) e = c • vecRec (linStep A B) 0 (pad u) e
    rw [pad_smul, linRec_smul]

/-- Shifting the initial state shifts both trajectories the same way, so it cannot
    create or destroy injectivity of the prefix map. -/
lemma linRec_sub (A : (Fin k → ℝ) →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ) (s₀ : Fin k → ℝ)
    (x y : ℕ → ℝ) : ∀ i,
    vecRec (linStep A B) s₀ x i - vecRec (linStep A B) s₀ y i
      = vecRec (linStep A B) 0 x i - vecRec (linStep A B) 0 y i := by
  intro i
  induction i with
  | zero => simp [linStep]
  | succ i ih =>
    have hA : A (vecRec (linStep A B) s₀ x i) - A (vecRec (linStep A B) s₀ y i)
        = A (vecRec (linStep A B) 0 x i) - A (vecRec (linStep A B) 0 y i) := by
      rw [← map_sub, ← map_sub, ih]
    simp only [vecRec_succ, linStep]
    calc A (vecRec (linStep A B) s₀ x i) + x (i + 1) • B
            - (A (vecRec (linStep A B) s₀ y i) + y (i + 1) • B)
        = (A (vecRec (linStep A B) s₀ x i) - A (vecRec (linStep A B) s₀ y i))
            + (x (i + 1) • B - y (i + 1) • B) := by abel
      _ = (A (vecRec (linStep A B) 0 x i) - A (vecRec (linStep A B) 0 y i))
            + (x (i + 1) • B - y (i + 1) • B) := by rw [hA]
      _ = A (vecRec (linStep A B) 0 x i) + x (i + 1) • B
            - (A (vecRec (linStep A B) 0 y i) + y (i + 1) • B) := by abel

/-- **#12 at general state dimension, linear case.** An LTI first-order system with
    state space `Fin k → ℝ` — any transition `A`, any input map `B`, any initial
    state `s₀`, any readout `C` — cannot realize a delay of `d = e + 1 > k`.

    Sharp, in this same file: `delay_realizable_at_dimension_d` realizes delay `d`
    at state dimension `d + 1`, so the impossible/possible boundary is bracketed
    between `k < d` and `k = d + 1`. -/
theorem linear_first_order_cannot_delay_beyond_state_dim (e : ℕ)
    (A : (Fin k → ℝ) →ₗ[ℝ] (Fin k → ℝ)) (B : Fin k → ℝ) (s₀ : Fin k → ℝ)
    (C : (Fin k → ℝ) → ℝ) (hk : k < e + 1)
    (h : ∀ x : ℕ → ℝ, IsDelay (e + 1) (fun i => C (vecRec (linStep A B) s₀ x i)) x) :
    False := by
  have hinj := delay_forces_state_injective (linStep A B) s₀ C e h
  have hinj0 : Function.Injective (prefixMap (k := k) e A B) := by
    intro u v huv
    apply hinj
    have hs : vecRec (linStep A B) s₀ (pad u) e - vecRec (linStep A B) s₀ (pad v) e = 0 := by
      rw [linRec_sub]
      exact sub_eq_zero_of_eq huv
    simpa using sub_eq_zero.mp hs
  have hle := LinearMap.finrank_le_finrank_of_injective hinj0
  rw [FiniteDimensional.finrank_fin_fun, FiniteDimensional.finrank_fin_fun] at hle
  omega

/-! ## #13 — the Grünwald-Letnikov weights -/

/-- The weights as the implementation computes them (`ceq/beds/bed_k.py`,
    `gl_weights`): `w_0 = 1`, `w_k = w_{k−1}·(α + k − 1)/k`. -/
noncomputable def glw (α : ℝ) : ℕ → ℝ
  | 0 => 1
  | k + 1 => glw α k * (α + (k : ℝ)) / ((k : ℝ) + 1)

@[simp] lemma glw_zero (α : ℝ) : glw α 0 = 1 := rfl

lemma glw_succ (α : ℝ) (k : ℕ) :
    glw α (k + 1) = glw α k * (α + (k : ℝ)) / ((k : ℝ) + 1) := rfl

/-- The generalized binomial coefficient `C(a, k) = (∏_{j<k}(a − j)) / k!`. -/
noncomputable def gbinom (a : ℝ) (k : ℕ) : ℝ :=
  (∏ j in range k, (a - (j : ℝ))) / (k.factorial : ℝ)

/-- The contract's closed form, `w_k = (−1)^k · C(−α, k)`. -/
noncomputable def glwClosed (α : ℝ) (k : ℕ) : ℝ := (-1) ^ k * gbinom (-α) k

/-- **#13 `GL_weights_ratio_recurrence`.**

    The closed form the contract writes and the ratio recurrence the implementation
    runs are the same sequence. This is the content: `bed_k.py` stopped calling
    `scipy.special.binom(-alpha, k)` after it was found to return NaN at `α = 1`
    and switched to the recurrence, so the object the corpus is built from is the
    recurrence, while the object the contract names is the binomial. Without this
    theorem the two are only conjecturally the same function. -/
theorem GL_weights_ratio_recurrence (α : ℝ) (k : ℕ) : glwClosed α k = glw α k := by
  induction k with
  | zero => simp [glwClosed, gbinom]
  | succ k ih =>
    have _hfac : ((k.factorial : ℝ)) ≠ 0 := Nat.cast_ne_zero.mpr k.factorial_ne_zero
    have _hk1 : ((k : ℝ) + 1) ≠ 0 := by positivity
    rw [glw_succ, ← ih]
    simp only [glwClosed, gbinom, Finset.prod_range_succ, Nat.factorial_succ, pow_succ,
      Nat.cast_mul, Nat.cast_succ]
    field_simp
    ring

/-- **Bind 1, `α = 0` gives the identity kernel `[1, 0, 0, …]`.** -/
theorem GL_weights_alpha_zero (k : ℕ) : glw 0 (k + 1) = 0 := by
  induction k with
  | zero => norm_num [glw_succ]
  | succ k ih => rw [glw_succ, ih]; ring

/-- **Bind 2, `α = 1` gives the cumulative sum, all weights `1`.** This is exactly
    the point where `scipy.special.binom(-1, k)` returns NaN; the value proved here
    is the one the ratio recurrence returns. -/
theorem GL_weights_alpha_one (k : ℕ) : glw 1 k = 1 := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hk1 : ((k : ℝ) + 1) ≠ 0 := by positivity
    rw [glw_succ, ih, one_mul, add_comm (1 : ℝ) (k : ℝ), div_self hk1]

/-- Positivity on the whole open ray `α > 0`, which contains BED-K's registered box
    `α = H − 1/2 ∈ (0, 1/2)`. -/
theorem GL_weights_pos (α : ℝ) (hα : 0 < α) (k : ℕ) : 0 < glw α k := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [glw_succ]
    have h1 : (0 : ℝ) < α + (k : ℝ) := by positivity
    have h2 : (0 : ℝ) < (k : ℝ) + 1 := by positivity
    exact div_pos (mul_pos ih h1) h2

/-- The DECAY half of #13 that is in reach: for `0 < α < 1` the weights are
    strictly decreasing. The RATE `w_k ~ k^{α−1}/Γ(α)` is NOT proved here; see the
    file header and `V15_JUPITER3_KERNEL.md`. -/
theorem GL_weights_strictAnti (α : ℝ) (h0 : 0 < α) (h1 : α < 1) (k : ℕ) :
    glw α (k + 1) < glw α k := by
  have hpos := GL_weights_pos α h0 k
  have hk1 : (0 : ℝ) < (k : ℝ) + 1 := by positivity
  rw [glw_succ, div_lt_iff hk1]
  nlinarith [hpos, hk1]

/-! ## #12, the power-law bed -/

/-- A causal convolution kernel applied to a drive: `z_i = ∑_{j ≤ i} w_{i−j} x_j`.
    BED-K's power-law bed is `z = K b` with `K i j = w_{i−j}`. -/
def kconv (w x : ℕ → ℝ) (i : ℕ) : ℝ := ∑ j in range (i + 1), w (i - j) * x j

lemma kconv_dirac_zero (w : ℕ → ℝ) (i : ℕ) : kconv w (dirac 0) i = w i := by
  rw [kconv, Finset.sum_eq_single 0]
  · simp [dirac]
  · intro j _ hj
    simp [dirac, hj]
  · intro hmem
    exact absurd (Finset.mem_range.mpr (Nat.succ_pos i)) hmem

/-- The impulse response of the scalar affine recurrence is GEOMETRIC: `β·γ^k`.
    This is the whole reason a scan cannot carry a power law — the shape of its
    kernel is fixed by two numbers. -/
lemma affineRec_impulse (γ β y₀ : ℝ) (hy : γ * y₀ = 0) (k : ℕ) :
    affineRec γ β y₀ (dirac 0) k = β * γ ^ k := by
  induction k with
  | zero =>
    show γ * y₀ + β * dirac 0 0 = β * γ ^ 0
    simp [dirac, hy]
  | succ k ih =>
    have hstep : affineRec γ β y₀ (dirac 0) (k + 1)
        = γ * affineRec γ β y₀ (dirac 0) k + β * dirac 0 (k + 1) := rfl
    have hd : dirac 0 (k + 1) = 0 := by simp [dirac]
    rw [hstep, ih, hd]
    simp only [Nat.succ_eq_add_one, pow_succ]
    ring

/-- **The GL weights are not geometric**, for every `α` strictly between `0` and
    `1` — which contains BED-K's registered box `α ∈ (0, 1/2)` entirely. Three
    weights settle it: `w_0 = 1` forces `β = 1`, `w_1 = α` forces `γ = α`, and then
    `w_2 = α(α+1)/2` must equal `α²`, i.e. `α = 1`. -/
theorem GL_not_geometric (α : ℝ) (h0 : 0 < α) (h1 : α < 1) :
    ¬ ∃ β γ : ℝ, ∀ k, glw α k = β * γ ^ k := by
  rintro ⟨β, γ, h⟩
  have e0 : (1 : ℝ) = β := by simpa using h 0
  have e1 : α = β * γ := by
    have := h 1
    rw [glw_succ] at this
    simpa using this
  have e2 : α * (α + 1) / 2 = β * γ ^ 2 := by
    have := h 2
    rw [show (2 : ℕ) = 1 + 1 from rfl, glw_succ, glw_succ] at this
    norm_num at this
    linarith [this]
  rw [← e0] at e1 e2
  rw [one_mul] at e1
  rw [one_mul, ← e1] at e2
  nlinarith [e2, h0, h1]

/-- **#12 for the power-law bed.**

    No scalar affine recurrence — any `α`-gate `γ`, any input gain `β`, any initial
    condition — reproduces the Grünwald-Letnikov power-law kernel for every drive,
    at any `α` strictly inside `(0, 1)`. The proof reads the machine's impulse
    response off the all-zero and unit-impulse drives and compares it to the
    weights: the recurrence's kernel is `β γ^k`, the bed's is `w_k`, and
    `GL_not_geometric` says those are different sequences.

    Note which regime this covers. VORT (arXiv:2605.08966) proves a related
    non-representability result for `α > 1/2` by an `L²`-energy divergence
    argument; BED-K's registered box is `α ∈ (0, 1/2)`, where that energy CONVERGES
    and the argument is unavailable. This theorem is stated on `(0, 1)` and so
    covers the stationary box the bed actually runs in. -/
theorem first_order_cannot_powerlaw (α : ℝ) (h0 : 0 < α) (h1 : α < 1) (γ β y₀ : ℝ) :
    ¬ ∀ (x : ℕ → ℝ) (i : ℕ), affineRec γ β y₀ x i = kconv (glw α) x i := by
  intro h
  have hy : γ * y₀ = 0 := by
    have hz := h (fun _ => 0) 0
    have hl : affineRec γ β y₀ (fun _ => 0) 0 = γ * y₀ + β * 0 := rfl
    have hr : kconv (glw α) (fun _ => (0 : ℝ)) 0 = 0 := by simp [kconv]
    rw [hl, hr, mul_zero, add_zero] at hz
    exact hz
  refine GL_not_geometric α h0 h1 ⟨β, γ, ?_⟩
  intro k
  rw [← kconv_dirac_zero (glw α) k, ← h (dirac 0) k, affineRec_impulse γ β y₀ hy k]

/-! ### The strictly causal alignment, which is the one `bed_k.py` builds -/

/-- `_powerlaw_kernel_matrix` sets `K[i, i−k] = w_k` for `k = 1..i` — no self term,
    strictly `j < i`. This is that kernel. -/
def kconvStrict (w x : ℕ → ℝ) (i : ℕ) : ℝ := ∑ j in range i, w (i - j) * x j

/-- The drive delayed one step. -/
def lag (x : ℕ → ℝ) : ℕ → ℝ := fun i => if i = 0 then 0 else x (i - 1)

/-- The first-order recurrence MATCHED to a strictly causal kernel: it reads the
    previous input, `y_i = γ·y_{i−1} + β·x_{i−1}`, so its own kernel is also
    strictly causal. Comparing `affineRec` (which reads `x_i`) against
    `kconvStrict` would win the theorem on an index convention instead of on the
    shape of the kernel, which is not what #12 is supposed to say. -/
def affineRecLag (γ β y₀ : ℝ) (x : ℕ → ℝ) : ℕ → ℝ := affineRec γ β y₀ (lag x)

lemma lag_dirac_zero : lag (dirac 0) = dirac 1 := by
  funext i
  rcases Nat.eq_zero_or_pos i with rfl | hi
  · simp [lag, dirac]
  · have h0 : i ≠ 0 := Nat.pos_iff_ne_zero.mp hi
    have hiff : (i - 1 = 0) ↔ (i = 1) := by omega
    simp [lag, dirac, h0, hiff]

lemma kconvStrict_dirac_zero (w : ℕ → ℝ) (i : ℕ) :
    kconvStrict w (dirac 0) (i + 1) = w (i + 1) := by
  rw [kconvStrict, Finset.sum_eq_single 0]
  · simp [dirac]
  · intro j _ hj
    simp [dirac, hj]
  · intro hmem
    exact absurd (Finset.mem_range.mpr (Nat.succ_pos i)) hmem

/-- The lagged recurrence's kernel is `β·γ^{k−1}` at lag `k ≥ 1` — geometric, with
    the same one-step shift as the target. -/
lemma affineRecLag_impulse (γ β y₀ : ℝ) (hy : γ * y₀ = 0) (k : ℕ) :
    affineRecLag γ β y₀ (dirac 0) (k + 1) = β * γ ^ k := by
  have hzero : affineRecLag γ β y₀ (dirac 0) 0 = 0 := by
    show γ * y₀ + β * lag (dirac 0) 0 = 0
    simp [lag, hy]
  induction k with
  | zero =>
    have hstep : affineRecLag γ β y₀ (dirac 0) 1
        = γ * affineRecLag γ β y₀ (dirac 0) 0 + β * lag (dirac 0) 1 := rfl
    rw [hstep, hzero, lag_dirac_zero]
    simp [dirac]
  | succ k ih =>
    have hstep : affineRecLag γ β y₀ (dirac 0) (k + 1 + 1)
        = γ * affineRecLag γ β y₀ (dirac 0) (k + 1) + β * lag (dirac 0) (k + 1 + 1) := rfl
    have hd : lag (dirac 0) (k + 1 + 1) = 0 := by
      rw [lag_dirac_zero]; simp [dirac]
    rw [hstep, ih, hd]
    simp only [Nat.succ_eq_add_one, pow_succ]
    ring

/-- **The GL weights are not geometric from lag 1 either.** The aligned refutation
    used `w_0, w_1, w_2`; this one uses `w_1, w_2, w_3`, so no alignment convention
    escapes it. `w_1 = α`, `w_2 = α(α+1)/2`, `w_3 = α(α+1)(α+2)/6`; geometry forces
    `(α+1)/4 = (α+2)/6`, i.e. `α = 1`. -/
theorem GL_not_geometric_tail (α : ℝ) (h0 : 0 < α) (h1 : α < 1) :
    ¬ ∃ β γ : ℝ, ∀ k, glw α (k + 1) = β * γ ^ k := by
  rintro ⟨β, γ, h⟩
  have hα : α ≠ 0 := ne_of_gt h0
  have e1 : α = β := by
    have hk := h 0
    rw [glw_succ] at hk
    simpa using hk
  have e2 : α * (α + 1) / 2 = β * γ := by
    have hk := h 1
    rw [show (1 : ℕ) + 1 = 1 + 1 from rfl, glw_succ, glw_succ] at hk
    norm_num at hk
    linarith [hk]
  have e3 : α * (α + 1) * (α + 2) / 6 = β * γ ^ 2 := by
    have hk := h 2
    rw [show (2 : ℕ) + 1 = 1 + 1 + 1 from rfl, glw_succ, glw_succ, glw_succ] at hk
    norm_num at hk
    linarith [hk]
  rw [← e1] at e2 e3
  have h2' : α * γ = α * ((α + 1) / 2) := by rw [← e2]; ring
  have hg : γ = (α + 1) / 2 := mul_left_cancel₀ hα h2'
  rw [hg] at e3
  have key : α * (α + 1) * (1 - α) = 0 := by linear_combination 12 * e3
  have hpos : 0 < α * (α + 1) * (1 - α) :=
    mul_pos (mul_pos h0 (by linarith)) (by linarith)
  linarith [key, hpos]

/-- **#12 for the power-law bed, at the exact kernel `bed_k.py` builds.**

    No first-order affine recurrence reading the previous input reproduces the
    strictly causal Grünwald-Letnikov kernel for every drive, at any `α ∈ (0,1)`. -/
theorem first_order_cannot_powerlaw_strict (α : ℝ) (h0 : 0 < α) (h1 : α < 1) (γ β y₀ : ℝ) :
    ¬ ∀ (x : ℕ → ℝ) (i : ℕ), affineRecLag γ β y₀ x i = kconvStrict (glw α) x i := by
  intro h
  have hy : γ * y₀ = 0 := by
    have hz := h (fun _ => 0) 0
    have hl : affineRecLag γ β y₀ (fun _ => (0 : ℝ)) 0 = γ * y₀ + β * lag (fun _ => (0:ℝ)) 0 :=
      rfl
    have hr : kconvStrict (glw α) (fun _ => (0 : ℝ)) 0 = 0 := by simp [kconvStrict]
    rw [hl, hr] at hz
    simpa [lag] using hz
  refine GL_not_geometric_tail α h0 h1 ⟨β, γ, ?_⟩
  intro k
  rw [← kconvStrict_dirac_zero (glw α) k, ← h (dirac 0) (k + 1),
    affineRecLag_impulse γ β y₀ hy k]

end CEQ.V15

