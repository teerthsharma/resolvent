# V15 X₃₆ — LEAN #16, AND THE TWO STATEMENTS AROUND IT THAT CARRY THE CLAIM

Node JUPITER-8, CEQ v15.3 round. Spec: `CEQ_V15_3_DELTA.md` §X₃₆ (THE GATE IS A
WAVE) and §LEAN #16.

Code: `lean/CEQ/V15Phase.lean`, new file, 30 declarations, no `sorry`.
`lean/CEQ.lean` is **not edited** — the coordinator holds the import list and
there are now five V15 modules; the file builds standalone against the already
compiled `CEQ.V15` olean.

Provenance for everything below: `lake env lean CEQ/V15Phase.lean` run in
`lean/` at working-tree HEAD `91639c3`, Lean 4.7.0
(`x86_64-w64-windows-gnu`, commit `6fce8f7d5cd1`), mathlib4 at
`a45ae63747140c1b2cbad9d46f518015c047047a` (`v4.7.0` pin from `lakefile.lean`),
Windows-11-10.0.26200, single machine. Wall clock 11.75 s on a warm cache.
Nothing was trained and nothing was measured for this file; every number in it
is either a proof or a quotation of a number R1 already measured.

---

## VERDICT — four calls

1. **#16 is proved, and it is as trivial as the delta says.** Three lines.
   `unit_phase_product` holds with no hypotheses at all, which is exactly why it
   is refused as the content of the node (§1).
2. **The pair the delta actually needs is proved, and it is stronger than the
   delta's phrasing.** `phase_path_le_one` bounds the modulus of the path
   product **for every phase schedule** — the `θ` are universally quantified
   inside the statement — and `phase_path_eq_one_iff_band` is an **iff**, not
   the implication the delta writes (§3).
3. **The delta's #6 → band step does NOT hold, and the file proves why.**
   `six_never_reaches_the_band`: under #6's own parametrization
   `m = exp(−softplus w)`, every non-empty path has modulus **strictly** below
   1, for every `w` and every `θ`. `m ≡ 1` is not in the image of
   `w ↦ exp(−softplus w)`, so the equality is not a corollary of #6 — it is a
   corollary of the hard cap, which is a different function (§4). The same
   proof retires LRU's open magnitude, which is the delta's own novelty claim,
   now at the strength of a strict inequality.
4. **`parity mask = Z₂ winding` is TRUE in one reading and FALSE in the one the
   delta writes.** The agreement holds after `exp` and only after `exp`;
   the phase-sum reading has a two-line counterexample, and the `Z₂` character
   does not determine the integer winding §X₃₇(a) asks for (§5).

---

## 1. #16 as the delta writes it, and the refusal

### The statement

```lean
theorem unit_phase_product (θ : ℕ → ℝ) (s : Finset ℕ) :
    Complex.abs (∏ k in s, Complex.exp ((θ k : ℂ) * Complex.I)) = 1
```

**Hypotheses: none.** `θ : ℕ → ℝ` arbitrary, `s : Finset ℕ` arbitrary,
including empty. Status: **PROVED**, two tactic lines
(`Complex.abs_prod`, then `Complex.abs_exp_ofReal_mul_I` pointwise).

Stated over `ℂ` rather than over mathlib's `circle` subgroup deliberately. The
gates of §X₃₆ are `m·e^{iθ}`; only the second factor lives on the circle, so a
statement inside the subgroup would be cleaner and would not compose with the
magnitude, which is where the entire content is.

### What it does not give — recorded in the file, not in the margin

```lean
theorem unit_phase_does_not_bound_the_gate (θ : ℝ) :
    Complex.abs (Complex.exp ((θ : ℂ) * Complex.I)) = 1 ∧
      Complex.abs ((285.07 : ℝ) * Complex.exp ((θ : ℂ) * Complex.I)) = 285.07
```

Status: **PROVED**. The left conjunct is #16 at one factor; the right conjunct
is the modulus of a gate carrying that same unit phase, and it is `285.07` — R1's
measured `â_max` on its worst divergent seed, the number X₃₆ exists to remove.
A unit phase is true of RoPE, of a Fourier basis, of any unitary RNN, and of the
number `1`; it bounds the phase and says nothing whatever about the magnitude,
so it cannot be what removes the divergence. The same point at path length:

```lean
theorem divergence_needs_an_open_magnitude (θ : ℕ → ℝ) :
    Complex.abs (Wp (fun _ => (285.07 : ℝ)) θ 1 0) = 285.07
```

Status: **PROVED**. Read against `phase_path_le_one` in §3: the two statements
differ in exactly one hypothesis, the cap on `m`, and that hypothesis is the
load-bearing one. This is the `V15Kernel`/`V15Source` discipline applied to a
statement that would otherwise have shipped as a result.

---

## 2. The prefix-PHASE construction

```lean
noncomputable def C (m θ : ℕ → ℝ) (i : ℕ) : ℂ :=
  ∑ k in range (i + 1), ((Real.log (m k) : ℂ) + (θ k : ℂ) * Complex.I)

noncomputable def Wp (m θ : ℕ → ℝ) (i j : ℕ) : ℂ := Complex.exp (C m θ i - C m θ j)
```

`C` is §X₃₆'s `C_i = Σ log m + i Σ θ` verbatim; `Wp` is `CEQ.V15.W` with the
gates made complex.

### The construction, stated once

```lean
theorem Wp_polar (m θ : ℕ → ℝ) (hm : ∀ k, 0 < m k) {i j : ℕ} (hij : j ≤ i) :
    Wp m θ i j = ((∏ k in Ico (j + 1) (i + 1), m k : ℝ) : ℂ)
        * Complex.exp (((∑ k in Ico (j + 1) (i + 1), θ k : ℝ) : ℂ) * Complex.I)
```

**Hypotheses:** `m` pointwise positive (needed for `exp ∘ log = id`), `j ≤ i`.
Status: **PROVED**. The carrier factors exactly — a real magnitude which is the
path product of the `m`, times a unit-modulus factor which is the path sum of
the `θ`. **There is no cross term**, and that is why every magnitude statement
below holds for every phase schedule and every phase statement below holds for
every magnitude schedule.

### The modulus is #2's carrier, unchanged

```lean
theorem phase_modulus_is_the_real_carrier (m θ : ℕ → ℝ) (i j : ℕ) :
    Complex.abs (Wp m θ i j) = CEQ.V15.W (fun k => Real.log (m k)) i j
```

**Hypotheses: none** — no positivity, no `j ≤ i`, all `i j`. Status: **PROVED**.
X₃₆ does not restate or weaken V15's magnitude semantics; it multiplies them by
a factor of modulus 1. Everything `CEQ.V15` proved about `W` survives verbatim.

```lean
theorem prefix_phase_modulus (m θ : ℕ → ℝ) (hm : ∀ k, 0 < m k) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp m θ i j) = ∏ k in Ico (j + 1) (i + 1), m k
```

**Hypotheses:** `m > 0`, `j ≤ i`; `θ` free. Status: **PROVED**, by composing the
previous theorem with `CEQ.V15.prefix_logit_mask` (#2). This is the delta's
`|exp(C_i − C_j)| = ∏ m_k`.

---

## 3. The pair that matters

```lean
theorem phase_path_le_one (m : ℕ → ℝ) (h0 : ∀ k, 0 < m k) (h1 : ∀ k, m k ≤ 1)
    (θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) : Complex.abs (Wp m θ i j) ≤ 1

theorem phase_path_eq_one_iff_band (m : ℕ → ℝ) (h0 : ∀ k, 0 < m k) (h1 : ∀ k, m k ≤ 1)
    (θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp m θ i j) = 1 ↔ ∀ k ∈ Ico (j + 1) (i + 1), m k = 1
```

Both **PROVED**. Hypotheses on the magnitudes only: `0 < m k` and `m k ≤ 1`
pointwise, plus `j ≤ i`. **`θ` is bound inside the statement**, so the bound is
not "holds for the phases the model learns" — there is no phase schedule,
learned or adversarial, that makes the path product exceed 1. The divergence is
not bounded, it is unavailable.

The second is stated as an **iff**, which the delta's "= 1 exactly on the band"
asks for and which the delta's `≤`/`=` phrasing does not deliver: the modulus is
1 **if and only if** every magnitude on that path is 1, so any gate strictly
inside the cap anywhere on the path makes it strictly less. The two supporting
real-number lemmas (`prod_lt_one_of_mem`, `prod_eq_one_iff`) are proved
separately over `ℝ` because that is where the content is; `Wp_polar` is what
lets the complex statements reduce to them exactly.

---

## 4. What #6 gives, what it does NOT give, and what the cap adds

The delta says #16 "replaces #6's bound with an equality on the band". The
first half is recovered; **the second half is false as an inference from #6.**

```lean
theorem six_recovered (w θ : ℕ → ℝ) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp (fun k => Real.exp (-CEQ.V15.softplus (w k))) θ i j) ≤ 1
```
Status: **PROVED**. #6's bound holds inside the complex carrier.

```lean
theorem softplus_gate_lt_one (w : ℝ) : Real.exp (-CEQ.V15.softplus w) < 1

theorem six_never_reaches_the_band (w θ : ℕ → ℝ) {i j : ℕ} (hij : j < i) :
    Complex.abs (Wp (fun k => Real.exp (-CEQ.V15.softplus (w k))) θ i j) < 1
```

Both **PROVED**. Hypothesis on the path: `j < i`, i.e. non-empty. **For every
`w` and every `θ`, the modulus is STRICTLY below 1.** `CEQ.V15.bounded_gates_stable`
states `≤ 1`; the gap between `≤` and `<` is the whole of the band claim, and
`m ≡ 1` is on the wrong side of it — it is not in the image of
`w ↦ exp(−softplus w)` at all. So the delta's sentence *"on the band `m ≡ 1`
this becomes `= 1`"* is not a statement about #6's gates. The equality requires
the hard cap §X₃₆ calls for, and #6 does not have one:

```lean
noncomputable def cap (x : ℝ) : ℝ := min 1 (max 0 x)

theorem cap_band_attains (x θ : ℕ → ℝ) (hx : ∀ k, 1 ≤ x k) {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (Wp (fun k => cap (x k)) θ i j) = 1
```
Status: **PROVED**, with `cap_nonneg`, `cap_le_one`, `cap_eq_one_of_one_le`,
`cap_eq_zero_of_le_zero` giving the closed interval. This is the band case #16
is supposed to name, and it is the cap that delivers it, not the phase.

### The prior-art comparison, at proof strength

```lean
theorem lru_modulus_lt_one (ν θ : ℝ) :
    Complex.abs (Complex.exp (((-Real.exp ν : ℝ) : ℂ) + (θ : ℂ) * Complex.I)) < 1
```
Status: **PROVED**, no hypotheses. LRU (Orvieto 2023) parametrizes
`λ = exp(−exp ν + iθ)`, whose modulus is `exp(−exp ν) < 1` strictly, for every
`ν` and every `θ`. §X₃₆'s novelty claim — *"our closed `m` is the small
delta"* — is exactly this strict inequality: the LRU magnitude is open at 1 and
therefore cannot represent the band either. The claim is small and it is now
proved rather than asserted, and it applies to CEQ's own #6 with the same one
line, which is the finding of §4 and the thing this node did not expect to find.

---

## 5. `parity mask = Z₂ winding` — TRUE after `exp`, FALSE as written

### The reading that is true

```lean
noncomputable def phaseOf (p : ZMod 2) : ℝ := if p = 0 then 0 else Real.pi

lemma phase_gate_is_sign (p : ZMod 2) :
    Complex.exp ((phaseOf p : ℂ) * Complex.I) = ((CEQ.V15.chi p : ℝ) : ℂ)

theorem parity_is_Z2_winding (p : ℕ → ZMod 2) {i j : ℕ} (hij : j ≤ i) :
    Complex.exp (((∑ k in Ico (j + 1) (i + 1), phaseOf (p k) : ℝ) : ℂ) * Complex.I)
      = ((CEQ.V15.chi (CEQ.V15.pscan p i - CEQ.V15.pscan p j) : ℝ) : ℂ)
```

Both **PROVED**; hypothesis `j ≤ i` only. `θ ∈ {0, π}` gives `e^{iθ} ∈ {+1, −1}`
(`phase_gate_is_sign`), and the exponentiated prefix sum of phases along the
path equals `CEQ.V15.parity_sign`'s character of the prefix-XOR of the sign
bits. The `ZMod 2` prefix mask of V15 #3 and the complex phase scan of X₃₆ are
the same map. **This is the delta's `[RUN: True]`, and it holds.**

### The reading the delta writes is FALSE — the counterexample

The delta says "the prefix sum of phases mod `2π` corresponds to the prefix-XOR
of sign bits". Dropped to the phases themselves — which is what a `Z` winding
readout in §X₃₇(a) actually looks at — the correspondence fails:

```lean
theorem phase_sum_is_not_the_parity_phase :
    (∑ k in Ico 1 3, phaseOf ((fun _ => (1 : ZMod 2)) k)) = 2 * Real.pi ∧
      phaseOf (CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 2
          - CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 0) = 0 ∧
      (∑ k in Ico 1 3, phaseOf ((fun _ => (1 : ZMod 2)) k))
        ≠ phaseOf (CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 2
            - CEQ.V15.pscan (fun _ => (1 : ZMod 2)) 0)
```

Status: **PROVED**, and it is a refutation. **Counterexample in words:** take the
sign bits `p ≡ 1` and the path `Ico 1 3 = {1, 2}`. The phases accumulate
`π + π = 2π`. The prefix-XOR over the same path is `P₂ − P₀ = 1 − 1 = 0`, whose
phase is `0`. `2π ≠ 0`. The correspondence exists only after the quotient by
`2πℤ` that `exp` performs, which is the previous theorem and is not what the
delta's sentence says.

### And `Z₂` is not the `Z` winding §X₃₇(a) asks for

```lean
theorem Z2_forgets_the_winding :
    (∑ k in Ico 1 5, phaseOf ((fun _ => (1 : ZMod 2)) k)) = 2 * (2 * Real.pi) ∧
      Complex.exp (((∑ k in Ico 1 5, phaseOf ((fun _ => (1 : ZMod 2)) k) : ℝ) : ℂ)
        * Complex.I) = 1
```

Status: **PROVED**. The same all-ones parity sequence over `Ico 1 5` accumulates
`2·(2π)` — winding number **2** — and its character is `1`, the value the empty
path also gives. **The parity mask is the winding reduced mod 2 and does not
determine the integer.** §X₃₆'s `[RUN: True]` is therefore a true reading of a
statement strictly coarser than §X₃₇(a)'s per-instance integer certificate; the
identity does not discharge that certificate, and a node reporting `Z` winding
must read the phase sum, not the parity bit. Filed as a delta phrasing defect,
in the manner `CEQ.V15`'s header files #5's multiplicative reading.

---

## 6. Build — real output

Import line, at the top of `lean/CEQ/V15Phase.lean`:

```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.BigOperators
import CEQ.V15
```

`CEQ.V15` supplies `scan`, `W`, `prefix_logit_mask` (#2), `chi`, `pscan`,
`parity_sign` (#3), `softplus`, `softplus_pos` and `bounded_gates_stable` (#6);
nothing from `V15Fork`, `V15Kernel` or `V15Source` is used, and none of those
files, nor `lean/CEQ.lean`, is edited.

Command and output, verbatim:

```
$ cd lean && lake env lean CEQ/V15Phase.lean
EXIT=0
'CEQ.V15Phase.cap' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.C' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.Wp' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.phaseOf' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.unit_phase_product' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.unit_phase_does_not_bound_the_gate' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.cap_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.cap_le_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.cap_eq_one_of_one_le' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.cap_eq_zero_of_le_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.softplus_gate_lt_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.lru_modulus_lt_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.prod_lt_one_of_mem' depends on axioms: [propext, Quot.sound, Classical.choice]
'CEQ.V15Phase.prod_eq_one_iff' depends on axioms: [propext, Quot.sound, Classical.choice]
'CEQ.V15Phase.C_re' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.C_im' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.C_sub' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.Wp_polar' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.phase_modulus_is_the_real_carrier' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.prefix_phase_modulus' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.phase_path_le_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.phase_path_eq_one_iff_band' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.six_recovered' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.six_never_reaches_the_band' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.cap_band_attains' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.divergence_needs_an_open_magnitude' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.phase_gate_is_sign' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.parity_is_Z2_winding' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.phase_sum_is_not_the_parity_phase' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.Z2_forgets_the_winding' depends on axioms: [propext, Classical.choice, Quot.sound]
```

**30 declarations, 30 axiom lines, and the only axioms anywhere are Lean's own
three.** `sorryAx` appears nowhere, which is the check that matters and which
exit 0 alone would not give, since a `sorry` inside a macro still builds and
still exits 0. The output above is the *entire* output: no warnings, no `error`,
no `declaration uses 'sorry'`. `grep -nE "\bsorry\b|^axiom |admit"` over the
file returns one hit, the words "No `sorry`" in the module header.

---

## 7. WHAT THESE THEOREMS DO NOT COVER

Stated at the size of the gap, not at the size of the ambition.

**They do not say the arm trains.** `|a| ≤ 1` is proved for the carrier *as
parametrized*. Nothing here says gradient descent reaches a useful `m` inside
the cap, that the R1 re-run converges 8/8, or that the pre-registered interval
stops straddling. The author's own counter-prediction — that the cap removes
gain the divergent seeds were exploiting and the crossing shrinks to ≤ 3/8 — is
**not addressed and cannot be addressed by any theorem in this file**; it is a
measurement, and a proof that divergence is impossible is not evidence that
capability survives. Both remain open and both are still scored on the data.

**They do not cover the cap's gradient.** `cap = min 1 (max 0 ·)` is proved to
have the closed range `[0,1]` and to attain both ends. It is not differentiable
at either end, its subgradient is zero outside `(0,1)`, and nothing here says
that a magnitude pinned at the cap can move again. A hard cap is where the
"nothing to chase, nowhere to fall" claim of §X₃₆ is strongest and where its
optimization behaviour is least examined.

**They do not cover the phase's trainability.** `θ ∈ {0, π}` is a two-point set.
`phase_gate_is_sign` and `parity_is_Z2_winding` treat it as given, not as
learned; there is no continuous relaxation here and no claim that the `Z₂` phase
is reachable by gradient descent.

**They are single-channel and single-path.** Every statement is about one path
`j → i` in one scalar channel. There is no multi-channel statement, no Kuramoto
coupling, no order parameter `r`, no `β₁`, no Euler–Poincaré check, and nothing
about §X₃₇(b) or (c). §X₃₇(a)'s integer winding is *touched* only negatively, by
`Z2_forgets_the_winding` showing the parity bit does not supply it.

**They say nothing about `m = 0`.** `prefix_phase_modulus` and everything
downstream require `0 < m k`, because `log 0` is where the prefix-log carrier
stops existing. The cap's lower end is proved attainable (`cap_eq_zero_of_le_zero`)
and is then excluded by hypothesis from every theorem about the carrier. §X₃₆
claims `0` is an attainable value of the gate; on the log-prefix construction it
is attainable as a *gate* and is outside the *carrier's* domain, and no theorem
here closes that. An implementation that clamps `m` to `[ε, 1]` is covered; one
that admits exact `0` is not.

**They say nothing about the noisy or numerical regime.** Exact real arithmetic
throughout. `∏ m ≤ 1` in `ℝ` says nothing about float underflow of a long
prefix-log scan, and `= 1` on the band is an exact equality that float
accumulation of `Σ log m` will not reproduce. The delta's
`[RUN: 1e4 phases → modulus 1.000000000000]` is a measurement of a different
object than `phase_path_eq_one_iff_band`.

**And #16 itself covers nothing.** That is §1, and it is the reason the file is
not one theorem long.
