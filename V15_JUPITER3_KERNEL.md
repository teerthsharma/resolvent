# V15 JUPITER-3 — Lean #12 and #13, the kernel items

Node JUPITER-3 of the CEQ v15 composition round. Scope per dispatch:
`lean/CEQ/V15Kernel.lean` (new) and this file. `lean/CEQ.lean`,
`lean/CEQ/V15.lean` and `ceq/beds/**` were read, never written — confirmed by
`git diff --stat -- lean/CEQ.lean lean/CEQ/V15.lean`, which is empty. No git
write command was run. Nothing trained.

---

## STATUS TABLE

| # | statement | status | Lean name |
|---|---|---|---|
| 12 | delay bed, scalar state, ARBITRARY `f` | **GREEN** | `CEQ.V15.first_order_cannot_delay` |
| 12 | delay bed, the fitted affine model | **GREEN** | `CEQ.V15.affine_first_order_cannot_delay` |
| 12 | the naive reading, **REFUTED BY CONSTRUCTION** | **GREEN** | `CEQ.V15.delay_realizable_at_dimension_d` |
| 12 | the per-sequence reading, **REFUTED BY COUNTEREXAMPLE** | **GREEN** | `CEQ.V15.first_order_delays_constant_input` |
| 12 | the mechanism, no dimension and no linearity hypothesis | **GREEN** | `CEQ.V15.delay_forces_state_injective` |
| 12 | LTI state dimension `k` cannot delay `d > k` | **GREEN** | `CEQ.V15.linear_first_order_cannot_delay_beyond_state_dim` |
| 12 | power-law bed, both index alignments | **GREEN** | `CEQ.V15.first_order_cannot_powerlaw`, `..._strict` |
| 13 | closed form `= ` ratio recurrence | **GREEN** | `CEQ.V15.GL_weights_ratio_recurrence` |
| 13 | the two binds `α = 0`, `α = 1` | **GREEN** | `CEQ.V15.GL_weights_alpha_zero`, `..._alpha_one` |
| 13 | positivity and strict decay on `0 < α < 1` | **GREEN** | `CEQ.V15.GL_weights_pos`, `..._strictAnti` |
| 13 | the RATE `w_k ~ k^{α−1}/Γ(α)` | **`[S]`, NOT PROVED, NOT CLAIMED** | — |

No `sorry`. 30 declarations checked against the kernel's axiom list; `sorryAx`
appears nowhere. Build transcript and axiom transcript below.

**Verdict on the train-gate clause.** `CEQ_V15_CONTRACT.md` PART II says
"#12 before BED-K's scan verdict is called a proof." That clause is now
satisfiable **for the delay bed only**. It is NOT satisfiable for the power-law
bed in the sense the contract's PART III sentence uses, and the reason is a
measurement, not a proof gap — see WHAT THIS DOES NOT COVER, item 4.

---

## THE STATEMENT, AND WHY THIS ONE

### The chosen headline

```
CEQ.V15.first_order_cannot_delay : ∀ (f : ℝ → ℝ → ℝ) (y₀ : ℝ) {d : ℕ},
  1 ≤ d → ¬∀ (x : ℕ → ℝ), CEQ.V15.IsDelay d (CEQ.V15.firstOrder f y₀ x) x
```

with

```
CEQ.V15.IsDelay (d : ℕ) (y x : ℕ → ℝ) : Prop := ∀ i, y (i + d) = x i

CEQ.V15.vecRec (F : S → ℝ → S) (s₀ : S) (x : ℕ → ℝ) : ℕ → S
  | 0     => F s₀ (x 0)
  | i + 1 => F (vecRec F s₀ x i) (x (i + 1))

CEQ.V15.firstOrder (f : ℝ → ℝ → ℝ) (y₀ : ℝ) (x : ℕ → ℝ) : ℕ → ℝ := vecRec f y₀ x
```

**The hypotheses, stated rather than assumed away.**

1. **State dimension one, and the state IS the output.** `y_i = f (y_{i−1}) (x_i)`.
   This is not a convenience: it is the machine `ceq/beds/bed_k.py`'s
   `fit_first_order_recurrence` fits, which regresses the label on *the label's own
   true previous value*. The instrument's state is its own output, scalar, by
   construction.
2. **`f` is arbitrary.** Not affine, not linear, not continuous, not bounded. The
   theorem covers every nonlinear scalar recurrence at once, which is strictly
   stronger than the measurement — `R² = −0.000170` rules out the affine `f` only,
   and only on one realization.
3. **`y₀` is arbitrary.** Every initial condition.
4. **`d ≥ 1`.** At `d = 0` the claim is false and the theorem says so by omission:
   the identity recurrence delays by 0.
5. **The identity is required for EVERY input sequence.** Load-bearing — see
   the per-sequence refutation below.

**The witness, checkable by hand.** Realizing the delay forces, at every input,
`y_d = x_0` and `y_{d+1} = x_1`; the recurrence's own step gives
`y_{d+1} = f (y_d) (x_{d+1})`, so every input must satisfy

```
x_1 = f (x_0) (x_{d+1}).
```

The all-zero drive reads `0 = f 0 0`. The unit impulse at position 1 reads
`1 = f 0 0`, because `d ≥ 1` puts index `d + 1` off the impulse. Two explicit
sequences, two indices, no unrolling, no series. The `d ≥ 1` hypothesis is used
exactly once and visibly.

### Which of the dispatch's candidates this is, and which it is not

The dispatch named three readings. The file's position on each:

* **(a) scalar-state first-order cannot equal `y_i = x_{i−d}` for `d ≥ 1`** —
  **PROVED, and strengthened.** The dispatch's (a) is stated for
  `y_i = α y_{i−1} + β x_i`; that is `affine_first_order_cannot_delay`, a
  one-line corollary. The headline proves it for arbitrary `f`.
* **(b) state dimension `k` cannot produce delay `d > k`** — **PROVED for LINEAR
  (LTI) systems**, and the mechanism behind it proved with no linearity at all:

```
CEQ.V15.linear_first_order_cannot_delay_beyond_state_dim :
  ∀ {k : ℕ} (e : ℕ) (A : (Fin k → ℝ) →ₗ[ℝ] Fin k → ℝ) (B s₀ : Fin k → ℝ)
    (C : (Fin k → ℝ) → ℝ),
    k < e + 1 →
      (∀ x, IsDelay (e + 1) (fun i => C (vecRec (linStep A B) s₀ x i)) x) → False

CEQ.V15.delay_forces_state_injective :
  ∀ {S : Type u_1} (F : S → ℝ → S) (s₀ : S) (g : S → ℝ) (e : ℕ),
    (∀ x, IsDelay (e + 1) (fun i => g (vecRec F s₀ x i)) x) →
      Function.Injective fun u => vecRec F s₀ (pad u) e
```

  `delay_forces_state_injective` is the content of #12 with every dimension and
  linearity hypothesis removed: **a `d`-delay forces the state after the first `d`
  inputs to determine those `d` inputs.** Any state type, any update, any readout.
  The LTI theorem is that lemma plus one counting step (`finrank`), and it covers
  arbitrary `A`, `B`, readout `C` and arbitrary initial state `s₀` — the initial
  state is not fixed to zero, which cost the extra lemma `linRec_sub`.
* **(c) the transfer-function form `z^{−d} ≠ β/(1 − α z^{−1})`** — **not
  formalized as such.** It is the same fact as (a) in the frequency domain, and
  formalizing formal power series to say it would add machinery and no content.
  What replaces it, and is stronger for the power-law bed, is the impulse-response
  argument: `affineRec_impulse` proves the scalar recurrence's kernel is exactly
  `β γ^k`, and `GL_not_geometric` proves the target is not of that form.

### Why NOT the naive statement, proved rather than argued

The dispatch's warning is correct and the file discharges it as theorems, not as
prose:

```
CEQ.V15.delay_realizable_at_dimension_d : ∀ (d : ℕ) (x : ℕ → ℝ),
  IsDelay d (fun i => vecRec (shiftStep d) 0 x i (Fin.last d)) x
```

A shift register **is** a first-order recurrence — `shiftStep d s u =
Fin.cons u (fun k => s k.castSucc)` depends on the previous state and the current
input and on nothing else — and it produces a pure `d`-delay exactly, at state
dimension `d + 1`. So "no first-order recurrence delays" is FALSE, and the file
that proves #12 also contains its refutation. This is why the headline is stated
at dimension one and why the general statement is stated as a dimension bound.

The bracket is: **impossible for `k < d` (linear), possible at `k = d + 1`
(constructively).** The exact threshold `k = d` is not decided by this file.

Second false reading, also proved false:

```
CEQ.V15.first_order_delays_constant_input : ∀ (c y₀ : ℝ) (d : ℕ),
  IsDelay d (affineRec 0 1 y₀ fun _ => c) fun _ => c
```

On a **constant** drive the delayed label equals the undelayed one and `α = 0,
β = 1` reproduces it exactly, at every `d`. So "a first-order recurrence cannot
fit this delayed label" is false as a statement about one fixed input sequence.
`fit_first_order_recurrence`'s `R² = −0.000170` is evidence about one realization
of an iid drive; the quantifier `∀ x` is what turns it into a statement about the
delay MAP, and it is load-bearing.

Verified numerically at the same time as the theorem (`α = 0, β = 1`, `c = 3.7`,
10 steps): output `[3.7, 3.7, 3.7, 3.7, …]`, bitwise equal to the delayed label
at every `d`.

---

## #12, THE POWER-LAW BED

The contract attaches "scan-blind by Lean #12" to BED-K's *both* variants. The
delay half is above. The power-law half is proved separately and by a different
mechanism, because the delay argument does not reach it.

```
CEQ.V15.first_order_cannot_powerlaw : ∀ (α : ℝ), 0 < α → α < 1 →
  ∀ (γ β y₀ : ℝ), ¬∀ (x : ℕ → ℝ) (i : ℕ), affineRec γ β y₀ x i = kconv (glw α) x i

CEQ.V15.first_order_cannot_powerlaw_strict : ∀ (α : ℝ), 0 < α → α < 1 →
  ∀ (γ β y₀ : ℝ), ¬∀ (x : ℕ → ℝ) (i : ℕ),
    affineRecLag γ β y₀ x i = kconvStrict (glw α) x i
```

Two alignments are proved because `_powerlaw_kernel_matrix` builds the STRICTLY
causal kernel (`K[i, i−k] = ψ_k` for `k = 1..i`, no self term) while the contract
writes the weights `w_k = (−1)^k C(−α,k)` starting at `k = 0`. Comparing a
recurrence that reads `x_i` against a kernel that starts at lag 1 would win the
theorem on an index convention rather than on the shape of the kernel, which is
not what #12 is supposed to say. `affineRecLag` reads the previous input, so its
kernel is strictly causal too, and the refutation is forced to be about shape.

The mechanism, in one line each:

* `affineRec_impulse` / `affineRecLag_impulse` — the scalar recurrence's kernel is
  **geometric**, `β γ^k`. Two numbers fix the whole sequence.
* `GL_not_geometric` — `w_0 = 1` forces `β = 1`, `w_1 = α` forces `γ = α`, and then
  `w_2 = α(α+1)/2` must equal `α²`, i.e. `α = 1`. Three weights.
* `GL_not_geometric_tail` — the same at lag 1: `w_1 = α`, `w_2 = α(α+1)/2`,
  `w_3 = α(α+1)(α+2)/6` force `(α+1)/4 = (α+2)/6`, i.e. `α = 1`. So no shift of
  the alignment escapes.

Numeric instance, at BED-K's own registered point `H = 0.75`, `α = 0.25`, weights
read from `ceq/beds/bed_k.gl_weights`:

```
w = [1.0, 0.25, 0.15625, 0.1171875, ...]
w0*w2 − w1^2 = 0.09375          (the theorem: ≠ 0)
w1*w3 − w2^2 = 0.0048828125     (the theorem: ≠ 0)
closed form α(α+1)/2 − α² at α=0.25 = 0.09375   — matches to the digit
```

---

## #13 — `GL_weights_powerlaw`

```
CEQ.V15.GL_weights_ratio_recurrence : ∀ (α : ℝ) (k : ℕ), glwClosed α k = glw α k

  glw α 0 = 1,  glw α (k+1) = glw α k * (α + k) / (k + 1)          -- the implementation
  gbinom a k   = (∏ j in range k, (a − j)) / k!                     -- C(a, k)
  glwClosed α k = (−1)^k * gbinom (−α) k                            -- the contract
```

**Why this is the theorem and not a definitional unfolding.** Stated as "the
weights satisfy the ratio recurrence" the item would be true by `rfl` and would
compile in one token — the failure mode the dispatch forbids. The content is that
the closed form the CONTRACT writes and the recurrence the IMPLEMENTATION runs are
the same sequence. That gap is real and recent: `bed_k.py` stopped calling
`scipy.special.binom(-alpha, k)` after it was found to return NaN at `α = 1` and
switched to the recurrence, so the object the corpus is built from is no longer
the object the contract names. `GL_weights_ratio_recurrence` closes it.

The two binds:

```
CEQ.V15.GL_weights_alpha_zero : ∀ (k : ℕ), glw 0 (k + 1) = 0
CEQ.V15.GL_weights_alpha_one  : ∀ (k : ℕ), glw 1 k = 1
```

`α = 1` is exactly the point where `scipy.special.binom(-1, k)` returns NaN. The
value proved here is the value the ratio recurrence returns, so the theorem
adjudicates in favour of the implementation against the library at the one place
they disagree.

Numeric instance, run against `ceq/beds/bed_k.gl_weights` at the working tree:

```
alpha=0.25: impl   [1.0, 0.25, 0.15625, 0.1171875, 0.0952148438, 0.0809326172, 0.07081604]
            closed [1.0, 0.25, 0.15625, 0.1171875, 0.0952148438, 0.0809326172, 0.07081604]   maxdiff 0.000e+00
alpha=0.3 : maxdiff 5.551e-17
alpha=0.0 : impl and closed both [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     maxdiff 0.000e+00
alpha=1.0 : impl and closed both [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]     maxdiff 0.000e+00
```

**The asymptotic is `[S]`, not done, and is not claimed anywhere in the file.**
`w_k ~ k^{α−1}/Γ(α)` needs Stirling and Gamma-ratio asymptotics that this
toolchain's mathlib does not put within a node's reach. What is proved instead is
the part the round actually consumes: `GL_weights_pos` (positive for `α > 0`) and
`GL_weights_strictAnti` (strictly decreasing for `0 < α < 1`). The power-law half
of #12 does not need the RATE — it needs the failure of GEOMETRIC decay, and that
is proved outright from three weights.

---

## VORT (arXiv:2605.08966) — DOES ITS RESULT (iii) COVER BED-K'S BOX?

**Determination: NO. The regimes are disjoint, and the boundary between them is
the same object in both papers.**

Basis, in order.

**1. The abstract was read verbatim, not through a summarizer.** `WebFetch` on the
arXiv HTML abstract page returned a PARAPHRASE that dropped the `α` range entirely
— the failure the coordinator warned about, reproduced. The verbatim text was
obtained from the arXiv API (`export.arxiv.org/api/query?id_list=2605.08966`),
whose `<summary>` element is the abstract as submitted. The two sentences that
decide the question:

> "each ingested token is assigned a learnable fractional order α_i∈[δ,1] that
> governs a Grünwald–Letnikov power-law retention kernel"

> "(iii) a direct L^2 energy argument (Proposition) showing that for α>1/2 any
> mixture with fixed minimum decay rate Λ>0 incurs L^2([1,T]) error at least
> N_α(T)-C(Λ)→∞, with the Λ-dependence made explicit"

**2. VORT's `α` and the contract's `α` are the SAME parameter.** Three
independent checks, all agreeing:

* *Direction of the parametrization.* VORT calls it a **retention** kernel with
  `α ∈ [δ, 1]`. Under the fractional-INTEGRAL convention — the contract's
  `w_k = (−1)^k C(−α,k)`, coefficients of `(1−z)^{−α}` — `α → 0` gives
  `[1,0,0,…]` (no memory) and `α = 1` gives all ones (full retention). Retention
  increases monotonically across `[δ,1]`, which is what a learnable retention
  order should do. Under the GL **derivative** convention the weights alternate in
  sign and `α = 1` is the first difference, which is not a retention kernel at
  all. Both binds are theorems in this file (`GL_weights_alpha_zero`,
  `GL_weights_alpha_one`), so this is not a reading of prose.
* *Positivity / Laplace representation.* VORT approximates the weights by a
  **sum of exponentials from a Laplace-type integral representation**. That
  representation exists for completely monotone, positive weight sequences. The
  integral-convention weights are positive for `α > 0` (`GL_weights_pos`); the
  derivative-convention weights are not.
* *The critical exponent.* VORT's threshold is `α > 1/2` and its engine is the
  divergence of `N_α(T)`, the `L²([1,T])` energy. For weights decaying as
  `k^{α−1}`, `∑_k w_k² ~ ∑ k^{2α−2}` diverges **iff `α ≥ 1/2`**. That is
  numerically the same threshold, and analytically the same computation, as
  ARFIMA's stationarity condition `|d| < 1/2` with `H = d + 1/2`, which is where
  BED-K's box comes from and why `_powerlaw_kernel_matrix` raises outside it. Two
  papers landing on `1/2` for the same `∑ k^{2α−2}` reason is strong evidence the
  `α` is one object.

**3. Therefore result (iii) does not reach BED-K.** BED-K's registered box is
`α = H − 1/2 ∈ (0, 1/2)`, `H ∈ (0.5, 1.0)`. VORT's Proposition is stated for
`α > 1/2`. The regimes are disjoint, and not incidentally: in BED-K's box the
target kernel's `L²` energy **converges**, so `N_α(T)` does not diverge and the
argument cannot even be stated there, let alone applied. The exclusion is
structural, not a matter of the authors not having bothered.

**Consequence for this node.** `first_order_cannot_powerlaw` is stated on
`α ∈ (0, 1)` — it covers BED-K's box entirely, and it also covers VORT's regime up
to `α = 1`, by a different mechanism (exact non-representability of a geometric
kernel, not an `L²` divergence). It is not a restatement of published work in the
stationary box, which was the coordinator's condition (2).

**What VORT DOES occupy, and the round should cite rather than claim.** The
architecture — a learnable per-token fractional order driving a GL power-law
retention kernel inside a transformer, with an `O(S d_v)` per-step Markovian
recurrence per exponential component — is published and spans BED-K's whole `α`
range. PART I §S-K's fractional head is occupied. Only the non-representability
THEOREM is regime-limited.

---

## WHAT THIS DOES NOT COVER — read before citing #12

**1. Hidden scalar state with a nonlinear update.** `first_order_cannot_delay`
requires the state to BE the output. A machine `s_i = f(s_{i−1}, x_i)`,
`y_i = g(s_i)` with `s_i ∈ ℝ` and pathological `f, g` CAN delay: `ℝ²` injects into
`ℝ` set-theoretically, so `delay_forces_state_injective` — which is exactly the
statement that the state must retain the inputs — is satisfiable at scalar state
by a digit-interleaving encoder. No theorem here rules that out, and none can
without a regularity hypothesis (measurability, continuity, or linearity). This is
why the scalar theorem is stated for output-equals-state and the dimension theorem
is stated for linear maps, and why neither is stated in the other's generality.
The instrument fits output-equals-state, so this gap does not touch the BED-K
verdict; it would touch a claim about "any RNN with a scalar hidden unit".

**2. Nonlinear machines at state dimension `k`.**
`linear_first_order_cannot_delay_beyond_state_dim` assumes an LTI system. A
nonlinear `k`-dimensional machine is covered only by
`delay_forces_state_injective`, which gives injectivity, not a dimension bound.

**3. The threshold is bracketed, not pinned.** Impossible for `k < d` (linear),
possible at `k = d + 1` (constructive). `k = d` is not decided here. Kronecker's
theorem gives minimal realization order `= d` for the pure delay; that is not
formalized.

**4. NOTHING HERE IS AN APPROXIMATION BOUND, AND THE POWER-LAW BED IS NOT
`R²`-SCAN-BLIND.** Every theorem in the file refutes an EXACT identity for EVERY
input. None of them bounds how well a first-order recurrence can approximate the
label on one realization, and on the power-law bed the difference is decisive.
Measured, at the working tree, with `bed_k.fit_first_order_recurrence` (the same
instrument N4 used, state = the true previous label, `n = 2048`, `seed = 0`):

```
delay bed    (d=5)              α̂=-0.0109  β̂=-0.0216  R² = -0.000166
power-law bed (H=0.75, α=0.25)  α̂= 0.7975  β̂=-0.0022  R² =  0.603580
   H=0.60 (α=0.10) R² = 0.452312
   H=0.70 (α=0.20) R² = 0.551956
   H=0.80 (α=0.30) R² = 0.655309
   H=0.90 (α=0.40) R² = 0.754833
```

The delay-bed number reproduces `V15_N4_BEDK.md`'s `−0.000170` to rounding, which
is the cross-check that the corpus is unchanged. The power-law number does not
support the contract's PART III sentence. **"Attention-native, scan-blind by Lean
#12" is licensed by this file for the DELAY bed and is NOT licensed for the
POWER-LAW bed**: a first-order recurrence explains 60% of the power-law label's
variance at BED-K's own registered `H = 0.75`, rising to 75% at `H = 0.9`, and
`first_order_cannot_powerlaw` is entirely consistent with that — it says the
recurrence never EQUALS the kernel, not that it fits badly. Citing #12 for
power-law scan-blindness would be exactly the `[V]`-as-theorem defect L-EQ was
written against, with a theorem underneath instead of a page, which is worse
rather than better.

**5. Two contract arithmetic defects, filed.**

* `CEQ_V15_CONTRACT.md` PART III tags the BED-K sentence
  `[RUN: best first-order recurrence 0.990]`. The measurements above are
  `−0.000166` (delay) and `0.604` (power-law, `H = 0.75`). Neither is `0.990`, at
  any `H` in the registered box. The tag is unsourced against the current
  implementation and should be re-derived or struck.
* PART IV **R3** reads "BED-K delay bed: scan-only `>= 0.95` (theorem-backed),
  attention / fractional head near 0". That is the exact inverse of PART III's
  "attention-native, scan-blind" for the same bed, and of N4's measurement
  (attention reaches `8.67e-19`; the scan reaches `R² ≈ 0`). Read literally, R3
  asks for a scan-only score of `≥ 0.95` on the delay bed, which
  `first_order_cannot_delay` proves is unreachable in the exact sense and which
  N4 measured at `0.000` in the fitted sense. The two arms appear to be
  transposed in R3. Which one the author intended is not something this node can
  settle; that R3 and PART III cannot both stand is.

**6. The `d = 0` case.** Excluded by hypothesis and false without it.

**7. Continuous time, and multi-input/multi-output systems.** Everything is a
discrete-time single-input single-output recurrence.

---

## BUILD OUTPUT — real, pasted

The file was verified **STANDALONE**, not as part of `lake build`, per the
dispatch's collision-avoidance instruction. `lean/CEQ.lean` was not edited, so
`lake build`'s default target does not reach `CEQ/V15Kernel.lean` yet.

```
$ cd lean && lake env lean CEQ/V15Kernel.lean; echo "=== EXIT CODE: $? ==="
=== EXIT CODE: 0 ===
```

No output at all: no errors, no warnings, no `sorry` warning, no linter warning.
Toolchain `leanprover/lean4:v4.7.0`, mathlib vendored at the matching tag —
identical to the toolchain `lean/CEQ/V15.lean` was built with.

```
$ grep -n sorry lean/CEQ/V15Kernel.lean
72:  No `sorry`.
```

The single hit is the sentence in the module doc comment.

**IMPORT LINE FOR THE COORDINATOR** — to be added to `lean/CEQ.lean` by whoever
owns that file, next to the existing `import CEQ.V15`:

```
import CEQ.V15Kernel
```

`CEQ/V15Kernel.lean` reopens `namespace CEQ.V15`; every name it introduces is new
(`vecRec`, `firstOrder`, `affineRec`, `IsDelay`, `dirac`, `window`, `shiftStep`,
`pad`, `linStep`, `prefixMap`, `glw`, `gbinom`, `glwClosed`, `kconv`,
`kconvStrict`, `lag`, `affineRecLag`), so there is no clash with `CEQ/V15.lean`'s
`chain`, `scan`, `W`, `Wc`, `chi`, `pscan`, `softplus`, `Aff`, `affApply`,
`affComp`. The two files do not import each other.

---

## AXIOM CHECK — real, pasted

Exit 0 is not sufficient: a `sorry` inside a macro, or a theorem closed by an
axiom, still builds. Every declaration was therefore printed against the kernel's
axiom list, by appending `#print axioms` lines to the file, running it, and
removing them again.

```
$ cd lean && lake env lean CEQ/V15Kernel.lean      # with #print axioms appended
'CEQ.V15.first_order_cannot_delay' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.affine_first_order_cannot_delay' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.first_order_delays_constant_input' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.window_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.window_step' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.window_eq_vecRec' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.window_isDelay' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.delay_realizable_at_dimension_d' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.delay_forces_state_injective' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.linRec_add' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.linRec_smul' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.pad_add' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.pad_smul' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.linRec_sub' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.linear_first_order_cannot_delay_beyond_state_dim' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.glw_succ' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_weights_ratio_recurrence' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_weights_alpha_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_weights_alpha_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_weights_pos' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_weights_strictAnti' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.kconv_dirac_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.affineRec_impulse' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_not_geometric' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.first_order_cannot_powerlaw' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.lag_dirac_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.kconvStrict_dirac_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.affineRecLag_impulse' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.GL_not_geometric_tail' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.first_order_cannot_powerlaw_strict' depends on axioms: [propext, Classical.choice, Quot.sound]
=== EXIT CODE: 0 ===
```

30 of 30. `propext`, `Classical.choice`, `Quot.sound` are Lean's three standard
axioms. **`sorryAx` appears nowhere.**

---

## THE STATEMENTS AS THE KERNEL ELABORATED THEM

Pasted from `#check`, not retyped, so no weakening can hide in the transcription.

```
CEQ.V15.first_order_cannot_delay : ∀ (f : ℝ → ℝ → ℝ) (y₀ : ℝ) {d : ℕ},
  1 ≤ d → ¬∀ (x : ℕ → ℝ), CEQ.V15.IsDelay d (CEQ.V15.firstOrder f y₀ x) x

CEQ.V15.affine_first_order_cannot_delay : ∀ (α β y₀ : ℝ) {d : ℕ},
  1 ≤ d → ¬∀ (x : ℕ → ℝ), CEQ.V15.IsDelay d (CEQ.V15.affineRec α β y₀ x) x

CEQ.V15.first_order_delays_constant_input : ∀ (c y₀ : ℝ) (d : ℕ),
  CEQ.V15.IsDelay d (CEQ.V15.affineRec 0 1 y₀ fun x => c) fun x => c

CEQ.V15.delay_realizable_at_dimension_d : ∀ (d : ℕ) (x : ℕ → ℝ),
  CEQ.V15.IsDelay d (fun i => CEQ.V15.vecRec (CEQ.V15.shiftStep d) 0 x i (Fin.last d)) x

@CEQ.V15.delay_forces_state_injective : ∀ {S : Type u_1} (F : S → ℝ → S) (s₀ : S) (g : S → ℝ) (e : ℕ),
  (∀ (x : ℕ → ℝ), CEQ.V15.IsDelay (e + 1) (fun i => g (CEQ.V15.vecRec F s₀ x i)) x) →
    Function.Injective fun u => CEQ.V15.vecRec F s₀ (CEQ.V15.pad u) e

@CEQ.V15.linear_first_order_cannot_delay_beyond_state_dim : ∀ {k : ℕ} (e : ℕ)
  (A : (Fin k → ℝ) →ₗ[ℝ] Fin k → ℝ) (B s₀ : Fin k → ℝ) (C : (Fin k → ℝ) → ℝ),
  k < e + 1 →
    (∀ (x : ℕ → ℝ), CEQ.V15.IsDelay (e + 1)
      (fun i => C (CEQ.V15.vecRec (CEQ.V15.linStep A B) s₀ x i)) x) → False

CEQ.V15.GL_weights_ratio_recurrence : ∀ (α : ℝ) (k : ℕ), CEQ.V15.glwClosed α k = CEQ.V15.glw α k

CEQ.V15.GL_weights_alpha_zero : ∀ (k : ℕ), CEQ.V15.glw 0 (k + 1) = 0

CEQ.V15.GL_weights_alpha_one : ∀ (k : ℕ), CEQ.V15.glw 1 k = 1

CEQ.V15.GL_weights_strictAnti : ∀ (α : ℝ), 0 < α → α < 1 →
  ∀ (k : ℕ), CEQ.V15.glw α (k + 1) < CEQ.V15.glw α k

CEQ.V15.GL_not_geometric : ∀ (α : ℝ), 0 < α → α < 1 →
  ¬∃ β γ, ∀ (k : ℕ), CEQ.V15.glw α k = β * γ ^ k

CEQ.V15.GL_not_geometric_tail : ∀ (α : ℝ), 0 < α → α < 1 →
  ¬∃ β γ, ∀ (k : ℕ), CEQ.V15.glw α (k + 1) = β * γ ^ k

CEQ.V15.first_order_cannot_powerlaw : ∀ (α : ℝ), 0 < α → α < 1 → ∀ (γ β y₀ : ℝ),
  ¬∀ (x : ℕ → ℝ) (i : ℕ), CEQ.V15.affineRec γ β y₀ x i = CEQ.V15.kconv (CEQ.V15.glw α) x i

CEQ.V15.first_order_cannot_powerlaw_strict : ∀ (α : ℝ), 0 < α → α < 1 → ∀ (γ β y₀ : ℝ),
  ¬∀ (x : ℕ → ℝ) (i : ℕ),
    CEQ.V15.affineRecLag γ β y₀ x i = CEQ.V15.kconvStrict (CEQ.V15.glw α) x i
```

---

## PROVENANCE OF THE NUMERIC INSTANCES

All numbers above were produced this session by inline `python -` invocations
against the repository's own modules — no file was written outside the two in
scope. `ceq/beds/bed_k.py` was modified concurrently by another node during this
one (an X35 "planted hidden cause" field on `build`, since landed as `b0df56e`);
every reading above was re-run against the committed tree afterwards and
reproduces to the digit:

```
$ git log --oneline -1
b0df56e Build the residual onset detector, and show its flatness gate can reject
delay    R2 = -0.000166
powerlaw R2 =  0.603580
gl_weights(0.25, 3) = [1.0, 0.25, 0.15625, 0.1171875]
```

`gl_weights`, `build_delay`, `build_powerlaw` and `fit_first_order_recurrence`
are untouched by that diff, and the delay-bed `R²` reproducing N4's published
`−0.000170` is the check that the corpus these numbers came from is the corpus N4
measured.

---

## FILES TOUCHED

- `lean/CEQ/V15Kernel.lean` — new, 640 lines, 30 top-level `theorem`/`lemma`
  declarations plus 17 definitions.
- `V15_JUPITER3_KERNEL.md` — this file.

`lean/CEQ.lean`, `lean/CEQ/V15.lean`, `ceq/beds/**`, `tests/**` were read and not
written. No git write command was run.

---

## DISTANCE TO THE NORTH STAR

Unchanged in direction, shortened by one gate. "Attention that is EQUAL to
self-attention on its own ground and capable on ground it cannot occupy" needs a
bed where the two differ and a proof that they differ by mechanism rather than by
seed. #12 supplies exactly half of that: the DELAY bed's split is now a theorem —
no first-order recurrence of any shape reaches the label, while N4's hand-set head
reaches it to `8.67e-19`. The POWER-LAW bed's split is not a theorem and, at
`R² = 0.60` for a scan, is not currently a measurement either; PART III's sentence
covers a bed it has not earned. No floor, no arm and no training number moves
here.
