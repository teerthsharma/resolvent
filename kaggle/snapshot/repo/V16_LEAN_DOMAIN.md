# V16 JUPITER (it. 2–3) — L-DOM: #2 RE-STATED ON THE CORPUS'S SUPPORT, THE DOMAIN CENSUS, #5a AND #5b

Node `JUPITER`, iterations 2–3 of CEQ v16 (the Banking Round). Discharges
`CEQ_V16_CONTRACT.md` PART II items **#2 (re-statement clause)**, **#5a**,
**#5b**, PART V debt **D-DOM**, and the **TRAIN-GATE** line.

**Files written.** `lean/CEQ/V16Domain.lean` (new, 594 lines, 47 declarations,
no `sorry`), this file. `lean/CEQ.lean` was **not** edited — the import line is
handed to the coordinator in §7. No existing `.lean` file, no `MISTAKES.md`, no
`scale/`, no `ceq/` file was touched. No git command that writes was run.
Repo HEAD at run time `5ed3091b8b2cd00bc5b396b1802cf37412d183c0`; the working
tree also carried another node's edit to `scale/negation_scope.py` (a `device=`
keyword on `calibrate_bar`, lines 1409–1483) which does **not** touch the gate
draw this census reads. Lean `leanprover/lean4:v4.7.0`, mathlib4 `v4.7.0`.
Nothing trained.

---

## VERDICT TABLE

| question | answer |
|---|---|
| **(a) #2 re-stated over the full support?** | **YES.** `prefix_logit_mask_restated`, hypotheses `0 ≤ m k ≤ 1` for every `k`, `θ` free, `j ≤ i`. Five clauses, both endpoints of `[0,1]` included. `m = 0` is clause 4 — a **theorem** (the product annihilates), not an excluded hypothesis. |
| **(b) Was the conclusion weakened to buy the domain?** | **NO.** `pathProd_abs` has a hypothesis strictly weaker than #2's (`0 ≤ m` vs `0 < a`) and the same conclusion (the path product). `restatement_subsumes_phase_path_le_one` proves the two agree wherever #2 had a value. Both are in the file, diffable. |
| **(c) What WAS lost?** | **The log-domain computational route, and only on windows containing a zero gate.** `no_prefix_scan_represents_a_zero_gate`: `Complex.exp` is never `0`, so **no** prefix scan `C` whatsoever represents a hop that annihilates. Not a weaker theorem — a form that does not exist there. Repair is a segmented scan that resets at zeros, or the product itself. |
| **(d) #2 on BED-M** | Old hypothesis `0 < a`: **1 of 3** drawn values (`#eval` reads `1`). Re-stated `0 ≤ \|a\| ≤ 1`: **3 of 3** (`#eval` reads `3`). |
| **(e) #6 on BED-M** | **0 of 3. `bounded_gates_stable` is DECORATION on BED-M.** `six_misses_every_bedM_value` proves it for the whole family at once: no `w` sends `exp(−softplus w)` to `−1`, `0`, or `+1`. |
| **(f) #12 on BED-K** | **PARTIAL.** `d ≥ 1` covered; **`d = 0` is a legal BED-K(a) cell and is outside the theorem, necessarily** — `delay_zero_is_first_order` exhibits the recurrence that realizes it. **Empty** on BED-K(b) and on BED-M. |
| **(g) #16 on both beds** | **Overlap TOTAL, content NULL.** `#16` has no hypotheses, so no overlap count can ever fail it. `sixteen_is_silent_on_the_zero_draw` shows it reads `1` on the very draw where BED-M's gate is `0`. **L-DOM's test is necessary, not sufficient**, and #16 is the witness. |
| **(h) #5a** | **PROVED.** `three_corners_containment`, three settings of one family. `corners_are_distinct` included because a containment whose corners coincide is decoration: `β=1` reads `1/2` at `(1,0)`, `β=0` reads `1`. |
| **(i) #5b** | **PROVED, both halves.** `gate_zero_beta_zero_is_linear_attention` = the identification **and** the separation witness. `gate_zero_beta_zero_row_not_one` composes with `CEQ.V15.gate_zero_not_stochastic` for the whole-row separation; `softmax_row_sum_one` is the class marker it is separated from. |
| **(j) Train-gate** | **GREEN.** `#1, #2(original), #2(re-stated), #3, #5a, #5b, #6, #7, #16` all `[propext, Classical.choice, Quot.sound]`, no `sorryAx`. **Nothing previously green was broken.** |
| **(k) Build** | `lake build` **exit 0**. `lake env lean CEQ/V16Domain.lean` **exit 0**. All five existing V15 files force-re-elaborated from source, **exit 0** each. |

---

## 1. THE RE-STATED #2

### 1.1 The problem, at its exact size

`CEQ.V15.prefix_logit_mask` is

```lean
theorem prefix_logit_mask (a : ℕ → ℝ) (ha : ∀ k, 0 < a k) {i j : ℕ} (hij : j ≤ i) :
    W (fun k => Real.log (a k)) i j = ∏ k in Ico (j + 1) (i + 1), a k
```

BED-M's gate support, **read by running the generator**, not assumed:

```
make_equilibrium_batch  a support: [-1.0, 0.0, 1.0]     scale/negation_scope.py:428-429
make_propagate_batch    a support: [-1.0, 1.0]          scale/negation_scope.py:658
```

`a = +1` satisfies `0 < a`. `a = 0` and `a = −1` do not. One of three.

### 1.2 The construction, and why it is a case split and not a weakened hypothesis

`a_k = m_k · e^{iθ_k}`, `m_k ∈ [0,1]` **closed**, `θ_k` free. The hop is taken
as the path product directly:

```lean
noncomputable def gateOf (m θ : ℝ) : ℂ := (m : ℂ) * Complex.exp ((θ : ℂ) * Complex.I)

noncomputable def pathProd (m θ : ℕ → ℝ) (i j : ℕ) : ℂ :=
  ∏ k in Ico (j + 1) (i + 1), gateOf (m k) (θ k)
```

Nothing in that definition mentions `log`. That is the whole of the repair.

**The re-statement.**

```lean
theorem prefix_logit_mask_restated (m θ : ℕ → ℝ) (h0 : ∀ k, 0 ≤ m k) (h1 : ∀ k, m k ≤ 1)
    {i j : ℕ} (hij : j ≤ i) :
    Complex.abs (pathProd m θ i j) = ∏ k in Ico (j + 1) (i + 1), m k
      ∧ Complex.abs (pathProd m θ i j) ≤ 1
      ∧ (Complex.abs (pathProd m θ i j) = 1 ↔ ∀ k ∈ Ico (j + 1) (i + 1), m k = 1)
      ∧ ((∃ k ∈ Ico (j + 1) (i + 1), m k = 0) → pathProd m θ i j = 0)
      ∧ ((∀ k, 0 < m k) → pathProd m θ i j = CEQ.V15Phase.Wp m θ i j)
```

| clause | content | hypothesis it needs |
|---|---|---|
| 1 | `\|∏ a\| = ∏ m` | `0 ≤ m` **only** — `m = 0` and `m = 1` both in |
| 2 | `\|∏ a\| ≤ 1` | `0 ≤ m ≤ 1` |
| 3 | `\|∏ a\| = 1` ⟺ the path is on the band | `0 ≤ m ≤ 1` |
| 4 | **off the band the product is exactly `0`** | none beyond a zero on the path |
| 5 | on `m > 0` the exponential form of #2 is recovered **unchanged** | `0 < m` |

Clause 4 is the one #2 could not state. Clause 5 is the receipt that nothing
was traded for it.

**Sign lives in the phase, and it is an ordinary point.** `bedM_gate_exact`
proves, for each of `{−1, 0, +1}` separately, that `gateOf |a| (arg a) = a`
exactly and `|a| ∈ [0,1]`; `negative_draw_is_on_the_band` isolates
`gateOf 1 π = −1` with modulus `1`. No branch of `Real.log` appears.

### 1.3 What was lost, stated exactly

**Not the conclusion.** `pathProd_abs`'s hypothesis (`0 ≤ m`) is strictly
weaker than #2's (`0 < a`), its conclusion is the same path product, and
`pathProd_eq_Wp` plus `restatement_subsumes_phase_path_le_one` prove the two
objects are literally equal wherever #2 had a value. On the overlap the pair is
identical; off it only the re-statement has a value. That is the diff.

**What was lost is the log-domain COMPUTATIONAL ROUTE.**

```lean
theorem no_prefix_scan_represents_a_zero_gate (C : ℕ → ℂ) (m θ : ℕ → ℝ) {i j : ℕ}
    (hz : ∃ k ∈ Ico (j + 1) (i + 1), m k = 0) :
    Complex.exp (C i - C j) ≠ pathProd m θ i j
```

Two lines: `Complex.exp` is never zero, the path product is. This is
**structural, not a Lean artifact and not a hypothesis one could relax**. On a
window containing a zero gate the parallel prefix scan plus subtraction does
not compute the hop, and there is no `C` — no repaired `log`, no extended-real
convention, no re-parametrization — that makes it. The route survives on such
windows only as a segmented scan that resets at zeros, or as the product
itself. That is the cost, and it is the engineering consequence the census
exists to surface.

**The `−∞` reading is worse than local, and this is why the `nan` in
`V15_ARM_PHASE.md` (e) is not a numerical accident.** Under the honest
extended-real convention `log 0 = −∞`, a single `m_k = 0` sends `C_n = −∞` for
**every** `n ≥ k`, so `C_i − C_j` is `−∞ − (−∞)` for every pair `(i, j)` lying
entirely **after** the zero — pairs whose path product is perfectly
well-defined and generically nonzero. One zero gate poisons the difference
form globally downstream, not locally. `V15_ARM_PHASE.md`'s band residual of
`nan` is that arithmetic made visible on real hardware.

### 1.4 The refusal, filed in-file

`0 ≤ a` with the conclusion kept is **refused**, and refuted twice.

```lean
theorem lean_log_junk_makes_the_scan_form_silently_false :
    CEQ.V15Phase.Wp (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 = 1
      ∧ pathProd (fun _ => (0 : ℝ)) (fun _ => 0) 1 0 = 0
      ∧ ... ≠ ...
```

Lean's `Real.log 0 = 0` is a junk value. The naive relaxation therefore does
**not** fail loudly — it returns `1` where the path product is `0`. Filed so
that no later file reads a green `Real.log`-based statement on `[0,1]` as
evidence about the `m = 0` draw. `no_prefix_scan_represents_a_zero_gate` is the
reason underneath it and does not depend on Lean's convention at all.

### 1.5 A defect inherited by V15Phase, named

`CEQ.V15Phase.Wp_polar`, `prefix_phase_modulus`, `phase_path_le_one` and
`phase_path_eq_one_iff_band` all carry `∀ k, 0 < m k`. They inherited #2's
domain defect verbatim: **the entire X₃₆ magnitude chain is decoration on
BED-M's `a = 0` draw**, including the two theorems `V15_X36_LEAN.md` presents
as "the pair the delta is asking for". `pathProd_abs` and clauses 1–4 above are
their replacements on the closed support; clause 5 and
`restatement_subsumes_phase_path_le_one` are the proof that nothing in them is
contradicted. No V15 file was edited to say so.

---

## 2. THE DOMAIN CENSUS

**Value support, read from the generators.**

| bed | object | support | how read |
|---|---|---|---|
| BED-M `make_equilibrium_batch` | gate `a` | `{−1, 0, +1}` | ran the builder: `torch.unique` over `512×64` = `[-1.0, 0.0, 1.0]`; source `scale/negation_scope.py:428-429` (`randint(0,2)*2−1`, then `a[:, :head+1] = 0.0`) |
| BED-M `make_propagate_batch` | gate `a` | `{−1, +1}` | ran the builder: `[-1.0, 1.0]`; source line 658, no prefix zeroing |
| BED-M | drive `b` | `ℝ`, `N(0,1)`, `b[s−1] = 0` | line 427, 430 |
| BED-K (a) delay | kernel `K` | `{0, 1}`; `d ≥ 0` accepted (`d < 0` rejected) | ran `kernel_matrix('delay', 8, d=3)` → `unique = [0., 1.]`; `d=0` builds and gives `K = I` |
| BED-K (b) power-law | kernel `K` | `(0, α]` off-diagonal, strictly decreasing, `α = H − ½ ∈ (0, ½)`; zero on and above the diagonal | ran `kernel_matrix('powerlaw', 8, H)`: `H=0.60 → [0.018123, 0.100000]`, `H=0.75 → [0.063229, 0.250000]`, `H=0.90 → [0.137871, 0.400000]`, zero off-triangle entries `0` |
| BED-K both | drive `b` | `ℝ`, `N(0,1)` unbounded | `build()`; observed `[−3.8994, 3.0660]` at `n=2048, seed=0` |

### 2.1 #2 `prefix_logit_mask`

| | theorem's hypotheses | bed's support | overlap | verdict |
|---|---|---|---|---|
| **#2 original**, BED-M (equil.) | `∀ k, 0 < a k` | `{−1, 0, +1}` | **1 of 3** (`+1` only) | **DECORATION on two thirds of BED-M's draws.** Green in Lean, inapplicable to `0` and `−1`. |
| **#2 original**, BED-M (propagate) | `∀ k, 0 < a k` | `{−1, +1}` | **1 of 2** | Decoration on half the draws. |
| **#2 re-stated**, BED-M (equil.) | `0 ≤ m k ≤ 1`, `θ` free | `{−1, 0, +1}` ↦ `(m,θ) ∈ {(1,π), (0,·), (1,0)}` | **3 of 3** | **Full.** `bedM_gate_exact` proves the mapping is exact, value by value. |
| **#2 re-stated**, BED-M (propagate) | as above | `{−1, +1}` | **2 of 2** | Full. |
| **#2 original**, BED-K (a) delay | `∀ k, 0 < a k` | there is no gate sequence `a`; the label generator is `z = K b`, `K ∈ {0,1}` | **EMPTY** | **#2 IS DECORATION ON BED-K(a).** Two independent reasons: the bed instantiates no `a`, and reading `K` as the hop is refuted outright — `K[i,j] = 0` off the delay diagonal while `exp(C_i − C_j)` is never zero (`no_prefix_scan_represents_a_zero_gate` with `m` the zero row). |
| **#2 original**, BED-K (b) power-law | `∀ k, 0 < a k` | `K` entries in `(0, α]` — positive, but not a path product of any gate sequence | **EMPTY as stated** | **#2 IS DECORATION ON BED-K(b).** The entries are positive, so the *hypothesis shape* is satisfiable, but the bed has no `a` for the theorem to be about and `K[i,j]` depends on `i−j` in a way `exp(C_i − C_j)` can in principle match — see the limit in §8. Nothing in this round licenses the claim. |
| **#2 re-stated**, BED-K (a)/(b) | `0 ≤ m ≤ 1` | as above | **EMPTY, same reason** | The re-statement fixes the value-support hole on BED-M; it does **not** give #2 a subject on BED-K. That is a different debt. |

Checked in Lean on decidable literals (`#eval` output reproduced verbatim in §6):

```lean
theorem bedM_overlap_old_two : bedM.countP satOldTwo = 1 := by decide   -- 0 < a
theorem bedM_overlap_new_two : bedM.countP satNewTwo = 3 := by decide   -- 0 ≤ |a| ≤ 1
theorem bedM_overlap_six     : bedM.countP satSix    = 0 := by decide   -- 0 < a < 1
```

### 2.2 #6 `bounded_gates_stable`

Hypothesis, as proved: the gate is `exp(−softplus w)` for some real `w`. Image:
`(0, 1)` **open at both ends** — `CEQ.V15.bounded_gates_stable` gives `≤ 1` and
`CEQ.V15Phase.softplus_gate_lt_one` closes the upper end to `< 1` strictly.

| | image | bed's support | overlap | verdict |
|---|---|---|---|---|
| **#6**, BED-M (equil.) | `(0,1)` open | `{−1, 0, +1}` | **0 of 3 — EMPTY** | **#6 IS DECORATION ON BED-M.** Not "partial", not "mostly": zero. |
| **#6**, BED-M (propagate) | `(0,1)` open | `{−1, +1}` | **0 of 2 — EMPTY** | Decoration. |
| **#6**, BED-K (a) delay | `(0,1)` open | `K ∈ {0,1}`; no gate sequence | **EMPTY** | **#6 IS DECORATION ON BED-K(a).** |
| **#6**, BED-K (b) power-law | `(0,1)` open | `K` entries in `(0, α] ⊂ (0, ½)`; no gate sequence | **shape-compatible, subject absent** | The entries lie inside `(0,1)`, so a census on values alone would pass. There is still no first-order gate sequence in the bed for #6 to bound — `CEQ.V15.first_order_cannot_delay` is the theorem that says so for (a), and (b) is not a recurrence either. **#6 IS DECORATION ON BED-K(b)**, on the subject test rather than the value test. |

Proved for the whole family at once rather than on three literals:

```lean
theorem six_misses_every_bedM_value (w : ℝ) :
    Real.exp (-CEQ.V15.softplus w) ≠ -1
      ∧ Real.exp (-CEQ.V15.softplus w) ≠ 0
      ∧ Real.exp (-CEQ.V15.softplus w) ≠ 1
```

**Consequence for PART III's identity bind.** BED-M's bind is "oracle gates in
⇒ label `≤ 1e-6`". An arm whose gates are #6-parametrized **cannot be set to
BED-M's oracle gates at all** — not approximately, not at any `w`. The bind is
not a test that #6's family can be given. This is precisely what §X₃₆'s hard
cap exists to fix, and it is a stronger statement than
`V15_X36_LEAN.md`'s `six_never_reaches_the_band`, which covers only the `+1`
end.

### 2.3 #12 `first_order_cannot_delay`

Hypotheses, as proved in `CEQ/V15Kernel.lean`: state dimension one and the
state IS the output; `f : ℝ → ℝ → ℝ` arbitrary; **`1 ≤ d`**; the identity
required for **every** input sequence.

| | hypotheses | bed's support | overlap | verdict |
|---|---|---|---|---|
| **#12**, BED-K (a) delay | `1 ≤ d`, all inputs | `_delay_kernel_matrix` rejects only `d < 0`, so **`d ≥ 0`** — `d = 0` is a legal cell and `build_delay(n, 0, seed)` returns `K = I` (verified by running it) | **PARTIAL: `d ≥ 1` in, `d = 0` out** | The gap is **necessary, not an oversight**: at `d = 0` the delay IS a first-order recurrence. `delay_zero_is_first_order` exhibits `α = 0, β = 1` realizing it, so `1 ≤ d` is load-bearing. Any BED-K(a) cell registered at `d = 0` is outside #12 and must not cite it. |
| **#12**, BED-K (b) power-law | pure delay `y_{i+d} = x_i` | `z_i = Σ_{j<i} ψ_{i−j} b_j`, never a pure delay | **EMPTY** | **#12 IS DECORATION ON BED-K(b).** The scan-blindness of the power-law bed is a separate claim with no theorem in this round. `V15Kernel.lean`'s `linear_first_order_cannot_delay_beyond_state_dim` also speaks only about delay. |
| **#12**, BED-M | pure delay, all inputs | BED-M's label is a first-order chain **by construction** (`equilibrium_oracle`: `z ← a_i z + b_i`) | **EMPTY** | **#12 IS DECORATION ON BED-M** — expected, since #12 is a BED-K theorem, but L-DOM requires it printed rather than assumed. |

A second, already-filed partiality that this census inherits:
`CEQ.V15.first_order_delays_constant_input` shows the per-sequence reading of
#12 is FALSE (a constant drive is delayed exactly by `α=0, β=1` at every `d`).
So #12's overlap with BED-K(a) is `d ≥ 1` **and** the quantifier over inputs —
a single realization of an iid drive is evidence, not the theorem.

### 2.4 #16 `unit_phase_product`

| | hypotheses | bed's support | overlap | verdict |
|---|---|---|---|---|
| **#16**, BED-M | **NONE** (`θ : ℕ → ℝ` arbitrary, `s : Finset ℕ` arbitrary) | phases `{0, π}` from `a = ±1`; the `a = 0` draw **has no phase** | **TOTAL — and vacuously so** | **#16 IS DECORATION ON BED-M, and no overlap count can say it.** |
| **#16**, BED-K (a)/(b) | NONE | all kernel entries real and non-negative, phases `≡ 0` | **TOTAL, vacuously** | **#16 IS DECORATION ON BED-K.** |

**This is a hole in L-DOM as written, and it is worth the round's attention.**
L-DOM says "No overlap ⇒ the theorem is decoration". The converse does not
hold: a theorem with **no hypotheses** has total overlap with every corpus and
constrains nothing. `sixteen_is_silent_on_the_zero_draw` is the witness — #16
reads `1` on the very draw where BED-M's gate is `0` and the path product
annihilates:

```lean
theorem sixteen_is_silent_on_the_zero_draw (θ : ℕ → ℝ) (s : Finset ℕ) :
    Complex.abs (∏ k in s, Complex.exp ((θ k : ℂ) * Complex.I)) = 1
      ∧ pathProd (fun _ => (0 : ℝ)) θ 1 0 = 0
```

`V15_X36_LEAN.md` already refused #16 as content
(`unit_phase_does_not_bound_the_gate`, modulus `285.07` at unit phase). This
adds the L-DOM reading: **a census must have two columns — hypothesis overlap,
and what the theorem constrains on the drawn value.** The first alone passes
#16.

---

## 3. #5a `three_corners_containment`

**The family.** `Hop β g qk i j = exp((C_i − C_j) + q_i·k_j) / Z_i^β` for
`j ≤ i`, `0` above the diagonal, `Z_i = Σ_{j≤i} num i j`. `β` enters through
`Real.rpow`, so `Z^0 = 1` for **every** `Z` including `Z = 0` and no positivity
side condition leaks into the `β = 0` corner.

| corner | setting | named operator | theorem |
|---|---|---|---|
| softmax attention | `β = 1, g ≡ 0, QK-on` | `softmaxAttn qk i j = exp(q·k) / Σ_{j'≤i} exp(q·k)` | `corner_softmax` |
| linear attention | `β = 0, g ≡ 0, QK-on` | `linearAttn qk i j = exp(q·k)`, no normalizer | `corner_linear` |
| the path product | `β = 0, QK-off` | `CEQ.V15.Wc g i j` | `corner_path_product`, bound to `∏ a_k` by `corner_path_product_is_the_gate_product` via `CEQ.V15.prefix_logit_mask` |

```lean
theorem three_corners_containment {g₀ : ℕ → ℝ} (hg : ∀ k, g₀ k = 0) (g : ℕ → ℝ)
    (qk : ℕ → ℕ → ℝ) :
    (∀ i j, Hop 1 g₀ qk i j = softmaxAttn qk i j)
      ∧ (∀ i j, Hop 0 g₀ qk i j = linearAttn qk i j)
      ∧ (∀ i j, Hop 0 g (fun _ _ => 0) i j = CEQ.V15.Wc g i j)
```

**Refused as the content of #5a: a containment whose corners coincide.**
`corners_are_distinct` reads the `β=1` corner at `1/2` and the `β=0` corner at
`1`, at `(i,j) = (1,0)` under the common setting `g ≡ 0, QK-off`. The family
has interior; the three names are not three names for one operator.

**And corner 3 inherits #2's domain defect, unrepaired.**
`path_product_corner_fails_at_a_zero_gate`: wherever a zero magnitude sits on
the path, the real exponential corner cannot equal the corpus's path product,
**for any `g`**. The `β = 0, QK-off` corner of §S-M′ is the corner §1 replaces,
not one this file certifies.

---

## 4. #5b `gate_zero_beta_zero_is_linear_attention` — the refutation

```lean
theorem gate_zero_beta_zero_is_linear_attention {g : ℕ → ℝ} (hg : ∀ k, g k = 0)
    (qk : ℕ → ℕ → ℝ) :
    (∀ i j, Hop 0 g qk i j = linearAttn qk i j)
      ∧ ∃ (qk' : ℕ → ℕ → ℝ) (i j : ℕ), Hop 0 g qk' i j ≠ softmaxAttn qk' i j
```

Both halves: the identification with **linear** attention, and an explicit
separation witness (`qk ≡ 0`, `(i,j) = (1,0)`: `1 ≠ 1/2`).

**The whole-row separation, composed with `V15.lean` as instructed.**

```lean
theorem softmax_row_sum_one (qk) (i) : ∑ j in range (i+1), softmaxAttn qk i j = 1

theorem gate_zero_beta_zero_row_not_one (g) (hg : ∀ k, g k = 0) {i} (hi : 1 ≤ i) :
    ∑ j in range (i + 1), Hop 0 g (fun _ _ => 0) i j ≠ 1
```

The second is `CEQ.V15.gate_zero_not_stochastic` re-used through
`corner_path_product`: at `g ≡ 0, β = 0, QK-off` the hop **is** `CEQ.V15.Wc`,
whose row `i` sums to `i + 1`. Every softmax row sums to `1`. So the `β = 0`
corner is outside the softmax class for **every** `i ≥ 1`, not merely different
at one entry.

**`beta_one_row_is_one` completes the diagnosis.** The same `g ≡ 0` gate at
`β = 1` gives a row that sums to `1`. So **`β`, not `g`, is the switch that
decides membership in the softmax class** — which is the precise sense in which
the contract's original parity clause named the wrong parameter. R11 refuted
that clause three independent ways; this pair is what stops it being
re-claimed by anyone reading `g ≡ 0` as "standard attention".

---

## 5. TRAIN-GATE

`CEQ_V16_CONTRACT.md` PART II: **`#1, #2(re-stated), #3, #5a, #6, #7, #16`
green.** Printed from `V16Domain.lean` itself, so the gate is checked **in the
presence of** the re-statement rather than beside it. `#5b` is printed with
`#5a`; `#2` is printed twice so the pair is diffable.

```
'CEQ.V15.chain_path_product' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.prefix_logit_mask' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.prefix_logit_mask_restated' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.parity_sign' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.three_corners_containment' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.gate_zero_beta_zero_is_linear_attention' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.bounded_gates_stable' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15.scan_assoc' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V15Phase.unit_phase_product' depends on axioms: [propext, Classical.choice, Quot.sound]
```

**Nothing previously green was broken by the re-statement**, and the mechanism
is that no existing file was edited: the re-statement is a new theorem on a
weaker hypothesis, not a replacement of `prefix_logit_mask`. Both are green,
both are printed, and `restatement_subsumes_phase_path_le_one` proves they
agree on the overlap.

The finding that IS a finding, restated once here and once in §1.5: **#6 and
V15Phase's magnitude chain remain green and remain decoration on BED-M.** Green
and applicable are different properties and this round is the one that
separates them.

---

## 6. BUILD OUTPUT AND AXIOM CHECK

### 6.1 Standalone

```
$ cd lean && lake env lean CEQ/V16Domain.lean
EXIT=0
```

`#eval` output — the census counted, verbatim, in file order:

```
1        -- bedM.countP satOldTwo : {−1,0,+1} against #2's `0 < a`        → 1 of 3
3        -- bedM.countP satNewTwo : {−1,0,+1} against re-stated `0 ≤ |a| ≤ 1` → 3 of 3
0        -- bedM.countP satSix    : {−1,0,+1} against #6's image `(0,1)`  → 0 of 3
```

### 6.2 Full build

```
$ cd lean && lake build
EXIT=0
```

No output (warm cache, every target up to date). Because a no-op exit 0 is weak
evidence, all five existing `V15*` files were **force-re-elaborated from
source**:

```
V15          exit=0  error/sorryAx-lines=0
V15Fork      exit=0  error/sorryAx-lines=0
V15Kernel    exit=0  error/sorryAx-lines=0
V15Source    exit=0  error/sorryAx-lines=0
V15Phase     exit=0  error/sorryAx-lines=0
```

### 6.3 Axiom check — all 47 declarations of `V16Domain.lean`

Every one reads `[propext, Classical.choice, Quot.sound]` or less. **No
`sorryAx` anywhere.** `grep -c '\bsorry\b'` over the file returns `1`, and the
single hit is the words "No `sorry`" in the header prose.

```
'CEQ.V16Domain.gateOf' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.pathProd' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.abs_gateOf' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.pathProd_polar' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.pathProd_abs' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.pathProd_eq_zero_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.lean_log_junk_makes_the_scan_form_silently_false' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
'CEQ.V16Domain.no_prefix_scan_represents_a_zero_gate' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.pathProd_eq_Wp' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.restatement_subsumes_phase_path_le_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.prod_eq_one_iff_of_nonneg' depends on axioms: [propext, Quot.sound, Classical.choice]
'CEQ.V16Domain.prefix_logit_mask_restated' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.magOf' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.argOf' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.bedM_gate_exact' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.negative_draw_is_on_the_band' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.bedM' does not depend on any axioms
'CEQ.V16Domain.bedMProp' does not depend on any axioms
'CEQ.V16Domain.satOldTwo' does not depend on any axioms
'CEQ.V16Domain.satNewTwo' depends on axioms: [propext]
'CEQ.V16Domain.satSix' does not depend on any axioms
'CEQ.V16Domain.bedM_overlap_old_two' does not depend on any axioms
'CEQ.V16Domain.bedM_overlap_new_two' depends on axioms: [propext]
'CEQ.V16Domain.bedM_overlap_six' does not depend on any axioms
'CEQ.V16Domain.bedMProp_overlap_old_two' does not depend on any axioms
'CEQ.V16Domain.bedMProp_overlap_new_two' depends on axioms: [propext]
'CEQ.V16Domain.bedMProp_overlap_six' does not depend on any axioms
'CEQ.V16Domain.six_misses_every_bedM_value' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.delay_zero_is_first_order' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.sixteen_is_silent_on_the_zero_draw' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.num' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.Znorm' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.Hop' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.softmaxAttn' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.linearAttn' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.scan_zero_of_zero' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.corner_softmax' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.corner_linear' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.corner_path_product' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.corner_path_product_is_the_gate_product' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.path_product_corner_fails_at_a_zero_gate' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.three_corners_containment' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.corners_are_distinct' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.softmax_row_sum_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.gate_zero_beta_zero_is_linear_attention' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.gate_zero_beta_zero_row_not_one' depends on axioms: [propext, Classical.choice, Quot.sound]
'CEQ.V16Domain.beta_one_row_is_one' depends on axioms: [propext, Classical.choice, Quot.sound]
```

56 `#print axioms` lines total: 47 for this file's declarations, 9 for the
train-gate list of §5.

---

## 7. THE IMPORT LINE

`lean/CEQ.lean` was **not** edited. The coordinator adds, as the seventh V-file
import, after `import CEQ.V15Phase`:

```lean
import CEQ.V16Domain
```

`V16Domain.lean`'s own imports are

```lean
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import CEQ.V15
import CEQ.V15Kernel
import CEQ.V15Phase
```

`Pow.Real` is the only new mathlib dependency in the tree and carries
`Real.rpow`, which is what lets `β` be a real switch with `Z^0 = 1`
unconditionally.

---

## 8. LIMITS

The census reads the generators as they stand at HEAD
`5ed3091b8b2cd00bc5b396b1802cf37412d183c0`; a builder change moves the census
and the tables must be re-run, not re-quoted. BED-M's support was read by
running two builders at one shape (`n=512, s=64, d=24`, seeds `0`) and by
reading the two draw lines; the `{−1,0,+1}` support is a property of those
lines, not a sampling estimate, but the run is a check on the reading and not a
proof about every shape. `Hop` is a single-head, single-channel, real-logit
object: no multi-head coupling, no value contraction, no `β` strictly between
`0` and `1`, and no claim that trained switches land on a corner. #5a proves
containment, not that the family is *exactly* the union of its corners, and it
proves nothing about approximation quality between them. The BED-K rows of
§2.1 record that #2 has **no subject** on that bed; they do **not** settle
whether some `C` reproduces the power-law kernel's `i−j` dependence — that is
open and is not claimed either way here. #12's power-law row records an empty
overlap, not a proof that a first-order recurrence cannot fit the power-law
label; that theorem does not exist in this round. Nothing here trains, and
nothing here says the `{0, π}` phase restriction is differentiable or that the
cap's endpoints are reachable by an optimizer. The `−∞` analysis of §1.3 is
argued in prose over the extended reals; what is *proved* in Lean is the
stronger and convention-free `no_prefix_scan_represents_a_zero_gate` plus the
junk-value witness.
