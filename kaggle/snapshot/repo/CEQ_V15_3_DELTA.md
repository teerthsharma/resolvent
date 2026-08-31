# CEQ v15.2 → v15.3 — X₃₆ PHASE GATES (the leap), X₃₇ TOPOLOGICAL CERTIFICATES

Filed verbatim from the author's message, 2026-08-31, immediately after R11
closed. Extends `CEQ_V15_2_DELTA.md`. All three deltas stay verbatim-of-record
under RUL-5.

**This delta is aimed at a measurement, not at a hunch.** R11's R1 returned
5 of 8 seeds below `floor₁` and 3 divergent, with `â_max` of `20.31`, `49.66` and
`285.07` — above 1, where the amended operator's `1/(1−a)` is undefined. X₃₆
makes `|a| ≤ 1` hold **by construction**.

---

## X₃₆ — THE GATE IS A WAVE

`a_i = m_i · e^{iθ_i}`

- **magnitude** `m_i ∈ [0,1]` **CLOSED** — the band mask via the even feature
  (`R² = 1.000000`) with a **hard cap**, so `0` and `1` are attainable values,
  not limits.
- **phase** `θ_i` on the circle. BED-M's `±1` gates are `θ ∈ {0, π}`.
  **Nothing to chase, nowhere to fall.**

**The prefix construction becomes prefix-PHASE.** `C_i` complex
`= Σ log m + i Σ θ`. Then `|path product| ≤ 1` **by construction**, and `= 1`
exactly on the band. `[RUN: 1e4 phases → modulus 1.000000000000]`

**Parity mask = Z₂ winding** (identity, `[RUN: True]`).

### PRIOR ART — state before naming

- unitary / orthogonal RNNs (Arjovsky 2016)
- **LRU (Orvieto 2023)** — `λ = exp(−exp ν + iθ)`, **OPEN magnitude**
- complex diagonal S4 / Mamba eigenvalues near the circle
- RoPE (rotation = norm 1)

**The novelty claim, stated at its actual size by the author:** *"our closed `m`
is the small delta, and it is exactly the delta R1's three divergent seeds paid
for."* A closed magnitude against LRU's open one is a small difference that a
measurement has already priced.

### PRE-REGISTERED — the R1 re-run

Same cell, phase gates, `N = 8`:

> 8/8 converge · `â_max ≤ 1` **by construction, not by luck** · the interval no
> longer straddles · the 5/8 crossing **banks** (`+12`).

### COUNTER-PREDICTION, filed by the author against himself

Under the same-sign law from R11's round entry — *nine contract statements
checked, six wrong, all six over-crediting the project*:

> **"Phase gates converge 8/8 but the crossing shrinks to ≤ 3/8, because the
> magnitude cap removes gain the divergent seeds were exploiting."**
>
> If that is what the table shows, **the amendment bought stability at the price
> of capability, stated.**

Both predictions are filed before the data. **Both are scored.** This is the
first time in the campaign a contract has carried a competing prediction against
its own optimism, and it is the correct response to the same-sign finding.

---

## X₃₇ — TOPOLOGICAL CERTIFICATES

**(a) `Z` winding** of the learned phase sequence, printed **per instance**.
Integer. **A non-integer reading is an instrument defect**, not a result.

**(b) Persistent `β₁`** of the carrier trajectory via the author's Rips toolkit.
**Recurrent node iff `β₁ ≥ 1`**, with RPS / coordination must-fires
`[RUN: 1 vs 0]`.

**(c) Euler–Poincaré.** `Σ(−1)^k β_k` from the toolkit **must equal** the
Poincaré–Hopf index sum of the equilibrium census. **Mismatch ⇒ census defect.**

Berry phase `[U]` as the continuous frame. **Kuramoto** (verified earlier) as the
coupling law for multi-channel phase gates — *"waves coupling around a point"* is
phase locking, and the order parameter `r` is printed.

---

## LEAN #16 `[M]`

`unit_phase_product` — `|Π e^{iθ_k}| = 1`.

Trivial, and **it replaces #6's bound with an equality on the band.** `#6`
(`bounded_gates_stable`) gives `∏ ≤ 1`; on the band `m ≡ 1` this becomes `= 1`,
which is what removes the divergence R1 measured.

---

## WHAT THIS INHERITS FROM R11, AND MUST NOT REPEAT

- **L-EQ.** The four prior-art items above are named, not yet `[V-eq]`. Each
  needs its equation with hypotheses and one numeric instance run before it can
  be cited in a claim.
- **V-24.** Any identity bind for the phase arm must have a demonstrated
  rejection region — planted mutilations that fail at `O(1)`. A bind the answer
  key passes is not a bind.
- **M-18 and its own correction.** R1 showed that M-18's *prescribed replacement*
  diagnostic was itself non-discriminating, sign flipped. **Any diagnostic
  registered for the re-run must be run on the corpus alone, and against a
  zero-step control, before it is trusted.**
- **M-20.** State each prediction **once**. This delta already carries two
  predictions for the same cell — they are a *prediction and a counter*, which is
  legitimate, and both are named as such and both are scored. That is not the
  M-20 defect, which was two registrations each claiming to be the prediction.
