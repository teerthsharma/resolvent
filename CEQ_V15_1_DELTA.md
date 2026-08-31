# CEQ v15 → v15.1 — CONTRACT DELTA

Filed verbatim from the author's message of 2026-08-31, mid-round at R11 it.11.
Amends `CEQ_V15_CONTRACT.md`, which stays verbatim-of-record under RUL-5. Where
this delta and v15 disagree, this delta is the later authored text and governs.

---

## STANDING PERMISSION ADDED — L-AMEND

> "don't just take a loss sometimes if something can be fixed with editing the
> formula u are allowed to do that if it comes closer to North Star"

**L-AMEND.** A refuted clause may be REPAIRED by amending its formula, not only
recorded as a loss, provided the amendment moves toward the north star and the
amendment is stated as an amendment — with the original clause, the refutation,
the new formula, and what the new formula gives up.

This is a change of disposition, not of standard. An amended formula still needs
its identity theorems before it trains (L-LEAN), its equations before it is
named (L-PRIOR, L-EQ), and its bind must still carry information — an amendment
that restores a clause by making its bind vacuous is a worse outcome than the
refutation it replaces.

**First application.** The v15 PARITY clause is refuted: §S-M's unnormalized hop
has row sums `i + 1` at `g ≡ 0`, never `1` (`lean/CEQ/V15.lean`,
`gate_zero_not_stochastic`). Under L-AMEND the round no longer stops at that
refutation. The additive-logit form `logit_ij = q_ij + (C_i − C_j)` is proved to
restore the bind (`gate_zero_logit_identity`), and the open question — whether
one operator can carry both that bind and exact path-product label reproduction
— becomes a design question rather than an adjudication.

---

## X₃₅ — RESIDUAL INFERENCE OF HIDDEN CAUSES

### X₃₅a THE INSTRUMENT

`r = z_obs − z_model(visible)`. Onset by a **MEMORYLESS comparator on residual
energy** (Shewhart).

**Why memoryless and not CUSUM.** CUSUM's dependence inflation killed X₂₆; the
memoryless detector was the one that survived at matched `ARL₀`. This is a
settled result of the campaign and is not to be relitigated by a node that finds
CUSUM more sensitive in isolation — the comparison that matters is at matched
`ARL₀` on dependent data.

Threshold calibrated on **no-plant runs**. Report `(onset, magnitude, downstream
gate estimates)`.

**MUST-FIRES** — on production-path residuals (L-SCOPE), not on a reimplementation:

1. **Planted hidden cause ⇒ onset within ±1.** `[RUN: exact at 0 noise, 36 vs 37
   at sd 0.05]`.
2. **NO plant ⇒ no onset called** at the calibrated false-alarm rate.

**The second half is the SCOPE control and it is the one that decides the
instrument.** A bad model's approximation error masquerades as a hidden cause
everywhere, so the no-plant residual must be flat or every onset is void.

### X₃₅b THE ARCHITECTURE STEP

S2 posits a **latent node** at a called onset; S-M re-propagates with it.

New capability column: **LATENT LOCALIZATION ACCURACY**, on beds with planted
hidden causes. The bed variant costs one extra field in the generator.

**What this makes measurable.** The New York demo becomes a measurement rather
than a demo: the "travel" is a **latent node the model must place**, not a token
it must emit. That is the north star's "predicts the NEXT STATE toward
equilibrium" with a number attached — which basin, which transition, and now
also *what unobserved cause moved it*.

### X₃₅c PRIOR ART `[U, before names]`

Fetched at equation level **before** anything is named:

- latent-variable inference `[V]`
- hidden-confounder discovery (FCI-class algorithms)
- residual / anomaly-detection lineages
- missing-mass inference in astronomy — **as the cited motivation only**, not as
  a method claim

---

## X₈′ — MÖBIUS INVERSION (X₈ FORMALIZED)

The four-point estimator **is** the Möbius inversion of `f` on the Boolean
lattice `[RUN: 0.600]`. The general `k`-order interaction is `μ`-weighted
inclusion–exclusion over subsets (Rota `[U]`).

- **Must-fire:** additive `f ⇒ 0`.
- **Positive control:** epistasis (Lean #9).
- **Renamed in code and docs.**

---

## KILLS

- **No-plant residual fails flatness ⇒ X₃₅ VOID** until the visible model is good
  enough that its residual means something. *The detector cannot outrun the model
  it subtracts.*
- **Localization below the pre-set bar on planted beds ⇒ the residual is
  uninformative at that noise**, stated **with the noise level**.

---

## CONSEQUENCE FOR THE ROUND'S ORDER

X₃₅'s first kill is a precondition, not a result: it requires a visible model
whose residual is flat under no plant. No arm is trained yet (L-LEAN), so the
instrument is built and calibrated against a **known** visible model — the
oracle — where "flat under no plant" is checkable exactly. The instrument
therefore lands before the arm, and its bar is set before any arm's residual is
ever read, which is the correct order and the one L-FIRST asks for.
