# done5 — Round 5, CEQ v7, the two-spheres round

**Verdict: `TWOSPHERES: BROKEN` — ARM A, K1's dual slope, displacement clause.**

Closed at iteration 23 of 30. The loop stopped because the promise became true,
not because it ran out. Round 1 archived in `DONE_ARCHIVE_ROUND1.md`; rounds 2–4
in `workdone2.md`, `done3.md`, `done4.md`. The full negative result with its
numbers is `D1.md`; the iteration-by-iteration log is `DONE.md`.

---

## 1. What the round proposed, and the one sentence that killed it

Measure a token's consequence as an **angle** on the unit sphere. The square-root
map `φ(p) = √p` carries the simplex isometrically onto the positive orthant, and
by Čencov the Fisher–Rao metric is the unique invariant choice, so displacement
would be **scale-free by radius rather than by tuning**.

It fails for a reason that is arithmetic, not empirical. Masking one token and
renormalising leaves the row a **single degree of freedom**: with `p = A^c[i,c]`,

    A^0[i,j] = A^c[i,j] / (1 − p)     BC_i = √(1 − p)     TV_i = p

    θ_i = arcsin(√(TV_i))      exactly, row by row

Measured three times independently — float32 residual **8.457280e-04**, float64
**6.828570e-08**, total-variation residual **2.980e-07**. The four-order collapse
from single to double precision is the signature of an exact identity read
through an ill-conditioned `arccos` near argument 1.

**The angle carries no information total variation does not.** Every
ablate-and-measure probe on this geometry compares two aggregations of one
number.

---

## 2. The kill that fired

**K1, dual slope, displacement clause.** 400 draws per cell, six pivot counts,
bucketed, determinism replay matching each bucket.

| statistic | slope | 95% CI | vs the −0.30 trigger | vs K1's bar ≥ −0.10 |
|---|---|---|---|---|
| as computed | **−0.4137** | [−0.4579, −0.3704] | excludes, CI below | not met |
| live rows only | **−0.4654** | [−0.5173, −0.4160] | excludes, CI below | not met |

Both intervals lie entirely below the pre-registered *"displacement dies with
flip ⇒ no leap"* line. **ARM A did not survive, so ARM B was never authorized and
was never built** — the contract's `build only if ARM A survives` was honoured,
not relaxed. `TWOSPHERES: KEPT` therefore fails on two independent grounds.

**Caveat that travels with this number:** the slope rests on a **journal replay,
not a fresh derivation**. The driver reported every unit already journalled and
recomputed none. The replay matched bitwise, so the journal is intact, but nobody
re-derived that slope from draws this pass.

---

## 3. Binding facts for the next contract — do not relitigate

**F-identity.** `θ = arcsin(√TV)` exactly on a one-token mask. Three independent
measurements. **This is a constraint on probe design, not a tool.** A round that
wants a geometry to earn something must change the **probe**, not the metric.

**F-curvature.** The sphere's one distinguishing feature is measured as a
**liability**. The chord beats the geodesic by **0.86% / 0.65% / 1.25%**, and
`sin θ = √TV` beats it by **3.97% / 2.87% / 5.70%**, on K3's own criterion at
every pivot count. Both tangent-plane readings win.

**F-K3-void.** K3 as written cannot distinguish geometry from row-wise concavity.
θ beats **raw** TV at every k (`+0.0157 / +0.0282 / +0.0387`, CIs excluding zero)
and **loses at k=8 to a held-out `TV^p`** (`−0.0643 [−0.0981, −0.0159]`, fit and
score on disjoint halves). Beating raw TV is evidence of a square root.

**F-selector.** `select_pivots` ranks by `key.norm(dim=-1)`
(`scale/pivot_probe.py:88`) — a pure function of a token's own representation,
with no reference to any downstream effect, and nothing in an ARM A draw is
causal. **Every causal-vs-filler contrast in this project is confounded by
construction until a key-norm-matched filler is applied.** The published filler
pool is at least **two populations**: band-vs-tail separates at **1.0716 / 0.8782
/ 0.4868**, all CIs excluding zero.

**F-aggregator.** `max_i θ_i = arcsin(√(max_i A^c[i,c]))` — a monotone read of
the largest attention weight any row places on the masked token, AUC gap **~1e-03**
against that raw quantity. Real as a comparison (`+1.1347 [+0.7833, +1.5877]`
over the mean) and **largely confounded as a result**: retains **2.9% / 13.3% /
58.9%** against a key-norm-matched filler, where the mean it replaced retained
**35%**. **More confounded than the statistic it was meant to improve on.**

**F-equilibrium.** X6 survives its own kill — the glance is **not** the fixed
point (residual `0.599101 / 0.388587 / 0.321843`, 100% converged in 28–53 steps).
But `τ = 0` has a **one-pass closed form**: `m = x̄/‖x̄‖` reads `2.454507e-16 /
9.675157e-16 / 5.176001e-15`. The contract defined equilibrium twice and those
were **different points**; the repair is in `LOOP_PROMPT.md` — the certificate is
the **Karcher residual**, and `τ` is a **displacement statistic**, never an
equilibrium test.

**F-uniqueness.** The Karcher mean is unique only inside the injectivity radius,
and that holds on **0.9333 / 0.8167 / 0.5167** of draws at k = 8 / 32 / 128. **At
k=128 the settled reading is not well defined on 48.3% of draws**, and settling
slows (28.43 → 52.55 steps). Both trends run against more pivots.

**F-B1.** Not a rename of the key-norm (`ρ = −0.0253`, top-k overlap below chance,
reproduced twice) — **and ill-posed as written**. `ξ` is defined relative to a
chosen masked token while the selector is documented to use *"ONLY CONTENT —
never `c`, never `i`, `j`"*. Its top-k retains **5.4% / 19.6% / 63.2%** under a
change of `c` on the same draw. Also: `‖ξ_p‖ = 0` for **68.04%** of candidates by
causality — row `p` moves only if it attended to `c`, which needs `p > c`.

**F-lam (corrects round 4's F16).** `lam` is **not** a threshold at 1.0. At ARM
A's geometry `_causal_sgate_operator` is signed at **every** lam tested including
0.10 — 30/30 readings, `min A = −1.363636e-01`. The cause is **logit scale**: mean
causal `|w|` is `1.171e+01` / `1.320e+01` against the harness's `2.682399e-03`.
F16 was a true reading of one geometry generalised into a property of the
operator.

**F-basis.** The additive-basis arm's third birth gate **failed on a measurement,
not a theorem** — off-schedule 0/384 flips, but the cell is **live**: 109/384
draws differ, max separation `1.505102e-02`. Separately, the **coverage** result
stands and is worth keeping, **with its provenance split because the two halves
are not equally established**:

  * `[1,56]` needs **minimum k = 12, proven**, and it is reproducible from the
    committed log `scale/arm_a_rebuild.txt` — k=10 refuted in 2,912 nodes, k=11
    in 407,051, witness at k=12 in 934,038.
  * `[1,127]` has a **k = 18 witness**, but **minimality below 18 is NOT
    reproducible from this repository.** The in-repo search **capped at k=16**
    (600,000 nodes) and its author explicitly carried the value rather than
    claiming to have derived it. A separate exhaustive refutation of k=17 was
    reported, but its scripts lived in a scratchpad outside the repo and were
    never committed. **Treat k=18 as a witness with an unverified lower bound.**
  * The **unreachable fraction driven to exactly 0.0000** at the covering
    schedules is in the committed log.

**F-deadrow.** `scale/torque_probe.py::theta_rows` contradicts its own docstring —
it promises 0 for a row with no mass either side and returns `arccos(0) = π/2`.
Row 0 has no visible key under causal masking, so every draw carries a constant
**0.0015340 rad**: 5.0–11.6% of the causal reading and **46.2–55.1% of the filler
reading**. Refinement: rows reading exactly π/2 number **2, 3 or 4** in some
cells and **those extras are live**, so detecting dead rows by `θ == π/2`
over-subtracts by up to three rows.

---

## 4. What is genuinely open, in order of worth

1. **K1's sign-flip clause is the only pre-registered kill left undecided.**
   8 events in 2400 draws, `0.003333` with exact interval `[0.001440, 0.006557]`.
   Its slope CI `[−1.0000, +0.0000]` is **count discreteness, not an interval**,
   and the unsigned arm's exact interval `[0, 0.010195]` **contains** the signed
   arm's rate — the two arms are not separated at these sample sizes even though
   one is zero by theorem. Needs the contract's own **20,000 draws per cell**,
   about eight hours at the measured per-draw cost. Raising `lam` does not buy
   flip events (4/300, 5/300, 5/300 at lam 0.10/0.50/1.00).

2. **What peak attention tracks, once the key-norm confound is removed, is not
   established.** Retention is 2.9% / 13.3% / 58.9%, and the match is by **rank,
   not value** (band/causal key-norm 0.9120 / 0.8900 / 0.8347), so those are
   **upper** bounds. Only the k=128 residual has room in it.

3. **A fresh derivation of the K1 slope from draws.** See the caveat in §2.

4. **The provenance bind is left RED, deliberately.** It now fails listing nine
   shipped cost figures with no run evidence anywhere in `DONE*.md` — all of them
   present in `DONE_ARCHIVE_ROUND1.md`. The repair is to re-point at the archive,
   **not** to paste numbers into the log, which would manufacture the defect the
   strike names.

---

## 5. Instrument ledger

**Ten defects introduced by this round's own work and caught by its own output.**
The four gate failures share a shape worth carrying forward:

| iteration | the gate measured | what was pre-registered |
|---|---|---|
| 16 | bulk rank stability | a **top-k** overlap, since selection takes a top-k |
| 17 | `argmax` agreement | an identity about **values** |
| 18 | half of a two-part condition | interval **and** ratio |
| 21 | a helper the control could not reach | the filter, which lived elsewhere |

**Each measured something adjacent to the pre-registration, and each erred toward
the flattering reading.** All four were caught by the probes' own output.

Six earlier ones: a tolerance below the float32 floor so `converged` could never
be true; a step cap averaged into a convergence time; a control returning `True`
unconditionally; `c = i` chosen as an "inert" control when it is the one position
guaranteed to move the row; an unbound draw loop whose disagreement was reported
as a contradiction; and a `k`-independent statistic printed as three rows.

**Three struck by the log audit — 37 claims checked.** Two were mine: a residual
figure asserted `[RUN]` with **no live producer** (it lived only in a code comment
and in prose; now in the `STRUCK` registry with its provenance), and a sentence
claiming a fellow's two figures failed to reproduce when both reproduce to four
decimals — the two agents had measured different quantities at different
geometries.

**Four found by other agents.** The `theta_rows` docstring defect; the health
check's clean bill covering **107 of 1278 tests, 8.37%** (it now prints its own
coverage, with a must-fire control that the fraction is measured rather than
assumed); the self-satisfying provenance bind; and `scale/valuation.py`'s
docstring arguing its underflow defect at the wrong dynamic range — the defect is
real but needs ~1e-200, and at 1e-30 the float path is correct because the
function takes a Python float64.

---

## 6. What the process cost, and what worked

**The escalation chain, set mid-round:** fellows (Chase, Cameron, Foreman) →
Wilson → Health Inspector → Dr House. Three fellows dispatched **in one message**
on the **same question** from different stances, each test-bound.

**It worked, and the evidence is that it overruled almost everyone.** Wilson
killed three fellow claims and one of his own predictions. The Inspector struck
three claims including two of mine. My own matched-filler control largely
dissolved the round's most promising finding, which was Cameron's.

**What did not work, for six iterations:** agents were dispatched **serially, one
at a time**, and each report written up on its own. That is sequential
delegation, not a differential — no parallel tier, no reconciliation, no
prognosis. The same failure had been flagged twice before and patched at the
symptom.

**Dr House was never released, and that was a decision.** The trigger requires the
cause of death to be **missing innovation**. Every kill this round fired from a
measurement with an interval, and the audit re-ran the probes that produced them.
Releasing him would have been using a leap to argue with a measurement.

---

## 7. State of the deliverable

`results/arm_a.jsonl` is byte-identical to its committed version — **G2 intact**,
no published number moved across 23 iterations and five concurrent agents.

Shipped this round: `D1.md` (the negative result, with acceptance criteria that
did not previously exist anywhere); the coverage line in `inspector.py`; the
struck-constant scan extended to cover `D1.md` and de-duplicated; the provenance
bind repaired at the class; and twelve probe files with eight journals committed
so every figure quoted in `D1.md` can be re-derived from the repository rather
than from a transcript.

**The terminal deliverable is unchanged and unmet:** a module trainable in Colab
and uploaded to HuggingFace, with a capability table. Round 5 produced no
capability. What it produced is a constraint — the identity — that rules out a
family of probe designs, and an instrument ledger that is now three rounds deep.
