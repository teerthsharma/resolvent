# Work done — round 3 (CEQ v5), live

Round 1 archived in `DONE_ARCHIVE_ROUND1.md`. Round 2 summarised in
`workdone2.md`. This file tracks round 3 and is updated every iteration until
the loop stops.

**Goal, as restated:** not the best token predictor — the **next-equilibrium
predictor**. An attention module that understands causality and consequences on
Turing-grade problems at the smallest scale.

---

## The round in one line, so far

The contract went to the room before a line was built, which **struck two of four
routes before they cost anything** — and then the first substantive measurement
**struck a constant I had pinned myself**.

---

## Iteration 0 — the contract went to the room first

`/differential-planning` on CEQ v4. Three fellows, questions only.

**Foreman and Chase converged independently** on the question that decided the
round: which route changes the **EVENT**, not the **STATISTIC**?

    common positive rescale (R5, R8)   event changed in      0 / 20000
    ECDF rank transform      (R7)      event changed in    723 / 20000  (3.6%)

**R5 and R8 struck before build.** R8 was scheduled **first**, as the cheapest
one-line change. It is a provable no-op — dividing both sides of an inequality by
the same positive scalar leaves the event set invariant — and this repository had
already proved it once as instrument #16. Building it would have produced a
guaranteed null that read as evidence.

**A published claim withdrawn.** Foreman demanded the interval on the comparison
that declared pivot routing dead. Bootstrap, B=20000:

    pivot − dense = −0.2099,  95% CI [−0.7497, +0.2651]   — does not exclude zero

*"Routing makes it worse than dense"* is unsupported and had appeared in every
document since round 2. Pivot routing stays dead — both arms sit far past the
−0.3 bar — but that sentence does not.

**Foreman killed the twin test.** In `ã = sign(t)·F(|t|)^γ` the sign is a
*multiplicative prefactor*, so sign-sensitivity is a property of the algebra, not
the aggregation: `sign(t)·1` — pure sign, all magnitude destroyed, plainly
useless — passes the identical gate. Rebuilt with a third **NOT-VACUOUS** arm in
which that control must be seen losing.

**All three demanded the same reordering, conceded:** M3 on the windowed arm
moves to Phase 0. F4's w=8 arm is already flat to s=2048 — the exact property
four routes chase — and has never been capability-tested. It is a live
counterexample to the sufficiency of the gate, sitting inside the contract's own
facts table.

---

## Iterations 1–3 — the capability harness was measuring the wrong operator

**Iteration 1.** `scale/m3_capability.py` — the harness whose numbers decide
everything — resolved its only signed arm to `_causal_tgate_operator`, which
ships nowhere. Round 2 caught that failure mode and bound it, but the bind
resolves arms from `pivot_probe.py::build_arm` and **never covered the capability
harness**. So round 2's M3 readings — softmax 1.855584, pivot_signed 1.342215,
pivot_unsigned 1.956147 — are instrument-#17 readings.

The lesson is not "we missed one." **A bind covers the call site it names and
nothing else.** That one was written against a *module*; it needed to be written
against the *question*.

**Iteration 2.** Moved onto shipped `sgate`. The tgate-only `g[s] + tau`
parameters went with it — and that fixed a second problem nobody was tracking.
`HIDDEN = 128` had been sized specifically to keep those extras under a 10%
parameter-match bar (a comment records `HIDDEN = 32` breaking it at 12.05%).
All arms now report **n_params = 4769 exactly**: matched params became structural
rather than tuned.

**Iteration 3.** Added `windowed_signed` — the shipped sgate with `window=8`,
one argument rather than new operator code, because a separate operator would
mean F4's flatness and F4's capability were measured on two different objects.
Bound by four value assertions, including that hop-2 reaches exactly `2w` **and
carries nonzero mass in the `w..2w` ring** — a windowed arm whose second hop
bought no reach would pass every structural check while being a one-hop arm under
a multi-hop name.

---

## Iteration 4–5 — the reading was held, and the audit was clean

The first capability reading was **held** rather than taken: four house-mode
agents were auditing that harness, and one hazard would have made the reading
meaningless rather than merely wrong — at s=64 with w=8, hop-2 reaches 16
positions, so if the task's flipper-to-payload distance exceeds that, the
windowed arm cannot see the flipper and its result is arithmetic.

`python inspector.py 5` → exit 0. **8 checks, 8 must-fire controls all fired.**
The published-number rotation landed on M2's two-point slope — the figure the
entire kill rests on — and it reproduced at **−1.2977**.

---

## Iteration 6 — `A_8 = 2.187500` IS STRUCK. The signs do not cancel.

**This corrects a constant this project pinned, and it was pinned by me.**

`A_k = E|Σ ε_p|` was enumerated over `2^k` sign patterns **assuming independent
symmetric signs**. Under top-k salience selection the survivors are
**concomitants of order statistics** — the value attached to a token chosen for
its magnitude — and independence is an assumption, not a fact. The governing
prompt already said *"sign-independence is TESTED, not assumed"*; it was written
and then not honoured.

**Calibrated at both ends before being believed.** A plain Gaussian row must read
independent, and a planted correlation must be detected:

| control | E\|Σε\| | vs A_8 | mean pair corr |
|---|---|---|---|
| independent | 2.1830 | 0.998× | −0.0011 |
| 30% aligned | 2.5080 | 1.147× | +0.0377 |
| all aligned | 8.0000 | 3.657× | +1.0000 |

**Measured, 4000 draws per cell, k=8:**

| operator | s | E\|Σε\| | vs A_8 | P(+) | pair corr |
|---|---|---|---|---|---|
| gaussian | 16 | 2.2185 | 1.014× | 0.4983 | +0.0033 |
| gaussian | 128 | 2.1695 | 0.992× | 0.4960 | −0.0020 |
| gaussian | 512 | 2.1990 | 1.005× | 0.4993 | +0.0009 |
| **sgate** | 16 | **7.7065** | **3.52×** | 0.9817 | **+0.9284** |
| **sgate** | 128 | **7.9850** | **3.65×** | 0.9991 | **+0.9963** |
| **sgate** | 512 | **7.9970** | **3.66×** | 0.9998 | **+0.9993** |

The Gaussian control reads independence exactly where theory requires it, so the
probe is not broken. **`sgate` reads 7.997 against the all-aligned control's
8.000.** At s=512 the selected signs are **99.98% positive** with pairwise
correlation **0.9993**. They do not cancel at all.

**Why.** `sgate` is `A = ρ(softmax(w) − λ·softmax(−w))/(1+λ)` at `λ = 0.10`. The
negative half is scaled down tenfold, so the largest-magnitude entries are the
large *positive* ones, and selecting the top-k by `|A|` selects them. The
operator is **signed in name while its large entries are essentially all one
sign** — which is precisely the integrity failure the arsenal's frustration /
switching-class item exists to catch, and it had never been run on this operator.

**What it costs, and what it does not.**

- `A_8 = 2.187500` is **struck as a certificate for sgate**. The measured
  background is `≈ k = 8`, not `√k ≈ 2.19` — **3.66× larger**, so the perturbed
  token competes against roughly 8, not 2.19.
- The boundedness claim **survives unconditionally**: `|B_k| ≤ k` is sign-free,
  so only the constant moves — which is exactly what the pre-registration said
  would happen if dependence were found.
- The *s*-dependence gets **flatter**, not steeper: measured `E|Σε|` grows
  7.7065 → 7.9970 over s = 16 → 512, a ratio of **1.038**, against the **1.357**
  the independent model predicts. Sign alignment saturates.

So the finding cuts both ways and must be reported as both: the absolute flip
rate is worse than the independent model implies, and the slope — the thing M2″
actually measures — is better. Neither half may be quoted without the other.

---

## Iteration 8 — F1 is scoped to a stack nobody ships

Foreman resolved the contradiction iteration 0 left in Open. It was never one
contradiction — **both sides were measuring objects that do not correspond.**

**The benchmark ran a different operator.** `ceq/capability.py` never sets
`lm.RHO / SGATE_LAM / HOPS`, so COGS trained the pre-campaign globals
`(0.9, 1.0, 3)` rather than the parity point `(1.5, 0.10, 2)`;
`max|A_run − A_parity| = 1.2273`. At `λ = 1` both softmax halves sum to 1, so
`A = ρ(p⁺ − p⁻)/2` has **row sum exactly 0**, and row 1 is identically zero.

| | mean row-L1 | ‖Av‖/‖v‖ |
|---|---|---|
| as-run `sgate(0.9, 1.0)` | 0.210988 | **0.054957** |
| parity `sgate(1.5, 0.10)` | 1.220934 | 0.517922 |
| softmax | 1.000000 | 0.545308 |

The benchmarked signed arm **mixed 9.9× less** than the softmax arm it was
parameter-matched against, and its hop-2/hop-3 masses were `4.04e-03` and
`3.22e-04` — the multi-hop path sum this campaign exists to test contributed
**0.4% and 0.03%** of the signal in the arm that was benchmarked.

**The theorem's zero is measured with the MLP deleted.** Verified directly:

```
softmax       depth=1  0.0          softmax_gelu  depth=1  0.0
softmax       depth=2  0.0          softmax_gelu  depth=2  0.0546875
sgate         depth=1  0.1484375    sgate         depth=2  0.09375
```

Plain softmax reads 0.0 at **both** depths, so the theorem is about the operator
rather than shallowness — that part of F1 is sound. But `ceq/lm.py:214` puts
`nn.Linear → nn.GELU → nn.Linear` in every block and trains four of them, and
with that nonlinearity softmax reads **0.0546875**.

**Like-for-like at depth 2 the ratio is 1.71×, not exclusivity** — and `sgate`
gets *worse* with depth while `softmax_gelu` gets better, so the gap closes from
both sides. The `0.0546875` has been asserted in this repository since round 1
at `tests/cameron/test_parity_is_the_wrong_target.py:61` and was never carried
into the capability framing.

**Sixth appearance of one shape** — a correct statement about an object other
than the one that ships: instrument #17, M4's kill, M2's vacuous clause, M5's
hypothesis, the random-init scope, and now F1.

What survives: the theorem is true and M1 stays GREEN as a precondition. What
does not survive is *"softmax cannot do this at all."* The honest number is
**1.71× at matched depth**, and it travels with the claim from here on.

---

*Updated through round 3, iteration 8. Four house-mode agents still running;
their findings and the representation-theorem challenge reconcile into a
prognosis when they land.*
