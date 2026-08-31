# M2′ R2 — what each outcome will mean, written BEFORE the run

**Written 2026-08-25, loop iteration 3, with `scale/r2_units.py` under
construction and no R2 number yet in existence.** Nothing here may be edited
after the first real R2 number lands — that is the entire point of writing it
now.

The predecessor file, `M2_PREREGISTERED_READING.md`, earned itself in a single
iteration: it predicted that S2's unsigned arm would come out structurally zero
and be misread as "signedness wins", and it stopped exactly that. R2 has a
sharper trap than S2 had, so it gets the same treatment.

---

## R2 as written

> **R2 sign-determinacy** (breaks "magnitude statistic") — enforce a
> sign-nonsingular pattern on the k×k pivot block; test = magnitude-
> randomization invariance, 10⁴ resamples, sign pattern bitwise constant.
> **Kill:** determined fraction ~0 in trained blocks, or constraint destroys
> training.

The idea: if the sign of $(I-A)^{-1}_{ij}$ is fixed by the sign **pattern** of A
alone, no magnitude can change it — so context cannot dilute it by
combinatorics rather than by measurement. Brualdi & Shader, *Matrices of
Sign-Solvable Linear Systems*, Cambridge Tracts in Mathematics 116, CUP **1995**
[VERIFIED by direct fetch]. A square sign pattern is **sign-nonsingular (SNS)**
if every matrix in its qualitative class is nonsingular.

---

## THE TRAP, fixed in advance

**Sign-determinacy is insensitivity to magnitude. The property M2 measured is a
third token FLIPPING a sign. These pull in opposite directions.**

A fully sign-determined operator **cannot flip at all** — its sign is fixed by
the pattern, and intervening on token $c$ changes magnitudes. Pushed to its
limit, R2 drives the flip rate to **zero**, which is softmax's number and the
death of the only property this project has.

So the naive reading — "determined fraction high ⇒ R2 GREEN ⇒ context cannot
dilute it" — **is wrong**, and it is wrong in the direction that feels like
winning. That is the shape of the multizoom `c = s//2` artifact, the dfloor
`min(4, nblk)` ceiling, and S2's vacuous unsigned arm: a number that is what it
is by construction, dressed as a result.

**Therefore R2 is not GREEN on a determined fraction alone. It is GREEN only if
the determined fraction is high AND the flip rate survives.** Both, measured in
the same run, in the same table. Any R2 report that shows one without the other
is incomplete and must be read as incomplete.

---

## THE OUTCOMES, fixed before the run

**A. Determined fraction ≈ 0.**
→ The sign pattern determines nothing; there is no magnitude-independent
structure to stand on. **R2 RED by its own kill.** Owe a replacement per the
standing order — the next-cheapest route is R4 (a sweep), and the named
alternative in `ARSENAL.md` is **path coherence** (below).

**B. Determined fraction high AND flip rate collapses toward softmax's zero.**
→ **R2 RED, and it is the informative death.** It means sign-determinacy and
sign-sensitivity are genuinely incompatible for this operator class, which is a
real result about the design space and should be written up as one, not buried.
Replacement is **path coherence**, which is the weaker and correct condition:
not "the sign is determined regardless of magnitude", but **"the $j\to i$ paths
through $c$ SHARE a sign, so $c$'s contribution adds instead of cancelling"**.
That attacks the measured death mechanism (the $k{=}2$ term averaging over
intermediates) without demanding magnitude-independence, and it is computable on
the $k\times k$ pivot block where the global version never was.

**C. Determined fraction high AND flip rate holds.**
→ **R2 GREEN.** This is the outcome that would matter. Even then it is a
**statistic, not a capability** — `CHECKLIST.md`'s preamble is explicit, and M3
is UNTESTED. A GREEN R2 buys the right to run M3, nothing more.

**D. Determined fraction is high but so is softmax's.**
→ **VACUOUS.** Softmax's pattern is all-positive, so its determined fraction
should be **1.0 by construction**. That is the calibration, not a comparison —
see below.

---

## CALIBRATION — both ends, before any R2 number is believed

Per standing doctrine: *calibrate every instrument against a case where it must
fire and one where it must not.*

- **Must read 1.0:** softmax. An all-positive pattern is sign-determined
  trivially. If the instrument does not return 1.0 for softmax, **it is broken**
  and every R2 number from it is void.
- **Must read low:** a random sign pattern at the same density. If a random
  pattern also reads high, the measurement is not detecting structure.
- **The kill must be able to fire in both directions.** Exhibit an input giving
  ≈0 and one giving ≈1 before trusting any value between. A kill clause that
  cannot fire is instrument #15 repeating.

---

## THE DEFECT IN R2 AS WRITTEN — stated now, not after

R2's kill says **"determined fraction ~0 in TRAINED blocks"**.

**This project has no trained pivot blocks.** Nothing has trained above 3.65M
parameters, the pivot operator has never been trained at all, and every M2/S2
number on record is measured on **random projections**. So R2's kill, read
literally, cannot be evaluated today — the same class of defect as M2's clause
2, which is why M2 is superseded.

Two honest options, and the choice must be recorded when it is made:

1. Run R2 on **random projections**, state plainly that the kill's "trained"
   qualifier is unmet, and treat the result as a screening number that cannot
   close M2′ on its own.
2. Train a small pivot block first — which pulls the ~15 lines of
   checkpoint/resume and the free 25.7M T4 run *ahead* of R2, and changes the
   cost order.

Option 1 is cheaper and is probably right first, but **it must not be reported
as satisfying R2's kill.** Recording that here so a screening number cannot
quietly become a GREEN later.

---

## WHAT R2 CANNOT DECIDE, whatever it returns

- **Novelty.** The Littlewood–Offord resolution in iteration 2 is
  **sign-agnostic** — it explains *routing*, and Star-Transformer 1902.09113 had
  routing in 2019, unsigned. R2 is about the sign pattern of the block; it does
  not touch whether the combination is new. G1's per-route sweep
  (Brualdi–Shader applications, tropical attention, hierarchical/RG attention,
  group-testing attention) is a separate obligation.
- **Capability.** M3 is UNTESTED. Every route in M2′ produces a statistic.
- **Anything about trained models.** See the defect above.
