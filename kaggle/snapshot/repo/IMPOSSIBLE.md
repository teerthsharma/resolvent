# DR HOUSE TODO — the impossibility list

**What belongs here.** An item lands on this list when a fellow finds that the
way forward needs a LEAP — not more evidence, not more caution, but an
innovation nobody in the room has. That is precisely and only what Dr House is
for.

**What does NOT belong here** (from `dispatching-house-mode`):
- Wilson refuted it with a verified fact. Dead is dead; no leap un-refutes a fact.
- Chase found it dangerous. That is a risk decision, not an innovation gap.
- Cameron found a cheaper path. The theory did not die, it lost.
- Nobody wants the thing any more. Let it go.

**The rule that put this file here:** a fellow who kills an M point must name its
successor. When they honestly cannot — when the successor requires a leap — they
say "this needs a leap" and name what it would have to DO. That statement lands
here. **Inventing a weak successor to avoid writing on this list is the failure
this file exists to prevent.**

**Dr House produces a HYPOTHESIS, not a finding.** He is exempt from RED-first
because five minutes does not fit a test — and that exemption is exactly why his
output re-enters the differential as a candidate and gets bound later by a fellow
with a RED test. A leap that never gets bound stays OPEN forever.

---

## OPEN — awaiting a five-minute fable run

### I1. M6 — bound hop growth with no denominator, or prove it cannot be done

**Raised by:** Chase, iteration 27. He explicitly refused to invent a successor
here, which is the correct behaviour and is why this file exists.

**The impossibility, stated precisely.** M6 asks for
`||A^h|| <= 2 w(A)^h` via a numerical radius `w(A) <= rho` — one scalar, no sum
over `s`, so nothing for context to dilute.

Measured [RUN, ARSENAL A2]: on the shipped `tgate` at s=64, seed 0,
**`w(A) = 1.499315`**, `||A|| = 2.837508`. The chain HOLDS at every hop tested —
but it is **VACUOUS**, because `w(A) > 1` makes `2 w(A)^h` GROW with h while the
true `||A^h||` FALLS (the operator is nilpotent):

    h=1  ||A^h|| = 2.838e+00   2w^h = 2.999e+00
    h=2  ||A^h|| = 1.777e+00   2w^h = 4.496e+00
    h=3  ||A^h|| = 7.119e-01   2w^h = 6.741e+00
    h=4  ||A^h|| = 2.099e-01   2w^h = 1.011e+01

A certificate whose bound grows while the quantity shrinks certifies nothing.

**Why it may be impossible rather than merely unbuilt.** The operator's stated
price is row-L1 growing as **O(s·g)** — that is the cost of having no
denominator, and it is the same absence that M2's flatness depends on. So:

> **Any scalar bound tight enough to publish may be exactly the denominator whose
> absence M2's flatness requires.**

If that is so, M6 and M2 are not two items — they are one tension, and it cannot
be satisfied twice.

**What the leap would have to DO:**
1. bound `||p(A)||` for the whole path sum `p(A) = sum_h A^h` (Crouzeix-Palencia
   territory — Berger alone only gives the term-by-term bound), with
2. **no sum over `s` anywhere** in the bounding quantity, and
3. no per-row normalizer,
4. **OR** prove goals 2 and 3 are formally incompatible with M2's flatness.

**The fourth branch is worth as much as the first three.** A proof that a
denominator-free magnitude certificate and context-stable signed influence cannot
coexist would fold M6 into M2, explain the entire 1/s programme in one statement,
and be publishable as a negative result with a machine-checkable core. This
project's best assets are already of that shape.

**Provenance note:** the operator's spectrum is `{0}` (nilpotent), so every
eigenvalue-based guard is vacuous — that is exactly how T1 died. `W(A)` is the
only non-vacuous control object anyone has proposed, which is why the leap has to
go through it or explicitly around it.

**Status:** OPEN. Not yet dispatched. Trigger check: (1) M6 is dying — yes,
the guard does not exist in any file and the bound as stated is vacuous;
(2) cause of death is missing innovation — yes, Chase found no successor and said
so; (3) somebody still wants it — yes, it is a MANDATORY checklist item and the
denominator-free story is the whole mechanism claim.

---

## BOUND — a leap that a fellow later bound with a RED test

*(none yet)*

## RETIRED — proposed, and the thing it was for is no longer wanted

*(none yet)*
