# M2 — what each outcome will mean, written BEFORE the control finished

**Written 2026-08-25, loop iteration 25, with `dense_signed__at_pivots/s2048` at
2 of 8 batches.** The claim arm is complete; the control is not. Nothing here may
be edited after the remaining batches land — that is the entire point of writing
it now.

This project has been wrong fifteen times about instruments, and several of those
were *interpretations* that hardened after the number arrived. The defence is to
fix the reading first.

## The measured state at the time of writing

| s | 8 | 32 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| **pivot** (claim) | 0.024658 | 0.028564 | 0.031006 | 0.028809 | 0.029663 | 0.029907 |
| **dense** (control) | 0.024658 | 0.021240 | 0.010254 | 0.003174 | 0.000488 | 0/1024 so far |

Claim-arm slope **+0.0270**, R² **0.5222**, over a 256× context growth.

## What the frozen kill can and cannot decide

`LOCK M2 efadc390c93f`. The kill reads:

> slope(c ∈ P) < −0.3, OR `c ∉ P` is ALSO flat, OR any published bench number moves.

- **Clause 1 is evaluable** and currently reads +0.0270, far from −0.3.
- **Clause 2 is NOT evaluable and never will be.** `c ∉ P` is zero by
  construction: `hop2[i,j] = Σ_{p∈P} A[i,p]·A[p,j]` contains no term with index
  `c` when `c ∉ P`. A quantity that cannot be nonzero cannot be "also flat" in
  the sense the clause intends. **This is a defect in the kill I wrote, not a
  pass.** It must be reported as unevaluable, never as satisfied.
- **Clause 3** is checked every bucket; calibration has been bit-identical
  throughout.

`dense_signed__at_pivots` was added *because* clause 2 is dead. It is the
comparator that can actually falsify, and it is not part of the frozen text.

## The readings, fixed in advance

**A. Control finishes near zero at s=2048** (current trajectory: 0/1024, CP upper
0.00292).
→ The separation is real and monotone: 1.00× / 1.3× / 3.0× / 9.1× / 61× / ≥10×.
→ **What this establishes:** routing hop-2 through a fixed-size pivot set holds
the content-conditional sign rate flat in context, where the same operator with
dense hop-2 at the *same intervened tokens* decays. That is a mechanism result.
→ **What it does NOT establish, and must not be written as if it does:**
   - It is **not a capability.** `CHECKLIST.md`'s preamble is explicit —
     "statistics are not capabilities". M3 is the capability item and is UNTESTED.
   - It is **not novelty.** Star-Transformer (1902.09113) already routes hop-2
     through a relay with an unsigned operator. What is unoccupied is the
     *signed* version, and S2 — pivots+unsigned vs pivots+signed — has not run.
     Until it does, the honest attribution of the flatness is **routing**, not
     signedness.
   - It is measured on **random projections**, not trained ones.
   - The property it holds flat is one a **single GELU between two softmax layers
     restores** (`test_a_nonlinearity_between_softmax_layers_gives_the_sign_flip_back`),
     so the surviving claim is depth/parameter efficiency, not something softmax
     cannot do at all.

**B. Control rises at s=2048 instead of staying near zero.**
→ The separation is not monotone and the mechanism story is weaker than the
   s=1024 point suggested. Report the non-monotonicity plainly; do not average
   it away, and do not drop s=2048 for being "noisy".

**C. Claim arm had come out with slope < −0.3.**
→ M2 RED, all build work stops, and ARSENAL A1 and A4 freeze with it. This did
   not happen, but it is written here so the bar is on record as having been
   real rather than decorative.

**D. Control turns out flat and nonzero** — i.e. it does *not* decay.
→ Then routing explains nothing, the flatness is a property of the operator or
   the harness, and the entire pivot design is unsupported. **This is the
   outcome that would kill the design**, and it is why the control had to be
   built to be capable of it.

## The one number that would change the verdict

S2's ablation: **pivots + unsigned** at the same geometry. If an unsigned
operator routed through the same pivots is *also* flat, then the contribution is
routing — which is Star-Transformer's, from 2019 — and the claim sentence must be
rewritten before anything is published. That is stopping condition **G4**, and it
is not yet run.

## Standing facts that do not change whatever M2 says

- ARC-AGI has never been scored; a Turing-style eval has never been attempted.
- Nothing has trained above 3.65M parameters against a 300M gate.
- The one capability comparison ever run at matched parameters went **against**
  this operator: COGS-gen softmax 0.0293 vs sgate 0.0000, Fisher p = 2.75e-05,
  and behind in-distribution too.
- The whole-suite `pytest tests/ -q` has never completed in nine attempts across
  three agents. **No total pass/fail count exists.**
