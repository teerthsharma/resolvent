# A trained measurement — what each outcome will mean, written BEFORE any run

**Written 2026-08-25, loop iteration 41. No trained operator has been measured.
Nothing here may be edited after one is.** That is the entire point of writing it
now.

Iteration 39 established a fact nothing in this repository had recorded: **every
M2 and M5 number is measured on a randomly-initialized operator.**
`scale/pivot_probe.py:143-145` draws `q`, `k` and the gate vector from
`torch.randn`; `scale/` loads no trained weights anywhere.

*"The test measured an untrained tensor"* is a true sentence. It is also exactly
the sentence that rescues a dead theory when nobody writes the reading down
first. `M2_PREREGISTERED_READING.md` predicted M2's vacuous control **before the
run finished**, and that is the only reason the defect was a finding rather than
an excuse. This document does the same job for the trained question.

## What is already measured, and how solid it is

Shipped operator `sgate`, random initialization, `PROTOCOL: SCALING`,
placement `in_P`:

| s | k/n | rate | CP95 |
|---|---|---|---|
| 8 | 124/751 | 0.16511 | [0.13925, 0.19364] |
| 32 | 15/549 | 0.02732 | [0.01537, 0.04466] |
| 128 | 0/213 | 0.00000 | [0.00000, **0.01717**] |
| 512 | 0/60 | 0.00000 | [0.00000, **0.05963**] |

Fitted slope **−1.298** against M2's **−0.3** bar. Dense control **−1.088**, so
routing is *worse* than not routing.

**THE HONEST WEAKNESS, STATED BEFORE ANYONE ELSE RAISES IT.** The tail zeros rest
on 213 and **60** draws. At s=512 the CP upper bound is **0.05963** — the true
rate could be *higher* than the well-measured s=32 rate. Writing "0.00000" reads
far stronger than 60 draws warrant.

**AND IT DOES NOT MATTER, [DERIVED] with the arithmetic shown.** The slope
through the two well-measured points alone is

    log10(0.02732 / 0.16511) / log10(32 / 8) = −1.2977

against the published full-fit **−1.2980**. **The two weak zeros move the slope
by 0.0003.** The kill is carried entirely by 124/751 and 15/549, both with tight
intervals. A tail re-measurement at higher draw counts cannot rescue M2 and is
not worth buying.

## What the frozen LOCK can and cannot decide

`LOCK M2 efadc390c93f`. The item text is frozen and **says nothing about
initialization**. So:

- A trained measurement is **NOT a re-run of M2.** M2 was evaluated, the kill
  fired, and its status is immutable. Re-running the same item against a
  different object is the ParaFormer hazard (G3) wearing a new costume.
- A trained measurement is therefore a **NEW checklist item** with its own
  frozen text, its own pre-registered kill, and softmax in the same table.
  It does not inherit M2's identity, and a GREEN on it is not a GREEN on M2.
- The contract is explicit: **a RED is overturned only by convicting the
  INSTRUMENT, never by adjusting the arm.**

## What would convict the instrument — and what would not

**WOULD NOT.** "The trained operator reads a flatter slope." That does not
convict anything. It would mean the instrument correctly measured the object it
was pointed at, and the *claim's scope* was mis-stated. Scope errors are repaired
by restating the claim, not by lifting a RED.

**WOULD.** A demonstration that the random-init measurement is **not
informative** about the quantity at all — for instance that `run_arm`'s flip
statistic is dominated by the Gaussian draw's scale rather than by the operator's
structure, so it would read −1.298 for *any* operator including ones known to be
context-stable. That is a property of the instrument, testable **without
training anything**, and it is strictly cheaper than a Colab run.

**That cheaper test comes first.** If the instrument survives it, a trained run
cannot overturn M2 no matter what it reads.

## The readings, fixed in advance

**A. Trained slope ≤ −0.3 (still steep).** → The dilution is a property of the
architecture, not of initialization. **M2's RED becomes permanent and the scope
question closes.** This is the outcome that ends the line of enquiry, and it is
the one I expect: the decay is structural — a third token is 1 of ~s
intermediates in the `k ≥ 2` term of `J = Σ A^k` — and training changes the
entries of `A`, not how many of them there are.

**B. Trained slope > −0.3 AND the unsigned/softmax control is steeper in the same
table.** → A genuine finding, and it belongs to the NEW item, not to M2. It
would require the full Phase-2 treatment before any claim sentence: 5 seeds,
matched params, matched lr sweep, softmax timestamped first.

**C. Trained slope > −0.3 but the control is comparably flat.** → Nothing.
Flatness shared with the control is not a capability, it is a property of the
probe. Reads as **instrument**, and the cheaper test above should have caught it.

**D. Anything at a single seed.** → Nothing, in every direction. One seed is not
a measurement, and this project has already published a slope that moved 0.26
across **draw count alone** (`128 → −1.295, 256 → −1.551, 512 → −1.402`).

## The bar problem — stated because it decides whether to run at all

The standing constraint on this project is **"if it doesn't survive over 300M
then it's false."** Nothing here has trained above **3.65M**. The free-T4 shape
`LOOP_PROMPT.md` names is **25,707,520 parameters** — **8.6% of the 300M gate**,
and it runs at **1.6% of a Chinchilla budget**.

**So even outcome B at 25.7M does not settle the question the bar asks.** It
would be a provisional reading at one twelfth of the required scale, and it must
be labelled that way in the same sentence it is reported. Any document that
quotes a 25.7M result without the 300M gate beside it is overclaiming.

## Cost, honestly

| item | figure |
|---|---|
| trained run | free T4, 2.45 GiB against 14.5, one 12-hour session |
| resume | exists, **bitwise-verified** (`tests/chase/test_resume_checkpoint.py`, 3 passed) |
| the cheaper instrument test above | CPU minutes, no training |
| what 25.7M buys against the bar | **8.6% of 300M** |

## What this document commits me to

1. **The instrument test runs first**, and it is cheap. If the instrument
   survives, M2's RED stands regardless of any trained number.
2. **No trained run is authorised by this document.** It fixes the reading; the
   decision to spend the run is separate and is the user's.
3. **A trained measurement gets a new item, never M2's.** Its text freezes before
   its first run.
4. **Outcome A closes the question.** If the slope stays steep, the scope gap is
   answered and nothing further is owed to it.
