# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **23 - ROUND CLOSED** |
| phase | **CLOSED - `TWOSPHERES: BROKEN`, ARM A, K1 dual slope** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5 is CLOSED)

**Round 5 ended at iteration 23 with `TWOSPHERES: BROKEN` - ARM A, K1's dual
slope, displacement clause.** Line 1 of `DONE.md` carries the verdict and `D1.md`
ships.

**NOTHING FURTHER IS OWED ON THIS ROUND.** The chain ran to completion, the audit
struck three claims and all three were applied, and the deliverable meets its own
acceptance criteria.

**WHAT A NEXT ROUND WOULD INHERIT, in the order it is worth having:**

  1. **K1's sign-flip clause is alive and unresolved.** 8 events in 2400 draws;
     the interval is **count discreteness**, and the unsigned arm's exact interval
     **contains** the signed arm's rate, so the two arms are not separated at
     these sample sizes. **Needs the contract's own 20000 draws per cell**, about
     eight hours at the measured per-draw cost. This is the only pre-registered
     kill in the round that remains genuinely undecided.
  2. **What peak attention tracks, once the key-norm confound is removed, is not
     established.** It retains **2.9% / 13.3% / 58.9%** against a key-norm-matched
     filler, and the match is by **rank, not value**, so those are **upper**
     bounds. The residual at k=128 is the only part with room in it.
  3. **The K1 slope rests on a journal replay, not a fresh derivation.** The
     replay matched bitwise, so the journal is intact - but **nobody re-derived
     that slope from draws**. A round that wanted to lean on it should.
  4. **The identity is the durable result and it is a constraint, not a tool.**
     `theta = arcsin(sqrt(TV))` on a one-token mask means **any** future
     ablation-and-measure probe on this geometry compares two aggregations of one
     number. A design that wants the geometry to earn something must change the
     PROBE, not the metric.

**THE INSTRUMENT LESSON, since it repeated four times and is the most transferable
thing here:** every verdict gate that failed this round measured something
**ADJACENT** to what had been pre-registered - bulk rank where selection takes a
top-k, argmax where the claim was about values, half of a two-part condition, and
a control that could not reach the logic it guarded. **Each erred toward the
flattering reading.** All four were caught by the probes' own output, which is the
only reason they are recorded rather than shipped.

## Open REDs

None new. Round-4 carries: F-core (every additive route dead), F-cover (arm A
redirected to an additive basis of order 2), F-journ (13/37 drift in sigma/term,
rate intact 37/37).

## Carried, and load-bearing

- **F-green is the only positive result in four rounds, and it is UNSIGNED.**
- **A sign measurement without its logit scale is not a measurement.**
- **n_train >= 8192 or the reading ranks overfitting.**
- **No multiplication inside a sign decision** (G8, new) - `lo*hi` underflows to
  exactly -0.0 in float32 while both factors are healthy.
- **A leap binds before it counts.** House's displacement frame is MOTIVATION
  until a fellow writes the RED test.

## Board in flight

G1 fetches -> ARM A torque probe (K1 dual slope, K2 filler twin, K3 geometry
earns itself) -> G-b reproducible summation in parallel -> ARM B birth gates ->
M3 at n_train=8192.
