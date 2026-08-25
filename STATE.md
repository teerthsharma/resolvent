# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **16 complete, 17 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 17)

**Wait for the Health Inspector's verdicts, and while he runs, close the
self-satisfying provenance bind - it is the last live hazard in the tree.**

**THE HAZARD, from Wilson:** `1.44x`, `1.0334`, `0.379x` and `3,319,296` now
appear in `DONE.md` exactly once each - **inside the text of the finding that
reported their absence.** `tests/chase/test_hub_package_hardening.py` asserts
`"3,319,296" in done`. **That assertion can now be satisfied by the report of the
absence itself.** It is the same class as the fabricated number: a bind that
passes for the wrong reason. **A provenance test must match a MEASUREMENT, not a
mention** - the fix is to require the number to appear in a context that carries
its own run evidence, or to point the bind at the archive where the measurement
actually lives.

**THE ROUND'S SHAPE IS DECIDED AND SHOULD NOT BE RE-ARGUED.** K1's `D_FR` clause
is resolved against an interval - **-0.4137 [-0.4579,-0.3704]** and **-0.4654
[-0.5173,-0.4160]**, both entirely below **-0.30** - so the pre-registered *"no
leap"* branch has **FIRED**, **ARM A has not survived K1**, and **ARM B is not
authorized**. `TWOSPHERES: KEPT` is **not available**.

**B1 is doubly settled and neither settlement helps ARM B:** it is **not** a
rename of the key-norm (R1 = -0.025318, reproduced twice), **and** it is
**ill-posed as written** - the top-k it selects retains **5.4%** at k=8 under a
change of the very token the selector is forbidden to see.

**DR HOUSE REMAINS IN THE BOX, and the reason must be stated rather than
assumed.** K1 died of **EVIDENCE** - a measured slope with an interval. The skill
is explicit: *"Wilson refuted it with a verified fact - dead is dead, and no leap
un-refutes a fact."* Releasing him here would be using a leap to argue with a
measurement.

**WHAT STILL WANTS AN ANSWER AND IS NOT BLOCKED:** the aggregator win is **real,
large and unexplained** (+1.1347 / +0.9355 / +0.4959, CIs excluding zero, Wilson
confirming Cameron) - and **it is not a sphere result**, since the sphere-vs-TV
term at the same aggregator is only **~5%** of it. **That is the live thread, and
it belongs to whatever comes after this round.**

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
