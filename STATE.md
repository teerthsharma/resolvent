# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **6 complete, 7 next** |
| phase | **G1 fetches, then ARM A (the torque probe)** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 7)

**Read Wilson's TASK 2/3 K1 numbers when the buckets land, and read the D_FR
slope's CI before its point estimate.**

TASK 1 and TASK 4 are in and both changed the ground:
  * **the flip instrument is calibrated** - planted sign change fires, softmax
    reads zero, and the `1e-200` case shows the float path missing a flip the
    valuation catches;
  * **F16 does not generalise.** `_causal_sgate_operator` is signed at **every**
    lam including 0.10 at ARM A's geometry - 30/30 readings, `min A =
    -1.363636e-01` - because the mean causal `|w|` here is **1.171e+01** against
    the harness's **2.682399e-03**. **`lam` is a threshold at 1.0 only when the
    logits are near zero.** K1's flip half is therefore a measurement now.

**The reading order is fixed:** whether the `D_FR` slope's bootstrap CI
**excludes**, **includes** or **straddles** the pre-registered **-0.3** line
comes before the point estimate. Iteration 4 read **-0.3061** with no error bar
and that is the entire reason K1 was voided. **Do not round toward a verdict in
either direction** - an unfavourable reading pulled out of noise is worth exactly
what a favourable one is.

**Gate 3 is closed and it is a FAIL, not a void.** The audit's repaired control
fires at 0/48 with separation exactly zero, so the 109/384 live off-schedule
draws are the arm, not the probe. **The additive-basis arm's flip behaviour
tracks lattice placement rather than content.** That does not touch the basis's
COVERAGE result - k=12 proven for `[1,56]`, k=18 for `[1,127]`, unreachable
fraction driven to exactly 0.0000 - which is a separate object and survives.

**A defect to repair when the instrument is next opened:** `scale/valuation.py`'s
docstring argues the underflow with float32's `1.18e-38` while the function takes
a Python float64. The instrument is GREEN; **its stated reason does not reproduce
at the magnitude it names.**

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
