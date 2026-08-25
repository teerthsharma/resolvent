# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **9 complete, 10 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 10)

**Collect the fellows if any have landed. Otherwise: DECIDE WHICH EQUILIBRIUM
THE CONTRACT MEANS, because it currently asserts two and they are not the same
point.**

X6 is fully measured now, both halves:
  * the Karcher **residual** reaches **5.4944e-13** and the glance is NOT the
    fixed point (residual 0.599101 / 0.388587 / 0.321843) - **the clause
    SURVIVES**;
  * the **`||tau||`** trajectory **plateaus NONZERO** (7.255e-02 / 2.962e-01 /
    1.019e+00) at that same fixed point.

**Both cannot be the equilibrium condition.** `tau = 0` holds iff the reading is
parallel to `xbar`, the normalised **Euclidean** mean; the Karcher mean is the
**geodesic** one, and the two sit **0.031648 rad [0.026318, 0.037550]** apart at
k=8 with the CI excluding zero at every k. **The contract asserts both, so one of
its sentences is false of whatever ARM B is built on.** This is a specification
repair, not a measurement, and it is cheap: pick the geodesic mean (and cut or
restate the `tau = 0` sentence), or pick `tau = 0` (and stop calling the settled
reading a Karcher mean). **Do not build ARM B while both stand.**

**Carry into that decision, from iteration 8:** the uniqueness precondition holds
on only **0.9333 / 0.8167 / 0.5167** of draws at k = 8 / 32 / 128. **At k=128 the
Karcher mean is not unique on 48.3% of draws** - so the geodesic branch of the
repair has to say what happens there, and "unique for theta < pi/2" is a
precondition that fails on half the draws at the pivot count ARM B wants.

**Tier reading order unchanged:** Wilson's facts settle disputes; a fellow finding
without its RED test goes to Open; **the Health Inspector runs BEFORE the
prognosis**; the Chart carries every fellow including overruled ones. **Dr House
(`model: fable`, 5 min, no nurses) only if the problem survives all four rungs
AND the gap is INNOVATION, not evidence.**

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
