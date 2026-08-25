# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **8 complete, 9 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 9)

**Collect whichever fellows have landed; if none has, measure the `||tau||`
trajectory that X6 actually names.**

**X6 is GREEN and the equilibrium clause SURVIVES** - residual at the glance
0.599101 / 0.388587 / 0.321843, 100% converged in 28-53 steps. But the probe
measured the **Karcher residual** trajectory, not the **`||tau||_F`** trajectory
the clause names. `||tau||` was taken **at the glance only**. That gap is stated
in `CHECKLIST.md` and is the cheapest remaining thing to close.

**THE FINDING THAT MATTERS MORE IS THE ONE X6 DRAGGED IN.** Uniqueness holds on
**0.9333 / 0.8167 / 0.5167** of draws at k = 8 / 32 / 128 - **at k=128 the
Karcher mean is not unique on 48.3% of draws**, and settling also slows (28.43 ->
52.55 steps). **Both trends run against ARM B, which wants k pivots.** Before ARM
B is built on a Karcher mean, decide what happens on the draws where that mean
does not exist uniquely - it is a precondition, not a caveat.

**Reading order for the tier, fixed:** Wilson's facts settle factual disputes; a
fellow finding without its RED test goes to **Open**; the **Health Inspector runs
BEFORE the prognosis**; the Chart carries **every** fellow including overruled
ones, with the reason. **Dr House (`model: fable`, 5 min, no nurses) only if the
problem survives all four rungs AND the gap is INNOVATION, not evidence.**

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
