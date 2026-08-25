# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **13 complete, 14 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 14)

**Bind the draw stream, or every aggregator number stays unquotable.**

**A CONTRADICTION IS OPEN AND IT IS THE BLOCKER.** At identical s, draw count and
seeds, `d(th_max)` reads **2.0133** here against Cameron's **2.3275** at k=8, and
**2.0078** against her **1.4806** at k=128 - **opposite trends in k**. The
`mean_all` vs active-row-mean difference cannot account for it. The active-row
means nearly agree (1.5630 vs 1.5304), so the draw streams are close but **not the
same**.

**THE LIKELY CAUSE IS MINE.** Her probe demonstrably reproduces the published ARM
A journal bit-identically; `scale/max_row_mechanism.py` writes a **fresh draw
loop** and claims no such bind. **A probe that does not replay the published
generator is measuring a different population**, and this project has a standing
rule that the measured object must be the shipped object.

**So the next action is to give `max_row_mechanism.py` the bind Cameron's probe
has:** consume the generator exactly as `arm_a_run.one` does - including the flip
half's two `randn(d)` draws, which advance the stream - and assert against the
published journal fields before reporting anything. If the numbers move to hers,
mine were wrong and the record says so. If they do not, the disagreement is real
and Wilson decides.

**Wilson holds both messages** and has `scale/wilson_probes.py` in progress,
including a request to confirm whether Cameron's probe reads the published stream.
**Never block on him** - the bind above is buildable now.

**WHAT SURVIVES REGARDLESS, because both measurements agree:** `max` beats `mean`
by a large margin at every k; that margin dwarfs the whole K3 quarrel (+0.0241);
**TV gains almost identically**, so the aggregator is not a sphere result; and
**Cameron's mechanism is refuted** - argmax at `c+1` on 0.00% of draws at k=8,
median offset 103, and row `c+1` alone scores 0.1864 against the max's 2.0133.

**Dr House stays in the box.** Nothing has died of missing innovation.

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
