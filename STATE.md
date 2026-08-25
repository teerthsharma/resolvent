# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **17 complete, 18 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 18)

**Run the key-norm-matched filler against the AGGREGATOR. It is the one control
that decides whether iteration 17's finding is a result or an artifact.**

The aggregator win is now explained: `theta.max()` is a monotone read of
`max_i A[i,c]`, the largest attention weight any row places on `c`, with an AUC
gap of **~1e-03** against the raw attention statistic - **the sphere contributes
nothing to it.** The mechanism is concentration: **0.881909 causal vs 0.161140
filler**, a ratio of **5.473**.

**BUT THE DECISIVE CONTROL IS MISSING AND IT IS NAMED IN THE RECORD RATHER THAN
GLOSSED.** Wilson measured that a **key-norm-matched filler** (ranks k+1..2k)
removes **~65%** of the mean-based K2 effect - 1.2267 collapses to 0.4301, though
it survives with a CI excluding zero. **Nobody has run that filler against
`max_i A[i,c]`.** If the aggregator's 5.473 ratio collapses the same way, then
peak attention is largely the key-norm returning and **the finding is Chase's F2
firing a third time**. If it survives, peak attention tracks something the
selector's score does not, and **that is worth carrying past this round.**

Within-arm `rho(||k_c||, max A)` reads **+0.25 to +0.45**, which is why the
question is open rather than settled. **The pooled +0.76 must not be quoted as
evidence** - it mixes the two arms and therefore IS the separation under test.

**HOLD EVERYTHING ELSE.** K1's `D_FR` clause is resolved against an interval,
the *"no leap"* branch has **FIRED**, **ARM A has not survived**, **ARM B is not
authorized**, and `TWOSPHERES: KEPT` is **not available**. The aggregator finding
is a fact about softmax attention, **not** about the two-spheres frame.

**The Health Inspector is still out.** He audits the log, and he was pointed
hardest at my own six probes and at the self-satisfying provenance bind. **The
prognosis is not written until he reports**, and **his strikes are strikes** - no
appeal, straight to Open.

**Dr House stays in the box.** K1 died of EVIDENCE. *"Dead is dead, and no leap
un-refutes a fact."*

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
