# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **10 complete, 11 next** |
| phase | **ARM A under the escalation chain; autonomous to iteration 30** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 11)

**Collect Chase and Cameron; hold Foreman at the Wilson rung.**

**FOREMAN'S REPORT IS NOT A VERDICT AND MUST NOT BE TREATED AS ONE.** It is
RED-bound and it replays ARM A's own draw stream, but the chain is
**fellows -> Wilson -> Health Inspector -> Dr House**, and it has passed exactly
one rung. **Wilson has been sent F4/F5** - the dead-row floor and the contaminated
K1 slope - because his K1 run reads through the same statistic and his facts, not
Foreman's, settle it.

**THE FINDING THAT REFRAMES THE ROUND, once it clears Wilson:**
**`theta_i = arcsin(sqrt(TV_i))` IDENTICALLY** on a one-token mask - one degree of
freedom, residual **2.980e-07** at the float32 floor. **K3 as written cannot
distinguish geometry from row-wise concavity**, because theta is a fixed monotone
function of TV **by construction**. And the sphere's own curvature is measured as
a **liability**: the chord beats the geodesic by 0.65-1.25%, `sqrt(TV)` by
2.87-5.70%.

**WHAT K3 NEEDS IF IT IS TO MEAN ANYTHING:** a control arm that is a concave
reparametrisation of TV **with no geometric story** - `TV^p` - and theta must beat
**that**, not raw TV. **Foreman's `TV^0.2` numbers do NOT support shipping it**:
he searched 8 exponents on the draws that scored them, with no bootstrap CI and no
multiplicity correction, and he says so himself. **That arm has to be built
properly before it can kill or save K3.**

**Equilibrium is settled and written into the contract:** the certificate is the
**Karcher residual**; `tau` is a **displacement statistic**; `tau = 0` may not be
called equilibrium. **The price rides with it** - the Karcher mean is not unique
on **48.3% of draws at k=128**, and ARM B may not be built on it until that is
answered.

**Dr House (`model: fable`, 5 min, no nurses) is NOT triggered yet.** Foreman did
not die of missing innovation; he produced a result. Release him only if the
problem survives all four rungs AND the gap is INNOVATION rather than evidence.

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
