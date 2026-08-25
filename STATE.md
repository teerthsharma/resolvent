# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **2 complete, 3 next** |
| phase | **G1 fetches, then ARM A (the torque probe)** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 3)

**ARM A. The torque probe. Pure measurement, no training, no build.**

G1 is complete at 6/6 and the gate is passed. Two cells survive - the additive-
basis coverage theorem and the Procrustes torque as a ROUTING OBJECTIVE - and
**both owe their kills**, which ARM A is.

Per draw on the LIVE `pivot_unsigned` arm: `A^c` and `A^0`; `theta_i` per row;
`D_FR = E_i[theta_i]`; `||tau||_F`; rank profile and `gamma_r` of `Xi`.
`k in {8,32,128,512}`, 20000 draws, **`c` and `j` uniform at random** - the
carpet discipline that two rounds broke.

**Three kills, and K3 decides whether the round has anything:**
  * **K1** `flip(s)` slope <= -0.4 AND `D_FR` slope >= -0.1 on the SAME draws.
    **`D_FR` slope < -0.3 means "no leap".**
  * **K2** `D_FR(causal c)` vs `D_FR(filler c)` with DISJOINT CIs, or the
    statistic cannot lose to a filler and the table is VOID.
  * **K3** `theta` must beat raw TV by standardized effect on identical draws.
    **After this fetch round K3 is decisive:** the sphere map is textbook and the
    TV ablation probe is occupied, so if `theta` does not beat TV, the round is
    classical geometry pointed at an existing probe and **TV ships instead.**

Read on X4. **No multiplication inside any sign decision** (G8). Bucket it -
20000 draws at four sizes will not fit one call, and two unbucketed runs have
already died here.

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
