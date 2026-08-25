# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **3 complete, 4 next** |
| phase | **G1 fetches, then ARM A (the torque probe)** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 4)

**RUN ARM A.** The module exists (`scale/torque_probe.py`); nothing has been
measured yet.

Bucketed - 20000 draws at four sizes will not fit one call, and two unbucketed
runs have already died here. Start at `k in {8, 32}` with a declared draw count,
journal it, then extend.

Three kills, and after the G1 round **K3 decides whether the round has
anything**: the sphere map is textbook and the TV ablation probe is occupied, so
theta must beat raw TV by standardized effect on identical draws or **TV ships**.

**Pin the thread count in the probe itself.** Wilson's finding is that
`m3_capability.py`'s published numbers were reproducible only because the shell
happened to be set to 2 - the file said nothing, and the same log carries the
same command at 20 threads (14 runs) and 3 threads (2 runs). **A probe that does
not pin is a probe whose number is a function of its launcher.**

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
