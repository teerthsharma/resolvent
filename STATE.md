# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **4 complete, 5 next** |
| phase | **G1 fetches, then ARM A (the torque probe)** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 5)

**Iteration 5 is an Inspector pass (every 5th) - and it has a repair to make
that Foreman found.**

`inspector.py:299`'s *"published: M2 two-point slope"* check computes
`log10(0.02732/0.16511)/log10(4)` **from two constants it holds itself** and
compares to a third constant. **It re-verifies `math.log10`. It has never checked
a measurement**, and it has passed at every pass this round. Those two numbers
are **sgate** values that appear in **no journalled unit** - grep returns zero
hits, and no unit carries n=751 or n=549.

**Then re-run K1 on a SIGNED arm.** The flip half is vacuous on `pivot_unsigned`
by F1 - Jacobian min entry exactly 0.000000e+00 - so K1 needs `pivot_signed` at
`lam=1.00`, the regime where the operator is genuinely signed (bulk negative
fraction 0.500564). And more draws and more k points: **-0.3061 against a -0.3
line on a 3-point fit is noise, in either direction.**

**Do not read a verdict out of that margin while it stands.** K1 is UNTESTED.

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
