# State

**Updated: 2026-08-25 - ROUND 5 open under CEQ v7, the two-spheres round.**

| field | value |
|---|---|
| round | **5** - CEQ v7, promise `TWOSPHERES`, **30 iterations** |
| iteration | **5 complete, 6 next** |
| phase | **G1 fetches, then ARM A (the torque probe)** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 5, iteration 6)

**Read Wilson's K1-on-a-signed-arm report and act on TASK 4 FIRST.**

Wilson is running `scale/arm_a_k1.py`. **The order of reading is fixed by the
contract and by this project's own history: if his planted-sign-change control
did not read nonzero, the flip instrument is broken and every other number in
his report is void.** That is read before the slopes, not after.

If the control fired, then K1 has a real reading for the first time:

  * **the flip half** measured on `_causal_sgate_operator` at whichever `lam`
    is genuinely signed at this geometry - **F16 says `lam=0.10` has min entry
    exactly 0.000e+00 at harness scale, so `lam` is a threshold at 1.0, not a
    dial**, and Wilson is verifying or refuting that first;
  * **the D_FR half** on 6 k points and >=400 draws, **with a bootstrap CI on
    the slope itself**. The only reason K1 was voided is that **-0.3061 against
    a -0.3 line had no error bar.** Read whether the CI **excludes**, **includes**
    or **straddles** -0.3, and do not round toward a verdict in either direction.

**Do not declare the pre-registered "no leap" branch on a point estimate.** The
unfavourable reading taken from noise is worth exactly as little as the
favourable one, and this project has already spent two rounds learning that.

**Cameron's arm_a_rebuild landed and its gate 3 FAILED** - on-schedule X4
0.044271 [0.027821, 0.069748] vs off-schedule 0.000000 [0, 0.009905], CIs
disjoint. That is the test that killed the dilation arm firing again, and it
needs its own iteration; it does not get folded into K1's.

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
