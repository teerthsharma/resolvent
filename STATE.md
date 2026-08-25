# State

**Updated: 2026-08-25 - ROUND 4 open under CEQ v6', the chosen-sign round.**

| field | value |
|---|---|
| round | **4** - CEQ v6', promise `CHOSENSIGN` |
| iteration | **3 (round 4) complete, 4 next** |
| phase | **X4 instrument first - everything downstream reads through it** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor, not token predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 4, iteration 4)

**Build arm A as the DIFFERENCE-SET COVERAGE construction - and describe it that
way, never as "co-prime spacing", which G1 has taken.**

The birth gates are already specified and one is already GREEN:
  * `|D-D| = v-1` as a VALUE - **GREEN**, exact at v = 7, 13, 21, 31, 57;
  * **flipper placed UNIFORMLY AT RANDOM, never on the schedule** - the
    discipline `carpet_probe.py:24` demanded and round 3 broke, which is how a
    lattice-aligned `c` produced a flat reading that was an artifact;
  * **off-schedule flip rate within CI of on-schedule** - the exact test that
    killed the dilation, promoted to a birth gate.

Kill: any bitwise-identical gradient pair (severing), OR `|slope| >= 0.01` over
s=512->2048 **on the X4 instrument**, OR M3 fail at n=8192.

**Read it on X4, not on the float path.** The float statistic misses flips whose
product underflows, and at depth median |grad| is 2.8e-32.

## Open REDs

**F16** - the signed arm was not signed at harness scale; round 4's entry point.

**SPENCER IS SLACK BY 7.4x** at this geometry, so arm B's stated theoretical
warrant does not bind. Not a kill - a correction to the reason. The background is
dominated (**max term 50.7% of total mass**), which is extreme-order-statistic
selection doing what round 2 already identified.

## Carried, and load-bearing

- **A sign measurement without its logit scale is not a measurement** (F17).
- **Routing beats softmax and survives F16** (F18) - the one live positive result.
- **n_train >= 8192 or the reading ranks overfitting** (F19).
- **Co-prime severance 0.1277 at unchanged support** (F20); severance is INERT,
  never UNREACHED - an influence defect.
- **Batched hop-2 is bitwise and 374x on fwd+bwd** (F21); gradients bound to n <= 64.
- **G1 owed before any name.** Six novelty claims have already died here.

## Board in flight

X4 instrument -> G1 fetches -> arm A birth gates -> arm B reference pass -> M3 at 8192.
