# State

**Updated: 2026-08-25 - ROUND 4 open under CEQ v6', the chosen-sign round.**

| field | value |
|---|---|
| round | **4** - CEQ v6', promise `CHOSENSIGN` |
| iteration | **8 (round 4) complete, 9 next** |
| phase | **X4 instrument first - everything downstream reads through it** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor, not token predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 4, iteration 9)

**Back to the round's actual subject: arm A, rebuilt as an ADDITIVE BASIS OF
ORDER 2.** The instrument work is done - the replay asserts what it verifies, the
census is recorded, and `rate` is intact at all 37, so the published verdict
stands.

Arm A needs `D + D` covering `Z_v`, not `D - D`. In a causal DAG both hops point
the same way, so only sums compose - measured: `|D-D| = 56/56` while
`|D+D| = 36/57` for the same Singer set. Two corrections travel with it:
  * **`j` must stop being fixed at `s//4`.** That pins the offset at `3s/4`,
    unreachable for any bounded-offset schedule at low hop count, which is what
    made every schedule read severance 1.0000. The uniform-at-random discipline
    `carpet_probe.py:24` demands for `c` applies to `j`.
  * **read on X4**, not the float path.

Search for a small additive 2-basis mod v directly - for v = 57 a greedy or
exhaustive search over k = 8..12 offsets is cheap, and the coverage condition is
a VALUE to assert exactly as `|D-D| = v-1` was.

## Open REDs

**JOURNAL DRIFT [Wilson, r4 iter 4].** `dense_signed__at_pivots/s1024/b0` and
`/s128` do not replay bitwise. Journal mtime 12:38:59, code mtime 16:22:33.
G2-class. Census running.

**F16** - the signed arm was not signed at harness scale; round 4's entry point.

**X3 DOES NOT BIND AS STATED.** At lam=1.00 the selected set reads P(+) 0.4968,
pairwise correlation **-0.039893**. The alignment measured at iteration 6 was
**lambda, not FKG**.

**ARM A REDIRECTED.** Difference sets describe differences; causal composition
uses sums. Target is an additive basis of order 2.

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
