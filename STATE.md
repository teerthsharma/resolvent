# State

**Updated: 2026-08-25 - ROUND 4 open under CEQ v6', the chosen-sign round.**

| field | value |
|---|---|
| round | **4** - CEQ v6', promise `CHOSENSIGN` |
| iteration | **7 (round 4) complete, 8 next** |
| phase | **X4 instrument first - everything downstream reads through it** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor, not token predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 4, iteration 8)

**Repair the replay instrument, and the repair is NOT a better constant.**

No thread count among 1, 2, 4, 8, 16, 20, 24 replays the whole journal:
`s1024/b0` needs 1 or 4, `s128` needs none of them. `JOURNAL_THREADS = 2` is
unfixable by choosing differently.

Two honest forms, and the second is what the evidence supports:
  * **per-unit thread counts recorded in the journal** - correct, but it
    re-journals 37 units against code that may itself have moved;
  * **assert on `rate` alone, report `sigma`/`term` as advisory.** `rate` has
    matched on every unit at every count tried, it is an integer, and **every
    published M2 number is built from it**. The float reductions carry no claim.

Take the second, and say plainly in the check's own docstring that it is a
WEAKER assertion than it was - a replay that no longer verifies float
reproducibility is not the same instrument, and pretending otherwise is how a
check becomes decoration.

**Then the must-fire control has to change with it**: mutating `rate` must fail,
and mutating only `sigma` must NOT - otherwise the new check would still be
claiming what it no longer verifies.

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
