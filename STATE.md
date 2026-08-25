# State

**Updated: 2026-08-25 - ROUND 4 open under CEQ v6', the chosen-sign round.**

| field | value |
|---|---|
| round | **4** - CEQ v6', promise `CHOSENSIGN` |
| iteration | **5 (round 4) complete, 6 next** |
| phase | **X4 instrument first - everything downstream reads through it** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor, not token predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 4, iteration 6)

**The bucketed journal census - one unit per call, journalled as it goes, so a
timeout leaves evidence.** A whole-census attempt timed out at 10 minutes this
iteration and I stopped rather than retried; that is ADR-001's whole point and it
has now been learned twice.

For each of the 37 units, record which thread counts reproduce the journalled
value. Three outcomes and all are useful:
  * **one thread count reproduces ALL 37** -> pin it, fix `JOURNAL_THREADS`, and
    the replay instrument is sound again;
  * **different units need different counts** -> the journal was written across a
    session with varying threads, and **bitwise replay is not available as a
    single-setting check** - it must record the count per unit;
  * **some unit reproduces at NO count** -> that unit's code changed after it was
    journalled, and the journal is stale for it.

**Do not weaken the replay check while this runs.** It is one of three
instruments here that has never given a false reading, and the finding is that it
has been sampling one unit per pass out of 37 - not that it is wrong.

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
