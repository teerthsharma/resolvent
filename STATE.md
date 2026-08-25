# State

**Updated: 2026-08-25 - ROUND 4 open under CEQ v6', the chosen-sign round.**

| field | value |
|---|---|
| round | **4** - CEQ v6', promise `CHOSENSIGN` |
| iteration | **6 (round 4) complete, 7 next** |
| phase | **X4 instrument first - everything downstream reads through it** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor, not token predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 4, iteration 7)

**Continue the census - it resumes, so this is one more bucket, not a restart.**

`python scale/journal_census.py --budget 420` at `OMP_NUM_THREADS=1`, then the
same at 2. It skips what is already journalled, so each call is pure progress.

**The sharpest single question is `s128`**, which drifted at 1 thread here and at
Wilson's count. If it drifts at 1, 2, 4 and 8, it is not reduction order and that
unit's journal is stale - which is a different defect with a different remedy
(re-journal that unit, do not re-pin a thread count). That is four cheap runs of
one unit and it can go in the same bucket.

**Hold the distinction that is already earned:** `rate` has matched on every unit
at every count tried. Any statement of the form "the journal drifts" must carry
"in `sigma`/`term`, not in `rate`", or it impeaches numbers that are sound.

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
