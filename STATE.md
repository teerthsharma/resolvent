# State

**Updated: 2026-08-25 - ROUND 4 open under CEQ v6', the chosen-sign round.**

| field | value |
|---|---|
| round | **4** - CEQ v6', promise `CHOSENSIGN` |
| iteration | **0 (round 4) complete, 1 next** |
| phase | **X4 instrument first - everything downstream reads through it** |
| goal | **match or SUPERSEDE self-attention**; next-equilibrium predictor, not token predictor |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | tri-state; INDETERMINATE exits nonzero |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (round 4, iteration 1)

**Build the X4 valuation instrument. Everything downstream reads through it, so
it is first and nothing is measured before it exists.**

The float instrument is provably wrong at depth: median `|grad|` **2.8e-32**, and
`floor = 1e-6` discarded **100%** of a live arm's flips (0.386719 at floor=0
against 0.000000 floored). A float comparison cannot straddle 30 orders.

Two calibration ends, both required before any reading:
  * **must reproduce** every published `floor=0` number **exactly**;
  * **must-fire** - a planted **30-order-spread** arm read correctly where the
    float instrument **provably misreads it**. Not "differently". Provably.

All prior floored numbers stay quoted as **historical instrument readings**, not
withdrawn - they were correct readings of a float instrument.

## Open REDs

**F16 - the signed arm was not signed at harness scale.** Every signed-vs-unsigned
comparison there is void as sign evidence. This is round 4's entry point, not a
defect to repair: lam=1.00 gives bulk negative fraction **0.500564** and
frustration **0.520525**.

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
