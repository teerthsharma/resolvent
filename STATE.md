# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **1 complete, 2 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 2)

**Turn the M3 harness bind GREEN by the minimum change: move `pivot_signed` onto
the shipped `sgate` operator.** That is the smallest edit that makes the
capability harness measure the module. It will move round 2's M3 numbers, which
is correct - they were readings of a non-shipped operator, so they are re-scoped,
not corrected (the round-2 precedent: the tgate measurements REPRODUCE, they were
simply measuring the wrong object).

Then, iteration 3, add the **windowed arm on sgate with `window=8`** - built on
sgate because [READ, DONE_ARCHIVE_ROUND1:1272] F4's flatness was measured there.

## Open REDs

**M3 harness operator bind - RED [RUN, iter 1].** `pivot_signed` in
`scale/m3_capability.py` is `tgate`, which ships nowhere. 1 failed, 4 passed;
must-fire control fires. R5 and R8 remain STRUCK before build (0/20000).

## Carried, and load-bearing

- **The withdrawn claim.** *"Routing is worse than dense"* is unsupported:
  bootstrap 95% CI **[-0.7497, +0.2651]** includes zero. Do not repeat it.
- **The twin test was vacuous** and is rebuilt with a NOT-VACUOUS third arm.
- **The ECDF has zero gradient a.e.** R7 is not trainable as written; the soft
  surrogate is sign-blind natively, so the rebuilt twin test runs on the SOFT arm
  before any training number is believed.
- **TACTiS is owed as a G1 fetch** before the name "Copula Attention" is written.
- **Nine structure instruments failed; three value instruments never have.**
  Run `inspector.py`; do not retype its checks into shell.

## Board in flight

Phase 0 M3w -> Phase 1 R7 (R6 behind it) -> Phase 2 M6, M7 -> Phase 3 write-up.
