# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **5 complete, 6 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 6)

**Test the sign-independence assumption behind `A_8 = 2.187500`. It is mine and
it is owed.**

That constant was enumerated over 256 sign patterns **assuming Rademacher
independence**. Under top-k salience selection the surviving signs are
**concomitants of order statistics** and may be dependent. The v5 prompt already
requires this - *"Sign-independence is TESTED, not assumed"* - and the
requirement was written and then not honoured.

RED-first. Measure the joint law of (rank, sign) among selected tokens:
  * `E|B_k|` under the MEASURED sign law vs the independent enumeration;
  * a must-fire control - a deliberately correlated sign pattern must move it
    VISIBLY, or the test cannot detect dependence and its green means nothing.

If dependence is found, `A_k` is re-derived under the measured copula and the
boundedness claim survives regardless, since `|B_k| <= k` is sign-free - only the
constant moves. **Pin whatever comes out at abs=5e-7.**

## Open REDs

**None.** The M3 harness bind went GREEN this iteration (5 passed, must-fire
control firing). R5 and R8 remain STRUCK before build (0/20000 event change).

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
