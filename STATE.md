# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **4 complete, 5 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 5)

**Iteration 5 is an Inspector pass (every 5th): `python inspector.py`.** It is
also the mandatory pre-prognosis audit for the house-mode run, so the two
coincide - re-run every test the fellows claim green, and check every finding has
a matching RED before it.

Then reconcile the four reports into ONE prognosis. Contradictions get resolved,
not averaged: Wilson's verified facts settle any factual dispute, and anything
genuinely undecided goes to Open rather than into the verdict.

**The first capability reading stays HELD until the audit clears the harness.**
The specific blocker to clear is Chase's: at s=64, w=8, hop-2 reaches 16
positions - if the task's flipper-to-payload distance exceeds that, the windowed
arm cannot see the flipper and its reading is predetermined.

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
