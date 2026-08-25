# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **3 complete, 4 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 4)

**Take the first capability reading, and take SOFTMAX first.** The harness is
now correct: four arms, all on shipped operators, all at n_params=4769, bar
already calibrated RED-first (`predict_the_mean` 1.000000, `payload_only`
1.414204 failing, `oracle` 0.000000).

Run at the SMALL setting first (s=64, d=24) to reproduce the harness end to end
on the shipped operator, with **softmax written to `results/` before any signed
arm runs** so the timestamps prove the order. That reading is not M3 - M3 needs
d in {256,512,1024} and 5 seeds - it is the check that the corrected harness
still runs and that the bar still fires.

**Pre-register the reading before running it.** Round 2's habit: on the old
tgate harness every arm sat ABOVE the 1.0 bar (best 1.342215, 34% worse than
predicting the mean). If the sgate arms also sit above it, that is Outcome B for
Phase 0 - flatness does not produce capability - and it must be read that way
rather than as a tuning problem.

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
