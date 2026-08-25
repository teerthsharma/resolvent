# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **7 complete, 8 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 8)

**Repair the M3 bar, because it is the gate that guards the round and two of its
three checks are algebraic identities.**

Chase [RED, 5 failed / 9 passed]: `predict_the_mean = nrmse(y.mean(), y)` is
identically 1.0 and `oracle = nrmse(oracle(x,f,p), y)` is `nrmse(t,t)`,
identically 0.0 - **y was produced by that same call**. Only `payload_only`
reads the task, and the bar printed **BAR CALIBRATED** on a label with zero
flipper dependence. That is *"zero BY CONSTRUCTION mapped to GREEN"* inside the
gate that decides Phase 0.

The repair needs a **model-level positive control**: something TRAINED that must
pass, so that "arm failed" and "harness cannot produce a pass" stop being the
same printout. Today no arm has ever passed - softmax 0.581/1.477, pivot_signed
0.127/1.389, pivot_unsigned 0.584/1.501 - and the oracle is an identity, not a
model.

**The windowed arm's geometry is NOT a bug to fix by widening w.** Reach `2w=16`
against `d=24..54` is a fact about F4's construction. Either the task distance
comes down inside `2w`, or the arm gets depth (L layers reach `L*w`) - and that
second option is Cameron's composition question, still running.

## Open REDs

**M3 bar - RED [Chase, r3 iter 7].** Two of three checks are identities; the bar
calibrates on a task with no long-range dependence. No model-level positive
control exists.

**M3w BLOCKED.** `windowed_signed` cannot see the flipper at any distance in the
recorded sweep - `d(out)/d(x[flipper]) = 0.0` exactly.

R5 and R8 remain STRUCK before build (0/20000 event change).

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
