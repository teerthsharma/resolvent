# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **12 complete, 13 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 13)

**Reconcile the two in-flight streams, which the policy requires and iteration 12
did not reach:** `results/r3_it11_pivot_8192.log` (signed + unsigned arms at the
budget where softmax reads eval 0.877168) and Wilson's three engineering diffs.

**If the severed fraction has an obvious repair, it is the next build after
that.** Cameron named the candidate and did not test it: severance is a property
of the RIGID per-layer power-of-two lattice, not of bounded row width - so
**overlapping, co-prime, or randomised dilations** may keep the reach result
(support exactly s, row width 8) while killing the severance. That is a cheap
one-line change to `log_schedule` and it is testable with the probe that already
exists.

Pre-register before running it: **co-prime dilations must lower the severed
fraction WITHOUT lowering gradient support**, or the trade is not a repair, it is
a different arm.

## Open REDs

**M3 bar - RED [Chase].** Two of three checks are identities; calibrates on a
task with no long-range dependence; no model-level positive control.

**M3w BLOCKED [Chase].** `d(out)/d(x[flipper]) = 0.0` exactly - reach `2w=16`
against a sweep of `d=24..54`.

**COGS capability number - SCOPE BROKEN [Foreman].** The run trained
`(0.9, 1.0, 3)`, not the parity point: row sums exactly 0, mixing 9.9x below the
softmax arm, hop-2 mass 4.04e-03. **The headline 0/512 vs 15/512 is not a
measurement of the parity operator.**

R5 and R8 remain STRUCK before build (0/20000).

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
