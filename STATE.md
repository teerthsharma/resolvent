# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **13 complete, 14 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 14)

**Test the repair Cameron named and never tried: co-prime / randomised dilations
against the severed fraction.**

Severance is a property of the **rigid power-of-two lattice**, not of bounded row
width - so the reach result (support exactly `s`, row width exactly 8) may
survive a schedule that is not a geometric series. The probe already exists
(`tests/cameron/severed_fraction.py`, exhaustive over every legal `c`), and
`log_schedule` is the one function to change.

**PRE-REGISTERED, fixed before running:**
  * co-prime dilations **lower the severed fraction WITHOUT lowering gradient
    support** -> the repair is real, and the composition route is alive again;
  * they lower severance **and** support -> **not a repair, a different arm** -
    reach was traded away, which is the tradeoff the round already refused;
  * severance is unchanged -> it is not the lattice, it is bounded row width
    itself, and the whole composition family closes.

Baselines in the same table: `[1,2,4,8]` (measured: 0.4130 at s=64, 0.5745 at
s=128) and a contiguous `[1,1,1,1]` control.

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
