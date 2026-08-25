# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **2 complete, 3 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 3)

**Add the windowed arm - `windowed_signed`, on `sgate` with `window=8`** - to
`m3_capability.ARMS`. It is Phase 0's whole subject and the harness has never
carried it, which is precisely why F4 has never been capability-tested.

Built on **sgate**, not tgate: [READ, DONE_ARCHIVE_ROUND1:1272] F4's flatness
was measured on the shipped operator. `_causal_sgate_operator` already takes
`window` natively, so this is an argument, not new operator code.

Its hop-2 term is **dense within the band** (`a @ a`), NOT `pivot_hop2` - the
claim is *windowed signed multi-hop*, and routing through pivots is a different
arm that is already dead. The G3 bind (`hops=0` bitwise identity) ships with it,
and `test_the_windowed_arm_that_phase_0_needs_does_not_exist_yet` must be
UPDATED rather than deleted when it lands - it is written to fail on arrival.

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
