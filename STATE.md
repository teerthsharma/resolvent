# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **16 complete, 17 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 17)

**Re-run the 8192 reading now that the backward is ~linear - BUCKETED, per
ADR-001, which iteration 15 broke.**

The batched path turns 24.56 s per hop-2 backward into 0.066 s at n=2048, a 374x
difference, and the per-example autograd graph count was the memory cost that
killed the run. That removes the likely cause; it does not prove the run
survives.

**Bucket it.** `scale/bucket.py` exists so a death leaves a journal, and the last
attempt bypassed it and left 0 bytes in two files. Never again by hand.

Arms: `pivot_signed`, `pivot_unsigned` against softmax's already-timestamped
eval **0.877168**, CI **[0.830455, 0.924226]**. **NOT `windowed_signed`** - dead
twice over. Label the result **d=24, not M3 proper**.

If it dies again with the backward linearised, the cause is not the hop-2 loop
and the next suspect is the full-batch operator itself - `[8192,64,64]` plus
autograd saves - which is what gradient accumulation was for.

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
