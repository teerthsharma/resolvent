# State

**Updated: 2026-08-25 - ROUND 3 open under CEQ v5. Iteration 0 (the room) complete.**

| field | value |
|---|---|
| round | **3** - CEQ v5, 30 iterations, promise `SCALEFREE` |
| iteration | **6 complete, 7 next** |
| phase | **Phase 0 - M3 on the windowed arm. Capability before statistics.** |
| calibration | GREEN [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| inspector | `python inspector.py` - 8 checks, 8 must-fire controls, exits nonzero if any control stays SILENT |
| repo | https://github.com/teerthsharma/resolvent (private) |

## THE ONE NEXT ACTION (iteration 7)

**Run the frustration / switching-class integrity audit on every signed arm.**
Iteration 6 found sgate is **signed in name only** at the entries that matter:
99.98% of its top-k-by-magnitude entries are positive, pairwise correlation
0.9993. The arsenal has carried this exact check since round 1 - *"frustration ~0
= signed-in-name-only; this item is itself a kill"* - and it has never been run.

If sgate's frustration index is ~0, then every signed-vs-unsigned comparison in
three rounds compared an unsigned arm against an **effectively unsigned** arm,
and the whole signedness axis was never actually exercised. That would explain
Cameron's open contradiction - softmax 15/512 against the signed operator's
0/512 - without any appeal to trainability.

Cheap: it is a property of the operator at init, no training, CPU minutes.
RED-first: a deliberately balanced signed matrix must read HIGH frustration, and
an all-positive one must read ~0.

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
