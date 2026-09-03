# V20 R15 it.17 — MERCURY (LESTRADE)

Deliverable: `tests/mercury/arena_rig.py` such that
`tests/mercury/test_v20_r15_it15_arena_rig.py` runs green.

**Status: GREEN. 17 passed in 0.35s.**

`[RUN]` `python -m pytest tests/mercury/test_v20_r15_it15_arena_rig.py -q`
→ `17 passed in 0.35s`

## Inherited state
`tests/mercury/test_v20_r15_it15_arena_rig.py` survived it.15 at 6,601 bytes,
RED by `ImportError` on `tests.mercury.arena_rig`. That was the captured RED.
The module was written to the surviving test, not the reverse. Two constants in
the test were retyped off the wrong corpus at it.15; both are corrected below
with the defect named in place, and neither touches the docstring design.

## The two controls

| control | `[RUN]` | result |
|---|---|---|
| **MUST-FIRE** | `gpu_seconds_to_floor(plant_crossing(at=2, secs=(1,2,4,8,16)))` | `value=7.0`, `first_crossing_index=2`, `why=None` |
| **MUST-NOT-FIRE** (flat) | `gpu_seconds_to_floor(plant_non_crossing(secs=(1,2,4,8,16)))` | `value=None`, `index=None`, why = *"no cell crosses floor_1=0.7071 under the runner's own rule (m + half < floor1, v15_r1.py:909): 0 of 5 cells. The instrument declines rather than interpolate a crossing that is not in the data."* |
| **MUST-NOT-FIRE** (near miss) | `plant_non_crossing(secs=(1,2), near_miss=True)` | every `eval_nrmse < FLOOR_1` = `True`; `any(crosses)` = `False`; `value=None` |

The near miss is the sharp negative: an `eval_nrmse < floor` instrument fires
there and is wrong. The two plants differ in the bootstrap width alone, which
is what makes the pair a control rather than two unrelated fixtures.

## The rule, taken from the runner and not retyped
`crosses(cell) = bool(cell["boot_hi"] < FLOOR_1)` — the per-cell reading of
`scripts/v15_r1.py:909` `crosses=bool(m + half < floor1)`, where the cell's
interval is the bootstrap. It reproduces the journalled counts exactly:
`arm_pl` **12 of 16**, `arm_smprime` **1 of 16**. `FLOOR_1 = sqrt((2-1)/2)` is
recomputed from `T_STAR`, not typed as `0.7071`.

## Both clause-(1) readings, always
`[RUN]` `cp_lower_both_tails(12, 16)`

| reading | value | verdict at target 0.5 |
|---|---|---|
| two-sided (α/2 = 0.025) | `0.476229` | **FAILS** by `0.0238` |
| one-sided (α = 0.05) | `0.515604` | **CLEARS** |
| ruling | — | **`UNRULED`** |

Both ship on every row, labelled, so `⟨CLAUSE_1_TAIL⟩` costs no re-run when it
is ruled **and no office can pick the tail after seeing which wins** (`M-2`).
The inverse incomplete beta is computed in-module by Lentz continued fraction
plus bisection — no scipy dependency for one number. Matches
`scipy.stats.beta.ppf` to 6 dp.

## The run-order confound, in every row's own output
`arena_row(...)["run_order_confound"]` is emitted on every row and names all
three mechanisms: run order (`+0.7029` published, `+0.6971` this loader),
stronger than the gate correlation `+0.5197` it must be separated from; the
`early_warning` arm-dependent forward passes inside the timed window
(`scripts/v15_r1.py:712,:747`), which confound `secs` with **arm** directly;
and the absent `torch.cuda.synchronize()`.

Measured, not quoted:

| pair | ρ | what it measures |
|---|---|---|
| seed vs `secs`, within `arm_smprime` (n=16) | **`+0.6971`** | run order |
| index-across-arms vs `secs` (n=40) | **`-0.1224`** | the **arm**, not run order |

The rig asserts both and asserts they differ by more than 0.5. The published
`+0.7029` reproduces only under a **fresh-file-first** loader that exists
nowhere in the tree; every committed loader is retake-first
(`tests/jupiter/test_v20_r15_it12_constants.py:61-69`) and gives `+0.6971`.
Both figures are carried; neither is preferred here.

## `None` with a reason, never a plausible number
Four contract columns have no object, and each carries its own text (>20 chars,
asserted):

| column | reason |
|---|---|
| `dist_to_skyline` | no v15/v16 scan-skyline module exists (R-SKY); `None` on all 40 banked cells |
| `w1_to_oracle` | **both wings emit a point prediction**, so there is no state distribution for a W1 distance to have as its object |
| `peak_bytes` | `peak_wset_gib` is journalled once per run at `t=wall`, host-wide, never per cell |
| `cert_grade` | neither arm emits a mask certificate in this bed |

`w1_to_oracle` is the one that stops the leap chasing a ghost at it.35.

## Two defects found in this office's own it.15 test file

| # | defect | mechanism | fix |
|---|---|---|---|
| **D-1** | `assert len(cells) == 34` | `34` is the it.8 **undeduplicated** `retake + it6` record count (`tests/jupiter/test_v20_r15_it8_q4_q5.py:60-62`, *"32 distinct + seeds 0,1 measured twice"*), whose `arm_smprime` count is **18**. It cannot coexist with the `(1, 16)` assertion twelve lines above. A count retyped off a neighbouring corpus. | `40` — the bank of 16 `arm_pl` + 16 `arm_smprime` + 8 `softmax`, which `tests/mercury/phase_c_price.py:18` already calls `BANKED_40` |
| **D-2** | the rho test named **neither operand** | it correlated index-**across-arms** with `secs`, which measures the arm (`early_warning`), not run order, and returns `-0.1224` — reproducing nothing. This is the same unnamed-operand defect the Inspector filed against `V20_R15_IT13_MERCURY.md:83` (DEFECT B), repeated by this office one iteration later in a test written to bind it. | operands named: seed vs `secs` within one arm; the across-arm reading kept alongside as the confound demonstration |

## Limits
The `+0.6971` / `+0.7029` split is unresolved and is not resolved here: the
published figure's loader is not in the tree, so the rig binds the loader the
repo actually uses and carries the published one as a labelled string. `n=16`
is sixteen correlated readings of one eval condition, not sixteen Bernoulli
trials (`V20_R15_IT9_MARS.md:360-368`), so the CP figures are emitted as
priced readings and not as a licence to rule — `ruling` is `UNRULED` on every
row by construction. `dist_to_floor` is the arm mean of `eval_nrmse - floor_1`
and inherits every `secs`-independent caveat of the cells themselves. No GPU
was touched, nothing was uploaded, `scripts/v15_r1.py` is unmodified
(`instrument_hash 5d41a63d…` intact), `results/v17k_r4_retake.jsonl` is intact
at sha256 `26fb180b…` (asserted by the rig's own last test), and no git write
was made.
