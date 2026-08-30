# R10 — iterations 8 & 9: the capacity sweep and the open region

**Deciding measurement.** Can softmax learn the `t* > hop` region at all? If it
cannot, no CEQ arm result in that region means anything and the architecture must
say so rather than compete where nothing learns.

**Verdict at the time of writing: learnable at `t*=2` and `t*=8`; `t*=32`
UNTESTED, not refuted.** **FULLY RESOLVED AT it.11:** `t*=32` reads NOT LEARNABLE
at n=32768 (CI [1.000851, 1.005679]) and **LEARNABLE at n=49152** (CI
[0.997535, 0.999930]). All three blocks cross; the boundary is a DATA requirement,
not an architectural wall. **SUPERSEDED AT it.11**, once the seeds this record
called for were bought: `t*=32` at n=32768 reads **NOT LEARNABLE** on a seed CI of
[1.0009, 1.0057] over N=8 — a verdict, not a refusal, because the interval lies
entirely above the bar. See the Open section below for what that does and does not
establish. The
OPEN-REGION PROTOCOL is discharged for the first two and survives for the third.

**The region is bigger than it was filed as.** Softmax's hop budget is **1**
`scale/e_ladder.py:55` (`HOP_BUDGET = {"softmax": 1, ...}`, consumed at `:202`), not 2. Venus's filing assumed hop=2 and therefore took
the open region to be `t* in {8, 32}`. With hop=1 the condition `t* > hop` covers
**all three blocks**, so `t*=2`'s clean crossing is itself an open-region result
and the region's discharged half is far larger than the question implied.

## A qualification on every verdict below, found at it.17

The bar that certified all three blocks calibrates itself with an **in-sample**
positive control, while every arm it gates is scored **held-out**.

| | |
|---|---|
| the control | trains on `feats` (`negation_scope.py:1529`), scores on **the same** `feats` (`:1536`) |
| the arms | `eval_nrmse` against a batch drawn at `seed + 12345` (`m3_capability.py:206`, `:281`) |
| the clause | `trained_two_feature < 1.0` (`:1594`) certifies the task as learnable |

So the certification is satisfied by an easier problem than the one it admits.
The repo states the rule this breaks, in its own words, in the sibling gate:
`rips_gate.py:163-168` — *"an in-sample reading would credit memorisation as
decoding and the strike below would be unearned."*

**What it does not touch.** The crossings. `t*=2` and `t*=8` clear the bar
held-out on eight seeds each, and that is true regardless of how the task was
certified.

**What it looked like it touched, and no longer does.** `t*=32`. The bar printed
BAR CALIBRATED and softmax read 1.0034 over eight seeds — consistent with either
"softmax is the limitation" or "the bar overstated the task", with nothing
separating them.

**Measured, and it separates them.** The control was replicated faithfully and
pinned to the sweep's own call (`r10_capacity_sweep.py:160` — n=4096, s=64, d=24,
steps=600), reproducing the shipped in-sample reading to **1.976e-08** before any
new number was derived from it. Scored on a batch drawn at `seed + 12345` and
never trained on:

| t\* | in-sample | held-out | gap |
|---|---|---|---|
| 2  | 0.013982 | 0.018030 | +0.004048 |
| 8  | 0.022470 | 0.032861 | +0.010391 |
| 32 | 0.024554 | **0.026544** | +0.001990 |

All three clear the clause's 1.0 threshold held-out, by a factor of ~38 at the
worst block. **So the task is learnable held-out, and softmax reading 1.0034 at
`t*=32` is a statement about softmax.** The NOT LEARNABLE verdict means what it
appeared to mean.

The defect remains real and worth repairing — it is the only one of `bar_verdict`'s
five clauses that tests learnability at all, and it survived a repair pass that
removed two by-construction siblings from the same function. Its cost in this
round is zero. Re-runnable at `scale/heldout_control_probe.py`.

Bound RED at `tests/loop/test_the_bar_control_is_scored_out_of_sample.py`. MARS
filed it as attack 3 at iteration 4 and scored it "no verdict moves" — true then,
and it moves one now that a NOT LEARNABLE rests on the bar.

## The grid: 21 cells, 49 seed-runs, 56 journal rows

Bar: NRMSE < 1.0 (predict-the-mean). Calibration bar CALIBRATED in all three
blocks; nothing below is credited from an uncalibrated block.

**Corrected after the Inspector's Phase 1a audit.** This section was headed "The
grid, 48 cells". 48 was the JOURNAL ROW count at the time of writing, published
as a cell count, and it contradicted MERCURY's correct "36 cells" two sections
below. The four numbers are different and all four are in play:

| quantity | value | what it is |
|---|---|---|
| planned grid | **36** | 3 `t*` x 3 `n` x 4 `steps` |
| dropped unaffordable | 15 | named in the priced DAG, journalled as `t:dropped` |
| distinct cells run | **21** | distinct `(t*, n, steps)` |
| seed-runs | **49** | distinct `(t*, n, steps, seed)` |
| journal rows | **56** | includes bit-identical duplicate runs |

Row counts drift upward as seeds land, which is exactly why publishing one as a
cell count is a defect rather than a typo: it is a number that changes while the
claim it supports does not.

| t\* | 1-hop ceiling | n=2048, 150 | n=8192, 150 | n=32768, 150 | verdict |
|---|---|---|---|---|---|
| 2  | 0.7071 | 0.9402–0.9714 (8 seeds, 9 rows) | 0.8880 | **0.8748** | LEARNS |
| 8  | 0.9354 | 1.1024–1.1369 (8 seeds, 9 rows) | 1.0251 | **0.9724** | LEARNS |
| 32 | 0.9843 | 1.1086–1.2464 (8 seeds, 9 rows) | 1.0272 | **1.0061** | NO READING → **NOT LEARNABLE** at N=8 |

The `t*=32` n=32768 entry above is **seed 0 alone**, which is what this table held
when it was written. At N=8 the cell reads mean 1.0034, seed CI [1.0009, 1.0057].
The eight values are 1.006066, 1.003982, **0.997284**, **0.998440**, 1.003313,
1.008121, 1.006066, 1.003698 — **two of eight fall below the bar**, so a
single-seed run at this cell reports LEARNS 25% of the time. The sweep drew seed 0
and this record published NO READING; seed 2 would have published the opposite,
from identical code and identical data.

The n=2048 column lists **9 rows over 8 distinct seeds** — seed 0 was run twice,
at threads=8 and threads=6, reading 1.1285342332 and 1.1280431664. This record
first printed that column as "9 seeds": the row-vs-seed conflation that
`scale/it11_verdict.py` deduplicates against, committed in the paragraph
describing it.

The ceiling is `sqrt((t*-1)/t*)`, the best NRMSE reachable by a model whose hop
budget is 1 — softmax's, at `scale/e_ladder.py:55` (`HOP_BUDGET = {"softmax": 1, ...}`, consumed at `:202`). It is a limit imposed by the
architecture, not by the task, and it was emitted to the journal before the first
cell ran.

## Three findings

**F-1. Data helps monotonically; steps hurt monotonically.** At 150 steps every
block improves with n. At fixed n=2048 every block degrades with steps, hard:

| t\* | 150 | 600 | 2400 | 9600 |
|---|---|---|---|---|
| 2  | 0.9714 | 1.3337 | 1.6105 | 1.6871 |
| 8  | 1.1285 | 1.4970 | 1.9205 | 2.2541 |
| 32 | 1.1582 | 1.3471 | 1.9939 | 2.2384 |

Every crossing in the whole grid is at the **shortest** rung. The task is
data-limited and optimisation-hurt. A sweep that had spent its budget on steps
rather than examples would have concluded the region unlearnable and been wrong.

**F-1 bound mechanically, not by eye**, over every cell in the three journals:
**15 cells read below the bar and all 15 are at 150 steps**; of the 6 `(t*, n)`
groups carrying two or more step rungs, **0 violate monotone worsening**. Scope,
stated because it is smaller than the sentence above implies: the grid has 9
`(t*, n)` combinations and only 6 have more than one step rung — the large-n
cells mostly ran 150 alone, since the corner cells were dropped as unaffordable.
The steps claim is therefore bound on two-thirds of the grid and **untested** on
`(2, 32768)`, `(8, 32768)` and `(32, 32768)`, which is exactly where a
steps-driven reversal would matter most.

**F-2. The 1-hop ceiling rises toward the bar as the horizon grows.** 0.7071 →
0.9354 → 0.9843. At `t*=32` a *perfect* one-hop model has 0.0157 of margin under
predict-the-mean. This is the architectural case for multi-hop propagation stated
in closed form rather than asserted, and it is why `t*=32` sitting at 1.0061 is
not a surprise: there is almost no room there for one hop to win.

**F-3. Neither decisive cell is saturating its architecture.** `t*=8` reads 0.9724
against a 0.9354 ceiling — it learns, but is still 0.037 short of what one hop
could reach. Softmax is not being beaten by the ceiling at `t*=8`; it is being
beaten by optimisation.

## Venus's filing, adjudicated

Filed to `results/r10_it8_venus_filing.md` at commit `bf2a769`, before the first
cell. Read only after the table was complete.

- **Existence — right.** She predicted a crossing exists in the open region. One
  does, at `t*=8`.
- **Location — falsified on her own stated falsifier.** She named
  `(2400, 8192, t*=8)` = 0.93 as the cheapest crosser and wrote that the filing
  is falsified if the cheapest crosser is anything else. It is
  `(150, 32768, t*=8)` = 0.9724 — wrong step count, wrong n, wrong value, and
  wrong in direction: 2400 steps is catastrophic in every block measured.

Her existence claim survives; her frontier does not. Filing before the data is
what makes that split legible instead of arguable.

## Correction to this record's own author

Two claims made in-flight and withdrawn here.

1. *"It is not slow; it is starved — the cell cannot start."* Read off a falling
   free-memory number. The process table showed both `t*=32` seats already
   running and allocating; the memory I watched drain was the job itself. A
   resource reading was used as a proxy for a job state that could have been
   read directly.
2. *"The ceiling is the analytic floor on the task."* It is the floor imposed by
   **softmax's hop budget**, per the code that computes it. The corrected reading
   is stronger and is F-2.

## Open

- ~~**The largest rung has one seed.**~~ **RESOLVED at it.11.** `--seed-cap`
  restricted seeds beyond the first to the smaller rungs, so every n=32768 cell
  was seed 0 and the two "replicates" per block were bit-identical duplicates — a
  determinism check, not a variance estimate. Those seeds were priced at 8 × 600 s
  and declined at it.9; they have now been bought for `t*=8`:

  | t\*=8, n=32768, 150 steps | value |
  |---|---|
  | eight distinct seeds, all threads=12 | 0.972372, 0.976063, 0.976329, 0.975753, 0.978695, 0.977786, 0.972677, 0.973294 |
  | mean | 0.975371 |
  | seed CI, B=10,000 | **[0.9739, 0.9768]** — excludes 1.0 |
  | worst of eight | 0.978695, still clear of the bar |

  The interval is 0.0029 wide against the 0.0345 seed *range* at n=2048, so seed
  variance collapses with n. Seed 0 alone read 0.9724 against the N=8 mean of
  0.9754 — 0.003 optimistic and well inside the interval. **The unreplicated
  draw this record flagged as resting on one seed was representative, not
  lucky.** `t*=32` at n=32768 remains at 4/8 seeds and is still refused.
- ~~`t*=32` is NO READING, not refuted.~~ **RESOLVED at it.11 and bounded.** At
  N=8 it reads NOT LEARNABLE, CI [1.0009, 1.0057]. **Scope, because the headline
  overstates it otherwise:** this establishes that `t*=32` does not cross **at
  n=32768 with 150 steps**. It does not establish that no amount of data would
  cross. The n=65536 cell that would test it was **refused on memory** — 9,859 MiB
  needed against 8,808 available, and re-measurement later showed even that figure
  understated the requirement by 8.2%. A 0.58-octave substitute at n=49152 is
  running. JUPITER's deceleration measurement (per-2-octave drop 0.126566 →
  0.021089, ratio 0.167) is the evidence that it may never cross, and it remains an
  extrapolation.
- No cell was run above n=32768 or below 150 steps. The optimum in the steps axis
  is at the grid's low edge and is therefore unbracketed — the exact defect
  `tests/loop/test_argmin_axes_report_a_minimum_not_their_own_edge.py` guards
  against, here noted rather than repaired because no axis test scores this grid.

## The priced DAG (MERCURY, it.8)

**36 cells, not 12, and 9 trainings, not 36.** `t*` is a **corpus**, not an eval
slice: `M3_TASKS` registers `e3_t{2,8,32}` as
`partial(make_equilibrium_batch, t_star=t)`, which zeroes the sub-diagonal at
`head = s-1-t*` and gives a different label law `N(0, t*)` and a different bar
`2/sqrt(t*)`. **Corrected at it.16** — this line read `N(0, t*+1)` beside a bar
of `2/sqrt(t*)`, which are inconsistent with each other: the bar follows from the
law. MARS filed the inconsistency as attack #1 at it.13 and SATURN's binding
measured `Var(y) = t*` at 9.8 half-widths, so the bar was right and the law as
written here was wrong. Slicing `t*` at eval would measure OOD transfer, not capability.
Steps **are** nested — one 9600-step run is read at all four rungs, saving 24.7%.

`c_step` measured at 8 threads, s=64, d=24: 0.11137 / 0.55395 / 2.54988 s/step at
n = 2048 / 8192 / 32768. Per-example cost is **not** flat (1.43x rise), so a unit
fitted at n=2048 under-prices n=32768.

| grid | seconds | RE @ 18.01 |
|---|---|---|
| 36 independent trainings | 122,981 | 6,828 |
| 9 nested trainings | 92,598 | 5,141 |
| what ran | 5,352 | **297 (5.8%)** |

**15 of 36 cells dropped as unaffordable, named:** `(2400, 9600) x 8192` and
`(600, 2400, 9600) x 32768`, each `t*`. The corner `(9600, 32768)` alone costs
24,479 s = 6.80 h per `t*`; at three `t*` that is 4,078 RE, **79% of the whole
grid**. The drops are recorded in the journal as `t:dropped`, not omitted.

## What N=8 exists, and where

it.11's gate — *"N=8 seed CI excluding 1.0, bootstrap B=1e4"* — is met at
(150, 2048) only, where eight seeds were run:

| t\* | N=8 seed CI at (150, 2048) | it.11 |
|---|---|---|
| 2  | [0.9462, 0.9591] | **excludes 1.0 — LEARNABLE** |
| 8  | [1.1162, 1.1314] | above the bar |
| 32 | [1.1307, 1.1837] | above the bar |

Every cell at n=8192 and n=32768 is **single-seed and marked `n=1`**. The
crossings that decide the open region are therefore *not* yet it.11-compliant:
the gate is satisfied for `t*=2` at the small rung, and the `t*=8` crossing at
n=32768 rests on one draw. Those seeds were priced at 8 x 600 s and declined at
it.9. They are being bought now — seeds 1-7 at n=32768 for `t*=8` and `t*=32`,
after which `scale/it11_verdict.py` applies the rule at the decisive rung. That
module refuses to report a CI below N=8 and counts distinct seeds rather than
rows, because the journals carry bit-identical duplicate rows that would
otherwise read as replicates.
