# R10 — iteration 10: the frontier, and a block that may never cross

**Script task.** *"MERCURY+JUPITER: extend one octave along argmax-gradient axis;
fit frontier n\*(steps) by isotonic regression (monotone by assumption of
more-data-never-hurts) and report the first crossing NRMSE(n\*, s\*) = 1."*

**Which axis.** The steps axis is worse-going at every measured pair, so the
argmax-gradient direction is **n**. The octave extension is therefore
`n = 65536`, and the block worth spending it on is `t*=32` — the only one still
NO READING.

## The frontier (JUPITER)

Interpolation is linear **in log2(n)**, stated as such.

| t\* | n\* at steps=150 | status |
|---|---|---|
| 2  | **below the grid** | 0.952349 < 1.0 already at n=2048, the smallest n run. No number is extrapolated backwards. |
| 8  | **≈ 16,238** (log2 13.99) | **Interpolated**, bracketed by measured n=8192 (1.025127) and n=32768 (0.974218). Seed-0-only variant 15,854. |
| 32 | ≈ 48,823 (log2 15.58) | **Extrapolated 0.575 octaves past the grid**, on the single 8192→32768 segment. No cell brackets it. |

## The fit is the identity map, and says so

Monotonicity held with **max violation exactly 0.0** in all nine sequences
(3 `t*` × {8-seed, seed-0-only} at steps=150, plus three at steps=600). The
isotonic regression returned its input unchanged. Every crossing number above
comes from linear interpolation between raw points, not from the regression.

This is worth stating as a result rather than a footnote: the round's standing
hazard is an instrument that cannot fail being read as a passing instrument, and
a fit that changes nothing reported as though it had done work is that hazard in
its mildest form. Backend was `sklearn.isotonic 1.9.0`, with the PAVA fallback
kept, separately named, and asserted against sklearn on a genuinely violating
sequence so the "which backend ran" line is itself checked.

## The steps axis, bound

`dNRMSE/d(steps)` is **positive — more steps is monotonically worse — in 12 of 12
adjacent pairs across 6 of 6 eligible cells.** Smallest delta +0.087107
(`t*=32, n=8192, 150→600`), largest +0.646806 (`t*=32, n=2048, 600→2400`); every
one 33×–276× the noise floor. `train_nrmse` falls while `eval_nrmse` rises in the
same rows, so the mechanism is overfitting rather than instability.

**n=32768 carries only steps=150 for all three `t*` and is NOT MEASURED on this
axis.** No slope is reported there and none is inferred.

## t\*=32 cannot be answered from the measured cells

The margin being extrapolated to zero is **6.066e-03, only 2.59× the 2.345e-03
noise floor**. The curve decelerates measurably: per-2-octave drop
0.126566 → 0.021089, a ratio of 0.167. Continued geometrically it **asymptotes at
1.00185 and never crosses**, and that asymptote's 1.85e-03 gap to the bar is
*below* the noise floor.

So `n* ≈ 48,823` is best read as a **lower bound conditional on a crossing
existing at all**. The honest statement is that the measured cells do not
distinguish "crosses at ~49k" from "never crosses", and the distinction is
smaller than the instrument's own drift.

**The single cell that settles it: `t*=32, n_train=65536, steps=150`**, about 2×
the n=32768 wall time (measured range 312–681 s at 12 threads).

### That cell was REFUSED, and the refusal is the gate working

It waited 66 minutes for its dependency — and did **not** release on the two
occasions occupancy read `0 sweep procs` while the durable seeds-complete
condition was still false, which is the conjunction repair holding on live cases.
Then:

```
it10-octave t*=32 n=65536 (RSS EXTRAPOLATED, not measured): HOST DOES NOT FIT
-- needs 9859 MiB (7888 x 1.25 contention margin), only 8808 MiB available
REFUSED at second zero. The octave is unaffordable on this box.
```

Refused before spending, not OOM'd 13 minutes in. After killing every
non-essential process the box reached 8,370 MiB — still 1,489 short. `n=57344`
refuses too, at 8,731.

### The cell that fits is the better experiment anyway

`n=49152` prices at 7,603 MiB and **fits**. It is not a consolation rung: the
frontier fit above puts the predicted crossing at **n\* ≈ 48,823**, so `n=49152`
sits **0.7% above it** and tests that prediction directly instead of extending the
axis generically. All 8 seeds are running, so the result will be it.11-admissible
rather than another single draw.

**Labelled honestly:** 49152/32768 = 1.5× is **0.58 octaves**, not the one octave
the script asks for. The script's wording is not satisfied by this cell.

**And it was not expected to adjudicate.** The two readings of this same data —
linear-interpolation crossing at 48,823, and geometric deceleration asymptoting at
1.00185 — predict ≈1.0 and ≈1.002 at this cell, differing by ~0.002 against a
measured noise floor of 2.345e-03.

**It adjudicated anyway.** The N=8 seed CI came in at [0.997535, 0.999930] — width
0.0024, tighter than the ~0.005 anticipated, because seed variance fell 2.08×
between the rungs where 1/√n predicts 1.22×. The interval excludes 1.0, so the
deceleration reading is refuted. The caution was right in advance and the
measurement beat it.

**And the refused octave was the wrong cell anyway.** `n=65536` sits *beyond* the
crossing, and a cell beyond a crossing does not bracket it. The affordable
substitute did, because it was aimed at the prediction rather than at the next
power of two. The gate's refusal cost nothing and redirected the experiment to a
better one.

## RESOLVED: t*=32 crosses, and the prediction landed

The octave was refused; `n=49152` ran instead, at 0.58 octaves and 0.7% above the
predicted crossing. **All eight seeds, all at threads=12:**

```
0.999940  0.995362  0.998811  1.000584
1.000205  0.998849  1.000233  0.996743      5 under, 3 over
```

| cell | seed CI (N=8, B=10⁴) | verdict |
|---|---|---|
| n=32768 | [1.000851, 1.005679] | **NOT LEARNABLE** |
| n=49152 | **[0.997535, 0.999930]** | **LEARNABLE** |

**The crossing is bracketed**, and `n* ≈ 48,823` — interpolated in log2(n) from
three rungs, extrapolated 0.58 octaves past the grid — **falls inside the bracket
and within 0.7% of the measured cell.**

**The competing reading is refuted.** Geometric deceleration asymptoting at
1.00185 and never crossing does not survive a measured CI entirely below 1.0.
Both readings were filed; only one survived, which is what filing both is for.

### The verdict splits, and only one half is robust

**Robust — the crossing.** Both cells ran entirely at threads=12 (verified via
`thread_split`), so the cross-thread systematic offset **cancels in the
comparison**. The 0.0045 difference between the rungs is far above any drift.

**Not robust — the absolute reading.** The margin below the bar is **7.00e-05**
against a measured cross-thread drift of up to **2.345e-03**, thirty-three times
larger. Re-running at threads=6 or 8 could flip LEARNABLE to NOT LEARNABLE.

**A category error worth naming, because this record's author nearly made it:**
comparing 7e-5 against 2.345e-3 as if they were the same kind of uncertainty.
They are not. At a fixed thread count the drift is a *systematic offset moving all
eight seeds together*, not random error on their mean — so the seed CI is the
correct uncertainty for the mean, **and** the whole result sits on a
thread-count-dependent baseline. Both facts hold at once.

## Correction: the noise floor was quoted from one pair of three

This record's author reported the cross-thread drift as **4.911e-04** and
concluded that `t*=32`'s margin is 12× it and that no verdict turns on it. There
are three cross-thread pairs, not one:

| pair | drift |
|---|---|
| `t*=2, n=2048, seed 0`, threads 6 vs 8 | **2.345e-03** |
| `t*=8, n=2048, seed 0`, threads 6 vs 8 | 4.911e-04 ← the quoted one |
| `t*=32, n=2048, seed 0`, threads 6 vs 8 | 7.311e-06 |

Max is 2.345e-03, **4.8× the quoted figure**; `t*=32`'s margin is **2.59×** that
floor, not 12×. A single measurement was presented as a population statistic —
the same shape as the N=1 seed defect this round was concurrently objecting to,
committed in the sentence objecting to it. Caught by JUPITER, verified
independently before being accepted.

A second correction, in this author's favour and recorded anyway: of two
static-analysis defects flagged to JUPITER mid-run, only one was real. The `None`
unpack would have crashed on any missing steps=150 cell and was fixed. The
"redeclared `isofit`" was a guarded `except ImportError` fallback that never
shadowed anything — a linter hit reported as a defect without checking whether
the second binding was reachable.

## Venus, again

The steps axis makes `n*(steps)` **increasing**, refuting the it.8 filing's
frontier in direction on measured cells: it predicted `t*=8` would *never* cross
at steps=150 (measured 0.974218 — crossed), and would cross at n=2048 by
steps=9600 (measured 2.254118, the worst cell in that journal).

## Open

- `t*=32` is undecided and may be undecidable at this noise level. The octave
  cell is queued.
- The `t*=8` crossing is interpolated between two measured cells, which is the
  strongest status any crossing in this grid has. It still rests on few seeds at
  n=32768; it.11 governs.
- The t8 journal declares seeds `[1,2,3]` in one header while carrying a cell for
  seed 1 only — a run in progress at read time, noted so a later reader does not
  mistake the header for a completed set.
