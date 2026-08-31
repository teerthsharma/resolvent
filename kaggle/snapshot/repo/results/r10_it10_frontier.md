# R10 it.10 — frontier n*(steps) by isotonic regression, and the NRMSE = 1.0 crossing

JUPITER's half. No training was run and no allocation was made; this is a re-read of
cells already on disk.

**Sources**, pinned by hash because the it.8 journals are append-only and one of them was
still being written at read time (see Limits):

| file | lines | sha256 (first 16) |
|---|---|---|
| `results/r10_it8_capacity_softmax_t2.jsonl`  | 35 | `f5ea87693684743d` |
| `results/r10_it8_capacity_softmax_t8.jsonl`  | 43 | `a45e93658ca53b99` |
| `results/r10_it8_capacity_softmax_t32.jsonl` | 35 | `09e0f75d37db90ba` |

Read at 2026-08-30T21:46:36. Machine-readable form of everything below:
`results/r10_it10_frontier.jsonl` (92 records). Regenerate with
`python scale/r10_it10_frontier.py`.

**Isotonic backend: `sklearn.isotonic.IsotonicRegression 1.9.0`, `increasing=False`.**
sklearn is installed on this box, so the in-repo pool-adjacent-violators fallback was not
used for the reported fits. Both backends are defined under distinct names and the
selected one is asserted against the other on a sequence that does violate monotonicity
(`scale/r10_it10_frontier.py`, the `_VX/_VY` assertions), so the sentence above is checked
rather than assumed.

---

## 0. Resolution floor — read this before any difference below

The harness is deterministic given (seed, threads) but not across thread counts. Three
cells were run twice at the same seed and different thread counts, and the float reduction
order alone moves the reading:

| t* | n | steps | seed | threads 6 | threads 8 | drift | rows |
|---|---|---|---|---|---|---|---|
| 2  | 2048 | 150 | 0 | 0.9690874072 | 0.9714324269 | **2.345e-03** | `t2.jsonl:4`, `t2.jsonl:17` |
| 8  | 2048 | 150 | 0 | 1.1280431664 | 1.1285342332 | 4.911e-04 | `t8.jsonl:4`, `t8.jsonl:17` |
| 32 | 2048 | 150 | 0 | 1.1581741681 | 1.1581668575 | 7.311e-06 | `t32.jsonl:4`, `t32.jsonl:17` |

**Noise floor = 2.345e-03 absolute NRMSE**, the largest of the three. The grid was swept at
mixed thread counts (8, 6, 6, 12, 12), so any two cells differenced below carry this floor.
Every difference this report treats as signal is stated as a multiple of it.

---

## 1. Isotonic fit at steps = 150

steps=150 is the only rung with all three n. Fitted as NRMSE against `log2(n_train)`,
monotone decreasing.

| t* | n=2048 | n=8192 | n=32768 | fit changed anything? |
|---|---|---|---|---|
| 2  | 0.952349 (8 seeds) | 0.888001 (1 seed) | 0.874834 (1 seed) | **no — identity** |
| 8  | 1.124057 (8 seeds) | 1.025127 (1 seed) | 0.974218 (2 seeds) | **no — identity** |
| 32 | 1.153720 (8 seeds) | 1.027155 (1 seed) | 1.006066 (1 seed) | **no — identity** |

`max_abs_change = 0.0` for all three, and for both the primary and the seed-0-only variant.
**The isotonic regression performed no pooling and returned its input unchanged.** It did no
work. Nothing in the crossing numbers below comes from the fit; they come from linear
interpolation between the raw measured points. Reporting the fit as if it had shaped the
curve would be false.

The n=2048 column reproduces `results/r10_it8_table.txt` exactly (0.9523 / 1.1241 / 1.1537),
which is the cross-check that this aggregation matches the it.8 aggregation.

Seed counts differ down each row and are stated per cell. n=2048 is a mean over 8 seeds
(seeds 0–7, `t*.jsonl:17,19-25`); n=8192 is a single run at seed 0 (`t*.jsonl:11`); n=32768
is a single run at seed 0 for t*=2 and t*=32, and a mean over 2 seeds for t*=8 (see §5).

---

## 2. The crossing of NRMSE = 1.0

Interpolated **in log2(n)**, i.e. linearly in octaves of training rows, not linearly in n.

| t* | n* | status | basis |
|---|---|---|---|
| 2  | **below the grid** | not a number | 0.952349 < 1.0 already at the smallest measured n=2048 |
| 8  | **≈ 16,238** (log2 = 13.99) | **interpolated**, bracketed | between measured n=8192 (1.025127) and n=32768 (0.974218) |
| 32 | **≈ 48,823** (log2 = 15.58) | **EXTRAPOLATED, 0.575 octaves beyond the grid** | last measured segment only; no measured cell brackets it |

**t\* = 2.** The curve is below the bar at every measured n. The crossing lies below
n=2048 and is *not* reported as a backwards-extrapolated number; the measured statement is
"already crossed at the smallest n run."

**t\* = 8.** n* ≈ 16,238 sits inside the measured bracket [8192, 32768], so this is
interpolation and nothing else. Seed-0-only sensitivity: n* ≈ 15,854 (2.4% lower). The
curve is convex in log2(n) (second difference +0.0480), and a convex decreasing curve lies
below its own chord, so the chord estimate is an upper bound on n* under that curvature.

**t\* = 32. The curve has not crossed anywhere in the measured grid, and the extrapolated
number should not be trusted as a crossing.** Three reasons, all measured:

1. It is 0.575 octaves past the largest n ever run, on a two-point slope of
   −0.010544 NRMSE per octave taken from the single segment 8192 → 32768.
2. The margin being extrapolated to zero is 1.006066 − 1.0 = **6.066e-03, only 2.59x the
   2.345e-03 noise floor of §0**, and the two cells defining that segment were run at
   different thread counts (6 and 12).
3. The curve is decelerating, and the deceleration is measured, not assumed. The drop per
   two octaves falls from 0.126566 (2048→8192) to 0.021089 (8192→32768), a ratio of 0.167.
   Continue that geometric decrement and the curve **asymptotes at 1.00185 and never reaches
   1.0 at any n**. The gap from that asymptote to the bar, 1.85e-03, is *below* the noise
   floor — so the measured cells cannot even distinguish "asymptotes just above the bar"
   from "asymptotes just below it."

The linear-in-log2 n* ≈ 48,823 is therefore best read as a **lower bound on n\* if a crossing
exists at all**, and the honest answer for t*=32 is in §5: the question cannot be settled
from the measured cells. Context that makes this tight rather than surprising: the 1-hop
NRMSE ceiling at t*=32 is 0.984251 (`t32.jsonl:2`), so the entire available headroom below
the bar is 0.0157 — the t*=32 crossing, if real, has to happen inside a band 6.7x the noise
floor wide.

### n*(steps) as a function — mostly not measured

| steps | t*=2 | t*=8 | t*=32 | n rungs measured |
|---|---|---|---|---|
| 150  | below grid | 16,238 interp. | 48,823 extrap. (0.58 oct) | 3 |
| 600  | 8,615 extrap. (0.07 oct) | 13,115 extrap. (0.68 oct) | 16,173 extrap. (0.98 oct) | 2 — n=32768 **not measured** |
| 2400 | **not measured** | **not measured** | **not measured** | 1 |
| 9600 | **not measured** | **not measured** | **not measured** | 1 |

At steps=600 only n=2048 and n=8192 were run (n=32768 at 600 was dropped as unaffordable,
`t*.jsonl:33,35`), so those three numbers are two-point straight lines with no curvature check
and no bracketing cell. At steps=2400 and 9600 only n=2048 exists; a single point admits no
monotone fit and no crossing, and those six entries read "not measured", not "never crosses".

---

## 3. The monotonicity assumption, tested rather than inherited

Isotonic regression here assumes more-data-never-hurts. The assumption was checked against
the data at steps=150, per t*:

| t* | 2048 → 8192 | 8192 → 32768 | violations | smallest gap ÷ noise floor |
|---|---|---|---|---|
| 2  | −0.064347 | −0.013168 | **0** | 5.62x |
| 8  | −0.098930 | −0.050909 | **0** | 21.71x |
| 32 | −0.126566 | −0.021089 | **0** | 8.99x |

**Zero violations. All six observed steps=150 sequences (three t*, each in the 8-seed and
the seed-0-only variant) are already strictly monotone decreasing in n**, as are the three
steps=600 sequences, one gap each over the only two n measured at that rung. Maximum violation
magnitude is exactly 0.0 in every case. The assumption held, so — restating §1 — the
isotonic fit is the identity map and contributed nothing to any number in this document.

Every gap clears the noise floor, the tightest being t*=2's 8192→32768 gap of 0.013168 at
5.62x floor, and those two cells were run at 6 and 12 threads respectively.

---

## 4. The steps axis

Sign of dNRMSE/d(steps) at every (t*, n) where two or more step rungs were measured.
**Confirmed, without exception: NRMSE rises with steps — more optimisation is monotonically
worse — in 12 of 12 adjacent step pairs across 6 of 6 eligible cells.**

| t* | n | 150→600 | 600→2400 | 2400→9600 | sign |
|---|---|---|---|---|---|
| 2  | 2048 | +0.381373 (163x) | +0.276804 (118x) | +0.076604 (33x) | **+ worse** |
| 2  | 8192 | +0.123684 (53x) | not measured | not measured | **+ worse** |
| 8  | 2048 | +0.372992 (159x) | +0.423431 (181x) | +0.333638 (142x) | **+ worse** |
| 8  | 8192 | +0.100836 (43x) | not measured | not measured | **+ worse** |
| 32 | 2048 | +0.193423 (83x) | +0.646806 (276x) | +0.244458 (104x) | **+ worse** |
| 32 | 8192 | +0.087107 (37x) | not measured | not measured | **+ worse** |
| 2, 8, 32 | 32768 | **not measured** | **not measured** | **not measured** | no slope computable |

Multiples in parentheses are the delta over the 2.345e-03 noise floor; the smallest is 33x,
so no pair is anywhere near the resolution limit. Rows at n=2048 use the 8-seed mean at
steps=150 and single seed-0 runs at 600/2400/9600, which if anything understates the rise,
since the 8-seed mean at 150 is the better-estimated end of each row.

At n=32768 only steps=150 was ever run for all three t* (600/2400/9600 dropped as
unaffordable, `t*.jsonl:33,35`). **The steps axis is untested at the largest n** — that is
absence of measurement, not a passing reading.

The direction is unambiguous over 1.5 decades of steps, and it inverts the frontier's usual
shape: n*(steps) is *increasing*, so more optimisation demands more rows rather than fewer.
This refutes the it.8 VENUS filing's frontier in direction, on measured cells: that filing
predicted t*=8 would "never" cross at steps=150 (measured 0.974218 at n=32768, crossed) and
would cross at n=2048 by steps=9600 (measured 2.254118, the worst cell in the t*=8 journal).

The mechanism is visible in the same rows: `train_nrmse` falls monotonically as `eval_nrmse`
rises — at t*=8, n=2048 it runs 0.865999 → 0.559578 → 0.301694 → 0.125027 across the four
rungs while eval goes 1.128534 → 1.497049 → 1.920480 → 2.254118 (`t8.jsonl:4-7`).

---

## 5. What the measured cells cannot answer, and the one cell that would

**The t\* = 32 crossing cannot be answered from the measured cells.** The curve is above the
bar at every n that was run, the extrapolation runs 0.575 octaves past the grid, the margin
being extrapolated away is 2.59x the noise floor, and the curvature-aware continuation says
there may be no crossing at all. Both readings are consistent with the data on disk.

**The single cell that would answer it: t\* = 32, n_train = 65536, steps = 150, softmax arm,
same (s=64, d=24, n_eval=4096, lr=0.02, d_model=16) config.** One octave beyond the grid, it
brackets the linear estimate n* ≈ 48,823 from above: an eval NRMSE below 1.0 confirms a
crossing and pins n* inside [32768, 65536]; a reading at or above 1.0 kills the linear
extrapolation and promotes the deceleration reading. Cost is roughly 2x the measured
n=32768 cell, whose wall time on this box ranged 312–681 s at 12 threads across the seven
n=32768 journal rows (`t2.jsonl:32,34`, `t8.jsonl:32,34,39`, `t32.jsonl:32,34`), so on the order of 10–25 minutes, load-dependent. It was not run
here: the round forbids it while another agent holds the box's RAM.

Two further cells would matter but were not requested and are named only so their absence is
not read as a passing result: (t*=2 or 8 or 32, n=32768, steps=600) would give the steps axis
a second rung at the largest n, and (t*=8 or 32, n=8192, steps=2400) would give the n*(steps)
frontier a second point at any rung above 600.

---

## Limits

The n=2048 points at steps=150 are means over 8 seeds while every other point in the
steps=150 row is 1 seed (2 for t*=8 at n=32768), so each curve mixes an 8-seed estimate at
its left end with single-run points to its right; the seed-0-only variant in
`r10_it10_frontier.jsonl` (`variant: "seed0_only"`) is the apples-to-apples version, and it
moves t*=8's n* by 2.4% and changes no verdict. `boot_lo`/`boot_hi` in the journals resample
eval examples inside one trained model and are not seed intervals; they are not used as
evidence about seed variability anywhere above, and no interval is quoted at n=32768. At
n=32768 the duplicate journal rows for t*=2, t*=8 seed 0, and t*=32 are bit-identical in
`eval_nrmse`, `boot_lo` and `boot_hi` and differ only in `secs` (624.46/617.67, 607.76/602.85,
679.95/681.47) — one run journaled twice by overlapping writers, not replicates, and they
are deduplicated rather than averaged. The t*=8 journal was live during this analysis: a
seed sweep at n=32768 landed seed 1 (0.976063, `t8.jsonl:39`) and had opened a header for
seeds 4–7 at 21:41:30 when the file was hashed, so the t*=8 n=32768 mean of 0.974218 over 2
seeds will move as that sweep completes and should be recomputed against a fresh hash; the
2-seed spread there is 3.691e-03, which already exceeds the cross-thread noise floor and is
a genuine seed effect, but two seeds do not support an interval and none is given. That
same t*=8 journal declares seeds [1,2,3] in its 21:36:03 header but carries a cell for seed 1
only before the next header opens, so seeds 2 and 3 at n=32768 are not measured rather than
missing-because-they-failed. All step-axis and crossing arithmetic differences two cells swept
at different thread counts, whose float-reduction drift is the 2.345e-03 floor of §0; the
only quantity in this report that does not comfortably clear it is the t*=32 margin to the
bar, which is precisely the quantity §5 declines to answer. Cost figures in §5 are wall-clock
seconds from a shared, loaded box at differing thread counts and are indicative only.
