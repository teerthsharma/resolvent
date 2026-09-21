# Floors row (Cameron, CPU only)

Two floors from the same 1940s generation, applied to this project's tables.
Code: `wiener_floor.py`, `cramer_rao.py`. Board events logged to
`house-events.jsonl` as agent Cameron. No git touched, no `results/*.pt`,
no `tests/chase/deq_run.jsonl`, no `docs/canon/`, no `lean/`, no GPU.

## (1) W1 -- Wiener 1942, reproduced

`default_rng(0)`, AR(2) `a=(1.2,-0.5)`, process noise sd 0.5, observation
noise sd 0.3, N=200,000, one-step-ahead prediction of the hidden state
`x_t` (oracle uses the true noise-free `x_{t-1},x_{t-2}`; last-value and
Wiener use only the noisy observation `y_{t-1..t-p}`, never `y_t`):

| | spec | got | rel diff |
|---|---|---|---|
| last-value | 0.4594 | 0.4603 | 0.19% |
| WIENER p=8 | 0.3498 | 0.3480 | 0.50% |
| oracle | 0.2502 | 0.2506 | 0.16% |

All three MATCH (sampling noise at this N is a fraction of a percent; not
tuned to match -- this is the first and only run). The oracle number is
exactly the theoretical process-noise variance (0.25); last-value's
theoretical value is `Var(x_t - x_{t-1}) + obs_sd^2 = 0.3704 + 0.09 = 0.4604`,
which also lines up.

**p=32 extension: the gap does NOT close.** WIENER(32) MSE = 0.3480, bit for
bit the same as WIENER(8). Gap to oracle stays at 0.0974 both orders. An
AR(2)+white-observation-noise process is already fully captured by an
order-8 linear predictor; more lags buy nothing. (Finding, not a bug: don't
expect p=32 to help on beds whose true generating order is this low --
p=32 only pays off when the underlying dependency genuinely extends past 8
lags.)

## (2) L-WIENER as a reusable row

`wiener_row(series, target=None, p_values=(8,32))` in `wiener_floor.py`
solves the empirical Wiener-Hopf normal equations (Yule-Walker
autocovariance of the regressor series, cross-covariance with the target)
-- no ground-truth model required, so it works on any 1-D time-indexed
sequence, defaulting to self-prediction when there's no separate target
channel.

**It cannot be attached to either named bed as specified:**
- **Chess bed**: nurse-confirmed, `ply_idx` resets to 0 every game and "carries
  no meaning across games" (chess bed source, lines 48-52). Worse: every ply
  of one game carries the *same* one-hot outcome label -- there is no
  time-varying target within a game to predict one-step-ahead. No
  time-indexed sequence exists for a Wiener floor to sit beside.
- **MDP-CAL bed**: does not exist anywhere in `ceq/` or `ceqjepa/` (source or
  git history) -- exhaustive nurse search, zero hits.

This is itself the finding for part (2): the row was asked to fit two beds,
and neither currently supports it. The function is ready for the first bed
that actually produces a real time-indexed target series.

## (3) W2 -- Cramer-Rao table

`n_cr(p,eps) = ceil(z^2 p(1-p)/eps^2)`, `n_hoeffding(eps) = ceil(ln(2/delta)/(2 eps^2))`,
z=1.9600 (delta=0.05, matching the project's existing 95% convention).

Spec points reproduced exactly: n=2401 (CR, p=0.5) vs 4612 (Hoeffding) =
1.92x; n=865 (CR, p=0.1) vs 4612 = 5.33x.

`eps` cancels in the ratio: `ratio(p) = ln(2/delta) / (2 z^2 p(1-p))`,
symmetric in `p <-> 1-p`:

| p | ratio |
|---|---|
| 0.01 / 0.99 | 48.50x |
| 0.02 / 0.98 | 24.50x |
| 0.05 / 0.95 | 10.11x |
| 0.10 / 0.90 | 5.33x |
| 0.20 / 0.80 | 3.00x |
| 0.30 / 0.70 | 2.29x |
| 0.40 / 0.60 | 2.00x |
| 0.50 | 1.92x |

**Loosest corner: p -> 0 or 1 (rare-event probabilities), diverging.**
Hoeffding is calibrated to its own worst case (p=0.5) and is *always* at
least 1.92x looser than necessary; at the extreme probabilities this
project's committor/resolution work actually lives at (rare wins, rare
draws, rare sink states), it is 10-50x looser. Every place this project
picked an `n` from the Hoeffding table instead of the Cramer-Rao table
over-sampled, most severely for rare outcomes.

## (4) Audit: published resolution numbers vs their own Cramer-Rao floor

Ground truth pulled directly from
`tests/wilson/resolution/wil_chess400_results.json` (not from the dispatch
text -- see first finding below).

**Finding: the dispatch's own numbers don't match the published file.**
The dispatch cites "chess committor resolution 0.001469 against an oracle
ceiling of 0.10117." The file actually contains `RES = 0.0012548892290271023`
(`part2_operator_murphy.operator_committor`) and
`oracle_ceiling_RES = 0.024116693788186205` (`part1_bed_floor.max_plies_400`).
Neither number in the dispatch is the number in the file. The audit below
uses the file, not the dispatch paraphrase.

**Finding: three named result files don't exist.** `wil_res_ceiling.py`
defines `measure(300, 80, "ship_default_cap80")` and
`measure(150, 400, "control_cap400")` (the "size-sweep tiers at 300 games"
and piece-count numbers the dispatch asks to audit) but was never run --
`wil_res_ceiling.json` is absent, as are `wil_bar_can_fire.json` and
`wil_recal_race.json`. Confirmed by directory listing, matches the nurse
report. There is nothing to check a floor against; the absence *is* the
finding for that row.

**Finding: the n used for the resolution numbers is not an independent-draw
count, so both naive floor checks below are themselves too generous.**
`part2_operator_murphy` reports `RES` at `n_eval = 6000` (positions). But
the chess bed's own source confirms every ply within one game carries the
*same* one-hot outcome label -- positions inside a game are perfectly
correlated copies of a single draw, not 6000 independent Monte-Carlo
samples. Cramer-Rao/Hoeffding floors are only valid for independent draws;
the defensible `n` is `n_games`, not `n_positions`, and `n_games <= n_eval`
by a large factor (`part1_bed_floor` used exactly 1000 games to produce
308,561 rows -- a 308x inflation of raw row-count over game-count).

Numbers, both ways:

| quantity | value | n used in file | naive floor @ p=0.5 | naive verdict | floor @ n=n_games | corrected verdict |
|---|---|---|---|---|---|---|
| `operator_committor.RES` | 0.0012548892 | 6000 positions | 0.01265 | ~10x BELOW floor | n_games unknown but <=6000, floor only gets worse | further below its real floor |
| per-class `sink.res_k` | 0.0007011879 | 6000 positions | 0.01265 | BELOW floor | -- | further below |
| per-class `draw.res_k` | 0.0004429054 | 6000 positions | 0.01265 | BELOW floor | -- | further below |
| `oracle_ceiling_RES` (part1) | 0.0241166938 | 308,561 rows / 1000 games | 0.00176 (n=rows) | ~13.7x ABOVE floor | 0.0310 (n=games) | BELOW floor |

The `oracle_ceiling_RES` row is the sharpest case: which `n` you use
*flips the verdict* -- resolved by row-count, noise by game-count. Given
the bed's own documented correlation structure (one label per game, shared
across every ply), game-count is the right count, and this number is
noise with a confidence interval computed on the wrong n.

**Not noise: the bootstrap gap.** `comparison_bootstrap.gap_mean =
0.001934666269042538` over `n_boot=400` resamples has `gap_ci95 =
[0.001125, 0.003002]`, which excludes zero. That is a legitimate,
already-computed resolved result for the operator-vs-marginal *gap*
specifically (bootstrapped directly, not derived from a raw-proportion
floor) -- it is not rescued by, and does not rescue, the RES-vs-floor
calls above, which are a different question (is RES itself distinguishable
from the position-count noise floor).

## Bottom line

- Wiener instance reproduces to within 0.5% on first run; p=32 buys
  nothing over p=8 on this generating process.
- L-WIENER is built and reusable, but neither the chess bed nor an
  MDP-CAL bed currently gives it a real target to run against.
- The Cramer-Rao table matches the spec exactly; Hoeffding is 1.92x-48.5x
  looser than necessary across the probability range, worst at rare
  events.
- The audit surfaces three separate problems in the published resolution
  numbers: the dispatch's own citation doesn't match the file, three
  named result files were never produced, and the position-count `n`
  used for every RES figure overstates independence -- correcting for
  that flips `oracle_ceiling_RES` from "13.7x above floor" to "below
  floor."
