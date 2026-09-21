# C25 -- scorer invariance

Owner: Chase. Files: `c25_scorer.py` (the scorer + wrapper), `c25_rescore.py`
(re-score the chess row), `c25_ceiling.py` (the ceiling question),
`c25_rescore_results.json`, `c25_ceiling_results.json`. Board events under
agent `Chase`. No git, no `results/*.pt`, no `tests/chase/deq_run.jsonl`, no
`docs/canon/`, no `lean/` touched. `tests/wilson/resolution/*.py` imported
only, not edited or reimplemented.

## L-REFLECTOR (printed by `c25_rescore.py`, before any arm is scored)

| fixed structure | value |
|---|---|
| initializer | torch default per-layer init, no custom init |
| parameterization | TinyCEQ n=16 d_enc=32 z_dim_state=8 rank=8, 46,349 params |
| corpus regime | ChessBed self-play, uniform-random legal moves, n_games=300 seed=0 max_plies=400 |
| scorer functional | one-vs-rest Murphy REL/RES/UNC, `c25_scorer.score_forecasts` |
| bin scheme | fixed_width_10 / equal_count_200 / equal_count_50, all three per C25 |
| dtype path | model forward float32; scorer accumulation float64 |
| torch build | 2.14.0+cu126, CUDA available |
| eval subsample size | 6000 rows x 4 classes |

## (1) The scorer module

`c25_scorer.score_forecasts(q_pred, k_star, K, scheme)` returns REL, RES, UNC,
a per-class `res_k` table, and `flagged_single_bin` per class (true when every
forecast for that class landed in one populated bin). `score_all_schemes`
runs `fixed_width_10`, `equal_count_200`, `equal_count_50` and returns the
per-scheme ordering plus a verdict: `STABLE` only if all three orderings
agree, else `ORDERING UNSTABLE -- NO VERDICT`.

REUSE, not reimplementation: the REL/RES/UNC accumulation is the same formula
as `tests/wilson/resolution/wil_chess400.murphy_binned_onevsrest` (imported,
not copied); only the bin-edge computation is new (quantile edges for
equal-count, vs that file's fixed `linspace`). `demo()` cross-checks
`score_forecasts(..., "fixed_width_10")` against the imported original on
random data: **max abs diff over REL/RES/UNC = 0.00e+00** (bit-identical).

## (3) Check the check -- run BEFORE any real re-scoring, per the task's own order

Two synthetic pairs, both in `c25_scorer.demo()`:

- **Known-ordering pair** (`check_the_check()`): a genuinely discriminating
  arm (true per-item class probabilities) against a base-rate arm
  (RES = 0 by construction). All three schemes ordered `discriminating >
  base_rate`; wrapper reports **STABLE**.
- **Adversarial pair** (modelled on the task's own rare-class motivating
  instance: one arm concentrates 99% of mass in one bin with a 1% near-
  saturation tail, the other carries a tiny genuine signal spread evenly):
  fixed-width orders `arm_b > arm_a`, both equal-count schemes order
  `arm_a > arm_b`. Wrapper reports **ORDERING UNSTABLE -- NO VERDICT**.

Both directions fire. The wrapper is not one of the twelve checks that could
not fail: `python c25_scorer.py` reproduces both results on every run
(fixed seeds 0 and 1).

## (2) Re-scoring a chess row (operator vs. marginal baseline)

`c25_rescore.py` trains a fresh TinyCEQ (46,349 params, real
`ceqjepa.operator`/`TinyCEQ`/`stage1_loss`/`heldout_split`/`draw` code, none
re-derived) at **reduced scale** -- 300 games, 800 steps, vs. the published
row's 1,000 games / 4,000 steps -- because a full reproduction did not fit
this task's wall-clock budget. This is a fresh number, not a reproduction of
the published `0.001469`, and is reported as such:

| scheme | operator RES | marginal RES | order |
|---|---|---|---|
| fixed_width_10 | 0.001368 | 0.000000 | operator > marginal |
| equal_count_200 | 0.023142 | 0.000000 | operator > marginal |
| equal_count_50 | 0.007410 | 0.000000 | operator > marginal |

**Verdict: STABLE.** No per-class flag fired (no class landed entirely in
one bin) at this scale/seed -- unlike the task's cited tier-2/chess400
row, which is a different training run, different seed budget, and (per the
task text) is exactly where the rare-class flag and the fixed-width/equal-
count inversion were originally observed. This run does not reproduce that
inversion; it is offered as an independent, smaller-scale STABLE point, not
as a contradiction of the cited numbers, which this task's budget could not
regenerate at published scale (1,000 games / 4,000 steps / 5,900-game full
race not run here either).

The published numbers quoted in the task prompt (operator recalibrated RES
0.001469 CI95 [0.001041, 0.004009]; piece-count histogram gap -0.000662;
ply-bucket -0.014518; unbinned ceiling 0.10116955630126778; tier-2 sweep
operator 0.002155 fixed-width / 0.003769 equal-count-200, floor 0.006761 /
0.006993) were **not re-derived** here -- re-scoring them under
`equal_count_50` specifically (the third leg C25 requires) would need the
original run's saved per-item forecast arrays, which are not present in the
tree (only aggregate numbers are saved in
`tests/wilson/resolution/wil_chess400_results.json`). **This is a gap, named
here rather than papered over**: C25's stability verdict cannot be computed
for the *published* tier-2 row without those raw forecasts. Recommendation:
the next run that produces the tier-2 numbers should also save the raw
`q_pred`/`k_star` arrays (not just aggregates), so C25 can be applied to the
exact published row rather than to a fresh substitute.

## (4) The ceiling question

`c25_ceiling.py` recomputes the unbinned rollent-based oracle ceiling at
reduced scale (60 games vs. the published 150, R=16 rollouts, same seed 0,
same functions imported from `wil_res_ceiling.py`) as a reproducibility
gate, then applies the SAME binned scorer functional to the identical
`phat`/outcome data:

| quantity | RES | vs. published unbinned 0.10117 |
|---|---|---|
| unbinned, recomputed here (n=60) | 0.113322 | gate: within published CI95 order of magnitude |
| binned, fixed_width_10 | 0.165855 | 163.9% |
| binned, equal_count_200 | 0.242032 | 239.2% |
| binned, equal_count_50 | 0.211476 | 209.0% |

**The finding is the opposite of what the task text anticipated, and it is
a real mechanism, not noise:** the binned ceiling does not fall short of the
unbinned one here -- it exceeds it, worse as bin count grows. The unbinned
estimator explicitly subtracts a rollout-noise correction term,
`mean(phat*(1-phat))/(R-1)`; the binned scorer functional (correctly, per
its C25 specification -- it is the same functional the arms are scored
with) applies no such correction, so `phat`'s own sampling noise (only
R=16 rollouts per position) inflates the between-bin spread of realised
outcomes. `equal_count_200` on 60 items requests 200 quantile bins,
degenerating toward one item per bin -- close to an exact-partition
scorer, which is exactly the in-sample overfitting `wil_res_null.py`'s
permutation null exists to catch (a fitted-in-sample partition buys free
resolution). This is scale-dependent: it should shrink as `n_games` grows
well past the requested bin count and as `R` grows past the noise floor,
but that was not tested here (time budget). Flagged, not resolved.

**Consequence for every RES-over-ceiling ratio this project has quoted
against a binned ceiling:** none can be taken as-is. A ratio needs its
ceiling computed by the SAME functional AND at comparable `n_games`-to-bin-
count and rollout-noise conditions as the numerator, or the ceiling itself
carries an unquantified, scheme-dependent inflation (as measured here,
1.6x-2.4x the unbinned figure at this scale). Restating the published
ratios needs a ceiling run at matched scale, which was not done here.

## Limits

Every number in this file is from a fresh, reduced-scale run inside this
task's scratchpad (300 games / 800 steps for the operator re-score, 60
games / R=16 for the ceiling), not a reproduction of the published tier-2 or
chess400 rows -- those require raw per-item forecast arrays that are not
saved in the tree. The equal_count_50 leg of C25, specifically requested
against the published tier-2 row, could not be computed against that row
for the same reason. The ceiling-inflation finding in (4) is measured at
n_games=60 only; whether it shrinks or persists at n_games in the
hundreds-to-thousands range (where the published ceiling was measured) is
untested here.
