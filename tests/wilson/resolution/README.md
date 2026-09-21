# Resolution: the only score component a recalibrator cannot fake

Murphy's decomposition is `Brier = REL − RES + UNC`. Temperature scaling is
monotone on the logits, and resolution scores the ranking that a monotone map
preserves exactly — so recalibration **cannot move resolution and can zero
reliability**, by construction. Measured here: a mis-scaled arm's reliability
falls `0.02407 → 0.00012`, a factor of 200, while resolution stays flat.
Foster–Vohra says the same from the other side, and a base-rate forecaster scores
the *best* reliability of any arm, `0.0`, alongside the *worst* resolution, `0.0`.

So any architecture claim this project makes has to live in resolution. This
directory measures whether it can.

## Running it

```bash
python wil_res_ceiling.py      # the bed's oracle ceiling, both caps
python wil_res_null.py         # 1000-permutation shuffle null
python wil_bar_can_fire.py     # the degenerate controls, before trusting anything
python wil_chess400.py --part 1 --n-games 1000 --seed 0
python wil_chess400.py --part 2 --n-games 1000 --seed 0 --steps 4000
```

Run with the repository root as the working directory so `ceqjepa` imports.

## The bed had to be repaired before it could measure anything

At the shipped `max_plies=80` the chess bed **cannot discriminate**: 98.67% of
games end in SINK, the oracle ceiling — the hard bound for *any* forecaster — is
`0.000501` with CI95 `[6.09e-05, 1.04e-03]`, and a 1000-permutation shuffle null
posts `0.000579`. A discriminating 8-bin arm scores `0.000645`: ratio `1.114×`,
`p = 0.337`. It does not clear chance.

At `max_plies=400`, same generator and seed, the ceiling is `0.10116955630126778`
with CI95 `[0.0708, 0.1329]` — **202×, non-overlapping intervals** — `RES/UNC`
rises from 1.90% to 17.99%, SINK falls to 26.0%, and the same arm clears the null
at `3.995×`, `p = 0.000`. The analytic null mean `(B−1)·UNC/N` matches the
permutation mean to 1.2% at both caps, which cross-validates the permutation
implementation rather than trusting it.

`max_plies` was already an argument of `generate_selfplay_game` and
`ChessBed.build`. The fix was a keyword.

## The result: no resolution worth the name

`ceqjepa.operator.committor`, `TinyCEQ` at 1,973 parameters, 4,000 steps, held
out **by game**. Recalibrated resolution `0.001469`, game-paired bootstrap CI95
`[0.001041, 0.004009]`.

| against | RES gap | CI95 | reading |
|---|---|---|---|
| base rate | **+0.001469** | [+0.001041, +0.004009] | real, 1.45% of ceiling |
| pre-registered bar `0.00506` | — | — | **29% of it** |
| ply-bucket histogram | **−0.014518** | [−0.020625, −0.006879] | loses 11× |
| piece-count histogram | **−0.000662** | [−0.002380, +0.001812] | **tied** |

The bar of `0.00506` is 5% of the bed's ceiling, set before any arm was scored,
because recalibrator drift on this bed is 2.0–2.2% and a gap inside that is
noise.

The ply-bucket arm reads game depth, which `fen_to_vec`
(`ceqjepa/beds/chess.py:83`) drops from the operator's input — so that row prices
the *encoding*, not the operator.

**The piece-count arm is the fair control, and it ties.** Eight bins of the total
piece count, computed from the operator's own 769-float input — strictly less
information than the operator receives — matches it on resolution with the point
estimate in the histogram's favour, and beats it on Brier, `0.5020` against
`0.6334`. Even after temperature recalibration at `T = 2.9395` (NLL
`1.6336 → 1.1052`), the operator's reliability is `0.130097` against the marginal
predictor's `0.001334`, a factor of 98, and its raw Brier is worse than
predicting the base rate.

1,973 parameters and 4,000 steps bought nothing over one scalar summary of the
operator's own input.

## What is honest about the measurement, and what is not settled

The binned decomposition's identity residual is `0.00188`, 0.26% of Brier. That
is **not machine epsilon** and it is not hidden: it is a real discretization gap
from finite bin width, and it is not an arithmetic fault, because the
marginal-predictor row on the same code path residuals at `1e-16` — that arm is
piecewise constant. An exact-partition convention was avoided deliberately for a
continuous committor, where it degenerates to near-singleton bins and the
identity becomes a tautology.

Only `committor` can carry a proper score at all. Exact refusal raises a
`SingularTransientBlockError`, and an exception is not a point on the simplex;
decidable NEVER is a proof-checker fact with no probability in its statement.
Neither gates a forecast — they gate *which items receive one*.

**Not run:** the full pre-registered race — 5,900 games, isotonic recalibration on
both arms, operator against a `G ≡ 1` softmax baseline, paired by game. This is a
first measurement in a fifty-minute window, one bed, one generator, self-play
chess, and a committor read rather than a trained model.
