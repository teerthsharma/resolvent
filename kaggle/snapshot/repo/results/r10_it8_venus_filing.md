# VENUS filing — R10 P1a it.8

One numeric prediction per deciding measurement, **committed before any cell exists**.
Target: MERCURY's it.9 softmax-only sweep, steps {150, 600, 2400, 9600} x n_train
{2048, 8192, 32768} at t* {2, 8, 32}. Bar: NRMSE < 1.0. Open region: t* > hop, hop = 2,
so the region under test is t* in {8, 32}. NO READING counts as not-crossing.

## 1. EXISTENCE CLAIM

**YES.** There exists a (steps, n_train) cell in the grid with NRMSE < 1.0 at t* > hop.

Basis, not vibes: gate (b) already measured a *static local* decoder (radius 5,
held-out half, ~2048 rows) at NRMSE 0.756967 on t8 and 0.731839 on t32
(`results/e4_harmonic_ladder.txt`). The labels carry local signal well under the
1.0 bar at 2048-row scale. The trained softmax arm sits at 1.131993 (t8) and
1.163365 (t32) at n_train=2048 / 5 seeds (`results/e_ladder_reading.txt`). A 0.375
gap between what a ridge fit on local features extracts and what the trained arm
reaches at the same row count is an optimisation/sample gap, not an information
ceiling. Sample gaps close with samples.

Corroborating: the rho axis says parity is *worse* at 12.8M than at 3.3M at every
rho — capacity is not the binding constraint on this substrate, so the frontier
should move with n_train and steps rather than sit flat.

## 2. POINT ESTIMATE

Cheapest crossing cell: **steps = 2400, n_train = 8192, t* = 8.**
Predicted **NRMSE = 0.93**.

## 3. FRONTIER — n*(steps), smallest n_train predicted to clear 1.0

At t* = 8 (the primary frontier):

| steps | 150   | 600   | 2400 | 9600 |
|-------|-------|-------|------|------|
| n*    | never | 32768 | 8192 | 2048 |

At t* = 32: **never** at 150 and 600; **32768** at 2400; **8192** at 9600.
(t* = 2 is not in the open region; it is predicted to cross at every cell.)

## 4. WHAT WOULD FALSIFY ME

Falsified if MERCURY's grid prints zero cells with NRMSE < 1.0 at t* in {8, 32}
(existence dead), or if the cheapest crossing cell there is anything other than
(2400, 8192, t*=8) (point estimate dead), or if any cell contradicts a "never"
or beats a listed n* in the table above (frontier dead) — judged on MERCURY's
printed cells only, NO READING counting as not-crossing, no re-reading of the bar.

Filed before a single cell of the it.9 sweep existed. No partial results consulted.
