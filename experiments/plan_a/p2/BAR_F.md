# P2-F bar — Lorenz-63 forecasters: skill before T*, reliability after T* (registered 2026-09-27, before any run)

Every run records the sha256 of this file.

## Bed

- Lorenz-63 as `sun/sun_maths.py` (RK4, dt 0.01), lambda = 0.9059 (S2).
- Observations every dt_obs = 0.05 of the full state with noise delta = 0.1 per coordinate.
  T* = ln(1/delta)/lambda = 2.542 (S2's law, Delta = 1).
- Input: the last L = 20 observations (1.0 time unit). Leads tau/T* in {0.25, 0.5, 1, 1.5, 2, 2.5, 3}.
- Targets at each lead: the true state (3 numbers) and the event e = 1[x(tau) > 0] (the wing).
- Seed s in {0, 1, 2} draws the training trajectory (10,000 time units after a 50-unit burn-in) and
  its noise; one fixed test trajectory (seed 99, 2,000 time units), shared by all arms and seeds.

## Metrics

- **Skill before T***: state MSE skill S = 1 - MSE/Var_clim (3 coordinates pooled, clim from train),
  averaged over leads tau <= T* (0.25, 0.5, 1).
- **Reliability after T***: Murphy decomposition of the event Brier score, BS = REL - RES + UNC,
  10 equal-width probability bins, averaged over leads tau > T* (1.5, 2, 2.5, 3). BSS = 1 - BS/UNC.

## Arms

| arm | what | probability of the event |
|---|---|---|
| `clim` | floor: train base rate; state = train mean | base rate |
| `persist` | floor: last observation as the forecast | 1[x_last > 0] |
| `oracle` | reference, not a ceiling: true model, 256-member ensemble from last obs +- delta | ensemble fraction |
| `tf` | transformer encoder (2 layers, width 64, 4 heads) on the 20 observation tokens; per-lead heads for state (MSE) and event (BCE) | sigmoid of its event head |
| `ngrc` | next-generation RC (Gauthier et al. 2021): constant + 2-tap delay linear + unique quadratic monomials, ridge, one step of dt_obs, iterated | 1[x_hat > 0] (deterministic, as published) |
| `edmd` | EDMD: monomials to degree 3 of the 2-tap delay state, least squares Koopman matrix, iterated | 1[x_hat > 0] |
| `canon` | the Canon P2 element: `ngrc` mean + a spread sigma(tau) = min(sigma0 * exp(lam_hat*tau), sigma_clim) per coordinate, lam_hat and sigma0 fitted on train residual growth (no test data), mean shrunk to climatology by the same factor | Phi(mu_x / sigma_x) |
| `ngrc-p` | owner's strongest probabilistic variant: logistic regression per lead on the ngrc features | logistic |

Deep Koopman (Lusch et al. 2018) is **not run** in this pass; L-OWNER therefore bars a PASS for P2
until it is (a PASS here can only be "PASS pending deep Koopman").

## Learned gate (L-LEARN)

An arm with state skill < 0.90 at tau = 0.25 T* on test is **void**: printed, not compared.
(`clim` and `oracle` are exempt; `persist` is a floor and is expected to fail it.)

## Bars (mean over 3 seeds)

- **Prediction (plan):** |S(canon) - S(tf)| <= 0.05 before T*; after T*, REL(canon) <= 0.01 and
  REL(tf) >= 0.05.
- **Counter (plan):** REL(tf) <= 0.02 after T* — calibration comes from the proper scoring rule, not
  the architecture.
- **House addition (registered now):** `clim` has REL = 0 by construction, so the REL bar alone is
  passed by a forecaster with no information. Canon's REL claim counts only if
  BSS(canon) >= BSS(tf) - 0.02 after T*.
- **PASS** iff prediction and House addition hold and the counter does not (and then only "pending
  deep Koopman"). **KILL** iff the counter holds. **OPEN** otherwise.

## Lead prior (D-CALIB)

Expected: the counter holds (a BCE-trained head is calibrated in distribution), so P2 is KILLED as
an architectural claim: the horizon is a training rule (proper scoring), not architecture.

## Amendment A1 (2026-09-27, after House's shortcut hunt, before any arm run)

Declared: House ran the single-observation oracle (no arm) before this amendment. It measured oracle
BSS 0.40 / 0.07 / 0.01 / -0.01 at 1.5 / 2 / 2.5 / 3 T*.

1. **Live leads:** REL and BSS bars are evaluated **per lead**, only on leads > T* where the
   20-observation oracle (item 4) has BSS >= 0.05. Dead leads are printed and never averaged into a
   bar. If there is no live lead > T*, the verdict is OPEN.
2. **New arm `ngrc-platt`:** a per-lead logistic on the ngrc mean mu_x (2 parameters per lead), fitted
   on train. Canon's sigma0 and lam_hat are fitted on leads <= T* only; a Canon fitted on leads > T*
   is void. Canon's claim needs REL(canon) <= 0.01 **and** BSS(canon) >= BSS(ngrc-platt) - 0.02 at
   every live lead > T*.
3. **tf early-stopping and new arm `tf-platt`:** `tf` early-stops on the last 1,000 train time units
   (validation slice). `tf-platt` is a per-lead logistic recalibration of tf's event logit on that
   slice. **KILL** also fires if REL(tf-platt) <= 0.02 with BSS(tf-platt) >= BSS(canon) - 0.02.
4. **Oracle:** an importance-weighted ensemble conditioned on all 20 observations, using the true
   model. The single-observation oracle is printed beside it.
5. **Learned gate:** S(0.25 T*) >= 0.95.
6. **Skill match before T*:** MSE(canon)/MSE(tf) in [0.8, 1.25] at every lead <= T*. This replaces
   |dS| <= 0.05.
7. **Definitions:**
   - Var_clim = mean over coordinates of each coordinate's variance about its own train mean.
   - Canon's shrink factor is rho = sqrt(max(0, 1 - sigma^2/sigma_clim^2)), with mean
     = clim + rho*(mu - clim).
   - REL from deterministic 0/1 arms is printed and never cited.
   - CIs use a block bootstrap with 10-time-unit blocks on the test trajectory.
