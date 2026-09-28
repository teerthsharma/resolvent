# R-JEPA Foreman bar (registered 2026-09-28 00:58 IST, before any run of `rjepa.py`)

Question: what is underneath D-JEPA's decision-local gap (arXiv 2609.24749, facts from the brief, [U] here),
and what is the resolvent's role in closing it?

## Prior read before registering (not a run of this bed)

Canon's `sun/canon/p2/results_dj_n300.json` already holds both decision rules on Lorenz-63 with a shared
(common-random-number) start posterior: `hit_bayes` (top-1 Bayes vote) and `hit_bayes_meanargmax` (plug-in:
argmax of the posterior-mean prediction, i.e. what an MSE-trained predictor ranks by). Their difference is
the part of the gap any operator could close. It is ~0 before 0.75 T*, peaks at 0.05-0.09 at 1.25-1.5 T*,
and returns to noise by 3-4 T* (N = 300, void/underpowered bed; read as a pilot, not evidence).

## Hypothesis under test

H-deep: the gap is a decision-rule mismatch, not prediction error. Latent distance is the plug-in rule.
When the error is SHARED across candidates (it comes from the start state, which all candidates share) and
is comparable to the candidate spread, the Bayes-optimal rank is a function of the whole candidate set
(Gaussian mass of each candidate's Voronoi cell); in the limit it is the exterior angle of each candidate's
Voronoi cell (a convex-hull property, independent of the start), and nearest-to-goal then favours the
hull-interior candidate, whose cell is bounded - latent distance becomes anti-informative.

H-tail (brief's hypothesis 1): the operator is a resolvent (I - gamma W)^-1 on the candidate graph and the
gap is its Neumann tail (multi-hop).

## Bed `shift` (exact truth, Bayes ceiling by Monte Carlo)

D = 2, K = 4, goal g = 0. Start s ~ N(0, 9 I). Observation y = s + sigma*xi. Planner proposes
a_k = -y + rho*r_k, r_k ~ N(0, I), rho = 1 (candidates around the goal: the top-K regime).
Executed outcome z_k = s + a_k + sigma_e*eta_k, sigma_e = 0.1. Truth best = argmin_k |z_k|.
JEPA predictor: MLP f(y, a_k) -> z_hat_k trained by latent MSE (target encoder = identity, frozen).
Bayes: posterior s | y ~ N(kappa y, kappa sigma^2 I), kappa = 9/(9 + sigma^2); M = 1024 draws of (s, eta);
pick argmax_k P(k best). Hull-angle policy: argmax_k of the share of directions u with k = argmax_j r_j.u.
Train 20,000 starts, eval 4,000 held-out starts, seeds {0,1,2}. sigma in {0.1, 0.3, 1, 3, 10}; main cell sigma = 1.

Heads (all ranked on goal-relative predicted features z_hat_k - g; trained by cross-entropy on the executed
best index, i.e. from executed outcomes as D-JEPA does):
- `dist`: -|z_hat_k| (no training) - D-JEPA's native latent-distance planning.
- `point`: MLP per candidate, no interaction (strongest control lacking the set property).
- `hop1`: set head, h = enc(z_hat_k); W = row-softmax of pair MLP over j != i; m = h + gamma W h.
- `resolvent`: same parameters, m = (I - gamma W)^-1 h, gamma = 0.99*sigmoid(theta). Bounded:
  |(I - gamma W)^-1|_inf <= 1/(1 - gamma) for row-stochastic W.

## Bars (mean over 3 seeds, held-out eval)

- B0 null, must fire (else bed void): sigma = 0 (y = s), sigma_e = 0.3 -> agreement(Bayes pick, dist pick) >= 0.97.
- B1 ceiling, must fire: sigma = 0.1 -> Bayes hit >= 0.90.
- B2 gap exists: sigma = 1 -> Bayes hit - dist hit >= 0.03.
- B3 anti-information past the horizon: sigma = 10 -> dist hit <= 0.23 (chance 0.25) and
  hull-angle hit >= Bayes hit - 0.02.
- B4 learn gate (before any comparative number): resolvent agreement with the Bayes pick >= 0.90 at sigma = 1.
- B5 set head closes the gap: closure (head - dist)/(Bayes - dist) >= 0.80 for `resolvent` at sigma = 1.
- B6 set property needed: `point` closure <= 0.50 at sigma = 1. If point closes > 0.5, kill "the fix is a set operator".
- B7 Neumann tail: resolvent - hop1 hit >= 0.01 with all 3 seeds positive. If |resolvent - hop1| < 0.01, H-tail is
  killed at K = 4.
- T-A (theorem, property test): for any permutation-equivariant LINEAR W = alpha I + beta J with
  spectral radius of gamma W < 1, (I - gamma W)^-1 s has the ranks of s. If it holds, a linear resolvent on
  exchangeable candidates is rank-inert and H-tail can only live through a data-dependent W.

## Amendment A1 (00:53 IST, before any run)

1. Train size 20,000 -> 100,000 starts (labels are executed outcomes, noisy at sigma = 1; 20k risks a learn-gate
   failure for sample size alone). Eval stays 4,000.
2. Generation order made exact for a flat posterior: y ~ N(0, 9 I) first, then s = y - sigma*xi, so
   s | y ~ N(y, sigma^2 I) exactly and kappa = 1. The N(0, 9 I) start prior line above is replaced.
3. B0-B3 are read with the ideal plug-in z_hat = y + a (the MSE-optimal predictor); B4-B7 with the learned
   JEPA predictor's z_hat, and `dist` there also uses the learned z_hat.

Kills and their replacement routes are written into the verdict, never left bare.

## Amendment A2 (00:58 IST, AFTER the stage-1 run; declared post-run)

Stage-1 run (heads without set interaction): B1 FAILED (Bayes hit 0.86 at sigma = 0.1), so per B1 the bed was
void; B2 read 0.021 (void with it). Cause: idiosyncratic execution noise sigma_e = 0.1 is irreducible for every
rule and is comparable to the nearest-pair gap of 4 unit-Gaussian candidates. Change: sigma_e = 0.1 -> 0.02
(fixed by that argument, before any run at 0.02). Every other bar, including the sigma = 1 main cell and B2's 0.03,
is unchanged. The 0.021 stage-1 reading is reported beside any new reading.

Added before building them (Wilson's facts, sun/rjepa/wilson/facts.json):
- Head `djepa`: D-JEPA's operator re-implemented from the paper's spec (not the released code): token
  v_i = [d_i; r_i], d_i = z_hat_i - g (2-d; LayerNorm on a 2-d vector is degenerate so it is dropped), r_i =
  rank of |z_hat_i| scaled to [0,1]; one horizon so b_i = r_i; 64-wide encoder (LN, GELU), 2 pre-norm Transformer
  layers, 4 heads, FF 128, no positional encoding; head 64->8->1 zero-init; delta = 0.2 tanh(W_up tanh(W_down h));
  s = b + delta; loss -log sum_{best} softmax(-s/0.05) + 0.1 mean delta^2 (the local-margin term is dropped).
- B8 (bound vs horizon): at K = 4, 2 eps = 0.4 lets D-JEPA select only base ranks {0, 1}. Prediction: at sigma = 1
  `djepa` hit >= Bayes - 0.03; at sigma = 10 `djepa` hit <= Bayes - 0.03 while the fraction of Bayes picks with
  base rank >= 2 is >= 0.25. If `djepa` stays within 0.03 of Bayes at sigma = 10, B8 is killed.
- B9 (speed, this box only): median forward latency, batch 16 starts, K = 63, 386-d tokens for D-JEPA and the
  same 386-d input to the resolvent head, CUDA fp32, 200 timed runs after 50 warm-up: resolvent head <= djepa.
  Descriptive ratio reported either way; no cross-machine claim.

## Amendment A3 (01:20 IST, AFTER the 12-epoch learned run; declared post-run)

12-epoch run (`results_cell_sigma{1,10}_ep12.json`): B4 learn gate FAILED (resolvent agreement 0.8685 / 0.8913 /
0.8998, mean 0.8865 < 0.90), so B5-B7 at sigma = 1 are void from that run. `djepa` learned no swap on any seed
(closure 0.000 at sigma = 1; 0.002 at sigma = 10), so it fails its own learn gate and B8's learned reading is void.
Changes: (1) every non-djepa head retrains at 36 epochs (same for all kinds) at sigma = 1; B4-B7 are read from
that run only. (2) B8b, learning-free, replaces B8 as the deciding test of the bound: the bound-restricted Bayes
rule (argmax of P(k best) over candidates whose base score is within 2 eps of the base minimum, i.e. distance
ranks {0, 1} at K = 4) is the ceiling of ANY operator obeying D-JEPA's Prop 2 / Cor 1. B8b: at sigma = 10 that
ceiling <= Bayes - 0.03; at sigma = 1 it is >= Bayes - 0.01. If either fails, B8b is killed.

## Amendment A4 (01:31 IST, before any run of it): B10, the shared-error share

Generalised cause under test: the closable gap is carried by the part of the prediction error that is SHARED
across candidates (a common translation acting on the whole candidate set); idiosyncratic error leaves nothing
to close (B0). Bed change (default path unchanged): total error variance sigma^2 split into a shared part
sigma^2 (1 - m) (start estimate) and an idiosyncratic part sigma^2 m per candidate; Bayes knows both.
- B10: at sigma = 3, gap(m) = Bayes hit - dist hit on 3 seeds x 4,000 starts: gap(0) >= 0.10, gap(1) <= 0.01,
  and gap(0) > gap(0.5) > gap(1). Kill if any fails.
Descriptive (1 seed, not a claim): D-JEPA's head at sigma = 10, 4 epochs, eps = 0.2 saturates |delta| (mean 0.164,
max 0.194) and agrees with dist on 0.998 of starts, hit 0.1695; the same head at eps = 4 reaches hit 0.3715
(Bayes 0.381). Neither passes the 0.9 agreement gate; logged as Open.

## Amendment A5 (01:30 IST, before the 36-epoch sigma = 10 run; post-pilot, declared)

Measured: two independent M = 1024 Bayes picks agree on 0.931 / 0.935 / 0.930 of starts at sigma = 1 and
0.837 / 0.839 / 0.826 at sigma = 10, so the 0.90 agreement gate cannot pass at sigma = 10 by construction.
Gate at sigma = 10: agreement >= 0.9 x mean Bayes self-agreement (0.9 x 0.833 = 0.750).
- B11: sigma = 10, 36 epochs, 3 seeds: resolvent passes that gate and closure >= 0.80.
- B12 (post-pilot: the 12-epoch run read point closure 0.82): point closure >= 0.50 at sigma = 10, i.e. most of the
  horizon correction is a per-candidate inversion of distance, not a set interaction.
B8 (learned, 12 epochs) is marked skip in the test file: void per A3, kept for the record.

## Amendment A6 (01:35 IST, before the run): B13, the bound not the architecture

36-epoch sigma = 10 read: B11 FAILED (resolvent agreement 0.725 < 0.750; closure 0.958); hop1 agreement 0.781,
closure 0.989; B12 point closure 0.813.
- B13: D-JEPA's head with eps = 4 (every base gap at K = 4 is < 2 eps, so Cor 1 no longer restricts it), all else
  identical, 12 epochs, sigma = 10, 3 seeds: agreement >= 0.750 and closure >= 0.80. The eps = 0.2 reference is the
  12-epoch run already stored (closure 0.002 / 0.000 / 0.007). Kill: if eps = 4 also fails, the architecture or its
  training, not the bound, blocks the horizon rule.
