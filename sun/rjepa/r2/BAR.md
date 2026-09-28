# R-JEPA round 2 bar: the shared-error dial with a learned predictor at K = 63 (Foreman)

Registered 2026-09-28 02:14 IST, before `r2.py` or `test_r2.py` exist. Every result file records the sha256 of this file.

Question: on a bed with a shared-error dial f and a LEARNED predictor at K = 63, which R-JEPA arm wins, and does
the winner flip with f as round-1 B10 predicts?

## Bed `dial`

- Latent D = 8, action m = 4 per step, horizon H = 5, K = 63 candidates per start.
- True dynamics (fixed across seeds, generated once from seed 12345): z' = A z + 0.5 tanh(C z + B a),
  A = 1.1 Q with Q Haar-orthogonal (spectral radius exactly 1.1, errors grow 1.1^5 = 1.61x plus the tanh term),
  C ~ N(0, 1/D), B ~ N(0, 1/m). F(x, a_{1:H}) = H-step rollout.
- Order of generation (flat posterior exact, as round-1 A1): planner estimate y ~ N(0, I_D); candidates
  a_k ~ N(0, I) in R^{H x m}; goal g = F(y, a*) with a fresh a* (goal inside the candidate cloud, no truth leak).
- Error dial: realised start of candidate k is x_k = y - sigma e_k, e_k = sqrt(f) s + sqrt(1-f) u_k, s, u_k ~ N(0, I_D).
  f = 1: one unknown true start shared by every candidate. f = 0: independent per-candidate start noise (a
  stochastic environment). Reading chosen because it matches B10's structure; the other reading (the planner
  observes a separate estimate per candidate) would let Bayes pool 63 estimates and is a different question.
  Marginally e_k ~ N(0, I) for every f.
- Executed outcome z_k = F(x_k, a_k) (true dynamics). Label best = argmin_k |z_k - g| (the truly best candidate).
  Secondary label success_k = |z_k - g| < r, r = median over TRAIN starts of min_k |z_k - g| (per cell).
- Scale: rho = candidate spread = RMS over dims of std_k F(y, a_k), averaged over starts. sigma for a ratio
  sigma/rho in {1, 3, 10} is set by a pilot (4,000 starts, bisection) so that the RMS per-dim std of
  F(y - sigma e, a) - F(y, a) equals ratio x rho. The pilot values are written to the results.
- Cells: f in {0, 0.25, 0.5, 0.75, 1} x sigma/rho in {1, 3, 10}. Seeds {0, 1, 2} vary data, predictor init and
  head init. n_train = 20,000 starts per cell, n_eval = 2,000 held-out starts per cell per seed.
  Run order (cut cells, never seeds): (f=0, 3), (f=1, 3); then f in {0.25, 0.5, 0.75} at 3; then f in {0, 1} at 1
  and 10; then the rest.

## Learned predictor (L-LEARN)

MLP (D + m) -> 256 -> 256 -> D, GELU, trained by MSE on true one-step transitions from 100,000 rollouts of
y ~ N(0, I) with random actions (500,000 transitions), fixed budget 2,000 Adam steps, batch 2,048, lr 1e-3 cosine.
Rolled out H steps from y for every candidate: z_hat_k. One predictor per seed, shared by every cell.
Gate L-LEARN: one-step R^2 >= 0.90 on 50,000 held-out transitions. Also reported: H-step R^2 of z_hat vs F(y, a).
If the gate fails, every learned-predictor reading is void (kill).

## Floors, ceiling, null

- blind = 1/K for `best`; NS = (hit - 1/K) / (hit_bayes - 1/K).
- Bayes: argmax_k P(k best | y, a) by Monte Carlo over (s, u_1..u_K), M = 256, TRUE dynamics (it knows the
  model and the error law, including f). Secondary: argmax_k P(success_k) (marginal).
- Null: arm 2 (`point`) trained on labels shuffled across starts; must score NS within +-0.05 of 0.

## Arms (every learned arm sees the same 12-d token per candidate and the same training budget)

Token v_k = [LN(z_hat_k - g) (8), d_k = |z_hat_k - g|, log s_k, r_k, l_k], with s_k = sigma |J_k^T n_k| (J_k the
Jacobian of the learned rollout wrt y, n_k the unit goal direction, one VJP per candidate), r_k = distance rank
in [0, 1] (0 = nearest), l_k = LIN1 rank in [0, 1] (0 = highest chance).
Training: 4,000 AdamW steps, 64 starts per batch, lr 1e-3 one-cycle, wd 1e-4, clip 1.0, on the executed `best`.

1. `dist` (floor): argmin_k d_k. D-JEPA's native latent-distance planning.
2. `point`: MLP 12 -> W -> W -> 1 per candidate, no mixing; CE on best.
3. `dj02`: D-JEPA as specified (facts.json): 12 -> 64 encoder (LN, GELU), 2 pre-norm Transformer layers, 4 heads,
   FF 128, no positional encoding, head 64 -> 8 -> 1 zero-init, delta = 0.2 tanh(.), s = b + delta, b = r_k,
   loss CE(-s / 0.05) + 0.1 mean delta^2 (local-margin term dropped, as round 1; one positive per start).
4. `dj4L`: same net and loss, eps = 4 (2 eps = 8 > the maximum base gap 1, so Cor 1 never binds), b = l_k.
5. `hop1`: round-1 one-hop set head: h = enc(v), W = softmax over j != i of q(h).k(h), m = h + g W h,
   g = 0.99 sigmoid(theta), score = out([h, m]).
6. `lin1`: argmax_k Phi((r - d_k) / s_k), ties to `dist`. Zero parameters.
Disentanglers at the deciding cells only: `dj02L` (b = l_k, eps = 0.2), `dj4` (b = r_k, eps = 4).
Parameter match: `point` and `hop1` widths chosen so their counts are within 10 % of `dj02`'s; counts reported.

## Contract tests (RED before code; `test_r2.py`)

C1 permutation equivariance of point / hop1 / dj heads. C2 |delta| <= eps for dj02, and its pick has base rank
within 2 eps of the base minimum. C3 LIN1 = exact marginal P(success) in the linear-Gaussian limit (linear
dynamics, s_k <= 0.02 d_k): max |LIN1 - MC| <= 0.02. C4 Bayes hit >= every arm - 2 SE on a small cell.
C5 null point head on shuffled labels: hit within 3 SE of 1/K. C6 (theorem T-F) marginal P(success_k) is
f-invariant: MC at f = 0 and f = 1 for the same (y, a) agree within MC error. C7 L-LEARN gate.

T-F consequence, stated before any run: e_k ~ N(0, I) marginally for every f, so each candidate's outcome law
is f-invariant. Under a success-indicator label (D-JEPA's), the Bayes pick is argmax of per-candidate marginals
and every per-candidate rule's expected score is f-invariant: f can matter only under the relative (argmin) label.

## Bed validity (else every comparative row is void)

V1 gap exists: at (f = 1, 3), Bayes hit - dist hit >= 0.02 (mean of 3 seeds). V2 null passes. V3 L-LEARN passes.

## Pre-registered prediction and kills (deciding cells (f = 0, 3) and (f = 1, 3); 3 seeds each)

- P1: at f = 0, |NS(point) - NS(best set arm of hop1, dj4L)| <= 0.01 and |NS(lin1) - that| <= 0.01 (mean of seeds).
- P2: at f >= 0.75 and sigma/rho >= 3, dj4L and hop1 each beat dj02 by >= 0.05 NS on 3/3 seeds.
- K1 (trust region is the wall): if dj02 is within 0.01 NS of the best arm at (f = 1, 3) on 3/3 seeds, the claim dies.
- K2: L-LEARN fails -> every learned-predictor row void.
Every kill ships a replacement route in the verdict.

## Amendment A1 (02:16 IST, after the contract tests, before any cell run)

- C5 fired on its registered form: a `point` head trained on shuffled labels (3,000 train / 2,000 eval starts,
  f = 1, sigma = 1, exact dynamics) hit 0.0300 vs chance 0.0159 (5.0 SE). Mechanism: the argmax of a near-constant
  head inherits the token's distance order, so "chance" is the wrong null for any head that sees d_k. Replaced:
  C5 and V2 now require null hit <= dist hit + 3 SE (a head trained on shuffled labels must not beat the untrained
  floor); the null's NS is still reported.
- C6's registered max|diff| <= 4 SE ignored the 1,260-way max and the two independent MC draws; it read 0.0385.
  Replaced by mean standardised z^2 in [0.7, 1.3] plus a control (sigma 1.3 vs 1.0) that must exceed 3.
- C3 / C6 test plumbing: CPU matrices moved to the device; r set to the cell's median distance (r = 1 in 8-D gave
  an all-zero table, the degeneracy guard fired).
- L-LEARN read (results/pred.json): one-step R^2 0.99945 / 0.99937 / 0.99943, H-step R^2 0.9973 / 0.9968 / 0.9971.
  Calibration (results/calib.json): rho = 0.638; sigma = 0.368 / 1.143 / 3.977 for sigma/rho = 1 / 3 / 10.
  The predictor's H-step error (RMS ~ 0.05 x the state scale) is small beside rho; recorded, not a gate.

## Amendment A2 (02:19 IST, after one pilot cell-seed; declared post-pilot)

Pilot (f = 0, sigma/rho = 3, seed 0, M = 256, run killed after this seed; its numbers are discarded from every table):
the marginal-success Bayes pick scored NS 1.282 on the argmin label, i.e. it beat the M = 256 argmax P(k best) pick,
so the MC ceiling was short at K = 63 (P(k best) ~ 1/63 is too small for 256 draws). Pilot NS for the record:
dist 0.718, lin1 0.435, point 0.671, hop1 0.765, dj02 0.718, dj4L 1.000, dj02L 0.435, dj4 0.671, null -0.223.
Changes: (1) Bayes M = 256 -> 2048 for every cell. NS > 1 or bayes_succ > bayes is reported as a ceiling-shortfall
flag, never hidden. (2) Data generated on the GPU: the CPU version peaked at 1.69 GB host RAM, over the 1.5 GB cap.
(3) Every cell also saves its picks (picks_f*_q*_s*.pt) so a ceiling can be recomputed without retraining.

## Amendment A3 (02:25 IST, after 2 cell-seeds at n_eval = 2,000 and a power sweep; declared post-run)

Read at the registered size (results/registered_n2000/, seed 0 only, both runs killed): at (f = 0, 3) Bayes hit 0.0310,
dist 0.0285, blind 0.0159; at (f = 0.75, 3) Bayes 0.0385, dist 0.0265. One start moves NS by 0.033, so NS per
seed carried SE ~ 0.2: the registered bed could not resolve the 0.01 / 0.05 NS bars. Power sweep (results/sweep.json,
exact model, no training, 1,000 starts, M = 2,048): Bayes - dist hit at f = 0 is <= 0.015 at every sigma in
0.01..1.14; at f = 1 it is 0.004 / 0.010 / 0.018 / 0.038 / 0.045 at sigma = 0.03 / 0.2 / 0.37 / 0.7 / 1.14,
with Bayes hit 0.218 at sigma/rho = 1 and 0.081 at sigma/rho = 3.
Changes: (1) n_eval 2,000 -> 20,000 and n_train 20,000 -> 60,000 per cell-seed; data stays on the GPU.
(2) The f = 0 reading of P1 moves to sigma/rho = 1 (Bayes 10x chance there; at 3 every rule is within 0.013 of
chance). (0, 3) stays in the plan, run last, flagged underpowered. (3) P2 and K1 stay at (f = 1, 3); sigma/rho = 10
is cut (the sweep puts Bayes at chance well before it). (4) Run order: A = (1, 3), (0, 1); B = (1, 1), (0.75, 3),
(0.5, 3); then (0, 3), (0.5, 1), (0.25, 3) if time. The bars' thresholds are unchanged.

## Amendment A4 (02:34 IST, after seed 0 of (f = 1, 3) at the A3 size; before seeds 1 and 2 exist)

Seed 0 read (NS): Bayes 1.000, bayes_eps02 (the Bayes pick restricted to base ranks within 2 eps = 0.4) 0.944,
dj4L 0.732, dj4 0.724, hop1 0.501, point 0.355, dj02 0.352, dist 0.338, dj02L 0.266, lin1 0.243, null 0.132.
It suggests the eps = 0.2 reach is NOT what blocks dj02 at K = 63 (ranks are scaled to [0, 1], so 2 eps covers 25 of 63
ranks); the loss is in learning. Registered now, read on seeds 1 and 2 only (seed 0 suggested them):
- M1 reach is not the wall: NS(bayes_eps02) >= 0.90 at (1, 3) on seeds 1 and 2.
- M2 the bound costs learning: NS(dj4) - NS(dj02) >= 0.05 at (1, 3) on seeds 1 and 2 (same net, base, loss; eps only).
- M3 speed or saturation: arm `dj02f` = dj02 with lr x 20 (so a parameter step moves the logits as much as in dj4).
  If NS(dj02f) - NS(dj02) >= 0.5 (NS(dj4) - NS(dj02)) on 3/3 seeds, the bound's cost is optimisation speed under a
  fixed budget; otherwise it is saturation of the tanh correction (reported: mean |delta| / eps on eval).
Kill of the "trust region is the wall" reading as a REACH claim: M1 passing kills it; the replacement is M2/M3.

## Amendment A5 (02:57 IST, resume after the session crash; declared post-run for A3r, pre-run for everything else)

- Resume state: (1, 3) 3/3 seeds, (1, 1) 3/3, (0.75, 3) 2/3 on disk; the orphan `r2.py cell 0 1` (PID 12700, 02:44) is
  kept, not duplicated; (0, 1) seed 0 written at 02:54. Queue after it, one job at a time, >= 1.5 GB free host RAM before
  each start: M3 arms (`R2_KINDS=dj02f,dj02`, (1, 3), 3 seeds; the dj02 re-run supplies the registered mean |delta| / eps
  and a run-to-run repeat of dj02), then (0, 3), then (0.75, 3) seed 2. torch threads 4 -> 2. No bar changes.
- `table` merge no longer lets an `_extra` arm overwrite a main-file arm of the same name.
- A3 re-stated as a testable claim (the struck A3 finding stays struck): at the registered n_eval = 2,000 the SD over
  disjoint 2,000-start blocks of NS(dj4L) - NS(dj02) at (1, 3) is >= 0.05 on 3/3 seeds
  (`test_claims.py::test_A3r_registered_n2000_cannot_resolve_P2_bar`, RED on a dj4L == dj02 control at 02:56, read on
  the existing picks files, so post-run).
- F1 (registered 02:58, pre-run: no (0, 3) file at the A3 size exists): the set arms' edge over dj02 vanishes without
  shared error. At (f = 0, sigma/rho = 3), mean over 3 seeds of hit(a) - hit(dj02) <= 0.005 for a in {dj4L, hop1}
  (hit units, not NS: at f = 0 the Bayes - blind denominator is ~0.015 and NS noise is unbounded). For scale, the
  same difference at (1, 3) was 0.0243-0.0296 for dj4L and 0.0079-0.0101 for hop1 (read before this registration). `test_claims.py::test_F1_*`.
- M3b (registered 03:03, after M3 seed 0 at (1, 3), before seeds 1 and 2 of `_extra` exist; read on seeds 1 and 2 only).
  Seed 0 read: NS dj02f 0.338 = dist 0.338 exactly, dj02f mean |delta| / eps 6.3e-5; dj02 re-run NS 0.352 (original 0.352),
  mean |delta| / eps 0.792. Claims: (a) the lr x 20 probe kills the correction rather than speeding it: dj02f mean
  |delta| / eps < 0.01 and NS(dj02f) within 0.01 of NS(dist); (b) dj02 is near saturation: mean |delta| / eps >= 0.6;
  (c) dj02 is reproducible: re-run NS within 0.02 of the main-file NS. `test_claims.py::test_M3b_*`.
- S1 (registered 03:43, pre-run; the M3 kill's replacement route): the eps = 0.2 cost is saturation of the bounded
  correction. Arms `dj0.5`, `dj1` (net, base r_k, loss and lr exactly as dj02; only eps changes) at (1, 3), 3 seeds,
  written to `res_f1.0_q3.0_s*_eps.json`. Claims on 3/3 seeds: (a) dj1 mean |delta| / eps < dj02's (0.79);
  (b) NS(dj1) - NS(dj02) >= 0.05. dj0.5 reported, not tested. `test_claims.py::test_S1_*`.

## Amendment A6 (round 3, Foreman, registered 2026-09-28 19:55 IST, before any round-3 code or run exists)

Code and results live in `sun/rjepa/r3/foreman/` (r2.py and r2 results are imported read-only). Three items.

**A6.1 Bayes tie-break (fix, applies to every round-3 number).** r2 picks the Bayes arm as `(Pb - 1e-9 d_learned).argmax`,
so an exact MC tie in P(k best) (resolution 1/2048) is broken by the LEARNED predictor's distance: the ceiling borrows
the arm-side model. Fix: ties at the maximum of Pb are broken by the smallest posterior-mean TRUE distance
E[|z_k - g|] from the same 2,048 true-dynamics draws (`bayes_eps02` likewise). RED first (`test_r3.py`). Audit: all 15
round-2 cell-seeds are regenerated (same generators), the old pick must reproduce the stored `bayes` pick exactly
(else the audit is void), and the report states the tie rate and every NS that moves by >= 0.005.

**A6.2 Gap-onset map (P2 re-stated conditionally).** Cells (f, sigma/rho) = (0.85, 3), (0.9, 3), (0.95, 3); seeds 0, 1, 2;
n_train 60,000, n_eval 20,000 per cell-seed (the A3 size: paired SE of Bayes - dist hit ~0.002 at n = 20,000, so 0.02 hit
is ~9 SE; A3r showed 2,000 cannot resolve the 0.05 NS bar). Arms: dist, dj02, dj0.5, dj4L, hop1, Bayes (A6.1 tie-break),
null (`point` on shuffled labels); plus lin1 and the true-dynamics arms of A6.3, reported only. Training, nets, tokens,
M = 2,048 exactly as r2. Run order 0.9, 0.95, 0.85 (cut cells, never seeds).
- gap(cell) = mean over 3 seeds of hit(Bayes) - hit(dist).
- P2c: in each new cell, [dj4L - dj02 >= 0.05 NS on 3/3 seeds] == [gap >= 0.02 hit]. Kill if the edge appears
  (3/3 seeds >= 0.05) where gap < 0.02, or does not appear where gap >= 0.02. The already-read cells (0.75, 3) and (1, 3)
  are consistent with P2c and are not evidence for it.
- Forecast (not a bar): onset between f = 0.85 and 0.95 (gap < 0.02 at 0.85, >= 0.02 at 0.95), because the independent
  part sigma sqrt(1 - f) = 0.44 / 0.36 / 0.26 (sigma = 1.143, rho = 0.638) shrinks toward the shared-shift regime.
- Read before this registration (post-hoc, recorded, not a claim): at (1, 1) gap = 0.024 / 0.024 / 0.029 >= 0.02 yet
  dj4L - dj02 = 0.004 / 0.005 / 0.023 NS. So the hit-unit gap is not sufficient off sigma/rho = 3: in NS units the
  closable room there is 1 - NS(dist) ~ 0.1. Reported for every new cell: closable_ns = 1 - NS(dist) and
  frac = (NS(dj4L) - NS(dj02)) / closable_ns.
- V: null hit <= dist hit + 3 SE in each new cell-seed, else that cell-seed is void.

**A6.3 LIN1 on the true dynamics at (0, 1).** Arms `lin1_true` (d_k, s_k from the TRUE rollout and its VJP, same r) and
`dist_true` (argmin true noiseless distance) on the regenerated r2 eval sets, 3 seeds; `bayes_succ` is the exact marginal
rule. Registered prediction (Foreman's): the LIN1 shortfall is LINEARISATION, not the learned predictor, because the
perpendicular spread sigma^2 |J_perp|_F^2 (~ 8 x 1.6^2 x 0.135 ~ 2.4 for sigma = 0.368) is of the order of d^2 (~ r^2 = 2)
and LIN1's 1-D Gaussian ignores it.
- L1: NS(bayes_succ) - NS(lin1_true) >= 0.02 on 3/3 seeds.
- L2: |NS(lin1_true) - NS(lin1)| <= 0.01 on 3/3 seeds.
Kill: L1 fails -> the shortfall is the predictor (route: LIN1 on a sharper predictor, or error-aware s_k). L2 fails with
L1 passing -> both costs are real (route: report the split). L1 passes -> route: LIN2, P(|z - g| < r) under the full
8-D linear Gaussian (noncentral chi-square from the full Jacobian, D VJPs per candidate). Claim tests in
`test_r3_claims.py`, RED against a control directory before any read.

**A6.4 (registered 20:03 IST, after (0.9, 3) seed 0 only; before any other round-3 cell-seed is read).** Seed 0 at (0.9, 3):
gap 0.0105 hit < 0.02 yet dj4L - dj02 = 0.170 NS. If seeds 1 and 2 agree, P2c dies as registered. Suspected cause: the
hit gap is the wrong unit; NS divides by Bayes - blind, so the room an arm can close in NS is closable = 1 - NS(dist)
(read so far: (1, 1) ~0.10 no edge, (0.75, 3) ~0.18 no edge, (0.9, 3) s0 0.39 edge, (1, 3) ~0.66 edge).
- P2n (replacement, pre-run for (0.85, 3), (0.95, 3) and seeds 1, 2 of (0.9, 3); seed 0 of (0.9, 3) suggested it and is
  flagged): per cell, [dj4L - dj02 >= 0.05 NS on 3/3 seeds] == [mean over seeds of 1 - NS(dist) >= 0.25].
  Kill: edge where closable < 0.25, or no edge where closable >= 0.25. `test_r3_claims.py::test_P2n_*`.

**A6 label correction (2026-09-28 20:32 box time, Inspector 30-min pass; amendment text unchanged).** The labels "19:55 IST"
on A6 and "20:03 IST" on A6.4 were written by hand and are wrong. Box file times: A6 was appended before
`r3/foreman/test_r3.py` (19:52:37) and before the first r3 job (19:53:42); A6.4 was appended at 19:58:55 (BAR.md mtime,
same write as `test_P2n`), after `res_f0.9_q3.0_s0.json` (19:58:27) and before `res_f0.9_q3.0_s1.json` (20:02:01).
Also recorded: the first `audit 0 1` job wrote `audit_f0.0_q1.0_s{0,1}.json` (19:54:17, 19:54:51; hashing A6 without A6.4)
before it was stopped at 19:54:59 for host RAM and before `test_r3_claims.py` existed; neither file was opened before
L1/L2 went RED on the control directory.

**A6 read (20:38 box time; files `r3/foreman/results/`, `table.json`; tests `test_r3.py` 4/4, `test_r3_claims.py` 13/16).**
- A6.1: old rule reproduces r2's stored Bayes pick on 15/15 cell-seeds; tie rate 0.3-8.6 % (highest at (0, 3)); Bayes hit
  moves <= 0.0007; NS moves >= 0.005 only at (0, 3) s0/s1 (<= 0.013) and (0.75, 3) s0/s1 (<= 0.030). No round-2 verdict
  (V1, P1, P2, F1, M1-M3b, S1) changes sign or side of its bar.
- A6.2: P2c KILLED at (0.9, 3) and (0.95, 3): hit gap 0.0086 / 0.0160 (< 0.02) with dj4L - dj02 = +0.170/+0.120/+0.079 and
  +0.158/+0.244/+0.254 NS. (0.85, 3) consistent. Forecast wrong: hit gap < 0.02 at 0.95 too, so its onset is in (0.95, 1].
  P2n (A6.4) consistent in 3/3 cells, closable 0.244 / 0.309 / 0.424; (0.85, 3) is a knife edge (0.244 vs 0.25; edges
  +0.105 / -0.023 / +0.149, paired SE 0.062-0.064 NS per seed, so the 0.05 bar is ~1 SE). Seed-level edge rises with
  closable NS across all 9 points (Spearman ~0.98). V passes 3/3 cells.
- A6.3: L1 PASSES (bayes_succ - lin1_true = 0.034 / 0.024 / 0.044); L2 KILLED (lin1_true - lin1 = 0.026 / 0.026 / 0.022).
  Both costs are real. Repricing P1's LIN1 leg with true dynamics: mean NS(lin1_true) 0.964 vs best set arm (dj4L) 0.969,
  |diff| 0.005 <= 0.01: the round-2 LIN1 half-kill was the learned predictor, not LIN1.
- Routes: P2n on fresh cells with the 0.25 threshold frozen ((0.8, 2), (0.9, 2)) at n_eval 60,000 so per-seed edge SE
  <= 0.6 x bar; LIN2 (full-Jacobian noncentral chi-square) on true dynamics against bayes_succ for the 0.024-0.044 residue.
