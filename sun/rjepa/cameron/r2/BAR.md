# R-JEPA / Cameron round 2 bar (registered 2026-09-28 02:35 IST, before any r2 run; runs record this file's sha256)

Bed: the round-1 torus standard-map bed, unchanged (`../BAR.md`: Ks 1.5, K = 8, delta 1e-3, r = sqrt(pi/2), goal (0.5, 1.0)),
code reused from `../rj.py`. NEW seeds only: eval/train seeds {3, 4, 5}; tuning seed {9}. Seeds 0/1/2 are never read.
Band (fixed by the brief, not re-derived): T in {12, 16, 20, 24, 32}. N_eval 10,000, N_train 20,000 per lead per seed.
A band lead is void if, on the 3-seed mean of the new seeds, bayes - floor < 0.03 or bayes - blind < 0.10.
Normalised score NS = (s - blind)/(bayes - blind), per seed per lead. Label-shuffle null: every arm scored against
within-start shuffled successes must land within 0.01 of blind, else every row of that lead is void.

## Claim A: LIN1 rebound (fresh RED; round-1 LIN1 result is struck and not used as evidence)

Arms: `floor`, `ens3` (3 posterior members, ties to floor), `lin1` (Phi((r - d)/s), s = delta ||J^T n||, ties to floor),
`dj` and `djl` (the Amendment-A1 D-JEPA-spec operator, 1,000 updates of 8 and of 256 starts); `best_dj` = max(dj, djl) per lead per seed.
Learn gate: dj and djl trained at T = 2 on each seed must reach train NS >= 0.90, else that seed's dj rows are void.

- **A1 (beats the D-JEPA spec):** on EACH of seeds 3, 4, 5: band-mean NS_lin1 >= band-mean NS_best_dj + 0.05, and at every
  band lead NS_lin1 >= NS_best_dj - 0.01.
- **A2 (beats the floor):** on each seed, band-mean NS_lin1 >= band-mean NS_floor + 0.05.
- **A3 (holds against the ensemble control):** 3-seed mean NS_lin1 >= NS_ens3 - 0.01 at every band lead.
  Registered knowing round 1 lost this at T = 24 and 32; it is tested, not assumed.

Verdict: LIN1 REBOUND iff A1 and A2 hold. A3 decides the scope: if A3 fails, the rebound claim is only "beats the D-JEPA
spec at 2T calls per candidate", and at the leads where it fails ENS3 (3T calls per candidate) is the better cheap ranker.
KILL (A1 or A2 fails): replacement route = ENS3 as the cheap ranker, repriced by calls (3T vs 2T per candidate).

## Claim B: fewer predictor calls at Bayes quality

Call accounting (per decision): one application of the one-step predictor to one state = 1 call; one tangent-linear
(JVP) or adjoint (VJP) step = 1 call (sensitivity with 2 calls per tangent step reported). Hits are computed once, and
every arm's picks are read from the same member draws, so an arm that spends all calls reproduces `bayes` exactly.

- `bayes` (the full-rollout reference): M = 256 posterior members per candidate, all K candidates, T steps, plus the
  K centre rollouts for the tie-break: C_full = K (M + 1) T = 2,056 T.
- `prune_m` (descriptive, m in {1, 2, 3, 4}): LIN1 for all K (forward + 1 VJP: 2 K T), then all M members only for the top-m by
  LIN1 chance; argmax of in-ball fraction among them, ties to floor. Cost 2 K T + m M T.
- `cascade` (deciding): stage 0 rolls each candidate's centre with its full 2x2 tangent (3 T per candidate, stopped early
  at the step t where delta ||J_t||_2 first exceeds kappa: that candidate is "mixed", scored p = 1/K (the ball's area
  fraction), cost 3 t). A candidate is "settled" if s < R/2 and |(R - d)/s| > z_c (p = 1 or 0, no members). If a settled-1
  candidate exists, pick the settled-1 with the smallest d and stop. Otherwise race the unsettled candidates on the
  shared member draws in rounds of b members: after n members, drop k if p_k + c/sqrt(n) < max_j p_j - c/sqrt(n);
  stop when one is left or n = M. Pick = argmax over {raced p, mixed 1/K, settled-0 zero}, ties to floor. Cost = stage 0
  + T x (members spent).
  Grid: z_c in {2, 3, 5, inf}, kappa in {inf, 2 pi, 25}, c in {0.2, 0.35, 0.5, 0.75, 1.0}, b in {8, 16, 32}.
  Selection (tuning seed 9 only, before any eval seed is read): the config with the smallest band-summed calls among
  those with NS >= 0.995 at every band lead. Written to `tune.json` and frozen.

- **B (FEWER WIN):** the frozen cascade config, on EACH of seeds 3, 4, 5, has NS >= 0.99 (i.e. within 0.01 of `bayes`)
  at every band lead, and mean calls per decision <= C_full / 4 at every band lead. The reported number is the
  3-seed calls ratio C_full / C_cascade per lead.
- **B KILL:** NS < 0.99 at any band lead on any seed, or ratio < 4 at any band lead. Replacement route: `prune_m` with the
  smallest m reaching NS >= 0.99 on 3/3 seeds (reprice at its measured ratio), else retire with the measured
  Pareto front (calls vs NS) as the reason.
- Null for B: the cascade pick scored on shuffled successes lands within 0.01 of blind.

## Routes priced without a run

- Shared-prefix rollouts: in this bed the impulse acts at t = 0 and members differ at t = 0, so candidates and members
  share no state prefix; the saving is 0 by construction. The only shared prefix is in tangent space (roll the centre, spawn
  members late); that is stage 0 of `cascade` in its limit and is not run separately this round.

## Tests (RED first)

`test_r2.py`: implementation tests (cascade with c = z_c = kappa = inf (no pruning) reproduces bayes at K (M + 3) T calls; prune_K equals
bayes; call accounting; settle and mix rules) and claim tests (A1, A2, A3, B read from `eval.json`). Logged RED against
the stub `r2.py` before any run.

## Amendment B2 (2026-09-28 02:44 IST, after the seed-9 tune of B, before any eval seed was generated or read)

Seen on tuning seed 9 (`tune.json`): the frozen B config is the pure Hoeffding race (z_c = kappa = inf, c = 0.35, b = 32) at
calls ratio 2.4 / 3.0 / 3.5 / 3.6 / 3.4 over the band, so B is expected to die at the ratio bar (< 4). The mixing stop
(kappa = 2 pi or 25, p := 1/K) caps NS at 0.71-0.80 on seed 9: dead as priced. LIN1 settling reaches 23-60x but caps NS at
0.976-0.985 with s < R/2. B stays registered and is run as frozen.
Added arm `cascade2` (deciding under the same bar as B, "B2"): kappa = inf; empirical-Bernstein radius
rad = c sqrt((p (1 - p) + 1/n)/n); settle threshold s < tau; epsilon-good stop (a row stops when the leader's
p - rad >= every other live p + rad - eps). Grid: z_c in {3, 5, inf}, tau in {R/2, 0.2, 0.05}, c in {0.5, 1, 1.5, 2},
b in {8, 16, 32}, eps in {0, 0.02, 0.05}. Same selection rule on seed 9 (NS >= 0.995 at every band lead, min calls), frozen in
`tune2.json`. B2 WIN / KILL exactly as B, with replacement route the same.

## Amendment R1 (2026-09-28 02:57 IST, PRE-RUN: after a session crash, while `tune2` on seed 9 was still running, before any eval seed was generated or read)

The previous session died mid-run; `tune2` (PID 26336) survived and is left to finish; its `tune2.json` is used as written.
Changes to `r2.py` that touch no bar and no arm: `torch.set_num_threads(4)` -> 2 (host shared, RAM critical), and the
eval writes `eval_partial.json` after each seed and resumes from it, so an OOM kill loses one seed, not three; a seed
killed mid-run is rerun from scratch (same rng streams), never partially reused. `eval.json` therefore records a
different `script_sha256` from `tune.json`/`tune2.json`. No bar, grid, seed, arm or threshold changes.

## Amendment R2 (2026-09-28 03:06 IST, PRE-RUN: no eval row has been read into any claim; two eval attempts were stopped)

The single-process eval (attempt 1, 02:57:42) printed one seed-3 row (T = 12) and was then stopped by its author at
03:02:37: host free commit fell to 765 MB of 60 GB, and the process held 2.7 GB of private commit from importing the
torch cu126 build. A stray relaunch (attempt 2, 03:02:59) was stopped at 03:03:48 before it wrote any row. Both attempts'
seed-3 cache files were deleted. The seed-3 T = 12 row stays in `eval.log` and is not used. The run is split by memory, not
by result: `evnp` (every arm except dj/djl; the torch import is stubbed out; ~150 MB) writes `eval_np.json`, `evdj` (the
D-JEPA-spec arms and their learn gate, on the same cached eval batches and the same train rng streams) writes `eval_dj.json`,
and `eval` merges them into `eval.json`. Same arms, same rng streams, same numbers as the single-process design. No bar,
grid, seed or threshold changes.

## Amendment R3 (2026-09-28 03:41 IST, POST-RUN: records verdicts only, changes nothing above)

`eval.json` (bar sha cec21d0e, script sha 73883184), `pytest test_r2.py` -> 10 passed, 2 failed (`green_eval.log`).
A1 GREEN, A2 GREEN, A3 RED at T = 32 only (3-seed NS lin1 0.7589 vs ens3 0.7791): LIN1 REBOUND, scoped - at T = 32 ENS3 is
the better cheap ranker. B RED (Hoeffding cascade ratio 2.34-3.56 < 4; NS 0.9812 at seed 4 T = 32): KILLED; its
registered replacement `prune_m` (m <= 4) does not reach NS >= 0.99 on 3/3 seeds at T = 24, 32 either. B2 GREEN:
Bernstein epsilon-good race, NS >= 0.9908 on all 15 seed-lead rows, 3-seed calls ratio 7.78 / 7.87 / 6.98 / 5.94 / 4.53.
