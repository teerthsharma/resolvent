# R-JEPA / Cameron round 3 bar (registered 2026-09-28 19:52 IST, before seed 10 or seeds 6/7/8 were generated; runs record this file's sha256)

Bed: the round-1 torus standard-map bed, unchanged (`../BAR.md`, code `../rj.py`); arms and batch generator imported
read-only from `../r2/r2.py` (`r2.batch`, `r2.cascade`, `r2.prune`, `r2.GRID2`, `r2.tune`). Round-3 batches cache under
`r3/cache/`, never under `r2/`. NEW seeds only: tuning seed {10}; eval seeds {6, 7, 8}. Seeds 0-5 and 9 are never read by
any claim (seed 9 was read once before registration, see "Seen before registering").
Band as round 2: T in {12, 16, 20, 24, 32}; long leads for claim F: T in {40, 48}. N_eval 10,000 per lead per seed, M = 256.
Lead validity (as round 2): void if, on the 3-seed mean, bayes - floor < 0.03 or bayes - blind < 0.10, or any seed's
label-shuffle null (floor, bayes, lin1, ens3 and the deciding cascade arms) deviates more than 0.01 from blind.
NS = (s - blind)/(bayes - blind). C_full = K (M + 1) T = 2,056 T calls per decision; call accounting exactly as round 2.

## Seen before registering

Round 2 (`../r2/BAR.md`, `../r2/eval.json`): B2 (c = 1.5, b = 8, eps = 0.05, z_c = inf) on seeds 3/4/5 has mean ratio 4.52-7.89,
p90 ratio 3.84 at T = 12-20, 3.10 at T = 24, 2.33-2.34 at T = 32. On spent seed 9 only (B2 config, eps varied):
T = 24 eps 0.05 / 0.10 / 0.15 -> NS 0.9955 / 0.9972 / 0.9889, p90 ratio 3.06 / 5.04 / 6.59; T = 32 -> NS 1.000 / 0.9978 /
0.9843, p90 ratio 2.32 / 3.84 / 5.59. That sized the eps grid below. No seed-10, 6, 7 or 8 value has been seen.

## Claim D: B2 survives re-derivation (B2')

Procedure identical to round 2's `tune2`: grid `r2.GRID2`, Bernstein radius, selection on tuning seed 10 only = the config
with the smallest band-summed mean calls among those with NS >= 0.995 at every band lead. Frozen in `tune.json`.
- **D (B2' WIN):** on EACH of seeds 6, 7, 8, at every valid band lead: NS_B2' >= 0.99 and mean calls <= C_full / 4, and
  the B2' pick scored on shuffled successes lands within 0.01 of blind.
- Descriptive, not deciding: the round-2 B2 config (seed-9 tune) on seeds 6/7/8, and whether B2' equals B2.
- **D KILL:** any of the above fails. Replacement route: the round-2 B2 config priced on seeds 6/7/8 if it passes there
  (two tunes, one survives -> tune on the union of seeds 9 + 10); else retire the ratio claim to the measured
  (calls, NS) Pareto front of the seed-10 grid and price `prune_m` as round 2.

## Claim E: lead-dependent eps lifts the tail

Rule (registered here, tuned on seed 10 only, after D's tune): (z_c, tau, c, b) frozen at B2'. For each band lead T,
eps_T = the SMALLEST eps in {0.05, 0.075, 0.10, 0.125, 0.15, 0.20} whose seed-10 row has NS >= 0.995 and p90 ratio
C_full / p90(calls) >= 3.3 (a 10% margin over the bar); if none reaches 3.3, the eps with the largest p90 ratio among those
with NS >= 0.995. Frozen in `leadeps.json`. Arm `B3` = r2.cascade with eps = eps_T.
- **E (TAIL WIN):** on EACH of seeds 6, 7, 8, at every valid band lead: NS_B3 >= 0.99, p90 ratio >= 3, and null within
  0.01 of blind.
- **E KILL:** any of the above fails. Replacement route: a hard member cap n_max(T) on the race (stop at n_max, pick the
  leader): p90 calls <= K T (3 + n_max), so n_max <= 2056/(3 K) - 3 = 82 guarantees ratio >= 3 by construction and the
  question becomes NS only; priced on seed 10 next round.

## Claim F: LIN1 vs ENS3 beyond the band

Seeds 6, 7, 8 at T in {40, 48}, same validity rule. Arms: floor, bayes, lin1 (2T calls per candidate), ens3 (3T),
ens2 (2T, the price-matched control; descriptive).
- **F (LIN1 HOLDS):** at each valid long lead, 3-seed mean NS_lin1 >= NS_ens3 - 0.01.
  Registered knowing round 2 lost this at T = 32 by 0.020; the stance is that it is tested, not assumed.
- **F KILL:** fails at any valid long lead. Replacement route: ENS3 as the cheap ranker for T >= 32, and LIN1's
  scope stated as T <= 24; if ens2 also beats lin1 there, LIN1 loses at equal price and the route is ENS2.
- If both long leads are void, F is VOID (neither win nor kill), with the bed's bayes - floor gap as the reason.

## Tests (RED first)

`test_r3.py`: implementation tests (cache path is r3; lead-eps rule on a toy table; B3 with a constant eps table equals
r2.cascade; seed-10 tune reproduces r2.tune's selection rule) and claim tests D, E, F read from `eval.json`/`long.json`.
Logged RED (`red.log`) before `r3.py` exists.

## Amendment E' (2026-09-28 19:57 IST, after tune.json and leadeps.json on seed 10, while `eval` on seeds 6/7/8 runs, before any eval or long row was read)

Seen on seed 10: B2' = B2 exactly (c 1.5, b 8, eps 0.05, z_c inf). Lead-eps table: eps_T = 0.05 at T 12/16/20/32 and 0.075 at
T = 24; at T = 32 no eps > 0.05 keeps NS >= 0.995 (0.075 -> 0.9878, p90 2.99), so the rule falls back to 0.05 and B3 = B2' at
T = 32 with seed-10 p90 ratio 2.32. E is therefore expected to die at T = 32; it stays registered and is read as frozen.
Seed 9 had put eps 0.10 at NS 0.9978 for T = 32 while seed 10 puts it at 0.9742: the eps-vs-NS curve at long leads does not
transfer across tuning seeds.
The E replacement route is promoted to a tested arm, with NO tuning: `B3cap` = B3 with a hard member cap n_max = 80 at every
lead (race stops at n = 80, pick = leader among the survivors, ties to floor). p90 ratio >= 2056/(8 (3 + 80)) = 3.10 by
construction, so E' is a test of NS only.
- **E' (CAP WIN):** on EACH of seeds 6, 7, 8, at every valid band lead: NS_B3cap >= 0.99, p90 ratio >= 3, null within 0.01.
- **E' KILL:** replacement route = a cap that grows with lead only where the tail lives (n_max = 256 at T <= 20 where B2'
  already has p90 >= 3.7; cap at T >= 24), or retire the p90 claim with the seed-10 (n_max, NS) curve as the reason.
Seed 10 B3cap numbers are computed in the same job and are descriptive only.
Test amendment T1 (same time, before any claim test read eval.json): the validity rule in `test_r3.py` now reads the
registered null arms (floor, bayes, lin1, ens3 + the deciding arms) from each row's per-arm nulls instead of
`null_max_dev` (long rows also folded the descriptive ens2 into it), and claims D/E/E' assert a non-empty band (nurse audit).
Timestamp correction (19:58): the header originally read 19:55, a guess written before the clock was checked; the file was written before `tune` started at 19:52:55 (run_all.log), and tune.json/leadeps.json record the pre-amendment sha256. Header corrected to 19:52; no bar text changed.

## Amendment G (2026-09-28 20:18 IST, POST-READ of D/E/F/E' on seeds 6/7/8; registers a new arm on FRESH seeds 11/12/13, none generated yet)

Read so far (verdicts recorded in Amendment V below): E and E' are killed at T = 24/32. Priced on seed 10 (exploration, not
evidence): common-random-number pairing is worth nothing here (top-2 member-hit correlation in the p90 tail: median 0.005),
so a paired race is retired without a run; the tail rows are 3-5-way near-ties (top-2 gap median 0.02, leader P ~ 0.31).
The per-candidate cap wastes budget: in a 4-way tie it stops at 4 x 80 members when the ratio bar allows ~660.
Arm `B4` = the B2' race (Bernstein radius, c 1.5, b 8, eps 0.05, z_c inf, stage 0 3T per candidate) with a TOTAL member
budget per decision: a row stops before any round whose members would push its total past B = 640, and picks the leader
among the survivors (ties to floor). B is not tuned: it is the largest multiple of 64 with K T (3) + T B <= C_full / 3,
i.e. p90 ratio >= 2056 / (24 + 640) = 3.10 by construction. With B = infinity, B4 is r2.cascade exactly (tested).
Seed-10 exploration numbers for this exact arm: NS 1.0002 / 1.0021 / 0.9952 at T = 12 / 24 / 32, mean ratio 7.77 / 6.35 / 5.17.
- **G (BUDGET WIN):** on EACH of fresh seeds 11, 12, 13, at every valid band lead: NS_B4 >= 0.99, p90 ratio >= 3, mean
  ratio >= 4, null within 0.01 of blind. Seeds 6/7/8 are run in the same job and reported, descriptive only.
- **G KILL:** replacement route = sequential halving over the race survivors with the same budget (spends on the top half
  instead of all survivors), else retire the p90 claim at T = 32 with the measured (B, NS) curve as the reason.

## Amendment V (2026-09-28 20:33 IST, POST-RUN verdicts, changes nothing above)

`pytest test_r3.py` -> 8 passed, 4 failed (`green_eval.log`). tune.json, leadeps.json, eval.json record bar sha 188d7c20.
- D GREEN: B2' (seed-10 retune) = B2 exactly; NS >= 0.9922 on all 15 seed-lead rows of seeds 6/7/8, mean ratio 4.54-7.87.
- E KILLED: B3 p90 ratio 2.32 at T = 32 on 3/3 seeds (eps_32 fell back to 0.05), and NS 0.98998 at seed 7 T = 24.
- E' KILLED: cap-80 NS 0.9728-0.9896 on 7/15 rows (T = 20, 24, 32).
- F KILLED at T = 40 (3-seed NS lin1 0.7543 vs ens3 0.7701); at T = 48 LIN1 wins (0.7572 vs 0.7353, 3/3 seeds); lin1 beats
  the price-matched ens2 at both long leads (0.7278, 0.7143).
- G KILLED at T = 32: B4 NS 0.9836 / 0.9922 / 0.9789 on fresh seeds 11 / 12 / 13 (T = 16 void on the fresh set: seed-11
  lin1 null 0.0112). Descriptive seeds 6/7/8 at T = 32: 0.9930 / 0.9838 / 0.9866. p90 ratio 3.13, mean ratio 5.13-5.20 at T = 32.
Seen after the verdicts (numpy on cached batches): a full rollout on members 1-128 only (ratio 2.0, below every bar here)
scores NS 0.9812-0.9940 at T = 32 over seeds 6/7/8/11/12/13 (7/12 half-rows below 0.99). The realised-outcome NS cannot
resolve 0.01 at T = 32: an arm that disagrees with bayes on a fraction f of rows carries NS noise
~ sqrt(f 2 p(1 - p) / N) / (bayes - blind) ~ 0.011 at f = 0.15, p = 0.3, N = 10,000.

## Amendment H (2026-09-28 20:33 IST, before any 1,024-member reference is generated)

Instrument, not arm. Score each pick by its EXPECTED success under an independent reference: P_ref = hit fraction of
1,024 fresh posterior members per candidate (rng stream [seed, T, 3], never seen by any arm), and
NS' = (mean P_ref[pick] - mean P_ref) / (mean P_ref[bayes] - mean P_ref), where bayes is the 256-member full rollout the
arms are priced against. This removes the Bernoulli outcome noise and does not reward matching the arms' own draws.
T = 32 only (every kill above lives there), seeds 6, 7, 8, 11, 12, 13. Arms re-scored unchanged: b2p, b4 (B = 640),
b3cap (n_max 80), half (members 1-128, ratio 2.0), ens80, lin1, ens3, floor.
- **H (TAIL WIN UNDER A SHARP INSTRUMENT):** NS'_b4 >= 0.99 on EACH of the six seeds (p90 ratio 3.13 is fixed by construction).
- **H KILL:** the T = 32 tail-cost tradeoff is real, not instrument noise; route = retire p90 >= 3 at T = 32 and keep
  the mean-ratio claim (D) only, pricing the tail as B2' p90 2.32 at T = 32.
The D, E, E', G verdicts above stand as read; H does not revive them. It decides what the next bar should measure.

## Amendment V2 (2026-09-28 21:02 IST, POST-RUN verdict for H, changes nothing above)

`sharp.json` (T = 32, 1,024 fresh members, 270 s per seed). `pytest test_r3.py` -> 9 passed, 5 failed (E, E', F, G, H).
- H KILLED by 0.0002-0.0008: NS'_b4 = 0.9917 / 0.9912 / 0.9898 / 0.9906 / 0.9909 / 0.9892 on seeds 6 / 7 / 8 / 11 / 12 / 13.
- The sharp instrument works: B4's spread over six seeds fell from 0.0141 (realised, 0.9789-0.9930) to 0.0025. The T = 32
  tail tradeoff is real and small: p90 ratio 3.13 costs ~1.0% NS'.
- B4 beats the 128-member full rollout (ratio 2.0) on 6/6 seeds (0.9889-0.9905), at mean ratio 5.1 and p90 3.13.
  B2' scores NS' 0.9957-0.9972 (p90 2.32). cap-80 0.9766-0.9795, ens80 0.9776-0.9799.
Route (as registered): retire "p90 >= 3 within 0.01" at T = 32; D (mean ratio >= 4 within 0.01) stands. Next bar, if the tail
is pursued: score by expected success under fresh members (this instrument) and set the tolerance against the
half-rollout control, not a fixed 0.01.
