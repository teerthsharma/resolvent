# R-JEPA / Cameron round 4 bar (registered 2026-09-28 21:20 IST, before any seed-14/15/16 value exists; runs record this file's sha256)
(Header time corrected before any run: the file was written between 21:19:19 and 21:20:41 by the shell clock; the first draft said 21:21.)

Bed: the round-1 torus standard-map bed, unchanged (`../BAR.md`, `../rj.py`); batch generator and arms imported read-only from
`../r2/r2.py` and `../r3/r3.py` (`r2.batch`, `r2.cascade`, `r3.race_budget`, `r3.p_ref`). Round-4 batches cache ONLY under
`r4/cache/` (gitignored). FRESH seeds {14, 15, 16}; seeds 0-13 are never read by a claim (spent seeds 6/7/8/11/12/13 are read
only by the record fixes in section 3). N_eval 10,000 rows per (seed, lead), M = 256 members, K = 8, C_full = K (M + 1) T.
Frozen config from `../r3/tune.json`: c 1.5, b 8, eps 0.05, z_c = kappa = inf (B2' = B2).

## Seen before registering

Round 3 (`../r3/BAR.md` V/V2, `../r3/sharp.json`, `../r3/long.json`): NS'_b4 at B = 640 on seeds 6/7/8/11/12/13 =
0.99170 / 0.99116 / 0.98981 / 0.99061 / 0.99086 / 0.98916 (mean 0.99055); half-rollout NS' 0.98861-0.99048.
Long leads (realised NS, seeds 6/7/8): T = 40 lin1 0.7511 / 0.7643 / 0.7474 vs ens3 0.7887 / 0.7838 / 0.7377;
T = 48 lin1 0.7386 / 0.7588 / 0.7743 vs ens3 0.7200 / 0.7293 / 0.7564. T = 32 sharp: lin1 0.761-0.775 < ens3 0.773-0.782 on 6/6.
Nothing on seeds 14/15/16 or at T = 36/44/56 has been generated.

## 1. Claim B: B4 at budget 656 under the sharp instrument (T = 32, seeds 14/15/16)

Arm `b4_656` = `r3.race_budget(ev, 1.5, 8, 0.05, B = 656)`. B = 656 is the largest multiple of b = 8 with
C_full / (3 K T + B T) = 2056 / 680 = 3.024 >= 3, so p90 ratio >= 3 holds by construction (tested, not assumed).
Instrument (r3 Amendment H, unchanged): P_ref = hit fraction of 1,024 fresh posterior members per candidate, rng stream
[seed, 32, 3]; NS'_a = (mean P_ref[a] - mean P_ref) / (mean P_ref[bayes] - mean P_ref), bayes = 256-member full rollout.
Validity: void if bayes_ref - blind_ref < 0.10 or bayes_ref - floor_ref < 0.03 on a seed.
- **B (WIN):** on EACH of seeds 14, 15, 16: NS'_b4_656 >= 0.99 and C_full / p90(calls) >= 3.
- **B-route (registered now, decided in the same run):** on EACH of seeds 14, 15, 16: NS'_b4_656 >= NS'_half, where
  half = full rollout on members 1-128 (ratio 2.0). This is the control-relative tolerance r3 V2 named.
- **B KILL:** replacement route = B-route if it passes (the claim becomes "B4 at p90 3.02 dominates the 2.0-ratio full
  rollout", tolerance set by a control, not a constant). If B-route also dies: retire p90 >= 3 at T = 32 and price the
  tail from the (B, NS') curve below (next bar: smallest B with NS' >= 0.99 on 14/15/16, frozen, priced on fresh seeds).
- Descriptive only: B in {640, 704, 768, 896, 1024}; b2p (uncapped B2'); ens80; lin1; ens3; floor.

## 2. Claim R: the LIN1/ENS3 reversal and its mechanism (T in {36, 40, 44, 48, 56}, seeds 14/15/16)

Instrument for R (member split, no winner's curse): members 1-3 = ENS3; members 4-128 = the ceiling pick `ceil`
(125-member rollout) and the regime class; members 129-256 = reference P_ref128. LIN1 reads no member. Neither arm, nor the
class, touches the reference, so every expected-success difference below is unbiased.
NS''_a = (mean P_ref128[a] - mean P_ref128) / (mean P_ref128[ceil] - mean P_ref128). gap(T) = NS''_lin1 - NS''_ens3 (3-seed mean).
Validity per lead: void if on the 3-seed mean ceil_ref - blind_ref < 0.10 or ceil_ref - floor_ref < 0.03.

**R0 (reversal replicates):** gap < 0 at T = 36 and 40, gap > 0 at T = 48 and 56 (valid leads only). T = 44 carries no
prediction (the crossing is placed in (40, 48] by round 3).
- R0 KILL: the round-3 T = 48 LIN1 win was seed noise. Route: ENS3 is the cheap ranker for T >= 32 and LIN1's scope is T <= 24
  (the round-3 F route), priced on these seeds.

**RM (mechanism: regime composition).** Hypothesis: LIN1's Gaussian tail is right where the linearisation is valid and again
once the predictive has wrapped the torus (it then ranks "certain hit > mixed ~ 1/K > certain miss" correctly, where ENS3's
3 draws of a ~1/K probability are noise that falls to the floor tie-break), and wrong only in between, where the predictive is
a folded filament. The reversal is the fraction of decisions in the folding regime rising and then falling with T.
Regime of a row = final linearised spread s* = delta ||J_T||_2 (`stretch[..., -1]`) of the `ceil` pick:
S (s* < R/2, the cascade's registered tau), F (R/2 <= s* < 2 pi), X (s* >= 2 pi, one torus period). Thresholds fixed here, a priori.
g_c = mean over the class-c rows (3 seeds x 5 leads pooled) of (P_ref128[lin1] - P_ref128[ens3]) / (ceil_ref - blind_ref)
of that row's (seed, T).
- **RM1 (sign pattern):** g_F < 0 and g_X > 0, each over >= 300 rows; g_S >= 0 if S has >= 300 rows (else not read).
- **RM2 (composition explains the lead dependence):** with w_c(T) the class fractions at lead T and g_c(T) the per-lead class
  gaps, pred(T) = sum_c w_c(T) g_c (pooled g_c, lead-free) matches gap(T) within 0.01 at every valid lead, and has the same sign
  wherever |gap(T)| >= 0.01.
- RM KILL: route = the ENS3-saturation hypothesis next round (gap(T) tracks the rate at which ENS3's top score is a tie broken
  by the floor; registered then, with its own RED), and the reversal stays unexplained in the record.
- Descriptive: ENS3 floor-tie rate per lead; w_c(T); g_c(T); realised NS for every arm (as rounds 2-3).

## 3. Record fixes

**I (bind the instrument finding, RED first).** r3's "realised-outcome NS cannot resolve 0.01 at T = 32 (~ +-0.011)" was
logged before any test (Inspector: unbound). It is kept and bound here. Prediction: conditional on y and the picks, the realised
success of any arm is one posterior draw, so for arm a vs bayes
D_real = mean(succ[a] - succ[bayes]) = D_exp + noise, D_exp = mean(P_ref[a] - P_ref[bayes]),
sd = sqrt(sum_i V_i) / N, V_i = P_a + P_b - 2 P_ab - (P_a - P_b)^2 with the joint P_ab from the same 1,024 fresh members.
z = (D_real - D_exp) / sd over arms {b4_656, b4_640, b2p, half, ens80, lin1, ens3, floor} x seeds {14, 15, 16} (24 values).
- **I (BOUND):** RMS(z) in [0.6, 1.5], max |z| <= 3.5, and the NS-scale noise sd / (bayes_ref - blind_ref) of b4_656 lies in
  [0.005, 0.02] on each seed (the realised noise is the size of the 0.01 tolerance).
- I KILL: the finding is dropped from the record; the sharp instrument stays (it needs no noise claim: it is the conditional
  expectation of the same quantity) and the retirement is written into the verdict.

**Record fix 1 (dated 2026-09-28, Inspector r3):** in `../r3/BAR.md` Amendment V2, "B4 beats the 128-member full rollout ...
(0.9889-0.9905)" should read **NS'_half 0.98861-0.99048** (`../r3/sharp.json`, seeds 8 and 12). `../r3/BAR.md` is not edited:
its sha256 is recorded by every r3 result file.
**Record fix 2:** "7/12 half-rows below 0.99" (r3 Amendment V) is in no file. It is recomputed from the cached r3 batches into
`fixes.json` (realised NS at T = 32, members 1-128 and members 129-256, seeds 6/7/8/11/12/13) and the line is replaced by that
file's count; if the recount differs, the recount stands.

## Tests (RED first)

`test_r4.py`: implementation tests (cache under r4; B = 656 obeys its budget and the budget-infinite race is r2.cascade; the
member split is disjoint and covers 1-256; the class thresholds; the z formula on a synthetic case with known answer) and the
claim tests B, B-route, R0, RM1, RM2, I read from `sharp.json` / `rev.json`. Logged RED (`red.log`) before `r4.py` exists.

## Amendment V (2026-09-28 21:55 IST, POST-RUN verdicts, changes nothing above)

Every result file (`fixes.json`, `sharp.json`, `rev.json`) records bar sha256 7e977e9c... and script sha256 b52d736c...
(`registered_sha.txt`, taken before any claim was read; this amendment changes the file's sha, the registration text above
does not change). Nurse audit (sonnet, read-only, before any claim test ran): no discrepancy on B, B-route, I, R0, RM1, RM2;
two low notes, both the tests failing loud on an empty validity set (R0 needs a valid lead on each side, RM2 a non-empty set),
stricter than the bar text. Neither guard fired. `pytest test_r4.py` -> 9 passed, 3 failed (R0, RM1, RM2) (`green_eval.log`).

- **B GREEN.** NS'_b4_656 = 0.99101 / 0.99213 / 0.99021 on seeds 14 / 15 / 16 at T = 32; p90 ratio 3.06, mean ratio
  5.10 / 5.11 / 5.05; validity bayes_ref - blind_ref 0.224-0.226, bayes_ref - floor_ref 0.098-0.102. Margin on seed 16 is
  0.0002. Descriptive: B = 640 on the same seeds scores 0.99051 / 0.99186 / 0.99007, so these three seeds would also have
  passed at 640; across the nine seeds scored so far at B = 640 (r3 six + these three) 2/9 sit below 0.99. B is a pass on
  the line, not clear of it.
- **B-route GREEN.** NS'_b4_656 beats the 128-member full rollout on 3/3 (half 0.98921 / 0.98853 / 0.98911).
- **I GREEN (bound).** RMS z = 0.675 over 24 arm-seed values, max |z| = 1.46; realised-NS noise of b4_656 is
  sd 0.0081 / 0.0079 / 0.0083 in NS units. The realised instrument cannot resolve 0.01 at T = 32; seed 14 shows it: realised
  NS_b4_656 = 0.9893 (a fail) against NS' 0.99101 (a pass).
- **R0 KILLED at T = 40.** gap(T) = NS''_lin1 - NS''_ens3, 3-seed mean: T 36 -0.0008, 40 +0.0056, 44 +0.0038, 48 +0.0140,
  56 +0.0182 (all five leads valid: ceil - blind 0.133-0.199, ceil - floor 0.061-0.091). No ENS3 win at T = 40 on fresh seeds;
  LIN1 ties at 36 and wins 3/3 at 48 and 56. The round-3 T = 40 ENS3 win (realised, 0.016) did not replicate. On the same rows
  the realised gap at T = 44 is -0.0142 against +0.0038 under the member split, so the realised instrument flips signs at this
  scale. Read with r3 sharp T = 32 (lin1 < ens3 on 6/6 by 0.004-0.012), the data fit one crossing in (32, 40] with the gap
  growing after it. The round-3 "non-monotone" F pattern has no support once the instrument is sharp.
- **RM1 KILLED.** Pooled over 150,000 rows: g_S = +0.0290 (23,855 rows), g_F = **+0.0476** (8,641; predicted < 0),
  g_X = +0.0010 (117,504). LIN1 wins most where the pick sits in the folding regime, which is the reverse of the hypothesis.
- **RM2 KILLED at T = 36.** pred(T) = 0.0103 / 0.0088 / 0.0080 / 0.0072 / 0.0064 against gap -0.0008 / 0.0056 / 0.0038 /
  0.0140 / 0.0182; |error| 0.0111 > 0.01 at T = 36. pred is flat and falling while gap rises, so class composition does not
  carry the lead dependence. The within-class gap does: g_X(T) = -0.0108 / -0.0025 / -0.0059 / +0.0073 / +0.0149.
  ENS3 tie rate is almost flat (0.502 -> 0.527).

Routes:
- R0: the registered route (ENS3 for T >= 32, LIN1 scope T <= 24) assumed the T = 48 win was the noise; the fresh data say the
  T = 40 loss was the noise. Replacement, registered for round 5: under the member-split instrument, a single crossing
  T* in (32, 40] (T in {32, 34, 36, 38, 40}, fresh seeds), LIN1 >= ENS3 - 0.002 at every T >= 36, ENS3 > LIN1 at T = 32.
  Until then LIN1's scope stays T <= 24 plus T >= 48 (3/3 here), with T in (24, 48) unresolved.
- RM: the registered route (ENS3 saturation, gap tracking the ENS3 floor-tie rate) goes forward with its own RED next round.
  Its evidence is thin: the tie rate moves 0.025 while gap moves 0.019. Sharper form, seen here post-hoc and so unbound: the
  lead dependence lives in the X class (g_X crosses 0 between T = 44 and 48). Round-5 hypothesis: within X rows, g_X tracks
  how close the candidates' P values are to one another (across-candidate spread of P_ref128 falls with T, ENS3's 3 draws lose
  resolution before LIN1's ordering does). RED: g_X(T) is monotone decreasing in that spread across T, 3 fresh seeds.

Record fixes:
- Fix 1 (dated 2026-09-28): `../r3/BAR.md` V2 "(0.9889-0.9905)" reads **0.98861-0.99048** (`fixes.json` fix1, from
  `../r3/sharp.json`).
- Fix 2 (dated 2026-09-28): r3 V "7/12 half-rows below 0.99" reads **8/12**. Realised NS at T = 32 of the 128-member
  rollout, members 1-128 / 129-256, seeds 6/7/8/11/12/13: 0.9930/0.9830, 0.9872/0.9902, 0.9845/0.9819, 0.9845/0.9783,
  0.9812/0.9890, 0.9940/0.9922 (`fixes.json` fix2). The r3 range "0.9812-0.9940" covered members 1-128 only; both halves
  span 0.9783-0.9940.
- Instrument finding: bound (claim I GREEN above). The Inspector's "unbound" mark is cleared as of this round.
