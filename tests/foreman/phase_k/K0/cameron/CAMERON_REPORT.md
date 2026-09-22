# CAMERON K0 — R-DEPTH bed, best responses, Y3

**Verdict.** The pinned `wald.py` and `yukawa.py` numbers are real. An independent rebuild reproduces every number to the printed digit. Two problems in R-DEPTH as written come out of the audit:

- **The L = 6 bar fires on the opponent's own best response.** The perfect-doubling twin scores 0.2988 once it is allowed to guess on the tokens it cannot resolve, and the bar is "stays under 0.254".
- **A single training length cannot pin the logit law.** The logit scale that just fits training scores 0.2551 at 16k.

Both have measured replacements. My optimist route (a centered resolvent that would remove the ln n law) died on its own test.

All code and outputs are in `SP/phase_k/K0/cameron/` (SP = the session scratchpad). The instance scripts are read-only at `tests/foreman/phase_k/instances/` (wald.py sha256 31271d64…, yukawa.py feb03137…).

## §3 table: pinned vs rebuild

The rebuild is independent. It builds the logit matrix, applies a row softmax (or a real sort-based sparsemax), then does a scipy triangular solve of (I − γW) z = V. It does not use the author's recurrence. It runs on the pinned bed (`default_rng(23)`, n = 4096, 16 lanes) with γ = 0.999. The seed band is the 2.5 / 50 / 97.5 percentiles over 200 draws of the same generator (seeds 0..199).

| Arm | Contract / wald.out | Rebuild, pinned bed | Seed band q2.5 / median / q97.5 |
|---|---|---|---|
| Max depth / median depth | 269 / 127 | 269 / 127.5 | median max depth 283; minimum of 200 draws is 269 (one tie) |
| Position/recency floor (most recent root) | 0.0645 (chance 0.0625) | 0.064453 | 0.0598 / 0.0664 / 0.0742 |
| L = 4 doubling ceiling (depth ≤ 2^L, zero credit) | 0.0664 | 0.066406 (= 272/4096) | exact whenever every chain has more than 16 tokens |
| L = 6 doubling ceiling | 0.2539 | 0.253906 (= 1040/4096) | exact whenever every chain has more than 64 tokens |
| L = 8 doubling ceiling | 0.9856 | 0.985596 | 0.9680 / 0.9777 / **0.98536**: pinned is above q97.5 |
| L = 4 / 6 / 8, same opponent guessing at its 2^L-th ancestor | not in contract | **0.1230 / 0.2988 / 0.9861** | — |
| Resolvent, s = 8 | 0.0869 | 0.086914 | 0.0676 / 0.0841 / 0.0935 |
| Resolvent, s = 12 | 0.9954 | 0.995361 | 0.9809 / 0.9961 / 1.0000 |
| Resolvent, s = 16 / 20 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | — |
| Resolvent, sparsemax | exact by construction | 1.0000 (real sparsemax, margin 2) | — |

What "perfect pointer doubling" reaches. The input gives each token its parent (jump 2^0). Each layer composes two pointers, so the set of known jump lengths D satisfies D_{ℓ+1} = D_ℓ + D_ℓ and the longest jump doubles. After L layers a token holds its 2^L-th ancestor, saturating at the root. It resolves exactly the tokens at depth ≤ 2^L, so the exact ceiling is mean(depth ≤ 2^L). wald.py counts the root as depth 0 and gives zero credit elsewhere. Extra heads do not beat this: the sumset of known jump lengths still at most doubles per layer.

## Y3: pinned vs rebuild

My rebuild is a vectorised recurrence over the whole s-grid at once. On n ≤ 4096 it agrees with the dense solve through the W rows above. The reads use yukawa.py's own definitions.

| n | Contract s_50 / width / offset | Rebuild on yukawa's beds (seed = n) | 12 fresh beds: s_50 min / median / max | Width, median | Offset, median |
|---|---|---|---|---|---|
| 1k | 8.92 / 1.81 / −1.48 | 8.92 / 1.81 / −1.48 | 8.81 / 9.02 / 9.22 | 1.66 | −1.38 |
| 4k | 10.73 / 2.61 / −2.44 | 10.73 / 2.61 / −2.44 | 10.45 / 10.64 / 10.74 | 2.81 | −2.53 |
| 16k | 13.00 / 3.82 / −2.94 | 13.00 / 3.82 / −2.94 | 12.77 / 12.87 / 13.02 | 3.69 | −3.07 |

- **The struck law stays struck.** Bar: struck if the seed-median offset falls by more than 0.5 from 1k to 16k. It falls by 1.69.
- **The broadening survives seeds.** The median width grows by 2.03 from 1k to 16k.
- **The surviving slope is 1.17 per ln n from 1k to 4k and 1.61 from 4k to 16k (seed medians).** The first is below the contract's "about 1.3–1.6".
- **acc(s) is monotone on all three pinned beds**, so the `np.interp` read is valid.
- **Notes on yukawa.py:**
  - Its `default_rng(31)` is never consumed: Y1, Y2 and Y4 are deterministic, and Y3 seeds each bed with n.
  - s_10 is read at the chance-adjusted level 0.15625, but s_90 at the raw 0.9.

## The K1 bed and its floors (`bed_k.py`, `k1_floors.py` → `k1_floors.json`)

Conventions are the author's (`make_bed` draw, verbatim):
- Every token picks one of 16 lanes uniformly and independently (`rng.integers(0, 16, n)`); its parent is the previous token in that lane.
- Lane identity is not in the content.
- Every token must output its root's id, and every token counts.

My K1 additions:
- **Ids:** n distinct ids drawn from a vocabulary of V = 32768 (`rng.permutation(V)[:n]`), so ids carry neither position nor lane. A root's parent id is NULL = 32768.
- **Training bed:** n = 1024. Depth is capped at 32 by restarting a lane with a new root. This gives about 38 roots and exact depth ≤ 32.
- **Test beds:** n ∈ {4096, 8192, 16384}, uncapped, 16 roots each.
- **Evaluation:** `evaluate(pred_ids, bed, Ls)` returns overall accuracy, accuracy on depth > 2^L, and accuracy by depth bin.
- **Floor seeds:** training beds `default_rng([1, k])`, test beds `default_rng([2, n, k])`, k = 0..7. Training streams must use a different seed family.

Floors are the mean over 8 beds; ranges are in the JSON.

| Bed | Max depth | Chance | Recency floor | Beyond-2^L opponent (chance-credited) | Resolvent s_99 | Resolvent acc at s = 16 / 20 |
|---|---|---|---|---|---|---|
| Train n = 1024 | 32 | 0.0263 | 0.1016 | L=3: 0.080, L=4: 0.100 | 11.25–11.5 | 1.0 / 1.0 |
| Test 4k | 275–296 | 0.0625 | 0.0653 | 0.041–0.062 (every L ≤ 8) | 12.0–12.25 | 1.0 / 1.0 |
| Test 8k | 529–555 | 0.0625 | 0.0636 | 0.049–0.062 | 13.0–13.25 | 1.0 / 1.0 |
| Test 16k | 1067–1120 | 0.0625 | 0.0637 | 0.063 (L ≤ 9), 0.115 at L=10 (1.2% of tokens) | 14.25–14.5 | 1.0 / 1.0 |

Doubling ceilings on the test beds, exact / chance-credited (the (a_loop) arm with T iterations reads the L = T row):

| L | 4k | 8k | 16k |
|---|---|---|---|
| 3 (f_R's ALiBi part at R0) | 0.0352 / 0.0946 | 0.0176 / 0.0782 | 0.0088 / 0.0710 |
| 4 (a_L, R0) | 0.0664 / 0.1239 | 0.0332 / 0.0929 | 0.0166 / 0.0783 |
| 6 (a_L, R1) | 0.2539 / 0.2997 | 0.1270 / 0.1808 | 0.0635 / 0.1223 |
| 7 (a_{L+3}, R0) | 0.5039 / 0.5341 | 0.2520 / 0.2979 | 0.1260 / 0.1809 |
| 8 (a_L, R2) | 0.9778 / 0.9786 | 0.5020 / 0.5323 | 0.2510 / 0.2981 |
| 9 (a_{L+3}, R1) | 1.0 | 0.9845 / 0.9853 | 0.5010 / 0.5324 |
| 11 (a_{L+3}, R2) | 1.0 | 1.0 | 1.0 |

## Findings: RED path, exact RED line, status, replacement

Every test was run once as a stub before its implementation existed. Each stub RED is `ModuleNotFoundError: No module named 'bed_k'` (or `'k1_floors'`, `'centered'`), and each is logged in `red_*.log` and on the board. The clock times in the test headers are my estimates and run 1–7 minutes late. The file mtimes (tests 01:45:48–01:46:31, bed_k.py 01:47:34, test_logit_law 01:51:13 before k1_floors.py 01:51:42) and the board order are the record.

1. **K0_wald_rebuild**: **still RED.**
   - Path: `…/test_wald_rebuild.py`.
   - RED line after implementation: `FAIL seed band L8: pinned 0.9856 in [q2.5, q97.5] : [0.967999, 0.977661, 0.985358]`.
   - 13 of 14 bars pass, including every pinned number reproduced exactly.
   - The pinned bed is an upper-tail draw on the L = 8 ceiling: its deepest chain (269) ties the shallowest of 200 other draws (median 283).
   - Route: every K1 floor and ceiling is reported over at least 8 beds per n. The W row stays the pinned instance (L-REPRO) and is not the bed's typical value.
2. **K0_doubling_ceiling_chance_credit**: **GREEN.** `PASS L=6 credited opponent 0.2988 > 0.2739`.
   - R-DEPTH's clause "(a_L) stays under its doubling ceiling (0.254 at L = 6)" is broken by the doubling opponent itself once it guesses.
   - Route: score (a_L) on depth > 2^L of the arm being scored. The opponent's best response there is about chance (bed means 0.041–0.063 over L and n). The counter "(a_L) > 0.5 beyond 2^L" stands. The overall-accuracy ceiling becomes the chance-credited one.
3. **K0_gamma_window**: **GREEN.** `PASS pinned bed s=16: gamma 0.9 -> 0.7798, 0.99/0.999/0.99999 -> 1.0000, 1-1e-10 -> 0.0605`.
   - γ → 1 collapses onto token 0's chain: W_00 = 1 puts a pole at γ = 1.
   - **Strike on my own board event.** Two sentences in my first gamma-window event were written before being measured. Both were wrong, and I struck them in a follow-up event (`K0_gamma_window_correction`):
     - "the lower edge needs 1/(1−γ) ≳ depth": γ = 0.99 scores 1.0 at depth 269.
     - "use 0.9999 at 16k": γ = 0.99–0.99999 all score ≥ 0.9964 at 16k.
   - Route: K1 keeps γ = 0.999, which is inside the window at 4k and 16k. Never use γ = 1.
4. **K0_y3_rebuild**: **GREEN**, 9/9 bars (numbers in the Y3 table above).
   - Route: the contract should read "seed-median slope 1.2–1.6 per ln n" and "Y3 beds seeded by n".
5. **K0_logit_law_unidentified**: **GREEN.** `PASS s_train = 11.50; 16k median acc at s_train = 0.2551; 16k s_99 = 14.50; implied a = 1.082 per ln n`.
   - Training at a single n cannot identify a in s = a·ln n + b.
   - Route, measured on the hand-set head (no bar): train f_R and (a_ss) on n ∈ {256, 512, 1024} at depth ≤ 32.
     - s_99 is 9.75 / 11.00 / 11.50 at those lengths, which fits a = 1.26, b = 2.87.
     - The extrapolated s = 13.38 at 4k and 15.13 at 16k scores 1.0000 / 1.0000 (median of 8 test beds).
6. **K0_centered_resolvent** (my optimist route): **RED, killed.**
   - Exact RED lines:
     - `FAIL (a) n=4096 centered sigma0 = 1.0 at s=8,12 : (0.5596, 1.0)`
     - `FAIL (b) n=4096 sigma1 s50 centered <= softmax - 1 : (10.80, 11.72)`
     - `FAIL (c) sigma1 slope centered < softmax : {softmax 1.528, centered 1.600}`
   - Subtracting the row background is exact only in exact arithmetic. The true-root signal is a product of parent weights and decays exponentially with depth, so float64 residue of the cancellation swamps it. Centering moves s_50 by a constant but grows faster with n (slope 2.31 vs 1.28 at σ = 0).
   - Replacement route: the learned SSMax coefficient from multi-length training (item 5). That is one scalar per head and gives up nothing.
7. **K0_k1_bed**: **GREEN**, 10/10 checks:
   - invariants over 50 training draws;
   - 16 roots on every test bed;
   - max depth 1087 at 16k;
   - an oracle scores 1.0;
   - `ceiling_credit` equals a slow parent-walking reference at L = 2, 3, 5.
   - Route, for what K1 must fix: at R2 the L + 3 = 11 twin has no tokens beyond 2^11 at any n ≤ 16k. At R1 (twin at 9), 4k has no tokens beyond 2^9 and 8k has 1.6%. For those arms, test at n = 32k–64k or score them on L only.

## OPEN

- Trained arms (f_R, a_L at L and L + 3, a_loop, a_ss) are K1 work. Whether the *trained* SSMax a lands at ≥ 1.1 is K1's first read.
- The resolvent solve in fp32/bf16 at 16k is unmeasured here; the rebuild is float64 only. This is Chase's lane.
- The 2^L ceiling is still the best *known* strategy, not a proven lower bound. The Sanford–Hsu–Telgarsky lower-bound content is unread beyond the abstract.
- The FLOP-matched loop count T for (a_loop) needs the K1 R-COST measurement.
- No nurses were dispatched; every number above is mine.

**Standing state.** The pinned instances hold, and the K1 bed and floors are fixed. Two R-DEPTH clauses need the replacement wording before any K1 bar is written:
- (a_L) is scored beyond 2^L of the arm itself;
- f_R and a_ss train on n ∈ {256, 512, 1024}.

Default: K1 proceeds on `bed_k.make_train` / `make_test` / `evaluate` with both replacements and γ = 0.999.
