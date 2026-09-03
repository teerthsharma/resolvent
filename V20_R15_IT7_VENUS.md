# V20 R15 — it.7 — VENUS (IRENE)

**Contract** `CEQ_V20_R15_CONTRACT.md:125` (clause (1), absolute `CP-lower > 0.5`).
**Basis** `results/v17k_r4_retake.jsonl` (24 cells, seeds 0–7 × 3 arms) +
`results/v20_r15_it6_seeds8_15.jsonl` (10 cells, `arm_smprime`, seeds 0–1 control + 8–15).
**Test** `tests/venus/test_v20_r15_it7_venus.py` — **16 passed**. Every number below
is asserted there. `[RUN]` unless marked.

---

## 0. VERIFICATION — the brief re-measured before anything is built on it

Every claim in the it.7 brief checks out. Verified independently, not accepted:

| brief's claim | verified | where |
|---|---|---|
| controls (seeds 0,1) reproduce bitwise | **yes — 15 fields, `repr`-identical, 0 diffs** | `test_controls_reproduce_bitwise` |
| fresh seeds 8–15 crossings | **0 of 8** | `test_rate_prediction_survived` |
| pooled `arm_smprime` | **1 of 16** (only seed 2) | ″ |
| seeds 11, 13 failed at `0.99462890625`, `0.976806640625` | **yes** | `test_mechanism_prediction_falsified` |
| `0.4967041015625` in the fresh eight | **0 occurrences** | ″ |
| `0.0` in the fresh eight | **0 occurrences** | ″ |
| distinct `frac_gate_annihilated` over 16 cells | **5** — `0.0, 0.4967041015625, 0.5032958984375, 0.976806640625, 0.99462890625` | `test_two_basins_is_dead` |
| `lambda_hat == −inf` on all ten it.6 cells | **yes, incl. 11 & 13 where `unit_root == False`** | `test_L1_*` |
| `arm_pl` 5/8, `softmax` 0/8, no fresh cells | **yes — it.6 header `arms: ["arm_smprime"]`** | `test_it6_ran_only_w1`, `test_arm_rates_unchanged_*` |
| symmetric `arm_pl` experiment ≈ `8 × 1.780` | **14.240 GPU-s exactly; mean 1.7800** | `test_symmetric_experiment_price` |

**One correction to the brief, and it favours nothing of mine.** The symmetric W3
experiment is **14.240 / 178.460 = 7.98 %** of the it.6 run's cell-seconds
(7.67 % of its 185.746 s wall). That is **about a twelfth, not a tenth.** The
argument gets *stronger*, not weaker, and the number should be the true one.

### 0.1 An unplanned RED, verbatim, and what it found

The lattice test (§2, L2) failed on first run — not a planted negative, a real one:

```
E               KeyError: 'frac_gate_annihilated_0step'
tests\venus\test_v20_r15_it7_venus.py:117: KeyError
1 failed, 15 passed in 0.54s
```

Cause: **`softmax` cells carry no `frac_gate_annihilated_0step` key at all —
8/8 absent, while `arm_pl` and `arm_smprime` carry it on 26/26.** `[RUN]` The
0-step probe MARS's predictor reads is **not instrumented on one of the three
arms.** MARS's rule therefore cannot be evaluated on `softmax` even in principle,
and any pooled statement over "all cells" touching the 0-step probe is scoped to
26, not 34. Filed here because it is a live hole in the instrument, not the test.

---

## 1. TASK A — THE SCORE, TAKEN EXACTLY

### 1.1 What survived

**The rate.** Forecast **1 of 8**, 95 % predictive interval **0–3**. Observed **0**.
Inside the interval, and at the mode of the Beta-Binomial the it.5 posterior
implies. **SURVIVED.**

**Seed 2's uniqueness on the clause-(2) statistic.** it.5 §2 claimed seed 2 is
*"the only cell in the tournament with a finite `lambda_hat` and `unit_root ==
False`"*. Over 24 cells then; **over 34 cells now, still exactly one:**
`[('arm_smprime', 2)]`. `[RUN]` `test_L1_is_independent_of_unit_root`. Seeds 11
and 13 carry `unit_root == False` but `lambda_hat == −inf`, so they do **not**
add a second candidate. **This is the load-bearing half of the 0.20 I placed on
W1, and the retake did not touch it.**

**Every crosser anywhere has a live gate.** Across all 34 cells and all three
arms, `eval_nrmse < floor₁ ⟹ frac_gate_annihilated == 0.0`. 6/6, zero
exceptions. `[RUN]` `test_every_crosser_anywhere_has_a_live_gate`.

### 1.2 What died, stated without softening

**FALSIFIED — the failure-set claim.** it.5, filed 2026-09-02:
*"Every new seed that fails to cross carries `frac_gate_annihilated ∈
{0.4967041015625, 0.5032958984375}`."* **Seeds 11 and 13 failed outside that
set.** Not a near miss — `0.99462890625` and `0.976806640625` are roughly
**double** the predicted values.

**FALSIFIED — "two basins, not a continuum."** it.5 §1 reason 3. Sixteen cells,
**five** distinct values. The inference *"eight seeds produce exactly two
non-zero values, a continuum of bad fits would not do that"* was drawn from a
sample of eight and did not survive doubling it.

**UNTESTED — the crossing half.** *"Every new seed that crosses carries
`frac == 0.0` and a finite `lambda_hat`."* Zero crossings occurred, so this
clause was never exercised. It is **not confirmed by the fresh eight** and this
office does not count it.

### 1.3 The falsifier's blindness — the actual finding, and it is mine

The named killer: *"one seed that crosses with an annihilated gate — or one that
**fails with a live gate** — falsifies it outright."*

The world produced **failures with a gate deader than predicted.** My killer
watched one tail of one variable and the claim broke in the other tail. The
mechanism I asserted was **set membership on `frac`**; the killer I wrote was a
**predicate on gate liveness**. Those are different variables. A killer that does
not range over the same quantity the claim constrains cannot fire on the claim.

**This is M-7 exactly** (`MISTAKES.md:529-540`): *"Rows must be **exhaustive
before the data**."* it.5 enumerated the two observed values and called the
enumeration a law. That is the E-ladder hole recommitted in a new file, by the
office that had read M-7. **The office's own rule caught it, one iteration late.**

### 1.4 The repair — falsifiers as complements, not as predicates

For a claim *"failures carry `X ∈ S`"*, the killer is *"any failure with
`X ∉ S`"*, `S` fixed before the run. Every claim below is stated that way. Where
`S` was previously an enumeration of sightings, it is now a **lattice** — a set
defined by a rule, whose complement is everything else.

---

## 2. THE RE-FILED MECHANISM — three laws, each with a complement-falsifier

Pre-registered **2026-09-02, it.7, before any seed 16+ exists.**

### L1 — gate death alone determines `−inf`

> For every cell of every arm:
> **`frac_gate_annihilated > 0` ⟺ `lambda_hat == −inf`.**

`[RUN]` **34/34 cells, zero violations.** 17 with `frac > 0`, all `−inf`; 17 with
`frac == 0`, all finite.

**Falsifier (complement, both directions, exhaustive):** *any* cell with
`frac > 0` and `lambda_hat` finite, **or** *any* cell with `frac == 0` and
`lambda_hat == −inf`. `lambda_hat` is `−inf` or it is not; there is no third
branch, so the two conditions partition the outcome space.

**This corrects it.5, and the correction is a strengthening.** it.5 read `−inf`
alongside a unit root. **`unit_root` is irrelevant** — seeds 3, 11, 13 all carry
`unit_root == False` with `−inf`. The counterexample (seed 3) was **already in
the table it.5 printed** and this office did not read its own table. `[READ]`
`V20_R15_IT5_VENUS.md:122`.

### L2 — the annihilation lattice

> `frac_gate_annihilated × 8192` is an integer, for every cell that reports the
> field. `S = {k/8192 : k ∈ ℤ, 0 ≤ k ≤ 8192}`, stated in full **before** the run.

`[RUN]` 26/26 cells, trained and 0-step (softmax omits 0-step, §0.1).
Observed `k` over the 16 `arm_smprime` cells: **`{0, 4069, 4123, 8002, 8148}`**.
`8192 = 2·s² = 2·64²`. `[DERIVED]` `4069 + 4123 = 8192` exactly — the two
half-values are **complementary**, which is why it.5 saw them as a pair.

**Falsifier:** *any* cell whose `frac × 8192` is not an integer. The complement
of a lattice is the rest of the reals, so this fires on every possible break.
**This is the shape it.5's `S` should have had.**

### L3 — dose-response, which replaces "two basins"

> `eval_nrmse` is monotone increasing in `frac_gate_annihilated` over
> `arm_smprime`. Three separated clusters, no overlap:

| cluster | `frac` | n | `eval_nrmse` range | reading |
|---|---|---|---|---|
| live | `0.0` | 1 | **0.203920** | crosses floor₁ |
| half-dead | `≈0.50` | 13 | 0.852061 – 0.926717 | skyline neighbourhood |
| near-total | `≥0.9` | 2 | **1.120603 – 1.203324** | **worse than predict-the-mean (1.0000000843)** |

`[RUN]` Spearman **ρ = 0.5564, p = 0.0252**; Kendall **τ = 0.4472, p = 0.0316**;
n = 16. Each cluster's maximum is strictly below the next cluster's minimum.

**Falsifier (three disjoint conditions covering both tails and the middle):**
1. *any* cell with `frac ≥ 0.9` and `eval_nrmse ≤ 1.0`; **or**
2. *any* cell with `frac == 0` and `eval_nrmse ≥ 0.7071067811865476`; **or**
3. pooled Spearman over 24 cells reaching `p ≥ 0.05`.

**Prediction for seeds 16–23** (`arm_smprime`, same cell): all three hold.

**Why this is better than what it replaces.** "Two basins" was a count of
sightings. L3 is a **direction**: the M1 corner descent does not stop at half —
on 2/16 cells it runs to `8148/8192` and `8002/8192`, and those cells land *past
predict-the-mean*. it.5 had the descent right and its **extent** wrong.

### 2.1 The rate, re-forecast

Pooled **1/16**. Jeffreys posterior `Beta(1.5, 15.5)`; Beta-Binomial over 8 fresh:

> **Expected crossings in eight fresh seeds: 0.706. 95 % predictive interval 0–3.**
> `P(0) = 0.5399`, `P(≥3) = 0.05489`, `P(≥4) = 0.01487`.

`[RUN]` `test_predictive_interval_for_next_eight`.
**Falsifier: ≥ 4 crossings in eight fresh seeds** (`p = 0.0149` under this
posterior). The interval is unchanged from it.5 at 0–3 while the mean fell
1 → 0.706; doubling n narrowed the centre, not the tail.

---

## 3. TASK B — WHAT 1-IN-16 DOES TO THE RANKING

### 3.1 The numbers, both conventions named

| quantity | it.5 (1/8) | it.7 (1/16) | move |
|---|---|---|---|
| `p̂` W1 | 0.1250 | **0.0625** | halved |
| CP-lower, one-sided 95 % | 0.0064 | **0.0032** | halved; **156× short of clause (1)'s 0.5** |
| CP-upper, one-sided 95 % | 0.4707 | **0.2640** | already below 0.5 both times |
| Fisher exact, W1 vs W3 | 0.11888 | **0.00686** | **the one that moved** |
| `P(θ_W1 < θ_W3)`, Jeffreys | 0.9828 | **0.9986** | +0.0158 |

`arm_pl` CP-lower **0.2892** — unchanged, and **also fails clause (1)**.
`softmax` 0/8, CP-upper 0.3123.

### 3.2 The re-filed ranking

> **1. W3 `arm_pl` — 2. W1 `arm_smprime`**

| ordering at it.29 | it.5 | **it.7** | Δ |
|---|---|---|---|
| **W3 ≻ W1** | 0.72 | **0.78** | **+0.06** |
| **W1 ≻ W3** | 0.20 | **0.14** | **−0.06** |
| no ordering — both eliminated on clause (1) | 0.08 | **0.08** | 0 |

### 3.3 What moved me, and the honest size

**It barely moved, and it should not have.** +0.06 is the whole update.

**The datum that moved it — one, named:** the Fisher exact between the arms went
`0.11888 → 0.00686`. That is the first iteration in which W1 and W3 separate at
conventional significance rather than merely differing in point estimate.
Everything else in §3.1 is the same conclusion carried to more digits.

**Why it did not move more — the part it would be dishonest to skip.** My 0.20 on
W1 **never rested on the crossing rate.** It rested on clause (2), BED-K(a),
unrun, where seed 2 is the only cell in the tournament carrying the statistic
that bed reads. **The retake did not test that, and the statistic survived the
enlargement from 24 cells to 34 unchanged.** A rate datum cannot discharge a
position that was not held on rate grounds. Treating 1-in-16 as a large update
would be **double-counting a prior already conditioned on** — the it.5 ranking
sat at 0.72 *because* W1's rate was bad.

**The −0.06 that did come off W1 is L3's, not the rate's.** Under "two basins",
seed 2's basin was one of a small number of discrete outcomes and plausibly
reachable at a fixed rate. Under L3 it is the extreme tail of a monotone descent
whose typical realisation is *half* the gate and whose worst is *nearly all* of
it. **A tail is rarer than a basin.** That is a mechanism update, and it is the
only thing this iteration produced that is entitled to move a probability.

### 3.4 The asymmetry, priced — and it does matter

**16 W1 cells against 8 W3 cells.** The resampled arm is the one whose rate fell.

**Does it change the ranking? No. Does it weaken the evidence for it? Yes, in a
way the ranking's own margin conceals.** W3's 5/8 rests on eight cells with
CP-lower `0.2892` and CP-upper `0.8889` — an interval **0.60 wide**. W1's is now
`0.0032–0.2640`, **0.26 wide**. The comparison's uncertainty is now dominated
almost entirely by the arm nobody resampled. Three of W3's eight cells fail by
divergence (`a_hat_max` `12.77 / 49.66 / 116.01`); whether that is 3-in-8 or
6-in-16 is **unmeasured**, and it is the number clause (1) actually turns on.

**What it would take to fix: 14.240 GPU-s.** `[RUN]` eight fresh `arm_pl` seeds
8–15 at the cells' own measured basis (`8 × 1.7800`), **7.98 % of what the W1
experiment cost**. This is the cheapest experiment on the board by an order of
magnitude and the only one that makes the round's headline comparison balanced.

**This office's recommendation, a preference not a ruling:** run it before any
further W1 sampling. A round that resamples only the losing arm and reports the
pooled loss has **chosen** its asymmetry, and 14 GPU-s is too little to justify
keeping it. `[NOTE]` GPU launch requires the author's explicit yes; none is
assumed here, and nothing was run.

**Pre-registered forecast for those eight, filed now:** **5 of 8 crossings**
(point), 95 % predictive interval **2–7**; `frac_gate_annihilated == 0.0` on
**8 of 8** (L1/L2: `arm_pl`'s gate has never died, 8/8 cells).
**Falsifier: any `arm_pl` cell with `frac_gate_annihilated ≠ 0.0`** — complement
form, fires on every value including the ones I have not seen.

---

## 4. TASK C — THE FORECASTER LEDGER, it.7 UPDATE

Scored against the it.6 measurement. **A ledger, not a judgement.**

| # | office | claim | outcome | grade |
|---|---|---|---|---|
| 15a | **VENUS** | rate: 1 of 8, PI 0–3 | observed **0 of 8** | **SURVIVED** |
| 15b | **VENUS** | failures carry `frac ∈ {0.4967041015625, 0.5032958984375}` ∧ `λ = −inf` | seeds 11, 13 at `0.9946`, `0.9768`; **λ-conjunct held 10/10** | **FALSIFIED in `frac`; the λ half survived** |
| 15c | **VENUS** | crossings carry `frac == 0` ∧ finite `λ` | zero crossings | **UNTESTED — not confirmed** |
| 15d | **VENUS** | "two basins, not a continuum" | 5 values / 16 cells | **FALSIFIED** |
| 15e | **VENUS** | seed 2 unique in finite `λ` ∧ `unit_root == False` | 1/34 cells, still unique | **SURVIVED, scope doubled** |
| 15f | **VENUS** | the named killer *"fails with a live gate"* | claim broke via a **deader** gate; killer blind | **HOLE — M-7 class** |
| 17 | **MARS** | `frac_0step < 0.15` ⟹ crossing at `nrmse < 0.3` | **zero positive predictions** over the fresh eight (min `0.380859375`) | **UNTESTED, not confirmed — vacuous agreement.** One supporting instance in 16 (seed 2, `0.124755859375`). Also **uninstrumented on `softmax`**, 8/8 (§0.1) |
| 18 | **it.6 instrument** | controls reproduce | seeds 0,1 identical on 15 fields | **SURVIVED** |

**On row 17:** an implication with a false antecedent is true and says nothing.
MARS's predictor did not pass a test; **it was not given one.** It is worth
~2 GPU-s to test properly — the pooled basis shows `frac_0step` spanning
`0.1248 – 0.9946`, so a wider seed sweep will eventually produce an antecedent.
Until one does, this row stays open.

---

## 5. THE LEAP ROW

`V20_R15_LEAP_LEDGER.md` was absent when §0 was written and present when this row
was filed — JUPITER/MARS opened it mid-iteration (`02:51`). **Appended, not
opened.** Reproduced here.

> **Failed claim.** "W1's failing cells occupy two discrete basins,
> `frac ∈ {0.4967041015625, 0.5032958984375}`" — `V20_R15_IT5_VENUS.md` §2.2,
> filed 2026-09-02, falsified the same day by seeds 11 and 13.
>
> **Grade.** **LEAPABLE.**
>
> **The field, not the theorem:** **the extent of corner descent in
> multiplicative-gate landscapes** — how far along the annihilating face a
> gradient descent travels before it stalls, as a function of depth, and whether
> the stalling points form a lattice. `[NOTE]` The field is named; no theorem is
> claimed, and none is available from 16 cells.
>
> **Why leapable rather than terminal.** The it.5 claim failed by being *too
> small*, not by pointing the wrong way. Annex M1's β-gradient descent to the
> corner is **observed on 15 of 16 cells**; what it.5 got wrong was assuming the
> descent halts at half the gate. It does not — it reaches `8148/8192` and
> `8002/8192`, and those cells land **past predict-the-mean**. A claim that fails
> by underestimating the reach of a mechanism it correctly identified leaves the
> mechanism standing and hands the next iteration a **quantity to measure** (the
> descent's stopping distribution over the `k/8192` lattice) rather than a
> **bound to respect**.
>
> **What converts it to TERMINAL.** If seeds 16–23 return `k` values off the
> `8192` lattice, the stopping points are not discrete, there is no distribution
> over lattice sites to characterise, and this row becomes TERMINAL with the
> bound *"gate annihilation is a continuous quantity and the corner is not
> isolated."* **That is L2's falsifier, and it is why L2 is stated as a lattice.**

---

## 6. SCOREBOARD

- **Filed, testable, dated 2026-09-02:** L1, L2, L3, the 0.706 / PI-0–3 rate, the
  `arm_pl` 5-of-8 forecast. Each carries a complement-form falsifier.
- **Retracted:** two basins; `−inf` tied to `unit_root`; the `S` of two values.
- **Ranking:** W3 ≻ W1 at **0.78** (+0.06). Both arms fail clause (1).
- **`⟨AUTHOR_COUNTER_RANKING⟩ NOT MEASURED.`** D-CALIB-3 authorises sign only;
  the author's measured bias is POSITIVE (7 `+` / 1 `−`, one-sided sign test
  `p = 0.0352`, `V16_CALIBRATION.md:19,:50,:94-100`). No size is invented.
- **Cheapest next measurement on the board: 14.240 GPU-s**, eight fresh `arm_pl`.
- **Open instrument hole filed:** `frac_gate_annihilated_0step` uninstrumented on
  `softmax`, 8/8 cells.
- **No git writes. Nothing touched Kaggle. No GPU run was launched.**
