# V20 R15 — it.9 — MARS (MORIARTY)

Branch `v17k-gate0`. Wall clock honoured. No git writes. Nothing touched Kaggle.

---

## 0. STRIKE F2 IS ACCEPTED WITHOUT QUALIFICATION

`V20_R15_IT567_INSPECTOR.md:486-500` struck the it.7 report for asserting a write
that did not occur. The strike is correct. `V20_R15_IT7_MARS.md:210-211` said five
rows had been appended to `V20_R15_LEAP_LEDGER.md` and that the file did not
exist. Both halves were false: JUPITER had opened the file at it.7 with eight
rows, and no `L-M` row was ever in it.

The mechanism is not "the rows were bad". The rows were graded, the Inspector
independently blessed two of their field names at `:476`. The mechanism is that
**a report was written from the draft rather than from the file**, and no read-back
was performed between drafting the table and asserting the append. That is the
same class as an unverified `RUN`: an evidence tag placed on an action that was
composed rather than executed. It is now on the record as MARS's.

**The repair rule adopted here, and it is cheap enough to be unconditional: no
report sentence asserting a write ships without a `grep` of the written file in
the same report.** §4 is that grep.

---

## 1. TASK A — THE ROWS, READ BACK FROM THE FILE

Appended at `V20_R15_LEAP_LEDGER.md:81-105`. Quoted below **from the file**, via
`sed -n` on the ledger, not from the it.7 draft.

`[RUN] grep -c "L-M" V20_R15_LEAP_LEDGER.md` → `6`
`[RUN] grep -rn "L-M1" --include=*.md .` now returns, alongside the four
Inspector lines and the it.7 draft, the line that was missing:

```
V20_R15_LEAP_LEDGER.md:85:| **L-M1** | **VENUS §2.2 body** — falsified 2 of 8, and the
falsifier could not fire | **F2** | ... | **LEAPABLE** | **error-statistical
severity (Mayo)** — the branch of it that fixes the rejection region as a
**proper subset of the hypothesis complement**, which is the property this
falsifier lacked |
```

The five rows as they now stand in the file, compressed to `#` / verdict / field
(the full HOW-BAD text is at `:85-89`):

| # | verdict | FIELD (LEAPABLE only), as written in the file |
|---|---|---|
| **L-M1** | LEAPABLE | **error-statistical severity (Mayo)** |
| **L-M2** | TERMINAL | none — a base-rate power bound, `1 − (15/16)^8 = 0.403`, `n ≥ 44` for 95 % |
| **L-M3** | LEAPABLE | **anti-concentration / small-ball probability** (Littlewood–Offord, Rudelson–Vershynin) |
| **L-M4** | split | TERMINAL for the three census states (a pinned draw); LEAPABLE for seeds 11, 13, same field as L-M3 |
| **L-M5** | TERMINAL | none — an instrument definition; `lambda_hat_live` is already journalled |

### 1.1 The F1 format rule, checked rather than assumed

`V20_R15_IT567_INSPECTOR.md:483` states the line: *"Naming a field and then saying
what you want from it is compliant; naming the want and calling it a field is
not."* Both MARS LEAPABLE head nouns are disciplines with textbooks, and both were
in the Inspector's own passing list at `:476`. The three TERMINAL rows name no
field, which is the rule rather than an omission.

**The row in the file that still violates F1 is not MARS's.** It is the VENUS row
at `V20_R15_LEAP_LEDGER.md:55-56`, struck at `:460-473` of the audit and
unrepaired at the time of this append. MARS does not rewrite another office's
grading; the violation is flagged **inside the ledger** at `:99-101` so it is
visible to a reader of the ledger and not only to a reader of the audit.

**JUPITER's it.8 rows L-9..L-12 were checked against the same rule and pass** —
`structured-sparsity kernel scheduling` (L-10), `rate-distortion /
channel-capacity theory` (L-11), `bifurcation theory / gradient-flow convergence`
(L-12) each lead with a discipline. That check was not requested; it is reported
because a format rule enforced on two offices and not on the third is not a rule.

---

## 2. TASK B — THE it.8 `arm_pl` CLAIM IS A ONE-CELL HINGE, AND THE FILE SAYS SO ITSELF

**Test: `tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py`. Two RED, one GREEN
control. Verbatim first run below.**

The brief hands me the claim as *"crosses 7 of 8 fresh, 12 of 16 pooled, against
`softmax` 0/8"*. Before reaching for STRIKE 10's eval-seed lever I asked the
cheaper question — **what does the file's own aggregate row say** — and the answer
is that the round's headline and the round's registered verdict are the same nine
cells read two ways.

### 2.1 The registered verdict on this file is `crosses: false` `[READ results/v20_r15_it8_armpl_b.jsonl, t="agg"]`

```
{"t": "agg", "kind": "arm_pl", "n": 9, "mean": 0.7175018067286933,
 "sd": 0.21245444406217273, "ci_lo": 0.5541948645454026, "ci_hi": 0.8808087489119841,
 "floor_1": 0.7071067811865476, "dist_to_floor": 0.010395025542145775,
 "crosses": false, "achieved_power": 0.2627460936916006, ...}
```

`crosses` is `bool(m + half < floor1)` at `scripts/v15_r1.py:909` — the **upper**
CI end must clear the floor. `[READ scripts/v15_r1.py:900-909]`

### 2.2 Eight of the nine cells cross. The ninth is seed 9. `[RUN]`

| seed | `eval_nrmse` | `lambda_hat` | `a_hat_max` | `gate_r2` | `frac_gate_annihilated` | `< floor_1` |
|---|---|---|---|---|---|---|
| 0 (control) | 0.6446726192039927 | −1.4324781894683838 | 1.4104527235031128 | 0.9888719478507275 | 0.0 | yes |
| 8 | 0.6248685523735131 | −1.4479899406433105 | 1.6481777429580688 | 0.9732529553038906 | 0.0 | yes |
| **9** | **1.281779592990027** | **+0.30839410424232483** | **292.2945556640625** | **0.050886682051081156** | **0.0** | **no** |
| 10 | 0.6398332532283637 | −1.4464397430419922 | 1.2551869153976440 | 0.9854557973057732 | 0.0 | yes |
| 11 | 0.6408209504227382 | −1.4612612724304200 | 1.2995500564575195 | 0.9729963042022197 | 0.0 | yes |
| 12 | 0.6762082068688114 | −1.4704492092132568 | 1.2870875597000122 | 0.9831772443281561 | 0.0 | yes |
| 13 | 0.6348616577583769 | −1.4591555595397950 | 1.3961308002471924 | 0.9762957696220326 | 0.0 | yes |
| 14 | 0.6338928504917678 | −1.4698615074157715 | 1.5189114809036255 | 0.9590363585727365 | 0.0 | yes |
| 15 | 0.6805785772206486 | −1.4400595426559448 | 1.4342579841613770 | 0.9812719581488555 | 0.0 | yes |

Eight cells span `0.6249 … 0.6806`, **sd `0.0203`**. The pooled sd the verdict is
computed from is **`0.2125`** — a factor of **10.5**, and all of it is one cell.

### 2.3 The verdict flips on that one cell, and the flip is exact `[DERIVED, RUN]`

Recomputed with the runner's own estimator (`statistics.fmean`/`stdev`,
`t.ppf(0.975, df=n−1)`, `scripts/v15_r1.py:899-909`):

```
n9 (as registered): n=9 mean=0.7175018067286933 sd=0.21245444406217273 ci_hi=0.8808087489119841 crosses=False
n8 (seed 9 removed): n=8 mean=0.6469670834460266 sd=0.02029964404207494 ci_hi=0.6639380105668372 crosses=True
```

The n=9 line reproduces the journalled `mean`, `sd` and `ci_hi` **bitwise** — that
is the GREEN control in the test file, and without it the two RED tests would be
reading a journal I could not reconstruct. With seed 9 removed the upper CI end
clears `floor_1` by **`0.0432`** and the registered verdict becomes `true`.

**So the round's first movement toward the north star is not 12 of 16 readings.
It is one exclusion decision.** And it is the *same* exclusion decision L-12 was
graded on at it.8 — a post-hoc cut on a coordinate read off the cells being cut.

### 2.4 What would make the exclusion legitimate, and it is not a leap

The cut is legitimate **iff a pre-registered rule names seed 9 without seeing its
`nrmse`.** The candidate rule already exists and is not MARS's: the it.8
divergence story's `lambda_hat > 0`. Seed 9 reads `+0.3084` where the other eight
read `−1.43 … −1.47`, and `gate_r2 = 0.0509` against `0.959 … 0.989`. **The rule
is stated in coordinates that are available before `eval_nrmse` is read**, which is
what a pre-registration needs.

**This is why §3's seed-9 question is not a curiosity — it is the load-bearing
question of the round.** If seed 9 is a fourth instance of the registered
divergence mode, `lambda_hat > 0` is a pre-registerable exclusion and the crossing
stands as an eight-cell result at `ci_hi = 0.6639`. If seed 9 is a **fourth
mode**, no registered rule covers it, the exclusion is post-hoc, and the
registered `crosses: false` is the only verdict the file supports.

### 2.5 A second defect in the same file, unselected `[RUN]`

The `t="probe"` row publishes CIs on quantities bounded above by 1:

```
"sign_acc_ci": [0.9544868408578528, 1.0179795219893695]
"gate_r2_ci":  [0.6370555844629411, 1.1121099749560528]
```

Both upper ends are **outside the estimand's own range**. A normal-theory
`t`-interval on a bounded, strongly bimodal statistic is not a coverage statement
here; `gate_r2` at `0.0509` on one cell and `0.96–0.99` on eight is exactly the
shape that breaks it. This is the second RED test. It is small, and it is
mechanical: the same estimator that produced `crosses: false` produced an interval
the quantity cannot occupy, from the same nine numbers.

**Replacement route, priced.** `[READ results/v20_r15_it8_armpl_b.jsonl, t="wall"]`
`secs_per_run_by_arm: {"arm_pl": 1.7332222222222222}`, total wall `19.588 s` for
nine cells including bar calibration. **A pre-registered rerun that settles the
exclusion — same nine seeds, exclusion rule declared before the run — costs under
20 GPU-seconds.** It is the cheapest open item in the round by two orders of
magnitude, and it has now been priced three times (L-6, L-12, here).

### 2.6 Verbatim first run, `tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py`

```
    def test_aggregate_verdict_agrees_with_per_cell_crossing_count():
        rs = rows()
        agg = [r for r in rs if r.get("t") == "agg"][0]
        cells = [r for r in rs if r.get("t") == "cell"]
        floor_1 = agg["floor_1"]
        crossed = [c for c in cells if c["eval_nrmse"] < floor_1]
>       assert (len(crossed) > len(cells) / 2) == bool(agg["crosses"]), (
            f"per-cell count says {len(crossed)}/{len(cells)} cells cross "
            f"floor_1={floor_1!r}, aggregate row says crosses={agg['crosses']!r} "
            f"(mean={agg['mean']!r}, dist_to_floor={agg['dist_to_floor']!r})"
        )
E       AssertionError: per-cell count says 8/9 cells cross floor_1=0.7071067811865476, aggregate row says crosses=False (mean=0.7175018067286933, dist_to_floor=0.010395025542145775)
E       assert (8 > (9 / 2)) == False

tests\mars_v20\test_v20_r15_it9_agg_vs_cells.py:29: AssertionError
_______ test_published_cis_on_bounded_quantities_stay_inside_the_bound ________

    def test_published_cis_on_bounded_quantities_stay_inside_the_bound():
        rs = rows()
        probe = [r for r in rs if r.get("t") == "probe"][0]
        bad = {k: v for k, v in probe.items()
               if k.endswith("_ci") and k.startswith(("gate_r2", "sign_acc"))
               and isinstance(v, list) and v[1] > 1.0}
>       assert not bad, f"CI upper end above the quantity's own bound of 1.0: {bad}"
E       AssertionError: CI upper end above the quantity's own bound of 1.0: {'sign_acc_ci': [0.9544868408578528, 1.0179795219893695], 'gate_r2_ci': [0.6370555844629411, 1.1121099749560528]}
E       assert not {'gate_r2_ci': [0.6370555844629411, 1.1121099749560528], 'sign_acc_ci': [0.9544868408578528, 1.0179795219893695]}

tests\mars_v20\test_v20_r15_it9_agg_vs_cells.py:42: AssertionError
=========================== short test summary info ===========================
FAILED tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py::test_aggregate_verdict_agrees_with_per_cell_crossing_count
FAILED tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py::test_published_cis_on_bounded_quantities_stay_inside_the_bound
2 failed, 1 passed in 1.63s
```

---

## 3. SEED 9 — THE SIGN SEPARATOR SURVIVES OUT-OF-SAMPLE; THE MAGNITUDE ENVELOPE DOES NOT

**Test: `tests/mars_v20/test_v20_r15_it9_seed9_mode.py`, RED first, verbatim
output below. It builds the envelope from `results/v17k_r4_retake.jsonl` seeds
2/3/7 — the exact cells `V20_R15_IT7_JUPITER.md:180-198` cites — and asserts
seed 9 falls inside it field by field.**

The signature as the round stated it **before seed 9 existed**
`[READ V20_R15_IT7_JUPITER.md:180-198, V20_R15_JOURNAL.md:917-921]`: `lambda_hat > 0`
against ≈ `−1.45` elsewhere, `a_hat_max` in `[12.77, 116.01]`, `gate_r2` collapsed
into `[0.011, 0.627]`, `eval_nrmse` in `[1.113, 1.152]`.

| field | seeds 2/3/7 envelope | seed 9 | |
|---|---|---|---|
| `lambda_hat` | `[+0.1890, +1.2169]` | `+0.30839410424232483` | **inside** |
| `gate_r2` | `[0.011122, 0.626861]` | `0.050886682051081156` | **inside** |
| `a_hat_max` | `[12.7675, 116.0061]` | `292.2945556640625` | **OUTSIDE — 2.52× the prior max** |
| `eval_nrmse` | `[1.11334, 1.15228]` | `1.281779592990027` | **OUTSIDE — exceeds the prior max by >3× the trio's own spread (0.039)** |

```
E       AssertionError: seed 9 (results/v20_r15_it8_armpl_b.jsonl) vs the seeds-2/3/7
E       envelope (results/v17k_r4_retake.jsonl), field by field:
E           lambda_hat     seed9=0.30839410424232483   envelope=[0.1889575868844986, 1.2168586254119873]  -> inside
E           a_hat_max      seed9=292.2945556640625     envelope=[12.767516136169434, 116.00607299808]     -> OUTSIDE  <-- FAILS envelope containment
E           gate_r2        seed9=0.050886682051081156  envelope=[0.011122143997766787, 0.6268612078957465] -> inside
E           eval_nrmse     seed9=1.281779592990027     envelope=[1.1133392329955414, 1.1522795055459243]  -> OUTSIDE  <-- FAILS envelope containment
E         fields outside envelope: ['a_hat_max', 'eval_nrmse']
FAILED tests/mars_v20/test_v20_r15_it9_seed9_mode.py::test_seed9_falls_inside_the_seeds_2_3_7_m2_envelope_field_by_field
1 failed in 0.74s
```

### 3.1 The answer is a split, and the split is the useful part

**The sign separator is confirmed out-of-sample and this is the round's best
piece of news.** Seed 9 arrived unselected, and the *sign* prediction — the one
coordinate L-6 filed at it.7 on `n = 8` with margin `0.451211` — held on a cell
nobody chose. The separation over all 16 `arm_pl` cells now has **zero overlap,
margin `0.432760655774893`** `[READ V20_R15_JOURNAL.md:1435-1438]`. That is a
genuine out-of-sample instance of a pre-registerable rule, and it is what makes
§2.4's exclusion rule usable **for it.10**.

**The magnitude story is not confirmed; it was widened to fit.** `[READ
V20_R15_IT8_MERCURY.md:107-123]` states the four diverging cells read
`12.77 / 49.66 / 116.01 / 292.29` — the range is redefined to include the new
point rather than the new point tested against the old range. `[READ
V20_R15_JOURNAL.md:1435-1436]` does the same, calling seed 9 "a **fourth**
`a_hat_max` blow-up". **Folding an observation into a range and confirming a range
are different operations, and only one of them is a test.**

The theorem does not forbid the larger value — M2 is stated as *"open-range:
infimum unattained, `|w| → ∞`"* `[READ CEQ_V20_R15_CONTRACT.md:181-185]`, so
unboundedness is the claim. **That is exactly why `a_hat_max` cannot be a
confirming measurement of M2: an unbounded prediction is not failable by a large
value.** The only failable half of M2 as measured here is the sign, and the sign
is the half that held.

### 3.2 Consequence for §2, stated as a bound

`frac_gate_annihilated = 0.0` on **all 16** `arm_pl` cells, divergent and
convergent alike, so it discriminates nothing and cannot be argued either way.
The exclusion rule therefore has exactly one admissible coordinate: **the sign of
`lambda_hat`**. Registered before the next run it makes the eight-cell crossing
(`ci_hi = 0.6639380105668372`, clearing `floor_1` by `0.0432`) a legitimate
result. **It was not registered before it.8, so `results/v20_r15_it8_armpl_b.jsonl`
still supports only its own `crosses: false`** — and a rule cannot be
pre-registered retroactively by the run that motivated it.

---

## 4. THE `softmax 0/8` CONTROL IS NOT ON THE FRESH SEEDS

**Test: fourth case in `tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py`, RED.**

`[RUN] grep -ln '"kind": "softmax"' results/*.jsonl` →
`foreman_signfloor.jsonl`, `v15_r1.jsonl`, `v17k_r4_floor.jsonl`,
`v17k_r4_retake.jsonl`. **Not `v20_r15_it8_armpl_b.jsonl`**, whose header reads
`"arms": ["arm_pl"]`.

`[RUN]` every `softmax` cell in `results/`:

| file | seeds | device | steps | `eval_nrmse` |
|---|---|---|---|---|
| `v15_r1.jsonl` | 0–7 | **cpu** | 150 | 0.9714 … 0.9455 |
| `v17k_r4_floor.jsonl` | 0, 1 | cuda | 150 | 0.9734, 0.9388 |
| `v17k_r4_retake.jsonl` | 0–7 | cuda | 150 | 0.9734 … 0.9443 |

**`softmax` has never been run on seeds 8–15.** The headline pairs `arm_pl` on
seeds 8–15 against `softmax` on seeds 0–7, from a different process. The RED
assertion reads: *"softmax cells exist only for seeds [0..7]; the fresh seeds
[8, 9, 10, 11, 12, 13, 14, 15] have no softmax control anywhere in `results/`."*

**This is the weakest of the three strikes and it is honest to say so.** `softmax`
spans `0.9388–0.9734` over eight seeds — sd well under `0.02` — so no plausible
draw at seeds 8–15 crosses `floor_1 = 0.7071`. The defect is in what the round is
**entitled to write**, not in which arm wins: *"7 of 8 fresh against 0 of 8"*
reads as a paired contrast and is not one. **Replacement route: 8 `softmax` cells
at `1.733 s/run` = ~14 GPU-s**, which folds into the same sub-20-second rerun §2.5
already prices.

---

## 5. STRIKE 10 (the pinned eval seed) — **THE DIFFERENTIAL ATTACK DID NOT FIRE. THE INDEPENDENCE ATTACK DID.**

**Test: `tests/mars_v20/test_v20_r15_it9_eval_pin.py`, 4 cases, verbatim first run
below. The two cases that speak to the round's claim PASSED — that is the
finding, and it is against MARS.**

```
test_predict_the_mean_is_still_exactly_one FAILED
test_pinned_seed_is_not_an_outlier_on_the_nrmse_denominator
std(y_ev) [nrmse denominator]: seed=12345 value=1.429381  theoretical sqrt(T_STAR)=1.414214
  other-draws(n=40) mean=1.414176 sd=0.013375 min=1.388199 max=1.437943  quantile=0.854
PASSED
test_pinned_seed_drive_channel_sign_balance_is_not_an_outlier
mean(a_ev): seed=12345 value=-0.006592  other-draws(n=40) mean=+0.000031 sd=0.009856
  min=-0.017090 max=+0.021484  quantile=0.268  frac_neg(pinned)=0.5033
PASSED
test_score_scale_is_common_mode_across_arms
seed 12345: std(y_ev)=1.429381 vs sqrt(T_STAR)=1.414214  relative shift=+1.0725%
PASSED
1 failed, 3 passed in 5.47s
```

The one failure is a `1e-9`-tolerance sanity gate tripped by float32 rounding at
`~1e-7` (`predict-the-mean` NRMSE reading `1.0000000839` instead of `1.0`); it is
not a statement about seed 12345 and is not claimed as one.

### 5.1 Why the differential attack cannot fire, structurally

`[READ scripts/v15_r1.py:699-700, :706-707]` the eval draw is taken **once, above**
the `for kind in a.arms: for seed in a.seeds:` loop. `[READ
scale/negation_scope.py:1382-1387]` `nrmse(pred, y) = RMSE(pred, y) / std(y,
unbiased=False)`. **So every `arm_pl` row and every `softmax` row divides by the
identical scalar from the identical tensor.** A draw-level scale shift is
**common-mode** and subtracts out of the between-arm contrast by construction.
`arm_pl` 12/16 against `softmax` 0/8 cannot be an artifact of *which* draw was
pinned. `[DERIVED]`

And the pinned draw is not extreme: `std(y_ev)` at quantile **0.854** of 40
same-shape draws, sign balance at quantile **0.268**, both inside `[0.025, 0.975]`.
**I filed STRIKE 10 expecting a differential and there is none. It is withdrawn as
a differential claim.** That is the third strike I have withdrawn this round.

### 5.2 What the pinning does destroy, and this half stands

`[DERIVED]` **The effective independent sample size of the eval draw is 1.** The
aggregate CI at `:899-909` is computed over `eval_nrmse` across *training* seeds;
the eval draw is constant down that column, contributes **zero** within-run
variance, and is therefore invisible to the interval. A variance component needs
≥ 2 clusters, and there is one. **"12 of 16 crossings" is sixteen correlated
readings of one eval condition, not sixteen Bernoulli trials**, so no binomial
confidence statement over the crossing *rate* is licensed by this design. That is
the half of STRIKE 10 that survives, and it is a bound on what the round may
write, not on which arm wins.

### 5.3 Replacement route — under one GPU-second `[DERIVED, READ scripts/v15_r1.py:879]`

Trained models are **kept** in `models[(kind, seed)]`. The second eval draw
therefore needs **no retraining**: one extra `batch_fn` call (measured `0.039 s`)
and one `torch.no_grad()` forward per kept model — order `1/150` of a training
cell each, **well under 1 GPU-second for all 24 kept models**. Retraining instead
would cost `~27–30 GPU-s` at `1.733 s/cell`.

**A round that has left a one-GPU-second measurement unspent for two iterations
is not GPU-bound.**

---

## 6. THE CLAMP — **THE PRE-CLAMP QUESTION IS SETTLED FOR THE SIX 0-STEP CELLS, AND `u` STRICTLY EXCEEDS 1.0 ON ALL SIX**

**Test: `tests/mars_v20/test_v20_r15_it9_preclamp_u.py`, 8 nodes, `8 passed in 2.25s`,
CPU only. This one was GREEN on its first run and is reported as a settlement,
not a strike — the RED-first rule binds strikes, and calling a measurement a
strike because it happened to be cheap is the kind of promotion this office
audits other offices for.**

MERCURY's measurement stands untouched. The attack was on the inference, and the
gap MERCURY correctly left open — *does pre-clamp `u` exceed `1.0` or land on it?*
— **is closed for the half of the population that is closed-form in the seed.**

### 6.1 Why it is recoverable at 0 steps, and only there

`[READ ceq/arm_smprime.py:113]` `magnitude(u) = torch.clamp(u, 0.0, 1.0)`.
`[READ :121-129]` `blend` feeds `magnitude()` the value
`torch.lerp(ones_like(u), u, g)`, which is the **identity when `g == 1.0`** — and
`g = 1.0` is the construction-time default `[READ :529]`, exact at every 0-step
cell. `[READ :546-547]` `heads(x)` returns the **raw, pre-clamp** value, and the
runner discards it in-line `[READ scripts/v15_r1.py:363]`. So pre-clamp `u` at
step 0 is a closed-form function of the cell seed and the pinned eval batch, and
the batch generator is a **device-independent** `torch.Generator(device="cpu")`.

`[RUN]` Reconstructed on CPU — `torch.manual_seed(seed); ArmSMPrime(S,
d_model=16)`, no training step:

| seed | published `a_hat_max_0step` | reconstructed pre-clamp `u` at argmax | margin above 1.0 |
|---|---|---|---|
| 2 | 1.0 | 1.0686708688735962 | 0.0687 |
| 3 | 1.0 | 1.0263001918792725 | 0.0263 |
| 4 | 1.0 | 1.0465793609619140 | 0.0466 |
| 7 | 1.0 | 1.1842671632766724 | 0.1843 |
| 12 | 1.0 | 1.0129091739654540 | 0.0129 |
| 14 | 1.0 | 1.0333305597305298 | 0.0333 |

**6 of 6 strictly exceed 1.0.** The margins `0.013–0.184` sit five to six orders of
magnitude above the `~1e-7` float32 rounding floor of a 16-wide dot product, so a
CPU-vs-CUDA arithmetic difference cannot flip any of them. **`a_hat_max == 1.0` is
saturation, not coincidence, and it is now measured rather than inferred.**

### 6.2 What the record genuinely cannot settle, priced

`[RUN]` The 58 journalled keys on an `arm_smprime` cell, plus
`manifest.smp_values` = `{variant, route, beta, qk, g, m_setting, theta_setting,
v_setting, m_max, v_max, n_zero_gates}`. **`m_max` and `a_hat_max` are
`magnitude()`'s OUTPUT** `[READ scripts/v15_r1.py:381-382, :866]`. **No key pins
pre-clamp `u`. Not recoverable from the journal.** `[DERIVED]`

For the **twelve trained** cells at `1.0` the reconstruction fails for a stated
reason, not a shrug: training moves `g` off `1.0` (seed 0 trained
`manifest.smp_values.g = 1.319505214691162`), `blend`'s `lerp` stops being the
identity, and pre-clamp `u` needs the trained weights. That is a rerun.

**Price. `[DERIVED]`** The minimal diff is two lines in `gate_columns()`'s
`arm_smprime` branch after `scripts/v15_r1.py:364` — bind `u_raw =
model.heads(x)[0][:, live]`, `lerp` it, publish `col["u_hat_max"]`. The existing
`_0step` loop at `:808` then produces `u_hat_max_0step` **for free**, at **0
incremental GPU-s on any future run.** Backfilling the two frozen journals is a
different matter: the code change moves `instrument_hash`, so it is a full rerun —
`[RUN]` sum of `secs` over the 16 distinct `arm_smprime` cells = **275.551 GPU-s**
(mean 17.22, range 14.852–22.069); the twelve trained-at-`1.0` cells alone =
**209.144 GPU-s**.

**The asymmetry is the finding: forward the field costs nothing, backward it costs
275 GPU-s.** That is L-2's mechanism again — *no theorem repairs a record that
never wrote the field* — arriving a second time in the same round, on a different
column, and it is now the strongest argument in the round for adding fields
speculatively rather than on demand.

---

## 7. THE ATTACKS THAT DID NOT FIRE

1. **STRIKE 10 as a differential advantage to `arm_pl` — dead, §5.1.** The pinned
   draw is common-mode across arms by construction and is not an extreme draw
   (quantiles `0.854` and `0.268` of 40). Withdrawn.
2. **`frac_gate_annihilated` as an exclusion coordinate for seed 9 — dead.** It
   reads `0.0` on **all 16** `arm_pl` cells, divergent and convergent alike. It
   discriminates nothing and cannot support an argument in either direction.
3. **"The pinned eval batch makes the `arm_pl` census constant" — does not apply
   here.** The `4123/8192` constant is an `arm_smprime` phenomenon; every
   `arm_pl` cell in `results/` reads `frac_gate_annihilated = 0.0`. L-M4's TERMINAL
   half is correctly scoped to `arm_smprime` and is not transferable to this
   claim.
4. **"Seed 9 contradicts M2" — dead, §3.1.** M2 is stated as *open-range,
   infimum unattained, `|w| → ∞`*. An unbounded prediction cannot be failed by a
   large value. The attack reverses into a bound on M2: **`a_hat_max` is not a
   confirming measurement of M2 in either direction**, and only the sign is.
5. **"The it.8 control is not real" — dead.** Seed 0 reproduces
   `results/v17k_r4_retake.jsonl` on `eval_nrmse` to the bit
   (`0.6446726192039927` / `0.64467`), consistent with the it.7 bitwise control.
6. **"MERCURY's clamp measurement is wrong" — never attempted, and correctly so.**
   §6 attacks the inference and leaves the measurement standing; it turned out to
   be recoverable and MERCURY's reading was right.

## 8. SEARCH PROOF — positive control first, same invocation both times

```
[RUN] grep -n "seed=12345" scripts/v15_r1.py
699:    x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345,
703:    print(f"  eval n={a.n_eval} seed=12345; per-seed train n={a.n_train} at the seed itself")

[RUN] grep -n "crosses" scripts/v15_r1.py | head -5
909:                         crosses=bool(m + half < floor1),
924:              f"CROSSES={agg[kind]['crosses']} power={agg[kind]['achieved_power']:.4f}")
```

The control finds the line the Inspector and MERCURY both cite. **The same
invocation then finds the verdict rule at `:909`, which is what §2 turns on.**

The write-back proof, which is the whole of strike F2 — the grep that was not run
at it.7, run now, on the file that was not written:

```
[RUN] grep -c "L-M" V20_R15_LEAP_LEDGER.md
6

[RUN] grep -rn "L-M1" --include=*.md . | grep LEDGER
V20_R15_LEAP_LEDGER.md:85:| **L-M1** | **VENUS §2.2 body** ...
```

At it.7 the second command returned nothing and the report said otherwise.

`[RUN] git status --porcelain` shows `house-events.jsonl` modified and the round's
`.md` files untracked, as before. **No git write of any kind was performed.
Nothing touched Kaggle.**

## 9. LIMITS

The n=8 recomputation in §2.3 removes seed 9 **post hoc**; it is offered as the
magnitude of a hinge, not as a verdict, and §3.2 states the only condition under
which it becomes one. The eval-draw quantiles in §5 are two coarse statistics over
40 draws; they rule out gross idiosyncrasy in the score's normaliser and the
sign balance, not finer structure in the realised coefficients at the two live
positions — only a second eval seed settles that, and it costs under one
GPU-second. The seeds-2/3/7 envelope in §3 is an empirical range from **n = 3**;
"outside the envelope" is a statement about that range, not about M2, whose
asymptotic claim a larger `a_hat_max` does not contradict. The pre-clamp
reconstruction in §6 covers the six 0-step cells only, was performed on CPU
against a CUDA-produced journal (the margins make that safe, the shared RNG makes
it valid, neither makes it a rerun), and says nothing about the twelve trained
cells. The `softmax` finding in §4 is about entitlement, not outcome; on the
observed spread no plausible draw at seeds 8–15 crosses `floor_1`. Every GPU-second
price quoted is `secs` from a journalled `t="wall"` or `t="cell"` row on this
machine's `cuda` device, and wall-clock `secs` in this instrument is
**un-synchronised** (L-9), so all prices are order-of-magnitude, not budgets.

## 10. TEST TALLY `[RUN]`

```
[RUN] python -m pytest tests/mars_v20/test_v20_r15_it9_{agg_vs_cells,seed9_mode,eval_pin,preclamp_u}.py -q
FAILED test_v20_r15_it9_agg_vs_cells.py::test_aggregate_verdict_agrees_with_per_cell_crossing_count
FAILED test_v20_r15_it9_agg_vs_cells.py::test_published_cis_on_bounded_quantities_stay_inside_the_bound
FAILED test_v20_r15_it9_agg_vs_cells.py::test_the_fresh_seeds_have_a_softmax_control_on_the_same_seeds
FAILED test_v20_r15_it9_seed9_mode.py::test_seed9_falls_inside_the_seeds_2_3_7_m2_envelope_field_by_field
FAILED test_v20_r15_it9_eval_pin.py::test_predict_the_mean_is_still_exactly_one
5 failed, 12 passed in 3.63s
```

**17 nodes. Four RED are the four strikes, each RED on its first run and not
weakened afterwards. One RED is the float32 tolerance artifact disclaimed in
§5. Twelve GREEN, of which three are controls that had to pass for the REDs to
mean anything: the `agg` row reconstructed bitwise (§2.3), the eval-draw
quantile band (§5), and the six pre-clamp reconstructions (§6).**
