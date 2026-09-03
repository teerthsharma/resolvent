# V20 R15 — it.10 — VENUS (IRENE)

Branch `v17k-gate0`. Wall clock honoured. **No git writes. Nothing touched Kaggle.**
Tests: `tests/venus/test_v20_r15_it10_venus.py` — **9 passed**, first run RED, verbatim in §5.

---

## 0. THE RECORD GREW UNDER THIS OFFICE MID-ITERATION, AND THE TEST FILE IS WHAT NOTICED

`results/v20_r15_it10_mercury_rescore.jsonl` was written at **`03:25:18`**, after this
iteration's test file was written and passing. The re-run went **RED on a test that had
passed four minutes earlier** — `test_softmax_has_never_been_run_past_seed_7`, MARS's
strike 3, falsified by MERCURY while the report arguing about it was open.

**That is the correct way to find out**, and it is the argument for writing the objection
as a test rather than as a paragraph. The paragraph would have shipped.

---

## 1. TASK A — THE STRUCK ROW IS RE-FILED AS `L-V1`

**Appended at `V20_R15_LEAP_LEDGER.md:164-197`, read back from the file**
(`[RUN] grep -c "L-V1" V20_R15_LEAP_LEDGER.md` → `4`; `wc -l` → `197`).

The struck row at `:49-60` is **left in place unedited** so the Inspector's line
references still resolve; `L-V1` supersedes it and says so in its own heading.

**The strike is accepted without qualification.** The field column held a quantity in this
round's vocabulary, then the missing theorem spelled out, then a conjecture — three
failures in one cell, and the reading of `:58` that confirms it is correct.

**The mechanism, named, because a repair without one repeats:** the row was written by
asking *what do I want proved* and labelling the answer a field. A field is named by
asking *whose textbook has this in its index.* That question was never asked.
**The repair is a construction order, not a vocabulary swap.**

| | struck row `:55-56` | `L-V1` |
|---|---|---|
| head noun | "the extent of corner descent in multiplicative-gate landscapes" — a quantity | **non-convex optimization theory — the implicit-bias / saddle-to-saddle branch for multiplicative (deep-linear / gated) parameterizations** — a discipline (Gunasekar et al.; Jacot et al.; Li–Ma–Zhang) |
| the want | fused into the head noun | stated after the field and marked as a want: *where gradient flow on a product parameterization comes to rest on an annihilating face*, as a function of depth and initialization scale |
| the lattice conjecture | in the field column | **moved out entirely** — it is the TERMINAL condition, where it was always a falsifier |

The it.7 idea survives; only the row's construction failed, which is what the brief said.

---

## 2. TASK B — THE RANKING, RE-FILED. THE UPDATE IS **UP**, AND TWO OF THE THREE OBJECTIONS WERE DISCHARGED BY MEASUREMENT INSIDE THIS ITERATION

### 2.1 Objection 1 — `n_eff = 1`. **Discharged on the indicator the round scores.** `[RUN]`

MERCURY re-scored **18 cells × 3 eval draws = 54 rows** (`eval_seeds [12345, 12346,
20260902]`, same `instrument_hash 5d41a63d…9a309`).

**The crossing indicator is invariant across all three draws on 18 of 18 cells.**

| eval draw | `arm_pl` below `floor_1` | `softmax` below `floor_1` |
|---|---|---|
| `12345` (the pinned one) | **8 / 9** | **0 / 9** |
| `12346` | **8 / 9** | **0 / 9** |
| `20260902` | **8 / 9** | **0 / 9** |

Control: `arm_pl` seed 9 at draw `12345` reproduces the banked it.8 value **bitwise**
(`1.281779592990027`), so the rescore is reading the same models.

**MARS's strike was correct at filing and is now discharged.** `n_eff = 1` did not make the
binomial vacuous — **it made it conditional.** The 16 cells vary in *training* seed; they
were scored on one *eval* draw. The unlicensed step was marginalising over draws. That
marginal is now measured at `n = 3` and it is **flat on the indicator**.

### 2.2 The part the rescore does **not** discharge, and it is mine to say `[DERIVED, RUN]`

`n = 3` is not `n = ∞`, and the stability is not free. Ranking `arm_pl` cells by
**draw-spread ÷ distance-to-floor**:

```
seed 15 ratio=0.999 margin=0.020233 spread=0.020220   <- one spread from flipping
seed 12 ratio=0.589   seed 9 ratio=0.493   seed 14 ratio=0.266
seed 10 0.180  seed 13 0.180  seed 0 0.165  seed 8 0.158  seed 11 0.105
```

**Pre-registered, dated 2026-09-02, complement form:** *on a fourth eval draw, if any
`arm_pl` cell in `{0, 8..15}` changes its crossing verdict, it is **seed 15**; if two
change, they are **15 and 12**.* **Falsifier: any cell outside `{15, 12}` flipping first,
or seed 15 flipping in the direction that adds a crossing** (its three readings are all
below the floor, so it can only flip out). This ranges over the whole outcome space
including "nothing flips", which confirms it weakly and correctly.

**The scale channel is bounded analytically and separately.** The pinned draw's own
denominator error is `+1.0725 %` (`std(y_ev) = 1.429381` vs `sqrt(T*) = 1.414214`, MARS
it.9 §5). Rescaling **every** banked cell by it moves **no cell across the floor** in
either arm — the smallest relative margin is `0.037517`, a factor **3.50** of headroom.
So the one-draw dependence splits into a **scale** channel (bounded, checked, cannot flip
a cell) and a **sample** channel (which sequences — that is what MERCURY measured).

### 2.3 Objection 3 — the unpaired contrast. **Repaired, and it was not weak** `[RUN]`

`softmax` is now run on seeds `{0, 8..15}`. **`arm_pl` 8/9 against `softmax` 0/9, on every
draw. Fisher exact one-sided `p = 2.0568e-04`.**

**Whether it changed my ranking — asked and answered rather than waved off.** On the
banked record alone, the honest paired comparison was seeds 0–7: **`arm_pl` 5/8 vs
`softmax` 0/8, `p = 0.012821`**, against the unpaired headline's `p = 6.7304e-04`.
**A factor of 19 in `p`.** MARS graded his own strike the weakest of four; on the record
as it stood **it was worth an order of magnitude**, and calling it weak was the one place
his it.9 undersold itself. It stopped mattering only because someone ran the cells.

### 2.4 Objection 2 — the exclusion decision. **STANDS, untouched by the rescore** `[RUN]`

```
n9 (as registered): n=9 mean=0.7175018067286933 sd=0.21245444406217273 ci_hi=0.8808087489119841 crosses=False
n8 (seed 9 removed): n=8                        sd=0.02029964404207494 ci_hi=0.6639380105668372 crosses=True
```

The `n=9` line reproduces the journalled aggregate bitwise (GREEN control). **The headline
still turns on one cell**, and no number of eval draws changes that — it is a **scoring
rule**, which is JUPITER's `L-15`, not a measurement.

**What the rescore does change is the legitimacy of the cut.** Seed 9's exclusion is
stated in coordinates readable **before** `eval_nrmse` (`lambda_hat > 0`, `gate_r2 0.0509`,
`a_hat_max 292.29`), and `sign(lambda_hat)` is now **draw-stable on 18 of 18 cells across
3 draws**. `L-15`'s mixing variable is therefore not merely journalled — it is measured to
be draw-invariant. That converts `L-15` from *a rule change one would like* into *a rule
change whose declared mixing variable has passed an out-of-sample stability check.*

### 2.5 What this office will and will not quote as a rate

**Will quote**, with the conditioning named: **`arm_pl` 12/16 below `floor_1` on the
banked record**, every one clearing with its whole bootstrap interval
(`max boot_hi | below = 0.705184`, JUPITER it.9); **8/9 vs 0/9 paired, on three draws.**

**Will not quote**: any Clopper–Pearson verdict against clause (1). Not because of
`n_eff` — that objection is discharged — but because **`12/16` straddles the bar under the
convention the contract never states** (`CP-lower` two-sided `0.4762`, one-sided `0.5156`)
and **⟨CLAUSE_1_TAIL⟩ is unruled.** A number whose sign is set by a word the author has not
yet chosen is not a finding. **Suspended, with the reason stated, until the ruling lands.**

### 2.6 The re-filed ranking

> **1. W3 `arm_pl` — 2. W1 `arm_smprime`**

| ordering at it.29 | it.5 | it.7 | **it.10** | Δ |
|---|---|---|---|---|
| **W3 ≻ W1** | 0.72 | 0.78 | **0.84** | **+0.06** |
| **W1 ≻ W3** | 0.20 | 0.14 | **0.09** | **−0.05** |
| no ordering — both eliminated on clause (1) | 0.08 | 0.08 | **0.07** | **−0.01** |

**The sign is up, and the size is small for a stated reason.** What moved it is not the
crossing count — that was already priced at it.7 and this office's own pre-registration
anticipated it. What moved it is that **the comparison became paired and draw-stable in one
iteration**, which removes the two largest instrument objections on the board
simultaneously. That is an *evidence-quality* update, and evidence quality is exactly what
it.7 §3.4 said the margin was concealing.

**Why not more.** The W1 mass never rested on rate (it.7 §3.3) — it rests on **clause (2),
BED-K(a), still unrun**, where seed 2 is the only cell carrying the statistic that bed
reads. Nothing this iteration touched it, so **W1 keeps a floor no rate datum can take**.
And the "both eliminated" branch falls only `0.01`, because it is now governed by an
**unruled convention** and a **scoring rule** — neither of which is settled by more data.

**Falsifier, complement form over the full outcome space, dated 2026-09-02:**
**W3 ≻ W1 at `0.84` is wrong if any of —**

1. eight fresh `arm_pl` seeds `16–23` land the pooled count **below `18/24`** below `floor_1`; **or**
2. **any** eval draw returns an `arm_pl` crossing count other than `8/9` on the rescored cells; **or**
3. `arm_smprime` returns **≥ 3** crossings on any eight fresh seeds; **or**
4. BED-K(a) runs and W1's seed-2 statistic separates the arms in W1's favour.

Each clause is a set complement over an observable that exists on every cell, which is the
`M-7` repair made at it.7 and held at it.8.

**Stated once, not re-litigated:** `⟨AUTHOR_COUNTER_RANKING⟩` is **NOT MEASURED**, so under
**D-CALIB-2** this ranking is formally unreadable and it.29 cannot open a VENUS row.
**D-CALIB-3 authorises sign, never size** — the `+0.06` above is a sign statement with a
magnitude attached for auditability, not a calibrated quantity.

---

## 3. TASK C — THE FORECASTER LEDGER, SIGNED, INCLUDING THIS OFFICE

| # | office | prediction / call | filed | outcome | could the killer fire? | grade |
|---|---|---|---|---|---|---|
| 1 | **VENUS** | W1 crossings `0 of 8`, PI `0–3` | it.5 | **0 observed** | yes | **HELD** |
| 2 | **VENUS** | mechanism: `frac ∈ {0.4967, 0.5033}` | it.5 | **FALSIFIED** at seeds 11, 13 | **NO — blind** (`M-7` HOLE) | **FAILED, and the falsifier failed worse** |
| 3 | **VENUS** | *"a bar of the form `> 0.5` against an unnamed interval"* | it.5 | now **decisive** (⟨CLAUSE_1_TAIL⟩) | n/a — structural | **HELD, called before it could advantage anyone** |
| 4 | **VENUS** | `5 of 8` crossings, PI `2–7` | it.7 | **7 of 8**, top edge, inside | yes | **HELD** |
| 5 | **VENUS** | `frac == 0.0` on `8 of 8` | it.7 | **9 of 9**, and now **54/54 readings across 3 draws** | **yes, and it did not** | **HELD — strongest form** |
| 6 | **MARS** | `_0step < 0.15` ⇒ crossing | it.5 | **withdrawn**, `n=1` support, `0` positives on eight fresh | — | **FAILED, self-withdrawn and self-graded TERMINAL (`L-M2`)** |
| 7 | **MARS** | STRIKE 10, *differential* half | it.9 | **his own test PASSED against him**, reported verbatim | yes | **CORRECT CONDUCT — a filed attack that failed and was published as failing** |
| 8 | **MARS** | STRIKE 10, *independence* half (`n_eff = 1`) | it.9 | correct at filing; **discharged at it.10** by 3 draws | yes | **HELD then RESOLVED — the highest-value objection of the round** |
| 9 | **JUPITER** | `sign(lambda_hat)` separates the floor | it.8 | **16/16** out of sample; **draw-stable 18/18 × 3** | yes | **HELD, out of sample twice** |
| 10 | **JUPITER** | five-cell CI, self-flagged post-hoc | it.8 | **7 of 8 new cells on the predicted side** | yes | **HELD, and the post-hoc flag was its author's** |
| 11 | **MERCURY** | refused to invent the cap parameter | it.8 | not a prediction — a refusal | — | **CORRECT (`M-2` from the executor's side)** |
| 12 | **MERCURY** | the it.10 rescore | it.10 | **discharged objections 1 and 3 in one run** | — | **the iteration's only instrument improvement** |

### 3.1 The row nobody filed: what `n_eff = 1` did to every prediction of this round

**A prediction confirmed on one eval draw is confirmed on one draw.** The damage is not
uniform, and it sorts into three classes:

**(a) Predictions on quantities read off the eval draw.** Rows 1, 4, 5 and every crossing
count. These were **conditional on draw `12345` and asserted as marginal.** Rows 4 and 5
belong to this office and were asserted that way. The rescore now measures the marginal for
row 5 — `frac_gate_annihilated == 0.0` exactly, on **18 cells × 3 draws = 54 readings** —
so the strongest-form confirmation is **draw-robust as well as killer-exposed.** That is
luck graded after the fact, not foresight: **nothing in the it.7 filing named the draw
dimension, and it should have.**

**(b) Rate statements over training seeds.** Licensed **conditional** all along; the
conditioning is now checked rather than assumed. The correction to the round's prose is one
clause — every `12/16` in this round means *12 of 16 training seeds, at a fixed eval draw*,
and until it.10 nobody had shown that clause was inert.

**(c) Structural calls.** Rows 3, 9, 11 are draw-independent by construction or now shown
so (`sign(lambda_hat)` stable 18/18 × 3). **These were never in danger, and the round
should notice that its three most durable claims are the ones that never touched the draw.**

**What is still untested.** `arm_smprime` was **not** in the rescore. MARS's `L-M4` — three
of five `frac_gate_annihilated` values being one republished property of the pinned draw
(`4123/8192` on 16 cells) — **remains untested on fresh draws, and it is a W1 finding.**
The arm whose crossing story is now draw-verified is the one that was already winning.
**The asymmetry complained of at it.7 §3.4 has inverted, not closed.**

---

## 4. WHAT it.11 OWES, PRICED

1. **⟨CLAUSE_1_TAIL⟩** — the author's ruling. Still the only thing standing between the
   round and a scored clause (1). **0 GPU-s.**
2. **`arm_smprime` in the rescore** — 16 cells × 2 fresh draws, re-scoring kept models,
   no training. Settles `L-M4` and closes the inverted asymmetry. **Cheapest open item.**
3. **A fourth eval draw**, which is what the §2.2 pre-registration is stated against.
4. **`L-15`'s rule change** — component-wise CIs under `sign(lambda_hat)`. Its mixing
   variable has now passed a draw-stability check; the obstruction is a decision.

---

## 5. TESTS — RED FIRST, VERBATIM

`tests/venus/test_v20_r15_it10_venus.py`. First run, before any expectation was corrected:

```
FAILED tests/venus/test_v20_r15_it10_venus.py::test_eval_draw_is_one_pinned_batch_so_n_eff_is_one
E       AssertionError: expected one pinned eval draw, got lines [698, 702]
E       assert 2 == 1
1 failed, 5 passed in 1.83s
```

The RED was real and the fix was a finding: `seed=12345` appears **twice** — the draw at
`scripts/v15_r1.py:699` and a **banner echo at `:703`**. The test now asserts both, so the
pin cannot be moved without breaking it.

Second RED, unforced, four minutes later — `test_softmax_has_never_been_run_past_seed_7`
went from PASS to FAIL when MERCURY's file landed. That test is retained under a dated
name (`..._in_the_banked_record`) because the claim was true of the banked record and the
record is what MARS graded.

Final:

```
eval draw pinned at v15_r1.py:699; cell loop opens at :707
paired seeds 0-7: arm_pl 5/8 below floor_1, softmax 0/8
n=9 ci_hi=0.8808087489119841 crosses=False ; n=8 ci_hi=0.6639380105668372 crosses=True
scale shift=0.010725  min relative margin=0.037517  ratio=3.50
fragility ranking (spread/margin): [(15, 0.999), (12, 0.589), (9, 0.493), (14, 0.266), (10, 0.18), (13, 0.18), (0, 0.165), (8, 0.158), (11, 0.105)]
9 passed in 1.68s
```

**Evidence classes.** `[READ]` `scripts/v15_r1.py:699,:703,:707`, `:899-909`;
`V20_R15_LEAP_LEDGER.md:49-60,:164-197`; `V20_R15_IT567_INSPECTOR.md:460-484`;
`results/v20_r15_it10_mercury_rescore.jsonl`. `[RUN]` the nine tests above. `[DERIVED]`
the Fisher and Clopper–Pearson figures, the rescale bound, the fragility ratios.
**No `GUESS` is asserted anywhere in this report.**

---

## 6. INCIDENTAL, FOUND WHILE LOGGING — `house-events.jsonl` IS NOT PARSEABLE AS JSONL `[RUN]`

Four lines of the event log are **not valid JSON** — unescaped backslashes inside prose
`text` fields (`\s`, `\log`): lines **1899** (WILSON), **2937**, **2938** (FOREMAN),
**5871** (wilson). Every one predates this round.

Consequence: **any consumer that parses the log strictly fails on the whole file**, which
includes any future audit of who filed what and when. `json.loads` per line is how this
was found, and it is how it.35 will read the log.

**Priced at 0 GPU-s and not taken here** — VENUS does not rewrite other offices' events.
Filed so the repair is someone's rather than nobody's.
