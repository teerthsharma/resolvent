# V20 R15 — it.5 — VENUS (IRENE)

**The ranking, pre-registered 2026-09-02, on branch `v17k-gate0` at HEAD `207e7b9`.**
Contract slot: `CEQ_V20_R15_CONTRACT.md:91` — *"it.5 VENUS's ranking; the author's
counter-ranking (the point estimate, D-CALIB); MERCURY prices the arena."*
MERCURY priced at it.3. The ranking is this office's and is overdue by four
iterations.

Node: `tests/venus/test_v20_r15_it5_ranking_record.py`, **15 passed**. Every
number below that is a claim about the record is bound there. The planted
negative is at §0.4.

---

## 0. VERIFICATION — the briefed table, re-measured before it is ranked on

`[RUN]` `results/v17k_r4_retake.jsonl`, 24 `t="cell"` records, header
`floor_1 = 0.7071067811865476`, `steps 150`, `seeds 0..7`,
`deterministic_algorithms true`.

| | W1 `arm_smprime` | W3 `arm_pl` | skyline `softmax` | verdict |
|---|---|---|---|---|
| crossings of floor₁ | 1/8 (seed 2) | 5/8 (seeds 0,1,4,5,6) | 0/8 | **CONFIRMED** |
| CP-lower, two-sided 95% | 0.0032 | 0.2449 | 0.0 | **CONFIRMED** |
| CP-lower, one-sided 95% | 0.0064 | 0.2892 | 0.0 | **the briefed row's convention was unstated** |
| best `eval_nrmse` | **0.203920** (seed 2) | 0.633739 (seed 4) | 0.938796 (seed 1) | **CONFIRMED**, ratio **3.108** |
| briefed cost, s/150 | 15.970 | 1.614 | 1.497 | **DIFFERENT RUN — see §0.2** |
| arena cells' own mean `secs` | **16.161** | **1.780** | **1.681** | `[RUN]`, the 24 cells |
| E[cost to a crossing], MERCURY basis | 118.816 | 2.8358 | undefined | **reproduces**, §0.3 |
| E[cost to a crossing], all-cell basis | **129.29** | 2.85 | undefined | `DERIVED` |

### 0.1 The CP convention is load-bearing enough to name

The briefed `0.0032 / 0.2449` are **two-sided** 95% Clopper–Pearson lower limits
(`beta.ppf(0.025, x, n−x+1)`). One-sided 95% reads `0.0064 / 0.2892`. Neither
convention moves either arm within reach of clause (1)'s `0.5`, so **nothing
flips** — but a bar of the form `> 0.5` compared against an unnamed interval is
the shape this round has already been burned by, and it is now named.

**MERCURY's *"Only a clean sweep clears `0.5`"* (`V20_R15_IT3_MERCURY.md:47`) is
CONFIRMED, with the margin he did not print.** `[RUN]` the whole ladder at N=8:

`0:0.0000  1:0.0032  2:0.0319  3:0.0852  4:0.1570  5:0.2449  6:0.3491  7:0.4735  8:0.6306`

**7/8 reads `0.4735` and fails. Only 8/8 clears, at `0.6306`.** The bar is not
approached asymptotically at this N; it is a step function with one step in it.

### 0.2 KILL — the briefed cost row and the briefed crossing row are two runs

`15.970 / 1.614 / 1.497` live at `V17_R4_RETAKE_PRICE.md:194-196`, under the
header *"s / 150 steps, **mean of 2**"* — a six-cell timing run, 3 arms ×
seeds 0,1, `t="wall" = 41.842 s`. They are **not** the 24 arena cells' own
journalled `secs`. Those read `16.161 / 1.780 / 1.681`.

| arm | price basis | arena cells | arena is slower by |
|---|---|---|---|
| `arm_smprime` | 15.970 | 16.161 | +1.2 % |
| `arm_pl` | 1.614 | 1.780 | **+10.3 %** |
| `softmax` | 1.497 | 1.681 | **+12.3 %** |

The table that ranks the wings therefore takes its crossing row from one run and
its cost row from another, and the cheap arm is the one the substitution
flatters most. **Replacement route, and it costs nothing:** criterion (3) is
repriced off the `secs` field the 24 arena cells already journal. No run, no
GPU-seconds, one column. The W1:W3 cost ratio moves `9.89× → 9.08×`; the
direction of every clause is unchanged, which is why this is a bookkeeping kill
and not a re-ranking.

**A second consequence the scoreboard should hear.** SCOREBOARD carries
*"the winner's cost ≤1/10 incumbent with a certificate +4"*. Under **either**
basis W3 costs **more** than the incumbent `softmax` — `1.614/1.497 = 1.078`,
`1.780/1.681 = 1.059`. That `+4` is unreachable by W3 as measured, and W1 is
9× the wrong way. Bound at
`test_no_wing_reaches_the_scoreboards_one_tenth_incumbent_cost`.

### 0.3 MERCURY's 118.8 and 2.85 reproduce, from a basis worth stating

`V20_R15_IT3_MERCURY.md:274,:278-279`. `[RUN]`, exactly:

- W1: `14.852 × 8/1 = 118.816` — `14.852` is **seed 2's own `secs`**, the single
  crossing cell.
- W3: `mean(1.884, 1.739, 1.759, 1.745, 1.735) = 1.7724`, `× 8/5 = 2.8358`.

So the statistic is *mean seconds over the crossing cells × n/x*, not mean over
all cells. Defensible, and it is what he says he did. On the all-cell basis W1
reads `129.29` and W3 is unchanged at `2.85`; the advantage moves `41.90× →
45.4×`. **Not load-bearing. Recorded so it is not re-derived a third time.**

### 0.4 The planted negative, verbatim RED

Four expected values mutated to what the round would have to hold instead
(`crossings 1→2`, `alive == [2] → [2,3]`, `0.4735 → 0.5100`, the cost inequality
reversed). `[RUN]` `pytest tests/venus/planted_red.py -q`:

```
E       assert [2] == [2, 3]
E       assert 0.4734903291247935 == 0.51 ± 5.0e-05
E       AssertionError: seed 2 must be the unique live-gate cell
E       assert (1.7799999999999998 < 1.614)
4 failed, 11 passed
```

Mutation reverted; `pytest tests/venus/ -q` → `15 passed`.

---

## 1. IS SEED 2 A WING OR AN ARTIFACT? — **A WING, and the mechanism names itself**

MARS established seed 2 as the unique cell whose gate reaches neither endpoint of
its cap. The gate columns say more than that, and what they say decides the
ranking.

`[RUN]` `arm_smprime`, all eight cells:

| seed | `eval_nrmse` | `frac_gate_annihilated` | `a_hat_min` | `a_hat_max` | `lambda_hat` | `unit_root` |
|---|---|---|---|---|---|---|
| 0 | 0.926082 | 0.5032958984375 | 0.0000 | 1.0000 | **−inf** | True |
| 1 | 0.881247 | 0.5032958984375 | 0.0000 | 1.0000 | **−inf** | True |
| **2** | **0.203920** | **0.0** | **0.3408** | **0.5400** | **−0.7953** | **False** |
| 3 | 0.916258 | 0.4967041015625 | 0.0000 | 0.8454 | **−inf** | False |
| 4 | 0.911926 | 0.5032958984375 | 0.0000 | 1.0000 | **−inf** | True |
| 5 | 0.918157 | 0.5032958984375 | 0.0000 | 1.0000 | **−inf** | True |
| 6 | 0.893239 | 0.5032958984375 | 0.0000 | 1.0000 | **−inf** | True |
| 7 | 0.926717 | 0.5032958984375 | 0.0000 | 1.0000 | **−inf** | True |

**Seed 2 is the only cell in which the path-product gate is alive at every
position.** On the other seven, half the gate mass sits at the annihilating
corner `m = 0`, and `lambda_hat` is `−inf` — the arm has no fitted decay because
the hop has been switched off. Those seven land at `0.88–0.93`, the skyline's
own neighbourhood (`softmax` best `0.9388`). **When the gate dies the arm becomes
the incumbent.**

**Three measured reasons it is a wing and not an artifact:**

1. **It differs by a gate state, not by an eval draw.** `frac_gate_annihilated`
   is a training outcome read off the model; `0.0` versus `≈0.50` is a discrete
   difference in what was learned, not a difference in what was scored.
2. **The instrument carries no noise for a fluke to hide in.** `[RUN]`
   `results/v17k_r4_floor.jsonl` is an independent run and reproduces the
   retake's `arm_smprime` seeds 0 and 1 to every printed digit — `0.926082`,
   `0.881247`. The pipeline is bit-reproducible per seed.
3. **The failure is a structural constant, not a spread.** Eight seeds produce
   exactly **two** non-zero values of `frac_gate_annihilated`
   (`0.5032958984375` six times, `0.4967041015625` once, `0.0` once). A continuum
   of bad fits would not do that. This is two basins.

**And the corner has a theorem.** Annex **M1** (`CEQ_V20_R15_CONTRACT.md`,
β-GRADIENT THEOREM): *"β descends toward the exact corner"*, `[RUN: +20.87 on
path-product data]`. W1's seven failures are the annex's own predicted descent,
observed. That is a landscape result, not an accident.

**The corollary that kills the cheapest next move.** Because the pipeline is
deterministic per seed, **re-running seed 2 returns `0.203920` and proves
nothing.** Only *new* seeds carry information. Anyone proposing to "check seed 2"
is proposing to re-read the same cell. Replacement route: §2.2.

**W3 fails by a different theorem, and this is the ranking's spine.** `[RUN]`
`arm_pl`: `frac_gate_annihilated == 0.0` on **all eight** cells — the prefix-scan
gate never dies. Its three failures are a blow-up:

| seed | `eval_nrmse` | `a_hat_max` | `gate_r2` | `lambda_hat` |
|---|---|---|---|---|
| 2 | 1.152280 | **12.77** | **0.0111** | +0.7584 |
| 3 | 1.113339 | **49.66** | 0.6269 | +0.1890 |
| 7 | 1.148927 | **116.01** | **0.0466** | +1.2169 |
| *(the five crossings)* | 0.634–0.662 | 1.10–1.51 | 0.972–0.990 | −1.38 to −1.47 |

That is annex **M2** (GATE-LANDSCAPE THEOREM): *"open-range `a = 2σ(w)−1`:
infimum unattained, `|w| → ∞`"*, `[RUN: w still growing at 400 steps]`.

**So the two wings fail by two different annex theorems, and only one of the two
is bounded by a cap.** M2's divergence is a runaway a norm cap arrests — the
round already carries the cap concept (`V16_CALIBRATION.md:151-160`). M1's corner
is the direction the gradient *points*; a cap does not undo a descent direction.
**W3's three losses are repairable in the sense a bound is repairable. W1's seven
losses are repairable only by changing where the gradient goes.**

**Verdict: seed 2 is a wing whose basin the optimizer reaches at `p̂ = 0.125`,
with a named reason for the other seven. "Wing" does not buy clause (1). A real
basin at 1/8 against a real basin at 5/8 still loses clause (1), and loses
clause (3) by 41.9×.**

---

## 2. THE RANKING — a forecast, with probabilities

**Scored at it.29 against measurements not yet taken.**

> **1. W3 `arm_pl`**
> **2. W1 `arm_smprime`**

| ordering at it.29 | probability |
|---|---|
| **W3 ≻ W1** | **0.72** |
| **W1 ≻ W3** | **0.20** |
| **no ordering filed — both eliminated on clause (1), `N` → 0** | **0.08** |

**Why W3, in one sentence:** it leads on clause (1) by 76× on the CP-lower and on
clause (3) by 41.9×, and its losing seeds fail by a mechanism a bound arrests
while W1's fail by a mechanism the gradient chooses.

**Why the 0.20 is not smaller, which is the honest half.** Clause (1) is
**unreachable for both arms at N=8** — §0.1, only 8/8 clears. A lexicographic
criterion whose first clause no entrant meets does not decide anything; it gets
re-read, and every re-read descends to clause (2), **which has never been run**
(BED-K(a) is scheduled it.25–27). On clause (2) — delay `d = 20`, Hankel ceiling
`1/d = 0.05`, explicitly *attention-native* — W1's seed-2 cell is **the only cell
in the tournament with a finite `lambda_hat` and `unit_root == False`**, which is
the statistic that bed is built to read. Twenty percent is what an unrun bed with
a 3.1× headline is worth.

### 2.1 What would have to be true for the ordering to flip — as quantities

| route | the quantity | flips me if | measurable |
|---|---|---|---|
| **FLIP-1** clause (1) | W1's crossing rate on fresh seeds | **≥ 4 of 8 new `arm_smprime` seeds cross floor₁** (pooled ≥ 5/16, rejecting `p = 0.125` upward) | **yes — 8 cells, ≈ 129 GPU-s** |
| **FLIP-2** clause (2) | conditional NRMSE on BED-K(a), `d = 20` | **W1 ≤ 0.10 and W3 > 0.20** against the `1/d = 0.05` ceiling | **yes — the bed exists, unrun** |
| **FLIP-3** the criterion | the coordinator's reading of a clause no arm meets | clause (1) ruled a **tie** rather than a double elimination (`P ≈ 0.55`) | **no — a ruling, not a measurement** |

FLIP-3 is the precondition for FLIP-2 mattering, and it is not mine to measure.
It is stated so that when it happens it is not mistaken for evidence.

### 2.2 THE CHEAPEST KILLER OF MY OWN RANKING

Not the crossing count — the **mechanism**, because the mechanism is the whole
content of the ranking.

**Run `arm_smprime` on eight fresh seeds 8–15, same BED-M cell, matched params.
≈ 129 GPU-s — about 2.2 minutes on this box.** `DERIVED`: `8 × 16.161`.

**Pre-registered joint prediction, 2026-09-02, filed before the run:**

> Every new seed that **fails** to cross carries
> `frac_gate_annihilated ∈ {0.4967041015625, 0.5032958984375}` and
> `lambda_hat == −inf`.
> Every new seed that **crosses** carries `frac_gate_annihilated == 0.0` and a
> **finite** `lambda_hat`.
> Expected crossings: **1 of 8**; 95 % predictive interval **0–3**.

**One new seed that crosses with an annihilated gate — or fails with a live gate
— falsifies me outright**, because it would show W1's failure is not the corner
M1 predicts, and the whole basis for preferring W3's bounded divergence
collapses. **≥ 4 of 8 crossings independently refutes the rate.** This is a
sharper falsifier than a count, it is filed before the run, and it costs
**1/10 of MERCURY's EXIT A at `N = 65` (1,243.8 GPU-s,
`V20_R15_JOURNAL.md:429`).**

**Stated plainly: if that run comes back 4/8, the ranking above is wrong and this
office wants it recorded as wrong, not reinterpreted.**

---

## 3. TASK B — THE AUTHOR'S COUNTER-RANKING

```
⟨AUTHOR_COUNTER_RANKING⟩ NOT MEASURED
```

| field | value |
|---|---|
| **what is owed** | an ordering over {W1, W3}, **plus a point probability on that ordering** |
| **why a point estimate** | **D-CALIB-1**, `V16_CALIBRATION.md:142` — *"the counter is the point estimate"* |
| **who owes it** | the author. He is not present in this loop |
| **who may supply it** | **no one.** Not this office, not any planet, not the coordinator |
| **status** | OPEN |

**A slot is not a prediction. Nothing in this section forecasts what the author
would say.**

**What the calibration column cannot compute without it: all of it.**
**D-CALIB-2**, `V16_CALIBRATION.md:162` — *"a prediction with no counter is not
read at all."* By the round's own rule **the ranking in §2 is unreadable until
the counter lands.** It is dated, falsifiable, pre-registered and formally inert.
That is the rule working, not a complaint: the counter is what makes a forecast
scorable, and a self-scored forecast is exactly what D-CALIB exists to refuse.
**The it.29 forecaster score cannot open a VENUS row until the slot is filled.**

The three-column form the slot must be filled in already exists in the record —
`V16_VENUS_R1PRIME.md:51`:

`| quantity | AUTHOR's PREDICTION | AUTHOR's COUNTER (D-CALIB-1: the point estimate) | VENUS |`

`V15_VENUS_PREDICTIONS.md` EXISTS at HEAD. `[READ]`

### 3.1 What IS derivable without him — the author's measured bias, signed and cited

This is a reading of the record, not an invention.

| quantity | value | source |
|---|---|---|
| round audited | R11 | `V16_CALIBRATION.md:19` |
| statements checked | 9 as filed (`V15_LEDGER.md:649-661`); ≥ 17 with the confirmations that table omits | `V16_CALIBRATION.md:19` |
| adverse verdicts | 9 of 9 filed rows | `V16_CALIBRATION.md:50` |
| **signs** | **7 `+` (optimistic), 1 `−` (pessimistic), 1 unsigned** | `V16_CALIBRATION.md:50` |
| **direction test** | **7/8 optimistic, one-sided sign test `p = 0.0352`** — established at α = 0.05 | `V16_CALIBRATION.md:94-100` |

**SIGN OF THE AUTHOR'S MEASURED BIAS: POSITIVE (OPTIMISTIC), `p = 0.0352`,
n = 8 signed rows, R11.** `CITED`

Two things the same file makes clear, and both bind this office:

- The filed prose *"six wrong, all six optimistic"* is **not what the rows say**:
  honestly signed, those six read `5/6`, `p = 0.1094`, **not significant**
  (`V16_CALIBRATION.md:94-100`). The direction survives on the full nine; the
  round entry's own evidence for it does not.
- **D-CALIB-3**, `V16_CALIBRATION.md:171` — *"sign, never size. No numeric shrink
  factor is authorised."* The only licensed use of the bias is directional: when
  the counter arrives, expect it to sit **above** the measured outcome. **No
  multiplier has been applied to anything in this report, and none may be applied
  to his counter when it lands.**

---

## 4. TASK C — THE FORECASTER LEDGER, OPENED AT it.5

The contract scores forecasters at it.29. A calibration column opened after the
outcome is worthless, so it opens now. **This is a ledger, not a judgement.**
Sign convention as `V16_CALIBRATION.md`: `+` = optimistic, `−` = pessimistic.

| # | forecaster | prediction, as filed | what happened | verdict | sign |
|---|---|---|---|---|---|
| 1 | **coordinator** | **`N=65`** (`41/65`, CP-lower `0.5020`), `V20_R15_JOURNAL.md:278` | self-corrected three ways, `:410-421`: convention-dependent (`floor` gives **`N=72`**); *"a coin flip, not a design — a 51.7 %-chance experiment"*, honestly powered **`N=125`**; the blocker does not exist | **WRONG, self-caught** | **`+`** |
| 2 | **SATURN** | *"Every pair separates by ≥ `0.30`"*, `V20_R15_IT2_SATURN.md:99` | MARS STRIKE 5 upheld; `V20_R15_IT3_SATURN.md:74` — *"does NOT survive. It is withdrawn."* | **WRONG, withdrawn in writing** | **`+`** |
| 3 | **SATURN** | **K5** — *"GPU-seconds-to-floor is not measurable for any wing today"*, `V20_R15_IT1_SATURN.md:180` | `V20_R15_IT2_SATURN.md:127` — *"K5: WITHDRAWN, refuted by its own node"* | **WRONG, self-refuted** | **`−`** |
| 4 | **SATURN** | **K6** — four annex `[RUN]` instances have no producer | halved to two survivors, `d=65` and `1.3e-3`, `V20_R15_IT2_INSPECTOR.md:283,:349` | **HALF RIGHT** (2 of 4) | `+` on the two that fell |
| 5 | **MARS** | **STRIKE 5** — A.6's distinctness is measured on a gate W1 cannot occupy, `V20_R15_IT2_MARS.md:199` | **FIRED**, REROUTE; upheld by SATURN, `V20_R15_IT3_SATURN.md:3` | **RIGHT** | — |
| 6 | **MARS** | **STRIKE 6** — criterion (3)'s `25.0×` is a row-order artifact, `:233` | **FIRED**, REPRICE; order-invariant reprice W3 `2.85` / W1 `118.8`, `V20_R15_JOURNAL.md:532-533` | **RIGHT** | — |
| 7 | **MARS** | **STRIKE 7** — the merge hides a distinction, `:329` | **FIRED**, REROUTE | **RIGHT** | — |
| 8 | **MARS** | **four attacks filed as NOT FIRING**, `V20_R15_IT2_MARS.md:263-289`: skyline-as-contender *NOT FOUND*; multiplier-as-arm *NOT FOUND*; *"W2 and W3 are one code path"* **NOT STRUCK** (SATURN struck it first); the DISTANCE line **OPEN, NOT FILED** | journalled as misses at the foot of his own report | **the honest form** — MARS is the only forecaster this round whose misses are in his own filing | — |
| 9 | **JUPITER** | it.2 merge: *"N = 1 primitive … The it.4 freeze should not freeze N=3 as three primitives"*, `V20_R15_IT2_JUPITER.md:112-122` | it.4 froze **N=2**; he narrowed himself: *"W1 and W3 are one primitive and two arena entries. `N = 2` measured. The isomorphism is conceded and it is unexercised"*, `V20_R15_IT4_JUPITER.md:20-24` | **RIGHT on direction** (`N < 3`), **too far on size**, corrected by his own measurement | **`−`** |
| 10 | **JUPITER** | cost law predicts `1.524 s` vs measured `1.497 s`, ratio `0.982` (−1.8 %), `V20_R15_IT2_JUPITER.md:153` | held | **RIGHT** | small `−` |
| 11 | **JUPITER** | *"prediction: **≥2×** at fixed `n`"*, `V20_R15_IT2_JUPITER.md:182` | not yet measured | **OPEN** | — |
| 12 | **JUPITER** | C4 RED count | *"retracted rather than restated"*, `V20_R15_IT4_JUPITER.md:177` | **WRONG, retracted** | unsigned |
| 13 | **MERCURY** | *"Only a clean sweep clears `0.5`"*, `V20_R15_IT3_MERCURY.md:47` | **CONFIRMED this iteration** `[RUN]`: 7/8 → `0.4735 < 0.5`; 8/8 → `0.6306` | **RIGHT**, now with its margin | — |
| 14 | **MERCURY** | E[cost to a crossing] `2.8358` / `118.8160`, `41.90×`, `V20_R15_IT3_MERCURY.md:274` | **reproduces exactly** from his own stated basis `[RUN]`; all-cell basis reads `2.85` / `129.29`, `45.4×` | **RIGHT within its stated basis**; the basis is not the arena's own seconds (§0.3) | — |
| 15 | **VENUS** | the §2 ranking + the §2.2 mechanism pre-registration | **OPEN** — scored at it.29, falsifier is the eight fresh seeds | filed **2026-09-02**; unreadable until the author's counter (D-CALIB-2) | — |
| 16 | **VENUS** | *(nothing filed at it.1–it.4)* | this office ran no iteration before it.5 | **NO ROW EXISTS**, and that absence is the ledger's first honest entry about me | — |

**Running tally, R15 to date: 5 right · 6 wrong · 1 half · 2 open · 1 retracted ·
1 filed-as-non-firing.** Of the wrong-with-a-sign: **3 `+`, 2 `−`**, one-sided
sign test `p = 0.5`. **The round's own bias is NOT established and no one may
cite one.** That is the point of opening the column early: it is currently empty
of a direction, and it will not be back-filled with one.

---

## 5. KILLS AND THEIR REPLACEMENT ROUTES

| kill | measured reason | replacement route | price |
|---|---|---|---|
| the briefed cost row as a single-basis table | `15.970/1.614/1.497` is a 2-seed timing run (`V17_R4_RETAKE_PRICE.md:194-196`, *"mean of 2"*); the arena cells read `16.161/1.780/1.681` | **reprice criterion (3) off the `secs` field the 24 arena cells already journal** | **0 GPU-s** — the numbers are in the file |
| *"re-run seed 2 to see if it holds"* | the pipeline is bit-reproducible per seed — an independent run reproduces retake seeds 0,1 to every printed digit; a re-run of seed 2 returns `0.203920` and carries no information | **8 fresh seeds 8–15, same cell**, against the §2.2 joint pre-registration | **≈ 129 GPU-s, ~2.2 min** |
| the SCOREBOARD's *"cost ≤1/10 incumbent +4"* as reachable this round | W3 costs **more** than `softmax` under both bases (`1.078×`, `1.059×`) | retire the `+4` from the expected total, or restate it as *cost ≤ incumbent at a crossing rate the incumbent cannot reach* — which W3 **does** satisfy, `5/8` against `0/8` | **0 GPU-s** |

---

## 6. DISTANCE, AND THE SCOREBOARD LINE

**DISTANCE.** The north star is attention equal to self-attention on its own
ground. One cell of twenty-four is below floor₁ by more than a rounding:
`arm_smprime` seed 2, `eval_nrmse 0.203920`, `dist_to_floor −0.503187`. The
skyline's best cell is `softmax` seed 1 at `0.938796`, `dist_to_floor +0.231689`
— **above** the floor. The tournament's best cell beats the floor by `0.503`; the
incumbent misses it by `0.232`; and the gap between them is carried by one seed
out of eight. **The distance to the north star this iteration is the distance
between a basin and a rate: the capability exists and is reached 12.5 % of the
time.** BED-K(a) and the witness are unrun; no state-distribution W1 number
exists on any bed.

**SCOREBOARD.** Moves at it.4, 14, 29, 35, 42, 43, 45 — **it.5 is not a scoring
iteration and nothing is claimed.** Standing: **+2** (wing list frozen, four
citations each, `V20_R15_WING_MANIFEST.md`) of ceiling **48**. The `+4` for
*"the winner's cost ≤1/10 incumbent"* is now measured **unreachable** by either
surviving wing (§5), which lowers the realistic ceiling to **44** until a wing is
repriced or a cheaper one is found.

---

*Filed by VENUS (IRENE), it.5, 2026-09-02. One competing prediction filed before
the deciding measurement: §2.2, eight fresh seeds, joint gate-state falsifier.
Node: `tests/venus/test_v20_r15_it5_ranking_record.py`, 15 passed, RED shown at
§0.4. No git writes. Nothing touched Kaggle.*
