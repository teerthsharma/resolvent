# V20 R15 — HEALTH INSPECTOR, it.5 / it.6 / it.7 (BACKLOG AUDIT)

Authority: **the LOG ONLY.** Every verdict below is whether a claim is BOUND to
data or source — never whether it is correct. No findings are filed. No git
writes. Nothing touched Kaggle. No mutation was applied, so none required
reversion.

Three iterations were unaudited on entry. This is a **partial audit inside one
wall clock**; §10 lists what was not reached.

---

## 1. LEDGER

| # | check | claim | verdict |
|---|---|---|---|
| A1 | retake integrity | `results/v17k_r4_retake.jsonl` untouched, 424 lines | **BOUND** |
| A2 | it.6 control | seeds 0/1 reproduce the retake, 59/60 columns, only `secs` | **BOUND** |
| A3 | it.6 regime diff | 17 (MERCURY) vs 18 (MARS) header fields | **BOTH BOUND** — different scopes |
| A4 | it.6 regime diff | MERCURY: "the only **two** header fields that differ" | **RED — four differ** |
| A5 | it.6 fresh cells | 8 new seeds + 2 controls = 10 cells | **BOUND** |
| B1 | STRIKE 9 source | `lambda_hat = float(lg.mean())` at `v15_r1.py:383` | **BOUND** |
| B2 | STRIKE 9 source | `unit_root = bool(a_max >= 1.0)` at `v15_r1.py:387` | **BOUND** |
| B3 | STRIKE 9 law | the iff, on 18 / 32 / 34 cells | **ALL THREE BOUND** — no contradiction |
| C1 | STRIKE 10 source | eval batch at hardcoded `seed=12345`, `v15_r1.py:699` | **BOUND** |
| C2 | STRIKE 10 | `4123/8192 = 0.5032958984375`, 8 of 10 fresh cells | **BOUND** |
| C3 | STRIKE 10 | `8148/8192`, `8002/8192` genuine off-census readings | **BOUND** |
| D1 | regime table | crossing `−0.7953` | **BOUND** |
| D2 | regime table | intermediate `−0.4411` | **BOUND** |
| D3 | regime table | flat band lower `−0.0009` | **RED — truncated, rounds to −0.0010** |
| D4 | regime table | flat band upper `−0.0436` | **RED — matches no datum anywhere** |
| D5 | regime table | over-decayed `−2.4989` / `−4.1869` | **BOUND** |
| D6 | regime table | over-decayed row's three columns | **RED — two different seed orders** |
| D7 | regime prose | "too fast (`λ̂_live ≤ −2.5`)" | **RED — excludes seed 13** |
| D8 | regime table | flat band `eval_nrmse 0.852–0.927` | **BOUND** |
| D9 | regime table | `qk` three regimes, no overlap | **BOUND** |
| E1 | coordinator | `ρ(beta, nrmse) = −0.0324 p=0.9053` | **BOUND — reproduced** |
| E2 | coordinator | `ρ(qk, nrmse) = +0.7176 p=0.0017` | **BOUND — reproduced** |
| E3 | coordinator | 16 distinct cells, dedup by `(seed, round(nrmse,6))` | **BOUND — dropped 2, doubled 0** |
| E4 | coordinator | β clusters `0.588–1.001` on the flat band | **BOUND — and it corroborates D4** |
| F1 | leap ledger | LEAPABLE must name a FIELD, not a theorem | **RED — VENUS row names the theorem** |
| F2 | leap ledger | MARS rows `L-M1..L-M5` appended to the ledger | **RED — they are not in the file** |
| G1 | withdrawal | MARS withdrew `_0step` entirely | **BOUND — written, unhedged** |
| G2 | withdrawal | `1 − (15/16)^8 = 0.403` | **BOUND** |
| G3 | withdrawal | "95 % confidence needs `n ≥ 44`" | **RED — n = 47** |
| G4 | self-correction | JUPITER corrected his own it.6 §6 seed | **BOUND — written in it.7** |
| G5 | self-correction | seed 11 is "the worst cell in the tournament" | **RED — seed 13 is worse** |
| G6 | self-correction | the sentence quoted as it.6 §6's | **RED — composite, not verbatim** |
| G7 | self-correction | VENUS repaired her falsifier | **BOUND — written and applied** |
| G8 | self-correction | VENUS admitted seed 3 was in her own it.5 table | **BOUND — written** |
| G9 | self-correction | that admission's citation `IT5_VENUS.md:122` | **RED — the row is `:120`** |
| H1 | journal | records MARS's and VENUS's withdrawals | **BOUND** |
| H2 | journal | records JUPITER's self-correction | **RED — absent** |
| H3 | journal | "live-band decay under 3 % per position" | **RED — seed 4 is 4.76 %** |
| H3b | MARS it.7 | the same "under 3 %", four lines below his own table | **RED — his seed 8 row is 3.20 %** |
| H4 | round consistency | the withdrawn predictor is retired round-wide | **RED — VENUS still scores it pending** |
| H5 | control table | MARS it.7 publishes 10 of 16 cells at 4 dp | **BOUND — all ten exact** |

**N audited = 39. M struck = 15** (A4, D3, D4, D6, D7, F1, F2, G3, G5, G6, G9,
H2, H3, H3b, H4).

---

## 2. THE it.6 EXPERIMENT — the round's load-bearing measurement

`results/v20_r15_it6_seeds8_15.jsonl`, 178 records: 1 header, 1 identity,
2 bind, 1 bar, 160 trace, **10 `t="cell"`**, 1 agg, 1 probe, 1 wall.

**A1 — the retake is genuinely untouched.** `sha256 =
26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e`, 424 lines,
tracked, last written by commit `847857e`, `git diff --quiet HEAD` clean, and
absent from `git status --porcelain`. **BOUND.**

**A2 — the control reproduces.** MARS's claim of 59 of 60 shared columns is
exact. The two cell schemas share **60** keys; for seed 0 and seed 1 alike,
**59/60 are byte-identical** and the sole difference is wall clock:

```
seed 0  secs: it6=16.104  retake=16.164
seed 1  secs: it6=15.957  retake=16.032
```

`lambda_hat_live` agrees to full precision (`−0.019537825137376785`,
`−0.011798891425132751`). **BOUND, MARS's count exact.**

**A3 — the 17/18 dispute is not a dispute.** Both headers carry the **same 22
keys**; **18 are equal**, 4 differ (`arms`, `seeds`, `tag`, `when`).
MERCURY's 17 is the length of `REGIME_FIELDS` in
`tests/mercury/test_v20_r15_it6_seeds8_15.py:27-30` — a curated comparability
whitelist, all 17 equal. `equal_set − REGIME_FIELDS == {'t'}`, the record-type
discriminator, whose value is `"header"` in both files and so matches
trivially. **MARS's 18 = MERCURY's 17 + `t`.** Both counts are correct under
their own stated definition, and each definition is stated in the office's own
test file rather than only in prose. Settled: **no strike either way.**

**A4 — STRUCK.** `V20_R15_IT6_MERCURY.md:62-63` states "The only two header
fields that differ are `arms` and `seeds`." **Four differ.** `tag`
(`v17k_r4_retake` → `v20_r15_it6_seeds8_15`) and `when` (`2026-08-31 23:08:44`
→ `2026-09-02 02:28:38`) also differ. Aggravating: MERCURY's own displayed
header block at `:52-57` prints only 14 of the 22 keys, omitting `tag` — but it
*does* print `when` with a visibly unequal value, directly above the sentence
that denies it. MARS's correction at `V20_R15_IT7_MARS.md:77` ("**four** differ
rather than being absent") is right on the substance. **RED against MERCURY.**
MARS's own `:76` puts "all seventeen header fields match" in MERCURY's mouth;
MERCURY wrote that the test *diffs* seventeen fields, which is true. Loose
paraphrase, sound correction underneath — **no strike on MARS for it.**

**A5 — the eight fresh cells are what both reports say.** Header
`seeds: [0,1,8,9,10,11,12,13,14,15]`, `arms: ["arm_smprime"]`, and exactly 10
`t="cell"` records match that list one-for-one. **BOUND.**

---

## 3. STRIKE 9 — the three denominators are three scopes, not three answers

**B1/B2 read from source and BOUND:**

```
scripts/v15_r1.py:380   fin = torch.isfinite(lg)
scripts/v15_r1.py:381   a_max = float(lg.max().exp())
scripts/v15_r1.py:383   lambda_hat=float(lg.mean()),
scripts/v15_r1.py:386   frac_gate_annihilated=float((~fin).double().mean()),
scripts/v15_r1.py:387   unit_root=bool(a_max >= 1.0),
```

`lambda_hat` is a mean over **all** positions including `−inf`; `unit_root` is
read off a **max**. Both exactly as struck.

**B3 — the count.** All three offices are arithmetically right about different
populations, and the whole spread is one duplication:

| office | scope, from the office's own test file | records | distinct cells | violations |
|---|---|---|---|---|
| MARS **18** | `arm_smprime` cells, it6 + retake, no dedup | 18 | 16 | **0** |
| VENUS **34** | all cells all arms, it6 + retake, no dedup | 34 | 32 | **0** |
| JUPITER **32** | the same union, deduped on `(arm, seed)` | 32 | 32 | **0** |

`34 = 24 + 10`; `32 = 34 − 2` (it.6's seeds 0/1 are the reproduction control —
the same cell counted twice); `18 = 8 + 10`, the `arm_smprime` slice of the same
34. JUPITER's fixture states the dedup rule in its own docstring, so 32 is the
only count of *distinct experimental cells* and is the strongest scope. VENUS's
sub-tallies inherit the duplication — her "17 with `frac > 0` / 17 with
`frac == 0`" is 15/17 on the deduped 32 — but her iff verdict is unaffected
because the duplicated rows are bit-identical and satisfy the law.

**The law holds on every scope.** Over the maximal population — every record in
`results/` carrying both fields, `t="cell"` and per-step `t="trace"` alike,
including the it.8 `arm_pl` run that landed mid-audit —
`lambda_hat == −Infinity` ⟺ `frac_gate_annihilated > 0` with **zero exceptions
in either direction**. Serialization is bare `-Infinity`, never `null` and never
a string, so the `== float('-inf')` test all three offices use is sound.

**Three offices, three scopes, one law. No strike. No office is wrong.** The
brief's premise that "three different denominators for one law" is a defect does
not survive contact with the test files. What it is instead: three scopes, each
declared, none reconciled by anyone before now.

**Attribution correction.** The brief places the 18 and the 34 in the it.5
reports and the 32 in `V20_R15_IT6_JUPITER.md`. All three are **it.7** claims —
`V20_R15_IT7_MARS.md:165-166`, `V20_R15_IT7_VENUS.md:124`,
`V20_R15_IT7_JUPITER.md:45-51`. `V20_R15_IT6_JUPITER.md` contains no "32 cells"
string; `V20_R15_IT5_VENUS.md` works over 24 cell records, not 34.

---

## 4. STRIKE 10 — every arithmetic identity confirmed, one label corrected

**C1 — BOUND.** `scripts/v15_r1.py:699` draws the eval batch at a literal
`seed=12345` that does not move with the cell seed, and `:703` prints the fact:

```
699:  x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345,
703:  print(f"  eval n={a.n_eval} seed=12345; per-seed train n={a.n_train} at the seed itself")
```

`12345` occurs at exactly those two lines and nowhere else in the file.

**C2 — BOUND, and the field is named precisely.** `4123/8192 =
0.5032958984375` exactly. It is published by **8 of the 10** fresh it.6 cells —
seeds 0, 1, 8, 9, 10, 12, 14, 15 — in the field **`frac_gate_annihilated`**,
not in any sign field. (`sign_acc` on those cells runs 0.999–1.0 and
`sign_acc_0step` runs 0.502–0.926; neither carries the constant. The audit
brief's phrase "no-arm sign census" mislabels the column. The *identity* is
exact and the *count of 8* is exact.) The same constant is published by 6 retake
cells and 2 `v17k_r4_floor.jsonl` cells: **16 cells across three separate runs
publish one identical float**, which is the mechanism the strike names — a fixed
eval batch gives a fixed annihilation census.

**C3 — BOUND.** The two cells that break the constant are genuine readings:

```
seed 11  frac = 0.99462890625    = 8148/8192 = 2037/2048  (exact)
seed 13  frac = 0.976806640625   = 8002/8192 = 4001/4096  (exact)
```

Both are `t="cell"` values with matching `t="trace"` histories (16 and 8 trace
rows), and both are the two `unit_root == False` cells of the ten.

---

## 5. THE `lambda_hat_live` REGIME TABLE — four strikes, one of them load-bearing

Re-derived from `results/` alone, never from the report.
`V20_R15_IT7_JUPITER.md:67-71` publishes the table; the `flat band` row reads
`n = 12`, `−0.0009 … −0.0436`, `eval_nrmse 0.852–0.927`.

| figure | data | round(4) | verdict |
|---|---|---|---|
| crossing `−0.7953` | seed 2 `−0.795290470123291` | `−0.7953` | **BOUND** |
| intermediate `−0.4411` | seed 3 `−0.4411417841911316` | `−0.4411` | **BOUND** |
| flat lower `−0.0009` | seed 6 `−0.0009529261151328683` | **`−0.0010`** | **RED (D3)** |
| flat upper `−0.0436` | *no datum, anywhere* | — | **RED (D4)** |
| over-decayed `−2.4989` | seed 13 `−2.4988784790039062` | `−2.4989` | **BOUND** |
| over-decayed `−4.1869` | seed 11 `−4.186896800994873` | `−4.1869` | **BOUND** |

**D3 — STRUCK, minor.** `−0.00095292…` truncates to `−0.0009` but *rounds* to
`−0.0010`; the 5th decimal is a 5 followed by nonzero. Every other figure in the
table is correctly rounded, so one row mixes two conventions. `−0.0009` does
occur at 4 dp on seed 6's **mid-training trace** (steps 100/130/140) and never
on any terminal cell — so if it was read off a trace, it was read off the wrong
row of a table whose other fifteen entries are terminal.

**D4 — STRUCK, and it is the load-bearing one.** A sweep of all **878** finite
`lambda_hat_live` occurrences in `results/` — every file, every step, `_0step`
twins included — returns **zero values that round to `−0.0436`**. The endpoint
is bound to nothing. The nearest cell is seed 8 at `−0.031969`; the nearest
value in the entire corpus is an unrelated `arm_pl` seed-6 step-0 reading of
`−0.04192`.

**The row contradicts itself, and its own other columns say which half is
wrong.** `n = 12`, the published `nrmse` range, and the coordinator's β range
all require **seed 4** (`λ̂_live = −0.047636`) inside the flat band:

- 12 cells land in the band only if seed 4 is admitted; `[−0.0436, −0.0009]` as
  printed admits **11**.
- `nrmse` over the 12 including seed 4 is `0.852061 … 0.926717` → **0.852–0.927**,
  exactly as published (D8, BOUND).
- seed 4's `λ̂_live` rounds to **`−0.0476`**.

`−0.0436` is a transcription of `−0.0476`. Three independent columns of the same
table are bound to the 12-cell reading; only the printed endpoint is not.
**RED on the endpoint, not on the banding.** Corrected band: **`−0.0010 …
−0.0476`.**

**Nothing in the suite pins the bad endpoints.**
`tests/jupiter/test_v20_r15_it7_q3.py:96` asserts the flat band as
`all(abs(live[s]) < 0.05 for s in flat)`, which the data satisfies (max
0.0476). So the `[RUN]` green on
`::test_w1_relaxation_rate_is_a_WINDOW_not_an_extreme` **does not certify the
published interval** — the test is wider than the prose, and the prose is what
is wrong. A report whose narrow figures are not the figures its own test
asserts is exactly the gap the log exists to catch.

**D6 — STRUCK.** The over-decayed row pairs its columns in two different seed
orders under one header reading "(11, 13)": `frac` `0.9768 / 0.9946` is seed
13/11, `lambda_hat_live` `−2.4989 / −4.1869` is seed 13/11, but `eval_nrmse`
`1.121 / 1.203` is seed **11/13**. Read left to right the row asserts
`λ̂ = −2.4989` goes with `nrmse = 1.121`; in the data `−2.4989` is seed 13,
whose `nrmse` is `1.203324`. One row, two orderings, neither matching the
header.

**D7 — STRUCK.** `V20_R15_IT7_JUPITER.md:78` writes the over-decayed regime as
"too fast (`λ̂_live ≤ −2.5`)". Seed 13 is `−2.49888`, which is **greater** than
`−2.5`. The stated threshold admits only seed 11 and excludes one of the two
cells the row describes.

**A control table exists, and it explains why nobody caught D3 or D4.**
`V20_R15_IT7_MARS.md:181-192` independently publishes 10 of the 16 cells at
4 dp, same round, same instrument. **Every one of its ten values matches the
re-derivation exactly** — `0:−0.0195, 1:−0.0118, 8:−0.0320, 9:−0.0035,
10:−0.0254, 12:−0.0137, 14:−0.0072, 15:−0.0072, 11:−4.1869, 13:−2.4989`. MARS
covers only the ten it.6 cells and omits retake seeds **4, 5, 6, 7**. The two
cells JUPITER misreports are **seed 4 and seed 6** — precisely the window the
control does not reach. Two offices published the same quantity in the same
iteration and the overlap was blind exactly where the error was. MARS also
pairs the over-decayed cells correctly (`11 → −4.1869 → 1.120603`;
`13 → −2.4989 → 1.203324`), which makes D6 a direct contradiction of a control
table sitting in the same round.

**H3b — STRUCK, and it is MARS's, not only the journal's.**
`V20_R15_IT7_MARS.md:193-195` writes "the surviving half of the band decays by
**under 3 % per position**". His own table row directly above says seed 8 is
`−0.0320`, i.e. **3.20 %**. The claim is contradicted by the evidence printed
four lines earlier, and the same sentence is what the journal repeats at
`:1191-1194` where seed 4 pushes it to 4.76 %. One understatement, three
documents, and it is the same understatement that produced `−0.0436`.

**The banding itself is clean.** With seed 4 admitted, the four regimes
partition all 16 distinct `arm_smprime` cells with no overlap and none left
out: crossing {2}, intermediate {3}, flat {0,1,4,5,6,7,8,9,10,12,14,15},
over-decayed {11,13}. Membership matches `frac_gate_annihilated` exactly — the
12 flat cells are precisely the 12 at `0.5032958984375`, the 2 over-decayed are
precisely the 2 above 0.9, and seeds 2 and 3 are singletons at `0.0` and
`0.4967041015625`.

**D9 — BOUND.** The `qk` separation at `:103`:
`max(qk|crossing) = 0.5006 < min(qk|flat) = 1.1694`, and
`max(qk|flat) = 1.6525 < min(qk|over-decayed) = 2.0305`. Strict, no overlap.

---

## 6. CHECK — THE COORDINATOR

**E1/E2/E3 — reproduced independently, to the digit.** Cells: the 18
`arm_smprime` `t="cell"` records from it.6 + retake; dedup on
`(seed, round(nrmse,6))`; `beta` and `qk` from `manifest.smp_values`; Spearman
by average-rank Pearson with a two-sided t test on `n−2 = 14` df.

```
arm_smprime cells before dedup: 18
  DROPPED dup: retake seed 0  nrmse 0.926082
  DROPPED dup: retake seed 1  nrmse 0.881247
after dedup: 16     seeds kept: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

rho(beta, nrmse) = -0.0324  p=0.9053  n=16
rho(qk,   nrmse) = +0.7176  p=0.0017  n=16
```

**The dedup dropped exactly 2 and doubled 0.** The 16 survivors are seeds 0–15,
none twice, none missing. Both coefficients and both p-values match the
coordinator's `[RUN]` at the published precision, and `+0.717647` matches
`V20_R15_IT7_JUPITER.md:91` at 6 dp. **BOUND.** The dedup key is also the right
one: it is only safe because the duplicated rows are bit-identical, which A2
independently established.

**E4 — the clustering is correct arithmetic, and it indicts D4.**

```
beta over the 12-cell flat band (seed 4 admitted):  0.587580 … 1.000812  ->  0.588-1.001
beta over the 11-cell band as printed (-0.0436):    0.696192 … 1.000812  ->  0.696-1.001
beta outside:  seed 11 = -0.060234   seed 2  = 1.343933
               seed 3  =  1.509306   seed 13 = 1.989505
```

The coordinator's `0.588` **is seed 4** — the cell JUPITER's printed endpoint
excludes. So the coordinator and JUPITER are reading the same 12-cell band, and
only JUPITER's endpoint is mistyped. The coordinator's figure is independent
corroboration of D4, arrived at without noticing it.

The separation is real either way: largest flat-band β is `1.000812`, smallest
non-flat β is `1.343933`, a clean gap, with seed 11 far below at `−0.060234`.
**BOUND as arithmetic.** Whether a β near 1 *means* the arm converges to softmax
rather than to its own corner is a mechanism claim and is **not mine to rule
on** — the log records only that the interval and its separation are computed
correctly, and that its lower endpoint is a cell the published regime table
excludes.

**The coordinator adjudicated a contradiction that was real.** JUPITER's it.6 §6
transfer and the it.7 β reading genuinely conflict, and the `[RUN]` settles it
with a number rather than a preference. That is the right shape. Two log defects
attach to it, both in §7 and §8 below: the journal does not record that JUPITER
himself struck the it.6 seed (H2), and the journal repeats MARS's claim of a
ledger append that did not occur (F2).

---

## 7. WITHDRAWALS AND SELF-CORRECTIONS — all three are written

The rule applied: a withdrawal needs no RED, but it must be **in a report
file**, not only in a return message. All three pass that bar. Each carries a
smaller defect alongside.

**G1 — MARS's `_0step` withdrawal is WRITTEN and unhedged.**
`V20_R15_IT7_MARS.md:16`, under its own heading `## 0. MY OWN PREDICTOR CAME
BACK VACUOUS, AND IT IS WORTH NOTHING`, at `:33-36`:

> **Verdict: nothing, not one supporting instance in sixteen.** An instance that
> only re-scores the cell the rule was cut from is not an instance. **I withdraw
> `_0step < 0.15` as a predictor** and restate it as what it is: a description
> of seed 2. It should not be scored again at it.29.

Three independent marks of a full withdrawal: the predicate is renamed a
description of one cell, it is de-scheduled from its own scoring iteration, and
it is graded TERMINAL in his ledger row. The §1.3 replacement offers the
*trained* columns, a different quantity, so nothing in the file keeps it alive.
**BOUND.** `V20_R15_IT5_MARS.md` is not corrected in place and read alone still
asserts the predictor at `:105` — which is permitted, since the withdrawal lives
in a report file.

**G2 — BOUND.** `1 − (15/16)^8 = 0.4032805261667818` → `0.403`. The companion
"~60 % of the time an 8-seed run issues no positive" is the complement,
`0.5967`, also correct. The scoring behind it checks out: across all 16
`arm_smprime` cells only seed 2 falls below 0.15 (`_0step = 0.124755859375`) and
it crosses (`nrmse = 0.20392`), so the positive branch really is n = 1.

**G3 — STRUCK.** The same paragraph (`:40`, repeated in ledger row L-M2 at
`:216` and in the journal at `:1214`) says "Reaching 95 % confidence of one
crossing needs `n ≥ 44`." At p = 1/16, `1 − (15/16)^44 = 0.9416`; 95 % is first
reached at **n = 47** (`ln 0.05 / ln(15/16) = 46.42`). The withdrawal does not
depend on the number, but the bound is understated by three seeds and it has
already propagated into two other documents.

**G4 — JUPITER's self-correction is WRITTEN, in it.7.**
`V20_R15_IT7_JUPITER.md:106`, headed `### 2.2 THE it.5 SEED IS CORRECTED,
AGAINST THIS OFFICE'S OWN §6`, at `:108-110` and `:124-127`. Terminology note
that changes the question: the "seed" here is the **Q3 seeded question**
(`V20_R15_IT6_JUPITER.md:276` is headed `## 6. Q3 SEED`), not a numeric random
seed. What is struck is the transfer of annex M1's predicted descent to W1's
sixteen cells, and the number that kills it is `ρ = −0.032`. He files it as an
instance against his own office. **BOUND.** `V20_R15_IT6_JUPITER.md:276-300` is
not corrected in place and in fact sharpens the claim, with no erratum and no
forward reference — permitted, but it means the it.6 report read alone is
misleading.

**G5 — STRUCK.** `V20_R15_IT7_JUPITER.md:117-118` calls seed 11 at
`beta = −0.06023` "the **worst cell in the tournament** (`nrmse = 1.120603`)".
Seed 13 is worse at `1.2033239267331008`. Seed 11 is second-worst. The
corner-cell argument survives — both are the only two cells above 1.0 — the
superlative does not.

**G6 — STRUCK.** The sentence presented in quotation marks as it.6 §6's —
*"W1's failures are annex M1's predicted descent — β descends toward the exact
corner"* — is **not verbatim in it.6 §6**. it.6 §6 quotes VENUS/journal as "W1's
seven failing cells are the predicted descent to the exact corner"; the phrase
"β descends toward the exact corner" is at `V20_R15_JOURNAL.md:817`. It is a
composite quotation stitched across two documents and attributed to one. The
substance of the correction is unaffected; the citation is not sound.

**G7/G8 — VENUS's repairs are BOTH WRITTEN.** The falsifier repair at
`V20_R15_IT7_VENUS.md:90-111`, headed `### 1.3 The falsifier's blindness — the
actual finding, and it is mine` and `### 1.4 The repair — falsifiers as
complements, not as predicates`:

> The mechanism I asserted was **set membership on `frac`**; the killer I wrote
> was a **predicate on gate liveness**. Those are different variables.

She names it as M-7 recommitted by the office that had read M-7, and the three
re-filed laws L1/L2/L3 each carry a complement-form falsifier — so the repair is
**applied, not merely announced.** The seed-3 admission at `:132-136`:

> The counterexample (seed 3) was **already in the table it.5 printed** and this
> office did not read its own table.

Verified against the data: seed 3 is `frac 0.4967041015625`, `lambda_hat −inf`,
`unit_root False`. **Both BOUND**, and G8 is the strongest self-correction of
the three because it names the reading failure rather than only the wrong
conclusion.

**G9 — STRUCK, minor.** That admission cites `V20_R15_IT5_VENUS.md:122`. The
seed-3 row is at `:120`; `:122` is the seed-5 row. Right table, wrong row.

---

## 8. THE LEAP LEDGER

Contract, `CEQ_V20_R15_CONTRACT.md:142-145`:

> it.35 EVALUATION-FOR-LEAP: JUPITER + MARS grade every F1/F2/F3 failure
> LEAPABLE (name the FIELD likely to hold the missing theorem, not the
> theorem) or TERMINAL (a bound; no theorem removes a bound).

**RULING ON THE QUESTION PUT TO ME: a LEAPABLE grade that names a theorem
rather than a FIELD is INADMISSIBLE.** The clause is not stylistic. A field is a
body of existing mathematics a leap can be pointed at and made to read; a
theorem statement is the output the leap is supposed to produce. Grading a
failure LEAPABLE while naming the theorem asserts the conclusion as if it were
the resource, and converts an open conjecture into a citation. Whether the named
text has literature behind it is checkable, so the log will enforce it.

**F1 — STRUCK. One row violates it: the VENUS row,
`V20_R15_LEAP_LEDGER.md:55-56`.** Under a column whose own header reads "the
field (never the theorem)", it puts:

> the extent of corner descent in multiplicative-gate landscapes — how far along
> the annihilating face a gradient descent travels before it stalls, as a
> function of depth, and whether the stalling points form a lattice

Three failures in one cell: a quantity in this round's own system rather than a
discipline; then the statement of the missing theorem, spelled out; then a
conjecture. Nothing is named that a leap could go read. Its own falsifier line
at `:58` treats the lattice clause as a conjecture to be tested, confirming the
reading. Compare the rows that pass, each led by a head noun with textbooks
behind it: realization theory for LTI systems (L-1), formalized linear algebra
over Mathlib (L-4), transfer-operator / Koopman spectral theory (L-5),
non-equilibrium statistical mechanics — projection-operator formalism (L-7),
error-statistical severity (L-M1), anti-concentration / small-ball probability
(L-M3). **RED against VENUS.**

Nearest the line but **admissible**: L-M3's parenthetical "(Littlewood–Offord,
Rudelson–Vershynin)" cites a named problem and two authors, and L-1/L-5 append
clauses describing what the theorem should say. All three keep a genuine field
as the head noun, which is what the clause requires. **Naming a field and then
saying what you want from it is compliant; naming the want and calling it a
field is not.** That is the line, and it is now on the record.

**F2 — STRUCK. `L-M1..L-M5` are not in the ledger, and two documents say they
are.** `V20_R15_IT7_MARS.md:210-211`: "Five rows appended to
`V20_R15_LEAP_LEDGER.md` (the file did not exist; created with MARS's rows under
their own heading so JUPITER's append merges)". `V20_R15_JOURNAL.md:1349-1350`
repeats it: "Leap ledger open with five MARS rows (`L-M1..L-M5`)". The ledger is
60 lines and holds nine rows — L-1..L-8 (JUPITER, `:22-29`) and one VENUS row
(`:49-60`). Repo-wide grep finds `L-M1..L-M5` only inside `V20_R15_IT7_MARS.md`
and that one journal line. **The ledger itself contradicts both** at `:43-47`:
"Every row above is JUPITER's; MARS has produced graded failures this round that
are not yet here." The parenthetical "the file did not exist" is also false —
JUPITER opened it at it.7 with eight rows. A report asserting a write that did
not occur is unbound regardless of the rows' quality, and the coordinator
inherited the claim without checking the file. **RED against MARS, and the
journal repeats it.**

**Not struck, recorded for the coordinator, who owns the vocabulary:** three
grades outside the contract's two-value set — L-2 "TERMINAL as a leap target,
LEAPABLE by 0 GPU-s of bookkeeping", L-6 "LEAPABLE, but cheaper to measure than
to leap", L-M4 "split". L-2 additionally claims a LEAPABLE half while leaving
its field column "none". `V20_R15_JOURNAL.md:1350` carries "split" forward as a
grade rather than resolving it. L-M5 is TERMINAL with no bound, only a reason.
And it.35 names **JUPITER + MARS** as the grading offices: the only non-JUPITER
row actually in the file is VENUS's — an office the contract does not authorize
to grade, and the one row that breaks the naming rule.

---

## 9. THE JOURNAL, AND ONE ROUND-WIDE INCONSISTENCY

**H1 — BOUND.** `V20_R15_JOURNAL.md:1209-1215` records MARS's withdrawal and
corrects the coordinator's own it.6 line in doing so ("**Worth nothing** — not
'one instance in sixteen', which is what this office wrote at it.6 and is too
generous", against `:1090-1092`). `:1245-1249` and `:1259-1261` record both
halves of VENUS's. Self-correction by the coordinator, in the journal, on the
record — the right shape.

**H2 — STRUCK.** The journal does **not** record JUPITER's self-correction. The
it.7 JUPITER section (`:1224-1236`) carries only the Q3 grades and the green
`lake build`. The M1-transfer kill, the `ρ = −0.032`, and "it is this office's
own" appear nowhere. The only trace is `:1186`, where the it.6 §6 seed is listed
as *collateral of MARS's STRIKE 9* — i.e. the journal attributes to an external
strike what the office did to itself. **Two of three self-corrections are in the
journal; the third, by the office that has produced the most of the round's
current mechanism story, is not.**

**H3 — STRUCK.** `V20_R15_JOURNAL.md:1191-1194` claims "live-band decay under
3 % per position" for the 0.88–0.93 band cells. The data contradicts it: seed 4
is 4.76 % and seed 8 is 3.20 %. Same underlying error as D4 — both understate
the flat band's true spread, and the journal reached it independently of
JUPITER's table, which means the understatement is now in two places.

**H4 — STRUCK.** `V20_R15_IT7_VENUS.md:285` still scores MARS's withdrawn
predictor as "**UNTESTED, not confirmed — vacuous agreement.** One supporting
instance in 16", and `:290` prices "~2 GPU-s to test properly". That is VENUS's
scoreboard, not a MARS hedge, but the round's it.7 record is internally
inconsistent about whether `_0step < 0.15` is retired or pending. A withdrawal
that is not propagated to the offices scoring it has not finished landing.

---

## 10. WHAT I DID NOT REACH

Named plainly, because a partial audit that hides its gaps is worse than none.

1. **`V20_R15_IT6_WILSON.md` (37 KB) — entirely unread.** The largest report of
   the three iterations, and it received no audit at all. This is the biggest
   single hole in this filing.
2. **it.5 in its own right.** `V20_R15_IT5_VENUS.md` and `V20_R15_IT5_MARS.md`
   were read only where they bear on STRIKE 9, STRIKE 10, the it.6 control and
   the withdrawals. Their other claims are unaudited.
3. **The coordinator's it.5–it.7 journal narrative** was searched for the
   Spearman block, the leap grades, the withdrawals and the flat band. It was
   not audited as a whole against the reports; H2 and H3 were found by targeted
   search and there may be more of the same kind.
4. **JUPITER's it.7 §2/§3 Lean citations** — `lean/CEQ/V16Domain.lean:129` and
   the Koopman/BED-1 adoption table at `:227` — not opened. The `[RUN] lake
   build` green is recorded in the journal but not independently reproduced.
5. **No mutation-and-revert was performed this round.** Every check above is
   read-only recomputation. Claims that need a mutation to test — that a test
   goes RED *for the reason stated* rather than incidentally — are unverified.
   Three offices quote deliberate-RED assertion messages as evidence; I
   confirmed the arithmetic **in** those messages, not that each assertion fires
   for its stated cause. D4 shows why that matters: a green test whose predicate
   is wider than the prose it is cited for certifies less than it appears to.
6. **The it.8 artefacts** (`results/v20_r15_it8_armpl_b.jsonl`,
   `results/v20_r15_it8_armpl_seeds8_15.jsonl`) were examined only far enough to
   confirm they postdate every report audited here and satisfy the STRIKE 9 law.
   They are not audited.

---

## 11. TREE STATEMENT

No mutation was applied, so none required reverting and no digest restoration
was needed. The only file this audit wrote is this report.
`results/v17k_r4_retake.jsonl` verified byte-identical to its HEAD blob.

`git status --porcelain` at close, reported in full and honestly, including
paths from concurrent it.8 work by MERCURY and JUPITER:

```
 M house-events.jsonl
 M pytest.ini
?? CEQ_V20_R15_CONTRACT.md
?? V20_R15_IT1_INSPECTOR.md
?? V20_R15_IT1_JUPITER_M14.md
?? V20_R15_IT1_MARS.md
?? V20_R15_IT1_SATURN.md
?? V20_R15_IT1_WILSON.md
?? V20_R15_IT2_INSPECTOR.md
?? V20_R15_IT2_JUPITER.md
?? V20_R15_IT2_MARS.md
?? V20_R15_IT2_SATURN.md
?? V20_R15_IT3_INSPECTOR.md
?? V20_R15_IT3_MERCURY.md
?? V20_R15_IT3_SATURN.md
?? V20_R15_IT4_INSPECTOR.md
?? V20_R15_IT4_JUPITER.md
?? V20_R15_IT4_SATURN.md
?? V20_R15_IT567_INSPECTOR.md
?? V20_R15_IT5_MARS.md
?? V20_R15_IT5_VENUS.md
?? V20_R15_IT6_JUPITER.md
?? V20_R15_IT6_MERCURY.md
?? V20_R15_IT6_WILSON.md
?? V20_R15_IT7_JUPITER.md
?? V20_R15_IT7_MARS.md
?? V20_R15_IT7_VENUS.md
?? V20_R15_IT8_JUPITER.md
?? V20_R15_IT8_MERCURY.md
?? V20_R15_JOURNAL.md
?? V20_R15_LEAP_LEDGER.md
?? V20_R15_WING_MANIFEST.md
?? results/v20_m14_cheeger.txt
?? results/v20_r15_it6_seeds8_15.jsonl
?? results/v20_r15_it8_armpl_b.jsonl
?? results/v20_r15_it8_armpl_seeds8_15.jsonl
?? scripts/iteration_timer.sh
?? scripts/v20_m14_cheeger.py
?? tests/jupiter/test_m14_cheeger.py
?? tests/jupiter/test_v20_r15_it2_ldom_census.py
?? tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py
?? tests/jupiter/test_v20_r15_it6_ldom_census.py
?? tests/jupiter/test_v20_r15_it6_q1_exact_class.py
?? tests/jupiter/test_v20_r15_it6_q2_outside_bound.py
?? tests/jupiter/test_v20_r15_it7_q3.py
?? tests/jupiter/test_v20_r15_it8_q4_q5.py
?? tests/mars_v20/
?? tests/mercury/arena_price.py
?? tests/mercury/test_v20_r15_it3_arena_price.py
?? tests/mercury/test_v20_r15_it6_seeds8_15.py
?? tests/mercury/test_v20_r15_it8_armpl_and_clamp.py
?? tests/saturn/test_v20_r15_freeze_manifest.py
?? tests/saturn/test_v20_r15_wing_rubric.py
?? tests/saturn/test_v20_r15_wings_distinct.py
?? tests/saturn/test_v20_r15_wings_distinct_percell.py
?? tests/venus/
```

Paths belonging to concurrent it.8 work, not to this audit:
`results/v20_r15_it8_armpl_seeds8_15.jsonl` (header + identity only, 0 cells —
an aborted start) and `results/v20_r15_it8_armpl_b.jsonl` (an `arm_pl` run
written at 02:58 **during** this audit and still appending; 9 cells, all
`frac_gate_annihilated == 0` and all `lambda_hat` finite, consistent with the
STRIKE 9 law but postdating every report audited here and outside every office's
stated scope). That concurrency is deliberate and is not a defect. The audit
precedes the verdict, not all work.

---

## 12. VERDICT

**39 audited, 15 struck.**

**The it.6 experiment is sound and the round may keep building on it.** The
control reproduces bitwise on 59 of 60 columns with only wall clock differing;
the regime is unchanged on every field that decides comparability; the retake is
untouched and byte-identical to its committed blob; the eight fresh cells are
what both reports say. The apparent three-way contradiction on STRIKE 9 was
three correctly-counted scopes, and the law survives all of them with zero
exceptions over every record in `results/`. STRIKE 10's identities are exact to
the last bit. The coordinator's adjudication reproduces to the digit and his
dedup dropped exactly the two rows it should have and doubled none.

**The strikes are almost entirely transcription, propagation and bookkeeping —
not measurement.** Not one of the fourteen overturns a number that was actually
computed. That is worth saying plainly: the offices' instruments are honest and
their arithmetic holds. What fails is the trip from a computed value to a
printed one, and from a printed one to the next document.

**Two strikes are load-bearing and should be fixed before anything else builds
on them:**

- **D4.** The regime table's flat-band ceiling `−0.0436` is bound to nothing in
  878 measurements. It should read `−0.0476`, and the table's own `n = 12`, its
  own `nrmse` range, and the coordinator's β range each independently say so.
  The regime table is what the round's current mechanism story rests on.
- **F2.** Five leap-ledger rows are asserted as appended by MARS and re-asserted
  by the journal, and they are not in the file. The ledger says so itself.

**And one ruling is now on the record:** a LEAPABLE grade that names a theorem
instead of a field is inadmissible, and the VENUS row is the instance.

**Priority 4 was reached after all** — all three withdrawals are written, and
that is the cleanest result in this filing. **Priority 3 cost the round four
strikes.** `V20_R15_IT6_WILSON.md` is the largest known gap.
