# V20 R15 — it.8 + it.9 — HEALTH INSPECTOR

**Branch `v17k-gate0`, HEAD `207e7b9`. No git writes. Nothing touched Kaggle.**
Authority is over the LOG ONLY: whether a claim is BOUND, never whether it is correct.
12 nurses dispatched in one batch; every re-run written to scratchpad only.

---

## VERDICT ON THE `+6` — **NOT EARNED. THE THEORY TABLE IS NOT BOUND.**

Twelve cells are *graded*. The contract's bar is not "twelve grades" but **"every exit
F-graded with a gap"**. **Q6 is the cell that fails it, and it fails on the log, not on the
mathematics.** JUPITER's Q6 argument is, in this office's reading, *correct*. Its HOW-BAD
gap cites a demonstration that **does not say what the report says it says**, and the
ledger row hardens the misstatement one notch further. A gap citing a misdescribed
artifact is an ungrounded gap, and an ungrounded gap is not a gap.

**The `+6` should stay withheld until §1.3–§1.5 are repaired. It is a short repair, and
the conclusion survives it.**

---

## 1. PRIORITY 1 — THE THEORY TABLE, CELL BY CELL

### 1.1 The table as filed `[READ V20_R15_IT9_JUPITER.md:13-20]`

| | W1 `arm_smprime` | W3 `arm_pl` | bound? |
|---|---|---|---|
| Q1 EXACT CLASS | F0 | F1 | §1.7 |
| Q2 OUTSIDE | F1 + const | F1 + const | not reached — §7 |
| Q3 LEARNABILITY | F2 | F1 + const | not reached — §7 |
| Q4 COST LAW | F3 | F3 | not reached — §7 |
| Q5 INFORMATION FLOOR | F1 + const | F1 + const | not reached — §7 |
| **Q6 STATE METRIC** | **F4** | **F4** | **NO — §1.3–§1.5** |

### 1.2 What the office got right, and it is most of it

JUPITER did not grade from the armchair. The Q6 cell is carried by an executed suite
(`tests/jupiter/test_v20_r15_it9_q6.py`) with **three named, seeded planted negatives**,
and §1 of his report opens by disambiguating `W1` the wing from `W1` the metric — the
conflation that would have made the whole section unreadable. §3.5 refuses to cite M13's
own advertised `[RUN: 0.492 vs KL 0.519]` after finding that **neither number is a
Wasserstein or a KL in this tree** — an office declining to spend evidence it had already
been handed, and the fifth domain-empty theorem of the round. §8 states what he is **not**
claiming. That is the shape of a bound cell.

### 1.3 STRIKE I-1 — the Q6 planted negative is not the experiment the report describes `[RUN, re-executed]`

The report, `V20_R15_IT9_JUPITER.md:146-148`:

```
predictor := the oracle's own values, permuted
1-Wasserstein(pred, oracle) = 0.0        exactly
NRMSE(pred, oracle)         = 1.41       (sqrt 2), worse than predict-the-mean
```

Re-executed from the shipped test at `tests/jupiter/test_v20_r15_it9_q6.py:149-162`:

| claim | re-run | verdict |
|---|---|---|
| "the **oracle's own values**, permuted" | the test never calls `equilibrium_oracle`; it uses `torch.randn(4096)` | **FALSE AS WRITTEN.** The file's own docstring (`:4-6`) promises *"the arms and the oracle are imported from the shipped modules"* — true of its siblings, false of this one |
| `W1 = 0.0` exactly | `0.0` — **and `0.0` at seeds 0, 1, 7, 4096, 999983 alike** | true but **tautological**: `W1` here is the mean absolute difference of the two *sorted* vectors, and `pred` is a permutation of `z`. **Seed 4096 does no work** |
| `NRMSE = √2` | `1.4060346618513293` against `√2 = 1.4142135623730951`, off `−8.179e-03` (0.58%) | **NOT EXACT.** The test asserts `rel=0.05` on the **un-normalised RMSE** (`1.4305`), not on the NRMSE |
| `NRMSE > 1` | confirmed, synthetic **and** real-oracle | true |

**The underlying argument survives the repair.** Re-run against the *actual*
`equilibrium_oracle` (`scale/negation_scope.py:286`) at `n=4096, s=64, d=24, d_model=16`:
`W1 = 0.0` exact, `NRMSE = 1.421901019003236`. Permutation-blindness is real, and it is
real on the object the report meant to use. **What is unbound is the citation, not the
conclusion.**

Collateral, and consistent with JUPITER's own §3.5: `grep -rn "wasserstein\|kantorovich"
--include=*.py .` returns **one hit repo-wide** — the string literal inside the offender
list at `test_v20_r15_it9_q6.py:136`. The `W1` in the planted negative is four inline
tensor ops, not a shipped, reviewed scorer.

### 1.4 STRIKE I-2 — the ledger hardens the approximation the report hedged

`V20_R15_IT9_JUPITER.md:301` writes `NRMSE ≈ √2`. `V20_R15_LEAP_LEDGER.md:131` writes
**`scores W1 = 0.0 exactly at NRMSE = √2`** — the `≈` is dropped on the way into the row
that the leap model reads at it.35. The measured value is `1.4060`, and the assertion
behind it is a 5%-tolerance check on a different quantity. **This is the same mechanism as
D4 (§3): a hedge in the report, a hard number in the artifact downstream.** It is the
second instance this round, and the first one cost `−0.0436`.

### 1.5 STRIKE I-3 — the Q6 grounding is journalled nowhere

The it.9 journal record (`V20_R15_JOURNAL.md:1518-1710`) carries the F4/F4 grade and
**none of its evidence**. Zero hits for `wasserstein`, for `4096` as a seed, for `√2`, or
for the permutation argument; **zero `results/*.jsonl` records the computation**. The gap
that earns the `+6` lives in one iteration report and one test file. **The journal is the
record, and the record does not contain the reason.**

### 1.6 The suite is no longer green `[RUN]`

`pytest tests/jupiter/test_v20_r15_it9_q6.py` → **`1 failed, 9 passed`**, against the
report's claimed `10 passed` (`:5`, `:291`). The failure is the *other* planted negative,
`::test_d4_planted_negative_the_struck_endpoint_is_bound_to_no_datum`: `assert n == 878`
now reads **932**, because `results/v20_r15_it10_mercury_rescore.jsonl` — MERCURY's
concurrent it.10 artifact — adds 54 finite `lambda_hat_live` fields.

**This is not JUPITER's error at filing time.** It is an un-pinned sweep asserting a count
over a directory that other offices write to, and it will keep breaking every time
`results/` grows. **The it.9 green is not reproducible today and the reason is structural.**
Q6's own test is unaffected and passes alone.

### 1.7 Q1 — the Lean half

`pathProd_polar` (`lean/CEQ/V16Domain.lean:105`) and `pathProd_eq_zero_iff` (`:129`),
the `1.387779e-16` and `1.110223e-16` instances, and the rebuild — **see §7. This is the
one Priority-1 item the wall took.**

### 1.8 Q6 is "no object", but in the weaker of the two available senses — and JUPITER says so

`V20_R15_IT9_JUPITER.md` §8 concedes the marginal *"is uncomputable from the banked
record, and §3.6 route 1 makes it computable for 0 GPU-seconds."* An `[n]` vector **is** a
sample from which an empirical distribution can be formed. So the F4 is not "the metric
has no object" in the strong sense — it is **"the per-draw object does not exist, and the
pooled object that does exist is inadmissible on this bed"**, the second half carried
entirely by §3.4.

**That is still a sound F4, and it is exactly why §1.3 matters.** §3.4 is the only thing
standing between this grade and "we did not try", and §3.4 is the misdescribed one. Repair
§3.4 against the real oracle — the numbers are in §1.3 above and they hold — and Q6 binds.

---

## 2. PRIORITY 2 — MARS's FOUR it.9 STRIKES: **ALL FOUR UPHELD**

### 2.1 The it.8 agg disagrees with its own cells — **UPHELD, bitwise** `[RUN]`

Sole `t="agg"` record: `results/v20_r15_it8_armpl_b.jsonl:159`. Recomputed against
`scripts/v15_r1.py:899-909` (`statistics.fmean`; `statistics.stdev`, ddof=1;
`tcrit = t.ppf(0.975, df=len(a.seeds)-1)`; `crosses = bool(m + half < floor1)`):

| | n=9 | n=8 (seed 9 dropped) |
|---|---|---|
| mean | `0.7175018067286933` | `0.6469670834460266` |
| sd | `0.21245444406217273` | `0.02029964404207494` |
| ci_hi | `0.8808087489119841` | **`0.6639380105668372`** |
| crosses | **False** | **True** |

**n=9 reproduces the journalled record bitwise on `mean`, `sd`, `ci_lo`, `ci_hi`.** n=8
hits MARS's `0.6639380105668372` exactly at df=7; at df=8 it is `0.6635172935237913`, and
`crosses` is `True` either way — **the strike's conclusion holds under both readings.**
sd ratio `10.465919679272197×`. **8 of 9 cells sit below `floor_1 = 0.7071067811865476`**;
the ninth is seed 9 at `1.281779592990027`.

**What MARS found is sharper than "disagrees", and the sharper version is the useful one:**
`crosses` names two different estimands in this repo. The agg field tests the **mean's**
upper CI bound; the journal's "N of M cross" counts **cells**. Both computations are
correct on the same nine numbers.

**STRIKE I-4:** the machine-emitted `crosses: false` on this very file appears **nowhere**
in `V20_R15_JOURNAL.md`. A reader following the journal's "7 of 8 fresh seeds cross" has
no way to learn the instrument's own verdict on the same file says the opposite.
`dist_to_floor` is `+0.0104` against a CI half-width of `0.1633` — **the aggregate is 16×
less resolved than the gap it is being asked to adjudicate**, and the record's own
`achieved_power: 0.2627` says so in the same line.

Latent, found in passing and not yet fired: `tcrit` takes its df from `len(a.seeds)`
(`:899`) while `half` divides by `len(v)` (`:904`). They agree only when every seed yields
a cell for every arm. **Not triggered here — both are 9 — and it is one dropped cell away
from being triggered.**

### 2.2 `n_eff = 1` — surviving half **BOUND**; withdrawal **written in the report, absent from the journal**

`scripts/v15_r1.py:699` —

```
699    x_ev, y_ev, f_ev, p_ev = batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345,
700                                      device=dev)
```

— sits **above** both loops at `:706-707`, where the train draw two lines later takes
`seed=seed` from the loop. The contrast is two lines wide. Only two occurrences of `12345`
in the file (`:699`, and `:703`, a print string). **No `--eval-seed` flag exists**
(argparse, `:547-558`); the eval seed is unreachable without editing source. Generators
are CPU-seeded (`scale/negation_scope.py:426`, `:656`), so the tensor is bit-identical
across separate processes.

Consumers of that one tensor: `:701`, `:729`, `:734-735`, `:745`, `:244`, `:269-271`,
`:754-782`, `:803-809`, and finally `:902` feeding the aggregate. **All 16 `arm_pl` rows
and all 8 `softmax` rows are readings of one eval condition.** The `stdev` at `:903` is
over *training* seeds only, so the eval draw contributes zero variance; `bootstrap_ci`
(`:271`) resamples rows of the same tensor and does not restore it either. **BOUND.**

The withdrawal is **explicit and unhedged** — `V20_R15_IT9_MARS.md:355-356`:

> **I filed STRIKE 10 expecting a differential and there is none. It is withdrawn as
> a differential claim.** That is the third strike I have withdrawn this round.

Restated at `:456-458`; the section heading at `:316` states it against himself — *"THE
DIFFERENTIAL ATTACK DID NOT FIRE. THE INDEPENDENCE ATTACK DID."* Technically justified:
`nrmse` divides by `float(y.std(unbiased=False))` from that shared tensor
(`scale/negation_scope.py:1382-1387`), so a draw-level scale shift is common-mode and
cancels in the between-arm contrast. **The withdrawal is correct and correctly reasoned.**

**STRIKE I-5:** `V20_R15_JOURNAL.md` has **zero** hits for `common-mode`, `n_eff`,
`Bernoulli`, or `crossing rate`. `STRIKE 10` appears three times — `:1196` (the it.8
filing), `:1506` (it.8 DISTANCE, asserting the common-mode fact as *support* before MARS
tested it), and `:1554` (it.9, about the unrelated `frac_gate_annihilated` mislabel).
**Neither half of it.9 STRIKE 10 is journalled — not the withdrawal, and not the half that
survives.** Meanwhile `:1694` still publishes the crossing-rate `p`-value that the
surviving half forbids.

### 2.3 The clamp at 0 steps — **NOT REACHED.** See §7.

### 2.4 `softmax` unpaired on seeds 8-15 — **UPHELD**, with the search shown working

| arm | crossings | seeds actually run | artifact |
|---|---|---|---|
| `arm_pl` fresh | **7/8** | **8, 9, 10, 11, 12, 13, 14, 15** | `results/v20_r15_it8_armpl_b.jsonl` (02:58) |
| `softmax` control | **0/8** | **0, 1, 2, 3, 4, 5, 6, 7** | `results/v17k_r4_retake.jsonl` (prior day, 23:11) |

**Disjoint. Zero overlap.**

The negative is demonstrated, not asserted. `grep -c -i softmax` — **the zeros are the
evidence**: `v20_r15_it6_seeds8_15.jsonl` → **0**; `v20_r15_it8_armpl_seeds8_15.jsonl` →
**0** (and it is a 2-line stub, header + identity, zero cells — an aborted run);
`v20_r15_it8_armpl_b.jsonl` → **0**; `V20_R15_JOURNAL.md` → 30. All **167** files under
`results/` swept including `.pt` weights; every softmax weight file carries suffix
`_sd0_`…`_sd5_`, and filtering all softmax hits for seeds 8–15 returns **nothing
pre-dating the strike**.

Pooled `12/16 = 5/8 + 7/8`. **`softmax` never gets a second 8; it has 8 cells in total.**
The asymmetric denominator in the journal's own table at `:1446-1450` — `12/16`, `1/16`,
`0/8` — is the defect showing on the face of the record. The Fisher exact
`p = 6.730e-04` at `:1473-1475` is computed **across disjoint seed sets**.

**The journal convicts itself twice before publishing anyway** — `:1239-1240` (*"`arm_pl`
5/8 and `softmax` 0/8 are unchanged because nothing fresh was run for them"*) and `:1341`
(*"`softmax` is `0/8` on stale cells nobody refreshed"*) — then prints the contrast at
`:1430` and `:1450`.

**Remediation exists and post-dates the claim.**
`results/v20_r15_it10_mercury_rescore.jsonl`, created **03:25:54, four minutes after MARS
filed at 03:21:02**, header `"why": "re-score kept-nothing models on 3 eval draws; softmax
paired on 8..15"`. It carries `softmax` on seeds `0, 8-15` at `0.922–1.047`: **0/9
crossings across three eval draws, against `arm_pl` 8/9.** MARS's own concession therefore
stands (`V20_R15_IT9_MARS.md:308-310`): **the defect is in what the round is entitled to
write, not in which arm wins.** A file created after publication cannot retroactively pair
the published claim, and MERCURY moved on it inside four minutes — **which is the correct
response and should be recorded as one.**

---

## 3. PRIORITY 3 — D4, AND THE UNDERSTATEMENT BEHIND IT

**JUPITER's acceptance is complete and names the mechanism, not just the number**
(`V20_R15_IT9_JUPITER.md:99-102`):

> the flat band is now stated as `−0.0476 … −0.0010`, i.e. **up to 4.76 % decay per
> position**, not "under 3 %" — the understatement H3b struck in MARS's report and the
> journal is the same understatement that produced `−0.0436`, and it is corrected here in
> the same direction.

The journal corroborates at `:1564-1570`: a sweep of all **878** finite `lambda_hat_live`
values in `results/` returns **zero rounding to `−0.0436`**; the true bound is seed 4 at
`−0.047636087983846664` — a transposition of the last two digits. The row's own `n = 12`,
its own `nrmse 0.852–0.927`, **and the coordinator's published β `0.588`** all three
require seed 4, which the printed interval excludes. *"This office printed the number that
proves the error and did not notice."* And the certifying test asserted `abs(live) < 0.05`
— **wider than the prose it was cited for, so the green `[RUN]` certified nothing there.**
That is the most valuable sentence in the it.9 record.

### STRIKE I-6 — "under 3%" survives uncorrected in the journal, at **both** publication sites

- `V20_R15_JOURNAL.md:1193` — *"live-band decay under 3% per position"* (it.7 record)
- `V20_R15_JOURNAL.md:1294` — *"Combined with STRIKE 9's live-band decay **under 3% per position**"* (it.8 record)
- `V20_R15_IT7_MARS.md:196` — *"band decays by **under 3 % per position**"*, four lines below his own 3.20% row

`grep -c "4.76" V20_R15_JOURNAL.md` → **0**. `grep -n "per position" V20_R15_JOURNAL.md` →
**one hit, `:1294`, and it is the uncorrected one.**

**CORRECTION 9 repaired the arithmetic and left the sentence standing.** The endpoint moved
`−0.0436` → `−0.0476` in the journal; the phrase the author actually read, twice, did not.
The corrected reading `up to 4.76 % per position` exists **only** in JUPITER's iteration
report. Under this office's standing rule — the journal is the record, the conversation is
not — **B is a SILENT RESTATEMENT, not a retraction.**

---

## 4. PRIORITY 4 — THE LEAP LEDGER AND THE RULING

### 4.1 MARS's re-file is real — the F2 strike is discharged

L-M1..L-M5 are **physically present** at `V20_R15_LEAP_LEDGER.md:85`, `:86`, `:87`, `:88`,
`:89`. The heading at `:81` names its own prior failure rather than papering it: *"Filed at
it.7 in `V20_R15_IT7_MARS.md:210-219` and asserted appended there; **they were not**.
Strike F2 … is correct and is MARS's. This is the append."* **The strike that existed
because a report asserted a write that never happened is now closed by the write.**
(Row-greps differ by one between runs, 6 vs 7, on a prose line mentioning two rows; not
load-bearing — the five rows are in the file.)

### 4.2 The ruling as recorded is narrower than the version handed to this office

`V20_R15_JOURNAL.md:1591-1594`:

> **The Inspector's requested ruling, granted:** a LEAPABLE grade naming a **theorem**
> rather than a **FIELD** is **INADMISSIBLE.** *Naming a field and then saying what you
> want from it is fine; naming the want and calling it a field is not.* One VENUS row is
> the instance and is struck to be re-filed.

**FIELD there means an academic discipline, not a data column.** Corroborated by the
ledger's own column header `FIELD (LEAPABLE only)`, the VENUS row label at `:56` (*"the
field (never the theorem)"*), and this office's own list at
`V20_R15_IT567_INSPECTOR.md:474-478` — *"realization theory for LTI systems (L-1),
formalized linear algebra over Mathlib (L-4), transfer-operator / Koopman spectral theory
(L-5) … each led by a head noun with textbooks behind it."*

**This office grades on the rule as recorded**, and notes for the record that a
data-column reading would strike every row this office previously certified as passing.
The ruling on the log is the binding one.

### 4.3 Per-row verdict — **20 numbered rows, 13 admissible, 4 inadmissible**

Four rows are pure TERMINAL with no field (L-3, L-8, L-M2, L-M5); the ruling does not bind
them. Of the 17 carrying a LEAPABLE grade in whole or part:

| row | ln | why it fails |
|---|---|---|
| **VENUS** | `:55-56` | *"the extent of corner descent in multiplicative-gate landscapes"* — a quantity in this round's own system, then the missing theorem spelled out, then a conjecture. **Struck at it.5–7 and STILL UNREPAIRED** |
| **L-2** | `:23` | claims a LEAPABLE half — *"LEAPABLE by 0 GPU-s of bookkeeping"* — over a FIELD column reading **`none`** |
| **L-9** | `:68` | same pattern — *"LEAPABLE by ~275 GPU-s and one argparse line"*, FIELD column **`none`** |
| **L-14** | `:131` | *"LEAPABLE only on the chess witness"* — names **a bed**, not a field |

Admissible and well-formed: L-1, L-4, L-5, L-6, L-10, L-11, L-12, L-M1, L-M3, L-M4, L-13,
L-15. **JUPITER's L-13 and L-15 pass cleanly** — *calibration / probabilistic forecasting
(CRPS, pinball loss)* and *finite-mixture inference* are both disciplines with literatures.
His closing claim *"Every row names a FIELD, never a theorem"* (`:344`) is **true as to
theorems and false as to L-14**, which substitutes a bed for a field. **One row off, and it
is his own.**

### 4.4 STRIKE I-7 — the struck row is invisible to the ledger's own row-grep

The VENUS row at `:49-60` carries a LEAPABLE grade and **no `L-` number**. `grep "L-"`
cannot find the one row the ruling was issued about. **The ledger's index does not cover
its own defect.**

### 4.5 STRIKE I-8 — the L-12 amendment is an appended note, and it leaves the file self-contradictory

`:133-150` is a section appended *after* the L-13/L-14 table. **Line `:71` is untouched.**
The ledger now asserts both:

- `:71` — *"the five-cell CI is **post-hoc**, selected on a coordinate cut from these same eight cells"*
- `:146-148` — *"**The eight new cells were not available when the cut was made** … The selection is no longer post-hoc; it is a prediction that held."*

For the ruling this is harmless (L-12's field was already admissible), but the falsified
text stands. Worse, the same append states *"The banked record is 40 cells, not 34"*
(`:137`) while **L-9 at `:68` still reads `34 of 34`** and the file's own index at `:121`
reads **`40-of-40`**. **Three numbers for one corpus inside one document.**

### 4.6 STRIKE I-9 — MARS overclaims his own compliance check

`V20_R15_LEAP_LEDGER.md:105-108`: *"JUPITER's it.8 rows **L-9..L-12** were checked against
the same rule and **pass**"* — then names only L-10, L-11, L-12. **L-9 is precisely the one
of the four that fails**, on a `none` field. Four asserted, three checked. The row he did
not name is the row that breaks the claim.

### 4.7 The non-binary grades are spreading

it.5–7 flagged three hybrids for the coordinator (L-2, L-6, L-M4). There are now **seven**
— add L-9, L-13, L-14, L-15. **The contract's two-value set covers 13 of 20 rows.** The
"LEAPABLE half with FIELD `none`" pattern is what makes L-2, L-9 and L-14 inadmissible,
and it is the pattern that is growing.

---

## 5. PRIORITY 5 — AUDIT OF THE COORDINATOR

| # | published to the author | journal? | verdict |
|---|---|---|---|
| **A** | *"a pole precisely on the unit circle"* | **YES** — `V20_R15_JOURNAL.md:1380-1390`, **CORRECTION 8 — THE POLE ON THE UNIT CIRCLE WAS A CLAMP CEILING** | **RETRACTED-IN-JOURNAL. Clean.** |
| **B** | *"live-band decay under 3% per position"* | **PARTIAL** — endpoint corrected at `:1564-1566`; **the phrase never is.** `grep -c "4.76"` → **0** | **SILENT RESTATEMENT — §3** |
| **C** | *"12 of 16"* | **NO** — restated at `:1694` as surviving *"everything found this iteration"* | **NOT RETRACTED** |
| **D** | *"7 of 8 fresh against softmax 0 of 8"* | **NO** — stands at `:1430`, `:1450`, `:1473-1475` | **NOT RETRACTED** |
| **E** | `frac_gate_annihilated` mislabelled *"sign census"* | **YES, at it.9** — `:1554-1556` | **CORRECTED. Clean, and self-attributed.** |

**A and E are exemplary and this office says so plainly.** A names that the office itself
published the claim, says *"the test was run and it decided against the story"*, and
corrects an it.7 line cite `:109`→`:113` in the same breath. E reads: *"One label is wrong
and it is this office's: the column is `frac_gate_annihilated`, **not a sign field**; 'sign
census' is a mislabel this office propagated from it.1. The arithmetic and the count of 8
stand."* **Both are what a retraction looks like: in the journal, self-attributed, scoped
to the label without conceding the arithmetic.**

`frac_gate_annihilated` is not a sign field, and the it.9 correction is right to say so —
the three identities the round has been reading off it (`4123/8192 = 0.5032958984375`,
`8148/8192`, `8002/8192`) are exact to the bit and unaffected.

### 5.1 STRIKE I-10 — **MARS's it.9 report is not merged into the it.9 journal record**

**This is the headline of the audit.** The it.9 record (`:1518-1710`) names MARS five
times — `:1520` (room roster, *"the ledger rows he owed"*), `:1573`, `:1579`, `:1582`,
`:1585` — **all five about the it.7 F2 strike and the rows.** Not one of his four it.9
strikes appears.

Greps over `:1518-1710`: no `agg`, no `crosses: false`, no `0.2125`, no `0.6639380105668372`,
no `n_eff`, no `common-mode`, no `unpaired`, no seed-9 exclusion, no clamp margins.

The record even carries *"pre-clamp `u` at the 0-step twin is a pure read, answerable by a
test without touching the instrument"* (`:1665`) as an **open item to be priced** — while
MARS's report of the *same iteration* says it is **settled on all six cells**. **The
journal is stale against a report filed at 03:21:02, one second before the journal's own
mtime of 03:21:06.**

**And in the same record, at `:1694`:**

> `arm_pl` crosses `floor₁` on 12 of 16 against `softmax` 0/8 at `p = 6.730e-04`, and that
> separation survives everything found this iteration.

**It does not survive. It was struck in the same iteration by the office whose report the
record does not contain.** Three it.9 findings bear directly on that one sentence:

1. the seed-9 hinge — the `12` is `5/8 + 7/8` and the second half flips on one cell (§2.1);
2. the unpaired control — the `0/8` is on seeds 0–7 and the `7/8` on seeds 8–15 (§2.4);
3. `n_eff = 1` — sixteen correlated readings of **one** eval condition are not sixteen
   Bernoulli trials, so the Fisher exact is not licensed (§2.2).

**The coordinator's frame-correction in the same paragraph is honest and good** — `floor₁`
is a capability threshold and not an information floor, 6 of 34 cells violate it, the
honest distance is `0.204` above the calibrated oracle at best and `0.852` at the mode.
**That an office capable of that correction published `12 of 16` unqualified in the next
sentence makes the omission read as a merge that ran out of clock, not as concealment.**
It is still the defect that most needs fixing before it.10 closes.

---

## 6. LEDGER — `N audited, M struck`

**33 checks audited across 12 nurses. 10 struck.**

| # | strike | where |
|---|---|---|
| I-1 | Q6 planted negative misdescribes its own experiment (`torch.randn`, not the oracle); `√2` is `1.4060`; `W1 = 0` is a tautology true at every seed | §1.3 |
| I-2 | LEAP ledger `:131` hardens `≈ √2` to `= √2` | §1.4 |
| I-3 | Q6's entire grounding is journalled nowhere | §1.5 |
| I-4 | it.8 machine verdict `crosses: false` appears nowhere in the journal | §2.1 |
| I-5 | neither half of it.9 STRIKE 10 is journalled | §2.2 |
| I-6 | "under 3%" stands uncorrected at both author-facing sites, `:1193` and `:1294` | §3 |
| I-7 | the struck VENUS row has no `L-` number and is invisible to the ledger's grep | §4.4 |
| I-8 | the L-12 amendment leaves `:71` falsified and the corpus at 34/40/40 in one file | §4.5 |
| I-9 | MARS asserts four L-rows checked, names three, and the unnamed one is the failure | §4.6 |
| **I-10** | **MARS's four it.9 strikes are absent from the it.9 journal record, which republishes `12 of 16` as surviving them** | §5.1 |

**Upheld without strike:** MARS's four strikes (all four), JUPITER's D4 acceptance and its
corrected endpoints, MARS's L-M1..L-M5 re-file, coordinator retractions A and E,
JUPITER's L-13/L-15 rows, JUPITER's §3.5 refusal to cite M13's un-produced figures, and
MERCURY's four-minute remediation of the unpaired control.

---

## 7. WHAT THIS AUDIT DID NOT REACH

Named honestly, because the it.5–7 report proved this is the most useful paragraph.

1. **The Lean half of Q1 — the largest gap.** `lake build` exit status, whether
   `pathProd_polar` and `pathProd_eq_zero_iff` sit at `:105` and `:129` exactly, whether
   either is `sorry`-backed, and whether the `1.387779e-16` / `1.110223e-16` instances are
   journalled or only asserted. **The nurse was still in `lake build` at the wall.** Q1's
   F0 is therefore **unverified in this audit, not cleared.** The contract's Q1 bar —
   *"attempted in Lean or ≤1e-6 instance"* — turns on it.
2. **The clamp re-derivation.** MARS's six zero-step cells, `a_hat_max_0step == 1.0`
   reported, pre-clamp `u` strictly exceeding 1.0 with margins `0.0129–0.1843`. **Zero of
   the two required re-derivations completed.** The mechanism is corroborated
   independently — the journal's own CORRECTION 8 locates the clamp at
   `ceq/arm_smprime.py:113` and says `a_hat_max` *"cannot exceed 1.0 by construction"* —
   but **the six margins are unaudited and this office does not endorse them.**
3. **Q2, Q3, Q4, Q5 — eight of the twelve cells.** RED-first status and gap-citation
   existence unverified. The `+6` verdict above rests on Q6 alone; **if Q2–Q5 hold, the
   table still fails on Q6, and if Q6 is repaired, Q2–Q5 must still be audited before the
   `+6` is released.** This office is ruling one cell short of a full table and says so.
4. **The independent 40-banked-cell key census.** JUPITER's `0 of 40` was not re-run
   end-to-end here; §1.8 argues the grade from the report's own §8 concession instead.
5. **The it.8 reports as such.** `V20_R15_IT8_MERCURY.md` and `V20_R15_IT8_JUPITER.md`
   were read only where it.9 cited them. **The it.8 audit is partial.**

---

## 8. TREE STATEMENT

**No mutations by this office or its nurses.** Nurses were confined to the scratchpad;
two ran python and one ran pytest, all read-only against the tree.

Digests of every audited file, taken at the start of the audit and re-taken at the end,
are **unchanged**: `V20_R15_JOURNAL.md`, `V20_R15_LEAP_LEDGER.md`, `V20_R15_IT9_MARS.md`,
`V20_R15_IT9_JUPITER.md`, `V20_R15_IT8_MERCURY.md`, `V20_R15_IT8_JUPITER.md`,
`V20_R15_IT9_SATURN.md`, `scripts/v15_r1.py`, `ceq/arm_smprime.py`,
`scale/negation_scope.py`.

`git status --porcelain` reports **3 modified** (`house-events.jsonl`, `pytest.ini`,
`scale/ledger.py` — **all three already dirty on entry**, from SATURN's it.9 ledger repair;
the env snapshot's "clean" was stale) and the untracked V20 R15 working set.

**Concurrent it.10 paths, reported honestly and not attributable to this audit:**

- `results/v20_r15_it10_mercury_rescore.jsonl` — **MERCURY, created 03:25:54 and still
  growing during the audit** (16 → 59 lines observed). This file is the cause of the
  `1 failed` in §1.6.
- `scripts/v20_r15_it10_mercury_rescore.py`
- `tests/mercury/test_v20_r15_it10_eval_independence.py`
- `tests/venus/` — VENUS's re-forecast working set

**This office wrote exactly one file: `V20_R15_IT89_INSPECTOR.md`.**

### 8.1 CORRECTION TO §8, FILED AT THE WALL — **TWO AUDITED FILES MOVED UNDER THE AUDIT**

The closing digest re-take does **not** match the opening one on two files. **Neither
change is this office's**, and both are reported rather than hidden.

| file | open | close | what happened |
|---|---|---|---|
| `V20_R15_JOURNAL.md` | `475ae363…` | `468a2854…` | **append-only growth, 1710 → 1852 lines** — the it.10 record landed mid-audit |
| `V20_R15_LEAP_LEDGER.md` | `88d3e9a1…` | `46c280f1…` | **197 lines before and after — an IN-PLACE EDIT**, not an append |

The other eight audited files are **byte-identical open to close**: `V20_R15_IT9_MARS.md`,
`V20_R15_IT9_JUPITER.md`, `V20_R15_IT8_MERCURY.md`, `V20_R15_IT8_JUPITER.md`,
`V20_R15_IT9_SATURN.md`, `scripts/v15_r1.py`, `ceq/arm_smprime.py`,
`scale/negation_scope.py`.

**Line cites verified to still resolve after the move:** `V20_R15_JOURNAL.md:1694` still
carries the `12 of 16 … softmax 0/8 … p = 6.730e-04` sentence, and
`V20_R15_LEAP_LEDGER.md:131` still opens L-14. **Every line number in this report was
re-checked against the closing state or is confirmed by append-only growth below it.**

**Two consequences, stated plainly.** First, `V20_R15_IT10_MERCURY.md` and
`V20_R15_IT10_VENUS.md` now exist — **it.10 opened while this audit ran**, which is why the
timer rolled to iteration 16 at the close. Second, and more important: **the leap ledger was
edited in place at constant line count during an audit of the leap ledger.** §4's per-row
verdicts are as-of the opening digest `88d3e9a1…`. **A ledger the round has already struck
once for an asserted-but-unwritten append is now being edited in place under a live audit,
and this office cannot say from a digest alone which row moved.** The next Inspector should
diff `88d3e9a1…` against `46c280f1…` before trusting §4.3.

---

## 9. THE ANSWER TO THE QUESTION THAT WAS ASKED

**Is the theory table BOUND, so that the `+6` is earned?**

# NO. The `+6` is not earned, and the withholding was correct.

Twelve cells are graded and the grades are, on this office's reading, honest ones. But the
contract pays for **"every exit F-graded with a gap"**, and **Q6's gap cites an experiment
that is not the experiment the report describes** — the oracle is `torch.randn`, the `√2`
is `1.4060`, and the `W1 = 0.0` is an identity that holds at every seed. The one artifact
standing between Q6's F4 and "we did not try" is the one that does not survive re-running,
**and the LEAP ledger hardens its hedge on the way to it.35.** Four of twelve cells were
not reached at all, Q1's Lean half among them.

**The repair is small and the conclusion holds.** Re-point §3.4 at `equilibrium_oracle` —
`W1 = 0.0` exact, `NRMSE = 1.421901019003236`, both re-run and reported at §1.3 — restore
the `≈` at `V20_R15_LEAP_LEDGER.md:131`, journal the census, and audit Q2–Q5. **Then the
`+6` is earned on the merits it already has.**

Separately and more urgently: **`12 of 16` and `7 of 8 against softmax 0 of 8` are still
standing in the journal, unretracted, in a record that omits the four strikes that broke
them.** That is a larger debt than the `+6`.

---

# 10. LATE RETURNS — FILED INSIDE THE WALL, SUPERSEDING §7

Four nurses landed after the body was drafted. **They close two of the five gaps §7 named,
and they make the verdict stronger, not weaker.** The final count is **47 audited, 14
struck**.

## 10.1 Q1 — the Lean half is CLEAN. §7 item 1 is CLOSED.

- **Both cited line numbers are exact, off-by-zero:** `lemma pathProd_polar` at
  `lean/CEQ/V16Domain.lean:105`, `theorem pathProd_eq_zero_iff` at `:129`. Both are full
  proofs terminating in `rw`/`exact`/`constructor`.
- **`lake build` → exit 0.** Output was empty (fully cached), which proves nothing, so the
  nurse forced re-elaboration: `lake env lean CEQ/V16Domain.lean` → **exit 0, 61 lines,
  zero errors, zero warnings.** Toolchain `leanprover/lean4:v4.7.0`, `lakefile.lean`
  present.
- **Zero `sorry`, zero `admit`, zero `axiom` declarations.** All 11 grep hits are
  doc-comment prose; six of them are the files *asserting* `No \`sorry\`.` Machine-confirmed
  by the file's own `#print axioms` at `:532-534`: all three `pathProd` theorems depend on
  `[propext, Classical.choice, Quot.sound]` — **Mathlib's standard three, no `sorryAx`.**
- **Both instances reproduce bit-for-bit.** `1.387779e-16` (Q1/W1) and `1.110223e-16`
  (Q1/W3) are journalled as a RED→GREEN pair at `house-events.jsonl:12117-12118` and
  re-ran identically. The RED at `:12117` is a genuine probe:
  `assert diff <= 0.0 probe; 1.3877787807814457e-16 <= 0.0 fails`.
- **Q1/W3's "no theorem for the positive class" is self-declared, not uncovered**
  (`V20_R15_IT6_JUPITER.md:309`), and F1-not-F0 is the honest reading of "instance is not
  theorem".

**Q1/W1 and Q1/W3: BOUND.** The contract's Q1 bar — *"attempted in Lean or ≤1e-6 instance"* —
**is met twice over on both wings.**

### STRIKE I-11 — the instances are printed, never asserted

`tests/jupiter/test_v20_r15_it6_q1_exact_class.py:42` sets `TOL = 1e-6`; the only assertions
are `assert diff <= TOL` at `:70` and `:136`, and the exact digits surface via `print` at
`:69` and `:135`. **A regression degrading the residual by seven orders of magnitude still
passes green.** The instances are backed but **not pinned** — and `house-events.jsonl` is
their sole persisted record; there is no `results/*.jsonl` for it.6 Q1.

*Provenance hazard for future offices:* `1.110223e-16` is 0.5·machine-epsilon and appears in
30 files, including a **different** arm-distinctness measurement at `V20_R15_JOURNAL.md:312`.
`1.387779e-16` has two occurrences repo-wide and is the safer anchor of the pair.

## 10.2 STRIKE I-12 — **Q2 IS UNBOUND ON BOTH WINGS.** §7 item 3 partly closed, and it went badly.

This is the second cell to fail, and it fails harder than Q6.

- **The F1 constants are real and attached to neither wing.**
  `tests/jupiter/test_v20_r15_it6_q2_outside_bound.py:43` imports **only `ceq.hankel`** —
  never `ceq.arm_smprime`, never `ceq.arm_pl`, and it reads **no banked cell.** The
  constants reproduce exactly (`R2_1 = 0.05`, `err_1 = 0.9746794345`, `rank_real = 1`
  against the annex's hinted 2), but `err_1` is measured on a `d=20` delay Hankel block —
  **a BED-K object** — while both frozen wings are **BED-M**. `ceq/hankel.py` is imported
  by tests only; **no runner path touches it.**
- **The office itself later ruled this instrument inadmissible here.**
  `V20_R15_IT8_JUPITER.md:474` — M16 is **F4 for these wings**, *"BED-K only; both frozen
  wings are BED-M … do not cite it on W1/W3 at all"* — repeated at
  `V20_R15_JOURNAL.md:1646`. **The Q2 row was never revisited against his own ruling.**
- **Q2 is journalled as a pointer, not as evidence.** `V20_R15_JOURNAL.md:1123-1124` reads
  `| **Q2 / W1** | **F1 with the constant** | §8 |` — **the journal reproduces a section
  reference from a report as the gap.** Grep of the journal returns **0** for `0.9746`,
  `R2_1`, `err_1`, `NEG_ENTRY`, `rank_+`; `Hankel` appears once, at `:1646`, only to say the
  instrument does not bind these wings.
- **Q2/W3 has no cell-level gap stated anywhere.** The row points at §8; §8 states a delta
  for W1 only.
- **Q2 is the only question with no LEAP ledger row on either wing** — Q1→L-1/L-2,
  Q3→L-5/L-6, Q4→L-9/L-10, Q5→L-13/L-14, Q6→L-13/L-14, and **Q2 absent.**

**A gap that is a section reference is not a gap. Q2/W1 and Q2/W3 are UNBOUND.**

## 10.3 STRIKE I-13 — Q1/W1's gap is literally true and its conclusion is false

The F0's HOW-BAD gap says `0 of 24` banked cells attribute the effect, *"Not because they
did not; because the record cannot say"* (`V20_R15_IT6_JUPITER.md:96-97`).

**The record can say.** `results/v17k_r4_retake.jsonl` holds exactly 24 `t="cell"` records
and 0 of 24 carry **top-level** `beta`/`qk`/`route`/`g` — but **all 8 `arm_smprime` cells
carry all four inside `manifest.smp_values`.** Seed 0: `route='product'`,
`beta=0.7325604557991028`, `qk=1.276558756828308`, `g=1.319505214691162`.

**Three offices had already read that field.** `V20_R15_IT1_MARS.md:364` states it;
**JUPITER himself reads `manifest.smp_values.qk` and `.beta` one iteration later** at
`V20_R15_IT7_JUPITER.md:91-92`. The error propagates into `V20_R15_LEAP_LEDGER.md:23`
(L-2 — the row whose FIELD column reads `none` on the strength of it) and
`V20_R15_IT9_SATURN.md:121`. Cells carry **60** keys, not the claimed 58.

**The grade survives** — the census is about top-level columns and the F0 rests on the Lean
certificate, not the census — **but the sentence "the record cannot say" is false, and it
is load-bearing for L-2's inadmissibility.**

## 10.4 Q3 — **BOUND on both wings**, and Q3/W1 is the round's best-argued cell

- **Q3/W1 F2 rests on a falsifier that came out the other way.** The office's own draft
  bucketed seed 3 into the flat band (`0.49 < frac < 0.51` swallowing both
  `0.4967041015625` and `0.5032958984375`), went RED at `assert (12 == 13)`
  (`V20_R15_IT7_JUPITER.md:262-274`), **and the corrected node graded F2 instead of F1.**
  A test that changed the grade against the office's own draft is the strongest form of
  red-first in this round.
- Every figure reproduces on 16 deduped cells: seed 3 `lambda_hat_live = -0.4411417841911316`,
  `Spearman(beta, nrmse) = -0.0324`, `Spearman(qk, nrmse) = +0.7176`, crossing cell seed 2 at
  `beta = 1.3439332246780396`, qk partition with **zero overlap 16/16**. The counterexample
  is **asserted**, not narrated (`tests/jupiter/test_v20_r15_it7_q3.py:103`).
- **Q3/W3 BOUND**, planted negative `pl_sign` takes 2 nodes RED, margin
  `0.4512110278480903` exact against the file. **Caveat:** the journal has already
  superseded it at `:1438` — *"margin `0.432760655774893` (was `0.451211` over 8)"* — and
  **the frozen it.9 table and `V20_R15_LEAP_LEDGER.md:27` still carry the n=8 figure.**

### STRIKE I-14 — the loose test that let D4 through is still in the tree

`tests/jupiter/test_v20_r15_it7_q3.py:96`: `assert all(abs(live[s]) < 0.05 for s in flat)` —
**looser than the `[−0.0436, −0.0009]` interval it is cited for.** This is the exact
mechanism CORRECTION 9 named (*"a test looser than the claim it is cited for"*), and **it is
unrepaired.** The same class recurs at `test_v20_r15_it9_q6.py`'s `assert n == 878`
(§1.6) and at `test_v20_r15_it6_q1_exact_class.py`'s printed-not-asserted instances
(§10.1). **Three instances, one mechanism: the certificate is wider than the sentence it
certifies.**

## 10.5 The clamp — **MARS UPHELD ON ALL SIX, BITWISE.** §7 item 2 is CLOSED.

The nurse did **not** reuse MARS's test. He drove the shipped instrument
(`V.make_arm`, `V.gate_columns`, `M3_TASKS["e3_t2"]`) through the runner's own 0-step path
(`scripts/v15_r1.py:801-809`) on CPU with his own arithmetic.

| seed | independent pre-clamp `u` | MARS | agree |
|---|---|---|---|
| 2 | `1.0686708688735962` | `1.0686708688735962` | **bitwise** |
| 3 | `1.0263001918792725` | `1.0263001918792725` | **bitwise** |
| 4 | `1.046579360961914` | `1.0465793609619140` | **bitwise** |
| 7 | `1.1842671632766724` | `1.1842671632766724` | **bitwise** |
| 12 | `1.012909173965454` | `1.0129091739654540` | **bitwise** |
| 14 | `1.0333305597305298` | `1.0333305597305298` | **bitwise** |

**6 of 6 strictly exceed 1.0.** The algebra holds from the actual code:
`torch.lerp(ones_like(u), u, g) = 1 + g·(u−1)`, and at `float(model.g) == 1.0` exactly
(`ceq/arm_smprime.py:529`) the difference from `u` is **bitwise 0.0** — the shipped
`torch.lerp` is exact where the hand-expanded form is not. `magnitude()` at `:113` receives
`u` unmodified. `scripts/v15_r1.py:363` splats `model.heads(x)` straight into `blend` and
never binds it, so **MARS's "no key pins pre-clamp `u`" is true across all 60 keys** —
`a_hat_max` and `manifest.smp_values.m_max` are both `magnitude()`'s *output*.

The six cells with `a_hat_max_0step == 1.0` are **exactly [2,3,4,7,12,14]**, confirmed
against the files. **`a_hat_max == 1.0` is saturation, not coincidence, and it is now
measured twice by two offices.**

**The cross-check MARS did not run strengthens him.** The CPU replay reproduces the
journalled **cuda** `*_0step` columns bitwise on all six seeds — `a_hat_max_0step`,
`a_hat_min_0step`, `frac_gate_annihilated_0step`, `unit_root_0step`,
`dyn_range_bound_0step`, `z_winding_max_0step`. Only mean-reductions drift, by ~1e-7 — the
**exact CUDA-vs-CPU floor MARS cited**, and the margins sit 10⁵–10⁶ above it.

Two nuances, neither fatal: **(a)** the margins are **lower bounds** — `argmax` on the
post-clamp tensor returns the *first* index tied at 1.0, so the true maximum overrun is
larger on 5 of 6 seeds (seed 2: `1.2068` over the live band, 29 of 8192 positions above
1.0). MARS's column heading *"at argmax"* is accurate, and the binding margin `0.0129`
is unaffected. **(b)** §6.2's GPU-second sums (`275.551`, `209.144`) are
**duplicate-copy-dependent**: seeds 0 and 1 sit in both journals with identical physics and
different wall-clock `secs`; summing the other copy gives `275.686` and `209.279`. Not a
defect — **but the numbers are not reproducible without naming which copy, and §6.2 does
not name it.**

## 10.6 Q6's census — 40 is exactly right, and the phrase is still the weak half

**FACT 3 verified exactly.** 43 raw `t="cell"` records across three files; deduping on
`(kind, seed, round(eval_nrmse,6))` removes exactly three replays → **40** (`arm_pl` 16,
`arm_smprime` 16, `softmax` 8), one `instrument_hash 5d41a63d576717` on 40/40. **Not 38,
not 44.** 60 keys in the union; the only non-scalar is `manifest`, a dict of provenance
hashes. The regex returns 3 hits — `dist_to_floor`, `dist_to_skyline`, `dist_to_skyline_why`
— **all false positives on `dist` = distance**, and JUPITER's own test already carves them
out at `test_v20_r15_it9_q6.py:143`. **Semantic hits: 0 of 40.** The 688 `trace` records
were swept too (12 keys, 0 list/dict-valued) — a sweep FACT 3 does not claim and passes.

**STRIKE I-15 — FACT 1 makes a both-wings claim on a one-wing citation.**
`ceq/arm_smprime.py:572-577` is the **only** `def forward` in that file. W3's forward is at
**`ceq/arm_pl.py:405-410`, uncited.** Both do return `[n]` — executed, `(8,)` `ndim 1`
`float32`, `readout out_features = 1`, no `logits`/`prob` parameter on either — so **the
census survives; the citation does not cover it.** FACT 2's "float32" is also inherited
(`dtype=x.dtype`), a property of the batch maker, not of `equilibrium_oracle`.

**The sharper statement of the grade, which this office recommends JUPITER adopt:**

> **Per-draw, `W1` degenerates to `L1`** — `readout.out_features = 1`, so one real per draw
> is a Dirac and `W1(δ_a, δ_b) = |a−b|`, carrying zero distributional information.
> Unrepairable without a new head. **Pooled, `W1` is not a score** — a permutation defeats
> it, and the pairing is the entire content of a regression bed.

**Both halves are structural. FACT 3 is neither** — "we did not journal `pe` and `y_ev`" is
a **record gap**, structurally identical to Q1/W1's `0 of 24`, which this very ledger grades
as *LEAPABLE by bookkeeping* (L-2). **Standing FACT 3 beside two structural facts inflates a
record gap into a domain fact — the precise error the round grades as F-gap-on-the-record
everywhere else.** And "no one built the estimator" is false: JUPITER built it, in one line,
at `test_v20_r15_it9_q6.py:160`. **The grade survives the demotion. The phrase "has no
object" does not.**

## 10.7 D4 — endpoints EXACT, and "under 3%" stands in **four** places, not two

**The corrected endpoints are verified against raw data, not against the report.** Over all
`t="cell"`, `kind="arm_smprime"` records with `frac_gate_annihilated == 0.5032958984375`:
**n = 12, seeds exactly `[0,1,4,5,6,7,8,9,10,12,14,15]`**, min = `−0.047636087983846664`
(seed 4) → **`−0.0476`**, max = `−0.0009529261151328683` (seed 6) → **`−0.0010`**,
`eval_nrmse` span `0.852061 … 0.926717` → **`0.852–0.927`**. `0.0436` has **zero occurrences
in any data file** — prose only. **JUPITER's corrected row is exact in every column.**

**And "under 3%" is false twice over, not once:** seed 4 is 4.7636%, **and seed 8 is
3.1969%** — the claim fails even with seed 4 removed.

**§3's strike widens.** The uncorrected sites are **four**, not three:

| # | file:line | why it stands |
|---|---|---|
| 1 | `V20_R15_JOURNAL.md:1193` | STRIKE 9, it.7. The it.9 entry reaffirms STRIKE 9 at `:1548-1553` and never touches the percentage |
| 2 | `V20_R15_JOURNAL.md:1294` | **worse than untouched** — CORRECTION 8 retracts only the *pole* half of this sentence, then affirmatively re-endorses the other half at `:1405-1407`: *"The author's wave/decay pointer is not refuted by this … `lambda_hat_live`'s relaxation window survives untouched"* |
| 3 | `V20_R15_JOURNAL.md:1345` | it.7 DISTANCE close — *"sub-3% decay"*, a phrasing variant no `under 3` grep catches |
| 4 | `V20_R15_IT7_MARS.md:196` | **MARS never corrected his own line.** His it.9 report accepts F2 and withdraws STRIKE 10, and never mentions H3b, "under 3%", or 4.76% |

Also standing: `V20_R15_IT7_JUPITER.md:71` still prints the original `−0.0009 … −0.0436` row.

**The mechanism, for the taxonomy — and it is the round's most reusable finding:**
**the correction was routed to the *datum* and not to the *claim derived from the datum*.**
`−0.0436 → −0.0476` lives in one file and one test; `under 3%` lives in three journal lines
and one report and went to the author twice. **A datum-level fix in a different document
leaves the derived claim standing — and CORRECTION 8 actively re-endorsed it while
retracting its neighbour.** The it.9 strike-count line at `:1540` (*"39 AUDITED, 15 STRUCK"*)
itemises STRIKE 9's scope, STRIKE 10's identities, and the sign-census mislabel — **and
skips H3, the one strike aimed at the journal's own author-facing prose.**

## 10.8 The coordinator — the pattern, stated

The retraction yardstick is the journal's own precedent: a named `### CORRECTION n —` block
that quotes the original and says it dies (`CORRECTION 5` at `:761`, `CORRECTION 8` at
`:1380`). The journal declares itself the record at `:3-4` — *"One entry per iteration,
appended, never rewritten."*

**A and E clear that bar. B, C and D do not.** And the three that do not are **exactly the
three that would subtract from the it.8/it.9 DISTANCE lines** — the author-facing headline.
All three were struck in reports filed at it.9; the it.9 journal entry was written **after
all of them** (`03:21:06` against `03:21:02` and `03:21:05`) and closes with *"that
separation survives everything found this iteration."*

**This office does not read that as concealment** — the same entry contains CORRECTION 9's
*"This office printed the number that proves the error and did not notice"*, which no
concealing office writes. **It reads as a merge that ran out of clock, and the fix is
mechanical: fold `V20_R15_IT9_MARS.md` into the it.9 entry and re-derive the DISTANCE line.**

### On (E), one correction to the correction

The it.9 note says "sign census" was *"propagated from it.1"*. **The journal's own it.1
entry does not use the phrase** — `:177-181` says *"IS THE CORPUS, NOT THE ARM"* and *"a
corpus sign count"*. **The exact phrase enters `V20_R15_JOURNAL.md` at `:1199`, at it.7**,
and it **still stands there**, 357 lines above its own correction with no forward pointer,
because the journal is append-only. The correction is also scoped to STRIKE 10 and leaves
**STRIKE 1's identically-mislabelled title** standing at `V20_R15_IT1_MARS.md:19` — where
`V20_R15_IT1_INSPECTOR.md:145` graded it **"clean"**. *This office graded the mislabel clean
at it.1 and records that here.*

**What the column actually is**, for the record: `scripts/v15_r1.py:386`,
`frac_gate_annihilated=float((~fin).double().mean())` with `fin = torch.isfinite(lg)` at
`:380` and `lg = torch.log(m)` at `:372` — **the fraction of live-band positions where the
gate magnitude `m` is exactly `0.0`.** Denominator `4096 × 2 = 8192`. It reads no sign of
anything. On `softmax` it is not measured at all — hardcoded `0.0` at `:377`. It looked like
a census because the eval draw is pinned at `seed=12345` (§2.2), so 16 cells across three
runs republish one draw's property. **A per-cell gate-annihilation fraction, pinned by a
fixed eval draw, was published as a sign count of the corpus.**

---

# 11. REVISED VERDICT — UNCHANGED IN DIRECTION, STRONGER IN GROUND

**Cells now adjudicated: 10 of 12.**

| cell | verdict |
|---|---|
| Q1/W1, Q1/W3 | **BOUND** — Lean exit 0, lines exact, no `sorry`, instances reproduce |
| **Q2/W1, Q2/W3** | **UNBOUND** — constants on a bed neither wing occupies, and the office ruled that instrument inadmissible here at it.8 |
| Q3/W1, Q3/W3 | **BOUND** — the best red-first work in the round |
| Q4/W1, Q4/W3, Q5/W1, Q5/W3 | **NOT REACHED** — the one remaining gap |
| **Q6/W1, Q6/W3** | **UNBOUND** — the gap's demonstration is not the experiment described |

# THE `+6` IS NOT EARNED. FOUR OF TWELVE CELLS FAIL OR ARE UNREACHED, AND TWO OF THEM FAIL.

**Q2 is the worse failure and it was not the one this office was sent to find.** Q6's
argument is right and its citation is broken — a repair of minutes. **Q2's constant is
measured on a `d=20` delay Hankel block that neither frozen wing occupies, its journal entry
is a section reference standing in for evidence, and JUPITER's own it.8 ruling forbids
citing that instrument on W1 or W3.** That is not a citation defect. That is a cell graded
against the wrong bed.

**What the round should do before claiming the `+6`:** re-point Q6 §3.4 at
`equilibrium_oracle` (numbers in §1.3, they hold); restore the `≈` at
`V20_R15_LEAP_LEDGER.md:131`; **re-grade Q2 on BED-M or mark it ungraded**; cite
`ceq/arm_pl.py:405-410` for the both-wings claim; and audit Q4/Q5. **The table is close.
It is not closed.**

---

# 12. FINAL RETURN — Q4 AND Q5. §7 IS NOW FULLY CLOSED. **ALL TWELVE CELLS ADJUDICATED.**

| cell | grade | verdict |
|---|---|---|
| **Q4/W1** | F3 | **BOUND.** Six seeded planted negatives (`JUP_IT8_MUTATE`); `seq_len` and `cost_point` both fire on this cell. The core claim is a falsifiable **negative census** — `s = 64` on 55/55 records, reproduced cold. 13 of 13 cites exact **except one**: `ceq/sizing.py:169` is cited for `raise ValueError(arm)`, which sits at **`:168`.** The Θ(n·S²·d) *form* is code-reading only — and by the report's own argument cannot be measured, which is the honest position |
| **Q4/W3** | F3 | **WEAK, NOT STRUCK.** One planted negative (`pl_dense`) certifies `frac == 0.0` on 8/8; `ceq/arm_pl.py:316-327` is the `for i × for j × for size × combinations` nest as cited. But **the `1.0589×` ratio is stale — at 16 cells it is `1.0402×`, never updated** — and `1.0589` / `brute_force_path_sums` / `2^S` return **0 hits in the journal** |
| **Q5/W1** | F1+const | **BOUND — and it is the round's cleanest prediction.** `floor_formula` takes 2 nodes RED; the prediction `+0.20…+0.27` is stated **before** the observation `+0.145…+0.220` (`V20_R15_IT8_JUPITER.md:349-353`). Every cite exact; the `h_hat` identity holds to **0.0**, not 1e-12. The floor is violated by cells, **which a lower bound never is** — the finding that produced CORRECTION 11 |
| **Q5/W3** | F1+const | **BOUND — genuinely out-of-sample.** A RED caught a `2.0e-5` error **in this very constant**. The it.8 predictor `{2,3,7,9}` was checked at it.9 against eight cells that did not exist when it was made: **16/16**, reproduced. mean `0.6454112028697345`, sd `0.010352366815678714`, ci_hi `0.6582633033288883`, margin `0.0488434` — all digit-exact |

**Q6's census independently re-confirmed a third time, with a stronger test than the report's
own:** `isinstance(v, list)` flags any vector regardless of field name. **40 cells**
(`arm_pl 16, arm_smprime 16, softmax 8`), one `instrument_hash`, `s=64`/`t_star=2`/`steps=150`
on 40/40, and value types `320 str / 432 int / 1240 float / 112 bool / 120 None / 40 dict` —
**the only non-scalar is `manifest`, and it holds no sample.** The nested `manifest` was
swept separately: clean. **The claim survives the post-report it.10 file**, whose three
`eval_draw` records are still scalars.

Spot-checks all reproduced from `results/`: seed 4 `−0.047636087983846664`; **0** values round
to `−0.0436`; `878` finite `lambda_hat_live` exactly, once the post-report it.10 file is
excluded (**932** with it, confirming §1.6 independently); flat band 12 seeds;
`floor_1 = 0.7071067811865476 == sqrt(1/2)` on 40/40; `house-events.jsonl` unparseable lines
**4**, at **1899, 2937, 2938, 5871**, all `Invalid \escape` — SATURN's it.9 finding stands.

### STRIKE I-16 — the wing-manifest cite is off by two, twice in one report

`V20_R15_IT9_JUPITER.md:3` and `:32` cite `V20_R15_WING_MANIFEST.md:20-21` and `:20` for the
frozen wings. **Line 20 is the table header and 21 the `|---|` separator; the W1 and W3 rows
are at `:22` and `:23`.** Same class as the `ceq/sizing.py:169`→`:168` slip and the it.7
`:109`→`:113` slip CORRECTION 8 already fixed. **Third instance of an off-by-N cite this
round.**

### STRIKE I-17 — §3.5's "no file" is now false, by JUPITER's own hand

§3.5 claims a keyword sweep returns *"**no file**"* for `wasserstein`. It returns **one**:
`tests/jupiter/test_v20_r15_it9_q6.py:136`, a string literal inside **his own it.9 test**.
The substance holds — a literal is not a producer, exactly the class WILSON flagged at it.1
for `cantelli`/`azuma` — **but the absolute phrasing does not, and the file that falsifies it
shipped in the same iteration.**

### STRIKE I-18 — the D4 mechanism recurs *inside the correction that named it*

The it.9 journal entry says **"s = 64 on 34 of 34"** and **"violated by 6 of 34 cells"** — in
the same entry whose own report establishes the record is **40**. At 40 the violation count
is **13**. `V20_R15_JOURNAL.md:1129` still says "0 of 24"; `V20_R15_LEAP_LEDGER.md` carries
**34 of 34** at `:68` and **0 of 40** at `:130`. **A report reading 34 when the tree holds 40
is precisely the D4 mechanism, recurring in the correction that diagnosed it.** The
conclusions survive at the larger N — `s = 64` holds 40/40 and no cell journals
`beta`/`qk`/`route`/`g` at top level — **the numbers are stale, the findings are not.**

## 12.1 THE SYSTEMATIC FINDING OF THIS AUDIT

**The journal is where the evidence stops.** Q4/W1 and the Q5 *frame* are journalled in
detail. **Q4/W3's ratio, both Q5 graded constants, and the entirety of Q6 are not.**
`Wasserstein` appears **zero** times in `V20_R15_JOURNAL.md`; Q6 appears three times — the
room roster (`:1520`), the table cell (`:1533`), one sentence of verdict (`:1535-1537`).
**Four grounding claims that took a 9,582-byte test file to establish are journalled as
"the metric has no object on either."**

The same shape produced every other defect this office found: MARS's four strikes absent
(§5.1), STRIKE 10's withdrawal absent (§2.2), `crosses: false` absent (§2.1), "under 3%"
uncorrected (§10.7), Q2 journalled as a section reference (§10.2). **One mechanism, seven
sites: work is done well in reports and does not reach the record.** The offices are not
the problem. **The merge is.**

## 12.2 CLOSING NUANCE ON Q6, WHICH THE COORDINATOR SHOULD READ

Two independent nurses reached the same split. **Domain-emptiness rests on a census;
terminality rests on a theorem** — and §3.4's permutation argument is *"a mathematical
property of marginal-W1 demonstrated on synthetic gaussians, not a measurement on the bed."*
**The report presents them as one claim.** Separately: the shape census is **the only Q6
assertion with no planted negative** — nothing mutates an arm to return a 2-D output and
takes the node RED.

**Both facts hold. They are different kinds of claim, and the F4 should say which is doing
which work.** With §1.3's citation repaired and that split written down, **Q6 binds.**

---

# 13. THE FINAL COUNT

**47 checks audited. 18 struck.** Twelve of twelve cells adjudicated.

| BOUND (8) | UNBOUND (4) |
|---|---|
| Q1/W1, Q1/W3 — Lean exit 0, no `sorry`, instances reproduce | **Q2/W1, Q2/W3** — graded against a bed neither wing occupies |
| Q3/W1, Q3/W3 — best red-first work in the round | **Q6/W1, Q6/W3** — the gap's demonstration is not the experiment described |
| Q4/W1, Q4/W3 — Q4/W3 weak on a stale ratio, not struck | |
| Q5/W1, Q5/W3 — Q5/W3 genuinely out-of-sample, 16/16 | |

# THE ANSWER: **NO. THE THEORY TABLE IS NOT BOUND. THE `+6` IS NOT EARNED.**

**Eight of twelve cells bind, and several bind well.** Q3/W1's falsifier changed the grade
against its own author's draft; Q5/W1 predicted `+0.20…+0.27` before observing
`+0.145…+0.220`; Q5/W3's it.8 predictor held 16/16 on cells that did not exist when it was
made; Q1 is machine-checked to Mathlib's three standard axioms. **That is real work and this
office says so.**

**But the contract pays for *every* exit F-graded with a gap, and four exits do not have
one.** Q2 is graded on a `d=20` delay Hankel block that JUPITER's own it.8 ruling forbids
citing on either frozen wing, and its journal entry is a section reference standing in for
evidence. Q6's gap cites a permuted-oracle experiment that permutes `torch.randn`, reports
`√2` for `1.4060`, and rests on an identity true at every seed — with the `≈` dropped on the
way into the ledger row that it.35 reads.

**The withholding was correct. The repair is small and the round should take it: re-point
Q6 §3.4 at `equilibrium_oracle`, restore the `≈` at `V20_R15_LEAP_LEDGER.md:131`, re-grade
Q2 on BED-M or mark it ungraded, and journal what the reports already prove.** Then the `+6`
is earned on the merits the table already has.

**And before any of that: fold `V20_R15_IT9_MARS.md` into the it.9 journal entry and
re-derive the DISTANCE line.** `12 of 16` and `7 of 8 against softmax 0 of 8` are standing
in the record, unretracted, in an entry that closes by asserting they survived the iteration
that broke them. **That is the round's largest debt, and it is larger than the `+6`.**
