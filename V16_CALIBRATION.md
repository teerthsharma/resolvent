# V16_CALIBRATION — the calibration column

**D-CALIB**, `99777ab:CEQ_V16_CONTRACT.md:251`, owner SATURN. Opened at R12 it.1.

L-SIGN requires a calibration column *kept across rounds*: statements checked,
how many were wrong, and the **sign** of the error. This file is that column. It
is append-only: each round adds one summary row and one detail section, and no
earlier row is edited after its round closes. A row whose numbers were
transcribed from another document rather than re-read against the producing file
is worthless for this purpose, so every cell below carries the file it was read
from and the audit of the transcription is itself recorded (§3).

---

## 1. THE COLUMN

| round | statements checked | wrong | sign of error | notes |
|---|---|---|---|---|
| R11 | **9** as filed (`99777ab:V15_LEDGER.md:649-661`); **≥ 17** with the confirmations that table omits | **9 of 9** filed rows; **10 of ≥ 17** with confirmations | **7 `+`, 1 `−`, 1 unsigned.** Direction holds: `7/8` signed rows optimistic, one-sided sign test `p = 0.0352` | The filed prose says *"six wrong, all six optimistic."* Both halves are off — see §2 and §3. Denominator was chosen after the audit and lists no confirmations, so `9/9` is a count, not a rate |
| R12 | — | — | — | to be filed at it.38 (`99777ab:CEQ_V16_CONTRACT.md:298`); every R12 prediction and counter scored, both halves |

**Sign convention, fixed here and used in every later round.**

- `+` — the statement, if believed, **credits the project more** than the
  measurement supports (optimistic).
- `−` — the statement **understates** the project relative to the measurement
  (pessimistic).
- *unsigned* — the error has no direction. An internal contradiction is the
  canonical case: one half of it is right, so it cannot be one-sided.

---

## 2. R11's NINE, ROW BY ROW

Source of the list: `99777ab:V15_LEDGER.md:649-661`. Each row was verified against its
producing document, not against the ledger's summary.

| # | claimed | where | measured | producing file | sign |
|---|---|---|---|---|---|
| 1 | *"`g == 0` gives bitwise standard attention (Lean #5)"* | `99777ab:CEQ_V15_CONTRACT.md:139-140` | **FALSE**, three independent routes. At `g ≡ 0` the unnormalized causal hop's row `i` sums to `i + 1`, not `1` | `lean/CEQ/V15.lean:221` `gate_zero_row_sum`; `MISTAKES.md` [[V-23]], [[V-24]] | **+** |
| 2 | *"two-thirds of the pre-v13 section-5 strikes were `[V]`-as-theorem"* | `99777ab:CEQ_V15_CONTRACT.md:54` | **`4/40 = 10%`.** 44 rows, 4 record a claim that held, 40 genuine strikes, 4 are `[V]`-as-theorem. `66.7 / 10 = 6.67×` overstated | `MISTAKES.md` [[P-10]]; `99777ab:attic/workdonenew.pre-v13.md:160-211`; `99777ab:V15_N2_SATURN.md` Task B | **+** |
| 3 | *"the sizing model replaced by the calibrated one everywhere it was cited"* — filed as it.1-4 work | `99777ab:CEQ_V15_CONTRACT.md:251` | **Already satisfied.** `ceq/sizing.py` was calibrated (`C_OPERATOR = 3.9` at `:48`; `bf16_autocast = (2.2, 3.4)` at `:68`). All 5 flagged sites are the *record* of the repair, not uses of the stale model | `MISTAKES.md` [[M-17]]; `99777ab:V15_N2B_SIZING_CITATIONS.md:25-33`, `:139` | **−** |
| 4 | *"Ceiling approximately 38 from the current 22 percent."* | `99777ab:CEQ_V15_CONTRACT.md:285` | **No live producer anywhere in the tree.** The two `22%` hits in the repo are prose about a gap and a different quantity. Baseline under RUL-7 is `0 of 39 = 0%` | `99777ab:V15_CONTRACT_ARITHMETIC_AUDIT.md:225-253` (A-4) | **+** |
| 5 | *"TOST retires to `N >= 23` runs"* | `99777ab:CEQ_V15_CONTRACT.md:140` | **Achieved power at `N = 23` is `0.0669`.** The CI first *fits* at `N = 23` (confirmed) but power `0.80` first arrives at `N = 70` (confirmed). The clause licenses a verdict from a design with `6.7%` power | `99777ab:V15_CONTRACT_ARITHMETIC_AUDIT.md:76`, `:126` (A-1) | **+** |
| 6 | `H-hat = alpha-hat + 1/2` as BED-K's recovery target | `99777ab:CEQ_V15_CONTRACT.md` PART III | **Shipped without its stationarity hypothesis.** ARFIMA(0,d,0) is stationary and invertible iff `\|d\| < 0.5`, so the relation's domain is `α ∈ (0, 0.5)`, `H ∈ (0.5, 1.0)`. The contract gives a lower bound on `H` and no upper bound | `99777ab:V15_CONTRACT_ARITHMETIC_AUDIT.md:192-200` (A-3) | **+** |
| 7 | `[RUN: best first-order recurrence 0.990]` on the delay bed | `99777ab:CEQ_V15_CONTRACT.md:180` | **Matches neither bed at any tested `H`.** Delay bed `R² = −0.000166` (`−0.000170` in the independent N4 run, agreeing to rounding); power-law bed `0.604` at `H = 0.75`, `0.755` at `H = 0.9` | `99777ab:V15_JUPITER3_KERNEL.md:363`, `:371`, `:387`, `:556`; `99777ab:V15_N4_BEDK.md` | **+** |
| 8 | PART III *"attention-native, scan-blind"* vs PART IV R3 *"scan-only ≥ 0.95, attention near 0"* | `99777ab:CEQ_V15_CONTRACT.md:180` and PART IV | **Mutually inverse.** The arms are transposed; whichever number the cell returns, one section is confirmed and the other refuted, so the registration constrains nothing. PART III is right for the delay bed; PART IV is the inverse of the truth | `MISTAKES.md` [[M-20]]; `99777ab:V15_JUPITER3_KERNEL.md` | **none** |
| 9 | *"Carriers conserve mass to `1e-12`"* | `99777ab:CEQ_V15_CONTRACT.md:125` | **False for the contract's own central carrier.** §S-M's "ONE unnormalized causal hop" has row `i` summing to `i + 1` at `g ≡ 0`. Under the narrow reading it is true and silently omits the one carrier a reader needs | `MISTAKES.md` [[V-23]]; `lean/CEQ/V15.lean:221` | **+** |

**Totals: 9 rows, 9 adverse verdicts, 7 `+`, 1 `−`, 1 unsigned.**

---

## 3. WHAT THE VERIFICATION FOUND THAT THE SUMMARY DOES NOT SAY

A calibration column that transcribes is worse than none, so this section records
where the filed summary and the sources disagree. Three findings.

### 3.1 The count is not six. The table's own nine rows are nine adverse verdicts.

`99777ab:V15_LEDGER.md:649` reads *"Nine contract statements checked; six were wrong"* —
and then lists **nine** rows, every one of which carries an adverse verdict. The
R11 round entry (`MISTAKES.md`) enumerates six of them: rows 1, 2, 3, 4, 5 and 9.
Rows **6** (`H = α + ½` without `|d| < ½`), **7** (`[RUN: 0.990]` matching neither
bed) and **8** (PART III against PART IV) are dropped from the prose list and
appear nowhere in it as wrong.

This is not a considered subset. All three are discussed elsewhere in the same
round entry — §2 records `H = α + ½` entering v15 without its hypothesis, and §3
records M-20 — so they were known and simply not carried into the six-item
sentence. **The correct filed number is nine wrong of nine listed.**

### 3.2 The signs are not all one way. Row 3 goes the other direction, row 8 has none.

- **Row 3 is `−`.** The contract scheduled a repair that had already been made
  (`ceq/sizing.py` was calibrated before the round opened). Believing the contract
  makes the repo look *less* repaired than it is. That understates the project;
  it does not over-credit it. The round entry lists it among "all six optimistic"
  as *"a repair reported as outstanding that was already done"* — which is an
  accurate description of the fact and the wrong sign for it.
- **Row 8 is unsigned.** A pair of mutually inverse registrations cannot be
  one-sided: PART III is right for the delay bed. An error whose two halves point
  opposite ways contributes no direction.
- **Row 2 is `+` on the round's reading and the reading is worth stating.**
  Overstating one's own past defect rate is, read literally, self-critical. It is
  scored `+` here because the statement's function in the document is to be the
  evidence base for **L-EQ**, a law the same contract introduces: a `6.67×`
  overstatement makes the new law look better founded than it is. Sign scored
  against what the statement is *for*, and the ambiguity is recorded rather than
  hidden.

### 3.3 The direction is real, and the round entry's own six does not establish it.

One-sided sign test, `p = 0.5` under the null that error direction is chance:

| set | optimistic / signed | one-sided `p` | verdict |
|---|---|---|---|
| all nine, honestly signed | **7 / 8** | **`0.0352`** | direction established at `α = 0.05` |
| the round entry's six, honestly signed | **5 / 6** | `0.1094` | **not significant** |
| the round entry's six as filed ("all six") | 6 / 6 | `0.0156` | the claim as written, and it is not what the rows say |

**R11's headline is right about the direction and wrong about the evidence it
offers for it.** The finding clears `0.05` only once the three rows the prose
dropped are put back in. The paragraph the round entry actually needed is the one
it did not write.

### 3.4 The denominator was chosen after the audit and lists no confirmations.

`9/9` is a count of adverse rows in a hand-assembled table. It is not an error
rate, because nothing fixed the set of statements *before* the checking started
and the table lists no statement that survived.

One sub-census in R11 does have a stated denominator with confirmations in it —
`99777ab:V15_CONTRACT_ARITHMETIC_AUDIT.md`, which re-derived each claim numerically:

| sub-census | checked | wrong | rate | Wilson 95% |
|---|---|---|---|---|
| arithmetic audit (`A-0a`–`A-0g` CONFIRMED, `A-1`–`A-4` FINDING) | **11** | **4** | `36.4%` | `[0.152, 0.646]` |
| ledger's nine (adverse rows only, no confirmations listed) | 9 | 9 | `100%` | `[0.701, 1.000]` |
| union, de-duplicated (`A-1`, `A-3`, `A-4` appear in both) | **≥ 17** | **10** | `58.8%` | `[0.360, 0.784]` |

The union is `≥` because no node kept a full list of what it checked and passed.
The audit also found a **tenth** contract defect the ledger's nine omits — `A-2`,
the Grünwald–Letnikov weights `w_k = (−1)^k C(−α, k)` returning **NaN** at the
contract's own `α = 1` bind under `scipy.special.binom`
(`99777ab:V15_CONTRACT_ARITHMETIC_AUDIT.md:152-190`). So the nine is a selection from a
larger set, in both directions.

**For R12 this is a procedural requirement, not a caveat:** the set of statements
to be checked is listed *before* the checking, and confirmations are filed in the
same table as findings. Otherwise the next round's row measures the auditor's
appetite, not the author's calibration.

---

## 4. THE DISCOUNT — what R12 does with this, operationally

L-SIGN says *"a one-sided sign discounts the author."* This is what discounting
means as a rule a later node can apply without judgement. It is deliberately not
a sentiment and deliberately not a multiplier.

### D-CALIB-1 — the counter is the point estimate.

Where `99777ab:CEQ_V16_CONTRACT.md` files a prediction and a counter-prediction, **a node
reading the pair before data treats the counter as the expected outcome and the
prediction as the optimistic tail.** The counter is by construction the
pessimistic half, and `7 of 8` signed R11 misses were optimistic.

Applied to the round's largest filing, PART IV R1′:

| half | statement | how R12 reads it before data |
|---|---|---|
| PREDICTION | 8/8 converge, interval no longer straddles, the 5/8 crossing **banks** (`+12`) | the optimistic tail |
| COUNTER (author, filed against himself) | 8/8 converge but the crossing **shrinks to ≤ 3/8**, the cap having removed gain the divergent seeds exploited | **the expected outcome** |

Concretely: **the scoreboard may not carry `+12` as an expected value at any point
before it.8.** Planning, pricing and the it.8 branch text are written against
"stability bought at capability's price" as the base case, and the `+12` banking
is the upside branch. The same applies to R3's composed-arm prediction against
its counter (`composed loses on (a), matches on (b)`).

### D-CALIB-2 — a prediction with no counter is not read at all.

Already a KILL (`99777ab:CEQ_V16_CONTRACT.md`: *"Prediction without counter-prediction ⇒
not filed"*). The operational form: a node that encounters a bare directional
prediction in a contract clause does **not** apply a discount to it and does not
plan against it. It reports the missing counter and blocks the cell. Discounting
an unpaired number is a judgement call, and judgement calls are what this column
exists to remove.

### D-CALIB-3 — sign, never size. No numeric shrink factor is authorised.

Nine observations are enough to see a direction and not enough to estimate it.

- Direction: `7/8`, one-sided sign test `p = 0.0352`. **Established.**
- Size: the Wilson 95% interval on the optimism fraction at `7/8` is
  `[0.5291, 0.9776]`. It excludes `0.5` and fixes nothing else.

So a rule of the form *"multiply every predicted gain by 0.6"* or *"halve every
predicted crossing count"* is **forbidden** — the data cannot distinguish `0.53`
from `0.98`, and a multiplier invented at this `N` would be exactly the kind of
unsourced constant `P-1` catches. What the interval licenses is an **ordering**,
not a scaling: where the prediction and the counter disagree, the counter is
ranked first. Nothing more.

### D-CALIB-4 — order the round so the optimistic half is refuted cheaply.

A one-sided prior has one free consequence: **the cheapest discriminating
measurement of the optimistic half runs first.** An optimistic prediction is
refuted more cheaply than it is confirmed, so scheduling the refutation early
converts the bias into saved wall-clock rather than into commentary. R12's it.7
early-warning column is already this shape — `λ̂` and gate-`R²` from step 0 per
seed, which reads on the divergence story before the loss curve does.

### D-CALIB-5 — the row is appended whether or not it flatters.

At it.38 R12 files its own row: statements checked (**listed before the checking
began**), wrong, sign per statement, and the sign test. Confirmations are listed
beside findings. The author's counters are scored as well as his predictions
(`99777ab:CEQ_V16_CONTRACT.md:297`, VENUS's forecasting record) — a counter that was also
wrong, and wrong pessimistically, is a `−` and goes in the column. Two rounds
gives a second point; it does not give a size, and D-CALIB-3 stands until the
column has enough rows to estimate one.

---

## 5. LIMITS

Collected once, here.

- **`n = 9` is the whole sample, and its denominator was chosen after the fact.**
  The rate `9/9` is a count of adverse rows in a hand-assembled table, not an
  error rate. The union figure `10 of ≥ 17` is better founded but mixes one
  sub-census with a stated denominator against a list that recorded only
  failures; adding adverse-only items to a denominator biases it upward.
- **The sign convention is this file's, applied retrospectively.** R11 signed its
  six as a group and did not sign row by row, so §2's per-row signs are a reading
  taken here from each row's measured verdict. Row 2's sign depends on whether
  the statement is scored against the past it describes or the law it supports,
  and §3.2 records both readings.
- **`p = 0.0352` is a one-sided sign test on eight rows.** It is not a test of
  "the author is optimistic" in general; it is a test on eight statements from
  one document in one round, selected by the people auditing it.
- **Rows 6, 7 and 8 were signed here for the first time.** They were checked in
  R11 and their verdicts stand on their producing files, but no R11 document
  assigned them a direction, so their contribution to `7/8` is this file's
  reading, not R11's finding.
- **The `−0.000166` / `−0.000170` pair in row 7 is two runs, not a discrepancy.**
  `99777ab:V15_JUPITER3_KERNEL.md:371` states the reproduction explicitly and uses the
  agreement as the check that both numbers came from the same corpus.
- **`99777ab:house-events.jsonl` was not consulted.** It is never grepped; nothing in
  this file needed the ledger, so no `scale/ledger.py` parse was run.
