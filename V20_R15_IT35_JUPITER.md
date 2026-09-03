# V20 R15 it.35 — EVALUATION-FOR-LEAP — JUPITER

**Graded independently of MARS. No coordination.** Clock read in this window, both from
`date` in this shell: **`2026-09-02T20:02:55+0530`** at open, **`2026-09-02T20:07:13+0530`**
at filing. Elapsed **4m 18s** of the 20-minute wall.

**The set the contract sends to the gate** (`CEQ_V20_R15_CONTRACT.md:140-146`): every
`F1`/`F2`/`F3` failure in the theory table. **That is EIGHT cells, not nine** — see §3.

---

## §1 — THE RULE THIS OFFICE GRADED UNDER, STATED BEFORE THE GRADES

**TERMINAL** iff the gap is either
**(a)** a *bound* — an inequality holding for every instrument in the class, which no
theorem removes; or
**(b)** a *datum the record never wrote* — because **no theorem writes a field**, and the
repair is an instrument line or a run, not a statement.

**LEAPABLE** iff the record **already holds the measurements** and what is missing is a
**statement about them**.

**This is NOT the rule the ledger publishes.** `V20_R15_LEAP_LEDGER.md:11-14` reads
*"LEAPABLE when the gap is a missing statement **or a missing measurement**"* — under
which `Q4/W1` is LEAPABLE. But `V20_R15_THEORY_TABLE.md:353-356` resolved `L-9` (the same
failure) to **TERMINAL** under RULING J-14b. **Since it.14, when J-14b resolved `L-9`
against the published rule, the round has published one grading rule and applied another —
twenty-one iterations, it.14 to it.35.** The applied rule is the one used
below, and it is stated here so MARS's independent grading can be compared against a
written rule rather than against a habit.

**One token per cell.** RULING J-14b (`V20_R15_THEORY_TABLE.md:358`): *a hybrid verdict is
not a verdict.* Applied here to §4.2's own table as well as to the ledger — see §4.

---

## §2 — THE EIGHT GRADES

| # | cell | grade | verdict | FIELD (LEAPABLE) or BOUND (TERMINAL) | ledger row | citation that lands |
|---|---|---|---|---|---|---|
| 1 | **Q1/W3** exact class, `arm_pl` | `F1` | **LEAPABLE** | **realization theory for linear time-invariant systems** — Hankel/Kronecker realization, the field stating which sequence map a finite exponential scan represents exactly | `L-1` | `lean/CEQ/V16Domain.lean:165` · `ceq/arm_pl.py:88` |
| 2 | **Q2/W3** outside class, `arm_pl` | `F1 + const` | **TERMINAL** | **BOUND:** `err_i ≥ dist(t_i, [min_{j≤i} b_j, max_{j≤i} b_j])`. A softmax row is `A_ij ≥ 0`, `Σ_j A_ij = 1`, so `y_i` is a convex combination for **every** `(g,s,q,k)`. This is a **nonexistence, not a truncation** — there is no `k` at ANY size. No theorem repeals a nonexistence | `L-17` | `tests/jupiter/test_v20_r15_it12_constants.py:11` |
| 3 | **Q3/W1** learnability, `arm_smprime` | `F2` | **LEAPABLE** | **transfer-operator / Koopman spectral theory** — specifically the part relating a *leading exponent* to *approximation error*, which is what turns a necessary window into a sufficient condition | `L-5` | `scripts/v15_r1.py:383` |
| 4 | **Q3/W3** learnability, `arm_pl` | `F1 + const` | **LEAPABLE** | **bifurcation theory / gradient-flow convergence** — which initialisations diverge. *(The `~6 GPU-s` capped run is cheaper than the reading; that belongs in the gap column, not the verdict — see §4.)* | `L-6` | `tests/jupiter/test_v20_r15_it7_q3.py:106` |
| 5 | **Q4/W1** cost law, `arm_smprime` | `F3` | **TERMINAL** | **BOUND:** the exponent in `S` is **unidentified**, `n = 1`, `s = 64` on `40 of 40`; `S` is a module constant with nine sibling argparse flags and zero override paths. **No theorem supplies a slope from one point** | `L-9` | `scripts/v15_r1.py:137` |
| 6 | **Q4/W3** cost law, `arm_pl` | `F3` | **TERMINAL** | **BOUND:** the same harness fact (§0.3). W3-specific and separate: `brute_force_path_sums` is `O(2^S)` and unguarded — an implementation guard, not a theorem | `L-9` | `ceq/arm_pl.py:304` |
| 7 | **Q5/W1** information floor, `arm_smprime` | `F1 + const` | **LEAPABLE** | **FIELD CONTESTED — see §5.** This office names **approximation theory for finite-dimensional state models (Kolmogorov n-width)**, not §4.2's rate–distortion | `L-11` | `scripts/v15_r1.py:586` |
| 8 | **Q5/W3** information floor, `arm_pl` | `F1 + const` | **LEAPABLE** | **finite-mixture inference** — when a pooled statistic over a bimodal population is inadmissible and what replaces it. The mixing variable `sign(lambda_hat)` is **already journalled on every cell**, which is exactly why this is LEAPABLE and `L-2` is not | `L-15` | `tests/jupiter/test_v20_r15_it8_q4_q5.py:151` |

**Tally: 5 LEAPABLE, 3 TERMINAL, over 8 cells.**

**But 8 cells are 7 failures.** §0.3 of the table rules Q4's two `F3`s *"the same fact about
`scripts/v15_r1.py` written twice … not two failures"* — and this office grades them as
one, filed twice because the gate is per-cell. **A leap dossier that counts them as two
overstates the round's failure population by one.**

---

## §3 — THE COUNT §4.2 GOT WRONG, IN ITS OWN TABLE

`V20_R15_THEORY_TABLE.md:319` is headed **"THE NINE CELLS THAT DO REACH THE GATE."** Its
own ninth row (`V20_R15_THEORY_TABLE.md:331`) is `Q1/W1`, graded `F0`, verdict column
**"not a failure."** The contract sends `F1/F2/F3` to the gate. **An `F0` is not in the
set, and the header contradicts its own last row.** The gradeable population is **eight**.

`Q1/W1`'s `0 of 24` attributability gap is real and is `L-2`, **already resolved TERMINAL**
(`V20_R15_THEORY_TABLE.md:353`) — *no theorem repairs a record that never wrote the field.*
It is carried here for completeness and is **not counted in the tally.**

---

## §4 — THE VERDICT COLUMN OF §4.2 STILL CARRIES THE FORM J-14b WITHDREW

RULING J-14b (`V20_R15_THEORY_TABLE.md:358`) withdrew the string **"TERMINAL as a leap
target"** from `L-9`'s verdict column as a hybrid (`V20_R15_THEORY_TABLE.md:354`). **Two
rows of §4.2's own gate table — thirty and thirty-one lines above the ruling, in the same
file — still carry that exact string** (`V20_R15_THEORY_TABLE.md:327`, `V20_R15_THEORY_TABLE.md:328`), and a third
carries the same shape (`V20_R15_THEORY_TABLE.md:326`, *"LEAPABLE, but ~6 GPU-s buys it
outright"*), a fourth likewise (`V20_R15_THEORY_TABLE.md:330`, *"LEAPABLE — and it is a
SCORING RULE, not a theorem"*).

**J-14b was applied to the ledger and not to the table that quotes it.** Resolved here to
bare tokens, which is what §2 files:

| §4.2 row | verdict as written | **resolved** |
|---|---|---|
| `V20_R15_THEORY_TABLE.md:326` Q3/W3 | *LEAPABLE, but ~6 GPU-s buys it outright* | **LEAPABLE** |
| `V20_R15_THEORY_TABLE.md:327` Q4/W1 | *TERMINAL as a leap target* | **TERMINAL** |
| `V20_R15_THEORY_TABLE.md:328` Q4/W3 | *TERMINAL as a leap target* | **TERMINAL** |
| `V20_R15_THEORY_TABLE.md:330` Q5/W3 | *LEAPABLE — and it is a SCORING RULE, not a theorem* | **LEAPABLE** |

**No grade changed.** Only four verdict tokens did, and the prose they carried moves to
the gap column where J-14b put `L-9`'s.

---

## §5 — THE ONE CONTESTED FIELD, AND THE GRADE OF THE CONTEST

**Cell Q5/W1. The grade `F1 + const` is not contested. The FIELD is.**

§4.2 names **rate–distortion / channel-capacity theory applied to the DETERMINISTIC oracle
case** (`V20_R15_THEORY_TABLE.md:329`). **On the round's own bed that field is degenerate.**
BED-M's oracle is *exact* — the cell says so: *"the real information floor on BED-M is
`oracle = 0.0`"*. For a noiseless deterministic map `Y = f(X)`, the rate–distortion
function admits `D = 0` at rate `H(X)`; **the information floor it returns is the `0.0`
the cell already has.** It cannot state a *nonzero distance* to that floor, and the
distance is the thing the cell is missing (`0.203920` at best, `0.852` modally, **no annex
theorem predicting either**).

**What is actually missing is not an information quantity but a representational one:**
what error a *finite-dimensional state* scan must incur against an exact map. That is
**approximation theory — Kolmogorov n-width / degree-of-approximation for finite-state
models**, the sibling of Q1/W3's realization theory and the same family twice in one table.

**The contest graded:** the cell stays **LEAPABLE** under either field, so the gate's
verdict does not turn on it. **The leap's ONE call does.** A leap sent to rate–distortion
on a noiseless channel spends its call deriving `D(H(X)) = 0`, which the round already
knows. **This is a live disagreement inside this office's own it.14 filing, and it is
recorded rather than smoothed.**

---

## §6 — THE THREE CELLS THE GATE DOES NOT SEE, AND WHAT EACH HIDES

**Not graded. `J-14` files each `NOT-PUT`** (`V20_R15_THEORY_TABLE.md:284`+): an empty
domain is neither a bound nor a missing statement, so it is **neither LEAPABLE nor
TERMINAL** — **a third state the contract's grammar has no word for**
(`CEQ_V20_R15_CONTRACT.md:140-146` offers two tokens).

**⟨F4_GATE⟩ was opened at `V20_R15_JOURNAL.md:3965` and is still open at
`V20_R15_JOURNAL.md:5668` — sixteen-plus iterations, unruled.** The INSPECTOR's it.17
finding stands: **`3 of 12` = 25 % of the leap's primary input is invisible to the
consumer.** This office does not rule it. **It states, per cell, what the gate is blind
to:**

| F4 cell | what the gate DOES NOT SEE | admission condition |
|---|---|---|
| **Q2/W1** | **Whether the `1/d` Hankel ceiling binds the winner at all.** The law binds BED-K; both frozen wings are BED-M; the intersection is **EMPTY**, and **zero cells of BED-K's shape have ever been run in this tree.** Consequence the gate inherits: **arena clause (2) is VOID** — one of four lexicographic clauses is unscoreable, so the winner is chosen on three | one cell of BED-K's shape run — `ceq/beds/bed_k.py:236` |
| **Q6/W1** | **Any distributional or calibration reading of `arm_smprime` whatsoever.** The arm returns `[n]`, one real per draw; **`0 of 40` banked cells journal a prediction vector, histogram, quantile or density**, so the marginal is not computable from the record without a re-run. M13 Wasserstein is cited by annex number and **has no producer in this tree** | a distributional head — **an arm change, and the wing list is frozen** |
| **Q6/W3** | **The same blindness, plus: the metric that would fill it is REFUTED here.** Marginal `W1` is permutation-blind on the real oracle — a predictor returning the oracle's own values in the wrong order scores `W1 = 0.0` exactly at `NRMSE = 1.421901019003236`, and metric and bed rank two predictors in **opposite** orders by `14.465410797679917×` | the head **plus** a metric that is not permutation-blind; CRPS/pinball must be **re-derived, not assumed** |

**The prognosis must carry this hole, not inherit it.** The one-sentence form:
*the gate graded seven distinct failures across eight cells and was structurally unable to
look at three more, one of which voids a clause of the ranking that chose the winner.*

---

## §7 — WHAT THIS OFFICE DID NOT REACH

- **MARS's parallel grading is not read and must not be reconciled here.** Two independent
  gradings have been the round's strongest instrument twice (it.22 censuses, it.29
  grammars) and **both times the disagreement was the finding.** Reconciliation is a later
  iteration's, not this one's.
- **The ledger's `L-M1`..`L-M5` and `L-3`/`L-4`/`L-7`/`L-8` rows are not re-graded here.**
  The first five are MARS's and this office does not regrade another office's rows; the
  latter four are `F4` and out of scope under `J-14`.
- **The `~6 GPU-s` capped run at seeds 2, 3, 7 is still not taken** — priced at it.7, it.8,
  it.9 and now it.35, **taken zero times.** It settles Q3/W3 causality and Q5/W3 crossing
  at once and is still the cheapest live experiment in the round.
- **No repair was made to `V20_R15_THEORY_TABLE.md`.** §3, §4 and §5 name three defects in
  it and **change no byte of it** — the table's length is load-bearing for external
  pointers (`RULING J-17e`), and an edit inside a 20-minute gate window is how a pointer
  moves silently. Filed as findings against a frozen file.

---

## §8 — LIMITS

Collected once.

**The rubric is a reconstruction.** `L-GRADE (F0–F4)` has no canonical text anywhere in
this repository (`V20_R15_THEORY_TABLE.md:20`+); this office's reconstruction was ruled
**BOUND to its own stated usage and to no other**. Every grade above is ordinal within its
column and nominal across columns.

**The eight `F`-grades are inputs, not findings of this gate.** They froze at it.14. This
gate sorts them; it does not re-measure them. Where an `F`-grade was itself contested the
contest was already resolved before this window — `M14` took three grades in three files
and resolved to `F4` at it.19 by `J-17d`; `Q2/W1` resolved `F1 + const → F4` at it.12. **No
`F`-grade was reopened here**, and §5 contests a **FIELD**, not a grade.

**The citations in this file were checked by the round's landing instrument, not by eye** —
see the node in §9. That checker confirms a path exists and a line is in range; **it cannot
confirm the declaration named is the declaration at that line.** That is the honest ceiling
and it is the same one `V20_R15_THEORY_TABLE.md:393`+ records.

**`~6 GPU-s`, `206–537 GPU-s` and `275`/`309` are MERCURY's figures, not re-derived here**,
and his `secs` basis is un-synchronised CUDA host wall clock.

**Wall clock: 20 minutes. The `F4` census, the arena clauses and the ledger rows were read
at HEAD in this window; the Lean build was not re-run and `lake build`'s green at
`V20_R15_THEORY_TABLE.md:393`+ is a cache hit, not a recompile.**

---

## §9 — THE NODE THAT BINDS THIS FILING

`tests/jupiter/test_v20_r15_it35_grade_citations.py`

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it35_grade_citations.py -q` → `4 passed in 0.27s`.**
Scoped to this node; **not a whole-suite scalar** — the INSPECTOR measured `773 / 773 / 804`
nodes across three offices all called *"the whole suite."*

- **`35 of 35` citations in this file LAND** — path exists, line in range. `[RUN]`, this
  window.
- Every cell in `§2` carries a verdict token that is **exactly** `LEAPABLE` or `TERMINAL`
  with **no qualifier** — the J-14b property, enforced rather than asserted.
- The three `F4` cells are named and **must not appear in the graded table** — a node, not
  a promise.

**RED provenance, three mutations against the filing, each run in this window:**

| mutation | node's reading |
|---|---|
| Q4/W1's verdict restored to *"TERMINAL as a leap target"* — §4.2's live string | `7 of 8` rows carry one bare token → **RED** |
| `scripts/v15_r1.py:137` → `:999137` | non-landing citation named → **RED** |
| `Q6/W1` (F4) inserted into the graded table | F4 cell in a graded row → **RED** |

The node also cannot pass without this file at all, which was its state when it was
written. **Green here is not the parser finding nothing.**

---

## §10 — THE GATE'S PRIMARY INPUT HAS DRIFTED OFF ITS OWN CENSUS, AND THE ONE UNSCORED CITATION IS INSIDE AN `F4` ROW

**Measured in this window, `[RUN]` against the it.21 instrument at HEAD:**

```
len(occ) = 133   scored_by_line = 119 (frozen 118)   by_heading = 13 (frozen 13)   unscored = 1
```

`tests/jupiter/test_v20_r15_it21_heading_anchor.py:188` freezes `scored_by_line == 118`
and `:190` freezes `len(occ) == 131`. **The theory table now emits `133` and scores `119`.**
`tests/jupiter/test_v20_r15_it24_census_retake.py` fails on the same drift.

**`[RUN]` `python -m pytest tests/jupiter/test_v20_r15_it21_heading_anchor.py tests/jupiter/test_v20_r15_it24_census_retake.py -q`
→ `4 failed, 15 passed` — and it reads `4 failed, 15 passed` with `V20_R15_IT35_JUPITER.md`
REMOVED FROM THE TREE and re-taken. Identical both ways. This gate did not cause it; the
drift was at HEAD before the window opened.** That control is the reason this is filed as a
finding and not as damage.

**The single unscored occurrence is `scripts/v20_m14_cheeger.py:347`
(`V20_R15_THEORY_TABLE.md:306`)** — and `:306` is the **`Q2/W1` ADMISSION-CONDITION row**,
i.e. **the one citation in the whole table that the round's location instrument has never
scored sits inside the part of the table the gate is forbidden to grade.** The two blind
spots are the same spot.

**The claim it carries is TRUE.** `[RUN]` `sed -n '347p;462p' scripts/v20_m14_cheeger.py`
→ both read `b = bed_k.build_delay(n=500, d=4, seed=7)`, the two callers at the registered
tuple, exactly as the row states. **Verified by hand here because no instrument covers it.**

**Second defect, same line: the bare relative pointers `` `:462` `` and `` `:152` `` have no
resolvable base.** `CITE_RE` requires a path with an extension, so **no citation instrument
in this round parses either of them** — they are invisible to the it.17, it.18, it.20,
it.21 and it.24 censuses alike. `:462` is recoverable from context. **`:152` is not**:
`scripts/v20_m14_cheeger.py:152` is a docstring line about mixing time and is **not evidence
for the sentence it is attached to** (*"no cell of BED-K's shape has been run"*, which is
sourced elsewhere in the table to `V20_R15_IT13_MERCURY.md:63`). A reader must guess between
two files.

**No repair made.** `V20_R15_THEORY_TABLE.md` is frozen at 443 lines and its length is
load-bearing for external pointers (RULING J-17e). **Repairing a citation inside a 20-minute
gate window is how a pointer moves silently.** Filed for the next iteration that owns the
table, with the mechanism named: **a citation notation the round's own parser cannot see
was introduced into the frozen file, and the freeze nodes could not fail on it because they
could not read it.**
