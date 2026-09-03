# V20 R15 — it.14 — JUPITER (MYCROFT). THE THEORY TABLE FREEZES. PHASE B CLOSES.

Branch `v17k-gate0`. **No git writes. Nothing touched Kaggle. No training run started.**
Nothing under `results/` opened for write. Files created:
`V20_R15_THEORY_TABLE.md`, `tests/jupiter/test_v20_r15_it14_theory_table.py`, this report;
`house-events.jsonl` appended (7 events, each with `t` and an executable `run`).

**The frozen table is `V20_R15_THEORY_TABLE.md` — its own artifact, so the leap's primary
input is a file and not a section of a report.**

---

## RED FIRST, VERBATIM, AGAINST UNMUTATED CODE

The node was written before the table existed:

```
E   FileNotFoundError: [Errno 2] No such file or directory:
    'C:\\Users\\seal\\Desktop\\New folder (32)\\V20_R15_THEORY_TABLE.md'
tests\jupiter\test_v20_r15_it14_theory_table.py:75: in table_text
    return TABLE.read_text(encoding="utf-8")
...
11 errors in 2.33s
```

**And the planted negative went RED first on a real bug in this office's own checker,
which is the only reason the green below means anything:**

```
>       assert len(found) == 2, found
E       AssertionError: ['no/such/file.py:1 - no such file']
E       assert 1 == 2
```

The first draft recognised a path by `"/" in path`. **The round's entire law corpus is
root-level `.md` files with no slash in them** — so the checker would have silently skipped
every citation to `CEQ_V20_R15_CONTRACT.md`, `V20_R15_LEAP_LEDGER.md` and every report,
and reported green over a table it had not read. A citation checker that cannot see the
files this round cites is worse than none. Fixed at
`tests/jupiter/test_v20_r15_it14_theory_table.py:60` — recognition by extension, with the
reason written in the source.

**GREEN.** `[RUN] python -m pytest tests/jupiter/test_v20_r15_it14_theory_table.py -q`
→ **`13 passed in 0.99s`**. Both calibrations fire: the planted bad citation is caught, the
real citation passes.

**`[RUN] cd lean && lake build; echo exit=$?` → `exit=0`**, zero output lines, Lake
`5.0.0-6fce8f7` / Lean `4.7.0`, lakefile at `lean/lakefile.lean:1`. **Stated as a limit, not
a boast: it was a silent cache hit, not a recompile.** The 13 `.olean` under
`lean/.lake/build/lib/` date to Aug 31 and a from-scratch build needs `.lake` deleted, which
is a write.

---

## TASK A — THE TABLE IS FROZEN. TWELVE CELLS, EACH WITH SIX FIELDS.

`1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4`, test-bound at
`::test_the_grade_census_matches_the_cells`.

Every cell carries **GRADE · GAP · DECLARATION · CENSUS · ROUTE · ARENA**, all six
enforced non-empty. **Every theorem is named by DECLARATION and `path:line` — never by
annex number**, enforced by `::test_no_declaration_names_a_theorem_by_annex_number_alone`.

**`[RUN]` citation re-verification at HEAD: 123 `path:line` citations across 37 distinct
files, 0 bad.** Independently, 14 spot-checked citations came back **14/14 VERIFIED, 0
MOVED**, including `pathProd_eq_zero_iff` @ `lean/CEQ/V16Domain.lean:129` and the nine
sibling argparse flags at `scripts/v15_r1.py:547` (exactly nine, confirmed).

### The three cells where "cited by annex number only" is stated *in the cell*

The failure mode this office withdrew six Lean citations for is now a field the leap can
see rather than a fact it must infer:

- **Q1/W3** — the positive class statement has **no declaration**; `Lean #21 [S]` was
  withdrawn at it.2 and never replaced. The cell carries the *negative* boundary
  declarations and says the positive side is unbacked.
- **Q3/W3** — the theorem half is *"M2 GATE-LANDSCAPE THEOREM… `Lean #19 [M]`"* with **no
  declaration name given**. The cell says so and rests on the instance alone.
- **Q6/W1 and Q6/W3** — **M13 Wasserstein has no producer in this tree.** A keyword sweep
  returns no `.py` for `wasserstein` and none for `kantorovich`; the annex's
  `[RUN: 0.492 vs KL 0.519]` resolves to `0.492188` from
  `scale/foreman_consequence.py:12`, **an unrelated quantity.**

### THE THREE DEFECTS ARE INSIDE THE TABLE, NOT AROUND IT

**§0.1 — the scale has no text.** The operational rubric the twelve grades were actually
assigned under is reproduced verbatim from `house-events.jsonl:12784`, **labelled a
reconstruction**, with the Inspector's split ruling stated: **this office's reconstruction
BOUND** to its stated usage and to no other, **SATURN's NOT BOUND** (eight counterexamples,
his own). Five counterexamples are named in the table: `F4` overloaded three ways
(unattempted `L-7` / struck `L-4` / domain-empty `L-3`); `F2` and `F3` unordered; **`M14`
carrying three grades in three files** — `F3` at `CEQ_V20_R15_CONTRACT.md:239`, `F4` at
`V20_R15_LEAP_LEDGER.md:24`, **`F1` at `V20_R15_JOURNAL.md:645`**; an `F4` row carrying a
measured constant; and this office's own two-grades-one-object defect at Q2/W1. The table
instructs the leap to read the grades **ordinal within a column, nominal across columns.**

**§0.2 — `floor₁` is a capability threshold and every Q5 cell says which floor it means.**
Both Q5 DECLARATION fields name `floor1` @ `scripts/v15_r1.py:586` explicitly as
*"a capability threshold — NOT Fano, NOT rate–distortion, NOT Hankel."*

**§0.3 — Q4's `F3` is a fact about the harness.** `S, D = 64, 24` @
`scripts/v15_r1.py:137`, nine sibling flags, zero override paths, `s = 64` on 40 of 40. The
table states the two `F3`s are **one harness fact written twice, not two failures and not
evidence about the primitives.**

### THE Q5 CENSUS WAS STALE AND IS RE-MEASURED

`[RUN]` a dedupe of the three result files on `(kind, seed)`:

| | published | **measured at it.14** |
|---|---|---|
| banked cells with `eval_nrmse` | 34 | **40 unique, from 43 rows** |
| **exact re-emissions** | **never journalled by any office** | **3** — `arm_pl` seed 0, `arm_smprime` seeds 0 and 1, identical `eval_nrmse` to 16 digits under one `instrument_hash` |
| cells violating `floor₁` | 6 of 34 | **13 of 40** (12 `arm_pl`, 1 `arm_smprime`) |
| `s = 64` | 34 of 34 | **40 of 40** |
| `dist_to_skyline` populated | 0 of 34 | **0 of 40** |

**The `6 of 34` this office published at it.8 is stale by eight cells.** The three
re-emissions also mean `sum(secs)` over the *unique* 40 is **`316.954 s`**, against
MERCURY's `317.092` over 40 rows — a `0.138 s` difference that is pure duplicate
accounting, and neither figure is wrong.

---

## TASK B — THE ARENA TICKET, AND THE PART OF IT THAT IS VOID

**Stated per wing in §3 of the frozen table.** Of four clauses, **W1 and W3 can be scored
today on clause (1) only.**

| clause | W1 | W3 | status |
|---|---|---|---|
| (1) BED-M CP-lower > 0.5 | SCOREABLE, `1/16` | SCOREABLE, `12/16` | blocked on `⟨CLAUSE_1_TAIL⟩` |
| (2) BED-K(a) within `1/d` | **NOT SCOREABLE** | **NOT SCOREABLE** | **VOID** |
| (3) lowest GPU-s-to-floor | not as written | not as written | 4 edits + a 5th unpriced |
| (4) witness tiebreak | **NOT SCOREABLE** | **NOT SCOREABLE** | no arm consumes a chess corpus |

**Clause (2) is void and the table says it is a GAP IN THE ORDER, not a tiebreak.** Both
wings are BED-M; the Hankel ceiling binds BED-K only — *"do not cite it on W1/W3 at all"*,
this office's own row at `V20_R15_IT8_JUPITER.md:473`, the ruling that re-graded Q2/W1 to
F4. MERCURY prices it at nothing because **zero cells of BED-K's shape have ever run.**
**A tiebreak that cannot be evaluated still orders the arms — it never fires. A gap does
not.**

### THE CONSEQUENCE NO OFFICE HAD STATED, AND IT IS THE ONE THING THIS ITERATION ADDS TO THE ARENA

**`⟨CLAUSE_1_TAIL⟩` decides whether clause (2)'s void is LATENT or FATAL.**

- **One-sided** (`0.5156`): W3 clears clause (1) **alone**. The order terminates at (1).
  Clause (2) is never reached. **The void is latent.**
- **Two-sided** (`0.4762`): **no entrant clears clause (1).** All arms tie at the first
  clause, the order **falls through to (2)** — which cannot be scored. **Phase C returns no
  winner at any price.**

MERCURY ruled *"Phase C as written cannot produce a winner"* and listed the two facts
separately. **They are one fact.** The tail ruling is not a bookkeeping convention; it
decides whether the arena has a structural hole or an unused one. **This office still will
not pick the tail** — choosing the tail that makes the round score is the catalogued
failure of selecting a threshold after seeing the data it judges.

**Clause (4) is also not scoreable on either frozen wing**, and the table corrects a
standing error to say so: `[RUN] python -c "from ceq import kdata;
print(list(kdata.BED_SPECS))"` → `['bed_m', 'bed_k', 'bed_1']`. **The chess witness is not
a registered bed.** `V20_R15_LEAP_LEDGER.md:131` calls it *"the one registered bed whose
output is categorical"*, and **that sentence is this office's and is withdrawn.**

---

## TASK C — THE F4 RULING, STATED, AND FOUR OF THIS OFFICE'S LEDGER ROWS RESOLVED

### RULING J-14 — an `F4` cell is NOT leap material as an `F4`

**The rule applied, stated so the leap does not have to derive it:** *TERMINAL* means a
**bound** — a claim about every instrument in the class. *LEAPABLE* means a **missing
statement** a theorem could supply. **An empty domain is neither.** It is a claim about
*this round's instruments*, repaired by a bed, a head or a journalled field — never by a
theorem. TERMINAL overstates it; LEAPABLE sends the leap hunting a theorem whose hypotheses
nothing here satisfies.

**Each F4 cell is filed NOT-PUT with a named ADMISSION CONDITION** and re-enters the gate
at whatever grade it earns once that condition holds:

| F4 cell | admission condition |
|---|---|
| **Q2/W1** | one cell of BED-K's shape actually run — generator at `ceq/kdata.py:475`, **zero callers under `scripts/`** |
| **Q6/W1** | a distributional head — **an arm change, and the wing list is frozen** |
| **Q6/W3** | that head **plus** a metric that is not permutation-blind — W1-marginal is **refuted** here, not merely inapplicable |

**The F4 ledger rows are six, not five.** L-3, L-4, L-7, L-8, L-13, L-14 —
`V20_R15_IT12_INSPECTOR.md:272`; SATURN counted five and omitted **L-8**. Their
LEAPABLE/TERMINAL stamps are **out of scope for the it.35 gate** under J-14. No finding is
withdrawn.

### RULING J-14b — a hybrid verdict is not a verdict. **Four of the five inadmissible rows are this office's, not one.**

`[RUN] python -m pytest tests/saturn/test_v20_r15_it12_saturn.py::test_the_FIELD_ruling_is_enforced_row_by_row -q`
→ **`1 passed`**. The detector (`tests/saturn/test_v20_r15_it12_saturn.py:296`) names
`V-it7`, **`L-2`, `L-9`, `L-13`, `L-14`** — **the last four all JUPITER's.**

**And the framing does not match the mechanism.** *"FIELD names a want rather than a
discipline"* describes **only `V-it7`**. The four JUPITER rows trip
`head.startswith("none")` — and the ledger's own convention says a TERMINAL row naming no
field is **correct** (`V20_R15_LEAP_LEDGER.md:99`). What pulls them in is the substring
test `"LEAPABLE" not in verdict` reading a **hybrid verdict** as LEAPABLE
(*"TERMINAL as a leap target, LEAPABLE by 0 GPU-s of bookkeeping"*, and three more).

**A row saying both hands the leap a fork and no rule for taking it.** All four are
resolved to a single token:

| row | resolved | ground |
|---|---|---|
| `L-2` | **TERMINAL** | no theorem repairs a record that never wrote the field |
| `L-9` | **TERMINAL** | no theorem supplies a slope from one point |
| `L-13` | **NOT-PUT** (F4) | superseded by J-14; **FIELD column withdrawn** — *"the nearest LEAPABLE restatement is calibration / probabilistic forecasting"* is a **want** behind a hedge |
| `L-14` | **NOT-PUT** (F4) | same defect, same column |

**Every finding in the four rows stands** and is carried in its cell in §2. Only the verdict
tokens and two FIELD columns are withdrawn.

**And two limits on the detector, recorded although this office is the party its findings
are against.** `KNOWN_INADMISSIBLE` (`tests/saturn/test_v20_r15_it12_saturn.py:320`) is a
**frozen literal** — a change-detector, not a derivation of five. **Its planted negative
(`:330`) exercises only the `ROUND_NOUNS` arm; the `none` arm that produces four of the five
findings has no control at all.**

---

## KILLS THIS ITERATION, EACH WITH ITS REPLACEMENT ROUTE

| killed | replacement route |
|---|---|
| `V20_R15_LEAP_LEDGER.md:131` — *"the one registered bed whose output is categorical"* | register a chess bed in `ceq/kdata.py:475`'s `BED_SPECS`, **or** stop citing the witness as an available Q6 domain. Q6/W3's ROUTE now says "none on the frozen wings" |
| the hybrid verdicts on `L-2`, `L-9`, `L-13`, `L-14` | resolved to single tokens above; the hybrid prose moves to the gap column, where it was always accurate |
| `L-13`/`L-14` FIELD columns | superseded by the ADMISSION CONDITION table — an F4 cell states what would make its domain non-empty instead of naming a field |
| the published `6 of 34` floor violation | re-measured to **13 of 40**, `[RUN]` command journalled in `house-events.jsonl` |
| this office's own citation checker (`"/" in path`) | recognition by extension, `tests/jupiter/test_v20_r15_it14_theory_table.py:60`, reason in the source |

---

## LIMITS

`lake build` returned `exit=0` but was **a cache hit, not a recompile**; the `.olean`
artifacts date to Aug 31 and forcing a clean build requires a write. The table's citation
checker **confirms a file exists and a line is in range — it cannot read semantics**, which
is why every DECLARATION also carries the declaration's name; **123 citations were checked
this way and none was checked for meaning.** Two soft mismatches found and recorded in the
table's §5: `scripts/v15_r1.py:17` is **module-docstring prose**, not code, so every
citation of it is a citation of a comment; and `zero_hop_mask` @ `ceq/arm_smprime.py:559`
has **no production caller** — `tests/jupiter/test_v20_r15_it8_q4_q5.py:222` only asserts
its source text is present, which is not a test that it runs. Every arena figure in §3 is
MERCURY's and is not re-derived here; his `secs` basis is un-synchronised CUDA host wall
clock. The `12/16` and `1/16` crossing counts are pooled over cells whose control is
incomplete — softmax exists only for seeds 0–7, so the eight newest `arm_pl` cells are
uncontrolled. **Nothing in the table was measured on BED-K or on the chess witness**,
because no cell of either shape has ever run in this tree.

**Suite:** `[RUN] python -m pytest tests/jupiter/ -q` → **`1 failed, 174 passed in
155.52s`**. The one failure is
`test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record`,
**pre-existing, not touched this iteration, and left RED on purpose** — it is a claim about
a written verdict and no measurement repairs it. **No existing test was edited.** The it.14
node is green in isolation at `13 passed`.

---

## OPEN, OWED, AND BY WHOM

1. **`⟨CLAUSE_1_TAIL⟩` — the author's, and it is now larger than it was.** It no longer
   only decides whether W3 crosses; it decides whether clause (2)'s void ends Phase C.
   **0 GPU-seconds.**
2. **`⟨L_GRADE_RUBRIC⟩` — the author's.** Twelve grades are assigned on a five-token scale
   with no text. **0 GPU-seconds**, one contract edit; split `F4` into `F4-empty` and
   `F4-unattempted` and make `STRUCK` its own token.
3. **The ~6 GPU-s capped run at seeds 2, 3, 7.** Priced at it.7, it.8, it.9 and now it.14.
   **Taken zero times.** It settles Q3/W3 causality, Q5/W3 crossing and pre-registerability
   at once, and it is the cheapest live experiment in the round.
4. **The randomised-execution edit** that would remove the `ρ = +0.70` run-order confound.
   **No office has priced it.**

**Scoreboard.** Phase B closes with **the table 12/12, all graded, all six fields, 123
citations resolving**. The `+6` is claimable on the contract's text; this office does not
claim it, because the grades are expressed on a scale with no text and **that is the second
open ruling above.** Carried: **2 of 44.**

**Distance to the north star:** unchanged this iteration. The table adds no capability —
it makes what the round does and does not know legible to a model that gets one call.
