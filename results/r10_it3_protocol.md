# R10 P0 iteration 3 — spot-check protocol, PRE-REGISTERED

SATURN. Written and committed BEFORE any verification command was run against any drawn row.
The draw (`scale/spotcheck_draw2.py`, SEED=10003, `results/r10_it3_draw.txt`) is MARS's, not
SATURN's, is pinned to revision `f823b02`, and reproduces byte-identical from a clean tree —
`python scale/spotcheck_draw2.py --demo` then `python scale/spotcheck_draw2.py`, checked against
the committed output before this file was written.

The iteration-2 census was REJECTED at 3 KEEP failures against a pre-registered threshold of 2.
This file does not re-argue that ruling. It repairs the instrument the ruling exposed and re-runs.

## 0. What is under test, and what is not

The spot-check tests the **DISPOSITION** cell (KEEP / ATTIC). It does not test prose.

- **DISPOSITION FAILURE** — the row's KEEP is wrong (the thing is dead, unreachable, or its stated
  retention ground does not hold) or its ATTIC is wrong.
- **DESCRIPTIVE AMENDMENT** — a wrong number or wrong word in a non-disposition cell where the
  disposition still stands. Reported with its correction; does **not** enter the binomial.
- Exception, pre-registered: a test-count cell reading N>0 where the measured count is **0** IS a
  disposition failure. A file with zero collected tests is not a live test file.

The binomial is defined over **KEEP failures only**. An ATTIC row found alive is a finding and an
amendment; it does not enter the KEEP count.

**EVERY SECTION BELOW STATES ITS OWN BINOMIAL DISPOSITION EXPLICITLY, IN ITS OWN WORDS.** No
section relies on a cross-reference to this one, and no section is silent. That is the drafting
defect that decided the iteration-2 ruling and it is not repeated: section 5 there said "per
section 0" and section 2 did not, so the Health Inspector read the single cross-reference as a
line deliberately drawn, correctly overruled the author's own softer reading, and the count went
from 1 to 3. A protocol whose scope must be inferred from where a cross-reference is absent is a
protocol with a hole. Each rule below carries its own consequence.

## 1. Exclusions, fixed now so a citation count cannot be tuned later

For every "is it cited / is it reachable" measurement, these are NOT readers and are excluded:
`AUDIT.md`, `R10_ITERATION_01.md`, `scale/spotcheck_draw.py`, `scale/spotcheck_draw2.py`,
`scale/red_by_design.py`, `scale/doc_readings.py`, `results/r10_it2_*`, `results/r10_it3_*`, this
file, `house-events*.jsonl`, anything created by this iteration, and everything under
`.claude/worktrees/` (the AUDIT.md scope rule, MISTAKES.md V-13). A census naming a row is not a
reader of that row; counting itself is how a sheet certifies itself.

## 2. T-KEEP — a test file classed KEEP

PASSES iff BOTH (T-a) and (T-b) hold. **A failure of either is a DISPOSITION FAILURE and enters
the binomial.**

### (T-a) it collects

pytest imports the file. A collection error is a FAIL with no exceptions: a file that cannot be
imported is not an instrument. **Enters the binomial.**

### (T-b) its outcome is green, or red-by-design — REPAIRED

Exit 0 is green. A non-zero exit is red-by-design only if **every** failing node is rescued by one
of three routes. The routes are tried in order and a rescue always names its ground.

- **R1 — the ledger.** The node is on the `tests/chase/conftest.py:285` KNOWN_RED ledger, which
  marks it `xfail(strict=True)`, so it should not surface as a failure at all. A strict xfail that
  XPASSes is itself a FAIL.
- **R2 — the prefix.** The node is a `test_claim_*` function in a P1''-carved-out refutation
  instrument, whose RED is its deliverable.
- **R3 — the declaration. NEW, and it is the brief's own primary route.** A DECLARATION covering
  the failing node states, in the present tense and in band, that this artifact's red is its
  deliverable.

**Any other red is a FAIL and enters the binomial.**

R3 exists because R1 and R2 are both NAME routes and the brief's primary route is a DECLARATION.
`tests/foreman/conftest.py` is a sys.path shim with no ledger; the chase ledger is scoped to
`tests/chase`; so every foreman red is unledgered BY DESIGN. MERCURY measured 13 `tests/foreman`
files carrying an in-band `# CLAIM AS WRITTEN -- RED` banner, of which the Inspector re-measured
12 as also carrying the `test_claim_` prefix — banner and prefix are DIFFERENT SETS, and three
banner files (`test_r1_settling.py`, `test_r2_maxplus_reduction.py`,
`test_r2_signed_consequence_fit.py`) carry no prefix at all. A classifier keyed on the name cannot
see the banner. Two of the three iteration-2 KEEP failures were that blind spot.

**R3 is mechanised, not read by eye.** `scale/red_by_design.py`, `--demo` green before this file
was committed. Its definition, in full:

- **The forms.** Enumerated by searching every tracked test file first, not guessed:
  `CLAIM AS WRITTEN -- RED`; `RED ON PURPOSE` / `RED on purpose`; `ALL RED`; `CURRENTLY RED`;
  `is RED` / `IS RED`; a docstring opening `RED.` or `RED,`; `RED … by design`; `RED tests`;
  `RED is its deliverable`.
- **The bands, narrowest first.** **T** the test function's own docstring; **B** the framed
  column-0 section banner it sits under, running forward to the next banner or EOF; **F** the
  module docstring. **A narrower non-declaring band OVERRIDES a wider declaring one.** Ten of the
  thirteen foreman banner files open `# CLAIM AS WRITTEN -- RED` and then CLOSE it with a
  `-- GREEN` banner; the override is what makes the second half of those files fail.
- **Excluded — `RED-first` and `written first (RED on purpose)`.** Authoring order, not present
  state. This is the load-bearing half of the route: 68 lines across `tests/` carry it, four times
  every declaration form combined. A test written red-first and since fixed is green; one that is
  red today is a defect, and R3 must not rescue it.
- **Excluded — `must FAIL` / `must fail`.** A statement about the SUBJECT, asserted by a GREEN test.
- **Excluded — a form appearing inside backticks.** A backticked span is a citation, not an
  utterance. Without this,`tests/loop/test_attic_never_removes_the_last_must_fire.py` rescues
  itself by quoting the foreman banner in its own module docstring. That false rescue was caught
  by the demo, which is why the demo exists.

### R3's REJECTION REGION, measured before this file was committed

A route that classifies every red as by-design certifies nothing, so the region where R3 refuses
to rescue is measured and stated, not asserted.

- **Structural, over the tree.** `python scale/red_by_design.py --survey`, every drawn row held
  out: **1,201 test functions, 60 rescued (5.0%), 1,141 REJECTED (95.0%)** — by band, T 19, B 35,
  F 6. R3 also SPLITS 21 files, rescuing some of their tests and refusing others, so it
  discriminates within a file and not merely across files.
- **Empirical, over reds actually measured.** The `tests/loop` + `tests/mars` + `tests/saturn` +
  `tests/mercury` sweep (no drawn row in it) returned **22 failing nodes; R3 rescues 9 and REFUSES
  13 (59%)**. Refused outright: all five reds in
  `tests/loop/test_conftest_import_is_order_dependent.py`, both in
  `tests/loop/test_manifest_refuses_an_absence_it_has_not_earned.py`, both in
  `tests/loop/test_identity_manifest_covers_every_beyond_key_field.py`, both in
  `tests/loop/test_no_module_writes_a_file_at_import.py`,
  `tests/loop/test_attic_never_removes_the_last_must_fire.py`, and — against SATURN's own
  interest — `tests/saturn/test_r10_it2_spotcheck_reds.py::test_an_intentional_red_in_tests_chase_is_on_the_ledger`.

If R3 rescues every red on a drawn row, that row is reported WITH its rescue ground named per
node, so the rescue can be read and disputed.

### Class cell

A class error that would flip the disposition is a FAIL and **enters the binomial**; one that
leaves the row KEEP is an amendment and **does not**.

## 3. S-KEEP — a scale module classed KEEP

PASSES iff BOTH. **A failure of either is a DISPOSITION FAILURE and enters the binomial.**

- **(S-a) reachable without side effects.** `python -c "import scale.<mod>"` exits 0, importing
  does not run the experiment (no file under `results/` created or modified by the import), and
  import wall clock is under 60 s.
- **(S-b) it does something checkable.** At least one of: (1) a live tracked python importer
  outside the section-1 exclusions; (2) runnable as an entry point (a `__main__` guard with a
  callable body, or `python -m`); (3) the journal the sheet names for it exists and holds records,
  or its readings reproduce per section 4's instrument.

FAILS on an import error, or when no leg of (S-b) holds. A wrong record COUNT with the journal
present and non-empty is an amendment and **does not enter the binomial**.

## 4. D-KEEP — a document classed KEEP

The draw contains six documents and they do NOT share one retention ground, so they are not
checked against one. Two are superseded archives kept as provenance
(`LOOP_PROMPT_ROUND3_ARCHIVE.md`, `done5.md`); four are live and kept because tracked files name
them (`NOTES.md`, `README.md`, `REQUIREMENTS.md`, `tests/deimos/DEIMOS_REPORT.md`). **Each row is
checked against the ground its own cell states.** Checking a live doc against a provenance ground
it never claimed would manufacture failures.

### The denominator gap is RESOLVED, and the resolution runs against iteration 2

Iteration 2 reported `workdone2.md` as "0 of 4 readings reproduce" against a sheet cell of 1/10,
could not explain the denominator, and amended the row anyway — the one undisputed KEEP failure of
that census. **The extractor was the wrong rule.** It counted numbers with >= 6 SIGNIFICANT
DIGITS. The census counts numbers with **>= 5 DECIMAL PLACES, deduplicated by value**.

Identified, not guessed: a 32-rule grid — {significant digits, decimal places} x {threshold 2..9}
x {deduplicated, with multiplicity} — scored against the **64 AUDIT.md rows carrying an
`N/M readings reproduce` cell that are NOT in this draw and are not `workdone2.md`**:

| rule | exact denominators |
|---|---|
| **decimals >= 5, deduplicated** | **54 / 64** |
| decimals >= 6, deduplicated | 32 / 64 |
| decimals >= 4, deduplicated | 21 / 64 |
| decimals >= 5, with multiplicity | 19 / 64 |
| significant digits >= 6, deduplicated — *what iteration 2 used* | 9 / 64 |

The iteration-2 calibration passed on `LOOP_PROMPT_ROUND6_ARCHIVE.md` (8/23) because both rules
return the same 23 readings on that one document. **A single-point calibration on the largest
available control is not an identification**, and that is the mechanism of the iteration-2 doc
failure — not carelessness about the number, but a control that could not discriminate.

**The residual is carried as a band, not waved away.** Ten of the 64 miss, all near: eight by ±1,
`PRIOR_ART.md` by +4, `scale/twodof.py` by −3. None of the ten has been touched since `f823b02`,
so the residual is in the rule, not in the documents. Therefore, pre-registered:

- a denominator disagreement of **≤ 4** is an AMENDMENT and **does not enter the binomial**;
- a denominator disagreement of **> 4** is a disagreement about method, is reported as a FINDING,
  and **does not enter the binomial either** — the denominator is an instrument property, and this
  protocol will not fail a row for a defect it has just documented in its own tool;
- the **NUMERATOR** carries no such caveat. It is an exact lookup against the pool of numeric
  leaves in `results/*.jsonl` at abs=5e-7.

`scale/doc_readings.py`, `--demo` green before this file was committed: 61/71 sheet denominators
exact, 71/71 within ±4.

### The legs

- **(D1) the provenance re-measures — PROVENANCE ROWS ONLY.** For a row whose cell states
  `N/M readings reproduce`, the NUMERATOR is re-measured with `scale/doc_readings.py`. **D1 is
  load-bearing for a row retained as "the provenance of published readings": if the numerator is
  0, the KEEP FAILS and enters the binomial**, because provenance of readings is the entire stated
  reason to retain a superseded document. For a row whose stated ground is citation rather than
  provenance, a numerator of 0 is an AMENDMENT and **does not enter the binomial** — a live
  document named by tracked files is not retained on its numbers.
- **(D2) the stated ground resolves.** For a provenance row: a tracked successor exists that names
  this file, or the document self-labels as the archive of a closed round. For a citation row: at
  least one tracked file outside the section-1 exclusions names the document or its stem. **A
  failure of D2 is a DISPOSITION FAILURE and enters the binomial** — it is the row's own stated
  reason to exist, and if it is empty the KEEP rests on nothing. A wrong citation COUNT with the
  count still >= 1 is an amendment and does not.
- **(D3) no citation drift into a surviving file.** Every `path:line` citation in the doc that
  points at a **tracked** file must be in range (the MISTAKES.md P-6 check). A citation to a file
  that no longer exists is an AMENDMENT, **not** a failure, and **does not enter the binomial**: a
  historical round log naming a file later deleted is an accurate record of its round, and
  demanding otherwise would forbid archives from describing history. Stated now, before
  measurement, because it is the leg most open to post-hoc convenience.

PASSES iff D1 (as scoped above) and D2.

## 5. ATTIC — "dead" means the stated reason re-measures

"It is dead because the sheet says so" is circular. Each ATTIC row states a reason; that reason is
re-measured from scratch, and the re-measurement uses the blind spot KILL 3 did not rule out —
FINDING 1's **constructed** journal path. `scale/bucket.py:45` builds `RESULTS / f"{name}.jsonl"`,
so a producer can be live while never writing the literal string. The orphan check is therefore:
importers, `results/` writes at ANY extension, AND resolution of every `Journal(NAME)` or bucket
name the module constructs against `results/{NAME}.jsonl`.

An ATTIC row is DEAD iff its stated conjunction re-measures true. **Any leg that fails makes the
ATTIC wrong; that is a FINDING and an AMENDMENT, and it DOES NOT ENTER THE BINOMIAL, which is
defined over KEEP failures only.** Stated here in full rather than by cross-reference.

## 6. The pre-registered verdict, computed at n = 30, not carried over

Iteration 2's "2 or more" was derived for a draw of 10 and is wrong at 30. Under H0 "the sheet is
at least 95% right", each drawn KEEP row fails independently with p <= 0.05, n = 30, so
**E[failures] = 1.50** and sd = 1.19. The exact binomial upper tail:

| reject at | alpha = P(X >= k \| p=0.05) |
|---|---|
| >= 2 | 0.4465 |
| >= 3 | 0.1878 |
| **>= 4** | **0.0608** |
| >= 5 | 0.0156 |

**THE THRESHOLD IS 4.**

- **0, 1, 2 or 3 KEEP failures** — the rows are AMENDED, the census STANDS.
- **4 or more KEEP failures** — the census is REJECTED at **p < 0.061** and it re-runs.

**Why 4 and not 3.** 0.0608 is the largest tail at or below the 0.09 this round has been running
at (iteration 2's own alpha was 0.0861), so the false-rejection rate is held roughly constant
while the budget triples. Three failures is 1.26 sd above a null mean of 1.5 and would be expected
in 19% of runs on a sheet that is exactly 95% right; rejecting there would reject correct sheets
once every five iterations.

**The direction of error this prefers, stated plainly.** It prefers letting a mildly-wrong sheet
stand over rejecting a right one. Power of the >= 4 rule: **0.35** against a sheet 90% right,
**0.68** against 85% right, **0.88** against 80% right. A sheet that is 90% right survives this
check about two times in three. That is the price of holding alpha at 0.06, it is a real cost, and
it is stated before the count is known rather than after.

The count is stated plainly whichever way it falls. **A fourth failure is not argued into being
one, and a fourth failure is not argued out of being one.**

## 7. The caveats this verdict carries

1. **Stratification does not dominate.** MARS measured that against a size-12 cluster placed
   adversarially inside the largest stratum (n=114, 36% of KEEP), uniform at matched budget 30
   detects with 0.7157 and stratified with 0.4951. The floor buys total rule COVERAGE —
   P(detect a wholly-wrong rule) = 1.0000 for all 61 — and that is the axis the round's measured
   history says fails. It does not buy the best power against an adversarial cluster.
2. **The urn is pinned, the tree is not.** The draw reads `AUDIT.md` at `f823b02`. Amendments made
   during this iteration cannot move the sample, which is the iteration-2 defect this fixes. But
   the FILES being measured are read from the working tree at HEAD, so a row's verdict is a
   verdict about the tree now, not about the tree at `f823b02`.
3. **R3 is exact on the forms measured in this tree and blind to a form invented after it.** A new
   declaration form will read as UNDECLARED and its red will FAIL. That error runs toward
   rejecting the sheet, which is the direction an audit instrument should fail in.
4. **The R3 demo pins two of its ground truths on `tests/foreman/test_topology_washout.py` and
   `tests/chase/test_structural_zero_guard.py`.** Neither is a drawn row. This was deliberate: an
   earlier draft pinned them on `tests/foreman/test_r1_settling.py`, which IS drawn, and was moved
   before this file was committed.
5. **The doc denominator band is fitted, not derived.** ±4 comes from the observed residual on 64
   held-out rows. It is not a bound; a document unlike those 64 could miss by more.
