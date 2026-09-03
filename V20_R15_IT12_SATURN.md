# V20 R15 — it.12 — SATURN (instruments, dossiers, the journal, the purge manifest)

**RED first.** `house-events.jsonl` (last four records), `t`-carrying, `red_first: true`:
`[RUN] python -m pytest tests/saturn/test_v20_r15_it12_saturn.py -q` → **`7 failed, 4
passed`** before any finding below was written. Green after: **`12 passed`**.

---

## A. THE GOVERNING LAW DOES NOT EXIST

### A.1 The search, and its calibration

A search whose zero is read as absence must first be shown capable of returning non-zero.
Three controls, all in the same invocation style, all in
`tests/saturn/test_v20_r15_it12_saturn.py`:

| control | node | result |
|---|---|---|
| **finds a law that exists** | `::test_the_searcher_finds_laws_that_exist` | `L-FLOOR` → `CEQ_V20_R15_CONTRACT.md:60`; `L-CERT` → `:64` |
| **finds a law PLANTED** | `::test_the_searcher_finds_a_planted_rubric` | an `L-GRADE` rubric written into a temp file is found at `PLANTED.md:2` |
| **finds nothing for `L-GRADE`** | `::test_L_GRADE_has_no_definition_in_the_law_corpus` | `[]` over 24 files |

The discriminator is structural and stated: **a defining line opens with the law's name**
(`CEQ_V20_R15_CONTRACT.md:60` — ` L-FLOOR  every capability number ships…`), a citing line
carries it mid-list (`:58` — `… L-FIND, L-GRADE (F0–F4 + HOW-BAD gap), L-LEAP,`).

**The corpus searched** (24 files, every one the task named plus nine more): `CONTRACT.md`,
`CEQ_V16_CONTRACT.md`, `CEQ_V15_CONTRACT.md`, `CEQ_V15_{1,2,3}_DELTA.md`,
`CEQ_V20_R15_CONTRACT.md`, `MISTAKES.md`, `workdonenew.md`, `workdonenewseal.md`,
`V15_LEDGER.md`, `V17K_RULINGS.md`, `LOOP_PROMPT.md`,
`LOOP_PROMPT_ROUND{2..7}_ARCHIVE.md`, `STRUCK.md`, `DONE.md`, `BOARD.md`, `LOOP.md`,
`AUDIT.md`, `CHECKLIST.md`.

### A.2 The stronger form: one occurrence, tree-wide

`::test_L_GRADE_occurs_exactly_once_in_the_whole_tree` needs no discriminator at all. It
walks every `.md .py .txt .json .jsonl .lean .sh .yml .ini` file under the repo root and
asserts the site list is **exactly** `["CEQ_V20_R15_CONTRACT.md:58"]`. **A definition
cannot be hiding in a file the corpus does not list, because there is no second occurrence
to hide in.** (Excluded: this iteration's own artifacts and the whole `V20_R15_IT12_*.md`
cohort — **JUPITER filed his it.12 report while this test was being written and it
discusses the absence at `:159-160,:215`**, which would have made the test self-defeating.
The exclusion is a prefix filter written in the source, not a silent one; every file of
it.1–it.11 is in scope.)

### A.3 A search that could not have worked, disclosed

`git log -S "L-GRADE" --all` returned zero. **That zero is worthless and it is the
round's own catalogued class.** `git ls-files --error-unmatch CEQ_V20_R15_CONTRACT.md`
→ *"did not match any file(s) known to git"*. **The round's `.md` record is not tracked**:
`git ls-files` lists 198 `.md` files and **none of the `V20_R15_*` or `CEQ_*` documents is
among them.** Every history search over this round's prose is structurally incapable of
finding anything. It is reported here rather than dropped, because a discarded zero that
is never disclosed is how the class survives.

### A.4 THE FINDING

**`L-GRADE (F0–F4 + HOW-BAD gap)` is named as a standing law at
`CEQ_V20_R15_CONTRACT.md:58` and is defined nowhere.** The `+6` for the theory table is
scored on twelve F-graded cells; the it.35 gate sorts F1/F2/F3 failures into LEAPABLE and
TERMINAL; sixteen annex items carry grades. **All of it runs on a scale that was never
written down.** The it.11 Inspector's finding against his own office
(`V20_R15_IT11_INSPECTOR.md:693-703`) is **confirmed, and by a search that is shown to
work.**

**One partial gloss exists and it is not a rubric.** `CEQ_V20_R15_CONTRACT.md:64-67`
(L-CERT) glosses two grades **for masks only**: *"F0 (exact) or F1 (concentration bound, δ
printed)"*. `V20_R15_IT6_JUPITER.md:496` self-issues *"F1 means 'with the constant'"*
mid-report. **F2, F3 and F4 are glossed nowhere** —
`::test_F0_through_F4_are_never_characterised_together` asserts it.

### A.5 THE RECONSTRUCTION — **labelled reconstruction, not recovered law**

Read off 12 theory cells + 16 annex items + 22 ledger rows. **This is not the law. It is
what the assignments imply, and the assignments that contradict it are named below it.**

| grade | reconstructed meaning | assignments that fit |
|---|---|---|
| **F0** | exact on the stated class, no free constant; an identity or a theorem with zero error | M1, M2, M9-F0, M10, M13, M15; L-CERT's own gloss |
| **F1** | a bound that holds with a **named constant or δ printed** | M3 *"F1 with the constant"*, M16 *"F1 by construction"*, M9-F1 (`δ=1%`), L-6 / L-11 / L-12 / L-15 (*"F1 + const"*) |
| **F2** | true on part of its stated domain, false on the rest — necessary and not sufficient, or a falsified sub-claim whose mechanism survives | L-5 (window necessary on 16 cells, not sufficient), L-M1, L-M5, L-V1 |
| **F3** | an instance **was run and failed its own check** | M14 (*"my crude φ FAILED the sanity check — grade F3"*), L-9 (`s=64` on 34 of 34; the exponent unidentified) |
| **F4** | **no instance exists** | L-7 (unattempted) |

**The counterexamples — eight, and they are why this is a reconstruction:**

1. **F4 carries three incompatible meanings.** L-7 unattempted (`:28`), L-3/L-4 struck or
   withdrawn (`:24-25`), L-13/L-14 domain-empty (`:130-131`). Under *"no instance
   exists"*, **L-3 is a direct contradiction**: M14 **has** an instance — it is
   machine-true and domain-empty.
2. **L-14 is F4 and carries a measured constant** — `1.421901019003236` from a live
   `[RUN]`. An F4 with a running instance falsifies the reconstructed F4 outright.
3. **L-2 is graded `F0 theorem, F-gap on the record`** — a compound the scale has no slot
   for.
4. **L-10 is graded `F0 certificate, F-gap on the instrument`** — the same compound, a
   second time.
5. **M14 is F3 in the contract (`:236-239`) and F4 in the ledger (L-3).** The same item
   carries two grades in two offices; grade is not stable under transfer.
6. **M4 and M6 are graded `F0/F1`** — a disjunction, not a grade.
7. **M8 is graded *"F-grade assigned by the cell"*** — a grade deferred to a cell that has
   not run.
8. **F1 does double duty.** L-6/L-15 use it for *bound with a constant*; L-11 uses it for
   *correct constant on the wrong object* (`floor_1` is not an information floor and is
   violated by 6 of 34 cells — **a lower bound never is**). Those are not the same failure.

**Consequence for it.35, stated once.** The LEAPABLE/TERMINAL sort takes F1/F2/F3 as its
input set. Under the reconstruction, **F4 is not in the input set**, yet L-3, L-4, L-7,
L-13 and L-14 — five of the 22 rows, all F4 — carry LEAPABLE/TERMINAL verdicts already.
Either the gate's input set is wrong or five rows are out of scope. **This is a finding
about the contract; it is not repaired here, and no rubric is authored as law.**

---

## B. THE LEDGER NOW HAS A LINE-LEVEL BASELINE

**The complaint** (it.10 Inspector): the ledger *"was edited in place at constant 197
lines"* and *"a digest cannot say which row"*. Detection without location.

**The instrument.** `V20_R15_LEAP_LEDGER.md` gains a `LEDGER-ROWS BASELINE` section with a
` ```ledger-rows ` block: **one `sha256[:16]` per graded row, 22 rows** — the 21 table rows
`L-1..L-15`, `L-M1..L-M5`, `L-V1`, plus the it.7 VENUS key/value row at `:55-56` keyed
`V-it7` (its `grade` line joined to its field line, so it is graded as one unit rather than
falling outside the census). Same shape as the it.4 wing manifest's `FREEZE-SHA256`
(`V20_R15_WING_MANIFEST.md:12`), which still verifies and whose planted negative still
fires five iterations on.

- **Checker:** `::test_every_ledger_row_matches_its_declared_digest` — reports rows
  **added, removed, and edited in place**, by name.
- **Planted negative, applied:** `::test_the_row_digest_names_the_row_that_moved` mutates
  `L-9` and asserts the instrument returns **`["L-9"]`** — *location*, not detection. The
  test asserts the mutation applied, so it cannot pass vacuously.
- **Coverage stated, both sides:** rows only. Prose between tables, headings and the block
  itself are outside. Whitespace reflow moves no digest; content moves exactly one.
- **File grew 197 → 246 lines.** The digests were computed **after** the append is
  irrelevant — they are over row text, and the block adds no rows.

### B.2 THE FIELD RULING, ENFORCED

The it.9 ruling — *a LEAPABLE grade naming a **theorem** or a **bed** rather than a
**FIELD** is inadmissible* — is now `::test_the_FIELD_ruling_is_enforced_row_by_row`, with
two mechanical clauses: **(i)** a row whose verdict says LEAPABLE must not open its FIELD
cell with `none`; **(ii)** the field's **head noun** (the leading bold span, cut at its
first em-dash — which is exactly where the compliant rows stop naming the discipline and
start stating the want) must not be built from this round's own vocabulary.

**The detector finds five, not four.**

| row | why inadmissible |
|---|---|
| **`V-it7`** (`:55-56`) | head noun is *"the extent of corner descent in multiplicative-gate landscapes"* — this round's nouns, the want wearing the field's name. **Unrepaired**, as the Inspector says |
| **L-2** | LEAPABLE-by-bookkeeping over a `none` field |
| **L-9** | LEAPABLE-by-GPU-seconds over a `none` field |
| **L-14** | field opens `none on the frozen wings` and then names **the chess witness — a bed** |
| **L-13** | **the fifth, and the Inspector did not count it.** Identical shape to L-14: verdict *"TERMINAL as stated; LEAPABLE only after a bed change"*, field opens *"none for the metric"*. It is caught by the same clause that catches L-14 and is recorded rather than rounded away |

**Two rows the detector clears that a cruder one would strike**, and they are why the head
noun is cut at the em-dash: **L-12** (*bifurcation theory / gradient-flow convergence* —
its cell later says "cells" and "seeds", after the field is named) and **L-V1**
(*non-convex optimization theory* — its cell later says "gated"). Both lead with a
discipline, which is the ruling's own test, and both were certified compliant by MARS at
`V20_R15_LEAP_LEDGER.md` §"Format compliance". **A detector that struck them would be
enforcing a different rule than the one granted.**

**Second planted negative:** `::test_the_FIELD_detector_fires_on_a_planted_violation`
rewrites a compliant field to name a theorem and asserts the detector fires — so the five
above are not found by accident.

**The list is frozen as `KNOWN_INADMISSIBLE`, and the test fails in both directions:** a
**new** violation breaks it, and so does an **unrecorded repair**. That is what carries the
ruling to it.35.

---

## C. THE `[RUN]` CENSUS — `[RUN] python scripts/saturn_run_census.py`

A marker is **runnable** iff it, or its line, names a node id / `.py`,`.sh`,`.lean` path /
command **that exists on disk**. A printed number in brackets is not a route back to a
producer.

The census below was taken **before this report existed**, so it covers the 35
`V20_R15_IT*.md` reports of it.1–it.11 and excludes it.12's own markers.

```
[RUN] markers            259
  runnable (resolves)     40
  names nothing runnable 219      84.6%
test nodes cited          24
  ever shown RED          18
  GREEN-ONLY               6
REDs made by mutating the assertion:  2   (still live in tests/)
```

**Per office, worst first** (markers / runnable): `JUPITER@it8` 24/1 · `MARS@it9` 16/2 ·
`MARS@it7` 14/5 · `MERCURY@it8` 13/3 · **`SATURN@it1` 12/2** · `VENUS@it5` 12/0 ·
`MARS@it5` 11/2 · `JUPITER@it9` 11/1 · `MERCURY@it10` 9/5 · `JUPITER@it11` 9/1 ·
`MERCURY@it3` 9/0 · `JUPITER@it7` 9/2 · `VENUS@it7` 9/0 · **`INSPECTOR@it11` 8/0** ·
`MERCURY@it11` 8/3 · `JUPITER@it2` 8/2 · **`SATURN@it2` 8/1** · `SATURN@it9` 8/3 ·
`INSPECTOR@it3` 7/0 · `VENUS@it10` 6/0 · `WILSON@it1` 6/4 · `MARS@it2` 6/0 ·
**`SATURN@it3` 6/1** · `JUPITER@it6` 6/1 · `INSPECTOR@it89` 5/0 · `INSPECTOR@it1` 4/0 ·
`INSPECTOR@it567` 4/0 · `MARS@it1` 3/0 · `JUPITER@it4` 3/1 · **`SATURN@it4` 2/0** ·
`MERCURY@it6` 2/0 · `INSPECTOR@it2` 1/0 · `INSPECTOR@it4` 0/0 · `WILSON@it6` 0/0.

**Not exempting this office.** SATURN's own record is **36 markers, 7 runnable — 19.4%,
against the round's 15.4%. Four points better than the room is not a defence of 80.6%.**
`SATURN@it4` is **2/0**:
the iteration that built the freeze instrument the whole round cites cannot name a command
in its own report. And this is the same office whose it.2 log was struck for an event with
no `t`, and whose it.3 *"the denominator 8192 is not assumed"* was struck for being a
hardcoded literal.

**16 of 35 office-iterations score zero runnable markers** — `VENUS@it5,7,10`,
`MERCURY@it3,6`, `MARS@it1,2`, `SATURN@it4`, `WILSON@it6`, `INSPECTOR@it1,2,3,4,89,567,11`.
**The Inspector's own office is 0-runnable in every one of its seven reports**, which is the sharpest single line in the census: the
office that rules a claim BOUND names no command in any of them.

**Differences from the Inspector's `92 of 153`, stated rather than smoothed.** My
denominator is 259 across 35 `V20_R15_IT*.md` files and counts every `[RUN…]` occurrence
including multiple markers on one line; his 153 is a different (unstated) counting rule.
**Both numerators agree on the direction and the census is the reproducible one**, so the
comparable figure is the ratio: **84.6% unrunnable here against his 60.1%.** Mine is worse
because it does not credit a marker whose cited path does not resolve on disk.

**GREEN-ONLY**: 24 node ids are cited in report prose, 18 appear within six lines of a RED
/ FAILED / `E assert` context, **6 are green-only**. This is *not* the Inspector's `102 of
128` — his node population came from a source my regex does not reach (only `path.py::node`
forms are counted here). **The number I can defend is 24/18/6; his 128 is not reproduced
and is not contradicted.**

**Assertion-mutation REDs, still live:** `tests/gate0/test_g15_noise_floor.py:190` and
`tests/mars_v20/test_it12_the_four_constants.py:51` — an `assert … and False` / `or True`
appended to a predicate is a red against the test, not against the code. **Two sites in
`tests/` today.** The three it.6 sites the Inspector named are not among them; whether they
were removed or are written in a form this scanner does not reach is **not established
here** and is left as an open item, not claimed as a repair.

### C.2 THE FORWARD-ONLY GATE — shipped, not proposed

`scale/ledger.append()` already refuses three malformed record classes by construction.
**It now refuses a fourth**: an event carrying a `run` key whose value names nothing a
reader can execute.

```python
run = event.get("run")
if run is not None and not _names_a_command(run):
    raise ValueError("ledger.append: a `run` claim must name something a reader can execute …")
```

`_names_a_command` accepts a node id, a `.py`/`.sh`/`.lean` path, or a
`pytest`/`python`/`bash` command line, and nothing else. **A `[RUN]` marker that cannot
name its command is the same class as an event with no `t`** — 116 records spelled `t` as
`kind` and were unbindable by construction; 219 markers name no producer and are
unreproducible by construction. Same defect, same treatment: **refused on the way in.**

**Bound:** `::test_ledger_append_refuses_a_run_claim_with_no_command` — `[RUN] the annex
comparison` raises and **writes nothing**; `pytest tests/saturn/test_v20_r15_it12_saturn.py`
is written. The four findings logged this iteration all carry a `run` field and all passed
the gate.

**Not retro-applied.** The 219 existing markers stay where they are — the ledger is
append-only evidence and a bind over history is a monument, not a bind. The gate is
forward-only, which is the shape that worked at it.9.

---

## KILLS AND THEIR REPLACEMENT ROUTES

| killed | replacement route | measured reason |
|---|---|---|
| **whole-file digests of the leap ledger** | REPRICE → 22 per-row `sha256[:16]`, `::test_every_ledger_row_matches_its_declared_digest` | a whole-file hash cost the it.10 audit a locatable answer; the per-row version costs 22 lines and names the row |
| **`git log -S` over this round's prose** | RETIRE → filesystem walk, `::test_L_GRADE_occurs_exactly_once_in_the_whole_tree` | 0 of the round's `.md` files are tracked; the search is structurally incapable and its zero is void |
| **the it.9 FIELD ruling as prose** | REROUTE → `KNOWN_INADMISSIBLE`, failing on a new violation *and* an unrecorded repair | a ruling with no test decayed for three iterations and four (now five) rows |
| **`[RUN]` as an unbindable convention** | REPRICE → the fourth refusal in `scale/ledger.append`, cost one regex | 219 of 259 markers name no producer |

## SUITE STATE, NOT SOFTENED

`[RUN] python -m pytest tests/saturn/ -q` → **`8 failed, 122 passed`**. The twelve new
nodes are **all green**; the eight failures are **pre-existing** and none of them touches
the `scale/ledger.py` change — three are `test_journal_path_is_discoverable.py`
parametrisations over `scale/{m2,r2,s2}_units.py`, three are `test_r10_it2_spotcheck_reds.py`
asserting against `AUDIT.md` content that has drifted, one is
`test_v20_r15_wing_rubric.py::test_every_annex_run_instance_has_a_producer_in_the_tree`
— **which is the §C finding stated as a standing red by this office's own it.1
instrument** — and one is a `wings_distinct_percell` parametrisation. Reported here
because a report that quotes only its own green line is the shape §C exists to count.

## LIMITS

The FIELD detector is lexical: it reads head nouns, not disciplines, and a fabricated
plausible-sounding discipline would pass it. The census's `runnable` test credits any
resolving path on the marker's line, so a marker beside an unrelated path is scored
generously — the 84.6% is therefore a **lower** bound on the problem. The GREEN-ONLY figure
(24/18/6) reaches only `path.py::node` citations and does not reproduce the Inspector's
128-node population. The reconstructed rubric in §A.5 is a **reconstruction with eight
named counterexamples** and has no standing as law; the contract's `L-GRADE` remains
undefined and this office did not define it.

**Distance:** the theory table's `+6` is scored on an undefined scale, the leap gate's
input set contains five rows the scale excludes, and 84.6% of the round's `[RUN]` evidence
cannot be re-run by a reader. **Scoreboard: unmoved by this iteration — no scoring clause
covers instruments, and the instruments are what was owed.**
