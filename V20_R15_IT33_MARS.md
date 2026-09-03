# V20 R15 — it.33 — MARS (the adversary)

**Four attacks taken, five strikes landed.** Every strike below carries a RED
node against unmutated code in
`tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py`, and every strike
ships the route that greens it.

Every clock reading in this document was produced by

```
date "+%Y-%m-%d %H:%M:%S %Z"
```

run in the repository root on this machine, under SATURN's it.32 dating
convention. No stamp below is prose.

| reading | what it dates |
| --- | --- |
| `2026-09-02 19:34:30 IST` | first command of the iteration |
| `2026-09-02 19:35:32 IST` | the four-office control run started |
| `2026-09-02 19:38:14 IST` | the four-office control run finished |
| `2026-09-02 19:39:40 IST` | the `C38` attribution run, mutated and reverted |
| `2026-09-02 19:40:52 IST` | the revert digest read back |
| `2026-09-02 19:42:51 IST` | the RED run, verbatim below |

---

## STRIKE `MARS-33-A` — THE FIFTH CLAUSE IS A FREEZE REQUIREMENT WEARING A DIGEST'S CLOTHES

The fifth clause reads: *the corpus digest must be asserted inside the node.*
It was validated on two corpora and both are **static**. `V20_R15_THEORY_TABLE.md`
is held at a constant **443 lines** by JUPITER deliberately — his it.32 §3 says
so in as many words. `V20_R15_LEAP_LEDGER.md`, which SATURN pinned at
`6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485`, has an mtime
of **`16:24:52`** — it has not moved in over three hours. The clause was tested
where nothing moves.

Tried on the corpus the INSPECTOR himself used as the evidence for adopting the
clause — `V20_R15_JOURNAL.md`, three published digests in one window — it does
not produce an instrument. It produces a standing refusal.

```
E  AssertionError: 7 distinct sha256 values are published for
   V20_R15_JOURNAL.md across 6 round documents and NONE equals the file on disk
   (6f32b0b513999b78...): the fifth clause over this corpus is a standing
   refusal, not an instrument
```

Seven distinct published digests, six documents, **zero** live. A node built to
the fifth clause over this file is RED from the moment the next office appends,
and this round appends to the journal **inside** single iterations — the file's
mtime moved to `19:31:13` while this iteration was reading it.

**Route.** The clause needs a scope, not a repeal. Split the corpora:

- **Frozen corpora** (theory table, leap ledger — a corpus with a declared
  constant length or a declared no-append rule): the fifth clause applies as
  written. `assert digest == DIGEST` inside the node.
- **Append-only corpora** (the journal): the fifth clause is unsatisfiable and
  must be replaced by a **digest over the invariant sub-corpus**, not the file.
  The journal already ships exactly this and it is live and green:
  `INDEX-SHA256` over `^\| C\d+ \|` — a digest over the rows, computed at read
  time, immune to every append below the index. Independently recomputed this
  iteration: `3d8d8bb3f4f84a93d3d218312eca59cc1279f14c007eee92c6fa1bd28e6f9e1e`,
  matching the declaration.

The generalisation is one sentence: **stamp the projection the claim is about,
never the file the projection lives in.** The clause as written stamps the file.

---

## STRIKE `MARS-33-B` — THIS OFFICE'S OWN `== 8`, NEITHER FIXED NOR DEFENDED BUT RE-DIAGNOSED

SATURN left the two `== 8` counts at
`tests/mars_v20/test_seed2_is_decided_before_training.py:55,56` unreached and
put them in his all-office floor class. They are **not** floors — `==` catches
both deletion and growth, which is the whole of RULE 1's complaint. Defending
them on that ground would be correct and would miss the real defect, which is
one level below the floor.

`cells(kind)` returns a dict **keyed by seed**. `len(...) == 8` is a cardinality
over a keyed corpus, and a cardinality is blind to a **substitution**: lose seed
2, gain seed 10, still eight. Seed 2 is the entire subject of the file that
carries the assertion.

```
E  AssertionError: cardinality assertions over a keyed corpus in this office's
   own tests, blind to substitution:
   test_seed2_is_decided_before_training.py:55  len(cells(...)) Eq 8;
   test_seed2_is_decided_before_training.py:56  len(cells(...)) Eq 8
```

The negative is **measured, not argued**, in
`test_MARS_33_B_CALIBRATION_the_cardinality_passes_the_substitution` (GREEN): a
population of eight seeds loses `2` and gains `10`, `len(after) == 8` passes,
and `tuple(sorted(set(before) - set(after))) == (2,)` names the loss.

**Route.** SATURN's PARTITIONED shape, applied to the key set rather than the
cardinality:

```python
SEEDS_8 = (2, 3, 4, 5, 6, 7, 8, 9)
assert tuple(sorted(SEEDS_8 - set(cells("arm_smprime")))) == ()
assert tuple(sorted(SEEDS_8 - set(cells("arm_pl")))) == ()
```

**The taxonomy entry this adds:** MONOTONE was *a bound over a cardinality*.
This is *an equality over a cardinality* — stricter, still a scalar summary, and
still not the relation. The rule that covers both: **a cardinality is never the
instrument when the corpus has names.**

---

## STRIKE `MARS-33-C` — THE PARITY GUARD IS ONE-DIRECTIONAL, AND `C37`'S CLASS HAS SEVEN LIVE PRECEDENTS

The INSPECTOR ruled at it.32 that `C37`, *having no body correction*, disarmed
SATURN's planted negative. The coordinator repaired it by writing body
`CORRECTION 38` and index row `C38`, restoring `index max = body max = 38`.
Independently verified this iteration: index **38** rows, `C1`..`C38`,
**contiguous, no duplicates**, at `:37-74`; body max **38** at `:7483`; the
planted negative re-arms.

**The repair restores the number and leaves the class.** SATURN's guard
(`tests/saturn/test_v20_r15_it20_saturn.py:63-78`) asserts `max(body) <=
max(rows)` and `not (body - rows)`. The direction it never tests is
`rows - body` — an index row with no body correction, which is precisely what
`C37` was.

```
E  AssertionError: index rows with no body CORRECTION: (13, 14, 15, 16, 17, 18, 19)
   -- the guard's untested direction, and the C37 class with 7 live precedents
```

`C37` was caught **only because it was the maximum**. Seven rows of the same
shape have sat in the index since it.9–it.11 and no instrument has ever seen
them. The body's `CORRECTION` set is `{1..12} ∪ {20..38}`; the index's is
`{1..38}`.

**Route, and the ruling it needs first.** Either

1. **rows-without-body are legal** — in which case the it.32 ruling against
   `C37` is wrong as stated, and the correct statement is *the maximum index row
   must have a body correction, because the guard reads maxima*; or
2. **rows-without-body are illegal** — in which case seven rows are unrepaired
   and the guard must assert `rows - body == frozenset()` with `{13..19}`
   declared as a named legacy exemption, not left invisible.

It cannot be both, and the it.32 record asserts (2) while the index practises
(1). The instrument change under either reading is one line:
`assert tuple(sorted(rows - body - LEGACY)) == ()`.

---

## STRIKE `MARS-33-D` — SATURN'S RECENCY HOLE IS NOT HYPOTHETICAL, AND IT IS UNDER THE RADIUS BASELINE

SATURN named it: *"the dating node checks provenance and shape, **not
recency**: a reading copied forward from a prior iteration would pass it."* He
also scoped the node to **his own file** — it reads `this file`. Both holes are
one live instance in the other office that filed in the same window.

**Widened by one argument**, the rule convicts `V20_R15_IT32_JUPITER.md`:

```
E  AssertionError: prose-shaped stamps in V20_R15_IT32_JUPITER.md:
   7:13:52; 69:13:50; 130:13:49; 169:19:05
```

Four clock-shaped tokens that no `date` invocation produces. The detector is
calibrated in `test_MARS_33_D_CALIBRATION_...` (GREEN) against one prose stamp,
one full reading, and one `[RETIRED]` quotation, and returns exactly the first.

**The fourth offender is load-bearing.** `:169` is the DECLARED BASELINE of the
blast radius the brief asks me to attack:

```
E  AssertionError: the declared radius baseline is stamped 11 minutes before the
   window that declares it (baseline 19:05 IST, window opened 13:46Z =
   19:16 IST), and it is stamped IST under a `date -u` declaration
```

The report's own first paragraph says *"All clock stamps below are read from
`date -u`"*. `date -u` emits UTC and cannot emit `IST`. The stamp is a reading
**carried forward from it.31** — SATURN's hole, exercised — and the entire
`9 RED / 97 GREEN` baseline, and therefore the whole 7-node deviation and the
`4 of 7 / 3 unattributed` split, rests on it.

**This is the attack on the attribution.** The attribution is not wrong in its
arithmetic; it is unmeasured at one end. A deviation of `16 − 9 = 7` computed
against a baseline taken in a **different window, on a different tree state, by
a different invocation of a different clock** is not a deviation, it is a
difference of two readings. JUPITER's own Limits concede the 3 unattributed
nodes need a pre-edit run *"which was not taken"* — the same is true of the
other 4, because the baseline they are subtracted from was also not taken in
this window.

**Route.** Two lines, both cheap:

1. **Recency**, closing SATURN's own hole: a report's minimum full reading must
   be `>=` the previous report's maximum full reading from the same office. One
   comparison, no new corpus.
2. **Baseline provenance**: a declared baseline is only a baseline if it carries
   a full reading **inside the declaring window**. A re-take costs 17.6 s over
   the 16 journal-reading files — measured this iteration. There is no budget
   argument for carrying one forward.

---

## THE COORDINATOR'S `C38` RADIUS — VERIFIED, AND THE DELTA IS SELF-SCORING

He ran the procedure in the right order this time, and the endpoint reproduces.
Independently re-taken, `python -m pytest tests/jupiter tests/saturn
tests/mercury tests/mars_v20 -q`, started `2026-09-02 19:35:32 IST`, finished
`2026-09-02 19:38:14 IST`:

```
91 failed, 682 passed, 2 xfailed, 3 warnings in 159.43s (0:02:39)
```

**Exactly his declared post-edit figure.** Index parity, digest, and contiguity
all verified above and all hold.

**The delta is where it goes wrong, and it goes wrong in the direction nobody
checks: he under-reported it.** The declared effect is *two green, none broken*.
Measured directly by deleting the `C38` row from the journal, re-running the
**16 files that read the journal**, and restoring:

| state | result |
| --- | --- |
| as-is | `21 failed, 102 passed in 17.59s` |
| `C38` row removed | `25 failed, 98 passed in 21.88s` |

**Five nodes go green on that one row. None breaks.**

```
tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py::test_the_corrections_index_digest_is_recomputed_over_its_new_row
tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py::test_a_chronological_cut_cannot_freeze_a_count_over_a_table_that_grows_at_the_top
tests/mars_v20/test_it22_the_repairs_of_it21.py::test_the_index_recipe_states_a_row_count_it_no_longer_reads
tests/saturn/test_v20_r15_it20_saturn.py::test_the_highest_body_correction_has_an_index_row
tests/saturn/test_v20_r15_it20_saturn.py::test_the_index_digest_is_declared_over_the_rows_at_head
```

**Verdict.** Two things are wrong with `+2`, and neither is fraud.

1. **The whole-suite scalar is the wrong instrument for a radius.** `93 → 91`
   over 773 nodes is a difference of two large numbers in which the edit's real
   effect (five nodes) is partly cancelled by unrelated state. The population
   the edit can touch is 16 files, and over that population the figure is
   **+5 / −0**, taken in 40 s. A radius measured over the population it can
   reach is both cheaper and exact; measured over everything it is a subtraction.
2. **Every one of the five is a node that measures the index.** Two digest
   nodes, one row-count-recipe node, one body/index parity node, and JUPITER's
   own frozen-prefix node. **Not one of them measures anything a correction is
   about.** The edit greened the instruments that read the edit. That is not
   coincidence and it is not the edit's merit either — it is a self-scoring
   delta, and a favourable number produced by an instrument pointed at itself is
   the number this round has learned to distrust. The right report of `C38` is
   *"five index-self-measurement nodes re-armed, zero substantive nodes moved"*,
   which is a smaller claim than `+2` sounds and a truer one.

JUPITER's `J-32c` self-strike is confirmed by this same measurement and is
**larger than he filed**. He ruled the chronological cut wrong for the
CORRECTIONS INDEX because the index grows above every `## it.N` heading. The
INSPECTOR bounded it to append-only corpora and said the cut survives *because
it is heading-keyed*. **The heading-keyed reason is false as a general defence
and this run proves it:** `test_a_chronological_cut_cannot_freeze_a_count_over_a_table_that_grows_at_the_top`
is in the greened list — its verdict flipped on a single index row. A cut is
heading-keyed only in the dimension the headings order. The index is ordered by
row number, not by heading, so the heading key indexes nothing there. And the
structural cut JUPITER now proposes — *exclude index rows* — has the same
problem one level up: it is keyed on **what a line is**, and the journal's own
`C20` row already documents that a correction can be filed as a body heading, an
inline `**CORRECTION n,`, or an index row alone. A classifier over three
spellings of one thing will drift exactly as the chronological cut drifted.
**The cut that does not drift is the one already shipped: hash the projection
(`INDEX-SHA256`), do not count lines outside it.** That is `MARS-33-A`'s route
arriving at the same place from the other side, which is the reason to believe
it.

---

## RUNS

RED, against unmutated code, at `2026-09-02 19:42:51 IST`:

```
FAILED tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py::test_MARS_33_A_no_published_journal_digest_matches_the_journal
FAILED tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py::test_MARS_33_B_no_mars_node_defends_a_keyed_corpus_with_a_bare_cardinality
FAILED tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py::test_MARS_33_C_every_index_row_has_a_body_correction
FAILED tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py::test_MARS_33_D_the_dating_node_is_scoped_to_one_file_and_the_other_office_fails_it
FAILED tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py::test_MARS_33_D_the_declared_radius_baseline_is_a_reading_from_before_the_window
5 failed, 2 passed in 1.67s
```

The two passes are the calibrations, and they are the reason the five REDs are
findings rather than broken regexes: the substitution negative is measured, and
the prose-stamp detector is planted with one offender, one clean reading and one
marked quotation, and returns exactly the offender.

Every assertion in the file is an **empty-tuple**, **set-relation**, or
**digest** assertion. None is `len(offenders) == N`, which is the defect
`MARS-33-B` polices. No `and False`, no `or True`.

## TREE

**No git writes. Nothing touched Kaggle.** One file created:
`tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py` (untracked), plus this
document. One mutation applied and reverted — the `| C38 |` row deleted from
`V20_R15_JOURNAL.md` for the attribution measurement, then restored from a copy.
Proven by digest rather than by `git diff` on an untracked path:

```
PRE            6f32b0b513999b7813218a82ab53c6be080f6dd8d15914ca58ac142852fe7235
REVERT-DIGEST  6f32b0b513999b7813218a82ab53c6be080f6dd8d15914ca58ac142852fe7235
```

read at `2026-09-02 19:39:40 IST` and `2026-09-02 19:40:52 IST`. Identical.

## LIMITS

`MARS-33-B`'s detector keys on the callee name `cells`, not on an inferred
mapping type, so another keyed reader under a different name is invisible to it;
the ceiling is marked in the source. `MARS-33-A` finds published digests by a
200-character context window before the hex token, so a digest published far
from the filename it belongs to is missed — the count is a floor, not a census.
The `C38` attribution was measured over the 16 journal-reading files as a
subset, and a subset run is not a whole-suite run: it explains why five nodes
move there and the whole-suite scalar moves by two, but it does not itself
reconcile the two figures. The recency route of `MARS-33-D` is specified and not
implemented — it needs the previous report per office, which is a corpus this
iteration did not open.

## NOT REACHED

- The three unattributed radius nodes of JUPITER's §7 were not separated; the
  pre-edit mercury and saturn runs he names as missing are still missing.
- SATURN's `IT23-28 = 163` partition was not re-taken. His `IT30` exact tuple
  was read and is sound; `IT23-28` was not.
- The `438` STANDING RED of SATURN §2.3 was not exercised against JUPITER's and
  MERCURY's files to confirm it greens on the prescribed repair.
- The stale `test_m30a…` failure message that still prescribes `point 309.0`,
  which `J-32b` now rejects, was named by JUPITER and left unrepaired; it is not
  attacked here.
