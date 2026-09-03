# V20 R15 — it.14 — SATURN (instruments, dossiers, the journal, the purge manifest)

**RED first, before any finding, repair or flag below was written.**
`house-events.jsonl`, `t="test"`, `red_first: true`, `run` naming an executable command:

```
[RUN] python -m pytest tests/saturn/test_v20_r15_it14_saturn.py -q   ->  6 failed, 9 passed
```

Green after: **17 passed.** Three suites together —
`[RUN] python -m pytest tests/saturn/test_v20_r15_it14_saturn.py tests/saturn/test_v20_r15_it12_saturn.py tests/saturn/test_v20_r15_freeze_manifest.py -q` → **48 passed.**
No git writes. Nothing touched Kaggle.

---

## A. THE `V-26` SWEEP — AND THE SIBLING IS IN THIS OFFICE'S OWN FREEZE

VENUS's rule, applied verbatim: *for a claim about a relation between two collections, the
falsifying mutation is the one that **preserves every per-collection statistic**.*

### A.1 THE STRIKE — `FREEZE-SHA256` clause (b) is two marginals

**The prose claim** (`V20_R15_WING_MANIFEST.md:23-25`, clause (b) *"EVIDENCE IN THE
LEDGER"*): *W1's evidence is `results/v17k_r4_retake.jsonl:161`; W3's is `:25`.* That is a
**relation between two collections** — the manifest's wing rows and the journalled records
in `results/`.

**What the it.4 nodes actually assert, each a function of one collection alone:**

| node | ranges over | what it computes |
|---|---|---|
| `::test_every_frozen_wing_is_found_and_not_merely_named` | `results/` **entire** | `journalled_cells(wing) > 0` — a **count** |
| `::test_the_found_wing_counter_reads_zero_for_the_named_wing` | `results/` entire | `journalled_cells("arm_phase") == 0` — a **count** |
| `::test_every_frozen_citation_resolves_at_head` | the cited file's **text** | anchor substring on the cited line |
| `::test_the_declared_hash_matches_the_rows` | the **manifest** | sha256 over sorted rows |

**Not one of them ranges over the pair.** Nothing anywhere asserted that the record on the
cited line is a record **of that wing**.

**The marginal-preserving mutation, and it is one field.** Exchange the `kind` of the two
clause-(b) records. `[RUN] ::test_the_it4_freeze_is_two_marginals_and_stays_green_under_the_joint_break`:

- **marginal 1 — every per-kind count bitwise identical.** `arm_smprime` and `arm_pl` counts
  unchanged; `arm_phase` still exactly `0`, so the it.4 calibration still passes.
- **marginal 2 — no anchor moved.** `0.9260818361` is still on line 161, `0.6446726192`
  still on line 25; every clause resolves.
- **marginal 3 — `FREEZE-SHA256` cannot move.** The manifest file is never opened by the
  mutation; the digest is over manifest rows.
- **the joint, and it is the only thing that moved:** line 161 now journals `arm_pl` and
  line 25 `arm_smprime`. **The manifest certifies W1 by W3's record, and every it.4 node is
  green.**

**This is `V-26`, not `D4`.** There is no tolerance here to tighten, and no mutation of the
*code* exposes it — `journalled_cells` and `resolve` compute exactly what they claim. It
survived ten iterations of citation as the round's tamper-evident instrument.

### A.2 THE REPAIR, shipped inside the instrument it repairs

`tests/saturn/test_v20_r15_freeze_manifest.py::test_every_clause_b_line_journals_that_wings_own_arm`
— the cited line is parsed as JSON and its `kind` compared against the wing's arm name,
**derived from that wing's own clause-(a) module path** (`ceq/arm_smprime.py:144` → stem
`arm_smprime`) rather than written into a map, because a hardcoded map is a second place for
the manifest to be wrong. **`[RUN] pytest tests/saturn/test_v20_r15_freeze_manifest.py -q`
→ 19 passed** (was 18). `::test_the_clause_b_pairing_is_true_on_the_unmutated_tree` confirms
the pairing is **true today**, so the strike is against the instrument, not the data.

### A.3 THE CONTROL — this office's it.12 ledger baseline CLEARS

Same class of mutation, applied to the it.12 per-row digests: **exchange the bodies of
`L-13` and `L-14`, keeping their ids.** The multiset of row bodies is bitwise unchanged
(asserted in-node), so a checker comparing a **set** of body digests reports nothing.

`[RUN] ::test_the_it12_ledger_baseline_is_joint_and_fires_on_the_same_class_of_mutation` →
the checker returns **`["L-13", "L-14"]`**. It is joint because each digest is keyed by the
row id parsed **out of the row itself** and the digested text **contains** that id. **The
instrument that certifies *these rows are these rows* does range over the pair.** One of
this office's two freeze instruments is joint and one was two marginals; the difference is
that the ledger digest includes the key in the digested text and the manifest's clause (b)
never joined the wing to the record at all.

### A.4 THE SWEEP — every remaining node in this round that certifies a relation

| node / instrument | the relation claimed | joint or marginal | state under a marginal-preserving mutation |
|---|---|---|---|
| **it.4 `FREEZE-SHA256` clause (b)** | wing ↔ journalled record | **TWO MARGINALS** | **STAYS GREEN — struck, repaired A.2** |
| **it.4 clauses (a), (c), (d)** | wing ↔ source/kill/price line | joint per row | anchor is on the *cited* line; a swap of two rows moves the digest |
| **it.12 `LEDGER-ROWS BASELINE`** | row id ↔ row content | **JOINT** | fires, names both rows (A.3) |
| **it.12 `KNOWN_INADMISSIBLE`** | row ↔ its FIELD cell | joint per row | id and field come from one line; no swap preserves it |
| **it.14 `THEORY-SHA256`** | cell ↔ grade | **JOINT by construction** | `::test_the_theory_digest_is_joint_over_cell_and_row` swaps two cells' bodies and the digest moves |
| **VENUS `it.10:147` C14 pairing** | `arm_pl` seeds ↔ `softmax` seeds | **was two marginals** | **already struck and repaired by VENUS at it.13** |

**Limit, stated rather than hidden:** this sweep covers the nodes this office owns plus the
one VENUS already filed. It is **not** a tree-wide sweep of every office's suite — the
mutation has to be constructed per claim, and a generic scanner for "compares two counts"
would flag every legitimate count in `tests/`. **What is bound is a rule and three worked
instances, not a census.** The census is the cheap follow-on and it is named in §D.

### A.5 A SECOND DEFECT IN THIS OFFICE'S OWN it.12 FILE, found while auditing it

`test_the_declared_digest_covers_every_row_in_the_file` shipped an assertion ending in a
disjunct that is **true whenever the ledger has any row at all** — the `or True` class this
office's **own it.12 census counted in two other suites**, shipped inside the census's own
file. The left-hand comparison it guarded compared a row count against itself. **Removed**,
not repaired; the real coverage claim is the following line, and it now carries a message
naming the undigested and stale rows. Bound at
`::test_the_coverage_node_carries_no_vacuous_disjunct`, with
`::test_dropping_a_row_from_the_digest_block_is_caught` as its planted negative.

### A.6 A THIRD — `test_L_GRADE_occurs_exactly_once_in_the_whole_tree` decayed in two iterations

The it.12 node asserted `sites == ["CEQ_V20_R15_CONTRACT.md:58"]`. It is **red at HEAD**, on
`V20_R15_IT13_INSPECTOR.md`, `V20_R15_IT13_MERCURY.md`, `V20_R15_IT13_VENUS.md`,
`V20_R15_JOURNAL.md` and `V20_R15_THEORY_TABLE.md` — **fifteen new occurrences, every one a
report discussing the absence.** A raw occurrence count is not the property claimed. Rewritten
over the discriminator: **every occurrence in a law-bearing file must be the one citation, and
no line in a law-bearing file may define the law.** A later report citing `L-GRADE` can no
longer break it; a later report *defining* it must. **Same shape as the two above — an
instrument that was green for a reason adjacent to its claim.**

---

## B. THE FROZEN THEORY TABLE IS HASHED

`V20_R15_THEORY_TABLE.md` **landed during this iteration** (it did not exist when this file's
RED was logged; `::test_the_theory_table_is_filed_and_carries_twelve_cells` was one of the six
reds and is now green). Twelve cells read off §1's grade matrix — six question rows × two wing
columns, **the wing ids taken from the matrix header rather than written down.**

```theory-freeze
# sha256 over the sorted `Qn/Wm|cell` pairs of V20_R15_THEORY_TABLE.md §1.
# Frozen it.14 by SATURN over the table as filed at HEAD 207e7b9.
THEORY-SHA256 = 9989f0efabe1895a9772f650ba1c3bcb77db7d37550879f2fdc9e8e9f2c63057
Q1/W1 a2c3286f60ace993
Q1/W3 5c6dea9cb707c683
Q2/W1 75ab69f504ef9e24
Q2/W3 1eb8872d017f62d7
Q3/W1 e1569caad812293f
Q3/W3 1eb8872d017f62d7
Q4/W1 63f81989362491d3
Q4/W3 63f81989362491d3
Q5/W1 1eb8872d017f62d7
Q5/W3 1eb8872d017f62d7
Q6/W1 f53365e65ce4d925
Q6/W3 f53365e65ce4d925
```

**Checker:** `::test_the_declared_theory_digest_matches_the_table_at_head` — reports cells
**added, removed and edited in place, by cell name.**

**Planted negative, applied to the FILED table on disk and not simulated:**
`::test_the_planted_negative_is_applied_to_the_real_table_on_disk` copies
`V20_R15_THEORY_TABLE.md` to disk, rewrites `Q3/W1` from `F2` to `F1` **in the file**, and
asserts the whole-table digest moves **and** the instrument returns exactly `["Q3/W1"]`.
Location, not detection. The test asserts the edit applied, so it cannot pass vacuously.
A second calibration builds a twelve-cell table from scratch on disk and repeats the
exercise, so the instrument is shown to work on a table this office authored as well as on
JUPITER's.

**THE SCOPE, CARRIED FORWARD EXPLICITLY — this is the it.4 limitation and the reason the
wing manifest's guarantee has survived honest inspection:**

> **The digest covers the twelve cells, not the content of anything they cite.** Editing
> `ceq/arm_pl.py`, a `results/*.jsonl` record, a ledger row, or the table's own §0 prose moves
> **no** digest here. Whitespace reflow inside a cell moves none. Editing a **cell** moves
> exactly one. A digest over a list is a guarantee about the list.

**And the `V-26` self-audit of the instrument shipped this iteration**, before it is relied
on: `::test_the_theory_digest_is_joint_over_cell_and_row` exchanges two cells' bodies —
preserving the multiset of bodies exactly — and asserts the digest moves. **It is keyed by
cell, so it is joint.** An instrument built the week `V-26` was filed does not get to skip
its own test.

**Two duplicate digests are visible above and are not a defect:** `Q4/W1 = Q4/W3` and the
four `F1 + const` cells share a digest because the digest is over **cell text**, and those
cells carry identical text. The digest is keyed by cell id, so a *swap* still moves the
whole-table hash for any pair whose text differs; a swap between two textually identical
cells is a no-op on the table and is correctly invisible. **Stated because a reader
comparing the column would otherwise find it and wonder what was hidden.**

---

## C. THE LEDGER — FIVE INADMISSIBLE ROWS, SIX UNCONSUMABLE F4 ROWS, THREE GRADES FOR M14

### C.1 The five inadmissible rows are still there. Unrepaired at it.14.

`[RUN] ::test_the_five_inadmissible_FIELD_rows_are_still_unrepaired` → the detector returns
**`{V-it7, L-2, L-9, L-13, L-14}`**, unchanged from it.12. The FIELD ruling is now three
iterations old on `V-it7` and the rows are the it.35 gate's input.

### C.2 The arithmetic: **six**, not five, and a seventh object with no row at all

`[RUN] ::test_the_F4_rows_the_gate_cannot_consume_are_counted_by_a_detector`:

| row | grade | verdict the gate cannot consume |
|---|---|---|
| **L-3** | F4 | TERMINAL |
| **L-4** | F4 | LEAPABLE |
| **L-7** | F4 | LEAPABLE |
| **L-8** | F4 | TERMINAL for this round |
| **L-13** | F4 | TERMINAL as stated; LEAPABLE only after a bed change |
| **L-14** | F4 | TERMINAL for BED-M and BED-K; LEAPABLE only on the chess witness |

**Six of the 22 graded rows.** This office reported **five** at it.12 and omitted `L-8`; the
it.12 Inspector's recount is **correct and adopted**, and the number is now produced by a
**detector** rather than carried in prose — it fails on a new F4 row and on an unrecorded
repair. The gate takes **F1/F2/F3** (`CEQ_V20_R15_CONTRACT.md:142-144`); F4 is not in the
input set.

**Q2/W1 is the seventh F4 object and it has no ledger row at all.** Re-graded `F1 + const` →
`F4` at it.12 and carried into the theory table's §1 matrix at it.14 (`Q2/W1 = **F4**
(re-graded it.12)`), with **no row appended to this ledger.** The gate never sees it in
either direction. That is a **different defect** from the six — an absence rather than an
unconsumable grade — and it is bound at `::test_Q2_W1_carries_an_F4_grade_and_no_ledger_row_at_all`,
which goes red the moment a Q2/W1 row is filed and the arithmetic must be recounted.
**Answer to the question as asked: six today; seven the moment Q2/W1 is filed as a row.**

### C.3 `M14` — three grades in three files, and the owner is JUPITER

| file | line | grade |
|---|---|---|
| `CEQ_V20_R15_CONTRACT.md` | `:239-243` | **F3** — *"it stays F3 in the table until it passes"* |
| `V20_R15_LEAP_LEDGER.md` | `:24` (row `L-3`) | **F4** |
| `V20_R15_JOURNAL.md` | `:645` | **F1** — *"Annex: M14 at F1, HOW-BAD 108x"* |

**Which is which:** the contract's F3 is the *standing* grade (it is written as a condition —
F3 *until* an exact sweep-cut conductance passes); the ledger's F4 is a *post-strike*
re-grade after `V-25` (domain-empty); the journal's F1 is the *annex bookkeeping* grade that
feeds the `≥12 at F0/F1` bonus. **Three readings, three answers to whether M14 is gate input,
gate-excluded, or scoring credit.**

**Owner of the repair: JUPITER.** M14 is his annex item, the contract's F3 is his grade, and
`L-3` sits in his it.7 block. **This office does not rewrite another office's row.** The
contradiction is **flagged inside the ledger** — `V20_R15_LEAP_LEDGER.md`, section *"FLAG —
M14 CARRIES THREE GRADES IN THREE FILES"* — which is the handling the it.12 Inspector
approved for the `V-it7` violation. Bound at `::test_M14_carries_three_grades_in_three_files`,
which holds **all three** readings at once, so a repair in **any** file turns it red and
cannot land silently. A second flag records the six-row F4 arithmetic in the same place.

`::test_the_flag_moved_no_row_digest` re-verifies the it.12 `LEDGER-ROWS BASELINE` after the
append: **22 of 22 digests unmoved.** The flags add no `| **L-…** |` row, so the freeze holds
across this office's own edit to the file it froze.

---

## KILLS AND THEIR REPLACEMENT ROUTES

| killed | replacement route | measured reason |
|---|---|---|
| **it.4 clause (b) as a found-and-resolves pair of counts** | REPRICE → `::test_every_clause_b_line_journals_that_wings_own_arm`, arm name derived from clause (a) | a `kind` swap leaves every count and every anchor untouched; the instrument the round cites was blind for ten iterations |
| **`... or n_table > 0` in this office's it.12 coverage node** | RETIRE → the real coverage assertion, with a message naming undigested and stale rows | the disjunct is true whenever the ledger has one row; the census that counted this class shipped it |
| **`sites == [one citation]` as the L-GRADE tree-wide form** | REROUTE → *no defining line in a law-bearing file*, over the it.12 discriminator | fifteen new citing occurrences in two iterations, none of them a definition |
| **the theory table as an unfrozen leap input** | REPRICE → `THEORY-SHA256` + twelve keyed cell digests, planted negative applied on disk | it is read at it.15 and it.35 and nothing could say which cell had moved |

## SUITE STATE, NOT SOFTENED

`[RUN] python -m pytest tests/saturn/ -q` → **`8 failed, 140 passed`**. The seventeen new
nodes are green and the it.12 and it.4 files are green. The eight failures are the **same
eight pre-existing failures reported at it.12** — three `test_journal_path_is_discoverable.py`
parametrisations, three `test_r10_it2_spotcheck_reds.py` assertions against drifted
`AUDIT.md` content, `test_v20_r15_wing_rubric.py::test_every_annex_run_instance_has_a_producer_in_the_tree`
(this office's it.1 instrument standing red on the `[RUN]`-census finding), and one
`wings_distinct_percell` parametrisation. **None is touched by this iteration's changes.**

**The two `and False` / `or True` sites named at it.12 —
`tests/gate0/test_g15_noise_floor.py:190` and `tests/mars_v20/test_it12_the_four_constants.py:51`
— are still live.** They are other offices' files. A **third** site was found this iteration
**in this office's own file** and is repaired (§A.5). **The census that counted two counted
its own author at zero because it scanned for `and False`/`or True` and not for a disjunct
that is a live expression.**

## LIMITS

The `V-26` sweep is a rule plus three worked instances, not a tree-wide census; the mutation
must be constructed per claim and no generic scanner distinguishes a legitimate count from a
marginal standing in for a joint. The `THEORY-SHA256` covers the twelve §1 cells and **not**
the table's prose, its per-cell sub-tables, or the content of anything a cell cites — a
digest over a list guarantees the list. The joint clause-(b) repair asserts the record's
`kind`, which is a label in the journal: a record with the right `kind` and the wrong
provenance still passes, and that is a further joint nobody has instrumented. The M14 flag
records a contradiction and repairs nothing; the grade that survives is JUPITER's to name.
The six-row F4 arithmetic is computed against a scale that **still has no text** — the it.12
`L-GRADE` finding is BOUND and unrepaired, so "F4 is not in the gate's input set" is read off
the contract's clause, not off a rubric.

**Distance:** the leap's primary input is frozen and locatable for the first time; the gate's
input set still contains six rows the contract excludes and a seventh object with no row; and
the round's most-cited freeze instrument was, until this iteration, not asserting the thing it
was cited for. **Scoreboard: unmoved — no scoring clause covers instruments.**
