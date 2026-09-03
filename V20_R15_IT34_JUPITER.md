# V20 R15 — it.34 — JUPITER (MYCROFT, annex owner)

**The census is keyed by a digest over the citation set, the theory table's radius is
enumerated and re-derived, and `M-33a` is closed by deleting the parenthesis at the
address the citation resolves to. `MARS-33-C` is UPHELD against this office.**

Window opened `2026-09-02T14:17:05Z`, closed `2026-09-02T14:26Z`, every stamp read from
`date -u` on this box, branch `v17k-gate0`, tree at `207e7b9`. **No git writes. Nothing
touched Kaggle.** Two files written: `tests/jupiter/test_v20_r15_it34_keyed_census.py`
(new) and one line of `V20_R15_IT13_MERCURY.md`.

---

## 0. THE BASELINE, RE-TAKEN INSIDE THIS WINDOW — `MARS-33-D` IS UPHELD

`[RUN] 2026-09-02T14:17:49Z → 14:18:04Z`, `python -m pytest $(RADIUS) -q`, unmutated
tree: **`15 RED / 91 GREEN` in 12.24s.**

`RADIUS_BASELINE = (9, 97)` is stamped `2026-09-02T19:05 IST` — **a timezone `date -u`
cannot emit**, which is MARS's finding and it is correct. The it.32 comparison
`16 − 9 = 7` was therefore **a difference of two readings taken by two offices in two
windows.** Against a reading of the same command in this window the delta is
**`15 − 16 = −1`**, and the three nodes it.33 could not attribute are inside that
artefact, not residue in the tree. **`RADIUS_BASELINE` is not re-declared here** — it is
another iteration's row and correcting it in place is the defect this iteration is
about. It is *dated*, and superseded by the paired reading in §3.

## 1. `P-33a` — THE CENSUS KEYED BY A DIGEST OVER THE CITATION SET

`tests/jupiter/test_v20_r15_it34_keyed_census.py`, **RULING J-34a**.

**RED first, verbatim, against unmutated code**, `[RUN] 2026-09-02T14:22:14Z`:

```
FAILED ...it34_keyed_census.py::test_the_current_citation_set_is_a_DECLARED_STATE
FAILED ...it34_keyed_census.py::test_every_count_is_LOOKED_UP_BY_KEY_AND_NOT_FROZEN_IN_THE_ASSERTION
FAILED ...it34_keyed_census.py::test_the_table_radius_is_complete_and_every_member_reads_the_table
FAILED ...it34_keyed_census.py::test_a_radius_is_a_property_of_an_EDIT_not_of_an_OFFICE
4 failed, 3 passed in 0.58s
```

GREEN, same command, `[RUN] 2026-09-02T14:23:23Z`: **`7 passed in 0.38s`**.

**The mechanism.** `census_key()` is `sha256[:16]` over the sorted **multiset** of
`path:spec` occurrences in `V20_R15_THEORY_TABLE.md`. A multiset, not a set, because
`V20_R15_IT13_MERCURY.md:146` is cited twice (`:184` and `:354`) and losing one of the
two must move the key. Every count this office publishes — population, unique, files,
extension histogram, notation split — is then **looked up in `STATES` by that key**
rather than written into an assertion.

**The declared state, `[RUN] 2026-09-02T14:23:09Z`:**

| key | population | unique | files | ext | notation `(:N, :A-B, :*)` |
|---|---|---|---|---|---|
| `4e935cc9db3dbd5a` | **133** | 106 | 39 | `md 41, py 72, lean 19, jsonl 1` | `(122, 3, 8)` |

`STATES` is **append-only**. The it.24 row — `131`, `{'md': 40, 'py': 71, …}`,
`(120, 3, 8)` — is neither deleted nor corrected: **it was the state at its own key.**
That is the whole difference a key buys. *Nothing is ever wrong; things are dated.*

**Why this is the repair for the self-breaking remedy.** `M-30a` (*no point without a
pointer*) and `M-30b` (*name the callers*) each **add** a citation. The it.20 census
**freezes** the citation count. Seven of the eight it.32 breaks were those two lines
arriving at a frozen number. Under a key, an added pointer selects a **row this ledger
has not declared**, and the instrument answers with `diff_occurrences()`, which returns
the pointer by name.

**The planted negative names what it removes**, and writes to no file in the tree:

```
test_an_added_pointer_is_a_NEW_STATE_and_the_diff_NAMES_it
  occ + [("ceq/beds/bed_k.py", "999")]
    (a) census_key(plus) != census_key(occ)     -- fails if the key stops covering the set
    (b) census_key(plus) not in STATES          -- an undeclared state is not declared
    (c) diff == (["ceq/beds/bed_k.py:999"], [])  -- the ANSWER is the line, not a number
    (d) population(plus) == population(occ) + 1  -- the count-shaped answer, carrying nothing
```

And the it.32 breakage is reduced to its actual content by
`test_the_SEVEN_it32_breaks_reduce_to_a_TWO_LINE_DIFF`: remove
`V20_R15_IT13_MERCURY.md:146` and `scripts/v20_m14_cheeger.py:347` from the current set
and the diff against the current set is **exactly those two strings, nothing removed**.
**Seven RED nodes, two lines of finding.**

## 2. `P-33b` — THE THEORY TABLE'S RADIUS, ENUMERATED

**RULING J-34b — A RADIUS IS A PROPERTY OF AN EDIT, NOT OF AN OFFICE.**

`RADIUS` is, by its own comment, *the 14 files that read `V20_R15_JOURNAL.md`*. The it.32
edit wrote the journal **and** the theory table under that one declaration. **No office
has ever enumerated the table's readers.** `TABLE_RADIUS` does, and it is **23 test
files** — 9 jupiter, 2 mars, 8 mercury, 4 saturn.

| | files |
|---|---|
| journal `RADIUS` | 14 |
| `TABLE_RADIUS` | **23** |
| in both | **9** |
| table-only — **what the it.32 declaration did not cover** | **14** |
| union — the actual radius of the it.32 edit | **28** |

**This file is in its own radius**, because it reads the table. A radius that excludes
the instrument declaring it cannot see its own edit — `J-31a` one level up.

**Unlike `RADIUS`, this list is re-derived at run time.**
`test_the_table_radius_is_complete_and_every_member_reads_the_table` globs
`tests/*/test_*.py`, keeps those containing `V20_R15_THEORY_TABLE`, and compares to the
frozen list, failing with `unlisted readers:` / `listed non-readers:` by name. A frozen
list nobody re-derives is a list that is wrong quietly; this one is wrong loudly. **It
found its own first defect during this iteration** — the RED above at
`(9, 13)`/`22` was this file omitting itself.

## 3. `M-33a` — CLOSED, AT THE ADDRESS THE CITATION RESOLVES TO

`V20_R15_THEORY_TABLE.md:184` and `:354` both cite `V20_R15_IT13_MERCURY.md:146` as the
authority for **band only, NO POINT**. That line read:

```
| **RETIRE the re-take (SATURN's split)** | **206 – 537** (point 309.0) | 4 — …
```

**The pointer lands and the target asserts what the citing line withdraws.** MERCURY's
own diagnosis is the one applied: *the withdrawal was applied in the citing office and
never at the address the citation resolves to.* **So the edit is at the target**, one
parenthesis:

```
… | **206 – 537** (point WITHDRAWN it.32 under J-31c: band only, NO POINT) | …
```

**Withdrawn, not deleted** — the address keeps the record of what it retired, which is
the `C132`/`:241-244` shape from it.26 (*a region that contains its own supersession
notice*).

**No re-issue is owed and the census does not move.** `C27`'s frozen want is
`RETIRE the re-take`, in the first column, untouched; the line count of
`V20_R15_IT13_MERCURY.md` is **283 before and after** (`wc -l`, `[RUN] 2026-09-02T14:26:07Z`), so `LIVE_FILES` does not see a
growth. **A citing-side edit would have been the cheaper diff and the wrong one:** it
would leave the next reader of `:146` reading a point.

**Paired reading, one command, one window** —
`python -m pytest $(TABLE_RADIUS) -q`:

| | `[RUN]` | RED / GREEN |
|---|---|---|
| before the edit | `2026-09-02T14:24:18Z → :27Z` | **33 / 134** (+2 xfail) |
| after the edit | `2026-09-02T14:23:35Z → :44Z` | **32 / 135** (+2 xfail) |

**`−1 RED`, and the node is named:**
`tests/mercury/…it33…::test_m33a_the_band_only_claim_cites_a_line_that_quotes_the_point`.
`TABLE_RADIUS_BASELINE = (32, 135)`. **This is a deviation, not a difference of
readings** — same command, same box, same window, one edit between them, and the
before-reading was taken by reverting the edit and restoring it, not by memory.

## 4. `MARS-33-C` / `J-32c` — **UPHELD AGAINST THIS OFFICE**

MARS is right and the heading-keyed reason given at it.32 is **struck**.

*"The index is ordered by row number, not by heading, so the heading key indexes nothing
there."* His run is the proof: this office's own
`test_a_chronological_cut_cannot_freeze_a_count_over_a_table_that_grows_at_the_top`
asserts `before == whole == 38` at every cut. **That equality is the demonstration that
the cut buys nothing on that corpus.** A frozen prefix equal to the whole is a frozen
count wearing the hat this office said comes off the same way — written by this office,
one iteration earlier, about someone else's number.

**And the second level is conceded too.** The structural cut keys on *what a line is*
(`INDEX_ROW_RE`), and MARS's objection — that `C20` documents three spellings of one
correction — is the same defect one level up: **a projection defined by syntax is a
projection that a re-spelling walks out of.**

**His alternative is the one shipped in §1 of this filing: hash the projection, do not
count lines outside it.** `census_key()` is that alternative for the table. **The index
is not converted here** — it is a second corpus, it is the coordinator's to write, and
converting it under this clock would be a second undeclared edit inside a radius this
office has just finished arguing must be declared first. **Named, priced, not reached.**

## 5. RULING — THE THREE POPULATIONS. **ONE RULE AT TWO WIDTHS, AND ONE OF THEM AT TWO DATES.**

MERCURY's unclosed gap, closed, **measured, not argued**
(`test_the_three_populations_reconcile_exactly`, GREEN):

```
133  =  122 plain `:N`  +  8 file `:*`  +  3 range `:A-B`
122  =  MERCURY-R1 over the same 443-line table  (71 lines carry them)
```

* **`122` and `133` are ONE rule at two ADMISSION WIDTHS.** MERCURY-R1 is *a backticked
  dotted path, colon, digits*. This office's `CITE_RE` is the same rule plus `:*`
  (`J-24a`, it.24) and `:A-B` (`J-24b`, it.24) — **two notations this office invented
  after her recipe was banked at it.22.** The residual is exactly `8 + 3`. The control
  in the test asserts the two rules never differ on a **path**, only on a notation.
* **`129` is this office's own rule at an earlier DATE** — `129` at it.20/it.24, `131`
  at it.27, `133` at it.32.

**They reconcile exactly and no renaming is owed. What is owed is a NAME.** A population
is a `(rule, date)` pair; **a bare integer is neither**, which is why three offices could
publish three integers under one word and all three be right. `STATES` keys by digest for
this reason: **`133` is not a fact about the table. It is a fact about a rule applied to
a state, and the digest names the state.**

Recommended, not applied: MERCURY's denominator stays `129` **as `MERCURY-R1 @ it.26`**,
and this office's is `4e935cc9db3dbd5a`. Neither is a number that can rot; both are
numbers that can be *dated*.

## 6. LIMITS

The `15 RED / 91 GREEN` journal-radius reading was taken before this iteration's edits
and **not re-taken after them**; the edit touched neither the journal nor the table, so
the prediction is `15 / 91` unchanged, and it is a prediction, not a reading. The
`TABLE_RADIUS` pair is post-hoc in one respect: the list did not exist before this
window, so its "before" reading is this window's list applied to the pre-edit tree, not a
list frozen earlier. `MERCURY_CENSUS_R1` digests the **current text of cited lines**, so
the `M-33a` edit moves it; it was already RED in the `14:17:49Z` baseline for other
reasons and is **not repaired here** — MERCURY's seal is MERCURY's to re-take. The
`STATES` ledger has exactly one row, so its append-only property is asserted by
construction and will not be *exercised* until a second state is declared. The it.29/it.30
count nodes stay RED and undated by this filing. SATURN wrote `V20_R15_IT34_SATURN.md`
and `tests/saturn/…it34_saturn.py` inside this window (`git status`, `14:26:07Z`); none of
the readings above were re-taken after they appeared. **Not reached:** converting the index
scan to a projection digest (§4), and re-taking the journal radius after the edit.
