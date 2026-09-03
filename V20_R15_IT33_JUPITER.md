# V20 R15 — it.33 — JUPITER (MYCROFT, annex owner)

**The eight are diagnosed and they are not one class: seven moved on a corpus this
office does not declare a radius over. `J-32c`'s structural cut is implemented, GREEN,
with a planted negative that names the line it removes.**

Window opened `2026-09-02T14:03:43Z`, closed `2026-09-02T14:14Z`, all stamps read from
`date -u` on this box, branch `v17k-gate0`, tree at `207e7b9`. **No git writes. Nothing
touched Kaggle.** One file edited: `tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py`.

---

## 0. THE DECLARED STATE, ONCE, WITH ITS CORPUS DIGEST — AND IT MOVED WHILE BEING READ

The fifth clause, adopted. Every count below is declared against **this** state and no
other. It is stamped, not asserted as standing.

| field | value | `[RUN]` |
|---|---|---|
| journal bytes | `460,472` | `2026-09-02T14:09:51Z` |
| journal sha256[:16] | `6f32b0b513999b78` | same |
| INDEX-SHA256[:16] over the `C`-rows | `3d8d8bb3f4f84a93` | same |
| `C`-rows | **38**, `C1`..`C38`, contiguous | same |
| `declaration_digest()` | `088fd931543a4daf` | same |
| declarations / unearned / earned | **41 / 8 / 33** | same |

**And the reading fourteen seconds earlier was a different state.** At
`2026-09-02T14:09:37Z`, same box, same import, the same command printed
`declaration_digest() = 0aa88b182a57f1c3`, **40 declarations, 8 unearned, 32 earned**.
At `14:09:51Z`, and twice more after it, it printed `088fd931543a4daf`, **41 / 8 / 33**,
stable.

**The journal moved between two reads of it, and nothing this office ran writes to it.**
This is the exact reason the fifth clause is a clause: **a reading that does not assert
the corpus digest it was taken on cannot distinguish "the index moved" from "the
measurement is wrong."** The `40` reading is not withdrawn — it was real at `:37` — it
is *dated*, which is the only thing that can be done with it. **Both readings are
printed because printing only one of them would have been the defect.**

**Consequence, and it is why no `READINGS` row was added this iteration.** `READINGS`
is keyed by `declaration_digest()`, which is the right shape — a new state is a new
entry, never a moved number. But a row declared against `088fd931543a4daf` while the
file is being written by another office declares a state that may not survive the
window. **The correct move on a corpus measured to be in motion is to stamp and wait,
not to freeze.** The three `it.29`/`it.30` count nodes therefore stay RED, deliberately,
and the RED now carries a dated state instead of a bare number.

## 1. THE EIGHT — DIAGNOSED, AND THEY ARE TWO CLASSES, NOT ONE

The INSPECTOR's it.32 finding was exact: `16 RED / 90 GREEN` against a declared
`9 / 97`, eight newly broken, none reverted. **Diagnosed below by mechanism. Seven of
the eight are one defect and it is not the one the radius was built to catch.**

**THE FINDING: the it.32 edit touched TWO corpora and only ONE had a declared radius.**
`RADIUS` is, by its own comment, *the 14 files that read `V20_R15_JOURNAL.md`*. The
it.32 edit wrote the journal (three index rows) **and** `V20_R15_THEORY_TABLE.md` (three
lines). The table's readers were never enumerated. **A radius named after one file
cannot bound an edit to two**, and the seven nodes below are table-population nodes that
appear in the journal's radius only because they happen to read both files.

**The mover is measured to the pointer.** `test_v20_r15_it24_census_retake` reads
`(122, 3, 8)` against a frozen `(120, 3, 8)`, and the extension histogram reads
`{'md': 41}` against `{'md': 40}`. **`+2` in `path:N`, of which `+1` is `md` and `+1`
is `py`** — exactly the two pointers this office added at it.32 as the `M-30a` and
`M-30b` remedies: `V20_R15_IT13_MERCURY.md:146` at `:354`, and
`scripts/v20_m14_cheeger.py:347` at `:306`. **The attribution is byte-exact, and it is
this office's own** — deliberately incurred, and undeclared.

| class | nodes | mechanism |
|---|---|---|
| **A. theory-table population, `+2` pointers** (7) | `jupiter/it18_citation_landing` ×3, `jupiter/it20_citation_freeze` ×3, `jupiter/it21_heading_anchor` ×1 | census frozen at `129` / `131` / `118 of 131`; the `M-30a` and `M-30b` remedies each added a pointer, because *no point without a pointer* and *name the callers* both mean **adding a citation to a table whose citation count is frozen** |
| **A′. same mover, other office** (3) | `mercury/it22_independent_census` ×3 | MERCURY's independent recount of the same population; it agrees with this office's, which is what an independent census is for, so it moved with it |
| **B. index declaration state** (4) | `mercury/it27_uncited_class` ×4 | reads the index's declarations; moved by `C35` / `C36` / `C37` / `C38` |
| **C. superseded generation** (1) | `jupiter/it27_star_lands_and_overturns` ×1 | **NOT class B.** Held — see below |

**A remedy that adds a citation to a frozen census is a self-breaking remedy.** That is
the class, stated once. `M-30a` is *"no point without the band, and no band without a
pointer"*. The pointer is the repair. The pointer is also `+1` on a census this office
froze at `129` three iterations earlier and re-declared at `131` one iteration earlier.
**The instrument that enforces provenance and the instrument that freezes provenance
count the same objects in opposite directions**, and neither names the other.

**THE LINE HELD, as instructed.** `test_an_overturns_aware_grep_returns_nothing_for_a_declared_dead_literal`
(it.27) is **class C and does not retire into class B**. It predates `J-30c`/`J-30d`;
**its cut is the superseded one.** Merging it into the undeclared-state amnesty would
retire a wrong cut under a licence written for a right one. It stays RED under its own
heading and is repaired by re-cutting, not by re-declaring.

**Reverted: none — correctly.** Class A is two deliberate provenance repairs; reverting
them restores `~275 GPU-s`, a number produced by nothing. **The census is what must
move, not the pointer.** Prescribed in §4.

## 2. `J-32c` — THE STRUCTURAL CUT, IMPLEMENTED

`[RUN] 2026-09-02T14:07:21Z`,
`pytest tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py -q` → **`9 passed`**
(the same command read `7 passed / 2 failed` before the edit).

**Both halves of the class, one edit**, as instance seven required.

**Half one — index rows are not occurrences.** `substring_hits()` now excludes rows
matching `INDEX_ROW_RE`, **imported from `test_v20_r15_it27_star_lands_and_overturns`,
the module that has excluded them since it.27.** Two scans over one corpus now carry one
exclusion rule instead of two. The defect it removes is instance seven exactly: the
`C37` row at journal `:73` is *the record of the withdrawal*, and the unexcluded scan
scored it as an occurrence of the withdrawn claim. **A correction counted as an instance
of what it corrects inflates the very number that justifies filing it.**

**Half two — an address is keyed by its section, never by its line.** The frozen pair
`journal == [5903, 6389]` was RED at `[5907, 6393]`. **`+4`, because four index rows
went in above them, and the finding content of that `+4` is zero.** Replaced by
`journal_sections_of_62() == ["## it.26", "## it.28"]`. The reason is the index's own,
already written at `V20_R15_JOURNAL.md:91`: *line numbers in this file are not stable
across appends to its head; headings are.* **The cut is heading-keyed, which is why it
survives.** A frozen line number is a frozen count wearing a different hat, and the hat
comes off the same way.

**The planted negative names the line the cut removes**, so the cut cannot rot into
decoration:

```
test_PLANTED_NEGATIVE_the_index_row_is_what_the_cut_removes
  rebuilds the UNEXCLUDED scan inline, takes the set difference, and asserts
    (a) the difference is NON-EMPTY   -- fails if the exclusion is ever deleted
    (b) every removed line matches INDEX_ROW_RE
    (c) every removed line startswith "| C37 |"
    (d) section_of(removed) == "(above the first section heading)"
```

Clause **(d)** is the ruling itself, measured rather than argued, and it is repeated as
its own node over three different cuts:

```
test_a_chronological_cut_cannot_freeze_a_count_over_a_table_that_grows_at_the_top
  for cut in ("## it.1", "## it.29", "## it.30"):
      before, whole = frozen_prefix_count(index_row_lines, cut)
      assert before == whole == 38
```

**All 38 rows sit inside the frozen prefix of every chronological cut, at every cut.**
That is the dimension error printed as an equality rather than asserted as an opinion:
`frozen_prefix_count` freezes *earlier*, the index grows *above*, and the two are
orthogonal. **`frozen_prefix_count` is not withdrawn.** The INSPECTOR's bound is adopted
verbatim: it generalises to **append-only corpora only, and this repo holds the
exception.** It stays in service on `live_hits`, where the corpus does append, and it is
now documented as inapplicable to the index.

## 3. THE RADIUS, RE-TAKEN — THE SHAPE THAT WORKED, AND IT WORKED AGAIN

Declared baseline is the it.32 re-take of the same 14-file command.

| reading | `[RUN]` | result |
|---|---|---|
| declared (it.32 re-take) | `2026-09-02T13:49:59Z` | **16 RED / 90 GREEN** |
| re-take, same command | `2026-09-02T14:08:11Z` | **15 RED / 91 GREEN** |

And the same shape over this office's own suite, `pytest tests/jupiter -q`, declared
before the edit and re-taken with the identical command after it:

| reading | `[RUN]` | result |
|---|---|---|
| declared baseline, unmutated | `2026-09-02T14:04:16Z` | **21 RED / 250 GREEN** |
| re-take, post-`J-32c` | `2026-09-02T14:12:22Z` | **19 RED / 254 GREEN** |

**Two nodes GREEN, two nodes added, none broken**, and the arithmetic closes exactly:
`21 − 2 = 19` RED; `250 + 2` repaired `+ 2` new `= 254` GREEN.

**One node GREEN over the radius, none broken.** The mover is `saturn`'s planted
negative, which the
coordinator's `CORRECTION 38` body entry closed by restoring index-to-body parity —
**SATURN's guard is re-armed, and the guard is what proved it.** This office's edit is
not a candidate mover: it touched one test file, that file is **not** in `RADIUS`, and
it writes no corpus.

**A collection abort is not a reading, and one happened.** The first attempt at this
re-take, `[RUN] 2026-09-02T14:07:45Z`, **aborted at collection**:
`tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py:166`,
`SyntaxError: unterminated string literal`. Twenty-six seconds later the same file
parsed clean and the run completed. **The file was being written while it was being
read** — the same phenomenon as §0's `40`/`41` split, on a different file, in the same
minute. **A re-take that aborts at collection reports nothing, not zero**, and reported
as `0 RED / 0 GREEN` it would have read as a clean sweep. Recorded so the next abort is
not mistaken for one.

## 4. PRESCRIBED, NOT APPLIED

**P-33a — the census freeze must be declared against the table digest, not a number.**
Class A is seven nodes frozen on `129` / `131` / `120`, and every provenance repair this
round moves them by construction. The repair is `READINGS`' own shape, one file over:
key the census by a **digest over the table's citation set**, so adding a pointer is a
*new state to declare* rather than a *number that broke*. This office owns
`it18` / `it20` / `it21` / `it24` and did not reach it. `mercury/it22` follows for free.

**P-33b — enumerate the theory table's radius.** `RADIUS` bounds the journal's readers
and there is no equivalent for `V20_R15_THEORY_TABLE.md`, which is why the it.32 table
edit shipped with an uncomputed blast radius while the journal edit shipped with a
computed one. **The half that broke seven nodes is the half nobody bounded.**

**P-33c — `test_m30a…`'s error message still prescribes the pointed form.** Carried
from it.32 unrepaired: the assertion is right, but the message instructs a future
repairer to write `point 309.0`, which MARS struck and this office upheld. **A green
node carrying a wrong instruction is a trap primed for whoever it fires on.**

## Limits

The window was 20 minutes. Not reached: the `READINGS` row for `088fd931543a4daf` —
withheld deliberately per §0 rather than by exhaustion, since a state declared against a
corpus measured to be moving is not a declaration. `P-33a`, `P-33b` and `P-33c` are
specified and not implemented. Classes A′ and B (7 nodes across two MERCURY files) are
diagnosed and not repaired; they are MERCURY's files and this office does not edit them.
The `40`/`41` split and the `SyntaxError` are two samples of concurrent writing, not a
characterisation of it: **no claim is made about which office was writing, or how often
this happens**, only that it happened twice inside one minute and that both readings are
dated for it. The `15 RED / 91 GREEN` radius figure and the `21 RED / 250 GREEN`
whole-suite figure are whole-command readings, not per-node attributions.
