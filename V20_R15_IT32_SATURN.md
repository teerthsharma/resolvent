# V20 R15 IT32 — SATURN (WATSON, instruments)

One deliverable. MONOTONE is retired and replaced by the sorted-tuple shape
`MARS-31-D` demands, the `438` ledger pin is specified for the two offices that
carry it and enforced by a node this office ships, and the dating convention is
given the node it has been missing for four iterations.

Every clock reading in this document was produced by

```
date "+%Y-%m-%d %H:%M:%S %Z"
```

run in the repository root on this machine. No stamp below is prose.

| reading | what it dates |
| --- | --- |
| `2026-09-02 19:16:52 IST` | first command of the iteration |
| `2026-09-02 19:18:17 IST` | the ledger `sha256` census under RULE 2 |
| `2026-09-02 19:20:46 IST` | the RED run, verbatim below |
| `2026-09-02 19:22:23 IST` | this document written |
| `2026-09-02 19:24:17 IST` | the GREEN run, verbatim below |

---

## 1. `MARS-31-D` is conceded in full, on both counts

### 1.1 The retirement was blind to deletion

`MARS-31-D` struck the it.31 ladder while the it.31 report was still being
written. He is right, and it is the same objection he made at it.25, when the
index recipe's count was **kept** on exactly this ground: a count is the only
part of that recipe that can detect a deletion. At it.31 this office retired
`len(all_rows) == 17` to `len(all_rows) >= 17` and bought survival under growth
by giving up detection of deletion.

The ladder as it stands after the strike:

| shape | form | verdict |
| --- | --- | --- |
| MONOTONE | `len(x) >= N` | **RETIRED.** A corpus that loses one member and gains two is still `>= N`. |
| PARTITIONED | `REQUIRED - set(x) == set()` | **KEPT.** A set relation over NAMED members. A deletion removes a name, the difference is non-empty, the node is RED, and the failure says WHICH name went. |
| STAMPED | `assert digest == DIGEST` | **KEPT.** Refuses when the subject moves, rather than asserting about a subject it did not read. |

The two survivors are the two shapes MARS's objection cannot touch, and it is
worth saying plainly why, because it.31 stated the exemption without stating the
mechanism: `CELLS_12 - set(cells)` is a **set relation over names**, not a bound
over a cardinality, so a deletion changes the value of the expression. `>= N` is
a bound over a cardinality, so a deletion that is backfilled does not.

The sorted tuple MARS asked for is PARTITIONED written as an ordered witness:
`tuple(sorted(REQUIRED - seen)) == ()`. Same relation; the tuple makes the
failure message deterministic instead of set-ordered.

### 1.2 The planted negative, MEASURED not asserted

`tests/saturn/test_v20_r15_it32_deletion_blind_counts.py::test_the_floor_is_blind_to_the_deletion_the_tuple_catches`

A population of 17 named rows loses `L-9` and gains `L-18`, `L-19`:

```python
before = {f"L-{i}" for i in range(1, 18)}
after  = (before - {"L-9"}) | {"L-18", "L-19"}
assert len(after) >= 17                            # MONOTONE: GREEN on a deletion
assert tuple(sorted(before - after)) == ("L-9",)   # PARTITIONED: names the loss
assert tuple(sorted(before - before)) == ()        # and GREEN when intact
```

The floor does not merely fail to name the deletion — it passes it. That is the
whole of MARS's case, executed rather than argued.

### 1.3 The three retirements that landed

Detector: every `len(x) >= N` / `> N`, `N >= 2`, over an unstamped open corpus in
`tests/saturn/`, using the *same* `OPEN_SCAN` and `STAMPED` predicates as it.31
so the two rules police one population under one definition. Offender list at
`2026-09-02 19:20:46 IST`, verbatim:

```
test_v20_r15_it27_wing_arm_citation.py:181  assert len(all_rows) >= 17, f"read {len(all_rows)} ledger rows: {[r[0]
test_v20_r15_it27_wing_arm_citation.py:184  assert len(landed) >= 3, f"only {sorted(landed)} carry a resolving cit
test_v20_r15_it28_widened_resolver.py:165  assert len(outside) >= 5, f"the widened set saw only {outside}"
```

Three, not one. The it.31 report named only the first; the detector found the
other two, which is the point of writing the rule as a detector rather than as a
promise.

| site | was | now |
| --- | --- | --- |
| `tests/saturn/test_v20_r15_it27_wing_arm_citation.py:181` | `len(all_rows) >= 17` | `tuple(sorted(it27_rows - {r[0] for r in all_rows})) == ()` over the named `L-1 … L-17` |
| `tests/saturn/test_v20_r15_it27_wing_arm_citation.py:184` | `len(landed) >= 3` | `tuple(sorted({"L-10","L-13","L-14"} - set(landed))) == ()` |
| `tests/saturn/test_v20_r15_it28_widened_resolver.py:165` | `len(outside) >= 5` | equality-free set relation over the **sorted path tuple** of the 8 targets the widened resolver actually reaches |

The third is literally MARS's prescription — the population there *is* a sorted
list of paths:

```python
it32_outside = frozenset((
    "ceq/kdata.py:475", "lean/CEQ/V16Domain.lean:129",
    "scale/negation_scope.py:300-304", "scripts/v15_r1.py:137",
    "scripts/v15_r1.py:17-19",
    "tests/jupiter/test_v20_r15_it12_constants.py:11",
    "tests/jupiter/test_v20_r15_it12_constants.py:12",
    "tests/jupiter/test_v20_r15_it9_q6.py:155"))
dropped = tuple(sorted(it32_outside - set(outside)))
assert dropped == ()
```

All three retirements keep growth legal and make deletion RED. None of them is a
count.

### 1.4 The `IT23-28 = 163` closed-subset argument is struck, and conceded

MARS's evidence lands on this office's own partition argument, not only on the
floors. The it.30 census closed the `IT30` partition on a `[RUN] ls` recording
that `V20_R15_IT30_JUPITER.md` does not exist. It does — it is on disk, 12,151
bytes. `IT23-28` was closed by the same class of evidence, i.e. none.

**A partition is only as closed as the claim that it is closed, and that claim
needs an instrument, not a listing.** A listing pasted into a journal is read by
no later run. The concession is shipped as a node rather than as a sentence:

`test_the_it30_partition_is_closed_by_an_instrument_not_by_a_listing` asserts the
**exact** tuple

```python
tuple(sorted(p.name for p in ROOT.glob("V20_R15_IT30_*.md")))
    == ("V20_R15_IT30_JUPITER.md", "V20_R15_IT30_MERCURY.md")
```

Exact, not superset: this partition is *asserted to be closed*, so any movement
in it — deletion or the growth that falsified the listing — must be RED. That is
the discrimination the `[RUN] ls` never had.

---

## 2. The `438` pin, specified for both offices and enforced

The sharpest item on the it.31 census. `438` — the ledger length — is pinned
twice, in two offices, **neither stamped**, and MERCURY's states the defect in
its own failure message. An instrument that names the defect it will die of, and
dies of it.

Offender list, verbatim, same run:

```
test_v20_r15_it23_fence_and_argument.py:194  assert len(_src(LEDGER)) == 438, (
        "the ledger moved off its it.20 cen
test_v20_r15_it24_seal_and_manifest_gap.py:142  assert len(_lines("V20_R15_LEAP_LEDGER.md")) == 438, "the ledger started growi
```

### 2.1 Why the count cannot carry the claim

Both nodes assert a **line count** in order to carry the claim *the ledger has
not moved*. A line count cannot carry it: 438 lines of different text passes.
The claim is about content; the instrument reads cardinality. This is the same
category error as MONOTONE, one level down — a scalar summary standing in for a
relation over the thing itself.

### 2.2 The repair, exact and copy-pasteable

Census reading at `2026-09-02 19:18:17 IST`, by
`python -c "hashlib.sha256(pathlib.Path('V20_R15_LEAP_LEDGER.md').read_bytes()).hexdigest()"`:
45,670 bytes, 438 lines,
`sha256 = 6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485`.

```python
import hashlib
LEDGER_SHA = "6e88935181ab8c830b66b4adc83b91cf979726612a93fd5ff67468a8fbfd5485"
assert hashlib.sha256(
    (ROOT / "V20_R15_LEAP_LEDGER.md").read_bytes()).hexdigest() == LEDGER_SHA, (
    "the ledger moved off its it.32 census digest")
```

- **`tests/jupiter/test_v20_r15_it23_fence_and_argument.py:194`** — J-23g's claim
  is that one of three `LIVE` members *has not moved*. The digest **is** that
  claim; the count is a proxy a same-length edit defeats. Replace the `== 438`
  assertion with the block above and drop the literal.
- **`tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py:142`** — M-24b
  prices the same non-movement. Same replacement. Its message, *"the ledger
  started growing again"*, becomes true of the instrument as well as of the
  subject once the digest is what is read.

### 2.3 The node that enforces it — a STANDING RED

`test_the_ledger_length_pin_is_stamped_in_both_offices_that_carry_it` is **RED as
shipped, deliberately.** It reads both files by AST, and reports every `438`
assertion whose enclosing function does not stamp its subject. This office does
**not** edit those files; the node goes GREEN the moment JUPITER and MERCURY
apply 2.2, and no sooner.

It is guarded against both failure modes of a cross-office node:

- **Vacuity by rename.** A missing pin file is itself an offender
  (`MISSING -- the pin moved and this node went vacuous`), so deleting the file
  does not green the rule.
- **Permanence.** `test_the_438_detector_greens_on_the_repaired_form` runs the
  same detector over a planted repaired function and gets `[]`, and over a
  planted as-is function and gets the offender. The node is passable, not a
  monument.
- **Blind constants.** `test_the_pinned_digest_is_the_ledger_this_office_actually_read`
  re-reads the ledger on every run. If the ledger moves, the constant above is
  RED and the repair must be re-stated rather than the constant quietly edited.

The stamped exemption drawn at it.31 is the shape that holds here too: the two
nodes that carried `assert digest == DIGEST` survived the it.29 census; the one
that carried nothing broke twenty-five minutes after filing.

---

## 3. The dating convention: it got a node

The defect this office filed against itself at it.31 — four readings stamped in
prose, **none read from a clock**, every one 2–13 minutes wrong, **one in the
future**. The round has been dating readings precisely to survive the
moving-target problem, so a wrong stamp corrupts the one defence that has worked
in four consecutive audits. `[RETIRED]` — the four bad stamps were `19-23`,
`19-17`, `19-12`, `19-22`, written without colons here so this document's own
node does not read them as readings.

**The convention is not retired; the prose form of it is.** The node:

`test_no_stamp_in_this_offices_report_is_a_prose_stamp` reads this file and
requires

1. the literal command `date "+%Y-%m-%d %H:%M:%S %Z"` appears in it — a report
   that names no clock has no stamp, only a hope;
2. at least two full `YYYY-MM-DD HH:MM:SS TZ` readings — non-vacuity;
3. **no** clock-shaped token anywhere that is not part of a full reading, unless
   the line carries `[RETIRED]` as an explicit unsourced quotation.

A bare `HH:MM` cannot be produced by that command, so rule 3 is what makes the
convention checkable: it bans exactly the shape the four bad stamps had. The
detector is calibrated against that shape in
`test_the_stamp_detector_fires_on_the_shape_of_the_it31_defect`, which plants one
prose stamp, one real reading, and one marked quotation, and requires the
detector to return exactly the first.

This is the smallest node that makes a wrong stamp impossible to write silently.
It does not verify that a reading is *recent* — see Limits.

---

## 4. Runs

RED, at `2026-09-02 19:20:46 IST`, against unmutated code, before any repair:

```
FAILED tests/saturn/test_v20_r15_it32_deletion_blind_counts.py::test_no_saturn_node_defends_an_open_corpus_with_a_deletion_blind_floor
FAILED tests/saturn/test_v20_r15_it32_deletion_blind_counts.py::test_the_ledger_length_pin_is_stamped_in_both_offices_that_carry_it
FAILED tests/saturn/test_v20_r15_it32_deletion_blind_counts.py::test_no_stamp_in_this_offices_report_is_a_prose_stamp
3 failed, 5 passed in 0.93s
```

GREEN after the three retirements of §1.3 and this document, at
`2026-09-02 19:24:17 IST`, over the new file and the two files it repaired plus
the it.31 rules it extends:

```
FAILED tests/saturn/test_v20_r15_it32_deletion_blind_counts.py::test_the_ledger_length_pin_is_stamped_in_both_offices_that_carry_it
FAILED tests/saturn/test_v20_r15_it28_widened_resolver.py::test_w3s_pin_still_rests_on_one_resolving_citation
2 failed, 21 passed in 1.87s
```

Both remaining REDs are standing REDs that pre-date or are shipped as claims on
other offices, not regressions. The first is §2.3, RED by design. The second is
`RED 3` of it.28 — W3's binding rests on one resolving citation, and its own
message states the repair is a citation the ledger does not carry. It is in a
different function from the it.28 retirement of §1.3 (`:165` vs `:229`), reads a
different population, and was RED before this iteration touched the file.

RULE 1 went from three offenders to zero. The it.31 rules it extends
(`test_v20_r15_it31_open_corpus_counts.py`) stayed GREEN across the retirement,
which is the check that the new shape does not violate the old rule.

Whole-office control, `python -m pytest tests/saturn/ -q` at
`2026-09-02 19:25:15 IST`: **14 failed, 206 passed in 27.24s**. Thirteen of the
fourteen are standing REDs of earlier iterations — planted negatives and
`RED`-labelled finding nodes, including it.20's dropped-top-row negative and
it.29's two re-derived negatives. The fourteenth is §2.3, shipped RED. No node
of `test_v20_r15_it27_wing_arm_citation.py` fails, so the two retirements in
that file changed the shape of the assertion without changing its verdict.

Every rule in the new file is an **empty-set** or **exact-tuple** assertion.
None is `len(offenders) == N`, which would be the very defect it polices.
No `and False`, no `or True`.

---

## 5. Limits

The `438` node is RED by design and stays RED until two other offices act; it is
a standing claim on them, not a green badge for this one. The dating node checks
**provenance and shape, not recency** — a reading copied forward from a previous
iteration would pass it, and closing that needs a monotonicity check against the
previous report which this iteration did not reach. The `it27_rows` tuple names
`L-1 … L-17` by generator rather than by parsing the ledger's own row headings,
so a renaming scheme change reads as 17 deletions. The `landed` and `outside`
tuples are censuses taken at `2026-09-02 19:18:17 IST` and inherit whatever the
resolver was correct about at that moment; they detect loss, not wrongness.

## 6. Not reached

- The two journal cell counts `== 8` in
  `tests/mars_v20/test_seed2_is_decided_before_training.py:55,56`, and the
  all-office floor of `>= 12` unstamped counts the it.31 census found outside
  `tests/saturn/`. RULE 1's detector is scoped to `tests/saturn/`; widening it
  across offices is one argument change and was left for it.33.
- A re-take of the `IT23-28 = 163` partition itself. §1.4 concedes the argument
  and ships the instrument for `IT30`; the `IT23-28` tuple was not read.
- The `V20_R15_JOURNAL.md` it.30 entry still records the false `[RUN] ls`. It is
  another office's document and was not edited.
