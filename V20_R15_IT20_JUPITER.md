# V20 R15 it.20 — JUPITER (MYCROFT, annex owner)

**The landing instrument's coverage, raised from 30 to 116 of 129, with the 13 it
refuses named and the reason it refuses them.**

Instrument: `tests/jupiter/test_v20_r15_it20_citation_freeze.py` (498 lines, 10 tests).

---

## 1. THE NUMBER, UNROUNDED

| | citations | how certified |
|---|---|---|
| census population at it.20 | **129** occurrences / **103** unique | measured, not assumed |
| opened at their line this iteration | **129 of 129** | four readers, in bulk, line returned verbatim |
| **SCORED and LANDING on a frozen anchor** | **116 of 129** | `test_every_scored_citation_lands_on_its_frozen_anchor` |
| **REFUSED under RULING J-20b** | **13 of 129** | into files this round is still writing |
| failing among the scored | **0 of 116** | — |
| coverage before this iteration | 30 of 123 (MARS's live RED at it.18) | — |

**116, not 129.** Every one of the 129 was opened; 13 of them are into files that
changed length *while this census was being taken*, and under the ruling below a
pointer into a moving file is not evidence. It is not counted. The 13 are listed
by cid in `WITHDRAWN` with the repair each needs.

The three refused files and their measured motion inside this one iteration:

| file | at first take | at second take | occurrences refused |
|---|---|---|---|
| `V20_R15_JOURNAL.md` | 3890 | 3908 | 2 |
| `V20_R15_LEAP_LEDGER.md` | 406 | 438 | 10 |
| `house-events.jsonl` | append-only | append-only | 1 |

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it20_citation_freeze.py -q`
→ `10 passed`.**

---

## 2. THE CONTROL, AND IT IS RED AT HEAD WITHOUT ANY MUTATION

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it17_citation_landing.py
tests/jupiter/test_v20_r15_it18_citation_landing.py -q`
→ `2 failed, 10 passed`**, unmutated tree, no `git stash`, verbatim:

```
FAILED tests/jupiter/test_v20_r15_it18_citation_landing.py::test_every_true_location_carries_what_the_table_claims
FAILED tests/jupiter/test_v20_r15_it18_citation_landing.py::test_the_three_repaired_pointers_are_present_in_the_table
E           assert 'THE FREEZE' in ''
E            +  where '' = line_at('V20_R15_JOURNAL.md', 650)
E       Left contains one more item: "C17: V20_R15_JOURNAL.md:650 does not carry 'THE FREEZE'"
```

This is the INSPECTOR's `15:59 green / 16:11 RED` observation, still standing at
it.20, on the same pointer, with the it.18 test file byte-identical. It is not a
defect introduced here and it is not repaired here. It is the evidence for §4.

---

## 3. RULING J-20a — THE WANT FREEZE

The INSPECTOR's finding, taken as ruled: a landing check cannot detect a repair
aimed at the wrong claim, because `want` is supplied by the repairer in the same
edit as the pointer. This office's own `J-18f` is the same finding from the other side.

**Bound:** `want` is frozen at census. `CENSUS` is the ground truth. A manifest
whose `want` for a cid differs from `CENSUS[cid]` is **not scored as a repair** —
it is named as a **WITHDRAWAL plus a NEW CITATION**. `WANT_SEAL` is a SHA-256
over the frozen wants, so an in-place `want` edit is a two-line diff, not a
one-line one.

### The planted negative, verbatim

`test_the_freeze_refuses_a_repair_aimed_at_a_DIFFERENT_CLAIM`. C42 is
`lean/CEQ/V16Domain.lean:302`, anchored `theorem bedM_overlap_old_two`. The
mutation moves pointer and want together to `:304` /
`theorem bedM_overlap_new_two` — a real line, real text, and the **opposite**
theorem (`bedM.countP satOldTwo = 1` against `bedM.countP satNewTwo = 3`).

```python
    mutated = dict(CENSUS)
    mutated[42] = ("lean/CEQ/V16Domain.lean", 304, "theorem bedM_overlap_new_two")

    assert not_landing(mutated) == [], (
        "the LANDING check must stay GREEN on the wrong-claim repair -- that is "
        "the defect being demonstrated, not a bug in the negative"
    )
    assert not_a_repair(mutated) == [
        "C42: want moved from 'theorem bedM_overlap_old_two' to "
        "'theorem bedM_overlap_new_two' -- WITHDRAWAL + NEW CITATION under J-20a, "
        "not a repair"
    ], "the freeze scored a wrong-claim repair as a repair"
```

The negative asserts the it.17/it.18 instrument **stays green** on the wrong-claim
repair, and that the freeze names it anyway. That green is the defect, held in
place so the fix has something to be measured against.

**LIMIT, stated once and not repeated:** the freeze is tamper-evident, not
tamper-proof. `CENSUS` and `WANT_SEAL` sit in one file written by the office
under audit; a determined office can recompute the seal. What it buys is that a
claim can no longer be altered *silently, inside a repair*. That is all it buys.
Only a second office re-taking the census closes the rest.

---

## 4. RULING J-20b — THIS OFFICE'S `J-18a` WAS UNDER-CLAIMED, AND THE ROUND WAS RIGHT

`J-18a` said census **totals** expire. That under-claims it. **Every pointer into
a file still being written expires**, and the journal grows at its head every
iteration.

The round offered two fixes and required one to be picked and bound. **The second
is bound, and the iteration widened it:**

> The instrument records each cited file's line count at census time and refuses
> to score any pointer whose file has changed length — **and it refuses every
> pointer into a file declared LIVE outright, without waiting for the growth.**

The widening is not a preference. It is measured. The first take of the
instrument recorded `V20_R15_JOURNAL.md` at 3890 and refused one pointer; minutes
later the same instrument, byte-identical, read 3908. The second take then read
`V20_R15_LEAP_LEDGER.md` at 438 against the 406 it had recorded four minutes
earlier. **Two more instances of the INSPECTOR's mid-audit drift, in two more
files, inside one 20-minute iteration.** Recording a length for a file the round
appends to every iteration only defers the refusal by one append — so those files
get no length recorded and no pointer into them is scored at all.

**Why not the first option.** Citing prose by heading — the fix the CORRECTIONS
INDEX already adopted for itself at `P-6` — is the correct *repair* for each of
the 13, and it is written into each `WITHDRAWN` entry as such. It is not adopted
as the *instrument's rule* because the instrument cannot check a heading
pointer's LINE, and line is the whole subject of this instrument.

"Refuses" is precise: the 13 leave the denominator and are reported as needing
re-anchoring. They are not marked failing. Each of their anchors was opened and
verified correct at this census. **They are refused for their file, not for their
content**, and `test_every_refusal_is_a_growing_file_and_not_a_typo` asserts that.

---

## 5. WHAT THIS DOES NOT REACH

* **13 of 129 are un-scored.** Re-anchoring them by heading is real work and is
  not done. Until it is, the it.35 leap reads a table 13 of whose citations this
  instrument declines to certify.
* **An anchor is a location, not a proof.** It certifies the cited line carries
  the words the cell points at. It does not certify the cell's argument.
* **Length changes, not in-place edits.** A file rewritten without changing its
  line count passes J-20b and is caught only by its anchors.
* **Concentration.** 21 of 129 occurrences are into `scripts/v15_r1.py`, 18 into
  `lean/CEQ/V16Domain.lean`. One rigid shift in either moves a sixth of the
  census at once — it.17's finding, and no resolvability check can see it.
* **The it.18 RED is left standing.** C17 is not repaired here; repairing it by
  digit is exactly the move J-20b rules expires on the next append, and it
  already did.
* **No `git` writes. Nothing touched Kaggle.** Two files added to the working
  tree: the instrument and this report.
