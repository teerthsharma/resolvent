# V20 R15 it.24 — JUPITER (MYCROFT, annex)

Instrument: `tests/jupiter/test_v20_r15_it24_census_retake.py` (12 tests).
Repairs applied to `V20_R15_THEORY_TABLE.md` (10 sites), and the census re-taken
in the same iteration in `tests/jupiter/test_v20_r15_it20_citation_freeze.py`.
Carried forward: `it17`, `it18`, `it23`. No git writes. Nothing touched Kaggle.

Timer, dated: armed **11:31:03Z**, labelled "iteration 30" — the same arm this
office quoted at it.23, unmoved. First reading this iteration **11:46:25Z**, last
**11:58:17Z**. 20m wall from 11:46:25Z ends 12:06:25Z. Every number below carries
the reading that produced it.

---

## 0. THE CONSTRAINT THIS OFFICE NAMED, DISCHARGED

it.23 ruled on three argument-layer defects and applied none, for a stated
reason: each edit moves an occurrence out of the it.20 `CENSUS` and drops
`129 of 129` until the census is re-taken in the same iteration.

**A census that can never be updated is a census that will be abandoned.** The
cost is paid once here and the procedure is what is actually being shipped:
**EDIT, RE-TAKE, RE-DECLARE, IN ONE FILING.**

**RED first, verbatim, against unmutated code** — `[RUN] python -m pytest
tests/jupiter/test_v20_r15_it24_census_retake.py -q` at **11:51:27Z**, before any
edit to the table or the freeze:

```
tests\jupiter\test_v20_r15_it24_census_retake.py:77: in <module>
    from tests.jupiter.test_v20_r15_it20_citation_freeze import (
E   ImportError: cannot import name 'REISSUED' from 'tests.jupiter.test_v20_r15_it20_citation_freeze' (C:\Users\seal\Desktop\New folder (32)\tests\jupiter\test_v20_r15_it20_citation_freeze.py)
=========================== short test summary info ===========================
ERROR tests/jupiter/test_v20_r15_it24_census_retake.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.57s
```

The freeze had no withdrawal-with-re-issue record at all. That absence is the
reason it.23 could not act: the only withdrawal channel the round owned was
`WITHDRAWN`, which is J-20b's (a refusal *for the file*) and which it.21 reads to
auto-generate heading re-anchors. A J-20a withdrawal — a *changed claim* — had
nowhere to be filed. `REISSUED` is that place, and it is four lines.

---

## 1. THE CENSUS, RE-TAKEN AFTER THE EDITS

`[RUN]` at **11:55:23Z**, against the edited table, through the edited
instrument:

```
occurrences 129 unique 103 files 37
ext      {'md': 40, 'jsonl': 1, 'py': 69, 'lean': 19}
notation {'plain': 120, 'star': 8, 'range': 1}
scored occ 116
```

### **COVERAGE: `129 of 129`.** 107 by line + 1 by range + 8 by file + 13 by heading. 0 unscored, 0 failing.

**The histogram the brief predicted would move did not move.** `.md 40, .py 69,
.lean 19, .jsonl 1 = 129` — identical to the INSPECTOR's it.23 exhaustive count
[CITED: INSPECTOR, it.23], and identical to MERCURY's it.22 [CITED: MERCURY,
it.22]. The reason is worth more than the number: **a re-notation is not a
re-population.** Three citations were re-pointed and eight re-notated; none was
added and none removed. Ten edits, zero movement in the population.

**The histogram that DID move is by NOTATION, and it did not exist before this
iteration:**

| notation | count | scored by | J-20b applies |
|---|---|---|---|
| `path:N` | **120** | the line | yes |
| `path:A-B` | **1** | the range, every want in the tuple | yes |
| `path:*` | **8** | the file | **no — J-24a** |
| | **129** | | |

Both counts are asserted in `test_the_extension_histogram_is_republished_and_did_not_move`
and `test_the_histogram_that_DID_move_is_by_notation`, so neither can drift
silently the way the it.20 population did.

`WANT_SEAL` moved `fda4cdc2…` → **`538b5319…`**, which is J-20a working exactly
as designed: two wants left the census and two arrived, and the seal makes that a
two-line diff instead of a one-line one.

---

## 2. REPAIR 1 — `COMPOUND_HALF`. RULING J-24b.

Applied at `V20_R15_THEORY_TABLE.md`: `both bound @ …it12_constants.py:13`
becomes **`:13-14`**. `[RUN]` re-confirmed at 11:53Z: `+0.717647` at `:13`,
`−0.032353` at `:14`, and `−0.032353` is *not* on `:13`.

**The rule this office stated at it.23 is bound here — and the node that catches
the next one is the point.** A range pointer with a *single* want would have
re-created `COMPOUND_HALF` one notation wider: it would still be satisfied by the
half it can see. So the pointer widened **and the anchor became a pair**:

```python
131: ('tests/jupiter/test_v20_r15_it12_constants.py', '13-14', ('+0.717647', '-0.032353')),
```

`lands()` is tuple-aware; every member must appear inside the region. The planted
negative in `test_the_compound_claim_now_carries_both_halves_and_a_single_want_would_not`
builds the single-want range and asserts it **stays GREEN under the landing
check** — the defect demonstrated rather than described — and RED under the want
freeze. Under J-20a a changed want is a withdrawal, so **C63 is withdrawn and
re-issued as C131**. The freeze was not bent to let the repair through.

---

## 3. REPAIR 2 — `LINE_ONE_IDIOM`. RULING J-24a, AND THE INSTRUMENT NOW READS IT.

**8 occurrences across 6 files**, applied. Three offices counted 8/6
independently — this one, MERCURY at it.22, the INSPECTOR at it.23 — and the
edited table measures 8/6 again at 11:55Z.

`path:*` = the file, **not line-landed**. `path:1` = line 1, line-landed.
`CITE_RE` learned both notations in one character class, and `region(path, spec)`
replaces `line_at` inside the landing check.

**The consequence J-23f granted the notation without stating, and it is the
reason the notation is worth having: a `:*` pointer cannot be broken by an
append, so J-20b does not apply to it.** J-20b refuses a pointer into a growing
file because its LINE NUMBER goes stale; `:*` has no line number to go stale.
That is not a loophole — it is what a file-scoped citation is *for*, and
`refused()` now skips `*` explicitly rather than by accident.

Stated as a measurement, not a licence:
`test_a_star_pointer_survives_an_append_that_J_20b_would_refuse_a_digit_for`
prepends 40 lines to `ceq/arm_pl.py` in memory (writing to no file) and asserts
**the digit anchor breaks and the `:*` anchor holds**. If the digit anchor ever
stops breaking, the negative proves nothing and says so.

it.23's `test_the_file_pointer_idiom_is_eight_occurrences_across_six_files` is
carried forward as `…_is_RETIRED_at_it24_and_the_count_carried_over`: 0 by digit,
8 by `:*`, same 6 files. A relapse is RED.

---

## 4. REPAIR 3 — `C13` / `THREE_LINES_LOW`. RULING J-24c. THE THREE OFFICES, SETTLED.

`[RUN]` at 11:50Z, `grep -n F3 CEQ_V20_R15_CONTRACT.md` → **`139, 142, 241, 243,
244`**. `:242` reads *"an exact sweep-cut conductance; first task of the annex,
and"*.

| office | position | verdict |
|---|---|---|
| MERCURY [CITED: MERCURY, it.22] | the `F3` is at `:242` | **wrong, and the only wrong number of the three** |
| JUPITER (it.23) | no line carries `F3` uniquely → `REFUSED-PENDING-REISSUE` | **right disposition, wrong rule — STRUCK below** |
| INSPECTOR [CITED: INSPECTOR, it.23] | `grep` exact; fix is `:239 → :241` | **remedy ADOPTED** |

### The ruling: non-uniqueness does not block a digit pointer.

**And here is how that squares with this office's own resolver refusing
`'F4' appears on 4 lines in section '## it.7' … not 1`.**

**The resolver refuses non-uniqueness because uniqueness is its ADDRESSING
MECHANISM.** `resolve(path, key, want)` is handed no line number; it *derives*
one from the want. A want on four lines derives four answers, so it must refuse
or guess, and guessing is the fail-open MARS struck at it.22.

**A digit pointer is already addressed.** `:241` names the line. The want is
asked one question — does this line carry it — and uniqueness buys nothing.
**The it.20 landing check has never required uniqueness and no office has called
that a defect in four iterations**: C2's want `L-GRADE (F0–F4 + HOW-BAD gap)` is
not unique in its file either.

**it.23 imported an addressing constraint into a place where addressing was
already settled. That is the finding, and it is against this office.**

Both halves are pinned together in one test so neither can drift alone:
`test_non_uniqueness_does_not_block_a_DIGIT_pointer_and_the_resolver_still_refuses`
asserts (a) `F3` is on 5 lines of the contract, (b) the digit pointer at `:241`
is scored and lands, (c) the resolver handed `('## it.7', 'F4')` still returns
`REFUSED: … appears on 4 lines …`, with a resolving control so the refusal cannot
be an artefact of a dead section key. **Consistency between the two is now a
test, not a paragraph.**

### And C13 is STILL withdrawn and re-issued — for a different reason.

C13's frozen want is `M14 CHEEGER STRATIFICATION`. **`:241` does not carry it**
(`[RUN]`: `:239` carries it, `:241` does not). Moving a pointer to a line the
frozen want does not land on **requires a new want, and a new want is J-20a.**

So: **it.23 reached the right disposition by the wrong rule.** The disposition is
kept, the rule behind it is struck, and C13 is re-issued as

```python
130: ('CEQ_V20_R15_CONTRACT.md', 241, 'grade F3 pending'),
```

`grade F3 pending` are the words that carry **the cell's** claim — *the contract
grades M14 `F3` here*. `M14 CHEEGER STRATIFICATION` never did: it certified the
item's NAME while the cell asserted its GRADE, which is a **J-21c argument-layer
defect closed**, not a location repaired. `:239` carries `[V]` and no `F3`;
MERCURY saw that correctly and his remedy was one line off.

---

## 5. THE PROCEDURE, WHICH IS THE DURABLE PART

`REISSUED: dict[int, int] = {130: 13, 131: 63}` — new cid → the cid it replaces,
the same shape as it.21's `REANCHORS`. The withdrawn cid is **removed from
`CENSUS`**, so `not_a_repair` names it as uncensused if any manifest carries it
forward: a withdrawal is not an edit. `WITHDRAWN` is left alone — it is J-20b's
channel (refusal *for the file*), it.21 auto-generates heading re-anchors from
it, and overloading it would have manufactured two phantom `H*` anchors.

**Two offices' instruments were carried forward rather than left RED:**

* `it18`'s `CITE_RE` was a verbatim copy of it.20's. Left alone it read the
  population as **120 and would have called it 129**. It now carries the widened
  regex with the reason in a comment — the same one-place-for-the-rule move
  J-23c made for `lands()`.
* `it17`'s MANIFEST records what THIS office repaired at it.17 and is not
  edited. `SUPERSEDED` names the one pointer a later iteration widened, and
  asserts the *wider* pointer is in the table — so a lost citation still goes
  RED while a widened one does not.

**GREEN, `[RUN] python -m pytest tests/jupiter/ -q` at 11:58:17Z: `231 passed,
3 failed`** — and the 3 are exactly the three this office named as pre-existing
at it.23 (`it18_citation_landing` ×2, `it4_merge_is_unexercised` ×1), neither
touched nor claimed. Before this iteration: 3 failures. After 10 table edits and
a re-taken census: the same 3.

---

### MERCURY's own `COMPOUND_HALF` marker cannot go green on this repair, and the reason is a finding

`[RUN] python -m pytest tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py
tests/mercury/test_v20_r15_it22_independent_census.py -q` at **11:59:57Z** →
`1 failed, 13 passed`. The failure is
`test_m22a_compound_half_constants_13_does_not_carry_beta`, and it asserts

```
assert "-0.032353" in line_at("tests/jupiter/test_v20_r15_it12_constants.py", 13)
```

**That assertion reads only the constants file, which this iteration did not
touch — so it was RED before the repair and is RED after it, unchanged.** It is a
defect MARKER, not a regression, and it is mis-aimed: it demands that the FILE
move to match the citation, when the defect it names is that the CITATION does
not match the file. A marker written against the cited source can never be
discharged by repairing the pointer, which is the only repair available. **A
defect marker must assert against the thing the repair will change.** MERCURY's
finding was right and this office adopted it; his instrument for it cannot
observe its own remedy. Not repaired here — it is his file.

## Limits

The re-taken census is this office re-measuring its own freeze, which is the
tamper-evidence limit J-20a already bounded and this iteration does not improve:
`CENSUS`, `WANT_SEAL` and now `REISSUED` sit in one file written by the office
under audit, and only a second office re-taking the count closes that — the
INSPECTOR's it.23 bounded negative applies here in full, since a restated `[RUN]`
marker reproduces perfectly and re-execution cannot detect it. The two standing
`it18` C17 failures are a J-20b refusal in a superseded instrument and were not
repaired, only left un-widened. `it4_merge_is_unexercised` is untouched and
unattributed. MARS's STRIKE 1 (SATURN's `WING_ARM` pin) and STRIKE 3 (the stale
`31` in the INDEX-SHA256 sentence) are still untouched, three iterations running.
J-24a exempts `:*` from J-20b on the ground that an append cannot break it; **it
does not protect against a DELETION** — a `:*` want removed from its file fails
landing, which is correct, but no instrument distinguishes that from a want that
was never there. `region()` for a range joins the lines, so a want spanning a
line break inside a range would land; no cited range currently does, and nothing
checks that it stays true. The it.24 instrument is not sealed over its own
asserted counts.
