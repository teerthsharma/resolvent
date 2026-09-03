# V20 R15 it.20 — SATURN (WATSON). THREE INSTRUMENT REPAIRS, EACH WITH A PLANTED NEGATIVE

**Branch** `v17k-gate0`. **No git writes. Nothing touched Kaggle.** Every `[RUN]` below is a
command that was executed and whose output is quoted verbatim.

**One node file carries all three planted negatives:**
`tests/saturn/test_v20_r15_it20_saturn.py`.

```
[RUN] python -m pytest tests/saturn/test_v20_r15_it20_saturn.py -q
RED FIRST, before any repair:   3 failed, 3 passed in 0.58s
AFTER the three repairs:        6 passed in 0.40s
```

---

## REPAIR 1 — THE CORRECTIONS INDEX RAN 11 ROWS BEHIND ITS OWN BODY

**The INSPECTOR's strike measured it at `C19` against a body at `C29`. At HEAD it was worse:
the index held 20 rows and the body ran to `CORRECTION 31`.** `C21`–`C31` had no rows, so a
reader checking whether a claim had been corrected was told **no**, silently, in the trusting
direction — on the structure the round designated as its guard against exactly that.

### The check that was missing

`tests/saturn/test_v20_r15_it20_saturn.py::test_the_highest_body_correction_has_an_index_row`
asserts *the highest `C`-number in the body has a row in the index*, and the whole range with
it. **Without it the index falls behind on every iteration that writes a correction, which is
every iteration** — it fell behind at it.14 and stayed behind for six.

**RED first, verbatim:**

```
[RUN] python -m pytest tests/saturn/test_v20_r15_it20_saturn.py -q
FAILED ::test_the_highest_body_correction_has_an_index_row
E  AssertionError: the CORRECTIONS INDEX stops at C20 while the body runs to
   CORRECTION 31: a lookup on the round's own anti-staleness instrument, failing
   silently in the trusting direction
   missing rows: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
```

### What landed

**Eleven rows, `C21`–`C31`, one per body correction**, each naming the claim as filed, where
it was filed, what is true, and the office that forced it — `C21`/`C22` (it.14, on itself),
`C23` (it.15, the `V-7` grep), `C24`/`C25`/`C26` (it.16, three corrections *of* corrections),
`C27` (it.17, the INSPECTOR on the vacuous heartbeat), `C28` (it.17), `C29` (it.18, MERCURY
and JUPITER falsifying their own briefs from disk), `C30` (it.19, the coordinator), `C31`
(it.19, this office's file-size-as-completion).

**One thing is disclosed rather than repaired.** Index row `C20` (the M14 grade) and the
body's `### CORRECTION 20` (a digest published in an append-only body) are **different
findings sharing a number** — the two sequences ran independently until this extension joined
them. Rewriting a published row to realign them is an author amendment of `C4`'s class, so the
collision is stated in the index instead, directly under the table.

### The property that already worked is preserved

The index's published recompute command is untouched and still governs. **It moved from
`:68-70` to `:87-89`, which is the index's own `P-6` drift, disclosed in the same paragraph.**

```
[RUN] python -c "import re,hashlib,io;s=io.open('V20_R15_JOURNAL.md',encoding='utf-8').read();
      print(hashlib.sha256(chr(10).join(re.findall(r'^\| C\d+ \|.*$',s,flags=re.M)).encode()).hexdigest())"
8a886db50dc36c9ffd67e5464b43058025b1890199396fb3d47fac181fb62757
```

**`INDEX-SHA256 = 8a886db50dc36c9ffd67e5464b43058025b1890199396fb3d47fac181fb62757` over 31
rows.** Was `1ec530205a46fc530f58eb7d752a528bd70e1b010ec6b1d1fd5c576e28e731be` over 20.
**Recomputed and re-declared in the index block, never restated inside a body that cannot be
revised** — that is `CORRECTION 20`'s finding and it is obeyed here.

### PLANTED NEGATIVE — fires

`::test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught` deletes the highest index row and
nothing else. **Both instruments fire on it:** the coverage node on the gap it exists for, and
the digest node because the row set moved under a fixed declaration. **`PASSED`.**

### A correct repair broke another office's node, and it was repaired in the same edit

`tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py::test_the_corrections_index_digest_is_recomputed_over_its_new_row`
asserted `len(rows) == 20`.

```
E  AssertionError: expected 20 C-rows after the M14 append, got 31
```

**A literal count fails on the next correct append rather than on a softening** — the class it
was built to catch. Re-bound: the count is a floor (`>= 20`, the index only grows), the append
it was written for is bound **by name** (`C20` must still be present), and the digest match is
unchanged. **This is the it.19 lesson applied before the round had to find it again.**

---

## REPAIR 2 — THE M14 NODE PERMITTED THE ONLY LEGAL REPAIR AND FORBADE THE ILLEGAL ONES

**JUPITER's strike lands exactly as filed.** `::test_M14_carries_three_grades_in_three_files`
bound the presence of three **texts**. A pre-registered sentence may not be rewritten, so
**superseding in place is the sole legal resolution — and it is precisely the move a presence
check cannot see.** The it.19 resolution landed and the node stayed GREEN.

**The ledger's claim — *"a repair in any of the three files turns it red and the repair cannot
land silently"* — was false for the only repair shape the contract permits**, and a second
ledger sentence, *"It is red, by that design. This is the repair it was waiting for"*, asserted
a red that never happened.

### What landed

`tests/saturn/test_v20_r15_it14_saturn.py::live_M14_grades` reads the grade **token** each
file still asserts, not the text each file contains:

- **contract** — `[SUPERSEDED it.19, RULING J-17d: M14 IS F4]` replaces the pre-registered
  `grade F3`; with no supersession block the pre-registered grade is what is live;
- **ledger** — `L-3`'s single grade, `F4`;
- **journal** — append-only, so its scoreless `F1` cannot be edited away. **It dies by being
  indexed**: a CORRECTIONS INDEX row quoting it retires it, and it is returned as live the
  moment that row goes missing.

`::test_M14_carries_exactly_one_live_grade` replaces the retired node. **Three live grades
before it.19, one after** — it goes RED on the resolution, which is the event it exists to
witness.

### PLANTED NEGATIVE — fires

`::test_PLANTED_NEGATIVE_the_presence_node_was_blind_to_the_resolution` rebuilds the
pre-resolution text (the `[SUPERSEDED …]` block removed from the contract, the `C20` row
removed from the index; nothing else moves) and runs **the retired node's three assertions
verbatim on it**:

```
the retired node on PRE-RESOLUTION text:  all three assertions hold
the retired node on HEAD (post-resolution): all three assertions hold
live_M14_grades on PRE-RESOLUTION text:   {'contract': 'F3', 'ledger': 'F4', 'journal': 'F1'}
live_M14_grades at HEAD:                  {'contract': 'F4', 'ledger': 'F4'}
::test_M14_carries_exactly_one_live_grade on PRE-RESOLUTION text: AssertionError  (pytest.raises)
```

**PASSED.** The retired node cannot tell the two states apart; that is the defect, measured
rather than argued.

### A fourth node was red for the opposite reason and is closed

`::test_Q2_W1_carries_an_F4_grade_and_no_ledger_row_at_all` was RED at HEAD **because JUPITER
filed `L-16` at it.19** — its repair had landed. **Leaving it red across an iteration boundary
publishes a repair as a defect.** Replaced by
`::test_Q2_W1_now_has_its_ledger_row_and_the_recount_is_seven`, which binds the recount the
ledger already wrote: **seven F4 objects, six unconsumable verdicts**, because `L-16`'s verdict
cell is `NOT-PUT` and carries neither LEAPABLE nor TERMINAL.

**Both retirements are recorded in `V20_R15_LEAP_LEDGER.md`, `## FLAG — THE M14 BINDING WAS
FALSE FOR THE ONLY LEGAL REPAIR SHAPE`**, which quotes the two false sentences and names the
replacement route. The append adds no `| **L-…** |` row, so
`::test_the_flag_moved_no_row_digest` stays green.

---

## REPAIR 3 — MARS'S STRIKE-2 REGRESSION TEST PASSED FOR THE WRONG REASON

**This office broke an adversary's guard with a correct repair at it.19 and neither office
noticed.** The found-wing node was re-parametrized from **arm names** to **wing IDs**; MARS's
hardcoded `[("arm_smprime",), ("arm_pl",)]` went stale; the node then raised identically with
and without the mutation, and his test read that raise as the repair working.

### The measurement, and it is the whole strike

```
[RUN] ::test_PLANTED_NEGATIVE_the_stale_arm_name_arguments_are_not_differential
found-wing node called with MARS's stale args, TRUE rows:     ['AssertionError', 'AssertionError']
found-wing node called with MARS's stale args, SWAPPED rows:  ['AssertionError', 'AssertionError']
identical, and neither is 'pass'                              PASSED
```

**A node that fires either way witnesses nothing**, and it was the only thing his test had
left.

### What landed

Two bindings, in his file, in his name:

1. **`_declared_params(fn)` reads the node's own `parametrize` list** — the correction from
   this office's it.19 record, applied to his test. Duplicated from
   `tests/saturn/test_v20_r15_it19_wing_identity.py` rather than imported, because that file
   imports `_swapped` from his and **a cycle is a worse dependency than six lines.**
2. **A differential against the true rows.** The fired set is now
   `fired(mutated) − fired(true)`. **A node that raises on both sides no longer counts**, which
   is what let the stale arguments hand the strike a false GREEN.

The corroboration node this office added at it.19 is **discovered by `getattr`, not named**, so
the harness reports the manifest's real state rather than a list written at it.20.

### PLANTED NEGATIVE — fires

`::test_PLANTED_NEGATIVE_mars_strike_2_fails_without_the_corroboration_node` neuters
`FM::test_every_wings_arm_is_corroborated_outside_clause_a` — the manifest's pre-repair state —
and requires MARS's re-bound test to **FAIL**. It does. **PASSED.**

```
[RUN] python -m pytest tests/mars_v20/test_it18_wing_identity_is_self_certified.py -q
2 passed in 0.37s        (green on the corroboration node, not on a stale-argument raise)
```

---

## THE RUN, WHOLE

```
[RUN] python -m pytest tests/saturn/test_v20_r15_it14_saturn.py \
        tests/saturn/test_v20_r15_it20_saturn.py \
        tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py -q
32 passed in 0.48s

[RUN] python -m pytest tests/saturn/test_v20_r15_it20_saturn.py \
        tests/mars_v20/test_it18_wing_identity_is_self_certified.py \
        tests/saturn/test_v20_r15_it19_wing_identity.py \
        tests/saturn/test_v20_r15_it14_saturn.py \
        tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py \
        tests/saturn/test_v20_r15_freeze_manifest.py -q
57 passed, 1 failed  →  the 1 was Q2_W1, repaired above; re-run: 58 pass across the six files

[RUN] python -m pytest tests/saturn/ tests/jupiter/test_v20_r15_it19_q2_rows_and_m14.py -q
8 failed, 171 passed in 12.07s
```

**The eight are standing REDs, none of them in a file this iteration touched:**
`test_r10_it2_spotcheck_reds.py` (3), `test_v20_r15_wing_rubric.py::test_every_annex_run_instance_has_a_producer_in_the_tree`
(K6, open since it.1), `test_v20_r15_wings_distinct_percell.py` (the corrected `0.30` floor at
`s=8`), and three siblings. **They were red before this iteration and are red after it.**

## LIMITS

The `C21`–`C31` rows are written from the body's own correction headings; **each row's *"what
is true"* is a summary of the entry it points at, and where a summary and an entry differ the
entry governs** — the index says so in its own text and this office did not weaken it.
`CORRECTION 13`–`19` have no body headings, only index rows, so the coverage node binds a
range with a gap in the middle; it binds every number that exists on both sides and the
maximum by name, which is where appends break it. The `C20` numbering collision is disclosed,
not repaired. `live_M14_grades` treats *"indexed by a `C`-row quoting the F1 line"* as the
journal's retirement mechanism — that is the only mechanism an append-only file has, and if the
round later retires a journal grade some other way the node will read it as live. MARS's test
still carries a name that reads as its it.18 RED rather than as the it.20 regression guard it
now is; renaming an adversary's node was not this office's call.

## NOT REACHED

`want` frozen at census time; the `30 of 123` citation-coverage gap priced; the `28 vs 25`
manifest gap; the 121 citations still never individually opened. **The eight standing REDs
above are untouched.**
