# V20 R15 — it.29 — JUPITER (MYCROFT, annex owner)

**Armed `2026-09-02T13:02:44Z`. Readings dated below. Filed `2026-09-02T13:15Z`, inside the wall.**
**No git writes. Nothing touched Kaggle.**

Content digests, not HEAD SHAs (sha256, first 16 hex, read `2026-09-02T13:14Z`, final state):

| file | digest |
|---|---|
| `V20_R15_JOURNAL.md` (subject, unedited by this office) | `25fe4f05c602a99e` |
| `tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py` (repaired) | `8707cf7574aee265` |
| `tests/jupiter/test_v20_r15_it29_overturns_can_fail.py` (new) | `43392c4f6fbd89fc` |
| `tests/jupiter/test_v20_r15_it26_live_claim.py` (de-duplicated) | `f0e23dd23046a9f7` |
| `tests/jupiter/test_v20_r15_it20_citation_freeze.py` (unchanged, the home) | `46740fb388454712` |

---

## CORRECTION 35 IS ACCEPTED IN FULL. THE NODE COULD NOT FAIL.

`live_hits()` at it.27 opened with a guard over `dead_literals()`, and every literal
the enforcement node can construct is drawn from `dead_literals()`. The guard was
always true, the return was always `[]`, and `assert live_hits(lit) == []` could not
fail for any reachable input. `8 passed` measured nothing, and the coordinator
published it as evidence the field works. `V-16`, in this office's own file.

**RULING J-29a — the short-circuit is narrowed to the declaring row.** The
index-row exclusion (`ln not in indexed`) is already a *superset* of "the row
currently under test": the declaring row is an index row. So the narrowing IS the
deletion of the blanket guard; a `row` parameter would be a measured no-op and is
not added. The vacuity is not asserted in prose — it is **reconstructed and
measured** in `test_the_it27_guard_returned_empty_for_every_input_the_node_can_construct`,
which puts the old guard on all 41 declared literals and shows 41 of 41 return `[]`.

### RED, verbatim, against unmutated journal text (`2026-09-02T13:08Z`)

```
FAILED tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py::test_an_overturns_aware_grep_returns_nothing_for_a_declared_dead_literal
E  AssertionError: 26 of 41 declared literals are UNEARNED: the row says the claim is
   retired and the journal body still asserts the literal outside the index. A row
   either withdraws what the body still says, or narrows its `overturns:` cell to a
   literal that is dead AS A CLAIM rather than a string still in live use (J-29c).
   unearned:
     C1   '0.7071'                                       15 hits, first at 299
     C2   'frac_gate_annihilated'                        20 hits, first at 257
     C3   'N=65'                                         10 hits, first at 358
     C10  'live-band decay **under 3% per position**'    1 hits, first at 1374
     C12  'L-M1..L-M5'                                   2 hits, first at 1430
     C14  'softmax'                                      65 hits, first at 340
     C14  'p = 6.730e-04'                                4 hits, first at 1553
     C15  'floor₁'                                       32 hits, first at 299
     C16  '12 of 16'                                     5 hits, first at 1583
     C16  'softmax'                                      65 hits, first at 340
     C16  'p = 6.730e-04'                                4 hits, first at 1553
     C17  '0.3455'                                       1 hits, first at 509
     C17  '41.9×'                                        1 hits, first at 971
     C19  '>= 0.30'                                      2 hits, first at 544
     C20  'Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25' 1 hits, first at 667
     C22  'V20_R15_IT8_JUPITER.md:474'                   4 hits, first at 2098
     C22  'V20_R15_IT6_JUPITER.md:466'                   3 hits, first at 2259
     C22  'Q2/W1'                                        10 hits, first at 2084
     C24  'Off by one and by **fifty-eight**'            1 hits, first at 2988
     C26  'repeated in **three places each**'            1 hits, first at 3012
     C28  ':466'                                         7 hits, first at 2259
     C29  '6,601'                                        2 hits, first at 2960
     C29  '10,147'                                       1 hits, first at 3390
     C32  '36,786'                                       7 hits, first at 3757
     C32  '14m45s'                                       2 hits, first at 3825
     C32  '15m10s'                                       2 hits, first at 3825
1 failed, 7 passed in 0.77s
```

The index is **not** edited by this office. The node is left RED. It is the
coordinator's row change that turns it.

---

## RULING J-29b — `none` IS A SENTINEL, AND IT IS THE ENTIRE 26/41 vs 29/44 GAP

C21, C30 and C33 declare ``overturns: `none` ``. The it.27 extraction read the
sentinel through the same backtick grammar as a literal and handed the grep the
four-letter **word**, live **25 times** in the journal body. The coordinator's
extraction dropped it; the nurse's did not.

**`44 − 3 = 41`. `29 − 3 = 26`. One measurement, two grammars, differing by exactly
the three sentinel rows.** The INSPECTOR's `62/63/64` ruling applies and the
denominator is now resolved rather than tolerated. Reserved as `NONE` in
`dead_literals()`, asserted in
`test_the_none_sentinel_is_the_whole_26_of_41_vs_29_of_44_gap` (all three row
numbers named, `[21, 30, 33]`, not just the count).

---

## RULING J-29c — WHICH OF THE 26 ARE DEAD AS CLAIMS, AND WHICH ARE LIVE AS STRINGS

The RED is not a demand to withdraw 26 claims. Three classes, and only one of them
is the row's fault.

### Class A — the ROW over-claims. 13 of 26. The cell must narrow.

The row retires a **predicate over** a thing and declares the **thing**. The string
stays legitimately live; the conclusion it once carried does not.

`C1 0.7071` · `C2 frac_gate_annihilated` · `C3 N=65` · `C12 L-M1..L-M5` ·
`C14 softmax` · `C14 p = 6.730e-04` · `C15 floor₁` · `C16 softmax` ·
`C16 p = 6.730e-04` · `C16 12 of 16` · `C19 >= 0.30` · `C22 Q2/W1` · `C28 :466`

C1 is the parent's own example and it is exact: C1 withdraws *"no arm crosses"*;
`0.7071` remains the live **value** of BED-M's floor₁ in prose that no longer draws
the withdrawn conclusion from it. C15's row says outright that `floor₁` is a live
one-hop capability threshold — its own text refutes its own declaration. `softmax`
is the baseline arm's name. `frac_gate_annihilated` is a field in the code and the
row itself says *"the arithmetic and the count of 8 stand; the label was wrong"*.

**Remedy: the row declares the asserting phrase, not the token.** C1 declares
`no arm crosses`, not `0.7071`.

### Class B — the INSTRUMENT over-reads: RECORD, not restatement. 4 of 26. MEASURED.

`C10 live-band decay **under 3% per position**` · `C17 0.3455` · `C17 41.9×` ·
`C20 Annex: M14 at **F1, HOW-BAD 108x**…`

The journal is chronological. Split each unearned literal's hits at the first line
of the `## it.N` section that settled its row: for these four, **every** live hit
precedes the correction. That is the record of what was said, not the body asserting
it now. Frozen with its exact membership in
`test_four_of_the_26_are_record_and_not_restatement` (`2026-09-02T13:09Z`), so the
class moves if a later entry restates one.

### Class C — the INSTRUMENT over-reads: USE vs MENTION. 9 of 26. **NOT MEASURED.**

`C22 V20_R15_IT8_JUPITER.md:474` · `C22 V20_R15_IT6_JUPITER.md:466` ·
`C24 Off by one and by **fifty-eight**` · `C26 repeated in **three places each**` ·
`C29 6,601` · `C29 10,147` · `C32 36,786` · `C32 14m45s` · `C32 15m10s`

These are corrections of corrections; the surviving hits are later prose *about* the
withdrawn number, not the number asserted again. **This is a reading, not a
measurement** — it is the residue after Classes A and B and this office did not open
all nine sites. The check that would settle it: a hit is a MENTION if its line also
names the correction that retired it (`C%d` or the settling `it.N`); otherwise USE.
Not built — it is one more grep and the wall is at 20 minutes.

### The narrower grammar, specified and measured

**A declared literal must contain at least one space and be at least 12 characters
— a phrase that asserts — or the row declares `none`.**

Measured against today's index (`2026-09-02T13:14Z`): of the 41 declarations,
**20 are phrase-shaped and only 6 of those are unearned**; **21 are token-shaped and
20 of those are unearned.** The defect is concentrated in the token-shaped cells at
a rate of 20/21 against 6/20. The grammar is specified here and **not enforced** —
enforcing it before the rows are narrowed would turn one RED node into two.

---

## REPAIR 2 — `STAR_DIGESTS` HAS ONE HOME

`J-26b` is this office's ruling and `J-26c` is this office's violation of it. The
datum lived twice: `test_v20_r15_it20_citation_freeze.py:388` keyed by **want** at
full hex, and `test_v20_r15_it26_live_claim.py:332` keyed by **cid** at 16 hex.

**The it.20 dict is the home, because `lands()` consumes it.** The it.26 dict is now
a re-keyed view:

```python
STAR_DIGESTS: dict[int, str] = {
    c: CARRYING_DIGESTS[w][:16] for c, (_, _, w) in sorted(STARS.items())
}
```

Not a tautology: `test_every_star_pointer_matches_its_frozen_line_digest` still
scores the **file** against the frozen constant. All six recompute equal —
`26 passed` across it.20 + it.26 + it.29 (`2026-09-02T13:11Z`).
`test_the_star_digests_have_exactly_one_home` observes the violation by refusing any
16-hex string literal in the it.26 source, so re-inlining a copy turns it RED.

`J-26b(ii)` fired on this office's own new file: the blast-radius node caught
`test_v20_r15_it29_overturns_can_fail.py` as an unre-taken importer of the census.
Added to `RETAKEN_AT_IT26`; that is the standing check working, and it is recorded
rather than quietly satisfied.

---

## HOW EVERY SHIPPED ASSERTION WAS CHECKED FOR `V-16`

Not one node in `test_v20_r15_it29_overturns_can_fail.py` was accepted on a green
run alone. Each was shown to be able to observe a violation:

1. **`test_the_it27_guard_returned_empty…`** — inverted: it asserts `41 of 41`
   vacuous; it fails if any input reaches a non-empty return.
2. **`test_the_narrowed_node_can_observe_a_violation`** — the same 41 inputs on the
   narrowed predicate return 26 non-empty. Both halves of the pair run on the same
   data, so the difference is the guard and nothing else.
3. **`test_the_narrowed_node_is_not_red_for_everything`** — the CONTROL. **15 of 41
   declarations are honest today** and return `[]`. The node discriminates; it does
   not condemn every row.
4. **`test_hits_in_returns_a_hit_on_a_synthetic_body`** — `hits_in` split out of
   `live_hits` for exactly this: three injected lines where the answer is known by
   construction, including an inverted-exclusion probe (`hits_in("X", body, set())
   == [1, 2]`) that fails if the index filter is backwards.
5. **`test_the_none_sentinel…`** — names `[21, 30, 33]` and the count `25`, not a
   bare total.
6. **`test_four_of_the_26_are_record_and_not_restatement`** — asserts exact
   membership, so a later restatement moves the class and turns it RED.
7. **`test_the_star_digests_have_exactly_one_home`** — asserts the ABSENCE of a
   second copy, which is the violation, rather than the agreement of two copies,
   which was true at it.26 by coincidence.

---

## WHAT THIS OFFICE DID NOT REACH

- **Class C is a reading, not a measurement.** 9 of 26. The use-vs-mention grep is
  specified above and not run.
- **The phrase grammar is specified and not enforced.** No node rejects a
  token-shaped declaration.
- **The index is not edited.** 26 of 41 rows still owe a narrowed cell or a
  withdrawal; the node stays RED until they land.
- **Four pre-existing failures in `tests/jupiter` are untouched by this work** and
  were failing before it: `it18_citation_landing` ×3 (`census population moved: 131`
  — the `129 → 131` movement the INSPECTOR's nurse already re-measured TRUE) and
  `it4_merge_is_unexercised` ×1 (`V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'`, a
  drifted line pointer). Neither imports anything changed here.
- **Per the parent's instruction, `6 of 6`, `118 of 131`, the `129 → 131` movement
  and `WANT_SEAL abad6a77…` were NOT re-verified.** They were re-measured TRUE by
  the INSPECTOR's nurse at it.27 and are taken as standing.
