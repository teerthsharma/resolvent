# V20 R15 it.21 — JUPITER (MYCROFT, annex owner)

**The 13 citations it.20 refused, re-anchored by heading and scored. Coverage is
`129 of 129`.**

Instrument: `tests/jupiter/test_v20_r15_it21_heading_anchor.py` (7 tests). It
imports the it.20 freeze rather than copying it, so an edit to `CENSUS`,
`WITHDRAWN` or `WANT_SEAL` is RED here as well as there.

---

## 1. THE NUMBER, UNROUNDED

| | occurrences | how certified |
|---|---|---|
| census population | **129** / 103 unique | measured at it.20, re-asserted here |
| SCORED BY LINE (it.20) | **116** | `test_every_scored_citation_lands_on_its_frozen_anchor` |
| **SCORED BY HEADING (it.21)** | **13** | `test_all_thirteen_refusals_resolve_by_heading_and_land` |
| **unscored** | **0 of 129** | `test_coverage_is_129_of_129_with_nothing_left_unscored` |
| failing, either population | **0** | — |
| coverage leaving it.18 | 30 of 123 | MARS's live RED |

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it21_heading_anchor.py
tests/jupiter/test_v20_r15_it20_citation_freeze.py -q` → `17 passed`.**

Whole-annex control, same tree, same minute:
**`[RUN] python -m pytest tests/jupiter/ -q` → `3 failed, 210 passed`.** All three
are named, none is new: two are the `C17` digit RED quoted in §2, superseded by
`H17` and left standing as the evidence for this iteration; the third is
`test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record`,
marked **RED BY DESIGN** in its own docstring since it.4 — the strike this office
filed against its own it.2 verdict — and untouched here.

The 13 are 10 unique pointers: `V20_R15_LEAP_LEDGER.md` ×8 cids (10 occurrences),
`V20_R15_JOURNAL.md` ×2, `house-events.jsonl` ×1. Every one is still into a LIVE
file. **J-20b is not repealed.** What changed is that a pointer into a LIVE file
now has an anchor form the instrument can check, so refusing it is no longer the
only honest option.

---

## 2. THE RED, AT HEAD, UNMUTATED

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it18_citation_landing.py -q`
→ `2 failed, 4 passed`**, no mutation, no `git stash`, verbatim:

```
>           assert want in line_at(path, lineno), f"{cid}: {path}:{lineno} lacks {want!r}"
E           AssertionError: C17: V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'
E           assert 'THE FREEZE' in '   the extreme — permuting the arms independently, the ratio spans **2.09x to'
E            +  where '   the extreme — permuting the arms independently, the ratio spans **2.09x to' = line_at('V20_R15_JOURNAL.md', 650)
```

`C17` is the round's worked example of the digit failure. At it.18 the heading was
at `:650`. At the it.20 census it was at `:651` and `:650` was blank. At this
iteration it is at `:669` and `:650` carries an unrelated sentence. **Three
different wrong answers from one unchanged pointer in three iterations.** The
it.20 note is confirmed by a third data point: a digit repair expires on the next
append, and it did, twice.

`H17` is the repair: `V20_R15_JOURNAL.md` § `## it.4 — THE FREEZE`. It resolves to
`:669` now and it resolved to `:651` then, from the same anchor text.

---

## 3. RULING J-21a — A HEADING ANCHOR IS RESOLVED, THEN LANDED

`J-20b` said a heading pointer is the correct repair and could not be the rule
"because a heading pointer has no LINE to check". An anchor is now
`(path, section_key, want)`:

1. **RESOLVE** — `section_key` must match **exactly one** heading line. 0 (renamed
   or deleted) or 2 (duplicated) is a REFUSAL, not a pass.
2. **SCOPE** — the section runs to the next heading of same-or-shallower level, or
   EOF. A file with no headings (`house-events.jsonl`) has `section_key = None`
   and the section is the whole file.
3. **LAND** — `want` must appear on **exactly one** line inside that section. That
   line is the resolution, and **the it.20 landing check is then applied to it
   verbatim**: `want in line_at(path, lineno)`.

Uniqueness at both steps is the entire mechanism. A resolver taking the first
match would follow a duplicated heading into the wrong section and report a line
number with full confidence — the failure the round has been punishing all
iteration, moved one level up.

### Planted negative 1 — the append that breaks the digit and not the heading

`test_the_heading_anchor_SURVIVES_AN_APPEND_THAT_BREAKS_THE_DIGIT`. 40 lines are
prepended **in memory**; no file the round owns is written. Verbatim:

```
H15 heading  before append -> :663   after +40 -> :703
H17 heading  before append -> :669   after +40 -> :709
C15 digit :645 after +40 -> carries 'measured on the corrected gate'? False
C17 digit :650 after +40 -> carries 'THE FREEZE'? False
```

Both heading anchors track the append exactly; both digit anchors are destroyed by
it. The test asserts the digit breakage as well as the heading survival — a
negative that only showed the heading working would prove nothing about what the
heading bought.

### Planted negative 2 — refusal, not a guess

`test_an_unresolvable_or_ambiguous_heading_is_REFUSED_not_guessed`, verbatim:

```
REFUSED: section key '## it.99 — never written' matches 0 headings in V20_R15_LEAP_LEDGER.md, not 1
REFUSED: 'F4' appears on 4 lines in section '## it.7' of V20_R15_LEAP_LEDGER.md, not 1
REFUSED: section key '## it.7' matches 2 headings in V20_R15_LEAP_LEDGER.md, not 1
```

The third is the duplicate-heading case and it is the one that matters: the same
anchor that returns `:28` against the real file returns a REFUSAL the moment a
second `## it.7` exists. Three of the ten re-anchors (`H10`, `H18`, `H117`) needed
a longer `want` or a longer key precisely because the short form hit this.

---

## 4. RULING J-21b — THE 13 ARE RE-ISSUED, NOT REPAIRED

`J-20a` holds against this office. Changing a `want` is a withdrawal plus a new
citation, so the 13 are issued as **new cids `H*`** under their own
`HEADING_SEAL`, with `REANCHORS` recording which withdrawn `C*` each replaces.
No `C*` in the it.20 census is edited. `test_every_withdrawn_cid_has_exactly_one_re_anchor`
asserts the map is total and that no re-anchor silently moved to a different file.

---

## 5. RULING J-21c — LOCATION IS NOT ARGUMENT. **RETIRE.**

The round has not touched this and it is the deeper of the two limits filed at
it.20. Stated exactly:

> The landing check, the want freeze, and the heading resolver all certify that a
> cited line carries certain words. **None certifies that those words support the
> claim the cell makes from them.**

The standing instance is `C110`, and it is **GREEN under every instrument the
round owns**. The cell (`V20_R15_THEORY_TABLE.md`) says choosing the tail would be
"the catalogued failure of selecting a threshold after seeing the data it judges
(`MISTAKES.md:451`)". `MISTAKES.md:451` carries "A threshold refitted to the data
it judges". The anchor certifies **that the failure has a name in the taxonomy at
that line**. The cell's claim is **that this choice would be an instance of it** —
a judgment about a decision not yet made. The two are different propositions and
no arrangement of substrings separates them. The cell does not even quote the
line; it paraphrases the rule and applies it, which is why the freeze cannot
detect the gap either — there is no wrong `want` to catch.

**Disposition: RETIRE.** Closing it means reading the cited passage, reading the
claim, and judging entailment. That is a reader, and this office should not ship
an instrument that can be mistaken for one. What is shipped instead is
`test_the_instrument_cannot_see_the_ARGUMENT_and_says_so`, which asserts *both*
that the location check passes *and* that the cell only paraphrases — so the green
and the gap are recorded in the same place and the green cannot be read as
support.

The scope of what this retires is not small and is stated rather than softened:
**129 of 129 citations are certified for location and 0 of 129 for argument.** The
leap reads the table for argument.

---

## 6. THE `WANT_SEAL` LIMIT, PRICED

The it.20 concession: *"the seal sits in the file the audited office writes."*
Asked directly — is there a seal location not written by the audited office?

**Inside this repo, no, and the reason is structural rather than a matter of which
file is chosen.** Every candidate was checked against the one question that
matters — *can JUPITER write it?*

| location | writable by JUPITER | what it would buy |
|---|---|---|
| the instrument itself (today) | yes | tamper-evident: a want edit is a 2-line diff |
| a second file in the same tree | yes | nothing — same hand, one more file |
| another office's report | yes | nothing enforced; only convention |
| `house-events.jsonl` (append-only log) | yes (append) | an append cannot be silently un-appended; a contradicting later record is visible |
| git history | out of scope — **no git writes this iteration** | a rewrite is detectable, not prevented |

The append-only log is the only one that changes anything, and it changes one
thing: **an appended seal cannot be edited in place, only contradicted by a later
append, and the contradiction is itself a record.** That is still tamper-evident,
one notch harder. It is not adopted here because it costs a write to a shared log
mid-round and because it would still be verified by this office's own instrument.

**The real price of tamper-proof is one iteration of a second planet re-taking the
census independently and publishing its own digest.** Two independently produced
digests over the same 129 anchors either agree or name the discrepancy, and
neither office can produce the other's. Nothing cheaper than that changes the
class of the guarantee, and this office cannot buy it from inside its own annex —
which is the same shape as MARS's strike 2 and the INSPECTOR's ruling, and is
conceded on the same terms.

---

## 7. LIMITS

* **129 of 129 for location, 0 of 129 for argument** (§5). RETIRED, not deferred.
* A heading anchor is only as stable as its heading. `H9`/`H10`/`H11` all key on
  `## it.7` in `V20_R15_LEAP_LEDGER.md`; renaming that one heading refuses three
  citations at once. The refusal is loud, which is the point, but the coupling is
  real and is the heading analogue of it.20's "21 pointers into one file".
* The resolver scores **the current line**. It does not detect that a section's
  *content* was rewritten in place under an unchanged heading, exactly as
  `FILE_LINES_AT_CENSUS` did not detect an in-place edit at it.20. The `want`
  freeze is the only thing standing there and it is tamper-evident only (§6).
* `H6` is into `house-events.jsonl`, which has no headings; its section is the
  whole file and its `want` is unique across 12,000+ lines *today*. A second
  `finding_contract` event for JUPITER would refuse it. That is correct behaviour
  and it means `H6` is the most fragile of the ten.
* Every number here is dated to this iteration. `129` was `123` at it.18.

---

## 8. NOT REACHED

* **The append-only seal was priced, not built** (§6). No record was written to
  `house-events.jsonl`.
* **The 13 are scored but the table still carries their digit text.** The
  citations in `V20_R15_THEORY_TABLE.md` still read `V20_R15_JOURNAL.md:650`; the
  instrument now scores them by heading through `REANCHORS`, but the table's own
  prose was not rewritten to the `§ heading` form. Rewriting a frozen table is not
  this iteration's authority to take alone.
* **No second-office cross-census**, which §6 names as the only thing that makes
  the freeze tamper-proof.
