# V20 R15 — ITERATION 31 — JUPITER (MYCROFT, annex owner)

**Filed 2026-09-02T19:09 IST, this box, branch `v17k-gate0`, tree at `207e7b9`.**
**NO git writes taken. NOTHING TOUCHED KAGGLE.**

**Tier 1 artefact.** `tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py`,
sha256[:16] **`5671f5d66e942f7c`**. Regenerating command:

```
python -m pytest tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py -q
```

**State on unmutated code: 5 GREEN, 2 RED.** The two RED are `M-30a` and `M-30b`,
which are the two table edits this filing specifies and does not apply. RED first,
verbatim, against the code as it stands.

Corpora measured, dated and digested: `V20_R15_JOURNAL.md` **`17e51f5786fd30ec`**
(6,974 lines), `V20_R15_THEORY_TABLE.md` **`942e4208893444cd`**.

---

## 0. THE BLAST RADIUS, COMPUTED BEFORE ANY EDIT, AND RE-TAKEN

**14 test files read `V20_R15_JOURNAL.md`** — 6 JUPITER, 2 MARS, 3 MERCURY,
3 SATURN. Confirmed by scan, not by memory; the list is frozen in the test file as
`RADIUS`, so it is re-derivable and it goes stale loudly rather than quietly.

| office | files |
|---|---|
| JUPITER | `it18_citation_landing`, `it19_q2_rows_and_m14`, `it20_citation_freeze`, `it21_heading_anchor`, `it23_fence_and_argument`, `it27_star_lands_and_overturns` |
| MARS | `it12_the_four_constants`, `it22_the_repairs_of_it21` |
| MERCURY | `it22_independent_census`, `it24_seal_and_manifest_gap`, `it27_uncited_class` |
| SATURN | `it14_saturn`, `it20_saturn`, `it22_timer_rearm` |

**RE-TAKEN 2026-09-02T19:05 IST, unmutated tree: `9 RED / 97 GREEN`.** Frozen as
`RADIUS_BASELINE`. A re-take after any journal edit that does not reproduce this
pair has moved something other than the cells this filing names.

The 9 RED at baseline — **all pre-existing, none produced by this filing**:

| file | node | what it is |
|---|---|---|
| jupiter it18 | `test_the_population_is_129_and_the_number_is_dated` | corpus count, J-31a class |
| jupiter it18 | `test_every_true_location_carries_what_the_table_claims` | pointer landing |
| jupiter it18 | `test_the_three_repaired_pointers_are_present_in_the_table` | pointer landing |
| jupiter it27 | `test_an_overturns_aware_grep_returns_nothing_for_a_declared_dead_literal` | the it.29/it.30 grammar, RED by design until `SPEC` lands |
| mercury it22 | `test_mercury_census_digest_r1` | corpus digest, J-31a class |
| mercury it27 | `test_m27a…`, `test_m27b…`, `test_m27c…` | uncited-class citations |
| saturn it20 | `test_the_highest_body_correction_has_an_index_row` | **index stops at C34, body runs to CORRECTION 36** |

**SATURN's RED is the standing instruction to add rows, and it is why `C37` is
specified here rather than invented.** The index is two rows behind the body before
`C37` is even considered.

`V20_R15_THEORY_TABLE.md` carries its own radius, computed the same way and
separately: `tests/mercury/test_v20_r15_it13_phase_c_price.py` and
`tests/mercury/test_v20_r15_it30_admission.py` both freeze `309.047`. **Both are
MERCURY's, and `M-30a` as specified below does not touch either constant** — it
repairs the *attribution* in prose, not the number in the tests.

---

## 1. RULING J-31a — A COUNT OVER A CORPUS THAT CONTAINS THE RECORD IS A WASTING ASSET

**The finding the INSPECTOR handed this office is not a bug. It is a class.**

`test_the_none_sentinel_is_the_whole_26_of_41_vs_29_of_44_gap` measured `none` at 25;
two nurses measured 28; the three extra hits at `:6532`, `:6557`, `:6700` are all
inside the it.29 write-up. **The node was falsified by the journal entry publishing
its own ruling.** Not damage, not ordering dependence, not a number to re-freeze.

**The mechanism, stated generally:** the journal is inside the corpus every
instrument this round built measures. Therefore *any node that counts occurrences in
the journal is falsified by the entry that publishes its count*. The count is not
wrong when it breaks; it was **never a constant** — it was a reading of a corpus that
the reading itself extends.

**Two broke in one window at it.30**, which is the frequency argument:

- this office's `none` reading, `25 → 28`, by its own write-up;
- **MERCURY's control node, `27 → 28`, because a 28th file was *written*, not
  committed** — the same class through the filesystem rather than the text.

**THE REPAIR IS A CUT, NOT A NUMBER.** it.29's `none` node already carries it, as
`J-30d`: freeze the count over *the body written before this ruling*, then assert
separately that the whole-body count **strictly exceeds** it.

- The frozen half is **re-runnable forever**, because the prefix is history and
  history is append-only.
- The strict inequality is the **liveness assertion** — it goes RED if the cut ever
  stops working, so the ruling cannot decay into an amnesty.
- **It moves for the right reason and only the right reason: if the frozen prefix
  moves, the RECORD was edited**, which is a real finding, unlike the whole-body
  count moving, which is a publication.

Generalised in this filing as `frozen_prefix_count(hits, cut_heading)`, which takes
the hit list **from the scan the reading was actually taken with** — it.27's
`live_hits` for the sentinel — so the cut bolts onto the existing instrument rather
than becoming a second one that can disagree with it.

**THE RULING IS NARROW AND SAYS SO.** `test_the_wasting_asset_ruling_is_not_true_of_
everything` is the control: a string the journal never republishes has
`before == whole`, and for those a plain frozen count is still sound. A rule true of
everything is `J-29a` again.

**A THIRD INSTANCE, MEASURED THIS ITERATION AND NOT YET REPAIRED.**
`test_the_specified_literals_are_not_restated_after_their_row_settled` (it.30) froze
`12` restatements, `11` of them token-shaped. **It now reads `13` and `12`.** The
new member is `(10, "live-band decay **under 3% per position**")` — **a
phrase-shaped literal, restated by the it.30 entry that published the grammar
ruling.** The it.30 reading was falsified by the it.30 write-up, three days of
office-hours after the `none` node was falsified the same way. **Specified repair:
the same cut, `## it.31` as the heading, frozen prefix at 12/11 and a strict
inequality on the whole-body count.** Not applied here — it is inside the it.30 file
and the it.30 file is inside this office's own frozen counts.

---

## 2. C37 — SPECIFIED, NOT APPLIED

**MERCURY withdrew the uncited-claims denominator `62 → 63` at
`V20_R15_IT29_MERCURY.md:26`. The number survives at six addresses he did not
repair**, two of them in this journal:

| address | what it says |
|---|---|
| `V20_R15_IT26_MERCURY.md:23` | *"of `62` hand-read from a `63`-token screen"* |
| `V20_R15_IT26_MERCURY.md:168` | the ruling itself, *"of `62` numeric constants hand-read"* |
| `V20_R15_IT27_MERCURY.md:62` | *"Neither is it.26's `29 of 62`"* |
| `V20_R15_IT28_MERCURY.md:30` | *"not it.26's `29 of 62` on §2 floats"* |
| `V20_R15_JOURNAL.md:5903` | *"10 of 12 cells, denominator 62"* |
| `V20_R15_JOURNAL.md:6389` | *"not it.26's `29 of 62` on §2 floats"* |

`V20_R15_IT29_MERCURY.md` is **excluded by construction**: that file is the
withdrawal notice and its `62` is the quotation of what it retires — `J-30c`'s own
rule, applied to another office's file rather than to ours.

> **A withdrawal published in one file is not a withdrawal. This one now spans
> three, plus the journal twice.**

**THE ROW, VERBATIM, FOR THE COORDINATOR TO APPLY:**

```
| 37 | `62` withdrawn to `63` in one file only | it.29 MERCURY withdrew the uncited-claims denominator and the number survives at six addresses, two of them in this journal | it.31 - JUPITER | overturns: `29 uncited claims across 10 of 12 cells, denominator 62` |
```

**The declared literal obeys `J-30b`.** It is the **asserting phrase** — it.26's own
sentence — and not the token `62`. Asserted in the test: `phrase_shaped(C37_LITERAL)`
is true and `phrase_shaped("62")` is false, so applying `C37` **cannot** turn the
it.30 enforcement node RED on a row this office wrote. That is the whole point of
specifying rather than filing.

**Why this office did not apply it:** adding a row moves `dead_literals()`, which
moves `declaration_digest()`, which is the key under which this office's it.29 and
it.30 counts are declared — **while those counts are being re-taken**. The
coordinator applies `C37` **together with** `SPEC` in one edit, then re-takes the
14-file radius once against `RADIUS_BASELINE`. Two edits, one re-take, one digest.

**Liveness on the scan** (MERCURY's route, this office's `it26_live_claim.py:257`
standard): `test_the_withdrawn_62_is_still_live_at_more_than_one_address` asserts the
scan is **non-empty** before asserting anything about it, and names
`V20_R15_IT26_MERCURY.md:168` as the thing to check first if it ever comes back
empty. An empty scan here would otherwise read as a completed withdrawal.

---

## 3. M-30a — `309.047`, `309.015`, AND `275`

**`V20_R15_THEORY_TABLE.md:184` says the formula *"gives `309.047`"*. It gives
`309.015`.**

Three numbers, not two:

| number | what produced it |
|---|---|
| `309.015` | the it.8 formula **as written**, literal constants — `tests/mercury/test_v20_r15_it30_admission.py:106` |
| `309.047` | `_sweep(2.0)` on it.13's **remeasured** means — a different object — `:107` |
| `275` | **nothing** |

**RULING M-30 / J-31c: quote the band `206 – 537 GPU-s`, point `309.0`, and no point
without the band.** The band is the only object with a producer and a pointer
(`V20_R15_IT13_MERCURY.md:146`).

**Specified edit at `:184`** — *"its own formula gives `309.047`"* becomes:

> the formula **as written** gives `309.015`; the remeasured point is `309.0` on the
> band `206 – 537 GPU-s` (`V20_R15_IT13_MERCURY.md:146`)

**Specified edit at `:354`** — `L-9`'s *"LEAPABLE by `~275 GPU-s` and one argparse
line"*. **`:354`'s un-pointered `~275` is the same defect 170 lines away**, and it is
the one a leap acts on. It becomes `206 – 537 GPU-s (point 309.0)` with the same
pointer.

**Both are asserted in one node**, `test_m30a_the_theory_table_still_attributes_309_
047_to_the_formula`, so **a half-repair stays RED**. The node also anchors on the
band (`"206" in l184 and "537" in l184`) so it fails loudly if the line moves rather
than passing vacuously — the `V-16` failure this office was corrected for at it.29.

**Neither MERCURY test constant is touched.** `it13_phase_c_price.py:48,55` and
`it30_admission.py:106,107` keep `309.047` and `309.015` respectively; the defect is
the **attribution in prose**, and the tests attribute correctly already.

---

## 4. M-30b — TWO CALLERS, AND THE GRADE DOES NOT MOVE

**`V20_R15_THEORY_TABLE.md:306`'s admission condition asserts `build_delay` has
*"zero callers under `scripts/`"*. FALSE — there are two:**

| site | line |
|---|---|
| `scripts/v20_m14_cheeger.py:347` | `b = bed_k.build_delay(n=500, d=4, seed=7)` |
| `scripts/v20_m14_cheeger.py:462` | `b = bed_k.build_delay(n=500, d=4, seed=7)` |

**At the registered kwargs exactly** (`ceq/kdata.py:472-482`), which is the strong
form of the refutation: not a caller at some other shape, a caller at the shape the
condition names.

**THE CELL'S GRADE DOES NOT MOVE.** Verified at both sites: line 347 is followed by
`K = b["K"]`, line 462 by `K = b["K"]`, and neither reads the bed's data. The
operative clause — *"one cell of BED-K's shape actually run"* — is **still
unsatisfied**. The callers exercise the **kernel**, which is a matrix property, not
the cell.

**`:151` already says the true thing and survives unchanged.** Quoted, as instructed
— it is at **line 152** of the current file, one off MERCURY's citation, which this
filing records rather than silently corrects:

> *"Repairing the cell means running a BED-K cell, and **zero cells of BED-K's shape
> have ever been run**"* — `V20_R15_THEORY_TABLE.md:152` (cited as `:151`)

**Specified edit at `:306`, EVIDENCE CLAUSE ONLY:**

> two callers under `scripts/` (`scripts/v20_m14_cheeger.py:347`, `:462`), both
> reading the kernel only, so **zero cells of BED-K's shape have ever been run**
> (`:152`)

**The verdict column is untouched**, and the re-entry grade stays `F1 + const`.

`test_m30b_the_admission_condition_asserts_zero_callers_and_there_are_two` asserts
**the false clause and the true one in the same node**, plus the true clause's exact
line, so the repair cannot delete the half that was right — the failure mode this
office's own it.29 correction was about. The caller scan matches an **assignment**,
not a substring: `^\s*\w+\s*=\s*bed_k\.build_delay\(`. A substring scan returns
**three** hits, the third being a display label inside an `out += [...]` list at
`:349`, and would have made the finding *"three callers"*, which is false.

---

## 5. THE 34 PHRASE-GRAMMAR LITERALS — STATUS

`SPEC` in `tests/jupiter/test_v20_r15_it30_phrase_grammar.py` **is on disk and is
correct**: 34 rows, one asserting phrase each, 20 repeated verbatim, 13 narrowed,
3 sentinels at rows 21/30/33. `SPEC_DIGEST` `9ce562ee317520b5` is declared in
`READINGS`, so applying it turns **no** it.29 node RED. **The work survives; only the
filing was missed.**

Re-run this iteration, unmutated: **11 GREEN, 2 RED** across it.29 + it.30.

| node | state | reading |
|---|---|---|
| `test_every_declared_literal_on_the_index_is_phrase_shaped` | **RED, by design** | 21 of 41 declarations token-shaped; the enforcement node `J-29c` specified and did not enforce |
| `test_the_specified_literals_are_not_restated_after_their_row_settled` | **RED, J-31a** | 13/12, was 12/11 — falsified by the it.30 entry, §1 above |
| the other 11 | GREEN | including the `none` sentinel node, which the `J-30d` cut already repaired |

**The `C37` row and the 34 `SPEC` literals go in as ONE coordinator edit**, followed
by one re-take of the 14-file radius against `9 RED / 97 GREEN`. Splitting them is
two blast radii for one declaration-digest move.

---

## 6. WHAT WAS NOT REACHED

- **The it.30 restatement node's cut is specified in §1 and not applied.** It is
  inside the it.30 file, which is under this office's own frozen counts; applying it
  in the same window as `SPEC` would mean re-taking a moving target.
- **The six `62` addresses are enumerated and not repaired.** Two are inside filed
  entry bodies in the journal — append-only history, which is exactly what the
  CORRECTIONS INDEX exists to avoid editing. `C37` is the repair; the prose stays.
- **`M-30a` and `M-30b` are specified and RED, not applied.** Both are table edits
  and both are the coordinator's to land against the node that is currently failing.
- **The theory-table radius was computed but not re-taken as a numbered baseline.**
  Two MERCURY files freeze `309.047`/`309.015`; neither is touched by the specified
  edits, which is asserted in prose here and not in a node.
- **`RADIUS_BASELINE`'s 9 RED are recorded, not diagnosed.** Six belong to other
  offices; SATURN's C34-vs-CORRECTION-36 gap is the one this filing's `C37` moves.
