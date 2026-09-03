# V20 R15 — it.32 — JUPITER (MYCROFT, annex owner)

**Both table edits landed. Two rulings on other offices' deviations, one against this
office's own instrument.**

All clock stamps below are read from `date -u`, not inferred. Window opened
`2026-09-02T13:46:32Z`, closed `2026-09-02T13:52Z`, this box, branch `v17k-gate0`,
tree at `207e7b9` plus this iteration's three theory-table line edits. No git writes.
Nothing touched Kaggle.

---

## 1. RULING J-32a — THE `C37` GRAMMAR DEVIATION WAS RIGHT, AND THE SPEC WAS DEFECTIVE TWICE

**The coordinator was right to apply `| C37 |` rather than the specified `| 37 |`.**
The index grammar is `^\| C\d+ \|`; the digest recipe and `dead_literals()` both
require the `C`. Applied verbatim, `C37_ROW` would have been invisible to every
instrument that reads the index — a row that publishes a correction no instrument can
see is worse than no row, because it reports coverage it does not have. **A
coordinator that applies a spec into invisibility has obeyed the letter and destroyed
the function. The deviation is affirmed and the record of it is the right form.**

**The spec was defective in a second way the deviation report did not name.** The
applied row carries **six** columns; `C37_ROW` specified **five** and its own node
asserted `C37_ROW.count("|") == 6`, which is 5 columns. The settling-iteration column
(`it.29`) is missing from the spec entirely — and `settled_at()`, this office's own
predicate at it.30, reads exactly that column to decide when a literal counts as
restated. **A spec written by the office that owns `restated()` omitted the field
`restated()` runs on.** The coordinator supplied it. Both deviations were repairs,
neither was discretionary, and the count of them is two.

**Mechanism, for the taxonomy:** the spec was validated by a node this office wrote
(`test_c37_is_specified_and_its_literal_obeys_the_it30_grammar`) that checked the
literal's shape and the pipe count and never checked the row against the grammar of
the table it is inserted into. **A shape check on a row is not a check that the row
parses.** The node passed on a row that would not have been read.

## 2. RULING J-32b — MARS'S NO-POINT BAND IS UPHELD, AND IT IS WHAT LANDED

MARS struck `M-30`'s remedy while this office was filing, and he is right.

`_sweep(e) = 3 · Σ_{S∈{32,64,128}} (S/64)^e · Σ means` is one monotone family on
`e ∈ [1,3]`. The published band endpoints are `_sweep(1.0)` and `_sweep(3.0)`; the
published point is `_sweep(2.0)`. **Containment of the point by the band is arithmetic
in `e`, not a measurement.** A band that cannot fail to contain its point corroborates
nothing about the point, and a price quoted as *band + point* claims an independent
centre it does not have. His second strike is the same blade: `309.015` (the formula
as written) and `309.047` (the withdrawn re-measured object) **both round to `309.0`**,
so the prescribed one-decimal point reinstates at one decimal exactly the distinction
`M-30` was made to draw.

**The remedy that landed is MARS's: quote `206–537 GPU-s`, band only, NO POINT.**
`M-30`'s `point 309.0` is rejected at both addresses. This office's own error message
in `test_m30a…` still prescribes the pointed form; the message is now wrong and the
assertion is not — noted below under what was not reached.

## 3. THE TWO TABLE EDITS, LANDED

`V20_R15_THEORY_TABLE.md`, three lines, **no line inserted or deleted — the file is
443 lines before and after**, so every `file:line` pointer into it survives the edit.
That was the constraint, and it was met by editing in place rather than adding rows.

| line | before | after |
|---|---|---|
| `:184` | *"its own formula gives `309.047`"*, band `206–537` unqualified | *"Price: `206–537 GPU-s`, band only — NO POINT"*, endpoints named as `_sweep(1.0)`/`_sweep(3.0)` with `e ∈ [1,3]` unmeasured; *"the formula as written gives `309.015`; the withdrawn `309.047` is `_sweep(2.0)` on it.13's re-measured means — a different object"* |
| `:354` | *"LEAPABLE by `~275 GPU-s` and one argparse line"*, no pointer | *"LEAPABLE by `206–537 GPU-s`, band only, no point (`V20_R15_IT13_MERCURY.md:146`)"* |
| `:306` | *"has **zero callers under `scripts/`**"* | *"has **two callers under `scripts/`** — `scripts/v20_m14_cheeger.py:347` and `:462`, both at the registered `(n=500, d=4, seed=7)`, both consuming the kernel `b["K"]` and never the bed's data, so no cell of BED-K's shape has been run (`:152`)"* |

**Both nodes are GREEN**, `[RUN] 2026-09-02T13:50Z`:

```
tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py  5 passed, 2 failed
  test_m30a_the_theory_table_still_attributes_309_047_to_the_formula   PASSED
  test_m30b_the_admission_condition_asserts_zero_callers_and_there_are_two  PASSED
```

**The `:306` repair could not use the wording this office prescribed.** The prescribed
text ended *"so zero cells of BED-K's shape have ever been run (`:151`)"*, and the same
node asserts `true_at == [152]` — the true clause must occur **once**. Writing the
prescription verbatim would have made `true_at == [152, 306]` and turned the node RED
on the repair it demanded. **This is J-31a firing inside a single file: the remedy text
republished the literal the node counts.** The clause was reworded (*"no cell of
BED-K's shape has been run"*) and pointed at `:152` — MERCURY's `:151` is off by one
and is corrected in passing, not silently. **The grade does not move: `Q2/W1` stays at
its admission condition unsatisfied.** The edit is to the EVIDENCE clause only.

## 4. `MARS-31-C` — NAMED, NOT EDITED

`309.047` survives as a **green assertion** at
`tests/mercury/test_v20_r15_it13_phase_c_price.py:48` and `:55`, in an instrument whose
docstring at `:47` says `309.02` while its assertion says `309.047`. **These are
MERCURY's files and this office does not edit them.** MARS's framing is exact and is
adopted: *the `62` survived in prose; `309.047` survives in a green assertion*, which
is the worse case, because a green assertion is read as a measurement.

The `M-30a` withdrawal now has **three** repaired addresses (`:184`, `:354`, and the
`:306` pointer correction) and **two** live unrepaired ones. **A withdrawal published
at three of five addresses is not a withdrawal** — J-31b, turned on this office's own
remedy.

## 5. RULING J-32c — THE PREFIX CUT IS WRONG FOR A TABLE THAT GROWS AT THE TOP

Applying J-31a to this office's own instruments, as instructed, and it convicts one.

`frozen_prefix_count(hits, cut_heading)` freezes hits **before the first `## it.N`
heading named**. `test_the_c37_literal_is_live_where_the_row_says_it_is` froze
`before == 1` with the cut at `## it.30`. It is now **2**, and the second hit is the
`C37` row itself.

**The CORRECTIONS INDEX sits above every `## it.N` heading in the journal and grows
downward from the top.** Every row ever added lands inside the frozen prefix of every
chronological cut. **A chronological cut cannot freeze a count over a table that grows
above the cut.** The mechanism is J-31a's — the corpus republishes the thing counted —
but the repair J-31a prescribes does not work here, because the cut is in the wrong
dimension: chronological, when the growth is structural.

**The correct cut is structural: exclude index rows.** `hits_in()` already takes the
`indexed` line set and already does this; `substring_hits()`, written three days later
for the same journal, does not. **Two scans over one corpus with two exclusion rules,
and the newer one is the weaker.** The cut is specified here and not applied — see
what was not reached.

`test_the_withdrawn_62_is_still_live_at_more_than_one_address` is RED for the same
reason at a different address: `journal == [5903, 6389]` is a frozen pair of line
numbers into a file that has since had three index rows and an it.31 entry written
into it. **A frozen line number is a frozen count wearing a different hat.**

## 6. NODE STATE, RECONCILED ACROSS THREE GENERATIONS

`[RUN] 2026-09-02T13:49Z`, `pytest tests/jupiter -q`, **17 RED / 254 GREEN**
pre-edit; the two `M-30` nodes have since gone GREEN, leaving **15 RED / 256 GREEN**
by attribution (the full re-run did not fit the window — see below).

| generation | node | why RED | class |
|---|---|---|---|
| it.31 | `test_m30a…` | — | **REPAIRED this iteration** |
| it.31 | `test_m30b…` | — | **REPAIRED this iteration** |
| it.31 | `test_the_withdrawn_62_is_still_live…` | frozen journal line pair `[5903, 6389]` moved | J-32c, structural |
| it.31 | `test_the_c37_literal_is_live…` | frozen prefix `1 → 2`, the `C37` row is the second hit | J-32c, structural |
| it.30 | `test_the_specified_replacements…` | `SPEC` has 34 rows, the index now has 37 | undeclared state |
| it.30 | `test_the_specified_literals_are_not_restated…` | frozen `13` restatements, measured against the pre-`SPEC` index | undeclared state |
| it.30 | `test_the_blast_radius…` | `declaration_digest()` is now a fourth state, in neither `READINGS` entry | undeclared state |
| it.30 | `test_every_declared_literal_is_phrase_shaped` | C35/C36/C37 literals not in `SPEC` | undeclared state |
| it.29 | `test_the_none_sentinel_is_the_whole_26_of_41…` | `41` denominators, index is 37 rows / 40 declarations | undeclared state |
| it.29 | `test_the_narrowed_node_can_observe_a_violation` | same | undeclared state |
| it.29 | `test_the_narrowed_node_is_not_red_for_everything` | same | undeclared state |
| it.27 | `test_an_overturns_aware_grep…` | predates the chronology cut J-30c/J-30d established | superseded generation |

**Nine of the twelve are one defect: an undeclared declaration state.** The index moved
from 34 rows to 37 and from 41 declarations to 40 in one window (13 token cells became
phrases, three rows were added), and every node that froze a count against the old set
went RED without saying anything about whether the move was good. **That is precisely
what `READINGS` exists to prevent, and `READINGS` carries two states where it needed
four.** The mechanism is J-30a, restated: *a reading is declared against the
declaration set it was taken on* — and this office declared two sets, then let a third
and a fourth land.

**The it.27 node is not in that class and should not be retired into it.** It predates
J-30c/J-30d and its cut is the superseded one. **It is a superseded generation, not an
undeclared state**, and merging the two would let a wrong cut retire under an amnesty
written for a right one.

## 7. THE BLAST RADIUS, RE-TAKEN

The 14 files that read `V20_R15_JOURNAL.md`, run as a set.

| reading | when | result |
|---|---|---|
| declared baseline (it.31, `207e7b9`) | `2026-09-02T19:05 IST` | **9 RED / 97 GREEN** |
| **re-take, post-`SPEC` / post-C35–C37 / post-table-edits** | **`2026-09-02T13:49:59Z`** | **16 RED / 90 GREEN** |

**The re-take does not reproduce the baseline pair, and the deviation is attributed,
not excused.** The theory table's citation population is **131 pre-edit, 133
post-edit**: exactly two pointers added, `V20_R15_IT13_MERCURY.md:146` at `:354` and
`scripts/v20_m14_cheeger.py:347` at `:306`. **Both were added deliberately** — the
`M-30a` remedy is *no point without a pointer* and the `M-30b` remedy is *name the
callers* — and both move a census frozen at **129**, which had **already** moved to
131 before this iteration touched anything.

**Attribution, exact:** of the 7-node deviation, **4 are this office's**
(`it20_citation_freeze` ×3, `it21_heading_anchor` ×1 — all census-population nodes
that the `+2` pushed past their frozen totals). The remaining 3
(`mercury/test_v20_r15_it22_independent_census` ×3 net,
`mercury/test_v20_r15_it27_uncited_class` ×4, `saturn` planted negative ×1, against
a baseline that already contained some of them) **were not separated within the
window** and are named as unattributed rather than assigned.

`saturn::test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught` fires for the reason
the coordinator recorded: the index leads the body, 37 to 36, because `C37` is a
ruling filed in a report and not a body `CORRECTION`. **The planted negative is
correct and the gap is real.** The body entry that closes it is `CORRECTION 37` in the
it.32 body write-up, which this office does not own.

## Limits

The window was 20 minutes and closed with the following not reached. The full
`tests/jupiter` re-run after the table edits did not fit; the **15 RED / 256 GREEN**
figure in §6 is the pre-edit 17 minus the two nodes measured GREEN individually, not a
whole-suite reading, and is marked as such. The structural cut specified in J-32c is
specified and **not implemented** — `substring_hits()` still lacks the index-row
exclusion `hits_in()` has. No fourth `READINGS` entry was written, so the nine
undeclared-state nodes are diagnosed and still RED. The `test_m30a…` failure message
still prescribes `point 309.0`, which J-32b now rejects; the assertion is correct and
the message is stale. The 3 unattributed radius nodes in §7 need a pre-edit run of the
mercury and saturn thirds to separate, which was not taken. Two green assertions of
`309.047` in `tests/mercury/test_v20_r15_it13_phase_c_price.py` are named and, being
MERCURY's, untouched.
