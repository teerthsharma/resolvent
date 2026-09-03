# V20 R15 — it.30 — JUPITER (MYCROFT, annex owner)

**Armed `2026-09-02T18:49Z`. Readings dated below. Filed `2026-09-02T19:06Z`, inside the 20-minute wall. The journal MOVED under the readings mid-filing (detected 19:05 local) and every reading was re-taken against the moved file before filing.**
**No git writes. Nothing touched Kaggle. The CORRECTIONS INDEX was READ, never written.**

Content digests (sha256, first 16 hex, read `2026-09-02T19:02Z`, final state):

| file | digest |
|---|---|
| `V20_R15_JOURNAL.md` at 18:56Z (subject, unedited by this office) | `11b77f9c0e96b2da` |
| `V20_R15_JOURNAL.md` at 19:05Z — **it MOVED under the readings mid-filing; all re-taken** | `2bdcfc57a332bbff` |
| CORRECTIONS INDEX rows only (34 rows, joined) | `28a30ce6f6dcdbb0` |
| declaration set (row, literal) pairs only (unmoved) | `6817b208c2304e9d` |
| `tests/jupiter/test_v20_r15_it30_phrase_grammar.py` (new) | `b4276eb5e1d6f828` |
| `tests/jupiter/test_v20_r15_it29_overturns_can_fail.py` (re-declared) | `e9b7b122fc7f39b1` |
| `tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py` (unchanged) | `8707cf7574aee265` |

---

## 1. THE `none`-SENTINEL NODE: DIAGNOSED. NOT A ROUND-TRIP RESIDUE, NOT AN ORDERING DEPENDENCE.

The index reverted **byte-identical** — index-row digest `28a30ce6f6dcdbb0`, as the
coordinator reports. The node still failed. It is not the index:

```
E  AssertionError: the word `none` is live 28 times in the body; the it.27 grammar
   handed that to the grep as a retired claim
E  assert 28 == 25
```

**The node froze a count over the whole journal BODY, and the body is not the file it
was at it.29.** The three new hits, located (`2026-09-02T18:52Z`):

| line | section | text |
|---|---|---|
| 6532 | `## it.29` | ``C21`, `C30`, `C33` declare `` overturns: `none` ``, and **the it.27 grammar handed the grep the…`` |
| 6557 | `## it.29` | `…must **contain a space and be ≥12 characters**, or declare `none`.` |
| 6700 | `## it.29` | ``…against `29 of 44` was **the `none` sentinel**…`` |

**All three are the journal's own entry recording ruling J-29b.** The instrument's
subject contains the record of the instrument's finding, so **stating the finding
raised the number the finding was frozen at.**

**RULING J-30d — A READING OVER A GROWING SUBJECT MUST CARRY ITS CUT.** The datum
J-29b actually established is *"the sentinel word was live in the body the it.27
grammar greped"*, and that body is the body **written before it.29**. The node now
splits there: `25` before line 6470, and a second assertion that the total is
**strictly greater**, so the self-reference is asserted rather than tolerated. Both
halves can fail: the first on any edit to pre-it.29 text, the second if a later
entry stops discussing the sentinel.

`[RUN] 2026-09-02T19:00Z` — `test_v20_r15_it29_overturns_can_fail.py`: **7 passed**.

---

## 2. THE BLAST RADIUS, COMPUTED BEFORE THE EDIT — AND THE REASON IT BROKE

The four nodes that broke froze `26 of 41`, `15 of 41`, `25`, and an exact
membership, **naming no state**. Any row edit turns them RED and **the RED says
nothing about whether the edit was right**. That is this office's defect, not the
coordinator's: the procedure the coordinator was corrected for is the procedure this
office's own nodes made impossible to follow.

**RULING J-30a — A READING IS DECLARED AGAINST THE DECLARATION SET IT WAS TAKEN ON.**
`declaration_digest()` is a sha256 over the **(row, literal) pairs only** — not the
row prose — so it is invariant to how the coordinator words the cell and moves
exactly when a declaration moves. The it.29 count nodes now read `READINGS`:

| declaration digest | declarations | unearned | earned | state |
|---|---|---|---|---|
| `6817b208c2304e9d` | 41 | 26 | 15 | the index as it stands |
| `9ce562ee317520b5` | 31 | **7** | **24** | after the 34 specified literals land |

An unknown third digest is **RED with the blast-radius instruction**, which is the
correct answer to an un-re-taken edit — the node discriminates *un-re-taken* from
*wrong*, which it could not do at it.29.

**Both states RE-TAKEN `2026-09-02T19:05Z` against journal `2bdcfc57a332bbff`** (the counts of declarations, unearned and earned did not move; the restatement columns did, by exactly one row each, `C10`):

| | declarations | token-shaped | unearned | restated after settling |
|---|---|---|---|---|
| index today | 41 | **21** | 26 | **13** |
| after `SPEC` | 31 | **0** | 7 | **3** |

**11 of today's 13 restatements are token-shaped cells.** That is what the grammar
buys, and it is a measurement, not a prediction.

---

## 3. THE 34 LITERALS, SPECIFIED EXACTLY (RULING J-30b)

One asserting phrase per row. **20 rows already carry a phrase and are repeated
VERBATIM**, so the coordinator's edit is **13 cells**, not 34, and 3 sentinels stay.
Frozen in `SPEC` in `tests/jupiter/test_v20_r15_it30_phrase_grammar.py`; the table
below is that dict. **The 13 marked ▸ are the edit.**

| row | `overturns:` literal |
|---|---|
| ▸C1 | `no arm crosses` |
| ▸C2 | `the eval draw's own **sign census**` |
| ▸C3 | `an **8x arena cost**` |
| C4 | `**EXIT B costs zero and is already met**` |
| C5 | `SATURN and MERCURY **independently** refuted the determinism blocker` |
| C6 | `JUPITER and SATURN reached N=2 **independently, from opposite directions**` |
| C7 | `two wings the record produced, **one only prose did**` |
| C8 | `K6 shrank — two tokens **gained producers**` |
| C9 | `a **pole precisely on the unit circle**, marginal stability, an almost-all-pass filter` |
| C10 | `live-band decay **under 3% per position**` |
| C11 | `W1's fifteen failures are **M1's predicted descent to the corner**` |
| ▸C12 | `leap ledger open with **five MARS rows` |
| C13 | `**12 of 16**` |
| ▸C14 | `7 of 8 fresh vs` |
| ▸C15 | `used as an **information floor**` |
| ▸C16 | `and that separation survives everything found this iteration` |
| ▸C17 | `every GPU-second and cost ratio quoted from it.3 onward` |
| C18 | `**N = 1 primitive**, not 3` |
| ▸C19 | `every pair separates by` |
| C20 | `Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25` |
| C21 | `none` |
| ▸C22 | `the Q2/W1 pointers` |
| C23 | `**the check confirms him**` |
| C24 | `Off by one and by **fifty-eight**` |
| C25 | `this office quoted **JUPITER's unprompted count** and did not check it` |
| C26 | `repeated in **three places each**` |
| C27 | `**both branches fired in test**` |
| ▸C28 | `five :466 sites` |
| ▸C29 | `premises, read as file sizes` |
| C30 | `none` |
| C31 | `the INSPECTOR is **still auditing**` |
| ▸C32 | `Correction 31's own numbers` |
| C33 | `none` |
| C34 | `Occurrences 129 -> 120, pointers 103 -> 96` |

**The shape rule, enforced:** a space, **≥12 characters**, **no backtick** (the
field's own delimiter — a literal containing one cannot be parsed back out) and **no
pipe** (the table's). `test_every_declared_literal_on_the_index_is_phrase_shaped` is
**RED now** and names all 21 offending cells; it lands GREEN when the 13 rows change.

**MEASURED LIMIT, filed against this office's own rule:** two of the coordinator's
four mechanical truncations — `the eval draw's` (15 chars) and `EXIT B costs` (12) —
**PASS the shape rule.** Shape is *necessary and not sufficient*, which is why the 34
are specified by judgment and frozen, not derived. Asserted in
`test_the_grammar_rejects_the_cells_it_is_meant_to_reject`, as a positive, so it
cannot be read as the rule working.

---

## 4. CLASS C, SETTLED. THE READING WAS WRONG.

**RUN, `2026-09-02T18:53Z`, exactly as J-29c specified it:** a hit is a MENTION if its
line also names the retiring correction (`C{n}`) or the settling `it.N`; else USE.

**23 live hits across the 9 literals. 22 USE, 1 MENTION** (`C32 36,786` at line 4525,
*"All three of `C32`'s claims hold…"*).

**Class C is not a mention class. The reading is withdrawn and Class C collapses into
Class A** — the same over-claim, the same remedy: `C22` → `the Q2/W1 pointers`,
`C29` → `premises, read as file sizes`, `C32` → `Correction 31's own numbers`
(`C24` and `C26` already carry phrases and need no change). Three thirds, one remedy.

**RULING J-30c — THE SETTLING ENTRY'S OWN QUOTE IS THE WITHDRAWAL, NOT A
RESTATEMENT.** The it.29 chronology cut was the **start** of the settling section,
which counts the correction quoting the claim it retires as the body still asserting
it — `C24` and `C26` are exactly that (it.16 quoting it.14 in order to withdraw it).
The cut moves to the **end** of the settling section. `restated()` is that predicate;
`test_restated_is_narrower_than_live_and_both_can_observe_a_violation` asserts it is
a **proper** subset of the live hits and non-empty, so the ruling is neither a no-op
nor an amnesty. The it.29 membership node is **retired into** the successor, which
freezes membership for **both** declaration sets.

**What survives `SPEC`, named, not counted:** `C1 no arm crosses`, `C10 live-band decay **under 3% per position**` and `C19 every pair separates by`. `C10` entered the class **during this filing**, when the coordinator's mid-filing journal edit restated it after it.9 — the node observed the movement and the reading was re-taken rather than re-frozen. `C1`'s two are **J-30d again** — the
it.28 and it.29 entries quoting the phrase while ruling on it. **`C19`'s is a real
live restatement at line 2530 and is the one row this grammar does not repair.**

---

## 5. EVERY SHIPPED ASSERTION, SHOWN ABLE TO OBSERVE A VIOLATION

1. `test_every_declared_literal_on_the_index_is_phrase_shaped` — **RED right now**, 21 named cells. Reachability is not argued; it is the current state.
2. `test_the_specified_replacements_satisfy_the_grammar_they_demand` — the demand is not unsatisfiable; fails if `SPEC` drifts off the index one-for-one.
3. `test_the_grammar_rejects_the_cells_it_is_meant_to_reject` — CONTROL, both directions, including the two truncations that pass.
4. `test_the_specified_literals_are_not_restated_after_their_row_settled` — exact membership for both states; moves if either exception is repaired.
5. `test_restated_is_narrower_than_live_and_both_can_observe_a_violation` — CONTROL on the cut itself: proper subset, non-empty, and `C24` specifically retired.
6. `test_the_blast_radius_of_the_spec_edit_is_declared_before_it_lands` — fails if `READINGS` lacks the post-`SPEC` entry, i.e. if this office asks for an edit it has not re-taken.
7. it.29's `reading()` — an undeclared declaration set is RED **with the instruction**, separating *un-re-taken* from *wrong*.

`[RUN] 2026-09-02T19:01Z` — `python -m pytest tests/jupiter/ -q` → **`6 failed, 258 passed in 55.40s`**.
Failures: **it.30's shape node** (RED first, lands with the 13 cells) · **it.27's
enforcement node** (RED since it.29, same landing) · **4 pre-existing** — `it18_citation_landing` ×3, `it4_merge_is_unexercised` ×1, both failing before this work and untouched by it.

---

## 6. WHAT THIS OFFICE DID NOT REACH

- **`C19`'s live restatement at line 2530 is not repaired.** The grammar narrows the cell; the body still asserts the phrase after it.3 settled. That is a row change or a withdrawal, and it is the coordinator's.
- **`C1`'s two post-cut hits are self-reference (J-30d) and are not subtracted by any predicate here.** The rule "a hit on a line that is itself ruling on the literal is not a restatement" is specified and **not built** — it needs a marker the journal does not carry, and inventing one on the last minute of the wall is how it.27 shipped a guard that could not fail.
- **The 13 replacement phrases were chosen from each row's own claim column by judgment.** They are frozen and testable but they are not derived; a second office reading the same rows could pick different phrases and the shape rule would accept those too.
- **The index is not edited.** All 34 literals are specified above and enforced by a node that is RED until they land.
- **`6 of 6`, `118 of 131`, the `129 → 131` movement and `WANT_SEAL abad6a77…` were not re-verified**, per standing instruction.
