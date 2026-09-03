# V20 R15 it.18 — JUPITER (MYCROFT): THE LANDING CENSUS, RE-RUN ON THE REPAIRED TABLE

**Status: WRITTEN AS CHUNKS LAND.** it.16 died holding the census in context. it.17 did not
die — the brief for this iteration was cut against a 10,147-byte snapshot of
`V20_R15_IT17_JUPITER.md`; that file is now 21,774 bytes and its `## 2 — CENSUS RESULT`
is populated with 28 failures and 29 repairs. **The brief's premise is stale.**

That changes what this iteration owes. it.17's census is not re-run for its own sake. It is
re-run because **it.17 exhibited only its failures.** The 95 it declared LANDS were never
shown, its instrument's `MANIFEST` holds **29** citations and not the full population, and
the office's own §1 precedent — *the citation layer is unasserted in a file whose purpose is
to end unasserted claims* — applies to it with full force. A census that reports a count
and shows one quarter of it is the `123/123, 0 bad` failure in a different costume.

## 0 — FIRST FINDING, BEFORE A SINGLE RULING: THE POPULATION IS NOT 123

**`[RUN]`** mechanical extraction from the table at HEAD, same regex and same extension
filter as `tests/jupiter/test_v20_r15_it14_theory_table.py::bad_citations`
(`CITE_RE = r"`([^`]+?):(\d+)`"`, filtered to `.py|.md|.lean|.jsonl|.json|.txt|.toml|.hs|.ipynb|.pgn`):

```
TOTAL CITATIONS = 128
```

**128, not 123.** The five are it.17's own repairs: `C44` replaced one citation
(`ceq/hankel.py:1`) with **four** (`:108`/`:131`/`:279`/`:75`), and `C59` replaced one with
two (`:13`–`:14`). Repairing a citation by splitting it changes the size of the population
being censused. **Every count in the it.17 report — `123`, `95`, `28`, `22.8%` — is stated
against a population that no longer exists at HEAD.** RULING J-17e held the table's
*line count* fixed at 443 and it did; nobody held the *citation count* fixed, and it moved.

> **RULING J-18a. The census population is defined by extraction at HEAD, not by a carried
> constant.** `123` is now a historical figure naming the pre-repair table. This iteration
> censuses **128**.

## 1 — CENSUS RESULT

*(chunks appended as they land)*

| chunk | citations | LANDS | DOES NOT LAND |
|---|---|---|---|
| 1 | C1–C33 | 33 | **0** |
| 2 | C34–C66 | 31 | **2** |
| 3 | C67–C98 | 32 | **0** |
| 4 | C99–C128 | 29 | **1** |
| **TOTAL** | **128** | **125** | **3 — 2.3%** |

### CHUNK 1 — C1–C33. **33 LAND, 0 FAIL.**

Every it.17 repair in this range is confirmed at HEAD by reading the cited line:
`WING_MANIFEST.md:3` carries *"Frozen at it.4 on branch `v17k-gate0`"* (C1, the repaired
`:12`); `LEAP_LEDGER.md:25` carries **L-4**, the *withdrawn* sense, and `:24` carries **L-3**,
*domain-empty* (C10/C11, the repaired label swap); `IT13_MERCURY.md:146` carries the
`206–537` route row with *"one `argparse` line … two `cuda.synchronize()`"* (C27, the
repaired `:141`). The five Lean declarations of Q1/W1 (C28–C32) each land on their own
`theorem`/`lemma` keyword line.

### CHUNK 2 — C34–C66. **31 LAND, 2 FAIL.**

The `ceq/hankel.py` split (C45–C48) lands four for four: `:108` `def hankel_block`, `:131`
`def rank_real`, `:279` `def rank_plus_lower`, `:75` `NEG_ENTRY = _Sentinel("NEG_ENTRY")`.
The `it12_constants.py` `+10` displacement is gone: `:11`, `:12`, `:13` each carry their
constant (C51, C56, C63). `it7_q3.py:106` carries `def test_w3_lambda_hat_sign_separates_all_eight_arm_pl_cells` (C65).

**THE TWO FAILURES ARE BOTH NEW, AND BOTH ARE it.17's OWN.**

| # | cited as | the table's claim | what the cited line actually is | true location |
|---|---|---|---|---|
| **C17** | `V20_R15_JOURNAL.md:648` | *"the `F1` is an it.3 SCOREBOARD reading **on the far side of the it.4 freeze**"* | **`---`** — a horizontal rule | **`:650`** — `## it.4 — THE FREEZE. `FROZEN-N = 2`.` |
| **C64** | `scripts/v15_r1.py:801` | *"journal `manifest.smp_values` on the **`_0step` record**"* | `torch.manual_seed(seed)` | **`:804`** — `r["sign_acc_0step"] = probe(...)`, the first `_0step` field; the block runs `:804`–`:808` |

**C17 is the it.17 report's own decisive fact, and the report states it wrong in prose.**
`V20_R15_IT17_JUPITER.md` §3.4 reads: *"`V20_R15_JOURNAL.md:648` is `## it.4 — THE FREEZE.
FROZEN-N = 2.`"* It is not. `:648` is `---` and `:650` is the heading. **The ruling J-17d
survives intact** — the `F1` at `:645` really does sit before the freeze, and the freeze
really is two lines further on — **but the pointer carrying that ruling is off by two, onto
a horizontal rule.** This is mechanism 2 of it.17's own taxonomy (*off-by-a-row onto an
adjacent line of the same register*) committed **inside the repair that catalogued it**,
and it is in the table at `V20_R15_THEORY_TABLE.md:55` where the leap reads it.

**C64 is worse than a line slip.** `[RUN] grep -n "smp_values" scripts/v15_r1.py` returns
**nothing**. The cited symbol does not exist in the cited file at any line, so no repair of
the digits fixes this cell — the ROUTE field of Q3/W1 promises a journal field by a name the
producer never defines. The `_0step` record it means is real and is at `:804`–`:808`.

> **RULING J-18b. C64 is not a citation defect, it is an unbacked ROUTE.** A `path:line`
> whose named symbol is absent from the whole file is the `Lean #21 [S]` failure — cited by
> name, no source declaration behind it — which is the defect this round already withdrew
> six citations for (`V20_R15_LEAP_LEDGER.md:25`, L-4). It is recorded here and **not
> silently renumbered**, because renumbering it to `:804` would assert that `:804`
> journals `manifest.smp_values`, and it does not.

**C64 CORRECTED, ONE STEP FURTHER, BEFORE THE RULING WAS LEFT STANDING.**
`[RUN] grep -rn "smp_values" scripts/ ceq/ scale/` →

```
ceq/arm_smprime.py:409:    m["smp_values"] = values
```

The symbol exists — **in a different file.** `manifest.smp_values` is written by
`ceq/arm_smprime.py:409`, not by `scripts/v15_r1.py` at any line. So C64 is **mechanism 5
of it.17's taxonomy — ONE WRONG FILE ENTIRELY** — the same species as C98, and the second
instance of it in this table. **True location: `ceq/arm_smprime.py:409`.** RULING J-18b is
withdrawn: the ROUTE is backed, by a producer in the arm rather than in the runner, and the
citation is repairable after all. **The `_0step` record half of the same sentence still
points at `scripts/v15_r1.py:804`, so the cell needs two citations where it carries one.**


### CHUNK 3 — C67–C98. **32 LAND, 0 FAIL.**

The Inspector's named example is confirmed repaired at HEAD: **`CEQ_V20_R15_CONTRACT.md:123`
carries `W1 to the oracle where a state distribution exists`** verbatim (C92, the repaired
`:107` that was off by 16). `arm_smprime.py:577` is the `return self.readout(h).squeeze(-1)[:, seq - 1]`
and `:144` is `def path_product` (C93, C68) — the def-line/body-line rule holds in both
directions. `negation_scope.py:286` is the `def equilibrium_oracle` and `:304` is its
`return z` (C95, C94), the split it.17 ruled on its own. `it8_q4_q5.py:127` and `:151` carry
their two node names (C87, C90). `IT8_JUPITER.md:105` is the `O(n·S²)` row, not the
table rule above it (C67).

### CHUNK 4 — C99–C128. **29 LAND, 1 FAIL.**

The `tests/saturn/test_v20_r15_it12_saturn.py` `−23` displacement is gone at all four sites:
`:319` `def inadmissible_leapable_rows`, `:329` `if head.startswith("none")`, `:343`
`KNOWN_INADMISSIBLE = {...}`, `:353` `def test_the_FIELD_detector_fires_on_a_planted_violation`
(C113, C114, C117, C118). `MISTAKES.md:451` is `### M-2. A threshold refitted to the data it
judges` (C109). `LEAP_LEDGER.md:131` does carry *"The one registered bed whose output is
categorical is the chess witness"* — checked against the **full 1,960-character line**, not
its first 200 (C104, C119).

| # | cited as | the table's claim | what the cited line actually is | true location |
|---|---|---|---|---|
| **C103** | `V20_R15_IT89_INSPECTOR.md:63` | *"`[RUN] python -c "from ceq import kdata; print(list(kdata.BED_SPECS))"` returns `['bed_m', 'bed_k', 'bed_1']`"* | the `NRMSE = √2` row — `1.4060346618513293 against √2` | **`V20_R15_IT13_MERCURY.md:189`** — a different office's file |

**C103 is C98's mirror image, and the two point at each other.** it.17's C98 repair moved a
citation **out of** `V20_R15_IT13_MERCURY.md` **into** `V20_R15_IT89_INSPECTOR.md:63`, and it
was right to: `:63` does carry `1.4060346618513293`. **C103, forty lines earlier in the same
table, points the other way and is wrong** — its `BED_SPECS` run is in
`V20_R15_IT13_MERCURY.md:189`. `[RUN] grep -n "BED_SPECS" V20_R15_IT89_INSPECTOR.md` returns
**nothing**: the quoted command and its quoted output appear nowhere in the cited file.

> **RULING J-18c. Two citations between the same pair of files, each pointing at the content
> that is in the other, is not two independent slips.** it.17 repaired the one it sampled and
> the census caught the one it did not, which is the whole argument for censusing rather than
> spot-checking. **The wrong-file mechanism now has two confirmed instances (C98, C103) and
> both involve `IT13_MERCURY.md` ↔ `IT89_INSPECTOR.md`.**

### THE TWO MARGINALS, NAMED RATHER THAN COUNTED EITHER WAY

Neither is scored above. Both are recorded so a later reader does not discover them as
"missed":

- **C13 — `CEQ_V20_R15_CONTRACT.md:239` for M14's `F3`.** `:239` is the M14 entry's head
  line (`M14 CHEEGER STRATIFICATION. φ²/2 ≤ 1−λ₂ ≤ 2φ [V]`); the token `F3` is at **`:241`**.
  It lands under the block-head convention and fails under the strict value rule. The table
  cites the same entry as `:239-243` elsewhere, which is unambiguous — **the bare `:239` is
  the form that is not.**
- **C120 — `lean/lakefile.lean:1` carrying `toolchain leanprover/lean4:v4.7.0`.** `:1` is
  `import Lake`, a legitimate `:1`-as-module citation for the Lake build. **The toolchain
  string is not in that file at all** — it is the entire contents of `lean/lean-toolchain`,
  which the table never cites. Mechanism 3 at its boundary.

## 2 — THE COUNT, THE FAILURES, AND WHAT MOVED

**128 censused. 125 land. 3 do not.**

| # | cited as | true location | mechanism (it.17's own taxonomy) |
|---|---|---|---|
| **C17** | `V20_R15_JOURNAL.md:648` | **`:650`** | 2 — off-by-a-row onto an adjacent line |
| **C64** | `scripts/v15_r1.py:801` | **`ceq/arm_smprime.py:409`** (`m["smp_values"] = values`); the `_0step` record is `scripts/v15_r1.py:804` | 5 — wrong file entirely, **plus** a second claim needing its own citation |
| **C103** | `V20_R15_IT89_INSPECTOR.md:63` | **`V20_R15_IT13_MERCURY.md:189`** | 5 — wrong file entirely |

**Two of the three were introduced at it.17; the third was declared a landing by it.17
and is not.** C17 is prose it.17 wrote, C103 is a pointer it.17 repaired (see §4.2),
and **C64 is older than both — it sat in it.17's own 123-citation population and came
back LANDS**, which is the one failure mode a census exists to prevent.
The failure rate fell from **28/123 (22.8%)** to **3/128 (2.3%)** — and the residue is not
the tail of the old distribution. **It is the repair's own error rate.** Twenty-nine
citation strings were rewritten and two of them landed wrong, which is `6.9%` of the repairs:
**three times the failure rate of the table they were repairing.**

> **RULING J-18d. A citation repair is a citation, and inherits the standard.** it.17's
> instrument checks the 29 repaired pointers against hand-verified substrings and passes;
> **C17 and C64 are not in its `MANIFEST`**, because the manifest holds the citations it.17
> *repaired*, not the citations it.17 *wrote*. A prose sentence asserting `:648` is the
> freeze heading is exactly as load-bearing as a cell citation and was checked by nothing.

## 3 — THE DIGEST

**`THEORY-SHA256` is unchanged: `9989f0efabe1895a9772f650ba1c3bcb77db7d37550879f2fdc9e8e9f2c63057`.**
No repair was applied to the table this iteration, so nothing could have moved it. it.17
already established the stronger fact: **the digest could not move even if the three failures
above were repaired**, because `theory_cells()` reads only the twelve §1 grade tokens and is
blind to all 128 citations (`V20_R15_IT17_JUPITER.md:274-290`). The brief's expectation that
`9989f0ef…` *"will move — that is correct"* is **false on this table's construction**, and
its accompanying claim that *"SATURN's per-cell keyed digests identify which cells changed"*
is false on **eight of twelve cells**: `row_digest` hashes the row value alone, not
`cell|row`, so `1eb8872d017f62d7` is four cells at once.

## 4 — THE INSTRUMENT, RED FIRST, THEN THE REPAIRS

`tests/jupiter/test_v20_r15_it18_citation_landing.py`. Six nodes. The manifest is three
`(path, lineno, substring)` triples plus the three withdrawn pointers, hand-verified at
HEAD and **not** generated from the table.

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it18_citation_landing.py -q`** against
the **unmutated table, before any repair**: **`2 failed, 4 passed in 0.51s`** — RED on
`test_the_three_repaired_pointers_are_present_in_the_table` and
`test_no_withdrawn_pointer_survives_in_the_table`; GREEN on the three that prove the
*targets* were right before the table was touched, and on the planted negative.

**The planted negative is a wrong line inside a file that exists**, per the it.17 precedent:
it shifts `CEQ_V20_R15_CONTRACT.md:123` — a citation that **lands** — to `:133`, which still
resolves, so the it.14 checker stays green on it, and asserts this checker names it anyway.

**The three repairs, in place, table length unchanged.**

```
lines before/after: 443 443
```

| line | was | now |
|---|---|---|
| `:55` | `V20_R15_JOURNAL.md:648` | `V20_R15_JOURNAL.md:650` |
| `:168` | `manifest.smp_values` … @ `scripts/v15_r1.py:801` | `manifest.smp_values` @ `ceq/arm_smprime.py:409` … @ `scripts/v15_r1.py:804` |
| `:224` | `V20_R15_IT89_INSPECTOR.md:63` | `V20_R15_IT13_MERCURY.md:189` |

**`[RUN] python -m pytest tests/jupiter/ tests/saturn/test_v20_r15_it14_saturn.py -q`** after
the repairs: **`1 failed, 203 passed in 54.16s`**. The single failure is
`test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record`
— **a standing RED that predates this iteration**, touching no citation. **Control: it.17
records the same node red for the same reason before these edits.**

### 4.1 THE REPAIR MOVED THE POPULATION AGAIN, AND THIS OFFICE'S OWN NODE CAUGHT IT

`test_the_population_is_128_not_123` went **RED on the repair that had just been applied**.
C64's ROUTE carried one citation for two claims — the producer of `manifest.smp_values` and
the `_0step` record — so repairing it correctly **adds** a citation. **128 → 129.**

> **RULING J-18e. J-18a is not a remark about it.17, it is a property of citation repair.**
> This office charged it.17 with moving the population from 123 to 128 and then moved it from
> 128 to 129 in the same file, one section later. The node is left asserting
> `POPULATION_AT_IT18 = 129` **with the iteration in the constant's name**, because a census
> count is a dated measurement and every undated one in this round has gone stale.

### 4.2 C98 AND C103 ARE THE SAME CITATION. it.17's REPAIR AIMED AT THE WRONG CLAIM.

Applying the C103 repair turned `test_v20_r15_it17_citation_landing.py::test_every_repaired_pointer_is_present_in_the_table`
RED on exactly one entry: `V20_R15_IT89_INSPECTOR.md:63`. **That pointer occurs once in the
table, at `:224`, which is C103** — so it.17's C98 and this office's C103 are **one citation,
repaired twice, to two different files.**

it.17 read the cell's claim as *"STRUCK and KILLED — measured `1.4060346618513293`"* and
sent the pointer to the line carrying that number. **The sentence that actually holds the
citation is the `BED_SPECS` `[RUN]`**, and `[RUN] grep -n "BED_SPECS" V20_R15_IT89_INSPECTOR.md`
returns **nothing**. The original `V20_R15_IT13_MERCURY.md:196` was also wrong — it is a
`python-chess` pin row — but wrong by **7 lines in the right file**. **The repair moved it to
a different office's file, which is further from the truth than the defect it replaced.**

> **RULING J-18f. A repair aimed at the wrong claim is worse than the defect.** `:196` was
> recoverable by a reader who kept scrolling; `IT89_INSPECTOR.md:63` is not recoverable at
> all, and it reads *more* plausibly because the number it lands on is a real number from
> the same argument. **it.17's mechanism 5 ("one wrong file entirely") had one instance when
> it was written; it has one now — and it is the repair, not the original.**

it.17's `MANIFEST` and `WITHDRAWN` maps are corrected in place at
`tests/jupiter/test_v20_r15_it17_citation_landing.py:66` and `:96`, with the reason in a
comment beside them, rather than deleted. Its `V20_R15_IT13_MERCURY.md:196` withdrawal
**stands** — that pointer is still wrong; only its target changed.

## 5 — THE DIGEST

**`THEORY-SHA256` is unchanged: `9989f0efabe1895a9772f650ba1c3bcb77db7d37550879f2fdc9e8e9f2c63057`,
and `tests/saturn/test_v20_r15_it14_saturn.py` is green inside the 203 above.**

The brief expected it to move. **It cannot.** `theory_cells()` reads only the twelve §1 grade
matrix rows — its own docstring closes *"Rows only"* — so `THEORY-SHA256` is a digest over
**twelve grade tokens** and is blind to all 129 citations and to all six fields of all twelve
cell bodies. Three citation repairs inside the cells leave it bit-identical, exactly as
it.17's twenty-nine did.

**The brief's second half is also false.** *"SATURN's per-cell keyed digests identify which
cells changed"* fails on **eight of twelve cells**: `row_digest` hashes the row value alone
rather than `cell|row`, so `1eb8872d017f62d7` is Q2/W3, Q3/W3, Q5/W1 **and** Q5/W3 at once,
`63f81989362491d3` is both Q4 cells, and `f53365e65ce4d925` is both Q6 cells — **7 distinct
digests for 12 cells.** Swap Q4/W1 with Q4/W3 and nothing moves.

## 6 — LIMITS

The instrument asserts **3 landings and 3 withdrawals out of 129 citations**; the other 123
are ruled in this file and **checked by nothing**, which is the same standing it.17's census
had and the reason this one re-ran it. A node asserting all 129 needs 129 hand-verified
substrings and was not built inside the cap. C13 (`CONTRACT.md:239` for a grade token at
`:241`) and C120 (`lakefile.lean:1` carrying a toolchain string that lives in
`lean/lean-toolchain`) are marginals under a convention this office has still not written
down, and are unrepaired. The four open items from the brief — Q5's grade-and-disclaimer,
Q2/W1's forbidden estimate, Q2/W1's missing ledger row, and M14's three grades — were
**ruled at it.17 (J-17a–d) and are not re-opened here**; the census was the deliverable. No
git write was made. Nothing touched Kaggle.
