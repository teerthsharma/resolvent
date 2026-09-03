# V20 R15 it.17 — JUPITER (MYCROFT): THE 123-CITATION CENSUS BY THE INSPECTOR'S STANDARD

**Status: SKELETON FILED FIRST.** it.16 died holding the verification in context only.
This file is written as results land, not after.

## 0 — THE TWO STANDARDS, AND WHY THE COUNTS DIFFER

| | it.15 checker (`tests/jupiter/test_v20_r15_it14_theory_table.py::bad_citations`) | Inspector's standard (`V20_R15_IT16_INSPECTOR.md`) |
|---|---|---|
| asks | does the FILE exist and is `lineno` within its length? | does the CITED LINE carry the declaration/statement/value the table claims? |
| planted negative | a citation naming a file that does not exist | *(absent — this is the hole)* |
| result on the frozen table | **123/123 pass, 0 bad** | **8 of 20 spot-checked do not land** |

Both are true. `contract:107` is off by 16 onto a different question and passes the
weaker check trivially: the file exists and has more than 107 lines.

**The it.15 result is not withdrawn — it is re-scoped.** It certifies *resolvability*,
which it does correctly after the planted negative caught its `"/" in path` bug. It never
certified *landing*.

## 1 — THE PRECEDENT THIS CENSUS MUST NOT REPEAT

`tests/jupiter/test_v20_r15_it12_constants.py` names its authority in its own docstring —
constants *"published in prose before this file existed, at the `path:line` named beside
it"* — and three of those pointers are stale (`:12`, `:185`, `:223`). **No assertion in
that file touches a line number.** Every check is on a value or on source text. It is
green (`6 passed in 9.42s`) and its citations are wrong at once.

> The citation layer is unasserted in a file whose entire purpose is to end unasserted claims.

**Consequence for this iteration's instrument:** its planted negative fires on a citation
pointing at the WRONG LINE inside a file that exists, not merely at a missing file.

## 2 — CENSUS RESULT: 123 CENSUSED, **28 DO NOT LAND**, ALL 28 REPAIRED

**123 citations across 37 files**, extracted mechanically from the frozen table and split
into four dossiers, each record carrying the citation, the table's own claim context, and
the cited source lines with the cited line marked. The extraction count matches it.15
exactly — **the disagreement was never about how many citations there are.**

| chunk | citations | LANDS | **DOES NOT LAND** |
|---|---|---|---|
| 1 | C1–C31 | 28 | **3** |
| 2 | C32–C62 | 25 | **6** |
| 3 | C63–C93 | 21 | **10** |
| 4 | C94–C123 | 21 | **9** |
| **TOTAL** | **123** | **95** | **28 — 22.8%** |

**The Inspector's `8 of 20` was low, not high.** His rate projected to `49`; the census
finds `28`. Both readings agree on the only thing that mattered: **`0 bad` was an artefact
of the standard, not a property of the table.**

### 2.1 THE 28, WITH TRUE LOCATIONS

| # | cited as | claim | true location |
|---|---|---|---|
| C1 | `WING_MANIFEST.md:12` | *"the wing list froze at it.4"* | **`:3`** — `:12` is `FREEZE-SHA256 = fbf17e07…` |
| C10 | `LEAP_LEDGER.md:25` | the ***struck*** sense of `F4` | digits right, **label wrong**: `:25` is `L-4`, sense **withdrawn**; *struck* is at `:24` |
| C26, C74 | `IT13_MERCURY.md:141` ×2 | *"206–537 GPU-s … one `argparse` line plus two `synchronize()`"* | **`:146`** — `:141` is the heading above the table |
| C44 | `ceq/hankel.py:1` | `hankel_block`/`rank_real`/`rank_plus_lower`/`NEG_ENTRY` | **`:108`/`:131`/`:279`/`:75`** |
| C47 | `it12_constants.py:22` | `err_1 = 0.9746794345` @ `5e-11` | **`:12`** |
| C50 | `IT11_INSPECTOR.md:503` | *"theorem sound, the `n=4` test is two identities"* | **`:502`** — `:503` is the next row, about constants |
| C52 | `it12_constants.py:21` | excess `1.0845223424` @ `5e-10` | **`:11`** |
| C59 | `it12_constants.py:23` | Spearman `+0.717647` / `−0.032353` | **`:13`–`:14`** |
| C61 | `it7_q3.py:1` | node `::test_w3_lambda_hat_sign_separates…` | **`:106`** |
| C63 | `IT8_JUPITER.md:104` | the `O(n·S²)` row | **`:105`** — `:104` is `\|---\|---\|---\|` |
| C64 | `arm_smprime.py:163` | `path_product` | **`:144`** |
| C75 | `arm_pl.py:316` | `brute_force_path_sums` is `O(2^S)` | **`:304`** |
| C82 | `it8_q4_q5.py:1` | node `::test_q5_floor_1_is_a_ONE_HOP_THRESHOLD…` | **`:127`** |
| C83 | `it12_constants.py:83` | band `min=0.144954 max=0.219610 (n=12)` | **`:122`** |
| C85 | `it8_q4_q5.py:1` | node `::test_q5_no_arm_crosses…` | **`:151`** |
| **C87** | **`CEQ_V20_R15_CONTRACT.md:107`** | *"W1 to the oracle where a state distribution exists"* | **`:123` — OFF BY 16, onto Q6's heading line.** The Inspector's named example, confirmed and located |
| C88 | `arm_smprime.py:572` | `returns readout(h).squeeze(-1)[:, seq-1]` | **`:577`** — `:572` is the `def forward` |
| C89 | `negation_scope.py:300` | `equilibrium_oracle` returns shape `[n]` | **`:304`** (`return z`) |
| C98 | `IT13_MERCURY.md:196` | *"STRUCK and KILLED — measured `1.4060346618513293`"* | **`IT89_INSPECTOR.md:63` — A DIFFERENT FILE**, not a line slip |
| C100 | `CONTRACT.md:124` | *"CRITERION (lexicographic): (1) BED-M crossing CP-lower > 0.5"* | **`:125`** |
| C103 | `IT13_MERCURY.md:57` | *"no cell of BED-K's shape has ever been run"* | **`:63`** |
| C104 | `MISTAKES.md:1` | *"threshold selected after seeing the data it judges"* | **`:451`** (`### M-2`) — `:1` is the file's `#` heading |
| C107 | `IT567_INSPECTOR.md:460` | *"naming a field … calling it a field is not"* | **`:482`** |
| C108 | `it12_saturn.py:296` | `inadmissible_leapable_rows()` | **`:319`** |
| C109 | `it12_saturn.py:306` | `head.startswith("none")` | **`:329`** |
| C112 | `it12_saturn.py:320` | `KNOWN_INADMISSIBLE` frozen literal | **`:343`** |
| C113 | `it12_saturn.py:330` | the FIELD planted negative | **`:353`** |

**Plus one repair this office ruled on its own, not flagged by census:** `C95`
(`negation_scope.py:300`, *"on `equilibrium_oracle`"*) resolves to a body line. Under the
same named-symbol rule applied to C44/C61/C64/C75 it must point at the declaration,
**`:286`**. **29 citation strings changed; 28 were census failures.**

### 2.2 THE FIVE MECHANISMS — NOT 28 TYPOS

1. **TWO RIGID DISPLACEMENTS, EACH ONE ERROR APPLIED TO A WHOLE FILE (7 of 28).**
   Into `test_v20_r15_it12_constants.py`: `21→11`, `22→12`, `23→13` — **uniform `+10`.**
   Into `tests/saturn/test_v20_r15_it12_saturn.py`: `296→319`, `306→329`, `320→343`,
   `330→353` — **uniform `−23`.** Neither is N typos; each is one displacement, and
   **no resolvability check can ever see a rigid shift, because every shifted line still
   resolves.** Both target files are the round's own instrument files.
2. **OFF-BY-A-ROW ONTO AN ADJACENT ROW OF THE SAME TABLE (C50, C63, C100, C10).** The
   reader arrives somewhere plausible, in the same register, about a different object, and
   never learns it is the wrong row. **This is the `contract:107` species** — C87 is the
   same mechanism at amplitude 16 instead of 1.
3. **`:1` USED AS "THE MODULE", THEN REUSED FOR NAMED SYMBOLS (C44, C61, C82, C85, C104).**
   `ceq/arm_pl.py:1` is legitimate where the table means *the arm itself* (C51, C53 both
   land). Reusing it when the sentence names four functions, a test node, or a taxonomy
   entry is not. **The convention is sound; extending it past its meaning is the defect.**
4. **DEF-LINE vs BODY-LINE CONFUSION, IN BOTH DIRECTIONS (C64, C75, C88, C89, C95).**
   `:572` is `def forward` where the claim is about the `return`; `:163` is a body line
   where the claim names `path_product`. **Cited symbol → its `def`; cited behaviour → the
   statement.** Applied uniformly here for the first time.
5. **ONE WRONG FILE ENTIRELY (C98).** `IT13_MERCURY.md:196` for a finding that is in
   `IT89_INSPECTOR.md:63`. Not a line slip — **a different office's report.**

**None of the five is detectable by asking whether the file exists.** That is the entire
distance between `123/123, 0 bad` and this census.

### 2.3 THE INSTRUMENT, AND ITS CALIBRATION

`tests/jupiter/test_v20_r15_it17_citation_landing.py`. `MANIFEST` holds the 29 repaired
citations, **each with a verbatim substring of the line it must land on**, hand-verified —
deliberately **not** generated from the table, because a manifest derived from the thing it
checks asserts nothing.

**The calibration is the point.** The it.14 planted negative plants a citation to a
**missing file** — the weaker failure, and the one that was already caught. This file's is
`test_the_checker_fires_on_a_WRONG_LINE_inside_a_file_that_exists`: it shifts every
citation by the **`+10` the census actually measured**, so every shifted citation still
resolves and the it.14 checker stays green on all of them, **and asserts this checker names
every single one.** Without it, green would mean only that the parser found nothing —
which is exactly how `test_v20_r15_it12_constants.py` is green with three stale pointers.

**`[RUN] python -m pytest tests/jupiter/test_v20_r15_it17_citation_landing.py -q`**
against the **unmutated table, before any repair**:
`3 failed, 3 passed in 0.54s` — RED on `test_no_withdrawn_pointer_survives_in_the_table`,
`test_every_repaired_pointer_is_present_in_the_table`,
`test_the_F4_overload_labels_name_the_sense_actually_at_the_line`; GREEN on the three that
prove the *targets* were right before the table was touched.

**`[RUN] python -m pytest tests/jupiter/ tests/saturn/test_v20_r15_it14_saturn.py -q`**
after all 29 repairs: **`1 failed, 197 passed in 56.56s`**. The single failure is
`test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record`
— **a standing RED that predates this iteration** and asserts a fact about trained `arm_pl`
cells (`0 of 8 have a_hat_max <= 1.0`), touching no citation. **Control: it was red before
these edits and is red for the same reason after.**

## 3 — THE FOUR CELL DEFECTS

### 3.1 Q5 — grade and disclaimer of that grade in one cell. **THE GRADE STANDS. THE GRADE LINE'S SILENCE ABOUT ITS OBJECT IS THE DEFECT.**

Both Q5 cells read `GRADE: F1 + const` and then open GAP with a strike on the object:
**"THE FLOOR GRADED AGAINST IS NOT AN INFORMATION FLOOR"** (Q5/W1), and *"the same
capability threshold, with the same defect"* (Q5/W3). A one-call reader sees a bound and a
denial of the bound in one cell and cannot tell which survives.

> **RULING J-17a. `F1 + const` stands, scoped to its object on the GRADE line itself.**

*Why not `F4`.* `F4` in this round's own rubric is *"the theory's domain is empty for the
graded wings"*, and §0.2's measured finding — that zero of four candidate floors bind
either wing **as information floors** — reads as domain-emptiness. It is rejected anyway:
`floor₁` **is measured**, on 40 of 40 banked cells, with a constant band
`min=0.144954 max=0.219610 (n=12)` bound at `tests/jupiter/test_v20_r15_it12_constants.py:83`.
Re-grading a measured cell to `F4` asserts emptiness where a number exists. §0's own
counterexample 1 already records that `F4` is **overloaded three ways**; answering an
ambiguity by spending an overloaded token is not a resolution.

*Why the disclaimer is not a competing grade.* `F1` grades **how well a claim is pinned**.
`floor₁` is pinned, to a constant. The disclaimer says the pinned object is **not what Q5
asks about**. Those are statements about two different things, and the cell reads as a
contradiction only because the GRADE line never names its object. **The repair is on the
GRADE line, not on the token** — and the ledger already reads it exactly this way: `L-11`
is filed **`F1 + const on both wings, but on the wrong object`**
(`V20_R15_LEAP_LEDGER.md:70`). The cell is brought into line with its own ledger row, which
was right before the cell was.

### 3.2 Q2/W1 — does the `F4` rest on a forbidden estimate? **NO. GRADE AND JUSTIFICATION BOTH HOLD.**

MERCURY's standing rule is *no estimate without its measured basis*, and its bite here is
`V20_R15_IT13_MERCURY.md:63`, verified verbatim this iteration:

```
**(ii) No cell of BED-K's shape has ever been run.** The bed generator is registered
```

> **RULING J-17b. The `F4` rests on bed membership, and bed membership is not an estimate.**

The grade's basis is `V20_R15_JOURNAL.md:1703-1704`, verified verbatim: *"M16 Hankel binds
BED-K(a) only and **both frozen wings are BED-M**."* An empty domain intersection is
established by comparing two **registered** domains. It consumes **no BED-K measurement**,
so the standing rule cannot bite it — there is no estimate present to lack a basis.

Where the rule *does* bite is **pricing the repair**, and the cell already carries no
price: `ROUTE: none on this wing`. The admission-condition table likewise names the
instrument change with no number, and its pointer checks out —
`ceq/kdata.py:475` is `"bed_k": {"generator": "ceq.beds.bed_k.build_delay",`.

**But the failure mode this defect was hunting is real, and it lives one column over.** The
admission table's third column asserts `Q2/W1` would re-enter at **`F1 + const` (the `1/d`
law is already true there)** — a **grade prediction on a shape that has never been run**,
which is the same species MERCURY forbids, in a column no office has audited. It is not
load-bearing for the `F4`, so the `F4` survives; the prediction is marked in place.

### 3.3 Q2/W1 — the `F4` with no ledger row. **IT NEEDS A ROW. J-14 SAYS WHAT THE ROW CONTAINS; IT DOES NOT EXCUSE THE ROW.**

The enumeration settles it. The ledger's fifteen rows
(`V20_R15_LEAP_LEDGER.md:22-28`, `:68-71`, `:130-131`, `:160`) cover, by cell:

| question | W1 row | W3 row |
|---|---|---|
| Q1 | `L-2` | `L-1` |
| **Q2** | **none** | **none** |
| Q3 | `L-5` | `L-6` |
| Q4 | `L-9`, `L-10` | `L-9` |
| Q5 | `L-11` | `L-11`, `L-12` |
| Q6 | `L-13` (**F4**) | `L-14` (**F4**) |

> **RULING J-17c. The other two `F4` cells both have rows, so the ruling cannot be what
> excuses the missing one.**

`Q6/W1` is `L-13` and `Q6/W3` is `L-14`, and J-14 was **applied to** `L-13` rather than used
to delete it: `V20_R15_THEORY_TABLE.md:368` restamps it **NOT-PUT (F4), FIELD column
withdrawn**. J-14's own worked application therefore proves that an `F4` cell **keeps its
row and gets restamped**. `Q2/W1` sits on identical footing and is absent for no stated
reason.

**And the hole is larger than the defect as put.** `Q2` is the **only question with zero
ledger rows on either wing**. `Q2/W3` is graded `F1 + const`, bears on clauses (1) and (3)
by its own ARENA field, and is likewise absent. SATURN flagged `Q2/W1` alone — the
Inspector records it at `V20_R15_IT16_INSPECTOR.md:41`: *"SATURN flagged `Q2/W1` alone,
because it is the one with no ledger row."* **The correct count is two missing rows, not
one.** The leap reads the ledger; on Q2 it currently reads nothing at all.

### 3.4 M14 — three grades in three files. **M14 IS `F4`. THE `F1` IS DATED it.3 AND DIED AT it.4.**

The three readings, each verified verbatim this iteration:

| file | grade | what kind of statement it is |
|---|---|---|
| `CEQ_V20_R15_CONTRACT.md:239-243` | `F3` | **pre-registration**, self-conditioned: *"grade F3 pending an exact sweep-cut conductance; first task of the annex, and it stays F3 in the table until it passes"* |
| `V20_R15_JOURNAL.md:645` | `F1` | an **it.3 SCOREBOARD line**: *"Annex: M14 at **F1, HOW-BAD 108x**, death re-attributed to V-25."* |
| `V20_R15_LEAP_LEDGER.md:24` | `F4` | the **live verdict**: `L-3`, *"struck by V-25 … machine-true and domain-empty"*, `TERMINAL` |

> **RULING J-17d. These are not three verdicts on one object. They are one
> pre-registration, one superseded scoreboard reading, and one verdict — and only the last
> is live. `M14 = F4, TERMINAL, domain-empty.`**

The decisive fact sits three lines below the `F1`: `V20_R15_JOURNAL.md:648` is
`## it.4 — THE FREEZE. FROZEN-N = 2.` **The `F1` is inside the it.3 scoreboard block, on
the far side of the freeze** — in a paragraph whose own first clause reads *"Wing list
**not frozen**"*, a state the very next heading ends. It is a superseded reading, not a
competing grade. The contract's `F3` is a starting position the contract itself makes
conditional, and pre-registration is not retro-editable.

**Why it survived two Inspector approvals unrepaired: it is scoreless.** The `F1` reading
feeds `CEQ_V20_R15_CONTRACT.md:267` — *"annex M1–M16 with ≥12 at F0/F1 and M14 resolved
+4"* — which is a **conjunction**, and `V20_R15_JOURNAL.md:1763` already forfeits the other
conjunct: *"`≥12 of M1–M16 at F0/F1` is out of reach and the `+4` with it."* **M14's grade
cannot move the score in either direction.** That is precisely why naming it without
repairing it cost nothing, and precisely why it must still be repaired: **the leap does not
read the scoreboard, it reads the grade.** Under `:267`'s second conjunct M14 is now
*resolved*; the `+4` stays forfeit on the first.

### 3.5 — THE CONSTRAINT EVERY REPAIR IN THIS ITERATION OBEYS

Other offices cite **into** this table by line. `V20_R15_IT16_INSPECTOR.md:41` cites
`V20_R15_THEORY_TABLE.md:118,123`, and `:285-287` and `:289` in the lines below it.
**Inserting or deleting one line in the table breaks all of them and manufactures exactly
the defect this iteration exists to remove.**

> **RULING J-17e. Every citation repair is made IN PLACE — digits changed inside the line
> that already holds them. `V20_R15_THEORY_TABLE.md` keeps its 443-line length exactly.**
> The content digest moves, which is correct and intended. **No external pointer moves.**

## 4 — THE DIGEST. **IT DID NOT MOVE, AND THAT IS THE FINDING.**

The brief expected `THEORY-SHA256 = 9989f0ef…` to move once citations were fixed.

> **`[RUN]`** *(SATURN's own `theory_cells` / `theory_digest` / `row_digest`, imported, not
> reimplemented)*
> ```
> SATURN THEORY-SHA256 declared = 9989f0efabe1895a9772f650ba1c3bcb77db7d37550879f2fdc9e8e9f2c63057
> SATURN THEORY-SHA256 at HEAD  = 9989f0efabe1895a9772f650ba1c3bcb77db7d37550879f2fdc9e8e9f2c63057
> MOVED CELLS: NONE
> ```
> **`[RUN] python -m pytest tests/saturn/test_v20_r15_it14_saturn.py -q` → `17 passed`.**

**Twenty-nine citation repairs inside the cells, and the freeze digest is bit-identical.**
`theory_cells()` reads **only the §1 at-a-glance grade matrix** — its own docstring closes
*"Rows only; see the report's LIMITS."* So `THEORY-SHA256` is a digest over **twelve grade
tokens**. It is blind to all 123 citations and to all six fields of all twelve cell bodies.

**And the per-cell keys are not injective.** `distinct SATURN per-cell digests: 7 of 12`.
`1eb8872d017f62d7` is **four** cells (Q2/W3, Q3/W3, Q5/W1, Q5/W3), `63f81989362491d3` is
Q4/W1 **and** Q4/W3, `f53365e65ce4d925` is Q6/W1 **and** Q6/W3 — because `row_digest`
hashes the row **value alone**, not `cell|row`. The claim that *"per-cell keyed digests
identify exactly which cells changed"* **fails on eight of the twelve cells**, and
`::test_the_theory_digest_is_joint_over_cell_and_row` passes only because the pair it
swaps happens not to collide. **Swap Q4/W1 with Q4/W3 and nothing moves at all.**

**THE DIGEST THAT DOES COVER WHAT THE LEAP READS.** Same construction, over the twelve
`### CELL` bodies rather than the twelve grade tokens:

```
THEORY-CELLS-SHA256 = 6e3473503b8a8f08d412574a24a2df31d01d6c15545bd175b42e6d0286982189
distinct cell-body digests: 12 of 12
  Q1/W1 d0f8f648178d9285   Q1/W3 a7e7ea78c684b3f7
  Q2/W1 46539c42aceeb71f   Q2/W3 9302fcdd72742323
  Q3/W1 d084048e7f92ccea   Q3/W3 688f34b24c936c80
  Q4/W1 dc55d186d28fff4d   Q4/W3 b0e2a7fa3a7f1378
  Q5/W1 e34895c5b8f70979   Q5/W3 8db21ffe38e4ffdb
  Q6/W1 51624dd29132e6dc   Q6/W3 085ee797a4017629
```

**Twelve distinct of twelve**, and it moved on the repairs — mid-iteration, after the first
nine, it read `05b1a773e249f921…`; after all 29 it read `c79be822…`; after RULING J-17a and J-17d were applied to the Q5 GRADE lines and the M14 counterexample it reads `6e347350…`. **This is not filed as a
replacement for SATURN's digest, which is his row and his to move.** It is filed as the
measurement that shows what his does not cover.

**One further fact, found while trying to recover the pre-repair baseline:**

> **`[RUN] git show HEAD:V20_R15_THEORY_TABLE.md`** →
> `fatal: path 'V20_R15_THEORY_TABLE.md' exists on disk, but not in 'HEAD'`

**The leap's primary input is not in git.** The freeze is a digest, written in one office's
markdown, over an **untracked** file — recoverable only from the digest that, as measured
above, does not cover the file's contents. No git write was made to change this; it is
recorded as found.

## 5 — WHAT THIS ITERATION DID NOT DO

- **It did not re-grade any cell.** J-17a keeps Q5 at `F1 + const` and says why; the census
  `1×F0 / 5×F1 / 1×F2 / 2×F3 / 3×F4` is unchanged and `DECLARED_CENSUS` untouched.
- **It did not add the two missing Q2 ledger rows** (J-17c). The ledger is another office's
  file and its rows are digest-bound at it.12; the finding is filed, the row is not written.
- **It did not repair the three stale docstring pointers inside
  `tests/jupiter/test_v20_r15_it12_constants.py`** (`:12`, `:185`, `:223`). The census was
  of the table's citations; that file's own citation layer is still unasserted, and it is
  the file both rigid displacements point into.
- **It did not audit the admission table's `grade it would re-enter at` column**, which
  contains at least one grade prediction on a never-run bed (§3.2).
- **No git write. Nothing touched Kaggle.**
