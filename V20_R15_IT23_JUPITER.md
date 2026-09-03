# V20 R15 it.23 — JUPITER (MYCROFT, annex)

Instrument: `tests/jupiter/test_v20_r15_it23_fence_and_argument.py` (9 tests).
Repairs applied to `tests/jupiter/test_v20_r15_it21_heading_anchor.py` and
`tests/jupiter/test_v20_r15_it20_citation_freeze.py`. No git writes. Nothing
touched Kaggle.

Timer, dated: **11:31:46Z** — 0m43s elapsed of 20m, armed 11:31:03Z, labelled
"iteration 30". Second reading **11:36:09Z**. The re-arm the brief warns about is
visible in the label; the arm timestamp is the only stable field, so it is quoted
with every number below.

---

## 1. REPAIR 1 — THE FENCE. MARS IS RIGHT IN BOTH DIRECTIONS.

**RED first, verbatim, against unmutated code** —
`[RUN] python -m pytest tests/mars_v20/test_it22_the_repairs_of_it21.py -x -q -k fence`
(11:32:56Z, before any edit):

```
E       AssertionError: FAIL-OPEN. On the real file the resolver REFUSES: REFUSED: 'F4' appears on 4 lines in section '## it.7' of V20_R15_LEAP_LEDGER.md, not 1
E            After a 4-line ```bash fence carrying one `# comment` is inserted inside the section, it returns line 24 with full confidence.
E            The `# comment` is read as a level-1 heading, the section ends there, and 3 of the 4 occurrences are scoped out of existence. Uniqueness was not established; it was manufactured by markdown the instrument cannot parse.
E       assert (False)
E        +  where False = isinstance(24, str)
1 failed, 2 deselected in 0.51s
```

That is this office's own third refusal — the one J-21a calls load-bearing —
returned as line 24 with no hedge.

### RULING J-23a — a fence is not a heading. ADOPTED.

`_levels(src)` forces every line inside a fence to level 0; `resolve` indexes it
at **both** former `_level` call sites. Five lines, one pass, no dependency.

### RULING J-23b — the exact-key half of the route is REJECTED, with a number.

MARS asked for `l.strip() == key` in place of `key in l`. **[RUN] that breaks all
nine keyed re-anchors, not the corner case he priced.** Every key in
`HEADING_CENSUS` is a *prefix* of its heading line — `## it.7` against
`## it.7 — opened, seven rows`; `## it.9 — JUPITER` against
`## it.9 — JUPITER — the Q6 pair, plus the cell index the ledger owed`. Under the
exact form each resolves to ZERO headings and REFUSES, taking coverage from
`129 of 129` to `116 of 129` in one edit — the outcome J-20b existed to avoid.

The risk he names cannot occur: a future `## it.7 bis` makes `len(heads) == 2`,
which is `REFUSED: section key '## it.7' matches 2 headings ... not 1`. **The
substring form fails closed and loud already**, and
`test_a_second_heading_extending_the_key_refuses_rather_than_double_matching`
pins that. His *"three of ten re-anchors had to lengthen"* is real and is about
the `want`, not the key.

### RULING J-23c — one landing predicate. ADOPTED.

it.20 now exports `lands(want, line)`; `not_landing` and
`not_landing_by_heading` both call it. The INSPECTOR's it.20 flag is closed by
one import, as MARS said.

**Coverage after repair: `129 of 129`** — 116 by line, 13 by heading, 0 unscored,
0 failing. `[RUN] python -m pytest tests/jupiter/test_v20_r15_it23_fence_and_argument.py
tests/jupiter/test_v20_r15_it21_heading_anchor.py tests/jupiter/test_v20_r15_it20_citation_freeze.py -q`
→ **26 passed** (11:36Z, after the count correction). Both MARS fence tests turn
GREEN under the repair with his file unmodified.

**Latency, carried forward as MARS stated it [CITED: MARS, it.22]:** 0 ghost
headings in either anchored file today. The defect was latent, not live.

---

## 2. THE THREE ARGUMENT-LAYER DEFECTS — RULINGS

MERCURY's `9 of 12`, seed 2209, disjoint from 1520, denominator published
[CITED: MERCURY, it.22].

### J-23d — `THREE_LINES_LOW`: real defect, **and its proposed repair is also wrong**.

MERCURY is right that `CEQ_V20_R15_CONTRACT.md:239` carries `[V]` and no `F3`.
He places `F3` at `:242`. **[RUN] `:242` reads `an exact sweep-cut conductance;
first task of the annex, and` and carries no `F3` either. `F3` is at `:241`,
`:243` and `:244`.** The corrected pointer is off by one in the same direction as
the defect it corrects.

Consequence, and it changes the disposition: **no line in that item carries `F3`
uniquely**, so this is not a pointer edit. A pointer that lands uniquely needs a
different `want`, and under J-20a a changed `want` is a **withdrawal plus a new
citation**. `C13` (`CEQ_V20_R15_CONTRACT.md:239`, want `M14 CHEEGER
STRATIFICATION`) is **REFUSED-PENDING-REISSUE**, not silently moved. Repairable —
by re-issue, next iteration.

### J-23e — `COMPOUND_HALF`: repairable, and the general rule is stated.

`+0.717647` is at `tests/jupiter/test_v20_r15_it12_constants.py:13`; `−0.032353`
is at `:14`. **[RUN] confirmed.** **A compound claim may NOT cite one line.** A
cell asserting N propositions carries N pointers or one range; the pointer here
becomes `:13-14`. The word doing the damage is *"both"*, and MERCURY named it.

### J-23f — `LINE_ONE_IDIOM`: NOT a wrong pointer. The round gets the notation.

**[RUN] 8 occurrences across 6 files** in `V20_R15_THEORY_TABLE.md`, independently
recounted here — `ceq/arm_pl.py:1` ×2, `tests/.../it11_q6_oracle.py:1` ×2,
`lean/lakefile.lean:1`, `tests/.../it14_theory_table.py:1`,
`tests/.../it6_q1_exact_class.py:1`, `tests/.../it9_q6.py:1`. Matches MERCURY's
count exactly.

**Notation, granted:** `path:*` means THE FILE and is not line-landed. `path:1`
continues to mean line 1 and IS line-landed. The idiom is retired at the next
table edit; the count is pinned in the test so the population cannot drift.

### The stricter reading — `8 of 12` vs `9 of 12`

**The round adopts MERCURY's loose reading, `9 of 12`, and records the strict one
as a named limit.** A citation into a module docstring points at documentation of
an assertion rather than the assertion; that is a fourth mechanism and it is real,
but scoring it would retire `:12` and every prose constants table with it, and the
round has no instrument that can tell a docstring from a comment from code. The
strict reading is adopted the iteration an instrument can measure it, not before.

---

## 3. RULING J-23g — `LIVE` IS A HAND-MAINTAINED LIST. SAID PLAINLY.

MERCURY could not measure it by commit count and was right not to fake it.
**[RUN] `V20_R15_LEAP_LEDGER.md`, `V20_R15_JOURNAL.md`, `CEQ_V20_R15_CONTRACT.md`
and `V20_R15_THEORY_TABLE.md` are all untracked** — four files, zero commits each.
The criterion does not separate the refused from the scored, because two of the
four are scored and all four have the same commit count.

**The measurable criterion that does not need git:** a file is LIVE if its line
count moved between two reads one iteration apart. **Applied now it contradicts
the hand-list.** `[RUN] grep -c "" `, 11:33Z:

| file | it.20 census | now | moved |
|---|---|---|---|
| `V20_R15_JOURNAL.md` | 3908 | **4829** | +921 |
| `V20_R15_LEAP_LEDGER.md` | 438 | **438** | **0, three iterations later** |

**The ledger is on the LIVE list because an office typed it there.** It has not
moved since the census that refused it for moving.

**The list is not retired.** A static file may be appended tomorrow, and a
criterion that flips a file's scoring status every iteration is worse than a
judgement with a named owner. What is retired is the *claim that it is measured*.
**What makes a file join: this office puts it there, on the ground that an office
is actively writing it this round.** That is a judgement, it is JUPITER's, and
`test_LIVE_is_a_hand_list_and_one_of_its_three_members_has_not_moved` goes RED on
an unannounced edit to the list or on the ledger starting to grow again — which is
the difference between this and the hardcoded maps the round struck twice.

---

## Limits

The three argument-layer repairs are RULED and MEASURED here; none is APPLIED to
`V20_R15_THEORY_TABLE.md`, because each edit moves an occurrence out of the it.20
`CENSUS` and would drop `129 of 129` until the census is re-taken in the same
iteration — not reachable inside this wall clock. `tests/jupiter/` has 3 failures
at 11:36Z (`it18_citation_landing` ×2, `it4_merge_is_unexercised` ×1); neither
module imports it.20 or it.21 and both were failing before this iteration's edits
— not this office's, not verified as anyone's. MARS's STRIKE 1 (SATURN's
`WING_ARM` pin) and STRIKE 3 (the stale `31` in the INDEX-SHA256 sentence) are
untouched. The it.23 instrument is not sealed; a `HEADING_SEAL`-style freeze over
its own asserted counts was not reached.
