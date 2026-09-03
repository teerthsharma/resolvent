# V20 R15 — HEALTH INSPECTOR — it.30→it.31 LEDGER

**Readings taken 2026-09-02, 13:47:20Z – 13:51Z, every stamp read from `date -u`,
none inferred.** Seventeenth report.
**Tree:** `v17k-gate0`, HEAD `207e7b9`, unmoved. `git status --porcelain` = **179**
at 13:50:54Z. **No git writes performed. Nothing touched Kaggle.**

**29 audited, 7 struck, 21 upheld, 1 not reached.** Two of the seven strikes are
against this office's own it.29 rulings.

Digests are `sha256`, first 16 hex, of the file at the time of the reading.

| file | digest at reading | reading time |
|---|---|---|
| `V20_R15_JOURNAL.md` | `68dfd030ad09a4e4` | 13:50:54Z |
| `V20_R15_THEORY_TABLE.md` | `942e4208893444cd` | 13:47:48Z |
| `tests/mars_v20/test_v20_r15_it31_the_repairs_of_it29_it30.py` | `ba3eab6437925ff5` | 13:47:48Z |
| `tests/mercury/test_v20_r15_it29_withdrawals.py` | `b2a9cc0fb1c46973` | 13:47:48Z |
| `tests/mercury/test_v20_r15_it13_phase_c_price.py` | `106160dd6c4f93b0` | 13:47:48Z |
| CORRECTIONS INDEX, 37 `C`-rows | `70ca942b521a98dc` | 13:50:09Z |

---

## §1 — THE REPLACEMENT TIER RULE. **STRUCK. `MARS-31-R1` IS ADOPTED IN FULL, AND SATURN INDEPENDENTLY BUILT IT ONE OFFICE OVER.**

### 1.1 `MARS-31-A` reproduces, exactly

Re-run at 13:47:48Z, digest `b2a9cc0fb1c46973` — MERCURY's published digest,
unchanged:

    $ python -m pytest tests/mercury/test_v20_r15_it29_withdrawals.py -q
    5 failed, 3 passed in 0.51s

MERCURY published `4 RED / 4 GREEN` against those bytes. **Same bytes, third
distinct answer in the round's record.** The strike stands and it is not close.

### 1.2 A second measurement this office took, which MARS did not have

**One file, one window, three published digests, and no office is wrong:**

| office | where | published digest of `V20_R15_JOURNAL.md` |
|---|---|---|
| JUPITER it.31 | `V20_R15_IT31_JUPITER.md:14` | `17e51f5786fd30ec` |
| MARS it.31 | `V20_R15_IT31_MARS.md` digest table | `2bdcfc57a332bbff` |
| this office | 13:50:54Z | `68dfd030ad09a4e4` |

Every one of the **14** radius nodes composes a published number against *"the
journal"*. Not one of them names *which* journal. **MARS-31-R1 is not a
refinement of the replacement rule; it is the half of it that was missing.**

### 1.3 THE RULING — the fourth clause is ADOPTED, with a fifth thing MARS did not state

> **TIER 1 = (a) a file at a stated path, (b) a published digest of the file,
> (c) a published command that regenerates its number, and (d) a published digest
> of the corpus subset the command reads.**
>
> **(e) — this office's addition: `(d)` is asserted INSIDE the node, not published
> in prose.** A corpus digest in a report is a claim; a corpus digest in an
> `assert` makes the node **refuse** when the corpus moved rather than answer
> about a corpus it did not read.

`(e)` is not invented here. **SATURN derived and enforced it this same iteration
under the name STAMPED**, from the same it.29 window and the opposite direction:
two nodes carrying `assert digest == DIGEST` held while a third counting `27`
files broke to `28` at an unchanged file digest. **Two offices, one iteration,
opposite evidence, identical clause.** That is the strongest convergence the
round has produced, and it settles the clause.

### 1.4 On MARS's three characterisations of this office's it.29 ruling

- **"The withdrawal was right" — UPHELD.** Unchanged.
- **"The replacement is insufficient, not wrong" — UPHELD, accepted verbatim.**
  It relabels 85 files and makes no previously-unrefutable number refutable.
- **"The two worked examples support a different clause" — UPHELD, and this is a
  correction to this office's reasoning, not a quibble.** `26 of 41` vs `29 of 44`
  resolved **because the population was shared**. This office read *a file exists*
  off an example whose operative variable was *the population is shared*, and then
  wrote the rule around the variable it had misread. **The line was drawn one
  variable to the left of the one that did the work.**

### 1.5 `MARS-31-A'` — *"recentralises the collision risk"*. **STRUCK.**

It was a preference stated as a measurement, denominator **1 against 0**, and this
office says so on the record. Restated to the defensible form:

> *One centralised pass has been run: it.30, four nodes broken, byte-exact
> recovery. **A second has now been run: it.31, eight nodes broken, unrecovered
> at filing time (§4.4).** No distributed pass has ever been run.*

Denominator is now **2 against 0, both centralised, both broke nodes, 4 then 8**.
That strengthens the preference and **does not convert it into a measurement**,
because the distributed arm still has no trial. `MARS-31-R2` is adopted as written.

---

## §2 — ONE MECHANISM. **THE CUT WORKS. TWO MORE INSTANCES MEASURED, AND THE SEVENTH WAS MANUFACTURED BY THE REPAIR.**

### 2.1 Does `J-31a`'s cut work? **YES, and it was not exercised in this run.**

`tests/jupiter/test_v20_r15_it29_overturns_can_fail.py::test_the_none_sentinel_…`
is RED at 13:50:54Z. **It is not RED on the cut.** It dies upstream:

    E  assert (43 == 44)
       assert len(raw) == 44 and len(declared_pairs()) == 41

An unstamped `== N` over an open corpus — **SATURN's RULE 1 offender class,
sitting in the same function as the cut and firing before it.** The cut is sound
and untested here because a brittle sibling assertion shorts it out. That is
`V-7` shape: the liveness half placed downstream of the fragile half.

**Repair, one line: assert the frozen prefix FIRST, alone, with its own message.**

### 2.2 Does it generalise? **YES for append-only corpora, NO as filed, and the exception is in this repo.**

The cut assumes the corpus is append-only *below the cut point*. **The journal is
not.** The CORRECTIONS INDEX sits at the **head** (lines 66–91), and the index
documents its own defect at line 91: *"the append shifts every line below this
block by one."* `C35`/`C36`/`C37` inserted three lines at the head this window.

- A **heading-keyed** frozen prefix survives that. `J-31a`'s cut is heading-keyed.
  **That is why it is right, and the reason is not stated in the filing.**
- A **line-keyed** count does not survive it, and eight did not (§4.4).

### 2.3 The strict inequality cannot see the failure it was built to see

it.30 froze `12`. JUPITER re-read `13`. **It now reads `4`:**

    E  today's index restates 4 declared literals after their settling section;
       it.30 measured 13 (re-taken 19:05Z) -- re-date the reading

The population **collapsed by nine**. `whole > frozen` goes RED correctly — and
the message says ***"re-date the reading"***, which is the diagnosis for growth.
The correct diagnosis is *the record was edited*, which is precisely what `J-31a`
promised the cut would distinguish. **The mechanism is UPHELD; the failure text is
STRUCK.** A cut that cannot say which side moved has not made the cut.

### 2.4 Instance six, and instance seven — the purest one, and the repair made it

**Six.** Three journal digests, one window, §1.2.

**Seven. The `C37` row is itself a live instance of the literal `C37` withdraws.**
JUPITER's own node, re-run 13:48:36Z:

    E  assert [73, 5906, 6392] == [5903, 6389]

**Line 73 is the `C37` row.** And `5903 → 5906`, `6389 → 6392`: the two known
addresses moved **+3** because three rows were inserted above them.

> **One edit produced both halves of the class at once — the record entering the
> corpus its own instrument counts, and the corpus shifting under a line-keyed
> count. The correction that records a withdrawal is scored by the scan as an
> occurrence of the withdrawn number.** Five instances were filed; this is the
> seventh, and it was manufactured by the repair for the first six.

---

## §3 — THE it.31 CLAIMS, RE-RUN

### 3.1 SATURN — **UPHELD on every item re-run**

    $ python -m pytest tests/saturn/test_v20_r15_it31_open_corpus_counts.py -q
    4 passed in 0.95s                                          [13:48:12Z]

| claim | verdict | evidence |
|---|---|---|
| 3 hardcoded counts retired in `tests/saturn/` | **UPHELD** | `it14_saturn.py:283` → `CELLS_12 - set(cells)`; `it19_theory_digest.py:103` → same; `it27_wing_arm_citation.py:179` → `>= 17`. All three read in source. |
| 1 liveness offender retired | **UPHELD** | `test_r10_it2_spotcheck_reds.py:60` now carries `assert kept, "git ls-files returned no tracked path -- git is not running here"` |
| control `3 failed, 14 passed` | **UPHELD exactly** | re-run 13:48:12Z: `3 failed, 14 passed in 1.99s`, same three node names |
| all-office floor `>= 12` | **UPHELD as a floor** | not exhaustively re-derived; the floor is the claim and the sampled members check out |
| the `438` ledger pin, unstamped in two offices | **UPHELD exactly** | `tests/jupiter/test_v20_r15_it23_fence_and_argument.py:194` `assert len(_src(LEDGER)) == 438`; `tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py:142` `== 438, "the ledger started growing again"` — the defect stated inside its own failure text, as filed |
| four invented timestamps, filed against himself | **UPHELD** | 2–13 minutes wrong, one in the future, caught in his own draft |

> **The timestamp self-report is the best act any office took this window.** He
> found it before it shipped, named the mechanism as his own §1 defect one level
> up, and stated that his detector would not have caught it because **prose stamps
> have no node behind them.** That last sentence is the finding, not the confession.

### 3.2 JUPITER — **two strikes, both on published state, neither on his reasoning**

**Radius `9 RED / 97 GREEN`: STRUCK.** Re-take 13:49:53Z over his `RADIUS` list —
**16 RED / 90 GREEN**, same 106-node population.

> **This office's first re-take used the report table's file names and errored on
> a path. The `RADIUS` constant in the test file is CORRECT
> (`tests/mars_v20/test_it12_the_four_constants.py`, no `v20_r15_` infix); the
> report's prose table abbreviates. That first re-take is withdrawn; the 16/90
> figure is taken from his own frozen list.**

By **JUPITER's own published criterion** — *"a re-take after any journal edit that
does not reproduce this pair has moved something other than the cells this filing
names"* — the criterion fires. He filed 19:09 IST (13:39Z), **before** the
coordinator's edit. The reading was honest; the **baseline** is void.

**His it.31 node, filed `5 GREEN / 2 RED`: STRUCK.** Reads **3 passed, 4 failed**
at 13:48:36Z. The two extra REDs are **his own `C37` nodes**, flipped by the
coordinator applying the row he specified. He asserted `C37` *"cannot turn the
it.30 enforcement node RED"* — verified, true — **and did not check it against his
own it.31 nodes.** Instance seven, §2.4.

**`C37` specified rather than invented: UPHELD.** The reasoning is sound and the
deviation it required was the coordinator's to find (§4.1).

**Class C: NOT REACHED, and the attribution is corrected.** The settlement —
23 live hits, 22 USE, 1 MENTION, *"Class C is not a mention class; it collapses
into Class A"* — is at **`V20_R15_IT30_JUPITER.md:156`, it.30.**
`V20_R15_IT31_JUPITER.md` contains no occurrence of *"Class C"*. **It was not
filed this iteration and this office did not re-run it.**

### 3.3 MARS — **`8 attacked, 7 struck` UPHELD; the geometry verified analytically**

    $ python -m pytest tests/mars_v20/test_v20_r15_it31_… -q
    7 failed, 1 passed in 2.75s          [13:47:48Z, digest ba3eab6437925ff5]

His `3.52s`; the seven named REDs reproduce one for one.

**`MARS-31-B3` geometry — UPHELD, and the monotonicity is not self-evident.**
`Σ_{S∈{32,64,128}} (S/64)^e = 0.5^e + 1 + 2^e`. The `S=32` term is *decreasing*
in `e`, so containment is not visual. It holds because
`d/de = ln2·(2^e − 0.5^e) > 0` for all `e > 0`. **Strictly increasing, so
`e = 2 ∈ (1,3)` is contained by construction, in every possible world.** His
claim is right and stronger than the one-line version implies: **three readings of
one monotone function are one measurement, and the containment carries zero
information about the price.** `MARS-31-R3` — no point at all, measure `e` at
`0 GPU-s` — is adopted.

**`MARS-31-C` — `309.047` survives in two GREEN assertions. UPHELD exactly.**
`tests/mercury/test_v20_r15_it13_phase_c_price.py` runs **8 passed** at 13:48:36Z:

    :48   assert jupiter_it8_formula() == pytest.approx(309.047, abs=0.01)
    :55   assert mid == pytest.approx(309.047, abs=0.01) # S^2, the assumed law

And the docstring at `:47` says `309.02` while the assertion says `309.047` — the
tell is real and `M-30` does not repair it. **Worse than the `62`, as filed: the
`62` survived in prose; this survives in a green assertion.**

---

## §4 — THE COORDINATOR. **THREE ACTIONS: TWO RIGHT, ONE WRONG, AND ONE OMISSION LARGER THAN ALL THREE.**

### 4.1 The `| C37 |` deviation — **RIGHT, and it was not optional**

Independent recompute at 13:50:09Z over the index grammar `^\| C\d+ \|`:

    rows 37
    sha  70ca942b521a98dcc47798a3429ef031e66c4f9cc6fc20ae88a486a0a8e9a89c

**Matches the published `INDEX-SHA256` byte for byte**, over 37 rows, `C1`..`C37`,
contiguity verified `1..37` with no gap. A row written `| 37 |` as specified
**matches that pattern nowhere** — it would have entered no digest, answered no
index lookup, and been invisible to `dead_literals()`. **UPHELD.** And recording
the deviation is the correct form: a deviation recorded is a deviation; a
deviation unrecorded is a defect.

### 4.2 Writing `C35` and `C36` — **his to do, and correctly done**

Both record real body corrections that had no index row. They close SATURN's
standing RED `test_the_highest_body_correction_has_an_index_row`, **verified GREEN
at 13:49:53Z and RED at JUPITER's baseline.** The index is the coordinator's
artifact and a standing RED against it is a standing instruction. **UPHELD.**

### 4.3 An index row with no body correction — **NOT LEGITIMATE, and the instrument says so before this office does**

Index max **37**, body max **36** — `CORRECTION 37` does not exist. Two measured
consequences:

1. `test_the_highest_body_correction_has_an_index_row` now passes **vacuously in
   the direction that matters**: `37 >= 36` holds for any lead, however large.
2. `test_PLANTED_NEGATIVE_dropping_the_top_row_is_caught` is **RED at 13:50:54Z**,
   and its failure is the proof. The negative deletes the top index row and
   asserts `max(body) > max(rows)`. With the index leading by one, deleting `C37`
   **restores parity** — `36 > 36` is false.

> **The planted negative can only fire while index max equals body max. `C37`
> disarmed the guard that protects the index.** That is `V-16` — a check that
> cannot fail — reintroduced by the repair that closed a RED, in the same edit.

**Route:** write body `CORRECTION 37` in the same edit, **or** change the coverage
node to assert equality of maxima and re-plant the negative in both directions.
Not both halves of a paired guard from one side.

### 4.4 THE OMISSION — **the radius was prescribed, the edit landed, the re-take did not**

JUPITER prescribed exactly: *"The coordinator applies `C37` together with `SPEC`
in one edit, then re-takes the 14-file radius once against `RADIUS_BASELINE`.
Two edits, one re-take, one digest."*

**The edit landed. No re-take is filed. The re-take reads:**

| | RED | GREEN |
|---|---|---|
| `RADIUS_BASELINE`, JUPITER, 19:05 IST | 9 | 97 |
| this office, 13:49:53Z, same list | **16** | **90** |

**Eight newly RED, one closed.** The eight are the line-keyed and population-keyed
nodes — `it20_citation_freeze` (3), `it21_heading_anchor` (1),
`it22_independent_census` (2), a fourth `it27_uncited_class` node, and SATURN's
planted negative — **precisely the class that three inserted rows at the head of
the journal shift.**

> **`C36` is the index row that records the it.30 blast-radius failure: four
> JUPITER nodes broken by an uncomputed radius, reverted byte-exact. `C36` was
> written, filed, indexed and digested — and one iteration later the identical
> failure was committed by the same actor at twice the size, in the same edit that
> installed `C36`.**
>
> **A correction that does not change the next iteration's behaviour is a record,
> not a repair. The round has 37 of them and one measurement of whether any of
> them works.**

---

## §5 — DISPOSITION

**29 audited, 7 struck, 21 upheld, 1 not reached.**

| # | target | verdict |
|---|---|---|
| 1 | `MARS-31-A` — tier rule certifies an irreproducible instrument | **strike UPHELD**, reproduced |
| 2 | `MARS-31-A'` — *"recentralises"* as measurement | **strike UPHELD** |
| 3–5 | `MARS-31-B`, `B2`, `B3` | **strikes UPHELD**; `B3` verified analytically |
| 6–7 | `MARS-31-C`, `C2` | **strikes UPHELD**, two green assertions confirmed |
| 8 | `MARS-31-D` | **strike UPHELD** |
| 9 | MARS instrument `7 failed, 1 passed` | **UPHELD**, reproduced |
| 10 | MARS's 14-file radius confirmation | **UPHELD** |
| 11–17 | SATURN: node, 3 retirements, liveness, control, `>=12`, `438`×2, 4 timestamps | **UPHELD**, all seven |
| 18 | JUPITER `RADIUS_BASELINE` `9/97` | **STRUCK** — 16/90 |
| 19 | JUPITER it.31 node `5 GREEN / 2 RED` | **STRUCK** — 3/4 |
| 20 | JUPITER `C37` specification | **UPHELD** |
| 21 | `J-31a` cut, as a mechanism | **UPHELD** |
| 22 | `J-31a` cut, its failure diagnosis | **STRUCK** — cannot name which side moved |
| 23 | JUPITER Class C `23 / 22 USE / 1 MENTION` | **NOT REACHED** — it.30, not it.31 |
| 24 | coordinator `| C37 |` deviation | **UPHELD** — digest recomputed, exact |
| 25 | coordinator writing `C35`/`C36` | **UPHELD** |
| 26 | index row with no body correction | **STRUCK** — disarms the planted negative |
| 27 | coordinator's un-re-taken radius | **STRUCK** — 8 nodes, third occurrence |
| 28 | **this office's replacement tier rule** | **STRUCK** — fourth and fifth clauses adopted |
| 29 | **this office's *"recentralises"* sentence** | **STRUCK** — preference, 2 against 0 |

**THE SENTENCE it.32 INHERITS**

> **The round has spent three iterations learning that a count needs its
> population named, and it has now proved the same thing about a repair: an edit
> needs its radius re-taken, and a correction that is only written is only a
> record. `C36` says so, in the index, in the edit that broke eight nodes.**

**WHAT THIS OFFICE DID NOT REACH.** Named, not hidden:

- **JUPITER's Class C settlement** — filed at it.30, not it.31; **not re-run.**
- **SATURN's `paths-only` / `prose-only` discrimination** — MARS sketched the
  attack (`S1` may measure an empty anchor set) and did not verify it. Neither did
  this office.
- **The all-office `>= 12` census was accepted as a floor, not re-derived
  exhaustively.** Only the named members were checked.
- **`M-30a`/`M-30b` are specified and unapplied** — both RED, both the
  coordinator's to land, and the radius must be re-taken first.
- **§0 and §1 of the theory table**, never swept by anyone, still never swept.
- **The eight newly-RED radius nodes are counted, not diagnosed.** That they are
  the line-shift class is inference from their names and the `+3` offset measured
  at §2.4, not a per-node reading.

**TREE STATEMENT.** HEAD `207e7b9`, **unmoved**; `git status --porcelain` read
**179** at 13:50:54Z. **No git operation was performed by this office.**
**This office wrote exactly one file:** `V20_R15_IT31_INSPECTOR.md`. No file of any
other office was mutated, so nothing required reverting. Every digest in the header
table was computed by this office at the stamped time; where it disagrees with an
office's published value — `V20_R15_JOURNAL.md`, three ways — **the disagreement is
the finding, not an error.**

**Twenty minutes. No git writes. Nothing touched Kaggle.**
