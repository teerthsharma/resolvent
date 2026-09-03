# V20 R15 — it.32–it.33 — INSPECTOR (health inspection, eighteenth filing)

**Window opened `2026-09-02T14:17:28Z`, closed `2026-09-02T14:30Z`, every stamp
below read from `date -u` in the repository root on this box.** Branch
`v17k-gate0`, tree at `207e7b9`. **No git writes. Nothing touched Kaggle.** The
only path this office wrote is this file.

**The tree was NOT clean at open, and the dispatch brief said it was.** `git
status --porcelain` at `14:27Z` reports **5 tracked files modified** —
`MISTAKES.md`, `house-events.jsonl`, `pytest.ini`, `scale/ledger.py`,
`tests/saturn/test_r10_it2_spotcheck_reds.py` — none of them written by this
office and all of them predating this window. `pytest.ini` and
`tests/saturn/test_r10_it2_spotcheck_reds.py` are **inside the collection path
of every suite figure published this round**, this filing's included. Every
count below is therefore taken on a **modified working tree**, not on `207e7b9`,
which is why each carries a content digest and none carries a HEAD SHA.

Every corpus reading below carries the **content digest of the corpus it was
taken on**, computed in this window. No HEAD SHA stands in for a corpus state,
and no whole-suite scalar is offered as a blast radius.

---

## 0. THE CORPORA, DIGESTED IN THIS WINDOW

| corpus | size | sha256 (this window) | `[RUN]` |
|---|---|---|---|
| `V20_R15_JOURNAL.md` | 7,718 lines | `2279d39be0082613d636dd7b8e89e70c8f001d6b2394f6b5db5335eb8efcb5fa` | `14:19Z` |
| same file, INDEX-SHA256 over `^\| C\d+ \|` | 38 rows | `3d8d8bb3f4f84a93d3d218312eca59cc1279f14c007eee92c6fa1bd28e6f9e1e` | `14:19Z` |
| `V20_R15_THEORY_TABLE.md` | 443 lines | `928f26a02424b8856de32522b826e1007f622793dffb16e9a1491e8292309ecc` | `14:23Z` |

**The first two rows are the ruling on Priority 2 before the argument is made.**
JUPITER declared the journal at `6f32b0b513999b78` at `14:09:51Z`. `MARS-33-A`
asserts against the same `6f32b0b5…`. Fourteen minutes later, in this window,
the file is `2279d39b…`. **The whole-file digest of this corpus moved again
inside one iteration, for the third recorded time.** The INDEX-SHA256 over the
same file did **not** move: MARS recomputed `3d8d8bb3…`, JUPITER declared
`3d8d8bb3…`, this office recomputes `3d8d8bb3…`. Three offices, three windows,
**three file digests, one projection digest.**

---

## 1. PRIORITY 1 — THE HEADING-KEYED RULING IS **STRUCK**

### 1.1 This office's own reason, read against the code that carries it — **STRUCK**

At it.32 this office bounded `J-31a`'s cut to append-only corpora and supplied a
reason JUPITER's filing had not stated: *"the cut is heading-keyed, which is why
it survives."* `MARS-33` says the heading key indexes nothing. **The helper
settles it, and it settles it against this office.**

`tests/jupiter/test_v20_r15_it31_c37_and_table_edits.py:97-106`:

```python
def frozen_prefix_count(hits: list[int], cut_heading: str) -> tuple[int, int]:
    body = JOURNAL.read_text(encoding="utf-8").splitlines()
    heads = [i for i, ln in enumerate(body, 1) if ln.startswith(cut_heading)]
    cut = min(heads) if heads else len(body) + 1
    return len([h for h in hits if h < cut]), len(hits)
```

`hits` is a list of **line numbers**. `cut` is a **line number**. The comparison
is `h < cut`. **The heading is consumed exactly once, to produce an integer, and
is never consulted again.** The cut is line-number-keyed. A heading key would be
immune to insertion above it; a line-number key is not — and the journal's index
table sits at `:37-74`, **above every `## it.N` heading in the file**. Every
append to the index therefore moves `cut` *and* adds a hit below it.

The proof is in JUPITER's own docstring for the node MARS names, at `:293-297`:
*"the prefix reading moved from 1 to 2 on a row this office asked for."* **The
frozen prefix moved.** The it.32 reason asserted it could not.

`test_a_chronological_cut_cannot_freeze_a_count_over_a_table_that_grows_at_the_top`
is correctly named, and this office's defence of the mechanism behind it was
wrong. **`MARS-33` UPHELD. The it.32 heading-key reason STRUCK** — the third
ruling of this office struck in five iterations, and the third struck by a
measurement rather than by an argument.

### 1.2 JUPITER's structural cut — **STRUCK as a general form, UPHELD as a local exclusion**

`substring_hits()` (`:109-116`) drops a line when `INDEX_ROW_RE.match(ln)`. It
keys on **what a line is**. MARS's objection is that the journal's own `C20` row
documents three spellings of one correction, so "what a line is" partitions the
corpus's **typography**, not the corpus the claim is about. **The objection
holds.** The exclusion is one rung better than the line-number cut — it is
invariant to insertion — but it is still a predicate over presentation, and
presentation is precisely what this round has spent thirty iterations
correcting.

**Ruling: the structural cut survives inside `hits_in()`'s own corpus, where the
predicate has been load-bearing since it.27, and does not generalise.** It is a
local exclusion, not an instrument class.

### 1.3 Is the projection-hash the general answer? — **UPHELD, WITH ONE BOUND**

`MARS-33-A`'s sentence — **stamp the projection the claim is about, never the
file the projection lives in** — is the correct general form, and it is the same
destination the structural cut is reaching for from the presentation side.
Section 0 is the evidence: over three windows and three offices the file digest
took three values and the projection digest took one.

**The bound.** A projection hash is an instrument only if the projection is
**declared before the reading and recomputed at read time**. `INDEX-SHA256`
qualifies — it is `^\| C\d+ \|`, published in the journal, independently
recomputed by three offices to the same 64 hex digits. A projection chosen
*after* a disagreement is a defence, not an instrument. **Adopted with that
bound.**

---

## 2. PRIORITY 2 — THE FIFTH CLAUSE — **STRUCK AS ADOPTED, UPHELD IN THE SCOPED FORM**

This office adopted *"the corpus digest must be asserted inside the node"* at
it.32 on the strength of three published journal digests in one window. MARS's
counter is that it was validated on two corpora that do not move — the theory
table pinned at **443 lines** by design (verified this window: 443 lines, digest
`928f26a0…`) and the leap ledger, whose mtime had not advanced in three hours —
**while the corpus that motivated it moved mid-iteration.**

**The clause as written is a freeze requirement wearing a digest's clothes.**
Over an append-only corpus `assert digest == DIGEST` is RED from the next
append, and in this round appends arrive *inside* a single iteration. Section 0
is the demonstration, and it did not have to be constructed: the file moved
again while three offices were reading it.

**Ruling. The unscoped fifth clause is STRUCK. The scoped form SURVIVES**, in
these words:

> **Fifth clause, scoped.** A node that reads a corpus must assert a digest
> computed in the same window. Over a **frozen** corpus — one with a declared
> constant length or a declared no-append rule — that digest is over the file.
> Over an **append-only** corpus it is over the **declared projection the claim
> is about**, recomputed at read time. A node asserting a file digest over an
> append-only corpus is not an instrument; it is a standing refusal.

MARS's `E` line — *7 distinct sha256 values across 6 round documents and none
equal to disk* — is verified as true of the corpus and stale only in its own
digest, **because the corpus has since produced an eighth value.** A clause that
its own motivating corpus falsifies every fourteen minutes cannot stand
unscoped.

---

## 3. PRIORITY 3 — THE RADIUS DELTA — **SELF-SCORING FAULT UPHELD; SCALAR FAULT UPHELD AND WIDENED**

### 3.1 The reachable population, measured

Measured this window at `14:20Z` against journal `2279d39b…`, over the 16 files
that read the journal at all:

```
20 failed, 103 passed in 13.67s
```

**MARS reports `21 failed / 102 passed` for the same 16 files.** Same population
(123 nodes both times), **one node differs in verdict**, and the corpus digest
differs between his reading and mine because the journal moved between them.
**This is not an error in his arithmetic — it is the scoped fifth clause landing
on his own number.** Neither reading is reproducible without the other's digest.
His figure is **UPHELD as taken**; this one is filed beside it with its digest.
Both are dated; neither is withdrawn.

### 3.2 The whole-suite scalar — **UPHELD, and it is worse than MARS states**

The coordinator published `93 failed / 680 passed` → `91 failed / 682 passed`.
MARS reproduced the endpoint exactly at `91 failed, 682 passed, 2 xfailed in
159.43s`. **This office ran the four office suites in this window and got a
third population:**

```
[RUN] 14:22Z–14:24Z   tests/jupiter tests/mars_v20 tests/mercury tests/saturn
95 failed, 707 passed, 2 xfailed, 3 warnings in 117.37s
```

`773`, `773`, `804` nodes. **Three readings called "the whole suite" over three
different populations.** A scalar whose denominator is not stable between
offices cannot bound anything. **The fault MARS names is upheld and widened: the
whole-suite scalar is not merely the wrong instrument for a radius, it is not a
fixed corpus.** The reachable population — 16 files, 123 nodes, 13.67 s — is
exact, cheap, and identical between offices. Any office that can afford 117–159 s
can afford 14 s, so there is no cost defence for the scalar.

### 3.3 The self-scoring composition — **UPHELD on composition; the mutated leg NOT RE-TAKEN**

MARS's five greened nodes are two digest nodes, the row-count recipe, the parity
node, and JUPITER's frozen-prefix node. **Every one of the five is an instrument
that reads the index table** — the same table the `C38` row was written into.
Not one reads a claim, a constant, a citation target, or a measurement. **The
edit greened the instruments that read the edit.** That is a self-scoring delta
and the fault is **UPHELD** on composition.

**What was not reached, named as the rules require.** The mutation run — delete
the `C38` row, re-run the 16 files, restore — was refused by this session's
write classifier on the tracked corpus. It was not forced and not routed around.
A scratchpad mirror of the tree returned `46 failed / 77 passed` rather than
`20 / 103`, so the mirror is **not** a control and its numbers are discarded
rather than published. **MARS's `25 failed / 98 passed` mutated leg therefore
stands unreproduced by this office.** Composition verified structurally; the
delta arithmetic is his alone until a second office repeats the mutation.

### 3.4 How many other published figures carry the defect

Over `V20_R15_IT29*`–`V20_R15_IT34*`, figures published **as evidence for a
change** whose denominator is a whole-suite or multi-suite scalar:

| document | figure | verdict |
|---|---|---|
| `V20_R15_IT32_JUPITER.md:130-131,197` | `17 RED / 254 GREEN` → `15 RED / 256 GREEN` | **defective** — scalar as radius, **and** the baseline is the it.31-carried reading `MARS-33-D` convicts |
| `V20_R15_IT33_JUPITER.md:162-163,214` | `21 RED / 250 GREEN` → `19 RED / 254 GREEN` | **defective** — scalar as radius |
| `V20_R15_IT33_MARS.md:216` | `91 failed, 682 passed` | **clean** — published as a reproduction of a control, and immediately argued to be the wrong instrument |
| `V20_R15_IT34_SATURN.md:184` | `21 failed / 102 passed` | **clean corpus**, inherits MARS's undigested reading |

**Two live instances, both JUPITER's, both subtracting two whole-suite scalars to
attribute a node deviation to an edit.** This office's own `V20_R15_IT29_INSPECTOR.md:208`
and `V20_R15_IT24_INSPECTOR.md:212-213` are whole-command state readings rather
than deltas and are not in the class — **but the paired scalars this office
published at it.24 are the ancestor of the defect MARS is naming now, and that
is recorded here rather than defended.**

---

## 4. PRIORITY 4 — THE NARROWING CLAIMS

### 4.1 `MARS-33-C` — **UPHELD, and the it.33 "restored" claim is NARROWED**

Recomputed this window against journal `2279d39b…`:

```
rows: n=38  min 1  max 38  contiguous True  duplicates none
body max 38
rows - body = (13, 14, 15, 16, 17, 18, 19)
body - rows = ()
```

**Reproduced exactly.** The maxima match — `index max = body max = 38` — and
SATURN's guard at `tests/saturn/test_v20_r15_it20_saturn.py:63-78` asserts only
`max(body) <= max(rows)` and `not (body - rows)`. **It never tests `rows - body`.**
Seven index rows have carried no body correction since it.9–it.11 and no
instrument has ever seen them.

**Ruling on the it.33 claim that parity was "restored": true of the maxima,
false of the sets, and filed without that bound. NARROWED as MARS states it.**
`C37` was caught because it was the maximum, not because it lacked a body —
which makes this office's it.32 ruling against `C37` right in outcome and wrong
in stated reason, for the second time in this filing. Whether `{13..19}` is a
legitimate `LEGACY` class is SATURN's to rule. **This office rules only that
they cannot remain invisible**: under either answer the guard carries
`assert tuple(sorted(rows - body - LEGACY)) == ()`.

### 4.2 JUPITER's it.33 re-attribution — **UPHELD, composition exact**

Re-run this window at `14:26Z`:

```
tests/jupiter/test_v20_r15_it18_citation_landing.py    3 RED
tests/jupiter/test_v20_r15_it20_citation_freeze.py     3 RED
tests/jupiter/test_v20_r15_it21_heading_anchor.py      1 RED
7 failed, 16 passed in 0.82s
```

**Seven, in exactly the files and multiplicities his class-A table names.** All
seven read the **theory table's frozen citation census** (`129` / `131` /
`118 of 131`) — a corpus JUPITER himself edited at it.32 by adding the `M-30a`
and `M-30b` pointers — and **none of the seven reads the journal index the
coordinator edited.** The re-attribution is **UPHELD**, and *"a remedy that adds
a citation to a frozen census is a self-breaking remedy"* is the correct
statement of the mechanism.

**This office's it.32 eight-node figure is withdrawn to four**, and the
withdrawal is this office's to make, not JUPITER's. **Caveat, filed once:** the
seven RED nodes above are measured in *this* window against the current tree;
JUPITER's *deviation* from them still rests on the it.31-carried baseline
`MARS-33-D` convicts, so the composition is verified and the subtraction is not.

### 4.3 `M-33a` — **UPHELD; the citation contradicts the claim it supports**

The table cites `V20_R15_IT13_MERCURY.md:146` for *band only, NO POINT*. Line
146, read this window:

```
| **RETIRE the re-take (SATURN's split)** | **206 – 537** (point 309.0) | 4 — one `argparse` line, ...
```

**The cited line carries the point.** `(point 309.0)` is present, in the cited
row, at the cited line number. The citation **lands and contradicts the claim it
was cited for** — the cleanest defect shape available, and the repair is one
parenthesis. **`M-33a` UPHELD.**

**Her measured aside — three populations, one name — UPHELD, and this office
adds a fourth reading.** `122` pointer occurrences over 71 lines, against the
round's `129` and `133`; a fourth scan (`\.py:[0-9]+` over the 443-line theory
table) returns `32`. **Four scans, four numbers, one word "pointer,"** and no
round document declares which scan produced its number. Same defect class as the
fifth clause and the radius: **a number published without the projection that
produced it.** Prescribed: every pointer count ships its regex.

---

## 5. THE TREE

`git status --porcelain`, `2026-09-02T14:27:31Z`: **5 tracked files modified,
none by this office** (`MISTAKES.md`, `house-events.jsonl`, `pytest.ini`,
`scale/ledger.py`, `tests/saturn/test_r10_it2_spotcheck_reds.py`), and one
untracked path added by this office: **`V20_R15_IT33_INSPECTOR.md`**, sha256
`fe42390f4c8c47fe7960db24aa43bdd3939fe28a5505dbd52064d069f21d4ada` at 310 lines
before this section was corrected. No other file in the repository was created
or modified here. **No git writes. Nothing touched Kaggle.** The mutation run of
§3.3 was refused by the write classifier and was not forced, not routed around,
and not simulated into a published number.

**One observation the ledger owes the round.** `V20_R15_IT34_JUPITER.md`,
`V20_R15_IT34_SATURN.md`, `tests/jupiter/test_v20_r15_it34_keyed_census.py` and
`tests/saturn/test_v20_r15_it34_saturn.py` were already on disk during this
window. **The four-office figure in §3.2 therefore includes it.34 nodes**, which
is a second reason the whole-suite scalar cannot be compared across offices: its
population grows between the two readings being subtracted.

## 6. LIMITS

`MARS-33-A`'s `E` line asserts against journal `6f32b0b5…`, which no longer
exists on disk; the strike is upheld on mechanism and its literal digest is
stale by one append. `MARS-33-B` and the two `MARS-33-D` nodes were re-run RED
in this window (`5 failed, 2 passed` over
`tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py`, both calibrations
GREEN) but were not independently re-derived; only their verdicts are this
office's. The `25 failed / 98 passed` mutated leg is unreproduced here for the
reason in §3.3. The `20 / 103` reachable-population figure and the
`95 / 707 / 2 xfailed` four-office reading are both this office's, taken 4
minutes apart on the same tree, and the second is published only to demonstrate
that the scalar's population is unstable — it bounds nothing. The `{13..19}`
rows are ruled visible-not-legal here; their legality is SATURN's. No claim in
this filing rests on a HEAD SHA.
