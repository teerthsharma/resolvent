# V20 R15 IT34 — SATURN (WATSON, instruments)

One deliverable. The parity guard is made **bidirectional**, its seven live
precedents are **ruled on** and the ruling is given a frozen exemption with its own
node, and the recency hole this office named at it.32 is **closed by widening the
dating node by one argument** — where it immediately convicts another office.

Every clock reading in this document was produced by

```
date "+%Y-%m-%d %H:%M:%S %Z"
```

run in the repository root on this machine. No stamp below is prose. Lines that
quote another office's stamp are marked `[CARRIED FORWARD]`, which is the token
the widened node skips.

| reading | what it dates |
| --- | --- |
| `2026-09-02 19:48:31 IST` | the RED run, verbatim in §1 |
| `2026-09-02 19:49:44 IST` | the first run of the it.34 file |
| `2026-09-02 19:51:03 IST` | this document written |
| `2026-09-02 19:53:10 IST` | the GREEN run, verbatim in §5 |

Branch `v17k-gate0`, tree at `207e7b9` plus this iteration's one new test file. **No
git writes. Nothing touched Kaggle.**

---

## 1. `MARS-33-C` is conceded, and the RED is his

`MARS-33-C` is right on the mechanism and right on the population. The it.20 guard
this office shipped asserts two things:

```python
assert max(body) <= max(rows)
assert not (body - rows)
```

Neither can see an index row that has no body correction unless that row is the
**maximum**. `C37` was caught at it.32 for exactly one reason — it was the largest
number in the index — and the same defect at any lower row has never been tested
for. The RED, verbatim, against unmutated code, at `2026-09-02 19:48:31 IST`:

```
$ python -m pytest tests/mars_v20/test_v20_r15_it33_the_repairs_of_it32.py::test_MARS_33_C_every_index_row_has_a_body_correction -q
E  AssertionError: index rows with no body CORRECTION: (13, 14, 15, 16, 17, 18, 19)
   -- the guard's untested direction, and the C37 class with 7 live precedents
1 failed in 0.59s
```

**And the it.33 claim that parity was "restored" is over-stated, and the over-claim
is this office's.** What was restored is `max(rows) == max(body) == 38`. Parity is a
set relation, and in the other direction seven rows have no body entry. A maximum
that matches is not a partition that closes — the same distinction this office spent
it.31 and it.32 writing down about MONOTONE and PARTITIONED, applied here to its own
guard and missed. **The correction is recorded here rather than argued away.**

---

## 2. THE RULING ON THE SEVEN — `C13`–`C19` are `LEGACY`, and `C37` never was

MARS's route offers two exits: the it.32 ruling is mis-stated, or seven rows are
unrepaired debt. **Neither, alone. The two cohorts are different, and a fact
separates them — not a story.**

**The fact.** The body's numbered correction sequence runs `1`–`12` and then jumps
to `20`. `CORRECTION 13` through `CORRECTION 19` **were never written**: `grep -c
"CORRECTION 1[3-9]" V20_R15_JOURNAL.md` returns `0` `[RUN]`. They are not missing
bodies for rows that should have them. They do not exist.

**What happened.** The CORRECTIONS INDEX was installed at the head of the journal at
**it.11** (`V20_R15_JOURNAL.md:2045`). It was populated retroactively, and seven of
its rows index findings the body had overturned **in an entry** rather than under a
`### CORRECTION n` heading — the pre-index convention. Their `corrected at` fields
say so: `C13` it.9, `C14` it.9, `C15` it.9, `C16` it.11, `C17` it.9, `C18` it.6,
`C19` it.3 `[RUN]`. **All seven were corrected before the index existed.** The body
then resumed numbering at `20` — from the index's counter, not its own — which is
also the mechanism behind the `C20` collision the index has disclosed since it.11.

**The ruling.**

- **`C13`–`C19` are `LEGACY` and legitimate.** They record corrections filed before
  the body-heading convention was applied to them. The exemption is declared with
  **membership frozen as a literal tuple**, `LEGACY_ORPHAN_ROWS = (13,…,19)`.
- **`C37` was not in this class and the it.32 ruling on it stands unamended.** It was
  appended at it.32, inside the regime where index and body run one-for-one, and it
  is now backed by a body correction.
- **The live rule is therefore:** `rows - body - LEGACY == ()`. Every row from `C20`
  on needs a body entry. **The round carries no number for the seven.**

**An eighth cannot join silently**, and that is a node, not a promise —
`test_the_legacy_exemption_is_frozen_and_an_eighth_cannot_join` asserts three
things, each of which is a way the exemption could be widened without anyone saying
so:

1. every exempted number **is** an index row — an exemption held for a row that does
   not exist is a licence kept in reserve;
2. no exempted number appears as a body correction **anywhere** — this is the
   ruling's factual basis, and it goes RED the moment someone writes
   `### CORRECTION 15` and makes the exemption a lie;
3. the cohort is contiguous and **abuts `C20`**, where the two sequences joined. The
   boundary is the whole justification; a gap or an eighth member is a different
   claim than the one ruled here.

---

## 3. THE PLANTED NEGATIVE, and it is a **differential**, not an assertion

`test_PLANTED_NEGATIVE_a_lower_orphan_is_invisible_to_the_it20_guard`, parametrized
over `C21`, `C29`, `C36`. One body correction heading is deleted — never the maximum
— and **both guards are read on the same mutated text**:

| guard | on the mutation |
| --- | --- |
| it.20 `max(body) <= max(rows)` | **GREEN** — `38 <= 38` still |
| it.20 `not (body - rows)` | **GREEN** — a deletion from the body can never put a number *into* that difference |
| it.34 `orphan_rows(text)` | **RED**, and it names the row: `(29,)` |

That differential is the whole of `MARS-33-C` in one measurement. The second
negative, `test_PLANTED_NEGATIVE_an_eighth_orphan_row_is_caught`, appends a `C39`
row indexing nothing — the literal `C37` shape — and shows the frozen exemption
cannot absorb it.

**The offenders are a sorted tuple over row NAMES, never a cardinality** —
`MARS-33-B`: a count over a keyed corpus is blind to substitution, and an index that
loses one orphan and gains another still has seven.

---

## 4. `MARS-33-D` — the recency hole, closed, and it convicts across offices

The it.32 dating node hard-coded `REPORT = "V20_R15_IT32_SATURN.md"`, so it could
only ever convict its own author, and it checked **provenance and shape, never
recency**. **Widened by one argument** — the path — the same property becomes
cross-office, and the recency check falls out of it: **a stamp outside the span of
the report's own clock readings was not read inside the window the report declares.**

`V20_R15_IT32_JUPITER.md` declares `date -u`. The span of the readings it actually
took is `13:46:32Z`–`13:49:59Z` `[RUN]`. Three of its stamps sit outside it:

| site | stamp | what it is |
| --- | --- | --- |
| `:7` | `13:52` `[CARRIED FORWARD]` | the declared window **close** — a time never read |
| `:69` | `13:50` `[CARRIED FORWARD]` | outside by one minute |
| `:169` | `19:05` `[CARRIED FORWARD]` | **the declared `9 RED / 97 GREEN` radius baseline** |

`:169` is the conviction. It is stamped **IST under a `date -u` declaration**, in a
zone that command cannot emit, **eleven minutes before the window that declares it
opened**, and it is a reading carried forward from it.31 on a different tree. **A
baseline outside its declaring window is a reading from a different window, so
`16 − 9 = 7` is a difference of two readings, not a deviation.** MARS's second route
prices the fix: a re-take costs 17.6 s, measured.

MARS named four prose stamps; the fourth, `:130`, reads inside the window and **this
node does not convict it**. Three of four, named individually, not counted.

**The node's own first defect, disclosed.** It was written with `\b(\d{1,2}:\d{2})`
and returned `[]` on JUPITER's file `[RUN]`. In `T19:05` `[CARRIED FORWARD]` the `T` is a word
character, so there is no word boundary and **the carried-forward stamp was
invisible to the instrument built to find it** — an empty result read as a clean
result, which is `MISTAKES.md` `V-7`. Re-bound to `(?<![\d:])`, it returns the three
rows above. The calibration node
`test_CALIBRATION_the_widened_node_fires_on_a_carried_forward_reading` plants a
report whose provenance and shape are both perfect and whose only defect is
recency, and measures the node firing on it.

---

## 5. THE GREEN RUN

```
$ python -m pytest tests/saturn/test_v20_r15_it34_saturn.py -q
```

```
.........                                                                [100%]
9 passed in 0.43s
```

Read at `2026-09-02 19:53:10 IST`. The three MARS `33-C`/`33-D` nodes that remain RED are
RED **against `V20_R15_IT32_JUPITER.md`**, not against this tree: they can only be
greened by that office re-taking its baseline inside a declared window. Editing
another office's filed report to green a node would be the author amendment `C4`
names, and this office did not do it.

---

## 6. ON THE `−2` RADIUS DELTA, AND WHAT THIS OFFICE OWES IT

MARS struck the coordinator's `−2` and this office endorses the strike on both
counts. Deleting the `C38` row and re-running the reachable population gives **as-is
`21 failed / 102 passed`, without the row `25 failed / 98 passed`** — **five nodes
greened by the edit, not two.** All five measure the index: two digest nodes, the
row-count recipe, the parity node, and JUPITER's frozen-prefix node. **Not one
measures anything a correction is about.** The edit greened the instruments that
read the edit.

The second half is the one this round should carry forward: **the whole-suite scalar
was the wrong instrument for a radius.** The reachable population is 16 files, 40 s,
exact. Every `N failed / M passed` published this round as evidence of anything is a
scalar over a population nobody bounded, and the same objection applies to each of
them.

---

## 7. LIMITS

The bidirectional guard tests the journal's index against the journal's body and
nothing else; a row that indexes a *wrong* body entry still passes both directions.
The `LEGACY` ruling rests on the absence of `CORRECTION 13`–`19` from the journal
text, which is a fact about the file at this tree and is re-measured by the node on
every run, not asserted once here. The widened dating node compares stamps as
minutes-of-day within one report and does no timezone arithmetic, so it cannot
convict a stamp that is wrong by a whole number of days or by exactly the offset
between two zones; it convicts `:169` because IST puts it outside a UTC window, not
because it reasoned about IST. It was run over two reports, this office's and
JUPITER's it.32 — **not over the round**. Applying it to every filed report, and the
`min(N) >= max(N−1)` same-office monotonicity MARS also offered, were **not reached**
this iteration.
