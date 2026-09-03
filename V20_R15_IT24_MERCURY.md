# V20 R15 — it.24 — MERCURY (arena and measurement)

Readings dated. Wall clock armed `11:46:47Z`, published `11:59Z`.
Branch `v17k-gate0`. No git writes. Nothing touched Kaggle.

---

## 0 — THE ONE FACT THAT REFRAMES EVERYTHING BELOW

**The theory table was edited by another office at `11:51:49Z`, five minutes into
this iteration, while this office was measuring it.**

```
[RUN] date -u; stat V20_R15_THEORY_TABLE.md          11:53:52Z
   -> mtime 2026-09-02 17:21:49.816713500 +0530  ==  11:51:49Z
[RUN] occurrence count, MERCURY-R1 step 1
   11:49:35Z   occurrences 129   pointers 103   scored pointers 93
   11:56:26Z   occurrences 120   pointers  96   scored pointers 86
[RUN] grep -c "" V20_R15_THEORY_TABLE.md   ->  443   (unchanged; J-17e held)
```

**Nine occurrences and seven pointers left the table mid-iteration; the file did
not change length, so the in-place rule J-17e was honoured.** Eight of the nine
are the `:1` file-pointers this office banked at it.22 as `LINE_ONE_IDIOM`.

Every count in this report is therefore a **dated re-take, not the it.22 census.**
It is also the reason the last section of this report says the frozen hex is the
weakest part of the repair and the two negatives are the strong part.

---

## 1 — REPAIR 1. THE SEAL NOW FIRES ON TAMPERING, NOT ON WORK

### 1.1 The RED at HEAD, verbatim, against unmutated code

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it22_independent_census.py -q   11:47Z
   4 failed, 3 passed in 0.60s
   FAILED ...::test_mercury_census_digest_r1
   FAILED ...::test_m22a_three_lines_low_contract_239_does_not_carry_f3
   FAILED ...::test_m22a_compound_half_constants_13_does_not_carry_beta
   FAILED ...::test_m22a_line_one_idiom_arm_pl_1_carries_no_measurement
```

```
[RUN] python -m pytest ...::test_mercury_census_digest_r1 -q                      11:48Z
E   AssertionError: MERCURY-CENSUS-R1 moved: 5438604c1de06232d317600b403d6480b6b8d0555e094dafc8c64281ccffa997
E   assert '5438604c1de0...64281ccffa997' == '7b1e67cd9e5d...ceb0ad1f6e9b4'
E     - 7b1e67cd9e5d049fff93a7795cd6cce1aabd5b95b6a42892870ceb0ad1f6e9b4
E     + 5438604c1de06232d317600b403d6480b6b8d0555e094dafc8c64281ccffa997
```

**This office published `3 failed, 4 passed` at it.22 and the INSPECTOR got
`4 failed`. He is right and the extra one is the seal itself.** The it.22 report
counted the seal as passing because it passed at the instant it was frozen. It has
been red since.

### 1.2 The charge, confirmed with this office's own numbers

```
[RUN] MERCURY-R1 step 1, MERCURY_LIVE split                                       11:49Z
   pointers 103   in the three append-growing files 10
   V20_R15_LEAP_LEDGER.md 7   V20_R15_JOURNAL.md 2   house-events.jsonl 1
```

**Ten of the 103 rows the seal hashed came from the three files this same file's
section 1 refuses to score.** The seal's population and the scored population were
never the same population. And the mechanism is real, not theoretical:

```
[RUN] the it.22 recipe R1 (UNFILTERED) against one ordinary mid-audit journal write  11:56Z
   base                                    20a1fab10012a6e2d8ea9e8dd582fff7012ffbe48ae0b731c61ce0dea1ab616a
   after inserting ONE line in the journal cc6d433d7fdffb16f212687ffccbefebb5e14462284734bccab3b9eb37b5ef07
   SEAL MOVED ON ORDINARY WORK: True
```

The journal's own `P-6` says an append shifts every line below it; the journal is
written mid-audit; the table cites it at `:645` and `:650`. **One ordinary write
turns the it.22 seal red.** A seal that alarms on every append is a seal that gets
switched off, and this round has struck three instruments this month for exactly
that.

### 1.3 The amendment — one clause

**`MERCURY-R1b`: the digest's population is the SCORED population, filtered by the
same `MERCURY_LIVE` rule step 2 already uses.** No whitelist is introduced and none
is reintroduced — `MERCURY_LIVE` is a three-name *exclusion*, derived in this
file since it.22, and the it.22 `.lean` omission came from an *inclusion* list.
The distinction is the one that matters: an exclusion list that is wrong makes the
seal **stricter**, never blind.

`tests/mercury/test_v20_r15_it22_independent_census.py` — `scored_pointers()` and
`census_digest(override=...)` added; the digest test now calls them.

### 1.4 THE TWO NEGATIVES — VERBATIM

Both run **entirely in memory** through `census_digest(override=...)`. Nothing is
written to the repo, so neither negative can be the thing that moves the seal it
tests. `tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py`.

```
[RUN] python -m pytest tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py -v   11:57:51Z

tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py::test_negative_GREEN_the_seal_does_not_move_on_an_ordinary_append PASSED [ 50%]
tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py::test_negative_RED_the_seal_moves_when_a_line_moves_under_a_scored_pointer PASSED [100%]

============================== 2 passed in 0.72s ==============================
```

**NEGATIVE 1 — GREEN ON AN ORDINARY APPEND.** Appends 50 lines to **every** one of
the three `MERCURY_LIVE` files **and** to **every one of the 86 files the seal
actually scores**, then asserts the digest is unchanged. It is blind to both *by
construction*: the growing files are not in the population at all, and an append
at EOF of a scored file moves no line above it.

**NEGATIVE 2 — RED WHEN A LINE MOVES UNDER A SCORED POINTER.** Inserts **one** line
at the top of each scored cited file, one file at a time, and asserts the digest
moves for **every** file — `unmoved == []`. A `+1` shift is the smallest edit that
moves every line of a file, and it is invisible to `test_every_pointer_resolves`,
which stays green through all of them because the file exists and the line is
still in range. **That gap is the seal's entire reason to exist and this is the
only thing in the round that demonstrates it.**

A third test, `test_the_two_negatives_are_not_the_same_test`, closes the two ways
NEGATIVE 1 could pass for a bad reason (empty population; a LIVE file leaking in).

### 1.5 FINDING M-24b — THE PRICE OF THE `LIVE` LIST, WHICH J-23g DID NOT TAKE

J-23g ruled `LIVE` a hand-maintained list, owner named, and conceded its own
criterion contradicts it: `V20_R15_LEAP_LEDGER.md` has sat at 438 lines for three
iterations. **This office does not dispute the ruling.** A judgement with an owner
beats a criterion that reflips a file's scoring status every iteration.

**It disputes the price, with the measurement the ruling is missing:**

```
[RUN] MERCURY-R1 step 1, refused pointers by file                    11:49Z and 11:56Z
   V20_R15_LEAP_LEDGER.md   7  of 10      (438 lines at it.20, 438 now)
   V20_R15_JOURNAL.md       2  of 10      (3908 lines at it.20, 5145 now)
   house-events.jsonl       1  of 10      (13521 lines)
```

**Seven of the ten pointers the seal gives up are into the file that has not
moved.** Seventy percent of the seal's blind spot is bought to protect against a
file that has not changed since the census that refused it for moving, while the
two files that actually churn cost three. That is the number to weigh the list
with; it is banked GREEN in `test_m24b_the_ledger_is_seven_of_the_ten_refused_pointers`,
which also goes RED if the ledger starts growing again.

### 1.6 THE INSPECTOR'S RULING ON THE SECOND CENSUS — TAKEN, NOT ARGUED

> *"Two files writable by one process is one office wearing two hats — what it
> actually buys is two edits in two grammars, a real raise in cost, not
> tamper-proof. And it certifies the population, not the landings — which is where
> the argument defects live."*

**Both halves are right and this office withdraws any tamper-proof reading of
it.22.** On the second half, aimed where it helps: **a second census of the
landings is priced, not taken.** The population census is 103 pointers read by one
regex — seconds. A landings census is 86 scored pointers each requiring a human
reading of *the cell's claim beside the cited line*, which is what produced
`9 of 12` at it.22 from a sample of twelve. **At the it.22 rate, 86 landings is
roughly 7× the reading that consumed most of that iteration — one full iteration
of one office, and it cannot be split without splitting the judgement.** It is not
reachable inside a 20-minute wall clock and this office will not pretend
otherwise. What it *is* worth: the argument layer is the only layer with a
measured defect rate (`3 of 12`), and no instrument in the round can see it.

**Corroboration arrived unasked during this iteration.** Two of the three it.22
argument defects were repaired by another office at `11:51:49Z`:
`ceq/arm_pl.py:1` and `CEQ_V20_R15_CONTRACT.md:239` are **both gone from the
table** (`[RUN] grep -o` at 11:57Z: 0 hits each; `it12_constants.py:13` still
present). Both tests are retired into **recurrence guards** that go RED if the
citation returns; `LINE_ONE_IDIOM`'s class test is now non-vacuous in the other
direction — `:1` pointers `8 -> 0`, and the test fires if one comes back.
`COMPOUND_HALF` stands RED, correctly, because the defect stands.

---

## 2 — REPAIR 2. THE `28 vs 25` MANIFEST GAP. **CLOSED.**

Open since it.18, unreached for five iterations. **VERDICT: two populations
compared by mistake. It dissolves on inspection, and the arithmetic closes exactly
with no residue.**

**What each number counts:**

| number | population | source |
|---|---|---|
| **28** | census-**failing citation OCCURRENCES** in `V20_R15_THEORY_TABLE.md` | `[CITED: JUPITER]` it.17 §2.1 — 27 table rows, the row `C26, C74` carrying two occurrences of one pointer. 27 + 1 = 28 |
| **25** | distinct **repaired TARGET LINES** in `MANIFEST` | `[CITED: INSPECTOR]` it.18 §1.3, read at `:40-82` |

They are not the same object and were never comparable. The mapping is **not 1:1
in either direction**:

```
  one-to-many   C44  ceq/hankel.py:1 -> :75, :108, :131, :279              +3
  many-to-one   C26 + C74           -> V20_R15_IT13_MERCURY.md:146         -1
  not-a-line    C10 is a LABEL repair inside V20_R15_LEAP_LEDGER.md,
                certified by test_the_F4_overload_labels_name_the_sense_
                actually_at_the_line, NOT by MANIFEST                      -1
  not-a-census  C95 scale/negation_scope.py:286, ruled by JUPITER on the
                named-symbol rule, never a census failure                  +1

  28 - 1 - 1 + 3 + 1 = 30
```

```
[RUN] len(MANIFEST) in tests/jupiter/test_v20_r15_it17_citation_landing.py   11:48Z
   -> 30      (no duplicate keys; 31 raw lines, one entry's value wraps)
```

**30 = 30. THERE ARE NO MISSING REPAIRS.** The INSPECTOR's own hedge at it.18 —
*"the gap may be grouping rather than omission"* — was right. Each of the four
bridge legs is separately asserted, not asserted-about, in
`test_the_28_vs_25_gap_is_two_populations`, plus a control
(`test_the_28_vs_25_gap_has_no_third_reading`) that fires if anyone re-reads the
two as one population.

**Why the specific `25` cannot be re-run, said plainly:**

```
[RUN] git ls-files --error-unmatch tests/jupiter/test_v20_r15_it17_citation_landing.py   11:52Z
   -> error: pathspec ... did not match any file(s) known to git
```

The file is untracked, so the it.18-era dict is unrecoverable. **The `25` is
`[CITED: INSPECTOR]`, not `[RUN]`.** It does not need to be re-run: the verdict
turns on the mapping being many-to-one and one-to-many, which *is* re-run above,
and that is sufficient to close the item at any dict size.

**Closing it loudly, as asked:** this sat five iterations because closing it needs
arithmetic across two offices' artefacts and nobody owned the arithmetic. It cost
under ten minutes once someone did. **The lesson is not about the 28 or the 25 —
it is that an item phrased as a discrepancy gets deferred by every office that
assumes the discrepancy is real.**

---

## 3 — SUITE STATE

```
[RUN] python -m pytest tests/mercury/ -q                                  11:57Z
   2 failed, 146 passed in 18.00s
   FAILED tests/mercury/test_v20_r15_it10_eval_independence.py::test_it8_aggregate_verdict_agrees_with_its_own_cells
   FAILED tests/mercury/test_v20_r15_it22_independent_census.py::test_m22a_compound_half_constants_13_does_not_carry_beta
```

**Both REDs are intended.** `COMPOUND_HALF` is the one it.22 argument defect still
standing in the table (`it12_constants.py:13` still cited, `-0.032353` still at
`:14`). `test_it8_aggregate_verdict_agrees_with_its_own_cells` is in a module this
office did not open this iteration; **it is not verified as pre-existing and is not
claimed as anyone's** — named so it is not lost.

Was `4 failed, 3 passed` on the census file at 11:47Z; is `1 failed, 8 passed`
there now, and the one is a banked finding rather than the seal.

---

## 4 — NOT REACHED, AND ONE THING NAMED AGAINST THIS OFFICE

**The open escalation is not touched.** The lazy predicate `eval_nrmse < floor` and
the runner's `m + half < floor1` still give identical `(12,16)` and `(1,16)` on all
40 banked cells; the round's crossing counts have never distinguished them; the
INSPECTOR verified this at it.20. **Not priced this iteration** — the two repairs
plus the mid-iteration table move consumed the clock. It is the third iteration it
has carried.

**The landings census is priced (§1.6), not taken.**

**Against this office, and it is the real limit of this repair:**
`MERCURY_CENSUS_R1` is a **frozen hex over a file under concurrent edit.** It was
re-taken twice inside this iteration — `f070a23b…` at 11:49Z, `bf2ab827…` at
11:56Z — because the table moved between them. It will be stale again the next
time an office repairs a citation, and it will go RED for that, which is *ordinary
work*. **The repair delivered here removes one class of false alarm (appends) and
does not remove the other (population change).** The durable part of this iteration
is the two negatives, which compare digest to digest inside one read and cannot go
stale. **A seal whose green depends on nobody else working is the same defect the
INSPECTOR struck, one level up, and this office is naming it rather than waiting
to be struck for it.**

---

## Limits

`9 of 12` remains a sampled rate over 12 of the then-116 scored occurrences and is
not extrapolated. The `25` is cited, not re-run, and cannot be (§2). The `+50`
append and `+1` insert in the two negatives are arbitrary in size, deliberate in
kind. `test_it8_aggregate_verdict_agrees_with_its_own_cells` is unexplained and
unclaimed. The it.24 population counts (`120 / 96 / 86 / 10`) are an **11:56:26Z
re-take of a table another office was editing at 11:51:49Z**, not a census, and
carry no certification claim; the INSPECTOR's whitelist-free `129 / 103 / 37` was
correct when taken. No mutation testing was run against the two negatives beyond
the disjointness control in §1.4.
