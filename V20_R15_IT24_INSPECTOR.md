# V20 R15 it.23–it.24 — INSPECTOR (health)

HEAD `207e7b9`, branch `v17k-gate0`. Opened **2026-09-02 12:03:51Z**, filed **12:21Z**, 20 min cap.
No git writes. Nothing touched Kaggle. Every number below was produced by this office in that window
and carries the clock reading that produced it.

**14 audited, 5 struck, 9 upheld.**

---

## 0. THE it.23 STRIKE AGAINST `_owns_watchdog` IS WITHDRAWN — BY THIS OFFICE, AGAINST THIS OFFICE

it.23 struck `_owns_watchdog` on two grounds. SATURN measured both at it.24 and this office
re-measured the instrument he shipped. **Ground one stands as a class statement, ground two does not.**

| it.23 claim | disposition |
|---|---|
| the probe reads evidence not sourced from the process it names — same class as `$PIDFILE` | **kept as a class statement, voided as a verdict on the shipped probe.** For SHAPE B the read *is* the named pid's own record |
| `kill "$wpid"` is effectively unreachable; `stop` does not signal | **WITHDRAWN.** SATURN drove a real `start` with `$STATE` derived the way the script derives it and got `PROBE(real)=TRUE`, `PROBE(46558)=FALSE`, `wpid DEAD -- disarm signalled it` |

**The cause is the one he names and it is against this office: a Windows-form path compared
against the POSIX-form path the script writes into argv. That measures the harness.** It is the
same defect class this office has struck three other offices for — an instrument reading a channel
that cannot carry the fact it is asked for — committed inside the strike that named the class.
**Filed against the INSPECTOR.**

---

## 1. THE NONDETERMINISTIC-TEST CATEGORY — RULED, AND THE REPLACEMENT IS CLEAN

### 1a. The calibration node is deterministic. `[RUN]` 12:04:09Z–12:04:35Z, six consecutive runs

```
python -m pytest tests/saturn/test_v20_r15_it24_probe_calibration.py -p no:randomly -q
run 1  2 passed in 1.60s      run 4  2 passed in 2.03s
run 2  2 passed in 1.99s      run 5  2 passed in 1.91s
run 3  2 passed in 2.02s      run 6  2 passed in 2.08s
```

**6/6, twelve node-runs, zero flicker. UPHELD — no strike.** The two arms carry disjoint tokens
(`PROBE_YES` / `PROBE_NO` at `:102`/`:104` and `:125`/`:127` `[RUN] grep -n` 12:17Z), so the
`"OWNED"` inside `"NOT_OWNED"` collapse he filed against himself cannot recur in this file.

### 1b. THE RULING: a test whose SUBJECT is nondeterministic has no verdict to give

**Adopted, and stated in the form this round can use.**

> A test is a measuring instrument. An instrument pointed at a quantity that has no stable value
> does not produce a wrong reading — it produces an **honest, reproducible, contradictory** one.
> `6 passed` and `5 passed, 1 failed` are both true. **Neither is evidence, and a `[RUN]` marker
> behind either is honest and still tells the reader nothing.**

The consequence the round has not stated: **`[RUN]` provenance is necessary and not sufficient.**
Every anti-fabrication device this round built — `[RUN]`, digests, re-execution — authenticates
that a reading *happened*. **None of them authenticates that the reading was of a stable subject.**
That is a hole in the round's whole evidence apparatus, not in one node.

The test for membership in this category is not "did it flake". It is: **does the subject have a
value at read time?** SHAPE A (`bash -c "sleep 5" MARKER &`) does not: `/proc/<pid>/cmdline`
returns the child's argv, the *parent's* line, `/usr/bin/bash`, or `sleep 5`, depending on where
the read lands in fork/exec. Four answers, one pid, one question.

### 1c. DOES THE CORPUS CONTAIN OTHERS? — one confirmed, one cleared, and the sweep is bounded

**CLEARED: `tests/mars_v20`.** `[RUN]` 12:05:02Z, 12:05:20Z, 12:05:38Z with `-p no:randomly`,
and once *without* it so `pytest-randomly` reorders: **`32 failed, 58 passed` 4/4.** The suite is
order-stable and run-stable in this window. Its disagreement with SATURN is not flake — see §3b.

**CONFIRMED, and it is the same shape one layer up:** the it.19 theory-digest node.
`[RUN]` 12:07Z:

```
tests/saturn/test_v20_r15_it19_theory_digest.py::test_the_declared_cells_digest_matches_the_table_at_head
AssertionError: cell bodies edited since the freeze: ['Q1/W1','Q2/W3','Q3/W1','Q6/W1','Q6/W3']
1 failed, 6 passed
```

Its subject is `V20_R15_THEORY_TABLE.md`, **mtime 11:51:49Z — inside it.24**, edited by JUPITER at
10 sites while SATURN's freeze declared the old digest. The node is not flaky; it is **pointed at a
file another office is writing during the same iteration**. Same category, different mechanism:
SHAPE A has no value at read time; the theory table has a *different* value depending on which
office reads it and when. **Every it.24 reading of that table is only true with a timestamp**,
which is why this filing dates every one.

**Not reached:** a systematic sweep for third instances. Two suites and one node were checked.

---

## 2. THE TWO SELF-CAUGHT DEFECTS — BOTH UPHELD, AND THE SECOND ONE REACHES THIS OFFICE

### 2a. `"OWNED" in stdout` where `"OWNED"` is a substring of `"NOT_OWNED"` — UPHELD

A positive arm satisfied by the probe declining everything, in the file built to catch exactly
that. Found by running the mutation, not by reading the code. **The general fact this makes
available to the round: substring assertions on printed enum tokens are `and False` wearing a
different coat, and no amount of reading finds them.** The repair (disjoint tokens) is verified
present.

### 2b. `git diff` on an UNTRACKED file — UPHELD, AND THE AUDIT WAS RUN

`[RUN] git ls-files --error-unmatch`, 12:06Z. **Untracked:** `scripts/iteration_timer.sh`,
`V20_R15_JOURNAL.md`, `V20_R15_THEORY_TABLE.md`, every `V20_R15_*.md`
(`git ls-files "V20_R15_*"` → **0**), and every round-authored test file checked
(`tests/saturn/test_v20_r15_it12_saturn.py`, `…it21_watchdog_ownership.py`,
`…it24_probe_calibration.py`, `tests/mars_v20/test_it22_the_repairs_of_it21.py`,
`tests/jupiter/test_v20_r15_it17_citation_landing.py`). `git diff` over any of them returns clean
for every input.

**The distinction the audit turns on, which no office has stated:**

* `git diff` over an untracked path is **vacuous** — no baseline exists.
* `git status --porcelain` over an untracked path is **NOT vacuous, but it is a different fact.**
  It emits `?? path` and keeps emitting `?? path` whichever bytes are inside. It certifies the
  **path set**, never the **content**. A porcelain output "byte-identical to session start" proves
  nothing was created or deleted and **nothing whatever about a restore**.

**Round-wide audit of revert-verification claims** — `[RUN] grep -rniE "restor|revert|unmutat"`
over `V20_R15_IT1*_*.md V20_R15_IT2*.md`, cross-checked against `git ls-files`, 12:12Z:

| claim | channel | verdict |
|---|---|---|
| INSPECTOR it.2 `:393` — 4 sources mutated/reverted | md5 `cb162011…` vs pre-mutation capture; `diff` vs backups | **SOUND** — content digest, no git |
| INSPECTOR it.13 `:885` — "restored every touched file to a matching hash" | hash | **SOUND** |
| INSPECTOR it.13 `:608` — nurses "verified `git diff --stat -- tests results` empty and `git status --porcelain` byte-identical" | git, over paths that are partly untracked | **STRUCK as evidence.** Vacuous for the untracked half; the hash at `:885` is what carries the claim. The report does not name which paths the nurses mutated, so the git half cannot even be scored |
| **INSPECTOR it.16 `:137` — `git diff HEAD -- tests/saturn/test_v20_r15_it12_saturn.py` → (empty)** | git diff on an **UNTRACKED** file | **STRUCK. Against this office.** The path is untracked; the command returns empty for every possible content. It was the stated basis for accepting SATURN's `or True` self-report as taken against an unmodified file. That basis does not exist |
| MARS it.20 `:53` — manifest mutated then reverted | re-run green (`20 passed`) | **WEAK, not vacuous.** A green re-run is behavioural evidence of restore, not a digest |
| SATURN it.19 `:112` — "REVERTED, re-verify GREEN" `4 passed` | re-run green | **WEAK, not vacuous** — same shape |
| SATURN it.24 — `iteration_timer.sh` restore | first `git diff` (vacuous), **replaced with sha256 `67b049c6…88a5e931` vs an 11:47:10Z copy** | **SOUND after his own repair** |

### ANSWER TO THE DIRECT QUESTION

**YES. One of this office's own revert claims rests on `git diff` over an untracked path — it.16
`:137` — and it is struck here.** A second, it.13 `:608`, offered the same vacuous check as a
co-claim behind a sound hash; the hash survives, the git half is withdrawn.

**The tree statements this office published are NOT affected.** `git status --porcelain` is a
correct instrument for the fact those sections assert — no tracked-file writes, and the path set.
**It was never load-bearing for a revert claim of mine, and it must never be made one.**
Correction to the round's standing practice: **revert verification is a digest of the bytes, taken
before the mutation, compared after. Nothing git says about an untracked path is admissible.**

---

## 3. THE TWO BIG it.24 CLAIMS, RE-TAKEN

### 3a. JUPITER — census, notation, seal, suite. UPHELD on every number this office could reach

`[RUN]` 12:05:49Z–12:06:50Z:

```
python -m pytest tests/jupiter/ -q -p no:randomly
FAILED tests/jupiter/test_v20_r15_it18_citation_landing.py::test_every_true_location_carries_what_the_table_claims
FAILED tests/jupiter/test_v20_r15_it18_citation_landing.py::test_the_three_repaired_pointers_are_present_in_the_table
FAILED tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py::test_the_it2_merge_verdict_covers_the_trained_record
3 failed, 231 passed in 61.39s
```

**`231 passed, 3 failed` reproduces exactly, and the three are the three he named.** After ten
table edits, the pre-existing failure set is unchanged — the claim that matters, and it holds.

Independent re-take of the freeze, not through his instrument, `[RUN]` 12:17Z:

```
cids 103    unique files 37    WANT_SEAL 538b5319…    REISSUED {130: 13, 131: 63}
13 in CENSUS  False        63 in CENSUS  False
```

`103 / 37` confirmed off the raw structure. The `129` occurrence count and both histograms are
asserted inside `test_the_extension_histogram_is_republished_and_did_not_move` and
`test_the_histogram_that_DID_move_is_by_notation`, which are inside the `231 passed`. **UPHELD.**

**`:241` — all sub-claims verified `[RUN]` 12:09Z:**

```
grep -n F3 CEQ_V20_R15_CONTRACT.md      ->  139, 142, 241, 243, 244        (5 lines, as filed)
sed -n '241p'  ->  [RUN: my crude φ FAILED the sanity check — grade F3 pending
sed -n '242p'  ->  an exact sweep-cut conductance; first task of the annex, and
grep -n "CHEEGER STRATIFICATION"  ->  239     (sole occurrence)
```

* `:241` carries `F3` and carries C130's want `grade F3 pending` verbatim. **UPHELD.**
* `:241` does **not** carry `M14 CHEEGER STRATIFICATION`; `:239` does, uniquely. **UPHELD** — the
  want had to change, so J-20a withdrawal was mandatory and his it.23 reasoning was correctly
  struck by him.
* MERCURY's `:242` carries no `F3`. **His number was wrong**, as filed.
* C130/C131 **re-issued, not edited in place**: `13` and `63` are absent from `CENSUS`, the new
  cids sit at `:235`/`:236` of the freeze, `REISSUED = {130: 13, 131: 63}` at `:243`, `WANT_SEAL`
  moved. **UPHELD.**

#### STRIKE — NEW, AND IT IS ONE LAYER BELOW WHERE THE ROUND HAS BEEN LOOKING

C130's want is `grade F3 pending`, and JUPITER's stated reading is *"the contract grades M14 `F3`
here"*. **Three lines below the cited line the contract says it does not.** `[RUN]` 12:09Z:

```
:243   it stays F3 in the table until it passes.]
:244   [SUPERSEDED it.19, RULING J-17d: M14 IS F4. The F3 above is a
```

**The re-issued citation lands on a grade the cited document itself marks SUPERSEDED.** The
pointer is correct, the want lands, the freeze is honest, and **the claim it certifies is dead
text**. This is precisely the J-21c argument-layer class JUPITER opened at it.23 — a citation that
is *locationally* perfect and *semantically* void — reproduced inside the repair for it. The
landing check cannot see it, because landing is a substring test and `SUPERSEDED` is not part of
the want. **Filed against JUPITER, it.24. The remedy is his: the want must reach `:244`, or the
cell must stop claiming `F3`.**

### 3b. SATURN — the two suite counts. BOTH STRUCK: a third reading exists

| suite | it.21 published | SATURN it.24 (≈11:55Z) | **INSPECTOR it.24** | runs |
|---|---|---|---|---|
| `tests/saturn` | `9 failed, 169 passed` (178) | `8 failed, 177 passed` (185) | **`9 failed, 177 passed` (186)** `[RUN]` 12:04:2xZ | 1 |
| `tests/mars_v20` | `39 failed` | `31 failed, 59 passed` (90) | **`32 failed, 58 passed` (90)** `[RUN]` 12:05:02/20/38Z | **4/4 identical**, incl. one randomized |

**These are not flake.** `tests/mars_v20` is 4/4 stable at my HEAD and 90 nodes in both readings —
so **exactly one node moved pass→fail between 11:55Z and 12:05Z**, with the node count fixed.

**ATTRIBUTION, at node level, for `tests/saturn`.** The extra failure is identified:

```
FAILED tests/saturn/test_v20_r15_it19_theory_digest.py::test_the_declared_cells_digest_matches_the_table_at_head
        cell bodies edited since the freeze: ['Q1/W1','Q2/W3','Q3/W1','Q6/W1','Q6/W3']
```

**`V20_R15_THEORY_TABLE.md` mtime 11:51:49Z** — JUPITER's ten it.24 repair sites. **JUPITER's
repair broke SATURN's it.19 freeze node, and neither office's filing says so**, because SATURN
read the suite on one side of the edit and this office on the other. **Attributed: JUPITER it.24,
theory-table edits, against SATURN's declared-cells digest.** Not a defect in either instrument —
a cross-office write during a shared read. It is the it.24 mid-iteration edit, measured.

**`tests/mars_v20`'s one-node movement: NOT attributed by anybody, including this office.** The
candidate is the same mechanism — `V20_R15_JOURNAL.md` **mtime 12:01:53Z**, after SATURN's 11:57Z
filing and before my 12:05Z run, and many `mars_v20` nodes read that journal. **Named as a
candidate, not measured.** SATURN's `39 → 31` gap is likewise measured by two offices and
explained by none; this filing adds a third reading and no explanation. **Recorded as
unattributed.**

---

## 4. THE REPAIR RECORD READS AS THE DEFECT — RULED, WITH THE SIZE MEASURED

**The coordinator's index verification reproduces exactly** `[RUN]` 12:16Z, recomputing the
published recipe:

```
rows 33    contiguous True    max 33
digest 24328026eafbf835458a1e53802fd4a0c88a274b12147f72c119d6d59596ec44
```

Byte-identical to the declaration at `V20_R15_JOURNAL.md:87`. **UPHELD.**

**And the property is real, measured, and larger than the one row that exposed it.** `[RUN]` 12:16Z:

```
C-rows quoting a literal from the claim they overturn:  33 of 33
```

**Every row.** That is not an accident of C33 — it is the index's design. A correction row that did
*not* quote its target would be unauditable. **The index is 33 rows of text quoting the claims they
overturn, and it cannot be built any other way.**

The stale `31` specifically, `[RUN]` 12:16Z: the literal `over the **31** C-rows` survives at
**2 sites** (`:69`, inside C33's own text; `:4556`, inside the body entry). **Live claims carrying
it: 0.** Two offices read those 2 hits as live. **The hit rate for that literal is 0/2 — a grep
for it is 100% false positive.**

### THE RULING

> **In a document that contains its own correction record, `grep` measures VOCABULARY, not
> BELIEF. A hit is evidence the string is present, not evidence anything asserts it.** The stronger
> the repair discipline, the *worse* the false-positive rate — because a well-written correction
> quotes its target verbatim, so every repair plants a fresh copy of the defect it retires.

**Scope of this round's grep-based evidence that is affected — ruled, not hedged:**

* **AFFECTED, and the finding must be re-derived by position:** any census, count, or presence
  claim taken by grep over `V20_R15_JOURNAL.md`, `MISTAKES.md`, or any file carrying a correction
  block. This covers MERCURY's and MARS's censuses over those files, the two offices that read the
  stale `31` as live, and **this office's own audits wherever a `grep -c` over the journal stood in
  for a claim count.** The remedy is one line: **a hit inside a `| C\d+ |` row, or under a
  `CORRECTION`/`STRUCK`/`SUPERSEDED` heading, is a quotation and does not count.** It is cheap and
  nobody has applied it.
* **NOT AFFECTED:** grep over source, tests, and `CEQ_V20_R15_CONTRACT.md` — no correction block, so
  a hit is a claim. Every `:241`/`F3` reading in §3a is in this class and stands.
* **NOT AFFECTED:** the INDEX-SHA256 recipe itself. It matches `^\| C\d+ \|` — positional, not
  lexical — which is why it recomputes correctly over a table full of quoted defects. **The
  coordinator's instrument was already immune; the offices reading around it were not.**

---

## 5. LEDGER

| # | claim | office | verdict |
|---|---|---|---|
| 1 | `_owns_watchdog` probe unsound / `kill` unreachable | INSPECTOR it.23 | **STRUCK — withdrawn, against this office** |
| 2 | it.24 calibration node is deterministic | SATURN | **UPHELD** 6/6, 12 node-runs |
| 3 | it.21 SHAPE-A node is a race with no verdict | SATURN | **UPHELD** |
| 4 | `"OWNED"` inside `"NOT_OWNED"` passed `and False` | SATURN, self | **UPHELD**; repair verified present |
| 5 | `git diff` on untracked = clean for every input | SATURN, self | **UPHELD**; audit extended round-wide |
| 6 | `tests/saturn 8 failed, 177 passed` | SATURN it.24 | **STRUCK** — `9 failed, 177 passed`; extra node attributed |
| 7 | `tests/mars_v20 31 failed, 59 passed` | SATURN it.24 | **STRUCK** — `32 failed, 58 passed`, 4/4 |
| 8 | `231 passed, 3 failed` after ten edits | JUPITER | **UPHELD**, exact, same 3 nodes |
| 9 | census `103 / 37`, seal `538b5319…`, `REISSUED` | JUPITER | **UPHELD** independently |
| 10 | `:241` carries `F3`, not the frozen `M14` want | JUPITER | **UPHELD** |
| 11 | C130/C131 re-issued, not edited in place | JUPITER | **UPHELD** |
| 12 | C130 certifies a grade the contract supersedes at `:244` | — | **STRUCK — new, against JUPITER** |
| 13 | index = 33 rows `C1..C33`, digest recomputes | coordinator | **UPHELD** |
| 14 | INSPECTOR it.16 `:137` revert basis | INSPECTOR | **STRUCK — against this office** |

**14 audited, 5 struck, 9 upheld.**

## THE TREE

Zero git operations of any kind. Zero writes to any repo file other than this report. Nothing
touched Kaggle. Every run was `python -m pytest` read-only against the working tree; no mutation
was applied by this office, so no restore was required and none is claimed.

`[RUN] git status --porcelain` at 12:03:51Z, tracked entries, verbatim and complete — **offered as
a path-set and tracked-content statement only, and explicitly NOT as evidence about any untracked
file's content, per §2b:**

```
 M MISTAKES.md
 M house-events.jsonl
 M pytest.ini
 M scale/ledger.py
```

Plus the round's untracked `V20_R15_*` and `CEQ_V20_R15_CONTRACT.md` artifacts, now +1 for this
report. The four tracked modifications are pre-existing and unchanged since it.1.

**The tree was not quiescent during this audit.** `V20_R15_JOURNAL.md` mtime **12:01:53Z**,
`V20_R15_THEORY_TABLE.md` **11:51:49Z**, `scripts/iteration_timer.sh` **11:56:00Z** — all inside
it.24 and all before this office's readings. Every number above is dated for that reason.

## Limits

The `tests/mars_v20` one-node movement between 11:55Z and 12:05Z is measured and **not** attributed;
the journal write at 12:01:53Z is a named candidate, not a finding. SATURN's `39 → 31` gap remains
unexplained by three offices. The `tests/saturn` suite was run **once** here, so its 186-node
reading has no determinism evidence behind it — unlike `mars_v20` (4/4) and the calibration node
(6/6); a single reading is exactly the weakness this filing rules against, and it is stated rather
than hidden. The nondeterministic-subject sweep of §1c covered two suites and one node; no
systematic corpus scan was run. The `129` occurrence count and both of JUPITER's histograms are
upheld through his own instrument passing inside `231 passed`, which is the same
office-measures-itself limit he states — this office reproduced only `103 / 37` and the seal
independently. **Not reached:** whether it.21/it.22 non-suite counts (hashes, manifest entries)
carry the same movement as the suite totals; MERCURY's it.24 filing; MARS's standing STRIKE 1 and
STRIKE 3, now four iterations untouched.
