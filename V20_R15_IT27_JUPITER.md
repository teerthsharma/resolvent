# V20 R15 it.27 — JUPITER (MYCROFT, annex)

`armed 2026-09-02T12:32:30Z`. Every reading below is dated. No git writes. Nothing
touched Kaggle.

Three repairs were ordered. **Two are closed and one is closed to the boundary the
coordinator set for it** — the `overturns:` field is specified and its enforcement
node ships RED, because the index is not this office's file to edit.

---

## REPAIR 1 — `J-26c` IS NOW IN THE PRODUCTION PREDICATE, AND MARS'S STRIKE C IS GREEN

### RED FIRST, AGAINST UNMUTATED CODE — `[RUN] 2026-09-02T12:37:06Z`

`tests/jupiter/test_v20_r15_it27_star_lands_and_overturns.py`, `3 failed, 5 passed`,
against `lands()` exactly as it.26 left it. The failure that names the largest thing
open, verbatim:

```
E  AssertionError: `lands()` still LANDS on a file where the want was deleted and
E  only the line recording its removal remains, for 6 of 6 `:*` pointers. J-26c
E  froze the carrying lines in the it.26 instrument and left the PRODUCTION
E  predicate unmodified, so every office that imports `lands` still greps.
E  missed: ['C33: tests/jupiter/test_v20_r15_it6_q1_exact_class.py:*',
E  'C55: ceq/arm_pl.py:*', 'C98: tests/jupiter/test_v20_r15_it9_q6.py:*',
E  'C99: tests/jupiter/test_v20_r15_it11_q6_oracle.py:*',
E  'C121: lean/lakefile.lean:*',
E  'C129: tests/jupiter/test_v20_r15_it14_theory_table.py:*']
```

**6 of 6, not 1 of 6.** The it.26 filing said STRIKE C was still RED; it did not say
the defect was live on every `:*` pointer in the census, because the it.26 device was
only ever asserted for C55 in that shape. The other two failures are the missing
`STAR_DIGESTS` import and the `overturns:` node (REPAIR 3).

### RULING J-27a — `:*` IS SCORED AGAINST THE LINES THAT CARRY THE WANT, IN `lands()`

`STAR_DIGESTS` and `carrying_digest()` now live in
`tests/jupiter/test_v20_r15_it20_citation_freeze.py`, and `lands()` reads them:

```python
def lands(want: str, line: str) -> bool:
    if isinstance(want, tuple):
        return all(lands(w, line) for w in want)
    if want in STAR_DIGESTS:
        return carrying_digest(want, line) == STAR_DIGESTS[want]
    return want in line
```

**Keyed on the WANT, not on the cid, and that is a decision rather than a
convenience.** `lands(want, line)` is the two-argument signature four offices already
call — MARS's it.25 node calls it positionally. Widening it to take a cid would have
restated the rule in every caller, which is precisely how it.21 shipped two copies of
one rule and how it.20's INSPECTOR strike arose. The cost is that the key must be
unique, so uniqueness is **asserted, not assumed**:
`test_the_star_wants_are_unique_so_keying_on_them_is_sound` checks both that the six
`:*` wants are distinct and that none of them is also a line-pointer want.

### MARS'S IT.25 NODE, HIS FILE UNMODIFIED — `[RUN] 2026-09-02T12:37:43Z`

```
tests/mars_v20/test_it25_the_repair_record_reads_as_the_defect.py
  ::test_the_control_a_star_anchor_breaks_on_a_silent_deletion    PASSED
  ::test_a_star_pointer_can_tell_a_want_from_its_own_obituary     PASSED   <- STRIKE C
  ::test_the_control_most_of_it17s_manifest_is_still_censused     PASSED
  ::test_a_withdrawal_does_not_strand_a_pointer_another_office_reads PASSED <- STRIKE B
  ::test_the_control_a_git_sha_DOES_bind_a_tracked_suite          PASSED
  ::test_a_head_sha_is_a_provenance_for_the_suite_it_counts       FAILED   <- STRIKE A
```

No line of `tests/mars_v20/` was written by this office. `git status` cannot
corroborate that — the whole directory is `?? tests/mars_v20/`, which is STRIKE A
itself — so the evidence is the re-take above, run against his file as it stands. This is the
it.23 standard met a second time. **STRIKE A stays RED and is still true** — it is
SATURN's HEAD-SHA provenance defect, not this office's, and nothing here closes it.

### WHAT THE DEVICE AUTHENTICATES, AGAINST THE INSPECTOR'S IT.25 RULING

> `[RUN]` provenance is necessary and not sufficient. … None authenticates that the
> SUBJECT HAD A VALUE.

**A `:*` digest freeze authenticates the subject, and it is the first device this
round has built that does.** `WANT_SEAL` seals what this office *claims* the want is.
`FILE_LINES_AT_CENSUS` seals the file's *shape*. Both certify the reading. The line
digest is taken over the bytes of the lines in the cited file that carry the claim, so
it moves when the claim's own text moves and stays still when anything else in that
file does — J-24a's append exemption survives, asserted in
`test_the_control_a_star_anchor_lands_unmutated_and_survives_an_append`.

**What it still cannot see, stated rather than left to be found:**

1. **A line that keeps its bytes and loses its meaning.** Same silent-supersession
   boundary J-26a moved and did not retire.
2. **A want whose carrying line was already wrong when the census froze it.** The
   digest certifies *unchanged since census*, never *correct at census*. The census is
   still this office's hand-taken reading, and only a second office re-taking it fixes
   that.
3. **Which of several carrying lines is the one meant**, if a want ever acquires a
   second occurrence. The digest then covers both and moves if either moves — stricter
   than intended, not weaker, but no longer addressing one line.

### THE COUNT ROUTE, RE-MEASURED AND STILL DEAD

MARS's proposed remedy was a frozen occurrence count. `[RUN 12:37:06Z]` all six `:*`
wants occur exactly once, and the obituary line quotes the want, so the count reads
`1 → 1` on the exact mutation STRIKE C was built from. Asserted as a planted negative
against this office's own device in
`test_a_frozen_COUNT_would_be_GREEN_on_the_obituary_for_all_six`, with an escape
clause: if a `:*` want ever acquires a second occurrence the node goes RED and the
count route becomes viable for that pointer alone.

---

## REPAIR 2 — MERCURY'S THREE REPLACEMENTS, APPLIED UNDER THE CORRECTED PROCEDURE

All three applied verbatim as he specified them. No `want` was edited, so J-20a
required no withdrawal: **`M-25a` is a POINTER repair and `M-25b`/`M-25c` each ADD a
citation to a clause rather than moving one.** That distinction is his ruling — *a
registry entry and its artefact are two citations* — and it is the reason the
population moves rather than the seal alone.

| ruling | table | edit | census |
|---|---|---|---|
| `M-25a` | `:97` | `scripts/v15_r1.py:547` → `:547-558` | `CENSUS[24]` spec widened, want `'--seeds'` untouched |
| `M-25b` | `:71-72` | clause gains `T_STAR = 2` @ `scripts/v15_r1.py:138` | **`C133` issued** — `('scripts/v15_r1.py', 138, 'T_STAR = 2')` |
| `M-25c` | `:306` | clause gains `build_delay` @ `ceq/beds/bed_k.py:236`, `:475` **keeps** its place as *registered @* | **`C134` issued** — `('ceq/beds/bed_k.py', 236, 'def build_delay(')` |

The two re-cites at `:198`/`:206` were **not touched**, per his ruling that they cite
the formula only and are correct as they stand. `WANT_SEAL` re-taken to
`abad6a77d9da0793c54d19f092a30005cb3daa40aa0dad77d8d44ce5e01212c4`;
`FILE_LINES_AT_CENSUS` gains `ceq/beds/bed_k.py: 397`.

### COVERAGE AFTER THE THREE EDITS — `[RUN] 2026-09-02T12:42:22Z`

**`118 of 131` occurrences SCORED and landing; `13 of 131` REFUSED under J-20b;
`0 of 118` failing.** `105 of 105` unique pointers censused, across **38** files.
`not_landing(CENSUS) == []` and `not_a_repair(CENSUS) == []`.

| extension | it.24 | it.27 | | notation | it.24 | it.27 |
|---|---|---|---|---|---|---|
| `.md` | 40 | 40 | | plain `:N` | 119 | **120** |
| `.py` | 69 | **71** | | range `:A-B` | 2 | **3** |
| `.lean` | 19 | 19 | | file `:*` | 8 | 8 |
| `.jsonl` | 1 | 1 | | | | |
| **total** | 129 | **131** | | **total** | 129 | **131** |

Both histograms moved and both movements are accounted for exactly: `.py +2` is
`scripts/v15_r1.py:138` and `ceq/beds/bed_k.py:236`; plain `+1` is those two additions
less `:547`, which was promoted into the range column by `M-25a`.

### THE BLAST RADIUS, COMPUTED — `[RUN] 2026-09-02T12:42:22Z`

Two things were edited: the census module and **`V20_R15_THEORY_TABLE.md` itself**.
The radius of a table edit is far wider than the radius of a census edit, and the
procedure surfaced that rather than this office guessing it:

```
19 files, 4 offices  --  jupiter 10, mercury 4, saturn 3, mars_v20 2
11 failed, 141 passed
```

Content digest of the collected files (**not a HEAD SHA**): `499dce1ae831da7b6ae3199715f1a002`.

**The procedure earned its keep for a second time.** Re-taking the computed radius
found **six failures inside JUPITER files that an office-scoped re-take of
`tests/jupiter` alone would still have caught, and five outside it that it would
not.** All six of the internal ones were the same defect it.24 shipped and it.26
named: an instrument restating the census instead of reading it — `129`, `103`, `37`
files and two histograms hard-coded in `test_v20_r15_it24_census_retake.py` and
`test_v20_r15_it21_heading_anchor.py`. Each is now read from `POPULATION_AT_IT20` /
`UNIQUE_AT_IT20` or re-declared with the movement annotated in the source.

**One repair was found only because `lands()` changed under it.** it.26's count
negative counted lines with `lands(want, ln)` as a presence test. Once `lands` became
digest-scored for `:*` wants, that count read `1 → 0` and the negative asserted the
wrong thing. It now uses `want in ln` explicitly, with the reason in the source:
scoring MARS's COUNT route through the predicate that retires it measures the repair,
not the route. **A caller that used the shared predicate for a purpose it no longer
serves is exactly what an office-scoped re-take ships.**

### THE FIVE FAILURES OUTSIDE THIS OFFICE, NAMED AND NOT ABSORBED

**None is repaired here and none is silently carried.** Three are foreign freezes
doing their job against a table this office moved; two are not mine at all.

1. `tests/saturn/test_v20_r15_it19_theory_digest.py::test_the_declared_cells_digest_matches_the_table_at_head`
   — **his freeze firing correctly.** Five cell bodies moved: `Q1/W1, Q2/W3, Q3/W1,
   Q6/W1, Q6/W3`. His digest is his to re-take; the movement is MERCURY's three
   rulings, applied.
2. `tests/mercury/test_v20_r15_it22_independent_census.py::test_mercury_census_digest_r1`
   — the same, for his independent census digest.
3. `tests/mars_v20/test_it20_the_census_still_certifies_its_own_coverage.py::test_the_landing_instrument_opens_most_of_the_census`
   — his coverage node reads the table population.
4. `tests/mercury/test_v20_r15_it27_uncited_class.py` — **three nodes from MERCURY's
   live it.27**, on §4 `BED_SPECS`, the `none` arm, and §3.2's Fisher `p`. Not this
   office's edits, and not attributed to them.
5. `tests/mars_v20/...::test_a_head_sha_is_a_provenance_for_the_suite_it_counts` —
   STRIKE A, still true, still SATURN's.

---

## REPAIR 3 — THE `overturns:` FIELD, SPECIFIED AND ENFORCED WITHOUT TOUCHING THE INDEX

**`V20_R15_JOURNAL.md` was not edited.** The field is specified in
`test_v20_r15_it27_star_lands_and_overturns.py` and the node that enforces it ships,
so the coordinator's row change lands against a test that already exists.

**RULING J-27b — THE FIELD.** One trailing cell per CORRECTIONS INDEX row:

```
| C1 | "no arm crosses ..." | it.1 | **6 of 24 cross.** | it.2 | overturns: `no arm crosses`
```

The literals are the **search keys a later reader would grep for**, not a prose
restatement of the correction. A row with no machine-checkable literal declares
`overturns: none` — a claim the round can audit. **An absent field is not a claim, and
that is the whole reason the field must be mandatory rather than optional.**

**What a grep-based instrument must do to honour it**, shipped as `live_hits()`:

1. read the field off every index row — `dead_literals()`;
2. return **nothing** for a literal any row declares dead;
3. never count a hit that is itself an index row.

**(3) alone is not sufficient, and that is the finding.** Excluding the index from the
grep is the obvious repair and it is wrong: a body entry may restate the withdrawn
claim in its own words, and no positional rule reaches it. Only the declaration does.

**THE MEASUREMENT, re-taken — `[RUN] 12:42:22Z`.** `34 of 34` index rows quote a
literal from the claim they overturn (the INSPECTOR measured `33 of 33`; the index has
since grown by one row and the property held across it). The round's designated
guard against stale claims is the densest source of stale-claim false positives in the
repository.

`test_an_overturns_aware_grep_returns_nothing_for_a_declared_dead_literal` is **RED
and is meant to be**, verbatim:

```
E  AssertionError: no CORRECTIONS INDEX row carries an `overturns:` field, so a
E  stale-claim grep has nothing to subtract and every one of its hits on the index
E  is a false positive. Add the trailing cell specified above OVERTURNS_RE --
E  `overturns: `literal`, `literal`` -- starting with C1's `no arm crosses`.
```

Once a row carries the field, the node also checks the declaration is honest: every
literal a row declares dead must have **zero** live hits in the journal body outside
the index. A row cannot retire a claim the body still asserts.

---

## LIMITS

- **The `:*` digest certifies unchanged-since-census, never correct-at-census.** It
  authenticates the subject against drift and not against a bad first reading. The
  census remains one office's hand-taken reading sealed in a file that office writes;
  only a second office re-taking it closes that, and nothing this iteration built
  moves it.
- **`lands()` is now keyed on want text.** A future census that gives two `:*`
  citations the same want silently scores one against the other's digest. The premise
  node asserts against it today; it is an invariant the census must preserve, and it
  is stated here so the next office does not learn it from a green suite.
- **`tests/jupiter/test_v20_r15_it18_citation_landing.py` has 3 failures**, one more
  than at it.26: its dated `129` expired when the population moved to `131`. That is
  J-18a working on a retired instrument — it.20 supersedes its landing check entirely
  — and it is **reported, not repaired**. Repairing a superseded instrument's dated
  total would be restating the census in a third place.
- **`tests/jupiter/test_v20_r15_it4_merge_is_unexercised.py` has 1 failure**, outside
  the radius, pre-existing, run anyway. Not this office's and not claimed.
- **The three foreign digests are RED and stay RED.** SATURN's cell digest, MERCURY's
  census digest and MARS's coverage node all read a table this office moved under
  MERCURY's own rulings. Re-taking another office's freeze is the thing the freeze
  exists to prevent; each is named above with what moved and why.
- **The `overturns:` field is specified, not adopted.** No row carries it, so the
  measurement stands and the remedy does not. The node is the whole of what this
  office can ship without editing the coordinator's file.

---

## NOT REACHED

- **The `overturns:` field on the rows themselves.** Coordinator's file, by the brief.
- **A second office's re-take of the census.** Named as the only fix for
  tamper-evidence at it.20, it.26 and again here; still nobody's iteration.
- **Whether MERCURY's three it.27 failures interact with these table edits.** They are
  in the radius and they are his live iteration; separating cause would need his
  filing, which is not written yet.
- **SATURN's it.27 `test_v20_r15_it27_subject_provenance.py`** appeared in the tree
  mid-iteration and left the radius before the final take. Two of its nodes were RED
  at `12:40:16Z` and it is not in the `12:42:22Z` collection. **The tree moved under
  this census during the iteration, for the fourth measured time this round** — J-20b's
  original finding, now happening to test files rather than to cited prose.
