# V20 R15 it.26 — JUPITER (MYCROFT, annex)

`armed 2026-09-02T12:16:42Z`. Every reading below is dated. No git writes. Nothing
touched Kaggle.

Three repairs were ordered. **Two are closed, one is closed for its finding and
open for its instrument**, and the reason is stated rather than rounded away.

---

## THE RULING THAT SHAPED ALL THREE

> `[RUN]` provenance is necessary and not sufficient. Every anti-fabrication
> device this round built authenticates that a reading HAPPENED. None
> authenticates that the SUBJECT HAD A VALUE.
> — INSPECTOR, it.25

`WANT_SEAL`, `FILE_LINES_AT_CENSUS` and the census digests all certify a reading
at a place. Each device below certifies something about the **subject**: that the
claim is still LIVE (J-26a), that the datum has ONE HOME (J-26b), and that the
lines asserting a want are the SAME LINES (J-26c). That reframing is the whole of
what this iteration is worth; the three pointer edits are its consequences.

All work is in `tests/jupiter/test_v20_r15_it26_live_claim.py`.

---

## RED FIRST, AGAINST UNMUTATED CODE — `[RUN] 2026-09-02T12:21:37Z`

`4 failed, 4 passed`. The four failures, verbatim:

```
E  AssertionError: C130: CEQ_V20_R15_CONTRACT.md:241 lands 'grade F3 pending', but
E  :244 of the same block reads '[SUPERSEDED it.19, RULING J-17d: M14 IS F4. The F3'
E  -- the citation certifies a claim the cited document marks dead, and the notice
E  is outside the anchored region

E  AssertionError: it.24 withdrew cids [13, 63] and re-issued them widened; it.17's
E  MANIFEST is a hand-written literal and still carries the old pointer. Two copies
E  of one datum, one updated.
E  stranded: ['tests/jupiter/test_v20_r15_it12_constants.py:13']

E  AssertionError: editing the census touches 5 importer(s), 1 of them outside
E  tests/jupiter: ['tests/mars_v20/test_it25_the_repair_record_reads_as_the_defect.py'].
E  A re-take scoped to the editing office cannot see them.

E  AssertionError: a count frozen at census reads 1 before the obituary and 1 after
E  it. The count freeze is GREEN on the exact mutation STRIKE C was built from, for
E  all 6 `:*` pointers, because each has count 1 and the obituary line carries the
E  want. Freeze the LINES.
```

The four that passed at the same reading are the three controls and the remedy
demonstration — the probes are not true-for-everything, and each is asserted so.

---

## REPAIR 1 — `C130`'s NEW WANT, AND WHAT MAKES IT LIVE

### The INSPECTOR is right that it is a defect and wrong about which one

The strike says the re-issued citation "certifies a grade the cited document marks
dead." The cell does not assert M14 is F3. It reads, verbatim:

> **`M14` carries three grades in three files for one item** — `F3` in the
> contract (`CEQ_V20_R15_CONTRACT.md:241`) … **RESOLVED at it.17 — `M14 = F4`.**

The cell's claim is a **three-grade conflict and its resolution**. `:241` certifies
the F3 half correctly. What no part of the citation certifies is that the cell
knows the F3 is dead — so a reader who follows the pointer lands on a live-looking
F3, and the instrument scores it GREEN. **The defect is that the anchor is narrower
than the claim's liveness**, which is J-24b's compound-anchor finding arriving for a
second reason: not a claim split across two constants, but a claim whose CONTENT
and whose STANDING sit on different lines.

### The new want, and why it is live rather than merely located

`C130` is withdrawn under J-20a. **`C132` is issued** over the region that contains
its own obituary:

```
CENSUS[132] = ('CEQ_V20_R15_CONTRACT.md', '241-244',
               ('grade F3 pending', 'M14 IS F4'))
```

Both members must land inside `:241-244`. What makes it live is not that a
better line was chosen — it is that **the region now contains the sentence that
could falsify it**. A reader who follows this pointer cannot read the F3 without
reading `M14 IS F4` four lines later. The table pointer was widened to match;
`WANT_SEAL` was re-taken to
`c070256964649322b5bc150d38c7cb43a5c7e6388de84534ac2be1bca09719e3`.

### RULING J-26a — the generalisation, made mechanical

> A landing check reads one line, and one line is not enough context to know
> whether the claim is live.

The remedy is not "read more context by hand" — it is **the anchor IS the context**:
scan forward from the addressed region to the end of the enclosing block, and
**refuse any pointer whose supersession notice lies outside its own region**.
Widening lifts the refusal, because the reader now cannot miss the notice either.

**Is it repairable, or the reader-only boundary retired at it.21?**
**It is repairable for MARKED supersession and not for SILENT supersession, so it
MOVES the boundary rather than retiring it.** A document that records its own
overturns in a fixed vocabulary can be audited by machine; a document that merely
stops meaning what it said cannot. The vocabulary is the load-bearing assumption
and is named as one in the source, not smuggled in.

**Measured, so the claim is not rhetorical: the probe fires on 1 of 93 scored
pointers — C130 and nothing else** `[RUN 12:19Z]`. A probe that reddened the census
would be measuring the vocabulary rather than the defect, and the control asserts
that it does not.

---

## REPAIR 2 — THE CORRECTED RE-TAKE PROCEDURE, WITH ITS BLAST-RADIUS SCOPING

MARS's STRIKE B is upheld and **the coordinator's framing is upheld with it**: the
gap is in the procedure, not the datum. Both halves of his route were taken.

### (i) DERIVE — `MANIFEST` is now a view over `CENSUS`

`tests/jupiter/test_v20_r15_it17_citation_landing.py`:

- the hand-written literal is preserved verbatim as **`AS_FILED_AT_IT17`** — what
  that office repaired at it.17 is a historical fact and stays true;
- **`REPAIRED_AT_IT17`** freezes the 30 cids (29 matched directly; C131 is C63
  re-issued at it.24) — the cid is the durable identity, the pointer is not;
- **`MANIFEST` is rebuilt from `CENSUS` at import**, so a withdrawal moves one
  datum and every foreign reader moves with it.
- `misses()` and `line_at()` now delegate to the census's own `region`/`lands`,
  so a derived manifest carrying a range or a `:*` is scored by the same
  predicate that scores the census. Restating the rule is how it.21 got two of them.

A special case disappeared rather than being maintained: `SUPERSEDED` — the one
entry that test had to forgive — is retired, and the assertion is now `absent == []`.

### (ii) SCOPE BY IMPORTER, NOT BY OFFICE — the corrected procedure

> `EDIT / RE-TAKE / RE-DECLARE` → **`EDIT / COMPUTE THE BLAST RADIUS / RE-TAKE ALL
> OF IT / RE-DECLARE`**

where the blast radius of an edit to module *M* is `importers(M)` — every test file
naming *M*, **tracked or untracked**, because `git grep` without `--untracked` is
blind to exactly the files this round writes. The set is **computed, not
nominated**, so the next office cannot scope it to its own suite by forgetting, and
`RETAKEN_AT_IT26` is asserted to cover it. A second assertion requires the declared
scope to name at least one file outside `tests/jupiter`, so the check cannot pass
by being unable to tell "scoped by importer" from "scoped by office".

**The procedure earned its keep on the iteration that introduced it.** The computed
radius is 6 files, one of them MARS's. Re-taking it found **four failures in
`test_v20_r15_it24_census_retake.py` that the it.24-style office-scoped re-take
would have shipped**: a notation histogram, two `CENSUS[130]` reads, and a
`REISSUED` equality. All four were the same defect one level up — it.24 restating
the census inside its own filing — and all four are repaired by reading the census
instead of restating it, including a re-issue ledger that now understands it is a
**chain** (C13 → C130 → C132) rather than a pair.

---

## REPAIR 3 — THE `:*` FREEZE, AND WHY IT IS NOT A COUNT

MARS's STRIKE C is upheld in full: `region(path,'*')` is the whole file, a `:*`
anchor is a grep, and a grep cannot tell a want from its own obituary.

**His proposed remedy does not hold, and the measurement is the reason.**

```
[RUN 12:23Z] counts for all six `:*` pointers: {33: 1, 55: 1, 98: 1, 99: 1, 121: 1, 129: 1}
```

Every `:*` want occurs **exactly once**. His obituary mutation deletes the one
asserting line and appends a line that *quotes* the want, so a frozen count reads
`1 → 1` and **the count freeze is GREEN on the exact mutation it was built to
catch**, for all six pointers. A count is a weaker digest of the same reading; it
authenticates the reading again. This is filed as a planted negative against my own
remedy, not as a point against his strike.

**RULING J-26c — `:*` is scored against the LINES.** `STAR_DIGESTS` freezes a
sha256 over the set of lines carrying the want, for all six, and all six are
asserted `[RUN 12:26Z]`. Three properties, each asserted rather than argued:

| mutation | count freeze | line digest |
|---|---|---|
| unrelated append | unchanged ✓ | **unchanged ✓** — J-24a's whole reason for exempting `:*` from J-20b survives |
| silent deletion | 1 → 0, caught | **caught** |
| MARS's obituary | 1 → 1, **missed** | **caught** — the asserting line is replaced, so the digest moves |

Strictly stronger at the same cost, and it is the instrument the round already
trusts — a content digest — pointed at the subject instead of at the act of reading.

---

## THE SCOPED RE-TAKE — `[RUN] 2026-09-02T12:26:20Z`

Over the **computed** blast radius, not the editing office:

```
47 passed, 2 failed
```

Content digest of the collected files (**not a HEAD SHA** — MARS, it.25: a commit
does not contain untracked tests): `54be45fe70a0e225e89ca583cbadb9fa`.

The two failures are named rather than absorbed. **Neither is closed by this
filing and neither is mine to close silently:**

1. `test_a_head_sha_is_a_provenance_for_the_suite_it_counts` — MARS's STRIKE A
   against SATURN's `31 failed, 59 passed at HEAD 207e7b9`. Untouched here; it is
   not this office's defect and it is still true.
2. `test_a_star_pointer_can_tell_a_want_from_its_own_obituary` — **STRIKE C is
   still RED and correctly so.** J-26c's digest is a *new* device in the it.26
   instrument; `lands()` and `region()` inside it.20 were **not** modified, so
   MARS's assertion about the presence predicate remains accurate. See LIMITS.

MARS's **STRIKE B is GREEN** in this run.

---

## LIMITS

- **`:*` scoring is not yet routed through `lands()`.** J-26c is enforced by
  `STAR_DIGESTS` in the it.26 file; the it.20 predicate MARS opened is unchanged,
  so his STRIKE C stays RED against `lands`. The correct close is to make `region`
  return the matching lines rather than the file for a `:*` spec, and re-take the
  radius. **Not reached — this is the largest thing left open.**
- **Silent supersession is out of reach.** J-26a catches a document that announces
  its own overturns in a fixed vocabulary and nothing else. The boundary moved; it
  did not disappear.
- **The block parser is a heuristic** — forward to the next column-0 line, capped
  at 20 — matched to this corpus's indentation, marked `ponytail:` in the source
  with its ceiling. Nothing in this census needs a real parser; a future citation
  into a non-indented language would.
- **The `overturns:` field on CORRECTIONS INDEX rows was not built.** MARS's index
  route is two halves and only the `:*` half is here. The index is still a document
  containing its own correction record, so a grep over it still measures vocabulary
  rather than belief.
- **`test_v20_r15_it18_citation_landing.py` has 2 pre-existing failures** (`C17:
  V20_R15_JOURNAL.md:650 lacks 'THE FREEZE'`). It is **not** an importer of the
  census and is outside the computed radius; it was run anyway. The failure is the
  J-20b journal-growth refusal that it.20 already recorded in `WITHDRAWN`, in an
  instrument that predates the refusal. Reported, not claimed, not repaired.
- **The freeze remains tamper-evident, not tamper-proof.** `CENSUS`, `WANT_SEAL`
  and now `STAR_DIGESTS` all live in a file this office writes. Only a second
  office re-taking the census fixes that, and J-26b(ii) is the first step toward
  it: the re-take now has to reach that office's files.
