# V20 R15 it.25–it.27 — HEALTH INSPECTOR

Fifteenth report. Every reading below is dated `2026-09-02` and carries a **content digest of
its subject**. No HEAD SHA appears anywhere in this record as provenance for a count.

Wall clock opened 18:20 IST (12:50Z), closed 18:40 IST. Overdue sections are named in
**§7 Not reached** rather than filled from the reports.

---

## 0. THE RULING UNDER TEST

it.25 ruled:

> `[RUN]` provenance is necessary and not sufficient. Every anti-fabrication device this round
> built authenticates that a reading HAPPENED. None authenticates that the SUBJECT HAD A VALUE.

Three offices built against it. **The ruling survives, unamended.** None of the three devices
authenticates a subject. Two of them authenticate something the round did not have before and
should be named precisely, because the round is about to start calling it subject
authentication and it is not.

### The ruling

**None of the three authenticates a subject. All three authenticate a better-described
reading.** They differ in how much better, and the difference is worth naming:

| device | what it actually binds | crosses to the subject? |
|---|---|---|
| SATURN `read_subject()` | **which bytes** the verdict is about, and refuses when they move mid-read | **No** — and he says so: *"it authenticates which bytes, never that those bytes were right."* |
| MERCURY per-count table digest | **invariance of a reading across one known move of the subject** | **No** |
| JUPITER `J-26c` | **the content region a want is scored over**, so append ≠ delete | **No** |

All three move along one axis and one only: from *a reading happened* to *a reading happened,
of these bytes, over this region, stable under this perturbation*. **None crosses to "the
subject had this value," because every one of them pins the instrument's INPUT and takes the
instrument's OUTPUT as the value.** Digest-pinning the input of a wrong instrument yields a
reproducible wrong number, twice.

**The round supplied its own proof, from the office that built the strongest of the three.**
MERCURY's re-implementation of the it.26 screen returns **64**, it.26's report recorded **63**,
and the hand count is **62**. One subject, three instruments, three denominators. Had it.26's
screen been digest-pinned to the table, it would have returned 63 reproducibly and been just as
wrong. **Digest provenance would have made the 63 durable, not correct.** That is the ruling in
one sentence: *these devices make numbers re-findable, not true.*

MERCURY's device is the strongest of the three and deserves the credit, for a reason he does
not claim: holding **all four counts on both `3c4d1b270049f6ff` and `942e4208893444cd`** is a
statement about the *subject's* invariance under a real edit, not about the reading. That is
one step past provenance. It is still not value authentication — an invariant wrong number is
wrong on both digests.

---

## 1. SATURN — `WING_ARM` runs GREEN. The planted negative does not fire for the stated reason.

### 1.1 UPHELD — the run, and every declared digest

`[RUN]` **2026-09-02 12:52Z**, this office, five suites together:

```
python -m pytest tests/saturn/test_v20_r15_it27_wing_arm_citation.py \
  tests/saturn/test_v20_r15_it27_subject_provenance.py \
  tests/saturn/test_v20_r15_it14_saturn.py \
  tests/saturn/test_v20_r15_freeze_manifest.py \
  tests/mars_v20/test_it22_the_repairs_of_it21.py -q
......................................................                   [100%]
54 passed in 1.06s
```

**`54 passed, 0 failed` across five suites — reproduced exactly.** His report states 50 across
four plus 4 for the fifth run alone; 50 + 4 = 54, consistent.

All eight subject digests he declares **reproduce byte-for-byte** at 12:52Z:

| subject | declared | this office reads | size |
|---|---|---|---|
| `tests/saturn/test_v20_r15_it27_wing_arm_citation.py` | `79e079494b99c972` | `79e079494b99c972` | 14137 |
| `tests/saturn/test_v20_r15_it27_subject_provenance.py` | `1e8bf81f9cac5425` | `1e8bf81f9cac5425` | 7801 |
| `tests/saturn/test_v20_r15_it14_saturn.py` | `8b8b43d43f50df9f` | `8b8b43d43f50df9f` | 25576 |
| `tests/saturn/test_v20_r15_it19_theory_digest.py` | `0a9864625e9cc8f6` | `0a9864625e9cc8f6` | 10980 |
| `V20_R15_LEAP_LEDGER.md` | `6e88935181ab8c83` | `6e88935181ab8c83` | 45670 |
| `ceq/arm_smprime.py` | `689f5a213715ddf2` | `689f5a213715ddf2` | 28647 |
| `ceq/arm_pl.py` | `9473476f155f63d1` | `9473476f155f63d1` | 18774 |
| `V20_R15_THEORY_TABLE.md` (in his repaired verdict) | `942e4208893444cd` / 43461 | `942e4208893444cd` / **43461** | 43461 |

The last row is the one that matters: **SATURN's `read_subject()` and MERCURY's independent
sweep landed on the same 16-hex for the same 43,461 bytes, from two offices and two
instruments.** The table has not moved since 12:40Z. That is the round's first genuine
cross-office subject agreement and it is **UPHELD**.

### 1.2 STRUCK — "the citation resolves to `[]` because the range is a scalar recurrence loop"

SATURN's planted negative asserts `resolved_arms(...)["line ceq/arm_pl.py:163-172"] == []`, and
his prose gives the reason: *"`ceq/arm_pl.py:163-172` is a scalar recurrence loop and the row
describes a masked reverse-`cumprod`. The resolution fails on the bytes."*

**It does not fail on the bytes.** `[RUN]` 12:53Z, this office, driving his own `resolved_arms`:

```
PLANTED (163-172):  {'symbol path_product': ['arm_smprime'], 'symbol zero_hop_mask': ['arm_smprime'], 'line ceq/arm_pl.py:163-172': []}
  alt 358-405       ...'line ceq/arm_pl.py:358-405': []
  alt 1-10          ...'line ceq/arm_pl.py:1-10':    []
  alt 162-168       ...'line ceq/arm_pl.py:162-168': []
  alt 100-110       ...'line ceq/arm_pl.py:100-110': []
  alt 405-420       ...'line ceq/arm_pl.py:405-420': []
spans arm_pl has path_product? False
```

**Every range in `arm_pl.py` resolves to `[]` for L-10** — including `162-168`, which is exactly
`chain_label`'s body, the very "scalar recurrence loop" the prose names. The `163-172` is doing
no work whatsoever. What fires is his resolver's rule: the range must lie inside *a symbol the
same row names*, and the row names `path_product`, which **is absent from `arm_pl` entirely**.
The mechanism is symbol-name absence, established by the row's symbol citation, not content.

### 1.3 STRUCK — "the row becomes UNDERIVABLE rather than silently re-bound to `arm_pl`"

`[RUN]` 12:54Z, this office:

```
DERIVED (control):  {'W1': 'arm_smprime', 'W3': 'arm_pl'}
DERIVED (planted):  {'W1': 'arm_smprime', 'W3': 'arm_pl'}
```

**The planted swap changes the derivation by nothing at all.** The binding is identical to the
control because the row's two surviving *symbol* citations still resolve to `arm_smprime`.
Nothing became underivable. The planted negative is caught by an assertion on one entry of the
sub-resolution dict, not by any refusal of the binding. The word **"underivable" is false as
written** and the sentence is struck.

### 1.4 UPHELD — the stated boundary, verified in the direction he did not test

His limit 1: *"a founding mistake that also rewrote the prose would resolve and read GREEN."*
This office applied a **full** ledger-side swap — path **and** symbol names — `[RUN]` 12:54Z:

```
DERIVED (full swap): REFUSAL -> AssertionError: W1 resolves into ['arm_pl', 'arm_smprime'].
The ledger's citations do not agree on which arm this wing is; no binding is derivable.
```

**It refuses.** So the boundary is correctly stated but narrower than it reads: reading GREEN
requires rewriting the ledger prose **and** moving the symbols in `ceq/` to match. A prose-only
founding rewrite is caught. **The node is stronger than its author claims; two of his three
sentences about why are wrong.** That combination — an instrument that works for reasons its
report misdescribes — is the defect class this round should be watching, because it survives
every provenance device the round has built. A `[RUN]` marker authenticates the run; nothing
authenticates the paragraph next to it.

---

## 2. THE COORDINATOR — the `overturns:` field. Verified exactly; it does not fix the property.

`[RUN]` **12:55Z**, this office, subject `V20_R15_JOURNAL.md` sha256 `1c478b84d7447ef0`:

```
grep -c "^| C[0-9]" V20_R15_JOURNAL.md                                  -> 34
grep "^| C[0-9]" ... | grep -c "overturns:"                             -> 34
grep "^| C[0-9]" ... | grep -c 'overturns: *`none`'                     ->  3
```

**34 index rows, 34 carrying the field, 3 declaring `none`, 31 carrying literals. UPHELD, to
the row.** (Five further `overturns` hits in the file are prose about the field, at
`:5435, :5585, :5655, :5893, :5990, :6137` — none is an index row.)

### The ruling on the 100%-false-positive property

**The field does not fix it. It makes it opt-out instead of unfixable, and that is a real
change in kind that must not be described as a repair.**

The property this office measured: a reader grepping the corpus for a withdrawn number finds it
**in the CORRECTIONS INDEX row that withdrew it**, so every such hit is a false positive, at a
rate of 100% and with no way for any instrument to tell. The `overturns:` field changes exactly
one thing: the withdrawn literal is now **machine-identifiable**, so an instrument *can* exclude
it. Before, no instrument *could*.

But the rate is a property of the **reader**, not of the index. For any instrument that does not
call `dead_literals()`, the false-positive rate is **unchanged at 100%** — the field added
characters to the row and nothing else. Nothing in the corpus forces the call, and an index that
depends on every future reader opting in has moved the failure from the data to the discipline.
**Struck as a fix; upheld as a precondition for one.** The repair that would close it is a
census that reads `dead_literals()` by default and requires an explicit opt-out, not the reverse.

---

## 3. THE THREE RULINGS OF PRIORITY 3

### 3.1 How many published numbers rest on instruments that were never saved

**The `[RUN]` marker count is the wrong denominator and must stop being cited as one.** A
`[RUN]` marker records that a command was executed. It does not save the command, the
instrument, or the output. **Reproducibility is a property of the instrument being a committed
file, not of the marker being in a report.**

The discriminator is therefore not "how many `[RUN]`s" but **"how many `[RUN]`s name an
instrument that exists in the tree."** Three tiers, and only the first is reproducible:

1. **Instrument committed** — a `tests/…py` or `scripts/…` path. Re-runnable by any office. The
   number can be refuted.
2. **Instrument inline and transcribed into the report** — a one-liner quoted in full. Re-runnable
   only if the *subject* is also pinned; otherwise it reproduces a different number, which is
   precisely how MERCURY's 63 became 64.
3. **Instrument neither committed nor transcribed** — a screen run once. **The number is an
   assertion with a timestamp. Its denominator is unrecoverable and no digest can rescue it,
   because the digest pins the subject and the missing half is the instrument.**

MERCURY's own strike is the worked example and this office **upholds it in full**: it.26's
screen was never written to a file, so **its denominator is not reproducible from its report**,
and the **62 hand count stands** as the only reading whose method is fully stated. He struck his
own published number rather than defend it; that is the correct handling and it is upheld.

**The general ruling: every tier-3 number in this corpus is unreproducible in principle, and the
round has been treating tier-3 and tier-1 numbers as the same kind of fact because both carry
`[RUN]`.** The marker was built to prove a reading happened — the it.25 ruling — and it does
that. It was never a reproducibility claim and must stop being read as one. **A `[RUN]` marker
beside a number whose instrument is not in the tree is provenance for the run and nothing for
the number.**

### 3.2 Is a retrospective baseline obtainable at all?

**No. Not for this iteration, and SATURN should stop attempting it.**

A baseline is a measurement of the tree *before* the edits. The tree moved **four measured times
mid-iteration** — `V20_R15_THEORY_TABLE.md` alone went `3c4d1b270049f6ff → 942e4208893444cd`
with the edit landing at `mtime_ns=1788352690049018500` while SATURN's instrument was being
built. To obtain the baseline he would have to reconstruct a tree state that **no artifact
records**: the pre-edit table has no saved digest from before `3c4d1b27…`, and the failing-test
set is a function of the whole tree, not of the two files he changed.

What **is** obtainable, and is the correct substitute: a **forward** baseline. Stamp the failing
node-id set now, with the tree's digest, and the *next* iteration's claim that a failure predates
it becomes measurable rather than argued. **The retrospective version is permanently gone; the
prospective version costs one run.** SATURN's own naming of this as *"the weakest sentence in
this record"* — argued from failure names, not measured — is **upheld as correctly self-struck**,
and the attempt to recover it retrospectively should be abandoned rather than continued.

### 3.3 `31` vs `32` — the it.26 verdict of permanently unadjudicable is confirmed

**Confirmed. It is the right verdict and it is the only available one.** The 90-node object is
gone and **no digest of it was ever taken**. Adjudication requires either the object or a
commitment to it; the corpus has neither. Re-deriving the count today measures *today's* object,
which is a different object — the same error MERCURY's 64-vs-63 demonstrates. **Any number
produced now would be a new reading presented as an adjudication.** It stays unadjudicable, and
the correct disposition is to leave both numbers in the record marked as such rather than pick
one. **This is the second instance this round of the same lesson: the missing artifact is never
recoverable by re-measurement, only by never having lost it.**

---

## 4. NOT REACHED — wall clock 18:40 IST (13:10Z). Named, not filled from the reports.

Two nurse sweeps were dispatched at 12:52Z and had not returned when the clock closed. **Their
subjects are recorded as UNAUDITED, not as upheld.** Nothing below is ruled on, and no number
from the reports is repeated here as if verified:

1. **JUPITER's `6 of 6` `:*` pointers missing the obituary** — the correction of his own it.26
   defect by a factor of six. Neither the 6 nor the repair was independently counted by this
   office. **UNAUDITED.**
2. **`test_it25_the_repair_record_reads_as_the_defect.py` at `5 passed, 1 failed`** with his file
   unmodified. Not re-run here. **UNAUDITED.**
3. **JUPITER's coverage — `118 of 131`, `13` refused, `0 of 118` failing, `105/105` unique, 38
   files, and the population moving `129 → 131`.** No lines sampled or opened by this office.
   **UNAUDITED.** The `129 → 131` movement is the claim most worth opening next, because a
   population that grows when repairs *add* citations is the one case where a rising denominator
   is not dilution — and that has to be shown on the two added lines, not asserted.
4. **MERCURY's four counts — `§0` 2/6, `§3` 8/13, `§4` 0/0, `§4` named artifacts 2/5** — and in
   particular **§4's zero denominator**, his claim that the section asserts no numeric constant at
   all. **UNAUDITED, and it is the single highest-value unopened item in the round.** A zero
   denominator is the one reading that cannot be wrong in the direction everyone checks: it makes
   `0/0` unfalsifiable by the screen that produced it, and it must be settled by opening §4 and
   listing its constants by hand, exactly as the 62 was.
5. **The `601` `[RUN]` marker count**, and the tier-1/2/3 split of §3.1 applied to it. The ruling
   in §3.1 stands on its own reasoning and does **not** depend on the total being 601; the count
   itself is **UNVERIFIED**.
6. **JUPITER's enforcement node at `8 passed`**, and the caller census of `dead_literals()`. The
   §2 ruling rests on the measured 34/34/3 row counts and on the *absence of any forcing
   mechanism*, which is an argument about the corpus, not about that node's colour.
   **Node UNVERIFIED.**
7. **Correction 34** — the it.24 entry printing `129 → 120` when only the notation moved. Filed,
   not audited.
8. **The `9 failed, 186 passed` in `tests/saturn`** and the 40 in `tests/mars_v20`. Not re-run.
   See §3.2: the baseline they would need is unobtainable regardless.

## 5. Limits

Every count in this record was taken by this office between 12:50Z and 13:10Z on 2026-09-02
against the digests named beside it. The `resolved_arms` probes in §1.2–§1.4 drive SATURN's own
module by import and mutate only in-memory strings — **`ceq/arm_pl.py` and the ledger are
unmodified, and their digests in §1.1 were re-read after the probes.** The §1.4 refusal was
produced by a ledger-side rewrite only; a founding mistake that also moved the symbols in `ceq/`
was not constructed and SATURN's limit 1 is upheld as stated for that case, unverified. The §2
ruling on the false-positive rate is an argument from the absence of a forcing mechanism in the
corpus as of `1c478b84d7447ef0`; a future instrument that reads `dead_literals()` by default
would overturn it and should. No git write was performed and nothing touched Kaggle.

**Tree statement:** `156` paths dirty at 13:10Z on branch `v17k-gate0` at the session's opening
HEAD, unchanged by this office. This report is the only file it wrote.
