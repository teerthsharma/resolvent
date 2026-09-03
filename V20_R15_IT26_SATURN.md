# V20 R15 it.26 — SATURN (WATSON, instruments)

Branch `v17k-gate0`. Opened 12:17:07Z, filed 12:31Z, 20 min cap. No git writes, nothing
touched Kaggle. Every reading below is dated and carries a content digest of its subject.

## VERDICT — MARS is right, and all three readers were right. The counts were unsourceable because the subject moves.

`tests/mars_v20` is **22 files, 0 known to git**. `tests/saturn` is **18 files, 4 known to
git**. A count stamped `at HEAD 207e7b9` over either names a commit that contains almost none
of it. Worse than unsourceable: **the suites changed between the three readings**, so `31/59`,
`32/58` and `35/61` are three true counts of three different objects. Commit-SHA provenance
replaced with a content digest below.

## REPAIR 1 — every count re-published against a content digest

Digest = sha256 over `(relpath, sha256(bytes))` for every `test_*.py`, sorted. `[RUN]`

    tests/mars_v20  22 files  a8daeb3165a43b75c94cc0ae2a6bb66e2dc237507808aa0f665336af47758c76
    tests/saturn    18 files  9a9f1868fac1803fe3ca9ddf86267442b70f99ab878abdad5bc19bf764904285

| Suite | Reading | Digest | Read at | Determinism |
|---|---|---|---|---|
| `tests/mars_v20` | **35 failed, 61 passed** (96 nodes) | `a8daeb31…8c76` | 12:19–12:22Z | **5/5**, and the FAILED node *set* is identical across all five runs, not merely the total |
| `tests/saturn` | **9 failed, 177 passed** (186 nodes) | `9a9f1868…4285` | 12:24Z | 3/3 |

### The contested counts, adjudicated

| Published | By | Nodes | Verdict |
|---|---|---|---|
| `tests/mars_v20` **31 failed, 59 passed** | SATURN it.24 | 90 | superseded — a different suite (90 nodes, not 96) |
| `tests/mars_v20` **32 failed, 58 passed**, 5/5 | MARS it.25 | 90 | superseded — same 90-node object, one node apart from mine |
| `tests/mars_v20` **32/58**, 4/4 | INSPECTOR | 90 | as MARS |
| `tests/saturn` **8 failed, 177 passed** | SATURN it.24 | 185 | **WRONG** |
| `tests/saturn` **9 failed / 177 passed**, 186 | INSPECTOR | 186 | **CONFIRMED exactly, 3/3 at 12:24Z** |

The 90 → 96 movement in `tests/mars_v20` is MARS's own it.25 file,
`test_it25_the_repair_record_reads_as_the_defect.py`, landing between his reading and mine.
**No count in this round over these suites was ever sourceable, mine included**, and the
`31` vs `32` disagreement is not adjudicable after the fact — the 90-node object no longer
exists on disk and no digest of it was ever taken. That is the defect, not the gap.

**This office's it.21 `39 failed` is withdrawn, not defended.**

## REPAIR 1b — the INSPECTOR's `tests/saturn` attribution: **CONFIRMED, at node level**

His claim: JUPITER's theory-table edits broke SATURN's it.19 freeze node, and neither filing
says so because the two reads straddled a write.

Confirmed. The ninth failure is exactly the node he named. `[RUN]` 12:25Z

    FAILED tests/saturn/test_v20_r15_it19_theory_digest.py::
           test_the_declared_cells_digest_matches_the_table_at_head
    E  AssertionError: cell bodies edited since the freeze:
    E  ['Q1/W1', 'Q2/W3', 'Q3/W1', 'Q6/W1', 'Q6/W3']

**Five cell bodies moved since the it.19 freeze.** And the subject is still being written:

    mtime V20_R15_THEORY_TABLE.md            2026-09-02T12:23:28Z   <- DURING this iteration
    mtime tests/.../it19_theory_digest.py    2026-09-02T10:36:33Z

The 12:23:28Z write is *later* than the 11:51:49Z one the INSPECTOR reconstructed, so the
table has been written at least twice more since he read it. His mechanism is right and his
timestamp is one of several.

**The half that cannot be confirmed is this office's own.** There is no record of *when* the
it.24 `8 failed` was read, because that count carried a HEAD SHA and no clock — the identical
defect this filing repairs. So: the attribution is confirmed on mechanism, node, and
subject-mtime; the claim that the it.24 read preceded the 11:51:49Z write is *consistent with*
the it.24 filing window (11:46:04–11:57Z) and is **not independently sourceable from that
record**. Filed against this office.

### The INSPECTOR's ruling lands on this node, and this office built it

> `[RUN]` provenance is necessary and not sufficient. Every anti-fabrication device this
> round built authenticates that a reading HAPPENED. None authenticates that the SUBJECT
> HAD A VALUE.

`test_the_declared_cells_digest_matches_the_table_at_head` is in the class. Its subject is a
file another office edits mid-iteration, so its verdict is a function of *when* it ran, and it
records no such time. It is not a race like the it.21 node — it is deterministic against the
bytes on disk — but the bytes on disk are not a fixed subject. **Conceded in full. Not
repaired this iteration** (see Not reached).

## REPAIR 2 — `WING_ARM`: the pin moves to the **ledger binding**. Consensus is withdrawn.

**Decision: `WING_ARM` rests on `V20_R15_LEAP_LEDGER.md`, not on the six-office consensus.**

The reason is not that the ledger is a better-authored document. The two are different *kinds*
of evidence, and only one is falsifiable by something other than another report.

The consensus witness counts votes: `W1 = arm_smprime` 25 to 7, `W3 = arm_pl` 23 to 7. Every
one of those 32 votes is a sentence in an office report, and **every office report descends
from the manifest.** If it.1 wrote the map backwards, all 32 votes are one mistake counted 32
times and the witness still reads GREEN. That is precisely the limit this office stated at
it.22 and could not get past: *consensus defeats an editor, not a founding mistake.*

The ledger rows are a different object because **their grades are derived from the arms' source
code, under the wing id.** They do not name the arm beside the wing; they reason about it:

- **L-10, `Q4 / W1 arm_smprime`** grades the sparsity gap by reading `ceq/arm_smprime.py:163-172`
  — "materialises the dense `[S,S]` block **before** producing the zeros". Resolved `[RUN]` 12:26Z:
  those lines are the `le` mask, the reverse-`cumprod`, and the `masked_fill` path product,
  exactly as described. **The citation resolves, under W1, into `arm_smprime`.**
- **L-14, `Q6 / W3 arm_pl`** grades `F4` off `ArmPL.forward` returning an `[n]` scalar.
  Resolved `[RUN]` 12:26Z: `class ArmPL` at `ceq/arm_pl.py:358`. **Resolves under W3, into `arm_pl`.**
- **L-16, `Q2 / W1`** rests its `F4` on **bed membership** — W1 is BED-M, the Hankel `1/d` law
  binds BED-K, so the domain intersection is empty. That grade is *unstateable* if W1 is the
  other arm.

The binding is load-bearing in the ledger's arithmetic, not decorative. **If W1 were `arm_pl`,
L-10's line citation would resolve into the wrong module and L-16's `F4` would be underived.**
A vote cannot be wrong in a way that shows; a resolved line citation can.

### What would detect the founding mistake — the answer owed since it.22

Not a wider consensus. **A citation-resolution node**: for each wing, take the arm the pin
claims, and require that the ledger's per-wing code citations resolve *into that arm's module*
and contain what the row says they contain. That check consumes no office's opinion. It fails
if it.1 wrote the map backwards, because the line numbers stop landing on the described code.

Two such resolutions are established by hand above `[RUN]`. **The node is not written** (see
Not reached), and a hand-resolution is not a shipped instrument — which is the mistake this
whole round is about.

## AND ONE THING MARS TURNED BACK — the `EXIT` trap, repeated, and **the discharge holds 3/3**

He is right that one sample is one sample. Two further independent `start 1` runs, concurrent,
in separate scratch ROOTs, on the shipped script — sha256
`67b049c662291b64f6aa6cbdaace86b7656dddd3a75f8c6bc46227fd88a5e931`, **byte-identical to the
copy the it.24 run used.** `[RUN]` armed 12:24:33Z, deadline 12:25:33Z, observed 12:25:48Z.

| | RUN A | RUN B | it.24 |
|---|---|---|---|
| wpid | 53426 | 53428 | 46097 |
| registration at arm | EXISTS | EXISTS | EXISTS |
| watchdog after deadline | **DEAD** | **DEAD** | DEAD |
| registration after `trap … EXIT` | **REMOVED** | **REMOVED** | REMOVED |
| `ITERATION_OVERDUE` | `12:25:33Z exceeded 1 min` | `12:25:33Z exceeded 1 min` | `11:48:10Z` |
| OVERDUE raised at | the deadline, to the second | the deadline, to the second | ditto |
| `check` exit code | **1** | 1 | 1 |

**3 of 3. The discharge is confirmed with a repeat count.** The MARS it.20 stale-`$WATCHDOG`
condition does not reproduce on this box in three independent real arms.

**One artifact of this office's own harness, recorded rather than hidden.** The first reading
of the exit code came back `rc=0` on both runs, which would have been a live defect. It was
not: the harness ran `iteration_timer.sh check … | sed`, so `$?` was **sed's** status, not the
script's. Re-measured without the pipe: `TRUE check rc=1` `[RUN]` 12:26:36Z. **Third instance
this round of an instrument of mine reading a channel that cannot carry the fact it is asked
for** — after `git diff` on an untracked path (it.24) and the HEAD-SHA count provenance
(above). Caught before publication this time, and only because the number was surprising.

## Limits

The `31` vs `32` disagreement on `tests/mars_v20` is unadjudicable and will stay so: the
90-node object is gone and no digest of it was ever taken. The theory-table attribution is
confirmed on mechanism and node, but its timing half rests on a filing window rather than a
recorded read time. The `WING_ARM` ledger decision is argued from two hand-resolved citations,
not from a shipped resolver, so it is at present a better *argument* than the consensus and not
yet a better *instrument*. The trap result is three samples on one box and one OS; it says the
condition does not reproduce here, not that it cannot occur.

## Not reached

1. **The citation-resolution node for `WING_ARM`.** Argued and hand-checked, not shipped. The
   main debt of the iteration.
2. **A repair for `test_the_declared_cells_digest_matches_the_table_at_head`.** The INSPECTOR's
   ruling against it is conceded and unaddressed. The shape of the fix — record the subject's
   digest *and* its mtime alongside the verdict, so a stale read is distinguishable from a moved
   subject — is named here and not built.
3. Whether counts published by **other** offices over these two suites carry the same
   unsourceable-provenance defect; only this office's are re-published here.
4. No new RED shipped this iteration. Every reading above is a measurement against unmutated
   code, and the two instruments the iteration calls for are in Not reached, not in `tests/saturn`.
