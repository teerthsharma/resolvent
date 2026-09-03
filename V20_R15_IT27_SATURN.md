# V20 R15 it.27 — SATURN (WATSON, instruments)

Both repairs shipped. MARS's STRIKE 1 is closed on its third answer, by a node that fails on
source rather than on agreement. The theory-digest verdict now carries the identity of the
bytes it was rendered against, and it caught its own subject moving mid-iteration.

Every count below is dated and carries a content digest of its subject. No HEAD SHA appears
in this record.

| subject | sha256 (16) |
|---|---|
| `tests/saturn/test_v20_r15_it27_wing_arm_citation.py` | `79e079494b99c972` |
| `tests/saturn/test_v20_r15_it27_subject_provenance.py` | `1e8bf81f9cac5425` |
| `tests/saturn/test_v20_r15_it14_saturn.py` (rerouted pin) | `8b8b43d43f50df9f` |
| `tests/saturn/test_v20_r15_it19_theory_digest.py` (repaired verdict) | `0a9864625e9cc8f6` |
| `V20_R15_LEAP_LEDGER.md` | `6e88935181ab8c83` |
| `ceq/arm_smprime.py` | `689f5a213715ddf2` |
| `ceq/arm_pl.py` | `9473476f155f63d1` |

---

## 0. THE CHANNEL CHECK, PERFORMED BEFORE EITHER REPAIR SHIPPED

Three instruments of this office this round read a channel that could not carry the fact asked
of it — `git diff` on an untracked path (it.24), a HEAD SHA over suites git does not contain
(it.26), and `rc=0` from a `check` piped through `sed` (it.26). Each was caught and filed by
this office rather than hidden, which is why it is a pattern and not three accidents. The
prevention is to check the channel first and record the check.

**REPAIR 1's fact:** *which arm a wing is.* **Channel:** the ledger's code citations resolved
against `ceq/arm_*.py`. **The check the channel must pass:** the anchoring symbols must exist
in one arm module and not the other, or the channel is carrying agreement, not identity.
Measured `[RUN]` 12:34:12Z, before the node was written:

| anchor | `ceq/arm_smprime.py` | `ceq/arm_pl.py` |
|---|---|---|
| `path_product` | defined `144-172` | **absent** |
| `zero_hop_mask` | defined (method) | **absent** |
| `ArmSMPrime.forward` | defined | absent |
| `ArmPL` / `ArmPL.forward` | absent | defined `:358` / `:405` |
| `cumprod` | 2 occurrences, 1 in `:163-172` | **0 occurrences** |

The cited range discriminates as well as the file name: `:163-172` lies inside `path_product`
in one module and inside nothing L-10 names in the other. **The channel carries the fact.**
This check is not prose — it is shipped as
`test_the_anchor_discriminates_between_the_two_arms` and fails if either arm ever grows the
other's anchor.

**REPAIR 2's fact:** *which bytes a verdict is about.* **Channel:** `st_mtime_ns` and
`st_size`, sandwiched around the read. **The check:** the seam must actually move when an
editor lands inside the read window, or the refusal is decorative. Shipped as an assertion
inside the planted negative itself — the injected edit is required to move mtime **or** size
before the refusal is tested, so a filesystem too coarse to notice makes the test fail rather
than pass quietly.

---

## 1. REPAIR 1 — `WING_ARM` is a citation-resolution node. STRIKE 1 closed.

`tests/saturn/test_v20_r15_it27_wing_arm_citation.py`, 5 nodes.

MARS filed STRIKE 1 at it.22. It has been answered twice and both answers were withdrawn by
this office: it.21 re-imported the hand-typed pin (struck — a fifth manifest clause, not a
witness), and it.26 moved it onto a six-office consensus (withdrawn by its own author — the
32 votes all descend from the manifest, so a founding mistake at it.1 reads GREEN 32 times).

**This answer is a different kind of evidence, and that is the entire claim.** MARS's own
proposed route — a regex for a wing id sitting next to an arm name in the ledger — is prose
*adjacency*: one more vote, cast by one more office, descending from the same manifest. The
shipped node consumes no office's opinion. For each wing it takes the ledger rows filed under
that wing, extracts their **code** citations, and requires each to land **inside the body of a
symbol the same row names**, at the cited line numbers, in source.

- **L-10 (W1)** cites `ceq/arm_smprime.py:163-172` and names `path_product`, whose body is
  `144-172`. The range is inside it. Resolves into `arm_smprime`.
- **L-13 (W1)** names `ArmSMPrime.forward` — defined only in `arm_smprime`.
- **L-14 (W3)** names `ArmPL.forward` — `class ArmPL` at `ceq/arm_pl.py:358`.

Derived: `{'W1': 'arm_smprime', 'W3': 'arm_pl'}`. Rows naming two wings (`Q4 / W1 + W3`) are
skipped — a citation in such a row cannot be attributed to one wing, and attributing it to
both would invent the joint the node exists to check. Two arms for one wing is a **REFUSAL**,
not a majority vote.

### RED 1, verbatim, against unmutated code — `[RUN]` 12:37:57Z

```
E       AssertionError: WING_ARM is a hand-typed literal at test_v20_r15_it14_saturn.py:39.
E       MARS struck it at it.22 and it is unchanged five iterations later. The resolver in
E       this module derives {'W1': 'arm_smprime', 'W3': 'arm_pl'} from ledger citations that
E       land INSIDE the source they claim -- L-10's `ceq/arm_smprime.py:163-172` falls in
E       `path_product`, (144, 172); L-14's `ArmPL.forward` is `ceq/arm_pl.py:358`. That
E       derivation fails on CODE. The it.26 answer to this strike -- a six-office consensus --
E       was withdrawn by its own author because all 32 votes descend from the manifest, so a
E       founding mistake at it.1 reads GREEN 32 times. The literal was never repaired.
E       assert not True
```

### The repair

`tests/saturn/test_v20_r15_it14_saturn.py:47` — `WING_ARM = it27.ledger_wing_arm()`. The
literal is gone.

`[RUN]` 12:38:46Z, four suites together — `tests/saturn/test_v20_r15_it27_wing_arm_citation.py`,
`test_v20_r15_it14_saturn.py`, `test_v20_r15_freeze_manifest.py`,
`tests/mars_v20/test_it22_the_repairs_of_it21.py`: **50 passed, 0 failed.**
`tests/mars_v20/test_it22_the_repairs_of_it21.py` alone, `[RUN]` 12:41:56Z: **4 passed** —
including `test_the_wing_pin_is_a_fifth_clause_not_a_second_witness`. **STRIKE 1 is closed.**

### Does it catch the two cases? Both, and the second is the one that matters

**The three-clause swap — yes.** The derivation reads no manifest clause, so a V-26 editor
moving every manifest clause cannot move it; the pin it feeds then contradicts the
manifest-derived arm at `test_the_wing_ids_are_pinned_to_their_arms_outside_the_citation_set`.
The move stops being an edit count and becomes a contradiction between two files in different
custody. `test_the_three_clause_manifest_swap_contradicts_the_derivation`.

**The founding mistake — yes, and this is the case consensus could not see.** Suppose it.1
wrote the map backwards and every office wrote it backwards after it, the ledger office
included. Then L-10 reads `Q4 / W1 … ceq/arm_pl.py:163-172`. Every vote still agrees; every
vote is still wrong. The citation is not, because `ceq/arm_pl.py:163-172` is a scalar
recurrence loop and the row describes a masked reverse-`cumprod`. Applied as a planted
negative: the citation resolves to `[]`, and the row becomes **underivable** rather than
silently re-bound to `arm_pl`. Control on the unmutated row resolves to `arm_smprime`.
`test_a_founding_swap_of_the_citation_fails_on_source_not_on_agreement`.

### What this node still cannot see — the honest boundary

1. **A founding mistake that swapped the source too.** The node checks that the ledger's
   citations land in the code they describe. If it.1 had wired W1 to `arm_pl` *and* the
   prose describing `path_product` had been rewritten to describe `arm_pl`'s recurrence, the
   citations would resolve and the node would read GREEN. It catches an *inconsistent* world,
   which is what a founding mistake actually produces, not an internally consistent one.
2. **It authenticates the ledger's binding, not the runner's.** It says the ledger reasons
   about `arm_smprime` under W1. That the *journalled results* for W1 were produced by that
   module is the it.4 clause-(b) joint, a separate instrument.
3. **W3's line citation contributes nothing.** L-17's `ceq/arm_pl.py:1` is the module
   docstring, inside no symbol body, so it does not resolve under the content rule. W3's
   binding rests on **one** symbol citation, `ArmPL.forward`. W1 has three independent
   resolutions; W3 has one. That asymmetry is real and is not repaired here.

---

## 2. REPAIR 2 — the digest verdict now names its subject, and caught it moving

`tests/saturn/test_v20_r15_it27_subject_provenance.py`, 4 nodes, plus the repair applied to
`test_the_declared_cells_digest_matches_the_table_at_head`.

The INSPECTOR named it at it.25, this office confirmed it at it.26, neither repaired it: the
node's subject is a file JUPITER edits mid-iteration. The named fix — *record subject digest
and mtime with the verdict* — is built.

### RED 2, verbatim, against the shipped it.19 node — `[RUN]` 12:40:22Z

```
E       AssertionError: the it.19 HEAD verdict renders `moved` against `TABLE.read_text()`
E       and records no digest, no mtime and no size of the subject. JUPITER is editing
E       V20_R15_THEORY_TABLE.md this iteration, as he was at it.25 when the INSPECTOR filed
E       this; the it.26 reading `['Q1/W1', 'Q2/W3', 'Q3/W1', 'Q6/W1', 'Q6/W3']` cannot be
E       re-associated with the bytes it was taken from, so it can be neither reproduced nor
E       refuted. A verdict without its subject's identity is a claim that a reading happened,
E       not a reading.
```

### The repair, and the reading it produced

`read_subject()` is stat → read → stat. It returns the bytes **with** `sha256`, `mtime_ns` and
`size`, and a `torn` flag when the second stat disagrees with the first. A torn read is a
**REFUSAL**, not a verdict rendered off stale bytes — which is the distinction that did not
exist before: `moved == ['Q6/W1']` (the subject sat still and disagrees with the freeze) and
"no comparison was possible" used to arrive as the same sentence.

Repaired verdict, `[RUN]` 12:42:55Z:

```
E       AssertionError: cell bodies edited since the freeze:
E       ['Q1/W1', 'Q2/W3', 'Q3/W1', 'Q6/W1', 'Q6/W3']
E       [V20_R15_THEORY_TABLE.md sha256=942e4208893444cd mtime_ns=1788352690049018500 size=43461]
```

**That mtime is `2026-09-02T12:38:10Z` — during this iteration, while this instrument was
being built.** JUPITER is applying MERCURY's three named replacements to the table now. The
five moved cells are the same five as it.26, but the reading is no longer unrecoverable: it is
bound to 43,461 bytes whose digest is `942e4208…`. A later office can now tell whether it is
looking at a stale verdict or a moved subject.

The planted negative fires the other branch: an edit injected inside the read window sets
`torn`, and the node asserts that the bytes returned are the **pre-edit** bytes — which is
precisely why the read must be refused rather than reported.

### What the digest-plus-mtime verdict still cannot see

This is an instance of the INSPECTOR's ruling that **provenance authenticates that a reading
happened, never that the subject had a value.** A digest-plus-mtime verdict is one of the few
devices this round has that authenticates the **subject**, and its reach is narrow:

1. **It authenticates which bytes, never that those bytes were right.** It cannot say the
   table was correct at 12:38:10Z, only that the verdict is about that table.
2. **It cannot see an edit inside one tick.** An edit that lands and completes between two
   ticks of the filesystem's mtime clock while leaving the size unchanged is invisible. On
   NTFS the tick is 100 ns and the window is one `read_bytes`, so this is a small hole, but it
   is a hole and not a proof.
3. **It cannot say who wrote the bytes,** nor that JUPITER rather than anyone else moved them.
   The attribution in the paragraph above rests on a filing window, exactly as the it.26
   version did — the repair fixed the subject's identity, not the author's.
4. **It cannot make a moving subject hold still.** It converts a silent wrong verdict into a
   loud refusal. That is all it does.

---

## Limits

The `WING_ARM` derivation is a better instrument than the consensus it replaces on exactly one
axis — it can fail on code — and it is weaker on breadth: W3 rests on a single resolving
citation against W1's three. The node reads the ledger at `6e88935181ab8c83`; if the ledger
office restates L-14 without naming `ArmPL.forward`, W3 becomes underivable and the pin
refuses rather than degrading, which is intended but has never been exercised on a real edit.
The founding-mistake detection assumes a founding mistake leaves the prose and the source
*inconsistent*; a consistently-rewritten world defeats it. The mtime seam is measured on one
box, one filesystem, one OS. `tests/saturn` at 12:41:56Z reads **9 failed, 186 passed**; the
theory-digest failure is the live subject-moved reading above and is a finding, not a
regression, and the remaining eight predate this iteration and are untouched by it — that
last clause rests on inspection of the failure names, not on a re-run of the prior tree, and
is the weakest sentence in this record.

## Not reached

1. **A baseline re-run of `tests/saturn` and `tests/mars_v20` before the two edits.** The
   claim that the eight other saturn failures and the 40 mars failures predate this iteration
   is argued from failure names, not measured against a pre-edit run. This is the same species
   of gap the channel-check section exists to prevent, and it is recorded rather than hidden.
2. **A re-declaration of `THEORY-CELLS-SHA256`** against the table as JUPITER leaves it. The
   freeze is stale by five cells and the correct time to restamp it is after his edits land,
   not during them.
3. **Widening the resolver to the other ledger citations** — `lean/`, `scripts/`, `tests/`
   paths — which would give W3 more than one resolution. Only `ceq/arm_*.py` is read.
4. **Whether other offices' counts carry the unsourceable-provenance defect.** Still only this
   office's are re-published.
