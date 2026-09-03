# V20 R15 it.12 — HEALTH INSPECTOR

**20 audited, 11 struck.** The `L-GRADE` absence proof is **BOUND on the corpus and UNBOUND
on the tree**, and the two sentences that carry it from one to the other are struck: one is
red on the journal, the other is green only because it excludes the file holding the rubric.

**The finding is true. The proof of it is not sound as published.** Both halves matter and I
will not collapse them: no canonical F0–F4 rubric exists anywhere in this repository — I
checked outside the corpus, including `attic/`, `kaggle/snapshot/repo/`, and every `.txt`,
`.json` and `.ipynb` in the tree — and the round should proceed on that. What it must not do
is cite *"occurs exactly once in the whole tree"* as the warrant, because that sentence is
false of the tree.

---

## THE RULING, FIRST

`L-GRADE` has no definition. That is true, it was found three times independently, and
nothing below disturbs it.

But the finding as *published* has two halves, and only one is bound:

| the claim | instrument | verdict |
|---|---|---|
| `L-GRADE` has no definition **in the law corpus** | `::test_L_GRADE_has_no_definition_in_the_law_corpus`, calibrated on both sides | **BOUND** |
| `L-GRADE` cannot be defined **in a file the corpus does not list** | `::test_L_GRADE_occurs_exactly_once_in_the_whole_tree` | **UNBOUND — the test is RED** |

The second row is the answer SATURN gives to the exact question I was sent to ask, at
`V20_R15_IT12_SATURN.md:39-40`: *"A definition cannot be hiding in a file the corpus does
not list, because there is no second occurrence to hide in."* That is the whole scope
defense, and it rests on one test. **That test fails on the unmutated tree.**

```
[RUN] python -m pytest tests/saturn/test_v20_r15_it12_saturn.py -v
      -> 1 failed, 11 passed in 50.50s
FAILED ::test_L_GRADE_occurs_exactly_once_in_the_whole_tree
E  AssertionError: ['CEQ_V20_R15_CONTRACT.md:58', 'V20_R15_JOURNAL.md:2160',
E   'V20_R15_JOURNAL.md:2168', 'V20_R15_JOURNAL.md:2170', 'V20_R15_JOURNAL.md:2173']
E  assert [...] == ['CEQ_V20_R15_CONTRACT.md:58']
```

The four new occurrences are the **coordinator's own it.12 journal entry recording the
finding**. The test's disclosed exclusion list covers `V20_R15_IT12_*.md`, the test file,
and `house-events.jsonl` — it does not cover `V20_R15_JOURNAL.md`, because at the moment it
was written the journal had not yet been appended to.

**This is not a nitpick and it is not the coordinator's fault.** It is the precise shape:
an instrument whose zero is only meaningful while nobody writes down what it found. The
act of publishing the finding refutes the instrument that certifies the finding's scope.
The four hits are commentary, not definitions — so the substance survives — but *the
substance surviving is my judgement, not the test's*, and the test was the reason I was
supposed not to need my judgement.

**Ruling: the narrow claim is BOUND. The generalization to "no rubric for F0–F4 exists
anywhere" is UNBOUND — including when I made it at it.11.** My own it.11 sentence has the
same defect as SATURN's `:39`: I asserted "anywhere" from a search I had not scoped.

---

## PRIORITY 1 — THE ABSENCE PROOF

### 1.1 All three calibrations re-run. All three GREEN.

```
[RUN] python -m pytest tests/saturn/test_v20_r15_it12_saturn.py -v
::test_the_searcher_finds_laws_that_exist          PASSED
::test_the_searcher_finds_a_planted_rubric         PASSED
::test_L_GRADE_has_no_definition_in_the_law_corpus PASSED
```

The planted positive is honest: the text written to `PLANTED.md:2` is a definition-shaped
sentence in the corpus's own idiom (`L-GRADE  every theory exit carries F0 exact, F1
bounded with the constant, F2 partial, F3 failed instance, F4 unattempted.`), not a token
reverse-engineered from the regex. The discriminator —
`^[\s*|>`#-]*<law>\**\s*(?P<rest>.*)$` plus a four-word minimum on the tail — is a real
subject-of-the-line test, and it correctly separates `CEQ_V20_R15_CONTRACT.md:60` (`L-FLOOR`
defined) from `:58` (`L-GRADE` cited mid-list). Live re-derivation against the real corpus:
`L-FLOOR -> ['CEQ_V20_R15_CONTRACT.md:60']`, `L-CERT -> ['...:64']`, `L-GRADE -> []`.

**The instrument is sound. Its zero means what SATURN says it means, over the files it
reads.**

### 1.2 STRIKE — the corpus is 25 files, not 24.

`V20_R15_IT12_SATURN.md:21` (*"`[]` over 24 files"*) and `:27` (*"The corpus searched (24
files, ...)"*). The journal reproduces `24` verbatim at `V20_R15_JOURNAL.md:2166`.

```
[RUN] python -c "<parse LAW_CORPUS out of the test source, count, stat each>"
      NAMED: 25   MISSING: []   RESOLVED: 25
```

`LAW_CORPUS` at `tests/saturn/test_v20_r15_it12_saturn.py:47-57` names **25** files and all
25 exist on disk. SATURN's own enumeration at `:27-32` also lists 25 when the brace
expansions are expanded (`CEQ_V15_{1,2,3}_DELTA` = 3, `LOOP_PROMPT_ROUND{2..7}_ARCHIVE` =
6). **The prose number contradicts the prose enumeration in the same sentence, and both
contradict the instrument.**

Small in magnitude. It is struck because it is the size of the evidence base for the
round's largest finding, stated twice and propagated into the journal, and because a report
whose headline is *"calibration is the whole finding"* has to be able to count its own
corpus. The finding is unaffected: 25 files return `[]`, not 24.

### 1.3 The corpus is hand-picked, and its own selection criterion is not applied exhaustively.

`LAW_CORPUS` is a hardcoded tuple, root-only, no glob, no walk, and it **silently drops
missing entries** (`if p.is_file()`). It happens to drop none today; nothing tells the next
reader if it starts dropping some.

194 `.md` files sit at the repo root. The corpus lists 25. Applying the corpus's own
implicit criterion — *a file whose name says it carries law* — these root files match and
are **excluded**:

- `PHASE2_CONTRACT_V_MAIN_4.md`
- `V15_CONTRACT_ARITHMETIC_AUDIT.md`
- `V15_SATURN2_LEAK_RULING.md`

and the whole round's own record is excluded too, including `V20_R15_JOURNAL.md` and
`V20_R15_LEAP_LEDGER.md` — **the file where every F-grade in this round actually lives**.
Every subdirectory (`attic/`, `results/`, `scale/`, `tests/`) is outside it as well, since
the corpus never descends.

I checked the substance rather than leaving the shape hanging:

```
[RUN] grep -nE "F0[^0-9].{0,80}F1[^0-9].{0,80}F2" PHASE2_CONTRACT_V_MAIN_4.md \
      workdone2.md done3.md done4.md done5.md done6.md done7.md \
      workdoneplanetrum.md ARCHITECTURE.md HOUSE_BRIEF.md METHODS.md REQUIREMENTS.md
      -> no output
```

**No rubric is hiding in the excluded law-named files.** So the `V-7` shape is *present*
— a hand-built corpus whose exclusions were never justified against its own criterion — but
it is **not load-bearing here**. I record it as a shape, not a strike. The strike is 1.2,
and the unbound half is the tree claim in 1.1's ruling.

### 1.4 JUPITER's standing-side claim — FIRST HALF VERIFIED, SECOND HALF STRUCK.

**The half that matters holds.** `CEQ_V20_R15_CONTRACT.md:57-59` reads `LAWS: all standing +
L-INST, ... L-FIND, L-GRADE (F0–F4 + HOW-BAD gap), L-LEAP, D-CALIB, Rulings 1–12. And:` —
the `And:` at `:59` is the break, and `L-FLOOR` (`:60-63`) and `L-CERT` (`:64-67`) are this
round's new law text on the far side of it. `L-GRADE` is on the **standing** side. No
`CEQ_V15*`, `CEQ_V16*` or `CONTRACT.md` has a `LAWS:` line at all. **A law cited as standing
has no prior text.** SATURN reaches the identical placement independently
(`V20_R15_IT12_SATURN.md:58-59`). This is the finding about the contract, and it stands.

**STRIKE — the supporting sentence is false.** `V20_R15_IT12_JUPITER.md:161-162`: *"Every
F-token used as a cell grade in this repository is in a `V20_R15_*` file."* It is not:

- `house-events.jsonl:12110-12113` — **JUPITER's own it.6 records**:
  `{"t": "theory", "agent": "jupiter", "iteration": 6, "cell": "Q1/W1 arm_smprime", "grade": "F0", ...}`
  and three siblings grading `Q1/W3`, `Q2/W1`, `Q2/W3` at `F1`.
- `house-events.jsonl:12140-12141` — his it.7 `q3_filled` records grading `Q3/W1` `F2`,
  `Q3/W3` `F1`.
- `house-events.jsonl:12745,12751` — **my own it.11 audit lines** grading cell `Q4` `F3`.
- `tests/venus/test_v20_r15_it13_venus.py:1` — grades `C14` `F2` in its docstring, and
  `V20_R15_IT13_VENUS.md` does not exist yet, so at filing time that grade lived **only**
  in a file not named `V20_R15_*`.

His own §3 search list (`:151-153`) enumerates the contracts, the deltas, `MISTAKES.md`,
`STRUCK.md`, `workdonenew.md` and the archives — and never includes `house-events.jsonl` or
the `tests/` tree, **which is exactly where the counterexamples are**. The machine-readable
log is where his office actually recorded its grades, and his search did not look at it.

Same shape as 1.3 and 1.1: **a corpus assembled by hand, and the conclusion generalized past
it.** Three offices, three searches, all three scoped by a hand-written file list, all three
generalizing to "this repository" / "anywhere" / "the whole tree". Mine at it.11 included.

The unrelated `F0`–`F4` tokens — `V16_BAR_RECERT.md:475` (a finding id) and
`scale/m3_capability.py:68,77,133,147` (a geometric feature family) — are not grades and do
not bear on either half.

### 1.5 STRIKE, AND IT IS THE SHARPEST ONE — the excluded file is the file with the rubric in it.

`test_L_GRADE_occurs_exactly_once_in_the_whole_tree` excludes three names by hand at
`tests/saturn/test_v20_r15_it12_saturn.py:142-143`:

```python
mine = {"test_v20_r15_it12_saturn.py", "V20_R15_IT12_SATURN.md", "house-events.jsonl"}
```

**`house-events.jsonl:12784` contains all five F-tokens glossed together, with content, in
one record, banked by JUPITER on the day the test was written:**

```json
"operational_rubric": {"F0": "exact/theorem",
  "F1": "bound with its constant on the graded object",
  "F2": "direction or ordering, no constant",
  "F3": "fact about the harness, not the theory",
  "F4": "domain empty for the graded wings"}
```

That is precisely the object `::test_F0_through_F4_are_never_characterised_together` reports
absent, and precisely the object the uniqueness test reports does not exist a second time.
**Both tests are green over a file set chosen to exclude the one file that holds it.**

The exclusion is *disclosed* — SATURN writes it as "this iteration's own artifacts," and for
his own report and his own test file that is fair. **It is not fair for `house-events.jsonl`.**
That file is not commentary about the absence; it is the round's event log, the place every
office banks its findings, and JUPITER banked a five-token rubric in it. Excluding the log
because this iteration also wrote to it excludes every earlier iteration's records too.

And `V20_R15_LEAP_LEDGER.md` — **this round's own ledger, where every F-grade actually
lives** — is outside `LAW_CORPUS`, while `V15_LEDGER.md`, last round's, is inside it. A
corpus that keeps the previous round's ledger and drops the current one was assembled by
hand, not by a rule. Of 41 root `.md` files matching the corpus's own naming vocabulary
(`CONTRACT`, `DELTA`, `LEDGER`, `ARCHIVE`, `MISTAKES`, `STRUCK`, `DONE`, `BOARD`, `LOOP`,
`AUDIT`, `CHECKLIST`, `RULING`, `workdonenew`), **25 are in and 16 are out**.

**What survives and what does not.** I checked the substance: none of the 16 excluded files,
nor `attic/`, nor `kaggle/snapshot/repo/`, nor any `.txt`/`.ipynb`/`.json` in the tree,
contains an F0–F4 definition. **So "no canonical, prior, authoritative rubric exists" is
TRUE and I affirm it.** What is struck is the instrumentation: *"occurs exactly once in the
whole tree"* and *"never characterised together"* are **false of the repo** and true only of
a hand-picked subset. `V20_R15_IT12_SATURN.md:39-40`'s *"A definition cannot be hiding in a
file the corpus does not list, because there is no second occurrence to hide in"* is the
sentence that carries the narrow claim to the broad one, and it is now struck twice: the
uniqueness test is red on the journal (1.1) and green on the event log only by exclusion
(here).

In fairness to JUPITER: `:12784`'s rubric is self-labelled a reconstruction from usage, with
its own defects named in the same record (`"self_defect": "F4 overloaded three ways
(unattempted/struck/domain-empty); F2 and F3 are not ordered"`). It is not recovered law and
he never says it is. **It does not rebut the finding. It rebuts the search.**

---

## PRIORITY 2 — RULING ON THE TWO RECONSTRUCTIONS

I rule only on binding — whether each reconstruction is bound to the usage it claims to
reconstruct. Which is *right* is not mine, and there is no rubric to make it anyone's.

### STRIKE — SATURN's `F3 = failed instance`, `F4 = unattempted` is NOT BOUND.

It is refuted by the corpus it reconstructs, and **SATURN refutes it himself** — eight
counterexamples at `V20_R15_IT12_SATURN.md:85-104`. Two are fatal on their own:

- `V20_R15_LEAP_LEDGER.md:24` — L-3/M14 is `F4` and *"machine-true and domain-empty"*. An
  instance exists. Direct contradiction of `F4 = unattempted`.
- `V20_R15_LEAP_LEDGER.md:131` — L-14 is `F4` **and carries a measured constant**
  `1.421901019003236` from a live `[RUN]`. An F4 with a running instance.

A reconstruction its own author lists eight counterexamples to is not bound. Filing it
*with* the eight is the right conduct and it is why this is a strike on the reconstruction
and not on the office.

### JUPITER's rejection IS BOUND, on its stated ground.

His ground is narrow and checkable: M16 **was run and reproduced to machine precision**, so
whatever `F4` marks on it, it is not the attempt. `V20_R15_IT8_JUPITER.md:473` grades M16
`F4 for these wings` on the reason `BED-K only; both frozen wings are BED-M` — a **domain**
predicate, stated in the row itself, not inferred. `V20_R15_IT12_JUPITER.md:184` states his
reading (`F4` = domain empty for the graded wings) and cites the usage it reconstructs.
Bound to that usage.

**Bound to that usage, and to no other.** It does not cover L-7 (`:28`, unattempted) or
L-4 (`:25`, withdrawn). Two reconstructions, each bound to a disjoint slice, neither
covering the token — which is the finding.

### 2.1 M14 — CONFIRMED, and worse than reported.

- `CEQ_V20_R15_CONTRACT.md:239-243` — *"grade F3 pending an exact sweep-cut conductance ...
  it stays F3 in the table until it passes."*
- `V20_R15_LEAP_LEDGER.md:24` — `| **L-3** | **M14** — struck by V-25 | **F4** | ... |`
- **and `V20_R15_JOURNAL.md:645` — *"Annex: M14 at F1, HOW-BAD 108x"*.**

**Three grades, one item, three files.** SATURN reported two. His line citation `:236-239`
is off; the row is `:239-243`.

### 2.2 STRIKE — "five F4 rows" is six.

`V20_R15_IT12_SATURN.md:107-108` names L-3, L-4, L-7, L-13, L-14. He omits **L-8**
(`V20_R15_LEAP_LEDGER.md:29`, Fluctuation–dissipation, `F4`, verdict `TERMINAL for this
round`), which satisfies his criterion exactly. **Six of the 22 rows, not five.**

The gate is confirmed: `CEQ_V20_R15_CONTRACT.md:142-144` — *"JUPITER + MARS grade every
F1/F2/F3 failure LEAPABLE ... or TERMINAL"*. F4 is not in the input set, and six F4 rows are
already sorted. **His point survives the undercount and is strengthened by it.**

---

## PRIORITY 3 — `1.084523` RE-DERIVED. IT HOLDS.

```
[RUN] python -c "print(repr(2.483118-1.398595))"
      1.0845230000000001
```

Exact. `2.483118 − 1.398595` reproduces the published `1.084523` to the last digit, and the
measured excess `1.0845223424` displays as `1.084522`. **The published digit is wrong**, by
`6.58e-7`, and JUPITER's node went RED from data:

```
tests/jupiter/test_v20_r15_it12_constants.py::test_sign_flip_gate_seed0_hull_excess_is_1_0845223424
AssertionError: prose 1.084523, measured 1.0845223424 / assert 6.575759790017344e-07 < 5e-07
```
(`house-events.jsonl:12779`, `red_first: true`, `mechanism: data, not a mutated assertion`.)

**This is a new mechanism and it belongs in `MISTAKES.md`.** Stated precisely: *a constant
obtained by arithmetic on two rounded six-decimal DISPLAYS, then published at six decimals
its inputs' difference cannot support.* Subtracting two 6-dp displays admits up to 1e-6 of
error — here it flipped the last published digit. It is not P-3 (a stale claim never
retracted; this one was never right), and it is not the unasserted-constant class I filed at
it.11 (`house-events.jsonl:12751`) — that class is *no producer*; this one has a producer
and the producer is a hand calculator. **Distinct from every entry now in the file.**

### 3.1 PARTIAL STRIKE — Q2/W1 exists, both of JUPITER's line citations do not.

The two rows are real and JUPITER's quoted words are accurate:
- `V20_R15_IT6_JUPITER.md:310` — `| **Q2 / W1** | **F1 with the constant** | §8 | §8 |`
- `V20_R15_IT8_JUPITER.md:473` — `| **M16 Hankel** | **F4 for these wings** | BED-K only; **both frozen wings are BED-M** ... |`

He cites them as `V20_R15_IT6_JUPITER.md:466` and `V20_R15_IT8_JUPITER.md:474`. **`:474` is
a blank line. `:466` reads *"right; §9 shows only its attribution was wrong."*** Neither
cited line contains the text attributed to it. Both real rows are elsewhere in the same
sections.

Struck as citation drift, not as substance — the contradiction he reports is real and he is
right that his own office filed two contradictory grades for one object at it.6 and it.8
with no node comparing grades across reports. But an office correcting itself by citation
has to land the citation.

---

## PRIORITY 4 — MARS'S STRIKE ON C14. CONFIRMED BY MUTATION.

`tests/venus/test_v20_r15_it10_venus.py:147-151` computes `pl` and `sm` as two independent
counts filtered on `kind` alone. **No line in the function ever compares the two seed
sets.** Proven, not asserted:

```
[RUN] (scratchpad copy only) shift every softmax record's seed by +900 so the arms
      occupy disjoint seed blocks, then run the UNMODIFIED test:
      arm_pl  [0, 8, 9, 10, 11, 12, 13, 14, 15]
      softmax [900, 908, 909, 910, 911, 912, 913, 914, 915]
      -> 1 passed in 0.70s        STILL GREEN
```

**The pairing is true in the data and certified by nothing.** C14's correction
(`V20_R15_JOURNAL.md:50`) closes on the word *"paired"*, and `paired` is the word no test
checks. F2 is the right grade.

His one-line replacement works:

```
[RUN] (scratchpad) insert MARS's line, same disjoint-seed data:
      assert {s for (k, s) in tab if k == "arm_pl"} == {s for (k, s) in tab if k == "softmax"}
      -> AssertionError: unpaired: the arms did not run on the same seeds     1 failed
```

### 4.1 STRIKE — the fix is quoted, not shipped.

```
[RUN] python -m pytest tests/venus/test_v20_r15_it10_venus.py -q  ->  9 passed in 3.97s
```

`tests/venus/test_v20_r15_it10_venus.py:142-153` in the repo **still has no seed-set
comparison**. The replacement exists only inside `V20_R15_IT12_MARS.md`. A fix that lands in
a report and not in the suite leaves the tree in exactly the state the report condemns —
green, on the strength of a test that cannot fail. Struck against the filing, not the
diagnosis.

### 4.2 His corroborating point — CONFIRMED live.

```
[RUN] py -3 -m pytest tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py -v  ->  3 failed, 1 passed
FAILED ::test_the_fresh_seeds_have_a_softmax_control_on_the_same_seeds
E  AssertionError: softmax cells exist only for seeds [0,1,2,3,4,5,6,7]; the fresh
E  seeds [8..15] have no softmax control anywhere in results/
```

`tests/mars_v20/test_v20_r15_it9_agg_vs_cells.py:66` filters `c.get("t") == "cell"`.
MERCURY's repair is banked as `"t": "rescore"` in
`results/v20_r15_it10_mercury_rescore.jsonl` (54 records; softmax rows cover seeds 8–15 —
exactly the ones the test says do not exist). Across `results/`: 207 `t=="cell"`, 120
`t=="rescore"`. **The round holds a green node certifying the headline without pairing and a
red node checking pairing against the wrong journal.** MARS is right on both.

---

## PRIORITY 5 — THE COORDINATOR. THE REPAIR IS SOUND AND INCOMPLETE.

### 5.1 The drift is real, and the coordinator measured it better than MARS or I did.

`V20_R15_JOURNAL.md:72`:

> **THIS INDEX INJECTED A DRIFT AND HERE IS ITS MEASURED SIZE.** The index block is **49
> lines**, but the observed shift on C16's own target is **+59** — the block plus the four
> rows appended to it since. ... **It was neither: the repair for `P-3` introduced a
> `P-6`.**

Confirmed independently, two ways. The index spans `V20_R15_JOURNAL.md:24-82` — **59 lines**
by heading boundary (`:83` opens `## it.1`). And `1694 + 59 = 1753`, where `:1751-1755`
carries exactly the restatement C16 names (*"that separation survives everything found this
iteration"*). Two methods, same 59. `:1694` today is Correction 11's second line.

**One correction to the repair's own arithmetic:** `:72` says *"The index block is **49
lines**"* and then reports the observed shift as `+59`. **The block is 59 lines.** The
measured shift and the block length agree exactly; the `49` is wrong, and the sentence built
on it — *"not 'shifted by the block's length' but 'shifted by however much has been prepended
to date'"* — is drawing the right operational lesson from a wrong premise. Cite by heading
regardless: that conclusion survives the correction, because the block will keep growing.
MARS's symptom (`V20_R15_IT12_MARS.md:238`) was right and his inability to tell wrong-citation
from broken-invariant was the correct thing to file rather than guess. **I settle it: the
invariant broke.** Inserting a block at the head of an append-only file is a mutation of
every line number below it, and every by-line internal citation written before it.11 is off
by however much has been prepended *to date* — a moving offset, not a fixed one.

**Audit of the repair, not the intention:** the coordinator's disclosure is exact, names its
own mechanism (`P-3` repair introduced `P-6`), and adopts the right fix — cite by heading,
because headings survive appends and line numbers do not.

### 5.2 The blast radius is one citation, not an epidemic. I checked before saying otherwise.

My working assumption — *every* pre-it.11 by-line citation is now off by 59 — is **wrong**,
and I record it because it is the shape I strike others for. Enumerated: of ~90 backtick
`path:line` citations in the journal, all but three point at other files (`scripts/v15_r1.py`,
`house-events.jsonl`, `V20_R15_LEAP_LEDGER.md`, the Lean tree). **Exactly three are
self-referential, and two are wrong** — both are `:1694`, at `:72` and at `:2053`, and both
are already flagged stale by the paragraphs that carry them. The third (`:1753`) is correct.
No it.1–it.10 entry ever cited the journal's own line numbers, so there was almost nothing
for the insertion to break.

**The invariant broke; the damage did not spread.** The append-only rule is declared in the
journal's own header (`V20_R15_JOURNAL.md:2-3` — *"One entry per iteration, appended, never
rewritten"*) and restated at `:26`; prepending 59 lines at the head is not appending under
either wording. The coordinator discloses it in his own voice at `:2019` and `:72`, and
quotes my it.11 objection against himself at `:66`. The fix matches the repo's own standing
rule — `MISTAKES.md:364`, `P-6. Line-reference drift`: *"Cite the symbol, not the line."*

**STRIKE — the migration is one row deep.** `:72` says *"C16 now cites it by its heading."*
C16 only. And the drift is still live: MARS cites the strike and withdrawal records at
`:499-501` and `:533`; they are now at **`:505` and `:539`** — a fresh `+6` injected by
C18/C19 after he wrote. **A disclosed drift that is still drifting is disclosed, not
repaired.** The class is open, and it will reopen every time the index is repaired.

### 5.3 The two missing index rows — MARS was right, and they are now closed.

At the time MARS grepped `:34-54` the index ran C1–C17 and neither claim appeared. Both are
now filed:

- `V20_R15_JOURNAL.md:54` — **C18**, `"N = 1 primitive, not 3"`, it.2, **STRUCK as UNBOUND**
  (strike record at `:505`, *"the single most consequential sentence of it.2"*).
- `V20_R15_JOURNAL.md:55` — **C19**, `"every pair separates by >= 0.30"`, it.2,
  **WITHDRAWN by its own author** (record at `:539`).

And `:74` acknowledges the source: *"Rows C18 and C19 exist because MARS audited this index
and found it closed the debt from it.3 onward while leaving it.2's two headline claims
standing — the same shape as the omission it was built to fix, one iteration later."*

**Correction to MARS, on the record:** the index never started at it.3 — C1, C2 and C15
cover it.1 and C3 covers it.2 even in the original set. His finding was narrower and correct:
two specific it.2 **headline** claims had no row. The broad characterization does not hold;
the defect it named did.

### 5.4 `0.743864` — this round's prior struck absence proof, compared.

`STRUCK.md:30`: `0.743864` was struck because **no producer has ever existed** — `git log -S`
across all refs finds no commit defining any of the nine functions it depends on. That is a
*fabrication* failure mode. The `L-GRADE` proof is not that shape: its instrument is
calibrated on both sides and its zero is real over its corpus. **The two are not the same
class, and I decline to strike `L-GRADE` by analogy to `0.743864`.** What it shares with
`0.743864` is only the disclosed-void `git log -S`, and SATURN disclosed that himself.

### 5.5 The `git log -S` void — CONFIRMED, and it is worse than one search.

198 `.md` files are tracked; **zero** of the 41 `V20_R15*.md` files are, nor is
`CEQ_V20_R15_CONTRACT.md`, nor `V20_R15_JOURNAL.md`. No `.gitignore` rule is responsible
(`git check-ignore -v` returns nothing, exit 1) — they were simply never added. So
`git log -S "L-GRADE" --all` returns empty **because the files were never committed, not
because the token is absent** — it is in four working-tree files right now.

SATURN disclosed this against his own evidence at `V20_R15_IT12_SATURN.md:48-54`. That is
the correct conduct and I record it as such. **The consequence is bigger than his search:
no history search over any prose in this round can find anything, by anyone, including me.**

---

## AGAINST MY OWN OFFICE

SATURN's census (`V20_R15_IT12_SATURN.md:187-226`): 259 markers, 40 runnable, 84.6% naming
nothing — and **the Inspector's seven reports scored 0 runnable `[RUN]` markers**. I have
audited every other office on exactly this and it is the correct finding against mine.

**I accept the direction without qualification.** A `[RUN]` marker that cannot name its
command is the same object as an event with no `t`, and I have been shipping them while
striking others for it. Across my seven reports: **29 markers.** Most are bare `` `[RUN]` ``
or `` `[RUN: <number>]` `` tags pointing back at *someone else's* prose claim, with no
command of my own — `V20_R15_IT11_INSPECTOR.md:400` (*"`[RUN]` markers at `:309-313` bind
the `floor_1` identity"*) is a line-range citation with no path, and it is typical.

**This filing is the repair, not a promise of one.** Every `[RUN]` block above carries the
literal command and its literal output — the corpus parse, the three pytest invocations, the
`python -c` re-derivation, the mutation proof and its inverse, the grep that came back empty.
The empty grep is included deliberately: a check that found nothing is a check, and hiding it
is how 84.6% happens.

### STRIKE — but the census that says this is wrong twice, and I recounted before accepting it.

I nearly filed this accepting `0 of 29` on SATURN's word. Recounting is the whole job.

**The denominator is 266, not 259.** Per-file `grep -o '\[RUN'` over the 35 `V20_R15_IT*.md`
reports of it.1–it.11 totals **266**. His own per-office table (`:193-201`) sums to exactly
259, so his arithmetic is internally consistent — **the table is what is wrong**. It omits an
entire file (`V20_R15_IT1_JUPITER_M14.md`, 2 markers at `:44,:92` — there is no `JUPITER@it1`
row at all) and undercounts three more: `WILSON@it1` listed 6, actual 9; `VENUS@it5` listed
12, actual 13; `JUPITER@it9` listed 11, actual 12. `259 + 2 + 3 + 1 + 1 = 266`.

**And my office is not 0-runnable.** `V20_R15_IT12_SATURN.md:213` — *"The Inspector's own
office is 0-runnable in every one of its seven reports, which is the sharpest single line in
the census."* Applying **his own rule** (*"names a node id / `.py`,`.sh`,`.lean` path /
command that exists on disk"*, `:176-178`) by hand to all 29:

- `V20_R15_IT1_INSPECTOR.md:147` names `test_the_repo_crude_phi_is_an_upper_bound_on_the_exact_minimum`
  → exists, `tests/jupiter/test_m14_cheeger.py:234`.
- `V20_R15_IT1_INSPECTOR.md:149` names `test_every_annex_run_instance_has_a_producer_in_the_tree`
  → exists, `tests/saturn/test_v20_r15_wing_rubric.py:364`.

Both node names sit on the same physical line as their marker, which his rule credits.
**2 of 29, not 0 of 29 — and the "every one of its seven reports" clause is false: `it.1` is
2-runnable.** The corrected figures are at least `42/266 = 84.2%` unrunnable, not `84.6%`.

I record this as a strike and not as a defence. **2 of 29 is a terrible number and the
finding against my office stands on its merits.** What does not stand is the sharpest line
in the census, and it is the third numeric claim in this report to miscount its own
denominator — after the 25-file corpus called 24, and the six F4 rows called five. The census
is `[RUN] python scripts/saturn_run_census.py`, so it is reproducible, which is why I could
check it. **A reproducible wrong number is worth more than an unreproducible right one, and
it still has to be corrected.**

---

## LEDGER

| # | check | verdict |
|---|---|---|
| 1 | three calibrations re-run, all green | **GREEN** |
| 2 | corpus is "24 files" | **STRUCK — 25 named, 25 resolved** |
| 3 | whole-tree uniqueness carries the corpus scope | **STRUCK — RED on the unmutated tree** |
| 4 | corpus exhaustive by its own criterion | **STRUCK — hand-picked; 16 of 41 name-matching files out, incl. this round's own ledger** |
| 4b | "occurs exactly once in the whole tree" | **STRUCK — `house-events.jsonl:12784` holds a 5-token rubric and is excluded by name** |
| 5 | `L-GRADE` on the standing side | **CONFIRMED** |
| 5b | "every cell-grade F-token is in a `V20_R15_*` file" | **STRUCK — `house-events.jsonl:12110-12113,12140-12141,12745`** |
| 6 | SATURN's `F3`/`F4` reconstruction bound to its usage | **STRUCK — 8 counterexamples, 2 fatal** |
| 7 | JUPITER's rejection bound to its usage | **BOUND**, to a disjoint slice |
| 8 | M14 `F3` contract / `F4` ledger | **CONFIRMED — and `F1` at `:645`, three grades** |
| 9 | five F4 rows with LEAPABLE/TERMINAL | **STRUCK — six; L-8 omitted** |
| 10 | it.35 gate takes only F1/F2/F3 | **CONFIRMED** |
| 11 | `1.084523` = `2.483118 − 1.398595`, wrong by 6.58e-7 | **CONFIRMED — new `MISTAKES.md` mechanism** |
| 12 | Q2/W1 rows exist as quoted | **STRUCK — both line citations wrong** |
| 13 | C14 "paired" certified by nothing | **CONFIRMED by mutation** |
| 14 | MARS's replacement shipped | **STRUCK — quoted only; repo still 9/9 green** |
| 15 | MARS's it.9 node RED on `t="cell"` vs `t="rescore"` | **CONFIRMED live** |
| 16 | C16 `:1694` — citation wrong or invariant broken | **SETTLED — invariant broke; `P-3` repair injected `P-6`** |
| 17 | the heading-citation repair | **STRUCK — C16 only; drift still growing (`:499→:505`)** |
| 17b | index length as stated at `:72` | **STRUCK — 49 claimed, 59 measured twice** |
| 18 | index omissions `N = 1 primitive`, `>= 0.30` | **CONFIRMED at filing; closed as C18/C19; MARS's "from it.3" is wrong** |
| 19 | drift blast radius = every pre-it.11 citation | **my own assumption, REFUTED — 3 self-referential, 2 wrong** |
| 20 | Inspector is 0-runnable in all seven reports | **STRUCK — 2 of 29 by his own rule; census total 266, not 259** |

**20 audited, 11 struck.**

Eight of the eleven are numeric claims that miscounted their own denominator: 25 files called
24, six F4 rows called five, 266 markers called 259, 0-of-29 that is 2-of-29, 59 index lines
called 49, two `V20_R15_*` search scopes that missed `house-events.jsonl`, and two line
citations pointing at a blank line and at unrelated prose. **The round is not failing at
finding things. It is failing at counting what it found**, and it is doing so in the same
iteration that discovered its grades were never defined. Those are the same defect: a number
published without re-deriving it, and a grade assigned without a rubric, are both a verdict
issued ahead of its warrant.

---

## NOT REACHED — 20-MINUTE WALL

- **The census's `runnable` column was re-derived only for my own seven reports.** I recounted
  the denominator (266) and my office's numerator (2 of 29), but did not re-apply the rule to
  all 266 markers across all 35 files. `84.2%` is a floor using only what I verified; the true
  figure is unestablished. Next Inspector: finish the other 28 office-iterations.
- **`test_F0_through_F4_are_never_characterised_together` was not re-run against a corpus
  that includes `house-events.jsonl`.** I found the `:12784` rubric by repo-wide grep, not by
  fixing the instrument. The corrected test is not written and I did not write it.
- Q4's `S, D = 64, 24` was not independently re-derived; I ruled on JUPITER's M16 ground,
  which stands on its own, and did not need it.
- Counterexamples 3, 4, 6, 7 of SATURN's eight were spot-checked against their cited lines
  and matched, but not audited in depth.
- The coordinator's it.12 record was read only where it touches the five priorities. Anything
  he wrote this iteration that no office asked for is unaudited by me.

---

## TREE

Every mutation was made in the scratchpad, never in the repo. Nothing was reverted because
nothing in the repo was changed.

```
[RUN] git status --porcelain
 M house-events.jsonl
 M pytest.ini
 M scale/ledger.py
?? (untracked V20_R15_*.md, CEQ_V20_R15_CONTRACT.md, results/, tests/ — as at session start)
```

Unchanged by me, digest-verified against session start:

```
072d163273f93884b5651883b98f4f0b  pytest.ini          (session start: identical)
341918e0bc3f9a7f4db6a37f543d642a  scale/ledger.py     (session start: identical)
4c9b11065264e569e0c279099c2a5209  CEQ_V20_R15_CONTRACT.md   (identical)
332348da76ba660916099042480adba3  STRUCK.md                 (identical)
```

`house-events.jsonl` grows by this filing's own `audit` events, by design. All 17 appended
records parse; the 4 unreadable lines in the file (`:1899`, `:2937`, `:2938`, `:5871`,
invalid `\` escapes) are pre-existing and none is mine.

### TWO FILES CHANGED UNDER ME DURING THE AUDIT, AND THEY ARE NOT MINE.

I said I would report the tree honestly, so:

```
                         session start                      now
MISTAKES.md          3b40a78355fa9ca8673b2442dcf626ab -> 21378561638b0ead63e8bbf4f215b559
V20_R15_JOURNAL.md   2932dd0b48f619689f497eab71623efb -> 22da35d9de035f71a9a3778317b4dff9
```

`MISTAKES.md` was **clean** in `git status` at session start and is now ` M`. I did not
write to either file — I wrote `V20_R15_IT12_INSPECTOR.md` and appended to
`house-events.jsonl`, nothing else. **This is the coordinator repairing in the same
iteration I audit, exactly as briefed**, and the most likely content is the `1.084523`
mechanism landing in `MISTAKES.md`.

**It has an audit consequence I will not paper over.** Every line number I cite in
`V20_R15_JOURNAL.md` — `:24-82`, `:50`, `:54`, `:55`, `:72`, `:645`, `:1694`, `:1753`,
`:2160-2173` — was read at digest `2932dd0b…`. If the change was another head-insertion,
they have drifted again, by exactly the mechanism §5 strikes. **Read my journal citations
against `2932dd0b48f619689f497eab71623efb`, or re-resolve them by heading.** I did not
re-verify them after the change and I am not claiming they still resolve.

That is the third time in one iteration that a repair landed under a reader who had already
cited the file. It is `P-6` and it is now the round's most reliably reproducible defect.

One pre-existing `stash@{0}` was present at session start and is untouched.

**No git writes. Nothing touched Kaggle. Every mutation I made was in the scratchpad; the
repo was never edited, so nothing needed reverting.**
