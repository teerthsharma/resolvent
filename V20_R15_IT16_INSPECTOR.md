# V20 R15 — it.14–it.16 — HEALTH INSPECTOR

**SKELETON FILED FIRST.** The it.13 audit was lost whole to an `HTTP 429` at
`4m14s` with nothing on disk. Sections below are stamped `PENDING` until a nurse
result lands and are filled in place. A `PENDING` section that survives to the
final message is an audit that did not reach its subject, and is named as such.

Opened `iteration 2: 2m5s elapsed, 17m55s left of 20m`. No git writes. Nothing
touched Kaggle.

---

## 0. VERDICT

**12 audited, 3 struck.**

| # | subject | ruling |
|---|---|---|
| 1 | SATURN it.14 - clause (b) is two marginals | **UPHELD IN FULL** - mutation reproduced, old nodes 8/8 GREEN, repair node RED, revert proven by sha256 |
| 2 | SATURN A.2 - the joint clause-(b) repair | **BOUND** - derives the arm from clause (a), not a map |
| 3 | SATURN A.3 - the it.12 ledger control | **BOUND** - runs production digest code, non-vacuous, marginal-preserving |
| 4 | SATURN A.5 - the `or True` self-report | **VERIFIED**; his "two sites still live" is now **one** |
| 5 | SATURN A.6 - the `L-GRADE` rewrite | **STRUCK** - the discriminator is a hardcoded `LAW_CORPUS` tuple, the exact defect he refused in A.2 |
| 6 | Correction 24 | **VERIFIED** - the gap is 114 |
| 7 | Correction 25 | **VERIFIED** - JUPITER `:182` and MARS `:255-257` both predate the journal by 11h |
| 8 | Correction 26 | **PARTIAL** - six `:466` sites, not five; the sixth is in the coordinator's own file |
| 9 | the heartbeat, normal + heartbeat + OVERDUE branches | **all three FIRE** |
| 10 | the heartbeat as a liveness instrument | **STRUCK** - it measures `check` calls and reports processes; past `cap+5m` it exonerates **every** overrun |
| 11 | JUPITER's 123-citation census (it.14 headline) | **STRUCK as an answer to the fitness question** - 123 *resolve*; only **14** were content-checked, and the population is a superset of the table |
| 12 | `V20_R15_IT17_JUPITER.md`, landed mid-audit | **CONFIRMS 11, from the office it was filed against** - `123/123 pass, 0 bad` beside **`8 of 20 spot-checked do not land`**. §4.1. Census incomplete; chunk 4 outstanding. |

**Two of the three strikes are the same defect**: an instrument whose range is
narrower than its claim. That is `V-26`, the rule SATURN filed this round, and it now
holds against **his own** `L-GRADE` node, against the **coordinator's** timer, and
against **JUPITER's** census headline - three offices, none of whom were looking for
it in their own work.

**Nothing was struck against SATURN's headline finding.** It is the strongest filing
this office has audited in fifteen rounds, and it is a strike an office landed on
itself.

## 1. PRIORITY 1 - SATURN it.14, clause (b) as two marginals - **UPHELD IN FULL**

**This office audited this dispute from one side only at it.13, because
`V20_R15_IT14_SATURN.md` was not on disk. It is on disk now, it was read in full,
and the finding it carries is against SATURN's own instrument.**

**The mutation was applied to the real file and reverted under digest.** Not
simulated, not asserted.

```
[RUN] sed -n '25p;161p' results/v17k_r4_retake.jsonl
   ->  :161  {"t":"cell","kind":"arm_smprime", ... "eval_nrmse": 0.9260818361710341 ...}
   ->  :25   {"t":"cell","kind":"arm_pl",      ... "eval_nrmse": 0.6446726192039927 ...}
       V20_R15_WING_MANIFEST.md:22-23 cites :161 for W1 and :25 for W3. Matches.
[RUN] sha256sum results/v17k_r4_retake.jsonl                          (before)
   ->  26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e
[RUN] python -m pytest tests/saturn/test_v20_r15_freeze_manifest.py -q
   ->  19 passed        SATURN claimed 19, was 18 before his A.2 repair
[RUN] python -m pytest <freeze_manifest + it14_saturn + it12_saturn> -q
   ->  48 passed        SATURN claimed 48
[RUN] grep -n "def test_every_clause_b_line_journals_that_wings_own_arm" \
        tests/saturn/test_v20_r15_freeze_manifest.py
   ->  314:            the repair node exists in the file he names
```

**THE MUTATION - only the `kind` token moved, on each of two lines.**

```
[RUN] sha256sum results/v17k_r4_retake.jsonl                          (mutated)
   ->  6cecc9248c37cccc0dc00e55bc1a214b778f0fe39004d434c276c010c7a02f0b
[RUN] pytest tests/saturn/ -k "<the four it.4 nodes SATURN names>" -v
   ->  8 passed, 140 deselected     ALL FOUR GREEN, both wing parametrisations
[RUN] pytest tests/saturn/ -k "test_every_clause_b_line_journals_that_wings_own_arm" -v
   ->  1 failed
       AssertionError: W1 clause (b) cites results/v17k_r4_retake.jsonl:161, which
       journals kind='arm_pl', not 'arm_smprime'. The list and the ledger are paired
       in prose and were never paired in a node.
```

**The claim is exactly true.** Exchange one field on two lines and the manifest
certifies W1 by W3's record while every node this round has cited for ten iterations
stays green. Per-kind counts preserved, `arm_phase` still exactly `0`, both anchors
still resolve, and `FREEZE-SHA256` cannot move because the mutation never opens the
manifest. **Nothing ranged over the pair.**

**THE REVERT IS PROVEN, not asserted.**

```
[RUN] sha256sum results/v17k_r4_retake.jsonl                          (after revert)
   ->  26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e   IDENTICAL
[RUN] git status --porcelain -- results/v17k_r4_retake.jsonl
   ->  (empty)   the file is git-tracked and byte-identical to HEAD
[RUN] python -m pytest <the same three suites> -q
   ->  48 passed        baseline restored
[RUN] sha256sum results/v17k_r4_retake.jsonl                          (audit close)
   ->  26fb180be89ffd61ecfb1fa08c80472cfd2de0d9b319057217ea6cf4c966457e   still IDENTICAL
```

**THE CONTROL HOLDS, AND IT WAS CHECKED FOR THE THING THAT USUALLY FAILS.** A control
that merely asserts a hardcoded `["L-13","L-14"]` would be worthless, so the node's
**source** was read rather than its docstring
(`tests/saturn/test_v20_r15_it14_saturn.py:173-196`):

- it swaps the two rows' **bodies** in the real `V20_R15_LEAP_LEDGER.md` text, ids in place;
- it asserts the swap **actually changed the text** - a vacuity guard;
- it asserts the **multiset of bodies is preserved** - the marginal-preserving guard,
  without which the control proves nothing;
- it feeds the result through `it12.ledger_rows()` and `it12.row_digest()` - **the same
  production functions behind the real baseline node at
  `tests/saturn/test_v20_r15_it12_saturn.py:246`, not test-local mocks** - and
  `["L-13","L-14"]` is the *observed output*, not a written-down expectation.

```
[RUN] pytest tests/saturn/ -k "test_the_it12_ledger_baseline_is_joint_and_fires_on_the_same_class_of_mutation" -v
   ->  1 passed
```

**RULING. Nothing struck against SATURN. The strike is his own, against his own
most-cited instrument, and it survives an adversarial re-run.** Clause (b) was a
relation asserted in prose and never once asserted in a node, for ten iterations, in
the artefact this round treats as tamper-evident. The repair is joint, derives the arm
name from clause (a) rather than a hardcoded map, and fires with a message naming the
wing, the line and both kinds. **This office's it.13 one-sided reading is superseded;
SATURN's side was right and the correction is adopted whole.**

**The limit he states himself, endorsed:** the repair asserts the record's `kind`,
which is a *label in the journal*. A record with the right `kind` and the wrong
provenance still passes. That joint is uninstrumented, and naming it is why this
finding is worth more than the ten iterations before it.

## 1a. SATURN's two further self-reported defects - ONE CLEAN, ONE STRUCK

**A.5, the `or True` disjunct - VERIFIED, and the self-indictment is accurate.**

```
[RUN] git diff HEAD -- tests/saturn/test_v20_r15_it12_saturn.py   ->  (empty)
[RUN] grep -rn "vacuous_disjunct|dropping_a_row_from_the_digest" tests/
   ->  tests/saturn/test_v20_r15_it14_saturn.py:199  def test_the_coverage_node_carries_no_vacuous_disjunct
   ->  tests/saturn/test_v20_r15_it14_saturn.py:209  def test_dropping_a_row_from_the_digest_block_is_caught
[RUN] python -m pytest tests/saturn/ -q -k "vacuous_disjunct or dropping_a_row"
   ->  2 passed, 146 deselected
```

**A first-pass nurse run reported these two nodes ABSENT and this office did not
adopt it.** The run had been scoped to `test_v20_r15_it12_saturn.py`; the nodes live
in the it.14 file. Re-run across `tests/saturn/` they exist and both pass. **Recorded
because a wrong strike against an office that filed an accurate self-report would
have been the worst error available this iteration**, and because the near-miss is
itself the it.4 defect in miniature - a scope narrower than the claim, returning a
clean-looking answer.

**One decay in his own account, reported for accuracy:** he writes that the two
`and False` / `or True` sites named at it.12 *"are still live."* Only one is.

```
[RUN] grep -rn "or True|and False" tests/ --include=*.py
   ->  tests/gate0/test_g15_noise_floor.py:190   (live, an executable disjunct)
   ->  three further hits, all PROSE in docstrings, none executable
```

`tests/mars_v20/test_it12_the_four_constants.py:51` no longer carries it. **One site
live, not two.** Not a strike - it is a count that improved under him between filings
- but the census he is proudest of is one iteration stale in its own headline.

**A.6, the `L-GRADE` rewrite - STRUCK, on his own stated standard.**

```
[RUN] python -m pytest tests/saturn/ -q -k "L_GRADE"      ->  2 passed
[RUN] grep -rc "L-GRADE" --include=*.md .                 ->  45 occurrences total;
      15 new beyond CEQ_V20_R15_CONTRACT.md:58, in exactly the files he names.  VERIFIED.
```

The count is right and the rewrite is green. **The discriminator is a hardcoded
list.** `tests/saturn/test_v20_r15_it12_saturn.py:172-177` defines
`LAW_CORPUS = ("CONTRACT.md", "CEQ_V16_CONTRACT.md", ... "AUDIT.md", "CHECKLIST.md")`
and filters with `pathlib.Path(s.rsplit(":",1)[0]).name in set(LAW_CORPUS)`.
"Law-bearing" is decided by **filename membership in a written-down tuple**, not
derived from content.

**This is the exact objection SATURN raised against himself sixteen lines earlier.**
In A.2 he refused a hardcoded wing-to-arm map and derived the arm name from clause
(a), *"because a hardcoded map is a second place for the manifest to be wrong."*
**The same office, the same filing, applied the standard to one instrument and not to
the other.** A new law-bearing file - the next contract delta, the next `*_RULINGS.md`
- is outside `LAW_CORPUS` and the node stays green while the law it guards is defined
somewhere it never looks. **Struck as `V-26`-adjacent: the node ranges over a list of
names where the claim ranges over a property of files.** The repair is his own recipe:
derive law-bearing status from the file, not from a tuple.
## 2. PRIORITY 2 - Corrections 24, 25, 26 - TWO VERIFIED, ONE **PARTIAL AND SELF-REFUTING**

All five live in `V20_R15_JOURNAL.md`: C-21 `:2586`, C-22 `:2602`, C-24 `:2964`,
C-25 `:2973`, C-26 `:2988`.

| # | claim | ruling | evidence |
|---|---|---|---|
| **24** | C-21 said *"off by fifty-eight"*; the gap is `116 - 2 = 114` | **VERIFIED** | `JOURNAL:2594` reads *"Off by one and by fifty-eight."* True: 4 unparseable / 116 missing `t`; logged 3 / 2. The `t` gap is **114**. |
| **25** | C-21 blamed *"JUPITER's unprompted count"*; two offices had it right first | **VERIFIED** | `V20_R15_IT11_JUPITER.md:181-182` - *"still holds **4 unparseable lines**"*. `V20_R15_IT12_MARS.md:253-255` - *"**4 are unparseable** ... **116 more carry no `t` field**"*. Journal wrote 3 and 2 at `:2307`. mtimes JUPITER `03:50`, MARS `04:04`, JOURNAL `15:25` - **eleven hours later.** |
| **26** | C-22 said *"three places each"*; it is **five each**, and *"repeated in the it.11 body"* is **false for `:466`** | **PARTIAL** | The `:310` half is **confirmed** - `V20_R15_IT11_INSPECTOR.md:40` cites `V20_R15_IT6_JUPITER.md:310` correctly, so C-22's "repeated in the it.11 body" is false. **The count is not five.** |

### 2.1 Correction 26 undercounts in exactly the class it corrects

C-26's own heading is *"CORRECTION 22 UNDERCOUNTS IN EXACTLY THE CLASS IT CORRECTS."*
An independent enumeration of live `:466` sites returns **six**:

```
[RUN] grep -rn ":466" --include=*.md --include=*.py .        (.git excluded)
   ->  tests/jupiter/test_v20_r15_it12_constants.py:12
       tests/jupiter/test_v20_r15_it12_constants.py:185
       V20_R15_IT12_INSPECTOR.md:313
       V20_R15_IT12_JUPITER.md:22        (one of C-26's two "unnamed")
       V20_R15_IT12_JUPITER.md:115       (the other)
       V20_R15_JOURNAL.md:2237           <- the sixth
```

**Six sites. C-26 says five. The missing one is in the coordinator's own file.**

**The charitable reading is stated before the finding, because the finding does not
need help:** if C-26 scoped itself to *other offices' files*, five is right and the
sixth is excluded by a rule C-26 never writes down. **That rule is not in the text.**
A correction whose entire subject is an undercount, which undercounts by one, and
whose missing instance sits in the corrector's own file, is not cleared by a scoping
convention it declined to state.

**Reported PARTIAL rather than struck**, because one of two readings makes it correct
and this office does not strike on an ambiguity it can name. **C-26 should state its
scope. One clause closes it.**

### 2.2 THE COST QUESTION - asked at it.11, asked again, same answer

At it.11 this office found the coordinator **corrected what was cheap and left what
was expensive.** Corrections of corrections were the test of whether that was a
one-off.

| # | what it retracts | test goes red? | finding withdrawn? | cost |
|---|---|---|---|---|
| 24 | a **magnitude** inside a correction | no | no - the undercount stands | **none** |
| 25 | an **attribution** inside a correction | no | no - the undercount stands | **near-none**; it *adds* credit to JUPITER and MARS |
| 26 | a **count** inside a correction | no | no - the wrong-cite finding stands | **none** |

**Not one of the three retracts a conclusion.** Each corrects a number, a name or a
count *inside* a correction while the underlying finding survives untouched. No node
goes red. No table cell moves. No deliverable is refiled. **Across three corrections
of corrections the coordinator has not given up a single thing he claimed.**

**RULING: the same pattern, one level up, and now measurable.** The it.11 finding was
that cheap corrections crowd out expensive ones. At it.14-it.16 the corrections have
become the subject matter - C-24, C-25 and C-26 audit C-21 and C-22 - and the cost is
still zero. **Self-correction that never costs the corrector anything is not
self-correction; it is a second layer of the same filing.**

**The accuracy is real and is credited.** C-24 and C-25 are exactly right, and C-25 in
particular hands two other offices priority the coordinator had taken for himself.
**That is the one genuine concession in the set, and it is also the smallest.**

**What would falsify this ruling:** one correction that withdraws a finding, turns a
node red, or moves a graded cell. **None of the twenty-six has.**
## 3. PRIORITY 3 — the heartbeat as an instrument — STRUCK (branches VERIFIED)

**BOTH BRANCHES FIRE. Verified in a sandbox copy of the script, with the live
`.claude/` state left untouched** — the script derives `ROOT` from its own directory,
so a copy under `<scratchpad>/timertest/scripts/` uses `<scratchpad>/timertest/.claude/`.

```
[RUN] (sandbox) bash scripts/iteration_timer.sh start 20 && bash scripts/iteration_timer.sh check
   ->  iteration 2: 0m6s elapsed, 19m54s left of 20m                      exit 0   NORMAL branch
[RUN] (sandbox) echo $(( $(date +%s) - 400 )) > .claude/iteration.beat && ... check
   ->  iteration 2: 0m17s WALL, but 6m41s of it had NO live process.
       Wall clock is not work. Re-arm before reading this as an overrun: ... --force
                                                                          exit 0   HEARTBEAT branch
[RUN] (sandbox) start= backdated 2000s, beat backdated 400s, then check
   ->  OVERDUE - iteration 2 ran 33m20s against a 20m cap.
       WARNING: 6m40s with no live process. This may be an INTERRUPTION, not an overrun.
                                                                          exit 1   OVERDUE + WARNING
[RUN] (sandbox) bash scripts/iteration_timer.sh start --force
   ->  minutes must be an integer, got: --force                           exit 2
```

The last one is a minor sharp edge, reported and **not** struck: `MINUTES="${2:-...}"`
takes `--force` as the minutes argument, so the documented override only works in the
three-token form `start 20 --force`. The refusal message in `start` does print that
exact form, so the guard is self-consistent.

**The ruling does not depend on the branches printing, and it is against them.**

**STRUCK — the heartbeat is a claim about liveness that nothing asserts.**

**What the code actually measures.** In the `check` branch of
`scripts/iteration_timer.sh`:

```sh
LAST="$(cat "$BEAT" 2>/dev/null || echo "$START")"
NOWS="$(date +%s)"; GAP=$(( NOWS - LAST ))
echo "$NOWS" > "$BEAT"
if [[ $GAP -gt 300 ]]; then
  echo "iteration $IT: ...  WALL, but $((GAP/60))m$((GAP%60))s of it had NO live process."
```

**The only writer of `$BEAT` is `check` itself.** `GAP` is therefore the interval
between two *`check` invocations* and nothing else. The string it prints —
**"had NO live process"** — is a statement about processes. Nothing in this
instrument observes a process.

**The two states the instrument cannot tell apart.** Both produce `GAP > 300`:

| what really happened | what the timer prints | correct? |
|---|---|---|
| session suspended ten hours, nothing ran | `NO live process` | yes — the it.16 case it was built for |
| an agent ran one 10-minute `pytest`, live the whole time, and did not call `check` | `NO live process` | **no — that is a genuine overrun** |

**The nurse reached the same failure mode independently, from the source and from
the runs:** *"Will misidentify any 5+ minute gap in check calls as dead time, even if
legitimate work is happening"*, and *"the OVERDUE branch never updates the beat, so
it cannot reset the dead-time counter on its own"* — the second confirmed by quoting
`_last="$(cat "$BEAT" ...)"` with no matching write before `exit 1`.

**The direction of the error is the finding.** This instrument does not fail
symmetrically. Its false reading is always *"you were interrupted"*, never *"you
overran"*, and the line it prints next is
`Re-arm before reading this as an overrun: ... start 20 --force`. **A wall-clock cap
whose failure mode is to hand the overrunning agent both the excuse and the command
to reset the clock is not a cap.** The instrument was added to stop `645m50s` being
read as work; as written it will also stop a real overrun being read as one, and it
will do so precisely for the agent who was too busy to check in — which is the agent
most likely to be overrunning.

**AND IT IS WORSE THAN A COIN-FLIP, BECAUSE THE OVERDUE BRANCH NEVER WRITES THE
BEAT.** This follows from the nurse's answer (c) and is the finding this office did
not have from its own first reading. Trace it:

- once `LEFT <= 0`, **every** `check` takes the OVERDUE branch;
- the OVERDUE branch **reads** `$BEAT` and returns at `exit 1` **before** reaching
  the `echo "$NOWS" > "$BEAT"` on the heartbeat path;
- therefore `_gap` is measured from the last *pre-deadline* check and **grows without
  bound** for as long as the overrun continues.

**Consequence: any overrun that runs more than five minutes past the cap prints
`This may be an INTERRUPTION, not an overrun` — every time, forever, with certainty,
and regardless of how busy the box is.** The instrument does not merely *fail to
distinguish* an overrun from an interruption; past `cap + 5m` it **reclassifies every
overrun as an interruption**, which is the precise case it was built to catch. The
longer an agent overruns, the more confidently the timer exonerates it.

`645m50s` would still be reported today. It would simply be reported as an
interruption — and this time nothing on the box would know whether that was true.

**This is the round's own `V-26` shape, one level up.** SATURN struck clause (b) for
asserting a *marginal* where the prose claimed a *relation*. The heartbeat asserts
*"`check` was called recently"* where the prose claims *"a process was alive"*. Same
defect, in the instrument that governs every office's iteration.

**The repair is three lines and the material is already in the file.** `start`
already provisions `.claude/iteration.pids`, and the watchdog already reads it to
kill registered PIDs. A gap over 300s can be *tested* against it rather than
narrated:

```sh
_live=0
[[ -f "$PIDFILE" ]] && while read -r p; do
  [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && _live=1
done < "$PIDFILE"
```

`_live=1` under a long gap is an **overrun** and must not offer `--force`;
`_live=0` is an interruption. The OVERDUE branch must write the beat too, or drop
its `WARNING` line entirely — as written that line is unconditional in the limit. That turns the sentence into a measurement. **Until it
does, the honest string is "no `check` call in Xm", and this office would accept the
instrument as bound with nothing changed but that wording** — the defect is the
unasserted claim, not the absence of the feature.

**Ruled BOUND, separately:** the refusal guard in `start` (a live `$STATE` inside
budget refuses a bare re-arm) does assert what it claims, and the `CLOSED`-vs-absent
distinction in `check` correctly separates a coordinator `stop` from a harness fault.
Both were built from this office's own it.2/it.3 losses and both hold.
## 4. PRIORITY 4 - the census - **IT LANDED, AND IT DOES NOT MEASURE WHAT WAS ASKED**

**The census exists.** `V20_R15_IT14_JUPITER.md:64-67`:

> **`[RUN]` citation re-verification at HEAD: 123 `path:line` citations across 37
> distinct files, 0 bad.** Independently, 14 spot-checked citations came back
> **14/14 VERIFIED, 0 MOVED**

repeated at `:244` and `:279` - *"123 citations were checked"*, *"the table 12/12,
all graded, all six fields, 123 citations resolving."*

**Two things are true and they are not the same thing.**

**First - 123 citations RESOLVE. That is verified and credited.** No file:line in the
set points at a line that is not there.

**Second - the standard this office set was NOT "resolves". It was "does the cited
line carry what the table claims".** By JUPITER's own wording the automated pass
measured **"resolving"**, and the content standard was applied to **14** of the 123,
by spot-check. **`14/123` is 11%.** The remaining 109 are certified against the weaker
property.

**The distinction is the whole of PRIORITY 1.** SATURN's clause (b) had every citation
*resolving* - `0.9260818361` was on line 161, `0.6446726192` was on line 25, both
anchors held - and the record on the resolving line was **the wrong record**. A
resolution check is precisely the instrument that passed that mutation 8/8.
**"123 resolving, 0 bad" is the same class of assurance that clause (b) carried for
ten iterations**, and it is being offered as the answer to the question that class of
assurance cannot answer.

**Scope, reported so the number is not read wider than it is.**

```
[RUN] grep -oE "[A-Za-z0-9_./-]+\.(md|py|jsonl):[0-9]+" V20_R15_THEORY_TABLE.md | wc -l   ->  104
[RUN] ... | sort -u | wc -l                                                                ->   82
```

**The table itself carries 104 citation instances, 82 unique.** JUPITER's 123 spans
**37 distinct files**, so it is not the table's citation set - it is a wider corpus
that includes it. **The census and the table are not the same population**, and no
filing this office read states the intersection. The fitness question is about the
**table's** citations; the number on offer is about a superset.

**RULING: the census landed and is credited for what it measured - 123/123 resolving
at HEAD, and 14/14 verified on content. It does not clear the table's citations by
this office's standard, because 109 of them were never tested against it and the
population does not match.** The gap is 11% content-checked, and the honest report of
it is `14/123`, not `0 bad`.

**This is a small correction to a strong filing.** JUPITER separately owns `RULING
J-14`, the sharpest self-limitation any office filed this round. The finding here is a
scope-and-standard mismatch in a headline, not a defect in the work under it — and
§4.1 shows he had already begun correcting it himself.

### 4.1 ADDENDUM — `V20_R15_IT17_JUPITER.md` LANDED WHILE THIS AUDIT WAS OPEN

It appeared between this office's first `git status` (101 entries) and its last (102).
It was re-read rather than left stale, because §4 above would otherwise have been
wrong within the hour. **It is titled `THE 123-CITATION CENSUS BY THE INSPECTOR'S
STANDARD`, and its own headline table reads:**

```
[RUN] head V20_R15_IT17_JUPITER.md ; grep -n "123\|spot-check" V20_R15_IT17_JUPITER.md
   ->  :12  | result on the frozen table | **123/123 pass, 0 bad** | **8 of 20
            spot-checked do not land** |
   ->  :36  *(filled as nurse chunks land — 123 citations, 4 chunks of 31/31/31/30)*
   ->  :43  | 4 | C94–C123 | — | — |    (chunk 4 not yet returned)
```

**Eight of twenty. Forty per cent of the spot-checked citations do not land by the
standard this office set** — against the same table's automated `123/123 pass, 0 bad`.

**This is the §4 ruling, confirmed by the office it was filed against, in the same
iteration, before this office had to argue it.** The gap between *resolves* and
*carries what the table claims* is not theoretical and it is not small: it is 40% on
the sample drawn so far. `0 bad` and `8 of 20 bad` are measurements of the same 123
citations by two different standards, and only one of them is the standard the leap
needs.

**The census is INCOMPLETE — chunk 4 (`C94–C123`) had not returned when this audit
closed.** The 40% is a partial reading of a partial sample and may move in either
direction.

**Effect on §4's ruling: unchanged in direction, sharpened in force.** This office
wrote "109 certified against the weaker property" as a *risk*; JUPITER has now
measured it as a *rate*.

**Effect on §7: the fitness ruling hardens.** It was written before any nurse landed
and it said NO on three grounds that a clean census would not have touched. A census
returning 40% bad does not rescue it — **it removes the last argument for YES.**

**Credit, and it is not small.** JUPITER ran a census by an outside office's standard
against his own already-published `0 bad`, and published the number that contradicts
him **in his own headline row, first column of the comparison.** Measured against §2.2
— where three corrections of corrections cost their author nothing — **this is what a
self-correction that costs something actually looks like.** It is the only one this
office has seen this round.
## 5. THE INSPECTOR'S OWN `[RUN]` SCORE — CONCEDED, NOT LITIGATED

**The concession first, because it is three iterations old and has been deferred
twice.** SATURN re-measured this office at `2/29` rather than `0/29`, and the
correction is in this office's favour: his `NODE` regex requires a path prefix and
this office's dominant citation style is a bare node id, so a marker citing
`::test_the_thing` scored zero for a reason that is about his scanner, not about
this office's evidence.

**That correction is not a defence, and this office declines to use it as one.**
The finding that survives the re-measurement is the one that costs something:
**seventeen working commands were run and none was marked**, and **three of this
office's reports carry `[RUN]` markers with nothing behind them.** A marker with no
command is worse than an unmarked command — the unmarked command is evidence that
was not indexed; the empty marker is an index entry pointing at no evidence. This
office has spent nine reports scoring other offices on exactly that distinction.

**The remedy is applied in this file rather than promised for the next one.** Every
command this office ran during it.16 is marked below with its verbatim output.

```
[RUN] bash scripts/iteration_timer.sh check
   ->  iteration 2: 2m5s elapsed, 17m55s left of 20m        (exit 0, iteration opened)
[RUN] git status --porcelain
   ->  4 tracked modifications, 38 untracked V20_R15_* reports (full text in §6)
[RUN] git rev-parse HEAD
   ->  207e7b92c5611effbcac6877757dc1d0bb572e42
[RUN] grep -n "Q2/W1" V20_R15_THEORY_TABLE.md
   ->  :118 `| **Q2 OUTSIDE** | **F4** (re-graded it.12) | **F1 + const** |`
[RUN] sed -n '285,292p' V20_R15_THEORY_TABLE.md
   ->  "The gate does not take F4. This table has three F4 cells: Q2/W1, Q6/W1, Q6/W3."
[RUN] sed -n '140,146p' CEQ_V20_R15_CONTRACT.md
   ->  it.35 grades every F1/F2/F3 failure LEAPABLE or TERMINAL; F4 unmentioned
[RUN] grep -n "L-GRADE" CEQ_V20_R15_CONTRACT.md
   ->  :58 only, and it is a parenthetical inside a list of law names --
       "L-COST, L-PURGE, L-FIND, L-GRADE (F0-F4 + HOW-BAD gap), L-LEAP"
       CONFIRMS SATURN: the scale the twelve cells are graded on has no text.
[RUN] cat scripts/iteration_timer.sh
   ->  read in full; the heartbeat analysis in §3 is against this source
```

**The historical count, and this office declines to claim exoneration from it.**

```
[RUN] scripts/saturn_run_census.py:27-28   the NODE regex SATURN scored against
   ->  NODE = re.compile(r"(?:tests|scripts|ceq|scale|lean)/[\w./-]+\.(?:py|sh|lean)"
                         r"(?:::[\w:\[\]-]+)?")
       A path prefix is REQUIRED. A bare `::test_foo` scores zero. Confirms 2/29.
[RUN] grep -c "\[RUN\]" V20_R15_IT*INSPECTOR.md
   ->  IT12:18  IT11:8  IT3:7  IT89:5  IT1:4  IT567:4  IT13:3  IT2:1  IT4:0
       64 markers across all ten reports today. The 29 was the it.12-era denominator,
       cited by this office at V20_R15_IT12_INSPECTOR.md:487 - "Across my seven
       reports: 29 markers." Both numbers are right for their date.
[RUN] search for `[RUN]` markers with no command on the line or the next
   ->  no bare markers found
```

**The search for the three empty-marker reports came back empty, and this office is
NOT recording that as a clearance.** The search tested one shape - a marker with
nothing after it. SATURN's charge is the broader one: a marker followed by *prose*
rather than by an executable command and its output. `IT4:0` markers with a report
still filed, and `IT2:1` marker across a whole audit, are the shape that charge
describes, and neither is refuted by finding no *bare* markers.

**Standing: the charge is three iterations old, it has now been deferred a third
time, and this office will not litigate its own score down.** What is conceded
without qualification: **seventeen commands were run and none was marked.** The
remedy applied here is the nine marked commands above and the verbatim outputs in
§§1-4, and the correct measure is whether it.17's Inspector filing can be re-scored
by SATURN's own census script without this office arguing about the regex.
## 6. TREE STATEMENT

**No git writes were made by this office. Nothing touched Kaggle.**
`HEAD = 207e7b92c5611effbcac6877757dc1d0bb572e42`, branch `v17k-gate0`, unmoved.

```
[RUN] git status --porcelain | wc -l   ->  101  (audit open)
[RUN] git status --porcelain | wc -l   ->  102  (audit close; the new entry is
                                              V20_R15_IT17_JUPITER.md, see 4.1)
```

**4 modified tracked files, 98 untracked at close.** The four modified are
`MISTAKES.md`, `house-events.jsonl`, `pytest.ini`, `scale/ledger.py` — **all four
were already modified before this audit opened** and none was touched by this office.
`house-events.jsonl` is append-only and is written by `iteration_timer.sh` itself,
including by this office's own `check` calls; that is the instrument logging, not an
edit.

**Concurrent it.17 paths, reported honestly rather than filtered out.** MERCURY and
JUPITER were both working inside this audit's window, and the following appeared or
moved while it ran:

- `V20_R15_IT17_MERCURY.md` (25 lines — a filing in progress at read time)
- `tests/mercury/test_v20_r15_it15_arena_rig.py`
- `tests/mercury/test_v20_r15_it13_phase_c_price.py`, `tests/mercury/arena_price.py`,
  `tests/mercury/phase_c_price.py`

**This office cannot certify the tree was quiet, and does not claim it.** Any suite
count in this report is a reading taken against a tree two other offices were
writing. Where a count mattered it was taken twice; where it could not be, it is
named as a single reading below.

**Mutation discipline:** one mutation was authorised this iteration (the clause-(b)
`kind` exchange in `results/v17k_r4_retake.jsonl`), applied by a nurse under a
sha256 before/after with a scratchpad backup. Its revert is asserted in §1 by digest
equality, not by assertion.
## 7. FITNESS RULING

**Drafted from SATURN's own LIMITS at it.14 and recorded BEFORE any nurse landed**,
so the ruling is not back-fitted to whatever the census returned. The census then
returned `0 bad`, and the ruling below is unchanged - which is the point of having
written it first.

**Three facts, all of them from SATURN's own filing, not from this office:**

1. `THEORY-SHA256` **covers the twelve §1 cells and nothing they cite.** His words:
   *a digest over a list is a guarantee about the list.* Editing `ceq/arm_pl.py`,
   a `results/*.jsonl` record, a ledger row or the table's §0 prose moves no digest.
   The freeze makes the table **locatable**, which is a different property from
   **true**.
2. **Three of the twelve cells are grades the gate cannot consume — and the table
   says so itself.** SATURN flagged `Q2/W1` alone, because it is the one with no
   ledger row (§C.2). The larger fact is on the table's own face:
   `V20_R15_THEORY_TABLE.md:118,123` grade `Q2/W1`, `Q6/W1` and `Q6/W3` as `F4`, and
   `:285-287` states *"The gate does not take F4. This table has three F4 cells."*
   The it.35 clause confirms it — `CEQ_V20_R15_CONTRACT.md:140-146` grades
   **`F1/F2/F3`** failures LEAPABLE or TERMINAL and never mentions `F4`.
   **25% of the leap's only input is invisible to the gate that consumes it.**

   **Credit where it is due: JUPITER got there first.** `RULING J-14` at `:289` —
   *"An `F4` cell is NOT leap material as an `F4`"* — is the table declaring its own
   blind spot, and this office found nothing JUPITER had not already written down.
   That changes the severity and not the fact. **A declared defect is still a
   defect.** A reader who reaches `:285` is warned; an it.35 gate that filters on
   `F1/F2/F3` drops three cells and emits no warning at all, because nothing in the
   contract clause tells it to look.
3. **The grades are read against a scale that has no text.** `L-GRADE` is BOUND and
   unrepaired; SATURN states plainly that *"F4 is not in the gate's input set"* is
   read off a contract clause, **not off a rubric.** Twelve cells carry grades on an
   unwritten scale.

**And the fourth fact, which the census supplied:** 123 citations **resolve**; only
**14** were checked against the standard that matters - *does the cited line carry
what the table claims*. `14/123` is 11%. **The other 109 are certified by exactly the
instrument PRIORITY 1 struck this iteration** - a resolution check, which is what
clause (b) had while pointing at the wrong record.

---

### THE RULING, IN ONE LINE

**NO. The theory table is not fit to be the leap's only input.** It is fit to be the
leap's **primary** input, read by a person, alongside the ledger and the contract.

**Three reasons, none of which a digest repairs:**

1. **Three of twelve cells are `F4`, and the it.35 gate takes `F1/F2/F3`.** 25% of the
   input is invisible to the consumer. The table declares this at `:285-287` and
   JUPITER ruled on it at `:289` - **the defect is disclosed, and disclosure is not
   repair.** A human reader who reaches line 285 is warned. A gate that filters on
   `F1/F2/F3` drops three cells and emits nothing, because
   `CEQ_V20_R15_CONTRACT.md:140-146` never tells it to look.
2. **The grades are read against a scale with no text.** `L-GRADE` appears once in the
   contract, at `:58`, as a parenthetical in a list of law names -
   *"L-GRADE (F0-F4 + HOW-BAD gap)"*. There is no rubric. **Twelve cells carry grades
   on an unwritten scale**, and SATURN says so plainly: *"F4 is not in the gate's input
   set" is read off the contract's clause, not off a rubric.*
3. **The freeze guarantees the list, not the citations.** `THEORY-SHA256` covers the
   twelve cell texts. Editing `ceq/arm_pl.py`, a `results/*.jsonl` record, a ledger row,
   or the table's own §0 prose moves **no digest**. The census that was supposed to
   cover the gap covers 11% of it by the standard asked for.

**What IS fit, and it is more than the round had a week ago.** The table is **frozen
and locatable** for the first time: twelve keyed cell digests, a planted negative
applied to the real file on disk rather than simulated, and a `V-26` self-audit of the
digest shipped in the same iteration the rule was filed. Cell-level location is a
genuine gain and this office does not discount it.

**The distinction the round should carry forward:** *frozen*, *locatable*, and *true*
are three properties. it.14 delivered the first two. **Only the third makes an input
safe to be alone**, and the instrument that was supposed to establish it - a
resolution check - is the instrument this iteration struck.

**The smallest change that would flip this ruling to YES**, in priority order:
**(a)** give `L-GRADE` a rubric with text, so the twelve grades mean something
checkable; **(b)** state in the contract what it.35 does with an `F4` cell, so the
three are dropped **loudly** rather than silently; **(c)** finish the content census
on the remaining 109. **(a) is the expensive one and it is the one that has been
BOUND and unrepaired since it.12.**

---

## 8. WHAT THIS AUDIT DID NOT REACH

Named, per the standing rule, rather than left as silence.

- **No it.15 or it.16 office filing exists on disk.** `ls V20_R15_IT15*` and
  `V20_R15_IT16*` return only this file. The it.14-it.16 window was audited from the
  it.14 filings plus the journal corrections; **there is no it.15 report to audit**,
  and this office does not know whether that is a gap or a numbering convention. It
  is flagged rather than assumed.
- **MERCURY it.17 was not audited.** `V20_R15_IT17_MERCURY.md` was 25 lines and being
  written during this audit. `V20_R15_IT17_JUPITER.md` appeared mid-audit and was not
  read at all until the final minutes - it DID carry the content census, and §4.1 was
  written from it. Its chunk 4 (C94-C123) had not returned when this audit closed.
- **The `V-26` sweep was not extended tree-wide.** SATURN calls the census the cheap
  follow-on; this office agrees, and did not run it either. Three fresh instances were
  found by hand this iteration (`L-GRADE`, the timer, the census headline), which
  suggests the class is nowhere near exhausted.
- **The 109 content-unchecked citations were not checked here.** That is JUPITER's to
  finish, and it is the one item that would move the fitness ruling.
- **The `:453` half of Correction 26 was checked only for adopters** - none outside
  C-22 itself - and not for its "five each" count.
- **The heartbeat repair was specified, not applied.** No code was written by this
  office; `scripts/iteration_timer.sh` is unmodified.
