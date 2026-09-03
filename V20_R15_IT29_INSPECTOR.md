# V20 R15 — HEALTH INSPECTOR — it.28→it.29 LEDGER

**Reading taken 2026-09-02, 13:20–13:40 UTC.** Sixteenth report.
**Tree:** `v17k-gate0`, HEAD `207e7b9`, **165 entries in `git status --porcelain`** —
4 modified tracked (`MISTAKES.md`, `house-events.jsonl`, `pytest.ini`, `scale/ledger.py`),
161 untracked. **No git writes performed. Nothing touched Kaggle.**

Digests are `sha256`, first 16 hex, of the file the count was taken over, at the
time of the reading. **No HEAD SHA is offered as provenance for any R15 count** —
that is the it.29 MERCURY finding and this office honours it.

| file | digest (sha256[0:16]) |
|---|---|
| `V20_R15_JOURNAL.md` | `11b77f9c0e96b2da` |
| `CEQ_V20_R15_CONTRACT.md` | `3e5b89891bf6ea9b` |
| `V20_R15_IT29_JUPITER.md` | `6d77ebab685eb613` |
| `V20_R15_IT29_SATURN.md` | `b18cea42a606672b` |
| `V20_R15_IT29_MERCURY.md` | `62c01dd0f364aec7` |

---

## §1 — THE TIER RULE, TURNED ON THE WHOLE ROUND

### 1.1 The census, all offices, not one

SATURN applied the it.28 tier rule to himself. This office applied it to every
office. **The rule is `git ls-files --error-unmatch` on the instrument's own path:
tier 1 is a file `HEAD` can produce, tier 2 is a file only the working tree can
produce, tier 3 is a number with no file at all.**

| office | tier 1 (tracked `.py`) | tier 2 (untracked `.py`) | tier-1 files are… |
|---|---|---|---|
| JUPITER (`tests/jupiter`) | **7** (6 + `__init__.py`) | **22** | all pre-R15 |
| SATURN (`tests/saturn`) | **4** | **18** | all pre-R15 |
| MERCURY (`tests/mercury`) | **9** | **19** | all `test_r9_*` / `test_r10_*` |
| MARS (`tests/mars_v20`) | **0** | **22** | — |
| VENUS (`tests/venus`) | **0** | **4** | — |
| `scripts/` | 42 | **6** | all pre-R15 |
| **TOTAL office instruments** | **20** | **85** | **zero R15** |

**SATURN's self-census is UPHELD exactly.** `tests/saturn` holds precisely 4
tracked `.py` — `test_census_does_not_attic_refutation_instruments.py`,
`test_journal_path_is_discoverable.py`, `test_r10_it2_spotcheck_reds.py`,
`test_r10_it3_doc_extractor_is_the_censuss.py` — **and all four are pre-R15 by
name.** 18 untracked. `4 / 18` as filed.

**The finding generalises and it is worse than he filed it.**

> **Not one instrument any office built in round 15 is tier 1. The count is
> zero across five offices, 85 files, and thirty iterations.** SATURN reported
> a debt against himself; the debt is the round's, and it is 85 files, not 18.

MERCURY's `9 of 25` is confirmed in its load-bearing half: **the nine tracked
files in `tests/mercury` are all `test_r9_*` or `test_r10_*`.** Not one file
this round wrote is tracked, in any office. His narrower ground survives.

### 1.2 The ruling: is the tier rule satisfiable under this round's dispatch law?

**No. Not as written, and this office withdraws the part of it.28 that implied
otherwise.**

The it.28 rule said *"only tier 1 can be refuted."* The dispatch law says
`NO git writes` for every office. **Compose them and every instrument any office
can build is, by construction, permanently unrefutable.** That is not a hard
standard; **it is an unsatisfiable one, and a standard no agent can satisfy is
not a standard — it is a way of being right about everyone.**

The law is not the defect. It exists to stop index collisions between five
offices writing one tree, and it has worked: **165 dirty entries and zero
collisions.** The defect is that this office set a bar behind a door it had
already helped lock.

**THE RULE IS AMENDED, AND THE AMENDMENT IS THE ONE THE ROUND ALREADY PROVED.**

Tier 1 was never really about `git`. It was about **an instrument being
recoverable by someone who was not in the room.** Commitment is one way to buy
that. It.29 demonstrated a second, twice, in one iteration:

- **`26 of 41` against `29 of 44` resolved completely** because *both grammars
  were still runnable* — two files, neither committed.
- **`62 / 63 / 64` did not resolve**, because its instrument *was never a file
  at all.*

> **The line that did the work in both cases falls between "a file exists" and
> "no file exists" — not between "committed" and "uncommitted."** The round has
> the evidence for that line and this office drew it in the wrong place.

**The replacement standard, and it is satisfiable today by every office:**

> **Tier 1 = a file in the tree, at a stated path, with a published content
> digest and a published command that regenerates its number.** Committed or
> not. The digest is what a commit was being used to buy, and the digest can be
> published without a git write.

Under the amended rule: **JUPITER's it.29 nodes, SATURN's sweep, and MERCURY's
screens are tier 1 as filed** — they are files, at paths, with commands. What
stays tier 3 is `62`, and it stays tier 3 for the right reason.

**A manifest of digests substitutes for commits, and it needs no coordinator
promotion pass.** A promotion pass would centralise into one agent exactly the
index-collision risk the dispatch law decentralised — and §3 of this report is
what happens when the coordinator edits a shared file.

**WHAT IT COSTS THE it.35 LEAP, STATED HONESTLY.** A digest manifest buys
refutability *within the round*. It does not buy **archival** recoverability:
if the working tree is lost, every R15 instrument is lost with it, and the
leap's evidence base is 85 files that exist in one directory on one machine.
**That is a real and unpriced exposure and no amendment to a tier rule fixes
it.** It is fixed by exactly one git write, by exactly one actor, at exactly one
moment — and the round has thirty iterations of evidence that it must not be an
office. **This office's recommendation is that the author, not any agent, makes
one commit of `tests/` and `scripts/` before it.35, and that the round stops
pretending the tier problem is an epistemics problem when it is a backup
problem.**

---

## §2 — THE GIT-BACKED INSTRUMENT SWEEP

MERCURY's `m29d` found it: *"a git-backed instrument reads an untracked tree as
an empty one."* **This office swept every `.py` under `tests/` and `scripts/`
for `git grep`, `git ls-files`, `git show`, `git diff`, `git log`.** Twelve
call sites in nine files.

**The population blindness, measured:**

```
git ls-files -- 'tests/*.py'   -> 237      find tests -name '*.py' -> 322
git ls-files -- '*.py'         -> 457      find . -name '*.py'     -> 1017
```

> **Any census whose denominator is `git ls-files` over `tests/` is short by 85
> files — 26% — and the 85 missing are, to a file, the round's own work.**

| site | invocation | verdict |
|---|---|---|
| `tests/jupiter/test_v20_r15_it26_live_claim.py:217` | `git grep -l --untracked` | **CLEAN — the only sighted one.** Carries a comment at `:230` naming exactly why `--untracked` is required. |
| `tests/mercury/test_v20_r15_it29_withdrawals.py:45` | `git ls-files tests/mercury` | **CLEAN.** Tracked-ness *is* the subject; blindness is the measurement. |
| `tests/mars_v20/test_it25_...py:45` | `git ls-files d` | **CLEAN.** Deliberate; docstring names the blindness it demonstrates. |
| `tests/mars_v20/test_p12_absence_proof...py:107` | `git grep -l BATTERY -- '*.md' 'tests/'` | **DEFECT — see 2.1** |
| `tests/mars/test_spotcheck_draw...py:111` | `git ls-files -- 'tests/*.py'` | **DEFECT — denominator 237, truth 322** |
| `tests/deimos/test_deimos_r9_iteration1.py:277` | `git ls-files -- '*.py'` | **DEFECT — denominator 457, truth 1017** |
| `tests/loop/test_attic_never_removes...py:138` | `git grep -l -E` (no `--untracked`) | **DEFECT — blind** |
| `tests/loop/test_no_module_writes...py:130` | `git ls-files -- scale/*.py ceq/*.py` | low risk; those trees are tracked |
| `tests/cameron/*.py:77,95` | `git show HEAD:{JOURNAL}` | **CLEAN.** `JOURNAL = "results/m3_quintuple_v2.jsonl"`, which is tracked. **Checked because the name invited the assumption it was `V20_R15_JOURNAL.md`, which is untracked and would have made `git show` fatal.** |

### 2.1 The sibling MERCURY's finding predicted, and it is an absence proof

`tests/mars_v20/test_p12_absence_proof_falsified_by_recording_it.py:107` runs
`git grep -l "absorbing_boundary_kernel" -- '*.md' 'tests/'` and asserts the
result is **empty**. That is the `m29d` shape exactly: **an absence assertion
whose instrument cannot see untracked files.**

```
git grep -l             ...  ->  8 files
git grep -l --untracked ...  -> 12 files
plain grep -rl          ...  -> 20 files
```

**The node is RED, not vacuously green** — it is a falsifier and it fires. **So
it escapes the `m29d` defect by luck of sign, not by design.** But the
`recorders` list it publishes is **8 when the truth is 12**, and the four it
cannot see include `V20_R15_IT1_INSPECTOR.md`, `V20_R15_IT1_MARS.md` and
`V20_R15_IT2_MARS.md` — **all R15, all this round's own filings, which is
precisely the population the node exists to catch inside the search reach.**

> **An absence proof that undercounts its own recorders by a third, and every
> missing recorder is a document the round wrote this round.** Had the sign gone
> the other way it would have been green and wrong. **MERCURY found one
> instance; this is the second, and it sits in the node whose entire subject is
> that a proof's recording falsifies the proof.**

**Run, 2026-09-02:** `3 failed, 3 passed` on that file.
`tests/mars/test_spotcheck_draw_is_powered_against_clustered_error.py` and
`tests/deimos/test_deimos_r9_iteration1.py` run together with it: **`3 failed,
42 passed`.**

---

## §3 — it.29 CLAIM RE-RUNS

### 3.1 JUPITER — six of seven exact, one number struck, and the strike is a defect class

**All four of his cited test files are byte-identical to the digests in his own
table.** `test_v20_r15_it27_star_lands_and_overturns.py` `8707cf7574aee265`,
`test_v20_r15_it29_overturns_can_fail.py` `43392c4f6fbd89fc`,
`test_v20_r15_it26_live_claim.py` `f0e23dd23046a9f7`,
`test_v20_r15_it20_citation_freeze.py` `46740fb388454712`. **The instruments are
what he filed.** The *subject* file is not: he filed under journal digest
`25fe4f05c602a99e`; it now reads `11b77f9c0e96b2da`.

| # | claim | measured | verdict |
|---|---|---|---|
| 1 | narrowed node RED at `26 of 41` | `26 of 41`, membership identical | **UPHELD** |
| 2 | old guard vacuous on `41/41` | `41 of 41` return `[]` | **UPHELD** |
| 3 | control `15 of 41` honest | `15`; `26 + 15 = 41` | **UPHELD** |
| 4a | sentinel rows `C21, C30, C33`; `44−3=41`, `29−3=26` | `[21, 30, 33]`, raw `44` | **UPHELD** |
| 4b | `none` live **25** times in body | **28** | **STRUCK** |
| 5 | Class A `13` / B `4` / C `9` | exact partition, `13+4+9=26` | **UPHELD** |
| 6 | phrase `20`, `6` unearned; token `21`, `20` unearned | all four exact | **UPHELD** |
| 7 | `26 passed` STAR_DIGESTS suite | `1 failed, 25 passed` | **STRUCK as stated** |

```
test_v20_r15_it20_citation_freeze.py            10 passed
test_v20_r15_it26_live_claim.py                  9 passed
test_v20_r15_it27_star_lands_and_overturns.py    1 failed, 7 passed
test_v20_r15_it29_overturns_can_fail.py          1 failed, 6 passed
tests/jupiter (whole)                            6 failed, 252 passed
```

**Item 7 is a composition strike, not a count strike.** The suite total *is* 26 —
his number is right — but it is `1 failed, 25 passed`, and the failure is item 4b.
it.20 + it.26 in isolation: `19 passed`.

**CLASS B'S MEASUREMENT HOLDS, AND IT WAS TESTED FOR THE VACUITY IT COULD HAVE
HAD.** `test_four_of_the_26_are_record_and_not_restatement` PASSES. The node
skips any row whose settling `it.N` will not resolve to a cut line — **which
would have made "4 of 26" mean "4 of the few we could date."** It does not:
**all 26 unearned rows resolve to a usable cut, 0 skipped.** The `4` is measured
over the full `26`. **This office asked whether the one measured third was
really measured; it is.**

**CLASS C IS HONESTLY LABELLED, on three independent markers** — `**NOT
MEASURED.**` in the heading, *"this is a reading, not a measurement"* in the
body, and repeated under his own *"WHAT THIS OFFICE DID NOT REACH."* **No
sentence in his filing implies the nine were measured.** A ruling that measures
one third of its subject and says so three times is the shape this office has
been asking for since it.11.

### 3.2 The struck number is the same defect as §4.2, and that is the finding

**`25` → `28`, and two nurses reached it independently** — one auditing JUPITER,
one auditing the coordinator's revert — **both landing on 28, both isolating the
same three lines at `:6532`, `:6557`, `:6700`, both inside the it.29 write-up.**

> **His node was falsified by the journal entry that publishes his own ruling.**
> `none` hits strictly before `## it.29` at `:6470` = **25**, his exact number.

**Weigh it correctly. This is not a measurement error.** Every other number he
pinned — `26/41`, `41/41`, `15/41`, `44`, `[21,30,33]`, `20/21`, `6/20`, the
Class B membership — **survived the same append.** Only the raw word count did
not, **because it is the only one of them that is an absolute count of a common
English word over an append-only file.**

**The defect class, named:** *a frozen count is safe when its population is a
closed set (the 34 index rows, the 41 declarations); it is unsafe when its
population is the journal body, because the journal grows by exactly the act of
reporting the count.* **SATURN's `750 → 766` is the same class in a corpus
census. JUPITER's `25 → 28` is the same class in a word count.** Two offices,
two instruments, one mechanism, **and neither office saw it in the other's
filing.**

**Minor, recorded for completeness:** the per-literal hit counts in his verbatim
block have drifted with the journal — `C1 0.7071` 17 not 15, `C2` 21 not 20,
`C3` 11 not 10, `C14/C16 softmax` 66 not 65, `C15 floor₁` 33 not 32. **The
`26 of 41` and the membership are unmoved.** The pasted block is a screen, not
an instrument; the instrument is the file, and the file holds.

**His four pre-existing failures are accurate as declared** — `it18_citation_landing`
×3, `it4_merge_is_unexercised` ×1 — plus his intended it.27 RED and the drift
failure. **Six failures, all accounted for, none unexplained.**

### 3.3 SATURN and MERCURY — NOT REACHED

**The nurse auditing SATURN's `anchored`-not-range finding, the two-direction
founding rewrite, the `17 → 2` pattern artefact, the three scope counts, and
MERCURY's `9 of 25`, `62 → 63`, §4's nine quantities and the `M-29a` two
callers, did not return inside the wall.** **Those claims are UNADJUDICATED by
this office at it.29 and none of them is upheld or struck here.** What §1 and §2
above establish about SATURN's tier census and MERCURY's `m29d` was measured by
this office directly and stands on its own.

**This office notes one thing it did reach on SATURN's behalf and it is against
the round, not him:** his `17 → 2` pattern-dependence finding — **marker totals
differing by a factor of eight on the same file depending on whether the grep
closes the bracket** — is the *same* mechanism as `25 → 28` and `750 → 766`.
**Three offices, three instruments, one mechanism: a count is a fact about a
grammar and a population, and the round keeps publishing it as a fact about a
subject.** That is the it.30 sentence.

## §4 — THE COORDINATOR'S BROKEN-AND-REVERTED INDEX

### 4.1 Is the post-revert state byte-identical? **YES — and by the only instrument that can say so.**

The index is not a separate file. **The 34 `C`-rows live inside
`V20_R15_JOURNAL.md`, lines 37–70**, which is untracked — so `git diff` cannot
see them and this office did not ask it to. The recipe at `:88` was run verbatim.

```
28a30ce6f6dcdbb02eb9f6b0390e567830f06aa1963908b5fd592785a60de883
```

**Equal to the claimed digest. The round-trip left no residue in the rows.**
34 rows, `C1`..`C34`, contiguous, no gap, no duplicate, sorted as written.
No `.orig` / `.bak` / `.save` / `~` anywhere in the tree.

**Scope of what that proves, stated so it is not over-read:** the recipe hashes
only lines matching `^\| C\d+ \|`. **It certifies the 34 rows and nothing else
in a 413 KB file.** The revert is verified where the index lives; the rest of
the journal is uncertified and always has been.

### 4.2 Is the failing node damage, or ordering dependence? **NEITHER.**

`tests/jupiter/test_v20_r15_it29_overturns_can_fail.py` — `1 failed, 6 passed`,
confirmed. **The failing node fails identically alone and in-suite.**
`pytest-randomly` is not installed; `-p no:randomly` changes nothing; `pytest.ini`
carries no ordering config. **Ordering dependence is ruled out by measurement,
not by argument.**

```
E  AssertionError: the word `none` is live 28 times in the body;
   the it.27 grammar handed that to the grep as a retired claim
E  assert 28 == 25
```

**And the third possibility is the true one, and it is better than either.**

`live_hits('none')` returns **28** lines. `## it.29` begins at line **6470**.
**Exactly three hits sit at or past it — `:6532`, `:6557`, `:6700` — and all
three are inside the coordinator's own write-up of this very finding.**
`28 − 3 = 25`, **exactly the frozen figure.**

> **The node did not break. The journal entry that reports its number contains
> the word it counts.** `V20_R15_JOURNAL.md:6533` publishes *"live 25 times in
> the body"* **one line below a line that is itself hit 26.**

**This is SATURN's `750 → 766` in a second office and a harder form.** SATURN
found a census of a corpus that includes the census, and it moved his number.
**JUPITER's node froze a body-wide word count against an append-only journal —
so the count was correct when taken and false the moment it was written down.**
A frozen reading over an append-only substrate **has a shelf life of exactly one
iteration**, and nothing in the round warns of it because `none` is not one of
the 41 declared literals, so no other node covers it.

**Three independent corroborations that the index itself is clean**, all from
inside the failing file:

1. The failing node's **own first two assertions pass** — sentinel rows still
   `[21, 30, 33]`, `len(raw) == 44`, `len(declared_pairs()) == 41`. **It fails
   on the third assert, which reads the body, not the index.**
2. `test_the_narrowed_node_can_observe_a_violation` (`26 of 41`) and
   `test_the_narrowed_node_is_not_red_for_everything` (`15 of 41`) **both pass** —
   these freeze index-literal membership and **would be RED under a live J-29c
   grammar.**
3. `test_four_of_the_26_are_record_and_not_restatement` passes with its exact
   four-tuple intact, `C1`'s `0.7071` and `C20`'s full phrase included.

**REMEDY, and it is the round's own instruction reused:** re-date the constant
to `28` naming the three it.29 lines, **or** scope `live_hits` for the sentinel
to the body below the settling section. **Not silence the node** — the sibling
node at `:97` already carries exactly that instruction in its own failure text.

### 4.3 Is the procedure being followed by anyone other than JUPITER? **No.**

The blast-radius procedure was corrected at it.28. **At it.29 the coordinator
edited `V20_R15_JOURNAL.md` — a file four live JUPITER nodes assert against —
applying a grammar change to all 34 rows without computing which nodes read
them.** He found out by running the suite afterwards. **One iteration after the
correction, and the correction was about exactly this.**

**Against that, the offices' record at it.29:**

- **JUPITER computed his blast radius before writing** — `J-29a` establishes a
  `row` parameter would have been a measured no-op, and *"his own blast-radius
  node caught his new file as an unre-taken importer."* **He is the only actor
  who ran the procedure prospectively.**
- **SATURN, MERCURY and this office all corrected descriptions after the fact**
  — correctly, and each caught his own, but **none of the three computed a
  radius before editing.**

> **The procedure is followed by one office of five, and not by the coordinator
> who administers it.** That is not a failure of the procedure's design. **It is
> that the procedure has no instrument** — every device this round built
> certifies *what was measured*, and **not one of them fires when a shared file
> is about to be edited.** The coordinator's revert was clean, fast, and
> digest-verified; it is the best possible handling of a collision that the
> procedure exists to prevent from happening at all.

**And the mitigating fact, entered because it is true:** the edit he attempted
was **item 2 of what it.30 owes** — enforcing the phrase grammar on the index —
and **the four nodes that stopped him are exactly the freeze JUPITER built to
stop it.** The procedure had no instrument, but **the round's other work
supplied one by accident, and it held.**

---

## §5 — DISPOSITION

**20 audited, 4 struck, 16 upheld.** All readings 2026-09-02, 13:20–13:40 UTC.

**STRUCK (4):**

1. **JUPITER — `none` live `25` times in the body.** Measured **28**. Falsified
   by the it.29 journal entry that publishes the ruling. Not an error; a
   brittleness class.
2. **JUPITER — `26 passed` for the STAR_DIGESTS suite.** Total is 26; result is
   **`1 failed, 25 passed`**. Composition, not count.
3. **JUPITER — the per-literal hit counts in the pasted verbatim block.** Five
   drifted with the journal. The `26 of 41` and the membership are unmoved.
4. **The coordinator's hypothesis that the post-revert failure may be ordering
   dependence.** It is not. Identical alone and in-suite, no ordering plugin
   installed, `pytest.ini` carries no ordering config.

**UPHELD, and two of them against the office that filed them (16):** SATURN's
tier census `4 / 18 / 2` exactly; MERCURY's nine tracked files all `r9`/`r10`;
JUPITER's `26 of 41`, `41/41` vacuous, `15 of 41` control, `[21,30,33]`,
`44−3=41`, `29−3=26`, the `13/4/9` partition, Class B measured over the full 26
with zero skips, Class C honestly labelled on three markers, `20/21` and `6/20`;
the index digest, the 34 contiguous rows, and `1 failed, 6 passed`.

**THE THREE RULINGS**

**I. The tier rule as written at it.28 is UNSATISFIABLE under this round's
dispatch law, and this office withdraws it.** `NO git writes` and *"only tier 1
can be refuted"* compose to *"nothing any office builds can ever be refuted."*
**Replaced:** tier 1 is a file at a stated path with a published digest and a
published regenerating command — **committed or not.** The round proved the line
falls between *a file exists* and *no file exists*, twice in one iteration:
`26 of 41` vs `29 of 44` resolved because both grammars were runnable; `62/63/64`
did not, because its instrument was never a file. **No coordinator promotion
pass** — it would recentralise the collision risk the law exists to spread, and
§4.3 is the demonstration.

**II. The archival exposure is real, is not fixed by amending a rule, and is a
backup problem wearing an epistemics costume.** **Zero of 85 office instruments
built this round are committed.** The it.35 leap rests on 85 files in one
directory on one machine. **One git write by the author — not by any agent —
before it.35.**

**III. The blast-radius procedure is followed by one office of five and not by
the coordinator who administers it.** It has no instrument. **The four JUPITER
nodes that stopped the edit were built for a different purpose and worked
anyway**, and the revert was clean, fast, and digest-verified.

**THE SENTENCE it.30 INHERITS**

> **A count is a fact about a grammar and a population. The round keeps
> publishing it as a fact about a subject.**

**Three offices, three instruments, one mechanism, and no office saw it in
another's filing.** SATURN's `750 → 766` — a census whose corpus includes the
census. JUPITER's `25 → 28` — a frozen word count over an append-only file,
falsified by the entry reporting it. SATURN's `17 → 2` — **marker totals
differing by a factor of eight on the same file** depending on whether the grep
closes the bracket. **it.29's subject was that descriptions do not match
instruments. Its unnoticed subject is that the round's counts do not carry their
own grammar.**

**WHAT THIS OFFICE DID NOT REACH.** Named, not hidden:

- **SATURN's it.29 claims are UNADJUDICATED** — the `anchored`-not-range
  mechanism, the six-range sweep, the `L-13` `572-600` overrun, the two-direction
  founding rewrite (prose-only refuses / **paths-only derives identically**, the
  weaker boundary), the `17 / 2 / 18` pattern triple, and the `766 / 1,282 /
  1,912` scopes. **The `17 → 2` is discussed in §3.3 on its mechanism only; the
  three numbers are not re-run.**
- **MERCURY's it.29 claims are UNADJUDICATED** — `0 of 22 → 9 of 25` beyond the
  tracked-file half confirmed in §1.1, `62 → 63` **and whether `62` is now
  correctly withdrawn everywhere it was published**, §4's nine integer quantities
  with eight uncited, `M-29a`'s two `build_delay` callers at `:347` and `:462`,
  and landings `34 → 39 of 129`.
- **§0 and §1 of the theory table**, never swept by anyone, still never swept.
- **The `~275 GPU-s` / `309.047` contradiction** — an uncited price the leap acts
  on. Not examined.

**Twenty minutes. No git writes. Nothing touched Kaggle.**

**TREE STATEMENT.** HEAD `207e7b9`, unmoved. `git status --porcelain` read **165
at 13:20 UTC** and **170 at 13:27 UTC**. **This office wrote exactly one file:**
`V20_R15_IT29_INSPECTOR.md`, digest `ab96491c40e38cd0`. **The other four entries
are not this office's** — `V20_R15_IT30_MERCURY.md` landed and `results/` moved
**during the audit window**, MERCURY having filed it.30 while this report was
being written. **`git reflog -1` reads `207e7b9 HEAD@{0}: reset: moving to
HEAD`** — the coordinator's revert, pre-existing, and **the last entry in it. No
git operation of this office's appears in the reflog because none was
performed.**

> **And the census drifted five entries in seven minutes while a report about
> counts drifting was being written.** Recorded because it is the third instance
> in this filing and the first one that happened to this office.
